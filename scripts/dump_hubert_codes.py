import torch
import torch.nn.functional as F
import torchaudio
import os
import argparse
import tqdm
import pickle
import numpy as np
import fairseq
import joblib
import csv

MIN_WAV_LEN = 720

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOWNSAMPLING_FACTOR = 640  # maybe we'll want to un-hardcode it later
MODEL_SR = 16000


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "data", help="Path with .csv files pointing to the audio data"
    )
    parser.add_argument("--split", help="Which split", required=True)
    parser.add_argument(
        "--audio-column-idx",
        type=int,
        default=2,
        help="Column in the .csv files containing the audio files",
    )
    parser.add_argument(
        "--audio-start-column-idx",
        type=int,
        default=None,
        help="Column in the .csv files containing the audio offset (if any)",
    )
    parser.add_argument(
        "--audio-duration-column-idx",
        type=int,
        default=None,
        help="Column in the .csv files containing the audio duration (if any)",
    )

    parser.add_argument(
        "--audio-prefix",
        type=str,
        default="",
        help="Prefix for the audio file e.g. '${LIBRILIGHT_PATH}/'",
    )
    parser.add_argument(
        "--audio-suffix",
        type=str,
        default="",
        help="Prefix for the audio file e.g. '.flac'",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to the directory containing the HuBERT checkpoint",
        required=True,
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument(
        "--km_path",
        help="Path to the directory containing the kmeans checkpoint",
        required=True,
    )
    parser.add_argument(
        "--save-dir", help="Output path to store the features", required=True
    )
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max_chunk", type=int, default=3200000)
    parser.add_argument("--dtype", type=str, default="float32")
    return parser


class PseudoTextTokenizer(torch.nn.Module):
    def __init__(self, feat_extractor_path, layer, km_path, max_chunk=1600000):
        super().__init__()
        # Feature extractor
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [feat_extractor_path]
        )
        self.model = model[0]
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        # Quantizer
        km_model = joblib.load(km_path)
        self.C_np = km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.register_buffer("C", torch.from_numpy(self.C_np))
        self.register_buffer("Cnorm", torch.from_numpy(self.Cnorm_np))
        self.sample_rate = MODEL_SR

    def wav2code(self, x):
        with torch.no_grad():
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                if x_chunk.size(1) < MIN_WAV_LEN:
                    continue
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
            feat = torch.cat(feat, 1).squeeze(0)
            dist = (
                feat.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(feat, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).unsqueeze(0)


def print_args(args):
    print("*" * 40)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    print(f"\tdevice: {device}")
    print("*" * 40)


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invalid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    if start >= end:
        start = end - 1
    print(
        f"rank {rank + 1} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_iterator(args, mdl, device, ptdtype, n_processed=0):
    with open(os.path.join(args.data, args.split) + ".csv", newline="") as fp:
        reader = csv.reader(fp)

        # Skip the header
        lines = list(reader)[1:]

        # Group lines by audio file
        file_to_segments = {}
        for line in lines:
            if len(line) > 0:
                fname = args.audio_prefix + line[args.audio_column_idx] + args.audio_suffix
                offset = (
                    float(line[args.audio_start_column_idx])
                    if args.audio_start_column_idx is not None
                    else 0
                )
                duration = (
                    float(line[args.audio_duration_column_idx])
                    if args.audio_duration_column_idx is not None
                    else -1
                )
                if fname not in file_to_segments:
                    file_to_segments[fname] = []
                file_to_segments[fname].append((offset, duration))

        # Get list of files and apply sharding
        files = list(file_to_segments.keys())
        start, end = get_shard_range(len(files), args.nshard, args.rank)
        files = files[start:end]

        # Total number of segments in this shard
        total_segments = sum(len(file_to_segments[fname]) for fname in files)

        if n_processed >= total_segments:
            # All segments in this shard have been processed
            print(f"All segments in this shard have been processed.")
            return lambda: iter([]), 0  # Return an empty iterator

        # Skip already processed segments
        skipped_segments = 0
        new_files = []
        for fname in files:
            segments = file_to_segments[fname]
            num_segments = len(segments)

            if skipped_segments >= n_processed:
                # No need to skip segments in this file
                new_files.append(fname)
                continue
            elif skipped_segments + num_segments <= n_processed:
                # Skip the entire file
                skipped_segments += num_segments
                continue
            else:
                # Skip some segments in this file
                num_to_skip_in_file = n_processed - skipped_segments
                file_to_segments[fname] = segments[num_to_skip_in_file:]
                skipped_segments += num_to_skip_in_file
                new_files.append(fname)

        # Update the list of files to process
        files = new_files

        # Recalculate the number of segments to process
        num = total_segments - n_processed

        def iterate():
            for fname in files:
                try:
                    wav, sr = torchaudio.load(fname)
                    n_channels, wav_len = wav.size()
                    assert n_channels < 3
                    if n_channels == 2:
                        wav = (wav[0, :] + wav[1, :]) / 2
                    if sr != MODEL_SR:
                        wav = torchaudio.functional.resample(
                            wav, sr, mdl.sample_rate
                        )
                    wav = wav.to(device)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    continue

                segments = file_to_segments[fname]
                for offset, duration in segments:
                    offset_s = (int(offset * mdl.sample_rate) if offset > 0 else 0)
                    duration_s = (int(duration * mdl.sample_rate) if duration > 0 else -1)
                    if duration_s > 0:
                        wav_chunk = wav[:, offset_s : offset_s + duration_s]
                    else:
                        wav_chunk = wav[:, offset_s:]
                    wav_chunk = wav_chunk.view(1, 1, -1)
                    codes = mdl.wav2code(wav_chunk)
                    yield codes.squeeze().cpu()

        return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()
    print_args(args)

    os.makedirs(args.save_dir, exist_ok=True)
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    encoder = PseudoTextTokenizer(
        args.checkpoint, args.layer, args.km_path, args.max_chunk
    )
    encoder.to(device)
    encoder.eval()

    save_path = os.path.join(args.save_dir, args.split) + (
        f"_{args.rank}_{args.nshard}" if args.nshard > 1 else ""
    )

    # Check how many files have been processed
    save_len_path = save_path + ".len"
    if os.path.exists(save_len_path):
        n_processed = len(np.memmap(save_len_path, dtype=np.int16, mode='r'))
        print(f"Resuming from file {n_processed}")
    else:
        n_processed = 0

    generator, num = get_iterator(args, encoder, device, ptdtype, n_processed)
    iterator = generator()

    n_codes = encoder.C_np.shape[1]
    eos_token = n_codes

    with open(save_path + ".bin", "ab") as bin_data_f, open(
        save_path + ".dur", "ab"
    ) as dur_f, open(save_path + ".len", "ab") as len_f:
        # batch_size = 1000  # Flush after every 1000 files
        for idx, codes in enumerate(tqdm.tqdm(iterator, total=num + n_processed, initial=n_processed)):
            codes = torch.cat((codes, torch.tensor([eos_token], dtype=torch.long)))  # Append EOS token
            codes, duration = torch.unique_consecutive(
                codes, return_counts=True
            )  # Remove adjacent duplicates
            codes = codes.numpy().astype(np.int16)
            duration = duration.numpy().astype(np.int8)
            length = np.array([len(codes)], dtype=np.int16)
            bin_data_f.write(codes.tobytes())
            dur_f.write(duration.tobytes())
            len_f.write(length.tobytes())
            # if idx % batch_size == 0:
            bin_data_f.flush()
            dur_f.flush()
            len_f.flush()

    if args.rank == 0:
        meta = {
            "vocab_size": n_codes + 2,  # Accounting for EOS and PAD tokens
            "eos": n_codes,
            "pad": n_codes + 1,
            "n_q": 1,
            "frame_rate": MODEL_SR // DOWNSAMPLING_FACTOR,
        }
        with open(os.path.join(args.save_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)


if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    main()
