#!/usr/bin/env python
"""
Tokenizes audio files with Mimi and saves the discrete tokens to disk as binary files.
"""

import torch
import torchaudio
import argparse
import csv
import os
import pickle
import tqdm
import numpy as np

# Enable TF32 on supported hardware
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="Path with .csv files pointing to the audio data")
    parser.add_argument("--split", help="Which split", required=True)
    parser.add_argument("--audio-column-idx", type=int, default=2,
                        help="Column in the .csv files containing the audio files")
    parser.add_argument("--audio-start-column-idx", type=int, default=None,
                        help="Column in the .csv files containing the audio offset (if any)")
    parser.add_argument("--audio-duration-column-idx", type=int, default=None,
                        help="Column in the .csv files containing the audio duration (if any)")
    parser.add_argument("--audio-prefix", type=str, default="",
                        help="Prefix for the audio file e.g. '${LIBRILIGHT_PATH}/'")
    parser.add_argument("--audio-suffix", type=str, default="",
                        help="Suffix for the audio file e.g. '.flac'")
    # For Mimi, the --checkpoint argument is used as the Hugging Face model identifier.
    parser.add_argument("--checkpoint", type=str, default="kyutai/mimi",
                        help="Hugging Face model identifier for Mimi")
    parser.add_argument("--save-dir", help="Output path to store the features", required=True)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max_chunk", type=int, default=3200000)
    parser.add_argument("--dtype", type=str, default="float32")
    return parser

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
    print(f"rank {rank+1} of {nshard}, processing {end-start} items ({start}-{end}) out of {tot}")
    return start, end

def get_iterator(args, mdl, device, ptdtype, n_processed=0):
    # Open the CSV (assuming a header row) and group segments by audio file.
    with open(os.path.join(args.data, args.split) + ".csv", newline="") as fp:
        reader = csv.reader(fp)
        lines = list(reader)[1:]  # skip header

        file_to_segments = {}
        for line in lines:
            if len(line) > 0:
                fname = args.audio_prefix + line[args.audio_column_idx] + args.audio_suffix
                offset = float(line[args.audio_start_column_idx]) if args.audio_start_column_idx is not None else 0
                duration = float(line[args.audio_duration_column_idx]) if args.audio_duration_column_idx is not None else -1
                if fname not in file_to_segments:
                    file_to_segments[fname] = []
                file_to_segments[fname].append((offset, duration))

        files = list(file_to_segments.keys())
        start, end = get_shard_range(len(files), args.nshard, args.rank)
        files = files[start:end]

        total_segments = sum(len(file_to_segments[fname]) for fname in files)
        if n_processed >= total_segments:
            print("All segments in this shard have been processed.")
            return lambda: iter([]), 0

        # Skip already processed segments.
        skipped_segments = 0
        new_files = []
        for fname in files:
            segments = file_to_segments[fname]
            num_segments = len(segments)
            if skipped_segments >= n_processed:
                new_files.append(fname)
                continue
            elif skipped_segments + num_segments <= n_processed:
                skipped_segments += num_segments
                continue
            else:
                num_to_skip = n_processed - skipped_segments
                file_to_segments[fname] = segments[num_to_skip:]
                skipped_segments += num_to_skip
                new_files.append(fname)
        files = new_files
        num = total_segments - n_processed

        def iterate():
            for fname in files:
                try:
                    wav, sr = torchaudio.load(fname)
                    n_channels, _ = wav.size()
                    if n_channels == 2:  # Convert stereo to mono
                        wav = (wav[0:1, :] + wav[1:2, :]) / 2
                    if sr != mdl.sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, mdl.sample_rate)
                    wav = wav.to(device)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    continue

                segments = file_to_segments[fname]
                for offset, duration in segments:
                    
                    offset_s = int(offset * mdl.sample_rate) if offset > 0 else 0
                    duration_s = int(duration * mdl.sample_rate) if duration > 0 else -1
                    if duration_s > 0:
                        wav_chunk = wav[:, offset_s:offset_s+duration_s]
                    else:
                        wav_chunk = wav[:, offset_s:]
                    wav_chunk = wav_chunk.view(1, 1, -1)
                    codes = mdl.wav2code(wav_chunk)
                    yield codes.squeeze().cpu()
        return iterate, num

# The MimiTokenizer wraps the Hugging Face MimiModel and feature extractor.
class MimiTokenizer(torch.nn.Module):
    def __init__(self, model_id, max_chunk=3200000):
        super().__init__()
        from transformers import MimiModel, AutoFeatureExtractor
        self.model = MimiModel.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.max_chunk = max_chunk
        self.sample_rate = self.feature_extractor.sampling_rate
        # We assume a codebook size (for discrete tokens) and number of quantizers.
        # (Adjust these if needed based on the model’s configuration.)
        self.codebook_size = 2048
        self.n_q = 8

    def wav2code(self, x):
        """
        Convert a waveform tensor (shape: [1, 1, T]) to discrete tokens.
        The raw audio is processed with the feature extractor and then encoded
        in chunks by the Mimi model.
        """
        # Convert waveform to a 1D numpy array.
        audio_np = x.squeeze().cpu().numpy()
        # Extract model inputs.
        inputs = self.feature_extractor(raw_audio=audio_np, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        total_length = inputs["input_values"].shape[-1]
        codes_chunks = []
        # Process in chunks if needed.
        for start in range(0, total_length, self.max_chunk):
            input_chunk = inputs["input_values"][:, start:start+self.max_chunk]
            attn_chunk = inputs.get("attention_mask", None)
            if attn_chunk is not None:
                attn_chunk = attn_chunk[:, start:start+self.max_chunk]
            with torch.no_grad():
                encoder_outputs = self.model.encode(input_chunk, attn_chunk)
                # Assume encoder_outputs.audio_codes has shape (batch, n_q, T')
                codes_chunks.append(encoder_outputs.audio_codes)
        if codes_chunks:
            codes = torch.cat(codes_chunks, dim=-1)
        else:
            codes = torch.empty(0, device=x.device)
        # Remove the batch dimension so that codes has shape (n_q, T')
        codes = codes.squeeze(0)
        return codes[:self.n_q]

def main():
    parser = get_parser()
    args = parser.parse_args()
    print_args(args)
    os.makedirs(args.save_dir, exist_ok=True)
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # Initialize the Mimi tokenizer using the Hugging Face model identifier.
    tokenizer = MimiTokenizer(args.checkpoint, args.max_chunk)
    tokenizer.to(device)
    tokenizer.eval()

    save_path = os.path.join(args.save_dir, args.split) + (f"_{args.rank}_{args.nshard}" if args.nshard > 1 else "")
    # Check for resuming (how many segments have already been processed).
    save_len_path = save_path + ".len"
    if os.path.exists(save_len_path):
        n_processed = len(np.memmap(save_len_path, dtype=np.uint16, mode='r'))
        print(f"Resuming from {n_processed} segments")
    else:
        n_processed = 0

    generator, num = get_iterator(args, tokenizer, device, ptdtype, n_processed)
    iterator = generator()

    # Define EOS token (we use the codebook size as EOS, similar to your SpeechTokenizer script).
    eos_token = tokenizer.codebook_size  
    with open(save_path + ".bin", "ab") as bin_f, open(save_path + ".len", "ab") as len_f:
        for codes in tqdm.tqdm(iterator, total=num+n_processed, initial=n_processed):
            # `codes` is expected to be a tensor of shape (n_q, T').
            # Append an EOS token for each quantizer.
            eos_col = torch.full((tokenizer.n_q, 1), eos_token, dtype=torch.int16)
            codes_np = torch.cat([codes, eos_col], dim=1).numpy().astype(np.int16)
            # Save the tokens transposed (so that the time dimension is contiguous in memory).
            bin_f.write(codes_np.T.tobytes())
            sample_len = np.array([codes_np.shape[1]], dtype=np.int64)
            len_f.write(sample_len.tobytes())
            bin_f.flush()
            len_f.flush()

    if args.rank == 0:
        codebook_size = 2**tokenizer.model.bits_per_codebook
        meta = {
            "vocab_size": codebook_size + 2,  # Accounting for EOS and PAD tokens.
            "eos": codebook_size,
            "pad": codebook_size + 1,
            "n_q": tokenizer.n_q,
            "frame_rate": tokenizer.model.config.frame_rate
        }
        with open(os.path.join(args.save_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    main()