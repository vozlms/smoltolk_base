import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from torch.utils.data import IterableDataset, DataLoader


@dataclass
class DatasetArgs:
    audio_datasets: List[str] = field(default_factory=list)
    txt_datasets: List[str] = field(default_factory=list)
    audiotxt_datasets: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    p_strategies: List[float] = field(default_factory=list)
    audio_tokens: str = ""
    txt_tokens: str = ""
    n_codebooks_to_use: int = -1
    block_size: int = 1024
    interleaved_p_audio: Optional[float] = 0.5
    debug: Optional[bool] = False
    pred_txt_in_interleaved: Optional[bool] = True
    audio_datasets_probs: Optional[List[float]] = None
    txt_datasets_probs: Optional[List[float]] = None
    audiotxt_datasets_probs: Optional[List[float]] = None


class TextAudioCodecLMDataset(IterableDataset):
    def __init__(self, config: DatasetArgs):
        super().__init__()
        self.config = config
        # Sanity check args
        assert any(config.p_strategies), "Some training task must be selected, i.e. p_strategies cannot be all zeros"
        assert all(x >= 0 for x in config.p_strategies), "p_strategies elements must be >= 0"
        models_audio = False
        models_txt = False
        self.is_crossmodal = is_crossmodal = False
        if config.p_strategies[0]:
            models_audio = True
            assert config.audio_tokens, "If doing audio language modeling setting audio_tokens is required"
        if config.p_strategies[1]:
            models_txt = True
            assert config.txt_tokens, "If doing text language modeling setting txt_tokens is required"
        if any(config.p_strategies[2:]):
            models_audio = True
            models_txt = True
            is_crossmodal = True
            assert config.txt_tokens, \
                "If doing ASR, TTS or interleaving, setting audio_tokens and txt_tokens is required"
        # Define eval strategies according to selected training tasks
        sampling_strategies = [
            ('get_mono_sample', "audio", "mono_audio"),
            ('get_mono_sample', "txt", "mono_txt"),
            ('get_prefix_sample', "audio", "prefix_audio"),
            ('get_prefix_sample', "txt", "prefix_txt"),
            ('get_interleaved_sample', config.interleaved_p_audio, "interleaved")
        ]
        self.samplers = []
        self.sampler_probs = []
        for train_task_prob, task_sampler in zip(config.p_strategies, sampling_strategies):
            if train_task_prob:
                self.samplers.append(task_sampler)
                self.sampler_probs.append(train_task_prob)
        # define defaults 
        self.audio_ncodebooks = 1
        self.audio_vocabsize, self.audio_pad_token, self.audio_eos_token, self.audio_sr = None, None, None, None
        self.txt_vocabsize, self.txt_eos_token = None, None
        self.bop_token, self.eop_token, self.swt_token = None, None, None
        # Load metadata
        if models_audio:
            # Load audio metadata
            audio_meta_path = None
            for dataset in self.config.audio_datasets + self.config.audiotxt_datasets:
                audio_meta_path = os.path.join("data", dataset, self.config.audio_tokens, 'meta.pkl')
                if os.path.isfile(audio_meta_path):
                    break # all audio datasets must use the same metadata
            if audio_meta_path is not None:
                with open(audio_meta_path, 'rb') as f:
                    audio_meta = pickle.load(f)
                self.audio_vocabsize = audio_meta['vocab_size']
                self.audio_pad_token = audio_meta['pad']
                self.audio_eos_token = audio_meta['eos']
                self.audio_sr = audio_meta['frame_rate']
                self.audio_ncodebooks = audio_meta['n_q']
                # If dropping some codebooks, update the n_codebooks variable accordingly
                if self.config.n_codebooks_to_use > 0:
                    self.audio_ncodebooks = self.config.n_codebooks_to_use
        if models_txt:
            # Load text metadata if needed
            txt_meta_path = None
            for dataset in self.config.txt_datasets + self.config.audiotxt_datasets:
                txt_meta_path = os.path.join("data", dataset, self.config.txt_tokens, 'meta.pkl')
                if os.path.isfile(txt_meta_path):
                    break # all text datasets must use the same metadata
            if txt_meta_path is not None:
                with open(txt_meta_path, 'rb') as f:
                    txt_meta = pickle.load(f)
                self.txt_vocabsize = txt_meta["vocab_size"]
                self.txt_eos_token = txt_meta["eos"]
        if is_crossmodal:
            self.bop_token = self.txt_vocabsize
            self.eop_token = self.txt_vocabsize + 1
            self.swt_token = self.txt_vocabsize + 2
            self.txt_vocabsize += 3
        ##################################################
        #  1) AUDIO LOADING
        ##################################################
        if config.p_strategies[0]:
            print("Loading audio data ...")
            # If user provided dataset-level probabilities, we group shards by dataset
            if config.audio_datasets_probs:
                # Build a dict: dataset -> probability
                audio_datasets_probs = dict(zip(config.audio_datasets, config.audio_datasets_probs))
                # Group shards by dataset
                dataset_to_shards = defaultdict(list)
                found_data = False
                for dataset in config.audio_datasets:
                    for split in config.splits:
                        audio_mmap = self.get_data_mmaps(split, dataset, config.audio_tokens)
                        if audio_mmap is not None:
                            found_data = True
                            dataset_to_shards[dataset].append(audio_mmap)
                assert found_data, (
                    f"No audio data found for audio datasets '{config.audio_datasets}', splits {config.splits}"
                )
                # Build self.audio_shards & self.audio_shards_probs
                self.audio_shards = []
                self.audio_shards_probs = []
                for dataset, shards_list in dataset_to_shards.items():
                    dataset_size = sum(len(shard['audio_data']) for shard in shards_list)
                    # Probability weight assigned by the user to this entire dataset:
                    dataset_weight = audio_datasets_probs[dataset]
                    for shard in shards_list:
                        self.audio_shards.append(shard)
                        # Probability = dataset_weight * ( shard_size / sum_of_shard_sizes_in_dataset )
                        self.audio_shards_probs.append(
                            dataset_weight * (len(shard['audio_data']) / dataset_size)
                        )
            else:
                # No dataset-level probs -> treat all shards from all datasets as one pool.
                all_audio_shards = []
                all_audio_lens = []
                found_data = False
                for dataset in config.audio_datasets:
                    for split in config.splits:
                        audio_mmap = self.get_data_mmaps(split, dataset, config.audio_tokens)
                        if audio_mmap is not None:
                            found_data = True
                            all_audio_shards.append(audio_mmap)
                            all_audio_lens.append(len(audio_mmap['audio_data']))
                assert found_data, (
                    f"No audio data found for audio datasets '{config.audio_datasets}', splits {config.splits}"
                )
                total_size = sum(all_audio_lens)
                self.audio_shards = all_audio_shards
                self.audio_shards_probs = [
                    shard_len / total_size for shard_len in all_audio_lens
                ]
        ##################################################
        #  2) TEXT LOADING
        ##################################################
        if config.p_strategies[1]:
            print("Loading text data ...")
            if config.txt_datasets_probs:
                # Build dataset->prob dict
                txt_datasets_probs = dict(zip(config.txt_datasets, config.txt_datasets_probs))
                dataset_to_shards = defaultdict(list)
                found_data = False
                for dataset in config.txt_datasets:
                    for split in config.splits:
                        txt_mmap = self.get_data_mmaps(split, dataset, config.txt_tokens)
                        if txt_mmap is not None:
                            found_data = True
                            dataset_to_shards[dataset].append(txt_mmap)
                assert found_data, (
                    f"No text data found for dataset '{config.txt_datasets}', splits {config.splits}"
                )
                self.txt_shards = []
                self.txt_shards_probs = []
                for dataset, shards_list in dataset_to_shards.items():
                    dataset_size = sum(len(shard['txt_data']) for shard in shards_list)
                    dataset_weight = txt_datasets_probs[dataset]
                    for shard in shards_list:
                        self.txt_shards.append(shard)
                        self.txt_shards_probs.append(
                            dataset_weight * (len(shard['txt_data']) / dataset_size)
                        )
            else:
                # No user-provided dataset-level probabilities
                all_txt_shards = []
                all_txt_lens = []
                found_data = False
                for dataset in config.txt_datasets:
                    for split in config.splits:
                        txt_mmap = self.get_data_mmaps(split, dataset, config.txt_tokens)
                        if txt_mmap is not None:
                            found_data = True
                            all_txt_shards.append(txt_mmap)
                            all_txt_lens.append(len(txt_mmap['txt_data']))
                assert found_data, (
                    f"No text data found for dataset '{config.txt_datasets}', splits {config.splits}"
                )
                total_size = sum(all_txt_lens)
                self.txt_shards = all_txt_shards
                self.txt_shards_probs = [
                    shard_len / total_size for shard_len in all_txt_lens
                ]
        ##################################################
        #  3) AUDIO+TEXT LOADING (CROSSMODAL)
        ##################################################
        if any(config.p_strategies[2:]):
            print("Loading audio-text data ...")
            if config.audiotxt_datasets_probs:
                audiotxt_datasets_probs = dict(
                    zip(config.audiotxt_datasets, config.audiotxt_datasets_probs)
                )
                dataset_to_shards = defaultdict(list)
                found_data = False
                for dataset in config.audiotxt_datasets:
                    for split in config.splits:
                        audio_mmap = self.get_data_mmaps(split, dataset, config.audio_tokens)
                        txt_mmap = self.get_data_mmaps(split, dataset, config.txt_tokens)
                        if audio_mmap is not None and txt_mmap is not None:
                            assert len(audio_mmap["audio_lens"]) == len(txt_mmap["txt_lens"]), (
                                f"Not a 1-to-1 correspondence between audio samples "
                                f"({len(audio_mmap['audio_lens'])}) and text samples "
                                f"({len(txt_mmap['txt_lens'])})"
                            )
                            # Merge them so that each shard can be used as a single structure
                            audio_mmap.update(txt_mmap)
                            dataset_to_shards[dataset].append(audio_mmap)
                            found_data = True
                assert found_data, (
                    f"No paired data found for audio-text dataset '{config.audiotxt_datasets_probs}', splits {config.splits}"
                )
                # Build self.audiotxt_shards & self.audiotxt_shards_probs
                self.audiotxt_shards = []
                self.audiotxt_shards_probs = []
                self.audiotxt_shards_lens = []
                for dataset, shards_list in dataset_to_shards.items():
                    dataset_size = sum(len(shard['txt_data']) for shard in shards_list)
                    dataset_weight = audiotxt_datasets_probs[dataset]
                    for shard in shards_list:
                        self.audiotxt_shards.append(shard)
                        shard_len = len(shard['txt_data'])
                        self.audiotxt_shards_lens.append(shard_len)
                        self.audiotxt_shards_probs.append(
                            dataset_weight * (shard_len / dataset_size)
                        )
            else:
                # No dataset-level probabilities for crossmodal
                all_audiotxt_shards = []
                all_audiotxt_lens = []
                found_data = False
                for dataset in config.audiotxt_datasets:
                    for split in config.splits:
                        audio_mmap = self.get_data_mmaps(split, dataset, config.audio_tokens)
                        txt_mmap = self.get_data_mmaps(split, dataset, config.txt_tokens)
                        if audio_mmap is not None and txt_mmap is not None:
                            assert len(audio_mmap["audio_lens"]) == len(txt_mmap["txt_lens"]), (
                                f"Not a 1-to-1 correspondence between audio samples "
                                f"({len(audio_mmap['audio_lens'])}) and text samples "
                                f"({len(txt_mmap['txt_lens'])})"
                            )
                            audio_mmap.update(txt_mmap)
                            all_audiotxt_shards.append(audio_mmap)
                            all_audiotxt_lens.append(len(audio_mmap['txt_data']))
                            found_data = True
                assert found_data, (
                    f"No paired data found for audio-text dataset '{config.audiotxt_datasets_probs}', splits {config.splits}"
                )
                total_size = sum(all_audiotxt_lens)
                self.audiotxt_shards = all_audiotxt_shards
                self.audiotxt_shards_lens = all_audiotxt_lens
                self.audiotxt_shards_probs = [
                    shard_len / total_size for shard_len in all_audiotxt_lens
                ]
            # If the last p_strategy indicates "interleaving", gather aligned subsets
            if config.p_strategies[-1]:
                self.alignedaudiotxt_shards = []
                self.alignedaudiotxt_shards_lens = []
                for shard, shard_len in zip(self.audiotxt_shards, self.audiotxt_shards_lens):
                    if "wrds_timestamps" in shard:
                        self.alignedaudiotxt_shards.append(shard)
                        self.alignedaudiotxt_shards_lens.append(shard_len)
                assert self.alignedaudiotxt_shards, (
                    f"No paired data with timestamps found for audio-text datasets "
                    f"{config.audiotxt_datasets}, splits {config.splits}"
                )
                # If the user gave crossmodal dataset probabilities, we must also build
                # alignedaudiotxt_shards_probs accordingly:
                if config.audiotxt_datasets_probs:
                    aligned_dataset_to_shards = defaultdict(list)
                    for shard in self.alignedaudiotxt_shards:
                        dset_name = shard["dataset"]
                        aligned_dataset_to_shards[dset_name].append(shard)

                    self.alignedaudiotxt_shards_probs = []
                    for dataset, shards_list in aligned_dataset_to_shards.items():
                        dataset_size = sum(len(s['txt_data']) for s in shards_list)
                        dataset_weight = audiotxt_datasets_probs[dataset]
                        for s in shards_list:
                            self.alignedaudiotxt_shards_probs.append(
                                dataset_weight * (len(s['txt_data']) / dataset_size)
                            )
                else:
                    # No dataset-level probs => just proportion by shard size
                    total_size_aligned = sum(self.alignedaudiotxt_shards_lens)
                    self.alignedaudiotxt_shards_probs = [
                        length / total_size_aligned for length in self.alignedaudiotxt_shards_lens
                    ]
        # This ensures we sample exactly block_size tokens + 1 for next-token prediction
        self.sample_len = config.block_size + 1

    def get_data_mmaps(self, split, dataset, tokens_source):
        data_dir = os.path.join("data", dataset, tokens_source, split)
        if os.path.exists(data_dir + '.bin'):
            vocab_size = self.txt_vocabsize if tokens_source == self.config.txt_tokens else self.audio_vocabsize
            tokens_dtype = np.uint32 if vocab_size > 2**16 else np.uint16
            data_mmap, m_bnds, m_sample_lens, data_durs = self.load_data(data_dir, tokens_dtype)
        else:
            return None
        modality = 'audio' if tokens_source == self.config.audio_tokens else 'txt'
        if modality == 'audio':
            try:
                data_mmap = data_mmap.reshape(np.sum(m_sample_lens), self.audio_ncodebooks)
            except ValueError: # TODO: ugly hack, need to find out why ls960.val lengths are in uint16 instead of uint64
                m_sample_lens = np.memmap(data_dir + '.len', dtype=np.uint16, mode='r')
                m_bnds = np.concatenate((np.array([0]), np.cumsum(m_sample_lens)), dtype=int)
                data_mmap = data_mmap.reshape(np.sum(m_sample_lens), self.audio_ncodebooks)
        assert len(data_mmap) > self.config.block_size, "This shard is way too small? Investigate."
        print(f"Length of {dataset}.{tokens_source}.{split}: {len(data_mmap)} tokens")
        
        data = {
            'dataset': dataset,
            'split': split,
            f'{modality}_data': data_mmap,
            f'{modality}_durs': data_durs,
            f'{modality}_bnds': m_bnds,
            f'{modality}_lens': m_sample_lens,
            "overfit_idx": np.argmax(m_sample_lens),
            f'{modality}_probs': m_sample_lens / np.sum(m_sample_lens)  # If sampling over individual samples, each will be sampled according to its length, i.e., uniform
        }
        if self.config.p_strategies[-1]: # if we're doing interleaving
            # If there are timestamps, load them
            if os.path.exists(data_dir + '.wrds_tms'):
                wrds_timestamps = np.memmap(data_dir + '.wrds_tms', dtype=np.float32, mode='r')
                wrds_timestamps_lens = np.memmap(data_dir + '.wrds_tms_len', dtype=np.uint64, mode='r')
                wrds_timestamps_bnds = np.concatenate((np.array([0], dtype=np.uint64), np.cumsum(wrds_timestamps_lens)))
                data.update({
                    'wrds_timestamps': wrds_timestamps,
                    'wrds_timestamps_lens': wrds_timestamps_lens,
                    'wrds_timestamps_bnds': wrds_timestamps_bnds
                })

                nan_inds = np.where(np.isnan(data['wrds_timestamps']))[0]    
                if len(nan_inds) > 0:
                    print(f"WARNING: Dataset {data['dataset']} wrds_timestamps contains nans at: {nan_inds}. These samples will lead to a resampling.")

                if os.path.exists(data_dir + '.wrd_smps'):
                    data['wrd_smps'] = np.memmap(data_dir + '.wrd_smps', dtype=np.uint64, mode='r')
                    data['wrd_lens'] = np.memmap(data_dir + '.wrd_lens', dtype=np.uint64, mode='r')
                    data['wrd_bnds'] = np.memmap(data_dir + '.wrd_bnds', dtype=np.uint64, mode='r')
                else:
                    # Pre-compute word-alignment related data
                    self.prepare_words_counts(data, data_dir)
            # If audio tokens are variable rate, and there are text timestamps
            # get mappings from fixed to variable rate indices
            txt_data_dir = os.path.join("data", dataset, self.config.txt_tokens, split)
            if data[f'{modality}_durs'] is not None and os.path.exists(txt_data_dir + '.wrds_tms'):
                if os.path.exists(data_dir + '.short_idxs'):
                    data[f'{modality}_short_idxs'] = np.memmap(data_dir + '.short_idxs', dtype=np.uint64, mode='r')
                    data[f'{modality}_long_bnds'] = np.memmap(data_dir + '.long_bnds', dtype=np.uint64, mode='r')
                else:
                    # Pre-compute duration indexing needed for interleaved 
                    self.prepare_collapsed2uncollapsed_idxs(data, modality, data_dir)
        return data

    def load_data(self, data_dir, tokens_dtype):
        try:
            m_sample_lens = np.memmap(data_dir + '.len', dtype=np.uint64, mode='r')
        except ValueError:
            print(f".len file stored in {data_dir} does not match dtype uint64, trying uint16")
            m_sample_lens = np.memmap(data_dir + '.len', dtype=np.uint16, mode='r')
        data_mmap = np.memmap(data_dir + '.bin', dtype=tokens_dtype, mode="r")
        # Compute sample boundaries
        m_bnds = np.concatenate((np.array([0]), np.cumsum(m_sample_lens)), dtype=int)
        # load durations
        data_durs = None
        if os.path.exists(data_dir + '.dur'):
            data_durs = np.memmap(data_dir + '.dur', dtype=np.uint8, mode="r")
        return data_mmap, m_bnds, m_sample_lens, data_durs

    def prepare_words_counts(self, data, data_dir):
        smp_ix = 1
        total = len(data['wrds_timestamps_bnds'])
        total_token_bnds = len(data['txt_bnds'])
        last_smp_ix = 0
        # Create memory-mapped arrays with appropriate shapes
        wrd_smps = np.memmap(data_dir + '.wrd_smps', dtype=np.uint64, mode='w+', shape=(total,))
        wrd_lens = np.memmap(data_dir + '.wrd_lens', dtype=np.uint64, mode='w+', shape=(len(data['txt_lens']),))
        # Compute word alignment data
        for i in tqdm(range(total), total=total, desc='Preparing word alignment data for interleaving.'):
            txt_ix = data['wrds_timestamps_bnds'][i]
            while smp_ix < (total_token_bnds - 1) and txt_ix >= data['txt_bnds'][smp_ix]:
                wrd_lens[smp_ix - 1] = i - last_smp_ix
                smp_ix += 1
                last_smp_ix = i
            wrd_smps[i] = smp_ix - 1
        wrd_lens[smp_ix - 1] = i - last_smp_ix
        # Compute cumulative sum for wrd_bnds
        wrd_bnds_array = np.concatenate([np.array([0], dtype=np.uint64), np.cumsum(wrd_lens)], dtype=np.uint64)
        wrd_bnds = np.memmap(data_dir + '.wrd_bnds', dtype=np.uint64, mode='w+', shape=wrd_bnds_array.shape)
        wrd_bnds[:] = wrd_bnds_array[:]
        wrd_bnds.flush()
        # Flush changes to disk
        wrd_smps.flush()
        wrd_lens.flush()
        # Assign memory-mapped arrays to data dictionary
        data['wrd_smps'] = wrd_smps
        data['wrd_lens'] = wrd_lens
        data['wrd_bnds'] = wrd_bnds
        return wrd_smps, wrd_lens, wrd_bnds

    def prepare_collapsed2uncollapsed_idxs(self, data, modality, data_dir):
        uncompressed_len = np.sum(data['audio_durs'])
        index_to_compressed = np.memmap(data_dir + '.short_idxs', dtype=np.uint64, mode='w+',shape=(uncompressed_len,))
        uncompressed_bnds = np.memmap(data_dir + '.long_bnds', dtype=np.uint64, mode='w+', shape=(len(data['audio_bnds']),))
        uncollapsed_pointer = 0
        current_sample = 0        
        for idx, dur in enumerate(tqdm(data[f'{modality}_durs'], desc='Preparing uncollapsed to collapsed token indexing for interleaving.')):
            if idx == data[f'{modality}_bnds'][current_sample]:
                uncompressed_bnds[current_sample] = uncollapsed_pointer
                current_sample += 1
                in_sample_collapsed_idx = 0
            index_to_compressed[uncollapsed_pointer:uncollapsed_pointer + dur] = in_sample_collapsed_idx
            in_sample_collapsed_idx += 1
            uncollapsed_pointer += dur
        index_to_compressed.flush()
        uncompressed_bnds.flush()
        data[f'{modality}_short_idxs'] = index_to_compressed
        data[f'{modality}_long_bnds'] = uncompressed_bnds

    def get_mono_sample(self, data, modality):
        # select a random chunk of length block_size + 1 (because of next token prediction) from the dataset
        ix = random.choice(range(len(data[modality + '_data']) - self.sample_len))
        sample = data[modality + '_data'][ix:ix + self.sample_len]
        audio_input_mask = np.zeros((self.sample_len,), dtype=bool)
        audio_preds_mask = np.zeros((self.sample_len,), dtype=bool)
        txt_preds_mask = np.zeros((self.sample_len,), dtype=bool)
        if modality == 'txt':
            sample = sample.reshape(-1, 1).repeat(self.audio_ncodebooks, 1)
            txt_preds_mask[:] = True
        else:
            audio_input_mask[:] = True
            audio_preds_mask[:] = True
        return sample[:, :self.audio_ncodebooks], audio_input_mask, audio_preds_mask, txt_preds_mask
            
    def get_prefix_sample(self, data, condition_modality):
        prefix_sample_list = []
        audio_input_mask_list = []
        audio_preds_mask_list = []
        txt_preds_mask_list = []
        total_length = 0
        while total_length < self.sample_len:
            sample_ix = random.choice(range(len(data['audio_lens'])))
            # Get text corresponding to the sample
            txt_chunk = data['txt'][data['txt_bnds'][sample_ix]:data['txt_bnds'][sample_ix + 1] - 1]  # -1 to omit eos
            # Get audio corresponding to the sample
            sample_audio = audio_chunk = data['audio_data'][data['audio_bnds'][sample_ix]:data['audio_bnds'][sample_ix + 1] - 1]  # -1 to omit eos
            # Use only up to n_codebooks
            audio_chunk = audio_chunk[:, :self.audio_ncodebooks]
            sample_audio = sample_audio[:, :self.audio_ncodebooks]
            # Define conditioning and target sequence
            if condition_modality == 'txt':
                condition_chunk = txt_chunk.reshape(-1, 1).repeat(self.audio_ncodebooks, 1)
                # Use eos token to indicate the end of target generation
                target_chunk = np.concatenate((audio_chunk, np.array([self.audio_eos_token]).repeat(self.audio_ncodebooks).reshape(1, -1)))
            else:
                condition_chunk = audio_chunk
                # Use eos token to indicate the end of target generation
                target_chunk = np.concatenate((
                    txt_chunk.reshape(-1, 1).repeat(self.audio_ncodebooks, 1),
                    np.array([self.txt_eos_token]).repeat(self.audio_ncodebooks).reshape(1, -1)
                ))
            # Create the prefix sample as <bop>condition<eop>target<eos>
            condition_chunk = np.concatenate((
                np.array([self.bop_token]).repeat(self.audio_ncodebooks).reshape(1, -1),
                condition_chunk,
                np.array([self.eop_token]).repeat(self.audio_ncodebooks).reshape(1, -1)
            ))
            prefix_sample = np.concatenate((condition_chunk, target_chunk))
            # Create masks
            audio_input_mask = np.zeros((len(prefix_sample),), dtype=bool)
            audio_preds_mask = np.zeros((len(prefix_sample),), dtype=bool)
            txt_preds_mask = np.zeros((len(prefix_sample),), dtype=bool)
            if condition_modality == 'txt':
                audio_input_mask[len(condition_chunk):] = True  # The rest is audio
                audio_preds_mask[len(condition_chunk):] = True  # We want to predict the audio target
            else:
                audio_input_mask[1:len(condition_chunk) - 1] = True  # The rest is audio
                txt_preds_mask[len(condition_chunk):] = True  # We want to predict the text target
            prefix_sample_list.append(prefix_sample)
            audio_input_mask_list.append(audio_input_mask)
            audio_preds_mask_list.append(audio_preds_mask)
            txt_preds_mask_list.append(txt_preds_mask)
            total_length += len(prefix_sample)
        prefix_sample = np.concatenate(prefix_sample_list)[:self.sample_len]
        audio_input_mask = np.concatenate(audio_input_mask_list)[:self.sample_len]
        audio_preds_mask = np.concatenate(audio_preds_mask_list)[:self.sample_len]
        txt_preds_mask = np.concatenate(txt_preds_mask_list)[:self.sample_len]
        return prefix_sample, audio_input_mask, audio_preds_mask, txt_preds_mask

    def get_interleaved_sample(self, data, p_audio=0.5, margin_samples=0, max_tries=3):
        # Find the end of data margin (worst case scenario 1 sample of block_size words made of a single token + <eos>)
        # TODO: this margin is too conservative, the last samples will never be sampled
        #       a better option is to retry if we run out of data, or even better cycle back to the beginning of the samples when running out
        #margin = np.where(np.cumsum(wrd_lens[-block_size//2::-1])>block_size)[0].min()

        def sample_interleaved(data, p_audio, margin_samples):
            # Find a random sample
            # txt_bnds has one more element than the number of samples
            # That is why we have the -1
                # Find a random sample
            sample_count = data['txt_bnds'].shape[0] - 1
            smp_ix = random.randint(0, sample_count - margin_samples - 1)
            # Find a random word in the sample
            wrd_smp_ix = random.randint(0, data['wrd_lens'][smp_ix] - 1)
            wrd_ix = int(data['wrd_bnds'][smp_ix]) + wrd_smp_ix
            # Total word count
            wrd_count = data['wrds_timestamps_bnds'].shape[0]
            # Adjust wrd_ix to be within bounds
            wrd_ix = wrd_ix % (wrd_count - 1)

            # Generate sequences of modalities with specified lengths
            wrd_is_text = []
            current_modality = random.choice([True, False]) # Start randomly 
            # current_modality = True  # Start with text
            while len(wrd_is_text) < self.sample_len:
                if current_modality:  # Text sequence
                    seq_len = random.randint(10, 31)  # Between 10 and 30 words
                else:  # Audio sequence
                    seq_len = random.randint(5, 16)  # Between 5 and 15 words
                # Adjust seq_len if it exceeds the remaining sample length
                seq_len = min(seq_len, self.sample_len - len(wrd_is_text))
                # Extend the modality list
                wrd_is_text.extend([current_modality] * seq_len)
                # Switch modality for the next sequence
                current_modality = not current_modality
            wrd_is_text = np.array(wrd_is_text[:self.sample_len])  # Ensure length matches sample_len

            # Identify runs of the same modality
            modality_changes = np.where(wrd_is_text[:-1] != wrd_is_text[1:])[0] + 1
            run_starts = np.concatenate(([0], modality_changes))
            run_ends = np.concatenate((modality_changes, [len(wrd_is_text)]))

            # Resulting array
            r = np.zeros((self.sample_len, self.audio_ncodebooks), dtype=np.int64)
            # Prepare the empty masks
            audio_input_mask = np.zeros(self.sample_len, dtype=bool)
            audio_preds_mask = np.zeros(self.sample_len, dtype=bool)
            txt_preds_mask = np.zeros(self.sample_len, dtype=bool)

            # Index (offset) on the resulting array
            r_off = 0
            prev_modality = None

            for start, end in zip(run_starts, run_ends):
                if r_off >= self.sample_len:
                    break
                next_modality = wrd_is_text[start]
                if prev_modality is not None and next_modality != prev_modality:
                    # Output the switching token
                    r[r_off, :] = self.swt_token
                    audio_input_mask[r_off] = False
                    audio_preds_mask[r_off] = False
                    txt_preds_mask[r_off] = False
                    r_off += 1
                    if r_off >= self.sample_len:
                        break
                run_len = end - start
                run_len = min(run_len, self.sample_len - r_off)
                if run_len <= 0:
                    continue

                if next_modality:  # Text sequence
                    total_tokens = 0
                    while total_tokens < run_len and r_off < self.sample_len:
                        next_wrd_ix = (wrd_ix + 1) % (wrd_count - 1)  # Wrap-around
                        if (wrd_ix >= wrd_count) or \
                        (data['txt_data'][data['wrds_timestamps_bnds'][wrd_ix]] == self.txt_eos_token) or \
                        (data['wrd_smps'][wrd_ix] != data['wrd_smps'][next_wrd_ix]):
                            # Handle <eos> token or end of sample
                            pos = min(data['wrds_timestamps_bnds'][wrd_ix], data['txt_data'].shape[0]-1)
                            tok = data['txt_data'][pos]
                            assert tok == self.txt_eos_token, f"Expecting EOS ({self.txt_eos_token}), but got {tok}"
                            r[r_off, :] = tok
                            audio_input_mask[r_off] = False
                            audio_preds_mask[r_off] = False
                            txt_preds_mask[r_off] = True if self.config.pred_txt_in_interleaved else False
                            r_off += 1
                            wrd_ix = (wrd_ix + 1) % (wrd_count - 1)
                            total_tokens += 1
                        else:
                            b = data['wrds_timestamps_bnds'][wrd_ix].astype(np.int64)
                            e = data['wrds_timestamps_bnds'][next_wrd_ix].astype(np.int64)
                            token_len = e - b
                            token_len = min(token_len, self.sample_len - r_off)
                            chunk = data['txt_data'][b:b+token_len]
                            r[r_off:r_off+token_len, :] = chunk[:, np.newaxis]
                            audio_input_mask[r_off:r_off+token_len] = False
                            audio_preds_mask[r_off:r_off+token_len] = False
                            txt_preds_mask[r_off:r_off+token_len] = True if self.config.pred_txt_in_interleaved else False
                            r_off += token_len
                            total_tokens += 1
                            wrd_ix = (wrd_ix + 1) % (wrd_count - 1)
                else:  # Audio sequence
                    total_tokens = 0
                    while total_tokens < run_len and r_off < self.sample_len:
                        if (wrd_ix + 1 >= wrd_count) or \
                        (wrd_ix > 0 and data['wrd_smps'][wrd_ix - 1] != data['wrd_smps'][wrd_ix]):
                            # Handle <eos> token or end of sample
                            pos = min(data['wrds_timestamps_bnds'][wrd_ix - 1], data['txt_data'].shape[0]-1)
                            tok = data['txt_data'][pos]
                            assert tok == self.txt_eos_token, f"Expecting EOS ({self.txt_eos_token}), but got {tok}"
                            r[r_off, :] = self.audio_eos_token
                            audio_input_mask[r_off] = True
                            audio_preds_mask[r_off] = True
                            txt_preds_mask[r_off] = False
                            r_off += 1
                            wrd_ix = (wrd_ix + 2) % (wrd_count - 1)
                            total_tokens += 1
                        else:
                            prev_wrd_ix = (wrd_ix - 1) if wrd_ix > 0 else None
                            if prev_wrd_ix is not None and data['wrd_smps'][prev_wrd_ix] == data['wrd_smps'][wrd_ix]:
                                a_be = data['wrds_timestamps'][prev_wrd_ix:wrd_ix+1]
                            else:
                                a_be = np.array([0, data['wrds_timestamps'][wrd_ix]])
                            smp_ix = data['wrd_smps'][wrd_ix]
                            a_b, a_e = (a_be * self.audio_sr).astype(np.int64)
                            a_off = data['audio_bnds'][smp_ix]
                            if data['audio_durs'] is not None:
                                max_a_e = int(data['audio_long_bnds'][int(smp_ix) + 1]) - int(data['audio_long_bnds'][smp_ix]) - 1
                                a_e = min(a_e, max_a_e)
                                a_off_long = int(data['audio_long_bnds'][smp_ix])
                                b_idx = a_off_long + a_b
                                e_idx = a_off_long + a_e
                                b = a_off + int(data['audio_short_idxs'][b_idx])
                                e = a_off + int(data['audio_short_idxs'][int(e_idx)])
                                token_len = e - b
                            else:
                                b = a_off + a_b
                                token_len = a_e - a_b
                            token_len = max(min(token_len, self.sample_len - r_off), 1)
                            chunk = data['audio_data'][b:b+token_len]
                            r[r_off:r_off+token_len] = chunk[:, :self.audio_ncodebooks]
                            audio_input_mask[r_off:r_off+token_len] = True
                            audio_preds_mask[r_off:r_off+token_len] = True
                            txt_preds_mask[r_off:r_off+token_len] = False
                            r_off += token_len
                            total_tokens += 1
                            wrd_ix = (wrd_ix + 1) % (wrd_count - 1)
                prev_modality = next_modality
    
            return r, audio_input_mask, audio_preds_mask, txt_preds_mask

        for i in range(max_tries):
            try:
                return sample_interleaved(data, p_audio, margin_samples)
            except Exception:
                if i == max_tries-1:
                    raise

    def __iter__(self):
        while True:
            # Datasets are sampled according to their size, i.e. uniformly across tokens
            # TODO: maybe allow setting per-dataset sampling ratios?
            sampler, sampler_arg, sampler_label = random.choices(self.samplers, weights=self.sampler_probs, k=1)[0]
            if sampler == "get_mono_sample" and sampler_arg == "txt":
                data = random.choices(self.txt_shards, weights=self.txt_shards_probs, k=1)[0]
            elif sampler == "get_prefix_sample":
                data = random.choices(self.audiotxt_shards, weights=self.audiotxt_shards_probs, k=1)[0]
            elif sampler == "get_interleaved_sample":
                data = random.choices(self.alignedaudiotxt_shards,
                                        weights=self.alignedaudiotxt_shards_probs, k=1)[0]
            else:
                data = random.choices(self.audio_shards, weights=self.audio_shards_probs, k=1)[0]
            # Save chosen sampler + dataset + split for external tracking
            self.last_choice = {
                "sampler_label": sampler_label,     # e.g. "mono_audio", "mono_txt", "prefix_audio", etc.
                "dataset": data["dataset"],         # from the shard dictionary
                "split": data["split"],             # from the shard dictionary
            }
            tokens, audio_input_mask, audio_preds_mask, txt_preds_mask = getattr(self, sampler)(data, sampler_arg)
            # TODO: make sure that samples are always up to block_size without this hack
            tokens = torch.from_numpy(tokens.astype(np.int64))[:self.sample_len]
            audio_input_mask = torch.from_numpy(audio_input_mask)[:self.sample_len]
            audio_preds_mask = torch.from_numpy(audio_preds_mask)[:self.sample_len]
            txt_preds_mask = torch.from_numpy(txt_preds_mask)[:self.sample_len]
            yield tokens, audio_input_mask, audio_preds_mask, txt_preds_mask

def collate_fn(batch):
    tokens, audio_input_masks, audio_preds_masks, txt_preds_masks = zip(*batch)

    tokens = torch.stack(tokens).permute(0, 2, 1)
    audio_input_masks = torch.stack(audio_input_masks)
    audio_preds_masks = torch.stack(audio_preds_masks)
    txt_preds_masks = torch.stack(txt_preds_masks)
    # define inputs and targets
    input_tokens = tokens[..., :-1]
    target_tokens = tokens[..., 1:]
    audio_input_masks = audio_input_masks[..., :-1]
    audio_preds_masks = audio_preds_masks[..., 1:]
    txt_preds_masks = txt_preds_masks[..., 1:]
    return input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks

def worker_init_fn(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers, dataset):
        dl = DataLoader(
            dataset, 
            batch_size=batch_size,
            pin_memory=False, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn
        )
        for input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks in dl:
            if device == 'cuda':
                # Pin batch, which allows us to move them to GPU asynchronously (non_blocking=True)
                input_tokens = input_tokens.pin_memory().to(device, non_blocking=True)
                target_tokens = target_tokens.pin_memory().to(device, non_blocking=True)
                audio_input_masks = audio_input_masks.pin_memory().to(device, non_blocking=True)
                audio_preds_masks = audio_preds_masks.pin_memory().to(device, non_blocking=True)
                txt_preds_masks = txt_preds_masks.pin_memory().to(device, non_blocking=True)
            else:
                input_tokens = input_tokens.to(device)
                target_tokens = target_tokens.to(device)
                audio_input_masks = audio_input_masks.to(device)
                audio_preds_masks = audio_preds_masks.to(device)
                txt_preds_masks = txt_preds_masks.to(device)

            yield input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks
