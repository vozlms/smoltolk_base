import os
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from model import TextAudioLMFromPretrained, ModelArgs

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# (These globals are used both by this script and can be overridden via configurator.py)
out_dir = "out"
eval_dataset = 'scloze_multimodal'
eval_split = 'test'
audio_feats = 'mhubert25hzl11'
txt_feats = 'smollm'
batch_size = 32              # use a batch size larger than 1 for batched evaluation
device = "cuda"
dtype = "bfloat16"
model_name = "ours_135m_2stg_s200-ricard"
checkpoint_id = "iter17000_final"
num_workers = 16

# (Optional: load overrides from a configurator file)
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
# Uncomment the next line if you use a configurator:
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

# import ptvsd
# ptvsd.enable_attach(('0.0.0.0', 7310))
# print("Attach debugger now")
# ptvsd.wait_for_attach()

# -----------------------------------------------------------------------------
# SET UP DEVICE & AMP CONTEXT
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# LOAD TEXT AND AUDIO DATA, AND MANIFEST
# -----------------------------------------------------------------------------
# Load text data
eval_data_dir_txt = os.path.join('data', eval_dataset, txt_feats)
eval_data_txt = np.memmap(os.path.join(eval_data_dir_txt, f'{eval_split}.bin'), dtype=np.uint16, mode='r')
eval_lens_txt = np.memmap(os.path.join(eval_data_dir_txt, f'{eval_split}.len'), dtype=np.uint16, mode='r')
eval_data_txt = np.split(eval_data_txt, np.cumsum(eval_lens_txt)[:-1])

# Load audio data
eval_data_dir_audio = os.path.join('data', eval_dataset, audio_feats)
eval_data_audio = np.memmap(os.path.join(eval_data_dir_audio, f'{eval_split}.bin'), dtype=np.uint16, mode='r')
eval_lens_audio = np.memmap(os.path.join(eval_data_dir_audio, f'{eval_split}.len'), dtype=np.uint16, mode='r')
eval_data_audio = np.split(eval_data_audio, np.cumsum(eval_lens_audio)[:-1])

# Load manifest CSV
manifest_path = os.path.join('data', eval_dataset, "cross-modal_manifest.csv")
manifest = pd.read_csv(manifest_path)

# -----------------------------------------------------------------------------
# MODEL WRAPPER
# -----------------------------------------------------------------------------
class GetLogProbWrapper(torch.nn.Module):
    def __init__(self, model: TextAudioLMFromPretrained):
        super().__init__()
        self.model = model

    def forward(self,
                input_tokens: torch.Tensor,
                target_tokens: torch.Tensor,
                audio_input_masks: torch.Tensor,
                audio_preds_masks: torch.Tensor,
                txt_preds_masks: torch.Tensor) -> torch.Tensor:
        # sanity check
        bsize, ncodebooks, seqlen = input_tokens.size()
        assert seqlen <= self.model.params.block_size, f"Sequence beyond maximum length of {self.model.params.block_size}"
        assert ncodebooks == self.model.n_codebooks, "Sequence shape must match the specified number of codebooks"
        # determine if multimodal
        is_all_audio = audio_input_masks.all()
        has_audio = audio_input_masks.any()
        logits_audio = None
        logits_txt = None
        if is_all_audio:
            target_audio_tokens = target_tokens
            # apply codebook pattern
            input_audio_tokens = self.model.apply_delay_pattern(input_tokens)
            # compute the frame audio embeddings as the sum of codebook embeddings
            h_raw = sum([self.model.audio_embed[k](input_audio_tokens[:, k]) for k in range(ncodebooks)])
            h = self.model.forward_audio_only(h_raw, audio_input_masks, is_output=False) if self.model.audio_in_layers is not None else h_raw
        elif not has_audio: # is all text
            target_txt_tokens = target_tokens[:, 0, :]
            input_txt_tokens = input_tokens[:, 0, :]
            # compute text embeddings
            h = self.model.txt_embed(input_txt_tokens)
        else: # multimodal input
            input_txt_tokens = input_tokens[:, 0, :]
            target_txt_tokens = target_tokens[:, 0, :]
            input_audio_tokens = input_tokens.clone()
            target_audio_tokens = target_tokens.clone()
            h_raw = self.model.embed_multimodal_input(input_audio_tokens, input_txt_tokens, audio_input_masks)
            h = self.model.forward_audio_only(h_raw, audio_input_masks, is_output=False, multimodal_input=True) if self.model.audio_in_layers is not None else h_raw
        # obtain contextual embeddings
        ctx_out = self.model.context_model(
            inputs_embeds=h,
            use_cache=False,
            output_hidden_states=self.model.params.layer_wa_audio or self.model.params.layer_selwa_audio
        )
        h_ctx = ctx_out['last_hidden_state']
        # compute loss
        audio_loss = torch.zeros(bsize, device=input_tokens.device)
        if audio_preds_masks.any():
            h_audio = h_ctx
            if self.model.params.layer_wa_audio or self.model.params.layer_selwa_audio:
                stacked_ctx_h = torch.stack(ctx_out['hidden_states'])[1:]
                if self.model.params.layer_wa_audio:
                    # compute context as weighted average of contextual representations in all layers
                    h_audio = (stacked_ctx_h * F.softmax(self.model.layer_weights, dim=0)).sum(0)
                elif self.model.params.selwa_in_layer is not None:
                    h_audio = ctx_out['hidden_states'][self.model.params.selwa_in_layer]
                if self.model.params.layer_selwa_audio:
                    # compute context as an input-dependent weighted average of contextual representations in all layers
                    if self.model.params.selwa_downproj:
                        # the input to the weight predictor is the concatenation of downprojected layer representations
                        stacked_ctx_h_downprojed = self.model.selwa_downproj(stacked_ctx_h) # n_layers x B x T x low_proj_dim
                        stacked_ctx_h_downprojed = stacked_ctx_h_downprojed.permute(1, 2, 0, 3).reshape(
                            bsize, seqlen, self.model.n_layers * self.model.params.selwa_downproj) # B x T x (n_layers * low_proj_dim)
                        selected_layer_weights = F.softmax(
                            self.model.layer_selector(stacked_ctx_h_downprojed).permute(2, 0, 1).unsqueeze(-1), dim=0
                        )
                    else:
                        # if layer_wa_audio is set, the input to the weight predictor is the non-input dependent weighted average
                        # if layer_wa_audio is not set, the input to the weight predictor is the last layer context representation
                        selected_layer_weights = F.softmax(self.model.layer_selector(h_audio).permute(2, 0, 1).unsqueeze(-1), dim=0)
                    h_audio = (stacked_ctx_h * selected_layer_weights).sum(0)
            if self.model.params.raw_speech_residual:
                h_audio = h_audio + h_raw
            if self.model.audio_out_layers is not None:
                h_audio = self.model.forward_audio_only(h_audio, audio_input_masks, is_output=True,
                                            multimodal_input=not is_all_audio)
            # apply each audio prediction head to obtain logits per codebook
            logits_audio = torch.stack([self.model.audio_unembed[k](h_audio) for k in range(self.model.n_codebooks)], dim=1).float()
            # Compute per-token loss for audio using reduction='none'
            per_token_audio_loss = F.cross_entropy(
                logits_audio[:, 0].view(-1, logits_audio.size(-1)),
                target_audio_tokens[:, 0].view(-1),
                ignore_index=-1,
                reduction='none'
            )  # shape: (bsize, seqlen)
            per_token_audio_loss = per_token_audio_loss.view(bsize, seqlen)
            # Avoid division by zero (assume at least one valid token per sample)
            per_sample_audio_loss = (per_token_audio_loss * audio_preds_masks).sum(1) / (audio_preds_masks.sum(1) + 1e-8)
            audio_loss = per_sample_audio_loss

        # --- TEXT LOSS ---
        txt_loss = torch.zeros(bsize, device=input_tokens.device)
        if txt_preds_masks.any():
            # For text we assume logits_txt was computed from h_ctx
            logits_txt = self.model.txt_unembed(h_ctx).float()  # shape: (bsize, seqlen, vocab_size)
            # Optionally, forbid predicting special tokens (if needed)
            if self.model.params.bop_token is not None:
                logits_txt[..., self.model.params.bop_token:self.model.params.bop_token + 3] = -float('inf')
            # Compute per-token loss with reduction='none'
            per_token_txt_loss = F.cross_entropy(
                logits_txt.view(-1, logits_txt.size(-1)),
                target_txt_tokens.view(-1),
                ignore_index=-1,
                reduction='none'
            )  # shape: (bsize, seqlen)
            per_token_txt_loss = per_token_txt_loss.view(bsize, seqlen)
            per_sample_txt_loss = (per_token_txt_loss * txt_preds_masks).sum(1) / (txt_preds_masks.sum(1) + 1e-8)
            txt_loss = per_sample_txt_loss

        # --- COMBINE LOSSES ---
        # Here we combine the audio and text losses weighted by the number of tokens used in each.
        txt_token_counts = txt_preds_masks.float().sum(1)
        audio_token_counts = audio_preds_masks.float().sum(1)
        total_token_counts = txt_token_counts + audio_token_counts + 1e-8
        lm_loss = (txt_loss * txt_token_counts + audio_loss * audio_token_counts) / total_token_counts
        # Return the per-sample loss (i.e. NLL) so that later logp = -lm_loss.
        return lm_loss

# -----------------------------------------------------------------------------
# LOAD CHECKPOINT AND INITIALIZE MODEL
# -----------------------------------------------------------------------------
ckpt_path = os.path.join(out_dir, model_name,
                         f"ckpt_{checkpoint_id}.pt" if checkpoint_id != 'ckpt' else "ckpt.pt")
checkpoint = torch.load(ckpt_path)
modelconf = ModelArgs(**checkpoint["model_args"])
model_instance = TextAudioLMFromPretrained(modelconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)
model_instance.load_state_dict(state_dict)
model = GetLogProbWrapper(model_instance)
model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# TOKENIZER (for e.g. swt token)
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")

# -----------------------------------------------------------------------------
# TCLozeDataset (unchanged except for our new parameters)
# -----------------------------------------------------------------------------
class TCLozeDataset(Dataset):
    """
    Produces a list of samples, each sample corresponding to exactly one prompt,
    returning BOTH its correct continuation and an incorrect continuation.
    For each continuation, you get 4 modality-variants (audio_audio, text_text, audio_text, text_audio).
    The returned structure looks like:

        {
          "correct": {
            "audio_audio": {
              "input_tokens": Tensor[int64] of shape [seq_len],
              "target_tokens": Tensor[int64] of shape [seq_len],
              "audio_input_masks": Tensor[bool] of shape [seq_len],
              "audio_preds_masks": Tensor[bool] of shape [seq_len],
              "txt_preds_masks":   Tensor[bool] of shape [seq_len],
              "correctness": int  # 1 for correct
            },
            "text_text": {...},
            "audio_text": {...},
            "text_audio": {...}
          },
          "incorrect": {
            "audio_audio": {...},
            ...
          }
        }
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        eval_data_txt: list[np.ndarray],
        eval_data_audio: list[np.ndarray],
        swt_token: int,
        ignore_index: int = -1,
        do_audio_text_pad: bool = False,
        n_padding: int = 0,
        seed: int = 42
    ):
        """
        Args:
          manifest_df: a DataFrame with columns:
              ["pair_id", "type", "correctness"],
              where type ∈ {"prompt","continuation"},
              correctness ∈ {"-", "correct", "incorrect"}.
          eval_data_txt: list of 1D text token arrays (one per row in manifest_df).
          eval_data_audio: list of 1D audio token arrays (one per row in manifest_df).
          swt_token: integer ID for "switch-to-text" token.
          ignore_index: used to mask out the prompt portion in target tokens.
        """
        super().__init__()
        self.manifest = manifest_df
        self.eval_data_txt = eval_data_txt
        self.eval_data_audio = eval_data_audio
        self.swt_token = swt_token
        self.ignore_index = ignore_index

        # Store new params
        self.do_audio_text_pad = do_audio_text_pad
        self.n_padding = n_padding
        self.rng = np.random.default_rng(seed)

        # Gather indices of all text CONTINUATIONS for random padding
        # (We'll exclude them at use-time if they belong to the current sample.)
        self.all_text_cont_indices = [
            i for i, row in self.manifest.iterrows()
            if row["type"] == "prompt"
        ]

        # --- 1) Identify prompt rows and continuation rows (by correctness) for each pair_id. ---
        prompts_by_pid = {}
        conts_by_pid = {}  # We'll store all continuation row indices, grouped by correctness

        for i, row in self.manifest.iterrows():
            pid = row["pair_id"]
            if row["type"] == "prompt":
                prompts_by_pid[pid] = i
            elif row["type"] == "continuation":
                conts_by_pid.setdefault(pid, []).append(i)

        # --- 2) Build a list of (prompt_idx, correct_idx, incorrect_idx) for each pair_id. ---
        self.samples = []
        for pid, pidx in prompts_by_pid.items():
            if pid not in conts_by_pid:
                continue
            # find exactly one correct
            correct_idxs = [c for c in conts_by_pid[pid] if self.manifest.loc[c, "correctness"] == "correct"]
            # find exactly one incorrect
            incorrect_idxs = [c for c in conts_by_pid[pid] if self.manifest.loc[c, "correctness"] == "incorrect"]
            if len(correct_idxs) > 0 and len(incorrect_idxs) > 0:
                # pick the first correct and the first incorrect
                cidx = correct_idxs[0]
                iidx = incorrect_idxs[0]
                self.samples.append((pid, pidx, cidx, iidx))

    def __len__(self):
        return len(self.samples)

    def _prepend_random_text(self, prompt_part: np.ndarray, cont_part: np.ndarray, exclude_idx: int):
        """Fetch the last n_padding tokens of some *other* text continuation and prepend it to prompt_part."""
        # Pick a random index different from exclude_idx
        while True:
            rand_idx = self.rng.choice(self.all_text_cont_indices)
            if rand_idx != exclude_idx:
                break
        # Grab the text array, including its final eos if present
        random_text_array = self.eval_data_txt[rand_idx]
        # We'll take the last self.n_padding tokens
        pad_snippet = random_text_array[-self.n_padding:]  # shape [n_padding]
        # Prepend to prompt_part
        # NOTE: prompt_part is audio tokens, but you asked to insert text tokens
        # in front of them. If the model can handle “mixed” tokens in the prompt
        # dimension, we just do a direct concat:
        new_prompt_part = np.concatenate([pad_snippet, prompt_part], axis=0)
        return new_prompt_part, cont_part

    def _prepend_random_text_to_prompt_audio(self, prompt_audio_part: np.ndarray, exclude_idx: int):
        """
        Pick a random text continuation (excluding 'exclude_idx'),
        take the last `n_padding` tokens, and prepend them to `prompt_audio_part`.
        Return (new_prompt, text_pad_len).
        """
        if not self.do_audio_text_pad or self.n_padding <= 0:
            return prompt_audio_part, 0
    
        while True:
            rand_idx = self.rng.choice(self.all_text_cont_indices)
            if rand_idx != exclude_idx:
                break
    
        # Grab the random text array
        rand_text_arr = self.eval_data_txt[rand_idx]
        # Slice last n_padding tokens
        pad_snippet = rand_text_arr[-self.n_padding:]  # shape [n_padding]
    
        # Prepend text tokens to the front of the audio prompt
        new_prompt = np.concatenate([pad_snippet, prompt_audio_part], axis=0)
        text_pad_len = len(pad_snippet)  # i.e. self.n_padding
    
        return new_prompt, text_pad_len

    def __getitem__(self, idx: int):
        """
        Returns a dictionary with 'correct' and 'incorrect' keys.
        Each is itself a dict of the 4 sub-variants:
          "audio_audio", "text_text", "audio_text", "text_audio".
        """
        pid, prompt_idx, correct_idx, incorrect_idx = self.samples[idx]

        # Extract the raw arrays (minus final eos, if you like)
        prompt_txt = self.eval_data_txt[prompt_idx][:-1]
        prompt_audio = self.eval_data_audio[prompt_idx][:-1]

        correct_txt = self.eval_data_txt[correct_idx][:-1]
        correct_audio = self.eval_data_audio[correct_idx][:-1]

        incorrect_txt = self.eval_data_txt[incorrect_idx][:-1]
        incorrect_audio = self.eval_data_audio[incorrect_idx][:-1]


        # Helper to mark prompt as ignore_index in target, etc.
        def build_variant(
            prompt_part: np.ndarray,
            cont_part: np.ndarray,
            is_prompt_audio: bool,
            is_cont_audio: bool,
            insert_swt: bool = False,
            text_pad_len: int = 0):
            """
            Build a single variant:
              - Optionally, the first 'text_pad_len' tokens of the prompt are text (audio_input_masks=False).
              - The rest of the prompt is audio if is_prompt_audio=True, or text if =False.
              - ...
            """
            prompt_part = prompt_part.astype(np.int64)
            cont_part   = cont_part.astype(np.int64)
        
            # 1) Concatenate
            if insert_swt:
                input_tokens = np.concatenate([prompt_part, [self.swt_token], cont_part], axis=0)
            else:
                input_tokens = np.concatenate([prompt_part, cont_part], axis=0)
        
            # 2) Shift
            target_tokens = input_tokens[1:].copy()
            input_tokens  = input_tokens[:-1]
        
            seq_len = len(input_tokens)
            audio_input_masks = np.zeros(seq_len, dtype=bool)
            audio_preds_masks = np.zeros(seq_len, dtype=bool)
            txt_preds_masks   = np.zeros(seq_len, dtype=bool)
        
            prompt_len = len(prompt_part)
            swt_len = 1 if insert_swt else 0
            cont_len = len(cont_part)
        
            # A) Mark how many tokens at the front are text due to padding
            #    => Those positions get audio_input_masks=False
            #    => Then any remaining portion of the prompt is set according to is_prompt_audio
            front_text = min(text_pad_len, seq_len)  # clamp in case the prompt is too short
            if front_text > 0:
                audio_input_masks[:front_text] = False
            # The remainder of the prompt (up to prompt_len) is audio if is_prompt_audio=True
            # but note that the total prompt length might exceed seq_len by 1 if short:
            nonpadding_start = front_text
            nonpadding_end   = min(prompt_len, seq_len)
            if nonpadding_start < nonpadding_end:
                audio_input_masks[nonpadding_start:nonpadding_end] = is_prompt_audio
        
            # B) If swt token is in range
            swt_index = prompt_len
            if insert_swt and swt_index < seq_len:
                # swt is text
                audio_input_masks[swt_index] = False
        
            # C) Mark the continuation portion
            cont_start = prompt_len + swt_len
            if cont_start < seq_len:
                if is_cont_audio:
                    audio_input_masks[cont_start:] = True
                else:
                    audio_input_masks[cont_start:] = False
        
                # Predict only continuation
                if is_cont_audio:
                    audio_preds_masks[cont_start - 1:] = True
                else:
                    txt_preds_masks[cont_start - 1:] = True
        
            # D) Hide prompt portion in target
            if (prompt_len - 1) > 0:
                target_tokens[: prompt_len - 1] = self.ignore_index
        
            # If swt inserted, hide it in target as well
            if insert_swt and (prompt_len) < len(target_tokens):
                target_tokens[prompt_len - 1] = self.ignore_index
        
            # Convert to torch Tensors
            out = {
                "input_tokens":        torch.tensor(input_tokens,  dtype=torch.long),
                "target_tokens":       torch.tensor(target_tokens, dtype=torch.long),
                "audio_input_masks":   torch.tensor(audio_input_masks, dtype=torch.bool),
                "audio_preds_masks":   torch.tensor(audio_preds_masks, dtype=torch.bool),
                "txt_preds_masks":     torch.tensor(txt_preds_masks,   dtype=torch.bool)
            }
            return out

        # --------------------------------------------------------
        # Build the correct set of 4 variants
        # --------------------------------------------------------
        # audio_audio
        correct_audio_audio = build_variant(
            prompt_part=prompt_audio,
            cont_part=correct_audio,
            is_prompt_audio=True,
            is_cont_audio=True,
            insert_swt=False,
        )
    
        # text_text
        correct_text_text = build_variant(
            prompt_part=prompt_txt,
            cont_part=correct_txt,
            is_prompt_audio=False,
            is_cont_audio=False,
            insert_swt=False,
        )
    
        # audio_text (ONLY here we do optional left-padding)
        #   We'll modify the 'prompt_audio' by prepending random text if needed
        # For correct "audio_text"
        at_padded_prompt, pad_len = self._prepend_random_text_to_prompt_audio(prompt_audio, exclude_idx=correct_idx)
        correct_audio_text = build_variant(
            prompt_part=at_padded_prompt,
            cont_part=correct_txt,
            is_prompt_audio=True,
            is_cont_audio=False,
            insert_swt=(self.swt_token is not None),
            text_pad_len=pad_len  # <--- pass it here!
        )
    
        # text_audio
        correct_text_audio = build_variant(
            prompt_part=prompt_txt,
            cont_part=correct_audio,
            is_prompt_audio=False,
            is_cont_audio=True,
            insert_swt=(self.swt_token is not None),
        )
    
        correct_variants = {
            "audio_audio": correct_audio_audio,
            "text_text":   correct_text_text,
            "audio_text":  correct_audio_text,
            "text_audio":  correct_text_audio,
        }
    
        # --------------------------------------------------------
        # Build the incorrect set of 4 variants
        # --------------------------------------------------------
        # audio_audio
        incorrect_audio_audio = build_variant(
            prompt_part=prompt_audio,
            cont_part=incorrect_audio,
            is_prompt_audio=True,
            is_cont_audio=True,
            insert_swt=False
        )
    
        # text_text
        incorrect_text_text = build_variant(
            prompt_part=prompt_txt,
            cont_part=incorrect_txt,
            is_prompt_audio=False,
            is_cont_audio=False,
            insert_swt=False
        )
    
        # audio_text (ONLY here we do optional left-padding)
        at_padded_prompt_incorrect, pad_len_incorrect = self._prepend_random_text_to_prompt_audio(prompt_audio, exclude_idx=incorrect_idx)
        incorrect_audio_text = build_variant(
            prompt_part=at_padded_prompt_incorrect,
            cont_part=incorrect_txt,
            is_prompt_audio=True,
            is_cont_audio=False,
            insert_swt=(self.swt_token is not None),
            text_pad_len=pad_len_incorrect  # <--- pass it here!
        )
    
        # text_audio
        incorrect_text_audio = build_variant(
            prompt_part=prompt_txt,
            cont_part=incorrect_audio,
            is_prompt_audio=False,
            is_cont_audio=True,
            insert_swt=(self.swt_token is not None)
        )
    
        incorrect_variants = {
            "audio_audio": incorrect_audio_audio,
            "text_text":   incorrect_text_text,
            "audio_text":  incorrect_audio_text,
            "text_audio":  incorrect_text_audio,
        }
    
        return {
            "correct": correct_variants,
            "incorrect": incorrect_variants
        }

# Create the dataset
dataset = TCLozeDataset(
    manifest_df=manifest,
    eval_data_txt=eval_data_txt,        # list of 1D np.arrays for text tokens
    eval_data_audio=eval_data_audio,      # list of 1D np.arrays for audio tokens
    swt_token=model.model.params.swt_token if model.model.params.swt_token is not None else tokenizer.eos_token_id,
    ignore_index=-1,
    do_audio_text_pad=True,
    n_padding=10
)

# -----------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION FOR BATCHING
# -----------------------------------------------------------------------------
def collate_fn(batch):
    """
    Given a list of samples (each is a dict with keys "correct" and "incorrect",
    each mapping to a dict of 4 variants whose values are dicts of tensors),
    collate the fields across samples. (Since each tensor was unsqueezed to include a “channel” dimension,
    we first squeeze the extra dimension, then pad along the sequence dimension.)
    """
    batched = {"correct": {}, "incorrect": {}}
    variants = ["audio_audio", "text_text", "audio_text", "text_audio"]
    for key in ["correct", "incorrect"]:
        batched[key] = {}
        for var in variants:
            batched[key][var] = {}
            for field in batch[0][key][var].keys():
                field_list = [sample[key][var][field] for sample in batch]
                # Choose the pad value: 0 for input tokens, -1 for target tokens, and False for masks.
                if field == "target_tokens":
                    pad_val = -1
                elif field == "input_tokens":
                    pad_val = 0
                elif field == "audio_input_masks":
                    pad_val = True
                else:
                    pad_val = False
                batched[key][var][field] = pad_sequence(field_list, batch_first=True, padding_value=pad_val)
                if "tokens" in field:
                    batched[key][var][field] = batched[key][var][field].unsqueeze(1)
    return batched

# Create a DataLoader for batched evaluation.
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        shuffle=False, num_workers=num_workers)

# -----------------------------------------------------------------------------
# BATCHED EVALUATION
# -----------------------------------------------------------------------------
# We will compute scores for each variant: a score of 1 if the log-probability
# (i.e. negative loss) of the correct continuation is greater than the incorrect.
variants = ["audio_audio", "text_text", "audio_text", "text_audio"]
scores = {var: [] for var in variants}

pbar = tqdm(total=len(dataset), desc="Evaluating batches")
with torch.no_grad():
    for batch in dataloader:
        for var in variants:
            # Move the batched tensors to device (they are already on device if you did .to(device)
            # in __getitem__, but it is safer to call .to(device) here as well).
            correct_batch = {k: v.to(device) for k, v in batch["correct"][var].items()}
            incorrect_batch = {k: v.to(device) for k, v in batch["incorrect"][var].items()}
            with ctx:
                nll_correct = model(**correct_batch)
                nll_incorrect = model(**incorrect_batch)
            logp_correct = -nll_correct
            logp_incorrect = -nll_incorrect
            batch_scores = (logp_correct > logp_incorrect).int().view(-1).cpu().tolist()
            scores[var].extend(batch_scores)
        # Here we use the batch size from one of the variants (they all have the same number of samples)
        pbar.update(batch["correct"]["audio_text"]["input_tokens"].size(0))
pbar.close()

# Compute the mean score (accuracy) for each variant.
mean_scores = {var: float(np.mean(scores[var])) for var in variants}
print("Mean scores by variant:", mean_scores)

# -----------------------------------------------------------------------------
# SAVE THE SCORES INTO A JSON FILE
# -----------------------------------------------------------------------------
# We will log the scores to a JSON file using the same convention as your other script:
# a main key equal to eval_dataset, and subkeys for each variant.
out_path_eval = os.path.join(out_dir, model_name, "evals", checkpoint_id)
os.makedirs(out_path_eval, exist_ok=True)
scores_json_path = os.path.join(out_path_eval, "scores.json")

def load_or_create_scores_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_scores_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

scores_dict = load_or_create_scores_json(scores_json_path)
scores_dict[eval_dataset] = mean_scores
save_scores_json(scores_json_path, scores_dict)
print(f"Scores saved to {scores_json_path}")