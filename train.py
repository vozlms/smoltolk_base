"""
Code largely adapted from https://github.com/karpathy/llama2.c

LICENSE: MIT License

Copyright (c) 2023 Andrej

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
import math
import os
import time
from contextlib import nullcontext
import numpy as np
import random
from functools import partial

import torch
from model import TextAudioLMFromPretrained, ModelArgs
from dataset import DatasetArgs, TextAudioCodecLMDataset, Task
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

torch._dynamo.config.optimize_ddp=False
# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
n_evals = 50
log_interval = 10
eval_interval = 200
eval_iters = 1
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
keep_last_n_checkpoints = 1  # keep only the last n checkpoints
resume_from = "last"  # or "best" or "warmdown"
force_load_last = False
ckpt_warmdown_at = []
# wandb logging
wandb_log = True  # disabled by default
wandb_project = 'uniaudiogen'
wandb_run_name = 'run' + str(time.time())
wandb_entity = None
wandb_offline = True
# data
audio_datasets = ['ls960', 'librilight-large', 'people', 'stinystories', 'swiki', 'tedlium', 'voxpopuli']
txt_datasets = ['finewebedu', 'cosmopedia2', 'pythonedu', 'finemath']
audiotxt_datasets = ['libriheavy', 'stinystories']
train_audio_datasets_probs = []
train_txt_datasets_probs = []
train_audiotxt_datasets_probs = []
val_audio_datasets_probs = []
val_txt_datasets_probs = []
val_audiotxt_datasets_probs = []
train_splits = ['train', 'small', 'medium']
val_splits = ['dev-clean', 'dev-other', 'dev', 'val']
audio_tokens = 'mhubert25hzl11'
txt_tokens = 'gptneox'
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
num_workers = 0
block_size = 2048
p_strategies = [1., 1., 0., 0., 1.] # weight of each sampling strategy
pred_txt_in_interleaved = True
force_multimodal_model = False
# model
backbone = "EleutherAI/pythia-410m-deduped"
warm_init = True
freeze_backbone = False
freeze_txt_inout = False
tie_audio_embeddings = True
n_audio_in_layers = 0
n_audio_out_layers = 0
rope_theta = -1
layer_wa_audio = False
layer_selwa_audio = False
selwa_in_layer = None
selwa_linear = True
selwa_downproj = 0
selwa_downproj_linear = True
raw_speech_residual = False
entropy_reg = 0.
#   codebook args
fine_warmup_iters = 0 # by default don't use warmup for fine tokens' lm loss
max_fine_weight = 1.
n_codebooks_to_use = -1 # by default use all codebooks
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
audio_learning_rate = None # learning rate used for audio-specific layers
min_lr = 5e-5
min_audio_lr = None 
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
load_optimizer = True
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
decay_audio_lr = None
schedule = 'halfcos' # halfcos or trapezoidal (https://arxiv.org/pdf/2405.18392)
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
warmup_iters = 1000  # how many steps to warm up for
warmdown_iters = 0 # how many steps to warm down before the end of training
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster
use_idr_torch = True  # use jean-zay
seed_offset = 0
debug = False
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and ((isinstance(v, (int, float, bool, str, list)) or v is None))
]
exec(open("configurator.py").read())  # overrides from command line or config file

# set some defaults
if audio_learning_rate is None:
    audio_learning_rate = learning_rate
if min_audio_lr is None:
    min_audio_lr = min_lr
if decay_audio_lr is None:
    decay_audio_lr = decay_lr

if use_idr_torch:
    import idr_torch

if debug:
    import ptvsd
    ptvsd.enable_attach(('0.0.0.0', 7310))
    print("Attach debugger now")
    ptvsd.wait_for_attach()
    wandb_log = False
    out_dir = f"out/debug_{str(time.time())}"
    compile = False
    num_workers = 0
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Various inits, derived attributes, I/O setup
ddp = idr_torch.size > 1 if use_idr_torch else int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    ddp_rank = idr_torch.rank if use_idr_torch else int(os.environ["RANK"])
    ddp_local_rank = idr_torch.local_rank if use_idr_torch else int(os.environ["LOCAL_RANK"])
    ddp_world_size = idr_torch.size if use_idr_torch else int(os.environ["WORLD_SIZE"])
    init_process_group(backend="nccl", world_size=ddp_world_size, rank=ddp_rank)
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = seed_offset + ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    ddp_world_size = 1

# logging
if wandb_log and master_process:
    import wandb
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, entity=wandb_entity)

# Misc stuff
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"saving to: {out_dir}")
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {block_size} block size")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# Note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

is_multitask = sum(1 for x in p_strategies if x != 0) > 1

# Init these up here, can override if init_from='resume' (i.e., from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Create the datasets 
dataset_args = dict(
    audio_datasets=audio_datasets,
    txt_datasets=txt_datasets,
    audiotxt_datasets=audiotxt_datasets,
    p_strategies=p_strategies,
    audio_tokens=audio_tokens,
    txt_tokens=txt_tokens,
    n_codebooks_to_use=n_codebooks_to_use,
    block_size=block_size,
    pred_txt_in_interleaved=pred_txt_in_interleaved
)
train_dataset_args = {**dataset_args,
                      'splits': train_splits,
                      'audio_datasets_probs': train_audio_datasets_probs,
                      'txt_datasets_probs': train_txt_datasets_probs,
                      'audiotxt_datasets_probs': train_audiotxt_datasets_probs}
train_data_conf = DatasetArgs(**train_dataset_args)
train_dataset = TextAudioCodecLMDataset(train_data_conf)
val_dataset_args = {**dataset_args,
                    'splits': val_splits,
                    'audio_datasets_probs': val_audio_datasets_probs,
                    'txt_datasets_probs': val_txt_datasets_probs,
                    'audiotxt_datasets_probs': val_audiotxt_datasets_probs}
val_data_conf = DatasetArgs(**val_dataset_args)
val_dataset = TextAudioCodecLMDataset(val_data_conf)

# Model configuration
# if force_multimodal, create a dummy multimodal dataset:
model_ref_dataset = train_dataset
if force_multimodal_model:
    print("-" * 80)
    print(f"Dummy dataset because of {force_multimodal_model=}")
    dummy_dataset_args = {**dataset_args, 'splits': ['train']}
    dummy_dataset_args["p_strategies"] = [1., 1., 0., 0., 0.]  # force interleaving -> multimodal
    dummy_data_conf = DatasetArgs(**dummy_dataset_args)
    dummy_dataset = TextAudioCodecLMDataset(dummy_data_conf)
    model_ref_dataset = dummy_dataset
    print("-" * 80)

model_args = dict(
    audio_vocabsize=model_ref_dataset.audio_vocabsize,
    txt_vocabsize=model_ref_dataset.txt_vocabsize,
    block_size=block_size,
    audio_pad_token=model_ref_dataset.audio_pad_token,
    bop_token=model_ref_dataset.bop_token,
    eop_token=model_ref_dataset.eop_token,
    swt_token=model_ref_dataset.swt_token,
    n_codebooks=model_ref_dataset.audio_ncodebooks,
    tie_audio_embeddings=tie_audio_embeddings,
    backbone=backbone,
    warm_init=warm_init,
    freeze_backbone=freeze_backbone,
    freeze_txt_inout=freeze_txt_inout,
    n_audio_in_layers=n_audio_in_layers,
    n_audio_out_layers=n_audio_out_layers,
    rope_theta=rope_theta,
    layer_wa_audio=layer_wa_audio,
    layer_selwa_audio=layer_selwa_audio,
    selwa_in_layer=selwa_in_layer,
    raw_speech_residual=raw_speech_residual,
    selwa_linear=selwa_linear,
    selwa_downproj=selwa_downproj,
    selwa_downproj_linear=selwa_downproj_linear,
    entropy_reg=entropy_reg
)

def get_latest_checkpoint(out_dir, tag_pattern):
    """
    Finds the latest checkpoint matching a specific tag pattern.
    """
    files = os.listdir(out_dir)
    checkpoints = []
    for file in files:
        match = re.match(rf"ckpt_iter(\d+){tag_pattern}\.pt", file)
        if match:
            iter_num = int(match.group(1))
            checkpoints.append((iter_num, file))
    if checkpoints:
        # Sort by iteration number and return the latest
        return max(checkpoints, key=lambda x: x[0])[1]
    return None

def resume_checkpoint(out_dir, resume_from="last"):
    """
    Resumes training from the specified checkpoint type.
    Returns the loaded `checkpoint` (or None if not found) 
    and the checkpoint file path used.
    """
    checkpoint = None
    ckpt_path = None

    if resume_from == "best":
        ckpt_path = os.path.join(out_dir, "ckpt_best.pt")
    elif resume_from == "last":
        latest_file = get_latest_checkpoint(out_dir, '')
        if latest_file:
            ckpt_path = os.path.join(out_dir, latest_file)
    elif resume_from == "warmdown":
        latest_file = get_latest_checkpoint(out_dir, "_warmdown")
        if latest_file:
            ckpt_path = os.path.join(out_dir, latest_file)

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        # if we loaded the latest checkpoint, check that it is not a post-warmdown start checkpoint
        # if we are loading a post warmdown start checkpoint, load instead the latest warmdown start checkpoint,
        # unless force_load_last is set (useful e.g. to resume from stage 1 training)
        if resume_from == "last" and not force_load_last:
            checkpoint_cfg = checkpoint["config"]
            is_trapezoidal_ckpt = checkpoint_cfg["schedule"] == "trapezoidal" \
                and (checkpoint_cfg["decay_lr"] or checkpoint_cfg["decay_audio_lr"])
            ckpt_warmdown_start = checkpoint_cfg["lr_decay_iters"] - checkpoint_cfg["warmdown_iters"]
            is_warmdown_ckpt = is_trapezoidal_ckpt and (checkpoint["iter_num"] > ckpt_warmdown_start)
            if is_warmdown_ckpt:
                print("Post-warmdown start checkpoint detected, loading the latest warmdown start checkpoint instead ...")
                ckpt_path = get_latest_checkpoint(out_dir, "_warmdown")
                if ckpt_path:
                    ckpt_path = os.path.join(out_dir, ckpt_path)
                    checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"Resuming training from {ckpt_path}")
    return checkpoint, ckpt_path

def override_args_from_checkpoint(current_args, ckpt_args, fields_to_override):
    for field in fields_to_override:
        if field in ckpt_args and field in current_args:
            old_val = current_args[field]
            new_val = ckpt_args[field]
            if old_val != new_val:
                print(f"Overriding from checkpoint '{field}': {old_val} -> {new_val}")
                current_args[field] = new_val
    return current_args

checkpoint, ckpt_path = resume_checkpoint(out_dir, resume_from)
if checkpoint is not None:
    # If we have a checkpoint, override the model args from it that are needed to match so that resuming is possible
    checkpoint_model_args = checkpoint["model_args"]
    fields_to_override = ["audio_vocabsize",
        "txt_vocabsize",
        "backbone",
        "n_audio_in_layers",
        "n_audio_out_layers",
        "layer_wa_audio",
        "layer_selwa_audio",
        "selwa_linear",
        "selwa_downproj",
        "selwa_downproj_linear",
    ]
    model_args = override_args_from_checkpoint(
        model_args, checkpoint_model_args, fields_to_override
    )

modelconf = ModelArgs(**model_args)
model = TextAudioLMFromPretrained(modelconf)

if checkpoint is not None:
    state_dict = checkpoint["model"]
    # Fix any unwanted prefix in the state dictionary
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            new_key = k[len(unwanted_prefix):]
            state_dict[new_key] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # Also restore iteration number and best val loss if relevant
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    torch.manual_seed(1337 + seed_offset + iter_num)

print(model)
model.to(device)

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    device=device,
    num_workers=num_workers
)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, audio_learning_rate)
if checkpoint is not None and load_optimizer and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)
    print("model compiled!")

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

def get_fine_weight(it):
    curr_fine_weight = max_fine_weight
    if it < fine_warmup_iters:
        curr_fine_weight = max_fine_weight * it / fine_warmup_iters
    return curr_fine_weight

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    def compute_losses(batch_iter, eval_iters, dataset=None):
        if batch_iter is None:
            assert dataset is not None, "You must provide either a batch iterator or a dataset"
            batch_iter = iter_batches(dataset=dataset)
        total_losses = torch.zeros(eval_iters)
        losses = {key: 0.0 for key in raw_model.losses.keys()}
        for k in range(eval_iters):
            input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks = next(batch_iter)
            with ctx:
                raw_model(input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks)
            total_losses[k] = raw_model.last_loss
            for key in raw_model.losses.keys():
                losses[key] += raw_model.losses[key]

        mean_total_loss = total_losses.mean().item()
        mean_losses = {key: losses[key] / eval_iters for key in losses}
        return mean_total_loss, mean_losses
    raw_model.eval()
    # estimate training and validation losses
    train_total_loss, train_losses = compute_losses(eval_train_batch_iter, eval_iters)
    val_total_loss, val_losses = compute_losses(eval_val_batch_iter, eval_iters)
    total_out = {"train": train_total_loss, "val": val_total_loss}
    losses_out = {"train": train_losses, "val": val_losses}
    task_losses = {}
    if is_multitask:
        # estimate the validation loss per training task
        orig_sampler_probs = val_dataset.sampler_probs # save the original sampler probabilities
        for task_ix, (sampler_cls, task_args, task_name) in enumerate(val_dataset.samplers):
            # Force the sampler to only produce samples for the current task
            val_dataset.sampler_probs = [1. if i == task_ix else 0. for i in range(len(val_dataset.samplers))]
            # here we create a fresh batch iterator as otherwise sampler_probs would have to be changed per worker
            task_loss, _ = compute_losses(None, eval_iters, val_dataset)
            task_losses[task_name] = task_loss
        # reset validation sampler probabilities to the original ones
        val_dataset.sampler_probs = orig_sampler_probs
    raw_model.train()
    return total_out, losses_out, task_losses

# learning rate decay scheduler (cosine with warmup, inverse sqrt, trapezoidal)
def get_lr(it, lr, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, apply the selected schedule
    if schedule == 'halfcos':
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (lr - min_lr)
    elif schedule == 'trapezoidal':
        if it < (lr_decay_iters - warmdown_iters):
            return lr
        else: # start warmdown to min learning rate
            decay_ratio = (lr_decay_iters - it) / warmdown_iters
            return lr * decay_ratio
    elif schedule == 'invsqrt':
        # inverse sqrt decay schedule
        decay_factor = lr * math.sqrt(warmup_iters)
        return max(decay_factor / math.sqrt(it), min_lr)
    else:
        raise NotImplementedError(f"Unknown {schedule} schedule, options: 'halfcos', 'trapezoidal', or 'invsqrt'")

def save_checkpoint(tag=''):
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config
    }
    out_path = os.path.join(out_dir, f"ckpt{tag}.pt")
    print(f"saving checkpoint to {out_path}")
    torch.save(checkpoint, out_path)

# evaluation iterators
eval_train_batch_iter = iter_batches(dataset=train_dataset)
eval_val_batch_iter = iter_batches(dataset=val_dataset)
# training loop
train_batch_iter = iter_batches(dataset=train_dataset)
input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = None
lr, audio_lr = None, None
while True:
    # determine and set the learning rate for this iteration
    for param_group in optimizer.param_groups:
        if param_group['is_audio']:
            audio_lr = get_lr(iter_num, audio_learning_rate, min_audio_lr) if decay_audio_lr else audio_learning_rate
            param_group['lr'] = audio_lr
        else:
            lr = get_lr(iter_num, learning_rate, min_lr) if decay_lr else learning_rate
            param_group['lr'] = lr
    # determine the weight of the fine tokens lm loss for this iteration
    fine_weight = get_fine_weight(iter_num) if fine_warmup_iters else 1.

    # evaluate the loss on train/val sets and write checkpoints
    is_start_of_warmdown = (schedule == 'trapezoidal' and (iter_num == (lr_decay_iters - warmdown_iters) or iter_num in ckpt_warmdown_at))
    is_eval_step = iter_num % eval_interval == 0
    if (iter_num % eval_interval == 0 or is_start_of_warmdown or iter_num == max_iters) and master_process:
        prefix = "[start of warmdown] " if is_start_of_warmdown else ''
        total_losses, loss_terms, task_losses = estimate_loss()
        log_str = f"{prefix}step {iter_num} | "
        for split in ['train', 'val']:
            split_str = f"{split} total {total_losses[split]:.4f} "
            for term in loss_terms[split]:
                split_str += f"{term} : {loss_terms[split][term]} "
            log_str += split_str + ('| ' if task_losses else '')
        task_str = ''
        for task in task_losses:
            task_str += f"{task} : {task_losses[task]} "
        log_str += task_str
        print(log_str)
        if wandb_log and is_eval_step:
            log_dict = {
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,
                "losses/train": total_losses["train"],
                "losses/val": total_losses["val"],
                "lr": lr if lr is not None else audio_lr,
                "fine_weight": fine_weight,
                **{f"losses/train_{key}": loss_terms['train'][key] for key in loss_terms['train']},
                **{f"losses/val_{key}": loss_terms['val'][key] for key in loss_terms['val']}
            }

            if running_mfu is not None:
                log_dict["mfu"] = running_mfu * 100

            if task_losses:
                log_dict.update({
                    **{f"losses/val_{key}": task_losses[key] for key in task_losses}
                })
            if lr is not None and audio_lr is not None and lr != audio_lr:
                log_dict["audio_lr"] = audio_lr
            wandb.log(log_dict, step = iter_num)
        new_best_val_loss = False
        if total_losses["val"] < best_val_loss:
            best_val_loss = total_losses["val"]
            new_best_val_loss = True
        if not debug and iter_num > 0:
            if new_best_val_loss:
                save_checkpoint("_best")
            if always_save_checkpoint:
                save_checkpoint(f"_iter{iter_num}")
                if keep_last_n_checkpoints is not None:
                    # remove previous checkpoint
                    pattern = r"^ckpt_iter(\d+)\.pt$"
                    iteration_ckpt_files = [
                        f for f in os.listdir(out_dir)
                        if re.match(pattern, f)
                    ]
                    iteration_ckpt_files.sort(
                        key=lambda filename: int(re.match(pattern, filename).group(1))
                    )
                    for old_ckpt in iteration_ckpt_files[:-keep_last_n_checkpoints]:
                        os.remove(os.path.join(out_dir, old_ckpt))
            if is_start_of_warmdown:
                save_checkpoint(f"_iter{iter_num}_warmdown")

            if iter_num == max_iters:
                save_checkpoint(f"_iter{iter_num}_final")

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            model(input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks, fine_weight)
            loss = raw_model.last_loss / gradient_accumulation_steps     
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        input_tokens, target_tokens, audio_input_masks, audio_preds_masks, txt_preds_masks = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu is None else 0.9 * running_mfu + 0.1 * mfu
        
        msg = ''
        if lr is not None and audio_lr is not None and lr != audio_lr:
            msg += f"{iter_num} | loss {lossf:.4f} | lr txt/audio {lr:e}/{audio_lr:e} | {dt*1000:.2f}ms"
        else:
            msg += f"{iter_num} | loss {lossf:.4f} | lr {lr if lr is not None else audio_lr:e} | {dt*1000:.2f}ms"

        if running_mfu is not None:
            msg += f" | mfu {running_mfu:.2%}"
        
        print(msg)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    torch.cuda.empty_cache()
    torch.distributed.barrier()  # Ensure all processes synchronize
    destroy_process_group()
