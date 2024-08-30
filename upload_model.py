import torch
from huggingface_hub import upload_file

# def push_to_hf_repo(local_file_path, repo_name, file_name):
#     # Load the original checkpoint
#     checkpoint = torch.load(local_file_path)

#     # Create a new dictionary with only 'config' and 'optimizer'
#     reduced_checkpoint = {
#         "config": checkpoint["config"],
#         "optimizer": checkpoint["optimizer"]
#     }

#     # Save the reduced checkpoint to a temporary file
#     temp_file_path = "temp_reduced_checkpoint.pt"
#     torch.save(reduced_checkpoint, temp_file_path)

#     # Upload the temporary file
#     upload_file(
#         path_or_fileobj=temp_file_path,
#         path_in_repo=file_name,
#         repo_id=repo_name,
#         repo_type="model",
#         commit_message="Uploaded config and optimizer"
#     )
#     print(f"Uploaded reduced checkpoint to Hugging Face repo {repo_name} as {file_name}")

#     # Optional: Remove the temporary file
#     import os
#     os.remove(temp_file_path)

# # Example usage
# local_file_path = "other_checkpoints_20000_large.pt"
# repo_name = "windsornguyen/aladdin-2b"  # Your specific repository
# file_name = "other_checkpoints_20000.pt"  # The name in the Hugging Face repository

# push_to_hf_repo(local_file_path, repo_name, file_name)

import json
from huggingface_hub import notebook_login

notebook_login()
from flash_stu.config import FlashSTUConfig
from flash_stu.model import FlashSTU
from transformer import Transformer, TransformerConfig
from safetensors import safe_open


FlashSTUConfig.register_for_auto_class()
FlashSTU.register_for_auto_class("AutoModel")

# with open("config.json", "r") as file:
#     config = json.load(file)
with open("transformer_small.json", "r") as file:
    config = json.load(file)

# Model configurations
# n_embd = config["n_embd"]
# n_heads = config["n_heads"]
# n_layers = config["n_layers"]
# seq_len = config["seq_len"]
# window_size = config["window_size"]
# vocab_size = config["vocab_size"]
# mlp_scale = config["mlp_scale"]
# bias = config["bias"]
# dropout = config["dropout"]
# num_eigh = config["num_eigh"]
# use_hankel_L = config["use_hankel_L"]

# # Optimizations
# use_flash_fft = config["use_flash_fft"]
# use_approx = config["use_approx"]
# softcap = config["softcap"]
# torch_compile = config["torch_compile"]

# # Training configurations
# dilation = config["dilation"]
# warmup_steps = config["warmup_steps"] // dilation
# eval_period = config["eval_period"] // dilation
# save_period = config["save_period"] // dilation
# num_epochs = config["num_epochs"]
# max_lr = config["max_lr"]
# min_lr = config["min_lr"]
# max_norm = config["max_norm"]

# global_bsz = config["global_bsz"]
# bsz = config["bsz"]


# Distributed
fsdp = config["fsdp"]
ddp = config["ddp"]
assert not (fsdp and ddp), "FSDP and DDP are both enabled which is not allowed"

cache_enabled = not ddp

mixed_precision = config["mixed_precision"]
torch_dtype     = config["torch_dtype"]
use_cpu_offload = config["use_cpu_offload"]
sharding_strategy = config["sharding_strategy"]
auto_wrap_policy = config["auto_wrap_policy"]
backward_prefetch = config["backward_prefetch"]
forward_prefetch = config["forward_prefetch"]
sync_module_states = config["sync_module_states"]
use_orig_params = config["use_orig_params"]
device_id = config["device_id"]
precision = config["precision"]
fsdp_modules = config["fsdp_modules"]
use_activation_checkpointing = config["use_activation_checkpointing"]

# config = FlashSTUConfig(
#     bsz=bsz,
#     n_embd=n_embd,
#     n_heads=n_heads,
#     n_layers=n_layers,
#     seq_len=seq_len,
#     window_size=window_size,
#     vocab_size=vocab_size,
#     mlp_scale=mlp_scale,
#     bias=bias,
#     dropout=dropout,
#     num_eigh=num_eigh,
#     use_hankel_L=use_hankel_L,
#     use_flash_fft=use_flash_fft,
#     use_approx=use_approx,
#     softcap=softcap,
#     torch_dtype=getattr(torch, torch_dtype),
# )
# model = FlashSTU(config)

config = TransformerConfig(torch_dtype=getattr(torch, torch_dtype))
model = Transformer(config)

device = torch.device("cuda")

# Load the state dict from safetensors
# with safe_open(
#     "/home/ubuntu/flash-stu/log_tensordot/model_04096.safetensors",
#     framework="pt",
# ) as f:
#     state_dict = {k: f.get_tensor(k) for k in f.keys()}

# # Load the state dict into the model
# model.load_state_dict(state_dict, strict=False)

# Push to Hugging Face Hub
model.push_to_hub(
    "windsornguyen/flash-stu-test",
    safe_serialization=True,
)
