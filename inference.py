import torch
import torch.nn.functional as F
import tiktoken

import logging
import json
from time import time
from tqdm import tqdm
from flash_stu.model import FlashSTU
from flash_stu.config import FlashSTUConfig
from distributed import setup_distributed
from safetensors import safe_open

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device, local_rank, rank, world_size, main_process = setup_distributed(seed=1337)

print(f'Using device: {device}')

# Load the checkpoint
print('Loading the checkpoint...')
start_time = time()
state_dict = {}
with safe_open('/home/ubuntu/flash-stu/fsdp_notensordot/model_06016.safetensors', framework="pt", device='cuda') as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)
print(f'Successfully loaded the checkpoint in {time() - start_time:.2f} seconds')

torch.set_float32_matmul_precision("high")

with open("config.json", "r") as file:
    configs  = json.load(file)

# Model configurations
n_embd             = configs['n_embd']
n_heads            = configs['n_heads']
n_layers           = configs['n_layers']
seq_len            = configs['seq_len']
window_size        = configs['window_size']
vocab_size         = configs['vocab_size']
mlp_scale          = configs['mlp_scale']
bias               = configs['bias']
dropout            = configs['dropout']
num_eigh           = configs['num_eigh']
use_hankel_L       = configs['use_hankel_L']

# Optimizations
use_flash_fft      = configs['use_flash_fft']
use_approx         = configs['use_approx']
torch_compile      = configs['torch_compile']

# Training configurations
dilation           = configs['dilation']
warmup_steps       = configs['warmup_steps']
eval_period        = configs['eval_period']
save_period        = configs['save_period']
num_epochs         = configs['num_epochs']
max_lr             = configs['max_lr']
min_lr             = configs['min_lr']
max_norm           = configs['max_norm']

global_bsz         = configs['global_bsz']
bsz                = configs['bsz']
assert (
    global_bsz % (bsz * seq_len * world_size) == 0
), f"global_bsz ({global_bsz}) must be divisible by bsz * seq_len * world_size ({bsz * seq_len * world_size}),"
f" got {global_bsz % (bsz * seq_len * world_size)}"
gradient_accumulation_steps = global_bsz // (bsz * seq_len * world_size)

# Distributed
fsdp               = configs['fsdp']
ddp                = configs['ddp']
assert not (fsdp and ddp), "FSDP and DDP are both enabled which is not allowed"

distributed        = (fsdp or ddp) and world_size > 1
cache_enabled      = not ddp

mixed_precision    = configs['mixed_precision']
use_cpu_offload    = configs['use_cpu_offload']
sharding_strategy  = configs['sharding_strategy']
auto_wrap_policy   = configs['auto_wrap_policy']
backward_prefetch  = configs['backward_prefetch']
forward_prefetch   = configs['forward_prefetch']
sync_module_states = configs['sync_module_states']
use_orig_params    = configs['use_orig_params']
device_id          = configs['device_id']
precision          = configs['precision']
fsdp_modules       = configs['fsdp_modules']
use_activation_checkpointing = configs['use_activation_checkpointing']

if main_process:
    logging.info(f"Training configs: {configs}\n")

if world_size == 1 and fsdp:
    if main_process:
        logging.info("World size is 1, disabling sharding.")
    sharding_strategy = "no_shard"

configs = FlashSTUConfig(
    n_embd=n_embd,
    n_heads=n_heads,
    n_layers=n_layers,
    seq_len=seq_len,
    window_size=window_size,
    vocab_size=vocab_size,
    mlp_scale=mlp_scale,
    bias=bias,
    dropout=dropout,
    num_eigh=num_eigh,
    use_hankel_L=use_hankel_L,
    use_flash_fft=use_flash_fft,
    use_approx=use_approx,
    device=device,
)
model = FlashSTU(configs)

# Load the saved states into the model
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# Prepare generation
tokenizer = tiktoken.get_encoding('o200k_base')
num_return_sequences = 5
max_length = 16
prompt = "Hi, I'm a language model,"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.int)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

print(
    f"\nGenerating {num_return_sequences} sequences of maximum length {max_length} based on the prompt: '{prompt}'"
)

with torch.no_grad():
    with tqdm(total=max_length - xgen.size(1), desc='Generating') as pbar:
        while xgen.size(1) < max_length:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(xgen)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(
                topk_probs, 1, generator=sample_rng
            )  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            xgen = torch.cat((xgen, xcol), dim=1)
            pbar.update(1)

# Print the generated text
print()
for i in range(num_return_sequences):
    tokens = xgen[i, :].tolist()
    decoded = tokenizer.decode(tokens)
    print(f'Sample {i+1}: {decoded}')
    print()


# import torch
# import torch.nn.functional as F
# import tiktoken

# import logging
# import json
# from safetensors import safe_open
# from time import time
# from tqdm import tqdm
# from transformer import Transformer, ModelConfigs
# from distributed import setup_distributed

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# device, local_rank, rank, world_size, main_process = setup_distributed(seed=1337)

# print(f'Using device: {device}')

# # Load the checkpoint
# print('Loading the checkpoint...')
# start_time = time()
# tensors = {}
# with safe_open('/scratch/gpfs/mn4560/flash-stu/log_transformer/model_20000_large.safetensors', framework="pt", device='cuda') as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
# print(f'Successfully loaded the checkpoint in {time() - start_time:.2f} seconds')

# torch.set_float32_matmul_precision("high")

# with open("config.json", "r") as file:
#     configs = json.load(file)

# # Model configurations
# n_embd = 2304
# n_heads = 9
# n_layers = 6
# seq_len = 8192
# window_size = 4096
# vocab_size = 200064
# mlp_scale = 12
# bias = False
# dropout = 0.01
# num_eigh = 24
# use_hankel_L = False

# # Optimizations
# use_flash_fft = True
# use_approx = True
# torch_compile = False

# # Training configurations
# dilation = 1
# warmup_steps = 715
# eval_period = 250
# save_period = 5000
# num_epochs = 3
# max_lr = 3e-4
# min_lr = 3e-5
# max_norm = 1.0

# global_bsz = 524288
# bsz = 2
# assert (
#     global_bsz % (bsz * seq_len * world_size) == 0
# ), f"global_bsz ({global_bsz}) must be divisible by bsz * seq_len * world_size ({bsz * seq_len * world_size}),"
# f" got {global_bsz % (bsz * seq_len * world_size)}"
# gradient_accumulation_steps = global_bsz // (bsz * seq_len * world_size)

# # Distributed
# fsdp = configs['fsdp']
# ddp = configs['ddp']
# assert not (fsdp and ddp), "FSDP and DDP are both enabled which is not allowed"

# distributed = (fsdp or ddp) and world_size > 1
# cache_enabled = not ddp

# mixed_precision = configs['mixed_precision']
# use_cpu_offload = configs['use_cpu_offload']
# sharding_strategy = configs['sharding_strategy']
# auto_wrap_policy = configs['auto_wrap_policy']
# backward_prefetch = configs['backward_prefetch']
# forward_prefetch = configs['forward_prefetch']
# sync_module_states = configs['sync_module_states']
# use_orig_params = configs['use_orig_params']
# device_id = configs['device_id']
# precision = configs['precision']
# fsdp_modules = configs['fsdp_modules']
# use_activation_checkpointing = configs['use_activation_checkpointing']

# if main_process:
#     logging.info(f"Training configs: {configs}\n")

# if world_size == 1 and fsdp:
#     if main_process:
#         logging.info("World size is 1, disabling sharding.")
#     sharding_strategy = "no_shard"

# config = ModelConfigs(
#     n_embd=n_embd,
#     n_heads=n_heads,
#     n_layers=n_layers,
#     seq_len=seq_len,
#     vocab_size=vocab_size,
#     mlp_scale=mlp_scale,
#     bias=bias,
#     dropout=dropout,
#     device=device,
# )
# model = Transformer(config)

# # Load the saved states into the model
# model.load_state_dict(tensors)
# model.to(device)
# model.eval()

# # Prepare generation
# tokenizer = tiktoken.get_encoding('o200k_base')
# num_return_sequences = 5
# max_length = 16
# prompt = "Twinkle, twinkle, little ,"
# tokens = tokenizer.encode(prompt)
# tokens = torch.tensor(tokens, dtype=torch.int32)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# xgen = tokens.to(device)
# sample_rng = torch.Generator(device=device)
# sample_rng.manual_seed(42)

# print(
#     f"\nGenerating {num_return_sequences} sequences of maximum length {max_length} based on the prompt: '{prompt}'"
# )

# with torch.no_grad():
#     with tqdm(total=max_length - xgen.size(1), desc='Generating') as pbar:
#         while xgen.size(1) < max_length:
#             with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
#                 logits, _ = model(xgen)  # (B, T, vocab_size)
#             logits = logits[:, -1, :]  # (B, vocab_size)
#             probs = F.softmax(logits, dim=-1)
#             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#             ix = torch.multinomial(
#                 topk_probs, 1, generator=sample_rng
#             )  # (B, 1)
#             xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
#             xgen = torch.cat((xgen, xcol), dim=1)
#             pbar.update(1)

# # Print the generated text
# print()
# for i in range(num_return_sequences):
#     tokens = xgen[i, :].tolist()
#     decoded = tokenizer.decode(tokens)
#     print(f'Sample {i+1}: {decoded}')
#     print()