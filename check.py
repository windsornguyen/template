import os
import torch
import tiktoken
from tqdm import tqdm

# Configuration
local_dir = "data/fineweb-edu-10B"
shard_index = 95  # Change this to read different shards

# Initialize the tokenizer
enc = tiktoken.get_encoding("o200k_base")
eot = enc._special_tokens['<|endoftext|>']

def decode_tokens(tokens):
    return enc.decode(tokens.tolist())

def read_shard(file_path):
    return torch.load(file_path)

def analyze_shard(tokens):
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens.tolist()))
    eot_count = (tokens == eot).sum().item()
    
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {unique_tokens}")
    print(f"Number of EOT tokens: {eot_count}")
    
    # Sample first 100 tokens
    print("\nFirst 100 tokens:")
    print(tokens[:100])
    
    # Decode and print first 500 characters
    decoded_text = decode_tokens(tokens[:500])
    print("\nFirst 500 characters of decoded text:")
    print(decoded_text)

def main():
    # Construct the file path
    split = "val" if shard_index == 0 else "train"
    file_name = f"fineweb-edu_{split}_{shard_index:06d}.pt"
    file_path = os.path.join(local_dir, file_name)
    
    print(f"Reading shard: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Read the shard
    tokens = read_shard(file_path)
    
    # Analyze the shard
    analyze_shard(tokens)

if __name__ == "__main__":
    main()