import os
import torch
import tiktoken
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
local_dir = "data/fineweb-edu-10B"

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
    
    logger.info(f"Shard Analysis:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Unique tokens: {unique_tokens}")
    logger.info(f"  Number of EOT tokens: {eot_count}")
    
    logger.info("\nFirst 100 tokens:")
    logger.info(tokens[:100])
    
    decoded_text = decode_tokens(tokens[:500])
    logger.info("\nFirst 500 characters of decoded text:")
    logger.info(decoded_text)

class DistributedDataloader:
    def __init__(
        self,
        bsz: int,
        seq_len: int,
        rank: int,
        world_size: int,
        dataset: str,
        split: str,
        main_process: bool = False,
    ):
        self.bsz = bsz
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        assert split in {'train', 'val', 'test'}, f"Invalid split: {split}"

        self.dataset = dataset
        self.split = split
        
        self.shards = [s for s in os.listdir(dataset) if split in s and s.endswith('.pt')]
        self.shards = [os.path.join(dataset, s) for s in sorted(self.shards)]
        assert len(self.shards) > 0, f'No shards found for split {split}'
        if main_process:
            logger.info(f'Found {len(self.shards)} shards for split {split}')

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = read_shard(self.shards[self.current_shard])
        self.current_position = self.bsz * self.seq_len * self.rank

    def set_epoch(self, epoch):
        self.generator = torch.Generator()
        self.generator.manual_seed(epoch)
        self.shard_order = torch.randperm(len(self.shards), generator=self.generator).tolist()
        self.current_shard = self.shard_order[self.rank % len(self.shards)]
        self.tokens = read_shard(self.shards[self.current_shard])
        self.current_position = self.bsz * self.seq_len * self.rank

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_position + (self.bsz * self.seq_len + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = read_shard(self.shards[self.current_shard])
            self.current_position = self.bsz * self.seq_len * self.rank

        buf = self.tokens[self.current_position : self.current_position + self.bsz * self.seq_len + 1]
        x = buf[:-1].view(self.bsz, self.seq_len)
        y = buf[1:].view(self.bsz, self.seq_len)
        
        self.current_position += self.bsz * self.seq_len * self.world_size
        return x, y.to(torch.long)

def simulate_epoch(dataloader, num_batches):
    shard_usage = defaultdict(int)
    for i in range(num_batches):
        x, y = next(dataloader)
        shard_usage[dataloader.current_shard] += 1
        if i == 0:  # Print sample from the first batch
            logger.info(f"    Sample input (x) from first batch:")
            logger.info(f"      Raw tokens: {x[0, :20]}")
            logger.info(f"      Decoded: {decode_tokens(x[0, :20])}")
            logger.info(f"    Sample target (y) from first batch:")
            logger.info(f"      Raw tokens: {y[0, :20]}")
            logger.info(f"      Decoded: {decode_tokens(y[0, :20])}")
            
            logger.info(f"\n    Extended sample (first 100 tokens):")
            logger.info(f"      Input (x):  {decode_tokens(x[0, :100])}")
            logger.info(f"      Target (y): {decode_tokens(y[0, :100])}")
    return shard_usage

def main():
    # Simulation parameters
    bsz = 4
    seq_len = 128
    world_size = 4
    split = "train"
    num_batches = 100
    num_epochs = 3

    for epoch in range(num_epochs):
        logger.info(f"\nSimulating Epoch {epoch}")
        for rank in range(world_size):
            logger.info(f"  Rank {rank}")
            dataloader = DistributedDataloader(bsz, seq_len, rank, world_size, local_dir, split, rank == 0)
            dataloader.set_epoch(epoch)
            
            shard_usage = simulate_epoch(dataloader, num_batches)
            
            logger.info(f"    Shard usage: {dict(shard_usage)}")
            logger.info(f"    First batch x shape: {next(iter(dataloader))[0].shape}")
            logger.info(f"    First batch y shape: {next(iter(dataloader))[1].shape}")

if __name__ == "__main__":
    main()