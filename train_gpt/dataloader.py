"""
DataLoader modules for GPT training.
"""
import os
import numpy as np
import torch


def load_tokens(filename):
    """Load tokens from a .npy file and return as a torch tensor."""
    arr = np.load(filename)
    arr = arr.astype(np.int32)
    return torch.tensor(arr, dtype=torch.long)


class DataLoaderLite:
    """
    DataLoader for loading from sharded .npy files.
    Efficiently handles large datasets split across multiple files.
    """
    def __init__(self, B, T, split, proc_idx, num_procs, data_root="./train_gpt/dataset"):
        self.B = B
        self.T = T
        assert split in {"train", "val"}
        self.proc_idx = proc_idx
        self.num_procs = num_procs
        # load only .npy shards for this split
        shards = [
            os.path.join(data_root, s)
            for s in sorted(os.listdir(data_root))
            if split in s and s.endswith(".npy")
        ]

        assert len(shards) > 0, f"No shards found for split: {split}"

        self.shards = shards
        self.reset()

    def reset(self):
        """Reset to the first shard and position."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.proc_idx

    def _advance_shard(self):
        """Load next shard and reset position."""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.proc_idx

    def skip_batches(self, n):
        """
        Skip n batches without loading the data.
        Useful for resuming training from a specific batch.
        
        Args:
            n: Number of batches to skip
        """
        B, T = self.B, self.T
        tokens_to_skip = n * B * T * self.num_procs
        
        while tokens_to_skip > 0:
            remaining_in_shard = len(self.tokens) - self.current_position
            
            if tokens_to_skip < remaining_in_shard:
                # Can skip within current shard
                self.current_position += tokens_to_skip
                tokens_to_skip = 0
            else:
                # Need to advance to next shard
                tokens_to_skip -= remaining_in_shard
                self._advance_shard()

    def next_batch(self):
        """
        Get the next batch of data.
        
        Returns:
            tuple: (x, y) where x is input tokens and y is target tokens
        """
        B, T = self.B, self.T
        needed = B * T + 1  # need B*T+1 tokens for this process's batch

        # If not enough tokens left in this shard → move to next.
        if self.current_position + needed > len(self.tokens):
            self._advance_shard()

        # Now guaranteed enough tokens
        buf = self.tokens[self.current_position : self.current_position + needed]

        # Create x,y. buf length is exactly B*T+1 so reshaping is safe.
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_procs
        return x, y

