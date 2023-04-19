import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    One head of self-attention
    """

    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out
