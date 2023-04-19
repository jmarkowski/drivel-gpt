import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Each token directly reads off the logits for the next token from a
        # lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers

        # logits are "scores" for the next character in the sequence
        logits = self.token_embedding_table(idx)
        # The logits form a (Batch, Time, Channel) tensor
        # In our example, that would be  (4, 8, 65=vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
