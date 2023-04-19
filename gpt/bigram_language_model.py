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

    def generate(self, idx, max_new_tokens):
        """
        This is the generate function for the model.

        idx is (B, T) array of indices in the current context

        The purpose is to take a (B, T), and turn it into a
        (B, T+1 [+2, +3, ..., +max_new_tokens]) -> This is the generation part.
        """
        for _ in range(max_new_tokens):
            # Get the predictions, but ignore the loss.
            logits, _loss = self(idx)

            # Focus only on the last time step, we pluck out the last element
            # in the time direction (because those are the predictions for what
            # comes next).
            logits = logits[:, -1, :]                           # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)                   # (B, C)

            # Sample from the distribution (just one sample!) a single prediction
            # for what comes next!
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        return idx
