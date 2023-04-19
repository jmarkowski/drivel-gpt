import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import Head
from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class BigramLanguageModel(nn.Module):

    def __init__(self, block_size, vocab_size, n_embed, device='cpu'):
        super().__init__()

        # Each token directly reads off the logits for the next token from a
        # lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # Self-attention head
        self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embed//4, n_embed=n_embed, block_size=block_size)

        # Feed forward network
        self.feed_fwd = FeedForward(n_embed)

        # Language-modeling head
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.device = device
        self.block_size = block_size

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        k = torch.arange(T, device=self.device) # Integers from 0 to T-1

        # logits are "scores" for the next character in the sequence
        tok_emb = self.token_embedding_table(idx) # (B, T, C=n_embed)
        pos_emb = self.position_embedding_table(k) # (T, C)

        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # apply one head of self-attention. (B, T, C)
        x = self.feed_fwd(x) # (B, T, C)

        logits = self.lm_head(x) # (B, T, C=vocab_size)
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
            # Crop the idx that is fed to forward block
            idx_cropped = idx[:, -self.block_size:]

            # Get the predictions, but ignore the loss.
            logits, _loss = self(idx_cropped)

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
