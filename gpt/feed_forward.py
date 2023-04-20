import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, n_embed, dropout_amount):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(n_embed, 4 * n_embed),
                nn.ReLU(),
                nn.Linear(4 * n_embed, n_embed), # Projection layer
                nn.Dropout(dropout_amount)
            )

    def forward(self, x):
        return self.net(x)
