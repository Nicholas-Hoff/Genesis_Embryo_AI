import torch
import torch.nn as nn
from typing import List

class EpisodeSummarizer:
    """Simple Transformer-based encoder that produces an embedding for an episode."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_heads: int = 4, num_layers: int = 2):
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def summarize(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return a mean pooled representation of a sequence.
        Args:
            sequence: Tensor of shape (seq_len, batch, input_dim)
        """
        with torch.no_grad():
            enc = self.encoder(sequence)
            return enc.mean(dim=0)
