import logging
import torch
import torch.nn as nn
from typing import List

logger = logging.getLogger(__name__)

class EpisodeSummarizer:
    """Simple Transformer-based encoder that produces an embedding for an episode."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_heads: int = 4, num_layers: int = 2):
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        logger.debug("EpisodeSummarizer initialized input_dim=%s", input_dim)

    def summarize(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return a mean pooled representation of a sequence.
        Args:
            sequence: Tensor of shape (seq_len, batch, input_dim)
        """
        with torch.no_grad():
            enc = self.encoder(sequence.transpose(0, 1))
            out = enc.mean(dim=1)
            logger.debug("Episode summarized to shape %s", out.shape)
            return out
