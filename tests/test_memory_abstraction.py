import pytest

torch = pytest.importorskip("torch")

from memory_abstraction import EpisodeSummarizer


def test_episode_summarizer_output_shape():
    seq_len, batch, dim = 4, 2, 8
    summarizer = EpisodeSummarizer(input_dim=dim)
    x = torch.randn(seq_len, batch, dim)
    out = summarizer.summarize(x)
    assert out.shape == (batch, dim)
    assert not out.requires_grad
