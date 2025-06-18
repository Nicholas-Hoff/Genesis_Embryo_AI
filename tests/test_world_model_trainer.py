import pytest

torch = pytest.importorskip("torch")

from world_model import WorldModel, WorldModelTrainer


def test_world_model_trainer_runs_single_step():
    model = WorldModel(state_dim=2, choice_emb_dim=1, hidden_dim=4)
    trainer = WorldModelTrainer(model, lr=0.01)

    state = torch.randn(5, 2)
    choice = torch.randn(5, 1)
    delta = torch.randn(5, 2)

    loss = trainer.train_step(state, choice, delta)
    assert isinstance(loss, float)
