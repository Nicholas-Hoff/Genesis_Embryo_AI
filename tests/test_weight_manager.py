import numpy as np
import pytest
from weight_manager import WeightManager


def test_update_adjusts_weights_and_history():
    wm = WeightManager(init_weights=[0.25, 0.25, 0.25, 0.25], lr=0.1)
    metrics = np.array([1.0, 0.0, 0.0, 0.0])
    wm.update(metrics, reward_delta=1.0)
    weights = wm.get_weights()
    assert pytest.approx(weights.sum(), rel=1e-6) == 1.0
    assert weights[0] > 0.25
    assert len(wm.history) == 1


def test_update_mutation_probs_normalizes_and_clamps():
    wm = WeightManager(lr=0.5)
    wm.update_mutation_probs("gaussian", reward_delta=1.0)
    probs = wm.get_mutation_probs()
    assert pytest.approx(sum(probs.values()), rel=1e-6) == 1.0
    assert probs["gaussian"] > 0.0

    wm.update_mutation_probs("gaussian", reward_delta=-10.0)
    probs = wm.get_mutation_probs()
    assert probs["gaussian"] >= 0.0
    assert probs["gaussian"] < 0.5
