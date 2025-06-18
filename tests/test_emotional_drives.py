import math
import pytest
from emotional_drives import EmotionalDrives


def test_curiosity_score_is_variance():
    ed = EmotionalDrives()
    errors = [1.0, 2.0, 4.0]
    for e in errors:
        ed.record_prediction_error(e)
    mean = sum(errors) / len(errors)
    expected_var = sum((e - mean) ** 2 for e in errors) / len(errors)
    assert ed.curiosity_score == pytest.approx(expected_var)


def test_novelty_score_matches_kl_term():
    ed = EmotionalDrives()
    ed.record_observation(0.5)
    ed.record_observation(1.0)
    expected = 1.0 * math.log(1.0 / 0.5)
    assert ed.novelty_score == pytest.approx(expected)
