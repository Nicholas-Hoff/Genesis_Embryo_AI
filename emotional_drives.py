from collections import deque
import math

class EmotionalDrives:
    """Track curiosity and novelty signals."""
    def __init__(self, window: int = 50):
        self.pred_errors = deque(maxlen=window)
        self.obs_history = deque(maxlen=window)

    def record_prediction_error(self, err: float):
        self.pred_errors.append(err)

    def record_observation(self, val: float):
        self.obs_history.append(val)

    @property
    def curiosity_score(self) -> float:
        if not self.pred_errors:
            return 0.0
        mean = sum(self.pred_errors)/len(self.pred_errors)
        var = sum((e-mean)**2 for e in self.pred_errors)/len(self.pred_errors)
        return var

    @property
    def novelty_score(self) -> float:
        if len(self.obs_history) < 2:
            return 0.0
        p = self.obs_history[-2]
        q = self.obs_history[-1]
        if p <= 0 or q <= 0:
            return 0.0
        return q * math.log(q/p)
