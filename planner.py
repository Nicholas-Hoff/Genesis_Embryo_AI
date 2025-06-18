import random
from typing import Any, List

class MCTSPlanner:
    """Very small Monte Carlo tree search planner using a world model."""
    def __init__(self, world_model, rollout_depth: int = 3, samples: int = 10):
        self.world_model = world_model
        self.rollout_depth = rollout_depth
        self.samples = samples

    def plan(self, state: Any, actions: List[Any]) -> Any:
        """Return the best action by random rollout using the world model."""
        best_a, best_r = None, float('-inf')
        for a in actions:
            total = 0.0
            for _ in range(self.samples):
                s = state
                r = 0.0
                for _ in range(self.rollout_depth):
                    inp = self.world_model.prepare_input(s, a)
                    delta = self.world_model(inp)
                    s = self.world_model.post_process(s, delta)
                    r += self.world_model.estimate_reward(s)
                total += r / self.rollout_depth
            if total > best_r:
                best_r = total
                best_a = a
        return best_a
