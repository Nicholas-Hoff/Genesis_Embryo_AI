# =============================================================================
# New: prioritized_replay.py
# =============================================================================
import random
from collections import deque

class PrioritizedReplay:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        # capacity = max buffer size, alpha controls how much prioritization (0 = uniform)
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, sample, priority: float):
        # priority should be >0; we’ll store priority^alpha
        self.buffer.append(sample)
        self.priorities.append(priority ** self.alpha)

    def sample(self, k: int):
        if len(self.buffer) == 0:
            return []
        # normalize to a probability distribution
        probs = [p / sum(self.priorities) for p in self.priorities]
        # sample indices with weighted randomness
        idxs = random.choices(range(len(self.buffer)), weights=probs, k=k)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return len(self.buffer)
