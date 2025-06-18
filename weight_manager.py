# weight_manager.py
"""
Dynamically adjusts the multi-objective weights based on performance feedback.
Learns to allocate importance to system sub-scores: cpu, memory, disk, network.
Also adapts mutation strategy probabilities over time.
"""
import numpy as np

class WeightManager:
    def __init__(self, init_weights=None, lr=1e-2, init_mutation_probs=None):
        """
        init_weights: list of 4 floats for [cpu, memory, disk, network]
        lr: learning rate for gradient updates
        init_mutation_probs: dict mapping mutation strategy names to initial probabilities
        """
        # Default equal weighting across four categories
        if init_weights is None:
            init_weights = [1/4] * 4
        if len(init_weights) != 4:
            raise ValueError("init_weights must be a list of 4 floats")
        self.weights = np.array(init_weights, dtype=np.float64)
        self.lr = lr
        self.history = []

        # Mutation strategy probabilities
        if init_mutation_probs is not None:
            # copy to avoid external mutation
            self.mutation_probs = dict(init_mutation_probs)
        else:
            self.mutation_probs = {}
        # track history of mutation_probs over updates
        self.mutation_history = []

    def normalize(self):
        """
        Normalize multi-objective weights to sum to 1.
        """
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total

    def update(self, metrics: np.ndarray, reward_delta: float):
        """
        Update multi-objective weights based on performance feedback.

        metrics: np.array([cpu_s, memory_s, disk_s, network_s])
        reward_delta: float = new_composite - old_composite
        """
        if metrics.shape[0] != 4:
            raise ValueError("metrics array must have length 4")

        grad = metrics * reward_delta
        self.weights += self.lr * grad
        # ensure non-negative
        self.weights = np.clip(self.weights, 0.0, None)
        self.normalize()
        self.history.append(self.weights.copy())

    def get_weights(self) -> np.ndarray:
        """
        Return a copy of the current multi-objective weight vector.
        """
        return self.weights.copy()

    def update_mutation_probs(self, strategy: str, reward_delta: float):
        """
        Update mutation strategy probabilities based on feedback.

        strategy: the mutation strategy key to update
        reward_delta: float = improvement in survival/composite score
        """
        if strategy not in self.mutation_probs:
            # initialize unseen strategies with zero before update
            self.mutation_probs[strategy] = 0.0
        # gradient-like update
        factor = 1.5 if reward_delta < 0 else 1.0
        self.mutation_probs[strategy] = max(
            0.0,
            self.mutation_probs[strategy] + self.lr * reward_delta * factor
        )
        # normalize probabilities
        total = sum(self.mutation_probs.values())
        if total > 0:
            for k in self.mutation_probs:
                self.mutation_probs[k] /= total
        self.mutation_history.append(self.mutation_probs.copy())

    def get_mutation_probs(self) -> dict:
        """
        Return a copy of current mutation strategy probabilities.
        """
        return dict(self.mutation_probs)
