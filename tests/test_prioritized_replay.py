import random
from collections import Counter

from prioritized_replay import PrioritizedReplay


def test_prioritized_sampling_and_length():
    random.seed(42)
    replay = PrioritizedReplay(capacity=10)
    items = ['low', 'medium', 'high']
    priorities = [1.0, 2.0, 5.0]
    for item, pr in zip(items, priorities):
        replay.add(item, pr)

    assert len(replay) == len(items)

    # Sample many times to observe frequency differences
    samples = [replay.sample(1)[0] for _ in range(1000)]
    counts = Counter(samples)

    assert counts['high'] > counts['medium'] > counts['low']
