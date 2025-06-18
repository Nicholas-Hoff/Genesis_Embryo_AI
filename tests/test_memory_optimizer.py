import pytest

from memory_optimizer import EfficientStateManager, TimeSimEngine, MemoryAIManager


def test_efficient_state_manager_restores_with_recompute():
    mgr = EfficientStateManager(checkpoint_interval=2)
    state = {'val': 0}
    mgr.save_state(state.copy())  # index 0 checkpoint
    state['val'] += 1
    mgr.save_state(state.copy(), lambda s: {'val': s['val'] + 1})  # index 1 rule
    state['val'] += 1
    mgr.save_state(state.copy())  # index 2 checkpoint
    state['val'] += 1
    mgr.save_state(state.copy(), lambda s: {'val': s['val'] + 1})  # index 3 rule

    assert mgr.load_state(1) == {'val': 1}
    assert mgr.load_state(2) == {'val': 2}
    assert mgr.load_state(3) == {'val': 3}


def test_timesimengine_descending_score_order():
    engine = TimeSimEngine()
    engine.enqueue(1, 0.5)
    engine.enqueue(2, 0.8)
    engine.enqueue(3, 0.3)

    results = [engine.get_next() for _ in range(3)]
    scores = [-r[0] for r in results]
    indices = [r[1] for r in results]

    assert scores == sorted(scores, reverse=True)
    assert indices == [2, 1, 3]


def test_memory_ai_manager_usage_and_summarize():
    manager = MemoryAIManager()
    manager.summarizer = None  # Ensure summarizer unavailable

    manager.track_usage('a')
    manager.track_usage('a')
    manager.track_usage('b')

    assert manager.usage_stats['a'] == 2
    assert manager.usage_stats['b'] == 1

    assert manager.should_retain('a', 1)
    assert not manager.should_retain('b', 2)

    assert manager.summarize_episode(None) is None
