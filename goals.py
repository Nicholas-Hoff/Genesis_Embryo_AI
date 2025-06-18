import time
import random
import duckdb 
import math
import json
import heapq
import logging
from typing import Callable, Any, List, Dict, Optional, Tuple, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
from colorama import Fore, Style

logger = logging.getLogger(__name__)
import pickle
from pathlib import Path

def default_q_values(n: int) -> List[float]:
    """
    Helper that returns a fresh list of length n, each initialized to 0.0
    (or any default you prefer).
    """
    return [0.0 for _ in range(n)]

@dataclass(slots=True)
class Goal:
    name: str
    evaluate: Callable[[], float]
    reward: Callable[[], float]
    weight: float = 1.0
    duration: int = 10
    meta: Dict[str, Any] = field(default_factory=dict)
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=100), init=False)
    age: int = 0

    def tick(self) -> bool:
        self.age += 1
        return self.age >= self.duration

    def update(self) -> Tuple[float, float]:
        dist = self.evaluate()
        rew = self.reward()
        self.history.append((time.time(), dist, rew))
        return dist, rew

class RollingStats:
    __slots__ = ('window', 'min_q', 'max_q')

    def __init__(self, maxlen: int):
        self.window = deque(maxlen=maxlen)
        self.min_q = deque()
        self.max_q = deque()

    def push(self, value: float) -> None:
        if len(self.window) == self.window.maxlen:
            old = self.window[0]
            if self.min_q and self.min_q[0] == old:
                self.min_q.popleft()
            if self.max_q and self.max_q[0] == old:
                self.max_q.popleft()
        self.window.append(value)
        while self.min_q and self.min_q[-1] > value:
            self.min_q.pop()
        self.min_q.append(value)
        while self.max_q and self.max_q[-1] < value:
            self.max_q.pop()
        self.max_q.append(value)

    def min(self) -> float:
        return self.min_q[0] if self.min_q else float('inf')

    def max(self) -> float:
        return self.max_q[0] if self.max_q else float('-inf')

    def range(self) -> float:
        return self.max() - self.min()

    def __len__(self) -> int:
        return len(self.window)

    # ─── NEW: Export to JSON ────────────────────────────────────────────
    def to_json(self) -> str:
        """
        Serialize RollingStats as JSON. We store:
          • maxlen (so we can reconstruct)
          • the current contents of 'window'
        """
        data = {
            "maxlen": self.window.maxlen,
            "window": list(self.window)
        }
        return json.dumps(data)

    # ─── NEW: Reconstruct from JSON ─────────────────────────────────────
    @classmethod
    def from_json(cls, json_str: str) -> "RollingStats":
        """
        Re-create a RollingStats instance from its JSON string.
        This will rebuild the window, and 'push' each element back so that
        min_q / max_q get reconstructed properly.
        """
        parsed = json.loads(json_str)
        maxlen = parsed["maxlen"]
        window_values = parsed["window"]
        inst = cls(maxlen)
        for v in window_values:
            inst.push(v)
        return inst

class GoalGenerator:
    """
    Proposes new goals based on embryo metrics, plateau detection, crashes,
    and open-ended procedural tasks.
    """
    def __init__(self, embryo: Any) -> None:
        self.embryo    = embryo
        self.generated = set()
        # late import to avoid circular
        from procedural_tasks import ProceduralTaskGenerator
        self.task_gen  = ProceduralTaskGenerator(embryo)

    def detect_plateau(self, window: int = 5, thresh: float = 1e-3) -> bool:
        hist = getattr(self.embryo, 'score_stats', None)
        if not hist or len(hist) < window:
            return False
        values = list(hist.window)[-window:]
        return (max(values) - min(values)) < thresh

    def propose_emergent_goals(self) -> List[Goal]:
        goals = []
        hist_window = getattr(self.embryo, 'score_stats', None)
        training = getattr(self.embryo.launch_args, "training", False)

        # Recover from sudden drop
        if hist_window and len(hist_window.window) >= 2:
            last, prev = hist_window.window[-1], hist_window.window[-2]
            if last < prev * 0.9:
                goals.append(Goal(
                    name="Recover_from_Drop",
                    evaluate=lambda: 1 - last,
                    reward=lambda: last,
                    weight=2.5,
                    duration=20,
                    meta={'reason': 'sudden_drop'}
                ))

        # Reduce volatility
        if hist_window and len(hist_window.window) >= 5:
            recent = list(hist_window.window)[-5:]
            vol = max(recent) - min(recent)
            if vol > 0.2:
                goals.append(Goal(
                    name="Reduce_Volatility",
                    evaluate=lambda: vol,
                    reward=lambda: 1 - vol,
                    weight=2.0,
                    duration=15,
                    meta={'reason': 'volatility'}
                ))

        # Sustain improvement
        if hist_window and len(hist_window.window) > 10:
            last10 = list(hist_window.window)[-10:]
            if all(x < y for x, y in zip(last10, last10[1:])):
                goals.append(Goal(
                    name="Sustain_Improvement",
                    evaluate=lambda: 0.1,
                    reward=lambda: last10[-1],
                    weight=3.0,
                    duration=30,
                    meta={'reason': 'sustained_improvement'}
                ))

        # Filter crash-prone
        recent = self.embryo.CrashTracker.recent_crashes(limit=10)
        crash_goals = {c['goal'] for c in recent}
        filtered = []
        for g in goals:
            if g.name in self.generated:
                continue
            if not training and g.name in crash_goals:
                logger.warning(f"[FILTER] Skipping crash-prone goal: {g.name}")
                continue
            filtered.append(g)
            self.generated.add(g.name)
        return filtered

    def propose_goals(self) -> List[Goal]:
        # 1) Emergent goals
        goals = self.propose_emergent_goals()
        # 2) Procedural open-ended tasks
        goals += self.task_gen.propose_tasks()

        # 3) Plateau breaker
        if self.detect_plateau():
            name = "Break_Plateau"
            if name not in self.generated:
                last = list(self.embryo.score_stats.window)[-1]
                goals.append(Goal(
                    name=name,
                    evaluate=lambda: 1 - last,
                    reward=lambda: last,
                    weight=2.0,
                    duration=25,
                    meta={'reason': 'plateau'}
                ))
                self.generated.add(name)

        # 4) Boost exploration
        mr = getattr(self.embryo, 'mutation_rate', 0)
        floor = self.embryo.cfg.get('mutation_rate_floor', 0)
        name = "Boost_Exploration"
        if mr < floor * 2 and name not in self.generated:
            goals.append(Goal(
                name=name,
                evaluate=lambda: (floor * 2 - mr),
                reward=lambda: mr,
                weight=1.5,
                duration=20,
                meta={'reason': 'low_explore'}
            ))
            self.generated.add(name)

        return goals


class GoalEngine:
    def __init__(self, goals: List[Any], db_path: str, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):

        self.goals = goals
        self.db_path = db_path
        # Q-learning hyperparameters
        self.alpha = alpha    # learning rate for Q-updates
        self.gamma = gamma    # discount factor
        self.epsilon = epsilon  # exploration probability

        # Ensure the 'goal_q_table' exists
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goal_q_table (
                state_key  VARCHAR,
                goal_index INTEGER,
                q_value    DOUBLE,
                PRIMARY KEY (state_key, goal_index)
            )
        """)
        conn.close()

        # Load existing Q-values or initialize
        conn = duckdb.connect(self.db_path)
        rows = conn.execute("SELECT state_key, goal_index, q_value FROM goal_q_table").fetchall()
        conn.close()

        # Build in-memory Q-table
        self.q_table: Dict[Tuple[Any, ...], List[float]] = defaultdict(lambda: default_q_values(len(self.goals)))
        for state_key_str, goal_index, q_value in rows:
            key_tuple = tuple(state_key_str.split(","))
            self.q_table[key_tuple] = default_q_values(len(self.goals))
            self.q_table[key_tuple][goal_index] = q_value

        # Action selection state
        self.action_space   : List[str]          = []
        self.last_state     : Optional[Tuple[Any, ...]] = None
        self.last_action    : Optional[int]       = None
        self.current_goal   : Optional[str]       = None

    def _state_key(self, state: List[float]) -> Tuple[Any, ...]:
        # Convert numeric state list to a hashable tuple key
        return tuple(round(s, 4) for s in state)

    def choose_action(self, state: List[float], goals: List[Any]) -> str:
        key = self._state_key(state)
        self.action_space = [g.name for g in goals]

        # Ensure Q-row matches current goals length
        if len(self.q_table[key]) != len(goals):
            self.q_table[key] = default_q_values(len(goals))

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            idx, mode = random.randrange(len(goals)), 'explore'
        else:
            q_vals = self.q_table[key]
            idx = int(max(range(len(q_vals)), key=lambda i: q_vals[i]))
            mode = 'exploit'

        self.last_state, self.last_action = key, idx
        self.current_goal = self.action_space[idx]
        logging.debug(f"{Fore.CYAN}[GOAL ENGINE] Selected '{self.current_goal}' via {mode}{Style.RESET_ALL}")
        return self.current_goal

    def reward_goal(self, new_state: List[float], reward: float) -> None:
        key_new = self._state_key(new_state)
        if self.last_state is None or self.last_action is None:
            return

        prev_q = self.q_table[self.last_state][self.last_action]
        future_max = max(self.q_table[key_new]) if self.q_table[key_new] else 0.0
        # Apply crash penalty if needed
        crashes = self.embryo.CrashTracker.recent_crashes_for_goal(
            goal=self.current_goal,
            phase="mutate_cycle",
            limit=1
        )
        if crashes:
            logger.warning(
                f"[CRASH PENALTY] Recent crash tied to '{self.current_goal}' — penalty applied"
            )
            reward -= 0.5

        # Q-learning update
        updated = prev_q + self.alpha * (reward + self.gamma * future_max - prev_q)
        self.q_table[self.last_state][self.last_action] = updated

    def plan_for_goal(self) -> Dict[str, str]:
        return {
            "Recover_from_Drop":   "survival_threshold",
            "Reduce_Volatility":   "heartbeat_interval",
            "Sustain_Improvement": "mutation_rate",
            "Break_Plateau":       "mutation_rate",
            "Boost_Exploration":   "mutation_rate"
        }
