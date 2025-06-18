# procedural_tasks.py
"""
Open-ended environment & intrinsic-drive tasks generator.
Produces dynamically created mini-tasks (stress tests) as Goals.
"""

import time
import random
from typing import Any, List
from goals import Goal
from health import SystemHealth, Survival
from genesis_embryo_core import Embryo
from colorama import Fore, Style

class ProceduralTaskGenerator:
    """
    Generates short-lived, procedurally defined tasks (Goals) to expand the embryo's objective space.
    """
    def __init__(self, embryo: Embryo, task_interval: int = 50):
        self.embryo = embryo
        self.counter = 0
        self.interval = task_interval

    def propose_tasks(self) -> List[Goal]:
        """
        Every `interval` heartbeats, propose a new random stress-test goal.
        Allows dynamic threshold deltas and durations via embryo.task_params.
        """
        self.counter += 1
        tasks: List[Goal] = []
        # Retrieve task-specific parameters that embryo can evolve
        params = getattr(self.embryo, 'task_params', {})

        if self.counter % self.interval == 0:
            metrics = self.embryo.metrics
            # Build available scenarios based on metrics present
            available = ['cpu_burst', 'memory_spike', 'io_stress']
            if 'network' in metrics:
                available.append('network_spike')


            scenario = random.choice(available)
            # Fetch dynamic threshold delta and duration (embryo can mutate these)
            delta = params.get(scenario, {}).get(
                'threshold_delta',
                10.0 if 'spike' in scenario else 0.1
            )
            dur = params.get(scenario, {}).get('duration', 10)

            if scenario == 'cpu_burst':
                name = f"CPUBurst_{self.embryo.hb.count}"
                base = metrics.get('cpu', 0.0)
                threshold = min(100.0, base + delta)
                def eval_fn(): return max(0.0, (base - threshold) / 100)
                def reward_fn(): return max(0.0, (threshold - base) / 100)
                tasks.append(Goal(
                    name, eval_fn, reward_fn,
                    weight=2.0, duration=dur,
                    meta={'scenario': scenario, 'threshold': threshold}
                ))

            elif scenario == 'memory_spike':
                name = f"MemSpike_{self.embryo.hb.count}"
                base = metrics.get('memory', 0.0)
                threshold = min(100.0, base + delta)
                def eval_fn(): return max(0.0, (base - threshold) / 100)
                def reward_fn(): return max(0.0, (threshold - base) / 100)
                tasks.append(Goal(
                    name, eval_fn, reward_fn,
                    weight=2.0, duration=dur,
                    meta={'scenario': scenario, 'threshold': threshold}
                ))

            elif scenario == 'io_stress':
                name = f"IOStress_{self.embryo.hb.count}"
                dio = metrics.get('disk_io')
                interval = self.embryo.cfg.get('metrics_interval', 1.0)
                io_rate = (
                    (dio.read_bytes + dio.write_bytes) / (1024 * 1024 * interval)
                    if dio else 0.0
                )
                def eval_fn(): return 1.0 - min(1.0, io_rate / (io_rate + 1e-6))
                def reward_fn(): return min(1.0, io_rate / (io_rate + 1e-6))
                tasks.append(Goal(
                    name, eval_fn, reward_fn,
                    weight=1.5, duration=dur,
                    meta={'scenario': scenario, 'rate_mb_s': io_rate}
                ))

            elif scenario == 'network_spike':
                name = f"NetSpike_{self.embryo.hb.count}"
                base = metrics.get('network', 0.0)
                threshold = min(1.0, base + delta)
                def eval_fn(): return max(0.0, base - threshold)
                def reward_fn(): return max(0.0, threshold - base)
                tasks.append(Goal(
                    name, eval_fn, reward_fn,
                    weight=2.0, duration=dur,
                    meta={'scenario': scenario, 'threshold': threshold}
                ))

        return tasks
