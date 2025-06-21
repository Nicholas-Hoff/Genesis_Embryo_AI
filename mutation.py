import random
import copy
import logging
import heapq
import itertools
import json

from typing import Any, Dict, Tuple, List, Callable, Optional, Union

logger = logging.getLogger(__name__)
from colorama import Fore, Style
from strategy import SynthStrategy, StrategyRegistry   # ← add SynthStrategy here
from pygad_strategy import pygad_mutation

# Mutation strategies module

# ─── Constants ────────────────────────────────────────────────────────────────
GENE_PARAMS = [
    "heartbeat_interval",
    "survival_threshold",
    "mutation_rate",
    "mutation_interval",
    "rollback_required",
    "dynamic_strategy_prob",
    "gene_min",
    "gene_max",
]

DEFAULT_META_WEIGHTS: Dict[str, float] = {
    "gaussian":       1.0,
    "random_uniform": 1.0,
    "creep":          1.0,
    "pygad":         1.0,
    "explore":        0.5,
    "reset":          0.0,
}

# Each task_param subkey and how to sample its delta
TASK_PARAM_SAMPLERS: List[Tuple[str, Callable[[Any], float]]] = [
    ('threshold_delta', lambda _: random.gauss(0, 1.0)),
    ('duration',        lambda _: random.choice([-1, 1])),
]

# ─── Module‐level mutation functions ───────────────────────────────────────
def gaussian(embryo: Any) -> Tuple[str, Dict[str, Any]]:
    """Gaussian perturbation on a random parameter."""
    param = random.choice(embryo.mutator.params)
    delta = random.gauss(0, 0.1)
    return embryo_mutation(embryo, param, delta, "gaussian")


def random_uniform(embryo: Any) -> Tuple[str, Dict[str, Any]]:
    """Uniform random step on a random parameter."""
    param = random.choice(embryo.mutator.params)
    delta = random.choice([-1, 1]) * random.uniform(0.05, 0.5)
    return embryo_mutation(embryo, param, delta, "random_uniform")


def creep(embryo: Any) -> Tuple[str, Dict[str, Any]]:
    """Small creep step on a random parameter."""
    param = random.choice(embryo.mutator.params)
    delta = random.uniform(-0.02, 0.02)
    return embryo_mutation(embryo, param, delta, "creep")


def explore(embryo: Any) -> Tuple[str, Dict[str, Any]]:
    """Larger multi-parameter perturbation used when stagnant."""
    picks = random.sample(embryo.mutator.params, k=min(2, len(embryo.mutator.params)))
    ctx = {"strategy": "explore", "param": [], "old": [], "new": [], "delta": []}
    for p in picks:
        old = getattr(embryo, p)
        delta = random.uniform(-0.5, 0.5)
        new = embryo.apply_param_bounds(p, old + delta)
        setattr(embryo, p, new)
        ctx["param"].append(p)
        ctx["old"].append(round(old, 4))
        ctx["new"].append(round(new, 4))
        ctx["delta"].append(round(new - old, 4))
    desc = "explore: " + ", ".join(
        f"{p} {o:.2f}->{n:.2f}" for p, o, n in zip(ctx["param"], ctx["old"], ctx["new"])
    )
    return desc, ctx


def reset(embryo: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Random reset of one parameter—used only when stuck.
    """
    param = random.choice(embryo.mutator.params)
    old = getattr(embryo, param)
    if param == "heartbeat_interval":
        new = random.uniform(0.5, 30.0)
    else:
        new = random.uniform(0.4, 0.95)
    setattr(embryo, param, new)
    desc = f"{param}: {old:.3f} → {new:.3f} (reset)"
    ctx = {"strategy": "reset", "param": param, "old": old, "new": new, "delta": new - old}
    return desc, ctx

# ─── Helper to apply a tweak ────────────────────────────────────────────────
def embryo_mutation(
    embryo: Any,
    param: str,
    delta: float,
    label: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Apply a single‐parameter tweak in place on the embryo,
    enforcing its bounds and returning a description & context.
    If 'param' is empty or invalid, return a no-op rather than crashing.
    """
    # ─── Guard against empty/invalid 'param' ────────────────────────────────────
    if not isinstance(param, str) or param == "":
        # no valid parameter name: treat as a no-op
        return "", {}
    old = getattr(embryo, param)
    new = embryo.apply_param_bounds(param, old + delta)
    setattr(embryo, param, new)
    desc = f"{param}: {old:.3f} → {new:.3f} ({label})"
    ctx = {
        "strategy": label,
        "param": param,
        "old": round(old, 4),
        "new": round(new, 4),
        "delta": round(new - old, 4),
    }
    return desc, ctx


def tweak_task_param(
    embryo: Any,
    key: str,
    subkey: str,
    delta: float
) -> Tuple[str, Dict[str, Any]]:
    """
    Mutate embryo.task_params[key][subkey] by delta, respecting lower bound of 0.
    """
    old = embryo.task_params[key][subkey]
    new = max(0.0, old + delta)
    embryo.task_params[key][subkey] = new
    desc = f"task_params[{key}][{subkey}]: {old:.3f} → {new:.3f}"
    ctx = {
        "strategy": "tweak_task_param",
        "param": f"{key}.{subkey}",
        "old": round(old, 4),
        "new": round(new, 4),
        "delta": round(new - old, 4)
    }
    return desc, ctx

# ─── The main mutation‐cycle logic ─────────────────────────────────────────
def mutation_cycle(
    embryo: Any,
    meta_weights: Dict[str, float],
    stagnant_cycles: int,
    *,
    strategy_idx: Optional[int] = None,
    stuck_threshold: int = 5,
    max_reset_weight: float = 0.2,
    reset_penalty: float = 0.0,
    alpha: float = 0.1,
    return_strategy: bool = False
) -> Union[Tuple[float, int], Tuple[float, int, str, Dict[str, Any]]]:
    """
    Single mutation step applied directly on the Embryo object.
    If strategy_idx is provided, uses that action index from embryo.action_space
    instead of stochastic pick_strategy.
    Returns: (new_score, new_stagnant_cycles) or
             (new_score, new_stagnant_cycles, strategy_name, ctx)
             if return_strategy=True. The returned `ctx` is the mutation
             context dictionary recorded in the database.
    """
    from health import SystemHealth, Survival

    # 1) Snapshot & measure before
    pre_state      = copy.deepcopy(embryo)
    before_metrics = SystemHealth.check()
    before_score   = Survival.score(before_metrics)["composite"]
    is_stuck       = stagnant_cycles >= stuck_threshold
    logger.warning(f"[MUTATION CYCLE] is_stuck={is_stuck} (stagnant={stagnant_cycles})")

    # 2) Pick and apply strategy (planner‐driven if strategy_idx given)
    if strategy_idx is not None:
        choice = embryo.action_space[strategy_idx]
        strat  = embryo.mutator.get_strategy(choice)
    else:
        choice, strat = embryo.mutator.pick_strategy(meta_weights, is_stuck)
    desc, ctx = strat.apply(embryo)

    # 3) Log and measure after
    raw_after    = SystemHealth.check()
    scored_after = Survival.score(raw_after)
    new_score    = scored_after["composite"]

    # Record mutation context
    params = ctx.get("param")
    olds   = ctx.get("old", 0.0)
    news   = ctx.get("new", 0.0)
    if isinstance(params, list) or isinstance(olds, list) or isinstance(news, list):
        if not isinstance(params, list):
            params = [params]
        if not isinstance(olds, list):
            olds = [olds] * len(params)
        if not isinstance(news, list):
            news = [news] * len(params)
        for p, o, n in zip(params, olds, news):
            embryo.db.record_mutation_context(
                param=p,
                strategy=choice,
                old=o,
                new=n,
                before=before_score,
                after=new_score,
                cpu=scored_after["cpu"],
                mem=scored_after["memory"],
                disk=scored_after["disk"],
                network=scored_after.get("network", 0.0)
            )
    else:
        embryo.db.record_mutation_context(
            param=params,
            strategy=choice,
            old=olds,
            new=news,
            before=before_score,
            after=new_score,
            cpu=scored_after["cpu"],
            mem=scored_after["memory"],
            disk=scored_after["disk"],
            network=scored_after.get("network", 0.0)
        )

    logger.info(f"[SCORE] before={before_score:.4f}, after={new_score:.4f}")

    # 4) Compute reward & update weights
    reward = new_score - before_score
    if choice == "reset" and not is_stuck:
        reward = -abs(reset_penalty)
    factor = 1.5 if reward < 0 else 1.0
    meta_weights[choice] = max(0.0, meta_weights.get(choice, 0.0) + alpha * reward * factor)
    meta_weights["reset"] = min(meta_weights.get("reset", 0.0), max_reset_weight)
    total = sum(meta_weights.values()) or 1.0
    for name in meta_weights:
        meta_weights[name] /= total
    logger.info(f"[META] normalized weights: {meta_weights}")

    # restore previous state on negative outcome
    if new_score < before_score:
        embryo.__dict__.clear()
        embryo.__dict__.update(pre_state.__dict__)

    # 5) Update stagnant counter
    new_stagnant = 0 if new_score > before_score else stagnant_cycles + 1

    if return_strategy:
        return new_score, new_stagnant, choice, ctx
    return new_score, new_stagnant

# ─── Archive of top‐k states ───────────────────────────────────────────────
# mutation.py

class Archive:
    """
    Keeps a min-heap of top-k states by score, with a tie-breaker counter.
    """
    def __init__(self, k: int):
        self.k = k
        self._heap: List[Tuple[float,int,Any]] = []
        self._counter = itertools.count()

    def consider(self, state: Any, score: float) -> None:
        cnt = next(self._counter)
        entry = (score, cnt, copy.deepcopy(state))
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        else:
            heapq.heappushpop(self._heap, entry)
        logger.debug(f"[ARCHIVE] considered score={score}")

    def seed(self, embryo: Any) -> None:
        if not self._heap:
            return
        if random.random() < 0.2:
            score, _, state = random.choice(self._heap)
            embryo.__dict__.update(state)
            logger.info(f"[ARCHIVE] seeded from champion score={score}")

    def replay_success(self, embryo: Any) -> None:
        if not self._heap:
            return
        best_score, _, best_state = max(self._heap, key=lambda x: x[0])
        embryo.__dict__.update(best_state)
        logger.info(f"[REPLAY] restored best state score={best_score}")

    # ─── NEW: Export to JSON ────────────────────────────────────────────
    def to_json(self) -> str:
        """
        Serialize the Archive as a JSON list of (score, state_dict).
        We drop the tie-counter.  Each 'state' must itself be JSON-serializable.
        """
        serializable_heap = []
        for score, _, state in self._heap:
            # here `state` is a dict (thanks to how Embryo.__getstate__ packages it)
            # but if you need further conversion, do it here.
            serializable_heap.append({
                "score": score,
                "state": state
            })
        return json.dumps(serializable_heap)

    # ─── NEW: Rebuild an Archive from JSON ─────────────────────────────
    @classmethod
    def from_json(cls, json_str: str) -> "Archive":
        """
        Create an Archive of size k by deserializing the JSON we stored.
        Note: this will not restore the original k exactly; we default to len(parsed).
        """
        data = json.loads(json_str)  # data is a list of {"score":..., "state":...}
        arch = cls(k=len(data))
        for entry in data:
            score = entry["score"]
            state_dict = entry["state"]
            # We create a dummy "count" for tie-breaking; it doesn’t matter
            cnt = next(arch._counter)
            arch._heap.append((score, cnt, state_dict))
        heapq.heapify(arch._heap)
        return arch


# ─── Stateful engine that holds strategies & weights ───────────────────────
from strategy import StrategyRegistry

class MutationEngine:
    def __init__(self, sample_state: Any, cfg: Any, registry: StrategyRegistry):
        # Receive the shared StrategyRegistry via DI
        self.registry = registry

        # Keep a reference to the embryo and its config
        self.sample = sample_state
        self.cfg    = cfg

        # Initialize your strategy weights & parameter lists
        self.params  = GENE_PARAMS.copy()
        self.params += [
            "cpu_threshold", "mem_threshold", "disk_threshold",
            "cpu_ok_streak", "mem_ok_streak", "disk_ok_streak",
            "cpu_bonus_val", "mem_bonus_val", "disk_bonus_val",
        ]
        self.weights = DEFAULT_META_WEIGHTS.copy()

    # ─── NEW: Export this MutationEngine to JSON ────────────────────────────
    def to_json(self) -> str:
        """
        Serialize just the 'weights' dictionary (and any other minimal state)
        so that later we can reconstruct the MutationEngine with exactly the
        same learned weights.  The registry and sample are reattached by Embryo's __setstate__.
        """
        data = {
            'weights': self.weights,
            # If you have other fields to persist, add them here.
        }
        return json.dumps(data)

    # ─── NEW: Rebuild a MutationEngine from JSON ────────────────────────────
    @classmethod
    def from_json(cls, json_str: str) -> "MutationEngine":
        """
        Create a new MutationEngine with default sample and registry placeholders.
        The Embryo.__setstate__ method should reassign 'sample' and 'registry'
        on the returned object, then restore 'weights'.
        """
        data = json.loads(json_str)
        # Create with dummy placeholders; Embryo.__setstate__ will override sample and registry.
        engine = cls(sample_state=None, cfg=None, registry=StrategyRegistry())
        engine.weights = data.get('weights', {}).copy()
        return engine

    def _register_default_strategies(self) -> None:
        from mutation import DEFAULT_STRATEGIES_MAP
        for name, fn in DEFAULT_STRATEGIES_MAP.items():
            self.registry.register(SynthStrategy(name, fn))

    def make_tweak_fn(self, scenario, subkey, sampler):
        def tweak_fn(em):
            return tweak_task_param(em, scenario, subkey, sampler(scenario))
        return tweak_fn

    def _register_task_param_strategies(self) -> None:
        for scenario, params in (self.sample or {}).get('task_params', {}).items():
            for subkey, sampler in TASK_PARAM_SAMPLERS:
                name = f"tweak_{scenario}_{subkey}"
                thunk = self.make_tweak_fn(scenario, subkey, sampler)
                self.registry.register(SynthStrategy(name, thunk))

    def pick_strategy(
        self,
        meta_weights: Dict[str, float],
        is_stuck: bool
    ) -> Tuple[str, SynthStrategy]:
        names = ["gaussian", "random_uniform", "creep", "pygad"]
        if is_stuck:
            names.append("explore")
            names.append("reset")
        wts   = [meta_weights.get(n, 0.0) for n in names]
        total = sum(wts) or 1.0
        probs = [w/total for w in wts]
        choice = random.choices(names, weights=probs, k=1)[0]
        logging.debug(f"[STRATEGY_ENGINE] chosen '{choice}' weights={probs}")
        logger.debug(f"[STRATEGY_ENGINE] choosing {choice} weights={probs}")
        return choice, self.registry.get(choice)

    def register(
        self,
        name: str,
        fn: Callable[..., Tuple[str, Dict[str, Any]]],
        weight: float = 0.05
    ) -> None:
        strat = SynthStrategy(name, fn)
        self.registry.register(strat)
        self.weights[name] = weight
        logging.debug(f"[REGISTER] {name} weight={weight}")
        logger.debug(f"[REGISTER] {name} weight={weight}")

    def get_strategy(self, name: str) -> Optional[SynthStrategy]:
        return self.registry.get(name)

    def get_all_strategies(self) -> List[SynthStrategy]:
        return self.registry.get_all()

# ─── The one canonical map of names → module‐level functions ────────────────
DEFAULT_STRATEGIES_MAP: Dict[str, Callable[..., Tuple[str, dict]]] = {
    "gaussian":       gaussian,
    "random_uniform": random_uniform,
    "creep":          creep,
    "explore":        explore,
    "pygad":          pygad_mutation,
    "reset":          reset,
}

__all__ = [
    "gaussian", "random_uniform", "creep", "explore", "pygad_mutation", "reset",
    "embryo_mutation", "mutation_cycle", "Archive", "MutationEngine",
    "DEFAULT_STRATEGIES_MAP",
]
