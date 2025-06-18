import logging
from typing import Callable, Dict, List, Any, Optional
from colorama import Fore, Style

logger = logging.getLogger(__name__)

# ─── SynthStrategy ──────────────────────────────────────────────────────────

class SynthStrategy:
    """
    Wraps a mutation function so we can track errors/successes
    and auto-prune any that misfire too often.
    """
    MAX_ERRORS = 2

    def __init__(self, name: str, fn: Callable[..., tuple[str, dict]]):
        self.name = name
        self.fn = fn
        self.func = fn        # preserve original function reference
        self.error_count = 0
        self.success_count = 0

    def apply(self, embryo: Any) -> tuple[str, dict]:
        try:
            desc, ctx = self.fn(embryo)
            ctx["strategy"] = self.name
        except Exception as e:
            self.error_count += 1
            logging.warning(f"[STRATEGY ERROR] {self.name}: {e}")
            logger.warning(f"[STRATEGY ERROR] {self.name}: {e}")
            raise
        else:
            self.success_count += 1
            logger.info(f"[STRATEGY APPLIED] {self.name} -> {desc}")
            return desc, ctx

    @property
    def is_dead(self) -> bool:
        return self.error_count >= self.MAX_ERRORS


# ─── Strategy Registry ──────────────────────────────────────────────────────

class StrategyRegistry:
    """
    Manages a collection of ``SynthStrategy`` objects and prunes dead ones.

    Registration events are logged at ``DEBUG`` level.
    """
    # Names never auto-pruned; we now also keep tweak_ strategies alive
    DEFAULT_STRATEGIES = {"gaussian", "random_uniform", "creep", "reset"}

    def __init__(self, sample_state: Any):
        # Keep a copy of the sample state for resets
        self.sample = sample_state
        self._strategies: Dict[str, SynthStrategy] = {}

    def register(self, strategy: SynthStrategy) -> None:
        # never register the same strategy twice
        if strategy.name in self._strategies:
            return
        self._strategies[strategy.name] = strategy
        logging.debug(f"[REGISTER] strategy '{strategy.name}'")
        # print(f"{Fore.CYAN}[REGISTER] strategy {strategy.name}{Style.RESET_ALL}")

    def get(self, name: str) -> Optional[SynthStrategy]:
        """Retrieve a single registered strategy by name."""
        return self._strategies.get(name)

    def get_all(self) -> List[SynthStrategy]:
        return list(self._strategies.values())

    def prune_dead(self) -> None:
        """
        Remove any strategy that has exceeded error limit,
        except default and tweak_ strategies.
        """
        for name, strat in list(self._strategies.items()):
            if strat.is_dead and (name not in self.DEFAULT_STRATEGIES) and not name.startswith("tweak_"):
                logging.info(f"[PRUNE] strategy '{name}' died (errors={strat.error_count})")
                logger.info(f"[PRUNE] strategy {name}")
                del self._strategies[name]

    def remove(self, name: str) -> None:
        """Remove a strategy explicitly by name."""
        self._strategies.pop(name, None)

    def register_defaults(self, task_params: Dict[str, Any]) -> None:
        """
        Register both core strategies and all tweak_* strategies.
        """
        # Import here to avoid circular dependency
        from mutation import DEFAULT_STRATEGIES_MAP, TASK_PARAM_SAMPLERS, tweak_task_param

        # 1) Core strategies
        for name, fn in DEFAULT_STRATEGIES_MAP.items():
            self.register(SynthStrategy(name, fn))

        # 2) Tweak-parameter strategies
        for scenario, params in task_params.items():
            for subkey, sampler in TASK_PARAM_SAMPLERS:
                strat_name = f"tweak_{scenario}_{subkey}"
                def thunk(em, s=scenario, sk=subkey, samp=sampler):
                    return tweak_task_param(em, s, sk, samp(s))
                self.register(SynthStrategy(strat_name, thunk))


# ─── MetaMutator ─────────────────────────────────────────────────────────────

class MetaMutator:
    @staticmethod
    def adapt(
        weights: Dict[str, float],
        history: Dict[str, int],
        synth_stats: Dict[str, List[float]],
        min_floor: float = 0.05
    ) -> None:
        """
        1) Reweight by usage history
        2) Prune under-performing synthetic strategies
        3) Renormalize all weights
        """
        total_usage = sum(history.values()) or 1.0
        for strat in list(weights):
            usage = history.get(strat, 1)
            weights[strat] = max(min_floor, usage / total_usage)

        # Prune unhelpful synths
        to_remove = [s for s, deltas in synth_stats.items()
                     if (sum(deltas) / len(deltas)) <= 0]
        for s in to_remove:
            synth_stats.pop(s, None)
            history.pop(s, None)
            weights.pop(s, None)
            logging.info(f"[META PRUNE] dropped strategy '{s}'")
            logger.info(f"[META PRUNE] strategy {s}")

        # Renormalize
        total_w = sum(weights.values()) or 1.0
        for strat in weights:
            weights[strat] /= total_w
        logging.info(f"[META] weights normalized: {weights}")
        logger.info(f"[META] weights normalized: {weights}")

    @staticmethod
    def optimize_hyperparameters(
        cfg: Any,
        mutation_history: Dict[str, int]
    ) -> None:
        """
        Auto-tune key hyperparameters based on overall success rate.
        """
        successes = mutation_history.get("success", 0)
        attempts = max(mutation_history.get("attempts", 0), 1)
        success_rate = successes / attempts

        delta = 0.05 * (0.5 - success_rate)
        new_init = max(0.1, min(0.5, cfg.get("mutation_rate_initial") + delta))
        new_floor = max(0.01, min(0.2, cfg.get("mutation_rate_floor") + delta * 0.2))
        cfg.set("mutation_rate_initial", new_init)
        cfg.set("mutation_rate_floor", new_floor)

        if success_rate > 0.75:
            cfg.set("mutation_rate_anneal", min(cfg.get("mutation_rate_anneal") + 0.005, 0.999))
        else:
            cfg.set("mutation_rate_anneal", max(cfg.get("mutation_rate_anneal") - 0.005, 0.900))

        curr_rb = cfg.get("rollback_required_cycles")
        if success_rate < 0.3:
            cfg.set("rollback_required_cycles", min(curr_rb + 1, 10))
        elif success_rate > 0.8:
            cfg.set("rollback_required_cycles", max(curr_rb - 1, 1))

        msg = (
            f"[HYPER] success_rate={success_rate:.2f} -> init={new_init:.3f}, "
            f"floor={new_floor:.3f}, anneal={cfg.get('mutation_rate_anneal'):.3f}, "
            f"rollback_cycles={cfg.get('rollback_required_cycles')}"
        )
        logging.info(msg)
        logger.info(msg)
