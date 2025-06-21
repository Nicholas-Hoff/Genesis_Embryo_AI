'''GENESIS EMBRYO CORE v5.2.1 — Phase 3.3'''
# Purpose: Free the intelligence — adapt mutation strategies (including self-generated), archive champions, dynamically adjust gene_count, track mutation rate, log metrics, and snapshots with enhanced console feedback.
import os
import sys
import time
import copy
import json
import math
import uuid
import shutil
import pickle
import psutil
import tempfile
import random
import heapq
import logging, traceback
from pathlib import Path
from threading import Thread
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Literal
import tracemalloc
from pydantic import BaseModel, Field, ValidationError
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from colorama import init as colorama_init, Fore, Style

from health import SystemHealth, Survival, Heartbeat
from health_monitor import HealthMonitor
from resources import spawn_resource_controllers, DynamicResourceManager
from mutation import Archive, MutationEngine, mutation_cycle, DEFAULT_STRATEGIES_MAP, tweak_task_param, TASK_PARAM_SAMPLERS
from strategy import StrategyRegistry, SynthStrategy
from goals import Goal, RollingStats, GoalGenerator, GoalEngine, default_q_values
from crash_tracker import CrashTracker
from world_model import WorldModel, WorldModelTrainer
from weight_manager import WeightManager
from prioritized_replay import PrioritizedReplay
from memory_optimizer import EfficientStateManager, TimeSimEngine, MemoryAIManager
from persistence import MemoryArchive, MemoryDB, SnapshotManager, DuckdbStateIO
from meta_strategy_engine import MetaStrategyEngine
from logging_config import configure_logging
from emotional_drives import EmotionalDrives
from env_scanner import scan_environment
import settings

from multiprocessing import current_process

colorama_init(autoreset=True)

BEHAVIOR_KEYS = ('cpu','memory','disk','network')
BEHAVIOR_DIM  = len(BEHAVIOR_KEYS)

if "--training" not in sys.argv:
    tracemalloc.start(10)

def _curiosity_default():
    """Top‐level factory so that pickle can find it."""
    return {"succ": 0, "att": 0}

def parse_args():
    p = argparse.ArgumentParser(
        prog="genesis_embryo",
        description="Run the Genesis Embryo Core."
    )
    p.add_argument(
        "--mode",
        choices=["stabilize", "aggressive"],
        default="stabilize",
        help="High-level run mode"
    )
    p.add_argument(
        "--reset-heartbeat",
        action="store_true",
        help="Reset the heartbeat counter to zero at start"
    )
    p.add_argument(
        "--goal_mode",
        dest="goal_mode_override",
        choices=["shadow", "full"],
        help="Override goal engine mode"
    )
    p.add_argument(
        "--ram-usage",
        type=float,
        metavar="PCT",
        help="Target RAM usage percentage (0–100)"
    )
    p.add_argument(
        "--cpu-usage",
        type=float,
        metavar="PCT",
        help="Target CPU usage percentage (0–100)"
    )
    p.add_argument(
        "--beats",
        type=int,
        default=10_000,
        help="Number of additional heartbeats to run"
    )
    p.add_argument(
        "--training",
        action="store_true",
        help="Disable all snapshots (tracemalloc & DuckDB exports)"
    )
    return p.parse_args()

GAMMA      = 0.9
BATCH_SIZE = 32

def build_db_path(args):
    """
    Construct a duckdb filename that encodes run parameters.
    e.g. godseed_aggressive_full_ram90_cpu80.db
    """
    parts = ["godseed", args.mode, args.goal_mode_override or "shadow"]
    if args.ram_usage is not None:
        parts.append(f"ram{int(args.ram_usage)}")
    if args.cpu_usage is not None:
        parts.append(f"cpu{int(args.cpu_usage)}")
    return "_".join(parts) + ".db"

class Critic(nn.Module):
    def __init__(self, input_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x)

class Config(BaseModel):
    # ─── Heartbeat & mutation timing ─────────────────────────────
    heartbeat_interval: float  = Field(5.0, gt=0, description="Seconds between heartbeats")
    mutation_interval: float   = Field(30.0, gt=0, description="Seconds between mutation cycles")
    autosave_seconds: int      = Field(300, ge=1, description="Autosave frequency in seconds")
    narration_interval: int    = Field(20, ge=1, description="Heartbeats between reflections")

    # ─── Survival & exploration controls ─────────────────────────
    survival_threshold: float           = Field(0.7, ge=0.0, le=1.0, description="Minimum survival score")
    critical_survival_threshold: float  = Field(0.85, ge=0.0, le=1.0, description="Critical survival drop multiplier")
    rollback_required_cycles: int       = Field(2, ge=1, description="Cycles before rollback")
    exploration_base: float             = Field(0.2, ge=0.0, le=1.0, description="Base exploration probability")
    dynamic_strategy_prob: float        = Field(0.01, ge=0.0, le=1.0, description="Prob. of new strategy")

    # ─── Resource-control thresholds ─────────────────────────────
    metrics_interval: int               = Field(5, ge=1, description="Seconds between health metrics samples")
    target_ram_usage_pct: float         = Field(95.0, ge=0.0, le=100.0, description="Target RAM usage %")
    snapshot_ram_threshold_pct: float   = Field(50.0, ge=0.0, le=100.0, description="Only take tracemalloc snapshots if RAM% < this")
    training_ram_threshold_pct: float   = Field(85.0, ge=0.0, le=100.0, description="Only run critic training if RAM% < this")
    mutation_ram_threshold_pct: float   = Field(90.0, ge=0.0, le=100.0, description="Only run mutation cycles if RAM% < this")
    target_cpu_usage_pct: float         = Field(100.0, ge=0.0, le=100.0, description="Target CPU usage %")

    # ─── Genetic Algorithm hyper-parameters ─────────────────────
    mutation_rate: float  = Field(0.1, ge=0.0, le=1.0, description="Probability of mutation per gene")
    crossover_rate: float = Field(0.5, ge=0.0, le=1.0, description="Probability of crossover")

    # ─── Gene-count evolution bounds ─────────────────────────────
    gene_count_min: int    = Field(1, ge=0, description="Minimum gene count")
    gene_count_max: int    = Field(10, ge=0, description="Maximum gene count")
    elite_archive_size: int= Field(5, ge=1, description="Number of champions to keep")

    # ─── Mutation-rate annealing ─────────────────────────────────
    mutation_rate_initial: float = Field(0.3, ge=0.0, le=1.0, description="Starting mutation rate")
    mutation_rate_floor:   float = Field(0.05, ge=0.0, le=1.0, description="Minimum mutation rate")
    mutation_rate_anneal:  float = Field(0.99, ge=0.0, le=1.0, description="Annealing factor per cycle")

    # ─── Goal engine mode ────────────────────────────────────────
    goal_mode: Literal["shadow", "full"] = Field(
        "shadow",
        description="‘shadow’ (Q-updates only) or ‘full’ (plus act_on_goal)"
    )

    # ─── TASK PARAMETERS (for tweak_* strategies) ─────────────────
    task_params: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            'cpu_burst':    {'threshold_delta': 10.0, 'duration': 10},
            'memory_spike': {'threshold_delta': 10.0, 'duration': 10},
            'io_stress':    {'threshold_delta': 0.0,  'duration': 10},
            'network_spike':{'threshold_delta': 0.1,  'duration': 10},
        },
        description="Default parameters for procedural stress-test tasks"
    )

    # ─── Micro-goal thresholds & bonuses ─────────────────────────
    cpu_threshold: float = Field(0.8, ge=0.0, le=1.0, description="CPU health to consider OK")
    mem_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Memory health to consider OK")
    disk_threshold: float= Field(0.8, ge=0.0, le=1.0, description="Disk health to consider OK")

    cpu_ok_streak: int    = Field(3, ge=1, description="Consecutive OK cycles on CPU for bonus")
    mem_ok_streak: int    = Field(3, ge=1, description="Consecutive OK cycles on Memory for bonus")
    disk_ok_streak: int   = Field(3, ge=1, description="Consecutive OK cycles on Disk for bonus")

    cpu_bonus: float      = Field(0.05, ge=0.0, description="Reward bonus on CPU metric")
    mem_bonus: float      = Field(0.05, ge=0.0, description="Reward bonus on Memory metric")
    disk_bonus: float     = Field(0.05, ge=0.0, description="Reward bonus on Disk metric")

    # ─── Initial multi-objective weights ─────────────────────────
    initial_weights: List[float] = Field(
        default_factory=lambda: [1/4] * 4,
        description="Starting weights for [cpu, memory, disk, network]"
    )


class ConfigManager:
    """
    Wraps a Pydantic Config model and persists it to disk as JSON.
    Loads existing config or creates defaults, and provides get/set accessors.
    """

    def __init__(self, path: str = "config.json"):
        self.path = Path(path)
        self.cfg = self._load_or_create()

    def _load_or_create(self) -> Config:
        if not self.path.exists():
            logging.info(f"[CONFIG] No config found at {self.path!r}, creating default.")
            return self._create_default()

        try:
            raw = self.path.read_text(encoding="utf-8")
            return Config.model_validate_json(raw)
        except (ValidationError, ValueError) as e:
            logging.warning(f"[CONFIG] Failed to parse {self.path!r}: {e!r}, resetting to defaults.")
            return self._create_default()

    def _create_default(self) -> Config:
        cfg = Config()
        self._write_config(cfg)
        return cfg

    def _write_config(self, cfg: Config) -> None:
        json_data = cfg.model_dump_json(indent=4)
        try:
            self.path.write_text(json_data, encoding="utf-8")
        except OSError as e:
            logging.error(f"[CONFIG] write_text failed for {self.path!r}: {e!r}, retrying with open().")
            try:
                with open(self.path, "w", encoding="utf-8", newline="") as f:
                    f.write(json_data)
            except OSError as e2:
                logging.critical(f"[CONFIG] fallback write also failed for {self.path!r}: {e2!r}")

    def save(self) -> None:
        """Persist the current config to disk."""
        self._write_config(self.cfg)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a config value or return default."""
        return getattr(self.cfg, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value and immediately persist."""
        setattr(self.cfg, key, value)
        self._write_config(self.cfg)


# ─── Build and populate global StrategyRegistry ─────────────────────────
cfg_mgr     = ConfigManager()
task_params = cfg_mgr.get(
    "task_params",
    {
        'cpu_burst':     {'threshold_delta': 10.0, 'duration': 10},
        'memory_spike':  {'threshold_delta': 10.0, 'duration': 10},
        'io_stress':     {'threshold_delta':  0.0, 'duration': 10},
        'network_spike': {'threshold_delta':  0.1, 'duration': 10},
    }
)
registry    = StrategyRegistry(sample_state=None)
registry.register_defaults(task_params)

# ───────────────────────────────────────────────────────────────────────────────
# CURIOSITY

class Curiosity:
    def __init__(self, base):
        self.base = base
        self.count = 0
        self.records = defaultdict(_curiosity_default)
        self.fatal = 0

    def update(self, target, success: bool):
        r = self.records[target]
        r["att"] += 1
        r["succ"] += int(success)

    def record_fatal(self, desc=None):
        """
        Just increment a counter (no lambda in here).
        """
        self.fatal += 1

    def exploration_rate(self):
        total = sum(r["att"] for r in self.records.values()) or 1
        success = sum(r["succ"] for r in self.records.values())
        factor = 0.5 if self.fatal > 5 else 1.0
        rate = self.base * factor * (1 / (success / total + 1e-4))
        return min(0.5, max(0.05, rate))

    def choose(self, targets):
        if random.random() < self.exploration_rate():
            t = random.choice(targets)
            logger.info(f"[CURIOSITY] Exploring {t}")
        else:
            best = max(
                ((self.records[t]["succ"] / max(1, self.records[t]["att"]), t)
                 for t in targets),
                key=lambda x: x[0]
            )[1]
            logger.info(f"[CURIOSITY] Exploiting {best}")
            t = best
        return t

logger = logging.getLogger(__name__)

# EMBRYO CORE
class Embryo:
    def __init__(
        self,
        config,
        db_path: str = "godseed_memory.db",
        goal_mode_override=None,
        disable_snapshots: bool = False,
        critic: nn.Module = None,
        launch_args=None
    ):

        # Device & critic
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[EMBRYO INIT] using device: {self.device}")
        self.critic = (
            critic.to(self.device)
            if critic is not None
            else Critic(input_dim=5, hidden=32).to(self.device)
        )
        self.critic.eval()

        # Config, modes, args
        self.cfg = config
        self.launch_args = launch_args or argparse.Namespace()

            # Crash tracker
        self.CrashTracker = CrashTracker()
        if getattr(self.launch_args, "training", False):
            self.CrashTracker.clear()
        self.past_crashes = self.CrashTracker.crashes
        # ─── NETWORK DELTA SEEDING ─────────────────────────
        # so that first call to collect_metrics has something to diff against
        self._last_net_counters   = psutil.net_io_counters()
        self._prev_pernic_counters= psutil.net_io_counters(pernic=True)
        self._iface_speeds        = {
            nic: getattr(s, 'speed', 0) or 0
            for nic, s in psutil.net_if_stats().items()
        }

        # ─── MICRO-GOAL CONFIG ──────────────────────────────
        self.cpu_threshold  = config.get("cpu_threshold")
        self.mem_threshold  = config.get("mem_threshold")
        self.disk_threshold = config.get("disk_threshold")

        self.cpu_ok_streak  = config.get("cpu_ok_streak")
        self.mem_ok_streak  = config.get("mem_ok_streak")
        self.disk_ok_streak = config.get("disk_ok_streak")

        # store the raw bonus values under *_bonus_val so we NEVER override
        self.cpu_bonus_val  = config.get("cpu_bonus")
        self.mem_bonus_val  = config.get("mem_bonus")
        self.disk_bonus_val = config.get("disk_bonus")

        # ─── BUILD THE micro_goals DICTIONARY ─────────────────────────
        # Used by _metric_bonus() to track counts and parameters
        self.micro_goals = {}
        for metric, thr_attr, streak_attr, bonus_attr in [
            ("cpu",    "cpu_threshold",  "cpu_ok_streak",  "cpu_bonus_val"),
            ("memory", "mem_threshold",  "mem_ok_streak",  "mem_bonus_val"),
            ("disk",   "disk_threshold", "disk_ok_streak", "disk_bonus_val"),
        ]:
            self.micro_goals[metric] = {
                "threshold": getattr(self, thr_attr),
                "ok_streak": getattr(self, streak_attr),
                "bonus":     getattr(self, bonus_attr),
                "ok_count":  0
            }

        # ── NEW: initialize attrs your logic expects ─────────────
        self.prev_state = None
        self.action_embedding = None
        self.goal_mode = goal_mode_override or self.cfg.get("goal_mode") or "shadow"

        # ─── CONFIGURED INTERVALS ──────────────────────
        self.heartbeat_interval = self.cfg.get("heartbeat_interval")
        self.mutation_interval  = self.cfg.get("mutation_interval")
        self.autosave_seconds   = self.cfg.get("autosave_seconds")
        self.narration_interval = self.cfg.get("narration_interval")
        self.metrics_interval   = self.cfg.get("metrics_interval")

        # Core systems
        self.mem           = MemoryArchive()              # In‐memory log
        # --- initialize your state + memory backends ---
        self.state_path = db_path  # existing DuckDB file for survival_details, transitions, etc.
        self.memory_path = db_path.replace(".duckdb", "_memory.db")  
        # or wherever you want to store godseed_memory.db

        # 1) in-memory metrics DB
        self.db = MemoryDB(path=self.memory_path)

        # 2) snapshot manager now tracks *both* databases
        from persistence import SnapshotManager
        self.snap_mgr = SnapshotManager(
            db_paths=[ self.state_path, self.memory_path ],
            snap_dir=self.cfg.get("snap_dir", "snapshots")
)
        self._log_pool = ThreadPoolExecutor(max_workers=8)
        self.cur          = Curiosity(self.cfg.get("exploration_base"))
        self.emotions     = EmotionalDrives()
        self.disable_snapshots = disable_snapshots
        self.health       = HealthMonitor(sample_interval=self.metrics_interval)
        self.state_path = db_path
        self.env_info    = scan_environment()

        # Dynamic resource manager
        self.resource_manager = DynamicResourceManager(
            crash_tracker=self.CrashTracker,
            sample_interval=self.metrics_interval,
            target_cpu_pct=self.cfg.get("target_cpu_usage_pct"),
            target_mem_pct=self.cfg.get("target_ram_usage_pct"),
        )
        self.resource_manager.start()


        
        # ─── HEARTBEAT COUNTERS & METRICS ────────────────────────
        # these back private counters used in think()
        self.hb = Heartbeat()      
        self.last_save = time.time()   # track when we last autosaved
        self.last_mut  = time.time()   # track when we last mutated
        self._hb_counter     = 0
        self._metrics_interval = self.metrics_interval
        # seed last metrics so think() can reference it immediately
        self._last_metrics   = SystemHealth.check()
        self._last_raw_metrics = SystemHealth.check()
        self.metrics = {**self._last_raw_metrics, **Survival.score(self._last_raw_metrics)}

        # 1. mutation strategy history (attempts & successes)
        self.archive = Archive(config.get("elite_archive_size", 5))
        self.strategy_history = {
            'attempts': 0,
            'success': 0,
        }

        # 2. stats for synthesized strategies
        self.synth_strategy_stats = {}

        # 3. champion‐state archive (min‐heap of top‐k)
        archive_size = self.cfg.get("archive_size", 20)  # or whatever default you want
        self.archive = Archive(archive_size)

        # 4. rolling‐window of composite survival scores
        plateau_window = self.cfg.get("plateau_window", 50)
        self.score_stats = RollingStats(plateau_window)

        self.current_goal    = None

        # ─── Initialize RL / planning pointers so think() won't crash ─────────
        self.prev_state      = None          # for critic replay & TD updates
        self.prev_reward     = 0.0           # reward from last step
        self.prev_composite  = 0.0           # last composite MO-score

        # ─── STATE I/O BACKENDS ─────────────────────────────────────      
        self.run_id        = f"run_{uuid.uuid4().hex}"

        # EfficientStateManager will track small‐delta checkpoints in memory
        self.state_manager = EfficientStateManager()

        # Now pass both to DuckdbStateIO:
        self.duckdb_state_io = DuckdbStateIO(self.state_manager, self.db_path)


        self.mode = getattr(self.launch_args, "mode", "stabilize")
        logger.info(f"[EMBRYO INIT] Run mode: {self.mode.upper()}")

        # Control parameters
        self.survival_threshold      = self.cfg.get("survival_threshold")
        self.rollback_required       = self.cfg.get("rollback_required_cycles")
        self.mutation_rate           = self.cfg.get("mutation_rate_initial")
        self.dynamic_strategy_prob   = self.cfg.get("dynamic_strategy_prob")
        self.gene_min                = self.cfg.get("gene_count_min")
        self.gene_max                = self.cfg.get("gene_count_max")
        self.gene_count              = self.gene_min

        # Per-task dynamic parameters (pulled from config)
        self.task_params = self.cfg.get("task_params", {
            'cpu_burst':           {'threshold_delta': 10.0, 'duration': 10},
            'memory_spike':        {'threshold_delta': 10.0, 'duration': 10},
            'io_stress':           {'threshold_delta': 0.0,  'duration': 10},
            'network_spike':       {'threshold_delta': 0.1,  'duration': 10},
        })

        """
        sample_state — an Embryo (for seeding any state‐based strategies)
        cfg          — ConfigManager
        registry     — the globally‐shared StrategyRegistry
        """

        # ─── STATE DIMENSION (for world‐model & goal‐engine) ─────────
        # must be set before we compute wm_cfg or instantiate GoalEngine
        self.state_size = 5

        # registry is the module-level StrategyRegistry you built at the bottom of this file
        self.mutator = MutationEngine(
            sample_state=self,
            cfg=self.cfg,
            registry=registry
        )

        # 1) Define initial mutation probabilities for meta-strategies
        init_probs = {
            'const': 0.3,
            'try': 0.1,
            'comprehension': 0.2
        }
        # 2) Instantiate the MetaStrategyEngine with those probabilities
        self.meta_engine = MetaStrategyEngine(
            registry,
            batch_size=10,
            interval=30.0,
            mutation_probs=init_probs
        )

        # 3) Instantiate the WeightManager, seeding it with the same initial mutation_probs
        #    so that it can begin to learn and adjust them over time
        self.weight_mgr = WeightManager(
            init_weights=self.cfg.get("initial_weights", [1/4] * 4),
            lr=self.cfg.get("weight_lr", 5e-3),
            init_mutation_probs=self.meta_engine.mutation_probs
        )

        # Parameter bounds
        self.param_bounds = self.cfg.get("param_bounds", {
            "heartbeat_interval":   [0.5, 30.0],
            "survival_threshold":   [0.4, 0.95],
            "mutation_rate":        [0.01, 0.5],
            "mutation_interval":    [10.0, 120.0],
            "rollback_required":    [1, 5],
            "dynamic_strategy_prob":[0.001, 0.1],
            "gene_min":             [1, 5],
            "gene_max":             [5, 20],
                # add micro‐goal bounds
            "cpu_threshold":  [0.0, 1.0],
            "mem_threshold":  [0.0, 1.0],
            "disk_threshold": [0.0, 1.0],
            "cpu_ok_streak":  [1, 10],
            "mem_ok_streak":  [1, 10],
            "disk_ok_streak": [1, 10],
            "cpu_bonus_val":      [0.0, 1.0],
            "mem_bonus_val":      [0.0, 1.0],
            "disk_bonus_val":     [0.0, 1.0], })

        # Action/Goal spaces
        self.state_size   = 5
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.goal_engine  = GoalEngine([], self.db_path)
        self.goal_engine.embryo = self
        self.goal_gen     = GoalGenerator(self)

        # ─── ACTION & EMBEDDING ──────────────────────
        # (We do _not_ build the embedding until after we know choice_emb_dim.)
        self.sync_actions()

        # ─── WORLD‐MODEL & TRAINER ────────────────────
        # (a) Pull dims from config early so we can use choice_emb_dim below.
        state_dim = self.state_size
        choice_emb_dim = int(self.cfg.get("world_model_choice_dim", 8))
        hidden_dim     = int(self.cfg.get("world_model_hidden", 128))
        learning_rate  = float(self.cfg.get("world_model_lr", 1e-3))
        grad_clip      = self.cfg.get("world_model_grad_clip", settings.WORLD_MODEL_GRAD_CLIP)

        # (b) Build a single WorldModel instance and move to device:
        self.world_model = WorldModel(state_dim, choice_emb_dim, hidden_dim).to(self.device)

        # (c) Instantiate the WorldModelTrainer on exactly that same instance:
        self.world_trainer = WorldModelTrainer(
            self.world_model,
            lr=learning_rate,
            grad_clip=grad_clip
        )

        # ─── NOW that choice_emb_dim is in scope, build the action_embedding ──
        num_actions = len(self.action_space)
        self.action_embedding = nn.Embedding(num_actions, choice_emb_dim).to(self.device)

        # ─── Initialize planning weights (size = state_dim) ───────────────────
        self.planning_weights = [1.0 / state_dim] * state_dim

        # ─── Other replay buffers, record‐keeping, etc. ───────────────────────
        self.critic_replay = PrioritizedReplay(capacity=10_000, alpha=0.6)
        self.wm_replay     = PrioritizedReplay(capacity=5_000, alpha=0.6)

        # Strategy history & stats
        self.strategy_history = {k: 0 for k in self.mutator.weights}
        self.strategy_history.update(attempts=0, success=0)
        self.score_stats = RollingStats(maxlen=self.cfg.get("PLATEAU_WINDOW", 50))

        # Metrics & novelty
        self.metrics          = {}
        self.behavior_archive = []
        self.novelty_k        = self.cfg.get("novelty_k", 5)
        self.min_archive_size = self.cfg.get("min_archive_size", 20)

        # Generation & mutation counters
        self.bad_cycles = 0

        logger.info(f"[EMBRYO INIT] Goal engine in {self.goal_mode.upper()} mode")


    def _mg(self, metric: str, field: str):
        """Micro-goal helper: e.g. self._mg('cpu', 'bonus')"""
        return self.micro_goals[metric][field]

    def _metric_bonus(self, metric_name: str, score: float) -> float:
        # use the helper to pull each value
        thr      = self._mg(metric_name, "threshold")
        streak   = self._mg(metric_name, "ok_streak")
        bonus    = self._mg(metric_name, "bonus")
        ok_count = self._mg(metric_name, "ok_count")

        if score < thr:
            # reset the counter
            self.micro_goals[metric_name]["ok_count"] = 0
            return -bonus

        ok_count += 1
        if ok_count >= streak:
            # give bonus and reset
            self.micro_goals[metric_name]["ok_count"] = 0
            return +bonus
        else:
            # update the counter
            self.micro_goals[metric_name]["ok_count"] = ok_count
            return 0.0

    def apply_action_by_index(self, idx: int):
        """
        Execute the mutation corresponding to action_space[idx],
        via the SynthStrategy stored in the registry.
        """
        try:
            name = self.action_space[idx]
        except IndexError:
            raise ValueError(f"Invalid action index {idx}")

        strat = self.mutator.get_strategy(name)
        if strat is None:
            raise ValueError(f"No strategy '{name}' registered")

        # apply() will run the underlying fn, track errors, successes, etc.
        desc, ctx = strat.apply(self)
        return desc, ctx


    def cpu_bonus(self, cpu_score: float) -> float:
        return self._metric_bonus("cpu", cpu_score)

    def memory_bonus(self, mem_score: float) -> float:
        return self._metric_bonus("memory", mem_score)

    def disk_bonus(self, disk_score: float) -> float:
        return self._metric_bonus("disk", disk_score)


    def _compute_novelty(self, vec: List[float]) -> float:
        """
        Novelty = average distance to the k nearest behaviors in archive.
        """
        if len(self.behavior_archive) < self.min_archive_size:
            return 0.0

        # compute distances
        dists = [math.dist(vec, past) for past in self.behavior_archive]
        k = min(self.novelty_k, len(dists))
        nearest = heapq.nsmallest(k, dists)
        return sum(nearest) / k

    def _compute_efficiency(self, surv_dict: Dict[str, float]) -> float:
        """
        Efficiency = average of the 4 survival sub-scores
        (cpu, memory, disk, network).
        """
        scores = [surv_dict[k] for k in BEHAVIOR_KEYS]
        return sum(scores) / BEHAVIOR_DIM

    def __deepcopy__(self, memo):
        """
        Return a deep copy of self without re-running __init__,
        skipping non-pickleable components (DB, threads, models, etc.).
        """
        cls = self.__class__
        new = cls.__new__(cls)

        # 1) Shallow-copy the long-lived/non-pickleable resources:
        skip_attrs = (
            'db', 'snap_mgr', 'state_manager', 'duckdb_state_io',
            'world_model', 'world_trainer', 'action_embedding',
            'critic', 'goal_engine', 'goal_gen', 'health',
            '_log_pool', 'resource_manager',
            'mutator', 'meta_engine', 'weight_mgr', 'CrashTracker'
        )
        for attr in skip_attrs:
            if hasattr(self, attr):
                setattr(new, attr, getattr(self, attr))

        # 2) Deep-copy everything else into the new instance:
        for key, val in self.__dict__.items():
            if key not in skip_attrs:
                setattr(new, key, copy.deepcopy(val, memo))

        return new



    def __getstate__(self):
        """
        Return a dict containing only pickleable state.
        We do **not** pickle a live DuckDB connection; instead we store db_path.
        """
        state = {
            # ─── Basic configuration flags ────────────────────────────────
            'mode':                   self.mode,
            'current_goal':           getattr(self, 'current_goal', None),

            # ─── Heartbeat & Curiosity ────────────────────────────────────
            'hb':                     type('HB', (), {'count': self.hb.count})(),
            'cur':                    copy.deepcopy(self.cur),

            # ─── Mutation parameters ──────────────────────────────────────
            'mutation_interval':      self.mutation_interval,
            'heartbeat_interval':     self.heartbeat_interval,
            'survival_threshold':     self.survival_threshold,
            'rollback_required':      self.rollback_required,
            'dynamic_strategy_prob':  self.dynamic_strategy_prob,

            # ─── Gene‐count / bounds ──────────────────────────────────────
            'gene_count':             self.gene_count,
            'gene_min':               self.gene_min,
            'gene_max':               self.gene_max,

            # ─── Strategy history & stats ─────────────────────────────────
            'strategy_history':       self.strategy_history.copy(),
            'synth_strategy_stats':   dict(getattr(self, 'synth_strategy_stats', {})),
            'bad_cycles':             self.bad_cycles,

            # ─── Archives & scoring ───────────────────────────────────────
            'archive':                self.archive,
            'score_stats':            copy.deepcopy(self.score_stats),
            'metrics':                dict(self.metrics),

            # ─── Parameter bounds ─────────────────────────────────────────
            'param_bounds':           {k: tuple(v) for k, v in self.param_bounds.items()},

            # ─── Task‐specific & novelty settings ─────────────────────────
            'task_params':            copy.deepcopy(self.task_params),
            'behavior_archive':       list(self.behavior_archive),
            'novelty_k':              self.novelty_k,
            'min_archive_size':       self.min_archive_size,

            # ─── Memory & database ────────────────────────────────────────
            'mem':                    self.mem,
            'db_path':                self.db_path,      # <<< store only the path—no live connection

            # ─── Mutator & CrashTracker ───────────────────────────────────
            'mutator':                self.mutator,
            'CrashTracker':           self.CrashTracker,
        }
        return state

    def __setstate__(self, state: dict):
        """
        Restore every attribute from __getstate__.
        Re‐instantiate anything (like the DB connection) that we did not pickle.
        """
        # ─── Basic configuration flags ────────────────────────────────
        self.mode                   = state['mode']
        self.current_goal           = state['current_goal']

        # ─── Heartbeat & Curiosity ────────────────────────────────────
        hb_state = state['hb']
        if isinstance(hb_state, Heartbeat):
            self.hb = hb_state
        else:
            self.hb = Heartbeat()
            self.hb.count = getattr(hb_state, 'count', 0)
        self.cur                    = state['cur']

        # ─── Mutation parameters ──────────────────────────────────────
        self.mutation_interval      = state['mutation_interval']
        self.heartbeat_interval     = state['heartbeat_interval']
        self.survival_threshold     = state['survival_threshold']
        self.rollback_required      = state['rollback_required']
        self.dynamic_strategy_prob  = state['dynamic_strategy_prob']

        # ─── Gene‐count / bounds ──────────────────────────────────────
        self.gene_count             = state['gene_count']
        self.gene_min               = state['gene_min']
        self.gene_max               = state['gene_max']

        # ─── Strategy history & stats ─────────────────────────────────
        self.strategy_history       = state['strategy_history']
        self.synth_strategy_stats   = state['synth_strategy_stats']
        self.bad_cycles             = state['bad_cycles']

        # ─── Archives & scoring ───────────────────────────────────────
        self.archive                = state['archive']
        self.score_stats            = state['score_stats']
        self.metrics                = state['metrics']

        # ─── Parameter bounds ─────────────────────────────────────────
        self.param_bounds           = {k: list(v) for k, v in state['param_bounds'].items()}

        # ─── Task‐specific & novelty settings ─────────────────────────
        self.task_params            = state['task_params']
        self.behavior_archive       = state['behavior_archive']
        self.novelty_k              = state['novelty_k']
        self.min_archive_size       = state['min_archive_size']

        # ─── Memory & database ────────────────────────────────────────
        self.mem                    = state['mem']
        # Re‐open the DuckDB connection from that path now that we are unpickling:
        db_path = state['db_path']
        self.db = MemoryDB(path=db_path)

        # ─── Mutator & CrashTracker ───────────────────────────────────
        self.mutator                = state['mutator']
        self.CrashTracker           = state['CrashTracker']

        # ─── Executor (re‐create anything __deepcopy__ would) ──────────
        self._log_pool = ThreadPoolExecutor(max_workers=4)

        # ─── Dynamic Resource Manager ───────────────────────────────
        self.resource_manager = DynamicResourceManager(
            crash_tracker=self.CrashTracker,
            sample_interval=self.metrics_interval,
            target_cpu_pct=self.cfg.get("target_cpu_usage_pct"),
            target_mem_pct=self.cfg.get("target_ram_usage_pct"),
        )
        self.resource_manager.start()


    def adjust_gene_count(self, result):
        if result == "SUCCESS" and self.gene_count < self.gene_max:
            self.gene_count += 1
        elif result == "FAILURE" and self.gene_count > self.gene_min:
            self.gene_count -= 1
        logger.info(f"[GENE COUNT] adjusted to {self.gene_count}")

    def evolve_param_bounds(self, param, success):
        lower, upper = self.param_bounds[param]
        range_span = upper - lower
        adjustment = range_span * (0.02 if success else -0.01)
        self.param_bounds[param][0] = max(lower - adjustment, 0.01)
        self.param_bounds[param][1] = upper + adjustment

    def apply_param_bounds(self, param, value):
        lower, upper = self.param_bounds[param]
        return min(max(value, lower), upper)

    # 7. Save Q-table after run ends:
    def save_q_table(self):
        self.goal_engine.save_q_table()
        logger.info("[GOAL ENGINE] Q-table saved to disk")

    def save_all(self):
        self.duckdb_state_io.save(self)
        self.cfg.save()

    # 8. Fix missing save_state method:
    def save_state(self):
        self.save_all()
        self.save_q_table()

    def reflect(self):
        try:
            recent_survival = self.score_history[-1] if self.score_history else None
            recent_trend = ''.join([
                "↑" if self.score_history[i] < self.score_history[i + 1] else "↓"
                for i in range(len(self.score_history) - 1)
            ]) if len(self.score_history) > 1 else "N/A"
            most_used = max(
                (k for k in self.strategy_history if k not in ["success","attempts"]),
                key=lambda k: self.strategy_history[k], default="N/A"
            )
            summary = f"""
{Fore.YELLOW}[REFLECTION] Heartbeat #{self.hb.count}{Style.RESET_ALL}
• Mode: {self.mode}
• Gene Count: {self.gene_count}
• Heartbeat Interval: {round(self.heartbeat_interval,3)}s
• Survival Threshold: {round(self.survival_threshold,3)}
• Recent Survival Score: {recent_survival}
• Trend (last {len(self.score_history)}): {recent_trend}
• Most Used Strategy: {most_used}
• Mutation Rate: {round(self.mutation_rate,4)}
"""
            self.db.record_reflection(
                self.hb.count, self.mode, self.gene_count,
                round(self.heartbeat_interval,3), round(self.survival_threshold,3),
                recent_survival, recent_trend, most_used, round(self.mutation_rate,4)
            )
            logger.info(summary.strip())
            self.mem.log("REFLECTION", summary.strip().replace("\n"," "))
        except Exception as e:
            logger.error(f"[REFLECTION ERROR] {e}")

    PLATEAU_WINDOW = 50
    IMPROVEMENT_THRESH = 0.01

    def has_plateaued(self) -> bool:
        return len(self.score_stats) == self.PLATEAU_WINDOW \
               and self.score_stats.range() < self.IMPROVEMENT_THRESH

    def collect_state(self):
        return [
            self.metrics.get("survival", 0.0),
            self.metrics.get("novelty", 0.0),
            self.metrics.get("efficiency", 0.0),
            self.metrics.get("mutation_error", 0.0),
            self.metrics.get("cycle", 0.0)
        ]

    def _generate_recompute_rule(self):
        return None  # Or a lambda that takes a state and returns a transformed state


    def act_on_goal(self, goal_name):
        logger.info(f"[GOAL EXECUTION] Acting on goal: {goal_name}")
        param_map = {
            "increase_survival_rate": "survival_threshold",
            "reduce_mutation_error": "mutation_rate",
            "maximize_novelty_score": "dynamic_strategy_prob",
            "improve_efficiency": "heartbeat_interval",
        }
        if goal_name in param_map:
            param = param_map[goal_name]
            value = getattr(self, param, None)
            if value is not None:
                updated = value * 1.05 if "increase" in goal_name or "maximize" in goal_name else value * 0.95
                updated = self.apply_param_bounds(param, updated)
                setattr(self, param, updated)
                logger.info(f"[PARAM ADJUST] {param} -> {updated:.4f}")

    def evaluate_reward(self):
        return self.metrics.get("reward", 0)

        
    def think(self):
        """
        One heartbeat: collect fresh metrics, model-based planning, compute composite with learned weights,
        update world-model, prioritized Q-learner, dynamic weights, emergent goals, autosave & schedule mutate.
        """
        proc = psutil.Process()
        mem_pct = proc.memory_percent()
        if mem_pct > self.cfg.get("target_ram_usage_pct"):
            logging.warning(f"[MEMORY] at {mem_pct:.1f}% RAM — skipping heavy work")
            self.hb.beat()
            logging.debug(f"[HEARTBEAT] #{self.hb.count} (skipped)")
            return

        try:
            # ─── HEARTBEAT & SNAPSHOT ───────────────────────────────────
            self.hb.beat()
            logging.debug(f"[HEARTBEAT] #{self.hb.count}")

            if not getattr(self, 'disable_snapshots', False) and hasattr(self, 'snap_mgr'):
                # every N beats, export into the *single* parquet_export dir
                if self.hb.count % 10 == 0 and mem_pct < self.cfg.get("snapshot_ram_threshold_pct"):
                    logging.info(f"[SNAPSHOT] exporting to {self.snap_mgr.parquet_dir}")
                    # no argument → uses self.parquet_dir
                    self.snap_mgr.export_snapshot()
            # ─── PERIODIC REFLECTION ───────────────────────────────────────────────
                # every N heartbeats, run reflect() so it writes into the new reflections table
                if self.hb.count % self.cfg.get("reflect_interval", 100) == 0:
                    self.reflect()

            # ─── METRICS SAMPLING ────────────────────────────────────────
            if self._hb_counter % self._metrics_interval == 0:
                raw_metrics = self.collect_metrics()
            else:
                raw_metrics = self._last_raw_metrics
            self._last_raw_metrics = raw_metrics

            # compute survival sub-scores
            score = Survival.score(raw_metrics)
            combined = {**raw_metrics, **score}
            self.raw_metrics = raw_metrics
            self.metrics     = combined

            surv_dict   = score
            survival    = surv_dict["composite"]
            cpu_s       = surv_dict['cpu']
            memory_s    = surv_dict['memory']
            disk_s      = surv_dict['disk']
            network_s   = surv_dict['network']

            logger.info(
                f"[SURVIVAL SCORE] composite={survival:.4f}, cpu={cpu_s:.4f}, "
                f"memory={memory_s:.4f}, disk={disk_s:.4f}, network={network_s:.4f}"
            )

            # ─── BEHAVIOR, NOVELTY, EFFICIENCY, BONUS ───────────────────
            behavior_vec = [surv_dict[k] for k in BEHAVIOR_KEYS]
            novelty      = self._compute_novelty(behavior_vec)
            self.behavior_archive.append(behavior_vec)
            efficiency   = self._compute_efficiency(surv_dict)
            bonus = (self.cpu_bonus(cpu_s)
                   + self.memory_bonus(memory_s)
                   + self.disk_bonus(disk_s))
            composite_with_bonus = surv_dict['composite'] + bonus

            # ─── Emotional drives update ───────────────────────────────
            pred_err = abs(survival - self.prev_composite)
            self.emotions.record_prediction_error(pred_err)
            self.emotions.record_observation(survival)
            curiosity_score = self.emotions.curiosity_score
            novelty_drive   = self.emotions.novelty_score

            # Record metrics for collect_state
            self.metrics.update({
                "survival":        survival,
                "novelty":         novelty,
                "efficiency":      efficiency,
                "bonus":           bonus,
                "cycle":           self.hb.count,
                "mutation_error":  0.0,
                "curiosity_score": curiosity_score,
                "novelty_score":   novelty_drive
            })

            # ─── MULTI-OBJECTIVE SCORING ────────────────────────────────
            w_cpu, w_mem, w_disk, w_net = self.weight_mgr.get_weights()
            composite = (
                w_cpu   * surv_dict['cpu']
              + w_mem   * surv_dict['memory']
              + w_disk  * surv_dict['disk']
              + w_net   * surv_dict['network']
            )
            logger.info(
                f"[MO-SCORE] w=[{w_cpu:.2f},{w_mem:.2f},{w_disk:.2f},{w_net:.2f}] -> {composite:.3f}"
            )
            logger.info(
                f"[HB] {self.hb.count} composite={composite:.4f}"
            )

            # Record basic heartbeat info for persistence
            self.db.record_heartbeat(self.hb.count, composite)

            if self.hb.count == 250:
                # throttle down statement-level mutations
                self.meta_engine.mutation_probs['stmt'] = 0.05
                # maybe boost try/except wrapping a bit
                self.meta_engine.mutation_probs['try'] = 0.2

            # ─── HIGH-FREQUENCY DB INSERT ─────────────────
            # Instead, buffer rows and flush in bulk every N rows or T seconds:
            cpu_pct   = raw_metrics.get('cpu_percent', 0.0)
            dio       = raw_metrics.get('disk_io')
            interval  = raw_metrics.get('sample_interval', settings.INTERVAL_SECONDS) or settings.INTERVAL_SECONDS
            disk_io   = ((getattr(dio, 'read_bytes', 0)+getattr(dio, 'write_bytes', 0))/(1024*1024*interval)) if dio else 0.0
            self.db.insert_many('survival_details', [(
                self.hb.count,
                self.gene_count,
                cpu_s,
                memory_s,
                disk_s,
                network_s,
                composite,
                cpu_pct,
                disk_io
            )])

            # ─── PRIORITIZED Q-LEARNING REPLAY & CRITIC UPDATE ─────────
            state = self.collect_state()
            if self.prev_state is not None:
                with torch.no_grad():
                    prev_q = self.critic(
                        torch.tensor(self.prev_state, dtype=torch.float32)
                             .unsqueeze(0).to(self.device)
                    ).item()
                    next_q = self.critic(
                        torch.tensor(state, dtype=torch.float32)
                             .unsqueeze(0).to(self.device)
                    ).item()
                    target = self.prev_reward + GAMMA * next_q
                td_error = abs(target - prev_q)
                self.critic_replay.add((self.prev_state, self.prev_reward, state), td_error)

            self.prev_state, self.prev_reward = state, composite

            if (len(self.critic_replay) >= BATCH_SIZE
                and mem_pct < self.cfg.get("training_ram_threshold_pct")):
                batch = self.critic_replay.sample(BATCH_SIZE)
                s_batch, r_batch, s_next_batch = zip(*batch)
                states      = torch.tensor(s_batch, dtype=torch.float32).to(self.device)
                rewards     = torch.tensor(r_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
                next_states = torch.tensor(s_next_batch, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    targets = rewards + GAMMA * self.critic(next_states)
                preds = self.critic(states)
                loss = F.mse_loss(preds, targets)
                self.critic_optim.zero_grad()
                loss.backward()
                self.critic_optim.step()
                logger.info(f"[CRITIC] training loss={loss.item():.6f}")

            # ─── DYNAMIC WEIGHT ADAPTATION ───────────────────────────────
            delta = composite - self.prev_composite
            vec4  = np.array([cpu_s, memory_s, disk_s, network_s], dtype=np.float64)
            self.weight_mgr.update(vec4, delta)
            self.prev_composite = composite

            # ─── EMERGENT GOALS & EXECUTION ─────────────────────────────
            goals = self.goal_gen.propose_goals()
            if goals:
                action = self.goal_engine.choose_action(self.collect_state(), goals)
                logger.info(f"[GOAL] chosen -> {action}")
                if self.goal_mode == "full":
                    self.act_on_goal(action)
                self.current_goal = action
            else:
                self.current_goal = None

            # ─── AUTOSAVE & SCHEDULED MUTATION ──────────────────────────
            now_ts = time.time()
            if now_ts - self.last_save > self.cfg.get("autosave_seconds"):
                logger.info("[AUTOSAVE] saving…")
                self.save_all()
                self.last_save = now_ts

            if now_ts - self.last_mut > self.mutation_interval \
               and mem_pct < self.cfg.get("mutation_ram_threshold_pct"):
                logger.info("[MUTATE] interval reached…")
                self.mutate_cycle()
                self.last_mut = now_ts

        except Exception as e:
            traceback.print_exc()
            goal = getattr(self, "current_goal", "unknown")
            tb   = traceback.format_exc()
            self.CrashTracker.record_crash(goal, "think", {"error": str(e), "trace": tb})
            ctx  = {
                "phase":      "think",
                "goal":       goal,
                "message":    str(e),
                "traceback":  tb,
                "heartbeat":  self.hb.count,
                "metrics":    self.metrics.copy(),
            }
            self.CrashTracker.log_crash(ctx)
            logger.error(f"[THINK ERROR] {e}")
        finally:

            gc.collect()


    def _initialize_state(self):
            return {
                "genes": {},
                "metrics": {},
                "performance": [],
            }

    def mutate_cycle(self):
        """
        One mutation cycle: grow strategies, plan via world-model, apply mutation,
        train world-model, update weights & mutation-probs, record transition,
        adjust parameters, and save state.
        """
        # ─── 0) Autonomous strategy growth ─────────────────────────────
        self.meta_engine.generate_and_register()
        self.sync_actions()

        # ─── Seed from archive at start of cycle ────────────────────────
        if hasattr(self.archive, "seed"):
            logger.debug("[ARCHIVE] attempting seed from archive")
            self.archive.seed(self)

        # 0.1) ensure our embedding matches the new action_space size
        n_actions = len(self.action_space)
        old_n, dim = self.action_embedding.num_embeddings, self.action_embedding.embedding_dim
        if n_actions != old_n:
            new_emb = nn.Embedding(n_actions, dim).to(self.device)
            with torch.no_grad():
                # carry over old weights if they exist
                if old_n > 0:
                    new_emb.weight[:old_n].copy_(self.action_embedding.weight)
                # init any new rows
                new_emb.weight[old_n:].normal_(0, 0.01)
            self.action_embedding = new_emb

        if not self.action_space:
            logger.warning("[MUTATE] no strategies registered, skipping plan")
            return

        try:
            # ─── 1) Emergent goals ────────────────────────────────────────
            goals = self.goal_gen.propose_goals()
            if goals:
                choice = self.goal_engine.choose_action(self.collect_state(), goals)
                if self.goal_mode == "full":
                    self.act_on_goal(choice)
                    self.current_goal = choice
                else:
                    self.current_goal = None

            # ─── 2) Snapshot & survival ───────────────────────────────────
            before_state = self.collect_state()
            prev_surv    = Survival.score(SystemHealth.check())['composite']

            # ─── 3) Batched world-model planning ──────────────────────────
            st_t      = torch.tensor(before_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            n_actions = len(self.action_space)
            action_idx = torch.arange(n_actions, device=self.device)
            embs       = self.action_embedding(action_idx)          # [n, param_dim]
            states     = st_t.expand(n_actions, -1)                 # [n, state_dim]
            inputs     = torch.cat([states, embs], dim=1)           # [n, state+param]
            delta_preds= self.world_model(inputs)                   # [n, state_dim]
            pw_tensor  = torch.tensor(self.planning_weights, device=self.device)
            scores     = (delta_preds * pw_tensor.unsqueeze(0)).sum(dim=1)
            act_i      = torch.argmax(scores).item()
            self.apply_action_by_index(act_i)
            choice_emb = embs[act_i].unsqueeze(0)

            # ─── 4) Mutation & strategy apply ─────────────────────────────
            new_score, self.bad_cycles, strat, ctx = mutation_cycle(
                embryo=self,
                meta_weights=self.mutator.weights,
                stagnant_cycles=self.bad_cycles,
                stuck_threshold=self.rollback_required,
                max_reset_weight=self.dynamic_strategy_prob,
                reset_penalty=0.1,
                alpha=0.05,
                return_strategy=True
            )
            logger.info(f"[MUTATION APPLIED] {strat}")

            # If we've been stuck for several cycles, replay the best state
            if self.bad_cycles >= self.rollback_required:
                logger.info(
                    f"[ARCHIVE] replaying best state after {self.bad_cycles} stagnant cycles"
                )
                if hasattr(self.archive, "replay_success"):
                    self.archive.replay_success(self)

            # ─── 5) World-model update ────────────────────────────────────
            after_state  = self.collect_state()
            actual_delta = (
                torch.tensor(after_state, dtype=torch.float32).to(self.device)
                - torch.tensor(before_state, dtype=torch.float32).to(self.device)
            )
            wm_loss = self.world_trainer.train_step(
                torch.tensor(before_state, dtype=torch.float32).unsqueeze(0).to(self.device),
                choice_emb,
                actual_delta.unsqueeze(0)
            )
            logger.info(f"[WORLD MODEL] train loss={wm_loss:.6f}")

            # ─── 6) Adapt planning_weights (single-step) ──────────────────
            reward = new_score - prev_surv
            alpha = 0.01
            delta_np = actual_delta.cpu().numpy()
            pw = np.array(self.planning_weights)
            signal = delta_np / (np.abs(delta_np).sum() + 1e-8)
            pw_new = np.clip(pw + alpha * signal, 1e-6, None)
            self.planning_weights = (pw_new / pw_new.sum()).tolist()

            # ─── 7) Update multi-objective weights ─────────────────────────
            surv_dict = Survival.score(SystemHealth.check())
            metrics = np.array([surv_dict['cpu'], surv_dict['memory'], surv_dict['disk'], surv_dict['network']])
            self.weight_mgr.update(metrics, reward)

            # ─── 8) Adapt mutation-probabilities ──────────────────────────
            self.weight_mgr.update_mutation_probs(strat, reward)
            self.meta_engine.mutation_probs = self.weight_mgr.get_mutation_probs()

            # ─── 9) Persist transition & reward ───────────────────────────
            self.db.record_transition(
                self.hb.count, before_state, strat, self.current_goal or "", reward, after_state
            )
            if self.goal_mode == "full":
                self.goal_engine.reward_goal(after_state, reward)

            # ─── 10) Gene-count & parameter evolution ──────────────────────
            result = "SUCCESS" if new_score  >= prev_surv else "FAILURE"
            self.strategy_history['attempts'] += 1
            if result == "SUCCESS":
                self.strategy_history['success'] += 1
            self.adjust_gene_count(result)
            for param in [
                'heartbeat_interval', 'survival_threshold', 'mutation_rate',
                'mutation_interval', 'dynamic_strategy_prob'
            ]:
                self.evolve_param_bounds(param, new_score >= prev_surv)

            # Summarize mutation outcome
            delta = new_score - prev_surv
            self.db.record_mutation(result, strat, delta)

            # summarize the mutation episode for dashboards
            param_field = ctx.get('param')
            if isinstance(param_field, list):
                param_count = len(param_field)
            elif param_field is None:
                param_count = 0
            else:
                param_count = 1
            self.db.record_mutation_episode(
                str(self.hb.count),
                1,
                param_count,
                {"composite": prev_surv},
                {"composite": new_score},
            )

            self.duckdb_state_io.save(self)


        except Exception as e:
            # robust crash handling
            goal = getattr(self, 'current_goal', 'unknown')
            tb   = traceback.format_exc()
            self.CrashTracker.record_crash(goal, 'mutate_cycle', {'error': str(e), 'trace': tb})
            ctx = {
                'phase': 'mutate_cycle', 'goal': goal,
                'metrics': self.metrics.copy(), 
                'message': str(e), 'traceback': tb, 'heartbeat': self.hb.count
            }
            self.CrashTracker.log_crash(ctx)
            logger.error(f"[MUTATE ERROR] {e}")
            self.save_all()

    def sync_actions(self):
        # pull strategy names out of registry
        self.action_space = [s.name for s in self.mutator.get_all_strategies()]

    def collect_metrics(self):
        """
        Gather all system metrics *including* real MB/s network deltas.
        """
        m = SystemHealth.check(
            prev_net=self._last_net_counters,
            prev_pernic=self._prev_pernic_counters,
            iface_speeds=self._iface_speeds,
            compute_connections=False
        )
        # update our stored counters for the next call
        self._last_net_counters    = m.pop('_net_counters')
        self._prev_pernic_counters = m.pop('_pernic_counters')
        self._iface_speeds         = m.pop('_iface_speeds')
        return m

    def shutdown(self):
        """Call on clean exit."""
        self.snap_mgr.shutdown()
        self._log_pool.shutdown(wait=True)

if __name__ == "__main__":
    # 1) Parse CLI, override resources, spawn controllers
    args        = parse_args()
    config      = ConfigManager()
    db_path     = build_db_path(args)

    if args.ram_usage is not None:
        config.set("target_ram_usage_pct", args.ram_usage)
    if args.cpu_usage is not None:
        config.set("target_cpu_usage_pct", args.cpu_usage)

    controllers = spawn_resource_controllers(
        cpu_pct=args.cpu_usage,
        ram_pct=args.ram_usage
    )

    # 2) Load or initialize Critic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = Critic(input_dim=5, hidden=32).to(device)
    try:
        ckpt = torch.load("critic_pretrained.pt", map_location=device)
        critic.load_state_dict(ckpt)
        logger.info("[MAIN] Loaded pretrained Critic")
    except FileNotFoundError:
        logger.info("[MAIN] No pretrained critic found; using fresh init")
    critic.eval()

    # 3) SafeRotatingFile Handler Logging
    configure_logging()

    # 4) Instantiate a single Embryo and start fresh
    embryo = Embryo(
        config=config,
        db_path=db_path,
        goal_mode_override=args.goal_mode_override,
        disable_snapshots=args.training,
        critic=critic,
        launch_args=args
    )
    embryo.mode = args.mode
    logger.info(f"[START] new run, mode={embryo.mode}")

    # 5) Heartbeat loop
    max_beats = args.beats
    next_beat = time.time() + embryo.heartbeat_interval

    try:
        while embryo.hb.count < max_beats:
            if time.time() >= next_beat:
                embryo.think()
                next_beat += embryo.heartbeat_interval
    except KeyboardInterrupt:
        logger.info("[EXIT] interrupted—shutting down.")
        for c in controllers:
            c.stop()
        embryo.resource_manager.stop()
        sys.exit(0)

    logger.info(f"[EXIT] reached {max_beats} beats, shutting down.")
    for c in controllers:
        c.stop()
    embryo.resource_manager.stop()
    sys.exit(0)
