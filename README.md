# Genesis Embryo

An autonomous, evolutionary AI framework that continuously adapts and improves itself—much like a living organism navigating abstract goals, selecting mutations, and reflecting on outcomes. It combines reinforcement learning, genetic algorithms, procedural task generation, and self-modifying code to forge a self-directed intelligence.

## 🚀 Key Features

### Embryo Core (`genesis_embryo_core.py`)
- Orchestrates the heartbeat loop and evolution cycles via `run_heartbeat` and `mutation_cycle`.
- `parse_args` exposes command line options for mode selection and resource limits.
- Maintains persistence and world-model planning through `ConfigManager` and `WorldModelTrainer`.
- Self-mutates via AST-level strategies and meta-strategy generation.
- Computes multi-objective survival scores across CPU, memory, disk, and network metrics.

### Health Monitoring (`health.py` & `health_monitor.py`)
- `SystemHealth.check` gathers CPU, memory, disk and network metrics via `psutil`.
- `Survival.score` converts raw metrics into sub-scores for each resource.
- `HealthMonitor` periodically logs metrics and calibrates thresholds.

### Resource Controllers (`resources.py`)
- Includes CPU throttling and RAM-burning daemons for procedural stress tests.
- `spawn_resource_controllers` launches these daemons when resource usage exceeds configured targets.
- `DynamicResourceManager` monitors the process tree, adjusts priorities, and can terminate low‑priority tasks when memory pressure rises.

### Mutation Engine (`mutation.py`, `weight_manager.py`, `meta_strategy_engine.py`, `strategy.py`)
- Implements mutation strategies such as `gaussian`, `creep`, `random_uniform`, `explore`, and `reset`.
- `MutationEngine.pick_strategy` selects a strategy based on meta weights that update after each cycle.
- `Archive` preserves top‑performing states and can reseed the embryo from champions.
- Meta‑strategies mutate the strategies themselves using AST crossover to invent new heuristics.

### Dynamic Skill Synthesis (`skill_synthesizer.py`)
- `SkillSynthesizer` uses RedBaron AST manipulation to generate new Python modules on the fly.
- These modules expand the embryo’s action space and are registered as additional strategies.

### Memory & Replay Buffers (`prioritized_replay.py`, `memory_optimizer.py`)
- `PrioritizedReplay` ranks transitions by temporal-difference error for efficient critic updates.
- `EfficientStateManager` in `MemoryOptimizer` compresses checkpoints and can recompute intermediate states on demand.

### Environment Awareness (`env_scanner.py`)
- `scan_environment` inspects the host OS, virtualization layers, and container hints.
- This information can inform migration strategies or adaptive configuration.

### Emotional Drives (`emotional_drives.py`)
- Maintains running measures of curiosity and novelty for the embryo.
- These drives feed into the overall survival score to encourage exploration.

### World Model (`world_model.py`)
- Implements a lightweight MLP that predicts state deltas given an action embedding.
- `WorldModelTrainer.train_step` performs mixed‑precision optimization with optional gradient clipping.

### Meta-Cognition (`planner.py`)
- Contains a lightweight Monte Carlo planner that rolls out potential future mutations using the world model.
- Helps evaluate which strategy choices may lead to higher survival before actually applying them.

### Memory Abstraction (`memory_abstraction.py`)
- Provides an `EpisodeSummarizer` transformer that compresses sequences of mutation events into fixed-size embeddings.
- These embeddings can be stored in the database and used for later analysis or training.

### Persistence Layer (`persistence.py`, `merge_duckdb.py`, `pretrain_critic.py`)
- `MemoryDB` uses DuckDB to store metrics, transitions, and reflection logs.
- `SnapshotManager` can export state to JSON or Parquet for offline analysis.
- `merge_duckdb.py` merges multiple runs, and `pretrain_critic.py` trains the critic network from stored transitions.

### Mission Control UI (`genesis_monitor.py`)
- Dash + Plotly dashboard for live visualization of heartbeat metrics, survival scores, and mutation analytics.
- Provides controls to start, stop, and inspect ongoing runs.

### External Feedback (`feedback_hooks.py`)
- Parses log files for errors or denials and converts them into reinforcement signals for the mutation engine.

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/genesis-embryo.git
cd genesis-embryo
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Optional) GPU support: ensure you have CUDA-enabled PyTorch installed if you intend to run on GPU.

## ⚙️ Configuration

`settings.py` – global tunables for I/O calibration, sampling intervals, and max throughput.

`config.json` – auto-generated by `ConfigManager` on first run. It contains all hyperparameters:

- Heartbeat/mutation intervals
- Survival thresholds
- Gene-count bounds
- Mutation-rate schedules
- Procedural task defaults

Modify `config.json` directly or via the CLI flags (see Usage below).

## 🎯 Usage Examples

Run the Embryo:

```bash
python genesis_embryo_core.py \
  --mode aggressive \
  --goal_mode full \
  --beats 10000 \
  --cpu-usage 80 \
  --ram-usage 90
```

- `--mode`: `stabilize` or `aggressive`
- `--goal_mode`: `shadow` (only Q-updates) or `full` (apply on-goal changes)
- `--beats`: number of heartbeats to execute
- `--cpu-usage` / `--ram-usage`: target resource stress levels

Pretrain the critic network (requires PyTorch):

```bash
python pretrain_critic.py
```

Merge multiple DuckDB files:

```bash
python merge_duckdb.py
```

Launch the Mission Control dashboard:

```bash
python genesis_monitor.py
```

## 📁 Project Structure
```
.
├── README.md
├── requirements.txt
├── config.json          # generated on first run
├── settings.py
├── genesis_embryo_core.py
├── health.py
├── health_monitor.py
├── resources.py
│   └── DynamicResourceManager class for adaptive process control
├── mutation.py
├── meta_strategy_engine.py
├── strategy.py
├── weight_manager.py
├── world_model.py
├── prioritized_replay.py
├── memory_optimizer.py
├── persistence.py
├── merge_duckdb.py
├── pretrain_critic.py
├── genesis_monitor.py
├── procedural_tasks.py
├── goals.py
├── logging_config.py
└── crash_tracker.py
```

## 📜 Requirements

The core dependencies are listed in `requirements.txt`:

- psutil
- numpy
- torch
- duckdb
- pandas
- dash
- dash-daq
- plotly
- scikit-learn
- colorama
- pydantic
- astor

(See `requirements.txt` for exact versions.)

## 👍 Contributing

Contributions, bug reports, and feature requests are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add X feature"`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 🧪 Running Tests

Install the dependencies and then execute the test suite:

```bash
scripts/setup_tests.sh
pytest
```

Tests that rely on PyTorch are skipped automatically if the library is missing.

Genesis Embryo is still in its infancy—every heartbeat and mutation brings us closer to a truly adaptive, evolving intelligence. Enjoy exploring its ever-growing capabilities!
