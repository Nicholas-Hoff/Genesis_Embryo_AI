import pickle
import tempfile
import json
import torch
import datetime
import duckdb
import gc, queue, os, logging
import psutil
from multiprocessing import Process
from typing import Any, List, Tuple, Dict, Optional
from colorama import Fore, Style
from memory_optimizer import EfficientStateManager
from threading import Thread
from mutation import MutationEngine

# ——— Helpers —————————————————————————————————————————————————

def _now() -> str:
    """ISO-formatted current timestamp for logging or embeds."""
    return datetime.datetime.now().isoformat()

def _placeholders(count: int) -> str:
    """Generate a comma-separated list of '?' placeholders of length `count`."""
    return ", ".join("?" for _ in range(count))

def _make_recorder(table: str):
    """Return a function that inserts into `table`."""
    def record_fn(self, *args):
        self.insert(table, *args)
    return record_fn


# ——— Memory Archive —————————————————————————————————————————————

class MemoryArchive:
    def __init__(self, path: str = "godseed_memory.log") -> None:
        self.path = path

    def log(self, res: str, desc: str) -> None:
        """Append a result + description to the rolling memory log."""
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(f"{_now()} | {res} | {desc}\n")
        except Exception as e:
            logging.error(f"[MEMORY LOG ERROR] {e}")

    def to_json(self) -> str:
        """
        Serialize this MemoryArchive as JSON.  We only need to store its path.
        """
        return json.dumps({"path": self.path})

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryArchive":
        """
        Reconstruct a MemoryArchive from its JSON string.
        """
        d = json.loads(json_str)
        return cls(path=d["path"])

# ——— MemoryDB —————————————————————————————————————————————————————

class MemoryDB:
    # Centralized DDL for all tables
    TABLE_SCHEMAS: Dict[str, str] = {
        "heartbeats": (
            "ts TIMESTAMP, heartbeat INT, survival_score DOUBLE"
        ),
        "mutations": (
            "ts TIMESTAMP, result TEXT, description TEXT, survival_change DOUBLE"
        ),
        "mutation_metrics": (
            "ts TIMESTAMP, target_rate DOUBLE, observed_rate DOUBLE"
        ),
        "fatal_events": (
            "ts TIMESTAMP, description TEXT"
        ),
        "mutation_contexts":  (
            "ts TIMESTAMP, "
            "param TEXT, strategy TEXT, "
            "old_value DOUBLE, new_value DOUBLE, "
            "survival_before DOUBLE, survival_after DOUBLE, "
            "survival_change DOUBLE, "
            "cpu DOUBLE, mem DOUBLE, disk DOUBLE, "
            "network DOUBLE"
        ),
        "mutation_episodes": (
            "ts TIMESTAMP, episode_id TEXT, "
            "strategies_applied INT, parameters_changed INT, "
            "survival_before DOUBLE, survival_after DOUBLE, survival_change DOUBLE"
        ),
        "reflections": (
            "ts TIMESTAMP, heartbeat INT, mode TEXT, gene_count INT, "
            "heartbeat_interval DOUBLE, survival_threshold DOUBLE, "
            "recent_survival DOUBLE, trend TEXT, "
            "most_used_strategy TEXT, mutation_rate DOUBLE"
        ),
        "survival_details": (
            "ts TIMESTAMP, "
            "heartbeat INT, "
            "gene_count INT, "
            "cpu DOUBLE, "
            "memory DOUBLE, "
            "disk DOUBLE, "
            "network DOUBLE, "
            "composite DOUBLE, "
            "cpu_pct DOUBLE, "
            "disk_io DOUBLE"
        ),
        "transitions": (
            "ts TIMESTAMP, hb INT, "
            "survival_before DOUBLE, novelty_before DOUBLE, efficiency_before DOUBLE, "
            "mutation_error_before DOUBLE, cycle_before DOUBLE, action TEXT, goal TEXT, reward DOUBLE, "
            "survival_after DOUBLE, novelty_after DOUBLE, efficiency_after DOUBLE, mutation_error_after DOUBLE, cycle_after DOUBLE"
        ),
        "reflections": (
            "ts TIMESTAMP, hb INT, mode TEXT, gene_count INT, hb_interval DOUBLE,"
            " survival_threshold DOUBLE, recent_survival DOUBLE, trend TEXT,"
            " strategy TEXT, mutation_rate DOUBLE"
        ),
    }
    def fetch_for_episode(self, hb_index: int) -> torch.Tensor:
        # pull all transitions from the last mutation cycle
        rows = self.conn.execute(
            "SELECT survival_before, novelty_before, efficiency_before, "
            "mutation_error_before, cycle_before FROM transitions "
            "WHERE hb = ?", [hb_index]
        ).fetchall()
        # convert to a Tensor of shape [seq_len, 1, state_dim]
        data = torch.tensor(rows, dtype=torch.float32).unsqueeze(1)
        return data

    def __init__(self, path: str = "godseed_memory.db") -> None:
        # Connect to DuckDB and enforce memory limits
        self.conn = duckdb.connect(path)
        self.conn.execute("PRAGMA memory_limit='8GB';")
        self.state_manager = EfficientStateManager(path + ".state")
        self.init_tables()

        # Auto-generate simple record_* methods for basic tables
        for table in ("heartbeats", "mutations", "mutation_metrics", "fatal_events"):
            method_name = f"record_{table[:-1]}"
            setattr(self, method_name, _make_recorder(table).__get__(self, MemoryDB))

    def init_tables(self) -> None:
        """Create any missing tables and columns based on TABLE_SCHEMAS."""
        for table, schema in self.TABLE_SCHEMAS.items():
            info = self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name=?",
                [table],
            ).fetchall()
            if not info:
                self.conn.execute(f"CREATE TABLE {table} ({schema})")
                continue

            existing_cols = {row[0] for row in info}
            for col_def in schema.split(','):
                col_def = col_def.strip()
                col_name = col_def.split()[0]
                if col_name not in existing_cols:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

    def insert(self, table: str, *values: Any) -> None:
        """Insert a single row into `table` with NOW() timestamp + provided values."""
        ph = _placeholders(len(values))
        sql = f"INSERT INTO {table} VALUES (now(), {ph})"
        self.conn.execute(sql, values)

    def insert_many(self, table: str, rows: List[Tuple[Any, ...]]) -> None:
        """Batch-insert multiple rows into `table` in one call."""
        if not rows:
            return
        ph = _placeholders(len(rows[0]))
        sql = f"INSERT INTO {table} VALUES (now(), {ph})"
        # duckdb will autocommit for us, so just call executemany()
        self.conn.executemany(sql, rows)

    # Pre-defined record methods:
    def record_heartbeat(self, cnt: int, score: float) -> None:
        self.insert("heartbeats", cnt, score)

    def record_mutation(self, res: str, desc: str, chg: float) -> None:
        self.insert("mutations", res, desc, chg)

    def record_metric(self, target: float, observed: float) -> None:
        self.insert("mutation_metrics", target, observed)

    def record_fatal(self, desc: str) -> None:
        self.insert("fatal_events", desc)

    def record_mutation_context(
        self,
        param: str,
        strategy: str,
        old: float,
        new: float,
        before: float,
        after: float,
        cpu: float,
        mem: float,
        disk: float,
        network: float
    ) -> None:
        change = round(after - before, 4)
        self.insert(
            "mutation_contexts",
            param, strategy, old, new,
            before, after, change,
            cpu, mem, disk,
            network
        )

    def record_mutation_episode(
        self,
        ep_id: str,
        strat_count: int,
        ctx_count: int,
        before: Dict[str, float],
        after: Dict[str, float]
    ) -> None:
        change = round(after["composite"] - before["composite"], 4)
        self.insert(
            "mutation_episodes",
            ep_id, strat_count, ctx_count,
            before["composite"], after["composite"], change
        )

    def record_reflection(
        self,
        hb: int,
        mode: str,
        gene_count: int,
        hb_int: float,
        surv_th: float,
        recent: float,
        trend: str,
        strat: str,
        m_rate: float
    ) -> None:
        self.insert(
            "reflections",
            hb, mode, gene_count, hb_int,
            surv_th, recent, trend, strat, m_rate
        )

    def record_survival_detail(
        self,
        cnt: int,
        gene_count: int,
        cpu: float,
        memory: float,
        disk: float,
        network: float,
        composite: float,
        cpu_pct: float,
        disk_io: float
    ) -> None:
        self.insert(
            "survival_details",
            cnt,
            gene_count,
            cpu,
            memory,
            disk,
            network,
            composite,
            cpu_pct,
            disk_io
        )

    def record_transition(
        self,
        hb: int,
        state_b: List[float],
        action: str,
        goal: str,
        reward: float,
        state_a: List[float]
    ) -> None:
        self.insert(
            "transitions",
            hb,
            *state_b,
            action,
            goal,
            reward,
            *state_a
        )

class SnapshotManager:
    """
    Handles periodic exports of each DuckDB table to Parquet,
    using a background thread.  Updated to snapshot multiple DBs.
    """
    def __init__(self, db_paths: List[str], snap_dir: str = "snapshots"):
        # db_paths: list of full paths to .db files (state DB, memory DB, etc.)
        self.db_paths    = db_paths
        self.snap_dir    = snap_dir
        os.makedirs(self.snap_dir, exist_ok=True)

        # All Parquet exports will go here:
        self.parquet_dir = os.path.join(self.snap_dir, "parquet_export")
        os.makedirs(self.parquet_dir, exist_ok=True)

        self._q = queue.Queue()
        Thread(target=self._worker, daemon=True).start()

    def export_snapshot(self, export_dir: str = None) -> None:
        """
        Enqueue a snapshot.  If export_dir is omitted, uses self.parquet_dir.
        """
        if export_dir is None:
            export_dir = self.parquet_dir
        self._q.put(export_dir)

    def _worker(self):
        """
        Background thread: for each enqueued export_dir,
        iterate over each DB in self.db_paths and dump all tables.
        """
        while True:
            export_dir = self._q.get()
            if export_dir is None:
                # shutdown signal
                self._q.task_done()
                break

            # For each DB, make a subfolder named after the DB file (without extension)
            for db_path in self.db_paths:
                db_name       = os.path.splitext(os.path.basename(db_path))[0]
                db_export_dir = os.path.join(export_dir, db_name)
                os.makedirs(db_export_dir, exist_ok=True)

                try:
                    conn = duckdb.connect(db_path)
                    conn.execute("PRAGMA memory_limit='8GB';")
                    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
                    for tbl in tables:
                        out_path = os.path.join(db_export_dir, f"{tbl}.parquet")
                        conn.execute(
                            f"COPY (SELECT * FROM {tbl}) TO '{out_path}' (FORMAT PARQUET)"
                        )
                    conn.close()
                except Exception as e:
                    logging.error(f"[SNAPSHOT WORKER] export for {db_path} → {db_export_dir} failed: {e}")

            gc.collect()
            logging.info(f"[SNAPSHOT] export to {export_dir} done")
            self._q.task_done()

    def shutdown(self):
        """
        Signal the worker to stop, and wait for it.
        """
        self._q.put(None)
        self._q.join()


class DuckdbStateIO:
    """
    Write the full snapshot into DuckDB PLUS let
    EfficientStateManager track small‐delta checkpoints in memory.

    We no longer use pickle for 'mutator' or 'CrashTracker'.
    Instead, both are stored as JSON in 'mutator_state' and 'crash_tracker_state'.
    """

    def __init__(self, state_manager: Any, db_path: str):
        """
        state_manager: an instance of EfficientStateManager or similar,
                       which knows how to handle incremental checkpoints.
        db_path: path to the DuckDB file (e.g. "embryo.db").
        """
        self.state_manager = state_manager
        self.db_path = db_path

        # Ensure the same table as before, but replace the two BLOB columns with JSON
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embryo_state (
                run_id                   VARCHAR PRIMARY KEY,
                mode                     VARCHAR,
                current_goal             VARCHAR,
                hb_count                 INTEGER,
                cur_count                INTEGER,
                mutation_interval        INTEGER,
                heartbeat_interval       INTEGER,
                survival_threshold       DOUBLE,
                rollback_required        BOOLEAN,
                dynamic_strategy_prob    DOUBLE,
                gene_count               INTEGER,
                gene_min                 INTEGER,
                gene_max                 INTEGER,
                strategy_history         JSON,
                synth_strategy_stats     JSON,
                bad_cycles               INTEGER,
                archive_state            JSON,   -- changed from BLOB to JSON
                score_stats_state        JSON,   -- changed from BLOB to JSON
                metrics                  JSON,
                param_bounds             JSON,
                task_params              JSON,
                behavior_archive         JSON,
                novelty_k                INTEGER,
                min_archive_size         INTEGER,
                mem                      JSON,
                db_path_str              VARCHAR,
                mutator_state            JSON,   -- new: store mutator as JSON
                crash_tracker_state      JSON    -- new: store crash tracker as JSON
            )
        """)
        conn.close()

    def save(self, embryo: Any) -> None:
        """
        Exactly same DuckDB save logic as before, but:
          • Instead of pickling `archive` and `score_stats`, we JSON‐serialize them.
          • Instead of pickling `mutator` and `CrashTracker`, we JSON‐serialize them.
        """
        # 1) Get the raw state dict
        state = embryo.__getstate__()

        # 2) Tell the state_manager to record incremental changes
        self.state_manager.save_state(state)

        # 3) Convert sub‐objects to JSON strings
        try:
            archive_json = state['archive'].to_json()
        except AttributeError:
            raise RuntimeError("Archive must implement a to_json() method returning valid JSON.")

        try:
            score_stats_json = state['score_stats'].to_json()
        except AttributeError:
            raise RuntimeError("RollingStats must implement a to_json() method returning valid JSON.")

        metrics_json       = json.dumps(state['metrics'])
        param_bounds_json  = json.dumps(state['param_bounds'])
        task_params_json   = json.dumps(state['task_params'])
        behavior_json      = json.dumps(state['behavior_archive'])
        mem_json           = state['mem'].to_json()

        # ─── NEW: serialize 'mutator' and 'CrashTracker' to JSON ───
        try:
            mutator_json = state['mutator'].to_json()
        except AttributeError:
            raise RuntimeError("Mutator must implement a to_json() method returning valid JSON.")

        try:
            crash_tracker_json = state['CrashTracker'].to_json()
        except AttributeError:
            raise RuntimeError("CrashTracker must implement a to_json() method returning valid JSON.")

        # 4) Upsert into DuckDB (now with exactly 28 placeholders)
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO embryo_state VALUES (
                ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,
                ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?
            )
        """, (
            embryo.run_id,                         # run_id                      → 1
            state['mode'],                         # mode                        → 2
            state['current_goal'] or "",           # current_goal                → 3
            state['hb'].count,                     # hb_count                    → 4
            state['cur'].count,                    # cur_count                   → 5
            state['mutation_interval'],            # mutation_interval           → 6
            state['heartbeat_interval'],           # heartbeat_interval          → 7
            state['survival_threshold'],           # survival_threshold          → 8
            state['rollback_required'],            # rollback_required           → 9
            state['dynamic_strategy_prob'],        # dynamic_strategy_prob       → 10
            state['gene_count'],                   # gene_count                  → 11
            state['gene_min'],                     # gene_min                    → 12
            state['gene_max'],                     # gene_max                    → 13
            json.dumps(state['strategy_history']), # strategy_history            → 14
            json.dumps(state.get('synth_strategy_stats', {})), # synth_strategy_stats → 15
            state['bad_cycles'],                   # bad_cycles                  → 16
            archive_json,                          # archive_state (JSON)        → 17
            score_stats_json,                      # score_stats_state (JSON)    → 18
            metrics_json,                          # metrics (JSON)              → 19
            param_bounds_json,                     # param_bounds (JSON)         → 20
            task_params_json,                      # task_params (JSON)          → 21
            behavior_json,                         # behavior_archive (JSON)     → 22
            state['novelty_k'],                    # novelty_k                   → 23
            state['min_archive_size'],             # min_archive_size            → 24
            mem_json,                              # mem (JSON)                  → 25
            state['db_path'],                      # db_path_str                 → 26
            mutator_json,                          # mutator_state (JSON)        → 27
            crash_tracker_json                     # crash_tracker_state (JSON)  → 28
        ))
        conn.close()


    def load(self, run_id: str) -> dict:
        """
        Fetch the row from DuckDB, parse each JSON field, and reconstruct
        'mutator' and 'CrashTracker' from their JSON blobs.
        """
        conn = duckdb.connect(self.db_path)
        row = conn.execute("""
            SELECT
                mode, current_goal, hb_count, cur_count,
                mutation_interval, heartbeat_interval,
                survival_threshold, rollback_required,
                dynamic_strategy_prob, gene_count, gene_min,
                gene_max, strategy_history, synth_strategy_stats,
                bad_cycles, archive_state, score_stats_state,
                metrics, param_bounds, task_params,
                behavior_archive, novelty_k, min_archive_size,
                mem, db_path_str, mutator_state, crash_tracker_state
            FROM embryo_state
            WHERE run_id = ?
        """, (run_id,)).fetchone()
        conn.close()

        if row is None:
            raise RuntimeError(f"No saved state found for run_id='{run_id}'")

        (
            mode, current_goal, hb_count, cur_count,
            mutation_interval, heartbeat_interval,
            survival_threshold, rollback_required,
            dynamic_strategy_prob, gene_count, gene_min,
            gene_max, strategy_history_json, synth_stats_json,
            bad_cycles, archive_json, score_stats_json,
            metrics_json, param_bounds_json, task_params_json,
            behavior_json, novelty_k, min_archive_size,
            mem_json, db_path_str, mutator_json, crash_tracker_json
        ) = row

        # Re‐build the raw state dict:
        state = {
            'mode':                   mode,
            'current_goal':           None if current_goal == "" else current_goal,
            'hb':                     type('HB', (), {'count': hb_count})(),
            'cur':                    type('CUR', (), {'count': cur_count})(),
            'mutation_interval':      mutation_interval,
            'heartbeat_interval':     heartbeat_interval,
            'survival_threshold':     survival_threshold,
            'rollback_required':      rollback_required,
            'dynamic_strategy_prob':  dynamic_strategy_prob,
            'gene_count':             gene_count,
            'gene_min':               gene_min,
            'gene_max':               gene_max,
            'strategy_history':       json.loads(strategy_history_json),
            'synth_strategy_stats':   json.loads(synth_stats_json),
            'bad_cycles':             bad_cycles,
            'archive':                json.loads(archive_json),
            'score_stats':            json.loads(score_stats_json),
            'metrics':                json.loads(metrics_json),
            'param_bounds':           json.loads(param_bounds_json),
            'task_params':            json.loads(task_params_json),
            'behavior_archive':       json.loads(behavior_json),
            'novelty_k':              novelty_k,
            'min_archive_size':       min_archive_size,
            'mem':                    MemoryArchive.from_json(mem_json),
            'db_path':                db_path_str,
        }

        # ───── Reconstruct the Mutator object from its JSON ─────────
        try:
            state['mutator'] = MutationEngine.from_json(mutator_json)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize mutator: {e}")

        # ─── Reconstruct the CrashTracker from its JSON ───────────────
        try:
            state['CrashTracker'] = CrashTracker.from_json(crash_tracker_json)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize CrashTracker: {e}")

        return state

