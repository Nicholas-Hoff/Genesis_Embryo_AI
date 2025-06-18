# resources.py

import threading
import time
import os
import signal
import logging
import json
import psutil
from typing import Optional, Dict, List, Set

class CPUThrottler:
    __slots__ = (
        '_limit', '_interval', '_stop_evt', '_last_reset', 
        '_accum_busy', '_prev_cpu', '_thread'
    )

    def __init__(self, limit_pct: float, interval: float = 1.0) -> None:
        self._limit      = max(0.0, min(limit_pct, 100.0))
        self._interval   = interval
        self._stop_evt   = threading.Event()
        self._last_reset = time.perf_counter()
        self._accum_busy = 0.0

        # initialize prev_cpu so first delta is zero
        proc = psutil.Process()
        self._prev_cpu = sum(proc.cpu_times()[:2])

        # keep a reference so we can join later if we want
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        proc = psutil.Process()
        while not self._stop_evt.is_set():
            now     = time.perf_counter()
            elapsed = now - self._last_reset

            if elapsed >= self._interval:
                self._last_reset = now
                self._accum_busy = 0.0
                elapsed = 0.0

            busy_target = (self._limit / 100.0) * self._interval

            # measure CPU time since last loop
            used_cpu   = sum(proc.cpu_times()[:2])
            delta_busy = used_cpu - self._prev_cpu
            self._prev_cpu = used_cpu

            to_burn = busy_target - delta_busy
            if to_burn > 0:
                # split spin/sleep to approximate duty-cycle
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < to_burn * 0.5:
                    pass
                time.sleep(to_burn * 0.5)
            else:
                # over budget: sleep until next window
                time.sleep(self._interval - elapsed)


class RAMBurner:
    __slots__ = ('_limit', '_chunk', '_interval', '_buffer', '_stop_evt', '_thread')

    def __init__(self, limit_pct: float, chunk_mb: int = 10, interval: float = 1.0) -> None:
        self._limit    = max(0.0, min(limit_pct, 100.0))
        self._chunk    = chunk_mb * 1024 * 1024
        self._interval = interval
        self._buffer   = []
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        vm = psutil.virtual_memory
        while not self._stop_evt.is_set():
            used = vm().percent
            if used < self._limit:
                try:
                    self._buffer.append(bytearray(self._chunk))
                except MemoryError:
                    time.sleep(self._interval)
            else:
                time.sleep(self._interval)


def spawn_resource_controllers(
    cpu_pct: Optional[float], 
    ram_pct: Optional[float]
) -> list:
    """
    Start CPU- and/or RAM- controllers as daemons.
    Returns controller objects, so you can .stop() them later.
    """
    controllers = []
    if cpu_pct:
        cpu = CPUThrottler(limit_pct=cpu_pct)
        cpu.start()
        controllers.append(cpu)
    if ram_pct:
        ram = RAMBurner(limit_pct=ram_pct)
        ram.start()
        controllers.append(ram)
    return controllers


class DynamicResourceManager:
    """Dynamic CPU/Memory manager with crash learning."""

    def __init__(
        self,
        crash_tracker=None,
        sample_interval: float = 1.0,
        target_cpu_pct: float = 80.0,
        target_mem_pct: float = 90.0,
    ) -> None:
        self.crash_tracker = crash_tracker
        self.sample_interval = sample_interval
        self.target_cpu = target_cpu_pct
        self.target_mem = target_mem_pct
        self.whitelist: Set[int] = {os.getpid()}
        self.blacklist: Set[str] = set()
        self.priorities: Dict[int, str] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

    # ─── Process Metrics ──────────────────────────────────────────
    def _collect(self) -> Dict[int, Dict[str, float]]:
        procs = {}
        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'nice', 'cmdline']):
            try:
                procs[p.pid] = p.info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return procs

    def _categorize(self, info: Dict[str, float]) -> str:
        if info['pid'] in self.whitelist:
            return 'critical'
        mem = info.get('memory_percent', 0.0)
        if mem > 10:
            return 'high'
        elif mem > 5:
            return 'medium'
        return 'low'

    # ─── Priority & Affinity Adjustments ──────────────────────────
    def _adjust(self, procs: Dict[int, Dict[str, float]]) -> None:
        total_cores = list(range(psutil.cpu_count(logical=True)))
        win_platform = os.name == 'nt'
        changed = 0
        for pid, info in procs.items():
            cat = self._categorize(info)
            self.priorities[pid] = cat
            try:
                proc = psutil.Process(pid)
                name = proc.name()

                if win_platform:
                    mapping = {
                        'critical': psutil.HIGH_PRIORITY_CLASS,
                        'high': psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                        'medium': psutil.NORMAL_PRIORITY_CLASS,
                        'low': psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    }
                else:
                    mapping = {
                        'critical': -5,
                        'high': 0,
                        'medium': 5,
                        'low': 10,
                    }

                desired_nice = mapping.get(cat, mapping['low'])
                old_nice = proc.nice()
                changed_nice = False
                if old_nice != desired_nice:
                    proc.nice(desired_nice)
                    changed_nice = True

                desired_affinity = None
                changed_affinity = False

                if hasattr(proc, 'cpu_affinity'):
                    current_affinity = proc.cpu_affinity()
                    if cat == 'critical':
                        desired_affinity = total_cores
                    elif cat == 'medium' and len(total_cores) > 1:
                        desired_affinity = total_cores[:-1]
                    elif cat == 'low' and len(total_cores) > 1:
                        desired_affinity = [total_cores[-1]]

                    if desired_affinity is not None and current_affinity != desired_affinity:
                        proc.cpu_affinity(desired_affinity)
                        changed_affinity = True

                    # After potential change
                    new_affinity = proc.cpu_affinity()
                else:
                    new_affinity = []

                if changed_nice or changed_affinity:
                    changed += 1
                    self.logger.debug(
                        f"Adjusted PID {pid} ({name}) priority {proc.nice()} affinity {new_affinity}"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self.logger.debug(f"Processed {len(procs)} processes; adjusted {changed}")

    # ─── Safe Termination ────────────────────────────────────────
    def safe_terminate(self, pid: int) -> bool:
        if pid in self.whitelist:
            self.logger.warning("Attempted to terminate whitelisted PID")
            return False
        try:
            proc = psutil.Process(pid)
            name = proc.name()
        except psutil.NoSuchProcess:
            return True
        if name in self.blacklist:
            self.logger.warning("Process is blacklisted, skipping termination")
            return False
        try:
            proc.terminate()
            proc.wait(5)
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        except psutil.TimeoutExpired:
            proc.kill()
            return True

    def _handle_memory_pressure(self, procs: Dict[int, Dict[str, float]]) -> None:
        vm = psutil.virtual_memory()
        if vm.percent <= self.target_mem:
            return
        # sort by memory usage descending
        victims = sorted(procs.items(), key=lambda x: x[1].get('memory_percent', 0.0), reverse=True)
        for pid, info in victims:
            if self.priorities.get(pid) == 'low':
                if self.safe_terminate(pid):
                    self.logger.info(f"Terminated PID {pid} to reclaim memory")
                    if self.crash_tracker:
                        ctx = {"pid": pid, "name": info.get('name'), "action": "terminate"}
                        self.crash_tracker.record_crash("resource", "terminate", ctx)
                if psutil.virtual_memory().percent <= self.target_mem:
                    break

    # ─── Main Loop ───────────────────────────────────────────────
    def _run(self) -> None:
        while self._running:
            procs = self._collect()
            self._adjust(procs)
            self._handle_memory_pressure(procs)
            time.sleep(self.sample_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join()

