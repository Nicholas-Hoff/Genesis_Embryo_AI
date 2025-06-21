# crash_tracker.py — Tracks fatal events for learning

import os
import json
import time
import logging
from pathlib import Path
from colorama import Fore, Style

logger = logging.getLogger(__name__)

_CACHE_FILE = os.path.expanduser("~/.embryo_crash_log.json")

class CrashTracker:
    def __init__(self):
        self.crashes = []
        self._load()

    def _load(self):
        if not os.path.exists(_CACHE_FILE):
            self.crashes = []
            return

        try:
            with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                self.crashes = json.load(f)
        except json.JSONDecodeError as e:
            # Corrupted JSON: back it up and start fresh
            backup = _CACHE_FILE + ".bak"
            os.replace(_CACHE_FILE, backup)
            logger.warning(f"[CRASH TRACKER] Corrupted log; backed up to {backup}")
            self.crashes = []
            # write a new empty log
            try:
                with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[CRASH TRACKER] Failed to load crash log: {e}")
            self.crashes = []

    def _save(self):
        try:
            with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.crashes, f, indent=2)
        except Exception as e:
            logger.error(f"[CRASH TRACKER] Failed to save crash log: {e}")

    def record_crash(self, goal: str, phase: str, context: dict = None):
        event = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "goal": goal,
            "phase": phase,
            "context": context or {}
        }
        self.crashes.append(event)
        self._save()
        logger.error(f"[CRASH] Logged fatal event: {goal} during {phase} — {event['context']}")

    def log_crash(self, context: dict):
        goal  = context.get("goal",  "unknown")
        phase = context.get("phase", "unknown")
        self.record_crash(goal, phase, context)

    def recent_crashes(self, limit: int = 5):
        return self.crashes[-limit:]

    def recent_crashes_for_goal(self, goal: str, phase: str = None, limit: int = 10):
        return [
            c for c in reversed(self.crashes)
            if c.get("goal") == goal and (phase is None or c.get("phase") == phase)
        ][:limit]

    def crash_count(self):
        return len(self.crashes)

    def clear(self):
        self.crashes = []
        self._save()
        logger.info("[CRASH TRACKER] Cleared crash history")

    # ─── NEW: Export to JSON ────────────────────────────────────────────
    def to_json(self) -> str:
        """
        Convert the entire crash list to a JSON string.
        """
        return json.dumps(self.crashes)

    # ─── NEW: Reconstruct from JSON ─────────────────────────────────────
    @classmethod
    def from_json(cls, json_str: str) -> "CrashTracker":
        # create fresh and bypass __init__
        inst = cls.__new__(cls)
        # set up the path and load current on‐disk cursors
        inst.__dict__.update({
            'crashes': json.loads(json_str),
            '_CACHE_FILE': _CACHE_FILE,
            'paths': [Path(_CACHE_FILE)],
            'cursors': { Path(_CACHE_FILE): 0 },
            'logger': logging.getLogger(__name__)
        })
        # optionally write out the new crash list immediately
        inst._save()
        return inst

