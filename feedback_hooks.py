import logging
from pathlib import Path

class FeedbackHooks:
    """Monitor external signals like log files for feedback events."""
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.cursors = {p: 0 for p in self.paths}
        self.logger = logging.getLogger(__name__)

    def poll(self) -> int:
        """Return count of negative feedback signals detected."""
        negatives = 0
        for p in self.paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(self.cursors[p])
                for line in f:
                    if any(word in line.lower() for word in ["error", "fail", "denied"]):
                        negatives += 1
                self.cursors[p] = f.tell()
        if negatives:
            self.logger.info(f"[FEEDBACK] Detected {negatives} negative signals")
        return negatives
