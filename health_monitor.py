import time
import logging
import psutil
from health import SystemHealth, Survival
from settings import SAMPLE_INTERVAL

class HealthMonitor:
    def __init__(
        self,
        sample_interval: float = SAMPLE_INTERVAL,
        logger: logging.Logger = None
    ):
        self.sample_interval = sample_interval
        self.logger = logger or logging.getLogger(__name__)
        self._sample_count = 0
        # choose how often to do a full socket scan:
        self._conn_every = max(1, int(60.0 / sample_interval))
        # seed previous overall counters
        self._last_net_counters   = psutil.net_io_counters()
        # seed previous per-interface counters
        self._prev_iface_counters = psutil.net_io_counters(pernic=True)
        # cache interface speeds once
        stats = psutil.net_if_stats()
        self._iface_speeds = {
            nic: getattr(s, 'speed', 0) or 0
            for nic, s in stats.items()
        }
        self._running = False

    def _collect_metrics(self) -> dict:
        # on every call, bump count
        self._sample_count += 1
        # only request a fresh scan every _conn_every samples:
        do_conn = (self._sample_count % self._conn_every) == 0

        # store previous net counters for scoring
        prev_net = self._last_net_counters

        m = SystemHealth.check(
            prev_net=prev_net,
            prev_pernic=self._prev_iface_counters,
            iface_speeds=self._iface_speeds,
            compute_connections=do_conn
        )

        # extract new raw counters
        new_net = m.pop('_net_counters')
        self._last_net_counters = new_net

        # update per-interface data
        self._prev_iface_counters = m.pop('_pernic_counters')
        self._iface_speeds        = m.pop('_iface_speeds')

        # inject raw counters into metrics for Survival.score
        m['_prev_net']     = prev_net
        m['_net_counters'] = new_net

        return m

    def start(self):
        self._running = True
        self.logger.info(f"HealthMonitor started, collecting every {self.sample_interval}s")
        while self._running:
            try:
                metrics = self._collect_metrics()
                score = Survival.score(metrics)
                self.logger.info(f"Survival composite score: {score['composite']}")
            except Exception as e:
                self.logger.error(f"Error during health monitoring: {e}")
            time.sleep(self.sample_interval)

    def stop(self):
        self._running = False
        self.logger.info("HealthMonitor stopped")

    def get_snapshot(self) -> dict:
        try:
            metrics = self._collect_metrics()
            score = Survival.score(metrics)
            return {'metrics': metrics, 'score': score}
        except Exception as e:
            self.logger.error(f"Error taking health snapshot: {e}")
            return {}
