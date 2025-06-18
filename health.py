import time
import logging
import psutil
import tempfile
import json
import os
from collections import deque
from colorama import Fore, Style

logger = logging.getLogger(__name__)
from sklearn.preprocessing import MinMaxScaler
import settings
from typing import Dict, Any, Optional

# Constants imported from settings for centralized tuning
INTERVAL_SECONDS = settings.INTERVAL_SECONDS
IO_CALIBRATION_SIZE_MB = settings.IO_CALIBRATION_SIZE_MB

# Initialize MAX_IO_MBPS placeholder
MAX_IO_MBPS = None

# Static helper caches
_static_cache: Dict[str, Any] = {}
_static_keys = {'cpu_count', 'disk_parts'}

# Calibration function for max I/O
def calibrate_max_io() -> float:
    """
    Measure and return the maximum I/O throughput in MB/s.
    """
    data = b"x" * 1024 * 1024 * IO_CALIBRATION_SIZE_MB
    start = time.time()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
    os.unlink(f.name)
    duration = time.time() - start
    return round(IO_CALIBRATION_SIZE_MB / max(duration, 1e-6), 2)

# If running as a script, calibrate once
if __name__ == '__main__':
    from logging_config import configure_logging
    configure_logging()
    MAX_IO_MBPS = calibrate_max_io()
    logger.info(f"[CALIBRATION] MAX_IO_MBPS = {MAX_IO_MBPS:.2f} MB/s")


# def collect_process_metrics() -> dict[int, dict]:
#     # 1) map pid → listening ports
#     pid_ports: dict[int, set[int]] = {}
#     for conn in psutil.net_connections(kind='inet'):
#         if conn.pid and conn.laddr:
#             pid_ports.setdefault(conn.pid, set()).add(conn.laddr.port)

#     # 2) warm up CPU counters
#     for proc in psutil.process_iter():
#         try:
#             proc.cpu_percent(interval=None)
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             pass
#     time.sleep(INTERVAL_SECONDS)

#     # 3) collect metrics with oneshot
#     proc_metrics: dict[int, dict] = {}
#     for proc in psutil.process_iter():
#         try:
#             with proc.oneshot():
#                 pid   = proc.pid
#                 name  = proc.name()
#                 cpu   = proc.cpu_percent(interval=None)
#                 mem   = proc.memory_percent()
#                 dio   = proc.io_counters()
#                 try:
#                     nio = proc.net_io_counters()
#                     net_sent = getattr(nio, 'bytes_sent', 0)
#                     net_recv = getattr(nio, 'bytes_recv', 0)
#                 except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
#                     net_sent = net_recv = 0
#                 ports = sorted(pid_ports.get(pid, []))
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue

#         proc_metrics[pid] = {
#             'name':             name or '<unknown>',
#             'cpu_percent':      cpu,
#             'memory_percent':   mem,
#             'disk_read_bytes':  dio.read_bytes,
#             'disk_write_bytes': dio.write_bytes,
#             'net_bytes_sent':   net_sent,
#             'net_bytes_recv':   net_recv,
#             'ports':            ports,
#         }

#     return proc_metrics


# def _simplify_network(metrics: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
#     net = metrics.get('net_io', None)
#     total_net = {
#         'bytes_sent': getattr(net, 'bytes_sent', 0),
#         'bytes_recv': getattr(net, 'bytes_recv', 0)
#     }

#     # reuse pid_ports mapping for summary
#     pid_ports: dict[int, set[int]] = {}
#     for conn in psutil.net_connections(kind='inet'):
#         if conn.pid and conn.laddr:
#             pid_ports.setdefault(conn.pid, set()).add(conn.laddr.port)

#     proc = metrics.get('proc_metrics', {})
#     lst = []
#     for pid, info in proc.items():
#         sent = info.get('net_bytes_sent', 0)
#         recv = info.get('net_bytes_recv', 0)
#         lst.append({
#             'pid': pid,
#             'name': info.get('name', '<unknown>'),
#             'bytes_sent': sent,
#             'bytes_recv': recv,
#             'total_bytes': sent + recv,
#             'ports': sorted(pid_ports.get(pid, []))
#         })

#     lst.sort(key=lambda x: x['total_bytes'], reverse=True)
#     return {
#         'total_net_io': total_net,
#         'top_process_network_usage': lst[:top_n]
#     }


def _simplify_network_from_conns(conns) -> Dict[str, Any]:
    # Summarize connection states
    state_counts: Dict[str, int] = {}
    for c in conns:
        state_counts[c.status] = state_counts.get(c.status, 0) + 1
    return {'connection_states': state_counts}

class Heartbeat:
    def __init__(self, logger: logging.Logger = None):
        self.count = 0
        self.log = logger or logging.getLogger(__name__)

    def beat(self) -> None:
        self.count += 1
        self.log.debug(f"[HEARTBEAT] #{self.count}")

class SystemHealth:
    """
    Gathers raw system metrics for CPU, memory, disk, and network.
    """
    METRIC_FUNCS = {
        'cpu_times':      lambda: psutil.cpu_times(),
        'cpu_percent':    lambda: psutil.cpu_percent(interval=None),
        'cpu_times_pct':  lambda: psutil.cpu_times_percent(interval=None),
        'cpu_count':      lambda: psutil.cpu_count(logical=True),
        'cpu_stats':      lambda: psutil.cpu_stats(),
        'cpu_freq':       lambda: psutil.cpu_freq(),
        'load_avg':       lambda: psutil.getloadavg(),
        'virtual_mem':    lambda: psutil.virtual_memory(),
        'swap_mem':       lambda: psutil.swap_memory(),
        'disk_parts':     lambda: psutil.disk_partitions(),
        'disk_usage':     lambda: psutil.disk_usage("/"),
        'disk_io':        lambda: psutil.disk_io_counters()
    }

    _last_conn_time: float = 0.0
    _last_conn_summary: Dict[str, Any] = {}
    _conn_interval: float = 60.0  # rescan connections every 60s
    _iface_speeds_cache: Optional[Dict[str, float]] = None

    @classmethod
    def check(
        cls,
        prev_net: Optional[psutil._common.snetio] = None,
        prev_pernic: Optional[Dict[str, psutil._common.snetio]] = None,
        iface_speeds: Optional[Dict[str, float]] = None,
        compute_connections: bool = False
    ) -> Dict[str, Any]:
        """
        Return a dict of raw system metrics plus network deltas and summaries.
        Uses psutil.Process.oneshot() to batch costly process calls.
        """
        # ─── 1) PRIME & BATCH PROCESS METRICS ─────────────────────────
        proc = psutil.Process()
        # prime the internal CPU counter (no-block)
        proc.cpu_percent(None)
        with proc.oneshot():
            cpu = proc.cpu_percent(None)
            memory = proc.memory_percent()
            try:
                io_counters = proc.io_counters()  # namedtuple(read_bytes, write_bytes, ...)
                read_bytes, write_bytes = io_counters.read_bytes, io_counters.write_bytes
            except AttributeError:
                # Some psutil implementations lack io_counters
                read_bytes = write_bytes = 0

        # ─── 2) COLLECT STATIC METRICS (e.g. disk, load) ─────────────
        metrics: Dict[str, Any] = {}
        for name, fn in cls.METRIC_FUNCS.items():
            try:
                metrics[name] = fn()
            except (AttributeError, NotImplementedError, psutil.AccessDenied):
                # skip unsupported metrics
                continue

        # inject our batched process stats
        metrics['cpu'] = cpu
        metrics['memory'] = memory
        metrics['proc_read_bytes'] = read_bytes
        metrics['proc_write_bytes'] = write_bytes

        # ─── 3) NETWORK I/O COUNTERS & DELTAS ────────────────────────
        now_total  = psutil.net_io_counters()
        now_pernic = psutil.net_io_counters(pernic=True)
        interval   = metrics.get('sample_interval', 1.0) or 1e-6

        if prev_net:
            sent_mb_s = (now_total.bytes_sent - prev_net.bytes_sent) / interval / 1e6
            recv_mb_s = (now_total.bytes_recv - prev_net.bytes_recv) / interval / 1e6
        else:
            sent_mb_s = recv_mb_s = 0.0

        metrics['net_sent_mb_s'] = sent_mb_s
        metrics['net_recv_mb_s'] = recv_mb_s

        # ─── 4) PER-INTERFACE UTILIZATION ────────────────────────────
        # cache speeds to avoid repeated net_if_stats()
        if iface_speeds is None:
            if cls._iface_speeds_cache is None:
                stats = psutil.net_if_stats()
                cls._iface_speeds_cache = {
                    nic: getattr(s, 'speed', 0) or 0 for nic, s in stats.items()
                }
            iface_speeds = cls._iface_speeds_cache

        util: Dict[str, float] = {}
        for nic, counters in now_pernic.items():
            prev = prev_pernic.get(nic) if prev_pernic else None
            speed = iface_speeds.get(nic, 0)
            if prev and speed > 0:
                delta_bytes = (
                    (counters.bytes_sent + counters.bytes_recv)
                    - (prev.bytes_sent + prev.bytes_recv)
                )
                max_bps = speed * 1e6 / 8
                util[nic] = min(1.0, delta_bytes / (interval * max_bps))
            else:
                util[nic] = 0.0

        metrics['net_utilization'] = util

        # ─── 5) OPTIONAL EXPENSIVE SOCKET SCAN ───────────────────────
        # now = time.time()
        # if compute_connections and (now - cls._last_conn_time) >= cls._conn_interval:
        #     conns = psutil.net_connections(kind='inet')
        #     cls._last_conn_summary = _simplify_network_from_conns(conns)
        #     cls._last_conn_time = now

        # metrics['network_summary'] = (
        #     cls._last_conn_summary
        #     if compute_connections or cls._last_conn_summary
        #     else _simplify_network(metrics)
        # )

        # ─── 6) CARRY FORWARD RAW DATA FOR NEXT DELTA CALC ───────────
        metrics['_net_counters']    = now_total
        metrics['_pernic_counters'] = now_pernic
        metrics['_iface_speeds']    = iface_speeds

        return metrics

class Survival:
    """
    Computes a normalized health score across CPU, memory, disk, and network.
    """
    _ctx_history: deque = deque(maxlen=2)
    _disk_scaler: MinMaxScaler = MinMaxScaler()
    DISK_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

    # maximum expected aggregate network rate in MB/s
    MAX_NET_MB_S = settings.MAX_NET_MB_S
    # maximum expected disk I/O in MB/s
    MAX_IO_MBPS = settings.MAX_IO_MBPS or 100.0

    @staticmethod
    def score(metrics: dict) -> dict:
        # ─── CPU sub-score
        cpu_pct   = metrics.get('cpu_percent', 0.0)
        times_pct = metrics.get('cpu_times_pct')
        load1,_,_ = metrics.get('load_avg', (0.0,0.0,0.0))
        cores     = metrics.get('cpu_count', 1)
        freq      = metrics.get('cpu_freq')

        busy     = max(0.0, 1.0 - cpu_pct/100.0)
        idle     = (times_pct.idle/100.0) if times_pct else busy
        usr     = times_pct.user if times_pct else 0.0
        sys_    = times_pct.system if times_pct else 0.0
        user_sys = usr/(usr + sys_ + 1e-6)
        load_s   = max(0.0, 1.0 - load1/cores)
        freq_s   = (freq.current/freq.max) if freq and freq.max else 1.0

        now_cs  = (metrics.get('cpu_stats').ctx_switches if metrics.get('cpu_stats') else 0)
        Survival._ctx_history.append(now_cs)
        delta   = min(1.0, (Survival._ctx_history[-1] - Survival._ctx_history[0]) / 1e6) if len(Survival._ctx_history)==2 else 0.0
        cs_s    = 1.0 - delta

        cpu_s = busy*0.3 + idle*0.2 + user_sys*0.1 + load_s*0.2 + freq_s*0.1 + cs_s*0.1

        # ─── Memory sub-score
        vm    = metrics.get('virtual_mem')
        swap  = metrics.get('swap_mem')
        procs = metrics.get('proc_metrics', {})
        avail = (vm.available/vm.total) if vm and vm.total else 1.0
        swap_s= 1.0 - (swap.percent/100.0) if swap else 1.0
        rss   = sorted((p['memory_percent'] for p in procs.values()), reverse=True)
        top3  = sum(rss[:3]) if rss else 0.0
        rss_s = max(0.0, 1.0 - top3/100.0)
        pf    = metrics.get('page_faults', 0)
        pf_s  = 1.0 - min(1.0, pf/1e6)

        mem_s = avail*0.4 + swap_s*0.3 + rss_s*0.2 + pf_s*0.1

        # ─── Disk sub-score
        du   = metrics.get('disk_usage')
        dio  = metrics.get('disk_io')
        pct  = du.percent/100.0 if du else 0.0
        interval = metrics.get('sample_interval', INTERVAL_SECONDS) or INTERVAL_SECONDS
        rb_s = (dio.read_bytes/(1024*1024))/interval if dio else 0.0
        wb_s = (dio.write_bytes/(1024*1024))/interval if dio else 0.0
        io_u = min(1.0, (rb_s+wb_s)/max(Survival.MAX_IO_MBPS,1.0))
        head = 1.0 - io_u
        bal  = 1.0 - abs(rb_s-wb_s)/(rb_s+wb_s+1.0)
        parts = [(p['disk_read_bytes']+p['disk_write_bytes'])/interval for p in procs.values()]
        top1  = max(parts) if parts else 0.0
        tot   = sum(parts)+1.0
        top1_s = top1/tot

        vec = [pct, head, bal, top1_s]
        if not hasattr(Survival, '_disk_scaler_fitted'):
            Survival._disk_scaler.fit([[0,0,0,0], [1,1,1,1]])
            Survival._disk_scaler_fitted = True
        disk_v = Survival._disk_scaler.transform([vec])[0]
        disk_s = sum(w*x for w,x in zip(Survival.DISK_WEIGHTS, disk_v))

        # ─── Network sub-score
        sent = metrics.get('net_sent_mb_s', 0.0)
        recv = metrics.get('net_recv_mb_s', 0.0)
        tot_n = sent + recv
        net_s = 1.0 - min(1.0, tot_n/max(Survival.MAX_NET_MB_S,1.0))

        # ─── Emotional drives (optional) ──────────────────────────────
        curiosity = metrics.get('curiosity_score', 0.0)
        novelty   = metrics.get('novelty_score', 0.0)

        comp = (cpu_s + mem_s + disk_s + net_s + curiosity + novelty) / 6.0
        return {
            'composite': round(comp,4),
            'cpu': round(cpu_s,4),
            'memory': round(mem_s,4),
            'disk': round(disk_s,4),
            'network': round(net_s,4),
            'curiosity': round(curiosity,4),
            'novelty': round(novelty,4)
        }

    @staticmethod
    def to_dict(metrics: dict) -> dict:
        return {**metrics, **Survival.score(metrics)}