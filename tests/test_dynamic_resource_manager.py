import os
import subprocess
import psutil

from resources import DynamicResourceManager


def test_safe_terminate_kills_process():
    drm = DynamicResourceManager()
    proc = subprocess.Popen([psutil.Process().exe(), "-c", "import time; time.sleep(5)"])
    try:
        assert psutil.pid_exists(proc.pid)
        assert drm.safe_terminate(proc.pid)
        assert not psutil.pid_exists(proc.pid)
    finally:
        if psutil.pid_exists(proc.pid):
            psutil.Process(proc.pid).kill()


def test_safe_terminate_respects_whitelist_and_blacklist():
    drm = DynamicResourceManager()
    drm.whitelist.add(os.getpid())
    assert not drm.safe_terminate(os.getpid())

    proc = subprocess.Popen([psutil.Process().exe(), "-c", "import time; time.sleep(5)"])
    try:
        name = psutil.Process(proc.pid).name()
        drm.blacklist.add(name)
        assert not drm.safe_terminate(proc.pid)
        assert psutil.pid_exists(proc.pid)
    finally:
        if psutil.pid_exists(proc.pid):
            psutil.Process(proc.pid).kill()
