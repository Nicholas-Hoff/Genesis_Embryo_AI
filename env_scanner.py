import os
import platform
import socket
from dataclasses import dataclass

@dataclass
class EnvironmentInfo:
    os: str
    release: str
    version: str
    machine: str
    hostname: str
    is_container: bool


def scan_environment() -> EnvironmentInfo:
    """Return basic information about the current host environment."""
    os_name = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    hostname = socket.gethostname()
    is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    return EnvironmentInfo(os_name, release, version, machine, hostname, is_container)
