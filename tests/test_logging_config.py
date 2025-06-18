import logging
import gzip
from logging_config import configure_logging, SafeRotatingFileHandler


def test_rotated_file_gzipped(tmp_path):
    log_file = tmp_path / "test.log"
    configure_logging(log_file_path=str(log_file), max_bytes=50, backup_count=1)
    logger = logging.getLogger()
    handler = next(h for h in logger.handlers if isinstance(h, SafeRotatingFileHandler))

    for _ in range(10):
        logger.info("x" * 10)

    handler.doRollover()
    rotated = log_file.with_name(log_file.name + ".1.gz")
    assert rotated.exists()
    with gzip.open(rotated, "rb") as f:
        data = f.read()
    assert b"x" in data
