import os
import shutil
import gzip
import logging
from logging.handlers import RotatingFileHandler

class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Extends RotatingFileHandler with a fallback for locked files,
    plus post-rotate compression of the backup.
    """
    def rotate(self, source: str, dest: str) -> None:
        """
        Attempt to rename; on PermissionError, fallback to copy & truncate.
        """
        try:
            os.rename(source, dest)
        except PermissionError:
            try:
                shutil.copy2(source, dest)
                # truncate original so logging can continue
                with open(source, 'w'):
                    pass
            except Exception as e:
                logging.getLogger().error(f"SafeRotate fallback failed: {e}")

    def doRollover(self) -> None:
        """
        Perform the rollover; then compress the rotated file.
        """
        # close current stream
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self.stream = None

        # let base class do its thing (may raise FileNotFoundError)
        try:
            super().doRollover()
        except FileNotFoundError:
            return

        # compress the just-created backup
        if self.backupCount > 0:
            backup_name = f"{self.baseFilename}.1"
            if os.path.exists(backup_name):
                with open(backup_name, 'rb') as f_in, gzip.open(f"{backup_name}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(backup_name)

        # reopen log file
        self.stream = self._open()


def configure_logging(
    log_file_path: str = "godseed_embryo.log",
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 7,
    level: int = logging.DEBUG,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Configure the root logger with SafeRotatingFileHandler.

    - log_file_path: path to the active log file
    - max_bytes:    file size threshold to trigger a rotation
    - backup_count: how many rotated files to keep before oldest is purged
    - level:        minimum logging level
    - fmt, datefmt: formatting strings for log records
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # Create and attach the SafeRotatingFileHandler
    handler = SafeRotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
