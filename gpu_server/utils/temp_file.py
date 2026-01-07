"""
Temporary File Management

Provides robust temporary file handling with guaranteed cleanup,
including atexit registration and orphan file detection.
"""
import atexit
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional, Set
import logging

logger = logging.getLogger(__name__)

# Global registry of active temp files for atexit cleanup
_active_temp_files: Set[Path] = set()
_registry_lock = threading.Lock()

# Prefix for our temp files to identify orphans
TEMP_FILE_PREFIX = "gpu_server_audio_"

# Default temp directory (can be overridden)
_temp_directory: Optional[Path] = None


def set_temp_directory(path: Optional[Path]) -> None:
    """
    Set a custom temp directory for audio files.

    Args:
        path: Directory path, or None to use system default
    """
    global _temp_directory
    if path is not None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    _temp_directory = path


def get_temp_directory() -> Path:
    """
    Get the temp directory for audio files.

    Returns:
        Path to temp directory
    """
    if _temp_directory is not None:
        return _temp_directory
    return Path(tempfile.gettempdir())


def _register_temp_file(path: Path) -> None:
    """Register a temp file for cleanup tracking."""
    with _registry_lock:
        _active_temp_files.add(path)


def _unregister_temp_file(path: Path) -> None:
    """Unregister a temp file after cleanup."""
    with _registry_lock:
        _active_temp_files.discard(path)


def _cleanup_all_temp_files() -> None:
    """
    Cleanup all registered temp files.
    Called by atexit to ensure cleanup on normal shutdown.
    """
    with _registry_lock:
        files_to_clean = list(_active_temp_files)

    for path in files_to_clean:
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up temp file on shutdown: {path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {path} on shutdown: {e}")

    with _registry_lock:
        _active_temp_files.clear()


# Register atexit handler
atexit.register(_cleanup_all_temp_files)


def cleanup_orphaned_temp_files(max_age_seconds: int = 3600) -> int:
    """
    Clean up orphaned temp files from previous runs.

    Should be called on server startup to clean files that weren't
    properly cleaned up (e.g., due to crash or SIGKILL).

    Args:
        max_age_seconds: Only delete files older than this (default 1 hour)

    Returns:
        Number of orphaned files cleaned up
    """
    temp_dir = get_temp_directory()
    cleaned = 0
    now = time.time()

    try:
        for path in temp_dir.iterdir():
            if not path.name.startswith(TEMP_FILE_PREFIX):
                continue

            try:
                # Check file age
                stat = path.stat()
                age = now - stat.st_mtime

                if age > max_age_seconds:
                    path.unlink()
                    cleaned += 1
                    logger.info(f"Cleaned orphaned temp file: {path} (age: {age:.0f}s)")
            except Exception as e:
                logger.warning(f"Failed to check/clean orphan {path}: {e}")
    except Exception as e:
        logger.warning(f"Failed to scan temp directory for orphans: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned {cleaned} orphaned temp files from previous runs")

    return cleaned


class TempAudioFile:
    """
    Context manager for temporary audio files with guaranteed cleanup.

    Features:
    - Automatic cleanup on context exit
    - atexit registration for cleanup on process shutdown
    - Unique prefix for orphan detection
    - Thread-safe

    Usage:
        with TempAudioFile(audio_data, suffix=".opus") as audio_path:
            # audio_path is a Path to the temp file
            process_audio(audio_path)
        # File is automatically deleted here
    """

    def __init__(
        self,
        data: bytes,
        suffix: str = ".opus",
        prefix: str = TEMP_FILE_PREFIX,
    ):
        """
        Initialize temp audio file.

        Args:
            data: Audio data to write
            suffix: File extension (default .opus)
            prefix: File prefix for identification
        """
        self._data = data
        self._suffix = suffix
        self._prefix = prefix
        self._path: Optional[Path] = None
        self._cleaned = False

    def __enter__(self) -> Path:
        """Create and write the temp file."""
        temp_dir = get_temp_directory()

        # Create temp file with our prefix
        fd, path_str = tempfile.mkstemp(
            suffix=self._suffix,
            prefix=self._prefix,
            dir=temp_dir,
        )

        self._path = Path(path_str)

        # Register for atexit cleanup
        _register_temp_file(self._path)

        try:
            # Write data
            os.write(fd, self._data)
        finally:
            os.close(fd)

        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up the temp file."""
        self._cleanup()
        return None  # Don't suppress exceptions

    def _cleanup(self) -> None:
        """Internal cleanup method."""
        if self._cleaned or self._path is None:
            return

        try:
            if self._path.exists():
                self._path.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {self._path}: {e}")
        finally:
            _unregister_temp_file(self._path)
            self._cleaned = True

    @property
    def path(self) -> Optional[Path]:
        """Get the path to the temp file (None if not created yet)."""
        return self._path
