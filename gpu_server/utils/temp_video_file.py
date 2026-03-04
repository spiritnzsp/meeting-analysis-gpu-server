"""
Temporary Video File Management

Streams binary frame data directly to a temp file (not memory) with
size limit and timeout enforcement. Designed for large video files
that can be gigabytes in size.
"""
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMP_VIDEO_PREFIX = "gpu_server_video_"


class TempVideoFile:
    """
    Streams binary frame data to a temp file with size limit and timeout.

    Unlike TempAudioFile which takes all data upfront, this class accepts
    incremental appends as binary WebSocket frames arrive. This prevents
    accumulating potentially gigabytes of video data in memory.
    """

    def __init__(
        self,
        suffix: str = ".mp4",
        max_size: int = 2 * 1024 * 1024 * 1024,  # 2GB
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize a temp video file for streaming writes.

        Args:
            suffix: File extension
            max_size: Maximum allowed file size in bytes
            temp_dir: Directory for temp files (must NOT be tmpfs for large files)
        """
        self._suffix = suffix
        self._max_size = max_size
        self._temp_dir = temp_dir
        self._current_size = 0
        self._path: Optional[Path] = None
        self._fd: Optional[int] = None
        self._finalized = False
        self._cleaned = False
        self._last_write_time = time.monotonic()

        # Create the temp file immediately
        self._create()

    def _create(self) -> None:
        """Create the temp file."""
        fd, path_str = tempfile.mkstemp(
            suffix=self._suffix,
            prefix=TEMP_VIDEO_PREFIX,
            dir=self._temp_dir,
        )
        self._fd = fd
        self._path = Path(path_str)
        self._last_write_time = time.monotonic()

    def append(self, data: bytes) -> None:
        """
        Append data to the temp file.

        Args:
            data: Bytes to append

        Raises:
            ValueError: If file is finalized or exceeds max size
            RuntimeError: If file has been cleaned up
        """
        if self._cleaned:
            raise RuntimeError("TempVideoFile has been cleaned up")
        if self._finalized:
            raise ValueError("TempVideoFile is already finalized")
        if self._fd is None:
            raise RuntimeError("TempVideoFile not initialized")

        new_size = self._current_size + len(data)
        if new_size > self._max_size:
            raise ValueError(
                f"Video data exceeds maximum size "
                f"({new_size / 1024 / 1024:.0f} MB > {self._max_size / 1024 / 1024:.0f} MB)"
            )

        os.write(self._fd, data)
        self._current_size = new_size
        self._last_write_time = time.monotonic()

    def finalize(self) -> Path:
        """
        Finalize the temp file (close for writing).

        Returns:
            Path to the completed temp file

        Raises:
            ValueError: If already finalized or no data written
        """
        if self._finalized:
            return self._path

        if self._current_size == 0:
            raise ValueError("No data written to temp video file")

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

        self._finalized = True
        return self._path

    def cleanup(self) -> None:
        """Delete the temp file and release resources."""
        if self._cleaned:
            return

        # Close FD if still open
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        # Delete file
        if self._path is not None:
            try:
                if self._path.exists():
                    self._path.unlink()
                    logger.debug(f"Cleaned up temp video file: {self._path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp video file {self._path}: {e}")

        self._cleaned = True

    @property
    def current_size(self) -> int:
        """Current size of data written."""
        return self._current_size

    @property
    def is_complete(self) -> bool:
        """Whether the file has been finalized."""
        return self._finalized

    @property
    def path(self) -> Optional[Path]:
        """Path to the temp file."""
        return self._path

    @property
    def seconds_since_last_write(self) -> float:
        """Seconds elapsed since last write (for inactivity timeout)."""
        return time.monotonic() - self._last_write_time

    def __del__(self):
        """Best-effort cleanup on garbage collection."""
        self.cleanup()
