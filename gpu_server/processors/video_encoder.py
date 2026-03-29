"""
Video Encoder Processor

FFmpeg NVENC-based video encoding using asyncio.subprocess.
Standalone processor (not extending BaseProcessor) because it uses
asyncio.subprocess rather than ThreadPoolExecutor, and cancels by
killing the FFmpeg process rather than checking a flag.
"""
import asyncio
import re
from pathlib import Path
from typing import Optional, List, Callable, Awaitable, Tuple

from ..config import VideoEncodingConfig
from ..logging_config import get_logger

logger = get_logger(__name__)


class CodecCapabilities:
    """
    Probes FFmpeg once at startup, caches available NVENC codecs.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self._ffmpeg_path = ffmpeg_path
        self._available_codecs: List[str] = []
        self._probed = False

    @property
    def available_codecs(self) -> List[str]:
        return list(self._available_codecs)

    async def detect_available_codecs(self) -> List[str]:
        """
        Probe FFmpeg for available NVENC encoders.

        Returns:
            List of available NVENC codec names
        """
        if self._probed:
            return self._available_codecs

        try:
            proc = await asyncio.create_subprocess_exec(
                self._ffmpeg_path, "-encoders", "-hide_banner",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            output = stdout.decode('utf-8', errors='replace')
            nvenc_codecs = []

            for line in output.splitlines():
                # FFmpeg encoder lines look like: " V....D h264_nvenc           NVIDIA NVENC H.264 encoder"
                line = line.strip()
                if 'nvenc' in line.lower():
                    # Extract codec name (second whitespace-delimited token after the flags)
                    parts = line.split()
                    if len(parts) >= 2:
                        codec_name = parts[1]
                        if 'nvenc' in codec_name:
                            nvenc_codecs.append(codec_name)

            self._available_codecs = nvenc_codecs
            self._probed = True

            logger.info(f"Detected NVENC codecs: {nvenc_codecs}")
            return nvenc_codecs

        except FileNotFoundError:
            logger.error(f"FFmpeg not found at: {self._ffmpeg_path}")
            self._probed = True
            return []
        except asyncio.TimeoutError:
            logger.error("FFmpeg codec detection timed out")
            self._probed = True
            return []
        except Exception as e:
            logger.error(f"FFmpeg codec detection failed: {e}")
            self._probed = True
            return []

    def select_codec(self, requested: Optional[str], preferred_order: List[str]) -> Optional[str]:
        """
        Select the best available codec.

        Args:
            requested: Client-requested codec (None = auto-select)
            preferred_order: Server's preferred codec order

        Returns:
            Codec name or None if no NVENC codec is available
        """
        if requested and requested in self._available_codecs:
            return requested

        for codec in preferred_order:
            if codec in self._available_codecs:
                return codec

        return None


class VideoEncoderProcessor:
    """
    FFmpeg NVENC video encoder using asyncio.subprocess.

    Does NOT extend BaseProcessor because:
    - Uses asyncio.create_subprocess_exec (not ThreadPoolExecutor)
    - Cancels by killing the FFmpeg process (not a checked flag)
    - Different concurrency model would violate LSP if forced into BaseProcessor
    """

    def __init__(self, config: VideoEncodingConfig):
        self.config = config
        self._codec_caps = CodecCapabilities(config.ffmpeg_path)
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._current_request_id: Optional[str] = None
        self._is_processing = False
        self._shutdown = False

    @property
    def codec_capabilities(self) -> CodecCapabilities:
        return self._codec_caps

    async def initialize(self) -> None:
        """Initialize by detecting available codecs. Call at startup."""
        codecs = await self._codec_caps.detect_available_codecs()
        if not codecs:
            logger.error("No NVENC codecs detected. Video encoding will not work.")

    async def encode(
        self,
        input_path: Path,
        output_path: Path,
        options: 'VideoCodecOptions',
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> Tuple[bool, str, str]:
        """
        Encode a video file using FFmpeg NVENC.

        Args:
            input_path: Path to input video file
            output_path: Path for output encoded file
            options: Video codec options
            progress_callback: Callback(percent, message) for progress
            request_id: For logging and cancellation

        Returns:
            Tuple of (success, error_message, codec_used)
        """
        if self._shutdown:
            return False, "Encoder has been shut down", ""

        self._is_processing = True
        self._current_request_id = request_id

        try:
            # Select codec
            codec = self._codec_caps.select_codec(
                options.codec,
                self.config.preferred_codecs,
            )
            if not codec:
                return False, "No compatible NVENC codec available", ""

            # Build FFmpeg command
            cmd = self._build_ffmpeg_cmd(input_path, output_path, codec, options)
            logger.info(f"FFmpeg command: {' '.join(cmd)}")

            if progress_callback:
                await progress_callback(5, f"Starting encoding with {codec}")

            # Run FFmpeg with progress parsing
            # Use 1MB buffer limit (default 64KB is too small for large files
            # where FFmpeg can emit very long stderr lines with warnings/diagnostics)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )
            self._current_process = proc

            # Read stderr for progress/errors
            # FFmpeg writes progress to stderr by default
            stderr_lines = []
            duration_seconds = None

            async def read_stderr():
                nonlocal duration_seconds
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').strip()
                    stderr_lines.append(decoded)

                    # Parse duration from input info
                    if duration_seconds is None:
                        dur_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+)\.(\d+)', decoded)
                        if dur_match:
                            h, m, s, cs = dur_match.groups()
                            duration_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0

                    # Parse progress from time= in stderr
                    if progress_callback and duration_seconds and duration_seconds > 0:
                        time_match = re.search(r'time=(\d+):(\d+):(\d+)\.(\d+)', decoded)
                        if time_match:
                            h, m, s, cs = time_match.groups()
                            current_time = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0
                            percent = min(95, int((current_time / duration_seconds) * 100))
                            await progress_callback(percent, f"Encoding: {percent}%")

            try:
                await read_stderr()
                await proc.wait()
            except asyncio.CancelledError:
                proc.kill()
                await proc.wait()
                return False, "Encoding cancelled", codec

            self._current_process = None

            if proc.returncode == 0:
                if progress_callback:
                    await progress_callback(100, "Encoding complete")
                return True, "", codec
            else:
                # Get last few lines of stderr for error context
                error_context = "\n".join(stderr_lines[-5:]) if stderr_lines else "Unknown error"
                return False, f"FFmpeg exited with code {proc.returncode}: {error_context}", codec

        except Exception as e:
            logger.error(f"Encoding error: {e}", exc_info=True)
            return False, str(e), ""
        finally:
            self._is_processing = False
            self._current_request_id = None
            self._current_process = None

    def _build_ffmpeg_cmd(
        self,
        input_path: Path,
        output_path: Path,
        codec: str,
        options: 'VideoCodecOptions',
    ) -> List[str]:
        """Build the FFmpeg command line."""
        cmd = [
            self.config.ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", codec,
        ]

        # Preset
        preset = options.preset or self.config.default_preset
        cmd.extend(["-preset", preset])

        # Quality (CQ mode) or bitrate
        if options.bitrate:
            cmd.extend(["-b:v", options.bitrate])
        else:
            quality = options.quality if options.quality is not None else self.config.default_quality
            cmd.extend(["-cq", str(quality)])

        # Resolution
        if options.resolution:
            w, h = options.resolution.split("x")
            cmd.extend(["-vf", f"scale={w}:{h}"])

        # Framerate
        if options.framerate:
            cmd.extend(["-r", str(options.framerate)])

        # Pixel format
        if options.pixel_format:
            cmd.extend(["-pix_fmt", options.pixel_format])

        # Copy audio stream
        cmd.extend(["-c:a", "copy"])

        # Output
        cmd.append(str(output_path))

        return cmd

    def cancel(self, request_id: str = "") -> None:
        """
        Cancel encoding by killing the FFmpeg process.

        Args:
            request_id: Request to cancel, or empty for any current request
        """
        proc = self._current_process
        if proc is None:
            return

        if request_id and self._current_request_id != request_id:
            return

        try:
            proc.kill()
            logger.info(f"Killed FFmpeg process for request {request_id or self._current_request_id}")
        except ProcessLookupError:
            pass  # Already exited

    async def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """
        Wait for encoding to complete.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if idle, False if timeout
        """
        if not self._is_processing:
            return True

        deadline = asyncio.get_event_loop().time() + timeout
        while self._is_processing:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.1, remaining))

        return True

    def shutdown(self) -> None:
        """Shutdown the encoder."""
        if self._shutdown:
            return
        self._shutdown = True
        self.cancel()  # Kill any running FFmpeg process
        logger.info("VideoEncoderProcessor shut down")
