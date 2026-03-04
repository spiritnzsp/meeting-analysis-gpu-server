"""Tests for CodecCapabilities."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from gpu_server.processors.video_encoder import CodecCapabilities


class TestCodecDetection:
    """Test NVENC codec detection."""

    @pytest.mark.asyncio
    async def test_detect_codecs_from_ffmpeg_output(self):
        """Test parsing real FFmpeg encoder output."""
        ffmpeg_output = (
            "Encoders:\n"
            " V..... = Video\n"
            " ------\n"
            " V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n"
            " V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)\n"
            " V....D av1_nvenc            NVIDIA NVENC AV1 encoder (codec av1)\n"
            " V..... libx264              libx264 H.264 / AVC / MPEG-4 (codec h264)\n"
        )

        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(ffmpeg_output.encode(), b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            caps = CodecCapabilities("ffmpeg")
            codecs = await caps.detect_available_codecs()

        assert "h264_nvenc" in codecs
        assert "hevc_nvenc" in codecs
        assert "av1_nvenc" in codecs
        assert "libx264" not in codecs

    @pytest.mark.asyncio
    async def test_detect_no_nvenc(self):
        """Test when no NVENC codecs are available."""
        ffmpeg_output = (
            "Encoders:\n"
            " V..... libx264              libx264 H.264\n"
            " V..... libx265              libx265 HEVC\n"
        )

        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(ffmpeg_output.encode(), b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
            caps = CodecCapabilities("ffmpeg")
            codecs = await caps.detect_available_codecs()

        assert codecs == []

    @pytest.mark.asyncio
    async def test_detect_ffmpeg_not_found(self):
        """Test when FFmpeg is not installed."""
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError):
            caps = CodecCapabilities("/nonexistent/ffmpeg")
            codecs = await caps.detect_available_codecs()

        assert codecs == []

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test that detection results are cached."""
        ffmpeg_output = b" V....D h264_nvenc           NVIDIA NVENC H.264\n"
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(ffmpeg_output, b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            caps = CodecCapabilities("ffmpeg")
            await caps.detect_available_codecs()
            await caps.detect_available_codecs()  # Second call

        # Should only call FFmpeg once
        assert mock_exec.call_count == 1


class TestCodecSelection:
    """Test codec selection logic."""

    def test_select_requested_if_available(self):
        caps = CodecCapabilities()
        caps._available_codecs = ["h264_nvenc", "hevc_nvenc"]
        caps._probed = True

        result = caps.select_codec("hevc_nvenc", ["h264_nvenc", "hevc_nvenc"])
        assert result == "hevc_nvenc"

    def test_select_preferred_order(self):
        caps = CodecCapabilities()
        caps._available_codecs = ["h264_nvenc", "hevc_nvenc"]
        caps._probed = True

        result = caps.select_codec(None, ["av1_nvenc", "hevc_nvenc", "h264_nvenc"])
        assert result == "hevc_nvenc"  # First available in preferred order

    def test_select_none_when_unavailable(self):
        caps = CodecCapabilities()
        caps._available_codecs = []
        caps._probed = True

        result = caps.select_codec(None, ["av1_nvenc", "hevc_nvenc"])
        assert result is None

    def test_requested_not_available_falls_back(self):
        caps = CodecCapabilities()
        caps._available_codecs = ["h264_nvenc"]
        caps._probed = True

        # Requested av1_nvenc not available, falls back to preferred order
        result = caps.select_codec("av1_nvenc", ["h264_nvenc"])
        assert result == "h264_nvenc"
