"""
GPU Server Entry Point

Run with: python -m gpu_server
"""
import argparse
import asyncio
import signal
import sys
from pathlib import Path

from .config import load_config, validate_config, ConfigurationError, VideoEncodingConfig
from .server import GPUServer
from .logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Meeting Analysis GPU Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--host",
        help="Host to bind to (overrides config)",
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port to listen on (overrides config)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides config)",
    )

    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Continue despite configuration errors (not recommended for production)",
    )

    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format (recommended for production/log aggregation)",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored console output",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.log_level:
        config.logging.level = args.log_level

    # Setup structured logging
    configure_logging(
        level=config.logging.level,
        json_format=args.json_logs,
        use_colors=not args.no_color,
    )

    logger.info("=" * 60)
    logger.info("Meeting Analysis GPU Server")
    logger.info("=" * 60)

    # Validate configuration (strict=True by default)
    try:
        validate_config(config, strict=not args.no_strict)
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        sys.exit(1)

    # Check GPU availability and auto-compute max input size from VRAM
    from .gpu_info import detect_gpu
    gpu = detect_gpu()
    if gpu:
        logger.info(f"GPU: {gpu.name} ({gpu.total_memory_gb:.1f} GB)")
        # Auto-set max_input_size to 80% of VRAM if user hasn't overridden
        default_max = VideoEncodingConfig().max_input_size
        if config.video_encoding.max_input_size == default_max:
            config.video_encoding.max_input_size = int(gpu.total_memory_bytes * 0.80)
            logger.info(
                f"  Max input size (auto): "
                f"{config.video_encoding.max_input_size / 1024 / 1024:.0f} MB "
                f"(80% of VRAM)"
            )

    # Create and start server
    server = GPUServer(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.start()

        logger.info("-" * 60)
        logger.info(f"Server ready: ws://{config.server.host}:{config.server.port}")
        logger.info(f"Auth enabled: {config.auth.enabled}")
        logger.info(f"Whisper model: {config.whisper.model}")
        logger.info(f"PyAnnote model: {config.pyannote.model}")
        if config.video_encoding.enabled:
            logger.info(f"Video encoding: ENABLED")
            logger.info(f"  FFmpeg: {config.video_encoding.ffmpeg_path}")
            logger.info(f"  Preferred codecs: {', '.join(config.video_encoding.preferred_codecs)}")
            logger.info(f"  Max input size: {config.video_encoding.max_input_size / 1024 / 1024:.0f} MB")
        else:
            logger.info(f"Video encoding: disabled")
        logger.info("-" * 60)

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        await server.stop()
        logger.info("Server shutdown complete")


def main_sync():
    """Synchronous entry point for console script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main_sync()
