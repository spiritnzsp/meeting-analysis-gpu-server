"""
GPU Server Entry Point

Run with: python -m gpu_server
"""
import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from .config import load_config, setup_logging, validate_config, ConfigurationError
from .server import GPUServer

logger = logging.getLogger(__name__)


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
        "--strict",
        action="store_true",
        help="Fail on configuration errors instead of using defaults (recommended for production)",
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

    # Setup logging
    setup_logging(config.logging)

    logger.info("=" * 60)
    logger.info("Meeting Analysis GPU Server")
    logger.info("=" * 60)

    # Validate configuration
    try:
        validate_config(config, strict=args.strict)
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        sys.exit(1)

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("CUDA not available - will use CPU (slow!)")
    except ImportError:
        logger.warning("PyTorch not installed - GPU detection skipped")

    # Create and start server
    server = GPUServer(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
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
