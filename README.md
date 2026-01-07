# Meeting Analysis GPU Server

A WebSocket-based GPU processing service for meeting audio analysis. Provides centralised Whisper transcription and PyAnnote speaker diarisation for the [Meeting Analysis](https://github.com/spiritnzsp/meeting-analysis) application.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU Server                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │  WebSocket  │──▶│   Queue     │──▶│  GPU Worker     │   │
│  │   Server    │◀──│  Manager    │◀──│  - Whisper      │   │
│  └─────────────┘   └─────────────┘   │  - PyAnnote     │   │
│        :8765                         └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Whisper transcription** with word-level timestamps
- **PyAnnote speaker diarisation** with voice embeddings
- **Request queuing** with priority support
- **Progress streaming** via WebSocket
- **API key authentication** for remote access
- **Automatic GPU detection** (CUDA/CPU fallback)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended: 8GB+ VRAM)
- PyTorch with CUDA support
- ~10GB disk space for models

## Installation

```bash
# Clone the repository
git clone https://github.com/spiritnzsp/meeting-analysis-gpu-server.git
cd meeting-analysis-gpu-server

# Create conda environment
conda env create -f environment.yml
conda activate gpu-server

# Or with pip
pip install -e .

# Copy and edit configuration
cp config.example.yaml config.yaml
```

## Configuration

Edit `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8765

auth:
  enabled: true
  api_keys:
    - "your-secret-api-key-here"

whisper:
  model: "large-v3"
  device: "cuda"
  compute_type: "float16"

pyannote:
  device: "cuda"
  huggingface_token: "hf_xxxxx"  # Required for PyAnnote models
```

## Usage

### Start the server

```bash
# Using module
python -m gpu_server

# Or with custom config
python -m gpu_server --config /path/to/config.yaml

# As systemd service (see scripts/gpu-server.service)
sudo systemctl start gpu-server
```

### Client connection

The Meeting Analysis application connects via WebSocket:

1. Open **Settings > GPU Server**
2. Enter server URL: `ws://your-server:8765` (or `wss://` for TLS)
3. Enter API key
4. Enable remote processing

## Protocol

### Connection

```json
{"type": "auth", "api_key": "your-key"}
```

### Processing request

```json
{
  "type": "process",
  "request_id": "uuid",
  "audio_data": "<base64 encoded opus>",
  "options": {
    "transcribe": true,
    "diarize": true,
    "extract_embeddings": true,
    "whisper_model": "large-v3",
    "num_speakers": null
  }
}
```

### Progress updates

```json
{"type": "progress", "request_id": "uuid", "stage": "transcribing", "percent": 45, "message": "Processing..."}
```

### Result

```json
{
  "type": "result",
  "request_id": "uuid",
  "transcript": {...},
  "diarization": {...},
  "embeddings": [...]
}
```

## Remote Access (VPN)

For secure remote access, use a VPN solution like:

- **Tailscale** (recommended - easy setup)
- **WireGuard**
- **ZeroTier**

The server can then be accessed via VPN IP without exposing to public internet.

## Development

```bash
# Run tests
pytest

# Run with debug logging
python -m gpu_server --log-level DEBUG
```

## License

MIT License - See LICENSE file
