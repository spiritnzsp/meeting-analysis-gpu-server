# Meeting Analysis GPU Server

A WebSocket-based GPU processing service for meeting audio analysis and video encoding. Provides centralised Whisper transcription, PyAnnote speaker diarisation, and NVENC hardware video encoding for the [Meeting Analysis](https://github.com/spiritnzsp/meeting-analysis) application.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GPU Server                                │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  WebSocket  │──▶│ Audio Queue  │──▶│  Audio Worker        │  │
│  │   Server    │◀──│  Manager     │◀──│  - Whisper           │  │
│  │   :8765     │   └──────────────┘   │  - PyAnnote          │  │
│  │             │   ┌──────────────┐   └──────────────────────┘  │
│  │  JSON + ────│──▶│ Video Queue  │──▶┌──────────────────────┐  │
│  │  Binary     │◀──│  Manager     │◀──│  Video Worker        │  │
│  └─────────────┘   └──────────────┘   │  - FFmpeg/NVENC      │  │
│                                       └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Features

- **Whisper transcription** with word-level timestamps
- **PyAnnote speaker diarisation** with voice embeddings
- **NVENC video encoding** with AV1, HEVC, and H.264 codec support
- **Binary frame protocol** for streaming large video files (up to 5GB)
- **Request queuing** with priority support (separate audio and video queues)
- **Progress streaming** via WebSocket
- **API key authentication** for remote access
- **Automatic GPU detection** (CUDA/CPU fallback)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended: 8GB+ VRAM)
- PyTorch with CUDA support
- FFmpeg with NVENC support (for video encoding)
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

video_encoding:
  enabled: true
  ffmpeg_path: "ffmpeg"
  preferred_codecs: ["av1_nvenc", "hevc_nvenc", "h264_nvenc"]
  default_preset: "p4"       # p1 (fastest) to p7 (best quality)
  default_quality: 23        # CQ value, 0-51
  max_sessions: 2            # NVENC hardware session limit
  max_input_size: 5368709120 # 5GB
  data_upload_timeout: 600   # 10 min for large files
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

### Video encoding request

Video encoding uses a binary frame protocol for efficient streaming of large files.

**Step 1: Send JSON control message**

```json
{
  "type": "video_encode",
  "request_id": "uuid",
  "filename": "recording.mp4",
  "input_size": 3500000000,
  "options": {
    "codec": null,
    "preset": null,
    "quality": 28,
    "resolution": null,
    "framerate": null,
    "bitrate": null,
    "pixel_format": null
  }
}
```

All options are optional (null = server defaults). Valid codecs: `av1_nvenc`, `hevc_nvenc`, `h264_nvenc`. Valid presets: `p1`-`p7`.

**Step 2: Stream video data as binary frames**

After receiving the `queued` response, the client streams the file as binary WebSocket messages:

```
Binary frame format: [1B frame_type][1B id_length][N B request_id_utf8][payload]

Frame types:
  0x01 = VIDEO_INPUT  (client -> server)
  0x02 = VIDEO_OUTPUT (server -> client)
```

The client sends `VIDEO_INPUT` frames with file data in chunks (recommended 512KB). An empty-payload `VIDEO_INPUT` frame signals upload complete.

**Step 3: Receive encoding result**

The server sends progress updates during encoding, then the result:

```json
{
  "type": "video_result",
  "request_id": "uuid",
  "success": true,
  "output_size": 500000000,
  "codec_used": "hevc_nvenc",
  "encoding_time": 45.2
}
```

On success, the server streams the encoded output as `VIDEO_OUTPUT` binary frames (64KB chunks), terminated by an empty-payload frame.

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
