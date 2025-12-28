# GLM-ASR-STT

A real-time Speech-to-Text (STT) system using the GLM-ASR-Nano model. It supports local or remote ASR processing, making it ideal for offloading computation to a more powerful machine (like an AMD laptop with a GPU/NPU).

## Project Structure

```
.
├── client/
│   ├── main.py              # Primary entry point (VAD, recording, typing)
│   ├── keyboard_listener.py # Captures F12 key events (requires sudo)
│   └── config.json          # Client configuration
├── server/
│   └── server.py            # ASR HTTP server (GLM-ASR model)
├── assets/                  # Notification sounds
├── requirements.txt         # Python dependencies
└── README.md
```

## Features

- **Real-time VAD**: Uses Silero VAD to detect speech and automatically stop recording.
- **Global Hotkey**: Press and hold **F12** to start recording.
- **Automatic Typing**: Transcribed text is automatically typed into your active window.
- **Distributed Architecture**: Run the ASR model on a separate machine to save resources on your main workstation.
- **Configurable**: Easily customize audio devices and notification sounds via `config.json`.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/GLM-ASR-STT.git
    cd GLM-ASR-STT
    ```

2.  **Install dependencies**:
    This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.
    ```bash
    uv sync
    ```
    *Note: You may still need to install system-level dependencies like `portaudio` for `sounddevice` and `libuinput-dev` for `python-uinput`.*

3.  **Configure**:
    Edit `client/config.json` to set your preferred audio input devices and sound paths.

## Usage

### 1. Single-Machine Mode (Local ASR)

Simply run the client using `uv`. It will automatically start the local ASR server and keyboard listener.

```bash
uv run client/main.py
```

### 2. Distributed Mode (Remote ASR)

**On the ASR Server (e.g., AMD Laptop):**
```bash
uv run server/server.py --port 8000
```

**On the Client Machine:**
```bash
uv run client/main.py --asr-server http://<server-ip>:8000
```

## Configuration (`client/config.json`)

- `audio_devices`: A list of substrings to match your preferred microphone name.
- `sound_up`: Path to the sound played when recording starts.
- `sound_down`: Path to the sound played when recording ends.
- `socket_path`: Path for the internal communication socket.
- `default_asr_server`: The default ASR server URL if none is provided via CLI.

## Requirements

- Linux (for `uinput` and Unix domain sockets)
- Python 3.8+
- `sudo` privileges (for keyboard listener)
- `pw-play` (PipeWire) or similar for audio playback
