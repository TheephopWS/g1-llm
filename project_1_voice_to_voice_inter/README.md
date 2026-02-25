# Robot Voice Brain - LLM Robotic Integration

Interactive voice-to-voice robot system with LLM brain for Unitree G1.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Laptop (Server)                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Whisper  │───▶│  Ollama  │───▶│ Validate │───▶│ Edge-TTS │  │
│  │   STT    │    │   LLM    │    │  Action  │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│        ▲                                               │        │
│        │         FastAPI Server (:8000)                ▼        │
└────────┼───────────────────────────────────────────────┼────────┘
         │                                               │
    [WAV Audio]                              [JSON + MP3 Audio URL]
         │                                               │
         │                   Network                     │
         │                                               │
┌────────┼───────────────────────────────────────────────┼────────┐
│        │           Jetson Orin (Robot)                 ▼        │
│  ┌──────────┐                                   ┌──────────┐    │
│  │   Mic    │                                   │  Speaker │    │
│  │ Recording│                                   │ Playback │    │
│  └──────────┘                                   └──────────┘    │
│                                                        │        │
│                    ┌──────────────────┐               │        │
│                    │  Unitree G1 SDK  │◀──────────────┘        │
│                    │  Action Dispatch │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Conda Environment

```bash
# Create environment
conda create -n llm_env python=3.10
conda activate llm_env
```

### 2. Install Ollama

Ollama is the local LLM runtime that runs the language model.

**Windows:**
1. Download the installer from: https://ollama.ai/download/windows
2. Run the installer and follow the prompts
3. Ollama will start automatically as a background service

**Linux/WSL:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
# Or download from: https://ollama.ai/download/mac
```

**Verify installation:**
```bash
ollama --version
```

### 3. Pull Required Models

```bash
# Pull the LLM model (qwen2.5:7b recommended for quality)
ollama pull qwen2.5:7b

# Alternative: qwen2.5:3b-instruct for faster speed (lower quality)
# ollama pull qwen2.5:3b-instruct

# Verify the model is downloaded
ollama list

# (Optional) Test the model
ollama run qwen2.5:7b "Hello, how are you?"
```

**Note:** The first pull may take a few minutes depending on your internet speed (~4GB for 7b model).

### 4. Server Setup (Laptop)

```bash
conda activate llm_env

# Install dependencies
pip install -r requirements_server.txt

# Run server
python server.py
```

Server will be available at `http://localhost:8000`

### 4. Test Client (Same Laptop)

```bash
# In another terminal
conda activate llm_env

# Install client dependencies
pip install -r requirements_client.txt

# Run interactive test
python test_client.py

# Or quick text test
python test_client.py --text "Hello, can you wave at me?"

# Or quick voice test
python test_client.py --voice --duration 5
```

### 5. Robot Client (Jetson Orin)

```bash
# On Orin
conda activate llm_env  # or your robot's environment

pip install -r requirements_client.txt

# Run with server IP
python robot_client.py --server http://<4090_LAPTOP_IP>:8000 --session g1-001
```

## Configuration

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM model to use |
| `K_TURNS` | `5` | Conversation history length (turns) |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `OUTPUT_DIR` | `./outputs` | Directory for generated audio |

Example:
```bash
OLLAMA_MODEL=llama3:8b K_TURNS=10 python server.py
```

### Client Arguments

```bash
# Test client
python test_client.py --server http://localhost:8000 --session my-session

# Robot client
python robot_client.py \
    --server http://192.168.1.100:8000 \
    --session g1-001 \
    --sample-rate 16000 \
    --no-simulate  # Execute real robot actions
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/actions` | GET | List allowed actions |
| `/process_voice` | POST | Main voice processing (audio → response) |
| `/process_text` | POST | Text-only processing (for testing) |
| `/audio/{id}.mp3` | GET | Download generated audio |
| `/session/{id}` | GET | Get session history |
| `/session/{id}/reset` | POST | Reset session |
| `/sessions` | GET | List all sessions |

### Process Voice Request

```bash
curl -X POST "http://localhost:8000/process_voice" \
  -F "file=@input.wav" \
  -F "session_id=robot-1" \
  -F 'robot_status_json={"battery":0.85,"location":"living_room"}'
```

Response:
```json
{
  "session_id": "robot-1",
  "user_text": "Hello, can you wave at me?",
  "speech": "Hello! Sure, I'll wave at you!",
  "auxiliary": {
    "action": "WAVE_HAND",
    "params": {}
  },
  "detected_language": "en",
  "audio_id": "abc123...",
  "audio_url": "/audio/abc123....mp3"
}
```

## Available Actions

| Action | Description |
|--------|-------------|
| `NONE` | No physical action |
| `MOVE_FORWARD` | Walk forward |
| `DANCE` | Dance routine |

## Adding Custom Actions

1. **Server side** - Edit `ALLOWED_ACTIONS` in [server.py](server.py):

```python
ALLOWED_ACTIONS: Dict[str, str] = {
    ...
    "MY_NEW_ACTION": "Description for LLM to understand when to use it.",
}
```

2. **Robot side** - Add handler in [robot_client.py](robot_client.py):

```python
class UnitreeActionDispatcher:
    def __init__(self, ...):
        self._action_handlers = {
            ...
            "MY_NEW_ACTION": self._action_my_new,
        }
    
    def _action_my_new(self, params: Dict) -> bool:
        # Your Unitree SDK call here
        # unitree_g1.my_new_action()
        return True
```

## Conversation Memory

- Server maintains conversation history per `session_id`
- Default: 5 turns (10 messages: 5 user + 5 assistant)
- Reset with: `POST /session/{id}/reset`
- Only text is stored, not actions (to avoid prompt bloat)

## Language Support

The system auto-detects language from speech:
- **English** → `en-US-JennyNeural` voice
- **Mandarin** → `zh-CN-XiaoxiaoNeural` voice
- **Cantonese** → `zh-HK-HiuGaamNeural` voice

The LLM is instructed to respond in the same language as the user.

## Troubleshooting

### Server Issues

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Test with CPU (slower)
WHISPER_DEVICE=cpu python server.py
```

### Client Issues

```bash
# Test audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Check server connectivity
curl http://<server>:8000/health
```

### Common Errors

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Use smaller Whisper model: `WHISPER_MODEL=medium` |
| `Connection refused` | Check server is running and firewall allows port 8000 |
| `Ollama timeout` | Increase timeout or check Ollama status |
| `No audio device` | Check microphone permissions, install PortAudio |

## Project Structure

```
project_1_voice_to_voice_inter/
├── server.py              # FastAPI server (laptop)
├── robot_client.py        # Production client (Orin/robot)
├── test_client.py         # Development test client
├── requirements_server.txt
├── requirements_client.txt
├── README.md
└── outputs/               # Generated audio files (auto-created)
```

## Next Steps

1. **Integrate Unitree G1 SDK**: Replace simulation in `robot_client.py`
2. **Add wake word detection**: Snowboy, Porcupine, or custom
3. **Improve VAD**: Use Silero VAD for better speech detection
4. **Add vision**: Camera input for spatial awareness (very optional)
5. **Safety system**: Hardware interlocks for emergency stop
