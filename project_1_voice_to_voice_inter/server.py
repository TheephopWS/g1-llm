"""
Robot Voice Brain Server
========================
FastAPI server running on laptop (4090 GPU):
- Audio upload -> Whisper STT -> Build prompt with last K turns
- Ollama JSON response -> Validate/normalize action -> Edge-TTS
- Returns JSON + audio URL

Supports two modes:
1. Batch mode: POST /process_voice (upload complete audio)
2. Streaming mode: WebSocket /ws/stream (real-time audio streaming)

Run with: python server.py
Or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import json
import os
import re
import struct
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field


# =============================================================================
# Configuration (can be set via environment variables)
# =============================================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Keep last K turns (a "turn" = user + assistant). Internally that's 2*K messages.
K_TURNS = int(os.getenv("K_TURNS", "5"))

# Output directory for generated audio files
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Whisper model configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Concurrency guards (tune as needed)
STT_CONCURRENCY = int(os.getenv("STT_CONCURRENCY", "1"))   # Whisper on GPU
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "2"))   # Ollama
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "2"))   # Edge TTS


# =============================================================================
# Unitree G1 High-Level SDK Actions
# Replace/extend these with your actual Unitree G1 high-level SDK action set.
# =============================================================================
ALLOWED_ACTIONS: Dict[str, str] = {
    "NONE": "Do nothing / no physical action. Use this for normal conversation.",
    "MOVE_FORWARD": "Walk forward toward the user.",
    "DANCE": "Perform a dance routine.",
}

DEFAULT_ACTION = "NONE"


# =============================================================================
# Pydantic Models
# =============================================================================
class AuxiliaryCommand(BaseModel):
    """Represents the action command for the robot."""
    action: str = Field(default=DEFAULT_ACTION, description="Action name from ALLOWED_ACTIONS")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the action")


class ProcessVoiceResponse(BaseModel):
    """Response from /process_voice endpoint."""
    session_id: str
    user_text: str
    speech: str
    auxiliary: AuxiliaryCommand
    detected_language: Optional[str] = None
    audio_id: str
    audio_url: str


class SessionInfo(BaseModel):
    """Information about a session."""
    session_id: str
    message_count: int
    messages: List[Dict[str, str]]


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="Robot Voice Brain",
    description="Voice-to-voice interactive robot server with LLM brain",
    version="1.0.0",
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Global Resources
# =============================================================================
memory_lock = asyncio.Lock()
sessions: Dict[str, deque] = {}  # session_id -> deque of {role, content}

stt_sem = asyncio.Semaphore(STT_CONCURRENCY)
llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)
tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)

# Whisper model (loaded lazily to allow module import without GPU)
_stt_model: Optional[WhisperModel] = None


def get_stt_model() -> WhisperModel:
    """Lazy load Whisper model."""
    global _stt_model
    if _stt_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}...")
        _stt_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        print("Whisper model loaded successfully!")
    return _stt_model


# =============================================================================
# Language Detection & Voice Selection Helpers
# =============================================================================
def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _looks_cantonese(text: str) -> bool:
    """Lightweight heuristic for Cantonese text detection."""
    hints = ["係", "唔", "喺", "咩", "嘅", "啲", "嗰", "冇", "佢", "咗", "啦", "喇", "嘢", "乜", "噉"]
    return any(h in (text or "") for h in hints)


def pick_voice(detected_lang: Optional[str], speech_text: str) -> str:
    """
    Select TTS voice. Always Chinese since we configured to reply in Chinese.
    Uses Cantonese voice if text looks Cantonese, otherwise Mandarin.
    """
    # Prefer Cantonese voice if text looks Cantonese
    if _looks_cantonese(speech_text):
        return "zh-HK-HiuGaamNeural"
    
    # Default to Mandarin Chinese
    return "zh-CN-XiaoxiaoNeural"


# =============================================================================
# JSON Parsing & Action Normalization
# =============================================================================
def safe_json_loads(maybe_json_text: str) -> Dict[str, Any]:
    """
    Parse JSON with fallback to extract outer {...} block.
    Handles cases where model wraps JSON in markdown or extra text.
    """
    try:
        return json.loads(maybe_json_text)
    except json.JSONDecodeError:
        # Try to extract JSON object from text
        s = maybe_json_text.find("{")
        e = maybe_json_text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(maybe_json_text[s : e + 1])
        raise


def normalize_aux(raw_aux: Any) -> AuxiliaryCommand:
    """
    Normalize auxiliary command to ensure it's valid.
    Hard clamps action to ALLOWED_ACTIONS for robot safety.
    """
    action = DEFAULT_ACTION
    params: Dict[str, Any] = {}

    if isinstance(raw_aux, str):
        action = raw_aux
    elif isinstance(raw_aux, dict):
        action = raw_aux.get("action", DEFAULT_ACTION)
        params_val = raw_aux.get("params", {}) or {}
        params = params_val if isinstance(params_val, dict) else {}
    elif raw_aux is None:
        action = DEFAULT_ACTION
    else:
        action = DEFAULT_ACTION

    # Normalize and validate action
    action = str(action).strip().upper()
    if action not in ALLOWED_ACTIONS:
        print(f"[WARN] Unknown action '{action}' requested, defaulting to {DEFAULT_ACTION}")
        action = DEFAULT_ACTION
        params = {}

    return AuxiliaryCommand(action=action, params=params)


# =============================================================================
# Conversation History & Prompt Building
# =============================================================================
def format_history(history: List[Dict[str, str]]) -> str:
    """Format conversation history for the prompt."""
    lines: List[str] = []
    for m in history:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").replace("\n", " ").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(No previous conversation)"


def build_prompt(
    user_text: str,
    robot_status: Dict[str, Any],
    history: List[Dict[str, str]],
) -> str:
    """Build the prompt for the LLM with all context."""
    actions_text = "\n".join([f"- {k}: {v}" for k, v in ALLOWED_ACTIONS.items()])
    history_text = format_history(history)
    status_json = json.dumps(robot_status, ensure_ascii=False, indent=2)

    return f"""You are a responsive robot assistant.

OUTPUT FORMAT (JSON only, no markdown):
{{"speech": "短回應", "auxiliary": {{"action": "ACTION_NAME", "params": {{}}}}}}

CRITICAL RULES:
1. "speech" MUST be under 20 words. Be brief and direct.
2. Reply in Chinese (廣東話 or 普通話 matching user).
3. Pick action from: {", ".join(ALLOWED_ACTIONS.keys())}
4. DO NOT explain yourself. Just answer.
5. No greetings in every response. Get to the point.

History: {history_text}
User: {user_text}

JSON:""".strip()


def parse_robot_status(robot_status_json: str) -> Dict[str, Any]:
    """Parse robot status JSON, with graceful error handling."""
    if not robot_status_json:
        return {}
    try:
        obj = json.loads(robot_status_json)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        return {"raw": robot_status_json}


# =============================================================================
# Core Processing Functions
# =============================================================================
def stt_transcribe_sync(audio_path: str) -> Tuple[str, Optional[str]]:
    """Synchronous STT transcription (runs in thread pool)."""
    model = get_stt_model()
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    lang = getattr(info, "language", None)
    return text, lang


async def call_ollama(prompt: str) -> str:
    """Call Ollama API asynchronously."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    # First request can take a long time as Ollama loads the model into GPU
    # 300s timeout for first load, subsequent requests are much faster
    timeout = httpx.Timeout(300.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


# Sentence boundary punctuation for chunked streaming
SENTENCE_BOUNDARIES = {',', '.', '!', '?', '，', '。', '！', '？', '；', '：', '、'}


async def stream_ollama_sentences(prompt: str):
    """
    Stream LLM response and yield complete sentences/chunks.
    
    Yields tuples of (sentence_text, is_final).
    This allows the caller to start TTS as soon as a sentence is complete.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": True,  # Enable streaming
        "options": {
            "temperature": 0.2,
        },
    }
    
    timeout = httpx.Timeout(300.0, connect=30.0)
    sentence_buffer = ""
    full_response = ""
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                token = chunk_data.get("response", "")
                done = chunk_data.get("done", False)
                
                full_response += token
                sentence_buffer += token
                
                # Check if we hit a sentence boundary
                if not done and any(punc in token for punc in SENTENCE_BOUNDARIES):
                    # Yield the current sentence chunk
                    if sentence_buffer.strip():
                        yield (sentence_buffer.strip(), False)
                        sentence_buffer = ""
                
                if done:
                    # Yield any remaining text
                    if sentence_buffer.strip():
                        yield (sentence_buffer.strip(), True)
                    break


async def tts_to_mp3(text: str, voice: str, out_path: Path) -> None:
    """Generate TTS audio and save to MP3."""
    # Ensure text is not empty (Edge TTS fails on empty text)
    if not text or not text.strip():
        text = "..."  # Fallback to ellipsis
    
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(out_path))
    except Exception as e:
        print(f"[TTS] Error generating audio: {e}")
        # Create a short silence file as fallback
        communicate = edge_tts.Communicate(text="...", voice=voice)
        await communicate.save(str(out_path))


# =============================================================================
# Session Memory Management
# =============================================================================
async def get_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session."""
    async with memory_lock:
        dq = sessions.get(session_id)
        if dq is None:
            dq = deque(maxlen=2 * K_TURNS)
            sessions[session_id] = dq
        return list(dq)


async def append_memory(session_id: str, role: str, content: str) -> None:
    """Append a message to session memory."""
    content = (content or "").strip()
    if not content:
        return
    async with memory_lock:
        dq = sessions.get(session_id)
        if dq is None:
            dq = deque(maxlen=2 * K_TURNS)
            sessions[session_id] = dq
        dq.append({"role": role, "content": content})


# =============================================================================
# API Routes
# =============================================================================
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True, "model": OLLAMA_MODEL, "whisper": WHISPER_MODEL}


@app.get("/actions")
async def get_actions():
    """Get list of allowed actions."""
    return {"actions": ALLOWED_ACTIONS}


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset/clear a session's conversation history."""
    async with memory_lock:
        sessions.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information and history."""
    history = await get_history(session_id)
    return SessionInfo(
        session_id=session_id,
        message_count=len(history),
        messages=history
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    async with memory_lock:
        return {
            "sessions": [
                {"session_id": sid, "message_count": len(dq)}
                for sid, dq in sessions.items()
            ]
        }


@app.get("/audio/{audio_id}.mp3")
async def get_audio(audio_id: str):
    """Serve generated audio file."""
    # Sanitize audio_id to prevent path traversal
    if not re.match(r"^[a-f0-9]+$", audio_id):
        raise HTTPException(status_code=400, detail="Invalid audio ID")
    
    path = OUTPUT_DIR / f"{audio_id}.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename="response.mp3")


@app.post("/process_voice", response_model=ProcessVoiceResponse)
async def process_voice(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    session_id: str = Form("robot-1", description="Session ID for conversation continuity"),
    robot_status_json: str = Form("{}", description="JSON string of robot's current status"),
):
    """
    Main endpoint: Process voice input and return speech response with action.
    
    1. Receive audio file
    2. Transcribe with Whisper (STT)
    3. Build prompt with conversation history
    4. Get LLM response via Ollama
    5. Validate and normalize action
    6. Generate TTS audio (Edge-TTS)
    7. Return JSON response with audio URL
    """
    # 1) Save uploaded audio to a unique temp file
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    try:
        # 2) STT - Speech to Text
        print(f"[{session_id}] Processing audio: {file.filename}")
        async with stt_sem:
            user_text, detected_lang = await asyncio.to_thread(stt_transcribe_sync, input_path)

        user_text = (user_text or "").strip()
        print(f"[{session_id}] Transcribed ({detected_lang}): {user_text}")

        # Handle empty transcription
        if not user_text:
            user_text = "(silence or unclear audio)"

        # 3) Build prompt with memory
        robot_status = parse_robot_status(robot_status_json)
        history = await get_history(session_id)
        prompt = build_prompt(user_text=user_text, robot_status=robot_status, history=history)

        # 4) LLM - Get response from Ollama
        print(f"[{session_id}] Calling Ollama ({OLLAMA_MODEL})...")
        async with llm_sem:
            llm_raw = await call_ollama(prompt)

        print(f"[{session_id}] LLM response: {llm_raw[:200]}...")

        # 5) Parse/normalize LLM JSON
        try:
            obj = safe_json_loads(llm_raw)
        except Exception as e:
            print(f"[{session_id}] JSON parse error: {e}, using fallback")
            obj = {
                "speech": "Sorry, I had trouble understanding. Could you say that again?",
                "auxiliary": {"action": "NONE", "params": {}},
            }

        speech = str(obj.get("speech") or "").strip()
        aux = normalize_aux(obj.get("auxiliary", None))

        # Ensure we have something to say
        if not speech:
            speech = "I'm not sure how to respond to that."
            aux = AuxiliaryCommand(action="NONE", params={})

        print(f"[{session_id}] Speech: {speech}")
        print(f"[{session_id}] Action: {aux.action}")

        # 6) Update memory (store only text, not actions)
        await append_memory(session_id, "user", user_text)
        await append_memory(session_id, "assistant", speech)

        # 7) TTS - Text to Speech
        audio_id = uuid.uuid4().hex
        out_path = OUTPUT_DIR / f"{audio_id}.mp3"
        voice = pick_voice(detected_lang, speech)
        
        print(f"[{session_id}] Generating TTS with voice: {voice}")
        async with tts_sem:
            await tts_to_mp3(speech, voice, out_path)

        return ProcessVoiceResponse(
            session_id=session_id,
            user_text=user_text,
            speech=speech,
            auxiliary=aux,
            detected_language=detected_lang,
            audio_id=audio_id,
            audio_url=f"/audio/{audio_id}.mp3",
        )

    finally:
        # Clean up temp file
        try:
            os.remove(input_path)
        except OSError:
            pass


@app.post("/process_text", response_model=ProcessVoiceResponse)
async def process_text(
    text: str = Form(..., description="Text input (for testing without audio)"),
    session_id: str = Form("robot-1", description="Session ID"),
    robot_status_json: str = Form("{}", description="Robot status JSON"),
    language: str = Form("en", description="Language hint for TTS"),
):
    """
    Text-only endpoint for testing without audio.
    Useful for development and debugging.
    """
    user_text = text.strip()
    
    if not user_text:
        raise HTTPException(status_code=400, detail="Text input is required")

    # Build prompt
    robot_status = parse_robot_status(robot_status_json)
    history = await get_history(session_id)
    prompt = build_prompt(user_text=user_text, robot_status=robot_status, history=history)

    # LLM
    async with llm_sem:
        llm_raw = await call_ollama(prompt)

    # Parse
    try:
        obj = safe_json_loads(llm_raw)
    except Exception:
        obj = {
            "speech": "Sorry, I had trouble processing that.",
            "auxiliary": {"action": "NONE", "params": {}},
        }

    speech = str(obj.get("speech") or "").strip()
    aux = normalize_aux(obj.get("auxiliary", None))

    if not speech:
        speech = "I'm not sure how to respond."
        aux = AuxiliaryCommand(action="NONE", params={})

    # Update memory
    await append_memory(session_id, "user", user_text)
    await append_memory(session_id, "assistant", speech)

    # TTS
    audio_id = uuid.uuid4().hex
    out_path = OUTPUT_DIR / f"{audio_id}.mp3"
    voice = pick_voice(language, speech)
    
    async with tts_sem:
        await tts_to_mp3(speech, voice, out_path)

    return ProcessVoiceResponse(
        session_id=session_id,
        user_text=user_text,
        speech=speech,
        auxiliary=aux,
        detected_language=language,
        audio_id=audio_id,
        audio_url=f"/audio/{audio_id}.mp3",
    )


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================
class StreamingTranscriber:
    """
    Handles streaming audio transcription.
    
    Accumulates audio chunks and runs Whisper periodically to provide
    partial transcriptions while recording continues.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.last_transcription = ""
        self.last_process_time = 0
        self.min_process_interval = 0.5  # Process at most every 0.5s
        self.min_audio_for_process = 0.8  # Need at least 0.8s of audio
    
    def add_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to buffer."""
        self.audio_buffer.append(audio_data)
    
    def get_audio_duration(self) -> float:
        """Get total buffered audio duration in seconds."""
        if not self.audio_buffer:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        return total_samples / self.sample_rate
    
    def get_complete_audio(self) -> np.ndarray:
        """Get all buffered audio as single array."""
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_buffer)
    
    def should_process(self) -> bool:
        """Check if we should run transcription now."""
        now = time.time()
        duration = self.get_audio_duration()
        
        # Need minimum audio and time since last process
        if duration < self.min_audio_for_process:
            return False
        if now - self.last_process_time < self.min_process_interval:
            return False
        return True
    
    def transcribe_partial(self) -> Tuple[str, Optional[str]]:
        """
        Run transcription on current buffer.
        Returns (text, detected_language)
        """
        audio = self.get_complete_audio()
        if len(audio) == 0:
            return "", None
        
        self.last_process_time = time.time()
        
        # Run whisper - limit to English and Chinese only
        model = get_stt_model()
        segments, info = model.transcribe(
            audio,
            beam_size=3,  # Faster for partial
            vad_filter=True,
            language=None,  # Auto-detect first
        )
        
        text = "".join(seg.text for seg in segments).strip()
        detected_lang = getattr(info, "language", None)
        
        # If detected language is not en/zh, default to zh
        if detected_lang not in ("en", "zh", "yue"):
            detected_lang = "zh"
        
        # Filter out common Whisper hallucinations
        text = self._filter_hallucinations(text)
        
        self.last_transcription = text
        return text, detected_lang
    
    def _filter_hallucinations(self, text: str) -> str:
        """Filter out known Whisper hallucination patterns."""
        if not text:
            return text
        
        # Common hallucination phrases (Whisper artifacts from training data)
        hallucinations = [
            "На этом всё",  # Russian: "That's all"
            "До новых встреч",  # Russian: "See you next time"
            "Подписывайтесь",  # Russian: "Subscribe"
            "Thanks for watching",
            "Please subscribe",
            "Like and subscribe",
            "See you next time",
            "Thank you for watching",
            "字幕",  # Chinese: "subtitles"
            "謝謝收看",  # Chinese: "Thanks for watching"
            "請訂閱",  # Chinese: "Please subscribe"
            "ご視聴ありがとう",  # Japanese: "Thanks for watching"
        ]
        
        for phrase in hallucinations:
            if phrase.lower() in text.lower():
                print(f"[FILTER] Removed hallucination: {text}")
                return ""
        
        # If text is very short and looks like noise
        if len(text) < 3:
            return ""
        
        return text
    
    def transcribe_final(self) -> Tuple[str, Optional[str]]:
        """
        Final transcription with higher quality settings.
        Limited to English and Chinese only.
        """
        audio = self.get_complete_audio()
        if len(audio) == 0:
            return "", None
        
        model = get_stt_model()
        segments, info = model.transcribe(
            audio,
            beam_size=5,  # Higher quality for final
            vad_filter=True,
            language=None,  # Auto-detect
        )
        
        text = "".join(seg.text for seg in segments).strip()
        detected_lang = getattr(info, "language", None)
        
        # If detected language is not en/zh, default to zh
        if detected_lang not in ("en", "zh", "yue"):
            detected_lang = "zh"
        
        # Filter hallucinations in final too
        text = self._filter_hallucinations(text)
        
        return text, detected_lang
    
    def clear(self):
        """Clear buffer."""
        self.audio_buffer = []
        self.last_transcription = ""
        self.last_process_time = 0


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio.
    
    Protocol:
    1. Client connects
    2. Client sends JSON: {"type": "start", "session_id": "...", "robot_status": {...}}
    3. Client sends binary audio chunks (float32, 16kHz, mono)
    4. Server sends JSON: {"type": "partial", "text": "..."} periodically
    5. Client sends JSON: {"type": "end"} when done speaking
    6. Server sends JSON: {"type": "result", ...} with final response
    
    This overlaps STT with recording, reducing total latency.
    """
    await websocket.accept()
    
    transcriber = StreamingTranscriber(sample_rate=16000)
    session_id = "ws-default"
    robot_status = {}
    is_recording = False
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                # JSON control message
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "start":
                    # Start new recording session
                    session_id = data.get("session_id", "ws-default")
                    robot_status = data.get("robot_status", {})
                    transcriber.clear()
                    is_recording = True
                    await websocket.send_json({"type": "started", "session_id": session_id})
                    print(f"[WS] Stream started: {session_id}")
                
                elif msg_type == "end":
                    # End of speech - process final result
                    is_recording = False
                    
                    # Check if we have audio
                    audio_duration = transcriber.get_audio_duration()
                    print(f"[WS] Stream ended, audio duration: {audio_duration:.1f}s")
                    
                    if audio_duration < 0.3:
                        print(f"[WS] Audio too short, skipping")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Audio too short. Please speak longer."
                        })
                        transcriber.clear()
                        continue
                    
                    # ========== TIMING: STT ==========
                    stt_start = time.time()
                    print(f"[WS] Processing final transcription...")
                    try:
                        async with stt_sem:
                            user_text, detected_lang = await asyncio.to_thread(
                                transcriber.transcribe_final
                            )
                    except Exception as e:
                        print(f"[WS] STT error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Speech recognition failed: {e}"
                        })
                        transcriber.clear()
                        continue
                    stt_time = time.time() - stt_start
                    
                    print(f"[WS] Final transcription ({detected_lang}): {user_text}")
                    print(f"[TIMING] STT: {stt_time:.2f}s")
                    
                    if not user_text.strip():
                        user_text = "(silence or unclear audio)"
                    
                    # ========== TIMING: LLM ==========
                    llm_start = time.time()
                    history = await get_history(session_id)
                    prompt = build_prompt(
                        user_text=user_text,
                        robot_status=robot_status,
                        history=history
                    )
                    
                    async with llm_sem:
                        llm_raw = await call_ollama(prompt)
                    llm_time = time.time() - llm_start
                    print(f"[TIMING] LLM: {llm_time:.2f}s")
                    
                    # Parse LLM response
                    try:
                        obj = safe_json_loads(llm_raw)
                    except Exception:
                        obj = {
                            "speech": "對不起，我聽不太清楚。",
                            "auxiliary": {"action": "NONE", "params": {}},
                        }
                    
                    speech = str(obj.get("speech") or "").strip()
                    aux = normalize_aux(obj.get("auxiliary", None))
                    
                    if not speech:
                        speech = "我不太確定怎麼回應。"
                        aux = AuxiliaryCommand(action="NONE", params={})
                    
                    # Update memory
                    await append_memory(session_id, "user", user_text)
                    await append_memory(session_id, "assistant", speech)
                    
                    # ========== TIMING: TTS ==========
                    tts_start = time.time()
                    audio_id = uuid.uuid4().hex
                    out_path = OUTPUT_DIR / f"{audio_id}.mp3"
                    voice = pick_voice(detected_lang, speech)
                    
                    async with tts_sem:
                        await tts_to_mp3(speech, voice, out_path)
                    tts_time = time.time() - tts_start
                    print(f"[TIMING] TTS: {tts_time:.2f}s")
                    
                    # Total time
                    total_time = stt_time + llm_time + tts_time
                    print(f"[TIMING] TOTAL (STT+LLM+TTS): {total_time:.2f}s")
                    
                    # Send final result with timing info
                    await websocket.send_json({
                        "type": "result",
                        "session_id": session_id,
                        "user_text": user_text,
                        "speech": speech,
                        "auxiliary": {"action": aux.action, "params": aux.params},
                        "detected_language": detected_lang,
                        "audio_id": audio_id,
                        "audio_url": f"/audio/{audio_id}.mp3",
                        "timing": {
                            "stt_seconds": round(stt_time, 2),
                            "llm_seconds": round(llm_time, 2),
                            "tts_seconds": round(tts_time, 2),
                            "total_seconds": round(total_time, 2),
                        }
                    })
                    
                    print(f"[WS] Result sent: {speech[:50]}...")
                    transcriber.clear()
                
                elif msg_type == "cancel":
                    # Cancel current recording
                    transcriber.clear()
                    is_recording = False
                    await websocket.send_json({"type": "cancelled"})
            
            elif "bytes" in message and is_recording:
                # Binary audio data (float32 samples)
                audio_bytes = message["bytes"]
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                transcriber.add_chunk(audio_data)
                
                # Check if we should run partial transcription
                if transcriber.should_process():
                    async with stt_sem:
                        partial_text, _ = await asyncio.to_thread(
                            transcriber.transcribe_partial
                        )
                    if partial_text:
                        await websocket.send_json({
                            "type": "partial",
                            "text": partial_text
                        })
                        print(f"[WS] Partial: {partial_text}")
    
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {session_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


@app.websocket("/ws/stream-fast")
async def websocket_stream_fast(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio with CHUNKED response.
    
    This endpoint uses sentence-streaming to reduce perceived latency:
    - LLM streams its output
    - As soon as a sentence boundary is detected, TTS runs on that chunk
    - Audio chunk is sent to client immediately
    - Client can start playing while server generates more audio
    
    Protocol:
    1. Client connects
    2. Client sends JSON: {"type": "start", "session_id": "...", "robot_status": {...}}
    3. Client sends binary audio chunks (float32, 16kHz, mono)
    4. Server sends JSON: {"type": "partial", "text": "..."} periodically
    5. Client sends JSON: {"type": "end"} when done speaking
    6. Server sends multiple JSON: {"type": "audio_chunk", "chunk_index": N, "text": "...", "audio_url": "..."}
    7. Server sends JSON: {"type": "result", ...} with final complete response
    """
    await websocket.accept()
    
    transcriber = StreamingTranscriber(sample_rate=16000)
    session_id = "ws-fast-default"
    robot_status = {}
    is_recording = False
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "start":
                    session_id = data.get("session_id", "ws-fast-default")
                    robot_status = data.get("robot_status", {})
                    transcriber.clear()
                    is_recording = True
                    await websocket.send_json({"type": "started", "session_id": session_id})
                    print(f"[WS-FAST] Stream started: {session_id}")
                
                elif msg_type == "end":
                    is_recording = False
                    audio_duration = transcriber.get_audio_duration()
                    print(f"[WS-FAST] Stream ended, audio duration: {audio_duration:.1f}s")
                    
                    if audio_duration < 0.3:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Audio too short. Please speak longer."
                        })
                        transcriber.clear()
                        continue
                    
                    # ========== STT ==========
                    stt_start = time.time()
                    try:
                        async with stt_sem:
                            user_text, detected_lang = await asyncio.to_thread(
                                transcriber.transcribe_final
                            )
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"STT failed: {e}"})
                        transcriber.clear()
                        continue
                    stt_time = time.time() - stt_start
                    print(f"[WS-FAST] STT: {user_text} ({stt_time:.2f}s)")
                    
                    if not user_text.strip():
                        user_text = "(silence or unclear audio)"
                    
                    # ========== LLM + TTS (Streamed/Chunked) ==========
                    history = await get_history(session_id)
                    prompt = build_prompt(
                        user_text=user_text,
                        robot_status=robot_status,
                        history=history
                    )
                    
                    llm_start = time.time()
                    chunk_index = 0
                    full_speech = ""
                    full_llm_response = ""
                    voice = pick_voice(detected_lang, "")
                    first_audio_time = None
                    tts_total_time = 0
                    
                    # Stream LLM and send audio chunks as sentences complete
                    async with llm_sem:
                        try:
                            async for sentence, is_final in stream_ollama_sentences(prompt):
                                full_llm_response += sentence
                                
                                # For streaming mode, we can't parse JSON until the end
                                # So we just accumulate the text chunks
                                # We'll extract speech from JSON at the end
                                
                                # Skip JSON structure tokens
                                if sentence.startswith('{') or sentence.startswith('"speech"'):
                                    continue
                                
                                # Clean up sentence (remove JSON artifacts)
                                clean_sentence = sentence
                                for artifact in ['"', ':', '{', '}', 'speech', 'auxiliary', 'action', 'params', 'NONE']:
                                    clean_sentence = clean_sentence.replace(artifact, '')
                                clean_sentence = clean_sentence.strip()
                                
                                if not clean_sentence or len(clean_sentence) < 2:
                                    continue
                                
                                full_speech += clean_sentence + " "
                                
                                # Generate TTS for this chunk
                                tts_chunk_start = time.time()
                                audio_id = f"{uuid.uuid4().hex}"
                                out_path = OUTPUT_DIR / f"{audio_id}.mp3"
                                
                                async with tts_sem:
                                    await tts_to_mp3(clean_sentence, voice, out_path)
                                
                                tts_chunk_time = time.time() - tts_chunk_start
                                tts_total_time += tts_chunk_time
                                
                                if first_audio_time is None:
                                    first_audio_time = time.time() - llm_start
                                    print(f"[WS-FAST] First audio ready in {first_audio_time:.2f}s (chunk: {clean_sentence[:30]}...)")
                                
                                # Send audio chunk to client
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "chunk_index": chunk_index,
                                    "text": clean_sentence,
                                    "audio_url": f"/audio/{audio_id}.mp3",
                                })
                                chunk_index += 1
                                
                                if is_final:
                                    break
                        except Exception as e:
                            print(f"[WS-FAST] Streaming LLM error: {e}")
                            # Fall back to regular call
                            full_llm_response = await call_ollama(prompt)
                    
                    llm_time = time.time() - llm_start
                    
                    # Parse the complete LLM JSON response
                    try:
                        obj = safe_json_loads(full_llm_response)
                    except Exception:
                        obj = {
                            "speech": full_speech.strip() or "對不起，出了點問題。",
                            "auxiliary": {"action": "NONE", "params": {}},
                        }
                    
                    final_speech = str(obj.get("speech") or full_speech or "").strip()
                    aux = normalize_aux(obj.get("auxiliary", None))
                    
                    if not final_speech:
                        final_speech = "我不太確定怎麼回應。"
                    
                    # Update memory
                    await append_memory(session_id, "user", user_text)
                    await append_memory(session_id, "assistant", final_speech)
                    
                    total_time = stt_time + llm_time
                    print(f"[WS-FAST] TIMING: STT={stt_time:.2f}s LLM+TTS={llm_time:.2f}s FirstAudio={first_audio_time or 0:.2f}s")
                    
                    # Send final result
                    await websocket.send_json({
                        "type": "result",
                        "session_id": session_id,
                        "user_text": user_text,
                        "speech": final_speech,
                        "auxiliary": {"action": aux.action, "params": aux.params},
                        "detected_language": detected_lang,
                        "chunks_sent": chunk_index,
                        "timing": {
                            "stt_seconds": round(stt_time, 2),
                            "llm_tts_seconds": round(llm_time, 2),
                            "first_audio_seconds": round(first_audio_time or llm_time, 2),
                            "total_seconds": round(total_time, 2),
                        }
                    })
                    
                    print(f"[WS-FAST] Complete: {final_speech[:50]}... ({chunk_index} chunks)")
                    transcriber.clear()
                
                elif msg_type == "cancel":
                    transcriber.clear()
                    is_recording = False
                    await websocket.send_json({"type": "cancelled"})
            
            elif "bytes" in message and is_recording:
                audio_bytes = message["bytes"]
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                transcriber.add_chunk(audio_data)
                
                if transcriber.should_process():
                    async with stt_sem:
                        partial_text, _ = await asyncio.to_thread(
                            transcriber.transcribe_partial
                        )
                    if partial_text:
                        await websocket.send_json({"type": "partial", "text": partial_text})
    
    except WebSocketDisconnect:
        print(f"[WS-FAST] Client disconnected: {session_id}")
    except Exception as e:
        print(f"[WS-FAST] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# =============================================================================
# Startup/Shutdown Events
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup - preload models."""
    print("=" * 60)
    print("Robot Voice Brain Server Starting...")
    print(f"  Ollama URL: {OLLAMA_URL}")
    print(f"  Ollama Model: {OLLAMA_MODEL}")
    print(f"  Whisper Model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    print(f"  Conversation Memory: {K_TURNS} turns")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Allowed Actions: {len(ALLOWED_ACTIONS)}")
    print("=" * 60)
    
    # 1) Preload Whisper STT model
    print("\n[STARTUP] Loading Whisper STT model...")
    try:
        model = get_stt_model()
        print(f"[STARTUP] ✓ Whisper model loaded: {WHISPER_MODEL}")
    except Exception as e:
        print(f"[STARTUP] ✗ Failed to load Whisper: {e}")
        raise
    
    # 2) Warm up Ollama LLM (loads model into GPU)
    print(f"\n[STARTUP] Warming up Ollama LLM ({OLLAMA_MODEL})...")
    print("[STARTUP] This may take 30-60 seconds on first run...")
    try:
        warmup_payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Say hi",
            "stream": False,
            "options": {"num_predict": 5},  # Very short response for warmup
        }
        timeout = httpx.Timeout(300.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(OLLAMA_URL, json=warmup_payload)
            r.raise_for_status()
            data = r.json()
            response = data.get("response", "")
            print(f"[STARTUP] ✓ Ollama ready: {repr(response[:50])}")
    except httpx.ConnectError:
        print(f"[STARTUP] ✗ Cannot connect to Ollama at {OLLAMA_URL}")
        print("[STARTUP]   Make sure Ollama is running: ollama serve")
        raise
    except Exception as e:
        print(f"[STARTUP] ✗ Ollama warmup failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✓ Server ready! All models loaded.")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Server shutting down...")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    # Keep workers=1 for GPU Whisper (multiple workers will duplicate VRAM usage)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
