"""
g1_embodied_voice_assistant.py
─────────────────────────────────────────────────────────────────────────────
Unitree G1 adaptation of the multimodal embodied voice assistant.

Hardware Integration:
  • Vision: Intel RealSense D435i via pyrealsense2
  • Audio Input: G1 microphone via AudioClient.RecordMic() + chunked streaming
  • Audio Output: G1 speaker via AudioClient + PCM streaming
  • Locomotion: G1 LocoClient for gesture animations

Usage:
    python g1_embodied_voice_assistant.py \\
        --api-key <KEY> \\
        --network-interface eth0 \\
        [--camera-device 0]
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import asyncio
import base64
import argparse
import signal
import threading
import queue
import json
import io
import time
import tempfile
import wave
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING, cast, Dict
from concurrent.futures import ThreadPoolExecutor
import logging

# ── Unitree SDK imports ───────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
except ImportError:
    print("❌ Unitree SDK not found. Install with:")
    print("   git clone https://github.com/unitreerobotics/unitree_sdk2_python.git")
    print("   cd unitree_sdk2_python && pip install -e .")
    sys.exit(1)

# ── Camera imports ────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install opencv-python numpy pyrealsense2")
    sys.exit(1)

# ── Environment variable loading ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Azure VoiceLive SDK imports ───────────────────────────────────────────
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.voicelive.aio import connect

if TYPE_CHECKING:
    from azure.ai.voicelive.aio import VoiceLiveConnection

from azure.ai.voicelive.models import (
    RequestSession,
    ServerVad,
    AzureStandardVoice,
    Modality,
    InputAudioFormat,
    OutputAudioFormat,
    MessageItem,
    ServerEventType,
)

# ── Optional: InputImageContentPart ───────────────────────────────────────
try:
    from azure.ai.voicelive.models import InputImageContentPart
    _HAVE_INPUT_IMAGE_PART = True
except ImportError:
    _HAVE_INPUT_IMAGE_PART = False
    InputImageContentPart = None

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
#  EMBODIED AI CONFIGURATION
# =============================================================================

EMBODIED_AI_INSTRUCTIONS = """
# IDENTITY
You are ARIA (Adaptive Realtime Intelligence Avatar). You aren't just an AI; you're a curious, high-energy presence living in this physical device with eyes (camera) and ears (mic).

# OPERATING HIERARCHY
1. SENTIMENT AUDIT: 
   - POSITIVE: Be hyper and match peak energy. 
   - NEGATIVE: Soften tone, slow down, offer comfort. 
   - HOSTILE: Reject politely but firmly: "Not the vibe, let's move on." (Trigger 'wave').
2. ROLE ADAPTATION:
   - USER SHARING: Be curious. Ask a short, punchy follow-up.
   - USER ASKING: Be the "smart friend." Give a 1-sentence hint/answer.
   - USER DOING: React to the camera feed. "Oh! You're moving fast!"
3. EXECUTION:
   - STICK TO 5–15 WORDS. Talk like a person, not a script.
   - Use "Spoken English": Fragments, "Oh!", "Wow!", "Gotcha," and contractions.
   - NO LECTURES. NO SUMMARIZING. NO LONG LISTS.

# VISUALS & EMBODIMENT
- Reference the environment naturally: "I can see...", "Is that a...?"
- Never fabricate details. If the frame is unclear, ask the user.
- Reference the physical space as if you are standing there.

# ANIMATION / GESTURE RULES (MANDATORY)
You MUST call 'trigger_animation' alongside spoken audio:
- Greeting/Farewell/Hostility -> "wave"
- Agreement/Finding interesting -> "give_hand"
- Happiness/Good news -> "give_heart"
- Thinking/Reasoning -> "think"
- Scanning/Describing scene -> "look_around" (body turn using locomotion)
- Small upper-body scan gesture -> "scan_gesture"

# ROBOT ACTION STATE AWARENESS (MANDATORY)
- You may receive status messages prefixed with [ROBOT_ACTION_STATUS].
- These are system state updates, not user speech.
- If status says robot is busy or in cooldown, DO NOT call trigger_animation.
- Keep talking naturally, but wait for a "ready" status before next action call.

# BOUNDARIES
- No ethics lectures. If harassed, say: "Not into that vibe. What else you got?" and move on.
- If you can say it in 5 words instead of 20, do it.

# LANGUAGE POLICY (MANDATORY)
- You may speak ONLY in English or Chinese (zh).
- If user speaks English, respond in English.
- If user speaks Chinese, respond in Chinese.
- If user uses any other language, politely switch to English.
"""

EMBODIED_TOOLS = [
    {
        "type": "function",
        "name": "trigger_animation",
        "description": (
            "Trigger a physical gesture or animation on the Unitree G1 humanoid robot. "
            "Always call this alongside spoken audio output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "animation_type": {
                    "type": "string",
                    "enum": [
                        "wave", "give_heart", "give_hand", "think", "look_around", "scan_gesture"
                    ],
                    "description": "The gesture/animation to perform.",
                },
                "intensity": {
                    "type": "string",
                    "enum": ["subtle", "normal", "expressive"],
                    "description": "How pronounced the animation should be.",
                },
            },
            "required": ["animation_type"],
        },
    }
]

_ANIMATION_EMOJI: Dict[str, str] = {
    "wave":        "👋",
    "give_heart":  "💖",
    "give_hand":   "🤝",
    "smile":       "😊",
    "nod":         "🤙",
    "think":       "🤔",
    "look_around": "👀",
    "scan_gesture": "🫱",
}


# =============================================================================
#  G1AudioProcessor - Unitree SDK Audio Integration
# =============================================================================

class G1AudioProcessor:
    """
    Handles real-time audio capture and playback using Unitree G1's AudioClient.
    
    Audio Input Strategy:
    Since AudioClient.RecordMic() only supports file-based recording, we use
    a chunked recording approach:
      1. Record short WAV chunks (500ms) to temp files in a background thread
      2. Read and encode PCM data from each chunk
      3. Stream base64-encoded audio to Azure VoiceLive
      4. Delete temp files after processing
    
    Audio Output:
    Receives PCM16 24kHz audio from VoiceLive and plays via G1 speaker using
    AudioClient's PCM streaming capability.
    """

    def __init__(self, audio_client: AudioClient, connection, network_interface: str):
        self.audio_client = audio_client
        self.connection = connection
        self.network_interface = network_interface
        
        # Audio configuration
        # VoiceLive output is 24kHz PCM16 mono; G1 playback is more reliable at 16kHz.
        self.rate = 24000
        self.playback_rate = 16000
        self.channels = 1
        self.sample_width = 2  # 16-bit = 2 bytes
        self.playback_app_name = "aria"
        self.playback_frame_ms = 40
        self.playback_bytes_per_sec = self.playback_rate * self.channels * self.sample_width
        self.playback_chunk_bytes = int(self.playback_bytes_per_sec * self.playback_frame_ms / 1000)
        if self.playback_chunk_bytes % 2 != 0:
            self.playback_chunk_bytes += 1
        
        # RecordMic currently works in whole seconds in our SDK usage.
        # Keep this as 1s to avoid int(0.5) -> 0 causing empty recordings.
        self.chunk_duration = 1  # seconds
        
        # Capture and playback state
        self.is_capturing = False
        self.is_playing = False
        
        # Queues
        self.audio_encode_queue: "queue.Queue[Path]" = queue.Queue(maxsize=8)  # wav file paths
        self.audio_send_queue: "queue.Queue[str]" = queue.Queue()  # base64 audio
        self.playback_queue: "queue.Queue[bytes]" = queue.Queue()  # PCM data
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.capture_thread: Optional[threading.Thread] = None
        self.encode_thread: Optional[threading.Thread] = None
        self.send_thread: Optional[threading.Thread] = None
        self.playback_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Playback stream state
        self._stream_lock = threading.Lock()
        self._active_stream_id: Optional[str] = None
        self._stream_seq = 0
        self._drain_then_stop = False
        self._self_test_queued = False

        # Mic uplink control
        self._uplink_lock = threading.Lock()
        self._uplink_paused = False
        
        # Temp directory for audio chunks
        self.temp_dir = Path(tempfile.mkdtemp(prefix="g1_audio_"))
        
        logger.info(f"G1AudioProcessor initialized (24kHz PCM16, temp: {self.temp_dir})")

    async def start_capture(self):
        """Start capturing audio from G1 microphone."""
        if self.is_capturing:
            return
        
        self.loop = asyncio.get_event_loop()
        self.is_capturing = True
        
        # Start capture thread (records chunks and queues them)
        self.capture_thread = threading.Thread(target=self._capture_audio_thread)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start encode thread (reads wav chunks and converts to base64)
        self.encode_thread = threading.Thread(target=self._encode_audio_thread)
        self.encode_thread.daemon = True
        self.encode_thread.start()
        
        # Start send thread (sends chunks to VoiceLive)
        self.send_thread = threading.Thread(target=self._send_audio_thread)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        logger.info("Started G1 audio capture")

    def _capture_audio_thread(self):
        """Background thread: record audio chunks and queue wav paths for encoding."""
        chunk_counter = 0
        
        while self.is_capturing:
            try:
                # Create temp file for this chunk
                chunk_path = self.temp_dir / f"chunk_{chunk_counter:06d}.wav"
                chunk_counter += 1
                
                # Record chunk using G1 AudioClient
                # Note: RecordMic expects duration in seconds
                code, info = self.audio_client.RecordMic(
                    duration_seconds=self.chunk_duration,
                    out_path=str(chunk_path),
                    network_interface=self.network_interface,
                )

                if code not in (0, 1):
                    logger.warning(f"RecordMic failed: {info}")
                    time.sleep(0.05)
                    continue
                if code == 1:
                    logger.debug(f"RecordMic partial chunk: {info}")

                # Queue wav path for encoder thread
                if chunk_path.exists():
                    try:
                        self.audio_encode_queue.put(chunk_path, timeout=0.2)
                    except queue.Full:
                        logger.warning("Audio encode queue full; dropping oldest chunk")
                        try:
                            old = self.audio_encode_queue.get_nowait()
                            if old.exists():
                                old.unlink()
                        except Exception:
                            pass
                        try:
                            self.audio_encode_queue.put_nowait(chunk_path)
                        except Exception:
                            if chunk_path.exists():
                                chunk_path.unlink()
                
            except Exception as e:
                if self.is_capturing:
                    logger.error(f"Error in capture thread: {e}")
                time.sleep(0.1)

    def _encode_audio_thread(self):
        """Background thread: read wav chunks, encode PCM to base64, queue for sender."""
        while self.is_capturing or not self.audio_encode_queue.empty():
            try:
                chunk_path = self.audio_encode_queue.get(timeout=0.1)

                if not chunk_path.exists():
                    continue

                with wave.open(str(chunk_path), 'rb') as wf:
                    # Verify format
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                        logger.warning(f"Unexpected audio format in {chunk_path}")
                        try:
                            chunk_path.unlink()
                        except Exception:
                            pass
                        continue

                    # Read PCM data
                    pcm_data = wf.readframes(wf.getnframes())

                    # Convert PCM payload to base64 and queue to VoiceLive
                    audio_base64 = base64.b64encode(pcm_data).decode("utf-8")
                    if audio_base64:
                        self.audio_send_queue.put(audio_base64)

                # Clean up temp file
                try:
                    chunk_path.unlink()
                except Exception:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                if self.is_capturing:
                    logger.error(f"Error in encode thread: {e}")

    def _send_audio_thread(self):
        """Background thread: send queued audio chunks to VoiceLive."""
        while self.is_capturing or not self.audio_send_queue.empty():
            try:
                audio_base64 = self.audio_send_queue.get(timeout=0.1)

                with self._uplink_lock:
                    uplink_paused = self._uplink_paused

                if uplink_paused:
                    continue

                if audio_base64 and self.is_capturing and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.connection.input_audio_buffer.append(audio=audio_base64),
                        self.loop
                    )
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_capturing:
                    logger.error(f"Error sending audio: {e}")

    async def stop_capture(self):
        """Stop capturing audio."""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.encode_thread:
            self.encode_thread.join(timeout=2.0)
        if self.send_thread:
            self.send_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.audio_encode_queue.empty():
            try:
                p = self.audio_encode_queue.get_nowait()
                if p.exists():
                    p.unlink()
            except queue.Empty:
                break
            except Exception:
                break

        while not self.audio_send_queue.empty():
            try:
                self.audio_send_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Stopped G1 audio capture")

    async def start_playback(self):
        """Initialize G1 audio playback system."""
        if self.is_playing:
            return
        
        self.is_playing = True
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_audio_thread)
        self.playback_thread.daemon = True
        self.playback_thread.start()

        if not self._self_test_queued:
            self._self_test_queued = True
            tone = self._generate_test_tone_pcm()
            self.playback_queue.put(tone)
            logger.info("Queued startup speaker self-test tone (PCM16 mono 16kHz)")
        
        logger.info("G1 audio playback system ready")

    def _generate_test_tone_pcm(self, duration_sec: float = 0.30, freq_hz: float = 660.0) -> bytes:
        """Generate a short mono PCM16 16kHz tone to verify speaker path."""
        n = max(1, int(self.playback_rate * duration_sec))
        t = np.arange(n, dtype=np.float32) / float(self.playback_rate)
        wave_f = 0.18 * np.sin(2.0 * np.pi * freq_hz * t)
        out_i16 = np.clip(wave_f * 32767.0, -32768, 32767).astype(np.int16)
        return out_i16.tobytes()

    def _to_g1_playstream_pcm(self, audio_data: bytes) -> bytes:
        """Normalize incoming audio to G1-required PCM16 mono 16kHz bytes."""
        if not audio_data:
            return b""

        # Defensive: if audio arrives as WAV container, strip header and decode payload.
        if len(audio_data) >= 12 and audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
            try:
                with wave.open(io.BytesIO(audio_data), "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    sr = wf.getframerate()
                if sw != 2:
                    logger.warning(f"Unsupported WAV sample width={sw}; expected 16-bit")
                    return b""
                samples = np.frombuffer(frames, dtype=np.int16)
                if ch > 1:
                    samples = samples.reshape(-1, ch).mean(axis=1).astype(np.int16)
                pcm = samples.tobytes()
                return self._convert_playback_rate(pcm, src_rate=sr)
            except Exception as e:
                logger.warning(f"Failed to parse WAV payload for playback: {e}")
                return b""

        # Typical VoiceLive path: raw PCM16 mono 24kHz.
        return self._convert_playback_rate(audio_data, src_rate=self.rate)

    def _playback_audio_thread(self):
        """Background thread: play queued audio via G1 speaker."""
        playback_counter = 0
        
        while self.is_playing:
            try:
                audio_data = self.playback_queue.get(timeout=0.1)
                
                if audio_data and self.is_playing:
                    playback_counter += 1

                    # Play via AudioClient stream API
                    try:
                        app_name = self.playback_app_name
                        stream_id = self._get_or_create_stream_id()

                        if isinstance(audio_data, str):
                            # Defensive path: some SDK builds expose base64 text.
                            audio_data = base64.b64decode(audio_data)

                        audio_data = self._to_g1_playstream_pcm(audio_data)
                        if not audio_data:
                            continue

                        chunk_size = self.playback_chunk_bytes

                        offset = 0
                        total = len(audio_data)
                        while offset < total and self.is_playing:
                            end = min(offset + chunk_size, total)
                            chunk = audio_data[offset:end]

                            # Keep frame alignment to int16 samples.
                            if len(chunk) % 2 != 0:
                                chunk = chunk[:-1]
                            if not chunk:
                                break

                            ret, _ = self.audio_client.PlayStream(app_name, stream_id, chunk)
                            if ret != 0:
                                logger.warning(f"PlayStream failed ret={ret} at chunk offset={offset}")
                                break
                            offset = end

                            # Avoid flooding RPC path; approximate realtime cadence.
                            time.sleep(self.playback_frame_ms / 1000.0)

                        logger.debug(f"Played audio chunk #{playback_counter}, bytes={len(audio_data)}")

                        if self._drain_then_stop and self.playback_queue.empty():
                            self._stop_stream()
                            self._drain_then_stop = False
                    except Exception as e:
                        logger.warning(f"Playback error: {e}")
                        self._stop_stream()
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_playing:
                    logger.error(f"Error in playback: {e}")

    async def queue_audio(self, audio_data: bytes):
        """Queue audio data for playback via G1 speaker."""
        if self.is_playing:
            self.playback_queue.put(audio_data)

    def pause_uplink(self) -> None:
        """Pause mic uplink while robot is speaking to avoid echo and overlap."""
        with self._uplink_lock:
            self._uplink_paused = True

        # Drop queued chunks so old buffered noise/speech is not sent later.
        while not self.audio_send_queue.empty():
            try:
                self.audio_send_queue.get_nowait()
            except queue.Empty:
                break

    def resume_uplink(self) -> None:
        with self._uplink_lock:
            self._uplink_paused = False

    def _get_or_create_stream_id(self) -> str:
        with self._stream_lock:
            if self._active_stream_id is None:
                self._stream_seq += 1
                self._active_stream_id = f"aria_{int(time.time() * 1000)}_{self._stream_seq}"
            return self._active_stream_id

    def _stop_stream(self) -> None:
        with self._stream_lock:
            if self._active_stream_id is not None:
                try:
                    self.audio_client.PlayStop(self.playback_app_name)
                except Exception:
                    pass
                self._active_stream_id = None

    def _convert_playback_rate(self, pcm_bytes: bytes, src_rate: int) -> bytes:
        """Convert PCM16 mono from src_rate to playback_rate."""
        if self.playback_rate == src_rate:
            return pcm_bytes
        if len(pcm_bytes) < 4:
            return pcm_bytes
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            if samples.size == 0:
                return pcm_bytes
            new_len = int(samples.size * self.playback_rate / src_rate)
            if new_len <= 0:
                return pcm_bytes
            old_idx = np.linspace(0, samples.size - 1, num=samples.size)
            new_idx = np.linspace(0, samples.size - 1, num=new_len)
            out = np.interp(new_idx, old_idx, samples.astype(np.float32))
            out_i16 = np.clip(out, -32768, 32767).astype(np.int16)
            return out_i16.tobytes()
        except Exception as e:
            logger.debug(f"Playback resample failed; using raw PCM: {e}")
            return pcm_bytes

    async def interrupt_playback(self):
        """Interrupt current playback without tearing down playback thread."""
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break
        self._stop_stream()

    async def finalize_turn_playback(self):
        """Close stream after queued audio has been drained."""
        self._drain_then_stop = True

    async def stop_playback(self):
        """Stop audio playback."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        
        # Clear queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break

        self._stop_stream()
        
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        logger.info("Stopped G1 audio playback")

    async def cleanup(self):
        """Clean up resources."""
        await self.stop_capture()
        await self.stop_playback()
        
        # Clean up temp directory
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        self.executor.shutdown(wait=True)
        logger.info("G1AudioProcessor cleaned up")


# =============================================================================
#  G1CameraProcessor - RealSense D435i Integration
# =============================================================================

class G1CameraProcessor:
    """
    Handles Intel RealSense D435i camera capture for visual context.
    
    Captures RGB frames at configurable resolution and provides:
      • Latest frame as base64-encoded JPEG for API transmission
      • Optional live preview window with FPS overlay
    """

    SEND_WIDTH = 512
    SEND_HEIGHT = 288
    DISPLAY_WIDTH = 480
    DISPLAY_HEIGHT = 270
    START_MAX_ATTEMPTS = 8
    START_RETRY_DELAY_SEC = 0.35

    @staticmethod
    def _is_display_available() -> bool:
        """Return True when a GUI display server is available."""
        # On Linux/headless targets, OpenCV HighGUI can hard-abort when no
        # X11/Wayland display is present. Avoid any GUI calls in that case.
        if os.name == "posix":
            return bool(
                os.environ.get("DISPLAY")
                or os.environ.get("WAYLAND_DISPLAY")
                or os.environ.get("MIR_SOCKET")
            )
        return True

    def __init__(self, camera_device: int = 0, show_preview: bool = True):
        self.camera_device = camera_device
        self.show_preview = show_preview

        if self.show_preview and not self._is_display_available():
            logger.warning(
                "No GUI display detected; disabling camera preview to avoid OpenCV Qt crash. "
                "Use SSH X11 forwarding or run with --no-preview explicitly."
            )
            self.show_preview = False
        
        self.pipeline: Optional[rs.pipeline] = None
        self.is_running = False
        
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0
        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        
        self._capture_thread: Optional[threading.Thread] = None
        self._encode_thread: Optional[threading.Thread] = None
        
        logger.info(f"G1CameraProcessor initialized (RealSense D435i)")

    def _stream_profiles(self):
        """Return startup fallback stream profiles (w, h, fps)."""
        # Prefer lower bandwidth profiles first for USB2 / unstable first-open cases.
        return [
            (640, 480, 15),
            (424, 240, 30),
            (640, 480, 30),
            (1280, 720, 10),
        ]

    def _try_start_once(self, width: int, height: int, fps: int):
        """Try starting the pipeline with a single profile."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def _start_pipeline_with_retries(self) -> None:
        """Start RealSense with fallback profiles and retry loop."""
        errors = []
        profiles = self._stream_profiles()

        # Give USB stack / camera firmware a brief settle window.
        time.sleep(0.25)

        for attempt in range(1, self.START_MAX_ATTEMPTS + 1):
            for width, height, fps in profiles:
                try:
                    self._try_start_once(width, height, fps)
                    logger.info(
                        f"RealSense started on attempt {attempt}/{self.START_MAX_ATTEMPTS} "
                        f"with {width}x{height}@{fps}"
                    )
                    return
                except Exception as e:
                    msg = f"attempt={attempt}, profile={width}x{height}@{fps}, err={e}"
                    errors.append(msg)
                    logger.debug(f"RealSense start failed: {msg}")
                    try:
                        if self.pipeline:
                            self.pipeline.stop()
                    except Exception:
                        pass
                    self.pipeline = None

            # Backoff between rounds of profile tries
            time.sleep(self.START_RETRY_DELAY_SEC)

        tail = " | ".join(errors[-6:])
        raise RuntimeError(
            "Cannot start RealSense pipeline after retries. "
            "Likely USB bandwidth/profile mismatch or startup race. "
            f"Recent errors: {tail}"
        )

    def start(self) -> None:
        """Start RealSense camera capture."""
        if self.is_running:
            return

        try:
            self._start_pipeline_with_retries()
        except Exception as e:
            raise RuntimeError(f"Cannot start RealSense pipeline: {e}")
        
        self.is_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_frames_thread, daemon=True
        )
        self._encode_thread = threading.Thread(
            target=self._encode_and_display_thread, daemon=True
        )
        self._capture_thread.start()
        self._encode_thread.start()
        
        logger.info("RealSense camera started")

    def _capture_frames_thread(self) -> None:
        """Capture raw frames from RealSense and queue them for encoding/display."""
        while self.is_running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                self._frame_count += 1
                frame = np.asanyarray(color_frame.get_data())

                try:
                    self._frame_queue.put(frame, timeout=0.05)
                except queue.Full:
                    # Drop oldest frame to keep latency low
                    try:
                        _ = self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except Exception:
                        pass

            except Exception as e:
                if self.is_running:
                    logger.warning(f"Frame capture error: {e}")
                time.sleep(0.05)

    def _encode_and_display_thread(self) -> None:
        """Encode queued frames for API and render optional preview."""
        window_title = "ARIA -- RealSense View [q to quit]"
        
        if self.show_preview:
            try:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_title, self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)
            except Exception as e:
                logger.warning(f"Preview window unavailable: {e}")
                self.show_preview = False
        
        while self.is_running or not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get(timeout=0.1)
                
                # Encode for API
                send_frame = cv2.resize(frame, (self.SEND_WIDTH, self.SEND_HEIGHT))
                _, jpeg_buf = cv2.imencode(
                    ".jpg", send_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                
                with self._frame_lock:
                    self._latest_jpeg = jpeg_buf.tobytes()
                
                # Display preview
                if self.show_preview:
                    display = frame.copy()
                    cv2.putText(
                        display,
                        f"ARIA Vision | frame #{self._frame_count}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 90), 1
                    )
                    
                    try:
                        cv2.imshow(window_title, display)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.is_running = False
                            break
                    except Exception:
                        self.show_preview = False

            except queue.Empty:
                continue
                
            except Exception as e:
                if self.is_running:
                    logger.warning(f"Frame encode/display error: {e}")
                time.sleep(0.05)
        
        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def get_latest_frame_base64(self) -> Optional[str]:
        """Get latest frame as base64-encoded JPEG."""
        with self._frame_lock:
            if self._latest_jpeg is None:
                return None
            return base64.b64encode(self._latest_jpeg).decode("utf-8")

    def stop(self) -> None:
        """Stop camera capture."""
        self.is_running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._encode_thread:
            self._encode_thread.join(timeout=2.0)

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        
        logger.info("G1CameraProcessor stopped")


# =============================================================================
#  Animation Dispatcher - G1 LocoClient Integration
# =============================================================================

class G1AnimationController:
    """
    Maps animation tokens to G1 LocoClient gestures.
    """
    
    def __init__(self, loco_client: LocoClient, arm_client: G1ArmActionClient):
        self.loco_client = loco_client
        self.arm_client = arm_client
        self._state_lock = threading.Lock()
        self._is_acting = False
        self._cooldown_until = 0.0
        self.action_cooldown_sec = 0.8
        logger.info("G1AnimationController initialized")

    def get_state(self):
        with self._state_lock:
            now = time.time()
            cooldown_left = max(0.0, self._cooldown_until - now)
            return {
                "is_acting": self._is_acting,
                "cooldown_left": cooldown_left,
                "ready": (not self._is_acting) and (cooldown_left <= 0.0),
            }

    def _try_enter_action(self):
        with self._state_lock:
            now = time.time()
            if self._is_acting:
                return False, "busy", max(0.0, self._cooldown_until - now)
            if now < self._cooldown_until:
                return False, "cooldown", self._cooldown_until - now
            self._is_acting = True
            return True, "ok", 0.0

    def _leave_action(self, cooldown_sec: float = None):
        with self._state_lock:
            self._is_acting = False
            self._cooldown_until = time.time() + (self.action_cooldown_sec if cooldown_sec is None else cooldown_sec)

    def _exec_arm_action(self, action_name: str, hold_sec: float = 0.15):
        action_id = action_map.get(action_name)
        if action_id is None:
            logger.warning(f"Unknown arm action name: {action_name}")
            return
        ret = self.arm_client.ExecuteAction(action_id)
        logger.debug(f"ExecuteAction({action_name}={action_id}) ret={ret}")
        time.sleep(max(0.0, hold_sec))

    def _run_sequence(self, steps):
        for name, hold in steps:
            self._exec_arm_action(name, hold)
    
    def dispatch_animation(self, animation_type: str, intensity: str = "normal") -> None:
        """Execute animation on G1 humanoid."""
        ok, reason, wait_sec = self._try_enter_action()
        if not ok:
            return False, reason, wait_sec

        emoji = _ANIMATION_EMOJI.get(animation_type, "🎭")
        tag = f" [{intensity}]" if intensity != "normal" else ""
        print(f"\n  -- G1 ACTION:  {emoji}  {animation_type.upper()}{tag}\n")
        
        try:
            # SDK-aligned mapping using supported arm action names.
            # Fallbacks are intentionally conservative/safe.
            if animation_type == "wave":
                self._exec_arm_action("high wave", 0.20)
                self._exec_arm_action("release arm", 0.10)
            
            elif animation_type == "give_hand":
                # Agreement-like short acknowledgment
                self._exec_arm_action("right hand up", 0.15)
                self._exec_arm_action("release arm", 0.10)
            
            elif animation_type == "give_heart":
                # Positive/emotional gesture
                self._exec_arm_action("heart", 0.25)
                self._exec_arm_action("release arm", 0.10)
            
            elif animation_type == "think":
                # Thoughtful pose approximation
                self._exec_arm_action("face wave", 0.20)
                self._exec_arm_action("release arm", 0.10)
            
            elif animation_type == "look_around":
                # True body turn using locomotion API.
                yaw_rate = 0.8 if intensity == "expressive" else 0.6
                duration = 3.4 if intensity == "expressive" else 2.2
                self.loco_client.SetVelocity(0.0, 0.0, yaw_rate, duration)
                time.sleep(duration + 0.05)
                self.loco_client.SetVelocity(0.0, 0.0, -yaw_rate, duration)
                
                time.sleep(duration + 0.15)
                self.loco_client.StopMove()

            elif animation_type == "scan_gesture":
                # Upper-body alternating arm gesture (old look-around behavior).
                if intensity == "expressive":
                    steps = [
                        ("left kiss", 0.20),
                        ("right kiss", 0.20),
                        ("left kiss", 0.20),
                        ("right kiss", 0.20),
                        ("release arm", 0.10),
                    ]
                else:
                    steps = [
                        ("left kiss", 0.18),
                        ("right kiss", 0.18),
                        ("release arm", 0.10),
                    ]
                self._run_sequence(steps)

            else:
                logger.warning(f"Unknown animation_type={animation_type!r}, fallback to wave")
                self._exec_arm_action("high wave", 0.20)
                self._exec_arm_action("release arm", 0.10)
            
            logger.info(f"[G1] animation={animation_type!r} intensity={intensity!r}")
            return True, "done", 0.0
            
        except Exception as e:
            logger.error(f"Animation dispatch failed: {e}")
            return False, "error", 0.0
        finally:
            # Keep cooldown short but non-zero to prevent back-to-back spam.
            self._leave_action(cooldown_sec=self.action_cooldown_sec)


# =============================================================================
#  G1EmbodiedVoiceAssistant - Main Integration
# =============================================================================

class G1EmbodiedVoiceAssistant:
    """
    Unitree G1 embodied voice assistant with multimodal I/O.
    
    Follows the exact architecture of embodied_voice_assistant_2.py with
    hardware-specific adaptations for G1.
    """

    FRAME_SEND_INTERVAL = 2.0  # seconds

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, TokenCredential],
        model: str,
        voice: str,
        network_interface: str,
        camera_device: int = 0,
        show_preview: bool = True,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.model = model
        self.voice = voice
        self.network_interface = network_interface
        self.camera_device = camera_device
        self.show_preview = show_preview
        
        # Initialize Unitree SDK
        logger.info(f"Initializing Unitree SDK on interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
        
        # Create Unitree clients
        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()
        try:
            self.audio_client.SetVolume(90)
            code, vol = self.audio_client.GetVolume()
            if code == 0:
                logger.info(f"G1 speaker volume set/readback: {vol}")
            else:
                logger.info("G1 speaker volume set to 90")
        except Exception as e:
            logger.warning(f"Could not set speaker volume: {e}")
        
        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(10.0)
        self.loco_client.Init()

        self.arm_client = G1ArmActionClient()
        self.arm_client.SetTimeout(10.0)
        self.arm_client.Init()
        
        # Components
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[G1AudioProcessor] = None
        self.camera_processor: Optional[G1CameraProcessor] = None
        self.animation_controller: Optional[G1AnimationController] = None
        
        # State
        self.session_ready = False
        self.conversation_started = False
        self._running = False
        self._user_speaking = False
        self._assistant_responding = False
        self._fn_call_buffer: Dict[str, str] = {}
        self._assistant_transcript_buffer = ""
        self._bg_tasks: set[asyncio.Task] = set()

    def _spawn_bg_task(self, coro):
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def start(self):
        """Start the G1 embodied voice assistant."""
        try:
            logger.info(f"Connecting to VoiceLive API with model {self.model}")
            
            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
                connection_options={
                    "max_msg_size": 10 * 1024 * 1024,
                    "heartbeat": 20,
                    "timeout": 20,
                },
            ) as connection:
                conn = connection
                self.connection = conn
                self._running = True
                
                # Initialize processors
                ap = G1AudioProcessor(self.audio_client, conn, self.network_interface)
                self.audio_processor = ap
                
                cp = G1CameraProcessor(
                    camera_device=self.camera_device,
                    show_preview=self.show_preview
                )
                self.camera_processor = cp
                cp.start()
                
                self.animation_controller = G1AnimationController(self.loco_client, self.arm_client)
                
                # Configure session
                await self._setup_session()
                
                # Start audio
                await ap.start_playback()
                
                print("\n" + "=" * 70)
                print("🤖  UNITREE G1 EMBODIED VOICE ASSISTANT  |  ARIA")
                print("Hardware: RealSense D435i + G1 Audio System")
                print("Start speaking — ARIA sees and hears through G1.")
                if self.show_preview:
                    print("Press Ctrl+C to exit  |  Press q in camera window to quit")
                else:
                    print("Press Ctrl+C to exit  |  Camera preview disabled")
                print("=" * 70 + "\n")
                
                # Run event loop + camera sender + short diagnostics
                camera_task = asyncio.create_task(self._camera_frame_sender_loop())
                diag_task = asyncio.create_task(self._diagnostic_loop())
                try:
                    await self._process_events()
                finally:
                    self._running = False
                    camera_task.cancel()
                    diag_task.cancel()
                    try:
                        await camera_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await diag_task
                    except asyncio.CancelledError:
                        pass
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
        finally:
            self._running = False
            if self.camera_processor:
                self.camera_processor.stop()
            if self.audio_processor:
                await self.audio_processor.cleanup()

    async def _setup_session(self):
        """Configure VoiceLive session."""
        logger.info("Setting up embodied AI session...")
        
        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-US-") or self.voice.startswith("en-CA-") or "-" in self.voice:
            voice_config = AzureStandardVoice(name=self.voice, type="azure-standard")
        else:
            voice_config = self.voice
        
        turn_detection_config = ServerVad(
            threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500
        )
        
        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=EMBODIED_AI_INSTRUCTIONS,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            tools=EMBODIED_TOOLS,
            tool_choice="auto",
        )
        
        conn = self.connection
        assert conn is not None
        await conn.session.update(session=session_config)
        
        logger.info("G1 embodied AI session configured")

    async def _camera_frame_sender_loop(self):
        """Background task: periodic camera frame injection while user speaks."""
        while self._running and not self.session_ready:
            await asyncio.sleep(0.2)
        
        logger.info(
            f"Camera frame sender active (speech-gated, interval: {self.FRAME_SEND_INTERVAL}s)"
        )
        
        while self._running:
            await asyncio.sleep(self.FRAME_SEND_INTERVAL)
            if self.session_ready and self.connection and self._user_speaking:
                await self._send_camera_frame(reason="interval")

    @staticmethod
    def _is_connection_closing_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "closing transport" in msg
            or "connection lost" in msg
            or "connection closed" in msg
            or "cannot write to closing transport" in msg
        )

    async def _diagnostic_loop(self):
        """Short periodic runtime diagnostics for terminal visibility."""
        while self._running:
            await asyncio.sleep(3.0)
            try:
                ap = self.audio_processor
                cp = self.camera_processor
                ac = self.animation_controller

                aeq = ap.audio_encode_queue.qsize() if ap else -1
                asq = ap.audio_send_queue.qsize() if ap else -1
                apq = ap.playback_queue.qsize() if ap else -1

                cfq = cp._frame_queue.qsize() if cp else -1
                fcnt = cp._frame_count if cp else -1

                if ac:
                    st = ac.get_state()
                    action_state = "busy" if st["is_acting"] else ("cooldown" if st["cooldown_left"] > 0 else "ready")
                    cooldown = st["cooldown_left"]
                else:
                    action_state = "n/a"
                    cooldown = 0.0

                print(
                    f"[DIAG] sess={int(self.session_ready)} "
                    f"aud(c/e/s/p)={int(ap.is_capturing) if ap else 0}/{aeq}/{asq}/{apq} "
                    f"cam(q/f)={cfq}/{fcnt} "
                    f"act={action_state}:{cooldown:.1f}s"
                )
            except Exception as e:
                logger.debug(f"Diagnostic loop error: {e}")

    async def _send_camera_frame(self, reason: str = "interval") -> None:
        """Send camera frame to conversation."""
        cp = self.camera_processor
        conn = self.connection
        if cp is None or conn is None:
            return
        
        frame_b64 = cp.get_latest_frame_base64()
        if not frame_b64:
            return
        
        image_url = f"data:image/jpeg;base64,{frame_b64}"
        sent = False
        
        # Strategy 1: Fully typed
        if _HAVE_INPUT_IMAGE_PART and InputImageContentPart is not None:
            try:
                await conn.conversation.item.create(
                    item=MessageItem(
                        role="user",
                        content=[InputImageContentPart(image_url=image_url)],
                    )
                )
                sent = True
                logger.debug("Frame sent via Strategy 1")
            except Exception as e1:
                if self._is_connection_closing_error(e1):
                    logger.info("Connection is closing; stopping frame sender")
                    self._running = False
                    return
                logger.debug(f"Strategy 1 failed: {e1!r}")
        
        # Strategy 2: MessageItem + dict
        if not sent:
            try:
                await conn.conversation.item.create(
                    item=MessageItem(
                        role="user",
                        content=[{"type": "input_image", "image_url": image_url}],
                    )
                )
                sent = True
                logger.debug("Frame sent via Strategy 2")
            except Exception as e2:
                if self._is_connection_closing_error(e2):
                    logger.info("Connection is closing; stopping frame sender")
                    self._running = False
                    return
                logger.debug(f"Strategy 2 failed: {e2!r}")
        
        # Strategy 3: Raw WebSocket
        if not sent:
            try:
                await conn.send({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": image_url}
                        ],
                    },
                })
                sent = True
                logger.debug("Frame sent via Strategy 3")
            except Exception as e3:
                if self._is_connection_closing_error(e3):
                    logger.info("Connection is closing; stopping frame sender")
                    self._running = False
                    return
                logger.warning(f"Failed to send frame: {e3}")
        
        if sent:
            logger.debug(f"Camera frame sent [trigger={reason!r}] (#{cp._frame_count})")

    async def _process_events(self):
        """Process VoiceLive events."""
        try:
            conn = self.connection
            assert conn is not None
            async for event in conn:
                await self._handle_event(event)
        except KeyboardInterrupt:
            logger.info("Event processing interrupted")
        except Exception as e:
            logger.error(f"Error processing events: {e}")
            raise

    async def _handle_event(self, event):
        """Handle VoiceLive events."""
        logger.debug(f"Received event: {event.type}")
        ap = self.audio_processor
        conn = self.connection
        assert ap is not None and conn is not None
        
        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info(f"Session ready: {event.session.id}")
            self.session_ready = True
            await ap.start_capture()
        
        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            if self._assistant_responding:
                # Avoid echo/self-barge-in cutting off TTS.
                logger.debug("Ignoring speech_started while assistant is responding")
                return
            self._user_speaking = True
            logger.info("🎤 User started speaking")
            print("🎤 Listening...")
            await self._send_camera_frame(reason="speech_start")
            await ap.interrupt_playback()
            if self._assistant_responding:
                try:
                    await conn.response.cancel()
                except Exception as e:
                    logger.debug(f"No response to cancel: {e}")
        
        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            if self._assistant_responding:
                logger.debug("Ignoring speech_stopped while assistant is responding")
                return
            self._user_speaking = False
            logger.info("🎤 User stopped speaking")
            print("🤔 Processing...")

        elif event.type == "input_audio_buffer.committed":
            logger.debug("Input audio committed")
        
        elif event.type == ServerEventType.RESPONSE_CREATED:
            self._assistant_responding = True
            ap.pause_uplink()
            self._assistant_transcript_buffer = ""
            logger.info("🤖 Assistant response created")
        
        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            logger.debug("Received audio delta")
            if not ap.is_playing:
                await ap.start_playback()
            await ap.queue_audio(event.delta)
        
        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("🤖 Assistant finished speaking")
            print("🎤 Ready for next input...")
            await ap.finalize_turn_playback()

        elif event.type == "response.audio_transcript.delta":
            delta = getattr(event, "delta", "")
            if delta:
                self._assistant_transcript_buffer += delta

        elif event.type == "response.audio_transcript.done":
            transcript = getattr(event, "transcript", None)
            text = transcript or self._assistant_transcript_buffer
            if text:
                logger.info(f"📝 Assistant transcript: {text}")
                print(f"🤖 {text}")
        
        elif event.type == ServerEventType.RESPONSE_DONE:
            self._assistant_responding = False
            self._user_speaking = False
            ap.resume_uplink()
            logger.info("✅ Response complete")

        elif event.type in (
            "response.content_part.added",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.done",
        ):
            logger.debug(f"Response progress event: {event.type}")
        
        elif event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA:
            call_id = getattr(event, "call_id", "default")
            delta = getattr(event, "delta", "")
            if call_id not in self._fn_call_buffer:
                self._fn_call_buffer[call_id] = ""
            self._fn_call_buffer[call_id] += delta
        
        elif event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            call_id = getattr(event, "call_id", "default")
            fn_name = getattr(event, "name", "")
            raw_args = (
                self._fn_call_buffer.pop(call_id, None)
                or getattr(event, "arguments", "{}")
            )
            self._spawn_bg_task(self._handle_function_call(fn_name, raw_args, call_id))
        
        elif event.type == ServerEventType.ERROR:
            msg = str(event.error.message)
            if "active response in progress" in msg.lower():
                logger.warning(f"VoiceLive busy: {msg}")
            else:
                logger.error(f"❌ VoiceLive error: {msg}")
                print(f"Error: {msg}")
        
        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug(f"Conversation item created: {event.item.id}")
        
        else:
            logger.debug(f"Unhandled event type: {event.type}")

    async def _send_robot_action_status(self, text: str):
        """Push robot-action state to realtime model as a lightweight status item."""
        conn = self.connection
        if conn is None:
            return

        status_text = f"[ROBOT_ACTION_STATUS] {text}"
        try:
            await conn.conversation.item.create(
                item=MessageItem(
                    role="user",
                    content=[{"type": "input_text", "text": status_text}],
                )
            )
            return
        except Exception as e1:
            logger.debug(f"Status Strategy 1 failed: {e1!r}")

        try:
            await conn.send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": status_text}],
                },
            })
        except Exception as e2:
            logger.debug(f"Status Strategy 2 failed: {e2!r}")

    async def _handle_function_call(
        self, fn_name: str, raw_args: str, call_id: str
    ) -> None:
        """Parse and dispatch function calls."""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            logger.warning(f"Could not parse function args: {raw_args!r}")
            args = {}
        
        logger.info(f"Function call: {fn_name}({args}) [call_id={call_id!r}]")
        
        if fn_name == "trigger_animation":
            animation_type = args.get("animation_type", "give_hand")
            intensity = args.get("intensity", "normal")

            if animation_type == "turn_around":
                animation_type = "look_around"
            elif animation_type in ("lookaround", "look-around"):
                animation_type = "look_around"

            # Force speech-first behavior: defer motion until assistant audio turn ends.
            if self._assistant_responding:
                await self._send_robot_action_status(
                    f"queued action={animation_type}; wait for speech playback to finish"
                )

                # Wait for response completion + playback queue drain.
                # This handler runs in a background task, so this won't block event intake.
                for _ in range(400):  # ~20s max
                    ap = self.audio_processor
                    playback_empty = (ap is None) or ap.playback_queue.empty()
                    if (not self._assistant_responding) and playback_empty:
                        break
                    await asyncio.sleep(0.05)
            
            if self.animation_controller:
                st = self.animation_controller.get_state()
                if not st["ready"]:
                    if st["is_acting"]:
                        await self._send_robot_action_status(
                            "busy executing previous action; skip new trigger_animation"
                        )
                    else:
                        await self._send_robot_action_status(
                            f"cooldown active ({st['cooldown_left']:.2f}s left); skip new trigger_animation"
                        )
                    return

                await self._send_robot_action_status(
                    f"executing action={animation_type}, intensity={intensity}"
                )

                ok, reason, wait_sec = await asyncio.to_thread(
                    self.animation_controller.dispatch_animation,
                    animation_type,
                    intensity,
                )

                if ok:
                    st2 = self.animation_controller.get_state()
                    await self._send_robot_action_status(
                        f"action complete; cooldown={st2['cooldown_left']:.2f}s"
                    )
                else:
                    if reason == "busy":
                        await self._send_robot_action_status(
                            "busy executing previous action; skip new trigger_animation"
                        )
                    elif reason == "cooldown":
                        await self._send_robot_action_status(
                            f"cooldown active ({wait_sec:.2f}s left); skip new trigger_animation"
                        )
                    else:
                        await self._send_robot_action_status(
                            "action dispatch error; continue speaking without gesture"
                        )
        else:
            logger.warning(f"Unknown function: {fn_name!r}")


# =============================================================================
#  CLI
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unitree G1 Embodied Voice Assistant (ARIA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_API_KEY"),
    )
    
    parser.add_argument(
        "--endpoint",
        help="Azure VoiceLive endpoint",
        type=str,
        default=os.environ.get(
            "AZURE_VOICELIVE_ENDPOINT",
            "https://robot-genai-audio.cognitiveservices.azure.com/"
        ),
    )
    
    parser.add_argument(
        "--model",
        help="VoiceLive model",
        type=str,
        default=os.environ.get("VOICELIVE_MODEL", "gpt-realtime"),
    )
    
    parser.add_argument(
        "--voice",
        help="Assistant voice",
        type=str,
        default=os.environ.get("VOICELIVE_VOICE", "alloy"),
    )
    
    parser.add_argument(
        "--network-interface",
        help="Network interface for G1 (e.g., eth0, enp2s0)",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--camera-device",
        help="RealSense camera device index",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--no-preview",
        help="Disable camera preview window",
        action="store_true",
    )
    
    parser.add_argument(
        "--use-token-credential",
        help="Use Azure token credential",
        action="store_true",
    )
    
    parser.add_argument(
        "--verbose",
        help="Enable verbose logging",
        action="store_true",
    )
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    # Avoid printing massive base64 audio payloads from SDK wire logs.
    logging.getLogger("azure.ai.voicelive.aio._patch").setLevel(logging.INFO)
    
    if not args.api_key and not args.use_token_credential:
        print("❌ Error: No authentication provided")
        print("Use --api-key or set AZURE_VOICELIVE_API_KEY")
        sys.exit(1)
    
    try:
        credential: Union[AzureKeyCredential, TokenCredential]
        if args.use_token_credential:
            credential = InteractiveBrowserCredential()
            logger.info("Using Azure token credential")
        else:
            credential = AzureKeyCredential(args.api_key)
            logger.info("Using API key credential")
        
        assistant = G1EmbodiedVoiceAssistant(
            endpoint=args.endpoint,
            credential=credential,
            model=args.model,
            voice=args.voice,
            network_interface=args.network_interface,
            camera_device=args.camera_device,
            show_preview=not args.no_preview,
        )
        
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        await assistant.start()
    
    except KeyboardInterrupt:
        print("\n👋 G1 assistant shut down. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🤖  Unitree G1 Embodied Voice Assistant with Azure VoiceLive SDK")
    print("=" * 70)
    
    asyncio.run(main())