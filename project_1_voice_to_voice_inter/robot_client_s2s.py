"""
Socket-Based Robot Client with Audio Caching for S2S Pipeline
=============================================================
Smooth, lag-free voice interaction via cached audio playback.

Interaction cycle (like robot_client.py):
  1. Stream mic audio -> pipeline (pipeline VAD detects speech)
  2. Cache TTS audio response into memory buffer
  3. Play complete cached audio smoothly (no chunk-by-chunk jitter)
  4. Execute robot actions from command channel

During playback, mic sends silence to keep the pipeline socket alive
without triggering VAD (half-duplex echo prevention).

Architecture:
   Robot Client (this)              S2S Pipeline (GPU server)
  ┌──────────────────────┐        ┌─────────────────────────────┐
  │ Mic ─> audio :12345 ────────> SocketReceiver -> VAD -> STT  │
  │ (silence during play)│        │               -> LLM        │
  │                      │        │                             │
  │ Cache <- audio :12346 <─────── TTS -> SocketSender          │
  │ sd.play() smooth     │        │                             │
  │                      │        │                             │
  │ Actions <- cmd :12347 <─────── LMOutputProcessor            │
  │ UnitreeDispatcher    │        │   -> SocketCommandSender    │
  └──────────────────────┘        └─────────────────────────────┘

Usage:
  # 1) Start S2S pipeline (GPU server):
  cd speech-to-speech
  python s2s_pipeline.py --mode socket --device cuda

  # 2) Start this client:
  python robot_client_s2s.py --host localhost
  python robot_client_s2s.py --host 192.168.1.100 --no-simulate

Requirements: sounddevice, numpy
"""

import argparse
import json
import logging
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, Optional

import numpy as np
import sounddevice as sd


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ClientConfig:
    host: str = "localhost"
    send_port: int = 12345       # Audio TO pipeline (mic)
    recv_port: int = 12346       # Audio FROM pipeline (speaker)
    cmd_port: int = 12347        # Commands FROM pipeline (actions)
    send_rate: int = 16000       # Mic sample rate (Hz)
    recv_rate: int = 16000       # Speaker sample rate (Hz)
    chunk_size: int = 1024       # Audio chunk size (samples of int16)
    simulate: bool = True        # Simulate actions (True=log only)
    cache_timeout: float = 0.8   # Seconds of silence before flushing audio cache
    audio_cache_dir: str = "./audio_cache"  # Directory for cached audio files


# =============================================================================
# Allowed Actions (mirrors speech-to-speech/actions/allowed_actions.py)
# =============================================================================
ALLOWED_ACTIONS: Dict[str, str] = {
    "NONE": "Do nothing / no physical action.",
    "MOVE_FORWARD": "Walk forward toward the user.",
    "DANCE": "Perform a dance routine.",
}

DEFAULT_ACTION = "NONE"


# =============================================================================
# UnitreeActionDispatcher
# =============================================================================
class UnitreeActionDispatcher:
    """
    Dispatches actions to Unitree G1 robot.

    When simulate=True, actions are logged but not executed.
    Replace placeholder methods with actual Unitree G1 SDK calls.
    """

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self._handlers: Dict[str, Callable] = {
            "NONE": self._action_none,
            "MOVE_FORWARD": self._action_move_forward,
            "DANCE": self._action_dance,
        }
        self._action_count = 0

    def dispatch(self, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        params = params or {}
        action = str(action).strip().upper()

        if action not in ALLOWED_ACTIONS:
            logger.warning(f"Unknown action '{action}', defaulting to {DEFAULT_ACTION}")
            action = DEFAULT_ACTION
            params = {}

        handler = self._handlers.get(action)
        if handler is None:
            logger.warning(f"No handler for '{action}'")
            return False

        try:
            self._action_count += 1
            return handler(params)
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return False

    # -- Action implementations (replace with Unitree G1 SDK calls) -----------
    def _action_none(self, params: Dict) -> bool:
        return True

    def _action_move_forward(self, params: Dict) -> bool:
        distance = params.get("distance", 1.0)
        if self.simulate:
            logger.info(f"[ACTION] MOVE_FORWARD {distance}m (simulated)")
        else:
            # TODO: unitree_g1.walk_forward(distance)
            logger.info(f"[ACTION] MOVE_FORWARD {distance}m (executing)")
        return True

    def _action_dance(self, params: Dict) -> bool:
        if self.simulate:
            logger.info("[ACTION] DANCE (simulated)")
        else:
            # TODO: unitree_g1.dance()
            logger.info("[ACTION] DANCE (executing)")
        return True


# =============================================================================
# Robot Status Provider
# =============================================================================
class RobotStatusProvider:
    """Mock robot status. Replace with real Unitree G1 SDK queries."""

    def __init__(self):
        self.battery = 0.85
        self.location = "home"
        self.is_standing = True
        self.last_action = "NONE"

    def get_status(self) -> Dict[str, Any]:
        return {
            "battery": self.battery,
            "location": self.location,
            "is_standing": self.is_standing,
            "last_action": self.last_action,
        }

    def update(self, action: str):
        self.last_action = action


# =============================================================================
# Socket Robot Client — Cached Audio Playback
# =============================================================================
class SocketRobotClient:
    """
    Socket-based robot client with cached audio playback.

    Audio flow:
      - Mic streams continuously to pipeline (pipeline VAD handles speech detection)
      - TTS audio chunks are accumulated into a memory buffer
      - When a response is complete (no audio for cache_timeout), the buffer is
        played smoothly using sd.play() — no chunk-by-chunk callback jitter
      - During playback, mic sends silence (half-duplex echo gating)

    Connections:
      1. Audio send socket  — mic -> pipeline
      2. Audio recv socket  — pipeline -> audio cache -> smooth playback
      3. Command recv socket — pipeline -> action dispatch
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.stop_event = threading.Event()
        self._is_playing = threading.Event()  # Set while playing cached audio
        self._send_queue: Queue = Queue()

        self.action_dispatcher = UnitreeActionDispatcher(simulate=config.simulate)
        self.status_provider = RobotStatusProvider()

        self._send_socket: Optional[socket.socket] = None
        self._recv_socket: Optional[socket.socket] = None
        self._cmd_socket: Optional[socket.socket] = None

        self._interaction_count = 0
        self._stats: Dict[str, Any] = {
            "audio_chunks_sent": 0,
            "audio_chunks_recv": 0,
            "commands_recv": 0,
            "actions_dispatched": 0,
            "responses_played": 0,
            "start_time": 0.0,
        }

        # Create audio cache directory
        Path(config.audio_cache_dir).mkdir(parents=True, exist_ok=True)

    # ---- Connection setup ---------------------------------------------------
    def _connect(self):
        """Connect to all three S2S pipeline sockets."""
        host = self.config.host

        logger.info(f"Connecting to S2S pipeline at {host}...")

        # Audio send socket (mic -> pipeline)
        self._send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._send_socket.connect((host, self.config.send_port))
        logger.info(f"  Audio send  -> {host}:{self.config.send_port}")

        # Audio recv socket (pipeline -> speaker)
        self._recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._recv_socket.connect((host, self.config.recv_port))
        logger.info(f"  Audio recv  <- {host}:{self.config.recv_port}")

        # Command socket (pipeline -> actions)
        self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_socket.connect((host, self.config.cmd_port))
        logger.info(f"  Command     <- {host}:{self.config.cmd_port}")

        logger.info("All sockets connected!")

    # ---- Mic input ----------------------------------------------------------
    def _mic_callback(self, indata, frames, time_info, status):
        """
        Sounddevice mic callback.
        
        During playback: sends silence to keep pipeline socket alive without
        triggering VAD (half-duplex echo gating).
        Otherwise: sends real mic audio.
        """
        if self._is_playing.is_set():
            # Send silence during playback to keep SocketReceiver unblocked
            self._send_queue.put(b"\x00" * len(bytes(indata)))
        else:
            self._send_queue.put(bytes(indata))

    def _mic_send_worker(self):
        """Send mic audio (or silence) to S2S pipeline continuously."""
        assert self._send_socket is not None
        logger.debug("Mic send thread started")

        while not self.stop_event.is_set():
            try:
                data = self._send_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._send_socket.sendall(data)
                self._stats["audio_chunks_sent"] += 1
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.warning("Audio send socket disconnected")
                break

    # ---- Audio cache & playback ---------------------------------------------
    def _audio_cache_worker(self):
        """
        Receive TTS audio chunks and cache them into a buffer.

        When no audio arrives for cache_timeout seconds (after receiving some),
        the accumulated buffer is played smoothly using sd.play().

        This eliminates the chunk-by-chunk callback jitter of the old approach.
        """
        assert self._recv_socket is not None
        logger.debug("Audio cache thread started")

        # Use non-blocking recv with timeout so we can detect pauses
        self._recv_socket.settimeout(0.2)

        buffer = bytearray()
        last_recv_time = 0.0
        receiving = False

        while not self.stop_event.is_set():
            try:
                data = self._recv_socket.recv(self.config.chunk_size * 2)
                if not data:
                    logger.warning("Audio recv socket closed")
                    break
                buffer.extend(data)
                last_recv_time = time.time()
                receiving = True
                self._stats["audio_chunks_recv"] += 1

            except socket.timeout:
                # No data arrived in this 200ms window — check if we should flush
                if receiving and buffer and last_recv_time > 0:
                    elapsed_since_last = time.time() - last_recv_time
                    if elapsed_since_last >= self.config.cache_timeout:
                        # Response complete — flush and play
                        self._play_cached_audio(bytes(buffer))
                        buffer.clear()
                        receiving = False
                        last_recv_time = 0.0

            except (ConnectionResetError, OSError):
                logger.warning("Audio recv socket disconnected")
                break

        # Flush anything remaining
        if buffer:
            self._play_cached_audio(bytes(buffer))

    def _play_cached_audio(self, audio_bytes: bytes):
        """
        Play cached TTS audio smoothly.

        Converts raw int16 bytes -> numpy float32 -> sd.play() (blocking).
        Also saves to disk cache for debugging/replay.
        """
        if len(audio_bytes) < 4:
            return

        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        duration = len(audio_float) / self.config.recv_rate

        self._interaction_count += 1
        self._stats["responses_played"] += 1

        logger.info(
            f"[PLAY] Response #{self._interaction_count}: "
            f"{duration:.2f}s ({len(audio_bytes)} bytes)"
        )

        # Save to disk cache (optional, useful for debugging)
        try:
            cache_path = os.path.join(
                self.config.audio_cache_dir,
                f"response_{self._interaction_count:04d}.raw",
            )
            with open(cache_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            logger.debug(f"Cache save failed (non-critical): {e}")

        # Smooth playback — mute mic during play
        self._is_playing.set()
        try:
            sd.play(audio_float, samplerate=self.config.recv_rate, blocking=True)
        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._is_playing.clear()

    # ---- Command channel ----------------------------------------------------
    def _command_recv_worker(self):
        """
        Receive action/text commands from S2S pipeline.

        Protocol: 4-byte big-endian length prefix + JSON bytes.
        """
        assert self._cmd_socket is not None
        logger.debug("Command recv thread started")

        while not self.stop_event.is_set():
            try:
                # Read 4-byte length header
                header = self._recv_exact(self._cmd_socket, 4)
                if header is None:
                    logger.warning("Command socket closed")
                    break

                msg_len = struct.unpack(">I", header)[0]
                if msg_len > 1_000_000:
                    logger.error(f"Command message too large: {msg_len}")
                    break

                json_bytes = self._recv_exact(self._cmd_socket, msg_len)
                if json_bytes is None:
                    logger.warning("Command socket closed mid-message")
                    break

                message = json.loads(json_bytes.decode("utf-8"))
                self._stats["commands_recv"] += 1
                self._handle_command(message)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from command socket: {e}")
            except (ConnectionResetError, OSError):
                logger.warning("Command socket disconnected")
                break

    def _handle_command(self, message: Dict[str, Any]):
        """Process a command message from the S2S pipeline."""
        msg_type = message.get("type", "")

        if msg_type == "speech_started":
            logger.info("[EVENT] User started speaking")

        elif msg_type == "speech_stopped":
            logger.info("[EVENT] User stopped speaking — processing...")

        elif msg_type == "assistant_text":
            text = message.get("text", "")
            actions = message.get("actions", [])

            if text:
                print(f"  ASSISTANT: {text}")

            # Dispatch robot actions immediately (don't wait for playback)
            for action_info in actions:
                action_name = action_info.get("action", "NONE")
                params = action_info.get("params", {})

                if action_name and action_name != "NONE":
                    logger.info(f"[DISPATCH] {action_name} params={params}")
                    success = self.action_dispatcher.dispatch(action_name, params)
                    self.status_provider.update(action_name)
                    self._stats["actions_dispatched"] += 1

                    if not success:
                        logger.warning(f"Action {action_name} failed")
        else:
            logger.debug(f"Unknown command type: {msg_type}")

    # ---- Socket helpers -----------------------------------------------------
    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> Optional[bytes]:
        """Receive exactly `size` bytes from a socket."""
        data = b""
        while len(data) < size:
            try:
                packet = conn.recv(size - len(data))
            except (ConnectionResetError, OSError):
                return None
            if not packet:
                return None
            data += packet
        return data

    # ---- Main run loop ------------------------------------------------------
    def run(self):
        """
        Start the client.

        Interaction cycle:
          1. Mic streams audio to pipeline (pipeline VAD detects speech)
          2. TTS audio is cached into memory buffer
          3. Complete cached audio plays smoothly via sd.play()
          4. Actions dispatched in real-time from command channel
        """
        self._stats["start_time"] = time.time()

        try:
            self._connect()
        except ConnectionRefusedError:
            logger.error(
                f"Cannot connect to S2S pipeline at {self.config.host}. "
                "Make sure the pipeline is running: "
                "python s2s_pipeline.py --mode socket"
            )
            return
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return

        print()
        print("=" * 60)
        print("  Robot S2S Client — CACHED PLAYBACK MODE")
        print(f"  Pipeline:      {self.config.host}")
        print(f"  Audio send:    :{self.config.send_port}")
        print(f"  Audio recv:    :{self.config.recv_port}")
        print(f"  Commands:      :{self.config.cmd_port}")
        print(f"  Actions:       {'SIMULATED' if self.config.simulate else 'LIVE'}")
        print(f"  Cache timeout: {self.config.cache_timeout}s")
        print(f"  Audio cache:   {self.config.audio_cache_dir}")
        print("-" * 60)
        print("  Speak naturally. TTS audio is cached then played smoothly.")
        print("  Actions execute immediately from command channel.")
        print("  Press Enter (or Ctrl+C) to stop.")
        print("=" * 60)
        print()

        # Start mic input stream (sounddevice callback -> send_queue)
        mic_stream = sd.RawInputStream(
            samplerate=self.config.send_rate,
            channels=1,
            dtype="int16",
            blocksize=self.config.chunk_size,
            callback=self._mic_callback,
        )
        mic_stream.start()

        # Start worker threads
        send_thread = threading.Thread(target=self._mic_send_worker, daemon=True, name="mic-send")
        cache_thread = threading.Thread(target=self._audio_cache_worker, daemon=True, name="audio-cache")
        cmd_thread = threading.Thread(target=self._command_recv_worker, daemon=True, name="cmd-recv")

        send_thread.start()
        cache_thread.start()
        cmd_thread.start()

        # Wait for user to stop
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass

        self.stop()
        mic_stream.stop()
        mic_stream.close()

        # Join threads
        send_thread.join(timeout=2)
        cache_thread.join(timeout=2)
        cmd_thread.join(timeout=2)

        self._print_stats()

    def stop(self):
        """Graceful shutdown."""
        if self.stop_event.is_set():
            return
        logger.info("Shutting down...")
        self.stop_event.set()

        for sock in (self._recv_socket, self._send_socket, self._cmd_socket):
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass

        logger.info("All sockets closed.")

    def _print_stats(self):
        elapsed = time.time() - self._stats["start_time"]
        print()
        print("-" * 40)
        print(f"  Session duration:   {elapsed:.1f}s")
        print(f"  Audio chunks sent:  {self._stats['audio_chunks_sent']}")
        print(f"  Audio chunks recv:  {self._stats['audio_chunks_recv']}")
        print(f"  Responses played:   {self._stats['responses_played']}")
        print(f"  Commands received:  {self._stats['commands_recv']}")
        print(f"  Actions dispatched: {self._stats['actions_dispatched']}")
        print("-" * 40)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Socket-based robot client with cached audio playback for the S2S pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot_client_s2s.py --host localhost
  python robot_client_s2s.py --host 192.168.1.100 --no-simulate
  python robot_client_s2s.py --cache-timeout 0.5 --cache-dir ./my_cache
        """,
    )
    parser.add_argument("--host", default="localhost", help="S2S pipeline host (default: localhost)")
    parser.add_argument("--send-port", type=int, default=12345, help="Audio send port (default: 12345)")
    parser.add_argument("--recv-port", type=int, default=12346, help="Audio recv port (default: 12346)")
    parser.add_argument("--cmd-port", type=int, default=12347, help="Command recv port (default: 12347)")
    parser.add_argument("--send-rate", type=int, default=16000, help="Mic sample rate Hz (default: 16000)")
    parser.add_argument("--recv-rate", type=int, default=16000, help="Speaker sample rate Hz (default: 16000)")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size in samples (default: 1024)")
    parser.add_argument("--no-simulate", action="store_true", help="Execute real robot actions (default: simulate)")
    parser.add_argument("--cache-timeout", type=float, default=0.8,
                        help="Seconds of silence before flushing audio cache (default: 0.8)")
    parser.add_argument("--cache-dir", default="./audio_cache",
                        help="Directory for cached audio files (default: ./audio_cache)")

    args = parser.parse_args()

    config = ClientConfig(
        host=args.host,
        send_port=args.send_port,
        recv_port=args.recv_port,
        cmd_port=args.cmd_port,
        send_rate=args.send_rate,
        recv_rate=args.recv_rate,
        chunk_size=args.chunk_size,
        simulate=not args.no_simulate,
        cache_timeout=args.cache_timeout,
        audio_cache_dir=args.cache_dir,
    )

    client = SocketRobotClient(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        client.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    client.run()


if __name__ == "__main__":
    main()
