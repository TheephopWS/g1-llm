"""
Socket-Based Robot Client for Speech-to-Speech Pipeline
========================================================
Ultra-low latency voice interaction with real-time action dispatch.

Connects directly to the S2S pipeline via TCP sockets, bypassing all
HTTP/WebSocket overhead. Audio streams in/out while action commands
arrive on a separate channel for immediate robot execution.

Architecture:
   Robot Client (this)              S2S Pipeline (GPU server)
  ┌──────────────────────┐        ┌─────────────────────────────┐
  │ Mic ──> audio send ──────────> SocketReceiver -> VAD -> STT │
  │                      │        │               -> LLM        │
  │ Speaker <─ audio recv <──────── TTS -> SocketSender         │
  │                      │        │                             │
  │ UnitreeAction <─ cmd <──────── LMOutputProcessor            │
  │  Dispatcher    recv  │        │   -> SocketCommandSender    │
  └──────────────────────┘        └─────────────────────────────┘

Usage:
  # 1) Start S2S pipeline (GPU server):
  cd speech-to-speech
  python s2s_pipeline.py --mode socket --device cuda

  # 2) Start this client:
  python robot_client_s2s.py --host localhost

  # Or on a remote robot:
  python robot_client_s2s.py --host 192.168.1.100 --no-simulate

Requirements: sounddevice, numpy
"""

import argparse
import json
import logging
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
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
    chunk_size: int = 1024       # Audio chunk size (bytes of int16 samples)
    simulate: bool = True        # Simulate actions (True=log only)


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
# Socket Robot Client
# =============================================================================
class SocketRobotClient:
    """
    Socket-based robot client that connects to the S2S pipeline.

    Three concurrent connections:
      1. Audio send (mic -> pipeline)
      2. Audio receive (pipeline -> speaker)
      3. Command receive (pipeline -> action dispatch)
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.stop_event = threading.Event()
        self.recv_queue: Queue = Queue()
        self.send_queue: Queue = Queue()

        self.action_dispatcher = UnitreeActionDispatcher(simulate=config.simulate)
        self.status_provider = RobotStatusProvider()

        self._send_socket: Optional[socket.socket] = None
        self._recv_socket: Optional[socket.socket] = None
        self._cmd_socket: Optional[socket.socket] = None

        self._stats: Dict[str, Any] = {
            "audio_chunks_sent": 0,
            "audio_chunks_recv": 0,
            "commands_recv": 0,
            "actions_dispatched": 0,
            "start_time": 0.0,
        }

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

    # ---- Audio callbacks (sounddevice) --------------------------------------
    def _callback_recv(self, outdata, frames, time_info, status):
        """Speaker output callback — play received audio."""
        if not self.recv_queue.empty():
            data = self.recv_queue.get()
            if len(data) >= len(outdata):
                outdata[:] = data[: len(outdata)]
            else:
                outdata[: len(data)] = data
                outdata[len(data) :] = b"\x00" * (len(outdata) - len(data))
        else:
            outdata[:] = b"\x00" * len(outdata)

    def _callback_send(self, indata, frames, time_info, status):
        """Mic input callback — queue audio for sending. Skip if playing."""
        if self.recv_queue.empty():
            self.send_queue.put(bytes(indata))

    # ---- Thread workers -----------------------------------------------------
    def _audio_send_worker(self):
        """Send mic audio to S2S pipeline."""
        assert self._send_socket is not None
        logger.debug("Audio send thread started")
        while not self.stop_event.is_set():
            try:
                data = self.send_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._send_socket.sendall(data)
                self._stats["audio_chunks_sent"] += 1
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.warning("Audio send socket disconnected")
                break

    def _audio_recv_worker(self):
        """Receive TTS audio from S2S pipeline."""
        assert self._recv_socket is not None
        logger.debug("Audio recv thread started")
        chunk_bytes = self.config.chunk_size * 2  # int16 = 2 bytes per sample

        while not self.stop_event.is_set():
            data = self._recv_full_chunk(self._recv_socket, chunk_bytes)
            if data is None:
                logger.warning("Audio recv socket closed")
                break
            self.recv_queue.put(data)
            self._stats["audio_chunks_recv"] += 1

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
                if msg_len > 1_000_000:  # Safety: reject >1MB messages
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

    # ---- Command handling ---------------------------------------------------
    def _handle_command(self, message: Dict[str, Any]):
        """Process a command message from the S2S pipeline."""
        msg_type = message.get("type", "")

        if msg_type == "speech_started":
            logger.info("[EVENT] User started speaking")

        elif msg_type == "speech_stopped":
            logger.info("[EVENT] User stopped speaking")

        elif msg_type == "assistant_text":
            text = message.get("text", "")
            actions = message.get("actions", [])

            if text:
                print(f"\n  ASSISTANT: {text}")

            # Dispatch robot actions
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
    def _recv_full_chunk(conn: socket.socket, chunk_size: int) -> Optional[bytes]:
        """Receive exactly chunk_size bytes."""
        data = b""
        while len(data) < chunk_size:
            try:
                packet = conn.recv(chunk_size - len(data))
            except (ConnectionResetError, OSError):
                return None
            if not packet:
                return None
            data += packet
        return data

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> Optional[bytes]:
        """Receive exactly `size` bytes."""
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
        """Start the client: connect, stream audio, receive commands."""
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
        print("  Robot S2S Client — CONNECTED")
        print(f"  Pipeline: {self.config.host}")
        print(f"  Audio:    send:{self.config.send_port}  recv:{self.config.recv_port}")
        print(f"  Commands: {self.config.cmd_port}")
        print(f"  Actions:  {'SIMULATED' if self.config.simulate else 'LIVE'}")
        print("-" * 60)
        print("  Speak naturally. Actions execute in real-time.")
        print("  Press Enter (or Ctrl+C) to stop.")
        print("=" * 60)
        print()

        # Start audio streams (sounddevice)
        send_stream = sd.RawInputStream(
            samplerate=self.config.send_rate,
            channels=1,
            dtype="int16",
            blocksize=self.config.chunk_size,
            callback=self._callback_send,
        )
        recv_stream = sd.RawOutputStream(
            samplerate=self.config.recv_rate,
            channels=1,
            dtype="int16",
            blocksize=self.config.chunk_size,
            callback=self._callback_recv,
        )

        # Start audio device threads
        threading.Thread(target=send_stream.start, daemon=True).start()
        threading.Thread(target=recv_stream.start, daemon=True).start()

        # Start network threads
        send_thread = threading.Thread(target=self._audio_send_worker, daemon=True)
        recv_thread = threading.Thread(target=self._audio_recv_worker, daemon=True)
        cmd_thread = threading.Thread(target=self._command_recv_worker, daemon=True)

        send_thread.start()
        recv_thread.start()
        cmd_thread.start()

        # Wait for user to stop
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass

        self.stop()

        # Join threads
        send_thread.join(timeout=2)
        recv_thread.join(timeout=2)
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
        print(f"  Commands received:  {self._stats['commands_recv']}")
        print(f"  Actions dispatched: {self._stats['actions_dispatched']}")
        print("-" * 40)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Socket-based robot client for the S2S pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robot_client_s2s.py --host localhost
  python robot_client_s2s.py --host 192.168.1.100 --no-simulate
  python robot_client_s2s.py --send-rate 16000 --recv-rate 16000
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
