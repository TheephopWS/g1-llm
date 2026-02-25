"""
Robot Client with Streaming STT
================================
Production client with WebSocket streaming for lower latency.

This client streams audio to the server while recording, allowing
Whisper STT to process audio in parallel with recording.

Latency improvement: ~1-2 seconds faster than batch mode.

Requirements:
- sounddevice (for mic recording)
- soundfile (for WAV handling)
- websockets (for streaming)
- requests (for HTTP fallback)
- pygame (for playback)
"""

import argparse
import asyncio
import json
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# WebSocket client
try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("[WARN] websockets not installed. Install: pip install websockets")

# Audio playback
try:
    import pygame
    pygame.mixer.init()
    AUDIO_BACKEND = "pygame"
except ImportError:
    AUDIO_BACKEND = None
    print("[WARN] pygame not installed for audio playback")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class RobotConfig:
    """Robot client configuration."""
    server_url: str = "http://localhost:8000"
    ws_url: str = "ws://localhost:8000/ws/stream"
    session_id: str = "g1-001"
    sample_rate: int = 16000
    channels: int = 1
    
    # VAD settings (higher thresholds = less sensitive, fewer false triggers)
    speech_threshold: float = 0.04
    silence_threshold: float = 0.02
    silence_duration: float = 1.2
    min_speech_duration: float = 0.5
    max_record_seconds: float = 30.0
    pre_speech_buffer: float = 0.3
    
    # Streaming settings
    chunk_duration: float = 0.1  # Send audio every 100ms
    
    request_timeout: int = 300
    audio_output_dir: str = "./audio_cache"


# =============================================================================
# Robot Status Provider
# =============================================================================
class RobotStatusProvider:
    """Provides current robot status."""
    
    def __init__(self):
        self._battery = 0.85
        self._location = "home"
        self._is_standing = True
        self._last_action = "NONE"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "battery": self._battery,
            "location": self._location,
            "is_standing": self._is_standing,
            "last_action": self._last_action,
        }
    
    def update_after_action(self, action: str):
        self._last_action = action


# =============================================================================
# Action Dispatcher (Simplified for testing)
# =============================================================================
class UnitreeActionDispatcher:
    """Dispatches actions to robot (simulated)."""
    
    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self._action_handlers = {
            "NONE": self._action_none,
            "MOVE_FORWARD": self._action_move_forward,
            "DANCE": self._action_dance,
        }
    
    def dispatch(self, action: str, params: Dict[str, Any] = None) -> bool:
        params = params or {}
        handler = self._action_handlers.get(action)
        if handler is None:
            print(f"[ACTION] Unknown action: {action}")
            return False
        return handler(params)
    
    def _action_none(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] NONE - No action")
        return True
    
    def _action_move_forward(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] MOVE_FORWARD - Walking forward")
        return True
    
    def _action_dance(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] DANCE - Dancing!")
        return True


# =============================================================================
# Audio Playback
# =============================================================================
def play_audio_file(file_path: str) -> bool:
    """Play audio file."""
    if AUDIO_BACKEND == "pygame":
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"[AUDIO] Playback error: {e}")
            return False
    else:
        print(f"[AUDIO] No playback. File: {file_path}")
        return False


# =============================================================================
# Streaming Robot Client
# =============================================================================
class StreamingRobotClient:
    """
    Robot client with WebSocket streaming for lower latency.
    
    Audio is streamed to server while recording, overlapping
    STT with recording for faster response.
    """
    
    def __init__(
        self,
        config: RobotConfig,
        action_dispatcher: UnitreeActionDispatcher,
        status_provider: RobotStatusProvider,
    ):
        self.config = config
        self.action_dispatcher = action_dispatcher
        self.status_provider = status_provider
        self._running = False
        self._audio_queue = queue.Queue()
        
        Path(config.audio_output_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_amplitude(self, data: np.ndarray) -> float:
        """Calculate RMS amplitude."""
        return np.sqrt(np.mean(data ** 2))
    
    async def stream_interaction(self) -> Optional[Dict[str, Any]]:
        """
        Perform one interaction with streaming STT.
        
        1. Connect WebSocket
        2. Start recording and streaming audio chunks
        3. Receive partial transcriptions (for feedback)
        4. On silence, send end signal
        5. Receive final result
        """
        if not WEBSOCKET_AVAILABLE:
            print("[ERROR] websockets not installed")
            return None
        
        robot_status = self.status_provider.get_status()
        
        try:
            async with websockets.connect(self.config.ws_url) as ws:
                # Start streaming session
                await ws.send(json.dumps({
                    "type": "start",
                    "session_id": self.config.session_id,
                    "robot_status": robot_status,
                }))
                
                # Wait for confirmation
                response = await ws.recv()
                data = json.loads(response)
                if data.get("type") != "started":
                    print(f"[ERROR] Unexpected response: {data}")
                    return None
                
                print("[MIC] Waiting for speech (streaming mode)...")
                
                # Recording state
                is_recording = True
                speech_started = False
                silent_samples = 0
                total_samples = 0
                pre_speech_buffer = []
                
                samples_per_second = self.config.sample_rate
                max_samples = int(self.config.max_record_seconds * samples_per_second)
                silence_samples_needed = int(self.config.silence_duration * samples_per_second)
                min_speech_samples = int(self.config.min_speech_duration * samples_per_second)
                pre_speech_samples = int(self.config.pre_speech_buffer * samples_per_second)
                chunk_samples = int(self.config.chunk_duration * samples_per_second)
                
                # Audio callback
                def audio_callback(indata, frame_count, time_info, status):
                    if status:
                        print(f"[MIC] Status: {status}")
                    self._audio_queue.put(indata.copy())
                
                # Task to receive partial transcriptions
                partial_text = ""
                async def receive_partials():
                    nonlocal partial_text
                    try:
                        while is_recording:
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                                data = json.loads(msg)
                                if data.get("type") == "partial":
                                    partial_text = data.get("text", "")
                                    print(f"[PARTIAL] {partial_text}")
                            except asyncio.TimeoutError:
                                pass
                    except Exception:
                        pass
                
                # Start receiving partials in background
                receive_task = asyncio.create_task(receive_partials())
                
                # Start recording
                with sd.InputStream(
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype='float32',
                    callback=audio_callback,
                    blocksize=chunk_samples,
                ):
                    speech_samples = 0
                    
                    while total_samples < max_samples and is_recording:
                        try:
                            # Get audio chunk
                            data = self._audio_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        
                        amplitude = self._get_amplitude(data)
                        
                        if not speech_started:
                            # Waiting for speech
                            pre_speech_buffer.append(data)
                            buffer_samples = sum(len(d) for d in pre_speech_buffer)
                            while buffer_samples > pre_speech_samples and len(pre_speech_buffer) > 1:
                                pre_speech_buffer.pop(0)
                                buffer_samples = sum(len(d) for d in pre_speech_buffer)
                            
                            if amplitude > self.config.speech_threshold:
                                speech_started = True
                                print("[MIC] Speech detected, streaming...")
                                
                                # Send buffered pre-speech audio
                                for chunk in pre_speech_buffer:
                                    await ws.send(chunk.tobytes())
                                total_samples = buffer_samples
                        else:
                            # Recording speech - stream to server
                            await ws.send(data.tobytes())
                            total_samples += len(data)
                            speech_samples += len(data)
                            
                            # Check for silence
                            if amplitude < self.config.silence_threshold:
                                silent_samples += len(data)
                                if silent_samples >= silence_samples_needed:
                                    print("[MIC] End of speech detected")
                                    is_recording = False
                            else:
                                silent_samples = 0
                
                # Cancel partial receiver
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                
                # Check minimum duration
                if speech_samples < min_speech_samples:
                    print("[MIC] Audio too short, cancelling")
                    await ws.send(json.dumps({"type": "cancel"}))
                    return None
                
                duration = total_samples / samples_per_second
                print(f"[MIC] Streamed {duration:.1f} seconds")
                
                # Signal end of speech
                await ws.send(json.dumps({"type": "end"}))
                
                # Wait for final result
                print("[CLIENT] Waiting for response...")
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(response)
                    
                    if data.get("type") == "result":
                        return data
                    elif data.get("type") == "error":
                        print(f"[ERROR] Server error: {data.get('message')}")
                        return None
                    # Ignore other messages (partials still coming)
                
        except websockets.exceptions.ConnectionClosed:
            print("[ERROR] WebSocket connection closed")
            return None
        except asyncio.TimeoutError:
            print("[ERROR] Timeout waiting for response")
            return None
        except Exception as e:
            print(f"[ERROR] Streaming error: {e}")
            return None
    
    def download_audio(self, audio_url: str) -> Optional[str]:
        """Download audio file from server."""
        try:
            # Convert ws:// to http://
            http_base = self.config.server_url
            full_url = f"{http_base}{audio_url}"
            
            response = requests.get(full_url, timeout=60)
            response.raise_for_status()
            
            filename = audio_url.split("/")[-1]
            local_path = os.path.join(self.config.audio_output_dir, filename)
            
            with open(local_path, "wb") as f:
                f.write(response.content)
            
            return local_path
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return None
    
    async def run_continuous_async(self, calibrate: bool = True):
        """Run continuous streaming interaction loop."""
        self._running = True
        
        print("\n" + "=" * 60)
        print("Robot Voice Interaction - STREAMING MODE")
        print(f"Server: {self.config.ws_url}")
        print(f"Session: {self.config.session_id}")
        print("-" * 60)
        print("Streaming STT enabled - lower latency!")
        print("Just start speaking. Press Ctrl+C to stop.")
        print("=" * 60 + "\n")
        
        # Optional calibration
        if calibrate:
            print("[SETUP] Calibrating microphone...")
            try:
                audio = sd.rec(
                    int(2 * self.config.sample_rate),
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype='float32',
                )
                sd.wait()
                ambient = self._get_amplitude(audio)
                self.config.silence_threshold = max(ambient * 2.5, 0.015)
                self.config.speech_threshold = max(ambient * 5.0, 0.035)
                print(f"[SETUP] Ambient={ambient:.4f}, Thresholds: speech={self.config.speech_threshold:.4f}, silence={self.config.silence_threshold:.4f}")
            except Exception as e:
                print(f"[SETUP] Calibration failed: {e}")
            print()
        
        interaction_count = 0
        
        while self._running:
            try:
                # Perform streaming interaction
                response = await self.stream_interaction()
                
                if response is None:
                    await asyncio.sleep(0.5)
                    continue
                
                interaction_count += 1
                print(f"\n--- Interaction #{interaction_count} ---")
                
                # Process response
                user_text = response.get("user_text", "")
                speech = response.get("speech", "")
                auxiliary = response.get("auxiliary", {})
                audio_url = response.get("audio_url", "")
                detected_lang = response.get("detected_language", "?")
                timing = response.get("timing", {})
                
                print(f"[LANG] {detected_lang}")
                print(f"[USER] {user_text}")
                print(f"[ROBOT] {speech}")
                
                action = auxiliary.get("action", "NONE")
                params = auxiliary.get("params", {})
                if action != "NONE":
                    print(f"[ACTION] {action}")
                
                # Show timing info if available
                if timing:
                    print(f"[TIMING] STT={timing.get('stt_seconds', '?')}s | LLM={timing.get('llm_seconds', '?')}s | TTS={timing.get('tts_seconds', '?')}s | Total={timing.get('total_seconds', '?')}s")
                
                # Play audio
                if audio_url:
                    audio_path = self.download_audio(audio_url)
                    if audio_path:
                        play_audio_file(audio_path)
                
                # Execute action
                if action and action != "NONE":
                    self.action_dispatcher.dispatch(action, params)
                    self.status_provider.update_after_action(action)
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n[CLIENT] Stopping...")
                self._running = False
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                await asyncio.sleep(0.5)
    
    def run_continuous(self, calibrate: bool = True):
        """Run continuous loop (sync wrapper)."""
        asyncio.run(self.run_continuous_async(calibrate))
    
    def stop(self):
        """Stop the client."""
        self._running = False


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Streaming Robot Voice Client")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Server HTTP URL",
    )
    parser.add_argument(
        "--session",
        default="g1-001",
        help="Session ID",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip microphone calibration",
    )
    parser.add_argument(
        "--speech-threshold",
        type=float,
        default=None,
        help="Speech detection threshold",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=None,
        help="Silence detection threshold",
    )
    
    args = parser.parse_args()
    
    # Derive WebSocket URL from HTTP URL
    ws_url = args.server.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/stream"
    
    config = RobotConfig(
        server_url=args.server,
        ws_url=ws_url,
        session_id=args.session,
    )
    
    if args.speech_threshold:
        config.speech_threshold = args.speech_threshold
    if args.silence_threshold:
        config.silence_threshold = args.silence_threshold
    
    dispatcher = UnitreeActionDispatcher(simulate=True)
    status_provider = RobotStatusProvider()
    
    client = StreamingRobotClient(config, dispatcher, status_provider)
    
    try:
        client.run_continuous(calibrate=not args.no_calibrate)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
