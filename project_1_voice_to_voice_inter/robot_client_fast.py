"""
Robot Client with FAST Chunked Streaming
=========================================
Production client with sentence-level streaming for lowest latency.

This client uses the /ws/stream-fast endpoint which:
1. Streams LLM output
2. Sends TTS audio chunks as sentences complete
3. Client plays first chunk immediately while server generates more

Perceived latency improvement: ~50-60% faster (first audio in ~1.2s vs 3s)

Requirements:
- sounddevice, soundfile, websockets, requests, pygame
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
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("[WARN] websockets not installed")

try:
    import pygame
    pygame.mixer.init()
    AUDIO_BACKEND = "pygame"
except ImportError:
    AUDIO_BACKEND = None
    print("[WARN] pygame not installed")


@dataclass
class RobotConfig:
    """Robot client configuration."""
    server_url: str = "http://localhost:8000"
    ws_url: str = "ws://localhost:8000/ws/stream-fast"  # Fast endpoint!
    session_id: str = "g1-fast-001"
    sample_rate: int = 16000
    channels: int = 1
    
    # VAD settings (higher = less sensitive)
    speech_threshold: float = 0.04
    silence_threshold: float = 0.02
    silence_duration: float = 1.2
    min_speech_duration: float = 0.5
    max_record_seconds: float = 30.0
    pre_speech_buffer: float = 0.3
    
    chunk_duration: float = 0.1
    request_timeout: int = 300
    audio_output_dir: str = "./audio_cache"


class RobotStatusProvider:
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


class UnitreeActionDispatcher:
    def __init__(self, simulate: bool = True):
        self.simulate = simulate
    
    def dispatch(self, action: str, params: Dict[str, Any] = None) -> bool:
        if self.simulate:
            print(f"[ACTION] {action}")
        return True


class AudioPlayer:
    """
    Threaded audio player that queues and plays audio files sequentially.
    Allows starting playback immediately while more audio is being generated.
    """
    
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._playing = False
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
    
    def start(self):
        """Start the playback thread."""
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop playback and clear queue."""
        self._stop_flag.set()
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    
    def enqueue(self, file_path: str):
        """Add audio file to playback queue."""
        self._queue.put(file_path)
    
    def wait_until_done(self):
        """Wait until all queued audio is played."""
        self._queue.join()
    
    def _playback_loop(self):
        """Background thread that plays queued audio."""
        while not self._stop_flag.is_set():
            try:
                file_path = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if self._stop_flag.is_set():
                self._queue.task_done()
                break
            
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not self._stop_flag.is_set():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO] Playback error: {e}")
            finally:
                self._queue.task_done()


class FastStreamingRobotClient:
    """
    Robot client with sentence-level chunked streaming for lowest latency.
    
    Uses /ws/stream-fast endpoint which sends audio chunks as LLM generates
    sentences, allowing audio playback to start before full response is ready.
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
        self._audio_player = AudioPlayer()
        
        Path(config.audio_output_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_amplitude(self, data: np.ndarray) -> float:
        return np.sqrt(np.mean(data ** 2))
    
    def download_audio(self, audio_url: str) -> Optional[str]:
        """Download audio file from server."""
        try:
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
    
    async def fast_stream_interaction(self) -> Optional[Dict[str, Any]]:
        """
        Perform one interaction with sentence-chunked streaming.
        
        Flow:
        1. Connect to /ws/stream-fast
        2. Record and stream audio
        3. Receive audio_chunk messages → download → enqueue for playback
        4. Start playing first chunk immediately
        5. Receive final result message
        """
        if not WEBSOCKET_AVAILABLE:
            print("[ERROR] websockets not installed")
            return None
        
        robot_status = self.status_provider.get_status()
        
        try:
            async with websockets.connect(self.config.ws_url) as ws:
                # Start session
                await ws.send(json.dumps({
                    "type": "start",
                    "session_id": self.config.session_id,
                    "robot_status": robot_status,
                }))
                
                response = await ws.recv()
                data = json.loads(response)
                if data.get("type") != "started":
                    print(f"[ERROR] Unexpected response: {data}")
                    return None
                
                print("[MIC] Waiting for speech (fast streaming mode)...")
                
                # Recording variables
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
                
                def audio_callback(indata, frame_count, time_info, status):
                    if status:
                        print(f"[MIC] Status: {status}")
                    self._audio_queue.put(indata.copy())
                
                # Background task to receive partial transcriptions
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
                            data = self._audio_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        
                        amplitude = self._get_amplitude(data)
                        
                        if not speech_started:
                            pre_speech_buffer.append(data)
                            buffer_samples = sum(len(d) for d in pre_speech_buffer)
                            while buffer_samples > pre_speech_samples and len(pre_speech_buffer) > 1:
                                pre_speech_buffer.pop(0)
                                buffer_samples = sum(len(d) for d in pre_speech_buffer)
                            
                            if amplitude > self.config.speech_threshold:
                                speech_started = True
                                print("[MIC] Speech detected, streaming...")
                                for chunk in pre_speech_buffer:
                                    await ws.send(chunk.tobytes())
                                total_samples = buffer_samples
                        else:
                            await ws.send(data.tobytes())
                            total_samples += len(data)
                            speech_samples += len(data)
                            
                            if amplitude < self.config.silence_threshold:
                                silent_samples += len(data)
                                if silent_samples >= silence_samples_needed:
                                    print("[MIC] End of speech detected")
                                    is_recording = False
                            else:
                                silent_samples = 0
                
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                
                if speech_samples < min_speech_samples:
                    print("[MIC] Audio too short, cancelling")
                    await ws.send(json.dumps({"type": "cancel"}))
                    return None
                
                duration = total_samples / samples_per_second
                print(f"[MIC] Streamed {duration:.1f} seconds")
                
                # Signal end of speech
                await ws.send(json.dumps({"type": "end"}))
                
                # Start audio player thread
                self._audio_player.start()
                
                # Receive audio chunks and final result
                print("[CLIENT] Processing... (audio will play as ready)")
                first_audio_received = False
                first_audio_time = None
                speech_start_time = time.time()
                chunks_received = 0
                result = None
                
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(response)
                    msg_type = data.get("type")
                    
                    if msg_type == "audio_chunk":
                        # Download and queue for immediate playback
                        audio_url = data.get("audio_url", "")
                        chunk_text = data.get("text", "")
                        chunk_idx = data.get("chunk_index", 0)
                        
                        audio_path = self.download_audio(audio_url)
                        if audio_path:
                            self._audio_player.enqueue(audio_path)
                            chunks_received += 1
                            
                            if not first_audio_received:
                                first_audio_received = True
                                first_audio_time = time.time() - speech_start_time
                                print(f"[FAST] 🎵 First audio in {first_audio_time:.2f}s: \"{chunk_text[:30]}...\"")
                            else:
                                print(f"[CHUNK {chunk_idx}] {chunk_text[:40]}...")
                    
                    elif msg_type == "result":
                        result = data
                        break
                    
                    elif msg_type == "error":
                        print(f"[ERROR] Server error: {data.get('message')}")
                        self._audio_player.stop()
                        return None
                
                # Wait for all audio to finish playing
                self._audio_player.wait_until_done()
                self._audio_player.stop()
                
                return result
                
        except websockets.exceptions.ConnectionClosed:
            print("[ERROR] WebSocket connection closed")
            return None
        except asyncio.TimeoutError:
            print("[ERROR] Timeout waiting for response")
            return None
        except Exception as e:
            print(f"[ERROR] Streaming error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_continuous_async(self, calibrate: bool = True):
        """Run continuous fast streaming interaction loop."""
        self._running = True
        
        print("\n" + "=" * 60)
        print("Robot Voice Interaction - FAST CHUNKED STREAMING")
        print(f"Server: {self.config.ws_url}")
        print(f"Session: {self.config.session_id}")
        print("-" * 60)
        print("🚀 Sentence-level streaming enabled!")
        print("   Audio plays as soon as first sentence is ready.")
        print("   Expected first-audio latency: ~1.0-1.5s")
        print("Just start speaking. Press Ctrl+C to stop.")
        print("=" * 60 + "\n")
        
        # Calibration
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
                response = await self.fast_stream_interaction()
                
                if response is None:
                    await asyncio.sleep(0.5)
                    continue
                
                interaction_count += 1
                print(f"\n--- Interaction #{interaction_count} ---")
                
                user_text = response.get("user_text", "")
                speech = response.get("speech", "")
                auxiliary = response.get("auxiliary", {})
                detected_lang = response.get("detected_language", "?")
                timing = response.get("timing", {})
                chunks_sent = response.get("chunks_sent", 0)
                
                print(f"[LANG] {detected_lang}")
                print(f"[USER] {user_text}")
                print(f"[ROBOT] {speech}")
                
                action = auxiliary.get("action", "NONE")
                params = auxiliary.get("params", {})
                if action != "NONE":
                    print(f"[ACTION] {action}")
                
                if timing:
                    print(f"[TIMING] STT={timing.get('stt_seconds', '?')}s | LLM+TTS={timing.get('llm_tts_seconds', '?')}s | FirstAudio={timing.get('first_audio_seconds', '?')}s | Chunks={chunks_sent}")
                
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
        asyncio.run(self.run_continuous_async(calibrate))
    
    def stop(self):
        self._running = False
        self._audio_player.stop()


def main():
    parser = argparse.ArgumentParser(description="Fast Chunked Streaming Robot Voice Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server HTTP URL")
    parser.add_argument("--session", default="g1-fast-001", help="Session ID")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip microphone calibration")
    parser.add_argument("--speech-threshold", type=float, default=None)
    parser.add_argument("--silence-threshold", type=float, default=None)
    
    args = parser.parse_args()
    
    ws_url = args.server.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/stream-fast"  # Use fast endpoint
    
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
    
    client = FastStreamingRobotClient(config, dispatcher, status_provider)
    
    try:
        client.run_continuous(calibrate=not args.no_calibrate)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
