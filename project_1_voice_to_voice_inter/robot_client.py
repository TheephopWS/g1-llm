"""
Robot Client for Unitree G1
===========================
Production client running on Jetson Orin (robot side).

Features:
- Continuous voice interaction loop
- Audio recording from microphone
- Communication with server
- Audio playback
- Unitree G1 action dispatch

Requirements:
- sounddevice (for mic recording)
- soundfile (for WAV handling)
- requests (for HTTP)
- pygame or simpleaudio (for playback)
"""

import argparse
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

# Try to import audio playback library
try:
    import pygame
    pygame.mixer.init()
    AUDIO_BACKEND = "pygame"
except ImportError:
    try:
        import simpleaudio
        AUDIO_BACKEND = "simpleaudio"
    except ImportError:
        AUDIO_BACKEND = None
        print("[WARN] No audio playback library found. Install pygame or simpleaudio.")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class RobotConfig:
    """Robot client configuration."""
    server_url: str = "http://localhost:8000"
    session_id: str = "g1-001"
    sample_rate: int = 16000
    channels: int = 1
    record_seconds: float = 5.0
    
    # VAD (Voice Activity Detection) settings
    # These may need tuning based on your microphone and environment
    speech_threshold: float = 0.015      # Amplitude to detect speech start (increase if too sensitive)
    silence_threshold: float = 0.008     # Amplitude to detect silence (lower = more sensitive)
    silence_duration: float = 1.2        # Seconds of silence before stopping recording
    min_speech_duration: float = 0.3     # Minimum speech duration to process (filter out noise)
    max_record_seconds: float = 30.0     # Maximum recording time
    pre_speech_buffer: float = 0.3       # Seconds of audio to keep before speech detected
    
    request_timeout: int = 300           # Increased for first request / slow models
    audio_output_dir: str = "./audio_cache"


# =============================================================================
# Robot Status Provider (customize for your setup)
# =============================================================================
class RobotStatusProvider:
    """
    Provides current robot status.
    Replace with actual Unitree G1 SDK calls.
    """
    
    def __init__(self):
        self._battery = 0.85
        self._location = "home"
        self._is_standing = True
        self._last_action = "NONE"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current robot status as dictionary."""
        return {
            "battery": self._battery,
            "location": self._location,
            "is_standing": self._is_standing,
            "last_action": self._last_action,
        }
    
    def update_after_action(self, action: str):
        """Update status after action execution."""
        self._last_action = action
        if action == "SIT_DOWN":
            self._is_standing = False
        elif action == "STAND_UP":
            self._is_standing = True


# =============================================================================
# Unitree G1 Action Dispatcher (customize for your SDK)
# =============================================================================
class UnitreeActionDispatcher:
    """
    Dispatches actions to Unitree G1 robot.
    
    TODO: Replace placeholder implementations with actual Unitree G1 SDK calls.
    """
    
    def __init__(self, simulate: bool = True):
        """
        Initialize dispatcher.
        
        Args:
            simulate: If True, only print actions instead of executing.
        """
        self.simulate = simulate
        self._action_handlers: Dict[str, Callable] = {
            "NONE": self._action_none,
            "MOVE_FORWARD": self._action_move_forward,
            "DANCE": self._action_dance,
        }
    
    def dispatch(self, action: str, params: Dict[str, Any] = None) -> bool:
        """
        Dispatch action to robot.
        
        Args:
            action: Action name
            params: Optional parameters
            
        Returns:
            True if action executed successfully
        """
        params = params or {}
        handler = self._action_handlers.get(action)
        
        if handler is None:
            print(f"[ACTION] Unknown action: {action}")
            return False
        
        try:
            return handler(params)
        except Exception as e:
            print(f"[ACTION] Error executing {action}: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Action Implementations - Replace with actual Unitree G1 SDK calls
    # -------------------------------------------------------------------------
    def _action_none(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] NONE - No action taken")
        return True
    
    def _action_stop(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] STOP - Emergency stop")
        else:
            # TODO: Call Unitree SDK emergency stop
            # unitree_g1.emergency_stop()
            pass
        return True
    
    def _action_stand_up(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] STAND_UP - Standing up")
        else:
            # TODO: unitree_g1.stand_up()
            pass
        return True
    
    def _action_sit_down(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] SIT_DOWN - Sitting down")
        else:
            # TODO: unitree_g1.sit_down()
            pass
        return True
    
    def _action_move_forward(self, params: Dict) -> bool:
        distance = params.get("distance", 1.0)
        if self.simulate:
            print(f"[ACTION] MOVE_FORWARD - Walking forward {distance}m")
        else:
            # TODO: unitree_g1.walk_forward(distance)
            pass
        return True
    
    def _action_move_backward(self, params: Dict) -> bool:
        distance = params.get("distance", 0.5)
        if self.simulate:
            print(f"[ACTION] MOVE_BACKWARD - Walking backward {distance}m")
        else:
            # TODO: unitree_g1.walk_backward(distance)
            pass
        return True
    
    def _action_turn_left(self, params: Dict) -> bool:
        degrees = params.get("degrees", 90)
        if self.simulate:
            print(f"[ACTION] TURN_LEFT - Turning left {degrees} degrees")
        else:
            # TODO: unitree_g1.turn_left(degrees)
            pass
        return True
    
    def _action_turn_right(self, params: Dict) -> bool:
        degrees = params.get("degrees", 90)
        if self.simulate:
            print(f"[ACTION] TURN_RIGHT - Turning right {degrees} degrees")
        else:
            # TODO: unitree_g1.turn_right(degrees)
            pass
        return True
    
    def _action_wave_hand(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] WAVE_HAND - Waving hand")
        else:
            # TODO: unitree_g1.wave_hand()
            pass
        return True
    
    def _action_nod_head(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] NOD_HEAD - Nodding head")
        else:
            # TODO: unitree_g1.nod_head()
            pass
        return True
    
    def _action_shake_head(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] SHAKE_HEAD - Shaking head")
        else:
            # TODO: unitree_g1.shake_head()
            pass
        return True
    
    def _action_raise_hand(self, params: Dict) -> bool:
        hand = params.get("hand", "right")
        if self.simulate:
            print(f"[ACTION] RAISE_HAND - Raising {hand} hand")
        else:
            # TODO: unitree_g1.raise_hand(hand)
            pass
        return True
    
    def _action_clap_hands(self, params: Dict) -> bool:
        times = params.get("times", 3)
        if self.simulate:
            print(f"[ACTION] CLAP_HANDS - Clapping {times} times")
        else:
            # TODO: unitree_g1.clap_hands(times)
            pass
        return True
    
    def _action_bow(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] BOW - Bowing")
        else:
            # TODO: unitree_g1.bow()
            pass
        return True
    
    def _action_battery_status(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] BATTERY_STATUS - Checking battery")
        # This is usually just a verbal response, no physical action
        return True
    
    def _action_dance(self, params: Dict) -> bool:
        if self.simulate:
            print("[ACTION] DANCE - Dancing!")
        else:
            # TODO: unitree_g1.dance()
            pass
        return True


# =============================================================================
# Audio Recording with Voice Activity Detection
# =============================================================================
class AudioRecorder:
    """
    Records audio from microphone with voice activity detection.
    
    Supports two modes:
    1. record_with_vad() - Wait for speech, record until silence
    2. continuous_listen() - Always listening, yields audio when speech detected
    """
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self._recording = False
        self._audio_queue = queue.Queue()
        self._stop_continuous = False
    
    def _get_amplitude(self, data: np.ndarray) -> float:
        """Calculate RMS amplitude of audio data."""
        return np.sqrt(np.mean(data ** 2))
    
    def record_with_vad(self, wait_for_speech: bool = True) -> Optional[np.ndarray]:
        """
        Record audio with voice activity detection.
        
        Args:
            wait_for_speech: If True, wait indefinitely for speech to start.
                           If False, start recording immediately.
        
        Returns:
            Numpy array of audio data, or None if failed/no speech
        """
        frames = []
        pre_speech_buffer = []  # Keep some audio before speech starts
        silent_samples = 0
        speech_started = False
        
        samples_per_second = self.config.sample_rate
        max_samples = int(self.config.max_record_seconds * samples_per_second)
        silence_samples_needed = int(self.config.silence_duration * samples_per_second)
        min_speech_samples = int(self.config.min_speech_duration * samples_per_second)
        pre_speech_samples = int(self.config.pre_speech_buffer * samples_per_second)
        
        blocksize = 1024
        
        def callback(indata, frame_count, time_info, status):
            if status:
                print(f"[MIC] Status: {status}")
            self._audio_queue.put(indata.copy())
        
        if wait_for_speech:
            print("[MIC] Waiting for speech...")
        else:
            print("[MIC] Recording...")
        
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype='float32',
                callback=callback,
                blocksize=blocksize,
            ):
                total_samples = 0
                speech_samples = 0
                
                while total_samples < max_samples:
                    try:
                        data = self._audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    amplitude = self._get_amplitude(data)
                    
                    if not speech_started:
                        # Keep a rolling buffer of pre-speech audio
                        pre_speech_buffer.append(data)
                        buffer_samples = sum(len(d) for d in pre_speech_buffer)
                        while buffer_samples > pre_speech_samples and len(pre_speech_buffer) > 1:
                            pre_speech_buffer.pop(0)
                            buffer_samples = sum(len(d) for d in pre_speech_buffer)
                        
                        # Check if speech started
                        if amplitude > self.config.speech_threshold:
                            speech_started = True
                            # Add pre-speech buffer to frames
                            frames.extend(pre_speech_buffer)
                            total_samples = sum(len(d) for d in frames)
                            print("[MIC] Speech detected, recording...")
                    else:
                        # Recording speech
                        frames.append(data)
                        total_samples += len(data)
                        speech_samples += len(data)
                        
                        # Check for silence
                        if amplitude < self.config.silence_threshold:
                            silent_samples += len(data)
                            if silent_samples >= silence_samples_needed:
                                print("[MIC] End of speech detected")
                                break
                        else:
                            silent_samples = 0
            
            if not frames:
                if wait_for_speech:
                    print("[MIC] No speech detected")
                return None
            
            # Check minimum duration
            audio_data = np.concatenate(frames, axis=0)
            if len(audio_data) < min_speech_samples:
                print(f"[MIC] Audio too short ({len(audio_data)/samples_per_second:.1f}s), ignoring")
                return None
            
            duration = len(audio_data) / samples_per_second
            print(f"[MIC] Recorded {duration:.1f} seconds")
            return audio_data
            
        except Exception as e:
            print(f"[MIC] Recording error: {e}")
            return None
    
    def continuous_listen(self):
        """
        Generator that continuously listens and yields audio when speech is detected.
        
        Usage:
            for audio_data in recorder.continuous_listen():
                # process audio_data
                if should_stop:
                    recorder.stop_continuous()
        """
        self._stop_continuous = False
        
        while not self._stop_continuous:
            audio_data = self.record_with_vad(wait_for_speech=True)
            if audio_data is not None:
                yield audio_data
    
    def stop_continuous(self):
        """Stop the continuous listening loop."""
        self._stop_continuous = True
    
    def record_fixed_duration(self, duration: float = None) -> Optional[np.ndarray]:
        """Record for a fixed duration."""
        duration = duration or self.config.record_seconds
        print(f"[MIC] Recording for {duration} seconds...")
        
        try:
            audio_data = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype='float32',
            )
            sd.wait()
            return audio_data
        except Exception as e:
            print(f"[MIC] Recording error: {e}")
            return None
    
    def calibrate_silence(self, duration: float = 2.0) -> float:
        """
        Calibrate silence threshold based on ambient noise.
        
        Args:
            duration: Seconds to sample ambient noise
            
        Returns:
            Recommended silence threshold
        """
        print(f"[MIC] Calibrating... Please be quiet for {duration} seconds")
        time.sleep(0.5)  # Give user time to read
        
        try:
            audio_data = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype='float32',
            )
            sd.wait()
            
            # Calculate ambient noise level
            ambient_rms = self._get_amplitude(audio_data)
            
            # Set thresholds based on ambient noise
            # Speech threshold should be 2-3x ambient, silence threshold 1.5x
            recommended_silence = ambient_rms * 1.5
            recommended_speech = ambient_rms * 2.5
            
            print(f"[MIC] Ambient noise level: {ambient_rms:.4f}")
            print(f"[MIC] Recommended silence_threshold: {recommended_silence:.4f}")
            print(f"[MIC] Recommended speech_threshold: {recommended_speech:.4f}")
            
            return recommended_silence
            
        except Exception as e:
            print(f"[MIC] Calibration error: {e}")
            return self.config.silence_threshold


# =============================================================================
# Audio Playback
# =============================================================================
def play_audio_file(file_path: str) -> bool:
    """Play an audio file."""
    if AUDIO_BACKEND == "pygame":
        return _play_with_pygame(file_path)
    elif AUDIO_BACKEND == "simpleaudio":
        return _play_with_simpleaudio(file_path)
    else:
        print(f"[AUDIO] No playback backend. File saved at: {file_path}")
        return False


def _play_with_pygame(file_path: str) -> bool:
    """Play audio using pygame."""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        return True
    except Exception as e:
        print(f"[AUDIO] Pygame playback error: {e}")
        return False


def _play_with_simpleaudio(file_path: str) -> bool:
    """Play audio using simpleaudio."""
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return True
    except Exception as e:
        print(f"[AUDIO] Simpleaudio playback error: {e}")
        return False


# =============================================================================
# Robot Client
# =============================================================================
class RobotClient:
    """Main robot client for voice interaction."""
    
    def __init__(
        self,
        config: RobotConfig,
        action_dispatcher: UnitreeActionDispatcher,
        status_provider: RobotStatusProvider,
    ):
        self.config = config
        self.action_dispatcher = action_dispatcher
        self.status_provider = status_provider
        self.recorder = AudioRecorder(config)
        self._running = False
        
        # Ensure output directory exists
        Path(config.audio_output_dir).mkdir(parents=True, exist_ok=True)
    
    def send_audio(self, wav_path: str) -> Optional[Dict[str, Any]]:
        """Send audio to server and get response."""
        robot_status = self.status_provider.get_status()
        
        try:
            with open(wav_path, "rb") as f:
                response = requests.post(
                    f"{self.config.server_url}/process_voice",
                    files={"file": ("input.wav", f, "audio/wav")},
                    data={
                        "session_id": self.config.session_id,
                        "robot_status_json": json.dumps(robot_status, ensure_ascii=False),
                    },
                    timeout=self.config.request_timeout,
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            print("[ERROR] Server request timed out")
            return None
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to server at {self.config.server_url}")
            return None
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            return None
    
    def download_audio(self, audio_url: str) -> Optional[str]:
        """Download audio file from server."""
        try:
            full_url = f"{self.config.server_url}{audio_url}"
            response = requests.get(full_url, timeout=60)
            response.raise_for_status()
            
            # Save to cache directory
            filename = audio_url.split("/")[-1]
            local_path = os.path.join(self.config.audio_output_dir, filename)
            
            with open(local_path, "wb") as f:
                f.write(response.content)
            
            return local_path
            
        except Exception as e:
            print(f"[ERROR] Failed to download audio: {e}")
            return None
    
    def process_interaction(self, use_vad: bool = True) -> bool:
        """
        Process one interaction cycle:
        1. Record audio
        2. Send to server
        3. Play response
        4. Execute action
        
        Returns:
            True if interaction completed successfully
        """
        # 1. Record audio
        if use_vad:
            audio_data = self.recorder.record_with_vad()
        else:
            audio_data = self.recorder.record_fixed_duration()
        
        if audio_data is None or len(audio_data) == 0:
            print("[WARN] No audio recorded")
            return False
        
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, self.config.sample_rate)
            wav_path = tmp.name
        
        try:
            # 2. Send to server
            print("[CLIENT] Sending audio to server...")
            response_data = self.send_audio(wav_path)
            
            if response_data is None:
                return False
            
            # 3. Process response
            user_text = response_data.get("user_text", "")
            speech = response_data.get("speech", "")
            auxiliary = response_data.get("auxiliary", {})
            audio_url = response_data.get("audio_url", "")
            
            print(f"\n[USER] {user_text}")
            print(f"[ROBOT] {speech}")
            
            action = auxiliary.get("action", "NONE")
            params = auxiliary.get("params", {})
            print(f"[ACTION] {action} {params if params else ''}")
            
            # 4. Download and play audio
            if audio_url:
                audio_path = self.download_audio(audio_url)
                if audio_path:
                    print("[AUDIO] Playing response...")
                    play_audio_file(audio_path)
            
            # 5. Execute action
            if action and action != "NONE":
                print(f"[ROBOT] Executing action: {action}")
                self.action_dispatcher.dispatch(action, params)
                self.status_provider.update_after_action(action)
            
            return True
            
        finally:
            # Clean up temp file
            try:
                os.remove(wav_path)
            except:
                pass
    
    def run_loop(self, use_vad: bool = True):
        """Run interaction loop with ENTER key trigger."""
        self._running = True
        print("\n" + "=" * 60)
        print("Robot Voice Interaction Started (Press-to-Talk Mode)")
        print(f"Server: {self.config.server_url}")
        print(f"Session: {self.config.session_id}")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        while self._running:
            try:
                input("\n[Press ENTER to start speaking, Ctrl+C to quit]\n")
                self.process_interaction(use_vad=use_vad)
            except KeyboardInterrupt:
                print("\n[CLIENT] Stopping...")
                self._running = False
                break
            except Exception as e:
                print(f"[ERROR] Interaction failed: {e}")
                time.sleep(1)
    
    def run_continuous(self, calibrate: bool = True):
        """
        Run continuous listening mode - no button press needed.
        
        The robot will:
        1. Wait for speech to be detected
        2. Record until silence
        3. Process and respond
        4. Go back to waiting
        
        Args:
            calibrate: If True, calibrate silence threshold on startup
        """
        self._running = True
        print("\n" + "=" * 60)
        print("Robot Voice Interaction - CONTINUOUS LISTENING MODE")
        print(f"Server: {self.config.server_url}")
        print(f"Session: {self.config.session_id}")
        print("-" * 60)
        print("The robot is always listening. Just start speaking!")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        # Optional: calibrate for ambient noise
        if calibrate:
            print("[SETUP] Calibrating microphone for ambient noise...")
            suggested_threshold = self.recorder.calibrate_silence(duration=2.0)
            # Auto-apply if reasonable
            if 0.001 < suggested_threshold < 0.1:
                self.config.silence_threshold = suggested_threshold
                self.config.speech_threshold = suggested_threshold * 1.8
                print(f"[SETUP] Applied thresholds: speech={self.config.speech_threshold:.4f}, silence={self.config.silence_threshold:.4f}")
            print()
        
        # Continuous listening loop
        interaction_count = 0
        while self._running:
            try:
                # Wait for and record speech
                audio_data = self.recorder.record_with_vad(wait_for_speech=True)
                
                if audio_data is None:
                    continue
                
                interaction_count += 1
                print(f"\n--- Interaction #{interaction_count} ---")
                
                # Save to temp WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_data, self.config.sample_rate)
                    wav_path = tmp.name
                
                try:
                    # Send to server
                    print("[CLIENT] Processing...")
                    response_data = self.send_audio(wav_path)
                    
                    if response_data is None:
                        print("[ERROR] No response from server")
                        continue
                    
                    # Show what was understood
                    user_text = response_data.get("user_text", "")
                    speech = response_data.get("speech", "")
                    auxiliary = response_data.get("auxiliary", {})
                    audio_url = response_data.get("audio_url", "")
                    detected_lang = response_data.get("detected_language", "?")
                    
                    print(f"[LANG] Detected: {detected_lang}")
                    print(f"[USER] {user_text}")
                    print(f"[ROBOT] {speech}")
                    
                    action = auxiliary.get("action", "NONE")
                    params = auxiliary.get("params", {})
                    if action != "NONE":
                        print(f"[ACTION] {action} {params if params else ''}")
                    
                    # Play audio response
                    if audio_url:
                        audio_path = self.download_audio(audio_url)
                        if audio_path:
                            play_audio_file(audio_path)
                    
                    # Execute action
                    if action and action != "NONE":
                        self.action_dispatcher.dispatch(action, params)
                        self.status_provider.update_after_action(action)
                    
                    print("-" * 40)
                    
                finally:
                    try:
                        os.remove(wav_path)
                    except:
                        pass
                
            except KeyboardInterrupt:
                print("\n[CLIENT] Stopping continuous mode...")
                self._running = False
                break
            except Exception as e:
                print(f"[ERROR] Interaction failed: {e}")
                time.sleep(0.5)
    
    def stop(self):
        """Stop the interaction loop."""
        self._running = False


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Robot Voice Client")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--session",
        default="g1-001",
        help="Session ID (default: g1-001)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Simulate actions instead of executing (default: True)",
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="Execute real robot actions",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous listening mode (no button press needed)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip microphone calibration in continuous mode",
    )
    parser.add_argument(
        "--speech-threshold",
        type=float,
        default=None,
        help="Speech detection threshold (default: 0.015, increase if too sensitive)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=None,
        help="Silence detection threshold (default: 0.008)",
    )
    parser.add_argument(
        "--fixed-duration",
        type=float,
        default=None,
        help="Use fixed recording duration instead of VAD",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = RobotConfig(
        server_url=args.server,
        session_id=args.session,
        sample_rate=args.sample_rate,
    )
    
    if args.fixed_duration:
        config.record_seconds = args.fixed_duration
    
    if args.speech_threshold:
        config.speech_threshold = args.speech_threshold
    
    if args.silence_threshold:
        config.silence_threshold = args.silence_threshold
    
    # Create components
    simulate = not args.no_simulate
    action_dispatcher = UnitreeActionDispatcher(simulate=simulate)
    status_provider = RobotStatusProvider()
    
    # Create and run client
    client = RobotClient(config, action_dispatcher, status_provider)
    
    try:
        if args.continuous:
            # Continuous listening mode - no button press needed
            client.run_continuous(calibrate=not args.no_calibrate)
        elif args.fixed_duration:
            # Fixed duration recording
            client.run_loop(use_vad=False)
        else:
            # Press-to-talk with VAD
            client.run_loop(use_vad=True)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
