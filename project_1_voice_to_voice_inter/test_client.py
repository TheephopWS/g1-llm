"""
Test Client for Robot Voice Brain
=================================
Testing client for laptop development.

Features:
- Interactive menu-based testing
- Microphone recording with playback
- Text-only mode for quick testing
- File upload mode
- Session management

Use this to test the server before deploying to robot.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

# Audio libraries (optional for full features)
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARN] Audio libraries not available. Install: pip install sounddevice soundfile numpy")

# Playback library
try:
    import pygame
    pygame.mixer.init()
    PLAYBACK_AVAILABLE = True
except ImportError:
    PLAYBACK_AVAILABLE = False
    print("[WARN] Pygame not available for playback. Install: pip install pygame")


# =============================================================================
# Configuration
# =============================================================================
class TestConfig:
    """Test client configuration."""
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        session_id: str = "test-session",
        sample_rate: int = 16000,
        record_seconds: float = 5.0,
    ):
        self.server_url = server_url
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds


# =============================================================================
# Test Client
# =============================================================================
class TestClient:
    """Interactive test client for development."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.audio_cache_dir = Path("./test_audio_cache")
        self.audio_cache_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Server Communication
    # -------------------------------------------------------------------------
    def check_health(self) -> bool:
        """Check if server is healthy."""
        try:
            r = requests.get(f"{self.config.server_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            print(f"[OK] Server healthy: {data}")
            return True
        except Exception as e:
            print(f"[ERROR] Server health check failed: {e}")
            return False
    
    def get_actions(self) -> Optional[dict]:
        """Get available actions from server."""
        try:
            r = requests.get(f"{self.config.server_url}/actions", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to get actions: {e}")
            return None
    
    def reset_session(self) -> bool:
        """Reset current session."""
        try:
            r = requests.post(
                f"{self.config.server_url}/session/{self.config.session_id}/reset",
                timeout=5
            )
            r.raise_for_status()
            print(f"[OK] Session '{self.config.session_id}' reset")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to reset session: {e}")
            return False
    
    def get_session_history(self) -> Optional[dict]:
        """Get session history."""
        try:
            r = requests.get(
                f"{self.config.server_url}/session/{self.config.session_id}",
                timeout=5
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to get session: {e}")
            return None
    
    def send_audio_file(self, audio_path: str, robot_status: dict = None) -> Optional[dict]:
        """Send audio file to server."""
        robot_status = robot_status or {"battery": 0.85, "location": "test_room"}
        
        try:
            with open(audio_path, "rb") as f:
                r = requests.post(
                    f"{self.config.server_url}/process_voice",
                    files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                    data={
                        "session_id": self.config.session_id,
                        "robot_status_json": json.dumps(robot_status, ensure_ascii=False),
                    },
                    timeout=120,
                )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            print("[ERROR] Request timed out (120s)")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to send audio: {e}")
            return None
    
    def send_text(self, text: str, robot_status: dict = None, language: str = "en") -> Optional[dict]:
        """Send text directly to server (skip STT)."""
        robot_status = robot_status or {"battery": 0.85, "location": "test_room"}
        
        try:
            r = requests.post(
                f"{self.config.server_url}/process_text",
                data={
                    "text": text,
                    "session_id": self.config.session_id,
                    "robot_status_json": json.dumps(robot_status, ensure_ascii=False),
                    "language": language,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ERROR] Failed to send text: {e}")
            return None
    
    def download_audio(self, audio_url: str) -> Optional[str]:
        """Download audio from server."""
        try:
            r = requests.get(f"{self.config.server_url}{audio_url}", timeout=60)
            r.raise_for_status()
            
            filename = audio_url.split("/")[-1]
            local_path = self.audio_cache_dir / filename
            
            with open(local_path, "wb") as f:
                f.write(r.content)
            
            return str(local_path)
        except Exception as e:
            print(f"[ERROR] Failed to download audio: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Audio Recording & Playback
    # -------------------------------------------------------------------------
    def record_audio(self, duration: float = None) -> Optional[str]:
        """Record audio from microphone."""
        if not AUDIO_AVAILABLE:
            print("[ERROR] Audio libraries not available")
            return None
        
        duration = duration or self.config.record_seconds
        print(f"\n[MIC] Recording for {duration} seconds...")
        print("[MIC] Speak now!")
        
        try:
            audio_data = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=1,
                dtype='float32',
            )
            sd.wait()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, self.config.sample_rate)
                print(f"[MIC] Recording saved: {tmp.name}")
                return tmp.name
                
        except Exception as e:
            print(f"[ERROR] Recording failed: {e}")
            return None
    
    def play_audio(self, file_path: str) -> bool:
        """Play audio file."""
        if not PLAYBACK_AVAILABLE:
            print(f"[INFO] Playback not available. Audio saved at: {file_path}")
            return False
        
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            print("[AUDIO] Playing response...")
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"[ERROR] Playback failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Interactive Test Methods
    # -------------------------------------------------------------------------
    def test_voice_interaction(self, play_response: bool = True):
        """Test full voice interaction cycle."""
        # Record
        audio_path = self.record_audio()
        if not audio_path:
            return
        
        try:
            # Send to server
            print("\n[CLIENT] Sending to server...")
            response = self.send_audio_file(audio_path)
            
            if response:
                self._print_response(response)
                
                # Play response
                if play_response and response.get("audio_url"):
                    local_audio = self.download_audio(response["audio_url"])
                    if local_audio:
                        self.play_audio(local_audio)
        finally:
            # Cleanup
            try:
                os.remove(audio_path)
            except:
                pass
    
    def test_text_interaction(self, text: str, language: str = "en", play_response: bool = True):
        """Test text-only interaction (skip STT)."""
        print(f"\n[TEXT] Sending: {text}")
        
        response = self.send_text(text, language=language)
        
        if response:
            self._print_response(response)
            
            if play_response and response.get("audio_url"):
                local_audio = self.download_audio(response["audio_url"])
                if local_audio:
                    self.play_audio(local_audio)
    
    def test_file_interaction(self, file_path: str, play_response: bool = True):
        """Test with existing audio file."""
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return
        
        print(f"\n[FILE] Sending: {file_path}")
        
        response = self.send_audio_file(file_path)
        
        if response:
            self._print_response(response)
            
            if play_response and response.get("audio_url"):
                local_audio = self.download_audio(response["audio_url"])
                if local_audio:
                    self.play_audio(local_audio)
    
    def _print_response(self, response: dict):
        """Pretty print server response."""
        print("\n" + "=" * 60)
        print("SERVER RESPONSE")
        print("=" * 60)
        print(f"  Session:   {response.get('session_id', 'N/A')}")
        print(f"  Language:  {response.get('detected_language', 'N/A')}")
        print(f"  User said: {response.get('user_text', 'N/A')}")
        print("-" * 60)
        print(f"  Robot says: {response.get('speech', 'N/A')}")
        print("-" * 60)
        aux = response.get("auxiliary", {})
        print(f"  Action:    {aux.get('action', 'N/A')}")
        print(f"  Params:    {aux.get('params', {})}")
        print("=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # Interactive Menu
    # -------------------------------------------------------------------------
    def run_interactive(self):
        """Run interactive testing menu."""
        print("\n" + "=" * 60)
        print("Robot Voice Brain - Test Client")
        print(f"Server: {self.config.server_url}")
        print(f"Session: {self.config.session_id}")
        print("=" * 60)
        
        # Check server
        if not self.check_health():
            print("\n[WARN] Server not available. Start server first.")
        
        while True:
            print("\n--- TEST MENU ---")
            print("1. Voice interaction (record & send)")
            print("2. Text interaction (type & send)")
            print("3. Send audio file")
            print("4. View session history")
            print("5. Reset session")
            print("6. View available actions")
            print("7. Check server health")
            print("8. Change session ID")
            print("9. Quick text test (loop)")
            print("0. Exit")
            print("-" * 20)
            
            choice = input("Select option: ").strip()
            
            try:
                if choice == "1":
                    if not AUDIO_AVAILABLE:
                        print("[ERROR] Audio not available. Use text mode or install audio libraries.")
                    else:
                        duration = input(f"Recording duration (default {self.config.record_seconds}s): ").strip()
                        if duration:
                            self.config.record_seconds = float(duration)
                        self.test_voice_interaction()
                
                elif choice == "2":
                    text = input("Enter text: ").strip()
                    if text:
                        lang = input("Language (en/zh, default en): ").strip() or "en"
                        self.test_text_interaction(text, language=lang)
                
                elif choice == "3":
                    file_path = input("Audio file path: ").strip()
                    if file_path:
                        self.test_file_interaction(file_path)
                
                elif choice == "4":
                    history = self.get_session_history()
                    if history:
                        print(f"\nSession: {history.get('session_id')}")
                        print(f"Messages: {history.get('message_count')}")
                        for msg in history.get("messages", []):
                            role = msg.get("role", "?").upper()
                            content = msg.get("content", "")
                            print(f"  [{role}] {content}")
                
                elif choice == "5":
                    self.reset_session()
                
                elif choice == "6":
                    actions = self.get_actions()
                    if actions:
                        print("\nAvailable Actions:")
                        for action, desc in actions.get("actions", {}).items():
                            print(f"  {action}: {desc}")
                
                elif choice == "7":
                    self.check_health()
                
                elif choice == "8":
                    new_id = input(f"New session ID (current: {self.config.session_id}): ").strip()
                    if new_id:
                        self.config.session_id = new_id
                        print(f"[OK] Session ID changed to: {new_id}")
                
                elif choice == "9":
                    print("\n[TEXT LOOP] Type messages. Enter 'q' to quit.")
                    while True:
                        text = input("\nYou: ").strip()
                        if text.lower() == 'q':
                            break
                        if text:
                            self.test_text_interaction(text, play_response=PLAYBACK_AVAILABLE)
                
                elif choice == "0":
                    print("Goodbye!")
                    break
                
                else:
                    print("[WARN] Invalid option")
                    
            except KeyboardInterrupt:
                print("\n[Interrupted]")
            except Exception as e:
                print(f"[ERROR] {e}")


# =============================================================================
# Quick Test Functions
# =============================================================================
def quick_test_text(server_url: str, text: str, session_id: str = "quick-test"):
    """Quick one-shot text test."""
    config = TestConfig(server_url=server_url, session_id=session_id)
    client = TestClient(config)
    client.test_text_interaction(text, play_response=False)


def quick_test_voice(server_url: str, duration: float = 5.0, session_id: str = "quick-test"):
    """Quick one-shot voice test."""
    config = TestConfig(server_url=server_url, session_id=session_id, record_seconds=duration)
    client = TestClient(config)
    client.test_voice_interaction(play_response=PLAYBACK_AVAILABLE)


def quick_test_file(server_url: str, file_path: str, session_id: str = "quick-test"):
    """Quick one-shot file test."""
    config = TestConfig(server_url=server_url, session_id=session_id)
    client = TestClient(config)
    client.test_file_interaction(file_path, play_response=PLAYBACK_AVAILABLE)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Robot Voice Brain Test Client")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--session",
        default="test-session",
        help="Session ID (default: test-session)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Quick test with text input",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Quick test with audio file",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Quick test with voice recording",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration for voice mode (default: 5.0)",
    )
    
    args = parser.parse_args()
    
    # Quick test modes
    if args.text:
        quick_test_text(args.server, args.text, args.session)
        return
    
    if args.file:
        quick_test_file(args.server, args.file, args.session)
        return
    
    if args.voice:
        quick_test_voice(args.server, args.duration, args.session)
        return
    
    # Interactive mode
    config = TestConfig(
        server_url=args.server,
        session_id=args.session,
        record_seconds=args.duration,
    )
    
    client = TestClient(config)
    client.run_interactive()


if __name__ == "__main__":
    main()
