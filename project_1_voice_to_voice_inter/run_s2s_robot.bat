@echo off
REM ============================================================
REM  Robot S2S Client Launcher (Cached Audio Playback)
REM  Connects to the speech-to-speech pipeline via sockets.
REM  TTS audio is cached for smooth, lag-free playback.
REM
REM  Usage:
REM    1. Start the S2S pipeline first (in another terminal):
REM       run_s2s_server.bat
REM
REM    2. Then run this script:
REM       run_s2s_robot.bat
REM
REM    Or connect to a remote pipeline:
REM       run_s2s_robot.bat 192.168.1.100
REM ============================================================

call venv\Scripts\activate.bat

set HOST=%1
if "%HOST%"=="" set HOST=localhost

echo.
echo Starting Robot S2S Client (Cached Playback)...
echo   Pipeline host: %HOST%
echo   Audio send:    %HOST%:12345
echo   Audio recv:    %HOST%:12346
echo   Commands:      %HOST%:12347
echo   Mode:          Cached audio playback (smooth, no jitter)
echo.

python robot_client_s2s.py --host %HOST%
