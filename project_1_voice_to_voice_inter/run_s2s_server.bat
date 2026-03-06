@echo off
REM  Robot S2S Server
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

call venv\Scripts\activate.bat
cd /d speech-to-speech
python s2s_pipeline.py --mode socket