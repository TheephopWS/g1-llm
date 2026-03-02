"""
Socket Command Sender - sends text/action commands to a remote robot client.

Reads from text_output_queue and sends JSON messages via TCP socket
using a 4-byte big-endian length prefix protocol.

Messages include:
  - {"type": "speech_started"}       (from VAD)
  - {"type": "speech_stopped"}       (from VAD)
  - {"type": "assistant_text", "text": "...", "actions": [...]}  (from LMOutputProcessor)
"""

import json
import logging
import socket
import struct
from queue import Empty

logger = logging.getLogger(__name__)


class SocketCommandSender:
    """Sends command/action messages from the S2S pipeline to a remote robot client."""

    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12347):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info(f"Command sender waiting on {self.host}:{self.port}...")

        self.conn, addr = self.socket.accept()
        logger.info(f"Command sender connected: {addr}")

        while not self.stop_event.is_set():
            try:
                message = self.queue_in.get(timeout=0.1)
            except Empty:
                continue

            if isinstance(message, bytes) and message == b"END":
                break

            try:
                json_bytes = json.dumps(message, ensure_ascii=False).encode("utf-8")
                header = struct.pack(">I", len(json_bytes))
                self.conn.sendall(header + json_bytes)
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"Command socket disconnected: {e}")
                break
            except Exception as e:
                logger.error(f"Command sender error: {e}")

        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.socket.close()
        except Exception:
            pass
        logger.info("Command sender closed")
