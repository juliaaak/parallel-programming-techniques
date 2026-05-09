"""
Console chat client.
Runs two threads: receiver (background) + sender (main input loop).
Commands:
  /pm <user> <text>       — private message
  /join <group>           — join group
  /leave <group>          — leave group
  /group <group> <text>   — send to group
  /file <user> <path>     — send file to user
  /gfile <group> <path>   — send file to group
  /list                   — list online users
  /groups                 — list groups
  /quit                   — disconnect
"""

import socket
import threading
import base64
import os
import sys
import json

# allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.protocol import (
    HOST, PORT, HEADER_SIZE, BUFFER_SIZE, MAX_FILE_SIZE,
    MsgType, encode, decode_header
)

DOWNLOADS_DIR = os.path.join(os.path.dirname(__file__), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

HELP_TEXT = """\
Commands:
  /pm <user> <text>      — private message
  /join <group>          — join a group
  /leave <group>         — leave a group
  /group <group> <text>  — send message to group
  /file <user> <path>    — send file to user
  /gfile <group> <path>  — send file to group
  /list                  — show online users
  /groups                — show groups
  /help                  — this help
  /quit                  — disconnect
  <text>                 — broadcast to everyone
"""


# ══════════════════════════════════════════════════════════════════════════════
# ChatClient
# ══════════════════════════════════════════════════════════════════════════════

class ChatClient:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.username = ""
        self._running = False

    # ── connection ─────────────────────────────────────────────────────────────

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self._running = True

    def _send(self, payload: dict):
        try:
            self.sock.sendall(encode(payload))
        except OSError:
            self._running = False

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _recv_message(self) -> dict | None:
        header = self._recv_exact(HEADER_SIZE)
        if header is None:
            return None
        length = decode_header(header)
        raw = self._recv_exact(length)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    # ── receiver thread ────────────────────────────────────────────────────────

    def _receiver(self):
        while self._running:
            msg = self._recv_message()
            if msg is None:
                print("\n[!] Disconnected from server.")
                self._running = False
                break
            self._handle_incoming(msg)

    def _handle_incoming(self, msg: dict):
        t = msg.get("type")

        if t == MsgType.SERVER_MSG:
            time = msg.get("time", "")
            prefix = f"[{time}] " if time else ""
            print(f"\n{prefix}[SERVER] {msg.get('text', '')}")

        elif t == MsgType.CHAT_MSG:
            frm  = msg.get("from", "?")
            to   = msg.get("to", "")
            text = msg.get("text", "")
            time = msg.get("time", "")
            priv = msg.get("private", False)
            tag  = "(private)" if priv else f"→ {to}"
            print(f"\n[{time}] {frm} {tag}: {text}")

        elif t == MsgType.USER_LIST:
            users = msg.get("users", [])
            print(f"\n[Online users]: {', '.join(users) or 'none'}")

        elif t == MsgType.GROUP_LIST:
            groups = msg.get("groups", {})
            if not groups:
                print("\n[Groups]: none")
            else:
                lines = [f"  #{g}: {', '.join(m)}" for g, m in groups.items()]
                print("\n[Groups]:\n" + "\n".join(lines))

        elif t == MsgType.FILE_RECV:
            self._save_file(msg)

        # prompt stays on new line
        print(f"{self.username}> ", end="", flush=True)

    def _save_file(self, msg: dict):
        frm      = msg.get("from", "?")
        filename = os.path.basename(msg.get("filename", "file"))
        data_b64 = msg.get("data", "")
        time     = msg.get("time", "")
        try:
            data = base64.b64decode(data_b64)
            path = os.path.join(DOWNLOADS_DIR, filename)
            # avoid overwrite — append sender prefix
            if os.path.exists(path):
                name, ext = os.path.splitext(filename)
                path = os.path.join(DOWNLOADS_DIR, f"{name}_{frm}{ext}")
            with open(path, "wb") as f:
                f.write(data)
            print(f"\n[{time}] [FILE from {frm}] saved → {path}")
        except Exception as e:
            print(f"\n[!] Failed to save file: {e}")

    # ── command parser ─────────────────────────────────────────────────────────

    def _parse_and_send(self, line: str):
        line = line.strip()
        if not line:
            return

        if line.startswith("/pm "):
            parts = line[4:].split(" ", 1)
            if len(parts) < 2:
                print("Usage: /pm <user> <text>")
                return
            self._send({"type": MsgType.PRIVATE, "to": parts[0], "text": parts[1]})

        elif line.startswith("/join "):
            group = line[6:].strip()
            self._send({"type": MsgType.JOIN_GROUP, "group": group})

        elif line.startswith("/leave "):
            group = line[7:].strip()
            self._send({"type": MsgType.LEAVE_GROUP, "group": group})

        elif line.startswith("/group "):
            parts = line[7:].split(" ", 1)
            if len(parts) < 2:
                print("Usage: /group <group> <text>")
                return
            self._send({"type": MsgType.GROUP_MSG, "group": parts[0], "text": parts[1]})

        elif line.startswith("/file "):
            parts = line[6:].split(" ", 1)
            if len(parts) < 2:
                print("Usage: /file <user> <path>")
                return
            self._send_file(parts[0], parts[1], group=None)

        elif line.startswith("/gfile "):
            parts = line[7:].split(" ", 1)
            if len(parts) < 2:
                print("Usage: /gfile <group> <path>")
                return
            self._send_file(parts[0], parts[1], group=parts[0])

        elif line == "/list":
            self._send({"type": MsgType.LIST_USERS})

        elif line == "/groups":
            self._send({"type": MsgType.LIST_GROUPS})

        elif line == "/help":
            print(HELP_TEXT)

        elif line == "/quit":
            self._send({"type": MsgType.DISCONNECT})
            self._running = False

        elif line.startswith("/"):
            print(f"Unknown command: {line}. Type /help.")

        else:
            # plain text → broadcast
            self._send({"type": MsgType.BROADCAST, "text": line})

    def _send_file(self, target: str, path: str, group: str | None):
        path = path.strip()
        if not os.path.isfile(path):
            print(f"[!] File not found: {path}")
            return
        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            print(f"[!] File too large (max 50 MB).")
            return
        with open(path, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("ascii")
        filename = os.path.basename(path)
        if group:
            self._send({"type": MsgType.FILE_GROUP, "group": group,
                        "filename": filename, "data": data_b64})
        else:
            self._send({"type": MsgType.FILE_SEND, "to": target,
                        "filename": filename, "data": data_b64})
        print(f"[*] Sending '{filename}' ({size} bytes)...")

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print(f"Connecting to {self.host}:{self.port}...")
        try:
            self.connect()
        except ConnectionRefusedError:
            print("[!] Cannot connect to server.")
            return

        self.username = input("Enter username: ").strip()
        if not self.username:
            print("[!] Username cannot be empty.")
            return

        # register with server
        self._send({"type": MsgType.REGISTER, "username": self.username})

        # start background receiver thread
        rx = threading.Thread(target=self._receiver, daemon=True)
        rx.start()

        print(HELP_TEXT)

        # main input loop (runs in main thread)
        try:
            while self._running:
                try:
                    line = input(f"{self.username}> ")
                except EOFError:
                    break
                if self._running:
                    self._parse_and_send(line)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            try:
                self._send({"type": MsgType.DISCONNECT})
                self.sock.close()
            except Exception:
                pass
            print("\n[*] Disconnected.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chat console client")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    ChatClient(args.host, args.port).run()