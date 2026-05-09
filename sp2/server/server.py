"""
Chat server — one thread per client connection.

Handles:
  broadcast, private messages, groups, offline storage,
  file / multimedia transfer, disconnection detection,
  user registry, timestamped logging to file.

Parallel model: threading.Thread per connection + a single Lock
for all shared data structures.
"""

import socket
import threading
import logging
import logging.handlers
import base64
import os
import json
import time
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared.protocol import (
    HOST, PORT, HEADER_SIZE, MAX_FILE_SIZE, MAX_FRAME_SIZE,
    MsgType, encode, decode_header, guess_media_type,
)

# ── logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

_fmt = logging.Formatter("%(asctime)s [SERVER] %(levelname)s %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_file_h  = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_h.setFormatter(_fmt)

log = logging.getLogger("chat_server")
log.setLevel(logging.INFO)
log.addHandler(_console)
log.addHandler(_file_h)


# ══════════════════════════════════════════════════════════════════════════════
# Stats collector — used later for benchmark graphs
# ══════════════════════════════════════════════════════════════════════════════

class Stats:
    """Thread-safe counters written periodically to results/stats.json."""

    def __init__(self):
        self._lock       = threading.Lock()
        self.connections = 0     # total connections ever
        self.messages    = 0     # total messages routed
        self.bytes_sent  = 0     # total payload bytes sent
        self.files_sent  = 0
        self.peak_users  = 0
        self._timeline: list[dict] = []  # [{ts, users, msg_rate}]
        self._last_msg   = 0
        self._last_ts    = time.time()

    def conn(self):
        with self._lock:
            self.connections += 1

    def msg(self, n_bytes: int = 0):
        with self._lock:
            self.messages += 1
            self.bytes_sent += n_bytes

    def file(self):
        with self._lock:
            self.files_sent += 1

    def peak(self, n: int):
        with self._lock:
            if n > self.peak_users:
                self.peak_users = n

    def snapshot(self, current_users: int):
        now = time.time()
        with self._lock:
            dt       = now - self._last_ts or 1
            rate     = (self.messages - self._last_msg) / dt
            self._timeline.append({"ts": now, "users": current_users, "msg_rate": round(rate, 2)})
            self._last_msg = self.messages
            self._last_ts  = now

    def save(self):
        out = os.path.join(LOG_DIR, "stats.json")
        with self._lock:
            data = {
                "connections": self.connections,
                "messages":    self.messages,
                "bytes_sent":  self.bytes_sent,
                "files_sent":  self.files_sent,
                "peak_users":  self.peak_users,
                "timeline":    self._timeline,
            }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


_stats = Stats()


# ══════════════════════════════════════════════════════════════════════════════
# ClientHandler
# ══════════════════════════════════════════════════════════════════════════════

class ClientHandler(threading.Thread):
    def __init__(self, conn: socket.socket, addr, server: "ChatServer"):
        super().__init__(daemon=True)
        self.conn     = conn
        self.addr     = addr
        self.server   = server
        self.username: str | None = None
        self._running = True

    # ── low-level I/O ──────────────────────────────────────────────────────────

    def send(self, payload: dict) -> None:
        raw = encode(payload)
        try:
            self.conn.sendall(raw)
            _stats.msg(len(raw))
        except OSError:
            self._running = False

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.conn.recv(n - len(buf))
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
        if length > MAX_FRAME_SIZE:
            log.warning(f"Oversized frame {length} from {self.addr} — dropping.")
            return None
        raw = self._recv_exact(length)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    # ── thread entry ───────────────────────────────────────────────────────────

    def run(self):
        log.info(f"New connection from {self.addr}")
        _stats.conn()
        while self._running:
            msg = self._recv_message()
            if msg is None:
                break
            self._dispatch(msg)
        self._cleanup()

    # ── dispatcher ─────────────────────────────────────────────────────────────

    def _dispatch(self, msg: dict):
        t = msg.get("type")
        if t == MsgType.REGISTER:
            self._handle_register(msg)
            return
        if not self.username:
            self.send({"type": MsgType.SERVER_MSG,
                       "text": "Please register first."})
            return
        handlers = {
            MsgType.BROADCAST:   self._handle_broadcast,
            MsgType.PRIVATE:     self._handle_private,
            MsgType.GROUP_MSG:   self._handle_group_msg,
            MsgType.JOIN_GROUP:  self._handle_join_group,
            MsgType.LEAVE_GROUP: self._handle_leave_group,
            MsgType.LIST_USERS:  lambda _: self._handle_list_users(),
            MsgType.LIST_GROUPS: lambda _: self._handle_list_groups(),
            MsgType.FILE_SEND:   self._handle_file_send,
            MsgType.FILE_GROUP:  self._handle_file_group,
            MsgType.DISCONNECT:  lambda _: setattr(self, "_running", False),
        }
        fn = handlers.get(t)
        if fn:
            fn(msg)
        else:
            log.warning(f"Unknown message type '{t}' from {self.username}")

    # ── handlers ──────────────────────────────────────────────────────────────

    def _handle_register(self, msg: dict):
        name = msg.get("username", "").strip()
        if not name or len(name) > 32:
            self.send({"type": MsgType.SERVER_MSG,
                       "text": "Invalid username (1–32 non-empty chars)."})
            return
        with self.server.lock:
            if name in self.server.clients:
                self.send({"type": MsgType.SERVER_MSG,
                           "text": f"Username '{name}' is already taken."})
                return
            self.username = name
            self.server.clients[name] = self
            n = len(self.server.clients)
        _stats.peak(n)
        log.info(f"Registered: {name}  (active={n})")
        self.send({"type": MsgType.SERVER_MSG,
                   "text": f"Welcome, {name}! Type /help for commands.",
                   "time": _now()})
        self._deliver_offline()
        self.server.broadcast_server(f"{name} joined the chat.", exclude=name)
        self.server.push_user_list()

    def _handle_broadcast(self, msg: dict):
        text = msg.get("text", "").strip()
        fmt  = msg.get("fmt", "plain")
        if not text:
            return
        packet = {"type": MsgType.CHAT_MSG, "from": self.username,
                  "to": "ALL", "text": text, "fmt": fmt, "time": _now()}
        self.server.broadcast_all(packet)
        log.info(f"BROADCAST {self.username}: {text[:80]}")

    def _handle_private(self, msg: dict):
        to   = msg.get("to", "").strip()
        text = msg.get("text", "").strip()
        fmt  = msg.get("fmt", "plain")
        if not to or not text:
            return
        packet = {"type": MsgType.CHAT_MSG, "from": self.username,
                  "to": to, "text": text, "fmt": fmt,
                  "time": _now(), "private": True}
        with self.server.lock:
            target = self.server.clients.get(to)
        if target:
            target.send(packet)
            self.send(packet)
        else:
            self.server.store_offline(to, packet)
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"'{to}' is offline — message queued.", "time": _now()})
        log.info(f"PM {self.username}→{to}: {text[:80]}")

    def _handle_group_msg(self, msg: dict):
        group = msg.get("group", "").strip()
        text  = msg.get("text",  "").strip()
        fmt   = msg.get("fmt",   "plain")
        if not group or not text:
            return
        with self.server.lock:
            members = set(self.server.groups.get(group, set()))
        if self.username not in members:
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"You are not in group '{group}'."})
            return
        packet = {"type": MsgType.CHAT_MSG, "from": self.username,
                  "to": f"#{group}", "text": text, "fmt": fmt, "time": _now()}
        with self.server.lock:
            targets = [self.server.clients.get(m) for m in members]
        for t in targets:
            if t:
                t.send(packet)
        log.info(f"GROUP[{group}] {self.username}: {text[:80]}")

    def _handle_join_group(self, msg: dict):
        group = msg.get("group", "").strip()
        if not group:
            return
        with self.server.lock:
            self.server.groups[group].add(self.username)
        self.send({"type": MsgType.SERVER_MSG,
                   "text": f"Joined group '{group}'.", "time": _now()})
        self.server.push_group_list()
        log.info(f"{self.username} joined group '{group}'")

    def _handle_leave_group(self, msg: dict):
        group = msg.get("group", "").strip()
        with self.server.lock:
            was_member = self.username in self.server.groups.get(group, set())
            self.server.groups[group].discard(self.username)
            if not self.server.groups[group]:
                del self.server.groups[group]
        if was_member:
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"Left group '{group}'.", "time": _now()})
        else:
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"You were not in group '{group}'."})
        self.server.push_group_list()

    def _handle_list_users(self):
        with self.server.lock:
            users = list(self.server.clients.keys())
        self.send({"type": MsgType.USER_LIST, "users": users})

    def _handle_list_groups(self):
        with self.server.lock:
            groups = {g: list(m) for g, m in self.server.groups.items()}
        self.send({"type": MsgType.GROUP_LIST, "groups": groups})

    def _handle_file_send(self, msg: dict):
        to         = msg.get("to",         "").strip()
        filename   = msg.get("filename",   "file")
        data_b64   = msg.get("data",       "")
        media_type = msg.get("media_type") or guess_media_type(filename)
        packet = {
            "type":       MsgType.FILE_RECV,
            "from":       self.username,
            "to":         to,
            "filename":   filename,
            "data":       data_b64,
            "media_type": media_type,
            "time":       _now(),
        }
        with self.server.lock:
            target = self.server.clients.get(to)
        if target:
            target.send(packet)
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"File '{filename}' ({media_type}) sent to {to}.",
                       "time": _now()})
        else:
            self.server.store_offline(to, packet)
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"'{to}' is offline — file queued.", "time": _now()})
        _stats.file()
        log.info(f"FILE {self.username}→{to}: {filename} [{media_type}]")

    def _handle_file_group(self, msg: dict):
        group      = msg.get("group",      "").strip()
        filename   = msg.get("filename",   "file")
        data_b64   = msg.get("data",       "")
        media_type = msg.get("media_type") or guess_media_type(filename)
        with self.server.lock:
            members = set(self.server.groups.get(group, set()))
        if self.username not in members:
            self.send({"type": MsgType.SERVER_MSG,
                       "text": f"You are not in group '{group}'."})
            return
        packet = {
            "type":       MsgType.FILE_RECV,
            "from":       self.username,
            "to":         f"#{group}",
            "filename":   filename,
            "data":       data_b64,
            "media_type": media_type,
            "time":       _now(),
        }
        with self.server.lock:
            targets = [self.server.clients.get(m) for m in members
                       if m != self.username]
        for t in targets:
            if t:
                t.send(packet)
        self.send({"type": MsgType.SERVER_MSG,
                   "text": f"File '{filename}' ({media_type}) sent to #{group}.",
                   "time": _now()})
        _stats.file()
        log.info(f"GFILE {self.username}→#{group}: {filename} [{media_type}]")

    # ── offline delivery ───────────────────────────────────────────────────────

    def _deliver_offline(self):
        with self.server.lock:
            messages = self.server.offline.pop(self.username, [])
        if not messages:
            return
        self.send({"type": MsgType.SERVER_MSG,
                   "text": f"You have {len(messages)} offline message(s):",
                   "time": _now()})
        for m in messages:
            self.send(m)
        self.send({"type": MsgType.OFFLINE_DELIVERED,
                   "count": len(messages)})
        log.info(f"Delivered {len(messages)} offline msg(s) to {self.username}")

    # ── cleanup ────────────────────────────────────────────────────────────────

    def _cleanup(self):
        if not self.username:
            self.conn.close()
            return
        with self.server.lock:
            self.server.clients.pop(self.username, None)
            for members in self.server.groups.values():
                members.discard(self.username)
        log.info(f"Disconnected: {self.username}")
        self.server.broadcast_server(f"{self.username} left the chat.")
        self.server.push_user_list()
        self.conn.close()
        _stats.save()


# ══════════════════════════════════════════════════════════════════════════════
# ChatServer
# ══════════════════════════════════════════════════════════════════════════════

class ChatServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.lock    = threading.Lock()
        self.clients: dict[str, ClientHandler] = {}
        self.groups:  dict[str, set]           = defaultdict(set)
        self.offline: dict[str, list]          = defaultdict(list)
        self._snapshot_timer: threading.Timer | None = None

    # ── helpers ────────────────────────────────────────────────────────────────

    def broadcast_all(self, packet: dict):
        with self.lock:
            handlers = list(self.clients.values())
        for h in handlers:
            h.send(packet)

    def broadcast_server(self, text: str, exclude: str | None = None):
        packet = {"type": MsgType.SERVER_MSG, "text": text, "time": _now()}
        with self.lock:
            handlers = [h for u, h in self.clients.items() if u != exclude]
        for h in handlers:
            h.send(packet)

    def store_offline(self, username: str, packet: dict):
        with self.lock:
            self.offline[username].append(packet)

    def push_user_list(self):
        with self.lock:
            users    = list(self.clients.keys())
            handlers = list(self.clients.values())
        packet = {"type": MsgType.USER_LIST, "users": users}
        for h in handlers:
            h.send(packet)

    def push_group_list(self):
        with self.lock:
            groups   = {g: list(m) for g, m in self.groups.items()}
            handlers = list(self.clients.values())
        packet = {"type": MsgType.GROUP_LIST, "groups": groups}
        for h in handlers:
            h.send(packet)

    # ── periodic stats snapshot ────────────────────────────────────────────────

    def _schedule_snapshot(self):
        with self.lock:
            n = len(self.clients)
        _stats.snapshot(n)
        _stats.save()
        self._snapshot_timer = threading.Timer(5.0, self._schedule_snapshot)
        self._snapshot_timer.daemon = True
        self._snapshot_timer.start()

    # ── main accept loop ───────────────────────────────────────────────────────

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(128)
        log.info(f"Server listening on {self.host}:{self.port}")
        self._schedule_snapshot()
        try:
            while True:
                conn, addr = srv.accept()
                ClientHandler(conn, addr, self).start()
        except KeyboardInterrupt:
            log.info("Server shutting down.")
        finally:
            if self._snapshot_timer:
                self._snapshot_timer.cancel()
            _stats.save()
            srv.close()


# ── helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Chat server")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()
    ChatServer(a.host, a.port).start()
    