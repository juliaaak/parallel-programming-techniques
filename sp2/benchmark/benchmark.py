"""
benchmark/benchmark.py
──────────────────────
Automated load-test for the chat server.
Spawns N virtual clients that send messages as fast as possible,
then writes results/benchmark_raw.json for the graph generator.

Usage (server must be running first):
    python benchmark/benchmark.py --clients 10 --duration 15
"""

import socket
import threading
import time
import json
import os
import sys
import random
import string

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared.protocol import CLIENT_HOST, PORT, encode, decode_header, MsgType, HEADER_SIZE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── tiny bot client ────────────────────────────────────────────────────────────

class BotClient(threading.Thread):
    def __init__(self, host, port, name, duration, lock):
        super().__init__(daemon=True)
        self.host      = host
        self.port      = port
        self.name      = name
        self.duration  = duration
        self.lock      = lock
        self.latencies = []

        self.sent      = 0

    def _send_raw(self, sock, payload):
        sock.sendall(encode(payload))

    def _recv_msg(self, sock):
        header = b""
        while len(header) < HEADER_SIZE:
            chunk = sock.recv(HEADER_SIZE - len(header))
            if not chunk:
                return None
            header += chunk
        length = decode_header(header)
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return json.loads(data.decode("utf-8"))

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            self._send_raw(sock, {"type": MsgType.REGISTER, "username": self.name})
            # read welcome
            self._recv_msg(sock)

            deadline = time.time() + self.duration
            while time.time() < deadline:
                msg = "".join(random.choices(string.ascii_letters, k=40))
                t0  = time.perf_counter()
                self._send_raw(sock, {"type": MsgType.BROADCAST, "text": msg, "fmt": "plain"})

                with self.lock:
                    self.sent += 1
                self.latencies.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.05)   # ~20 msg/s per bot

            self._send_raw(sock, {"type": MsgType.DISCONNECT})
            sock.close()

        except Exception as e:
            print(f"[BOT {self.name}] error: {e}")


# ── timeline sampler ───────────────────────────────────────────────────────────

def run_benchmark(host, port, n_clients, duration):
    print(f"[bench] Connecting {n_clients} bots for {duration}s …")
    lock     = threading.Lock()
    timeline = []

    bots = [
        BotClient(host, port, f"bot_{i:03d}", duration, lock)
        for i in range(n_clients)
    ]

    t_start = time.time()
    for b in bots:
        b.start()
        time.sleep(0.05)   # stagger connections slightly

    while True:
        elapsed = time.time() - t_start
        alive   = sum(1 for b in bots if b.is_alive())
        with lock:
            current_sent = sum(b.sent for b in bots)
        timeline.append({
            "t":      round(elapsed, 1),
            "active": alive,
            "sent":   current_sent,
        })
        if not alive:
            break
        time.sleep(1.0)

    for b in bots:
        b.join()

    lats = []
    total_sent = 0
    for b in bots:
        lats.extend(b.latencies)
        total_sent += b.sent

    lats.sort()
    n    = len(lats)
    result = {
        "n_clients":   n_clients,
        "duration_s":  duration,
        "total_sent":  total_sent,
        "throughput":  round(total_sent / duration, 2),
        "latency_ms": {
            "min":  round(lats[0],          3) if lats else 0,
            "avg":  round(sum(lats) / n,    3) if lats else 0,
            "p95":  round(lats[int(n*.95)], 3) if lats else 0,
            "max":  round(lats[-1],         3) if lats else 0,
        },
        "timeline": timeline,
    }

    out = os.path.join(RESULTS_DIR, "benchmark_raw.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench] Done. Results → {out}")
    print(f"        throughput: {result['throughput']} msg/s")
    print(f"        latency p95: {result['latency_ms']['p95']} ms")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host",     default=CLIENT_HOST)
    p.add_argument("--port",     type=int, default=PORT)
    p.add_argument("--clients",  type=int, default=10,
                   help="number of concurrent bot clients")
    p.add_argument("--duration", type=int, default=20,
                   help="test duration in seconds")
    a = p.parse_args()
    run_benchmark(a.host, a.port, a.clients, a.duration)