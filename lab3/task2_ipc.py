"""
Lab 3 - Task 2
Inter-process data transfer using four methods:
  Method 1: multiprocessing.Queue        - message passing via OS pipe (Python)
  Method 2: multiprocessing.Value        - shared memory with Event sync (Python)
  Method 3: socket TCP loopback          - Python <-> Python via TCP
  Method 4: socket TCP loopback (Node.js)- Python <-> Node.js via TCP (cross-language)

The main process generates a random number, sends it to a helper process
which logs it and echoes it back. Each method is benchmarked independently.
Only the first LOG_FIRST_N received values are printed per method.
"""

import multiprocessing
import multiprocessing.synchronize
import socket
import subprocess
import sys
import time
import random
import struct
import os

ITERATIONS = 200
LOG_FIRST_N = 3
PORT_PYTHON = 54321
PORT_NODE   = 54322

# path to Node.js helper - same directory as this script
NODE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'helper_node.js')


# ──────────────────────────────────────────────────────
# Method 1: Queue (message passing via OS pipe)
# ──────────────────────────────────────────────────────

def helper_queue(req_q: multiprocessing.Queue, res_q: multiprocessing.Queue):
    """Helper process: receive value, log it, echo back."""
    count = 0
    while True:
        value = req_q.get()
        if value is None:
            break
        count += 1
        if count <= LOG_FIRST_N:
            print(f"  [Queue helper] received {value:.6f}", flush=True)
        elif count == LOG_FIRST_N + 1:
            print(f"  [Queue helper] ... (logging first {LOG_FIRST_N} only)", flush=True)
        res_q.put(value)


def benchmark_queue(iterations: int) -> float:
    req_q: multiprocessing.Queue = multiprocessing.Queue()
    res_q: multiprocessing.Queue = multiprocessing.Queue()

    proc = multiprocessing.Process(target=helper_queue, args=(req_q, res_q))
    proc.start()

    start = time.perf_counter()
    for _ in range(iterations):
        number = random.random()
        req_q.put(number)
        res_q.get()
    elapsed = time.perf_counter() - start

    req_q.put(None)
    proc.join()
    return elapsed


# ──────────────────────────────────────────────────────
# Method 2: Shared Memory (Value + Events)
# ──────────────────────────────────────────────────────

def helper_shared(
    shared_val,
    ready_event: multiprocessing.synchronize.Event,
    done_event:  multiprocessing.synchronize.Event,
    stop_event:  multiprocessing.synchronize.Event,
):
    """
    Helper process using shared memory.
    Without Events this would be a race condition between processes -
    both could read/write shared_val simultaneously.
    Events serve as synchronisation barriers.
    """
    count = 0
    while not stop_event.is_set():
        if not ready_event.wait(timeout=1.0):
            continue
        ready_event.clear()
        count += 1
        if count <= LOG_FIRST_N:
            print(f"  [SHM helper] received {shared_val.value:.6f}", flush=True)
        elif count == LOG_FIRST_N + 1:
            print(f"  [SHM helper] ... (logging first {LOG_FIRST_N} only)", flush=True)
        done_event.set()


def benchmark_shared(iterations: int) -> float:
    shared_val  = multiprocessing.Value('d', 0.0)
    ready_event = multiprocessing.Event()
    done_event  = multiprocessing.Event()
    stop_event  = multiprocessing.Event()

    proc = multiprocessing.Process(
        target=helper_shared,
        args=(shared_val, ready_event, done_event, stop_event),
    )
    proc.start()

    start = time.perf_counter()
    for _ in range(iterations):
        number = random.random()
        with shared_val.get_lock():
            shared_val.value = number
        ready_event.set()
        done_event.wait()
        done_event.clear()
    elapsed = time.perf_counter() - start

    stop_event.set()
    ready_event.set()
    proc.join()
    return elapsed


# ──────────────────────────────────────────────────────
# Method 3: Socket TCP - Python <-> Python
# ──────────────────────────────────────────────────────

def helper_socket_python(port: int, iterations: int):
    """Python TCP server: receive float, log it, echo back."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    conn, _ = server.accept()

    for i in range(iterations):
        data = b''
        while len(data) < 8:
            data += conn.recv(8 - len(data))
        value = struct.unpack('<d', data)[0]
        if i < LOG_FIRST_N:
            print(f"  [Socket/Python helper] received {value:.6f}", flush=True)
        elif i == LOG_FIRST_N:
            print(f"  [Socket/Python helper] ... (logging first {LOG_FIRST_N} only)", flush=True)
        conn.sendall(struct.pack('<d', value))

    conn.close()
    server.close()


def benchmark_socket_python(iterations: int) -> float:
    proc = multiprocessing.Process(
        target=helper_socket_python, args=(PORT_PYTHON, iterations))
    proc.start()
    time.sleep(0.15)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', PORT_PYTHON))

    start = time.perf_counter()
    for _ in range(iterations):
        number = random.random()
        client.sendall(struct.pack('<d', number))
        data = b''
        while len(data) < 8:
            data += client.recv(8 - len(data))
    elapsed = time.perf_counter() - start

    client.close()
    proc.join()
    return elapsed


# ──────────────────────────────────────────────────────
# Method 4: Socket TCP - Python <-> Node.js (cross-language)
# ──────────────────────────────────────────────────────

def benchmark_socket_node(iterations: int) -> float:
    """
    Starts helper_node.js as a subprocess, waits for READY signal,
    then performs round-trip float transfers via TCP socket.
    Demonstrates cross-language IPC: Python sends, Node.js logs and echoes.
    """
    node_proc = subprocess.Popen(
        ['node', NODE_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # wait for Node.js server to signal it is ready
    while True:
        line = node_proc.stdout.readline().strip()
        if line == 'READY':
            break

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', PORT_NODE))

    start = time.perf_counter()
    for _ in range(iterations):
        number = random.random()
        client.sendall(struct.pack('<d', number))
        data = b''
        while len(data) < 8:
            data += client.recv(8 - len(data))
    elapsed = time.perf_counter() - start

    client.close()

    # drain Node.js stdout (log lines) and print them
    node_proc.stdin = None
    remaining, _ = node_proc.communicate(timeout=5)
    for line in remaining.splitlines():
        if line.strip():
            print(line, flush=True)

    node_proc.wait()
    return elapsed


# ──────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────

def run_task2():
    print("=" * 60)
    print("TASK 2 - Inter-Process Communication Benchmark")
    print("=" * 60)
    print(f"\n  Iterations per method: {ITERATIONS}\n")

    methods = [
        ("Queue (message passing) [Python]      ", benchmark_queue),
        ("Shared Memory + Events  [Python]      ", benchmark_shared),
        ("Socket TCP              [Python->Python]", benchmark_socket_python),
        ("Socket TCP              [Python->Node.js]", benchmark_socket_node),
    ]

    results = {}
    for name, fn in methods:
        t = fn(ITERATIONS)
        avg_us = (t / ITERATIONS) * 1_000_000
        results[name] = (t, avg_us)
        print(f"  {name}")
        print(f"    total={t:.4f}s  avg/round-trip={avg_us:.1f} us\n")

    ranked = sorted(results.items(), key=lambda kv: kv[1][0])
    print("  Ranking (fastest -> slowest):")
    for rank, (name, (_, avg_us)) in enumerate(ranked, 1):
        print(f"    {rank}. {name.strip()}: {avg_us:.1f} us/round-trip")

    return results


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_task2()
    