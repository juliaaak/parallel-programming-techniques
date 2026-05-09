"""
main.py — launcher / demo runner
─────────────────────────────────
Usage:
  python main.py server              — start the server
  python main.py client              — connect as a client
  python main.py client --host <IP>  — connect to remote server
  python main.py bench               — run automated load test
  python main.py graphs              — generate PNG graphs from results
  python main.py demo                — run a self-contained local demo
                                       (server + 10 bots, no user input needed)
"""

import argparse
import os
import sys
import threading
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from shared.protocol import CLIENT_HOST, PORT


# ── helpers ────────────────────────────────────────────────────────────────────

def run_server(host, port):
    from server.server import ChatServer
    ChatServer(host, port).start()


def run_client(host, port):
    from client.client import ChatClient
    ChatClient(host, port).run()


def run_bench(host, port, clients, duration):
    from benchmark.benchmark import run_benchmark
    run_benchmark(host, port, clients, duration)


def run_graphs():

    graphs_path = os.path.join(BASE, "generate_graphs.py")

    if not os.path.exists(graphs_path):

        graphs_path = os.path.join(BASE, "results", "generate_graphs.py")

    if not os.path.exists(graphs_path):
        print("[!] generate_graphs.py not found.")
        print(f"    Searched: {BASE}/generate_graphs.py")
        print(f"    Searched: {BASE}/results/generate_graphs.py")
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_graphs", graphs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()


# ── demo ───────────────────────────────────────────────────────────────────────

def run_demo(port):
    """
    1. Start server in a thread.
    2. Run benchmark (10 bots, 10 s).
    3. Generate graphs.
    """
    print("=" * 60)
    print("  DEMO: server + 10 bots for 10 s, then graphs")
    print("=" * 60)

    # start server in background thread
    from server.server import ChatServer
    srv = ChatServer("127.0.0.1", port)
    t   = threading.Thread(target=srv.start, daemon=True)
    t.start()
    time.sleep(0.8)   # let server bind

    from benchmark.benchmark import run_benchmark
    run_benchmark("127.0.0.1", port, 10, 10)

    time.sleep(0.5)
    run_graphs()
    print("\nDemo complete!  See results/ for graphs and logs.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Chat system launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("mode", choices=["server", "client", "bench", "graphs", "demo"])
    p.add_argument("--host",     default=CLIENT_HOST, help="server host (client/bench)")
    p.add_argument("--port",     type=int, default=PORT)
    p.add_argument("--clients",  type=int, default=10,  help="bots for bench")
    p.add_argument("--duration", type=int, default=20,  help="bench duration (s)")
    a = p.parse_args()

    if a.mode == "server":
        run_server("0.0.0.0", a.port)
    elif a.mode == "client":
        run_client(a.host, a.port)
    elif a.mode == "bench":
        run_bench(a.host, a.port, a.clients, a.duration)
    elif a.mode == "graphs":
        run_graphs()
    elif a.mode == "demo":
        run_demo(a.port)


if __name__ == "__main__":
    main()