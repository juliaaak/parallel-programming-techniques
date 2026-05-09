"""
results/generate_graphs.py
──────────────────────────
Reads benchmark_raw.json (and server stats.json if present)
and produces PNG charts in the results/ folder.

Run after benchmark:
    python results/generate_graphs.py
"""

import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    sys.exit("Install matplotlib and numpy first:  pip install matplotlib numpy")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── styling ────────────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
ACCENT1   = "#4f8ef7"
ACCENT2   = "#f7a24f"
ACCENT3   = "#4ff7a2"
ACCENT4   = "#f74f8e"
TEXT_COL  = "#e0e4ef"
GRID_COL  = "#2a2d3a"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.titlecolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COL,
    "legend.labelcolor": TEXT_COL,
    "font.family":       "monospace",
    "font.size":         11,
})


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Throughput over time
# ══════════════════════════════════════════════════════════════════════════════

def plot_throughput(data):
    tl   = data["timeline"]
    ts   = [p["t"]    for p in tl]
    sent = [p["sent"] for p in tl]
    # msg/s per interval
    rates = [0]
    for i in range(1, len(sent)):
        dt = ts[i] - ts[i-1] or 1
        rates.append((sent[i] - sent[i-1]) / dt)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(ts, rates, alpha=0.25, color=ACCENT1)
    ax.plot(ts, rates, color=ACCENT1, linewidth=2, label="msg / s")
    ax.axhline(data["throughput"], color=ACCENT2, linestyle="--",
               linewidth=1.4, label=f"avg {data['throughput']} msg/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Messages / s")
    ax.set_title(f"Throughput — {data['n_clients']} concurrent clients")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save("graph_throughput.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Latency distribution (histogram)
# ══════════════════════════════════════════════════════════════════════════════

def plot_latency_hist(data):
    lms = data["latency_ms"]
    # reconstruct approximate sample set from summary stats for display
    # (real samples aren't stored, so we show a bar chart of percentile buckets)
    labels  = ["min", "avg", "p95", "max"]
    values  = [lms["min"], lms["avg"], lms["p95"], lms["max"]]
    colors  = [ACCENT3, ACCENT1, ACCENT2, ACCENT4]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val} ms", ha="center", va="bottom", color=TEXT_COL, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Send latency percentiles")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save("graph_latency.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Active clients over time
# ══════════════════════════════════════════════════════════════════════════════

def plot_active_clients(data):
    tl     = data["timeline"]
    ts     = [p["t"]      for p in tl]
    active = [p["active"] for p in tl]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(ts, active, color=ACCENT3, linewidth=2, where="post", label="active bots")
    ax.fill_between(ts, active, step="post", alpha=0.15, color=ACCENT3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active clients")
    ax.set_title("Concurrent clients over time")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save("graph_active_clients.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Server stats (if available): total messages, files, peak users
# ══════════════════════════════════════════════════════════════════════════════

def plot_server_stats(stats):
    labels = ["Peak users", "Files sent", "Connections"]
    values = [stats["peak_users"], stats["files_sent"], stats["connections"]]
    colors = [ACCENT1, ACCENT2, ACCENT3]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, height=0.5, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", color=TEXT_COL, fontsize=10)
    ax.set_xlabel("Count")
    ax.set_title("Server statistics (session)")
    ax.grid(True, axis="x")
    fig.tight_layout()
    save("graph_server_stats.png")


def plot_server_timeline(stats):
    tl    = stats.get("timeline", [])
    if not tl:
        return
    ts    = [p["ts"] - tl[0]["ts"] for p in tl]   # relative seconds
    users = [p["users"]    for p in tl]
    rates = [p["msg_rate"] for p in tl]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(ts, users, color=ACCENT1, linewidth=2)
    ax1.fill_between(ts, users, alpha=0.2, color=ACCENT1)
    ax1.set_ylabel("Online users")
    ax1.set_title("Server activity over session")
    ax1.grid(True)

    ax2.plot(ts, rates, color=ACCENT2, linewidth=2)
    ax2.fill_between(ts, rates, alpha=0.2, color=ACCENT2)
    ax2.set_ylabel("msg / s (server-side)")
    ax2.set_xlabel("Elapsed (s)")
    ax2.grid(True)

    fig.tight_layout()
    save("graph_server_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Scalability: synthetic throughput vs client count
#    (linear model from benchmark results — shown if only one data point)
# ══════════════════════════════════════════════════════════════════════════════

def plot_scalability(data):
    n0         = data["n_clients"]
    tput0      = data["throughput"]
    client_counts = list(range(1, max(n0 * 2, 20) + 1, 1))

    # approximate linear scaling up to ~CPU saturation, then slight drop
    import math
    def model(n):
        # linear up to n0, then log-linear falloff
        if n <= n0:
            return tput0 * n / n0
        return tput0 * (1 + math.log(n / n0) * 0.35)

    tputs = [model(n) for n in client_counts]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(client_counts, tputs, color=ACCENT1, linewidth=2, label="estimated throughput")
    ax.axvline(n0, color=ACCENT4, linestyle="--", linewidth=1.3,
               label=f"measured ({n0} clients, {tput0} msg/s)")
    ax.scatter([n0], [tput0], color=ACCENT4, s=60, zorder=5)
    ax.set_xlabel("Concurrent clients")
    ax.set_ylabel("Messages / s")
    ax.set_title("Scalability model (threading, 1 lock)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save("graph_scalability.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating graphs…")
    bench = load("benchmark_raw.json")
    stats = load("stats.json")

    if bench:
        plot_throughput(bench)
        plot_latency_hist(bench)
        plot_active_clients(bench)
        plot_scalability(bench)
    else:
        print("  [!] benchmark_raw.json not found — run benchmark first.")

    if stats:
        plot_server_stats(stats)
        plot_server_timeline(stats)
    else:
        print("  [!] stats.json not found — start the server to generate it.")

    print("Done. All graphs saved to results/")


if __name__ == "__main__":
    main()