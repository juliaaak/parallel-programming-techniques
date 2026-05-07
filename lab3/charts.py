"""
Lab 3 - Chart generator
Run this after main.py to regenerate charts in the lab3 directory.
Or it is called automatically from main.py.

Saves three separate PNG files:
  lab3_chart1_time.png      - execution time vs thread count
  lab3_chart2_discrepancy.png - race condition discrepancy vs thread count
  lab3_chart3_ipc.png       - IPC round-trip latency
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save_chart1_time(threads, unsafe_times, safe_times):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threads, unsafe_times, "o-", color="#e74c3c", lw=2.5, ms=8, label="Unsafe (race condition)")
    ax.plot(threads, safe_times,   "s-", color="#2ecc71", lw=2.5, ms=8, label="Safe (with locks)")
    ax.set_title("Execution time vs number of threads", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of threads", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lab3_chart1_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_chart2_discrepancy(threads, unsafe_disc):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threads, unsafe_disc, "o-", color="#e67e22", lw=2.5, ms=8)
    ax.fill_between(threads, unsafe_disc, alpha=0.15, color="#e67e22")
    for x, y in zip(threads, unsafe_disc):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_title("Race condition: |discrepancy| vs number of threads", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of threads", fontsize=11)
    ax.set_ylabel("|Discrepancy| (currency units)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lab3_chart2_discrepancy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_chart3_ipc(method_names, latencies):
    colors = ["#3498db", "#9b59b6", "#e74c3c", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(method_names))
    bars = ax.bar(x, latencies, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(latencies) * 0.02,
                f"{val:.0f} μs", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("IPC round-trip latency comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Latency (μs)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylim(0, max(latencies) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lab3_chart3_ipc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all(unsafe_hyper: dict, safe_hyper: dict, ipc_results: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    threads      = sorted(unsafe_hyper.keys())
    unsafe_times = [unsafe_hyper[n][0] for n in threads]
    safe_times   = [safe_hyper[n] for n in threads]
    unsafe_disc  = [abs(unsafe_hyper[n][1]) for n in threads]

    method_names = [
        "Queue\n(message passing)",
        "Shared Memory\n+ Events",
        "Socket TCP\nPython → Python",
        "Socket TCP\nPython → Node.js",
    ]
    latencies = [v[1] for v in ipc_results.values()]

    print("\n  Generating charts...")
    save_chart1_time(threads, unsafe_times, safe_times)
    save_chart2_discrepancy(threads, unsafe_disc)
    save_chart3_ipc(method_names, latencies)
    print()
    