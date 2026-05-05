"""
benchmark.py — Runs all tasks, collects results, builds speedup charts.
Saves PNG files for the report.

Usage: python benchmark.py
"""

import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no GUI — save to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

# Import all implementations
from task1_html_tags   import (get_html_files, sequential as html_seq,
                                map_reduce as html_mr, fork_join as html_fj,
                                worker_pool as html_wp)
from task1_array_stats import (sequential as arr_seq,
                                map_reduce as arr_mr, fork_join as arr_fj,
                                worker_pool as arr_wp)
from task1_matrix_mult import (sequential as mat_seq,
                                map_reduce as mat_mr, fork_join as mat_fj,
                                worker_pool as mat_wp)
from task2_transactions import (sequential as tx_seq, pipeline as tx_pl,
                                 producer_consumer as tx_pc)

Path("results").mkdir(exist_ok=True)

WORKERS = [2, 4, 8]
COLORS  = {"Map-Reduce": "#4C72B0", "Fork-Join": "#DD8452", "Worker Pool": "#55A868"}
COLORS2 = {"Pipeline": "#C44E52", "Prod-Consumer": "#8172B2"}


def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def plot_speedup(title: str, seq_time: float,
                 data: dict,   # {"Pattern": {workers: time}}
                 filename: str,
                 colors: dict):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    x = np.arange(len(WORKERS))
    width = 0.25
    offsets = np.linspace(-(len(data)-1)*width/2, (len(data)-1)*width/2, len(data))

    for (pattern, times), offset in zip(data.items(), offsets):
        speedups = [seq_time / times.get(w, seq_time) for w in WORKERS]
        bars = ax.bar(x + offset, speedups, width,
                      label=pattern, color=colors.get(pattern, "#888"),
                      edgecolor="white", linewidth=0.8)
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{sp:.2f}x", ha="center", va="bottom", fontsize=8)

    # Ideal speedup line
    ideal = WORKERS
    ax.plot(x, ideal, "k--", linewidth=1, alpha=0.4, label="Ideal speedup")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w} workers" for w in WORKERS])
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_ylim(0, max(WORKERS) * 1.2)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=1, color="gray", linewidth=0.8, linestyle=":")

    plt.tight_layout()
    path = f"results/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved: {path}")


def plot_task2(seq_time: float, pl_time: float,
               pc_times: dict, filename: str):
    """Chart for Task 2 (Pipeline vs Producer-Consumer)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Task 2: Transaction processing — speedup", fontsize=13, fontweight="bold")

    # Left — method comparison with 4 consumers
    methods  = ["Sequential", "Pipeline", "Prod-Consumer\n(4 threads)"]
    times    = [seq_time, pl_time, pc_times[4]]
    bar_colors = ["#95A5A6", COLORS2["Pipeline"], COLORS2["Prod-Consumer"]]
    axes[0].bar(methods, times, color=bar_colors, edgecolor="white")
    axes[0].set_ylabel("Time (seconds)", fontsize=11)
    axes[0].set_title("Execution time", fontsize=11)
    for i, t in enumerate(times):
        axes[0].text(i, t + 0.01, f"{t:.2f}s", ha="center", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Right — Producer-Consumer speedup with varying number of consumers
    workers_list = sorted(pc_times.keys())
    speedups = [seq_time / pc_times[w] for w in workers_list]
    axes[1].plot(workers_list, speedups, "o-",
                 color=COLORS2["Prod-Consumer"], linewidth=2, markersize=7, label="Prod-Consumer")
    axes[1].axhline(y=seq_time/pl_time, color=COLORS2["Pipeline"],
                    linestyle="--", linewidth=1.5, label=f"Pipeline ({seq_time/pl_time:.2f}x)")
    axes[1].set_xlabel("Number of consumers", fontsize=11)
    axes[1].set_ylabel("Speedup (×)", fontsize=11)
    axes[1].set_title("Prod-Consumer scaling", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    for w, sp in zip(workers_list, speedups):
        axes[1].annotate(f"{sp:.2f}x", (w, sp), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)

    plt.tight_layout()
    path = f"results/{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved: {path}")


def run_all():
    print("\n" + "="*65)
    print("  FULL BENCHMARK — LAB WORK #2")
    print("="*65)

    # ── Task 1a: HTML ──────────────────────────────────────────
    print("\n[1/4] Task 1a: Counting tags in HTML files...")
    files = get_html_files()
    seq_t, _ = timeit(html_seq, files)
    print(f"  Sequential: {seq_t:.3f}s")

    html_data = {"Map-Reduce": {}, "Fork-Join": {}, "Worker Pool": {}}
    for w in WORKERS:
        t, _ = timeit(html_mr, files, w);  html_data["Map-Reduce"][w]  = t; print(f"  Map-Reduce  w={w}: {t:.3f}s  x{seq_t/t:.2f}")
        t, _ = timeit(html_fj, files, w);  html_data["Fork-Join"][w]   = t; print(f"  Fork-Join   w={w}: {t:.3f}s  x{seq_t/t:.2f}")
        t, _ = timeit(html_wp, files, w);  html_data["Worker Pool"][w] = t; print(f"  Worker Pool w={w}: {t:.3f}s  x{seq_t/t:.2f}")

    plot_speedup("Task 1a: Counting tags in 1200 HTML files",
                 seq_t, html_data, "task1a_html.png", COLORS)

    # ── Task 1b: Array ─────────────────────────────────────────
    print("\n[2/4] Task 1b: Statistics for array of 2,000,000 numbers...")
    arr = np.load("data/array.npy")
    seq_t, _ = timeit(arr_seq, arr)
    print(f"  Sequential: {seq_t:.3f}s")

    arr_data = {"Map-Reduce": {}, "Fork-Join": {}, "Worker Pool": {}}
    for w in WORKERS:
        t, _ = timeit(arr_mr, arr, w);  arr_data["Map-Reduce"][w]  = t; print(f"  Map-Reduce  w={w}: {t:.3f}s  x{seq_t/t:.2f}")
        t, _ = timeit(arr_fj, arr, w);  arr_data["Fork-Join"][w]   = t; print(f"  Fork-Join   w={w}: {t:.3f}s  x{seq_t/t:.2f}")
        t, _ = timeit(arr_wp, arr, w);  arr_data["Worker Pool"][w] = t; print(f"  Worker Pool w={w}: {t:.3f}s  x{seq_t/t:.2f}")

    plot_speedup("Task 1b: Array statistics (2,000,000 numbers)",
                 seq_t, arr_data, "task1b_array.png", COLORS)

    # ── Task 1c: Matrices ───────────────────────────────────────
    print("\n[3/4] Task 1c: Matrix multiplication 1200×1200...")
    a = np.load("data/matrix_a.npy").astype(np.float64)
    b = np.load("data/matrix_b.npy").astype(np.float64)
    seq_t, _ = timeit(mat_seq, a, b)
    print(f"  Sequential (numpy BLAS): {seq_t:.3f}s")

    mat_data = {"Map-Reduce": {}, "Fork-Join": {}, "Worker Pool": {}}
    for w in WORKERS:
        t, _ = timeit(mat_mr, a, b, w);  mat_data["Map-Reduce"][w]  = t; print(f"  Map-Reduce  w={w}: {t:.3f}s  x{seq_t/t:.2f}")
        t, _ = timeit(mat_fj, a, b, w);  mat_data["Fork-Join"][w]   = t; print(f"  Fork-Join   w={w}: {t:.3f}s  x{seq_t/t:.2f}")
        t, _ = timeit(mat_wp, a, b, w);  mat_data["Worker Pool"][w] = t; print(f"  Worker Pool w={w}: {t:.3f}s  x{seq_t/t:.2f}")

    plot_speedup("Task 1c: Matrix multiplication 1200×1200",
                 seq_t, mat_data, "task1c_matrix.png", COLORS)

    # ── Task 2: Transactions ──────────────────────────────────────
    print("\n[4/4] Task 2: Processing 500,000 transactions...")
    filepath = "data/transactions.csv"
    seq_t, _ = timeit(tx_seq, filepath)
    print(f"  Sequential: {seq_t:.3f}s")

    t_pl, _ = timeit(tx_pl, filepath)
    print(f"  Pipeline:          {t_pl:.3f}s  x{seq_t/t_pl:.2f}")

    pc_times = {}
    for n_c in [1, 2, 4, 8]:
        t, _ = timeit(tx_pc, filepath, n_c)
        pc_times[n_c] = t
        print(f"  Prod-Consumer c={n_c}: {t:.3f}s  x{seq_t/t:.2f}")

    plot_task2(seq_t, t_pl, pc_times, "task2_transactions.png")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BENCHMARK COMPLETE. Charts saved in the results/ folder")
    print("="*65)


if __name__ == "__main__":
    run_all()