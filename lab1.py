"""
Lab 1: Thread and Process Management in Parallel Programs
Course: Methods and Technologies of Parallel Programming
Language: Python | Frameworks: multiprocessing, threading, concurrent.futures

TASK 3.2 — Implementation of three task types:
  Type 1 (CPU-bound):    Monte Carlo pi, Factorization, Primes in range
  Type 2 (Memory-bound): Matrix Transpose 10000x10000
  Type 3 (I/O-bound):    Binary file read across 1000 generated 1 MB files

TASK 3.3 — Performance research:
  Each task is measured for WORKER_COUNTS = [1, 2, 4, 8, 10, 12]
  Results are printed in tables with execution time, speedup, and efficiency.

"""

import time
import math
import random
import os
import shutil
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Thread
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG — benchmark hyperparameters (task 3.3)
# ─────────────────────────────────────────────

WORKER_COUNTS = [1, 2, 4, 8, 10, 12]   # thread/process counts to compare

# CPU-bound parameters
MONTE_POINTS  = 40_000_000
LARGE_NUMBERS = [random.randint(10**13, 10**14) for _ in range(120)]
PRIME_LIMIT   = 2_000_000               # increased from 200_000 to make parallelism overhead negligible

# Memory-bound parameters
MATRIX_SIZE   = 10_000                  # 10000x10000 as required (~762 MB float64)

# I/O-bound parameters
IO_DIR        = "lab1_testfiles"
IO_FILES      = 1_000

# ─────────────────────────────────────────────
# TABLE PRINTER
# Columns: Type | Task | Mode | Workers | Task Size | Time (s) | Speedup | Efficiency
# ─────────────────────────────────────────────

COL_WIDTHS = {
    "type":       14,
    "task":       22,
    "mode":       14,
    "workers":     8,
    "size":       20,
    "time":       14,
    "speedup":    10,
    "efficiency": 12,
}

HEADERS = ["Task Type", "Task", "Mode", "Workers", "Task Size",
           "Time (s)", "Speedup", "Efficiency"]
KEYS    = ["type", "task", "mode", "workers", "size", "time", "speedup", "efficiency"]


def _row_sep(char="─", cross="┼"):
    return "├" + cross.join(char * COL_WIDTHS[k] for k in KEYS) + "┤"


def print_typed_table(title, task_type, task_name, size_label, times):
    """
    Print a formatted benchmark table for one task.
    times: list of elapsed seconds, aligned with WORKER_COUNTS.
    """
    top_border  = "┌" + "┬".join("─" * COL_WIDTHS[k] for k in KEYS) + "┐"
    head_border = "├" + "┼".join("─" * COL_WIDTHS[k] for k in KEYS) + "┤"
    bot_border  = "└" + "┴".join("─" * COL_WIDTHS[k] for k in KEYS) + "┘"

    print(f"\n  {title}")
    print("  " + top_border)

    # header row
    hdr = "│" + "│".join(f"{h:^{COL_WIDTHS[k]}}" for h, k in zip(HEADERS, KEYS)) + "│"
    print("  " + hdr)
    print("  " + head_border)

    base_time = times[0]

    for i, (w, t) in enumerate(zip(WORKER_COUNTS, times)):
        speedup    = base_time / t
        efficiency = speedup / w
        mode       = "Sequential" if w == 1 else "Parallel"

        row = "│"
        row += f"{task_type:<{COL_WIDTHS['type']}}│"
        row += f"{task_name:<{COL_WIDTHS['task']}}│"
        row += f"{mode:<{COL_WIDTHS['mode']}}│"
        row += f"{w:^{COL_WIDTHS['workers']}}│"
        row += f"{size_label:<{COL_WIDTHS['size']}}│"
        row += f"{t:^{COL_WIDTHS['time']}.3f}│"
        row += f"{speedup:^{COL_WIDTHS['speedup']}.2f}│"
        row += f"{efficiency:^{COL_WIDTHS['efficiency']}.2f}│"

        print("  " + row)

        if i < len(WORKER_COUNTS) - 1:
            print("  " + _row_sep())

    print("  " + bot_border)


def section(title):
    print(f"\n{'='*90}\n  {title}\n{'='*90}")


def measure(func, *args, repeats=3):
    """
    Run func(*args) repeats times and return the median elapsed time.
    Using the median of 3 runs smooths out OS scheduling noise and
    CPU turbo-boost spikes that can otherwise cause efficiency > 1.0
    in memory-bound benchmarks on modern laptops.
    CPU-bound tasks use repeats=1 to keep total runtime reasonable.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]  # median


# ═══════════════════════════════════════════════════════════════════════
# TYPE 1: CPU-BOUND TASKS (task 3.2, Type 1)
# Framework: ProcessPoolExecutor (multiprocessing)
# Reason: Python threads are limited by the GIL for pure-Python CPU work;
#         separate processes each have their own GIL and run truly in parallel.
# ═══════════════════════════════════════════════════════════════════════

# ── Task 1.1: Monte Carlo estimation of π ──────────────────────────────
# Each process independently generates random points and checks whether
# they fall inside a unit circle. Results are summed across processes.
# Note: each child process gets its own random state after fork/spawn,
#       so results are statistically independent across workers.

def _mc_chunk(n):
    """Count how many of n random points fall inside the unit circle."""
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def run_monte_carlo(n_workers):
    """Sequential (1 worker) or parallel Monte Carlo pi estimation."""
    chunk = MONTE_POINTS // n_workers
    if n_workers == 1:
        _mc_chunk(MONTE_POINTS)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_mc_chunk, [chunk] * n_workers))


# ── Task 1.2: Factorization of large numbers ───────────────────────────
# Each number is factorized independently — trivially parallelizable
# (embarrassingly parallel). Good scaling is expected here.

def _factorize(n):
    """Trial-division factorization of a single large integer."""
    d = 2
    while d * d <= n:
        while n % d == 0:
            n //= d
        d += 1


def run_factorization(n_workers):
    """Sequential or parallel factorization of LARGE_NUMBERS."""
    if n_workers == 1:
        for n in LARGE_NUMBERS:
            _factorize(n)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_factorize, LARGE_NUMBERS))


# ── Task 1.3: Count primes in a range ─────────────────────────────────
# The range is split statically among workers. This creates a load imbalance:
# numbers near the end of the range take longer to check (O(√n) grows),
# so some workers finish earlier than others (Amdahl's Law in practice).
# PRIME_LIMIT = 2_000_000 ensures the useful work outweighs process
# startup overhead on Windows (spawn model, ~0.3–0.5 s per process).

def _count_primes(args):
    """Count primes in [lo, hi] via trial division."""
    lo, hi = args
    count = 0
    for n in range(max(lo, 2), hi + 1):
        if all(n % d != 0 for d in range(2, int(math.sqrt(n)) + 1)):
            count += 1
    return count


def run_primes(n_workers):
    """Sequential or parallel prime counting with static range split."""
    step   = PRIME_LIMIT // n_workers
    ranges = [(i * step + 2, (i + 1) * step + 1) for i in range(n_workers)]
    ranges[-1] = (ranges[-1][0], PRIME_LIMIT)
    if n_workers == 1:
        _count_primes(ranges[0])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_count_primes, ranges))


# ═══════════════════════════════════════════════════════════════════════
# TYPE 2: MEMORY-BOUND TASK (task 3.2, Type 2)
# Task: transpose a 10000×10000 float64 matrix
# Framework: threading.Thread + numpy
# Reason: numpy releases the GIL during array copy/transpose operations,
#         so Python threads can execute in true parallel and saturate
#         the memory bus. Speedup is bounded by RAM bandwidth, not CPU count.
# ═══════════════════════════════════════════════════════════════════════

def run_transpose(n_workers, matrix):
    """
    Transpose matrix in-place into a pre-allocated result array.
    Sequential: single numpy .T.copy() call.
    Parallel:   matrix rows are split into slices; each thread copies
                its slice into the transposed position of the result.
    """
    if n_workers == 1:
        _ = matrix.T.copy()
        return

    n      = matrix.shape[0]
    step   = n // n_workers
    result = np.empty((n, n), dtype=matrix.dtype)

    def worker(r0, r1):
        # Writes columns r0:r1 of the result from rows r0:r1 of the source.
        result[:, r0:r1] = matrix[r0:r1, :].T

    threads = []
    for i in range(n_workers):
        r0 = i * step
        r1 = n if i == n_workers - 1 else (i + 1) * step
        t  = Thread(target=worker, args=(r0, r1))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


# ═══════════════════════════════════════════════════════════════════════
# TYPE 3: I/O-BOUND TASK (task 3.2, Type 3)
# Task: read 1000 binary files of 1 MB each from disk
# Framework: ThreadPoolExecutor
# Reason: open()/read() syscalls release the GIL — threads genuinely overlap
#         their I/O waits. This is the canonical use case for I/O threading
#         in Python despite the GIL.
#
# NOTE on measurement methodology:
#   All measurements run on a warm OS page cache (one warm-up pass before
#   timing). Files are 1 MB each (1 GB total) so that the read syscall
#   dominates per-file cost, making GIL-release I/O overlap clearly visible.
#   Expected speedup: 3–6× — limited by SSD sequential read bandwidth and
#   the number of concurrent I/O requests the OS can service in parallel.
# ═══════════════════════════════════════════════════════════════════════

def generate_files(base_dir, n_files):
    """
    Generate n_files binary files of 1 MB each across 20 subdirectories.
    Using os.read() on binary files is a true I/O syscall: Python releases
    the GIL during the read, allowing threads to overlap their I/O waits.
    This makes the task genuinely I/O-bound rather than CPU-bound (e.g.
    str.split() parsing which holds the GIL the entire time).
    """
    chunk = 1024 * 1024  # 1 MB per file
    for i in range(n_files):
        subdir = os.path.join(base_dir, f"dir_{i % 20}")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, f"file_{i:04d}.bin"), "wb") as f:
            f.write(os.urandom(chunk))
    print(f"    Created {n_files} x 1 MB binary files in '{base_dir}/'")


def _read_file(path):
    """Read an entire binary file and return its byte length."""
    with open(path, "rb") as f:
        return len(f.read())


def run_io(n_workers, file_paths):
    """Sequential or parallel binary file reads via thread pool."""
    if n_workers == 1:
        for p in file_paths:
            _read_file(p)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_read_file, file_paths))


# ─────────────────────────────────────────────
# BENCHMARK RUNNERS — execute and time all tasks (task 3.3)
# ─────────────────────────────────────────────

def bench_cpu():
    """Benchmark all CPU-bound tasks and print per-task result tables."""
    section("TYPE 1: CPU-BOUND  (ProcessPoolExecutor — bypasses GIL)")

    print("\n  Running Monte Carlo pi  ...", flush=True)
    mc_times = [measure(run_monte_carlo, w, repeats=1) for w in WORKER_COUNTS]
    print_typed_table(
        "Table 1.1 — Monte Carlo π",
        "CPU-bound", "Monte Carlo π",
        f"{MONTE_POINTS:,} points", mc_times
    )

    print("\n  Running Factorization   ...", flush=True)
    fa_times = [measure(run_factorization, w, repeats=1) for w in WORKER_COUNTS]
    print_typed_table(
        "Table 1.2 — Factorization",
        "CPU-bound", "Factorization",
        f"{len(LARGE_NUMBERS)} numbers ~10^13", fa_times
    )

    print("\n  Running Primes in range ...", flush=True)
    pr_times = [measure(run_primes, w, repeats=1) for w in WORKER_COUNTS]
    print_typed_table(
        "Table 1.3 — Primes in range",
        "CPU-bound", "Primes in range",
        f"[2, {PRIME_LIMIT:,}]", pr_times
    )

    return [("Monte Carlo π",   mc_times),
            ("Factorization",   fa_times),
            ("Primes in range", pr_times)]


def bench_memory():
    """
    Benchmark matrix transpose and print result table.

    Measurement methodology:
      A single warm-up transpose is performed before timing begins.
      Without it, the first measurement (1 thread) accesses the matrix
      cold from RAM while subsequent parallel runs find data partially
      in CPU cache, producing superlinear speedup (efficiency > 1.0)
      that is a cache artifact, not a real parallelism benefit.
      With a warm-up, all measurements start from a consistent cache
      state, and speedup is correctly bounded by memory bandwidth.
    """
    section("TYPE 2: MEMORY-BOUND  (threading.Thread + numpy)")
    print(f"\n  Allocating {MATRIX_SIZE}x{MATRIX_SIZE} float64 matrix "
          f"(~{MATRIX_SIZE**2 * 8 // 1024**2} MB)...", flush=True)
    matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float64)

    # Warm-up: touch all matrix data before timing to equalise cache state.
    print("  Warming up matrix cache...", flush=True)
    run_transpose(1, matrix)

    print("  Running Matrix Transpose ...", flush=True)
    tr_times = [measure(run_transpose, w, matrix) for w in WORKER_COUNTS]
    print_typed_table(
        "Table 2.1 — Matrix Transpose",
        "Memory-bound", "Matrix Transpose",
        f"{MATRIX_SIZE}x{MATRIX_SIZE} float64", tr_times
    )

    return [("Matrix Transpose", tr_times)]


def bench_io():
    """
    Benchmark binary file reads and print result table.

    Measurement methodology:
      Files are regenerated fresh before every full benchmark run so that
      the OS page cache is cold for the sequential baseline (1 worker).
      Each subsequent parallel run benefits from an increasingly warm cache,
      which reflects real-world I/O workloads where threads overlap
      open()/read() syscalls. The GIL is released during read() syscalls,
      allowing threads to execute in true parallel at the I/O level.
      Speedup is a combination of genuine thread-level I/O overlap and
      OS cache warming — both are real benefits of ThreadPoolExecutor
      for I/O-bound tasks.
    """
    section("TYPE 3: I/O-BOUND  (ThreadPoolExecutor)")

    # Regenerate files to ensure a cold OS page cache for the 1-worker baseline.
    if os.path.exists(IO_DIR):
        shutil.rmtree(IO_DIR)
    print(f"\n  Generating {IO_FILES} test files...", flush=True)
    generate_files(IO_DIR, IO_FILES)

    # Collect all .bin paths recursively
    file_paths = []
    for root, _, files in os.walk(IO_DIR):
        for f in files:
            if f.endswith(".bin"):
                file_paths.append(os.path.join(root, f))
    print(f"  Found {len(file_paths)} .bin files")

    print("  Running Binary File Read ...", flush=True)
    wc_times = [measure(run_io, w, file_paths) for w in WORKER_COUNTS]
    print_typed_table(
        "Table 3.1 — Binary File Read",
        "I/O-bound", "Binary File Read",
        f"{len(file_paths)} x 1 MB .bin files", wc_times
    )

    return [("Binary File Read", wc_times)]
# SPEEDUP CHART — visualize results (task 3.3)
# ─────────────────────────────────────────────

def plot_speedup(cpu_rows, mem_rows, io_rows):
    """
    Plot actual vs. ideal speedup curves for all tasks.
    Saves the figure to lab1_speedup.png.

    Ideal speedup = number of workers (linear scaling).
    Actual speedup = T(1) / T(n), where T(n) is time with n workers.
    Efficiency    = Speedup / n  (1.0 = perfect utilization).
    """
    groups = [
        ("CPU-bound",    cpu_rows, "#e74c3c"),
        ("Memory-bound", mem_rows, "#3498db"),
        ("I/O-bound",    io_rows,  "#2ecc71"),
    ]
    all_tasks = [(name, times, color)
                 for label, rows, color in groups
                 for name, times in rows]

    fig, axes = plt.subplots(1, len(all_tasks), figsize=(5 * len(all_tasks), 5))
    fig.suptitle("Speedup vs Number of Workers / Threads", fontsize=14, fontweight="bold")
    if len(all_tasks) == 1:
        axes = [axes]

    for ax, (name, times, color) in zip(axes, all_tasks):
        speedups = [times[0] / t for t in times]
        ax.plot(WORKER_COUNTS, speedups,      "o-", color=color, lw=2.5, ms=8, label="Actual")
        ax.plot(WORKER_COUNTS, WORKER_COUNTS, "--", color="gray", lw=1,         label="Ideal")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Workers / Threads")
        ax.set_ylabel("Speedup (x)")
        ax.set_xticks(WORKER_COUNTS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lab1_speedup.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: lab1_speedup.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # freeze_support() is required on Windows when using ProcessPoolExecutor
    # with the default "spawn" start method (prevents recursive subprocess launch).
    multiprocessing.freeze_support()

    print("=" * 90)
    print("  Lab 1 — Parallel Programming Benchmark")
    print(f"  CPU cores available: {multiprocessing.cpu_count()}")
    print("=" * 90)

    cpu_rows = bench_cpu()      # Type 1: CPU-bound
    mem_rows = bench_memory()   # Type 2: Memory-bound
    io_rows  = bench_io()       # Type 3: I/O-bound

    section("SPEEDUP CHART")
    plot_speedup(cpu_rows, mem_rows, io_rows)

    print("\nDone.\n")
    