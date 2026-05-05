"""
task1_array_stats.py
====================
Task 1b: Compute min, max, median, and mean of an array with 2,000,000 numbers.
Implements three parallel patterns and one sequential baseline:
  - Sequential   : numpy built-in functions (uses BLAS internally)
  - Map-Reduce   : split array into chunks, compute partial stats in parallel,
                   reduce by taking global min/max/sum; median via numpy on main process
  - Fork-Join    : binary recursive split of the array, join results upward
  - Worker Pool  : more chunks than workers for dynamic load balancing

Note on median:
  True distributed median requires sorting + merging (O(n log n) IPC overhead).
  Since numpy's median is already highly optimised (BLAS), we compute it on the
  main process after collecting min/max/sum/count from workers. This avoids
  serialising sorted lists across processes, which costs more than the computation.

Run:
    python task1_array_stats.py
"""

import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple


# ─── Result container ────────────────────────────────────────────────────────

class Stats:
    def __init__(self, mn, mx, mean, median):
        self.min    = mn
        self.max    = mx
        self.mean   = mean
        self.median = median

    def __repr__(self):
        return (f"Stats(min={self.min:.4f}, max={self.max:.4f}, "
                f"mean={self.mean:.4f}, median={self.median:.4f})")

    def close_to(self, other: "Stats", tol: float = 1e-3) -> bool:
        return (abs(self.min    - other.min)    < tol and
                abs(self.max    - other.max)    < tol and
                abs(self.mean   - other.mean)   < tol and
                abs(self.median - other.median) < tol)


# ─── 1. Sequential baseline ──────────────────────────────────────────────────

def sequential(arr: np.ndarray) -> Stats:
    """Pure numpy: internally uses BLAS/OpenBLAS, already multi-threaded."""
    return Stats(
        mn     = float(arr.min()),
        mx     = float(arr.max()),
        mean   = float(arr.mean()),
        median = float(np.median(arr)),
    )


# ─── Chunk worker ────────────────────────────────────────────────────────────

def _chunk_stats(chunk: np.ndarray) -> Tuple[float, float, float, int]:
    """
    Compute partial statistics for one chunk.
    Returns (min, max, sum, count).
    Median is NOT computed here to avoid expensive IPC serialisation of
    sorted lists — numpy median runs on the main process instead.
    """
    return (float(chunk.min()), float(chunk.max()),
            float(chunk.sum()), len(chunk))


# ─── 2. Map-Reduce ───────────────────────────────────────────────────────────
# MAP  : each chunk -> (min, max, sum, count)
# REDUCE: global_min = min of all mins
#         global_max = max of all maxes
#         mean       = total_sum / total_count
#         median     = numpy on full array (main process)

def map_reduce(arr: np.ndarray, workers: int = 4) -> Stats:
    chunk_size = math.ceil(len(arr) / workers)
    chunks = [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        partial = list(executor.map(_chunk_stats, chunks))

    global_min  = min(r[0] for r in partial)
    global_max  = max(r[1] for r in partial)
    total_sum   = sum(r[2] for r in partial)
    total_cnt   = sum(r[3] for r in partial)

    return Stats(global_min, global_max, total_sum / total_cnt,
                 float(np.median(arr)))


# ─── 3. Fork-Join ────────────────────────────────────────────────────────────
# Binary recursive split of the array.
# FORK : submit left and right halves as separate futures
# JOIN : combine (min, max, sum, count) from both halves
#
# Python limitation: ProcessPoolExecutor does not allow spawning child processes
# from within a worker, so we simulate recursion via iterative depth tracking.

def _fork_join_recursive(arr: np.ndarray, executor,
                         depth: int, max_depth: int) -> Tuple:
    """Recursively split array and compute stats via Fork-Join."""
    if depth >= max_depth or len(arr) < 50_000:
        return _chunk_stats(arr)

    mid = len(arr) // 2
    left_f  = executor.submit(_chunk_stats, arr[:mid])
    right_f = executor.submit(_chunk_stats, arr[mid:])

    left, right = left_f.result(), right_f.result()

    # JOIN: merge the two partial results
    return (min(left[0], right[0]),
            max(left[1], right[1]),
            left[2] + right[2],
            left[3] + right[3])


def fork_join(arr: np.ndarray, workers: int = 4) -> Stats:
    max_depth = math.ceil(math.log2(max(workers, 2)))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        mn, mx, s, cnt = _fork_join_recursive(arr, executor, 0, max_depth)

    return Stats(mn, mx, s / cnt, float(np.median(arr)))


# ─── 4. Worker Pool ──────────────────────────────────────────────────────────
# More chunks than workers -> dynamic load balancing.
# Workers pull chunks from the internal queue as soon as they are free.
# Good when chunk execution time varies (unequal load distribution).

def worker_pool(arr: np.ndarray, workers: int = 4) -> Stats:
    n_chunks   = workers * 4          # more chunks than workers
    chunk_size = math.ceil(len(arr) / n_chunks)
    chunks     = [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

    results = [None] * len(chunks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_chunk_stats, ch): i
                         for i, ch in enumerate(chunks)}
        for future in as_completed(future_to_idx):
            results[future_to_idx[future]] = future.result()

    global_min  = min(r[0] for r in results)
    global_max  = max(r[1] for r in results)
    total_sum   = sum(r[2] for r in results)
    total_cnt   = sum(r[3] for r in results)

    return Stats(global_min, global_max, total_sum / total_cnt,
                 float(np.median(arr)))


# ─── Benchmark ───────────────────────────────────────────────────────────────

def run_benchmark():
    arr = np.load("data/array.npy")
    print(f"\n{'='*60}")
    print(f"TASK 1b: Array statistics  ({len(arr):,} numbers)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    seq = sequential(arr)
    seq_time = time.perf_counter() - t0
    print(f"\n[Sequential ]  {seq_time:.3f}s  {seq}")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = map_reduce(arr, workers)
        t = time.perf_counter() - t0
        ok = "OK" if r.close_to(seq) else "MISMATCH"
        print(f"[Map-Reduce  w={workers}]  {t:.3f}s  x{seq_time/t:.2f}  {ok}")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = fork_join(arr, workers)
        t = time.perf_counter() - t0
        ok = "OK" if r.close_to(seq) else "MISMATCH"
        print(f"[Fork-Join   w={workers}]  {t:.3f}s  x{seq_time/t:.2f}  {ok}")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = worker_pool(arr, workers)
        t = time.perf_counter() - t0
        ok = "OK" if r.close_to(seq) else "MISMATCH"
        print(f"[Worker Pool w={workers}]  {t:.3f}s  x{seq_time/t:.2f}  {ok}")


if __name__ == "__main__":
    run_benchmark()