"""
task1_matrix_mult.py
====================
Task 1c: Multiply two 1200x1200 matrices using three parallel patterns.

Strategy: split matrix A into horizontal row-bands.
Each task computes its band multiplied by the full matrix B.
REDUCE: stack the result bands back in order (np.vstack).

Sequential baseline uses numpy @ (which calls BLAS/OpenBLAS internally).
Parallel versions use ProcessPoolExecutor to bypass the GIL.

Why parallel is slower here:
  numpy's BLAS is already internally parallelised and extremely optimised.
  The multiprocessing overhead (process startup, pickle serialisation of
  large numpy arrays via IPC) dominates when the baseline is already fast.
  This is an important real-world observation: parallelism is not always better.

Run:
    python task1_matrix_mult.py
"""

import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple


# ─── Row-band multiplication worker ──────────────────────────────────────────

def _multiply_rows(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Multiply a sub-matrix (row band) by the full matrix B.
    numpy @ inside a single process uses BLAS — that is fine,
    because we parallelise at the level of row bands across processes.
    """
    a_chunk, b = args
    return a_chunk @ b


# ─── 1. Sequential baseline ──────────────────────────────────────────────────

def sequential(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard numpy matrix multiplication (uses BLAS internally)."""
    return a @ b


# ─── 2. Map-Reduce ───────────────────────────────────────────────────────────
# MAP  : each process multiplies its row band of A by full B
# REDUCE: np.vstack — concatenate result bands in the correct row order

def map_reduce(a: np.ndarray, b: np.ndarray, workers: int = 4) -> np.ndarray:
    n_rows     = a.shape[0]
    chunk_size = math.ceil(n_rows / workers)
    chunks     = [(a[i : i + chunk_size], b)
                  for i in range(0, n_rows, chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        mapped = list(executor.map(_multiply_rows, chunks))

    return np.vstack(mapped)


# ─── 3. Fork-Join ────────────────────────────────────────────────────────────
# FORK : recursively split A by rows into two halves
# JOIN : vstack the two computed halves back together
#
# Difference from Map-Reduce: binary recursive split instead of linear N-way.

def _fork_join_matrix(a_rows: np.ndarray, b: np.ndarray,
                      executor, depth: int, max_depth: int) -> np.ndarray:
    """Recursively split row band and multiply via Fork-Join."""
    if depth >= max_depth or a_rows.shape[0] <= 50:
        return _multiply_rows((a_rows, b))

    mid = a_rows.shape[0] // 2

    # FORK: submit both halves
    left_f  = executor.submit(_multiply_rows, (a_rows[:mid], b))
    right_f = executor.submit(_multiply_rows, (a_rows[mid:], b))

    # JOIN: wait and combine
    return np.vstack([left_f.result(), right_f.result()])


def fork_join(a: np.ndarray, b: np.ndarray, workers: int = 4) -> np.ndarray:
    max_depth = math.ceil(math.log2(max(workers, 2)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return _fork_join_matrix(a, b, executor, 0, max_depth)


# ─── 4. Worker Pool ──────────────────────────────────────────────────────────
# More chunks than workers -> finer-grained queue, better load balancing.
# Results collected by index to preserve row order.

def worker_pool(a: np.ndarray, b: np.ndarray, workers: int = 4) -> np.ndarray:
    n_rows     = a.shape[0]
    n_chunks   = workers * 4
    chunk_size = math.ceil(n_rows / n_chunks)

    indexed_chunks = [(i, a[i : i + chunk_size])
                      for i in range(0, n_rows, chunk_size)]
    results = [None] * len(indexed_chunks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_multiply_rows, (chunk, b)): idx
            for idx, (_, chunk) in enumerate(indexed_chunks)
        }
        for future in as_completed(future_to_idx):
            results[future_to_idx[future]] = future.result()

    return np.vstack(results)


# ─── Correctness check ───────────────────────────────────────────────────────

def matrices_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-2) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=tol))


# ─── Benchmark ───────────────────────────────────────────────────────────────

def run_benchmark():
    a = np.load("data/matrix_a.npy").astype(np.float64)
    b = np.load("data/matrix_b.npy").astype(np.float64)

    print(f"\n{'='*60}")
    print(f"TASK 1c: Matrix multiplication  {a.shape[0]}x{a.shape[1]}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    expected = sequential(a, b)
    seq_time = time.perf_counter() - t0
    print(f"\n[Sequential (numpy BLAS)]  {seq_time:.3f}s  (baseline)")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = map_reduce(a, b, workers)
        t = time.perf_counter() - t0
        ok = "OK" if matrices_close(r, expected) else "MISMATCH"
        print(f"[Map-Reduce  w={workers}]  {t:.3f}s  x{seq_time/t:.2f}  {ok}")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = fork_join(a, b, workers)
        t = time.perf_counter() - t0
        ok = "OK" if matrices_close(r, expected) else "MISMATCH"
        print(f"[Fork-Join   w={workers}]  {t:.3f}s  x{seq_time/t:.2f}  {ok}")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = worker_pool(a, b, workers)
        t = time.perf_counter() - t0
        ok = "OK" if matrices_close(r, expected) else "MISMATCH"
        print(f"[Worker Pool w={workers}]  {t:.3f}s  x{seq_time/t:.2f}  {ok}")


if __name__ == "__main__":
    run_benchmark()