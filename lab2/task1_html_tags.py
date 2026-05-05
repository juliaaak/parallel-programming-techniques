"""
task1_html_tags.py
==================
Task 1a: Count HTML tag frequencies across 1200 HTML documents.
Implements three parallel patterns and one sequential baseline:
  - Sequential   : single loop over all files
  - Map-Reduce   : each process maps one file -> Counter, then reduce via +
  - Fork-Join    : split file list into chunks, fork tasks, join results
  - Worker Pool  : fixed thread pool pulls files from a queue dynamically

Framework: concurrent.futures (Python standard library, no install needed)

Why concurrent.futures?
  - Unified API for threads (ThreadPoolExecutor) and processes (ProcessPoolExecutor)
  - submit() / map() naturally express all three patterns
  - No third-party dependencies

Python GIL note:
  - Threads do NOT give true CPU parallelism for CPU-bound work
  - For CPU-bound tasks use ProcessPoolExecutor (separate OS processes)
  - HTML parsing is I/O-bound (reading files) -> threads work well here too

Run:
    python task1_html_tags.py
"""

import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import reduce
from pathlib import Path
from typing import List


# ─── Helpers ─────────────────────────────────────────────────────────────────

TAG_PATTERN = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)[^>]*?>")


def count_tags_in_file(filepath: str) -> Counter:
    """Read one HTML file and return a Counter of lowercase tag names."""
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        tags = TAG_PATTERN.findall(text)
        return Counter(tag.lower() for tag in tags)
    except Exception:
        return Counter()


def get_html_files(directory: str = "data/html_docs") -> List[str]:
    return sorted(str(p) for p in Path(directory).glob("*.html"))


# ─── 1. Sequential baseline ──────────────────────────────────────────────────

def sequential(files: List[str]) -> Counter:
    """Process files one by one, no parallelism."""
    total = Counter()
    for f in files:
        total.update(count_tags_in_file(f))
    return total


# ─── 2. Map-Reduce ───────────────────────────────────────────────────────────
# MAP  : each process independently maps one file -> Counter
# REDUCE: merge all Counters into one with +
#
# Best for: large independent datasets, natural aggregation
# Overhead: process startup cost per task (mitigated by ProcessPool reuse)

def map_reduce(files: List[str], workers: int = 4) -> Counter:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # MAP: each file processed in parallel by a worker process
        mapped = list(executor.map(count_tags_in_file, files))

    # REDUCE: merge all partial Counters (O(k) per merge, k = unique tags)
    return reduce(lambda a, b: a + b, mapped, Counter())


# ─── 3. Fork-Join ────────────────────────────────────────────────────────────
# FORK : split file list into equal chunks, submit each chunk as one task
# JOIN : wait for all tasks to finish, then merge results
#
# Difference from Map-Reduce: task granularity is a chunk (not a single file).
# Reduces task-spawn overhead when there are many small files.

def _process_chunk(chunk: List[str]) -> Counter:
    """Process a list of files and return an aggregated Counter."""
    result = Counter()
    for f in chunk:
        result.update(count_tags_in_file(f))
    return result


def fork_join(files: List[str], workers: int = 4) -> Counter:
    chunk_size = max(1, len(files) // workers)
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # FORK: submit each chunk as an independent task
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]

        # JOIN: collect results as they complete
        total = Counter()
        for future in as_completed(futures):
            total.update(future.result())

    return total


# ─── 4. Worker Pool ──────────────────────────────────────────────────────────
# A fixed pool of N workers pulls tasks from a queue dynamically.
# executor.map() implements this pattern natively in Python.
#
# Best when tasks have variable duration: the pool auto-balances load.
# Uses ThreadPoolExecutor here because file reading is I/O-bound.

def worker_pool(files: List[str], workers: int = 4) -> Counter:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(count_tags_in_file, files))

    total = Counter()
    for r in results:
        total.update(r)
    return total


# ─── Benchmark ───────────────────────────────────────────────────────────────

def run_benchmark(files: List[str]):
    print(f"\n{'='*60}")
    print(f"TASK 1a: HTML tag frequency across {len(files)} files")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    seq_result = sequential(files)
    seq_time = time.perf_counter() - t0
    print(f"\n[Sequential ]  {seq_time:.3f}s  (baseline)")

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = map_reduce(files, workers)
        t = time.perf_counter() - t0
        print(f"[Map-Reduce  w={workers}]  {t:.3f}s  speedup={seq_time/t:.2f}x")
        assert r == seq_result, "Map-Reduce: result mismatch!"

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = fork_join(files, workers)
        t = time.perf_counter() - t0
        print(f"[Fork-Join   w={workers}]  {t:.3f}s  speedup={seq_time/t:.2f}x")
        assert r == seq_result, "Fork-Join: result mismatch!"

    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        r = worker_pool(files, workers)
        t = time.perf_counter() - t0
        print(f"[Worker Pool w={workers}]  {t:.3f}s  speedup={seq_time/t:.2f}x")
        assert r == seq_result, "Worker Pool: result mismatch!"

    print(f"\nTop-10 most frequent tags:")
    for tag, count in seq_result.most_common(10):
        print(f"  <{tag}>: {count:,}")


if __name__ == "__main__":
    files = get_html_files()
    if not files:
        print("Run first: python generate_data.py html")
    else:
        run_benchmark(files)