"""
task2_transactions.py
=====================
Task 2: Financial transaction processing system.
Input: data/transactions.csv  (user_id, amount, currency, date, product_type)

Processing steps (same logic in both patterns):
  1. Parse CSV row -> Transaction object
  2. Convert amount to USD using fixed exchange rates
  3. Apply cashback: users with ID > 50,000 receive 20% back
  4. Aggregate: compute final total per user_id

Two parallel patterns implemented:
  Pipeline        — 4 stages connected by queues, each stage runs in its own thread
  Producer-Consumer — one producer reads CSV, N consumer threads process in parallel

Framework: threading + queue.Queue (Python standard library)

Why threading here (not multiprocessing)?
  - The bottleneck is CSV I/O and simple arithmetic, not CPU computation
  - queue.Queue is thread-safe out of the box, no IPC overhead
  - GIL releases during I/O, so threads do overlap work meaningfully
  - multiprocessing would add pickle/IPC cost with no real gain for this workload

Run:
    python task2_transactions.py
"""

import csv
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# ─── Data structures ─────────────────────────────────────────────────────────

EXCHANGE_RATES: Dict[str, float] = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27,
    "UAH": 0.024, "PLN": 0.25, "CHF": 1.12, "JPY": 0.0067,
}

_DONE = object()   # sentinel value: signals worker threads to stop


@dataclass
class Transaction:
    user_id:      int
    amount:       float
    currency:     str
    date:         str
    product_type: str


@dataclass
class ProcessedTx:
    user_id:      int
    amount_usd:   float   # after currency conversion
    cashback_usd: float   # cashback amount


# ─── Shared processing logic ─────────────────────────────────────────────────

def parse_row(row) -> Optional[Transaction]:
    try:
        return Transaction(
            user_id      = int(row["user_id"]),
            amount       = float(row["amount"]),
            currency     = row["currency"].strip().upper(),
            date         = row["date"].strip(),
            product_type = row["product_type"].strip(),
        )
    except (ValueError, KeyError):
        return None


def convert_to_usd(tx: Transaction) -> float:
    return tx.amount * EXCHANGE_RATES.get(tx.currency, 1.0)


def calc_cashback(user_id: int, amount_usd: float) -> float:
    """Users with ID > 50,000 receive 20% cashback."""
    return amount_usd * 0.20 if user_id > 50_000 else 0.0


# ─── 1. Sequential baseline ──────────────────────────────────────────────────

def sequential(filepath: str) -> Dict[int, float]:
    """Read, convert, cashback, aggregate — all in a single loop."""
    totals: Dict[int, float] = defaultdict(float)

    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tx = parse_row(row)
            if tx is None:
                continue
            amount_usd = convert_to_usd(tx)
            cashback   = calc_cashback(tx.user_id, amount_usd)
            totals[tx.user_id] += amount_usd - cashback

    return dict(totals)


# ─── 2. Pipeline ─────────────────────────────────────────────────────────────
# Four stages connected by three queues, each running in its own thread.
# Data flows in one direction: Stage1 -> q1 -> Stage2 -> q2 -> Stage3 -> q3 -> Stage4
#
#   [Stage1: read CSV]  ->  [Stage2: convert]  ->  [Stage3: cashback]  ->  [Stage4: aggregate]
#
# Advantages:
#   + Low latency: Stage2 processes batch N while Stage1 reads batch N+1 (overlap)
#   + Easy to add/remove stages without changing other stages
#   + Memory bounded by queue maxsize
#
# Disadvantages:
#   - Throughput is limited by the slowest stage (bottleneck)
#   - Harder to debug than a simple loop

def _stage1_read(filepath: str, q_out: queue.Queue, batch: int = 500):
    """Read CSV and push batches of Transaction objects into the queue."""
    with open(filepath, newline="", encoding="utf-8") as f:
        buf = []
        for row in csv.DictReader(f):
            tx = parse_row(row)
            if tx:
                buf.append(tx)
            if len(buf) >= batch:
                q_out.put(buf)
                buf = []
        if buf:
            q_out.put(buf)
    q_out.put(_DONE)


def _stage2_convert(q_in: queue.Queue, q_out: queue.Queue):
    """Convert each transaction's amount to USD."""
    while True:
        item = q_in.get()
        if item is _DONE:
            q_out.put(_DONE)
            return
        q_out.put([(tx, convert_to_usd(tx)) for tx in item])


def _stage3_cashback(q_in: queue.Queue, q_out: queue.Queue):
    """Compute cashback for each transaction."""
    while True:
        item = q_in.get()
        if item is _DONE:
            q_out.put(_DONE)
            return
        processed = []
        for tx, amount_usd in item:
            cashback = calc_cashback(tx.user_id, amount_usd)
            processed.append(ProcessedTx(tx.user_id, amount_usd, cashback))
        q_out.put(processed)


def _stage4_aggregate(q_in: queue.Queue, result: dict):
    """Aggregate final amounts per user_id."""
    while True:
        item = q_in.get()
        if item is _DONE:
            return
        for ptx in item:
            result[ptx.user_id] = (result.get(ptx.user_id, 0.0)
                                   + ptx.amount_usd - ptx.cashback_usd)


def pipeline(filepath: str, q_size: int = 200) -> Dict[int, float]:
    q1, q2, q3 = (queue.Queue(maxsize=q_size) for _ in range(3))
    result: Dict[int, float] = {}

    threads = [
        threading.Thread(target=_stage1_read,     args=(filepath, q1),  daemon=True),
        threading.Thread(target=_stage2_convert,  args=(q1, q2),        daemon=True),
        threading.Thread(target=_stage3_cashback, args=(q2, q3),        daemon=True),
        threading.Thread(target=_stage4_aggregate,args=(q3, result),    daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    return result


# ─── 3. Producer-Consumer ────────────────────────────────────────────────────
# One producer reads CSV and pushes batches into a shared queue.
# N consumer threads pull batches, process them, and update a shared aggregate.
#
# Key optimisation: each consumer maintains a LOCAL dict and merges into
# the global dict only once at the end (under Lock). This minimises lock contention.
#
# Advantages:
#   + Easy to scale: just add more consumers
#   + Natural backpressure: queue.maxsize blocks producer if consumers are slow
#   + Flexible: consumers can do complex, independent processing
#
# Disadvantages:
#   - Lock on the global aggregate is a bottleneck at very high consumer counts
#   - GIL limits CPU-bound work in threads (use ProcessPoolExecutor for heavy CPU)

def _producer(filepath: str, shared_q: queue.Queue,
              n_consumers: int, batch: int = 500):
    with open(filepath, newline="", encoding="utf-8") as f:
        buf = []
        for row in csv.DictReader(f):
            tx = parse_row(row)
            if tx:
                buf.append(tx)
            if len(buf) >= batch:
                shared_q.put(buf)
                buf = []
        if buf:
            shared_q.put(buf)
    # Send one sentinel per consumer so each knows when to stop
    for _ in range(n_consumers):
        shared_q.put(_DONE)


def _consumer(shared_q: queue.Queue, totals: dict, lock: threading.Lock):
    """Pull batches, process locally, merge into global totals once at the end."""
    local: Dict[int, float] = defaultdict(float)

    while True:
        item = shared_q.get()
        if item is _DONE:
            break
        for tx in item:
            amount_usd = convert_to_usd(tx)
            cashback   = calc_cashback(tx.user_id, amount_usd)
            local[tx.user_id] += amount_usd - cashback

    # Single lock acquisition per consumer thread (not per transaction)
    with lock:
        for uid, val in local.items():
            totals[uid] = totals.get(uid, 0.0) + val


def producer_consumer(filepath: str, n_consumers: int = 4,
                      q_size: int = 200) -> Dict[int, float]:
    shared_q = queue.Queue(maxsize=q_size)
    totals:   Dict[int, float] = {}
    lock = threading.Lock()

    prod = threading.Thread(
        target=_producer,
        args=(filepath, shared_q, n_consumers),
        daemon=True,
    )
    consumers = [
        threading.Thread(target=_consumer, args=(shared_q, totals, lock), daemon=True)
        for _ in range(n_consumers)
    ]

    prod.start()
    for c in consumers: c.start()
    prod.join()
    for c in consumers: c.join()

    return totals


# ─── Result comparison ───────────────────────────────────────────────────────

def results_close(a: Dict[int, float], b: Dict[int, float], tol: float = 0.01) -> bool:
    return set(a.keys()) == set(b.keys()) and all(abs(a[k] - b[k]) < tol for k in a)


# ─── Benchmark ───────────────────────────────────────────────────────────────

def run_benchmark():
    filepath = "data/transactions.csv"
    if not Path(filepath).exists():
        print("Run first: python generate_data.py transactions")
        return

    n_lines = sum(1 for _ in open(filepath)) - 1
    print(f"\n{'='*60}")
    print(f"TASK 2: Financial transaction processing  ({n_lines:,} rows)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    seq_result = sequential(filepath)
    seq_time   = time.perf_counter() - t0
    print(f"\n[Sequential          ]  {seq_time:.3f}s  total=${sum(seq_result.values()):,.2f}")

    t0 = time.perf_counter()
    pl_result  = pipeline(filepath)
    pl_time    = time.perf_counter() - t0
    ok = "OK" if results_close(seq_result, pl_result) else "MISMATCH"
    print(f"[Pipeline            ]  {pl_time:.3f}s  x{seq_time/pl_time:.2f}  {ok}")

    for n_c in [2, 4, 8]:
        t0 = time.perf_counter()
        pc_result = producer_consumer(filepath, n_consumers=n_c)
        pc_time   = time.perf_counter() - t0
        ok = "OK" if results_close(seq_result, pc_result) else "MISMATCH"
        print(f"[Prod-Consumer  c={n_c} ]  {pc_time:.3f}s  x{seq_time/pc_time:.2f}  {ok}")

    print(f"\nUnique users: {len(seq_result):,}")
    print("Top-5 users by total spend:")
    for uid, total in sorted(seq_result.items(), key=lambda x: -x[1])[:5]:
        print(f"  user_id={uid}: ${total:,.2f}")


if __name__ == "__main__":
    run_benchmark()