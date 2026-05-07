"""
Lab 3 - Task 1 (SAFE version)
Fixes Race Condition with per-account locks.
Fixes Deadlock with ordered lock acquisition.
"""

import threading
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait

NUM_ACCOUNTS = 150
NUM_THREADS = 1001
TRANSFERS_PER_WORKER = 10

THREAD_COUNTS = [1, 10, 100, 500, 1001]


class BankAccountSafe:
    def __init__(self, account_id: int, balance: float):
        self.account_id = account_id
        self.balance = balance
        self.lock = threading.Lock()


def transfer_safe(accounts: list, from_id: int, to_id: int, amount: float):
    src = accounts[from_id]
    dst = accounts[to_id]
    first, second = (src, dst) if src.account_id < dst.account_id else (dst, src)
    with first.lock:
        with second.lock:
            if src.balance >= amount:
                src.balance -= amount
                dst.balance += amount


def worker_safe(accounts: list, n_transfers: int):
    n = len(accounts)
    for _ in range(n_transfers):
        from_id = random.randint(0, n - 1)
        to_id = random.randint(0, n - 1)
        if from_id == to_id:
            continue
        transfer_safe(accounts, from_id, to_id, random.uniform(1, 50))


def run_with_threads(n_threads: int) -> tuple:
    accounts = [BankAccountSafe(i, random.uniform(100, 1000))
                for i in range(NUM_ACCOUNTS)]
    total_before = sum(a.balance for a in accounts)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker_safe, accounts, TRANSFERS_PER_WORKER)
                   for _ in range(n_threads)]
        wait(futures)
    elapsed = time.perf_counter() - start

    total_after = sum(a.balance for a in accounts)
    discrepancy = abs(total_after - total_before)
    return elapsed, discrepancy


def run_safe():
    print("=" * 60)
    print("TASK 1 - SAFE: Fixed Race Condition & Deadlock")
    print("=" * 60)
    print("\n  Race condition fix : per-account threading.Lock")
    print("  Deadlock fix       : acquire locks in ascending account_id order\n")

    accounts = [BankAccountSafe(i, random.uniform(100, 1000))
                for i in range(NUM_ACCOUNTS)]
    total_before = sum(a.balance for a in accounts)
    total_ops = NUM_THREADS * TRANSFERS_PER_WORKER
    print(f"  Accounts       : {NUM_ACCOUNTS}")
    print(f"  Worker threads : {NUM_THREADS} x {TRANSFERS_PER_WORKER} transfers = {total_ops:,} ops")
    print(f"  Total before   : {total_before:,.2f}")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        futures = [pool.submit(worker_safe, accounts, TRANSFERS_PER_WORKER)
                   for _ in range(NUM_THREADS)]
        wait(futures)
    elapsed = time.perf_counter() - start

    total_after = sum(a.balance for a in accounts)
    discrepancy = abs(total_after - total_before)
    print(f"  Total after    : {total_after:,.2f}")
    print(f"  Discrepancy    : {discrepancy:,.6f}")
    print(f"  Duration       : {elapsed:.3f}s\n")

    print("  Hyperparameter research - effect of thread count:\n")
    print(f"  {'Threads':>8}  {'Time (s)':>10}  {'Discrepancy':>14}  {'Ops':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*8}")

    hyper_results = {}
    for n in THREAD_COUNTS:
        t, disc = run_with_threads(n)
        ops = n * TRANSFERS_PER_WORKER
        print(f"  {n:>8}  {t:>10.3f}  {disc:>14.6f}  {ops:>8,}")
        hyper_results[n] = t

    print()
    return elapsed, hyper_results


if __name__ == "__main__":
    run_safe()
    