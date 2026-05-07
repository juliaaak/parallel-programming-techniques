"""
Lab 3 - Task 1 (UNSAFE version)
Demonstrates Race Condition and Deadlock in bank transfer simulation.
"""

import threading
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait

NUM_ACCOUNTS = 150
NUM_THREADS = 1001
TRANSFERS_PER_WORKER = 10

THREAD_COUNTS = [1, 10, 100, 500, 1001]


class BankAccountUnsafe:
    def __init__(self, account_id: int, balance: float):
        self.account_id = account_id
        self.balance = balance


def transfer_unsafe(accounts: list, from_id: int, to_id: int, amount: float):
    src = accounts[from_id]
    dst = accounts[to_id]
    temp = src.balance
    time.sleep(0)
    if temp >= amount:
        src.balance = temp - amount
        dst.balance += amount


def worker_unsafe(accounts: list, n_transfers: int):
    n = len(accounts)
    for _ in range(n_transfers):
        from_id = random.randint(0, n - 1)
        to_id = random.randint(0, n - 1)
        if from_id == to_id:
            continue
        transfer_unsafe(accounts, from_id, to_id, random.uniform(1, 20))


def demonstrate_deadlock():
    lock_a = threading.Lock()
    lock_b = threading.Lock()
    detected = threading.Event()

    def thread1():
        with lock_a:
            time.sleep(0.05)
            acquired = lock_b.acquire(timeout=0.3)
            if not acquired:
                detected.set()
                print("  [Thread-1] holds lock_A, timed out waiting for lock_B -> DEADLOCK")
            else:
                lock_b.release()

    def thread2():
        with lock_b:
            time.sleep(0.05)
            acquired = lock_a.acquire(timeout=0.3)
            if not acquired:
                detected.set()
                print("  [Thread-2] holds lock_B, timed out waiting for lock_A -> DEADLOCK")
            else:
                lock_a.release()

    t1 = threading.Thread(target=thread1)
    t2 = threading.Thread(target=thread2)
    t1.start(); t2.start()
    t1.join();  t2.join()

    if detected.is_set():
        print("  Deadlock detected: both threads timed out waiting for each other.\n")


def run_with_threads(n_threads: int) -> tuple:
    accounts = [BankAccountUnsafe(i, random.uniform(100, 500))
                for i in range(NUM_ACCOUNTS)]
    total_before = sum(a.balance for a in accounts)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker_unsafe, accounts, TRANSFERS_PER_WORKER)
                   for _ in range(n_threads)]
        wait(futures)
    elapsed = time.perf_counter() - start

    total_after = sum(a.balance for a in accounts)
    discrepancy = total_after - total_before
    return elapsed, total_before, discrepancy


def run_unsafe():
    print("=" * 60)
    print("TASK 1 - UNSAFE: Race Condition & Deadlock")
    print("=" * 60)

    print("\n[1] Deadlock demo:")
    demonstrate_deadlock()

    print("[2] Race Condition demo - bank transfers:\n")
    elapsed, total_before, discrepancy = run_with_threads(NUM_THREADS)
    total_after = total_before + discrepancy

    print(f"  Accounts       : {NUM_ACCOUNTS}")
    print(f"  Worker threads : {NUM_THREADS} x {TRANSFERS_PER_WORKER} transfers"
          f" = {NUM_THREADS * TRANSFERS_PER_WORKER:,} ops")
    print(f"  Total before   : {total_before:,.2f}")
    print(f"  Total after    : {total_after:,.2f}")
    print(f"  Discrepancy    : {discrepancy:+,.2f}")
    print(f"  Duration       : {elapsed:.3f}s\n")

    print("[3] Hyperparameter research - effect of thread count:\n")
    print(f"  {'Threads':>8}  {'Time (s)':>10}  {'Discrepancy':>14}  {'Ops':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*8}")

    results = {}
    for n in THREAD_COUNTS:
        t, _, disc = run_with_threads(n)
        ops = n * TRANSFERS_PER_WORKER
        print(f"  {n:>8}  {t:>10.3f}  {disc:>+14.2f}  {ops:>8,}")
        results[n] = (t, disc)

    print()
    return elapsed, results


if __name__ == "__main__":
    run_unsafe()
    