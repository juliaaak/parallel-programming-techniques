"""
Lab 3 - Task 1 (SEQUENTIAL version)
Same bank transfer logic executed in a single thread - no parallelism.
Used as a baseline to compare against parallel (safe and unsafe) versions.
"""

import random
import time

NUM_ACCOUNTS = 150
TOTAL_TRANSFERS = 10_000  # same total ops as parallel versions (1000 threads x 10)


class BankAccount:
    def __init__(self, account_id: int, balance: float):
        self.account_id = account_id
        self.balance = balance


def transfer_sequential(accounts: list, from_id: int, to_id: int, amount: float):
    """
    Single-threaded transfer - no race condition possible,
    no synchronization needed. Correct by design.
    """
    src = accounts[from_id]
    dst = accounts[to_id]
    if src.balance >= amount:
        src.balance -= amount
        dst.balance += amount


def run_sequential():
    print("=" * 60)
    print("TASK 1 - SEQUENTIAL: Single-threaded baseline")
    print("=" * 60)
    print("\n  No parallelism - no race condition, no deadlock possible.\n")

    accounts = [BankAccount(i, random.uniform(100, 1000))
                for i in range(NUM_ACCOUNTS)]

    total_before = sum(a.balance for a in accounts)
    n = len(accounts)

    print(f"  Accounts    : {NUM_ACCOUNTS}")
    print(f"  Transfers   : {TOTAL_TRANSFERS:,} (single thread)")
    print(f"  Total before: {total_before:,.2f}")

    start = time.perf_counter()
    for _ in range(TOTAL_TRANSFERS):
        from_id = random.randint(0, n - 1)
        to_id = random.randint(0, n - 1)
        if from_id == to_id:
            continue
        transfer_sequential(accounts, from_id, to_id, random.uniform(1, 50))
    elapsed = time.perf_counter() - start

    total_after = sum(a.balance for a in accounts)
    discrepancy = abs(total_after - total_before)

    print(f"  Total after : {total_after:,.2f}")
    print(f"  Discrepancy : {discrepancy:,.6f}")
    print(f"  Duration    : {elapsed:.3f}s\n")

    return elapsed


if __name__ == "__main__":
    run_sequential()
    