"""
Lab 3 - Main runner
Executes all tasks, prints summary, saves speedup charts.
"""

import multiprocessing
from task1_sequential import run_sequential
from task1_unsafe import run_unsafe, THREAD_COUNTS
from task1_safe import run_safe
from task2_ipc import run_task2
from charts import generate_all


def main():
    print("\n" + "#" * 60)
    print("#   Lab 3: Deadlock, Race Condition, IPC Benchmark      #")
    print("#   Course: Parallel Programming Methods & Technologies  #")
    print("#" * 60)

    print()
    t_seq = run_sequential()

    print()
    t_unsafe, unsafe_hyper = run_unsafe()

    print()
    t_safe, safe_hyper = run_safe()

    print()
    ipc_results = run_task2()

    generate_all(unsafe_hyper, safe_hyper, ipc_results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    speedup_unsafe = t_seq / t_unsafe if t_unsafe > 0 else 0
    speedup_safe   = t_seq / t_safe   if t_safe   > 0 else 0

    print(f"\n  Task 1 - Bank transfers (150 accounts, {1001*10:,} ops):")
    print(f"    Sequential      : {t_seq:.3f}s  (baseline)")
    print(f"    Parallel unsafe : {t_unsafe:.3f}s  (speedup x{speedup_unsafe:.2f}, INCORRECT)")
    print(f"    Parallel safe   : {t_safe:.3f}s  (speedup x{speedup_safe:.2f}, correct)")

    print(f"\n  Hyperparameter research - unsafe vs safe by thread count:")
    print(f"  {'Threads':>8}  {'Unsafe (s)':>12}  {'Safe (s)':>10}  {'Ops':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*8}")
    for n in sorted(unsafe_hyper.keys()):
        t_u, _disc = unsafe_hyper[n]
        t_s = safe_hyper.get(n, 0)
        ops = n * 10
        print(f"  {n:>8}  {t_u:>12.3f}  {t_s:>10.3f}  {ops:>8,}")

    print(f"\n  Task 2 - IPC round-trip latency (200 iterations each):")
    for name, (_, avg_us) in ipc_results.items():
        print(f"    {name.strip()}: {avg_us:.1f} us/round-trip")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
    