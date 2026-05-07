# Lab 3 — Deadlock, Race Condition, IPC Benchmark

## Run

```bash
pip install matplotlib
python main.py
```

Runs all tasks and saves charts to `results/`.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point, runs all tasks and generates charts |
| `task1_sequential.py` | Single-threaded bank transfer baseline |
| `task1_unsafe.py` | Unsafe transfers — demonstrates race condition and deadlock |
| `task1_safe.py` | Safe transfers — fixed with locks and ordered acquisition |
| `task2_ipc.py` | IPC benchmark: Queue, Shared Memory, Socket TCP (Python↔Python, Python↔Node.js) |
| `helper_node.js` | Node.js TCP server — receives float, logs it, echoes back |
| `charts.py` | Generates 3 PNG charts saved to `results/` |

## Output

- `results/lab3_chart1_time.png` — execution time vs thread count
- `results/lab3_chart2_discrepancy.png` — race condition discrepancy vs thread count  
- `results/lab3_chart3_ipc.png` — IPC round-trip latency comparison