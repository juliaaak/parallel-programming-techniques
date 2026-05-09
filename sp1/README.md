# Project 1 — Brownian Motion Simulation (Multithreaded)

## Run

```bash
C:\Users\julia\AppData\Local\Programs\Python\Python314\python.exe -m pip install rich matplotlib numpy
C:\Users\julia\AppData\Local\Programs\Python\Python314\python.exe main.py
```

Runs all demos and saves charts to `results/`.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — runs all demos and generates charts |
| `results/` | Auto-created folder with saved PNG plots |

## Dependencies

| Package | Purpose |
|---------|---------|
| `rich` | Pretty console output with tables and colours |
| `matplotlib` | Animated visualisation and saved charts |
| `numpy` | Grid array operations for plotting |

Install all at once:
```bash
C:\Users\julia\AppData\Local\Programs\Python\Python314\python.exe -m pip install rich matplotlib numpy
```

## What it does

| Demo | Description |
|------|-------------|
| Demo 1 | Race condition — simulation without locks, shows particle loss |
| Demo 2 | Safe simulation — per-cell locks + barrier snapshots, particle count preserved |
| Demo 3 | Deadlock — two threads with opposite lock order, fixed via canonical ordering |
| Demo 4 | Reproducibility — same seed produces identical results across runs |
| Demo 5 | Performance analysis — simulation time vs number of threads |
| Demo 6 | Animated heatmap — interactive matplotlib window |

## Output

| File | Description |
|------|-------------|
| `results/01_snapshot_gallery.png` | 6 evenly-spaced snapshots of particle distribution |
| `results/02_particle_conservation.png` | Particle count over time (flat line = correct) |
| `results/03_diffusion_comparison.png` | Diffusion: initial → mid → final state |
| `results/04_performance.png` | Simulation time vs particle/thread count |
| `results/05_race_condition_comparison.png` | Unsafe vs safe final state comparison |