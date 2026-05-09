"""
Brownian Motion Simulation in a 2D Crystal
Course: Methods and Technologies of Parallel Programming
Project #1 - Full implementation (15 points)

Features:
- 2D grid simulation with multithreading (1 thread per particle)
- Snapshots of crystal state at regular intervals
- Demonstrates race condition (broken version) vs. fixed version (with locks)
- Deadlock demonstration and resolution
- Reproducible simulation via per-thread seeded RNG (hard task #1)
- Rich console output with tables
"""

import threading
import random
import time
import copy
from pathlib import Path
from typing import List, Tuple

# ── try to import rich for pretty tables ──────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── simulation parameters ──────────────────────────────────────────────────────
GRID_ROWS    = 10       # N
GRID_COLS    = 10       # M
NUM_PARTICLES = 20      # K
NUM_STEPS    = 50       # total simulation steps
SNAPSHOT_EVERY = 10     # take a snapshot every N steps
BASE_SEED    = 42       # global seed for reproducibility

# movement probabilities [up, down, left, right] — must sum to 1.0
MOVE_PROBS = [0.25, 0.25, 0.25, 0.25]

# directions as (row_delta, col_delta)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES  = ["UP", "DOWN", "LEFT", "RIGHT"]

console = Console() if HAS_RICH else None

# ── results folder — always next to this script file ──────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str) -> None:
    if HAS_RICH:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)


def print_info(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[green]•[/green] {msg}")
    else:
        print(f"  • {msg}")


def print_warning(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[yellow]⚠[/yellow]  {msg}")
    else:
        print(f"  WARNING: {msg}")


def print_error(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"  ERROR: {msg}")


def render_grid_table(grid: List[List[int]], title: str = "Crystal State") -> None:
    """Print the 2D grid as a table. Cells show particle count."""
    if HAS_RICH:
        table = Table(title=title, box=box.SQUARE, show_header=False,
                      border_style="cyan", padding=(0, 1))
        for _ in range(GRID_COLS):
            table.add_column(justify="center", min_width=3)
        for row in grid:
            styled = []
            for val in row:
                if val == 0:
                    styled.append("[dim]·[/dim]")
                elif val == 1:
                    styled.append("[bold green]1[/bold green]")
                elif val <= 3:
                    styled.append(f"[bold yellow]{val}[/bold yellow]")
                else:
                    styled.append(f"[bold red]{val}[/bold red]")
            table.add_row(*styled)
        console.print(table)
    else:
        print(f"\n  [{title}]")
        for row in grid:
            print("  " + "  ".join(f"{v:2}" for v in row))


def count_particles(grid: List[List[int]]) -> int:
    return sum(sum(row) for row in grid)


def choose_direction(rng: random.Random) -> int:
    """Pick direction index using the particle's own RNG."""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(MOVE_PROBS):
        cumulative += p
        if r < cumulative:
            return i
    return len(MOVE_PROBS) - 1


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — RACE CONDITION DEMO (broken, no synchronisation)
# ══════════════════════════════════════════════════════════════════════════════

class ParticleUnsafe(threading.Thread):
    """
    Thread for one particle — NO locking.
    Demonstrates race condition: read-modify-write on shared grid
    is not atomic, so particles can be lost or counted twice.
    """

    def __init__(self, pid: int, grid: List[List[int]],
                 row: int, col: int, steps: int, seed: int):
        super().__init__(daemon=True)
        self.pid   = pid
        self.grid  = grid
        self.row   = row
        self.col   = col
        self.steps = steps
        # each particle gets its OWN seeded RNG — reproducibility feature
        self.rng   = random.Random(seed + pid)

    def run(self):
        for _ in range(self.steps):
            d = choose_direction(self.rng)
            dr, dc = DIRECTIONS[d]
            new_r = clamp(self.row + dr, 0, GRID_ROWS - 1)
            new_c = clamp(self.col + dc, 0, GRID_COLS - 1)

            # ── RACE CONDITION HERE ──────────────────────────────────────────
            # Between reading grid[old] and writing grid[new], another thread
            # may have already modified those cells. Result: lost particles.
            self.grid[self.row][self.col] -= 1   # leave old cell
            time.sleep(0)                         # yield → makes races more likely
            self.grid[new_r][new_c]       += 1   # enter new cell
            # ─────────────────────────────────────────────────────────────────

            self.row, self.col = new_r, new_c


def demo_race_condition() -> None:
    print_section("DEMO 1: Race Condition (no synchronisation)")
    print_info("Running simulation WITHOUT locks — particles may be lost or duplicated.")

    # initialise grid: place all particles in the centre
    grid = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
    start_r, start_c = GRID_ROWS // 2, GRID_COLS // 2
    grid[start_r][start_c] = NUM_PARTICLES

    before = count_particles(grid)
    print_info(f"Particles BEFORE: {before}")
    render_grid_table(grid, "Initial State")

    threads = []
    for pid in range(NUM_PARTICLES):
        t = ParticleUnsafe(pid, grid, start_r, start_c, NUM_STEPS, BASE_SEED)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    after = count_particles(grid)
    render_grid_table(grid, "Final State (UNSAFE)")

    if HAS_RICH:
        table = Table(box=box.SIMPLE_HEAD, border_style="red")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Particles before", str(before))
        table.add_row("Particles after",  f"[{'green' if after==before else 'red'}]{after}[/]")
        table.add_row("Lost / gained",    str(before - after))
        console.print(table)
    else:
        print(f"\n  Particles before : {before}")
        print(f"  Particles after  : {after}")
        print(f"  Lost / gained    : {before - after}")

    if after != before:
        print_warning("Race condition confirmed — particle count changed!")
    else:
        print_info("Race condition did not manifest this run (try increasing NUM_PARTICLES).")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — FIXED SIMULATION WITH LOCKS + SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════

class Crystal:
    """
    Shared crystal grid with per-cell locks.
    Using per-cell locking instead of one global lock → better parallelism.
    Deadlock is avoided by always acquiring locks in (row, col) order.
    """

    def __init__(self, rows: int, cols: int):
        self.rows  = rows
        self.cols  = cols
        self.grid  = [[0]*cols for _ in range(rows)]
        # one lock per cell — fine-grained locking
        self.locks = [[threading.Lock() for _ in range(cols)] for _ in range(rows)]
        # snapshot list protected by its own lock
        self.snapshots: List[Tuple[int, List[List[int]]]] = []
        self.snapshot_lock = threading.Lock()
        # barrier: all particles wait here between steps → consistent snapshots
        self.step_barrier  = threading.Barrier(NUM_PARTICLES)
        self.current_step  = 0
        self.step_lock     = threading.Lock()

    def move_particle(self, old_r: int, old_c: int,
                      new_r: int, new_c: int) -> None:
        """
        Atomically move one particle from (old_r, old_c) to (new_r, new_c).
        Locks are always acquired in canonical order (min-cell first)
        to prevent deadlock.
        """
        if (old_r, old_c) == (new_r, new_c):
            return  # reflected at boundary — no move needed

        # canonical lock order → no deadlock
        cell_a = (min(old_r, new_r), min(old_c, new_c)) \
            if (old_r, old_c) < (new_r, new_c) else (old_r, old_c)
        cell_b = (new_r, new_c) if cell_a == (old_r, old_c) else (old_r, old_c)

        # always lock lower-index cell first
        first  = (old_r, old_c) if (old_r, old_c) <= (new_r, new_c) else (new_r, new_c)
        second = (new_r, new_c) if first == (old_r, old_c) else (old_r, old_c)

        with self.locks[first[0]][first[1]]:
            with self.locks[second[0]][second[1]]:
                self.grid[old_r][old_c] -= 1
                self.grid[new_r][new_c] += 1

    def take_snapshot(self, step: int) -> None:
        """Deep-copy the grid for a consistent snapshot (no simulation pause needed)."""
        with self.snapshot_lock:
            snap = copy.deepcopy(self.grid)
            self.snapshots.append((step, snap))

    def place_particle(self, row: int, col: int) -> None:
        with self.locks[row][col]:
            self.grid[row][col] += 1


class ParticleSafe(threading.Thread):
    """
    Thread for one particle — uses Crystal's per-cell locks.
    Also uses a barrier so snapshots are taken between steps.
    Each particle has its own seeded RNG for reproducibility.
    """

    def __init__(self, pid: int, crystal: Crystal,
                 row: int, col: int, steps: int, seed: int):
        super().__init__(daemon=True, name=f"Particle-{pid}")
        self.pid     = pid
        self.crystal = crystal
        self.row     = row
        self.col     = col
        self.steps   = steps
        # reproducibility: unique seed per particle
        self.rng     = random.Random(seed + pid)

    def run(self):
        for step in range(1, self.steps + 1):
            d = choose_direction(self.rng)
            dr, dc = DIRECTIONS[d]
            new_r = clamp(self.row + dr, 0, self.crystal.rows - 1)
            new_c = clamp(self.col + dc, 0, self.crystal.cols - 1)

            self.crystal.move_particle(self.row, self.col, new_r, new_c)
            self.row, self.col = new_r, new_c

            # barrier: wait for ALL particles to finish this step
            self.crystal.step_barrier.wait()

            # particle 0 is responsible for taking snapshots
            if self.pid == 0 and step % SNAPSHOT_EVERY == 0:
                self.crystal.take_snapshot(step)

            # second barrier: ensure snapshot is done before next step
            self.crystal.step_barrier.wait()


def demo_safe_simulation() -> None:
    print_section("DEMO 2: Safe Simulation (per-cell locks + barrier snapshots)")
    print_info(f"Grid: {GRID_ROWS}×{GRID_COLS}  |  Particles: {NUM_PARTICLES}  |  Steps: {NUM_STEPS}")
    print_info(f"Snapshot every {SNAPSHOT_EVERY} steps  |  Base seed: {BASE_SEED}")

    crystal = Crystal(GRID_ROWS, GRID_COLS)

    # place all particles in centre
    start_r, start_c = GRID_ROWS // 2, GRID_COLS // 2
    for _ in range(NUM_PARTICLES):
        crystal.place_particle(start_r, start_c)

    before = count_particles(crystal.grid)
    print_info(f"Particles BEFORE: {before}")
    render_grid_table(crystal.grid, "Initial State")

    # create and start threads
    threads = []
    for pid in range(NUM_PARTICLES):
        t = ParticleSafe(pid, crystal, start_r, start_c, NUM_STEPS, BASE_SEED)
        threads.append(t)

    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    after = count_particles(crystal.grid)

    # print all snapshots
    print_section("Snapshots")
    for step, snap in crystal.snapshots:
        render_grid_table(snap, f"Step {step}")
        cnt = count_particles(snap)
        print_info(f"Particle count at step {step}: {cnt}")

    render_grid_table(crystal.grid, "Final State (SAFE)")

    # summary table
    if HAS_RICH:
        table = Table(title="Simulation Summary", box=box.ROUNDED, border_style="green")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")
        table.add_row("Grid size",        f"{GRID_ROWS} × {GRID_COLS}")
        table.add_row("Particles",         str(NUM_PARTICLES))
        table.add_row("Steps",             str(NUM_STEPS))
        table.add_row("Threads used",      str(NUM_PARTICLES))
        table.add_row("Snapshots taken",   str(len(crystal.snapshots)))
        table.add_row("Elapsed time",      f"{elapsed:.3f}s")
        table.add_row("Particles before",  str(before))
        table.add_row("Particles after",
                      f"[{'green' if after==before else 'red'}]{after}[/]")
        table.add_row("Conservation OK?",
                      "[green]YES ✓[/green]" if after == before else "[red]NO ✗[/red]")
        console.print(table)
    else:
        print(f"\n  Grid size       : {GRID_ROWS}x{GRID_COLS}")
        print(f"  Particles       : {NUM_PARTICLES}")
        print(f"  Threads used    : {NUM_PARTICLES}")
        print(f"  Elapsed time    : {elapsed:.3f}s")
        print(f"  Particles before: {before}")
        print(f"  Particles after : {after}")
        print(f"  Conservation OK : {'YES' if after==before else 'NO'}")

    if after == before:
        print_info("Particle count preserved — synchronisation works correctly.")
    else:
        print_error("Particle count changed — check synchronisation!")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — DEADLOCK DEMO + FIX
# ══════════════════════════════════════════════════════════════════════════════

def demo_deadlock() -> None:
    """
    Deadlock scenario: two threads each hold one lock and wait for the other.
    Fixed by always acquiring locks in canonical (sorted) order.
    """
    print_section("DEMO 3: Deadlock — demonstration and fix")

    lock_a = threading.Lock()
    lock_b = threading.Lock()
    results = []

    def thread_bad_1():
        """Acquires A then tries to get B — classic deadlock pattern."""
        with lock_a:
            time.sleep(0.05)           # give thread_bad_2 time to grab B
            acquired = lock_b.acquire(timeout=0.5)
            results.append(("T1-bad", acquired))
            if acquired:
                lock_b.release()

    def thread_bad_2():
        """Acquires B then tries to get A — opposite order → deadlock."""
        with lock_b:
            time.sleep(0.05)
            acquired = lock_a.acquire(timeout=0.5)
            results.append(("T2-bad", acquired))
            if acquired:
                lock_a.release()

    print_info("Running DEADLOCK scenario (threads with opposite lock order)...")
    t1 = threading.Thread(target=thread_bad_1)
    t2 = threading.Thread(target=thread_bad_2)
    t1.start(); t2.start()
    t1.join();  t2.join()

    deadlocked = any(not ok for _, ok in results)
    if deadlocked:
        print_warning("Deadlock detected — one or both threads timed out waiting for a lock!")
    else:
        print_info("No deadlock this run (timing-dependent, try again).")

    # ── FIX: canonical lock order ──────────────────────────────────────────
    def thread_good(lock_first, lock_second, name):
        """Always acquires locks in the same canonical order → no deadlock."""
        with lock_first:
            time.sleep(0.05)
            with lock_second:
                results.append((name, True))

    # sort locks by id() to get consistent order
    ordered = sorted([lock_a, lock_b], key=id)
    results.clear()
    print_info("Running FIXED scenario (canonical lock order)...")
    t3 = threading.Thread(target=thread_good, args=(ordered[0], ordered[1], "T1-good"))
    t4 = threading.Thread(target=thread_good, args=(ordered[0], ordered[1], "T2-good"))
    t3.start(); t4.start()
    t3.join();  t4.join()

    if HAS_RICH:
        table = Table(title="Deadlock Results", box=box.SIMPLE_HEAD, border_style="magenta")
        table.add_column("Thread")
        table.add_column("Lock acquired?", justify="center")
        for name, ok in results:
            table.add_row(name, "[green]YES[/green]" if ok else "[red]TIMEOUT[/red]")
        console.print(table)
    else:
        for name, ok in results:
            print(f"  {name}: {'OK' if ok else 'TIMEOUT'}")

    print_info("Fix: always acquire multiple locks in the same sorted order — deadlock impossible.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — REPRODUCIBILITY (hard task #1)
# ══════════════════════════════════════════════════════════════════════════════

def demo_reproducibility() -> None:
    """
    Show that using per-thread seeded RNGs makes the simulation reproducible.
    Two runs with the same seed produce identical final grids.
    """
    print_section("DEMO 4: Reproducibility via per-thread seeded RNG (Hard Task #1)")

    def run_sim(seed: int) -> List[List[int]]:
        crystal = Crystal(GRID_ROWS, GRID_COLS)
        sr, sc = GRID_ROWS // 2, GRID_COLS // 2
        for _ in range(NUM_PARTICLES):
            crystal.place_particle(sr, sc)
        threads = [
            ParticleSafe(pid, crystal, sr, sc, NUM_STEPS, seed)
            for pid in range(NUM_PARTICLES)
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        return crystal.grid

    grid_run1 = run_sim(BASE_SEED)
    grid_run2 = run_sim(BASE_SEED)
    grid_diff = run_sim(BASE_SEED + 999)   # different seed → different result

    same_12   = grid_run1 == grid_run2
    same_1diff = grid_run1 == grid_diff

    if HAS_RICH:
        table = Table(title="Reproducibility Check", box=box.ROUNDED, border_style="blue")
        table.add_column("Comparison", style="bold")
        table.add_column("Result", justify="center")
        table.add_row("Run 1 vs Run 2 (same seed)",
                      "[green]IDENTICAL ✓[/green]" if same_12 else "[red]DIFFERENT ✗[/red]")
        table.add_row("Run 1 vs Run 3 (different seed)",
                      "[green]DIFFERENT ✓[/green]" if not same_1diff else "[yellow]SAME (unlikely)[/yellow]")
        console.print(table)
    else:
        print(f"  Run1 vs Run2 (same seed)     : {'IDENTICAL' if same_12 else 'DIFFERENT'}")
        print(f"  Run1 vs Run3 (diff seed)     : {'DIFFERENT' if not same_1diff else 'SAME'}")

    print_info("Each particle thread gets rng = random.Random(BASE_SEED + particle_id)")
    print_info("This guarantees deterministic, reproducible results regardless of thread scheduling.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def demo_performance() -> None:
    """
    Compare simulation time for different particle counts.
    Shows the 1-thread-per-particle scalability issue in Python (GIL).
    """
    print_section("DEMO 5: Performance Analysis (1 thread per particle)")

    counts = [5, 10, 20, 40]
    timings = []

    for k in counts:
        crystal = Crystal(GRID_ROWS, GRID_COLS)
        # rebuild barrier for correct k
        crystal.step_barrier = threading.Barrier(k)
        sr, sc = GRID_ROWS // 2, GRID_COLS // 2
        for _ in range(k):
            crystal.place_particle(sr, sc)
        threads = [
            ParticleSafe(pid, crystal, sr, sc, NUM_STEPS, BASE_SEED)
            for pid in range(k)
        ]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.perf_counter() - t0
        timings.append((k, elapsed))
        print_info(f"{k:3d} particles → {elapsed:.3f}s")

    if HAS_RICH:
        table = Table(title="Scalability (1 thread per particle)", box=box.ROUNDED,
                      border_style="yellow")
        table.add_column("Particles", justify="right", style="bold")
        table.add_column("Time (s)",  justify="right")
        table.add_column("Relative",  justify="right")
        base_t = timings[0][1]
        for k, t in timings:
            bar = "█" * int((t / base_t) * 10)
            table.add_row(str(k), f"{t:.3f}", bar)
        console.print(table)

    print_warning("Python's GIL limits true parallelism for CPU-bound threads.")
    print_info("For CPU-bound work, consider multiprocessing or task-pool approaches.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — MATPLOTLIB VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_vis_simulation(vis_steps: int) -> tuple:
    """Helper: run a full simulation and return (crystal, frames list)."""
    import numpy as np
    crystal = Crystal(GRID_ROWS, GRID_COLS)
    crystal.step_barrier = threading.Barrier(NUM_PARTICLES)
    sr, sc = GRID_ROWS // 2, GRID_COLS // 2
    for _ in range(NUM_PARTICLES):
        crystal.place_particle(sr, sc)

    class ParticleVis(threading.Thread):
        def __init__(self, pid, cryst, row, col):
            super().__init__(daemon=True)
            self.pid   = pid
            self.cryst = cryst
            self.row   = row
            self.col   = col
            self.rng   = random.Random(BASE_SEED + pid)

        def run(self):
            for step in range(1, vis_steps + 1):
                d = choose_direction(self.rng)
                dr, dc = DIRECTIONS[d]
                new_r = clamp(self.row + dr, 0, self.cryst.rows - 1)
                new_c = clamp(self.col + dc, 0, self.cryst.cols - 1)
                self.cryst.move_particle(self.row, self.col, new_r, new_c)
                self.row, self.col = new_r, new_c
                self.cryst.step_barrier.wait()
                if self.pid == 0:
                    self.cryst.take_snapshot(step)
                self.cryst.step_barrier.wait()

    threads = [ParticleVis(pid, crystal, sr, sc) for pid in range(NUM_PARTICLES)]
    for t in threads: t.start()
    for t in threads: t.join()

    frames = [(step, np.array(snap)) for step, snap in crystal.snapshots]
    return crystal, frames


def save_results() -> None:
    """
    Save all report-ready plots to results/ folder.
    Called automatically — no display window needed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        print_warning("matplotlib/numpy not installed — skipping save_results.")
        return

    print_section("Saving plots to results/")

    STYLE = {
        "bg":      "#0d1117",
        "panel":   "#161b22",
        "border":  "#30363d",
        "text":    "white",
        "dim":     "#8b949e",
        "green":   "#3fb950",
        "blue":    "#58a6ff",
        "yellow":  "#e3b341",
        "red":     "#f85149",
    }

    def _apply_dark(ax):
        ax.set_facecolor(STYLE["panel"])
        ax.tick_params(colors=STYLE["dim"])
        for sp in ax.spines.values():
            sp.set_color(STYLE["border"])
        ax.xaxis.label.set_color(STYLE["dim"])
        ax.yaxis.label.set_color(STYLE["dim"])
        ax.title.set_color(STYLE["text"])

    def _heatmap(ax, grid, title, vmax=None):
        vm = vmax or max(int(np.max(grid)), 1)
        im = ax.imshow(grid, cmap="plasma", vmin=0, vmax=vm,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title, color=STYLE["text"], fontsize=11)
        _apply_dark(ax)
        return im

    # ── 1. Snapshot gallery ────────────────────────────────────────────────────
    print_info("Generating snapshot gallery...")
    VIS_STEPS = 80
    _, frames = _run_vis_simulation(VIS_STEPS)

    # pick 6 evenly spaced frames for the gallery
    indices  = [int(i * (len(frames) - 1) / 5) for i in range(6)]
    selected = [frames[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Brownian Motion — Snapshot Gallery",
                 color=STYLE["text"], fontsize=14, fontweight="bold")

    vmax = max(int(np.max(g)) for _, g in selected) or 1
    for ax, (step, grid) in zip(axes.flat, selected):
        im = _heatmap(ax, grid, f"Step {step}", vmax=vmax)

    fig.colorbar(im, ax=axes.flat[-1], fraction=0.046, pad=0.04).set_label(
        "Particles / cell", color=STYLE["text"])
    fig.tight_layout(pad=2)
    path = RESULTS_DIR / "01_snapshot_gallery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print_info(f"Saved: {path}")

    # ── 2. Particle count conservation ────────────────────────────────────────
    print_info("Generating particle conservation chart...")
    steps_list  = [s for s, _ in frames]
    counts_list = [int(g.sum()) for _, g in frames]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(STYLE["bg"])
    _apply_dark(ax)
    ax.plot(steps_list, counts_list, color=STYLE["green"], linewidth=2,
            label="Actual count (safe simulation)")
    ax.axhline(NUM_PARTICLES, color=STYLE["blue"], linestyle="--",
               linewidth=1.5, label=f"Expected = {NUM_PARTICLES}")
    ax.set_title("Particle Count Conservation Over Time",
                 color=STYLE["text"], fontsize=13)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Particle count")
    ax.set_ylim(0, NUM_PARTICLES + 5)
    ax.legend(facecolor=STYLE["bg"], labelcolor=STYLE["text"],
              edgecolor=STYLE["border"])
    fig.tight_layout(pad=2)
    path = RESULTS_DIR / "02_particle_conservation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print_info(f"Saved: {path}")

    # ── 3. Diffusion: initial vs mid vs final heatmaps side by side ───────────
    print_info("Generating diffusion comparison...")
    snap_initial = frames[0][1]
    snap_mid     = frames[len(frames) // 2][1]
    snap_final   = frames[-1][1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Particle Diffusion: Initial → Mid → Final",
                 color=STYLE["text"], fontsize=13, fontweight="bold")
    vmax = max(int(np.max(snap_initial)), 1)
    labels = [("Initial (step 0)", snap_initial),
              (f"Mid (step {frames[len(frames)//2][0]})", snap_mid),
              (f"Final (step {frames[-1][0]})", snap_final)]
    for ax, (title, grid) in zip(axes, labels):
        im = _heatmap(ax, grid, title, vmax=vmax)
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04).set_label(
        "Particles / cell", color=STYLE["text"])
    fig.tight_layout(pad=2)
    path = RESULTS_DIR / "03_diffusion_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print_info(f"Saved: {path}")

    # ── 4. Performance / scalability ──────────────────────────────────────────
    print_info("Generating performance chart...")
    counts_perf = [5, 10, 20, 40]
    timings_perf = []
    for k in counts_perf:
        cryst = Crystal(GRID_ROWS, GRID_COLS)
        cryst.step_barrier = threading.Barrier(k)
        sr2, sc2 = GRID_ROWS // 2, GRID_COLS // 2
        for _ in range(k):
            cryst.place_particle(sr2, sc2)
        ts = [ParticleSafe(pid, cryst, sr2, sc2, NUM_STEPS, BASE_SEED)
              for pid in range(k)]
        t0 = time.perf_counter()
        for t in ts: t.start()
        for t in ts: t.join()
        timings_perf.append(time.perf_counter() - t0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Performance: 1 Thread per Particle",
                 color=STYLE["text"], fontsize=13, fontweight="bold")

    # left: time vs particle count
    ax = axes[0]
    _apply_dark(ax)
    ax.plot(counts_perf, timings_perf, color=STYLE["yellow"],
            marker="o", linewidth=2, markersize=7)
    ax.set_title("Simulation Time vs Particle Count", color=STYLE["text"])
    ax.set_xlabel("Number of particles (threads)")
    ax.set_ylabel("Time (seconds)")

    # right: bar chart
    ax2 = axes[1]
    _apply_dark(ax2)
    bars = ax2.bar([str(k) for k in counts_perf], timings_perf,
                   color=STYLE["blue"], edgecolor=STYLE["border"])
    ax2.set_title("Elapsed Time per Configuration", color=STYLE["text"])
    ax2.set_xlabel("Number of particles")
    ax2.set_ylabel("Time (seconds)")
    for bar, t in zip(bars, timings_perf):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{t:.3f}s", ha="center", va="bottom",
                 color=STYLE["text"], fontsize=9)

    fig.tight_layout(pad=2)
    path = RESULTS_DIR / "04_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print_info(f"Saved: {path}")

    # ── 5. Race condition: unsafe final state vs safe final state ─────────────
    print_info("Generating race condition comparison...")

    # unsafe run
    grid_unsafe = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
    sr3, sc3 = GRID_ROWS // 2, GRID_COLS // 2
    grid_unsafe[sr3][sc3] = NUM_PARTICLES
    unsafe_threads = [
        ParticleUnsafe(pid, grid_unsafe, sr3, sc3, NUM_STEPS, BASE_SEED)
        for pid in range(NUM_PARTICLES)
    ]
    for t in unsafe_threads: t.start()
    for t in unsafe_threads: t.join()

    # safe run
    cryst_safe = Crystal(GRID_ROWS, GRID_COLS)
    cryst_safe.step_barrier = threading.Barrier(NUM_PARTICLES)
    for _ in range(NUM_PARTICLES):
        cryst_safe.place_particle(sr3, sc3)
    safe_threads = [
        ParticleSafe(pid, cryst_safe, sr3, sc3, NUM_STEPS, BASE_SEED)
        for pid in range(NUM_PARTICLES)
    ]
    for t in safe_threads: t.start()
    for t in safe_threads: t.join()

    unsafe_arr = np.array(grid_unsafe)
    safe_arr   = np.array(cryst_safe.grid)
    unsafe_cnt = int(unsafe_arr.sum())
    safe_cnt   = int(safe_arr.sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.suptitle("Race Condition: Unsafe vs Safe Final State",
                 color=STYLE["text"], fontsize=13, fontweight="bold")

    vmax2 = max(int(unsafe_arr.max()), int(safe_arr.max()), 1)
    _heatmap(axes[0], unsafe_arr,
             f"UNSAFE  (count={unsafe_cnt})", vmax=vmax2)
    im2 = _heatmap(axes[1], safe_arr,
                   f"SAFE  (count={safe_cnt})", vmax=vmax2)

    # difference map
    diff = safe_arr.astype(int) - unsafe_arr.astype(int)
    ax_d = axes[2]
    _apply_dark(ax_d)
    im_d = ax_d.imshow(diff, cmap="RdYlGn", vmin=-3, vmax=3,
                        interpolation="nearest", aspect="equal")
    ax_d.set_title("Difference (safe − unsafe)", color=STYLE["text"], fontsize=11)
    fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04).set_label(
        "Δ particles", color=STYLE["text"])

    fig.tight_layout(pad=2)
    path = RESULTS_DIR / "05_race_condition_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print_info(f"Saved: {path}")

    # ── summary ───────────────────────────────────────────────────────────────
    saved = sorted(RESULTS_DIR.glob("*.png"))
    if HAS_RICH:
        table = Table(title="Saved plots", box=box.SIMPLE_HEAD, border_style="green")
        table.add_column("File", style="cyan")
        table.add_column("Description")
        descs = [
            "Snapshot gallery (6 evenly-spaced frames)",
            "Particle count conservation over time",
            "Diffusion: initial / mid / final heatmaps",
            "Performance: time vs thread count",
            "Race condition: unsafe vs safe comparison",
        ]
        for f, d in zip(saved, descs):
            table.add_row(f.name, d)
        console.print(table)
    else:
        print(f"\n  Plots saved to: {RESULTS_DIR.resolve()}")
        for f in saved:
            print(f"    {f.name}")


def visualize_simulation() -> None:
    """
    Run animated heatmap in an interactive window.
    Left: heatmap per step. Right: particle count over time.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
    except ImportError:
        print_warning("matplotlib/numpy not installed — skipping visualisation.")
        print_info("Run:  pip install matplotlib numpy")
        return

    print_section("DEMO 6: Visualisation (matplotlib animated heatmap)")
    print_info("Running simulation to collect frames...")

    VIS_STEPS = 80
    _, frames = _run_vis_simulation(VIS_STEPS)
    print_info(f"Collected {len(frames)} frames — launching animation window...")

    STYLE_BG    = "#0d1117"
    STYLE_PANEL = "#161b22"
    STYLE_BORDER= "#30363d"
    STYLE_DIM   = "#8b949e"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(STYLE_BG)

    ax_heat = axes[0]
    ax_heat.set_facecolor(STYLE_BG)
    ax_heat.set_xlabel("Column", color=STYLE_DIM)
    ax_heat.set_ylabel("Row",    color=STYLE_DIM)
    ax_heat.tick_params(colors=STYLE_DIM)
    for sp in ax_heat.spines.values(): sp.set_color(STYLE_BORDER)

    im = ax_heat.imshow(
        frames[0][1], cmap="plasma", vmin=0, vmax=max(NUM_PARTICLES // 3, 1),
        interpolation="nearest", aspect="equal"
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Particles per cell", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    title_heat = ax_heat.set_title("Step 0", color="white", fontsize=12)

    ax_line = axes[1]
    ax_line.set_facecolor(STYLE_PANEL)
    ax_line.set_title("Particle Count Over Time", color="white", fontsize=12)
    ax_line.set_xlabel("Step",  color=STYLE_DIM)
    ax_line.set_ylabel("Count", color=STYLE_DIM)
    ax_line.tick_params(colors=STYLE_DIM)
    for sp in ax_line.spines.values(): sp.set_color(STYLE_BORDER)
    ax_line.set_xlim(0, VIS_STEPS)
    ax_line.set_ylim(0, NUM_PARTICLES + 5)
    ax_line.axhline(NUM_PARTICLES, color="#58a6ff", linestyle="--",
                    linewidth=1.5, label=f"Expected = {NUM_PARTICLES}")
    ax_line.legend(facecolor=STYLE_BG, labelcolor="white", edgecolor=STYLE_BORDER)

    steps_h, counts_h = [], []
    line_plot, = ax_line.plot([], [], color="#3fb950", linewidth=2)

    fig.suptitle("Brownian Motion — 2D Crystal Simulation",
                 color="white", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=2)

    def update(idx):
        step, grid = frames[idx]
        im.set_data(grid)
        title_heat.set_text(f"Step {step}  |  Particles: {int(grid.sum())}")
        steps_h.append(step)
        counts_h.append(int(grid.sum()))
        line_plot.set_data(steps_h, counts_h)
        return im, line_plot, title_heat

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=120, blit=False, repeat=True
    )
    plt.show()
    print_info("Visualisation window closed.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold white]Brownian Motion Simulation[/bold white]\n"
            "[dim]Parallel Programming Project #1[/dim]",
            border_style="bright_cyan"
        ))
    else:
        print("\n" + "="*60)
        print("  Brownian Motion Simulation")
        print("  Parallel Programming Project #1")
        print("="*60)

    if not HAS_RICH:
        print("\n  TIP: run  pip install rich  for a nicer output!\n")

    demo_race_condition()
    demo_deadlock()
    demo_safe_simulation()
    demo_reproducibility()
    demo_performance()
    save_results()           # save all plots to results/ folder
    visualize_simulation()   # interactive animation window

    print_section("Done")
    print_info("All demonstrations complete.")


if __name__ == "__main__":
    main()
