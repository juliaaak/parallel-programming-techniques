"""
generate_data.py
================
Generates all test datasets required for Lab 2:
  - 1200 HTML documents of varying sizes  -> data/html_docs/
  - Array of 2,000,000 floats             -> data/array.npy
  - Two 1200x1200 matrices                -> data/matrix_a.npy, data/matrix_b.npy
  - 500,000 financial transactions CSV    -> data/transactions.csv

Usage:
    python generate_data.py              # generate everything
    python generate_data.py html         # only HTML files
    python generate_data.py array        # only the numeric array
    python generate_data.py matrices     # only matrices
    python generate_data.py transactions # only CSV
"""

import random
import string
import csv
import numpy as np
from pathlib import Path

# ─── HTML documents ──────────────────────────────────────────────────────────

HTML_TAGS = [
    "div", "span", "p", "a", "h1", "h2", "h3", "h4", "ul", "li",
    "table", "tr", "td", "th", "form", "input", "button", "img",
    "section", "article", "header", "footer", "nav", "main", "aside",
    "strong", "em", "br", "hr", "label", "select", "option",
]

def random_text(words: int = 10) -> str:
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(words)
    )

def generate_html_doc(size: str = "medium") -> str:
    """Generate a single HTML document with random tags."""
    depth_limits = {"small": 15, "medium": 60, "large": 200}
    n = depth_limits.get(size, 60)

    lines = ["<!DOCTYPE html>", "<html>", "<head>",
             f"<title>{random_text(3)}</title>", "</head>", "<body>"]

    for _ in range(n):
        tag = random.choice(HTML_TAGS)
        content = random_text(random.randint(2, 8))
        attrs = ""
        if tag == "a":
            attrs = ' href="#"'
        elif tag == "img":
            attrs = ' src="img.png" alt="img"'
            lines.append(f"<{tag}{attrs}/>")
            continue
        lines.append(f"<{tag}{attrs}>{content}</{tag}>")

    lines += ["</body>", "</html>"]
    return "\n".join(lines)


def generate_html_files(out_dir: str = "data/html_docs", n: int = 1200):
    """Save n HTML files of mixed sizes (small / medium / large)."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    sizes = ["small"] * (n // 3) + ["medium"] * (n // 3) + ["large"] * (n - 2 * (n // 3))
    random.shuffle(sizes)

    for i, size in enumerate(sizes):
        doc = generate_html_doc(size)
        (path / f"doc_{i:05d}.html").write_text(doc, encoding="utf-8")

    print(f"[OK] Generated {n} HTML files in '{out_dir}'")


# ─── Numeric array ───────────────────────────────────────────────────────────

def generate_array(out_file: str = "data/array.npy", n: int = 2_000_000):
    """Generate array of n numbers using exponential distribution (non-normal)."""
    arr = np.random.exponential(scale=1000.0, size=n).astype(np.float64)
    np.save(out_file, arr)
    print(f"[OK] Array of {n:,} numbers saved to '{out_file}'")
    return arr


# ─── Matrices ────────────────────────────────────────────────────────────────

def generate_matrices(out_a: str = "data/matrix_a.npy",
                      out_b: str = "data/matrix_b.npy",
                      size: int = 1200):
    """Generate two random float32 matrices of shape (size x size)."""
    a = np.random.uniform(-10, 10, (size, size)).astype(np.float32)
    b = np.random.uniform(-10, 10, (size, size)).astype(np.float32)
    np.save(out_a, a)
    np.save(out_b, b)
    print(f"[OK] Matrices {size}x{size} saved to '{out_a}', '{out_b}'")


# ─── Financial transactions ──────────────────────────────────────────────────

CURRENCIES    = ["USD", "EUR", "GBP", "UAH", "PLN", "CHF", "JPY"]
PRODUCT_TYPES = ["electronics", "clothing", "food", "software", "services",
                 "books", "travel", "health"]

def generate_transactions(out_file: str = "data/transactions.csv", n: int = 500_000):
    """Generate CSV with n rows: user_id, amount, currency, date, product_type."""
    Path("data").mkdir(exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "amount", "currency", "date", "product_type"])

        from datetime import date, timedelta
        for _ in range(n):
            user_id      = random.randint(1, 100_000)
            amount       = round(random.uniform(1.0, 5000.0), 2)
            currency     = random.choice(CURRENCIES)
            day_offset   = random.randint(0, 365 * 2)
            tx_date      = (date(2023, 1, 1) + timedelta(days=day_offset)).isoformat()
            product_type = random.choice(PRODUCT_TYPES)
            writer.writerow([user_id, amount, currency, tx_date, product_type])

    print(f"[OK] {n:,} transactions saved to '{out_file}'")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    Path("data").mkdir(exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "html"):
        generate_html_files()
    if target in ("all", "array"):
        generate_array()
    if target in ("all", "matrices"):
        generate_matrices()
    if target in ("all", "transactions"):
        generate_transactions()

    print("\nAll data generated successfully.")