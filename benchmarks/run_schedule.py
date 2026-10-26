"""Scheduling of the chunked passes: time and peak memory by thread count,
chunk rule and grid length.

The peer review's P1 table (``dev/peer-review.qmd``) measured the quantiles
method at 262,144 rows with the fixed 65,536-row chunk against 4,096-row
chunks and found a 32-thread machine mostly idle at the default: four chunks
for 32 workers. From 2.4.2 the default chunk is a rule of the row count and
the grid length, ``clamp(max(ceil(rows / 64), ceil(n / 64)), 4096, 65536)``,
and a pass runs its chunks in waves
of ``min(2 * threads, 1 GiB / partial size)``, so the partials in flight are
bounded by memory as well as by threads. This script reruns that table with
the grid length as a third axis and the peak working set beside each time,
since a quantiles partial is two grid-length vectors (256 MiB at 2^24
buckets).

Usage::

    uv run python benchmarks/run_schedule.py [--rows 262144 1000000]
        [--threads 1 8 16 32] [--grids 16 20 24] [--repeats 3] [--markdown]

Each configuration runs in a child process, so its peak working set is its
own. A run over ``--long`` seconds (15 by default) is not repeated; the
survival method is skipped below 16 threads at 2^24 buckets, where every
tail is walked across millions of edges and one run takes minutes. The
record goes to ``benchmarks/results.json`` under ``"kind": "schedule"`` and
the table to stdout; ``benchmarks/report-2.4.2.md`` holds the figures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eltq

sys.path.insert(0, str(Path(__file__).parent))
from run_end_to_end import (RESULTS_PATH, cols_tuple, machine_info,  # noqa: E402
                            peak_working_set_mb, synthetic_elt)

LOSS_CAP = 5e6   # the synthetic portfolio's largest exposed value
# (label, method, chunk_rows): the fixed chunk of versions before 2.4.2, the
# rule, and the survival method at its fixed 256.
CONFIGS = (
    ("quantiles, chunk 65,536", "quantiles", 65_536),
    ("quantiles, default rule", "quantiles", None),
    ("survival, chunk 256", "survival", None),
)


def bucket_size(grid_log2: int) -> float:
    """The h at which 2**grid_log2 buckets cover the loss cap (the grid
    length is also passed explicitly; nothing overflows)."""
    return LOSS_CAP / (2.0 ** grid_log2 - 1.5)


def child(rows: int, threads: int, grid_log2: int, method: str, chunk, repeats: int, long: float) -> None:
    """One configuration, timed in this process; JSON on stdout. A first run
    longer than ``long`` seconds stands alone."""
    cols = cols_tuple(synthetic_elt(rows))
    before = peak_working_set_mb()
    kwargs = {"bucket_size": bucket_size(grid_log2), "grid_length": 2 ** grid_log2, "method": method,
              "threads": threads, "quadrature_points": 32}
    if chunk is not None:
        kwargs["chunk_rows"] = chunk
    times = []
    meta = None
    for _ in range(repeats):
        start = time.perf_counter()
        _, _, meta = eltq.quantize_arrays(*cols, **kwargs)
        times.append(time.perf_counter() - start)
        if times[-1] > long:
            break
    after = peak_working_set_mb()
    print(json.dumps({
        "elapsed_s": statistics.median(times),
        "repeats": len(times),
        "grid_length": meta.grid_length,
        "peak_working_set_mb": after,
        "run_working_set_mb": None if before is None or after is None else round(max(0.0, after - before), 1),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[262_144, 1_000_000])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--grids", type=int, nargs="+", default=[16, 20, 24],
                        help="log2 of the grid lengths")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--long", type=float, default=15.0, help="seconds beyond which a run is not repeated")
    parser.add_argument("--quantiles-only", action="store_true", help="skip the survival method's column")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--child", nargs=7, metavar=("ROWS", "THREADS", "GRID", "METHOD", "CHUNK", "REPEATS", "LONG"))
    args = parser.parse_args()
    if args.child:
        rows, threads, grid, method, chunk, repeats, long = args.child
        child(int(rows), int(threads), int(grid), method, None if chunk == "none" else int(chunk), int(repeats), float(long))
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "schedule",
        "machine": machine_info(),
        "quadrature_points": 32,
        "runs": [],
    }
    for rows in args.rows:
        for grid in args.grids:
            for threads in args.threads:
                for label, method, chunk in CONFIGS:
                    if method == "survival" and (args.quantiles_only or (grid >= 24 and threads < 16)):
                        print(f"rows={rows:>9,} grid=2^{grid} threads={threads:>2} {label:<24} skipped", flush=True)
                        continue
                    cmd = [sys.executable, __file__, "--child", str(rows), str(threads), str(grid),
                           method, "none" if chunk is None else str(chunk), str(args.repeats), str(args.long)]
                    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    result = json.loads(out.stdout.strip().splitlines()[-1])
                    result.update({"rows": rows, "threads": threads, "grid_log2": grid,
                                   "method": method, "chunk_rows": chunk, "label": label})
                    record["runs"].append(result)
                    print(f"rows={rows:>9,} grid=2^{grid} threads={threads:>2} {label:<24} "
                          f"{result['elapsed_s']:8.3f} s  peak {result['peak_working_set_mb']} MiB", flush=True)
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    results.append(record)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    if args.markdown:
        for rows in args.rows:
            print(f"\n{rows:,} rows, K = 32; seconds (median of {args.repeats}, one run beyond {args.long:g} s) and peak working set in MiB.\n")
            header = ["grid", "threads"] + [f"{label} s" for label, _, _ in CONFIGS] + [f"{label} MiB" for label, _, _ in CONFIGS]
            print("| " + " | ".join(header) + " |")
            print("|" + "|".join("---:" for _ in header) + "|")
            for grid in args.grids:
                for threads in args.threads:
                    cells = [f"$2^{{{grid}}}$", str(threads)]
                    picked = {r["label"]: r for r in record["runs"]
                              if r["rows"] == rows and r["grid_log2"] == grid and r["threads"] == threads}
                    cells += [f"{picked[label]['elapsed_s']:.3f}" if label in picked else "not run" for label, _, _ in CONFIGS]
                    cells += [f"{picked[label]['peak_working_set_mb']:,.0f}" if label in picked else "" for label, _, _ in CONFIGS]
                    print("| " + " | ".join(cells) + " |")
    print(f"\neltq {eltq.__version__}; {np.__version__=}")


if __name__ == "__main__":
    main()
