"""End-to-end multi-unit benchmark harness (1.11.0).

Run from the repository root (release wheel installed by `uv sync`):

    uv run python benchmarks/run_multiunit.py [--events 100000] [--units 10]
        [--quick] [--no-append]

Generates a seeded synthetic unit-level portfolio ELT (events x units, mixed
regimes: correlated betas, independent-only events, comonotone-only events,
atom units, single-unit events, dropped rows, zero-rate events) with typical
term sparsity (four of ten units carry policy terms, two of those also
occurrence reinsurance), measures the workloads below, prints a summary, and
appends a machine-readable record to ``benchmarks/results.json`` under
``"suite": "multiunit"``:

- all four views (default), convolution auto / direct / fft;
- ground_up only (the all-four delta);
- z_nodes 32 versus 8 (the z wall);
- thread scaling at 1 / 2 / 4 / all cores;
- the two-unit joint at reference accuracy (Q = 1001 and 4001);
- bitwise determinism across thread counts;
- peak RSS and machine metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eltq

sys.path.insert(0, str(Path(__file__).parent))
from run_end_to_end import machine_info, peak_working_set_mb, timed  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "results.json"
SEED = 20260826
BUCKET = 40000.0  # n ~ 128 per unit at M = 5e6

TERMS = """\
unit u00 2000000 xs 200000 occurrence net of 50% po 800000 xs 200000 and 1000000 xs 1000000
unit u01 3000000 xs 0 occurrence net of 400000 po 1600000 xs 400000
unit u02 4000000 xs 120000
unit u03 occurrence net of 25% po 2000000 xs 0
"""


def synthetic_portfolio(events: int, units: int) -> dict[str, np.ndarray]:
    """Seeded unit-level ELT with mixed regimes per event."""
    rng = np.random.default_rng(SEED)
    n = events * units
    event_id = np.repeat(np.arange(events), units)
    unit_id = np.tile(np.array([f"u{i:02d}" for i in range(units)], dtype=object), events)
    rate = np.repeat(np.exp(rng.uniform(np.log(1e-6), np.log(1e-2), size=events)), units)
    m = np.exp(rng.uniform(np.log(1e5), np.log(5e6), size=n))
    p = 0.05 + 0.9 * rng.beta(2.0, 2.0, size=n)
    f = rng.uniform(0.05, 0.6, size=n)
    mu = p * m
    sigma = np.sqrt(f * p * (1.0 - p)) * m
    share = rng.uniform(0.0, 1.0, size=n)
    kind = rng.choice(
        ["mixed", "independent", "comonotone", "atoms", "single", "zero_rate"],
        size=events, p=[0.68, 0.10, 0.05, 0.05, 0.10, 0.02],
    )
    kind_rows = np.repeat(kind, units)
    share[kind_rows == "independent"] = 1.0     # sd_corr = 0
    share[kind_rows == "comonotone"] = 0.0      # sd_indep = 0
    sigma[kind_rows == "atoms"] = 0.0
    rate[kind_rows == "zero_rate"] = 0.0
    keep = np.ones(n, dtype=bool)
    single = kind_rows == "single"
    keep[single & (np.tile(np.arange(units), events) != 0)] = False
    # A sprinkle of dropped rows (mean above max) inside live events.
    dropped = rng.random(n) < 0.005
    mu[dropped] = m[dropped] * 1.5
    cols = {
        "event_id": event_id[keep],
        "unit_id": unit_id[keep],
        "rate": rate[keep],
        "mean_loss": mu[keep],
        "sd_indep": (sigma * share)[keep],
        "sd_corr": (sigma * (1.0 - share))[keep],
        "max_loss": m[keep],
    }
    return cols


def run(cols: dict, **options):
    return eltq.multiunit_views(
        cols["event_id"], cols["unit_id"], cols["rate"], cols["mean_loss"],
        cols["sd_indep"], cols["sd_corr"], cols["max_loss"],
        terms=TERMS, bucket_size=BUCKET, **options,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--units", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="skip the slow sweeps")
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()
    if eltq.BACKEND != "rust":
        raise SystemExit("the benchmark needs the Rust backend")

    cols = synthetic_portfolio(args.events, args.units)
    rows = cols["rate"].size
    print(f"portfolio: {args.events} events x {args.units} units -> {rows} rows")
    record: dict = {
        "suite": "multiunit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        "events": args.events,
        "units": args.units,
        "rows": rows,
        "bucket_size": BUCKET,
        "runs": {},
    }

    def measure(label: str, repeats: int = 1, **options):
        seconds, (views, meta) = timed(lambda: run(cols, **options), repeats)
        entry = {
            "seconds": round(seconds, 3),
            "rows_per_s": round(rows / seconds),
            "grid_length": meta.grid_length,
            "valid_rows": meta.valid_rows,
            "valid_events": meta.valid_events,
            "engine": meta.engine,
            "max_view_mean_rel_error": max(
                abs(meta.view_mean_error[v]) / max(abs(meta.analytic_view_mean[v]), 1e-300)
                for v in meta.views
            ),
            "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in options.items()},
        }
        record["runs"][label] = entry
        print(f"{label:28s} {seconds:8.2f} s  {rows / seconds:9.0f} rows/s  "
              f"paths direct/fft {meta.engine['direct_paths']}/{meta.engine['fft_paths']}  "
              f"n={meta.grid_length}")
        return views, meta

    base_views, base_meta = measure("all4 auto q32", repeats=2)
    measure("ground_up only q32", views=("ground_up",))
    measure("all4 auto q8", z_nodes=8)
    if not args.quick:
        measure("all4 direct q32", convolution="direct")
        measure("all4 fft q32", convolution="fft")
        measure("all4 gauss_hermite q32", z_rule="gauss_hermite")

    # Thread scaling.
    cores = machine_info()["logical_cores"] or 1
    scaling = {}
    for threads in sorted({1, 2, 4, cores}):
        if args.quick and threads not in (1, cores):
            continue
        _, meta = measure(f"all4 auto threads={threads}", threads=threads)
        scaling[str(threads)] = record["runs"][f"all4 auto threads={threads}"]["seconds"]
    record["thread_scaling_seconds"] = scaling

    # Determinism: bitwise identical across thread counts.
    one, _ = run(cols, threads=1)
    identical = all(np.array_equal(one[v], base_views[v]) for v in base_meta.views)
    record["deterministic_across_threads"] = bool(identical)
    print(f"bitwise identical across thread counts: {identical}")

    # Two-unit joint at reference accuracy.
    units = ((200.0, 100.0, 150.0, 1000.0), (500.0, 300.0, 200.0, 2000.0))
    joint = {}
    for q in (1001, 4001):
        seconds, (pmf, _, _, meta) = timed(
            lambda q=q: eltq.joint_two_units(*units, bucket_size=10.0, n1=101, n2=201, z_nodes=q),
            repeats=3,
        )
        joint[str(q)] = {"seconds": round(seconds, 4), "mass": meta.total_mass}
        print(f"joint two units Q={q:5d}  {seconds * 1e3:7.1f} ms")
    record["joint_two_units"] = joint
    record["peak_rss_mb"] = peak_working_set_mb()
    print(f"peak RSS {record['peak_rss_mb']} MiB")

    if not args.no_append:
        history = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
        history.append(record)
        RESULTS_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
