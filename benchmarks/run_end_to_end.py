"""End-to-end benchmark harness for eltq.

Run from the repository root (release wheel installed by `uv sync`):

    uv run python benchmarks/run_end_to_end.py [--rows 1000000] [--quick]
        [--scratch-dir PATH]

Generates a seeded synthetic portfolio ELT (mixture of well-conditioned,
skewed alpha < 1, concentrated large-parameter, degenerate, inadmissible and
zero-rate rows per specification section 10), measures the workloads below,
prints a summary, and appends a machine-readable record to
``benchmarks/results.json``:

- in-memory quantize_arrays across (K, bucket) configurations;
- thread scaling at the headline configuration (1 M rows, K = 32);
- non-kernel overhead (same rows with sigma = 0: validation + deposits only);
- CSV route (parse + compute) and memory-mapped .npy output;
- SciPy reference throughput on a subsample, extrapolated for the speedup;
- bitwise determinism check across thread counts.

Scratch files (the benchmark CSV) go to --scratch-dir, defaulting to the
system temp dir; keep them off Synology-synced folders.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eltq

REPO_ROOT = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "results.json"
SEED = 20260814

HEADLINE = {"quadrature_points": 32, "bucket_size": 5000.0}
CONFIGS = [
    {"quadrature_points": 32, "bucket_size": 5000.0},   # headline, n ~ 1e3
    {"quadrature_points": 32, "bucket_size": 50.0},     # large grid, n ~ 1e5
    {"quadrature_points": 8, "bucket_size": 5000.0},
    {"quadrature_points": 128, "bucket_size": 5000.0},
]


def synthetic_elt(n: int) -> dict[str, np.ndarray]:
    """Seeded mixture ELT at portfolio scale (loss cap 5e6)."""
    rng = np.random.default_rng(SEED)
    m = np.exp(rng.uniform(np.log(1e4), np.log(5e6), size=n))
    lam = np.exp(rng.uniform(np.log(1e-6), np.log(1e-2), size=n))
    p = 0.05 + 0.9 * rng.beta(2.0, 2.0, size=n)
    f = rng.uniform(0.05, 0.6, size=n)
    kind = rng.choice(
        ["well", "skew", "conc", "sig0", "bound", "inadm", "zrate"],
        size=n,
        p=[0.70, 0.10, 0.05, 0.06, 0.03, 0.04, 0.02],
    )
    skew = kind == "skew"
    p[skew] = rng.uniform(0.005, 0.05, size=skew.sum())
    f[skew] = rng.uniform(0.5, 0.95, size=skew.sum())
    conc = kind == "conc"
    f[conc] = rng.uniform(0.001, 0.01, size=conc.sum())
    inadm = kind == "inadm"
    f[inadm] = rng.uniform(1.0, 1.5, size=inadm.sum())
    mu = p * m
    sigma = np.sqrt(f * p * (1.0 - p)) * m
    sigma[kind == "sig0"] = 0.0
    bound = kind == "bound"
    at_max = rng.random(size=n) < 0.5
    mu[bound & at_max] = m[bound & at_max]
    mu[bound & ~at_max] = 0.0
    lam[kind == "zrate"] = 0.0
    share = rng.uniform(0.0, 1.0, size=n)
    return {
        "rate": lam,
        "max_loss": m,
        "mean_loss": mu,
        "sd_indep": sigma * np.sqrt(share),
        "sd_corr": sigma * np.sqrt(1.0 - share),
    }


def cols_tuple(cols: dict[str, np.ndarray]) -> tuple:
    return (cols["rate"], cols["max_loss"], cols["mean_loss"],
            cols["sd_indep"], cols["sd_corr"])


def timed(fn, repeats: int = 1):
    """Best-of-N wall time plus the last result."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def beta_row_count(meta) -> int:
    """Rows that exercised the inverse-beta kernel."""
    skipped = sum(
        issues.count
        for category, issues in meta.diagnostics.items()
        if category in ("nonfinite", "negative_field", "mean_exceeds_max",
                        "zero_rate", "degenerate_atom", "variance_dropped",
                        "variance_atom")
    )
    return meta.input_rows - skipped


def peak_working_set_mb() -> float | None:
    """Peak RSS of this process in MiB (Windows psapi)."""
    if platform.system() != "Windows":
        return None

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(PMC),
                                           ctypes.c_uint32]
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / (1024 * 1024), 1)


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "eltq": eltq.__version__,
    }
    try:
        out = subprocess.run(["rustc", "--version"], capture_output=True,
                             text=True, timeout=10)
        info["rustc"] = out.stdout.strip()
    except OSError:
        info["rustc"] = None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20)
        if out.stdout.strip():
            info["cpu_name"] = out.stdout.strip()
    except OSError:
        pass
    return info


def scipy_reference_rate(cols: dict, subsample: int, params: dict) -> float:
    """SciPy reference rows/s measured on a subsample."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from reference import quantize_reference

    sub = {k: v[:subsample] for k, v in cols.items()}
    elapsed, _ = timed(lambda: quantize_reference(*cols_tuple(sub), **params))
    return subsample / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--quick", action="store_true",
                        help="headline config only, no CSV/scaling sweeps")
    parser.add_argument("--scratch-dir", type=Path,
                        default=Path(tempfile.gettempdir()))
    parser.add_argument("--scipy-subsample", type=int, default=20_000)
    args = parser.parse_args()

    print(f"generating {args.rows:,} synthetic rows ...")
    cols = synthetic_elt(args.rows)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": args.rows,
        "machine": machine_info(),
        "runs": [],
    }

    # In-memory configurations.
    configs = [HEADLINE] if args.quick else CONFIGS
    headline_rate = None
    for params in configs:
        elapsed, (raw, secondary, meta) = timed(
            lambda p=params: eltq.quantize_arrays(*cols_tuple(cols), **p),
            repeats=2,
        )
        kernel_rows = beta_row_count(meta)
        quantiles = kernel_rows * params["quadrature_points"]
        run = {
            "kind": "in_memory",
            **params,
            "grid_length": meta.grid_length,
            "elapsed_s": round(elapsed, 4),
            "rows_per_s": round(args.rows / elapsed),
            "beta_rows": kernel_rows,
            "quantiles_per_s": round(quantiles / elapsed),
            "mass_error": abs(meta.raw_mass - meta.total_rate),
            "secondary_mean_error": meta.secondary_mean_error,
        }
        record["runs"].append(run)
        print(f"in-memory K={params['quadrature_points']:>3} "
              f"h={params['bucket_size']:>6}: {elapsed:7.3f} s  "
              f"{run['rows_per_s']:>10,} rows/s  "
              f"{run['quantiles_per_s']:>12,} quantiles/s  n={meta.grid_length}")
        if params == HEADLINE:
            headline_rate = run["rows_per_s"]

    # Non-kernel overhead: same rows, sigma zeroed (validation + deposits).
    zeroed = dict(cols, sd_indep=np.zeros_like(cols["sd_indep"]),
                  sd_corr=np.zeros_like(cols["sd_corr"]))
    elapsed, _ = timed(lambda: eltq.quantize_arrays(*cols_tuple(zeroed),
                                                    **HEADLINE))
    record["runs"].append({"kind": "non_kernel_overhead", **HEADLINE,
                           "elapsed_s": round(elapsed, 4)})
    print(f"non-kernel overhead (sigma=0 pass):   {elapsed:7.3f} s")

    if not args.quick:
        # Thread scaling at the headline configuration.
        for threads in (1, 2, 4, 8, None):
            elapsed, _ = timed(
                lambda t=threads: eltq.quantize_arrays(*cols_tuple(cols),
                                                       **HEADLINE, threads=t))
            record["runs"].append({"kind": "thread_scaling",
                                   "threads": threads,
                                   "elapsed_s": round(elapsed, 4),
                                   "rows_per_s": round(args.rows / elapsed)})
            label = threads if threads is not None else "all"
            print(f"threads={label:>3}: {elapsed:7.3f} s  "
                  f"{args.rows / elapsed:>12,.0f} rows/s")

        # CSV route: parse + compute in Rust.
        scratch = args.scratch_dir / "eltq-bench"
        scratch.mkdir(parents=True, exist_ok=True)
        csv_path = scratch / f"bench_{args.rows}.csv"
        if not csv_path.exists():
            print("writing benchmark CSV ...")
            import pandas as pd

            pd.DataFrame(cols).to_csv(csv_path, index=False)
        elapsed, _ = timed(lambda: eltq.quantize_csv(csv_path, **HEADLINE))
        record["runs"].append({"kind": "csv_route", **HEADLINE,
                               "elapsed_s": round(elapsed, 4),
                               "rows_per_s": round(args.rows / elapsed),
                               "csv_bytes": csv_path.stat().st_size})
        print(f"csv route (parse + compute):          {elapsed:7.3f} s")

        # Memory-mapped output.
        npy_path = scratch / "bench_grids.npy"
        elapsed, _ = timed(lambda: eltq.quantize_to_npy(cols_tuple(cols),
                                                        npy_path, **HEADLINE))
        record["runs"].append({"kind": "npy_output", **HEADLINE,
                               "elapsed_s": round(elapsed, 4)})
        print(f"memmap .npy output:                   {elapsed:7.3f} s")

        # Determinism: bitwise across thread counts.
        r1, s1, _ = eltq.quantize_arrays(*cols_tuple(cols), **HEADLINE,
                                         threads=1)
        r2, s2, _ = eltq.quantize_arrays(*cols_tuple(cols), **HEADLINE)
        deterministic = bool(np.array_equal(r1, r2) and np.array_equal(s1, s2))
        record["determinism_bitwise_across_threads"] = deterministic
        print(f"bitwise deterministic across threads: {deterministic}")

    # SciPy reference for the speedup headline.
    scipy_rate = scipy_reference_rate(cols, args.scipy_subsample, HEADLINE)
    record["scipy_reference"] = {
        "subsample_rows": args.scipy_subsample,
        "rows_per_s": round(scipy_rate),
        "extrapolated_s_for_full_input": round(args.rows / scipy_rate, 1),
    }
    if headline_rate:
        record["speedup_vs_scipy"] = round(headline_rate / scipy_rate, 1)
        print(f"scipy reference: {scipy_rate:,.0f} rows/s  ->  "
              f"eltq speedup {record['speedup_vs_scipy']}x")

    record["peak_working_set_mb"] = peak_working_set_mb()

    history = []
    if RESULTS_PATH.exists():
        history = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    history.append(record)
    RESULTS_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"appended record to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
