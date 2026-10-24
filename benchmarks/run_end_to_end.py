"""End-to-end benchmark harness for eltq.

Run from the repository root (release wheel installed by `uv sync`):

    uv run python benchmarks/run_end_to_end.py [--rows 1000000] [--quick]
        [--scratch-dir PATH] [--method quantiles|survival]
        [--nodes-per-decade N] [--survival-sweep] [--small-alpha | --large-m]

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
- bitwise determinism check across thread counts;
- with ``--survival-sweep``, the survival method at 25, 50, 100 and 150
  nodes per decade against the quantiles method at K = 32, on the mixture
  portfolio and (``--small-alpha``) on a small-alpha portfolio with the
  recipe of ``fixtures/small_alpha_elt.csv`` at scale, recording the
  interpolation error and the secondary mean error of each; with
  ``--large-m`` (``--rows 100000``) the survival method at the default
  density only against K = 32 on a large-exposed-value portfolio at
  ``h = 25`` and ``2**22`` buckets, the tail-cost case of
  ``dev/done/plan-survival-tail-cost.md``.

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
    # Linear split: the total sd is sd_indep + sd_corr (Burke rule, 1.6.0),
    # so the regime mix above is preserved exactly.
    share = rng.uniform(0.0, 1.0, size=n)
    return {
        "rate": lam,
        "max_loss": m,
        "mean_loss": mu,
        "sd_indep": sigma * share,
        "sd_corr": sigma * (1.0 - share),
    }


def small_alpha_elt(n: int) -> dict[str, np.ndarray]:
    """Small-alpha portfolio at scale: the recipe of the test fixture
    (alpha loguniform on [0.003, 3], M / mu on [1e2, 1e4] with a tenth of
    the rows at 1e5), ground-up rows only."""
    rng = np.random.default_rng(SEED + 1)
    m = np.exp(rng.uniform(np.log(1e4), np.log(1e6), size=n))
    ratio = np.exp(rng.uniform(np.log(1e2), np.log(1e4), size=n))
    deep = rng.random(n) < 0.1
    m[deep] = 1e6
    ratio[deep] = 1e5
    alpha = np.exp(rng.uniform(np.log(0.003), np.log(3.0), size=n))
    lam = np.exp(rng.uniform(np.log(1e-5), np.log(1e-2), size=n))
    p = 1.0 / ratio
    kappa = alpha / p
    v = p * (1.0 - p) / (kappa + 1.0)
    sigma = m * np.sqrt(v)
    share = rng.uniform(0.0, 1.0, size=n)
    return {
        "rate": lam,
        "max_loss": m,
        "mean_loss": p * m,
        "sd_indep": sigma * share,
        "sd_corr": sigma * (1.0 - share),
    }


def large_m_elt(n: int) -> dict[str, np.ndarray]:
    """Large-exposed-value portfolio: M log-uniform on [1e6, 1e9], 98% of
    the rows low-mean (mean damage ratio 1e-5 to 1e-2, CV 1 to 20) and 2%
    top-heavy (ratio 0.5 to 0.9, CV 0.3 to 0.6), the recipe of reproduction
    1 in ``dev/done/plan-survival-tail-cost.md``. At ``h = 25`` and 2**22
    buckets the window ends at 1.05e8, below a third of the endpoints."""
    rng = np.random.default_rng(SEED + 2)
    rate = 10.0 ** rng.uniform(-6, -2, n)
    m = 10.0 ** rng.uniform(6, 9, n)
    top_heavy = rng.random(n) < 0.02
    p = np.where(top_heavy, rng.uniform(0.5, 0.9, n), 10.0 ** rng.uniform(-5, -2, n))
    mean = p * m
    cv = np.where(top_heavy, rng.uniform(0.3, 0.6, n), 10.0 ** rng.uniform(0, np.log10(20), n))
    sd = np.minimum(cv * mean, 0.95 * np.sqrt(mean * (m - mean)))
    return {
        "rate": rate,
        "max_loss": m,
        "mean_loss": mean,
        "sd_indep": 0.6 * sd,
        "sd_corr": 0.4 * sd,
    }


LARGE_M = {"bucket_size": 25.0, "grid_length": 2 ** 22}


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
                        "variance_atom", "variance_bernoulli")
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
    from eltq.reference import quantize_reference

    sub = {k: v[:subsample] for k, v in cols.items()}
    elapsed, _ = timed(lambda: quantize_reference(*cols_tuple(sub), **params))
    return subsample / elapsed


def survival_sweep(args) -> None:
    """Survival at four node densities against quantiles at K = 32, on the
    mixture portfolio or the small-alpha one; appends a ``survival_sweep``
    record to the results history."""
    label = "small_alpha" if args.small_alpha else "large_m" if args.large_m else "mixture"
    print(f"generating {args.rows:,} {label} rows ...")
    if args.large_m:
        cols = large_m_elt(args.rows)
    elif args.small_alpha:
        cols = small_alpha_elt(args.rows)
    else:
        cols = synthetic_elt(args.rows)
    h = LARGE_M["bucket_size"] if args.large_m else 100.0 if args.small_alpha else HEADLINE["bucket_size"]
    window = {"grid_length": LARGE_M["grid_length"]} if args.large_m else {}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": "survival_sweep",
        "portfolio": label,
        "rows": args.rows,
        "bucket_size": h,
        **window,
        "machine": machine_info(),
        "runs": [],
    }
    configs = [{"method": "quantiles", "quadrature_points": 32}]
    if args.large_m:
        # The default density only, once: the survival grid here is the
        # tail-cost case and runs for minutes before the rate cut.
        configs += [{"method": "survival"}]
        repeats = 1
    else:
        configs += [{"method": "survival", "nodes_per_decade": d} for d in (25, 50, 100, 150)]
        repeats = 2
    base = None
    for params in configs:
        elapsed, (raw, secondary, meta) = timed(
            lambda p=params: eltq.quantize_arrays(*cols_tuple(cols), bucket_size=h, **window, **p),
            repeats=repeats,
        )
        run = {
            **params,
            "grid_length": meta.grid_length,
            "elapsed_s": round(elapsed, 4),
            "rows_per_s": round(args.rows / elapsed),
            "secondary_mean_error_rel": meta.secondary_mean_error / meta.analytic_aal,
            "interpolation_error": meta.interpolation_error,
            "nodes": meta.nodes,
            "mass_error": abs(meta.secondary_mass - meta.total_rate),
        }
        if params["method"] == "quantiles":
            base = elapsed
            run["ratio_to_k32"] = 1.0
        else:
            run["ratio_to_k32"] = round(elapsed / base, 2)
        record["runs"].append(run)
        density = params.get("nodes_per_decade", "-")
        print(f"{params['method']:>9} density={density:>4}: {elapsed:7.3f} s "
              f"({run['ratio_to_k32']:>5}x K=32)  mean err {run['secondary_mean_error_rel']:+.2e}  "
              f"interp err {meta.interpolation_error if meta.interpolation_error is not None else float('nan'):.1e}  "
              f"nodes {meta.nodes}  n={meta.grid_length}")
    if args.large_m:
        # A single-thread survival run would take too long here; the test
        # suite covers determinism.
        record["determinism_bitwise_across_threads"] = None
    else:
        r1, s1, _ = eltq.quantize_arrays(*cols_tuple(cols), bucket_size=h, method="survival", threads=1)
        r2, s2, _ = eltq.quantize_arrays(*cols_tuple(cols), bucket_size=h, method="survival")
        record["determinism_bitwise_across_threads"] = bool(np.array_equal(r1, r2) and np.array_equal(s1, s2))
        print(f"bitwise deterministic across threads (survival): {record['determinism_bitwise_across_threads']}")
    record["peak_working_set_mb"] = peak_working_set_mb()
    history = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    history.append(record)
    RESULTS_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"appended record to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--quick", action="store_true",
                        help="headline config only, no CSV/scaling sweeps")
    parser.add_argument("--scratch-dir", type=Path,
                        default=Path(tempfile.gettempdir()))
    parser.add_argument("--scipy-subsample", type=int, default=20_000)
    parser.add_argument("--method", choices=["quantiles", "survival"],
                        default="quantiles",
                        help="method for the standard runs")
    parser.add_argument("--nodes-per-decade", type=int, default=None,
                        help="survival node density for the standard runs")
    parser.add_argument("--survival-sweep", action="store_true",
                        help="survival at 25/50/100/150 nodes per decade "
                             "against quantiles at K = 32, then stop")
    parser.add_argument("--small-alpha", action="store_true",
                        help="use the small-alpha portfolio for the sweep")
    parser.add_argument("--large-m", action="store_true",
                        help="use the large-exposed-value portfolio for the sweep "
                             "(h = 25, 2**22 buckets, default density only)")
    args = parser.parse_args()

    if args.survival_sweep:
        survival_sweep(args)
        return

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
    method = {"method": args.method}
    if args.nodes_per_decade is not None:
        method["nodes_per_decade"] = args.nodes_per_decade
    headline_rate = None
    for params in configs:
        elapsed, (raw, secondary, meta) = timed(
            lambda p=params: eltq.quantize_arrays(*cols_tuple(cols), **p, **method),
            repeats=2,
        )
        kernel_rows = beta_row_count(meta)
        quantiles = kernel_rows * params["quadrature_points"]
        run = {
            "kind": "in_memory",
            **params,
            **method,
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
