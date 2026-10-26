"""Accuracy and cost of both methods, read together.

The speed suite (``run_end_to_end.py``) compares times; this suite reports
the error the time buys, so that a speed comparison is read at a stated
accuracy (``dev/done/plan-peer-review-response.md``, [mesh] and [benchmark]).
Every error below is against SciPy's regularized incomplete beta (the normal
law from ``min(alpha, beta) = 1e16``, where SciPy itself is off), summed
exactly over the events, and is the largest relative error by band of the
exceedance rate as a share of the total rate.

Sections (``--section``, default ``all``):

- ``bell``: the fine mesh's accuracy knob. Bell-shaped events (both beta
  parameters above 1) are the regime the shared cubic cannot follow on the
  base mesh alone; the fine mesh gives each ``nodes_per_sigma`` nodes per
  standard deviation around its mean, and the table shows what each value
  buys, at every fine-interval midpoint and at ``mu + z sigma`` for ``z``
  from -3 to 7.
- ``tails``: endpoint tails. The curve of a rate-one event with its mass
  against the exposed value, evaluated at distances ``M - x`` from 1e-8 M
  to 0.1 M, where the per-event tail interpolant and its power law act.
- ``large``: large parameters, ``min(alpha, beta)`` from 2e5 to 1e20, where
  the survival method takes the narrow route and the quantiles method the
  Cornish-Fisher expansion: the largest bucket-mass error as a share of the
  event's rate, on a grid of a quarter standard deviation (at least 1e-6 of
  the exposed value), the atoms of the fixtures set aside as described below.
- ``mixtures``: the two fixtures and samples of the two million-row benchmark
  portfolios under both methods, the bucketed tail sums of the beta events
  against their exact exceedance at every bucket edge (the atoms, which the
  linear rule splits across two nodes, are set aside), beside the engine's
  time on the full million rows at every core.

Usage::

    uv run python benchmarks/run_accuracy.py [--section all] [--rows 1000000]
        [--markdown]

The output is recorded in ``benchmarks/report-<version>.md``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

import eltq
from eltq import reference

sys.path.insert(0, str(Path(__file__).parent))
from run_end_to_end import cols_tuple, small_alpha_elt, synthetic_elt  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BANDS = ((1e-2, 1.0), (1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6), (1e-12, 1e-8))
BAND_HEADERS = [f"[{lo:.0e},{hi:.0e})" for lo, hi in BANDS]
K = 32


def fmt(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.1e}"


def table(header, rows, markdown: bool) -> None:
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---:" if i else ":---" for i, _ in enumerate(header)) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
        for r in rows:
            print("  ".join(c.rjust(w) for c, w in zip(r, widths)))
    print()


def band_errors(got: np.ndarray, exact: np.ndarray, total: float) -> np.ndarray:
    """Largest relative error by band of ``exact / total``; NaN where the
    band is empty."""
    worst = np.full(len(BANDS), np.nan)
    for i, (lo, hi) in enumerate(BANDS):
        keep = (exact >= lo * total) & (exact < hi * total)
        if keep.any():
            worst[i] = np.max(np.abs(got[keep] / exact[keep] - 1.0))
    return worst


def tail_sums(grid: np.ndarray) -> np.ndarray:
    """The mass above the edge ``e_i = (i + 1/2) h`` for ``i = 0 .. n - 2``."""
    return np.cumsum(grid[::-1])[::-1][1:]


def one_row(alpha: float, beta: float, maximum: float = 1.0, rate: float = 1.0):
    p = alpha / (alpha + beta)
    sd = maximum * math.sqrt(p * (1.0 - p) / (alpha + beta + 1.0))
    return tuple(np.array([v]) for v in (rate, maximum, maximum * p, sd, 0.0)), maximum * p, sd


# ---------------------------------------------------------------- bell

BELL_ALPHAS = (1.5, 3.0, 10.0, 1e2, 1e3, 1e4)
BELL_CVS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1)
SIGMAS = (8, 12, 16, 20)


def bell_row(alpha: float, cv: float):
    """A rate-one event on [0, 1] with shape ``alpha`` and coefficient of
    variation ``cv``; None when no beta with ``beta > 1`` has them."""
    if cv * cv * alpha >= 1.0:
        return None
    beta = cv * cv * alpha * (alpha + 1.0) / (1.0 - cv * cv * alpha)
    if beta <= 1.0:
        return None
    return beta


def bell_errors(nodes_per_sigma: int):
    worst = np.zeros(len(BANDS))
    fine_nodes = 0
    for alpha in BELL_ALPHAS:
        for cv in BELL_CVS:
            beta = bell_row(alpha, cv)
            if beta is None:
                continue
            cols, p, sd = one_row(alpha, beta)
            curve = eltq.exceedance_curve(*cols, min_loss=p * 1e-3, nodes_per_sigma=nodes_per_sigma)
            fine_nodes += curve.meta.fine_nodes
            loss = np.asarray(curve.loss)
            xs = np.concatenate([np.sqrt(loss[:-1] * loss[1:]), p + np.linspace(-3.0, 7.0, 201) * sd])
            xs = xs[(xs > loss[0]) & (xs < 1.0)]
            got = np.asarray(curve.exceedance(xs))
            exact = special.betaincc(alpha, beta, xs)
            worst = np.fmax(worst, np.nan_to_num(band_errors(got, exact, 1.0), nan=0.0))
    return worst, fine_nodes


def bell_section(markdown: bool) -> None:
    print(f"Bell-shaped events by nodes_per_sigma: alpha in {BELL_ALPHAS}, coefficients of variation in {BELL_CVS}; "
          "the curve against SciPy by band of the event's survival.\n")
    rows = []
    base, _ = bell_errors(1)
    rows.append(["base only"] + [fmt(w) for w in base] + ["0"])
    for t in SIGMAS:
        worst, nodes = bell_errors(t)
        rows.append([str(t)] + [fmt(w) for w in worst] + [f"{nodes:,}"])
    table(["nodes_per_sigma"] + BAND_HEADERS + ["fine nodes"], rows, markdown)


# ---------------------------------------------------------------- tails

TAIL_ALPHAS = (1e2, 1e4, 1e6)
TAIL_BETAS = (0.5, 1.0, 2.0, 5.0, 20.0)


def tails_section(markdown: bool) -> None:
    print(f"Endpoint tails: rate-one events with alpha in {TAIL_ALPHAS} and beta in {TAIL_BETAS}, the curve at "
          "distances M - x from 1e-8 M to 0.1 M against SciPy by band of the event's survival; "
          "the survival method on both backends (the quantiles method has no curve).\n")
    worst = {"engine": np.zeros(len(BANDS)), "reference": np.zeros(len(BANDS))}
    agreement = 0.0
    xs = 1.0 - np.logspace(-8, -1, 141)
    for alpha in TAIL_ALPHAS:
        for beta in TAIL_BETAS:
            cols, p, _ = one_row(alpha, beta)
            exact = special.betaincc(alpha, beta, xs)
            got = {}
            for name, build in (("engine", eltq.exceedance_curve),
                                ("reference", lambda *c, **k: reference.exceedance_curve_reference(*c, **k)[0])):
                curve = build(*cols, min_loss=p * 1e-2)
                got[name] = np.asarray(curve.exceedance(xs), dtype=np.float64)
                worst[name] = np.fmax(worst[name], np.nan_to_num(band_errors(got[name], exact, 1.0), nan=0.0))
            keep = exact >= 1e-12
            agreement = max(agreement, float(np.max(np.abs(got["engine"][keep] - got["reference"][keep]) / exact[keep])))
    rows = [[f"survival, {name}"] + [fmt(w) for w in worst[name]] for name in ("engine", "reference")]
    table(["backend"] + BAND_HEADERS, rows, markdown)
    print(f"The two backends agree within {agreement:.1e} relative where the survival is at least 1e-12.\n")


# ---------------------------------------------------------------- large

LARGE_SMALL = (2e5, 1e6, 1e9, 1e16, 1e20)
LARGE_RATIOS = (1.0, 20.0)


def large_section(markdown: bool) -> None:
    print("Large parameters: one-row portfolios with exposed value 1, min(alpha, beta) as shown, on a grid of a "
          "quarter standard deviation (at least 1e-6) covering the mean plus and minus 12; the largest bucket-mass error as a share "
          "of the event's rate against the exact law (SciPy, the normal from 1e16).\n")
    rows = []
    for small in LARGE_SMALL:
        for ratio in LARGE_RATIOS:
            a, b = small, small * ratio
            cols, mu, sd = one_row(a, b)
            h = max(sd / 4.0, 1e-6)
            n = int(math.ceil((mu + 12.0 * sd) / h)) + 2
            law = reference._law(a, b)
            edges = (np.arange(n - 1) + 0.5) * h
            survival = law.sf(np.minimum(edges, 1.0))
            exact_mass = np.diff(np.concatenate([[1.0], survival, [0.0]])) * -1.0
            cells = [f"{small:.0e}", f"{ratio:g}"]
            for label, run, extra in (("survival, engine", eltq.quantize_arrays, {"method": "survival"}),
                                      ("survival, reference", reference.quantize_reference, {"method": "survival"}),
                                      ("quantiles K = 32, engine", eltq.quantize_arrays, {"method": "quantiles", "quadrature_points": K})):
                _, sec, _ = run(*cols, bucket_size=h, grid_length=n, **extra)
                cells.append(fmt(float(np.max(np.abs(sec - exact_mass)))))
            rows.append(cells)
    table(["min(alpha, beta)", "beta / alpha", "survival, engine", "survival, reference", "quantiles K = 32, engine"],
          rows, markdown)


# ---------------------------------------------------------------- mixtures

def fixture_columns(name: str):
    df = pd.read_csv(FIXTURES / f"{name}_elt.csv", float_precision="round_trip")
    return tuple(df[c].to_numpy(dtype=np.float64) for c in ("rate", "max_loss", "mean_loss", "sd_indep", "sd_corr"))


def timed_median(fn, repeats: int = 3) -> float:
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def mixtures_section(markdown: bool, rows_for_timing: int) -> None:
    print("Mixtures: the bucketed tail sums of the beta events against their exact exceedance at every bucket edge, by band of the "
          "exceedance as a share of the total rate; the engine under both methods and the reference under the "
          f"survival method, with the engine's time on {rows_for_timing:,} rows of the benchmark recipe at every core.\n")
    portfolios = [
        ("medium fixture, h = 5,000", fixture_columns("medium"), 5000.0, None),
        ("small-alpha fixture, h = 100", fixture_columns("small_alpha"), 100.0, None),
        ("mixture portfolio, 20,000-row sample, h = 5,000", cols_tuple(synthetic_elt(20_000)), 5000.0,
         cols_tuple(synthetic_elt(rows_for_timing))),
        ("small-alpha portfolio, 5,000-row sample, h = 100", cols_tuple(small_alpha_elt(5_000)), 100.0,
         cols_tuple(small_alpha_elt(rows_for_timing))),
    ]
    rows = []
    for label, cols, h, full in portfolios:
        ref_curve, _ = reference.exceedance_curve_reference(*cols, min_loss=0.5 * h)
        grids = {}
        times = {}
        for method in ("survival", "quantiles"):
            extra = {"quadrature_points": K} if method == "quantiles" else {}
            _, grids[method], meta = eltq.quantize_arrays(*cols, bucket_size=h, method=method, **extra)
            if full is not None:
                times[method] = timed_median(lambda: eltq.quantize_arrays(*full, bucket_size=h, method=method, **extra))
        _, ref_grid, _ = reference.quantize_reference(*cols, bucket_size=h, method="survival")
        total = meta.total_rate
        n = grids["survival"].size
        edges = (np.arange(n - 1) + 0.5) * h
        exact = ref_curve.exact(edges)
        # The atoms (degenerate and Bernoulli rows) go on the grid by the
        # linear rule, which splits one across two nodes, where the exact
        # exceedance counts it whole; they are the same in every grid and
        # are set aside before the comparison.
        atoms = np.zeros(n)
        for x, w in zip(ref_curve.atom_loss, ref_curve.atom_rate):
            reference._deposit(atoms, float(x), float(w), 1.0 / h)
        for method, grid in grids.items():
            name = "survival" if method == "survival" else f"quantiles K = {K}"
            cells = [label, f"{name}, engine"] + [fmt(w) for w in band_errors(tail_sums(grid - atoms), exact, total)]
            cells.append(f"{times[method]:.2f}" if method in times else "")
            rows.append(cells)
        agree = float(np.max(np.abs(grids["survival"] - ref_grid)) / total)
        cells = [label, "survival, reference"] + [fmt(w) for w in band_errors(tail_sums(ref_grid - atoms), exact, total)]
        cells.append(f"agrees within {agree:.0e}")
        rows.append(cells)
    table(["portfolio", "method, backend"] + BAND_HEADERS + ["time, s"], rows, markdown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["bell", "tails", "large", "mixtures", "all"], default="all")
    parser.add_argument("--rows", type=int, default=1_000_000, help="rows for the mixtures' timings")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    sections = {"bell": lambda: bell_section(args.markdown),
                "tails": lambda: tails_section(args.markdown),
                "large": lambda: large_section(args.markdown),
                "mixtures": lambda: mixtures_section(args.markdown, args.rows)}
    for name, run in sections.items():
        if args.section in ("all", name):
            run()
    print(f"eltq {eltq.__version__}, backend {eltq.BACKEND}.")


if __name__ == "__main__":
    main()
