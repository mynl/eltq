"""Shared-factor rule convergence study (1.7.0; extended at 2.6.0 by
`dev/done/plan-multiunit-coupling.md` [z-rule]).

Run from the repository root:

    uv run python benchmarks/z_convergence.py > benchmarks/z-convergence-<version>.md

For the three rules (midpoint quantile nodes, Gauss-Hermite, stratified) and
Q in 8, 16, 32, 64, 128, reports against exact references:

- two units: the bivariate normal rectangle probabilities of the edge scores
  (`exact_two_unit_joint`), exact for beta and two-point units at any
  loading, with no quadrature over the factor;
- one unit: the direct bucketing of its marginal;
- more than two units (the fixture): the stratified rule at Q = 4096, whose
  two-unit error against the exact reference is recorded beside it.

Metrics per case and rule: the relative error of the event total's
exceedance probability at the losses where the exact exceedance is 1e-2,
1e-3, 1e-4 and 1e-5; the relative errors of the total's mean, sd and 99th
percentile; the largest absolute marginal error of any unit. Cases: A to D
of the multi-unit article (`tbl-cases`), the three events of Homer and Li
2017, Table 11 with their two-point rows (`tbl-homer`), the fixture, a lone
beta unit at loadings 0.5 to 0.99, and units at loading one (a beta and a
two-point row) beside a partner with w < 1. The study sets the default rule
and Q (spec section 4A, requirement (ii)); the reading at the end records
the choice.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eltq
from eltq.reference_multiunit import (
    Unit, exact_two_unit_joint, marginal_pmf, multiunit_views_reference, prepare_unit,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

H = 10.0
RULES = ("midpoint", "gauss_hermite", "stratified")
Q_VALUES = (8, 16, 32, 64, 128)
LEVELS = (1e-2, 1e-3, 1e-4, 1e-5)

A1, A2 = Unit(200, 100, 150, 1000), Unit(500, 300, 200, 2000)
CASES = {
    "A balanced": (A1, A2),
    "B correlation-dominant": (Unit(200, 30, 220, 1000), Unit(500, 60, 440, 2000)),
    "C independent": (Unit(200, 250, 0, 1000), Unit(500, 500, 0, 2000)),
    "D comonotone, common shape": (Unit(200, 0, 250, 1000), Unit(400, 0, 500, 2000)),
    "Homer event 1 (C two-point)": (Unit(300, 400, 300, 3000), Unit(200, 300, 200, 1000)),
    "Homer event 2 (P two-point)": (Unit(100, 371, 267, 1000), Unit(200, 150, 533, 4000)),
    "Homer event 3": (Unit(100, 224, 200, 2000), Unit(100, 200, 200, 2000)),
    "beta at loading one beside A2": (Unit(200, 0, 250, 1000), A2),
    "two-point at loading one beside A2": (Unit(30, 0, 80, 100), A2),
}


def grid_length(unit):
    return int(math.ceil(unit.max_loss / H)) + 1


def total_of(pmf):
    n1, n2 = pmf.shape
    idx = np.add.outer(np.arange(n1), np.arange(n2)).ravel()
    return np.bincount(idx, weights=pmf.ravel(), minlength=n1 + n2 - 1)


def tail_errors(got, exact):
    """Relative error of the exceedance probability at the losses where the
    exact exceedance is nearest each level (NaN where the exact total never
    reaches the level)."""
    s_exact = 1.0 - np.cumsum(exact)
    s_got = 1.0 - np.cumsum(got)
    out = []
    for level in LEVELS:
        candidates = np.nonzero(s_exact >= level)[0]
        if candidates.size == 0 or s_exact[candidates[-1]] > 10 * level:
            out.append(math.nan)
            continue
        k = candidates[-1]
        out.append((s_got[k] - s_exact[k]) / s_exact[k])
    return out


def moments(ps):
    xs = H * np.arange(ps.size)
    mean = ps @ xs
    sd = math.sqrt(max(ps @ xs**2 - mean * mean, 0.0))
    cdf = np.cumsum(ps)
    q99 = xs[min(int(np.searchsorted(cdf, 0.99 * cdf[-1])), xs.size - 1)]
    return mean, sd, q99


def fmt(x):
    return "nan" if x != x else f"{x:+.1e}"


def two_unit_study():
    print("## Two units against the bivariate normal rectangles\n")
    print("Units as (mu, sd_indep, sd_corr, M); bucket 10. Columns: relative error of the event "
          "total's exceedance probability at the exact 1e-2, 1e-3, 1e-4 and 1e-5 losses; relative "
          "errors of its mean, sd and q99; the largest absolute marginal error of either unit.\n")
    summary = []
    for name, (u1, u2) in CASES.items():
        n1, n2 = grid_length(u1), grid_length(u2)
        exact = exact_two_unit_joint([u1, u2], H, n1, n2)
        exact_total = total_of(exact)
        direct = [marginal_pmf(prepare_unit(u, "u"), H, n) for u, n in ((u1, n1), (u2, n2))]
        e_mean, e_sd, e_q = moments(exact_total)
        print(f"### {name}: {tuple(u1)} and {tuple(u2)}\n")
        print(f"Exact total: mean {e_mean:.4f}, sd {e_sd:.4f}, q99 {e_q:.0f}; normal correlation "
              f"{prepare_unit(u1, 'u').w_corr * prepare_unit(u2, 'u').w_corr:.3f}.\n")
        print("| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for rule in RULES:
            for q in Q_VALUES:
                pmf, _, _, _ = eltq.joint_two_units(u1, u2, bucket_size=H, n1=n1, n2=n2,
                                                     z_nodes=q, z_rule=rule)
                tot = total_of(pmf)
                tails = tail_errors(tot, exact_total)
                mean, sd, q99 = moments(tot)
                marg = max(np.abs(pmf.sum(axis=1) - direct[0]).max(),
                           np.abs(pmf.sum(axis=0) - direct[1]).max())
                row = [name, rule, q, *tails, (mean - e_mean) / e_mean, (sd - e_sd) / e_sd,
                       (q99 - e_q) / e_q, marg]
                summary.append(row)
                print(f"| {rule} | {q} | " + " | ".join(fmt(x) for x in tails)
                      + f" | {fmt(row[7])} | {fmt(row[8])} | {fmt(row[9])} | {marg:.1e} |")
        print()
    return pd.DataFrame(summary, columns=["case", "rule", "Q", "s2", "s3", "s4", "s5", "mean", "sd",
                                          "q99", "marginal"])


def lone_unit_study():
    print("## A lone beta unit against its direct bucketing\n")
    print("A one-unit event (200, sd_indep, sd_corr, 1000) at total sd 250 and loading w; the "
          "mixture over the rule should return the marginal. Largest absolute bucket error and "
          "relative exceedance error at the 1e-3 and 1e-5 losses.\n")
    print("| w | rule | Q | marginal | S 1e-3 | S 1e-5 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for w in (0.5, 0.8, 0.9, 0.99):
        sd_c = 250.0 * w / (w + math.sqrt(1 - w * w))  # sd_indep + sd_corr = 250 at loading w
        unit = Unit(200.0, 250.0 - sd_c, sd_c, 1000.0)
        n = grid_length(unit)
        direct = marginal_pmf(prepare_unit(unit, "u"), H, n)
        for rule in RULES:
            for q in (8, 32, 128):
                views, meta = eltq.multiunit_views(["e"], ["a"], [1.0], [unit.mean_loss],
                                                   [unit.sd_indep], [unit.sd_corr],
                                                   [unit.max_loss], bucket_size=H,
                                                   views=("ground_up",), z_nodes=q, z_rule=rule)
                got = views["ground_up"]
                tails = tail_errors(got, direct)
                print(f"| {w} | {rule} | {q} | {np.abs(got - direct).max():.1e} | "
                      f"{fmt(tails[1])} | {fmt(tails[3])} |")
    print()


def fixture_study():
    from conftest import load_multi_fixture  # noqa: E402

    df, terms = load_multi_fixture()
    print("## The fixture against the stratified rule at Q = 4096\n")
    print("`fixtures/multiunit_small_elt.csv` with its terms, all four views; the reference is "
          "the stratified rule at Q = 4096 (its two-unit error against the exact rectangles is "
          "in the tables above). Columns: the ground-up total's relative exceedance error at the "
          "1e-2 to 1e-5 fractions of the total rate, the largest relative view-mean error, and "
          "the engine's wall time for the four views.\n")
    ref, ref_meta = multiunit_views_reference(*(df[c] for c in df.columns), terms=terms,
                                              bucket_size=H, z_nodes=4096, z_rule="stratified")
    ref_total = ref["ground_up"] / ref_meta.total_rate
    print("| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | view mean | time (ms) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rule in RULES:
        for q in Q_VALUES:
            start = time.perf_counter()
            views, meta = eltq.multiunit_views_dataframe(df, terms=terms, bucket_size=H,
                                                         z_nodes=q, z_rule=rule)
            elapsed = (time.perf_counter() - start) * 1e3
            tails = tail_errors(views["ground_up"] / meta.total_rate, ref_total)
            mean_err = max(abs(meta.view_mean_error[v]) / abs(meta.analytic_view_mean[v])
                           for v in meta.views)
            print(f"| {rule} | {q} | " + " | ".join(fmt(x) for x in tails)
                  + f" | {mean_err:.1e} | {elapsed:.1f} |")
    print()


def main() -> None:
    print(f"# Shared-factor rule convergence ({eltq.__version__}, backend {eltq.BACKEND})\n")
    summary = two_unit_study()
    lone_unit_study()
    fixture_study()
    print("## Worst case over the two-unit table by rule and Q\n")
    print("Largest absolute relative exceedance error over the four levels and the nine cases, "
          "and the largest marginal error.\n")
    print("| rule | Q | worst tail | worst marginal |")
    print("| --- | --- | --- | --- |")
    for rule in RULES:
        for q in Q_VALUES:
            sub = summary[(summary.rule == rule) & (summary.Q == q)]
            worst = np.nanmax(np.abs(sub[["s2", "s3", "s4", "s5"]].to_numpy(dtype=float)))
            print(f"| {rule} | {q} | {worst:.1e} | {sub.marginal.max():.1e} |")
    print()


if __name__ == "__main__":
    main()
