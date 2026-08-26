"""Shared-factor quadrature convergence study (plan decision 5, 1.7.0).

Run from the repository root:

    uv run python benchmarks/z_convergence.py > benchmarks/z-convergence-<version>.md

For the four regimes of the additivity note (balanced, correlation-dominant,
independent, comonotone) and both z rules (midpoint quantile nodes,
Gauss-Hermite), reports against a Q = 4001 midpoint reference:

- the maximum absolute error of the joint's marginals against direct beta
  bucketing (the exact target: the z mixture of the conditionals is the
  marginal);
- the relative error of the sum distribution's mean, sd and 99th percentile;
- the maximum absolute cell difference of the sum distribution.

The study validates the default (midpoint, Q = 32) rather than picks it: see
the reading at the bottom of the generated report.
"""

from __future__ import annotations

import math

import numpy as np

from eltq.reference_multiunit import Unit, marginal_pmf, prepare_unit, secondary_two_units

CASES = {
    "A balanced": (Unit(200, 100, 150, 1000), Unit(500, 300, 200, 2000)),
    "B correlation-dominant": (Unit(200, 30, 220, 1000), Unit(500, 60, 440, 2000)),
    "C independent": (Unit(200, 250, 0, 1000), Unit(500, 500, 0, 2000)),
    "D comonotone": (Unit(200, 0, 250, 1000), Unit(400, 0, 500, 2000)),
}
BS, N1, N2 = 10.0, 101, 201
Q_VALUES = (8, 16, 32, 64, 128)


def stats(result):
    xs, ps = result.sum_distribution()
    mean = ps @ xs
    sd = math.sqrt(max(ps @ xs**2 - mean * mean, 0.0))
    cdf = np.cumsum(ps)
    return mean, sd, xs[min(int(np.searchsorted(cdf, 0.99 * cdf[-1])), xs.size - 1)], ps


def main() -> None:
    print("# Shared-factor quadrature convergence\n")
    print("Reference: midpoint rule at Q = 4001. Units (mu, sd_indep, sd_corr, M); "
          f"bucket {BS}, grids {N1} x {N2}.\n")
    for name, (u1, u2) in CASES.items():
        direct = [marginal_pmf(prepare_unit(u, "u"), BS, n) for u, n in ((u1, N1), (u2, N2))]
        ref = secondary_two_units([u1, u2], BS, N1, N2, n_z=4001)
        r_mean, r_sd, r_q, r_ps = stats(ref)
        print(f"## {name}: {tuple(u1)} and {tuple(u2)}\n")
        print(f"Reference sum: mean {r_mean:.4f}, sd {r_sd:.4f}, q99 {r_q:.1f}\n")
        print("| rule | Q | max marginal err | mean rel | sd rel | q99 rel | max sum cell diff |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for rule in ("midpoint", "gauss_hermite"):
            for q in Q_VALUES:
                res = secondary_two_units([u1, u2], BS, N1, N2, n_z=q, z_rule=rule)
                marg = max(np.abs(res.marginal(1) - direct[0]).max(),
                           np.abs(res.marginal(2) - direct[1]).max())
                mean, sd, q99, ps = stats(res)
                print(f"| {rule} | {q} | {marg:.1e} | {(mean - r_mean) / r_mean:+.1e} | "
                      f"{(sd - r_sd) / r_sd:+.1e} | {(q99 - r_q) / r_q:+.1e} | "
                      f"{np.abs(ps - r_ps).max():.1e} |")
        print()
    print("## Reading\n")
    print("Gauss-Hermite is spectrally accurate where the conditional cdf is smooth in z "
          "(balanced regime: marginal error 1e-10 at Q = 32 against 1e-3 for midpoint) but "
          "loses that advantage as the conditional approaches a step in z "
          "(correlation-dominant and comonotone units), where its clustered nodes leave "
          "visible artifacts in the pmf while the equal-weight midpoint rule degrades "
          "gracefully and keeps the better distribution shape. Neither rule dominates; "
          "the midpoint rule is the robust default for pmf shape at Q = 32, Gauss-Hermite "
          "the option for moment accuracy on smooth portfolios.")


if __name__ == "__main__":
    main()
