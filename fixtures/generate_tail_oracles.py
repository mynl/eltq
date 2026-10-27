"""Exact tails of two-unit event totals, the oracles of spec test 38.

Run from the repository root:

    uv run python fixtures/generate_tail_oracles.py

For two beta units under a Gaussian factor at normal-score correlation
``rho = w_u w_w``, conditioning on the first unit's score ``s`` leaves the
second unit's score normal with mean ``rho s`` and variance ``1 - rho^2``,
so (`dev/done/plan-model-moments-and-windowed-views.md`, test 38)

    P(S^r; S > t) = int phi(s) sum_k C(r, k) x(s)^(r-k) P(X_w^k; X_w > t - x(s) | s) ds,

``r = 0..3``, with ``x(s) = M_u F_u^-1(Phi(s))``. Both integrals are taken
on dense uniform grids of the scores (Simpson's rule outside, the trapezoid
rule inside), the quantile functions tabulated once by SciPy's inverses,
the inner partial moments by cumulative sums from the top with the
conditional normal weights, the threshold's own score exact and its partial
cell by interpolation; printed at three grid densities so that their
agreement shows the accuracy. Nothing here is shared with the union bound the engine
evaluates; the test file embeds the printed values.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

from eltq import reference_multiunit as rm


def prep(unit):
    return rm.couple(rm.prepare_unit(rm.Unit(*unit), "u"), "burke")


def quantile(law, v):
    """``F^-1(Phi(v))``, by the inverse survival function above the median
    so that the upper tail does not round to the largest loss."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.0, law.ppf(special.ndtr(np.minimum(v, 0.0))),
                    law.isf(special.ndtr(-np.maximum(v, 0.0))))


def exact_tails(u, w, edges, n_s=4001, n_v=40001, reach=9.5):
    """``P(S^r; S > t)`` for r = 0..3 at each edge."""
    pu, pw = prep(u), prep(w)
    rho = pu.w_corr * pw.w_corr
    sig = math.sqrt((1.0 - rho) * (1.0 + rho))
    law_w = stats.beta(pw.alpha, pw.beta)
    s = np.linspace(-reach, reach, n_s)
    v = np.linspace(-reach - 3.0, reach + 3.0, n_v)
    dv = v[1] - v[0]
    x = pu.max_loss * quantile(stats.beta(pu.alpha, pu.beta), s)
    g = pw.max_loss * quantile(law_w, v)
    # Simpson weights on the outer grid (n_s odd).
    ws = np.full(n_s, 2.0)
    ws[1::2] = 4.0
    ws[[0, -1]] = 1.0
    ws *= (s[1] - s[0]) / 3.0 * np.exp(-0.5 * s * s) / math.sqrt(2 * math.pi)
    powers = np.stack([g ** k for k in range(1, 4)])

    def score(y):
        if y <= 0.0:
            return -math.inf
        if y >= 1.0:
            return math.inf
        sf = law_w.sf(y)
        return -special.ndtri(sf) if sf < 0.5 else special.ndtri(law_w.cdf(y))

    out = np.zeros((len(edges), 4))
    for i in range(n_s):
        dens = np.exp(-0.5 * ((v - rho * s[i]) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
        f = powers * dens[None, :]
        # Trapezoid cumulative from the top: above[:, j] = int_{v_j}^{inf} f.
        cells = 0.5 * (f[:, 1:] + f[:, :-1]) * dv
        above = np.zeros((3, n_v))
        above[:, :-1] = np.cumsum(cells[:, ::-1], axis=1)[:, ::-1]
        for e, t in enumerate(edges):
            va = score((t - x[i]) / pw.max_loss)
            if va == math.inf:
                continue
            tail = np.zeros(4)
            tail[0] = special.ndtr(-(va - rho * s[i]) / sig)
            if va <= v[0]:
                tail[1:] = above[:, 0]
            elif va < v[-1]:
                j = int((va - v[0]) // dv)
                frac = (va - v[j]) / dv
                tail[1:] = above[:, j + 1] + (1.0 - frac) * cells[:, j]
            for r in range(4):
                out[e, r] += ws[i] * sum(math.comb(r, k) * x[i] ** (r - k) * tail[k] for k in range(r + 1))
    return out


BETAS = [(200.0, 100.0, 150.0, 1000.0), (500.0, 300.0, 200.0, 2000.0)]
REGRESSION = [(100.0, 5000.0, 5000.0, 1e9), (100.0, 5000.0, 5000.0, 1e9)]


def main():
    np.set_printoptions(precision=12)
    for label, units, edges in (("two betas", BETAS, [1000.0, 2000.0, 2900.0]),
                                ("two regression rows", REGRESSION, [4e5, 4e6, 4e7])):
        for n_s, n_v in ((2001, 20001), (4001, 40001), (8001, 80001)):
            print(label, n_s, n_v)
            print(exact_tails(*units, edges, n_s, n_v))


if __name__ == "__main__":
    main()
