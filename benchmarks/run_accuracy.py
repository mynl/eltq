"""Accuracy of the survival method's curve on one-row portfolios, by regime.

The speed suite (``run_end_to_end.py``) compares times; this suite reports
the error the time buys, so that a speed comparison is read at a stated
accuracy (``dev/done/plan-peer-review-response.md``, [mesh] and [benchmark]).

Bell-shaped events (both beta parameters above 1) are the regime the shared
cubic cannot follow on the base mesh alone: its relative error in an event's
tail falls as about the 2.7th power of the event's width in node intervals.
The fine mesh gives each such event ``nodes_per_sigma`` nodes per standard
deviation around its mean; this script measures what each value of that knob
buys. For each event the curve is evaluated at every fine-interval midpoint
and at ``mu + z sigma`` for ``z`` from -3 to 7 against SciPy's regularized
incomplete beta, and the largest relative error is reported by band of the
event's survival.

Usage::

    uv run python benchmarks/run_accuracy.py            # the table to stdout
    uv run python benchmarks/run_accuracy.py --markdown # as a Markdown table

The output is recorded in ``benchmarks/report-<version>.md``.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy import special

import eltq

BANDS = ((1e-2, 1.0), (1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6), (1e-12, 1e-8))
ALPHAS = (1.5, 3.0, 10.0, 1e2, 1e3, 1e4)
CVS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1)
SIGMAS = (8, 12, 16, 20)


def bell_row(alpha: float, cv: float):
    """A rate-one event on [0, 1] with shape ``alpha`` and coefficient of
    variation ``cv``; None when no beta with ``beta > 1`` has them."""
    if cv * cv * alpha >= 1.0:
        return None
    beta = cv * cv * alpha * (alpha + 1.0) / (1.0 - cv * cv * alpha)
    if beta <= 1.0:
        return None
    p = alpha / (alpha + beta)
    sd = math.sqrt(p * (1.0 - p) / (alpha + beta + 1.0))
    cols = tuple(np.array([v]) for v in (1.0, 1.0, p, sd, 0.0))
    return cols, beta, p, sd


def bell_errors(nodes_per_sigma: int):
    """Largest relative error by survival band over the bell sweep, and the
    fine node count summed over the sweep."""
    worst = np.zeros(len(BANDS))
    fine_nodes = 0
    for alpha in ALPHAS:
        for cv in CVS:
            made = bell_row(alpha, cv)
            if made is None:
                continue
            cols, beta, p, sd = made
            curve = eltq.exceedance_curve(*cols, min_loss=p * 1e-3, nodes_per_sigma=nodes_per_sigma)
            fine_nodes += curve.meta.fine_nodes
            loss = np.asarray(curve.loss)
            xs = np.concatenate([np.sqrt(loss[:-1] * loss[1:]), p + np.linspace(-3.0, 7.0, 201) * sd])
            xs = xs[(xs > loss[0]) & (xs < 1.0)]
            got = np.asarray(curve.exceedance(xs))
            exact = special.betaincc(alpha, beta, xs)
            for i, (lo, hi) in enumerate(BANDS):
                keep = (exact >= lo) & (exact < hi)
                if keep.any():
                    worst[i] = max(worst[i], np.max(np.abs(got[keep] / exact[keep] - 1.0)))
    return worst, fine_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    header = ["nodes_per_sigma"] + [f"[{lo:.0e},{hi:.0e})" for lo, hi in BANDS] + ["fine nodes"]
    rows = []
    for t in SIGMAS:
        worst, nodes = bell_errors(t)
        rows.append([str(t)] + [f"{w:.1e}" for w in worst] + [str(nodes)])
    base, _ = bell_errors(1)   # the base mesh alone: no event asks for density
    rows.insert(0, ["base only"] + [f"{w:.1e}" for w in base] + ["0"])
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---:" for _ in header) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
        for r in rows:
            print("  ".join(c.rjust(w) for c, w in zip(r, widths)))
    print(f"\neltq {eltq.__version__}; largest relative error of the curve against SciPy over "
          f"alpha in {ALPHAS} and coefficients of variation in {CVS}, by band of the event's survival.")


if __name__ == "__main__":
    main()
