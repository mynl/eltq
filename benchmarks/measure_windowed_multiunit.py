"""Measurement ahead of [Eltq-Windowed-Shared-Events] (2026-10-27).

Run from the repository root (release wheel installed):

    uv run python benchmarks/measure_windowed_multiunit.py [--probe 16 17 18 19]

The plan (`dev/done/plan-model-moments-and-windowed-views.md`) asks, before
the windowed multi-unit design is committed, for the cost of the 0.91M-row
portfolio of ``run_multiunit.py`` with exposures raised to 1e9 at windows of
2^20 and 2^22. Two parts:

- the probe: today's ``multiunit_views`` (ground_up) on one ten-unit event of
  the raised portfolio, one thread, at bucket sizes ``1e10 / 2^k`` so that
  the event's full support is ``2^k`` buckets; seconds and peak working set
  per run. Memory grows by about 13.4 KiB per bucket: 2^20 needs 14 GiB, and
  2^22 would need about 57 GiB, so the default stops at 2^19. Each size runs
  in its own process, so that each peak is its own.
- the projection: each unit row's span under the rate floor of the one-row
  route (``TAIL_RATE_FLOOR`` = 1e-20 of the total rate, spec section 5A),
  the event lengths after that cut, and the one-thread seconds of today's
  route and of the design, both at the probe's measured cost per bucket of a
  ten-unit event's grid.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

RAISED_MAX_LOSS = 1e9
SECONDS_PER_BUCKET = 1.21e-5   # the probe's slope from 2^19 to 2^20 (one event, ten units, one thread)


def raised_portfolio():
    from run_multiunit import synthetic_portfolio

    cols = synthetic_portfolio(100_000, 10)
    raised = cols["mean_loss"] <= cols["max_loss"]   # dropped rows stay dropped
    cols["max_loss"] = np.where(raised, RAISED_MAX_LOSS, cols["max_loss"])
    return cols


def probe(k: int) -> None:
    """One ten-unit event at full support 2^k, one thread."""
    import eltq
    from run_end_to_end import peak_working_set_mb

    cols = raised_portfolio()
    event = np.asarray(cols["event_id"])
    first = np.flatnonzero(np.bincount(event) == 10)[0]
    rows = event == first
    sub = {name: np.asarray(values)[rows] for name, values in cols.items()}
    n = 2 ** k
    start = time.perf_counter()
    eltq.multiunit_views(sub["event_id"], sub["unit_id"], sub["rate"], sub["mean_loss"], sub["sd_indep"],
                         sub["sd_corr"], sub["max_loss"], bucket_size=10 * RAISED_MAX_LOSS / n,
                         views=("ground_up",), threads=1)
    seconds = time.perf_counter() - start
    print(f"probe 2^{k}: {seconds:.2f} s, peak {peak_working_set_mb():.0f} MiB")


def projection() -> None:
    from scipy import stats

    cols = raised_portfolio()
    mu, m = cols["mean_loss"], cols["max_loss"]
    live = (mu <= m) & (cols["rate"] > 0) & (mu > 0)
    event = np.asarray(cols["event_id"])[live]
    rate, mu = cols["rate"][live], mu[live]
    sd = (cols["sd_indep"] + cols["sd_corr"])[live]   # Burke's total sd
    event_rate = np.zeros(event.max() + 1)
    event_rate[event] = rate
    floor = 1e-20 * event_rate.sum()
    x_floor = mu.copy()                               # an atom's span ends at its position
    beta = sd > 0
    s = mu[beta] * (RAISED_MAX_LOSS - mu[beta]) / sd[beta] ** 2 - 1.0
    a, b = s * mu[beta] / RAISED_MAX_LOSS, s * (1.0 - mu[beta] / RAISED_MAX_LOSS)
    x_floor[beta] = RAISED_MAX_LOSS * stats.beta.isf(np.minimum(floor / rate[beta], 1.0), a, b)
    multi = np.bincount(event) >= 2
    print(f"rate floor {floor:.2e}; span ends, quantiles 0.5 0.9 0.99 1: "
          f"{np.quantile(x_floor, [0.5, 0.9, 0.99, 1.0]).round(-3)}")
    for n in (2 ** 20, 2 ** 22):
        for label, h in (("full support = window", 10 * RAISED_MAX_LOSS / n),
                         ("unit support = window", RAISED_MAX_LOSS / n)):
            full = np.ceil(RAISED_MAX_LOSS / h) + 1
            today = np.minimum(n, np.bincount(event, weights=np.full(event.size, full)))[multi]
            cut = np.minimum(n, np.bincount(event, weights=np.minimum(full, np.ceil(x_floor / h) + 1)))[multi]
            print(f"2^{int(np.log2(n))}, h {h:.0f} ({label}): event length median {np.median(today):.3g} today, "
                  f"{np.median(cut):.3g} after the cut (max {cut.max():.3g}); one-thread seconds "
                  f"{SECONDS_PER_BUCKET * today.sum():.3g} today, {SECONDS_PER_BUCKET * cut.sum():.3g} after the cut")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=int, nargs="*", default=[16, 17, 18, 19],
                        help="grid sizes 2^k to probe, each in its own process (2^20 needs 14 GiB)")
    parser.add_argument("--one", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one is not None:
        probe(args.one)
        return
    for k in args.probe:
        subprocess.run([sys.executable, __file__, "--one", str(k)], check=True)
    projection()


if __name__ == "__main__":
    main()
