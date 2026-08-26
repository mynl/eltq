"""Generate the synthetic two-unit ELT samples used by ``dev/demo-bivariate.qmd``.

Run from the repository root::

    uv run python dev/features/make_two_unit_elt.py

Fully deterministic (fixed seed). Every number is invented; no real data ever
enters the repository (see the data firewall in CLAUDE.md).

Outputs (canonical multi-unit columns, so the same files feed both
``eltq.joint_two_units`` and ``eltq.multiunit_views``)::

    dev/features/two_unit_elt_10.csv     10 handcrafted events x 2 units
    dev/features/two_unit_elt_1000.csv   1000 seeded events x 2 units

Two units share every event: ``coastal`` (max loss 1200) and ``inland``
(max loss 800), the two sites of one notional operator hit by the same storm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
COLUMNS = ["event_id", "unit_id", "rate", "mean_loss", "sd_indep", "sd_corr", "max_loss"]
SEED = 20260826
MAX_LOSS = {"coastal": 1200.0, "inland": 800.0}

# event_id, rate, (mu, sd_indep, sd_corr) coastal, (mu, sd_indep, sd_corr) inland
HANDCRAFTED = [
    # Typical storm: both sites load on the shared factor.
    ("ev01", 0.020, (240.0, 90.0, 150.0), (160.0, 70.0, 90.0)),
    # Large but mostly site-specific uncertainty.
    ("ev02", 0.015, (600.0, 200.0, 100.0), (400.0, 150.0, 60.0)),
    # No shared factor at all: conditionally independent given the event.
    ("ev03", 0.030, (120.0, 60.0, 0.0), (80.0, 40.0, 0.0)),
    # Pure shared factor: the two sites move together (comonotone).
    ("ev04", 0.008, (500.0, 0.0, 300.0), (300.0, 0.0, 180.0)),
    # No secondary uncertainty at all: atoms at the mean, with or without.
    ("ev05", 0.012, (300.0, 0.0, 0.0), (200.0, 0.0, 0.0)),
    # Severe, both sites close to their caps.
    ("ev06", 0.005, (900.0, 150.0, 150.0), (640.0, 80.0, 80.0)),
    # Small mean, heavy relative spread (alpha < 1, strongly skewed).
    ("ev07", 0.025, (60.0, 100.0, 80.0), (40.0, 60.0, 50.0)),
    # Asymmetric loading: coastal mostly shared, inland mostly its own.
    ("ev08", 0.010, (400.0, 120.0, 200.0), (100.0, 30.0, 20.0)),
    # Near the maximum loss on both sites.
    ("ev09", 0.004, (1000.0, 100.0, 60.0), (700.0, 60.0, 40.0)),
    # Wide spread relative to the mean on a low mean.
    ("ev10", 0.018, (200.0, 250.0, 150.0), (150.0, 180.0, 100.0)),
]


def _rows(event_id, rate, coastal, inland):
    for unit, (mu, sd_i, sd_c) in (("coastal", coastal), ("inland", inland)):
        yield (event_id, unit, rate, mu, sd_i, sd_c, MAX_LOSS[unit])


def handcrafted() -> pd.DataFrame:
    """Ten events chosen to cover the interesting corners of the model."""
    rows = [r for spec in HANDCRAFTED for r in _rows(*spec)]
    return pd.DataFrame(rows, columns=COLUMNS)


def seeded(n_events: int = 1000) -> pd.DataFrame:
    """A seeded portfolio: lognormal severities, a mixed shared-factor share.

    ``sd_corr / (sd_indep + sd_corr)`` is drawn per event and applied to both
    units, so the shared-factor loading (and hence the dependence between the
    sites) varies across the portfolio rather than being one fixed number.
    """
    rng = np.random.default_rng(SEED)
    # Rate: small, right-skewed; total rate lands near 0.5 events per year.
    rate = np.round(rng.lognormal(mean=np.log(0.0004), sigma=0.8, size=n_events), 7)
    share = rng.beta(2.0, 2.0, size=n_events)          # shared-factor share
    cv = rng.uniform(0.30, 0.80, size=n_events)        # total sd / mean
    rows = []
    for i in range(n_events):
        event_id = f"e{i + 1:04d}"
        severity = rng.lognormal(mean=np.log(0.13), sigma=0.60)   # of max loss
        # The inland site follows the same storm, at its own scale.
        scale = {"coastal": severity, "inland": severity * rng.uniform(0.4, 1.0)}
        spec = {}
        for unit, maximum in MAX_LOSS.items():
            # Cap the mean well below the maximum: a mean at the cap leaves a
            # beta with almost all its mass at the endpoint, which crowds the
            # tail of the demo without illustrating anything.
            mu = float(np.clip(scale[unit] * maximum, 1.0, 0.80 * maximum))
            # Keep sigma inside the beta bound sqrt(mu (M - mu)) with room to spare.
            sd = min(cv[i] * mu, 0.9 * np.sqrt(mu * (maximum - mu)))
            spec[unit] = (round(mu, 2), round(sd * (1.0 - share[i]), 2),
                          round(sd * share[i], 2))
        rows.extend(_rows(event_id, float(rate[i]), spec["coastal"], spec["inland"]))
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    for name, frame in (("two_unit_elt_10.csv", handcrafted()),
                        ("two_unit_elt_1000.csv", seeded())):
        path = OUT_DIR / name
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows, "
              f"{frame.event_id.nunique()} events, total rate "
              f"{frame.groupby('event_id').rate.first().sum():.4f})")


if __name__ == "__main__":
    main()
