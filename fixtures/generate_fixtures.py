"""Generate synthetic ELT fixtures and golden outputs from the SciPy reference.

Run from the repository root:

    uv run python fixtures/generate_fixtures.py

Fully deterministic (fixed seed); regenerated files are committed. All data is
synthetic (data firewall: no real data ever enters the repository).

Outputs:
    fixtures/small_elt.csv                handcrafted rows covering every category
    fixtures/medium_elt.csv               5000-row seeded mixture of regimes
    fixtures/golden/<name>.npz            raw + secondary golden arrays
    fixtures/golden/<name>.meta.json      golden metadata + diagnostics
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eltq.reference import quantize_reference

FIXTURES_DIR = Path(__file__).parent

COLUMNS = ["event_id", "rate", "max_loss", "mean_loss", "sd_indep", "sd_corr"]
SEED = 20260814

# Golden run parameters, shared with the acceptance suite via the meta files.
SMALL_GOLDEN = {"bucket_size": 0.5, "quadrature_points": 8}
MEDIUM_GOLDEN = {"bucket_size": 5000.0, "quadrature_points": 32}


def small_elt() -> pd.DataFrame:
    """Handcrafted rows exercising every diagnostic category and fast path."""
    rows = [
        # event_id, rate, M, mu, sd_indep, sd_corr
        # Well-conditioned betas (various shapes).
        ("well_1", 0.020, 100.0, 40.0, 15.0, 10.0),
        ("well_2", 0.015, 80.0, 20.0, 8.0, 6.0),
        ("well_3", 0.010, 60.0, 45.0, 7.0, 3.0),
        # Uniform beta: mu = M/2, sigma^2 = M^2/12.
        ("uniform", 0.008, 96.0, 48.0, 96.0 / np.sqrt(12.0), 0.0),
        # Symmetric non-uniform: alpha = beta = 2.
        ("symmetric", 0.006, 50.0, 25.0, 25.0 / np.sqrt(5.0), 0.0),
        # Skewed, alpha < 1.
        ("skew_low", 0.030, 100.0, 2.0, 6.0, 4.0),
        # Concentrated, large alpha and beta.
        ("concentrated", 0.004, 100.0, 55.0, 1.0, 0.5),
        # Degenerate atoms.
        ("sigma_zero", 0.012, 100.0, 33.3, 0.0, 0.0),
        ("mu_zero", 0.011, 100.0, 0.0, 5.0, 0.0),
        ("mu_max", 0.009, 75.0, 75.0, 0.0, 3.0),
        ("m_zero", 0.007, 0.0, 0.0, 0.0, 0.0),
        # Inadmissible variance (sigma^2 >= mu (M - mu)): clamped by default.
        ("inadmissible_1", 0.005, 100.0, 30.0, 50.0, 30.0),
        ("inadmissible_2", 0.003, 40.0, 20.0, 20.0, 0.0),
        # Zero rate.
        ("zero_rate", 0.0, 90.0, 30.0, 10.0, 5.0),
        # Duplicate event ids (allowed; rows are independent).
        ("dup", 0.002, 30.0, 10.0, 4.0, 2.0),
        ("dup", 0.002, 30.0, 12.0, 4.0, 2.0),
        # Atom exactly on a grid node for the small golden (h = 0.5).
        ("on_node", 0.013, 100.0, 42.5, 0.0, 0.0),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def medium_elt(n: int = 5000) -> pd.DataFrame:
    """Seeded mixture of parameter regimes at portfolio-like loss scales."""
    rng = np.random.default_rng(SEED)
    m = np.exp(rng.uniform(np.log(1e4), np.log(5e6), size=n))
    lam = np.exp(rng.uniform(np.log(1e-6), np.log(1e-2), size=n))
    p = 0.05 + 0.9 * rng.beta(2.0, 2.0, size=n)
    # Variance as a fraction f of the admissible maximum p (1 - p).
    f = rng.uniform(0.05, 0.6, size=n)

    kind = rng.choice(
        ["well", "skew", "conc", "sig0", "bound", "inadm", "zrate", "m0"],
        size=n,
        p=[0.70, 0.10, 0.05, 0.05, 0.03, 0.04, 0.02, 0.01],
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
    sig0 = kind == "sig0"
    sigma[sig0] = 0.0
    bound = kind == "bound"
    at_max = rng.random(size=n) < 0.5
    mu[bound & at_max] = m[bound & at_max]
    mu[bound & ~at_max] = 0.0
    lam[kind == "zrate"] = 0.0
    m0 = kind == "m0"
    m[m0] = 0.0
    mu[m0] = 0.0
    sigma[m0] = 0.0

    # Split sigma into independent and correlated components. The total sd
    # is the arithmetic sum sd_indep + sd_corr (Burke 2016 rule, 1.6.0), so a
    # linear split keeps f the fraction of the admissible variance.
    share = rng.uniform(0.0, 1.0, size=n)
    sd_indep = sigma * share
    sd_corr = sigma * (1.0 - share)
    return pd.DataFrame(
        {
            "event_id": [f"ev{i:05d}" for i in range(n)],
            "rate": lam,
            "max_loss": m,
            "mean_loss": mu,
            "sd_indep": sd_indep,
            "sd_corr": sd_corr,
        }
    )


def write_golden(df: pd.DataFrame, name: str, params: dict) -> None:
    raw, secondary, meta = quantize_reference(
        df["rate"], df["max_loss"], df["mean_loss"], df["sd_indep"],
        df["sd_corr"], event_ids=list(df["event_id"]), **params,
    )
    golden_dir = FIXTURES_DIR / "golden"
    golden_dir.mkdir(exist_ok=True)
    np.savez(golden_dir / f"{name}.npz", raw=raw, secondary=secondary)
    # Full-precision text twin of the arrays for the Rust golden tests.
    np.savetxt(golden_dir / f"{name}.arrays.csv",
               np.column_stack([raw, secondary]), fmt="%.17g", delimiter=",",
               header="raw,secondary", comments="")
    meta_dict = dataclasses.asdict(meta)
    meta_dict["params"] = params
    (golden_dir / f"{name}.meta.json").write_text(
        json.dumps(meta_dict, indent=2), encoding="utf-8"
    )
    print(f"golden {name}: n={meta.grid_length} rows={meta.input_rows} "
          f"valid={meta.valid_rows} mass_err="
          f"{abs(meta.raw_mass - meta.total_rate):.3e}")


def main() -> None:
    # Goldens are computed from the CSV as re-read, so the committed golden
    # matches what any implementation sees after parsing the fixture file.
    for name, df, params in (
        ("small", small_elt(), SMALL_GOLDEN),
        ("medium", medium_elt(), MEDIUM_GOLDEN),
    ):
        path = FIXTURES_DIR / f"{name}_elt.csv"
        df.to_csv(path, index=False)
        # round_trip parsing: bit-identical to what the Rust CSV route sees.
        reread = pd.read_csv(path, float_precision="round_trip")
        write_golden(reread, name, params)


if __name__ == "__main__":
    main()
