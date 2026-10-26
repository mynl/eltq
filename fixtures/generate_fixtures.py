"""Generate synthetic ELT fixtures and golden outputs from the SciPy reference.

Run from the repository root:

    uv run python fixtures/generate_fixtures.py

Fully deterministic (fixed seed); regenerated files are committed. All data is
synthetic (data firewall: no real data ever enters the repository).

Outputs:
    fixtures/small_elt.csv                handcrafted rows covering every category
    fixtures/medium_elt.csv               5000-row seeded mixture of regimes
    fixtures/small_alpha_elt.csv          2000 small-alpha rows for the survival
                                          tests (spec tests 15 to 20)
    fixtures/golden/<name>.npz            raw + secondary golden arrays, for
                                          small and medium under the quantiles
                                          method and small_survival and
                                          medium_survival under the default
    fixtures/golden/<name>.meta.json      golden metadata + diagnostics
    fixtures/multiunit_small_elt.csv      handcrafted event x unit rows (1.7.0)
    fixtures/multiunit_small_terms.txt    DecL-style unit terms for it
    fixtures/golden/multiunit_small.*     per-view golden arrays + metadata
    fixtures/golden/joint_small.*         two-unit joint pmf golden + metadata
    fixtures/golden/terms_parse.json      terms grammar parse goldens + errors
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eltq.reference import quantize_reference
from eltq.reference_multiunit import multiunit_views_reference, secondary_two_units
from eltq.terms import KEYWORDS, TermsError, parse_terms_line

FIXTURES_DIR = Path(__file__).parent

COLUMNS = ["event_id", "rate", "max_loss", "mean_loss", "sd_indep", "sd_corr"]
SEED = 20260814

# Golden run parameters, shared with the acceptance suite via the meta files.
SMALL_GOLDEN = {"bucket_size": 0.5, "method": "quantiles", "quadrature_points": 8}
MEDIUM_GOLDEN = {"bucket_size": 5000.0, "method": "quantiles", "quadrature_points": 32}
SMALL_SURVIVAL_GOLDEN = {"bucket_size": 0.5, "method": "survival"}
MEDIUM_SURVIVAL_GOLDEN = {"bucket_size": 5000.0, "method": "survival"}


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
        # Inadmissible variance (sigma^2 >= mu (M - mu)): Bernoulli rows by
        # default from 1.13.0 (two atoms, at 0 and at M).
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


def small_alpha_elt(n: int = 1000) -> pd.DataFrame:
    """Small-alpha events for the survival tests: ground-up rows (``gu``)
    and a gross-like copy (``gr``) with mu and alpha both shrunk fivefold
    and M kept, as deductibles do.

    Shapes are drawn directly: alpha loguniform on [0.003, 3], M / mu
    loguniform on [1e2, 1e4] with a tenth of the rows at 1e5 on M = 1e6, so
    that the variance is admissible by construction. A few rows are atoms,
    inadmissible (Bernoulli by default) or zero-rate.
    """
    rng = np.random.default_rng(SEED + 1)
    m = np.exp(rng.uniform(np.log(1e4), np.log(1e6), size=n))
    ratio = np.exp(rng.uniform(np.log(1e2), np.log(1e4), size=n))
    deep = rng.random(n) < 0.1
    m[deep] = 1e6
    ratio[deep] = 1e5
    alpha = np.exp(rng.uniform(np.log(0.003), np.log(3.0), size=n))
    lam = np.exp(rng.uniform(np.log(1e-5), np.log(1e-2), size=n))
    kind = rng.choice(["beta", "atom", "inadm", "zrate"], size=n,
                      p=[0.94, 0.03, 0.02, 0.01])
    share = rng.uniform(0.0, 1.0, size=n)
    frames = []
    for label, shrink in (("gu", 1.0), ("gr", 0.2)):
        p = shrink / ratio
        a = shrink * alpha
        kappa = a / p
        v = p * (1.0 - p) / (kappa + 1.0)
        mu = p * m
        sigma = m * np.sqrt(v)
        sigma[kind == "atom"] = 0.0
        inadm = kind == "inadm"
        sigma[inadm] = 1.2 * np.sqrt(mu[inadm] * (m[inadm] - mu[inadm]))
        rate = lam.copy()
        rate[kind == "zrate"] = 0.0
        frames.append(pd.DataFrame({
            "event_id": [f"{label}{i:04d}" for i in range(n)],
            "rate": rate,
            "max_loss": m,
            "mean_loss": mu,
            "sd_indep": sigma * share,
            "sd_corr": sigma * (1.0 - share),
        }))
    return pd.concat(frames, ignore_index=True)


def write_golden(df: pd.DataFrame, name: str, params: dict, csv_name: str | None = None) -> None:
    """Write the golden arrays and metadata for ``name``; ``csv_name`` is the
    fixture CSV the golden is computed from (``name`` when None)."""
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
    meta_dict["fixture"] = csv_name or name
    (golden_dir / f"{name}.meta.json").write_text(
        json.dumps(meta_dict, indent=2), encoding="utf-8"
    )
    print(f"golden {name}: n={meta.grid_length} rows={meta.input_rows} "
          f"valid={meta.valid_rows} mass_err="
          f"{abs(meta.raw_mass - meta.total_rate):.3e}")


MULTI_COLUMNS = ["event_id", "unit_id", "rate", "mean_loss", "sd_indep",
                 "sd_corr", "max_loss"]
MULTI_GOLDEN = {"bucket_size": 10.0, "z_nodes": 32, "z_rule": "stratified"}
MULTI_TERMS = """\
# Synthetic unit terms for fixtures/multiunit_small_elt.csv (1.7.0).
unit a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500
unit b 1000 xs 0 occurrence net of 100 po 400 xs 100
unit c
unit d occurrence net of 50% po 300 xs 100 and 50% po 200 xs 200
unit e 600 xs 50
unit spare 100 xs 0
"""

JOINT_GOLDEN = {
    "units": [[200.0, 100.0, 150.0, 1000.0], [500.0, 300.0, 200.0, 2000.0]],
    "bucket_size": 20.0, "n1": 51, "n2": 101, "z_nodes": 1001, "z_rule": "midpoint",
}

TERMS_PARSE_OK = [
    "unit a",
    "unit plant_a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500",
    "unit plant_b 1000 xs 0 occurrence net of 100 po 400 xs 100",
    "unit fleet occurrence net of .2 po 5 xs 1",
    "unit x inf xs 100",
    "unit y 1e6 xs 2.5e5 occurrence net of 12.5% po 1e5 xs 5e4 and 25 po 100 xs 0",
    "unit z 500 xs 0 occurrence net of 100% po 500 xs 0",
]
TERMS_PARSE_ERRORS = [
    "",
    "plant_a 750 xs 250",
    "unit",
    "unit xs",
    "unit a 750 xs",
    "unit a 750 of 250",
    "unit a occurrence net 400 xs 100",
    "unit a occurrence net of 400 xs 100 and",
    "unit a occurrence net of 400 xs 100 500 xs 500",
    "unit a occurrence net of 150% po 400 xs 100",
    "unit a occurrence net of 500 po 400 xs 100",
    "unit a occurrence net of 0 po 400 xs 100",
    "unit a occurrence net of 0 xs 100",
    "unit a occurrence net of inf xs 100",
    "unit a 0 xs 0",
    "unit a 750 xs -1",
    "unit a nan xs 0",
    "unit a 750 xs 250 occurrence net of 400 xs -5",
    "unit a occurrence net of 300 xs 100 and 200 xs 200",
    "unit a occurrence net of 60% po 300 xs 100 and 50% po 200 xs 200",
]


def multiunit_small_elt() -> pd.DataFrame:
    """Handcrafted event x unit rows covering every unit kind and policy."""
    rows = [
        # event_id, unit_id, rate, mu, sd_indep, sd_corr, M
        # e1: three beta units sharing the factor, mixed loadings.
        ("e1", "a", 0.010, 200.0, 100.0, 150.0, 1000.0),
        ("e1", "b", 0.010, 500.0, 300.0, 200.0, 2000.0),
        ("e1", "c", 0.010, 300.0, 50.0, 100.0, 800.0),
        # e2: comonotone unit (sd_indep = 0) with an independent unit.
        ("e2", "a", 0.020, 250.0, 0.0, 200.0, 1000.0),
        ("e2", "c", 0.020, 100.0, 120.0, 0.0, 500.0),
        # e3: single-unit event.
        ("e3", "b", 0.005, 400.0, 80.0, 60.0, 1200.0),
        # e4: four units incl. an atom (sigma = 0); unit b is admissible
        # (900^2 < 600 * 1400), so the fixture has no inadmissible row.
        ("e4", "a", 0.003, 300.0, 0.0, 0.0, 1000.0),
        ("e4", "b", 0.003, 600.0, 400.0, 500.0, 2000.0),
        ("e4", "d", 0.003, 150.0, 60.0, 40.0, 600.0),
        ("e4", "e", 0.003, 350.0, 90.0, 70.0, 900.0),
        # e5: zero-rate event, skipped entirely.
        ("e5", "a", 0.000, 200.0, 50.0, 50.0, 1000.0),
        ("e5", "b", 0.000, 300.0, 50.0, 50.0, 1000.0),
        # e6: a dropped row (mean > max) alongside a live unit.
        ("e6", "d", 0.007, 700.0, 10.0, 10.0, 600.0),
        ("e6", "e", 0.007, 200.0, 40.0, 30.0, 900.0),
        # e7: boundary means (atoms at 0 and at M) with an ordinary unit.
        ("e7", "a", 0.004, 0.0, 5.0, 5.0, 1000.0),
        ("e7", "c", 0.004, 800.0, 0.0, 3.0, 800.0),
        ("e7", "e", 0.004, 300.0, 100.0, 50.0, 900.0),
    ]
    return pd.DataFrame(rows, columns=MULTI_COLUMNS)


def write_multiunit_golden() -> None:
    golden_dir = FIXTURES_DIR / "golden"
    path = FIXTURES_DIR / "multiunit_small_elt.csv"
    multiunit_small_elt().to_csv(path, index=False)
    terms_path = FIXTURES_DIR / "multiunit_small_terms.txt"
    terms_path.write_text(MULTI_TERMS, encoding="utf-8", newline="\n")
    df = pd.read_csv(path, float_precision="round_trip")
    views, meta = multiunit_views_reference(
        *(df[c] for c in MULTI_COLUMNS), terms=MULTI_TERMS, **MULTI_GOLDEN,
    )
    names = list(meta.views)
    np.savez(golden_dir / "multiunit_small.npz", **{v: views[v] for v in names})
    np.savetxt(golden_dir / "multiunit_small.arrays.csv",
               np.column_stack([views[v] for v in names]), fmt="%.17g",
               delimiter=",", header=",".join(names), comments="")
    meta_dict = dataclasses.asdict(meta)
    meta_dict["params"] = MULTI_GOLDEN
    (golden_dir / "multiunit_small.meta.json").write_text(
        json.dumps(meta_dict, indent=2), encoding="utf-8"
    )
    print(f"golden multiunit_small: n={meta.grid_length} rows={meta.input_rows} "
          f"valid={meta.valid_rows} events={meta.valid_events}/{meta.events}")


def write_joint_golden() -> None:
    golden_dir = FIXTURES_DIR / "golden"
    p = JOINT_GOLDEN
    result = secondary_two_units(p["units"], p["bucket_size"], p["n1"], p["n2"],
                                 n_z=p["z_nodes"], z_rule=p["z_rule"])
    np.savetxt(golden_dir / "joint_small.pmf.csv", result.pmf, fmt="%.17g",
               delimiter=",")
    meta_dict = dataclasses.asdict(result.meta)
    meta_dict["params"] = p
    (golden_dir / "joint_small.meta.json").write_text(
        json.dumps(meta_dict, indent=2), encoding="utf-8"
    )
    print(f"golden joint_small: shape={result.pmf.shape} mass={result.meta.total_mass:.15f}")


def write_terms_parse_golden() -> None:
    ok = []
    for line in TERMS_PARSE_OK:
        terms = parse_terms_line(line)
        ok.append({
            "line": line,
            "unit_id": terms.unit_id,
            "limit": None if np.isinf(terms.limit) else terms.limit,
            "ded": terms.ded,
            "clauses": [[c.share, c.limit, c.attach] for c in terms.clauses],
            "text": terms.text(),
        })
    for line in TERMS_PARSE_ERRORS:
        try:
            parse_terms_line(line)
        except TermsError:
            continue
        raise AssertionError(f"expected a parse error for {line!r}")
    (FIXTURES_DIR / "golden" / "terms_parse.json").write_text(
        json.dumps({"ok": ok, "errors": TERMS_PARSE_ERRORS,
                    "keywords": sorted(KEYWORDS)}, indent=2),
        encoding="utf-8",
    )
    print(f"golden terms_parse: {len(ok)} parses, {len(TERMS_PARSE_ERRORS)} errors")


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
        survival = SMALL_SURVIVAL_GOLDEN if name == "small" else MEDIUM_SURVIVAL_GOLDEN
        write_golden(reread, f"{name}_survival", survival, csv_name=name)
    # No golden yet: the survival tests check this fixture against exact
    # evaluation; goldens for it arrive with the 2.0.0 default switch.
    small_alpha_elt().to_csv(FIXTURES_DIR / "small_alpha_elt.csv", index=False)
    write_multiunit_golden()
    write_joint_golden()
    write_terms_parse_golden()


if __name__ == "__main__":
    main()
