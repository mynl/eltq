"""SciPy reference implementation of ELT secondary-uncertainty quantization.

Transparent and accurate rather than fast. This module is the mathematical
oracle for the Rust implementation (``eltq``): it follows the contract in
``secondary-uncertainty-mathematical-specification.md`` row by row, and is used
both by the acceptance suite and to generate the golden fixtures under
``fixtures/``. Reference totals use exact summation (``math.fsum``) so oracle
values are not polluted by summation-order noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist

DEFAULT_CLAMP_FACTOR = 0.999
MAX_EXAMPLES = 10

# Canonical diagnostic categories, in canonical order (mirrored by eltq-core).
CATEGORIES = (
    "nonfinite",         # any field NaN/Inf: row dropped
    "negative_field",    # rate, max_loss, mean_loss, or an sd < 0: row dropped
    "mean_exceeds_max",  # mu > M (after optional loss-rate scaling): row dropped
    "zero_rate",         # rate == 0: row carries no mass, skipped
    "degenerate_atom",   # sigma == 0, mu == 0, mu == M, or M == 0: single atom
    "sigma_clamped",     # inadmissible variance repaired by the clamp policy
    "variance_dropped",  # inadmissible variance, row dropped by the drop policy
    "variance_atom",     # inadmissible variance collapsed to an atom by policy
)

VARIANCE_POLICIES = ("clamp", "error", "drop", "atom")


class EltqError(Exception):
    """Base class for reference-implementation errors."""


class VarianceError(EltqError):
    """Inadmissible beta variance under the strict ("error") policy."""


class GridOverflowError(EltqError):
    """Grid does not cover the maximum contributing loss."""

    def __init__(self, message, *, required_length=None, examples=None):
        super().__init__(message)
        self.required_length = required_length
        self.examples = examples or []


@dataclass
class RowIssues:
    """Count plus a bounded set of representative event ids for one category."""

    count: int = 0
    examples: list = field(default_factory=list)

    def add(self, event_id) -> None:
        self.count += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(event_id)


@dataclass
class QuantizeMeta:
    """Metadata and diagnostics accompanying the two grid distributions.

    Masses and means describe the arrays actually returned (so they are
    normalized when ``normalize=True``); ``total_rate``, ``analytic_aal``,
    ``quadrature_secondary_mean`` and ``secondary_mean_error`` are always in
    rate-weighted units.
    """

    bucket_size: float
    grid_length: int
    quadrature_points: int
    normalized: bool
    total_rate: float
    analytic_aal: float
    raw_mass: float
    raw_mean: float
    secondary_mass: float
    secondary_mean: float
    quadrature_secondary_mean: float
    secondary_mean_error: float
    input_rows: int
    valid_rows: int
    diagnostics: dict[str, RowIssues]


# Row classification kinds (internal).
_DROP = 0    # contributes to neither output
_ATOM = 1    # atom at mu in both raw and secondary
_BETA = 2    # atom at mu in raw; K beta midpoint quantiles in secondary


def beta_params(max_loss: float, mean_loss: float, variance: float):
    """Return ``(alpha, beta)`` for admissible ``(M, mu, sigma^2)``.

    Uses ``kappa = mu (M - mu) / sigma^2 - 1`` to avoid forming ``M**2``.
    Caller must ensure ``0 < mu < M`` and ``0 < sigma^2 < mu (M - mu)``.
    """
    p = mean_loss / max_loss
    kappa = mean_loss * (max_loss - mean_loss) / variance - 1.0
    return p * kappa, (1.0 - p) * kappa


def _deposit(hist: np.ndarray, x: float, w: float, inv_h: float) -> None:
    """Linear mass- and mean-preserving deposit of weight ``w`` at loss ``x``."""
    r = x * inv_h
    i = int(r)
    n = hist.size
    if i >= n - 1:
        # Only reachable for x at the top node (coverage is checked upfront).
        hist[n - 1] += w
    else:
        f = r - i
        hist[i] += w * (1.0 - f)
        hist[i + 1] += w * f


def quantize_reference(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    bucket_size,
    grid_length=None,
    quadrature_points=32,
    normalize=False,
    mean_is_loss_rate=False,
    variance_policy="clamp",
    clamp_factor=DEFAULT_CLAMP_FACTOR,
    event_ids=None,
):
    """Quantize an ELT onto an even loss grid; return ``(raw, secondary, meta)``.

    Parameters mirror the specification: ``bucket_size`` is the grid spacing h,
    ``grid_length`` the number of nodes n (auto-derived from the maximum
    contributing loss when None), ``quadrature_points`` the number K of midpoint
    beta quantiles per nondegenerate event. ``variance_policy`` is one of
    ``clamp`` (default, sigma clamped to ``clamp_factor * sqrt(mu (M - mu))``),
    ``error``, ``drop``, or ``atom``.
    """
    rate = np.asarray(rate, dtype=np.float64)
    max_loss = np.asarray(max_loss, dtype=np.float64)
    mean_loss = np.asarray(mean_loss, dtype=np.float64)
    sd_indep = np.asarray(sd_indep, dtype=np.float64)
    sd_corr = np.asarray(sd_corr, dtype=np.float64)
    n_rows = rate.size
    for name, arr in (("max_loss", max_loss), ("mean_loss", mean_loss),
                      ("sd_indep", sd_indep), ("sd_corr", sd_corr)):
        if arr.size != n_rows:
            raise EltqError(f"column {name} has {arr.size} rows, expected {n_rows}")
    if event_ids is None:
        event_ids = list(range(n_rows))
    h = float(bucket_size)
    if not (h > 0.0) or not math.isfinite(h):
        raise EltqError(f"bucket_size must be finite and positive, got {h}")
    K = int(quadrature_points)
    if K < 1:
        raise EltqError(f"quadrature_points must be >= 1, got {K}")
    if variance_policy not in VARIANCE_POLICIES:
        raise EltqError(f"unknown variance_policy {variance_policy!r}")
    if not (0.0 < clamp_factor < 1.0):
        raise EltqError(f"clamp_factor must be in (0, 1), got {clamp_factor}")

    diagnostics: dict[str, RowIssues] = {}

    def note(category, event_id):
        diagnostics.setdefault(category, RowIssues()).add(event_id)

    # Pass 1: classify every row; only then is the grid coverage requirement
    # (max contributing M) known, and dropped rows never touch either output.
    kinds = np.empty(n_rows, dtype=np.int8)
    mu_eff = np.empty(n_rows, dtype=np.float64)
    var_eff = np.empty(n_rows, dtype=np.float64)
    max_contrib_loss = 0.0
    for j in range(n_rows):
        lam, m, mu, s_i, s_c = (rate[j], max_loss[j], mean_loss[j],
                                sd_indep[j], sd_corr[j])
        kinds[j] = _DROP
        if not all(map(math.isfinite, (lam, m, mu, s_i, s_c))):
            note("nonfinite", event_ids[j])
            continue
        if lam < 0.0 or m < 0.0 or mu < 0.0 or s_i < 0.0 or s_c < 0.0:
            note("negative_field", event_ids[j])
            continue
        mu = mu * m if mean_is_loss_rate else mu
        if mu > m:
            note("mean_exceeds_max", event_ids[j])
            continue
        if lam == 0.0:
            note("zero_rate", event_ids[j])
            continue
        sigma2 = s_i * s_i + s_c * s_c
        mu_eff[j] = mu
        if m == 0.0 or sigma2 == 0.0 or mu == 0.0 or mu == m:
            # Boundary/degenerate treatment: a single atom at mu. (For mu at a
            # boundary with sigma > 0 the variance is formally inadmissible;
            # the specified degenerate treatment is the resolution.)
            kinds[j] = _ATOM
            note("degenerate_atom", event_ids[j])
        else:
            v_max = mu * (m - mu)
            if sigma2 >= v_max:
                if variance_policy == "error":
                    raise VarianceError(
                        f"row {event_ids[j]}: sigma^2 = {sigma2} >= "
                        f"mu (M - mu) = {v_max} is inadmissible for a beta"
                    )
                if variance_policy == "drop":
                    note("variance_dropped", event_ids[j])
                    continue
                if variance_policy == "atom":
                    kinds[j] = _ATOM
                    note("variance_atom", event_ids[j])
                else:  # clamp
                    sigma2 = clamp_factor * clamp_factor * v_max
                    kinds[j] = _BETA
                    note("sigma_clamped", event_ids[j])
            else:
                kinds[j] = _BETA
            var_eff[j] = sigma2
        max_contrib_loss = max(max_contrib_loss, m)

    # Grid setup and upfront coverage check: require (n - 1) h >= max M over
    # contributing rows (every deposit satisfies x <= M).
    if grid_length is None:
        n = int(math.ceil(max_contrib_loss / h)) + 1 if max_contrib_loss > 0.0 else 1
    else:
        n = int(grid_length)
        if n < 1:
            raise EltqError(f"grid_length must be >= 1, got {n}")
        if max_contrib_loss > (n - 1) * h:
            required = int(math.ceil(max_contrib_loss / h)) + 1
            examples = [event_ids[j] for j in range(n_rows)
                        if kinds[j] != _DROP
                        and max_loss[j] > (n - 1) * h][:MAX_EXAMPLES]
            raise GridOverflowError(
                f"grid of {n} nodes at spacing {h} covers only {(n - 1) * h}, "
                f"but the maximum contributing loss is {max_contrib_loss}; "
                f"grid_length >= {required} required",
                required_length=required,
                examples=examples,
            )

    # Pass 2: deposits plus exact reference totals.
    inv_h = 1.0 / h
    raw = np.zeros(n, dtype=np.float64)
    secondary = np.zeros(n, dtype=np.float64)
    u = (np.arange(K, dtype=np.float64) + 0.5) / K
    rates_kept, aal_terms, quad_terms = [], [], []
    valid_rows = 0
    for j in range(n_rows):
        kind = kinds[j]
        if kind == _DROP:
            continue
        valid_rows += 1
        lam, m, mu = rate[j], max_loss[j], mu_eff[j]
        rates_kept.append(lam)
        aal_terms.append(lam * mu)
        _deposit(raw, mu, lam, inv_h)
        if kind == _ATOM:
            _deposit(secondary, mu, lam, inv_h)
            quad_terms.append(lam * mu)
        else:
            a, b = beta_params(m, mu, var_eff[j])
            xs = m * beta_dist.ppf(u, a, b)
            w = lam / K
            for x in xs:
                _deposit(secondary, x, w, inv_h)
            quad_terms.append(w * math.fsum(xs))

    total_rate = math.fsum(rates_kept)
    analytic_aal = math.fsum(aal_terms)
    quadrature_secondary_mean = math.fsum(quad_terms)

    if normalize:
        if not total_rate > 0.0:
            raise EltqError("cannot normalize: total contributing rate is zero")
        raw /= total_rate
        secondary /= total_rate

    grid = h * np.arange(n, dtype=np.float64)
    meta = QuantizeMeta(
        bucket_size=h,
        grid_length=n,
        quadrature_points=K,
        normalized=bool(normalize),
        total_rate=total_rate,
        analytic_aal=analytic_aal,
        raw_mass=float(raw.sum()),
        raw_mean=float(raw @ grid),
        secondary_mass=float(secondary.sum()),
        secondary_mean=float(secondary @ grid),
        quadrature_secondary_mean=quadrature_secondary_mean,
        secondary_mean_error=quadrature_secondary_mean - analytic_aal,
        input_rows=n_rows,
        valid_rows=valid_rows,
        diagnostics=diagnostics,
    )
    return raw, secondary, meta
