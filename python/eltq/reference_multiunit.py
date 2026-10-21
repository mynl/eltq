"""SciPy reference for joint multi-unit secondary uncertainty.

Transparent and accurate rather than fast; the oracle the Rust engine is
tested against and the fallback backend for the multi-unit API. Two entry
points:

- :func:`secondary_two_units`: exact bucketed bivariate pmf of one event's
  losses across two units (the ``dev/`` reference of 2026-08-25, promoted);
- :func:`multiunit_views_reference`: for a multi-unit ELT (event x unit rows)
  with per-unit DecL-style terms (:mod:`eltq.terms`), the rate-accumulated
  total event severity by view (ground_up, gross, ceded, net).

Model (Burke, Rau-Chaplin, Varghese 2016, section III.E; derivations and the
additivity analysis in ``dev/secondary-uncertainty-additivity.md``). Per unit
i of an event: total sd ``sigma = sd_indep + sd_corr``, damage ratio
``X_i / M_i ~ Beta(alpha_i, beta_i)`` moment-matched to ``(mu_i / M_i,
sigma_i / M_i)``; one shared standard normal ``Z_c`` per event occurrence and
an independent ``Z_i`` per unit give ``U_i = Phi((sd_corr Z_c + sd_indep
Z_i) / t_i)``, ``t_i = hypot(sd_indep, sd_corr)``, and ``X_i = M_i
F_i^{-1}(U_i)``. Conditional on ``Z_c = z`` the units are independent with
``P(X_i <= x | z) = Phi((Phi^{-1}(F_i(x / M_i)) - w_c z) / w_I)``.

Discretization conventions, chosen to match eltq's deposit rule:

- a beta unit's conditional pmf is bucketed by cdf differences at half-bucket
  edges (exact in probability; bucket k holds ``((k - 1/2) h, (k + 1/2) h]``);
- point values (degenerate atoms, comonotone units' per-z atoms, and every
  transformed view value) use the linear mass- and mean-preserving deposit;
- the shared factor is integrated by midpoint quantile nodes
  ``z_q = Phi^{-1}((q + 1/2) / Q)`` with equal weights (default) or
  Gauss-Hermite nodes and weights (``z_rule="gauss_hermite"``).

Fresh ``(Z_c, Z_i)`` per occurrence make occurrences iid, so the
rate-weighted mixture severity plus a Poisson frequency reproduces the
compound distribution exactly; accumulation across events is the ordinary
eltq pattern.
"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
from scipy.special import betainc
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from eltq._meta import JointMeta, MultiMeta, RowIssues
from eltq.reference import (
    DEFAULT_CLAMP_FACTOR,
    MAX_EXAMPLES,
    EltqError,
    GridOverflowError,
    VarianceError,
    _deposit,
)
from eltq.terms import VIEWS, TermsError, UnitTerms, check_grid_alignment, parse_terms

VARIANCE_POLICIES = ("clamp", "error", "drop", "atom")
Z_RULES = ("midpoint", "gauss_hermite")

#: Input record for one unit, in ELT field order. ``mean_loss`` and the sds are
#: in loss currency units (not damage ratios); ``max_loss`` is the beta upper
#: bound M. Plain 4-tuples in this order are accepted everywhere a Unit is.
Unit = namedtuple("Unit", ["mean_loss", "sd_indep", "sd_corr", "max_loss"])


class MultiUnitError(EltqError):
    """Structural problem in a multi-unit ELT: events not contiguous, a rate
    varying within an event, or a unit repeated within an event."""


# Unit kinds after validation.
_DROP = 0        # contributes nothing (invalid or dropped by policy)
_ATOM = 1        # degenerate: all conditional mass at one loss value
_COMONOTONE = 2  # sd_indep == 0: loss is a deterministic function of Z_c
_BETA = 3        # general case


@dataclass
class _Prepped:
    """Validated, derived per-unit parameters."""

    kind: int
    mean_loss: float
    max_loss: float
    alpha: float = math.nan
    beta: float = math.nan
    w_corr: float = 0.0      # sd_corr / t, the shared-factor loading
    w_indep: float = 0.0     # sd_indep / t
    sigma_used: float = 0.0  # total sd after any clamp, loss units
    notes: list = field(default_factory=list)

    @property
    def support(self) -> float:
        """Largest loss the unit can produce."""
        return self.mean_loss if self.kind == _ATOM else self.max_loss


def _note(diagnostics: dict, category: str, example) -> None:
    issues = diagnostics.setdefault(category, RowIssues())
    issues.count += 1
    if len(issues.examples) < MAX_EXAMPLES:
        issues.examples.append(example)


def prepare_unit(unit, label, variance_policy="clamp",
                 clamp_factor=DEFAULT_CLAMP_FACTOR, diagnostics=None) -> _Prepped:
    """Validate one unit and derive beta parameters and copula weights.

    Structural problems (nonfinite, negative, mean above max) raise
    ``EltqError`` when ``diagnostics`` is None (the two-unit joint) and are
    recorded as dropped rows otherwise (the multi-unit ELT route, mirroring
    the one-unit reference's categories).
    """
    mu, s_i, s_c, m = (float(x) for x in unit)

    def problem(category, message):
        if diagnostics is None:
            raise EltqError(f"{label}: {message}")
        _note(diagnostics, category, label)
        return _Prepped(kind=_DROP, mean_loss=0.0, max_loss=0.0)

    if not all(map(math.isfinite, (mu, s_i, s_c, m))):
        return problem("nonfinite", "all fields must be finite")
    if mu < 0.0 or s_i < 0.0 or s_c < 0.0 or m < 0.0:
        return problem("negative_field", "negative field")
    if mu > m:
        return problem("mean_exceeds_max", f"mean_loss {mu} exceeds max_loss {m}")
    sigma = s_i + s_c
    prep = _Prepped(kind=_ATOM, mean_loss=mu, max_loss=m, sigma_used=sigma)
    if m == 0.0 or sigma == 0.0 or mu == 0.0 or mu == m:
        # Boundary treatment as in the one-unit spec: a single atom at mu.
        prep.notes.append("degenerate_atom")
        if diagnostics is not None:
            _note(diagnostics, "degenerate_atom", label)
        return prep
    v_max = mu * (m - mu)
    if sigma * sigma >= v_max:
        if variance_policy == "error":
            raise VarianceError(
                f"{label}: sigma^2 = {sigma * sigma} >= mu (M - mu) = {v_max} "
                "is inadmissible for a beta")
        if variance_policy == "drop":
            prep.kind = _DROP
            prep.notes.append("variance_dropped")
            if diagnostics is not None:
                _note(diagnostics, "variance_dropped", label)
            return prep
        if variance_policy == "atom":
            prep.notes.append("variance_atom")
            if diagnostics is not None:
                _note(diagnostics, "variance_atom", label)
            return prep
        sigma = clamp_factor * math.sqrt(v_max)
        prep.sigma_used = sigma
        prep.notes.append("sigma_clamped")
        if diagnostics is not None:
            _note(diagnostics, "sigma_clamped", label)
    p = mu / m
    kappa = mu * (m - mu) / (sigma * sigma) - 1.0
    prep.alpha = p * kappa
    prep.beta = (1.0 - p) * kappa
    t = math.hypot(s_i, s_c)
    prep.w_corr = s_c / t
    prep.w_indep = s_i / t
    prep.kind = _COMONOTONE if s_i == 0.0 else _BETA
    return prep


def z_quadrature(n_z: int, rule: str = "midpoint"):
    """Nodes and weights for integrating out the shared factor ``Z_c``.

    ``midpoint``: ``z_q = Phi^{-1}((q + 1/2) / n_z)`` with equal weights, the
    midpoint-quantile style of the one-unit specification. ``gauss_hermite``:
    probabilists' Gauss-Hermite nodes with weights normalized to sum to one.
    """
    q_n = int(n_z)
    if q_n < 1:
        raise EltqError(f"z_nodes must be >= 1, got {n_z}")
    if rule == "midpoint":
        z = norm.ppf((np.arange(q_n) + 0.5) / q_n)
        return z, np.full(q_n, 1.0 / q_n)
    if rule == "gauss_hermite":
        return _gauss_hermite(q_n)
    raise EltqError(f"unknown z_rule {rule!r}; expected one of {Z_RULES}")


def _gauss_hermite(n: int):
    """Probabilists' Gauss-Hermite nodes and weights by Golub-Welsch.

    Eigen-decomposition of the symmetric tridiagonal Jacobi matrix (zero
    diagonal, off-diagonal sqrt(k)); stable for large n where the polynomial
    recurrence in ``numpy.polynomial.hermite_e.hermegauss`` overflows.
    Weights sum to one (the standard normal density is the weight function).
    """
    if n == 1:
        return np.zeros(1), np.ones(1)
    off = np.sqrt(np.arange(1, n, dtype=np.float64))
    jacobi = np.diag(off, 1) + np.diag(off, -1)
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = vectors[0] ** 2
    # Symmetrize exactly: nodes come in +/- pairs with equal weights.
    nodes = 0.5 * (nodes - nodes[::-1])
    weights = 0.5 * (weights + weights[::-1])
    return nodes, weights / weights.sum()


def grid_length_for(max_loss: float, bucket_size: float) -> int:
    """Smallest grid length covering ``max_loss``: ceil(M / h) + 1."""
    return int(math.ceil(max_loss / bucket_size)) + 1 if max_loss > 0.0 else 1


def marginal_pmf(prep: _Prepped, bucket_size: float, n: int) -> np.ndarray:
    """Unconditional bucket pmf of one unit (beta cdf differences at
    half-bucket edges; the linear deposit for an atom)."""
    if prep.kind == _ATOM:
        row = np.zeros(n)
        _deposit(row, prep.mean_loss, 1.0, 1.0 / bucket_size)
        return row
    edges = (np.arange(n + 1) - 0.5) * bucket_size
    return np.diff(beta_dist.cdf(edges / prep.max_loss, prep.alpha, prep.beta))


def conditional_pmfs(prep: _Prepped, bucket_size: float, n: int, z: np.ndarray) -> np.ndarray:
    """Conditional bucket pmf of one unit for each shared-factor value.

    Returns a ``(len(z), n)`` array whose row q is the pmf of the unit's loss
    on buckets ``0, h, ..., (n - 1) h`` conditional on ``Z_c = z[q]``. Every
    row sums to one. Beta units use cdf differences at half-bucket edges;
    atoms and comonotone units use the linear deposit of their point value.
    """
    q_n = z.size
    inv_h = 1.0 / bucket_size
    pmf = np.zeros((q_n, n))
    if prep.kind == _ATOM:
        row = np.zeros(n)
        _deposit(row, prep.mean_loss, 1.0, inv_h)
        pmf[:] = row
        return pmf
    if prep.kind == _COMONOTONE:
        # U = Phi(w_corr z) with no independent noise: an atom per z value.
        u = norm.cdf(prep.w_corr * z)
        xs = prep.max_loss * beta_dist.ppf(u, prep.alpha, prep.beta)
        for q, x in enumerate(xs):
            _deposit(pmf[q], min(x, prep.max_loss), 1.0, inv_h)
        return pmf
    # General case: the normal score of each half-bucket edge,
    # y = Phi^{-1}(F_beta(edge / M)), is computed once; edges below 0 map to
    # -inf and edges at or above M to +inf, so each row is complete.
    edges = (np.arange(n + 1) - 0.5) * bucket_size
    y = norm.ppf(beta_dist.cdf(edges / prep.max_loss, prep.alpha, prep.beta))
    with np.errstate(invalid="ignore"):
        cdf = norm.cdf((y[None, :] - prep.w_corr * z[:, None]) / prep.w_indep)
    return np.diff(cdf, axis=1)


# ---------------------------------------------------------------------------
# Two-unit joint distribution.

@dataclass
class BivariateSU:
    """Bucketed bivariate loss distribution for one event and two units.

    ``pmf[j, k]`` is the probability that unit 1 loses ``x1[j]`` and unit 2
    loses ``x2[k]``, both grids sharing the common bucket size. ``meta``
    carries the derived parameters and moment diagnostics.
    """

    pmf: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    meta: JointMeta

    def frame(self):
        """Return the pmf as a pandas DataFrame indexed by the two loss grids."""
        import pandas as pd

        return pd.DataFrame(self.pmf, index=pd.Index(self.x1, name="x1"),
                            columns=pd.Index(self.x2, name="x2"))

    def marginal(self, unit: int) -> np.ndarray:
        """Return the marginal pmf of unit 1 or unit 2 (1-based)."""
        if unit not in (1, 2):
            raise ValueError("unit must be 1 or 2")
        return self.pmf.sum(axis=1) if unit == 1 else self.pmf.sum(axis=0)

    def sum_distribution(self):
        """Return ``(xs, ps)`` for the total loss ``X1 + X2``.

        Exact given the joint pmf: both units share one bucket size, so bucket
        indices add and the anti-diagonals of ``pmf`` accumulate directly.
        """
        n1, n2 = self.pmf.shape
        idx = np.add.outer(np.arange(n1), np.arange(n2))
        ps = np.bincount(idx.ravel(), weights=self.pmf.ravel(), minlength=n1 + n2 - 1)
        return self.meta.bucket_size * np.arange(n1 + n2 - 1), ps


def secondary_two_units(units, bucket_size, n1, n2, *, n_z=1001, z_rule="midpoint",
                        variance_policy="clamp", clamp_factor=DEFAULT_CLAMP_FACTOR):
    """Joint bucketed loss distribution for one event across two units.

    Parameters
    ----------
    units : sequence of two Unit or 4-tuples
        Per-unit ELT fields ``(mean_loss, sd_indep, sd_corr, max_loss)`` in
        loss currency units.
    bucket_size : float
        Common grid spacing h; bucket k represents the loss ``k h``.
    n1, n2 : int
        Grid lengths for unit 1 and unit 2. Each must cover its unit's
        maximum loss, ``(n - 1) h >= M``; anything less raises.
    n_z : int, optional
        Number of shared-factor quadrature nodes. Accuracy control; raise it
        when ``sd_corr`` dominates ``sd_indep``.
    z_rule : {"midpoint", "gauss_hermite"}, optional
        Quadrature rule for the shared factor.
    variance_policy : {"clamp", "error", "atom"}, optional
        Treatment of a unit whose sd is inadmissible for a beta: clamp sigma
        to ``clamp_factor`` times the bound (Burke's own repair, the
        default), raise, or collapse the unit to an atom at its mean.
    clamp_factor : float, optional
        Factor in (0, 1) applied to the sd bound under the clamp policy.

    Returns
    -------
    BivariateSU
        Joint pmf on the two loss grids plus :class:`JointMeta`.
    """
    if len(units) != 2:
        raise EltqError(f"expected exactly 2 units, got {len(units)}")
    h = float(bucket_size)
    if not (h > 0.0) or not math.isfinite(h):
        raise EltqError(f"bucket_size must be finite and positive, got {h}")
    if variance_policy not in ("clamp", "error", "atom"):
        raise EltqError(f"unknown variance_policy {variance_policy!r} for the joint")
    if not (0.0 < clamp_factor < 1.0):
        raise EltqError(f"clamp_factor must be in (0, 1), got {clamp_factor}")
    z, w = z_quadrature(n_z, z_rule)
    lengths = (int(n1), int(n2))
    preps = []
    for i, (unit, n) in enumerate(zip(units, lengths), start=1):
        if n < 1:
            raise EltqError(f"n{i} must be >= 1, got {n}")
        prep = prepare_unit(unit, f"unit {i}", variance_policy, clamp_factor)
        if prep.max_loss > (n - 1) * h:
            raise GridOverflowError(
                f"unit {i}: grid of {n} nodes at spacing {h} covers only "
                f"{(n - 1) * h}, but max_loss is {prep.max_loss}; "
                f"n{i} >= {grid_length_for(prep.max_loss, h)} required",
                required_length=grid_length_for(prep.max_loss, h))
        preps.append(prep)

    cond1 = conditional_pmfs(preps[0], h, lengths[0], z)
    cond2 = conditional_pmfs(preps[1], h, lengths[1], z)
    pmf = np.einsum("q,qi,qj->ij", w, cond1, cond2)

    x1 = h * np.arange(lengths[0])
    x2 = h * np.arange(lengths[1])
    marg1, marg2 = pmf.sum(axis=1), pmf.sum(axis=0)
    mean1, mean2 = marg1 @ x1, marg2 @ x2
    sd1 = math.sqrt(max(marg1 @ x1**2 - mean1**2, 0.0))
    sd2 = math.sqrt(max(marg2 @ x2**2 - mean2**2, 0.0))
    cross = x1 @ pmf @ x2
    corr = ((cross - mean1 * mean2) / (sd1 * sd2)
            if sd1 > 0.0 and sd2 > 0.0 else math.nan)
    meta = JointMeta(
        bucket_size=h,
        z_nodes=int(n_z),
        z_rule=z_rule,
        alpha=tuple(p.alpha for p in preps),
        beta=tuple(p.beta for p in preps),
        sigma_used=tuple(p.sigma_used for p in preps),
        copula_normal_correlation=preps[0].w_corr * preps[1].w_corr,
        total_mass=float(pmf.sum()),
        grid_mean=(float(mean1), float(mean2)),
        analytic_mean=tuple(p.mean_loss for p in preps),
        grid_sd=(sd1, sd2),
        grid_correlation=float(corr),
        notes=tuple(tuple(p.notes) for p in preps),
    )
    return BivariateSU(pmf=pmf, x1=x1, x2=x2, meta=meta)


# ---------------------------------------------------------------------------
# Analytic layer means (closed form via the incomplete beta).

def beta_limited_mean(y: float, alpha: float, beta: float) -> float:
    """``E[min(Y, y)]`` for ``Y ~ Beta(alpha, beta)`` and ``y`` in [0, 1]."""
    if y <= 0.0:
        return 0.0
    if y >= 1.0:
        return alpha / (alpha + beta)
    return y * (1.0 - betainc(alpha, beta, y)) + alpha / (alpha + beta) * betainc(alpha + 1.0, beta, y)


def layer_mean(prep: _Prepped, attach: float, limit: float) -> float:
    """Analytic ``E[min(max(X - attach, 0), limit)]`` for a unit's marginal."""
    if limit <= 0.0:
        return 0.0
    if prep.kind == _ATOM:
        return float(min(max(prep.mean_loss - attach, 0.0), limit))
    m = prep.max_loss
    lower = attach / m
    upper = (attach + limit) / m
    return m * (beta_limited_mean(upper, prep.alpha, prep.beta)
                - beta_limited_mean(lower, prep.alpha, prep.beta))


def analytic_view_mean(prep: _Prepped, terms: UnitTerms, view: str) -> float:
    """Analytic mean of a view of one unit's loss (the LEV identity)."""
    if view == "ground_up":
        return prep.mean_loss
    ded, lim = terms.ded, terms.limit
    gross = layer_mean(prep, ded, min(lim, prep.support))
    if view == "gross":
        return gross
    # A clause on gross is a layer on X: attach shifts by ded, the limit is
    # cut by whatever of the policy limit remains above the attachment.
    ceded = 0.0
    for clause in terms.clauses:
        effective = min(clause.limit, max(lim - clause.attach, 0.0))
        ceded += clause.share * layer_mean(prep, ded + clause.attach, effective)
    if view == "ceded":
        return ceded
    if view == "net":
        return gross - ceded
    raise TermsError(f"unknown view {view!r}")


# ---------------------------------------------------------------------------
# Multi-unit ELT: per-event engine and rate accumulation.

def _view_transform(cond: np.ndarray, bucket_size: float, terms: UnitTerms,
                    view: str) -> np.ndarray:
    """Map a ``(Q, n)`` conditional pmf on the ground-up grid through a view.

    Each source bucket's mass moves to ``v(k h)`` by the linear deposit, so
    the result (same length; views never exceed X) is mass-exact and
    mean-exact for the transformed point values.
    """
    if view == "ground_up":
        return cond
    n = cond.shape[1]
    values = terms.view(view, bucket_size * np.arange(n))
    transform = np.zeros((n, n))
    inv_h = 1.0 / bucket_size
    for k in range(n):
        _deposit(transform[k], float(values[k]), 1.0, inv_h)
    return cond @ transform


def _event_pmf(rows: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Average over z of the convolution across units of per-z pmfs."""
    length = sum(r.shape[1] - 1 for r in rows) + 1
    out = np.zeros(length)
    for q, w in enumerate(weights):
        acc = np.ones(1)
        for r in rows:
            acc = np.convolve(acc, r[q])
        out += w * acc
    return out


def _events(event_id: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` row ranges per event; rejects regrouping."""
    n = event_id.size
    ranges = []
    seen = set()
    start = 0
    while start < n:
        end = start + 1
        while end < n and event_id[end] == event_id[start]:
            end += 1
        if event_id[start] in seen:
            raise MultiUnitError(
                f"event {event_id[start]!r} rows are not contiguous (reappears at row {start})"
            )
        seen.add(event_id[start])
        ranges.append((start, end))
        start = end
    return ranges


def _quantile(xs: np.ndarray, ps: np.ndarray, q: float) -> float:
    cdf = np.cumsum(ps)
    return float(xs[min(int(np.searchsorted(cdf, q * cdf[-1])), xs.size - 1)])


def multiunit_views_reference(
    event_id,
    unit_id,
    rate,
    mean_loss,
    sd_indep,
    sd_corr,
    max_loss,
    *,
    terms=None,
    bucket_size,
    grid_length=None,
    views=VIEWS,
    z_nodes=32,
    z_rule="midpoint",
    normalize=False,
    variance_policy="clamp",
    clamp_factor=DEFAULT_CLAMP_FACTOR,
    roll_up_diagnostic=False,
):
    """Total event severity by view for a multi-unit ELT; return ``(views, meta)``.

    Parameters
    ----------
    event_id, unit_id : array_like
        Row identifiers. Rows of one event must be contiguous (validated,
        never sorted for the caller); a unit may appear once per event.
    rate : array_like
        Event rate, constant within an event (validated).
    mean_loss, sd_indep, sd_corr, max_loss : array_like
        Per-unit ELT fields in loss currency units.
    terms : str | sequence of str | Mapping | None
        DecL-style unit terms (:func:`eltq.terms.parse_terms`); units
        without a line get no terms (gross = ground_up, ceded = 0).
    bucket_size : float
        Grid spacing h. Every monetary breakpoint in ``terms`` must be a
        multiple of h.
    grid_length : int, optional
        Number of grid nodes; auto-derived from the largest event span
        (each unit's support rounded up to the grid, summed) when None.
        Insufficient explicit grids raise ``GridOverflowError``.
    views : sequence of str
        Views to return, any of ground_up, gross, ceded, net.
    z_nodes, z_rule
        Shared-factor quadrature (see :func:`z_quadrature`).
    normalize : bool
        Divide every view by the total contributing rate.
    variance_policy : {"clamp", "error", "drop", "atom"}
        Treatment of a unit row with inadmissible beta variance.
    roll_up_diagnostic : bool
        Also compute, per event with two or more contributing units, the
        sd and 99th-percentile ratios of the exact ground-up total against
        the rolled single-row beta (the additivity note's diagnostic).

    Returns
    -------
    views : dict[str, numpy.ndarray]
        View name -> length-n float64 rate-weight array on the grid i * h.
    meta : MultiMeta
    """
    event_id = np.asarray(event_id, dtype=object)
    unit_id = np.asarray(unit_id, dtype=object)
    columns = {
        "rate": np.asarray(rate, dtype=np.float64),
        "mean_loss": np.asarray(mean_loss, dtype=np.float64),
        "sd_indep": np.asarray(sd_indep, dtype=np.float64),
        "sd_corr": np.asarray(sd_corr, dtype=np.float64),
        "max_loss": np.asarray(max_loss, dtype=np.float64),
    }
    n_rows = event_id.size
    if unit_id.size != n_rows:
        raise EltqError(f"column unit_id has {unit_id.size} rows, expected {n_rows}")
    for name, arr in columns.items():
        if arr.size != n_rows:
            raise EltqError(f"column {name} has {arr.size} rows, expected {n_rows}")
    h = float(bucket_size)
    if not (h > 0.0) or not math.isfinite(h):
        raise EltqError(f"bucket_size must be finite and positive, got {h}")
    views = tuple(views)
    for view in views:
        if view not in VIEWS:
            raise EltqError(f"unknown view {view!r}; expected one of {VIEWS}")
    if not views:
        raise EltqError("at least one view is required")
    if variance_policy not in VARIANCE_POLICIES:
        raise EltqError(f"unknown variance_policy {variance_policy!r}")
    if not (0.0 < clamp_factor < 1.0):
        raise EltqError(f"clamp_factor must be in (0, 1), got {clamp_factor}")
    z, z_weights = z_quadrature(z_nodes, z_rule)

    try:
        term_table = parse_terms(terms)
        for unit_terms in term_table.values():
            check_grid_alignment(unit_terms, h)
    except TermsError as error:
        raise EltqError(str(error)) from error
    unit_order: dict[str, None] = {}
    diagnostics: dict[str, RowIssues] = {}

    # Pass 1: group and classify. Grid coverage needs every event's span
    # before any deposit happens: each contributing unit's pmf has
    # grid_length_for(support, h) nodes, so the convolution across units has
    # span + 1 nodes with span = sum(grid_length_for(support, h) - 1).
    # Summing supports in money first undercounts off-grid supports.
    ranges = _events(event_id)
    preps: list[_Prepped | None] = [None] * n_rows
    event_span = np.zeros(len(ranges), dtype=np.int64)
    event_live = np.zeros(len(ranges), dtype=bool)
    rates = columns["rate"]
    valid_rows = 0
    for e, (start, end) in enumerate(ranges):
        eid = event_id[start]
        if not np.all(rates[start:end] == rates[start]):
            raise MultiUnitError(f"event {eid!r}: rate varies across its rows")
        units_here = set()
        for j in range(start, end):
            uid = str(unit_id[j])
            if uid in units_here:
                raise MultiUnitError(f"event {eid!r}: unit {uid!r} appears twice")
            units_here.add(uid)
            unit_order.setdefault(uid, None)
        lam = rates[start]
        if not math.isfinite(lam) or lam < 0.0:
            category = "nonfinite" if not math.isfinite(lam) else "negative_field"
            for j in range(start, end):
                _note(diagnostics, category, f"{eid}/{unit_id[j]}")
            continue
        if lam == 0.0:
            for j in range(start, end):
                _note(diagnostics, "zero_rate", f"{eid}/{unit_id[j]}")
            continue
        span = 0
        for j in range(start, end):
            unit = Unit(columns["mean_loss"][j], columns["sd_indep"][j],
                        columns["sd_corr"][j], columns["max_loss"][j])
            prep = prepare_unit(unit, f"{eid}/{unit_id[j]}", variance_policy,
                                clamp_factor, diagnostics)
            if prep.kind == _DROP:
                continue
            preps[j] = prep
            valid_rows += 1
            span += grid_length_for(prep.support, h) - 1
        if any(preps[j] is not None for j in range(start, end)):
            event_live[e] = True
            event_span[e] = span

    max_span = int(event_span.max()) if len(ranges) else 0
    required = max_span + 1
    if grid_length is None:
        n = required
    else:
        n = int(grid_length)
        if n < 1:
            raise EltqError(f"grid_length must be >= 1, got {n}")
        if required > n:
            examples = [str(event_id[start]) for e, (start, _) in enumerate(ranges)
                        if event_live[e] and event_span[e] + 1 > n][:MAX_EXAMPLES]
            raise GridOverflowError(
                f"grid of {n} nodes at spacing {h} covers only {(n - 1) * h}, "
                f"but the largest event spans {max_span} buckets (each unit's "
                f"support rounded up to the grid, summed); "
                f"grid_length >= {required} required",
                required_length=required, examples=examples)

    # Pass 2: per-event pipeline, rate-weighted accumulation.
    out = {view: np.zeros(n) for view in views}
    rate_terms: list[float] = []
    aal_terms: list[float] = []
    analytic_terms = {view: [] for view in views}
    roll_up = [] if roll_up_diagnostic else None
    valid_events = 0
    default_terms: dict[str, UnitTerms] = {}
    for e, (start, end) in enumerate(ranges):
        if not event_live[e]:
            continue
        valid_events += 1
        eid = event_id[start]
        lam = rates[start]
        rate_terms.append(lam)
        live = [(j, preps[j]) for j in range(start, end) if preps[j] is not None]
        aal_terms.append(lam * math.fsum(p.mean_loss for _, p in live))
        conds = []
        unit_terms_list = []
        for j, prep in live:
            uid = str(unit_id[j])
            t = term_table.get(uid)
            if t is None:
                t = default_terms.setdefault(uid, UnitTerms(uid))
            unit_terms_list.append(t)
            conds.append(conditional_pmfs(prep, h, grid_length_for(prep.support, h), z))
        for view in views:
            rows = [_view_transform(c, h, t, view) for c, t in zip(conds, unit_terms_list)]
            pmf = _event_pmf(rows, z_weights)
            out[view][:pmf.size] += lam * pmf
            analytic_terms[view].append(
                lam * math.fsum(analytic_view_mean(p, t, view)
                                for (_, p), t in zip(live, unit_terms_list)))
        if roll_up is not None and len(live) >= 2:
            roll_up.append(_roll_up_entry(eid, live, columns, conds, z_weights, h))

    total_rate = math.fsum(rate_terms)
    if normalize:
        if not total_rate > 0.0:
            raise EltqError("cannot normalize: total contributing rate is zero")
        for view in views:
            out[view] /= total_rate
    grid = h * np.arange(n)
    scale = total_rate if normalize else 1.0
    view_mass = {view: float(out[view].sum()) for view in views}
    view_mean = {view: float(out[view] @ grid) for view in views}
    analytic = {view: math.fsum(analytic_terms[view]) for view in views}
    meta = MultiMeta(
        bucket_size=h,
        grid_length=n,
        z_nodes=int(z_nodes),
        z_rule=z_rule,
        views=views,
        normalized=bool(normalize),
        total_rate=total_rate,
        analytic_aal=math.fsum(aal_terms),
        view_mass=view_mass,
        view_mean=view_mean,
        analytic_view_mean=analytic,
        view_mean_error={view: view_mean[view] * scale - analytic[view] for view in views},
        input_rows=n_rows,
        valid_rows=valid_rows,
        events=len(ranges),
        valid_events=valid_events,
        unit_ids=tuple(unit_order),
        terms_echo={uid: t.text() for uid, t in term_table.items()},
        unused_terms=tuple(uid for uid in term_table if uid not in unit_order),
        diagnostics=diagnostics,
        roll_up=roll_up,
    )
    return out, meta


def _roll_up_entry(eid, live, columns, conds, z_weights, h):
    """Exact ground-up total versus the rolled single-row beta for one event."""
    exact = _event_pmf(conds, z_weights)
    xs = h * np.arange(exact.size)
    mu_r = math.fsum(columns["mean_loss"][j] for j, _ in live)
    m_r = math.fsum(columns["max_loss"][j] for j, _ in live)
    sd_c = math.fsum(columns["sd_corr"][j] for j, _ in live)
    sd_i = math.sqrt(math.fsum(columns["sd_indep"][j] ** 2 for j, _ in live))
    rolled_prep = prepare_unit(Unit(mu_r, sd_i, sd_c, m_r), f"{eid}/rolled")
    rolled = marginal_pmf(rolled_prep, h, exact.size)

    def sd(ps):
        mean = ps @ xs
        return math.sqrt(max(ps @ xs**2 - mean**2, 0.0))

    sd_exact, sd_rolled = sd(exact), sd(rolled)
    q_exact, q_rolled = _quantile(xs, exact, 0.99), _quantile(xs, rolled, 0.99)
    return {
        "event_id": eid,
        "units": len(live),
        "sd_exact": sd_exact,
        "sd_rolled": sd_rolled,
        "sd_ratio": sd_exact / sd_rolled if sd_rolled > 0.0 else math.nan,
        "q99_exact": q_exact,
        "q99_rolled": q_rolled,
        "q99_ratio": q_exact / q_rolled if q_rolled > 0.0 else math.nan,
    }
