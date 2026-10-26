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
sigma_i / M_i)``, or the two-point law on 0 and M with ``P(X_i = M_i) = mu_i
/ M_i`` when the stated variance is inadmissible (the Bernoulli policy, spec
section 4A); one shared standard normal ``Z_c`` per event occurrence and
an independent ``Z_i`` per unit give ``U_i = Phi((sd_corr Z_c + sd_indep
Z_i) / t_i)``, ``t_i = hypot(sd_indep, sd_corr)``, and ``X_i = M_i
F_i^{-1}(U_i)``. Conditional on ``Z_c = z`` the units are independent with
``P(X_i <= x | z) = Phi((Phi^{-1}(F_i(x / M_i)) - w_c z) / w_I)``; for a
two-point unit this is the mass ``1 - Phi((Phi^{-1}(1 - p) - w_c z) / w_I)``
at M and the rest at 0.

Discretization conventions, chosen to match eltq's deposit rule:

- a beta unit's conditional pmf is bucketed by cdf differences at half-bucket
  edges (exact in probability; bucket k holds ``((k - 1/2) h, (k + 1/2) h]``);
- point values (degenerate atoms, the two atoms of a two-point unit,
  comonotone units' per-z atoms, and every transformed view value) use the
  linear mass- and mean-preserving deposit;
- the shared factor is integrated by midpoint quantile nodes
  ``z_q = Phi^{-1}((q + 1/2) / Q)`` with equal weights, by Gauss-Hermite
  nodes and weights (``z_rule="gauss_hermite"``), or by the stratified rule
  (``z_rule="stratified"``, 2.6.0): ``Q`` equal-probability strata of
  ``Z_c``, each unit's conditional law averaged over the stratum exactly
  (through the bivariate normal cdf below loading one, through the marginal
  restricted to the stratum at loading one), the units independent given
  the stratum; every marginal is then exact at any ``Q``.

Fresh ``(Z_c, Z_i)`` per occurrence make occurrences iid, so the
rate-weighted mixture severity plus a Poisson frequency reproduces the
compound distribution exactly; accumulation across events is the ordinary
eltq pattern.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
from scipy.special import betainc, owens_t
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from eltq._meta import (
    STRATA_BODY_TOP, STRATA_TAIL_FLOOR, STRATA_TAIL_RATIO, JointMeta, MultiMeta, RowIssues,
)
from eltq.reference import (
    DEFAULT_CLAMP_FACTOR,
    MAX_EXAMPLES,
    EltqError,
    GridOverflowError,
    VarianceError,
    _deposit,
)
from eltq.terms import VIEWS, Clause, TermsError, UnitTerms, check_grid_alignment, parse_terms

VARIANCE_POLICIES = ("bernoulli", "clamp", "error", "drop", "atom")
Z_RULES = ("midpoint", "gauss_hermite", "stratified")
COUPLINGS = ("burke", "comonotone", "independent", "homer")

CLAMP_DEPRECATION = (
    "the clamp variance policy on the multi-unit route is deprecated from 2.5.0 "
    "and will be removed at 3.0.0; the default is now bernoulli (spec section 4A)"
)


def warn_if_clamp(variance_policy: str) -> None:
    """Emit the deprecation warning for the ``clamp`` policy (decision 5 of
    ``dev/done/plan-multiunit-coupling.md``)."""
    if variance_policy == "clamp":
        warnings.warn(CLAMP_DEPRECATION, DeprecationWarning, stacklevel=3)

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
_TWO_POINT = 4   # inadmissible variance under the Bernoulli policy: 0 or M


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

    @property
    def two_point_p(self) -> float:
        """``P(X = M) = mu / M`` of a two-point unit."""
        return self.mean_loss / self.max_loss

    @property
    def depends_on_factor(self) -> bool:
        """Whether the conditional law varies with the shared factor."""
        if self.kind == _COMONOTONE:
            return True
        return self.kind in (_BETA, _TWO_POINT) and self.w_corr > 0.0


def _note(diagnostics: dict, category: str, example) -> None:
    issues = diagnostics.setdefault(category, RowIssues())
    issues.count += 1
    if len(issues.examples) < MAX_EXAMPLES:
        issues.examples.append(example)


def prepare_unit(unit, label, variance_policy="bernoulli",
                 clamp_factor=DEFAULT_CLAMP_FACTOR, diagnostics=None) -> _Prepped:
    """Validate one unit and derive beta parameters and copula weights.

    Structural problems (nonfinite, negative, mean above max) raise
    ``EltqError`` when ``diagnostics`` is None (the two-unit joint) and are
    recorded as dropped rows otherwise (the multi-unit ELT route, mirroring
    the one-unit reference's categories). An inadmissible variance follows
    ``variance_policy``: ``bernoulli`` (the default from 2.5.0) makes the
    unit the two-point law on 0 and M with the maximizer's sd, keeping
    Burke's loadings; ``clamp`` is deprecated.
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
        if variance_policy == "bernoulli":
            prep.kind = _TWO_POINT
            prep.sigma_used = math.sqrt(v_max)
            prep.notes.append("variance_bernoulli")
            if diagnostics is not None:
                _note(diagnostics, "variance_bernoulli", label)
            t = math.hypot(s_i, s_c)
            prep.w_corr = s_c / t
            prep.w_indep = s_i / t
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


def couple(prep: _Prepped, coupling: str) -> _Prepped:
    """The unit under a coupling rule (spec section 4A): Burke's loadings as
    prepared, or every loading forced to one (``comonotone``) or zero
    (``independent``); a beta unit at loading one takes the comonotone kind
    and at loading zero the beta kind with no factor dependence. Atoms, drops
    and ``homer`` (which rolls the event up instead) leave the unit alone."""
    if coupling not in COUPLINGS:
        raise EltqError(f"unknown coupling {coupling!r}; expected one of {COUPLINGS}")
    if coupling in ("burke", "homer") or prep.kind in (_ATOM, _DROP):
        return prep
    out = dataclasses.replace(prep, notes=list(prep.notes))
    if coupling == "comonotone":
        if out.kind == _BETA:
            out.kind = _COMONOTONE
        out.w_corr, out.w_indep = 1.0, 0.0
    else:
        if out.kind == _COMONOTONE:
            out.kind = _BETA
        out.w_corr, out.w_indep = 0.0, 1.0
    return out


def comonotone_merge(rows, deposit) -> None:
    """Merge the cumulative probabilities of bucketed marginals on one grid:
    every unit is a nondecreasing function of one uniform, so between two
    consecutive breakpoints each unit sits in one bucket, and ``deposit(indices,
    mass)`` receives each combination with the interval's length. Each row's
    last cumulative value is taken as exactly one."""
    cums = []
    for r in rows:
        c = np.cumsum(np.asarray(r, dtype=np.float64))
        c[-1] = 1.0
        cums.append(c)
    idx = [0] * len(rows)
    u = 0.0
    while True:
        nxt = min((c[i] for c, i in zip(cums, idx)), default=1.0)
        deposit(tuple(idx), max(nxt - u, 0.0))
        if nxt >= 1.0:
            return
        u = nxt
        for j, c in enumerate(cums):
            while c[idx[j]] <= u and idx[j] + 1 < c.size:
                idx[j] += 1


def comonotone_sum(rows) -> np.ndarray:
    """The comonotone sum's bucket pmf: the indices add on the shared grid."""
    out = np.zeros(sum(len(r) - 1 for r in rows) + 1)

    def deposit(indices, mass):
        out[sum(indices)] += mass

    comonotone_merge(rows, deposit)
    return out


def rolled_unit(units) -> Unit:
    """Homer and Li 2017, section 3.2.2 (and Mitchell-Wallace et al. 2017,
    section 1.10.5.10): means, exposures and correlated sds added, the
    independent sds in quadrature."""
    return Unit(math.fsum(u.mean_loss for u in units),
                math.sqrt(math.fsum(u.sd_indep ** 2 for u in units)),
                math.fsum(u.sd_corr for u in units),
                math.fsum(u.max_loss for u in units))


def terms_on_rolled(terms: UnitTerms, share: float) -> UnitTerms:
    """Unit terms expressed on the rolled loss: with the allocation ``X_k = s
    X_R``, ``term_k(s x) = s term_k'(x)`` where every monetary breakpoint of
    ``term_k'`` is the original divided by ``s``."""
    return UnitTerms(terms.unit_id, terms.limit / share, terms.ded / share,
                     tuple(Clause(c.share, c.limit / share, c.attach / share) for c in terms.clauses))


def homer_view_values(unit_terms, shares, view: str, bucket_size: float, n: int) -> np.ndarray:
    """The value of a view of the event total under ``homer`` at each node of
    the rolled grid, ``sum_k term_k(s_k x)``; zero-mean units receive nothing."""
    x = bucket_size * np.arange(n)
    values = np.zeros(n)
    for t, s in zip(unit_terms, shares):
        if s > 0.0:
            values += t.view(view, s * x)
    return values


def homer_excess(rolled: _Prepped, shares, max_losses) -> float:
    """The expected allocation over the exposures, ``sum_k P((s_k X_R -
    M_k)^+)``, from the rolled row's limited expected values."""
    return math.fsum(s * layer_mean(rolled, m / s, math.inf)
                     for s, m in zip(shares, max_losses) if s > 0.0)


def z_quadrature(n_z: int, rule: str = "midpoint"):
    """Nodes and weights for integrating out the shared factor ``Z_c``.

    ``midpoint``: ``z_q = Phi^{-1}((q + 1/2) / n_z)`` with equal weights, the
    midpoint-quantile style of the one-unit specification. ``gauss_hermite``:
    probabilists' Gauss-Hermite nodes with weights normalized to sum to one.
    The stratified rule has no nodes and is refused here (see
    :class:`FactorRule`).
    """
    q_n = int(n_z)
    if q_n < 1:
        raise EltqError(f"z_nodes must be >= 1, got {n_z}")
    if rule == "midpoint":
        z = norm.ppf((np.arange(q_n) + 0.5) / q_n)
        return z, np.full(q_n, 1.0 / q_n)
    if rule == "gauss_hermite":
        return _gauss_hermite(q_n)
    if rule == "stratified":
        raise EltqError("the stratified rule has strata, not nodes; use FactorRule")
    raise EltqError(f"unknown z_rule {rule!r}; expected one of {Z_RULES}")


def stratum_edges(n_z: int) -> np.ndarray:
    """Boundaries in probability of the default stratified partition:
    ``n_z`` equal strata over ``[0, STRATA_BODY_TOP]``, then tail
    probabilities ``(1 - STRATA_BODY_TOP) / STRATA_TAIL_RATIO**k`` for ``k =
    1, 2, ...`` down to the floor, then 1 (mirrors ``stratum_edges`` in
    ``quadrature.rs``)."""
    body = STRATA_BODY_TOP * np.arange(n_z + 1) / n_z
    tails = []
    k = 1
    while True:
        tail = (1.0 - STRATA_BODY_TOP) * STRATA_TAIL_RATIO ** (-k)
        if tail <= STRATA_TAIL_FLOOR:
            break
        tails.append(1.0 - tail)
        k += 1
    return np.concatenate([body, tails, [1.0]])


@dataclass
class FactorRule:
    """The shared factor's rule as the conditional pmfs consume it: ``Q``
    nodes with weights (the node rules), or strata with their boundaries in
    probability (``edges_u``) and in ``z`` (``edges_z``, from ``-inf`` to
    ``+inf``) and their widths as weights (the stratified rule)."""

    rule: str
    nodes: np.ndarray
    weights: np.ndarray
    edges_u: np.ndarray
    edges_z: np.ndarray

    @classmethod
    def new(cls, n_z: int, rule: str = "stratified") -> FactorRule:
        """The rule with ``n_z`` nodes, or ``n_z`` body strata plus the tail
        ladder."""
        q_n = int(n_z)
        if rule == "stratified":
            if q_n < 1:
                raise EltqError(f"z_nodes must be >= 1, got {n_z}")
            return cls.from_edges(stratum_edges(q_n))
        z, w = z_quadrature(q_n, rule)
        return cls(rule, z, w, np.empty(0), np.empty(0))

    @classmethod
    def from_edges(cls, edges_u) -> FactorRule:
        """The stratified rule on explicit boundaries in probability,
        ascending from 0 to 1 (the convergence study's equal strata)."""
        edges_u = np.asarray(edges_u, dtype=np.float64)
        return cls("stratified", np.empty(0), np.diff(edges_u), edges_u, norm.ppf(edges_u))

    @classmethod
    def single(cls) -> FactorRule:
        """One node at ``z = 0`` with weight one: a unit independent of the
        factor."""
        return cls("midpoint", np.zeros(1), np.ones(1), np.empty(0), np.empty(0))

    def __len__(self) -> int:
        return self.weights.size

    @property
    def stratified(self) -> bool:
        return self.rule == "stratified"


def bivariate_normal_cdf(h, k, rho: float):
    """``Phi_2(h, k; rho) = P(X <= h, Y <= k)`` for standard normals with
    correlation ``rho``, elementwise over broadcast ``h`` and ``k``.

    Owen (1956): ``Phi_2 = (Phi(h) + Phi(k)) / 2 - T(h, a_h) - T(k, a_k) -
    beta`` with Owen's T function, ``a_h = (k - rho h) / (h sqrt(1 -
    rho^2))``, ``a_k`` likewise, and ``beta`` 1/2 when ``h k < 0`` or ``h k =
    0`` with ``h + k < 0``, else 0; ``T(0, a) = atan(a) / (2 pi)`` and the
    infinities and ``rho = +-1`` are the exact limits. Agrees with
    ``scipy.stats.multivariate_normal`` to 2e-16.
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    h, k = np.broadcast_arrays(h, k)
    if rho >= 1.0:
        return norm.cdf(np.minimum(h, k))
    if rho <= -1.0:
        return np.maximum(norm.cdf(h) + norm.cdf(k) - 1.0, 0.0)
    s = math.sqrt((1.0 - rho) * (1.0 + rho))
    out = np.empty(h.shape)
    finite = np.isfinite(h) & np.isfinite(k)
    # The infinities: -inf on either axis gives 0, +inf the other marginal.
    out[~finite] = np.where(
        (h[~finite] == -np.inf) | (k[~finite] == -np.inf), 0.0,
        np.where(h[~finite] == np.inf, norm.cdf(k[~finite]), norm.cdf(h[~finite])))
    hf, kf = h[finite], k[finite]

    def term(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (y - rho * x) / (x * s)
        t = np.where(x == 0.0, 0.0, owens_t(np.where(x == 0.0, 1.0, x), np.where(x == 0.0, 0.0, a)))
        # T(0, a) with a = sign(y) inf: +-1/4; both zero: the limit below.
        return np.where(x == 0.0, np.where(y > 0.0, 0.25, np.where(y < 0.0, -0.25, 0.0)), t)

    beta = np.where((hf * kf < 0.0) | ((hf * kf == 0.0) & (hf + kf < 0.0)), 0.5, 0.0)
    value = 0.5 * (norm.cdf(hf) + norm.cdf(kf)) - term(hf, kf) - term(kf, hf) - beta
    both_zero = (hf == 0.0) & (kf == 0.0)
    value = np.where(both_zero, 0.25 + math.asin(rho) / (2.0 * math.pi), value)
    out[finite] = np.clip(value, 0.0, 1.0)
    return out


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
    half-bucket edges; the linear deposit for an atom; the two atoms of a
    two-point unit)."""
    if prep.kind == _ATOM:
        row = np.zeros(n)
        _deposit(row, prep.mean_loss, 1.0, 1.0 / bucket_size)
        return row
    if prep.kind == _TWO_POINT:
        row = np.zeros(n)
        p = prep.two_point_p
        _deposit(row, 0.0, 1.0 - p, 1.0 / bucket_size)
        _deposit(row, prep.max_loss, p, 1.0 / bucket_size)
        return row
    edges = (np.arange(n + 1) - 0.5) * bucket_size
    return np.diff(beta_dist.cdf(edges / prep.max_loss, prep.alpha, prep.beta))


def two_point_conditional(p: float, w_corr: float, w_indep: float, z: np.ndarray) -> np.ndarray:
    """``P(X = M | Z_c = z)`` of a two-point unit: ``1 - Phi((Phi^{-1}(1 - p)
    - w_corr z) / w_indep)``, the step ``1{z > Phi^{-1}(1 - p)}`` at loading
    one."""
    y0 = norm.ppf(1.0 - p)
    z = np.asarray(z, dtype=np.float64)
    if w_indep == 0.0:
        return (z > y0).astype(np.float64)
    return norm.sf((y0 - w_corr * z) / w_indep)


def edge_cdf(prep: _Prepped, bucket_size: float, n: int) -> np.ndarray:
    """The unit's marginal distribution function at the ``n + 1`` half-bucket
    edges: 0 below zero, 1 at or above M, the beta or the two-point step
    between."""
    edges = (np.arange(n + 1) - 0.5) * bucket_size
    if prep.kind == _TWO_POINT:
        return np.where(edges < 0.0, 0.0, np.where(edges < prep.max_loss, 1.0 - prep.two_point_p, 1.0))
    return beta_dist.cdf(edges / prep.max_loss, prep.alpha, prep.beta)


def conditional_pmfs(prep: _Prepped, bucket_size: float, n: int, z) -> np.ndarray:
    """Conditional bucket pmf of one unit for each node or stratum.

    ``z`` is a :class:`FactorRule` or an array of nodes. Returns a ``(Q, n)``
    array whose row q is the pmf of the unit's loss on buckets ``0, h, ...,
    (n - 1) h`` given ``Z_c = z[q]`` (a node rule) or given ``Phi(Z_c)`` in
    the q-th stratum (the stratified rule). Every row sums to one. Beta units
    use cdf differences at half-bucket edges; atoms, two-point units and
    comonotone units use the linear deposit of their point values.

    Under the stratified rule the stratum average of the conditional
    distribution function at an edge score ``y`` is ``(Phi_2(y, z_{s+1}; w)
    - Phi_2(y, z_s; w)) / (b_s - a_s)`` for a loading ``w`` below one, so the
    rows telescope to the marginal exactly; at loading one the stratum's law
    is the marginal restricted to the stratum, ``(clip(F(x), a, b) - a) /
    (b - a)``.
    """
    if isinstance(z, FactorRule):
        rule = z
        z = rule.nodes
    else:
        z = np.asarray(z, dtype=np.float64)
        rule = None
    q_n = len(rule) if rule is not None else z.size
    inv_h = 1.0 / bucket_size
    pmf = np.zeros((q_n, n))
    if prep.kind == _ATOM:
        row = np.zeros(n)
        _deposit(row, prep.mean_loss, 1.0, inv_h)
        pmf[:] = row
        return pmf
    if rule is not None and rule.stratified:
        f = edge_cdf(prep, bucket_size, n)
        inv_width = (1.0 / rule.weights)[:, None]
        if prep.kind == _COMONOTONE or prep.w_indep == 0.0:
            a, b = rule.edges_u[:-1, None], rule.edges_u[1:, None]
            clipped = np.clip(f[None, :], a, b)
            return inv_width * np.diff(clipped, axis=1)
        y = norm.ppf(f)
        # U(y_e, z_s) = P(V <= y_e, Z_c > z_s) = Phi_2(y_e, -z_s; -w) over
        # the (Q + 1) x (n + 1) table of boundaries and edges, the upper-tail
        # form the engine uses so that the two agree in the narrow tail
        # strata; U(y, -inf) = Phi(y), U(y, +inf) = 0, U(+inf, z) = Phi(-z).
        table = np.zeros((q_n + 1, n + 1))
        table[0] = norm.cdf(y)
        for s in range(1, q_n):
            table[s] = bivariate_normal_cdf(y, -rule.edges_z[s], -prep.w_corr)
            table[s, -1] = norm.sf(rule.edges_z[s])
        averaged = inv_width * (table[:-1] - table[1:])
        return np.maximum(np.diff(averaged, axis=1), 0.0)
    if prep.kind == _TWO_POINT:
        at_max = two_point_conditional(prep.two_point_p, prep.w_corr, prep.w_indep, z)
        for q, a in enumerate(at_max):
            _deposit(pmf[q], 0.0, 1.0 - a, inv_h)
            _deposit(pmf[q], prep.max_loss, a, inv_h)
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


def secondary_two_units(units, bucket_size, n1, n2, *, n_z=32, z_rule="stratified",
                        variance_policy="bernoulli", clamp_factor=DEFAULT_CLAMP_FACTOR,
                        coupling="burke"):
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
        Shared-factor nodes, or body strata under the stratified rule
        (:class:`FactorRule`).
    z_rule : {"midpoint", "gauss_hermite", "stratified"}, optional
        Rule for the shared factor (:class:`FactorRule`).
    variance_policy : {"bernoulli", "clamp", "error", "atom"}, optional
        Treatment of a unit whose sd is inadmissible for a beta: the
        two-point law on 0 and M (the default), clamp sigma to
        ``clamp_factor`` times the bound (deprecated; removed at 3.0.0),
        raise, or collapse the unit to an atom at its mean.
    clamp_factor : float, optional
        Factor in (0, 1) applied to the sd bound under the clamp policy.
    coupling : {"burke", "comonotone", "independent", "homer"}, optional
        The coupling rule (spec section 4A). ``comonotone`` and ``homer`` are
        computed exactly with no factor rule: the merged breakpoints of the
        two bucketed marginals, or the rolled row deposited at the pair of its
        allocations (uncapped, so a unit's grid must reach ``s_k M_R``).

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
    if variance_policy not in ("bernoulli", "clamp", "error", "atom"):
        raise EltqError(f"unknown variance_policy {variance_policy!r} for the joint")
    warn_if_clamp(variance_policy)
    if not (0.0 < clamp_factor < 1.0):
        raise EltqError(f"clamp_factor must be in (0, 1), got {clamp_factor}")
    if coupling not in COUPLINGS:
        raise EltqError(f"unknown coupling {coupling!r}; expected one of {COUPLINGS}")
    rule = FactorRule.new(n_z, z_rule)
    w = rule.weights
    lengths = (int(n1), int(n2))
    preps = []
    for i, (unit, n) in enumerate(zip(units, lengths), start=1):
        if n < 1:
            raise EltqError(f"n{i} must be >= 1, got {n}")
        preps.append(couple(prepare_unit(unit, f"unit {i}", variance_policy, clamp_factor), coupling))
    units = [Unit(*(float(v) for v in u)) for u in units]
    rolled = rolled_unit(units)
    rolled_prep = (prepare_unit(rolled, "rolled", variance_policy, clamp_factor)
                   if coupling == "homer" else None)
    shares = ([u.mean_loss / rolled.mean_loss for u in units] if rolled.mean_loss > 0.0
              else [0.0, 0.0])
    for i, (prep, n) in enumerate(zip(preps, lengths), start=1):
        # Under homer unit k receives s_k X_R up to s_k M_R, beyond M_k when
        # its share of the mean exceeds its share of the exposure.
        reach = shares[i - 1] * rolled_prep.support if rolled_prep is not None else prep.max_loss
        if reach > (n - 1) * h:
            raise GridOverflowError(
                f"unit {i}: grid of {n} nodes at spacing {h} covers only "
                f"{(n - 1) * h}, but the unit reaches {reach}; "
                f"n{i} >= {grid_length_for(reach, h)} required",
                required_length=grid_length_for(reach, h))

    if coupling == "homer":
        # Each bucket of the rolled row at the pair of its allocations,
        # deposited linearly on each axis.
        n_r = grid_length_for(rolled_prep.support, h)
        base = marginal_pmf(rolled_prep, h, n_r)
        pmf = np.zeros(lengths)
        inv_h = 1.0 / h
        for i, p in enumerate(base):
            if p == 0.0:
                continue
            rows = []
            for s, n in zip(shares, lengths):
                row = np.zeros(n)
                _deposit(row, s * i * h, 1.0, inv_h)
                rows.append(row)
            pmf += p * np.outer(rows[0], rows[1])
    elif coupling == "comonotone":
        marginals = [marginal_pmf(p, h, n) for p, n in zip(preps, lengths)]
        pmf = np.zeros(lengths)

        def deposit(indices, mass):
            pmf[indices] += mass

        comonotone_merge(marginals, deposit)
    else:
        cond1 = conditional_pmfs(preps[0], h, lengths[0], rule)
        cond2 = conditional_pmfs(preps[1], h, lengths[1], rule)
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
        coupling=coupling,
        z_nodes=int(n_z),
        z_rule=z_rule,
        z_strata=len(rule),
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


def exact_two_unit_joint(units, bucket_size, n1, n2, *, variance_policy="bernoulli",
                         clamp_factor=DEFAULT_CLAMP_FACTOR) -> np.ndarray:
    """The joint bucket pmf of two units of one event with no quadrature over
    the factor: the bivariate normal rectangle probabilities of the two
    units' normal scores at the bucket edges (spec section 4A, the reference
    of the convergence study and of spec test 28).

    The scores ``(V_1, V_2)`` are standard normal with correlation ``w_1
    w_2``, and ``X_i`` lies in bucket ``k`` exactly when ``V_i`` lies between
    the edge scores ``y_i(e_k)`` and ``y_i(e_{k+1})``, so the cell
    probability is the second difference of ``Phi_2`` over the edge scores.
    Exact for beta and two-point units at any loadings, comonotone pairs
    included (``rho = 1``); an atom is an outer product with its deposited
    row.
    """
    h = float(bucket_size)
    preps = [prepare_unit(u, f"unit {i}", variance_policy, clamp_factor)
             for i, u in enumerate(units, start=1)]
    lengths = (int(n1), int(n2))
    if any(p.kind == _ATOM for p in preps):
        rows = [marginal_pmf(p, h, n) if p.kind == _ATOM else None for p, n in zip(preps, lengths)]
        if all(r is not None for r in rows):
            return np.outer(rows[0], rows[1])
        other = 1 if rows[0] is not None else 0
        rows[other] = marginal_pmf(preps[other], h, lengths[other])
        return np.outer(rows[0], rows[1])
    scores = [norm.ppf(edge_cdf(p, h, n)) for p, n in zip(preps, lengths)]
    rho = preps[0].w_corr * preps[1].w_corr
    table = bivariate_normal_cdf(scores[0][:, None], scores[1][None, :], rho)
    return np.maximum(np.diff(np.diff(table, axis=0), axis=1), 0.0)


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
    if prep.kind == _TWO_POINT:
        # The loss is M with probability p and 0 otherwise.
        return prep.two_point_p * float(min(max(prep.max_loss - attach, 0.0), limit))
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
    return _transform_values(cond, bucket_size, terms.view(view, bucket_size * np.arange(n)))


def _transform_values(cond: np.ndarray, bucket_size: float, values) -> np.ndarray:
    """Move each source bucket's mass to ``values[k]`` by the linear deposit
    (the composed allocation map of the ``homer`` coupling, 2.8.0)."""
    n = cond.shape[1]
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
    z_rule="stratified",
    normalize=False,
    variance_policy="bernoulli",
    clamp_factor=DEFAULT_CLAMP_FACTOR,
    roll_up_diagnostic=False,
    coupling="burke",
    pair=None,
    first_moments=False,
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
        Shared-factor rule and its Q (see :class:`FactorRule`): ``midpoint``,
        ``gauss_hermite`` or ``stratified``.
    normalize : bool
        Divide every view by the total contributing rate.
    variance_policy : {"bernoulli", "clamp", "error", "drop", "atom"}
        Treatment of a unit row with inadmissible beta variance: the
        two-point law of spec section 4A (the default from 2.5.0), the
        deprecated clamp (removed at 3.0.0), an error, a dropped row, or an
        atom at the mean.
    roll_up_diagnostic : bool
        Also compute, per event with two or more contributing units, the
        sd and 99th-percentile ratios of the exact ground-up total against
        the rolled single-row beta (the additivity note's diagnostic).
    coupling : {"burke", "comonotone", "independent", "homer"}
        The coupling rule of spec section 4A, applied to every event.
        ``burke`` and ``independent`` set the loadings and run the factor
        rule; ``comonotone`` is the comonotone sum of the bucketed view
        marginals, exact, and ``homer`` the rolled row bucketed directly and
        pushed through the composed allocation map per view (uncapped, as
        published), both with no factor rule.
    pair : tuple of two unit ids, optional
        Also return the rate-weighted joint occurrence severity of the two
        units for the first view in ``views`` (spec section 4A): an event
        exposing one of them puts the other at zero, one exposing neither
        contributes nothing.
    first_moments : bool
        Also return, for every unit, the rate-weighted first-moment vector
        ``m_u(y) = sum_j lambda_j P(X_uj; X_j = y)`` on the grid of the first
        view's total.

    Returns
    -------
    views : dict[str, numpy.ndarray]
        View name -> length-n float64 rate-weight array on the grid i * h.
    meta : MultiMeta
    extras : dict
        Only when ``pair`` or ``first_moments`` is requested: ``"pair"`` is
        ``(pmf, x_u, x_w, rate, events)`` and ``"first_moments"`` is ``(dict
        unit id -> array, identity_error)``.
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
    warn_if_clamp(variance_policy)
    if coupling not in COUPLINGS:
        raise EltqError(f"unknown coupling {coupling!r}; expected one of {COUPLINGS}")
    rule = FactorRule.new(z_nodes, z_rule)
    z_weights = rule.weights

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
    reach: dict[str, float] = {}
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
        live_rows = 0
        exposure = 0.0
        mean_sum = 0.0
        for j in range(start, end):
            unit = Unit(columns["mean_loss"][j], columns["sd_indep"][j],
                        columns["sd_corr"][j], columns["max_loss"][j])
            prep = prepare_unit(unit, f"{eid}/{unit_id[j]}", variance_policy,
                                clamp_factor, diagnostics)
            if prep.kind == _DROP:
                continue
            preps[j] = prep
            valid_rows += 1
            live_rows += 1
            exposure += unit.max_loss
            mean_sum += prep.mean_loss
            span += grid_length_for(prep.support, h) - 1
            uid = str(unit_id[j])
            reach[uid] = max(reach.get(uid, 0.0), prep.support)
        # Under homer the rolled row's beta reaches the summed exposures,
        # beyond the supports of any atoms among the units, and each unit's
        # allocation its share of them.
        if coupling == "homer" and live_rows >= 2:
            span = max(span, grid_length_for(exposure, h) - 1)
            if mean_sum > 0.0:
                for j in range(start, end):
                    if preps[j] is not None:
                        uid = str(unit_id[j])
                        reach[uid] = max(reach[uid], preps[j].mean_loss / mean_sum * exposure)
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
    if pair is not None:
        pair = tuple(str(u) for u in pair)
        if len(pair) != 2:
            raise EltqError(f"pair must name two units, got {pair}")
        pair_n = tuple(grid_length_for(reach.get(u, 0.0), h) for u in pair)
        pair_pmf = np.zeros(pair_n)
        pair_rate_terms: list[float] = []
        pair_events = 0
    moments = {uid: np.zeros(n) for uid in unit_order} if first_moments else None
    identity_error = 0.0
    rate_terms: list[float] = []
    aal_terms: list[float] = []
    analytic_terms = {view: [] for view in views}
    roll_up = [] if roll_up_diagnostic else None
    valid_events = 0
    one_unit_events = 0
    excess_terms: list[float] = []
    default_terms: dict[str, UnitTerms] = {}
    for e, (start, end) in enumerate(ranges):
        if not event_live[e]:
            continue
        valid_events += 1
        eid = event_id[start]
        lam = rates[start]
        rate_terms.append(lam)
        live = [(j, couple(preps[j], coupling)) for j in range(start, end) if preps[j] is not None]
        aal_terms.append(lam * math.fsum(p.mean_loss for _, p in live))
        conds = []
        unit_terms_list = []
        # An event with one contributing unit never enters the mixture: its
        # row is the unit's marginal bucketed by cdf differences at the
        # half-bucket edges (spec section 4A, requirement (i)). The comonotone
        # and homer couplings need no factor rule either: every unit carries
        # its marginal as one row (comonotone, merged below) or the event is
        # rolled up (homer, below).
        one_unit = len(live) == 1
        one_unit_events += one_unit
        exact_path = not one_unit and coupling in ("comonotone", "homer")
        weights = np.ones(1) if (one_unit or exact_path) else z_weights
        for j, prep in live:
            uid = str(unit_id[j])
            t = term_table.get(uid)
            if t is None:
                t = default_terms.setdefault(uid, UnitTerms(uid))
            unit_terms_list.append(t)
            n_unit = grid_length_for(prep.support, h)
            conds.append(marginal_pmf(prep, h, n_unit)[None, :] if (one_unit or exact_path)
                         else conditional_pmfs(prep, h, n_unit, rule))
        if exact_path and coupling == "homer":
            # The rolled row bucketed directly, pushed through the composed
            # allocation map per view (uncapped, as published).
            specs = [Unit(columns["mean_loss"][j], columns["sd_indep"][j],
                          columns["sd_corr"][j], columns["max_loss"][j]) for j, _ in live]
            rolled = rolled_unit(specs)
            rolled_prep = prepare_unit(rolled, f"{eid}/rolled", variance_policy, clamp_factor)
            shares = [u.mean_loss / rolled.mean_loss if rolled.mean_loss > 0.0 else 0.0 for u in specs]
            n_r = grid_length_for(rolled_prep.support, h)
            base = marginal_pmf(rolled_prep, h, n_r)
            for view in views:
                if view == "ground_up":
                    pmf = base
                else:
                    values = homer_view_values(unit_terms_list, shares, view, h, n_r)
                    pmf = _transform_values(base[None, :], h, values)[0]
                out[view][:pmf.size] += lam * pmf
                # Each unit's view acts on its allocation s_k X_R, a layer of
                # the rolled row with the breakpoints divided by s_k.
                analytic_terms[view].append(
                    lam * math.fsum(s * analytic_view_mean(rolled_prep, terms_on_rolled(t, s), view)
                                    for s, t in zip(shares, unit_terms_list) if s > 0.0))
            excess_terms.append(lam * homer_excess(rolled_prep, shares, [u.max_loss for u in specs]))
        else:
            for view in views:
                rows = [_view_transform(c, h, t, view) for c, t in zip(conds, unit_terms_list)]
                pmf = comonotone_sum([r[0] for r in rows]) if exact_path else _event_pmf(rows, weights)
                out[view][:pmf.size] += lam * pmf
                analytic_terms[view].append(
                    lam * math.fsum(analytic_view_mean(p, t, view)
                                    for (_, p), t in zip(live, unit_terms_list)))
        if roll_up is not None and len(live) >= 2:
            roll_up.append(_roll_up_entry(eid, live, columns, conds, weights, h))
        if pair is not None or moments is not None:
            _occurrence_outputs(
                views[0], live, unit_id, unit_terms_list, conds, weights, h, lam,
                exact_path and coupling == "homer", exact_path,
                (rolled_prep, shares) if exact_path and coupling == "homer" else None,
                pair, pair_pmf if pair is not None else None, moments,
            )
            if pair is not None and any(str(unit_id[j]) in pair for j, _ in live):
                pair_rate_terms.append(lam)
                pair_events += 1
            if moments is not None:
                identity_error = max(identity_error, lam * _moment_identity_error(
                    views[0], live, unit_id, unit_terms_list, conds, weights, h,
                    exact_path and coupling == "homer", exact_path,
                    (rolled_prep, shares) if exact_path and coupling == "homer" else None))

    total_rate = math.fsum(rate_terms)
    if normalize:
        if not total_rate > 0.0:
            raise EltqError("cannot normalize: total contributing rate is zero")
        for view in views:
            out[view] /= total_rate
        if pair is not None:
            pair_pmf /= total_rate
        if moments is not None:
            for m in moments.values():
                m /= total_rate
    grid = h * np.arange(n)
    scale = total_rate if normalize else 1.0
    view_mass = {view: float(out[view].sum()) for view in views}
    view_mean = {view: float(out[view] @ grid) for view in views}
    analytic = {view: math.fsum(analytic_terms[view]) for view in views}
    meta = MultiMeta(
        bucket_size=h,
        grid_length=n,
        coupling=coupling,
        z_nodes=int(z_nodes),
        z_rule=z_rule,
        z_strata=len(rule),
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
        one_unit_events=one_unit_events,
        homer_excess_mean=math.fsum(excess_terms) if coupling == "homer" else None,
        unit_ids=tuple(unit_order),
        terms_echo={uid: t.text() for uid, t in term_table.items()},
        unused_terms=tuple(uid for uid in term_table if uid not in unit_order),
        diagnostics=diagnostics,
        roll_up=roll_up,
    )
    if pair is None and moments is None:
        return out, meta
    extras = {}
    if pair is not None:
        extras["pair"] = (pair_pmf, h * np.arange(pair_n[0]), h * np.arange(pair_n[1]),
                          math.fsum(pair_rate_terms), pair_events)
    if moments is not None:
        extras["first_moments"] = (moments, identity_error)
    return out, meta, extras


def _first_view_rows(view, live, unit_terms_list, conds, h, homer):
    """Per unit of the event, its rows for the view: the view maps of its
    conditional rows, or under homer its allocation values at the rolled
    nodes (``None`` for the rows, the values returned second)."""
    if homer is not None:
        rolled_prep, shares = homer
        n_r = grid_length_for(rolled_prep.support, h)
        base = marginal_pmf(rolled_prep, h, n_r)
        x = h * np.arange(n_r)
        values = [t.view(view, s * x) if s > 0.0 else np.zeros(n_r) for t, s in zip(unit_terms_list, shares)]
        return None, (base, values)
    return [_view_transform(c, h, t, view) for c, t in zip(conds, unit_terms_list)], None


def _occurrence_outputs(view, live, unit_id, unit_terms_list, conds, weights, h, lam, homer_path,
                        exact_path, homer, pair, pair_pmf, moments):
    """Add one event's rate-weighted contribution to the pair joint and to
    the first-moment vectors of the first view (spec section 4A)."""
    uids = [str(unit_id[j]) for j, _ in live]
    rows, homer_values = _first_view_rows(view, live, unit_terms_list, conds, h, homer)
    inv_h = 1.0 / h
    if homer_path:
        base, values = homer_values
        if pair is not None:
            iu = uids.index(pair[0]) if pair[0] in uids else None
            iw = uids.index(pair[1]) if pair[1] in uids else None
            if iu is not None or iw is not None:
                for i, p in enumerate(base):
                    if p == 0.0:
                        continue
                    row_u, row_w = np.zeros(pair_pmf.shape[0]), np.zeros(pair_pmf.shape[1])
                    _deposit(row_u, values[iu][i] if iu is not None else 0.0, 1.0, inv_h)
                    _deposit(row_w, values[iw][i] if iw is not None else 0.0, 1.0, inv_h)
                    pair_pmf += lam * p * np.outer(row_u, row_w)
        if moments is not None:
            # Each rolled bucket's mass at the two deposit nodes of the view
            # total, split among the units by their allocation values' shares
            # of it, so that the moments at a node sum to the node's loss
            # times its mass as the total's own deposit has it.
            total = sum(values)
            grid = h * np.arange(moments[uids[0]].size)
            for i, p in enumerate(base):
                if p == 0.0 or total[i] <= 0.0:
                    continue
                split = np.zeros(grid.size)
                _deposit(split, float(total[i]), 1.0, inv_h)
                for uid, v in zip(uids, values):
                    moments[uid] += lam * p * (v[i] / total[i]) * split * grid
        return
    if exact_path:
        marginals = [r[0] for r in rows]
        if pair is not None:
            iu = uids.index(pair[0]) if pair[0] in uids else None
            iw = uids.index(pair[1]) if pair[1] in uids else None
            if iu is not None or iw is not None:
                def deposit_pair(indices, mass):
                    a = indices[iu] if iu is not None else 0
                    b = indices[iw] if iw is not None else 0
                    pair_pmf[a, b] += lam * mass

                comonotone_merge(marginals, deposit_pair)
        if moments is not None:
            def deposit_moments(indices, mass):
                y = sum(indices)
                for uid, k in zip(uids, indices):
                    moments[uid][y] += lam * mass * k * h

            comonotone_merge(marginals, deposit_moments)
        return
    # The factor path (one-unit events included, with their single row).
    if pair is not None:
        iu = uids.index(pair[0]) if pair[0] in uids else None
        iw = uids.index(pair[1]) if pair[1] in uids else None
        if iu is not None or iw is not None:
            def rows_of(i, q):
                if i is None:
                    return np.ones(1)
                r = rows[i]
                return r[0] if r.shape[0] == 1 else r[q]

            for q, w in enumerate(weights):
                ru, rw = rows_of(iu, q), rows_of(iw, q)
                pair_pmf[:ru.size, :rw.size] += lam * w * np.outer(ru, rw)
    if moments is not None:
        for q, w in enumerate(weights):
            per_unit = [r[0] if r.shape[0] == 1 else r[q] for r in rows]
            for i, uid in enumerate(uids):
                others = np.ones(1)
                for k, r in enumerate(per_unit):
                    if k != i:
                        others = np.convolve(others, r)
                x_row = h * np.arange(per_unit[i].size) * per_unit[i]
                m = np.convolve(x_row, others)
                moments[uid][:m.size] += lam * w * m


def _moment_identity_error(view, live, unit_id, unit_terms_list, conds, weights, h, homer_path,
                           exact_path, homer):
    """``max_y |sum_u m_u(y) - y p(y)|`` for one event (unit rate), with
    ``p`` the event's own pmf of the view."""
    rows, homer_values = _first_view_rows(view, live, unit_terms_list, conds, h, homer)
    if homer_path:
        base, values = homer_values
        total = sum(values)
        n = int(math.ceil(float(total.max()) / h)) + 2
        pmf = np.zeros(n)
        inv_h = 1.0 / h
        for i, p in enumerate(base):
            if p == 0.0:
                continue
            _deposit(pmf, float(total[i]), p, inv_h)
        # The moments are split at the deposit nodes by the allocations'
        # shares of the node's loss, so the identity holds by construction.
        weighted = h * np.arange(n) * pmf
    elif exact_path:
        marginals = [r[0] for r in rows]
        pmf = comonotone_sum(marginals)
        weighted = np.zeros(pmf.size)

        def deposit(indices, mass):
            weighted[sum(indices)] += mass * sum(indices) * h

        comonotone_merge(marginals, deposit)
    else:
        pmf = _event_pmf(rows, weights)
        weighted = np.zeros(pmf.size)
        for q, w in enumerate(weights):
            per_unit = [r[0] if r.shape[0] == 1 else r[q] for r in rows]
            for i in range(len(per_unit)):
                others = np.ones(1)
                for k, r in enumerate(per_unit):
                    if k != i:
                        others = np.convolve(others, r)
                m = np.convolve(h * np.arange(per_unit[i].size) * per_unit[i], others)
                weighted[:m.size] += w * m
    return float(np.abs(weighted - h * np.arange(pmf.size) * pmf).max())


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
