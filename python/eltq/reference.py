"""SciPy reference implementation of ELT secondary-uncertainty quantization.

Transparent and accurate rather than fast. This module plays two roles:

- the mathematical oracle for the Rust extension: it follows the contract in
  ``secondary-uncertainty-mathematical-specification.md`` row by row, and the
  acceptance suite runs both implementations against it (reference totals use
  exact ``math.fsum`` summation, so oracle values carry no summation-order
  noise);
- the pure-Python fallback backend: when the compiled ``eltq._eltq``
  extension is not importable, the package API is served by this module
  (``eltq.BACKEND == "reference"``). Requires SciPy.

Two methods build the ``secondary`` grid (specification sections 5A and 5):
``survival`` (the default from 2.0.0) evaluates the portfolio exceedance rate
exactly at log-spaced nodes, interpolates it monotonically and buckets it by
differences at half-bucket edges (the nearest rule), each event's endpoint
handled per event; ``quantiles`` replaces each beta event by K equally
weighted midpoint quantiles deposited by the linear rule. The survival
machinery lives in ``SurvivalCurve`` so that the curve can be built once and
bucketed at any grid.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betainc, betaincc, betaln, ndtri
from scipy.stats import beta as beta_dist

from eltq._meta import (DEFAULT_NODES_PER_DECADE, LARGE_PARAMETER_FROM, NARROW_STEP_PARAMETER_SUM,
                        NARROW_STEPS, TAIL_INTERVALS, TAIL_RATE_FLOOR, CurveMeta, GridMeta, Meta,
                        RowIssues)

DEFAULT_CLAMP_FACTOR = 0.999
MAX_EXAMPLES = 10
METHODS = ("quantiles", "survival")
OVERFLOW_POLICIES = ("top_bucket", "drop", "error")

# Below this probability the closed-form lower-tail quantile marks the loss
# under which an event's survival function rounds to exactly 1 (1 - F is 1
# in double precision once F < 2^-54; the margin covers the closed form's
# error for shapes with beta < 1).
LOWER_SKIP_PROBABILITY = 2.0 ** -60
#: Edges per block of a tail walk (reference only; the Rust walk is per edge).
TAIL_BLOCK = 4096

# Canonical diagnostic categories, in canonical order (mirrored by eltq-core).
CATEGORIES = (
    "nonfinite",         # any field NaN/Inf: row dropped
    "negative_field",    # rate, max_loss, mean_loss, or an sd < 0: row dropped
    "mean_exceeds_max",  # mu > M (after optional loss-rate scaling): row dropped
    "zero_rate",         # rate == 0: row carries no mass, skipped
    "degenerate_atom",   # sigma == 0, mu == 0, mu == M, or M == 0: single atom
    "variance_dropped",  # inadmissible variance, row dropped by the drop policy
    "variance_atom",     # inadmissible variance collapsed to an atom by policy
    "variance_bernoulli",  # inadmissible variance replaced by the Bernoulli maximizer
)

VARIANCE_POLICIES = ("bernoulli", "error", "drop", "atom")


class EltqError(Exception):
    """Base class for reference-implementation errors."""


class VarianceError(EltqError):
    """Inadmissible beta variance under the strict ("error") policy."""


class GridOverflowError(EltqError):
    """Grid does not cover the maximum contributing loss (overflow="error")."""

    def __init__(self, message, *, required_length=None, examples=None):
        super().__init__(message)
        self.required_length = required_length
        self.examples = examples or []


def _check_variance_policy(variance_policy: str) -> None:
    if variance_policy == "clamp":
        raise EltqError(
            "the clamp variance policy was removed from the one-row route at "
            "2.0.0 (use bernoulli, error, drop, or atom)")
    if variance_policy not in VARIANCE_POLICIES:
        raise EltqError(f"unknown variance_policy {variance_policy!r}")


def _note(diagnostics: dict, category: str, event_id) -> None:
    issues = diagnostics.setdefault(category, RowIssues())
    issues.count += 1
    if len(issues.examples) < MAX_EXAMPLES:
        issues.examples.append(event_id)


# Row classification kinds (internal).
_DROP = 0    # contributes to neither output
_ATOM = 1    # atom at mu in both raw and secondary
_BETA = 2    # atom at mu in raw; a beta on [0, M] in secondary
_BERNOULLI = 3  # atom at mu in raw; rate (1 - p) at 0 and rate p at M in secondary


def beta_params(max_loss: float, mean_loss: float, variance: float):
    """Return ``(alpha, beta)`` for admissible ``(M, mu, sigma^2)``.

    Uses ``kappa = mu (M - mu) / sigma^2 - 1`` to avoid forming ``M**2``.
    Caller must ensure ``0 < mu < M`` and ``0 < sigma^2 < mu (M - mu)``.
    """
    p = mean_loss / max_loss
    kappa = mean_loss * (max_loss - mean_loss) / variance - 1.0
    return p * kappa, (1.0 - p) * kappa


def _deposit(hist: np.ndarray, x: float, w: float, inv_h: float) -> None:
    """Linear mass- and mean-preserving deposit of weight ``w`` at loss ``x``.

    A loss at or above the top node goes wholly to the top node: the linear
    rule has no upper node there, and the caller has already decided, by the
    overflow policy, that the loss belongs on the grid.
    """
    r = x * inv_h
    i = int(r)
    n = hist.size
    if i >= n - 1:
        hist[n - 1] += w
    else:
        f = r - i
        hist[i] += w * (1.0 - f)
        hist[i + 1] += w * f


class _Overflow:
    """Overflow accounting for one output under a policy (spec section 6).

    ``place(hist, x, w)`` deposits an atom, or diverts it when ``x`` is above
    the top bucket's upper edge: to the top node under ``top_bucket``, to
    nowhere under ``drop``; either way its rate and its moment in excess of
    the top node are recorded.
    """

    def __init__(self, policy: str, h: float, n: int):
        self.policy = policy
        self.inv_h = 1.0 / h
        self.top_node = (n - 1) * h
        self.top_edge = (n - 0.5) * h
        self.rate_terms: list[float] = []
        self.moment_terms: list[float] = []

    def place(self, hist: np.ndarray, x: float, w: float) -> None:
        if x > self.top_edge:
            self.rate_terms.append(w)
            self.moment_terms.append(w * (x - self.top_node))
            if self.policy == "top_bucket":
                hist[-1] += w
            return
        _deposit(hist, x, w, self.inv_h)

    def divert(self, hist: np.ndarray, rate: float, moment_excess: float) -> None:
        """Record an overflow already measured (the survival curve's)."""
        self.rate_terms.append(rate)
        self.moment_terms.append(moment_excess)
        if self.policy == "top_bucket":
            hist[-1] += rate

    @property
    def rate(self) -> float:
        return math.fsum(self.rate_terms)

    @property
    def moment(self) -> float:
        return math.fsum(self.moment_terms)


def _limit_slopes(d: float, m0: float, m1: float):
    """Fritsch and Carlson limiting of the two end slopes of one interval
    against its secant slope ``d`` (all nonpositive for a nonincreasing
    function): zero slopes on a flat interval, slopes of the wrong sign set
    to zero, and both scaled down together when ``a^2 + b^2 > 9`` in units
    of the secant, which guarantees a monotone cubic Hermite."""
    if d == 0.0:
        return 0.0, 0.0
    a = max(m0 / d, 0.0)
    b = max(m1 / d, 0.0)
    s = a * a + b * b
    if s > 9.0:
        tau = 3.0 / math.sqrt(s)
        a *= tau
        b *= tau
    return a * d, b * d


def _hermite(t: np.ndarray, y0, y1, dx, m0, m1) -> np.ndarray:
    """Cubic Hermite on one interval: values ``y0, y1``, width ``dx``,
    end slopes ``m0, m1``, at local coordinate ``t`` in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * dx * m0
            + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * dx * m1)


def _tail_moment(lam: float, m: float, a: float, b: float, e: float) -> float:
    """``lam P(X; X > e)`` for ``X = m Beta(a, b)``: the partial expectation
    is the mean times the survival function of the beta with ``a + 1``."""
    if e >= m:
        return 0.0
    return lam * m * (a / (a + b)) * float(betaincc(a + 1.0, b, e / m))


def _narrow_bulk(a: float, b: float, m: float, mu: float, sigma: float,
                 lam: float, floor: float, min_loss: float):
    """The bulk of a candidate narrow event by steps of ``sigma`` out from
    the mean: ``start``, the first step down where ``lam F`` is at or below
    the floor (``min_loss`` if reached first), and ``stop``, the first step
    up where ``lam S`` is at or below it (``M`` if reached first); None
    when either side needs more than ``NARROW_STEPS`` steps."""
    start = stop = None
    for i in range(1, NARROW_STEPS + 1):
        x = mu - i * sigma
        if x <= 0.0:
            start = 0.0
            break
        if lam * float(betainc(a, b, x / m)) <= floor:
            start = x
            break
    for i in range(1, NARROW_STEPS + 1):
        x = mu + i * sigma
        if x >= m:
            stop = m
            break
        if lam * float(betaincc(a, b, x / m)) <= floor:
            stop = x
            break
    if start is None or stop is None:
        return None
    return max(start, min_loss), stop


def _tail_survival(m: float, a: float, b: float, xk: float, s: float,
                   x: np.ndarray, min_distance: float) -> np.ndarray:
    """``S(x)`` of one beta event on its tail interval ``(x_k, M]`` by the
    endpoint interpolant: the log survival against the log distance to the
    endpoint, ``u = log(M - x)``, a monotone cubic Hermite with the exact
    slopes ``(1 - y) f(y) / S(y)`` on nodes shrinking the distance by
    ``sqrt(2)`` per step from ``M - x_k`` down to ``min_distance``, and the
    power law ``S ~ d^beta`` (the exact asymptote) below the last node.

    Notes
    -----
    Near its endpoint a beta's survival is ``(1 - y)^beta R(y)`` with
    ``R`` analytic and nonzero, so ``log S`` is nearly linear in ``u`` and
    the cubic follows it to a few units in 1e-6 over a tail interval, where
    no interpolant in ``x`` can; the shared curve stops ``TAIL_INTERVALS``
    node intervals before the endpoint for that reason (spec section 5A).
    ``s`` is the exact ``S(x_k)`` and is used as the first node's value, so
    the tail joins the curve bitwise.
    """
    d0 = m - xk
    count = 2 * max(1, int(math.ceil(math.log2(d0 / min_distance)))) + 1
    d = d0 * np.power(0.5, np.arange(count, dtype=np.float64) / 2.0)
    y = 1.0 - d / m
    surv = betaincc(a, b, y)
    surv[0] = s
    ln_b = betaln(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_surv = np.log(surv)
        # slope d log S / du = (1 - y) f(y) / S(y), finite for every d > 0.
        slope = np.exp((a - 1.0) * np.log(y) + b * np.log(d / m) - ln_b - log_surv)
    # Keep the nodes with a positive survival (underflow ends the list).
    keep = np.isfinite(log_surv)
    u = np.log(d[keep])[::-1]           # ascending in u
    g = log_surv[keep][::-1]
    sl = slope[keep][::-1]
    out = np.zeros(x.shape)
    if u.size == 0:
        return out   # no mass beyond x_k
    dist = m - x
    inside = dist > 0.0
    uq = np.log(dist[inside])
    res = np.empty(uq.shape)
    if u.size == 1:
        # One node: the power law alone.
        res[:] = g[0] + b * (uq - u[0])
    else:
        below = uq < u[0]
        res[below] = g[0] + b * (uq[below] - u[0])
        q = ~below
        k = np.clip(np.searchsorted(u, uq[q], side="right") - 1, 0, u.size - 2)
        du = u[k + 1] - u[k]
        m0 = np.empty(k.shape)
        m1 = np.empty(k.shape)
        for j, kk in enumerate(k):
            m0[j], m1[j] = _limit_slopes((g[kk + 1] - g[kk]) / du[j], sl[kk], sl[kk + 1])
        t = (uq[q] - u[k]) / du
        res[q] = _hermite(t, g[k], g[k + 1], du, m0, m1)
    out[inside] = np.exp(res)
    return np.minimum(out, s)


class SurvivalCurve:
    """The portfolio exceedance rate of the beta events (spec section 5A).

    Each beta event ``j`` is split at its truncation node ``x_k``, the node
    ``TAIL_INTERVALS`` intervals below the first node at or above its
    endpoint ``M_j``: the **curve** carries ``lambda_j (S_j(x) - S_j(x_k))``
    for ``x <= x_k``, continuous and ending exactly at a node, and the
    **tail**, the mass ``lambda_j S_j(x_k)`` on ``(x_k, M_j]``, is kept per
    event and deposited or evaluated exactly. The summed curve is therefore
    smooth between nodes, with kinks only at nodes, which carry one-sided
    slopes; the interpolant is a monotone cubic Hermite per interval in
    plain coordinates with the exact one-sided slopes.

    Attributes: ``loss`` (the nodes), ``curve`` (the summed truncated
    exceedance rate at them), ``slope_out`` and ``slope_in`` (``-x`` times
    the derivative from the right and from the left), ``tail_above`` (the
    total tail mass of the events truncated at or above each node, so that
    ``exceedance`` can add it), ``atom_loss`` and ``atom_rate`` (the atoms,
    not part of the curve), ``beta_rate`` (the total rate of the beta
    events, the value of the exceedance rate below every event).
    """

    def __init__(self, loss, curve, slope_out, slope_in, groups, below_rate,
                 beta_rate, atom_loss, atom_rate, whole=(), narrow=()):
        self.loss = np.asarray(loss, dtype=np.float64)
        self.curve = np.asarray(curve, dtype=np.float64)
        self.slope_out = np.asarray(slope_out, dtype=np.float64)
        self.slope_in = np.asarray(slope_in, dtype=np.float64)
        # groups[k]: list of (lam, M, alpha, beta, s) for the events whose
        # last node below M is node k, s = S(x_k); their tails live on
        # (x_k, M] inside the node interval k.
        self._groups = groups
        # Events whose tail region is their whole support above the lowest
        # node: (lam, M, alpha, beta, 1.0), deposited exactly from edge 0.
        self._whole = list(whole)
        # The narrow events (lam, M, alpha, beta, start, stop, S(start)),
        # sorted by start: kept out of the curve and walked exactly on
        # their bulk; in full below it, nothing above it.
        self._narrow = sorted(narrow, key=lambda t: t[4])
        self._narrow_start = np.array([t[4] for t in self._narrow], dtype=np.float64)
        above = np.cumsum([t[0] for t in self._narrow][::-1])[::-1]
        self._narrow_above = np.append(above, 0.0)
        self.narrow_mass = math.fsum(t[0] * t[6] for t in self._narrow)
        self._below_rate = float(below_rate)
        self.beta_rate = float(beta_rate)
        # Set by exceedance_curve_reference; the grid's metadata needs them.
        self.total_rate = float("nan")
        self.analytic_aal = float("nan")
        self.atom_loss = np.asarray(atom_loss, dtype=np.float64)
        self.atom_rate = np.asarray(atom_rate, dtype=np.float64)
        m1 = self.loss.size
        tails = np.zeros(m1 + 1)
        for k in range(m1):
            tails[k] = math.fsum(lam * s for lam, _, _, _, s in groups[k])
        tails[0] += math.fsum(lam * s for lam, _, _, _, s in self._whole)
        # tail_above[k] = total tail mass of the events truncated at node k
        # or later: the part of Lambda above x < x_k that the curve omits.
        self.tail_above = np.cumsum(tails[::-1])[::-1]
        self._prepare_interpolant()

    # -- construction ---------------------------------------------------

    def _prepare_interpolant(self) -> None:
        """Per interval, the limited end slopes ``dy/dx`` of the cubic
        Hermite in plain coordinates (the slopes are stored as ``-x y'``).

        A log-log Hermite was tried first and rejected: near a node where
        several truncated survival functions vanish together the logarithm
        magnifies the curvature, and on portfolios of J-shaped events it
        missed by 1e-2 where the plain cubic is within 1e-5.
        """
        x, y = self.loss, self.curve
        n_int = max(x.size - 1, 0)
        self._m0 = np.zeros(n_int)
        self._m1 = np.zeros(n_int)
        for i in range(n_int):
            if y[i] <= 0.0:
                continue
            d = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            self._m0[i], self._m1[i] = _limit_slopes(
                d, -self.slope_out[i] / x[i], -self.slope_in[i + 1] / x[i + 1])

    @property
    def nodes(self) -> int:
        return int(self.loss.size)

    @property
    def narrow_events(self) -> int:
        """The beta events kept out of the curve as narrow."""
        return len(self._narrow)

    @property
    def exceedance_rate(self) -> np.ndarray:
        """The exact exceedance rate of the beta events at the nodes."""
        return self.exceedance_betas(self.loss)

    @property
    def max_loss(self) -> float:
        """Largest contributing loss, beta endpoint or atom."""
        top = float(self.loss[-1]) if self.loss.size else 0.0
        if self.atom_loss.size:
            top = max(top, float(self.atom_loss.max()))
        return top

    def grid(self, bucket_size, grid_length=None, *, normalize=False,
             overflow="top_bucket"):
        """The secondary grid at ``bucket_size`` with ``grid_length`` nodes
        (derived to cover the largest loss when None) under the overflow
        policy; returns ``(array, GridMeta)``. Requires ``bucket_size >= 2
        loss[0]``."""
        h = float(bucket_size)
        if not (h > 0.0) or not math.isfinite(h):
            raise EltqError(f"bucket_size must be finite and positive, got {h}")
        if overflow not in OVERFLOW_POLICIES:
            raise EltqError(
                f"unknown overflow {overflow!r} (expected top_bucket, drop, or error)")
        top = self.max_loss
        auto_length = int(math.ceil(top / h)) + 1 if top > 0.0 else 1
        n = auto_length if grid_length is None else int(grid_length)
        if n < 1:
            raise EltqError(f"grid_length must be >= 1, got {n}")
        top_edge = (n - 0.5) * h
        if overflow == "error" and top > top_edge:
            raise GridOverflowError(
                f"grid of {n} nodes at spacing {h} covers losses up to "
                f"{top_edge}, but the maximum contributing loss is {top}; "
                f"grid_length >= {auto_length} required "
                f"(or select the top_bucket or drop overflow policy)",
                required_length=auto_length)
        weights, over_rate, over_moment = self.bucket(h, n)
        over = _Overflow(overflow, h, n)
        if over_rate > 0.0:
            over.divert(weights, over_rate, over_moment)
        for x, w in zip(self.atom_loss, self.atom_rate):
            over.place(weights, float(x), float(w))
        grid = h * np.arange(n, dtype=np.float64)
        moment = float(weights @ grid)
        mass = float(weights.sum())
        total = self.total_rate
        if normalize:
            if not total > 0.0:
                raise EltqError("cannot normalize: total contributing rate is zero")
            weights /= total
        meta = GridMeta(
            bucket_size=h, grid_length=n, normalized=bool(normalize),
            overflow=overflow,
            secondary_mass=mass / total if normalize else mass,
            secondary_mean=moment / total if normalize else moment,
            secondary_mean_error=moment - self.analytic_aal,
            rate_above_grid=over.rate, mean_above_grid=over.moment,
        )
        return weights, meta

    # -- the curve part ---------------------------------------------------

    def interpolate(self, x) -> np.ndarray:
        """The summed truncated exceedance rate (the curve part of
        ``Lambda``) by the interpolant; NaN below the lowest node, 0 above
        the highest."""
        x = np.asarray(x, dtype=np.float64)
        out = np.full(x.shape, np.nan)
        loss, y = self.loss, self.curve
        if loss.size == 0:
            return out
        out[x >= loss[-1]] = 0.0
        inside = (x >= loss[0]) & (x < loss[-1])
        if loss.size == 1 or not inside.any():
            return out
        xi = x[inside]
        k = np.searchsorted(loss, xi, side="right") - 1
        x0, x1 = loss[k], loss[k + 1]
        t = (xi - x0) / (x1 - x0)
        res = np.maximum(_hermite(t, y[k], y[k + 1], x1 - x0, self._m0[k], self._m1[k]), 0.0)
        # Exact at the nodes themselves.
        at_node = x0 == xi
        res[at_node] = y[k[at_node]]
        out[inside] = res
        return out

    def exact_curve(self, x) -> np.ndarray:
        """The curve part summed exactly over the events (the oracle for the
        interpolation error)."""
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros(x.shape)
        for k, group in enumerate(self._groups):
            xk = self.loss[k]
            sel = x < xk
            if not sel.any():
                continue
            for lam, m, a, b, s in group:
                out[sel] += lam * (betaincc(a, b, x[sel] / m) - s)
        return out

    # -- the whole exceedance rate ------------------------------------------

    def exact(self, x) -> np.ndarray:
        """``Lambda(x)`` summed exactly over the beta events (the oracle)."""
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros(x.shape)
        for group in list(self._groups) + [self._whole]:
            for lam, m, a, b, _ in group:
                out += lam * betaincc(a, b, np.minimum(x / m, 1.0))
        for lam, m, a, b, _, _, _ in self._narrow:
            out += lam * betaincc(a, b, np.minimum(x / m, 1.0))
        return out

    def exceedance_betas(self, x) -> np.ndarray:
        """``Lambda(x)`` of the beta events: the interpolated curve plus the
        tails, exact for the events whose tail interval contains ``x``."""
        x = np.asarray(x, dtype=np.float64)
        out = self.interpolate(x)
        loss = self.loss
        if loss.size == 0:
            return out
        inside = (x >= loss[0]) & (x < loss[-1])
        k = np.searchsorted(loss, x[inside], side="right") - 1
        add = self.tail_above[k + 1].copy()
        xi = x[inside]
        # Tails of the events truncated at this node or up to
        # TAIL_INTERVALS - 1 nodes below it may reach x; exact.
        for kk in np.unique(k):
            sel = k == kk
            for g in range(max(0, kk - TAIL_INTERVALS + 1), kk + 1):
                for lam, m, a, b, _ in self._groups[g]:
                    add[sel] += lam * betaincc(a, b, np.minimum(xi[sel] / m, 1.0))
        # The whole-tail events are counted in tail_above at their value
        # at the lowest node; replace that by their exact survival.
        for lam, m, a, b, s in self._whole:
            add += lam * (betaincc(a, b, np.minimum(xi / m, 1.0)) - s)
        # The narrow events: in full above x, exact where x is in the bulk.
        if self._narrow:
            add += self._narrow_above[np.searchsorted(self._narrow_start, xi, side="right")]
            for lam, m, a, b, start, stop, _ in self._narrow:
                sel = (xi >= start) & (xi < stop)
                if sel.any():
                    add[sel] += lam * betaincc(a, b, xi[sel] / m)
        out[inside] += add
        return out

    def exceedance(self, x) -> np.ndarray:
        """``Lambda(x)`` with the atoms included."""
        x = np.asarray(x, dtype=np.float64)
        total = self.exceedance_betas(x)
        if self.atom_loss.size:
            order = np.argsort(self.atom_loss, kind="stable")
            losses = self.atom_loss[order]
            tail = np.cumsum(self.atom_rate[order][::-1])[::-1]
            tail = np.append(tail, 0.0)
            total = total + tail[np.searchsorted(losses, x, side="right")]
        return total

    def interpolation_error(self, every_interval: bool = False,
                            samples: int = 64) -> float:
        """Largest error of the interpolated curve against the exact curve
        at interval log-midpoints, relative to the exact exceedance rate,
        over the points where that rate exceeds ``1e-12 beta_rate``: every
        interval, or ``samples`` intervals spread over the node range."""
        if self.loss.size < 2:
            return 0.0
        log_mid = 0.5 * (np.log(self.loss[:-1]) + np.log(self.loss[1:]))
        if not every_interval and log_mid.size > samples:
            pick = np.unique(np.linspace(0, log_mid.size - 1, samples).round().astype(int))
            log_mid = log_mid[pick]
        x = np.exp(log_mid)
        scale = self.exact(x)
        keep = scale > 1e-12 * self.beta_rate
        if not keep.any():
            return 0.0
        err = np.abs(self.interpolate(x) - self.exact_curve(x))
        return float(np.max(err[keep] / scale[keep]))

    # -- bucketing -----------------------------------------------------------

    def bucket(self, h: float, n: int, *, exact_edges: bool = False,
               tail_rate_floor: float = TAIL_RATE_FLOOR):
        """Beta part of the secondary grid by the nearest rule.

        Returns ``(weights, overflow_rate, overflow_moment_excess)``: the
        length-``n`` array with bucket ``i`` holding the mass in
        ``(e_{i-1}, e_i]`` (bucket 0 everything up to ``e_0``), the rate
        beyond the top edge ``e_{n-1}``, and that overflow's first moment in
        excess of the top node (by partial expectations). The curve part is
        differenced at the edges; each event's tail is differenced exactly
        at the edges inside its tail interval. ``exact_edges`` evaluates the
        curve part exactly at every edge (tests only); ``tail_rate_floor``
        is the rate cut of the tail walks (tests only; 0 walks every tail
        to its endpoint or the top edge). Requires ``h >= 2 loss[0]``.
        """
        loss = self.loss
        if loss.size and h < 2.0 * loss[0] * (1.0 - 1e-12):
            raise EltqError(
                f"bucket_size {h} is below twice the curve's lowest node "
                f"{loss[0]}; the curve knows nothing below that loss"
            )
        edges = (np.arange(n, dtype=np.float64) + 0.5) * h
        top_edge = edges[-1]
        top_node = (n - 1) * h
        weights = np.zeros(n, dtype=np.float64)
        if loss.size:
            values = self.exact_curve(edges) if exact_edges else self.interpolate(edges)
            values = np.where(np.isnan(values), self.curve[0], values)
            # Monotone by construction; enforce it against rounding so that
            # no bucket is negative and the telescoping sum stays exact.
            values = np.minimum.accumulate(np.minimum(values, self.curve[0]))
            # Below e_0 = x_0: everything the curve and the tails do not hold.
            weights[0] = self.beta_rate - values[0] - self.tail_above[0] - self.narrow_mass
            weights[1:] = values[:-1] - values[1:]
            over_rate_terms = [float(values[-1])]
        else:
            weights[0] = self.beta_rate - self.narrow_mass
            over_rate_terms = []
        over_moment_terms = []
        # The curve part above the top edge: events truncated above it.
        if loss.size and values[-1] > 0.0:
            for k, group in enumerate(self._groups):
                xk = loss[k]
                if xk <= top_edge:
                    continue
                for lam, m, a, b, s in group:
                    mass = lam * (float(betaincc(a, b, top_edge / m)) - s)
                    moment = _tail_moment(lam, m, a, b, top_edge) - _tail_moment(lam, m, a, b, xk)
                    over_moment_terms.append(moment - top_node * mass)
        # The tails and the narrow events, per event: the survival walked
        # along the edges inside its interval and on the grid (a tail by its
        # endpoint interpolant, exactly under ``exact_edges``; a narrow
        # event exactly) to the top edge or to the first edge where the
        # event's rate reaches the floor; the remainder goes to the next
        # bucket, and beyond the grid its moment is taken by partial
        # expectations. The whole-tail events start at the lowest node.
        threshold = tail_rate_floor * self.beta_rate
        x_min = 0.5 * loss[0]

        def deposit(lam, m, a, b, start, end, s, survival_at):
            first = int(math.floor(start / h - 0.5)) + 1      # e_{first-1} <= start < e_first
            stop = min(int(math.ceil(end / h - 0.5)) - 1, n - 1)   # e_stop < end, on the grid
            if stop < first or lam * s <= threshold:
                # No edge to walk, or the whole mass at or below the floor:
                # one deposit in the bucket containing ``start``.
                self._put(weights, first, lam * s, over_rate_terms)
                rest = first
            else:
                rest = self._walk(weights, over_rate_terms, first, stop, lam, s, h,
                                  threshold, survival_at)
            if rest >= n:
                begin = max(start, top_edge)
                mass = lam * (float(betaincc(a, b, begin / m)) if begin > start else s)
                over_moment_terms.append(_tail_moment(lam, m, a, b, begin) - top_node * mass)

        for k, group in list(enumerate(self._groups)) + [(0, self._whole)]:
            xk = loss[k]
            for lam, m, a, b, s in group:
                if s <= 0.0:
                    continue   # the survival underflowed before x_k: no tail
                if exact_edges:
                    def survival_at(e, a=a, b=b, m=m):
                        return betaincc(a, b, e / m)
                else:
                    def survival_at(e, a=a, b=b, m=m, xk=xk, s=s):
                        return _tail_survival(m, a, b, xk, s, e, x_min)
                deposit(lam, m, a, b, xk, m, s, survival_at)
        for lam, m, a, b, start, stop, s in self._narrow:
            def survival_at(e, a=a, b=b, m=m):
                return betaincc(a, b, e / m)
            deposit(lam, m, a, b, start, stop, s, survival_at)
        return weights, math.fsum(over_rate_terms), math.fsum(over_moment_terms)

    def _walk(self, weights, over_rate_terms, first, stop, lam, s, h, threshold,
              survival_at):
        """Walk one event's survival along the edges ``first .. stop`` in
        blocks (given ``first <= stop`` and ``lam s`` above the floor):
        deposit ``lam`` times the drop across each edge, break at the last
        edge or at the first edge where ``lam S`` reaches the floor, and
        lump the remainder in the next bucket; returns that bucket."""
        prev, i = s, first
        while True:
            # The same edge arithmetic as the Rust walk, so the values
            # agree bitwise.
            block = (np.arange(i, min(i + TAIL_BLOCK, stop + 1), dtype=np.float64) + 0.5) * h
            surv = survival_at(block)
            cut = np.flatnonzero(lam * surv <= threshold)
            used = int(cut[0]) + 1 if cut.size else surv.size
            values = np.concatenate([[prev], surv[:used]])
            for j, w in zip(range(i, i + used), lam * (values[:-1] - values[1:])):
                self._put(weights, j, w, over_rate_terms)
            prev, i = float(surv[used - 1]), i + used
            if cut.size or i > stop:
                break
        self._put(weights, i, lam * prev, over_rate_terms)
        return i

    @staticmethod
    def _put(weights, i, w, over_rate_terms):
        """Deposit ``w`` into bucket ``i``, or record it as overflow when the
        bucket lies beyond the grid (its moment is added per event later)."""
        if i < weights.size:
            weights[i] += w
        else:
            over_rate_terms.append(w)


def build_survival_curve(beta_rows, atoms, min_loss: float,
                         nodes_per_decade: int) -> SurvivalCurve:
    """Evaluate the truncated exceedance rate and its one-sided slopes at
    geometric nodes; group the events' tails by node.

    Parameters
    ----------
    beta_rows : sequence of (rate, max_loss, alpha, beta)
    atoms : sequence of (loss, rate)
    min_loss : float
        The lowest node ``x_0`` (the smallest loss the consumer uses).
    nodes_per_decade : int
        Node density; the nodes run from ``min_loss`` to the largest beta
        ``max_loss`` with ``ceil(nodes_per_decade * log10(ratio))`` intervals.

    Notes
    -----
    Per event the two skips of the specification apply: nodes at or above
    its endpoint contribute nothing (its tail is kept per event), and nodes
    below its lower-tail plateau contribute exactly its rate less its tail
    and nothing to the slope. Slopes are accumulated as ``-x Lambda'(x)``,
    with an event's slope at its truncation node counted on the incoming
    side only.
    """
    if not (min_loss > 0.0) or not math.isfinite(min_loss):
        raise EltqError(f"min_loss must be finite and positive, got {min_loss}")
    if int(nodes_per_decade) < 1:
        raise EltqError(f"nodes_per_decade must be >= 1, got {nodes_per_decade}")
    beta_rows = [tuple(float(v) for v in row) for row in beta_rows]
    atom_loss = np.asarray([x for x, _ in atoms], dtype=np.float64)
    atom_rate = np.asarray([w for _, w in atoms], dtype=np.float64)
    beta_rate = math.fsum(row[0] for row in beta_rows)
    max_m = max((row[1] for row in beta_rows), default=0.0)
    if max_m <= min_loss:
        # Every event lies below the lowest node: nothing to interpolate.
        return SurvivalCurve(np.array([min_loss]), np.zeros(1), np.zeros(1), np.zeros(1),
                             [[]], beta_rate, beta_rate, atom_loss, atom_rate)
    intervals = max(1, int(math.ceil(nodes_per_decade * math.log10(max_m / min_loss))))
    log_x = np.linspace(math.log(min_loss), math.log(max_m), intervals + 1)
    loss = np.exp(log_x)
    loss[0] = min_loss
    loss[-1] = max_m
    m1 = loss.size
    curve = np.zeros(m1)
    slope_both = np.zeros(m1)   # -x Lambda' from events continuing past the node
    slope_end = np.zeros(m1)    # -x Lambda' from events truncated at the node
    groups = [[] for _ in range(m1)]
    below_rate = 0.0
    whole = []   # events whose tail region starts at the lowest node
    narrow = []  # events narrower than a node interval, walked exactly
    # The narrow criterion, one node interval at the mean, and the rate
    # floor of the bulk search.
    narrow_ratio = 10.0 ** (1.0 / nodes_per_decade) - 1.0
    narrow_floor = TAIL_RATE_FLOOR * beta_rate
    for lam, m, a, b, mu, sigma in beta_rows:
        if m <= min_loss:
            below_rate += lam
            continue
        # Only bounded, log-concave densities (both parameters above 1)
        # qualify; beyond NARROW_STEP_PARAMETER_SUM the event is a step at
        # its mean, as in the Rust engine.
        if sigma < mu * narrow_ratio and a > 1.0 and b > 1.0:
            if a + b > NARROW_STEP_PARAMETER_SUM:
                bulk = (max(mu, min_loss), mu)
            else:
                bulk = _narrow_bulk(a, b, m, mu, sigma, lam, narrow_floor, min_loss)
            if bulk is not None:
                start, stop = bulk
                if stop > min_loss:
                    s = 1.0 if stop <= start else float(betaincc(a, b, start / m))
                    narrow.append((lam, m, a, b, start, stop, s))
                continue
        # loss[k] < m <= loss[k + 1], then TAIL_INTERVALS - 1 nodes lower.
        k = int(np.searchsorted(loss, m, side="left")) - 1 - (TAIL_INTERVALS - 1)
        if k < 0:
            # Its tail starts at the lowest node: S there is its "s".
            whole.append((lam, m, a, b, float(betaincc(a, b, loss[0] / m))))
            continue
        y = loss[:k + 1] / m
        ln_b = betaln(a, b)
        # Lower skip: below the closed-form lower-tail quantile at
        # LOWER_SKIP_PROBABILITY the survival function is exactly 1.
        ln_plateau = (math.log(LOWER_SKIP_PROBABILITY) + math.log(a) + ln_b) / a
        plateau = y < math.exp(ln_plateau) if ln_plateau > -745.0 else np.zeros(y.shape, dtype=bool)
        surv = np.ones(y.shape)
        active = ~plateau
        surv[active] = betaincc(a, b, y[active])
        s = float(surv[k])
        curve[:k + 1] += lam * (surv - s)
        # x f(x / M) / M = exp(a ln y + (b - 1) ln(1 - y) - ln B): the
        # density times y, finite as y -> 0 for every alpha > 0.
        g = np.zeros(y.shape)
        g[active] = lam * np.exp(a * np.log(y[active]) + (b - 1.0) * np.log1p(-y[active]) - ln_b)
        slope_both[:k] += g[:k]
        slope_end[k] += g[k]
        groups[k].append((lam, m, a, b, s))
    curve[curve < 0.0] = 0.0
    return SurvivalCurve(loss, curve, slope_both, slope_both + slope_end, groups,
                         below_rate, beta_rate, atom_loss, atom_rate, whole, narrow)


def _classify_rows(rate, max_loss, mean_loss, sd_indep, sd_corr, event_ids,
                   mean_is_loss_rate, variance_policy):
    """Classify every row (spec section 4): kinds, effective mean and
    variance, the largest contributing loss, and the diagnostics. Dropped
    rows never touch an output."""
    n_rows = rate.size
    diagnostics: dict[str, RowIssues] = {}
    kinds = np.empty(n_rows, dtype=np.int8)
    mu_eff = np.empty(n_rows, dtype=np.float64)
    var_eff = np.empty(n_rows, dtype=np.float64)
    max_contrib_loss = 0.0
    for j in range(rate.size):
        lam, m, mu, s_i, s_c = (rate[j], max_loss[j], mean_loss[j],
                                sd_indep[j], sd_corr[j])
        kinds[j] = _DROP
        if not all(map(math.isfinite, (lam, m, mu, s_i, s_c))):
            _note(diagnostics, "nonfinite", event_ids[j])
            continue
        if lam < 0.0 or m < 0.0 or mu < 0.0 or s_i < 0.0 or s_c < 0.0:
            _note(diagnostics, "negative_field", event_ids[j])
            continue
        mu = mu * m if mean_is_loss_rate else mu
        if mu > m:
            _note(diagnostics, "mean_exceeds_max", event_ids[j])
            continue
        if lam == 0.0:
            _note(diagnostics, "zero_rate", event_ids[j])
            continue
        # Total conditional sd is the arithmetic sum of the two components
        # (Burke, Rau-Chaplin, Varghese 2016, section III.E, step i).
        sigma = s_i + s_c
        sigma2 = sigma * sigma
        mu_eff[j] = mu
        if m == 0.0 or sigma2 == 0.0 or mu == 0.0 or mu == m:
            # Boundary/degenerate treatment: a single atom at mu. (For a
            # boundary mean with sigma > 0 the variance is formally
            # inadmissible; the specified degenerate treatment resolves it.)
            kinds[j] = _ATOM
            _note(diagnostics, "degenerate_atom", event_ids[j])
        else:
            v_max = mu * (m - mu)
            if sigma2 >= v_max:
                if variance_policy == "error":
                    raise VarianceError(
                        f"row {event_ids[j]}: sigma^2 = {sigma2} >= "
                        f"mu (M - mu) = {v_max} is inadmissible for a beta"
                    )
                if variance_policy == "drop":
                    _note(diagnostics, "variance_dropped", event_ids[j])
                    continue
                if variance_policy == "atom":
                    kinds[j] = _ATOM
                    _note(diagnostics, "variance_atom", event_ids[j])
                else:  # bernoulli
                    kinds[j] = _BERNOULLI
                    sigma2 = v_max
                    _note(diagnostics, "variance_bernoulli", event_ids[j])
            else:
                kinds[j] = _BETA
            var_eff[j] = sigma2
        max_contrib_loss = max(max_contrib_loss, m)
    return kinds, mu_eff, var_eff, max_contrib_loss, diagnostics


def exceedance_curve_reference(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    min_loss,
    nodes_per_decade=DEFAULT_NODES_PER_DECADE,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
    event_ids=None,
):
    """Build the exceedance curve of an ELT from ``min_loss`` up (spec
    section 5A); return ``(SurvivalCurve, CurveMeta)``.

    Rows are classified as in ``quantize_reference``; the beta events go
    on the curve and the atoms (degenerate rows, policy atoms, both atoms
    of a Bernoulli row) beside it. The curve then serves ``exceedance``,
    ``grid`` at any ``bucket_size >= 2 min_loss``, and the OEP points.
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
    x0 = float(min_loss)
    if not (x0 > 0.0) or not math.isfinite(x0):
        raise EltqError(f"min_loss must be finite and positive, got {x0}")
    npd = int(nodes_per_decade)
    if npd < 1:
        raise EltqError(f"nodes_per_decade must be >= 1, got {npd}")
    _check_variance_policy(variance_policy)
    kinds, mu_eff, var_eff, _, diagnostics = _classify_rows(
        rate, max_loss, mean_loss, sd_indep, sd_corr, event_ids,
        mean_is_loss_rate, variance_policy)
    rates_kept, aal_terms, beta_rows, atoms = [], [], [], []
    valid_rows = 0
    for j in range(n_rows):
        kind = kinds[j]
        if kind == _DROP:
            continue
        valid_rows += 1
        lam, m, mu = rate[j], max_loss[j], mu_eff[j]
        rates_kept.append(lam)
        aal_terms.append(lam * mu)
        if kind == _ATOM:
            atoms.append((mu, lam))
        elif kind == _BERNOULLI:
            p = mu / m
            atoms.append((0.0, lam * (1.0 - p)))
            atoms.append((m, lam * p))
        else:
            a, b = beta_params(m, mu, var_eff[j])
            beta_rows.append((lam, m, a, b, mu, math.sqrt(var_eff[j])))
    curve = build_survival_curve(beta_rows, atoms, x0, npd)
    curve.total_rate = math.fsum(rates_kept)
    curve.analytic_aal = math.fsum(aal_terms)
    meta = CurveMeta(
        min_loss=x0, max_loss=curve.max_loss, nodes=curve.nodes,
        nodes_per_decade=npd,
        interpolation_error=curve.interpolation_error(),
        total_rate=curve.total_rate, analytic_aal=curve.analytic_aal,
        input_rows=n_rows, valid_rows=valid_rows, diagnostics=diagnostics,
        narrow_events=curve.narrow_events,
    )
    return curve, meta


def large_parameter_quantiles(a, b, u):
    """Beta quantiles at large parameters by the Cornish-Fisher expansion.

    Both backends take this route when ``min(a, b) >= LARGE_PARAMETER_FROM``
    (spec section 5): the inversion of the incomplete beta drifts there (its
    continued fraction needs about ``sqrt(min(a, b))`` iterations and stops
    at 400) and fails beyond ``a + b`` of about 5e18, and SciPy's ``ppf`` is
    off by hundreds of standard deviations at ``a = 5e20``. The expansion to
    second order in the cumulants (Abramowitz and Stegun 26.2.49) is within
    4e-9 standard deviations of SciPy at the threshold and improves as
    ``min(a, b) ** -1.5``.

    Parameters
    ----------
    a, b : float
        The beta's shape parameters.
    u : array_like
        Probability levels in [0, 1]; 0 and 1 give 0 and 1.

    Returns
    -------
    ndarray
        The quantiles on [0, 1], the same shape as ``u``.
    """
    u = np.asarray(u, dtype=np.float64)
    s = a + b
    mean = a / s
    sd = math.sqrt(a * b / (s * s * (s + 1.0)))
    # (b - a) / sqrt(a b), the shape factor of the odd cumulants, in a form
    # that stays finite for parameters up to about 1e150.
    r = math.sqrt(b / a) - math.sqrt(a / b)
    skew = 2.0 * r * math.sqrt(s + 1.0) / (s + 2.0)
    kurt = 6.0 * (r * r * (s + 1.0) - (s + 2.0)) / ((s + 2.0) * (s + 3.0))
    with np.errstate(all="ignore"):
        z = ndtri(u)
        z2 = z * z
        w = (z + skew * (z2 - 1.0) / 6.0 + kurt * z * (z2 - 3.0) / 24.0
             - skew * skew * z * (2.0 * z2 - 5.0) / 36.0)
        x = np.clip(mean + sd * w, 0.0, 1.0)
    return np.where(u <= 0.0, 0.0, np.where(u >= 1.0, 1.0, x))


def quantize_reference(
    rate,
    max_loss,
    mean_loss,
    sd_indep,
    sd_corr,
    *,
    bucket_size,
    grid_length=None,
    method="survival",
    quadrature_points=32,
    nodes_per_decade=DEFAULT_NODES_PER_DECADE,
    overflow="top_bucket",
    normalize=False,
    mean_is_loss_rate=False,
    variance_policy="bernoulli",
    event_ids=None,
    exact_edges=False,
    check_every_interval=False,
):
    """Quantize an ELT onto an even loss grid; return ``(raw, secondary, meta)``.

    Parameters mirror the specification: ``bucket_size`` is the grid spacing h,
    ``grid_length`` the number of nodes n (auto-derived from the maximum
    contributing loss when None). ``method`` selects how the secondary grid
    is built: ``survival`` (default: the portfolio exceedance rate at
    ``nodes_per_decade`` geometric nodes, interpolated and bucketed by the
    nearest rule, each event's endpoint handled per event; section 5A) or
    ``quantiles`` (K = ``quadrature_points`` midpoint beta quantiles per
    event, deposited linearly; section 5). ``overflow`` is the policy for
    mass above the top bucket's upper edge: ``top_bucket``, ``drop`` or
    ``error``. ``variance_policy`` is one of ``bernoulli`` (default: the row
    becomes two atoms, rate ``1 - p`` at 0 and rate ``p`` at M with
    ``p = mu / M``, the variance maximizer on [0, M]), ``error``, ``drop``,
    or ``atom``; the ``clamp`` of versions before 2.0.0 is refused.

    ``exact_edges`` (survival, tests only) evaluates ``Lambda`` exactly at
    every bucket edge instead of interpolating; ``check_every_interval``
    reports the interpolation error over every node interval rather than
    the 64-point sample.
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
    if method not in METHODS:
        raise EltqError(f"unknown method {method!r} (expected quantiles or survival)")
    K = int(quadrature_points)
    if K < 1:
        raise EltqError(f"quadrature_points must be >= 1, got {K}")
    npd = int(nodes_per_decade)
    if npd < 1:
        raise EltqError(f"nodes_per_decade must be >= 1, got {npd}")
    if overflow not in OVERFLOW_POLICIES:
        raise EltqError(
            f"unknown overflow {overflow!r} (expected top_bucket, drop, or error)")
    _check_variance_policy(variance_policy)

    kinds, mu_eff, var_eff, max_contrib_loss, diagnostics = _classify_rows(
        rate, max_loss, mean_loss, sd_indep, sd_corr, event_ids,
        mean_is_loss_rate, variance_policy)

    # Grid setup. The derived length covers max M, (n - 1) h >= max M, so
    # the overflow is exactly zero; an explicit shorter grid is a window,
    # handled by the overflow policy (spec section 6).
    if grid_length is None:
        n = int(math.ceil(max_contrib_loss / h)) + 1 if max_contrib_loss > 0.0 else 1
    else:
        n = int(grid_length)
        if n < 1:
            raise EltqError(f"grid_length must be >= 1, got {n}")
        top_edge = (n - 0.5) * h
        if overflow == "error" and max_contrib_loss > top_edge:
            required = int(math.ceil(max_contrib_loss / h)) + 1
            examples = [event_ids[j] for j in range(n_rows)
                        if kinds[j] != _DROP
                        and max_loss[j] > top_edge][:MAX_EXAMPLES]
            raise GridOverflowError(
                f"grid of {n} nodes at spacing {h} covers losses up to "
                f"{top_edge}, but the maximum contributing loss is "
                f"{max_contrib_loss}; grid_length >= {required} required "
                f"(or select the top_bucket or drop overflow policy)",
                required_length=required,
                examples=examples,
            )

    # Pass 2: deposits plus exact reference totals.
    raw = np.zeros(n, dtype=np.float64)
    secondary = np.zeros(n, dtype=np.float64)
    raw_overflow = _Overflow(overflow, h, n)
    sec_overflow = _Overflow(overflow, h, n)
    u = (np.arange(K, dtype=np.float64) + 0.5) / K
    rates_kept, aal_terms, quad_terms = [], [], []
    beta_rows, atoms = [], []
    valid_rows = 0
    for j in range(n_rows):
        kind = kinds[j]
        if kind == _DROP:
            continue
        valid_rows += 1
        lam, m, mu = rate[j], max_loss[j], mu_eff[j]
        rates_kept.append(lam)
        aal_terms.append(lam * mu)
        raw_overflow.place(raw, mu, lam)
        if kind == _ATOM:
            sec_overflow.place(secondary, mu, lam)
            quad_terms.append(lam * mu)
            atoms.append((mu, lam))
        elif kind == _BERNOULLI:
            # The variance maximizer on [0, M]: rate p at M and the rest at
            # 0, so the row's mean lam * mu is exact.
            p = mu / m
            sec_overflow.place(secondary, 0.0, lam * (1.0 - p))
            sec_overflow.place(secondary, m, lam * p)
            quad_terms.append(lam * mu)
            atoms.append((0.0, lam * (1.0 - p)))
            atoms.append((m, lam * p))
        else:
            a, b = beta_params(m, mu, var_eff[j])
            if method == "quantiles":
                if min(a, b) >= LARGE_PARAMETER_FROM:
                    xs = m * large_parameter_quantiles(a, b, u)
                else:
                    xs = m * beta_dist.ppf(u, a, b)
                w = lam / K
                for x in xs:
                    sec_overflow.place(secondary, x, w)
                quad_terms.append(w * math.fsum(xs))
            else:
                beta_rows.append((lam, m, a, b, mu, math.sqrt(var_eff[j])))

    curve = None
    interpolation_error = None
    if method == "survival":
        curve = build_survival_curve(beta_rows, atoms, 0.5 * h, npd)
        weights, over_rate, over_moment = curve.bucket(h, n, exact_edges=exact_edges)
        secondary += weights
        if over_rate > 0.0:
            sec_overflow.divert(secondary, over_rate, over_moment)
        interpolation_error = curve.interpolation_error(every_interval=check_every_interval)

    total_rate = math.fsum(rates_kept)
    analytic_aal = math.fsum(aal_terms)
    quadrature_secondary_mean = math.fsum(quad_terms) if method == "quantiles" else None

    grid = h * np.arange(n, dtype=np.float64)
    secondary_moment = float(secondary @ grid)  # rate units, before normalizing
    if normalize:
        if not total_rate > 0.0:
            raise EltqError("cannot normalize: total contributing rate is zero")
        raw /= total_rate
        secondary /= total_rate

    meta = Meta(
        bucket_size=h,
        grid_length=n,
        quadrature_points=K if method == "quantiles" else None,
        normalized=bool(normalize),
        total_rate=total_rate,
        analytic_aal=analytic_aal,
        raw_mass=float(raw.sum()),
        raw_mean=float(raw @ grid),
        secondary_mass=float(secondary.sum()),
        secondary_mean=float(secondary @ grid),
        quadrature_secondary_mean=quadrature_secondary_mean,
        secondary_mean_error=secondary_moment - analytic_aal,
        input_rows=n_rows,
        valid_rows=valid_rows,
        diagnostics=diagnostics,
        method=method,
        nodes=curve.nodes if curve is not None else None,
        nodes_per_decade=npd if method == "survival" else None,
        interpolation_error=interpolation_error,
        narrow_events=curve.narrow_events if curve is not None else None,
        overflow=overflow,
        rate_above_grid={"raw": raw_overflow.rate, "secondary": sec_overflow.rate},
        mean_above_grid={"raw": raw_overflow.moment, "secondary": sec_overflow.moment},
    )
    return raw, secondary, meta
