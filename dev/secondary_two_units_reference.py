"""Reference implementation: bivariate secondary uncertainty for one event, two units.

Given one catastrophe event and two units (programs, portfolios, or any two
exposure cells sharing the event), each unit i supplies the RMS-style ELT
fields: mean loss ``mu_i``, independent standard deviation ``sigma_I,i``,
correlated standard deviation ``sigma_C,i``, and maximum loss ``M_i``. The
module returns the joint distribution of the two conditional losses on a
common, evenly spaced loss grid: an aggregate-style bucketed bivariate pmf.

Method (Burke, Rau-Chaplin, Varghese 2016, Section III.E, "Computing probable
maximum loss in catastrophe reinsurance portfolios ...", confirmed against the
paper 2026-08-25):

1. Total sd is the arithmetic sum ``sigma_i = sigma_I,i + sigma_C,i``. This is
   the RMS convention, not the root sum of squares.
2. Marginal severity: the damage ratio ``X_i / M_i`` is Beta(alpha_i, beta_i)
   with mean ``p_i = mu_i / M_i`` and sd ``s_i = sigma_i / M_i`` (both
   normalized by ``M_i``), so ``alpha = p kappa``, ``beta = (1 - p) kappa``,
   ``kappa = p (1 - p) / s^2 - 1``. Feasibility requires ``s^2 < p (1 - p)``;
   Burke clamps sigma to just below the bound, mirrored here as the default
   ``variance_policy="clamp"`` (same policy name and default factor as
   ``eltq.reference``).
3. Dependence: draw one shared standard normal ``Z_c`` for the event and an
   independent ``Z_i`` per unit, and set
   ``U_i = Phi((sigma_C,i Z_c + sigma_I,i Z_i) / sqrt(sigma_C,i^2 + sigma_I,i^2))``.
   Each ``U_i`` is uniform by construction, so the marginals are exactly the
   betas above, and the pair has a Gaussian copula with normal correlation
   ``rho = (sigma_C,1 sigma_C,2) / (t_1 t_2)``, ``t_i = sqrt(sigma_C,i^2 + sigma_I,i^2)``.
   Note the mismatch that looks dubious but is the published method: the
   marginal sd adds the sigmas arithmetically while the copula weights
   normalize in quadrature (they must, for ``U_i`` to be uniform).
4. Loss: ``X_i = M_i F_i^{-1}(U_i)`` with ``F_i`` the Beta(alpha_i, beta_i) cdf.

Computation is deterministic, no simulation. Conditional on ``Z_c = z`` the
two losses are independent with conditional cdf
``P(X_i <= x | z) = Phi((Phi^{-1}(F_i(x / M_i)) - w_c,i z) / w_I,i)`` where
``w_c,i = sigma_C,i / t_i`` and ``w_I,i = sigma_I,i / t_i``. Each unit's
conditional distribution is bucketed by cdf differences at half-bucket edges
(the aggregate "round" method: bucket k holds mass in
``((k - 1/2) h, (k + 1/2) h]``), and the joint pmf is the average of outer
products over a midpoint quantile grid of ``z`` values, matching the midpoint
quadrature style of the one-unit specification.

Self-contained on purpose: depends only on numpy, scipy, pandas, and does not
import ``eltq``. It lives in ``dev`` as the mathematical reference for the
planned bivariate extension; folding it into the package is a separate,
explicit step.
"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm

DEFAULT_CLAMP_FACTOR = 0.999
VARIANCE_POLICIES = ("clamp", "error", "atom")

#: Input record for one unit, in ELT field order. ``mean_loss`` and the sds are
#: in loss currency units (not damage ratios); ``max_loss`` is the beta upper
#: bound M. Plain 4-tuples in this order are accepted everywhere a Unit is.
Unit = namedtuple("Unit", ["mean_loss", "sd_indep", "sd_corr", "max_loss"])

# Internal unit kinds after validation.
_ATOM = 0        # degenerate: all conditional mass at one loss value
_COMONOTONE = 1  # sd_indep == 0: loss is a deterministic function of Z_c
_BETA = 2        # general case


class SecondaryError(Exception):
    """Invalid input, inadmissible variance under the strict policy, or an
    insufficient grid."""


@dataclass
class _Prepped:
    """Validated, derived per-unit parameters."""
    kind: int
    mean_loss: float
    max_loss: float
    alpha: float = math.nan
    beta: float = math.nan
    w_corr: float = 0.0     # sigma_C / t, the shared-factor loading
    w_indep: float = 0.0    # sigma_I / t
    sigma_used: float = 0.0  # total sd after any clamp, loss units
    notes: list = field(default_factory=list)


def _prepare(unit, label, variance_policy, clamp_factor):
    """Validate one unit and derive beta parameters and copula weights."""
    mu, s_i, s_c, m = (float(x) for x in unit)
    if not all(map(math.isfinite, (mu, s_i, s_c, m))):
        raise SecondaryError(f"{label}: all fields must be finite")
    if mu < 0.0 or s_i < 0.0 or s_c < 0.0 or m < 0.0:
        raise SecondaryError(f"{label}: negative field")
    if mu > m:
        raise SecondaryError(f"{label}: mean_loss {mu} exceeds max_loss {m}")
    sigma = s_i + s_c
    prep = _Prepped(kind=_ATOM, mean_loss=mu, max_loss=m, sigma_used=sigma)
    if m == 0.0 or sigma == 0.0 or mu == 0.0 or mu == m:
        # Boundary treatment as in the one-unit spec: a single atom at mu.
        prep.notes.append("degenerate_atom")
        return prep
    p = mu / m
    s_max2 = mu * (m - mu)
    if sigma * sigma >= s_max2:
        if variance_policy == "error":
            raise SecondaryError(
                f"{label}: sigma^2 = {sigma * sigma} >= mu (M - mu) = {s_max2} "
                "is inadmissible for a beta")
        if variance_policy == "atom":
            prep.notes.append("variance_atom")
            return prep
        sigma = clamp_factor * math.sqrt(s_max2)
        prep.sigma_used = sigma
        prep.notes.append("sigma_clamped")
    kappa = mu * (m - mu) / (sigma * sigma) - 1.0
    prep.alpha = p * kappa
    prep.beta = (1.0 - p) * kappa
    t = math.hypot(s_i, s_c)
    prep.w_corr = s_c / t
    prep.w_indep = s_i / t
    prep.kind = _COMONOTONE if s_i == 0.0 else _BETA
    return prep


def _conditional_pmfs(prep, bucket_size, n, z):
    """Conditional bucket pmf of one unit for each shared-factor value.

    Returns a ``(len(z), n)`` array whose row q is the pmf of the unit's loss
    on buckets ``0, h, ..., (n - 1) h`` conditional on ``Z_c = z[q]``, bucketed
    by the round method. Every row sums to exactly 1.
    """
    q_n = z.size
    if prep.kind == _ATOM:
        pmf = np.zeros((q_n, n))
        pmf[:, int(round(prep.mean_loss / bucket_size))] = 1.0
        return pmf
    if prep.kind == _COMONOTONE:
        # U = Phi(w_corr z) with no independent noise: an atom per z value.
        u = norm.cdf(prep.w_corr * z)
        x = prep.max_loss * beta_dist.ppf(u, prep.alpha, prep.beta)
        k = np.rint(x / bucket_size).astype(np.intp)
        pmf = np.zeros((q_n, n))
        pmf[np.arange(q_n), k] = 1.0
        return pmf
    # General case: cdf differences at half-bucket edges. The normal score of
    # each edge, y = Phi^{-1}(F_beta(edge / M)), is computed once; edges below
    # 0 map to -inf and edges at or above M map to +inf, so each row is a
    # complete probability distribution with no truncation.
    edges = (np.arange(n + 1) - 0.5) * bucket_size
    y = norm.ppf(beta_dist.cdf(edges / prep.max_loss, prep.alpha, prep.beta))
    with np.errstate(invalid="ignore"):
        cdf = norm.cdf((y[None, :] - prep.w_corr * z[:, None]) / prep.w_indep)
    return np.diff(cdf, axis=1)


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
    meta: dict

    def frame(self):
        """Return the pmf as a DataFrame indexed by the two loss grids."""
        return pd.DataFrame(self.pmf, index=pd.Index(self.x1, name="x1"),
                            columns=pd.Index(self.x2, name="x2"))

    def marginal(self, unit):
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
        ps = np.bincount(idx.ravel(), weights=self.pmf.ravel(),
                         minlength=n1 + n2 - 1)
        h = self.x1[1] - self.x1[0] if self.x1.size > 1 else self.x2[1] - self.x2[0]
        return h * np.arange(n1 + n2 - 1), ps


def secondary_two_units(units, bucket_size, n1, n2, *, n_z=1001,
                        variance_policy="clamp",
                        clamp_factor=DEFAULT_CLAMP_FACTOR):
    """Joint bucketed loss distribution for one event across two units.

    Parameters
    ----------
    units : sequence of two Unit or 4-tuples
        Per-unit ELT fields in the order ``(mean_loss, sd_indep, sd_corr,
        max_loss)``, all in loss currency units.
    bucket_size : float
        Common grid spacing h; bucket k represents the loss ``k h``.
    n1, n2 : int
        Grid lengths for unit 1 and unit 2. Each grid must cover its unit's
        maximum loss, ``(n - 1) h >= M``; anything less raises, silent
        truncation is unacceptable.
    n_z : int, optional
        Number of midpoint quantile nodes for the shared factor ``Z_c``.
        Accuracy control; raise it when ``sd_corr`` dominates ``sd_indep``.
    variance_policy : {"clamp", "error", "atom"}, optional
        Treatment of rows whose stated sd is inadmissible for a beta,
        ``sigma^2 >= mu (M - mu)``: clamp sigma to ``clamp_factor`` times the
        bound (Burke's own repair, the default), raise, or collapse the unit
        to an atom at its mean.
    clamp_factor : float, optional
        Factor in (0, 1) applied to the sd bound under the clamp policy.

    Returns
    -------
    BivariateSU
        Joint pmf on the two loss grids plus diagnostics; see the class for
        the DataFrame, marginal, and sum-of-losses views.

    Notes
    -----
    Follows Burke, Rau-Chaplin, and Varghese (2016), Section III.E: total sd
    is the arithmetic sum of the independent and correlated sds, marginals are
    betas on ``[0, M]`` fit by moment matching to the normalized mean and sd,
    and dependence is the one-factor Gaussian copula built from the shared
    event factor. The computation integrates out the shared factor by midpoint
    quadrature instead of simulating, so results are deterministic and the
    marginals of the returned pmf agree with directly bucketing each beta up
    to quadrature error of order ``n_z**-2``.
    """
    if len(units) != 2:
        raise SecondaryError(f"expected exactly 2 units, got {len(units)}")
    h = float(bucket_size)
    if not (h > 0.0) or not math.isfinite(h):
        raise SecondaryError(f"bucket_size must be finite and positive, got {h}")
    if variance_policy not in VARIANCE_POLICIES:
        raise SecondaryError(f"unknown variance_policy {variance_policy!r}")
    if not (0.0 < clamp_factor < 1.0):
        raise SecondaryError(f"clamp_factor must be in (0, 1), got {clamp_factor}")
    q_n = int(n_z)
    if q_n < 1:
        raise SecondaryError(f"n_z must be >= 1, got {n_z}")
    lengths = (int(n1), int(n2))
    preps = []
    for i, (unit, n) in enumerate(zip(units, lengths), start=1):
        if n < 1:
            raise SecondaryError(f"n{i} must be >= 1, got {n}")
        prep = _prepare(unit, f"unit {i}", variance_policy, clamp_factor)
        if prep.max_loss > (n - 1) * h:
            raise SecondaryError(
                f"unit {i}: grid of {n} nodes at spacing {h} covers only "
                f"{(n - 1) * h}, but max_loss is {prep.max_loss}; "
                f"n{i} >= {math.ceil(prep.max_loss / h) + 1} required")
        preps.append(prep)

    # Midpoint quantile grid for the shared factor, equal weights 1 / n_z.
    z = norm.ppf((np.arange(q_n) + 0.5) / q_n)
    cond1 = _conditional_pmfs(preps[0], h, lengths[0], z)
    cond2 = _conditional_pmfs(preps[1], h, lengths[1], z)
    pmf = np.einsum("qi,qj->ij", cond1, cond2) / q_n

    x1 = h * np.arange(lengths[0])
    x2 = h * np.arange(lengths[1])
    marg1, marg2 = pmf.sum(axis=1), pmf.sum(axis=0)
    mean1, mean2 = marg1 @ x1, marg2 @ x2
    sd1 = math.sqrt(max(marg1 @ x1**2 - mean1**2, 0.0))
    sd2 = math.sqrt(max(marg2 @ x2**2 - mean2**2, 0.0))
    cross = x1 @ pmf @ x2
    corr = ((cross - mean1 * mean2) / (sd1 * sd2)
            if sd1 > 0.0 and sd2 > 0.0 else math.nan)
    meta = {
        "bucket_size": h,
        "n_z": q_n,
        "alpha": tuple(p.alpha for p in preps),
        "beta": tuple(p.beta for p in preps),
        "sigma_used": tuple(p.sigma_used for p in preps),
        "copula_normal_correlation": preps[0].w_corr * preps[1].w_corr,
        "total_mass": float(pmf.sum()),
        "grid_mean": (mean1, mean2),
        "analytic_mean": tuple(p.mean_loss for p in preps),
        "grid_sd": (sd1, sd2),
        "grid_correlation": corr,
        "notes": tuple(tuple(p.notes) for p in preps),
    }
    return BivariateSU(pmf=pmf, x1=x1, x2=x2, meta=meta)


def _monte_carlo(units, bucket_size, n1, n2, n_sims=2_000_000, seed=20260825,
                 variance_policy="clamp", clamp_factor=DEFAULT_CLAMP_FACTOR):
    """Simulated joint pmf on the same grids, for cross-checking only."""
    rng = np.random.default_rng(seed)
    z_c = rng.standard_normal(n_sims)
    losses = []
    for i, unit in enumerate(units, start=1):
        prep = _prepare(unit, f"unit {i}", variance_policy, clamp_factor)
        if prep.kind == _ATOM:
            losses.append(np.full(n_sims, prep.mean_loss))
            continue
        score = prep.w_corr * z_c + prep.w_indep * rng.standard_normal(n_sims)
        u = norm.cdf(score)
        losses.append(prep.max_loss * beta_dist.ppf(u, prep.alpha, prep.beta))
    k1 = np.rint(losses[0] / bucket_size).astype(np.intp)
    k2 = np.rint(losses[1] / bucket_size).astype(np.intp)
    counts = np.zeros((n1, n2))
    np.add.at(counts, (k1, k2), 1.0)
    return counts / n_sims


if __name__ == "__main__":
    # Demonstration and verification. Two admissible units sharing an event.
    u1 = Unit(mean_loss=200.0, sd_indep=100.0, sd_corr=150.0, max_loss=1000.0)
    u2 = Unit(mean_loss=500.0, sd_indep=300.0, sd_corr=200.0, max_loss=2000.0)
    bs, n1, n2 = 10.0, 101, 201
    res = secondary_two_units([u1, u2], bs, n1, n2)
    m = res.meta
    print("alpha:", m["alpha"], " beta:", m["beta"])
    print(f"total mass          {m['total_mass']:.15f}")
    print(f"grid means          {m['grid_mean'][0]:9.4f} {m['grid_mean'][1]:9.4f}"
          f"   analytic {m['analytic_mean'][0]:.1f} {m['analytic_mean'][1]:.1f}")
    print(f"grid sds            {m['grid_sd'][0]:9.4f} {m['grid_sd'][1]:9.4f}"
          f"   sigma_I + sigma_C {u1.sd_indep + u1.sd_corr:.1f} "
          f"{u2.sd_indep + u2.sd_corr:.1f}")
    print(f"copula normal corr  {m['copula_normal_correlation']:.6f}")
    print(f"grid loss corr      {m['grid_correlation']:.6f}")
    xs, ps = res.sum_distribution()
    print(f"sum: mass {ps.sum():.15f}  mean {ps @ xs:9.4f}"
          f"  (unit means sum to {u1.mean_loss + u2.mean_loss:.1f})")

    # Marginals of the joint against directly bucketed betas.
    for unit, n, axis in ((u1, n1, 1), (u2, n2, 2)):
        prep = _prepare(unit, f"unit {axis}", "clamp", DEFAULT_CLAMP_FACTOR)
        edges = (np.arange(n + 1) - 0.5) * bs
        direct = np.diff(beta_dist.cdf(edges / prep.max_loss,
                                       prep.alpha, prep.beta))
        err = np.abs(res.marginal(axis) - direct).max()
        print(f"unit {axis} marginal vs direct beta bucketing: max abs err {err:.2e}")

    # Independence sanity check: no correlated sd gives an outer product.
    ind = secondary_two_units([u1._replace(sd_corr=0.0),
                               u2._replace(sd_corr=0.0)], bs, n1, n2)
    outer = np.outer(ind.marginal(1), ind.marginal(2))
    print(f"sd_corr = 0: pmf vs outer product: max abs err "
          f"{np.abs(ind.pmf - outer).max():.2e}")

    # Comonotone path: unit 1 with no independent sd.
    com = secondary_two_units([u1._replace(sd_indep=0.0), u2], bs, n1, n2)
    print(f"comonotone unit 1: mass {com.meta['total_mass']:.12f}  "
          f"grid corr {com.meta['grid_correlation']:.4f}")

    # Monte Carlo cross-check of the full joint.
    mc = _monte_carlo([u1, u2], bs, n1, n2)
    print(f"Monte Carlo (2e6 sims): max abs cell err "
          f"{np.abs(res.pmf - mc).max():.2e}, "
          f"mean abs {np.abs(res.pmf - mc).mean():.2e}")
