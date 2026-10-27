"""Independent oracles for spec test 36, the shared-event model moments.

Run from the repository root:

    uv run python fixtures/generate_moment_oracles.py

Prints the raw moments P(S), P(S^2), P(S^3) of each test event's total,
computed by methods that share nothing with the engine's score tables and
rules (`dev/done/plan-model-moments-and-windowed-views.md`, test 36):

- two-point units: ``P(X_u^a X_w^b) = M_u^a M_w^b P(V_u > y_u, V_w > y_w)``,
  the bivariate normal at correlation ``w_u w_w`` (Owen's T, the reference's
  ``bivariate_normal_cdf``), and the triple by a one-dimensional quadrature
  over the factor of the product of the conditional probabilities;
- a beta and a two-point unit (Homer and Li 2017, Table 11, events 1 and 2):
  ``P(X_u^a; V_w > y_w) = M_u^a int_0^1 a y^(a-1) P(Y_u > y, V_w > y_w) dy``,
  one-dimensional integrals of the bivariate normal;
- two beta units: Hoeffding's identity ``P(X_u^a X_w^b) = int int a x^(a-1)
  b x'^(b-1) P(X_u > x, X_w > x') dx dx'`` in the units' score coordinates,
  ``x = M F^-1(Phi(s))``, ``dx = M phi(s) / f(x / M) ds``, by tensor
  Gauss-Legendre on panels, converged by doubling;
- the comonotone coupling: ``P(S^r) = int (sum_u q_u(Phi(s)))^r phi(s) ds``,
  the units' quantile functions along one score.

The test file embeds the printed values.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special, stats

from eltq import reference_multiunit as rm


def prep(unit):
    return rm.couple(rm.prepare_unit(rm.Unit(*unit), "u"), "burke")


def two_point_pair(u, w):
    """``P(X_u^a X_w^b)`` for a, b in 1..2 of two two-point units."""
    pu, pw = prep(u), prep(w)
    yu, yw = special.ndtri(1.0 - pu.two_point_p), special.ndtri(1.0 - pw.two_point_p)
    both = float(rm.bivariate_normal_cdf(-yu, -yw, pu.w_corr * pw.w_corr))
    return {(a, b): pu.max_loss ** a * pw.max_loss ** b * both for a in (1, 2) for b in (1, 2)}


def two_point_triple(units):
    preps = [prep(u) for u in units]

    def integrand(z):
        out = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        for p in preps:
            out *= float(rm.two_point_conditional(p.two_point_p, p.w_corr, p.w_indep, np.array([z]))[0])
        return out

    prob = integrate.quad(integrand, -12.0, 12.0, epsabs=0.0, epsrel=1e-13, limit=400)[0]
    return math.prod(p.max_loss for p in preps) * prob


def beta_two_point(beta_unit, two_unit):
    """``P(X_b^a X_t^c)`` for a in 1..2 (and the mirror) of a beta unit and
    a two-point unit: ``M_t^c P(X_b^a; V_t > y_t)``."""
    b, t = prep(beta_unit), prep(two_unit)
    yt = special.ndtri(1.0 - t.two_point_p)
    rho = b.w_corr * t.w_corr
    law = stats.beta(b.alpha, b.beta)

    def joint_tail(y):
        s = -special.ndtri(law.sf(y)) if law.sf(y) < 0.5 else special.ndtri(law.cdf(y))
        return float(rm.bivariate_normal_cdf(-s, -yt, rho))

    partial = {}
    for a in (1, 2):
        f = lambda y, a=a: a * y ** (a - 1) * joint_tail(y)
        points = [law.ppf(q) for q in (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-6)]
        value = integrate.quad(f, 0.0, 1.0, points=sorted(set(points)), epsabs=0.0, epsrel=1e-12, limit=800)[0]
        partial[a] = b.max_loss ** a * value
    return {(a, c): t.max_loss ** c * partial[a] for a in (1, 2) for c in (1, 2)}


def beta_pair(u, w, panels=48, order=24):
    """``P(X_u^a X_w^b)`` of two beta units by Hoeffding's identity in score
    coordinates, tensor Gauss-Legendre with ``panels`` panels of ``order``
    points on [-9, 9] in each score."""
    pu, pw = prep(u), prep(w)
    rho = pu.w_corr * pw.w_corr
    x, wt = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-9.0, 9.0, panels + 1)
    s = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
    ws = np.concatenate([0.5 * (b - a) * wt for a, b in zip(edges[:-1], edges[1:])])

    def coords(p):
        law = stats.beta(p.alpha, p.beta)
        y = law.ppf(special.ndtr(s))
        with np.errstate(divide="ignore", invalid="ignore"):
            dy = np.exp(-0.5 * s * s) / math.sqrt(2 * math.pi) / law.pdf(y)
        # At the ends of [0, 1] in double precision the score's probability
        # is below 1e-18; those points carry nothing.
        dy = np.where(np.isfinite(dy) & (y > 0.0) & (y < 1.0), dy, 0.0)
        return p.max_loss * y, p.max_loss * dy * ws

    xu, du = coords(pu)
    xw, dw = coords(pw)
    joint = rm.bivariate_normal_cdf(-s[:, None], -s[None, :], rho)
    out = {}
    for a in (1, 2):
        for b in (1, 2):
            wa = a * xu ** (a - 1) * du
            wb = b * xw ** (b - 1) * dw
            out[(a, b)] = float(wa @ joint @ wb)
    return out


def event_moments(units, cross):
    """``(P S, P S^2, P S^3)`` of a two-unit total from the closed-form
    marginal moments and the cross moments ``cross[(a, b)]``."""
    m = [rm.law_moments_of(rm._unit_law(prep(u))) for u in units]
    first = m[0][0] + m[1][0]
    second = m[0][1] + m[1][1] + 2.0 * cross[(1, 1)]
    third = m[0][2] + m[1][2] + 3.0 * cross[(2, 1)] + 3.0 * cross[(1, 2)]
    return first, second, third


def comonotone(units):
    preps = [prep(u) for u in units]
    laws = [stats.beta(p.alpha, p.beta) for p in preps]

    def total(s):
        v = special.ndtr(s)
        return sum(p.max_loss * law.ppf(v) for p, law in zip(preps, laws))

    out = []
    for r in (1, 2, 3):
        f = lambda s, r=r: total(s) ** r * math.exp(-0.5 * s * s) / math.sqrt(2 * math.pi)
        out.append(integrate.quad(f, -12.0, 12.0, epsabs=0.0, epsrel=1e-13, limit=800, points=[0.0])[0])
    return tuple(out)


TWO_POINT = [(30.0, 50.0, 30.0, 100.0), (20.0, 40.0, 60.0, 200.0), (50.0, 100.0, 100.0, 400.0)]
BETAS = [(200.0, 100.0, 150.0, 1000.0), (500.0, 300.0, 200.0, 2000.0)]
HOMER = {
    1: ((300.0, 400.0, 300.0, 3000.0), (200.0, 300.0, 200.0, 1000.0)),
    2: ((100.0, 371.0, 267.0, 1000.0), (200.0, 150.0, 533.0, 4000.0)),
    3: ((100.0, 224.0, 200.0, 2000.0), (100.0, 200.0, 200.0, 2000.0)),
}


def main():
    np.set_printoptions(precision=17)
    pair = two_point_pair(TWO_POINT[0], TWO_POINT[1])
    print("two two-point units", event_moments(TWO_POINT[:2], pair))
    print("triple probability times M product", two_point_triple(TWO_POINT))
    for level in ((24, 16), (48, 24)):
        pair = beta_pair(*BETAS, *level)
        print("two betas", level, event_moments(BETAS, pair))
    for event, (personal, commercial) in HOMER.items():
        kinds = [prep(u).kind for u in (personal, commercial)]
        if kinds[1] == rm._TWO_POINT:
            cross = beta_two_point(personal, commercial)
        elif kinds[0] == rm._TWO_POINT:
            mirror = beta_two_point(commercial, personal)
            cross = {(a, b): mirror[(b, a)] for a, b in mirror}
        else:
            for level in ((48, 24), (96, 32)):
                cross = beta_pair(personal, commercial, *level)
                print(f"homer event {event} {level}", event_moments((personal, commercial), cross))
            continue
        print(f"homer event {event}", event_moments((personal, commercial), cross))
    print("comonotone betas", comonotone(BETAS))


if __name__ == "__main__":
    main()
