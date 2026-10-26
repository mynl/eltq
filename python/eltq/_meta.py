"""Result metadata dataclasses shared by the Rust and reference backends."""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default node density of the survival method (nodes per decade of loss),
#: shared by both backends: the smallest density whose exact-edge error is
#: within 1e-4 on every benchmark portfolio (`benchmarks/report-1.16.0.md`).
DEFAULT_NODES_PER_DECADE = 100

#: Survival method: nodes per standard deviation a bell-shaped event asks of
#: the fine mesh around its mean, the method's accuracy knob (spec section
#: 5A): twelve holds the shared cubic within 1e-4 relative where the event's
#: survival is at least 1e-4 and within 1e-3 where it is at least 1e-8.
#: Shared by both backends; mirrors ``DEFAULT_NODES_PER_SIGMA`` in
#: ``crates/eltq-core/src/config.rs``.
DEFAULT_NODES_PER_SIGMA = 12

#: Survival method: a bell-shaped event's upper skip, ``2**-60``; from the
#: first mesh node where its survival is below this it contributes nothing.
#: Shared by both backends; mirrors ``UPPER_SKIP_SURVIVAL`` in
#: ``crates/eltq-core/src/curve.rs``.
UPPER_SKIP_SURVIVAL = 2.0 ** -60

#: Survival method: the number of node intervals before an event's endpoint
#: that are left to its exact tail rather than to the interpolated curve
#: (spec section 5A). Shared by both backends.
TAIL_INTERVALS = 4

#: Survival method: the rate cut of a tail walk as a fraction of the beta
#: events' total rate; the walk along the bucket edges stops where the
#: event's exceedance rate is at or below it (spec section 5A). Shared by
#: both backends.
TAIL_RATE_FLOOR = 1e-20

#: Survival method: where the endpoint expansion's first correction ``c t``,
#: ``c = (alpha - 1) beta / (beta + 1)``, lies in this band and ``c > 1``,
#: the tail ladder steps by ``2**(1/4)`` instead of ``sqrt(2)`` (spec
#: section 5A). Shared by both backends; mirrors ``TAIL_FINE_ZONE`` in
#: ``crates/eltq-core/src/curve.rs``.
TAIL_FINE_ZONE = (1e-2, 1e2)

#: Survival method: the power law below the tail ladder's last node applies
#: only where ``c t`` at that node is at most this; otherwise the queried
#: point is evaluated exactly (spec section 5A). Shared by both backends;
#: mirrors ``TAIL_POWER_LAW_LIMIT`` in ``crates/eltq-core/src/curve.rs``.
TAIL_POWER_LAW_LIMIT = 1e-6

#: Survival method: the narrow route's quadrature needs a density whose
#: derivative is finite at the endpoints, so both parameters at least this;
#: between 1 and 2 the 8-point rule was off by up to 5e-4 (spec section 5A).
#: Shared by both backends; mirrors ``NARROW_MIN_PARAMETER`` in
#: ``crates/eltq-core/src/curve.rs``.
NARROW_MIN_PARAMETER = 2.0

#: Survival method: steps of one standard deviation each side of the mean
#: within which a narrow event's bulk must reach the rate floor; wider
#: events take the regular route (spec section 5A). Shared by both backends.
NARROW_STEPS = 40

#: Survival method: from this smaller parameter a narrow event's kernel is
#: the standardized normal law in closed form instead of the quadrature of
#: its density, whose two logarithms cancel to about
#: ``4e-15 sqrt(min(alpha, beta))`` (spec section 5A). Shared by both
#: backends; mirrors ``NARROW_NORMAL_FROM`` in ``crates/eltq-core/src/curve.rs``.
NARROW_NORMAL_FROM = 1e16

#: A beta row whose standard deviation is below this fraction of its mean
#: (2^-48, sixteen ulps of the loss at the mean) is an atom at the mean,
#: reported under ``atom_by_resolution`` (spec section 4). Shared by both
#: backends; mirrors ``ATOM_RESOLUTION`` in ``crates/eltq-core/src/validate.rs``.
ATOM_RESOLUTION = 2.0 ** -48

#: Quantiles method (and the multi-unit beta units): from this smaller
#: parameter a beta quantile comes from the Cornish-Fisher expansion in the
#: beta's cumulants, not from inverting the incomplete beta, whose continued
#: fraction no longer converges there (spec section 5). Shared by both
#: backends; mirrors ``LARGE_PARAMETER_FROM`` in ``crates/eltq-core/src/batched.rs``.
LARGE_PARAMETER_FROM = 2e5


#: Multi-unit route, the stratified factor rule (spec section 4A): the
#: probability of the factor covered by the ``z_nodes`` equal body strata;
#: above it the upper tail is a geometric ladder of ratio
#: ``STRATA_TAIL_RATIO`` down to ``STRATA_TAIL_FLOOR`` (2^-24). Shared by
#: both backends; mirror ``STRATA_BODY_TOP``, ``STRATA_TAIL_RATIO`` and
#: ``STRATA_TAIL_FLOOR`` in ``crates/eltq-core/src/quadrature.rs``
#: (`benchmarks/z-convergence-2.6.0.md`).
STRATA_BODY_TOP = 0.75
STRATA_TAIL_RATIO = 2.0 ** 0.5
STRATA_TAIL_FLOOR = 2.0 ** -24


def _zero_per_output() -> dict:
    return {"raw": 0.0, "secondary": 0.0}


@dataclass
class RowIssues:
    """Count plus a bounded set of representative event ids for one category."""

    count: int = 0
    examples: list = field(default_factory=list)


@dataclass
class Meta:
    """Metadata and diagnostics accompanying the two grid distributions.

    Masses and means describe the arrays actually returned (normalized when
    ``normalized``); ``total_rate``, ``analytic_aal``,
    ``quadrature_secondary_mean`` and ``secondary_mean_error`` are always in
    rate-weighted units. ``secondary_mean_error`` is the secondary grid's
    first moment less ``analytic_aal`` under either method. Diagnostic
    examples are event ids when available, 0-based row indices otherwise.

    ``method`` is ``quantiles`` or ``survival``. Under ``quantiles``,
    ``quadrature_points`` and ``quadrature_secondary_mean`` are set and the
    survival fields are None; under ``survival`` the reverse: ``nodes`` (the
    node count), ``nodes_per_decade`` and ``interpolation_error`` (the
    sampled relative error of the interpolant). ``overflow`` is the policy
    applied; ``rate_above_grid`` and ``mean_above_grid`` give, per output,
    the overflow's rate and its first moment in excess of the top node (both
    exactly zero when the grid length was derived from the input).
    """

    bucket_size: float
    grid_length: int
    quadrature_points: int | None
    normalized: bool
    total_rate: float
    analytic_aal: float
    raw_mass: float
    raw_mean: float
    secondary_mass: float
    secondary_mean: float
    quadrature_secondary_mean: float | None
    secondary_mean_error: float
    input_rows: int
    valid_rows: int
    diagnostics: dict[str, RowIssues]
    method: str = "quantiles"
    nodes: int | None = None
    nodes_per_decade: int | None = None
    interpolation_error: float | None = None
    narrow_events: int | None = None
    fine_nodes: int | None = None
    requesting_events: int | None = None
    overflow: str = "top_bucket"
    rate_above_grid: dict = field(default_factory=_zero_per_output)
    mean_above_grid: dict = field(default_factory=_zero_per_output)


@dataclass
class ModelMoments:
    """Model moments of an ELT with no loss grid (spec section 4B).

    ``raw_moments`` are the rate-weighted ``C_r = sum lambda P(X^r)`` for
    ``r = 1, 2, 3``, the annual cumulants under compound Poisson occurrence;
    ``severity_moments`` are ``C_r / total_rate``, the severity's raw
    moments (NaN when the total rate is zero); ``analytic_aal`` is ``C_1``;
    ``rate_at_zero`` is ``sum lambda P(X = 0)``, so the probability of a year
    without loss is ``exp(-(total_rate - rate_at_zero))``. ``support`` is the
    largest loss of each row's effective law (``mu`` for an atom, the exposed
    value otherwise; NaN for a dropped row), and ``max_support`` its
    maximum. An order whose moment leaves double precision is infinite and
    flagged in ``moment_overflow``. Diagnostic examples are event ids when
    available, 0-based row indices otherwise.

    The integration fields describe how the moments were obtained:
    ``method`` is ``closed_form`` for a one-row ELT, whose moments are exact
    (``achieved_error`` zero, ``converged`` True).
    """

    total_rate: float
    raw_moments: tuple
    severity_moments: tuple
    analytic_aal: float
    rate_at_zero: float
    max_support: float
    support: object
    input_rows: int
    valid_rows: int
    atom_rows: int
    beta_rows: int
    two_point_rows: int
    dropped_rows: int
    moment_overflow: tuple
    diagnostics: dict[str, RowIssues]
    method: str = "closed_form"
    #: Requested relative tolerance on ``C_2`` and ``C_3`` (None when closed form).
    rtol: float | None = None
    #: Achieved relative errors of ``C_2`` and ``C_3``.
    achieved_error: tuple = (0.0, 0.0)
    converged: bool = True


@dataclass
class CurveMeta:
    """Metadata of an exceedance curve: its node range and count, the
    sampled interpolation error, the totals over the contributing rows and
    the row diagnostics. ``total_rate`` and ``analytic_aal`` are in
    rate-weighted units."""

    min_loss: float
    max_loss: float
    nodes: int
    nodes_per_decade: int
    interpolation_error: float
    total_rate: float
    analytic_aal: float
    input_rows: int
    valid_rows: int
    diagnostics: dict[str, RowIssues]
    narrow_events: int = 0
    fine_nodes: int = 0
    requesting_events: int = 0


@dataclass
class GridMeta:
    """Metadata of a grid made from an exceedance curve. ``secondary_mass``
    and ``secondary_mean`` describe the array returned (normalized when
    ``normalized``); ``secondary_mean_error`` is the grid's first moment in
    rate units less the curve's analytic AAL; the overflow figures are the
    rate beyond the top bucket and its first moment in excess of the top
    node."""

    bucket_size: float
    grid_length: int
    normalized: bool
    overflow: str
    secondary_mass: float
    secondary_mean: float
    secondary_mean_error: float
    rate_above_grid: float
    mean_above_grid: float


@dataclass
class JointMeta:
    """Parameters and moment diagnostics of a two-unit joint pmf.

    Tuples are per unit (unit 1, unit 2). ``grid_*`` moments are computed
    from the returned pmf; ``analytic_mean`` is the input means.
    """

    bucket_size: float
    #: The coupling rule of spec section 4A.
    coupling: str
    #: Under ``coupling="matched"``: ``scale``, ``sd_target``,
    #: ``sd_achieved`` and ``saturated`` for the event; None otherwise.
    matched: dict | None
    z_nodes: int
    z_rule: str
    #: Nodes or strata the rule actually used (``z_nodes`` body strata plus
    #: the tail ladder under the stratified rule).
    z_strata: int
    alpha: tuple
    beta: tuple
    sigma_used: tuple
    copula_normal_correlation: float
    total_mass: float
    grid_mean: tuple
    analytic_mean: tuple
    grid_sd: tuple
    grid_correlation: float
    notes: tuple


@dataclass
class PairMeta:
    """Metadata of the rate-weighted joint occurrence severity of two units
    (spec section 4A, bivariate outputs): ``pmf[j, k]`` is the rate at which
    an occurrence loses ``x_u[j]`` on the first unit and ``x_w[k]`` on the
    second, for the view. ``mass`` is the rate of the events exposing either
    unit, the Poisson rate of the pair's own occurrences (an event exposing
    neither contributes nothing); the grid moments are those of ``pmf /
    mass``."""

    bucket_size: float
    units: tuple
    view: str
    coupling: str
    z_nodes: int
    z_rule: str
    z_strata: int
    grid_lengths: tuple
    mass: float
    events: int
    grid_mean: tuple
    grid_sd: tuple
    grid_correlation: float


@dataclass
class FirstMomentMeta:
    """Metadata of the rate-weighted first-moment vectors ``m_u(y) = sum_j
    lambda_j P(X_uj; X_j = y)`` on the grid of the view's total (spec
    section 4A): ``sum_u m_u(y) = y s(y)`` with ``s`` the view's severity,
    whose largest violation over the grid, rate-weighted and in loss units,
    is ``identity_error``."""

    bucket_size: float
    grid_length: int
    units: tuple
    view: str
    coupling: str
    z_nodes: int
    z_rule: str
    z_strata: int
    total_rate: float
    identity_error: float
    #: ``sum_u sum_y m_u(y)``, the rate-weighted mean of the view's total.
    total_mean: float


@dataclass
class MultiMeta:
    """Metadata and diagnostics accompanying the per-view grid distributions.

    ``view_mass`` and ``view_mean`` describe the arrays actually returned
    (normalized when ``normalized``); ``total_rate``, ``analytic_aal``,
    ``analytic_view_mean`` and ``view_mean_error`` are always in
    rate-weighted units (``view_mean_error`` compares the returned array's
    mean, rescaled by the total rate when normalized, to the analytic beta
    layer mean of the view). Diagnostic examples are ``event_id/unit_id``
    strings. ``roll_up`` is None unless the roll-up diagnostic was requested.
    """

    bucket_size: float
    grid_length: int
    #: The coupling rule of spec section 4A.
    coupling: str
    z_nodes: int
    z_rule: str
    #: Nodes or strata the rule actually used.
    z_strata: int
    views: tuple
    normalized: bool
    total_rate: float
    analytic_aal: float
    view_mass: dict
    view_mean: dict
    analytic_view_mean: dict
    view_mean_error: dict
    input_rows: int
    valid_rows: int
    events: int
    valid_events: int
    #: Events with one contributing unit, bucketed directly, outside the
    #: mixture over the factor (spec section 4A).
    one_unit_events: int
    #: Under ``coupling="homer"`` the rate-weighted expected allocation over
    #: the units' exposures, ``sum_j lambda_j P(sum_k (s_k X_R - M_k)^+)``;
    #: None otherwise.
    homer_excess_mean: float | None
    #: Under ``coupling="matched"`` the events whose target variance was out
    #: of reach, each ``{"event_id", "scale", "sd_target", "sd_achieved"}``
    #: at saturation; None otherwise.
    matched_shortfall: list | None
    unit_ids: tuple
    terms_echo: dict
    unused_terms: tuple
    diagnostics: dict[str, RowIssues]
    roll_up: list | None = None
    #: Rust engine statistics (max per-event mass error, convolution paths);
    #: None on the reference backend.
    engine: dict | None = None
