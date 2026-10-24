"""Result metadata dataclasses shared by the Rust and reference backends."""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default node density of the survival method (nodes per decade of loss),
#: shared by both backends: the smallest density whose exact-edge error is
#: within 1e-4 on every benchmark portfolio (`benchmarks/report-1.16.0.md`).
DEFAULT_NODES_PER_DECADE = 100

#: Survival method: the number of node intervals before an event's endpoint
#: that are left to its exact tail rather than to the interpolated curve
#: (spec section 5A). Shared by both backends.
TAIL_INTERVALS = 4

#: Survival method: the rate cut of a tail walk as a fraction of the beta
#: events' total rate; the walk along the bucket edges stops where the
#: event's exceedance rate is at or below it (spec section 5A). Shared by
#: both backends.
TAIL_RATE_FLOOR = 1e-20

#: Survival method: steps of one standard deviation each side of the mean
#: within which a narrow event's bulk must reach the rate floor; wider
#: events take the regular route (spec section 5A). Shared by both backends.
NARROW_STEPS = 40

#: Survival method: above this ``alpha + beta`` a narrow event's density
#: cannot be resolved in double precision and the event is a step at its
#: mean (spec section 5A). Shared by both backends.
NARROW_STEP_PARAMETER_SUM = 1e15


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
    overflow: str = "top_bucket"
    rate_above_grid: dict = field(default_factory=_zero_per_output)
    mean_above_grid: dict = field(default_factory=_zero_per_output)


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
    z_nodes: int
    z_rule: str
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
    z_nodes: int
    z_rule: str
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
    unit_ids: tuple
    terms_echo: dict
    unused_terms: tuple
    diagnostics: dict[str, RowIssues]
    roll_up: list | None = None
    #: Rust engine statistics (max per-event mass error, convolution paths);
    #: None on the reference backend.
    engine: dict | None = None
