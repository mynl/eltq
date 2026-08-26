"""Result metadata dataclasses shared by the Rust and reference backends."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    rate-weighted units. Diagnostic examples are event ids when available,
    0-based row indices otherwise.
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
