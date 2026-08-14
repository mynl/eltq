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
