"""DecL-style unit terms: grammar, parsing, and the four view maps.

One line per unit, borrowing the idioms of ``aggregate``'s DecL so the eye
reads it the same way (``xs``, ``po`` for part-of, ``occurrence net of``,
clauses joined by ``and``)::

    unit <unit_id> [<limit> xs <ded>] [occurrence net of <clause> (and <clause>)*]
    clause := [<share> po] <limit> xs <attach>
    share  := <number> | <number>%

Examples::

    unit plant_a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500
    unit plant_b 1000 xs 0 occurrence net of 100 po 400 xs 100
    unit fleet

Semantics (stated once here, tested in ``tests/test_terms.py``), with X the
unit's ground-up loss:

- ``ground_up`` = X, prior to policy terms;
- ``gross`` = min(max(X - ded, 0), limit), after deductible and limit;
- ``ceded`` = sum over clauses of share * min(max(gross - attach, 0), limit),
  every clause applied to gross independently (a standard XL tower reading;
  clauses may overlap as long as their combined share is at most 100% in
  every band, so that ceded never exceeds gross);
- ``net`` = gross - ceded.

Shares: a percent share is the fraction of the layer directly (``50% po 400
xs 100`` cedes half the layer); a plain-number share is an amount of the
layer limit, fraction amount / limit (``100 po 400 xs 100`` cedes 25%;
``.2 po 5 xs 1`` cedes 0.2 / 5). A clause with no share takes the whole
layer. A missing policy layer means ded = 0 and an unlimited limit (capped by
the unit's maximum loss in practice); a missing occurrence clause means ceded
is zero and net equals gross. ``inf`` is accepted as a limit.

Every view is a nondecreasing piecewise-linear map of X. Monetary breakpoints
(ded, limit, attach, layer limits) must be multiples of the grid spacing when
compiled onto a grid (:func:`check_grid_alignment`); there is no silent
rounding.

This module is pure Python with NumPy only (no SciPy), shared by both
backends. Structured input is accepted everywhere a terms string is: a
mapping ``unit_id -> UnitTerms`` or ``unit_id -> {"limit", "ded",
"clauses": [(share_fraction, limit, attach), ...]}``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

VIEWS = ("ground_up", "gross", "ceded", "net")
KEYWORDS = frozenset({"unit", "xs", "occurrence", "net", "of", "po", "and"})


class TermsError(ValueError):
    """Malformed terms text, invalid term values, or grid misalignment."""


@dataclass(frozen=True)
class Clause:
    """One occurrence reinsurance layer: ``share`` (fraction in (0, 1]) of
    ``limit`` xs ``attach``, applied to the unit's gross loss."""

    share: float
    limit: float
    attach: float

    def __post_init__(self):
        if not (0.0 < self.share <= 1.0):
            raise TermsError(f"clause share must be in (0, 1], got {self.share}")
        if not (math.isfinite(self.limit) and self.limit > 0.0):
            raise TermsError(f"clause limit must be finite and positive, got {self.limit}")
        if not (math.isfinite(self.attach) and self.attach >= 0.0):
            raise TermsError(f"clause attachment must be finite and >= 0, got {self.attach}")

    def apply(self, gross):
        """Ceded amount of this clause for gross loss(es) ``gross``."""
        return self.share * np.minimum(np.maximum(gross - self.attach, 0.0), self.limit)

    def text(self) -> str:
        share = "" if self.share == 1.0 else f"{_fmt(100.0 * self.share)}% po "
        return f"{share}{_fmt(self.limit)} xs {_fmt(self.attach)}"


@dataclass(frozen=True)
class UnitTerms:
    """Policy layer plus occurrence reinsurance clauses for one unit."""

    unit_id: str
    limit: float = math.inf
    ded: float = 0.0
    clauses: tuple[Clause, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise TermsError(f"unit_id must be a nonempty string, got {self.unit_id!r}")
        if math.isnan(self.limit) or self.limit <= 0.0:
            raise TermsError(f"unit {self.unit_id}: limit must be positive, got {self.limit}")
        if not (math.isfinite(self.ded) and self.ded >= 0.0):
            raise TermsError(f"unit {self.unit_id}: ded must be finite and >= 0, got {self.ded}")
        object.__setattr__(self, "clauses", tuple(self.clauses))
        self._check_overlap()

    def _check_overlap(self) -> None:
        """Clauses may overlap, but their combined share must not exceed
        100% of gross in any band: otherwise ceded exceeds gross and net
        decreases (or goes negative), which no loss grid can represent."""
        points = sorted({c.attach for c in self.clauses}
                        | {c.attach + c.limit for c in self.clauses})
        for lo, hi in zip(points, points[1:]):
            share = sum(c.share for c in self.clauses if c.attach <= lo and hi <= c.attach + c.limit)
            if share > 1.0 + 1e-12:
                raise TermsError(
                    f"unit {self.unit_id}: clauses cede {100.0 * share:.6g}% of gross "
                    f"between {_fmt(lo)} and {_fmt(hi)}; combined share must not exceed 100%"
                )

    @property
    def has_policy_terms(self) -> bool:
        return self.ded > 0.0 or math.isfinite(self.limit)

    @property
    def has_reinsurance(self) -> bool:
        return bool(self.clauses)

    # View maps: vectorized over NumPy arrays or scalars.
    def gross(self, x):
        return np.minimum(np.maximum(np.asarray(x, dtype=np.float64) - self.ded, 0.0), self.limit)

    def ceded(self, x):
        g = self.gross(x)
        c = np.zeros_like(g)
        for clause in self.clauses:
            c = c + clause.apply(g)
        return c

    def net(self, x):
        return self.gross(x) - self.ceded(x)

    def view(self, name: str, x):
        """Evaluate the named view map at ground-up loss(es) ``x``."""
        if name == "ground_up":
            return np.asarray(x, dtype=np.float64)
        if name == "gross":
            return self.gross(x)
        if name == "ceded":
            return self.ceded(x)
        if name == "net":
            return self.net(x)
        raise TermsError(f"unknown view {name!r}; expected one of {VIEWS}")

    def breakpoints(self) -> tuple[float, ...]:
        """Finite monetary breakpoints that must sit on the grid."""
        points = [self.ded]
        if math.isfinite(self.limit):
            points.append(self.limit)
        for clause in self.clauses:
            points.extend((clause.limit, clause.attach))
        return tuple(points)

    def text(self) -> str:
        """Canonical one-line rendering in the grammar above."""
        parts = [f"unit {self.unit_id}"]
        if self.has_policy_terms:
            parts.append(f"{_fmt(self.limit)} xs {_fmt(self.ded)}")
        if self.clauses:
            parts.append("occurrence net of " + " and ".join(c.text() for c in self.clauses))
        return " ".join(parts)


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.15g}"


def _number(token: str, what: str, line: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise TermsError(f"expected a number for {what}, got {token!r} in: {line}") from None
    if math.isnan(value):
        raise TermsError(f"{what} must not be NaN in: {line}")
    return value


def parse_terms_line(line: str) -> UnitTerms:
    """Parse one ``unit ...`` line into :class:`UnitTerms`."""
    tokens = line.split()
    if not tokens or tokens[0] != "unit":
        raise TermsError(f"terms line must start with 'unit': {line}")
    if len(tokens) < 2:
        raise TermsError(f"missing unit_id in: {line}")
    unit_id = tokens[1]
    if unit_id in KEYWORDS:
        raise TermsError(f"unit_id {unit_id!r} is a keyword in: {line}")
    pos = 2
    limit, ded = math.inf, 0.0
    # Optional policy layer: <limit> xs <ded>.
    if pos < len(tokens) and tokens[pos] != "occurrence":
        if pos + 2 >= len(tokens) or tokens[pos + 1] != "xs":
            raise TermsError(f"expected '<limit> xs <ded>' after unit_id in: {line}")
        limit = _number(tokens[pos], "policy limit", line)
        ded = _number(tokens[pos + 2], "deductible", line)
        pos += 3
    clauses: list[Clause] = []
    if pos < len(tokens):
        if tokens[pos:pos + 3] != ["occurrence", "net", "of"]:
            raise TermsError(f"expected 'occurrence net of' in: {line}")
        pos += 3
        while True:
            clause, pos = _parse_clause(tokens, pos, line)
            clauses.append(clause)
            if pos == len(tokens):
                break
            if tokens[pos] != "and":
                raise TermsError(f"expected 'and' between clauses in: {line}")
            pos += 1
            if pos == len(tokens):
                raise TermsError(f"dangling 'and' in: {line}")
    try:
        return UnitTerms(unit_id, limit, ded, tuple(clauses))
    except TermsError as error:
        raise TermsError(f"{error} in: {line}") from None


def _parse_clause(tokens: list[str], pos: int, line: str) -> tuple[Clause, int]:
    """Parse ``[<share> po] <limit> xs <attach>`` starting at ``pos``."""
    share_token = None
    if pos + 1 < len(tokens) and tokens[pos + 1] == "po":
        share_token = tokens[pos]
        pos += 2
    if pos + 2 >= len(tokens) or tokens[pos + 1] != "xs":
        raise TermsError(f"expected '<limit> xs <attach>' in clause of: {line}")
    limit = _number(tokens[pos], "clause limit", line)
    attach = _number(tokens[pos + 2], "clause attachment", line)
    pos += 3
    if not (math.isfinite(limit) and limit > 0.0):
        raise TermsError(f"clause limit must be finite and positive in: {line}")
    if share_token is None:
        share = 1.0
    elif share_token.endswith("%"):
        percent = _number(share_token[:-1], "percent share", line)
        if not (0.0 < percent <= 100.0):
            raise TermsError(f"percent share must be in (0, 100], got {percent} in: {line}")
        share = percent / 100.0
    else:
        amount = _number(share_token, "share amount", line)
        if not (0.0 < amount <= limit):
            raise TermsError(
                f"share amount must be in (0, limit], got {amount} for limit {limit} in: {line}"
            )
        share = amount / limit
    try:
        return Clause(share, limit, attach), pos
    except TermsError as error:
        raise TermsError(f"{error} in: {line}") from None


def _from_structured(unit_id: str, spec) -> UnitTerms:
    if isinstance(spec, UnitTerms):
        if spec.unit_id != unit_id:
            raise TermsError(f"terms for {unit_id!r} carry unit_id {spec.unit_id!r}")
        return spec
    if isinstance(spec, str):
        terms = parse_terms_line(spec)
        if terms.unit_id != unit_id:
            raise TermsError(f"terms for {unit_id!r} carry unit_id {terms.unit_id!r}")
        return terms
    if not isinstance(spec, Mapping):
        raise TermsError(f"terms for {unit_id!r} must be a string, mapping, or UnitTerms")
    unknown = set(spec) - {"limit", "ded", "clauses"}
    if unknown:
        raise TermsError(f"unknown keys {sorted(unknown)} in terms for {unit_id!r}")
    clauses = []
    for item in spec.get("clauses", ()):
        if isinstance(item, Clause):
            clauses.append(item)
        elif isinstance(item, Mapping):
            clauses.append(Clause(float(item.get("share", 1.0)), float(item["limit"]),
                                  float(item["attach"])))
        else:
            share, limit, attach = item
            clauses.append(Clause(float(share), float(limit), float(attach)))
    return UnitTerms(unit_id, float(spec.get("limit", math.inf)),
                     float(spec.get("ded", 0.0)), tuple(clauses))


def parse_terms(terms) -> dict[str, UnitTerms]:
    """Parse terms into ``{unit_id: UnitTerms}``.

    Parameters
    ----------
    terms : str | iterable of str | Mapping | None
        Terms text (one ``unit`` line per unit; blank lines and ``#``
        comments ignored), a sequence of such lines, a mapping from unit id
        to a line / structured dict / :class:`UnitTerms`, or None for no
        terms at all.

    Raises
    ------
    TermsError
        On malformed text, invalid values, or a unit id given twice.
    """
    parsed: dict[str, UnitTerms] = {}
    if terms is None:
        return parsed
    if isinstance(terms, Mapping):
        for unit_id, spec in terms.items():
            parsed[str(unit_id)] = _from_structured(str(unit_id), spec)
        return parsed
    lines = terms.splitlines() if isinstance(terms, str) else list(terms)
    for raw in lines:
        if not isinstance(raw, str):
            raise TermsError(f"terms lines must be strings, got {type(raw).__name__}")
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        unit_terms = parse_terms_line(line)
        if unit_terms.unit_id in parsed:
            raise TermsError(f"unit {unit_terms.unit_id!r} appears twice in terms")
        parsed[unit_terms.unit_id] = unit_terms
    return parsed


def check_grid_alignment(terms: UnitTerms, bucket_size: float) -> None:
    """Require every monetary breakpoint to be a multiple of ``bucket_size``."""
    for value in terms.breakpoints():
        ratio = value / bucket_size
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, abs(ratio)):
            raise TermsError(
                f"unit {terms.unit_id}: breakpoint {value} is not a multiple of "
                f"bucket_size {bucket_size}"
            )
