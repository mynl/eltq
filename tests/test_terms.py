"""Terms grammar: parsing, canonical rendering, view semantics, alignment.

Pure-Python tests of ``eltq.terms`` (no backend involved) plus the committed
parse goldens in ``fixtures/golden/terms_parse.json`` that the Rust parser
(1.9.0) is held to as well.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from eltq.terms import (
    VIEWS,
    Clause,
    TermsError,
    UnitTerms,
    check_grid_alignment,
    parse_terms,
    parse_terms_line,
)

from conftest import GOLDEN_DIR

TOWER = "unit plant_a 750 xs 250 occurrence net of 50% po 400 xs 100 and 500 xs 500"


def test_parse_full_line():
    t = parse_terms_line(TOWER)
    assert t.unit_id == "plant_a"
    assert (t.limit, t.ded) == (750.0, 250.0)
    assert t.clauses == (Clause(0.5, 400.0, 100.0), Clause(1.0, 500.0, 500.0))
    assert t.has_policy_terms and t.has_reinsurance


def test_parse_minimal_and_partial_lines():
    bare = parse_terms_line("unit fleet")
    assert bare == UnitTerms("fleet")
    assert not bare.has_policy_terms and not bare.has_reinsurance
    assert math.isinf(bare.limit) and bare.ded == 0.0
    only_layer = parse_terms_line("unit e 600 xs 50")
    assert (only_layer.limit, only_layer.ded, only_layer.clauses) == (600.0, 50.0, ())
    only_reins = parse_terms_line("unit d occurrence net of 300 xs 100")
    assert math.isinf(only_reins.limit) and only_reins.clauses == (Clause(1.0, 300.0, 100.0),)
    unlimited = parse_terms_line("unit x inf xs 100")
    assert math.isinf(unlimited.limit) and unlimited.ded == 100.0


def test_share_conventions():
    # Percent: fraction of the layer. Amount: amount / layer limit.
    assert parse_terms_line("unit a occurrence net of 50% po 400 xs 100").clauses[0].share == 0.5
    assert parse_terms_line("unit a occurrence net of 100 po 400 xs 100").clauses[0].share == 0.25
    assert parse_terms_line("unit a occurrence net of .2 po 5 xs 1").clauses[0].share == 0.2 / 5.0
    assert parse_terms_line("unit a occurrence net of 100% po 5 xs 1").clauses[0].share == 1.0
    assert parse_terms_line("unit a occurrence net of 5 po 5 xs 1").clauses[0].share == 1.0


@pytest.mark.parametrize("line", [
    "unit a occurrence net of 150% po 400 xs 100",
    "unit a occurrence net of 0% po 400 xs 100",
    "unit a occurrence net of 500 po 400 xs 100",
    "unit a occurrence net of 0 po 400 xs 100",
])
def test_share_bounds_are_enforced(line):
    with pytest.raises(TermsError):
        parse_terms_line(line)


def test_canonical_text_round_trips():
    for line in (TOWER, "unit fleet", "unit e 600 xs 50",
                 "unit b 1000 xs 0 occurrence net of 100 po 400 xs 100"):
        t = parse_terms_line(line)
        assert parse_terms_line(t.text()) == t
    assert parse_terms_line("unit b 1000 xs 0 occurrence net of 100 po 400 xs 100").text() \
        == "unit b 1000 xs 0 occurrence net of 25% po 400 xs 100"


def test_parse_terms_text_forms_agree():
    text = f"# comment\n\n{TOWER}\nunit fleet   # trailing comment\n"
    from_text = parse_terms(text)
    from_lines = parse_terms([TOWER, "unit fleet"])
    structured = parse_terms({
        "plant_a": {"limit": 750, "ded": 250, "clauses": [(0.5, 400, 100), (1.0, 500, 500)]},
        "fleet": {},
    })
    mixed = parse_terms({"plant_a": TOWER, "fleet": UnitTerms("fleet")})
    assert from_text == from_lines == structured == mixed
    assert parse_terms(None) == {}


def test_parse_errors():
    with pytest.raises(TermsError):
        parse_terms([TOWER, TOWER])  # duplicate unit
    with pytest.raises(TermsError):
        parse_terms({"fleet": {"limit": 100, "bogus": 1}})
    with pytest.raises(TermsError):
        parse_terms({"fleet": UnitTerms("other")})
    with pytest.raises(TermsError):
        parse_terms({"fleet": "unit other"})
    with pytest.raises(TermsError):
        parse_terms([42])
    with pytest.raises(TermsError):
        UnitTerms("xs")  # keyword as unit id is fine structurally, but not in text
        parse_terms_line("unit xs 100 xs 0")


def test_view_semantics_on_points():
    t = parse_terms_line(TOWER)
    x = np.array([0.0, 100.0, 250.0, 500.0, 1000.0, 5000.0])
    gross = np.minimum(np.maximum(x - 250.0, 0.0), 750.0)
    ceded = 0.5 * np.minimum(np.maximum(gross - 100.0, 0.0), 400.0) \
        + np.minimum(np.maximum(gross - 500.0, 0.0), 500.0)
    assert np.array_equal(t.view("ground_up", x), x)
    assert np.array_equal(t.view("gross", x), gross)
    assert np.array_equal(t.view("ceded", x), ceded)
    assert np.array_equal(t.view("net", x), gross - ceded)
    # Worked value: X = 1000 -> gross 750, ceded 200 + 250 = 450, net 300.
    assert (float(t.gross(1000.0)), float(t.ceded(1000.0)), float(t.net(1000.0))) \
        == (750.0, 450.0, 300.0)
    with pytest.raises(TermsError):
        t.view("bogus", x)


def test_views_are_nondecreasing_and_bounded():
    t = parse_terms_line("unit o 900 xs 100 occurrence net of 60% po 300 xs 100 and 40% po 200 xs 200")
    x = np.linspace(0.0, 2000.0, 4001)
    for view in VIEWS:
        v = t.view(view, x)
        assert np.all(np.diff(v) >= -1e-12)
        assert np.all(v <= x + 1e-12) and np.all(v >= 0.0)
    assert np.all(t.ceded(x) <= t.gross(x) + 1e-12)


@pytest.mark.parametrize("line", [
    "unit o occurrence net of 300 xs 100 and 200 xs 200",          # 200% in [200, 400)
    "unit o occurrence net of 60% po 300 xs 100 and 50% po 200 xs 200",
    "unit o occurrence net of 100 xs 0 and 100 xs 0",
])
def test_overlapping_clauses_cannot_exceed_gross(line):
    # Without the guard net would decrease and could go negative.
    with pytest.raises(TermsError):
        parse_terms_line(line)
    parse_terms_line("unit o occurrence net of 300 xs 100 and 200 xs 400")  # stacked tower
    parse_terms_line("unit o occurrence net of 50% po 300 xs 100 and 50% po 300 xs 100")


def test_grid_alignment():
    t = parse_terms_line(TOWER)
    check_grid_alignment(t, 50.0)
    check_grid_alignment(t, 10.0)
    with pytest.raises(TermsError):
        check_grid_alignment(t, 300.0)
    with pytest.raises(TermsError):
        check_grid_alignment(parse_terms_line("unit a 75 xs 0"), 10.0)
    check_grid_alignment(parse_terms_line("unit a inf xs 0"), 7.0)  # inf is no breakpoint


def test_parse_goldens():
    golden = json.loads((GOLDEN_DIR / "terms_parse.json").read_text(encoding="utf-8"))
    for case in golden["ok"]:
        t = parse_terms_line(case["line"])
        assert t.unit_id == case["unit_id"]
        assert (case["limit"] is None and math.isinf(t.limit)) or t.limit == case["limit"]
        assert t.ded == case["ded"]
        assert [[c.share, c.limit, c.attach] for c in t.clauses] == case["clauses"]
        assert t.text() == case["text"]
    for line in golden["errors"]:
        with pytest.raises(TermsError):
            parse_terms_line(line)
