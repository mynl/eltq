"""Benchmark of the grid-free model moments and tail measures (2.11.0 on).

Run from the repository root (release wheel installed by `uv sync`):

    uv run python benchmarks/run_moments.py [--rows 1000000] [--edges 64] [--no-append]

Measures, on the three seeded one-row portfolios of ``run_end_to_end.py``
(mixture, small alpha, large exposed values), ``moments_arrays`` and
``tail_moments_arrays`` at ``--edges`` log-spaced candidate edges between a
thousandth of the largest exposed value and the value itself, with all
threads and with one; checks the tail query bitwise across thread counts;
prints a summary and appends a machine-readable record to
``benchmarks/results.json`` under ``"suite": "moments"``. With
``--multiunit`` it also times ``multiunit_moments`` under each coupling on
the portfolio of ``run_multiunit.py`` (``--events`` events of ten units),
one thread against all, beside one ``multiunit_views`` run of the ground-up
view. The plan's budget
(``dev/done/plan-model-moments-and-windowed-views.md``, open question 4):
a million rows by 64 edges well under a second.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eltq

sys.path.insert(0, str(Path(__file__).parent))
from run_end_to_end import (  # noqa: E402
    cols_tuple, large_m_elt, machine_info, peak_working_set_mb, small_alpha_elt, synthetic_elt, timed,
)

RESULTS_PATH = Path(__file__).parent / "results.json"

PORTFOLIOS = {
    "mixture": synthetic_elt,
    "small_alpha": small_alpha_elt,
    "large_m": large_m_elt,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--edges", type=int, default=64)
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--multiunit", action="store_true",
                        help="also time multiunit_moments on the multi-unit benchmark portfolio")
    parser.add_argument("--events", type=int, default=100_000)
    args = parser.parse_args()
    if eltq.BACKEND != "rust":
        raise SystemExit("the benchmark needs the Rust backend")
    record: dict = {
        "suite": "moments",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        "rows": args.rows,
        "edges": args.edges,
        "runs": {},
    }
    for name, make in PORTFOLIOS.items():
        cols = cols_tuple(make(args.rows))
        top = float(np.max(cols[1]))
        edges = np.geomspace(1e-3 * top, top, args.edges)
        nodes = edges * (1.0 - 1e-4)
        seconds_moments, moments = timed(lambda: eltq.moments_arrays(*cols), 3)
        seconds_tails, tails = timed(lambda: eltq.tail_moments_arrays(*cols, edge=edges, top_node=nodes), 3)
        seconds_one, one = timed(lambda: eltq.tail_moments_arrays(*cols, edge=edges, top_node=nodes, threads=1), 1)
        bitwise = all(
            np.array_equal(getattr(tails, field), getattr(one, field))
            for field in ("rate_above", "moments_above", "excess_above", "top_half_rate", "top_half_excess")
        )
        entry = {
            "moments_seconds": round(seconds_moments, 4),
            "tails_seconds": round(seconds_tails, 3),
            "tails_seconds_one_thread": round(seconds_one, 3),
            "tails_bitwise_across_threads": bitwise,
            "valid_rows": moments.valid_rows,
            "regimes": tails.regimes,
            "max_support": moments.max_support,
        }
        record["runs"][name] = entry
        print(f"{name:12s} moments {seconds_moments:7.4f} s  tails {seconds_tails:6.3f} s "
              f"(1 thread {seconds_one:6.3f} s)  bitwise {bitwise}  {tails.regimes}")
    if args.multiunit:
        from run_multiunit import synthetic_portfolio

        cols = synthetic_portfolio(args.events, 10)
        multi = (cols["event_id"], cols["unit_id"], cols["rate"], cols["mean_loss"], cols["sd_indep"],
                 cols["sd_corr"], cols["max_loss"])
        entry = {"rows": int(cols["rate"].size)}
        for coupling in ("burke", "comonotone", "independent", "homer"):
            seconds, m = timed(lambda c=coupling: eltq.multiunit_moments(*multi, coupling=c), 1)
            entry[coupling] = {"seconds": round(seconds, 3), "evaluations": m.evaluations,
                               "achieved_error": list(m.achieved_error), "unconverged": m.unconverged_events,
                               "marginal_residual": m.marginal_residual}
            print(f"multiunit_moments {coupling:12s} {seconds:7.2f} s  evaluations {m.evaluations:.2e}  "
                  f"achieved {m.achieved_error[0]:.1e} {m.achieved_error[1]:.1e}  unconverged {m.unconverged_events}")
        seconds_one, one = timed(lambda: eltq.multiunit_moments(*multi, threads=1), 1)
        seconds_all, again = timed(lambda: eltq.multiunit_moments(*multi), 1)
        entry["burke_one_thread_seconds"] = round(seconds_one, 2)
        entry["bitwise_across_threads"] = one.raw_moments == again.raw_moments
        seconds_views, _ = timed(lambda: eltq.multiunit_views(*multi, bucket_size=40000.0, views=("ground_up",)), 1)
        entry["views_ground_up_seconds"] = round(seconds_views, 2)
        print(f"burke one thread {seconds_one:.1f} s, bitwise {entry['bitwise_across_threads']}; "
              f"multiunit_views ground_up {seconds_views:.2f} s")
        record["runs"]["multiunit"] = entry
    record["peak_rss_mb"] = peak_working_set_mb()
    print(f"peak RSS {record['peak_rss_mb']} MiB")
    if not args.no_append:
        history = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
        history.append(record)
        RESULTS_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
