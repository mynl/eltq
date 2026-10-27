# eltq 2.14.0 shared-event tail query performance report

Date: 2026-10-27. Machine: AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T), Windows 11, Python 3.14, release wheel of this tree at `x86-64-v3`. Harness: `python benchmarks/run_moments.py --shared-tails`; the record is appended to `benchmarks/results.json` (`"suite": "moments"`, `runs.shared_tails`). The run used a scratch environment holding the lock file's versions (NumPy 2.5.2, SciPy 1.18.0, pandas 3.0.5) and a wheel of this tree, because the project environment's extension was held open by a stray process. Correctness gated the run: the Python suite on both backends and the Rust suite were green on this build.

The plan (`dev/done/plan-model-moments-and-windowed-views.md`) asks for the 0.91M portfolio at a handful of edges and sets no budget; ruling 10 asks whether the union bound's looseness is material.

## Portfolio and edges

The seeded portfolio of `run_multiunit.py`, as in `report-2.13.0.md`: 100k events of up to ten units, 909,937 rows. Four edges, log-spaced from the 99th percentile of the event mean totals to half the largest event support.

## Results

| run | seconds | integrand evaluations |
| --- | --- | --- |
| `multiunit_tail_moments`, four edges, all threads | 20.4 | 1.0e10 |
| the same, eight threads | 41.9 | |
| `multiunit_views`, ground_up only, for comparison (`report-2.13.0.md`) | 6.95 | |

Bitwise identical across thread counts: yes. Peak RSS 476 MiB (the whole harness). Events: 19,750 exact (9,945 of them one-unit), 78,176 bounded. An exploratory run at eight edges took 57 s (2.5e10 evaluations), and one at a single unreachable edge 0.05 s.

| edge | lower rate | upper rate | trivial bound, the rate of events reaching the edge | upper / trivial | upper / lower |
| --- | --- | --- | --- | --- | --- |
| 12,687,518 | 0.431 | 36.7 | 43.7 | 0.84 | 85 |
| 14,107,243 | 0.225 | 25.7 | 31.3 | 0.82 | 114 |
| 15,685,834 | 0.0993 | 15.9 | 19.9 | 0.80 | 160 |
| 17,441,069 | 0.0368 | 8.76 | 11.2 | 0.78 | 238 |

## Reading

- The union bound is not informative on events of many units: its upper tail rate is only 16 to 22% below the trivial bound, and 85 to 240 times its lower. A bound this wide says nothing about whether a window is adequate. This is the measurement ruling 10 asked for; the plan's next evaluator is the lattice bracket, which is Steve's decision.
- Exact events cost nothing beyond the classification pass; the cost is the bounded events. It grows with the edges, because each threshold of a steep unit (Burke's loading wherever $\sigma_I\ll\sigma_C$) needs breakpoints in the factor where its tail switches on. At four edges the query takes 2.9 times a ground-up `multiunit_views` run. No budget was set, and the speed was not tuned given the bound's looseness.
- The one-row parts of the harness are unchanged from 2.12.0 (0.17 to 0.30 s for a million rows by 64 edges).
