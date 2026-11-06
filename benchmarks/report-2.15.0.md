# eltq 2.15.0 one-row density performance report

Date: 2026-11-06. Machine: AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T), Windows 11, Python 3.14, local build at `x86-64-v3` (the machine-level cargo config). Harness: `uv run python benchmarks/run_moments.py --density`; the record is appended to `benchmarks/results.json` (`"suite": "moments"`, `runs.density`). Correctness gated the run: the Python suite on both backends (487 passed, 3 skipped) and the Rust suite were green on this build.

The line of `dev/done/plan-density-arrays.md`: at 30 points, `density_arrays` no slower than `tail_moments_arrays` at the same 30 edges.

## Portfolios

The three seeded one-row portfolios of `run_end_to_end.py`, a million rows each: the mixture (85% betas, 9% atoms, 4% two-point rows), small alpha (alpha from 0.003 to 3, M / mu to 1e5) and large exposed values (M to 1e9, 98% low-mean rows); and the 909,937 unit rows of `run_multiunit.py`'s portfolio read as one one-row table. Both functions are asked at the same 30 points, log-spaced from a thousandth of the largest exposed value to the value itself.

## Results

| portfolio | rows | `density_arrays`, 32 threads | one thread | `tail_moments_arrays`, 32 threads | bitwise across threads |
| --- | --- | --- | --- | --- | --- |
| mixture | 1,000,000 | 0.031 s | 0.30 s | 0.092 s | yes |
| small alpha | 1,000,000 | 0.040 s | 0.47 s | 0.148 s | yes |
| large exposed values | 1,000,000 | 0.034 s | 0.37 s | 0.112 s | yes |
| multi-unit rows | 909,937 | 0.034 s | 0.41 s | 0.107 s | yes |

At every node of an even grid of $2^{16}$ points to the largest exposed value, on the mixture portfolio: 0.86 s on 100,000 rows and 8.0 s on a million, peak RSS 72 MiB and 168 MiB (measured separately, not part of the harness).

## Reading

- The density costs one exponential and two logarithms per beta row and point inside its support, against a continued fraction per row and edge for the tail query: a third to a quarter of the tail query's time at the same points, inside the plan's line on every portfolio. On 16 cores with SMT it runs 10 to 12 times faster than on one thread.
- A row is visited only at the points inside $(0, M)$, found by bisection, so the cost is the rows times the points each row covers, not the rows times all the points. On the grid of $2^{16}$ points the time grows in proportion to the rows, 0.86 s to 8.0 s for ten times as many.
- The pass folds its chunks in waves of 64, so the partials in flight are bounded at about 32 MiB at $2^{16}$ points whatever the row count; the fold is in chunk order, so the result is the same for any thread count and any wave.
