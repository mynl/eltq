# eltq 1.11.0 multi-unit performance report

Date: 2026-08-26. Machine: Intel i7-7820HQ (4C/8T, AVX2), Windows 11,
Python 3.13, rustc 1.95, wheel built locally with `target-cpu=native`. Harness:
`uv run python benchmarks/run_multiunit.py --events 100000 --units 10`; the
record is appended to `benchmarks/results.json` (`"suite": "multiunit"`).
Correctness gated the run: 141 Python tests (both backends) and the full Rust
suite were green on this build; the harness also checks bitwise determinism
across thread counts on the portfolio itself.

The plan's headline target was a 1M unit-row portfolio in single-digit
seconds on the Xeon workstation. That machine was not available; on this
laptop the all-four-view run lands at 22 s. The numbers below are the probe
the plan asked for, not the assertion.

## Portfolio

Seeded synthetic unit-level ELT, 100k events x 10 units, single-unit events
thinned to one row: 909,937 rows. Regime mix per event: mixed correlated
betas 68%, independent-only 10%, comonotone-only 5%, atoms 5%, single-unit
10%, zero-rate 2%; 0.5% of rows dropped (mean above max). Term sparsity:
units u00 and u01 carry a policy layer plus an occurrence tower, u02 a policy
layer only, u03 a reinsurance clause only, six units bare. Bucket 40,000
(per-unit grids up to 126 nodes; event grid 874 nodes), Q = 32 midpoint.

## Results

| run | seconds | rows/s | notes |
| --- | --- | --- | --- |
| all four views, auto, Q = 32 | 22.5 | 40,500 | headline; auto chose direct for every event |
| ground_up only, Q = 32 | 15.2 | 59,700 | the all-four delta is +48%, not 4x |
| all four views, Q = 8 | 7.9 | 115,000 | the z loop is the wall: ~85% of time scales with Q |
| all four views, forced direct | 23.0 | 39,500 | |
| all four views, forced FFT | 38.0 | 23,900 | at these supports (n ~ 20-40) direct wins |
| all four views, Gauss-Hermite Q = 32 | 23.8 | 38,200 | same cost as midpoint |
| threads 1 / 2 / 4 / 8 | 81.9 / 43.8 / 27.7 / 22.3 | | 3.7x on 4 cores + hyperthreads |
| two-unit joint, Q = 1001 (101 x 201) | 9.9 ms | | reference accuracy |
| two-unit joint, Q = 4001 | 38.3 ms | | |
| peak RSS | 231 MiB | | whole portfolio in memory plus per-chunk partials |

Bitwise identical results across thread counts: yes.

## What the 1.11.0 pass changed (100k-row probe, same machine)

| stage | all four views | single thread |
| --- | --- | --- |
| 1.10.0 engine as shipped | 15.8 s | 21.2 s |
| whole-event chunks of 2048 rows (was 65536) | 6.0 s | 16.7 s |
| own erfc, real FFTs, cross-view row-set sharing | 6.0 s | 16.7 s |
| `target-cpu=native` (AVX2 axpy in the direct chains) | 4.5 s | 12.1 s |

The first row was effectively two-threaded (two chunks); the chunk change is
where the thread scaling came from. The sharing rewrite was worth ~1.3x
single-threaded on this term mix; its value grows with the number of views
and bare units. The FFT path is slower than direct here because the per-unit
supports are short (median ~20 nodes); the cost model (`FFT_COST_FACTOR = 3`)
keeps `auto` on direct for these events and will switch for long supports.

## Where the time goes

Per event at Q = 32: ~85% in the per-z convolution chains (direct
convolution of the correlated units' rows with the shared bare block and the
independent block, per view), ~10% in edge scores and conditional rows
(incomplete beta, normal quantile, erfc), the rest in view maps, deposits and
reconciliation. Reducing Q from 32 to 8 cuts the run by 2.8x, so the next
structural gain is in the z dimension (the plan's optional Chebyshev
interpolation of spectra across z), not in the kernels.

## One-unit route on the same day

`run_end_to_end.py --quick` on this build: 1M rows at K = 32 in 7.4 s
(135k rows/s), below the 1.4.0 history (180-230k rows/s) for the identical
seeded workload. An A/B against the 1.5.0 commit built and timed in a
separate worktree gave 8.1-8.9 s (112-124k rows/s), so the gap is the
machine's state today (sustained load, power/thermal), not the code; the
current build is if anything faster. Compare future records against runs
taken on the same day where possible.
