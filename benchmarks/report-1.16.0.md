# eltq 1.16.0 survival method report: node density and the speed gate

Date: 2026-10-24. Machine: AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T), Windows 11,
Python 3.14.7, rustc 1.99.1, extension built locally with `target-cpu=native`.
Harness: `uv run python benchmarks/run_end_to_end.py --survival-sweep` and the
same with `--small-alpha`; both records are appended to
`benchmarks/results.json` (`"suite": "survival_sweep"`). Correctness gated the
runs: the full Python suite on both backends and the Rust suite were green on
this build, and the harness checks bitwise determinism across thread counts on
each portfolio.

The plan (`dev/done/plan-single-unit-survival-route.md`, [benchmark]) asked for
two things: the smallest node density whose interpolation error is at most
1e-4 on every benchmark case, and a gate, stop and talk if the survival method
at that density takes more than five times the quantiles method at K = 32 on
the million-row case.

## Portfolios

- **mixture**: the standard 1M-row seeded mixture of the end-to-end benchmark
  (70% well-conditioned betas, 10% skewed with alpha below one, 5%
  concentrated, 6% atoms, 3% boundary means, 4% inadmissible, 2% zero rate;
  mean damage ratios up to 0.95, so half the betas are J-shaped with their
  mass against the endpoint). Bucket 5,000, 1,001 nodes.
- **small_alpha**: 1M rows by the recipe of `fixtures/small_alpha_elt.csv`
  (alpha loguniform on [0.003, 3], M / mu on [1e2, 1e4] with a tenth of the
  rows at 1e5 on M = 1e6), the shapes the method exists for. Bucket 100,
  10,001 nodes.

## Speed

Best of two runs, all threads, `chunk_rows` 65,536 for quantiles and 4,096 for
survival.

| portfolio | method | density | seconds | ratio to K = 32 | nodes | mean error | interpolation error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixture | quantiles K = 32 | | 0.784 | 1.00 | | -2.6e-4 | |
| mixture | survival | 25 | 0.450 | 0.57 | 84 | -7.6e-5 | 1.1e-4 |
| mixture | survival | 50 | 0.635 | 0.81 | 167 | -6.7e-5 | 1.2e-5 |
| mixture | survival | 100 | 0.962 | 1.23 | 332 | -6.7e-5 | 2.9e-6 |
| mixture | survival | 150 | 1.321 | 1.68 | 497 | -6.7e-5 | 1.5e-6 |
| small_alpha | quantiles K = 32 | | 1.006 | 1.00 | | -2.0e-1 | |
| small_alpha | survival | 25 | 1.470 | 1.46 | 109 | -6.0e-3 | 7.9e-6 |
| small_alpha | survival | 50 | 1.756 | 1.75 | 217 | -6.0e-3 | 4.9e-7 |
| small_alpha | survival | 100 | 2.664 | 2.65 | 432 | -6.0e-3 | 1.3e-6 |
| small_alpha | survival | 150 | 3.638 | 3.62 | 647 | -6.0e-3 | 1.0e-6 |

The mean error is `secondary_mean_error / analytic_aal`; the interpolation
error is the 64-point diagnostic of the Rust engine. The quantiles method loses
20% of the expected loss on the small-alpha portfolio (the 1/(2K) cut of spec
section 5); the survival method's 0.6% there is the first-order bucket-0 error
of the nearest rule at h = 100 against means down to 10, and halves with h.

**Gate: met.** At the chosen density the survival method costs 1.23 times
K = 32 on the mixture portfolio and 2.65 times on the small-alpha one, against
the plan's limit of five. The plan's own estimate was 5 to 14 times; the
difference is the one-sided truncated curve, which needs no node above an
event's truncation point, and the batched survival evaluation sharing its
logarithms with the slope.

## Accuracy and the density

The 64-point diagnostic is a sample. The exact figure is the reference's
bucketed tail sums against the exact incomplete beta at every bucket edge where
the exceedance rate exceeds 1e-12 of the total rate, the criterion of spec test
15, on the three portfolios the method was tested against (the fixture at
bucket 500, the medium golden and a 20,000-row subsample of the mixture
portfolio at bucket 5,000):

| density | small-alpha fixture | medium golden | mixture subsample |
| --- | --- | --- | --- |
| 25 | 2.8e-4 | 7.8e-3 | 7.2e-3 |
| 50 | 6.7e-5 | 2.8e-4 | 4.7e-4 |
| 100 | 4.0e-6 | 4.8e-5 | 5.9e-5 |
| 150 | 1.2e-5 | 1.1e-5 | 1.1e-5 |

50 nodes per decade fails the 1e-4 criterion on the J-shaped portfolios; 100 is
the smallest density that meets it everywhere, and it becomes the default
(`DEFAULT_NODES_PER_DECADE` in `python/eltq/_meta.py` and
`crates/eltq-core/src/config.rs`). The per-event endpoint interpolant is
independent of the density; its own accuracy is a few units in 1e-6.

## Not measured

The plan asked for the time split into curve build and bucketing. The Rust
engine deposits each event's tail in the same chunk pass that evaluates its
nodes, so the split is not observable from outside; the node evaluation
dominates (the mixture run at 150 nodes per decade spends about 0.5 s more than
at 100 for 165 more nodes per event).
