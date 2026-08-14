# Performance report — eltq 1.4.0, one-million-row headline

Date: 2026-08-14. Machine-readable history: `benchmarks/results.json`.

## Environment

| | |
| --- | --- |
| CPU | Intel Core i7-7820HQ @ 2.90 GHz (4 physical / 8 logical cores) |
| OS | Windows 11 Pro 10.0.26200 |
| Toolchain | rustc 1.95.0, maturin release profile (thin LTO, 1 codegen unit) |
| Python | CPython 3.13.13, NumPy 2.5.2, abi3-py313 wheel |
| Input | seeded synthetic 1M-row mixture (spec §10): 70% well-conditioned, 10% skewed α<1, 5% concentrated large-parameter, 6% σ=0, 3% boundary means, 4% inadmissible (clamped), 2% zero-rate; losses to 5e6 |

Correctness gates performance: this build passes the full acceptance suite
(80 Python tests against the SciPy reference incl. golden fixtures, 48 Rust
unit/property/golden tests) and the bitwise thread-determinism checks.

## Headline (1M rows, K = 32, h = 5000, n = 1001)

| Metric | Value |
| --- | --- |
| End-to-end in-memory | **4.35 s** |
| Rows/s | **230,000** |
| Beta quantiles/s | **6.55M** (938k beta rows × 32) |
| Speedup vs SciPy reference | **~44×** (SciPy: 5,293 rows/s measured on 20k subsample) |
| Non-kernel overhead (σ=0 pass: validation + deposits + fold) | 0.023 s (~0.5%) |
| CSV route (parse + compute, 89 MB file) | 4.84 s |
| Memory-mapped `.npy` output | 4.24 s |
| Peak working set (harness process incl. 1M-row inputs) | 148 MB |
| Bitwise deterministic across thread counts | yes |

The inverse-beta kernel is effectively the entire cost, as the workload
analysis predicted; parsing, validation, deposits and the deterministic
reduction together are under one percent.

## Grid and K sweeps (in-memory, all cores)

| K | h | n | Time | Rows/s | Quantiles/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5000 | 1,001 | 4.35 s | 230k | 6.55M |
| 32 | 50 | 100,000 | 4.40 s | 227k | 6.47M |
| 8 | 5000 | 1,001 | 1.60 s | 626k | 4.45M |
| 128 | 5000 | 1,001 | 12.4 s | 81k | 9.17M |

Grid length is immaterial (deposits are O(1) per point); throughput in
quantiles/s grows with K as per-event setup amortizes.

## Thread scaling (K = 32, h = 5000)

| Threads | Time | Rows/s | Scaling |
| ---: | ---: | ---: | ---: |
| 1 | 25.3 s | 40k | 1.0× |
| 2 | 13.7 s | 73k | 1.8× |
| 4 | 7.32 s | 137k | 3.5× |
| 8 | 4.26 s | 235k | 5.9× |
| all (8) | 4.24 s | 236k | 6.0× |

Near-linear to the 4 physical cores; SMT adds a further ~1.6× because the
kernel is branch/latency-bound rather than throughput-bound. Results are
bitwise identical at every row of this table (deterministic in-order chunk
fold; chunks of 65,536 rows).

## Kernel microbenchmarks (criterion, `cargo bench -p eltq-core`)

Batched kernel (warm-start ODE predictor + safeguarded Halley + early exit)
versus the scalar statrs-based reference kernel, K = 32 per event (criterion
medians):

| Regime (α, β) | Batched | Reference | Speedup | Batched ns/quantile |
| --- | ---: | ---: | ---: | ---: |
| well-conditioned (2.5, 7.5) | 24.5 µs | 106.6 µs | 4.4× | 765 |
| skewed small α (0.5, 9.5) | 18.9 µs | 135.6 µs | 7.2× | 590 |
| clamped U-shape (6e-4, 1.4e-3) | 68.0 µs | 811.3 µs | 11.9× | 2,124 |
| concentrated (500, 500) | 39.6 µs | 404.5 µs | 10.2× | 1,237 |
| asymmetric large (300, 100) | 53.8 µs | 227.2 µs | 4.2× | 1,680 |

Support kernels: own incomplete-beta CDF 187–273 ns/eval across the same
regimes; linear deposit ~3.9 ns/point; beta parameterization ~2.9 ns/row —
consistent with the ~2–4 CDF evaluations per quantile the warm-start design
targets.

## Phase attribution method

The σ=0 pass measures everything except the inverse-beta kernel (identical
validation, classification, raw and secondary deposits, reduction, and
Python boundary). CSV parse cost is the difference between the CSV route and
the in-memory route on identical data. Kernel internals are covered by the
criterion suite above.

## Observations and deferred work (1.5.0+ candidates, profiling-gated)

- Clamped U-shapes (α, β ≪ 1) cost ~2–3× a well-conditioned event: the
  quantile cliff defeats the ODE predictor and falls back to the robust
  bracketed solve. A validated closed-form start for the two-atom regime
  would close most of the gap; at 4% of rows it is not the headline lever.
- The CSV route costs ~0.5 s over in-memory; parse/compute overlap would
  hide most of that but is not material at these throughputs.
- SIMD remains unattractive: convergence paths diverge per event (branchy);
  scalar setup reuse and threads already dominate.
- Memory: peak partial-histogram memory is bounded by waves of 2× workers;
  at n = 100,000 the full run fits comfortably (~tens of MB of partials).
