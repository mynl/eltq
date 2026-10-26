# Scheduling of the chunked passes, 2.4.2, the tidying of 2.4.3, and the equal-accuracy suite

Date: 2026-10-26. Machine: the DOOB workstation (AMD Ryzen 9 9950X3D, 16 cores, 32 threads, Windows 11), the same as `benchmarks/report-2.4.0.md`; `uv run python benchmarks/run_schedule.py --markdown` on the mixture portfolio of `benchmarks/run_end_to_end.py` (its fixed seed), $K=32$, bucket size $5\times10^6/(n-1.5)$ for $n$ buckets, each configuration in its own process so that the peak working set is its own. Times are medians of three after a warm-up, one run where a run exceeds 15 s; memory is the process's peak working set in MiB, which includes the input columns (about 40 MiB at a million rows) and the two output grids.

## The chunk rule

The peer review (finding P1) measured the quantiles method at 262,144 rows with its fixed chunk of 65,536 rows against 4,096-row chunks and found 0.65 s against 0.14 s on 32 threads: four chunks for 32 workers. From 2.4.2 the default chunk is a rule of the input and the grid, $\min(\max(\lceil N/64\rceil,\lceil n/64\rceil,4096),65536)$ rows for $N$ rows and $n$ buckets, and a pass runs its chunks in waves of $\min(2T,\lfloor B/b\rfloor)$ for $T$ threads, the budget $B$ of 1 GiB and $b$ the partial's size (two grid-length vectors here). The grid term is what the review asked for and the plan's draft left out: with the row term alone, 262,144 rows at $2^{24}$ buckets took 6.4 s against 3.0 s single-threaded, 64 chunks each allocating, zeroing and folding two 128 MiB vectors, and a million rows at $2^{22}$ lost 23% on 32 threads to the serial fold of 61 partials. The fold of a wave's vectors is now parallel over disjoint blocks of the grid, each element in chunk order, so it is bitwise the serial fold's; with that and the grid divisor at 64 nothing regresses.

The fixed chunk of versions before 2.4.2 against the rule, quantiles method:

262,144 rows (the rule gives 4,096-row chunks to $2^{18}$ buckets, 16,384 at $2^{20}$, 65,536 from $2^{22}$):

| grid | threads | chunk 65,536, s | default rule, s | chunk 65,536, MiB | default rule, MiB |
|:---|---:|---:|---:|---:|---:|
| $2^{16}$ | 1 | 2.550 | 2.552 | 63 | 63 |
| $2^{16}$ | 8 | 0.654 | 0.341 | 63 | 72 |
| $2^{16}$ | 16 | 0.657 | 0.214 | 63 | 89 |
| $2^{16}$ | 32 | 0.654 | 0.133 | 63 | 123 |
| $2^{20}$ | 1 | 2.588 | 2.651 | 103 | 103 |
| $2^{20}$ | 8 | 0.673 | 0.368 | 135 | 328 |
| $2^{20}$ | 16 | 0.677 | 0.238 | 136 | 328 |
| $2^{20}$ | 32 | 0.674 | 0.233 | 138 | 329 |
| $2^{22}$ | 1 | 2.671 | 2.660 | 246 | 247 |
| $2^{22}$ | 8 | 0.710 | 0.710 | 375 | 375 |
| $2^{22}$ | 16 | 0.716 | 0.715 | 376 | 376 |
| $2^{22}$ | 32 | 0.718 | 0.718 | 377 | 378 |
| $2^{24}$ | 1 | 2.944 | 2.948 | 823 | 823 |
| $2^{24}$ | 8 | 1.545 | 1.519 | 1,079 | 1,080 |
| $2^{24}$ | 16 | 1.541 | 1.550 | 1,080 | 1,080 |
| $2^{24}$ | 32 | 1.540 | 1.531 | 1,082 | 1,081 |

1,000,000 rows (the rule gives 15,625-row chunks to $2^{18}$ buckets, 16,384 at $2^{20}$, 65,536 from $2^{22}$):

| grid | threads | chunk 65,536, s | default rule, s | chunk 65,536, MiB | default rule, MiB |
|:---|---:|---:|---:|---:|---:|
| $2^{16}$ | 1 | 9.737 | 9.777 | 137 | 137 |
| $2^{16}$ | 8 | 1.350 | 1.291 | 136 | 136 |
| $2^{16}$ | 16 | 0.758 | 0.758 | 137 | 136 |
| $2^{16}$ | 32 | 0.763 | 0.494 | 138 | 163 |
| $2^{20}$ | 1 | 9.926 | 10.114 | 143 | 143 |
| $2^{20}$ | 8 | 1.376 | 1.447 | 368 | 368 |
| $2^{20}$ | 16 | 0.836 | 0.872 | 368 | 625 |
| $2^{20}$ | 32 | 0.816 | 0.578 | 370 | 1,106 |
| $2^{22}$ | 1 | 10.196 | 10.186 | 287 | 287 |
| $2^{22}$ | 8 | 1.693 | 1.792 | 1,120 | 1,120 |
| $2^{22}$ | 16 | 1.066 | 1.093 | 1,120 | 1,120 |
| $2^{22}$ | 32 | 1.065 | 1.099 | 1,122 | 1,122 |
| $2^{24}$ | 1 | 11.122 | 11.081 | 863 | 863 |
| $2^{24}$ | 8 | 4.238 | 4.086 | 1,120 | 1,120 |
| $2^{24}$ | 16 | 4.106 | 4.135 | 1,120 | 1,121 |
| $2^{24}$ | 32 | 4.080 | 4.058 | 1,122 | 1,122 |

The gains are where the fixed chunk starved the threads: 262,144 rows on 32 threads at $2^{16}$ run in a fifth of the time (the review's measurement), and at $2^{20}$ in a third; a million rows on 32 threads gain a third at $2^{16}$ and $2^{20}$. The rows at 8 and 16 threads with a million rows and $2^{20}$ to $2^{22}$ buckets are within noise of each other (the kernel's 16 chunks of 65,536 rows already fed 16 workers). The memory column shows the budget at work: at $2^{20}$ a wave of 64 partials of 16 MiB is 1 GiB, where the fixed chunk's four partials were 64 MiB; at $2^{24}$ the wave is four partials of 256 MiB and the peak is the same under either chunk. A caller who wants the old memory profile at $2^{20}$ sets `chunk_rows=65536`, or lowers `Config::wave_memory_bytes` from Rust.

The two constants behind the grid term, measured single-threaded on this machine: about 3.4 ns per grid element per chunk for the partial's allocation, zeroing and fold, against about 9.7 microseconds per row for the kernel at $K=32$.

## The survival method

The survival method's chunk is 256 rows and unchanged; its partials are node-length, so neither the rule nor the budget touches it. For the record, the same sweep's survival column (the trimmed run of the same day, all four thread counts; `not run` where one run takes minutes):

| rows | grid | 1 thread | 8 | 16 | 32 |
|:---|:---|---:|---:|---:|---:|
| 262,144 | $2^{16}$ | 12.5 s | 2.14 | 1.22 | 0.84 |
| 262,144 | $2^{20}$ | 48.9 | 10.8 | 6.59 | 4.58 |
| 262,144 | $2^{24}$ | not run | not run | 137 | 137 |
| 1,000,000 | $2^{16}$ | 48.0 | 9.24 | 4.95 | 3.13 |
| 1,000,000 | $2^{20}$ | 187 | 38.4 | 22.4 | 15.3 |
| 1,000,000 | $2^{24}$ | not run | not run | 469 | 350 |

These are not comparable with the review's 4.2 s single-threaded on 262,144 rows, which was at a bucket size of 5,000, about a thousand buckets: the survival method's cost grows with the bucket count, because every event's tail is walked across its bucket edges until the rate floor (`dev/done/plan-survival-tail-cost.md`), and at $2^{16}$ buckets of this portfolio the tails reach most of the grid. Of the 12.5 s at $2^{16}$, 8.6 s remain with the fine mesh off (`nodes_per_sigma=1`). The peak working set is 73 to 435 MiB across the table: the survival partials are small.

## Tidying, 2.4.3

Four pieces of redundant work removed (peer review, finding P2), each bitwise neutral on the outputs: the survival grid's edges walked with an interval cursor and differenced as they come, the tail interpolant's logarithm taken once, the midpoint rule's normal scores shared across the events on the Cornish-Fisher route, and the mass and moment of each output grid in one pass. Medians of five at 32 threads unless stated, 2.4.2 against 2.4.3, the secondary grid's checksum, mass and mean identical in every row:

| portfolio | 2.4.2 | 2.4.3 |
|:---|---:|---:|
| mixture, $10^6$ rows, survival, $h=5{,}000$ | 1.797 s | 1.763 s |
| mixture, $10^6$ rows, quantiles $K=32$, $h=5{,}000$ | 0.486 s | 0.469 s |
| small-alpha, $10^6$ rows, survival, $h=100$ | 2.529 s | 2.495 s |
| large exposure, $10^5$ rows, survival, $h=25$, $2^{22}$ buckets | 1.200 s | 1.153 s |
| large parameters, $10^6$ rows, quantiles $K=32$, one thread | 0.434 s | 0.110 s |

The large-parameter portfolio has every event on the Cornish-Fisher route (exposed value $10^6$, mean about $2.5\times10^5$, coefficient of variation $10^{-3.5}$ to $10^{-3}$), where the normal quantile was most of the per-event work; on 32 threads it runs in milliseconds either way.

## The equal-accuracy suite

`uv run python benchmarks/run_accuracy.py --markdown` at 2.4.3: the error each method's time buys, against SciPy's regularized incomplete beta (the normal law from $\min(\alpha,\beta)=10^{16}$) summed exactly over the events, as the largest relative error by band of the exceedance rate as a share of the total rate. The bell-shaped sweep by `nodes_per_sigma` is unchanged from `benchmarks/report-2.4.0.md`.

Endpoint tails: rate-one events with $\alpha\in\{10^2,10^4,10^6\}$ and $\beta\in\{0.5,1,2,5,20\}$, the curve at distances $M-x$ from $10^{-8}M$ to $0.1M$, by band of the event's survival; the survival method on both backends (the quantiles method has no curve). The two backends agree within $1.4\times10^{-8}$ relative where the survival is at least $10^{-12}$.

| backend | $[10^{-2},1)$ | $[10^{-4},10^{-2})$ | $[10^{-6},10^{-4})$ | $[10^{-8},10^{-6})$ | $[10^{-12},10^{-8})$ |
|:---|---:|---:|---:|---:|---:|
| survival, engine | 2.2e-06 | 9.6e-06 | 1.1e-08 | 4.1e-09 | 1.4e-08 |
| survival, reference | 2.2e-06 | 9.6e-06 | 1.5e-10 | 7.3e-11 | 1.0e-10 |

Large parameters: one-row portfolios with exposed value 1, on a grid of a quarter standard deviation (at least $10^{-6}$) covering the mean plus and minus 12; the largest bucket-mass error as a share of the event's rate. The quantiles method's figure is the midpoint rule's: 32 atoms against about a hundred buckets across the bulk.

| $\min(\alpha,\beta)$ | $\beta/\alpha$ | survival, engine | survival, reference | quantiles $K=32$, engine |
|:---|---:|---:|---:|---:|
| $2\times10^5$ | 1 | 3.5e-14 | 2.0e-22 | 1.6e-02 |
| $2\times10^5$ | 20 | 2.3e-14 | 2.4e-14 | 1.2e-02 |
| $10^6$ | 1 | 8.3e-14 | 3.3e-14 | 1.5e-02 |
| $10^6$ | 20 | 8.9e-14 | 2.4e-22 | 1.7e-02 |
| $10^9$ | 1 | 1.4e-12 | 9.0e-22 | 1.9e-02 |
| $10^9$ | 20 | 1.1e-12 | 3.7e-26 | 8.1e-03 |
| $10^{16}$ | 1 | 0 | 0 | 2.8e-03 |
| $10^{16}$ | 20 | 0 | 0 | 4.8e-02 |
| $10^{20}$ | 1 | 0 | 0 | 2.8e-05 |
| $10^{20}$ | 20 | 0 | 0 | 4.8e-02 |

Mixtures: the bucketed tail sums of the beta events against their exact exceedance at every bucket edge (the atoms, which the linear rule splits across two nodes, set aside), the engine under both methods and the reference under the survival method, with the engine's time on a million rows of the same recipe on 32 threads. A speed comparison between the methods is read off this table at a stated accuracy: on the small-alpha portfolio the quantiles method at $K=32$ runs in a quarter of the time and is wrong by its whole value from the hundred-year band on, where the survival method is within $5\times10^{-6}$.

| portfolio | method, backend | $[10^{-2},1)$ | $[10^{-4},10^{-2})$ | $[10^{-6},10^{-4})$ | $[10^{-8},10^{-6})$ | $[10^{-12},10^{-8})$ | time, s |
|:---|:---|---:|---:|---:|---:|---:|---:|
| medium fixture, $h=5{,}000$ | survival, engine | 9.4e-07 | 4.6e-06 | 1.8e-07 | 2.0e-05 | n/a | |
| medium fixture, $h=5{,}000$ | quantiles $K=32$, engine | 1.5e-02 | 1.6e-01 | 8.7e-01 | 1.7e-01 | n/a | |
| medium fixture, $h=5{,}000$ | survival, reference | 9.4e-07 | 4.6e-06 | 1.8e-07 | 2.0e-05 | n/a | agrees within 2e-15 |
| small-alpha fixture, $h=100$ | survival, engine | 1.2e-08 | 3.6e-08 | 4.0e-06 | 3.7e-06 | 6.3e-07 | |
| small-alpha fixture, $h=100$ | quantiles $K=32$, engine | 9.8e-02 | 9.2e-01 | 1.0 | 1.0 | 1.0 | |
| small-alpha fixture, $h=100$ | survival, reference | 1.2e-08 | 3.6e-08 | 4.0e-06 | 3.7e-06 | 6.3e-07 | agrees within 9e-16 |
| mixture portfolio, 20,000-row sample, $h=5{,}000$ | survival, engine | 1.2e-06 | 2.4e-06 | 7.5e-06 | n/a | n/a | 1.83 |
| mixture portfolio, 20,000-row sample, $h=5{,}000$ | quantiles $K=32$, engine | 8.9e-03 | 1.4e-01 | 1.0 | n/a | n/a | 0.45 |
| mixture portfolio, 20,000-row sample, $h=5{,}000$ | survival, reference | 1.2e-06 | 2.4e-06 | 7.5e-06 | n/a | n/a | agrees within 3e-15 |
| small-alpha portfolio, 5,000-row sample, $h=100$ | survival, engine | 1.6e-08 | 4.4e-08 | 4.6e-06 | 3.5e-06 | 8.3e-07 | 2.47 |
| small-alpha portfolio, 5,000-row sample, $h=100$ | quantiles $K=32$, engine | 9.5e-02 | 1.0 | 1.0 | 1.0 | 1.0 | 0.60 |
| small-alpha portfolio, 5,000-row sample, $h=100$ | survival, reference | 1.6e-08 | 4.4e-08 | 4.6e-06 | 3.5e-06 | 8.3e-07 | agrees within 2e-15 |

The agreement column is the largest difference between the two backends' survival grids as a share of the total rate.

## Build profile and target, 2026-10-26

Release-profile variants built with environment overrides and timed on the mixture portfolio, medians of five at 32 threads and three single-threaded, the default build timed before and after to bound the noise:

| build | survival $10^6$, 32 threads | quantiles $K=32$, $10^6$, 32 threads | survival 262,144, 1 thread | quantiles 262,144, 1 thread |
|:---|---:|---:|---:|---:|
| thin LTO, `target-cpu=native` | 1.796 s | 0.497 s | 7.837 s | 2.534 s |
| fat LTO, native | 2.133 s | 0.527 s | 9.598 s | 2.882 s |
| fat LTO, `panic=abort`, native | 2.104 s | 0.522 s | 9.598 s | 2.879 s |
| thin LTO, `target-cpu=x86-64-v3` | 1.677 s | 0.435 s | 7.424 s | 2.443 s |
| thin LTO, native, again | 1.813 s | 0.456 s | 7.818 s | 2.539 s |

Fat LTO is 12% to 22% slower than thin on every row; `panic=abort` changes nothing. The generic `x86-64-v3` build, the level of the published wheels, is 4% to 6% faster than native on this Zen 5: native enables AVX-512, whose longer instruction encodings and wider auto-vectorization buy a scalar continued fraction nothing and cost it code size. The multi-unit route goes the other way, native 3% to 9% faster (`run_multiunit.py --quick`: all four views at $Q=32$ in 1.98 s against 2.04 s, single-threaded 24.4 s against 26.7 s), where the convolutions vectorize. The machine-level cargo config now targets `x86-64-v3` for every local build; it also restores the test profile under rustc 1.99.1, which failed inside `libm` only under native. The profile stays thin LTO with one codegen unit.
