# Scheduling of the chunked passes, 2.4.2, and the tidying of 2.4.3

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
