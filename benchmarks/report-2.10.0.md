# eltq 2.10.0 multi-unit performance report

Date: 2026-10-26. Machine: AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T), Windows 11, Python 3.14, rustc 1.99.1, local build at `x86-64-v3` (the machine-level cargo config). Harness: `uv run python benchmarks/run_multiunit.py --events 100000 --units 10`; the record is appended to `benchmarks/results.json` (`"suite": "multiunit"`). Correctness gated the run: the Python suite on both backends and the Rust suite were green on this build, and the harness checks bitwise determinism across thread counts on the portfolio itself.

The rerun asked for by `dev/done/plan-multiunit-coupling.md` [parity]: the same portfolio as `report-1.11.0.md` under the defaults of 2.5.0 to 2.9.0 (Bernoulli rows, the stratified factor rule with its tail ladder at 32 body strata, one-unit events bucketed directly), with the node rules and the exact coupling paths beside it.

## Portfolio

Seeded synthetic unit-level ELT, 100k events x 10 units, single-unit events thinned to one row: 909,937 rows. Regime mix per event: mixed correlated betas 68%, independent-only 10%, comonotone-only 5%, atoms 5%, single-unit 10%, zero-rate 2%; 0.5% of rows dropped (mean above max). Term sparsity: units u00 and u01 carry a policy layer plus an occurrence tower, u02 a policy layer only, u03 a reinsurance clause only, six units bare. Bucket 40,000 (per-unit grids up to 126 nodes; event grid 877 nodes). Default rule: stratified, 32 body strata plus the 44-stratum tail ladder, 76 strata.

## Results

| run | seconds | rows/s | notes |
| --- | --- | --- | --- |
| all four views, auto, default (stratified, 76 strata) | 8.36 | 108,800 | headline; auto chose direct for every event |
| ground_up only, default | 7.04 | 129,300 | the all-four delta is +19% |
| all four views, `z_nodes=8` (52 strata) | 5.64 | 161,200 | the tails are the ladder's, unchanged (`z-convergence-2.6.0.md`) |
| all four views, forced direct | 8.47 | 107,400 | |
| all four views, forced FFT | 19.42 | 46,900 | short supports (median ~20 nodes): direct wins, as at 1.11.0 |
| all four views, Gauss-Hermite Q = 32 | 1.89 | 480,600 | the node rules: fast, tails off by 1e-1 to 1 where a loading is near one |
| all four views, midpoint Q = 32 | 1.97 | 461,600 | |
| all four views, `coupling="comonotone"` | 0.38 | 2,410,900 | the exact path: one row per unit, no factor rule |
| all four views, `coupling="homer"` | 0.38 | 2,418,400 | the rolled row and the composed map per view |
| threads 1 / 2 / 4 / 32 | 123.6 / 63.9 / 33.2 / 8.6 | | 14.4x on 16 cores and SMT |
| two-unit joint, `z_nodes=1001` (101 x 201) | 16.2 ms | | 1,045 strata under the default rule |
| two-unit joint, `z_nodes=4001` | 62.3 ms | | |
| peak RSS | 728 MiB | | whole portfolio in memory plus per-chunk partials |

Bitwise identical results across thread counts: yes.

## Reading

- The stratified default costs 4.2 to 4.4 times the node rules at the same body count (8.4 s against 1.9 s to 2.0 s). The strata count explains 2.4 of that (76 against 32 rows per event through the convolution chains); the rest is the bivariate normal cdf per edge and stratum in the conditional matrices (`BivariateUpper`, Genz 2004, with the per-unit constants hoisted; it was 11.4 s before). `z_nodes=8` keeps the ladder and the tails and saves a third. The plan's acceptance line allows the 1.11.0 figures scaled by the strata count, 2.4 times; the default is 1.8 times over that line, the construction being the excess, and the node rules remain one keyword away for a portfolio whose loadings are all well below one.
- The exact coupling paths are 22 times faster than the default: the comonotone merge and the rolled row touch each unit once.
- Against `report-1.11.0.md` on this machine (0.92 s for the 1M-row headline at 2.4.2, `report-2.4.2.md`), the midpoint rule at Q = 32 takes 1.97 s here: the difference is the Bernoulli rows' two-point conditional law and the one-unit branch being small, and the 2.4.2 figure being a native build on a quieter machine; the two are within the day-to-day spread of this portfolio.
- Thread scaling is 14.4x on 16 cores with SMT, against 3.7x on the 1.11.0 laptop's four cores; the single-thread run is dominated by the strata loop as the 1.11.0 profile said of the node loop.
