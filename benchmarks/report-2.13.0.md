# eltq 2.13.0 shared-event model moments performance report

Date: 2026-10-27. Machine: AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T), Windows 11, Python 3.14, local build at `x86-64-v3`. Harness: `uv run python benchmarks/run_moments.py --multiunit`; the record is appended to `benchmarks/results.json` (`"suite": "moments"`). Correctness gated the run: the Python suite on both backends and the Rust suite were green on this build.

The budget of `dev/done/plan-model-moments-and-windowed-views.md` (open question 4): `multiunit_moments` within one default `multiunit_views` run of the 0.91M unit-row portfolio, 8.4 s (`report-2.10.0.md`).

## Portfolio

The seeded portfolio of `run_multiunit.py`: 100k events of ten units, 909,937 rows after single-unit thinning; 68% of events mixed correlated betas with loadings spread over (0, 1), 10% independent, 5% comonotone (loading one), 5% atoms, 10% single-unit, 2% zero-rate; shapes alpha and beta from 0.03 to 18.

## Results

| run | seconds | integrand evaluations | achieved error in C_2, C_3 |
| --- | --- | --- | --- |
| `coupling="burke"` (default) | 9.49 | 9.5e9 | 2.5e-8, 4.4e-8 |
| `coupling="comonotone"` | 3.73 | 1.8e9 | 2.4e-8, 4.9e-8 |
| `coupling="independent"` | 0.05 | 0 | closed form |
| `coupling="homer"` | 0.05 | 0 | closed form |
| `burke`, one thread | 150.1 | | 15.8x on 16 cores and SMT |
| `multiunit_views`, ground_up only, for comparison | 6.95 | | |

Bitwise identical across thread counts: yes. Peak RSS 303 MiB (the whole portfolio and the one-row runs of the same harness). No event unconverged at the default `rtol` of 1e-6.

## Reading

- The default coupling is 13% over the plan's line (9.5 s against 8.4 s) and 1.4 times a ground-up `multiunit_views` run on the same build. The accuracy is 40 times inside `rtol` (the two levels differ by 2.5e-8 and 4.4e-8 relative to `C_2` and `C_3`; against independent oracles the error is about 1e-8, `tests/test_moments.py`).
- The first implementation took 91 s: units with a loading above about 0.9 are steep in the factor, and every unit of their event was evaluated at the dense factor nodes they need. Each beta unit is now evaluated at its own nodes (its kernel's width in the factor when smooth, its table's breakpoints when steep) and interpolated onto the event's in the logarithm, which took the cost to 9.5 s with the accuracy unchanged (the tuning is in the plan's execution log).
- The closed-form couplings cost the classification pass alone.
