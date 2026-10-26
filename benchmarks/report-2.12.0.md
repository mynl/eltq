# eltq 2.12.0 model moments and tail measures performance report

Date: 2026-10-26. Machine: AMD Ryzen 9 9950X3D2 (Zen 5, 16C/32T), Windows 11, Python 3.14, local build at `x86-64-v3` (the machine-level cargo config). Harness: `uv run python benchmarks/run_moments.py --rows 1000000 --edges 64`; the record is appended to `benchmarks/results.json` (`"suite": "moments"`). Correctness gated the run: the Python suite on both backends and the Rust suite were green on this build.

The budget of `dev/done/plan-model-moments-and-windowed-views.md` (open question 4, Steve's answer: "full quantization on 250K is mostly < 1 sec"): one-row moments and tails on a million rows by 64 edges well under a second.

## Portfolios

The three seeded one-row portfolios of `run_end_to_end.py`, a million rows each: the mixture (85% betas, 9% atoms, 4% two-point rows), small alpha (alpha from 0.003 to 3, M / mu to 1e5) and large exposed values (M to 1e9, 98% low-mean rows). The tail query asks at 64 edges log-spaced from a thousandth of the largest exposed value to the value itself, each with its node 1e-4 below.

## Results

| portfolio | `moments_arrays` | `tail_moments_arrays`, 32 threads | one thread | bitwise across threads |
| --- | --- | --- | --- | --- |
| mixture | 0.020 s | 0.161 s | 3.22 s | yes |
| small alpha | 0.020 s | 0.291 s | 6.11 s | yes |
| large exposed values | 0.020 s | 0.200 s | 4.26 s | yes |

Peak RSS 191 MiB.

## Reading

- The moments are one serial classification and a few multiplications per row: 20 ms per million rows, so the pass needs no threads.
- The tail query costs one continued fraction and one density per beta row and edge (the four orders from them by recurrence), 64 million evaluations here, 50 to 95 ns each on one thread and 20 times faster on 16 cores with SMT. A beta row stops at the first edge where its survival is zero, which saves little on these portfolios: their edges lie inside most rows' supports.
- Well under a second on every portfolio, the plan's line.
