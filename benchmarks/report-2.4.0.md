# Accuracy and cost of the fine mesh, 2.4.0

Date: 2026-10-26. Machine: the DOOB workstation (AMD Ryzen 9 9950X3D, 16 cores, Windows 11), the same for every figure below; `uv run python benchmarks/run_accuracy.py --markdown` and the survival timings of `dev/done/plan-peer-review-response.md`, [mesh].

## The accuracy knob

Largest relative error of the survival method's curve against SciPy's regularized incomplete beta on one-row portfolios of bell-shaped events, $\alpha\in\{1.5,3,10,10^2,10^3,10^4\}$ and coefficients of variation in $\{10^{-3},3\times10^{-3},10^{-2},3\times10^{-2},0.1,0.2\}$, at every fine-interval midpoint and at $\mu+z\sigma$ for $z$ from $-3$ to $7$, by band of the event's survival. The base mesh alone is 100 nodes per decade; `nodes_per_sigma` is the fine mesh's density around each event's mean.

| nodes_per_sigma | $[10^{-2},1)$ | $[10^{-4},10^{-2})$ | $[10^{-6},10^{-4})$ | $[10^{-8},10^{-6})$ | $[10^{-12},10^{-8})$ | fine nodes, summed |
|---:|---:|---:|---:|---:|---:|---:|
| base only | 1.1e-02 | 1.3e-01 | 2.3e-01 | 3.1e-01 | 1.4e+00 | 0 |
| 8 | 1.5e-05 | 2.1e-03 | 1.9e-04 | 4.2e-04 | 1.2e-03 | 1,799 |
| 12 (default) | 8.6e-06 | 1.5e-04 | 3.9e-05 | 1.0e-04 | 2.1e-04 | 2,290 |
| 16 | 8.6e-06 | 1.5e-04 | 1.6e-05 | 3.3e-05 | 1.0e-04 | 2,392 |
| 20 | 8.6e-06 | 2.6e-05 | 6.2e-06 | 1.5e-05 | 3.3e-05 | 2,558 |

The $1.5\times10^{-4}$ in the second band at 12 and 16 is one case, $\operatorname{Beta}(10,7.3)$ at a coefficient of variation of $0.2$, a skewed small-parameter bell whose upper tail is steeper than a normal's; every other case is within $10^{-4}$ there. The contract of spec test 33 is read off the row for 12: $10^{-4}$ where $S\ge10^{-4}$ (with that exception), $10^{-3}$ where $S\ge10^{-8}$, $3\times10^{-3}$ where $S\ge10^{-12}$.

## Cost

Survival method, medians of three after a warm run, 32 threads, the fine mesh at its default against the same build with it off (`nodes_per_sigma=1`, which is the 2.3.6 behavior to the last bit on these portfolios).

| portfolio | fine mesh off | default | requesting events | fine nodes | base nodes |
|:---|---:|---:|---:|---:|---:|
| mixture, $10^6$ rows, $h=5{,}000$ | 1.00 s | 1.87 s | 105,779 (10.6%) | 3,797 | 332 |
| small-alpha, $10^6$ rows, $h=100$ | 2.78 s | 2.68 s | 0 | 0 | 432 |
| large exposure, $10^5$ rows, $h=25$, $2^{22}$ buckets | 1.42 s | 1.31 s | 0 | 0 | 792 |

The mixture's cost is about 7 microseconds per requesting event, about 130 incomplete-beta evaluations each: the $10T$ fine nodes in the event's window plus the base nodes between its lower skip and the window. Of its requesting events, 56,691 have coefficients of variation between 0.14 and 0.28 and ask for two pieces per base interval; 23,536 below 0.07 ask for five to twelve. The two portfolios without bell-shaped events of moderate width pay nothing and gain a little from the plateau difference and the upper skip.
