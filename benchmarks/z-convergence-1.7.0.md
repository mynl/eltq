# Shared-factor quadrature convergence

Reference: midpoint rule at Q = 4001. Units (mu, sd_indep, sd_corr, M); bucket 10.0, grids 101 x 201.

## A balanced: (200, 100, 150, 1000) and (500, 300, 200, 2000)

Reference sum: mean 699.7954, sd 644.5582, q99 2520.0

| rule | Q | max marginal err | mean rel | sd rel | q99 rel | max sum cell diff |
| --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | 4.2e-03 | -1.2e-02 | -3.0e-02 | -4.4e-02 | 4.8e-03 |
| midpoint | 16 | 2.0e-03 | -5.2e-03 | -1.4e-02 | -2.4e-02 | 2.2e-03 |
| midpoint | 32 | 9.6e-04 | -2.2e-03 | -6.1e-03 | -1.2e-02 | 9.9e-04 |
| midpoint | 64 | 4.5e-04 | -9.2e-04 | -2.7e-03 | -4.0e-03 | 4.5e-04 |
| midpoint | 128 | 2.1e-04 | -3.9e-04 | -1.2e-03 | +0.0e+00 | 2.0e-04 |
| gauss_hermite | 8 | 7.6e-04 | +9.3e-06 | -1.3e-05 | +0.0e+00 | 1.7e-04 |
| gauss_hermite | 16 | 2.9e-06 | +6.3e-06 | +2.1e-05 | +0.0e+00 | 4.6e-06 |
| gauss_hermite | 32 | 1.3e-10 | +6.3e-06 | +2.1e-05 | +0.0e+00 | 3.8e-06 |
| gauss_hermite | 64 | 2.0e-16 | +6.3e-06 | +2.1e-05 | +0.0e+00 | 3.8e-06 |
| gauss_hermite | 128 | 2.0e-16 | +6.3e-06 | +2.1e-05 | +0.0e+00 | 3.8e-06 |

## B correlation-dominant: (200, 30, 220, 1000) and (500, 60, 440, 2000)

Reference sum: mean 699.7995, sd 744.9458, q99 2760.0

| rule | Q | max marginal err | mean rel | sd rel | q99 rel | max sum cell diff |
| --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | 2.1e-02 | -1.4e-02 | -3.6e-02 | -1.6e-01 | 2.2e-02 |
| midpoint | 16 | 3.9e-03 | -4.4e-03 | -1.2e-02 | -7.6e-02 | 2.3e-03 |
| midpoint | 32 | 2.7e-03 | -1.4e-03 | -3.8e-03 | -2.2e-02 | 1.5e-03 |
| midpoint | 64 | 1.8e-03 | -4.2e-04 | -1.2e-03 | +7.2e-03 | 1.0e-03 |
| midpoint | 128 | 1.3e-03 | -1.3e-04 | -3.8e-04 | +0.0e+00 | 7.2e-04 |
| gauss_hermite | 8 | 8.8e-02 | +2.1e-04 | -2.8e-03 | -8.7e-02 | 3.6e-02 |
| gauss_hermite | 16 | 4.9e-02 | -4.1e-05 | +3.7e-05 | -4.0e-02 | 3.8e-02 |
| gauss_hermite | 32 | 1.8e-02 | +7.9e-05 | -6.8e-05 | +1.4e-02 | 1.7e-02 |
| gauss_hermite | 64 | 4.6e-03 | -2.3e-05 | +2.1e-05 | -3.6e-03 | 5.2e-03 |
| gauss_hermite | 128 | 4.2e-04 | +2.1e-06 | +3.7e-07 | +3.6e-03 | 1.0e-03 |

## C independent: (200, 250, 0, 1000) and (500, 500, 0, 2000)

Reference sum: mean 699.7998, sd 559.1320, q99 2220.0

| rule | Q | max marginal err | mean rel | sd rel | q99 rel | max sum cell diff |
| --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | 3.3e-16 | +0.0e+00 | +8.1e-16 | +0.0e+00 | 1.2e-15 |
| midpoint | 16 | 3.3e-16 | +1.6e-16 | +6.1e-16 | +0.0e+00 | 1.2e-15 |
| midpoint | 32 | 3.3e-16 | +0.0e+00 | +8.1e-16 | +0.0e+00 | 1.2e-15 |
| midpoint | 64 | 3.3e-16 | +0.0e+00 | +8.1e-16 | +0.0e+00 | 1.2e-15 |
| midpoint | 128 | 3.3e-16 | +0.0e+00 | +8.1e-16 | +0.0e+00 | 1.2e-15 |
| gauss_hermite | 8 | 3.3e-16 | +1.6e-16 | +6.1e-16 | +0.0e+00 | 1.2e-15 |
| gauss_hermite | 16 | 3.3e-16 | +1.6e-16 | +6.1e-16 | +0.0e+00 | 1.2e-15 |
| gauss_hermite | 32 | 3.3e-16 | +0.0e+00 | +4.1e-16 | +0.0e+00 | 1.2e-15 |
| gauss_hermite | 64 | 3.3e-16 | +1.6e-16 | +6.1e-16 | +0.0e+00 | 1.2e-15 |
| gauss_hermite | 128 | 3.3e-16 | +1.6e-16 | +6.1e-16 | +0.0e+00 | 1.2e-15 |

## D comonotone: (200, 0, 250, 1000) and (400, 0, 500, 2000)

Reference sum: mean 599.9999, sd 750.0197, q99 2790.0

| rule | Q | max marginal err | mean rel | sd rel | q99 rel | max sum cell diff |
| --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | 1.2e-01 | -1.7e-02 | -3.9e-02 | -2.2e-01 | 1.2e-01 |
| midpoint | 16 | 5.7e-02 | -4.9e-03 | -1.2e-02 | -1.0e-01 | 5.5e-02 |
| midpoint | 32 | 2.8e-02 | -1.4e-03 | -3.5e-03 | -2.9e-02 | 2.5e-02 |
| midpoint | 64 | 1.3e-02 | -4.1e-04 | -1.0e-03 | +1.4e-02 | 1.0e-02 |
| midpoint | 128 | 1.1e-02 | -1.2e-04 | -3.0e-04 | -7.2e-03 | 6.1e-03 |
| gauss_hermite | 8 | 3.4e-01 | +2.9e-04 | -5.7e-03 | -1.8e-01 | 2.0e-01 |
| gauss_hermite | 16 | 2.4e-01 | +2.9e-06 | +2.9e-05 | -7.5e-02 | 1.6e-01 |
| gauss_hermite | 32 | 1.5e-01 | +2.4e-07 | +1.6e-06 | +2.2e-02 | 1.4e-01 |
| gauss_hermite | 64 | 1.1e-01 | +2.4e-07 | +6.4e-06 | -2.5e-02 | 7.0e-02 |
| gauss_hermite | 128 | 8.4e-02 | +2.4e-07 | +5.4e-06 | +7.2e-03 | 5.7e-02 |

## Reading

Gauss-Hermite is spectrally accurate where the conditional cdf is smooth in z (balanced regime: marginal error 1e-10 at Q = 32 against 1e-3 for midpoint) but loses that advantage as the conditional approaches a step in z (correlation-dominant and comonotone units), where its clustered nodes leave visible artifacts in the pmf while the equal-weight midpoint rule degrades gracefully and keeps the better distribution shape. Neither rule dominates; the midpoint rule is the robust default for pmf shape at Q = 32, Gauss-Hermite the option for moment accuracy on smooth portfolios.
