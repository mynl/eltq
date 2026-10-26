# Shared-factor rule convergence (2.6.0, backend rust)

## Two units against the bivariate normal rectangles

Units as (mu, sd_indep, sd_corr, M); bucket 10. Columns: relative error of the event total's exceedance probability at the exact 1e-2, 1e-3, 1e-4 and 1e-5 losses; relative errors of its mean, sd and q99; the largest absolute marginal error of either unit.

### A balanced: (200, 100, 150, 1000) and (500, 300, 200, 2000)

Exact total: mean 699.7998, sd 644.5714, q99 2520; normal correlation 0.462.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -4.6e-01 | -8.6e-01 | -9.7e-01 | -1.0e+00 | -1.2e-02 | -3.0e-02 | -4.4e-02 | 4.2e-03 |
| midpoint | 16 | -2.6e-01 | -6.9e-01 | -9.1e-01 | -9.9e-01 | -5.2e-03 | -1.4e-02 | -2.4e-02 | 2.0e-03 |
| midpoint | 32 | -1.3e-01 | -4.9e-01 | -7.9e-01 | -9.6e-01 | -2.2e-03 | -6.1e-03 | -1.2e-02 | 9.6e-04 |
| midpoint | 64 | -6.1e-02 | -3.1e-01 | -6.3e-01 | -8.9e-01 | -9.3e-04 | -2.7e-03 | -4.0e-03 | 4.5e-04 |
| midpoint | 128 | -2.7e-02 | -1.8e-01 | -4.6e-01 | -7.8e-01 | -4.0e-04 | -1.2e-03 | +0.0e+00 | 2.1e-04 |
| gauss_hermite | 8 | -1.8e-03 | +7.5e-03 | +5.6e-03 | -4.2e-02 | +3.0e-06 | -3.4e-05 | +0.0e+00 | 7.6e-04 |
| gauss_hermite | 16 | +2.9e-05 | -2.6e-05 | -1.8e-05 | +2.2e-04 | +6.7e-09 | -2.1e-09 | +0.0e+00 | 2.9e-06 |
| gauss_hermite | 32 | -6.1e-10 | +5.9e-09 | -2.0e-09 | -8.9e-09 | -4.6e-13 | +8.0e-13 | +0.0e+00 | 1.3e-10 |
| gauss_hermite | 64 | +0.0e+00 | +1.2e-12 | +4.9e-12 | +6.4e-11 | +8.1e-16 | -1.8e-16 | +0.0e+00 | 1.5e-15 |
| gauss_hermite | 128 | +0.0e+00 | +1.2e-12 | +4.9e-12 | +6.4e-11 | +1.1e-15 | -3.5e-16 | +0.0e+00 | 1.5e-15 |
| stratified | 8 | -3.3e-03 | -3.2e-03 | -3.0e-03 | -2.9e-03 | +6.5e-16 | -6.5e-04 | +0.0e+00 | 1.5e-15 |
| stratified | 16 | -3.0e-03 | -3.1e-03 | -3.0e-03 | -2.9e-03 | +8.1e-16 | -3.4e-04 | +0.0e+00 | 1.5e-15 |
| stratified | 32 | -3.0e-03 | -3.1e-03 | -3.0e-03 | -2.9e-03 | +6.5e-16 | -2.6e-04 | +0.0e+00 | 1.5e-15 |
| stratified | 64 | -3.0e-03 | -3.1e-03 | -3.0e-03 | -2.9e-03 | +8.1e-16 | -2.5e-04 | +0.0e+00 | 1.5e-15 |
| stratified | 128 | -3.0e-03 | -3.1e-03 | -3.0e-03 | -2.9e-03 | +1.1e-15 | -2.4e-04 | +0.0e+00 | 1.5e-15 |

### B correlation-dominant: (200, 30, 220, 1000) and (500, 60, 440, 2000)

Exact total: mean 699.7998, sd 744.9468, q99 2760; normal correlation 0.982.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -1.4e-02 | -3.6e-02 | -1.6e-01 | 2.1e-02 |
| midpoint | 16 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -4.4e-03 | -1.2e-02 | -7.6e-02 | 3.9e-03 |
| midpoint | 32 | -8.5e-01 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -1.4e-03 | -3.8e-03 | -2.2e-02 | 2.7e-03 |
| midpoint | 64 | +2.4e-01 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -4.3e-04 | -1.2e-03 | +7.2e-03 | 1.8e-03 |
| midpoint | 128 | -2.9e-02 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -1.3e-04 | -3.8e-04 | +0.0e+00 | 1.3e-03 |
| gauss_hermite | 8 | -6.9e-02 | -8.3e-01 | -5.0e-01 | +1.0e+00 | +2.1e-04 | -2.8e-03 | -8.7e-02 | 8.8e-02 |
| gauss_hermite | 16 | -2.5e-01 | -5.3e-01 | +7.8e-01 | -6.1e-01 | -4.1e-05 | +3.5e-05 | -4.0e-02 | 4.9e-02 |
| gauss_hermite | 32 | +1.4e-01 | +2.4e-01 | +2.7e-01 | -2.5e-01 | +7.9e-05 | -7.0e-05 | +1.4e-02 | 1.8e-02 |
| gauss_hermite | 64 | -8.1e-02 | -7.6e-02 | -1.0e-01 | -5.1e-02 | -2.4e-05 | +1.9e-05 | -3.6e-03 | 4.6e-03 |
| gauss_hermite | 128 | +1.8e-02 | -1.7e-02 | +5.4e-03 | -3.5e-03 | +1.6e-06 | -1.0e-06 | +3.6e-03 | 4.2e-04 |
| stratified | 8 | -3.8e-03 | -3.9e-03 | -6.7e-03 | -4.6e-04 | -4.0e-13 | -1.3e-03 | +0.0e+00 | 2.4e-15 |
| stratified | 16 | -3.8e-03 | -3.9e-03 | -6.7e-03 | -4.6e-04 | -4.0e-13 | -7.4e-04 | +0.0e+00 | 2.9e-15 |
| stratified | 32 | -3.8e-03 | -3.9e-03 | -6.7e-03 | -4.6e-04 | -4.0e-13 | -6.1e-04 | +0.0e+00 | 2.7e-15 |
| stratified | 64 | -3.8e-03 | -3.9e-03 | -6.7e-03 | -4.6e-04 | -3.8e-13 | -5.8e-04 | +0.0e+00 | 7.2e-15 |
| stratified | 128 | -3.8e-03 | -3.9e-03 | -6.7e-03 | -4.6e-04 | -3.6e-13 | -5.7e-04 | +0.0e+00 | 8.6e-15 |

### C independent: (200, 250, 0, 1000) and (500, 500, 0, 2000)

Exact total: mean 699.7998, sd 559.1320, q99 2220; normal correlation 0.000.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +1.6e-16 | +1.2e-15 | +0.0e+00 | 1.4e-15 |
| midpoint | 16 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +1.6e-16 | +1.2e-15 | +0.0e+00 | 1.4e-15 |
| midpoint | 32 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +0.0e+00 | +1.6e-15 | +0.0e+00 | 1.4e-15 |
| midpoint | 64 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +1.6e-16 | +1.2e-15 | +0.0e+00 | 1.4e-15 |
| midpoint | 128 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +1.6e-16 | +1.0e-15 | +0.0e+00 | 1.4e-15 |
| gauss_hermite | 8 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +1.6e-16 | +1.4e-15 | +0.0e+00 | 1.4e-15 |
| gauss_hermite | 16 | -8.9e-14 | +0.0e+00 | +0.0e+00 | +1.9e-11 | +1.6e-16 | +1.0e-15 | +0.0e+00 | 1.4e-15 |
| gauss_hermite | 32 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +1.6e-16 | +1.4e-15 | +0.0e+00 | 1.4e-15 |
| gauss_hermite | 64 | -1.0e-13 | -1.1e-13 | -1.1e-12 | +9.5e-12 | +0.0e+00 | +1.4e-15 | +0.0e+00 | 1.4e-15 |
| gauss_hermite | 128 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +3.2e-16 | +1.2e-15 | +0.0e+00 | 1.4e-15 |
| stratified | 8 | -7.7e-14 | +1.1e-13 | +1.1e-12 | +2.8e-11 | +1.6e-16 | +1.4e-15 | +0.0e+00 | 1.5e-15 |
| stratified | 16 | -8.9e-14 | +0.0e+00 | +0.0e+00 | +1.9e-11 | +1.6e-16 | +1.2e-15 | +0.0e+00 | 1.5e-15 |
| stratified | 32 | -1.1e-13 | -2.2e-13 | -2.1e-12 | +0.0e+00 | +3.2e-16 | +1.2e-15 | +0.0e+00 | 1.5e-15 |
| stratified | 64 | -1.0e-13 | -1.1e-13 | -1.1e-12 | +9.5e-12 | +1.6e-16 | +1.2e-15 | +0.0e+00 | 1.5e-15 |
| stratified | 128 | -7.7e-14 | +1.1e-13 | +1.1e-12 | +2.8e-11 | +1.6e-16 | +1.4e-15 | +0.0e+00 | 1.5e-15 |

### D comonotone, common shape: (200, 0, 250, 1000) and (400, 0, 500, 2000)

Exact total: mean 599.7374, sd 750.2182, q99 2790; normal correlation 1.000.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -1.0e+00 | -1.0e+00 | -1.0e+00 | nan | -1.6e-02 | -4.0e-02 | -2.2e-01 | 1.2e-01 |
| midpoint | 16 | -1.0e+00 | -1.0e+00 | -1.0e+00 | nan | -4.4e-03 | -1.2e-02 | -1.0e-01 | 5.7e-02 |
| midpoint | 32 | -1.0e+00 | -1.0e+00 | -1.0e+00 | nan | -9.7e-04 | -3.8e-03 | -2.9e-02 | 2.8e-02 |
| midpoint | 64 | +4.7e-01 | -1.0e+00 | -1.0e+00 | nan | +3.2e-05 | -1.3e-03 | +1.4e-02 | 1.3e-02 |
| midpoint | 128 | -2.7e-01 | -1.0e+00 | -1.0e+00 | nan | +3.2e-04 | -5.6e-04 | -7.2e-03 | 1.1e-02 |
| gauss_hermite | 8 | -8.5e-02 | -9.0e-01 | -3.6e-01 | nan | +7.2e-04 | -5.9e-03 | -1.8e-01 | 3.4e-01 |
| gauss_hermite | 16 | -2.7e-01 | -5.3e-01 | +3.4e-01 | nan | +4.4e-04 | -2.4e-04 | -7.5e-02 | 2.4e-01 |
| gauss_hermite | 32 | +1.5e-01 | +2.3e-01 | +3.6e-02 | nan | +4.4e-04 | -2.6e-04 | +2.2e-02 | 1.5e-01 |
| gauss_hermite | 64 | -1.7e-01 | -3.4e-01 | -2.8e-01 | nan | +4.4e-04 | -2.6e-04 | -2.5e-02 | 1.1e-01 |
| gauss_hermite | 128 | +2.1e-01 | -7.0e-02 | -1.1e-01 | nan | +4.4e-04 | -2.6e-04 | +7.2e-03 | 8.4e-02 |
| stratified | 8 | +1.7e-02 | -1.9e-13 | -7.2e-13 | nan | -1.3e-15 | -1.4e-03 | +3.6e-03 | 2.3e-15 |
| stratified | 16 | +1.7e-02 | -1.9e-13 | -7.2e-13 | nan | -1.3e-15 | -8.6e-04 | +3.6e-03 | 2.3e-15 |
| stratified | 32 | +1.7e-02 | -3.8e-13 | -2.1e-12 | nan | -1.3e-15 | -7.3e-04 | +3.6e-03 | 2.3e-15 |
| stratified | 64 | +1.7e-02 | -1.9e-13 | -7.2e-13 | nan | -1.3e-15 | -7.0e-04 | +3.6e-03 | 2.3e-15 |
| stratified | 128 | +1.7e-02 | -1.9e-13 | -7.2e-13 | nan | -1.3e-15 | -6.9e-04 | +3.6e-03 | 2.3e-15 |

### Homer event 1 (C two-point): (300, 400, 300, 3000) and (200, 300, 200, 1000)

Exact total: mean 499.8749, sd 870.0785, q99 3820; normal correlation 0.333.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -2.9e-01 | -5.1e-01 | nan | nan | -3.5e-02 | -3.7e-02 | -3.7e-02 | 3.9e-03 |
| midpoint | 16 | -1.7e-01 | -3.5e-01 | nan | nan | -1.6e-02 | -1.8e-02 | -1.8e-02 | 1.7e-03 |
| midpoint | 32 | -9.2e-02 | -2.3e-01 | nan | nan | -7.0e-03 | -8.6e-03 | -1.0e-02 | 7.0e-04 |
| midpoint | 64 | -4.9e-02 | -1.4e-01 | nan | nan | -3.1e-03 | -4.0e-03 | -5.2e-03 | 2.9e-04 |
| midpoint | 128 | -2.5e-02 | -8.2e-02 | nan | nan | -1.3e-03 | -1.8e-03 | -2.6e-03 | 1.2e-04 |
| gauss_hermite | 8 | -1.9e-05 | -8.7e-05 | nan | nan | -5.9e-07 | +4.7e-06 | +0.0e+00 | 5.6e-07 |
| gauss_hermite | 16 | +4.8e-09 | -8.2e-09 | nan | nan | +5.2e-13 | -7.3e-11 | +0.0e+00 | 2.1e-12 |
| gauss_hermite | 32 | -1.0e-13 | -6.3e-13 | nan | nan | +3.4e-15 | +7.8e-16 | +0.0e+00 | 2.2e-15 |
| gauss_hermite | 64 | -1.1e-13 | -6.3e-13 | nan | nan | +3.1e-15 | +6.5e-16 | +0.0e+00 | 2.2e-15 |
| gauss_hermite | 128 | -1.0e-13 | -5.5e-13 | nan | nan | +3.3e-15 | +7.8e-16 | +0.0e+00 | 2.2e-15 |
| stratified | 8 | -2.4e-03 | -2.1e-03 | nan | nan | +3.4e-15 | -3.5e-04 | +0.0e+00 | 2.1e-15 |
| stratified | 16 | -1.9e-03 | -1.8e-03 | nan | nan | +3.4e-15 | -2.2e-04 | +0.0e+00 | 2.1e-15 |
| stratified | 32 | -1.7e-03 | -1.7e-03 | nan | nan | +3.4e-15 | -1.8e-04 | +0.0e+00 | 2.1e-15 |
| stratified | 64 | -1.7e-03 | -1.7e-03 | nan | nan | +3.3e-15 | -1.8e-04 | +0.0e+00 | 2.1e-15 |
| stratified | 128 | -1.7e-03 | -1.7e-03 | nan | nan | +3.5e-15 | -1.7e-04 | +0.0e+00 | 2.1e-15 |

### Homer event 2 (P two-point): (100, 371, 267, 1000) and (200, 150, 533, 4000)

Exact total: mean 299.9190, sd 826.3284, q99 4350; normal correlation 0.562.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -9.5e-01 | -1.0e+00 | -1.0e+00 | nan | -1.8e-01 | -2.4e-01 | -3.0e-01 | 4.9e-03 |
| midpoint | 16 | -6.5e-01 | -1.0e+00 | -1.0e+00 | nan | -4.8e-02 | -8.0e-02 | -1.4e-01 | 2.2e-03 |
| midpoint | 32 | -2.2e-01 | -9.9e-01 | -1.0e+00 | nan | -1.1e-02 | -2.1e-02 | -5.1e-02 | 9.8e-04 |
| midpoint | 64 | -4.8e-02 | -8.9e-01 | -9.8e-01 | nan | -2.9e-03 | -5.6e-03 | -1.1e-02 | 9.1e-04 |
| midpoint | 128 | -1.7e-02 | -6.2e-01 | -8.5e-01 | nan | -8.9e-04 | -1.7e-03 | -4.6e-03 | 7.7e-04 |
| gauss_hermite | 8 | -2.4e-01 | +1.4e-01 | -2.8e-01 | nan | +3.8e-02 | -3.5e-03 | -8.3e-02 | 2.9e-02 |
| gauss_hermite | 16 | -3.6e-02 | -2.3e-02 | -1.1e-01 | nan | -2.1e-03 | +2.3e-03 | -9.2e-03 | 4.6e-03 |
| gauss_hermite | 32 | -4.0e-03 | +2.2e-03 | +5.1e-03 | nan | -8.7e-06 | -2.2e-05 | +0.0e+00 | 4.5e-04 |
| gauss_hermite | 64 | +3.0e-05 | +4.3e-05 | -4.3e-05 | nan | -7.8e-09 | +5.0e-09 | +0.0e+00 | 3.4e-06 |
| gauss_hermite | 128 | +1.3e-09 | +2.9e-09 | -2.9e-09 | nan | -2.8e-12 | +2.8e-13 | +0.0e+00 | 2.7e-10 |
| stratified | 8 | -2.2e-03 | -1.2e-03 | -9.8e-04 | nan | +1.5e-14 | -3.7e-04 | +0.0e+00 | 4.2e-15 |
| stratified | 16 | -2.2e-03 | -1.2e-03 | -9.8e-04 | nan | +1.9e-14 | -3.7e-04 | +0.0e+00 | 5.2e-15 |
| stratified | 32 | -2.2e-03 | -1.2e-03 | -9.8e-04 | nan | +3.4e-14 | -3.6e-04 | +0.0e+00 | 7.0e-15 |
| stratified | 64 | -2.2e-03 | -1.2e-03 | -9.8e-04 | nan | +1.2e-13 | -3.6e-04 | +0.0e+00 | 2.5e-14 |
| stratified | 128 | -2.2e-03 | -1.2e-03 | -9.8e-04 | nan | +3.2e-13 | -3.6e-04 | +0.0e+00 | 5.6e-14 |

### Homer event 3: (100, 224, 200, 2000) and (100, 200, 200, 2000)

Exact total: mean 199.9765, sd 639.9037, q99 3300; normal correlation 0.471.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -4.7e-01 | -6.4e-01 | nan | nan | -1.2e-01 | -1.0e-01 | -3.7e-01 | 6.4e-03 |
| midpoint | 16 | -2.8e-01 | -4.4e-01 | nan | nan | -5.8e-02 | -5.2e-02 | -2.7e-01 | 2.7e-03 |
| midpoint | 32 | -1.5e-01 | -2.7e-01 | nan | nan | -2.6e-02 | -2.6e-02 | -1.5e-01 | 1.3e-03 |
| midpoint | 64 | -7.6e-02 | -1.5e-01 | nan | nan | -1.1e-02 | -1.2e-02 | -7.0e-02 | 6.6e-04 |
| midpoint | 128 | -3.6e-02 | -8.2e-02 | nan | nan | -4.8e-03 | -5.4e-03 | -3.3e-02 | 3.1e-04 |
| gauss_hermite | 8 | -2.0e-03 | -2.0e-03 | nan | nan | +7.2e-05 | -1.3e-04 | +0.0e+00 | 4.3e-06 |
| gauss_hermite | 16 | +2.5e-06 | -3.5e-06 | nan | nan | -5.5e-09 | +2.4e-07 | +0.0e+00 | 9.7e-10 |
| gauss_hermite | 32 | +9.4e-12 | -1.2e-11 | nan | nan | +5.1e-15 | +5.5e-13 | +0.0e+00 | 2.3e-15 |
| gauss_hermite | 64 | -1.1e-14 | -1.9e-13 | nan | nan | +5.4e-15 | +2.0e-15 | +0.0e+00 | 2.3e-15 |
| gauss_hermite | 128 | +3.3e-14 | -8.0e-14 | nan | nan | +5.3e-15 | +2.0e-15 | +0.0e+00 | 2.3e-15 |
| stratified | 8 | -3.5e-03 | -3.4e-03 | nan | nan | +5.7e-15 | -3.8e-04 | -3.0e-03 | 2.3e-15 |
| stratified | 16 | -3.1e-03 | -3.2e-03 | nan | nan | +5.5e-15 | -3.4e-04 | -3.0e-03 | 2.3e-15 |
| stratified | 32 | -3.1e-03 | -3.2e-03 | nan | nan | +6.1e-15 | -3.3e-04 | -3.0e-03 | 2.3e-15 |
| stratified | 64 | -3.0e-03 | -3.2e-03 | nan | nan | +6.1e-15 | -3.3e-04 | -3.0e-03 | 2.3e-15 |
| stratified | 128 | -3.0e-03 | -3.2e-03 | nan | nan | +5.8e-15 | -3.2e-04 | -3.0e-03 | 2.3e-15 |

### beta at loading one beside A2: (200, 0, 250, 1000) and (500, 300, 200, 2000)

Exact total: mean 699.7998, sd 662.0006, q99 2570; normal correlation 0.555.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -5.6e-01 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -1.1e-02 | -3.0e-02 | -4.7e-02 | 1.2e-01 |
| midpoint | 16 | -2.5e-01 | -1.0e+00 | -1.0e+00 | -1.0e+00 | -4.1e-03 | -1.2e-02 | -1.9e-02 | 5.7e-02 |
| midpoint | 32 | -1.1e-01 | -6.6e-01 | -1.0e+00 | -1.0e+00 | -1.5e-03 | -5.2e-03 | -7.8e-03 | 2.5e-02 |
| midpoint | 64 | -4.9e-02 | -3.2e-01 | -1.0e+00 | -1.0e+00 | -5.2e-04 | -2.3e-03 | -3.9e-03 | 1.1e-02 |
| midpoint | 128 | -2.2e-02 | -1.5e-01 | -6.5e-01 | -1.0e+00 | -1.0e-04 | -1.1e-03 | +0.0e+00 | 1.1e-02 |
| gauss_hermite | 8 | +5.2e-02 | +6.7e-02 | +3.0e-01 | -5.9e-01 | +2.9e-04 | -1.4e-03 | +3.9e-03 | 2.4e-01 |
| gauss_hermite | 16 | -2.1e-02 | -1.4e-01 | +1.1e-01 | -1.3e-01 | +2.1e-04 | -1.4e-04 | +0.0e+00 | 1.6e-01 |
| gauss_hermite | 32 | -4.8e-03 | -3.1e-02 | -7.9e-02 | -1.6e-02 | +2.1e-04 | -1.5e-04 | +0.0e+00 | 1.5e-01 |
| gauss_hermite | 64 | +1.8e-03 | +1.3e-02 | -3.5e-02 | +6.0e-03 | +2.1e-04 | -1.5e-04 | +0.0e+00 | 1.1e-01 |
| gauss_hermite | 128 | -1.8e-03 | +1.1e-03 | +6.1e-03 | -1.4e-02 | +2.1e-04 | -1.5e-04 | +0.0e+00 | 5.9e-02 |
| stratified | 8 | -3.3e-03 | -2.8e-03 | -2.0e-03 | -2.5e-03 | +4.9e-16 | -7.3e-04 | +0.0e+00 | 1.6e-15 |
| stratified | 16 | -3.3e-03 | -2.8e-03 | -2.0e-03 | -2.5e-03 | +6.5e-16 | -4.3e-04 | +0.0e+00 | 1.6e-15 |
| stratified | 32 | -3.3e-03 | -2.8e-03 | -2.0e-03 | -2.5e-03 | +6.5e-16 | -3.5e-04 | +0.0e+00 | 1.6e-15 |
| stratified | 64 | -3.3e-03 | -2.8e-03 | -2.0e-03 | -2.5e-03 | +6.5e-16 | -3.3e-04 | +0.0e+00 | 1.6e-15 |
| stratified | 128 | -3.3e-03 | -2.8e-03 | -2.0e-03 | -2.5e-03 | +6.5e-16 | -3.2e-04 | +0.0e+00 | 1.6e-15 |

### two-point at loading one beside A2: (30, 0, 80, 100) and (500, 300, 200, 2000)

Exact total: mean 529.9452, sd 521.2756, q99 1930; normal correlation 0.555.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | mean | sd | q99 | marginal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -2.3e-01 | -3.8e-01 | -5.0e-01 | -6.1e-01 | -1.8e-02 | -1.8e-02 | -1.6e-02 | 5.0e-02 |
| midpoint | 16 | -1.0e-01 | -2.3e-01 | -3.4e-01 | -4.5e-01 | -1.4e-03 | -6.7e-03 | -1.0e-02 | 1.3e-02 |
| midpoint | 32 | -5.2e-02 | -1.4e-01 | -2.3e-01 | -3.3e-01 | +6.0e-04 | -2.8e-03 | -5.2e-03 | 1.3e-02 |
| midpoint | 64 | -3.4e-02 | -8.8e-02 | -1.5e-01 | -2.3e-01 | -1.4e-03 | -1.8e-03 | -5.2e-03 | 3.1e-03 |
| midpoint | 128 | -1.9e-02 | -5.2e-02 | -9.6e-02 | -1.5e-01 | -9.6e-04 | -9.1e-04 | -5.2e-03 | 3.1e-03 |
| gauss_hermite | 8 | +9.6e-02 | +5.7e-02 | +3.1e-02 | +1.7e-02 | +3.8e-02 | +5.5e-03 | +5.2e-03 | 2.0e-01 |
| gauss_hermite | 16 | -6.0e-02 | -4.0e-02 | -2.3e-02 | -1.3e-02 | -1.6e-02 | -4.9e-03 | -5.2e-03 | 8.7e-02 |
| gauss_hermite | 32 | -2.7e-03 | -5.8e-04 | +8.4e-05 | +2.7e-04 | -2.2e-03 | -9.4e-05 | -5.2e-03 | 1.2e-02 |
| gauss_hermite | 64 | +3.0e-02 | +2.0e-02 | +1.2e-02 | +6.8e-03 | +8.8e-03 | +2.2e-03 | +0.0e+00 | 4.7e-02 |
| gauss_hermite | 128 | -6.2e-03 | -3.9e-03 | -2.2e-03 | -1.2e-03 | -2.1e-03 | -4.4e-04 | -5.2e-03 | 1.1e-02 |
| stratified | 8 | -3.8e-03 | -3.4e-03 | -2.4e-03 | -1.6e-03 | +1.1e-15 | -3.6e-04 | -5.2e-03 | 1.4e-15 |
| stratified | 16 | -2.1e-04 | -1.9e-04 | -1.3e-04 | -8.2e-05 | +1.3e-15 | -2.2e-05 | +0.0e+00 | 1.5e-15 |
| stratified | 32 | -1.0e-04 | -9.3e-05 | -6.3e-05 | -4.1e-05 | +1.3e-15 | -1.0e-05 | +0.0e+00 | 1.5e-15 |
| stratified | 64 | -4.5e-05 | -4.0e-05 | -2.8e-05 | -1.8e-05 | +1.1e-15 | -4.4e-06 | +0.0e+00 | 1.5e-15 |
| stratified | 128 | -1.4e-05 | -1.3e-05 | -8.9e-06 | -5.8e-06 | +1.5e-15 | -1.4e-06 | +0.0e+00 | 1.5e-15 |

## A lone beta unit against its direct bucketing

A one-unit event (200, sd_indep, sd_corr, 1000) at total sd 250 and loading w; the mixture over the rule should return the marginal. Largest absolute bucket error and relative exceedance error at the 1e-3 and 1e-5 losses.

| w | rule | Q | marginal | S 1e-3 | S 1e-5 |
| --- | --- | --- | --- | --- | --- |
| 0.5 | midpoint | 8 | 3.4e-03 | -2.6e-01 | nan |
| 0.5 | midpoint | 32 | 6.5e-04 | -9.7e-02 | nan |
| 0.5 | midpoint | 128 | 1.2e-04 | -3.3e-02 | nan |
| 0.5 | gauss_hermite | 8 | 1.7e-08 | -3.5e-07 | nan |
| 0.5 | gauss_hermite | 32 | 1.4e-15 | -3.0e-13 | nan |
| 0.5 | gauss_hermite | 128 | 1.4e-15 | +3.0e-13 | nan |
| 0.5 | stratified | 8 | 1.5e-15 | +7.6e-14 | nan |
| 0.5 | stratified | 32 | 1.4e-15 | +0.0e+00 | nan |
| 0.5 | stratified | 128 | 1.4e-15 | +1.5e-13 | nan |
| 0.8 | midpoint | 8 | 3.0e-03 | -8.4e-01 | nan |
| 0.8 | midpoint | 32 | 4.6e-04 | -5.0e-01 | nan |
| 0.8 | midpoint | 128 | 1.5e-04 | -2.0e-01 | nan |
| 0.8 | gauss_hermite | 8 | 2.9e-04 | +7.3e-03 | nan |
| 0.8 | gauss_hermite | 32 | 3.2e-12 | -1.7e-11 | nan |
| 0.8 | gauss_hermite | 128 | 1.5e-15 | -7.6e-14 | nan |
| 0.8 | stratified | 8 | 1.4e-15 | +0.0e+00 | nan |
| 0.8 | stratified | 32 | 1.4e-15 | +7.6e-14 | nan |
| 0.8 | stratified | 128 | 1.4e-15 | +1.5e-13 | nan |
| 0.9 | midpoint | 8 | 1.3e-03 | -9.9e-01 | nan |
| 0.9 | midpoint | 32 | 8.2e-04 | -8.1e-01 | nan |
| 0.9 | midpoint | 128 | 2.9e-04 | -3.8e-01 | nan |
| 0.9 | gauss_hermite | 8 | 5.7e-03 | +7.9e-02 | nan |
| 0.9 | gauss_hermite | 32 | 3.6e-07 | -3.7e-06 | nan |
| 0.9 | gauss_hermite | 128 | 1.4e-15 | -7.6e-14 | nan |
| 0.9 | stratified | 8 | 1.5e-15 | +0.0e+00 | nan |
| 0.9 | stratified | 32 | 1.4e-15 | -7.6e-14 | nan |
| 0.9 | stratified | 128 | 1.4e-15 | -1.2e-12 | nan |
| 0.99 | midpoint | 8 | 5.8e-03 | -1.0e+00 | nan |
| 0.99 | midpoint | 32 | 2.5e-03 | -1.0e+00 | nan |
| 0.99 | midpoint | 128 | 1.2e-03 | -9.6e-01 | nan |
| 0.99 | gauss_hermite | 8 | 8.6e-02 | -4.2e-01 | nan |
| 0.99 | gauss_hermite | 32 | 1.6e-02 | +1.7e-01 | nan |
| 0.99 | gauss_hermite | 128 | 2.7e-04 | -9.6e-04 | nan |
| 0.99 | stratified | 8 | 1.4e-15 | -2.4e-12 | nan |
| 0.99 | stratified | 32 | 1.4e-15 | -3.0e-12 | nan |
| 0.99 | stratified | 128 | 1.5e-15 | -1.4e-11 | nan |

## The fixture against the stratified rule at Q = 4096

`fixtures/multiunit_small_elt.csv` with its terms, all four views; the reference is the stratified rule at Q = 4096 (its two-unit error against the exact rectangles is in the tables above). Columns: the ground-up total's relative exceedance error at the 1e-2 to 1e-5 fractions of the total rate, the largest relative view-mean error, and the engine's wall time for the four views.

| rule | Q | S 1e-2 | S 1e-3 | S 1e-4 | S 1e-5 | view mean | time (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midpoint | 8 | -1.8e-01 | -7.0e-01 | -9.8e-01 | -1.0e+00 | 4.1e-02 | 0.9 |
| midpoint | 16 | -8.3e-02 | -5.0e-01 | -9.5e-01 | -1.0e+00 | 2.2e-02 | 0.8 |
| midpoint | 32 | -3.5e-02 | -3.1e-01 | -8.5e-01 | -1.0e+00 | 9.3e-03 | 1.2 |
| midpoint | 64 | -1.3e-02 | -1.6e-01 | -6.7e-01 | -9.7e-01 | 3.2e-03 | 2.0 |
| midpoint | 128 | -4.1e-03 | -7.9e-02 | -4.6e-01 | -9.1e-01 | 1.4e-03 | 3.7 |
| gauss_hermite | 8 | -2.9e-04 | -5.5e-03 | +6.9e-02 | -1.8e-01 | 4.0e-02 | 0.6 |
| gauss_hermite | 16 | +2.1e-03 | +3.1e-03 | +7.6e-03 | +7.8e-03 | 7.5e-04 | 0.7 |
| gauss_hermite | 32 | +2.1e-03 | +4.0e-03 | +4.1e-03 | +3.9e-03 | 6.0e-03 | 1.1 |
| gauss_hermite | 64 | +2.1e-03 | +4.0e-03 | +4.2e-03 | +4.0e-03 | 5.9e-03 | 1.9 |
| gauss_hermite | 128 | +2.1e-03 | +4.0e-03 | +4.2e-03 | +4.0e-03 | 1.0e-03 | 3.4 |
| stratified | 8 | -1.5e-03 | -2.7e-04 | -5.2e-06 | -1.6e-08 | 2.1e-04 | 4.0 |
| stratified | 16 | -3.7e-04 | -6.8e-05 | -1.4e-06 | +4.1e-09 | 2.1e-04 | 4.8 |
| stratified | 32 | -9.4e-05 | -1.7e-05 | -3.5e-07 | +9.7e-09 | 2.1e-04 | 5.6 |
| stratified | 64 | -2.3e-05 | -4.3e-06 | -8.6e-08 | +1.1e-08 | 2.1e-04 | 7.8 |
| stratified | 128 | -5.9e-06 | -1.1e-06 | -2.1e-08 | +1.1e-08 | 2.1e-04 | 12.1 |

## Worst case over the two-unit table by rule and Q

Largest absolute relative exceedance error over the four levels and the nine cases, and the largest marginal error.

| rule | Q | worst tail | worst marginal |
| --- | --- | --- | --- |
| midpoint | 8 | 1.0e+00 | 1.2e-01 |
| midpoint | 16 | 1.0e+00 | 5.7e-02 |
| midpoint | 32 | 1.0e+00 | 2.8e-02 |
| midpoint | 64 | 1.0e+00 | 1.3e-02 |
| midpoint | 128 | 1.0e+00 | 1.1e-02 |
| gauss_hermite | 8 | 1.0e+00 | 3.4e-01 |
| gauss_hermite | 16 | 7.8e-01 | 2.4e-01 |
| gauss_hermite | 32 | 2.7e-01 | 1.5e-01 |
| gauss_hermite | 64 | 3.4e-01 | 1.1e-01 |
| gauss_hermite | 128 | 2.1e-01 | 8.4e-02 |
| stratified | 8 | 1.7e-02 | 4.2e-15 |
| stratified | 16 | 1.7e-02 | 5.2e-15 |
| stratified | 32 | 1.7e-02 | 7.0e-15 |
| stratified | 64 | 1.7e-02 | 2.5e-14 |
| stratified | 128 | 1.7e-02 | 5.6e-14 |

