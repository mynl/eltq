# Build Brief: Ferociously Fast Portfolio ELT Secondary Uncertainty

Date: 2026-08-14

## 1. Assignment

Build a production-quality utility implementing the mathematical contract in `secondary-uncertainty-mathematical-specification.md`. The primary goal is extraordinary end-to-end speed on portfolio ELTs containing approximately one million rows, without compromising numerical validation, deterministic behaviour or Python usability.

The deliverable must expose a Python package built from a Rust computational core. It must return the raw and secondary-uncertainty grid distributions directly as NumPy arrays, support writable NumPy-compatible memory maps, and optionally operate as a command-line utility. Treat performance as an empirical engineering problem: establish a trusted reference, measure every proposed optimisation, and retain complexity only when benchmarks justify it.

Do not materialize the secondary-uncertainty-expanded ELT. The expanded points are an intermediate mathematical concept, not an output data structure.

## 2. Required technology

Use the following baseline stack unless a benchmark demonstrates a compelling reason to replace a component:

- Rust for the computational core;
- Rayon for CPU data parallelism;
- PyO3 for Python bindings;
- the Rust `numpy` crate for NumPy array exchange;
- Maturin for building and packaging Python wheels;
- the Rust `csv` crate for the initial streaming CSV reader;
- SciPy in the test environment as an independent beta-quantile reference;
- Criterion or an equivalent rigorous Rust benchmark harness.

Keep the mathematical core independent of Python, PyO3, NumPy and command-line types. Python bindings, file adapters and the CLI must call the same tested core API.

Useful current documentation:

- Maturin: <https://www.maturin.rs/>
- PyO3: <https://pyo3.rs/>
- Rust NumPy bindings: <https://docs.rs/numpy/latest/numpy/>
- Rayon: <https://docs.rs/rayon/latest/rayon/>
- Rust CSV: <https://docs.rs/csv/latest/csv/>
- `statrs::Beta`: <https://docs.rs/statrs/latest/statrs/distribution/struct.Beta.html>
- `statrs` inverse-beta source: <https://docs.rs/statrs/latest/src/statrs/function/beta.rs.html>
- NumPy memory mapping: <https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>

## 3. Project structure

Create a Cargo workspace with narrowly separated components:

```text
eltq/
├── Cargo.toml
├── pyproject.toml
├── crates/
│   ├── eltq-core/
│   ├── eltq-python/
│   └── eltq-cli/
├── python/
│   └── eltq/
├── tests/
├── benchmarks/
└── README.md
```

`eltq-core` owns schemas, validation, beta parameterization, quantile generation, histogram accumulation, deterministic reduction and numerical diagnostics. `eltq-python` contains only PyO3 conversion and error translation. `eltq-cli` is optional for the first milestone but should remain a thin adapter. The pure-Python layer may provide conveniences for pandas and path handling, but it must not perform the expensive computation.

Use `eltq` as the provisional package name unless the project owner supplies a final name.

## 4. Core data model

Represent input data in a structure-of-arrays layout for the computational path:

- event identifiers or stable row numbers;
- event rates;
- maximum losses;
- mean losses;
- independent standard deviations;
- correlated standard deviations.

Use `f64` for every loss, rate, beta parameter, quantile and accumulator. Do not introduce `f32` without a benchmark and a quantified end-to-end error analysis.

Define explicit configuration and result types. The configuration must include grid spacing, grid length, $K$, normalization, validation policy, overflow policy, deterministic mode, thread limit and memory limit. The result must contain the two histograms plus all metadata and diagnostics required by the mathematical specification.

Use typed errors. Invalid rows must never panic the process. Collect counts and a bounded set of representative event identifiers for each error class.

## 5. Streaming computational pipeline

Process the ELT in large bounded chunks. For each chunk:

1. parse or receive the six input columns;
2. validate rows and classify degenerate cases;
3. apply the raw linear deposit at $\mu_j$;
4. calculate admissible beta parameters for nondegenerate rows;
5. produce the $K$ midpoint beta quantiles;
6. scale each quantile by $M_j$;
7. apply the linear grid deposit immediately;
8. discard the event-local quantiles;
9. reduce worker-local histograms in a defined order.

Never allocate an $N\times K$ array and never write an expanded ELT unless a separate diagnostic command is explicitly added later. Reuse event-local fixed-capacity buffers. Prefer stack storage for small fixed $K$ where practical and a reusable heap buffer otherwise.

Compute raw and secondary histograms in the same input pass. Check the grid upper bound before beginning expensive quantile work whenever the input mode permits; otherwise detect and report overflow during streaming without unsafe indexing.

## 6. Parallel histogram design

Use Rayon threads rather than operating-system processes. Each worker should accumulate into private dense histograms, eliminating locks and atomic floating-point additions from the hot loop. Reduce private histograms only after their assigned work is complete.

The memory cost of two private histograms is
$$
16nT\ \text{bytes},
$$
where $n$ is the grid length and $T$ is the worker count. Select the effective worker count using both available physical cores and the configured memory budget. If large grids make full private histograms impractical, benchmark a tiled or hybrid reduction; do not default to hash maps or atomics without evidence.

Partition work into fixed indexed chunks and combine partial histograms in a fixed tree order in deterministic mode. Provide a faster nondeterministic mode only if it produces a material measured advantage and remains numerically within tolerance.

Do not assume more threads are always faster. Benchmark physical-core counts and avoid oversubscription from nested parallel libraries.

## 7. Inverse-beta kernel

The inverse-beta calculation will probably dominate runtime. Treat it as a dedicated subsystem with a small interface that can support several implementations.

### 7.1 Reference kernel

Begin with a scalar, high-accuracy kernel using `statrs::Beta::inverse_cdf` or another validated implementation. Establish correctness across the entire parameter domain before optimising. Compare every later kernel against SciPy and the reference kernel.

### 7.2 Batched event kernel

Implement a specialised function conceptually equivalent to

```text
beta_midpoint_quantiles(alpha, beta, K, loss_tolerance, output_buffer)
```

The function receives sorted, fixed midpoint probabilities. It should:

- compute $\log B(\alpha,\beta)$ and other event-level constants once;
- reuse initialisation across the $K$ probabilities;
- exploit monotonicity of successive quantiles;
- use a robust initial approximation followed by safeguarded Newton or Halley refinement;
- maintain a bracket so difficult shapes cannot diverge;
- exploit beta reflection where useful;
- stop according to the required loss-grid accuracy rather than unconditional machine precision;
- return explicit status information for nonconvergence or boundary loss of precision.

The relevant normalized tolerance depends on $h/M_j$. Because the linear rule uses the fractional position within a grid interval, do not stop merely after identifying a bucket. Define and test a quantitative bound on the resulting histogram-weight error.

The current `statrs` inverse-beta source uses AS 64/AS 109 initial approximations and iterative refinement. It is a suitable starting reference, but its scalar API repeats event-level setup for each probability. The production kernel should avoid that repetition.

### 7.3 Fast paths

Implement cheap explicit paths for:

- zero variance;
- means at zero or $M$;
- $K=1$;
- uniform beta $\alpha=\beta=1$;
- parameter regimes for which a demonstrably safe closed form exists;
- events whose requested tolerance permits a cheaper validated approximation.

Do not enable adaptive $K$ or a lookup-table approximation in the default mathematical mode because either changes the specified quadrature. They may be added as explicit alternative modes after their errors and performance are characterized.

### 7.4 SIMD

The inverse-beta iterations are branchy and may converge at different rates. Do not assume SIMD will help. First optimise scalar setup reuse and thread-level parallelism. A masked SIMD kernel may be investigated only after profiling demonstrates sufficient uniformity across batches.

## 8. Linear-grid accumulation

Make the linear deposit a tiny inlinable function with checked validation outside the hot path. Handle exact grid points and the upper endpoint explicitly. It must never calculate or write an index outside the histogram.

Avoid division in the inner point loop where possible by computing the reciprocal grid spacing once. Measure whether fused multiply-add operations improve speed without harming reproducibility. Accumulate in `f64`.

The raw path requires one deposit per valid event. The secondary path requires $K$ deposits per nondegenerate event. Each deposit normally updates exactly two adjacent elements of a private dense array.

## 9. Input interfaces

Provide two first-class input routes.

### 9.1 File route

Rust must read CSV directly rather than routing a million rows through pandas or Python objects. Parse into reusable column buffers and dispatch large chunks to the core. Avoid per-cell string allocation where possible. Begin with the mature Rust `csv` crate; investigate `simd_csv`, Arrow or Parquet only if profiling identifies input as material.

Allow configurable column names and provide a strict canonical schema. Report missing columns, parse failures and nonfinite values with bounded diagnostics.

### 9.2 Array route

Accept contiguous one-dimensional NumPy arrays for each numeric column. Require `float64` initially. Borrow read-only slices without copying when the arrays are C-contiguous and aligned; either reject incompatible arrays clearly or let the thin Python wrapper make an explicit contiguous conversion.

Event identifiers may be omitted from the hot array API and replaced by row indices. Do not accept object arrays in the native computational path.

## 10. Python API

Build the extension with Maturin and PyO3. Provide at least these public operations:

```python
raw, secondary = eltq.quantize_csv(
    path,
    *,
    bucket_size,
    grid_length,
    quadrature_points,
    **options,
)
```

```python
raw, secondary = eltq.quantize_arrays(
    event_rate,
    maximum_loss,
    mean_loss,
    independent_sd,
    correlated_sd,
    *,
    bucket_size,
    grid_length,
    quadrature_points,
    **options,
)
```

```python
metadata = eltq.quantize_into(
    source,
    *,
    raw_out,
    secondary_out,
    bucket_size,
    quadrature_points,
    **options,
)
```

For ordinary return values, compute into Rust-owned `Vec<f64>` histograms while detached from the Python interpreter, then transfer ownership into NumPy using `IntoPyArray`. Do not copy the completed vectors into newly allocated Python arrays.

`quantize_into` must accept writable, C-contiguous `float64` NumPy arrays of the exact required length, including `numpy.memmap` subclasses. Validate dtype, shape, contiguity and writability before computing. Decide carefully whether to hold the Python lock while writing borrowed output memory or to compute into Rust-owned buffers and perform a final copy; never release Python protection around borrowed writable memory without a sound lifetime and concurrent-access argument.

Return structured metadata through a small Python class or dictionary. Convert Rust errors into specific Python exception classes. Long Rust-owned calculations must use the current PyO3 detachment mechanism so unrelated Python threads can run.

Do not make pandas a dependency of the compiled extension. A separate Python convenience function may accept a DataFrame and extract contiguous columns explicitly.

## 11. Memory-mapped output

Support two memory-mapped workflows.

1. Python creates two writable `numpy.memmap` or `.npy`-backed arrays and passes them to `quantize_into`.
2. A file-output operation writes a standard NumPy `.npy` array of shape `(2,n)`, with row zero containing the raw distribution and row one containing the secondary distribution. Python must be able to reopen it using `np.load(path, mmap_mode="r")`.

Do not use compressed `.npz` for the memory-mapped mode. Store metadata in a small adjacent JSON file or return it to the caller. Write files transactionally where practical so interruption cannot leave a completed-looking but partial result.

Memory mapping is an output and interoperability feature, not necessarily a speed improvement for modest grids. Benchmark it separately from in-memory output.

## 12. Validation and reference implementation

Create a short, transparent Python/SciPy reference implementation. It may be slow and must favour clarity and accuracy. Use it to generate golden outputs for small and medium test ELTs.

Test the complete list in Section 9 of the mathematical specification. Add property-based Rust tests for mass conservation, linear-rule mean preservation, scaling and admissible beta reconstruction. Test extreme beta shapes, probabilities near zero and one, very small and large $h/M$, exact maximum-loss endpoints, zero-rate rows and duplicated event identifiers.

For approximate inverse-beta stopping, measure both quantile error and induced histogram error. A quantile can cross a grid boundary while remaining numerically close, so compare final histograms as well as individual quantiles.

Use compensated or pairwise reference summation when evaluating validation totals; do not mistake ordinary summation-order noise for an algorithmic error.

## 13. Benchmark programme

Establish benchmarks before writing specialised optimisation code. Record release-build results only. Every report must state the CPU, physical and logical core counts, memory, compiler version, compilation flags, Python version, grid length, $K$, input format and input characteristics.

Measure:

- CSV parsing alone;
- beta parameter construction alone;
- scalar inverse-beta throughput by parameter regime;
- batched inverse-beta throughput by parameter regime;
- linear deposits alone;
- complete in-memory transformation;
- complete memory-mapped transformation;
- scaling at 1, 2, 4, 8 and all physical cores;
- peak resident memory;
- deterministic reduction cost;
- Python boundary and output-conversion cost.

Use at least the datasets required by Section 10 of the mathematical specification. The principal production benchmark should contain one million rows with $K=32$. Report rows/s, quantiles/s, elapsed time and speedup relative to the SciPy reference and the initial scalar Rust implementation.

Profile before and after every substantial optimisation. Keep a benchmark history in a machine-readable file so apparent improvements survive scrutiny.

## 14. GPU investigation

Do not put GPU support on the critical path for the first production CPU release. The workload combines iterative, divergent inverse-beta calculations with irregular histogram writes, so GPU advantage is not guaranteed.

After the CPU implementation is correct and profiled, conduct a bounded GPU spike only if the expected production workload is large enough to amortize transfer and launch costs. Compare at least:

- a direct inverse-beta GPU kernel;
- a validated lookup/interpolation quantile kernel;
- block-local histograms followed by reduction;
- end-to-end time including host-device transfer.

Retain GPU support only if it provides a substantial end-to-end improvement on hardware representative of actual users. Keep the CPU implementation as the universal reference and fallback.

## 15. Delivery milestones

### Milestone 1: mathematical reference

- Confirm unresolved assumptions in Section 11 of the mathematical specification.
- Produce the Python/SciPy reference and golden test fixtures.
- Demonstrate raw and secondary mass and moment diagnostics.

### Milestone 2: correct Rust core

- Implement parsing-independent core types and scalar transformation.
- Pass all golden and property tests.
- Add direct CSV input and the initial benchmark suite.

### Milestone 3: Python package

- Add PyO3 and NumPy bindings.
- Build installable Windows wheels using Maturin.
- Return Rust-owned histograms as NumPy arrays without a completed-array copy.
- Add writable array and memory-map workflows.

### Milestone 4: ferocious CPU optimisation

- Add Rayon chunk parallelism and private histograms.
- Implement and validate the batched inverse-beta kernel.
- Tune tolerance, chunk size, worker count and reduction strategy from benchmarks.
- Produce an end-to-end one-million-row performance report.

### Milestone 5: optional investigations

- Evaluate faster file formats and SIMD only when profiling supports them.
- Conduct the bounded GPU spike if the CPU benchmark and expected workloads justify it.

## 16. Definition of done

The initial production release is complete when:

- all mathematical acceptance tests pass against the independent reference;
- a one-million-row ELT is processed without materializing expanded points;
- the Python API returns ordinary NumPy arrays and supports memory-mapped output;
- invalid input is diagnosed without crashes or silent alteration;
- deterministic mode has a documented reproducibility guarantee;
- Windows wheels build reproducibly through Maturin;
- performance and memory benchmarks are reproducible and checked into the project;
- the optimised kernel demonstrates a material end-to-end improvement over both SciPy and the scalar Rust baseline;
- the README documents the mathematical assumptions, Python API, file schema, diagnostics and benchmark commands.

The standard for success is not merely “implemented in Rust.” The completed utility must demonstrate, with reproducible measurements, that the entire Python-callable workflow is ferociously fast.
