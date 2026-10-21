# eltq

Adds to `V:\dev\CLAUDE.md` (house rules). Project detail only — see **Deltas** at the foot.

**Note the name.** The directory is `secondary-corrosion`; the crate, package, and product are
**`eltq`**. Use `eltq` in prose and code.

**Ferociously fast ELT secondary-uncertainty quantization.** Converts a portfolio event loss table
(ELT, ~1M rows) into two discrete, evenly spaced, rate-weighted loss-grid distributions:

- **raw** — each event contributes an atom of rate λ at its conditional mean μ;
- **secondary** — each event's conditional loss is modeled as a beta on [0, M] matching
  (μ, σ = σ_I + σ_C) and approximated by K equally weighted midpoint quantiles
  x_k = M·Q(α, β; (k+½)/K).

Both are mapped to the grid g_i = i·h by a linear deposit rule that **preserves mass and first
moment exactly.**

## The authoritative documents

- **`secondary-uncertainty-mathematical-specification.md` is the mathematical contract.** It
  governs. Code conforms to it, not the other way round; if an implementation shortcut disagrees
  with the spec, the spec wins and the discrepancy is a bug to report.
- Implementation plans: `dev/done/plan-implementation.md` (1.0.0–1.5.0) and
  `dev/done/plan-multiunit.md` (1.6.0–1.11.0 — joint multi-unit secondary uncertainty, unit terms,
  view totals).
- `README.md` carries the milestone table and the layout.

## Layout

```
crates/eltq-core/     the Rust engine — scalar and batched kernels, parallelism, own erf, real FFTs
crates/eltq-python/   PyO3 bindings
python/eltq/          the Python package, CSV route, SciPy fallback backend
fixtures/             golden fixtures
benchmarks/           performance suite
notebooks/            demo + exploration
dev/                  plans; dev/done/ once execution starts
```

Correctness is anchored by a **SciPy oracle plus golden fixtures and an acceptance suite**
(milestone 1.0.0). That reference path is the thing that makes the fast path trustworthy — keep it
working and keep it honest. An optimization that cannot be checked against the oracle is not
finished.

## Deltas from house rules

- **Overrides the stack default.** This is a **Rust + Python** project: a Cargo workspace with
  PyO3 bindings, not a pure-Python uv project. Both `cargo` and the Python tooling are in play.
- **Overrides single-sourced versioning.** The version appears in `Cargo.toml`,
  `pyproject.toml` and `python/eltq/__init__.py` (`__version__`), and all three must be bumped
  together (`Cargo.lock` and `uv.lock` follow on build). Currently 1.11.3.
- **Speed is a product requirement, not a nice-to-have** — "ferociously fast" is in the project's
  own description, the work has a `[ferocious]` milestone, and there is a benchmark suite.
  Performance regressions are defects. Equally, per the house KISS rule: raise it before trading
  large complexity for small speed.
- **Numerical exactness claims are load-bearing.** Mass and first-moment preservation are exact
  properties of the deposit rule, not approximations. Any change touching the deposit rule, the
  quantile construction, or the σ_I + σ_C total-sd rule (Burke 2016) must be checked against the
  spec and the oracle before it lands.
- No web face, so the house localhost-bind and UI rules do not apply.
