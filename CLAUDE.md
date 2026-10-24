# eltq

Adds to `V:\dev\CLAUDE.md` (house rules). Project detail only — see **Deltas** at the foot.

**Ferociously fast ELT secondary-uncertainty quantization.** Converts a portfolio event loss table
(ELT, ~1M rows) into two discrete, evenly spaced, rate-weighted loss-grid distributions:

- **raw** — each event contributes an atom of rate λ at its conditional mean μ;
- **secondary** — each event's conditional loss is modeled as a beta on [0, M] matching
  (μ, σ = σ_I + σ_C); the portfolio exceedance curve of those betas is built exactly on geometric
  nodes, each event's endpoint handled per event, and bucketed by the nearest rule (the survival
  method, 2.0.0). The K-point midpoint-quantile rule x_k = M·Q(α, β; (k+½)/K) remains selectable.
  Rows whose stated variance exceeds μ(M − μ) become two atoms (the Bernoulli maximizer).

Atoms and the raw grid use a linear deposit rule that **preserves mass and first moment
exactly**; the survival grid is exact in rate and reports its first-moment error. The exceedance
curve itself is public (`exceedance_curve`).

## The authoritative documents

- **`secondary-uncertainty-mathematical-specification.md` is the mathematical contract.** It
  governs. Code conforms to it, not the other way round; if an implementation shortcut disagrees
  with the spec, the spec wins and the discrepancy is a bug to report.
- Implementation plans: `dev/done/plan-implementation.md` (1.0.0–1.5.0) and
  `dev/done/plan-multiunit.md` (1.6.0–1.11.0 — joint multi-unit secondary uncertainty, unit terms,
  view totals).
- `README.md` carries the milestone table and the layout.

## Sources

**The repository, its methods and its tools rest only on published sources** (Steve, 2026-10-24).
The allowed set is the bibliography tags `AAA2018`, `Burke2016`, `Homer2017a` (marked uncertain),
`Taylor2015` and `Varghese2013` in `D:\Projects\Biblio\uber-library.bib`, and these archivum
full texts (`D:\archivum\ShardedFullText\<first 2 hex>\<hash>_...pdftotext.md`, listed by the first
eight hex digits):

| hash | year | source |
|---|---|---|
| B9293139 | 2013 | Varghese, Rau-Chaplin: secondary uncertainty, portfolio risk measures on multi/many core |
| 5B70BC34 | 2016 | Burke, Rau-Chaplin, Varghese: PML in catastrophe reinsurance portfolios |
| 0698532C | 2017 | Mitchell-Wallace, Jones, Hillier: natural catastrophe risk management and modelling |
| 40DD75B5 | 2009 | Evans: introduction to catastrophe models, working with output |
| 70A03346 | 2020 | Jewson, Barnes, Cusack: adjusting catastrophe model ensembles by importance sampling |
| 02F828ED | 2010 | A.M. Best: supplemental rating questionnaire |
| 47C88738 | 2023 | Born, Dumm, Johnson: epistemic uncertainty in catastrophe models |
| D4A9ADCF | 2022 | Jewson: uncertain hurricane climate change projections in catastrophe models |
| F5C7C0C1 | 1985 | Williamson: economic institutions of capitalism |
| 90701826 | 2015 | Gollini, Rougier: rapidly bounding exceedance probabilities of high aggregate losses |
| B67BB58F | 2009 | AIR Worldwide: earthquake model for Chile |
| 339F787F | 2023 | S&P: insurer risk-based capital adequacy, methodology and assumptions |
| 604BFB6D | 2015 | Chang, Clark, Manghnani: IAA risk book chapter 5, catastrophe risk |
| 7C55637D | 2015 | Parodi: pricing in general insurance |
| DE03D9B8 | 2015 | PRA general insurance stress test 2015, scenario specification |
| 0D87E159 | 2010 | S&P: refined methodology for insurer capital adequacy |
| 42B2222F | 2014 | Association of British Insurers: non-modelled risks |
| 58749A1A | 2017 | Obersteadt: NAIC enhanced P&C RBC formula, catastrophe risk charges |
| 267AEB4C | 2018 | AAA Extreme Events and Property Lines Committee: uses of catastrophe model output |
| 70BBF1C6 | 2004 | RMS: Northridge earthquake ten-year retrospective |
| 91D24AA1 | 2016 | Kirk, Krvavych: ENID loading |
| C0E83EB3 | 2020 | Aon: catastrophe risk tolerance study |
| EE24EDBD | 2007 | Fitch: Merna Reinsurance Ltd |
| A7EB5909 | 2013 | Niehorster: warming oceans, implications for the reinsurance industry |

Added 2026-10-24 for the additivity note (`dev/secondary-uncertainty-additivity.md`): `Meyers2003a`
(Meyers, Klinker, Lalonde, The aggregation and correlation of insurance exposure, CAS Forum 2003;
Burke's reference [2]) and Lancaster 1957 (Some properties of the bivariate normal distribution
considered in the form of a contingency table, Biometrika 44; no bibliography tag yet).

**Nothing in `D:\Projects\aggregate\secondary-uncertainty` may be used**: not read for method, not
cited, not paraphrased. Material that reaches the repository from anywhere else (a conversation,
a client file, a vendor summary, a memory note) is not a source; if a method or convention cannot
be traced to the set above or derived mathematically in the specification, it does not go in.
Every convention taken from a source is cited to it at the point of use, as the
$\sigma_I+\sigma_C$ rule is cited to Burke 2016 section III.E.

## Layout

```
crates/eltq-core/     the Rust engine — scalar and batched kernels, parallelism, own erf, real FFTs
crates/eltq-python/   PyO3 bindings
python/eltq/          the Python package, CSV route, SciPy fallback backend
fixtures/             golden fixtures
benchmarks/           performance suite
notebooks/            demo + exploration
dev/                  plans; dev/done/ once execution starts
.github/workflows/    CI: wheels, tests against the oracle, PyPI publishing
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
  together (`Cargo.lock` and `uv.lock` follow on build). Currently 2.0.0.
- **Speed is a product requirement, not a nice-to-have** — "ferociously fast" is in the project's
  own description, the work has a `[ferocious]` milestone, and there is a benchmark suite.
  Performance regressions are defects. Equally, per the house KISS rule: raise it before trading
  large complexity for small speed.
- **Numerical exactness claims are load-bearing.** Mass and first-moment preservation are exact
  properties of the deposit rule, not approximations. Any change touching the deposit rule, the
  quantile construction, or the σ_I + σ_C total-sd rule (Burke 2016) must be checked against the
  spec and the oracle before it lands.
- No web face, so the house localhost-bind and UI rules do not apply.
- **Claude commits, Steve pushes** (ruled 2026-10-24, revising the same day's ruling that Steve
  made every commit). Overrides the house rule that each commit is pushed as it lands. Claude
  commits each plan step and version bump as it lands, staged by path, and never runs
  `git push` or creates a tag; the report says what is waiting to be pushed. The repository is
  public and a push to `master` runs CI.
- **Pushing a tag publishes.** `.github/workflows/wheels.yml` publishes to PyPI on a pushed `v*`
  tag; the `pypi` GitHub environment, which requires Steve's approval, is a second gate. x86_64
  wheels target `x86-64-v3`; local builds use `target-cpu=native` from the machine-level
  `V:\dev\.cargo\config.toml`, never from this repository (`dev/done/plan-pypi-release.md`).
