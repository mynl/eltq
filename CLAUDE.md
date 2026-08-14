# Standing New Project Instructions / Operating Procedure

* Stack
    * Python (>= 3.13)
    * uv for venvs
    * pathlib.Path for all file manipulations
    * csv, parquet/feather, pandas data storage, not sqlite; prefer visible files to dbs unless real need for db.
    * toml or yaml for config files (one choice for project - though see next)
    * pyproject.toml (always toml, but configs can be yaml; just not a mixture of yaml/toml outside pyproject)
    * use hatchling build backend
    * flask / Boostrap web / fastAPI as appropriate

* Dev drive **IMPORTANT** I have a windows dev drive V:\dev. Use that for venvs (symlink into it). Don't want the massive venv churn in ai - it is a synology drive nas synced location.

* docs
    * Numpy style doc strings.
    * Balanced level of docstrings and comments.
    * Code must be maintainable and readable.

* UI: no buttons that change meaning with state (the infamous play/pause)
    * use separate, explicitly-labeled actions instead.
    * clean design (see ai/fiscus_project, ai/csv-viewer SPA, ai/great-servers, T:/worktrees/aggreate_api)

* Do not plan to use env variables, use config files

* src/PACKGE project layout

* Semantic versioning, start at 1.0.0; expect each feature bump 1.x.0. I don't want to get 1.0.101! That should likely be 5.20.0, see?

* Label: use [human-readable] type labels for project steps and deliverables. No A1, A2, but [front-end], or []

* Maintain README.md and, critically, CHANGELOG.md

* Git: you do a commit after each version bump. **One line** commit message, typically "[version number] we did this and that" and CHANGELOG then has [version number]: long description.

* **Diagnose / design / propose before editing source or tests.** Don't change code until told to proceed ("go ahead"). "Can you see the issue?" means explain, not fix.
    * plan docs - go into ./dev with names plan-feature-1.md etc. When plan is completed move it to dev/done.

* Environment is **PowerShell on Windows**. No `awk`/`sed`/`head`/`tail` (even
  via the Bash tool). Use `rg` + the Read/Edit/Write tools.

* Prefer explicit, documented recipes over magic / auto-install behavior.


* Keep rendered output tight — no gratuitous blank lines in blocks.

* US spelling throughout (prose, docstrings, comments, identifiers).

* ISO dates

* Talking to me / working with me:
    * AQIN - ask questions if needed
    * SWIM - see what I mean
    * **gummage** — is or would be perfection. From Chandler Bing i Friends. High praise: "that's gummage" = exactly right.
    * I often multitask. Wait for input.

* **IMPORTANT**
    * **Consistency**: check my requests are consistent - naming etc.
    * I can forget or not think - **CHALLENGE** if something does not seem like a good idea! I want you as a though-partner / design consultant. I may not always ask for exactly what I want...I am human after all!

* Other examples
    * All the code in ai folder is ai generated. You can look at the other projects to get an idea of how i like things to go.

## Working style (KISS · Ask · Yell)

- **Keep It Simple.** Prefer getting 90% of the ask the easy way to getting 100% via a
  hellacious complication. Difficulty is a signal to **stop and talk**, not to push
  through cleverly with a fragile workaround.
- **Yell when something proves hard.** When the obvious/simple approach hits a wall,
  surface the tradeoff loudly — don't silently engineer around it. Example: if one
  table can't satisfy mixed formatting needs, propose splitting it into two; don't
  contrive a hack to force one.
- **Ask short questions** when a decision is genuinely Steve's, rather than guessing.
- **When you have enough to act, act.** Don't re-litigate settled decisions or narrate
  options you won't pursue. Give a recommendation, not an exhaustive survey.

## Versioning & changelog

- **Bump the version on every capability change** (new feature, behavior change, fix).
  Keep all version strings in sync (e.g. `pyproject.toml` + `__init__.__version__`).
- **Keep a matching CHANGELOG** entry (Keep a Changelog / SemVer: Added / Changed /
  Fixed / Notes), dated with the current date. Patch bump for fixes/small features.
- Steve reviews the version + CHANGELOG as the record of what changed.

## Git & commits

- **Claude commits each version but Claude never pushes**
- **Commit messages match the existing `git log` style** — terse, e.g.
  `X.Y.Z short summary of what changed` (literally one line), ending with the Co-Authored-By trailer. The CHANGELOG provides the detail, with a new section per version.

## Data firewall & privacy
Be aware of privacy issues.

- **Real/sensitive data never enters the repo or version control.** Develop and test
  ONLY against synthetic fixtures committed under `fixtures/`.
- **Never read, copy, or echo real data** into the repo, logs, or conversation. Real
  inputs live OUTSIDE the repo under an env-var-configured path.
- `.gitignore` blocks data files (`*.csv` except fixtures, DB/store files, `.env`,
  local config). **Before any commit, sanity-check the staged file list for leakage.**
- **No network egress in core code.** Any future outbound fetch must be opt-in and
  pull public data IN only — never send private data out.
- **Web/servers bind localhost by default** (`127.0.0.1`, never `0.0.0.0`); make the
  bind configurable and require a token for any non-localhost bind.

## Code & project conventions

- **Match the surrounding code's style** — comment density, naming, idioms. New code
  should read like the code already there - subject to having a standard minimum level of comments to ensure readability. 
- **Reuse Steve's own packages and sibling projects** before reinventing; clone the
  shape of an existing sibling app rather than inventing a new architecture.
- **Type hints where they aid clarity.** Keep modules thin with clear separation of
  concerns (e.g. parsing separate from business logic separate from reporting).
- **Never hardcode absolute paths.** Use env vars / config; created lazily.
- **Config is the single source of truth** for locations and behavior; edits write to
  the config / app-state dir, never into the repo or dev tree.
- **Plans live in `./dev`; move them to `./dev/done` when complete.**
- **Define canonical orderings once** (e.g. a category order) and enforce centrally —
  not ad-hoc per view.
