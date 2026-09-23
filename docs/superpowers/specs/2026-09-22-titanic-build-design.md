# Build Design: Titanic Take-Home (delta spec)

Date: 2026-09-22

## Why this document is short

The repository already has a full specification spread over four documents:

- `docs/ARCHITECTURE.md`: module contracts, artifact schema, model definitions, UI spec
- `docs/API.md`: service layer, endpoints, metrics, queue model
- `PLAN.md`: phase-by-phase execution plan with time boxes and cut order
- `docs/DECISIONS.md`: the rationale behind every non-trivial choice

Those documents are the design. This file only records what the pre-implementation
interview and the environment probe changed or added, so that the two never conflict.
Anything not listed here follows the existing docs as written.

## Scope confirmed

Full documented scope, including the optional pieces:

- Four models: `fast` (TitanicLinear), `deep` (TitanicMLP), `attn` (TitanicAttention),
  `gbdt` (HistGradientBoostingClassifier)
- Service layer (`metrics.py`, `service.py`) built before the Streamlit app
- FastAPI adapter (`api/`) plus `scripts/load_test.py`
- Streamlit app with all six tabs including Ops

Per `PLAN.md`, `attn` is still the first model to cut if time gets tight.

## Deltas from the existing docs

### 1. Python 3.14 replaces the 3.12 mandate

`CLAUDE.md §6` and `PLAN.md` Phase 0 require Python 3.12 because of torch wheel
availability on Windows. On this machine, 3.14 resolved the full stack
(`torch 2.14.0+cpu`, `streamlit 1.64.0`, `fastapi 0.141.1`) with no problems. The project
builds on 3.14; `pyproject.toml` declares `requires-python = ">=3.11"` so reviewers on
3.11 or 3.12 are unaffected.

### 2. Dependency pins are a hard constraint

Windows Smart App Control is enabled on the development machine and blocks
low-reputation native extension modules at import time. Three packages were affected and
are pinned below their latest release: `pandas==2.3.3`, `scipy==1.16.2`,
`scikit-learn==1.7.2`. `requirements.txt` explains this inline so the pins are not removed
later by someone tidying up. The full reasoning is in `docs/DECISIONS.md`.

### 3. Kaggle credentials: four accepted formats

The docs assume `~/.kaggle/kaggle.json`. Kaggle now also issues `KGAT_`-prefixed access
tokens, and the client does not read the `~/.kaggle/access_token` file its own setup
snippet writes. `data._load_kaggle_credentials` checks, in order: `KAGGLE_API_TOKEN`,
`~/.kaggle/access_token` (bridged into the env var), `KAGGLE_USERNAME`/`KAGGLE_KEY`,
`~/.kaggle/kaggle.json`. If none is present, it raises a single `KaggleAuthError` that lists
every remedy.

### 4. Documentation layout

`CLAUDE.md`, `PLAN.md` and `README.md` moved from `docs/` to the repository root, matching
`CLAUDE.md §4`. `ARCHITECTURE.md`, `API.md`, `DECISIONS.md` and `assignment.md` stay in
`docs/`.

## Working agreement

These decisions from the pre-implementation interview govern how the code is written:

| Topic | Agreement |
|---|---|
| Comment density | Google-style docstrings on every public function with Args/Returns/Raises, type hints throughout, and inline comments on every non-obvious line that explain why, instead of restating the code |
| Teaching artifact | `docs/CODE_WALKTHROUGH.md` grows with each phase, explaining every module line by line for the author's own understanding, so the source itself stays clean |
| Testing | Test-first (TDD) for `features.py`, `preprocessing.py`, `models.py`, `service.py`; tests written alongside for everything else |
| Checkpoints | Work stops after each `PLAN.md` phase for a code walkthrough before the next phase begins |
| Version control | `git init` at Phase 0; one conventional commit per completed phase |
| Screenshots | Captured automatically with Playwright against the running Streamlit app into `docs/screenshots/` |
| Language | English for all code, comments and documentation |

## Security note

`kaggle.txt` in the repository root holds a live Kaggle API token. It is listed in
`.gitignore` (along with `.kaggle/` and `*.token`) and its absence from `git status` was
verified before the first commit. The token is scheduled for revocation by its owner.
