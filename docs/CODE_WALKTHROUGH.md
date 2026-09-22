# CODE_WALKTHROUGH.md — every module, explained

A study companion to the source. For each module: **what it does**, **why it exists**, and a
line-by-line explanation of anything non-obvious. The code itself carries docstrings and
"why" comments; this document adds the longer reasoning that would be noise inside a source
file.

Read it top to bottom to understand the project, or jump to a module you are about to change.

**Contents**

- [Phase 0 — foundation](#phase-0--foundation)
  - [`pyproject.toml` and `requirements.txt`](#pyprojecttoml-and-requirementstxt)
  - [`src/titanic/utils.py`](#srctitanicutilspy)
  - [`src/titanic/config.py`](#srctitanicconfigpy)
  - [`src/titanic/data.py`](#srctitanicdatapy)
  - [`tests/test_data.py`](#teststest_datapy)

---

## Phase 0 — foundation

The goal of Phase 0 is that `python -m titanic.data --fetch` produces a validated
`data/train.csv`, and that everything downstream has one place to look for paths, seeds and
schema rules.

### `pyproject.toml` and `requirements.txt`

**What:** packaging metadata, lint/format configuration, and pinned dependencies.

**Why two files:** `requirements.txt` pins the exact runtime versions *and* carries the
PyTorch CPU index URL, which `pyproject.toml` cannot express. `pyproject.toml` describes the
package itself. Duplicating dependencies in both would create two sources of truth, so
`pyproject.toml` deliberately declares none.

Points worth understanding:

- **`--extra-index-url https://download.pytorch.org/whl/cpu` must stay line 1.** Without it
  pip resolves the CUDA build of torch — roughly 2.5 GB of GPU libraries that a CPU-only
  laptop cannot use.
- **`where = ["src"]` (src-layout).** The importable package lives in `src/titanic`, not
  `./titanic`. This guarantees that `import titanic` in a test resolves to the *installed*
  package. Without src-layout, a test run from the repository root silently imports the
  local folder, so a broken `pip install -e .` goes unnoticed until a reviewer clones the
  repo and nothing works.
- **`select = [..., "D"]`** turns missing docstrings into lint errors, which is how the
  documentation standard is enforced mechanically rather than by discipline.
- **`per-file-ignores` for `tests/*`** switches the docstring rules off in tests. A test
  named `test_rejects_missing_required_column` already states its intent; a docstring
  repeating it is noise.
- **The version pins are load-bearing on Windows.** See `docs/DECISIONS.md` — Smart App
  Control blocks the newest `pandas`, `scipy` and `scikit-learn` native modules at import
  time. Do not unpin them without re-testing on a machine with Smart App Control enabled.

### `src/titanic/utils.py`

**What:** seeding, logging and timing — the three things every other module needs.

**Why one module:** none of them belongs to a single domain concept, and scattering them
would mean three near-duplicate implementations.

#### `set_seed(seed, deterministic=True)`

Reproducibility is an explicit grading criterion, so the function seeds **four** sources of
randomness, not just torch:

| Source | What it affects |
|---|---|
| `os.environ["PYTHONHASHSEED"]` | Iteration order of sets and dicts |
| `random.seed` | scikit-learn internals that use the stdlib RNG |
| `np.random.seed` | Bootstrap resampling, train/test splits |
| `torch.manual_seed` | Weight initialisation, dropout masks, DataLoader shuffling |

Two details:

- **`import torch` is inside the function, not at module top.** `utils` is imported by
  lightweight consumers such as the Streamlit sidebar, and importing torch costs about a
  second. Deferring it keeps cheap imports cheap.
- **`use_deterministic_algorithms(True, warn_only=True)`.** A handful of torch operations
  have no deterministic kernel. `warn_only=True` degrades those to a warning instead of
  crashing training half-way through — the right trade-off when full determinism is a
  nice-to-have and a completed run is mandatory.

#### `get_logger(name, level)`

The project forbids bare `print` in library code. Logs carry a timestamp and module name,
can be silenced by the host process, and do not corrupt stdout when the API is serialising
JSON.

The `_LOGGING_CONFIGURED` module-level flag exists because `logging.basicConfig` attaches a
handler to the root logger every time it is called with a fresh configuration. Streamlit
re-executes the whole script on every interaction, so without the guard each rerun would add
another handler and every log line would print twice, then three times, then four.

#### `timer()`

A context manager that yields a dict filled in on exit:

```python
with timer() as t:
    model.predict(x)
print(t["ms"])
```

- **`perf_counter`, not `time.time`.** `perf_counter` is monotonic and immune to system
  clock adjustments — necessary for the per-stage latency metrics the service records.
- **The `finally` block.** Timing is recorded even when the wrapped block raises, so a
  failed request still reports how long it took before failing. Without `finally`, error
  latency would silently vanish from the metrics.

### `src/titanic/config.py`

**What:** typed configuration objects and the schema constants shared by every layer.

**Why:** two rules the project must never break — no hardcoded absolute paths, and one
source of truth for "what does a valid Titanic CSV look like".

#### The schema constants

`REQUIRED_COLUMNS`, `OPTIONAL_COLUMNS`, `TARGET_COLUMN`, `NUMERIC_COLUMNS` and
`TRAIN_BASE_RATE` are consumed by `data.validate_schema`, the Pydantic API models and the
Streamlit "expected schema" panel. Defining them once means the API and the app reject
exactly the same inputs with exactly the same messages — a stated requirement in
`docs/API.md`.

`TRAIN_BASE_RATE = 0.3838` is the survival rate of the full training set (342/891). The Ops
tab compares the live predicted positive rate against it as a cheap drift signal. It is a
constant rather than a runtime computation because inference must never need the training
data.

#### `Paths`

`root` defaults to `Path(__file__).resolve().parents[2]`. Counting from
`src/titanic/config.py`: `parents[0]` is `src/titanic`, `parents[1]` is `src`, `parents[2]`
is the repository root. Every other path is a property derived from `root`, so a test can
pass `Paths(root=tmp_path)` and redirect the entire project at a temporary directory.

`frozen=True` makes instances immutable and hashable — one function cannot corrupt shared
state for another by assigning to `paths.data`.

`field(default_factory=...)` rather than a plain default: a mutable default evaluated once
at class-definition time would bake in the path from whichever directory Python happened to
be started in.

#### `SplitConfig` and `TrainConfig`

Grouped dataclasses rather than loose keyword arguments, so a caller passes one object and a
new parameter does not require changing five function signatures. The docstrings carry the
*reasoning* for each default — for example why `batch_size=64` (about 11 optimisation steps
per epoch on 712 rows: enough gradient noise to regularise, few enough to stay fast).

### `src/titanic/data.py`

**What:** the only module that talks to Kaggle and the only module that decides whether a
dataframe is acceptable.

**Why that boundary matters:** because training and inference call the same
`validate_schema`, a CSV the app accepts is exactly a CSV training would have accepted.
Divergence there is a classic source of production bugs.

#### The two exception types

- `SchemaError(ValueError)` — the *data* is wrong. The API maps it to HTTP 422; the app
  renders it as an actionable message.
- `KaggleAuthError(RuntimeError)` — the *credentials* are wrong. A different type because
  the remedy is completely different: create a token, not fix a file.

Every message states what is wrong **and** how to fix it. `SchemaError("Missing required
columns: ['Pclass']...")` names the columns; a bare `"invalid input"` would not.

#### `_load_kaggle_credentials()`

Kaggle supports two credential formats and the client reads only some of them:

1. `KAGGLE_API_TOKEN` environment variable — the newer `KGAT_...` token.
2. `~/.kaggle/access_token` — the same token in a file. **The client does not read this
   file**, even though Kaggle's own setup snippet writes it, so the function loads it and
   sets the environment variable itself.
3. `KAGGLE_USERNAME` + `KAGGLE_KEY` — classic environment variables.
4. `~/.kaggle/kaggle.json` — classic file, which the client does read natively.

If all four are absent it raises one `KaggleAuthError` listing every remedy, including the
competition-rules acceptance step (downloads return 403 until you click accept) and the
`--data-path data/sample_train.csv` escape hatch. The function returns a description of
which source was used, purely so the log line can say so.

#### `fetch_from_kaggle(dest_dir, force=False)`

- Requests **only `train.csv`** via `competition_download_file`, never the competition zip.
  The assignment forbids `test.csv` and `gender_submission.csv`, and a single file is also
  faster.
- An existing file short-circuits the network call unless `force=True`, so repeated runs are
  instant and work offline.
- **`import kaggle` is inside the function.** Some versions of the kaggle package
  authenticate at *import* time, which would make merely importing `titanic.data` fail on a
  machine with no credentials — including for users who only ever pass `--data-path`.
- The broad `except Exception` is deliberate and immediately re-raised as `KaggleAuthError`
  with guidance. The kaggle client raises a wide variety of types; catching them
  individually would be a guessing game, and the user-facing remedy is the same in every
  case. `raise ... from exc` preserves the original traceback for the logs.

#### `validate_schema(df, require_target=False)`

Strict about missing required columns, permissive about extra ones. A user exporting from a
spreadsheet routinely carries extra columns along (those are logged and ignored), but must
never be allowed to run a model on the wrong features.

The numeric check is the subtle part:

```python
coerced = pd.to_numeric(df[col], errors="coerce")
became_nan = coerced.isna() & df[col].notna()
```

`to_numeric(errors="coerce")` turns anything unparseable into `NaN`. Comparing that against
the values which were *already* `NaN` separates two very different situations:

- **Legitimately missing** — `Age` has 177 blanks in the real dataset; the preprocessor
  imputes them.
- **Not a number** — someone typed `"twenty-two"`; that is a data error and must be rejected
  loudly, naming the column, the offending values and the row numbers.

The target check calls `.dropna()` first for the same reason: an unlabelled row is handled
downstream, but a label of `2` or `"yes"` is a data error.

#### `stratified_split(df, config)`

Stratifying on the label keeps the 38% survival rate identical in both halves. At n=179 an
unstratified split can move the class balance by several points and shift accuracy by 1–2
points through sampling noise alone — pure noise in the headline number.

Both halves get `reset_index(drop=True)` so downstream NumPy array positions line up with
dataframe rows. Forgetting this produces silent misalignment between predictions and labels
after any row-filtering operation.

`from sklearn.model_selection import train_test_split` is again a local import: it keeps
`import titanic.data` cheap for consumers that only need `validate_schema`.

#### `make_sample(df, n, seed)`

Builds the committed `data/sample_train.csv`. Stratified so it stays representative (38.0%
survived versus 38.38% in the full set), and **sorted by `PassengerId`** so the committed
file has a stable, reviewable diff instead of a random row order that churns on every
regeneration.

#### `main(argv)`

The CLI. It returns an exit code rather than calling `sys.exit` directly, so tests can call
`main([...])` and assert on the result.

The `except (KaggleAuthError, SchemaError, FileNotFoundError)` block logs the message and
returns 1 — **no traceback**. These are *handled* failures whose messages already tell the
user what to do; a stack trace would only bury the guidance. Unexpected exceptions are
deliberately left to propagate, because those are bugs and the traceback is the useful part.

### `tests/test_data.py`

17 tests over four areas: validation, loading, splitting and sampling.

Two are worth studying:

**`test_preserves_class_balance`** originally asserted a deviation below `0.02` and failed.
The cause was not a bug but arithmetic: with 20 validation rows and a 38% base rate, the
ideal 7.6 positives must round to a whole 8, which *is* a 0.02 deviation. The fix derives
the tolerance from row granularity — `0.5 / len(split)`, half a row — rather than nudging a
magic number until the test passes. A tolerance you cannot justify is a test that will
mislead you later.

**`test_sizes_and_disjointness`** asserts the two halves share no `PassengerId`. Row overlap
between training and validation is the single most damaging bug possible in this project,
and it would otherwise show up only as suspiciously good validation scores.
