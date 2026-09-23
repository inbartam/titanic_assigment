# CODE_WALKTHROUGH.md: every module, explained

A study companion to the source. For each module it covers what the module does, why it
exists, and a line-by-line explanation of anything non-obvious. The code itself carries docstrings and
"why" comments; this document adds the longer reasoning that would be noise inside a source
file.

Read it top to bottom to understand the project, or jump to a module you are about to change.

**Contents**

- [Phase 0: foundation](#phase-0-foundation)
  - [`pyproject.toml` and `requirements.txt`](#pyprojecttoml-and-requirementstxt)
  - [`src/titanic/utils.py`](#srctitanicutilspy)
  - [`src/titanic/config.py`](#srctitanicconfigpy)
  - [`src/titanic/data.py`](#srctitanicdatapy)
  - [`tests/test_data.py`](#teststest_datapy)
- [Phase 1: features and the preprocessor](#phase-1-features-and-the-preprocessor)
  - [`src/titanic/features.py`](#srctitanicfeaturespy)
  - [`src/titanic/preprocessing.py`](#srctitanicpreprocessingpy)
  - [What the fitted values actually look like](#what-the-fitted-values-actually-look-like)
- [Phase 2: the EDA notebook](#phase-2-the-eda-notebook)
  - [How the notebook is built](#how-the-notebook-is-built)
  - [Split before looking](#split-before-looking)
  - [What the analysis actually found](#what-the-analysis-actually-found)
  - [`tests/test_notebook.py`](#teststest_notebookpy)
- [Phase 3: models, training and artifacts](#phase-3-models-training-and-artifacts)
- [Phase 4: evaluation and plots](#phase-4-evaluation-and-plots)
- [Phase 5: the service, the API and the app](#phase-5-the-service-the-api-and-the-app)
- [Phase 6: what the finished system actually measured](#phase-6-what-the-finished-system-actually-measured)

---

## Phase 0: foundation

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
  pip resolves the CUDA build of torch, which is roughly 2.5 GB of GPU libraries that a
  CPU-only laptop cannot use.
- **`where = ["src"]` (src-layout).** The importable package lives in `src/titanic`, not
  `./titanic`. This guarantees that `import titanic` in a test resolves to the installed
  package. Without src-layout, a test run from the repository root silently imports the
  local folder, so a broken `pip install -e .` goes unnoticed until a reviewer clones the
  repo and nothing works.
- **`select = [..., "D"]`** turns missing docstrings into lint errors, which is how the
  documentation standard is enforced by the linter instead of by discipline.
- **`per-file-ignores` for `tests/*`** switches the docstring rules off in tests. A test
  named `test_rejects_missing_required_column` already states its intent; a docstring
  repeating it is noise.
- **The version pins matter on Windows.** As `docs/DECISIONS.md` explains, Smart App
  Control blocks the newest `pandas`, `scipy` and `scikit-learn` native modules at import
  time. Do not unpin them without re-testing on a machine with Smart App Control enabled.

### `src/titanic/utils.py`

**What:** seeding, logging and timing, which every other module needs.

**Why one module:** none of them belongs to a single domain concept, and scattering them
would mean three near-duplicate implementations.

#### `set_seed(seed, deterministic=True)`

Reproducibility is an explicit grading criterion, so the function seeds four sources of
randomness rather than just torch:

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
  crashing training half-way through. Full determinism is a nice-to-have here, while a
  completed run is mandatory, so that seemed the right trade-off.

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

- **`perf_counter` instead of `time.time`.** `perf_counter` is monotonic and unaffected by
  system clock adjustments, which the per-stage latency metrics in the service depend on.
- **The `finally` block.** Timing is recorded even when the wrapped block raises, so a
  failed request still reports how long it took before failing. Without `finally`, error
  latency would quietly disappear from the metrics.

### `src/titanic/config.py`

**What:** typed configuration objects and the schema constants shared by every layer.

**Why:** the project has two rules here: no hardcoded absolute paths, and a single source of
truth for "what does a valid Titanic CSV look like".

#### The schema constants

`REQUIRED_COLUMNS`, `OPTIONAL_COLUMNS`, `TARGET_COLUMN`, `NUMERIC_COLUMNS` and
`TRAIN_BASE_RATE` are consumed by `data.validate_schema`, the Pydantic API models and the
Streamlit "expected schema" panel. Defining them once means the API and the app reject
exactly the same inputs with exactly the same messages, which `docs/API.md` states as a
requirement.

`TRAIN_BASE_RATE = 0.3838` is the survival rate of the full training set (342/891). The Ops
tab compares the live predicted positive rate against it as a cheap drift signal. It is a
constant rather than a runtime computation because inference must never need the training
data.

#### `Paths`

`root` defaults to `Path(__file__).resolve().parents[2]`. Counting from
`src/titanic/config.py`: `parents[0]` is `src/titanic`, `parents[1]` is `src`, `parents[2]`
is the repository root. Every other path is a property derived from `root`, so a test can
pass `Paths(root=tmp_path)` and redirect the entire project at a temporary directory.

`frozen=True` makes instances immutable and hashable, so one function cannot corrupt shared
state for another by assigning to `paths.data`.

`field(default_factory=...)` rather than a plain default: a mutable default evaluated once
at class-definition time would bake in the path from whichever directory Python happened to
be started in.

#### `SplitConfig` and `TrainConfig`

Grouped dataclasses rather than loose keyword arguments, so a caller passes one object and a
new parameter does not require changing five function signatures. The docstrings carry the
reasoning for each default. For example, `batch_size=64` gives about 11 optimisation steps
per epoch on 712 rows: enough gradient noise to regularise, few enough to stay fast.

### `src/titanic/data.py`

**What:** the only module that talks to Kaggle and the only module that decides whether a
dataframe is acceptable.

**Why that boundary matters:** training and inference call the same `validate_schema`, so a
CSV the app accepts is exactly a CSV training would have accepted. When those two drift apart
you get a familiar kind of production bug.

#### The two exception types

- `SchemaError(ValueError)`: the data is wrong. The API maps it to HTTP 422; the app
  renders it as an actionable message.
- `KaggleAuthError(RuntimeError)`: the credentials are wrong. It is a separate type because
  the remedy is different (create a token instead of fixing a file).

Every message states what is wrong and how to fix it. `SchemaError("Missing required
columns: ['Pclass']...")` names the columns; a bare `"invalid input"` would not.

#### `_load_kaggle_credentials()`

Kaggle supports two credential formats and the client reads only some of them:

1. `KAGGLE_API_TOKEN` environment variable: the newer `KGAT_...` token.
2. `~/.kaggle/access_token`: the same token in a file. **The client does not read this
   file**, even though Kaggle's own setup snippet writes it, so the function loads it and
   sets the environment variable itself.
3. `KAGGLE_USERNAME` + `KAGGLE_KEY`: classic environment variables.
4. `~/.kaggle/kaggle.json`: classic file, which the client does read natively.

If all four are absent it raises one `KaggleAuthError` listing every remedy, including the
competition-rules acceptance step (downloads return 403 until you click accept) and the
`--data-path data/sample_train.csv` escape hatch. The function returns a description of
which source was used, purely so the log line can say so.

#### `fetch_from_kaggle(dest_dir, force=False)`

- Requests only `train.csv` via `competition_download_file`, never the competition zip.
  The assignment forbids `test.csv` and `gender_submission.csv`, and a single file is also
  faster.
- An existing file short-circuits the network call unless `force=True`, so repeated runs are
  instant and work offline.
- **`import kaggle` is inside the function.** Some versions of the kaggle package
  authenticate at import time, which would make merely importing `titanic.data` fail on a
  machine with no credentials, including for users who only ever pass `--data-path`.
- The broad `except Exception` is deliberate and immediately re-raised as `KaggleAuthError`
  with guidance. The kaggle client raises many different exception types. Catching them
  individually would mean guessing, and the user-facing remedy is the same in every case. `raise ... from exc` preserves the original traceback for the logs.

#### `validate_schema(df, require_target=False)`

Strict about missing required columns, permissive about extra ones. A user exporting from a
spreadsheet often carries extra columns along (those are logged and ignored), but should not
be able to run a model on the wrong features.

The numeric check is the subtle part:

```python
coerced = pd.to_numeric(df[col], errors="coerce")
became_nan = coerced.isna() & df[col].notna()
```

`to_numeric(errors="coerce")` turns anything unparseable into `NaN`. Comparing that against
the values which were already `NaN` separates two different situations:

- **Legitimately missing:** `Age` has 177 blanks in the real dataset, and the preprocessor
  imputes them.
- **Not a number:** someone typed `"twenty-two"`. That is a data error and gets rejected with
  a message naming the column, the offending values and the row numbers.

The target check calls `.dropna()` first for the same reason: an unlabelled row is handled
downstream, but a label of `2` or `"yes"` is a data error.

#### `stratified_split(df, config)`

Stratifying on the label keeps the 38% survival rate identical in both halves. At n=179 an
unstratified split can move the class balance by several points and shift accuracy by 1 to 2
points through sampling alone, which is noise in the headline number.

Both halves get `reset_index(drop=True)` so downstream NumPy array positions line up with
dataframe rows. Forgetting this leads to a silent misalignment between predictions and labels
after any row-filtering operation.

`from sklearn.model_selection import train_test_split` is again a local import: it keeps
`import titanic.data` cheap for consumers that only need `validate_schema`.

#### `make_sample(df, n, seed)`

Builds the committed `data/sample_train.csv`. Stratified so it stays representative (38.0%
survived versus 38.38% in the full set), and sorted by `PassengerId` so the committed
file has a stable, reviewable diff instead of a random row order that churns on every
regeneration.

#### `main(argv)`

The CLI. It returns an exit code rather than calling `sys.exit` directly, so tests can call
`main([...])` and assert on the result.

The `except (KaggleAuthError, SchemaError, FileNotFoundError)` block logs the message and
returns 1 without a traceback. These are handled failures whose messages already tell the
user what to do, and a stack trace would only bury the guidance. Unexpected exceptions are
left to propagate on purpose, because those are bugs and the traceback is the useful part.

### `tests/test_data.py`

17 tests over four areas: validation, loading, splitting and sampling.

Two of them are worth a closer look:

**`test_preserves_class_balance`** originally asserted a deviation below `0.02` and failed.
The cause was not a bug but arithmetic: with 20 validation rows and a 38% base rate, the
ideal 7.6 positives must round to a whole 8, which is a 0.02 deviation. The fix derives
the tolerance from row granularity (`0.5 / len(split)`, half a row) instead of nudging a
magic number until the test passes. A tolerance you cannot justify tends to mislead you
later.

**`test_sizes_and_disjointness`** asserts the two halves share no `PassengerId`. Row overlap
between training and validation would be about the most damaging bug this project could
have, and otherwise it would only show up as suspiciously good validation scores.


---

## Phase 1: features and the preprocessor

Phase 1 splits one job in two. Stateless transformations live in `features.py`, and fitted
ones live in `preprocessing.py`. Anything that has to learn a value from the training data (a
median, a mean, a vocabulary) belongs on the fitted side, because that is the code that can
leak.

### `src/titanic/features.py`

**What:** pure functions turning raw Kaggle columns into modelling features.

**Why pure:** the contract tested in `tests/test_features.py` is that every feature is
computable from a single row. `test_row_features_are_independent_of_the_batch` engineers
one row on its own and asserts it matches the same row engineered inside the full frame. That
test is what keeps `TicketGroupSize` and `FarePerPerson` out of the codebase. Both are counts
over the batch, so at inference on one passenger they are always 1, which is not the value the
model trained on.

#### `extract_title(names)`

The Kaggle name format is `"Surname, Title. Given Names"`, so the title is the text between
the comma and the first period:

```python
_TITLE_AFTER_COMMA = r",\s*([^.]+)\."
_TITLE_AT_START    = r"^\s*([A-Za-z]+)\."
```

The second pattern exists because not every name follows that format. `"Mme. Something"` has
no comma and would silently become `Rare`. Only rows the primary pattern missed are retried
with the looser one, so the fallback cannot reinterpret a well-formed name:

```python
raw = raw.where(raw.notna(), fallback)
```

`.where` is used instead of `.fillna` because `fillna` on an object-dtype column raises a pandas
downcasting `FutureWarning`. `.where` also states the intent more directly: keep the primary
match, otherwise take the fallback.

Then `.str.title()` normalises case so `"MR."` and `"mr."` both become `Mr` and the vocabulary
does not split on capitalisation, aliases fold `Mlle`/`Ms` into `Miss` and `Mme` into `Mrs`,
and anything left outside the four common titles becomes `Rare`. The function does not return
missing values. An unparseable name yields `Rare` instead of raising, because inference on messy
user data should degrade gracefully instead of crashing.

#### The smaller functions

- **`family_size`**: `SibSp + Parch + 1`. Survival is non-monotonic in this value (families of
  2 to 4 did best, solo travellers and very large families worst), which is why the raw counts
  are dropped in favour of the total.
- **`is_alone`**: kept as an explicit binary even though it is derivable from `FamilySize`,
  because the survival drop at exactly size 1 is a step, and a linear model cannot represent
  a step from one continuous input.
- **`deck_from_cabin`**: first letter, `U` when missing. It uses `str[:1]` instead of `str[0]`
  because that returns `""` for an empty string instead of raising.
- **`log_fare`**: `log1p` instead of `log`, because a fare of exactly 0 appears in the data and
  `log(0)` is undefined. Missing fares stay missing on purpose, since imputing them needs a
  fitted value and that belongs to the preprocessor.

#### `engineer(df)`

The single entry point used by training, inference and the notebook, so the notebook can never
drift from the feature logic the model consumes. It copies the input (callers reuse the raw
frame for display), then materialises absent `Cabin`/`Embarked` columns as NaN so that an
absent column and an all-NaN column behave identically, following the rule in
`docs/ARCHITECTURE.md` section 2.

### `src/titanic/preprocessing.py`

**What:** the class that learns imputation values, scaling statistics and category
vocabularies from the training split, and serialises them to JSON.

**Why it matters most:** this is where leakage would enter. If any fitted value were computed
over data the model is later evaluated on, every number in the README would be optimistic and
no other test would notice.

#### The leakage guard

`tests/test_preprocessing.py::test_transforming_validation_data_does_not_change_fitted_state`
serialises the entire fitted state, transforms the validation split, serialises again, and
asserts the two strings are identical. Because it compares the whole state instead of a few
named fields, a new fitted parameter added later is covered automatically.

You can also see the guard working in the real numbers: after transforming, the training split
has numeric mean exactly 0 and std exactly 1, while the validation split has mean
`[-0.047, 0.058, 0.066]`. If the validation columns also came out at 0 and 1, the scaler would
have been fitted on them. That asymmetry is what correct behaviour looks like here.

#### Order of operations in `transform`

The order is fixed, and each step depends on the previous one:

1. Impute `Embarked` with the fitted mode.
2. Impute `Fare` with the fitted median.
3. Recompute `LogFare` from the imputed fare.
4. Impute `Age` from the title median, falling back to the global median.
5. Standardise numerics, index-encode categoricals.

Step 3 is the easy one to get wrong. Imputing `LogFare` directly would apply a median taken on
the wrong scale, since `log1p(median(fare))` is not `median(log1p(fare))`. Recomputing from the
imputed raw fare is the correct order, and `test_missing_fare_is_imputed_before_log` pins it
down by blanking both columns and asserting nothing comes back NaN.

#### Why `fit` imputes before computing scaling statistics

```python
imputed = self._impute(df)
for column in self.numeric_cols:
    self.num_mean[column] = float(np.nanmean(values))
```

The statistics must describe exactly the values `transform` will later standardise. Computing
the mean on raw data full of holes would describe only the observed subset and bias the
result. `_impute` is shared by `fit` and `transform` for the same reason: two copies of this
logic would eventually disagree.

#### `<UNK>` at index 0

Every vocabulary reserves index 0 for `UNKNOWN_TOKEN`. Unseen categories at inference map
there instead of raising:

```python
keys = imputed[column].map(self._as_key)
categorical[:, position] = keys.map(vocabulary).fillna(0).astype(np.int64).to_numpy()
```

`.map()` leaves unrecognised levels as NaN, and `fillna(0)` sends them to the reserved slot,
which the embedding layer has a real weight row for. A user CSV with a deck letter that never
appeared in training produces a prediction, not a stack trace.

#### `_as_key` and the JSON key problem

JSON object keys must be strings, but categories arrive as a mix of types: `Pclass` is an int,
`Sex` a string, `IsAlone` a numpy int. Everything is stringified on both the fit and the
transform path, so the mapping survives the round trip. The `is_integer()` branch handles a
specific pandas behaviour: a column widens to float when any value is missing, which would
otherwise make `3` and `3.0` two different levels.

#### Persistence

`save` writes `indent=2, sort_keys=True` so artifact diffs stay reviewable, with
`encoding="utf-8"` stated explicitly because Windows would otherwise default to cp1252. `load`
rejects an unrecognised `version` with a message naming the fix rather than misreading an old
artifact. `test_load_rejects_an_unknown_schema_version` covers this.

The `NotFittedError` is defined in this module rather than imported from scikit-learn, so the
project owns its own exception hierarchy: the service layer maps its typed exceptions to HTTP
codes and should not depend on sklearn's tree.

### What the fitted values actually look like

Fitted on the real 712-row training split:

```
Age median by Title            Cardinalities (incl. <UNK>)
  Master      3.0                Pclass      4   <UNK> 1 2 3
  Miss       22.0                Sex         3   <UNK> female male
  Mr         30.0                Embarked    4   <UNK> C Q S
  Mrs        35.0                Title       6   <UNK> Master Miss Mr Mrs Rare
  Rare       49.0                Deck       10   <UNK> A B C D E F G T U
  GLOBAL     28.5                IsAlone     3   <UNK> 0 1

fare_median = 14.4542    embarked_mode = S
```

`Master` imputed at 3.0 against a global median of 28.5 sums up the case for title-based
imputation. Roughly 20% of ages are missing. Imputing them globally would have turned every boy
with an unrecorded age into a 28-year-old man and weakened the "children first" signal, which is
the second strongest effect in the dataset after sex.


---

## Phase 2: the EDA notebook

`notebooks/eda.ipynb` is the assignment's "exploratory data analysis in a Jupyter Notebook"
deliverable. It is organised as eleven sections, each a Question → Analysis → Finding →
Decision block, with seven figures. Every decision it reaches is one that
`src/titanic/features.py` or `src/titanic/preprocessing.py` actually implements.

### How the notebook is built

`notebooks/build_eda.py` generates the `.ipynb`; `jupyter nbconvert --execute --inplace` runs
it and stores the outputs. Two reasons:

- **Notebook diffs are hard to read.** `.ipynb` is JSON, so changing one word in a markdown cell
  produces a diff tangled up with base64 PNG blobs. The generator is plain Python and reviews
  like plain Python.
- **It is hard to leave half-executed.** Regenerating and re-running is one command, so the
  committed notebook is always a complete, top-to-bottom run.

The committed notebook is still an ordinary notebook: Restart & Run All works, and the outputs
are saved so it renders on GitHub for a reviewer who has neither the dataset nor Kaggle
credentials. The first cell falls back to `data/sample_train.csv` with a printed warning if
`data/train.csv` has not been fetched.

### Split before looking

The first analysis cell does this:

```python
train_raw, val_raw = stratified_split(raw, SplitConfig())
df = engineer(train_raw)
del val_raw
```

`del val_raw` is the point of the cell. Everything below explores 712 rows, and the 179
held-out rows are never plotted or summarised.

This matters more than it might seem. Every decision the notebook reaches (impute Age by Title,
treat Pclass as categorical, drop SibSp/Parch, exclude ticket-group features) is a modelling
choice made because of what a plot showed. If those plots included the validation rows, the
final "single unbiased evaluation" would be scoring a pipeline that was partly designed on the
data it is being scored against. The leak would come from the analyst rather than the code.

### What the analysis actually found

Numbers below are from the committed run on the real 712-row training split.

| Section | Finding | Decision it drove |
|---|---|---|
| Target balance | 38.3% survived; a "nobody survived" model scores 0.617 accuracy | Report PR-AUC and F1 alongside accuracy, all with bootstrap CIs |
| Missingness | Cabin 77%, Age 20%, Embarked 2 rows | Three different treatments instead of one blanket imputation |
| Duplicates / leakage | 0 duplicate rows; 33.1% share a ticket | Exclude batch-dependent features (see below) |
| Sex × Pclass | Effects are not additive; Pclass spacing is not linear | Pclass as categorical; train a model ladder to measure the interaction |
| Age by Title | Master ≈ 3, Mr ≈ 30, global ≈ 28.5; under-10s survived at 0.640 vs 0.383 overall | Impute Age by Title median |
| Fare | 35.4× max/median ratio; 14 fares of exactly 0 | `log1p` (not `log`); keep outliers |
| Family size | Peaks at 2 to 4, collapses at both ends | Use FamilySize; keep IsAlone as an explicit step |
| Correlation | SibSp/Parch redundant once FamilySize exists | Final 9-feature set |
| 5-fold CV | Expectation band ROC-AUC 0.86-0.89; fold spread 0.080 | Tiny hyperparameter grids; CIs on everything |

A few outputs deserve a closer look.

**The ticket-group demonstration.** Section 4 prints this:

```
TicketGroupSize computed on a single-row request: 1
The same passenger's true value in the training batch: 6
```

Those two lines make the case against the feature better than a paragraph of theory would.
The same passenger gets a different feature value depending on who else happens to be in the
file.

**The cross-validation is itself leak-free.** The preprocessor is refitted inside every fold:

```python
for fold_train_idx, fold_val_idx in folds.split(df, y):
    pre = Preprocessor().fit(fold_train)     # refit per fold, not once outside
    xtr = np.hstack(pre.transform(fold_train))
    xva = np.hstack(pre.transform(fold_val))
```

Fitting the preprocessor once outside the loop is a common and easy-to-miss mistake in
cross-validation code: each fold's held-out rows would contribute to the imputation medians and
scaling statistics. The effect is small on this dataset, but it is exactly the kind of error
this project tries to avoid, so the notebook shows the correct pattern.

**The band, and what it is for.** Both classical models land near ROC-AUC 0.86-0.89, and the
fold-to-fold spread (0.080) is wider than the gap between the two models. So a PyTorch model
scoring far below that band probably has a bug rather than a modelling problem, and one scoring
far above it has probably leaked. The band is a sanity check for Phase 3, not a target.

### `tests/test_notebook.py`

Six static checks over the notebook JSON. They parse it without executing it, so the suite
stays fast. They exist because `PLAN.md`'s risk register names notebook drift as a specific
risk:

- `test_notebook_imports_feature_logic_instead_of_redefining_it` fails if the notebook contains
  `def extract_title` or any other feature function. The notebook must import from
  `titanic.features`.
- `test_engineered_columns_cover_what_the_preprocessor_needs` asserts every modelled column is
  either a raw Kaggle column or produced by `engineer()`.
- `test_notebook_never_touches_the_forbidden_files` greps for `test.csv` and
  `gender_submission.csv`.
- `test_notebook_discards_the_validation_split` asserts `del val_raw` is still there, turning
  the discipline described above into a tested invariant instead of a promise in markdown.
- `test_notebook_ran_without_errors` fails if any cell has a stored traceback. A committed
  notebook with a visible exception is worse than no notebook.
- `test_notebook_outputs_are_committed` fails if the notebook was committed unexecuted, which
  would leave a reviewer with a blank document.


---

## Phase 3: models, training and artifacts

### `src/titanic/models.py`

Three architectures, one forward signature:

```python
forward(x_num: FloatTensor[B, 3], x_cat: LongTensor[B, 6]) -> FloatTensor[B]   # logits
```

**Logits instead of probabilities.** That lets the training loop use `BCEWithLogitsLoss`, which
folds the sigmoid into the loss in a numerically stable way. At extreme logits, a separate
sigmoid followed by BCE saturates and loses gradient. Callers apply `torch.sigmoid` once, at inference.

**`squeeze(-1)` instead of `squeeze()`.** Every model ends with it. A `(B, 1)` output compared
against a `(B,)` target broadcasts into a `(B, B)` loss matrix, and the model still trains
(badly) without any error. `squeeze(-1)` only removes the last dimension, so a batch of one stays
`(1,)` instead of collapsing to a scalar.

**`nn.ModuleList` instead of a plain list.** A plain Python list of embeddings would be invisible
to `.parameters()`, so the optimiser would never update them and `.to(device)` would leave them
behind. `test_gradients_reach_every_parameter` catches that kind of wiring bug by asserting no
parameter has `grad is None` after a backward pass.

**`TitanicLinear` is logistic regression on purpose.** It has 34 parameters: one per one-hot
slot, one per numeric feature, plus a bias. Implementing it in PyTorch instead of using sklearn
means it shares the loop, loss, optimiser, batching and seed with the MLP, so any gap between
them comes from the architecture.

**`TitanicAttention`** gives each numeric feature its own `Linear(1, d)`: the token is the value
times a learned direction plus a learned offset. A shared projection would make every numeric
feature collinear in token space. `norm_first=True` (pre-LN) lets a 2-layer transformer train
stably at this scale without a warmup schedule, and `enable_nested_tensor=False` is stated
explicitly because the fast path it controls applies only to padded variable-length sequences,
which these fixed 10-token rows are not.

### `src/titanic/training.py`

**`EarlyStopping` deep-copies the best weights.** Keeping a reference would alias the live
parameters, so the "best" snapshot would keep changing as training continued and restoring it
would do nothing. Without `restore_best`, a model that overfits after epoch 40 gets saved in its
overfitted state even though training correctly stopped at 60.

**The carve-out is stratified.** With a 38% positive rate, a random 10% of 712 rows can easily
land at 30% or 46% positive, making the stopping signal noisy for reasons that have nothing to
do with the model.

**`_run_epoch` serves both passes.** `torch.enable_grad()` or `torch.no_grad()` is chosen by
whether an optimiser was passed, so there is one loop instead of two that can drift apart. Losses are
weighted by batch size because the final batch is usually smaller and an unweighted mean would
over-count it.

**`cross_validate` refits the preprocessor inside every fold.** Fitting once outside the loop is
a common CV bug that is easy to miss, because each fold's held-out rows would contribute to the
imputation medians and scaling statistics. It also re-seeds per fold, so fold-to-fold variance
measures data variance rather than initialisation noise.

### `src/titanic/artifacts.py`

**`state_dict` instead of a pickled module.** A pickled module embeds the class path, so renaming
or moving a class breaks every previously saved artifact. `build_model()` reconstructs the
architecture from `model_config.json` and then loads the weights. That also means the config
must be enough to rebuild the model, which `test_round_trips_through_its_own_config` checks.

**`weights_only=True`** on `torch.load` refuses to execute arbitrary pickled code while reading
a file from disk.

**`Bundle.predict_proba` hides the framework.** Because of it, nothing in `app/` or `api/`
branches on torch versus sklearn.

**`gbdt/model.joblib` is the one non-JSON artifact.** scikit-learn has no clean JSON
serialisation. The version that wrote it is recorded, and a mismatch warns rather than refusing,
because a joblib artifact usually loads across minor versions and refusing would make the
committed bundle useless to a reviewer with a slightly different install.

### `train.py`

The whole pipeline in one file, guaranteeing two invariants: the validation split is touched
exactly once per model, and everything inference needs is written to disk.

Its `except Exception` around each model is deliberate. One model failing should not throw away
the models already trained, since the registry tolerates a partial set and the app renders it.

Two bugs turned up here:

- **`"%9,d"` is not valid printf.** Thousands separators are f-string grammar; the logging call
  raised `ValueError: unsupported format character ','` only when the summary table printed, at
  the very end of an otherwise successful run.
- **The registry stored `"dir": "artifacts/<name>"`**, which broke under `--artifacts-dir`
  because the path resolved against the artifacts directory's parent. Now stored relative to
  `registry.json` itself.

## Phase 4: evaluation and plots

`evaluation.py` produces numbers and arrays; `plots.py` turns them into figures. That separation
is what lets `train.py` write HTML files and the app render interactive charts from one
implementation.

**The bootstrap is stratified.** Positives and negatives are resampled separately to their
original counts, so every resample has the same class balance as the real evaluation set. An
unstratified resample of 179 rows occasionally produces a single-class sample for which ROC-AUC
is undefined, silently shrinking the sample the interval is computed from.

**`average_precision_score` instead of the trapezoid under the PR curve.** PR curves are not
monotonic, so trapezoidal interpolation systematically overestimates. Average precision sums the rectangles
exactly.

**Why Brier score is included.** `test_brier_rewards_calibration_not_ranking` shows the reason:
predictions of `[0.01, 0.02, 0.98, 0.99]` and `[0.45, 0.46, 0.54, 0.55]` both rank perfectly and
both score ROC-AUC 1.0, but only Brier notices that one is confident and the other is guessing.

**Plot details.** ROC gets `scaleanchor="x"` because a stretched ROC curve is misleading. The PR
baseline is drawn at the base rate instead of 0.5, because that is what no-skill means on
imbalanced data. `metrics_comparison_fig` zooms the y-axis to the region the bars occupy. On a
full 0-to-1 axis the 0.019 spread between these four models would be invisible, and that spread
is what the reader needs to judge. Calibration marker size encodes the bin count,
so a point resting on three passengers is visibly less trustworthy than one resting on fifty.

## Phase 5: the service, the API and the app

### `src/titanic/service.py`

The single object that touches a model at serve time.

**Queue depth is the main thing to get right.** `_acquire_slot` increments a waiting counter,
tries to acquire the semaphore, then decrements it in both branches:

```python
with self._counter_lock:
    self._waiting -= 1              # rejected requests are no longer waiting either
    self.metrics.set_queue_depth(self._waiting)
    if acquired:
        self._inflight += 1
```

Leaking that counter on the rejection path would permanently inflate the gauge the Ops tab
presents as an autoscaling signal. The distinction matters because in-flight saturates at
`max_concurrency` the instant the service is busy and tells you nothing more, while depth keeps
climbing and tells you how far behind you are. The load test shows it: in-flight capped at 2
while depth reached 13.

**The `finally` around the critical section** is what stops the service deadlocking after
`max_concurrency` failures. `test_slot_is_released_when_inference_raises` fails three
predictions in a row and then asserts a fourth still succeeds.

**Timers wrap each stage separately** because "the model is slow" and "preprocessing is slow"
need different fixes. Under load, the answer turned out to be neither.

### `src/titanic/metrics.py`

Two readers of one recording. `/metrics` serves Prometheus text for a real scrape target;
`/stats` serves exact p50/p95/p99 over a 2000-record `deque` because the app needs precise
recent percentiles without anyone running a Prometheus server.

Each registry owns a private `CollectorRegistry`. Using the process-global default would
raise a duplicate-timeseries error the second time a test built a service.

The probability histogram samples the first 200 rows per request: a 10 000-row request would
otherwise dominate the distribution and cost more to record than the inference itself.

### `api/main.py`

A thin adapter. Routes parse, call the service, and let one `ERROR_MAP` turn typed exceptions
into status codes, so no route handler contains a status code.

**`limiter.total_tokens = max_concurrency + max_queue + 8`** is easy to overlook but necessary. anyio's
default thread limiter is 40 threads; if it were smaller than what the service accepts, the
threadpool would become a hidden second queue and the queue-depth metric would stop describing
reality.

**A bug the tests caught:** `/admin/reload` used `Depends(get_settings)`, which re-reads the
environment instead of the settings the app was constructed with, so an app built with an
admin token still behaved as though reloading were disabled. It now reads
`request.app.state.settings`.

### `app/`

`client.py` defines a `Predictor` protocol with two implementations, and `ApiPredictor`
translates HTTP error bodies back into the project's own exception types so the app's error
handling is written once against one hierarchy.

`build_predictor` probes `/health` with a 2-second timeout and falls back to local mode with a
visible warning. The API is a bonus layer, and the app does not depend on it.

`state.py` uses `@st.cache_resource` for the predictor and `@st.cache_data` for dataframes. The
distinction matters because `cache_data` copies its return value, so caching the predictor that
way would fork the metrics registry and reset the Ops tab on every interaction.

`components.honest_verdict()` writes the comparison paragraph from the numbers: the leader, which
models fall inside its interval, the smallest model, and a recommendation. It is generated
instead of hand-written so it cannot go stale after a retrain, and so the project's main claim
is computed from the evidence.

## Phase 6: what the finished system actually measured

Held-out validation, n = 179:

| model | params | accuracy | ROC-AUC | to ship? |
|---|---|---|---|---|
| `fast` | 34 | 0.827 | 0.859 | **yes** |
| `deep` | 1,281 | 0.821 | 0.859 | |
| `gbdt` | 1,084 | 0.832 | 0.848 | |
| `attn` | 7,361 | 0.788 | 0.840 | |

Every point estimate falls inside every other model's 95% interval. The intervals are about
±0.06 wide, while the spread between best and worst is 0.019. A 34-parameter logistic
regression matches a 7,361-parameter transformer, so the sensible recommendation is the small
one.

The load test at concurrency 16 produced the other useful number:

```
p50=109.81 ms  p95=221.5 ms  p99=259.07 ms   ok=300  rejected=0
peak_queue_depth=13  peak_inflight=2
server p95 by stage: queue=154.267ms  preprocess=9.891ms  inference=16.608ms
```

Queue p95 is 154 ms against 16.6 ms of actual inference. Under load this service is limited by
capacity rather than compute. Per-stage timing is what makes that conclusion possible; HTTP
middleware timing alone would not have shown it.
