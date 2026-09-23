# PLAN.md: 10-Hour Execution Plan (Windows, local-only delivery)

Total budget: **10 h** (realistically 10 to 11 with the API layer, which the buffer absorbs).
Phases are ordered so that a runnable, submittable project exists from about hour 5.5 onward. If
you fall behind, cut from the "Cut first" list at the bottom, never from Phases 0 to 4.

**Scope decision:** the instrumented API (Phase 5a) is in, and `attn` is out unless Phase 5b ends
ahead of schedule. An instrumented service sets the project apart more than a fourth model would.

Legend: ⏱ time box · ✅ done-when · 💾 commit message

---

## Phase 0: Environment, Kaggle, scaffold (⏱ 0:00-1:00)

**Environment (Windows)**
- Confirm `py -3.12 --version`. If missing, install Python 3.12 from python.org (add to PATH
  unchecked is fine; the `py` launcher finds it). Do not build on 3.13/3.14.
- `py -3.12 -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → upgrade pip.
- `requirements.txt`: pinned; line 1 `--extra-index-url https://download.pytorch.org/whl/cpu`;
  then `torch`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `plotly`, `streamlit`, `kaggle`,
  `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `prometheus-client`,
  `httpx`, `python-multipart`, `psutil`, `pytest`, `ruff`, `black`, `jupyter`,
  `matplotlib`+`seaborn` (notebook only).
- `pyproject.toml` with `[tool.ruff]`, `[tool.black]`, setuptools `package-dir = {"" = "src"}`.

**Kaggle (≈10 min, do it now, not later)**
- kaggle.com → Settings → API → *Create New Token* → save to `%USERPROFILE%\.kaggle\kaggle.json`.
- Open https://www.kaggle.com/competitions/titanic/rules and click *I Understand and Accept*
  (the API returns 403 otherwise).
- `python -c "import kaggle"` must not raise. If corporate proxy blocks it, set
  `HTTPS_PROXY` or fall back to `--data-path` and note it.

**Scaffold**
- Repo layout from `CLAUDE.md §4`. `src/titanic/utils.py`: `set_seed`, `get_logger`.
- `src/titanic/data.py`: `fetch_from_kaggle(dest)` via `kaggle.api.competition_download_file
  ("titanic", "train.csv", path=dest)` with a clear error if credentials/rules are missing;
  `load_csv(path)`; `validate_schema(df, require_target=False)`; `stratified_split(df, test_size, seed)`.
- Generate `data/sample_train.csv` = stratified 100 rows of `train.csv` (seeded). Commit it.
- `.gitignore`: `data/train.csv`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`,
  `.pytest_cache/`, but **not** `artifacts/`.
- `git init`, first commit, create GitHub repo (private until submission).

✅ `python -m titanic.data --fetch` writes `data\train.csv`; without creds it prints the two
   setup steps above instead of a stack trace. `pytest -q` collects.
💾 `chore: scaffold project, kaggle fetch, sample dataset`

---

## Phase 1: Features & Preprocessor (⏱ 1:00-2:15)

- `features.py`: pure functions on the raw dataframe (`docs/ARCHITECTURE.md §3`):
  `extract_title`, `family_size`, `is_alone`, `deck_from_cabin`, `log_fare`, `engineer(df)`.
- `preprocessing.py`: `Preprocessor` with `fit(df_train)`, `transform(df) -> (X_num, X_cat)`,
  `to_dict()/from_dict()`, `save(path)/load(path)`. Learns: Title→median Age table (+ global
  fallback), Fare median, Embarked mode, numeric mean/std, category vocabularies with `<UNK>=0`.
- `tests/test_features.py`, `tests/test_preprocessing.py`:
  - fit → save → load → transform gives identical arrays.
  - unseen `Title`/`Deck` at transform time → index 0, no exception.
  - fitted parameters unchanged after transforming validation data (leakage guard).
  - `transform` before `fit` raises `NotFittedError`.
  - single-row inference works (shapes `(1, 3)`, `(1, 6)`).

✅ All tests pass. `Preprocessor` round-trips through JSON.
💾 `feat: feature engineering and leak-safe preprocessor with JSON persistence`

---

## Phase 2: EDA notebook (⏱ 2:15-3:30)

`notebooks/eda.ipynb`. It should read as a story, with at most 9 figures, each in a
*Question → Analysis → Finding → Decision* block. Import everything from `titanic.features`.
Matplotlib/seaborn is fine here because the output is static and renders on GitHub.

Required sections:
1. Setup & loading (`data/train.csv`; state we only use train.csv).
2. Shape, dtypes, target balance (38% survived → motivates F1 + PR-AUC alongside ROC-AUC).
3. Missingness: Age ~20%, Cabin ~77%, Embarked 2 rows. Decisions: Title-median Age,
   Cabin→Deck with `U`, Embarked mode.
4. Duplicates & leakage check (no duplicate rows; ticket/cabin group survival is
   batch-dependent → excluded, with the reasoning).
5. Survival vs `Sex`, `Pclass` and their interaction (one faceted figure).
6. Age by Title (justifies imputation) + Age distribution by survival.
7. Fare: skew → `log1p`; overlap across classes; outliers kept.
8. FamilySize / IsAlone: non-monotonic survival (peaks at sizes 2 to 4).
9. Engineered-feature correlation (SibSp/Parch redundant with FamilySize).
10. Classical sanity check: 5-fold CV on the *training split only* for sklearn
    LogisticRegression and HistGradientBoosting → expectation band (~0.83 to 0.87 AUC).
11. Conclusions: exact feature list, expected difficulty, why NNs may not win here.

✅ *Kernel → Restart & Run All* succeeds; outputs saved; every figure has a Decision line.
💾 `docs: EDA notebook with feature decisions and classical sanity checks`

---

## Phase 3: Models & training (⏱ 3:30-5:15)

Build in this order. Each step can ship on its own.

1. `models.py`: `TitanicLinear`, `TitanicMLP`, `build_model(config, cardinalities)`,
   `count_parameters`. Both `forward(x_num, x_cat) -> logits (B,)`.
2. `training.py`: `train_torch_model(...)` with `BCEWithLogitsLoss`, `AdamW`, `batch_size=64`,
   seeded `DataLoader` (`num_workers=0`), `EarlyStopping(patience, restore_best=True)` on an
   inner 10% carve-out of the training split; `cross_validate(config, X, y, k=5)`.
   Returns `history`.
3. `artifacts.py`: `save_bundle`, `load_bundle` (dispatches on `framework`), `update_registry`.
4. `train.py` CLI (`--model {fast,deep,attn,gbdt,all}`, `--data-path`, `--seed`, `--epochs`,
   `--test-size`, `--cv/--no-cv`, `--artifacts-dir`). Logs a final metrics table with CIs;
   writes `metrics.json`, `history.json`, `plots/*.html` per model; updates `registry.json`.
5. `sklearn_models.py`: `gbdt` = `HistGradientBoostingClassifier(categorical_features=mask,
   random_state=seed)`; tiny CV grid (`max_depth ∈ {3, None}`, `learning_rate ∈ {0.05, 0.1}`);
   saved with `joblib`. Same `metrics.json` format.
6. ~~`attn`~~: deferred to Phase 7 (only if ahead). Leave the `attn` branch in
   `build_model` as `NotImplementedError` with a comment, or omit it entirely. The CLI choice must
   not appear until the model exists.
7. `tests/test_models.py` (forward shapes, param counts for all torch models),
   `tests/test_train_smoke.py` (`train.py --model fast --data-path data/sample_train.csv
   --epochs 3 --no-cv` completes and produces the artifact files).

✅ `python train.py --model all` finishes in < 3 min on CPU; `registry.json` lists every model
   that trained; `load_bundle` works for both frameworks.
💾 `feat: linear/MLP PyTorch models, GBDT reference, train.py CLI`

> **Checkpoint 1 (≈5:15):** the project is submittable in minimal form. Push to GitHub.

---

## Phase 4: Evaluation & plots (⏱ 5:15-5:55)

- `evaluation.py`: `compute_metrics(y_true, y_prob, threshold=0.5)` (accuracy, precision,
  recall, F1, ROC-AUC, PR-AUC, Brier, confusion matrix); `bootstrap_ci(..., n=1000, seed)`;
  `curve_data(y_true, y_prob)` → dict of ROC/PR/threshold-sweep/calibration arrays.
  No plotting happens in this module.
- `plots.py` (Plotly): `confusion_matrix_fig`, `roc_fig(curves: dict[str, ...])` (overlay
  multiple models), `pr_fig`, `threshold_sweep_fig`, `calibration_fig`,
  `training_curves_fig(history)`, `prob_histogram_fig`. Consistent template/colors.
- `train.py` saves each figure as `artifacts/<model>/plots/<name>.html`.

✅ The same functions produce the HTML files and the app figures, with no duplicated plotting code.
💾 `feat: evaluation metrics with bootstrap CIs and shared Plotly figures`

---

## Phase 5a: Service layer & API (⏱ 5:55-7:10)

Spec: `docs/API.md`. Build bottom-up, since the app in Phase 5b is a client of this layer.

1. `schemas.py`: Pydantic models; `PassengerIn` validators reuse `data.validate_schema` logic
   (one source of truth for required/optional columns and value checks).
2. `metrics.py`: `MetricsRegistry` with the Prometheus objects from API.md §4, a
   `deque(maxlen=2000)` of per-request records, `record(...)`, `snapshot() -> dict`,
   `prometheus_text()`. Own `CollectorRegistry` (not the global one) so tests can create fresh ones.
3. `service.py`: `InferenceService(artifacts_dir, max_concurrency=2, max_queue=64,
   queue_timeout_s=5)`: loads all bundles from the registry (tolerates partial), `predict(df,
   model, threshold) -> PredictionResult`, `evaluate(df, model, threshold, n_boot)`,
   `stats()`, `reload()`. Queue accounting exactly as API.md §2; stage timers around
   preprocess / inference / postprocess; typed exceptions.
4. `api/settings.py`, `api/main.py`: app factory, routes, exception handlers → JSON error
   shape, `X-Request-ID`, JSON access log line, CORS for localhost, `/metrics` via
   `generate_latest(registry)`.
5. `scripts/load_test.py`: async httpx, `--n --concurrency --model --rows`, prints p50/p95/p99,
   error count, peak `queue_depth` read from `/stats` during the run.
6. `tests/test_service.py`, `tests/test_api.py` per API.md §8.

✅ `uvicorn api.main:app` serves `/predict` on a sample row in < 20 ms warm; load test at
   concurrency 16 shows `queue_depth` > 0 and zero 5xx; `/metrics` scrapes; all tests pass.
💾 `feat: instrumented InferenceService with bounded queue, FastAPI endpoints, load test`

## Phase 5b: Streamlit app (⏱ 7:10-8:30)

Build to `docs/ARCHITECTURE.md §7`. The app talks only to `app/client.py`'s `Predictor`
(`LocalPredictor` wraps `InferenceService`; `ApiPredictor` wraps httpx). Priorities, in order:
1. Sidebar: model selector from the registry (only models that exist), data source
   (bundled sample / upload / path), threshold slider (default 0.5, caption), mode badge
   (Local / API @ url) with fallback-to-local warning.
2. Tabs: **Overview · Data · Predictions · Evaluation · Compare models · Ops**.
3. Schema validation with specific errors + an "expected schema" expander.
4. Predictions table + probability histogram + CSV download.
5. Evaluation tab only if `Survived` present; otherwise `st.info` explaining why.
6. Compare tab: metrics table with CIs for all registered models, overlaid ROC/PR on the
   loaded CSV if labeled, training curves for torch models, param counts, inference time.
7. Ops tab from `predictor.stats()`: counters, latency percentiles per stage (Plotly), queue
   depth / in-flight, error rate, positive-rate drift vs 0.38; "Run load test" button in API mode.
8. `.streamlit/config.toml` theme; no raw tracebacks (`try/except` → `st.error` + details expander).

✅ App runs on the bundled sample with no Kaggle access and no server; on a CSV without
`Survived`; rejects a random CSV with a clear message; survives a registry with only `fast`
trained; switches to API mode via env var and back when the API dies; Ops tab shows numbers in
both modes; every tab renders in < 2 s after first load.
💾 `feat: Streamlit inference/evaluation app with model comparison and Ops dashboard`

---

## Phase 6: README, screenshots, fresh-clone test (⏱ 8:30-9:30)

- Fill `README.md` from the template (results table from every `metrics.json`, 5 screenshots:
  data validation, predictions, evaluation, compare, and the Ops tab during the load test), plus
  the API section (run command, endpoint table, one `curl`/PowerShell `Invoke-RestMethod` example,
  load-test output).
- **Fresh-clone test**: `git clone` into `%TEMP%\titanic-check`, open a *new* PowerShell,
  follow README top to bottom (including the no-Kaggle path), and fix every gap you hit.
- Final `ruff check .`, `black --check .`, `pytest -q`.

✅ README has Windows + Unix commands, Kaggle token steps, results with CIs, screenshots,
   an honest "which model to pick and why" paragraph.
💾 `docs: README with results, screenshots and run instructions`

---

## Phase 7: `attn` (only if ≥ 45 min ahead) / buffer / final review (⏱ 9:30-10:00+)

- If Phase 6 finished before 9:00: build `TitanicAttention` (ARCHITECTURE §5) in a 30-min hard
  box, retrain `--model attn`, add to README tables. Otherwise skip and keep README/DECISIONS
  consistent with three models.
- Walk `CLAUDE.md §8` line by line. Re-read `docs/DECISIONS.md`; every entry must still be true.
  Then make the repo public and submit.

---

## Cut first (in this order, if behind)

1. `attn` model. It is already deferred, so cutting it costs nothing.
2. Ops tab "Run load test" button (keep the script + README output).
3. `/admin/reload`, prediction-drift metrics (`positive_rate`, probability histogram).
4. Calibration plot + Brier.
5. GBDT CV grid → single default config; then `deep` grid (say so in DECISIONS.md).
6. Threshold-sweep plot.
7. **Last resort:** the whole FastAPI adapter (`api/`), keeping `service.py` + `metrics.py` so the
   Ops tab still works in local mode. Say in README that the service layer is API-ready.

## Never cut

Leak-safe preprocessor + its tests · `fast` and `deep` PyTorch models · `train.py` · artifacts ·
app runs without labels, with a partial registry, and with no server · metrics beyond accuracy
with CIs · README with Windows commands and Kaggle steps.

## Risk register

| risk                                              | mitigation                                                                 |
|---------------------------------------------------|----------------------------------------------------------------------------|
| Kaggle 403 (rules not accepted) / token missing    | Phase 0 does it first; `data/sample_train.csv` + `--data-path` fallback    |
| torch wheel wrong/slow on Windows                  | CPU index URL line 1 of requirements; Python 3.12 venv                     |
| API layer eats the schedule                        | Service layer first (needed by the app anyway); FastAPI adapter is thin; cut order in the list above |
| Queue metrics wrong under threads                  | `test_service.py` asserts peak depth with a blocked slot; single `threading.Semaphore` shared by both paths |
| Fancy model loses to `fast`/`gbdt`                 | Expected; report with CIs, explain in README + DECISIONS                   |
| Cross-machine nondeterminism                       | CPU only, seeded generators, `use_deterministic_algorithms(True)`; README says "~3 decimals across platforms" |
| Notebook drifts from src feature logic             | Notebook imports `titanic.features`; test asserts `engineer()` output columns == preprocessor expectation |
| PowerShell execution policy blocks venv activation | README one-liner: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`    |
