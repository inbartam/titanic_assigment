# CLAUDE.md — Titanic Survival: End-to-End DS Take-Home

This file is read by Claude Code at the start of every session. Follow it strictly.
Companion docs: `PLAN.md` (10-hour execution plan), `docs/ARCHITECTURE.md` (module contracts,
artifact schema, UI spec), `docs/DECISIONS.md` (why we did what we did — interviewer Q&A),
`docs/API.md` (FastAPI inference service, metrics, queue model), `docs/assignment.md` (the
original assignment text).

## 1. What this project is

An interview take-home (1 day; we budget **10 working hours**). Build an end-to-end binary
classification pipeline on the Kaggle Titanic dataset that will be judged on: reproducibility,
EDA depth, preprocessing soundness, PyTorch correctness, evaluation methodology, visualization,
Streamlit UX, code organization, documentation, error handling, robustness, originality.

**This is an interview submission, not a leaderboard exercise.** Every non-trivial choice must
have a one-sentence answer to "why did you do this?" Log those answers in `docs/DECISIONS.md`.

Delivery is **local-only**: the reviewer clones the GitHub repo and follows README on their own
machine. No cloud deployment. Therefore README + a fresh-clone test are part of the deliverable.

Beyond the assignment, the backend is an **instrumented inference service**: a
`titanic.service.InferenceService` (bounded concurrency queue + Prometheus metrics: latency per
stage, usage in requests and rows, error rate, queue depth, in-flight, prediction drift) exposed
by a FastAPI app (`api/main.py`) *and* used in-process by Streamlit. See `docs/API.md`. The
Streamlit app must keep working with **no server running** (assignment requirement); the API is
the bonus layer, not a dependency.

## 2. Hard requirements (never violate)

- Fetch the dataset **programmatically from Kaggle** (with a committed `data/sample_train.csv`
  fallback and a `--data-path` override so the pipeline never hard-fails without credentials).
- Use **only `train.csv`**. Never touch `test.csv` or `gender_submission.csv`.
- **Stratified** train/validation split from `train.csv` (80/20, `SEED = 42`).
- All learned preprocessing (imputation values, scaling stats, category vocabularies) is
  **fitted on the training split only** and serialized to disk. Validation/inference data is
  transformed with the fitted preprocessor. No exceptions.
- The required deliverable models are **PyTorch**. Training lives in a standalone `train.py`.
  Weights + all preprocessing artifacts are saved to `artifacts/`.
- `ds_app.py` (Streamlit) loads artifacts from disk, accepts a CSV (upload, path, or bundled
  sample), validates schema, runs inference, and shows metrics + plots when `Survived` exists.
  It must **not crash when `Survived` is absent** — it runs inference and explains that metrics
  need labels.
- Fixed seeds everywhere (`random`, `numpy`, `torch`, sklearn `random_state`); CPU-only;
  deterministic algorithms.
- Inference must **never need the original training dataframe**. Everything needed is in artifacts.
- Never repeatedly tune against the validation split. Any hyperparameter/model selection uses
  **5-fold stratified CV inside the training split**. Validation is scored **once**, at the end.
- Development happens on **Windows (native), Python 3.12, pip + venv**. All commands in docs
  and README must have a Windows (PowerShell) form first and a Unix form second.

## 3. The "model ladder" differentiator

Four models, one preprocessor, one artifact contract, all selectable in the app:

| name   | framework | class                        | what it is                                                        | role                                             |
|--------|-----------|------------------------------|-------------------------------------------------------------------|--------------------------------------------------|
| `fast` | torch     | `TitanicLinear`              | Logistic regression in PyTorch: one-hot cats + numerics → `Linear(→1)` | Interpretable, <5 s, honest in-framework baseline |
| `deep` | torch     | `TitanicMLP`                 | Categorical embeddings + numerics → MLP (2 hidden, dropout, weight decay, early stopping) | The main required NN                             |
| `attn` | torch     | `TitanicAttention`           | Per-feature tokens + CLS → 1–2 `TransformerEncoder` layers → head (tiny FT-Transformer) | "Originality" model; **first to cut if behind**  |
| `gbdt` | sklearn   | `HistGradientBoostingClassifier` | Trees on the same preprocessed features (categorical mask)     | Strongest classical reference, selectable too    |

`train.py --model {fast,deep,attn,gbdt,all}` trains and registers each in
`artifacts/registry.json`. The app's **Compare models** tab shows all trained models side by side.
Priority order if time runs short: `fast` → `deep` → `gbdt` → API+Ops → `attn`.
With the API in scope, **`attn` is built only if Phase 5 ends ahead of schedule.**

If a simpler model beats a fancier one on validation, **report it honestly** and explain why
(n=712, tabular data, strong low-order signal, NN variance). Never "fix" it by re-tuning against
validation. The interesting deliverable is the *comparison with confidence intervals*, not the
winner.

## 4. Repository layout (do not drift from this)

```
.
├── CLAUDE.md
├── PLAN.md
├── README.md
├── requirements.txt
├── pyproject.toml            # ruff/black config, package = src/titanic
├── train.py                  # CLI entry point for training
├── ds_app.py                 # Streamlit entry point (thin)
├── data/
│   └── sample_train.csv      # ~100 stratified rows of train.csv, committed (fallback + demo)
├── notebooks/
│   └── eda.ipynb             # imports feature logic from src/, never re-implements it
├── src/titanic/
│   ├── __init__.py
│   ├── config.py             # dataclasses: Paths, SplitConfig, ModelConfig, TrainConfig
│   ├── data.py               # fetch_from_kaggle(), load_csv(), validate_schema(), split()
│   ├── features.py           # pure functions: raw df -> engineered df (no fitting)
│   ├── preprocessing.py      # Preprocessor: fit / transform / save / load (JSON)
│   ├── models.py             # TitanicLinear, TitanicMLP, TitanicAttention, build_model()
│   ├── sklearn_models.py     # build_gbdt(), fit/predict wrappers with the same interface
│   ├── training.py           # train_torch_model(), EarlyStopping, cross_validate()
│   ├── evaluation.py         # compute_metrics(), bootstrap_ci(), curve data (no plotting)
│   ├── plots.py              # Plotly figures from curve data (used by train.py AND the app)
│   ├── artifacts.py          # save_bundle() / load_bundle() / registry helpers
│   ├── schemas.py            # Pydantic request/response models (API + app share validation)
│   ├── metrics.py            # MetricsRegistry: prometheus objects + ring buffer, snapshot()
│   ├── service.py            # InferenceService: bounded queue, stage timing, predict/evaluate
│   └── utils.py              # set_seed(), get_logger(), timers
├── api/
│   ├── __init__.py
│   ├── settings.py           # pydantic-settings (TITANIC_* env vars)
│   └── main.py               # FastAPI app: /health /models /predict /predict/csv /evaluate /stats /metrics
├── scripts/
│   └── load_test.py          # async load generator; prints p50/p95/p99 and peak queue depth
├── app/                      # Streamlit helpers
│   ├── components.py         # reusable UI blocks (metric tiles, schema panel, ...)
│   ├── client.py             # Predictor protocol: LocalPredictor (service) | ApiPredictor (httpx)
│   └── state.py              # cached loaders (@st.cache_resource / cache_data)
├── artifacts/                # COMMITTED (small): registry.json + one folder per model
│   ├── registry.json
│   ├── fast/  {model.pt, model_config.json, preprocessor.json, metrics.json, history.json, plots/}
│   ├── deep/  {same}
│   ├── attn/  {same}
│   └── gbdt/  {model.joblib, model_config.json, preprocessor.json, metrics.json, plots/}
├── tests/
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_train_smoke.py
│   ├── test_service.py       # queue depth / rejection / stage timing
│   └── test_api.py           # FastAPI TestClient
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DECISIONS.md
│   ├── assignment.md
│   └── screenshots/
└── .streamlit/config.toml    # theme
```

## 5. Coding standards

- Python 3.12 (see §6 for why not 3.13/3.14). Type hints on every public function.
  Google-style docstrings. `logging`, never bare `print` in `src/` or `train.py`.
- Pure functions for feature engineering. Stateful things (`Preprocessor`, models) have explicit
  `fit` / `transform` / `save` / `load`; `transform` before `fit` raises `NotFittedError`.
- Torch artifacts are **JSON + `state_dict`**. The single exception is `gbdt/model.joblib`
  (sklearn has no clean JSON serialization); the sklearn version is pinned and recorded in
  `model_config.json`. No pickled dataframes, no pickled preprocessors.
- Every user-facing error is a specific exception whose message says *what* is wrong and
  *how to fix it* (`SchemaError("Missing required columns: ['Pclass', 'Sex']. Expected the raw
  Kaggle Titanic schema; see README.")`).
- Unknown categories at inference map to a reserved `<UNK>` index (0), never crash.
- One plotting implementation: `titanic.plots` builds Plotly figures; `train.py` saves them as
  HTML (`fig.write_html`, no kaleido dependency) and the app renders them with `st.plotly_chart`.
- `ruff` clean, `black` formatted. Line length 100. `pathlib` everywhere (Windows paths!).
- Small, single-purpose commits with conventional messages (`feat:`, `fix:`, `docs:`, `test:`).
- No hardcoded absolute paths. All paths flow through `config.Paths`, relative to repo root.
- Keep `ds_app.py` < 200 lines; UI logic lives in `app/`.
- **One inference path.** Streamlit (local mode) and FastAPI both call
  `InferenceService.predict/evaluate`. Nothing in `app/` or `api/` touches a model directly.
- Metrics are recorded inside the service, never in route handlers; route handlers only map
  exceptions to HTTP codes. Every `InferenceService` error is a typed exception
  (`SchemaError`, `ModelNotFoundError`, `QueueFullError`, `QueueTimeoutError`, `NoArtifactsError`).
- API errors use one JSON shape `{"error", "message", "details"}`; never a traceback.

## 6. Commands (Windows PowerShell first, Unix second)

```powershell
# --- setup (Windows) ---
py -3.12 -m venv .venv            # install 3.12 from python.org if `py -3.12` is not found
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# --- Kaggle credentials (Windows) ---
# put kaggle.json in  %USERPROFILE%\.kaggle\kaggle.json   OR
$env:KAGGLE_USERNAME="..." ; $env:KAGGLE_KEY="..."

python -m titanic.data --fetch                       # downloads data\train.csv only

# --- train / app / quality ---
python train.py --model all                          # fast, deep, attn, gbdt -> artifacts\
python train.py --model deep --data-path data\sample_train.csv --epochs 5 --no-cv   # smoke
streamlit run ds_app.py                              # local mode (no server needed)

uvicorn api.main:app --port 8000                     # inference API + /metrics + /stats
$env:TITANIC_API_URL="http://127.0.0.1:8000"; streamlit run ds_app.py   # app in API mode
python scripts\load_test.py --n 300 --concurrency 16 --model deep       # watch queue depth move

pytest -q
ruff check . ; black --check .
```

```bash
# --- Unix equivalents ---
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m titanic.data --fetch && python train.py --model all && streamlit run ds_app.py
```

**Why 3.12, not the 3.13/3.14 already on the machine:** the safest combination for
`torch` + `streamlit` + `kaggle` on Windows CPU is 3.11/3.12 wheels. 3.13 may work with a recent
torch; 3.14 is too new to trust for a 10-hour deadline. The `py` launcher lets 3.12 live
side-by-side with whatever is installed. README states "Python 3.12 (3.11 also fine)".

`requirements.txt` line 1 must be `--extra-index-url https://download.pytorch.org/whl/cpu`
so `torch` resolves to the small CPU wheel on Windows.

## 7. Working rules for Claude Code

1. Read `PLAN.md` and work **phase by phase**. Finish the phase's "done when" checklist, run
   `pytest -q`, commit, then move on. Do not start Phase N+1 with Phase N failing.
2. Before writing a module, re-read its contract in `docs/ARCHITECTURE.md`. If you must deviate,
   update the doc in the same commit and add a line to `docs/DECISIONS.md`.
3. When the notebook needs feature logic, `from titanic.features import ...`. Never copy code
   into a notebook cell.
4. Every time you make a modeling/preprocessing choice, append an entry to `docs/DECISIONS.md`
   in the Q/A format already used there.
5. Time-box. If something is not converging in 20 minutes, choose the simpler option, note it as
   a limitation, and move on. `attn` in particular: if it is not training cleanly within its
   30-minute box, ship without it and delete every reference to it from README/app.
6. Never run training on the validation split, never load `test.csv`, never call
   `preprocessor.fit` on anything but the training split.
7. Do not add dependencies beyond `requirements.txt` without a reason written in DECISIONS.md.
8. Windows: use `pathlib.Path`, never string-concatenate paths; open files with
   `encoding="utf-8"`; no `os.fork`, no `num_workers>0` in DataLoader; uvicorn `--workers 1`.
8b. Build the service layer (`metrics.py`, `service.py`) *before* the Streamlit app, so the app
   is a client of the service from the first line. Do not build the app against bundles
   directly and "add the API later" — that creates the second code path this project forbids.
9. After Phase 6, do the **fresh-clone test**: clone into a temp dir, follow README verbatim in a
   *new* PowerShell window, confirm `train.py` and the app run. Fix anything that required "knowing".
10. Screenshots of the app go in `docs/screenshots/` and are referenced from README.

## 8. Things that quietly lose points — check them

- Preprocessing fitted on the full dataframe before splitting → leakage.
- Age imputed with a global median instead of a fitted, group-aware value (we use Title-median).
- Feature that cannot be computed for a single inference row (e.g., ticket-group counts over
  the batch) → we deliberately **do not** use batch-dependent features.
- Reporting validation numbers as if they were unbiased after selecting on them.
- Accuracy reported alone. Always accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion
  matrix, plus bootstrap 95% CIs.
- App that assumes `Survived` exists. App that assumes `PassengerId` exists.
- App that crashes when only some of the four models were trained (registry may be partial).
- Un-pinned requirements; torch CPU index missing; README commands that only work on bash.
- README that says "pip install" but never says where `train.csv` comes from or how to get a
  Kaggle token.
- Streamlit app that breaks when the API is down (must fall back to local mode with a warning).
- Metrics that only exist in the API path — the Ops tab must show numbers in local mode too.
- Queue depth measured as thread-pool size or in-flight count (wrong): it is *waiting*, not
  *executing*, requests.
