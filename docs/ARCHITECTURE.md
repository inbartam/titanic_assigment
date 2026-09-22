# ARCHITECTURE.md — Contracts and Specifications

This is the source of truth for module boundaries, the artifact format, the four models, and the
Streamlit UI. Code must match this doc; if the doc is wrong, fix the doc in the same commit.

---

## 1. Data flow

```
Kaggle train.csv ──► data.load_csv ──► data.validate_schema ──► data.stratified_split (80/20, seed 42)
                                                                          │
                                        ┌─────────────────────────────────┴────────────────┐
                                     train_df (712)                                      val_df (179)
                                        │                                                   │
                            features.engineer(df)  [pure]                       features.engineer(df)
                                        │                                                   │
                            Preprocessor.fit(train)  ──► preprocessor.json                  │
                            Preprocessor.transform ──► (X_num, X_cat, y)        Preprocessor.transform (loaded)
                                        │                                                   │
                 5-fold CV on train only ──► pick config ──► train (torch) / fit (sklearn)     │
                                        │        for each of: fast, deep, attn, gbdt         │
                                        │                                                   │
                       model.pt | model.joblib + model_config.json + history.json           │
                                        │                                                   │
                                        └──────────────► evaluation.compute_metrics  ◄───────┘  (scored ONCE)
                                                                     │
                                                           metrics.json + plots/*.png + registry.json
```

Inference (app and API): `InferenceService.predict(df, model, threshold)` →
`features.engineer` → `preprocessor.transform` → `bundle.predict_proba(X_num, X_cat)`
(torch: `sigmoid(model(...))`; sklearn: `predict_proba`) → threshold, with queue accounting and
stage timing recorded in `MetricsRegistry`. **No training data required.** All models consume
the *same* `(X_num, X_cat)`. Neither `app/` nor `api/` ever calls a bundle directly — see
`docs/API.md` for the service layer, endpoints and metrics.

---

## 2. Input schema

Raw Kaggle Titanic columns. Validation is done by `data.validate_schema(df, require_target)`.

| column        | dtype           | required | notes                                                   |
|---------------|-----------------|----------|---------------------------------------------------------|
| `Survived`    | int {0,1}       | no       | if present → evaluation enabled                         |
| `PassengerId` | int             | no       | echoed in predictions if present                        |
| `Pclass`      | int {1,2,3}     | yes      | treated as categorical (ordinal but non-linear effect)  |
| `Name`        | str             | yes      | for `Title`                                             |
| `Sex`         | str             | yes      | `male`/`female`; case-normalized                        |
| `Age`         | float, may be NaN | yes    | imputed                                                 |
| `SibSp`       | int             | yes      |                                                         |
| `Parch`       | int             | yes      |                                                         |
| `Fare`        | float, may be NaN | yes    | imputed, `log1p`                                        |
| `Cabin`       | str, mostly NaN | **no**   | absent column ≡ all-NaN                                 |
| `Embarked`    | str {C,Q,S}, may be NaN | **no** | absent column ≡ all-NaN → mode                     |
| `Ticket`      | str             | **no**   | not used by the model                                   |

Errors raised: `SchemaError` (missing required columns, listing them), `SchemaError` (non-numeric
values in numeric columns, naming the column), `SchemaError` (`Survived` present but not in
{0,1}). Extra columns are ignored with a logged warning.

---

## 3. Feature engineering (`features.py`, pure functions)

| feature      | source              | why it may carry signal                                  | leakage? | robust at single-row inference? | in model? |
|--------------|---------------------|----------------------------------------------------------|----------|---------------------------------|-----------|
| `Title`      | `Name` regex `,\s*([^\.]+)\.` → {Mr, Mrs, Miss, Master, Rare} (Mlle/Ms→Miss, Mme→Mrs, else Rare) | proxies sex×age×status; Master identifies boys | no       | yes; unseen → Rare              | **yes** (cat) |
| `FamilySize` | `SibSp + Parch + 1` | non-monotonic survival (2–4 best)                        | no       | yes                             | **yes** (num) |
| `IsAlone`    | `FamilySize == 1`   | sharp drop for solo travelers                            | no       | yes                             | **yes** (cat/binary) |
| `Deck`       | first letter of `Cabin`, NaN → `U` | deck ≈ price tier + lifeboat proximity; U itself is informative | no | yes                          | **yes** (cat) |
| `HasCabin`   | `Cabin.notna()`     | recorded cabin correlates with class/survival            | no       | yes                             | folded into `Deck=U`; not separate |
| `LogFare`    | `log1p(Fare)`       | heavy right skew                                         | no       | yes                             | **yes** (num) |
| `Age`        | raw                 | children prioritized                                     | no       | yes (imputed by Title)          | **yes** (num) |
| `Pclass`, `Sex`, `Embarked` | raw   | dominant effects                                         | no       | yes                             | **yes** (cat) |
| `SibSp`, `Parch` | raw             | redundant with `FamilySize`                              | no       | yes                             | **no** (redundancy; noted in EDA) |
| `TicketGroupSize` | count of ticket over batch | strong signal on Kaggle                           | **yes: batch-dependent; trivially different for a single row** | no | **no** |
| `FarePerPerson` | `Fare / TicketGroupSize` | ditto                                            | inherits above | no                         | **no** |
| `AgeBin`     | bucketed Age        | NN/linear can learn it from `Age`                        | no       | yes                             | **no** (keeps model simple) |

Final feature sets:
- **numeric** (standardized): `Age`, `LogFare`, `FamilySize`
- **categorical** (vocab with `<UNK>`=0): `Pclass`, `Sex`, `Embarked`, `Title`, `Deck`, `IsAlone`

`engineer(df) -> pd.DataFrame` adds the engineered columns and returns a copy. It does not
impute, scale, or encode.

---

## 4. `Preprocessor` (`preprocessing.py`)

```python
class Preprocessor:
    numeric_cols: list[str]        # ["Age", "LogFare", "FamilySize"]
    categorical_cols: list[str]    # ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]

    def fit(self, df: pd.DataFrame) -> "Preprocessor": ...
        # learns: age_median_by_title (dict) + age_global_median, fare_median, embarked_mode,
        #         num_mean, num_std (per numeric col), vocab per categorical col
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...
        # returns X_num float32 (n, 3), X_cat int64 (n, 6); raises NotFittedError if unfitted
    def to_dict(self) -> dict / from_dict(d) / save(path) / load(path)
    @property
    def cardinalities(self) -> list[int]   # len(vocab)+1 per categorical col (index 0 = <UNK>)
```

Order of operations inside `transform`: `engineer` is **not** called here (caller does it) →
impute `Embarked` (mode) → impute `Fare` (median) → recompute `LogFare` → impute `Age`
(Title median, fallback global) → standardize numerics → map categoricals to indices.

`preprocessor.json` schema:

```json
{
  "version": 1,
  "numeric_cols": ["Age", "LogFare", "FamilySize"],
  "categorical_cols": ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"],
  "imputation": {"age_median_by_title": {"Mr": 30.0, "...": 0}, "age_global_median": 28.0,
                  "fare_median": 14.45, "embarked_mode": "S"},
  "scaling": {"mean": {"Age": 29.4, "...": 0}, "std": {"Age": 13.2, "...": 0}},
  "vocab": {"Sex": {"<UNK>": 0, "female": 1, "male": 2}, "...": {}}
}
```

---

## 5. Models (`models.py`, `sklearn_models.py`)

Common interface (wrapped by `artifacts.Bundle`):
`predict_proba(X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray[n]` of P(survived).
Torch models: `forward(x_num: FloatTensor[B,3], x_cat: LongTensor[B,6]) -> FloatTensor[B]` (logits).

### `fast` — `TitanicLinear` (torch)
- `F.one_hot(x_cat[:, i], num_classes=card_i)` per categorical col, concatenated with `x_num`
  → `nn.Linear(in, 1)`. ≈ 25 parameters. This *is* logistic regression, trained with the same
  loop/loss/optimizer as the other torch models → apples-to-apples.
- Config: `{"type": "linear", "lr": 0.01, "weight_decay": 1e-4, "epochs": 200, "batch_size": 64}`.
  No CV (one obvious config).

### `deep` — `TitanicMLP` (torch)
- `nn.Embedding(card_i, dim_i)` per categorical col, `dim_i = min(8, ceil(card_i/2))`.
- concat(embeddings, `x_num`) → `Linear(→h1)` → `ReLU` → `Dropout(p)` → `Linear(→h2)` → `ReLU`
  → `Dropout(p)` → `Linear(→1)`.
- Default: `h1=64, h2=32, p=0.3, lr=1e-3, weight_decay=1e-3, batch_size=64, max_epochs=300,
  patience=20`. ≈ 3–4k parameters.
- **Selection grid** (5-fold stratified CV on the training split, metric = mean ROC-AUC,
  tie-break log-loss): `hidden ∈ {(32,16), (64,32)}`, `dropout ∈ {0.2, 0.4}`,
  `weight_decay ∈ {1e-4, 1e-3}` → 8 configs × 5 folds. Winner retrained on the full training
  split with early stopping on the inner 10% carve-out. Grid stored in `history.json["cv_grid"]`.

### `attn` — `TitanicAttention` (torch) — *cut-first model*
- Tokenize every feature: categorical col i → `nn.Embedding(card_i, d)`; numeric col j →
  `nn.Linear(1, d)` (value × learned vector + bias); prepend a learned `[CLS]` token;
  add a learned per-token position/feature embedding. `d = 16`, 10 tokens.
- `nn.TransformerEncoder(TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64,
  dropout=0.2, batch_first=True, norm_first=True), num_layers=2)` → take `[CLS]` →
  `LayerNorm` → `Linear(16, 1)`. ≈ 6–8k parameters, < 60 s on CPU.
- **Single fixed config, no grid** (time-boxed). Trained with the same loop, `lr=1e-3`,
  `weight_decay=1e-3`, `patience=25`. Report its 5-fold CV score for context only.
- If cut: remove the class, the CLI choice, the registry entry, and all README/DECISIONS
  mentions in one commit.

### `gbdt` — `HistGradientBoostingClassifier` (sklearn)
- Input: `np.hstack([X_num, X_cat])` with `categorical_features = [False]*3 + [True]*6`;
  the `<UNK>=0` index is just another category.
- Tiny grid via 5-fold CV on the training split: `max_depth ∈ {3, None}`,
  `learning_rate ∈ {0.05, 0.1}`, `max_iter=300`, `early_stopping=True` (its own internal 10%),
  `random_state=seed`. Saved with `joblib`; `model_config.json` records `sklearn.__version__`.
- Exists because it is the honest strongest classical reference on small tabular data and
  because the app should let a user *see* that, not read about it.

`count_parameters(model)` (torch) / `n_estimators × max_leaf_nodes` (sklearn) logged and shown.

## 6. Training (`training.py`) and artifacts (`artifacts.py`)

- Loss `BCEWithLogitsLoss`, optimizer `AdamW`, `DataLoader(shuffle=True, generator=seeded)`.
- `EarlyStopping` monitors inner-val loss, restores best weights.
- `history.json`: `{"epochs": [...], "train_loss": [...], "val_loss": [...], "train_acc": [...],
  "val_acc": [...], "best_epoch": k, "cv_grid": [...]}` (inner-val = the 10% carve-out, never
  the held-out validation).
- `metrics.json`:

```json
{
  "model": "deep", "framework": "torch", "seed": 42, "trained_at": "2026-09-23T10:00:00Z",
  "n_train": 712, "n_val": 179, "n_params": 3457, "threshold": 0.5,
  "validation": {"accuracy": 0.83, "precision": 0.80, "recall": 0.74, "f1": 0.77,
                 "roc_auc": 0.87, "pr_auc": 0.85, "brier": 0.13,
                 "confusion_matrix": [[100, 10], [20, 49]]},
  "validation_ci95": {"accuracy": [0.78, 0.88], "roc_auc": [0.81, 0.92], "...": []},
  "cv_train_split": {"roc_auc_mean": 0.86, "roc_auc_std": 0.03},
  "inference_ms_per_1k_rows": 3.1
}
```

- `model_config.json` always contains `"framework": "torch" | "sklearn"`, `"name"`, the
  architecture/hyperparameters, `"n_params"`, `"seed"`, and library versions.
- `registry.json`: `{"models": {"fast": {"dir": "fast", "framework": "torch",
  "trained_at": "...", "roc_auc": 0.86, "n_params": 25}, "deep": {...}, "attn": {...},
  "gbdt": {...}}, "default": "deep"}`. Entries are only written for models that actually trained;
  the app must tolerate any subset. **`dir` is relative to `registry.json` itself** (not to the
repository root, as an earlier draft of this document said), so the artifacts tree can be moved
or written by `--artifacts-dir` and still resolve; `artifacts.bundle_dir()` performs the lookup
and still accepts the older `artifacts/<name>` form.
- `load_bundle(dir) -> Bundle(model, model_config, preprocessor, metrics, history)`; dispatches
  on `framework`: torch → build from config, `load_state_dict(torch.load(model.pt,
  map_location="cpu"))`, `eval()`; sklearn → `joblib.load(model.joblib)` with a version-mismatch
  warning. `Bundle.predict_proba` hides the difference from the app.

---

## 7. Streamlit UI spec (`ds_app.py` + `app/`)

Page config: wide layout, title "Titanic Survival — Inference & Evaluation", 🚢 icon.

**Sidebar**
1. Model: radio built from `registry.json` (only models present): `fast — logistic regression
   (PyTorch)` / `deep — MLP with embeddings (PyTorch)` / `attn — tiny transformer (PyTorch)` /
   `gbdt — gradient boosting (sklearn)`, each with param count, validation ROC-AUC and
   inference time as a caption.
2. Data source: radio `Bundled sample` / `Upload CSV` / `Path on disk` (text input). Path input
   validates existence and `.csv` extension before reading.
3. Decision threshold: slider 0.05–0.95 step 0.05, default 0.50, caption: "Default 0.5. Lower
   → higher recall on survivors; higher → higher precision. Metrics below update live; the
   threshold is not tuned on this data."
4. "Run" button (predictions cached by `(model, file hash, threshold)`).

**Sidebar (cont.)**
5. Mode badge: `Local (in-process)` or `API @ http://…` (from `TITANIC_API_URL` or a text box);
   unreachable API → warning + automatic fallback to local.

**Tabs** — Overview · Data · Predictions · Evaluation · Compare models · Ops
- **Overview** — one paragraph on what the model does; feature list; how training was done
  (bullet list); link to README; validation metrics card of the selected model.
- **Data** — `st.dataframe(head(20))`, shape, missingness bar, schema check result (green
  "Schema OK: 11/11 required columns, target present" or red error + expander with expected
  schema).
- **Predictions** — table: `PassengerId` (if any), `Name` (if any), `p_survived`, `prediction`;
  histogram of `p_survived`; download button `predictions.csv`.
- **Evaluation** (only if `Survived` present; otherwise `st.info` explaining why) — metric tiles
  (accuracy, precision, recall, F1, ROC-AUC, PR-AUC) each with 95% bootstrap CI as caption;
  confusion matrix; ROC; PR curve; threshold sweep; calibration (if not cut).
- **Compare models** — table of every registered model's `metrics.json` (validation + CI,
  params, framework, inference time); overlaid ROC and PR curves for *all* models on the
  currently loaded CSV if labeled (one Plotly figure each, legend toggles); training curves for
  each torch model (tabs); a short honest paragraph auto-filled from the numbers ("On the
  held-out validation set the linear model is within the 95% CI of the MLP…"). Selecting a model
  in the sidebar highlights it in the table.

- **Ops** — from `predictor.stats()` (local: `InferenceService.stats()`; API: `GET /stats`):
  counters (requests, rows, per model/endpoint, RPS), latency p50/p95/p99 overall and per stage
  (queue / preprocess / inference / postprocess) as a Plotly grouped bar, queue depth + in-flight
  + rejections as metric tiles, error rate, predicted positive rate vs the training base rate
  (0.38) as a drift line, model info (framework, params, load time), process RSS/CPU. In API mode
  a **Run load test** button runs `scripts/load_test.py` (N=200, concurrency 16) and re-renders
  so the queue-depth number visibly moves. Refresh button; no auto-polling.

All figures are Plotly (`titanic.plots`) rendered with `st.plotly_chart(fig,
use_container_width=True)`; the same figures are saved as HTML by `train.py`.

**Error handling** — any exception inside a tab → `st.error(f"{e}")` with `st.expander("Details")`
containing the traceback. Never a raw red traceback. Missing/partial artifacts → error telling
the user to run `python train.py --model all`; a partial registry is *not* an error.

---

## 8. Delivery (local-only)

- The reviewer clones the repo on their own machine; README is the deployment.
- Committed `artifacts/` (all models, < 1 MB with `gbdt`) and `data/sample_train.csv` → the app
  runs immediately after `pip install -r requirements.txt` with **no Kaggle credentials**.
  Training (`train.py`) is the only step that needs Kaggle, and it has `--data-path` too.
- Dev platform is Windows 11 / Python 3.12 / pip + venv; README gives PowerShell commands first
  and bash equivalents second. `requirements.txt` line 1 is the torch CPU index URL.
- No secrets, no Docker, no cloud. (Both are listed under "Future work".)
- Optional second process: `uvicorn api.main:app --port 8000` for the API + `/metrics`; the
  app can point at it with `TITANIC_API_URL`. Not required to use the app.

## 9. Service layer (summary — full spec in `docs/API.md`)

`titanic.service.InferenceService` is the only object that touches bundles at inference time.
It owns: bundle loading (partial registry tolerated), a `threading.Semaphore(max_concurrency)`
with an explicit waiting counter (queue depth), `queue_timeout_s`, `max_queue` back-pressure,
stage timers, and a `MetricsRegistry` (Prometheus objects + a 2000-record ring buffer for exact
percentiles). Both `app/client.LocalPredictor` and `api/main.py` are adapters over it.
