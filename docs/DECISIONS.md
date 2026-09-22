# DECISIONS.md — Design Decision Log

Format: **Q** (what an interviewer would ask) → **A** (the one-paragraph defensible answer) →
*alternatives considered*. Append, never rewrite history; if a decision changes, add a new entry
that supersedes the old one.

---

### Data & split

**Q: Why an 80/20 stratified split instead of only cross-validation?**
A: The assignment asks for a held-out validation set and for the app to evaluate on it. We keep
20% (n=179) untouched for a single final evaluation, and do *all* model selection with 5-fold
stratified CV inside the 80%. This separates "choosing" from "reporting", so the reported numbers
are not optimistically biased by selection. *Alternative:* nested CV — more rigorous but overkill
for a 1-day task; noted as a limitation.

**Q: Why stratify?**
A: 38% positive rate; on n=179, an unstratified split can shift the class ratio by several points
and change accuracy by ~1–2 points purely from sampling. Stratifying removes that noise source.

**Q: Why commit a 100-row sample?**
A: The assignment asks for a `data/` sample. It also makes the app demo-able and the smoke test
runnable with no Kaggle credentials, which is essential for a reviewer who clones the repo and
wants to see the app in 60 seconds. Full `train.csv` is git-ignored and fetched programmatically.

### Features

**Q: Why not use ticket-group size / fare-per-person? They are among the strongest Kaggle features.**
A: They are computed over the *batch* (count of rows sharing a ticket). At inference on a single
passenger the value is always 1, and on a partial CSV it differs from training — the feature's
meaning depends on which other rows happen to be in the file. That is a train/serve skew, and in
the training split it also leaks group-level survival information across rows. We deliberately
excluded them and say so in the EDA. *Alternative:* fit a ticket→count lookup on the training
split and default unseen tickets to 1 — technically leak-safe but semantically brittle.

**Q: Why Title?**
A: It compresses sex × age × social status into 5 levels and is the best available signal for
imputing Age (Master ≈ 4 y vs Mr ≈ 32 y). Extracted with a regex that is robust to the Kaggle
name format; anything outside the 4 common titles maps to `Rare`, so unseen titles at inference
are handled without error.

**Q: Why impute Age by Title median rather than a global median or a model?**
A: A global median (28) assigns adult ages to children whose title is `Master`, destroying the
"children first" signal. Title-median is a one-line fitted lookup, transparent, and serialized in
`preprocessor.json`. A regression imputer adds complexity and another leakage surface for little gain.

**Q: Why drop SibSp/Parch after making FamilySize?**
A: They are linearly determined by FamilySize together and EDA showed the survival pattern is a
function of total family size. Keeping all three adds collinearity for the linear model without
adding information. *Alternative:* keep them for the MLP only — rejected to keep one feature set.

**Q: Deck has 77% missing. Why keep it?**
A: Missingness itself is informative (no recorded cabin ≈ lower class ≈ lower survival). We encode
NaN as its own level `U`, which lets every model use the missingness pattern without a separate
`HasCabin` flag.

**Q: Why treat Pclass as categorical?**
A: Its effect is not linear (1st ≫ 2nd > 3rd in a non-uniform way). Three levels cost nothing.

### Preprocessing

**Q: How do you guarantee no leakage?**
A: `Preprocessor.fit` is called exactly once, on the training split, in `train.py`. Its state is
serialized to JSON and reloaded for validation and for the app. A unit test asserts that
transforming validation data does not change any fitted parameter, and another asserts that
save→load→transform is bit-identical.

**Q: Why JSON artifacts instead of pickling a sklearn `ColumnTransformer`?**
A: JSON is human-readable in a code review, version-safe, and forces us to enumerate every learned
parameter explicitly. Pickle would hide the preprocessing state and tie inference to a specific
sklearn version.

### Models

**Q: Why four models instead of one?**
A: The assignment rewards "originality and problem-solving", and on Titanic the interesting
question is not *which* model wins but *whether the models are distinguishable at n=179*. A
ladder — logistic regression in PyTorch (`fast`), an MLP with embeddings (`deep`), a tiny
FT-Transformer-style attention model (`attn`), and gradient boosting (`gbdt`) — all on the same
preprocessor and artifact contract lets the app show that comparison with confidence intervals.
It also demonstrates that the training/artifact/inference pipeline is model-agnostic. The
required deliverable is still the PyTorch models; `gbdt` is the honest classical reference,
selectable so the reviewer can *see* it rather than read about it.

**Q: Isn't four models over-engineering for a one-day task?**
A: Each adds ~15 lines of model code and zero pipeline code, because the contract is shared.
The genuine risk is time, so `attn` has a hard 30-minute box and is the first thing cut. If it
is absent from the repo, that is why.

**Q: Why implement logistic regression in PyTorch rather than sklearn?**
A: Same loss, optimizer, batches, seed and preprocessing as the MLP → any gap is attributable
to the architecture, not the training recipe. It also gives a ~25-parameter, fully interpretable
model that trains in seconds, which is the right default for a dataset this size.

**Q: Why was `attn` deferred behind the API?**
A: Ten hours. A fourth model on 712 rows adds a talking point; an instrumented service adds a
capability the reviewer can *run*. If `attn` is in the repo, it was built in the last 30 minutes
after everything else passed the fresh-clone test.

**Q: Why a transformer on 9 features?**
A: Not because it should win — it usually won't on Titanic. It is there to show (a) that
attention over feature tokens is the standard modern tabular-DL architecture and we can
implement it correctly and small (~7k params), and (b) how a more expressive model behaves
under n=712: higher variance, CI overlapping the linear model. That is a more honest
demonstration of judgment than a big model with a suspiciously good number.

**Q: Why HistGradientBoosting and why is it saved with joblib when everything else is JSON?**
A: HGB handles categorical features natively via a mask, so it consumes exactly the same
`(X_num, X_cat)` as the torch models — no second preprocessing path. sklearn has no clean JSON
serialization; joblib with a pinned sklearn version recorded in `model_config.json` is the
standard, and it is the only non-JSON artifact in the repo, called out explicitly.

**Q: Why embeddings in the MLP rather than one-hot?**
A: Mostly for the demonstration that categorical handling is done properly; with cardinalities
≤ 10 the parameter difference is negligible. Dimension `min(8, ceil(card/2))` follows the common
rule-of-thumb. One-hot would be equally defensible.

**Q: Why is the MLP so small?**
A: 712 training rows. A 64→32 MLP with dropout already has ~3–4k parameters (~5 per sample).
Anything larger overfits faster and adds nothing. Weight decay + dropout + early stopping are the
three cheap regularizers that matter here.

**Q: How was early stopping done without touching the validation set?**
A: A 10% stratified carve-out *inside the training split* serves as the inner early-stopping set.
The held-out 20% is never seen during training or selection.

**Q: How did you choose hyperparameters?**
A: An 8-point grid (hidden size, dropout, weight decay) scored by mean ROC-AUC over 5 stratified
folds of the training split. The grid is tiny on purpose: with n=712 the CV standard deviation
(~0.03 AUC) is larger than most between-config differences, so a large search would mostly be
fitting noise. The full grid results are stored in `history.json` and shown in the notebook.

**Q: What if `gbdt` beats every PyTorch model?**
A: It often does by ~1 point. Trees model interactions and thresholds natively, which suits small
tabular data with mixed types; NNs need more data to learn those from scratch. We show it in the
app with CIs. The PyTorch models are the required deliverable; the comparison is there to show
judgment, not to hide it. The honest recommendation paragraph in README and the Compare tab is
auto-derived from the numbers.

### Evaluation

**Q: Why bootstrap confidence intervals?**
A: With n=179 an accuracy of 0.83 has a 95% CI of roughly ±0.055. Reporting a point estimate to
three decimals would be misleading. 1000 stratified bootstrap resamples of the validation set
give an interval for every metric at negligible cost, and let us say honestly whether the two
models are distinguishable.

**Q: Why ROC-AUC *and* PR-AUC?**
A: ROC-AUC measures ranking quality independent of threshold and class balance; PR-AUC is more
sensitive to performance on the minority (survived) class, which is what a user of this model
would care about. F1 at the threshold complements both.

**Q: Why threshold 0.5? Why a slider?**
A: 0.5 is the Bayes decision rule for a calibrated model under symmetric costs and there is no
stated cost asymmetry. The slider exists to *show* the precision/recall trade-off, not to tune:
the caption says so, and no threshold is ever optimized on validation and then reported.

### Service & observability

**Q: The assignment asks for Streamlit. Why is there also a FastAPI service?**
A: Because "evaluate on the validation set and run inference" is a *product*, and an interviewer
judging "originality, robustness, error handling" is asking how this would behave in production.
An HTTP API with schema validation, bounded concurrency and metrics is the smallest honest
answer. It is strictly additive: Streamlit runs in-process with no server, exactly as the
assignment specifies; API mode is a switch.

**Q: Why put the metrics in `InferenceService` instead of FastAPI middleware?**
A: Middleware only sees the HTTP path. Putting queue accounting and stage timers in the service
means (1) the Streamlit app in local mode reports the same latency/usage/queue numbers, (2) there
is one inference code path, and (3) the metrics describe *inference*, not *HTTP* — preprocessing
vs model time is the number a data scientist actually wants.

**Q: What exactly is "queue depth" here and why does it matter?**
A: Requests that have arrived but are not yet executing — waiting on the concurrency semaphore.
It is the autoscaling/back-pressure signal: in-flight count saturates at `max_concurrency` and
tells you nothing once you are busy; queue depth keeps growing and tells you *how* busy. We also
bound it (`max_queue`) and reject with 503 + `Retry-After` instead of letting latency explode.

**Q: Why `max_concurrency=2` on CPU?**
A: Batch inference of a few thousand rows takes milliseconds; two parallel slots overlap I/O and
GIL-released torch ops on a 4-core laptop without thrashing. It is a setting, not a constant, and
the load test in README shows the p95 curve that justified it.

**Q: Why both Prometheus `/metrics` and a JSON `/stats`?**
A: `/metrics` is the industry-standard scrape target (histograms, counters, process metrics) for
a real deployment. `/stats` exists because the Streamlit Ops tab needs exact p50/p95/p99 over a
recent window without running a Prometheus server; a 2000-record ring buffer gives that for free.
Both come from the same `MetricsRegistry`.

**Q: Why a prediction positive-rate metric?**
A: It is the cheapest drift signal there is: if the served positive rate wanders far from the
training base rate (0.38) the input distribution has probably changed. Full drift detection
(feature-level PSI, etc.) is future work.

**Q: What did you deliberately leave out of the API?**
A: Auth, rate limiting per client, multi-worker metrics aggregation, async model execution on
GPU, request batching across clients, OpenTelemetry traces. Each is a well-known next step; none
changes the design. They are listed in README future work.

### Engineering

**Q: Why commit `artifacts/`?**
A: They are < 1 MB, they make the app runnable immediately after `git clone` without Kaggle
credentials (delivery is local-only, so the reviewer's first impression is `streamlit run`), and
they make the submitted results verifiable. `train.py` regenerates them deterministically.

**Q: Why Python 3.12 when the machine has 3.13/3.14?**
A: torch/streamlit/kaggle wheels on Windows CPU are safest on 3.11/3.12. Under a one-day deadline
"works first try on the reviewer's machine" beats "newest interpreter". The `py` launcher makes
3.12 a side-by-side install with no conflict.

**Q: Why Plotly instead of matplotlib?**
A: The app is the reviewer's main interaction surface; interactive ROC/PR overlays with legend
toggles make a four-model comparison readable. One `plots.py` produces the app figures and the
HTML files `train.py` saves, so there is still a single plotting implementation. The notebook
uses matplotlib/seaborn because static figures render on GitHub.

**Q: Is training bit-for-bit reproducible?**
A: On the same machine/torch version, yes (seeded RNGs, CPU, deterministic algorithms, seeded
DataLoader generator). Across OS or torch versions, floating-point kernels can differ in the last
bits; metrics reproduce to ~3 decimals. Stated in README.

**Q: Why do plotting functions live in `src/` and not in the app?**
A: One implementation serves `train.py` (saves HTML to `artifacts/<model>/plots/`) and the app
(`st.plotly_chart`). No duplicated plotting code, identical figures in both places.

### Environment (added during implementation)

**Q: The docs mandate Python 3.12. Why does the repository build on 3.14?**
A: The development machine had only 3.14 installed. Rather than assume, we probed it: a 3.14
venv resolves CPU wheels for `torch 2.14.0+cpu`, `streamlit`, `plotly`, `fastapi` and the whole
stack without incident, so the reason for pinning 3.12 (wheel availability on Windows) no longer
applies. `requires-python` stays at `>=3.11`, so a reviewer on 3.11 or 3.12 is unaffected.
*Alternative:* install 3.12 side-by-side via the `py` launcher — rejected because it would have
cost setup time to avoid a problem that measurement showed does not exist.

**Q: Why are `pandas`, `scipy` and `scikit-learn` pinned below their newest releases?**
A: Windows **Smart App Control** is enabled on the development machine
(`VerifiedAndReputablePolicyState = 1`). It blocks native `.pyd` files that have not yet built
cloud reputation, and it blocked them at *import* time, not install time: `pandas 3.0.6`
(`ccalendar`), `scipy 1.18.1` (`_csparsetools`) and `scikit-learn 1.9.1` (`_argkmin`) all
installed successfully and then failed with `DLL load failed ... An Application Control policy
has blocked this file`. Pinning `pandas==2.3.3`, `scipy==1.16.2` and `scikit-learn==1.7.2` — all
mature, widely-deployed builds — resolves it completely. *Alternative:* disabling Smart App
Control, which is irreversible without a Windows reinstall and is not something a project should
ask of a reviewer's machine. The pins are good practice regardless: pandas 3.0 also carries
breaking API changes we have no reason to absorb during a one-day build.

**Q: Why does `data.py` support four Kaggle credential formats instead of just `kaggle.json`?**
A: Kaggle now issues a `KGAT_`-prefixed access token rather than the classic
username/key `kaggle.json`, and the `kaggle` Python client reads the `KAGGLE_API_TOKEN`
environment variable but *not* the `~/.kaggle/access_token` file that Kaggle's own setup snippet
writes. `_load_kaggle_credentials` therefore checks the env var, then that file (bridging it into
the env var), then the classic env vars, then `kaggle.json`, and raises one `KaggleAuthError`
listing every remedy if all four are absent. A reviewer with either token generation works
without reading the source.

### EDA (added during implementation)

**Q: Why does the EDA notebook create the train/validation split before exploring, rather than
exploring `train.csv` as a whole?**
A: Because looking at the held-out rows biases the analyst even when no code touches them. Every
decision in the notebook -- impute Age by Title, treat Pclass as categorical, drop SibSp/Parch,
exclude ticket-group features -- is a modelling choice informed by what the plots showed. If those
plots included the 179 validation rows, the final "single, unbiased evaluation" would be
evaluating a pipeline that was partly designed on the data it is being scored against. The
notebook therefore splits in its first analysis cell, runs `del val_raw`, and a test
(`tests/test_notebook.py::test_notebook_discards_the_validation_split`) asserts it. This is
stricter than most Titanic notebooks and costs nothing. *Alternative:* explore everything and
rely on the split only at training time -- common practice, but it weakens exactly the claim the
project is making about evaluation discipline.

**Q: Why is `notebooks/eda.ipynb` generated by `notebooks/build_eda.py` instead of hand-authored?**
A: A notebook is JSON, so its git diff is unreadable: a one-word change to a markdown cell shows
up alongside base64 image blobs. Generating it from a plain Python file means the analysis is
reviewable as source, regenerable after a feature change, and impossible to leave in a
half-executed state. The committed `.ipynb` is still a normal notebook -- Restart & Run All
works, outputs are saved so it renders on GitHub without the dataset, and a reader never needs
the generator. *Alternative:* commit the notebook alone and strip outputs -- rejected because a
reviewer cloning the repo should see the analysis immediately, without Kaggle credentials.

**Q: The notebook says a neural network probably will not win. Why build one then?**
A: Because the assignment requires a PyTorch classifier, and because the interesting result is
the comparison, not the winner. The classical 5-fold cross-validation at the end of the notebook
establishes the expectation band before any PyTorch is written, and it also shows that the
fold-to-fold spread (~0.08 ROC-AUC) is wider than the gap between a logistic regression and
gradient boosting. That is the finding: at n=712 these models are not reliably distinguishable,
which is why every number in this project is reported with a confidence interval.

### Implementation notes (added while building)

**Q: The registry stores `"dir": "fast"` rather than `"artifacts/fast"` as ARCHITECTURE.md
specified. Why the change?**
A: A test caught it. The hardcoded `artifacts/<name>` prefix breaks the moment anyone passes
`--artifacts-dir` to something not literally named `artifacts`, because the path was being
resolved against the artifacts directory's *parent*. Storing the directory relative to the
registry file means the whole tree can be moved, renamed or written anywhere and still resolve.
`artifacts.bundle_dir()` still accepts the old form by keeping only the leaf, so an artifact
produced before the change continues to load.

**Q: Why is the Compare tab's conclusion paragraph generated rather than written?**
A: Because a hand-written verdict goes stale the first time someone retrains. `honest_verdict()`
derives it from the metrics: it finds the leading model, checks which others fall inside its
95% interval, identifies the smallest model, and recommends the simplest model that is
statistically indistinguishable from the best. If a retrain reorders the models, the paragraph
reorders with them. It is also the project's central claim, so it should be computed from
evidence rather than asserted.

**Q: Why does `fast` sometimes beat `deep` and `attn`, and why leave that in the README?**
A: Because it is the result. With 712 training rows, 9 features and dominant low-order signal,
there is very little for extra capacity to learn, and the 95% intervals (roughly +/- 0.06 at
n=179) are three times wider than the spread between best and worst model. Reporting `fast` as
the model to ship is the defensible reading of those numbers. Hiding it by re-tuning against the
validation set would be the actual failure.

**Q: The service records metrics itself instead of using FastAPI middleware. Concretely, what
does that buy?**
A: Three things the load test makes visible. (1) The Streamlit app in local mode shows the same
latency, usage and queue numbers with no server running. (2) There is exactly one inference code
path, so app and API cannot drift. (3) The metrics describe *inference*, not HTTP: under load at
concurrency 16 the report reads `queue=154ms preprocess=9.9ms inference=16.6ms`, which says the
fix is capacity rather than a faster model. Middleware only sees total request time and could
not have told us that.

**Q: Why is `ds_app.py` only 169 lines when it renders six tabs?**
A: Because every tab body lives in `app/tabs.py`, the widgets in `app/components.py` and the
caching in `app/state.py`. The entry point does sidebar, data loading, one inference call and six
dispatches. Keeping it that short is what makes it reviewable, and it forced the tabs to become
independently testable functions rather than one long script.

---

## Change log

| date       | decision                                   | supersedes |
|------------|--------------------------------------------|------------|
| 2026-09-23 | Initial decisions above (model ladder, Windows/3.12, local-only, Plotly) | — |
| 2026-09-23 | Instrumented `InferenceService` + FastAPI in scope; `attn` deferred to "if ahead" | "4 models" priority |
| 2026-09-22 | Build on Python 3.14; pin pandas/scipy/scikit-learn below latest (Smart App Control); support `KGAT_` Kaggle tokens | "Python 3.12" rule |
| 2026-09-22 | EDA splits before exploring; notebook generated from `notebooks/build_eda.py` | — |
| 2026-09-22 | Registry stores bundle dirs relative to `registry.json`; Compare verdict generated from metrics | `"dir": "artifacts/<name>"` |
