# DECISIONS.md: Design Decision Log

Format: **Q** (what an interviewer would ask) → **A** (a one-paragraph answer we can defend),
with *alternatives considered* where relevant. Entries are appended, not rewritten. If a decision
changes, a new entry supersedes the old one.

---

### Data & split

**Q: Why an 80/20 stratified split instead of only cross-validation?**
A: The assignment asks for a held-out validation set and for the app to evaluate on it. We keep
20% (n=179) untouched for a single final evaluation and do all model selection with 5-fold
stratified CV inside the 80%. Choosing and reporting happen on different data, so the reported
numbers are not optimistically biased by selection. *Alternative:* nested CV. It is more rigorous
but too much for a 1-day task; noted as a limitation.

**Q: Why stratify?**
A: The positive rate is 38%. On n=179, an unstratified split can shift the class ratio by several
points and move accuracy by ~1 to 2 points from sampling alone. Stratifying removes that source of
noise.

**Q: Why commit a 100-row sample?**
A: The assignment asks for a `data/` sample. It also makes the app demo-able and the smoke test
runnable without Kaggle credentials, which matters for a reviewer who clones the repo and wants to
see the app within a minute. The full `train.csv` is git-ignored and fetched programmatically.

### Features

**Q: Why not use ticket-group size / fare-per-person? They are among the strongest Kaggle features.**
A: They are computed over the batch (the count of rows sharing a ticket). At inference on a single
passenger the value is always 1, and on a partial CSV it differs from training, so the feature's
meaning depends on which other rows happen to be in the file. That is train/serve skew, and in the
training split it also leaks group-level survival information across rows. We excluded them on
purpose and say so in the EDA. *Alternative:* fit a ticket→count lookup on the training split and
default unseen tickets to 1. That is leak-safe but semantically brittle.

**Q: Why Title?**
A: It compresses sex × age × social status into 5 levels and is the best available signal for
imputing Age (Master ≈ 4 y vs Mr ≈ 32 y). It is extracted with a regex that handles the Kaggle
name format. Anything outside the 4 common titles maps to `Rare`, so unseen titles at inference
are handled without error.

**Q: Why impute Age by Title median rather than a global median or a model?**
A: A global median (28) gives adult ages to children whose title is `Master`, which wipes out the
"children first" signal. Title-median is a one-line fitted lookup, easy to inspect, and serialized
in `preprocessor.json`. A regression imputer would add complexity and another place for leakage,
for little gain.

**Q: Why drop SibSp/Parch after making FamilySize?**
A: Together they are linearly determined by FamilySize, and the EDA showed the survival pattern
depends on total family size. Keeping all three adds collinearity for the linear model without
adding information. *Alternative:* keep them for the MLP only. Rejected so there is one feature
set.

**Q: Deck has 77% missing. Why keep it?**
A: The missingness is informative in itself (no recorded cabin ≈ lower class ≈ lower survival).
We encode NaN as its own level `U`, which lets every model use the missingness pattern without a
separate `HasCabin` flag.

**Q: Why treat Pclass as categorical?**
A: Its effect is not linear (1st ≫ 2nd > 3rd, with uneven gaps). Three levels cost nothing.

### Preprocessing

**Q: How do you guarantee no leakage?**
A: `Preprocessor.fit` is called exactly once, on the training split, in `train.py`. Its state is
serialized to JSON and reloaded for validation and for the app. One unit test asserts that
transforming validation data does not change any fitted parameter, and another asserts that
save→load→transform is bit-identical.

**Q: Why JSON artifacts instead of pickling a sklearn `ColumnTransformer`?**
A: JSON is readable in a code review, does not depend on library versions, and forces us to list
every learned parameter explicitly. Pickle would hide the preprocessing state and tie inference to
a specific sklearn version.

### Models

**Q: Why four models instead of one?**
A: The assignment rewards "originality and problem-solving", and on Titanic the interesting
question is less which model wins than whether the models can be told apart at n=179. The four
models are logistic regression in PyTorch (`fast`), an MLP with embeddings (`deep`), a tiny
FT-Transformer-style attention model (`attn`) and gradient boosting (`gbdt`). Because they share
the preprocessor and artifact contract, the app can show that comparison with confidence
intervals. It also shows that the training/artifact/inference pipeline does not depend on the
model. The required deliverable is still the PyTorch models. `gbdt` is the classical reference,
and it is selectable so the reviewer can see it in the app rather than just read about it.

**Q: Isn't four models over-engineering for a one-day task?**
A: Each one adds about 15 lines of model code and no pipeline code, because the contract is
shared. The real risk is time, so `attn` had a hard 30-minute box and would be the first thing
cut. If it is missing from the repo, that is why.

**Q: Why implement logistic regression in PyTorch rather than sklearn?**
A: It uses the same loss, optimizer, batches, seed and preprocessing as the MLP, so any gap can
be attributed to the architecture rather than the training recipe. It is also a ~25-parameter,
fully interpretable model that trains in seconds, which is a sensible default for a dataset this
size.

**Q: Why was `attn` deferred behind the API?**
A: We had ten hours. A fourth model on 712 rows gives us something to discuss; an instrumented
service gives the reviewer something to run. If `attn` is in the repo, it was built in the last
30 minutes after everything else passed the fresh-clone test.

**Q: Why a transformer on 9 features?**
A: We did not expect it to win, and on Titanic it usually won't. It is there for two reasons.
First, attention over feature tokens is the standard modern tabular-DL architecture, and we wanted
to show we can implement it correctly at a small size (~7k params). Second, it shows how a more
expressive model behaves at n=712: higher variance, with a CI that overlaps the linear model's. We
think that says more about judgment than a big model with a suspiciously good number.

**Q: Why HistGradientBoosting and why is it saved with joblib when everything else is JSON?**
A: HGB handles categorical features natively via a mask, so it consumes exactly the same
`(X_num, X_cat)` as the torch models and needs no second preprocessing path. sklearn has no clean
JSON serialization. joblib, with the pinned sklearn version recorded in `model_config.json`, is
the standard approach. It is the only non-JSON artifact in the repo, and we call that out.

**Q: Why embeddings in the MLP rather than one-hot?**
A: Mostly to show that categorical handling is done properly; with cardinalities ≤ 10 the
parameter difference is negligible. The dimension `min(8, ceil(card/2))` follows a common rule of
thumb. One-hot would be just as defensible.

**Q: Why is the MLP so small?**
A: There are 712 training rows. A 64→32 MLP with dropout already has ~3 to 4k parameters (~5 per
sample). Anything larger overfits faster and adds nothing. Weight decay, dropout and early
stopping are the cheap regularizers that matter here.

**Q: How was early stopping done without touching the validation set?**
A: A 10% stratified carve-out inside the training split serves as the early-stopping set. The
held-out 20% is not seen during training or selection.

**Q: How did you choose hyperparameters?**
A: With an 8-point grid (hidden size, dropout, weight decay) scored by mean ROC-AUC over 5
stratified folds of the training split. The grid is small on purpose. With n=712 the CV standard
deviation (~0.03 AUC) is larger than most between-config differences, so a large search would
mostly be fitting noise. The full grid results are stored in `history.json` and shown in the
notebook.

**Q: What if `gbdt` beats every PyTorch model?**
A: It often does, by about 1 point. Trees model interactions and thresholds natively, which suits
small tabular data with mixed types, while NNs need more data to learn those from scratch. We show
it in the app with CIs. The PyTorch models are the required deliverable, and the comparison is
there to show our reasoning openly. The recommendation paragraph in the README and the Compare tab
is derived automatically from the numbers.

### Evaluation

**Q: Why bootstrap confidence intervals?**
A: With n=179, an accuracy of 0.83 has a 95% CI of roughly ±0.055, so a bare point estimate to
three decimals would be misleading. 1000 stratified bootstrap resamples of the validation set give
an interval for every metric at negligible cost, and let us say plainly whether two models can be
distinguished.

**Q: Why ROC-AUC *and* PR-AUC?**
A: ROC-AUC measures ranking quality independent of threshold and class balance. PR-AUC is more
sensitive to performance on the minority (survived) class, which is what a user of this model
would care about. F1 at the threshold complements both.

**Q: Why threshold 0.5? Why a slider?**
A: 0.5 is the Bayes decision rule for a calibrated model under symmetric costs, and no cost
asymmetry is stated. The slider is there to show the precision/recall trade-off, not to tune it.
The caption says so, and we never optimize a threshold on validation and then report it.

### Service & observability

**Q: The assignment asks for Streamlit. Why is there also a FastAPI service?**
A: "Evaluate on the validation set and run inference" describes a product, and an interviewer
judging "originality, robustness, error handling" is asking how it would behave in production. An
HTTP API with schema validation, bounded concurrency and metrics is the smallest credible answer.
It is purely additive: Streamlit runs in-process with no server, as the assignment specifies, and
API mode is a switch.

**Q: Why put the metrics in `InferenceService` instead of FastAPI middleware?**
A: Middleware only sees the HTTP path. Putting queue accounting and stage timers in the service
means (1) the Streamlit app in local mode reports the same latency/usage/queue numbers, (2) there
is one inference code path, and (3) the metrics describe inference rather than HTTP. The split
between preprocessing time and model time is the number a data scientist actually wants.

**Q: What exactly is "queue depth" here and why does it matter?**
A: It is the number of requests that have arrived but are not yet executing, i.e. waiting on the
concurrency semaphore. It is the autoscaling and back-pressure signal. In-flight count saturates
at `max_concurrency` and tells you nothing once you are busy, while queue depth keeps growing and
tells you how busy. We also bound it (`max_queue`) and reject with 503 + `Retry-After` instead of
letting latency grow without limit.

**Q: Why `max_concurrency=2` on CPU?**
A: Batch inference of a few thousand rows takes milliseconds. Two parallel slots overlap I/O and
GIL-released torch ops on a 4-core laptop without thrashing. It is a setting, not a constant, and
the load test in the README shows the p95 curve that justified it.

**Q: Why both Prometheus `/metrics` and a JSON `/stats`?**
A: `/metrics` is the standard scrape target (histograms, counters, process metrics) for a real
deployment. `/stats` exists because the Streamlit Ops tab needs exact p50/p95/p99 over a recent
window without running a Prometheus server, and a 2000-record ring buffer provides that cheaply.
Both come from the same `MetricsRegistry`.

**Q: Why a prediction positive-rate metric?**
A: It is about the cheapest drift signal available. If the served positive rate moves far from
the training base rate (0.38), the input distribution has probably changed. Full drift detection
(feature-level PSI, etc.) is future work.

**Q: What did you deliberately leave out of the API?**
A: Auth, per-client rate limiting, multi-worker metrics aggregation, async model execution on GPU,
request batching across clients, and OpenTelemetry traces. Each is a well-known next step and none
of them changes the design. They are listed under future work in the README.

### Engineering

**Q: Why commit `artifacts/`?**
A: They are < 1 MB. They make the app runnable right after `git clone` without Kaggle credentials
(delivery is local-only, so the reviewer's first step is `streamlit run`), and they make the
submitted results verifiable. `train.py` regenerates them deterministically.

**Q: Why Python 3.12 when the machine has 3.13/3.14?**
A: torch/streamlit/kaggle wheels on Windows CPU are safest on 3.11/3.12. Under a one-day
deadline, working first time on the reviewer's machine matters more than having the newest
interpreter. The `py` launcher lets 3.12 sit side by side with other versions without conflict.

**Q: Why Plotly instead of matplotlib?**
A: The app is where the reviewer spends most of their time, and interactive ROC/PR overlays with
legend toggles make a four-model comparison readable. One `plots.py` produces both the app
figures and the HTML files `train.py` saves, so there is still a single plotting implementation.
The notebook uses matplotlib/seaborn because static figures render on GitHub.

**Q: Is training bit-for-bit reproducible?**
A: On the same machine and torch version, yes (seeded RNGs, CPU, deterministic algorithms, seeded
DataLoader generator). Across OS or torch versions, floating-point kernels can differ in the last
bits, and metrics reproduce to about 3 decimals. The README says this.

**Q: Why do plotting functions live in `src/` and not in the app?**
A: One implementation serves both `train.py` (which saves HTML to `artifacts/<model>/plots/`) and
the app (`st.plotly_chart`). There is no duplicated plotting code, and the figures are identical
in both places.

### Environment (added during implementation)

**Q: The docs mandate Python 3.12. Why does the repository build on 3.14?**
A: The development machine only had 3.14 installed. Rather than assume, we tested it: a 3.14 venv
resolves CPU wheels for `torch 2.14.0+cpu`, `streamlit`, `plotly`, `fastapi` and the rest of the
stack without trouble, so the reason for pinning 3.12 (wheel availability on Windows) no longer
applies. `requires-python` stays at `>=3.11`, so a reviewer on 3.11 or 3.12 is unaffected.
*Alternative:* install 3.12 side by side via the `py` launcher. Rejected because it would have
cost setup time to avoid a problem that testing showed does not exist.

**Q: Why are `pandas`, `scipy` and `scikit-learn` pinned below their newest releases?**
A: Windows Smart App Control is enabled on the development machine
(`VerifiedAndReputablePolicyState = 1`). It blocks native `.pyd` files that have not yet built up
cloud reputation, and it blocked them at *import* time, not install time. `pandas 3.0.6`
(`ccalendar`), `scipy 1.18.1` (`_csparsetools`) and `scikit-learn 1.9.1` (`_argkmin`) all
installed successfully and then failed with `DLL load failed ... An Application Control policy
has blocked this file`. Pinning `pandas==2.3.3`, `scipy==1.16.2` and `scikit-learn==1.7.2`, all
mature and widely deployed builds, fixes it completely. *Alternative:* disable Smart App Control.
That cannot be undone without reinstalling Windows, and a project should not ask it of a
reviewer's machine. The pins are reasonable practice anyway: pandas 3.0 also brings breaking API
changes we had no reason to take on during a one-day build.

**Q: Why does `data.py` support four Kaggle credential formats instead of just `kaggle.json`?**
A: Kaggle now issues a `KGAT_`-prefixed access token instead of the classic username/key
`kaggle.json`. The `kaggle` Python client reads the `KAGGLE_API_TOKEN` environment variable but
does *not* read the `~/.kaggle/access_token` file that Kaggle's own setup snippet writes.
`_load_kaggle_credentials` therefore checks the env var, then that file (copying it into the env
var), then the classic env vars, then `kaggle.json`. If all four are absent it raises one
`KaggleAuthError` that lists every remedy. A reviewer with either kind of token can fetch the data
without reading the source.

### EDA (added during implementation)

**Q: Why does the EDA notebook create the train/validation split before exploring, rather than
exploring `train.csv` as a whole?**
A: Looking at the held-out rows biases the analyst even when no code touches them. Every decision
in the notebook (impute Age by Title, treat Pclass as categorical, drop SibSp/Parch, exclude
ticket-group features) is a modelling choice informed by what the plots showed. If those plots
had included the 179 validation rows, the final "single, unbiased evaluation" would be scoring a
pipeline that was partly designed on the data it is scored against. So the notebook splits in its
first analysis cell and runs `del val_raw`, and a test
(`tests/test_notebook.py::test_notebook_discards_the_validation_split`) asserts it. This is
stricter than most Titanic notebooks and costs nothing. *Alternative:* explore everything and rely
on the split only at training time. That is common practice, but it weakens the very claim this
project makes about evaluation discipline.

**Q: Why is `notebooks/eda.ipynb` generated by `notebooks/build_eda.py` instead of hand-authored?**
A: A notebook is JSON, so its git diff is unreadable; a one-word change to a markdown cell shows
up next to base64 image blobs. Generating it from a plain Python file means the analysis can be
reviewed as source, regenerated after a feature change, and never left half-executed. The
committed `.ipynb` is still a normal notebook. Restart & Run All works, outputs are saved so it
renders on GitHub without the dataset, and a reader never needs the generator. *Alternative:*
commit the notebook alone and strip outputs. Rejected because a reviewer cloning the repo should
see the analysis immediately, without Kaggle credentials.

**Q: The notebook says a neural network probably will not win. Why build one then?**
A: The assignment requires a PyTorch classifier, and the comparison is more interesting than the
winner. The classical 5-fold cross-validation at the end of the notebook sets the expectation band
before any PyTorch is written. It also shows that the fold-to-fold spread (~0.08 ROC-AUC) is wider
than the gap between a logistic regression and gradient boosting. That is the finding: at n=712
these models cannot be reliably told apart, which is why every number in this project comes with
a confidence interval.

### Implementation notes (added while building)

**Q: The registry stores `"dir": "fast"` rather than `"artifacts/fast"` as ARCHITECTURE.md
specified. Why the change?**
A: A test caught it. The hardcoded `artifacts/<name>` prefix breaks as soon as anyone passes
`--artifacts-dir` pointing at a directory not literally named `artifacts`, because the path was
being resolved against the artifacts directory's parent. Storing the directory relative to the
registry file means the whole tree can be moved, renamed or written anywhere and still resolve.
`artifacts.bundle_dir()` still accepts the old form by keeping only the leaf, so artifacts
produced before the change still load.

**Q: Why is the Compare tab's conclusion paragraph generated rather than written?**
A: A hand-written verdict goes stale the first time someone retrains. `honest_verdict()` derives
it from the metrics. It finds the leading model, checks which others fall inside its 95%
interval, identifies the smallest model, and recommends the simplest model that is statistically
indistinguishable from the best. If a retrain reorders the models, the paragraph reorders with
them. It is also the project's central claim, so it should be computed from evidence rather than
asserted.

**Q: Why does `fast` sometimes beat `deep` and `attn`, and why leave that in the README?**
A: Because that is the result. With 712 training rows, 9 features and a signal dominated by
low-order effects, there is very little for extra capacity to learn. The 95% intervals (roughly
+/- 0.06 at n=179) are three times wider than the spread between the best and worst model.
Recommending `fast` is the defensible reading of those numbers. Hiding it by re-tuning against the
validation set would be the real mistake.

**Q: The service records metrics itself instead of using FastAPI middleware. Concretely, what
does that buy?**
A: Three things, all visible in the load test. (1) The Streamlit app in local mode shows the same
latency, usage and queue numbers with no server running. (2) There is exactly one inference code
path, so the app and the API cannot drift apart. (3) The metrics describe inference rather than
HTTP. Under load at concurrency 16 the report reads `queue=154ms preprocess=9.9ms
inference=16.6ms`, which says the fix is capacity rather than a faster model. Middleware only sees
total request time and could not have told us that.

**Q: Why is `ds_app.py` only 169 lines when it renders six tabs?**
A: Every tab body lives in `app/tabs.py`, the widgets in `app/components.py` and the caching in
`app/state.py`. The entry point handles the sidebar, data loading, one inference call and six
dispatches. Keeping it that short makes it easy to review, and it pushed the tabs into separate
functions that can be tested on their own.

---

## Change log

| date       | decision                                   | supersedes |
|------------|--------------------------------------------|------------|
| 2026-09-23 | Initial decisions above (model ladder, Windows/3.12, local-only, Plotly) | (none) |
| 2026-09-23 | Instrumented `InferenceService` + FastAPI in scope; `attn` deferred to "if ahead" | "4 models" priority |
| 2026-09-22 | Build on Python 3.14; pin pandas/scipy/scikit-learn below latest (Smart App Control); support `KGAT_` Kaggle tokens | "Python 3.12" rule |
| 2026-09-22 | EDA splits before exploring; notebook generated from `notebooks/build_eda.py` | (none) |
| 2026-09-22 | Registry stores bundle dirs relative to `registry.json`; Compare verdict generated from metrics | `"dir": "artifacts/<name>"` |
