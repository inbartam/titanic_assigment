# API.md: Inference Service & Observability

Scope: an HTTP inference API (`api/main.py`, FastAPI + uvicorn) in front of the same artifacts the
Streamlit app uses. It is instrumented with latency, usage, error and queue-depth metrics in
Prometheus format, plus a JSON `/stats` snapshot for the app's Ops tab.

Design rule: the service layer owns the metrics, not the web framework.
`titanic.service.InferenceService` wraps loaded bundles, enforces a bounded concurrency queue and
records every metric. FastAPI is a thin adapter, and the Streamlit app in local mode calls the
same service in-process. As a result the Ops tab works with or without a running server, and
there is exactly one code path for inference.

---

## 1. Components

```
src/titanic/
├── service.py      # InferenceService: load bundles, predict(df, model, threshold) -> PredictionResult
│                   #   - bounded queue (threading.Semaphore + queue counter), timeouts
│                   #   - records metrics via titanic.metrics
├── metrics.py      # MetricsRegistry: prometheus_client objects + ring buffer for percentiles
│                   #   - .snapshot() -> dict for /stats and the Ops tab
│                   #   - .prometheus_text() -> str for /metrics
└── schemas.py      # Pydantic: PassengerIn, PredictRequest, PredictResponse, EvaluateResponse,
                    #   StatsResponse, ErrorResponse (shared by API and app validation)

api/
├── main.py         # FastAPI app factory create_app(settings), routes, exception handlers
├── settings.py     # pydantic-settings: TITANIC_ARTIFACTS_DIR, TITANIC_MAX_CONCURRENCY=2,
│                   #   TITANIC_MAX_QUEUE=64, TITANIC_QUEUE_TIMEOUT_S=5, TITANIC_DEFAULT_MODEL
└── __init__.py

scripts/
└── load_test.py    # httpx+asyncio: N requests at concurrency C, prints p50/p95/p99, queue max
```

Run: `uvicorn api.main:app --host 127.0.0.1 --port 8000` (add `--workers 1`; metrics are
per-process, see §6).

---

## 2. Concurrency & queue model

CPU inference on small batches is fast (<10 ms) but not free, and uncontrolled concurrency on a
2- to 4-core laptop makes latency collapse. The service therefore works like this:

```
request ──► queue_depth += 1 ──► wait on Semaphore(MAX_CONCURRENCY) with QUEUE_TIMEOUT_S
                │                          │
                │ queue_depth >= MAX_QUEUE  │ acquired: queue_depth -= 1, inflight += 1
                ▼                          ▼
           503 + Retry-After        preprocess → model → postprocess → inflight -= 1
                                           (each stage timed separately)
```

- `MAX_CONCURRENCY` (default 2): parallel inferences. Torch releases the GIL inside ops, so 2
  gives real overlap on ≥4 cores; going higher mostly adds contention.
- `MAX_QUEUE` (default 64): requests allowed to wait. Beyond that the service returns
  `503 Service Unavailable` with `Retry-After: 1` and body
  `{"error": "queue_full", "queue_depth": 64, "max_queue": 64}`.
- `QUEUE_TIMEOUT_S` (default 5): a request that waits longer than this gets `503 queue_timeout`.
- Queue depth is measured as "requests that have arrived and are not yet executing". That is the
  number a load balancer or autoscaler would act on.
- In FastAPI, inference is dispatched with `run_in_threadpool`. The `anyio` thread limiter is set
  to `MAX_CONCURRENCY + MAX_QUEUE` so the threadpool itself does not turn into a hidden second
  queue.
- The in-process Streamlit path uses the same `threading.Semaphore`. With one user it never
  blocks, but the metrics are still recorded the same way.

---

## 3. Endpoints

All responses are JSON. Errors use one shape: `{"error": "<code>", "message": "<actionable text>",
"details": {...}}`. Stack traces are not sent to the client; they are logged server-side with a
request id.

| method | path                 | purpose                                                                 |
|--------|----------------------|-------------------------------------------------------------------------|
| GET    | `/health`            | `{"status":"ok","models_loaded":[...],"uptime_s":..}`; 503 if no models  |
| GET    | `/models`            | registry contents: name, framework, n_params, validation metrics, trained_at |
| POST   | `/predict`           | body `{"model": "deep", "threshold": 0.5, "passengers": [PassengerIn, ...]}` → per-row `{"passenger_id", "p_survived", "prediction"}` + `{"model", "threshold", "n", "latency_ms": {"queue", "preprocess", "inference", "total"}}` |
| POST   | `/predict/csv`       | multipart CSV upload, same query params `model`, `threshold`; returns CSV (predictions appended) or JSON via `Accept` |
| POST   | `/evaluate`          | like `/predict/csv` but requires `Survived`; returns `EvaluateResponse`: metrics, 95% bootstrap CIs (`n_boot` param, default 1000, capped 2000), confusion matrix, ROC/PR curve points |
| GET    | `/stats`             | JSON snapshot for dashboards (see §5)                                   |
| GET    | `/metrics`           | Prometheus exposition text                                              |
| POST   | `/admin/reload`      | reload artifacts from disk (guarded by `TITANIC_ADMIN_TOKEN` header; disabled if unset) |

`PassengerIn` mirrors the raw Kaggle schema (required: `Pclass, Name, Sex, Age, SibSp, Parch,
Fare`; optional: `PassengerId, Cabin, Embarked, Ticket, Survived`). It applies the same
validation as `data.validate_schema`, so the API and the app reject the same inputs with the same
messages. Request body limit: 10 000 rows (`413 payload_too_large` above).

Validation errors return `422` with the list of offending fields. An unknown model returns
`404 model_not_found` and lists the available models. Missing artifacts return `503 no_artifacts`
with the `train.py` command to run.

---

## 4. Metrics (Prometheus names)

| metric                                        | type      | labels                     | meaning                                              |
|-----------------------------------------------|-----------|----------------------------|------------------------------------------------------|
| `titanic_requests_total`                      | Counter   | `endpoint, model, status`  | usage; `status` = HTTP class (`2xx`,`4xx`,`5xx`)     |
| `titanic_request_duration_seconds`            | Histogram | `endpoint, model`          | end-to-end latency (buckets 1 ms … 5 s)              |
| `titanic_stage_duration_seconds`              | Histogram | `stage ∈ {queue, preprocess, inference, postprocess}, model` | where time goes           |
| `titanic_rows_predicted_total`                | Counter   | `model`                    | usage in rows, not requests                          |
| `titanic_batch_size`                          | Histogram | `model`                    | rows per request                                     |
| `titanic_inflight_requests`                   | Gauge     | `model`                    | currently executing                                  |
| `titanic_queue_depth`                         | Gauge     | (none)                     | waiting for a slot (the autoscaling signal)          |
| `titanic_queue_rejections_total`              | Counter   | `reason ∈ {full, timeout}` | back-pressure events                                 |
| `titanic_errors_total`                        | Counter   | `endpoint, error_code`     | `schema_error`, `model_not_found`, `internal`, …     |
| `titanic_prediction_positive_rate`            | Gauge     | `model`                    | mean predicted class over the last window; a cheap drift signal |
| `titanic_prediction_probability`              | Histogram | `model`                    | distribution of `p_survived` (0.0 to 1.0, 10 buckets) |
| `titanic_model_info`                          | Info      | `model, framework, n_params, trained_at, sklearn/torch version` | what is serving |
| `titanic_model_load_duration_seconds`         | Gauge     | `model`                    | cold-start cost                                      |
| `python_gc_*`                                 | default   | (none)                     | GC stats from `prometheus_client`                    |
| `process_*`                                   | default   | (none)                     | CPU and RSS from `prometheus_client`. Linux only: the collector reads `/proc` and emits nothing on Windows or macOS |

Histogram buckets for latency: `(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)`.

---

## 5. `/stats` snapshot (what the Ops tab renders)

Computed from `MetricsRegistry`. Counters are read from the Prometheus objects. Percentiles come
from a `collections.deque(maxlen=2000)` of `(ts, endpoint, model, total_ms, stage_ms…)` records,
so p50/p95/p99 are exact over the recent window. Prometheus histograms only give bucketed
estimates, and the app should not need a Prometheus server.

```json
{
  "uptime_s": 1234.5, "window_size": 2000, "window_seconds": 611.2,
  "requests": {"total": 1830, "per_endpoint": {"/predict": 1700, "/evaluate": 130},
               "per_model": {"fast": 400, "deep": 1200, "gbdt": 230}, "error_rate": 0.011,
               "rps_1m": 12.4},
  "rows_predicted": {"total": 184300, "per_model": {...}},
  "latency_ms": {"p50": 6.1, "p95": 18.7, "p99": 41.2, "max": 210.0,
                 "per_model": {"deep": {"p50": 5.9, "p95": 17.0}, ...},
                 "per_stage": {"queue": {"p50": 0.0, "p95": 3.1}, "preprocess": {...},
                               "inference": {...}, "postprocess": {...}}},
  "queue": {"depth": 0, "max_depth_window": 17, "inflight": 1,
            "rejections": {"full": 0, "timeout": 0}, "max_concurrency": 2, "max_queue": 64},
  "predictions": {"positive_rate_window": {"deep": 0.39}, "train_base_rate": 0.3838},
  "models": {"deep": {"framework": "torch", "n_params": 3457, "loaded_ms": 84}, ...}
}
```

The Streamlit Ops tab (see ARCHITECTURE §7) shows request/row counters, latency percentiles per
stage (Plotly bar), a live queue-depth/inflight gauge pair, error rate, and the positive-rate
drift line against the training base rate (0.3838). Process RSS/CPU is intentionally left out of
`/stats`. It is available on the Prometheus endpoint on Linux, and duplicating it as JSON would
have meant adding `psutil` for a single tile.

---

## 6. Operational notes (put in README)

- Metrics are per-process. Run `uvicorn --workers 1` (the default). Multi-worker support would
  need `prometheus_client.multiprocess`, which is out of scope and listed as future work.
- Structured logging: one JSON line per request (`request_id, endpoint, model, n_rows,
  status, total_ms, queue_ms`), with `X-Request-ID` echoed in responses.
- CORS: only `localhost` origins are allowed (the Streamlit app). There is no auth on read
  endpoints; the reload endpoint is token-guarded or disabled.
- Graceful shutdown: on SIGTERM, stop accepting new requests and drain in-flight ones
  (≤ `QUEUE_TIMEOUT_S`).
- Windows: uvicorn runs fine natively; `--reload` uses watchfiles (in requirements).

---

## 7. Streamlit ↔ API modes

| mode  | how                                                | inference path                       | Ops data           |
|-------|----------------------------------------------------|--------------------------------------|--------------------|
| local (default) | `streamlit run ds_app.py`                | in-process `InferenceService`        | `service.stats()`  |
| api   | `TITANIC_API_URL=http://127.0.0.1:8000 streamlit run ds_app.py` (or sidebar text box) | `httpx` client → `/predict/csv`, `/evaluate` | `GET /stats` |

The app shows which mode it is in (sidebar badge) and falls back to local mode with a warning if
the API is unreachable. Local mode covers the assignment's "load the trained model from disk"
requirement; API mode is extra.

---

## 8. Tests

`tests/test_service.py`
- queue depth gauge increments/decrements around a blocked slot (patch inference to sleep,
  fire 3 threads with `MAX_CONCURRENCY=1`, assert peak depth == 2).
- `MAX_QUEUE=1` + 3 concurrent → exactly one `QueueFullError`.
- stage timings sum ≈ total.

`tests/test_api.py` (FastAPI `TestClient`)
- `/health` 200 with models; `/models` lists registry.
- `/predict` single passenger → probability in [0,1], prediction consistent with threshold.
- `/predict` missing `Pclass` → 422 naming the field.
- `/predict?model=nope` → 404 listing models.
- `/evaluate` on `data/sample_train.csv` → metrics keys present, CI bounds ordered.
- `/predict/csv` without `Survived` → 200, no metrics, note in response.
- `/metrics` contains `titanic_requests_total` and `titanic_queue_depth`.
- `/stats` schema validates as `StatsResponse`.

`scripts/load_test.py --n 300 --concurrency 16 --model deep` is not a test, but the README shows
its output and the Ops-tab screenshot taken while it ran.
