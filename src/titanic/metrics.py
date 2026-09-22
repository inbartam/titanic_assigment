"""Observability: Prometheus instruments plus an exact-percentile ring buffer.

Two consumers, one source of truth:

* ``/metrics`` exposes Prometheus text for a real scrape target.
* ``/stats`` (and the Streamlit Ops tab) reads :meth:`MetricsRegistry.snapshot`,
  which computes p50/p95/p99 exactly over a recent window.

Both exist because they answer different questions. Prometheus histograms give
bucketed estimates and need a Prometheus server; the app needs precise recent
percentiles with no extra infrastructure, which a 2000-record
``collections.deque`` provides for free.

Crucially the registry is owned by the *service*, not by the web framework. The
same numbers are recorded whether a request arrived over HTTP or was made
in-process by Streamlit, so the Ops tab works with no server running.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from titanic.config import TRAIN_BASE_RATE
from titanic.utils import get_logger

logger = get_logger(__name__)

#: Records kept for exact percentiles. 2000 requests at a few hundred bytes
#: each is well under a megabyte, and covers any interactive session.
WINDOW_SIZE = 2000

#: Latency buckets in seconds, from 1 ms to 5 s. Chosen around the observed
#: range: a warm single-row prediction is ~2 ms, a 1000-row evaluation with
#: bootstrap intervals is a few hundred.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

#: The four stages every inference request passes through. Splitting them is
#: the point: "the model is slow" and "preprocessing is slow" call for
#: completely different fixes.
STAGES: tuple[str, ...] = ("queue", "preprocess", "inference", "postprocess")


@dataclass
class RequestRecord:
    """One completed request, retained in the ring buffer.

    Attributes:
        timestamp: Unix time when the request finished.
        endpoint: Logical endpoint name.
        model: Model used.
        status: HTTP-style class (``2xx``, ``4xx``, ``5xx``).
        n_rows: Rows predicted.
        total_ms: End-to-end duration.
        stages_ms: Per-stage durations.
        positive_rate: Fraction predicted positive, for drift tracking.
    """

    timestamp: float
    endpoint: str
    model: str
    status: str
    n_rows: int
    total_ms: float
    stages_ms: dict[str, float] = field(default_factory=dict)
    positive_rate: float | None = None


def _percentile(values: list[float], fraction: float) -> float:
    """Return a percentile using nearest-rank on a sorted list.

    Implemented directly rather than via numpy: this runs inside request
    handling, the lists are short, and nearest-rank is unambiguous about which
    observed value it returns (no interpolation between two real requests).

    Args:
        values: Sorted-or-not list of measurements.
        fraction: Percentile as a fraction, e.g. ``0.95``.

    Returns:
        The percentile, or 0.0 for an empty list.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return round(ordered[index], 3)


def _summarise(values: list[float]) -> dict[str, float]:
    """Summarise a list of latencies.

    Args:
        values: Durations in milliseconds.

    Returns:
        Dict with ``p50``, ``p95``, ``p99``, ``max``, ``mean`` and ``count``.
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "count": len(values),
    }


class MetricsRegistry:
    """Prometheus instruments plus a recent-request window.

    Each instance owns a private :class:`CollectorRegistry` rather than using
    the process-global default. Without that, constructing two registries in
    one process -- which every test that builds a fresh service does -- raises
    a duplicate-timeseries error.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        """Create the instruments and the ring buffer.

        Args:
            window_size: Requests retained for exact percentiles.
        """
        self.registry = CollectorRegistry()
        self.started_at = time.time()

        # A deque with maxlen evicts the oldest record automatically, so the
        # window is bounded without any cleanup code.
        self._window: deque[RequestRecord] = deque(maxlen=window_size)
        # The buffer is appended from request threads and read by /stats, so
        # every touch is guarded. deque.append is atomic, but the multi-step
        # reads in snapshot() are not.
        self._lock = threading.Lock()

        self._max_queue_depth_seen = 0
        self._model_info: dict[str, dict[str, Any]] = {}

        self.requests_total = Counter(
            "titanic_requests_total",
            "Requests served",
            ["endpoint", "model", "status"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "titanic_request_duration_seconds",
            "End-to-end request duration",
            ["endpoint", "model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.stage_duration = Histogram(
            "titanic_stage_duration_seconds",
            "Duration of each inference stage",
            ["stage", "model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.rows_predicted = Counter(
            "titanic_rows_predicted_total",
            "Rows predicted (usage in rows, not requests)",
            ["model"],
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "titanic_batch_size",
            "Rows per request",
            ["model"],
            buckets=(1, 5, 10, 50, 100, 500, 1000, 5000, 10000),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "titanic_inflight_requests",
            "Requests currently executing",
            registry=self.registry,
        )
        # The autoscaling signal: requests that have arrived but are *waiting*.
        # In-flight saturates at max_concurrency and stops being informative
        # the moment the service is busy; queue depth keeps growing.
        self.queue_depth = Gauge(
            "titanic_queue_depth",
            "Requests waiting for an execution slot",
            registry=self.registry,
        )
        self.queue_rejections = Counter(
            "titanic_queue_rejections_total",
            "Requests rejected by back-pressure",
            ["reason"],
            registry=self.registry,
        )
        self.errors_total = Counter(
            "titanic_errors_total",
            "Errors by type",
            ["endpoint", "error_code"],
            registry=self.registry,
        )
        self.prediction_positive_rate = Gauge(
            "titanic_prediction_positive_rate",
            "Mean predicted class over the recent window (drift signal)",
            ["model"],
            registry=self.registry,
        )
        self.prediction_probability = Histogram(
            "titanic_prediction_probability",
            "Distribution of predicted survival probability",
            ["model"],
            buckets=tuple(round(0.1 * i, 1) for i in range(1, 11)),
            registry=self.registry,
        )
        self.model_load_duration = Gauge(
            "titanic_model_load_duration_seconds",
            "Cold-start cost per model",
            ["model"],
            registry=self.registry,
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_model_loaded(self, name: str, seconds: float, info: dict[str, Any]) -> None:
        """Record that a model finished loading.

        Args:
            name: Model name.
            seconds: Load duration.
            info: Framework, parameter count and training timestamp.
        """
        self.model_load_duration.labels(model=name).set(seconds)
        self._model_info[name] = {**info, "loaded_ms": round(seconds * 1000, 2)}

    def record_request(
        self,
        *,
        endpoint: str,
        model: str,
        status: str,
        n_rows: int,
        total_ms: float,
        stages_ms: dict[str, float] | None = None,
        probabilities: list[float] | None = None,
    ) -> None:
        """Record one completed request in both the instruments and the window.

        Args:
            endpoint: Logical endpoint name.
            model: Model used, or ``"none"`` when the request failed earlier.
            status: ``2xx``, ``4xx`` or ``5xx``.
            n_rows: Rows predicted.
            total_ms: End-to-end duration in milliseconds.
            stages_ms: Per-stage durations in milliseconds.
            probabilities: Predicted probabilities, for the drift signal.
        """
        stages_ms = stages_ms or {}

        self.requests_total.labels(endpoint=endpoint, model=model, status=status).inc()
        self.request_duration.labels(endpoint=endpoint, model=model).observe(total_ms / 1000)
        for stage, milliseconds in stages_ms.items():
            self.stage_duration.labels(stage=stage, model=model).observe(milliseconds / 1000)

        positive_rate: float | None = None
        if n_rows:
            self.rows_predicted.labels(model=model).inc(n_rows)
            self.batch_size.labels(model=model).observe(n_rows)

        if probabilities:
            # Sample rather than observe every row: a 10 000-row request would
            # otherwise dominate the histogram and cost more than the inference.
            for probability in probabilities[:200]:
                self.prediction_probability.labels(model=model).observe(probability)
            positive_rate = sum(1 for p in probabilities if p >= 0.5) / len(probabilities)
            self.prediction_positive_rate.labels(model=model).set(positive_rate)

        with self._lock:
            self._window.append(
                RequestRecord(
                    timestamp=time.time(),
                    endpoint=endpoint,
                    model=model,
                    status=status,
                    n_rows=n_rows,
                    total_ms=total_ms,
                    stages_ms=dict(stages_ms),
                    positive_rate=positive_rate,
                )
            )

    def record_error(self, endpoint: str, error_code: str) -> None:
        """Increment the error counter.

        Args:
            endpoint: Where the error occurred.
            error_code: Machine-readable code such as ``schema_error``.
        """
        self.errors_total.labels(endpoint=endpoint, error_code=error_code).inc()

    def record_rejection(self, reason: str) -> None:
        """Record a back-pressure rejection.

        Args:
            reason: ``"full"`` or ``"timeout"``.
        """
        self.queue_rejections.labels(reason=reason).inc()

    def set_queue_depth(self, depth: int) -> None:
        """Publish the current queue depth and track its peak.

        Args:
            depth: Requests currently waiting for a slot.
        """
        self.queue_depth.set(depth)
        # The peak is what makes the Ops tab meaningful: a gauge sampled after
        # a load test has almost always fallen back to zero.
        if depth > self._max_queue_depth_seen:
            self._max_queue_depth_seen = depth

    def set_inflight(self, count: int) -> None:
        """Publish the number of requests currently executing.

        Args:
            count: In-flight request count.
        """
        self.inflight.set(count)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def prometheus_text(self) -> bytes:
        """Render the Prometheus exposition format.

        Returns:
            The scrape payload for ``GET /metrics``.
        """
        return generate_latest(self.registry)

    def snapshot(self, queue_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the JSON summary behind ``/stats`` and the Ops tab.

        Args:
            queue_state: Live queue figures from the service.

        Returns:
            A nested dict of counters, latency percentiles, queue state,
            prediction drift and per-model information.
        """
        with self._lock:
            records = list(self._window)

        uptime = time.time() - self.started_at
        total = len(records)
        errors = sum(1 for r in records if r.status != "2xx")

        per_endpoint: dict[str, int] = {}
        per_model: dict[str, int] = {}
        rows_per_model: dict[str, int] = {}
        latency_per_model: dict[str, list[float]] = {}
        latency_per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
        positive_rates: dict[str, list[float]] = {}

        for record in records:
            per_endpoint[record.endpoint] = per_endpoint.get(record.endpoint, 0) + 1
            per_model[record.model] = per_model.get(record.model, 0) + 1
            rows_per_model[record.model] = rows_per_model.get(record.model, 0) + record.n_rows
            latency_per_model.setdefault(record.model, []).append(record.total_ms)
            for stage, milliseconds in record.stages_ms.items():
                latency_per_stage.setdefault(stage, []).append(milliseconds)
            if record.positive_rate is not None:
                positive_rates.setdefault(record.model, []).append(record.positive_rate)

        recent = [r for r in records if r.timestamp > time.time() - 60]
        window_seconds = records[-1].timestamp - records[0].timestamp if len(records) > 1 else 0.0

        return {
            "uptime_s": round(uptime, 1),
            "window_size": total,
            "window_seconds": round(window_seconds, 1),
            "requests": {
                "total": total,
                "per_endpoint": per_endpoint,
                "per_model": per_model,
                "errors": errors,
                "error_rate": round(errors / total, 4) if total else 0.0,
                "rps_1m": round(len(recent) / 60, 2),
            },
            "rows_predicted": {
                "total": sum(rows_per_model.values()),
                "per_model": rows_per_model,
            },
            "latency_ms": {
                **_summarise([r.total_ms for r in records]),
                "per_model": {
                    name: _summarise(values) for name, values in latency_per_model.items()
                },
                "per_stage": {
                    stage: _summarise(values)
                    for stage, values in latency_per_stage.items()
                    if values
                },
            },
            "queue": {
                **(queue_state or {}),
                "max_depth_window": self._max_queue_depth_seen,
                "rejections": self._rejection_counts(),
            },
            "predictions": {
                "positive_rate_window": {
                    name: round(sum(values) / len(values), 4)
                    for name, values in positive_rates.items()
                },
                "train_base_rate": TRAIN_BASE_RATE,
            },
            "models": dict(self._model_info),
        }

    def _rejection_counts(self) -> dict[str, int]:
        """Read the rejection counter back out of Prometheus.

        Returns:
            Mapping from reason to count.
        """
        counts = {"full": 0, "timeout": 0}
        for metric in self.registry.collect():
            if metric.name != "titanic_queue_rejections":
                continue
            for sample in metric.samples:
                reason = sample.labels.get("reason")
                if sample.name.endswith("_total") and reason in counts:
                    counts[reason] = int(sample.value)
        return counts

    def reset_window(self) -> None:
        """Clear the recent-request window, keeping the Prometheus counters.

        Prometheus counters are monotonic by definition and must never be
        reset; only the app's recent-percentile window is cleared.
        """
        with self._lock:
            self._window.clear()
        self._max_queue_depth_seen = 0
