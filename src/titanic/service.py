"""The inference service: the only object that touches a model at serve time.

Both adapters -- ``app/client.LocalPredictor`` for Streamlit and ``api/main.py``
for FastAPI -- go through this class. That is deliberate: one inference code
path means the app and the API cannot drift, and the metrics describe
*inference* rather than *HTTP*, so preprocessing time and model time are
visible separately whether or not a server is running.

Concurrency model (``docs/API.md`` section 2)::

    request -> queue_depth += 1 -> wait on Semaphore(max_concurrency)
                    |                       |
      depth >= max_queue                acquired: depth -= 1, inflight += 1
                    v                       v
        QueueFullError (503)        preprocess -> model -> postprocess

Queue depth counts requests that have arrived and are **not yet executing**.
That is the number a load balancer or autoscaler acts on: in-flight saturates
at ``max_concurrency`` and tells you nothing once the service is busy, while
queue depth keeps growing and tells you how far behind it is.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from titanic.artifacts import (
    Bundle,
    ModelNotFoundError,
    NoArtifactsError,
    bundle_dir,
    load_bundle,
    load_registry,
)
from titanic.config import Paths
from titanic.data import SchemaError, validate_schema
from titanic.evaluation import bootstrap_ci, compute_metrics, curve_data
from titanic.features import engineer
from titanic.metrics import MetricsRegistry
from titanic.utils import get_logger, timer

logger = get_logger(__name__)

#: Refuse requests larger than this. A 10 000-row CSV takes well under a
#: second; beyond that the caller should batch, and an unbounded request is a
#: denial-of-service vector.
MAX_ROWS = 10_000


class QueueFullError(RuntimeError):
    """Raised when too many requests are already waiting for a slot.

    Maps to HTTP 503 with ``Retry-After``. Rejecting immediately is better
    than letting latency grow without bound: a client that knows it was
    rejected can retry or shed load, whereas one waiting 40 seconds cannot.
    """


class QueueTimeoutError(RuntimeError):
    """Raised when a request waited longer than ``queue_timeout_s`` for a slot."""


@dataclass
class PredictionResult:
    """The outcome of one prediction request.

    Attributes:
        model: Model used.
        threshold: Decision threshold applied.
        probabilities: ``P(survived)`` per row.
        predictions: Binary predictions per row.
        passenger_ids: Echoed ``PassengerId`` values when the CSV had them.
        n: Number of rows.
        latency_ms: Per-stage and total durations.
    """

    model: str
    threshold: float
    probabilities: np.ndarray
    predictions: np.ndarray
    passenger_ids: list[Any] | None = None
    n: int = 0
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_frame(self, source: pd.DataFrame | None = None) -> pd.DataFrame:
        """Assemble a results dataframe for display and CSV download.

        Args:
            source: The original dataframe, so identifying columns can be
                echoed back beside each prediction.

        Returns:
            A dataframe with ``p_survived`` and ``prediction`` plus whichever
            of ``PassengerId``, ``Name``, ``Sex``, ``Pclass``, ``Age`` and
            ``Survived`` were present in the input.
        """
        frame = pd.DataFrame(
            {
                "p_survived": np.round(self.probabilities, 4),
                "prediction": self.predictions.astype(int),
            }
        )
        if source is not None:
            # Identifying columns first, so the table reads left to right from
            # "who is this passenger" to "what did the model say".
            for column in ("PassengerId", "Name", "Sex", "Pclass", "Age", "Survived"):
                if column in source.columns:
                    frame.insert(0, column, source[column].to_numpy())
        return frame


@dataclass
class EvaluationResult:
    """The outcome of one evaluation request.

    Attributes:
        model: Model used.
        threshold: Decision threshold applied.
        metrics: Point estimates.
        ci95: Bootstrap intervals.
        curves: Arrays for the figures.
        n: Number of rows scored.
        latency_ms: Per-stage and total durations.
    """

    model: str
    threshold: float
    metrics: dict[str, Any]
    ci95: dict[str, list[float]]
    curves: dict[str, Any]
    n: int
    latency_ms: dict[str, float] = field(default_factory=dict)


class InferenceService:
    """Loads bundles once and serves predictions under bounded concurrency.

    Attributes:
        artifacts_dir: Directory holding ``registry.json`` and the bundles.
        max_concurrency: Inferences allowed to execute at once.
        max_queue: Requests allowed to wait.
        queue_timeout_s: How long a request may wait before being rejected.
        metrics: The registry recording everything this service does.
    """

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        *,
        max_concurrency: int = 2,
        max_queue: int = 64,
        queue_timeout_s: float = 5.0,
        metrics: MetricsRegistry | None = None,
        eager: bool = True,
    ) -> None:
        """Load the registry and, by default, every bundle it names.

        Args:
            artifacts_dir: Where the bundles live.
            max_concurrency: Parallel inference slots. Two gives real overlap
                on a 4-core laptop because torch releases the GIL inside its
                kernels; more mostly adds contention.
            max_queue: Maximum waiting requests before rejecting.
            queue_timeout_s: Maximum wait before rejecting.
            metrics: An existing registry; a fresh one is created otherwise.
            eager: Load all bundles now. ``False`` defers to first use, which
                keeps test startup fast.

        Raises:
            NoArtifactsError: If no model can be loaded and ``eager`` is set.
        """
        self.artifacts_dir = Path(artifacts_dir or Paths().artifacts)
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self.metrics = metrics or MetricsRegistry()

        self._bundles: dict[str, Bundle] = {}
        self._registry: dict[str, Any] = {}

        # The semaphore bounds execution; the counters describe what the
        # semaphore is doing, which the semaphore itself cannot report.
        self._slots = threading.Semaphore(max_concurrency)
        self._counter_lock = threading.Lock()
        self._waiting = 0
        self._inflight = 0

        self.reload(eager=eager)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def reload(self, *, eager: bool = True) -> list[str]:
        """Re-read the registry and reload every bundle from disk.

        A partial registry is tolerated by design: a reviewer who ran
        ``train.py --model fast`` must still get a working app.

        Args:
            eager: Load the bundles now rather than on first use.

        Returns:
            Names of the models that loaded successfully.

        Raises:
            NoArtifactsError: If ``eager`` and nothing could be loaded.
        """
        self._registry = load_registry(self.artifacts_dir / "registry.json")
        self._bundles = {}

        names = list(self._registry.get("models", {}))
        if not names:
            message = (
                f"No trained models found in {self.artifacts_dir}. Train them first:\n"
                "    python train.py --model all"
            )
            if eager:
                raise NoArtifactsError(message)
            logger.warning(message)
            return []

        if eager:
            for name in names:
                try:
                    self._load_one(name)
                except Exception:
                    # One corrupt bundle must not take down the service; the
                    # remaining models stay available and the app shows a
                    # partial list.
                    logger.exception("Failed to load model %s; skipping", name)

            if not self._bundles:
                raise NoArtifactsError(
                    f"Found {len(names)} registry entries in {self.artifacts_dir} but none "
                    "could be loaded. Retrain with 'python train.py --model all'."
                )

        logger.info(
            "InferenceService ready: %s (max_concurrency=%d, max_queue=%d)",
            ", ".join(self._bundles) or "lazy",
            self.max_concurrency,
            self.max_queue,
        )
        return list(self._bundles)

    def _load_one(self, name: str) -> Bundle:
        """Load a single bundle and record its cold-start cost.

        Args:
            name: Model name.

        Returns:
            The loaded bundle.

        Raises:
            ModelNotFoundError: If the registry does not list ``name``.
        """
        entry = self._registry.get("models", {}).get(name)
        if entry is None:
            raise ModelNotFoundError(
                f"Model {name!r} is not in the registry. Available: "
                f"{', '.join(self._registry.get('models', {})) or 'none'}."
            )

        with timer() as load_time:
            bundle = load_bundle(bundle_dir(self.artifacts_dir, name, entry), name)

        self._bundles[name] = bundle
        self.metrics.record_model_loaded(
            name,
            load_time["ms"] / 1000,
            {
                "framework": bundle.framework,
                "n_params": bundle.n_params,
                "trained_at": entry.get("trained_at"),
                "validation_roc_auc": entry.get("roc_auc"),
            },
        )
        return bundle

    def get_bundle(self, name: str | None = None) -> Bundle:
        """Return a loaded bundle, loading it on demand.

        Args:
            name: Model name; the registry default when omitted.

        Returns:
            The requested bundle.

        Raises:
            ModelNotFoundError: If the name is unknown, listing what is
                available so the caller can correct the request.
            NoArtifactsError: If no models exist at all.
        """
        name = name or self.default_model
        if name is None:
            raise NoArtifactsError(
                f"No models available in {self.artifacts_dir}. Run 'python train.py --model all'."
            )
        if name in self._bundles:
            return self._bundles[name]
        return self._load_one(name)

    @property
    def models(self) -> list[str]:
        """Names of every model in the registry, loaded or not."""
        return list(self._registry.get("models", {}))

    @property
    def loaded_models(self) -> list[str]:
        """Names of the models currently held in memory."""
        return list(self._bundles)

    @property
    def default_model(self) -> str | None:
        """The model the app should preselect."""
        default = self._registry.get("default")
        if default in self.models:
            return default
        return next(iter(self.models), None)

    def model_info(self) -> dict[str, dict[str, Any]]:
        """Summarise every registered model for ``/models`` and the sidebar.

        Returns:
            Mapping from model name to framework, parameter count, validation
            metrics and training timestamp.
        """
        info: dict[str, dict[str, Any]] = {}
        for name, entry in self._registry.get("models", {}).items():
            summary = dict(entry)
            summary["loaded"] = name in self._bundles
            if name in self._bundles:
                metrics = self._bundles[name].metrics
                summary["validation"] = metrics.get("validation", {})
                summary["validation_ci95"] = metrics.get("validation_ci95", {})
                summary["inference_ms_per_1k_rows"] = metrics.get("inference_ms_per_1k_rows")
            info[name] = summary
        return info

    # ------------------------------------------------------------------
    # Queue accounting
    # ------------------------------------------------------------------

    def _acquire_slot(self) -> float:
        """Wait for an execution slot, applying back-pressure.

        Returns:
            Milliseconds spent waiting, recorded as the ``queue`` stage.

        Raises:
            QueueFullError: If ``max_queue`` requests are already waiting.
            QueueTimeoutError: If the wait exceeded ``queue_timeout_s``.
        """
        with self._counter_lock:
            if self._waiting >= self.max_queue:
                self.metrics.record_rejection("full")
                raise QueueFullError(
                    f"Queue is full ({self._waiting}/{self.max_queue} waiting). "
                    "Retry in a moment or lower the request rate."
                )
            self._waiting += 1
            self.metrics.set_queue_depth(self._waiting)

        start = time.perf_counter()
        acquired = self._slots.acquire(timeout=self.queue_timeout_s)
        waited_ms = (time.perf_counter() - start) * 1000

        with self._counter_lock:
            # Decrement first in both branches: a rejected request is no
            # longer waiting, and leaking this counter would permanently
            # inflate queue depth.
            self._waiting -= 1
            self.metrics.set_queue_depth(self._waiting)
            if acquired:
                self._inflight += 1
                self.metrics.set_inflight(self._inflight)

        if not acquired:
            self.metrics.record_rejection("timeout")
            raise QueueTimeoutError(
                f"Waited {self.queue_timeout_s:g}s for an inference slot without success. "
                "The service is saturated; retry shortly."
            )

        return waited_ms

    def _release_slot(self) -> None:
        """Return an execution slot and update the in-flight gauge."""
        with self._counter_lock:
            self._inflight -= 1
            self.metrics.set_inflight(self._inflight)
        self._slots.release()

    def queue_state(self) -> dict[str, Any]:
        """Report the live queue figures.

        Returns:
            Current depth, in-flight count and the configured bounds.
        """
        with self._counter_lock:
            return {
                "depth": self._waiting,
                "inflight": self._inflight,
                "max_concurrency": self.max_concurrency,
                "max_queue": self.max_queue,
                "queue_timeout_s": self.queue_timeout_s,
            }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame, *, require_target: bool) -> None:
        """Check request size and schema before doing any work.

        Args:
            df: The incoming dataframe.
            require_target: Whether ``Survived`` must be present.

        Raises:
            SchemaError: If the frame is empty, too large, or malformed.
        """
        if df.empty:
            raise SchemaError("The uploaded data has no rows.")
        if len(df) > MAX_ROWS:
            raise SchemaError(
                f"Request has {len(df):,} rows but the limit is {MAX_ROWS:,}. "
                "Split the file into smaller batches."
            )
        validate_schema(df, require_target=require_target)

    def predict(
        self, df: pd.DataFrame, model: str | None = None, threshold: float = 0.5
    ) -> PredictionResult:
        """Run inference on a dataframe.

        Args:
            df: Raw passenger rows in the Kaggle schema. Labels are optional.
            model: Model name; the registry default when omitted.
            threshold: Probability above which a passenger is predicted to
                survive.

        Returns:
            A :class:`PredictionResult`.

        Raises:
            SchemaError: If the data does not match the expected schema.
            ModelNotFoundError: If the named model does not exist.
            QueueFullError: If back-pressure rejected the request.
            QueueTimeoutError: If the request waited too long for a slot.
        """
        endpoint = "predict"
        name = model or self.default_model or "none"
        stages: dict[str, float] = {}

        with timer() as total:
            try:
                self._validate(df, require_target=False)
                bundle = self.get_bundle(model)
                name = bundle.name

                stages["queue"] = self._acquire_slot()
                try:
                    with timer() as preprocess:
                        engineered = engineer(df)
                        x_num, x_cat = bundle.preprocessor.transform(engineered)
                    stages["preprocess"] = preprocess["ms"]

                    with timer() as inference:
                        probabilities = bundle.predict_proba(x_num, x_cat)
                    stages["inference"] = inference["ms"]

                    with timer() as postprocess:
                        predictions = (probabilities >= threshold).astype(int)
                        passenger_ids = (
                            df["PassengerId"].tolist() if "PassengerId" in df.columns else None
                        )
                    stages["postprocess"] = postprocess["ms"]
                finally:
                    # finally, not a plain call: an exception inside the
                    # critical section must still return the slot, or the
                    # service would deadlock after max_concurrency failures.
                    self._release_slot()
            except Exception as exc:
                self._record_failure(endpoint, name, exc, total)
                raise

        self.metrics.record_request(
            endpoint=endpoint,
            model=name,
            status="2xx",
            n_rows=len(df),
            total_ms=total["ms"],
            stages_ms=stages,
            probabilities=probabilities.tolist(),
        )

        return PredictionResult(
            model=name,
            threshold=threshold,
            probabilities=probabilities,
            predictions=predictions,
            passenger_ids=passenger_ids,
            n=len(df),
            latency_ms={
                **{k: round(v, 3) for k, v in stages.items()},
                "total": round(total["ms"], 3),
            },
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        model: str | None = None,
        threshold: float = 0.5,
        n_boot: int = 1000,
    ) -> EvaluationResult:
        """Score a labelled dataframe and compute metrics with intervals.

        Args:
            df: Raw passenger rows including ``Survived``.
            model: Model name; the registry default when omitted.
            threshold: Decision threshold.
            n_boot: Bootstrap resamples; capped at 2000 to bound the cost.

        Returns:
            An :class:`EvaluationResult`.

        Raises:
            SchemaError: If ``Survived`` is absent or the data is malformed.
        """
        endpoint = "evaluate"
        name = model or self.default_model or "none"
        # Cap rather than reject: a caller asking for 10 000 resamples wants
        # precision, and 2000 already gives a stable interval.
        n_boot = max(0, min(n_boot, 2000))

        with timer() as total:
            try:
                self._validate(df, require_target=True)
                result = self.predict(df, model, threshold)
                name = result.model

                y_true = pd.to_numeric(df["Survived"]).to_numpy().astype(int)
                with timer() as scoring:
                    metrics = compute_metrics(y_true, result.probabilities, threshold)
                    intervals = bootstrap_ci(
                        y_true, result.probabilities, threshold=threshold, n_boot=n_boot
                    )
                    curves = curve_data(y_true, result.probabilities)
            except Exception as exc:
                self._record_failure(endpoint, name, exc, total)
                raise

        # Recorded under the evaluate endpoint; predict() already recorded the
        # inference itself, so rows are not double-counted here.
        self.metrics.record_request(
            endpoint=endpoint,
            model=name,
            status="2xx",
            n_rows=0,
            total_ms=total["ms"],
            stages_ms={"postprocess": scoring["ms"]},
        )

        return EvaluationResult(
            model=name,
            threshold=threshold,
            metrics=metrics,
            ci95=intervals,
            curves=curves,
            n=len(df),
            latency_ms={
                **result.latency_ms,
                "scoring": round(scoring["ms"], 3),
                "total": round(total["ms"], 3),
            },
        )

    def _record_failure(self, endpoint: str, model: str, exc: Exception, total: dict) -> None:
        """Record a failed request with the right error code and status class.

        Args:
            endpoint: Where it failed.
            model: Model involved, if known.
            exc: The exception raised.
            total: The open timer dict; its duration is read after the fact.
        """
        codes = {
            SchemaError: ("schema_error", "4xx"),
            ModelNotFoundError: ("model_not_found", "4xx"),
            NoArtifactsError: ("no_artifacts", "5xx"),
            QueueFullError: ("queue_full", "5xx"),
            QueueTimeoutError: ("queue_timeout", "5xx"),
        }
        code, status = codes.get(type(exc), ("internal", "5xx"))
        self.metrics.record_error(endpoint, code)
        # total["ms"] is filled by the timer's finally block, which has
        # already run by the time this exception handler executes.
        self.metrics.record_request(
            endpoint=endpoint,
            model=model,
            status=status,
            n_rows=0,
            total_ms=total.get("ms", 0.0),
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return the JSON snapshot behind ``/stats`` and the Ops tab.

        Returns:
            Counters, latency percentiles per stage, queue state, drift signal
            and per-model information.
        """
        return self.metrics.snapshot(queue_state=self.queue_state())

    def prometheus_text(self) -> bytes:
        """Return the Prometheus exposition payload.

        Returns:
            Bytes for ``GET /metrics``.
        """
        return self.metrics.prometheus_text()
