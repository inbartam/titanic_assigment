"""Tests for the inference service, especially its queue accounting.

The queue tests are the interesting ones. They block the service deliberately
and assert on the gauges, because "queue depth" is the number the Ops tab
presents as an autoscaling signal. If it measured in-flight count or
thread-pool size instead, the dashboard would be wrong.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import train as train_module
from titanic.artifacts import ModelNotFoundError, NoArtifactsError
from titanic.config import Paths
from titanic.data import SchemaError, load_csv
from titanic.metrics import MetricsRegistry
from titanic.service import (
    InferenceService,
    QueueFullError,
    QueueTimeoutError,
)

SAMPLE = Paths().sample_csv


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> Path:
    """Train two small models once for the whole module."""
    directory = tmp_path_factory.mktemp("svc_artifacts")
    for model in ("fast", "gbdt"):
        assert (
            train_module.main(
                [
                    "--model",
                    model,
                    "--data-path",
                    str(SAMPLE),
                    "--artifacts-dir",
                    str(directory),
                    "--epochs",
                    "5",
                    "--no-cv",
                    "--n-boot",
                    "20",
                ]
            )
            == 0
        )
    return directory


@pytest.fixture
def service(artifacts: Path) -> InferenceService:
    """A fresh service with its own metrics registry."""
    # A fresh MetricsRegistry per test: prometheus_client raises on duplicate
    # timeseries if instances share the global default registry.
    return InferenceService(artifacts, metrics=MetricsRegistry())


@pytest.fixture
def df():
    """The committed sample dataframe."""
    return load_csv(SAMPLE)


class TestLoading:
    def test_loads_every_registered_model(self, service: InferenceService) -> None:
        assert set(service.loaded_models) == {"fast", "gbdt"}

    def test_tolerates_a_partial_registry(self, tmp_path) -> None:
        # A reviewer who ran `train.py --model fast` must still get a service.
        assert (
            train_module.main(
                [
                    "--model",
                    "fast",
                    "--data-path",
                    str(SAMPLE),
                    "--artifacts-dir",
                    str(tmp_path),
                    "--epochs",
                    "3",
                    "--no-cv",
                    "--n-boot",
                    "10",
                ]
            )
            == 0
        )
        assert InferenceService(tmp_path, metrics=MetricsRegistry()).models == ["fast"]

    def test_no_artifacts_names_the_fix(self, tmp_path) -> None:
        with pytest.raises(NoArtifactsError, match="train.py"):
            InferenceService(tmp_path / "empty", metrics=MetricsRegistry())

    def test_unknown_model_lists_what_exists(self, service: InferenceService) -> None:
        with pytest.raises(ModelNotFoundError) as exc:
            service.get_bundle("xgboost")
        assert "fast" in str(exc.value)

    def test_records_model_load_time(self, service: InferenceService) -> None:
        models = service.stats()["models"]
        assert set(models) == {"fast", "gbdt"}
        assert all("loaded_ms" in info for info in models.values())


class TestPredict:
    def test_returns_one_probability_per_row(self, service: InferenceService, df) -> None:
        result = service.predict(df, "fast")
        assert result.n == len(df)
        assert len(result.probabilities) == len(df)
        assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()

    def test_works_without_labels(self, service: InferenceService, df) -> None:
        # The assignment requires the app not to crash on an unlabelled CSV.
        unlabelled = df.drop(columns=["Survived"])
        assert service.predict(unlabelled, "fast").n == len(df)

    def test_threshold_controls_the_predicted_class(self, service: InferenceService, df) -> None:
        low = service.predict(df, "fast", threshold=0.1)
        high = service.predict(df, "fast", threshold=0.9)
        assert low.predictions.sum() >= high.predictions.sum()

    def test_records_every_stage(self, service: InferenceService, df) -> None:
        latency = service.predict(df, "fast").latency_ms
        for stage in ("queue", "preprocess", "inference", "postprocess", "total"):
            assert stage in latency

    def test_stage_times_sum_to_roughly_the_total(self, service: InferenceService, df) -> None:
        latency = service.predict(df, "fast").latency_ms
        staged = sum(latency[s] for s in ("queue", "preprocess", "inference", "postprocess"))
        # The total also covers validation and bundle lookup, so it is the
        # larger of the two; a stage total exceeding it would mean double
        # counting somewhere.
        assert staged <= latency["total"] + 1e-6

    def test_rejects_a_non_titanic_csv(self, service: InferenceService) -> None:
        import pandas as pd

        with pytest.raises(SchemaError, match="Missing required columns"):
            service.predict(pd.DataFrame({"foo": [1], "bar": [2]}), "fast")

    def test_rejects_an_empty_frame(self, service: InferenceService, df) -> None:
        with pytest.raises(SchemaError, match="no rows"):
            service.predict(df.iloc[0:0], "fast")

    def test_result_frame_echoes_identifying_columns(self, service: InferenceService, df) -> None:
        frame = service.predict(df, "fast").to_frame(df)
        assert "p_survived" in frame.columns
        assert "PassengerId" in frame.columns
        assert list(frame.columns).index("PassengerId") < list(frame.columns).index("p_survived")

    def test_unknown_category_does_not_crash(self, service: InferenceService, df) -> None:
        row = df.iloc[[0]].copy()
        row["Embarked"] = "Z"
        row["Cabin"] = "Z99"
        assert service.predict(row, "fast").n == 1


class TestEvaluate:
    def test_returns_metrics_with_intervals(self, service: InferenceService, df) -> None:
        result = service.evaluate(df, "fast", n_boot=50)
        assert 0 <= result.metrics["accuracy"] <= 1
        assert result.ci95["accuracy"][0] <= result.ci95["accuracy"][1]
        assert result.curves["roc"]

    def test_requires_labels(self, service: InferenceService, df) -> None:
        with pytest.raises(SchemaError, match="Survived"):
            service.evaluate(df.drop(columns=["Survived"]), "fast")

    def test_bootstrap_count_is_capped(self, service: InferenceService, df) -> None:
        # A caller asking for 10 000 resamples gets 2000 rather than an error:
        # the interval is already stable there and the cost is bounded.
        assert service.evaluate(df, "fast", n_boot=99_999).ci95


class TestQueueAccounting:
    def test_queue_depth_counts_waiting_not_executing(self, artifacts: Path) -> None:
        # docs/API.md section 8: with one slot and three concurrent requests,
        # one executes and two wait, so peak depth must be exactly 2.
        service = InferenceService(
            artifacts, max_concurrency=1, max_queue=64, metrics=MetricsRegistry()
        )
        observed_peak = 0
        release = threading.Event()
        inside = threading.Event()

        original = service.get_bundle("fast").predict_proba

        def slow_predict(x_num, x_cat):
            """Hold the single slot until the test releases it."""
            inside.set()
            release.wait(timeout=5)
            return original(x_num, x_cat)

        service.get_bundle("fast").predict_proba = slow_predict
        frame = load_csv(SAMPLE)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(service.predict, frame, "fast") for _ in range(3)]
            inside.wait(timeout=5)

            # Sample while the first request holds the slot.
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                observed_peak = max(observed_peak, service.queue_state()["depth"])
                if observed_peak >= 2:
                    break
                time.sleep(0.01)

            assert service.queue_state()["inflight"] == 1, "in-flight must cap at max_concurrency"
            release.set()
            for future in futures:
                future.result(timeout=10)

        assert observed_peak == 2, f"expected peak queue depth 2, saw {observed_peak}"
        # Everything drains once the slot is released.
        assert service.queue_state()["depth"] == 0
        assert service.queue_state()["inflight"] == 0

    def test_max_queue_rejects_with_queue_full(self, artifacts: Path) -> None:
        service = InferenceService(
            artifacts, max_concurrency=1, max_queue=1, metrics=MetricsRegistry()
        )
        release = threading.Event()
        inside = threading.Event()
        original = service.get_bundle("fast").predict_proba

        def slow_predict(x_num, x_cat):
            """Occupy the only slot."""
            inside.set()
            release.wait(timeout=5)
            return original(x_num, x_cat)

        service.get_bundle("fast").predict_proba = slow_predict
        frame = load_csv(SAMPLE)
        errors: list[Exception] = []

        def attempt() -> None:
            """Run one prediction, capturing any back-pressure error."""
            try:
                service.predict(frame, "fast")
            except (QueueFullError, QueueTimeoutError) as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=3) as pool:
            holder = pool.submit(attempt)
            inside.wait(timeout=5)
            # One executing + one allowed to wait => the third is rejected.
            waiter = pool.submit(attempt)
            time.sleep(0.2)
            rejected = pool.submit(attempt)
            time.sleep(0.2)
            release.set()
            for future in (holder, waiter, rejected):
                future.result(timeout=10)

        assert len(errors) == 1, f"expected exactly one rejection, got {len(errors)}"
        assert isinstance(errors[0], QueueFullError)
        assert service.stats()["queue"]["rejections"]["full"] == 1

    def test_queue_timeout_rejects_after_waiting(self, artifacts: Path) -> None:
        service = InferenceService(
            artifacts,
            max_concurrency=1,
            max_queue=8,
            queue_timeout_s=0.2,
            metrics=MetricsRegistry(),
        )
        release = threading.Event()
        inside = threading.Event()
        original = service.get_bundle("fast").predict_proba

        def slow_predict(x_num, x_cat):
            """Hold the slot for longer than the queue timeout."""
            inside.set()
            release.wait(timeout=5)
            return original(x_num, x_cat)

        service.get_bundle("fast").predict_proba = slow_predict
        frame = load_csv(SAMPLE)

        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(service.predict, frame, "fast")
            inside.wait(timeout=5)
            with pytest.raises(QueueTimeoutError):
                service.predict(frame, "fast")
            release.set()
            holder.result(timeout=10)

        assert service.stats()["queue"]["rejections"]["timeout"] == 1

    def test_slot_is_released_when_inference_raises(self, service: InferenceService, df) -> None:
        # Without the finally block around the critical section, the service
        # would deadlock after max_concurrency failures.
        bundle = service.get_bundle("fast")
        original = bundle.predict_proba

        def boom(x_num, x_cat):
            """Fail inside the critical section."""
            raise RuntimeError("model exploded")

        bundle.predict_proba = boom
        for _ in range(3):
            with pytest.raises(RuntimeError):
                service.predict(df, "fast")

        bundle.predict_proba = original
        assert service.queue_state()["inflight"] == 0
        assert service.predict(df, "fast").n == len(df)


class TestStats:
    def test_counts_requests_and_rows(self, service: InferenceService, df) -> None:
        service.predict(df, "fast")
        service.predict(df.head(10), "gbdt")
        stats = service.stats()
        assert stats["requests"]["total"] == 2
        assert stats["rows_predicted"]["total"] == len(df) + 10
        assert set(stats["requests"]["per_model"]) == {"fast", "gbdt"}

    def test_reports_percentiles_per_stage(self, service: InferenceService, df) -> None:
        for _ in range(5):
            service.predict(df, "fast")
        per_stage = service.stats()["latency_ms"]["per_stage"]
        for stage in ("preprocess", "inference"):
            assert per_stage[stage]["p95"] >= per_stage[stage]["p50"]

    def test_tracks_the_error_rate(self, service: InferenceService, df) -> None:
        service.predict(df, "fast")
        with pytest.raises(SchemaError):
            service.predict(df.drop(columns=["Pclass"]), "fast")
        assert service.stats()["requests"]["error_rate"] == pytest.approx(0.5)

    def test_reports_the_drift_signal(self, service: InferenceService, df) -> None:
        service.predict(df, "fast")
        predictions = service.stats()["predictions"]
        assert 0 <= predictions["positive_rate_window"]["fast"] <= 1
        # The training base rate is the reference the Ops tab compares against.
        assert predictions["train_base_rate"] == pytest.approx(0.3838)

    def test_prometheus_exposes_the_key_series(self, service: InferenceService, df) -> None:
        service.predict(df, "fast")
        text = service.prometheus_text().decode()
        assert "titanic_requests_total" in text
        assert "titanic_queue_depth" in text
        assert "titanic_stage_duration_seconds" in text
