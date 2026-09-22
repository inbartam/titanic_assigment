"""Tests for the FastAPI adapter.

These check the HTTP contract: status codes, the single error shape, and that
a traceback never reaches the client. The inference behaviour itself is tested
in ``test_service.py`` -- the API is a thin adapter and these tests treat it
as one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import train as train_module
from api.main import create_app
from api.settings import Settings
from titanic.config import Paths

SAMPLE = Paths().sample_csv

VALID_PASSENGER = {
    "Pclass": 3,
    "Name": "Braund, Mr. Owen Harris",
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> Path:
    """Train two small models once for the whole module."""
    directory = tmp_path_factory.mktemp("api_artifacts")
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
def client(artifacts: Path):
    """A TestClient over an app wired to the test artifacts."""
    # `with` triggers the lifespan handler, which is what loads the models.
    with TestClient(create_app(Settings(artifacts_dir=artifacts))) as test_client:
        yield test_client


class TestHealthAndDiscovery:
    def test_health_reports_loaded_models(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert set(body["models_loaded"]) == {"fast", "gbdt"}

    def test_models_lists_the_registry(self, client) -> None:
        body = client.get("/models").json()
        assert set(body["models"]) == {"fast", "gbdt"}
        assert body["models"]["fast"]["framework"] == "torch"
        assert body["models"]["gbdt"]["framework"] == "sklearn"

    def test_models_includes_validation_metrics(self, client) -> None:
        entry = client.get("/models").json()["models"]["fast"]
        assert "accuracy" in entry["validation"]
        assert entry["validation_ci95"]

    def test_schema_endpoint_documents_the_columns(self, client) -> None:
        body = client.get("/schema").json()
        assert "Pclass" in body["required"]
        assert "Survived" in body["optional"]

    def test_every_response_carries_a_request_id(self, client) -> None:
        # The id is what a user quotes when reporting an error, since the
        # traceback is deliberately kept server-side.
        assert client.get("/health").headers["X-Request-ID"]


class TestPredict:
    def test_single_passenger(self, client) -> None:
        response = client.post("/predict", json={"model": "fast", "passengers": [VALID_PASSENGER]})
        assert response.status_code == 200

        body = response.json()
        assert body["n"] == 1
        row = body["predictions"][0]
        assert 0.0 <= row["p_survived"] <= 1.0
        assert row["prediction"] in (0, 1)

    def test_reports_latency_per_stage(self, client) -> None:
        latency = client.post("/predict", json={"passengers": [VALID_PASSENGER]}).json()[
            "latency_ms"
        ]
        for stage in ("queue", "preprocess", "inference", "total"):
            assert stage in latency

    def test_prediction_is_consistent_with_the_threshold(self, client) -> None:
        for threshold in (0.1, 0.9):
            body = client.post(
                "/predict",
                json={"model": "fast", "threshold": threshold, "passengers": [VALID_PASSENGER]},
            ).json()
            row = body["predictions"][0]
            assert row["prediction"] == int(row["p_survived"] >= threshold)

    def test_missing_required_field_is_422_naming_the_field(self, client) -> None:
        broken = {k: v for k, v in VALID_PASSENGER.items() if k != "Pclass"}
        response = client.post("/predict", json={"passengers": [broken]})
        assert response.status_code == 422
        assert any("Pclass" in str(error["loc"]) for error in response.json()["detail"])

    def test_invalid_sex_is_rejected(self, client) -> None:
        response = client.post(
            "/predict", json={"passengers": [{**VALID_PASSENGER, "Sex": "unknown"}]}
        )
        assert response.status_code == 422

    def test_out_of_range_pclass_is_rejected(self, client) -> None:
        response = client.post("/predict", json={"passengers": [{**VALID_PASSENGER, "Pclass": 9}]})
        assert response.status_code == 422

    def test_unknown_model_is_404_listing_the_options(self, client) -> None:
        response = client.post("/predict", json={"model": "nope", "passengers": [VALID_PASSENGER]})
        assert response.status_code == 404

        body = response.json()
        assert body["error"] == "model_not_found"
        assert set(body["details"]["available_models"]) == {"fast", "gbdt"}

    def test_empty_passenger_list_is_rejected(self, client) -> None:
        assert client.post("/predict", json={"passengers": []}).status_code == 422

    def test_optional_fields_may_be_omitted(self, client) -> None:
        # Age and Fare are imputed with values fitted on the training split.
        minimal = {k: v for k, v in VALID_PASSENGER.items() if k not in ("Age", "Fare")}
        assert client.post("/predict", json={"passengers": [minimal]}).status_code == 200


class TestCsvEndpoints:
    def test_predict_csv_returns_json_by_default(self, client) -> None:
        with SAMPLE.open("rb") as handle:
            response = client.post(
                "/predict/csv?model=fast", files={"file": ("sample.csv", handle, "text/csv")}
            )
        assert response.status_code == 200
        assert response.json()["n"] == 100

    def test_predict_csv_can_return_csv(self, client) -> None:
        with SAMPLE.open("rb") as handle:
            response = client.post(
                "/predict/csv",
                files={"file": ("sample.csv", handle, "text/csv")},
                headers={"accept": "text/csv"},
            )
        assert response.status_code == 200
        assert "p_survived" in response.text.splitlines()[0]

    def test_unlabelled_csv_returns_predictions_with_a_note(self, client, tmp_path) -> None:
        import pandas as pd

        unlabelled = tmp_path / "unlabelled.csv"
        pd.read_csv(SAMPLE).drop(columns=["Survived"]).to_csv(unlabelled, index=False)

        with unlabelled.open("rb") as handle:
            response = client.post(
                "/predict/csv", files={"file": ("unlabelled.csv", handle, "text/csv")}
            )
        assert response.status_code == 200
        assert "Survived" in response.json()["note"]

    def test_rejects_a_non_csv_upload(self, client) -> None:
        response = client.post(
            "/predict/csv", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "schema_error"

    def test_rejects_an_empty_upload(self, client) -> None:
        response = client.post("/predict/csv", files={"file": ("empty.csv", b"", "text/csv")})
        assert response.status_code == 422

    def test_rejects_a_csv_with_the_wrong_columns(self, client) -> None:
        response = client.post(
            "/predict/csv", files={"file": ("wrong.csv", b"a,b\n1,2\n", "text/csv")}
        )
        assert response.status_code == 422
        assert "Missing required columns" in response.json()["message"]


class TestEvaluate:
    def test_returns_metrics_and_intervals(self, client) -> None:
        with SAMPLE.open("rb") as handle:
            response = client.post(
                "/evaluate?model=fast&n_boot=50",
                files={"file": ("sample.csv", handle, "text/csv")},
            )
        assert response.status_code == 200

        body = response.json()
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            assert key in body["metrics"]
        for low, high in body["ci95"].values():
            assert low <= high
        assert len(body["confusion_matrix"]) == 2

    def test_requires_labels(self, client, tmp_path) -> None:
        import pandas as pd

        unlabelled = tmp_path / "unlabelled.csv"
        pd.read_csv(SAMPLE).drop(columns=["Survived"]).to_csv(unlabelled, index=False)

        with unlabelled.open("rb") as handle:
            response = client.post(
                "/evaluate", files={"file": ("unlabelled.csv", handle, "text/csv")}
            )
        assert response.status_code == 422
        assert "Survived" in response.json()["message"]

    def test_returns_curve_arrays_for_the_figures(self, client) -> None:
        with SAMPLE.open("rb") as handle:
            body = client.post(
                "/evaluate?n_boot=20", files={"file": ("sample.csv", handle, "text/csv")}
            ).json()
        assert body["curves"]["roc"]["fpr"]
        assert body["curves"]["threshold_sweep"]["thresholds"]


class TestObservability:
    def test_metrics_exposes_the_key_series(self, client) -> None:
        client.post("/predict", json={"passengers": [VALID_PASSENGER]})
        text = client.get("/metrics").text
        assert "titanic_requests_total" in text
        assert "titanic_queue_depth" in text

    def test_stats_has_the_shape_the_ops_tab_expects(self, client) -> None:
        client.post("/predict", json={"model": "fast", "passengers": [VALID_PASSENGER]})
        stats = client.get("/stats").json()
        for section in ("requests", "rows_predicted", "latency_ms", "queue", "predictions"):
            assert section in stats
        assert stats["queue"]["max_concurrency"] >= 1
        assert stats["requests"]["total"] >= 1

    def test_errors_are_counted(self, client) -> None:
        client.post("/predict", json={"model": "nope", "passengers": [VALID_PASSENGER]})
        assert client.get("/stats").json()["requests"]["errors"] >= 1


class TestErrorContract:
    def test_errors_use_one_json_shape(self, client) -> None:
        body = client.post(
            "/predict", json={"model": "nope", "passengers": [VALID_PASSENGER]}
        ).json()
        assert set(body) == {"error", "message", "details"}

    def test_no_traceback_ever_reaches_the_client(self, client) -> None:
        response = client.post(
            "/predict/csv", files={"file": ("wrong.csv", b"a,b\n1,2\n", "text/csv")}
        )
        assert "Traceback" not in response.text
        assert 'File "' not in response.text

    def test_reload_is_disabled_without_a_token(self, client) -> None:
        response = client.post("/admin/reload")
        assert response.status_code == 404
        assert response.json()["error"] == "reload_disabled"

    def test_reload_requires_the_right_token(self, artifacts: Path) -> None:
        settings = Settings(artifacts_dir=artifacts, admin_token="s3cret")
        with TestClient(create_app(settings)) as guarded:
            assert guarded.post("/admin/reload").status_code == 403
            assert (
                guarded.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 403
            )


class TestBackPressure:
    def test_queue_full_returns_503_with_retry_after(self, artifacts: Path) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        settings = Settings(artifacts_dir=artifacts, max_concurrency=1, max_queue=1)
        with TestClient(create_app(settings)) as saturated:
            service = saturated.app.state.service
            release = threading.Event()
            inside = threading.Event()
            original = service.get_bundle("fast").predict_proba

            def slow_predict(x_num, x_cat):
                """Occupy the only execution slot."""
                inside.set()
                release.wait(timeout=5)
                return original(x_num, x_cat)

            service.get_bundle("fast").predict_proba = slow_predict
            body = {"model": "fast", "passengers": [VALID_PASSENGER]}

            with ThreadPoolExecutor(max_workers=4) as pool:
                calls = [pool.submit(saturated.post, "/predict", json=body) for _ in range(4)]
                inside.wait(timeout=5)
                import time

                time.sleep(0.4)
                release.set()
                responses = [call.result(timeout=15) for call in calls]

            service.get_bundle("fast").predict_proba = original

        rejected = [r for r in responses if r.status_code == 503]
        assert rejected, "expected at least one 503 under back-pressure"
        assert rejected[0].headers["Retry-After"] == "1"
        assert rejected[0].json()["error"] in {"queue_full", "queue_timeout"}
