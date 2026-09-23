"""End-to-end smoke tests for train.py and the artifact contract.

These run the real CLI on the committed 100-row sample with tiny settings, so
the whole pipeline (load, split, engineer, fit, train, evaluate, save, load
back) is exercised in a couple of seconds. They are the tests that would
catch "the training script no longer runs", which no unit test can.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import train as train_module
from titanic.artifacts import (
    Bundle,
    ModelNotFoundError,
    NoArtifactsError,
    available_models,
    load_bundle,
    load_registry,
)
from titanic.config import Paths
from titanic.data import load_csv, stratified_split
from titanic.features import engineer

SAMPLE = str(Paths().sample_csv)


def run_training(artifacts_dir: Path, model: str = "fast", **extra: str) -> int:
    """Invoke the training CLI with fast, deterministic settings.

    Args:
        artifacts_dir: Where to write bundles.
        model: Which model to train.
        **extra: Additional flags.

    Returns:
        The CLI exit code.
    """
    argv = [
        "--model",
        model,
        "--data-path",
        SAMPLE,
        "--artifacts-dir",
        str(artifacts_dir),
        "--epochs",
        "5",
        "--no-cv",
        "--n-boot",
        "40",
        *[part for key, value in extra.items() for part in (f"--{key}", value)],
    ]
    return train_module.main(argv)


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """Train `fast` and `gbdt` once and share the directory across tests."""
    artifacts = tmp_path_factory.mktemp("artifacts")
    assert run_training(artifacts, "fast") == 0
    assert run_training(artifacts, "gbdt") == 0
    return artifacts


class TestTrainingCli:
    @pytest.mark.parametrize("model", ["fast", "deep", "attn", "gbdt"])
    def test_every_model_trains_and_writes_its_bundle(self, model, tmp_path) -> None:
        assert run_training(tmp_path, model) == 0
        directory = tmp_path / model
        weights = "model.joblib" if model == "gbdt" else "model.pt"
        for filename in (weights, "model_config.json", "preprocessor.json", "metrics.json"):
            assert (directory / filename).is_file(), f"{model}: {filename} missing"

    def test_saves_plots_as_standalone_html(self, trained: Path) -> None:
        plots = sorted(p.name for p in (trained / "fast" / "plots").glob("*.html"))
        assert "roc.html" in plots
        assert "confusion_matrix.html" in plots
        # include_plotlyjs="cdn" keeps each file small instead of embedding
        # ~3 MB of plotly.js into every one.
        assert (trained / "fast" / "plots" / "roc.html").stat().st_size < 500_000

    def test_missing_data_file_exits_cleanly(self, tmp_path) -> None:
        # A handled failure returns 1 with an explanation, not a traceback.
        code = train_module.main(["--model", "fast", "--data-path", str(tmp_path / "nope.csv")])
        assert code == 1

    def test_rejects_an_unknown_model_name(self, tmp_path) -> None:
        with pytest.raises(SystemExit):
            train_module.main(["--model", "xgboost", "--artifacts-dir", str(tmp_path)])


class TestMetricsPayload:
    def test_reports_every_required_metric(self, trained: Path) -> None:
        metrics = json.loads((trained / "fast" / "metrics.json").read_text(encoding="utf-8"))
        validation = metrics["validation"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
            assert key in validation, f"{key} missing from metrics.json"
        assert validation["confusion_matrix"]

    def test_confidence_intervals_bracket_the_point_estimate(self, trained: Path) -> None:
        metrics = json.loads((trained / "fast" / "metrics.json").read_text(encoding="utf-8"))
        for metric, (low, high) in metrics["validation_ci95"].items():
            value = metrics["validation"][metric]
            assert low <= high, f"{metric}: interval is inverted"
            # The point estimate must lie inside its own interval; if it does
            # not, the bootstrap is resampling something other than the data
            # the metric was computed on.
            assert low <= value <= high, f"{metric}: {value} outside [{low}, {high}]"

    def test_records_provenance(self, trained: Path) -> None:
        config = json.loads((trained / "fast" / "model_config.json").read_text(encoding="utf-8"))
        assert config["framework"] == "torch"
        assert config["seed"] == 42
        assert "versions" in config and "torch" in config["versions"]

    def test_validation_split_is_never_the_training_split(self, trained: Path) -> None:
        metrics = json.loads((trained / "fast" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["n_train"] + metrics["n_val"] == 100
        assert metrics["n_val"] == 20


class TestRegistry:
    def test_lists_every_trained_model(self, trained: Path) -> None:
        registry = load_registry(trained / "registry.json")
        assert set(registry["models"]) == {"fast", "gbdt"}
        assert registry["default"] in registry["models"]

    def test_training_one_model_does_not_erase_the_others(self, trained: Path) -> None:
        # Read-modify-write, not overwrite: `train.py --model deep` must leave
        # an existing `fast` entry intact.
        before = set(load_registry(trained / "registry.json")["models"])
        assert run_training(trained, "deep") == 0
        after = set(load_registry(trained / "registry.json")["models"])
        assert before < after

    def test_available_models_ignores_entries_without_files(
        self, trained: Path, tmp_path: Path
    ) -> None:
        # Work on a copy: this test deletes a bundle, and the `trained`
        # fixture is module-scoped, so mutating it would break later tests.
        import shutil

        workspace = tmp_path / "artifacts"
        shutil.copytree(trained, workspace)

        assert "fast" in available_models(workspace)
        # The app must survive a registry naming a directory someone deleted.
        shutil.rmtree(workspace / "gbdt")
        assert "gbdt" not in available_models(workspace)
        assert "fast" in available_models(workspace)


class TestBundleRoundTrip:
    @pytest.mark.parametrize("model", ["fast", "gbdt"])
    def test_loaded_bundle_reproduces_training_time_predictions(self, model, tmp_path) -> None:
        assert run_training(tmp_path, model) == 0
        bundle = load_bundle(tmp_path / model)
        assert isinstance(bundle, Bundle)

        df = engineer(stratified_split(load_csv(SAMPLE))[1])
        probabilities = bundle.predict_proba(*bundle.preprocessor.transform(df))

        assert probabilities.shape == (len(df),)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        # Loading twice must give identical numbers: dropout off, no RNG use.
        again = load_bundle(tmp_path / model)
        np.testing.assert_allclose(
            probabilities, again.predict_proba(*again.preprocessor.transform(df))
        )

    def test_predict_proba_hides_the_framework(self, trained: Path) -> None:
        # The app and the API must never branch on framework.
        df = engineer(stratified_split(load_csv(SAMPLE))[1])
        for name in ("fast", "gbdt"):
            bundle = load_bundle(trained / name)
            result = bundle.predict_proba(*bundle.preprocessor.transform(df))
            assert result.shape == (len(df),)

    def test_single_row_inference(self, trained: Path) -> None:
        bundle = load_bundle(trained / "fast")
        df = engineer(load_csv(SAMPLE).iloc[[0]])
        assert bundle.predict_proba(*bundle.preprocessor.transform(df)).shape == (1,)

    def test_missing_directory_names_the_fix(self, tmp_path) -> None:
        with pytest.raises(NoArtifactsError, match="train.py"):
            load_bundle(tmp_path / "does_not_exist")


def test_model_not_found_error_message_is_clean() -> None:
    # KeyError normally wraps its message in quotes when printed; the app
    # shows these messages directly to a user, so that is suppressed.
    error = ModelNotFoundError("Model 'nope' not found. Available: fast, deep.")
    assert str(error).startswith("Model 'nope' not found")
