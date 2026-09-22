"""Saving and loading trained models, and the registry that indexes them.

One artifact contract serves both frameworks. A bundle directory contains:

===========================  ===================================================
``model.pt`` / ``.joblib``   weights (torch ``state_dict``) or the fitted estimator
``model_config.json``        framework, architecture, parameter count, versions
``preprocessor.json``        the fitted preprocessor for *this* model
``metrics.json``             held-out validation metrics with bootstrap CIs
``history.json``             per-epoch training curves and the CV grid
``plots/*.html``             standalone Plotly figures
===========================  ===================================================

Everything except the torch weights and the sklearn estimator is JSON, so an
artifact can be read in a code review. ``gbdt/model.joblib`` is the single
exception -- scikit-learn has no clean JSON serialisation -- and the sklearn
version that wrote it is recorded so a mismatch warns rather than misbehaves.

The crucial property: **inference never needs the training data.** A bundle
plus a CSV is sufficient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from titanic.preprocessing import Preprocessor
from titanic.utils import get_logger

logger = get_logger(__name__)

#: Schema version of ``model_config.json`` and ``registry.json``.
ARTIFACT_VERSION = 1

TORCH_WEIGHTS = "model.pt"
SKLEARN_WEIGHTS = "model.joblib"
MODEL_CONFIG = "model_config.json"
PREPROCESSOR = "preprocessor.json"
METRICS = "metrics.json"
HISTORY = "history.json"
PLOTS_DIR = "plots"


class NoArtifactsError(FileNotFoundError):
    """Raised when no trained model can be found on disk.

    Its message always carries the command that fixes it, because this is the
    first error a reviewer hits if they run the app before training.
    """


class ModelNotFoundError(KeyError):
    """Raised when a named model is not present in the registry."""

    def __str__(self) -> str:
        """Return the message without ``KeyError``'s surrounding quotes."""
        return self.args[0] if self.args else ""


@dataclass
class Bundle:
    """A trained model plus everything needed to run and describe it.

    Attributes:
        name: Model name (``fast``, ``deep``, ``attn``, ``gbdt``).
        model: The torch module or fitted sklearn estimator.
        model_config: Architecture and provenance.
        preprocessor: The preprocessor fitted alongside this model.
        metrics: Held-out validation metrics.
        history: Training curves and CV grid; empty for sklearn.
    """

    name: str
    model: Any
    model_config: dict[str, Any]
    preprocessor: Preprocessor
    metrics: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)

    @property
    def framework(self) -> str:
        """``"torch"`` or ``"sklearn"``."""
        return self.model_config.get("framework", "torch")

    @property
    def n_params(self) -> int:
        """Parameter count (torch) or total tree nodes (sklearn)."""
        return int(self.model_config.get("n_params", 0))

    def predict_proba(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        """Return ``P(survived)`` for preprocessed features.

        This is the method that hides the framework difference from every
        caller: the app and the API never branch on ``framework``.

        Args:
            x_num: Float array ``(n, n_numeric)``.
            x_cat: Int array ``(n, n_categorical)``.

        Returns:
            Float array ``(n,)`` of probabilities.
        """
        if self.framework == "sklearn":
            from titanic.sklearn_models import stack_features

            # Column 1 is P(class 1) = P(survived); classes_ is [0, 1].
            return self.model.predict_proba(stack_features(x_num, x_cat))[:, 1]

        from titanic.training import predict_proba_torch

        return predict_proba_torch(self.model, x_num, x_cat)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dict as indented UTF-8 JSON.

    Args:
        path: Destination file; parents are created.
        payload: JSON-compatible dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # default=str rescues numpy scalars and datetimes that slipped through,
    # so a training run never dies at the final write step.
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Read a JSON file, optionally tolerating its absence.

    Args:
        path: File to read.
        required: Raise if missing, rather than returning ``{}``.

    Returns:
        Parsed contents, or ``{}`` when optional and absent.

    Raises:
        NoArtifactsError: If a required file is missing.
        ValueError: If the file is not valid JSON.
    """
    if not path.is_file():
        if required:
            raise NoArtifactsError(
                f"Missing artifact file: {path}. Train the models first:\n"
                "    python train.py --model all"
            )
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}). Regenerate it with train.py.") from exc


def save_bundle(
    directory: Path,
    name: str,
    model: Any,
    model_config: dict[str, Any],
    preprocessor: Preprocessor,
    metrics: dict[str, Any],
    history: dict[str, Any] | None = None,
    figures: dict[str, Any] | None = None,
) -> Path:
    """Write a complete bundle to disk.

    Args:
        directory: Target directory, created if absent.
        name: Model name.
        model: Trained torch module or sklearn estimator.
        model_config: Architecture and provenance; ``framework`` is required.
        preprocessor: The preprocessor fitted for this model.
        metrics: Validation metrics.
        history: Training history; omitted for sklearn models.
        figures: Plotly figures to save as HTML, keyed by filename stem.

    Returns:
        The directory written.

    Raises:
        ValueError: If ``model_config["framework"]`` is not torch or sklearn.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    framework = model_config.get("framework")

    if framework == "torch":
        import torch

        # state_dict, not the pickled module: a pickled module embeds the
        # class path, so renaming or moving a class would break every old
        # artifact. build_model() reconstructs the architecture from config.
        torch.save(model.state_dict(), directory / TORCH_WEIGHTS)
    elif framework == "sklearn":
        import joblib

        joblib.dump(model, directory / SKLEARN_WEIGHTS)
    else:
        raise ValueError(
            f"Unknown framework {framework!r} in model_config; expected 'torch' or 'sklearn'."
        )

    enriched = {
        **model_config,
        "name": name,
        "artifact_version": ARTIFACT_VERSION,
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(directory / MODEL_CONFIG, enriched)
    preprocessor.save(directory / PREPROCESSOR)
    _write_json(directory / METRICS, metrics)
    if history:
        _write_json(directory / HISTORY, history)

    if figures:
        plots_dir = directory / PLOTS_DIR
        plots_dir.mkdir(exist_ok=True)
        for stem, figure in figures.items():
            # include_plotlyjs="cdn" keeps each file at a few KB instead of
            # embedding 3 MB of plotly.js into every one.
            figure.write_html(plots_dir / f"{stem}.html", include_plotlyjs="cdn")

    logger.info("Saved %s bundle to %s", name, directory)
    return directory


def load_bundle(directory: Path, name: str | None = None) -> Bundle:
    """Load a bundle from disk, dispatching on its recorded framework.

    Args:
        directory: A bundle directory.
        name: Model name; inferred from ``model_config.json`` or the directory
            name when omitted.

    Returns:
        A ready-to-use :class:`Bundle`.

    Raises:
        NoArtifactsError: If the directory or a required file is missing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NoArtifactsError(
            f"No artifact directory at {directory}. Train the models first:\n"
            "    python train.py --model all"
        )

    model_config = _read_json(directory / MODEL_CONFIG)
    preprocessor = Preprocessor.load(directory / PREPROCESSOR)
    metrics = _read_json(directory / METRICS, required=False)
    history = _read_json(directory / HISTORY, required=False)
    name = name or model_config.get("name") or directory.name
    framework = model_config.get("framework", "torch")

    if framework == "sklearn":
        import joblib
        import sklearn

        # joblib.load unpickles, which executes code embedded in the file, so
        # it is only ever pointed at artifacts this repository produced:
        # train.py writes them and they are committed alongside the source.
        # The project takes no user-supplied model files -- the app and the API
        # accept CSVs only -- so there is no untrusted path into this call.
        # It is the one non-JSON artifact; scikit-learn has no clean JSON form.
        model = joblib.load(directory / SKLEARN_WEIGHTS)
        saved_version = model_config.get("sklearn_version")
        if saved_version and saved_version != sklearn.__version__:
            # A warning rather than an error: joblib artifacts usually load
            # across minor versions, and refusing would make the committed
            # bundle useless to a reviewer with a slightly different install.
            logger.warning(
                "%s was saved with scikit-learn %s but %s is installed. "
                "If predictions look wrong, retrain with 'python train.py --model gbdt'.",
                name,
                saved_version,
                sklearn.__version__,
            )
    else:
        import torch

        from titanic.models import build_model

        model = build_model(
            model_config,
            len(preprocessor.numeric_cols),
            preprocessor.cardinalities,
        )
        # weights_only=True refuses to execute arbitrary pickled code while
        # loading -- the safe default for a file read from disk.
        model.load_state_dict(
            torch.load(directory / TORCH_WEIGHTS, map_location="cpu", weights_only=True)
        )
        # eval() disables dropout. Without it the app would return a different
        # probability for the same passenger on every request.
        model.eval()

    return Bundle(
        name=name,
        model=model,
        model_config=model_config,
        preprocessor=preprocessor,
        metrics=metrics,
        history=history,
    )


def update_registry(
    registry_path: Path, name: str, entry: dict[str, Any], default: str | None = None
) -> dict[str, Any]:
    """Add or replace one model's entry in the registry.

    The registry is read-modify-write rather than overwritten, so training a
    single model does not erase the others' entries.

    Args:
        registry_path: Path to ``registry.json``.
        name: Model name.
        entry: Summary for this model.
        default: Model the app should preselect; only set when provided.

    Returns:
        The updated registry contents.
    """
    registry_path = Path(registry_path)
    registry = _read_json(registry_path, required=False) or {
        "version": ARTIFACT_VERSION,
        "models": {},
    }
    registry.setdefault("models", {})[name] = entry

    if default:
        registry["default"] = default
    elif "default" not in registry:
        registry["default"] = name

    # If the recorded default was never trained, fall back to any model that
    # was -- the app must not open pointing at something that does not exist.
    if registry["default"] not in registry["models"]:
        registry["default"] = next(iter(registry["models"]), None)

    registry["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write_json(registry_path, registry)
    return registry


def bundle_dir(artifacts_dir: Path, name: str, entry: dict[str, Any] | None = None) -> Path:
    """Resolve a model's bundle directory.

    The registry stores ``dir`` relative to the registry file's own location,
    so the whole ``artifacts/`` tree can be moved, renamed, or written by
    ``--artifacts-dir`` without invalidating it. An absolute value is honoured
    as-is, and a missing one falls back to the model name.

    Args:
        artifacts_dir: Directory containing ``registry.json``.
        name: Model name.
        entry: That model's registry entry, if already loaded.

    Returns:
        Absolute path to the bundle directory.
    """
    artifacts_dir = Path(artifacts_dir)
    recorded = (entry or {}).get("dir", name)
    directory = Path(recorded)
    if directory.is_absolute():
        return directory
    # Tolerate the legacy "artifacts/<name>" form by keeping only the leaf.
    return artifacts_dir / directory.name


def load_registry(registry_path: Path) -> dict[str, Any]:
    """Read the registry, tolerating its absence.

    Args:
        registry_path: Path to ``registry.json``.

    Returns:
        Registry contents, or an empty skeleton if it does not exist yet.
    """
    registry = _read_json(Path(registry_path), required=False)
    if not registry:
        return {"version": ARTIFACT_VERSION, "models": {}, "default": None}
    return registry


def available_models(artifacts_dir: Path) -> list[str]:
    """List models that are actually loadable from disk.

    The registry can name a model whose directory was deleted, so entries are
    checked against the filesystem. The app must tolerate any subset -- a
    reviewer who runs ``train.py --model fast`` should still get a working app.

    Args:
        artifacts_dir: Root artifacts directory.

    Returns:
        Model names whose bundles are present, in registry order.
    """
    artifacts_dir = Path(artifacts_dir)
    registry = load_registry(artifacts_dir / "registry.json")

    present: list[str] = []
    for name, entry in registry.get("models", {}).items():
        directory = bundle_dir(artifacts_dir, name, entry)
        if (directory / MODEL_CONFIG).is_file():
            present.append(name)
        else:
            logger.warning("Registry lists %s but %s is missing; skipping.", name, directory)

    return present
