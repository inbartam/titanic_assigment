"""Cached loaders and session helpers for the Streamlit app.

Streamlit re-executes the entire script on every interaction, so anything
expensive must be cached or the app would reload four PyTorch models each time
a slider moves.

Two cache decorators, used deliberately:

* ``@st.cache_resource`` for objects that must be shared and not copied -- the
  predictor holds loaded models, a semaphore and a metrics registry, and
  copying it would fork the metrics.
* ``@st.cache_data`` for values that are safe to copy -- dataframes and
  prediction results, keyed by content.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.client import Predictor, build_predictor
from titanic.config import Paths
from titanic.data import load_csv, validate_schema
from titanic.utils import get_logger

logger = get_logger(__name__)


@st.cache_resource(show_spinner="Loading models...")
def get_predictor(
    api_url: str | None, artifacts_dir: str | None = None
) -> tuple[Predictor, str | None]:
    """Build and cache the predictor for the session.

    ``cache_resource`` rather than ``cache_data``: the predictor owns loaded
    models, a concurrency semaphore and a metrics registry. Copying it per
    call would reset the Ops tab's numbers on every interaction.

    Args:
        api_url: API root URL, or ``None`` for local mode. Part of the cache
            key, so switching modes rebuilds the predictor.
        artifacts_dir: Artifacts directory override.

    Returns:
        ``(predictor, warning)`` as returned by
        :func:`app.client.build_predictor`.
    """
    return build_predictor(api_url, artifacts_dir)


@st.cache_data(show_spinner=False)
def load_dataframe(source: str, content: bytes | None = None) -> pd.DataFrame:
    """Load a CSV from a path or from uploaded bytes, cached by content.

    Args:
        source: Filesystem path, or a label when ``content`` is supplied.
        content: Raw bytes of an uploaded file.

    Returns:
        The parsed dataframe.

    Raises:
        SchemaError: If the data is not a valid Titanic CSV.
        FileNotFoundError: If a path was given and does not exist.
    """
    if content is not None:
        import io

        frame = pd.read_csv(io.BytesIO(content))
        if frame.empty:
            from titanic.data import SchemaError

            raise SchemaError("The uploaded CSV has a header but no data rows.")
    else:
        frame = load_csv(source)

    validate_schema(frame)
    return frame


@st.cache_data(show_spinner=False)
def load_registry_metrics(artifacts_dir: str) -> dict[str, dict[str, Any]]:
    """Read every model's ``metrics.json`` for the Compare tab.

    Read straight from disk rather than through the predictor because the
    Compare tab describes the *training run*, not the current session, and it
    must work identically in local and API mode.

    Args:
        artifacts_dir: Root artifacts directory.

    Returns:
        Mapping from model name to its metrics payload. Models whose files are
        missing are skipped, so a partial registry still renders.
    """
    import json

    from titanic.artifacts import available_models, bundle_dir, load_registry

    root = Path(artifacts_dir)
    registry = load_registry(root / "registry.json")
    collected: dict[str, dict[str, Any]] = {}

    for name in available_models(root):
        metrics_path = bundle_dir(root, name, registry["models"].get(name)) / "metrics.json"
        if metrics_path.is_file():
            collected[name] = json.loads(metrics_path.read_text(encoding="utf-8"))

    return collected


@st.cache_data(show_spinner=False)
def load_history(artifacts_dir: str, model: str) -> dict[str, Any]:
    """Read one model's training history for the training-curve figure.

    Args:
        artifacts_dir: Root artifacts directory.
        model: Model name.

    Returns:
        The history dict, or ``{}`` for models that have none (sklearn).
    """
    import json

    from titanic.artifacts import bundle_dir, load_registry

    root = Path(artifacts_dir)
    registry = load_registry(root / "registry.json")
    path = bundle_dir(root, model, registry["models"].get(model)) / "history.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Return a short stable hash of a dataframe's contents.

    Used as part of a cache key so predictions are recomputed when the data
    changes but not when an unrelated widget moves.

    Args:
        df: The dataframe to fingerprint.

    Returns:
        A 12-character hex digest.
    """
    # Hash the CSV rendering rather than pandas' own hash: it is stable across
    # dtype changes that do not alter the values a user can see.
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:12]


def resolve_api_url() -> str | None:
    """Read the API URL from the environment.

    Returns:
        The configured URL, or ``None`` for local mode.
    """
    url = os.environ.get("TITANIC_API_URL", "").strip()
    return url or None


def artifacts_dir() -> str:
    """Return the artifacts directory as a cache-key-friendly string.

    Returns:
        Absolute path to the artifacts directory.
    """
    return str(os.environ.get("TITANIC_ARTIFACTS_DIR") or Paths().artifacts)
