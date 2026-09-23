"""Predictor adapters: one interface, two transports.

The Streamlit app never touches a model, a bundle or an HTTP client directly.
It holds a :class:`Predictor`, which is either:

* :class:`LocalPredictor` wraps :class:`titanic.service.InferenceService`
  in-process. This is the default and needs no server, which is what the
  assignment requires ("load the trained model from disk").
* :class:`ApiPredictor` wraps ``httpx`` calls to a running FastAPI server.
  Selected by setting ``TITANIC_API_URL``.

Both return the same dataclasses, so every tab is written once. If the API
becomes unreachable, :func:`build_predictor` falls back to local mode with a
visible warning rather than leaving the app broken. The API is an optional
layer, never a dependency.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
import pandas as pd

from titanic.artifacts import ModelNotFoundError
from titanic.data import SchemaError
from titanic.service import (
    EvaluationResult,
    InferenceService,
    PredictionResult,
    QueueFullError,
    QueueTimeoutError,
)
from titanic.utils import get_logger

logger = get_logger(__name__)

#: How long to wait for the API before treating it as unreachable. Generous
#: enough for a cold start with four models, short enough that a dead server
#: does not hang the UI.
API_TIMEOUT_S = 30.0

#: Timeout for the health probe that decides local vs API mode. Much shorter:
#: this runs on every app start and must not delay the first render.
HEALTH_TIMEOUT_S = 2.0


class Predictor(Protocol):
    """What every tab of the app is allowed to depend on."""

    mode: str

    def models(self) -> dict[str, dict[str, Any]]:
        """Return every registered model and its metadata."""
        ...

    def default_model(self) -> str | None:
        """Return the model the sidebar should preselect."""
        ...

    def predict(self, df: pd.DataFrame, model: str | None, threshold: float) -> PredictionResult:
        """Run inference on a dataframe."""
        ...

    def evaluate(
        self, df: pd.DataFrame, model: str | None, threshold: float, n_boot: int
    ) -> EvaluationResult:
        """Score a labelled dataframe."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return the observability snapshot for the Ops tab."""
        ...


class LocalPredictor:
    """In-process predictor backed by :class:`InferenceService`.

    Attributes:
        mode: Human-readable description shown in the sidebar badge.
        service: The underlying service.
    """

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        """Load every available bundle.

        Args:
            artifacts_dir: Where the trained bundles live.

        Raises:
            NoArtifactsError: If no model can be loaded.
        """
        self.service = InferenceService(artifacts_dir)
        self.mode = "Local (in-process)"

    def models(self) -> dict[str, dict[str, Any]]:
        """Return every registered model and its metadata."""
        return self.service.model_info()

    def default_model(self) -> str | None:
        """Return the model the sidebar should preselect."""
        return self.service.default_model

    def predict(
        self, df: pd.DataFrame, model: str | None = None, threshold: float = 0.5
    ) -> PredictionResult:
        """Run inference on a dataframe."""
        return self.service.predict(df, model, threshold)

    def evaluate(
        self,
        df: pd.DataFrame,
        model: str | None = None,
        threshold: float = 0.5,
        n_boot: int = 1000,
    ) -> EvaluationResult:
        """Score a labelled dataframe."""
        return self.service.evaluate(df, model, threshold, n_boot)

    def stats(self) -> dict[str, Any]:
        """Return the observability snapshot for the Ops tab."""
        return self.service.stats()


class ApiPredictor:
    """Predictor that calls a running FastAPI server over HTTP.

    Attributes:
        base_url: Root URL of the API.
        mode: Human-readable description shown in the sidebar badge.
    """

    def __init__(self, base_url: str) -> None:
        """Point the predictor at an API.

        Args:
            base_url: Root URL, for example ``http://127.0.0.1:8000``.
        """
        self.base_url = base_url.rstrip("/")
        self.mode = f"API @ {self.base_url}"
        self._client = httpx.Client(timeout=API_TIMEOUT_S)

    @staticmethod
    def is_reachable(base_url: str) -> bool:
        """Probe an API before committing the app to API mode.

        Args:
            base_url: Root URL to probe.

        Returns:
            ``True`` if ``/health`` answered successfully.
        """
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=HEALTH_TIMEOUT_S)
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("API at %s is unreachable: %s", base_url, exc)
            return False

    def _raise_for_error(self, response: httpx.Response) -> None:
        """Translate an API error body back into the project's exceptions.

        The app's error handling is written once, against the same exception
        types the local path raises, so a tab cannot behave differently
        depending on transport.

        Args:
            response: The HTTP response to check.

        Raises:
            SchemaError: For 422 validation failures.
            ModelNotFoundError: For 404 unknown models.
            QueueFullError: For 503 back-pressure rejections.
            QueueTimeoutError: For 503 queue timeouts.
            RuntimeError: For anything else.
        """
        if response.status_code < 400:
            return

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"API returned {response.status_code}: {response.text[:200]}"
            ) from exc

        code = body.get("error", "")
        message = body.get("message") or body.get("detail") or response.text

        # FastAPI's own 422 body has "detail" rather than the project's shape.
        if response.status_code == 422:
            if isinstance(message, list):
                fields = ", ".join(str(item.get("loc", [])[-1]) for item in message)
                raise SchemaError(f"The API rejected these fields: {fields}")
            raise SchemaError(str(message))
        if code == "model_not_found":
            raise ModelNotFoundError(str(message))
        if code == "queue_full":
            raise QueueFullError(str(message))
        if code == "queue_timeout":
            raise QueueTimeoutError(str(message))
        raise RuntimeError(str(message))

    def models(self) -> dict[str, dict[str, Any]]:
        """Return every registered model and its metadata."""
        response = self._client.get(f"{self.base_url}/models")
        self._raise_for_error(response)
        return response.json().get("models", {})

    def default_model(self) -> str | None:
        """Return the model the sidebar should preselect."""
        response = self._client.get(f"{self.base_url}/models")
        self._raise_for_error(response)
        return response.json().get("default")

    def _post_csv(self, path: str, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        """POST a dataframe as a CSV upload.

        Args:
            path: Endpoint path.
            df: Data to send.
            params: Query parameters.

        Returns:
            The decoded JSON body.
        """
        buffer = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        response = self._client.post(
            f"{self.base_url}{path}",
            files={"file": ("data.csv", buffer, "text/csv")},
            params={key: value for key, value in params.items() if value is not None},
        )
        self._raise_for_error(response)
        return response.json()

    def predict(
        self, df: pd.DataFrame, model: str | None = None, threshold: float = 0.5
    ) -> PredictionResult:
        """Run inference on a dataframe via the API."""
        body = self._post_csv("/predict/csv", df, {"model": model, "threshold": threshold})
        rows = body["predictions"]
        return PredictionResult(
            model=body["model"],
            threshold=body["threshold"],
            probabilities=np.array([row["p_survived"] for row in rows], dtype=float),
            predictions=np.array([row["prediction"] for row in rows], dtype=int),
            passenger_ids=[row["passenger_id"] for row in rows],
            n=body["n"],
            latency_ms=body["latency_ms"],
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        model: str | None = None,
        threshold: float = 0.5,
        n_boot: int = 1000,
    ) -> EvaluationResult:
        """Score a labelled dataframe via the API."""
        body = self._post_csv(
            "/evaluate", df, {"model": model, "threshold": threshold, "n_boot": n_boot}
        )
        return EvaluationResult(
            model=body["model"],
            threshold=body["threshold"],
            metrics=body["metrics"],
            ci95=body["ci95"],
            curves=body["curves"],
            n=body["n"],
            latency_ms=body["latency_ms"],
        )

    def stats(self) -> dict[str, Any]:
        """Return the observability snapshot from ``GET /stats``."""
        response = self._client.get(f"{self.base_url}/stats")
        self._raise_for_error(response)
        return response.json()


def build_predictor(
    api_url: str | None = None, artifacts_dir: Path | str | None = None
) -> tuple[Predictor, str | None]:
    """Choose a predictor, preferring the API but never depending on it.

    Args:
        api_url: API root URL, typically from ``TITANIC_API_URL``. When empty
            the app runs locally without probing anything.
        artifacts_dir: Artifacts directory for local mode.

    Returns:
        ``(predictor, warning)``. ``warning`` is ``None`` on success, or a
        message explaining why the app fell back to local mode. The sidebar
        shows it, so the user always knows which mode is active.
    """
    if not api_url:
        return LocalPredictor(artifacts_dir), None

    if ApiPredictor.is_reachable(api_url):
        return ApiPredictor(api_url), None

    return (
        LocalPredictor(artifacts_dir),
        f"Could not reach the API at {api_url}. Running in local mode instead. "
        "Start it with: uvicorn api.main:app --port 8000",
    )
