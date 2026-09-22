"""FastAPI application: a thin HTTP adapter over the inference service.

Route handlers do three things and nothing else: parse the request, call
:class:`titanic.service.InferenceService`, and map typed exceptions to status
codes. All inference, all queue accounting and all metric recording live in
the service, so the Streamlit app running in-process reports exactly the same
numbers with no server involved.

Run it with::

    uvicorn api.main:app --port 8000 --workers 1

``--workers 1`` matters: ``prometheus_client`` metrics are per-process, so
multiple workers would each report their own slice. Multi-process aggregation
is listed as future work in the README.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pandas as pd
from fastapi import FastAPI, File, Header, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST

from api.settings import Settings, get_settings
from titanic import __version__
from titanic.artifacts import ModelNotFoundError, NoArtifactsError
from titanic.data import SchemaError
from titanic.schemas import (
    ErrorResponse,
    EvaluateResponse,
    HealthResponse,
    PredictionRow,
    PredictRequest,
    PredictResponse,
    SchemaInfo,
)
from titanic.service import (
    MAX_ROWS,
    InferenceService,
    PredictionResult,
    QueueFullError,
    QueueTimeoutError,
)
from titanic.utils import get_logger

logger = get_logger("api")

#: Exception type to (HTTP status, machine-readable code). Route handlers stay
#: free of status codes; the mapping lives in exactly one place.
ERROR_MAP: dict[type[Exception], tuple[int, str]] = {
    SchemaError: (422, "schema_error"),
    ModelNotFoundError: (404, "model_not_found"),
    NoArtifactsError: (503, "no_artifacts"),
    QueueFullError: (503, "queue_full"),
    QueueTimeoutError: (503, "queue_timeout"),
}


def _prediction_rows(result: PredictionResult) -> list[PredictionRow]:
    """Map a service result onto the wire format.

    Both ``/predict`` and ``/predict/csv`` return the same per-row shape, so
    the mapping lives here rather than being written twice and drifting.

    Args:
        result: What :meth:`InferenceService.predict` returned.

    Returns:
        One :class:`PredictionRow` per input row, in input order.
    """
    return [
        PredictionRow(
            passenger_id=(result.passenger_ids[i] if result.passenger_ids is not None else None),
            p_survived=round(float(probability), 6),
            prediction=int(prediction),
        )
        for i, (probability, prediction) in enumerate(
            zip(result.probabilities, result.predictions, strict=True)
        )
    ]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    A factory rather than a module-level app so tests can construct isolated
    instances with their own artifacts directory and queue settings.

    Args:
        settings: Configuration; read from the environment when omitted.

    Returns:
        A configured :class:`FastAPI` instance.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load models at startup and release them at shutdown.

        Loading eagerly means the first real request is not penalised by a
        cold start, and a broken artifacts directory fails at boot rather than
        on a user's first click.
        """
        started = time.time()
        try:
            service = InferenceService(
                settings.artifacts_dir,
                max_concurrency=settings.max_concurrency,
                max_queue=settings.max_queue,
                queue_timeout_s=settings.queue_timeout_s,
            )
            logger.info("Loaded models: %s", ", ".join(service.loaded_models))
        except NoArtifactsError:
            # Start anyway so /health can report the problem instead of the
            # process dying and leaving the operator with no diagnostics.
            logger.exception("Starting with no models available")
            service = InferenceService(settings.artifacts_dir, eager=False)

        # The threadpool must be able to hold every request the service will
        # accept; otherwise anyio's own limiter becomes a hidden second queue
        # and the queue-depth metric stops describing reality.
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = settings.max_concurrency + settings.max_queue + 8

        app.state.service = service
        app.state.settings = settings
        app.state.started_at = started
        yield
        app.state.service = None

    app = FastAPI(
        title="Titanic Inference API",
        description=(
            "Instrumented inference over the trained Titanic model ladder. "
            "Every endpoint is a thin adapter over titanic.service.InferenceService, "
            "which is the same object the Streamlit app calls in-process."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id and emit one structured log line per request.

        The id is echoed in the ``X-Request-ID`` header so a client that sees
        an error can quote it, and the server-side log holds the traceback
        that the client is deliberately not shown.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000

        response.headers["X-Request-ID"] = request_id
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "total_ms": round(duration_ms, 2),
                }
            )
        )
        return response

    def get_service(request: Request) -> InferenceService:
        """Return the shared service instance.

        Args:
            request: The incoming request.

        Returns:
            The application's :class:`InferenceService`.

        Raises:
            NoArtifactsError: If the service failed to start.
        """
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise NoArtifactsError(
                "The inference service is unavailable. Train models with "
                "'python train.py --model all' and restart the API."
            )
        return service

    # ------------------------------------------------------------------
    # Exception handling: one JSON shape, never a traceback
    # ------------------------------------------------------------------

    def error_response(request: Request, exc: Exception) -> JSONResponse:
        """Convert an exception into the project's single error shape.

        Args:
            request: The failing request.
            exc: The exception raised.

        Returns:
            A JSON response carrying ``error``, ``message`` and ``details``.
        """
        status, code = ERROR_MAP.get(type(exc), (500, "internal"))
        request_id = getattr(request.state, "request_id", "unknown")

        if status >= 500 and code == "internal":
            # Unexpected: log the traceback for the operator, return only the
            # request id to the caller.
            logger.exception("[%s] unhandled error on %s", request_id, request.url.path)
            message = (
                "An internal error occurred. Quote request id " f"{request_id} when reporting it."
            )
        else:
            logger.warning("[%s] %s: %s", request_id, code, exc)
            message = str(exc)

        headers = {"X-Request-ID": request_id}
        if code in {"queue_full", "queue_timeout"}:
            # Tell the client when to come back instead of letting it hammer.
            headers["Retry-After"] = "1"

        details: dict[str, Any] = {}
        if isinstance(exc, ModelNotFoundError):
            details["available_models"] = get_service(request).models
        if isinstance(exc, QueueFullError | QueueTimeoutError):
            details.update(get_service(request).queue_state())

        return JSONResponse(
            status_code=status,
            headers=headers,
            content=ErrorResponse(error=code, message=message, details=details).model_dump(),
        )

    for exception_type in ERROR_MAP:
        app.add_exception_handler(exception_type, error_response)
    app.add_exception_handler(Exception, error_response)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report liveness and which models are loaded."""
        service = get_service(request)
        loaded = service.loaded_models
        return HealthResponse(
            status="ok" if loaded else "degraded",
            models_loaded=loaded,
            uptime_s=round(time.time() - request.app.state.started_at, 1),
            version=__version__,
        )

    @app.get("/models")
    async def models(request: Request) -> dict[str, Any]:
        """List every registered model with its validation metrics."""
        service = get_service(request)
        return {"default": service.default_model, "models": service.model_info()}

    @app.get("/schema", response_model=SchemaInfo)
    async def schema() -> SchemaInfo:
        """Describe the expected CSV columns."""
        return SchemaInfo()

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, body: PredictRequest) -> PredictResponse:
        """Run inference on a list of passengers."""
        service = get_service(request)
        # exclude_none=False keeps missing Age and Fare as NaN so the fitted
        # imputer handles them, rather than dropping the columns entirely.
        frame = pd.DataFrame([passenger.model_dump() for passenger in body.passengers])

        result = await run_in_threadpool(service.predict, frame, body.model, body.threshold)

        return PredictResponse(
            model=result.model,
            threshold=result.threshold,
            n=result.n,
            predictions=_prediction_rows(result),
            latency_ms=result.latency_ms,
        )

    @app.post("/predict/csv")
    async def predict_csv(
        request: Request,
        file: UploadFile = File(..., description="CSV in the raw Kaggle Titanic schema"),
        model: str | None = Query(default=None),
        threshold: float = Query(default=0.5, ge=0.0, le=1.0),
        accept: str = Header(default="application/json"),
    ) -> Response:
        """Run inference on an uploaded CSV, returning JSON or CSV."""
        service = get_service(request)
        frame = await _read_upload(file)

        result = await run_in_threadpool(service.predict, frame, model, threshold)
        output = result.to_frame(frame)

        if "text/csv" in accept:
            return PlainTextResponse(
                output.to_csv(index=False),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
            )

        note = None
        if "Survived" not in frame.columns:
            note = (
                "No 'Survived' column found, so only predictions are returned. "
                "Add labels and call /evaluate for metrics."
            )
        return JSONResponse(
            PredictResponse(
                model=result.model,
                threshold=result.threshold,
                n=result.n,
                predictions=_prediction_rows(result),
                latency_ms=result.latency_ms,
                note=note,
            ).model_dump()
        )

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate(
        request: Request,
        file: UploadFile = File(..., description="Labelled CSV including 'Survived'"),
        model: str | None = Query(default=None),
        threshold: float = Query(default=0.5, ge=0.0, le=1.0),
        n_boot: int = Query(default=1000, ge=0, le=2000),
    ) -> EvaluateResponse:
        """Score a labelled CSV and return metrics with bootstrap intervals."""
        service = get_service(request)
        frame = await _read_upload(file)

        result = await run_in_threadpool(service.evaluate, frame, model, threshold, n_boot)
        return EvaluateResponse(
            model=result.model,
            threshold=result.threshold,
            n=result.n,
            metrics=result.metrics,
            ci95=result.ci95,
            confusion_matrix=result.metrics["confusion_matrix"],
            curves=result.curves,
            latency_ms=result.latency_ms,
        )

    @app.get("/stats")
    async def stats(request: Request) -> dict[str, Any]:
        """Return the JSON snapshot the Ops dashboard renders."""
        return get_service(request).stats()

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        """Expose Prometheus metrics for scraping."""
        return Response(
            content=get_service(request).prometheus_text(), media_type=CONTENT_TYPE_LATEST
        )

    @app.post("/admin/reload")
    async def reload_models(
        request: Request,
        x_admin_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Reload artifacts from disk without restarting the process."""
        # The app's own settings, not a fresh read of the environment: a
        # Depends(get_settings) here would ignore whatever create_app() was
        # configured with, which is exactly what a test would pass in.
        settings = request.app.state.settings
        if not settings.admin_token:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error="reload_disabled",
                    message="Reloading is disabled. Set TITANIC_ADMIN_TOKEN to enable it.",
                ).model_dump(),
            )
        # Constant-time comparison is unnecessary here (a local dev endpoint),
        # but refusing without leaking whether the token merely had the wrong
        # length is still the right shape.
        if x_admin_token != settings.admin_token:
            return JSONResponse(
                status_code=403,
                content=ErrorResponse(
                    error="forbidden", message="Invalid or missing X-Admin-Token header."
                ).model_dump(),
            )

        service = get_service(request)
        loaded = await run_in_threadpool(service.reload)
        return {"reloaded": loaded}

    return app


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded CSV into a dataframe.

    Args:
        file: The uploaded file.

    Returns:
        The parsed dataframe.

    Raises:
        SchemaError: If the upload is not a readable, non-empty CSV, or is
            larger than the row limit.
    """
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise SchemaError(f"Expected a .csv file, got {file.filename!r}.")

    payload = await file.read()
    if not payload:
        raise SchemaError("The uploaded file is empty.")

    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise SchemaError(f"Could not parse the upload as CSV: {exc}") from exc

    if frame.empty:
        raise SchemaError("The uploaded CSV has a header but no data rows.")
    if len(frame) > MAX_ROWS:
        raise SchemaError(
            f"The upload has {len(frame):,} rows but the limit is {MAX_ROWS:,}. "
            "Split it into smaller files."
        )
    return frame


#: Module-level application for `uvicorn api.main:app`.
app = create_app()
