"""Pydantic models shared by the API and the app's validation messages.

These types exist so that the HTTP layer rejects exactly what
:func:`titanic.data.validate_schema` rejects, with the same wording. The
constants they validate against come from :mod:`titanic.config`, so there is
one definition of "a valid Titanic row" in the project rather than three.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from titanic.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS


class PassengerIn(BaseModel):
    """One passenger in the raw Kaggle schema.

    Required fields mirror :data:`titanic.config.REQUIRED_COLUMNS`; the rest
    are optional and default to ``None``, which the preprocessor imputes with
    values fitted on the training split.
    """

    # populate_by_name plus the exact Kaggle capitalisation: a client can post
    # the CSV's own column names without renaming anything.
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    Pclass: Annotated[int, Field(ge=1, le=3, description="Ticket class: 1, 2 or 3")]
    Name: Annotated[str, Field(min_length=1, description="Full name, used to extract Title")]
    Sex: Annotated[str, Field(description="male or female")]
    Age: Annotated[float | None, Field(default=None, ge=0, le=120, description="Years")]
    SibSp: Annotated[int, Field(ge=0, le=20, description="Siblings and spouses aboard")]
    Parch: Annotated[int, Field(ge=0, le=20, description="Parents and children aboard")]
    Fare: Annotated[float | None, Field(default=None, ge=0, description="Ticket fare")]

    PassengerId: int | None = None
    Cabin: str | None = None
    Embarked: str | None = None
    Ticket: str | None = None
    Survived: Annotated[int | None, Field(default=None, ge=0, le=1)] = None

    @field_validator("Sex")
    @classmethod
    def normalise_sex(cls, value: str) -> str:
        """Lower-case and check the sex value.

        Args:
            value: Raw input.

        Returns:
            ``"male"`` or ``"female"``.

        Raises:
            ValueError: If the value is neither, naming what was received.
        """
        normalised = value.strip().lower()
        if normalised not in {"male", "female"}:
            raise ValueError(f"Sex must be 'male' or 'female', got {value!r}.")
        return normalised

    @field_validator("Embarked")
    @classmethod
    def normalise_embarked(cls, value: str | None) -> str | None:
        """Upper-case and check the port of embarkation.

        Args:
            value: Raw input, possibly ``None``.

        Returns:
            ``"C"``, ``"Q"``, ``"S"`` or ``None``.

        Raises:
            ValueError: If the value is not a known port.
        """
        if value is None or value == "":
            return None
        normalised = value.strip().upper()
        if normalised not in {"C", "Q", "S"}:
            raise ValueError(
                f"Embarked must be C (Cherbourg), Q (Queenstown) or S (Southampton), "
                f"got {value!r}."
            )
        return normalised


class PredictRequest(BaseModel):
    """Body of ``POST /predict``."""

    model_config = ConfigDict(protected_namespaces=())

    model: str | None = Field(default=None, description="Model name; registry default if omitted")
    threshold: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5
    passengers: Annotated[list[PassengerIn], Field(min_length=1, max_length=10_000)]


class PredictionRow(BaseModel):
    """One row of a prediction response."""

    passenger_id: int | None = None
    p_survived: Annotated[float, Field(ge=0.0, le=1.0)]
    prediction: Literal[0, 1]


class LatencyBreakdown(BaseModel):
    """Per-stage timings, in milliseconds.

    Reported on every response because "where did the time go" is the first
    question anyone asks about a slow prediction, and the answer is usually
    preprocessing rather than the model.
    """

    queue: float = 0.0
    preprocess: float = 0.0
    inference: float = 0.0
    postprocess: float = 0.0
    total: float = 0.0


class PredictResponse(BaseModel):
    """Body of a successful ``POST /predict``."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    threshold: float
    n: int
    predictions: list[PredictionRow]
    latency_ms: LatencyBreakdown
    note: str | None = None


class EvaluateResponse(BaseModel):
    """Body of a successful ``POST /evaluate``."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    threshold: float
    n: int
    metrics: dict[str, Any]
    ci95: dict[str, list[float]]
    confusion_matrix: list[list[int]]
    curves: dict[str, Any]
    latency_ms: dict[str, float]


class ModelSummary(BaseModel):
    """One entry of ``GET /models``."""

    name: str
    framework: str
    n_params: int = 0
    loaded: bool = False
    trained_at: str | None = None
    roc_auc: float | None = None
    accuracy: float | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    validation_ci95: dict[str, list[float]] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["ok", "degraded"]
    models_loaded: list[str]
    uptime_s: float
    version: str


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns.

    One shape everywhere means a client can handle errors generically, and a
    stack trace never reaches the caller -- it is logged server-side against a
    request id instead.
    """

    error: str = Field(description="Machine-readable code, e.g. schema_error")
    message: str = Field(description="Actionable description of what to fix")
    details: dict[str, Any] = Field(default_factory=dict)


class SchemaInfo(BaseModel):
    """The expected input schema, served to the app's help panel."""

    required: list[str] = Field(default_factory=lambda: list(REQUIRED_COLUMNS))
    optional: list[str] = Field(default_factory=lambda: list(OPTIONAL_COLUMNS))
