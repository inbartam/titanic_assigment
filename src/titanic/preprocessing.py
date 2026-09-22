"""The fitted preprocessor: imputation, scaling and categorical encoding.

Everything this class learns -- median ages per title, the fare median, the
embarkation mode, per-column scaling statistics and the category vocabularies
-- is fitted **once, on the training split only**, and then serialised to JSON.
Validation data and every inference request are transformed with those frozen
values. That single rule is what keeps the reported metrics honest.

The serialised form is JSON rather than a pickle because JSON is reviewable in
a diff, survives library upgrades, and forces every learned parameter to be
written out explicitly instead of hidden inside an object graph.

Typical use::

    pre = Preprocessor().fit(engineer(train_df))
    pre.save(path)
    x_num, x_cat = Preprocessor.load(path).transform(engineer(any_df))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from titanic.features import RARE_TITLE, UNKNOWN_DECK
from titanic.utils import get_logger

logger = get_logger(__name__)

#: Schema version of ``preprocessor.json``. Bumped whenever the layout changes
#: so an old artifact fails loudly instead of being silently misread.
SCHEMA_VERSION = 1

#: Token reserved at index 0 of every vocabulary. Any category not seen during
#: fitting maps here, so inference on unexpected input never raises and the
#: embedding layer always has a weight row to use.
UNKNOWN_TOKEN = "<UNK>"

#: Standardised continuous features.
DEFAULT_NUMERIC_COLS: tuple[str, ...] = ("Age", "LogFare", "FamilySize")

#: Index-encoded discrete features. ``Pclass`` is here rather than in the
#: numeric list because its effect is not linear (1st class outperforms 2nd by
#: far more than 2nd outperforms 3rd) and three levels cost nothing.
DEFAULT_CATEGORICAL_COLS: tuple[str, ...] = (
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "IsAlone",
)

#: Guard against a degenerate scale factor. A constant column has std 0, and
#: dividing by it would produce inf; substituting 1.0 leaves such a column at 0
#: after centring, which is the correct "carries no information" encoding.
_MIN_STD = 1e-8


class NotFittedError(RuntimeError):
    """Raised when :meth:`Preprocessor.transform` is called before ``fit``.

    Defined here rather than imported from scikit-learn so the error belongs to
    this project's own exception hierarchy -- the service layer maps its own
    typed exceptions to HTTP codes and should not depend on sklearn's tree.
    """


class Preprocessor:
    """Learns imputation, scaling and encoding parameters from training data.

    The class deliberately exposes its learned state through
    :meth:`to_dict`, so tests can assert that transforming validation data
    leaves every fitted value untouched.

    Attributes:
        numeric_cols: Columns standardised to zero mean and unit variance.
        categorical_cols: Columns mapped to integer indices.
    """

    def __init__(
        self,
        numeric_cols: tuple[str, ...] = DEFAULT_NUMERIC_COLS,
        categorical_cols: tuple[str, ...] = DEFAULT_CATEGORICAL_COLS,
    ) -> None:
        """Create an unfitted preprocessor.

        Args:
            numeric_cols: Continuous columns to standardise.
            categorical_cols: Discrete columns to index-encode.
        """
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)

        # Learned state. All None until fit() runs; _fitted is the single flag
        # transform() checks, so a half-populated object can never be used.
        self._fitted = False
        self.age_median_by_title: dict[str, float] = {}
        self.age_global_median: float = 0.0
        self.fare_median: float = 0.0
        self.embarked_mode: str = "S"
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self.vocab: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Learn every parameter from the training split.

        Call this exactly once, on the training split, and never on validation
        or inference data.

        Args:
            df: An **engineered** training dataframe (the output of
                :func:`titanic.features.engineer`).

        Returns:
            ``self``, so the call can be chained: ``Preprocessor().fit(df)``.

        Raises:
            KeyError: If a required engineered column is absent, which means
                the caller forgot to run ``engineer`` first.
        """
        self._require_columns(df)

        # --- imputation values ----------------------------------------
        # Age by title: a global median (~28) would hand an adult age to every
        # child whose title is Master, erasing the "children first" signal.
        title_medians = df.groupby("Title")["Age"].median()
        self.age_median_by_title = {
            str(title): float(value) for title, value in title_medians.items() if pd.notna(value)
        }
        # Fallback for titles absent from the training split, and for the case
        # where a whole title group happens to have no recorded age.
        self.age_global_median = float(df["Age"].median())
        if not np.isfinite(self.age_global_median):
            # Pathological input (every age missing): 28 is the documented
            # population median and keeps the pipeline running.
            self.age_global_median = 28.0

        self.fare_median = float(df["Fare"].median())
        if not np.isfinite(self.fare_median):
            self.fare_median = 0.0

        embarked = df["Embarked"].dropna()
        # mode() can return several values or none; take the first, else "S",
        # which is Southampton, the origin of roughly 72% of passengers.
        self.embarked_mode = str(embarked.mode().iloc[0]) if not embarked.empty else "S"

        # --- scaling statistics ---------------------------------------
        # Computed on the *imputed* frame so the statistics describe exactly
        # the values transform() will later standardise. Computing them on raw
        # data with holes would bias the mean toward the observed subset.
        imputed = self._impute(df)
        for column in self.numeric_cols:
            values = pd.to_numeric(imputed[column], errors="coerce").to_numpy(dtype=float)
            self.num_mean[column] = float(np.nanmean(values))
            std = float(np.nanstd(values))
            self.num_std[column] = std if std > _MIN_STD else 1.0

        # --- category vocabularies ------------------------------------
        for column in self.categorical_cols:
            # sorted() makes the vocabulary deterministic: the same training
            # split always produces byte-identical JSON, which keeps artifact
            # diffs meaningful.
            levels = sorted({self._as_key(v) for v in imputed[column].dropna().unique()})
            self.vocab[column] = {UNKNOWN_TOKEN: 0}
            for index, level in enumerate(levels, start=1):
                self.vocab[column][level] = index

        self._fitted = True
        logger.info(
            "Preprocessor fitted on %d rows: %d numeric, %d categorical (cardinalities %s)",
            len(df),
            len(self.numeric_cols),
            len(self.categorical_cols),
            self.cardinalities,
        )
        return self

    # ------------------------------------------------------------------
    # Transforming
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply the fitted parameters to any dataframe.

        The order of operations matters and is fixed: impute ``Embarked``,
        impute ``Fare``, **recompute** ``LogFare`` from the imputed fare,
        impute ``Age`` by title, then standardise and encode. Recomputing
        ``LogFare`` after imputation is essential -- imputing the log column
        directly would apply a median taken on the wrong scale.

        This method must never modify fitted state. ``tests/test_preprocessing``
        asserts that by comparing the serialised state before and after.

        Args:
            df: An **engineered** dataframe (raw columns plus the engineered
                ones). May be a single row.

        Returns:
            ``(x_num, x_cat)`` where ``x_num`` is float32 of shape
            ``(n, len(numeric_cols))`` and ``x_cat`` is int64 of shape
            ``(n, len(categorical_cols))``. float32 feeds torch without a copy;
            int64 is what ``nn.Embedding`` requires.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            KeyError: If a required engineered column is absent.
        """
        if not self._fitted:
            raise NotFittedError(
                "Preprocessor.transform() called before fit(). Fit on the training split "
                "first, or load a fitted preprocessor with Preprocessor.load(path)."
            )
        self._require_columns(df)

        imputed = self._impute(df)

        # --- numeric block --------------------------------------------
        numeric = np.empty((len(imputed), len(self.numeric_cols)), dtype=np.float32)
        for position, column in enumerate(self.numeric_cols):
            values = pd.to_numeric(imputed[column], errors="coerce").to_numpy(dtype=float)
            # Any residual NaN (a column that imputation does not cover) falls
            # back to the fitted mean, which standardises to exactly 0.
            values = np.where(np.isnan(values), self.num_mean[column], values)
            numeric[:, position] = (values - self.num_mean[column]) / self.num_std[column]

        # --- categorical block ----------------------------------------
        categorical = np.zeros((len(imputed), len(self.categorical_cols)), dtype=np.int64)
        for position, column in enumerate(self.categorical_cols):
            vocabulary = self.vocab[column]
            # .map() leaves unseen levels as NaN; fillna(0) sends them to the
            # reserved <UNK> slot instead of crashing.
            keys = imputed[column].map(self._as_key)
            categorical[:, position] = keys.map(vocabulary).fillna(0).astype(np.int64).to_numpy()

        return numeric, categorical

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using the fitted parameters.

        Shared by :meth:`fit` (to compute scaling statistics on imputed values)
        and :meth:`transform`, so the two can never diverge.

        Args:
            df: Engineered dataframe.

        Returns:
            A copy with no missing values in the modelled columns.
        """
        out = df.copy()

        # Embarked: fitted mode. During fit() the mode is already set from the
        # training data, so this is well defined in both call sites.
        out["Embarked"] = out["Embarked"].fillna(self.embarked_mode)

        # Fare, then LogFare recomputed from it. Order matters (see transform).
        fare = pd.to_numeric(out["Fare"], errors="coerce").fillna(self.fare_median)
        out["Fare"] = fare
        out["LogFare"] = np.log1p(fare.to_numpy(dtype=float))

        # Age by title, falling back to the global median for titles unseen
        # during fitting (an "Archduke" at inference time, for instance).
        age = pd.to_numeric(out["Age"], errors="coerce")
        title_fill = (
            out["Title"].map(self.age_median_by_title).astype(float).fillna(self.age_global_median)
        )
        out["Age"] = age.where(age.notna(), title_fill)

        # Deck and Title never carry NaN out of engineer(), but a caller could
        # hand us a hand-built frame; normalise defensively rather than crash.
        out["Deck"] = out["Deck"].fillna(UNKNOWN_DECK)
        out["Title"] = out["Title"].fillna(RARE_TITLE)

        return out

    def _require_columns(self, df: pd.DataFrame) -> None:
        """Verify the dataframe carries every column this preprocessor needs.

        Args:
            df: Dataframe to check.

        Raises:
            KeyError: Naming the missing columns and the likely cause.
        """
        needed = set(self.numeric_cols) | set(self.categorical_cols) | {"Fare", "Age"}
        missing = sorted(needed - set(df.columns))
        if missing:
            raise KeyError(
                f"Preprocessor requires columns {missing}, which are absent. Run "
                "titanic.features.engineer(df) before fit/transform."
            )

    @staticmethod
    def _as_key(value: Any) -> str:
        """Normalise a category value to a JSON-safe dictionary key.

        Categories arrive as a mix of types -- ``Pclass`` is an int, ``Sex`` a
        string, ``IsAlone`` a numpy int. JSON object keys must be strings, so
        everything is stringified on both the fit and the transform path. This
        also makes ``3`` and ``"3"`` the same level, which is what a user
        hand-editing a CSV would expect.

        Args:
            value: A raw category value.

        Returns:
            Its string key.
        """
        if isinstance(value, float) and float(value).is_integer():
            # Avoid "3.0" and "3" becoming different levels when pandas widens
            # an integer column to float because of a missing value elsewhere.
            return str(int(value))
        return str(value)

    # ------------------------------------------------------------------
    # Introspection and persistence
    # ------------------------------------------------------------------

    @property
    def cardinalities(self) -> list[int]:
        """Vocabulary size per categorical column, including ``<UNK>``.

        Returns:
            One integer per categorical column, in the same order. Embedding
            layers are sized from this list.
        """
        return [len(self.vocab[column]) for column in self.categorical_cols]

    def to_dict(self) -> dict[str, Any]:
        """Serialise every learned parameter to a plain dictionary.

        Returns:
            A JSON-compatible dict matching the schema in
            ``docs/ARCHITECTURE.md §4``.

        Raises:
            NotFittedError: If called before :meth:`fit` -- an unfitted
                preprocessor has nothing meaningful to serialise.
        """
        if not self._fitted:
            raise NotFittedError("Cannot serialise an unfitted Preprocessor; call fit() first.")
        return {
            "version": SCHEMA_VERSION,
            "numeric_cols": list(self.numeric_cols),
            "categorical_cols": list(self.categorical_cols),
            "imputation": {
                "age_median_by_title": dict(self.age_median_by_title),
                "age_global_median": self.age_global_median,
                "fare_median": self.fare_median,
                "embarked_mode": self.embarked_mode,
            },
            "scaling": {"mean": dict(self.num_mean), "std": dict(self.num_std)},
            "vocab": {column: dict(levels) for column, levels in self.vocab.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Preprocessor:
        """Rebuild a fitted preprocessor from its serialised state.

        Args:
            state: A dict produced by :meth:`to_dict`.

        Returns:
            A fitted :class:`Preprocessor`.

        Raises:
            ValueError: If the schema version is unrecognised, or a required
                key is missing.
        """
        version = state.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preprocessor schema version {version!r}; this build expects "
                f"{SCHEMA_VERSION}. Retrain with 'python train.py --model all' to regenerate "
                "the artifacts."
            )

        try:
            instance = cls(
                numeric_cols=tuple(state["numeric_cols"]),
                categorical_cols=tuple(state["categorical_cols"]),
            )
            imputation = state["imputation"]
            instance.age_median_by_title = {
                str(k): float(v) for k, v in imputation["age_median_by_title"].items()
            }
            instance.age_global_median = float(imputation["age_global_median"])
            instance.fare_median = float(imputation["fare_median"])
            instance.embarked_mode = str(imputation["embarked_mode"])
            instance.num_mean = {str(k): float(v) for k, v in state["scaling"]["mean"].items()}
            instance.num_std = {str(k): float(v) for k, v in state["scaling"]["std"].items()}
            instance.vocab = {
                str(column): {str(level): int(index) for level, index in levels.items()}
                for column, levels in state["vocab"].items()
            }
        except KeyError as exc:
            raise ValueError(
                f"Malformed preprocessor state: missing key {exc}. The artifact is corrupt; "
                "regenerate it with 'python train.py'."
            ) from exc

        instance._fitted = True
        return instance

    def save(self, path: str | Path) -> Path:
        """Write the fitted state to a JSON file.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # indent=2 and sorted keys keep the artifact diff-friendly in review;
        # encoding is explicit because Windows would otherwise use cp1252.
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Saved preprocessor to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Preprocessor:
        """Read a fitted preprocessor back from JSON.

        Args:
            path: Path to a file written by :meth:`save`.

        Returns:
            The fitted :class:`Preprocessor`.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON, or its schema version is
                unrecognised.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {path.resolve()}. Train a model first with "
                "'python train.py --model all'."
            )
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not valid JSON ({exc}). Regenerate it with train.py."
            ) from exc
        return cls.from_dict(state)
