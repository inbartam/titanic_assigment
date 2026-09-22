"""Typed configuration objects and project-wide constants.

Every path in the project flows through :class:`Paths`. Nothing anywhere else
is allowed to hardcode an absolute path -- that is what makes the repository
work identically on the author's Windows machine and a reviewer's laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Schema constants. These describe the *raw* Kaggle Titanic CSV and are the
# single source of truth shared by data validation, the Pydantic API schemas
# and the Streamlit "expected schema" panel.
# --------------------------------------------------------------------------

#: Columns a CSV must contain before we will run inference on it.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
)

#: Columns we use when present but never require. ``Cabin`` and ``Embarked``
#: are heavily missing in the source data, so an absent column is treated as
#: "all values missing" rather than as an error.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "PassengerId",
    "Cabin",
    "Embarked",
    "Ticket",
    "Survived",
)

#: The label column. Its presence switches the app from inference-only mode to
#: full evaluation mode.
TARGET_COLUMN: str = "Survived"

#: Columns that must parse as numbers. Validation reports the offending column
#: by name instead of letting pandas raise a generic cast error later.
NUMERIC_COLUMNS: tuple[str, ...] = ("Age", "SibSp", "Parch", "Fare", "Pclass")

#: Survival rate of the full Kaggle training set (342/891). The Ops tab
#: compares the served positive rate against this value as a cheap drift
#: signal, so it is recorded here rather than recomputed at runtime.
TRAIN_BASE_RATE: float = 0.3838


@dataclass(frozen=True)
class Paths:
    """Filesystem layout, resolved relative to the repository root.

    ``frozen=True`` makes instances hashable and prevents a caller from
    mutating shared state by accident; a caller that needs a different layout
    constructs a new instance (``Paths(root=tmp_path)`` in tests).

    Attributes:
        root: Repository root. Defaults to two parents above this file
            (``src/titanic/config.py`` -> ``src/titanic`` -> ``src`` -> root).
        data: Directory holding ``train.csv`` (git-ignored) and the committed
            ``sample_train.csv``.
        artifacts: Output directory for trained models and their metadata.
        notebooks: EDA notebook location.
        docs: Documentation, including generated screenshots.
    """

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def data(self) -> Path:
        """Directory containing dataset CSVs."""
        return self.root / "data"

    @property
    def train_csv(self) -> Path:
        """Full Kaggle training file, downloaded on demand and git-ignored."""
        return self.data / "train.csv"

    @property
    def sample_csv(self) -> Path:
        """Committed 100-row stratified sample, used when Kaggle is unavailable."""
        return self.data / "sample_train.csv"

    @property
    def artifacts(self) -> Path:
        """Root of the per-model artifact directories."""
        return self.root / "artifacts"

    @property
    def registry(self) -> Path:
        """Index of every model that has been trained successfully."""
        return self.artifacts / "registry.json"

    @property
    def notebooks(self) -> Path:
        """Directory holding the EDA notebook."""
        return self.root / "notebooks"

    @property
    def docs(self) -> Path:
        """Documentation directory."""
        return self.root / "docs"


@dataclass(frozen=True)
class SplitConfig:
    """Parameters of the train/validation split.

    Attributes:
        test_size: Fraction held out for the single final evaluation. 0.2 of
            891 rows leaves 179 validation rows.
        seed: Seed for the split. Fixed so the held-out rows are identical on
            every run and on every machine.
        stratify: Keep the survival ratio identical in both halves. With a 38%
            positive rate and n=179, an unstratified split can move the class
            balance by several points and shift accuracy by 1-2 points purely
            through sampling noise.
        inner_val_size: Fraction of the *training* split carved out for early
            stopping. The held-out validation set is never used for this.
    """

    test_size: float = 0.2
    seed: int = 42
    stratify: bool = True
    inner_val_size: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters shared by every PyTorch model.

    Per-architecture settings (hidden sizes, dropout) live in the model config
    written to ``model_config.json``; these are the ones the training loop
    itself consumes.

    Attributes:
        lr: AdamW learning rate.
        weight_decay: L2 regularisation. One of the three cheap regularisers
            (with dropout and early stopping) that matter at n=712.
        batch_size: 64 gives ~11 optimisation steps per epoch on the training
            split -- enough gradient noise to regularise, few enough to be fast.
        max_epochs: Upper bound; early stopping almost always triggers first.
        patience: Epochs without inner-validation improvement before stopping.
        seed: Seed for weight initialisation and DataLoader shuffling.
        cv_folds: Folds for the stratified cross-validation used to *select*
            hyperparameters inside the training split.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 20
    seed: int = 42
    cv_folds: int = 5
