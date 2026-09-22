"""Guards that keep the EDA notebook honest.

``PLAN.md``'s risk register names one specific failure mode: the notebook
quietly re-implements feature logic, drifts from ``src/titanic/``, and then
documents decisions the model does not actually follow. These tests make that
drift a test failure rather than something a reviewer has to notice.

They are static checks over the notebook JSON -- they never execute it, so the
suite stays fast.
"""

from __future__ import annotations

import json

import pytest

from titanic.config import Paths
from titanic.features import ENGINEERED_COLUMNS
from titanic.preprocessing import DEFAULT_CATEGORICAL_COLS, DEFAULT_NUMERIC_COLS

NOTEBOOK = Paths().notebooks / "eda.ipynb"

#: Functions that belong to titanic.features. If the notebook defines one of
#: these itself, the two implementations can disagree.
FEATURE_FUNCTIONS = (
    "extract_title",
    "family_size",
    "is_alone",
    "deck_from_cabin",
    "log_fare",
    "engineer",
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    """Parsed notebook JSON."""
    if not NOTEBOOK.is_file():
        pytest.skip(f"{NOTEBOOK} not present")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_source(notebook: dict) -> str:
    """All code-cell source concatenated into one string."""
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_engineered_columns_cover_what_the_preprocessor_needs() -> None:
    # The features module and the preprocessor must agree on the feature set.
    # Every modelled column is either raw or produced by engineer().
    raw_columns = {"Pclass", "Sex", "Embarked", "Age", "Fare", "SibSp", "Parch"}
    modelled = set(DEFAULT_NUMERIC_COLS) | set(DEFAULT_CATEGORICAL_COLS)
    unaccounted = modelled - raw_columns - set(ENGINEERED_COLUMNS)
    assert unaccounted == set(), f"columns neither raw nor engineered: {unaccounted}"


def test_notebook_imports_feature_logic_instead_of_redefining_it(code_source: str) -> None:
    assert "from titanic.features import" in code_source
    for name in FEATURE_FUNCTIONS:
        assert f"def {name}" not in code_source, (
            f"The notebook defines {name}() itself. Import it from titanic.features so the "
            "notebook and the training pipeline cannot disagree."
        )


def test_notebook_never_touches_the_forbidden_files(code_source: str) -> None:
    # The assignment allows train.csv only.
    for forbidden in ("test.csv", "gender_submission.csv"):
        assert forbidden not in code_source, f"notebook references {forbidden}"


def test_notebook_discards_the_validation_split(code_source: str) -> None:
    # The notebook splits first and explores the training half only. Deleting
    # the validation frame is what makes that claim enforceable rather than a
    # promise in a markdown cell.
    assert "stratified_split" in code_source
    assert "del val_raw" in code_source


def test_notebook_ran_without_errors(notebook: dict) -> None:
    # A committed notebook with stored tracebacks is worse than no notebook.
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"cell {index} contains a stored {output.get('ename')}. "
                "Re-run the notebook before committing."
            )


def test_notebook_outputs_are_committed(notebook: dict) -> None:
    # Outputs must be saved so the analysis renders on GitHub without the
    # reviewer having to install anything or fetch the dataset.
    executed = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and cell.get("execution_count")
    ]
    assert len(executed) >= 10, "notebook appears not to have been run before committing"
