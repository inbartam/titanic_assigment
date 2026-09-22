"""Tests for dataset loading, schema validation and splitting.

These guard the contract that every other layer depends on: if a CSV passes
``validate_schema`` here, the preprocessor, the models, the app and the API can
all assume the columns they need exist and hold sane values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from titanic.config import SplitConfig
from titanic.data import (
    SchemaError,
    load_csv,
    make_sample,
    stratified_split,
    validate_schema,
)


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """A minimal dataframe that satisfies the raw Kaggle Titanic schema."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 1],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Allen, Mr. William Henry",
            ],
            "Sex": ["male", "female", "female", "male"],
            "Age": [22.0, 38.0, None, 35.0],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
            "Fare": [7.25, 71.28, 7.92, 8.05],
            "Cabin": [None, "C85", None, None],
            "Embarked": ["S", "C", "S", "S"],
        }
    )


class TestValidateSchema:
    def test_accepts_valid_dataframe(self, valid_df: pd.DataFrame) -> None:
        validate_schema(valid_df)  # must not raise

    def test_accepts_missing_age_values(self, valid_df: pd.DataFrame) -> None:
        # Age has 177 blanks in the real dataset. Missing is legitimate; the
        # preprocessor imputes it. Only non-numeric *text* is an error.
        assert valid_df["Age"].isna().any()
        validate_schema(valid_df)

    def test_rejects_missing_required_column(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.drop(columns=["Pclass", "Sex"])
        with pytest.raises(SchemaError) as exc:
            validate_schema(df)
        # The message must name the offending columns: an error the user can act on.
        assert "Pclass" in str(exc.value)
        assert "Sex" in str(exc.value)

    def test_tolerates_absent_optional_columns(self, valid_df: pd.DataFrame) -> None:
        # Cabin and Embarked absent entirely is equivalent to all-missing.
        validate_schema(valid_df.drop(columns=["Cabin", "Embarked", "PassengerId"]))

    def test_rejects_non_numeric_in_numeric_column(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        # astype(object) mirrors what pandas actually produces when a real CSV
        # has text in a numeric column, and avoids a dtype FutureWarning.
        df["Age"] = df["Age"].astype(object)
        df.loc[0, "Age"] = "twenty-two"
        with pytest.raises(SchemaError, match="Age"):
            validate_schema(df)

    def test_rejects_target_outside_zero_one(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df.loc[0, "Survived"] = 2
        with pytest.raises(SchemaError, match="Survived"):
            validate_schema(df)

    def test_require_target_flag(self, valid_df: pd.DataFrame) -> None:
        unlabelled = valid_df.drop(columns=["Survived"])
        validate_schema(unlabelled)  # inference-only: fine
        with pytest.raises(SchemaError, match="Survived"):
            validate_schema(unlabelled, require_target=True)

    def test_extra_columns_are_ignored(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.assign(MyNotes="hello")
        validate_schema(df)  # warns, does not raise


class TestLoadCsv:
    def test_reads_the_committed_sample(self) -> None:
        from titanic.config import Paths

        df = load_csv(Paths().sample_csv)
        assert len(df) == 100
        validate_schema(df, require_target=True)

    def test_missing_file_names_the_path(self, tmp_path) -> None:
        target = tmp_path / "nope.csv"
        with pytest.raises(FileNotFoundError, match="nope.csv"):
            load_csv(target)

    def test_empty_file_raises_schema_error(self, tmp_path) -> None:
        target = tmp_path / "empty.csv"
        target.write_text("", encoding="utf-8")
        with pytest.raises(SchemaError):
            load_csv(target)


class TestStratifiedSplit:
    def test_preserves_class_balance(self) -> None:
        from titanic.config import Paths

        df = load_csv(Paths().sample_csv)
        train_df, val_df = stratified_split(df)
        base_rate = df["Survived"].mean()
        # Stratification cannot do better than whole rows: with 20 validation
        # rows and 38 positives, the ideal 7.6 positives must round to 8, which
        # is already a 0.02 deviation. So the tolerance is half a row in each
        # half -- anything larger means stratification genuinely failed.
        for split in (train_df, val_df):
            tolerance = 0.5 / len(split)
            assert abs(split["Survived"].mean() - base_rate) <= tolerance

    def test_sizes_and_disjointness(self) -> None:
        from titanic.config import Paths

        df = load_csv(Paths().sample_csv)
        train_df, val_df = stratified_split(df, SplitConfig(test_size=0.2))
        assert len(train_df) + len(val_df) == len(df)
        assert len(val_df) == 20
        # No row may appear in both halves, or the validation score is leaked.
        overlap = set(train_df["PassengerId"]) & set(val_df["PassengerId"])
        assert overlap == set()

    def test_is_deterministic(self) -> None:
        from titanic.config import Paths

        df = load_csv(Paths().sample_csv)
        first, _ = stratified_split(df)
        second, _ = stratified_split(df)
        pd.testing.assert_frame_equal(first, second)

    def test_requires_labels(self, valid_df: pd.DataFrame) -> None:
        with pytest.raises(SchemaError, match="Survived"):
            stratified_split(valid_df.drop(columns=["Survived"]))


class TestMakeSample:
    def test_sample_is_stratified_and_sorted(self) -> None:
        from titanic.config import Paths

        df = load_csv(Paths().train_csv) if Paths().train_csv.exists() else None
        if df is None:
            pytest.skip("full train.csv not present (fetch it with python -m titanic.data --fetch)")
        sample = make_sample(df, n=100)
        assert len(sample) == 100
        assert abs(sample["Survived"].mean() - df["Survived"].mean()) < 0.02
        # Sorted output keeps the committed CSV's diff stable between runs.
        assert sample["PassengerId"].is_monotonic_increasing

    def test_returns_all_rows_when_n_exceeds_size(self, valid_df: pd.DataFrame) -> None:
        assert len(make_sample(valid_df, n=999)) == len(valid_df)
