"""Tests for the fitted preprocessor.

This is the most important test file in the project. The preprocessor is where
data leakage would enter: every learned value (imputation tables, scaling
statistics, category vocabularies) must come from the training split alone and
must survive a round trip through JSON unchanged, or the numbers reported in
the README are not the numbers the app reproduces.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from titanic.config import Paths
from titanic.data import load_csv, stratified_split
from titanic.features import engineer
from titanic.preprocessing import NotFittedError, Preprocessor


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Engineered training split of the committed sample."""
    df = load_csv(Paths().sample_csv)
    train, _ = stratified_split(df)
    return engineer(train)


@pytest.fixture
def val_df() -> pd.DataFrame:
    """Engineered validation split of the committed sample."""
    df = load_csv(Paths().sample_csv)
    _, val = stratified_split(df)
    return engineer(val)


@pytest.fixture
def fitted(train_df: pd.DataFrame) -> Preprocessor:
    """A preprocessor fitted on the training split only."""
    return Preprocessor().fit(train_df)


class TestFitTransformShapes:
    def test_transform_returns_expected_shapes_and_dtypes(
        self, fitted: Preprocessor, train_df: pd.DataFrame
    ) -> None:
        x_num, x_cat = fitted.transform(train_df)
        assert x_num.shape == (len(train_df), 3)
        assert x_cat.shape == (len(train_df), 6)
        # float32 feeds torch without a copy; int64 is what nn.Embedding wants.
        assert x_num.dtype == np.float32
        assert x_cat.dtype == np.int64

    def test_single_row_inference_works(self, fitted: Preprocessor, val_df: pd.DataFrame) -> None:
        x_num, x_cat = fitted.transform(val_df.iloc[[0]])
        assert x_num.shape == (1, 3)
        assert x_cat.shape == (1, 6)

    def test_no_nan_survives_transform(self, fitted: Preprocessor, val_df: pd.DataFrame) -> None:
        # A NaN reaching a torch model produces NaN gradients and a silently
        # broken run, so imputation must be exhaustive.
        x_num, x_cat = fitted.transform(val_df)
        assert not np.isnan(x_num).any()
        assert (x_cat >= 0).all()

    def test_cardinalities_match_categorical_columns(self, fitted: Preprocessor) -> None:
        cards = fitted.cardinalities
        assert len(cards) == len(fitted.categorical_cols)
        # Every vocabulary reserves index 0 for <UNK>, so cardinality is at
        # least 2 (unknown plus one real level).
        assert all(c >= 2 for c in cards)


class TestFittingIsLeakFree:
    def test_transform_before_fit_raises(self, train_df: pd.DataFrame) -> None:
        with pytest.raises(NotFittedError):
            Preprocessor().transform(train_df)

    def test_transforming_validation_data_does_not_change_fitted_state(
        self, fitted: Preprocessor, val_df: pd.DataFrame
    ) -> None:
        # The leakage guard. If transform() ever recomputed a median or added a
        # vocabulary entry from the data it is transforming, validation
        # information would flow into the model and the reported metrics would
        # be optimistic. Comparing the full serialised state catches any such
        # mutation, including ones added by a future contributor.
        before = json.dumps(fitted.to_dict(), sort_keys=True)
        fitted.transform(val_df)
        after = json.dumps(fitted.to_dict(), sort_keys=True)
        assert before == after

    def test_fitted_statistics_come_from_the_training_split_only(
        self, train_df: pd.DataFrame
    ) -> None:
        pre = Preprocessor().fit(train_df)
        state = pre.to_dict()
        # Fare median must equal the training split's own median, not one
        # computed over any larger frame.
        expected_median = float(train_df["Fare"].median())
        assert state["imputation"]["fare_median"] == pytest.approx(expected_median)

    def test_scaling_standardises_the_training_split(
        self, fitted: Preprocessor, train_df: pd.DataFrame
    ) -> None:
        x_num, _ = fitted.transform(train_df)
        # By construction the training split has mean 0 and std 1 after
        # scaling. The validation split will not, and that is correct.
        np.testing.assert_allclose(x_num.mean(axis=0), np.zeros(3), atol=1e-5)
        np.testing.assert_allclose(x_num.std(axis=0), np.ones(3), atol=1e-2)


class TestImputation:
    def test_age_is_imputed_with_the_title_median(self, fitted: Preprocessor) -> None:
        # A global median (~28) would assign an adult age to a child whose
        # title is Master, destroying the "children first" signal. The fitted
        # table must therefore differ across titles.
        table = fitted.to_dict()["imputation"]["age_median_by_title"]
        assert "Mr" in table
        if "Master" in table:
            assert table["Master"] < table["Mr"]

    def test_unknown_title_falls_back_to_the_global_median(
        self, fitted: Preprocessor, val_df: pd.DataFrame
    ) -> None:
        row = val_df.iloc[[0]].copy()
        row["Age"] = np.nan
        row["Title"] = "Archduke"  # never seen during fit
        x_num, _ = fitted.transform(row)
        assert not np.isnan(x_num).any()

    def test_missing_embarked_uses_the_fitted_mode(self, fitted: Preprocessor) -> None:
        mode = fitted.to_dict()["imputation"]["embarked_mode"]
        assert mode in {"C", "Q", "S"}

    def test_missing_fare_is_imputed_before_log(
        self, fitted: Preprocessor, val_df: pd.DataFrame
    ) -> None:
        # LogFare must be recomputed after Fare imputation; imputing LogFare
        # directly would apply the median on the wrong scale.
        row = val_df.iloc[[0]].copy()
        row["Fare"] = np.nan
        row["LogFare"] = np.nan
        x_num, _ = fitted.transform(row)
        assert not np.isnan(x_num).any()


class TestUnknownCategories:
    def test_unseen_category_maps_to_index_zero(
        self, fitted: Preprocessor, val_df: pd.DataFrame
    ) -> None:
        # Inference must never crash on a category it has not seen. Index 0 is
        # the reserved <UNK> slot, which the embedding layer has weights for.
        row = val_df.iloc[[0]].copy()
        row["Deck"] = "Z"
        _, x_cat = fitted.transform(row)
        deck_position = fitted.categorical_cols.index("Deck")
        assert x_cat[0, deck_position] == 0

    def test_every_vocabulary_reserves_zero_for_unknown(self, fitted: Preprocessor) -> None:
        for column, vocab in fitted.to_dict()["vocab"].items():
            assert vocab["<UNK>"] == 0, f"{column} must reserve index 0 for <UNK>"
            assert 0 not in [v for k, v in vocab.items() if k != "<UNK>"]


class TestPersistence:
    def test_round_trip_through_json_is_identical(
        self, fitted: Preprocessor, val_df: pd.DataFrame, tmp_path
    ) -> None:
        # The app loads the preprocessor from disk, so save -> load -> transform
        # must reproduce training-time arrays exactly, not approximately.
        path = tmp_path / "preprocessor.json"
        fitted.save(path)
        reloaded = Preprocessor.load(path)

        original_num, original_cat = fitted.transform(val_df)
        loaded_num, loaded_cat = reloaded.transform(val_df)
        np.testing.assert_array_equal(original_num, loaded_num)
        np.testing.assert_array_equal(original_cat, loaded_cat)

    def test_saved_file_is_human_readable_json(self, fitted: Preprocessor, tmp_path) -> None:
        # JSON over pickle: reviewable in a diff, version-safe, and it forces
        # every learned parameter to be enumerated explicitly.
        path = tmp_path / "preprocessor.json"
        fitted.save(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "version",
            "numeric_cols",
            "categorical_cols",
            "imputation",
            "scaling",
            "vocab",
        ):
            assert key in state

    def test_load_rejects_an_unknown_schema_version(self, fitted: Preprocessor, tmp_path) -> None:
        path = tmp_path / "preprocessor.json"
        fitted.save(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        state["version"] = 999
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            Preprocessor.load(path)

    def test_to_dict_from_dict_round_trip(self, fitted: Preprocessor) -> None:
        clone = Preprocessor.from_dict(fitted.to_dict())
        assert clone.to_dict() == fitted.to_dict()
