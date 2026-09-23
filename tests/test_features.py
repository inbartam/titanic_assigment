"""Tests for feature engineering.

Every function in ``titanic.features`` is pure: same input, same output, no
fitted state, no dependence on other rows in the batch. These tests pin that
contract down, because a feature that quietly depends on the rest of the batch
would behave differently at training time and at single-row inference time.
That is the train/serve skew that ``docs/DECISIONS.md`` rules out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titanic.features import (
    deck_from_cabin,
    engineer,
    extract_title,
    family_size,
    is_alone,
    log_fare,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small raw dataframe covering the interesting edge cases."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 1, 0],
            "Pclass": [3, 1, 3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                "Heikkinen, Miss. Laina",
                "Peel, Master. Edward",
                "Ostby, Dr. Engelhart Cornelius",
            ],
            "Sex": ["male", "female", "female", "male", "male"],
            "Age": [22.0, 38.0, None, 4.0, 65.0],
            "SibSp": [1, 1, 0, 3, 0],
            "Parch": [0, 0, 0, 2, 1],
            "Fare": [7.25, 71.2833, 7.925, 27.9, None],
            "Cabin": [None, "C85", None, "F G73", "B42"],
            "Embarked": ["S", "C", "S", None, "S"],
        }
    )


class TestExtractTitle:
    def test_extracts_the_four_common_titles(self) -> None:
        names = pd.Series(
            [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Peel, Master. Edward",
            ]
        )
        assert list(extract_title(names)) == ["Mr", "Mrs", "Miss", "Master"]

    def test_normalises_french_and_modern_variants(self) -> None:
        # Mlle and Ms are Miss; Mme is Mrs. Collapsing them keeps the rare
        # levels from fragmenting the vocabulary on a 712-row training split.
        names = pd.Series(
            [
                "Boulos, Mlle. Nourelain",
                "Hirvonen, Ms. Helga",
                "Mme. Something",
            ]
        )
        assert list(extract_title(names)) == ["Miss", "Miss", "Mrs"]

    def test_uncommon_titles_collapse_to_rare(self) -> None:
        names = pd.Series(
            [
                "Ostby, Dr. Engelhart",
                "Byles, Rev. Thomas",
                "Weir, Col. John",
                "Duff Gordon, Lady. Lucille",
            ]
        )
        assert list(extract_title(names)) == ["Rare", "Rare", "Rare", "Rare"]

    def test_unparseable_name_falls_back_to_rare(self) -> None:
        # Never raise on a malformed name: inference must degrade, not crash.
        names = pd.Series(["no comma or period here", "", None])
        assert list(extract_title(names)) == ["Rare", "Rare", "Rare"]

    def test_is_case_insensitive_on_the_title(self) -> None:
        names = pd.Series(["Smith, MR. John", "Smith, mrs. Jane"])
        assert list(extract_title(names)) == ["Mr", "Mrs"]


class TestFamilySize:
    def test_counts_siblings_spouses_parents_children_and_self(self) -> None:
        df = pd.DataFrame({"SibSp": [0, 1, 3], "Parch": [0, 0, 2]})
        assert list(family_size(df)) == [1, 2, 6]

    def test_is_alone_is_family_size_of_one(self) -> None:
        sizes = pd.Series([1, 2, 6])
        assert list(is_alone(sizes)) == [1, 0, 0]


class TestDeckFromCabin:
    def test_takes_the_first_letter(self) -> None:
        cabins = pd.Series(["C85", "B42", "E46"])
        assert list(deck_from_cabin(cabins)) == ["C", "B", "E"]

    def test_multiple_cabins_use_the_first(self) -> None:
        # "F G73" means two cabins; the deck letter is the first token.
        assert list(deck_from_cabin(pd.Series(["F G73", "B57 B59 B63"]))) == ["F", "B"]

    def test_missing_cabin_becomes_u(self) -> None:
        # 77% of Cabin is missing, and the missingness itself is informative
        # (no recorded cabin correlates with lower class and lower survival),
        # so it becomes its own level rather than being imputed away.
        assert list(deck_from_cabin(pd.Series([None, np.nan, ""]))) == ["U", "U", "U"]


class TestLogFare:
    def test_applies_log1p(self) -> None:
        fares = pd.Series([0.0, 7.25, 512.3292])
        expected = np.log1p(fares.to_numpy())
        np.testing.assert_allclose(log_fare(fares).to_numpy(), expected)

    def test_missing_fare_stays_missing(self) -> None:
        # Imputation is the Preprocessor's job (it must be a *fitted* value),
        # not a pure feature function's.
        assert log_fare(pd.Series([None])).isna().all()


class TestEngineer:
    def test_adds_every_engineered_column(self, raw_df: pd.DataFrame) -> None:
        out = engineer(raw_df)
        for col in ("Title", "FamilySize", "IsAlone", "Deck", "LogFare"):
            assert col in out.columns

    def test_does_not_mutate_the_input(self, raw_df: pd.DataFrame) -> None:
        before = raw_df.copy(deep=True)
        engineer(raw_df)
        pd.testing.assert_frame_equal(raw_df, before)

    def test_works_on_a_single_row(self, raw_df: pd.DataFrame) -> None:
        # The decisive property: every feature must be computable from one row
        # alone. Anything needing the rest of the batch is train/serve skew.
        out = engineer(raw_df.iloc[[0]])
        assert len(out) == 1
        assert out.loc[0, "Title"] == "Mr"
        assert out.loc[0, "FamilySize"] == 2

    def test_row_features_are_independent_of_the_batch(self, raw_df: pd.DataFrame) -> None:
        full = engineer(raw_df)
        single = engineer(raw_df.iloc[[2]]).reset_index(drop=True)
        engineered = ["Title", "FamilySize", "IsAlone", "Deck", "LogFare"]
        pd.testing.assert_frame_equal(
            full.loc[[2], engineered].reset_index(drop=True),
            single[engineered],
        )

    def test_absent_optional_columns_are_treated_as_all_missing(self, raw_df: pd.DataFrame) -> None:
        # docs/ARCHITECTURE.md: an absent Cabin/Embarked column is equivalent
        # to a column of NaN, not an error.
        out = engineer(raw_df.drop(columns=["Cabin", "Embarked"]))
        assert list(out["Deck"].unique()) == ["U"]
        assert out["Embarked"].isna().all()

    def test_does_not_add_batch_dependent_features(self, raw_df: pd.DataFrame) -> None:
        # Guard against a future contributor reintroducing ticket-group size or
        # fare-per-person, which cannot be computed for a single row.
        out = engineer(raw_df)
        assert "TicketGroupSize" not in out.columns
        assert "FarePerPerson" not in out.columns
