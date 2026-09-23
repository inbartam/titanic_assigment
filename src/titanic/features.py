"""Feature engineering: pure functions from a raw dataframe to an enriched one.

Everything here is a pure function. No state is fitted, nothing is imputed,
scaled or encoded. That is :mod:`titanic.preprocessing`'s job, because those
operations must *learn* from the training split and therefore cannot live in a
stateless helper.

The decisive property enforced by ``tests/test_features.py`` is that every
feature is computable **from a single row**. Features that depend on the rest
of the batch (ticket-group size, fare per person) are deliberately excluded:
at inference on one passenger their value is always 1, which differs from what
the model saw during training. See ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Titles kept as their own level. Everything else collapses to ``Rare`` so a
#: 712-row training split does not fragment into two dozen one-member levels.
COMMON_TITLES: frozenset[str] = frozenset({"Mr", "Mrs", "Miss", "Master"})

#: Historic and modern spellings folded into the common titles. Mlle
#: (Mademoiselle) and Ms are Miss; Mme (Madame) is Mrs.
TITLE_ALIASES: dict[str, str] = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}

#: Fallback level for any title outside :data:`COMMON_TITLES`.
RARE_TITLE = "Rare"

#: Deck letter used when ``Cabin`` is missing. 77% of cabins are unrecorded and
#: that missingness is itself informative (no recorded cabin correlates with
#: lower class and lower survival), so it becomes a level rather than being
#: imputed away.
UNKNOWN_DECK = "U"

#: Columns :func:`engineer` adds. Exported so the preprocessor and the notebook
#: can assert they agree on the feature set.
ENGINEERED_COLUMNS: tuple[str, ...] = ("Title", "FamilySize", "IsAlone", "Deck", "LogFare")

#: Optional raw columns that may be absent from a user-supplied CSV. An absent
#: column is equivalent to a column of NaN (docs/ARCHITECTURE.md §2).
_OPTIONAL_RAW_COLUMNS: tuple[str, ...] = ("Cabin", "Embarked")

# Primary pattern: the Kaggle name format is "Surname, Title. Given Names", so
# the title is the text between the comma and the first period.
_TITLE_AFTER_COMMA = r",\s*([^.]+)\."

# Fallback for names that do not follow that format, e.g. "Mme. Something".
# Without it such rows would silently become Rare.
_TITLE_AT_START = r"^\s*([A-Za-z]+)\."


def extract_title(names: pd.Series) -> pd.Series:
    """Extract a normalised social title from passenger names.

    Title is the single most useful engineered feature: it compresses sex, age
    and social status into five levels, and it is what makes age imputation
    sensible (``Master`` is a boy at roughly 4 years, ``Mr`` an adult at
    roughly 32).

    Args:
        names: Raw ``Name`` column.

    Returns:
        A series of ``{"Mr", "Mrs", "Miss", "Master", "Rare"}``, aligned to the
        input index. Never contains missing values: an unparseable name
        yields ``"Rare"`` rather than raising, because inference on messy user
        data must degrade rather than crash.
    """
    # astype(str) after fillna: a None in the column would otherwise make the
    # whole .str accessor return NaN for that row rather than an empty match.
    text = names.fillna("").astype(str)

    raw = text.str.extract(_TITLE_AFTER_COMMA, expand=False)
    # Only rows the primary pattern missed are retried with the fallback, so a
    # well-formed name can never be reinterpreted by the looser rule.
    fallback = text.str.extract(_TITLE_AT_START, expand=False)
    # .where rather than .fillna: fillna on an object-dtype column triggers a
    # pandas downcasting FutureWarning, and .where expresses the intent anyway
    # ("keep the primary match, otherwise take the fallback").
    raw = raw.where(raw.notna(), fallback)

    # .str.title() normalises case ("MR." and "mr." both become "Mr") so the
    # vocabulary does not split on capitalisation.
    cleaned = raw.fillna("").str.strip().str.title()
    mapped = cleaned.replace(TITLE_ALIASES)

    return mapped.where(mapped.isin(COMMON_TITLES), RARE_TITLE)


def family_size(df: pd.DataFrame) -> pd.Series:
    """Total family members aboard, including the passenger themselves.

    ``SibSp`` (siblings and spouses) plus ``Parch`` (parents and children) plus
    one. Survival is non-monotonic in this value (families of 2 to 4 fared
    best, while solo travellers and very large families fared worst), which is
    why the raw counts are dropped in favour of the total.

    Args:
        df: Dataframe containing ``SibSp`` and ``Parch``.

    Returns:
        Integer series of family sizes.
    """
    return (df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1).astype(int)


def is_alone(sizes: pd.Series) -> pd.Series:
    """Flag passengers travelling with no family aboard.

    Kept as an explicit binary feature even though it is derivable from
    ``FamilySize``: the survival drop at exactly size 1 is sharp, and a linear
    model cannot represent a step from a single continuous input.

    Args:
        sizes: Output of :func:`family_size`.

    Returns:
        Integer series of 1 (alone) and 0 (with family).
    """
    return (sizes == 1).astype(int)


def deck_from_cabin(cabins: pd.Series) -> pd.Series:
    """Reduce a cabin code to its deck letter.

    The deck approximates price tier and physical distance to the lifeboats.
    A cabin code like ``"C85"`` yields ``"C"``; ``"F G73"`` records two cabins
    and yields the first token's letter, ``"F"``.

    Args:
        cabins: Raw ``Cabin`` column, mostly missing.

    Returns:
        Single-letter deck codes, with :data:`UNKNOWN_DECK` where the cabin is
        missing or blank.
    """
    text = cabins.fillna("").astype(str).str.strip()
    # str[:1] rather than str[0]: it returns "" for an empty string instead of
    # raising, so blank cells flow into the fillna below.
    first_letter = text.str[:1].str.upper()
    return first_letter.replace("", UNKNOWN_DECK).fillna(UNKNOWN_DECK)


def log_fare(fares: pd.Series) -> pd.Series:
    """Compress the heavily right-skewed fare distribution.

    Fares range from 0 to 512 with a long tail, so a linear model would let a
    handful of first-class tickets dominate the coefficient. ``log1p`` (not
    ``log``) is used because a fare of exactly 0 appears in the data and
    ``log(0)`` is undefined.

    Missing fares are left missing on purpose: imputing them requires a value
    *fitted* on the training split, which belongs to the preprocessor.

    Args:
        fares: Raw ``Fare`` column.

    Returns:
        ``log1p``-transformed fares, preserving NaN.
    """
    numeric = pd.to_numeric(fares, errors="coerce")
    return pd.Series(np.log1p(numeric.to_numpy(dtype=float)), index=fares.index, name="LogFare")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered column to a copy of the raw dataframe.

    This is the single entry point used by training, inference and the EDA
    notebook, so the notebook can never drift from the feature logic the model
    actually consumes.

    Args:
        df: Raw dataframe matching the Kaggle Titanic schema. ``Cabin`` and
            ``Embarked`` may be absent; they are created as all-missing.

    Returns:
        A new dataframe with :data:`ENGINEERED_COLUMNS` added. The input is
        never mutated, because callers reuse the raw frame for display.
    """
    out = df.copy()

    # Materialise optional columns so every downstream step can assume they
    # exist. An absent column and an all-NaN column must behave identically.
    for column in _OPTIONAL_RAW_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out["Title"] = extract_title(out["Name"])
    out["FamilySize"] = family_size(out)
    out["IsAlone"] = is_alone(out["FamilySize"])
    out["Deck"] = deck_from_cabin(out["Cabin"])
    out["LogFare"] = log_fare(out["Fare"])

    return out
