"""Dataset acquisition, schema validation and splitting.

This module is the only place that talks to Kaggle and the only place that
decides whether a dataframe is acceptable. Both the training CLI and the
inference service reuse :func:`validate_schema`, so a CSV that the app accepts
is exactly a CSV that training would have accepted.

Run as a script to download the dataset::

    python -m titanic.data --fetch
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from titanic.config import (
    NUMERIC_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    Paths,
    SplitConfig,
)
from titanic.utils import get_logger

logger = get_logger(__name__)

#: Kaggle competition slug. Only ``train.csv`` is ever requested from it --
#: the assignment forbids using test.csv and gender_submission.csv.
COMPETITION = "titanic"
TRAIN_FILE = "train.csv"


class SchemaError(ValueError):
    """Raised when a dataframe does not match the raw Kaggle Titanic schema.

    Inherits from :class:`ValueError` so callers that catch broad value errors
    still behave sensibly, while the API can map this specific type to HTTP 422
    and the Streamlit app can render it as an actionable message. Every message
    states *what* is wrong and *how to fix it*.
    """


class KaggleAuthError(RuntimeError):
    """Raised when Kaggle credentials are missing, malformed or rejected.

    Separate from :class:`SchemaError` because the remedy is completely
    different: the user must create a token, not fix a file.
    """


# --------------------------------------------------------------------------
# Kaggle fetching
# --------------------------------------------------------------------------


def _load_kaggle_credentials() -> str:
    """Locate Kaggle credentials and expose them to the ``kaggle`` client.

    Kaggle currently supports two credential formats and the client only picks
    some of them up on its own, so we normalise here:

    1. ``KAGGLE_API_TOKEN`` environment variable (newer ``KGAT_...`` token).
    2. ``~/.kaggle/access_token`` file containing the same token. The client
       does *not* read this file itself, so we load it and set the env var.
    3. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables (classic).
    4. ``~/.kaggle/kaggle.json`` (classic), which the client reads natively.

    Returns:
        A short string naming which credential source was used, for logging.

    Raises:
        KaggleAuthError: If none of the four sources is present, with the exact
            steps needed to fix it.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN environment variable"

    access_token_file = Path.home() / ".kaggle" / "access_token"
    if access_token_file.is_file():
        token = access_token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise KaggleAuthError(
                f"{access_token_file} exists but is empty. Paste your Kaggle API token "
                "(it starts with KGAT_) into it, or set KAGGLE_API_TOKEN instead."
            )
        # The kaggle client reads the env var but not this file, so bridge it.
        os.environ["KAGGLE_API_TOKEN"] = token
        return str(access_token_file)

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY environment variables"

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        return str(kaggle_json)

    raise KaggleAuthError(
        "No Kaggle credentials found. Fix with either option:\n"
        "  (a) Create a token at https://www.kaggle.com/settings -> Create New Token\n"
        f"      and save it to {Path.home() / '.kaggle' / 'kaggle.json'}, or save the\n"
        f"      newer KGAT_ token to {Path.home() / '.kaggle' / 'access_token'}.\n"
        "  (b) Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY) in your shell.\n"
        "Then accept the competition rules once at\n"
        "https://www.kaggle.com/competitions/titanic/rules -- downloads 403 otherwise.\n"
        "No credentials? Every command accepts --data-path data/sample_train.csv instead."
    )


def fetch_from_kaggle(dest_dir: Path | None = None, *, force: bool = False) -> Path:
    """Download ``train.csv`` from the Kaggle Titanic competition.

    Only ``train.csv`` is requested: the assignment forbids ``test.csv`` and
    ``gender_submission.csv``, and downloading a single file is also faster and
    smaller than pulling the competition zip.

    Args:
        dest_dir: Directory to write into. Defaults to ``<repo>/data``.
        force: Re-download even if the file already exists. Without it, an
            existing file short-circuits the network call so repeated runs are
            instant and work offline.

    Returns:
        Path to the downloaded ``train.csv``.

    Raises:
        KaggleAuthError: If credentials are missing, or the API rejects them --
            most often because the competition rules have not been accepted.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else Paths().data
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / TRAIN_FILE

    if target.is_file() and not force:
        logger.info("Using existing %s (pass force=True to re-download)", target)
        return target

    source = _load_kaggle_credentials()
    logger.info("Authenticating with Kaggle using %s", source)

    # Imported lazily: the kaggle package authenticates at import time in some
    # versions, which would make merely importing this module fail without
    # credentials -- even for users who only ever pass --data-path.
    import kaggle

    try:
        kaggle.api.authenticate()
        kaggle.api.competition_download_file(
            COMPETITION, TRAIN_FILE, path=str(dest_dir), force=True
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, actionable error
        raise KaggleAuthError(
            f"Kaggle download failed ({type(exc).__name__}: {exc}).\n"
            "Most common cause: the competition rules have not been accepted. Visit\n"
            "https://www.kaggle.com/competitions/titanic/rules and accept them, then retry.\n"
            "Otherwise verify your token is current, or use --data-path data/sample_train.csv."
        ) from exc

    if not target.is_file():
        raise KaggleAuthError(
            f"Kaggle reported success but {target} is missing. Check write permissions "
            f"on {dest_dir}."
        )

    logger.info("Downloaded %s (%d bytes)", target, target.stat().st_size)
    return target


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV into a dataframe with clear errors for the common failures.

    Args:
        path: Path to a ``.csv`` file.

    Returns:
        The parsed dataframe.

    Raises:
        FileNotFoundError: If the path does not exist, naming the resolved path
            so the user can see where we actually looked.
        SchemaError: If the file is empty or cannot be parsed as CSV.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {path.resolve()}. Provide a valid path, or run "
            "'python -m titanic.data --fetch' to download the Kaggle training set."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(f"{path} is empty. Expected the raw Kaggle Titanic CSV.") from exc
    except pd.errors.ParserError as exc:
        raise SchemaError(
            f"{path} could not be parsed as CSV ({exc}). Check the delimiter and quoting."
        ) from exc

    if df.empty:
        raise SchemaError(f"{path} contains a header but no data rows.")

    logger.info("Loaded %s: %d rows x %d columns", path.name, len(df), df.shape[1])
    return df


def validate_schema(df: pd.DataFrame, *, require_target: bool = False) -> None:
    """Check a dataframe against the raw Kaggle Titanic schema.

    Validation is intentionally permissive about *extra* columns (they are
    logged and ignored) and strict about missing required ones, because a user
    exporting a CSV from a spreadsheet will routinely carry extra columns along
    but must never be allowed to run a model on the wrong features.

    Args:
        df: The dataframe to check.
        require_target: When ``True``, ``Survived`` must be present. Training
            and the ``/evaluate`` endpoint set this; plain inference does not.

    Raises:
        SchemaError: If required columns are missing, a numeric column contains
            non-numeric values, or ``Survived`` holds values outside {0, 1}.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Missing required columns: {missing}. Expected the raw Kaggle Titanic schema "
            f"({', '.join(REQUIRED_COLUMNS)} required; "
            f"{', '.join(OPTIONAL_COLUMNS)} optional). See README."
        )

    if require_target and TARGET_COLUMN not in df.columns:
        raise SchemaError(
            f"Column '{TARGET_COLUMN}' is required here but missing. Evaluation needs ground "
            "truth labels; use a labelled CSV, or run inference-only instead."
        )

    # Numeric checks: coerce a copy and look for values that became NaN without
    # having been NaN to begin with. That distinguishes "missing" (legitimate --
    # Age has 177 blanks) from "not a number" (a typo such as 'twenty').
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        became_nan = coerced.isna() & df[col].notna()
        if became_nan.any():
            bad_rows = df.index[became_nan][:3].tolist()
            examples = df.loc[became_nan, col].head(3).tolist()
            raise SchemaError(
                f"Column '{col}' must be numeric but contains non-numeric values "
                f"{examples} at rows {bad_rows}. Fix those cells or leave them blank."
            )

    if TARGET_COLUMN in df.columns:
        # dropna() first: an unlabelled row is handled downstream, but a label
        # of 2 or 'yes' is a data error we must reject loudly.
        labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().unique()
        invalid = sorted({v for v in labels if v not in (0, 1)})
        if invalid:
            raise SchemaError(
                f"Column '{TARGET_COLUMN}' must contain only 0 or 1, found {invalid}. "
                "1 = survived, 0 = did not survive."
            )

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if extra:
        logger.warning("Ignoring unrecognised columns: %s", extra)


# --------------------------------------------------------------------------
# Splitting and sampling
# --------------------------------------------------------------------------


def stratified_split(
    df: pd.DataFrame, config: SplitConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labelled dataframe into training and validation halves.

    The validation half is scored exactly once, at the end of training. All
    model selection happens with cross-validation *inside* the training half,
    which is what keeps the reported numbers unbiased.

    Args:
        df: Labelled dataframe containing :data:`~titanic.config.TARGET_COLUMN`.
        config: Split parameters. Defaults to 80/20, seed 42, stratified.

    Returns:
        ``(train_df, val_df)``, both with a reset index so downstream array
        positions line up with dataframe rows.

    Raises:
        SchemaError: If the target column is absent -- splitting without labels
            would silently produce an unusable training set.
    """
    config = config or SplitConfig()
    if TARGET_COLUMN not in df.columns:
        raise SchemaError(
            f"Cannot split without the '{TARGET_COLUMN}' column: the training set must be "
            "labelled. Use the Kaggle train.csv, not test.csv."
        )

    from sklearn.model_selection import train_test_split

    # Stratifying on the label keeps the 38% survival rate identical in both
    # halves, removing a noise source worth 1-2 accuracy points at n=179.
    stratify_on = df[TARGET_COLUMN] if config.stratify else None
    train_df, val_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=stratify_on,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info(
        "Split: train=%d (%.1f%% positive), val=%d (%.1f%% positive)",
        len(train_df),
        100 * train_df[TARGET_COLUMN].mean(),
        len(val_df),
        100 * val_df[TARGET_COLUMN].mean(),
    )
    return train_df, val_df


def make_sample(df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Draw a small stratified sample, used to create the committed demo CSV.

    The sample is stratified on the label so it stays representative, and
    sorted by ``PassengerId`` so the committed file has a stable, reviewable
    diff rather than a random row order.

    Args:
        df: The full labelled dataframe.
        n: Approximate number of rows to keep.
        seed: Sampling seed.

    Returns:
        A sampled copy of ``df``.
    """
    from sklearn.model_selection import train_test_split

    if n >= len(df):
        return df.copy()

    stratify_on = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    sample, _ = train_test_split(
        df, train_size=n, random_state=seed, stratify=stratify_on, shuffle=True
    )
    sort_col = "PassengerId" if "PassengerId" in sample.columns else sample.columns[0]
    return sample.sort_values(sort_col).reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for dataset management.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled, explained failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m titanic.data",
        description="Download the Kaggle Titanic training set and build the demo sample.",
    )
    parser.add_argument("--fetch", action="store_true", help="download train.csv from Kaggle")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--make-sample",
        action="store_true",
        help="regenerate data/sample_train.csv from data/train.csv",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="rows in the sample")
    args = parser.parse_args(argv)

    if not args.fetch and not args.make_sample:
        parser.print_help()
        return 0

    paths = Paths()
    try:
        if args.fetch:
            path = fetch_from_kaggle(paths.data, force=args.force)
            df = load_csv(path)
            validate_schema(df, require_target=True)
            logger.info("train.csv is valid: %d rows, %d columns", len(df), df.shape[1])

        if args.make_sample:
            df = load_csv(paths.train_csv)
            validate_schema(df, require_target=True)
            sample = make_sample(df, n=args.sample_size)
            sample.to_csv(paths.sample_csv, index=False)
            logger.info("Wrote %s (%d rows)", paths.sample_csv, len(sample))
    except (KaggleAuthError, SchemaError, FileNotFoundError) as exc:
        # Handled failures print their guidance, not a traceback: the message
        # already tells the user exactly what to do next.
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
