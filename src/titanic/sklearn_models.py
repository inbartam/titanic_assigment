"""The gradient-boosting reference model.

``HistGradientBoostingClassifier`` is included because it is the honest
strongest classical baseline on small tabular data, and because the app should
let a reviewer *see* that rather than read a claim about it.

It consumes exactly the same ``(X_num, X_cat)`` the torch models consume: the
categorical columns are passed through unchanged with a boolean mask telling
sklearn to treat them as categories, so the ``<UNK>`` index is simply another
category. There is no second preprocessing path anywhere in the project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from titanic.config import TARGET_COLUMN
from titanic.preprocessing import Preprocessor
from titanic.utils import get_logger

logger = get_logger(__name__)

#: Candidate configurations explored by cross-validation. Four points, for the
#: same reason the torch grid is eight: at n=712 the CV standard deviation
#: exceeds most between-configuration differences.
GBDT_GRID: tuple[dict[str, Any], ...] = (
    {"max_depth": 3, "learning_rate": 0.05},
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": None, "learning_rate": 0.05},
    {"max_depth": None, "learning_rate": 0.1},
)


def categorical_mask(n_numeric: int, n_categorical: int) -> list[bool]:
    """Build the boolean mask marking which columns are categorical.

    The feature matrix is ``np.hstack([X_num, X_cat])``, so the numeric columns
    come first and the categorical ones follow.

    Args:
        n_numeric: Number of numeric columns.
        n_categorical: Number of categorical columns.

    Returns:
        A list of booleans, one per column of the stacked matrix.
    """
    return [False] * n_numeric + [True] * n_categorical


def stack_features(x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
    """Combine the preprocessor's two arrays into one matrix for sklearn.

    Args:
        x_num: Float array ``(n, n_numeric)``.
        x_cat: Int array ``(n, n_categorical)``.

    Returns:
        Float array ``(n, n_numeric + n_categorical)``.
    """
    # float64 throughout: sklearn casts internally anyway, and keeping the
    # categorical indices as exact integers inside a float array is safe at
    # these cardinalities.
    return np.hstack([np.asarray(x_num, dtype=float), np.asarray(x_cat, dtype=float)])


def build_gbdt(
    config: dict[str, Any], n_numeric: int, n_categorical: int, seed: int = 42
) -> HistGradientBoostingClassifier:
    """Construct the gradient-boosting classifier from a configuration.

    Args:
        config: May contain ``max_depth``, ``learning_rate``, ``max_iter``.
        n_numeric: Number of numeric columns, for the categorical mask.
        n_categorical: Number of categorical columns.
        seed: Random state, for reproducibility.

    Returns:
        An unfitted classifier.
    """
    return HistGradientBoostingClassifier(
        max_depth=config.get("max_depth", 3),
        learning_rate=config.get("learning_rate", 0.1),
        max_iter=config.get("max_iter", 300),
        # HGB carves out its own internal validation set for early stopping,
        # which is the sklearn equivalent of the torch models' 10% carve-out.
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        categorical_features=categorical_mask(n_numeric, n_categorical),
        random_state=seed,
    )


def cross_validate_gbdt(
    config: dict[str, Any], df, *, k: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Score a GBDT configuration with leak-free stratified k-fold CV.

    As in :func:`titanic.training.cross_validate`, the preprocessor is refitted
    inside every fold.

    Args:
        config: Candidate hyperparameters.
        df: Engineered training split including the target column.
        k: Number of folds.
        seed: Seed for the fold split and the model.

    Returns:
        ``{"k", "fold_roc_auc", "roc_auc_mean", "roc_auc_std", "log_loss_mean"}``.
    """
    y = df[TARGET_COLUMN].to_numpy()
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    aucs: list[float] = []
    losses: list[float] = []

    for fit_idx, score_idx in folds.split(df, y):
        fold_fit, fold_score = df.iloc[fit_idx], df.iloc[score_idx]

        preprocessor = Preprocessor().fit(fold_fit)
        x_fit = stack_features(*preprocessor.transform(fold_fit))
        x_score = stack_features(*preprocessor.transform(fold_score))

        model = build_gbdt(
            config, len(preprocessor.numeric_cols), len(preprocessor.categorical_cols), seed
        )
        model.fit(x_fit, y[fit_idx])
        probabilities = model.predict_proba(x_score)[:, 1]

        aucs.append(float(roc_auc_score(y[score_idx], probabilities)))
        losses.append(float(log_loss(y[score_idx], probabilities, labels=[0, 1])))

    return {
        "k": k,
        "fold_roc_auc": [round(score, 6) for score in aucs],
        "roc_auc_mean": round(float(np.mean(aucs)), 6),
        "roc_auc_std": round(float(np.std(aucs)), 6),
        "log_loss_mean": round(float(np.mean(losses)), 6),
    }


def select_gbdt_config(
    df, *, grid: tuple[dict[str, Any], ...] = GBDT_GRID, k: int = 5, seed: int = 42
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the best GBDT configuration by cross-validated ROC-AUC.

    Args:
        df: Engineered training split including the target.
        grid: Candidate configurations.
        k: Folds per candidate.
        seed: Shared seed, so candidates are compared on identical folds.

    Returns:
        ``(winning_config, all_results)``.
    """
    results: list[dict[str, Any]] = []
    for position, candidate in enumerate(grid, start=1):
        scores = cross_validate_gbdt(candidate, df, k=k, seed=seed)
        logger.info(
            "cv %d/%d %s -> roc_auc %.4f +/- %.4f",
            position,
            len(grid),
            candidate,
            scores["roc_auc_mean"],
            scores["roc_auc_std"],
        )
        results.append({"config": candidate, **scores})

    best = max(results, key=lambda r: (r["roc_auc_mean"], -r["log_loss_mean"]))
    logger.info("selected %s (cv roc_auc %.4f)", best["config"], best["roc_auc_mean"])
    return best["config"], results


def count_gbdt_parameters(model: HistGradientBoostingClassifier) -> int:
    """Approximate a fitted GBDT's size as its total number of tree nodes.

    There is no parameter count for a tree ensemble in the sense a neural
    network has one; total node count across all boosting iterations is the
    closest comparable measure of model size, and it is what the app displays
    beside the torch models' parameter counts.

    Args:
        model: A fitted classifier.

    Returns:
        Total node count, or 0 if the model is not fitted.
    """
    predictors = getattr(model, "_predictors", None)
    if not predictors:
        return 0
    return sum(tree.get_n_leaf_nodes() * 2 - 1 for iteration in predictors for tree in iteration)
