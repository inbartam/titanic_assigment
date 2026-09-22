"""Metrics, bootstrap confidence intervals and curve data.

Deliberately contains **no plotting**. This module produces numbers and
arrays; :mod:`titanic.plots` turns them into figures. That separation is what
lets ``train.py`` write HTML files and the Streamlit app render interactive
charts from one implementation.

Why confidence intervals are not optional here: the held-out validation split
has 179 rows, so an accuracy of 0.83 carries a 95% interval of roughly +/-
0.055. Reporting a point estimate to three decimals would imply a precision
the data cannot support, and would make two models look different when they
are not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from titanic.utils import get_logger

logger = get_logger(__name__)

#: Metrics that get a bootstrap interval. Ordered as they are displayed.
CI_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)

#: Number of bootstrap resamples. 1000 is enough for a stable 95% interval at
#: this sample size and costs milliseconds.
DEFAULT_N_BOOTSTRAP = 1000


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute the full metric set at one decision threshold.

    Accuracy alone is inadequate at a 38% base rate -- a model predicting
    "nobody survived" scores 0.62 while being useless -- so threshold metrics,
    ranking metrics and a calibration metric are all reported together.

    Args:
        y_true: Binary ground truth, shape ``(n,)``.
        y_prob: Predicted ``P(survived)`` in ``[0, 1]``, shape ``(n,)``.
        threshold: Probability above which a passenger is predicted to survive.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc, brier,
        the confusion matrix as a nested list, the threshold and ``n``.

    Raises:
        ValueError: If the inputs have different lengths or are empty.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true has {len(y_true)} rows but y_prob has {len(y_prob)}; they must match."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty array.")

    y_pred = (y_prob >= threshold).astype(int)

    # zero_division=0: when a model predicts no positives at all, precision is
    # undefined. Reporting 0 is the honest reading and avoids a warning that
    # would otherwise appear mid-training.
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(y_pred.mean()),
    }

    # Ranking and calibration metrics need both classes present. A single-class
    # slice is legitimate input (a user uploads ten survivors), so degrade to
    # None rather than raising.
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present; ROC-AUC and PR-AUC are undefined.")
        metrics.update({"roc_auc": None, "pr_auc": None, "brier": None})
    else:
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                # average_precision_score is the correct PR-AUC: it sums
                # rectangles exactly rather than trapezoid-interpolating a
                # curve that is not monotonic.
                "pr_auc": float(average_precision_score(y_true, y_prob)),
                # Brier score = mean squared error of the probabilities. Low
                # values mean well-calibrated, not merely well-ranked.
                "brier": float(brier_score_loss(y_true, y_prob)),
            }
        )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = matrix.tolist()
    # Named cells so the app never has to remember sklearn's row/column order.
    (tn, fp), (fn, tp) = matrix
    metrics["confusion"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Estimate percentile confidence intervals by resampling the evaluation set.

    Each resample draws ``n`` rows with replacement and recomputes every
    metric; the interval is the 2.5th to 97.5th percentile of those values.
    Resampling is **stratified** -- positives and negatives are drawn
    separately, preserving the class balance -- because an unstratified
    resample of 179 rows occasionally produces a single-class sample for which
    ROC-AUC is undefined.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        threshold: Decision threshold for the threshold-dependent metrics.
        n_boot: Number of resamples.
        seed: Seed, so the published intervals are reproducible.
        alpha: Significance level; 0.05 gives a 95% interval.

    Returns:
        Mapping from metric name to ``[lower, upper]``. Metrics that could not
        be computed on enough resamples are omitted.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        logger.warning("Cannot bootstrap with a single class present.")
        return {}

    samples: dict[str, list[float]] = {metric: [] for metric in CI_METRICS}

    for _ in range(n_boot):
        # Draw each class to its original size: the resampled set has the same
        # n and the same class balance as the real evaluation set.
        resampled = np.concatenate(
            [
                rng.choice(positive_idx, size=len(positive_idx), replace=True),
                rng.choice(negative_idx, size=len(negative_idx), replace=True),
            ]
        )
        metrics = compute_metrics(y_true[resampled], y_prob[resampled], threshold)
        for metric in CI_METRICS:
            value = metrics.get(metric)
            if value is not None:
                samples[metric].append(value)

    lower_pct, upper_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        metric: [
            round(float(np.percentile(values, lower_pct)), 6),
            round(float(np.percentile(values, upper_pct)), 6),
        ]
        for metric, values in samples.items()
        if values
    }


def curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Compute the arrays behind every evaluation figure.

    Kept separate from plotting so the same numbers feed the HTML files
    ``train.py`` writes and the interactive charts the app renders.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.

    Returns:
        Dict with ``roc``, ``pr``, ``threshold_sweep``, ``calibration`` and
        ``probabilities``. Returns empty curve entries when only one class is
        present, which the plotting layer renders as an explanatory message.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    data: dict[str, Any] = {
        "probabilities": y_prob.tolist(),
        "y_true": y_true.tolist(),
    }

    if len(np.unique(y_true)) < 2:
        data.update({"roc": {}, "pr": {}, "threshold_sweep": {}, "calibration": {}})
        return data

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    data["roc"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": np.where(np.isinf(roc_thresholds), 1.0, roc_thresholds).tolist(),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    data["pr"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        # precision_recall_curve returns one fewer threshold than points; pad
        # so every array in the dict has a consistent length for plotting.
        "thresholds": [*pr_thresholds.tolist(), 1.0],
        "auc": float(average_precision_score(y_true, y_prob)),
        "baseline": float(y_true.mean()),
    }

    # Threshold sweep: how precision, recall and F1 trade off as the decision
    # boundary moves. This is what the app's threshold slider visualises.
    sweep_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep: dict[str, list[float]] = {
        "thresholds": sweep_thresholds.tolist(),
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for threshold in sweep_thresholds:
        point = compute_metrics(y_true, y_prob, float(threshold))
        for metric in ("accuracy", "precision", "recall", "f1"):
            sweep[metric].append(round(point[metric], 6))
    data["threshold_sweep"] = sweep

    data["calibration"] = _calibration_data(y_true, y_prob)
    return data


def _calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Bin predictions and compare predicted against observed frequency.

    A perfectly calibrated model puts 30% of the passengers it scores at 0.3
    into the survived class. Deviation from the diagonal shows over- or
    under-confidence, which ROC-AUC cannot reveal because it only measures
    ranking.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        Dict of per-bin mean predicted probability, observed fraction and
        count. Empty bins are dropped rather than plotted as zero.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # -1 because np.digitize returns 1-based bin numbers; clip folds a
    # probability of exactly 1.0 into the last bin rather than a new one.
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    mean_predicted: list[float] = []
    observed: list[float] = []
    counts: list[float] = []

    for index in range(n_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    return {"mean_predicted": mean_predicted, "observed": observed, "counts": counts}


def summarise(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute metrics, intervals and curves in one call.

    Convenience wrapper used by ``train.py`` and the inference service so the
    three stay consistent.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        n_boot: Bootstrap resamples; 0 skips the intervals.
        seed: Bootstrap seed.

    Returns:
        ``{"metrics": ..., "ci95": ..., "curves": ...}``.
    """
    return {
        "metrics": compute_metrics(y_true, y_prob, threshold),
        "ci95": (
            bootstrap_ci(y_true, y_prob, threshold=threshold, n_boot=n_boot, seed=seed)
            if n_boot
            else {}
        ),
        "curves": curve_data(y_true, y_prob),
    }
