"""Tests for metrics, bootstrap intervals and curve data.

The numbers this module produces are the numbers in the README, so the tests
check them against hand-computable cases rather than against whatever the code
currently returns.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from titanic.evaluation import bootstrap_ci, compute_metrics, curve_data, summarise


@pytest.fixture
def perfect() -> tuple[np.ndarray, np.ndarray]:
    """A perfectly separated problem."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y_true, y_prob


@pytest.fixture
def realistic() -> tuple[np.ndarray, np.ndarray]:
    """A 200-row problem with a 38% base rate and a usable but imperfect signal."""
    rng = np.random.default_rng(0)
    y_true = (rng.random(200) < 0.38).astype(int)
    # Probabilities correlated with the label, then clipped into range.
    y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.18, 200), 0.01, 0.99)
    return y_true, y_prob


class TestComputeMetrics:
    def test_perfect_separation_scores_one(self, perfect) -> None:
        metrics = compute_metrics(*perfect)
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]

    def test_hand_computable_confusion_matrix(self) -> None:
        # 2 true negatives, 1 false positive, 1 false negative, 2 true positives
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.2, 0.8, 0.9])
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["recall"] == pytest.approx(2 / 3)
        assert metrics["confusion"]["true_positive"] == 2

    def test_threshold_shifts_precision_and_recall(self, realistic) -> None:
        y_true, y_prob = realistic
        low = compute_metrics(y_true, y_prob, threshold=0.2)
        high = compute_metrics(y_true, y_prob, threshold=0.8)
        # Lowering the threshold predicts more positives: recall can only rise.
        assert low["recall"] >= high["recall"]
        assert low["positive_rate"] >= high["positive_rate"]

    def test_ranking_metrics_ignore_the_threshold(self, realistic) -> None:
        y_true, y_prob = realistic
        # ROC-AUC and PR-AUC measure ranking, so they must not move with the
        # decision threshold. If they did, the metric would be miscomputed.
        assert compute_metrics(y_true, y_prob, 0.2)["roc_auc"] == pytest.approx(
            compute_metrics(y_true, y_prob, 0.8)["roc_auc"]
        )

    def test_single_class_degrades_instead_of_raising(self) -> None:
        # A user uploading ten survivors is valid input, not an error.
        metrics = compute_metrics(np.ones(10, dtype=int), np.full(10, 0.7))
        assert metrics["roc_auc"] is None
        assert metrics["accuracy"] == 1.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            compute_metrics(np.array([0, 1]), np.array([0.5]))

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_brier_rewards_calibration_not_ranking(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        confident = compute_metrics(y_true, np.array([0.01, 0.02, 0.98, 0.99]))
        hedged = compute_metrics(y_true, np.array([0.45, 0.46, 0.54, 0.55]))
        # Both rank perfectly, so both have AUC 1.0 ...
        assert confident["roc_auc"] == hedged["roc_auc"] == 1.0
        # ... but only Brier notices that one is far better calibrated.
        assert confident["brier"] < hedged["brier"]


class TestBootstrapCi:
    def test_intervals_bracket_the_point_estimate(self, realistic) -> None:
        y_true, y_prob = realistic
        metrics = compute_metrics(y_true, y_prob)
        intervals = bootstrap_ci(y_true, y_prob, n_boot=300, seed=1)
        for metric, (low, high) in intervals.items():
            assert low <= metrics[metric] <= high, f"{metric} outside its own interval"

    def test_intervals_are_ordered(self, realistic) -> None:
        for low, high in bootstrap_ci(*realistic, n_boot=200, seed=1).values():
            assert low <= high

    def test_is_reproducible(self, realistic) -> None:
        # The published intervals must be the same on a reviewer's machine.
        first = bootstrap_ci(*realistic, n_boot=200, seed=7)
        second = bootstrap_ci(*realistic, n_boot=200, seed=7)
        assert first == second

    def test_smaller_samples_give_wider_intervals(self) -> None:
        rng = np.random.default_rng(3)
        big_true = (rng.random(600) < 0.4).astype(int)
        big_prob = np.clip(0.3 + 0.4 * big_true + rng.normal(0, 0.2, 600), 0.01, 0.99)

        wide = bootstrap_ci(big_true[:60], big_prob[:60], n_boot=400, seed=1)["accuracy"]
        narrow = bootstrap_ci(big_true, big_prob, n_boot=400, seed=1)["accuracy"]
        # This is the whole justification for reporting intervals: at n=179
        # the uncertainty is large enough to change conclusions.
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_single_class_returns_empty(self) -> None:
        assert bootstrap_ci(np.ones(10, dtype=int), np.full(10, 0.7), n_boot=10) == {}

    def test_resampling_preserves_class_balance(self) -> None:
        # Stratified resampling is what keeps ROC-AUC defined in every
        # resample; an unstratified draw of a small set sometimes yields one
        # class and would silently drop those samples.
        y_true = np.array([0] * 5 + [1] * 5)
        y_prob = np.linspace(0.1, 0.9, 10)
        intervals = bootstrap_ci(y_true, y_prob, n_boot=200, seed=2)
        assert "roc_auc" in intervals


class TestCurveData:
    def test_provides_every_curve_the_app_draws(self, realistic) -> None:
        curves = curve_data(*realistic)
        for key in ("roc", "pr", "threshold_sweep", "calibration", "probabilities"):
            assert key in curves

    def test_roc_starts_and_ends_at_the_corners(self, realistic) -> None:
        roc = curve_data(*realistic)["roc"]
        assert roc["fpr"][0] == 0.0 and roc["tpr"][0] == 0.0
        assert roc["fpr"][-1] == 1.0 and roc["tpr"][-1] == 1.0

    def test_roc_thresholds_are_finite(self, realistic) -> None:
        # sklearn emits +inf as the first threshold; plotting it would break
        # the hover text, so curve_data replaces it.
        assert np.isfinite(curve_data(*realistic)["roc"]["thresholds"]).all()

    def test_pr_arrays_share_one_length(self, realistic) -> None:
        pr = curve_data(*realistic)["pr"]
        # precision_recall_curve returns one fewer threshold than points, which
        # would misalign a plot unless padded.
        assert len(pr["precision"]) == len(pr["recall"]) == len(pr["thresholds"])

    def test_pr_baseline_is_the_base_rate(self, realistic) -> None:
        y_true, _ = realistic
        assert curve_data(*realistic)["pr"]["baseline"] == pytest.approx(y_true.mean())

    def test_threshold_sweep_covers_the_slider_range(self, realistic) -> None:
        sweep = curve_data(*realistic)["threshold_sweep"]
        assert sweep["thresholds"][0] == pytest.approx(0.05)
        assert sweep["thresholds"][-1] == pytest.approx(0.95)
        assert len(sweep["accuracy"]) == len(sweep["thresholds"])

    def test_recall_is_monotonically_non_increasing_across_thresholds(self, realistic) -> None:
        recall = curve_data(*realistic)["threshold_sweep"]["recall"]
        # Raising the threshold can only ever predict fewer positives.
        assert all(a >= b - 1e-9 for a, b in itertools.pairwise(recall))

    def test_calibration_drops_empty_bins(self, perfect) -> None:
        calibration = curve_data(*perfect)["calibration"]
        assert len(calibration["mean_predicted"]) == len(calibration["observed"])
        assert sum(calibration["counts"]) == 6

    def test_single_class_returns_empty_curves(self) -> None:
        curves = curve_data(np.ones(5, dtype=int), np.full(5, 0.6))
        assert curves["roc"] == {}
        assert curves["probabilities"]


def test_summarise_bundles_everything(realistic) -> None:
    result = summarise(*realistic, n_boot=100, seed=1)
    assert set(result) == {"metrics", "ci95", "curves"}
    assert result["ci95"]


def test_summarise_can_skip_the_bootstrap(realistic) -> None:
    # n_boot=0 is the fast path the service uses for interactive requests.
    assert summarise(*realistic, n_boot=0)["ci95"] == {}
