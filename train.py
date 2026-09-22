"""Train the Titanic model ladder and write everything needed for inference.

Usage::

    python train.py --model all
    python train.py --model deep --data-path data/sample_train.csv --no-cv

The script is the whole pipeline in one place: load, validate, split, engineer,
fit the preprocessor on the training split, select hyperparameters by
cross-validation *inside* that split, train, score the held-out split exactly
once, and write bundles to ``artifacts/``.

Two invariants it exists to guarantee:

1. The held-out validation split is used **once**, at the end, for reporting.
   Every selection decision is made by cross-validation on the training split.
2. Everything inference needs is written to disk. Nothing in the app or the
   API ever requires the training data again.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from titanic import plots
from titanic.artifacts import load_registry, save_bundle, update_registry
from titanic.config import TARGET_COLUMN, Paths, SplitConfig, TrainConfig
from titanic.data import KaggleAuthError, SchemaError, load_csv, stratified_split, validate_schema
from titanic.evaluation import bootstrap_ci, compute_metrics, curve_data
from titanic.features import engineer
from titanic.models import build_model, count_parameters
from titanic.preprocessing import Preprocessor
from titanic.sklearn_models import (
    build_gbdt,
    count_gbdt_parameters,
    select_gbdt_config,
    stack_features,
)
from titanic.training import cross_validate, predict_proba_torch, select_config, train_torch_model
from titanic.utils import get_logger, set_seed, timer

logger = get_logger("train")

#: Model names in the order they are trained. Cheapest and most robust first,
#: so an interrupted run still leaves something usable in artifacts/.
MODEL_ORDER: tuple[str, ...] = ("fast", "deep", "gbdt", "attn")

#: Hyperparameter grid for `deep`, searched by 5-fold CV on the training split.
#: Eight points on purpose: the CV standard deviation at n=712 is around 0.03
#: ROC-AUC, which is larger than most differences a bigger grid would find, so
#: a wider search would mostly be fitting noise.
DEEP_GRID: tuple[dict[str, Any], ...] = tuple(
    {"type": "deep", "hidden": hidden, "dropout": dropout, "weight_decay": weight_decay}
    for hidden in ([32, 16], [64, 32])
    for dropout in (0.2, 0.4)
    for weight_decay in (1e-4, 1e-3)
)

#: `fast` and `attn` are trained on a single fixed configuration. `fast` has
#: one obvious setup (it is logistic regression); `attn` is time-boxed and its
#: CV score is reported for context only.
FIXED_CONFIGS: dict[str, dict[str, Any]] = {
    "fast": {"type": "fast"},
    "attn": {
        "type": "attn",
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.2,
    },
}


def library_versions() -> dict[str, str]:
    """Record the library versions that produced an artifact.

    Written into every ``model_config.json`` so a future load can explain a
    behaviour change rather than leaving it a mystery.

    Returns:
        Mapping from library name to version string.
    """
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_figures(
    name: str, curves: dict[str, Any], metrics: dict[str, Any], history: dict[str, Any]
) -> dict[str, Any]:
    """Build every figure saved alongside a bundle.

    The same functions produce the app's interactive charts, so the HTML files
    and the app can never disagree.

    Args:
        name: Model name, used for the consistent per-model colour.
        curves: Output of :func:`titanic.evaluation.curve_data`.
        metrics: Output of :func:`titanic.evaluation.compute_metrics`.
        history: Training history; empty for sklearn models.

    Returns:
        Mapping from filename stem to Plotly figure.
    """
    figures = {
        "roc": plots.roc_fig({name: curves}),
        "pr": plots.pr_fig({name: curves}),
        "confusion_matrix": plots.confusion_matrix_fig(metrics["confusion_matrix"]),
        "threshold_sweep": plots.threshold_sweep_fig(curves.get("threshold_sweep", {})),
        "calibration": plots.calibration_fig(curves.get("calibration", {})),
        "probability_histogram": plots.prob_histogram_fig(
            curves.get("probabilities", []), curves.get("y_true")
        ),
    }
    if history.get("epochs"):
        figures["training_curves"] = plots.training_curves_fig(history)
    return figures


def evaluate_and_save(
    name: str,
    model: Any,
    framework: str,
    model_config: dict[str, Any],
    preprocessor: Preprocessor,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    n_train: int,
    history: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Score a trained model on the held-out split and write its bundle.

    This is the **only** place the validation split is used, and it is used
    once per model.

    Args:
        name: Model name.
        model: The trained model.
        framework: ``"torch"`` or ``"sklearn"``.
        model_config: Architecture config to serialise.
        preprocessor: The preprocessor fitted on the training split.
        val_df: Engineered held-out split.
        y_val: Held-out labels.
        n_train: Training split size, recorded in metrics.
        history: Training history and CV results.
        args: Parsed CLI arguments.

    Returns:
        The metrics dict that was written to ``metrics.json``.
    """
    x_num_val, x_cat_val = preprocessor.transform(val_df)

    # Time inference on the real validation batch, reported per 1000 rows so
    # the app can compare models on a common scale.
    with timer() as inference_time:
        if framework == "sklearn":
            probabilities = model.predict_proba(stack_features(x_num_val, x_cat_val))[:, 1]
        else:
            probabilities = predict_proba_torch(model, x_num_val, x_cat_val)

    metrics = compute_metrics(y_val, probabilities, args.threshold)
    intervals = bootstrap_ci(
        y_val, probabilities, threshold=args.threshold, n_boot=args.n_boot, seed=args.seed
    )
    curves = curve_data(y_val, probabilities)

    n_params = count_gbdt_parameters(model) if framework == "sklearn" else count_parameters(model)
    payload = {
        "model": name,
        "framework": framework,
        "seed": args.seed,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": n_train,
        "n_val": len(val_df),
        "n_params": n_params,
        "threshold": args.threshold,
        "validation": metrics,
        "validation_ci95": intervals,
        "cv_train_split": history.get("cv", {}),
        "inference_ms_per_1k_rows": round(inference_time["ms"] * 1000 / max(len(val_df), 1), 4),
    }

    full_config = {
        **model_config,
        "framework": framework,
        "n_params": n_params,
        "seed": args.seed,
        "versions": library_versions(),
    }
    if framework == "sklearn":
        import sklearn

        full_config["sklearn_version"] = sklearn.__version__

    save_bundle(
        directory=Path(args.artifacts_dir) / name,
        name=name,
        model=model,
        model_config=full_config,
        preprocessor=preprocessor,
        metrics=payload,
        history=history,
        figures=build_figures(name, curves, metrics, history),
    )

    update_registry(
        Path(args.artifacts_dir) / "registry.json",
        name,
        {
            # Relative to the registry file itself, so a registry written
            # with --artifacts-dir still resolves after the tree is moved.
            "dir": name,
            "framework": framework,
            "trained_at": payload["trained_at"],
            "roc_auc": metrics.get("roc_auc"),
            "accuracy": metrics.get("accuracy"),
            "n_params": n_params,
        },
    )
    return payload


def train_torch(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: Preprocessor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train one torch model end to end.

    Args:
        name: ``fast``, ``deep`` or ``attn``.
        train_df: Engineered training split.
        val_df: Engineered held-out split.
        preprocessor: Preprocessor already fitted on ``train_df``.
        args: Parsed CLI arguments.

    Returns:
        The model's metrics payload.
    """
    import torch

    train_config = TrainConfig(seed=args.seed)
    if args.epochs:
        train_config = TrainConfig(seed=args.seed, max_epochs=args.epochs)

    history: dict[str, Any] = {}

    # --- configuration selection (training split only) -----------------
    if name == "deep" and args.cv:
        logger.info(
            "[%s] selecting hyperparameters over %d configs x %d folds",
            name,
            len(DEEP_GRID),
            args.cv_folds,
        )
        config, grid_results = select_config(
            list(DEEP_GRID), train_df, train_config, k=args.cv_folds, seed=args.seed
        )
        history["cv_grid"] = grid_results
        history["cv"] = next(
            {"roc_auc_mean": r["roc_auc_mean"], "roc_auc_std": r["roc_auc_std"]}
            for r in grid_results
            if r["config"] == config
        )
        # weight_decay is a training setting, not an architecture one, so it
        # travels in TrainConfig rather than into build_model.
        train_config = TrainConfig(
            seed=args.seed,
            max_epochs=train_config.max_epochs,
            weight_decay=config.get("weight_decay", train_config.weight_decay),
        )
    else:
        config = FIXED_CONFIGS.get(name, {"type": name})
        if name == "deep":
            config = {"type": "deep", "hidden": [64, 32], "dropout": 0.3}
        if args.cv:
            # Reported for context: `fast` and `attn` are not selected on it.
            logger.info("[%s] cross-validating the fixed configuration", name)
            history["cv"] = cross_validate(
                config, train_df, train_config, k=args.cv_folds, seed=args.seed
            )

    # --- final fit on the whole training split -------------------------
    x_num, x_cat = preprocessor.transform(train_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()

    # Re-seed immediately before construction so weight initialisation does
    # not depend on how much randomness the CV above happened to consume.
    torch.manual_seed(args.seed)
    model = build_model(config, x_num.shape[1], preprocessor.cardinalities)
    logger.info("[%s] %s | %d parameters", name, config, count_parameters(model))

    with timer() as train_time:
        run_history = train_torch_model(model, x_num, x_cat, y_train, train_config)
    history.update(run_history)
    history["train_seconds"] = round(train_time["ms"] / 1000, 2)
    history["config"] = config
    logger.info(
        "[%s] trained in %.1fs, %d epochs, best epoch %d",
        name,
        history["train_seconds"],
        history["n_epochs_run"],
        history["best_epoch"],
    )

    return evaluate_and_save(
        name=name,
        model=model,
        framework="torch",
        model_config=config,
        preprocessor=preprocessor,
        val_df=val_df,
        y_val=val_df[TARGET_COLUMN].to_numpy(),
        n_train=len(train_df),
        history=history,
        args=args,
    )


def train_gbdt(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: Preprocessor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train the gradient-boosting reference model.

    Args:
        train_df: Engineered training split.
        val_df: Engineered held-out split.
        preprocessor: Preprocessor already fitted on ``train_df``.
        args: Parsed CLI arguments.

    Returns:
        The model's metrics payload.
    """
    history: dict[str, Any] = {}

    if args.cv:
        logger.info("[gbdt] selecting hyperparameters by %d-fold CV", args.cv_folds)
        config, grid_results = select_gbdt_config(train_df, k=args.cv_folds, seed=args.seed)
        history["cv_grid"] = grid_results
        history["cv"] = next(
            {"roc_auc_mean": r["roc_auc_mean"], "roc_auc_std": r["roc_auc_std"]}
            for r in grid_results
            if r["config"] == config
        )
    else:
        config = {"max_depth": 3, "learning_rate": 0.1}

    x_num, x_cat = preprocessor.transform(train_df)
    model = build_gbdt(
        config, len(preprocessor.numeric_cols), len(preprocessor.categorical_cols), args.seed
    )

    with timer() as train_time:
        model.fit(stack_features(x_num, x_cat), train_df[TARGET_COLUMN].to_numpy())
    history["train_seconds"] = round(train_time["ms"] / 1000, 2)
    history["config"] = config
    logger.info(
        "[gbdt] trained in %.1fs, %d boosting iterations",
        history["train_seconds"],
        model.n_iter_,
    )

    return evaluate_and_save(
        name="gbdt",
        model=model,
        framework="sklearn",
        model_config={"type": "gbdt", **config, "n_iter": int(model.n_iter_)},
        preprocessor=preprocessor,
        val_df=val_df,
        y_val=val_df[TARGET_COLUMN].to_numpy(),
        n_train=len(train_df),
        history=history,
        args=args,
    )


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    """Log the final comparison table with confidence intervals.

    Args:
        results: Model name to its metrics payload.
    """
    if not results:
        return

    header = f"{'model':<8}{'params':>9}{'accuracy':>22}{'roc_auc':>22}{'f1':>10}{'ms/1k':>9}"
    logger.info("\n%s\n%s", header, "-" * len(header))

    for name, payload in results.items():
        validation = payload["validation"]
        intervals = payload.get("validation_ci95", {})

        def formatted(metric: str, v=validation, c=intervals) -> str:
            """Render a metric with its interval, or a dash when undefined."""
            value = v.get(metric)
            if value is None:
                return f"{'n/a':>22}"
            bounds = c.get(metric)
            if not bounds:
                return f"{value:>22.3f}"
            return f"{value:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]".rjust(22)

        # Thousands separators are an f-string feature, not a printf one, so
        # the count is formatted before it reaches the logging call.
        logger.info(
            "%-8s%9s%s%s%10.3f%9.2f",
            name,
            f"{payload['n_params']:,}",
            formatted("accuracy"),
            formatted("roc_auc"),
            validation.get("f1", float("nan")),
            payload["inference_ms_per_1k_rows"],
        )

    logger.info(
        "\nValidation n=%d. Overlapping intervals mean the models are not "
        "distinguishable on this data.",
        next(iter(results.values()))["n_val"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    paths = Paths()
    parser = argparse.ArgumentParser(
        prog="python train.py",
        description="Train the Titanic model ladder and write artifacts to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=[*MODEL_ORDER, "all"],
        help="which model to train",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="CSV to train on (default: data/train.csv, else the bundled sample)",
    )
    parser.add_argument("--artifacts-dir", default=str(paths.artifacts), help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="validation fraction")
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    parser.add_argument("--cv-folds", type=int, default=5, help="cross-validation folds")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples (0 to skip)")
    parser.add_argument(
        "--cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the hyperparameter grids (--no-cv for a fast smoke run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the training pipeline.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled failure.
    """
    args = parse_args(argv)
    paths = Paths()

    # Seed before anything random happens, including the split.
    set_seed(args.seed)

    # --- load -----------------------------------------------------------
    if args.data_path:
        data_path = Path(args.data_path)
    elif paths.train_csv.is_file():
        data_path = paths.train_csv
    else:
        data_path = paths.sample_csv
        logger.warning(
            "data/train.csv not found; falling back to the 100-row sample. "
            "Run 'python -m titanic.data --fetch' for real results."
        )

    try:
        raw = load_csv(data_path)
        validate_schema(raw, require_target=True)
    except (SchemaError, KaggleAuthError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    # --- split, engineer, fit the preprocessor ---------------------------
    train_raw, val_raw = stratified_split(
        raw, SplitConfig(test_size=args.test_size, seed=args.seed)
    )
    train_df, val_df = engineer(train_raw), engineer(val_raw)

    # Fitted once, on the training split, and reused for every model so the
    # comparison is not confounded by different preprocessing.
    preprocessor = Preprocessor().fit(train_df)

    selected = MODEL_ORDER if args.model == "all" else (args.model,)
    logger.info(
        "Training %s on %d rows (validation: %d rows, scored once at the end)",
        ", ".join(selected),
        len(train_df),
        len(val_df),
    )

    # --- train ------------------------------------------------------------
    results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for name in selected:
        logger.info("=" * 70)
        try:
            if name == "gbdt":
                results[name] = train_gbdt(train_df, val_df, preprocessor, args)
            else:
                results[name] = train_torch(name, train_df, val_df, preprocessor, args)
        except Exception:
            # One model failing must not discard the models already trained:
            # the registry tolerates a partial set and the app renders it.
            logger.exception("[%s] training failed; continuing with the remaining models", name)

    # Prefer `deep` as the app's default: it is the assignment's required
    # neural network. Fall back to whatever did train if it is absent.
    preferred = "deep" if "deep" in results else next(iter(results), None)
    if preferred:
        update_registry(
            Path(args.artifacts_dir) / "registry.json",
            preferred,
            load_registry(Path(args.artifacts_dir) / "registry.json")["models"][preferred],
            default=preferred,
        )

    logger.info("=" * 70)
    print_summary(results)
    logger.info("Total wall time: %.1fs", time.perf_counter() - started)

    if not results:
        logger.error("No model trained successfully.")
        return 1

    summary_path = Path(args.artifacts_dir) / "last_run.json"
    summary_path.write_text(
        json.dumps(
            {
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "data_path": str(data_path),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "models": list(results),
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Artifacts written to %s", Path(args.artifacts_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
