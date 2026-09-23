"""One function per tab of the Streamlit app.

Splitting the tabs out keeps ``ds_app.py`` down to wiring (sidebar, data
loading, inference, then six calls), which keeps that file readable as it
grows.

Every function takes what it needs as arguments and returns nothing. None of
them loads a model or reads the filesystem directly: inference arrives as a
``PredictionResult`` or ``EvaluationResult``, and artifact metadata through
the cached loaders in :mod:`app.state`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import components as ui
from app.client import Predictor
from app.state import load_history, load_registry_metrics
from titanic import plots
from titanic.service import EvaluationResult, PredictionResult


def render_overview(selected_model: str, models: dict[str, dict[str, Any]]) -> None:
    """Explain what the app does and show the selected model's scorecard.

    Args:
        selected_model: Currently selected model name.
        models: Model metadata from the predictor.
    """
    entry = models.get(selected_model, {})

    st.header("Titanic survival: inference and evaluation")
    st.markdown(f"""
This app runs a trained classifier over any CSV in the raw Kaggle Titanic schema and, when the
file carries `Survived` labels, evaluates it.

**Selected model:** `{selected_model}`, {ui.MODEL_DESCRIPTIONS.get(selected_model, "model")},
{entry.get("n_params", 0):,} parameters.

**Features.** `Pclass`, `Sex`, `Embarked`, `Title` (from `Name`), `Deck` (from `Cabin`; missing
becomes `U`), `IsAlone`, `Age` (imputed by Title median), `log1p(Fare)` and `FamilySize`.
Batch-dependent features such as ticket-group size were left out. They cannot be computed
consistently for a single passenger, so using them would cause train/serve skew.

**How it was trained.**
- Stratified 80/20 split of `train.csv`, seed 42. The held-out 20% is scored exactly once.
- The preprocessor is fitted on the training split only, then serialised to JSON.
- Hyperparameters chosen by 5-fold cross-validation inside the training split.
- Early stopping on a 10% carve-out of the training split, never on the held-out set.
- Every metric carries a 95% bootstrap confidence interval.

See `README.md` for setup and `docs/DECISIONS.md` for the reasoning behind each choice.
""")

    validation = entry.get("validation", {})
    if validation:
        st.subheader(f"`{selected_model}` on the held-out validation split")
        ui.metric_tiles(validation, entry.get("validation_ci95", {}))


def render_data(frame: pd.DataFrame) -> None:
    """Preview the loaded file and report its schema and missingness.

    Args:
        frame: The loaded dataframe.
    """
    st.subheader("Loaded data")
    st.caption(f"{len(frame):,} rows x {frame.shape[1]} columns")

    ui.schema_panel(frame)
    st.dataframe(frame.head(20), use_container_width=True)

    st.subheader("Missing values")
    ui.missingness_chart(frame)


def render_predictions(result: PredictionResult, frame: pd.DataFrame, has_labels: bool) -> None:
    """Show per-row predictions, a probability histogram and a CSV download.

    Args:
        result: The prediction result.
        frame: The source dataframe, for identifying columns.
        has_labels: Whether the file carries ``Survived``.
    """
    st.subheader(f"Predictions from `{result.model}` at threshold {result.threshold:g}")

    columns = st.columns(4)
    columns[0].metric("Rows", f"{result.n:,}")
    columns[1].metric("Predicted survivors", f"{int(result.predictions.sum()):,}")
    columns[2].metric("Predicted rate", f"{result.predictions.mean():.3f}")
    columns[3].metric("Latency", f"{result.latency_ms.get('total', 0):.1f} ms")

    output = result.to_frame(frame)
    st.dataframe(output, use_container_width=True, height=380)
    st.download_button(
        "Download predictions.csv",
        output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.plotly_chart(
        plots.prob_histogram_fig(
            result.probabilities.tolist(),
            frame["Survived"].tolist() if has_labels else None,
            result.threshold,
        ),
        use_container_width=True,
    )


def render_evaluation(evaluation: EvaluationResult | None, has_labels: bool) -> None:
    """Show metrics, intervals and the four evaluation figures.

    Args:
        evaluation: The evaluation result, or ``None`` when unavailable.
        has_labels: Whether the loaded file carries ``Survived``.
    """
    if not has_labels:
        # The assignment requires this path to be graceful rather than a crash.
        st.info(
            "This file has no `Survived` column, so there is nothing to score against. "
            "Predictions are still available in the Predictions tab. Add a `Survived` "
            "column of 0/1 labels to see metrics here.",
            icon="ℹ️",
        )
        return

    if evaluation is None:
        st.warning("Evaluation did not complete; see the Predictions tab.")
        return

    st.subheader(f"`{evaluation.model}` on {evaluation.n:,} labelled rows")
    ui.metric_tiles(evaluation.metrics, evaluation.ci95)
    st.caption(
        "Intervals are 1000 stratified bootstrap resamples of this file. Overlapping "
        "intervals mean a difference is not measurable on this much data."
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plots.confusion_matrix_fig(evaluation.metrics["confusion_matrix"]),
            use_container_width=True,
        )
        st.plotly_chart(
            plots.roc_fig({evaluation.model: evaluation.curves}), use_container_width=True
        )
    with right:
        st.plotly_chart(
            plots.pr_fig({evaluation.model: evaluation.curves}), use_container_width=True
        )
        st.plotly_chart(
            plots.threshold_sweep_fig(
                evaluation.curves.get("threshold_sweep", {}), evaluation.threshold
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        plots.calibration_fig(evaluation.curves.get("calibration", {})),
        use_container_width=True,
    )
    st.caption(
        "Calibration compares predicted probability against observed frequency. A model can "
        "rank perfectly (high ROC-AUC) and still be badly calibrated."
    )


def render_compare(
    predictor: Predictor,
    artifacts_directory: str,
    selected_model: str,
    frame: pd.DataFrame,
    threshold: float,
    has_labels: bool,
) -> None:
    """Compare every trained model side by side.

    Args:
        predictor: Used to score all models on the loaded file.
        artifacts_directory: Where the metrics live.
        selected_model: Model highlighted in the table.
        frame: The loaded dataframe.
        threshold: Current decision threshold.
        has_labels: Whether the loaded file can be scored.
    """
    metrics_by_model = load_registry_metrics(artifacts_directory)
    if not metrics_by_model:
        st.info("No metrics found. Run `python train.py --model all` first.")
        return

    st.subheader("Held-out validation results")
    st.dataframe(
        ui.comparison_table(metrics_by_model, selected_model),
        use_container_width=True,
        hide_index=True,
    )
    # Written from the numbers rather than by hand, so it cannot drift from
    # the results after a retrain.
    st.markdown(ui.honest_verdict(metrics_by_model))

    st.plotly_chart(
        plots.metrics_comparison_fig(
            {name: payload.get("validation", {}) for name, payload in metrics_by_model.items()},
            {
                name: payload.get("validation_ci95", {})
                for name, payload in metrics_by_model.items()
            },
            metric="roc_auc",
        ),
        use_container_width=True,
    )

    if has_labels:
        st.subheader("All models on the currently loaded file")
        st.caption(
            "These curves are computed on the file selected in the sidebar, not on the "
            "held-out split, so they change with the data you load."
        )
        curves_by_model: dict[str, dict[str, Any]] = {}
        for name in metrics_by_model:
            with ui.error_boundary(f"Could not score {name}"):
                # n_boot=0: the intervals above already come from the training
                # run, and bootstrapping every model here would be slow.
                curves_by_model[name] = predictor.evaluate(frame, name, threshold, 0).curves

        if curves_by_model:
            left, right = st.columns(2)
            left.plotly_chart(plots.roc_fig(curves_by_model), use_container_width=True)
            right.plotly_chart(plots.pr_fig(curves_by_model), use_container_width=True)

    torch_models = [
        name for name, payload in metrics_by_model.items() if payload.get("framework") == "torch"
    ]
    if torch_models:
        st.subheader("Training curves")
        for tab, name in zip(st.tabs(torch_models), torch_models, strict=True):
            with tab:
                history = load_history(artifacts_directory, name)
                st.plotly_chart(
                    plots.training_curves_fig(history, f"{name}: training curves"),
                    use_container_width=True,
                )
                if history.get("cv_grid"):
                    st.caption(
                        f"Configuration selected from a {len(history['cv_grid'])}-point grid "
                        "by 5-fold cross-validation on the training split."
                    )


def render_ops(predictor: Predictor) -> None:
    """Show the service's own latency, queue and drift metrics.

    Args:
        predictor: Source of the ``/stats`` snapshot.
    """
    st.subheader("Service metrics")
    st.caption(
        "Recorded inside `InferenceService`, not in the web layer, so these numbers are "
        "populated in local mode with no server running."
    )
    if st.button("Refresh", icon="🔄"):
        st.rerun()

    with ui.error_boundary("Could not read service stats"):
        ui.ops_dashboard(predictor.stats())
