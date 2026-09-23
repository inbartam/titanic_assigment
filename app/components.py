"""Reusable UI blocks for the Streamlit app.

Each function renders one section and returns nothing, or returns the value a
sidebar control produced. Keeping them here is what allows ``ds_app.py`` to
stay short enough to read in one screen.

No function in this module touches a model or the filesystem: they all take
data that a caller already fetched through the predictor.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd
import streamlit as st

from titanic import plots
from titanic.config import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, TRAIN_BASE_RATE

#: Metrics shown as tiles on the Evaluation tab, in reading order.
TILE_METRICS: tuple[tuple[str, str], ...] = (
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
)

#: One-line descriptions shown beside each model in the sidebar.
MODEL_DESCRIPTIONS: dict[str, str] = {
    "fast": "logistic regression (PyTorch)",
    "deep": "MLP with embeddings (PyTorch)",
    "attn": "tiny transformer (PyTorch)",
    "gbdt": "gradient boosting (sklearn)",
}


@contextmanager
def error_boundary(context: str) -> Iterator[None]:
    """Render any exception as a readable message instead of a red traceback.

    Streamlit's default is a full traceback in the page, which is hard to read
    for anyone who did not write the code. This shows the actionable message and tucks
    the traceback into an expander for whoever wants it.

    Args:
        context: What was being attempted, used in the error heading.
    """
    try:
        yield
    except Exception as exc:
        st.error(f"{context}: {exc}")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)), language="text")


def metric_tile(
    label: str, value: float | None, interval: list[float] | None = None, help_text: str = ""
) -> None:
    """Render one metric with its confidence interval as the caption.

    Args:
        label: Display name.
        value: Point estimate, or ``None`` when undefined.
        interval: ``[low, high]`` bootstrap bounds.
        help_text: Tooltip text.
    """
    if value is None:
        st.metric(label, "n/a", help=help_text or "Needs both classes present.")
        return

    st.metric(label, f"{value:.3f}", help=help_text)
    if interval:
        # A caption keeps the interval next to the number instead of in a
        # separate table.
        st.caption(f"95% CI [{interval[0]:.3f}, {interval[1]:.3f}]")


def metric_tiles(metrics: dict[str, Any], ci95: dict[str, list[float]]) -> None:
    """Render the full metric row in two columns of three.

    Args:
        metrics: Point estimates.
        ci95: Bootstrap intervals.
    """
    helps = {
        "accuracy": "Fraction of passengers classified correctly.",
        "precision": "Of those predicted to survive, how many did.",
        "recall": "Of those who survived, how many were found.",
        "f1": "Harmonic mean of precision and recall.",
        "roc_auc": "Ranking quality, independent of the threshold.",
        "pr_auc": "Ranking quality on the minority (survived) class.",
    }
    for row_start in (0, 3):
        for column, (key, label) in zip(
            st.columns(3), TILE_METRICS[row_start : row_start + 3], strict=False
        ):
            with column:
                metric_tile(label, metrics.get(key), ci95.get(key), helps.get(key, ""))


def schema_panel(df: pd.DataFrame) -> None:
    """Show whether the loaded data matches the expected schema.

    Args:
        df: The loaded dataframe.
    """
    present = [column for column in REQUIRED_COLUMNS if column in df.columns]
    has_target = "Survived" in df.columns

    if len(present) == len(REQUIRED_COLUMNS):
        st.success(
            f"Schema OK: {len(present)}/{len(REQUIRED_COLUMNS)} required columns present. "
            + (
                "Labels found, so evaluation is available."
                if has_target
                else "No 'Survived' column, so only predictions are available."
            )
        )
    else:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        st.error(f"Missing required columns: {missing}")

    with st.expander("Expected schema"):
        st.markdown(
            f"**Required** ({len(REQUIRED_COLUMNS)}): `{'`, `'.join(REQUIRED_COLUMNS)}`\n\n"
            f"**Optional**: `{'`, `'.join(OPTIONAL_COLUMNS)}`\n\n"
            "This is the raw Kaggle Titanic schema. `Survived` is what switches "
            "the app from inference-only to full evaluation. Extra columns are ignored."
        )


def missingness_chart(df: pd.DataFrame) -> None:
    """Show which columns have missing values.

    Args:
        df: The loaded dataframe.
    """
    missing = (100 * df.isna().mean()).round(1)
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.info("No missing values in this file.")
        return

    st.bar_chart(missing, horizontal=True, x_label="% missing", use_container_width=True)
    st.caption(
        "Missing values are imputed with statistics fitted on the training split: "
        "Age by Title median, Fare by median, Embarked by mode, and Cabin becomes deck 'U'."
    )


def model_selector(models: dict[str, dict[str, Any]], default: str | None) -> str:
    """Render the sidebar model radio, annotated with each model's metrics.

    Only models actually present are offered, so a registry containing just
    ``fast`` produces a working single-option app rather than an error.

    Args:
        models: Model metadata from the predictor.
        default: Model to preselect.

    Returns:
        The selected model name.
    """
    names = list(models)
    index = names.index(default) if default in names else 0

    def describe(name: str) -> str:
        """Build the radio label for one model."""
        return f"{name}: {MODEL_DESCRIPTIONS.get(name, 'model')}"

    selected = st.radio("Model", names, index=index, format_func=describe)

    entry = models.get(selected, {})
    validation = entry.get("validation", {})
    roc_auc = validation.get("roc_auc") or entry.get("roc_auc")
    parts = [f"{entry.get('n_params', 0):,} params"]
    if roc_auc:
        parts.append(f"val ROC-AUC {roc_auc:.3f}")
    if entry.get("inference_ms_per_1k_rows"):
        parts.append(f"{entry['inference_ms_per_1k_rows']:.1f} ms/1k rows")
    st.caption(" · ".join(parts))

    return selected


def threshold_slider() -> float:
    """Render the decision-threshold slider with its caveat.

    Returns:
        The selected threshold.
    """
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption(
        "Default 0.5. Lower → higher recall on survivors; higher → higher precision. "
        "Metrics below update live, but the threshold is never tuned on this data and "
        "then reported as a result."
    )
    return threshold


def mode_badge(mode: str, warning: str | None) -> None:
    """Show which inference path the app is using.

    Args:
        mode: Description from the predictor.
        warning: Fallback explanation, if the API was unreachable.
    """
    if warning:
        st.warning(warning, icon="⚠️")
    if mode.startswith("API"):
        st.info(f"Mode: {mode}", icon="🌐")
    else:
        st.caption(f"Mode: {mode}. No server required.")


def comparison_table(metrics_by_model: dict[str, dict[str, Any]], selected: str) -> pd.DataFrame:
    """Build the Compare tab's table of every trained model.

    Args:
        metrics_by_model: Model name to its ``metrics.json`` payload.
        selected: Currently selected model, marked in the output.

    Returns:
        A display-ready dataframe.
    """
    rows = []
    for name, payload in metrics_by_model.items():
        validation = payload.get("validation", {})
        intervals = payload.get("validation_ci95", {})

        def with_interval(metric: str, v=validation, c=intervals) -> str:
            """Format a metric and its interval for the table."""
            value = v.get(metric)
            if value is None:
                return "n/a"
            bounds = c.get(metric)
            return f"{value:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else f"{value:.3f}"

        rows.append(
            {
                "model": f"▶ {name}" if name == selected else name,
                "framework": payload.get("framework", ""),
                "params": f"{payload.get('n_params', 0):,}",
                "accuracy": with_interval("accuracy"),
                "roc_auc": with_interval("roc_auc"),
                "pr_auc": with_interval("pr_auc"),
                "f1": f"{validation.get('f1', 0):.3f}",
                "ms/1k rows": f"{payload.get('inference_ms_per_1k_rows', 0):.1f}",
            }
        )

    return pd.DataFrame(rows)


def honest_verdict(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    """Write the comparison paragraph directly from the numbers.

    Generated rather than hand-written so it can never drift from the results,
    and so it says something true even if retraining changes the ranking.

    Args:
        metrics_by_model: Model name to its metrics payload.

    Returns:
        A short markdown paragraph.
    """
    scored = {
        name: payload
        for name, payload in metrics_by_model.items()
        if payload.get("validation", {}).get("roc_auc") is not None
    }
    if len(scored) < 2:
        return "Train more than one model to see a comparison."

    ranked = sorted(scored.items(), key=lambda item: item[1]["validation"]["roc_auc"], reverse=True)
    best_name, best = ranked[0]
    best_auc = best["validation"]["roc_auc"]
    best_ci = best.get("validation_ci95", {}).get("roc_auc")

    # "Indistinguishable" means inside the leader's interval. At n=179 that is
    # the only defensible reading of a 0.01 gap.
    overlapping = [
        name
        for name, payload in ranked[1:]
        if best_ci and payload["validation"]["roc_auc"] >= best_ci[0]
    ]

    simplest = min(scored.items(), key=lambda item: item[1].get("n_params", 0))
    n_val = best.get("n_val", "?")

    text = (
        f"**`{best_name}` leads on ROC-AUC ({best_auc:.3f}"
        + (f", 95% CI [{best_ci[0]:.3f}, {best_ci[1]:.3f}]" if best_ci else "")
        + f"), on n = {n_val} held-out passengers.** "
    )

    if overlapping:
        text += (
            f"But {', '.join(f'`{n}`' for n in overlapping)} "
            f"{'fall' if len(overlapping) > 1 else 'falls'} inside that interval, so on this "
            "data the models are not statistically distinguishable. "
        )
    else:
        text += "No other model falls inside that interval. "

    text += (
        f"`{simplest[0]}` is the smallest model at {simplest[1].get('n_params', 0):,} "
        f"parameters (ROC-AUC {simplest[1]['validation']['roc_auc']:.3f}). "
    )

    if simplest[0] != best_name and best_ci and simplest[1]["validation"]["roc_auc"] >= best_ci[0]:
        text += (
            f"Since it is inside the leader's confidence interval, `{simplest[0]}` is the "
            "one to ship: the same measured performance, and less to train, serve and explain."
        )
    else:
        text += f"On these numbers `{best_name}` is the one to ship."

    return text


def ops_dashboard(stats: dict[str, Any]) -> None:
    """Render the Ops tab from a ``/stats`` snapshot.

    Args:
        stats: Snapshot from the predictor.
    """
    requests = stats.get("requests", {})
    queue = stats.get("queue", {})
    latency = stats.get("latency_ms", {})

    st.subheader("Traffic")
    columns = st.columns(4)
    columns[0].metric("Requests", f"{requests.get('total', 0):,}")
    columns[1].metric("Rows predicted", f"{stats.get('rows_predicted', {}).get('total', 0):,}")
    columns[2].metric("Error rate", f"{100 * requests.get('error_rate', 0):.1f}%")
    columns[3].metric("Requests/s (1m)", f"{requests.get('rps_1m', 0):.2f}")

    st.subheader("Latency")
    columns = st.columns(4)
    columns[0].metric("p50", f"{latency.get('p50', 0):.1f} ms")
    columns[1].metric("p95", f"{latency.get('p95', 0):.1f} ms")
    columns[2].metric("p99", f"{latency.get('p99', 0):.1f} ms")
    columns[3].metric("max", f"{latency.get('max', 0):.1f} ms")

    per_stage = latency.get("per_stage", {})
    if per_stage:
        st.plotly_chart(plots.stage_latency_fig(per_stage), use_container_width=True)
        st.caption(
            "Timing each stage separately matters because 'the model is slow' and "
            "'preprocessing is slow' have different fixes."
        )

    st.subheader("Queue and back-pressure")
    columns = st.columns(4)
    columns[0].metric("Queue depth", queue.get("depth", 0), help="Requests waiting for a slot.")
    columns[1].metric("In-flight", queue.get("inflight", 0), help="Requests executing now.")
    columns[2].metric("Peak depth", queue.get("max_depth_window", 0))
    rejections = queue.get("rejections", {})
    columns[3].metric("Rejected (503)", sum(rejections.values()) if rejections else 0)
    st.caption(
        f"Bounded at max_concurrency={queue.get('max_concurrency', '?')}, "
        f"max_queue={queue.get('max_queue', '?')}. Queue depth counts requests that are "
        "waiting, not executing. In-flight saturates as soon as the service is busy and "
        "then stops being informative, which is why autoscalers watch depth instead."
    )

    predictions = stats.get("predictions", {})
    positive_rates = predictions.get("positive_rate_window", {})
    if positive_rates:
        st.subheader("Prediction drift")
        base_rate = predictions.get("train_base_rate", TRAIN_BASE_RATE)
        for name, rate in positive_rates.items():
            drift = rate - base_rate
            st.metric(
                f"{name}: predicted survival rate",
                f"{rate:.3f}",
                delta=f"{drift:+.3f} vs training base rate {base_rate:.3f}",
                delta_color="off",
            )
        st.caption(
            "A cheap drift signal: if the served positive rate moves far from the training "
            "base rate, the input distribution has probably changed."
        )

    models = stats.get("models", {})
    if models:
        st.subheader("Loaded models")
        st.dataframe(pd.DataFrame(models).T, use_container_width=True)
