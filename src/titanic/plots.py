"""Plotly figures built from :mod:`titanic.evaluation` output.

One plotting implementation serves two consumers: ``train.py`` saves each
figure as a standalone HTML file under ``artifacts/<model>/plots/``, and the
Streamlit app renders the same figures with ``st.plotly_chart``. No figure is
ever defined twice, so what a reviewer sees in the app is exactly what the
training run produced.

Plotly rather than matplotlib because the app is the reviewer's main
interaction surface: overlaying four models' ROC curves is only readable when
the legend can toggle traces. The EDA notebook keeps matplotlib, since static
images are what render on GitHub.

Every function takes plain dicts and returns a ``go.Figure``. Nothing here
touches a model, a dataframe or the filesystem.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

#: Colour-blind-safe qualitative palette, assigned per model so a model keeps
#: the same colour across every figure and tab.
MODEL_COLORS: dict[str, str] = {
    "fast": "#4C72B0",
    "deep": "#DD8452",
    "attn": "#55A868",
    "gbdt": "#C44E52",
}
FALLBACK_COLOR = "#8172B3"

#: Shared layout so the figures look like one system rather than seven.
_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 60, "r": 30, "t": 55, "b": 55},
    "height": 380,
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    "hovermode": "closest",
}


def model_color(name: str) -> str:
    """Return the fixed colour for a model name.

    Args:
        name: Model name such as ``"deep"``.

    Returns:
        A hex colour string.
    """
    return MODEL_COLORS.get(name, FALLBACK_COLOR)


def _empty_figure(message: str) -> go.Figure:
    """Build a placeholder figure carrying an explanation.

    Used when a figure cannot be drawn, typically a CSV with only one class,
    where ROC is undefined. Returning a labelled empty figure keeps the app
    layout stable instead of leaving a hole or raising.

    Args:
        message: Text to display in the centre of the plot area.

    Returns:
        A figure with no data and one annotation.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, font={"size": 13, "color": "#666"}, x=0.5, y=0.5
    )
    figure.update_layout(
        **_LAYOUT,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def roc_fig(curves: dict[str, dict[str, Any]], title: str = "ROC curve") -> go.Figure:
    """Overlay ROC curves for one or more models.

    Args:
        curves: Mapping from model name to that model's ``curve_data`` output.
        title: Figure title.

    Returns:
        A Plotly figure with one trace per model plus the chance diagonal.
    """
    figure = go.Figure()
    drawn = 0

    for name, data in curves.items():
        roc = data.get("roc") or {}
        if not roc:
            continue
        figure.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name=f"{name} (AUC {roc['auc']:.3f})",
                line={"color": model_color(name), "width": 2},
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
            )
        )
        drawn += 1

    if not drawn:
        return _empty_figure("ROC needs both classes present in the data.")

    # The diagonal is what a coin flip achieves; every curve is read as
    # distance above it.
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="chance",
            line={"color": "#bbb", "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        **_LAYOUT,
        title=title,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    # scaleanchor keeps the plot square: ROC curves are misleading when the
    # axes are stretched to different scales.
    figure.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 1.02])
    figure.update_xaxes(range=[0, 1.02])
    return figure


def pr_fig(curves: dict[str, dict[str, Any]], title: str = "Precision-recall curve") -> go.Figure:
    """Overlay precision-recall curves for one or more models.

    PR curves are more informative than ROC on an imbalanced problem: they
    describe performance on the minority (survived) class specifically.

    Args:
        curves: Mapping from model name to ``curve_data`` output.
        title: Figure title.

    Returns:
        A Plotly figure with one trace per model plus the base-rate baseline.
    """
    figure = go.Figure()
    baseline: float | None = None

    for name, data in curves.items():
        pr = data.get("pr") or {}
        if not pr:
            continue
        baseline = pr.get("baseline", baseline)
        figure.add_trace(
            go.Scatter(
                x=pr["recall"],
                y=pr["precision"],
                mode="lines",
                name=f"{name} (AP {pr['auc']:.3f})",
                line={"color": model_color(name), "width": 2},
                hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
            )
        )

    if baseline is None:
        return _empty_figure("Precision-recall needs both classes present in the data.")

    # A no-skill classifier sits at the base rate, not at 0.5.
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"base rate ({baseline:.2f})",
            line={"color": "#bbb", "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(**_LAYOUT, title=title, xaxis_title="Recall", yaxis_title="Precision")
    figure.update_yaxes(range=[0, 1.02])
    figure.update_xaxes(range=[0, 1.02])
    return figure


def confusion_matrix_fig(matrix: list[list[int]], title: str = "Confusion matrix") -> go.Figure:
    """Render a 2x2 confusion matrix as an annotated heatmap.

    Args:
        matrix: ``[[tn, fp], [fn, tp]]`` as produced by ``compute_metrics``.
        title: Figure title.

    Returns:
        A Plotly heatmap with counts and row-normalised percentages.
    """
    (tn, fp), (fn, tp) = matrix
    labels = ["Did not survive", "Survived"]

    # Row-normalised: "of the passengers who actually died, what fraction did
    # we call correctly?" is the question a confusion matrix should answer.
    row_totals = [max(tn + fp, 1), max(fn + tp, 1)]
    text = [
        [f"{tn}<br>{100 * tn / row_totals[0]:.1f}%", f"{fp}<br>{100 * fp / row_totals[0]:.1f}%"],
        [f"{fn}<br>{100 * fn / row_totals[1]:.1f}%", f"{tp}<br>{100 * tp / row_totals[1]:.1f}%"],
    ]

    figure = go.Figure(
        go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=[f"Predicted: {label}" for label in labels],
            y=[f"Actual: {label}" for label in labels],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Blues",
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>count %{z}<extra></extra>",
        )
    )
    figure.update_layout(**_LAYOUT, title=title)
    # autorange reversed puts the "actual negative" row on top, matching how
    # sklearn prints the matrix.
    figure.update_yaxes(autorange="reversed")
    return figure


def threshold_sweep_fig(
    sweep: dict[str, list[float]],
    current_threshold: float = 0.5,
    title: str = "Metrics vs decision threshold",
) -> go.Figure:
    """Plot accuracy, precision, recall and F1 across thresholds.

    This figure exists to *show* the precision/recall trade-off, not to pick a
    threshold. No threshold is ever optimised on the validation set and then
    reported.

    Args:
        sweep: ``curve_data(...)["threshold_sweep"]``.
        current_threshold: Threshold to mark with a vertical line.
        title: Figure title.

    Returns:
        A Plotly figure with one line per metric.
    """
    if not sweep:
        return _empty_figure("Threshold sweep needs labelled data.")

    colors = {
        "accuracy": "#4C72B0",
        "precision": "#DD8452",
        "recall": "#55A868",
        "f1": "#C44E52",
    }
    figure = go.Figure()
    for metric, color in colors.items():
        figure.add_trace(
            go.Scatter(
                x=sweep["thresholds"],
                y=sweep[metric],
                mode="lines+markers",
                name=metric,
                line={"color": color, "width": 2},
                marker={"size": 5},
            )
        )

    figure.add_vline(
        x=current_threshold,
        line={"color": "#444", "width": 1, "dash": "dot"},
        annotation_text=f"threshold {current_threshold:g}",
        annotation_position="top",
    )
    figure.update_layout(
        **_LAYOUT, title=title, xaxis_title="Decision threshold", yaxis_title="Score"
    )
    figure.update_yaxes(range=[0, 1.02])
    return figure


def calibration_fig(calibration: dict[str, list[float]], title: str = "Calibration") -> go.Figure:
    """Plot predicted probability against observed frequency.

    Args:
        calibration: ``curve_data(...)["calibration"]``.
        title: Figure title.

    Returns:
        A Plotly figure with the calibration line and the ideal diagonal.
    """
    if not calibration or not calibration.get("mean_predicted"):
        return _empty_figure("Calibration needs labelled data.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfectly calibrated",
            line={"color": "#bbb", "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=calibration["mean_predicted"],
            y=calibration["observed"],
            mode="lines+markers",
            name="model",
            line={"color": "#4C72B0", "width": 2},
            # Marker size carries the bin count, so a point resting on three
            # passengers is visibly less trustworthy than one resting on fifty.
            marker={
                "size": [
                    8 + 22 * count / max(calibration["counts"]) for count in calibration["counts"]
                ],
                "color": "#4C72B0",
                "opacity": 0.75,
            },
            customdata=calibration["counts"],
            hovertemplate=(
                "predicted %{x:.2f}<br>observed %{y:.2f}<br>%{customdata} passengers"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_LAYOUT,
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed survival rate",
    )
    figure.update_yaxes(range=[0, 1.02])
    figure.update_xaxes(range=[0, 1.02])
    return figure


def prob_histogram_fig(
    probabilities: list[float],
    y_true: list[int] | None = None,
    threshold: float = 0.5,
    title: str = "Predicted probability distribution",
) -> go.Figure:
    """Histogram of predicted probabilities, split by true label when known.

    Args:
        probabilities: Predicted ``P(survived)`` per row.
        y_true: Ground truth, when available. Without it a single series is
            drawn, which is the inference-only case.
        threshold: Decision threshold to mark.
        title: Figure title.

    Returns:
        A Plotly histogram.
    """
    if not probabilities:
        return _empty_figure("No predictions to display.")

    figure = go.Figure()
    bins = {"start": 0.0, "end": 1.0, "size": 0.05}

    if y_true is not None and len(set(y_true)) > 1:
        for label, name, color in ((0, "did not survive", "#C44E52"), (1, "survived", "#55A868")):
            values = [p for p, t in zip(probabilities, y_true, strict=True) if t == label]
            figure.add_trace(
                go.Histogram(x=values, name=name, marker_color=color, opacity=0.7, xbins=bins)
            )
        # Overlay rather than stack: the question is how far apart the two
        # distributions sit, which a stacked chart hides.
        figure.update_layout(barmode="overlay")
    else:
        figure.add_trace(
            go.Histogram(x=probabilities, name="predictions", marker_color="#4C72B0", xbins=bins)
        )

    figure.add_vline(
        x=threshold,
        line={"color": "#444", "width": 1, "dash": "dot"},
        annotation_text=f"threshold {threshold:g}",
    )
    figure.update_layout(
        **_LAYOUT, title=title, xaxis_title="P(survived)", yaxis_title="Passengers"
    )
    return figure


def training_curves_fig(history: dict[str, Any], title: str = "Training curves") -> go.Figure:
    """Plot training and inner-validation loss per epoch.

    The inner-validation series is the 10% carve-out from inside the training
    split, never the held-out validation set.

    Args:
        history: ``history.json`` contents.
        title: Figure title.

    Returns:
        A Plotly figure with both loss curves and the best epoch marked.
    """
    if not history or not history.get("epochs"):
        return _empty_figure("No training history (this model is not a neural network).")

    figure = go.Figure()
    for key, name, color in (
        ("train_loss", "train loss", "#4C72B0"),
        ("val_loss", "inner-val loss", "#DD8452"),
    ):
        figure.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history[key],
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
            )
        )

    best_epoch = history.get("best_epoch")
    if best_epoch:
        figure.add_vline(
            x=best_epoch,
            line={"color": "#55A868", "width": 1, "dash": "dot"},
            annotation_text=f"best epoch {best_epoch}",
            annotation_position="top right",
        )

    figure.update_layout(**_LAYOUT, title=title, xaxis_title="Epoch", yaxis_title="BCE loss")
    return figure


def metrics_comparison_fig(
    metrics_by_model: dict[str, dict[str, Any]],
    ci_by_model: dict[str, dict[str, list[float]]] | None = None,
    metric: str = "roc_auc",
    title: str | None = None,
) -> go.Figure:
    """Compare one metric across models, with confidence intervals as error bars.

    This is the figure that makes the project's central point: when the
    intervals overlap, the models are not distinguishable on this data.

    Args:
        metrics_by_model: Model name to its validation metrics dict.
        ci_by_model: Model name to its bootstrap intervals.
        metric: Which metric to compare.
        title: Figure title; derived from ``metric`` when omitted.

    Returns:
        A Plotly bar chart with asymmetric error bars.
    """
    ci_by_model = ci_by_model or {}
    names = [name for name, m in metrics_by_model.items() if m.get(metric) is not None]
    if not names:
        return _empty_figure(f"No model reports {metric}.")

    values = [metrics_by_model[name][metric] for name in names]
    # Error bars are distances from the bar top, and the bootstrap interval is
    # not symmetric around the point estimate, so both sides are computed.
    error_plus, error_minus = [], []
    for name, value in zip(names, values, strict=True):
        interval = ci_by_model.get(name, {}).get(metric)
        if interval:
            error_plus.append(max(interval[1] - value, 0))
            error_minus.append(max(value - interval[0], 0))
        else:
            error_plus.append(0)
            error_minus.append(0)

    figure = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[model_color(name) for name in names],
            error_y={
                "type": "data",
                "array": error_plus,
                "arrayminus": error_minus,
                "color": "#333",
                "thickness": 1.4,
                "width": 6,
            },
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>" + metric + " %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_LAYOUT,
        title=title or f"{metric} by model (95% bootstrap CI)",
        xaxis_title="",
        yaxis_title=metric,
        showlegend=False,
    )
    lowest = min(v - e for v, e in zip(values, error_minus, strict=True))
    # Zoom to the region the bars occupy: on a 0-1 axis a 0.02 difference is
    # invisible, and that difference is exactly what the reader must judge.
    figure.update_yaxes(range=[max(0.0, lowest - 0.08), 1.02])
    return figure


def stage_latency_fig(
    per_stage: dict[str, dict[str, float]], title: str = "Latency by stage"
) -> go.Figure:
    """Grouped bars of p50/p95/p99 latency per inference stage.

    Args:
        per_stage: ``{"queue": {"p50": .., "p95": ..}, "preprocess": {...}, ...}``.
        title: Figure title.

    Returns:
        A Plotly grouped bar chart.
    """
    if not per_stage:
        return _empty_figure("No requests recorded yet. Run a prediction first.")

    stages = list(per_stage)
    figure = go.Figure()
    for percentile, color in (("p50", "#4C72B0"), ("p95", "#DD8452"), ("p99", "#C44E52")):
        figure.add_trace(
            go.Bar(
                name=percentile,
                x=stages,
                y=[per_stage[stage].get(percentile, 0.0) for stage in stages],
                marker_color=color,
                hovertemplate="%{x} " + percentile + " %{y:.2f} ms<extra></extra>",
            )
        )
    figure.update_layout(
        **_LAYOUT, title=title, barmode="group", xaxis_title="", yaxis_title="milliseconds"
    )
    return figure
