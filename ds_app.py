"""Streamlit entry point: inference and evaluation over the trained models.

Deliberately thin. Every tab lives in ``app/tabs.py``, reusable widgets in
``app/components.py``, caching in ``app/state.py``, and all inference behind
``app/client.py``'s ``Predictor``. This file only wires them together.

Run it with::

    streamlit run ds_app.py

No server is required: local mode loads the bundles from ``artifacts/`` in
process, which is what the assignment asks for. Setting ``TITANIC_API_URL``
routes inference through the FastAPI service instead, and an unreachable API
falls back to local mode with a visible warning.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import components as ui
from app import tabs
from app.state import (
    artifacts_dir,
    frame_fingerprint,
    get_predictor,
    load_dataframe,
    resolve_api_url,
)
from titanic.config import Paths

st.set_page_config(
    page_title="Titanic Survival — Inference & Evaluation",
    page_icon="🚢",
    layout="wide",
)

PATHS = Paths()
ARTIFACTS = artifacts_dir()


def load_from_sidebar() -> tuple[object | None, str | None]:
    """Render the data-source controls and load the chosen CSV.

    Returns:
        ``(dataframe, error)``. Exactly one is ever non-``None``; both are
        ``None`` while the user has not yet chosen a file to upload.
    """
    source = st.radio("Source", ["Bundled sample", "Upload CSV", "Path on disk"])

    try:
        if source == "Bundled sample":
            frame = load_dataframe(str(PATHS.sample_csv))
            st.caption("100 stratified rows from the Kaggle training set.")
            return frame, None

        if source == "Upload CSV":
            upload = st.file_uploader("Choose a CSV", type=["csv"])
            if upload is None:
                return None, None
            return load_dataframe(upload.name, upload.getvalue()), None

        typed = st.text_input("CSV path", value=str(PATHS.sample_csv))
        if not typed:
            return None, None
        # Check the path before reading so the user gets a precise message
        # rather than a pandas error about a missing file.
        if not typed.lower().endswith(".csv"):
            return None, f"Expected a .csv file, got {typed!r}."
        if not Path(typed).is_file():
            return None, f"No such file: {Path(typed).resolve()}"
        return load_dataframe(typed), None

    except Exception as exc:
        return None, str(exc)


# --------------------------------------------------------------------------
# Predictor: the single dependency every tab shares
# --------------------------------------------------------------------------

try:
    predictor, mode_warning = get_predictor(resolve_api_url(), ARTIFACTS)
    models = predictor.models()
except Exception as exc:
    # The one unrecoverable state: no models on disk. Show the exact command
    # instead of a traceback.
    st.error(f"Could not load any trained model.\n\n{exc}")
    st.code("python train.py --model all", language="powershell")
    st.stop()

if not models:
    st.error("No models are registered yet.")
    st.code("python train.py --model all", language="powershell")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🚢 Titanic")
    selected_model = ui.model_selector(models, predictor.default_model())

    st.divider()
    st.subheader("Data")
    frame, load_error = load_from_sidebar()

    st.divider()
    threshold = ui.threshold_slider()

    st.divider()
    ui.mode_badge(predictor.mode, mode_warning)

if load_error:
    st.error(load_error)
    st.stop()
if frame is None:
    st.info("Choose a data source in the sidebar to begin.")
    st.stop()

has_labels = "Survived" in frame.columns

# --------------------------------------------------------------------------
# Inference, recomputed only when the model, data or threshold changes
# --------------------------------------------------------------------------

cache_key = (selected_model, frame_fingerprint(frame), threshold)

if st.session_state.get("cache_key") != cache_key:
    with st.spinner("Running inference..."), ui.error_boundary("Inference failed"):
        st.session_state["result"] = predictor.predict(frame, selected_model, threshold)
        st.session_state["evaluation"] = (
            predictor.evaluate(frame, selected_model, threshold, 1000) if has_labels else None
        )
        st.session_state["cache_key"] = cache_key

result = st.session_state.get("result")
evaluation = st.session_state.get("evaluation")
if result is None:
    st.stop()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

overview_tab, data_tab, predictions_tab, evaluation_tab, compare_tab, ops_tab = st.tabs(
    ["Overview", "Data", "Predictions", "Evaluation", "Compare models", "Ops"]
)

with overview_tab, ui.error_boundary("Could not render the overview"):
    tabs.render_overview(selected_model, models)

with data_tab, ui.error_boundary("Could not render the data preview"):
    tabs.render_data(frame)

with predictions_tab, ui.error_boundary("Could not render predictions"):
    tabs.render_predictions(result, frame, has_labels)

with evaluation_tab, ui.error_boundary("Could not render the evaluation"):
    tabs.render_evaluation(evaluation, has_labels)

with compare_tab, ui.error_boundary("Could not build the comparison"):
    tabs.render_compare(predictor, ARTIFACTS, selected_model, frame, threshold, has_labels)

with ops_tab:
    tabs.render_ops(predictor)
