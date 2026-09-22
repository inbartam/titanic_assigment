"""Streamlit UI helpers.

Nothing in this package touches a model directly: all inference goes through
``app.client.Predictor``, which wraps either the in-process
``InferenceService`` or a running FastAPI server.
"""
