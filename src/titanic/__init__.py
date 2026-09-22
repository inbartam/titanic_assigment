"""Titanic survival classification: data, features, models, service and artifacts.

The package is deliberately layered so that every consumer (the training CLI,
the Streamlit app, the FastAPI service and the notebook) reuses the *same*
code path:

``data`` -> ``features`` -> ``preprocessing`` -> ``models`` -> ``training``
-> ``artifacts`` -> ``service``.

Nothing in ``app/`` or ``api/`` is allowed to touch a model directly; both go
through :class:`titanic.service.InferenceService`.
"""

__version__ = "0.1.0"
