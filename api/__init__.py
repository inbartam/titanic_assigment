"""FastAPI adapter over titanic.service.InferenceService.

The package is intentionally thin: route handlers parse input, call the
service, and map typed exceptions to HTTP status codes. No inference logic and
no metric recording happens here -- both belong to the service, so the
Streamlit app in local mode reports identical numbers with no server running.
"""
