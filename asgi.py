"""ASGI entry point: the original Render AI app plus the NVIDIA integration.

Run with:  uvicorn asgi:app --host 0.0.0.0 --port $PORT

`main.py` is left untouched; `nvidia.install()` registers the NVIDIA routes and usage limits on it.
"""
from main import app  # noqa: F401  (re-exported for uvicorn)
import nvidia

nvidia.install(app)
