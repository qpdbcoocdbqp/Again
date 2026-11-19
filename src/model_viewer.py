"""Local copy of Google's <model-viewer> web component, shared by the Gradio apps (src/app.py, app/sdg_app.py)."""
import hashlib
import urllib.request

from src.simready import CACHE_DIR

MV_URL = "https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"  # versioned URL: content is fixed
MV_SHA256 = "774edda21e1be2a0934e460ca5943af1fe3f88da130a9f98bd6a9d611576cacf"
MV_PATH = CACHE_DIR / "_vendor" / "model-viewer.min.js"


def ensure_model_viewer(timeout: float = 30) -> None:
    """Download model-viewer once into data/_vendor/ (checked against MV_SHA256); later runs use the local copy."""
    if MV_PATH.exists():
        return
    MV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MV_URL, timeout=timeout) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != MV_SHA256:
        raise RuntimeError(f"{MV_URL} does not match the expected SHA-256; refusing to use it")
    tmp = MV_PATH.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(MV_PATH)
