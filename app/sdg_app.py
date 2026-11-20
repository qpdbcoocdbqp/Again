"""3D preview web UI for recipe-driven material/scene variation (MVP).

    python -m app.sdg_app            # http://127.0.0.1:7870
"""
import logging
from functools import lru_cache
from pathlib import Path

import click
import gradio as gr

from app import recipe as rc
from app.asset import asset_available, find_screen, load_parts, size_of
from app.glb import build_glb
from app.sampler import randomize as sample
from src.model_viewer import MV_PATH, ensure_model_viewer
from src.simready import CACHE_DIR

DEFAULT_ASSET = "TV_a01"  # opened first when it is downloaded
UI_DIR = Path(__file__).parent / "ui"
log = logging.getLogger(__name__)


def list_assets() -> list[str]:
    """Names of the assets downloaded under data/ (folders that carry the SimReady root-USD metadata)."""
    root = Path(CACHE_DIR)
    return sorted((d.name for d in root.iterdir() if d.is_dir() and asset_available(d.name)), key=str.lower) \
        if root.is_dir() else []


@lru_cache(maxsize=4)
def load(asset: str) -> dict:
    """Parts, default recipe and the GLB for an asset (built once per process, the 4 latest are kept)."""
    if asset not in list_assets():  # the browser passes asset names; never turn them into paths
        raise ValueError(f"unknown asset {asset!r}; downloaded: {', '.join(list_assets()) or 'none'}")
    parts = load_parts(asset)
    screen = find_screen(parts)
    defaults = rc.default_recipe(asset, [(p.name, p.material) for p in parts],
                                 parts[screen].name if screen is not None else None)
    glb = build_glb(parts, defaults, screen, CACHE_DIR / asset / "sdg_parts.glb")
    return {"parts": parts, "defaults": defaults, "glb": glb, "size": size_of(parts), "screen": screen}


def file_url(p: Path) -> str:
    return "/gradio_api/file=" + p.as_posix()


# ---------------------------------------------------------------- functions the browser calls
# Gradio hands a server function a single argument, so the browser sends one object; every call that works on a
# recipe also names the asset the page currently shows ("asset"), which load() checks against data/.
def init(asset: str | None = None) -> dict:
    """Everything the page needs to show `asset` (default: TV_a01 if downloaded, else the first one found). A failure
    comes back as {"error": message}, so the page can keep showing the previous asset."""
    assets = list_assets()
    if not assets:
        return {"error": f"no asset found under {CACHE_DIR}/; download one with: python -m src.simready show <name>"}
    if asset is None:
        asset = DEFAULT_ASSET if DEFAULT_ASSET in assets else assets[0]
    try:
        a = load(asset)
    except Exception as e:  # UI boundary: any unreadable asset must become a message, not a dead page
        log.exception("cannot load %s", asset)
        return {"error": f"{asset}: {e}"}
    return {
        "asset": asset,
        "assets": assets,
        "src": file_url(a["glb"]),
        "mv": file_url(MV_PATH),
        "size_m": [round(float(x), 4) for x in a["size"]],
        "parts": [{"name": p.name, "material": p.material, "faces": p.faces, "tris": len(p.triangles),
                   "screen": i == a["screen"]} for i, p in enumerate(a["parts"])],
        "recipe": a["defaults"],
    }


def check(recipe: dict, asset: str) -> list[str]:
    """Validation problems of a recipe for `asset`."""
    a = load(asset)
    return rc.validate(recipe, [p.name for p in a["parts"]], screen_part=a["defaults"]["screen"]["part"], asset=asset)


def validate(req: dict) -> list[str]:
    """req = {"recipe": ..., "asset": name}."""
    return check(req["recipe"], req["asset"])


def randomize(req: dict) -> dict:
    """req = {"recipe": ..., "seed": int, "asset": name}. Sample a new look from the recipe's randomize section; same
    seed -> same result."""
    recipe, seed, asset = req["recipe"], req["seed"], req["asset"]
    errs = check(recipe, asset)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        errs.append("seed must be an integer in 0..4294967295")
    if errs:
        return {"errors": errs}
    return {"recipe": sample(recipe, load(asset)["defaults"], seed), "errors": []}


def dump_yaml(req: dict) -> dict:
    """req = {"recipe": ..., "asset": name}."""
    errs = check(req["recipe"], req["asset"])
    return {"yaml": None if errs else rc.to_yaml(req["recipe"]), "errors": errs}


def load_yaml(req: dict) -> dict:
    """req = {"text": YAML text, "asset": name}. The YAML must be for that asset (switch the asset first)."""
    asset = req["asset"]
    try:
        r = rc.from_yaml(req["text"], load(asset)["defaults"])
    except ValueError as e:
        return {"errors": [str(e)]}
    errs = check(r, asset)
    return {"recipe": None if errs else r, "errors": errs}


# ---------------------------------------------------------------- page
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SDG Material Preview") as demo:
        gr.HTML(html_template=(UI_DIR / "ui.html").read_text(encoding="utf-8"),
                css_template=(UI_DIR / "ui.css").read_text(encoding="utf-8"),
                js_on_load=(UI_DIR / "ui.js").read_text(encoding="utf-8"),
                server_functions=[init, randomize, dump_yaml, load_yaml, validate])
    return demo


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7870, show_default=True)
def main(host: str, port: int):
    """Material / scene recipe preview in 3D (Gradio)."""
    ensure_model_viewer()
    first = init()  # fail early (and build the first GLB) before the page opens
    if "error" in first:
        raise click.ClickException(first["error"])
    build_demo().launch(server_name=host, server_port=port, allowed_paths=[str(CACHE_DIR)])


if __name__ == "__main__":
    main()
