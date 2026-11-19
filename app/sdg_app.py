"""3D preview web UI for recipe-driven material/scene variation (MVP).

    python -m app.sdg_app            # http://127.0.0.1:7870
"""
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

ASSET = "TV_a01"          # MVP: a single asset
UI_DIR = Path(__file__).parent / "ui"


@lru_cache(maxsize=2)
def load(asset: str) -> dict:
    """Parts, default recipe and the GLB for an asset (built once per process)."""
    if asset != ASSET:  # the browser passes asset names; never turn them into paths
        raise ValueError(f"unsupported asset {asset!r}; this MVP only handles {ASSET}")
    if not asset_available(asset):
        raise FileNotFoundError(f"{asset} is not downloaded; run: python -m src.simready show {asset}")
    parts = load_parts(asset)
    screen = find_screen(parts)
    defaults = rc.default_recipe(asset, [(p.name, p.material) for p in parts],
                                 parts[screen].name if screen is not None else None)
    glb = build_glb(parts, defaults, screen, CACHE_DIR / asset / "sdg_parts.glb")
    return {"parts": parts, "defaults": defaults, "glb": glb, "size": size_of(parts), "screen": screen}


def file_url(p: Path) -> str:
    return "/gradio_api/file=" + p.as_posix()


# ---------------------------------------------------------------- functions the browser calls
def init(asset: str = ASSET) -> dict:
    a = load(asset)  # rejects any asset but the supported one
    return {
        "asset": asset,
        "src": file_url(a["glb"]),
        "mv": file_url(MV_PATH),
        "size_m": [round(float(x), 4) for x in a["size"]],
        "parts": [{"name": p.name, "material": p.material, "faces": p.faces, "tris": len(p.triangles),
                   "screen": i == a["screen"]} for i, p in enumerate(a["parts"])],
        "recipe": a["defaults"],
    }


def check(recipe: dict) -> list[str]:
    """Validation problems of a recipe for the (single) supported asset."""
    a = load(ASSET)
    return rc.validate(recipe, [p.name for p in a["parts"]], screen_part=a["defaults"]["screen"]["part"], asset=ASSET)


def validate(recipe: dict) -> list[str]:
    return check(recipe)


def randomize(req: dict) -> dict:
    """req = {"recipe": ..., "seed": int}. Sample a new look from the recipe's randomize section; same seed -> same
    result. (Gradio hands a server function a single argument, so the browser sends one object.)"""
    recipe, seed = req["recipe"], req["seed"]
    errs = check(recipe)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        errs.append("seed must be an integer in 0..4294967295")
    if errs:
        return {"errors": errs}
    return {"recipe": sample(recipe, load(ASSET)["defaults"], seed), "errors": []}


def dump_yaml(recipe: dict) -> dict:
    errs = check(recipe)
    return {"yaml": None if errs else rc.to_yaml(recipe), "errors": errs}


def load_yaml(req: dict) -> dict:
    """req = {"text": YAML text}."""
    try:
        r = rc.from_yaml(req["text"], load(ASSET)["defaults"])
    except ValueError as e:
        return {"errors": [str(e)]}
    errs = check(r)
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
    load(ASSET)  # fail early (and build the GLB) before the page opens
    build_demo().launch(server_name=host, server_port=port, allowed_paths=[str(CACHE_DIR)])


if __name__ == "__main__":
    main()
