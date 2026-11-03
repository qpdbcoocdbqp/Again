import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import click
import gradio as gr
import numpy as np
import trimesh
from PIL import Image

from src.catalog import DB_PATH, SORTS, query
from src.simready import (CACHE_DIR, bounds, decimate, download_asset, find_thumbnail, load_meshes, part_color,
                          part_name, root_usd)

# Decimation levels: number of grid cells along the longest edge (None = original resolution)
LEVELS = {"原始": None, "高": 400, "中": 200, "低": 100}


def cached_assets() -> list[str]:
    """Assets already downloaded under data/ (folders containing root_usds.json)."""
    return sorted(p.parent.parent.name for p in CACHE_DIR.glob("*/.metadata/com.nvidia.simready.root_usds.json"))


@lru_cache(maxsize=4)
def meshes_of(asset: str) -> list[dict]:
    return load_meshes(root_usd(CACHE_DIR / asset))


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (trimesh's fallback is very slow without scipy)."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    idx = f.ravel()
    n = np.stack([np.bincount(idx, np.repeat(fn[:, k], 3), minlength=len(v)) for k in range(3)], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def build_glb(asset: str, level: str) -> Path:
    """USD -> GLB, one node and one color per part. Material "p<i>" belongs to part i (the viewer finds it by name)."""
    res = LEVELS[level]
    out = CACHE_DIR / asset / f"preview_{res or 'full'}_n.glb"
    if out.exists():
        return out

    meshes = meshes_of(asset)
    lo, hi = bounds(meshes)
    size = (hi - lo).max()
    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        verts, faces = m["vertices"], m["faces"]
        if res:
            verts, faces = decimate(verts, faces, size / res)
        if len(faces) == 0:
            continue
        verts = np.c_[verts[:, 0], verts[:, 2], -verts[:, 1]]  # Z-up -> glTF's Y-up
        material = trimesh.visual.material.PBRMaterial(
            name=f"p{i}", baseColorFactor=[*part_color(i), 1.0],
            metallicFactor=0.1, roughnessFactor=0.7, doubleSided=True,
        )
        mesh = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals(verts, faces), process=False,
                               visual=trimesh.visual.TextureVisuals(material=material))
        scene.add_geometry(mesh, node_name=part_name(m))

    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(scene.export(file_type="glb", include_normals=True))
    tmp.replace(out)
    return out


def asset_info(asset: str):
    """Thumbnail and summary text."""
    d = CACHE_DIR / asset
    meshes = meshes_of(asset)
    lo, hi = bounds(meshes)
    sx, sy, sz = hi - lo
    total = sum(len(m["faces"]) for m in meshes)
    pkg_file = d / "com.nvidia.simready.packaging.json"
    pkg = json.loads(pkg_file.read_text()) if pkg_file.exists() else {}

    md = (f"### {asset}\n"
          f"- 尺寸：{sx:.2f} × {sy:.2f} × {sz:.2f} m\n"
          f"- 零件：{len(meshes)}，三角形：{total:,}\n"
          f"- 授權：{pkg.get('license', '-')}\n"
          f"- package：`{pkg.get('package_id', '-')}`")
    thumb = find_thumbnail(d)
    return (Image.open(thumb) if thumb else None), md


# ---------------------------------------------------------------- 3D viewer (model-viewer, served from the local cache)
MV_URL = "https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"
MV_PATH = CACHE_DIR / "_vendor" / "model-viewer.min.js"
VIEWER_JS = (Path(__file__).parent / "viewer.js").read_text(encoding="utf-8")
VIEWER_HTML = """
<div class="sv">
  <div class="sv-view"></div>
  <div class="sv-side">
    <div class="sv-bar">
      <button data-act="solo">只看選取</button><button data-act="reset">全部還原</button>
    </div>
    <div class="sv-hint">點 3D 中的零件可選取；拖曳滑桿調透明度</div>
    <div class="sv-list"></div>
  </div>
</div>"""
VIEWER_CSS = """
.sv { display: flex; gap: 8px; }
.sv-view { flex: 3; height: 640px; background: var(--background-fill-secondary); border-radius: 6px; overflow: hidden; }
.sv-view model-viewer { width: 100%; height: 100%; --poster-color: transparent; }
.sv-side { flex: 1.3; min-width: 260px; height: 640px; display: flex; flex-direction: column; gap: 4px; }
.sv-bar { display: flex; gap: 6px; }
.sv-bar button { padding: 2px 10px; border: 1px solid var(--border-color-primary); border-radius: 4px; cursor: pointer;
                 background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); }
.sv-bar button:hover { background: var(--button-secondary-background-fill-hover); }
.sv-hint { font-size: 12px; color: var(--body-text-color-subdued); }
.sv-list { overflow: auto; flex: 1; }
.sv-row { display: grid; grid-template-columns: auto 14px 1fr 90px 38px; gap: 6px; align-items: center;
          padding: 2px 4px; border-radius: 4px; }
.sv-row { color: var(--body-text-color); }
.sv-row.sel { background: color-mix(in srgb, var(--color-accent) 30%, transparent); outline: 1px solid var(--color-accent); }
.sv-row .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; font-size: 13px; }
.sv-row .sw { width: 14px; height: 14px; border-radius: 3px; border: 1px solid var(--border-color-primary); }
.sv-row output { font-size: 12px; text-align: right; color: var(--body-text-color-subdued); }
"""


def ensure_model_viewer() -> None:
    """Download model-viewer once into data/_vendor/; later runs use the local copy."""
    if MV_PATH.exists():
        return
    MV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MV_PATH.with_suffix(".tmp")
    urllib.request.urlretrieve(MV_URL, tmp)
    tmp.replace(MV_PATH)


def file_url(p: Path) -> str:
    return "/gradio_api/file=" + p.as_posix()


def viewer_payload(asset: str, level: str) -> str:
    parts = [{"name": part_name(m), "verts": len(m["vertices"]), "tris": len(m["faces"]),
              "color": "#%02x%02x%02x" % tuple(round(c * 255) for c in part_color(i))}
             for i, m in enumerate(meshes_of(asset))]
    return json.dumps({"asset": asset, "src": file_url(build_glb(asset, level)), "mv": file_url(MV_PATH),
                       "parts": parts})


# ---------------------------------------------------------------- events
def on_select(asset: str | None, level: str):
    if asset and asset not in cached_assets():
        raise gr.Error(f"Asset {asset} is not downloaded yet")
    if not asset:
        return "", None, "No assets in data/ yet. Download one from the panel at the bottom left."
    thumb, md = asset_info(asset)
    return viewer_payload(asset, level), thumb, md


def on_level(asset: str | None, level: str):
    return viewer_payload(asset, level) if asset else ""


def on_refresh(current: str | None):
    choices = cached_assets()
    return gr.Dropdown(choices=choices, value=current if current in choices else (choices[0] if choices else None))


def on_download(name: str):
    name = name.strip()
    if not name:
        raise gr.Error("Please enter an asset name")
    gr.Info(f"Downloading {name}...")
    d = download_asset(name)
    if not (d / ".metadata/com.nvidia.simready.root_usds.json").exists():
        raise gr.Error(f"Asset {name} not found; look up names with `python -m src.catalog search <keyword>`")
    return gr.Dropdown(choices=cached_assets(), value=name)


def on_catalog(keyword: str, sort: str):
    cols, rows = query((keyword or "").strip(), sort)
    mb = [i for i, c in enumerate(cols) if c.endswith("_bytes")]  # show sizes in MB; the DB keeps bytes
    cols = [c.replace("_bytes", "_mb") if i in mb else c for i, c in enumerate(cols)]
    rows = [[round(v / 1e6, 1) if i in mb else v for i, v in enumerate(r)] for r in rows]
    if not DB_PATH.exists() or not rows:
        status = "No rows. If you haven't synced yet, run `python -m src.catalog sync` first."
    else:
        status = f"{len(rows):,} rows from `{DB_PATH}`"
    return gr.Dataframe(value=rows, headers=cols), status


# ---------------------------------------------------------------- UI
with gr.Blocks(title="SimReady Viewer") as demo:
    gr.Markdown("# SimReady 資產檢視器")
    with gr.Tab("檢視"):
        gr.Markdown("左鍵旋轉、右鍵平移、滾輪縮放；右側清單可個別調整零件透明度")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                choices = cached_assets()
                asset = gr.Dropdown(choices, value=choices[0] if choices else None, label="已下載的資產（data/）",
                                    allow_custom_value=True)  # choices change at runtime (downloads)
                refresh = gr.Button("重新整理清單", size="sm")
                level = gr.Radio(list(LEVELS), value="原始", label="減面（三角形太多、瀏覽器卡頓時調低）")
                thumb = gr.Image(label="官方縮圖", interactive=False, height=240)
                info = gr.Markdown()
            with gr.Column(scale=3):
                viewer = gr.HTML(html_template=VIEWER_HTML, css_template=VIEWER_CSS, js_on_load=VIEWER_JS)

    with gr.Tab("資料表"):
        with gr.Row():
            keyword = gr.Textbox(label="名稱關鍵字", placeholder="例如 pole", scale=3)
            sort = gr.Dropdown(list(SORTS), value="name", label="排序（name / 總大小 / 下載大小）", scale=2)
            reload = gr.Button("重新載入", scale=1)
            with gr.Accordion("下載新資產", open=False):
                new_name = gr.Textbox(label="資產名稱", placeholder="例如 IV_pole_a02")
                download = gr.Button("下載")
        catalog_status = gr.Markdown()
        catalog = gr.Dataframe(label="assets（catalog.db）", interactive=False, max_height=700)

    outputs = [viewer, thumb, info]
    asset.change(on_select, [asset, level], outputs)
    level.change(on_level, [asset, level], viewer)
    refresh.click(on_refresh, asset, asset)
    download.click(on_download, new_name, asset)  # updating asset triggers asset.change
    # pick up assets downloaded after the server started, then render the selection
    demo.load(on_refresh, asset, asset).then(on_select, [asset, level], outputs)

    catalog_inputs, catalog_outputs = [keyword, sort], [catalog, catalog_status]
    keyword.change(on_catalog, catalog_inputs, catalog_outputs)
    sort.change(on_catalog, catalog_inputs, catalog_outputs)
    reload.click(on_catalog, catalog_inputs, catalog_outputs)
    demo.load(on_catalog, catalog_inputs, catalog_outputs)


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to bind to")
@click.option("--port", default=7860, show_default=True, help="Port number")
@click.option("--share", is_flag=True, help="Create a public Gradio share link")
def main(host: str, port: int, share: bool):
    """SimReady asset viewer (Gradio)."""
    ensure_model_viewer()
    demo.launch(server_name=host, server_port=port, share=share, allowed_paths=[str(CACHE_DIR)])


if __name__ == "__main__":
    main()
