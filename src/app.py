import hashlib
import json
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


def build_glb(asset: str, level: str, alphas: tuple[float, ...] = ()) -> Path:
    """USD -> GLB, one node and one color per part. alphas: per-part opacity (empty = all opaque)."""
    res = LEVELS[level]
    if all(a >= 1.0 for a in alphas):
        alphas = ()
    tag = f"_a{hashlib.md5(repr(alphas).encode()).hexdigest()[:8]}" if alphas else ""
    out = CACHE_DIR / asset / f"preview_{res or 'full'}{tag}.glb"
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
        alpha = alphas[i] if i < len(alphas) else 1.0
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[*part_color(i), alpha],
            metallicFactor=0.1, roughnessFactor=0.7, doubleSided=True,
            alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
        )
        mesh = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals(verts, faces), process=False,
                               visual=trimesh.visual.TextureVisuals(material=material))
        scene.add_geometry(mesh, node_name=part_name(m))

    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(scene.export(file_type="glb", include_normals=True))
    tmp.replace(out)
    return out


def asset_info(asset: str):
    """Thumbnail, summary text, and parts table."""
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
    rows = [[part_name(m), len(m["vertices"]), len(m["faces"]), 1.0] for m in meshes]
    thumb = find_thumbnail(d)
    return (Image.open(thumb) if thumb else None), md, rows


# ---------------------------------------------------------------- events
def on_select(asset: str | None, level: str):
    if asset and asset not in cached_assets():
        raise gr.Error(f"Asset {asset} is not downloaded yet")
    if not asset:
        return None, None, "No assets in data/ yet. Download one from the panel at the bottom left.", []
    thumb, md, rows = asset_info(asset)
    return str(build_glb(asset, level)), thumb, md, rows


def alphas_of(parts) -> tuple[float, ...]:
    """Per-part opacity from the parts table's 4th column, clamped to [0, 1]."""
    out = []
    for v in parts.iloc[:, 3]:
        try:
            out.append(min(max(float(v), 0.0), 1.0))
        except (TypeError, ValueError):
            out.append(1.0)
    return tuple(out)


def on_level(asset: str | None, level: str, parts):
    return str(build_glb(asset, level, alphas_of(parts))) if asset else None


def on_opacity_reset(parts):
    parts = parts.copy()
    parts.iloc[:, 3] = 1.0
    return parts


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
        gr.Markdown("左鍵旋轉、右鍵平移、滾輪縮放")
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
                viewer = gr.Model3D(label="3D 檢視", height=640, clear_color=(0.95, 0.95, 0.95, 1.0))
                parts = gr.Dataframe(headers=["零件", "頂點", "三角形", "透明度"], label="零件（可編輯「透明度」：0 全透明 ~ 1 不透明）",
                                     datatype=["str", "number", "number", "number"], interactive=True,
                                     static_columns=[0, 1, 2])
                with gr.Row():
                    apply_opacity = gr.Button("套用透明度", variant="primary")
                    reset_opacity = gr.Button("全部不透明")

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

    outputs = [viewer, thumb, info, parts]
    asset.change(on_select, [asset, level], outputs)
    level.change(on_level, [asset, level, parts], viewer)
    apply_opacity.click(on_level, [asset, level, parts], viewer)
    reset_opacity.click(on_opacity_reset, parts, parts).then(on_level, [asset, level, parts], viewer)
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
    demo.launch(server_name=host, server_port=port, share=share, allowed_paths=[str(CACHE_DIR)])


if __name__ == "__main__":
    main()
