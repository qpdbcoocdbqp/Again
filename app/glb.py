"""Parts -> GLB: one node per part, material "p<i>" belongs to part i (the browser finds it by name).

The GLB only carries the initial look; the browser edits the materials live, so recipe changes never rebuild it."""
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from app.asset import Part, planar_uv


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (trimesh's fallback is very slow without scipy)."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    idx = f.ravel()
    n = np.stack([np.bincount(idx, np.repeat(fn[:, k], 3), minlength=len(v)) for k in range(3)], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _white() -> Image.Image:
    return Image.new("RGB", (2, 2), (255, 255, 255))


def build_glb(parts: list[Part], defaults: dict, screen: int | None, out: Path) -> Path:
    """Write the GLB to `out` (atomically). The screen part gets UVs and placeholder base/emissive textures so the
    browser can swap in generated screen content."""
    scene = trimesh.Scene()
    for i, p in enumerate(parts):
        d = defaults["parts"][p.name]
        v = np.c_[p.vertices[:, 0], p.vertices[:, 2], -p.vertices[:, 1]]  # Z-up -> glTF Y-up
        kw = {}
        if i == screen:
            kw = {"baseColorTexture": _white(), "emissiveTexture": _white(), "emissiveFactor": [0, 0, 0]}
        material = trimesh.visual.material.PBRMaterial(
            name=f"p{i}", baseColorFactor=[*d["base_color"], 1.0], metallicFactor=d["metallic"],
            roughnessFactor=d["roughness"], doubleSided=True, **kw)
        visual = trimesh.visual.TextureVisuals(uv=planar_uv(p) if i == screen else None, material=material)
        mesh = trimesh.Trimesh(v, p.triangles, vertex_normals=vertex_normals(v, p.triangles), process=False,
                               visual=visual)
        scene.add_geometry(mesh, node_name=p.name)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(scene.export(file_type="glb", include_normals=True))
    tmp.replace(out)
    return out
