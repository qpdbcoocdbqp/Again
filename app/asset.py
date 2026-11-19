"""Load an asset as parts (one per GeomSubset), in meters, Z-up, plus screen-part detection."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

from src.simready import CACHE_DIR, root_usd, triangulate


@dataclass
class Part:
    name: str                     # subset name
    material: str | None          # bound material name
    vertices: np.ndarray          # (N, 3) world coordinates, meters, Z-up
    triangles: np.ndarray         # (M, 3) indices into vertices
    faces: int = 0                # polygon face count in the USD

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(0), self.vertices.max(0)


def load_parts(asset: str) -> list[Part]:
    """One Part per materialBind GeomSubset of every mesh (faces no subset claims form an extra part; a mesh without
    subsets is a single part). Names are made unique across meshes."""
    usd = root_usd(CACHE_DIR / asset)
    stage = Usd.Stage.Open(str(usd))
    meters = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"{asset}: only Z-up stages are supported")
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    parts: list[Part] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [])
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not pts or not len(counts):
            continue
        m = np.array(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
        p = np.asarray(pts, dtype=np.float64)
        world = (np.c_[p, np.ones(len(p))] @ m)[:, :3] * meters
        starts = np.cumsum(counts) - counts

        # Material partition: GeomSubsets of the "materialBind" family. Overlapping faces go to the first subset;
        # faces no subset claims become an extra "unassigned" part bound to the mesh's own material.
        claimed = np.zeros(len(counts), dtype=bool)
        groups = []
        for sub in UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, "materialBind"):
            ids = sub.GetIndicesAttr().Get() or sub.GetIndicesAttr().Get(Usd.TimeCode.EarliestTime())
            faces = np.unique(np.asarray(ids if ids is not None else [], dtype=np.int64))
            faces = faces[~claimed[faces]]
            claimed[faces] = True
            groups.append((sub.GetPrim().GetName(), faces, UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]))
        rest = np.flatnonzero(~claimed)
        if len(rest):
            name = "unassigned" if groups else prim.GetName()
            groups.append((name, rest, UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]))

        for name, faces, mat in groups:
            if not len(faces):
                continue
            lens = counts[faces]
            # gather every polygon's vertex indices at once: start offset of each face + 0..len-1
            pos = np.repeat(starts[faces], lens) + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
            used, inv = np.unique(idx[pos], return_inverse=True)
            parts.append(Part(
                name=name,
                material=mat.GetPrim().GetName() if mat else None,
                vertices=world[used],
                triangles=triangulate(lens, inv.ravel()),
                faces=len(faces),
            ))
    if not parts:
        raise ValueError(f"{asset}: no renderable mesh found")
    return _unique_names(parts)


def _unique_names(parts: list[Part]) -> list[Part]:
    """Subset names repeat across meshes; recipes key parts by name, so make every name unique (name, name~2, ...)."""
    seen: dict[str, int] = {}
    for p in parts:
        seen[p.name] = seen.get(p.name, 0) + 1
        if seen[p.name] > 1:
            p.name = f"{p.name}~{seen[p.name]}"
    return parts


def size_of(parts: list[Part]) -> np.ndarray:
    """Overall (x, y, z) extent in meters."""
    lo = np.min([p.bbox[0] for p in parts], axis=0)
    hi = np.max([p.bbox[1] for p in parts], axis=0)
    return hi - lo


def find_screen(parts: list[Part]) -> int | None:
    """Index of the part that looks like a flat panel: very thin along the object's thinnest axis and covering
    most of the other two axes (the largest such part). None if nothing matches."""
    size = size_of(parts)
    t = int(np.argmin(size))
    a, b = [i for i in range(3) if i != t]
    best, best_area = None, 0.0
    for i, p in enumerate(parts):
        lo, hi = p.bbox
        ext = hi - lo
        if ext[t] > 0.1 * size[t] or ext[a] < 0.85 * size[a] or ext[b] < 0.85 * size[b]:
            continue
        if ext[a] * ext[b] > best_area:
            best, best_area = i, ext[a] * ext[b]
    return best


def planar_uv(p: Part) -> np.ndarray:
    """UV for a flat part: u along the lower-index in-plane axis, v along the higher one (Z unless the part is
    horizontal). Spans 0..1; trimesh flips V on export, so v=1 is the top. The picture reads left-to-right when
    viewed from the +thin-axis side (the front of TV_a01); from the back it is mirrored."""
    lo, hi = p.bbox
    t = int(np.argmin(hi - lo))
    a, b = (i for i in range(3) if i != t)
    rng = np.maximum(hi - lo, 1e-9)
    return np.stack([(p.vertices[:, a] - lo[a]) / rng[a], (p.vertices[:, b] - lo[b]) / rng[b]], axis=1)


def asset_available(asset: str) -> bool:
    return (Path(CACHE_DIR) / asset / ".metadata/com.nvidia.simready.root_usds.json").exists()
