import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "nvidia/simready-assets"
REPO_TYPE = "dataset"
CACHE_DIR = Path("data")

api = HfApi()


# ---------------------------------------------------------------- 1. Browse (no download)
def repo_files(path: str | None = None):
    """Yield every file in the repo (or under `path`) as an entry with .path and .size."""
    for e in api.list_repo_tree(REPO_ID, path_in_repo=path, repo_type=REPO_TYPE, recursive=True):
        if getattr(e, "size", None) is not None:
            yield e


def list_files(asset: str) -> list[tuple[str, int]]:
    """List all files of an asset with their sizes (bytes)."""
    return [(e.path, e.size) for e in repo_files(asset)]


# ---------------------------------------------------------------- 2. Selective download
def download_patterns(asset: str, textures: bool = False) -> tuple[list[str], list[str]]:
    """(allow, ignore) filename patterns used by download_asset (fnmatch; * also matches /)."""
    allow = [
        f"{asset}/.metadata/*.json",
        f"{asset}/com.nvidia.simready.packaging.json",
        f"{asset}/simready_usd/*.usd",   # also matches payloads/geometries.usd
        f"{asset}/simready_usd/*.usda",  # payloads/*.usda
        f"{asset}/simready_usd/.thumbs/*_thumbnail.png",
    ]
    if textures:
        allow.append(f"{asset}/simready_usd/Textures/*.png")
    return allow, [f"{asset}/simready_usd/source_assets/*"]


def download_asset(asset: str, textures: bool = False) -> Path:
    """Download only the files needed to read geometry; return the local folder."""
    allow, ignore = download_patterns(asset, textures)
    snapshot_download(
        REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=allow,
        ignore_patterns=ignore,
        local_dir=CACHE_DIR,
    )
    return CACHE_DIR / asset


def root_usd(asset_dir: Path) -> Path:
    """Find the entry USD from .metadata/com.nvidia.simready.root_usds.json."""
    meta = json.loads((asset_dir / ".metadata/com.nvidia.simready.root_usds.json").read_text())
    return asset_dir / meta["entries"][0]


# ---------------------------------------------------------------- 3. Read USD geometry
def load_meshes(usd_path: Path) -> list[dict]:
    """Return [{"name", "vertices"(N,3), "faces"(M,3)}, ...] in world coordinates, meters."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    meters = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up_axis = UsdGeom.GetStageUpAxis(stage)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    meshes = []
    # TraverseInstanceProxies: lets instanceable prims be traversed too
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        purpose = UsdGeom.Imageable(prim).ComputePurpose()
        if purpose in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):  # skip collision/helper geometry
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue

        # Transform to world coordinates (USD uses row vectors, right-multiplied: p' = p * M)
        m = np.array(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
        p = np.asarray(pts, dtype=np.float64)
        p = (np.c_[p, np.ones(len(p))] @ m)[:, :3] * meters
        if up_axis == UsdGeom.Tokens.y:  # normalize to Z-up, which suits matplotlib plotting
            p = np.c_[p[:, 0], -p[:, 2], p[:, 1]]

        meshes.append({
            "name": str(prim.GetPath()),
            "vertices": p,
            "faces": triangulate(np.asarray(counts), np.asarray(idx)),
        })
    return meshes


def bounds(meshes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box (lo, hi) of all parts."""
    v = np.vstack([m["vertices"] for m in meshes])
    return v.min(0), v.max(0)


def part_name(m: dict) -> str:
    return m["name"].rsplit("/", 1)[-1]


def part_color(i: int) -> tuple[float, float, float]:
    import matplotlib.pyplot as plt

    return plt.get_cmap("tab20")(i % 20)[:3]


def find_thumbnail(asset_dir: Path) -> Path | None:
    return next((asset_dir / "simready_usd/.thumbs").glob("*_thumbnail.png"), None)


def triangulate(counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Polygons -> triangles (fan triangulation, vectorized)."""
    if (counts == 3).all():
        return idx.reshape(-1, 3).astype(np.int64)
    starts = np.cumsum(counts) - counts
    ntri = np.clip(counts - 2, 0, None)
    face = np.repeat(np.arange(len(counts)), ntri)
    k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    s = starts[face]
    return np.stack([idx[s], idx[s + k + 1], idx[s + k + 2]], axis=1).astype(np.int64)


def decimate(v: np.ndarray, f: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation: merge vertices that fall in the same cell."""
    q = np.floor(v / cell).astype(np.int64)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    nv = np.zeros((len(cnt), 3))
    np.add.at(nv, inv, v)
    nv /= cnt[:, None]
    nf = inv[f]
    nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
    if len(nf):
        # Dedupe on the sorted key, but keep the original rows so the winding (and thus normals) is preserved
        _, keep = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)
        nf = nf[np.sort(keep)]
    return nv, nf


# ---------------------------------------------------------------- 4. Visualization
def plot_asset(asset_dir: Path, meshes: list[dict], resolution: int = 200, save: Path | None = None):
    """resolution: decimation grid cells along the longest edge."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    thumb = find_thumbnail(asset_dir)
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(asset_dir.name)

    # Left: official thumbnail
    ax0 = fig.add_subplot(1, 2, 1)
    if thumb:
        ax0.imshow(plt.imread(thumb))
    ax0.set_title("thumbnail")
    ax0.axis("off")

    # Right: decimated geometry with simple normal-based shading
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    lo, hi = bounds(meshes)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    cell = 2 * r / resolution
    light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
    total, shown = sum(len(m["faces"]) for m in meshes), 0
    for i, m in enumerate(meshes):
        dv, df = decimate(m["vertices"], m["faces"], cell)
        if len(df) == 0:
            continue
        tri = dv[df]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(n @ light)
        colors = np.asarray(part_color(i))[None, :] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, edgecolor="none"))
        shown += len(df)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    size = hi - lo
    ax.set_title(f"{len(meshes)} meshes, {total:,} tris (decimated to {shown:,})"
                 f"\nsize {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m")

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print(f"saved -> {save}")
    else:
        plt.show()


def view_asset(asset_dir: Path, meshes: list[dict], screenshot: Path | None = None):
    """Interactive polyscope window; with screenshot, save an image instead of opening a window."""
    import polyscope as ps

    ps.set_program_name(asset_dir.name)
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")

    used = set()
    for i, m in enumerate(meshes):
        name = part_name(m)
        if name in used:  # polyscope names must be unique
            name = f"{name}_{i}"
        used.add(name)
        ps.register_surface_mesh(name, m["vertices"], m["faces"], color=part_color(i), smooth_shade=True)

    # Initial camera: look down at the whole object from the front diagonal
    lo, hi = bounds(meshes)
    c, r = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    ps.look_at(tuple(c + r * np.array([1.2, -1.8, 1.0])), tuple(c))

    if screenshot:
        ps.screenshot(str(screenshot), transparent_bg=False)
        print(f"saved -> {screenshot}")
    else:
        ps.show()


# ---------------------------------------------------------------- CLI
def main(argv: list[str]):
    cmd = argv[0] if argv else ""
    if cmd == "files":
        files = list_files(argv[1])
        for path, size in files:
            print(f"{size / 1e6:9.2f} MB  {path}")
        print(f"\nTotal {sum(s for _, s in files) / 1e6:.1f} MB")
    elif cmd in ("show", "view"):
        asset = argv[1]
        save = Path(argv[2]) if len(argv) > 2 else None
        asset_dir = download_asset(asset)
        usd = root_usd(asset_dir)
        print(f"open {usd}")
        meshes = load_meshes(usd)
        for m in meshes:
            print(f"  {m['name']}: {len(m['vertices'])} verts, {len(m['faces'])} tris")
        if cmd == "show":
            plot_asset(asset_dir, meshes, save=save)
        else:
            view_asset(asset_dir, meshes, screenshot=save)
    else:
        print("usage: python -m src.simready {files ASSET | show ASSET [out.png] | view ASSET [out.png]}")


if __name__ == "__main__":
    main(sys.argv[1:])
