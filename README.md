# Again

Read data nvidia/simready-assets. Playing with [again](https://www.youtube.com/watch?v=S2IOrt-n2fc).

* **About Again**

> Again·YUI
>
> HOLIDAYS IN THE SUN


## Reference

- [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets)

## Install

```bash
uv sync
```

Dependencies are declared in `pyproject.toml`; add one with `uv add <package>`.

## Usage

Run all commands from the project root, inside the uv venv:

```bash
source .venv/bin/activate
```

```bash
python -m src.simready files IV_pole_a01          # list the asset's files and sizes (no download)
python -m src.simready show IV_pole_a01           # download the required files and open a matplotlib window
python -m src.simready show IV_pole_a01 out.png   # save as an image instead
python -m src.simready view IV_pole_a01           # polyscope interactive window (GPU, original resolution)
python -m src.simready view IV_pole_a01 out.png   # no window, save a polyscope screenshot only
```

`view` window controls: left-drag rotates, right-drag pans, scroll zooms; the left panel toggles each part and changes its color.

## Local catalog (SQLite)

```bash
python -m src.catalog sync                        # fetch the full asset list into data/catalog.db (~1-2 min)
python -m src.catalog search pole                 # offline query; * marks assets already downloaded
python -m src.catalog search --sort download --limit 20
```

Per asset it stores the file count, the total size, the size `download_asset()` actually downloads, and whether textures/thumbnail exist.

## Gradio web viewer

```bash
python -m src.app              # opens http://127.0.0.1:7860
python -m src.app --help       # --host, --port, --share
```

- Pick an asset already downloaded under `data/` from the dropdown to view it in 3D in the browser, with the official thumbnail, size, license and parts table
- Choose a decimation level (original / high / medium / low); lower it when there are too many triangles and the browser lags
- "Download new asset" at the bottom left downloads by name (geometry only, no textures)
- USD is converted to GLB and cached at `data/<asset>/preview_*.glb`, so reopening an asset does not convert it again

Downloaded files live in `data/<asset>/` and are not downloaded again on repeated runs.

## Structure of each asset

```
<asset>/
  .metadata/com.nvidia.simready.root_usds.json   # path of the entry USD
  simready_usd/
    sm_<asset>_01.usd          # entry point
    payloads/geometries.usd    # geometry (the main size, about 20-70 MB)
    payloads/*.usda            # materials, physics (PhysX / MuJoCo / Newton)
    Textures/*.png             # textures (very large, not downloaded by default)
    source_assets/             # duplicate of the textures + MDL (not downloaded)
```

## Use in your own code

```python
from src.simready import download_asset, root_usd, load_meshes, plot_asset

d = download_asset("appliance_dishwasher_a01")   # textures are fetched only with textures=True
meshes = load_meshes(root_usd(d))                # [{"name", "vertices"(N,3) in meters, "faces"(M,3)}]
plot_asset(d, meshes)
```
