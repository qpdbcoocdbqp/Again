"""Recipe: schema, defaults, validation, YAML. A recipe is a plain dict so it round-trips to JSON (browser) and YAML."""
import colorsys
import copy
import re

import yaml

SCREEN_CONTENTS = ("off", "solid", "color_bars", "noise")
ENVIRONMENTS = ("neutral", "legacy")
BACKGROUNDS = ("solid", "gradient")
RANGE_LIMITS = {"roughness": (0, 1), "metallic": (0, 1), "exposure": (0.01, 5),
                "yaw": (-360, 360), "pitch": (-360, 360), "roll": (-360, 360)}


def default_material(material: str | None) -> dict:
    """Initial base_color (sRGB 0-1) / roughness / metallic guessed from the material name, because the asset keeps
    these values in textures only. e.g. m_opaque__metal__aluminiumBrushed_a01."""
    n = (material or "").lower()
    kind = n.split("__")[1] if n.count("__") >= 2 else ""
    if "white" in n:
        color = [0.85, 0.85, 0.82]
    elif "aluminium" in n:
        color = [0.77, 0.78, 0.78]
    elif "steel" in n:
        color = [0.6, 0.6, 0.62]
    elif "black" in n:
        color = [0.03, 0.03, 0.03]
    else:
        color = [0.5, 0.5, 0.5]
    roughness = 0.5
    if "glossy" in n:
        roughness = 0.12
    elif "brushed" in n:
        roughness = 0.4
    elif "rough" in n:
        roughness = 0.7
    metallic = 1.0 if kind == "metal" and "painted" not in n else 0.0
    return {"base_color": color, "roughness": roughness, "metallic": metallic, "visible": True}


def default_recipe(asset: str, parts: list[tuple[str, str | None]], screen: str | None) -> dict:
    """parts: [(subset name, material name)]; screen: name of the screen subset or None."""
    return {
        "name": f"{asset.lower()}_demo",
        "asset": asset,
        "seed": 42,
        "parts": {name: default_material(mat) for name, mat in parts},
        "screen": {"part": screen, "content": "color_bars", "color": [0.1, 0.4, 0.9]},
        "scene": {"background": {"type": "gradient", "colors": ["#1f2430", "#4a5568"]},
                  "ground": True, "exposure": 1.0, "environment": "neutral"},
        "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "randomize": {
            "hue_jitter": 0.08, "value_jitter": 0.2,
            "roughness": [0.1, 0.9], "metallic": [0.0, 1.0], "exposure": [0.6, 1.6],
            "yaw": [-180.0, 180.0], "pitch": [-10.0, 10.0], "roll": [-5.0, 5.0],
            "screen_contents": list(SCREEN_CONTENTS), "background": True,
        },
    }


def hsv_to_rgb(h: float, s: float, v: float) -> list[float]:
    return [round(c, 4) for c in colorsys.hsv_to_rgb(h % 1.0, s, v)]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_color(c) -> bool:
    return isinstance(c, list) and len(c) == 3 and all(_is_num(v) and 0 <= v <= 1 for v in c)


def _is_hex(c) -> bool:
    return isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) is not None


def _section(parent: dict, key: str, errs: list[str], path: str | None = None) -> dict:
    """parent[key] as a dict; records an error (and returns {}) when it is missing or not a mapping."""
    v = parent.get(key)
    if not isinstance(v, dict):
        errs.append(f"{path or key} must be a mapping")
        return {}
    return v


def validate(recipe: dict, part_names: list[str], screen_part: str | None = None, asset: str | None = None) -> list[str]:
    """All problems with a recipe as readable messages (empty list = valid). Never raises on wrongly typed input.

    screen_part: the only part that can show screen content (the one with UVs in the GLB), if known.
    asset: the only asset name the recipe may refer to, if known."""
    errs: list[str] = []
    if not isinstance(recipe, dict):
        return ["the recipe must be a mapping"]
    if asset is not None and recipe.get("asset") != asset:
        errs.append(f"asset must be {asset!r}")
    seed = recipe.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        errs.append("seed must be an integer in 0..4294967295")

    parts = _section(recipe, "parts", errs)
    for name, p in parts.items():
        if name not in part_names:
            errs.append(f"parts.{name}: no such part; available: {', '.join(part_names)}")
            continue
        if not isinstance(p, dict):
            errs.append(f"parts.{name} must be a mapping")
            continue
        if not _is_color(p.get("base_color")):
            errs.append(f"parts.{name}.base_color must be three numbers in 0..1")
        for k in ("roughness", "metallic"):
            if not (_is_num(p.get(k)) and 0 <= p[k] <= 1):
                errs.append(f"parts.{name}.{k} must be a number in 0..1")
        if not isinstance(p.get("visible"), bool):
            errs.append(f"parts.{name}.visible must be true or false")

    scr = _section(recipe, "screen", errs)
    if scr.get("part") is not None and scr["part"] not in part_names:
        errs.append(f"screen.part {scr['part']!r}: no such part")
    elif screen_part is not None and scr.get("part") != screen_part:
        errs.append(f"screen.part must be {screen_part!r} (the only part prepared to show screen content)")
    if scr.get("content") not in SCREEN_CONTENTS:
        errs.append(f"screen.content must be one of {', '.join(SCREEN_CONTENTS)}")
    if not _is_color(scr.get("color")):
        errs.append("screen.color must be three numbers in 0..1")

    sc = _section(recipe, "scene", errs)
    bg = _section(sc, "background", errs, "scene.background")
    if bg.get("type") not in BACKGROUNDS:
        errs.append(f"scene.background.type must be one of {', '.join(BACKGROUNDS)}")
    if not (isinstance(bg.get("colors"), list) and len(bg["colors"]) == 2 and all(_is_hex(c) for c in bg["colors"])):
        errs.append("scene.background.colors must be two #rrggbb strings")
    if not isinstance(sc.get("ground"), bool):
        errs.append("scene.ground must be true or false")
    if not (_is_num(sc.get("exposure")) and 0 < sc["exposure"] <= 5):
        errs.append("scene.exposure must be a number in (0, 5]")
    if sc.get("environment") not in ENVIRONMENTS:
        errs.append(f"scene.environment must be one of {', '.join(ENVIRONMENTS)}")

    pose = _section(recipe, "pose", errs)
    for k in ("yaw", "pitch", "roll"):
        if not (_is_num(pose.get(k)) and -360 <= pose[k] <= 360):
            errs.append(f"pose.{k} must be a number in -360..360")

    rnd = _section(recipe, "randomize", errs)
    for k in ("hue_jitter", "value_jitter"):
        if not (_is_num(rnd.get(k)) and 0 <= rnd[k] <= 1):
            errs.append(f"randomize.{k} must be a number in 0..1")
    for k, (lo, hi) in RANGE_LIMITS.items():
        r = rnd.get(k)
        if not (isinstance(r, list) and len(r) == 2 and all(_is_num(v) for v in r)):
            errs.append(f"randomize.{k} must be [min, max]")
        elif r[0] > r[1]:
            errs.append(f"randomize.{k}: min {r[0]} is greater than max {r[1]}")
        elif r[0] < lo or r[1] > hi:
            errs.append(f"randomize.{k} must stay within {lo}..{hi}")
    contents = rnd.get("screen_contents")
    if not (isinstance(contents, list) and contents and all(c in SCREEN_CONTENTS for c in contents)):
        errs.append(f"randomize.screen_contents must be a non-empty list from {', '.join(SCREEN_CONTENTS)}")
    if not isinstance(rnd.get("background"), bool):
        errs.append("randomize.background must be true or false")
    return errs


def merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` into a copy of `base` (dicts merge, everything else is replaced)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def to_yaml(recipe: dict) -> str:
    return yaml.safe_dump(_round(recipe), sort_keys=False, allow_unicode=True, default_flow_style=None)


def from_yaml(text: str, defaults: dict) -> dict:
    """Parse YAML and fill missing keys from `defaults`. Raises ValueError with a readable message."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("the recipe must be a YAML mapping")  # noqa: TRY004 - callers catch ValueError only
    for key, default in defaults.items():  # sections must keep the type of the default (mapping stays mapping)
        if key in data and isinstance(default, dict) and not isinstance(data[key], dict):
            raise ValueError(f"{key} must be a mapping")
    given = data.get("parts") or {}
    for name, part in given.items():
        if part is not None and not isinstance(part, dict):
            raise ValueError(f"parts.{name} must be a mapping")
    full = merge(defaults, {k: v for k, v in data.items() if k != "parts"})
    # a part given partially is completed from that part's default; unknown names are kept so validate() reports them
    full["parts"] = {name: merge(defaults["parts"].get(name, default_material(None)), p or {})
                     for name, p in {**defaults["parts"], **given}.items()}
    return full
