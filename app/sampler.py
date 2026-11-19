"""Seeded randomization. Pure: the same (recipe config, defaults, seed) always gives the same result."""
import colorsys

import numpy as np

from app.recipe import hsv_to_rgb


def _jitter_color(rng: np.random.Generator, color: list[float], hue_j: float, val_j: float) -> list[float]:
    h, s, v = colorsys.rgb_to_hsv(*color)
    h += rng.uniform(-hue_j, hue_j)
    v = float(np.clip(v + rng.uniform(-val_j, val_j) * max(v, 0.2), 0.0, 1.0))
    return hsv_to_rgb(h, s, v)


def _uniform(rng: np.random.Generator, r: list[float]) -> float:
    return round(float(rng.uniform(r[0], r[1])), 4)


def _hex(c: list[float]) -> str:
    return "#" + "".join(f"{round(v * 255):02x}" for v in c)


def randomize(recipe: dict, defaults: dict, seed: int) -> dict:
    """Copy of `recipe` with materials, screen, scene and pose sampled from recipe["randomize"].

    Part colors jitter around `defaults` (not the current values), so the result depends on the seed only.
    The screen part keeps its material and only gets new content. Visibility and part names are kept."""
    cfg = recipe["randomize"]
    rng = np.random.default_rng(seed)
    out = {**recipe, "seed": seed, "parts": {}, "screen": dict(recipe["screen"]),
           "scene": dict(recipe["scene"]), "pose": dict(recipe["pose"])}
    screen = recipe["screen"].get("part")

    for name, cur in recipe["parts"].items():
        base = defaults["parts"].get(name, cur)
        if name == screen:
            out["parts"][name] = dict(base, visible=cur["visible"])
            continue
        out["parts"][name] = {
            "base_color": _jitter_color(rng, base["base_color"], cfg["hue_jitter"], cfg["value_jitter"]),
            "roughness": _uniform(rng, cfg["roughness"]),
            "metallic": _uniform(rng, cfg["metallic"]),
            "visible": cur["visible"],
        }

    if screen:
        out["screen"]["content"] = str(rng.choice(cfg["screen_contents"]))
        out["screen"]["color"] = hsv_to_rgb(rng.uniform(), rng.uniform(0.4, 1.0), rng.uniform(0.5, 1.0))
    out["scene"]["exposure"] = _uniform(rng, cfg["exposure"])
    if cfg["background"]:
        h = rng.uniform()
        out["scene"]["background"] = {"type": "gradient", "colors": [
            _hex(hsv_to_rgb(h, rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.3))),
            _hex(hsv_to_rgb(h + rng.uniform(-0.1, 0.1), rng.uniform(0.1, 0.5), rng.uniform(0.3, 0.6)))]}
    for k in ("yaw", "pitch", "roll"):
        out["pose"][k] = _uniform(rng, cfg[k])
    return out
