"""Checks for the SDG preview (need data/TV_a01 downloaded). Run: python -m tests.test_sdg  (also pytest-compatible)."""
import copy

from app import recipe as rc
from app.asset import find_screen, load_parts, size_of
from app.sampler import randomize

ASSET = "TV_a01"
PARTS = load_parts(ASSET)
NAMES = [p.name for p in PARTS]
SCREEN = find_screen(PARTS)
DEFAULTS = rc.default_recipe(ASSET, [(p.name, p.material) for p in PARTS], PARTS[SCREEN].name)


def test_parts_match_usd_subsets():
    assert len(PARTS) == 10
    assert sum(p.faces for p in PARTS) == 72160
    assert sum(len(p.triangles) for p in PARTS) == 136861  # same as the whole mesh in src.simready.load_meshes
    size = size_of(PARTS)
    assert abs(size[0] - 0.1062) < 1e-3 and abs(size[1] - 1.272) < 1e-3 and abs(size[2] - 0.7549) < 1e-3


def test_screen_is_the_thin_full_face_part():
    assert PARTS[SCREEN].name == "m_opaque__plastic__glossy_black_a01"


def test_default_recipe_is_valid():
    assert rc.validate(DEFAULTS, NAMES) == []


def test_same_seed_same_result_and_other_seed_differs():
    a, b, c = (randomize(DEFAULTS, DEFAULTS, s) for s in (7, 7, 8))
    assert a == b and a != c
    assert rc.validate(a, NAMES) == []


def test_randomize_depends_on_seed_not_on_current_values():
    edited = copy.deepcopy(DEFAULTS)
    edited["parts"][NAMES[1]]["base_color"] = [1.0, 0.0, 0.0]
    assert randomize(edited, DEFAULTS, 3) == randomize(DEFAULTS, DEFAULTS, 3)


def test_randomize_respects_ranges_and_keeps_screen_material():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["randomize"].update(roughness=[0.3, 0.4], metallic=[0.0, 0.0], exposure=[1.0, 1.0], yaw=[10.0, 10.0])
    for seed in range(20):
        r = randomize(cfg, DEFAULTS, seed)
        for name, p in r["parts"].items():
            if name == PARTS[SCREEN].name:
                assert p == DEFAULTS["parts"][name]
            else:
                assert 0.3 <= p["roughness"] <= 0.4 and p["metallic"] == 0.0
        assert r["scene"]["exposure"] == 1.0 and r["pose"]["yaw"] == 10.0


def test_validation_messages():
    bad = copy.deepcopy(DEFAULTS)
    bad["randomize"]["roughness"] = [0.9, 0.1]
    bad["screen"]["content"] = "banana"
    bad["parts"]["nope"] = copy.deepcopy(DEFAULTS["parts"][NAMES[0]])
    bad["parts"][NAMES[0]]["metallic"] = 2
    errs = "\n".join(rc.validate(bad, NAMES))
    for needle in ("min 0.9 is greater than max 0.1", "screen.content", "parts.nope: no such part", ".metallic"):
        assert needle in errs, needle


def test_yaml_round_trip_and_partial_yaml():
    r = randomize(DEFAULTS, DEFAULTS, 11)
    assert rc.from_yaml(rc.to_yaml(r), DEFAULTS) == rc.merge(r, {})  # rounding already applied by randomize
    partial = rc.from_yaml(f"seed: 5\nparts:\n  {NAMES[1]}:\n    roughness: 0.9\n", DEFAULTS)
    assert partial["seed"] == 5 and partial["parts"][NAMES[1]]["roughness"] == 0.9
    assert partial["parts"][NAMES[1]]["base_color"] == DEFAULTS["parts"][NAMES[1]]["base_color"]
    assert rc.validate(partial, NAMES) == []


def test_invalid_yaml_raises():
    for text in ("- a\n- b\n", "a: [1,\n"):
        try:
            rc.from_yaml(text, DEFAULTS)
        except ValueError:
            continue
        raise AssertionError(f"accepted {text!r}")


def test_wrongly_typed_input_gives_messages_not_crashes():
    for text, needle in (("scene: 5\n", "scene must be a mapping"), ("screen:\n", "screen must be a mapping"),
                         ("pose: null\n", "pose must be a mapping"),
                         (f"parts:\n  {NAMES[0]}: abc\n", "must be a mapping")):
        try:
            rc.from_yaml(text, DEFAULTS)
        except ValueError as e:
            assert needle in str(e), (text, str(e))
        else:
            raise AssertionError(f"accepted {text!r}")
    # types the YAML loader accepts but the recipe must still reject, without raising
    for path, value in ((("scene", "background"), 5), (("randomize",), []), (("parts",), []), (("seed",), -1),
                        (("seed",), 2**32), (("seed",), True), (("scene", "exposure"), "x")):
        r = copy.deepcopy(DEFAULTS)
        target = r
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
        assert rc.validate(r, NAMES), (path, value)
    assert rc.validate(["x"], NAMES) == ["the recipe must be a mapping"]


def test_asset_and_screen_part_are_pinned():
    other = copy.deepcopy(DEFAULTS)
    other["asset"] = "../x"
    assert any("asset must be" in e for e in rc.validate(other, NAMES, asset=ASSET))
    other = copy.deepcopy(DEFAULTS)
    other["screen"]["part"] = NAMES[1]
    assert any("only part prepared" in e for e in rc.validate(other, NAMES, screen_part=NAMES[SCREEN]))
    assert rc.validate(DEFAULTS, NAMES, screen_part=NAMES[SCREEN], asset=ASSET) == []


def test_duplicate_part_names_are_made_unique():
    from app.asset import Part, _unique_names
    mk = lambda n: Part(n, None, PARTS[0].vertices, PARTS[0].triangles)
    assert [p.name for p in _unique_names([mk("a"), mk("b"), mk("a"), mk("a")])] == ["a", "b", "a~2", "a~3"]


def test_sdg_app_server_functions():
    from app import sdg_app
    assert sdg_app.randomize({"recipe": DEFAULTS, "seed": -1})["errors"]
    ok = sdg_app.randomize({"recipe": DEFAULTS, "seed": 5})
    assert ok["errors"] == [] and ok["recipe"]["seed"] == 5
    assert sdg_app.load_yaml({"text": "asset: ../x\n"})["errors"]
    assert sdg_app.load_yaml({"text": "scene: 5\n"})["errors"] == ["scene must be a mapping"]
    assert sdg_app.dump_yaml(DEFAULTS)["yaml"].startswith("name:")
    try:
        sdg_app.init("../x")
    except ValueError:
        pass
    else:
        raise AssertionError("init accepted another asset")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} passed")
