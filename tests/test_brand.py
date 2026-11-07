import json
import re
from pathlib import Path

import numpy as np
import pytest

import kinescript as k
from kinescript import brand as B
from kinescript.components import THEME
from kinescript.docs import reference
from kinescript.render import Renderer
from kinescript.verify import check

DARK = {"bg": "#101820", "surface": "#203040", "text": "#fafafa", "muted": "#a0a8b0", "accent": "#00a3e0"}
LIGHT = {"bg": "#fafafa", "text": "#101820", "accent": "#0077a8"}


def make_brand(folder: Path, **parts) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    spec = {"kinescriptBrand": 1, "id": "test", "palettes": {"dark": DARK, "light": LIGHT}, "default": "dark"}
    spec.update(parts)
    (folder / "brand.json").write_text(json.dumps(spec), encoding="utf-8")
    return folder


DOT = {"id": "dot", "type": "circle", "r": 10, "x": "50%", "y": "50%", "fill": "$accent"}


def scene_at(folder: Path, layers=(DOT,), **top) -> Path:
    spec = {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 1}, "layers": list(layers), **top}
    p = folder / "scene.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def test_scene_without_brand_keeps_the_builtin_theme_and_background(tmp_path):
    s = k.load(str(scene_at(tmp_path)))
    assert s.brand is None
    assert s.theme == THEME
    assert s.bg == "#0a0c10"


def test_brand_palette_fills_roles_aliases_and_background(tmp_path):
    make_brand(tmp_path / "brand")
    s = k.load(str(scene_at(tmp_path, brand="brand")))
    assert s.theme["accent"] == "#00a3e0"
    assert s.theme["paper"] == s.theme["text"] == "#fafafa"
    assert s.theme["ink"] == "#101820"
    assert s.theme["accent2"] == "#00a3e0"  # falls back to accent
    assert s.bg == "#101820"


def test_variant_picks_the_light_palette(tmp_path):
    make_brand(tmp_path / "brand")
    s = k.load(str(scene_at(tmp_path, brand="brand", variant="light")))
    assert s.theme["bg"] == "#fafafa"
    assert s.theme["variant"] == "light"


def test_unknown_variant_is_an_error_naming_the_palettes(tmp_path):
    make_brand(tmp_path / "brand")
    with pytest.raises(k.SceneError, match="dark, light"):
        k.load(str(scene_at(tmp_path, brand="brand", variant="sepia")))


def test_scene_theme_overrides_an_unlocked_brand(tmp_path):
    make_brand(tmp_path / "brand")
    s = k.load(str(scene_at(tmp_path, brand="brand", theme={"accent": "#ff0000"})))
    assert s.theme["accent"] == "#ff0000"


def test_scene_theme_cannot_change_a_locked_part(tmp_path):
    make_brand(tmp_path / "brand", lock=["palettes"])
    with pytest.raises(k.SceneError, match="locks"):
        k.load(str(scene_at(tmp_path, brand="brand", theme={"paper": "#ff0000"})))


def test_brand_layer_values_resolve_by_dotted_name(tmp_path):
    make_brand(tmp_path / "brand", charts={"series": ["#111111", "#222222"], "highlight": "#333333"})
    s = k.load(str(scene_at(tmp_path, [{"id": "r", "type": "rect", "w": 10, "h": 10, "fill": "$charts.highlight"}],
                            brand="brand")))
    assert s.theme["charts.series"] == ["#111111", "#222222"]
    assert s.by_id["r"].raw["fill"] == "#333333"


def test_brand_file_paths_cannot_leave_the_folder(tmp_path):
    make_brand(tmp_path / "brand", fonts={"body": {"family": "X", "files": [{"file": "../x.ttf"}], "license": "OFL"}})
    codes = {i["code"] for i in B.validate(tmp_path / "brand")}
    assert "brand-path" in codes
    with pytest.raises(k.SceneError, match="out of the brand folder"):
        k.load(str(scene_at(tmp_path, brand="brand")))


def test_absolute_brand_file_path_is_refused(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        B.inside(tmp_path, str(tmp_path / "a.ttf"))


def test_palette_without_a_required_role_is_an_error(tmp_path):
    make_brand(tmp_path / "brand", palettes={"dark": {"bg": "#000000", "text": "#ffffff"}})
    errors = [i for i in B.validate(tmp_path / "brand") if i["level"] == "error"]
    assert any("no 'accent'" in i["message"] for i in errors)


def test_newer_brand_version_is_refused(tmp_path):
    make_brand(tmp_path / "brand", kinescriptBrand=B.VERSION + 1)
    assert any(i["code"] == "brand-version" and i["level"] == "error" for i in B.validate(tmp_path / "brand"))


def test_font_files_without_a_license_warn(tmp_path):
    folder = make_brand(tmp_path / "brand", fonts={"body": {"family": "X", "files": [{"file": "fonts/x.ttf"}]}})
    (folder / "fonts").mkdir()
    (folder / "fonts" / "x.ttf").write_bytes(b"\0\1\0\0")
    assert [i["code"] for i in B.validate(folder)] == ["brand-font-license"]


def test_digest_changes_when_a_named_file_changes(tmp_path):
    folder = make_brand(tmp_path / "brand", logos={"mark": {"file": "logos/m.png"}})
    (folder / "logos").mkdir()
    (folder / "logos" / "m.png").write_bytes(b"one")
    before = B.digest(folder)
    (folder / "logos" / "m.png").write_bytes(b"two")
    assert B.digest(folder) != before


def test_logo_paths_resolve_to_the_brand_folder(tmp_path):
    folder = make_brand(tmp_path / "brand", logos={"mark": {"file": "logos/m.png", "alt": "M"}})
    (folder / "logos").mkdir()
    (folder / "logos" / "m.png").write_bytes(b"png")
    theme, _ = B.resolve(B.load(folder))
    assert Path(theme["logos.mark"]["file"]) == (folder / "logos" / "m.png").resolve()
    assert theme["logos.mark"]["alt"] == "M"


def test_variant_without_a_brand_warns(tmp_path):
    s = k.load(str(scene_at(tmp_path, variant="light")))
    assert any(i["code"] == "brand-variant" for i in s.issues)


def test_missing_brand_folder_is_a_load_error_in_check(tmp_path):
    r = check(str(scene_at(tmp_path, brand="nowhere")))
    assert not r["ok"]
    assert any("brand.json" in i["message"] for i in r["issues"])


def test_branded_frame_is_pure_function_of_index(tmp_path):
    make_brand(tmp_path / "brand")
    layers = [{"id": "t", "type": "text", "text": "Brand", "x": "50%", "y": "50%", "size": 40, "fill": "$accent",
               "split": "chars", "enter": {"name": "rise", "stagger": 0.05}},
              {"id": "c", "type": "circle", "r": 10, "x": "=160+cos(t*3)*100", "y": 40, "fill": "$text"}]
    path = scene_at(tmp_path, layers, brand="brand")
    r = Renderer(k.load(str(path)), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(20)]
    alone = Renderer(k.load(str(path)), backend="cpu").frame(17).copy()
    assert np.array_equal(sequential[17], alone)


def test_brand_example_in_docs_is_a_valid_brand(tmp_path):
    doc = reference()
    start = doc.index("BRANDS\n")
    text = doc[start:doc.index("\nCOMPONENTS", start)]
    m = re.search(r'\{"kinescriptBrand".*?"lock": \["palettes"\]\}', " ".join(text.split()))
    spec = json.loads(m.group(0))
    folder = tmp_path / "brand"
    folder.mkdir()
    (folder / "brand.json").write_text(json.dumps(spec), encoding="utf-8")
    assert B.validate(folder) == []
    r = check(str(scene_at(tmp_path, [{"id": "t", "type": "text", "text": "Hi", "x": "50%", "y": "50%",
                                       "fill": "$accent"}], brand="brand", variant="light")))
    assert r["ok"], r["issues"]
