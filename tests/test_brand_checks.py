from kinescript import brand as B
from kinescript.brandcheck import delta_e
from kinescript import color as C
from kinescript.verify import check
from test_brand import DARK, make_brand, scene_at
from test_brand_logos import LOGOS, branded

TEXT = {"id": "t", "type": "text", "text": "Hello brand", "x": "50%", "y": "50%", "size": 20, "fill": "$text",
        "dur": 1}


def _codes(r) -> list[str]:
    return [i["code"] for i in r["issues"]]


def test_low_contrast_palette_pair_warns_with_the_ratio(tmp_path):
    folder = make_brand(tmp_path / "brand", palettes={"dark": {**DARK, "text": "#303840"}})
    found = [i for i in B.validate(folder) if i["code"] == "brand-contrast"]
    assert found and "text #303840 on bg #101820" in found[0]["message"]


def test_chart_series_colour_too_close_to_the_background_warns(tmp_path):
    folder = make_brand(tmp_path / "brand", charts={"series": ["accent", "#181f28"]})
    found = [i for i in B.validate(folder) if i["code"] == "brand-chart-contrast"]
    assert len(found) == 1 and "series[1]" in found[0]["message"]


def test_brand_warnings_reach_check(tmp_path):
    make_brand(tmp_path / "brand", charts={"series": ["#181f28"]})
    assert "brand-chart-contrast" in _codes(check(str(scene_at(tmp_path, [TEXT], brand="brand"))))


def test_layer_inside_a_logo_clear_space_warns(tmp_path):
    branded(tmp_path, logos={**LOGOS, "clearSpace": 0.5})
    near = {"id": "near", "type": "rect", "x": 280, "y": 160, "w": 10, "h": 10, "fill": "$accent", "dur": 1}
    r = check(str(scene_at(tmp_path, [{"use": "logo", "id": "l", "dur": 1}, near], brand="brand")))
    assert "logo-clear-space" in _codes(r)


def test_layer_clear_of_the_logo_and_backgrounds_do_not_warn(tmp_path):
    branded(tmp_path, logos={**LOGOS, "clearSpace": 0.5})
    far = {"id": "far", "type": "rect", "x": 40, "y": 40, "w": 10, "h": 10, "fill": "$accent", "dur": 1}
    back = {"id": "back", "type": "rect", "x": 0, "y": 0, "w": 320, "h": 180, "anchor": "topleft",
            "fill": "$surface", "dur": 1}
    r = check(str(scene_at(tmp_path, [back, {"use": "logo", "id": "l", "dur": 1}, far], brand="brand")))
    assert "logo-clear-space" not in _codes(r)


def test_off_brand_colour_and_font_only_in_strict_mode(tmp_path):
    make_brand(tmp_path / "brand")
    odd = dict(TEXT, fill="#ff00ff", font="Arial")
    path = str(scene_at(tmp_path, [odd], brand="brand"))
    assert not {"off-brand-colour", "off-brand-font"} & set(_codes(check(path)))
    strict = check(path, brand_strict=True)
    assert {"off-brand-colour", "off-brand-font"} <= set(_codes(strict))
    colour = next(i for i in strict["issues"] if i["code"] == "off-brand-colour")
    assert "nearest" in colour["message"]


def test_brand_colours_pass_strict_mode(tmp_path):
    make_brand(tmp_path / "brand")
    r = check(str(scene_at(tmp_path, [TEXT], brand="brand")), brand_strict=True)
    assert not {"off-brand-colour", "off-brand-font"} & set(_codes(r))


def test_strict_mode_uses_the_brand_safe_area(tmp_path):
    make_brand(tmp_path / "brand", safeArea="20%")
    edge = dict(TEXT, x=90, y=90)
    path = str(scene_at(tmp_path, [edge], brand="brand"))
    assert "safe-area" not in _codes(check(path))
    assert "safe-area" in _codes(check(path, brand_strict=True))


def test_delta_e_is_zero_for_the_same_colour_and_large_for_opposites():
    assert delta_e(C.parse("#123456"), C.parse("#123456")) == 0
    assert delta_e(C.parse("#000000"), C.parse("#ffffff")) > 99


def test_example_brand_scene_has_no_brand_issues():
    from pathlib import Path
    r = check(str(Path(__file__).parents[1] / "examples" / "brand.json"), brand_strict=True)
    brand = {"brand-contrast", "brand-chart-contrast", "logo-clear-space", "logo-small", "off-brand-colour",
             "off-brand-font", "safe-area", "brand-font-license"}
    assert r["ok"] and not [i for i in r["issues"] if i["code"] in brand], r["issues"]
