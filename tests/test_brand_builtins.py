import shutil
from pathlib import Path

import numpy as np
import pytest

import kinescript as k
from kinescript.components import CHART_COLORS
from kinescript.render import FONT_DIR, Renderer
from test_brand import make_brand, scene_at


def _load(tmp_path, layers, **top):
    return k.load(str(scene_at(tmp_path, layers, **top)))


def _raw(s, lid):
    return s.by_id[lid].raw


def test_text_style_fills_a_layer_without_a_brand(tmp_path):
    s = _load(tmp_path, [{"id": "h", "type": "text", "text": "Hi", "textStyle": "heading"},
                         {"id": "b", "type": "text", "text": "Hi", "textStyle": "heading", "size": 50}])
    assert (_raw(s, "h")["size"], _raw(s, "h")["weight"], _raw(s, "h")["font"]) == (72, 700, "Space Grotesk")
    assert _raw(s, "b")["size"] == 50


def test_unknown_text_style_is_an_error_listing_the_styles(tmp_path):
    with pytest.raises(k.SceneError, match="caption, heading, label, title"):
        _load(tmp_path, [{"id": "t", "type": "text", "text": "Hi", "textStyle": "huge"}])


def test_brand_text_style_changes_the_built_in_one_and_adds_its_own(tmp_path):
    make_brand(tmp_path / "brand", fonts={"display": {"family": "IBM Plex Mono"}},
               text={"title": {"size": 96, "tracking": 2, "case": "upper"}, "quote": {"font": "body", "italic": True}})
    s = _load(tmp_path, [{"id": "t", "type": "text", "text": "Hi", "textStyle": "title"},
                         {"id": "q", "type": "text", "text": "Hi", "textStyle": "quote"}], brand="brand")
    t = _raw(s, "t")
    assert (t["size"], t["weight"], t["font"], t["letterSpacing"], t["case"]) == (96, 800, "IBM Plex Mono", 2, "upper")
    assert _raw(s, "q")["italic"] is True


def test_brand_font_files_draw_text_in_that_face(tmp_path):
    folder = make_brand(tmp_path / "brand", fonts={"body": {"family": "Acme Mono", "license": "OFL", "files": [
        {"file": "fonts/a-bold.ttf", "weight": 700}, {"file": "fonts/a-regular.ttf", "weight": 400}]}})
    (folder / "fonts").mkdir()
    shutil.copy(FONT_DIR / "IBMPlexMono-Bold.ttf", folder / "fonts" / "a-bold.ttf")
    shutil.copy(FONT_DIR / "IBMPlexMono-Regular.ttf", folder / "fonts" / "a-regular.ttf")

    def frame(font, sub):
        layer = {"id": "t", "type": "text", "text": "Wide words", "x": "50%", "y": "50%", "size": 40, "weight": 700,
                 "fill": "#ffffff"}
        if font:
            layer["font"] = font
        d = tmp_path / sub
        d.mkdir()
        if (folder / "..").resolve() != d:
            shutil.copytree(folder, d / "brand")
        return Renderer(_load(d, [layer], brand="brand"), backend="cpu").frame(0).copy()

    assert np.array_equal(frame(None, "brand-face"), frame("IBM Plex Mono", "plex"))
    assert not np.array_equal(frame(None, "again"), frame("Space Grotesk", "grotesk"))
    s = _load(tmp_path, [{"id": "t", "type": "text", "text": "Hi"}], brand="brand")
    assert not [i for i in s.issues if i["code"] == "font-missing"]


def test_charts_take_the_brand_series_highlight_and_axis(tmp_path):
    make_brand(tmp_path / "brand", charts={"series": ["accent", "#123456"], "highlight": "#abcdef",
                                           "axis": {"color": "text", "width": 5}})
    s = _load(tmp_path, [{"use": "pieChart", "id": "pie", "data": [1, 1, 1]},
                         {"use": "barChart", "id": "bar", "data": [1, 2], "highlight": 2},
                         {"use": "lineChart", "id": "line", "data": [1, 2]}], brand="brand")
    assert [_raw(s, f"pie-slice{i}")["stroke"] for i in (1, 2, 3)] == ["#00a3e0", "#123456", "#00a3e0"]
    assert _raw(s, "bar-bar2")["fill"] == "#abcdef"
    assert _raw(s, "line-line")["stroke"] == "#abcdef"
    assert (_raw(s, "bar-axis")["stroke"], _raw(s, "bar-axis")["strokeWidth"]) == ("#fafafa", 5)


def test_charts_without_a_brand_keep_their_colours(tmp_path):
    s = _load(tmp_path, [{"use": "pieChart", "id": "pie", "data": [1, 1]}])
    assert [_raw(s, f"pie-slice{i}")["stroke"] for i in (1, 2)] == CHART_COLORS[:2]


def test_component_styles_set_looks_the_instance_leaves_out(tmp_path):
    make_brand(tmp_path / "brand", components={"lowerThird": {"plate": "surface", "accent": "#ff00ff"}})
    s = _load(tmp_path, [{"use": "lowerThird", "id": "a", "title": "A"},
                         {"use": "lowerThird", "id": "b", "title": "B", "accent": "#00ff00"}], brand="brand")
    assert _raw(s, "a-title")["box"]["fill"] == "#203040"
    assert _raw(s, "a-bar")["fill"] == "#ff00ff"
    assert _raw(s, "b-bar")["fill"] == "#00ff00"


def test_branded_lower_third_plate_follows_the_background(tmp_path):
    make_brand(tmp_path / "brand")
    s = _load(tmp_path, [{"use": "lowerThird", "id": "a", "title": "A"}], brand="brand")
    assert _raw(s, "a-title")["box"]["fill"] == "#101820d0"


def test_built_in_text_parts_take_the_brand_faces_but_keep_their_weights(tmp_path):
    make_brand(tmp_path / "brand", fonts={"display": {"family": "IBM Plex Mono"}, "body": {"family": "Space Grotesk"}})
    s = _load(tmp_path, [{"use": "titleCard", "id": "c", "title": "T", "subtitle": "S", "kicker": "K"}],
              brand="brand")
    assert (_raw(s, "c-title")["font"], _raw(s, "c-title")["weight"]) == ("IBM Plex Mono", 800)
    assert _raw(s, "c-sub")["font"] == "Space Grotesk"


def test_part_text_style_in_components_sets_weight_but_not_size(tmp_path):
    make_brand(tmp_path / "brand", components={"titleCard": {"titleText": "caption"}})
    s = _load(tmp_path, [{"use": "titleCard", "id": "c", "title": "T"}], brand="brand")
    assert (_raw(s, "c-title")["weight"], _raw(s, "c-title")["size"]) == (500, 128)


def test_motion_values_are_opt_in_by_name(tmp_path):
    make_brand(tmp_path / "brand", motion={"enter": "pop"})
    s = _load(tmp_path, [{"id": "a", "type": "circle", "r": 5, "enter": "$motion.enter"},
                         {"id": "b", "type": "circle", "r": 5}], brand="brand")
    assert _raw(s, "a")["enter"] == "pop"
    assert "enter" not in _raw(s, "b")


def test_motion_name_the_theme_does_not_set_is_still_an_error(tmp_path):
    with pytest.raises(k.SceneError, match="unknown motion"):
        _load(tmp_path, [{"id": "a", "type": "circle", "r": 5, "enter": "$motion.enter"}])
