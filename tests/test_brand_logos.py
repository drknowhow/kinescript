from pathlib import Path

import numpy as np
import pytest

import kinescript as k
from kinescript.render import Renderer
from test_brand import make_brand, scene_at

SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">'
       '<rect width="100" height="100" fill="{}"/></svg>')


def branded(tmp_path, **parts) -> Path:
    folder = make_brand(tmp_path / "brand", **parts)
    (folder / "logos").mkdir()
    (folder / "logos" / "m.svg").write_text(SVG.format("#ff0000"), encoding="utf-8")
    (folder / "logos" / "m-dark.svg").write_text(SVG.format("#00ff00"), encoding="utf-8")
    return folder


LOGOS = {"mark": {"file": "logos/m.svg", "onDark": "logos/m-dark.svg"}}


def _load(tmp_path, layers, **top):
    return k.load(str(scene_at(tmp_path, layers, **top)))


def test_logo_goes_to_the_bottom_right_corner_inside_the_safe_area(tmp_path):
    branded(tmp_path, logos=LOGOS)
    raw = _load(tmp_path, [{"use": "logo", "id": "l"}], brand="brand", variant="light").by_id["l"].raw
    assert (raw["anchor"], raw["x"], raw["y"], raw["h"]) == ("bottomright", 304.0, 171.0, 14.4)
    assert Path(raw["src"]).name == "m.svg"


def test_logo_takes_its_on_dark_file_on_a_dark_background(tmp_path):
    branded(tmp_path, logos=LOGOS)
    raw = _load(tmp_path, [{"use": "logo", "id": "l"}], brand="brand").by_id["l"].raw
    assert Path(raw["src"]).name == "m-dark.svg"


def test_brand_placement_sets_anchor_margin_and_height(tmp_path):
    branded(tmp_path, logos=LOGOS, placements={"corner": {"anchor": "topleft", "margin": 10, "height": "20%"}})
    raw = _load(tmp_path, [{"use": "logo", "id": "l", "placement": "corner"}], brand="brand").by_id["l"].raw
    assert (raw["anchor"], raw["x"], raw["y"], raw["h"]) == ("topleft", 10.0, 10.0, 36.0)


def test_unknown_placement_is_an_error_listing_the_choices(tmp_path):
    branded(tmp_path, logos=LOGOS)
    with pytest.raises(k.SceneError, match="bottomright"):
        _load(tmp_path, [{"use": "logo", "id": "l", "placement": "nowhere"}], brand="brand")


def test_unknown_logo_is_an_error_naming_the_brand_logos(tmp_path):
    branded(tmp_path, logos=LOGOS)
    with pytest.raises(k.SceneError, match="logos are: mark"):
        _load(tmp_path, [{"use": "logo", "id": "l", "logo": "full"}], brand="brand")


def test_logo_without_brand_logos_says_how_to_add_them(tmp_path):
    branded(tmp_path)
    with pytest.raises(k.SceneError, match="has no logos"):
        _load(tmp_path, [{"use": "logo", "id": "l"}], brand="brand")


def test_logo_below_the_brand_minimum_warns(tmp_path):
    branded(tmp_path, logos={**LOGOS, "minHeight": 200})
    s = _load(tmp_path, [{"use": "logo", "id": "l"}], brand="brand")
    assert any(i["code"] == "logo-small" for i in s.issues)


def test_bug_shows_the_brand_mark_for_the_whole_video(tmp_path):
    branded(tmp_path, logos=LOGOS, bug={"placement": "topright", "opacity": 0.5})
    s = _load(tmp_path, [{"use": "bug", "id": "b"}], brand="brand")
    L = s.by_id["b"]
    assert (L.t_in, L.t_out) == (0, s.duration)
    assert (L.raw["anchor"], L.raw["opacity"]) == ("topright", 0.5)


def test_logo_draws_its_file_in_the_corner(tmp_path):
    branded(tmp_path, logos=LOGOS)
    s = _load(tmp_path, [{"use": "logo", "id": "l", "h": 40}], brand="brand", variant="light")
    px = Renderer(s, backend="cpu").frame(0)
    assert tuple(px[160, 295][:3]) == (255, 0, 0)
    assert tuple(px[20, 20][:3]) == (250, 250, 250)
