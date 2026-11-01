"""See-through models: transparency on layers, parts and model files, the far side drawn through the near one,
a plane drawn back to back showing once, and the checks."""
import numpy as np
import pytest

import kinescript as k
from kinescript.render import Renderer
from kinescript.verify import check
from test_models import write_gltf

RED = np.array([230, 40, 40])


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 1, "bg": "#ffffff"}, **canvas),
            "layers": layers}


def frame(spec, n=0, backend="cpu"):
    try:
        return Renderer(k.Scene(spec), backend=backend).frame(n)[..., :3].astype(int)
    except RuntimeError:
        pytest.skip("no GPU")


def model(shape, **kw):
    """An unshaded red model over x 110..210, y 40..140."""
    return dict({"id": "m", "type": "model", "shape": shape, "w": 100, "r": 50, "x": 160, "y": 90, "light": "flat",
                 "fill": "#e62828"}, **kw)


def cover(px) -> float:
    """How much of the red covers white at this pixel."""
    return float(np.mean((255 - np.asarray(px)) / (255 - RED)))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_see_through_model_shows_its_far_side_and_what_is_behind(backend):
    solid = frame(scene([model("cube", rotateY=30)]), backend=backend)
    assert np.abs(solid[90, 160] - RED).max() <= 3
    plane = frame(scene([model("plane", transparency=0.6)]), backend=backend)
    assert cover(plane[90, 160]) == pytest.approx(0.4, abs=0.03)  # drawn back to back, it shows once
    cube = frame(scene([model("cube", rotateY=30, transparency=0.6)]), backend=backend)
    assert cover(cube[90, 160]) == pytest.approx(1 - 0.6 ** 2, abs=0.04)  # the near face over the far one
    behind = {"id": "bar", "type": "rect", "x": 160, "y": 90, "w": 20, "h": 180, "fill": "#2040e0"}
    f = frame(scene([behind, model("cube", rotateY=30, transparency=0.6)]), backend=backend)
    bar = np.array([32, 64, 224])
    assert np.abs(f[90, 160] - (0.6 ** 2 * bar + (1 - 0.6 ** 2) * RED)).max() <= 8  # the bar shows through both


def test_a_part_can_be_see_through_and_the_rest_solid():
    parts = [{"shape": "cube", "w": 120, "transparency": 0.7, "fill": "#88ccff"},
             {"shape": "sphere", "r": 30, "fill": "#e62828"}]
    f = frame(scene([{"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat", "parts": parts}]))
    alone = frame(scene([{"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat", "parts": parts[1:]}]))
    assert f[90, 160, 0] > 150 and f[90, 160, 0] > f[90, 160, 2]  # the ball shows through the case, tinted
    assert np.abs(f[90, 160] - alone[90, 160]).max() > 10
    assert f[20, 160].min() < 250 and f[20, 160, 2] > f[20, 160, 0]  # the case itself, above the ball


def test_frames_with_transparency_are_pure_functions_of_their_index():
    spec = scene([model("torus", r=60, tube=25, rotateX=60, rotateY=[[0, 0], ["end", 180]], light="studio",
                        transparency=[[0, 0.2], ["end", 0.8]], edges=True)])
    spec["fog"] = {"near": 0, "far": 3000}
    r = Renderer(k.Scene(spec), backend="cpu")
    seq = [r.frame(n).copy() for n in range(0, 30, 6)]
    for n, want in zip(range(24, -1, -6), seq[::-1]):
        assert (Renderer(k.Scene(spec), backend="cpu").frame(n) == want).all()


def test_model_files_say_how_see_through_they_are(tmp_path):
    from kinescript import models
    tri = ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], None, 0)
    blend = {"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 0.25]}, "alphaMode": "BLEND"}
    opaque = {"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 0.25]}}  # alphaMode OPAQUE: alpha ignored
    for mat, want in ((blend, 0.75), (opaque, None)):
        imp = models.load(write_gltf(tmp_path / "t.gltf", [tri], materials=[mat]))
        assert imp.materials[0].get("transparency") == want
    (tmp_path / "t.mtl").write_text("newmtl glass\nKd 0.5 0.8 1\nd 0.3\nnewmtl tr\nKd 1 1 1\nTr 0.6\n")
    (tmp_path / "t.obj").write_text("mtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nusemtl glass\nf 1 2 3\n"
                                    "usemtl tr\nf 2 4 3\n")
    got = sorted(m.get("transparency") for m in models.load(str(tmp_path / "t.obj")).materials)
    assert got == [0.6, 0.7]


def test_transparency_is_checked(tmp_path):
    from test_textures import quadrants
    img = quadrants(tmp_path / "q.png")

    def codes(layer):
        return [(i["level"], i["code"]) for i in check(scene([layer]))["issues"]]
    assert codes(model("cube", transparency=0.5)) == []
    assert codes(model("cube", transparency=[[0, 0], [1, 0.5]])) == []
    assert ("warning", "ignored-key") in codes(model("cube", transparency=0.5, texture=img))
    assert ("warning", "ignored-key") in codes({"id": "m", "type": "model", "parts": [
        {"shape": "cube", "transparency": 0.5}, {"shape": "sphere", "texture": img}]})
    assert any(lv == "error" for lv, _ in codes(model("cube", transparency="clear")))
    assert ("info", "renamed-key") in codes(model("cube", seeThrough=0.4))
