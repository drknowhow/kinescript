"""pin: a top-level layer placed each frame where a point on another layer lands on screen, through that layer's
keys, its parents' and the camera; a callout that follows a turning model; and the checks."""
import numpy as np
import pytest

import kinescript as k
from kinescript.render import Renderer
from kinescript.verify import check

BEAD = [60, -40, 70]


def scene(layers, **top):
    return dict({"canvas": {"w": 640, "h": 360, "fps": 30, "duration": 3, "bg": "#ffffff"}, "layers": layers}, **top)


def cell(**kw):
    """A turning model with a red bead at BEAD, clear of its sphere."""
    return dict({"id": "cell", "type": "model", "x": 320, "y": 180, "rotateY": "=t * 60", "rotateX": -15,
                 "light": "flat", "scale": 1.2,
                 "parts": [{"shape": "sphere", "r": 50, "fill": "#dddddd"},
                           {"shape": "sphere", "r": 8, "fill": "#ff0000", "x": BEAD[0], "y": BEAD[1],
                            "z": BEAD[2]}]}, **kw)


def centroid(img, rgb):
    ys, xs = np.nonzero(np.abs(img[..., :3].astype(int) - rgb).max(-1) < 40)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


@pytest.mark.parametrize("camera", [None, {"z": 200, "rotateY": 15, "x": "=t * 40"}])
def test_a_pinned_layer_follows_its_point_on_a_turning_model(camera):
    top = {"camera": camera} if camera else {}
    dot = {"id": "dot", "type": "circle", "r": 3, "fill": "#0000ff", "pin": {"layer": "cell", "at": BEAD}}
    plain = Renderer(k.Scene(scene([cell()], **top)), backend="cpu")
    spec = scene([cell(), dot], **top)
    pinned = Renderer(k.Scene(spec), backend="cpu")
    seen = []
    for n in (0, 30, 60, 89):
        bead, img = centroid(plain.frame(n), (255, 0, 0)), pinned.frame(n).copy()
        assert np.abs(centroid(img, (0, 0, 255)) - bead).max() < 1.0, n
        seen.append(img)
        assert np.array_equal(Renderer(k.Scene(spec), backend="cpu").frame(n), img)  # a pure function of the frame
    assert not np.array_equal(seen[0], seen[2])
    assert check(spec)["issues"] == []


def test_x_and_y_move_it_off_its_point_and_an_id_alone_is_the_middle():
    def blue(layer):
        return centroid(Renderer(k.Scene(scene([cell(rotateY=0), layer])), backend="cpu").frame(0), (0, 0, 255))
    dot = {"id": "dot", "type": "circle", "r": 3, "fill": "#0000ff"}
    middle = blue(dict(dot, pin="cell"))
    assert np.abs(middle - (320, 180)).max() < 1.0
    assert np.abs(blue(dict(dot, pin="cell", x=40, y=-20)) - (360, 160)).max() < 1.0


def test_a_point_behind_the_camera_hides_the_layer_and_check_warns_off_the_canvas():
    dot = {"id": "dot", "type": "circle", "r": 6, "fill": "#0000ff", "pin": {"layer": "cell", "at": [0, 0, 2000]}}
    img = Renderer(k.Scene(scene([cell(rotateY=0), dot])), backend="cpu").frame(0)
    assert centroid(img, (0, 0, 255)) is None
    far = dict(dot, pin={"layer": "cell", "at": [500, 0, 0]})
    assert ("warning", "pin-offscreen") in [(i["level"], i["code"]) for i in check(scene([cell(), far]))["issues"]]


def test_a_callout_pinned_to_a_model_rings_its_point():
    co = {"use": "callout", "id": "co", "pin": {"layer": "cell", "at": BEAD}, "r": 20, "label": "Nucleus",
          "size": 20, "accent": "#00a000", "side": "left"}
    spec = scene([cell(rotateY=30), co])
    img = Renderer(k.Scene(spec), backend="cpu").frame(45)
    bead = centroid(Renderer(k.Scene(scene([cell(rotateY=30)])), backend="cpu").frame(45), (255, 0, 0))
    ring = centroid(img, (0, 160, 0))
    assert np.abs(ring - bead).max() < 2.0
    assert [i["code"] for i in check(spec)["issues"]] == []


def test_pin_is_checked():
    def codes(pin, nested=False, **dot):
        d = dict({"id": "dot", "type": "circle", "r": 3, "pin": pin}, **dot)
        layers = [cell(), {"id": "g", "type": "group", "children": [d]} if nested else d]
        return [(i["level"], i["code"]) for i in check(scene(layers))["issues"]]
    for ok in ("cell", {"layer": "cell"}, {"layer": "cell", "at": [10, 20]}, {"layer": "#cell", "at": [1, 2, 3]}):
        assert codes(ok) == [], ok
    for bad in (3, {"at": [0, 0, 0]}, {"layer": "cell", "at": [1]}, {"layer": "cell", "at": "middle"}):
        assert any(lv == "error" for lv, _ in codes(bad)), bad
    assert ("error", "pin-unknown") in codes("cel")
    assert ("error", "bad-pin") in codes("cell", nested=True)
    assert ("error", "bad-pin") in codes("dot")
    assert ("info", "renamed-key") in codes({"model": "cell", "point": [0, 0, 0]})
