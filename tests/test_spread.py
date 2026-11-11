"""repeat's "spread": seeded places in a ball, on its shell, in a box, a disc or on a ring, kept apart by gap and
out of the balls in avoid; the same on every run; the copies' own x, y and z added on top; and the checks."""
import numpy as np
import pytest

import kinescript as k
from kinescript.props import spread
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 240, "fps": 30, "duration": 1, "bg": "#ffffff"}, **canvas),
            "layers": layers}


def apart(p):
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    return (d + np.eye(len(p)) * 1e9).min()


def test_each_shape_holds_its_places():
    r = np.linalg.norm
    p = spread({"in": "sphere", "r": 100}, 500)
    assert (r(p, axis=1) <= 100 + 1e-9).all() and r(p, axis=1).max() > 90
    assert np.allclose(r(spread({"in": "shell", "r": 100}, 50), axis=1), 100)
    assert np.allclose(r(spread({"on": "sphere", "r": 100}, 50), axis=1), 100)
    d = spread({"in": "disc", "r": 80}, 300)
    assert (d[:, 2] == 0).all() and (r(d, axis=1) <= 80 + 1e-9).all()
    assert np.allclose(r(spread({"in": "ring", "r": 80}, 30), axis=1), 80)
    b = spread({"in": "box", "w": 200, "h": 100, "depth": 40}, 400)
    assert (np.abs(b) <= (100, 50, 20)).all() and (np.abs(b) > (90, 45, 18)).any(0).all()


def test_the_same_seed_gives_the_same_places_and_another_seed_others():
    a = spread({"in": "sphere", "r": 100, "seed": 7, "gap": 12}, 100)
    assert np.array_equal(a, spread({"in": "sphere", "r": 100, "seed": 7, "gap": 12}, 100))
    assert not np.allclose(a, spread({"in": "sphere", "r": 100, "seed": 8, "gap": 12}, 100))


def test_gap_and_avoid_are_kept_where_there_is_room():
    p = spread({"in": "disc", "r": 200, "gap": 25, "seed": 2}, 80)
    assert apart(p) >= 25 - 1e-9
    p = spread({"in": "sphere", "r": 200, "gap": 20, "avoid": [{"x": 50, "y": -20, "z": 0, "r": 90}]}, 120)
    assert apart(p) >= 20 - 1e-9
    assert (np.linalg.norm(p - (50, -20, 0), axis=1) >= 90).all()
    crowded = spread({"in": "disc", "r": 50, "gap": 30}, 40)  # no room for that gap: every copy is still placed
    assert len(crowded) == 40 and np.isfinite(crowded).all()


def test_copies_take_their_places_and_their_own_keys_on_top():
    spot = {"in": "disc", "r": 90, "seed": 5, "gap": 24}
    p = spread(spot, 12)
    dots = {"id": "d", "type": "circle", "r": 6, "x": 160, "y": 120, "fill": "#d03030"}
    a = Renderer(k.Scene(scene([dict(dots, repeat={"count": 12, "spread": spot, "x": 4})])), backend="cpu").frame(0)
    b = Renderer(k.Scene(scene([dict(dots, repeat={"count": 12, "x": [float(v) + 4 for v in p[:, 0]],
                                                   "y": [float(v) for v in p[:, 1]]})])), backend="cpu").frame(0)
    assert np.array_equal(a, b)
    # A model's copies are placed in depth too, and a moving key moves them about their places.
    ball = {"id": "b", "type": "model", "shape": "sphere", "r": 10, "x": 160, "y": 120, "light": "flat",
            "detail": "low"}
    spec = scene([dict(ball, repeat={"count": 20, "spread": {"in": "sphere", "r": 80, "seed": 3},
                                     "y": "=sin(t * 6 + i) * 5"})])
    r = Renderer(k.Scene(spec), backend="cpu")
    first, later = r.frame(0).copy(), r.frame(15).copy()
    assert not np.array_equal(first, later)
    assert np.array_equal(Renderer(k.Scene(spec), backend="cpu").frame(0), first)


def test_spread_is_checked():
    def codes(sp, t="model", n=20):
        layer = ({"id": "m", "type": "model", "shape": "sphere", "r": 10} if t == "model"
                 else {"id": "m", "type": "circle", "r": 5})
        return [(i["level"], i["code"]) for i in check(scene([dict(layer, repeat={"count": n,
                                                                                    "spread": sp})]))["issues"]]
    for ok in ({"in": "sphere", "r": 100}, {"in": "box", "w": 300, "seed": 2}, {"on": "sphere", "r": 50},
               {"in": "disc", "r": 80, "gap": 10, "avoid": [{"x": 0, "y": 0, "r": 20}]}):
        assert codes(ok) == [], ok
    assert codes({"in": "disc", "r": 80}, t="circle") == []
    for bad in ("disc", {"in": "cloud"}, {"in": "sphere", "gap": -1}, {"seed": 1.5}, {"avoid": {"x": 1}},
                {"avoid": [{"x": "left"}]}, {"in": "disc", "on": "sphere"}):
        assert any(lv == "error" for lv, _ in codes(bad)), bad
    assert any(lv == "error" for lv, _ in codes({"in": "sphere"}, t="circle"))  # in depth: a model's only
    assert ("warning", "spread-crowded") in codes({"in": "disc", "r": 50, "gap": 30}, n=40)
    assert ("info", "renamed-key") in codes({"shape": "disc", "radius": 50})
