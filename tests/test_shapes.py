"""Shapes built from their sizes: a cube rounded by r, the capsule, a lathe turned from its profile and a tube along
a path, which draw grows along its length."""
import numpy as np
import pytest

import kinescript as k
from kinescript import mesh, schema
from kinescript.render import Renderer


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def _sized(shape, **keys):
    s3, form = mesh.size(shape, lambda key, d=None: keys.get(key, d))
    return mesh.placed(mesh.unit(shape, "medium", form), s3)


def _issues(layer):
    _, rep = schema.normalize(scene([dict({"id": "m", "type": "model", "x": 160, "y": 90}, **layer)]))
    return [(i["level"], i["code"], i["message"]) for i in rep.issues]


def test_a_cube_with_r_rounds_its_edges_and_corners_into_quarter_circles():
    m = _sized("cube", w=200, h=120, depth=80, r=30)
    assert np.allclose(m.P.min(0), [-100, -60, -40]) and np.allclose(m.P.max(0), [100, 60, 40])
    inner = np.array([70, 30, 10])  # the box the corners are rounded round
    corner = (np.abs(m.P) > inner - 1e-9).all(1)
    assert corner.any()
    dist = np.linalg.norm(np.abs(m.P[corner]) - inner, axis=1)
    assert np.allclose(dist, 30, atol=1e-6)
    # Normals point straight out from the rounding: along the corner's radius.
    away = (np.abs(m.P[corner]) - inner) / 30
    assert np.allclose(np.abs(m.N[corner]), away, atol=1e-6)
    assert np.array_equal(_sized("cube", w=200).P, mesh.placed(mesh.unit("cube"), (200, 200, 200)).P)  # r 0: square
    # r larger than half the smallest side rounds that side right round, and no more.
    m = _sized("cube", w=100, h=100, depth=20, r=50)
    assert np.allclose(m.P.max(0), [50, 50, 10])


def test_a_capsule_is_a_cylinder_with_half_spheres_on_its_ends():
    m = _sized("capsule", r=40, h=200)
    assert np.allclose(m.P.min(0), [-40, -100, -40], atol=1e-6) and np.allclose(m.P.max(0), [40, 100, 40], atol=1e-6)
    top = m.P[:, 1] < -60
    assert np.allclose(np.linalg.norm(m.P[top] - [0, -60, 0], axis=1), 40, atol=1e-6)
    assert ((m.N[top] * (m.P[top] - [0, -60, 0])).sum(1) > 0.99 * 40).all()  # normals out from the cap's centre
    ball = _sized("capsule", r=50, h=60)  # shorter than its width: a ball
    assert np.allclose(np.linalg.norm(ball.P, axis=1), 50, atol=1e-6)


def _inward(m) -> float:
    """The share of normals that point in toward the y axis (flat ends point neither way)."""
    radial = m.P * [1, 0, 1]
    far = np.linalg.norm(radial, axis=1) > 1
    return float(((m.N[far] * radial[far]).sum(1) < -1e-6).mean())


def test_a_lathe_turns_its_profile_round_the_vertical_axis():
    from kinescript.render3d import _profile
    lines, extent = _profile([[0, 0], [50, 0], [50, 200], [0, 200]], 1.0)
    m = mesh.lathe(lines, 48, 30)
    assert np.allclose(m.P.min(0), [-50, -100, -50], atol=1e-6) and np.allclose(m.P.max(0), [50, 100, 50], atol=1e-6)
    assert extent == 200 and _inward(m) == 0.0
    # Drawn bottom to top, the profile is turned round so the normals still face out.
    back, _ = _profile([[0, 200], [50, 200], [50, 0], [0, 0]], 1.0)
    assert _inward(mesh.lathe(back, 48, 30)) == 0.0
    # A sharp corner keeps a normal on each side: the top's normals all point up.
    top = m.P[:, 1] < -100 + 1e-6
    assert (m.N[top][:, 1] < -0.99).any() and (np.abs(m.N[top][:, 1]) < 0.01).any()
    # An open line off the axis (a cup with no base) is drawn on both sides.
    cup, _ = _profile([[80, -100], [60, 100]], 1.0)
    one = mesh.lathe([(cup[0][0], False)], 48, 400)
    assert 0.45 < _inward(one) < 0.55
    # Path data: its x is the distance from the axis.
    vase, _ = _profile("M 0 -100 L 40 -100 C 90 -40 90 40 40 100 L 0 100", 1.0)
    v = mesh.lathe(vase, 48, 30)
    assert 60 < v.P[:, 0].max() < 80 and np.allclose([v.P[:, 1].min(), v.P[:, 1].max()], [-100, 100], atol=1e-6)


def test_a_tube_runs_along_its_path_with_round_ends_and_grows():
    import skia
    line = mesh.polylines(skia.Path().moveTo(-100, 0).lineTo(100, 0), 5)
    assert len(line) == 1 and not line[0][1]
    m = mesh.tube(line, 10, 24)
    assert np.allclose(m.P.min(0), [-110, -10, -10], atol=1e-6) and np.allclose(m.P.max(0), [110, 10, 10], atol=1e-6)
    half = mesh.tube(line, 10, 24, 0.5)
    assert np.isclose(half.P[:, 0].max(), 10, atol=1e-6)  # half its length, and a round end
    assert len(mesh.tube(line, 10, 24, 0.0).T) == 0
    # A closed path has no ends: every point is r from the circle it runs round.
    ring = mesh.polylines(skia.Path().addCircle(0, 0, 100), 2)
    assert ring[0][1]
    m = mesh.tube(ring, 8, 16)
    off = np.hypot(np.hypot(m.P[:, 0], m.P[:, 1]) - 100, m.P[:, 2])
    assert np.allclose(off, 8, atol=0.2)
    # A corner keeps the tube's width: points on the bend's outside sit r out along its mitre.
    corner = mesh.tube([(np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]), False)], 10, 24)
    assert np.isclose(corner.P[:, 0].max(), 110, atol=1e-6)


def _shot(spec, T=0.0):
    r = Renderer(k.Scene(spec), backend="cpu", track=True)
    r.draw(T)
    return r


def test_draw_grows_a_tube_along_its_length_and_frames_stay_pure():
    spec = scene([{"id": "m", "type": "model", "shape": "tube", "r": 6, "d": "M -100 0 L 100 0", "x": 160, "y": 90,
                   "light": "flat", "draw": [[0, 0], [1, 1]]},
                  {"id": "v", "type": "model", "shape": "lathe", "profile": [[0, -30], [30, -30], [20, 30], [0, 30]],
                   "x": 60, "y": 90, "rotateX": "=t * 40"}])
    r = _shot(spec, 0.5)
    left, _, right, _ = r.boxes["m"]["rect"]
    assert 100 < right - left < 125 and left == pytest.approx(54, abs=4)  # half of 212, from the left end
    assert _shot(spec, 0.0).boxes.get("m") is None
    seq = Renderer(k.Scene(spec), backend="cpu")
    frames = [seq.frame(n).copy() for n in range(0, 24)]
    alone = Renderer(k.Scene(spec), backend="cpu").frame(17)
    assert np.array_equal(frames[17], alone)


@pytest.mark.parametrize("layer, want", [
    ({"shape": "tube", "r": 40, "h": 100}, ("info", "renamed-value", "shape 'tube' without d read as 'cylinder'")),
    ({"shape": "lathe"}, ("error", "missing-key", "a lathe needs a profile")),
    ({"shape": "lathe", "profile": [[-10, 0], [20, 50]]}, ("error", "bad-value", "profile has a point with r below 0")),
    ({"shape": "lathe", "profile": [[0, 0], [0, 50]]}, ("error", "bad-value", "profile's points are all on the axis")),
    ({"shape": "lathe", "profile": [10, 20, 30]}, ("error", "bad-value", "profile is [10, 20, 30]")),
    ({"shape": "lathe", "profile": "M 0 0 L 40"}, ("error", "bad-value", "profile: path command L needs")),
    ({"shape": "cube", "profile": [[0, 0], [10, 10]]}, ("warning", "ignored-key", "profile is ignored")),
    ({"shape": "tube", "d": "M 0 0 Q"}, ("error", "bad-value", "d: path command Q needs")),
    ({"shape": "pill"}, ("info", "renamed-value", "shape 'pill' read as 'capsule'")),
])
def test_shape_mistakes_are_named(layer, want):
    got = _issues(layer)
    assert any(lv == want[0] and code == want[1] and msg.startswith(want[2]) for lv, code, msg in got), got


def test_a_tube_part_without_d_is_a_cylinder_and_a_lathe_part_turns():
    spec = scene([{"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat", "parts": [
        {"shape": "tube", "r": 20, "h": 60, "x": -60},
        {"shape": "lathe", "profile": [[0, -30], [25, -30], [25, 30], [0, 30]], "x": 60}]}])
    fixed, _ = schema.normalize(spec)
    assert fixed["layers"][0]["parts"][0]["shape"] == "cylinder"
    left, top, right, bottom = _shot(spec).boxes["m"]["rect"]
    assert right - left == pytest.approx(165, abs=6) and 60 <= bottom - top < 72  # ends seen from above add some
