"""Tubes in depth: along points, a seeded random walk or rays; rings carried without twisting so a flat path keeps
its old tube; a closed loop that meets itself; draw growing rays together; GL matching Skia; and the checks."""
import numpy as np
import pytest

import kinescript as k
from kinescript import gl3d, mesh
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 240, "fps": 30, "duration": 2, "bg": "#f4efe6"}, **canvas),
            "layers": layers}


def off_curve(m, q, closed):
    """Each vertex's distance from the curve through q, drawn finely."""
    dense = mesh.smooth(q, closed, 0.5)
    return np.array([np.sqrt(((dense - p) ** 2).sum(1)).min() for p in m.P])


def wound(m):
    a, b, c = (m.P[m.T[:, i]] for i in range(3))
    return ((np.cross(b - a, c - a) * m.N[m.T].sum(1)).sum(1) > 0).all()


def test_a_flat_path_keeps_the_tube_it_had():
    """Rings carried along a line in the xy plane keep side in the plane and up along z, as before."""
    line = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [40.0, 160.0]])
    m = mesh.tube([(line, False)], 10, 24)
    side, up = mesh._carried(np.c_[line, np.zeros(4)], np.array([[1.0, 0, 0], [0.7071, 0.7071, 0], [0, 1.0, 0],
                                                                 [-0.7071, 0.7071, 0]]), False)
    assert np.allclose(side[:, 2], 0, atol=1e-3) and np.allclose(np.abs(up[:, 2]), 1, atol=1e-3)
    assert np.allclose(m.P[:, 2].max(), 10, atol=1e-9) and wound(m)


@pytest.mark.parametrize("closed", [False, True])
def test_a_tube_in_depth_keeps_its_radius_and_a_loop_meets_itself(closed):
    s = np.linspace(0, 2 * np.pi if closed else 6 * np.pi, 300, endpoint=not closed)
    q = (np.c_[np.sin(s) + 2 * np.sin(2 * s), np.cos(s) - 2 * np.cos(2 * s), -np.sin(3 * s)] * 40 if closed
         else np.c_[80 * np.cos(s), 30 * s / np.pi - 90, 80 * np.sin(s)])  # a trefoil knot, a helix
    m = mesh.tube([(q, closed)], 8, 24)
    assert wound(m)
    assert np.allclose(off_curve(m, q, closed), 8, atol=0.05)
    if closed:  # the frame comes back turned round a knot; that turn is spread so the seam closes
        rings = m.P.reshape(-1, 25, 3)
        assert np.abs(rings[0] - rings[-1]).max() < 1e-6


def test_walk_is_seeded_and_kept_within_its_ball():
    w = mesh.walk(300, 6, 3, 60, 40, (0, 0, 0))
    assert w.shape == (301, 3) and np.linalg.norm(w, axis=1).max() <= 60 + 1e-9
    assert np.array_equal(w, mesh.walk(300, 6, 3, 60, 40, (0, 0, 0)))
    assert not np.allclose(w, mesh.walk(300, 6, 4, 60, 40, (0, 0, 0)))
    step = np.linalg.norm(np.diff(w, axis=0), axis=1)
    assert (step <= 6 + 1e-9).all() and np.median(step) == pytest.approx(6)


def test_rays_go_out_every_way_from_inner():
    rs = mesh.rays(40, [30, 60], 20, 1, (0, 0, 0))
    a = np.array([q[0] for q in rs])
    assert np.allclose(np.linalg.norm(a, axis=1), 20)
    ln = np.array([np.linalg.norm(q[1] - q[0]) for q in rs])
    assert (ln >= 30 - 1e-9).all() and (ln <= 60 + 1e-9).all()
    assert np.linalg.norm(a.mean(0)) < 4  # spread evenly: they cancel out
    # Grown together, each ray is drawn to the same share of its own length.
    half = mesh.tube([(q, False) for q in rs], 2, 8, 0.5, together=True)
    each = half.P.reshape(len(rs), -1, 3)  # every ray is two points, so as many vertices
    tips = [p[np.argmax(p @ (q[1] - q[0]))] for p, q in zip(each, rs)]
    reach = np.array([np.dot(t - q[0], (q[1] - q[0]) / n) for t, q, n in zip(tips, rs, ln)])
    assert np.allclose(reach / ln, 0.5 + 2 / ln, atol=0.02)  # half way, plus the round end


def spikes(**kw):
    return dict({"id": "s", "type": "model", "x": 160, "y": 120, "light": "flat", "parts": [
        {"shape": "sphere", "r": 50, "fill": "#e7c35a"},
        {"shape": "tube", "rays": {"count": 20, "length": 40, "inner": 45, "seed": 3}, "r": 3, "fill": "#8a5cc8"}]},
        **kw)


def test_routes_draw_in_depth_and_frames_stay_pure():
    walk = {"id": "w", "type": "model", "x": 160, "y": 120, "shape": "tube", "r": 4, "fill": "#3a7ad0",
            "walk": {"steps": 80, "step": 9, "within": 80, "seed": 4}, "rotateY": "=t * 90",
            "draw": [[0, 0], [1.5, 1]]}
    for layer in (walk, spikes(rotateY="=t * 40")):
        spec = scene([layer])
        r = Renderer(k.Scene(spec), backend="cpu")
        seq = [r.frame(n).copy() for n in (0, 30, 59)]
        assert not np.array_equal(seq[1], seq[2])
        assert (np.abs(seq[2][..., :3].astype(int) - (244, 239, 230)).max(-1) > 40).mean() > 0.02
        for n, want in zip((59, 30, 0), seq[::-1]):
            assert np.array_equal(Renderer(k.Scene(spec), backend="cpu").frame(n), want)
    # A coil through points, wound round the y axis, is seen from its side as a band of crossings.
    s = np.linspace(0, 4 * np.pi, 40)
    coil = {"id": "c", "type": "model", "x": 160, "y": 120, "shape": "tube", "r": 6, "fill": "#d05a3a",
            "light": "flat", "smooth": True,
            "points": [[float(90 * np.cos(a)), float(20 * a / np.pi - 40), float(90 * np.sin(a))] for a in s]}
    px = Renderer(k.Scene(scene([coil])), backend="cpu").frame(0)[..., :3].astype(int)
    red = np.abs(px - (208, 90, 58)).max(-1) < 40
    ys, xs = np.nonzero(red)
    assert 150 < np.ptp(xs) < 210 and 80 < np.ptp(ys) < 110


def test_tubes_in_depth_match_on_opengl(monkeypatch):
    spec = scene([spikes(rotateX=-20, rotateY=30)])
    monkeypatch.setenv("KINE_3D", "skia")
    try:
        a = Renderer(k.Scene(spec), backend="gpu").frame(0)[..., :3].astype(int)
    except RuntimeError:
        pytest.skip("no GPU")
    monkeypatch.setenv("KINE_3D", "auto")
    r = Renderer(k.Scene(spec), backend="gpu")
    if not gl3d.available(r):
        pytest.skip("no moderngl")
    b = r.frame(0)[..., :3].astype(int)
    assert r.on_gl == {"s"}
    d = np.abs(a - b).max(-1)
    assert np.abs(a - b).mean() < 0.6 and (d > 64).mean() < 0.005, (np.abs(a - b).mean(), (d > 64).mean())


def test_tube_routes_are_checked():
    def codes(**part):
        layer = {"id": "m", "type": "model", "parts": [dict({"shape": "tube", "r": 3}, **part)]}
        return [(i["level"], i["code"]) for i in check(scene([layer]))["issues"]]
    for ok in ({"points": [[0, 0, 0], [50, 20, 10]]}, {"points": [[0, 0], [50, 20]], "closed": True},
               {"walk": {"steps": 40, "within": 50}}, {"walk": 30}, {"rays": 12},
               {"rays": {"count": 8, "length": [10, 30], "inner": 20, "seed": 2}}):
        assert codes(**ok) == [], ok
    for bad in ({"points": [[0, 0, 0]]}, {"points": [[0, 0], [1, 2, 3]]}, {"points": "M0 0"},
                {"walk": {"steps": 1.5}}, {"walk": {"step": 0}}, {"walk": {"steps": 5000}}, {"walk": "far"},
                {"rays": {"count": 0}}, {"rays": {"length": [30, 10]}}, {"rays": {"inner": -1}}):
        assert any(lv == "error" for lv, _ in codes(**bad)), bad
    assert ("info", "renamed-key") in codes(walk={"n": 20, "radius": 40})
    assert any(c == "unknown-key" for _, c in codes(rays={"spread": 2}))
    assert ("warning", "ignored-key") in codes(d="M0 0 L100 0", walk=20)  # one route: d wins
    assert ("info", "renamed-value") in codes(h=60)  # no route: a cylinder, as before
    sphere = [(i["level"], i["code"]) for i in check(scene([{"id": "m", "type": "model", "shape": "sphere",
                                                               "rays": 12}]))["issues"]]
    assert ("warning", "ignored-key") in sphere
