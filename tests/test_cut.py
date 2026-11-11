"""A model's "cut": the mesh it leaves (closed, the right side gone, the cap filling solids and leaving hollows
open), how it draws on both backends and on OpenGL, an animated cut, copies, and the checks."""
import numpy as np
import pytest

import kinescript as k
from kinescript import gl3d, mesh
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 240, "fps": 30, "duration": 2, "bg": "#f4efe6"}, **canvas),
            "layers": layers}


def ball(**kw):
    return dict({"id": "ball", "type": "model", "shape": "sphere", "r": 90, "x": 160, "y": 120, "light": "flat",
                 "fill": "#d04040"}, **kw)


def cap_area(m, n):
    a, b, c = (m.P[m.T[len(m.T) - n:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def closed(m):
    """Whether m is a closed surface (no border edges), and its Euler characteristic."""
    weld, face, first, ends, sides, same = mesh.edges(m.P, m.T)
    return not (sides[:, 1] == len(first)).any(), len(np.unique(weld[m.T])) - len(ends) + len(first)


@pytest.mark.parametrize("side", list(mesh.SIDES))
def test_a_cut_takes_off_its_side_and_closes_the_opening(side):
    axis, sign = mesh.SIDES[side]
    sphere = mesh.unit("sphere")
    for at in (0.5, 0.3):
        m, n = mesh.cut(sphere, axis, sign, at)
        lo, hi = sphere.P[:, axis].min(), sphere.P[:, axis].max()
        plane = lo + at * (hi - lo) if sign > 0 else hi - at * (hi - lo)
        assert ((m.P[:, axis] - plane) * sign >= -1e-9).all()  # nothing left on the side taken off
        assert closed(m) == (True, 2)
        face = m.N[m.T[len(m.T) - n:].ravel()]
        assert np.allclose(face[:, axis], -sign)  # the cap faces the side taken off
        a, b, c = (m.P[m.T[:, i]] for i in range(3))
        assert ((np.cross(b - a, c - a) * m.N[m.T].sum(1)).sum(1) > 0).all()  # every face wound to its normals
        r = np.sqrt(1 - (2 * at - 1) ** 2)
        assert cap_area(m, n) == pytest.approx(np.pi * r * r, rel=0.03)


def test_a_cut_leaves_a_hollow_open_and_fills_each_solid():
    torus = mesh.unit("torus")  # round the z axis, tube 0.35 across 1
    m, n = mesh.cut(torus, *mesh.SIDES["front"], 0.5)
    assert closed(m) == (True, 0)  # still a ring
    assert cap_area(m, n) == pytest.approx(np.pi * (1.35 ** 2 - 0.65 ** 2), rel=0.03)  # the hole stays open
    m, n = mesh.cut(torus, *mesh.SIDES["left"], 0.5)
    assert cap_area(m, n) == pytest.approx(2 * np.pi * 0.35 ** 2, rel=0.03)  # two ends of the tube
    # A part inside another: the outer one's cap goes round it, and it shows its own.
    outer, inner = mesh.unit("sphere"), mesh.placed(mesh.unit("sphere"), (0.4, 0.4, 0.4))
    both = mesh.merge([outer, inner], [0, 1])
    m, n = mesh.cut(both, *mesh.SIDES["front"], 0.5)
    slot = m.M[m.T[len(m.T) - n:, 0]]
    tri = m.T[len(m.T) - n:]
    a, b, c = (m.P[tri[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert area[slot == 0].sum() == pytest.approx(np.pi * (1 - 0.16), rel=0.03)
    assert area[slot == 1].sum() == pytest.approx(np.pi * 0.16, rel=0.03)
    m, n = mesh.cut(both, *mesh.SIDES["front"], 0.5, cap=2)
    assert set(m.M[m.T[len(m.T) - n:].ravel()]) == {2}


def test_no_cut_and_a_whole_cut():
    sphere = mesh.unit("sphere")
    assert mesh.cut(sphere, 2, -1.0, 0.0) == (sphere, 0)
    m, n = mesh.cut(sphere, 2, -1.0, 1.0)
    assert len(m.T) == 0 or cap_area(m, n) < 1e-6


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_the_cap_shows_where_the_front_was(backend):
    try:
        plain = Renderer(k.Scene(scene([ball()])), backend=backend).frame(0)[..., :3].astype(int)
        cut = Renderer(k.Scene(scene([ball(cut={"from": "front", "at": 0.5, "cap": "#20c040"})])),
                       backend=backend).frame(0)[..., :3].astype(int)
    except RuntimeError:
        pytest.skip("no GPU")
    assert np.abs(plain[120, 160] - (208, 64, 64)).max() < 12
    assert np.abs(cut[120, 160] - (32, 192, 64)).max() < 12
    # Cut from the back, the cap faces away: the front looks as it did.
    back = Renderer(k.Scene(scene([ball(cut={"from": "back", "at": 0.5, "cap": "#20c040"})])),
                    backend=backend).frame(0)[..., :3].astype(int)
    assert np.abs(back[120, 160] - plain[120, 160]).max() < 12


def test_each_copy_is_cut():
    spec = scene([ball(r=40, cut={"from": "front", "at": 0.5, "cap": "#20c040"},
                       repeat={"count": 3, "x": "=(i - 1) * 100"})])
    px = Renderer(k.Scene(spec), backend="cpu").frame(0)[..., :3].astype(int)
    for x in (60, 160, 260):
        assert np.abs(px[120, x] - (32, 192, 64)).max() < 12, x


def test_an_animated_cut_is_a_pure_function_of_the_frame():
    spec = scene([ball(rotateY=-30, edges=True, cut={"from": "top", "at": [[0, 0], [2, 0.7]], "cap": "#fbe3c8"})])
    r = Renderer(k.Scene(spec), backend="cpu")
    seq = [r.frame(n).copy() for n in (0, 20, 40, 59)]
    assert not np.array_equal(seq[0], seq[2])
    for n, want in zip((59, 40, 20, 0), seq[::-1]):
        assert np.array_equal(Renderer(k.Scene(spec), backend="cpu").frame(n), want)


def test_parts_cut_together_match_on_opengl(monkeypatch):
    spec = scene([{"id": "cell", "type": "model", "x": 160, "y": 120, "rotateX": -20, "rotateY": -35, "bands": 3,
                   "edges": {"width": 2.5, "color": "#20202a", "angle": 50}, "cut": {"from": "front", "at": 0.5},
                   "parts": [{"shape": "sphere", "r": 100, "fill": "#e88a6a"},
                             {"shape": "sphere", "r": 40, "fill": "#7a5cc8", "x": 20, "y": -10},
                             {"shape": "torus", "r": 30, "tube": 9, "fill": "#3aa37a", "x": -45, "y": 40,
                              "rotateX": 60}]}])
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
    assert r.on_gl == {"cell"}
    d = np.abs(a - b).max(-1)
    assert np.abs(a - b).mean() < 0.6 and (d > 64).mean() < 0.005, (np.abs(a - b).mean(), (d > 64).mean())
    assert np.abs(b[120, 175] - a[120, 175]).max() < 8  # the inner part's own cut face, purple on both


def test_cut_is_checked():
    def codes(cut):
        return [(i["level"], i["code"]) for i in check(scene([ball(cut=cut)]))["issues"]]
    for ok in ({"from": "front", "at": 0.5}, {"at": [[0, 0], [1, 0.5]], "cap": "#fff"}, {"from": "left"},
               {"cap": True}, {"at": "=0.2 + 0.1 * t"}):
        assert codes(ok) == [], ok
    for bad in ("front", {"from": "inside"}, {"at": 1.5}, {"at": "half"}, {"cap": "nope"}, {"cap": 3}):
        assert any(lv == "error" for lv, _ in codes(bad)), bad
    assert ("info", "renamed-key") in codes({"side": "front"})
    assert ("info", "renamed-key") in codes({"from": "near"})
    assert any(c == "unknown-key" for _, c in codes({"plane": 2}))
