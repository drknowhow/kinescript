"""Edge lines on models: which edges a mesh has, creases and outlines drawn on both backends, hidden lines staying
hidden, the wireframe, textured models, and the checks."""
import numpy as np
import pytest

import kinescript as k
from kinescript import mesh
from kinescript.render import Renderer
from kinescript.verify import check

INK = (10, 12, 16)


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 1, "bg": "#ffffff"}, **canvas),
            "layers": layers}


def frame(spec, n=0, backend="cpu"):
    try:
        return Renderer(k.Scene(spec), backend=backend).frame(n)[..., :3].astype(int)
    except RuntimeError:
        pytest.skip("no GPU")


def cube(**kw):
    """A white cube, unshaded, turned so two faces show and meet in a vertical crease at x 160."""
    return dict({"id": "box", "type": "model", "shape": "cube", "w": 90, "x": 160, "y": 90, "rotateY": 45,
                 "light": "flat", "fill": "#ffffff"}, **kw)


def dark(px):
    return bool(np.abs(np.asarray(px) - INK).max() <= 40)


def test_a_mesh_has_each_edge_once_with_the_faces_either_side():
    m = mesh.unit("cube")  # corners kept apart per face, for the normals
    weld, face, first, ends, sides, same = mesh.edges(m.P, m.T)
    assert weld.max() + 1 - len(ends) + len(first) == 2  # a closed surface: V - E + F
    assert (sides[:, 1] < len(first)).all() and not same.any()
    a, b, c = (m.P[m.T[first, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    crease = (n[sides[:, 0]] * n[sides[:, 1]]).sum(1) < 0.5
    p = {int(w): m.P[i] for i, w in enumerate(weld)}
    ab = np.array([[p[int(x)], p[int(y)]] for x, y in ends[crease]])
    assert np.isclose(np.abs(ab) - 0.5, 0, atol=1e-9).sum(2).min() >= 2  # every crease lies along the cube's edges
    assert np.isclose(np.linalg.norm(ab[:, 0] - ab[:, 1], axis=1).sum(), 12)  # and they add up to all twelve
    plane = mesh.unit("plane")  # drawn back to back: one face each
    weld, face, first, ends, sides, same = mesh.edges(plane.P, plane.T)
    assert len(first) == len(plane.T) // 2
    border = sides[:, 1] == len(first)
    ab = plane.P[[np.flatnonzero(weld == w)[0] for w in ends[border].ravel()]].reshape(-1, 2, 3)
    assert np.isclose(np.linalg.norm(ab[:, 0] - ab[:, 1], axis=1).sum(), 4)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_creases_and_the_outline_are_drawn_and_flat_faces_are_not(backend):
    plain, lined = frame(scene([cube()]), backend=backend), frame(scene([cube(edges=True)]), backend=backend)
    assert not dark(plain[90, 160]) and dark(lined[90, 160])  # the crease between the two faces
    assert dark(lined[90, 125]) is False and dark(lined[90, 195]) is False  # the faces themselves
    rows = np.flatnonzero((np.abs(lined[:, :, :] - INK).max(2) <= 40).any(1))
    assert rows.min() < 50 and rows.max() > 130  # the outline above and below
    cols = np.flatnonzero(np.abs(lined[90] - INK).max(1) <= 40)
    assert cols.min() < 110 and cols.max() > 210  # and either side
    wide = frame(scene([cube(edges={"width": 8, "color": "#ff0000"})]), backend=backend)
    assert np.abs(wide[90, 157] - (255, 0, 0)).max() <= 40 and np.abs(wide[90, 163] - (255, 0, 0)).max() <= 40


def test_lines_behind_other_faces_stay_hidden():
    # A card in front of the cube, in the same model: the cube's crease is behind it, the card's border is not.
    parts = [{"shape": "cube", "w": 90, "z": -80, "rotateY": 45, "fill": "#ffffff"},
             {"shape": "plane", "w": 200, "h": 140, "z": 80, "fill": "#88ccff"}]
    spec = scene([{"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat", "parts": parts, "edges": True}])
    f = frame(spec)
    cols = np.flatnonzero(np.abs(f[90] - INK).max(1) <= 40)
    assert np.abs(f[90, 160] - (136, 204, 255)).max() <= 8 and not ((cols > 100) & (cols < 220)).any()
    assert cols.min() < 60 and cols.max() > 260  # the card's border
    alone = frame(scene([{"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat", "parts": parts[:1],
                          "edges": True}]))
    assert dark(alone[90, 160])


def test_all_draws_every_edge_and_an_angle_picks_creases():
    def inked(edges):
        f = frame(scene([{"id": "ball", "type": "model", "shape": "sphere", "r": 70, "x": 160, "y": 90,
                          "light": "flat", "fill": "#ffffff", "edges": edges}]))
        return int((np.abs(f - INK).max(2) <= 40).sum())
    outline, wire, most = inked(True), inked("all"), inked({"angle": 1})
    assert outline > 0 and wire > 4 * outline  # the sphere's facets turn less than 30 degrees: only its outline
    assert most > 2 * outline  # its facets turn more than 1 degree


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_textured_model_takes_its_edges(tmp_path, backend):
    from test_textures import quadrants
    img = quadrants(tmp_path / "q.png")
    spec = scene([{"id": "card", "type": "model", "shape": "plane", "w": 160, "x": 160, "y": 90, "light": "flat",
                   "texture": img, "edges": {"width": 4, "color": "#ffffff"}}], bg="#000000")
    f = frame(spec, backend=backend)
    assert np.abs(f[90, 80] - 255).max() <= 30 and np.abs(f[30, 120] - (230, 40, 40)).max() <= 12


def test_frames_with_edges_are_pure_functions_of_their_index():
    spec = scene([cube(rotateY=[[0, 0], ["end", 180]], rotateX=-30, light="studio", fill="#ff633f",
                       explode=[[0.5, 0], ["end", 0.6]], draw=[[0, 0.3], [0.5, 1]],
                       edges={"width": [[0, 1], ["end", 6]], "angle": [[0, 60], ["end", 0]], "color": "#123456"})])
    spec["fog"] = {"near": 0, "far": 3000}
    r = Renderer(k.Scene(spec), backend="cpu")
    seq = [r.frame(n).copy() for n in range(0, 30, 6)]
    for n, want in zip(range(24, -1, -6), seq[::-1]):
        assert (Renderer(k.Scene(spec), backend="cpu").frame(n) == want).all()


def test_edges_are_checked():
    def codes(edges):
        return [(i["level"], i["code"]) for i in check(scene([cube(edges=edges)]))["issues"]]
    for ok in (True, False, "all", {"width": 3}, {"color": "$accent", "angle": [[0, 10], [1, 60]]}):
        assert codes(ok) == [], ok
    for bad in ("some", 3, {"width": "thick"}, {"color": "nope"}, {"angle": [1, 2]}):
        assert any(lv == "error" for lv, _ in codes(bad)), bad
    assert ("warning", "unknown-key") in codes({"widht": 2}) or ("error", "unknown-key") in codes({"widht": 2})
    issues = check(scene([dict({k: v for k, v in cube().items()}, outline={"colour": "#fff"})]))["issues"]
    assert {i["code"] for i in issues} == {"renamed-key"}


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_look_draws_the_triangles_and_a_stages_depth(tmp_path, backend):
    from kinescript.verify import contact_sheet
    kids = [{"id": "ball", "type": "model", "shape": "sphere", "r": 40, "z": 60, "light": "flat"},
            {"id": "card", "type": "rect", "w": 200, "h": 120, "fill": "#0000ff"}]
    spec = scene([{"id": "g", "type": "group", "x": 160, "y": 90, "sort": "pixel", "children": kids}])
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    r.frame(0)
    assert r.wires == [] and r.depths == []  # only when asked
    r.wire = r.depth_view = True
    r.frame(0)
    (segs,), (d,) = r.wires, r.depths
    assert segs.shape[1:] == (2, 2) and len(segs) > 100
    assert np.abs(segs.reshape(-1, 2) - (160, 90)).max() < 60  # round the ball
    assert d.shape == (180, 320) and d[90, 160] > d[90, 240] > 0 and d[5, 5] == 0  # the ball nearer than the card
    out = contact_sheet(spec, str(tmp_path / "a.png"), count=1, width=160, wire=True, depth=True)
    assert "notes" not in out
    flat = contact_sheet(scene([cube()]), str(tmp_path / "b.png"), count=1, width=160, depth=True)
    assert "no stage" in flat["notes"][0]
