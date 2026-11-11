"""Model layers on OpenGL (gl3d): which models draw there, that they match the Skia path, stages, a partial draw,
moving copies, repeatability, and KINE_3D. The drawing tests need the GPU backend and moderngl, and skip without
them (CI runs on the CPU)."""
import numpy as np
import pytest

import kinescript as k
from kinescript import gl3d
from kinescript.render import Renderer


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 240, "fps": 30, "duration": 2, "bg": "#f4efe6"}, **canvas),
            "layers": layers}


def ball(**kw):
    return dict({"id": "ball", "type": "model", "shape": "sphere", "w": 120, "x": 160, "y": 120, "fill": "#5b6cf0",
                 "bands": 3, "edges": {"width": 2.5, "color": "#20202a", "angle": 50}}, **kw)


def ring(**kw):
    return dict({"id": "ring", "type": "model", "shape": "torus", "w": 170, "x": 160, "y": 120, "rotateX": 70,
                 "fill": "#e2683c", "shine": 0.6}, **kw)


def renderer(spec, monkeypatch, mode="auto"):
    monkeypatch.setenv("KINE_3D", mode)
    try:
        r = Renderer(k.Scene(spec), backend="gpu")
    except RuntimeError:
        pytest.skip("no GPU")
    if mode != "skia" and not gl3d.available(r):
        pytest.skip("no moderngl")
    return r


def both(spec, monkeypatch, n=0):
    """Frame n on the Skia path and on OpenGL, and the models OpenGL drew."""
    a = renderer(spec, monkeypatch, "skia").frame(n)[..., :3].astype(int)
    r = renderer(spec, monkeypatch)
    b = r.frame(n)[..., :3].astype(int)
    return a, b, r.on_gl


def close(a, b, mean=1.0, far=0.02):
    """Mean difference under `mean` levels, and under `far` of the pixels more than 64 levels apart: the paths find
    edge lines differently, so the pixels along a line may differ, never the shading."""
    d = np.abs(a - b).max(-1)
    assert np.abs(a - b).mean() < mean and (d > 64).mean() < far, (np.abs(a - b).mean(), (d > 64).mean())


def test_a_lit_model_matches_the_skia_path(monkeypatch):
    a, b, gl = both(scene([ring()]), monkeypatch)
    assert gl == {"ring"}
    close(a, b, mean=0.2, far=0.001)


def test_toon_bands_and_edge_lines_match_the_skia_path(monkeypatch):
    a, b, gl = both(scene([ball()]), monkeypatch)
    assert gl == {"ball"}
    close(a, b)
    ink = [np.flatnonzero(np.abs(x[120] - (32, 32, 42)).max(-1) < 40) for x in (a, b)]
    assert len(ink[1]) and abs(ink[0].max() - ink[1].max()) <= 1  # the outline, at the sphere's right edge


def test_models_that_need_skia_draw_there(monkeypatch):
    for extra in ({"transparency": 0.4}, {"explode": 0.5}, {"edges": "all"}):
        r = renderer(scene([ball(**extra)]), monkeypatch)
        r.frame(0)
        assert r.on_gl == set(), extra


def test_kine_3d_skia_keeps_every_model_on_skia(monkeypatch):
    r = renderer(scene([ball()]), monkeypatch, "skia")
    r.frame(0)
    assert r.on_gl == set()


def test_parts_that_cross_in_one_model_meet_without_steps(monkeypatch):
    # Two boxes through each other: the Skia path sorts whole triangles, so their crossing steps; a depth buffer
    # draws it straight. Compare the two paths only away from that line.
    spec = scene([{"id": "x", "type": "model", "x": 160, "y": 120, "rotateY": 30, "rotateX": -20, "light": "flat",
                   "parts": [{"shape": "cube", "w": 160, "h": 40, "d": 40, "fill": "#d04040"},
                             {"shape": "cube", "w": 40, "h": 160, "d": 40, "fill": "#4060d0", "rotateY": 25}]}])
    a, b, gl = both(spec, monkeypatch)
    assert gl == {"x"}
    close(a, b, mean=3.0, far=0.05)


def test_a_stage_merges_models_drawn_on_opengl(monkeypatch):
    spec = scene([{"id": "st", "type": "group", "sort": "pixel", "children": [
        ring(), ball(w=90, edges=None)]}])
    a, b, gl = both(spec, monkeypatch)
    assert gl == {"ring", "ball"}
    close(a, b, mean=0.5, far=0.005)
    assert np.abs(b[120, 160] - a[120, 160]).max() < 8  # the ball, in front of the ring's hole


def test_a_model_drawn_in_part_matches_the_skia_path(monkeypatch):
    spec = scene([ring(draw=[[0, 0], [2, 1, "linear"]])])
    a, b, gl = both(spec, monkeypatch, n=24)
    assert gl == {"ring"}
    close(a, b, mean=0.3, far=0.002)
    assert (np.abs(b - b[0, 0]).max(-1) > 30).mean() < (np.abs(renderer(scene([ring()]), monkeypatch).frame(0)[..., :3]
                                                               .astype(int) - b[0, 0]).max(-1) > 30).mean()


def test_moving_copies_keep_their_buffers_and_draw_as_a_fresh_renderer_does(monkeypatch):
    spec = scene([{"id": "dots", "type": "model", "shape": "sphere", "w": 20, "x": 160, "y": 120, "fill": "#c03070",
                   "detail": "low", "repeat": {"count": 12, "x": "=i * 22 - 120", "y": "=sin(t * 3 + i) * 40"}}])
    r = renderer(spec, monkeypatch)
    r.frame(0)
    buf = r._gl_meshes["dots"]
    later = r.frame(20)[..., :3]
    assert r._gl_meshes["dots"] is buf  # the same buffers, the vertices written again
    assert np.array_equal(later, renderer(spec, monkeypatch).frame(20)[..., :3])


def test_a_frame_is_the_same_whatever_was_drawn_before_it(monkeypatch):
    spec = scene([{"id": "st", "type": "group", "sort": "pixel", "children": [ring(), ball(x=200)]}])
    r = renderer(spec, monkeypatch)
    first = r.frame(30)
    for n in (0, 59, 12):
        r.frame(n)
    assert np.array_equal(first, r.frame(30))


def test_kine_3d_is_checked(monkeypatch):
    monkeypatch.setenv("KINE_3D", "bogus")
    with pytest.raises(ValueError, match="KINE_3D"):
        gl3d.setting()
    monkeypatch.setenv("KINE_3D", "gl")
    r = Renderer(k.Scene(scene([ball()])), backend="cpu")
    with pytest.raises(RuntimeError, match="KINE_3D=gl"):
        r.frame(0)


def test_what_fits_on_opengl():
    mat = ((0.5, 0.5, 0.5), 0.0, 0.4, 20.0, 0.0, 0.5, 0.0, (0.0, 0.0, 0.0), 1.0)
    clear = mat[:8] + (0.6,)
    lines = (2.0, [0, 0, 0], 0.8)
    assert gl3d.fits([mat], [None], [None], 0.0, lines)
    assert gl3d.fits([mat] * gl3d.MATS, [None] * gl3d.MATS, [None] * gl3d.MATS, 0.0, None)
    assert not gl3d.fits([mat] * (gl3d.MATS + 1), [None] * 65, [None] * 65, 0.0, None)
    assert not gl3d.fits([clear], [None], [None], 0.0, None)
    assert not gl3d.fits([mat], [object()], [None], 0.0, None)
    assert not gl3d.fits([mat], [None], [object()], 0.0, None)
    assert not gl3d.fits([mat], [None], [None], 0.3, None)
    assert not gl3d.fits([mat], [None], [None], 0.0, (2.0, [0, 0, 0], 2.0))  # edges: "all"


def test_moving_copies_drawn_in_part_follow_the_build_order_of_each_frame(monkeypatch):
    spec = scene([{"id": "dots", "type": "model", "shape": "sphere", "w": 20, "x": 160, "y": 120, "fill": "#c03070",
                   "detail": "low", "draw": [[0, 0], [2, 1, "linear"]],
                   "repeat": {"count": 12, "x": "=i * 22 - 120", "y": "=sin(t * 3 + i) * 40"}}])
    r = renderer(spec, monkeypatch)
    r.frame(20)
    later = r.frame(40)[..., :3]
    assert r.on_gl == {"dots"}
    assert np.array_equal(later, renderer(spec, monkeypatch).frame(40)[..., :3])


def test_a_washed_model_and_its_lines_mix_toward_the_paper_on_both_paths(monkeypatch):
    """wash mixes the drawn model, lines and all, toward the canvas background and keeps it opaque."""
    plain, washed = scene([ball()]), scene([ball(wash=0.6)])
    a, b, gl = both(washed, monkeypatch)
    assert gl == {"ball"}
    close(a, b)
    full = renderer(plain, monkeypatch, "skia").frame(0)[..., :3].astype(float)
    paper = np.array([0xf4, 0xef, 0xe6], float)
    mask = np.abs(full - paper).max(-1) > 30  # where the ball draws
    assert np.abs(full[mask] * 0.4 + paper * 0.6 - a[mask]).mean() < 2
