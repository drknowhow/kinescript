"""Model layers on OpenGL (gl3d): which models draw there, that they match the Skia path, stages and their shadow
maps, a partial draw, moving copies, repeatability, what check and --boxes read, and KINE_3D. The drawing tests need
the GPU backend and moderngl, and skip without them (CI runs on the CPU)."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript import gl3d, stage
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


def renderer(spec, monkeypatch, mode="auto", **kw):
    monkeypatch.setenv("KINE_3D", mode)
    try:
        r = Renderer(k.Scene(spec), backend="gpu", **kw)
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


def tracked(spec, monkeypatch, mode, n):
    """Frame n as check and --boxes draw it, the renderer, and what they read from it afterwards."""
    renderer(spec, monkeypatch, mode)  # skips where a render would not draw on OpenGL either
    r = Renderer(k.Scene(spec), backend="gpu", track=True)
    px = r.frame(n)[..., :3].astype(int)
    return px, r, (r.boxes, r.tris, r.culled, r.inside, r.fogged, r.unlit)


def test_check_reads_the_same_from_models_drawn_on_opengl(monkeypatch):
    """A lined model, a stage with a flat layer in it, and models off the canvas, across the camera and past the
    fog: the boxes, the triangle count and the sets check warns from are the Skia path's, to the last digit."""
    spec = scene([{"id": "st", "type": "group", "sort": "pixel", "children": [
                      ring(), ball(x=200), {"id": "card", "type": "rect", "x": 90, "y": 60, "w": 80, "h": 50,
                                            "fill": "#33323a"}]},
                  ball(id="grows", x=60, y=60, w=60, draw=[[0, 0], [2, 1, "linear"]]),
                  ball(id="off", x=900), ball(id="near", z=1100, w=400), ball(id="far", x=40, y=200, z=-9000)],
                 duration=2)
    spec.update(camera={"rotateX": 8}, fog={"near": 1500, "far": 6000})
    for n in (0, 20, 45):
        _, a, seen_a = tracked(spec, monkeypatch, "skia", n)
        _, b, seen_b = tracked(spec, monkeypatch, "auto", n)
        assert not a.on_gl and b.on_gl >= {"ring", "ball", "far"}
        assert seen_a == seen_b
    assert "grows" in b.on_gl and b.tris and "far" in b.fogged and "near" in b.culled


def test_check_sees_the_pixels_a_render_draws(monkeypatch):
    """Tracked, a stage and its Skia layers stay at 1x while OpenGL keeps 2x: the frame is the render's."""
    spec = scene([{"id": "st", "type": "group", "sort": "pixel", "children": [ring(), ball(x=200)]}, ball(id="b2", x=70)])
    a = renderer(spec, monkeypatch).frame(10)[..., :3].astype(int)
    b, r, _ = tracked(spec, monkeypatch, "auto", 10)
    assert r.on_gl == {"ring", "ball", "b2"}
    close(a, b, mean=0.2, far=0.005)
    assert (np.abs(b - np.array([0xf4, 0xef, 0xe6])).max(-1) > 30).mean() > 0.1  # the models are in the frame


class _SkiaMap:
    """gl3d as stage.py sees it, with the shadow map left to Skia: what a stage drew before OpenGL made the map."""

    def __getattr__(self, name):
        return getattr(gl3d, name)

    def available(self, r):
        return False


def lit(kids, shadows=True, x=160, w=300):
    """A stage of kids over a floor w wide."""
    floor = {"id": f"floor{x}", "type": "rect", "w": w, "h": 300, "y": 40, "rotateX": 90, "fill": "#808080"}
    return {"id": f"st{x}", "type": "group", "x": x, "y": 110, "sort": "pixel", "shadows": shadows,
            "children": kids + [floor]}


def lit_scene(stages, **more):
    """Stages seen from above, the key light from the left."""
    spec = scene(stages, bg="#000000")
    spec.update({"light": {"azimuth": -70, "elevation": 40, "follow": "world"}, "camera": {"rotateX": 35}}, **more)
    return spec


def nearness(image) -> np.ndarray:
    """A shadow map's texels as numbers: 0 where nothing casts, up to 1 nearest the light."""
    px = image.toarray(colorType=skia.kRGBA_8888_ColorType).astype(np.int64)
    return ((px[..., 0] << 16) | (px[..., 1] << 8) | px[..., 2]) / 16777215.0


def cast(monkeypatch) -> list:
    """Count what gl3d.sun_map draws from here on: [casters from a model's buffers, casters sent loose, maps]."""
    n, real_map, real_shadow = [0, 0, 0], gl3d.sun_map, gl3d._Buffers.shadow

    def sun_map(r, casters, *a):
        before = n[0]
        out = real_map(r, casters, *a)
        n[1] += len(casters) - (n[0] - before)
        n[2] += 1
        return out

    def shadow(self, order):
        n[0] += 1
        return real_shadow(self, order)
    monkeypatch.setattr(gl3d, "sun_map", sun_map)
    monkeypatch.setattr(gl3d._Buffers, "shadow", shadow)
    return n


def shadowed(spec, monkeypatch, n=0, skia_map=False):
    """Frame n with its models on OpenGL, the shadow map there too or (skia_map) left to Skia, and the renderer."""
    with monkeypatch.context() as m:
        if skia_map:
            m.setattr(stage, "gl3d", _SkiaMap())
        r = renderer(spec, monkeypatch)
        return r.frame(n)[..., :3].astype(int), r


def same_shadows(a, b, bare):
    """Frames a and b hold the same shadows, and there are shadows: a few pixels may differ, where two ways of
    filling a triangle put a shadow's edge a texel apart."""
    d = np.abs(a - b).max(-1)
    dark = ((bare - b).min(-1) > 20).sum()
    assert dark > 200 and (d > 8).sum() < 0.01 * dark, ((d > 8).sum(), dark)


def test_a_shadow_map_drawn_on_opengl_casts_the_shadows_skias_does(monkeypatch):
    """A whole model and one drawn in part (from their buffers) and a transparent one (on the Skia path, so sent
    loose) cast onto a floor and each other: the frame is the one the Skia-drawn map gives."""
    kids = [ball(x=-70, y=-25, r=28), ring(x=10, y=-30, r=34, tube=10, rotateX=20, draw=[[0, 0.3], [2, 1, "linear"]]),
            ball(id="glass", x=85, y=-10, r=20, transparency=0.3, edges=None)]
    spec = lit_scene([lit(kids)])
    count = cast(monkeypatch)
    for n in (0, 30):
        a, _ = shadowed(spec, monkeypatch, n, skia_map=True)
        assert count == [0, 0, 0]
        b, r = shadowed(spec, monkeypatch, n)
        assert r.on_gl == {"ball", "ring"} and count == [2, 1, 1]
        same_shadows(a, b, shadowed(lit_scene([lit(kids, shadows=None)]), monkeypatch, n)[0])
        count[:] = [0, 0, 0]


def test_each_stage_and_each_sub_frame_reads_its_own_shadow_map(monkeypatch):
    """Two shadowed stages in a frame, and a frame of several sub-frames (motion blur): one texture holds every map,
    and each is read before the next is drawn into it. With transparent models, which Skia draws, no OpenGL model
    comes between the two maps."""
    def stages(shadows, glass):
        look = {"transparency": 0.2, "edges": None} if glass else {}
        return [lit([ball(id="a", x=-40, y=-30, r=22, z=[[0, -60], [2, 60, "linear"]], **look)], shadows, 80, 150),
                lit([ring(id="b", x=-35, y=-34, r=24, tube=8, rotateX=25, rotateY=[[0, 0], [2, 200, "linear"]],
                          **look)], shadows, 240, 150)]
    count = cast(monkeypatch)
    for glass, more in ((False, {}), (True, {}), (False, {"motionBlur": {"samples": 4, "shutter": 0.9}})):
        a, _ = shadowed(lit_scene(stages(True, glass), **more), monkeypatch, 20, skia_map=True)
        b, r = shadowed(lit_scene(stages(True, glass), **more), monkeypatch, 20)
        assert count[2] == (8 if more else 2) and r.on_gl == (set() if glass else {"a", "b"})
        bare = shadowed(lit_scene(stages(None, glass), **more), monkeypatch, 20)[0]
        same_shadows(a[:, :160], b[:, :160], bare[:, :160])  # each stage's half of the frame
        same_shadows(a[:, 160:], b[:, 160:], bare[:, 160:])
        count[:] = [0, 0, 0]


def test_a_caster_is_drawn_from_its_own_corners_once_its_buffers_hold_others(monkeypatch):
    """Moving copies write each frame's corners into the same buffers. What a model cast at one frame, mapped after
    the buffers have taken another frame's corners, is still what it cast: sent loose, as the map made at the time."""
    dots = {"id": "dots", "type": "model", "shape": "sphere", "r": 16, "y": -30, "fill": "#c03070",
            "repeat": {"count": 5, "x": "=i * 44 - 88", "z": "=sin(t * 3 + i) * 90"}}
    kept, real, count = [], gl3d.sun_map, cast(monkeypatch)
    counted = gl3d.sun_map

    def sun_map(r, casters, to_map, side):
        out = counted(r, casters, to_map, side)
        kept.append((list(casters), to_map, nearness(out)))
        return out
    monkeypatch.setattr(gl3d, "sun_map", sun_map)
    r = renderer(lit_scene([lit([dots])]), monkeypatch)
    r.frame(0)
    r.frame(25)
    assert count == [2, 0, 2]
    (casters, to_map, then), (_, _, later) = kept
    again = nearness(real(r, casters, to_map, stage.SUN))
    assert count == [2, 0, 2]  # nothing more from the buffers
    moved = (np.abs(then - later) > 1e-3).mean()
    assert moved > 0.02 and (np.abs(then - again) > 1e-3).mean() < 0.02 * moved


def test_check_and_a_render_draw_the_same_shadows(monkeypatch):
    """A tracked render casts from the triangles render3d finds, sent loose, and its stage is at 1x."""
    spec = lit_scene([lit([ball(x=-40, y=-25, r=28), ring(x=50, y=-30, r=30, tube=9, rotateX=20)])])
    count = cast(monkeypatch)
    a, _ = shadowed(spec, monkeypatch)
    assert count == [2, 0, 1]
    b, r, _ = tracked(spec, monkeypatch, "auto", 0)
    assert r.on_gl == {"ball", "ring"} and count == [2, 2, 2]
    bare, _ = shadowed(lit_scene([lit(spec["layers"][0]["children"][:2], shadows=None)]), monkeypatch)
    assert ((bare - b).min(-1) > 20).sum() > 300
    close(a, b, mean=0.3, far=0.005)


def test_a_shadowed_frame_is_the_same_whatever_was_drawn_before_it(monkeypatch):
    spec = lit_scene([lit([ball(x=-40, y=-25, r=28, z=[[0, -80], [2, 80, "linear"]]),
                           ring(x=50, y=-30, r=30, tube=9, rotateX=[[0, 0], [2, 300, "linear"]])])])
    r = renderer(spec, monkeypatch)
    first = r.frame(30)
    for n in (0, 59, 12):
        r.frame(n)
    assert np.array_equal(first, r.frame(30))
    assert np.array_equal(first, renderer(spec, monkeypatch).frame(30))


def test_skia_draws_the_shadow_map_when_opengl_is_off(monkeypatch):
    spec = lit_scene([lit([ball(x=-40, y=-25, r=28)])])
    count = cast(monkeypatch)
    renderer(spec, monkeypatch, "skia").frame(0)
    Renderer(k.Scene(spec), backend="cpu").frame(0)
    assert count == [0, 0, 0]
