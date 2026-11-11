"""Model layers on OpenGL (gl3d): which models draw there, that they match the Skia path, stages and their shadow
maps, a partial draw, moving copies, textures, see-through materials, repeatability, what check and --boxes read,
and KINE_3D. The drawing tests need the GPU backend and moderngl, and skip without them (CI runs on the CPU)."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript import gl3d, stage
from kinescript.render import Renderer
from test_models import write_gltf


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
    assert gl3d.fits([clear], [None], [None], 0.0, None)
    assert not gl3d.fits([mat, clear], [None] * 2, [None] * 2, 0.0, lines)  # lines behind it would not be found
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
    n, real_map, real_array = [0, 0, 0], gl3d.sun_map, gl3d._Buffers.array

    def sun_map(r, casters, *a):
        before = n[0]
        out = real_map(r, casters, *a)
        n[1] += len(casters) - (n[0] - before)
        n[2] += 1
        return out

    def array(self, name, m, order=None):
        n[0] += name == "sun"
        return real_array(self, name, m, order)
    monkeypatch.setattr(gl3d, "sun_map", sun_map)
    monkeypatch.setattr(gl3d._Buffers, "array", array)
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
    """A whole model and one drawn in part (from their buffers) and a see-through one with edge lines (on the Skia
    path, so sent loose) cast onto a floor and each other: the frame is the one the Skia-drawn map gives."""
    kids = [ball(x=-70, y=-25, r=28), ring(x=10, y=-30, r=34, tube=10, rotateX=20, draw=[[0, 0.3], [2, 1, "linear"]]),
            ball(id="glass", x=85, y=-10, r=20, transparency=0.3)]
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
    and each is read before the next is drawn into it. With see-through models with edge lines, which Skia draws,
    no OpenGL model comes between the two maps."""
    def stages(shadows, glass):
        look = {"transparency": 0.2, "edges": {"width": 2}} if glass else {}
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


# ---- textures ----

def picture(path, size=128, turned=False) -> str:
    """An image whose quarters are red, green, blue and clear, with a dark line along every fourth row: which way up
    it is read, where its edges fall, what shows through it and how sharply it is read when small all show.
    turned: the same image upside down."""
    h = size // 2
    px = np.zeros((size, size, 4), np.uint8)
    px[..., 3] = 255
    px[:h, :h, :3], px[:h, h:, :3], px[h:, :h, :3] = (230, 40, 40), (40, 200, 60), (40, 70, 230)
    px[::4, :, :3] //= 4
    px[h:, h:] = 0
    px = np.ascontiguousarray(px[::-1, ::-1]) if turned else px
    skia.Image.fromarray(px, skia.kRGBA_8888_ColorType).save(str(path))
    return str(path)


def card(texture, **kw):
    """A plane that faces the camera."""
    return dict({"id": "card", "type": "model", "shape": "plane", "w": 200, "h": 150, "x": 160, "y": 120,
                 "fill": "#c8a03c", "texture": texture}, **kw)


def quad(tmp_path, name, uv, extra=None, halves=None, **material):
    """A square model file that faces the camera, with picture() as its image 0 and the same upside down as its
    image 1: of one material, or its left and right halves of the two materials `halves`."""
    png = [open(picture(tmp_path / f"p{i}.png", turned=bool(i)), "rb").read() for i in range(2)]
    xs = [(-1, 1)] if halves is None else [(-1, 0), (0, 1)]
    prims = [([[a, 1, 0], [b, 1, 0], [b, -1, 0], [a, -1, 0]], [[0, 3, 2], [0, 2, 1]], [[0, 0, 1]] * 4, i,
              *(() if uv is None else (uv,))) for i, (a, b) in enumerate(xs)]
    src = write_gltf(tmp_path / name, prims, materials=halves or [material], binary=True, images=png, extra=extra)
    return {"id": "m", "type": "model", "src": src, "w": 160, "x": 160, "y": 120}


def same(a, b, what=None, levels=3):
    """No pixel more than `levels` apart: Skia keeps a textured model's light in 8 bits before it multiplies."""
    assert np.abs(a - b).max() <= levels, (what, int(np.abs(a - b).max()))
    assert (np.abs(b - b[0, 0]).max(-1) > 30).mean() > 0.02, what  # and the model is in the frame


def test_a_textured_model_that_faces_the_camera_is_the_skia_paths(tmp_path, monkeypatch):
    """With perspective out of it nothing tells the paths apart: the image the right way up, repeated and offset,
    its fill through its clear pixels, as sharp when it is drawn small, under more light than white takes and under
    none of its own (glow), in toon bands."""
    img = picture(tmp_path / "p.png")
    for kw in ({}, {"w": 90, "h": 70}, {"texture": {"src": img, "repeat": [3, 2], "offset": [0.25, 0.1]}},
               {"light": {"key": 1.6, "ambient": 1.4}, "fill": "#403020"}, {"glow": 1}, {"bands": 3, "shine": 0.9}):
        a, b, gl = both(scene([card(**dict({"texture": img}, **kw))]), monkeypatch)
        assert gl == {"card"}, kw
        same(a, b, kw)


def test_a_textured_model_turned_in_perspective_stays_within_a_pixel_of_the_skia_path(tmp_path, monkeypatch):
    """Skia spreads the image evenly over triangles it cuts until that is within a pixel of right; OpenGL places it
    at each pixel. They differ along the image's edges and nowhere else."""
    img = picture(tmp_path / "p.png")
    two = {"id": "card", "type": "model", "x": 160, "y": 120, "rotateX": 60, "parts": [
        {"shape": "sphere", "r": 50, "texture": {"src": img, "repeat": [3, 2]}},
        {"shape": "torus", "r": 80, "tube": 12, "fill": "#ffffff"}]}
    for layer in (card(img, rotateY=50, rotateX=20), card(img, shape="sphere", r=80, rotateY=30),
                  card(img, shape="cube", w=110, h=110, rotateY=35, rotateX=-25, edges=True), two):
        a, b, gl = both(scene([layer]), monkeypatch)
        assert gl == {"card"}
        close(a, b, mean=0.8, far=0.004)
        assert (np.abs(a - b).max(-1) > 16).mean() < 0.03


def test_a_model_files_tint_tiling_and_emission_texture_draw_as_on_skia(tmp_path, monkeypatch):
    """Coordinates that run a whole image past each edge show the tiling: repeated, mirrored, held at the edge,
    and one image tiled two ways by two materials. Then the image tinted, an image as the light a material gives
    off (the one it shows, then another over it, then one whose sampled mean is black, so that the material has no
    emission colour), and a file without coordinates."""
    wide, unit = [[-1, -1], [2, -1], [2, 2], [-1, 2]], [[0, 0], [1, 0], [1, 1], [0, 1]]
    pbr = {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1}
    layers = []
    for i, wrap in enumerate(((10497, 10497), (33648, 33071), (33071, 10497))):
        extra = {"samplers": [{"wrapS": wrap[0], "wrapT": wrap[1]}], "textures": [{"source": 0, "sampler": 0}]}
        layers.append(quad(tmp_path, f"wrap{i}.glb", wide, extra, pbrMetallicRoughness=pbr))
    extra = {"samplers": [{"wrapS": 33071, "wrapT": 33071}, {"wrapS": 33648, "wrapT": 33648}],
             "textures": [{"source": 0, "sampler": 0}, {"source": 0, "sampler": 1}]}
    layers.append(quad(tmp_path, "ways.glb", wide, extra, halves=[
        {"pbrMetallicRoughness": dict(pbr, baseColorTexture={"index": i})} for i in range(2)]))
    layers.append(quad(tmp_path, "tint.glb", unit, pbrMetallicRoughness=dict(pbr, baseColorFactor=[1, 0.6, 0.3, 1])))
    dark = {"baseColorFactor": [0.1, 0.1, 0.1, 1], "metallicFactor": 0, "roughnessFactor": 1}
    layers.append(quad(tmp_path, "lit.glb", unit, pbrMetallicRoughness=dark, emissiveFactor=[1, 0.5, 1],
                       emissiveTexture={"index": 0}))
    layers.append(quad(tmp_path, "over.glb", unit, pbrMetallicRoughness=pbr, emissiveFactor=[0.5, 0.5, 0.5],
                       emissiveTexture={"index": 1}))
    px = np.zeros((128, 128, 4), np.uint8)
    px[..., 3] = 255
    px[1::2, 1::2, :3] = 255  # the importer's mean reads the even rows and columns
    skia.Image.fromarray(px, skia.kRGBA_8888_ColorType).save(str(tmp_path / "dots.png"))
    (tmp_path / "q.obj").write_text("mtllib q.mtl\nv -1 1 0\nv 1 1 0\nv 1 -1 0\nv -1 -1 0\n"
                                    "vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nusemtl q\nf 1/1 4/4 3/3 2/2\n")
    (tmp_path / "q.mtl").write_text("newmtl q\nKd 0.2 0.1 0.4\nmap_Ke dots.png\n")
    layers.append(dict(layers[0], src=str(tmp_path / "q.obj")))
    layers.append(dict(quad(tmp_path, "bare.glb", None, pbrMetallicRoughness=dark),
                       texture=picture(tmp_path / "p.png")))
    seen = []
    for layer in layers:
        a, b, gl = both(scene([layer]), monkeypatch)
        assert gl == {"m"}, layer["src"]
        same(a, b, layer["src"])
        seen.append(b)
    assert all(np.abs(x - y).max() > 60 for i, x in enumerate(seen) for y in seen[:i])  # nine different pictures


def moving_ui():
    """A group to show on a model: a square that crosses it, under a band. The rest of it is clear."""
    return {"id": "ui", "type": "group", "w": 100, "h": 100, "x": 0, "y": 0, "children": [
        {"type": "rect", "x": 0, "y": 0, "w": 100, "h": 40, "anchor": "topleft", "fill": "#203040"},
        {"type": "rect", "x": [[0, 0], [2, 70]], "y": 55, "w": 30, "h": 30, "anchor": "topleft", "fill": "#e62828"}]}


def test_a_picture_that_changes_is_copied_again_and_one_that_is_shared_is_copied_once(tmp_path, monkeypatch):
    """A group drawn on a model is a new image each frame: each frame shows its own, in the texture the last one
    had, with nothing of the last left where it is clear. Two models that show one file share its copy."""
    img = picture(tmp_path / "p.png")
    spec = scene([moving_ui(), card("#ui", x=90, w=140, h=140), card(img, id="a", x=240, y=60, w=100, h=80),
                  card(img, id="b", x=240, y=180, w=100, h=80)])
    on_skia = renderer(spec, monkeypatch, "skia")
    wanted = [on_skia.frame(n)[..., :3].astype(int) for n in (0, 25, 50)]
    r = renderer(spec, monkeypatch)
    frames, textures = [], []
    for n, want in zip((0, 25, 50), wanted):
        frames.append(r.frame(n)[..., :3].astype(int))
        assert r.on_gl == {"card", "a", "b"}
        same(want, frames[-1], n)
        kept = r._gl_pictures
        assert sorted(hit[2] for hit in kept.copies.values()) == [1, 2] and not kept.spare
        textures.append({hit[2]: hit[1] for hit in kept.copies.values()})
    assert all(t[1] is textures[0][1] and t[2] is textures[0][2] for t in textures)
    assert np.abs(frames[0] - frames[2]).max() > 100  # the square moved on the model


def test_a_textured_frame_is_the_same_whatever_was_drawn_before_it(tmp_path, monkeypatch):
    img = picture(tmp_path / "p.png")
    spec = scene([moving_ui(), card("#ui", x=90, w=140, h=140, rotateY=[[0, -40], [2, 40]]),
                  card(img, id="ball", shape="sphere", r=50, x=240, rotateY=[[0, 0], [2, 200]])])
    r = renderer(spec, monkeypatch)
    first = r.frame(30).copy()
    r.frame(0)
    r.frame(45)
    assert r.on_gl == {"card", "ball"}
    assert np.array_equal(first, r.frame(30))
    assert np.array_equal(first, renderer(spec, monkeypatch).frame(30))


def test_textured_models_drawn_in_part_in_copies_and_in_a_lit_stage_match_the_skia_path(tmp_path, monkeypatch):
    img = picture(tmp_path / "p.png")
    part = card(img, shape="sphere", r=80, rotateY=30, draw=0.5)
    a, b, gl = both(scene([part]), monkeypatch)
    assert gl == {"card"}
    close(a, b, mean=0.8, far=0.01)
    whole = renderer(scene([dict(part, draw=1)]), monkeypatch).frame(0)[..., :3].astype(int)
    assert (np.abs(b - b[0, 0]).max(-1) > 30).mean() < 0.8 * (np.abs(whole - whole[0, 0]).max(-1) > 30).mean()
    dots = card(img, id="dots", shape="cube", w=40, h=40, rotateY=[[0, 0], [2, 180]], rotateX=20,
                repeat={"count": 5, "x": "=i * 56 - 112", "z": "=sin(t * 3 + i) * 90"})
    for n in (0, 30):
        a, b, gl = both(scene([dots]), monkeypatch, n)
        assert gl == {"dots"}
        close(a, b, mean=0.8, far=0.004)
    count = cast(monkeypatch)
    kids = [ball(x=-20, y=-30, r=40, texture=img, edges=None, bands=0)]
    a = shadowed(lit_scene([lit(kids)]), monkeypatch, skia_map=True)[0]
    assert count == [0, 0, 0]
    b, r = shadowed(lit_scene([lit(kids)]), monkeypatch)
    assert r.on_gl == {"ball"} and count == [1, 0, 1]  # its shadow from the buffers its picture is drawn from
    same_shadows(a, b, shadowed(lit_scene([lit(kids, shadows=None)]), monkeypatch)[0])


def test_check_reads_a_textured_model_as_from_the_skia_path(tmp_path, monkeypatch):
    """A floor that runs past the camera: Skia cuts its triangles until the image stays put, and check counts
    those and warns where that was not enough, whichever path draws the pixels."""
    img = picture(tmp_path / "p.png")
    spec = scene([card(img, id="floor", w=3000, h=3000, rotateX=85, y=200, z=150),
                  card(img, rotateY=40, w=120, h=90, y=60)])
    _, a, seen_a = tracked(spec, monkeypatch, "skia", 0)
    px, b, seen_b = tracked(spec, monkeypatch, "auto", 0)
    assert not a.on_gl and b.on_gl == {"floor", "card"}
    assert seen_a == seen_b and a.swim == b.swim == {"floor"}
    assert a.tris > 1000  # cut: the two planes are 8 triangles
    assert np.array_equal(renderer(spec, monkeypatch).frame(0)[..., :3], px)  # and the pixels are a render's


def test_what_a_textured_models_program_does_not_take_draws_with_skia(tmp_path, monkeypatch):
    img, other = picture(tmp_path / "p.png"), picture(tmp_path / "q.png", size=64)
    mat = ((0.5, 0.5, 0.5), 0.0, 0.4, 20.0, 0.0, 0.5, 0.0, (0.0, 0.0, 0.0), 1.0)
    two = {"id": "two", "type": "model", "x": 160, "y": 120, "parts": [
        {"shape": "sphere", "r": 40, "x": -50, "texture": img},
        {"shape": "sphere", "r": 40, "x": 50, "texture": other}]}
    r = renderer(scene([two, ball(x=40, y=40, w=40)]), monkeypatch)
    r.frame(0)
    assert r.on_gl == {"two", "ball"}
    tx = lambda i: (object(), (1.0, 1.0), (0.0, 0.0), None, ("repeat", "repeat"))  # noqa: E731
    one = tx(0)
    assert gl3d.fits([mat] * gl3d.TEX_MATS, [one] * gl3d.TEX_MATS, [one] * gl3d.TEX_MATS, 0.0, None)
    assert not gl3d.fits([mat] * (gl3d.TEX_MATS + 1), [one] + [None] * gl3d.TEX_MATS, [None] * 33, 0.0, None)
    assert gl3d.fits([mat] * 8, [tx(i) for i in range(8)], [tx(i) for i in range(8)], 0.0, None)
    assert not gl3d.fits([mat] * 9, [tx(i) for i in range(9)], [tx(i) for i in range(8)] + [None], 0.0, None)
    assert not gl3d.fits([mat], [one], [None], 0.3, None)
    monkeypatch.setattr(gl3d, "UNITS", 1)  # one picture a draw
    r.frame(0)
    assert r.on_gl == {"ball"}
    monkeypatch.undo()
    monkeypatch.setenv("KINE_3D", "auto")
    monkeypatch.setitem(gl3d._STATE, "tex", None)  # a driver that would not take the program
    r.frame(0)
    assert r.on_gl == {"ball"}


# ---- see-through materials ----

def glass(**kw):
    """A see-through torus, turned so that its far side shows through the near one."""
    return dict({"id": "glass", "type": "model", "shape": "torus", "r": 70, "tube": 28, "x": 160, "y": 120,
                 "rotateX": 60, "rotateY": 25, "fill": "#7b5cff", "shine": 0.8, "rim": 0.9, "transparency": 0.6}, **kw)


def case(**kw):
    """A see-through box with a solid ball in it and a see-through one beside it."""
    return dict({"id": "glass", "type": "model", "x": 150, "y": 120, "rotateY": 30, "rotateX": -15, "parts": [
        {"shape": "cube", "w": 110, "fill": "#9fd3ff", "transparency": 0.75, "shine": 0.9},
        {"shape": "sphere", "r": 30, "fill": "#ff633f"},
        {"shape": "sphere", "r": 22, "x": 100, "fill": "#b9f45d", "transparency": 0.3}]}, **kw)


def test_a_see_through_model_matches_the_skia_path(tmp_path, monkeypatch):
    """Drawn far to near over what is behind, its far side lit from inside: a torus seen through itself, a box with
    a solid ball in it, a plane (whose back is its other face: drawn once), a file's one-sided square from behind,
    a file's square that is red on one side and blue on the other, from each side, then in toon bands, glowing,
    under an environment, in fog, and as it goes from solid to see-through."""
    square = dict(quad(tmp_path, "sq.glb", None, pbrMetallicRoughness={"baseColorFactor": [0.9, 0.3, 0.2, 1]}),
                  id="glass", transparency=0.5, rotateY=150, rotateX=20)
    P, N = [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], [[0, 0, 1]] * 4
    sides = write_gltf(tmp_path / "sides.glb", [(P, [[0, 3, 2], [0, 2, 1]], N, 0), (P, [[0, 2, 3], [0, 1, 2]], N, 1)],
                       materials=[{"pbrMetallicRoughness": {"baseColorFactor": c, "metallicFactor": 0},
                                   "alphaMode": "BLEND"} for c in ([0.9, 0.1, 0.1, 0.6], [0.1, 0.1, 0.9, 0.6])],
                       binary=True)
    sides = {"id": "glass", "type": "model", "src": sides, "w": 150, "x": 160, "y": 120, "rotateX": 15}
    fading = glass(transparency=[[0, 0], [2, 0.8, "linear"]])
    for spec, n in ((scene([glass()]), 0), (scene([case()]), 0), (scene([square]), 0),
                    (scene([dict(sides, rotateY=25)]), 0), (scene([dict(sides, rotateY=205)]), 0),
                    (scene([glass(shape="plane", w=160, h=110, rotateX=20, rotateY=140)]), 0),
                    (scene([glass(bands=3, glow=0.5)]), 0), (dict(scene([case()]), env="studio"), 0),
                    (dict(scene([case(z=-600)]), fog={"near": 600, "far": 2400}), 0),
                    (scene([fading]), 0), (scene([fading]), 15)):
        a, b, gl = both(spec, monkeypatch, n)
        assert gl == {"glass"}
        same(a, b, (spec["layers"][0], n), levels=2)


def test_see_through_models_drawn_in_part_and_in_copies_match_the_skia_path(monkeypatch):
    part = glass(draw=[[0, 0.2], [2, 1, "linear"]])
    pane = glass(shape="plane", w=160, h=110, rotateX=20, rotateY=140, draw=[[0, 0.2], [2, 1, "linear"]])
    dots = glass(r=16, tube=7, repeat={"count": 5, "x": "=i * 56 - 112", "rotateY": "=t * 90 + i * 40"})
    for layer in (part, pane, dots):
        for n in (10, 40):
            a, b, gl = both(scene([layer]), monkeypatch, n)
            assert gl == {"glass"}
            same(a, b, n, levels=2)


def test_a_see_through_model_in_a_stage_merges_and_casts_as_on_the_skia_path(monkeypatch):
    """A card through a see-through ring in a stage: each pixel takes the nearer of the two, by the depth of the
    ring's nearest surface. And its shadow on a floor."""
    card = {"id": "card", "type": "rect", "x": 160, "y": 120, "w": 60, "h": 150, "fill": "#33323a", "rotateY": 60}
    spec = scene([{"id": "st", "type": "group", "sort": "pixel", "children": [glass(rotateX=75), card]}])
    a, b, gl = both(spec, monkeypatch)
    assert gl == {"glass"}
    close(a, b, mean=0.3, far=0.002)
    lone = both(scene([glass(rotateX=75), card]), monkeypatch)[1]
    assert (np.abs(b - lone).max(-1) > 40).mean() > 0.005  # the card passes behind part of the ring
    kids = [glass(x=-20, y=-40, r=30, tube=12)]
    count = cast(monkeypatch)
    a = shadowed(lit_scene([lit(kids)]), monkeypatch, skia_map=True)[0]
    b, r = shadowed(lit_scene([lit(kids)]), monkeypatch)
    assert r.on_gl == {"glass"} and count == [1, 0, 1]
    same_shadows(a, b, shadowed(lit_scene([lit(kids, shadows=None)]), monkeypatch)[0])


def test_what_draws_a_see_through_model(tmp_path, monkeypatch):
    """With edge lines it draws with Skia: OpenGL finds lines on the nearest surface, and those of what shows
    through would go. A textured model draws solid whatever its transparency, on OpenGL as with Skia."""
    r = renderer(scene([case(edges={"width": 2})]), monkeypatch)
    r.frame(0)
    assert r.on_gl == set()
    solid = card(picture(tmp_path / "p.png"), rotateY=30)
    a, b, gl = both(scene([dict(solid, transparency=0.6)]), monkeypatch)
    assert gl == {"card"}
    assert np.array_equal(b, renderer(scene([solid]), monkeypatch).frame(0)[..., :3])
    close(a, b, mean=0.8, far=0.004)


def test_a_see_through_frame_is_the_same_whatever_was_drawn_before_it(monkeypatch):
    spec = scene([glass(x=100, rotateY=[[0, 0], [2, 300, "linear"]]),
                  case(id="case", x=230, rotateY=[[0, 0], [2, -200, "linear"]], transparency=[[0, 0], [1, 0.5]])])
    r = renderer(spec, monkeypatch)
    first = r.frame(30).copy()
    for n in (0, 59, 12):
        r.frame(n)
    assert r.on_gl == {"glass", "case"}
    assert np.array_equal(first, r.frame(30))
    assert np.array_equal(first, renderer(spec, monkeypatch).frame(30))


def test_check_reads_a_see_through_model_as_from_the_skia_path(monkeypatch):
    spec = scene([glass(), case(id="case", x=240, scale=0.6)])
    _, a, seen_a = tracked(spec, monkeypatch, "skia", 0)
    px, b, seen_b = tracked(spec, monkeypatch, "auto", 0)
    assert not a.on_gl and b.on_gl == {"glass", "case"}
    assert seen_a == seen_b
    assert np.array_equal(renderer(spec, monkeypatch).frame(0)[..., :3], px)
