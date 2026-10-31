import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kinescript as k
from kinescript import ease, media, svgpath
from kinescript.props import Ctx, compile_prop
from kinescript.render import Renderer
from kinescript.timing import Timeline, TimingError
from kinescript.verify import check, probe


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30}, **canvas), "layers": layers}


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "3", "-pix_fmt", "yuv420p", "-shortest", str(p)],
                   check=True)
    return str(p)


def test_cue_reference_follows_upstream_duration():
    tl = Timeline(30)
    tl.add("a", 0.5, dur=2)
    tl.add("b", "a.out+0.25", dur=1, prev="a")
    tl.add("c", "prev.in+15f", until="b.out", prev="b")
    tl.solve()
    assert tl.resolved["b"] == (2.75, 3.75)
    assert tl.resolved["c"][0] == pytest.approx(3.25)


def test_a_group_can_end_with_one_of_its_children():
    tl = Timeline(30)
    tl.add("g", 1, until="kid.out")
    tl.add("kid", 0.5, dur=2, parent="g", default_end="g")
    tl.add("rest", 0, parent="g", default_end="g")  # no end of its own: it ends with the group
    tl.solve()
    assert tl.resolved["g"] == (1, 3.5) and tl.resolved["rest"] == (1, 3.5)


def test_timing_cycle_is_reported():
    tl = Timeline(30)
    tl.add("a", "b.in")
    tl.add("b", "a.in")
    with pytest.raises(TimingError, match="cycle"):
        tl.solve()


def test_unknown_reference_lists_known_ids():
    with pytest.raises(k.SceneError, match="Layer ids and markers: a, b"):
        k.Scene(scene([{"id": "a", "type": "rect"}, {"id": "b", "type": "rect", "at": "nope.out"}]))


def test_open_layers_end_with_scene():
    s = k.Scene(scene([{"id": "a", "type": "rect", "dur": 2}, {"id": "bg", "type": "rect"}]))
    assert s.duration == 2 and s.by_id["bg"].t_out == 2


def test_keyframes_negative_time_counts_from_layer_end():
    p = compile_prop("x", [[0, 0], [1, 100, "linear"], [-1, 100], ["end", 0, "linear"]], Ctx(100, 100, 30))
    assert p.at(0.5, 4) == pytest.approx(50)
    assert p.at(2.0, 4) == pytest.approx(100)
    assert p.at(3.5, 4) == pytest.approx(50)


def test_percent_lengths_use_matching_axis():
    ctx = Ctx(1920, 1080, 30)
    assert compile_prop("x", "50%", ctx).value == 960
    assert compile_prop("y", "50%+20", ctx).value == 560
    assert compile_prop("size", "10%", ctx).value == 108


def test_expression_rejects_attribute_access():
    with pytest.raises(ValueError, match="not allowed"):
        compile_prop("x", "=(1).__class__", Ctx(10, 10, 30))
    with pytest.raises(ValueError, match="unknown name"):
        compile_prop("x", "=open('f')", Ctx(10, 10, 30))
    assert compile_prop("x", "=clamp(t*2, 0, 1)", Ctx(10, 10, 30)).at(0.25, 1) == 0.5


def test_huge_powers_and_repeated_lists_fail_fast():
    # Exact integer power made 9**9**9 run for hours inside check; '**' is a float power now.
    assert compile_prop("x", "=2**3 + 4**0.5", Ctx(10, 10, 30)).at(0, 1) == 10
    with pytest.raises(k.SceneError, match="OverflowError"):
        k.Scene(scene([{"id": "c", "type": "circle", "r": 10, "x": "=9**9**9"}]))
    with pytest.raises(ValueError, match="not on text or lists"):
        compile_prop("x", "=[0]*999999999", Ctx(10, 10, 30))


def test_an_expression_that_fails_at_one_frame_is_reported_by_check():
    r = check(scene([{"id": "c", "type": "circle", "r": 10, "x": "=100+1/(t-0.7)", "y": 50}], duration=1))
    assert [(i["code"], i["layer"]) for i in r["issues"] if i["level"] == "error"] == [("bad-expression", "c")]


def test_an_expression_failing_mid_render_names_its_layer():
    s = k.Scene(scene([{"id": "c", "type": "circle", "r": 10, "x": "=100+1/(t-0.71)", "y": 50}], duration=1))
    with pytest.raises(k.SceneError, match=r"layer 'c': x failed at t=0\.710s"):
        Renderer(s, backend="cpu").draw(0.71)  # between frames, so check could not see it


def test_colour_tweens_animate_fill_and_stroke():
    tween = {"from": "#ff0000", "to": "#0000ff", "dur": 1}
    spec = scene([{"id": "r", "type": "rect", "x": "50%", "y": "50%", "w": 200, "h": 100, "fill": tween,
                   "stroke": dict(tween, to="#00ff00"), "strokeWidth": 20}], bg="#000000", duration=2)
    r = Renderer(k.Scene(spec), backend="cpu")
    assert tuple(r.frame(0)[90, 160][:3]) == (255, 0, 0)  # the fill used to draw as a flat grey gradient
    f = r.frame(40)
    assert tuple(f[90, 160][:3]) == (0, 0, 255) and tuple(f[90, 60][:3]) == (0, 255, 0)  # the stroke used to raise


def test_a_css_gradient_stroke_draws():
    spec = scene([{"id": "r", "type": "rect", "x": "50%", "y": "50%", "w": 200, "h": 100, "strokeWidth": 20,
                   "stroke": "linear-gradient(to right, #ff0000, #0000ff)"}], bg="#000000")
    f = _frame(spec, 0)
    assert f[90, 61][0] > 200 > f[90, 61][2] and f[90, 258][2] > 200 > f[90, 258][0]


def test_colour_keyframes_mix_in_oklab():
    p = compile_prop("fill", [[0, "#ff0000"], [1, "#0000ff", "linear"]], Ctx(10, 10, 30))
    mid = p.at(0.5, 1)
    assert mid[0] > 0.4 and mid[2] > 0.4  # stays saturated instead of a dark sRGB midpoint


def test_eases_hit_endpoints():
    for name in ("outCubic", "inOutQuint", "outBack", "spring", "bezier(.2,.8,.2,1)", "steps(4)", "outBounce"):
        f = ease.get(name)
        assert f(0) == pytest.approx(0, abs=1e-3) and f(1) == pytest.approx(1, abs=2e-3), name


def test_svg_path_parses_relative_and_arcs():
    p = svgpath.parse("M10 10 h80 v80 h-80 z M 50 50 a 20 20 0 1 0 1 0")
    b = p.computeTightBounds()
    assert b.left() == pytest.approx(10) and b.right() == pytest.approx(90)


def _frame(spec, n, backend="cpu"):
    r = Renderer(k.Scene(spec), backend=backend)
    return r.frame(n).copy()


def test_frame_is_pure_function_of_index():
    spec = scene([
        {"id": "t", "type": "text", "text": "Hi **there**", "x": "50%", "y": "50%", "size": 40, "split": "chars",
         "enter": {"name": "rise+blur", "stagger": 0.05}},
        {"id": "c", "type": "circle", "r": 10, "x": "=160+cos(t*3)*100", "y": 40, "fill": "#ff5a36"},
    ], duration=2)
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 30)]
    alone = _frame(spec, 17)
    assert np.array_equal(sequential[17], alone)


def _boxes(spec, T=0.0):
    r = Renderer(k.Scene(spec), track=True, backend="cpu")
    r.draw(T)
    return r


def test_depth_frame_is_pure_function_of_index():
    spec = scene([
        {"id": "t", "type": "text", "text": "Flip", "x": "50%", "y": "40%", "size": 40, "split": "chars",
         "enter": {"name": "flip", "stagger": 0.05}},
        {"id": "card", "type": "rect", "x": "50%", "y": "70%", "w": 120, "h": 40, "rotateY": "=t*120",
         "z": "=sin(t*3)*40"},
    ], duration=2)
    spec["camera"] = {"z": [[0, 60], ["end", -30]], "rotateY": [[0, -10], ["end", 10]]}
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 30)]
    assert np.array_equal(sequential[17], _frame(spec, 17))


def test_default_camera_leaves_flat_layers_unchanged():
    flat = scene([{"type": "rect", "x": 160, "y": 90, "w": 100, "h": 60, "fill": "#3ddbd9", "radius": 8, "rotate": 12},
                  {"type": "text", "text": "Same", "x": "50%", "y": "30%", "size": 30, "shadow": {"blur": 8}}])
    assert np.array_equal(_frame(flat, 0), _frame(dict(flat, camera={}), 0))


def test_rotate_y_turns_the_right_edge_away_and_backface_hides_it():
    r = _boxes(scene([{"id": "a", "type": "rect", "x": 160, "y": 90, "w": 100, "h": 60, "rotateY": 60}]))
    left, top, right, bottom = r.boxes["a"]["rect"]
    assert 45 < right - left < 58  # cos 60 halves the width; perspective widens the near half
    assert bottom - top > 65 and (left + right) / 2 < 159  # the near (left) edge is taller
    r = _boxes(scene([{"id": "a", "type": "rect", "x": 160, "y": 90, "w": 100, "h": 60, "rotateY": 180,
                       "backface": "hidden"},
                      {"id": "b", "type": "rect", "x": 160, "y": 90, "w": 100, "h": 60, "rotateY": 180}]))
    assert "a" not in r.boxes and r.boxes["b"]["mirrored"]


def test_group_sort_depth_draws_far_to_near():
    kids = [{"id": "near", "type": "rect", "x": 0, "y": 0, "w": 80, "h": 80, "z": 50, "fill": "#0000ff"},
            {"id": "far", "type": "rect", "x": 0, "y": 0, "w": 80, "h": 80, "z": -50, "fill": "#ff0000"}]
    listed = scene([{"type": "group", "x": 160, "y": 90, "children": kids}], bg="#000000")
    by_depth = scene([{"type": "group", "x": 160, "y": 90, "sort": "depth", "children": kids}], bg="#000000")
    assert tuple(_frame(listed, 0)[90, 160, :3]) == (255, 0, 0)
    assert tuple(_frame(by_depth, 0)[90, 160, :3]) == (0, 0, 255)


def test_camera_pan_moves_layers_but_not_fixed_ones():
    spec = scene([{"id": "a", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40},
                  {"id": "b", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40, "fixed": True}])
    spec["camera"] = {"x": 50}
    r = _boxes(spec)
    assert r.boxes["a"]["rect"][0] == pytest.approx(90, abs=0.5)
    assert r.boxes["b"]["rect"][0] == pytest.approx(140, abs=0.5)


def test_camera_orbit_and_dolly_bring_layers_closer():
    for cam in ({"rotateY": 30}, {"z": -100}):
        spec = dict(scene([{"id": "r", "type": "rect", "x": 240, "y": 90, "w": 40, "h": 40}]), camera=cam)
        left, top, right, bottom = _boxes(spec).boxes["r"]["rect"]
        assert bottom - top > 44, cam


def test_layer_behind_camera_is_skipped_and_check_warns():
    spec = scene([{"id": "a", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40, "z": 400}])
    r = _boxes(spec)
    assert "a" in r.culled and "a" not in r.boxes
    assert any(i["code"] == "behind-camera" and i["layer"] == "a" for i in check(spec)["issues"])


def test_mirrored_text_warns():
    spec = scene([{"id": "t", "type": "text", "text": "Hi", "x": "50%", "y": "50%", "size": 40, "rotateY": 180}])
    assert any(i["code"] == "mirrored-text" and i["layer"] == "t" for i in check(spec)["issues"])


def _model(layer, **canvas):
    return scene([dict({"id": "m", "type": "model", "x": 160, "y": 90}, **layer)], **dict({"bg": "#000000"}, **canvas))


def test_model_frame_is_pure_function_of_index():
    spec = _model({"shape": "torus", "r": 50, "detail": "low", "rotateX": "=t*90", "rotateY": "=t*40",
                   "enter": "build", "explode": [[1, 0], ["end", 0.6]],
                   "parts": [{"shape": "sphere", "r": 20, "fill": "#b9f45d"},
                             {"shape": "text", "text": "3D", "size": 40, "depth": 12, "bevel": 2, "z": 50}]},
                  duration=2)
    spec["camera"] = {"rotateY": [[0, -10], ["end", 10]]}
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 50)]
    assert np.array_equal(sequential[5], _frame(spec, 5)) and np.array_equal(sequential[45], _frame(spec, 45))


def _shades(frame):
    """The most common colours of the non-background pixels, and the share of pixels they cover."""
    px = frame[..., :3].reshape(-1, 3)
    px = px[px.any(1)]
    cols, counts = np.unique(px, axis=0, return_counts=True)
    top = np.argsort(-counts)
    return cols[top], counts[top] / counts.sum()


def test_cube_seen_from_a_corner_shows_three_differently_lit_faces():
    spec = _model({"shape": "cube", "w": 90, "detail": "low", "rotateX": -35, "rotateY": 45})
    r = Renderer(k.Scene(spec), backend="cpu")
    r.draw(0)
    assert r.tris == 3 * 18  # low detail: 3 x 3 cells, 18 triangles a face
    cols, share = _shades(r.frame(0))
    assert share[:3].sum() > 0.85 and min(np.abs(cols[i].astype(int) - cols[j]).max()
                                          for i, j in ((0, 1), (0, 2), (1, 2))) > 10
    cols, share = _shades(_frame(_model({"shape": "cube", "w": 90, "rotateX": -35, "rotateY": 45, "light": "flat",
                                         "fill": "#3366cc"}), 0))
    assert tuple(cols[0]) == (0x33, 0x66, 0xcc) and share[0] > 0.85  # flat light: the fill, unshaded


def test_model_parts_hide_each_other_by_depth_not_list_order():
    spec = _model({"light": "flat", "parts": [{"shape": "sphere", "r": 30, "z": 80, "fill": "#ff0000"},
                                              {"shape": "cube", "w": 100, "fill": "#0000ff"}]})
    f = _frame(spec, 0)
    assert tuple(f[90, 160, :3]) == (255, 0, 0)  # the sphere is in front, though listed first
    assert tuple(f[90, 160 + 55, :3]) == (0, 0, 255)


def test_model_build_reveals_triangles_and_explode_spreads_them():
    full = _boxes(_model({"shape": "sphere", "r": 40, "detail": "low"}))
    half = _boxes(_model({"shape": "sphere", "r": 40, "detail": "low", "draw": 0.5}))
    burst = _boxes(_model({"shape": "sphere", "r": 40, "detail": "low", "explode": 0.5}))
    assert 0.3 < half.tris / full.tris < 0.7
    width = lambda r: r.boxes["m"]["rect"][2] - r.boxes["m"]["rect"][0]  # noqa: E731
    assert 80 < width(full) < 100 and width(burst) > width(full) * 1.2


def test_extruded_caps_fill_the_outline_and_its_holes():
    import skia
    from kinescript import mesh
    p = skia.Path()
    p.addRect(skia.Rect(0, 0, 200, 100))
    p.addCircle(50, 50, 30, skia.PathDirection.kCCW)
    p.addCircle(150, 50, 30, skia.PathDirection.kCCW)
    p.addCircle(150, 50, 10)  # an island inside the second hole
    Q, starts, caps = mesh.shape2d(mesh.contours(p, 4.0))
    a, b, c = Q[caps[:, 0]], Q[caps[:, 1]], Q[caps[:, 2]]
    cross = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
    outline = sum(mesh._area(Q[s:e]) for s, e in zip(starts, [*starts[1:], len(Q)]))
    assert outline == pytest.approx(200 * 100 - 2 * np.pi * 30 ** 2 + np.pi * 10 ** 2, rel=0.01)
    assert (cross > -1e-9).all() and cross.sum() / 2 == pytest.approx(outline)  # no gaps, no overlaps
    o = _frame(_model({"shape": "text", "text": "O", "size": 120, "depth": 30, "light": "flat", "fill": "#3366cc"}), 0)
    assert tuple(o[90, 160, :3]) == (0, 0, 0) and (o[90, :, :3] == (0x33, 0x66, 0xcc)).all(-1).sum() > 20


def test_extruded_front_cap_is_never_hidden_by_its_own_walls():
    def cap_pixels(depth, colour=None):
        # The part sits so its front cap stays in one plane whatever the depth, so only the walls change.
        f = _frame(_model({"x": 320, "y": 180, "rotateY": 60, "rotateX": 40, "parts": [
            {"shape": "text", "text": "e%", "size": 220, "weight": 800, "depth": depth, "z": -depth / 2}]},
            w=640, h=360), 0)[..., :3]
        if colour is None:  # a 1 px deep solid is nearly all cap
            colour = _shades(f)[0][0]
        return int((f == colour).all(-1).sum()), colour
    thin, colour = cap_pixels(1)
    deep, _ = cap_pixels(120, colour)
    assert thin > 5000 and abs(deep - thin) <= 2  # with the caps' long ear-clipped triangles it loses about 46
    # ...and other parts still sort against the caps by depth: the ball's front is nearer than the caps
    # around it, though the turned text's far ends come nearer still.
    f = _frame(_model({"light": "flat", "rotateY": 35, "parts": [
        {"shape": "text", "text": "HHH", "size": 110, "weight": 800, "depth": 40, "fill": "#0000ff"},
        {"shape": "sphere", "r": 40, "fill": "#ff0000"}]}), 0)
    assert tuple(f[90, 160, :3]) == (255, 0, 0)


def test_model_crossing_the_camera_drops_triangles_and_check_warns(monkeypatch):
    spec = _model({"shape": "plane", "w": 200, "h": 900, "y": 140, "rotateX": 80}, duration=0.5)  # a floor
    r = _boxes(spec)
    assert "m" in r.culled and "m" in r.boxes  # its near end is dropped, its far end still draws
    issues = check(spec)["issues"]
    assert any(i["code"] == "behind-camera" and "triangles" in i["message"] for i in issues)
    from kinescript import verify
    monkeypatch.setattr(verify, "TRIANGLE_BUDGET", 50)
    assert any(i["code"] == "triangle-budget" for i in check(_model({"shape": "sphere"}, duration=0.5))["issues"])


def test_gpu_and_cpu_agree_roughly():
    spec = scene([{"type": "rect", "x": 160, "y": 90, "w": 100, "h": 60, "fill": "#3ddbd9", "radius": 8}])
    cpu = _frame(spec, 0, "cpu").astype(int)
    try:
        gpu = _frame(spec, 0, "auto").astype(int)
    except RuntimeError:
        pytest.skip("no GPU")
    assert np.abs(cpu - gpu).mean() < 1.0


def test_draw_trims_stroke():
    spec = scene([{"type": "line", "points": [[10, 90], [310, 90]], "stroke": "#ffffff", "strokeWidth": 6,
                   "draw": [[0, 0], [1, 1, "linear"]], "cap": "butt"}], bg="#000000", duration=1)
    f = _frame(spec, 15)  # half drawn
    row = f[90, :, 0]
    assert row[50] > 200 and row[250] < 20


def test_check_flags_offscreen_text_and_overlap():
    res = check(scene([
        {"id": "a", "type": "text", "text": "Far too wide a title", "x": 160, "y": 90, "size": 60},
        {"id": "b", "type": "text", "text": "Overlap", "x": 160, "y": 95, "size": 30},
    ], duration=2))
    codes = {i["code"] for i in res["issues"]}
    assert "text-offscreen" in codes and "text-overlap" in codes and not res["ok"]


def test_check_flags_media_mostly_off_canvas(clip):
    # A video sits by its top-left corner, so x and y at the middle without an anchor push it off to the bottom right.
    placed = {"id": "v", "type": "video", "src": clip, "x": "50%", "y": "50%", "w": 320, "h": 180, "dur": 2}
    res = check(scene([placed], duration=2))
    hit = [i for i in res["issues"] if i["code"] == "media-offscreen"]
    assert len(hit) == 1 and hit[0]["layer"] == "v" and hit[0]["level"] == "warning" and "anchor" in hit[0]["fix"]
    assert "75% off" in hit[0]["message"] and hit[0]["to"] - hit[0]["from"] >= 1.5
    for fixed in ({"anchor": "center"}, {"allowOffscreen": True}):
        res = check(scene([dict(placed, **fixed)], duration=2))
        assert not [i for i in res["issues"] if i["code"] == "media-offscreen"], fixed
    # A sequence's clips get its x, y and anchor, so they are reported once, as the sequence; its allowOffscreen
    # covers them.
    seq = {"id": "cut", "type": "sequence", "x": "50%", "y": "50%", "w": 320, "h": 180,
           "clips": [{"src": clip, "trim": [0, 1.5]}, {"src": clip, "trim": [1.5, 3]}]}
    res = check(scene([seq], duration=3))
    hit = [i for i in res["issues"] if i["code"] == "media-offscreen"]
    assert [i["layer"] for i in hit] == ["cut"] and hit[0]["to"] > 2.5
    res = check(scene([dict(seq, allowOffscreen=True)], duration=3))
    assert not [i for i in res["issues"] if i["code"] == "media-offscreen"]


def test_check_lets_media_pass_through_the_edge():
    # Sliding in from off the canvas is mostly off for under a second, which is not a misplaced layer.
    shader = {"id": "s", "type": "shader", "preset": "score", "w": 320, "h": 180, "dur": 2}
    res = check(scene([dict(shader, x=[[0, -320], [0.5, 0, "outCubic"]])], duration=2))
    assert not [i for i in res["issues"] if i["code"] == "media-offscreen"]
    res = check(scene([dict(shader, x=-240)], duration=2))  # left there, the same layer is flagged
    assert [i["layer"] for i in res["issues"] if i["code"] == "media-offscreen"] == ["s"]


def test_check_flags_low_contrast():
    res = check(scene([{"id": "a", "type": "text", "text": "Grey", "x": 160, "y": 90, "size": 30, "fill": "#20232a"}],
                      bg="#1b1d22", duration=2))
    assert any(i["code"] == "contrast" for i in res["issues"])


def test_check_reports_load_error_instead_of_raising():
    res = check(scene([{"type": "rect", "at": "ghost.out"}]))
    assert res["errors"] == 1 and res["issues"][0]["code"] == "load"


def test_probe_returns_boxes():
    out = probe(scene([{"id": "r", "type": "rect", "x": 100, "y": 50, "w": 40, "h": 20}]), at=0.5)
    assert out["visible"]["r"]["rect"] == [80.0, 40.0, 120.0, 60.0]


def test_components_expand_and_resolve():
    s = k.Scene(scene([{"use": "lowerThird", "id": "lt", "title": "Ada", "subtitle": "Engineer", "dur": 3},
                       {"use": "titleCard", "id": "tc", "title": "Hi", "at": "lt.out"}]))
    assert s.by_id["lt-title"].t_in == pytest.approx(0.12)
    assert s.by_id["tc"].t_in == 3


def test_theme_names_work_inside_a_scenes_own_components():
    spec = scene([{"use": "tag", "id": "t1", "word": "Hi"}, {"type": "group", "id": "g", "children": [
        {"use": "tag", "id": "t2", "word": "Yo"}]}])
    spec["theme"] = {"brand": "#123456"}
    spec["components"] = {"tag": {"params": {"word": ""}, "layers": [
        {"id": "w", "type": "text", "text": "{{word}}", "fill": "$muted", "stroke": "$brand"}]}}
    s = k.Scene(spec)
    assert [(s.by_id[i].raw["fill"], s.by_id[i].raw["stroke"]) for i in ("t1-w", "t2-w")] == [("#9299a8", "#123456")] * 2


def test_a_component_holding_a_group_can_be_used_twice():
    spec = scene([{"use": "card", "id": "a", "label": "A"}, {"use": "card", "id": "ab", "label": "B"}])
    spec["components"] = {"card": {"params": {"label": ""}, "layers": [
        {"id": "box", "type": "group", "children": [{"id": "abc", "type": "text", "text": "{{label}}"},
                                                    {"type": "group", "children": [{"id": "dot", "type": "circle", "r": 4}]}]}]}}
    s = k.Scene(spec)
    assert {"a-box", "a-abc", "a-dot", "ab-box", "ab-abc", "ab-dot"} <= set(s.by_id)
    assert s.by_id["ab-abc"].raw["text"] == "B"


def test_sequence_crossfades_and_video_renders(clip):
    spec = scene([{"type": "sequence", "id": "cut", "clips": [{"src": clip, "trim": [0, 1.5]}, {"src": clip, "trim": [1, 3]}],
                   "transition": {"type": "fade", "dur": 0.5}}])
    s = k.Scene(spec)
    assert s.by_id["cut-2"].t_in == pytest.approx(1.0) and s.duration == pytest.approx(3.0)
    f = Renderer(s, backend="cpu").frame(45)
    assert f[..., :3].std() > 20  # test pattern is visible, not a black frame
    s.close()


def test_clip_decoders_close_once_their_clip_ends(clip):
    spec = scene([{"type": "sequence", "id": "cut", "clips": [{"src": clip, "trim": [k * 0.5, k * 0.5 + 0.5]}
                                                              for k in range(4)]}])
    s = k.Scene(spec)
    r = Renderer(s, backend="cpu")
    for n in range(s.frames):
        r.frame(n)
    open_ = [L.id for L in s.all_layers() if L.video is not None and L.video.proc is not None]
    s.close()
    assert open_ == ["cut-4"]  # only the clip still on screen


def test_grade_keeps_a_transparent_canvas_transparent():
    layers = [{"id": "r", "type": "rect", "x": "50%", "y": "50%", "w": 100, "h": 60, "fill": "#4a7dff"}]
    grade = {"vignette": 0.6, "grain": 0.2}
    clear = Renderer(k.Scene(dict(scene(layers, bg="transparent"), grade=grade)), backend="cpu").frame(3).copy()
    solid = Renderer(k.Scene(dict(scene(layers, bg="#000000"), grade=grade)), backend="cpu").frame(3)
    assert clear[..., 3][:20, :20].max() == 0 and clear[..., 3][90, 160] == 255
    assert np.array_equal(clear[60:120, 110:210], solid[60:120, 110:210])  # the rect is graded as before


def test_grain_moves_on_every_frame():
    # 123 / 30 * 30 is 122.99999999999999: the grain tile used to be picked by truncating it, so frame 123
    # repeated frame 122's grain.
    dot = {"id": "d", "type": "circle", "r": 2, "x": 4, "y": 4, "fill": "#808080"}
    spec = dict(scene([dot], w=64, h=36, bg="#808080", duration=5), grade={"grain": 0.5})
    r = Renderer(k.Scene(spec), backend="cpu")
    assert not np.array_equal(r.frame(122).copy(), r.frame(123))


def test_sequence_id_is_a_cue_and_takes_layer_effects(clip):
    seq = {"type": "sequence", "id": "cut", "at": 0.5,
           "clips": [{"src": clip, "trim": [0, 1]}, {"src": clip, "trim": [1, 2.5]}]}
    cap = {"id": "cap", "type": "text", "text": "after", "x": "50%", "y": "50%", "size": 30, "at": "cut.out", "dur": 1}
    s = k.Scene(scene([seq, cap], bg="#000000"))
    assert (s.by_id["cut"].t_in, s.by_id["cut"].t_out) == (0.5, pytest.approx(3.0))  # the span of its clips
    assert s.by_id["cut-1"].t_in == 0.5 and s.by_id["cap"].t_in == pytest.approx(3.0)
    full = Renderer(s, backend="cpu").frame(30)
    s.close()
    assert check(scene([dict(seq, opacity=0.5, blur=2, fx={"exposure": -1}), cap]))["errors"] == 0
    s = k.Scene(scene([dict(seq, opacity=0.5)], bg="#000000"))
    half = Renderer(s, backend="cpu").frame(30)
    s.close()
    assert half[..., :3].mean() == pytest.approx(full[..., :3].mean() / 2, rel=0.05)


def test_missing_system_gl_libraries_name_the_packages(tmp_path):
    # What a bare Linux container gets from skia-python, standing in for the real library here.
    (tmp_path / "skia.py").write_text('raise ImportError("libEGL.so.1: cannot open shared object file")')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), root]))
    r = subprocess.run([sys.executable, "-c", "import kinescript"], env=env, capture_output=True, text=True)
    assert "libEGL.so.1" in r.stderr and "sudo apt install libegl1 libgl1 libfontconfig1" in r.stderr


def test_emphasis_can_start_and_stop_inside_a_word():
    s = k.Scene(scene([{"id": "t", "type": "text", "text": "g**P**u(doom) **diff**, done", "size": 30}]))
    L, r = s.by_id["t"], Renderer(s, backend="cpu")
    lay = r._text_layout(L, 0, 0, lambda name, default=None: L.get(name, 0, 0, 0, default), L.raw["text"])
    assert [(u.text, u.emph) for u in lay.runs] == [("g", False), ("P", True), ("u(doom)", False),
                                                    ("diff", True), (",", False), ("done", False)]
    assert lay.runs[0].x < lay.runs[1].x < lay.runs[2].x and lay.runs[2].char0 == 2


def test_render_mp4_with_audio(clip, tmp_path):
    out = tmp_path / "o.mp4"
    spec = scene([{"type": "video", "src": clip, "trim": [0, 1]},
                  {"type": "text", "text": "Title", "x": "50%", "y": "50%", "size": 30, "enter": "pop"}])
    res = k.render(spec, str(out), quality="draft")
    assert res["frames"] == 30
    info = media.probe(str(out))
    assert info["audio"] and info["duration"] == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("workers", [1, 3])
def test_audio_lands_at_its_time_with_any_worker_count(clip, tmp_path, workers):
    spec = scene([{"type": "rect", "w": 320, "h": 180, "fill": "#202020"}], duration=2)
    spec["audio"] = [{"src": clip, "at": 1, "trim": [0, 1]}]
    out = tmp_path / "o.mp4"
    assert k.render(spec, str(out), quality="draft", workers=workers)["frames"] == 60
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(out), "-map", "0:a", "-ac", "1", "-ar", "8000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, np.float32)
    assert np.sqrt(np.mean(pcm[:7200] ** 2)) < 1e-3 < 0.05 < np.sqrt(np.mean(pcm[8800:15200] ** 2))


@pytest.mark.parametrize("role", [None, "music"])  # one ffmpeg process, or the managed mix's thread
def test_audio_mixes_while_frames_render_and_stops_when_they_fail(clip, tmp_path, monkeypatch, role):
    from kinescript import output
    started, real_start = [], output._start_audio

    def start(*a):
        started.append(real_start(*a))
        return started[-1]

    def frames(job):
        assert started and started[0][0].poll() is None, "the audio mix should already be running"
        raise RuntimeError("frames failed")

    monkeypatch.setattr(output, "_start_audio", start)
    monkeypatch.setattr(output, "_render_range", frames)
    spec = scene([{"type": "rect", "w": 320, "h": 180}], duration=60)
    spec["audio"] = [dict({"src": clip, "at": t}, **({"role": role} if role else {}))
                     for t in range(0, 60, 3)]  # a minute of audio takes a while to mix
    with pytest.raises(RuntimeError, match="frames failed"):
        k.render(spec, str(tmp_path / "o.mp4"), quality="draft", workers=1)
    assert started[0][0].poll() is not None  # stopped, not left running


@pytest.fixture(scope="session")
def tone(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "tone.wav"
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.5",
                    str(p)], check=True)
    return str(p)


def _pcm(path, channels=1):
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a", "-ac", str(channels),
                          "-ar", "8000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels)


def _rms(pcm, t0, t1):
    return float(np.sqrt(np.mean(pcm[int(t0 * 8000):int(t1 * 8000)] ** 2)))


def test_every_mix_pads_by_samples_so_it_always_ends(tone, tmp_path, monkeypatch):
    # An endless apad was cut by timestamps; ffmpeg 8 sometimes got those wrong and the mix never ended.
    import re
    from kinescript import output
    graphs, real = [], output._graph

    def record(*a):
        graphs.append(real(*a))
        return graphs[-1]

    monkeypatch.setattr(output, "_graph", record)
    for role in (None, "music"):
        spec = scene([{"type": "rect"}], duration=1)
        spec["audio"] = [dict({"src": tone, "at": 0.5}, **({"role": role} if role else {}))]
        k.render(spec, str(tmp_path / "o.wav"))
    pads = [m for _, g in graphs for m in re.findall(r"apad[^,;\[]*", g)]
    assert len(graphs) == 2 and pads and all(m == "apad=whole_dur=1.0000" for m in pads)


def test_layers_cue_from_audio_and_only_voice_extends_auto_duration(tone):
    for role, duration in (("voice", 1.5), ("music", 1.0)):
        spec = scene([{"id": "a", "type": "rect", "dur": 1}, {"id": "b", "type": "rect", "at": "vo.out"}])
        spec["audio"] = [{"id": "vo", "role": role, "src": tone, "at": 1}]
        s = k.Scene(spec)
        assert s.by_id["b"].t_in == pytest.approx(1.5)
        assert s.duration == pytest.approx(duration)
        assert probe(spec)["audio"][0] == {"id": "vo", "role": role, "src": tone, "in": 1.0, "out": 1.5, "layer": None}


def test_audio_entries_chain_with_prev(tone):
    spec = scene([{"type": "rect", "dur": 3}])
    spec["audio"] = [{"id": "l1", "src": tone, "at": 0.2}, {"id": "l2", "src": tone, "at": "prev.out+0.3"}]
    assert k.Scene(spec).timeline.resolved["l2"] == pytest.approx((1.0, 1.5))


def test_layer_sound_moves_with_its_layer_and_skips_hidden_layers(tone):
    def sound_in(start, **extra):
        spec = scene([{"id": "t", "type": "rect", "at": start, "dur": 0.5, "sound": dict({"src": tone, "at": "out-0.1"},
                                                                                       **extra)}], duration=3)
        return [(x.id, x.role, x.t_in) for x in k.Scene(spec).audio]
    assert sound_in(0.8) == [("t-sound", "sfx", pytest.approx(1.2))]
    assert sound_in(1.4) == [("t-sound", "sfx", pytest.approx(1.8))]
    assert sound_in(1.4, at=0.25)[0][2] == pytest.approx(1.65)  # a number counts from the layer start
    spec = scene([{"id": "t", "type": "rect", "hidden": True, "sound": tone}])
    assert k.Scene(spec).audio == []


def test_components_and_sequences_carry_a_sound(tone, clip):
    spec = scene([{"use": "titleCard", "id": "card", "title": "Hi", "at": 0.5, "dur": 1, "sound": tone},
                  {"type": "sequence", "id": "cut", "at": 2, "clips": [{"src": clip, "trim": [0, 1]}],
                   "sound": {"src": tone, "at": "out"}}])
    assert [(x.id, x.t_in) for x in k.Scene(spec).audio] == [("card-sound", 0.5), ("cut-sound", pytest.approx(3.0))]


def test_layer_sound_renders_at_its_cue_and_a_short_mix_keeps_every_frame(tone, tmp_path):
    spec = scene([{"id": "t", "type": "rect", "at": 1, "dur": 0.5, "sfx": tone}], duration=2.5)
    out = tmp_path / "o.mp4"
    k.render(spec, str(out), quality="draft")
    pcm = _pcm(out)
    assert _rms(pcm, 0, 0.95) < 1e-3 < 0.05 < _rms(pcm, 1.05, 1.45)
    frames = subprocess.run([media.FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert int(frames) == 75


def test_looped_audio_repeats_to_the_scene_end(tone, tmp_path):
    spec = scene([{"type": "rect"}], duration=2)
    spec["audio"] = [{"role": "music", "src": tone, "loop": True}]
    out = tmp_path / "o.wav"
    assert k.render(spec, str(out))["seconds"] == 2
    pcm = _pcm(out)
    assert len(pcm) == pytest.approx(16000, abs=80) and _rms(pcm, 1.6, 1.95) > 0.05


def test_pan_moves_sound_between_channels(tone, tmp_path):
    spec = scene([{"type": "rect"}], duration=1)
    spec["audio"] = [{"src": tone, "pan": -1}]
    out = tmp_path / "o.wav"
    k.render(spec, str(out))
    pcm = _pcm(out, 2)
    left, right = _rms(pcm[:, 0], 0.1, 0.4), _rms(pcm[:, 1], 0.1, 0.4)
    assert left > 0.05 and right < left / 100


def test_audio_only_render_draws_no_frames(tone, tmp_path, monkeypatch):
    from kinescript import output

    def frames(job):
        raise AssertionError("no frame should be rendered")

    monkeypatch.setattr(output, "_render_range", frames)
    spec = scene([{"type": "rect"}], duration=1.5)
    spec["audio"] = [{"src": tone, "at": 0.5}]
    res = k.render(spec, str(tmp_path / "mix.m4a"))
    assert res["seconds"] == 1.5 and media.probe(res["out"])["audio"]
    with pytest.raises(ValueError, match="no audio"):
        k.render(scene([{"type": "rect"}], duration=1), str(tmp_path / "none.wav"))


@pytest.mark.parametrize("ext,codec", [(".mov", "pcm_s16le"), (".webm", "opus")])
def test_mov_and_webm_carry_the_mix(tone, tmp_path, ext, codec):
    spec = scene([{"type": "rect", "w": 100, "h": 60}], duration=1)
    spec["audio"] = [{"src": tone}]
    out = tmp_path / f"o{ext}"
    assert k.render(spec, str(out), quality="draft")["audio"]
    streams = subprocess.run([media.FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=codec_name", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert streams.strip() == codec


def test_range_render_cuts_the_full_mix(tone, tmp_path):
    spec = scene([{"type": "rect"}], duration=3)
    spec["audio"] = [{"src": tone, "at": 1.5, "fadeIn": 0.05}]
    out = tmp_path / "o.mp4"
    assert k.render(spec, str(out), quality="draft", start=1.0, end=2.5)["audio"]
    pcm = _pcm(out)
    assert len(pcm) == pytest.approx(12000, abs=400)
    assert _rms(pcm, 0, 0.45) < 1e-3 < 0.05 < _rms(pcm, 0.6, 0.95)


@pytest.fixture(scope="session")
def mix_media(tmp_path_factory):
    """A 200 Hz bed, a 2 kHz voice, and an effect whose 3 kHz burst starts 0.3 s into its file."""
    d, out = tmp_path_factory.mktemp("mix"), {}
    for name, expr, dur in (("bed", "0.5*sin(2*PI*200*t)", 2), ("voice", "0.3*sin(2*PI*2000*t)", 1.5),
                            ("hit", "if(between(t,0.3,0.4),0.8*sin(2*PI*3000*t),0)", 0.5)):
        out[name] = str(d / f"{name}.wav")
        subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={dur}",
                        out[name]], check=True)
    return out


def _band(pcm, t0, t1, hz):
    """Level in dB of one frequency between t0 and t1, so one sound can be measured under another."""
    seg = pcm[int(t0 * 8000):int(t1 * 8000)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return 20 * np.log10(spec[np.argmin(np.abs(np.fft.rfftfreq(len(seg), 1 / 8000) - hz))] + 1e-12)


def _measure(path):
    from kinescript import audio
    log = subprocess.run([media.FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f",
                          "null", "-"], capture_output=True, text=True).stderr
    return audio.loudness(log)


def test_roles_are_levelled_ducked_and_brought_to_the_loudness_target(mix_media, tmp_path):
    spec = scene([{"type": "rect"}], duration=6)
    spec["audio"] = [{"role": "music", "src": mix_media["bed"], "loop": True, "until": "end"},
                     {"role": "voice", "src": mix_media["voice"], "at": 2}]
    out = tmp_path / "o.wav"
    k.render(spec, str(out))
    got = _measure(out)
    assert got["lufs"] == pytest.approx(-16, abs=1) and got["true_peak"] <= -1.5
    pcm = _pcm(out)[:, 0]
    free = _band(pcm, 0.5, 1.5, 200)
    assert free - _band(pcm, 2.4, 3.4, 200) == pytest.approx(14, abs=1)  # the bed dips under the voice
    assert _band(pcm, 4.6, 5.6, 200) == pytest.approx(free, abs=0.5)  # and comes back after it
    spec["mix"] = {"peak": -6, "duck": False}
    k.render(spec, str(out))
    pcm = _pcm(out)[:, 0]
    assert _measure(out)["true_peak"] <= -6
    assert _band(pcm, 0.5, 1.5, 200) - _band(pcm, 2.4, 3.4, 200) == pytest.approx(0, abs=1)


def test_an_effect_lands_on_its_cue_past_its_leading_silence(mix_media, tmp_path):
    spec = scene([{"id": "t", "type": "rect", "at": 1, "dur": 0.5, "sound": mix_media["hit"]}], duration=2)
    out = tmp_path / "o.wav"
    k.render(spec, str(out))
    pcm = _pcm(out)[:, 0]
    assert np.argmax(np.abs(pcm) > 0.01) / 8000 == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("role", [None, "music"])
def test_keyframed_gain_changes_the_level_over_time(tmp_path, role):
    src = tmp_path / "t44.wav"  # not 48 kHz, so the gain curve has to meet it at another rate
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=1000:sample_rate=44100:duration=2", str(src)], check=True)
    spec = scene([{"type": "rect"}], duration=2)
    spec["audio"] = [dict({"src": str(src), "gain": [[0, -40], [1, 0]]}, **({"role": role} if role else {}))]
    out = tmp_path / "o.wav"
    k.render(spec, str(out))
    pcm = _pcm(out)[:, 0]
    db = lambda t0, t1: 20 * np.log10(_rms(pcm, t0, t1))  # noqa: E731
    assert db(1.2, 1.8) - db(0.45, 0.55) == pytest.approx(20, abs=1.5)
    assert db(1.2, 1.8) - db(0, 0.05) > 35 and len(pcm) == pytest.approx(16000, abs=80)


def test_a_changing_gain_must_give_decibels(tone):
    spec = dict(scene([{"type": "rect"}]), audio=[{"src": tone, "gain": [[0, "loud"], [1, 0]]}])
    with pytest.raises(k.SceneError, match="is not a number of dB"):
        k.Scene(spec)


def _heard(spec):
    """check's sound issues by code."""
    return {i["code"]: i for i in check(spec)["issues"] if i["code"].startswith(("audio", "voice", "sfx"))}


def test_check_hears_cut_overlapping_buried_late_and_silent_sounds(mix_media, tmp_path):
    silent = tmp_path / "silent.wav"
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "1",
                    str(silent)], check=True)
    spec = scene([{"type": "rect"}], duration=2.2)
    spec["audio"] = [{"id": "bed", "role": "music", "src": mix_media["bed"], "loop": True, "until": "end"},
                     {"id": "vo1", "role": "voice", "src": mix_media["voice"], "at": 0.2},
                     {"id": "vo2", "role": "voice", "src": mix_media["voice"], "at": 1.0},
                     {"id": "hit", "role": "sfx", "src": mix_media["hit"], "at": 1.6, "trim": [0, 0.5]},
                     {"id": "hush", "role": "ambience", "src": str(silent)}]
    spec["mix"] = {"duck": False}
    got = _heard(spec)
    assert (got["audio-cut"]["layer"], got["audio-cut"]["from"], got["audio-cut"]["to"]) == ("vo2", 2.2, 2.5)
    assert (got["voice-overlap"]["layer"], got["voice-overlap"]["from"], got["voice-overlap"]["to"]) == ("vo1+vo2", 1.0, 1.7)
    assert got["voice-masked"]["fix"] == "remove \"duck\": false from mix"
    assert (got["sfx-late"]["layer"], got["sfx-late"]["from"]) == ("hit", 1.6)
    assert got["sfx-late"]["to"] == pytest.approx(1.9, abs=0.02)
    assert got["audio-silent"]["layer"] == "hush"
    spec["audio"], spec["canvas"]["duration"] = spec["audio"][:2], 3
    del spec["mix"]
    assert _heard(spec) == {}  # levelled and ducked: nothing to report


def test_check_reports_a_mix_without_roles_that_is_quiet_or_at_its_limiter(tone):
    spec = scene([{"type": "rect"}], duration=1)
    spec["audio"] = [{"src": tone, "gain": -30}]
    assert _heard(spec)["audio-loudness"]["level"] == "info"
    spec["audio"] = [{"src": tone, "gain": 24}]
    assert _heard(spec)["audio-peak"]["level"] == "warning"


def test_look_draws_the_sound_and_returns_its_measurements(mix_media, tmp_path):
    import skia
    from kinescript.verify import contact_sheet
    spec = scene([{"type": "rect"}], duration=4)
    spec["audio"] = [{"role": "music", "src": mix_media["bed"], "loop": True, "until": "end"},
                     {"role": "voice", "src": mix_media["voice"], "at": 1}]
    plain = contact_sheet(spec, str(tmp_path / "a.png"), count=4, width=160)
    heard = contact_sheet(spec, str(tmp_path / "b.png"), count=4, width=160, audio=True)
    assert "audio" not in plain
    a = heard["audio"]
    assert a["mix"]["lufs"] == pytest.approx(-16, abs=1) and a["mix"]["true_peak"] <= -1.5
    assert [s["lufs_out"] for s in a["sounds"]] == [pytest.approx(-16, abs=0.5), pytest.approx(-18, abs=0.5)]
    assert a["duck"]["depth"] == 14 and a["duck"]["spans"][0][0] == pytest.approx(0.94, abs=0.03)
    # three lanes (voice, music, the mix) of 44 px under a 22 px time axis, and an 8 px margin
    assert skia.Image.open(str(tmp_path / "b.png")).height() - skia.Image.open(str(tmp_path / "a.png")).height() == 162
    silent = contact_sheet(scene([{"type": "rect"}], duration=1), str(tmp_path / "c.png"), count=1, audio=True)
    assert silent["audio"] is None


WORDS = [{"word": "Write", "start": 0.3, "end": 0.6}, {"word": "it.", "start": 0.7, "end": 0.9},
         {"word": "Write", "start": 1.0, "end": 1.3}]


def _voice(mix_media, tmp_path, words=WORDS):
    """The voice copied next to a words file, so the file is found by its name."""
    import shutil
    src = tmp_path / "vo.wav"
    shutil.copy(mix_media["voice"], src)
    if words is not None:
        (tmp_path / "vo.words.json").write_text(json.dumps({"words": words}))
    return str(src)


def test_layers_cue_from_the_words_a_voice_says(mix_media, tmp_path):
    spec = scene([{"id": "a", "type": "rect", "at": "vo:write_2", "dur": 0.2},
                  {"id": "b", "type": "rect", "at": "vo:it.out+0.1", "dur": 0.2},
                  {"id": "c", "type": "rect", "at": "vo:write.mid", "until": "vo:write_2.out"}], duration=3)
    spec["audio"] = [{"id": "vo", "role": "voice", "src": _voice(mix_media, tmp_path), "at": 0.5, "trim": [0.2, 1.4]}]
    s = k.Scene(spec)
    # word times are seconds in the file: the voice starts at 0.5 playing from 0.2 s into it
    assert s.by_id["a"].t_in == pytest.approx(1.3)
    assert s.by_id["b"].t_in == pytest.approx(1.3)
    assert (s.by_id["c"].t_in, s.by_id["c"].t_out) == (pytest.approx(0.75), pytest.approx(1.6))


@pytest.mark.parametrize("at, says", [
    ("vo:writ", r"has no word 'writ'\. Did you mean 'write'\?"),
    ("voo:write", r"'voo', which is not an audio id\. Did you mean 'vo'\?"),
    ("bed:write", r"'bed' has no word times\. Add \S*bed\.words\.json"),
    ("vo:write", r"at 0\.30s in the file of 'vo', outside the part its trim plays"),
])
def test_a_word_cue_that_cannot_be_found_says_why(mix_media, tmp_path, at, says):
    spec = scene([{"type": "rect", "at": at}], duration=1)
    spec["audio"] = [{"id": "vo", "src": _voice(mix_media, tmp_path), "trim": [0.5, None]},
                     {"id": "bed", "src": mix_media["bed"]}]
    with pytest.raises(k.SceneError, match=says):
        k.Scene(spec)


def test_words_read_from_a_list_or_whisper_json_and_get_ids(tmp_path):
    from kinescript import words as W
    (tmp_path / "w.json").write_text(json.dumps({"segments": [{"words": [
        {"word": " Don't", "start": 0.0, "end": 0.2}, {"word": " stop,", "start": 0.3, "end": 0.5}]},
        {"words": [{"word": " stop!", "start": 0.6, "end": 0.8}, {"word": "—", "start": 0.9, "end": 1.0}]}]}))
    assert [w["id"] for w in W.read(tmp_path / "w.json")] == ["dont", "stop", "stop_2", None]
    got = W.read([{"word": "go", "start": 0, "end": 0.1}, {"word": "Go", "start": 0.2, "end": 0.3, "id": "go"}])
    assert [w["id"] for w in got] == ["go_2", "go"]  # an id given in the file keeps its name
    for bad, says in (([{"word": "go", "start": 0.5, "end": 0.1}], "ends at 0.1s, before"),
                      ([{"word": "go", "start": "0"}], "word #0"), (tmp_path / "no.json", "words file not found")):
        with pytest.raises(W.WordsError, match=says):
            W.read(bad)


def test_a_bad_words_file_is_reported_with_its_sound(mix_media, tmp_path):
    spec = scene([{"type": "rect"}], duration=1)
    spec["audio"] = [{"id": "vo", "src": _voice(mix_media, tmp_path, [{"word": "x", "start": 1, "end": 0}])}]
    with pytest.raises(k.SceneError) as e:
        k.Scene(spec)
    assert [(i["code"], i["layer"]) for i in e.value.issues if i["level"] == "error"] == [("bad-words", "vo")]


def test_caption_lines_break_at_sentences_pauses_and_width():
    from kinescript import words as W
    ws = [{"word": w, "start": s, "end": s + 0.2} for w, s in
          (("One", 0.0), ("two.", 0.3), ("Three", 0.6), ("four", 1.6), ("five", 1.9))]
    ws += [{"word": "word", "start": 2.2 + 0.3 * i, "end": 2.4 + 0.3 * i} for i in range(10)]
    got = W.cues(ws, trim0=0.0)
    assert [c[2] for c in got] == ["One two.", "Three", "four five word word word word word word",
                                   "word word word word"]
    assert got[0][:2] == [0.0, 0.5] and got[1][:2] == [0.5, 1.3]  # held 0.5 s, unless the next starts first
    assert got[0][3] == [[0.0, 0.2], [0.3, 0.5]]
    assert [c[2] for c in W.cues(ws, trim0=0.3, trim1=1.0)] == ["two.", "Three"]


def test_captions_follow_a_voice_and_light_each_word_as_it_is_said(mix_media, tmp_path):
    spec = scene([{"id": "cc", "type": "captions", "voice": "vo", "mode": "pop", "size": 30}], w=640, h=360,
                 duration="auto")
    spec["audio"] = [{"id": "vo", "role": "voice", "src": _voice(mix_media, tmp_path), "at": 1.0}]
    s = k.Scene(spec)
    cc = s.by_id["cc"]
    assert cc.t_in == 1.0 and cc.raw["cues"][0][2] == "Write it." and cc.raw["cues"][1][2] == "Write"
    r = Renderer(s)
    lit = [int((r.frame(round(T * 30))[300:340, :, :3].max(axis=2) > 200).sum()) for T in (1.25, 1.45, 1.75)]
    assert lit[0] == 0 < lit[1] < lit[2]  # pop: nothing before "Write" is said, then one word, then two
    spec["layers"][0]["voice"] = "bed"
    spec["audio"].append({"id": "bed", "src": mix_media["bed"]})
    with pytest.raises(k.SceneError, match="'bed' has no word times for the captions"):
        k.Scene(spec)


def _fake_speak(req):
    """A voice backend for tests: a 0.3 s tone per word, and those times."""
    said = req["text"].split()
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=300:sample_rate=24000:duration={0.3 * len(said) + 0.2}", req["out"]], check=True)
    return {"words": [{"word": w, "start": round(0.3 * k, 3), "end": round(0.3 * k + 0.25, 3)}
                      for k, w in enumerate(said)], "speaker": req["speaker"] or "fake_one", "version": "1"}


@pytest.fixture
def fake_voice(monkeypatch):
    from kinescript import voice
    calls = []

    class Entry:
        name = "fake"

        def load(self):
            return lambda req: calls.append(req["text"]) or _fake_speak(req)
    monkeypatch.delenv("KINE_VOICE_CMD", raising=False)
    monkeypatch.setattr(voice, "backends", lambda: {"fake": Entry()})
    return calls


def test_kine_voice_speaks_text_once_and_names_marked_words(fake_voice, tmp_path):
    from kinescript.voice import generate
    f = tmp_path / "s.json"
    spec = scene([{"id": "go", "type": "rect", "at": "vo1:go", "dur": 0.2},
                  {"id": "cc", "type": "captions", "voice": "vo1"}], duration="auto")
    spec["audio"] = [{"id": "vo1", "role": "voice", "text": "Write a scene. [[go]]Check it."}]
    f.write_text(json.dumps(spec))
    with pytest.raises(k.SceneError, match="'vo1' has text but is not spoken yet"):
        k.load(str(f))
    assert generate(str(f))["voices"] == [{"id": "vo1", "src": "voice/vo1.wav", "status": "made", "words": 5,
                                           "seconds": 1.45, "took": pytest.approx(0, abs=5)}]
    kept = json.loads((tmp_path / "voice" / "vo1.words.json").read_text())
    assert kept["words"][3] == {"word": "Check", "start": 0.9, "end": 1.15, "id": "go"}
    assert kept["made"]["backend"] == "fake" and kept["made"]["speaker"] == "fake_one"
    s = k.load(str(f))
    assert s.by_id["go"].t_in == pytest.approx(0.9) and s.by_id["cc"].raw["cues"][1][2] == "Check it."
    assert generate(str(f))["voices"][0]["status"] == "fresh" and fake_voice == ["Write a scene. Check it."]
    spec["audio"][0]["text"] = "Write a scene. [[go]]Check it twice."
    f.write_text(json.dumps(spec))
    assert [i["code"] for i in k.load(str(f)).issues] == ["voice-stale"]
    assert generate(str(f))["voices"][0]["status"] == "made" and len(fake_voice) == 2
    assert generate(str(f), force=True)["voices"][0]["status"] == "made" and len(fake_voice) == 3
    with pytest.raises(ValueError, match="no sound with text has id vo2; sounds with text: vo1"):
        generate(str(f), only=["vo2"])


def test_kine_voice_runs_a_command_for_a_backend_in_another_python(tmp_path, monkeypatch):
    from kinescript import voice
    tool = tmp_path / "speak.py"
    tool.write_text("import json, sys\nsys.path.insert(0, %r)\nfrom test_core import _fake_speak\n"
                    "print(json.dumps([_fake_speak(r) for r in json.load(sys.stdin)]))\n" % str(Path(__file__).parent))
    monkeypatch.setenv("KINE_VOICE_CMD", json.dumps([sys.executable, str(tool)]))
    monkeypatch.setattr(voice, "backends", lambda: {})
    spec = scene([{"type": "rect"}], duration="auto")
    spec["audio"] = [{"id": "a", "text": "One; two"}, {"id": "b", "text": "three \"quoted\" & $(nothing)"}]
    got = voice.generate(spec, base=tmp_path)["voices"]
    assert [(v["id"], v["status"], v["words"]) for v in got] == [("a", "made", 2), ("b", "made", 4)]
    # the text went to the command as JSON on stdin, untouched by any shell
    assert json.loads((tmp_path / "voice" / "b.words.json").read_text())["words"][3]["word"] == "$(nothing)"
    monkeypatch.delenv("KINE_VOICE_CMD")
    spec["audio"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="no voice backend is installed"):
        voice.generate(spec, base=tmp_path)


def test_marks_find_their_word_when_the_voice_reads_it_differently():
    from kinescript.voice import place_marks, split_marks
    text, marks = split_marks("In 2026 [[ship]]we ship. [[end]]Done")
    assert (text, marks) == ("In 2026 we ship. Done", {2: "ship", 4: "end"})
    said = [{"word": w, "start": k, "end": k + 0.5} for k, w in enumerate(["In", "twenty", "twenty-six", "we", "ship.",
                                                                            "Done"])]
    assert [w.get("id") for w in place_marks(text, marks, said)] == [None, None, None, "ship", None, "end"]
    with pytest.raises(ValueError, match=r"\[\[a\]\] is used twice"):
        split_marks("[[a]]x [[a]]y")


def test_tags_are_not_words_and_reach_only_backends_that_act_on_them(monkeypatch, tmp_path):
    from kinescript import voice
    text, marks = voice.split_marks("Well [sighs] okay, [[go]]go [laughs]now.")
    assert (text, marks) == ("Well [sighs] okay, go [laughs]now.", {2: "go"})
    assert voice.untag(text) == "Well okay, go now."
    # marks and pronunciations are not tags
    assert voice.tags("Meet [Kinescript](/kˈɪn/) [[a]]now [clears throat] ok") == ["[clears throat]"]
    got = []

    def entry(acts):
        def speak(req):
            got.append((req["text"], req["tagged"]))
            return {**_fake_speak(req), "tags": acts}
        return type("E", (), {"load": lambda self: speak})()
    monkeypatch.delenv("KINE_VOICE_CMD", raising=False)
    monkeypatch.setattr(voice, "backends", lambda: {"plain": entry(False), "acts": entry(True)})
    spec = scene([{"type": "rect"}], duration="auto")
    spec["audio"] = [{"id": "a", "text": "Well [sighs] okay. [[go]]Go.", "backend": "plain"},
                     {"id": "b", "text": "Well [sighs] okay. [[go]]Go.", "backend": "acts"}]
    a, b = voice.generate(spec, base=tmp_path)["voices"]
    assert got == [("Well okay. Go.", "Well [sighs] okay. Go.")] * 2
    assert a["ignored"] == ["[sighs]"] and "ignored" not in b  # said without it, and said so
    words = json.loads((tmp_path / "voice" / "b.words.json").read_text())["words"]
    assert [w["word"] for w in words] == ["Well", "okay.", "Go."] and words[2]["id"] == "go"


def test_what_a_backend_prints_stays_off_stdout(monkeypatch, tmp_path, capsys):
    # kine voice answers in JSON on stdout and kine mcp speaks JSON-RPC there; Chatterbox prints as it loads
    from kinescript import voice

    def speak(req):
        print("loaded PerthNet")
        return _fake_speak(req)
    monkeypatch.delenv("KINE_VOICE_CMD", raising=False)
    monkeypatch.setattr(voice, "backends", lambda: {"loud": type("E", (), {"load": lambda self: speak})()})
    spec = scene([{"type": "rect"}], duration="auto")
    spec["audio"] = [{"id": "a", "text": "hello"}]
    assert voice.generate(spec, base=tmp_path)["voices"][0]["status"] == "made"
    out, err = capsys.readouterr()
    assert out == "" and "loaded PerthNet" in err


def test_each_sound_picks_its_backend_and_options(monkeypatch, tmp_path):
    import hashlib
    from kinescript import voice
    seen = []

    def entry(name):
        def speak(req):
            seen.append((name, req["options"], req["base"]))
            return {**_fake_speak(req), **({"note": "half untimed"} if name == "two" else {})}
        return type("E", (), {"load": lambda self: speak})()
    monkeypatch.delenv("KINE_VOICE_CMD", raising=False)
    monkeypatch.setattr(voice, "backends", lambda: {"one": entry("one"), "two": entry("two")})
    spec = scene([{"type": "rect"}], duration="auto")
    spec["audio"] = [{"id": "a", "text": "first"},
                     {"id": "b", "text": "second", "engine": "two", "settings": {"seed": 3}}]  # aliases read
    with pytest.raises(ValueError, match="several voice backends are installed"):
        voice.generate(spec, base=tmp_path)
    got = voice.generate(spec, base=tmp_path, backend="one")
    assert [v.get("note") for v in got["voices"]] == [None, "half untimed"]  # the backend's own note
    assert seen == [("one", {}, str(tmp_path.resolve())), ("two", {"seed": 3}, str(tmp_path.resolve()))]
    assert json.loads((tmp_path / "voice" / "b.words.json").read_text())["made"]["backend"] == "two"
    # options are part of what the file was made from; a sound without backend or options keeps its old hash
    f = tmp_path / "s.json"
    spec["audio"] = [{"id": "b", "text": "second", "backend": "two", "options": {"seed": 4}}]
    f.write_text(json.dumps(spec))
    assert [i["code"] for i in k.load(str(f)).issues] == ["voice-stale"]
    old = {"text": "x", "speaker": None, "rate": 1}
    assert voice.fingerprint({"text": "x"}) == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:16]
    with pytest.raises(ValueError, match="no voice backend 'three': pip install kinescript-three, or set "
                                         "KINE_VOICE_CMD_THREE"):
        voice.generate({**spec, "audio": [{"id": "c", "text": "x", "backend": "three"}]}, base=tmp_path)
    # a backend in another Python, by name: its own variable, or KINE_VOICE_CMD when it runs that module
    tool = tmp_path / "speak.py"
    tool.write_text("import json, sys\nsys.path.insert(0, %r)\nfrom test_core import _fake_speak\n"
                    "print(json.dumps([_fake_speak(r) for r in json.load(sys.stdin)]))\n" % str(Path(__file__).parent))
    monkeypatch.setenv("KINE_VOICE_CMD_THREE", json.dumps([sys.executable, str(tool)]))
    got = voice.generate({**spec, "audio": [{"id": "c", "text": "x y", "backend": "three"}]}, base=tmp_path)
    assert got["voices"][0]["words"] == 2
    assert voice._command("kokoro") is None
    monkeypatch.setenv("KINE_VOICE_CMD", "py -m kinescript_kokoro")
    assert voice._command("kokoro") == (["py", "-m", "kinescript_kokoro"], "KINE_VOICE_CMD")
    assert voice._command("other") is None and voice._command(None)[1] == "KINE_VOICE_CMD"


def _plugin(monkeypatch, name):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "plugins" / f"kinescript-{name}"))


def test_forced_alignment_places_words_around_a_laugh_and_a_number(monkeypatch):
    _plugin(monkeypatch, "chatterbox")
    from kinescript_chatterbox.align import ctc_path, times, units
    vocab = {"<pad>": 0, "|": 1, "A": 2, "B": 3, "C": 4, "D": 5}
    words, seq = units("AB [laugh] C 12.", vocab)
    assert words == ["AB", "C", "12."] and [s[0] for s in seq] == [2, 3, -2, 4, 1, -1]

    def frames(said):  # what each 20 ms frame sounds like most: a vocabulary id, 0 for nothing
        lp = np.full((len(said), len(vocab)), np.log(0.02))
        lp[np.arange(len(said)), said] = np.log(0.9)
        return lp
    # silence, A A B, a laugh (heard as D D D), silence, C C, a word gap, "twelve" (D D D), silence
    said = [0, 0, 2, 2, 3, 5, 5, 5, 0, 4, 4, 1, 5, 5, 5, 0]
    got = times(ctc_path(frames(said), seq, 0), seq, words, offset=1)
    assert got == [{"word": "AB", "start": 1.04, "end": 1.1}, {"word": "C", "start": 1.18, "end": 1.22},
                   {"word": "12.", "start": 1.24, "end": 1.3}]
    # no laugh at all: the tag takes no time
    said = [0, 2, 3, 0, 4, 1, 5, 0]
    assert [w["start"] for w in times(ctc_path(frames(said), seq, 0), seq, words)] == [0.02, 0.08, 0.12]
    with pytest.raises(ValueError, match="too short"):
        ctc_path(frames([2, 3]), seq, 0)
    # a line of more than 127 states (63 letters and gaps) backtracks without overflowing
    words, seq = units(" ".join(["AB"] * 30), vocab)
    said = [0] + [2, 3, 1] * 30
    assert [w["start"] for w in times(ctc_path(frames(said), seq, 0), seq, words)][-1] == 1.76


def test_forced_alignment_reads_numbers_as_they_are_said(monkeypatch):
    _plugin(monkeypatch, "chatterbox")
    from kinescript_chatterbox.align import spoken, units
    said = {w: " ".join(spoken(w)) for w in ["250", "2026", "2005", "1,000", "3.5", "v1.2.0", "$5", "40%", "21st",
                                             "100th", "Render.", "—"]}
    assert said == {"250": "TWO HUNDRED FIFTY", "2026": "TWENTY TWENTY SIX", "2005": "TWO THOUSAND FIVE",
                    "1,000": "ONE THOUSAND", "3.5": "THREE POINT FIVE", "v1.2.0": "V ONE POINT TWO POINT ZERO",
                    "$5": "FIVE DOLLARS", "40%": "FORTY PERCENT", "21st": "TWENTY FIRST", "100th": "ONE HUNDREDTH",
                    "Render.": "RENDER", "—": ""}
    vocab = {"<pad>": 0, "|": 1, **{c: 2 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")}}
    words, seq = units("took 12 — now", vocab)
    # each word of a number is its own run of letters, a lone dash says nothing
    assert words == ["took", "12", "now"]
    assert "".join("|" if s[0] == 1 else "?" if s[0] < 0 else chr(63 + s[0]) for s in seq) == "TOOK|TWELVE|NOW"


def test_chatterbox_reads_tags_written_for_other_backends(monkeypatch):
    _plugin(monkeypatch, "chatterbox")
    import kinescript_chatterbox as cb
    monkeypatch.setattr(cb, "TURBO_TAGS", {"[laugh]", "[sigh]", "[chuckle]", "[clear throat]", "[whispering]"})
    written = ["[laugh]", "[Laughs]", "[laughing]", "[sighs]", "[chuckles]", "[whispers]", "[clears throat]", "[sings]"]
    assert [cb.turbo_tag(t) for t in written] == ["[laugh]", "[laugh]", "[laugh]", "[sigh]", "[chuckle]",
                                                  "[whispering]", "[clear throat]", None]
    long = "One two three. " * 30
    assert all(len(c) <= 300 for c in cb.chunks(long)) and " ".join(cb.chunks(long)) == long.strip()


def test_elevenlabs_times_words_from_its_character_times(monkeypatch, tmp_path):
    import base64
    import wave
    _plugin(monkeypatch, "elevenlabs")
    import kinescript_elevenlabs as el
    text = "Hi [laughs] there, all."
    chars = list(text)
    starts = [0.1 * k for k in range(len(chars))]
    words = el.words_from_chars(chars, starts, [s + 0.08 for s in starts])
    assert [w["word"] for w in words] == ["Hi", "there,", "all."]
    assert words[1] == {"word": "there,", "start": 1.2, "end": 1.68}  # first to last letter, not the comma
    calls = []

    def call(path, key, body=None, ctype="application/json"):
        calls.append((path, json.loads(body) if ctype == "application/json" and body else body))
        if path == "/v1/voices":
            return {"voices": [{"voice_id": "a" * 20, "name": "Aria - calm"}]}
        if path.startswith("/v1/forced-alignment"):
            return {"words": [{"text": "Hi", "start": 0.0, "end": 0.2}, {"text": " ", "start": 0.2, "end": 0.3},
                              {"text": "there,", "start": 0.3, "end": 0.6}]}
        timing = None if "multilingual" in json.loads(body)["model_id"] else {
            "characters": chars, "character_start_times_seconds": starts,
            "character_end_times_seconds": [s + 0.08 for s in starts]}
        return {"audio_base64": base64.b64encode(b"\0\0" * 24000).decode(), "alignment": timing}
    monkeypatch.setattr(el, "_call", call)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    out = tmp_path / "v.wav"
    req = {"text": "Hi there, all.", "tagged": text, "speaker": "Aria", "rate": 1.1, "out": str(out),
           "options": {"stability": 0.3, "model": "eleven_v4", "loud": 1}}
    res = el.synth(req)
    assert calls[1] == (f"/v1/text-to-speech/{'a' * 20}/with-timestamps?output_format=pcm_24000",
                        {"text": text, "model_id": "eleven_v4", "voice_settings": {"stability": 0.3, "speed": 1.1}})
    assert res["tags"] and res["ignored"] == ["options.loud"] and res["seconds"] == 1.0 and len(res["words"]) == 3
    with wave.open(str(out)) as f:
        assert (f.getframerate(), f.getnframes()) == (24000, 24000)
    # a model without tags gets the text without them; with no character times, forced alignment times the words
    calls.clear()
    res = el.synth({**req, "rate": 2, "speaker": "b" * 20, "options": {"model": "eleven_multilingual_v2"}})
    assert calls[0][1]["text"] == "Hi there, all." and not res["tags"]
    assert res["ignored"] == ["rate 2: ElevenLabs speaks at 0.7 to 1.2"] and calls[1][0] == "/v1/forced-alignment"
    assert [w["word"] for w in res["words"]] == ["Hi", "there,"]
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    with pytest.raises(RuntimeError, match="set ELEVENLABS_API_KEY"):
        el.synth(req)


def test_kokoro_tokens_become_the_words_of_the_text(monkeypatch):
    from types import SimpleNamespace as T
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "plugins" / "kinescript-kokoro"))
    from kinescript_kokoro import words_from_tokens
    # As Kokoro 0.9.4 times "Write a scene. Check it, in 2026!" (punctuation is a token of its own)
    toks = [("Write", " ", 0.3, 0.5125), ("a", " ", 0.5125, 0.6125), ("scene", "", 0.6125, 1.15), (".", " ", 1.15, 1.2875),
            ("Check", " ", 1.2875, 1.55), ("it", "", 1.55, 1.65), (",", " ", 1.65, 1.6875), ("—", " ", None, None),
            ("in", " ", None, None), ("2026", "", 2.35, 3.55), ("!", "", 3.55, 3.65)]
    words = words_from_tokens([T(text=a, whitespace=b, start_ts=c, end_ts=d) for a, b, c, d in toks], offset=10)
    assert [w["word"] for w in words] == ["Write", "a", "scene.", "Check", "it,", "in", "2026!"]
    assert words[2] == {"word": "scene.", "start": 10.6125, "end": 11.15}  # not stretched over the pause after it
    assert words[5] == {"word": "in", "start": 11.65, "end": 11.65}  # untimed: the end of the word before
    assert words[6]["end"] == 13.55


def test_repeat_renders_reuse_warm_workers_until_idle(tmp_path, monkeypatch):
    import time
    from kinescript import output
    monkeypatch.setattr(output, "WARM", 0.5)
    spec = scene([{"type": "rect", "w": 100, "h": 60, "x": "=t*200", "y": 90}], duration=1)
    k.render(spec, str(tmp_path / "a.mp4"), quality="draft", workers=2)
    first = set(output._POOL["ex"]._processes)
    k.render(spec, str(tmp_path / "b.mp4"), quality="draft", workers=2)
    assert first <= set(output._POOL["ex"]._processes)  # the same processes drew the second render
    if os.environ.get("KINE_BACKEND") == "cpu":  # change the setting whichever backend this run uses
        monkeypatch.delenv("KINE_BACKEND")
    else:
        monkeypatch.setenv("KINE_BACKEND", "cpu")
    k.render(spec, str(tmp_path / "c.mp4"), quality="draft", workers=2)
    assert not first & set(output._POOL["ex"]._processes)  # started again with the new backend setting
    time.sleep(1.5)
    assert "ex" not in output._POOL  # shut down once idle


def test_failed_render_or_warm_zero_shuts_the_pool_down(tmp_path, monkeypatch):
    from kinescript import output
    output._pool(1)
    output._rest(failed=True)
    assert "ex" not in output._POOL and output._POOL["busy"] == 0
    monkeypatch.setattr(output, "WARM", 0)  # what `kine render` sets: a one-shot process times its own shutdown
    k.render(scene([{"type": "rect"}], duration=1), str(tmp_path / "o.mp4"), quality="draft", workers=2)
    assert "ex" not in output._POOL


def test_media_caches_read_a_replaced_file_again(tmp_path):
    import skia
    p = tmp_path / "img.png"

    def write(w, colour):
        s = skia.Surface(w, 8)
        s.getCanvas().clear(colour)
        s.makeImageSnapshot().save(str(p), skia.kPNG)

    write(8, skia.ColorRED)
    first = media.image(str(p))
    assert media.image(str(p)) is first  # unchanged: decoded once
    write(16, skia.ColorBLUE)
    assert media.image(str(p)).width() == 16 and media.probe(str(p))["w"] == 16


def test_captions_from_srt(tmp_path):
    srt = tmp_path / "c.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n2\n00:00:01,000 --> 00:00:02,500\nSecond line here\n")
    spec = {"canvas": {"w": 320, "h": 180, "fps": 30}, "layers": [{"id": "cap", "type": "captions", "src": "c.srt",
                                                                     "mode": "karaoke", "size": 16}]}
    (tmp_path / "s.json").write_text(json.dumps(spec))
    s = k.load(str(tmp_path / "s.json"))
    assert s.duration == pytest.approx(2.5)
    out = probe(str(tmp_path / "s.json"), at=1.5)
    assert out["visible"]["cap"]["text"] == "Second line here"


def test_arrow_draws_head_at_tip():
    spec = scene([{"type": "arrow", "x1": 20, "y1": 90, "x2": 300, "y2": 90, "stroke": "#ffffff", "strokeWidth": 4,
                   "head": 20}], bg="#000000", duration=1)
    f = _frame(spec, 10)
    assert f[90, 150, 0] > 200 and f[84, 290, 0] > 100  # shaft, and a head stroke just above the tip


def test_snap_motion_cuts_in_without_blur():
    from kinescript.presets import parse
    m = parse({"name": "snap", "dur": 0.3})
    start, early, end = m.mods(0.0), m.mods(0.25), m.mods(1.0)
    assert start.alpha == 0 and start.scale > 1.0
    assert early.alpha == 1.0  # fully visible after the first fifth of the move
    assert end.alpha == 1.0 and end.scale == pytest.approx(1.0) and end.blur == 0


def test_score_shader_moves_in_whole_beats():
    # rate 2 beats/s at 30 fps: frames 3 and 6 share a beat, frame 18 is in the next one.
    spec = scene([{"type": "shader", "preset": "score", "uniforms": {"rate": 2}}], duration=1)
    a, b, c = _frame(spec, 3), _frame(spec, 6), _frame(spec, 18)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_animated_weight_frame_is_independent_of_history():
    spec = scene([{"type": "text", "text": "Weight", "x": "50%", "y": "50%", "size": 60,
                   "weight": [[0, 300], [1, 700, "linear"]]}], duration=1)
    r = Renderer(k.Scene(spec), backend="cpu")
    for n in range(0, 30):
        r.frame(n)
    after_history = r.frame(15).copy()
    assert np.array_equal(after_history, _frame(spec, 15))


def test_doctor_reports_version_and_font(capsys):
    from kinescript.cli import main
    assert main(["doctor"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["version"] == k.__version__ and info["font"] == "Space Grotesk", info
