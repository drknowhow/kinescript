import json
import os
import subprocess
import sys

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


def test_sequence_crossfades_and_video_renders(clip):
    spec = scene([{"type": "sequence", "id": "cut", "clips": [{"src": clip, "trim": [0, 1.5]}, {"src": clip, "trim": [1, 3]}],
                   "transition": {"type": "fade", "dur": 0.5}}])
    s = k.Scene(spec)
    assert s.by_id["cut-2"].t_in == pytest.approx(1.0) and s.duration == pytest.approx(3.0)
    f = Renderer(s, backend="cpu").frame(45)
    assert f[..., :3].std() > 20  # test pattern is visible, not a black frame
    s.close()


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


def test_audio_mixes_while_frames_render_and_stops_when_they_fail(clip, tmp_path, monkeypatch):
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
    spec["audio"] = [{"src": clip, "at": t} for t in range(0, 60, 3)]  # a minute of audio takes a while to mix
    with pytest.raises(RuntimeError, match="frames failed"):
        k.render(spec, str(tmp_path / "o.mp4"), quality="draft", workers=1)
    assert started[0][0].poll() is not None  # stopped, not left running


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
