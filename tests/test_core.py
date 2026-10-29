import json
import subprocess

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


def test_timing_cycle_is_reported():
    tl = Timeline(30)
    tl.add("a", "b.in")
    tl.add("b", "a.in")
    with pytest.raises(TimingError, match="cycle"):
        tl.solve()


def test_unknown_reference_lists_known_ids():
    with pytest.raises(k.SceneError, match="Known ids"):
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


def test_render_mp4_with_audio(clip, tmp_path):
    out = tmp_path / "o.mp4"
    spec = scene([{"type": "video", "src": clip, "trim": [0, 1]},
                  {"type": "text", "text": "Title", "x": "50%", "y": "50%", "size": 30, "enter": "pop"}])
    res = k.render(spec, str(out), quality="draft")
    assert res["frames"] == 30
    info = media.probe.__wrapped__(str(out))
    assert info["audio"] and info["duration"] == pytest.approx(1.0, abs=0.1)


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
