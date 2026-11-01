import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kinescript as k
from kinescript import ease, media, svgpath
from kinescript.presets import turns
from kinescript.props import Ctx, compile_prop
from kinescript.render import _GL, Renderer
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


def test_keys_at_the_same_time_jump_and_any_key_count_interpolates_its_own_segment():
    p = compile_prop("x", [[0, 0], [1, 10, "linear"], [1, 50], [2, 60, "linear"]], Ctx(100, 100, 30))
    assert [p.at(t, 2) for t in (0.5, 1.0, 1.5, 3.0)] == [pytest.approx(5), 50, pytest.approx(55), 60]
    keys = [[i * 0.1, (i * 37) % 11, "linear"] for i in range(121)]
    dense = compile_prop("x", keys, Ctx(100, 100, 30))
    for t in (0.0, 0.05, 3.33, 7.95, 12.0, 13.0):
        j = min(int(t / 0.1), 119)
        u = min(1.0, (t - j * 0.1) / 0.1)
        assert dense.at(t, 12) == pytest.approx(keys[j][1] + (keys[j + 1][1] - keys[j][1]) * u)


def test_smooth_keys_pass_through_each_key_without_stopping_or_overshooting():
    ctx = Ctx(100, 100, 30)
    p = compile_prop("x", {"keys": [[0, 0], [1, 100, "outBack"], [2, 300], [3, 300], [4, 0]], "smooth": True}, ctx)
    v = [p.at(n / 100, 4) for n in range(401)]
    assert (v[0], v[100], v[200], v[300], v[400]) == (0, pytest.approx(100), pytest.approx(300), 300, 0)
    assert v[101] - v[99] > 2  # moving through the key at 1 s; the outBack on it is ignored
    assert all(b >= a for a, b in zip(v[:200], v[1:201])) and max(v) == pytest.approx(300) and min(v) == 0
    assert v[1] - v[0] < 0.1 and v[201] - v[199] < 0.1  # at rest where the run starts and where it turns
    held = compile_prop("x", {"keys": [[0, 0], [1, 100], [2, 200, "hold"], [3, 300], [3, 0], [4, 50]],
                              "smooth": True}, ctx)
    assert [held.at(t, 4) for t in (1.5, 1.99, 2.0, 3.0)] == [100, 100, 200, 0]  # a hold, and keys at one time jump
    assert 200 < held.at(2.5, 4) < 300
    with pytest.raises(ValueError, match="smooth"):
        compile_prop("fill", {"keys": [[0, "#000000"], [1, "#ffffff"]], "smooth": True}, ctx)


def test_check_warns_of_a_key_that_stops_a_move_and_smooth_or_linear_keys_clear_it():
    def issues(x):
        spec = scene([{"id": "dot", "type": "circle", "r": 10, "y": 90, "x": x}], duration=3)
        return [i for i in check(spec)["issues"] if i["code"] == "stop-and-go"]
    (hit,) = issues([[0, 20], [1, 120], [2, 300]])
    assert (hit["layer"], hit["from"]) == ("dot", 1) and "smooth" in hit["fix"]
    assert not issues({"keys": [[0, 20], [1, 120], [2, 300]], "smooth": True})
    assert not issues([[0, 20], [1, 120, "linear"], [2, 300]])  # moving into the key
    assert not issues([[0, 20], [1, 120], [2, 20]])  # a turn stops it anyway
    assert not issues([[0, 20], [1, 120], [1.5, 120], [2.5, 300]])  # a pause written as a held key
    assert not issues([[0, 20], [1, 120], [2, 300, "hold"]])


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
    with pytest.raises(k.SceneError, match="too large for a float"):
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


RED_BLUE = ["#ff0000", "#0000ff"]


def _lit(f, over=150):
    """(xs, colours) of the pixels brighter than `over` in some channel."""
    ys, xs = np.nonzero(f[..., :3].max(-1) > over)
    return xs, f[ys, xs, :3].astype(int)


def test_text_takes_a_gradient_fill_and_stroke_and_keeps_its_emphasis_colour():
    text = {"id": "t", "type": "text", "text": "HHHH **HH**", "x": "50%", "y": "50%", "size": 60, "weight": 800,
            "fill": {"angle": 90, "stops": RED_BLUE}, "emph": "#00ff00"}
    xs, cols = _lit(_frame(scene([text], bg="#000000"), 0), 200)
    green = (cols[:, 1] > 200) & (cols[:, 0] < 60)
    assert cols[xs < xs.min() + 20].mean(0)[0] > 200  # red at its left: it used to draw in the paper colour
    assert green.any() and xs[green].min() > 160 and not (cols.min(1) > 200).any()
    outline = dict(text, text="HHHHHH", fill="#000000", stroke={"linear": [[0, 0], [1, 0]], "stops": RED_BLUE},
                   strokeWidth=3)
    xs, cols = _lit(_frame(scene([outline], bg="#000000"), 0))
    left, right = cols[xs < xs.min() + 10].mean(0), cols[xs > xs.max() - 10].mean(0)
    assert left[0] > 200 > left[2] and right[2] > 200 > right[0]


@pytest.mark.parametrize("fill, red, blue", [
    ({"sweep": True, "stops": RED_BLUE}, (30, 165), (30, 155)),  # from the top, clockwise
    ({"sweep": [0.5, 0.5], "angle": 90, "stops": RED_BLUE}, (95, 220), (85, 220)),  # from the right
])
def test_a_sweep_gradient_turns_round_its_middle(fill, red, blue):
    f = _frame(scene([{"id": "c", "type": "circle", "x": 160, "y": 90, "r": 80, "fill": fill}], bg="#000000"), 0)
    assert f[red][0] > 240 > 15 > f[red][2] and f[blue][2] > 240 > 15 > f[blue][0]


@pytest.mark.parametrize("angle, red, blue", [(0, (138, 160), (42, 160)), (90, (90, 62), (90, 257)),
                                              (180, (42, 160), (138, 160))])
def test_an_angle_gradient_runs_as_css_reads_one(angle, red, blue):
    f = _frame(scene([{"id": "r", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 100,
                       "fill": {"angle": angle, "stops": RED_BLUE}}], bg="#000000"), 0)
    assert f[red][0] > 240 > 15 > f[red][2] and f[blue][2] > 240 > 15 > f[blue][0]


def test_a_gradient_animates_into_another():
    tween = {"from": {"angle": 90, "stops": RED_BLUE}, "to": {"angle": 90, "stops": RED_BLUE[::-1]}, "dur": 1,
             "ease": "linear"}
    ring = [[0, {"sweep": True, "stops": RED_BLUE}], [1, {"sweep": True, "angle": 180, "stops": RED_BLUE}, "linear"]]
    spec = scene([{"id": "r", "type": "rect", "x": 100, "y": 90, "w": 120, "h": 100, "fill": tween},
                  {"id": "c", "type": "circle", "x": 250, "y": 90, "r": 50, "fill": None, "strokeWidth": 10,
                   "stroke": ring}], bg="#000000", duration=1)
    assert check(spec)["ok"]
    r = Renderer(k.Scene(spec), backend="cpu")
    first, mid, last = (r.frame(n).astype(int) for n in (0, 15, 29))
    assert first[90, 42][0] > 240 and last[90, 42][2] > 240 and 60 < mid[90, 42][0] < 200  # the rect's left edge
    assert first[44, 252][0] > 240 and last[136, 248][0] > 240  # the ring starts at the top, then at the bottom
    assert np.array_equal(_frame(spec, 15), mid)  # a pure function of the frame


def _combine(op, kids, **group):
    return dict({"id": "g", "type": "group", "x": 160, "y": 90, "combine": op, "fill": "#ffffff", "children": kids},
                **group)


TWO_DISCS = [{"id": "a", "type": "circle", "x": -40, "r": 60}, {"id": "b", "type": "circle", "x": 40, "r": 60}]


@pytest.mark.parametrize("op, left, middle, right", [
    ("union", True, True, True), ("subtract", True, False, False), ("intersect", False, True, False),
    ("xor", True, False, True),
])
def test_combine_merges_a_groups_shapes_into_one(op, left, middle, right):
    f = _frame(scene([_combine(op, TWO_DISCS)], bg="#000000"), 0)
    assert [f[90, x, 0] > 200 for x in (80, 160, 240)] == [left, middle, right]


def test_a_combined_shape_strokes_only_its_outline_and_takes_text_lines_and_nested_groups():
    union = _combine("union", TWO_DISCS, fill="#000000", stroke="#ffffff", strokeWidth=6)
    f = _frame(scene([union], bg="#000000"), 0)
    assert f[90, 220, 0] < 20 and f[90, 140, 0] < 20 and f[90, 60, 0] > 200  # no stroke where the discs overlap
    cut = _combine("subtract", [{"id": "plate", "type": "rect", "w": 260, "h": 110},
                                {"id": "word", "type": "text", "text": "IIII", "size": 90, "weight": 800}])
    plate = _combine("subtract", cut["children"][:1])
    lit, whole = (_frame(scene([s], bg="#000000"), 0)[..., 0] > 200 for s in (cut, plate))
    assert 0.5 < lit.sum() / whole.sum() < 0.95 and not (lit & ~whole).any()  # the letters are holes in the plate
    bar = _combine("subtract", [{"id": "plate", "type": "rect", "w": 200, "h": 100},
                                {"id": "inner", "type": "group", "children": [
                                    {"id": "line", "type": "line", "points": [[-120, 0], [120, 0]], "strokeWidth": 10}]}])
    f = _frame(scene([bar], bg="#000000"), 0)
    assert f[90, 160, 0] < 20 and f[60, 160, 0] > 200  # a line takes the outline of its stroke


def test_a_combining_group_merges_again_only_when_a_child_changes():
    still = scene([_combine("subtract", [{"id": "p", "type": "rect", "w": 200, "h": 100},
                                         {"id": "h", "type": "circle", "r": 30}]),
                   _combine("subtract", [{"id": "q", "type": "rect", "w": 200, "h": 100},
                                         {"id": "m", "type": "circle", "r": 30, "x": [[0, -80], [1, 80]]}],
                            id="moving", y=40)], bg="#000000", duration=1)
    r = Renderer(k.Scene(still), backend="cpu")
    r.frame(0)
    kept, moved = r._combos["g"][1][-1], r._combos["moving"][1]
    f = r.frame(15)
    assert r._combos["g"][1][-1] is kept and r._combos["moving"][1][0] is moved[0]  # the rect, not merged again
    assert r._combos["moving"][1][-1] is not moved[-1]
    assert np.array_equal(f, _frame(still, 15))


def test_check_does_not_read_combined_text_for_contrast():
    badge = _combine("subtract", [{"id": "plate", "type": "rect", "w": 260, "h": 110},
                                  {"id": "word", "type": "text", "text": "SALE", "size": 60}], fill="#ff633f")
    issues = check(scene([badge], bg="#f2f0e8"))["issues"]
    assert not [i for i in issues if i["code"] in ("contrast", "text-hidden")], issues


def test_check_reads_masked_text_for_contrast_where_its_mask_shows_it():
    def contrast(window):
        layers = [{"id": "plate", "type": "rect", "x": 60, "y": 90, "w": 120, "h": 120, "fill": "#ffffff"},
                  {"id": "win", "type": "rect", "anchor": "left", "y": 90, "w": 160, "h": 180, **window},
                  {"id": "word", "type": "text", "text": "HELLO", "x": 160, "y": 90, "size": 40, "fill": "#ffffff",
                   "mask": "win"}]
        return [i["code"] for i in check(scene(layers, bg="#000000", duration=2))["issues"]].count("contrast")
    assert contrast({"x": 0}) == 1  # it shows over the white plate
    assert contrast({"x": 125}) == 0  # it shows only over the black past the plate
    assert contrast({"x": [[0, 400], [1, 400], [2, 125]]}) == 0  # nothing shows at the time check reads it


TRIANGLE, DISC = "M0 -70 L70 50 L-70 50 Z", "M0 -70 A70 70 0 1 1 -0.1 -70 Z"
SQUARE, RING = "M-60 -60 H60 V60 H-60 Z", "M-70 0 A70 70 0 1 0 70 0 A70 70 0 1 0 -70 0 Z M-35 0 A35 35 0 1 1 35 0 "     "A35 35 0 1 1 -35 0 Z"


def _morphing(d, **kw):
    return dict({"id": "m", "type": "path", "x": 160, "y": 90, "fill": "#ffffff", "d": d}, **kw)


def test_a_path_morphs_through_its_d_keyframes_and_is_each_key_on_its_time():
    spec = scene([_morphing([[0, TRIANGLE], [1, DISC, "linear"], [2, SQUARE]])], bg="#000000", duration=2)
    r = Renderer(k.Scene(spec), backend="cpu")
    for n, d in ((0, TRIANGLE), (30, DISC)):
        assert np.array_equal(r.frame(n), _frame(scene([_morphing(d)], bg="#000000", duration=2), 0)), n
    tri, disc = (_frame(scene([_morphing(d)], bg="#000000"), 0)[..., 0] > 128 for d in (TRIANGLE, DISC))
    mid = r.frame(15)[..., 0] > 128
    assert tri.sum() < mid.sum() < disc.sum() and (mid & tri).sum() > 0.9 * tri.sum()  # it grows from one to the other
    assert np.array_equal(_frame(spec, 15)[..., 0] > 128, mid)  # a pure function of the frame
    pair = r._morphs[("m", 0)]
    r.frame(20)
    assert r._morphs[("m", 0)] is pair  # matched once for the pair, then only mixed


def test_a_shape_with_no_partner_grows_from_the_middle_and_open_paths_morph():
    spec = scene([_morphing([[0, SQUARE], [1, RING, "linear"]])], bg="#000000", duration=1.5)
    r = Renderer(k.Scene(spec), backend="cpu")
    first, mid, last = (r.frame(n)[..., 0] > 128 for n in (0, 15, 30))
    assert first[90, 160] and not mid[90, 160] and not last[90, 160] and mid[90, 160 + 25] and not last[90, 160 + 25]
    wave = _morphing([[0, "M-100 0 L100 0"], [1, "M-100 0 C-40 -120 40 120 100 0", "linear"]], fill=None,
                     stroke="#ffffff", strokeWidth=4)
    f = _frame(scene([wave], bg="#000000", duration=1), 15)[..., 0]
    assert f[90, 70:100].max() < 20 and f[:90, 110:140].max() > 200  # off the straight line, above it on the left


def test_morph_matches_any_two_paths_point_for_point_from_where_they_move_least():
    from kinescript import morph, svgpath
    for a, b in ((TRIANGLE, DISC), (SQUARE, RING), ("M0 0 L100 50", "M0 0 Q50 -60 100 50"), (RING, TRIANGLE)):
        pa, pb = morph.match(svgpath.parse(a), svgpath.parse(b))
        assert pa.isInterpolatable(pb), (a, b)
    turned = "M60 -60 V60 H-60 V-60 Z"  # the same square, started at another corner and drawn the other way
    pa, pb = morph.match(svgpath.parse(SQUARE), svgpath.parse(turned))
    a, b = (np.array([[p.x(), p.y()] for p in q.getPoints(q.countPoints())]) for q in (pa, pb))
    assert np.abs(a - b).max() < 2  # each point stays where it is


DASHED = {"id": "l", "type": "line", "points": [[20, 90], [300, 90]], "strokeWidth": 6, "cap": "butt",
          "stroke": "#ffffff", "dash": [20, 20]}


@pytest.mark.parametrize("extra, lit", [
    ({}, (25, 105, 115)),
    ({"dash": [20]}, (25, 105, 115)),  # an odd list repeats, as in SVG
    ({"dashOffset": -10}, (45, 115)),  # the pattern 10 px further along
    ({"drawStart": 0.25}, (105, 115)),  # the dashes stay where they lie on the whole line
])
def test_dash_marks_out_a_stroke_and_dash_offset_slides_it(extra, lit):
    f = _frame(scene([dict(DASHED, **extra)], bg="#000000"), 0)
    assert [x for x in (25, 45, 95, 105, 115) if f[90, x, 0] > 200] == list(lit)


def test_a_dashed_arrow_keeps_its_head_whole():
    arrow = {"id": "a", "type": "arrow", "x1": 40, "y1": 90, "x2": 280, "y2": 90, "strokeWidth": 4,
             "stroke": "#ffffff", "head": 22, "dash": [6, 30]}
    f = _frame(scene([arrow], bg="#000000"), 0)
    assert f[95, 271, 0] > 200 and f[85, 271, 0] > 200  # 10 px down each side of the head, inside a gap of the dash


def test_corner_rounds_a_paths_corners():
    square = {"id": "p", "type": "path", "points": [[0, 0], [200, 0], [200, 120], [0, 120]], "closed": True,
              "x": 60, "y": 30, "fill": "#ffffff"}
    sharp, round_ = (_frame(scene([dict(square, **extra)], bg="#000000"), 0) for extra in ({}, {"corner": 30}))
    assert sharp[31, 61, 0] > 200 and round_[31, 61, 0] < 20 and round_[90, 61, 0] > 200


def test_rough_wobbles_a_shape_the_same_for_the_same_seed_and_boils_on_time():
    def frames(rough, *ns):
        layer = {"id": "r", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 100, "fill": "#ffffff", "rough": rough}
        r = Renderer(k.Scene(scene([layer], bg="#000000", duration=1)), backend="cpu")
        return [r.frame(n).astype(int)[..., 0] for n in ns]
    smooth = _frame(scene([{"id": "r", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 100, "fill": "#ffffff"}],
                          bg="#000000"), 0).astype(int)[..., 0]
    one, again, two = frames({"amount": 4, "seed": 1}, 0, 20)[0], *frames({"amount": 4, "seed": 1}, 0),         frames({"amount": 4, "seed": 2}, 0)[0]
    assert np.array_equal(one, again) and not np.array_equal(one, two) and not np.array_equal(one, smooth)
    assert (one[46:134, 66:254] > 200).all() and (one[:34] < 20).all() and (one[:, :54] < 20).all()  # within 4 px
    still, a, b, c = frames({"amount": 4, "seed": 1, "boil": 4}, 0, 0, 7, 8)  # boil 4 at 30 fps: anew at frame 8
    assert np.array_equal(a, b) and not np.array_equal(b, c)
    assert np.array_equal(frames({"amount": 4, "seed": 1}, 0, 25)[0], frames({"amount": 4, "seed": 1}, 25)[0])


def test_a_rough_line_stays_open():
    line = {"id": "l", "type": "line", "points": [[20, 40], [160, 150], [300, 40]], "strokeWidth": 4,
            "stroke": "#ffffff", "rough": {"amount": 2, "seed": 4}}
    f = _frame(scene([line], bg="#000000"), 0)
    assert f[40:46, 100:220, 0].max() < 20 and f[140:156, 150:170, 0].max() > 200  # no stroke back to its start


@pytest.mark.parametrize("layer", [
    {"id": "r", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 100, "fill": "#ffffff", "radius": [40, 0, 40, 0]},
    {"id": "g", "type": "group", "x": 160, "y": 90, "w": 200, "h": 100, "anchor": "center", "fill": "#ffffff",
     "radius": [40, 0, 40, 0], "children": []},
    {"id": "r", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 100, "fill": "#ffffff",
     "radius": [[0, [0, 0, 0, 0]], [1, [40, 0, 40, 0], "linear"]]},
])
def test_radius_takes_four_corners(layer):
    f = _frame(scene([layer], bg="#000000", duration=1), 29)
    assert [f[y, x, 0] > 200 for y, x in ((41, 61), (41, 258), (138, 258), (138, 61))] == [False, True, False, True]


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
        {"id": "w", "type": "text", "text": "one two three four", "y": 150, "x": "50%", "size": 24, "split": "words",
         "enter": {"name": "rise", "stagger": 0.1, "order": "random", "seed": 3}},
        {"id": "r", "type": "rect", "x": 60, "y": 120, "w": 60, "h": 40, "fill": "#4a7dff",
         "rough": {"amount": 3, "seed": 2, "boil": 8}},
        {"id": "lens", "type": "circle", "r": 30, "x": "=160+sin(t*2)*80", "y": 90},
        {"id": "sheet", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 60, "fill": "#3ddbd9",
         "mask": {"layer": "lens", "feather": 6}},
        {"id": "heat", "type": "text", "text": "heat", "x": 260, "y": 30, "size": 28, "fill": "#ffd166",
         "glow": {"size": "=6+4*sin(t*3)"}, "outline": 2, "displace": {"amount": 3, "scale": 20, "speed": 2}},
    ], duration=2)
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 30)]
    alone = _frame(spec, 17)
    assert np.array_equal(sequential[17], alone)


def _card(**looks):
    """A red card, x 120 to 200 and y 70 to 110, on a black canvas."""
    return scene([dict({"id": "b", "type": "rect", "x": 160, "y": 90, "w": 80, "h": 40, "fill": "#ff0000"}, **looks)],
                 bg="#000000", duration=1)


def test_a_glow_lights_round_a_layer_in_its_own_colour_or_the_glows():
    plain, own = _frame(_card(), 0), _frame(_card(glow=24), 0)
    tinted = _frame(_card(glow={"color": "#00ff00", "size": 24}), 0)
    half = _frame(_card(glow={"size": 24, "amount": 0.5}), 0)
    assert plain[90, 208, :3].max() == 0
    assert own[90, 208, 0] > 40 and own[90, 208, 1:3].max() == 0  # 8 px past the edge
    assert tinted[90, 208, 1] > 40 and tinted[90, 208, 0] == 0
    assert abs(int(half[90, 208, 0]) - int(own[90, 208, 0]) / 2) <= 2
    assert tuple(own[90, 160, :3]) == tuple(tinted[90, 160, :3]) == (255, 0, 0)  # the layer itself, on top
    grows = _card(glow={"size": [[0, 0], [1, 30, "linear"]]})
    assert _frame(grows, 0)[90, 208, 0] == 0 and _frame(grows, 29)[90, 208, 0] > 40


def test_an_outline_rings_a_layer_outside_its_edge_and_an_inner_shadow_shades_inside_it():
    f = _frame(_card(outline={"width": 6, "color": "#ffffff"}), 0)
    assert tuple(f[90, 203, :3]) == (255, 255, 255) and f[90, 210, :3].max() == 0
    assert tuple(f[90, 160, :3]) == (255, 0, 0)
    spec = _card(fill="#ffffff", innerShadow={"x": 0, "y": 6, "blur": 4, "color": "#000000"})
    spec["canvas"]["bg"] = "#808080"
    f = _frame(spec, 0)
    assert f[72, 160, 0] < 30 and tuple(f[100, 160, :3]) == (255, 255, 255)  # dark under the top edge only
    assert tuple(f[65, 160, :3]) == (128, 128, 128)  # nothing outside


def test_displace_bends_a_layer_smoothly_and_its_waves_slide():
    still = _frame(_card(fill="#ffffff"), 0)
    spec = _card(fill="#ffffff", displace={"amount": 6, "scale": 30, "speed": 2})
    a, b = _frame(spec, 0), _frame(spec, 15)
    assert not np.array_equal(a, still) and not np.array_equal(a, b)
    assert len(np.unique(a[60:120, 190:210, 0])) > 10  # sampled between pixels: a bent edge is smooth, not in steps
    assert tuple(a[90, 160, :3]) == (255, 255, 255)
    held = _card(fill="#ffffff", displace={"amount": 6, "scale": 30, "speed": 0})
    assert np.array_equal(_frame(held, 0), _frame(held, 15))
    assert np.array_equal(_frame(_card(fill="#ffffff", displace={"amount": 0}), 0), still)
    mixed = _card(fill="#ff8000", blend="multiply", displace=4)
    mixed["canvas"]["bg"] = "#808080"
    assert np.abs(_frame(mixed, 0)[90, 160, :3].astype(int) - (128, 64, 0)).max() <= 1  # blends with what is behind


def test_a_masked_layer_is_displaced_inside_its_mask():
    spec = _masked("lens")
    spec["layers"][1]["displace"] = {"amount": 8, "scale": 20}
    f, flat = _frame(spec, 0), _frame(_masked("lens"), 0)
    assert np.array_equal(f[:, 200:], flat[:, 200:]) and f[:, :120, :3].max() == 0  # cut where the mask is
    assert tuple(f[90, 140, :3]) == (255, 255, 255)


def _masked(mask, **lens):
    """A white sheet over the left half of a black canvas, cut by `mask`; the lens, a circle at x 160, overlaps it
    from x 120 to 160 and stands off it from 160 to 200."""
    return scene([dict({"id": "lens", "type": "circle", "r": 40, "x": 160, "y": 90, "fill": "#ff0000"}, **lens),
                  {"id": "sheet", "type": "rect", "x": 80, "y": 90, "w": 160, "h": 180, "fill": "#ffffff",
                   "mask": mask}], bg="#000000", duration=1)


def test_a_mask_shows_a_layer_through_another_and_is_not_drawn_itself():
    f = _frame(_masked("lens"), 0)
    assert tuple(f[90, 140, :3]) == (255, 255, 255)  # the sheet, where the lens is
    assert f[90, 60, :3].max() == 0 and f[90, 180, :3].max() == 0  # nothing where it is not, nor the lens itself
    inv = _frame(_masked({"layer": "lens", "mode": "alphaInvert"}), 0)
    assert inv[90, 140, 0] == 0 and inv[90, 60, 0] == 255 and inv[90, 180, 0] == 0
    soft = _frame(_masked({"layer": "lens", "feather": 30}), 0)
    assert soft[90, 135, 0] > 200 and 60 < soft[90, 120, 0] < 200 and 0 < soft[90, 110, 0] < 60  # the edge fades
    half = _frame(_masked({"layer": "lens", "mode": "luma"}, fill="#808080"), 0)
    assert abs(int(half[90, 140, 0]) - 128) <= 2 and half[90, 60, 0] == 0  # grey keeps half
    rest = _frame(_masked({"layer": "lens", "mode": "lumaInvert"}, fill="#808080"), 0)
    assert abs(int(rest[90, 140, 0]) - 127) <= 2 and rest[90, 60, 0] == 255
    faint = _frame(_masked("lens", opacity=0.5), 0)
    assert abs(int(faint[90, 140, 0]) - 128) <= 2  # a mask's opacity is how much it keeps


def test_a_masked_layer_blends_with_what_is_behind_it():
    for mode in ("alpha", "alphaInvert"):
        spec = scene([{"id": "lens", "type": "rect", "x": 80, "y": 90, "w": 160, "h": 180},
                      {"id": "m", "type": "rect", "x": 160, "y": 90, "w": 320, "h": 180, "fill": "#ff8000",
                       "blend": "multiply", "mask": {"layer": "lens", "mode": mode}}], bg="#808080", duration=1)
        f = _frame(spec, 0)
        kept = (40, 280) if mode == "alpha" else (280, 40)
        assert np.abs(f[90, kept[0], :3].astype(int) - (128, 64, 0)).max() <= 1, (mode, f[90, kept[0]])
        assert tuple(f[90, kept[1], :3]) == (128, 128, 128), mode
    alone = scene([{"id": "m", "type": "rect", "x": 160, "y": 90, "w": 320, "h": 180, "fill": "#ff8000",
                    "blend": "multiply", "mask": {"layer": "lens", "mode": "alphaInvert"}},
                   {"id": "lens", "type": "rect", "x": 80, "y": 90, "w": 160, "h": 180, "at": 0.5}],
                  bg="#808080", duration=1)
    assert np.abs(_frame(alone, 0)[90, 40, :3].astype(int) - (128, 64, 0)).max() <= 1  # no mask yet: all of it


def test_a_mask_moves_on_its_own_clock_and_cuts_motion_blur_as_it_moves():
    spec = _masked("lens", at=0.5, x=[[0, 40], [0.5, 140, "linear"]])
    r = Renderer(k.Scene(spec), backend="cpu")
    assert r.frame(0)[..., :3].max() == 0  # before the mask starts, nothing shows through it
    assert r.frame(15)[90, 40, 0] == 255 and r.frame(15)[90, 140, 0] == 0
    assert r.frame(29)[90, 140, 0] == 255 and r.frame(29)[90, 40, 0] == 0
    # A sheet that holds still behind a moving mask is drawn on every sub-frame: it smears as a moving disc does.
    blur = {"samples": 8, "shutter": 0.5}
    moving = dict(_masked("lens", x=[[0, 40], [1, 280, "linear"]], fill="#ffffff"), motionBlur=blur)
    disc = dict(scene([{"id": "d", "type": "circle", "r": 40, "y": 90, "x": [[0, 40], [1, 280, "linear"]],
                        "fill": "#ffffff"},
                       {"id": "cover", "type": "rect", "x": 240, "y": 90, "w": 160, "h": 180, "fill": "#000000"}],
                      bg="#000000", duration=1), motionBlur=blur)
    a, b = _frame(moving, 10), _frame(disc, 10)
    edge = a[90, 70:90, 0]  # the disc's left edge, at x 80 on this frame
    assert ((edge > 0) & (edge < 255)).any() and np.abs(a.astype(int) - b).max() <= 2


def test_check_warns_when_a_mask_never_overlaps_its_layer():
    def misses(spec):
        return [i["layer"] for i in check(spec)["issues"] if i["code"] == "mask-misses"]
    assert misses(_masked("lens")) == []
    assert misses(_masked("lens", x=300)) == ["sheet"]
    assert misses(_masked({"layer": "lens", "mode": "alphaInvert"}, x=300)) == []  # all of it shows then


def test_check_sees_the_mask_over_a_clip_it_does_not_decode(clip):
    spec = scene([{"id": "word", "type": "text", "text": "CLIP", "size": 80, "x": 160, "y": 90},
                  {"id": "v", "type": "video", "src": clip, "x": 160, "y": 90, "anchor": "center", "w": 320, "h": 180,
                   "mask": "word"}], duration=1)
    assert not [i for i in check(spec)["issues"] if i["code"] == "mask-misses"]
    cut, whole = _frame(spec, 0), _frame(dict(spec, layers=[dict(spec["layers"][1], mask=None)]), 0)
    same = (cut == whole).all(axis=2)
    assert 1000 < same.sum() < same.size / 2 and not same[5, 5] and not same[175, 315]  # the clip in the word only


def _boxes(spec, T=0.0):
    r = Renderer(k.Scene(spec), track=True, backend="cpu")
    r.draw(T)
    return r


def test_check_warns_of_a_move_too_fast_to_follow_unless_motion_blur_smears_it():
    def strobes(spec):
        return {(i["layer"], i["from"]) for i in check(spec)["issues"] if i["code"] == "strobe"}
    spec = scene([{"id": "zip", "type": "circle", "r": 5, "y": 90, "x": [[0.5, 0], [1, 320, "linear"]]},  # 21 px/f
                  {"id": "cut", "type": "circle", "r": 5, "y": 40, "x": [[0, 20], [1, 20], [1, 300]]},
                  {"id": "drift", "type": "circle", "r": 5, "y": 140, "x": [[0, 20], [2, 300, "linear"]]},
                  {"id": "rider", "type": "circle", "r": 5, "along": {"d": "M0 0 L320 180", "p": "=t * 2"}}],
                 duration=2)
    assert strobes(spec) == {("zip", 0.5), ("rider", 0)}
    assert not strobes(dict(spec, motionBlur={"samples": 4}))


def test_stagger_orders_put_each_unit_in_its_turn():
    assert turns(5, "start") == [0, 1, 2, 3, 4] and turns(5, "end") == [4, 3, 2, 1, 0]
    assert turns(5, "center") == [2, 1, 0, 1, 2] and turns(4, "center") == [1, 0, 0, 1]
    assert turns(5, "edges") == [0, 1, 2, 1, 0] and turns(4, "edges") == [0, 1, 1, 0]
    shuffled = turns(8, "random", seed=3)
    assert sorted(shuffled) == list(range(8)) and shuffled == turns(8, "random", seed=3) != turns(8, "random", 4)


def test_a_group_with_a_staggered_enter_brings_its_children_in_one_by_one():
    def lit(order, n):
        kids = [{"id": f"k{j}", "type": "rect", "x": 60 + j * 100, "y": 90, "w": 40, "h": 40, "fill": "#ffffff"}
                for j in range(3)]
        spec = scene([{"id": "row", "type": "group", "children": kids,
                       "enter": {"name": "fade", "dur": 0.2, "stagger": 0.3, "order": order}}], bg="#000000",
                     duration=1)
        img = _frame(spec, n)
        return [int(img[90, 60 + j * 100, 0] > 128) for j in range(3)]
    assert lit("start", 8) == [1, 0, 0] and lit("start", 17) == [1, 1, 0] and lit("start", 26) == [1, 1, 1]
    assert lit("end", 8) == [0, 0, 1] and lit("center", 8) == [0, 1, 0] and lit("edges", 8) == [1, 0, 1]
    s = k.Scene(scene([{"id": "g", "type": "group", "enter": {"name": "rise", "dur": 0.2, "stagger": 0.3},
                        "children": [{"type": "rect"}, {"type": "rect"}, {"type": "rect"}]}], duration=2))
    r = Renderer(s, backend="cpu")
    assert not r._holds(s.by_id["g"], 0.5, 0.52, 0.51) and r._holds(s.by_id["g"], 0.9, 0.92, 0.91)


def test_a_group_runs_its_children_on_its_own_clock():
    kid = {"id": "kid", "type": "rect", "w": 10, "h": 10, "y": 90, "x": [[0, 0], [1, 100, "linear"]]}

    def x(T, **clock):
        spec = scene([dict({"id": "g", "type": "group", "children": [kid]}, **clock)], duration=3)
        return _middle(_boxes(spec, T), "kid")
    assert (50, 90, 10) == x(0.25, speed=2) and (50, 90, 10) == x(1.5, loop=1) and (50, 90, 10) == x(2.5, loop=1)
    assert (75, 90, 10) == x(2.0, timeRemap=0.75)  # a number freezes them
    assert (25, 90, 10) == x(1.0, timeRemap=[[0, 0], [2, 0.5, "linear"]])
    assert (60, 90, 10) == x(1.0, timeRemap="=t * 0.6", speed=4)  # timeRemap wins over speed
    looped = scene([{"id": "g", "type": "group", "loop": 1, "children": [dict(kid, dur=1)]}], duration=3,
                   bg="#000000")
    for spec in (looped, dict(looped, motionBlur={"samples": 4})):  # a sub-frame sees the group's clock too
        assert _frame(spec, 75).max() > 0  # 2.5 s: the kid's 0.5 s again
        r = Renderer(k.Scene(spec), backend="cpu")
        sequential = [r.frame(n).copy() for n in range(0, 80, 5)]
        assert np.array_equal(sequential[-1], _frame(spec, 75))


def test_a_frozen_group_still_staggers_in_and_its_late_children_are_not_called_unseen():
    spec = scene([{"id": "g", "type": "group", "timeRemap": 0, "enter": {"name": "fade", "dur": 0.1, "stagger": 0.5},
                   "children": [{"id": f"k{j}", "type": "rect", "x": 60 + j * 100, "y": 90, "w": 40, "h": 40,
                                 "fill": "#ffffff"} for j in range(2)]}], duration=2, bg="#000000")
    img = _frame(spec, 30)
    assert img[90, 60, 0] > 128 and img[90, 160, 0] > 128
    late = scene([{"id": "g", "type": "group", "dur": 1, "speed": 3,
                   "children": [{"id": "late", "type": "rect", "at": 2, "dur": 0.5, "w": 10, "h": 10}]}], duration=2)
    assert "never-shown" not in [i["code"] for i in check(late)["issues"]]
    assert "late" in _boxes(late, 0.75).boxes


def test_split_text_enters_in_the_order_given():
    def middle_of_ink(order):
        spec = scene([{"id": "t", "type": "text", "text": "ABCDE", "x": 160, "y": 90, "size": 40, "split": "chars",
                       "fill": "#ffffff", "enter": {"name": "fade", "dur": 0.1, "stagger": 0.2, "order": order}}],
                     bg="#000000", duration=1)
        cols = np.nonzero(_frame(spec, 3)[:, :, 0].max(0) > 128)[0]
        return (cols.min() + cols.max()) / 2
    assert middle_of_ink("start") < 135 and 150 < middle_of_ink("center") < 170 and middle_of_ink("end") > 185


def _middle(r, lid):
    left, top, right, bottom = r.boxes[lid]["rect"]
    return pytest.approx(((left + right) / 2, (top + bottom) / 2, right - left), abs=0.5)


def test_a_layer_rides_its_path_round_a_corner_and_turns_with_it():
    def at(T, along, **kw):
        return _boxes(scene([dict({"id": "a", "type": "rect", "w": 20, "h": 6, "along": along}, **kw)], duration=2), T)
    line = {"d": "M40 40 L140 40 L140 140", "p": [[0, 0], [2, 1, "linear"]], "orient": True}  # 200 px long
    assert (90, 40, 20) == _middle(at(0.5, line), "a")
    assert (140, 90, 6) == _middle(at(1.5, line), "a")  # down the second side, turned to it
    assert (150, 85, 20) == _middle(at(1.5, dict(line, orient=False), x=10, y=-5), "a")  # x and y move the path
    assert (140, 140, 6) == _middle(at(0, dict(line, p=1.5)), "a")  # an open path stops at its end
    assert (140, 40, 20) == _middle(at(1, {"d": line["d"]}), "a")  # p runs 0 to 1 over the layer by default
    square = {"d": "M100 60 L140 60 L140 100 L100 100 Z", "p": 1.25}  # closed, 160 px: round again past 1
    assert (140, 60, 20) == _middle(at(0, square), "a")


def test_a_layer_rides_a_path_layer_where_it_is_drawn():
    route = {"id": "route", "type": "path", "d": "M0 0 L100 0", "anchor": "center", "x": 160, "y": 90, "rotate": 90,
             "scale": 0.5, "stroke": "#ffffff"}  # drawn from (160, 65) down to (160, 115)

    def at(p, **kw):
        dot = dict({"id": "dot", "type": "rect", "w": 20, "h": 6, "along": {"d": "#route", "p": p, "orient": True}},
                   **kw)
        return _boxes(scene([route, dot]))
    assert (160, 65, 6) == _middle(at(0), "dot")
    assert (165, 115, 6) == _middle(at(1, x=5), "dot")
    moving = scene([dict(route, x=[[0, 100], [2, 200]]), {"id": "dot", "type": "circle", "r": 6,
                                                          "along": {"d": "#route", "p": 0.5}}], duration=2)
    r = Renderer(k.Scene(moving), backend="cpu")
    assert not r._holds(r.s.by_id["dot"], 0.9, 1.1, 1.0)  # its path moves under it
    sequential = [r.frame(n).copy() for n in range(0, 40, 3)]
    assert np.array_equal(sequential[-1], _frame(moving, 39))


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


def _sides(frame):
    """Mean brightness of the left and right halves of a model at the centre of a 320 x 180 frame."""
    row = frame[60:120, :, :3].astype(float).sum(2)
    return row[:, 100:160].mean(), row[:, 160:220].mean()


def _key(light, **layer):
    return _model(dict({"shape": "sphere", "r": 60, "fill": "#808080", "shine": 0, "rim": 0,
                        "light": dict({"fill": 0, "ambient": 0}, **light)}, **layer), duration=1)


def test_light_azimuth_tweens_across_a_model():
    spec = _key({"azimuth": [[0, -80], ["end", 80]], "elevation": 0})
    first, last = _sides(_frame(spec, 0)), _sides(_frame(spec, 29))
    assert first[0] > 2 * first[1] and last[1] > 2 * last[0]  # lit from the left, then from the right
    assert _frame(_key({"from": "left"}), 0).tobytes() == _frame(_key({"from": "left", "follow": "camera"}), 0).tobytes()


def test_a_world_light_stays_put_as_the_camera_orbits():
    def frame(follow, ry):
        spec = _key({"from": "front", "follow": follow})
        spec["camera"] = {"rotateY": ry}
        return _frame(spec, 0)
    # A sphere lit from the camera looks the same from anywhere, but for where its triangles fall.
    assert np.abs(frame("camera", 0).astype(int) - frame("camera", 70)).mean() < 0.2
    left, right = _sides(frame("world", 70))
    assert left > 2 * right  # the camera went round to the right: the light is on the sphere's left now


def test_more_lights_add_their_colour_and_a_point_light_fades():
    dark = {"key": 0, "fill": 0, "ambient": 0, "rim": 0}
    red = _key(dark, lights=[{"from": "right", "color": "#ff0000", "intensity": 1}])
    left, right = _sides(_frame(red, 0))
    f = _frame(red, 0)
    assert right > 4 * max(left, 1) and f[90, 200, 0] > 100 and f[90, 200, 1:3].max() == 0

    def point(z):
        return _sides(_frame(_key(dark, lights=[{"type": "point", "x": 160, "y": 90, "z": z, "range": 200}]), 0))
    near, far = point(300), point(1200)
    assert sum(near) > 2 * sum(far) > 0


def test_glow_keeps_a_model_at_least_its_fill():
    f = _frame(_key({"key": 0, "rim": 0}, fill="#3366cc", glow=1), 0)
    assert tuple(f[90, 160, :3]) == (0x33, 0x66, 0xcc)  # unlit, it shows its fill
    dim = _frame(_key({"key": 0, "rim": 0}, fill="#3366cc", glow=0.5), 0)
    assert abs(int(dim[90, 160, 2]) - 0xcc // 2) <= 1


def test_gpu_and_cpu_agree_on_more_lights_and_glow():
    spec = _key({"azimuth": 30}, glow=0.3, lights=[{"from": "right", "color": "#ff5a36"},
                                                    {"type": "point", "x": 60, "y": 40, "z": 300, "color": "#4a7dff"}])
    cpu = _frame(spec, 0, "cpu").astype(int)
    try:
        gpu = _frame(spec, 0, "gpu").astype(int)
    except RuntimeError:
        pytest.skip("no GPU")
    assert np.abs(cpu - gpu).mean() < 1.0


def test_check_warns_when_the_key_light_is_behind_a_model():
    spec = _key({"azimuth": 180})
    assert [i["layer"] for i in check(spec)["issues"] if i["code"] == "light-behind"] == ["m"]
    assert not [i for i in check(_key({"azimuth": 60}))["issues"] if i["code"] == "light-behind"]


def _ball(**layer):
    return _model(dict({"shape": "sphere", "r": 60, "detail": "high", "fill": "#ffffff", "metal": 1, "shine": 1,
                        "light": {"key": 0, "fill": 0, "ambient": 0, "rim": 0}}, **layer), duration=1)


def test_a_chrome_ball_reflects_the_sky_above_and_the_floor_below():
    f = _frame(_ball(env={"top": "#ffffff", "horizon": "#808080", "bottom": "#000000"}), 0).astype(int)
    above, below = f[45:80, 140:180, :3].mean(), f[100:135, 140:180, :3].mean()
    assert above > 150 and below < 80
    red = _frame(_ball(env={"top": "#ff0000", "horizon": "#ff0000", "bottom": "#ff0000"}), 0)
    assert red[90, 160, 0] > 200 and red[90, 160, 1] < 20
    assert _frame(_ball(), 0)[60:120, 100:220, :3].max() == 0  # no env, no light: nothing to see


def test_a_matte_model_reflects_nothing():
    matte = dict(metal=0, shine=0, light="studio")
    assert _frame(_ball(**matte), 0).tobytes() == _frame(_ball(env="chrome", **matte), 0).tobytes()


def test_the_environment_stays_in_the_world_as_the_camera_tilts():
    def frame(rx):
        spec = _ball(env={"top": "#ffffff", "horizon": "#808080", "bottom": "#000000"})
        spec["camera"] = {"rotateX": rx}
        return _frame(spec, 0).astype(int)
    # Level, the ball's middle reflects the horizon; from above, it reflects the sky behind the camera.
    assert frame(50)[85:95, 150:170, :3].mean() > frame(0)[85:95, 150:170, :3].mean() + 60


def test_gpu_and_cpu_agree_on_reflections():
    spec = _ball(env="sunset", fill="#e0b54a", metal=0.8, shine=0.7, light="studio")
    cpu = _frame(spec, 0, "cpu").astype(int)
    try:
        gpu = _frame(spec, 0, "gpu").astype(int)
    except RuntimeError:
        pytest.skip("no GPU")
    assert np.abs(cpu - gpu).mean() < 1.5


FOG = {"color": "#0000ff", "near": 300, "far": 800}  # px from the camera, which is 247 px from z 0 at 320 x 180


def _fogged(z, fog, **layer):
    spec = _model(dict({"shape": "sphere", "r": 40, "light": "flat", "fill": "#ff0000", "z": z}, **layer), duration=1)
    spec["fog"] = fog
    return spec


def _floor(fog, sort=None):
    """A long red floor going away from the camera, alone in a group (a stage with sort "pixel")."""
    floor = {"id": "floor", "type": "rect", "w": 300, "h": 1200, "y": 40, "z": -400, "rotateX": 90, "fill": "#ff0000"}
    group = {"id": "g", "type": "group", "x": 160, "y": 90, "children": [floor]}
    if sort:
        group["sort"] = sort
    spec = scene([group], bg="#000000", duration=1)
    spec["fog"], spec["camera"] = fog, {"rotateX": 20}
    return spec


def test_fog_mixes_far_models_into_its_colour():
    near, mid, far = (_frame(_fogged(z, FOG), 0)[90, 160, :3].astype(int) for z in (0, -300, -1000))
    assert tuple(near) == (255, 0, 0) and tuple(far) == (0, 0, 255)
    assert 100 < mid[0] < 200 and 50 < mid[2] < 150 and mid[1] == 0
    assert any(i["code"] == "fog-hides" and i["layer"] == "m" for i in check(_fogged(-1000, FOG))["issues"])
    assert not any(i["code"] == "fog-hides" for i in check(_fogged(-300, FOG))["issues"])


def test_default_fog_is_the_background_and_starts_at_the_canvas_plane():
    def spec(z, fog=True):
        s = _fogged(z, fog)
        s["canvas"]["bg"] = "#204060"
        return s
    assert _frame(spec(0), 0).tobytes() == _frame(spec(0, None), 0).tobytes()
    assert tuple(_frame(spec(-5000), 0)[90, 160, :3]) == (0x20, 0x40, 0x60)
    assert tuple(_frame(spec(-5000, {}), 0)[90, 160, :3]) == (0x20, 0x40, 0x60)


def test_fog_mixes_a_flat_layer_by_the_depth_of_its_middle():
    def middle(**card):
        spec = scene([dict({"id": "c", "type": "rect", "x": 160, "y": 90, "w": 100, "h": 60, "z": -300,
                            "fill": "#ff0000"}, **card)], bg="#000000", duration=1)
        spec["fog"] = FOG
        return _frame(spec, 0)[90, 160, :3].astype(int)
    r, g, b = middle()
    assert abs(r - 129) <= 2 and g == 0 and abs(b - 126) <= 2  # (547 - 300) / 500 of the way to blue
    assert tuple(middle(stroke="#ffffff")) == (r, g, b)  # drawn in an offscreen layer: fogged once all the same


def test_a_stage_fogs_each_pixel_by_its_own_depth():
    flat, staged = (_frame(_floor(FOG, sort), 0)[:, 160, :3].astype(int) for sort in (None, "pixel"))
    assert np.array_equal(flat[50], flat[170])  # one mix for the whole layer
    assert staged[50, 2] > staged[50, 0] and tuple(staged[170]) == (255, 0, 0)  # far end blue, near end red


def test_gpu_and_cpu_agree_on_fog():
    for spec in (_fogged(-300, FOG, detail="high", light="studio"), _floor(FOG, "pixel"), _floor(FOG)):
        cpu = _frame(spec, 0, "cpu").astype(int)
        try:
            gpu = _frame(spec, 0, "gpu").astype(int)
        except RuntimeError:
            pytest.skip("no GPU")
        assert np.abs(cpu - gpu).mean() < 1.0


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


def _crossed(**group):
    """Two cards crossing in an X: the red one's left edge is near, the blue one's right edge."""
    kids = [{"id": "red", "type": "rect", "w": 160, "h": 80, "rotateY": 40, "fill": "#ff0000"},
            {"id": "blue", "type": "rect", "w": 160, "h": 80, "rotateY": -40, "fill": "#0000ff"}]
    return scene([dict({"id": "g", "type": "group", "x": 160, "y": 90, "children": kids}, **group)], bg="#000000")


def test_stage_shows_the_nearer_child_at_each_pixel():
    listed, staged = _frame(_crossed(), 0), _frame(_crossed(sort="pixel"), 0)
    assert tuple(listed[90, 130, :3]) == (0, 0, 255)  # in list order blue, drawn last, covers the crossing
    assert tuple(staged[90, 130, :3]) == (255, 0, 0)
    assert tuple(staged[90, 190, :3]) == (0, 0, 255)


def test_stage_lets_a_model_pass_through_a_card():
    kids = [{"id": "ball", "type": "model", "shape": "sphere", "r": 40, "light": "flat", "fill": "#ff0000"},
            {"id": "card", "type": "rect", "w": 200, "h": 120, "fill": "#0000ff"}]
    spec = lambda **g: scene([dict({"id": "g", "type": "group", "x": 160, "y": 90, "children": kids}, **g)],  # noqa: E731
                             bg="#000000")
    listed, staged = _frame(spec(), 0), _frame(spec(sort="pixel"), 0)
    assert tuple(listed[90, 160, :3]) == (0, 0, 255)
    assert tuple(staged[90, 160, :3]) == (255, 0, 0)  # the ball's front half is in front of the card
    assert tuple(staged[90, 240, :3]) == (0, 0, 255)


def test_stage_ground_shadow_lies_on_a_floor_listed_after_the_model():
    def frame(ground):
        ball = {"id": "ball", "type": "model", "shape": "sphere", "r": 30, "fill": "#ff0000", "ground": ground}
        floor = {"id": "floor", "type": "rect", "w": 300, "h": 300, "y": 40, "rotateX": 90, "fill": "#808080"}
        spec = scene([{"id": "g", "type": "group", "x": 160, "y": 90, "sort": "pixel", "children": [ball, floor]}],
                     bg="#000000")
        spec["camera"] = {"rotateX": 35}
        return _frame(spec, 0).astype(int)
    lit, shaded = frame(False), frame({"y": 40, "opacity": 0.8})
    assert ((lit - shaded)[..., :3].min(2) > 20).sum() > 50  # a floor in the shadow's plane does not hide it


def test_stage_frame_is_pure_function_of_index():
    kids = [{"id": "ball", "type": "model", "shape": "sphere", "r": 30, "detail": "low", "z": "=sin(t*3)*60",
             "ground": {"y": 40}},
            {"id": "card", "type": "rect", "w": 120, "h": 70, "rotateY": "=t*90", "fill": "#3ddbd9"},
            {"id": "floor", "type": "rect", "w": 260, "h": 200, "y": 40, "rotateX": 90, "fill": "#2a3040"}]
    spec = scene([{"id": "g", "type": "group", "x": 160, "y": 90, "sort": "pixel", "children": kids}], duration=1)
    spec["camera"] = {"rotateX": 20, "rotateY": [[0, -10], ["end", 10]]}
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 20)]
    assert np.array_equal(sequential[17], _frame(spec, 17))


def test_stage_without_depth_draws_as_a_group():
    kids = [{"type": "rect", "x": 0, "y": 0, "w": 80, "h": 60, "fill": "#ff0000"},
            {"type": "circle", "x": 30, "y": 10, "r": 30, "fill": "#0000ff"}]
    plain = scene([{"type": "group", "x": 160, "y": 90, "children": kids}])
    staged = scene([{"type": "group", "x": 160, "y": 90, "sort": "pixel", "children": kids}])
    assert np.array_equal(_frame(plain, 0), _frame(staged, 0))


def test_stage_gpu_and_cpu_agree_roughly():
    spec = _crossed(sort="pixel")
    spec["layers"][0]["children"].append({"id": "ball", "type": "model", "shape": "sphere", "r": 30, "z": 10})
    cpu = _frame(spec, 0, "cpu").astype(int)
    try:
        gpu = _frame(spec, 0, "gpu").astype(int)
    except RuntimeError:
        pytest.skip("no GPU")
    assert np.abs(cpu - gpu).mean() < 1.0


SIDE = {"azimuth": -70, "elevation": 40, "follow": "world"}  # a key light from the left


def _hovering(ground, light=SIDE):
    """A ball over a grey background, whose ground shadow lies 15 px below it."""
    ball = {"id": "ball", "type": "model", "shape": "sphere", "r": 30, "x": 160, "y": 70, "fill": "#ff0000"}
    if ground is not None:
        ball["ground"] = ground
    spec = scene([ball], bg="#808080", duration=1)
    spec["light"], spec["camera"] = light, {"rotateX": 30}
    return spec


def test_ground_from_the_light_falls_away_from_it():
    def middle(light):
        bare = _frame(_hovering(None), 0).astype(int)
        dark = (bare - _frame(_hovering({"y": 115, "from": "light", "opacity": 0.8}, light), 0).astype(int))[..., :3]
        ys, xs = np.nonzero(dark.min(2) > 8)
        return xs.mean()
    assert middle(SIDE) > 175 and middle(dict(SIDE, azimuth=70)) < 145  # the ball's middle is at x 160
    assert _frame(_hovering({"y": 115}), 0).tobytes() == _frame(_hovering({"y": 115, "from": "above"}), 0).tobytes()


def _lit_stage(shadows, light=SIDE, **ball):
    """A ball over a floor in a stage, lit from the left."""
    kids = [dict({"id": "ball", "type": "model", "shape": "sphere", "r": 30, "y": -20, "fill": "#ff0000"}, **ball),
            {"id": "floor", "type": "rect", "w": 300, "h": 300, "y": 40, "rotateX": 90, "fill": "#808080"}]
    group = {"id": "g", "type": "group", "x": 160, "y": 90, "sort": "pixel", "children": kids}
    if shadows is not None:
        group["shadows"] = shadows
    spec = scene([group], bg="#000000", duration=1)
    spec["light"], spec["camera"] = light, {"rotateX": 35}
    return spec


def test_stage_shadows_fall_on_the_floor_away_from_the_light():
    bare = _frame(_lit_stage(None), 0).astype(int)
    shaded = _frame(_lit_stage({"opacity": 0.6}), 0).astype(int)
    dark = (bare - shaded)[..., :3].min(2) > 20
    ys, xs = np.nonzero(dark)
    assert len(xs) > 300 and xs.mean() > 175  # on the floor, to the right of the ball (x 160)
    assert np.array_equal(bare[150:, :40], shaded[150:, :40])  # the floor in the light is as it was


def test_a_lone_ball_does_not_shadow_itself():
    def ball(shadows):
        spec = _lit_stage(shadows)
        spec["layers"][0]["children"].pop()  # no floor
        return _frame(spec, 0).astype(int)[40:100, 120:200]
    assert np.abs(ball(None) - ball(True)).mean() < 0.5


def test_stage_shadows_need_a_key_light_and_a_model():
    off = _frame(_lit_stage(None), 0).tobytes()
    assert _frame(_lit_stage(True, light="flat"), 0).tobytes() == _frame(_lit_stage(None, light="flat"), 0).tobytes()
    flat = _lit_stage(True)
    flat["layers"][0]["children"][0] = {"id": "card", "type": "rect", "w": 60, "h": 60, "fill": "#ff0000"}
    plain = _lit_stage(None)
    plain["layers"][0]["children"][0] = flat["layers"][0]["children"][0]
    assert _frame(flat, 0).tobytes() == _frame(plain, 0).tobytes()  # only models cast
    assert off != _frame(_lit_stage(True), 0).tobytes()


def test_gpu_and_cpu_agree_on_shadows():
    for spec in (_lit_stage(True), _hovering({"y": 115, "from": "light"})):
        cpu = _frame(spec, 0, "cpu").astype(int)
        try:
            gpu = _frame(spec, 0, "gpu").astype(int)
        except RuntimeError:
            pytest.skip("no GPU")
        assert np.abs(cpu - gpu).mean() < 1.0


DOT = {"id": "dot", "type": "circle", "x": 60, "y": 40, "r": 10, "fill": "#ff0000"}


def _filmed(camera, *layers, T=0.0):
    """Where the red dot is, at T, seen through camera."""
    spec = scene(list(layers) or [DOT], bg="#000000", duration=2)
    spec["camera"] = camera
    img = _frame(spec, round(T * 30)).astype(int)
    ys, xs = np.nonzero((img[..., 0] > 128) & (img[..., 1] < 100))
    return np.array([xs.mean(), ys.mean()])


def test_camera_target_puts_a_point_or_a_layer_in_the_middle():
    middle = np.array([159.5, 89.5])
    assert np.abs(_filmed({}) - [59.5, 39.5]).max() < 0.6
    assert np.abs(_filmed({"target": [60, 40]}) - middle).max() < 0.6
    assert np.abs(_filmed({"target": ["18.75%", "22.2%", 0]}) - middle).max() < 0.6
    assert np.abs(_filmed({"target": "dot", "rotateY": 30, "rotateX": -20}) - middle).max() < 1.0
    plain = scene([DOT], bg="#000000")
    plain["camera"] = {"rotateY": 20}
    centred = scene([DOT], bg="#000000")
    centred["camera"] = {"rotateY": 20, "target": [160, 90, 0]}
    assert _frame(plain, 0).tobytes() == _frame(centred, 0).tobytes()  # the canvas centre is the default


def test_camera_follows_a_layer_through_its_group():
    moving = dict(DOT, x=[[0, 40], ["end", 280]], z=[[0, -200], ["end", 150]])
    for T in (0, 0.9, 1.9):
        assert np.abs(_filmed({"target": "dot"}, moving, T=T) - [159.5, 89.5]).max() < 1.0
    group = {"id": "g", "type": "group", "x": 100, "y": 20, "rotate": 90, "scale": 1.5, "children": [dict(DOT, x=30, y=0)]}
    assert np.abs(_filmed({"target": "dot"}, group) - [159.5, 89.5]).max() < 1.0
    # A layer that comes in with a motion moves on screen; the camera follows its keys, not the motion.
    rising = dict(DOT, enter={"name": "rise", "dur": 1})
    assert _filmed({"target": "dot"}, rising, T=0.2)[1] > 92


def test_camera_target_keys_pan_from_a_point_to_a_layer():
    spot = _filmed({"target": [[0, [160, 90]], [1, "dot", "linear"]]}, T=0.5)
    assert np.abs(spot - [109.5, 64.5]).max() < 1.0  # halfway: the dot is halfway from where it was to the middle
    assert np.abs(_filmed({"target": [[0, [160, 90]], [1, "dot", "linear"]]}, T=1.5) - [159.5, 89.5]).max() < 0.6


def test_camera_shake_is_smooth_noise_of_the_scene_time():
    still = _filmed({})
    path = np.array([_filmed({"shake": 6}, T=n / 30) for n in range(0, 60, 3)])
    moved = np.abs(path - still).max(1)
    assert moved.max() > 2 and moved.max() < 7  # about amount px at most
    assert np.abs(np.diff(path, axis=0)).max() < 3  # smooth: a tenth of a second moves it little
    other = _filmed({"shake": {"amount": 6, "seed": 3}}, T=0.5)
    assert np.abs(other - path[5]).max() > 0.3
    spec = scene([DOT], bg="#000000", duration=2)
    spec["camera"] = {"shake": [[0, 0], [1, 0], [1.5, 6]]}
    calm = dict(spec, camera={})
    assert _frame(spec, 15).tobytes() == _frame(calm, 15).tobytes()  # amount 0: no shake


def test_check_warns_when_the_camera_is_inside_a_model():
    def codes(model):
        spec = scene([dict(model, x=160, y=90)], duration=1)
        spec["camera"] = {"z": -247}  # at z 0, where the model's middle is (320x180, fov 40)
        return {i["code"] for i in check(spec)["issues"]}
    assert "camera-inside-model" in codes({"id": "ball", "type": "model", "shape": "sphere", "r": 60})
    assert "camera-inside-model" not in codes({"id": "ring", "type": "model", "shape": "torus", "r": 80, "tube": 20})


CARDS = [{"id": "near", "type": "rect", "x": 100, "y": 50, "z": 80, "w": 50, "h": 36, "fill": "#ffffff"},
         {"id": "mid", "type": "rect", "x": 160, "y": 130, "w": 60, "h": 40, "fill": "#ffffff"},
         {"id": "far", "type": "rect", "x": 260, "y": 60, "z": -400, "w": 120, "h": 80, "fill": "#ffffff"}]


def _edges(camera, T=0.0, backend="cpu"):
    """How many px each card's left edge takes to rise from 10% to 90% white, seen through camera (320x180)."""
    spec = scene(CARDS, bg="#000000", duration=2)
    spec["camera"] = camera
    img = _frame(spec, round(T * 30), backend)[..., 0].astype(float)
    out = {}
    for lid, row, x0, x1 in (("near", 30, 0, 80), ("mid", 130, 100, 160), ("far", 78, 160, 200)):
        p = img[row, x0:x1]
        out[lid] = int(np.flatnonzero(p > 229.5)[0] - np.flatnonzero(p > 25.5)[0]) if p.max() > 229.5 else 99
    return out


def test_depth_of_field_blurs_layers_by_their_distance_from_the_focus():
    sharp = _edges({})
    assert max(sharp.values()) <= 1
    soft = _edges({"aperture": 20})
    assert soft["mid"] <= 1 and soft["near"] >= 4 and soft["far"] >= 4  # z 0 is the focus plane by default
    far = _edges({"aperture": 20, "focus": "far"})
    assert far["far"] <= 1 and far["mid"] >= 4 and far["near"] > soft["near"]
    assert _edges({"aperture": 20, "focus": -400}) == far  # the far card's z, as a number
    plain = scene(CARDS, bg="#000000")
    plain["camera"] = {"rotateY": 10}
    off = scene(CARDS, bg="#000000")
    off["camera"] = {"rotateY": 10, "aperture": 0, "focus": "far"}
    assert _frame(plain, 0).tobytes() == _frame(off, 0).tobytes()


def test_focus_keys_pull_focus_from_one_layer_to_another():
    pull = {"aperture": 20, "focus": [[0, "near"], [1, "far", "linear"]]}
    start, end = _edges(pull, 0), _edges(pull, 1.5)
    assert start["near"] <= 1 and start["far"] >= 4
    assert end["far"] <= 1 and end["near"] >= 4


def _focus_stage(camera):
    kids = [{"id": "front", "type": "model", "shape": "cube", "w": 40, "x": -70, "z": 60, "fill": "#ffffff",
             "light": "flat"},
            {"id": "back", "type": "model", "shape": "cube", "w": 70, "x": 70, "z": -300, "fill": "#ffffff",
             "light": "flat"},
            {"id": "card", "type": "rect", "y": 70, "w": 300, "h": 30, "fill": "#808080"}]
    spec = scene([{"id": "g", "type": "group", "x": 160, "y": 90, "sort": "pixel", "children": kids}],
                 bg="#000000", duration=1)
    spec["camera"] = camera
    return spec


def test_a_stage_blurs_each_pixel_by_its_own_depth():
    def edge(img, row, x0, x1):
        p = img[row, x0:x1, 0].astype(float)
        return int(np.flatnonzero(p > 229.5)[0] - np.flatnonzero(p > 25.5)[0]) if p.max() > 229.5 else 99
    flat = _frame(_focus_stage({}), 0)
    back = _frame(_focus_stage({"aperture": 25, "focus": "back"}), 0)
    front = _frame(_focus_stage({"aperture": 25, "focus": "front"}), 0)
    # front's left edge is at x 27 and back's at x 175: each is sharp only while focused
    assert edge(flat, 90, 0, 60) <= 1 and edge(flat, 90, 160, 200) <= 1
    assert edge(back, 90, 160, 200) <= 1 and edge(back, 90, 0, 60) >= 4
    assert edge(front, 90, 0, 60) <= 2 and edge(front, 90, 160, 200) >= 4  # its face is 20 px before its middle
    assert _frame(_focus_stage({"aperture": 0}), 0).tobytes() == flat.tobytes()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_smaller_render_blurs_as_much_of_the_picture(backend):
    """At --scale 0.5 (a draft, or kine look's small frames) a stage is blurred by half as many px: the frame is
    the full one made smaller, not one twice as blurred."""
    spec = _focus_stage({"aperture": 25, "focus": "back"})
    try:
        full = Renderer(k.Scene(spec), backend=backend).frame(0)[..., :3].astype(float)
    except RuntimeError:
        pytest.skip("no GPU")
    half = Renderer(k.Scene(spec), backend=backend, scale=0.5).frame(0)[..., :3].astype(float)
    small = full.reshape(90, 2, 160, 2, 3).mean((1, 3))
    rows = slice(30, 60)  # across both cubes, the front one blurred
    assert np.abs(half[rows] - small[rows]).mean() < 3  # 20 when the blur was not made smaller


def test_gpu_and_cpu_agree_on_depth_of_field():
    for spec in (_focus_stage({"aperture": 25, "focus": "front"}), dict(scene(CARDS, bg="#000000"),
                                                                       camera={"aperture": 20})):
        cpu = _frame(spec, 0, "cpu").astype(int)
        try:
            gpu = _frame(spec, 0, "gpu").astype(int)
        except RuntimeError:
            pytest.skip("no GPU")
        assert np.abs(cpu - gpu).mean() < 1.0


def test_check_warns_about_slow_effects_on_the_cpu(monkeypatch):
    import kinescript.verify as V
    soft = scene([{"id": "soft", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40, "blur": 4}], duration=1)
    glowing = scene([{"id": "g", "type": "text", "text": "Hi", "x": 160, "y": 90, "size": 40, "glow": 12}], duration=1)
    plain = scene([{"id": "p", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40}], duration=1)
    monkeypatch.setattr(V, "SLOW_EFFECTS", 0.0005)  # 320x180 frames are too small to be slow
    try:
        on_gpu = {i["code"] for i in check(soft)["issues"]}
    except RuntimeError:
        on_gpu = set()
    assert "slow-effects" not in on_gpu or _GL.get("ctx") is None
    monkeypatch.setitem(_GL, "ctx", None)  # as on a machine with no GPU
    warned = [i for i in check(soft)["issues"] if i["code"] == "slow-effects"]
    assert warned and "0.44 s for a 1080p layer" in warned[0]["message"], warned
    assert any(i["code"] == "slow-effects" for i in check(glowing)["issues"])
    assert not any(i["code"] == "slow-effects" for i in check(plain)["issues"])


def test_effects_are_counted_by_the_pixels_they_blur():
    def filtered(layer):
        r = Renderer(k.Scene(scene([dict({"id": "a", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40}, **layer)],
                                   duration=1)), backend="cpu")
        r.draw(0)
        return r.filtered
    assert filtered({}) == 0 and filtered({"blur": 4}) == 320 * 180  # blur's layer is the frame
    small = filtered({"glow": 8})  # glow's is the rect and 1.5 times its size round it
    assert (40 + 24) ** 2 <= small <= (42 + 26) ** 2, small
    assert filtered({"outline": 3}) < filtered({"outline": 9})


def test_check_warns_about_focus_on_nothing_and_slow_depth_of_field(monkeypatch):
    def codes(camera):
        return {i["code"] for i in check(_focus_stage(camera))["issues"]}
    assert "focus-on-nothing" not in codes({"aperture": 25, "focus": "back"})
    assert "focus-on-nothing" in codes({"aperture": 25, "focus": -5000})
    assert "focus-on-nothing" in codes({"aperture": 25, "focus": 240})  # 7 px from the camera
    monkeypatch.setitem(_GL, "ctx", None)  # as on a machine with no GPU
    assert "slow-depth-of-field" in codes({"aperture": 25})
    assert "slow-depth-of-field" not in codes({})


def test_check_warns_about_a_stage_with_many_children():
    kids = [{"type": "rect", "x": 10 * i, "y": 0, "w": 8, "h": 8, "z": i} for i in range(17)]
    spec = scene([{"id": "g", "type": "group", "x": 80, "y": 90, "sort": "pixel", "children": kids}], duration=0.5)
    assert any(i["code"] == "stage-cost" and i["layer"] == "g" for i in check(spec)["issues"])
    kids.pop()
    assert not any(i["code"] == "stage-cost" for i in check(spec)["issues"])


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


def test_check_places_clips_without_decoding_them(clip, monkeypatch):
    # check's samples read boxes, not pixels. A clip decoded at each 0.25 s sample reopened ffmpeg nearly every time
    # (60 cuts: 134 decoders, 17 s); now only the text checks, which read pixels, decode it.
    clips = [{"src": clip, "trim": [0.3 * n, 0.3 * n + 0.7]} for n in range(4)]
    layers = [{"id": "v", "type": "video", "src": clip, "x": 40, "y": 20, "w": 200, "h": 120, "radius": 8,
               "stroke": "#ff0000", "strokeWidth": 12, "rotate": [[0, 0], [2, 30]], "zoom": [[0, 1], [2, 1.4]]},
              {"id": "cut", "type": "sequence", "x": 100, "y": 60, "w": 160, "h": 90, "clips": clips,
               "transition": {"type": "slide", "dur": 0.2}}]
    s = k.Scene(scene(layers, duration=2))
    try:
        decoded, placed = Renderer(s, track=True), Renderer(s, track=True)
        placed.footage = False
        for T in (0.0, 0.5, 0.61, 1.37, 1.9):  # 0.61: two clips of the sequence in their slide
            decoded.draw(T)
            placed.draw(T)
            assert placed.boxes == decoded.boxes, T
    finally:
        s.close()
    opened = []
    real = media.VideoSource._open
    monkeypatch.setattr(media.VideoSource, "_open", lambda self, t: opened.append(t) or real(self, t))
    check(scene(layers, duration=2))
    probe(scene(layers, duration=2), at=1.0)
    assert opened == []
    # Text over a clip is measured against the clip's picture: grey reads on the black canvas, not on the test card.
    grey = {"id": "t", "type": "text", "text": "Grey", "x": 160, "y": 90, "size": 40, "fill": "#808080"}
    assert not [i for i in check(scene([grey], bg="#000000", duration=2))["issues"] if i["code"] == "contrast"]
    card = {"id": "v", "type": "video", "src": clip, "w": 320, "h": 180, "dur": 2}
    hit = [i for i in check(scene([card, grey], bg="#000000", duration=2))["issues"] if i["code"] == "contrast"]
    assert [i["layer"] for i in hit] == ["t"] and opened
    # A decoder that cannot start there is a render error, as it was in the samples, not a crash.
    monkeypatch.setattr(media, "FFMPEG", "kine-test-no-ffmpeg")
    res = check(scene([card, grey], bg="#000000", duration=2))
    assert [i["code"] for i in res["issues"] if i["level"] == "error"] == ["render"] and not res["ok"]


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


def test_look_draws_trails_and_onion_skins(tmp_path):
    import skia
    from kinescript.verify import contact_sheet
    spec = scene([{"id": "dot", "type": "circle", "r": 6, "y": 90, "x": [[0, 40], [1, 280, "linear"]], "fill": "#ffffff"},
                  {"id": "still", "type": "rect", "x": 160, "y": 40, "w": 20, "h": 20, "fill": "#4a7dff"}],
                 duration=1, bg="#000000")

    def sheet(name, **kw):
        info = contact_sheet(spec, str(tmp_path / name), times=[0.5], width=320, **kw)
        return info, skia.Image.open(info["out"]).toarray(colorType=skia.kRGBA_8888_ColorType)
    _, plain = sheet("p.png")
    info, trail = sheet("t.png", trails=True)
    assert info["trails"] == ["dot"]  # the rect stands still
    y = 34 + 90  # the tile starts 8 px in and 34 px down; the dot moves 8 px a frame, at x 160 in frame 15
    assert plain[y, 8 + 240, :3].max() == 0 and trail[y, 8 + 240, 0] > 150 > trail[y, 8 + 240, 1]  # frame 25's dot
    _, onion = sheet("o.png", onion=2)
    assert 60 < onion[y, 8 + 152, 0] < 120 and 25 < onion[y, 8 + 144, 0] < 65  # frames 14 and 13, fainter
    still = onion[34 + 40, 8 + 160].astype(int) - plain[34 + 40, 8 + 160]  # what holds still looks the same:
    assert np.abs(still).max() <= 1  # the same on the GPU, a level off at most where the CPU blends it
    assert contact_sheet(spec, str(tmp_path / "n.png"), count=1, width=320, trails=True)["trails"] == ["dot"]
    assert probe(spec, at=0.5)["visible"]["dot"]["pos"] == [160.0, 90.0]
def test_probe_curve_samples_a_key_and_its_speed():
    from kinescript.mcp import call
    from kinescript.verify import curve
    spec = scene([{"id": "a", "type": "rect", "w": 10, "h": 10, "y": 90,
                   "x": [[0, 0], [1, 100, "linear"], [1.5, 100], [2, 40, "linear"]],
                   "fill": [[0, "#000000"], [1, "#ffffff"]]},
                  {"id": "g", "type": "group", "speed": 2,
                   "children": [{"id": "kid", "type": "rect", "w": 10, "h": 10, "x": [[0, 0], [1, 100, "linear"]]}]}],
                 duration=2)
    c = curve(spec, "a.x")
    assert c["frames"] == 60 and c["value"][15] == 50 and c["speed"][15] == 100 and c["speed"][0] is None
    assert c["stops"] == [{"from": 1.0, "to": 1.5, "frames": [30, 45], "value": 100}]
    assert round(c["fastest"]["speed"]) == -120 and c["fastest"]["frame"] > 45 and c["range"] == [0, 100]
    kid = curve(spec, "kid.x")  # its group's clock runs twice as fast, and it ends at the group's second second
    assert kid["value"][15] == 100 and kid["speed"][5] == 200 and kid["value"][30] is None
    fill = curve(spec, "a.color")
    assert fill["curve"] == "a.fill" and fill["kind"] == "colour" and fill["value"][0] == "#000000"
    assert fill["value"][30] == "#ffffff" and fill["speed"][15] > 0 and fill["stops"][0]["from"] == 1.0
    assert curve(spec, "a.position")["value"][15] == [50, 90] and curve(spec, "a.opacity")["fastest"] is None
    for bad, says in (("b.x", "no layer 'b'"), ("a.wobble", "has no wobble"), ("a", "such as title.x")):
        with pytest.raises(ValueError, match=says):
            curve(spec, bad)
    assert json.loads(call("kine_probe", {"scene": spec, "curve": "a.x"})[0]["text"])["speed"][15] == 100


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


def _packets(path) -> int:
    out = subprocess.run([media.FFPROBE, "-v", "error", "-select_streams", "v", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def _rgb(path, w, h, pix="rgb24") -> np.ndarray:
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", pix, "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, -1)


def test_a_clip_opened_mid_way_shows_the_frame_playing_through_shows(clip):
    # 41/30 s was sent to ffmpeg as 1.3667, past frame 41's own time, so a clip opened there (a still, a contact
    # sheet, the first frame of a render worker) started on frame 42.
    play = media.VideoSource(clip, 64, 36, 30)
    frames = [play.at(n / 30).toarray().copy() for n in range(45)]
    play.close()
    for n in (41, 22, 44):
        cold = media.VideoSource(clip, 64, 36, 30)
        assert np.array_equal(cold.at(n / 30).toarray(), frames[n]), n
        cold.close()


def test_a_render_with_sound_keeps_its_last_frame(clip, tmp_path):
    from kinescript import output
    # 0.49 s at 30 fps is 15 frames (0.5 s) over 0.49 s of sound; the mux used to end with the sound.
    spec = scene([{"type": "video", "src": clip, "x": 0, "y": 0, "anchor": "topleft", "w": 64, "h": 36}],
                 w=64, h=36, duration=0.49)
    res = output.render(spec, str(tmp_path / "a.mp4"), quality="draft", workers=1)
    assert res["frames"] == 15 == _packets(tmp_path / "a.mp4") and res["audio"]


def test_layer_ends_fall_on_the_frame_grid():
    rect = {"type": "rect", "w": 10, "h": 10}
    s = k.Scene(scene([dict(rect, id="a", at=2.2, dur=8.9), dict(rect, id="b", at="a.out", dur=1)], duration=13))
    a, b = s.by_id["a"], s.by_id["b"]  # 2.2 + 8.9 is 11.100000000000001: a also showed on frame 333, b started on 334
    assert a.visible(332 / 30) and not a.visible(333 / 30) and b.visible(333 / 30)


def test_aurora_draws_the_same_on_the_cpu_as_its_formula():
    # pow() of a negative base is undefined; Skia's raster backend made it NaN and the bands came out wrong.
    spec = scene([{"type": "shader", "preset": "aurora", "x": 0, "y": 0, "anchor": "topleft", "w": "100%", "h": "100%"}],
                 w=64, h=64, bg="#000000")
    got = _frame(spec, 0)[:, 20, :3].astype(float) / 255
    uv = (np.arange(64) + 0.5) / 64, (20 + 0.5) / 64
    band = sum(np.exp(-((uv[0] - (0.35 + 0.12 * i + 0.08 * np.sin(uv[1] * (3 + i)))) * (9 - 2 * i)) ** 2) * (0.7 - 0.15 * i)
               for i in range(3))
    base, glow, glow2 = (np.array(k.color.parse(c)[:3]) for c in ("#0a0c10", "#b9f45d", "#ff633f"))
    g = glow + (glow2 - glow) * (0.5 + 0.5 * np.sin(uv[1] * 2))
    want = base + (g - base) * np.clip(band * 0.75, 0, 0.85)[:, None]
    assert np.abs(got - want).max() < 3 / 255


def test_a_transparent_canvas_in_an_mp4_is_shown_over_black(tmp_path):
    from kinescript import output
    half = {"type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": "100%", "h": "100%", "fill": "#ff0000", "opacity": 0.5}
    spec = scene([half], w=64, h=36, bg="transparent", duration=0.2)
    output.render(spec, str(tmp_path / "a.mp4"), quality="high", workers=1)
    assert 120 <= _rgb(tmp_path / "a.mp4", 64, 36)[18, 32, 0] <= 136  # was 254: straight colour with alpha dropped
    with pytest.raises(ValueError, match="keeps no alpha"):
        output.render(spec, str(tmp_path / "b.mp4"), transparent=True)


def test_a_half_transparent_background_is_straight_alpha_in_a_mov(tmp_path):
    from kinescript import output
    spec = scene([{"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2, "fill": "#00ff00"}], w=64, h=36, bg="#ff000080",
                 duration=0.2)
    output.render(spec, str(tmp_path / "a.mov"), quality="draft", workers=1)
    r, g, b, a = (int(v) for v in _rgb(tmp_path / "a.mov", 64, 36, "rgba")[18, 32])
    assert r >= 250 and 120 <= a <= 136, (r, g, b, a)  # was (128, 0, 0, 129): premultiplied read as straight


@pytest.mark.parametrize("w,h", [(1, 1), (300, 7), (37, 300)])
def test_a_png_decodes_to_exactly_the_pixels_written(tmp_path, monkeypatch, w, h):
    import skia
    from kinescript import output
    monkeypatch.setattr(output, "PNG_BAND", 1000)  # many bands: one row each at 300 wide, 6 rows at 37
    px = np.random.default_rng(w * h).integers(0, 256, (h, w, 4), np.uint8)  # alpha 0 to 255, any colour
    p = tmp_path / "a.png"
    output.save_png(px, p)
    data = p.read_bytes()
    assert data[24:26] == bytes([8, 6])  # 8-bit RGBA, as Skia wrote them
    codec = skia.Codec.MakeFromData(data)
    got = np.empty_like(px)
    info = skia.ImageInfo.Make(w, h, skia.kRGBA_8888_ColorType, skia.kUnpremul_AlphaType)
    assert codec.getPixels(info, got, w * 4) == skia.Codec.kSuccess
    assert np.array_equal(got, px)
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                         capture_output=True, check=True).stdout  # a second decoder, ffmpeg's own
    assert np.array_equal(np.frombuffer(raw, np.uint8).reshape(h, w, 4), px)


def test_an_opaque_scenes_png_keeps_straight_colour_where_a_pixel_is_not_opaque():
    from types import SimpleNamespace
    from kinescript import output
    px = np.array([[[64, 32, 0, 128], [9, 8, 7, 255]]], np.uint8)  # read premultiplied, as an opaque scene reads
    got = output._straight(SimpleNamespace(transparent=False), px)
    assert got.tolist() == [[[128, 64, 0, 128], [9, 8, 7, 255]]]  # as Skia's encoder wrote it


def test_3d_text_frames_do_not_depend_on_the_frames_drawn_before():
    # Early in a slow eased tween, frame 3's size rounds to frame 0's cache key; the outline was built from
    # whichever exact size was drawn first, so a worker starting at frame 3 drew different pixels.
    spec = scene([{"id": "m", "type": "model", "shape": "text", "text": "AB", "x": 120, "y": 90,
                   "size": [[0, 60], [4, 110]], "depth": [[0, 10], [4, 50]], "bevel": [[0, 0], [4, 6]],
                   "rotateY": "=T*40", "fill": "#ff633f", "shine": [[0, 0.1], [4, 0.9]]}],
                 w=480, h=270, bg="#101418", duration=4)
    after = Renderer(k.Scene(spec), scale=0.5, backend="cpu")
    after.frame(0)
    alone = Renderer(k.Scene(spec), scale=0.5, backend="cpu")
    assert np.array_equal(after.frame(3), alone.frame(3))


def test_motion_blur_keeps_the_first_frame_inside_the_scene():
    white = {"type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": "100%", "h": "100%", "fill": "#ffffff"}
    spec = dict(scene([white], w=64, h=36, bg="#000000", duration=1), motionBlur={"samples": 8, "shutter": 0.5})
    assert tuple(_frame(spec, 0)[18, 32, :3]) == (255, 255, 255)  # was 127: half its samples were before 0 s


def test_hidden_clips_are_silent_and_looped_clips_loop_their_trim(clip):
    from kinescript import output
    vid = {"type": "video", "src": clip, "x": 0, "y": 0, "anchor": "topleft", "w": 64, "h": 36}
    s = k.Scene(scene([dict(vid, id="h", hidden=True), {"type": "group", "id": "g", "hidden": True,
                                                         "children": [dict(vid, id="in")]},
                       dict(vid, id="l", trim=[0.5, 1.5], loop=True, dur=3)], w=64, h=36, duration=3))
    ins = output._audio_inputs(s)
    s.close()
    assert [a["id"] for a in ins] == ["l"] and ins[0]["loop"] == pytest.approx(1.0)
    spec = scene([dict(vid, id="l", trim=[0.5, 1.5], loop=True, dur=3)], w=64, h=36, duration=3)
    r = Renderer(k.Scene(spec), backend="cpu")
    assert np.array_equal(r.frame(15).copy(), r.frame(45))  # 1.5 s in is 1.0 s of the file again; it played on to 2.0


@pytest.mark.parametrize("bad, code", [({"trim": [-1, 2]}, "bad-trim"), ({"trim": [None, 2]}, "bad-trim"),
                                       ({"trim": 3}, "bad-trim"), ({"speed": 0}, "bad-value")])
def test_bad_trims_and_speeds_are_scene_errors(clip, bad, code):
    vid = dict({"id": "v", "type": "video", "src": clip, "w": 64, "h": 36}, **bad)
    with pytest.raises(k.SceneError) as e:
        k.Scene(scene([vid], w=64, h=36))
    assert code in [i["code"] for i in e.value.issues]


def test_media_tags_and_temp_paths_need_not_be_ascii(clip, tmp_path, monkeypatch):
    from kinescript import output
    tagged = tmp_path / "tagged.mp4"
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-i", clip, "-t", "0.5", "-c", "copy", "-metadata",
                    "title=ссс видео 日本", str(tagged)], check=True)
    assert media.probe(str(tagged))["duration"] > 0  # cp1252 could not decode the tag on Windows
    odd = tmp_path / "o'brien"
    odd.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(odd))  # a quote in the concat list's paths broke the join
    spec = scene([{"type": "rect", "x": "=t*100", "y": 10, "w": 10, "h": 10}], w=64, h=36, duration=0.4)
    res = output.render(spec, str(tmp_path / "j.mp4"), quality="draft", workers=2)
    assert res["workers"] == 2 and _packets(tmp_path / "j.mp4") == 12
