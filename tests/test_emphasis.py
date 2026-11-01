"""emphasis: motions in the middle of a layer's life (pulse, heartbeat, shake, wiggle, wobble, float, bob, flash, tada,
spin), at a time, for a number of cycles or in a loop, combined with enter and exit, and across a split text as a
wave."""
import math

import numpy as np
import pytest

import kinescript as k
from kinescript import presets as P
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 400, "h": 300, "fps": 20, "duration": 3, "bg": "#000000"}, **canvas),
            "layers": layers}


def card(emphasis=None, **kw):
    """A white card 120 wide and 80 tall about (200, 150): x 140 to 259, y 110 to 189 when at rest."""
    return dict({"id": "c", "type": "rect", "x": 200, "y": 150, "w": 120, "h": 80, "fill": "#ffffff",
                 "emphasis": emphasis}, **kw)


def frames(spec, n=(0,), backend="cpu"):
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    return [r.frame(i)[:, :, :3].astype(int).copy() for i in n]


def lit(img, level=128):
    ys, xs = np.nonzero(img.max(axis=2) > level)
    return (ys.min(), ys.max(), xs.min(), xs.max()) if len(ys) else None


def codes(spec):
    return [(i["level"], i["code"]) for i in check(spec)["issues"]]


REST = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)


def state(m):
    return m.alpha, m.dx, m.dy, m.scale, (m.rotate + 180) % 360 - 180, m.sx, m.sy  # a whole turn is at rest


@pytest.mark.parametrize("name", list(P.EMPHASES))
def test_each_cycle_starts_and_ends_at_rest(name):
    f = P.EMPHASES[name]
    assert state(f(0.0, {})) == pytest.approx(REST, abs=1e-9)
    assert state(f(1.0, {})) == pytest.approx(REST, abs=1e-9)
    assert any(state(f(u, {})) != pytest.approx(REST, abs=1e-3) for u in np.linspace(0.05, 0.95, 19))


def test_the_shapes_of_the_cycles():
    E = P.EMPHASES
    assert E["pulse"](0.5, {}).scale == pytest.approx(1.12) and E["pulse"](0.5, {"amount": 0.3}).scale == 1.3
    assert E["pulse"](0.05, {}).scale == pytest.approx(1 + 0.12 * math.sin(math.pi * 0.05) ** 2)  # still at first
    beats = [E["heartbeat"](u, {}).scale for u in np.linspace(0, 1, 401)]
    peaks = [b for a, b, c in zip(beats, beats[1:], beats[2:]) if a < b >= c]
    assert peaks == pytest.approx([1.15, 1.09], abs=1e-3) and beats[300] == 1  # two beats, then rest
    assert max(abs(E["shake"](u, {}).dx) for u in np.linspace(0, 1, 401)) == pytest.approx(11.26, abs=0.05)
    assert max(abs(E["wiggle"](u, {"deg": 20}).rotate) for u in np.linspace(0, 1, 401)) > 15
    w = E["wobble"](0.12, {})
    assert w.sx > 1.1 and w.sx + w.sy == pytest.approx(2)  # wide as it is short
    assert E["float"](0.5, {}).dy == pytest.approx(-10) and E["float"](0.25, {}).rotate == pytest.approx(2)
    assert E["bob"](0.5, {"dist": 30}).dy == pytest.approx(-30)
    assert [E["flash"](u, {}).alpha for u in (0.25, 0.5, 0.75)] == pytest.approx([0, 1, 0], abs=1e-9)
    assert E["flash"](0.25, {"amount": 0.5}).alpha == pytest.approx(0.5)
    assert E["tada"](0.1, {}).scale == pytest.approx(0.9) and E["tada"](0.6, {}).scale == pytest.approx(1.1)
    assert E["tada"](0.1, {}).rotate == pytest.approx(-4) and abs(E["tada"](0.3, {}).rotate) > 0.5  # rocks as it swells
    assert E["spin"](0.5, {}).rotate == 180 and E["spin"](0.5, {"deg": -90}).rotate == -45


def test_an_emphasis_plays_from_at_for_its_cycles():
    e = P.parse_emphasis({"name": "pulse", "at": 1, "repeat": 2})[0]
    assert (e.at, e.dur, e.repeat, e.end) == (1, 0.5, 2, 2)
    assert e.mods(0.9) is P.IDENT and e.mods(2.1) is P.IDENT and e.mods(1.0) is P.IDENT
    assert e.mods(1.25).scale == pytest.approx(1.12) and e.mods(1.75).scale == pytest.approx(1.12)
    half = P.parse_emphasis({"name": "pulse", "repeat": 1.5})[0]
    assert half.mods(0.6).scale > 1 and half.mods(0.76) is P.IDENT  # half a cycle more, then it stops
    loop = P.parse_emphasis({"name": "float", "loop": True})[0]
    assert loop.repeat == math.inf and loop.mods(1000.5).dy == pytest.approx(-10)
    spin = P.parse_emphasis([{"name": "spin"}, {"name": "spin", "loop": True}, {"name": "spin", "ease": "linear"}])
    assert spin[0].mods(0.25).rotate == pytest.approx(360 * 0.0625)  # inOutCubic, slow to start
    assert spin[1].mods(0.25).rotate == pytest.approx(90) and spin[2].mods(0.25).rotate == pytest.approx(90)
    both = P.parse_emphasis("pulse+wiggle")[0]
    assert both.names == ["pulse", "wiggle"] and both.dur == 0.6  # the longer of their cycles
    assert [P.parse_emphasis(n)[0].dur for n in ("heartbeat", "float", "tada")] == [0.9, 3.0, 1.0]
    assert P.parse_emphasis(None) == [] and P.parse_emphasis([]) == []


def test_names_other_tools_use():
    assert [P.canonical_emphasis(n) for n in ("headShake", "heart-beat", "rubberBand", "bounce", "blink", "jello",
                                              "Hover", "nope")] == [
        "shake", "heartbeat", "wobble", "bob", "flash", "wobble", "float", None]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_pulse_grows_the_layer_and_lets_it_go(backend):
    rest, peak, after = frames(scene([card({"name": "pulse", "at": 1, "amount": 0.25})]), (10, 25, 40), backend)
    assert lit(rest) == lit(after) == (110, 189, 140, 259)
    top, bottom, left, right = lit(peak)
    assert right - left == pytest.approx(150, abs=2) and bottom - top == pytest.approx(100, abs=2)


def test_emphasis_combines_with_enter_exit_and_itself():
    spec = scene([card([{"name": "pulse", "amount": 0.5, "dur": 1}, {"name": "bob", "dist": 30, "dur": 1}],
                       enter={"name": "fade", "dur": 1, "ease": "linear"})])
    (img,) = frames(spec, (10,))
    top, bottom, left, right = lit(img, 40)
    assert right - left == pytest.approx(180, abs=3) and (top + bottom) / 2 == pytest.approx(120, abs=2)
    assert img[150, 200].max() == pytest.approx(128, abs=4)  # half faded in
    out = scene([card({"name": "flash", "at": 2.5, "dur": 0.8}, exit={"name": "fade", "dur": 1})])
    assert frames(out, (54,))[0][150, 200].max() == 0  # flashed out a quarter of the way through the cycle


def test_a_staggered_emphasis_runs_across_a_split_text_as_a_wave():
    text = {"id": "t", "type": "text", "text": "HHHH", "x": 200, "y": 150, "size": 60, "fill": "#ffffff",
            "split": "chars", "emphasis": {"name": "bob", "dist": 30, "dur": 1, "stagger": 0.25}}
    (img,) = frames(scene([text]), (5,))  # 0.25 s: the first H a quarter up, the last not yet off
    cols = img.max(axis=2) > 128
    xs = np.nonzero(cols.any(axis=0))[0]
    tops = [np.nonzero(cols[:, x])[0].min() for x in (xs.min() + 4, xs.max() - 4)]
    still = lit(frames(scene([dict(text, emphasis=None)]))[0])[0]
    assert tops[1] == still and still - tops[0] == pytest.approx(30 * math.sin(math.pi / 4), abs=2)
    plain = frames(scene([dict(text, emphasis={"name": "bob", "dist": 30, "dur": 1})]), (5,))[0]
    assert still - lit(plain)[0] == pytest.approx(21, abs=2)  # without a stagger, one block
    back = frames(scene([dict(text, emphasis={"name": "bob", "dist": 30, "dur": 1, "stagger": 0.25, "order": "end"})]),
                  (5,))[0]
    assert np.array_equal(back, img[:, ::-1]) or lit(back[:, :200])[0] == still  # the last H goes first
    e = P.parse_emphasis({"name": "bob", "stagger": 0.1, "order": "random", "seed": 3})[0]
    assert (e.order, e.seed, e.stagger) == ("random", 3, 0.1)


def test_emphasis_holds_back_the_picture_of_a_still_group_until_it_ends():
    spec = scene([{"id": "g", "type": "group", "children": [
        {"id": "a", "type": "rect", "x": 100, "y": 150, "w": 60, "h": 60},
        card({"name": "wiggle", "at": 0.5, "repeat": 2})]}])
    a, b = Renderer(k.Scene(spec), backend="cpu"), Renderer(k.Scene(spec), backend="cpu")
    b._still = lambda *x: None
    for i in range(0, 60, 3):
        assert np.array_equal(a.frame(i), b.frame(i)), i
    assert a._rests["g"][0] == pytest.approx(1.7)  # 0.5 s in, two cycles of 0.6 s
    looped = scene([{"id": "g", "type": "group", "children": [card({"name": "float", "loop": True})]}])
    r = Renderer(k.Scene(looped), backend="cpu")
    r.frame(30)
    assert r._rests["g"][0] == math.inf


def test_emphasis_is_a_function_of_the_frame_alone():
    spec = scene([card([{"name": "shake", "loop": True}, {"name": "tada", "at": 0.4}]),
                  {"id": "t", "type": "text", "text": "wave", "x": 200, "y": 250, "size": 40, "split": "chars",
                   "emphasis": {"name": "float", "loop": True, "dur": 1, "stagger": 0.1, "order": "random"}}])
    a, b, c = frames(spec, (13, 4, 13))
    (d,) = frames(spec, (13,))
    assert np.array_equal(a, c) and np.array_equal(a, d) and not np.array_equal(a, b)


def test_check_reads_emphasis():
    assert codes(scene([card({"name": "pulse", "at": 1, "repeat": 2})])) == []
    assert ("error", "bad-motion") in codes(scene([card({"name": "pulze"})]))
    msg = [i["message"] for i in check(scene([card("pulze")]))["issues"]][0]
    assert "pulse" in msg
    for bad in ({"dur": 0}, {"at": -1}, {"repeat": 0}, {"loop": "yes"}, {"amount": "big"}):
        assert ("error", "bad-value") in codes(scene([card(dict({"name": "pulse"}, **bad))])), bad
    assert ("error", "bad-ease") in codes(scene([card({"name": "pulse", "ease": "outWobbly"})]))
    assert ("error", "unknown-key") in codes(scene([card({"name": "pulse", "bounces": 2})]))
    assert ("info", "renamed-key") in codes(scene([card({"name": "pulse", "times": 3})]))
    bare = {k: v for k, v in card().items() if k != "emphasis"}
    assert ("info", "renamed-key") in codes(scene([dict(bare, attention="pulse")]))
    assert ("warning", "late-emphasis") in codes(scene([card({"name": "pulse", "at": 2}, dur=1.5)]))
    assert ("warning", "late-emphasis") in codes(scene([card([{"name": "pulse"}, {"name": "tada", "at": 9}])]))
    assert ("warning", "ignored-key") in codes(scene([card({"name": "pulse", "loop": True, "repeat": 3})]))
    assert ("warning", "ignored-key") in codes(scene([card({"name": "pulse", "stagger": 0.1})]))
    with pytest.raises(k.SceneError, match="emphasis: each is a name like"):
        k.Scene(scene([card(3)]))
    for bad, why in ((3, "emphasis must be a name"), (["pulse", "pulze"], "emphasis.1.: unknown emphasis 'pulze'"),
                     ({"name": "bob", "dur": -1}, "emphasis: dur is -1")):
        with pytest.raises(ValueError, match=why):
            P.parse_emphasis(bad)


def test_components_pass_emphasis_on():
    spec = scene([{"use": "lowerThird", "id": "lt", "title": "Name", "emphasis": {"name": "pulse", "at": 1}}])
    s = k.Scene(spec)
    assert [e.names for e in s.by_id["lt"].emphasis] == [["pulse"]] and "emphasis" not in s.by_id["lt"].props


def test_check_reads_a_turned_text_at_its_size():
    spec = scene([{"id": "t", "type": "text", "text": "turned", "x": 200, "y": 150, "size": 40, "rotate": 89}],
                 w=1920, h=1080)
    assert ("warning", "tiny-text") not in codes(spec)  # a turn empties the matrix's diagonal, not its scale
    spec["layers"][0]["scale"] = 0.3
    assert ("warning", "tiny-text") in codes(spec)
