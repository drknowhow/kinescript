"""The enter and exit motions of 1.6.1: land and squash bounce in, elastic, swing and stretch spring in about a pivot,
iris, clock, blinds and rise-mask reveal, scramble settles text, glitch splits and slices, whip smears across."""
import numpy as np
import pytest

import kinescript as k
from kinescript import presets as P
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 400, "h": 300, "fps": 20, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def card(enter=None, **kw):
    """A white card 120 wide and 80 tall about (200, 150): x 140 to 260, y 110 to 190."""
    return dict({"id": "c", "type": "rect", "x": 200, "y": 150, "w": 120, "h": 80, "fill": "#ffffff", "enter": enter},
                **kw)


def words(enter, **kw):
    return dict({"id": "t", "type": "text", "text": "HELLO world 42", "x": 200, "y": 150, "size": 40,
                 "fill": "#ffffff", "enter": enter}, **kw)


def frames(spec, n=(0,), backend="cpu"):
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    return [r.frame(i)[:, :, :3].astype(int).copy() for i in n]


def lit(img, level=128):
    """The rows and columns where img is lit, as (top, bottom, left, right), or None."""
    ys, xs = np.nonzero(img.max(axis=2) > level)
    return (ys.min(), ys.max(), xs.min(), xs.max()) if len(ys) else None


def test_fall_drops_under_gravity_and_bounces_a_quarter_as_high_each_time():
    assert P.fall(0.0, 3) == (1.0, 0.0, 0.0) and P.fall(1.0, 3)[0] == 0.0
    ps = np.linspace(0, 1, 5501)
    h = np.array([P.fall(p, 3)[0] for p in ps])
    tops = [h[i] for i in range(1, len(h) - 1) if h[i - 1] < h[i] >= h[i + 1]]
    assert tops == pytest.approx([0.25, 0.0625, 0.015625], abs=1e-3)
    assert P.fall(1 / 2.75, 3) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)  # lands at full speed, as outBounce does
    assert P.fall(0.5, 0)[0] == pytest.approx(0.75)  # no bounces: one fall
    # Held on the ground at each landing, squashed as hard as it lands; the last one ends the motion.
    q = [P.fall(p, 2, 0.35)[2] for p in ps]
    assert max(q) == pytest.approx(1.0, abs=1e-3) and P.fall(1.0, 2, 0.35)[2] == pytest.approx(0.0, abs=1e-9)
    later = [s for p, s in zip(ps, q) if p > 0.6]  # the second landing's, at half the speed
    assert max(later) == pytest.approx(0.5, abs=0.01)
    assert all(P.fall(p, 2, 0.35)[0] == 0 for p, s in zip(ps, q) if s > 0)


def test_land_and_squash_drop_from_dist_and_end_in_place():
    land = P.parse("land")
    assert land.dur == 1.0 and land.mods(0).dy == -160 and land.mods(1).dy == 0 and land.mods(1).alpha == 1
    assert P.parse({"name": "land", "dist": 300}).mods(0).dy == -300
    sq = P.parse({"name": "squash", "amount": 0.5})
    m = [sq.mods(p) for p in np.linspace(0, 1, 401)]
    assert all(x.pivot == "bottom" and x.sx * x.sy == pytest.approx(1) for x in m)
    assert max(x.sy for x in m) > 1.2 and min(x.sy for x in m) == pytest.approx(0.5, abs=0.01)
    assert m[-1].sy == pytest.approx(1) and m[-1].dy == 0


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_squash_flattens_on_the_ground_and_swing_turns_from_its_top(backend):
    spec = scene([card({"name": "squash", "bounces": 0, "dur": 1})])
    land = int(round(1 / 1.35 * 20))  # the fall takes 1 of its 1.35 parts; then it squashes on the ground
    rest, flat = frames(spec, (39, land + 2), backend)
    assert lit(rest) == (110, 189, 140, 259)
    top, bottom, left, right = lit(flat)
    assert bottom == 189 and top > 125 and left < 130 and right > 270  # its bottom stays on the ground
    spec = scene([card({"name": "swing", "dur": 1.2})])
    turned, still = frames(spec, (3, 39), backend)
    assert lit(still) == (110, 189, 140, 259)
    top, bottom, left, right = lit(turned)
    assert turned[112, 200].max() > 128 and (left < 130 or right > 270)  # hung from the middle of its top edge
    swing = P.parse("swing")
    assert swing.mods(1).hinge == 0 and P.parse({"name": "swing", "deg": 30}).mods(0).hinge == 30
    assert swing.mods(1 / 3).hinge == pytest.approx(-60 * 4 / 9 * 5 / 3)  # each swing smaller
    assert abs(swing.mods(0.97).hinge) < 0.5  # it comes to rest, not to a stop


def test_combined_motions_add_up():
    a = P.Mods(sx=2, sy=0.5, hinge=10, pivot="top", smear=4, glitch=0.5, scramble=0.6, count=3, step=2, seed=1)
    b = P.Mods(sx=3, sy=0.5, hinge=5, smear=6, glitch=0.3, scramble=0.2, step=5)
    m = a.combine(b)
    assert (m.sx, m.sy, m.hinge, m.pivot, m.smear, m.glitch, m.scramble, m.count, m.step, m.seed) == (
        6, 0.25, 15, "top", 10, 0.5, 0.6, 3, 5, 1)
    assert b.combine(P.Mods(pivot="bottom", seed=4)).pivot == "bottom" and b.combine(P.Mods(seed=4)).seed == 4


def test_elastic_overshoots_and_stretch_opens_from_the_middle():
    el = P.parse("elastic")
    assert el.dur == 0.9 and max(el.mods(p).scale for p in np.linspace(0, 1, 200)) > 1.2 and el.mods(1).scale == 1
    st = P.parse({"name": "stretch", "ease": "linear"})
    assert st.mods(0.25).sx == 0.25 and st.mods(0.25).sy == pytest.approx(1.3) and st.mods(1).sx == 1
    narrow, = frames(scene([card({"name": "stretch", "ease": "linear"})]), (3,))  # 0.15 s of 0.5
    top, bottom, left, right = lit(narrow)
    assert right - left < 50 and abs((left + right) / 2 - 200) <= 1 and bottom - top > 90


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_iris_clock_and_blinds_reveal_the_box_in_their_shapes(backend):
    half = {"dur": 1, "ease": "linear"}
    (iris,) = frames(scene([card(dict(half, name="iris"))]), (10,), backend)
    assert iris[150, 200].max() > 250 and iris[112, 142].max() < 8  # the middle shown, a corner not yet
    # a circle of half of half the box's diagonal, 72 px: 36 px across and down, short of it on the slant
    assert iris[150, 233].max() > 250 and iris[150, 240].max() < 8 and iris[183, 200].max() > 250
    assert iris[175, 230].max() < 8
    (clock,) = frames(scene([card(dict(half, name="clock"))]), (5,), backend)  # a quarter: twelve to three
    assert clock[120, 250].max() > 250 and clock[180, 250].max() < 8 and clock[120, 150].max() < 8
    (blinds,) = frames(scene([card(dict(half, name="blinds", count=4))]), (5,), backend)  # each strip a quarter open
    column = blinds[110:190, 200].max(axis=1) > 128
    assert column.sum() == pytest.approx(20, abs=2) and column[0] and column[20] and not column[10]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_rise_mask_rises_from_behind_the_bottom_edge(backend):
    (img,) = frames(scene([card({"name": "rise-mask", "dur": 1, "ease": "linear"})]), (5,), backend)
    assert lit(img) == (170, 189, 140, 259)  # a quarter up, cut at the box's bottom edge
    spec = scene([words({"name": "rise-mask", "dur": 0.5, "stagger": 0.5}, text="ONE\nTWO", split="lines")])
    (first,) = frames(spec, (10,), backend)
    top, bottom, _, _ = lit(first)
    assert bottom < 150  # the second line waits behind its own edge, below the middle
    assert lit(frames(spec, (39,), backend)[0])[1] > 160


def test_wipes_on_split_text_wipe_each_unit():
    spec = scene([words({"name": "wipe", "dur": 0.5, "stagger": 0.5, "ease": "linear"}, text="HH HH", split="words")])
    plain, = frames(scene([words(None, text="HH HH")]))
    (img,) = frames(spec, (15,))  # the first word in, the second half way
    _, _, left, right = lit(plain)
    mid = (left + right) // 2
    assert np.array_equal(img[:, :mid], plain[:, :mid])
    seen = (img[:, mid:].max(axis=0).max(axis=1) > 128).nonzero()[0]
    assert 0 < len(seen) < (plain[:, mid:].max(axis=0).max(axis=1) > 128).sum() * 0.7


def test_scrambled_settles_letters_in_turn_and_keeps_their_kind():
    text = "HELLO world 42, ok?"
    assert P.scrambled(text, 0, 0.0, 0, len(text), "k", 0, "") == text
    mess = P.scrambled(text, 0, 1.0, 0, len(text), "k", 0, "")
    assert len(mess) == len(text) and sum(a != b for a, b in zip(mess, text)) >= 12
    for a, b in zip(mess, text):
        assert (a.isupper(), a.islower(), a.isdigit()) == (b.isupper(), b.islower(), b.isdigit()) and (
            b.isalnum() or a == b)
    half = P.scrambled(text, 0, 0.5, 0, len(text), "k", 0, "")
    assert half[:5] == "HELLO" and half[-4:] != text[-4:]  # left to right
    assert P.scrambled(text, 0, 1.0, 0, len(text), "k", 1, "") != mess  # a new pick at each step
    assert P.scrambled(text, 0, 1.0, 0, len(text), "j", 0, "") != mess  # and for each key
    assert set(P.scrambled("Привет", 0, 1.0, 0, 6, "k", 0, "Привет")) <= set("Привет")
    assert P.scrambled("a,b-c", 0, 1.0, 0, 5, "k", 0, "xyz")[1::2] == ",-"
    m = P.parse("scramble")
    assert len({m.mods(p).step for p in np.linspace(0, 0.99, 30)}) == 16  # a new pick 16 times over the motion


def test_scramble_settles_into_the_text_and_is_the_same_each_time():
    spec = scene([words({"name": "scramble", "seed": 2})])
    start, end = frames(spec, (2, 19))
    plain, = frames(scene([words(None)]))
    assert np.array_equal(end, plain) and not np.array_equal(start, plain)
    assert np.array_equal(start, frames(spec, (2,))[0])
    assert not np.array_equal(start, frames(scene([words({"name": "scramble", "seed": 3})]), (2,))[0])
    assert P.parse("scramble").dur == 0.8


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_glitch_splits_the_colours_and_jumps_slices(backend):
    spec = scene([card({"name": "glitch", "dur": 1})])
    img, done = frames(spec, (4, 39), backend)
    assert lit(done) == (110, 189, 140, 259)
    red = (img[:, :, 0] > 150) & (img[:, :, 1] < 100)
    cyan = (img[:, :, 2] > 150) & (img[:, :, 0] < 100)
    assert red[:, :200].sum() > 50 and cyan[:, 200:].sum() > 50  # red pulled left, cyan right
    assert red[:, 200:].sum() < red[:, :200].sum() / 4
    rows = [lit(img[y:y + 1]) for y in range(112, 188)]
    assert len({r[2] for r in rows if r}) > 1  # slices jumped by different amounts
    assert not np.array_equal(img, frames(scene([card({"name": "glitch", "dur": 1, "seed": 9})]), (4,), backend)[0])


def test_whip_smears_across_and_not_down():
    spec = scene([card({"name": "whip", "dur": 1, "dist": -200})])  # from the left
    img, done = frames(spec, (1, 39))
    top, bottom, left, right = lit(img, 8)
    assert lit(done) == (110, 189, 140, 259) and right - left > 160 and bottom - top < 100
    assert P.parse("whip").mods(0).dx == 400 and P.parse("whip").mods(0).smear == 48


def test_motion_options_are_checked_and_old_names_keep_their_meaning():
    assert P.option_problem("bounces", 2) is None and P.option_problem("count", 12) is None
    for key, v in (("bounces", -1), ("bounces", 2.5), ("bounces", 21), ("count", 0), ("count", "4")):
        assert P.option_problem(key, v), (key, v)
    spec = scene([card({"name": "blinds", "count": 0}), card({"name": "land", "bounces": 1}, id="d")])
    issues = check(spec)["issues"]
    assert [i["code"] for i in issues if i["level"] == "error"] == ["bad-value"]
    assert not [i for i in issues if i["code"] == "unknown-key"]
    seeded = [i["code"] for i in check(scene([words({"name": "scramble", "seed": 3}),
                                              card({"name": "fade", "seed": 3})]))["issues"]]
    assert seeded.count("ignored-key") == 1  # a seed picks scramble's letters; fade has no use for one
    assert P.canonical("bounce") == P.canonical("bounceIn") == "pop"  # bounce stays pop; the drop is land
    assert [P.canonical(n) for n in ("dropBounce", "radialWipe", "riseMask", "rgbSplit", "decode", "venetian")] == [
        "land", "clock", "rise-mask", "glitch", "scramble", "blinds"]
    assert P.parse("rise").dur == 0.5 and P.parse("fade+land").dur == 1.0
    assert P.parse({"name": "land", "dur": 0.3}).dur == 0.3


def test_new_motions_are_a_function_of_the_frame_alone():
    spec = scene([words({"name": "scramble+glitch", "dur": 1.2}),
                  card({"name": "squash+whip", "dur": 1.5}, y=60, h=40),
                  words({"name": "glitch", "stagger": 0.2, "dur": 0.6}, id="u", y=250, split="words")])
    a, b, c = frames(spec, (7, 3, 7))
    (d,) = frames(spec, (7,))
    assert np.array_equal(a, c) and np.array_equal(a, d) and not np.array_equal(a, b)


def test_an_exit_plays_the_motion_backwards():
    spec = scene([card(None, exit={"name": "iris", "dur": 1, "ease": "linear"})])
    (img,) = frames(spec, (30,))  # half way out
    assert img[150, 200].max() > 250 and img[112, 142].max() < 8


def test_split_units_glitch_smear_and_stretch_in_their_own_boxes():
    def spread(enter, n=2):
        (img,) = frames(scene([words(enter, text="HH HH", split="words")]), (n,))
        return img
    plain = frames(scene([words(None, text="HH HH", split="words")]))[0]
    top, bottom, left, right = lit(plain)
    img = spread({"name": "glitch", "dur": 1, "stagger": 0.01})
    assert ((img[:, :, 0] > 150) & (img[:, :, 1] < 100)).sum() > 30  # red copies behind each word
    img = spread({"name": "whip", "dur": 1, "dist": 0, "stagger": 0.01}, 1)
    t, b, lo, hi = lit(img, 8)
    assert hi - lo > (right - left) + 40 and b - t < (bottom - top) + 12  # each word smeared across, not down
    img = spread({"name": "stretch", "dur": 1, "ease": "linear", "stagger": 0.01}, 4)  # sx 0.2 about each word
    cols = (img.max(axis=(0, 2)) > 128).nonzero()[0]
    assert len(cols) < ((plain.max(axis=(0, 2)) > 128).sum()) * 0.4 and cols.min() > left + 10


def test_scramble_on_split_units_settles_each_unit_by_itself():
    spec = scene([words({"name": "scramble", "dur": 1, "stagger": 0.001, "seed": 5}, split="chars")])
    half, done = frames(spec, (10, 39))  # each one-letter unit settles at 0.13 of its own scramble, by seed 5
    assert np.array_equal(half, done)
    (whole,) = frames(scene([words({"name": "scramble", "dur": 1, "seed": 5})]), (10,))
    assert not np.array_equal(whole, done)  # the whole text settles left to right: half way, the right half is not


def test_depth_of_field_sees_a_stretch_as_the_frame_does():
    def tall(enter):  # how tall the card's blurred image reaches
        spec = scene([card(enter, z=-400)], w=400, h=300)
        spec["camera"] = {"aperture": 20, "focus": 0}
        (img,) = frames(spec, (4,))
        top, bottom, _, _ = lit(img, 8)
        return bottom - top
    still, stretched = tall(None), tall({"name": "stretch", "dur": 1, "ease": "linear"})  # sx 0.2, sy 1.32
    # 1.32 times as tall and more: its blur is the one for its smaller area on screen, and stretched with it
    assert stretched > still * 1.32 + 12
