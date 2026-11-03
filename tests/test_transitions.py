"""Sequence transitions iris, push, whip, luma and glitch, the luma motion they share with enter and exit, and
"fade": false, which keeps a moving layer opaque."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript import presets as P
from kinescript.render import Renderer
from kinescript.verify import check

RED, BLUE, GREEN = (255, 0, 0), (0, 0, 255), (0, 255, 0)


@pytest.fixture
def media(tmp_path):
    """Solid red and blue pictures to cut between, and a luma map white on its left half and black on its right."""
    out = {}
    for name, col in (("red", RED), ("blue", BLUE)):
        a = np.zeros((50, 100, 4), np.uint8)
        a[..., :3], a[..., 3] = col, 255
        out[name] = str(tmp_path / f"{name}.png")
        skia.Image.fromarray(a, colorType=skia.ColorType.kRGBA_8888_ColorType).save(out[name])
    a = np.full((10, 10, 4), 255, np.uint8)
    a[:, 5:, :3] = 0
    out["map"] = str(tmp_path / "map.png")
    skia.Image.fromarray(a).save(out["map"])
    a = np.full((50, 100, 4), 255, np.uint8)
    a[:, (np.arange(100) // 4) % 2 == 1, :3] = 0
    out["stripes"] = str(tmp_path / "stripes.png")  # black and white, 4 px each
    skia.Image.fromarray(a).save(out["stripes"])
    return out


def cut(media, transition, **seq):
    """Red for 2 s, then blue for 2 s, over a green background: a 1 s transition runs from frame 20 to 40."""
    return {"canvas": {"w": 200, "h": 100, "fps": 20, "duration": 3, "bg": "#00ff00"}, "layers": [
        dict({"type": "sequence", "id": "cut", "w": 200, "h": 100, "fit": "cover",
              "clips": [{"src": media["red"], "dur": 2}, {"src": media["blue"], "dur": 2}],
              "transition": dict({"dur": 1}, **transition)}, **seq)]}


def frames(spec, n, backend="cpu"):
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    return [r.frame(i)[:, :, :3].astype(int).copy() for i in n]


def near(px, col, tol=24):
    return bool(np.abs(np.asarray(px) - col).max() <= tol)


def codes(spec):
    return [(i["level"], i["code"]) for i in check(spec)["issues"]]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_push_moves_both_clips_left_side_by_side(media, backend):
    run = frames(cut(media, {"type": "push"}), range(18, 43), backend)
    assert all(f[..., 1].max() <= 8 for f in run)  # the background never shows between them
    mid = run[30 - 18][50]
    assert near(mid[90], RED) and near(mid[110], BLUE)  # half way, the seam is at the middle
    early = run[24 - 18][50]
    seam = int(np.argmax(early[:, 2] > 128))
    assert 150 < seam < 199 and all(near(p, BLUE) for p in early[seam + 1:])  # the incoming clip is opaque
    assert all(near(p, RED) for p in early[:seam - 1])


@pytest.mark.parametrize("kind, scale, x", [("push", 0.5, 320.25), ("push", 0.73, 150.3), ("whip", 1, 260.5),
                                            ("whip", 1.37, 260.6)])
def test_no_background_between_them_when_scaled_or_between_pixels(media, kind, scale, x):
    w = 640 if scale == 0.5 else 200
    seq = cut(media, {"type": kind}, w=w, anchor="center")["layers"][0]
    spec = {"canvas": {"w": 520, "h": 300, "fps": 20, "duration": 3, "bg": "#00ff00"}, "layers": [
        {"type": "group", "x": x, "y": 150.4, "anchor": "center", "w": w, "h": 100, "clip": True, "scale": scale,
         "children": [seq]}]}
    hw, hh = w * scale / 2 - 2, 50 * scale - 2
    for f in frames(spec, range(19, 42)):
        assert f[int(150.4 - hh) + 1:int(150.4 + hh), int(x - hw) + 1:int(x + hw), 1].max() <= 8


def test_push_moves_by_the_sequences_width(media):
    spec = cut(media, {"type": "push"}, w="50%")
    spec["canvas"]["w"] = 400
    row = frames(spec, (30,))[0][50]
    assert near(row[90], RED) and near(row[110], BLUE) and near(row[300], GREEN)
    enter, leave = __import__("kinescript.components", fromlist=["_joining"])._joining("whip", 0.6, {}, 1000)
    assert enter["amount"] == leave["amount"] == 40 and enter["dist"] == -leave["dist"] == 996


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_whip_blurs_inside_each_clip(media, backend):
    run = frames(cut(media, {"type": "whip"}), range(18, 43), backend)
    assert all(f[..., 1].max() <= 8 for f in run)  # their edges stay solid: no background at the seam
    mid = run[30 - 18][50]
    assert near(mid[20], RED) and near(mid[180], BLUE)
    striped = {"clips": [{"src": media["stripes"], "dur": 2}, {"src": media["stripes"], "dur": 2}]}
    push, whip = (frames(cut(media, {"type": t}, **striped), (30,), backend)[0][50] for t in ("push", "whip"))
    assert whip[10:190, 0].std() < 0.5 * push[10:190, 0].std()  # its stripes smeared across


def test_iris_opens_the_next_clip_from_the_middle(media):
    f = frames(cut(media, {"type": "iris"}), (30,))[0]
    assert near(f[50, 100], BLUE) and near(f[2, 2], RED) and near(f[97, 197], RED)


def test_glitch_breaks_the_outgoing_clip_up_and_settles_the_next(media):
    early, late, done = frames(cut(media, {"type": "glitch"}), (25, 36, 41))
    assert (np.abs(early - RED).max(axis=2) > 40).mean() > 0.02  # bands jumped and colours split
    assert (np.abs(late - BLUE).max(axis=2) > 40).mean() > 0.01  # still settling
    assert np.abs(done - BLUE).max() <= 2


def test_luma_wipes_by_the_maps_brightness(media):
    start, mid, end = frames(cut(media, {"type": "luma"}), (20, 30, 40))
    row = mid[50]
    assert near(row[40], BLUE) and near(row[160], RED)  # left to right by default, dark first
    assert 60 < row[100][0] < 200 and 60 < row[100][2] < 200  # soft: half way at the middle
    assert np.abs(start - RED).max() <= 2 and np.abs(end - BLUE).max() <= 2
    f = frames(cut(media, {"type": "luma", "map": {"radial": [0.5, 0.5, 0.75]}}), (30,))[0]
    assert near(f[50, 100], BLUE) and near(f[2, 2], RED)
    f = frames(cut(media, {"type": "luma", "map": media["map"]}), (30,))[0]
    assert near(f[50, 160], BLUE) and near(f[50, 40], RED) and near(f[50, 75], RED)  # its black right half first,
    # stretched over the clip


def test_a_luma_map_is_read_from_the_scenes_folder(media, tmp_path):
    spec = cut(media, {"type": "luma", "map": "map.png"})
    (tmp_path / "cut.json").write_text(__import__("json").dumps(spec))
    s = k.load(tmp_path / "cut.json")
    row = Renderer(s, backend="cpu").frame(30)[50, :, :3].astype(int)
    assert near(row[160], BLUE) and near(row[40], RED)


def test_luma_soft_spreads_the_edge(media):
    def mixed(soft):
        row = frames(cut(media, {"type": "luma", "soft": soft}), (30,))[0][50]
        return sum(1 for p in row if 30 < p[0] < 225 and 30 < p[2] < 225)
    assert mixed(0) <= 3 < mixed(0.2) < mixed(0.6)


def test_a_missing_luma_map_is_reported_and_falls_back(media, tmp_path):
    assert ("error", "media-missing") in codes(cut(media, {"type": "luma", "map": "nowhere.png"}))
    gone = tmp_path / "gone.png"
    gone.write_bytes(open(media["map"], "rb").read())
    r = Renderer(k.Scene(cut(media, {"type": "luma", "map": str(gone)})), backend="cpu")
    gone.unlink()  # after the scene was checked: it wipes by the default gradient instead
    row = r.frame(30)[50, :, :3].astype(int)
    assert near(row[40], BLUE) and near(row[160], RED)


def test_luma_enters_any_layer_and_a_split_text_whole():
    rect = {"type": "rect", "x": 100, "y": 50, "w": 200, "h": 100, "fill": "#ffffff",
            "enter": {"name": "luma", "dur": 1, "ease": "linear", "soft": 0}}
    spec = {"canvas": {"w": 200, "h": 100, "fps": 20, "duration": 2, "bg": "#000000"}, "layers": [rect]}
    row = frames(spec, (10,))[0][50]
    assert row[60].min() > 200 and row[140].max() < 20
    text = {"type": "text", "text": "HHHHHH", "size": 40, "x": 100, "y": 50, "fill": "#ffffff",
            "enter": {"name": "luma", "dur": 1, "ease": "linear", "soft": 0}}
    whole, split = (frames(dict(spec, layers=[dict(text, split=s)]), (10,))[0] for s in (None, "chars"))
    cols = np.nonzero(split.max(axis=(0, 2)) > 128)[0]
    assert cols.max() < 110 and abs(int(whole.sum()) - int(split.sum())) < 0.02 * whole.sum()


def test_fade_false_keeps_a_motion_opaque():
    m = P.parse({"name": "left", "dist": 300, "fade": False, "ease": "linear"}).mods(0.1)
    assert m.alpha == 1.0 and m.solid and m.dx == pytest.approx(270)
    assert P.parse("left").mods(0.1).alpha < 1 and not P.parse("left").mods(0.1).solid
    assert P.option_problem("fade", "no") and P.option_problem("soft", 1.5) and P.option_problem("map", 3)
    assert P.option_problem("fade", False) is None and P.option_problem("soft", 0) is None
    spec = {"canvas": {"w": 200, "h": 100, "fps": 20, "duration": 2, "bg": "#000000"}, "layers": [
        {"type": "rect", "x": 100, "y": 50, "w": 40, "h": 40, "fill": "#ffffff",
         "enter": {"name": "left", "dist": 60, "fade": False}}]}
    assert frames(spec, (1,))[0].max() == 255


def test_combine_keeps_a_luma_wipe_and_solid():
    luma = P.PRESETS["luma"](0.5, {"map": "a.png", "soft": 0.4})
    for m in (luma.combine(P.IDENT), P.IDENT.combine(luma)):
        assert (m.wipe, m.matte, m.soft, m.reveal) == ("luma", "a.png", 0.4, 0.5)
    assert P.IDENT.combine(P.Mods(solid=True)).solid and P.Mods(solid=True).combine(P.IDENT).solid
    assert P.PRESETS["luma"](0.5, {}).matte == {"angle": 90}


def test_check_reads_transitions(media):
    assert ("info", "renamed-value") in codes(cut(media, {"type": "circle"}))
    assert codes(cut(media, {"type": "whip"})) == []
    assert ("warning", "ignored-key") in codes(cut(media, {"type": "push", "soft": 0.3}))
    assert ("error", "bad-value") in codes(cut(media, {"type": "luma", "soft": 2}))
    assert ("error", "bad-value") in codes(cut(media, {"type": "luma", "map": 5}))
    one_stop = {"radial": [0.5, 0.5], "stops": ["#000"]}
    assert ("error", "bad-value") in codes(cut(media, {"type": "luma", "map": one_stop}))
    assert codes(cut(media, {"type": "luma", "map": {"radial": [0.5, 0.5]}, "softness": 0.1})) == \
        [("info", "renamed-key")]
    layer = {"type": "rect", "w": 10, "h": 10, "enter": {"name": "rise", "map": {"angle": 0}}}
    assert ("warning", "ignored-key") in codes({"canvas": {"w": 100, "h": 100}, "layers": [layer]})
    layer["enter"] = {"name": "luma", "map": {"angle": 0}, "fade": "no"}
    assert codes({"canvas": {"w": 100, "h": 100}, "layers": [layer]}) == [("error", "bad-value")]


def test_transitions_are_a_function_of_the_frame_alone(media):
    for kind in ("push", "whip", "glitch", "luma"):
        spec = cut(media, {"type": kind})
        ordered = frames(spec, range(20, 41, 4))
        r = Renderer(k.Scene(spec), backend="cpu")
        for i, n in reversed(list(enumerate(range(20, 41, 4)))):
            assert np.array_equal(r.frame(n)[:, :, :3].astype(int), ordered[i]), (kind, n)
