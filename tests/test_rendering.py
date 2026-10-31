"""Rendering fixes: group fades with blended children, motion blur at a layer's ends, grain under motion blur,
repeatable GPU models, boxes that hold the ink, check rules (hidden text, flashes, odd sizes, glyphs), and text in
scripts the layer's font does not have."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript.render import Renderer, _bidi_order, _fallback, _text_font
from kinescript.verify import check, probe


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 2}, **canvas), "layers": layers}


def _renderer(spec, backend="cpu", **kw):
    try:
        return Renderer(k.Scene(spec), backend=backend, **kw)
    except RuntimeError:
        pytest.skip("no GPU")


def _codes(res, code):
    return [i for i in res["issues"] if i["code"] == code]


# ---- B11: a group's opacity with blended children ---------------------------------------------------------------

@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_group_opacity_fades_blended_children_without_a_jump(backend):
    def centre(op):
        spec = scene([{"id": "g", "type": "group", "opacity": op, "children": [
            {"id": "c", "type": "rect", "x": 160, "y": 90, "w": 100, "h": 100, "fill": "#ff8000",
             "blend": "multiply"}]}], bg="#808080")
        return _renderer(spec, backend).frame(0)[90, 160, :3].astype(float)

    full, bg = centre(1.0), np.array([128.0, 128.0, 128.0])
    assert np.abs(full - [128, 64, 0]).max() <= 1  # multiply over grey
    for op in (0.999, 0.9989, 0.99, 0.5, 0.1):
        # Before: 0.999 drew (128,64,0) and 0.9989 (255,128,0), the child blending with nothing.
        assert np.abs(centre(op) - (bg + op * (full - bg))).max() <= 2, op


# ---- B12, B13: motion blur at a layer's first and last frames, and grain ---------------------------------------

def test_motion_blur_keeps_a_layer_to_its_own_frames():
    spec = {"canvas": {"w": 64, "h": 36, "fps": 30, "duration": 2, "bg": "#ffffff"},
            "motionBlur": {"samples": 8, "shutter": 0.5},
            "layers": [{"id": "red", "type": "rect", "x": 32, "y": 18, "w": 64, "h": 36, "fill": "#ff0000",
                        "at": 1.0, "dur": 0.5}]}
    r = _renderer(spec)
    got = {n: tuple(int(v) for v in r.frame(n)[18, 32, :3]) for n in (29, 30, 44, 45)}
    # Before: (255,127,127) at frames 30 and 45, half strength at the first frame and one frame after the end.
    assert got == {29: (255, 255, 255), 30: (255, 0, 0), 44: (255, 0, 0), 45: (255, 255, 255)}


def test_motion_blur_starts_at_full_strength_on_frame_zero():
    spec = {"canvas": {"w": 64, "h": 36, "fps": 30, "duration": 1, "bg": "#000000"},
            "motionBlur": {"samples": 8, "shutter": 0.5},
            "layers": [{"id": "w", "type": "rect", "x": 32, "y": 18, "w": 64, "h": 36, "fill": "#ffffff"}]}
    assert tuple(_renderer(spec).frame(0)[18, 32, :3]) == (255, 255, 255)


def test_grain_is_as_strong_with_motion_blur():
    spec = {"canvas": {"w": 160, "h": 90, "fps": 30, "duration": 2, "bg": "#606060"}, "grade": {"grain": 0.3},
            "layers": [{"id": "r", "type": "rect", "x": 40, "y": 45, "w": 40, "h": 40, "fill": "#ff5a36"}]}
    plain = _renderer(spec).frame(20)[:, 80:, 0].astype(float)
    spec["motionBlur"] = {"samples": 8, "shutter": 0.5}
    blurred = _renderer(spec).frame(20)[:, 80:, 0].astype(float)
    # Before: std 5.86 against 8.35, grain added to every sub-frame and averaged away.
    assert blurred.std() == pytest.approx(plain.std(), rel=0.02)
    assert np.abs(blurred - plain).max() <= 1  # a still scene: the same frame


# ---- B10: models on the GPU -----------------------------------------------------------------------------------

def test_gpu_model_frames_are_repeatable():
    spec = scene([{"id": "m", "type": "model", "shape": "torus", "x": 150, "y": 90, "r": 50, "rotateX": "=T*40",
                   "rotateY": "=T*60", "fill": "#ff5a36", "metal": 0.4}], w=480, h=270, duration=4)
    r = _renderer(spec, "gpu")
    for n in (0, 15, 45, 90):
        first = r.frame(n).copy()
        for _ in range(3):  # 4x MSAA gave edge pixels one level apart between draws
            assert np.array_equal(first, r.frame(n)), n
        assert np.array_equal(first, _renderer(spec, "gpu").frame(n)), n


# ---- D25: boxes hold the ink, check rules -------------------------------------------------------------------------

@pytest.mark.parametrize("layer", [
    {"type": "rect", "x": 160, "y": 90, "w": 80, "h": 40, "fill": "#000000", "stroke": "#ffffff", "strokeWidth": 12},
    {"type": "rect", "x": 160, "y": 90, "w": 80, "h": 40, "stroke": "#ffffff", "strokeWidth": 10, "rotate": 30},
    {"type": "line", "x": 100, "y": 90, "points": [[0, 0], [120, 30]], "stroke": "#ffffff", "strokeWidth": 14},
    {"type": "arrow", "x": 80, "y": 90, "x1": 0, "y1": 0, "x2": 160, "y2": 0, "stroke": "#ffffff", "strokeWidth": 6},
    {"type": "circle", "x": 160, "y": 90, "r": 40, "fill": "#000000", "stroke": "#ffffff", "strokeWidth": 16},
    {"type": "path", "x": 160, "y": 90, "anchor": "center", "points": [[0, 0], [80, 40], [0, 60]],
     "stroke": "#ffffff", "strokeWidth": 10},
])
def test_boxes_hold_the_stroke(layer):
    r = Renderer(k.Scene(scene([dict(layer, id="s")], bg="#000000")), track=True, backend="cpu")
    buf = r.frame(15)
    ys, xs = np.nonzero(buf[..., :3].max(axis=2) > 8)
    ink = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = r.boxes["s"]["rect"]
    # Before: the box was the geometry alone, half a stroke short on every side.
    assert x0 <= ink[0] + 1 and y0 <= ink[1] + 1 and x1 >= ink[2] - 1 and y1 >= ink[3] - 1, (ink, (x0, y0, x1, y1))
    assert x0 >= ink[0] - 3 and y0 >= ink[1] - 3 and x1 <= ink[2] + 3 and y1 <= ink[3] + 3, (ink, (x0, y0, x1, y1))


def test_probe_box_of_a_rect_takes_half_its_stroke():
    out = probe(scene([{"id": "r", "type": "rect", "x": 100, "y": 50, "w": 40, "h": 20, "stroke": "#ffffff",
                        "strokeWidth": 10}]), at=0.5)
    assert out["visible"]["r"]["rect"] == [75.0, 35.0, 125.0, 65.0]


def test_check_flags_text_under_a_later_opaque_layer():
    text = {"id": "t", "type": "text", "text": "Hidden", "x": 160, "y": 90, "size": 30, "fill": "#ffffff"}
    cover = {"id": "cover", "type": "rect", "x": 160, "y": 90, "w": 300, "h": 120, "fill": "#2a6df4"}
    res = check(scene([text, cover], bg="#101418"))
    hidden = _codes(res, "text-hidden")
    assert hidden and hidden[0]["layer"] == "t" and "'cover'" in hidden[0]["message"]
    assert not _codes(check(scene([cover, text], bg="#101418")), "text-hidden")  # a card under the text


def test_check_flags_a_one_word_flash_but_not_a_readable_line():
    flash = check(scene([{"id": "t", "type": "text", "text": "Now", "x": 160, "y": 90, "size": 30, "dur": 0.2}]))
    assert _codes(flash, "read-time") and "flash" in _codes(flash, "read-time")[0]["message"]
    four = check(scene([{"id": "t", "type": "text", "text": "four words right here", "x": 160, "y": 90, "size": 30,
                         "dur": 1.6}]))
    assert not _codes(four, "read-time")  # 1.6 s was flagged for being 12 ms short of 0.4 + 4 x 0.3
    two = check(scene([{"id": "t", "type": "text", "text": "Two words", "x": 160, "y": 90, "size": 30, "dur": 0.5}]))
    assert not _codes(two, "read-time")
    # A chip's highlighted state takes over from its plain one: the word was on screen all along.
    chip = {"type": "text", "text": "[laughs]", "x": 160, "y": 90, "size": 30}
    handover = check(scene([dict(chip, id="plain", dur=1.0), dict(chip, id="lit", at=1.0, dur=0.3)]))
    assert not _codes(handover, "read-time")
    alone = check(scene([dict(chip, id="lit", at=1.0, dur=0.3)]))
    assert _codes(alone, "read-time")


def test_check_does_not_note_a_shape_that_sweeps_across_the_canvas():
    # Off the canvas at its first and last sample only. Before, any sample off the canvas was noted.
    sweep = {"id": "wipe", "type": "rect", "x": "=-100+t*260", "y": 90, "w": 160, "h": 180, "fill": "#2a6df4"}
    assert not _codes(check(scene([sweep])), "offscreen")
    gone = {"id": "gone", "type": "rect", "x": -300, "y": 90, "w": 100, "h": 100, "fill": "#2a6df4"}
    assert _codes(check(scene([gone])), "offscreen")


# ---- A18: odd canvas sizes ------------------------------------------------------------------------------------

def test_check_notes_an_odd_canvas_size():
    res = check(scene([{"id": "r", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40}], w=321, h=181))
    odd = _codes(res, "odd-size")
    assert odd and odd[0]["level"] == "info" and "320x180" in odd[0]["message"] and "322x182" in odd[0]["fix"]
    assert res["ok"]
    assert not _codes(check(scene([{"id": "r", "type": "rect", "x": 160, "y": 90, "w": 40, "h": 40}])), "odd-size")


# ---- A17: characters the font does not have, shaping, direction ---------------------------------------------------

def _covered_by_fallback():
    style = skia.FontStyle(600, 5, skia.FontStyle.kUpright_Slant)
    for ch in ("ש", "世", "ж"):
        if _fallback(style, ord(ch)) is not None:
            return ch
    pytest.skip("no installed font found by the system fallback")


def test_text_draws_characters_the_font_lacks_with_a_system_font():
    ch = _covered_by_fallback()

    def draw(text):
        spec = scene([{"id": "t", "type": "text", "text": text, "x": 160, "y": 90, "size": 60, "fill": "#ffffff",
                       "font": "Space Grotesk"}], bg="#000000")
        return Renderer(k.Scene(spec), backend="cpu").frame(0).copy()

    tofu = draw("͸" * 3)
    got = draw(ch * 3)
    assert got[..., 0].max() > 200  # something draws
    assert not np.array_equal(got, tofu)  # and it is not the empty boxes


def test_bidi_order_of_words():
    ar = ["مرحبا", "بالعالم", "2026!"]
    order, rtl, ctx = _bidi_order(ar, True)
    assert order == [2, 1, 0] and rtl == [True, True, False]
    assert ctx[2] == (True, "R", "R")  # so the ! goes to the left of 2026, as a browser puts it
    mixed = ["Order:", "שלום", "עולם", "123", "done"]
    order, rtl, _ = _bidi_order(mixed, False)
    assert order == [0, 3, 2, 1, 4] and rtl == [False, True, True, False, False]
    assert _bidi_order(["abc", "def"], False)[0] == [0, 1]


def test_right_to_left_pieces_mirror_brackets_and_move_end_punctuation():
    r = Renderer(k.Scene(scene([{"id": "t", "type": "text", "text": "x"}])), backend="cpu")
    font = _text_font(r._typeface("Space Grotesk", 600, False), 40)
    pcs = r._pieces(font, "(שלום)!", True)
    assert pcs[0].text.startswith("!(") and pcs[-1].text.endswith(")")
    assert [p.text for p in r._pieces(font, "2026!", False, (True, "R", "R"))] == ["!2026"]
    assert r._pieces(font, "2026!", False) is None  # left-to-right text keeps drawString
    assert r._pieces(font, "plain text", False) is None


def test_check_warns_about_characters_no_font_has():
    res = check(scene([{"id": "t", "type": "text", "text": "box ͸ here", "x": 160, "y": 90, "size": 30}]))
    miss = _codes(res, "missing-glyph")
    assert miss and miss[0]["layer"] == "t" and "U+0378" in miss[0]["message"]
    assert not _codes(check(scene([{"id": "t", "type": "text", "text": "fine", "x": 160, "y": 90}])), "missing-glyph")


def test_check_warns_that_split_chars_unjoins_arabic():
    res = check(scene([{"id": "t", "type": "text", "text": "مرحبا", "x": 160, "y": 90,
                        "size": 30, "split": "chars"}]))
    assert _codes(res, "unjoined-letters")
    res = check(scene([{"id": "t", "type": "text", "text": "مرحبا", "x": 160, "y": 90,
                        "size": 30, "split": "words"}]))
    assert not _codes(res, "unjoined-letters")


def test_check_warns_that_3d_text_does_not_shape():
    style = skia.FontStyle(600, 5, skia.FontStyle.kUpright_Slant)
    if _fallback(style, ord("ש")) is None:
        pytest.skip("no installed font has Hebrew")
    res = check(scene([{"id": "m", "type": "model", "shape": "text", "text": "שלום", "x": 160,
                        "y": 90, "size": 60, "depth": 10}]))
    assert _codes(res, "unshaped-3d-text")


def test_text_in_other_scripts_is_pure_function_of_index():
    spec = scene([
        {"id": "a", "type": "text", "text": "مرحبا (שלום) 2026!",
         "x": "50%", "y": 50, "size": 28, "enter": {"name": "type", "dur": 1.5}, "caret": True},
        {"id": "b", "type": "text", "text": "a\U0001F600b é́ 世界", "x": "50%", "y": 120, "size": 28,
         "split": "chars", "enter": {"name": "rise", "stagger": 0.05}},
    ])
    r = Renderer(k.Scene(spec), backend="cpu")
    seq = [r.frame(n).copy() for n in range(0, 40, 3)]
    alone = Renderer(k.Scene(spec), backend="cpu").frame(21)
    assert np.array_equal(seq[7], alone)
