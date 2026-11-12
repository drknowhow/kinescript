"""Units of a text that are only moved along the page are drawn where they go, with the canvas left as it is, and
their runs of one glyph are drawn together, as one run of placed glyphs. A frame is the one drawn unit by unit:
here the same text with a stroke that cannot be seen, which is drawn that way, each unit under a canvas moved to
it."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript.render import Renderer

BOTH = pytest.mark.parametrize("backend", ["cpu", "gpu"])
RED_TO_BLUE = "=mix('#ff0000', '#0000ff', i / (n - 1))"


def scene(layers, **canvas):
    return {"canvas": dict({"w": 640, "h": 200, "fps": 10, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def text(**kw):
    return dict({"id": "t", "type": "text", "text": "HHHH", "size": 60, "x": 320, "y": 100, "fill": "#ffffff",
                 "split": "chars"}, **kw)


def one_by_one(spec):
    """spec with a stroke nobody can see on what it writes: that is drawn unit by unit."""
    return dict(spec, layers=[dict(L, stroke="#00000000") if L["type"] in ("text", "counter", "captions") else L
                              for L in spec["layers"]])


def drawn(monkeypatch, spec, n=0, backend="cpu"):
    """Frame n of spec, the runs of placed glyphs its text was drawn as (where each one's glyphs went) and the
    strings drawn one at a time (each with where it went)."""
    placed, strings, real, each = [], [], skia.TextBlobBuilder, Renderer._draw_pieces

    class Builder:
        def __init__(self):
            self.to = real()

        def allocRunPos(self, font, glyphs, places):
            assert len(glyphs) == len(places)
            placed.append(list(places))
            self.to.allocRunPos(font, glyphs, places)

        def make(self):
            return self.to.make()

    def pieces(c, pcs, s, x, y, font, paint):
        strings.append((s, x, y))
        each(c, pcs, s, x, y, font, paint)
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    with monkeypatch.context() as mp:
        mp.setattr(skia, "TextBlobBuilder", Builder)
        mp.setattr(Renderer, "_draw_pieces", staticmethod(pieces))
        img = r.frame(n)[:, :, :3].astype(int)
    return img, placed, strings


def alike(a, b) -> bool:
    """Two frames the same, but for a letter's edge that fell a part of a pixel over: a unit left out, drawn in
    the wrong place, colour or turn is hundreds of times that."""
    return float(np.abs(a - b).mean()) < 0.1


def as_one_by_one(monkeypatch, spec, n=(0,), backend="cpu"):
    """The frames n of spec with what drawn tells of each, once each is seen to be the frame drawn unit by unit."""
    out = []
    for i in n:
        img, placed, strings = drawn(monkeypatch, spec, i, backend)
        ref, none, _ = drawn(monkeypatch, one_by_one(spec), i, backend)
        assert none == [] and alike(img, ref), (i, float(np.abs(img - ref).mean()))
        out.append((img, placed, strings))
    return out


@BOTH
def test_units_that_are_only_moved_are_drawn_as_one_run_of_placed_glyphs(monkeypatch, backend):
    [(flat, level, _)] = as_one_by_one(monkeypatch, scene([text()]), backend=backend)
    spec = scene([text(unitX="=i * 6", unitY="=i * 24 - 36")])
    [(img, placed, strings)] = as_one_by_one(monkeypatch, spec, backend=backend)
    assert [len(run) for run in placed] == [4] == [len(run) for run in level] and strings == []
    for i, ((x, y), (x0, y0)) in enumerate(zip(placed[0], level[0])):
        assert (x - x0, y - y0) == pytest.approx((i * 6, i * 24 - 36))
    assert flat.max() > 200 and not alike(img, flat)  # the letters are there, and they did move


@pytest.mark.parametrize("more, runs", [
    ({"text": "HH**HH**HH"}, [2, 2, 2]),  # the letters in the middle are another colour
    ({"unitOpacity": "=1 - i * 0.3"}, [1, 1, 1, 1]),
    ({"unitFill": RED_TO_BLUE}, [1, 1, 1, 1]),
    ({"unitFill": "=mix('#ff0000', '#0000ff', floor(i / 2))"}, [2, 2]),
    ({"text": "HH**HH**HH", "unitOpacity": "=1 - floor(i / 3) * 0.6"}, [2, 1, 1, 2]),
])
def test_a_run_of_placed_glyphs_ends_where_the_paint_changes(monkeypatch, more, runs):
    spec = scene([text(unitY="=sin(i) * 20", **more)])
    [(img, placed, strings)] = as_one_by_one(monkeypatch, spec)
    assert [len(run) for run in placed] == runs and strings == []
    assert len({tuple(px) for px in img[img.max(axis=2) > 200].tolist()}) >= len(set(runs))  # as many colours


@BOTH
@pytest.mark.parametrize("more, runs", [({}, [2, 1]), ({"unitFill": RED_TO_BLUE}, [1, 1, 1])])
def test_a_unit_the_canvas_turns_is_drawn_in_its_turn(monkeypatch, backend, more, runs):
    """The letters lie over each other, so which was drawn last shows; the third is turned, which the canvas does
    for it alone: the two before it are drawn first, then it, then the last."""
    spec = scene([text(unitX="=-i * 30", unitRotate="=40 if i == 2 else 0", **more)])
    [(img, placed, strings)] = as_one_by_one(monkeypatch, spec, backend=backend)
    assert [len(run) for run in placed] == runs and [s for s, _, _ in strings] == ["H"]
    unturned = scene([text(unitX="=-i * 30", **more)])
    [(flat, level, none)] = as_one_by_one(monkeypatch, unturned, backend=backend)
    assert [len(run) for run in level] == ([4] if not more else [1, 1, 1, 1]) and none == [] and not alike(img, flat)


def test_under_a_gradient_a_unit_that_moves_takes_the_gradient_with_it(monkeypatch):
    fill = {"linear": [[0, 0], [0, 1]], "stops": ["#ff0000", "#0000ff"]}
    [(flat, placed, strings)] = as_one_by_one(monkeypatch, scene([text(fill=fill)]))
    assert [len(run) for run in placed] == [4] and strings == []
    [(img, placed, strings)] = as_one_by_one(monkeypatch, scene([text(fill=fill, unitY="=i * 24 - 36")]))
    assert placed == [] and [s for s, _, _ in strings] == ["H"] * 4
    red = []  # how red each letter is a little under its top: the gradient runs red to blue down the text
    for x0, x1 in ((220, 260), (380, 420)):  # the first letter and the last, which lies 72 px lower
        part = img[:, x0:x1]
        row = part[np.nonzero((part.sum(axis=2) > 200).any(axis=1))[0][0] + 4]
        r, _, b = row[row.sum(axis=1) > 200].mean(axis=0)
        red.append(r / (r + b))
    assert red[0] > 0.6 and red[1] == pytest.approx(red[0], abs=0.05)


@BOTH
@pytest.mark.parametrize("name", [
    "rise", "drop", "left", "right", "fade", "pop", "zoom", "blur", "spin", "track", "wipe", "snap", "flip",
    "flip-up", "tumble", "push", "land", "squash", "elastic", "swing", "stretch", "scramble", "glitch", "whip",
    "rise+blur", "type"])
def test_a_text_entering_unit_by_unit_is_drawn_as_it_was(monkeypatch, backend, name):
    """Through an entrance some units are in, some on their way and some still out; those on their way are turned,
    blurred, cut or faded each in its own way."""
    spec = scene([text(text="HH **HH** Hi", enter={"name": name, "dur": 0.6, "stagger": 0.15},
                       unitY="=sin(T * 4 + i) * 6")])
    shots = as_one_by_one(monkeypatch, spec, n=(1, 3, 5, 7, 9, 12, 16), backend=backend)
    assert sum(len(run) for run in shots[-1][1]) == 6  # all in at the end: every letter is a placed glyph
    assert not alike(shots[0][0], shots[-1][0])


@pytest.mark.parametrize("layer", [
    text(split="words", text="A few words I say", size=40, unitY="=i * 8 - 16"),
    text(split="lines", text="One line\nand I", size=40, unitX="=i * 30"),
    {"id": "t", "type": "text", "text": "No split, **a** word", "size": 40, "x": 320, "y": 100},
    {"id": "t", "type": "text", "text": "Typed as I go", "size": 40, "x": 320, "y": 100, "caret": True,
     "enter": {"name": "type", "dur": 1.5}},
    {"id": "t", "type": "text", "text": "Settles as I go", "size": 40, "x": 320, "y": 100,
     "enter": {"name": "scramble", "dur": 1.5}},
    text(text="Spaced out", size=40, letterSpacing=6, unitY="=i * 4 - 16"),
    text(text="שלום HH مرحبا", size=40, unitY="=i * 4 - 16"),
    text(text="Hi 👍 é", size=40, unitY="=i * 8 - 16"),
    {"id": "t", "type": "counter", "value": [[0, 0], [1.5, 1234, "linear"]], "size": 60, "x": 320, "y": 100,
     "split": "chars", "unitY": "=i * 6 - 9"},
    {"id": "t", "type": "captions", "x": 320, "y": 100, "size": 40, "mode": "pop",
     "cues": [[0.0, 0.9, "I am a cue"], [1.0, 1.9, "and so am I"]]},
    {"id": "t", "type": "captions", "x": 320, "y": 100, "size": 40, "mode": "karaoke",
     "cues": [[0.0, 0.9, "I am a cue"], [1.0, 1.9, "and so am **I**"]]},
], ids=["words", "lines", "whole", "typed", "scrambled", "spaced", "right to left", "pieces", "counter",
        "captions that pop", "captions"])
def test_words_and_whole_texts_are_drawn_as_they_were(monkeypatch, layer):
    """A run of several glyphs is drawn as a string where its unit puts it, and a run of one waits to be drawn with
    the next: whatever the text is split by, typed or scrambled, shaped or from another font."""
    shots = as_one_by_one(monkeypatch, scene([layer]), n=(0, 2, 4, 7, 11, 14, 18))
    assert any(img.max() > 150 for img, _, _ in shots)


def test_a_frame_of_placed_glyphs_is_the_same_whatever_was_drawn_before_it(monkeypatch):
    spec = scene([text(text="HH **HH** Hi", enter={"name": "rise+blur", "dur": 0.6, "stagger": 0.15},
                       unitY="=sin(T * 4 + i) * 6")])
    r = Renderer(k.Scene(spec), backend="cpu")
    for n in (14, 3, 18, 7):
        assert np.array_equal(r.frame(n), Renderer(k.Scene(spec), backend="cpu").frame(n)), n
