"""A 2D layer's "repeat": copies of a shape, text, image or group, each moved in the layer's space and drawn in its own
fill, stroke and opacity. A shape's copies draw in one loop; other layers replay one recording per copy."""
import numpy as np
import pytest

import kinescript as k
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def _renderer(spec, backend="cpu", track=False):
    try:
        return Renderer(k.Scene(spec), backend=backend, track=track)
    except RuntimeError:
        pytest.skip("no GPU")


def _pixels(spec, T=0.0, backend="cpu", track=False):
    r = _renderer(spec, backend, track)
    r.draw(T)
    return r.snapshot().toarray()[..., :3].astype(int), r


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_copies_are_moved_in_the_layers_space_and_take_their_own_fill(backend):
    dots = {"id": "d", "type": "circle", "r": 12, "x": 160, "y": 90, "fill": "#ffffff",
            "repeat": {"count": 3, "x": "=(i - 1) * 80", "fill": ["#ff0000", "#00ff00", "#0000ff"]}}
    img, _ = _pixels(scene([dots]), backend=backend)
    for x, ch in ((80, 0), (160, 1), (240, 2)):
        assert img[90, x].argmax() == ch and img[90, x].max() > 240, (x, img[90, x])
    assert img[90, 120].max() == 0 and img[90, 200].max() == 0
    turned, _ = _pixels(scene([dict(dots, rotate=90)]), backend=backend)  # copies go where the layer's x axis points
    assert turned[10, 160].argmax() == 0 and turned[170, 160].argmax() == 2 and turned[90, 80].max() == 0
    spun, _ = _pixels(scene([{"id": "bar", "type": "rect", "w": 60, "h": 6, "x": 160, "y": 90, "anchor": "left",
                              "fill": "#ffffff", "repeat": {"count": 4, "rotate": "=i * 90", "scale": [1, 0.5]}}]),
                      backend=backend)
    # Copies 0 and 2 point right and left at full length, 1 and 3 down and up at half.
    assert spun[90, 217].min() > 240 and spun[90, 103].min() > 240 and spun[90, 96].max() == 0
    assert spun[117, 160].min() > 240 and spun[63, 160].min() > 240 and spun[124, 160].max() == spun[56, 160].max() == 0


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_shapes_copies_draw_as_that_many_layers_would(backend):
    star = {"type": "path", "d": "M0 -30 L9 -9 L30 -9 L13 5 L19 27 L0 14 L-19 27 L-13 5 L-30 -9 L-9 -9 Z",
            "anchor": "center", "y": 90, "stroke": "#ffffff", "strokeWidth": 3, "dash": [6, 4], "draw": 0.8}
    fills = ["#ff633f", "#7dd3fc", "#fde047"]
    copies, _ = _pixels(scene([dict(star, id="s", x=60, fill="#000000",
                                    repeat={"count": 3, "x": "=i * 100", "rotate": "=i * 20", "fill": fills})]),
                        backend=backend)
    layers, _ = _pixels(scene([dict(star, id=f"s{i}", x=60 + i * 100, rotate=i * 20, fill=fills[i])
                               for i in range(3)]), backend=backend)
    assert np.abs(copies - layers).max() <= (0 if backend == "cpu" else 1)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_copys_opacity_fades_it_and_a_fill_and_stroke_fade_together(backend):
    spec = scene([{"id": "d", "type": "circle", "r": 20, "x": 160, "y": 90, "fill": "#ffffff",
                   "repeat": {"count": 2, "x": [-60, 60], "opacity": [1, 0.5]}}])
    img, _ = _pixels(spec, backend=backend)
    assert img[90, 100].min() > 250 and abs(img[90, 220].mean() - 128) < 3, (img[90, 100], img[90, 220])
    ring = scene([{"id": "d", "type": "circle", "r": 30, "x": 160, "y": 90, "fill": "#ff0000", "stroke": "#0000ff",
                   "strokeWidth": 12, "repeat": {"count": 1, "opacity": 0.5}}])
    img, _ = _pixels(ring, backend=backend)
    inside = img[90, 160 - 27]  # the stroke's inner half, over the fill: the stroke alone shows, at half
    assert inside[0] < 4 and abs(inside[2] - 128) < 4, inside
    recoloured, _ = _pixels(scene([dict(ring["layers"][0], repeat={"count": 1, "stroke": "#00ff00"})]),
                            backend=backend)
    assert recoloured[90, 160 - 30].argmax() == 1 and recoloured[90, 160].argmax() == 0


def test_a_copy_brings_a_stroke_the_layer_lacks_and_a_list_of_colours_goes_round():
    spec = scene([{"id": "d", "type": "rect", "w": 30, "h": 30, "x": 160, "y": 90, "fill": "#202020",
                   "repeat": {"count": 4, "x": "=(i - 1.5) * 60", "stroke": ["#ff0000", "#00ff00"]}}])
    img, _ = _pixels(spec)
    edges = [img[90, int(160 + (i - 1.5) * 60 - 15)] for i in range(4)]
    assert [e.argmax() for e in edges] == [0, 1, 0, 1] and min(e.max() for e in edges) > 200, edges


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_text_images_and_groups_replay_one_recording_as_each_copy_draws_alone(backend, tmp_path):
    import skia
    surf = skia.Surface(20, 20)
    surf.getCanvas().clear(skia.Color(255, 200, 0))
    surf.makeImageSnapshot().save(str(tmp_path / "chip.png"))
    spec = scene([
        {"id": "g", "type": "group", "x": 40, "y": 40, "repeat": {"count": 4, "x": "=i * 70", "rotate": "=i * 20",
                                                                   "opacity": [1, 0.6]},
         "children": [{"type": "rect", "w": 40, "h": 40, "radius": 6, "fill": "#334155"},
                      {"type": "text", "text": "K", "size": 26, "fill": "#f2f0e8"}]},
        {"id": "w", "type": "text", "text": "kine", "size": 22, "x": 60, "y": 110,
         "repeat": {"count": 3, "x": "=i * 90", "scale": "=1 + i * 0.2"}},
        {"id": "chip", "type": "image", "src": str(tmp_path / "chip.png"), "x": 40, "y": 155, "w": 20, "h": 20,
         "repeat": {"count": 4, "x": "=i * 70", "opacity": "=1 - i * 0.2"}},
        {"id": "off", "type": "group", "x": -100, "y": 140, "repeat": {"count": 2, "x": [0, 400]},  # off the frame, its
         "children": [{"type": "rect", "w": 30, "h": 10, "fill": "#00ff00"}]},  # second copy on it
    ])
    replayed, r = _pixels(spec, backend=backend)
    alone, _ = _pixels(spec, backend=backend, track=True)  # tracked frames draw each copy on its own
    assert np.abs(replayed - alone).max() <= (0 if backend == "cpu" else 1)
    assert np.abs(replayed[155, 40 + 210] - (102, 80, 0)).max() <= 2  # the last chip, #ffc800 at 0.4
    assert replayed[40, 40 + 210].max() > 30  # the last badge
    assert replayed[140, 300].argmax() == 1 and replayed[140, 300, 1] > 240


def test_copies_boxes_hold_every_copy_and_a_groups_children_join_theirs():
    spec = scene([{"id": "d", "type": "circle", "r": 10, "x": 160, "y": 90, "stroke": "#ffffff", "strokeWidth": 4,
                   "repeat": {"count": 3, "x": "=(i - 1) * 100", "y": [0, 30, 0]}},
                  {"id": "g", "type": "group", "x": 30, "y": 30, "repeat": {"count": 2, "x": [0, 50]},
                   "children": [{"id": "sq", "type": "rect", "w": 20, "h": 20, "fill": "#ff0000"}]}])
    _, r = _pixels(spec, track=True)
    assert r.boxes["d"]["rect"] == [48.0, 78.0, 272.0, 132.0], r.boxes["d"]  # stroke half its width past the circle
    assert r.boxes["sq"]["rect"] == [20.0, 20.0, 90.0, 40.0], r.boxes["sq"]
    assert check(spec)["ok"]


def test_copies_wipe_on_as_one_and_a_combining_group_takes_each_copys_outline():
    spec = scene([{"id": "bars", "type": "rect", "w": 20, "h": 60, "x": 60, "y": 90, "fill": "#ffffff",
                   "enter": {"name": "wipe", "dur": 1}, "repeat": {"count": 5, "x": "=i * 50"}}])
    img, _ = _pixels(spec, T=0.2)  # the wipe's eased reveal is 0.49: x 50 to 158 of the five's 50 to 270
    assert img[90, 60].min() > 240 and img[90, 110].min() > 240 and img[90, 165].max() == img[90, 210].max() == 0
    union = scene([{"id": "g", "type": "group", "combine": "union", "fill": "#00ff00", "children": [
        {"type": "circle", "r": 10, "x": 160, "y": 90, "repeat": {"count": 3, "x": "=(i - 1) * 60"}}]}])
    img, _ = _pixels(union)
    assert all(img[90, x].argmax() == 1 and img[90, x, 1] > 240 for x in (100, 160, 220))
    assert img[90, 130].max() == 0


def test_copies_that_move_do_not_hold_still_and_their_frames_are_pure():
    moving = {"id": "d", "type": "circle", "r": 5, "x": 160, "y": 90,
              "repeat": {"count": 24, "x": "=cos(i * 0.2618 + t * 2) * 70", "y": "=sin(i * 0.2618 + t * 2) * 50",
                         "fill": "=hsl(i * 15 + t * 90)", "opacity": "=0.3 + 0.7 * ((i + t * 12) % 24) / 24"}}
    still = dict(moving, id="s", repeat={"count": 3, "x": [-10, 0, 10]})
    spec = scene([moving, still,
                  {"id": "g", "type": "group", "x": 20, "y": 20, "repeat": {"count": 3, "x": "=i * 30 + t * 20"},
                   "children": [{"type": "text", "text": "a", "size": 16, "fill": "#ffd166"}]}], duration=1)
    r = Renderer(k.Scene(spec), backend="cpu")
    assert not r._holds(r.s.by_id["d"], 0.0, 0.01, 0.0) and r._holds(r.s.by_id["s"], 0.0, 0.01, 0.0)
    sequential = [r.frame(n).copy() for n in range(0, 20)]
    alone = Renderer(k.Scene(spec), backend="cpu").frame(13)
    assert np.array_equal(sequential[13], alone) and not np.array_equal(sequential[12], sequential[13])


def test_repeat_on_2d_layers_reads_and_checks_its_keys():
    s = k.Scene(scene([{"id": "a", "type": "circle", "copies": {"n": 2, "alpha": [1, 0.5], "colours": ["red", "blue"],
                                                                 "strokeColor": "white"}}]))
    assert s.by_id["a"].raw["repeat"] == {"count": 2, "opacity": [1, 0.5], "fill": ["red", "blue"], "stroke": "white"}
    for layer, words in (({"type": "circle", "repeat": {"count": 2, "z": "=i"}}, "places a model's copies in depth"),
                         ({"type": "image", "src": "x.png", "repeat": {"count": 2, "fill": "red"}}, "unknown key 'fill'"),
                         ({"type": "text", "text": "Hi", "repeat": {"count": 2, "stroke": "#zz0000"}}, "repeat stroke"),
                         ({"type": "rect", "repeat": {"count": 2, "opacity": "half"}}, "repeat opacity is"),
                         ({"type": "captions", "cues": [[0, 1, "Hi"]], "repeat": {"count": 2}}, "unknown key 'repeat'")):
        with pytest.raises(k.SceneError, match=words):
            k.Scene(scene([dict(layer, id="a")]))
    bad = check(scene([{"id": "a", "type": "circle", "repeat": {"count": 4, "x": "=1 / (i - 2)"}}]))
    assert any(i["code"] == "bad-expression" and "repeat x failed for copy 2" in i["message"] for i in bad["issues"])
    many = check(scene([{"id": "a", "type": "circle", "r": 2, "repeat": {"count": 200, "x": "=i - 100",
                                                                         "fill": "=hsl(i * 1.8)"}}]))
    assert not [i for i in many["issues"] if i["code"] == "repeat-colours"]  # a model's limit, not a shape's
