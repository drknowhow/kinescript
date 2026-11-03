"""Text on a path: textPath lays a text's letters along path data or a path layer, each turned to the path where its
middle is, from offset (a share of the path's length) with align saying which part of the text stands there."""
import numpy as np
import pytest

import kinescript as k
from kinescript.render import Renderer
from kinescript.verify import check

LINE = "M 0 200 L 640 200"


def scene(layers, **canvas):
    return {"canvas": dict({"w": 640, "h": 360, "fps": 10, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def text(on, **kw):
    return dict({"id": "t", "type": "text", "text": "HELLO PATH", "size": 32, "fill": "#ffffff", "textPath": on}, **kw)


def frames(spec, n=(0,), backend="cpu"):
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    return [r.frame(i)[:, :, :3].astype(int) for i in n]


def ink(img, box=None):
    """(left, top, right, bottom) of the white pixels, in box (left, top, right, bottom) when given."""
    lit = img.min(axis=2) > 96
    if box is not None:
        mask = np.zeros_like(lit)
        mask[box[1]:box[3], box[0]:box[2]] = True
        lit &= mask
    ys, xs = np.nonzero(lit)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None


def codes(spec):
    try:
        s = k.Scene(spec)
        return [i["code"] for i in s.issues], s
    except k.SceneError as e:
        return [i["code"] for i in e.issues], None


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_letters_stand_on_the_path_turned_to_it(backend):
    (img,) = frames(scene([text("M 100 200 L 600 200")]), backend=backend)
    left, top, right, bottom = ink(img)
    assert abs(left - 100) <= 4 and 196 <= bottom <= 201 and top < 180  # the baseline on the path, capitals above it
    (img,) = frames(scene([text("M 300 20 L 300 340")]), backend=backend)
    left, top, right, bottom = ink(img)
    assert 298 <= left <= 302 and right < 330 and abs(top - 20) <= 4  # going down, the letters' tops face right
    assert bottom - top > (right - left) * 4


def test_a_letter_turns_with_the_path_where_its_middle_is():
    corner = "M 0 200 L 300 200 L 300 360"  # 460 px, turning down at 300
    (img,) = frames(scene([text({"d": corner, "offset": 270 / 460}, text="H", size=40)]))
    assert ink(img)[3] <= 201  # its middle before the corner: upright
    (img,) = frames(scene([text({"d": corner, "offset": 290 / 460}, text="H", size=40)]))
    left, top, right, bottom = ink(img)
    assert left >= 298 and bottom >= 210  # its left edge before the corner, its middle past it: turned down


def test_offset_and_align_say_where_the_text_stands():
    def at(offset, align):
        (img,) = frames(scene([text({"d": LINE, "offset": offset, "align": align})]))
        return ink(img)
    left, _, right, _ = at(0.5, "center")
    assert abs((left + right) / 2 - 320) <= 3
    assert abs(at(1, "end")[2] - 640) <= 4
    assert abs(at(0.25, "start")[0] - 160) <= 4


def test_offset_animates_the_text_along_the_path():
    spec = scene([text({"d": LINE, "offset": [[0, 0], [1, 0.5, "linear"]]})])
    a, b, c = frames(spec, (0, 5, 10))
    assert ink(a)[0] <= 4 and abs(ink(b)[0] - 160) <= 4 and abs(ink(c)[0] - 320) <= 4


def test_an_open_path_drops_the_letters_past_its_end_and_check_says_so():
    short = scene([text("M 100 200 L 200 200")])
    (img,) = frames(short)
    assert ink(img)[2] <= 206  # HELLO PATH is about 180 px: what is past 200 is left out
    assert "text-off-path" in [i["code"] for i in check(short)["issues"]]
    assert "text-off-path" not in [i["code"] for i in check(scene([text(LINE)]))["issues"]]
    closed = scene([text("M 100 200 L 140 200 L 140 160 L 100 160 Z")])  # 160 px round, and it goes round again
    assert "text-off-path" not in [i["code"] for i in check(closed)["issues"]]


def test_check_reads_the_letters_not_the_box_round_the_curve():
    ring = text({"d": "M 220 180 A 100 100 0 1 1 420 180 A 100 100 0 1 1 220 180 Z"},
                text="ROUND AND ROUND AND ROUND AND ROUND ", size=20)
    label = {"id": "mid", "type": "text", "text": "42", "size": 40, "x": 320, "y": 180}
    assert "text-overlap" not in [i["code"] for i in check(scene([ring, label]))["issues"]]  # inside the ring
    across = dict(label, text="ACROSS THE RING", y=80)  # over its letters at the top
    assert "text-overlap" in [i["code"] for i in check(scene([ring, across]))["issues"]]


def test_a_closed_path_takes_the_letters_round_past_its_start():
    square = "M 200 100 L 440 100 L 440 260 L 200 260 L 200 100"  # 800 px round
    (opened,) = frames(scene([text({"d": square, "offset": 0.95})]))
    (closed,) = frames(scene([text({"d": square + " Z", "offset": 0.95})]))
    for img in (opened, closed):
        assert ink(img, (150, 100, 200, 160)) is not None  # going up the left side, the tops face left
    assert ink(opened, (230, 40, 440, 100)) is None
    assert ink(closed, (230, 40, 440, 100)) is not None  # and on along the top


def test_a_path_layer_is_taken_where_it_is_drawn():
    def with_route(**kw):
        route = dict({"id": "p", "type": "path", "d": "M 0 0 L 400 0", "stroke": "#000000", "x": 100, "y": 200}, **kw)
        (img,) = frames(scene([route, text("#p")]))
        return ink(img)
    left, top, right, bottom = with_route()
    assert abs(left - 100) <= 4 and 196 <= bottom <= 201
    assert abs(with_route(x=150)[0] - 150) <= 4
    route = {"id": "p", "type": "path", "d": "M 0 0 L 400 0", "stroke": "#000000", "x": [[0, 100], [1, 200, "linear"]],
             "y": 200}
    a, b = frames(scene([route, text("#p")]), (0, 10))
    assert abs(ink(a)[0] - 100) <= 4 and abs(ink(b)[0] - 200) <= 4  # a moving path layer takes its text with it
    left, top, right, bottom = with_route(rotate=90)
    assert bottom - top > (right - left) * 4
    line = {"id": "p", "type": "line", "points": [[100, 200], [500, 200]], "stroke": "#000000"}
    (img,) = frames(scene([line, text({"d": "#p", "offset": 0.5, "align": "center"})]))
    left, _, right, _ = ink(img)
    assert abs((left + right) / 2 - 300) <= 3
    (img,) = frames(scene([line, text("#p", x=40, y=-20)]))  # the text's own x and y still move it
    left, _, _, bottom = ink(img)
    assert abs(left - 140) <= 4 and 176 <= bottom <= 181
    (img,) = frames(scene([dict(line, type="path", d="M 0 0 L 400 0", x=320, y=200, anchor="center"), text("#p")]))
    assert abs(ink(img)[0] - 120) <= 4  # drawn from 120 to 520, as its anchor puts it


def test_a_counter_rides_a_path_too():
    (img,) = frames(scene([{"id": "c", "type": "counter", "value": 1234, "size": 32, "fill": "#ffffff",
                            "textPath": "M 300 20 L 300 340"}]))
    left, top, right, bottom = ink(img)
    assert 290 <= left <= 302 and bottom - top > (right - left) * 2  # the comma's tail goes below the line


def test_unit_moves_go_along_and_across_the_path():
    down = "M 300 20 L 300 340"
    (img,) = frames(scene([text(down, unitX=20, unitY=-30)]))
    left, top, right, bottom = ink(img)
    assert abs(left - 330) <= 4 and abs(top - 40) <= 4 and bottom - top > (right - left) * 4
    (img,) = frames(scene([text(down, unitOpacity=0.5)]))
    assert 120 <= img.max() <= 136


def test_fill_gradient_and_stroke_go_on_the_letters_where_they_stand():
    (img,) = frames(scene([text(LINE, size=60, fill={"stops": ["#ff0000", "#0000ff"], "angle": 90})]))
    ys, xs = np.nonzero(img.max(axis=2) > 96)
    first = img[ys[xs < xs.min() + 20], xs[xs < xs.min() + 20]].mean(axis=0)
    last = img[ys[xs > xs.max() - 20], xs[xs > xs.max() - 20]].mean(axis=0)
    assert first[0] > first[2] * 3 and last[2] > last[0] * 3  # red at the text's start, blue at its end
    (img,) = frames(scene([text("M 300 20 L 300 340", fill={"stops": ["#ff0000", "#0000ff"], "angle": 0})]))
    ys, xs = np.nonzero(img.max(axis=2) > 96)
    middle = (ys > 80) & (ys < 140)  # the letters run from 20 to 200: halfway between red and blue
    r, _, b = img[ys[middle], xs[middle]].mean(axis=0)
    assert abs(r - b) < 40 and r > 80
    (img,) = frames(scene([text(LINE, fill="#000000", stroke="#ffffff", strokeWidth=2)]))
    assert ink(img) is not None


def test_split_letters_still_enter_and_land_where_the_whole_text_stands():
    curve = "M 60 260 C 200 60 440 60 580 260"
    still = scene([text({"d": curve, "offset": 0.5, "align": "center"})])
    split = scene([text({"d": curve, "offset": 0.5, "align": "center"}, split="chars",
                        enter={"name": "rise", "dur": 0.5, "stagger": 0.02})])
    (whole,) = frames(still, (15,))
    moving, landed = frames(split, (2, 15))
    assert np.abs(landed - whole).max() <= 2
    assert np.abs(moving - whole).max() > 100


def test_path_text_is_a_function_of_the_frame_alone():
    spec = scene([text({"d": "M 60 260 C 200 60 440 60 580 260", "offset": [[0, 0], [2, 0.6]]}, split="words",
                       enter={"name": "pop", "stagger": 0.1})])
    a, b, c = frames(spec, (12, 3, 12))
    (d,) = frames(spec, (12,))
    assert np.array_equal(a, c) and np.array_equal(a, d) and not np.array_equal(a, b)


def test_a_box_goes_round_the_letters_where_they_stand():
    spec = scene([text({"d": "M 300 20 L 300 340"}, box={"fill": "#ff0000", "pad": 6})])
    (img,) = frames(spec)
    red = (img[:, :, 0] > 200) & (img[:, :, 1] < 60)
    ys, xs = np.nonzero(red)
    left, top, right, bottom = ink(img)
    assert xs.min() <= left - 4 and xs.max() >= right + 4 and ys.min() <= top - 4 and ys.max() >= bottom + 4
    assert xs.max() - xs.min() < 60  # round the turned letters, not the text laid flat


def test_schema_reads_the_short_forms_and_names_bad_values():
    got, s = codes(scene([text("M 0 200 L 640 200")]))
    assert s is not None and s.by_id["t"].raw["textPath"] == {"d": "M 0 200 L 640 200"}
    got, s = codes(scene([text({"path": LINE, "startOffset": 0.5, "anchor": "middle"})]))
    assert s is not None and s.by_id["t"].raw["textPath"] == {"d": LINE, "offset": 0.5, "align": "center"}
    assert "renamed-value" in got
    assert "bad-value" in codes(scene([text({"d": LINE, "align": "diagonal"})]))[0]
    assert "missing-key" in codes(scene([text({"offset": 0.5})]))[0]
    assert codes(scene([text({"d": "M 0 0 Q 5"})]))[1] is None
    assert "bad-value" in codes(scene([text(42)]))[0]
    got, s = codes(scene([text(LINE, anchor="center")]))
    assert s is not None and "ignored-key" in got


def test_a_route_must_be_a_path_in_the_same_group():
    assert "unknown-route" in codes(scene([text("#nope")]))[0]
    other = {"id": "o", "type": "text", "text": "x"}
    assert "bad-route" in codes(scene([other, text("#o")]))[0]
    assert "bad-route" in codes(scene([text("#t")]))[0]
    grouped = {"id": "g", "type": "group", "layers": [{"id": "p", "type": "path", "d": LINE, "stroke": "#000"}]}
    assert "bad-route" in codes(scene([grouped, text("#p")]))[0]
    try:
        k.Scene(scene([text("#nope")]))
    except k.SceneError as e:
        assert "textPath '#nope'" in str(e)
