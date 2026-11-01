"""Per-unit paint: unitFill colours each unit of a split text and unitBlur blurs it, by expressions of i, the unit's
number, and n, how many there are; sweep(i, n, p, soft) is 1 for the units a sweep has passed and 0 for the rest."""
import numpy as np
import pytest

import kinescript as k
from kinescript.props import sweep
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 640, "h": 200, "fps": 10, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def text(**kw):
    return dict({"id": "t", "type": "text", "text": "HHHH", "size": 60, "x": 320, "y": 100, "fill": "#ffffff",
                 "split": "chars"}, **kw)


def frames(spec, n=(0,), backend="cpu"):
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    return [r.frame(i)[:, :, :3].astype(int) for i in n]


def ends(img):
    """The mean colour of the lit pixels in the first and the last 20 px of the text."""
    ys, xs = np.nonzero(img.max(axis=2) > 96)
    first, last = xs < xs.min() + 20, xs > xs.max() - 20
    return img[ys[first], xs[first]].mean(axis=0), img[ys[last], xs[last]].mean(axis=0)


def codes(spec):
    try:
        s = k.Scene(spec)
        return [i["code"] for i in s.issues], s
    except k.SceneError as e:
        return [i["code"] for i in e.issues], None


def test_sweep_passes_the_units_in_turn_with_a_soft_front():
    assert sweep(0, 4, 0.0) == 0 and sweep(3, 4, 1.0) == 1  # before the first unit, past the last
    assert sweep(1, 4, 0.5, 0) == 1 and sweep(2, 4, 0.5, 0) == 0  # soft 0: a hard edge at p * n
    edge = [sweep(i, 8, 0.5, 3) for i in range(8)]
    assert edge == sorted(edge, reverse=True) and 0 < edge[4] < 1  # done behind the front, easing across it
    assert sweep(5, 8, 0.5, 3) == pytest.approx(0.0741, abs=1e-3)  # a sixth into the front, eased in
    assert all(sweep(2, 8, a) <= sweep(2, 8, b) for a, b in zip(np.linspace(0, 1, 20), np.linspace(0, 1, 20)[1:]))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_unit_fill_colours_each_unit_by_i_and_n(backend):
    (img,) = frames(scene([text(unitFill="=mix('#ff0000', '#0000ff', i / (n - 1))")]), backend=backend)
    first, last = ends(img)
    assert first[0] > first[2] * 3 and last[2] > last[0] * 3  # unit 0 red, unit n - 1 blue
    (words,) = frames(scene([text(text="H H H H", split="words", unitFill="=mix('#ff0000', '#0000ff', i / (n - 1))")]),
                      backend=backend)
    first, last = ends(words)  # words are units too: n is 4 again
    assert first[0] > first[2] * 3 and last[2] > last[0] * 3


def test_unit_fill_leaves_emphasis_and_takes_over_a_gradient():
    (img,) = frames(scene([text(text="**H**HHH", unitFill="#00ff00")]))
    first, last = ends(img)
    assert first[0] > first[1] and last[1] > last[0] * 3  # the emphasised H keeps the accent
    (img,) = frames(scene([text(fill={"stops": ["#ff0000", "#0000ff"], "angle": 90}, unitFill="#00ff00")]))
    first, last = ends(img)
    assert first[1] > first[0] * 3 and last[1] > last[2] * 3


def test_unit_fill_goes_on_letters_on_a_path_too():
    spec = scene([text(x=0, y=0, textPath={"d": "M 100 120 Q 320 40 540 120", "offset": 0.5, "align": "center"},
                       unitFill="=mix('#ff0000', '#0000ff', i / (n - 1))")])
    (img,) = frames(spec)
    first, last = ends(img)
    assert first[0] > first[2] * 3 and last[2] > last[0] * 3


def test_unit_blur_blurs_each_unit_and_zero_leaves_it_sharp():
    sharp, zero, blurred, half = frames(scene([text()]))[0], *(
        frames(scene([text(unitBlur=v)]))[0] for v in ("=i * 0", 6, "=6 * (i >= n / 2)"))
    assert np.array_equal(sharp, zero)
    assert blurred.max() < sharp.max() - 40  # spread out: no pixel stays white
    lit = lambda img: (img.max(axis=2) > 8).sum()  # noqa: E731
    assert lit(blurred) > lit(sharp) * 1.3
    left, right = half[:, :290], half[:, 330:]  # units 0 and 1, 2 and 3, past the blur's reach
    assert np.array_equal(left, sharp[:, :290]) and right.max() < sharp.max() - 40


def test_a_whole_text_is_one_unit():
    (img,) = frames(scene([text(split=None, unitOpacity="=1 / n")]))
    assert img.max() > 240  # n is 1


def test_n_belongs_to_unit_keys_and_unit_fill_is_a_colour():
    with pytest.raises(k.SceneError, match="unknown name 'n'"):
        k.Scene(scene([text(x="=n * 10")]))
    with pytest.raises(k.SceneError, match="unitFill is one colour for each unit, not a gradient"):
        k.Scene(scene([text(unitFill={"stops": ["#ff0000", "#0000ff"]})]))
    assert "bad-value" in codes(scene([text(unitBlur=-2)]))[0]
    assert "bad-colour" in [i["code"] for i in check(scene([text(unitFill="=i * 2")]))["issues"]]


def test_unit_paint_is_a_function_of_the_frame_alone():
    spec = scene([text(unitFill="=mix('#f2f0e8', '#ff633f', sweep(i, n, p, 2))", unitBlur="=4 * (1 - sweep(i, n, p))")])
    a, b, c = frames(spec, (12, 3, 12))
    (d,) = frames(spec, (12,))
    assert np.array_equal(a, c) and np.array_equal(a, d) and not np.array_equal(a, b)


def test_checks_try_a_unit_expression_on_the_first_unit_and_the_last():
    assert "bad-expression" in codes(scene([text(text="H", unitY="=10 / (n - 1)")]))[0]  # one unit: n - 1 is 0
    assert codes(scene([text(unitY="=10 / (n - 1)")]))[1] is not None  # four
    assert "bad-expression" in codes(scene([text(unitY="=10 / (3 - i)")]))[0]  # the last of four is unit 3
    assert codes(scene([text(text="a few words", split="words", unitY="=10 / (n - 2)")]))[1] is not None  # three
    assert "bad-expression" in codes(scene([text(text="H H H", unitY="=10 / (n - 3)")]))[0]  # spaces are no units
    late = text(unitY="=1 / ((i - 3) * 100 + t * 10 - 5)")  # only the last unit, only at 0.5 s
    assert codes(scene([late]))[1] is not None
    assert "bad-expression" in [i["code"] for i in check(scene([late]))["issues"]]
    counter = {"id": "c", "type": "counter", "value": [[0, 5], [2, 50000, "linear"]], "split": "chars", "x": 320,
               "y": 100, "unitY": "=10 / (n - 5)"}
    assert codes(scene([counter]))[1] is not None  # 5, 25,002 and 50,000 at the times loading tries
    assert "bad-expression" in [i["code"] for i in check(scene([counter]))["issues"]]  # 1,000 to 9,999 on the way
