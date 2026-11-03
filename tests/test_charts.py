"""The built-in charts barChart, lineChart and pieChart: the data they take, the layers they build and when those
layers come on."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import kinescript as k
from kinescript import components as C
from kinescript.docs import reference
from kinescript.ease import EASES
from kinescript.render import Renderer

ROOT = Path(__file__).resolve().parent.parent
PAPER, ACCENT, MUTED = C.THEME["paper"], C.THEME["accent"], C.THEME["muted"]


def build(**raw):
    raw.setdefault("id", "c")
    return C.expand(raw, {}, C.THEME)


def kids(**raw):
    return {L["id"]: L for L in build(**raw)["children"]}


def frame(layer, t, backend="cpu", w=400, h=300):
    spec = {"canvas": {"w": w, "h": h, "fps": 20, "duration": 3, "bg": "#000000"}, "layers": [layer]}
    try:
        r = Renderer(k.Scene(spec), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    return r.frame(round(t * 20))[:, :, :3].astype(int).copy()


def near(px, hexcol, tol=24):
    col = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    return bool(np.abs(np.asarray(px) - col).max() <= tol)


@pytest.mark.parametrize("data, labels", [([3, 5], ["", ""]), ([["A", 3], ["B", 5]], ["A", "B"]),
                                          ([{"label": "A", "value": 3}, {"label": "B", "value": 5}], ["A", "B"])])
def test_data_is_numbers_pairs_or_objects(data, labels):
    assert C._series(C.Params(id="c", data=data), "barChart") == (labels, [3.0, 5.0])


def test_labels_name_the_values_in_turn():
    got = C._series(C.Params(id="c", data=[["A", 1], 2, 3], labels=["x", "y"]), "barChart")
    assert got == (["x", "y", ""], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="labels is 'Q1'; it is a list of text"):
        build(use="barChart", data=[1], labels="Q1")


@pytest.mark.parametrize("name", ["barChart", "lineChart", "pieChart"])
@pytest.mark.parametrize("data", [None, [], ["a", 1], [True, 1], [float("nan"), 1], [[1, 2, 3], 1], {"a": 1}])
def test_bad_data_names_the_layer_and_the_shapes_it_takes(name, data):
    with pytest.raises(ValueError, match=rf"layer 'c': {name} data .*\[label, number\] pairs"):
        build(use=name, data=data)


def test_only_a_line_goes_below_zero_and_its_axis_stays_at_zero():
    for name in ("barChart", "pieChart"):
        with pytest.raises(ValueError, match="each 0 or more"):
            build(use=name, data=[2, -1])
    line = kids(use="lineChart", data=[2, -1], h=300)
    assert line["c-axis"]["points"] == [[-600, 50], [600, 50]]  # 0 is a third of the way up from -1 to 2
    assert line["c-line"]["points"] == [[-600, -150], [600, 150]]
    with pytest.raises(ValueError, match="a line needs 2 or more"):
        build(use="lineChart", data=[5])


def test_bars_share_the_width_and_grow_to_their_share_of_the_top():
    out = build(use="barChart", data=[["A", 25], ["B", 50], ["C", 100]], w=600, h=400, gap=0.5, max=200, x=10, y=20)
    assert (out["type"], out["anchor"], out["x"], out["y"], out["w"], out["h"]) == ("group", "center", 10, 20, 600, 400)
    b = {L["id"]: L for L in out["children"]}
    assert [b[f"c-bar{i}"]["x"] for i in (1, 2, 3)] == [-200, 0, 200]
    assert [b[f"c-bar{i}"]["h"]["to"] for i in (1, 2, 3)] == [50, 100, 200]
    bar = b["c-bar1"]
    assert (bar["w"], bar["y"], bar["anchor"], bar["h"]["from"]) == (100, 200, "bottom", 0)
    assert b["c-axis"]["points"] == [[-300, 200], [300, 200]] and b["c-axis"]["stroke"] == MUTED
    assert [b[f"c-label{i}"]["text"] for i in (1, 2, 3)] == ["A", "B", "C"]
    assert (b["c-label2"]["x"], b["c-label2"]["y"], b["c-label2"]["anchor"]) == (0, 220, "top")
    tall = kids(use="barChart", data=[25, 50, 100], h=400)
    assert [tall[f"c-bar{i}"]["h"]["to"] for i in (1, 2, 3)] == [100, 200, 400]  # the top is the largest value
    over = kids(use="barChart", data=[25, 100], h=400, max=50)
    assert [over[f"c-bar{i}"]["h"]["to"] for i in (1, 2)] == [200, 400]  # past max stops at the top
    assert "c-label1" not in tall
    assert [kids(use="barChart", data=[1, 2], w=400, gap=g)["c-bar1"]["w"] for g in (2, -1)] == [20, 200]


def test_bars_grow_in_turn_with_their_values_riding_on_top():
    b = kids(use="barChart", data=[10, 20, 40], h=400, grow=0.5, stagger=0.25, format="{:.1f}%", size=24)
    assert [b[f"c-bar{i}"]["at"] for i in (1, 2, 3)] == [0, 0.25, 0.5]
    v = b["c-value2"]
    assert b["c-bar2"]["h"] == {"from": 0, "to": 200, "dur": 0.5, "ease": "outQuad"}
    assert (v["type"], v["at"], v["x"], v["format"], v["size"]) == ("counter", 0.25, b["c-bar2"]["x"], "{:.1f}%", 32)
    assert v["y"] == {"from": 160, "to": -40, "dur": 0.5, "ease": "outQuad"}  # 40 px above the bar all the way
    assert v["value"] == {"from": 0, "to": 20, "dur": 0.5, "ease": "outQuad"}
    assert kids(use="barChart", data=[1], valueSize=50)["c-value1"]["size"] == 50
    bare = kids(use="barChart", data=[1, 2], values=False)
    assert sorted(bare) == ["c-axis", "c-bar1", "c-bar2"]


def test_highlight_picks_bars_by_number_from_1_or_label():
    def fills(**raw):
        b = kids(use="barChart", data=[["A", 1], ["B", 2], ["C", 3]], **raw)
        return [b[f"c-bar{i}"]["fill"] for i in (1, 2, 3)]
    assert fills() == [PAPER, PAPER, PAPER]
    assert fills(highlight=2) == [PAPER, ACCENT, PAPER]
    assert fills(highlight="C") == [PAPER, PAPER, ACCENT]
    assert fills(highlight=0) == [PAPER, PAPER, PAPER]
    assert fills(highlight=[1, "C"], accent="#123456", color="#abcdef") == ["#123456", "#abcdef", "#123456"]
    assert fills(colors=["#111111", "#222222"]) == ["#111111", "#222222", "#111111"]


def test_line_runs_through_its_points_across_the_width():
    b = kids(use="lineChart", data=[0, 5, 10], w=400, h=200)
    assert b["c-line"]["points"] == [[-200, 100], [0, 0], [200, -100]]
    assert (b["c-line"]["stroke"], b["c-line"]["strokeWidth"]) == (ACCENT, 6)
    b = kids(use="lineChart", data=[0, 5, 10], w=400, h=200, min=-10, max=20, color="#00ff00", strokeWidth=3)
    assert b["c-line"]["points"] == [[-200, 33.333], [0, 0], [200, -33.333]]
    assert b["c-axis"]["points"][0][1] == 33.333 and (b["c-line"]["stroke"], b["c-line"]["strokeWidth"]) == \
        ("#00ff00", 3)
    assert kids(use="lineChart", data=[10, 20], h=200)["c-line"]["points"] == [[-600, 0], [600, -100]]  # from 0
    flat = kids(use="lineChart", data=[3, 3], h=200, min=3)
    assert flat["c-line"]["points"] == [[-600, 100], [600, 100]]


def test_each_dot_and_value_pop_on_as_the_line_reaches_them():
    b = kids(use="lineChart", data=[["Mon", 0], ["Tue", 30], ["Wed", 40]], w=300, h=300, grow=2, format="{:.1f}")
    pts = b["c-line"]["points"]
    seg = [math.dist(p, q) for p, q in zip(pts, pts[1:])]
    assert b["c-line"]["at"] == 0.2 and b["c-line"]["enter"] == {"name": "draw", "dur": 2, "ease": "inOutSine"}
    for i, f in enumerate([0, seg[0] / sum(seg), 1]):
        dot, val = b[f"c-dot{i + 1}"], b[f"c-value{i + 1}"]
        assert EASES["inOutSine"]((dot["at"] - 0.2) / 2) == pytest.approx(f, abs=1e-3)
        assert (dot["x"], dot["y"], dot["r"], dot["fill"]) == (pts[i][0], pts[i][1], 9, ACCENT)
        assert (val["at"], val["y"], val["anchor"]) == (dot["at"], pts[i][1] - 22, "bottom")
    assert [b[f"c-value{i}"]["text"] for i in (1, 2, 3)] == ["0.0", "30.0", "40.0"]
    assert [b[f"c-label{i}"]["text"] for i in (1, 2, 3)] == ["Mon", "Tue", "Wed"]
    assert b["c-label3"]["y"] == 170 and b["c-label3"]["x"] == 150
    assert C._sine_at(0.5) == pytest.approx(0.5) and C._sine_at(-1) == 0 and C._sine_at(2) == 1
    bare = kids(use="lineChart", data=[1, 2], dots=False, values=False, r=4)
    assert sorted(bare) == ["c-axis", "c-line"]
    assert kids(use="lineChart", data=[1, 2], r=4)["c-dot1"]["r"] == 4


@pytest.mark.parametrize("name", ["barChart", "lineChart", "pieChart"])
@pytest.mark.parametrize("fmt", ["{:q}", 5])
def test_a_bad_format_is_reported(name, fmt):
    with pytest.raises(ValueError, match=r"layer 'c': format is .*where the number goes"):
        build(use=name, data=[1, 2], format=fmt)


def test_arc_runs_clockwise_and_a_whole_ring_in_two_halves():
    assert C._arc(10, 0, 90) == "M10.000 0.000 A10.000 10.000 0 0 1 0.000 10.000"
    assert C._arc(10, -90, 135).split()[5] == "1"  # the long way round
    ring = C._arc(10, -90, 270)
    assert ring.count("A") == 2 and ring.startswith("M0.000 -10.000") and ring.endswith("0.000 -10.000")


def test_slices_sweep_round_from_twelve_and_tuck_under_the_next():
    b = kids(use="pieChart", data=[["A", 1], ["B", 1], ["C", 2]], r=100, inner=0.5, grow=1)
    assert [b[f"c-slice{i}"]["d"] for i in (1, 2, 3)] == [C._arc(75, -90, 0.6), C._arc(75, 0, 90.6),
                                                           C._arc(75, 90, 270)]
    assert {(b[f"c-slice{i}"]["strokeWidth"], b[f"c-slice{i}"]["cap"]) for i in (1, 2, 3)} == {(50, "butt")}
    assert [b[f"c-slice{i}"]["stroke"] for i in (1, 2, 3)] == C.CHART_COLORS[:3]
    third = math.acos(0.5) / math.pi  # when inOutSine is a quarter of the way
    assert [b[f"c-slice{i}"]["at"] for i in (1, 2, 3)] == [0, round(third, 4), 0.5]
    assert [b[f"c-slice{i}"]["enter"] for i in (1, 2, 3)] == [
        {"name": "draw", "dur": round(third, 4), "ease": "linear"},
        {"name": "draw", "dur": round(0.5 - third, 4), "ease": "linear"},
        {"name": "draw", "dur": 0.5, "ease": "linear"}]
    assert [b[f"c-label{i}"]["text"] for i in (1, 2, 3)] == ["A  25%", "B  25%", "C  50%"]
    assert [b[f"c-label{i}"]["at"] for i in (1, 2, 3)] == [round(third, 4), 0.5, 1]
    assert (b["c-label1"]["x"], b["c-label1"]["y"]) == (round(128 * math.cos(math.radians(-45)), 3),
                                                         round(128 * math.sin(math.radians(-45)), 3))
    assert [b[f"c-label{i}"]["anchor"] for i in (1, 2, 3)] == ["left", "left", "right"]
    out = build(use="pieChart", data=[1], r=100, x=5, y=6)
    assert (out["x"], out["y"], out["w"], out["h"], out["anchor"]) == (5, 6, 200, 200, "center")


def test_pie_labels_sit_outside_on_their_side():
    b = kids(use="pieChart", data=[1, 2, 1], format="{:.1f} pc")
    assert [b[f"c-label{i}"]["anchor"] for i in (1, 2, 3)] == ["left", "top", "right"]
    assert b["c-label2"]["text"] == "50.0 pc"
    assert kids(use="pieChart", data=[98, 2])["c-label2"]["anchor"] == "bottom"


def test_a_pie_skips_empty_slices_and_needs_a_total():
    b = kids(use="pieChart", data=[0, 1], colors=["#111111", "#222222"])
    assert "c-slice1" not in b and b["c-slice2"]["d"] == C._arc(120, -90, 270)
    assert b["c-slice2"]["stroke"] == "#222222" and b["c-slice2"]["strokeWidth"] == 240
    with pytest.raises(ValueError, match="adds up to 0"):
        build(use="pieChart", data=[0, 0])
    assert kids(use="pieChart", data=[1], inner=2, r=100)["c-slice1"]["strokeWidth"] == 5
    assert kids(use="pieChart", data=[1], inner=-1, r=100)["c-slice1"]["strokeWidth"] == 100


def test_parameters_are_checked_by_name_and_type():
    with pytest.raises(ValueError, match="no parameter 'hightlight'. Did you mean 'highlight'"):
        build(use="barChart", data=[1], hightlight=1)
    with pytest.raises(ValueError, match="barChart w is 'wide'; it takes a number"):
        build(use="barChart", data=[1], w="wide")
    with pytest.raises(ValueError, match="pieChart inner is True; it takes a number"):
        build(use="pieChart", data=[1], inner=True)
    out = build(use="barChart", data=[1], dur=4, at=1)
    assert (out["dur"], out["at"]) == (4, 1)  # the layer's own time, not the bars'
    assert kids(use="barChart", data=[1], dur=4)["c-bar1"]["h"]["dur"] == 1


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_bars_draw_from_the_baseline_up(backend):
    layer = {"use": "barChart", "id": "c", "x": 200, "y": 150, "w": 300, "h": 200, "data": [1, 2], "highlight": 2,
             "values": False, "grow": 0.5}
    mid, end = frame(layer, 0.25, backend), frame(layer, 2, backend)
    assert near(end[200, 125], PAPER) and near(end[140, 125], "#000000")  # bar 1 is 100 px tall on y 250
    assert near(end[60, 275], ACCENT) and near(end[40, 275], "#000000")
    assert near(end[250, 60], MUTED, 40)
    assert near(mid[245, 125], PAPER) and near(mid[160, 125], "#000000")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_slices_meet_without_the_background_between_them(backend):
    layer = {"use": "pieChart", "id": "c", "x": 150, "y": 150, "r": 100, "inner": 0.4, "data": [1, 2, 3],
             "colors": ["#ff0000", "#0000ff", "#00ff00"], "format": " "}
    img = frame(layer, 2.9, backend, 300, 300)
    for deg, col in ((-60, "#ff0000"), (30, "#0000ff"), (180, "#00ff00")):
        a = math.radians(deg)
        assert near(img[round(150 + 75 * math.sin(a)), round(150 + 75 * math.cos(a))], col)
    a = math.radians(-30)  # where the first slice meets the second, across pixels
    seam = [img[int(150 + d * math.sin(a)), int(150 + d * math.cos(a))].sum() for d in range(48, 94)]
    assert min(seam) >= 235, seam


def test_charts_example_checks_clean_and_docs_list_the_charts():
    sc = k.load(str(ROOT / "examples/charts.json"))
    assert [i for i in sc.issues if i["level"] == "error"] == []
    sc.close()
    uses = {L.get("use") for L in json.loads((ROOT / "examples/charts.json").read_text())["layers"]}
    assert {"barChart", "lineChart", "pieChart"} <= uses
    ref = reference()
    part = ref[ref.index("Built in (parameters):"):ref.index("x and y place the component")]
    assert all(f"    {n}" in part for n in ("barChart", "lineChart", "pieChart"))
    assert max(len(line) for line in part.splitlines()) <= 108
    assert '{"use": "barChart", "id": "sales"' in ref
