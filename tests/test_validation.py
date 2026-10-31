"""Scene validation and its messages: a wrong value is a check issue that names its layer and field and says what
is accepted, never a Python traceback while loading or drawing, and never silently dropped."""
import pytest

import kinescript as k
from kinescript import color as C
from kinescript.verify import check

RECT = {"id": "r", "type": "rect", "w": 40, "h": 40}


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 1}, **canvas), "layers": layers}


def load(spec):
    """(issues, scene or None): what loading says, whether or not it succeeds."""
    try:
        s = k.Scene(spec)
        return s.issues, s
    except k.SceneError as e:
        return e.issues, None


def errors(spec):
    issues, s = load(spec)
    return [i for i in issues if i["level"] == "error"], s


def one_error(spec, layer=None):
    errs, s = errors(spec)
    assert s is None and errs, "expected an error"
    if layer is not None:
        assert errs[0]["layer"] == layer, errs
    return errs[0]


def warnings(spec, code):
    issues, s = load(spec)
    assert s is not None, issues
    return [i for i in issues if i["code"] == code]


def rect(**kw):
    return dict(RECT, **kw)


# ---- A9: values that used to crash while loading or drawing ----------------------------------------------------

@pytest.mark.parametrize("ease, words", [
    (5, "ease 5 is not a name"), (["outCubic"], "is not a name"), ("spring(0, 0, 0)", "above 0"),
    ("bezier(1, 2)", "bezier takes 4"), ("steps()", "whole number"), ("bezier(2, 0, 0.5, 1)", "from 0 to 1"),
])
def test_a_bad_ease_is_an_error_naming_layer_and_field(ease, words):
    e = one_error(scene([rect(x={"from": 0, "to": 100, "ease": ease})]), "r")
    assert e["message"].startswith("x") and words in e["message"], e


def test_a_bad_ease_in_a_keyframe_or_motion_is_an_error():
    assert "ease" in one_error(scene([rect(x=[[0, 0], [1, 100, "spring(0, 0, 0)"]])]), "r")["message"]
    assert "enter" in one_error(scene([rect(enter={"name": "rise", "ease": [1]})]), "r")["message"]


@pytest.mark.parametrize("opt", [{"dist": "far"}, {"amount": "lots"}, {"from": [1]}, {"dur": "slow"},
                                 {"stagger": -1}, {"dur": float("nan")}])
def test_motion_options_are_numbers(opt):
    e = one_error(scene([rect(enter=dict({"name": "rise"}, **opt))]), "r")
    assert e["message"].startswith("enter") and next(iter(opt)) in e["message"], e


@pytest.mark.parametrize("fx", [5, "warm", {"brightness": "x"}, {"brightness": float("nan")}])
def test_fx_is_an_object_of_numbers(fx):
    assert "fx" in one_error(scene([rect(fx=fx)]), "r")["message"]


@pytest.mark.parametrize("bad", [{"font": 5}, {"font": ["Arial"]}, {"box": {"pad": []}}, {"box": {"pad": "big"}},
                                 {"box": {"radius": "round"}}, {"anchor": ["center"]}, {"anchor": {}},
                                 {"blend": ["screen"]}, {"text": ["a", "b"]}])
def test_wrong_types_on_text_are_errors_not_crashes(bad):
    one_error(scene([dict({"id": "t", "type": "text", "text": "Hi"}, **bad)]), "t")


def test_a_lone_surrogate_in_text_is_an_error():
    assert one_error(scene([{"id": "t", "type": "text", "text": "a\ud800b"}]), "t")["code"] == "bad-text"


@pytest.mark.parametrize("timing", [{"at": 10 ** 400}, {"dur": 10 ** 400}, {"dur": [1]}, {"dur": {}},
                                    {"at": [0]}, {"until": {"x": 1}}])
def test_impossible_times_are_errors(timing):
    e = one_error(scene([rect(**timing)]), "r")
    assert next(iter(timing)) in e["message"] and len(e["message"]) < 300, e


def test_an_expression_too_big_for_a_float_is_an_error():
    errs, s = errors(scene([rect(x="=10**400")]))
    assert s is None and errs[0]["layer"] == "r" and "finite" in errs[0]["message"], errs
    r = check(scene([rect(x="=1e999")]))  # inf, not an exception: still an error
    assert not r["ok"] and r["issues"][0]["layer"] == "r", r["issues"]


@pytest.mark.parametrize("raw", [{"type": 5}, {"type": ["text"]}, {"use": ["lowerThird"]}])
def test_a_type_or_component_that_is_not_a_name_is_an_error(raw):
    assert one_error(scene([dict({"id": "x"}, **raw)]), "x")["code"] in ("bad-type", "bad-component")


def test_tween_dur_null_means_the_default():
    _, s = load(scene([rect(opacity={"from": 0, "to": 1, "dur": None, "delay": None})]))
    assert s is not None and s.by_id["r"].get("opacity", 0.6) == pytest.approx(1)


def test_an_out_of_range_colour_keyframe_mixes_without_complex_numbers():
    issues, s = load(scene([rect(fill=[[0, "#ff0000"], [1, "rgb(0, -50, 0)"]])]))
    assert s is not None and any(i["code"] == "colour-clamped" for i in issues), issues
    c = s.by_id["r"].get("fill", 0.5)
    assert all(isinstance(x, float) and 0 <= x <= 1 for x in c), c


# ---- A14: position ------------------------------------------------------------------------------------------------

def test_position_with_three_numbers_is_x_y_and_z():
    _, s = load(scene([rect(position=[10, 20, 30])]))
    assert s is not None and (s.by_id["r"].raw["x"], s.by_id["r"].raw["y"], s.by_id["r"].raw["z"]) == (10, 20, 30)


@pytest.mark.parametrize("pos", ["10,20", [1], {"left": 4}])
def test_a_position_that_cannot_be_read_is_an_error_not_a_note(pos):
    e = one_error(scene([rect(position=pos)]), "r")
    assert e["message"].startswith("position is") and "[x, y]" in e["message"], e


# ---- A15: colour channels -----------------------------------------------------------------------------------------

def test_colour_channels_are_clamped_with_a_warning():
    assert C.parse("rgb(300, 0, 0)") == (1.0, 0.0, 0.0, 1.0)
    assert C.to_argb_int((1.2, -0.1, 0.5, 1.0)) == 0xFFFF0080
    w = warnings(scene([rect(fill="rgb(300, 0, 0)")]), "colour-clamped")
    assert w and w[0]["layer"] == "r" and "rgba(255, 0, 0, 1)" in w[0]["message"], w


def test_extra_colour_channels_are_an_error():
    e = one_error(scene([rect(fill="rgb(1, 2, 3, 4, 5)")]), "r")
    assert e["code"] == "bad-colour" and "has 5 values" in e["message"], e


# ---- A16: component placeholders ----------------------------------------------------------------------------------

def card(text, inst, params=None):
    comp = {"layers": [{"id": "t", "type": "text", "text": text, "dur": 1}]}
    if params is not None:
        comp["params"] = params
    return dict(scene([dict({"use": "card", "id": "c1"}, **inst)]), components={"card": comp})


def test_a_misspelt_placeholder_names_component_placeholder_and_parameters():
    e = next(i for i in errors(card("{{titel}}", {"title": "Hello"}, {"title": "D"}))[0]
             if i["code"] == "missing-param")
    m = e["message"]
    assert e["layer"] == "c1" and "component 'card'" in m and "{{titel}}" in m and "Did you mean 'title'" in m, m


def test_a_placeholder_with_no_value_is_an_error():
    e = one_error(card("{{title}}", {}), "c1")
    assert e["code"] == "missing-param" and "{{title}}" in e["message"], e


def test_placeholders_may_have_spaces_and_null_means_the_default():
    _, s = load(card("{{ title }}", {"title": "Hello"}, {"title": "D"}))
    assert s is not None and s.by_id["c1-t"].raw["text"] == "Hello"
    _, s = load(card("{{title}}!", {"title": None}, {"title": "D"}))
    assert s is not None and s.by_id["c1-t"].raw["text"] == "D!"


def test_a_component_that_uses_itself_is_an_error():
    comps = {"a": {"layers": [{"use": "b"}]}, "b": {"layers": [{"use": "a"}]}}
    e = one_error(dict(scene([{"use": "a"}]), components=comps))
    assert "uses itself" in e["message"] and e["layer"] == "components", e


@pytest.mark.parametrize("comp", [[1], "text", None, {"layers": {"t": 1}}, {"layers": ["a"]}, {"layers": 5}])
def test_a_component_of_the_wrong_shape_is_an_error(comp):
    e = one_error(dict(scene([{"use": "c"}]), components={"c": comp}))
    assert e["message"].startswith("components.c"), e


@pytest.mark.parametrize("inst, words", [({"use": "kinetic", "words": ["a"], "every": "fast"}, "every"),
                                         ({"use": "kinetic", "words": 5}, "words"),
                                         ({"use": "callout", "r": "big"}, "callout")])
def test_builtin_component_parameters_of_the_wrong_type(inst, words):
    e = one_error(scene([dict(inst, id="k")]), "k")
    assert words in e["message"] and "Error:" not in e["message"].split("(")[0], e


# ---- A19: timing ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    scene([rect(id="a", dur=1)], duration="a.out-5"),
    scene([rect(at=-5, dur=1)], duration="auto"),
    scene([RECT], duration="0"),
    scene([rect(at=True)]), scene([rect(dur=True)]), scene([rect(until=False)]),
    scene([rect(at=float("nan"))]), scene([rect(dur=float("nan"))]), scene([rect(dur=float("inf"))]),
    dict(scene([RECT]), markers={"m": float("nan")}), dict(scene([RECT]), markers={"m": "a.out"}),
])
def test_impossible_timing_is_an_error(spec):
    errs, s = errors(spec)
    assert s is None and errs and len(errs[0]["message"]) < 300, errs


def test_a_layer_wholly_outside_the_scene_or_its_group_warns():
    for layer in (rect(at=-1, dur=0.5), rect(at=5)):
        w = warnings(scene([layer], duration=2), "never-shown")
        assert w and w[0]["layer"] == "r", w
    group = {"id": "g", "type": "group", "dur": 1, "children": [rect(at=3, dur=1)]}
    assert warnings(scene([group], duration=5), "never-shown")[0]["layer"] == "r"
    assert not warnings(scene([rect(at=5, hidden=True)], duration=2), "never-shown")


def test_dur_and_until_together_warn_that_until_is_ignored():
    w = warnings(scene([rect(dur=0.5, until=0.8)]), "ignored-key")
    assert w and "until is ignored" in w[0]["message"], w


@pytest.mark.parametrize("word", ["start", "prev"])
def test_an_id_a_cue_keyword_would_shadow_is_an_error(word):
    assert "cue" in one_error(scene([rect(id=word)]), word)["message"]


def test_a_layer_called_end_can_still_be_cued():
    # "end" is only special as the whole of "until": "end"; marketing/hero.json has a group called end.
    _, s = load(scene([rect(id="end", dur=0.4), rect(id="b", at="end.out", dur=0.2)]))
    assert s is not None and s.by_id["b"].t_in == pytest.approx(0.4)


def test_a_marker_named_like_a_layer_warns_and_reads_as_the_marker():
    spec = dict(scene([rect(id="m", dur=0.4), rect(id="b", at="m", dur=0.2)], duration=2), markers={"m": 1})
    w = warnings(spec, "marker-shadows")
    assert w and "'m'" in w[0]["message"], w
    assert load(spec)[1].by_id["b"].t_in == pytest.approx(1)


def test_marker_clocks_and_messages():
    _, s = load(dict(scene([rect(at="drop")], duration=3), markers={"drop": "0:01.5"}))
    assert s is not None and s.by_id["r"].t_in == pytest.approx(1.5)
    e = one_error(dict(scene([rect(at="drop.out")], duration=3), markers={"drop": 1}), "r")
    assert "is a marker, one moment in time, so it has no .out" in e["message"], e


# ---- A20: values that used to be ignored --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"shadow": 5}, {"shadow": "big"}, {"box": 5}, {"box": "dark"}, {"blend": 5},
                                 {"split": 5}, {"split": True}, {"align": 5}, {"anchor": 5}, {"text": {"a": 1}}])
def test_wrong_values_that_used_to_be_ignored_are_errors(bad):
    one_error(scene([dict({"id": "t", "type": "text", "text": "Hi"}, **bad)]), "t")


def test_a_negative_radius_is_an_error():
    assert "radius is -4" in one_error(scene([rect(radius=-4)]), "r")["message"]


def test_null_anchor_is_the_default_and_a_number_text_is_its_digits():
    issues, s = load(scene([{"id": "t", "type": "text", "text": 42, "anchor": None}]))
    assert s is not None and s.by_id["t"].raw["text"] == "42" and "anchor" not in s.by_id["t"].raw, issues


def test_polygon_sides_and_group_children_and_layers():
    assert "sides" in one_error(scene([{"id": "p", "type": "polygon", "sides": 2}]), "p")["message"]
    g = {"id": "g", "type": "group", "children": [RECT], "layers": [dict(RECT, id="r2")]}
    assert one_error(scene([g]), "g")["code"] == "duplicate-key"


@pytest.mark.parametrize("fill", [{"stops": ["#fff"]}, {"stops": []}, {"stops": "red"},
                                  {"stops": ["#fff", "#000"], "linear": [0, 1]},
                                  {"stops": ["#fff", "nope"]}, {"stops": [[0, "#fff"], "#000"]}])
def test_gradient_objects_are_checked(fill):
    assert "gradient" in one_error(scene([rect(fill=fill)]), "r")["message"]


def test_a_gradient_stop_past_the_end_warns_that_it_is_pinned():
    w = warnings(scene([rect(fill="linear-gradient(#f00 0%, #00f 150%)")]), "gradient-clamped")
    assert w and "drawn at 1" in w[0]["message"], w


def test_null_values_mean_their_defaults():
    _, s = load(dict(scene([rect(fill=None, x=None, fx={"brightness": None}),
                            {"id": "t", "type": "text", "text": "a **b**", "emph": None}]),
                     motionBlur={"samples": 0}))
    assert s is not None and "fill" not in s.by_id["r"].raw and s.by_id["r"].get("x", 0) == 0


def test_an_unknown_font_warns_that_it_falls_back():
    w = warnings(scene([{"id": "t", "type": "text", "text": "Hi", "font": "No Such Font Family 12345"}]),
                 "font-missing")
    assert w and w[0]["layer"] == "t" and "not installed" in w[0]["message"], w


# ---- A23: messages name the layer and field and say what is accepted ---------------------------------------------

@pytest.mark.parametrize("fill, words", [("#zzzzzz", "'z', which is not a hex digit"), ("#12345", "5 hex digits"),
                                         ("#ffffffff00", "10 hex digits"), ("rgb()", "0 values"),
                                         ("rgb(a, b, c)", "'a' is not a number"), ("rgb(1, 2)", "2 values")])
def test_a_bad_colour_says_why_and_what_is_accepted(fill, words):
    e = one_error(scene([rect(fill=fill)]), "r")
    assert e["message"].startswith("fill: ") and words in e["message"] and "#rrggbb" in e["fix"], e


def test_a_bad_canvas_bg_names_bg():
    e = one_error(scene([RECT], bg="#zzzzzz"), "canvas")
    assert e["message"].startswith("canvas.bg: "), e
    assert str(k.SceneError.from_issues([e])).startswith("canvas.bg: ")  # not "layer 'canvas': ..."


def test_keyframe_times_take_ms_and_say_which_forms_they_take():
    _, s = load(scene([rect(x=[[0, 0], ["500ms", 100]])]))
    assert s is not None and s.by_id["r"].get("x", 0.5) == pytest.approx(100)
    e = one_error(scene([rect(x=[[0, 0], ["soon", 100]])]), "r")
    assert "keyframe 2" in e["message"] and "seconds" in e["message"], e


@pytest.mark.parametrize("bad, field", [({"enter": {"name": "rise", "dur": "slow"}}, "enter"),
                                        ({"opacity": {"from": 0, "to": 1, "dur": "slow"}}, "opacity")])
def test_words_for_numbers_name_their_field(bad, field):
    e = one_error(scene([rect(**bad)]), "r")
    assert e["message"].startswith(field) and "dur" in e["message"] and "could not convert" not in e["message"], e


def test_huge_times_print_short():
    e = one_error(scene([rect(at=1e308)]), "r")
    assert len(e["message"]) < 200 and "inf" not in e["message"], e


def test_exponents_and_ids_cues_can_name():
    _, s = load(scene([rect(x="1e2")]))
    assert s is not None and s.by_id["r"].get("x", 0) == 100
    _, s = load(scene([rect(id="1st", dur=0.3), rect(id="título", at="1st.out", dur=0.2),
                       rect(id="z", at="título.out", dur=0.1)]))
    assert s is not None and s.by_id["z"].t_in == pytest.approx(0.5)


def test_an_id_a_cue_cannot_name_says_so():
    e = one_error(scene([rect(id="intro 2", dur=0.5), rect(id="b", at="intro 2.out")]), "b")
    assert "names 'intro 2', but a cue cannot name an id with spaces" in e["message"], e


def test_check_issues_name_their_layer():
    r = check(scene([{"use": "kinetic", "id": "k", "words": ["a"], "every": 0}]))
    assert r["issues"][0]["layer"] == "k" and not r["issues"][0]["message"].startswith("layer"), r["issues"]
    r = check(scene([5]))
    assert r["issues"][0]["layer"] == "layers[0]", r["issues"]


@pytest.mark.parametrize("layer, words", [
    ({"id": "s", "type": "shader", "preset": "mesh", "uniforms": {"speed": "fast"}}, "uniforms speed"),
    ({"id": "s", "type": "shader", "sksl": 5}, "sksl"),
    ({"id": "s", "type": "counter", "value": 5, "format": "{:s}"}, "format"),
    ({"id": "s", "type": "path", "d": "M0 0 A 1 2"}, "d: "),
    ({"id": "s", "type": "captions", "cues": [[0, 1]]}, "cues[0]"),
    ({"id": "s", "type": "text", "text": "x", "fill": "=0.5"}, "fill"),
])
def test_values_that_failed_while_drawing_are_found_by_check(layer, words):
    r = check(scene([layer]))
    errs = [i for i in r["issues"] if i["level"] == "error"]
    assert errs and errs[0]["layer"] == "s" and words in errs[0]["message"] and errs[0]["code"] != "render", errs


def test_an_expression_nested_too_deeply_is_an_error():
    e = one_error(scene([rect(x="=" + "+".join(["1"] * 5000))]), "r")
    assert "nested too deeply" in e["message"], e


def test_a_theme_colour_that_is_not_a_colour_is_found_in_nested_fields():
    spec = dict(scene([rect(shadow={"color": "$ink2"}), rect(id="b", fill="$nope")]), theme={"ink2": "inky"})
    errs, s = errors(spec)
    got = {(i["layer"], i["message"].split(":")[0].split(" is")[0]) for i in errs}
    assert s is None and ("r", "shadow color") in got and ("b", "fill") in got, errs


# ---- A24: colour names --------------------------------------------------------------------------------------------

def test_colour_names_are_the_ios_ones_in_any_case_and_the_message_lists_them():
    assert C.parse("RED") == C.parse("red") == C.parse("#ff3b30")
    e = one_error(scene([rect(fill="crimson")]), "r")
    assert "CSS colour" not in e["message"] + e["fix"] and "black, white" in e["fix"] and "#ff3b30" in e["fix"], e


# ---- B14: a model's light object ----------------------------------------------------------------------------------

def test_a_model_light_object_is_read():
    light = {"preset": "dramatic", "from": "left", "key": 0.9}  # "from" made it look like a tween
    _, s = load(scene([{"id": "m", "type": "model", "shape": "cube", "light": light}]))
    assert s is not None and s.by_id["m"].raw["light"] == light
    e = one_error(scene([{"id": "m", "type": "model", "shape": "cube", "light": {"preset": "nope"}}]), "m")
    assert e["message"].startswith("light preset"), e


def test_lights_and_tweening_light_keys_are_checked():
    cube = {"id": "m", "type": "model", "shape": "cube"}
    ok = dict(scene([cube]), light={"azimuth": [[0, -40], ["end", 40]], "color": [[0, "#ffffff"], [1, "#ffd166"]],
                                    "follow": "world", "key": "=0.8+0.2*sin(t)"},
              lights=[{"type": "point", "x": "50%", "y": 0, "z": 400, "range": 600}, {"from": "right"}])
    errs, s = errors(ok)
    assert s is not None and not errs, errs
    e = one_error(dict(scene([cube]), lights=[{"type": "spot"}]))
    assert "lights[0] type" in e["message"], e
    e = one_error(dict(scene([cube]), light={"follow": "scene"}))
    assert "follow" in e["message"], e
    e = one_error(scene([dict(cube, light={"key": [[0, "bright"]]})]), "m")
    assert e["message"].startswith("light key"), e
    assert warnings(dict(scene([cube]), lights=[{}, {}, {}, {}]), "ignored-key")
    with pytest.raises(k.SceneError, match="light key"):
        k.Scene(dict(scene([cube]), light={"key": "=0.5 +"}))


def test_env_is_checked():
    cube = {"id": "m", "type": "model", "shape": "cube"}
    errs, s = errors(dict(scene([dict(cube, env="night")]), env={"preset": "sunset", "top": "#202a50",
                                                                  "amount": [[0, 0], ["end", 1]]}))
    assert s is not None and not errs, errs
    e = one_error(dict(scene([cube]), env="beach"))
    assert e["message"].startswith("env:") and "studio" in e["message"] + e["fix"], e
    e = one_error(scene([dict(cube, env={"top": 5})]), "m")
    assert e["message"].startswith("env top"), e
