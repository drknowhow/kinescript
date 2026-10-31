"""Validation: aliases, unknown keys, value checks, tolerant JSON, and a guard that the key lists
in kinescript/schema.py cover every key the renderer reads."""
import collections
import inspect
import re
from pathlib import Path

import pytest

import kinescript as k
from kinescript import components, schema
from kinescript import scene as S
from kinescript.render import Renderer
from kinescript.verify import check

ROOT = Path(__file__).parents[1]


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 1}, **canvas), "layers": layers}


def codes(spec):
    try:
        s = k.Scene(spec)
        return [i["code"] for i in s.issues], s
    except k.SceneError as e:
        return [i["code"] for i in e.issues], None


def test_aliases_are_read_and_noted():
    got, s = codes(scene([{"id": "t", "type": "txt", "content": "Hi", "fontSize": 40, "color": "#fff",
                           "duration": 0.5, "position": [100, 50], "opacity": 50, "rotate": "45deg",
                           "anchor": "middle", "enter": "fadeIn"}]))
    assert s is not None and "renamed-key" in got
    r = s.by_id["t"].raw
    assert (r["type"], r["text"], r["size"], r["fill"], r["dur"]) == ("text", "Hi", 40, "#fff", 0.5)
    assert (r["x"], r["y"], r["opacity"], r["rotate"], r["anchor"]) == (100, 50, 0.5, 45, "center")
    assert s.by_id["t"].enter.names == ["fade"]


def test_depth_keys_are_read_and_the_camera_is_checked():
    spec, rep = schema.normalize(scene([{"id": "a", "type": "rect", "z": -100, "rotationX": 20, "rotateY": "0.5turn"}]))
    a = spec["layers"][0]
    assert (a["z"], a["rotateX"], a["rotateY"]) == (-100, 20, 180)
    assert not [i for i in rep.issues if i["level"] == "error"]
    spec, rep = schema.normalize(dict(scene([{"type": "rect"}]), camera={"orbit": 10, "zom": 2, "fov": 200}))
    assert spec["camera"]["rotateY"] == 10
    errs = [i["message"] for i in rep.issues if i["level"] == "error"]
    assert len(errs) == 2 and "zom" in errs[0] and "fov is 200" in errs[1]


def test_model_keys_parts_and_light_are_checked():
    spec, rep = schema.normalize(dict(scene([
        {"id": "a", "type": "Sphere", "radius": 40, "metallic": 0.5},
        {"id": "b", "type": "model", "shape": "donut", "light": {"from": "above"},
         "parts": [{"shape": "ball", "r": "big", "colr": "#fff"}, {"shape": "cube", "x": [[0, 0], [1, 50]]}]}]),
        light="studoi"))
    a, b = spec["layers"]
    assert (a["type"], a["shape"], a["r"], a["metal"]) == ("model", "sphere", 40, 0.5)
    assert b["shape"] == "torus" and b["parts"][0]["shape"] == "sphere"
    errs = sorted(i["message"] for i in rep.issues if i["level"] == "error")
    assert len(errs) == 5, errs
    assert [m for m in errs if "above" in m] and [m for m in errs if "studoi" in m and "studio" in m]
    assert [m for m in errs if "colr" in m] and [m for m in errs if "'big'" in m]
    assert [m for m in errs if "parts[1]: x" in m and "do not animate" in m]


def test_text_and_path_models_need_their_outline_and_keep_the_2d_types():
    spec, rep = schema.normalize(scene([
        {"id": "a", "shape": "text", "text": "Hi", "d": 30, "fontSize": 90},  # no type: a shape makes it a model
        {"id": "b", "type": "model", "shape": "svg", "w": 200},
        {"id": "c", "type": "model", "parts": [{"shape": "string", "extrude": 20}]},
        {"id": "d", "type": "text", "text": "flat"}, {"id": "e", "type": "path", "d": "M0 0 L10 10"}]))
    a, b, c, d, e = spec["layers"]
    assert (S._infer_type(a), a["depth"], a["size"]) == ("model", 30, 90) and "d" not in a
    assert b["shape"] == "path" and c["parts"][0]["shape"] == "text" and c["parts"][0]["depth"] == 20
    assert (d["type"], e["type"]) == ("text", "path")
    errs = sorted(i["message"] for i in rep.issues if i["level"] == "error")
    assert len(errs) == 2 and "path model needs 'd'" in errs[0] and "parts[0]: a text model needs 'text'" in errs[1]


def test_line_colour_means_stroke():
    _, s = codes(scene([{"id": "l", "type": "line", "points": [[0, 0], [10, 10]], "color": "#f00"}]))
    assert s.by_id["l"].raw["stroke"] == "#f00"


def test_unknown_key_suggests_the_right_one_first():
    with pytest.raises(k.SceneError) as e:
        k.Scene(scene([{"id": "i", "type": "rect", "w": 10, "h": 10, "strokeWidht": 3}]))
    msg = e.value.issues[0]["message"]
    assert msg.startswith("unknown key 'strokeWidht'. Did you mean 'strokeWidth'?")
    assert e.value.issues[0]["layer"] == "i"


def test_every_error_is_reported_at_once():
    with pytest.raises(k.SceneError) as e:
        k.Scene(scene([{"id": "a", "type": "rect", "fil": "#fff"}, {"id": "b", "type": "text", "size": "big"},
                       {"id": "c", "type": "circle", "r": 5, "fill": "redish"}]))
    errs = [(i["layer"], i["code"]) for i in e.value.issues if i["level"] == "error"]
    assert ("a", "unknown-key") in errs and ("b", "missing-key") in errs and ("b", "bad-value") in errs
    assert ("c", "bad-colour") in errs


@pytest.mark.parametrize("canvas,code", [({"fps": 0}, "bad-canvas"), ({"w": 100000}, "bad-canvas"),
                                         ({"duration": 0}, "bad-canvas")])
def test_canvas_limits(canvas, code):
    assert code in codes(scene([{"type": "rect"}], **canvas))[0]


def test_flat_list_is_not_keyframes():
    assert "flat-keyframes" in codes(scene([{"type": "rect", "x": [0, 100]}]))[0]


def test_bad_values_found_before_render_with_layer_and_field():
    with pytest.raises(k.SceneError) as e:
        k.Scene(scene([{"id": "r", "type": "rect", "x": [[0, 0], [1, "left"]]}]))
    i = e.value.issues[0]
    assert i["layer"] == "r" and i["message"].startswith("x is 'left'")


def test_zero_dur_explains_itself():
    with pytest.raises(k.SceneError, match="dur must be greater than 0"):
        k.Scene(scene([{"type": "rect", "dur": 0}]))


def test_component_param_typo_is_reported():
    with pytest.raises(k.SceneError, match="no parameter 'subtitel'. Did you mean 'subtitle'"):
        k.Scene(scene([{"use": "lowerThird", "title": "A", "subtitel": "B"}]))


def test_motion_and_ease_aliases():
    s = k.Scene(scene([{"type": "rect", "enter": {"name": "slideUp", "ease": "easeOutCubic"},
                        "x": {"from": 0, "to": 10, "duration": 1, "ease": "ease-in-out"}}]))
    assert s.layers[0].enter.names == ["slide-up"]


def test_tolerant_json_reads_comments_trailing_commas_and_single_quotes(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("{ // c\n 'canvas': {'w': 320, 'h': 180, 'duration': 1,},\n /* x */ \"layers\": [{\"type\": \"rect\", \"note\": \"it's // fine\"},]}")
    s = k.load(str(f))
    assert s.issues[0]["code"] == "json" and s.layers[0].raw["note"] == "it's // fine"


def test_missing_media_fails_before_render_with_nearby_names(tmp_path):
    (tmp_path / "music.mp3").write_bytes(b"")
    spec = scene([{"type": "rect"}])
    spec["audio"] = [{"src": "musik.mp3"}]
    f = tmp_path / "s.json"
    f.write_text(__import__("json").dumps(spec))
    r = check(str(f))
    assert not r["ok"] and r["issues"][0]["code"] == "media-missing"
    assert "did you mean 'music.mp3'" in r["issues"][0]["fix"]


def test_timing_reference_typo_suggests_id():
    with pytest.raises(k.SceneError, match="Did you mean 'title'"):
        k.Scene(scene([{"id": "title", "type": "rect", "dur": 1}, {"id": "s", "type": "rect", "at": "titel.out"}]))


def test_no_layers():
    assert "no-layers" in codes({"canvas": {}})[0]


def test_audio_roles_pan_and_layer_sounds_are_checked():
    spec = scene([{"type": "rect", "sfx": {"file": "pop.wav", "at": "out", "volum": 2}}])
    spec["audio"] = [{"src": "vo.wav", "type": "voice-over", "pan": 1.5}, {"src": "x.wav", "role": "drums"}, 7]
    out, rep = schema.normalize(spec)
    got = [(i["code"], i["message"]) for i in rep.issues]
    assert out["audio"][0]["role"] == "voice" and out["layers"][0]["sound"]["src"] == "pop.wav"
    assert any(c == "bad-value" and m.startswith("pan is 1.5") for c, m in got)
    assert any(c == "bad-value" and m.startswith("role: unknown value 'drums'") for c, m in got)
    assert any(c == "unknown-key" and m.startswith("sound: unknown key 'volum'") for c, m in got)
    assert any(c == "bad-value" and m.startswith("an audio entry is a file name") for c, m in got)


def test_the_mix_block_is_checked():
    spec = dict(scene([{"type": "rect"}]), mix={"lufs": -14, "peak": 3, "levels": {"music": -20, "drums": -3},
                                               "duck": {"amount": 50, "under": ["music", "pads"]}})
    out, rep = schema.normalize(spec)
    got = [(i["code"], i["message"]) for i in rep.issues]
    assert out["mix"]["loudness"] == -14 and out["mix"]["duck"]["depth"] == 50
    assert any(c == "bad-value" and m.startswith("mix.peak is 3") for c, m in got)
    assert any(c == "unknown-key" and m.startswith("mix.levels: unknown key 'drums'") for c, m in got)
    assert any(c == "bad-value" and m.startswith("mix.duck.depth is 50") for c, m in got)
    assert any(c == "bad-value" and m.startswith("mix.duck.under: unknown role 'pads'") for c, m in got)
    for ok in ({"duck": False}, {"duck": True}, {"duck": {"by": "voice", "under": "music"}}):
        assert not [i for i in schema.normalize(dict(scene([{"type": "rect"}]), mix=ok))[1].issues
                    if i["level"] == "error"]


def test_audio_and_layer_ids_share_one_namespace():
    with pytest.raises(k.SceneError, match="duplicate id 'title'"):
        k.Scene(dict(scene([{"id": "title", "type": "rect"}]), audio=[{"id": "title", "src": __file__}]))


# ---- guards: the hand-kept key lists in schema.py ---------------------------------------------
READ = collections.defaultdict(set)


class _Tracked(dict):
    def __init__(self, d, typ, pre=""):
        super().__init__(d)
        self.typ, self.pre = typ, pre

    def _wrap(self, key, v):
        if isinstance(v, dict) and not isinstance(v, _Tracked):
            v = _Tracked(v, self.typ, f"{self.pre}{key}.")
            dict.__setitem__(self, key, v)
        return v

    def get(self, key, default=None):
        READ[self.typ].add(self.pre + str(key))
        return self._wrap(key, dict.get(self, key, default))

    def __getitem__(self, key):
        READ[self.typ].add(self.pre + str(key))
        return self._wrap(key, dict.__getitem__(self, key))

    def __contains__(self, key):
        READ[self.typ].add(self.pre + str(key))
        return dict.__contains__(self, key)


NESTED = {"shadow": schema.SHADOW, "box": schema.BOX, "fx": schema.FX, "fill": schema.GRADIENT, "ground": schema.GROUND,
          "stroke": schema.GRADIENT}


def test_schema_lists_cover_every_key_the_renderer_reads(monkeypatch):
    orig = S.Layer.get

    def get(self, name, *a, **kw):
        READ[self.type].add(name)
        return orig(self, name, *a, **kw)
    monkeypatch.setattr(S.Layer, "get", get)
    scenes = [ROOT / "examples" / "hello.json", ROOT / "examples" / "edit.json", ROOT / "examples" / "3d.json",
              ROOT / "examples" / "model.json", ROOT / "examples" / "title3d.json", ROOT / "examples" / "import.json",
              ROOT / "examples" / "voiceover.json", ROOT / "marketing" / "promo.json"]
    for path in scenes:
        sc = S.load(str(path))
        for layer in sc.all_layers():
            layer.raw = _Tracked(layer.raw, layer.type)
        r = Renderer(sc, track=True, scale=0.2, backend="cpu")
        for n in range(0, sc.frames, max(1, sc.frames // 25)):
            r.draw(n / sc.fps)
        sc.close()
    missing = []
    for typ, keys in READ.items():
        for key in keys:
            head, _, sub = key.partition(".")
            if head.startswith("u_") or head in schema.COMMENT:
                continue
            allowed = schema.CAMERA if typ == "camera" else schema.COMMON | schema.TYPES[typ]
            ok = head in allowed and (not sub or sub in NESTED.get(head, {sub}))
            if not ok:
                missing.append(f"{typ}.{key}")
    assert not missing, (f"the renderer reads {sorted(missing)} but kinescript/schema.py does not list them; "
                         "add them to TYPES/COMMON (or the nested set) there, or check will reject valid scenes")


def test_component_param_lists_cover_every_param_a_builder_reads():
    for name, fn in components.BUILTIN.items():
        src = inspect.getsource(fn)
        used = set(re.findall(r"""\bp(?:\.get\(|\[)["'](\w+)["']""", src)) - {"id"}
        lacking = used - set(components.PARAMS[name])
        assert not lacking, f"components.PARAMS['{name}'] lacks {sorted(lacking)}; add them there"


def test_examples_and_promo_have_no_unknown_keys():
    for path in ("examples/hello.json", "examples/edit.json", "examples/3d.json", "examples/model.json",
                 "examples/title3d.json", "examples/import.json", "examples/voiceover.json", "marketing/promo.json",
                 "marketing/og.json"):
        sc = k.load(str(ROOT / path))  # loading runs every schema check; a full `check` renders and is slow on CPU
        bad = [i for i in sc.issues if i["level"] == "error" or i["code"] in ("renamed-key", "ignored-key")]
        sc.close()
        assert not bad, (path, bad)


RECT = [{"id": "r", "type": "rect", "w": 10, "h": 10}]


@pytest.mark.parametrize("top, layers", [
    ({"motionBlur": "yes"}, RECT), ({"audio": 5}, RECT), ({"audio": [{"src": "a.wav", "volume": [1]}]}, RECT),
    ({"audio": [{"src": "a.wav", "gain": "loud"}]}, RECT), ({"canvas": {"duration": float("inf")}}, RECT),
    ({"canvas": {"bg": 5}}, RECT), ({"theme": [1]}, RECT), ({"markers": "drop"}, RECT), ({"grade": "warm"}, RECT),
    ({"components": [1]}, RECT),
    ({}, [{"type": "sequence", "clips": [{"src": "a.png"}], "transition": 5}]),
    ({}, [{"type": "sequence", "clips": {"src": "a.png"}}]),
    ({}, [{"id": "s", "type": "shader", "preset": "aurora", "uniforms": [1]}]),
    ({}, [{"id": "p", "type": "path", "points": [[1], [2, 3]]}]), ({}, [{"id": "p", "type": "path", "d": 5}]),
])
def test_wrong_shapes_are_errors_not_crashes(top, layers):
    # Each of these used to pass the schema and then raise TypeError or AttributeError while loading or drawing.
    spec = dict(scene(layers), **top)
    spec["canvas"] = dict(spec["canvas"], **top.get("canvas", {}))
    got, s = codes(spec)
    assert s is None and {"bad-value", "bad-colour", "bad-canvas"} & set(got), got


def test_null_canvas_values_and_a_motion_blur_number_read_as_defaults():
    got, s = codes(dict(scene(RECT, fps=None, w=None), motionBlur=6))
    assert s is not None and (s.fps, s.w, s.motion_blur) == (30, 1920, {"samples": 6}), got
    assert schema.normalize(dict(scene(RECT), audio="a.wav"))[0]["audio"] == [{"src": "a.wav"}]


def test_capitalised_text_type_is_a_text_layer_not_a_model():
    _, s = codes(scene([{"id": "t", "type": "Text", "text": "Hi"}]))
    assert s is not None and s.by_id["t"].type == "text"


def test_a_model_fill_may_be_a_colour_tween():
    got, s = codes(scene([{"id": "m", "type": "model", "shape": "cube",
                           "fill": {"from": "#ff0000", "to": "#0000ff", "dur": 1}}]))
    assert s is not None and "bad-value" not in got and "fill" in s.by_id["m"].props


def test_a_component_used_twice_owns_ids_nested_under_layers():
    body = [{"id": "g", "type": "group", "layers": [{"id": "k", "type": "rect", "w": 10, "h": 10}]}]
    got, s = codes(dict(scene([{"use": "card", "id": "a"}, {"use": "card", "id": "b"}]),
                        components={"card": {"layers": body}}))
    assert s is not None and {"a-k", "b-k"} <= set(s.by_id), got  # was: duplicate layer id 'k'


def test_shapes_models_write_instead_of_kinescript_are_read():
    from kinescript.render import glsl_to_sksl
    _, s = codes(scene([{"id": "p", "type": "progress"},
                        {"id": "t", "type": "rect", "tween": {"opacity": {"from": 1, "to": 0}},
                         "fill": "linear-gradient(to right, #00f, #808)"}]))
    assert s is not None and s.by_id["p"].type == "rect"  # the progress component expanded
    assert s.by_id["t"].raw["fill"] == {"linear": [[0, 0.5], [1, 0.5]], "stops": ["#00f", "#808"]}
    assert "opacity" in s.by_id["t"].props
    assert "half4 main(float2 _kpos)" in glsl_to_sksl("void main(){ gl_FragColor = vec4(gl_FragCoord.xy, 0, 1); }")
    r = check(scene([{"type": "shader", "sksl": "void mainImage(out vec4 c, in vec2 f){ c = vec4(f / iResolution, 0, 1); }"}]))
    assert r["ok"], r["issues"]
