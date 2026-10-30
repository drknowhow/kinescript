"""Every JSON example in `kine docs` must load cleanly and use canonical names, because small
models copy these verbatim."""
import json
import re
from pathlib import Path

import numpy as np
import skia

import kinescript as k
from kinescript.docs import reference
from kinescript.verify import check

ROOT = Path(__file__).parents[1]


def _objects(text: str) -> list[str]:
    """JSON objects that start a line (after indentation), joined across their continuation lines."""
    out, buf, depth = [], [], 0
    for line in text.splitlines():
        s = line.strip()
        if not buf and not s.startswith("{"):
            continue
        buf.append(s)
        depth += s.count("{") - s.count("}")
        if depth <= 0:
            out.append(" ".join(buf))
            buf, depth = [], 0
    return out


def _section(name: str) -> str:
    doc = reference()
    start = doc.index(name + "\n")
    nxt = re.search(r"\n[A-Z][A-Z ]+(\(|\n)", doc[start + len(name) + 1:])
    return doc[start: start + len(name) + 1 + (nxt.start() if nxt else len(doc))]


def test_complete_scene_in_docs_passes_check_with_no_rewrites():
    spec = json.loads(_objects(_section("A COMPLETE SCENE (copy it and change it)"))[0])
    r = check(spec)
    assert r["ok"], r["issues"]
    assert not [i for i in r["issues"] if i["level"] == "info"], r["issues"]
    assert r["duration"] == 3.3


def test_depth_examples_in_docs_pass_check_with_no_rewrites():
    objs = [json.loads(o) for o in _objects(_section("DEPTH AND CAMERA (3D)"))]
    card = next(o for o in objs if o.get("id") == "card")
    whole = next(o for o in objs if "layers" in o)
    for spec in (whole, {"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 2}, "layers": [card]}):
        r = check(spec)
        assert r["ok"] and not r["issues"], r["issues"]


def test_model_examples_in_docs_pass_check_with_no_rewrites():
    models = [json.loads(o) for o in _objects(_section("3D MODELS"))]
    assert {m["id"] for m in models} == {"emblem", "title3d", "product"}
    for m in models:
        if m.get("src"):  # the sample model examples/make_media.py writes
            m["src"] = str(ROOT / "examples" / "media" / m["src"]).replace("\\", "/")
        r = check({"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 4}, "layers": [m]})
        assert r["ok"] and not r["issues"], (m["id"], r["issues"])


def test_sound_examples_in_docs_pass_check_with_no_rewrites():
    pad = str(ROOT / "examples" / "media" / "pad.m4a").replace("\\", "/")
    sounds = [json.loads(o) for o in _objects(_section("SOUND"))]
    assert {s["id"] for s in sounds} == {"vo1", "bed"}
    for s in sounds:
        s["src"] = pad
    spec = {"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": "auto"}, "audio": sounds,
            "layers": [{"id": "after", "type": "rect", "at": "vo1.out", "dur": 1}]}
    r = check(spec)
    assert r["ok"] and not r["issues"], r["issues"]


def test_every_recipe_loads_with_no_rewrites(tmp_path):
    img = tmp_path / "photo.jpg"
    arr = np.zeros((90, 160, 4), np.uint8)
    arr[..., 0], arr[..., 3] = 200, 255
    skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType).save(str(img), skia.kJPEG)
    clip = str(ROOT / "examples" / "media" / "cells.mp4").replace("\\", "/")
    layers = []
    for obj in _objects(_section("RECIPES (layers to paste into \"layers\")")):
        obj = obj.replace('"a.mp4"', json.dumps(clip)).replace('"b.mp4"', json.dumps(clip))
        layers.append(json.loads(obj))
    assert len(layers) >= 9
    spec = {"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 4}, "layers": layers}
    f = tmp_path / "recipes.json"
    f.write_text(json.dumps(spec))
    s = k.load(str(f))
    assert not [i for i in s.issues if i["level"] != "warning"], s.issues
    s.close()


def test_bar_chart_recipe_passes_check_as_a_three_bar_chart():
    """Small models copy the bar recipe once per bar, often with bigger text; the copies must not
    warn. At 64 px a centre-anchored label at y 920 reaches into the bar, so this guards the anchor."""
    bar = [json.loads(o) for o in _objects(_section("RECIPES (layers to paste into \"layers\")"))
           if json.loads(o).get("id", "").startswith("bar1")]
    assert {L["id"] for L in bar} == {"bar1", "bar1-value", "bar1-label"}
    layers = []
    for n, dx in enumerate((-400, 0, 400), 1):
        for L in json.loads(json.dumps(bar).replace("bar1", f"bar{n}")):
            L["x"] += dx
            if L["id"].endswith("-label"):
                L["size"] = 64
            layers.append(L)
    r = check({"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 3}, "layers": layers})
    assert r["ok"] and not [i for i in r["issues"] if i["level"] != "info"], r["issues"]


def test_contrast_warning_names_the_shape_behind_but_not_the_background():
    def warn(layers):
        r = check({"canvas": {"w": 1280, "h": 720, "fps": 30, "duration": 2}, "layers": layers})
        return [i for i in r["issues"] if i["code"] == "contrast"]
    text = {"id": "t", "type": "text", "text": "Hello", "x": 640, "y": 360, "size": 64, "fill": "#f2f0e8"}
    on_shape = warn([{"id": "slab", "type": "rect", "x": 640, "y": 360, "w": 600, "h": 200, "fill": "#ff633f"}, text])
    assert on_shape and "overlaps 'slab'" in on_shape[0]["message"] and "slab" in on_shape[0]["fix"]
    on_bg = warn([{"id": "bg", "type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": "100%", "h": "100%",
                   "fill": "#ff633f"}, text])
    assert on_bg and "overlaps" not in on_bg[0]["message"]
    default_fill = {k: v for k, v in text.items() if k != "fill"}  # the theme's paper colour
    assert warn([{"id": "slab", "type": "rect", "x": 640, "y": 360, "w": 600, "h": 200, "fill": "#ff633f"},
                 default_fill])
