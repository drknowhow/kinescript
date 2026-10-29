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
