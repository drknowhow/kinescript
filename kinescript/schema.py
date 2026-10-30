"""Scene validation written for agents, including small ones.

`normalize(spec)` runs before a scene is built. It
  - rewrites well-known synonyms from other tools to Kinescript's names ("color" -> "fill",
    "fontSize" -> "size", "duration" -> "dur", ...) and reports each rewrite as an info note,
  - rejects every key Kinescript does not read, with the closest valid name first,
  - catches values that would only fail (or silently do nothing) at render time,
and reports all problems at once, each with the layer id and field, so one `kine check`
round fixes them all.

The key lists below are hand-kept. Their failure polarity is fail-loud: a key the renderer
reads but a list lacks makes `kine check` reject valid scenes, and tests/test_schema.py
renders the examples and the promo with a tracer and fails, naming this file, when the
renderer reads a key that is missing here.
"""
from __future__ import annotations

import copy
import difflib
import re

from . import color as C

# Keys that are always allowed and ignored: notes to self.
COMMENT = {"note", "notes", "comment", "comments", "description", "name"}

TOP = {"canvas", "theme", "markers", "grade", "motionBlur", "audio", "components", "layers", "meta", "version",
       "$schema", "camera", "light"}
TOP_ALIAS = {"layer": "layers", "elements": "layers", "objects": "layers", "items": "layers", "scene": "layers",
             "children": "layers", "tracks": "layers", "motion_blur": "motionBlur", "colorGrade": "grade",
             "music": "audio", "sound": "audio", "marker": "markers"}
CANVAS = {"w", "h", "fps", "bg", "duration"}
CANVAS_ALIAS = {"width": "w", "height": "h", "background": "bg", "backgroundColor": "bg", "background_color": "bg",
                "bgColor": "bg", "color": "bg", "fill": "bg", "frameRate": "fps", "framerate": "fps",
                "frame_rate": "fps", "length": "duration", "dur": "duration", "seconds": "duration"}
FX = {"brightness", "contrast", "saturate", "saturation", "hue", "exposure", "temperature", "tint", "gray", "invert"}
GRADE = FX | {"vignette", "grain"}
SHADOW = {"x", "y", "blur", "color"}
SHADOW_ALIAS = {"colour": "color", "fill": "color", "offsetX": "x", "offsetY": "y", "dx": "x", "dy": "y",
                "radius": "blur", "spread": "blur"}
BOX = {"fill", "pad", "radius"}
BOX_ALIAS = {"color": "fill", "colour": "fill", "background": "fill", "bg": "fill", "padding": "pad",
             "borderRadius": "radius", "cornerRadius": "radius"}
MOTION = {"name", "dur", "ease", "stagger", "dist", "from", "amount", "deg"}
MOTION_ALIAS = {"type": "name", "preset": "name", "duration": "dur", "easing": "ease", "distance": "dist",
                "delay": "stagger"}
TWEEN = {"from", "to", "dur", "delay", "ease", "keys"}
TWEEN_ALIAS = {"duration": "dur", "easing": "ease", "start": "from", "end": "to", "begin": "from"}
GRADIENT = {"linear", "radial", "stops"}
AUDIO = {"src", "at", "trim", "gain", "volume", "fadeIn", "fadeOut", "id", "role", "dur", "until", "loop", "pan"}
AUDIO_ALIAS = {"source": "src", "file": "src", "path": "src", "start": "at", "fade_in": "fadeIn",
               "fade_out": "fadeOut", "fadein": "fadeIn", "fadeout": "fadeOut", "duration": "dur", "length": "dur",
               "end": "until", "type": "role", "kind": "role", "balance": "pan", "repeat": "loop"}
# A layer's "sound": cued from the layer, so it has no id, until or loop of its own.
SOUND = AUDIO - {"id", "until", "loop"}
MOTION_BLUR = {"samples", "shutter"}
CAMERA = {"x", "y", "z", "rotateX", "rotateY", "rotate", "fov"}
CAMERA_ALIAS = {"rotationX": "rotateX", "rotationY": "rotateY", "rotateZ": "rotate", "rotationZ": "rotate",
                "rotation": "rotate", "roll": "rotate", "orbit": "rotateY", "tilt": "rotateX", "dolly": "z",
                "fieldOfView": "fov"}
LIGHT = {"preset", "key", "fill", "ambient", "rim", "from", "color"}
LIGHT_ALIAS = {"colour": "color", "direction": "from", "dir": "from", "position": "from", "intensity": "key",
               "ambience": "ambient", "name": "preset"}
MATERIAL = {"fill", "shine", "metal", "rim", "bands"}
PART = MATERIAL | {"shape", "w", "h", "depth", "r", "tube", "x", "y", "z", "rotateX", "rotateY", "rotate", "scale",
                   "text", "size", "font", "weight", "italic", "letterSpacing", "lineHeight", "d", "strokeWidth",
                   "bevel"}
MODEL_ALIAS = {"radius": "r", "thickness": "tube", "geometry": "shape", "primitive": "shape", "extrude": "depth",
               "extrusion": "depth", "chamfer": "bevel",
               "mesh": "shape", "gloss": "shine", "glossiness": "shine", "shininess": "shine",
               "specular": "shine", "metallic": "metal", "metalness": "metal", "toon": "bands", "steps": "bands",
               "quality": "detail", "lighting": "light", "children": "parts", "meshes": "parts", "model": "src",
               "floor": "ground", "groundShadow": "ground", "contactShadow": "ground"}
GROUND = {"y", "opacity", "blur"}
GROUND_ALIAS = {"alpha": "opacity", "strength": "opacity", "softness": "blur", "radius": "blur", "level": "y"}

COMMON = {"id", "type", "at", "dur", "until", "x", "y", "anchor", "opacity", "scale", "scaleX", "scaleY", "rotate",
          "blur", "blend", "shadow", "fx", "enter", "exit", "hidden", "allowOffscreen", "z", "rotateX", "rotateY",
          "backface", "fixed", "sound"}
SHAPE = {"fill", "stroke", "strokeWidth", "draw", "drawStart", "cap", "join"}
TEXT = {"text", "size", "weight", "font", "italic", "fill", "emph", "align", "maxWidth", "lineHeight",
        "letterSpacing", "stroke", "strokeWidth", "box", "case", "textCase", "caret", "split",
        "unitX", "unitY", "unitRotate", "unitScale", "unitOpacity"}
MEDIA = {"src", "w", "h", "fit", "zoom", "focus", "focusX", "focusY", "radius", "stroke", "strokeWidth", "cap", "join"}
TYPES: dict[str, set[str]] = {
    "rect": SHAPE | {"w", "h", "radius", "dimOutside"},
    "circle": SHAPE | {"r"},
    "ellipse": SHAPE | {"w", "h"},
    "polygon": SHAPE | {"sides", "r", "inner"},
    "path": SHAPE | {"d", "points", "smooth", "closed"},
    "line": SHAPE | {"points", "smooth", "closed"},
    "arrow": SHAPE | {"x1", "y1", "x2", "y2", "bend", "head"},
    "text": TEXT,
    "counter": TEXT | {"value", "format"},
    "image": MEDIA,
    "video": MEDIA | {"trim", "speed", "volume", "loop"},
    "captions": TEXT | {"src", "cues", "mode", "active"},
    "shader": {"preset", "uniforms", "sksl", "w", "h"},
    "group": {"children", "layers", "w", "h", "clip", "radius", "fill", "sort"},
    "model": PART - {"x", "y", "z", "rotateX", "rotateY", "rotate", "scale"} | {"detail", "parts", "explode",
                                                                              "draw", "light", "src", "ground"},
}
SEQUENCE = COMMON | {"clips", "transition", "fit", "volume", "w", "h"}
TYPE_ALIAS = {"rectangle": "rect", "square": "rect", "box": "rect", "shape": "rect", "txt": "text", "label": "text",
              "title": "text", "heading": "text", "string": "text", "img": "image", "picture": "image",
              "photo": "image", "oval": "ellipse", "caption": "captions", "subtitles": "captions",
              "subtitle": "captions", "number": "counter", "count": "counter", "movie": "video", "clip": "video",
              "svg": "path", "polyline": "line", "star": "polygon", "background": "shader", "container": "group",
              "frame": "group", "mesh": "model", "3d": "model", "object": "model", "primitive": "model",
              "solid": "model"}
# Synonyms other tools use. A per-type entry wins over the general one.
ALIAS = {"color": "fill", "colour": "fill", "fillColor": "fill", "backgroundColor": "fill", "background": "fill",
         "fontSize": "size", "font_size": "size", "textSize": "size", "fontWeight": "weight",
         "font_weight": "weight", "fontFamily": "font", "font_family": "font", "fontStyle": "italic",
         "duration": "dur", "length": "dur", "start": "at", "startTime": "at", "start_time": "at",
         "begin": "at", "delay": "at", "time": "at", "end": "until", "endTime": "until", "end_time": "until",
         "width": "w", "height": "h", "rotation": "rotate", "angle": "rotate", "content": "text",
         "alpha": "opacity", "borderRadius": "radius", "cornerRadius": "radius", "border_radius": "radius",
         "lineWidth": "strokeWidth", "stroke_width": "strokeWidth", "borderWidth": "strokeWidth",
         "strokeColor": "stroke", "borderColor": "stroke", "source": "src", "file": "src", "url": "src",
         "textAlign": "align", "text_align": "align", "letter_spacing": "letterSpacing", "tracking": "letterSpacing",
         "line_height": "lineHeight", "leading": "lineHeight", "max_width": "maxWidth", "wrap": "maxWidth",
         "animation": "enter", "animate": "enter", "entrance": "enter", "in": "enter", "intro": "enter",
         "out": "exit", "outro": "exit", "exitAnimation": "exit", "transform": None, "kids": "children",
         "items": "children", "elements": "children", "visible": None, "zIndex": None,
         "rotationX": "rotateX", "rotationY": "rotateY", "rotateZ": "rotate", "rotationZ": "rotate",
         "translateZ": "z", "backfaceVisibility": "backface", "sfx": "sound", "audio": "sound",
         "soundEffect": "sound"}
TYPE_ALIASES = {
    "circle": {"radius": "r", "size": "r"}, "polygon": {"radius": "r", "points": "sides"},
    "line": {"color": "stroke", "colour": "stroke", "fill": "stroke", "width": "strokeWidth"},
    "arrow": {"color": "stroke", "colour": "stroke", "fill": "stroke", "width": "strokeWidth"},
    "image": {"path": "src", "image": "src"}, "video": {"path": "src", "video": "src", "mute": None},
    "counter": {"number": "value", "to": "value"},
    "captions": {"subtitles": "cues", "lines": "cues"},
    "shader": {"shader": "preset", "code": "sksl", "source": "sksl"},
    "model": MODEL_ALIAS,
}
ENUMS = {
    "anchor": ("center", "topleft", "top", "topright", "left", "right", "bottomleft", "bottom", "bottomright"),
    "align": ("left", "center", "right"),
    "split": ("chars", "words", "lines"),
    "fit": ("cover", "contain", "stretch"),
    "case": ("upper", "lower"), "textCase": ("upper", "lower"),
    "cap": ("butt", "round", "square"), "join": ("miter", "round", "bevel"),
    "mode": ("karaoke", "pop"),
    "blend": ("normal", "multiply", "screen", "overlay", "add", "lighten", "darken", "softLight", "difference",
              "color"),
    "backface": ("visible", "hidden"),
    "sort": ("order", "depth"),
    "shape": ("cube", "sphere", "cylinder", "cone", "torus", "plane", "text", "path"),
    "detail": ("low", "medium", "high"),
    "role": ("voice", "music", "sfx", "ambience"),
}
ENUM_ALIAS = {"middle": "center", "centre": "center", "centered": "center", "mid": "center",
              "characters": "chars", "letters": "chars", "char": "chars", "word": "words", "line": "lines",
              "uppercase": "upper", "lowercase": "lower", "fill": "cover", "crop": "cover", "scale": "contain",
              "plus": "add", "additive": "add", "soft-light": "softLight", "softlight": "softLight",
              "box": "cube", "cuboid": "cube", "ball": "sphere", "globe": "sphere", "donut": "torus",
              "doughnut": "torus", "ring": "torus", "tube": "cylinder", "can": "cylinder", "pipe": "cylinder",
              "card": "plane", "quad": "plane", "svg": "path", "string": "text", "med": "medium", "hi": "high",
              "lo": "low", "vo": "voice", "voiceover": "voice", "narration": "voice", "narrator": "voice",
              "speech": "voice", "dialogue": "voice", "bed": "music", "bgm": "music", "song": "music",
              "score": "music", "fx": "sfx", "effect": "sfx", "effects": "sfx", "soundeffect": "sfx",
              "ambient": "ambience", "ambiance": "ambience", "atmos": "ambience", "atmosphere": "ambience",
              "roomtone": "ambience"}
WEIGHTS = {"thin": 100, "extralight": 200, "light": 300, "normal": 400, "regular": 400, "book": 400,
           "medium": 500, "semibold": 600, "demibold": 600, "bold": 700, "extrabold": 800, "heavy": 900,
           "black": 900}
COLOR_KEYS = {"fill", "stroke", "emph", "active"}
# Objects some tools nest animated values under; their keys belong on the layer itself.
WRAPPERS = {"tween", "tweens", "keyframes", "animate", "animation", "animations", "props", "properties", "style"}
# Props whose value must be one number (after % and expressions are resolved).
SCALAR = {"x", "y", "w", "h", "r", "opacity", "scale", "scaleX", "scaleY", "rotate", "blur", "size", "weight",
          "strokeWidth", "draw", "drawStart", "zoom", "value", "radius", "dimOutside", "sides", "inner", "x1", "y1",
          "x2", "y2", "bend", "head", "maxWidth", "lineHeight", "letterSpacing", "focusX", "focusY", "speed",
          "volume", "unitX", "unitY", "unitRotate", "unitScale", "unitOpacity", "z", "rotateX", "rotateY", "fov",
          "depth", "tube", "shine", "metal", "rim", "bands", "explode", "bevel"}
REQUIRED = {"text": ("text",), "image": ("src",), "video": ("src",), "path": ("d", "points"), "line": ("points",),
            "captions": ("cues", "src"), "shader": ("preset", "sksl"), "model": ("shape", "parts", "src")}


class Report:
    def __init__(self):
        self.issues: list[dict] = []

    def add(self, level, code, layer, message, fix=""):
        self.issues.append({"level": level, "code": code, "layer": layer, "message": message, "fix": fix})

    def errors(self):
        return [i for i in self.issues if i["level"] == "error"]


def suggest(word: str, options) -> str | None:
    opts = list(options)
    low = {o.lower(): o for o in opts}
    w = str(word).strip()
    if w.lower() in low:
        return low[w.lower()]
    squashed = {re.sub(r"[-_ ]", "", o.lower()): o for o in opts}
    if re.sub(r"[-_ ]", "", w.lower()) in squashed:
        return squashed[re.sub(r"[-_ ]", "", w.lower())]
    m = difflib.get_close_matches(w, opts, n=1, cutoff=0.6) or difflib.get_close_matches(w.lower(), list(low), 1, 0.6)
    return (low.get(m[0], m[0])) if m else None


def unknown(kind: str, word, options) -> str:
    """'unknown motion 'slid'. Did you mean 'slide-up'? Known: ...' with the suggestion first."""
    s = suggest(word, options)
    head = f"unknown {kind} '{word}'."
    if s:
        head += f" Did you mean '{s}'?"
    return f"{head} Known: {', '.join(sorted(options))}"


def _keys(obj: dict, allowed: set, alias: dict, rep: Report, lid, where: str, extra_ok=()):
    """Rename aliases in place and report unknown keys of one object."""
    for k in list(obj):
        if k in allowed or k in COMMENT or k in extra_ok or str(k).startswith(("_", "//", "$")):
            continue
        if k in alias and (alias[k] is None or alias[k] in allowed):
            to = alias[k]
            if to is None:
                del obj[k]
                rep.add("info", "ignored-key", lid, f"{where}'{k}' has no meaning in Kinescript and was ignored")
                continue
            if to in obj:
                rep.add("error", "duplicate-key", lid, f"{where}both '{k}' and '{to}' are set",
                        f"keep only '{to}'")
                continue
            obj[to] = obj.pop(k)
            rep.add("info", "renamed-key", lid, f"{where}'{k}' read as '{to}'", f"write '{to}'")
            continue
        s = suggest(k, allowed)
        rep.add("error", "unknown-key", lid,
                f"{where}unknown key '{k}'." + (f" Did you mean '{s}'?" if s else "")
                + f" Valid here: {', '.join(sorted(allowed))}", f"rename or remove '{k}'")


def _enum(raw: dict, key: str, rep: Report, lid):
    v = raw.get(key)
    if not isinstance(v, str):
        return
    opts = ENUMS[key]
    if v in opts:
        return
    low = re.sub(r"[-_ ]", "", v.strip().lower())
    if key == "anchor":
        low = low.replace("middle", "center").replace("centre", "center").replace("upper", "top").replace(
            "lower", "bottom")
        for a, b in (("lefttop", "topleft"), ("righttop", "topright"), ("leftbottom", "bottomleft"),
                     ("rightbottom", "bottomright"), ("topcenter", "top"), ("centertop", "top"),
                     ("bottomcenter", "bottom"), ("centerbottom", "bottom"), ("centerleft", "left"),
                     ("leftcenter", "left"), ("centerright", "right"), ("rightcenter", "right"),
                     ("centercenter", "center")):
            low = low.replace(a, b)
    hit = ENUM_ALIAS.get(low, low)
    if hit not in opts:
        hit = next((o for o in opts if o.lower() == hit), None)
    if hit:
        raw[key] = hit
        rep.add("info", "renamed-value", lid, f"{key} '{v}' read as '{hit}'", f"write '{hit}'")
        return
    rep.add("error", "bad-value", lid, f"{key}: " + unknown("value", v, opts))


_DEG = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(deg|°|rad|turn)\s*$")


def _values(raw: dict, ltype: str, rep: Report, lid):
    """Fix values that have one obvious meaning; reject the ones that would fail later."""
    for k in ("x", "y"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip().lower() in ("center", "centre", "middle"):
            raw[k] = "50%"
            rep.add("info", "renamed-value", lid, f"{k} '{v}' read as '50%'", f"write \"{k}\": \"50%\"")
    for k in ("rotate", "rotateX", "rotateY"):
        v = raw.get(k)
        if isinstance(v, str) and (m := _DEG.match(v)):
            num, unit = float(m.group(1)), m.group(2)
            raw[k] = num * {"rad": 57.29577951308232, "turn": 360.0}.get(unit, 1.0)
            rep.add("info", "renamed-value", lid, f"{k} '{v}' read as {raw[k]:g} (degrees)",
                    f"{k} takes a number of degrees")
    v = raw.get("opacity")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 1 < v <= 100:
        raw["opacity"] = v / 100
        rep.add("info", "renamed-value", lid, f"opacity {v} read as {v / 100:g}", "opacity runs from 0 to 1")
    v = raw.get("weight")
    if isinstance(v, str) and not v.startswith("="):
        w = WEIGHTS.get(re.sub(r"[-_ ]", "", v.lower()))
        if w:
            raw["weight"] = w
            rep.add("info", "renamed-value", lid, f"weight '{v}' read as {w}", "weight takes 100..900")
    for k in ("italic", "hidden", "clip", "loop", "smooth", "closed", "allowOffscreen", "fixed"):
        if isinstance(raw.get(k), str) and raw[k].lower() in ("true", "false", "yes", "no"):
            raw[k] = raw[k].lower() in ("true", "yes")
    if isinstance(raw.get("italic"), str) and raw["italic"].lower() in ("italic", "oblique", "normal"):
        raw["italic"] = raw["italic"].lower() != "normal"
    for key in ENUMS:
        _enum(raw, key, rep, lid)
    for k in ("dur",):
        v = raw.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0:
            rep.add("error", "bad-dur", lid, f"dur must be greater than 0 (got {v})",
                    "give it a positive number of seconds, or remove dur to run to the end; to hide a layer use \"hidden\": true")
    for k, v in raw.items():
        if k in SCALAR and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            rep.add("error", "flat-keyframes", lid,
                    f"{k}: {v} is a flat list. Keyframes are a list of [time, value] pairs",
                    f"\"{k}\": [[0, {v[0]}], [\"end\", {v[-1]}]]")
        elif k in SCALAR and isinstance(v, str) and not v.startswith(("=", "$")) and not _lengthish(v):
            rep.add("error", "bad-value", lid, f"{k} is '{v}', which is not a number",
                    f"use a number{', a percentage like 50%' if k in ('x', 'y', 'w', 'h', 'maxWidth') else ''} "
                    "or an expression starting with '='")
        elif k in SCALAR and isinstance(v, bool):
            rep.add("error", "bad-value", lid, f"{k} is {str(v).lower()}, which is not a number", "use a number")
        elif k in ("fill", "stroke") and isinstance(v, str) and (g := css_gradient(v)) is not None:
            raw[k] = g
            rep.add("info", "renamed-value", lid, f"{k}: CSS gradient read as {g}", "write the gradient object")
        elif k in COLOR_KEYS and isinstance(v, str) and not v.startswith(("=", "$")) and not C.is_color(v):
            rep.add("error", "bad-colour", lid, f"{k} '{v}' is not a colour",
                    "use #rrggbb, #rrggbbaa, rgb(...), or a CSS colour name such as 'red'")
        elif isinstance(v, dict) and ("to" in v or "keys" in v or "from" in v):
            _keys(v, TWEEN, TWEEN_ALIAS, rep, lid, f"{k} tween: ")
        elif k in ("fill", "stroke") and isinstance(v, dict):
            _keys(v, GRADIENT, {"colors": "stops", "colours": "stops"}, rep, lid, f"{k} gradient: ")
            if "stops" not in v:
                rep.add("error", "bad-value", lid, f"{k}: a gradient needs 'stops'",
                        '{"linear": [[0, 0], [1, 1]], "stops": ["#ff633f", "#b9f45d"]}')


_CSS_DIR = {"to right": ([0, 0.5], [1, 0.5]), "to left": ([1, 0.5], [0, 0.5]), "to bottom": ([0.5, 0], [0.5, 1]),
            "to top": ([0.5, 1], [0.5, 0]), "to bottom right": ([0, 0], [1, 1]), "to top left": ([1, 1], [0, 0]),
            "to top right": ([0, 1], [1, 0]), "to bottom left": ([1, 0], [0, 1])}


def css_gradient(v: str) -> dict | None:
    """'linear-gradient(90deg, #00f, #808)' or 'radial-gradient(#fff, #000)' -> a gradient object."""
    import math
    m = re.fullmatch(r"\s*(linear|radial)-gradient\((.*)\)\s*", v, re.S)
    if not m:
        return None
    parts = [p.strip() for p in re.split(r",(?![^(]*\))", m.group(2)) if p.strip()]
    pts = ([0.5, 0], [0.5, 1])  # CSS default: top to bottom
    if parts and (parts[0] in _CSS_DIR or re.fullmatch(r"-?\d+(\.\d+)?deg", parts[0])):
        d = parts.pop(0)
        if d in _CSS_DIR:
            pts = _CSS_DIR[d]
        else:
            a = math.radians(float(d[:-3]))
            dx, dy = math.sin(a) / 2, -math.cos(a) / 2
            pts = ([round(0.5 - dx, 3), round(0.5 - dy, 3)], [round(0.5 + dx, 3), round(0.5 + dy, 3)])
    stops = []
    for p in parts:
        mm = re.fullmatch(r"(.+?)\s+(\d+(?:\.\d+)?)%", p)
        col, pos = (mm.group(1), float(mm.group(2)) / 100) if mm else (p, None)
        if not C.is_color(col):
            return None
        stops.append(col if pos is None else [pos, col])
    if len(stops) < 2:
        return None
    if any(isinstance(x, list) for x in stops):
        n = len(stops)
        stops = [x if isinstance(x, list) else [i / (n - 1), x] for i, x in enumerate(stops)]
    return {"radial": [0.5, 0.5, 0.7], "stops": stops} if m.group(1) == "radial" else {"linear": list(pts), "stops": stops}


_NUMLIKE = re.compile(r"^\s*-?\d+(\.\d+)?\s*(%|vw|vh|vmin|px)?\s*([+-]\s*\d+(\.\d+)?\s*(px)?)?\s*$")


def _lengthish(v: str) -> bool:
    return bool(_NUMLIKE.match(v))


def _motion(raw: dict, key: str, rep: Report, lid):
    from .presets import PRESETS, canonical
    v = raw.get(key)
    if v in (None, False, "", "none"):
        return
    spec = {"name": v} if isinstance(v, str) else v
    if not isinstance(spec, dict):
        rep.add("error", "bad-value", lid, f"{key} must be a motion name like \"rise\" or an object "
                f"{{\"name\": \"rise\", \"dur\": 0.5}}, got {type(v).__name__}")
        return
    if isinstance(v, dict):
        _keys(v, MOTION, MOTION_ALIAS, rep, lid, f"{key}: ")
    for n in str(spec.get("name", "fade")).split("+"):
        n = n.strip()
        if n and canonical(n) is None:
            rep.add("error", "bad-motion", lid, f"{key}: " + unknown("motion", n, PRESETS))


def _layer(raw, rep: Report, path: str, comps: dict):
    if not isinstance(raw, dict):
        rep.add("error", "bad-layer", None, f"{path} must be an object, got {type(raw).__name__}")
        return
    lid = raw.get("id") or path
    if "use" not in raw and isinstance(raw.get("type"), str):
        from .components import BUILTIN
        name = suggest(raw["type"], BUILTIN)
        if name and name.lower() == raw["type"].strip().lower():
            raw["use"] = name
            del raw["type"]
            rep.add("info", "renamed-key", lid, f"type '{name}' is a component; read as \"use\": \"{name}\"",
                    f"write \"use\": \"{name}\"")
    for wrap in [k for k in raw if k in WRAPPERS and isinstance(raw[k], dict)
                 and not {"name", "type", "preset"} & set(raw[k])]:  # a motion object is not a wrapper
        inner = raw.pop(wrap)
        for k, v in inner.items():
            if k in raw:
                rep.add("error", "duplicate-key", lid, f"'{k}' is set both directly and inside '{wrap}'")
            else:
                raw[k] = v
        rep.add("info", "renamed-key", lid, f"'{wrap}' unwrapped: its keys ({', '.join(inner)}) are layer keys",
                "put animated values directly on the layer, e.g. \"opacity\": {\"from\": 0, \"to\": 1}")
    if "use" in raw:
        from .components import BUILTIN
        name = raw["use"]
        if name not in BUILTIN and name not in comps:
            s = suggest(name, list(BUILTIN) + list(comps))
            if s and s.lower() == str(name).lower():
                raw["use"] = s
                rep.add("info", "renamed-value", lid, f"component '{name}' read as '{s}'", f"write '{s}'")
            else:
                rep.add("error", "bad-component", lid, unknown("component", name, list(BUILTIN) + list(comps)))
        for key in ("enter", "exit"):
            _motion(raw, key, rep, lid)
        _layer_sound(raw, rep, lid)
        return  # parameters are checked when the component expands (components.Params)
    t = raw.get("type")
    if isinstance(t, str) and t.strip().lower() in ENUMS["shape"] and t not in TYPES and "shape" not in raw:
        raw["type"], raw["shape"] = "model", t.strip().lower()
        rep.add("info", "renamed-value", lid, f"type '{t}' read as a model with shape '{raw['shape']}'",
                f"write \"type\": \"model\", \"shape\": \"{raw['shape']}\"")
        t = "model"
    if isinstance(t, str) and t not in TYPES and t != "sequence":
        s = t.strip()
        fixed = s if s in TYPES else TYPE_ALIAS.get(s.lower()) or (s.lower() if s.lower() in TYPES else None)
        if fixed:
            raw["type"] = fixed
            rep.add("info", "renamed-value", lid, f"type '{t}' read as '{fixed}'", f"write \"type\": \"{fixed}\"")
            t = fixed
        else:
            rep.add("error", "bad-type", lid, unknown("layer type", t, list(TYPES) + ["sequence"]))
            return
    if t == "sequence":
        _keys(raw, SEQUENCE, {"items": "clips", "videos": "clips", "children": "clips"}, rep, lid, "")
        _effects(raw, rep, lid)
        _layer_sound(raw, rep, lid)
        return
    if t is None:
        from .scene import _infer_type
        t = _infer_type(raw)
    lalias = dict(ALIAS, **TYPE_ALIASES.get(t, {}))
    if t == "group":
        lalias.pop("items", None)
    pos = raw.get("position") or raw.get("pos")
    if pos is not None and "x" not in raw and "y" not in raw:
        key = "position" if "position" in raw else "pos"
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            raw["x"], raw["y"] = pos
        elif isinstance(pos, dict):
            raw["x"], raw["y"] = pos.get("x", 0), pos.get("y", 0)
        del raw[key]
        rep.add("info", "renamed-key", lid, f"'{key}' read as x and y", "write \"x\" and \"y\"")
    _keys(raw, COMMON | TYPES[t], lalias, rep, lid, "")
    shape = raw.get("shape")
    if t == "model" and isinstance(shape, str) and shape.lower().endswith((".glb", ".gltf", ".obj")):
        raw["src"] = raw.pop("shape")
        rep.add("info", "renamed-key", lid, f"shape '{shape}' read as src (a model file)",
                f"write \"src\": \"{shape}\"")
    for need in [REQUIRED.get(t)]:
        if need and not any(raw.get(k) not in (None, "", []) for k in need):
            rep.add("error", "missing-key", lid, f"a {t} layer needs " + " or ".join(f"'{k}'" for k in need),
                    {"text": "\"text\": \"Hello\"", "image": "\"src\": \"photo.jpg\"", "video": "\"src\": \"clip.mp4\""}
                    .get(t, ""))
    _values(raw, t, rep, lid)
    _effects(raw, rep, lid)
    _layer_sound(raw, rep, lid)
    if isinstance(raw.get("box"), dict):
        _keys(raw["box"], BOX, BOX_ALIAS, rep, lid, "box: ")
    if t == "shader" and raw.get("preset"):
        from .shaders import PRESETS as SH
        if raw["preset"] not in SH:
            s = suggest(raw["preset"], SH)
            if s and s.lower() == str(raw["preset"]).lower():
                raw["preset"] = s
            else:
                rep.add("error", "bad-value", lid, "preset: " + unknown("shader preset", raw["preset"], SH))
    if t == "model":
        _model(raw, rep, lid)
    kids = raw.get("children") if raw.get("children") is not None else raw.get("layers")
    if t == "group" and isinstance(kids, list):
        for i, k in enumerate(kids):
            _layer(k, rep, f"{lid}.children[{i}]", comps)


def _effects(raw: dict, rep: Report, lid):
    """Motions, shadow and fx, which any layer and a sequence take."""
    for key in ("enter", "exit"):
        _motion(raw, key, rep, lid)
    if isinstance(raw.get("shadow"), dict):
        _keys(raw["shadow"], SHADOW, SHADOW_ALIAS, rep, lid, "shadow: ")
    if isinstance(raw.get("fx"), dict):
        _keys(raw["fx"], FX, {"saturation": "saturate", "grayscale": "gray", "greyscale": "gray", "hueRotate": "hue",
                              "warmth": "temperature"}, rep, lid, "fx: ")


def _light(obj: dict, rep: Report, lid):
    """`light` is a preset name or an object of light amounts; both are checked here."""
    from .render3d import FROM, LIGHTS
    v = obj.get("light")
    if isinstance(v, str):
        s = v if v in LIGHTS else suggest(v, LIGHTS)
        if s and s.lower() == v.lower():
            obj["light"] = s
        else:
            rep.add("error", "bad-value", lid, "light: " + unknown("light preset", v, LIGHTS))
    elif isinstance(v, dict):
        _keys(v, LIGHT, LIGHT_ALIAS, rep, lid, "light: ")
        if v.get("preset") is not None and v["preset"] not in LIGHTS:
            rep.add("error", "bad-value", lid, "light preset: " + unknown("light preset", v["preset"], LIGHTS))
        if v.get("from") is not None and v["from"] not in FROM:
            rep.add("error", "bad-value", lid, "light from: " + unknown("direction", v["from"], FROM))
        for k in ("key", "fill", "ambient", "rim"):
            if k in v and (isinstance(v[k], bool) or not isinstance(v[k], (int, float))):
                rep.add("error", "bad-value", lid, f"light {k} is {v[k]!r}; it takes a number (about 0 to 1)")
        if isinstance(v.get("color"), str) and not C.is_color(v["color"]):
            rep.add("error", "bad-colour", lid, f"light color '{v['color']}' is not a colour")
    elif v is not None:
        rep.add("error", "bad-value", lid, f"light must be a preset ({', '.join(LIGHTS)}) or an object, got {v!r}")


def _outline(obj: dict, rep: Report, lid, where: str = ""):
    """Text and path shapes need their outline; a number in `d` is a depth (d is the path data here)."""
    d = obj.get("d")
    if isinstance(d, (int, float)) and not isinstance(d, bool) and "depth" not in obj:
        obj["depth"] = obj.pop("d")
        rep.add("info", "renamed-key", lid, f"{where}'d' read as 'depth'", "write 'depth'")
    shape = obj.get("shape")
    if shape == "text" and not str(obj.get("text", "")).strip():
        rep.add("error", "missing-key", lid, f"{where}a text model needs 'text'",
                'add "text": "..."')
    if shape == "path" and not isinstance(obj.get("d"), str):
        rep.add("error", "missing-key", lid, f"{where}a path model needs 'd', SVG path data",
                'add "d": "M0 0 L100 0 L50 80 Z"')


def _model(raw: dict, rep: Report, lid):
    _light(raw, rep, lid)
    _outline(raw, rep, lid)
    if raw.get("src") and raw.get("shape"):
        rep.add("warning", "ignored-key", lid, f"shape '{raw['shape']}' is ignored: the model comes from src",
                "drop shape, or make the shape its own part")
    gr = raw.get("ground")
    if isinstance(gr, dict):
        _keys(gr, GROUND, GROUND_ALIAS, rep, lid, "ground: ")
        for k, v in gr.items():
            if k in GROUND and (isinstance(v, bool) or not isinstance(v, (int, float))):
                rep.add("error", "bad-value", lid, f"ground {k} is {v!r}; it takes a number")
    elif gr is not None and not isinstance(gr, bool):
        rep.add("error", "bad-value", lid, f"ground is {gr!r}; it takes true or an object like {{\"opacity\": 0.5}}")
    if isinstance(raw.get("fill"), dict):
        rep.add("error", "bad-value", lid, "a model's fill is one colour; gradients do not apply to models",
                "use a colour, and give parts their own fill for more than one")
    parts = raw.get("parts")
    if parts is not None and not isinstance(parts, list):
        rep.add("error", "bad-value", lid, "parts must be a list of objects, each with a shape")
        return
    for i, p in enumerate(parts or []):
        where = f"parts[{i}]: "
        if not isinstance(p, dict):
            rep.add("error", "bad-value", lid, f"{where}a part is an object like {{\"shape\": \"sphere\", \"r\": 40}}")
            continue
        _keys(p, PART, dict(ALIAS, **MODEL_ALIAS), rep, lid, where)
        _values(p, "model", rep, lid)
        _outline(p, rep, lid, where)
        for k, v in p.items():  # what _values lets through for layers but a part cannot animate or resolve
            if k in SCALAR and (isinstance(v, (list, dict)) or (isinstance(v, str) and (v.startswith("=") or _lengthish(v)))):
                rep.add("error", "bad-value", lid, f"{where}{k} is {v!r}; parts take plain numbers (they do not animate)",
                        "animate the model layer, or make the part its own model layer")


def _audio_values(a: dict, rep: Report, lid, where: str):
    _enum(a, "role", rep, lid)
    p = a.get("pan")
    if p is not None and (isinstance(p, bool) or not isinstance(p, (int, float)) or not -1 <= p <= 1):
        rep.add("error", "bad-value", lid, f"{where}pan is {p!r}; it must be a number from -1 (left) to 1 (right)")
    if not a.get("src"):
        rep.add("error", "missing-key", lid, f"{where}an audio entry needs 'src'", "\"src\": \"music.mp3\"")


def _layer_sound(raw: dict, rep: Report, lid):
    s = raw.get("sound")
    if s is None or isinstance(s, str):
        return
    if not isinstance(s, dict):
        rep.add("error", "bad-value", lid, f"sound must be a file name or an object like {{\"src\": \"whoosh.wav\", "
                f"\"at\": \"out\"}}, got {type(s).__name__}")
        return
    _keys(s, SOUND, AUDIO_ALIAS, rep, lid, "sound: ")
    _audio_values(s, rep, lid, "sound: ")


def normalize(spec) -> tuple[dict, Report]:
    """Return a cleaned copy of `spec` and a Report of notes and errors (nothing is raised here)."""
    rep = Report()
    if not isinstance(spec, dict):
        rep.add("error", "bad-scene", None, f"a scene is a JSON object with 'canvas' and 'layers', got {type(spec).__name__}")
        return {}, rep
    spec = copy.deepcopy(spec)
    cv = spec.get("canvas")
    if cv is None:
        cv = spec["canvas"] = {}
    for k in [k for k in spec if k in ("width", "height", "fps", "bg", "background", "backgroundColor",
                                       "duration", "frameRate", "w", "h")]:
        to = CANVAS_ALIAS.get(k, k)
        if isinstance(cv, dict) and to not in cv:
            cv[to] = spec.pop(k)
            rep.add("info", "renamed-key", None, f"top-level '{k}' read as canvas.{to}", "put it inside \"canvas\"")
        elif isinstance(cv, dict):
            spec.pop(k)
            rep.add("error", "duplicate-key", None, f"top-level '{k}' belongs inside canvas as '{to}', and canvas.{to} "
                    "is already set", f"remove the top-level '{k}' or set canvas.{to}")
    for k in list(spec):
        if k in TOP_ALIAS and TOP_ALIAS[k] not in spec and (TOP_ALIAS[k] != "layers" or isinstance(spec[k], list)):
            spec[TOP_ALIAS[k]] = spec.pop(k)
            rep.add("info", "renamed-key", None, f"'{k}' read as '{TOP_ALIAS[k]}'", f"write '{TOP_ALIAS[k]}'")
    _keys(spec, TOP, {}, rep, None, "scene: ")
    if not isinstance(cv, dict):
        rep.add("error", "bad-canvas", None, f"canvas must be an object like {{\"w\": 1920, \"h\": 1080, \"fps\": 30}}, "
                f"got {type(cv).__name__}")
        spec["canvas"] = cv = {}
    _keys(cv, CANVAS, CANVAS_ALIAS, rep, None, "canvas: ")
    for k, lo, hi in (("w", 16, 16384), ("h", 16, 16384), ("fps", 1, 240)):
        v = cv.get(k)
        if v is None:
            continue
        if isinstance(v, str) and _lengthish(v) and not v.strip().endswith("%"):
            v = cv[k] = float(re.sub(r"[a-z]+", "", v))
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            rep.add("error", "bad-canvas", None, f"canvas.{k} must be a number, got {v!r}")
        elif not lo <= v <= hi:
            rep.add("error", "bad-canvas", None, f"canvas.{k} is {v:g}; it must be between {lo} and {hi}")
    d = cv.get("duration")
    if isinstance(d, (int, float)) and not isinstance(d, bool) and d <= 0:
        rep.add("error", "bad-canvas", None, f"canvas.duration must be greater than 0 (got {d})",
                "use seconds, or \"auto\" to end with the last layer")
    if cv.get("bg") is not None and isinstance(cv["bg"], str) and not cv["bg"].startswith("$") \
            and not C.is_color(cv["bg"]) and cv["bg"] not in ("transparent", "none"):
        rep.add("error", "bad-colour", None, f"canvas.bg '{cv['bg']}' is not a colour", "use #rrggbb")
    if isinstance(spec.get("grade"), dict):
        _keys(spec["grade"], GRADE, {"saturation": "saturate"}, rep, None, "grade: ")
    if isinstance(spec.get("motionBlur"), dict):
        _keys(spec["motionBlur"], MOTION_BLUR, {}, rep, None, "motionBlur: ")
    cam = spec.get("camera")
    if isinstance(cam, dict):
        _keys(cam, CAMERA, CAMERA_ALIAS, rep, "camera", "camera: ")
        _values(cam, "camera", rep, "camera")
        fov = cam.get("fov")
        if isinstance(fov, (int, float)) and not isinstance(fov, bool) and not 5 <= fov <= 150:
            rep.add("error", "bad-value", "camera", f"fov is {fov:g}; it must be between 5 and 150 degrees",
                    "40 is the default; lower flattens perspective, higher exaggerates it")
    elif cam is not None:
        rep.add("error", "bad-value", "camera", f"camera must be an object like {{\"z\": -200, \"rotateY\": 10}}, "
                f"got {type(cam).__name__}")
        spec.pop("camera")
    _light(spec, rep, None)
    audio = spec.get("audio")
    if isinstance(audio, dict):
        audio = spec["audio"] = [audio]
    for i, a in enumerate(audio or []):
        if isinstance(a, str):
            audio[i] = a = {"src": a}
        if isinstance(a, dict):
            owner = a.get("id") or f"audio[{i}]"
            _keys(a, AUDIO, AUDIO_ALIAS, rep, owner, "")
            _audio_values(a, rep, owner, "")
        else:
            rep.add("error", "bad-value", f"audio[{i}]", f"an audio entry is a file name or an object with 'src', "
                    f"got {type(a).__name__}")
    layers = spec.get("layers")
    if layers is None or layers == []:
        rep.add("error", "no-layers", None, "the scene has no layers", "add \"layers\": [ ... ] with at least one layer")
        spec["layers"] = []
    elif not isinstance(layers, list):
        rep.add("error", "bad-layers", None, f"'layers' must be a list, got {type(layers).__name__}")
        spec["layers"] = []
    comps = spec.get("components") or {}
    for i, raw in enumerate(spec["layers"]):
        _layer(raw, rep, f"layers[{i}]", comps if isinstance(comps, dict) else {})
    return spec, rep


# ---- tolerant JSON ---------------------------------------------------------------------------
def loads(text: str) -> tuple[object, list[str]]:
    """json.loads that also accepts // and /* */ comments, trailing commas and 'single quotes'.
    Returns (value, notes); raises json.JSONDecodeError for anything else."""
    import json
    try:
        return json.loads(text), []
    except json.JSONDecodeError as first:
        out, i, n, notes = [], 0, len(text), set()
        while i < n:
            ch = text[i]
            if ch in "\"'":
                q, j, buf = ch, i + 1, []
                while j < n and text[j] != q:
                    if text[j] == "\\" and j + 1 < n:
                        buf.append(text[j:j + 2])
                        j += 2
                        continue
                    buf.append('\\"' if text[j] == '"' else text[j])
                    j += 1
                if q == "'":
                    notes.add("single quotes")
                out.append('"' + "".join(buf) + '"')
                i = j + 1
            elif text.startswith("//", i):
                notes.add("comments")
                while i < n and text[i] != "\n":
                    i += 1
            elif text.startswith("/*", i):
                notes.add("comments")
                e = text.find("*/", i + 2)
                i = n if e < 0 else e + 2
            elif ch == ",":
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j < n and text[j] in "]}":
                    notes.add("trailing commas")
                    i += 1
                    continue
                out.append(ch)
                i += 1
            else:
                out.append(ch)
                i += 1
        try:
            return json.loads("".join(out)), sorted(notes)
        except json.JSONDecodeError:
            raise first from None
