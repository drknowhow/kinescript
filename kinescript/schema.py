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
import json
import math
import re

from . import color as C
from . import ease as E
from .props import _LEN, _NUM, KEY_TIMES, _is_time
from .timing import KEYWORDS, LIMIT, finite as _finite_secs, parse_clock, show

# Keys that are always allowed and ignored: notes to self.
COMMENT = {"note", "notes", "comment", "comments", "description", "name"}

TOP = {"canvas", "theme", "markers", "grade", "motionBlur", "audio", "components", "layers", "meta", "version",
       "$schema", "camera", "light", "lights", "env", "fog", "mix"}
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
MOTION = {"name", "dur", "ease", "stagger", "dist", "from", "amount", "deg", "order", "seed"}
MOTION_ALIAS = {"type": "name", "preset": "name", "duration": "dur", "easing": "ease", "distance": "dist",
                "delay": "stagger"}
TWEEN = {"from", "to", "dur", "delay", "ease", "keys", "smooth"}
TWEEN_ALIAS = {"duration": "dur", "easing": "ease", "start": "from", "end": "to", "begin": "from"}
GRADIENT = {"linear", "radial", "sweep", "angle", "stops"}
AUDIO = {"src", "at", "trim", "gain", "volume", "fadeIn", "fadeOut", "id", "role", "dur", "until", "loop", "pan",
         "words", "text", "speaker", "rate", "backend", "options"}
AUDIO_ALIAS = {"source": "src", "file": "src", "path": "src", "start": "at", "fade_in": "fadeIn",
               "fade_out": "fadeOut", "fadein": "fadeIn", "fadeout": "fadeOut", "duration": "dur", "length": "dur",
               "end": "until", "type": "role", "kind": "role", "balance": "pan", "repeat": "loop",
               "script": "text", "say": "text", "tts": "text", "narrator": "speaker", "pace": "rate",
               "engine": "backend", "settings": "options", "voiceSettings": "options"}
# A layer's "sound": cued from the layer, so it has no id, until, loop, words or text of its own.
SOUND = AUDIO - {"id", "until", "loop", "words", "text", "speaker", "rate", "backend", "options"}
MIX = {"loudness", "peak", "levels", "duck"}
MIX_ALIAS = {"lufs": "loudness", "target": "loudness", "ceiling": "peak", "truePeak": "peak", "ducking": "duck",
             "level": "levels"}
DUCK = {"by", "under", "depth", "attack", "hold", "release"}
DUCK_ALIAS = {"amount": "depth", "db": "depth", "key": "by", "trigger": "by", "targets": "under", "duck": "under"}
MOTION_BLUR = {"samples", "shutter"}
CAMERA = {"x", "y", "z", "rotateX", "rotateY", "rotate", "fov", "target", "shake", "focus", "aperture"}
CAMERA_ALIAS = {"rotationX": "rotateX", "rotationY": "rotateY", "rotateZ": "rotate", "rotationZ": "rotate",
                "rotation": "rotate", "roll": "rotate", "orbit": "rotateY", "tilt": "rotateX", "dolly": "z",
                "fieldOfView": "fov", "lookAt": "target", "follow": "target", "subject": "target",
                "handheld": "shake", "wobble": "shake", "focusDistance": "focus", "focalPlane": "focus",
                "focusOn": "focus", "dof": "aperture", "defocus": "aperture", "bokeh": "aperture",
                "lensBlur": "aperture"}
SHAKE = {"amount", "speed", "seed"}
SHAKE_ALIAS = {"strength": "amount", "intensity": "amount", "px": "amount", "frequency": "speed", "freq": "speed",
               "hz": "speed", "rate": "speed"}
LIGHT = {"preset", "key", "fill", "ambient", "rim", "from", "color", "azimuth", "elevation", "follow"}
LIGHT_ALIAS = {"colour": "color", "direction": "from", "dir": "from", "position": "from", "intensity": "key",
               "ambience": "ambient", "name": "preset", "angle": "azimuth", "height": "elevation",
               "altitude": "elevation"}
# One of "lights", the lights past the key: directional (from, azimuth, elevation, follow) or point (x, y, z, range).
MORE_LIGHT = {"type", "from", "azimuth", "elevation", "follow", "x", "y", "z", "color", "intensity", "range"}
MORE_LIGHT_ALIAS = {"colour": "color", "kind": "type", "direction": "from", "dir": "from", "strength": "intensity",
                    "amount": "intensity", "falloff": "range", "angle": "azimuth", "height": "elevation"}
ENV = {"preset", "top", "horizon", "bottom", "amount"}
ENV_ALIAS = {"name": "preset", "sky": "top", "ground": "bottom", "floor": "bottom", "middle": "horizon",
             "strength": "amount", "intensity": "amount"}
FOG = {"color", "near", "far", "amount"}
FOG_ALIAS = {"colour": "color", "start": "near", "end": "far", "density": "amount", "strength": "amount"}
MATERIAL = {"fill", "shine", "metal", "rim", "bands", "glow", "texture", "transparency"}
TEXTURE = {"src", "repeat", "offset", "loop", "trim", "speed"}
CLIP_KEYS = ("loop", "trim", "speed")  # a video texture's, as on a video layer
TEXTURE_ALIAS = {"image": "src", "file": "src", "path": "src", "url": "src", "tile": "repeat", "tiles": "repeat",
                 "tiling": "repeat", "shift": "offset", "position": "offset"}
ALONG = {"d", "p", "orient"}
ALONG_ALIAS = {"path": "d", "route": "d", "progress": "p", "autoRotate": "orient", "autoOrient": "orient",
               "rotate": "orient"}
PART = MATERIAL | {"shape", "w", "h", "depth", "r", "tube", "x", "y", "z", "rotateX", "rotateY", "rotate", "scale",
                   "text", "size", "font", "weight", "italic", "letterSpacing", "lineHeight", "d", "strokeWidth",
                   "bevel", "profile"}
MODEL_ALIAS = {"radius": "r", "thickness": "tube", "geometry": "shape", "primitive": "shape", "extrude": "depth",
               "extrusion": "depth", "chamfer": "bevel",
               "mesh": "shape", "gloss": "shine", "glossiness": "shine", "shininess": "shine",
               "specular": "shine", "metallic": "metal", "metalness": "metal", "toon": "bands", "steps": "bands",
               "quality": "detail", "lighting": "light", "children": "parts", "meshes": "parts", "model": "src",
               "floor": "ground", "groundShadow": "ground", "contactShadow": "ground", "map": "texture",
               "image": "texture", "picture": "texture", "decal": "texture", "skin": "texture", "outline": "edges",
               "outlines": "edges", "lines": "edges", "creases": "edges", "transmission": "transparency",
               "seeThrough": "transparency", "translucency": "transparency", "copies": "repeat",
               "instances": "repeat", "array": "repeat", "clones": "repeat", "grid": "repeat"}
ANIMATION = {"name", "speed", "loop", "start"}  # a model file's clip, played on the layer's clock
ANIMATION_ALIAS = {"clip": "name", "track": "name", "rate": "speed", "timeScale": "speed", "repeat": "loop",
                   "offset": "start", "from": "start", "time": "start"}
MODEL_FILES = (".glb", ".gltf", ".obj")
# A model's copies: how many, and each one's place and fill (a number or colour, a list in turn, an expression of i).
REPEAT = {"count", "x", "y", "z", "rotateX", "rotateY", "rotate", "scale", "fill"}
REPEAT_ALIAS = {"n": "count", "copies": "count", "number": "count", "times": "count", "amount": "count",
                "color": "fill", "colour": "fill", "colors": "fill", "colours": "fill", "rotation": "rotate",
                "rotationX": "rotateX", "rotationY": "rotateY", "rotationZ": "rotate", "rotateZ": "rotate"}
MAX_COPIES = 10_000
COPY_SLOTS = 64  # the materials a model with copies draws: its own, and one per copy fill for those without a fill
GROUND = {"y", "opacity", "blur", "from"}
EDGES = {"width", "color", "angle"}
EDGES_ALIAS = {"colour": "color", "thickness": "width", "size": "width", "strokeWidth": "width", "crease": "angle",
               "threshold": "angle", "creaseAngle": "angle"}
SHADOWS = {"opacity", "soft"}
SHADOWS_ALIAS = {"alpha": "opacity", "strength": "opacity", "softness": "soft", "blur": "soft"}
GROUND_ALIAS = {"alpha": "opacity", "strength": "opacity", "softness": "blur", "radius": "blur", "level": "y"}

COMMON = {"id", "type", "at", "dur", "until", "x", "y", "anchor", "opacity", "scale", "scaleX", "scaleY", "rotate",
          "blur", "blend", "shadow", "fx", "enter", "exit", "hidden", "allowOffscreen", "z", "rotateX", "rotateY",
          "backface", "fixed", "sound", "along"}
SHAPE = {"fill", "stroke", "strokeWidth", "draw", "drawStart", "cap", "join", "dash", "dashOffset", "rough"}
ROUGH = {"amount", "length", "seed", "boil"}
ROUGH_ALIAS = {"roughness": "amount", "jitter": "amount", "wobble": "amount", "segment": "length",
               "size": "length", "rate": "boil", "fps": "boil"}
TEXT = {"text", "size", "weight", "font", "italic", "fill", "emph", "align", "maxWidth", "lineHeight",
        "letterSpacing", "stroke", "strokeWidth", "box", "case", "textCase", "caret", "split",
        "unitX", "unitY", "unitRotate", "unitScale", "unitOpacity"}
MEDIA = {"src", "w", "h", "fit", "zoom", "focus", "focusX", "focusY", "radius", "stroke", "strokeWidth", "cap", "join",
         "dash", "dashOffset"}
TYPES: dict[str, set[str]] = {
    "rect": SHAPE | {"w", "h", "radius", "dimOutside"},
    "circle": SHAPE | {"r"},
    "ellipse": SHAPE | {"w", "h"},
    "polygon": SHAPE | {"sides", "r", "inner", "corner"},
    "path": SHAPE | {"d", "points", "smooth", "closed", "corner"},
    "line": SHAPE | {"points", "smooth", "closed", "corner"},
    "arrow": SHAPE | {"x1", "y1", "x2", "y2", "bend", "head"},
    "text": TEXT,
    "counter": TEXT | {"value", "format"},
    "image": MEDIA,
    "video": MEDIA | {"trim", "speed", "volume", "loop", "fadeIn", "fadeOut"},
    "captions": TEXT | {"src", "cues", "mode", "active", "voice"},
    "shader": {"preset", "uniforms", "sksl", "w", "h"},
    "group": {"children", "layers", "w", "h", "clip", "radius", "fill", "sort", "shadows", "speed", "loop",
              "timeRemap", "combine"} | SHAPE,
    "model": PART - {"x", "y", "z", "rotateX", "rotateY", "rotate", "scale"} | {"detail", "parts", "explode",
                                                                              "draw", "light", "lights", "env",
                                                                              "src", "ground", "edges",
                                                                              "animation", "repeat"},
}
SEQUENCE = COMMON | {"clips", "transition", "fit", "volume", "w", "h"}
TRANSITION = {"type", "dur"}
TRANSITION_ALIAS = {"name": "type", "kind": "type", "style": "type", "duration": "dur", "length": "dur"}
TRANSITIONS = ("fade", "wipe", "slide", "zoom", "blur", "cut")
TRANSITION_VALUE_ALIAS = {"dissolve": "fade", "crossfade": "fade", "crossdissolve": "fade", "xfade": "fade",
                          "mix": "fade", "none": "cut", "hard": "cut", "hardcut": "cut", "push": "slide"}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")  # a sequence clip with one of these is an image
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
         "strokeDasharray": "dash", "dasharray": "dash", "dashArray": "dash", "strokeDashoffset": "dashOffset",
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
    "rect": {"corner": "radius"},
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
    "sort": ("order", "depth", "pixel"),
    "combine": ("union", "subtract", "intersect", "xor"),
    "shape": ("cube", "sphere", "cylinder", "cone", "torus", "plane", "capsule", "lathe", "tube", "text", "path"),
    "detail": ("low", "medium", "high"),
    "role": ("voice", "music", "sfx", "ambience"),
}
ENUM_ALIAS = {"middle": "center", "centre": "center", "centered": "center", "mid": "center",
              "characters": "chars", "letters": "chars", "char": "chars", "word": "words", "line": "lines",
              "uppercase": "upper", "lowercase": "lower", "fill": "cover", "crop": "cover", "scale": "contain",
              "plus": "add", "additive": "add", "soft-light": "softLight", "softlight": "softLight",
              "merge": "union", "minus": "subtract", "cut": "subtract", "intersection": "intersect",
              "exclude": "xor",
              "box": "cube", "cuboid": "cube", "ball": "sphere", "globe": "sphere", "donut": "torus",
              "doughnut": "torus", "ring": "torus", "can": "cylinder", "pipe": "cylinder", "pill": "capsule",
              "revolve": "lathe", "revolved": "lathe", "turned": "lathe", "spun": "lathe",
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
          "unitX", "unitY", "unitRotate", "unitScale", "unitOpacity", "z", "rotateX", "rotateY", "fov",
          "depth", "tube", "shine", "metal", "rim", "bands", "explode", "bevel", "transparency", "timeRemap",
          "dashOffset", "corner"}
REQUIRED = {"text": ("text",), "image": ("src",), "video": ("src",), "path": ("d", "points"), "line": ("points",),
            "captions": ("cues", "src", "voice"), "shader": ("preset", "sksl"), "model": ("shape", "parts", "src")}
# Keys whose value is an object of their own: never a tween, even with a "from" key (a light's direction).
OBJECTS = {"light", "lights", "env", "ground", "shadow", "shadows", "box", "fx", "uniforms", "sound", "enter", "exit",
           "parts", "transition", "texture", "edges", "animation", "repeat", "along", "rough"}
# Props that cannot be below 0, and the least each may be.
AT_LEAST = {"radius": 0, "r": 0, "sides": 3, "blur": 0, "strokeWidth": 0, "size": 0, "corner": 0}
COLOUR_FIX = f"write {C.FORMATS}"
TIME_FORMS = 'seconds like 1.5, "45f" (frames), "0:02.5", "title.out+0.2" or a marker name'


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


def _show(v) -> str:
    """A value as the scene wrote it (JSON), shortened, for a message."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return show(v)
    try:
        s = json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _finite(v) -> bool:
    """A number (not a bool) that is finite; 10**400 is not, as no float holds it."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _colour(v, field: str, rep: Report, lid) -> bool:
    """Report a value that is not a colour (or a $theme name); warn when a channel is clamped. True when it is fine."""
    if isinstance(v, str) and v.startswith("$"):
        return True
    why = C.problem(v)
    if why:
        rep.add("error", "bad-colour", lid, f"{field}: {why}" if isinstance(v, str) else
                f"{field} is {_show(v)}; a colour is text like \"#ff633f\"", COLOUR_FIX)
        return False
    if note := C.clamped(v):
        rep.add("warning", "colour-clamped", lid, f"{field}: {note}", "keep red, green and blue from 0 to 255")
    return True


def _numbers(obj: dict, keys, where: str, rep: Report, lid, least=None):
    """Each of `keys` set in `obj` must be a finite number (at least `least` when given); null is the default."""
    for k in keys:
        v = obj.get(k)
        if v is None:
            continue
        if not _finite(v):
            rep.add("error", "bad-value", lid, f"{where}{k} is {_show(v)}; it takes a number")
        elif least is not None and v < least:
            rep.add("error", "bad-value", lid, f"{where}{k} is {_show(v)}; it must be {least} or more")


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
    if key not in raw:
        return
    v = raw[key]
    opts = ENUMS[key]
    if v is None:
        del raw[key]  # null is the default
        return
    if not isinstance(v, str):
        rep.add("error", "bad-value", lid, f"{key} is {_show(v)}; it is one of {', '.join(opts)}")
        return
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
    for k, v in list(raw.items()):
        if k in OBJECTS:
            continue
        if v is None and (k in SCALAR or k in COLOR_KEYS):
            del raw[k]  # null is the default
            continue
        if k in SCALAR and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) and (
                k != "radius"):  # a radius list is its corners, checked below
            rep.add("error", "flat-keyframes", lid,
                    f"{k}: {v} is a flat list. Keyframes are a list of [time, value] pairs",
                    f"\"{k}\": [[0, {v[0]}], [\"end\", {v[-1]}]]")
            continue
        if k in ("fill", "stroke") and isinstance(v, str) and (g := css_gradient(v)) is not None:
            raw[k] = v = g
            rep.add("info", "renamed-value", lid, f"{k}: CSS gradient read as {g}", "write the gradient object")
        if isinstance(v, dict) and not _tween(v) and k in ("fill", "stroke"):
            if ltype != "model":  # _model says a model takes no gradient
                gradient(v, k, rep, lid)
        elif k in SCALAR or k in COLOR_KEYS or (isinstance(v, dict) and _tween(v)):
            _prop(k, v, rep, lid)


def _tween(v: dict) -> bool:
    return "to" in v or "keys" in v or "from" in v


def _keyframes(v) -> bool:
    """Shaped like keyframes: a list of [time, value] or [time, value, ease] (the times are checked apart)."""
    return isinstance(v, list) and bool(v) and all(isinstance(e, (list, tuple)) and 2 <= len(e) <= 3 for e in v)


def _prop(k: str, v, rep: Report, lid):
    """One animatable value: a constant, keyframes or a tween, each value in it checked for what `k` takes."""
    if isinstance(v, str) and v.startswith(("=", "$")):
        return
    if isinstance(v, dict):
        _keys(v, TWEEN, TWEEN_ALIAS, rep, lid, f"{k} tween: ")
        if "smooth" in v:
            if not isinstance(v["smooth"], bool) or "keys" not in v:
                rep.add("error", "bad-value", lid, f"{k}: smooth is true or false, beside keys",
                        f"\"{k}\": {{\"keys\": [[0, 0], [1, 100], [2, 300]], \"smooth\": true}}")
                return
            if v["smooth"] and k in COLOR_KEYS:
                rep.add("error", "bad-value", lid, f"{k}: smooth moves numbers through their keys; colours mix key "
                        "to key", "leave smooth out")
                return
        if "keys" in v:
            if not _keyframes(v["keys"]):
                rep.add("error", "bad-value", lid, f"{k} tween: keys is {_show(v['keys'])}; it is keyframes, a list of "
                        "[time, value] pairs", f"\"keys\": [[0, 0], [1, 100]]")
            else:
                _prop(k, v["keys"], rep, lid)
            return
        if "to" not in v:
            rep.add("error", "bad-value", lid, f"{k} tween has no 'to', the value it ends on",
                    f"\"{k}\": {{\"from\": 0, \"to\": 1, \"dur\": 0.6}}")
        for key, default in (("dur", 0.6), ("delay", 0)):
            d = v.get(key)
            if d is not None and (not _finite(d) or (key == "dur" and d < 0)):
                rep.add("error", "bad-value", lid, f"{k} tween: {key} is {_show(d)}; it takes a number of seconds"
                        + (", 0 or more" if key == "dur" else "") + f" ({default:g} when left out)")
        if v.get("ease") not in (None, "") and (why := E.problem(v["ease"])):
            rep.add("error", "bad-ease", lid, f"{k} tween: {why}")
        values = [(f" as the tween's {e}", v[e]) for e in ("from", "to") if e in v]
    elif _keyframes(v):
        values = []
        for j, key in enumerate(v, 1):
            t = key[0]
            if not _is_time(t) or (isinstance(t, (int, float)) and not _finite_secs(t)):
                rep.add("error", "bad-keyframe", lid, f"{k}: keyframe {j} has time {_show(t)}, which is not a time; "
                        f"a keyframe time is {KEY_TIMES}", "\"x\": [[0, 200], [\"500ms\", 960], [\"end\", 1700]]")
            if len(key) == 3 and key[2] not in (None, "") and (why := E.problem(key[2])):
                rep.add("error", "bad-ease", lid, f"{k}: keyframe {j}: {why}")
            values.append((f" at keyframe {j}", key[1]))
    else:
        values = [("", v)]
    for where, x in values:
        _constant(k, x, where, rep, lid)
    grads = [x for _, x in values if isinstance(x, dict)]
    if grads and (len(grads) < len(values) or len({(_kind(x), len(x.get("stops") or [])) for x in grads}) > 1):
        rep.add("error", "bad-value", lid, f"{k}: a gradient animates only into a gradient of the same kind (linear, "
                "radial, sweep, or an angle) with as many stops", "give every key the same kind of gradient and the "
                "same number of stops")


def _constant(k: str, x, where: str, rep: Report, lid):
    if isinstance(x, str) and x.startswith(("=", "$")):
        return
    if k == "radius" and isinstance(x, list):
        if len(x) != 4 or not all(_finite(r) and r >= 0 for r in x):
            rep.add("error", "bad-value", lid, f"radius{where} is {_show(x)}; it is one number, or four for the "
                    "corners: [top-left, top-right, bottom-right, bottom-left], each 0 or more",
                    '"radius": [24, 24, 0, 0] rounds the top corners')
    elif k in SCALAR or k == "along p":
        if isinstance(x, bool):
            rep.add("error", "bad-value", lid, f"{k}{where} is {str(x).lower()}, which is not a number", "use a number")
        elif isinstance(x, (int, float)):
            if not _finite(x):
                rep.add("error", "bad-value", lid, f"{k}{where} is {show(x)}; it must be a finite number")
            elif k in AT_LEAST and x < AT_LEAST[k]:
                rep.add("error", "bad-value", lid, f"{k}{where} is {show(x)}; it must be {AT_LEAST[k]} or more"
                        + (" (a polygon has at least 3 sides)" if k == "sides" else ""))
        elif isinstance(x, str) and not _lengthish(x):
            rep.add("error", "bad-value", lid, f"{k} is '{x}'{where}, which is not a number",
                    f"use a number{', a percentage like 50%' if k in ('x', 'y', 'w', 'h', 'maxWidth') else ''} "
                    "or an expression starting with '='")
    elif k in COLOR_KEYS and x is not None:
        if isinstance(x, dict) and not where:
            return  # a gradient, checked by the caller
        if isinstance(x, dict):  # a gradient that animates
            gradient(x, f"{k}{where}", rep, lid)
        else:
            _colour(x, f"{k}{where}", rep, lid)


def _kind(g: dict) -> str:
    """A gradient's kind, as Gradient.of reads it."""
    for k in ("radial", "sweep"):
        if k in g:
            return k
    return "angle" if "linear" not in g and g.get("angle") is not None else "linear"


def gradient(g: dict, where: str, rep: Report, lid):
    """A gradient object: 2 or more stops, and a linear pair of points or a radial centre and radius."""
    _keys(g, GRADIENT, {"colors": "stops", "colours": "stops"}, rep, lid, f"{where} gradient: ")
    example = '{"linear": [[0, 0], [1, 1]], "stops": ["#ff633f", "#b9f45d"]}'
    stops = g.get("stops")
    if stops is None:
        rep.add("error", "bad-value", lid, f"{where}: a gradient needs 'stops'", example)
    elif not isinstance(stops, list) or len(stops) < 2:
        rep.add("error", "bad-value", lid, f"{where} gradient: stops is {_show(stops)}; it is a list of 2 or more "
                "colours, or of [position, colour] pairs with positions from 0 to 1", example)
    else:
        pairs = [isinstance(s, (list, tuple)) for s in stops]
        if any(pairs) and not all(pairs):
            rep.add("error", "bad-value", lid, f"{where} gradient: stops mixes plain colours and [position, colour] "
                    "pairs; give every stop a position, or none",
                    '"stops": [[0, "#ff633f"], [0.7, "#7b5cff"], [1, "#b9f45d"]]')
        for j, s in enumerate(stops, 1):
            if isinstance(s, (list, tuple)):
                if len(s) != 2 or not _finite(s[0]):
                    rep.add("error", "bad-value", lid, f"{where} gradient: stop {j} is {_show(s)}; a stop with a "
                            "position is [position from 0 to 1, colour]")
                    continue
                if not 0 <= s[0] <= 1:  # Skia pins it into 0..1: a CSS "#00f 150%" is drawn at the end
                    rep.add("warning", "gradient-clamped", lid, f"{where} gradient: stop {j} is at {_show(s[0])}, "
                            f"outside 0 to 1, so it is drawn at {min(1, max(0, s[0])):g}",
                            "positions run from 0 (start) to 1 (end)")
                s = s[1]
            _colour(s, f"{where} gradient stop {j}", rep, lid)
    lin, rad, swp, ang = g.get("linear"), g.get("radial"), g.get("sweep"), g.get("angle")
    kinds = [k for k in ("linear", "radial", "sweep") if g.get(k) is not None]
    if len(kinds) > 1:
        rep.add("error", "bad-value", lid, f"{where} gradient has {' and '.join(kinds)}; give one")
    if ang is not None and not _finite(ang):
        rep.add("error", "bad-value", lid, f"{where} gradient: angle is {_show(ang)}; it is degrees, 0 running up and "
                "90 to the right", '{"angle": 90, "stops": ["#ff633f", "#7b5cff"]}')
    elif ang is not None and (lin is not None or rad is not None):
        rep.add("warning", "ignored-key", lid, f"{where} gradient: angle is ignored beside "
                f"{'linear points' if lin is not None else 'radial'}", "give the angle alone, or the points alone")
    if swp is not None and swp is not True and not (isinstance(swp, list) and len(swp) == 2 and all(map(_finite, swp))):
        rep.add("error", "bad-value", lid, f"{where} gradient: sweep is {_show(swp)}; it is its middle [x, y], each "
                "from 0 to 1 of the layer's box", '"sweep": [0.5, 0.5]')
    if lin is not None and not (isinstance(lin, list) and len(lin) == 2 and all(
            isinstance(p, (list, tuple)) and len(p) == 2 and all(map(_finite, p)) for p in lin)):
        rep.add("error", "bad-value", lid, f"{where} gradient: linear is {_show(lin)}; it is two points [[x1, y1], "
                "[x2, y2]], each from 0 to 1 of the layer's box", '"linear": [[0, 0], [1, 1]]')
    if rad is not None and rad is not True and not (
            isinstance(rad, list) and 2 <= len(rad) <= 3 and all(map(_finite, rad))):
        rep.add("error", "bad-value", lid, f"{where} gradient: radial is {_show(rad)}; it is [x, y] or [x, y, radius], "
                "each from 0 to 1 of the layer's box", '"radial": [0.5, 0.5, 0.6]')


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


def _lengthish(v: str) -> bool:
    """A number or length written as text that props.length reads: "12", "1e3", "50%", "50%-120", "10vw"."""
    return bool(_LEN.match(v))


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
        from .presets import option_problem
        for k, x in v.items():
            if why := option_problem(k, x):
                rep.add("error", "bad-ease" if k == "ease" else "bad-value", lid, f"{key}: {why}")
    if isinstance(v, dict) and (v.get("order") not in (None, "start") or v.get("seed") is not None)             and not v.get("stagger") and not (raw.get("split") and raw.get("type") in ("text", "counter")):
        rep.add("warning", "ignored-key", lid, f"{key}: order sets who goes first in a stagger, and there is none: "
                "a group enters child by child only with a stagger", "add \"stagger\": 0.08")
    for n in str(spec.get("name", "fade")).split("+"):
        n = n.strip()
        if n and canonical(n) is None:
            rep.add("error", "bad-motion", lid, f"{key}: " + unknown("motion", n, PRESETS))


def _id(obj: dict, rep: Report, path: str) -> str:
    """The id issues are reported under: the layer's or audio entry's own id, or its path when it has none."""
    v = obj.get("id")
    if v is None or (isinstance(v, (str, int, float)) and not isinstance(v, bool) and not v):
        return path  # the scene makes one up, as for no id
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        rep.add("error", "bad-id", path, f"{path}: id is {_show(v)}; an id is a name such as \"title\"")
        return path
    s = str(v)
    if s in KEYWORDS:
        rep.add("error", "bad-id", s, f"id '{s}' is a word cues read first (\"start\" is 0 s, \"prev\" the layer "
                "listed before), so no cue could name this one",
                f"rename it, e.g. '{s}-1'")
    return s


def _timing(obj: dict, rep: Report, lid, where: str = ""):
    """at, until and dur, so a wrong type or an impossible number is an error that names the field, not a crash."""
    for k in ("at", "until"):
        v = obj.get(k)
        if v is None or isinstance(v, str):
            continue  # text is read by the timeline, which names the layer when it cannot
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            rep.add("error", "bad-time", lid, f"{where}{k} is {_show(v)}; it is a time: {TIME_FORMS}")
        elif not _finite_secs(v):
            rep.add("error", "bad-time", lid, f"{where}{k} is {show(v)}; it must be a finite number of seconds, at "
                    f"most {LIMIT:g}")
    d = obj.get("dur")
    secs = parse_clock(d, 30) if isinstance(d, str) else d
    if d is None:
        pass
    elif isinstance(d, str) and secs is None:
        rep.add("error", "bad-dur", lid, f"{where}dur is '{d}', which is not a length of time",
                "use seconds like 2.5, \"45f\" (frames) or \"0:02.5\"; to end with another layer use \"until\"")
    elif isinstance(secs, bool) or not isinstance(secs, (int, float)):
        rep.add("error", "bad-dur", lid, f"{where}dur is {_show(d)}; it is a length of time: seconds like 2.5, \"45f\" "
                "or \"0:02.5\"")
    elif not _finite_secs(secs):
        rep.add("error", "bad-dur", lid, f"{where}dur is {show(d)}; it must be a finite number of seconds, at most "
                f"{LIMIT:g}")
    elif secs <= 0:
        rep.add("error", "bad-dur", lid, f"{where}dur must be greater than 0 (got {_show(d)})",
                "give it a positive number of seconds, or remove dur to run to the end; "
                "to hide a layer use \"hidden\": true")
    if d is not None and obj.get("until") is not None:
        rep.add("warning", "ignored-key", lid, f"{where}dur and until are both set, so until is ignored",
                "keep one: dur (how long) or until (when it ends)")


def _text(obj: dict, rep: Report, lid, where: str = ""):
    """text is a string Skia can draw; a number is read as its digits."""
    v = obj.get("text")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        obj["text"] = s = json.dumps(v)
        rep.add("info", "renamed-value", lid, f"{where}text {s} read as the text \"{s}\"", f"write \"text\": \"{s}\"")
    elif v is not None and not isinstance(v, str):
        rep.add("error", "bad-value", lid, f"{where}text is {_show(v)}; it is the words to show, as a string",
                "\"text\": \"Hello\" (\\n starts a new line)")
    elif isinstance(v, str):
        try:
            v.encode("utf-8")
        except UnicodeEncodeError as e:
            rep.add("error", "bad-text", lid, f"{where}text has {e.object[e.start]!r} at character {e.start + 1}: half "
                    "of a surrogate pair, which is not a character on its own",
                    "write the character itself, or both halves of its JSON escape, like \\ud83d\\ude00")
    f = obj.get("font")
    if f is not None and not isinstance(f, str):
        rep.add("error", "bad-value", lid, f"{where}font is {_show(f)}; it is a font family such as \"Space Grotesk\", "
                "or a .ttf or .otf file")


def _box(raw: dict, rep: Report, lid):
    b = raw.get("box")
    if b is None or isinstance(b, bool):
        return
    if not isinstance(b, dict):
        rep.add("error", "bad-value", lid, f"box is {_show(b)}; it is true (a dark box behind the text) or an object "
                "like {\"fill\": \"#000000b0\", \"pad\": [12, 20], \"radius\": 10}")
        return
    _keys(b, BOX, BOX_ALIAS, rep, lid, "box: ")
    if b.get("fill") is not None:
        _colour(b["fill"], "box fill", rep, lid)
    pad = b.get("pad")
    if pad is not None and not (_finite(pad) and pad >= 0) and not (
            isinstance(pad, list) and len(pad) == 2 and all(_finite(p) and p >= 0 for p in pad)):
        rep.add("error", "bad-value", lid, f"box pad is {_show(pad)}; it is px of space around the text: one number, "
                "or [vertical, horizontal] like [12, 20]")
    _numbers(b, ("radius",), "box ", rep, lid, least=0)


def _layer(raw, rep: Report, path: str, comps: dict):
    if not isinstance(raw, dict):
        rep.add("error", "bad-layer", path, f"{path} must be an object, got {type(raw).__name__}")
        return
    lid = _id(raw, rep, path)
    for key in ("type", "use"):
        v = raw.get(key)
        if (key in raw if key == "use" else v is not None) and not isinstance(v, str):
            rep.add("error", "bad-type" if key == "type" else "bad-component", lid, f"{key} is {_show(v)}; it is a "
                    + ("layer type such as \"text\" or \"rect\"" if key == "type" else "component name such as "
                       "\"lowerThird\""))
            return
    if "use" not in raw and isinstance(raw.get("type"), str):
        from .components import BUILTIN
        name = suggest(raw["type"], BUILTIN)
        if name and name.lower() == raw["type"].strip().lower():
            raw["use"] = name
            del raw["type"]
            rep.add("info", "renamed-key", lid, f"type '{name}' is a component; read as \"use\": \"{name}\"",
                    f"write \"use\": \"{name}\"")
    for wrap in [k for k in raw if k in WRAPPERS and isinstance(raw[k], dict)
                 and not {"name", "type", "preset"} & set(raw[k])  # a motion object is not a wrapper
                 and not (k == "animation" and _model_file(raw))]:  # nor a model file's clip
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
        _timing(raw, rep, lid)
        return  # parameters are checked when the component expands (components.Params)
    t = raw.get("type")
    if isinstance(t, str) and t.strip().lower() in ENUMS["shape"] and t.strip().lower() not in TYPES \
            and "shape" not in raw:  # "Text" is a text layer, not a model with shape text
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
        _clips(raw, rep, lid, comps)
        _transition(raw, rep, lid)
        _volume(raw, rep, lid)
        _effects(raw, rep, lid)
        _layer_sound(raw, rep, lid)
        _timing(raw, rep, lid)
        return
    if t is None:
        from .scene import _infer_type
        t = _infer_type(raw)
    lalias = dict(ALIAS, **TYPE_ALIASES.get(t, {}))
    if t == "group":
        lalias.pop("items", None)
    key = "position" if "position" in raw else "pos"
    pos = raw.get(key)
    if pos is not None and "x" not in raw and "y" not in raw:
        del raw[key]
        axes = _position(pos)
        if axes is not None and not ("z" in axes and "z" in raw):
            raw.update(axes)
            rep.add("info", "renamed-key", lid, f"'{key}' read as {' and '.join(sorted(axes))}",
                    " and ".join(f"write \"{k}\"" for k in sorted(axes)))
        else:
            rep.add("error", "bad-value", lid, f"{key} is {_show(pos)}; it is read as x and y only when it is "
                    "[x, y], [x, y, z], {\"x\": ..., \"y\": ...}, or keyframes of points such as "
                    "[[0, [100, 300]], [2, [540, 60]]]", "write \"x\" and \"y\"")
    # "animation" plays a clip of a model file; on any other layer it is read as "enter", as before.
    _keys(raw, COMMON | TYPES[t] - (set() if _model_file(raw) else {"animation"}), lalias, rep, lid, "")
    _timing(raw, rep, lid)
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
    if raw.get("src") not in (None, "") and not isinstance(raw["src"], str):
        rep.add("error", "bad-value", lid, f"src is {raw['src']!r}; it is one file name, as a string",
                f"\"src\": \"{ {'image': 'photo.jpg', 'captions': 'captions.srt', 'model': 'model.glb'}.get(t, 'clip.mp4')}\" "
                "(a path relative to the scene file)")
    if t == "video":
        _volume(raw, rep, lid)
    _values(raw, t, rep, lid)
    _structure(raw, t, rep, lid)
    _stroke(raw, t, rep, lid)
    _rough(raw, rep, lid)
    _along(raw, rep, lid)
    _effects(raw, rep, lid)
    _layer_sound(raw, rep, lid)
    if "text" in TYPES[t]:
        _text(raw, rep, lid)
    _box(raw, rep, lid)
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
    if t == "group":
        _shadows(raw, rep, lid)
        _clock(raw, rep, lid)
        _combine(raw, rep, lid)
    kids = raw.get("children") if raw.get("children") is not None else raw.get("layers")
    if t == "group" and raw.get("children") is not None and raw.get("layers") is not None:
        rep.add("error", "duplicate-key", lid, "a group lists its layers under 'children' or 'layers', not both",
                "put them all under \"children\"")
    elif t == "group" and kids is not None and not isinstance(kids, list):
        rep.add("error", "bad-value", lid, f"children is {_show(kids)}; it is a list of layers",
                "\"children\": [{\"type\": \"text\", \"text\": \"Hi\"}]")
    if t == "group" and isinstance(kids, list):
        for i, k in enumerate(kids):
            _layer(k, rep, f"{lid}.children[{i}]", comps)


def _point(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) in (2, 3) and not any(isinstance(c, (list, tuple, dict)) for c in v)


def _position(pos) -> dict | None:
    """A layer's position as its x, y and z keys: from [x, y], [x, y, z] or {"x": ..., "y": ...}, from keyframes of
    points, or from a tween or keys object of points. None when it is none of these."""
    if _point(pos):
        return dict(zip("xyz", pos))
    if isinstance(pos, dict) and pos and set(pos) <= {"x", "y", "z"}:
        return dict(pos)
    keys = pos.get("keys") if isinstance(pos, dict) else pos
    if _keyframes(keys) and all(_point(k[1]) for k in keys) and len({len(k[1]) for k in keys}) == 1:
        axes = {a: [[k[0], k[1][j], *k[2:]] for k in keys] for j, a in enumerate("xyz"[:len(keys[0][1])])}
        return {a: dict(pos, keys=v) for a, v in axes.items()} if isinstance(pos, dict) else axes
    if isinstance(pos, dict) and "keys" not in pos and _point(pos.get("to")):
        n = len(pos["to"])
        if "from" in pos and not (_point(pos["from"]) and len(pos["from"]) == n):
            return None
        return {a: dict(pos, to=pos["to"][j], **({"from": pos["from"][j]} if "from" in pos else {}))
                for j, a in enumerate("xyz"[:n])}
    return None


def _stroke(raw: dict, t: str, rep: Report, lid):
    """dash ([on, off, ...] px, as SVG's stroke-dasharray), the dashOffset that slides it, and a path's corner."""
    dash = raw.get("dash")
    if dash is not None:
        if not (isinstance(dash, list) and dash and all(_finite(d) and d >= 0 for d in dash) and any(dash)):
            rep.add("error", "bad-value", lid, f"dash is {_show(dash)}; it is lengths in px, dash then gap, "
                    "repeated along the stroke, at least one above 0", '"dash": [12, 8]; to march the dashes, animate '
                    "dashOffset")
        elif raw.get("stroke") is None and t not in ("line", "arrow"):
            rep.add("warning", "ignored-key", lid, "dash marks out a stroke and this layer has none",
                    '"stroke": "#f2f0e8", "strokeWidth": 4')
    elif "dashOffset" in raw:
        rep.add("warning", "ignored-key", lid, "dashOffset slides a dash pattern and this layer has no dash",
                '"dash": [12, 8]')
    if "corner" in raw and raw.get("smooth") is True:
        rep.add("warning", "ignored-key", lid, "corner rounds the corners between straight pieces, and smooth "
                "leaves none", "drop smooth to keep the points' straight lines and round their corners")


def _rough(raw: dict, rep: Report, lid):
    """rough: true, an amount in px, or {"amount", "length", "seed", "boil"}, made into the object."""
    v = raw.get("rough")
    if v is None or v is False:
        raw.pop("rough", None)
        return
    example = '"rough": {"amount": 3, "seed": 1, "boil": 8}'
    if v is True:
        raw["rough"] = v = {}
    elif _finite(v):
        raw["rough"] = v = {"amount": v}
    if not isinstance(v, dict):
        rep.add("error", "bad-value", lid, f"rough is {_show(v)}; it is true, an amount in px, or an object", example)
        return
    _keys(v, ROUGH, ROUGH_ALIAS, rep, lid, "rough: ")
    for k, least, what in (("amount", 0, "how far in px the line strays, 0 or more"),
                           ("length", 1, "the px between the points that stray, 1 or more"),
                           ("boil", 0, "how many times a second it is drawn anew, 0 or more")):
        if k in v and not (_finite(v[k]) and v[k] >= least):
            rep.add("error", "bad-value", lid, f"rough {k} is {_show(v[k])}; it is {what}", example)
    s = v.get("seed")
    if s is not None and not (isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2 ** 31):
        rep.add("error", "bad-value", lid, f"rough seed is {_show(s)}; it is a whole number, 0 or more, that picks the "
                "wobble", example)
    if v.get("amount") == 0:
        rep.add("warning", "ignored-key", lid, "rough with amount 0 draws the shape smooth", example)


COMBINES = {"rect", "circle", "ellipse", "polygon", "path", "line", "arrow", "text", "counter", "group"}


def _combine(raw: dict, rep: Report, lid):
    """combine on a group merges its shapes and text into one shape; without it the group draws no stroke."""
    kids = raw.get("children") if raw.get("children") is not None else raw.get("layers")
    kids = [k for k in kids if isinstance(k, dict)] if isinstance(kids, list) else []
    if not raw.get("combine"):
        drawn = sorted(SHAPE - {"fill"} & set(raw))
        if drawn:
            rep.add("warning", "ignored-key", lid, f"{', '.join(drawn)} on a group draw only with combine, which "
                    "merges its children into one shape", '"combine": "union"')
        return
    left = [str(k.get("id", k.get("type"))) for k in kids if k.get("type") not in COMBINES]
    if left:
        rep.add("warning", "ignored-key", lid, f"combine merges shapes and text, so {', '.join(left)} is left out",
                "move it out of the group")
    if len(kids) - len(left) < (2 if raw["combine"] != "union" else 1):
        rep.add("warning", "ignored-key", lid, f"combine {raw['combine']} needs at least two shapes or texts to "
                "combine", "put the shapes to merge in the group's children")


def _clock(raw: dict, rep: Report, lid):
    """A group's own clock for its children: speed (a number above 0), loop (seconds, above 0) and timeRemap."""
    sp, lp = raw.get("speed"), raw.get("loop")
    if sp is not None and not (_finite(sp) and sp > 0):
        rep.add("error", "bad-value", lid, f"speed is {_show(sp)}; on a group it is a number above 0, how fast its "
                "children's time runs", "\"speed\": 0.5 for half speed; to change it over time, write timeRemap")
    if lp is not None and not (_finite(lp) and lp > 0):
        rep.add("error", "bad-value", lid, f"loop is {_show(lp)}; on a group it is the seconds after which its "
                "children's time starts again", "\"loop\": 1.2")
    if "timeRemap" in raw and sp is not None:
        rep.add("warning", "ignored-key", lid, "speed is ignored: timeRemap gives the children's time itself",
                "drop speed, or fold it into timeRemap")


def _shadows(raw: dict, rep: Report, lid):
    """A stage's `shadows`: true or {"opacity", "soft"}."""
    v = raw.get("shadows")
    if v is None or v is False:
        return
    if isinstance(v, dict):
        _keys(v, SHADOWS, SHADOWS_ALIAS, rep, lid, "shadows: ")
        for k in sorted(SHADOWS & set(v)):
            if not _number(v[k]):
                rep.add("error", "bad-value", lid, f"shadows {k} is {_show(v[k])}; it takes a number")
    elif v is not True:
        rep.add("error", "bad-value", lid, f"shadows is {_show(v)}; it takes true or an object like "
                "{\"opacity\": 0.5, \"soft\": 4}")
        return
    if raw.get("sort") != "pixel":
        rep.add("warning", "ignored-key", lid, "shadows fall only in a stage", "add \"sort\": \"pixel\" to the group")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _volume(raw: dict, rep: Report, lid, where: str = ""):
    """A video's (or a sequence's) volume, and the seconds its sound fades in and out: numbers, 0 or more. Keyframes
    for volume reached the mix as a list, -1 flipped the sound's phase and null muted it."""
    for k in ("volume", "fadeIn", "fadeOut"):
        if k in raw and not (_number(raw[k]) and math.isfinite(raw[k]) and raw[k] >= 0):
            rep.add("error", "bad-value", lid, f"{where}{k} is {raw[k]!r}; it must be a number, 0 or more",
                    "1 keeps the sound as it is, 0.5 halves it and 0 mutes it" if k == "volume" else
                    "seconds; 0 for no fade")


def _clip_type(clip) -> str:
    src = clip.get("src") if isinstance(clip, dict) else clip
    return "image" if isinstance(src, str) and src.lower().endswith(IMAGE_EXTS) else "video"


def _clips(raw: dict, rep: Report, lid, comps: dict):
    """A sequence's clips: each a video (or, with a dur, an image), checked as the layer it becomes."""
    clips = raw.get("clips")
    example = "\"clips\": [{\"src\": \"a.mp4\"}, {\"src\": \"b.mp4\"}]"
    if clips is None or clips == []:
        rep.add("error", "missing-key", lid, "a sequence needs 'clips', a list of at least one clip", example)
        return
    if not isinstance(clips, list):
        rep.add("error", "bad-value", lid, "clips must be a list of files or objects with 'src'", example)
        return
    for i, c in enumerate(clips):
        where = f"clips[{i}]"
        if isinstance(c, str):
            if not c.strip():
                rep.add("error", "missing-key", lid, f"{where} is an empty file name", example)
            elif _clip_type(c) == "image":
                rep.add("error", "missing-key", lid, f"{where} '{c}' is an image, which has no length of its own",
                        f"give it as {{\"src\": \"{c}\", \"dur\": 2}}")
            continue
        if not isinstance(c, dict):
            rep.add("error", "bad-value", lid, f"{where} is {c!r}; a clip is a file name or an object with 'src'", example)
            continue
        ctype = c.get("type") or _clip_type(c)
        if ctype not in ("video", "image"):
            rep.add("error", "bad-type", lid, f"{where}: a clip is a video or an image, not {ctype!r}",
                    "leave out type; a clip's src says which it is")
            continue
        c["type"] = ctype
        _layer(c, rep, f"{lid}.{where}", comps)
        if ctype == "image" and c.get("dur") is None and c.get("until") is None:
            rep.add("error", "missing-key", lid, f"{where}: an image clip needs 'dur', since an image has no length "
                    "of its own", f"{{\"src\": \"{c.get('src')}\", \"dur\": 2}}")


def _transition(raw: dict, rep: Report, lid):
    """A sequence's transition: a known type, and a dur in seconds unless it is a cut."""
    tr = raw.get("transition")
    example = "\"transition\": {\"type\": \"fade\", \"dur\": 0.4}"
    if tr is None or tr == {}:
        return
    if not isinstance(tr, dict):
        rep.add("error", "bad-value", lid, f"transition is {tr!r}; it is an object", example)
        return
    _keys(tr, TRANSITION, TRANSITION_ALIAS, rep, lid, "transition: ")
    kind = "fade"
    if "type" in tr:
        v = tr["type"]
        low = re.sub(r"[-_ ]", "", v.strip().lower()) if isinstance(v, str) else None
        kind = v if v in TRANSITIONS else TRANSITION_VALUE_ALIAS.get(low, low if low in TRANSITIONS else None)
        if kind is None:
            rep.add("error", "bad-value", lid, "transition type: " + unknown("value", v, TRANSITIONS)
                    if isinstance(v, str) and v.strip() else
                    f"transition type is {v!r}; it is one of {', '.join(TRANSITIONS)}", example)
            return
        if kind != v:
            tr["type"] = kind
            rep.add("info", "renamed-value", lid, f"transition type '{v}' read as '{kind}'", f"write '{kind}'")
    d = tr.get("dur")
    if kind == "cut":
        if d is not None:
            del tr["dur"]
            rep.add("warning", "ignored-key", lid, f"transition: a cut has no dur, so dur {d!r} is ignored and the "
                    "clips join back to back", "remove dur, or use \"type\": \"fade\" to overlap the clips")
    elif "dur" in tr and not (_number(d) and math.isfinite(d) and d > 0):
        rep.add("error", "bad-value", lid, f"transition dur is {d!r}; it is seconds, a number above 0",
                "\"dur\": 0.5 overlaps the clips by half a second; for no overlap use \"type\": \"cut\"")


def _structure(raw: dict, t: str, rep: Report, lid):
    """Keys with a fixed shape, checked here so a wrong one is an error instead of a crash while drawing."""
    u = raw.get("uniforms")
    if u is not None and not isinstance(u, dict):
        rep.add("error", "bad-value", lid, f"uniforms must be an object like {{\"speed\": 2}}, got {type(u).__name__}")
    for k, v in (u or {}).items() if isinstance(u, dict) else ():
        if isinstance(v, str) and not v.startswith(("=", "$")):  # text in a uniform is a colour
            _colour(v, f"uniforms {k}", rep, lid)
    if t == "shader" and raw.get("sksl") is not None and not isinstance(raw["sksl"], str):
        rep.add("error", "bad-value", lid, f"sksl is {_show(raw['sksl'])}; it is the shader's source as text, like "
                "\"half4 main(float2 p) { return half4(1, 0, 0, 1); }\"")
    fmt = raw.get("format")
    if t == "counter" and fmt is not None:
        bad = None if isinstance(fmt, str) else "it is not text"
        for spec in re.findall(r"\{:([^{}]*)\}", fmt) if bad is None else ():
            try:
                format(1.0, spec)
            except ValueError:
                bad = f"{{:{spec}}} is not a number format"
        if bad:
            rep.add("error", "bad-value", lid, f"format is {_show(fmt)}; {bad}. It is text with {{:...}} where the "
                    "number goes, like \"{:,.0f}\", \"{:.1f}%\" or \"$ {:,.2f}\"")
    if t == "path":
        _path_data(raw.get("d"), "d", rep, lid)
    focus = raw.get("focus")
    if t in ("image", "video") and focus is not None and not isinstance(focus, str) and not (
            isinstance(focus, list) and len(focus) == 2 and all(map(_finite, focus))):
        rep.add("error", "bad-value", lid, f"focus is {_show(focus)}; it is [x, y], each from 0 to 1 of the picture, "
                "the point kept in view when it is cropped", "\"focus\": [0.5, 0.3]")
    cues = raw.get("cues")
    if t == "captions" and cues is not None:
        def secs(v):
            try:
                return not isinstance(v, bool) and math.isfinite(float(v))
            except (TypeError, ValueError, OverflowError):
                return False
        for j, c in enumerate(cues if isinstance(cues, list) else [cues], 1):
            ok = (isinstance(c, dict) and secs(c.get("at", c.get("start", 0))) and secs(c.get("end"))) or (
                isinstance(c, (list, tuple)) and len(c) == 3 and secs(c[0]) and secs(c[1]))
            if not ok or not isinstance(cues, list):
                rep.add("error", "bad-captions", lid, f"cues[{j - 1}] is {_show(c)}; a cue is [start, end, \"text\"] or "
                        "{\"start\": 0.5, \"end\": 2, \"text\": \"...\"}, in seconds",
                        "\"cues\": [[0, 1.5, \"Hello\"]]")
                break
    if t == "path" and _keyframes(raw.get("d")):
        _morph(raw["d"], rep, lid)
    elif t == "path" and raw.get("d") is not None and not isinstance(raw["d"], str):
        rep.add("error", "bad-value", lid, f"d is {_show(raw['d'])}; it is an SVG path like \"M0 0 L100 0\", or "
                "keyframes of paths to morph through", '"d": [[0, "M0 0 L100 0 L50 90 Z"], [1, "M50 0 A50 50 0 1 1 '
                '49.9 0 Z"]]')
    pts = raw.get("points")
    if t in ("path", "line") and pts is not None and not (
            isinstance(pts, list) and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(map(_number, p))
                                          for p in pts)):
        rep.add("error", "bad-value", lid, "points must be a list of [x, y] number pairs",
                "\"points\": [[0, 0], [100, 50]]")


def _morph(keys: list, rep: Report, lid):
    """d as keyframes of paths: each key's path, its time and ease, and whether the keys have as many shapes."""
    _prop("d", keys, rep, lid)  # the times and eases
    counts = []
    for j, key in enumerate(keys, 1):
        if isinstance(key[1], str):
            _path_data(key[1], f"d at keyframe {j}", rep, lid)
            counts.append(len(re.findall(r"[Mm]", key[1])))
        else:
            rep.add("error", "bad-value", lid, f"d at keyframe {j} is {_show(key[1])}; it is SVG path data like "
                    '"M0 0 L100 0 L50 90 Z"')
    if len(set(counts)) > 1:
        rep.add("info", "morph-contours", lid, f"d's keys are drawn in {' and '.join(map(str, sorted(set(counts))))} "
                "pieces: a piece with no partner grows from, or shrinks into, the middle of the nearest one",
                "give each key as many pieces (M commands) to pair them as you want")


def _effects(raw: dict, rep: Report, lid):
    """Motions, shadow and fx, which any layer and a sequence take."""
    for key in ("enter", "exit"):
        _motion(raw, key, rep, lid)
    sh = raw.get("shadow")
    if isinstance(sh, dict):
        _keys(sh, SHADOW, SHADOW_ALIAS, rep, lid, "shadow: ")
        _numbers(sh, ("x", "y"), "shadow ", rep, lid)
        _numbers(sh, ("blur",), "shadow ", rep, lid, least=0)
        if sh.get("color") is not None:
            _colour(sh["color"], "shadow color", rep, lid)
    elif sh is not None and not isinstance(sh, bool):
        rep.add("error", "bad-value", lid, f"shadow is {_show(sh)}; it is true (the default shadow) or an object like "
                "{\"x\": 0, \"y\": 6, \"blur\": 16, \"color\": \"#00000099\"}")
    _look(raw, "fx", FX, {"saturation": "saturate", "grayscale": "gray", "greyscale": "gray", "hueRotate": "hue",
                          "warmth": "temperature"}, rep, lid)


def _look(obj: dict, key: str, allowed: set, alias: dict, rep: Report, lid):
    """fx on a layer, or grade on the scene: an object of numbers."""
    v = obj.get(key)
    if v is None:
        return
    if not isinstance(v, dict):
        rep.add("error", "bad-value", lid, f"{key} is {_show(v)}; it is an object of numbers like "
                + ('{"vignette": 0.4}' if key == "grade" else '{"brightness": 0.1, "saturate": 1.2}'))
        return
    _keys(v, allowed, alias, rep, lid, f"{key}: ")
    for k in [k for k, x in v.items() if x is None]:
        del v[k]  # null is the default
    for k, x in v.items():
        if k in allowed and not (_finite(x) or isinstance(x, bool)):
            rep.add("error", "bad-value", lid, f"{key} {k} is {_show(x)}; it takes a number")


def _tweens(v, colour: bool = False) -> bool:
    """A value a light key can take: a number (a colour), keyframes of them, a tween object or an expression."""
    if isinstance(v, str) and v.startswith("="):
        return True
    if isinstance(v, dict) and ("to" in v or "keys" in v):
        return all(_tweens(v[k], colour) for k in ("from", "to") if k in v)
    if isinstance(v, list) and v and all(isinstance(k, list) and 2 <= len(k) <= 3 and _is_time(k[0]) for k in v):
        return all(_tweens(k[1], colour) for k in v)
    if colour:
        return isinstance(v, str)
    if isinstance(v, str):
        return bool(_LEN.match(v))  # "50%", "20vw"
    return isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v)


def _light_values(v: dict, nums, rep: Report, lid, where: str):
    from .render3d import FROM
    if v.get("from") is not None and v["from"] not in FROM:
        rep.add("error", "bad-value", lid, f"{where}from: " + unknown("direction", v["from"], FROM))
    if v.get("follow") is not None and v["follow"] not in ("camera", "world"):
        rep.add("error", "bad-value", lid, f"{where}follow is {v['follow']!r}; it takes \"camera\" or \"world\"")
    for k in nums:
        if k in v and not _tweens(v[k]):
            rep.add("error", "bad-value", lid, f"{where}{k} is {_show(v[k])}; it takes a number, or keyframes of them")
    if v.get("color") is not None:
        if isinstance(v["color"], str):
            _colour(v["color"], f"{where}color", rep, lid)
        elif not _tweens(v["color"], colour=True):
            rep.add("error", "bad-value", lid, f"{where}color is {_show(v['color'])}; it takes a colour, or keyframes")


def _light(obj: dict, rep: Report, lid):
    """`light` is a preset name or an object of light amounts, `lights` up to three more lights; both are
    checked here."""
    from .render3d import LIGHTS, MORE_LIGHTS
    _more_lights(obj, rep, lid, MORE_LIGHTS)
    _env(obj, rep, lid)
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
        _light_values(v, ("key", "fill", "ambient", "rim", "azimuth", "elevation"), rep, lid, "light ")
    elif v is not None:
        rep.add("error", "bad-value", lid, f"light must be a preset ({', '.join(LIGHTS)}) or an object, got {v!r}")


def _env(obj: dict, rep: Report, lid):
    """`env` is an environment preset name or an object of its colours and amount."""
    from .render3d import ENVS
    v = obj.get("env")
    if isinstance(v, str):
        s = v if v in ENVS else suggest(v, ENVS)
        if s and s.lower() == v.lower():
            obj["env"] = s
        else:
            rep.add("error", "bad-value", lid, "env: " + unknown("environment", v, ENVS))
    elif isinstance(v, dict):
        _keys(v, ENV, ENV_ALIAS, rep, lid, "env: ")
        if v.get("preset") is not None and v["preset"] not in ENVS:
            rep.add("error", "bad-value", lid, "env preset: " + unknown("environment", v["preset"], ENVS))
        if "amount" in v and not _tweens(v["amount"]):
            rep.add("error", "bad-value", lid, f"env amount is {_show(v['amount'])}; it takes a number, or keyframes")
        for k in ("top", "horizon", "bottom"):
            if isinstance(v.get(k), str):
                _colour(v[k], f"env {k}", rep, lid)
            elif v.get(k) is not None and not _tweens(v[k], colour=True):
                rep.add("error", "bad-value", lid, f"env {k} is {_show(v[k])}; it takes a colour, or keyframes")
    elif v is not None:
        rep.add("error", "bad-value", lid, f"env must be a preset ({', '.join(ENVS)}) or an object, got {_show(v)}")


def _fog(spec: dict, rep: Report):
    """`fog` is true or {color, near, far, amount}; near and far are px from the camera."""
    v = spec.get("fog")
    if v is None or isinstance(v, bool):
        return
    if not isinstance(v, dict):
        rep.add("error", "bad-value", "fog", f"fog is true or an object like {{\"near\": 1500, \"far\": 4000}}, "
                f"got {_show(v)}")
        spec.pop("fog")
        return
    _keys(v, FOG, FOG_ALIAS, rep, "fog", "fog: ")
    _light_values(v, ("near", "far", "amount"), rep, "fog", "fog ")
    near, far = v.get("near"), v.get("far")
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (near, far)) and far <= near:
        rep.add("error", "bad-value", "fog", f"fog far ({far:g}) must be past near ({near:g})",
                "far is where the fog hides everything, near where it starts")


def _target(cam: dict, rep: Report):
    """camera target: a point [x, y] or [x, y, z] in px or lengths, a layer id (Scene checks that it is one), an
    expression, or keyframes of them."""
    v = cam.get("target")
    if v is None:
        return

    def spot(x) -> bool:
        if isinstance(x, str):
            return True
        return isinstance(x, list) and 2 <= len(x) <= 3 and all(
            _finite(e) or (isinstance(e, str) and _lengthish(e)) for e in x)
    keys = [k[1] for k in v] if not spot(v) and _keyframes(v) else [v]
    for x in keys:
        if not spot(x):
            rep.add("error", "bad-value", "camera", f"target is {_show(x)}; it takes a point [x, y, z], a layer id, "
                    "or keyframes of them", '"target": [960, 540, -200] or "target": "logo"')
            cam.pop("target")
            return


def _focus(cam: dict, rep: Report):
    """camera focus: a z in px or a layer id (Scene checks that it is one), an expression, or keyframes of them;
    aperture: px, or keyframes of them."""
    v = cam.get("focus")
    if v is not None:
        keys = [k[1] for k in v] if _keyframes(v) else [v]
        for x in keys:
            if not (isinstance(x, str) or _finite(x)):
                rep.add("error", "bad-value", "camera", f"focus is {_show(x)}; it takes a z in px, a layer id, or "
                        "keyframes of them", '"focus": -300 or "focus": "logo"')
                cam.pop("focus")
                break
    v = cam.get("aperture")
    if v is not None and not _tweens(v):
        rep.add("error", "bad-value", "camera", f"aperture is {_show(v)}; it takes px of blur per 1000 px from the "
                "focus plane, or keyframes of them", '"aperture": 12')
        cam.pop("aperture")
    elif isinstance(v, (int, float)) and v < 0:
        rep.add("error", "bad-value", "camera", f"aperture is {_show(v)}; it must be 0 or more")
        cam.pop("aperture")
    elif cam.get("focus") is not None and v is None:
        rep.add("warning", "ignored-key", "camera", "focus does nothing without aperture",
                '"aperture": 12 blurs what is 1000 px from the focus plane by 12 px')


def _shake(cam: dict, rep: Report):
    """camera shake: px (a number, or keyframes of them), true, or {amount, speed, seed}."""
    v = cam.get("shake")
    if v is None or isinstance(v, bool):
        return
    if isinstance(v, dict) and not _tween(v):
        _keys(v, SHAKE, SHAKE_ALIAS, rep, "camera", "camera shake: ")
        if "amount" in v and not _tweens(v["amount"]):
            rep.add("error", "bad-value", "camera", f"shake amount is {_show(v['amount'])}; it takes px, or keyframes "
                    "of them")
        _numbers(v, ("speed",), "shake ", rep, "camera", least=0)
        _numbers(v, ("seed",), "shake ", rep, "camera")
    elif not _tweens(v):
        rep.add("error", "bad-value", "camera", f"shake is {_show(v)}; it takes px, true, or an object like "
                "{\"amount\": 8, \"speed\": 1.5}")
        cam.pop("shake")


def _more_lights(obj: dict, rep: Report, lid, most: int):
    v = obj.get("lights")
    if isinstance(v, dict):
        v = obj["lights"] = [v]
    if v is None:
        return
    if not isinstance(v, list):
        rep.add("error", "bad-value", lid, f"lights is a list of light objects, got {_show(v)}",
                '"lights": [{"type": "point", "x": 300, "y": 200, "z": 400, "color": "#7fb2ff"}]')
        obj.pop("lights")
        return
    if len(v) > most:
        rep.add("warning", "ignored-key", lid, f"lights has {len(v)} lights; only the first {most} light a model")
    for j, x in enumerate(v):
        where = f"lights[{j}] "
        if not isinstance(x, dict):
            rep.add("error", "bad-value", lid, f"{where}is {_show(x)}; each light is an object")
            continue
        _keys(x, MORE_LIGHT, MORE_LIGHT_ALIAS, rep, lid, where.rstrip() + ": ")
        if x.get("type", "directional") not in ("directional", "point"):
            rep.add("error", "bad-value", lid, f"{where}type is {x['type']!r}; it takes \"directional\" or \"point\"")
        _light_values(x, ("azimuth", "elevation", "x", "y", "z", "intensity", "range"), rep, lid, where)


def _along(raw: dict, rep: Report, lid):
    """along: the path a layer rides, as path data or "#id" of a path layer, the share p of its length it is at, and
    whether it turns with the path."""
    a = raw.get("along")
    if a is None:
        return
    example = "\"along\": {\"d\": \"M0 0 C300 -200 600 200 900 0\", \"p\": [[0, 0], [2, 1]], \"orient\": true}"
    if isinstance(a, str):
        raw["along"] = a = {"d": a}
        rep.add("info", "renamed-value", lid, f"along {_show(a['d'])} read as its d", f"write \"along\": {{\"d\": ...}}")
    if not isinstance(a, dict):
        rep.add("error", "bad-value", lid, f"along is {_show(a)}; it is an object with the path in d", example)
        return
    _keys(a, ALONG, ALONG_ALIAS, rep, lid, "along: ")
    d = a.get("d")
    if not (isinstance(d, str) and d.strip()):
        rep.add("error", "missing-key", lid, "along needs d: SVG path data, or \"#id\" of a path layer to ride",
                example)
    elif not d.startswith("#"):
        _path_data(d, "along d", rep, lid)
    if a.get("p") is not None:
        _prop("along p", a["p"], rep, lid)
    if a.get("orient") is not None and not isinstance(a["orient"], bool):
        rep.add("error", "bad-value", lid, f"along orient is {_show(a['orient'])}; it is true or false", example)


def _path_data(d, what: str, rep: Report, lid):
    """Path data that does not parse, named by what it is."""
    if not isinstance(d, str) or "$" in d:
        return
    from .svgpath import parse as svg
    try:
        svg(d)
    except (ValueError, IndexError) as e:
        m = re.search(r"could not convert string to float: '(.*)'", str(e))
        why = (f"'{m.group(1)}' is neither a number nor a path command (M, L, H, V, C, S, Q, T, A, Z)" if m
               else str(e) if isinstance(e, ValueError) else "it ends in the middle of a command")
        rep.add("error", "bad-value", lid, f"{what}: {why}; {what.split()[-1]} is SVG path data like "
                "\"M0 0 L100 0 Q150 50 100 100 Z\"")


def _turned(obj: dict, rep: Report, lid, where: str = ""):
    """A lathe turns its profile round the vertical axis: [r, y] points, r out from the axis, or path data."""
    prof = obj.get("profile")
    if obj.get("shape") != "lathe":
        if prof is not None:
            rep.add("warning", "ignored-key", lid, f"{where}profile is ignored: only a lathe turns a profile",
                    "\"shape\": \"lathe\"")
        return
    example = "\"profile\": [[0, -100], [70, -60], [50, 100], [0, 100]]"
    if prof in (None, "", []):
        rep.add("error", "missing-key", lid, f"{where}a lathe needs a profile: the outline it turns round the "
                "vertical axis, as [r, y] points (r out from the axis, y down) or path data", example)
    elif isinstance(prof, str):
        _path_data(prof, f"{where}profile", rep, lid)
    elif not (isinstance(prof, list) and len(prof) >= 2 and all(
            isinstance(p, list) and len(p) == 2 and all(map(_finite, p)) for p in prof)):
        rep.add("error", "bad-value", lid, f"{where}profile is {_show(prof)}; it is two or more [r, y] points, r out "
                "from the axis and y down, or path data", example)
    elif any(p[0] < 0 for p in prof):
        rep.add("error", "bad-value", lid, f"{where}profile has a point with r below 0; r is the distance out from "
                "the axis", "turn the profile to the right of the axis: [10, 0], not [-10, 0]")
    elif all(p[0] == 0 for p in prof):
        rep.add("error", "bad-value", lid, f"{where}profile's points are all on the axis (r 0), so the lathe has no "
                "surface", example)


def _outline(obj: dict, rep: Report, lid, where: str = ""):
    """Text, path, tube and lathe shapes need their outline; a number in `d` is a depth (d is the path data
    here). A tube without d is a cylinder, as it was read before tubes ran along a path."""
    d = obj.get("d")
    if isinstance(d, (int, float)) and not isinstance(d, bool) and "depth" not in obj:
        obj["depth"] = obj.pop("d")
        rep.add("info", "renamed-key", lid, f"{where}'d' read as 'depth'", "write 'depth'")
    shape = obj.get("shape")
    if shape == "tube" and not isinstance(obj.get("d"), str):
        obj["shape"] = shape = "cylinder"
        rep.add("info", "renamed-value", lid, f"{where}shape 'tube' without d read as 'cylinder'",
                "a tube runs along SVG path data: \"d\": \"M-200 0 C-100 -150 100 150 200 0\"; write cylinder for "
                "a straight one")
    if shape == "tube":
        _path_data(obj["d"], f"{where}d", rep, lid)
    _turned(obj, rep, lid, where)
    if shape == "text" and not str(obj.get("text", "")).strip():
        rep.add("error", "missing-key", lid, f"{where}a text model needs 'text'",
                'add "text": "..."')
    if shape == "path" and not isinstance(obj.get("d"), str):
        rep.add("error", "missing-key", lid, f"{where}a path model needs 'd', SVG path data",
                'add "d": "M0 0 L100 0 L50 80 Z"')


def texture_kind(src: str) -> str:
    """What a texture's src names: "group" ("#id"), "image" or "video"."""
    return "group" if src.startswith("#") else "image" if src.lower().endswith(IMAGE_EXTS + (".gif",)) else "video"


def _texture(obj: dict, rep: Report, lid, where: str = ""):
    """`texture` is an image or video file, or "#groupId", or {"src", "repeat", "offset"} with a video's "loop",
    "trim" and "speed"; false draws the fill instead of the layer's texture (on a part) or the model file's own."""
    v = obj.get("texture")
    if v is None or v is False:
        return
    if isinstance(v, dict):
        _keys(v, TEXTURE, TEXTURE_ALIAS, rep, lid, f"{where}texture: ")
        src = v.get("src")
        if src is None:
            rep.add("error", "missing-key", lid, f"{where}texture needs 'src', an image file",
                    '"texture": {"src": "wood.jpg", "repeat": 2}')
            return
        for k, least in (("repeat", 0.0), ("offset", None)):
            x = v.get(k)
            pair = isinstance(x, list) and len(x) == 2 and all(_number(n) for n in x)
            if x is not None and not (_number(x) or pair) or (
                    least is not None and x is not None and min(x if pair else [x] if _number(x) else [1]) <= least):
                what = "a number above 0, or [across, down]" if k == "repeat" else "a number, or [across, down]"
                rep.add("error", "bad-value", lid, f"{where}texture {k} is {_show(x)}; it takes {what}",
                        "\"repeat\": 2 shows the image twice across and down" if k == "repeat" else
                        "\"offset\": [0.5, 0] moves the image half its width to the right")
        if v.get("loop") is not None and not isinstance(v["loop"], bool):
            rep.add("error", "bad-value", lid, f"{where}texture loop is {_show(v['loop'])}; it takes true or false")
        sp = v.get("speed")
        if sp is not None and not (_finite(sp) and sp > 0):
            rep.add("error", "bad-value", lid, f"{where}texture speed is {_show(sp)}; it must be a number above 0",
                    "2 plays twice as fast, 0.5 at half speed")
    else:
        src = v
    if not (isinstance(src, str) and src.strip()) or src.strip() == "#":
        rep.add("error", "bad-value", lid, f"{where}texture is {_show(v)}; it takes an image or video file, "
                "\"#groupId\", or an object like {\"src\": \"wood.jpg\", \"repeat\": 2}")
    elif isinstance(v, dict) and texture_kind(src) != "video" and any(k in v for k in CLIP_KEYS):
        keys = ", ".join(k for k in CLIP_KEYS if k in v)
        rep.add("warning", "ignored-key", lid, f"{where}texture {keys}: only a video texture plays, and {src!r} is "
                f"{'a group' if src.startswith('#') else 'a still image'}", "drop them, or use a video file")


def _edges(raw: dict, rep: Report, lid):
    """`edges` is true, "all" or {"width", "color", "angle"}, each of which may animate."""
    v = raw.get("edges")
    if v is None or isinstance(v, bool) or v == "all":
        return
    if isinstance(v, str) and v.lower() in ("all", "wireframe", "every"):
        raw["edges"] = "all"
        return
    if not isinstance(v, dict):
        rep.add("error", "bad-value", lid, f"edges is {_show(v)}; it takes true, \"all\" or an object",
                '"edges": {"width": 2, "color": "#0a0c10", "angle": 30}')
        return
    _keys(v, EDGES, EDGES_ALIAS, rep, lid, "edges: ")
    for k, why in (("width", "px"), ("angle", "degrees, 0 for every edge")):
        if k in v and not _tweens(v[k]):
            rep.add("error", "bad-value", lid, f"edges {k} is {_show(v[k])}; it takes a number ({why}), or keyframes")
    if isinstance(v.get("color"), str):
        _colour(v["color"], "edges color", rep, lid)
    elif v.get("color") is not None and not _tweens(v["color"], colour=True):
        rep.add("error", "bad-value", lid, f"edges color is {_show(v['color'])}; it takes a colour, or keyframes")


def _see_through(raw: dict, rep: Report, lid):
    """A model with a texture draws solid: its passes are combined per pixel, the nearest face's alone."""
    parts = [p for p in raw.get("parts") or [] if isinstance(p, dict)] if isinstance(raw.get("parts"), list) else []
    if any(o.get("transparency") not in (None, 0) for o in [raw, *parts]) and any(
            o.get("texture") not in (None, False) for o in [raw, *parts]):
        rep.add("warning", "ignored-key", lid, "transparency is ignored on a model with a texture: it draws solid",
                "put the see-through shape in a model layer of its own")


def _model_file(raw: dict) -> bool:
    """Whether the layer draws a model file, under src or a name agents write for it."""
    return any(isinstance(raw.get(k), str) and raw[k].strip().lower().endswith(MODEL_FILES)
               for k in ("src", "source", "file", "url", "model", "shape"))


def _animation(raw: dict, rep: Report, lid):
    """`animation` names a clip of the model file: a string, its number from 0, true for the first, or
    {"name", "speed", "loop", "start"}. Whether the file has it is checked when the scene loads."""
    if "animation" not in raw:
        return
    v = raw["animation"]
    if v is None or v is False:
        del raw["animation"]
        return
    named = lambda x: (isinstance(x, str) and x.strip()) or x is True or (  # noqa: E731
        isinstance(x, int) and not isinstance(x, bool) and x >= 0)
    if not isinstance(v, dict):
        if not named(v):
            rep.add("error", "bad-value", lid, f"animation is {_show(v)}; it names a clip of the model file: "
                    "its name, its number from 0, or an object like {\"name\": \"Walk\", \"speed\": 1}",
                    "\"animation\": \"Walk\" (kine probe lists the file's clips)")
        return
    _keys(v, ANIMATION, ANIMATION_ALIAS, rep, lid, "animation: ")
    if "name" in v and not named(v["name"]):
        rep.add("error", "bad-value", lid, f"animation name is {_show(v['name'])}; it is the clip's name or its "
                "number from 0", "kine probe lists the file's clips")
    for k, what in (("speed", "how fast it plays: 1 as made, 2 twice as fast, -1 backwards"),
                    ("start", "the seconds into the clip at the layer's start")):
        if k in v and not _finite(v[k]):
            rep.add("error", "bad-value", lid, f"animation {k} is {_show(v[k])}; it is a number, {what}")
    if isinstance(v.get("loop"), str) and v["loop"].lower() in ("true", "false", "yes", "no"):
        v["loop"] = v["loop"].lower() in ("true", "yes")
    if "loop" in v and not isinstance(v["loop"], bool):
        rep.add("error", "bad-value", lid, f"animation loop is {_show(v['loop'])}; it takes true (the default: the "
                "clip repeats) or false (it holds its last pose)")


def _repeat(raw: dict, rep: Report, lid):
    """`repeat` draws `count` copies of the model, each placed by x, y, z, rotateX, rotateY, rotate and scale and
    coloured by fill: a number (a colour), a list taken in turn, or an expression of i (the copy, from 0) and n."""
    if "repeat" not in raw:
        return
    v = raw["repeat"]
    if v is None or v is False:
        del raw["repeat"]
        return
    example = '"repeat": {"count": 10, "x": "=(i - 4.5) * 60"}'
    if not isinstance(v, dict):
        rep.add("error", "bad-value", lid, f"repeat is {_show(v)}; it is an object: how many copies and where each "
                "one goes", example)
        return
    _keys(v, REPEAT, REPEAT_ALIAS, rep, lid, "repeat: ")
    c = v.get("count")
    if isinstance(c, float) and c.is_integer():
        v["count"] = c = int(c)
    if "count" not in v:
        rep.add("error", "missing-key", lid, "repeat needs a count: how many copies of the model to draw", example)
    elif isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= MAX_COPIES:
        rep.add("error", "bad-value", lid, f"repeat count is {_show(c)}; it is a whole number from 1 to "
                f"{MAX_COPIES:,}")
    for k in sorted(REPEAT - {"count"}):
        x = v.get(k)
        if x is None or (isinstance(x, str) and x.startswith("=")):
            continue
        items = x if isinstance(x, list) and x else [x]
        if k == "fill":
            if all(isinstance(y, str) for y in items):
                for y in items:
                    _colour(y, "repeat fill", rep, lid)
                continue
            what = "a colour, a list of colours taken in turn, or an expression of i such as \"=hsl(i * 30)\""
        elif all(_finite(y) for y in items):
            continue
        else:
            what = "a number, a list of numbers taken in turn, or an expression of i such as \"=i * 60\""
        rep.add("error", "bad-value", lid, f"repeat {k} is {_show(x)}; it is {what}",
                "i is the copy's number from 0, n the count; t, d, p and T are the layer's times")


def _model(raw: dict, rep: Report, lid):
    _animation(raw, rep, lid)
    _repeat(raw, rep, lid)
    _light(raw, rep, lid)
    _outline(raw, rep, lid)
    _texture(raw, rep, lid)
    _edges(raw, rep, lid)
    _see_through(raw, rep, lid)
    if raw.get("src") and raw.get("shape"):
        rep.add("warning", "ignored-key", lid, f"shape '{raw['shape']}' is ignored: the model comes from src",
                "drop shape, or make the shape its own part")
    gr = raw.get("ground")
    if isinstance(gr, dict):
        _keys(gr, GROUND, GROUND_ALIAS, rep, lid, "ground: ")
        for k, v in gr.items():
            if k == "from":
                if v not in ("above", "light"):
                    rep.add("error", "bad-value", lid, f"ground from is {v!r}; it takes \"above\" or \"light\"",
                            "\"light\" casts the footprint along the key light, \"above\" (the default) straight down")
            elif k in GROUND and (isinstance(v, bool) or not isinstance(v, (int, float))):
                rep.add("error", "bad-value", lid, f"ground {k} is {v!r}; it takes a number")
    elif gr is not None and not isinstance(gr, bool):
        rep.add("error", "bad-value", lid, f"ground is {gr!r}; it takes true or an object like {{\"opacity\": 0.5}}")
    if isinstance(raw.get("fill"), dict) and not {"to", "keys"} & set(raw["fill"]):
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
        _text(p, rep, lid, where)
        _outline(p, rep, lid, where)
        _texture(p, rep, lid, where)


def _audio_values(a: dict, rep: Report, lid, where: str):
    _enum(a, "role", rep, lid)
    p = a.get("pan")
    if p is not None and (isinstance(p, bool) or not isinstance(p, (int, float)) or not -1 <= p <= 1):
        rep.add("error", "bad-value", lid, f"{where}pan is {p!r}; it must be a number from -1 (left) to 1 (right)")
    for k in ("volume", "fadeIn", "fadeOut"):
        v = a.get(k)
        if k in a and not (_number(v) and math.isfinite(v) and v >= 0):  # null too: it reached the mix as None
            rep.add("error", "bad-value", lid, f"{where}{k} is {v!r}; it must be a number, 0 or more",
                    "use \"gain\" in dB for a level that changes over time" if k == "volume" else "use seconds")
    src = a.get("src")
    if src not in (None, "") and not isinstance(src, str):
        rep.add("error", "bad-value", lid, f"{where}src is {src!r}; it is one file name, as a string",
                "\"src\": \"music.mp3\" (a path relative to the scene file); give each file its own entry")
    g = a.get("gain")
    if g is not None and not (_number(g) or isinstance(g, list) or (isinstance(g, str) and g.startswith("="))
                              or (isinstance(g, dict) and {"to", "keys"} & set(g))):
        rep.add("error", "bad-value", lid, f"{where}gain is {g!r}; it is dB as a number, keyframes or an expression",
                "\"gain\": -6, or \"gain\": [[0, -20], [2, 0]]")
    t, r = a.get("text"), a.get("rate")
    if t is not None and not (isinstance(t, str) and t.strip()):
        rep.add("error", "bad-value", lid, f"{where}text is {t!r}; it is the words to speak, as a string")
    if r is not None and (isinstance(r, bool) or not isinstance(r, (int, float)) or not 0.5 <= r <= 2):
        rep.add("error", "bad-value", lid, f"{where}rate is {r!r}; it is how fast the text is spoken, from 0.5 to 2")
    b, o = a.get("backend"), a.get("options")
    if b is not None and not (isinstance(b, str) and b.strip()):
        rep.add("error", "bad-value", lid, f"{where}backend is {b!r}; it names the voice backend, like \"kokoro\"")
    if o is not None and not isinstance(o, dict):
        rep.add("error", "bad-value", lid, f"{where}options is {o!r}; it is an object the backend reads, like "
                "{\"exaggeration\": 0.7}")
    if not a.get("src") and not a.get("text"):
        rep.add("error", "missing-key", lid, f"{where}an audio entry needs 'src' (or 'text' to speak)",
                "\"src\": \"music.mp3\"")


def _mix(m, rep: Report):
    if not isinstance(m, dict):
        rep.add("error", "bad-value", "mix", f"mix must be an object like {{\"loudness\": -16}}, got {type(m).__name__}")
        return
    _keys(m, MIX, MIX_ALIAS, rep, "mix", "mix: ")
    nums = [("loudness", m.get("loudness"), -40, -5, "LUFS"), ("peak", m.get("peak"), -12, 0, "dBTP")]
    levels = m.get("levels")
    if isinstance(levels, dict):
        _keys(levels, set(ENUMS["role"]), {}, rep, "mix", "mix.levels: ")
        nums += [(f"levels.{k}", v, -60, 0, "dB") for k, v in levels.items() if k in ENUMS["role"]]
    elif levels is not None:
        rep.add("error", "bad-value", "mix", "mix.levels must be an object like {\"music\": -20}")
    duck = m.get("duck")
    if isinstance(duck, dict):
        _keys(duck, DUCK, DUCK_ALIAS, rep, "mix", "mix.duck: ")
        nums += [(f"duck.{k}", duck.get(k), lo, hi, unit) for k, lo, hi, unit in
                 (("depth", 0, 40, "dB"), ("attack", 0, 5, "s"), ("hold", 0, 5, "s"), ("release", 0, 5, "s"))]
        for k in ("by", "under"):
            roles = duck.get(k)
            for r in roles if isinstance(roles, list) else [] if roles is None else [roles]:
                if r not in ENUMS["role"]:
                    rep.add("error", "bad-value", "mix", f"mix.duck.{k}: " + unknown("role", r, ENUMS["role"]))
    elif not (duck is None or isinstance(duck, bool)):
        rep.add("error", "bad-value", "mix", "mix.duck is an object like {\"depth\": 14}, true or false")
    for name, v, lo, hi, unit in nums:
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi):
            rep.add("error", "bad-value", "mix", f"mix.{name} is {v!r}; it must be a number from {lo} to {hi} ({unit})")


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


def _uses(layers) -> list[str]:
    """The component names a list of layers uses, groups included."""
    out = []
    for L in layers if isinstance(layers, list) else []:
        if isinstance(L, dict):
            if isinstance(L.get("use"), str):
                out.append(L["use"])
            out += _uses(L.get("children") or L.get("layers"))
    return out


def _components(comps: dict, rep: Report):
    """Each scene component is {"params": {...}, "layers": [...]}, and none may use itself."""
    example = '{"badge": {"params": {"label": "NEW"}, "layers": [{"type": "text", "text": "{{label}}"}]}}'
    good = {}
    for name, c in comps.items():
        where = f"components.{name}"
        if not isinstance(c, dict):
            rep.add("error", "bad-component", "components", f"{where} is {_show(c)}; a component is an object with "
                    "\"layers\" (and \"params\" for defaults)", example)
            continue
        _keys(c, {"params", "layers", "note"}, {"children": "layers", "defaults": "params"}, rep, "components",
              f"{where}: ")
        if c.get("params") is not None and not isinstance(c["params"], dict):
            rep.add("error", "bad-component", "components", f"{where}.params is {_show(c['params'])}; it is an object "
                    "of default values like {\"label\": \"NEW\"}", example)
        body = c.get("layers")
        if not isinstance(body, list) or not all(isinstance(L, dict) for L in body):
            rep.add("error", "bad-component", "components", f"{where}.layers is {_show(body)}; it is a list of layer "
                    "objects", example)
            continue
        good[name] = _uses(body)

    def cycle(name, path):
        for u in good.get(name, ()):
            if u in path:
                return path[path.index(u):] + [u]
            if (found := cycle(u, path + [u])) is not None:
                return found
        return None

    seen = set()
    for name in good:
        loop = cycle(name, [name])
        if loop and loop[0] not in seen:
            seen.update(loop)
            rep.add("error", "bad-component", "components", f"component '{loop[0]}' uses itself ("
                    + " uses ".join(f"'{n}'" for n in loop) + "), which would never end",
                    "a component's layers can use other components, but not one that leads back to it")


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
            cv.pop(k, None)  # null is the default
            continue
        if isinstance(v, str) and (m := re.fullmatch(rf"\s*(-?{_NUM})\s*(px)?\s*", v)):
            v = cv[k] = float(m.group(1))
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            rep.add("error", "bad-canvas", "canvas", f"canvas.{k} must be a number, got {_show(v)}")
        elif not lo <= v <= hi:
            rep.add("error", "bad-canvas", "canvas", f"canvas.{k} is {show(v)}; it must be between {lo} and {hi}")
    d = cv.get("duration")
    if isinstance(d, (int, float)) and not isinstance(d, bool) and not (_finite_secs(d) and d > 0):
        rep.add("error", "bad-canvas", "canvas", f"canvas.duration must be a number of seconds greater than 0 and at "
                f"most {LIMIT:g} (got {show(d)})", "use seconds, or \"auto\" to end with the last layer")
    elif isinstance(d, bool) or (d is not None and not isinstance(d, (int, float, str))):
        rep.add("error", "bad-canvas", "canvas", f"canvas.duration is {_show(d)}; it is a number of seconds, a time "
                "such as \"outro.out\", or \"auto\" to end with the last layer")
    bg = cv.get("bg")
    if isinstance(bg, dict):
        gradient(bg, "canvas.bg", rep, "canvas")
    elif bg is not None and not isinstance(bg, str):
        rep.add("error", "bad-colour", "canvas", f"canvas.bg is {_show(bg)}; it is a colour like \"#0a0c10\" or a "
                "gradient object", "use #rrggbb, or \"transparent\"")
    elif bg is not None and bg.strip().lower() not in ("transparent", "none"):
        _colour(bg, "canvas.bg", rep, "canvas")
    for k, example in (("theme", '{"accent": "#ff633f"}'), ("markers", '{"drop": 2.5}'),
                       ("grade", '{"vignette": 0.4}'), ("components", '{"card": {"layers": [...]}}')):
        if spec.get(k) is not None and not isinstance(spec[k], dict):
            rep.add("error", "bad-value", k, f"{k} must be an object like {example}, got {type(spec[k]).__name__}")
    for name, v in (spec.get("markers") or {}).items() if isinstance(spec.get("markers"), dict) else ():
        t = parse_clock(v, 30) if isinstance(v, str) else v
        if not (_finite(t) and _finite_secs(t)):
            rep.add("error", "bad-marker", "markers", f"markers.{name} is {_show(v)}; a marker is one time: seconds "
                    "like 2.5, \"45f\" or \"0:02.5\"", "to start something when a layer ends, cue it from the layer: "
                    "\"at\": \"title.out\"")
    theme = spec.get("theme") if isinstance(spec.get("theme"), dict) else {}
    for k in ("paper", "accent", "ink", "muted"):  # the colours the renderer and components draw with
        if theme.get(k) is not None:
            _colour(theme[k], f"theme.{k}", rep, "theme")
    if theme.get("font") is not None and not isinstance(theme["font"], str):
        rep.add("error", "bad-value", "theme", f"theme.font is {_show(theme['font'])}; it is a font family such as "
                "\"Space Grotesk\"")
    mb = spec.get("motionBlur")
    if isinstance(mb, (int, float)) and not isinstance(mb, bool):
        spec["motionBlur"] = {"samples": int(mb)}
        rep.add("info", "renamed-value", None, f"motionBlur {mb} read as {{\"samples\": {int(mb)}}}",
                f"write \"motionBlur\": {{\"samples\": {int(mb)}}}")
    elif mb not in (None, False) and not isinstance(mb, dict):
        rep.add("error", "bad-value", None, f"motionBlur must be an object like {{\"samples\": 8, \"shutter\": 0.5}}, "
                f"got {mb!r}")
    if isinstance(spec.get("grade"), dict):
        _look(spec, "grade", GRADE, {"saturation": "saturate"}, rep, "grade")
    if isinstance(spec.get("motionBlur"), dict):
        _keys(spec["motionBlur"], MOTION_BLUR, {}, rep, "motionBlur", "motionBlur: ")
        _numbers(spec["motionBlur"], ("samples",), "motionBlur.", rep, "motionBlur", least=0)  # under 2 is off
        _numbers(spec["motionBlur"], ("shutter",), "motionBlur.", rep, "motionBlur", least=0)
    cam = spec.get("camera")
    if isinstance(cam, dict):
        _keys(cam, CAMERA, CAMERA_ALIAS, rep, "camera", "camera: ")
        _values(cam, "camera", rep, "camera")
        _target(cam, rep)
        _shake(cam, rep)
        _focus(cam, rep)
        fov = cam.get("fov")
        if isinstance(fov, (int, float)) and not isinstance(fov, bool) and not 5 <= fov <= 150:
            rep.add("error", "bad-value", "camera", f"fov is {fov:g}; it must be between 5 and 150 degrees",
                    "40 is the default; lower flattens perspective, higher exaggerates it")
    elif cam is not None:
        rep.add("error", "bad-value", "camera", f"camera must be an object like {{\"z\": -200, \"rotateY\": 10}}, "
                f"got {type(cam).__name__}")
        spec.pop("camera")
    _light(spec, rep, None)
    _fog(spec, rep)
    audio = spec.get("audio")
    if isinstance(audio, (dict, str)):
        audio = spec["audio"] = [audio]
    elif audio is not None and not isinstance(audio, list):
        rep.add("error", "bad-value", None, f"audio is a list of files or objects with 'src', got {audio!r}",
                "\"audio\": [{\"src\": \"music.mp3\"}]")
        audio = spec["audio"] = []
    for i, a in enumerate(audio or []):
        if isinstance(a, str):
            audio[i] = a = {"src": a}
        if isinstance(a, dict):
            owner = _id(a, rep, f"audio[{i}]")
            _keys(a, AUDIO, AUDIO_ALIAS, rep, owner, "")
            _timing(a, rep, owner)
            _audio_values(a, rep, owner, "")
        else:
            rep.add("error", "bad-value", f"audio[{i}]", f"an audio entry is a file name or an object with 'src', "
                    f"got {type(a).__name__}")
    if spec.get("mix") is not None:
        _mix(spec["mix"], rep)
    layers = spec.get("layers")
    if layers is None or layers == []:
        rep.add("error", "no-layers", None, "the scene has no layers", "add \"layers\": [ ... ] with at least one layer")
        spec["layers"] = []
    elif not isinstance(layers, list):
        rep.add("error", "bad-layers", None, f"'layers' must be a list, got {type(layers).__name__}")
        spec["layers"] = []
    comps = spec.get("components") or {}
    if isinstance(comps, dict):
        _components(comps, rep)
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
