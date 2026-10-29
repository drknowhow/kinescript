"""Scene loading: theme variables, components, sequences, cue timing, compiled properties."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from . import components, media
from . import presets as P
from .props import Ctx, Prop, compile_prop
from .timing import Timeline

LAYER_TYPES = {"rect", "circle", "ellipse", "path", "line", "arrow", "polygon", "text", "counter",
               "image", "video", "group", "captions", "shader"}
STRUCT = {"id", "type", "at", "dur", "until", "enter", "exit", "children", "layers", "src", "trim",
          "speed", "fit", "text", "cues", "d", "points", "font", "split", "format", "align", "anchor",
          "blend", "clip", "mode", "volume", "hidden", "note", "words", "stops", "cap", "join", "loop",
          "allowOffscreen", "style", "audio", "shadow", "box", "fx", "fill", "stroke", "emph", "active",
          "sksl", "preset", "uniforms", "caret", "case", "italic", "smooth", "closed", "focus", "use"}
# Property defaults by name; everything not listed defaults to None.
DEFAULTS = {"x": 0.0, "y": 0.0, "opacity": 1.0, "scale": 1.0, "scaleX": 1.0, "scaleY": 1.0, "rotate": 0.0,
            "blur": 0.0, "draw": 1.0, "zoom": 1.0}


class SceneError(ValueError):
    pass


class Layer:
    __slots__ = ("id", "type", "raw", "t_in", "t_out", "props", "enter", "exit", "children", "parent",
                 "index", "src", "video", "ctx")

    def __init__(self, lid: str, ltype: str, raw: dict, parent: "Layer | None", index: int):
        self.id, self.type, self.raw, self.parent, self.index = lid, ltype, raw, parent, index
        self.t_in = self.t_out = 0.0
        self.props: dict[str, Prop] = {}
        self.enter = self.exit = None
        self.children: list[Layer] = []
        self.src: str | None = None
        self.video: media.VideoSource | None = None
        self.ctx: Ctx | None = None

    @property
    def dur(self) -> float:
        return self.t_out - self.t_in

    def get(self, name: str, t: float, T: float = 0.0, i: int = 0, default=None):
        p = self.props.get(name)
        if p is None:
            return DEFAULTS.get(name, default)
        return p.at(t, self.dur, T, i)

    def visible(self, T: float) -> bool:
        return self.t_in <= T < self.t_out and not self.raw.get("hidden")

    def __repr__(self):
        return f"<{self.type} {self.id} {self.t_in:.2f}-{self.t_out:.2f}>"


class Scene:
    def __init__(self, spec: dict, base_dir: str | Path = "."):
        self.spec = spec
        self.base = Path(base_dir)
        cv = spec.get("canvas", {})
        self.w = int(cv.get("w", cv.get("width", 1920)))
        self.h = int(cv.get("h", cv.get("height", 1080)))
        self.fps = float(cv.get("fps", 30))
        self.ctx = Ctx(self.w, self.h, self.fps)
        theme = dict(components.THEME, **spec.get("theme", {}))
        self.theme = theme
        self.bg = cv.get("bg", "#0a0c10")
        self.markers = spec.get("markers", {})
        self.grade = spec.get("grade") or {}
        self.motion_blur = spec.get("motionBlur") or {}
        self.audio = spec.get("audio", [])
        raw_layers = _subst(copy.deepcopy(spec.get("layers", [])), theme)
        self.bg = _subst(self.bg, theme)
        user_comps = spec.get("components", {})
        raw_layers = [components.expand(l, user_comps, theme, self) for l in raw_layers]
        raw_layers = [x for l in raw_layers for x in (l if isinstance(l, list) else [l])]
        self.timeline = Timeline(self.fps, self.markers)
        self.layers: list[Layer] = []
        self.by_id: dict[str, Layer] = {}
        self._build(raw_layers, None, self.layers)
        try:
            self.duration = self.timeline.solve(cv.get("duration"))
        except ValueError as e:
            raise SceneError(str(e)) from e
        for layer in self.by_id.values():
            layer.t_in, layer.t_out = self.timeline.resolved[layer.id]
        self.frames = max(1, round(self.duration * self.fps))

    def path(self, src: str) -> str:
        p = Path(src)
        return str(p if p.is_absolute() else (self.base / p))

    def _build(self, raws: list, parent: Layer | None, out: list):
        prev = None
        for idx, raw in enumerate(raws):
            if not isinstance(raw, dict):
                raise SceneError(f"layer #{idx} must be an object, got {type(raw).__name__}")
            ltype = raw.get("type") or _infer_type(raw)
            if ltype not in LAYER_TYPES:
                raise SceneError(f"layer #{idx} ({raw.get('id', '?')}): unknown type '{ltype}'. "
                                 f"Known: {', '.join(sorted(LAYER_TYPES))}")
            lid = str(raw.get("id") or f"{ltype}{len(self.by_id) + 1}")
            if lid in self.by_id:
                raise SceneError(f"duplicate layer id '{lid}'")
            if ltype == "captions":
                raw = dict({"x": "50%", "y": "86%", "size": 46, "weight": 700}, **raw)
            layer = Layer(lid, ltype, raw, parent, len(self.by_id))
            self.by_id[lid] = layer
            natural = None
            if ltype in ("video", "image") or (ltype == "captions" and raw.get("src")):
                layer.src = self.path(raw["src"]) if raw.get("src") else None
            if ltype == "video" and layer.src:
                info = media.probe(layer.src)
                trim = raw.get("trim") or [0, None]
                end = trim[1] if len(trim) > 1 and trim[1] is not None else info["duration"]
                natural = max(0.04, (end - trim[0]) / float(raw.get("speed", 1)))
            if ltype == "captions":
                raw = layer.raw = dict(raw, cues=components.load_cues(raw, layer.src))
                if raw["cues"] and "dur" not in raw and "until" not in raw:
                    natural = max(c[1] for c in raw["cues"])
            self.timeline.add(lid, raw.get("at", 0), raw.get("dur"), raw.get("until"), natural,
                              prev=prev, parent=parent.id if parent else None,
                              default_end=parent.id if parent else None)
            ctx = self.ctx
            if ltype == "group" and ("w" in raw or "h" in raw):
                ctx = Ctx(_num(raw.get("w"), self.w), _num(raw.get("h"), self.h), self.fps)
            layer.ctx = ctx
            pctx = parent.ctx if parent and parent.ctx else self.ctx
            try:
                for k, v in raw.items():
                    if k in ("fill", "stroke", "emph", "active") and not isinstance(v, dict):
                        layer.props[k] = compile_prop(k, v, pctx)
                    elif k not in STRUCT:
                        layer.props[k] = compile_prop(k, v, pctx)
                for un, uv in (raw.get("uniforms") or {}).items():
                    if not isinstance(uv, (int, float, str)) or (isinstance(uv, str) and uv.startswith("=")):
                        layer.props["u_" + un] = compile_prop("u_" + un, uv, pctx)
                layer.enter = P.parse(raw.get("enter"))
                layer.exit = P.parse(raw.get("exit"))
            except (ValueError, SyntaxError) as e:
                raise SceneError(f"layer '{lid}': {e}") from e
            out.append(layer)
            kids = raw.get("children") or raw.get("layers")
            if ltype == "group" and kids:
                kids = [components.expand(k, self.spec.get("components", {}), self.theme, self) for k in kids]
                kids = [x for k in kids for x in (k if isinstance(k, list) else [k])]
                self._build(kids, layer, layer.children)
            prev = lid

    def all_layers(self):
        def walk(ls):
            for layer in ls:
                yield layer
                yield from walk(layer.children)
        return walk(self.layers)

    def summary(self) -> dict:
        return {"canvas": {"w": self.w, "h": self.h, "fps": self.fps, "duration": round(self.duration, 3),
                           "frames": self.frames},
                "layers": [{"id": l.id, "type": l.type, "in": round(l.t_in, 3), "out": round(l.t_out, 3),
                            "parent": l.parent.id if l.parent else None} for l in self.all_layers()]}

    def close(self):
        for layer in self.all_layers():
            if layer.video:
                layer.video.close()


def _num(v, ref):
    if v is None:
        return ref
    if isinstance(v, str) and v.endswith("%"):
        return float(v[:-1]) / 100 * ref
    return float(v)


def _infer_type(raw: dict) -> str:
    for key, t in (("text", "text"), ("cues", "captions"), ("d", "path"), ("points", "path"),
                   ("children", "group"), ("layers", "group"), ("r", "circle")):
        if key in raw:
            return t
    if "src" in raw:
        return "video" if Path(str(raw["src"])).suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v") else "image"
    return "rect"


_VAR = re.compile(r"\$([A-Za-z_]\w*)")


def _subst(v, theme: dict):
    if isinstance(v, str):
        if v.startswith("$") and v[1:] in theme:
            return copy.deepcopy(theme[v[1:]])
        if "$" in v and not v.startswith("="):
            return _VAR.sub(lambda m: str(theme.get(m.group(1), m.group(0))), v)
        return v
    if isinstance(v, list):
        return [_subst(x, theme) for x in v]
    if isinstance(v, dict):
        return {k: _subst(x, theme) for k, x in v.items()}
    return v


def load(src) -> Scene:
    """Load a scene from a .json path, a JSON string or a dict."""
    if isinstance(src, Scene):
        return src
    if isinstance(src, dict):
        return Scene(src)
    s = str(src)
    if s.lstrip().startswith("{"):
        return Scene(json.loads(s))
    p = Path(s)
    if not p.exists():
        raise SceneError(f"scene file not found: {p}")
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SceneError(f"{p.name}: invalid JSON at line {e.lineno} col {e.colno}: {e.msg}") from e
    return Scene(spec, p.parent)
