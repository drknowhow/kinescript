"""Scene loading: theme variables, components, sequences, cue timing, compiled properties."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from . import components, media, models, schema
from . import presets as P
from . import voice as V
from . import words as W
from .props import Ctx, Prop, compile_prop
from .timing import Timeline, parse_clock

LAYER_TYPES = {"rect", "circle", "ellipse", "path", "line", "arrow", "polygon", "text", "counter",
               "image", "video", "group", "captions", "shader", "model"}
STRUCT = {"id", "type", "at", "dur", "until", "enter", "exit", "children", "layers", "src", "trim",
          "speed", "fit", "text", "cues", "d", "points", "font", "split", "format", "align", "anchor",
          "blend", "clip", "mode", "volume", "hidden", "note", "words", "stops", "cap", "join", "loop",
          "allowOffscreen", "style", "audio", "shadow", "box", "fx", "fill", "stroke", "emph", "active",
          "sksl", "preset", "uniforms", "caret", "case", "italic", "smooth", "closed", "focus", "use",
          "backface", "fixed", "sort", "shape", "detail", "parts", "light", "span", "sound", "voice"}
# Property defaults by name; everything not listed defaults to None.
DEFAULTS = {"x": 0.0, "y": 0.0, "opacity": 1.0, "scale": 1.0, "scaleX": 1.0, "scaleY": 1.0, "rotate": 0.0,
            "blur": 0.0, "draw": 1.0, "zoom": 1.0, "z": 0.0, "rotateX": 0.0, "rotateY": 0.0}
DEPTH_PROPS = ("z", "rotateX", "rotateY")


class SceneError(ValueError):
    """A scene that cannot be built. `issues` lists every problem found, each with its layer id."""

    def __init__(self, message: str, issues: list[dict] | None = None, layer: str | None = None):
        super().__init__(message)
        self.issues = issues or [{"level": "error", "code": "load", "layer": layer, "message": message, "fix": ""}]

    @classmethod
    def from_issues(cls, issues: list[dict]) -> "SceneError":
        errs = [i for i in issues if i["level"] == "error"]
        lines = [(f"layer '{i['layer']}': " if i.get("layer") else "") + i["message"] for i in errs]
        msg = lines[0] if len(lines) == 1 else f"{len(lines)} problems:\n  " + "\n  ".join(lines)
        return cls(msg, issues)


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
        try:
            return p.at(t, self.dur, T, i)
        except (ArithmeticError, ValueError, TypeError) as e:  # a user expression, e.g. 1/(t-0.7) at t=0.7
            raise SceneError(f"layer '{self.id}': {name} failed at t={t:.3f}s (scene {T:.3f}s): "
                             f"{type(e).__name__}: {e}", layer=self.id) from e

    def visible(self, T: float) -> bool:
        return self.t_in <= T < self.t_out and not self.raw.get("hidden")

    def __repr__(self):
        return f"<{self.type} {self.id} {self.t_in:.2f}-{self.t_out:.2f}>"


class Sound:
    """An audio entry, or a layer's sound, placed on the timeline like a layer."""

    def __init__(self, sid: str, raw: dict, src: str | None, layer: str | None = None, natural: float | None = None):
        self.id, self.raw, self.src, self.layer = sid, raw, src, layer
        self.natural = natural  # seconds of trimmed file; None when it loops or can't be read
        self.role = raw.get("role")
        self.t_in = self.t_out = 0.0


class Scene:
    def __init__(self, spec: dict, base_dir: str | Path = ".", notes: list[dict] | None = None):
        spec, report = schema.normalize(spec)
        self.issues: list[dict] = list(notes or []) + report.issues
        self._raise_errors()
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
        self._sounding: list[Layer] = []
        raw_layers = _subst(copy.deepcopy(spec.get("layers", [])), theme)
        self.bg = _subst(self.bg, theme)
        self._components = user_comps = _subst(copy.deepcopy(spec.get("components", {})), theme)
        try:
            raw_layers = [components.expand(l, user_comps, theme, self) for l in raw_layers]
        except ValueError as e:
            raise SceneError(str(e)) from e
        raw_layers = [x for l in raw_layers for x in (l if isinstance(l, list) else [l])]
        self.words = self._read_words()
        self._raise_errors()
        self.timeline = Timeline(self.fps, self.markers, {sid: dict(w, ids=None if w["words"] is None else {
            x["id"]: (x["start"], x["end"]) for x in w["words"] if x["id"]}) for sid, w in self.words.items()})
        self.layers: list[Layer] = []
        self.by_id: dict[str, Layer] = {}
        self._build(raw_layers, None, self.layers)
        self.camera = self._camera(_subst(spec.get("camera"), theme))
        self.light = _subst(spec.get("light"), theme)
        self.audio = self._build_audio()
        self._check_media()
        self._raise_errors()
        try:
            self.duration = self.timeline.solve(cv.get("duration"))
        except ValueError as e:
            lid = getattr(e, "layer", None)
            raise SceneError((f"layer '{lid}': " if lid else "") + str(e), layer=lid) from e
        for layer in self.by_id.values():
            layer.t_in, layer.t_out = (_on_grid(t, self.fps) for t in self.timeline.resolved[layer.id])
        for s in self.audio:
            s.t_in, s.t_out = self.timeline.resolved[s.id]
        if self.camera is not None:
            self.camera.t_out = self.duration
        self.depth = {L.id for L in self.layers if uses_depth(L)}
        self.is3d = self.camera is not None or bool(self.depth)
        self.frames = max(1, round(self.duration * self.fps))
        self._check_values()
        self._raise_errors()

    def note(self, level: str, code: str, layer, message: str, fix: str = ""):
        self.issues.append({"level": level, "code": code, "layer": layer, "message": message, "fix": fix})

    def _raise_errors(self):
        if any(i["level"] == "error" for i in self.issues):
            raise SceneError.from_issues(self.issues)

    def _check_media(self):
        """Missing files fail here, before any frame is rendered, with the path that was tried."""
        def missing(src: str, owner, what: str) -> bool:
            p = Path(src)
            if p.exists():
                return False
            near = sorted(x.name for x in p.parent.iterdir()) if p.parent.is_dir() else []
            close = schema.suggest(p.name, near) if near else None
            fix = (f"did you mean '{close}'? " if close else "") + (
                f"files in {p.parent}: {', '.join(near[:12])}" if near else f"folder {p.parent} does not exist")
            self.note("error", "media-missing", owner, f"{what} not found: {p}",
                      fix + " (paths are relative to the scene file)")
            return True

        for layer in self.all_layers():
            if layer.src and not missing(layer.src, layer.id, f"{layer.type} src") and layer.type == "model":
                try:
                    models.load(layer.src)
                except ValueError as e:
                    self.note("error", "bad-model", layer.id, f"model src {e}",
                              "export it again as glTF 2.0 (.glb) or .obj")
        for s in self.audio:
            owner = s.layer or s.id
            if s.raw.get("text") and s.src and not Path(s.src).exists():
                continue  # not spoken yet: reported by _read_words
            if not s.src or missing(s.src, owner, "sound src" if s.layer else "audio src"):
                continue
            try:
                length = media.probe(s.src)["duration"]
            except media.MediaError as e:
                self.note("error", "bad-media", owner, str(e), "export it again as .wav, .mp3, .m4a or .flac")
                continue
            _check_trim(self, owner, s.raw.get("trim") or [0, None], length)

    def _check_values(self):
        """Evaluate every property at the start, middle and end of its layer, so a bad value is reported
        now, with its layer and field, instead of as a Python error halfway through a render."""
        for layer in self.all_layers():
            d = layer.dur
            for name, prop in layer.props.items():
                if name.startswith("u_"):
                    continue
                for t in (0.0, d / 2, max(0.0, d - 1e-3), d):  # d: the last keyframe value itself
                    try:
                        v = prop.at(t, d, layer.t_in + t, 0)
                    except Exception as e:  # noqa: BLE001 - any failure inside a user expression
                        self.note("error", "bad-expression", layer.id,
                                  f"{name} failed at t={t:.2f}s: {type(e).__name__}: {e}")
                        break
                    if name in schema.COLOR_KEYS:
                        if isinstance(v, str):
                            self.note("error", "bad-colour", layer.id, f"{name} '{v}' is not a colour",
                                      "use #rrggbb, #rrggbbaa, rgb(...) or a CSS colour name")
                            break
                    elif name in schema.SCALAR and (isinstance(v, bool) or not isinstance(v, (int, float))):
                        self.note("error", "bad-value", layer.id, f"{name} is {v!r} at t={t:.2f}s; expected a number",
                                  "use a number, a percentage like '50%', or an expression starting with '='")
                        break

    def check_expressions(self, budget: int = 200_000):
        """An expression can fail at one moment only, like 1/(t-0.7) at t=0.7. `kine check` evaluates each at every
        frame its layer is on (every few frames past `budget` evaluations in all), so it finds that, not the render.
        Not part of loading: render workers load the scene too, and 2400 expressions take about 0.15 s."""
        spans = [(L, name, prop, math.ceil(L.t_in * self.fps - 1e-6), math.ceil(L.t_out * self.fps - 1e-6))
                 for L in self.all_layers() for name, prop in L.props.items()
                 if not prop.static and isinstance(prop.value, str)]
        stride = max(1, -(-sum(n1 - n0 for *_, n0, n1 in spans) // budget))
        for layer, name, prop, n0, n1 in spans:
            for n in range(n0, n1, stride):
                T = n / self.fps
                t = T - layer.t_in
                try:
                    prop.at(t, layer.dur, T, 0)
                except Exception as e:  # noqa: BLE001 - any failure inside a user expression
                    self.note("error", "bad-expression", layer.id, f"{name} failed at t={t:.2f}s (frame {n}): "
                              f"{type(e).__name__}: {e}", "guard the expression, e.g. max(…, 0.001) under a division")
                    break

    def path(self, src: str) -> str:
        p = Path(src)
        return str(p if p.is_absolute() else (self.base / p))

    def _read_words(self) -> dict[str, dict]:
        """Each audio entry's word times, read before the layers so cues and captions can use them:
        {id: {"words": [...] or None, "trim0", "trim1" (the part of the file that plays), "file"}}."""
        out = {}
        for i, a in enumerate(self.spec.get("audio") or []):
            sid = str(a.get("id") or f"audio{i + 1}")
            trim = a.get("trim") if isinstance(a.get("trim"), (list, tuple)) and a.get("trim") else [0, None]
            try:
                t0 = float(trim[0])
                t1 = float(trim[1]) if len(trim) > 1 and trim[1] is not None else None
            except (TypeError, ValueError):
                t0, t1 = 0.0, None  # reported by _check_media
            src = a.get("src") or (V.default_src(sid) if a.get("text") else None)
            out[sid] = {"words": None, "trim0": t0, "trim1": t1,
                        "file": Path(src).with_suffix(".words.json").as_posix() if src else "a words file"}
            if src is None:
                continue
            path, near = V.paths(sid, a, self.base)
            if a.get("text") and not path.exists():
                self.note("error", "voice-missing", sid, f"'{sid}' has text but is not spoken yet: {src} does not exist",
                          "run kine voice on this scene; it writes the file and its word times")
                continue
            rec = V.made(near) if a.get("text") else None
            if rec and rec.get("hash") != V.fingerprint(a):
                self.note("warning", "voice-stale", sid, f"the text, speaker, rate, backend or options of '{sid}' "
                          f"changed after {src} was spoken, so it says the old text",
                          "run kine voice on this scene to speak it again")
            given = a.get("words")
            if given is None and not near.exists():
                continue
            try:
                out[sid]["words"] = W.read(near if given is None else self.path(given) if isinstance(given, str)
                                           else given)
            except W.WordsError as e:
                self.note("error", "bad-words", sid, str(e), "a words file is {\"words\": [{\"word\": \"Hello\", "
                          "\"start\": 0.1, \"end\": 0.4}]}, in seconds of the sound file")
        return out

    def _voice_cues(self, lid: str, raw: dict) -> dict:
        """Captions that follow a voice: a cue a line, each word lit when it is said, starting when the voice starts."""
        v = str(raw["voice"])
        w = self.words.get(v)
        if w is None:
            close = schema.suggest(v, self.words)
            raise SceneError(f"layer '{lid}': voice '{v}' is not an audio id." + (f" Did you mean '{close}'?" if close else "")
                             + (f" Audio ids: {', '.join(list(self.words)[:16])}" if self.words else ""), layer=lid)
        if w["words"] is None:
            raise SceneError(f"layer '{lid}': '{v}' has no word times for the captions to follow. Add {w['file']} "
                             "(kine voice writes it for a sound with text; for a recording, a speech-to-text tool that "
                             "times words, such as Whisper, writes one), or give the captions cues", layer=lid)
        return dict(raw, cues=W.cues(w["words"], w["trim0"], w["trim1"]), at=raw.get("at", v))

    def _build_audio(self) -> list[Sound]:
        """Audio entries and layer sounds as timeline nodes, so a layer can cue from a sound and a sound from a layer.
        Only voice sets an "auto" duration: a sentence cut off at the end is never wanted, a long music bed is."""
        entries: list[tuple[dict, str | None]] = [(a, None) for a in self.spec.get("audio") or []]
        for L in self._sounding:
            if not any(x.raw.get("hidden") for x in _lineage(L)):
                entries.append((self._layer_sound(L), L.id))
        out, prev = [], None
        for i, (a, layer) in enumerate(entries):
            sid = str(a.get("id") or (f"{layer}-sound" if layer else f"audio{i + 1}"))
            src = self.path(a.get("src") or V.default_src(sid)) if a.get("src") or a.get("text") else None
            natural = _natural(a, src)
            try:
                self.timeline.add(sid, a.get("at", 0), a.get("dur"), a.get("until"), natural,
                                  prev=None if layer else prev, counts=a.get("role") == "voice")
            except ValueError as e:
                raise SceneError(f"audio '{sid}': {e}", layer=sid) from e
            if not layer:
                prev = sid
            g = a.get("gain")
            if g is not None and (isinstance(g, bool) or not isinstance(g, (int, float))):
                self._check_gain(layer or sid, g)
            out.append(Sound(sid, a, src, layer, natural))
        return out

    def _check_gain(self, owner: str, g):
        """A gain that changes over time (keyframes, a tween or an expression) must give a number of dB."""
        try:
            prop = compile_prop("gain", g, self.ctx)
            for t in (0.0, 0.5, 1.0):
                v = prop.at(t, 1.0, t, 0)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"gives {v!r}")
        except Exception as e:  # noqa: BLE001 - any failure inside a user expression
            self.note("error", "bad-value", owner, f"gain {g!r} is not a number of dB: {e}",
                      "use a number, keyframes like [[0, -20], [1, 0]], or an expression starting with '='")

    def _layer_sound(self, L: Layer) -> dict:
        """A layer's sound as an audio entry: "at" counts from the layer ("in", "out-0.1", 0.2)."""
        s = L.raw["sound"]
        s = {"src": s} if isinstance(s, str) else dict(s)
        at = s.get("at", "in")
        if isinstance(at, str) and parse_clock(at, self.fps) is not None:
            at = parse_clock(at, self.fps)
        if isinstance(at, (int, float)) and not isinstance(at, bool):
            at = f"{L.id}.in{'-' if at < 0 else '+'}{abs(at):.6f}"
        elif isinstance(at, str) and re.match(r"\s*(in|out|mid)\b", at):
            at = f"{L.id}.{at.strip()}"
        s["at"] = at
        s.setdefault("role", "sfx")
        return s

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
            if raw.get("sound") is not None:
                self._sounding.append(layer)
            natural = None
            if ltype in ("video", "image", "model") or (ltype == "captions" and raw.get("src")):
                layer.src = self.path(raw["src"]) if raw.get("src") else None
            if ltype == "video" and layer.src and Path(layer.src).exists():
                info = media.probe(layer.src)
                trim = _check_trim(self, lid, raw.get("trim") or [0, None], info["duration"])
                end = trim[1] if trim[1] is not None else info["duration"]
                end = min(float(end), info["duration"]) if info["duration"] else end
                speed = raw.get("speed", 1)
                if not _number(speed) or speed <= 0:
                    self.note("error", "bad-value", lid, f"speed is {speed!r}; it must be a number above 0",
                              "2 plays twice as fast, 0.5 at half speed")
                    speed = 1
                natural = max(0.04, (end - trim[0]) / float(speed))
            if ltype == "captions" and raw.get("voice") is not None:
                raw = layer.raw = self._voice_cues(lid, raw)
            elif ltype == "captions" and (raw.get("cues") or (layer.src and Path(layer.src).exists())):
                raw = layer.raw = dict(raw, cues=components.load_cues(raw, layer.src))
            if ltype == "captions" and raw.get("cues") and "dur" not in raw and "until" not in raw:
                natural = max(c[1] for c in raw["cues"])
            try:
                self.timeline.add(lid, raw.get("at", 0), raw.get("dur"), raw.get("until"), natural,
                                  prev=prev, parent=parent.id if parent else None,
                                  default_end=parent.id if parent else None,
                                  span=ltype == "group" and bool(raw.get("span")))
            except ValueError as e:
                raise SceneError(f"layer '{lid}': {e}", layer=lid) from e
            ctx = self.ctx
            if ltype == "group" and ("w" in raw or "h" in raw):
                ctx = Ctx(_num(raw.get("w"), self.w), _num(raw.get("h"), self.h), self.fps)
            layer.ctx = ctx
            pctx = parent.ctx if parent and parent.ctx else self.ctx
            try:
                for k, v in raw.items():
                    if k in schema.COLOR_KEYS and (not isinstance(v, dict) or "to" in v or "keys" in v):
                        layer.props[k] = compile_prop(k, v, pctx)  # a colour, or a tween of one; not a gradient
                    elif k not in STRUCT:
                        layer.props[k] = compile_prop(k, v, pctx)
                for un, uv in (raw.get("uniforms") or {}).items():
                    if not isinstance(uv, (int, float, str)) or (isinstance(uv, str) and uv.startswith("=")):
                        layer.props["u_" + un] = compile_prop("u_" + un, uv, pctx)
                layer.enter = P.parse(raw.get("enter"))
                layer.exit = P.parse(raw.get("exit"))
            except (ValueError, SyntaxError) as e:
                raise SceneError(f"layer '{lid}': {e}", layer=lid) from e
            out.append(layer)
            kids = raw.get("children") or raw.get("layers")
            if ltype == "group" and kids:
                kids = [components.expand(k, self._components, self.theme, self) for k in kids]
                kids = [x for k in kids for x in (k if isinstance(k, list) else [k])]
                self._build(kids, layer, layer.children)
            prev = lid

    def _camera(self, raw) -> Layer | None:
        """The camera as a layer that spans the scene, so its keys animate like any layer's."""
        if raw is None:
            return None
        cam = Layer("camera", "camera", raw, None, -1)
        try:
            cam.props = {k: compile_prop(k, v, self.ctx) for k, v in raw.items()}
        except (ValueError, SyntaxError) as e:
            raise SceneError(f"camera: {e}", layer="camera") from e
        return cam

    def all_layers(self):
        def walk(ls):
            for layer in ls:
                yield layer
                yield from walk(layer.children)
        return walk(self.layers)

    def summary(self) -> dict:
        audio = []
        for s in self.audio:
            src = s.raw.get("src") or (V.default_src(s.id) if s.raw.get("text") else None)
            audio.append({"id": s.id, "role": s.role, "src": src, "in": round(s.t_in, 3),
                          "out": round(s.t_out, 3), "layer": s.layer})
            if said := self.said(s.id):  # the ids a cue can name, and when each is said
                audio[-1]["words"] = [[wid, round(a, 3), round(b, 3)] for wid, _, a, b in said]
        return {"canvas": {"w": self.w, "h": self.h, "fps": self.fps, "duration": round(self.duration, 3),
                           "frames": self.frames},
                "layers": [{"id": l.id, "type": l.type, "in": round(l.t_in, 3), "out": round(l.t_out, 3),
                            "parent": l.parent.id if l.parent else None} for l in self.all_layers()],
                "audio": audio}

    def said(self, sid: str) -> list[tuple[str | None, str, float, float]]:
        """The words sound `sid` plays, as (id, word, start, end) in scene seconds; [] when it has no words."""
        w, s = self.words.get(sid), next((s for s in self.audio if s.id == sid), None)
        if not w or not w["words"] or s is None:
            return []
        lag = s.t_in - w["trim0"]
        return [(x["id"], x["word"], lag + x["start"], lag + x["end"]) for x in w["words"]
                if x["start"] >= w["trim0"] and (w["trim1"] is None or x["start"] < w["trim1"])
                and lag + x["start"] < min(s.t_out, self.duration)]

    def close(self):
        for layer in self.all_layers():
            if layer.video:
                layer.video.close()


def uses_depth(L: Layer) -> bool:
    """True when the layer or anything inside it moves in depth."""
    motions = [m for m in (L.enter, L.exit) if m]
    return (L.type == "model" or any(k in L.props for k in DEPTH_PROPS) or any(P.DEPTH & set(m.names) for m in motions)
            or any(uses_depth(c) for c in L.children))


def _lineage(L: Layer):
    while L is not None:
        yield L
        L = L.parent


def _natural(a: dict, src: str | None) -> float | None:
    """An audio entry's length from its trimmed source; None (runs to the scene end) when it loops or can't be read."""
    trim = a.get("trim") or [0, None]
    if a.get("loop") or not src or not Path(src).exists() or not isinstance(trim, (list, tuple)):
        return None
    try:
        total = media.probe(src)["duration"]
    except media.MediaError:
        return None  # reported by _check_media
    end = trim[1] if len(trim) > 1 and trim[1] is not None else total
    return max(0.01, float(end) - float(trim[0]))


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_trim(scene: "Scene", owner, trim, length: float) -> list:
    """Note what is wrong with a trim; returns it as [start, end or None] in seconds, or [0, None] if unusable."""
    if not isinstance(trim, (list, tuple)) or not 1 <= len(trim) <= 2 or not _number(trim[0]) \
            or (len(trim) > 1 and trim[1] is not None and not _number(trim[1])):
        scene.note("error", "bad-trim", owner, f"trim must be [start, end] in source seconds, got {trim!r}",
                   "[1.5, 4] plays 1.5 s to 4 s of the file; [1.5] plays from 1.5 s to its end")
        return [0.0, None]
    a = float(trim[0])
    b = trim[1] if len(trim) > 1 else None
    if a < 0:  # the clip would start before its file does, and its sound would play a second early
        scene.note("error", "bad-trim", owner, f"trim starts at {a:g}s, before the start of the source",
                   "use a start of 0 or more; to start the clip later in the scene, use \"at\"")
        return [0.0, None]
    if not length:
        return [a, b]
    if a >= length:
        scene.note("error", "bad-trim", owner, f"trim starts at {a:g}s but the source is only {length:.2f}s long",
                   f"use a start below {length:.2f}")
    elif b is not None and float(b) <= a:
        scene.note("error", "bad-trim", owner, f"trim end {b} is not after its start {a:g}")
    elif b is not None and float(b) > length + 0.05:
        scene.note("warning", "trim-past-end", owner,
                   f"trim ends at {float(b):g}s but the source is {length:.2f}s long, so it stops at {length:.2f}s",
                   f"use [{a:g}, {length:.2f}] or leave out the end")
    return [a, b]


def _on_grid(t: float | None, fps: float) -> float | None:
    """t, or the frame boundary it is a float error away from. 2.2 + 8.9 is 11.100000000000001, so a layer at 2.2
    for 8.9 s also showed on frame 333 (11.1 s), and one at its out on the frame after it."""
    if t is None:
        return t
    k = round(t * fps)
    return k / fps if abs(t * fps - k) < 1e-6 else t


def _num(v, ref):
    if v is None:
        return ref
    if isinstance(v, str) and v.endswith("%"):
        return float(v[:-1]) / 100 * ref
    return float(v)


def _infer_type(raw: dict) -> str:
    for key, t in (("shape", "model"), ("parts", "model"), ("text", "text"), ("cues", "captions"), ("d", "path"),
                   ("points", "path"), ("children", "group"), ("layers", "group"), ("r", "circle")):
        if key in raw:
            return t
    if "src" in raw:
        ext = Path(str(raw["src"])).suffix.lower()
        return ("video" if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v") else
                "model" if ext in models.EXTENSIONS else "image")
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
        text, base, name = s, Path("."), "scene"
    else:
        p = Path(s)
        if not p.exists():
            raise SceneError(f"scene file not found: {p}")
        text, base, name = p.read_text(encoding="utf-8-sig"), p.parent, p.name
    try:
        spec, fixed = schema.loads(text)
    except json.JSONDecodeError as e:
        lines = text.splitlines()
        line = lines[e.lineno - 1].strip()[:120] if 0 < e.lineno <= len(lines) else ""
        raise SceneError(f"{name}: invalid JSON at line {e.lineno} col {e.colno}: {e.msg}. The line is: {line}") from e
    notes = [{"level": "info", "code": "json", "layer": None, "message": f"the JSON had {', '.join(fixed)}; read anyway",
              "fix": "write strict JSON"}] if fixed else []
    return Scene(spec, base, notes)
