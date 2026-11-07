"""Avatars: a character drawn once as parts, in a folder with an avatar.json, that scenes make talk and move.

    avatars/host/avatar.json
    avatars/host/parts/body.svg, head.svg, eyes-open.svg, eyes-closed.svg, mouth-X.svg, mouth-A.svg ...

Every part file is drawn at the avatar's full size, in place (as layers exported from one drawing), so a part's
pivot is a point in the avatar's own coordinates. Parts draw in the order avatar.json lists them, and a part with
a parent turns and moves with it.

    {"kinescriptAvatar": 1, "name": "Host", "size": [600, 900],
     "parts": {"body": {"src": "parts/body.svg", "pivot": [300, 700]},
               "head": {"src": "parts/head.svg", "parent": "body", "pivot": [300, 380]},
               "eyes": {"parent": "head", "states": {"open": "parts/eyes-open.svg", "closed": "parts/eyes-closed.svg"},
                        "look": [14, 8]},
               "mouth": {"parent": "head", "shapes": {"X": "parts/mouth-X.svg", "A": "parts/mouth-A.svg", ...}}},
     "colors": {"skin": "#e8c4a0", "shirt": "accent"},
     "expressions": {"smile": {"brows": {"y": -6}, "mouth": "C"}},
     "gestures": {"nod": {"dur": 0.5, "head": {"rotate": [[0, 0], [0.2, 6], [0.5, 0]]}}},
     "idle": {"blink": [2.5, 5], "breath": 0.012, "saccade": [0.8, 2.5]},
     "voice": {"backend": "kokoro", "speaker": "am_michael"}}

A part has one of src (one drawing), states (named drawings; "open" shows unless something else is asked,
"closed" is the blink) or shapes (the mouth: the shapes of mouth.py, by letter). The mouth draws X, A, C and D at
least; the others fall back as mouth.FALLBACK says. colors gives each name a colour or a palette role: a piece or
<g> in any part's svg whose id is that name is filled with it, so one avatar takes on each brand's palette.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import brand as B
from . import color as C
from . import mouth as M

VERSION = 1
FILE = "avatar.json"
TOP = {"kinescriptAvatar", "name", "description", "size", "parts", "colors", "expressions", "gestures", "idle",
       "voice", "mouthMode"}
PART_KEYS = {"src", "states", "shapes", "parent", "pivot", "look"}
POSE_KEYS = ("x", "y", "rotate", "scale", "opacity")  # what an expression or gesture may change on a part
IDLE_KEYS = {"blink", "breath", "saccade"}
EXTS = (".svg", ".png", ".webp")
MODES = ("swap", "morph")


class AvatarError(ValueError):
    def __init__(self, folder: Path, issues: list[dict]):
        self.folder, self.issues = folder, issues
        errs = [i for i in issues if i["level"] == "error"]
        super().__init__(f"the avatar in {folder} has {len(errs)} problem{'s' * (len(errs) != 1)}: "
                         + "; ".join(f"{i['message']} ({i['fix']})" if i["fix"] else i["message"] for i in errs))


@dataclass
class Part:
    name: str
    files: dict[str, Path]  # "" for a part with src, else its state or shape names
    kind: str  # "src" | "states" | "shapes"
    parent: str | None
    pivot: tuple[float, float]
    look: tuple[float, float] = (0.0, 0.0)


@dataclass
class Avatar:
    folder: Path
    spec: dict
    size: tuple[float, float]
    parts: dict[str, Part]
    mode: str = "swap"
    fallbacks: dict[str, str] = field(default_factory=dict)  # mouth shape -> the drawn shape it shows

    @property
    def mouth(self) -> Part | None:
        return next((p for p in self.parts.values() if p.kind == "shapes"), None)

    @property
    def eyes(self) -> list[Part]:
        return [p for p in self.parts.values() if p.kind == "states" and "closed" in p.files]

    def shape(self, s: str) -> str:
        """The drawn mouth shape that shows for shape s."""
        return s if self.mouth and s in self.mouth.files else self.fallbacks.get(s, "X")


def _issue(level: str, code: str, message: str, fix: str = "") -> dict:
    return {"level": level, "code": code, "layer": "avatar", "message": message, "fix": fix}


def read(folder: str | Path) -> dict:
    """avatar.json in `folder` as a dict. ValueError when it is missing or not a JSON object."""
    p = Path(folder) / FILE
    try:
        spec = json.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"no {FILE} in {folder}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} is not JSON: {e}") from None
    if not isinstance(spec, dict):
        raise ValueError(f"{p} must hold a JSON object")
    return spec


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _point(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(_num(x) for x in v)


def _file(folder: Path, rel, where: str, out: list) -> Path | None:
    try:
        p = B.inside(folder, rel)
    except ValueError as e:
        out.append(_issue("error", "avatar-file", f"{where}: {str(e).replace('brand folder', 'avatar folder')}"))
        return None
    if p.suffix.lower() not in EXTS:
        out.append(_issue("error", "avatar-file", f"{where}: '{rel}' is not {', '.join(EXTS)}",
                          "draw parts as .svg (or .png and .webp for pictures)"))
        return None
    if not p.is_file():
        out.append(_issue("error", "avatar-file", f"{where}: '{rel}' is not in the avatar folder"))
        return None
    return p


def _parts(folder: Path, spec: dict, out: list) -> dict[str, Part]:
    raw = spec.get("parts")
    if not isinstance(raw, dict) or not raw:
        out.append(_issue("error", "avatar-parts", "the avatar has no parts",
                          'add "parts": {"body": {"src": "parts/body.svg"}, ...}'))
        return {}
    parts: dict[str, Part] = {}
    for name, p in raw.items():
        where = f"part '{name}'"
        if not isinstance(p, dict):
            out.append(_issue("error", "avatar-part",
                              f"{where} must be an object such as {{\"src\": \"parts/{name}.svg\"}}"))
            continue
        for k in sorted(set(p) - PART_KEYS):
            out.append(_issue("warning", "avatar-key", f"{where} has an unknown key '{k}'",
                              "known keys: " + ", ".join(sorted(PART_KEYS))))
        kinds = [k for k in ("src", "states", "shapes") if k in p]
        if len(kinds) != 1:
            out.append(_issue("error", "avatar-part", f"{where} needs exactly one of src, states and shapes"
                              + (f"; it has {' and '.join(kinds)}" if kinds else "")))
            continue
        kind, files = kinds[0], {}
        if kind == "src":
            f = _file(folder, p["src"], where, out)
            files = {"": f} if f else {}
        elif not isinstance(p[kind], dict) or not p[kind]:
            example = ('{"open": "parts/eyes-open.svg", "closed": "parts/eyes-closed.svg"}' if kind == "states"
                       else '{"X": "parts/mouth-X.svg", "A": "parts/mouth-A.svg"}')
            out.append(_issue("error", "avatar-part", f"{where}: {kind} must name its drawings, such as {example}"))
            continue
        else:
            for k, rel in p[kind].items():
                if kind == "shapes" and k not in M.SHAPES:
                    out.append(_issue("error", "avatar-shape", f"{where}: '{k}' is not a mouth shape",
                                      f"the shapes are {', '.join(M.SHAPES)} (see kine probe --mouth)"))
                    continue
                f = _file(folder, rel, f"{where} {kind[:-1]} '{k}'", out)
                if f:
                    files[k] = f
        parent = p.get("parent")
        if parent is not None and (not isinstance(parent, str) or parent not in raw or parent == name):
            out.append(_issue("error", "avatar-parent", f"{where}: parent '{parent}' is not another part",
                              "parts: " + ", ".join(raw)))
            parent = None
        pivot = p.get("pivot", [0, 0])
        if not _point(pivot):
            out.append(_issue("error", "avatar-pivot", f"{where}: pivot must be [x, y] in the avatar's own pixels"))
            pivot = [0, 0]
        look = p.get("look", [0, 0])
        if not _point(look) or kind != "states" and "look" in p:
            out.append(_issue("error", "avatar-look", f"{where}: look is [x, y], how far eyes move to look aside, "
                                                      "and only eyes (a part with states) take it"))
            look = [0, 0]
        if kind == "states" and "open" not in p[kind]:
            out.append(_issue("error", "avatar-states", f"{where} has no \"open\" state",
                              "name the drawing that shows unless asked otherwise open"))
        parts[name] = Part(name, files, kind, parent, (float(pivot[0]), float(pivot[1])),
                           (float(look[0]), float(look[1])))
    for name in parts:  # a parent chain that comes back to itself
        seen, at = {name}, parts[name].parent
        while at in parts:
            if at in seen:
                out.append(_issue("error", "avatar-parent", f"part '{name}' is its own ancestor through {at}",
                                  "make the parents a tree, with one part (the body) at the root"))
                parts[name].parent = None
                break
            seen.add(at)
            at = parts[at].parent
    mouths = [p for p in parts.values() if p.kind == "shapes"]
    if len(mouths) > 1:
        out.append(_issue("error", "avatar-mouth", "parts " + " and ".join(p.name for p in mouths)
                          + " both have shapes; an avatar has one mouth"))
    if not any(p.kind == "states" and "closed" in (spec["parts"][p.name].get("states") or {}) for p in parts.values()):
        out.append(_issue("info", "avatar-no-blink", "no part has a \"closed\" state, so the avatar never blinks",
                          "give the eyes {\"states\": {\"open\": ..., \"closed\": ...}}"))
    return parts


def _fallbacks(parts: dict[str, Part], spec: dict, out: list) -> dict[str, str]:
    mouth = next((p for p in parts.values() if p.kind == "shapes"), None)
    if mouth is None:
        out.append(_issue("info", "avatar-no-mouth", "no part has mouth shapes, so the avatar does not talk",
                          'give the mouth {"shapes": {"X": ..., "A": ..., "C": ..., "D": ...}}'))
        return {}
    named = set(spec["parts"][mouth.name]["shapes"])
    missing = [s for s in M.REQUIRED if s not in named]
    if missing:
        out.append(_issue("error", "avatar-missing-shape", f"the mouth '{mouth.name}' lacks shape"
                          f"{'s' * (len(missing) > 1)} {', '.join(missing)}",
                          f"draw at least {', '.join(M.REQUIRED)}: rest, closed, open and wide"))
    falls = {}
    for s in M.SHAPES:
        if s not in named:
            to = M.FALLBACK.get(s, s)
            falls[s] = to if to in named else "X"
    return falls


def _pose(where: str, pose, parts: dict, out: list, keyed: bool) -> None:
    """An expression's or a gesture's changes, part by part."""
    for pname, keys in pose.items():
        if pname == "dur" and keyed or pname == "mouth" and not keyed:
            continue
        if pname not in parts:
            out.append(_issue("error", "avatar-pose", f"{where} moves '{pname}', which is not a part",
                              "parts: " + ", ".join(parts)))
            continue
        if not isinstance(keys, dict):
            out.append(_issue("error", "avatar-pose", f"{where}: '{pname}' must be an object of keys such as "
                                                      '{"rotate": 6}'))
            continue
        for k, v in keys.items():
            if k not in POSE_KEYS:
                out.append(_issue("error", "avatar-pose", f"{where} sets '{k}' on '{pname}'",
                                  "a pose sets " + ", ".join(POSE_KEYS)))
            elif keyed:
                dur = pose.get("dur")
                ok = isinstance(v, list) and v and all(isinstance(x, list) and len(x) == 2 and _num(x[0]) and
                                                       _num(x[1]) for x in v)
                if not ok:
                    out.append(_issue("error", "avatar-gesture", f"{where}: '{pname}.{k}' must be keys "
                                      "[[time, value], ...]", "such as [[0, 0], [0.2, 6], [0.5, 0]]"))
                elif _num(dur) and any(not 0 <= x[0] <= dur for x in v):
                    out.append(_issue("error", "avatar-gesture", f"{where}: a key of '{pname}.{k}' is outside 0 to its "
                                                                 f"dur {dur}"))
            elif not _num(v):
                out.append(_issue("error", "avatar-pose", f"{where}: '{pname}.{k}' must be a number"))


def validate(folder: str | Path, spec: dict | None = None) -> list[dict]:
    """Every problem with the avatar in `folder`, as check issues (level, code, message, fix)."""
    return _check(Path(folder), spec)[0]


def _check(folder: Path, spec: dict | None) -> tuple[list[dict], dict, dict, dict]:
    out: list[dict] = []
    if spec is None:
        try:
            spec = read(folder)
        except ValueError as e:
            return [_issue("error", "avatar-file", str(e), f"make {FILE} in the folder")], {}, {}, {}
    ver = spec.get("kinescriptAvatar")
    if not isinstance(ver, int) or isinstance(ver, bool):
        out.append(_issue("error", "avatar-version", f"{FILE} has no \"kinescriptAvatar\" version number",
                          f"add \"kinescriptAvatar\": {VERSION}"))
    elif ver > VERSION:
        out.append(_issue("error", "avatar-version", f"the avatar is version {ver}; this Kinescript reads up to "
                                                     f"{VERSION}", "upgrade Kinescript"))
    for k in sorted(set(spec) - TOP):
        if not str(k).startswith(("_", "//")):
            out.append(_issue("warning", "avatar-key", f"{FILE} has an unknown key '{k}'",
                              "known keys: " + ", ".join(sorted(TOP))))
    size = spec.get("size")
    if not _point(size) or min(size) <= 0:
        out.append(_issue("error", "avatar-size", "size must be [w, h], the avatar's own size in pixels",
                          "the size every part file is drawn at, such as [600, 900]"))
    parts = _parts(folder, spec, out)
    falls = _fallbacks(parts, spec, out) if parts else {}
    for name, v in (spec.get("colors") or {}).items():
        if not isinstance(v, str) or (v not in B.ROLES and not C.is_color(v)):
            out.append(_issue("error", "avatar-color", f"colors.{name} is {v!r}, neither a colour nor a palette role",
                              "palette roles: " + ", ".join(B.ROLES)))
    for kind, keyed in (("expressions", False), ("gestures", True)):
        group = spec.get(kind) or {}
        if not isinstance(group, dict):
            out.append(_issue("error", f"avatar-{kind[:-1]}", f"{kind} must be an object of named poses"))
            continue
        for name, pose in group.items():
            where = f"{kind[:-1]} '{name}'"
            if not isinstance(pose, dict):
                out.append(_issue("error", f"avatar-{kind[:-1]}", f"{where} must be an object"))
                continue
            if keyed and not (_num(pose.get("dur")) and pose["dur"] > 0):
                out.append(_issue("error", "avatar-gesture", f"{where} needs dur, its length in seconds"))
            if not keyed and "mouth" in pose and pose["mouth"] not in M.SHAPES:
                out.append(_issue("error", "avatar-expression",
                                  f"{where}: mouth {pose['mouth']!r} is not a mouth shape",
                                  f"the shapes are {', '.join(M.SHAPES)}"))
            if parts:
                _pose(where, pose, parts, out, keyed)
    idle = spec.get("idle", {})
    if idle is not False and not isinstance(idle, dict):
        out.append(_issue("error", "avatar-idle", "idle must be an object or false"))
    elif isinstance(idle, dict):
        for k in sorted(set(idle) - IDLE_KEYS):
            out.append(_issue("warning", "avatar-key", f"idle has an unknown key '{k}'",
                              "known: blink, breath, saccade"))
        for k in ("blink", "saccade"):
            v = idle.get(k)
            if v is not None and v is not False and not (_point(v) and 0 < v[0] <= v[1]):
                out.append(_issue("error", "avatar-idle", f"idle.{k} must be [shortest, longest] seconds between "
                                                          f"{'blinks' if k == 'blink' else 'glances'}, or false"))
        b = idle.get("breath")
        if b is not None and b is not False and not (_num(b) and 0 <= b <= 0.1):
            out.append(_issue("error", "avatar-idle", "idle.breath is how much the body swells, 0 to 0.1 (0.012 is "
                                                      "calm), or false"))
    voice = spec.get("voice", {})
    if not isinstance(voice, dict) or set(voice) - set(B.VOICE_KEYS):
        out.append(_issue("error", "avatar-voice", "voice takes " + ", ".join(B.VOICE_KEYS),
                          'such as {"backend": "kokoro", "speaker": "am_michael"}'))
    if spec.get("mouthMode", "swap") not in MODES:
        out.append(_issue("error", "avatar-mouth-mode", f"mouthMode is {spec['mouthMode']!r}",
                          "swap (each shape its own drawing) or morph (shapes whose paths can blend)"))
    return out, spec, parts, falls


def load(folder: str | Path) -> Avatar:
    """The avatar in `folder`. AvatarError (a ValueError) lists every error when it has any."""
    folder = Path(folder)
    issues, spec, parts, falls = _check(folder, None)
    if any(i["level"] == "error" for i in issues):
        raise AvatarError(folder, issues)
    return Avatar(folder, spec, (float(spec["size"][0]), float(spec["size"][1])), parts,
                  spec.get("mouthMode", "swap"), falls)


def summary(folder: str | Path) -> dict:
    """What kine avatar check prints: the issues, and for an avatar without errors what it has."""
    issues = validate(folder)
    out = {"ok": not any(i["level"] == "error" for i in issues), "issues": issues}
    if out["ok"]:
        av = load(folder)
        out.update(name=av.spec.get("name"), size=list(av.size), parts=list(av.parts),
                   mouth=av.mouth.name if av.mouth else None,
                   shapes=sorted(av.mouth.files) if av.mouth else [], fallbacks=av.fallbacks,
                   eyes=[p.name for p in av.eyes], expressions=list(av.spec.get("expressions") or {}),
                   gestures=list(av.spec.get("gestures") or {}))
    return out


_ID = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""")


def svg_ids(path: Path) -> set[str]:
    """The ids in an svg file, the pieces colors can fill."""
    try:
        return set(_ID.findall(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return set()


def colors(av: Avatar, theme: dict | None = None) -> dict[str, str]:
    """The avatar's colours with palette roles read from `theme` (a scene's theme values; the built-in theme
    without one)."""
    from .components import THEME
    theme = {**THEME, **(theme or {})}
    alias = {}
    for name, role in B.ALIASES.items():
        alias.setdefault(role, name)
    out = {}
    for name, v in (av.spec.get("colors") or {}).items():
        while v in B.ROLES and v not in theme and v not in alias:
            v = B.FALLBACK.get(v, "accent")
        out[name] = theme.get(v) or theme.get(alias.get(v, ""), v)
    return out


def _drawing(av: Avatar, part: Part, key: str, lid: str, fills: dict, shown: bool, shift=(0.0, 0.0)) -> dict:
    """One drawing of a part, placed so the avatar's own (0, 0) is at its part group's pivot."""
    path = part.files[key]
    w, h = av.size
    L = {"id": f"{lid}-{part.name}" + (f"-{key}" if key else "-art"), "src": path.as_posix(),
         "x": w / 2 - part.pivot[0] + shift[0], "y": h / 2 - part.pivot[1] + shift[1], "w": w, "h": h}
    if not shown:
        L["opacity"] = 0
    if path.suffix.lower() == ".svg":
        ids = svg_ids(path)
        paint = {n: {"fill": c} for n, c in fills.items() if n in ids}
        return {**L, "type": "svg", **({"parts": paint} if paint else {})}
    return {**L, "type": "image"}


def layers(av: Avatar, lid: str, pose: dict | None = None, theme: dict | None = None) -> dict:
    """The avatar in one still pose as a group of plain layers, its (0, 0) at the avatar's top left.
    pose: {"parts": {name: {x, y, rotate, scale, opacity}}, "mouth": shape, "eyes": state, "look": [gx, gy]}, with
    look from -1 to 1 of each eye part's look reach. Each part is a group at its pivot, so it turns about it."""
    pose = pose or {}
    fills = colors(av, theme)
    shape = av.shape(pose.get("mouth", "X"))
    look = pose.get("look") or [0, 0]
    kids: dict[str | None, list[str]] = {}
    for name, p in av.parts.items():
        kids.setdefault(p.parent, []).append(name)

    def build(name: str) -> dict:
        p = av.parts[name]
        par = av.parts[p.parent].pivot if p.parent else (0.0, 0.0)
        change = (pose.get("parts") or {}).get(name, {})
        G = {"id": f"{lid}-{name}", "type": "group", "x": p.pivot[0] - par[0] + change.get("x", 0),
             "y": p.pivot[1] - par[1] + change.get("y", 0)}
        for k in ("rotate", "scale", "opacity"):
            if k in change:
                G[k] = change[k]
        if p.kind == "src":
            art = [_drawing(av, p, "", lid, fills, True)]
        elif p.kind == "shapes":
            art = [_drawing(av, p, k, lid, fills, k == shape) for k in p.files]
        else:
            state = pose.get("eyes", "open") if p in av.eyes else "open"
            state = state if state in p.files else "open"
            shift = (look[0] * p.look[0], look[1] * p.look[1])
            art = [_drawing(av, p, k, lid, fills, k == state, shift) for k in p.files]
        own = list(av.parts).index(name)  # children listed before the part draw behind it
        behind = [build(c) for c in kids.get(name, []) if list(av.parts).index(c) < own]
        front = [build(c) for c in kids.get(name, []) if list(av.parts).index(c) > own]
        G["children"] = behind + art + front
        return G

    return {"id": lid, "type": "group", "children": [build(n) for n in kids.get(None, [])]}


def peak(gesture: dict) -> float:
    """The time in a gesture where its keys are furthest from where they start: the frame that shows it."""
    best, at = -1.0, 0.0
    for name, keys in gesture.items():
        if name == "dur" or not isinstance(keys, dict):
            continue
        for ks in keys.values():
            for t, v in ks:
                d = abs(v - ks[0][1])
                if d > best:
                    best, at = d, t
    return at


def sample(keys: list, t: float) -> float:
    """Keys [[time, value], ...] at time t, straight between keys and held past the ends."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0) if t1 > t0 else 1)
    return keys[-1][1]


def poses(av: Avatar) -> list[tuple[str, dict]]:
    """Every pose worth looking at once: neutral, each mouth shape, eyes closed and looking, each expression and
    each gesture at its peak."""
    out = [("neutral", {})]
    if av.mouth:
        out += [(f"mouth {s}" + (f" (as {av.shape(s)})" if av.shape(s) != s else ""), {"mouth": s})
                for s in M.SHAPES]
    if av.eyes:
        out.append(("eyes closed", {"eyes": "closed"}))
    if any(p.look != (0.0, 0.0) for p in av.eyes):
        out += [("look left", {"look": [-1, 0]}), ("look right", {"look": [1, 0]})]
    for name, e in (av.spec.get("expressions") or {}).items():
        parts = {k: v for k, v in e.items() if k != "mouth"}
        out.append((f"expression {name}", {"parts": parts, **({"mouth": e["mouth"]} if "mouth" in e else {})}))
    for name, g in (av.spec.get("gestures") or {}).items():
        t = peak(g)
        parts = {p: {k: sample(ks, t) for k, ks in keys.items()} for p, keys in g.items() if p != "dur"}
        out.append((f"gesture {name} at {t:g}s", {"parts": parts}))
    return out


def sheet(folder: str | Path, out: str, width: int = 240, cols: int | None = None) -> dict:
    """A contact sheet of the avatar's poses (see poses), each labelled, with pivots marked on the neutral one."""
    from .output import still
    scene, names = sheet_scene(load(folder), width, cols)
    got = still(scene, 0, out)
    return {"out": got["out"], "poses": names, "w": got["w"], "h": got["h"]}


def sheet_scene(av: Avatar, width: int = 240, cols: int | None = None) -> tuple[dict, list[str]]:
    """The scene sheet draws, and the names of its poses in order."""
    cells = poses(av)
    cols = cols or min(6, len(cells))
    w, h = av.size
    k = width / w
    cw, ch, label = width, h * k, 34
    rows = math.ceil(len(cells) / cols)
    layers_out = []
    for i, (name, pose) in enumerate(cells):
        x0, y0 = (i % cols) * cw, (i // cols) * (ch + label)
        g = layers(av, f"c{i}", pose)
        g.update(x=x0, y=y0, scale=k)
        layers_out += [{"id": f"c{i}-cell", "type": "rect", "anchor": "topleft", "x": x0 + 2, "y": y0 + 2,
                        "w": cw - 4, "h": ch - 4, "radius": 6, "fill": "#1b1f27"}, g,
                       {"id": f"c{i}-label", "type": "text", "text": name, "x": x0 + cw / 2, "y": y0 + ch + label / 2,
                        "size": 18, "fill": "#e8e2d0", "weight": 500}]
    for name, p in av.parts.items():  # the pivots the avatar sets, in its own pixels, on the neutral pose
        if "pivot" not in av.spec["parts"][name]:
            continue
        layers_out += [{"id": f"pivot-{name}", "type": "circle", "x": p.pivot[0] * k, "y": p.pivot[1] * k, "r": 4,
                        "fill": "#ff633f", "stroke": "#0a0c10", "strokeWidth": 1.5},
                       {"id": f"pivot-{name}-name", "type": "text", "text": name, "x": p.pivot[0] * k + 7,
                        "y": p.pivot[1] * k, "anchor": "left", "size": 13, "fill": "#f2f0e8", "weight": 600,
                        "box": {"fill": "#0a0c10c0", "pad": [2, 5], "radius": 4}}]
    scene = {"canvas": {"w": round(cols * cw), "h": round(rows * (ch + label)), "fps": 30, "duration": 0.1,
                        "bg": "#0a0c10"}, "layers": layers_out}
    return scene, [n for n, _ in cells]


PRE = 3600.0  # seconds a first line's mouth drawings start before it: before any avatar that says it
SAY_GAP = 0.4  # seconds between one line of a say list and the next, when the next gives no at
SAY_KEYS = {"id", "text", "at", "speaker", "backend", "rate", "options", "src", "gain", "volume", "pan", "words"}


def _avatar_layers(layers):
    for L in layers or []:
        if isinstance(L, dict):
            if L.get("use") == "avatar":
                yield L
            yield from _avatar_layers(L.get("children"))


def say_sounds(spec: dict, base) -> dict:
    """The scene with each avatar's say lines written as voice sounds in "audio", so kine voice speaks them and
    cues can name them, and each avatar's say as the list of their ids. A line takes its voice from its own keys,
    then the avatar's voice, then the brand's and the scene's, as any voice does. Starts with the avatar unless it
    gives an at; each later line starts SAY_GAP after the one before it ends."""
    avatars = list(_avatar_layers(spec.get("layers")))
    if not any("say" in L for L in avatars):
        return spec
    import copy
    spec = copy.deepcopy(spec)
    audio = spec.setdefault("audio", [])
    taken = {str(a.get("id")) for a in audio if isinstance(a, dict)}
    for n, L in enumerate(_avatar_layers(spec.get("layers"))):
        if "say" not in L:
            continue
        lid = str(L.get("id") or f"avatar{n + 1}")
        L.setdefault("id", lid)
        try:
            voice = read(Path(base) / str(L.get("avatar", ""))).get("voice") or {}
        except ValueError:
            voice = {}  # reported when the layer expands
        lines = L["say"] if isinstance(L["say"], list) else [L["say"]]
        ids, prev = [], None
        for k, line in enumerate(lines):
            if isinstance(line, str):
                ids.append(line)
                prev = line
                continue
            if not isinstance(line, dict) or not isinstance(line.get("text"), str):
                raise ValueError(f"layer '{lid}': each say is a sound id or {{\"text\": \"...\"}}; got {line!r}")
            extra = set(line) - SAY_KEYS
            if extra:
                raise ValueError(f"layer '{lid}': say takes {', '.join(sorted(SAY_KEYS))}; not {', '.join(sorted(extra))}")
            sid = str(line.get("id") or (f"{lid}-say" if len(lines) == 1 else f"{lid}-say{k + 1}"))
            if sid in taken:
                raise ValueError(f"layer '{lid}': say id '{sid}' is already a sound id; give the line its own id")
            taken.add(sid)
            at = line.get("at", f"{prev}.out+{SAY_GAP}" if prev else f"{lid}.in")
            entry = {"id": sid, "role": "voice", **{k2: v for k2, v in voice.items() if k2 not in line},
                     **{k2: v for k2, v in line.items() if k2 != "id"}, "at": at}
            audio.append(entry)
            ids.append(sid)
            prev = sid
        L["say"] = ids
    return spec


def _track(scene, sid: str) -> tuple[list[list], float]:
    """Sound sid's mouth track and where in its file it starts playing (its trim)."""
    from . import audio as AU
    from . import voice as V
    from . import words as W
    a = next((x for i, x in enumerate(scene.spec.get("audio") or [])
              if str(x.get("id") or f"audio{i + 1}") == sid), None)
    if a is None:
        raise ValueError(f"say names '{sid}', which is not an audio id")
    trim = a.get("trim") if isinstance(a.get("trim"), list) and a.get("trim") else [0]
    if isinstance(a.get("words"), list):
        return M.track(W.read(a["words"]))["mouth"], float(trim[0] or 0)
    src, words_file = V.paths(sid, a, scene.base) if a.get("text") else (None, None)
    if words_file is None:
        src = Path(str(a.get("src", "")))
        src = src if src.is_absolute() else scene.base / src
        w = a.get("words")
        words_file = scene.base / w if isinstance(w, str) else W.default_path(str(src))
    if words_file.exists():
        return M.track(*M.stored(words_file))["mouth"], float(trim[0] or 0)
    if src is not None and Path(src).exists():
        return M.levels(AU.analyze(str(src))["rms"], AU._WIN)["mouth"], float(trim[0] or 0)
    return [], 0.0  # a voice not spoken yet: scene load reports voice-missing


def _keys(track: list[list], av: Avatar, shape: str, t0: float, pre: float = 0.0) -> list | int:
    """Opacity keys that show the drawn `shape` while the track says it (from t0 in the file, `pre` seconds into
    the layer), at rest before and after, held between keys."""
    rest = 1 if av.shape("X") == shape else 0
    keys, last = [[0, rest, "hold"]], rest
    for s, a, _ in track:
        v = 1 if av.shape(s) == shape else 0
        if v != last:
            keys.append([round(pre + max(0.0, a - t0), 4), v, "hold"])
            last = v
    if track and last != rest:
        keys.append([round(pre + max(0.0, track[-1][2] - t0), 4), rest, "hold"])
    return keys if len(keys) > 1 else rest


def _find(G: dict, gid: str) -> dict | None:
    if G.get("id") == gid:
        return G
    for c in G.get("children") or []:
        if isinstance(c, dict) and (got := _find(c, gid)):
            return got
    return None


RAMP = 0.2  # seconds an expression takes to come on and to go
MOVE = 0.12  # seconds the eyes take to move to a new gaze
GAZE = {"camera": (0, 0), "ahead": (0, 0), "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
ENTRY_KEYS = {"expressions": {"name", "at", "dur", "weight"}, "gaze": {"at", "target"},
              "gestures": {"name", "at"}}


def _early(at, lid: str, fps: float) -> str:
    """A cue PRE seconds before `at`, for a layer whose keys are shifted by PRE so that, starting that early, it
    shows its first key until `at`. A number (or clock) counts from the avatar's start, as at does in a group;
    written as a cue on the avatar, so it means the same however deep the layer is."""
    from .timing import parse_clock
    t = at if isinstance(at, (int, float)) and not isinstance(at, bool) else (
        parse_clock(at, fps) if isinstance(at, str) else None)
    if t is not None:
        return f"{lid}.in{'+' if t >= 0 else '-'}{abs(t):g} - {PRE:g}"
    if not isinstance(at, str):
        raise ValueError(f"cannot read the time {at!r}")
    return f"{at} - {PRE:g}"  # the spaces end an id, which may hold a '-'


def _entries(p, kind: str, names: dict | None) -> list[dict]:
    lid, got = p["id"], p.get(kind) or []
    got = got if isinstance(got, list) else [got]
    for i, e in enumerate(got):
        where = f"layer '{lid}': {kind}[{i}]"
        if not isinstance(e, dict):
            raise ValueError(f"{where} must be an object such as " + {
                "expressions": '{"name": "smile", "at": "s1.in", "dur": 1}', "gaze": '{"at": 0, "target": "camera"}',
                "gestures": '{"name": "nod", "at": "s1.in+0.5"}'}[kind])
        extra = set(e) - ENTRY_KEYS[kind]
        if extra:
            raise ValueError(f"{where} takes {', '.join(sorted(ENTRY_KEYS[kind]))}; not {', '.join(sorted(extra))}")
        if names is not None and e.get("name") not in names:
            raise ValueError(f"{where} names {kind[:-1]} {e.get('name')!r}, which the avatar does not have. Its "
                             f"{kind}: {', '.join(names) or 'none'}")
        for k in ("dur", "weight"):
            v = e.get(k)
            if v is not None and (not _num(v) or v < 0 or k == "dur" and v == 0):
                raise ValueError(f"{where}: {k} is {v!r}; it takes a number above 0")
    return got


def _wrap(G: dict, gid: str, at: str, props: dict, only: set[str] | None = None) -> None:
    """Put G's children (or only those with ids in `only`) in a group that starts at `at` and carries props, so its
    keys add to the part's own: a group at (0, 0) moves, turns and scales what it holds about the part's pivot."""
    kids = G.get("children") or []
    inner = [c for c in kids if only is None or c.get("id") in only]
    if not inner:
        return
    W = {"id": gid, "type": "group", "at": at, **props, "children": inner}
    first = kids.index(inner[0])
    G["children"] = [c for c in kids[:first] if c not in inner] + [W] + [c for c in kids[first:] if c not in inner]


def _gaze(target, p, av: Avatar, k: float, scene, where: str) -> tuple[float, float]:
    """A gaze target as [gx, gy], each -1 to 1 of the eyes' reach."""
    if isinstance(target, str) and target in GAZE:
        return GAZE[target]
    if _point(target):
        return max(-1.0, min(1.0, float(target[0]))), max(-1.0, min(1.0, float(target[1])))
    if not (isinstance(target, str) and target.startswith("#")):
        raise ValueError(f"{where}: target is {target!r}; give {', '.join(GAZE)}, [gx, gy] from -1 to 1, or "
                         "\"#id\" of a layer")
    from . import props as P
    top = {str(L.get("id")): L for L in scene.spec.get("layers") or [] if isinstance(L, dict)}
    tid = target[1:]

    def at(L, what):
        xy = [P.length(L.get(a, d), a, scene.ctx) for a, d in (("x", "50%" if what == "avatar" else 0),
                                                               ("y", "50%" if what == "avatar" else 0))]
        if not all(_num(v) for v in xy):
            raise ValueError(f"{where}: the {what} needs a fixed x and y to look at '{tid}'; give the target as "
                             "[gx, gy] instead")
        return [float(v) for v in xy]
    if tid not in top:
        raise ValueError(f"{where}: no layer at the top of the scene has id '{tid}'" +
                         ("; it is inside a group, so give the target as [gx, gy]" if tid in _ids(scene.spec) else
                          ". Layers: " + ", ".join(top)))
    if p["id"] not in top:
        raise ValueError(f"{where}: the avatar is inside a group, so it cannot look at '{tid}'; give [gx, gy]")
    tx, ty = at(top[tid], "target")
    ax, ay = at(p, "avatar")
    q, eye = (av.eyes or [None])[0], (av.size[0] / 2, av.size[1] / 2)
    while q is not None:  # where the eyes are: the nearest pivot the avatar sets, from the eyes up
        if "pivot" in av.spec["parts"][q.name]:
            eye = q.pivot
            break
        q = av.parts.get(q.parent) if q.parent else None
    dx, dy = tx - (ax + (eye[0] - av.size[0] / 2) * k), ty - (ay + (eye[1] - av.size[1] / 2) * k)
    d = math.hypot(dx, dy)
    return (0.0, 0.0) if d < 1 else (dx / d, dy / d)


def _ids(spec) -> set[str]:
    out: set[str] = set()

    def walk(ls):
        for L in ls or []:
            if isinstance(L, dict):
                out.add(str(L.get("id")))
                walk(L.get("children"))
    walk(spec.get("layers"))
    return out


def _mouth(av: Avatar, mouth: dict, says: list, exprs: list, lid: str, fills: dict, scene) -> None:
    """The mouth part's drawings: each line's track shows its shapes, and an expression with a mouth shows that
    shape wherever the mouth would be at rest while it lasts (speech owns the mouth; the latest such expression wins).
    Every keyed layer starts early and holds its first key (see PRE), as cues cannot be read inside keys."""
    part, rig, rest = av.mouth, f"{lid}-rig", av.shape("X")
    biased = [(i, e, av.shape(av.spec["expressions"][e["name"]]["mouth"])) for i, e in enumerate(exprs)
              if "mouth" in av.spec["expressions"][e["name"]] and (e.get("weight", 1) or 0) > 0]
    biased = [b for b in biased if b[2] != rest]
    if not says and not biased:
        return
    kids = []
    for n, sid in enumerate(says or [None]):
        track, t0 = _track(scene, sid) if sid else ([], 0.0)
        pre = PRE if n == 0 else 0.0  # the first line's drawings start long before it, at rest until it starts

        def draw(s: str, keyed: str, tag: str) -> dict:
            L = _drawing(av, part, s, rig, fills, True)
            L.pop("opacity", None)
            L.update(id=f"{lid}-mouth-{tag}-{n + 1}", opacity=_keys(track, av, keyed, t0, pre))
            if sid:
                L["at"] = f"{sid}.in - {PRE:g}" if pre else f"{sid}.in"
                if n + 1 < len(says):
                    L["until"] = f"{says[n + 1]}.in"
            return L
        for s in part.files:
            L = draw(s, s, s)
            if s != rest or not biased:
                if sid or L["opacity"] != 0:
                    kids.append(L)
                continue
            slot = [L]
            for i, e, shape in biased:  # rest shows where no biasing expression is on, each one's shape where it is
                at = _early(e.get("at", 0), lid, scene.fps)
                on = [[0, 0, "hold"], [PRE, 1, "hold"]] + ([[PRE + e["dur"], 0, "hold"]] if e.get("dur") else [])
                off = [[t, 1 - v, ease] for t, v, ease in on]
                slot = [{"id": f"{lid}-mouth-rest{i + 1}-{n + 1}", "type": "group", "at": at, "opacity": off,
                         "children": slot},
                        {"id": f"{lid}-mouth-expr{i + 1}-{n + 1}", "type": "group", "at": at, "opacity": on,
                         "children": [draw(shape, rest, f"{shape}-expr{i + 1}")]}]
            kids += slot
    mouth["children"] = kids


def component(p, theme: dict, scene) -> dict:
    """{"use": "avatar", "avatar": FOLDER, "x", "y", "h" or "w", "say", "expressions", "gaze", "gestures"}: the
    avatar as plain layers, centred on x, y at h pixels tall (its own size without h or w), its mouth following the
    sounds say names, with the expressions, gaze and gestures given at their times."""
    lid = p["id"]
    folder = Path(scene.base) / str(p.get("avatar") or "")
    if not p.get("avatar"):
        raise ValueError(f"layer '{lid}': an avatar layer needs \"avatar\": the folder holding avatar.json")
    av = load(folder)
    w, h = av.size
    k = (float(p["h"]) / h if p.get("h") is not None else float(p["w"]) / w if p.get("w") is not None else 1.0)
    rig, fills = f"{lid}-rig", colors(av, theme)
    body = layers(av, rig, {}, theme)
    exprs = _entries(p, "expressions", av.spec.get("expressions") or {})
    gazes = _entries(p, "gaze", None)
    gests = _entries(p, "gestures", av.spec.get("gestures") or {})
    says = p.get("say") or []
    says = says if isinstance(says, list) else [says]
    mouth = av.mouth and _find(body, f"{rig}-{av.mouth.name}")
    if mouth:
        _mouth(av, mouth, says, exprs, lid, fills, scene)

    eyes = [q for q in av.parts.values() if q.kind == "states" and q.look != (0.0, 0.0)]
    if gazes and not eyes:
        raise ValueError(f"layer '{lid}': gaze moves the eyes by their look reach, and no part of the avatar has "
                         "look; give the eyes part \"look\": [x, y] in its avatar.json")
    was = (0.0, 0.0)
    for i, g in enumerate(gazes):  # each gaze moves the eyes on from the one before, so the moves add up
        gx, gy = _gaze(g.get("target", "camera"), p, av, k, scene, f"layer '{lid}': gaze[{i}]")
        at = _early(g.get("at", 0), lid, scene.fps)
        for e in eyes:
            G = _find(body, f"{rig}-{e.name}")
            dx, dy = (gx - was[0]) * e.look[0], (gy - was[1]) * e.look[1]
            if G is None or (dx, dy) == (0, 0):
                continue
            props = {a: [[0, 0], [PRE, 0], [PRE + MOVE, round(d, 4)]] for a, d in (("x", dx), ("y", dy)) if d}
            _wrap(G, f"{lid}-gaze{i + 1}-{e.name}", at, props, {f"{rig}-{e.name}-{s}" for s in e.files} |
                  {f"{lid}-gaze{j}-{e.name}" for j in range(1, i + 1)})
        was = (gx, gy)

    for i, e in enumerate(exprs):
        pose, wt, dur = av.spec["expressions"][e["name"]], float(e.get("weight", 1)), e.get("dur")
        r = min(RAMP, dur / 2) if dur else RAMP
        at = _early(e.get("at", 0), lid, scene.fps)
        for pname, change in pose.items():
            G = _find(body, f"{rig}-{pname}") if pname != "mouth" else None
            if G is None:
                continue
            props = {}
            for a, v in change.items():
                base = 1.0 if a in ("scale", "opacity") else 0.0
                to = round(base + wt * (v - base), 4)
                keys = [[0, base], [PRE, base], [PRE + r, to]]
                if dur:
                    keys += ([[PRE + dur - r, to]] if dur > 2 * r else []) + [[PRE + dur, base]]
                props[a] = keys
            _wrap(G, f"{lid}-expr{i + 1}-{pname}", at, props)

    spans = getattr(scene, "gestures", None)
    for i, g in enumerate(gests):
        pose = av.spec["gestures"][g["name"]]
        at = _early(g.get("at", 0), lid, scene.fps)
        for pname, change in pose.items():
            G = _find(body, f"{rig}-{pname}") if pname != "dur" else None
            if G is None:
                continue
            gid = f"{lid}-gesture{i + 1}-{pname}"
            _wrap(G, gid, at, {a: [[0, ks[0][1]]] + [[PRE + t, v] for t, v in ks] for a, ks in change.items()})
            if spans is not None:
                spans.append({"layer": lid, "name": g["name"], "part": pname, "id": gid, "dur": float(pose["dur"])})

    body.update(x=-w * k / 2, y=-h * k / 2, scale=k)
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "children": [body]}
