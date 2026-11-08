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

from . import avatar3d as A3
from . import brand as B
from . import color as C
from . import mouth as M

VERSION = 1
FILE = "avatar.json"
TOP = {"kinescriptAvatar", "name", "description", "size", "parts", "colors", "expressions", "gestures", "idle",
       "voice", "mouthMode", "model", "mouth", "blink", "look"}  # model, mouth and blink: a 3D avatar (avatar3d)
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
    model: Path | None = None  # a 3D avatar's glTF file (avatar3d); it has no parts

    @property
    def mouth(self) -> Part | None:
        return next((p for p in self.parts.values() if p.kind == "shapes"), None)

    @property
    def eyes(self) -> list[Part]:
        return [p for p in self.parts.values() if p.kind == "states" and "closed" in p.files]

    def shape(self, s: str) -> str:
        """The drawn mouth shape that shows for shape s."""
        if self.model is not None:
            return s if s in (self.spec.get("mouth") or {}) else self.fallbacks.get(s, "X")
        return s if self.mouth and s in self.mouth.files else self.fallbacks.get(s, "X")


def _issue(level: str, code: str, message: str, fix: str = "") -> dict:
    return {"level": level, "code": code, "layer": "avatar", "message": message, "fix": fix}


BUILT_IN = Path(__file__).parent / "avatars"  # kine:NAME avatars, drawn by brand/make_host.py


def starters() -> list[str]:
    """The names of the built-in avatars, as kine:NAME names them."""
    return sorted(d.name for d in BUILT_IN.iterdir() if (d / FILE).is_file()) if BUILT_IN.is_dir() else []


def locate(name, base: str | Path = ".", brand=None) -> tuple[Path, dict]:
    """The folder an avatar name means and its brand entry: brand:NAME the scene brand's avatar NAME (with the
    brand's voice and idle for it), kine:NAME a built-in one, anything else a folder from base."""
    s = str(name or "")
    if s.startswith("brand:"):
        if not isinstance(brand, str) or not brand:
            raise ValueError(f"'{s}' is a brand's avatar, and the scene names no brand; add \"brand\"")
        return B.avatar(Path(base) / brand, s[6:])
    return where(s, base), {}


def where(name, base: str | Path = ".") -> Path:
    """The folder an avatar name means: kine:NAME a built-in one, anything else a folder from base."""
    s = str(name or "")
    if s.startswith("kine:"):
        if s[5:] not in starters():
            raise ValueError(f"there is no built-in avatar '{s}'; built in: " + ", ".join(f"kine:{n}" for n in starters()))
        return BUILT_IN / s[5:]
    return Path(base) / s


def new(folder: str | Path, src: str = "kine:host") -> dict:
    """Copy an avatar (the starter unless src names another) into `folder` as files to edit. Refuses a folder that
    already holds an avatar."""
    import shutil
    dest, origin = Path(folder), where(src)
    if (dest / FILE).exists():
        raise ValueError(f"{dest} already holds an avatar; pick a new folder")
    load(origin)
    files = [f for f in origin.rglob("*") if f.is_file() and f.suffix.lower() in (".json", ".glb", ".gltf", ".bin",
                                                                                   *EXTS)]
    for f in files:
        to = dest / f.relative_to(origin)
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, to)
    return {"ok": True, "out": str(dest), "from": src, "files": len(files),
            "next": f"edit {dest / 'parts'}, then kine avatar check {dest} and kine look {dest}"}


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


def _file(folder: Path, rel, where: str, out: list, exts: tuple = EXTS) -> Path | None:
    try:
        p = B.inside(folder, rel)
    except ValueError as e:
        out.append(_issue("error", "avatar-file", f"{where}: {str(e).replace('brand folder', 'avatar folder')}"))
        return None
    if p.suffix.lower() not in exts:
        out.append(_issue("error", "avatar-file", f"{where}: '{rel}' is not {', '.join(exts)}",
                          "draw parts as .svg (or .png and .webp for pictures)" if exts == EXTS else
                          "export the model as .glb"))
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
            if k == "state":
                part = parts[pname]
                if keyed:
                    out.append(_issue("error", "avatar-gesture", f"{where} picks a state of '{pname}'",
                                      "an expression can pick a state; a gesture only moves parts"))
                elif part.kind != "states":
                    out.append(_issue("error", "avatar-pose", f"{where} picks a state of '{pname}', which has none",
                                      'give the part "states": {"open": ..., "' + str(v) + '": ...}'))
                elif v not in part.files:
                    out.append(_issue("error", "avatar-pose", f"{where}: '{pname}' has no state {v!r}",
                                      "its states: " + ", ".join(part.files)))
                continue
            if k not in POSE_KEYS:
                out.append(_issue("error", "avatar-pose", f"{where} sets '{k}' on '{pname}'",
                                  "a pose sets " + ", ".join(POSE_KEYS)
                                  + ("" if keyed else ", and state (one of a part's drawings)")))
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
    three = A3.is_model(spec)
    size = spec.get("size", [600, 600] if three else None)
    if not _point(size) or min(size) <= 0:
        out.append(_issue("error", "avatar-size", "size must be [w, h], the avatar's own size in pixels",
                          "the size every part file is drawn at, such as [600, 900]" if not three else
                          "the box the model is fitted in, such as [600, 600] (the default)"))
    if three:
        more, _, falls = A3.check(folder, spec, _file)
        out += more
        parts: dict = {}
    else:
        parts = _parts(folder, spec, out)
        falls = _fallbacks(parts, spec, out) if parts else {}
    for name, v in (spec.get("colors") or {}).items():
        if not isinstance(v, str) or (v not in B.ROLES and not C.is_color(v)):
            out.append(_issue("error", "avatar-color", f"colors.{name} is {v!r}, neither a colour nor a palette role",
                              "palette roles: " + ", ".join(B.ROLES)))
    for kind, keyed in () if three else (("expressions", False), ("gestures", True)):
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
    size = spec.get("size", [600, 600])
    model = B.inside(folder, spec["model"]["src"]) if A3.is_model(spec) else None
    return Avatar(folder, spec, (float(size[0]), float(size[1])), parts, spec.get("mouthMode", "swap"), falls, model)


def summary(folder: str | Path) -> dict:
    """What kine avatar check prints: the issues, and for an avatar without errors what it has."""
    issues = validate(folder)
    out = {"ok": not any(i["level"] == "error" for i in issues), "issues": issues}
    if out["ok"] and A3.is_model(read(folder)):
        av = load(folder)
        out.update(name=av.spec.get("name"), size=list(av.size), model=av.spec["model"]["src"], morphs=A3.used(av),
                   shapes=sorted(av.spec.get("mouth") or {}), fallbacks=av.fallbacks,
                   blink=bool(av.spec.get("blink")), expressions=list(av.spec.get("expressions") or {}),
                   gestures=list(av.spec.get("gestures") or {}))
    elif out["ok"]:
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


def placed(av: Avatar, lid: str, pose: dict, theme: dict | None, x: float, y: float, k: float) -> dict:
    """The avatar in one still pose with its top left at x, y, k times its own size. A 3D avatar's model layer is
    sized, not scaled: a group's scale does not reach the model it holds."""
    g = A3.still(av, lid, pose, k) if av.model is not None else layers(av, lid, pose, theme)
    g.update(x=x, y=y, **({} if av.model is not None else {"scale": k}))
    return g


def layers(av: Avatar, lid: str, pose: dict | None = None, theme: dict | None = None) -> dict:
    """The avatar in one still pose as a group of plain layers, its (0, 0) at the avatar's top left.
    pose: {"parts": {name: {x, y, rotate, scale, opacity}}, "mouth": shape, "eyes": state, "look": [gx, gy]}, with
    look from -1 to 1 of each eye part's look reach. Each part is a group at its pivot, so it turns about it."""
    pose = pose or {}
    if av.model is not None:
        return A3.still(av, lid, pose)
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
            state = change.get("state") or (pose.get("eyes", "open") if p in av.eyes else "open")
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
    if av.model is not None:
        return A3.poses(av)
    out = [("neutral", {})]
    if av.mouth:
        out += [(f"mouth {s}" + (f" (as {av.shape(s)})" if av.shape(s) != s else ""), {"mouth": s})
                for s in M.SHAPES]
    if av.eyes:
        out.append(("eyes closed", {"eyes": "closed"}))
    if any(p.look != (0.0, 0.0) for p in av.eyes):
        out += [("look left", {"look": [-1, 0]}), ("look right", {"look": [1, 0]})]
    for p in _looking(av):
        out += [(f"look {d}", {"parts": {p.name: {"state": d}}}) for d in LOOKS if d in p.files]
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
        g = placed(av, f"c{i}", pose, None, x0, y0, k)
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
            folder, mark = locate(L.get("avatar", ""), base, spec.get("brand"))
            voice = {**(read(folder).get("voice") or {}), **(mark.get("voice") or {})}  # the brand's choice wins
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


def _track(scene, sid: str) -> tuple[dict, float, Path | None]:
    """Sound sid's mouth track (as mouth.track gives it), where in its file it starts playing (its trim), and the
    words file it came from, if one."""
    from . import audio as AU
    from . import voice as V
    from . import words as W
    a = next((x for i, x in enumerate(scene.spec.get("audio") or [])
              if str(x.get("id") or f"audio{i + 1}") == sid), None)
    if a is None:
        raise ValueError(f"say names '{sid}', which is not an audio id")
    trim = a.get("trim") if isinstance(a.get("trim"), list) and a.get("trim") else [0]
    if isinstance(a.get("words"), list):
        return M.track(W.read(a["words"])), float(trim[0] or 0), None
    src, words_file = V.paths(sid, a, scene.base) if a.get("text") else (None, None)
    if words_file is None:
        src = Path(str(a.get("src", "")))
        src = src if src.is_absolute() else scene.base / src
        w = a.get("words")
        words_file = scene.base / w if isinstance(w, str) else W.default_path(str(src))
    if words_file.exists():
        return M.track(*M.stored(words_file)), float(trim[0] or 0), words_file
    if src is not None and Path(src).exists():
        return M.levels(AU.analyze(str(src))["rms"], AU._WIN), float(trim[0] or 0), None
    return {"mouth": [], "source": "none", "unknown": []}, 0.0, None  # not spoken yet: load reports voice-missing


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


def _wrap(G: dict, gid: str, at: str | None, props: dict, only: set[str] | None = None) -> None:
    """Put G's children (or only those with ids in `only`) in a group that starts at `at` and carries props, so its
    keys add to the part's own: a group at (0, 0) moves, turns and scales what it holds about the part's pivot."""
    kids = G.get("children") or []
    inner = [c for c in kids if only is None or c.get("id") in only]
    if not inner:
        return
    W = {"id": gid, "type": "group", **({"at": at} if at is not None else {}), **props, "children": inner}
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


def _notes(scene, lid: str, sid: str, got: dict, words_file: Path | None, av: Avatar) -> None:
    """What check says about the mouth a sound gives an avatar: a mouth that only follows loudness, words the
    spelling rules cannot read, shapes the avatar has no drawing for, and a stored track older than its words."""
    if got["source"] == "levels":
        scene.note("info", "mouth-approximate", lid, f"the mouth follows only how loud '{sid}' is, as it has no words",
                   "give the sound words (a words file or list), or text for kine voice to speak")
    if got["unknown"]:
        shown = ", ".join(got["unknown"][:8]) + (" ..." if len(got["unknown"]) > 8 else "")
        scene.note("warning", "mouth-unknown-words", lid, f"the spelling rules cannot read {len(got['unknown'])} "
                   f"word{'s' * (len(got['unknown']) != 1)} of '{sid}' ({shown}); the mouth opens to C for them",
                   "spell them out (\"thirty five\", not 35), or use a voice backend that gives phones, such as kokoro")
    drawn = {"X": {}, **(av.spec.get("mouth") or {})} if av.model is not None else av.mouth.files if av.mouth else {}
    used: dict[str, int] = {}
    for s_, _, _ in got["mouth"]:
        if s_ not in drawn:
            used[s_] = used.get(s_, 0) + 1
    if used:
        scene.note("info", "avatar-missing-shape", lid, f"'{sid}' uses mouth shape" + "s" * (len(used) > 1) + " " +
                   ", ".join(f"{s_} ({n}x, shown as {av.shape(s_)})" for s_, n in sorted(used.items())) +
                   ", which the avatar does not " + ("map" if av.model is not None else "draw"),
                   "map them in the avatar's mouth for a closer match" if av.model is not None else
                   "draw them in the avatar's mouth shapes for a closer match")
    if words_file is not None and M.path_for(words_file).exists() and not M.fresh(words_file):
        scene.note("info", "mouth-stale", lid, f"{M.path_for(words_file).name} was made from other words than "
                   f"{words_file.name} holds now; the video uses the words, the file is out of date",
                   "run kine voice, or delete the file")


def _mouth(av: Avatar, mouth: dict, says: list, exprs: list, lid: str, fills: dict, scene) -> None:
    """The mouth part's drawings: each line's track shows its shapes, and an expression with a mouth shows that
    shape wherever the mouth would be at rest while it lasts (speech owns the mouth; the latest such expression wins).
    Every keyed layer starts early and holds its first key (see PRE), as cues cannot be read inside keys."""
    part, rig, rest = av.mouth, f"{lid}-rig", av.shape("X")
    if part is None:
        return
    biased = [(i, e, av.shape(av.spec["expressions"][e["name"]]["mouth"])) for i, e in enumerate(exprs)
              if "mouth" in av.spec["expressions"][e["name"]] and (e.get("weight", 1) or 0) > 0]
    biased = [b for b in biased if b[2] != rest]
    if not says and not biased:
        return
    kids = []
    for n, sid in enumerate(says or [None]):
        track, t0 = [], 0.0
        if sid:
            got, t0, words_file = _track(scene, sid)
            track = got["mouth"]
            _notes(scene, lid, sid, got, words_file, av)
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


IDLE_VERSION = 1  # change it when the event table changes, as it changes every video's blinks
IDLE = {"blink": [2.5, 5.0], "breath": 0.012, "saccade": [0.8, 2.5]}
BLINK = 0.12  # seconds the eyes are closed in a blink
QUIET = 0.15  # seconds either side of a gesture's peak, and after a line starts, that no blink lands in
HORIZON = 600.0  # seconds of idle events when the scene's duration is not a number


def _idle(av: Avatar, p, brand=None) -> dict:
    """The idle channels that are on: the avatar's idle, then its brand entry's, then the layer's, over the
    defaults; false turns off."""
    mine = p.get("idle")
    if not (mine is None or mine is False or isinstance(mine, dict) and set(mine) <= set(IDLE)):
        raise ValueError(f"layer '{p['id']}': idle is false, or an object of blink, breath and saccade such as "
                         '{"blink": false}')
    out: dict = dict(IDLE)
    for src in (av.spec.get("idle", {}), {} if brand is None else brand, mine or {} if mine is not False else False):
        if src is False:
            out = {k: False for k in out}
        elif isinstance(src, dict):
            out.update(src)
    return {**out, "seed": p.get("seed", 0)}


def _rng(lid: str, seed, channel: str):
    import random
    return random.Random(f"kinescript-idle-{IDLE_VERSION}:{lid}:{seed}:{channel}")


def _horizon(scene) -> float:
    d = (scene.spec.get("canvas") or {}).get("duration")
    return float(d) + 1 if _num(d) else HORIZON


def _art(G: dict, av: Avatar, rig: str) -> set[str]:
    """The ids of a part group's own drawings (or what wraps them): its children that are not other parts."""
    parts = {f"{rig}-{n}" for n in av.parts}
    return {str(c.get("id")) for c in G.get("children") or [] if c.get("id") not in parts}


def _blink_times(lid: str, idle: dict, scene) -> list[float]:
    """When each blink starts, in seconds from the avatar's start: seeded gaps between idle.blink's shortest and
    longest, now and then two blinks close together."""
    lo, hi = idle["blink"]
    rng, end, out = _rng(lid, idle["seed"], "blink"), _horizon(scene), []
    t = rng.uniform(lo / 2, hi)
    while t < end:
        out += [t, t + BLINK + 0.1] if rng.random() < 0.12 else [t]
        t += BLINK + rng.uniform(lo, hi)
    return out


def _blinks(G: dict, eye: Part, av: Avatar, lid: str, says: list, gests: list, idle: dict, scene) -> None:
    """Blinks at seeded times between idle.blink's shortest and longest gap (now and then twice), as opacity keys
    that swap the open and closed drawings. No blink shows near a gesture's peak or as a line starts: there an
    open copy shows instead, in groups timed by those cues (cues cannot be read inside keys)."""
    keys = [[0, 0, "hold"]]
    for b in _blink_times(lid, idle, scene):
        keys += [[round(b, 4), 1, "hold"], [round(b + BLINK, 4), 0, "hold"]]
    rig = f"{lid}-rig"
    by = {c.get("id"): c for c in G["children"]}
    shut, wide = by.get(f"{rig}-{eye.name}-closed"), by.get(f"{rig}-{eye.name}-open")
    if shut is None or wide is None:
        return
    start = f"{lid}.in"  # keys count from the avatar's start, whatever early group may come to hold them
    shut.update(opacity=keys, at=start)
    wide.update(opacity=[[a, 1 - v, e] for a, v, e in keys], at=start)
    quiet = [(f"{sid}.in", 0.0, QUIET) for sid in says]
    quiet += [(g.get("at", 0), peak(av.spec["gestures"][g["name"]]) - QUIET, peak(av.spec["gestures"][g["name"]]) + QUIET)
              for g in gests]
    slot = [shut]
    for i, (at, a, b) in enumerate(quiet):
        on = [[0, 0, "hold"], [round(PRE + a - BLINK, 4), 1, "hold"], [round(PRE + b, 4), 0, "hold"]]
        cue = _early(at, lid, scene.fps)
        copy_ = {**wide, "id": f"{wide['id']}-quiet{i + 1}", "opacity": keys}
        slot = [{"id": f"{lid}-blink{i + 1}-{eye.name}", "type": "group", "at": cue,
                 "opacity": [[x, 1 - v, e] for x, v, e in on], "children": slot},
                {"id": f"{lid}-quiet{i + 1}-{eye.name}", "type": "group", "at": cue, "opacity": on,
                 "children": [copy_]}]
    if len(slot) > 1:
        at = G["children"].index(shut)
        G["children"] = G["children"][:at] + slot + G["children"][at + 1:]


def _glances(eye: Part, idle: dict, lid: str, scene) -> dict:
    """Small seeded glances of the eyes and back, between idle.saccade's shortest and longest gap."""
    lo, hi = idle["saccade"]
    rng, end = _rng(lid, idle["seed"], "saccade"), _horizon(scene)
    xs, ys, t = [[0, 0.0]], [[0, 0.0]], rng.uniform(lo, hi)
    while t < end:
        a, r = rng.uniform(0, math.tau), rng.uniform(0.15, 0.35)
        hold = rng.uniform(0.25, 0.7)
        for ks, v in ((xs, math.cos(a) * r * eye.look[0]), (ys, math.sin(a) * r * eye.look[1])):
            ks += [[round(t, 4), ks[-1][1]], [round(t + 0.05, 4), round(v, 3)], [round(t + 0.05 + hold, 4), round(v, 3)],
                   [round(t + 0.1 + hold, 4), 0.0]]
        t += 0.1 + hold + rng.uniform(lo, hi)
    return {"x": xs, "y": ys}


LOOKS = ("left", "right", "up", "down")  # eye states a gaze picks, for eyes drawn looking each way instead of moved
SWAP = 0.06  # seconds one drawing of a part takes to give way to another


def _looking(av: Avatar) -> list[Part]:
    """Parts with states drawn looking aside (left, right, up, down) and no look reach: a gaze picks the drawing."""
    return [q for q in av.parts.values() if q.kind == "states" and q.look == (0.0, 0.0)
            and any(d in q.files for d in LOOKS)]


def _renamed(node: dict, tag: str) -> dict:
    """A copy of a layer tree with every id given the suffix -tag, so it can stand beside the original."""
    out = {k: v for k, v in node.items() if k != "children"}
    if "id" in out:
        out["id"] = f"{out['id']}-{tag}"
    if node.get("children"):
        out["children"] = [_renamed(c, tag) for c in node["children"]]
    return out


def _shown(G: dict, rig: str, part: Part, state: str, tag: str) -> dict | None:
    """A copy of a part's drawing of `state`, shown (its own drawing stays hidden)."""
    src = _find(G, f"{rig}-{part.name}-{state}")
    if src is None:
        return None
    copy_ = _renamed(src, tag)
    copy_["opacity"] = 1
    return copy_


def _state_for(gx: float, gy: float, part: Part) -> str | None:
    """The drawing a gaze at [gx, gy] picks: the way it looks most, when it looks aside enough and that is drawn."""
    if max(abs(gx), abs(gy)) < 0.3:
        return None
    d = ("left" if gx < 0 else "right") if abs(gx) >= abs(gy) else ("up" if gy < 0 else "down")
    return d if d in part.files else None


def _look_states(G: dict, av: Avatar, part: Part, rig: str, lid: str, picks: list, fps: float) -> None:
    """Each gaze shows the part's drawing for its direction (None: its own drawings, blinks and all) from its time
    until the next gaze's: each span's content sits in a group that comes on at its start, inside one that goes at
    the next span's start. Before the first gaze the part is as it was. picks: [(at, state or None)]."""
    if not any(st for _, st in picks):
        return
    own = _art(G, av, rig)
    art = [c for c in G["children"] if c.get("id") in own]
    if not art:
        return
    on = [[0, 0], [PRE, 0], [PRE + SWAP, 1]]
    off = [[0, 1], [PRE, 1], [PRE + SWAP, 0]]
    spans = [{"id": f"{lid}-look0-{part.name}", "type": "group", "at": _early(picks[0][0], lid, fps),
              "opacity": off, "children": art}]
    for i, (at, state) in enumerate(picks):
        tag = f"look{i + 1}"
        got = _shown(G, rig, part, state, tag) if state else None
        content = [got] if got else [_renamed(c, tag) for c in art]
        span = {"id": f"{lid}-{tag}-{part.name}", "type": "group", "at": _early(at, lid, fps), "opacity": on,
                "children": content}
        if i + 1 < len(picks):
            span = {"id": f"{lid}-{tag}-{part.name}-end", "type": "group", "at": _early(picks[i + 1][0], lid, fps),
                    "opacity": off, "children": [span]}
        spans.append(span)
    first = G["children"].index(art[0])
    G["children"] = ([c for c in G["children"][:first] if c not in art] + spans
                     + [c for c in G["children"][first:] if c not in art])


def _expression_state(G: dict, av: Avatar, part: Part, state: str, rig: str, gid: str, at: str, dur) -> None:
    """While an expression lasts, the part shows its drawing of `state` in place of the others (blinks included);
    a later expression's state wins over an earlier one's."""
    got = _shown(G, rig, part, state, gid)
    if got is None:
        return
    show = [[0, 0], [PRE, 0], [PRE + SWAP, 1]]
    hide = [[0, 1], [PRE, 1], [PRE + SWAP, 0]]
    if dur:
        show += [[PRE + dur - SWAP, 1], [PRE + dur, 0]]
        hide += [[PRE + dur - SWAP, 0], [PRE + dur, 1]]
    _wrap(G, f"{gid}-hide", at, {"opacity": hide}, _art(G, av, rig))
    ids = [c.get("id") for c in G["children"]]
    where = ids.index(f"{gid}-hide") + 1 if f"{gid}-hide" in ids else len(ids)
    G["children"].insert(where, {"id": f"{gid}-show", "type": "group", "at": at, "opacity": show,
                                 "children": [got]})


def component(p, theme: dict, scene) -> dict:
    """{"use": "avatar", "avatar": FOLDER, "x", "y", "h" or "w", "say", "expressions", "gaze", "gestures"}: the
    avatar as plain layers, centred on x, y at h pixels tall (its own size without h or w), its mouth following the
    sounds say names, with the expressions, gaze and gestures given at their times."""
    lid = p["id"]
    if not p.get("avatar"):
        raise ValueError(f"layer '{lid}': an avatar layer needs \"avatar\": the folder holding avatar.json")
    folder, mark = locate(p["avatar"], scene.base, scene.spec.get("brand"))
    av = load(folder)
    w, h = av.size
    k = (float(p["h"]) / h if p.get("h") is not None else float(p["w"]) / w if p.get("w") is not None else 1.0)
    exprs = _entries(p, "expressions", av.spec.get("expressions") or {})
    if av.model is not None:
        says = p.get("say") or []
        return A3.component(av, p, scene, says if isinstance(says, list) else [says], exprs,
                            _idle(av, p, mark.get("idle", {})), k)
    rig, fills = f"{lid}-rig", colors(av, theme)
    body = layers(av, rig, {}, theme)
    gazes = _entries(p, "gaze", None)
    gests = _entries(p, "gestures", av.spec.get("gestures") or {})
    says = p.get("say") or []
    says = says if isinstance(says, list) else [says]
    mouth = av.mouth and _find(body, f"{rig}-{av.mouth.name}")
    if mouth:
        _mouth(av, mouth, says, exprs, lid, fills, scene)

    eyes = [q for q in av.parts.values() if q.kind == "states" and q.look != (0.0, 0.0)]
    if gazes and not eyes and not _looking(av):
        raise ValueError(f"layer '{lid}': gaze moves the eyes by their look reach, and no part of the avatar has "
                         "look; give the eyes part \"look\": [x, y] in its avatar.json")
    idle = _idle(av, p, mark.get("idle", {}))
    for e in av.eyes if idle["blink"] else []:
        G = _find(body, f"{rig}-{e.name}")
        if G is not None:
            _blinks(G, e, av, lid, says, gests, idle, scene)
    was, picked = (0.0, 0.0), []
    for i, g in enumerate(gazes):  # each gaze moves the eyes on from the one before, so the moves add up
        try:
            gx, gy = _gaze(g.get("target", "camera"), p, av, k, scene, f"gaze[{i}]")
        except ValueError as e:
            scene.note("error", "avatar-gaze-target", lid, str(e), "")
            continue
        at = _early(g.get("at", 0), lid, scene.fps)
        for e in eyes:
            G = _find(body, f"{rig}-{e.name}")
            dx, dy = (gx - was[0]) * e.look[0], (gy - was[1]) * e.look[1]
            if G is None or (dx, dy) == (0, 0):
                continue
            props = {a: [[0, 0], [PRE, 0], [PRE + MOVE, round(d, 4)]] for a, d in (("x", dx), ("y", dy)) if d}
            _wrap(G, f"{lid}-gaze{i + 1}-{e.name}", at, props, _art(G, av, rig))
        was = (gx, gy)
        picked.append((g.get("at", 0), gx, gy))
    for e in _looking(av):
        G = _find(body, f"{rig}-{e.name}")
        if G is not None and picked:
            _look_states(G, av, e, rig, lid, [(at, _state_for(gx, gy, e)) for at, gx, gy in picked], scene.fps)

    for i, e in enumerate(exprs):
        pose, wt, dur = av.spec["expressions"][e["name"]], float(e.get("weight", 1)), e.get("dur")
        r = min(RAMP, dur / 2) if dur else RAMP
        at = _early(e.get("at", 0), lid, scene.fps)
        for pname, change in pose.items():
            G = _find(body, f"{rig}-{pname}") if pname != "mouth" else None
            if G is None:
                continue
            if change.get("state") and wt > 0:
                _expression_state(G, av, av.parts[pname], change["state"], rig, f"{lid}-expr{i + 1}-{pname}-state",
                                  at, dur)
            props = {}
            for a, v in change.items():
                if a == "state":
                    continue
                base = 1.0 if a in ("scale", "opacity") else 0.0
                to = round(base + wt * (v - base), 4)
                keys = [[0, base], [PRE, base], [PRE + r, to]]
                if dur:
                    keys += ([[PRE + dur - r, to]] if dur > 2 * r else []) + [[PRE + dur, base]]
                props[a] = keys
            if props:
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
            _wrap(G, gid, at, {a: {"keys": [[0, ks[0][1]]] + [[PRE + t, v] for t, v in ks], "smooth": True}
                               for a, ks in change.items()})  # through each key, as a hand moves
            if spans is not None:
                spans.append({"layer": lid, "name": g["name"], "part": pname, "id": gid, "dur": float(pose["dur"])})

    if idle["saccade"] and not gazes:  # an explicit gaze overrides the idle one
        for e in eyes:
            G = _find(body, f"{rig}-{e.name}")
            if G is not None:
                _wrap(G, f"{lid}-glance-{e.name}", f"{lid}.in", _glances(e, idle, lid, scene), _art(G, av, rig))
    if idle["breath"]:
        period = 4.0  # seconds a breath takes
        for name, q in av.parts.items():
            G = _find(body, f"{rig}-{name}") if q.parent is None else None
            if G is not None:
                _wrap(G, f"{lid}-breath-{name}", f"{lid}.in",
                      {"scale": f"=1+{idle['breath']:g}*(0.5-0.5*cos(tau*t/{period:g}))"})
    body.update(x=-w * k / 2, y=-h * k / 2, scale=k)
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "children": [body]}
