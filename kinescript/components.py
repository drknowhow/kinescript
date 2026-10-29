"""Reusable building blocks: {"use": "lowerThird", "title": "...", ...} expands into plain layers.

Scene files may also declare their own under "components" with {{param}} placeholders.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

THEME = {
    # Brand palette ("Kinetic Score"), plus the short names components use.
    "void": "#0a0c10", "sheet": "#f2f0e8", "track": "#242832", "cue": "#ff633f", "key": "#b9f45d",
    "annotation": "#9299a8",
    "ink": "#0a0c10", "paper": "#f2f0e8", "accent": "#ff633f", "accent2": "#b9f45d", "muted": "#9299a8",
    "font": "Space Grotesk", "mono": "IBM Plex Mono",
}


def _lower_third(p, th):
    title, sub = p.get("title", "Title"), p.get("subtitle", "")
    x, y = p.get("x", "6%"), p.get("y", "78%")
    accent = p.get("accent", th["accent"])
    plate = p.get("plate", "#0a0c10d0")
    box = {"fill": plate, "pad": [8, 16], "radius": 6} if plate else None
    kids = [
        {"id": f"{p['id']}-bar", "type": "rect", "x": 0, "y": 0, "w": 8, "h": 124 if sub else 72,
         "anchor": "topleft", "fill": accent, "radius": 4, "enter": {"name": "wipe-down", "dur": 0.35},
         "exit": {"name": "fade", "dur": 0.3}},
        {"id": f"{p['id']}-title", "type": "text", "text": title, "x": 28, "y": 0, "anchor": "topleft",
         "align": "left", "size": p.get("size", 54), "weight": 700, "fill": p.get("color", th["paper"]),
         "at": 0.12, "enter": {"name": "left", "dur": 0.45, "dist": 30}, "exit": {"name": "fade", "dur": 0.3},
         "shadow": {"y": 3, "blur": 12, "color": "#00000088"}, "box": box},
    ]
    if sub:
        kids.append({"id": f"{p['id']}-sub", "type": "text", "text": sub, "x": 28, "y": 76, "anchor": "topleft",
                     "align": "left", "size": round(p.get("size", 54) * 0.55), "weight": 500,
                     "fill": p.get("subColor", th["muted"]), "at": 0.3,
                     "enter": {"name": "left", "dur": 0.45, "dist": 20}, "exit": {"name": "fade", "dur": 0.3},
                     "box": box})
    return {"type": "group", "x": x, "y": y, "children": kids}


def _title_card(p, th):
    kids = []
    if p.get("kicker"):
        kids.append({"id": f"{p['id']}-kicker", "type": "text", "text": p["kicker"], "y": -110, "size": 30,
                     "weight": 600, "fill": p.get("accent", th["accent"]), "letterSpacing": 6,
                     "enter": {"name": "track", "dur": 0.8}, "exit": "fade"})
    kids.append({"id": f"{p['id']}-title", "type": "text", "text": p.get("title", "Title"), "size": p.get("size", 128),
                 "weight": 800, "fill": p.get("color", th["paper"]), "split": "chars", "at": 0.15,
                 "enter": {"name": "rise+blur", "dur": 0.6, "stagger": 0.035, "dist": 60},
                 "exit": {"name": "fade", "dur": 0.4}, "maxWidth": "86%"})
    if p.get("subtitle"):
        kids.append({"id": f"{p['id']}-sub", "type": "text", "text": p["subtitle"], "y": 120, "size": 40,
                     "weight": 400, "fill": th["muted"], "at": 0.6, "enter": "rise", "exit": "fade"})
    kids.append({"id": f"{p['id']}-rule", "type": "line", "points": [[-160, 0], [160, 0]], "y": 70 if not p.get("subtitle") else 76,
                 "stroke": p.get("accent", th["accent"]), "strokeWidth": 6, "cap": "round", "at": 0.45,
                 "enter": {"name": "draw", "dur": 0.7}, "exit": "fade"})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "children": kids}


def _callout(p, th):
    r = p.get("r", 60)
    accent = p.get("accent", th["accent"])
    side = p.get("side", "right")
    lx = r + 40 if side == "right" else -(r + 40)
    kids = [
        {"id": f"{p['id']}-ring", "type": "circle", "r": r, "stroke": accent, "strokeWidth": 6,
         "enter": {"name": "draw", "dur": 0.5}, "exit": "fade"},
        {"id": f"{p['id']}-pulse", "type": "circle", "r": r, "stroke": accent, "strokeWidth": 3,
         "scale": "=1 + (t % 1.2) / 1.2 * 0.5", "opacity": "=0.8 * (1 - (t % 1.2) / 1.2)", "at": 0.5},
    ]
    if p.get("label"):
        kids.append({"id": f"{p['id']}-label", "type": "text", "text": p["label"], "x": lx, "size": p.get("size", 36),
                     "weight": 700, "fill": th["ink"], "anchor": "left" if side == "right" else "right",
                     "align": "left" if side == "right" else "right",
                     "box": {"fill": p.get("boxColor", th["paper"]), "pad": [14, 22], "radius": 10},
                     "at": 0.25, "enter": {"name": "pop", "dur": 0.45}, "exit": "fade"})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "children": kids}


def _focus(p, th):
    """Highlight a region of footage: dim everything else and draw a frame around it."""
    x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("w", 400), p.get("h", 300)
    return {"type": "group", "children": [
        {"id": f"{p['id']}-frame", "type": "rect", "x": x, "y": y, "w": w, "h": h, "anchor": "topleft",
         "radius": p.get("radius", 12), "stroke": p.get("accent", th["accent"]), "strokeWidth": 5,
         "enter": {"name": "draw", "dur": 0.6}, "exit": "fade", "dimOutside": p.get("dim", 0.55)},
    ]}


def _progress(p, th):
    h = p.get("h", 8)
    return {"type": "rect", "x": 0, "y": p.get("y", "100%"), "anchor": "bottomleft", "h": h,
            "w": "=W * p", "fill": p.get("color", th["accent"])}


def _kinetic(p, th):
    """Big words, one after another: {"use": "kinetic", "words": ["Write", "Check", "Render"], "every": 0.6}."""
    every = float(p.get("every", 0.6))
    kids = []
    for i, word in enumerate(p.get("words", [])):
        kids.append({"id": f"{p['id']}-w{i}", "type": "text", "text": word, "at": round(i * every, 4), "dur": every,
                     "size": p.get("size", 180), "weight": 900, "fill": p.get("color", th["paper"]),
                     "enter": {"name": p.get("motion", "pop"), "dur": min(0.35, every * 0.6)},
                     "exit": {"name": "fade", "dur": min(0.12, every * 0.2)}})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "dur": every * len(kids) or 1,
            "children": kids}


# Parameters a builder reads only in some branches (so a single call may not touch them).
PARAMS = {"lowerThird": ("title", "subtitle", "x", "y", "accent", "size", "plate", "color", "subColor"),
          "titleCard": ("title", "subtitle", "kicker", "size", "accent", "color", "x", "y"),
          "callout": ("x", "y", "r", "label", "side", "accent", "size", "boxColor"),
          "focus": ("x", "y", "w", "h", "dim", "accent", "radius"),
          "progress": ("color", "h", "y"),
          "kinetic": ("words", "every", "size", "motion", "color", "x", "y")}

BUILTIN = {"lowerThird": _lower_third, "titleCard": _title_card, "callout": _callout, "focus": _focus,
           "progress": _progress, "kinetic": _kinetic}
COMPONENT_DOCS = {
    "lowerThird": "title, subtitle, x [\"6%\"], y [\"78%\"] (top-left of its accent bar), accent, size [54], "
                  "color, subColor, plate (box colour, or false for none)",
    "titleCard": "title, subtitle, kicker (small line above), x, y [\"50%\"] (centre), size [128], accent, color",
    "callout": "x, y (centre of the ring), r [60], label, side [right] (left|right), accent, size, boxColor",
    "focus": "x, y, w, h (top-left and size of the region to highlight), dim [0.55], accent, radius",
    "progress": "color, h [8], y [\"100%\"] (a bar along the bottom edge that fills over the whole scene)",
    "kinetic": "words [list], every [0.6] (seconds per word), size [180], motion [pop], color, x, y [\"50%\"] (centre)",
}
_PASS = ("id", "at", "dur", "until", "enter", "exit", "opacity", "scale", "rotate", "hidden")


class Params(dict):
    """Component parameters that remember which names the builder asked for, so a parameter
    nobody reads (a typo such as "subtitel") is reported instead of silently dropped."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set[str] = set()

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.read.add(k)
        return super().__contains__(k)


def _unused(name: str, raw: dict, known: set[str], scene):
    from .schema import COMMENT, suggest
    for k in raw:
        if k in known or k in _PASS or k in COMMENT or k in ("use", "type", "x", "y") or str(k).startswith(("_", "//")):
            continue
        close = suggest(k, sorted(known - set(_PASS) - {"id"}))
        msg = (f"component '{name}' has no parameter '{k}'." + (f" Did you mean '{close}'?" if close else "")
               + f" Its parameters: {', '.join(sorted(known - {'id'})) or 'none'}")
        if scene is None:
            raise ValueError(msg)
        scene.note("error", "unknown-param", raw.get("id"), msg, f"rename or remove '{k}'")


def _fill_template(v, params):
    if isinstance(v, str):
        m = re.fullmatch(r"\{\{(\w+)\}\}", v)
        if m:
            return copy.deepcopy(params.get(m.group(1), v))
        return re.sub(r"\{\{(\w+)\}\}", lambda mm: str(params.get(mm.group(1), mm.group(0))), v)
    if isinstance(v, list):
        return [_fill_template(x, params) for x in v]
    if isinstance(v, dict):
        return {k: _fill_template(x, params) for k, x in v.items()}
    return v


_counter = [0]


def expand(raw, user: dict, theme: dict, scene=None):
    if not isinstance(raw, dict):
        return raw
    if raw.get("type") == "sequence":
        return _sequence(raw)
    name = raw.get("use")
    if not name:
        return raw
    _counter[0] += 1
    params = Params(raw)
    params.setdefault("id", f"{name}{_counter[0]}")
    if name in user:
        comp = user[name]
        merged = dict(comp.get("params", {}), **params)
        body = _fill_template(copy.deepcopy(comp.get("layers", [])), merged)
        names = set(comp.get("params", {})) | set(re.findall(r"\{\{(\w+)\}\}", repr(comp.get("layers", []))))
        _unused(name, raw, names, scene)
        for i, k in enumerate(body):
            k.setdefault("id", f"{params['id']}-{i}")
            k["id"] = f"{params['id']}-{k['id']}" if not str(k["id"]).startswith(params["id"]) else k["id"]
        out = {"type": "group", "x": raw.get("x", 0), "y": raw.get("y", 0), "children": body}
    elif name in BUILTIN:
        out = BUILTIN[name](params, theme)
        _unused(name, raw, params.read | set(PARAMS.get(name, ())), scene)
    else:
        raise ValueError(f"unknown component '{name}'. Built-in: {', '.join(BUILTIN)}; scene: {', '.join(user) or 'none'}")
    for k in _PASS:
        if k in raw:
            out[k] = raw[k]
    out.setdefault("id", params["id"])
    return out


def _sequence(raw: dict) -> list[dict]:
    """Cut list -> back-to-back video layers. "transition": {"type": "fade", "dur": 0.4} overlaps them."""
    tr = raw.get("transition") or {}
    overlap = float(tr.get("dur", 0)) if tr else 0.0
    kind = tr.get("type", "fade") if tr else "none"
    base = raw.get("id", "seq")
    out = []
    for i, clip in enumerate(raw.get("clips", [])):
        c = dict(clip) if isinstance(clip, dict) else {"src": clip}
        c.setdefault("type", "video")
        c.setdefault("id", f"{base}-{i + 1}")
        for k in ("fit", "volume", "w", "h", "x", "y", "anchor"):
            if k in raw and k not in c:
                c[k] = raw[k]
        if i == 0:
            c.setdefault("at", raw.get("at", 0))
        else:
            prev_id = out[-1]["id"]
            c.setdefault("at", f"{prev_id}.out-{overlap}" if overlap else f"{prev_id}.out")
            if overlap and kind != "cut":
                c.setdefault("enter", {"name": {"fade": "fade", "wipe": "wipe-right", "slide": "left",
                                                "zoom": "zoom", "blur": "blur"}.get(kind, "fade"),
                                       "dur": overlap, "ease": "inOutSine", "dist": 120})
        out.append(c)
    return out


def _ts(s: str) -> float:
    h, m, rest = s.strip().replace(",", ".").split(":") if s.count(":") == 2 else ["0", *s.strip().replace(",", ".").split(":")]
    return int(h) * 3600 + int(m) * 60 + float(rest)


def load_cues(raw: dict, src: str | None) -> list[list]:
    cues = raw.get("cues")
    if cues:
        out = []
        for c in cues:
            if isinstance(c, dict):
                out.append([float(c.get("at", c.get("start", 0))), float(c.get("end")), str(c.get("text", ""))])
            else:
                out.append([float(c[0]), float(c[1]), str(c[2])])
        return out
    if not src:
        return []
    text = Path(src).read_text(encoding="utf-8-sig")
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        for j, line in enumerate(lines):
            if "-->" in line:
                a, b = line.split("-->")
                body = " ".join(lines[j + 1:])
                out.append([_ts(a), _ts(b.split()[0]), re.sub(r"<[^>]+>", "", body)])
                break
    return out
