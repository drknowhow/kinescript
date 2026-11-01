"""Reusable building blocks: {"use": "lowerThird", "title": "...", ...} expands into plain layers.

Scene files may also declare their own under "components" with {{param}} placeholders.
"""
from __future__ import annotations

import copy
import html
import math
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
    given = p.get("every")
    try:
        every = 0.6 if given is None else float(given) if not isinstance(given, bool) else math.nan
    except (TypeError, ValueError):
        every = math.nan
    if not 0 < every < 1e6:  # NaN fails this too
        raise ValueError(f"layer '{p['id']}': kinetic every is {given!r}; it is seconds per word, a number above 0 "
                         "(0.6 when left out)")
    words = p.get("words")
    if words is not None and not (isinstance(words, list) and all(isinstance(w, (str, int, float)) for w in words)):
        raise ValueError(f"layer '{p['id']}': kinetic words is {words!r}; it is a list of words like "
                         "[\"Write\", \"Check\", \"Render\"]")
    kids = []
    for i, word in enumerate(words or []):
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

# Parameters a builder computes with, so text there would fail inside it.
NUMBER_PARAMS = {"callout": ("r",), "lowerThird": ("size",)}

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
_PASS = ("id", "at", "dur", "until", "enter", "exit", "emphasis", "opacity", "scale", "rotate", "hidden", "z",
         "rotateX", "rotateY", "fixed", "sound")


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


# A placeholder in a scene component: {{title}}, or {{ title }} with spaces inside the braces.
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _fill_template(v, params):
    if isinstance(v, str):
        m = PLACEHOLDER.fullmatch(v)
        if m:
            return copy.deepcopy(params.get(m.group(1), v))
        return PLACEHOLDER.sub(lambda mm: str(params.get(mm.group(1), mm.group(0))), v)
    if isinstance(v, list):
        return [_fill_template(x, params) for x in v]
    if isinstance(v, dict):
        return {k: _fill_template(x, params) for k, x in v.items()}
    return v


_counter = [0]


def _own(layers, pid: str) -> None:
    """Put the instance id in front of the ids of a scene component's layers, nested ones too, so it can be used
    more than once."""
    for k in layers:
        if not isinstance(k, dict):
            continue
        if "id" in k and not str(k["id"]).startswith(f"{pid}-"):
            k["id"] = f"{pid}-{k['id']}"
        _own(k.get("children") or k.get("layers") or [], pid)  # a group may list its layers under either key


def _placeholders(v) -> list[str]:
    """The placeholder names used anywhere in a component's layers, in order of first use."""
    if isinstance(v, str):
        return PLACEHOLDER.findall(v)
    if isinstance(v, dict):
        v = list(v.values())
    return list(dict.fromkeys(n for x in v for n in _placeholders(x))) if isinstance(v, list) else []


def _unfilled(name: str, raw: dict, holes: list[str], values: dict, defaults: dict, scene):
    """A placeholder with no value would be drawn as the text "{{titel}}" or "None": report it instead."""
    from .schema import COMMENT, suggest
    given = sorted({str(k) for k in raw if k not in _PASS and k not in COMMENT and k not in ("use", "type", "x", "y")}
                   | set(defaults))
    for h in holes:
        if values.get(h) is not None:
            continue
        close = suggest(h, [g for g in given if g not in holes])
        msg = (f"component '{name}' uses {{{{{h}}}}}, but " + (f"'{h}' is null" if h in raw else f"no '{h}' is given")
               + " and the component's params set no default for it." + (f" Did you mean '{close}'?" if close else "")
               + f" Parameters given or with a default: {', '.join(given) or 'none'}")
        fix = (f"rename {{{{{h}}}}} to {{{{{close}}}}} in the component, or give '{h}'" if close else
               f"give \"{h}\" on the layer, or a default in the component's \"params\"")
        if scene is None:
            raise ValueError(msg)
        scene.note("error", "missing-param", raw.get("id") or name, msg, fix)


def expand(raw, user: dict, theme: dict, scene=None):
    if not isinstance(raw, dict):
        return raw
    if raw.get("type") == "sequence":
        return _sequence(raw)
    if raw.get("type") == "svg":
        return _svg(raw, scene)
    name = raw.get("use")
    if not name:
        return raw
    _counter[0] += 1
    params = Params(raw)
    params.setdefault("id", f"{name}{_counter[0]}")
    if name in user:
        comp = user[name]
        defaults = comp.get("params") if isinstance(comp.get("params"), dict) else {}
        merged = dict(defaults, **{k: v for k, v in params.items() if v is not None or k not in defaults})
        holes = _placeholders(comp.get("layers", []))
        _unfilled(name, raw, holes, merged, defaults, scene)
        body = _fill_template(copy.deepcopy(comp.get("layers", [])), merged)
        names = set(defaults) | set(holes)
        _unused(name, raw, names, scene)
        for i, k in enumerate(body):
            k.setdefault("id", f"{params['id']}-{i}")
        _own(body, params["id"])
        out = {"type": "group", "x": raw.get("x", 0), "y": raw.get("y", 0), "children": body}
    elif name in BUILTIN:
        for k in NUMBER_PARAMS.get(name, ()):  # the builder does arithmetic on these
            v = raw.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not -1e9 < v < 1e9):
                raise ValueError(f"layer '{params['id']}': {name} {k} is {v!r}; it takes a number. "
                                 f"Its parameters: {COMPONENT_DOCS[name]}")
        try:
            out = BUILTIN[name](params, theme)
        except (TypeError, AttributeError, KeyError, IndexError, ArithmeticError) as e:  # a parameter of a wrong type
            raise ValueError(f"layer '{params['id']}': component '{name}' cannot use a parameter as given "
                             f"({type(e).__name__}: {e}). Its parameters: {COMPONENT_DOCS[name]}") from e
        _unused(name, raw, params.read | set(PARAMS.get(name, ())), scene)
    else:
        raise ValueError(f"unknown component '{name}'. Built-in: {', '.join(BUILTIN)}; scene: {', '.join(user) or 'none'}")
    for k in _PASS:
        if k in raw:
            out[k] = raw[k]
    if "until" in raw and "dur" not in raw:
        out.pop("dur", None)  # the instance's until sets the end, not the length the builder gave it
    out.setdefault("id", params["id"])
    return out


_CLIP_KEYS = ("fit", "volume", "w", "h", "x", "y", "anchor")
TRANSITION_DUR = 0.5  # seconds, when a transition gives a type but no dur
_ENTERS = {"fade": "fade", "wipe": "wipe-right", "slide": "left", "zoom": "zoom", "blur": "blur"}
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg")  # as schema.IMAGE_EXTS


def _svg(raw: dict, scene) -> dict:
    """An svg layer -> a group of path layers, one for each shape in its file (svgdoc.layer). A file that cannot be
    read is an error on the layer, an id in parts that the file does not have another; what the layer leaves out of
    the file is a warning."""
    from . import svgdoc
    from .schema import suggest
    _counter[0] += 1
    lid = str(raw.get("id") or f"svg{_counter[0]}")
    src = Path(scene.path(raw["src"]) if scene is not None else raw["src"])
    empty = {k: v for k, v in raw.items() if k in _PASS or k in ("x", "y", "id")} | {"type": "group", "id": lid}

    def note(level, code, message, fix):
        if scene is None:
            raise ValueError(f"layer '{lid}': {message}; {fix}")
        scene.note(level, code, lid, message, fix)
    if not src.is_file():
        note("error", "media-missing", f"svg src not found: {src}", "paths are relative to the scene file")
        return empty
    try:
        group, skipped, unknown = svgdoc.layer(dict(raw, id=lid), src)
    except (OSError, ValueError) as e:
        note("error", "bad-media", f"svg src {e}", "save it again as svg from a drawing program")
        return empty
    if skipped:
        note("warning", "svg-unsupported", f"svg src {src.name} has {', '.join(n for v in skipped.values() for n in v)}"
             ", which the svg layer leaves out", "; ".join(dict.fromkeys(svgdoc.SKIPPED[k] for k in skipped)))
    ids = sorted({e.get("id") for e in svgdoc.read(src).iter() if e.get("id")}) if unknown else []
    for k in unknown:
        close = suggest(str(k), ids) if ids else None
        note("error", "unknown-part", f"parts '{k}': {src.name} has no element with that id",
             (f"did you mean '{close}'? " if close else "") + (f"its ids: {', '.join(ids[:20])}" if ids else
                                                                "it has no ids"))
    return group


def _sequence(raw: dict) -> dict:
    """Cut list -> a group of back-to-back video layers, timed within the sequence. The group has the sequence's
    id and ends with its last clip, so "cut.out" works as a cue. Its other layer keys (opacity, blur, fx, enter, ...)
    apply to the whole cut; fit, volume, w, h, x, y and anchor go to each clip.
    "transition": {"type": "fade", "dur": 0.4} overlaps the clips: the incoming clip enters over the outgoing one,
    and their sound crossfades over the same time (the outgoing clip's fadeOut, the incoming one's fadeIn). A cut
    never overlaps. The group's "sequence" key records each join it timed, so the scene can check that every
    transition is shorter than the clips it joins."""
    tr = raw.get("transition") or {}
    kind = (tr.get("type") or "fade") if tr else "cut"
    try:
        overlap = 0.0 if kind == "cut" else float(tr.get("dur", TRANSITION_DUR))
    except (TypeError, ValueError):
        raise ValueError(f"sequence '{raw.get('id', 'seq')}': transition dur is {tr.get('dur')!r}; it is seconds, "
                         "a number above 0") from None
    if not (math.isfinite(overlap) and overlap >= 0):
        raise ValueError(f"sequence '{raw.get('id', 'seq')}': transition dur is {overlap!r}; it is seconds, a number "
                         "above 0")
    base = raw.get("id", "seq")
    clips = raw.get("clips") or []
    if not isinstance(clips, list):
        raise ValueError(f"sequence '{base}': clips must be a list of files or objects with 'src'")
    out, joins = [], []
    for i, clip in enumerate(clips):
        c = dict(clip) if isinstance(clip, dict) else {"src": clip}
        image = isinstance(c.get("src"), str) and c["src"].lower().endswith(_IMAGE_EXTS)
        c.setdefault("type", "image" if image else "video")
        c.setdefault("id", f"{base}-{i + 1}")
        for k in _CLIP_KEYS:
            if k in raw and k not in c:
                c[k] = raw[k]
        if i == 0:
            c.setdefault("at", 0)
        elif "at" not in c:
            prev = out[-1]
            c["at"] = f"{prev['id']}.out-{overlap!r}" if overlap else f"{prev['id']}.out"
            if overlap:
                joins.append([prev["id"], c["id"]])
                c.setdefault("enter", {"name": _ENTERS.get(kind, "fade"), "dur": overlap, "ease": "inOutSine",
                                       "dist": 120})
                c.setdefault("fadeIn", overlap)  # the sound crossfades with the picture
                prev.setdefault("fadeOut", overlap)
        out.append(c)
    group = {k: v for k, v in raw.items() if k not in ("clips", "transition") + _CLIP_KEYS}
    group.update(type="group", id=base, children=out, span=True,
                 sequence={"type": kind, "overlap": overlap, "joins": joins})
    return group


_TIME = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:[.,]\d+)?)")
_LOOSE_ARROW = re.compile(r"\s*\d[\d:.,]*\s*->\s*\d")  # "00:00:01,500 -> 00:00:02,500": a typo for -->


def _ts(s: str) -> float:
    """Seconds from an SRT or WebVTT time: 00:01:02,500, 01:02.500 or 1:2.5; ValueError otherwise."""
    m = _TIME.fullmatch(s.strip())
    if not m:
        raise ValueError(s.strip())
    h, mi, sec = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + float(sec.replace(",", "."))


def _decode(data: bytes) -> tuple[str, str | None]:
    """Caption file bytes as text, and the encoding's name when it was not UTF-8. A byte-order mark says UTF-8, -16 or
    -32; zero bytes near the start mean UTF-16 without one; text that is not UTF-8 is read as Windows-1252, the
    usual encoding of subtitle files that are not, and Latin-1 when even that fails."""
    for bom, enc, name in ((b"\xef\xbb\xbf", "utf-8-sig", None), (b"\xff\xfe\x00\x00", "utf-32", "UTF-32"),
                           (b"\x00\x00\xfe\xff", "utf-32", "UTF-32"), (b"\xff\xfe", "utf-16", "UTF-16"),
                           (b"\xfe\xff", "utf-16", "UTF-16")):
        if data.startswith(bom):
            return data.decode(enc), name
    if b"\x00" in data[:400]:
        head = data[:400]
        le = head[1::2].count(0) >= head[0::2].count(0)
        return data.decode("utf-16-le" if le else "utf-16-be"), "UTF-16"
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "Windows-1252"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "Latin-1"


def _clean(text: str) -> str:
    """Cue text as it reads: no HTML tags (<i>, <b>, <font>, WebVTT's <v Name> and <c.x>), no ASS override tags
    ({\\an8}), and entities such as &amp; as the character."""
    text = re.sub(r"\{\\[^}]*\}", "", text)
    return html.unescape(re.sub(r"<[^>]*>", "", text)).strip()


def _stack(cues: list[list]) -> list[list]:
    """Cues that overlap show together, the earlier one on top, while both run. The renderer draws one cue at a
    time, so before only the first showed: A at 0.5-2.5 s and B at 1.5-3 s showed B from 2.5 s."""
    cues = sorted(cues, key=lambda c: (c[0], c[1]))
    out, i = [], 0
    while i < len(cues):
        group, end, j = [cues[i]], cues[i][1], i + 1
        while j < len(cues) and cues[j][0] < end - 1e-9:
            group.append(cues[j])
            end = max(end, cues[j][1])
            j += 1
        if len(group) == 1:
            out.append(group[0])
        else:
            cuts = sorted({t for c in group for t in c[:2]})
            for a, b in zip(cuts, cuts[1:]):
                on = [c[2] for c in group if c[0] <= a + 1e-9 and c[1] >= b - 1e-9]
                if on:
                    out.append([a, b, "\n".join(on)])
        i = j
    return out


def _inline_cues(cues) -> list[list]:
    if not isinstance(cues, list):
        raise ValueError(f"cues is {cues!r}; it is a list of [start, end, \"text\"] in seconds from the layer start")
    out = []
    for k, c in enumerate(cues):
        if isinstance(c, dict):
            s, e, t, rest = c.get("at", c.get("start", 0)), c.get("end"), c.get("text", ""), []
        elif isinstance(c, (list, tuple)) and len(c) >= 3:
            s, e, t, rest = c[0], c[1], c[2], list(c[3:])
        else:
            raise ValueError(f"cues[{k}] is {c!r}; a cue is [start, end, \"text\"] in seconds from the layer start, "
                             "or {\"at\": 1, \"end\": 2.5, \"text\": \"...\"}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (s, e)):
            raise ValueError(f"cues[{k}]: its start and end are {s!r} and {e!r}; they are numbers of seconds from the "
                             "layer start")
        if s < 0:
            raise ValueError(f"cues[{k}] starts at {s:g}s, before its layer does; cue times count from the layer start")
        if e <= s:
            raise ValueError(f"cues[{k}] ends at {e:g}s, which is not after its start at {s:g}s")
        out.append([float(s), float(e), str(t), *rest])
    return out


def load_cues(raw: dict, src: str | None, notes: list | None = None) -> list[list]:
    """A captions layer's cues, [start, end, text] in seconds from the layer start, from its \"cues\" or its .srt or
    .vtt file. A cue the file gets wrong is a ValueError naming the file and line; `notes` collects info notes
    (level, code, message, fix), such as an encoding read for you."""
    cues = raw.get("cues")
    if cues:
        return _stack(_inline_cues(cues))
    if not src:
        return []
    name = Path(src).name
    try:
        text, enc = _decode(Path(src).read_bytes())
    except UnicodeDecodeError as e:
        raise ValueError(f"captions src {name} is not readable text ({e.reason} at byte {e.start}); save it as UTF-8") \
            from None
    except OSError as e:
        raise ValueError(f"captions src {name} could not be read: {e.strerror or e}") from None
    if enc and notes is not None:
        notes.append(("info", "caption-encoding", f"captions src {name} is not UTF-8; read it as {enc}",
                      "save it as UTF-8"))
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def timing(line: str) -> bool:  # a cue's time line starts with its start time
        return bool(re.match(r"\s*\d", line)) and ("-->" in line or bool(_LOOSE_ARROW.match(line)))

    def at(part: str, no: int) -> float:
        try:
            return _ts(part)
        except ValueError:
            raise ValueError(f"captions src {name} line {no}: '{part.strip()}' is not a time; write "
                             "hours:minutes:seconds,milliseconds, like 00:00:01,500") from None

    out, loose, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        if not timing(line):
            i += 1
            continue
        no = i + 1
        a, b = re.split(r"-->" if "-->" in line else r"->", line, maxsplit=1)
        if "-->" not in line:
            loose.append(no)
        if not b.split():
            raise ValueError(f"captions src {name} line {no}: the cue '{line.strip()}' has no end time; write "
                             "00:00:01,500 --> 00:00:03,000")
        start, end = at(a, no), at(b.split()[0], no)  # after the end time, WebVTT puts cue settings
        if end <= start:
            raise ValueError(f"captions src {name} line {no}: the cue ends at {end:.3f}s, which is not after its "
                             f"start at {start:.3f}s")
        body, i = [], i + 1
        while i < len(lines) and lines[i].strip() and not timing(lines[i]):
            body.append(lines[i].strip())
            i += 1
        if i < len(lines) and timing(lines[i]) and body and body[-1].isdigit():
            body.pop()  # the next cue's number, with no blank line before it
        if txt := _clean(" ".join(body)):
            out.append([start, end, txt])
    if loose and notes is not None:
        notes.append(("info", "caption-arrow", f"captions src {name}: '->' on line{'s' if len(loose) > 1 else ''} "
                      f"{', '.join(map(str, loose))} read as '-->'", "write -->"))
    return _stack(out)
