"""Reusable building blocks: {"use": "lowerThird", "title": "...", ...} expands into plain layers.

Scene files may also declare their own under "components" with {{param}} placeholders.
"""
from __future__ import annotations

import copy
import html
import math
import re
from pathlib import Path

from . import color as C
from .props import FIELD

THEME = {
    # Brand palette ("Kinetic Score"), plus the short names components use.
    "void": "#0a0c10", "sheet": "#f2f0e8", "track": "#242832", "cue": "#ff633f", "key": "#b9f45d",
    "annotation": "#9299a8",
    "ink": "#0a0c10", "paper": "#f2f0e8", "accent": "#ff633f", "accent2": "#b9f45d", "muted": "#9299a8",
    "font": "Space Grotesk", "fontDisplay": "Space Grotesk", "mono": "IBM Plex Mono",
}

# Text styles a layer names with "textStyle"; a brand's "text" part changes them or adds more.
TEXT_STYLES = {
    "title": {"font": "display", "size": 128, "weight": 800},
    "heading": {"font": "display", "size": 72, "weight": 700},
    "body": {"font": "body", "size": 40, "weight": 400},
    "caption": {"font": "body", "size": 28, "weight": 500},
    "label": {"font": "mono", "size": 24, "weight": 500},
}
FACES = {"display": "fontDisplay", "body": "font", "mono": "mono"}
_STYLE_KEYS = {"tracking": "letterSpacing", "letterSpacing": "letterSpacing", "size": "size", "weight": "weight",
               "case": "case", "italic": "italic", "lineHeight": "lineHeight"}


def group(theme: dict, name: str) -> dict:
    """A brand part (text, charts, components...) as an object, from its "name.key" theme values."""
    pre = name + "."
    return {k[len(pre):]: v for k, v in theme.items() if k.startswith(pre)}


def text_styles(theme: dict) -> list[str]:
    return sorted(set(TEXT_STYLES) | set(group(theme, "text")))


def text_style(name, theme: dict) -> dict | None:
    """The text keys style `name` sets (font as a family name), or None when there is no such style."""
    own = theme.get(f"text.{name}") if isinstance(name, str) else None
    if not isinstance(own, dict) and name not in TEXT_STYLES:
        return None
    out = {}
    for k, v in {**TEXT_STYLES.get(name, {}), **(own or {})}.items():
        if k == "font":
            out["font"] = theme.get(FACES[v], v) if v in FACES else v
        elif k in _STYLE_KEYS:
            out[_STYLE_KEYS[k]] = v
    return out


def apply_text_styles(layers, theme: dict) -> None:
    """Fill each layer's keys from its "textStyle" (keys the layer sets itself win), in groups too."""
    for L in layers if isinstance(layers, list) else ():
        if not isinstance(L, dict):
            continue
        if "textStyle" in L:
            st = text_style(L["textStyle"], theme)
            if st is None:
                raise ValueError(f"layer '{L.get('id', '?')}': textStyle is {L['textStyle']!r}; the styles are: "
                                 + ", ".join(text_styles(theme)))
            for k, v in st.items():
                L.setdefault(k, v)
        apply_text_styles(L.get("children") or L.get("layers"), theme)


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


# A pie's slice colours in turn: the brand's cue and keyframe, then sheet, annotation and a darker grey.
CHART_COLORS = ["#ff633f", "#b9f45d", "#f2f0e8", "#9299a8", "#5b6272"]


def _series(p, name: str, negative: bool = False) -> tuple[list[str], list[float]]:
    """A chart's data as labels and values: [12, 30, 22], [["Q1", 12], ...] or [{"label": "Q1", "value": 12}, ...],
    with labels from "labels" where given."""
    data = p.get("data")
    shape = ("a list of numbers like [12, 30, 22], of [label, number] pairs or of {\"label\": \"Q1\", \"value\": 12} "
             "objects")
    if not isinstance(data, list) or not data:
        raise ValueError(f"layer '{p['id']}': {name} data is {data!r}; it is {shape}")
    labels, values = [], []
    for d in data:
        lab, v = (d.get("label"), d.get("value")) if isinstance(d, dict) else \
            tuple(d) if isinstance(d, (list, tuple)) and len(d) == 2 else (None, d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v < 0 and not negative):
            raise ValueError(f"layer '{p['id']}': {name} data has {d!r}; it is {shape}"
                             + ("" if negative else ", each 0 or more"))
        labels.append("" if lab is None else str(lab))
        values.append(float(v))
    given = p.get("labels")
    if given is not None:
        if not isinstance(given, list):
            raise ValueError(f"layer '{p['id']}': {name} labels is {given!r}; it is a list of text, one for each value")
        labels = [str(given[i]) if i < len(given) else labels[i] for i in range(len(values))]
    return labels, values


def _shown(p, v: float, default: str) -> str:
    """v written with the chart's format, as a counter writes its value."""
    fmt = p.get("format", default)
    try:
        return FIELD.sub(lambda m: format(v, m.group(1)), fmt)
    except (TypeError, ValueError) as e:
        raise ValueError(f"layer '{p['id']}': format is {fmt!r}; it is text with {{:...}} where the number goes, "
                         f"like \"{{:,.0f}}\" or \"{{:.1f}}%\" ({e})") from None


def _sine_at(f: float) -> float:
    """When inOutSine reaches f (0..1), as a share of its time."""
    return math.acos(max(-1.0, min(1.0, 1 - 2 * f))) / math.pi


def _axis(pid: str, w: float, y: float, th) -> dict:
    axis = th.get("charts.axis") if isinstance(th.get("charts.axis"), dict) else {}
    return {"id": f"{pid}-axis", "type": "line", "points": [[-w / 2, round(y, 3)], [w / 2, round(y, 3)]],
            "stroke": _themed(axis.get("color", th["muted"]), th), "strokeWidth": axis.get("width", 3), "cap": "butt",
            "enter": {"name": "draw", "dur": 0.5}}


def _bar_chart(p, th):
    """Bars that grow from a baseline in turn, each with its value counting up above it and its label below."""
    labels, values = _series(p, "barChart")
    pid, n = p["id"], len(values)
    w, h = p.get("w", 1200), p.get("h", 600)
    top = p.get("max") or max(values) or 1.0
    slot = w / n
    bw = slot * (1 - min(0.9, max(0.0, p.get("gap", 0.3))))
    grow, stagger = p.get("grow", 1.0), p.get("stagger", 0.12)
    size, base = p.get("size", 30), h / 2
    colors, hi = p.get("colors"), p.get("highlight")
    picked = hi if isinstance(hi, list) else [] if hi is None else [hi]
    fmt = p.get("format", "{:,.0f}")
    _shown(p, values[0], fmt)
    kids = [_axis(pid, w, base, th)]
    for i, (lab, v) in enumerate(zip(labels, values)):
        x, bh, at = round(-w / 2 + slot * (i + 0.5), 3), round(h * min(1.0, v / top), 3), round(i * stagger, 4)
        fill = colors[i % len(colors)] if isinstance(colors, list) and colors else \
            p.get("accent", th["accent"]) if i + 1 in picked or (lab and lab in picked) else p.get("color", th["paper"])
        kids.append({"id": f"{pid}-bar{i + 1}", "type": "rect", "x": x, "y": base, "anchor": "bottom",
                     "w": round(bw, 3), "h": {"from": 0, "to": bh, "dur": grow, "ease": "outQuad"},
                     "radius": round(min(6, bw / 4), 3), "fill": fill, "at": at})
        if p.get("values", True):  # 40 px above the bar on the same tween, so it rides on top
            kids.append({"id": f"{pid}-value{i + 1}", "type": "counter", "x": x,
                         "y": {"from": base - 40, "to": base - bh - 40, "dur": grow, "ease": "outQuad"},
                         "value": {"from": 0, "to": v, "dur": grow, "ease": "outQuad"}, "format": fmt,
                         "size": p.get("valueSize", round(size * 4 / 3)), "weight": 700, "fill": th["paper"],
                         "at": at})
        if lab:
            kids.append({"id": f"{pid}-label{i + 1}", "type": "text", "text": lab, "x": x, "y": base + 20,
                         "anchor": "top", "size": size, "fill": th["muted"], "at": at, "enter": "fade"})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "anchor": "center", "w": w, "h": h,
            "children": kids}


def _line_chart(p, th):
    """A line that draws on through its points, with a dot and the value popping on at each as the line gets there."""
    labels, values = _series(p, "lineChart", negative=True)
    pid, n = p["id"], len(values)
    if n < 2:
        raise ValueError(f"layer '{pid}': lineChart data has 1 value; a line needs 2 or more")
    w, h = p.get("w", 1200), p.get("h", 600)
    lo = p.get("min", min(0.0, *values))
    hi = p.get("max", max(values))
    hi = hi if hi > lo else lo + 1
    grow, size, color = p.get("grow", 1.5), p.get("size", 30), p.get("color", th["accent"])
    pts = [(-w / 2 + w * i / (n - 1), h / 2 - h * (v - lo) / (hi - lo)) for i, v in enumerate(values)]
    run = [0.0]  # length of the line up to each point
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        run.append(run[-1] + math.hypot(bx - ax, by - ay))
    kids = [_axis(pid, w, h / 2 - h * (min(max(0.0, lo), hi) - lo) / (hi - lo), th),
            {"id": f"{pid}-line", "type": "line", "points": [[round(x, 3), round(y, 3)] for x, y in pts],
             "stroke": color, "strokeWidth": p.get("strokeWidth", 6), "cap": "round", "join": "round", "at": 0.2,
             "enter": {"name": "draw", "dur": grow, "ease": "inOutSine"}}]
    for i, ((x, y), lab, v) in enumerate(zip(pts, labels, values)):
        at = round(0.2 + grow * _sine_at(run[i] / (run[-1] or 1)), 4)  # when the line reaches it
        if p.get("dots", True):
            kids.append({"id": f"{pid}-dot{i + 1}", "type": "circle", "x": round(x, 3), "y": round(y, 3),
                         "r": p.get("r", 9), "fill": color, "at": at, "enter": {"name": "pop", "dur": 0.3}})
        if p.get("values", True):
            kids.append({"id": f"{pid}-value{i + 1}", "type": "text", "text": _shown(p, v, "{:,.0f}"),
                         "x": round(x, 3), "y": round(y - 22, 3), "anchor": "bottom", "size": size, "weight": 700,
                         "fill": th["paper"], "at": at, "enter": {"name": "rise", "dur": 0.4, "dist": 16}})
        if lab:
            kids.append({"id": f"{pid}-label{i + 1}", "type": "text", "text": lab, "x": round(x, 3), "y": h / 2 + 20,
                         "anchor": "top", "size": size, "fill": th["muted"], "enter": "fade"})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "anchor": "center", "w": w, "h": h,
            "children": kids}


def _arc(r: float, a0: float, a1: float) -> str:
    """SVG path data for an arc of radius r about (0, 0), clockwise from a0 to a1 degrees (0 is three o'clock)."""
    if a1 - a0 > 359.99:  # a whole ring, as two halves
        rest = _arc(r, a0 + 180, a1)
        return _arc(r, a0, a0 + 180) + " " + rest[rest.index("A"):]
    x0, y0 = r * math.cos(math.radians(a0)), r * math.sin(math.radians(a0))
    x1, y1 = r * math.cos(math.radians(a1)), r * math.sin(math.radians(a1))
    return f"M{x0:.3f} {y0:.3f} A{r:.3f} {r:.3f} 0 {int(a1 - a0 > 180)} 1 {x1:.3f} {y1:.3f}"


def _pie_chart(p, th):
    """Slices that sweep round from twelve o'clock in turn, each labelled outside with its share. inner above 0 leaves
    a hole in the middle (a doughnut)."""
    labels, values = _series(p, "pieChart")
    pid, total = p["id"], sum(values)
    if total <= 0:
        raise ValueError(f"layer '{pid}': pieChart data adds up to 0; a pie needs a value above 0")
    r, inner = p.get("r", 240), min(0.95, max(0.0, p.get("inner", 0)))
    grow, size = p.get("grow", 1.2), p.get("size", 30)
    colors = p.get("colors") if isinstance(p.get("colors"), list) and p.get("colors") else CHART_COLORS
    mid, width = r * (1 + inner) / 2, r * (1 - inner)  # each slice is a thick arc that draws on along its length
    kids, done = [], 0.0
    for i, (lab, v) in enumerate(zip(labels, values)):
        if v <= 0:
            continue
        f0, f1 = done / total, (done + v) / total
        done += v
        a0, a1 = -90 + 360 * f0, -90 + 360 * f1
        t0, t1 = grow * _sine_at(f0), grow * _sine_at(f1)
        end = a1 if f1 > 1 - 1e-9 else a1 + 0.6  # tucked under the next slice, so no gap shows between them
        kids.append({"id": f"{pid}-slice{i + 1}", "type": "path", "d": _arc(mid, a0, end),
                     "stroke": colors[i % len(colors)], "strokeWidth": round(width, 3), "cap": "butt",
                     "at": round(t0, 4), "enter": {"name": "draw", "dur": round(max(t1 - t0, 0.01), 4),
                                                   "ease": "linear"}})
        m = math.radians((a0 + a1) / 2)
        cx, cy = math.cos(m), math.sin(m)
        share = _shown(p, 100 * v / total, "{:.0f}%")
        kids.append({"id": f"{pid}-label{i + 1}", "type": "text", "text": f"{lab}  {share}" if lab else share,
                     "x": round((r + 28) * cx, 3), "y": round((r + 28) * cy, 3),
                     "anchor": "left" if cx > 0.25 else "right" if cx < -0.25 else "top" if cy > 0 else "bottom",
                     "size": size, "fill": th["paper"], "at": round(t1, 4), "enter": "fade"})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "anchor": "center", "w": 2 * r,
            "h": 2 * r, "children": kids}


# Where a logo goes when the brand names no such placement: a corner or edge, inside the safe area.
PLACES = ("topleft", "top", "topright", "left", "center", "right", "bottomleft", "bottom", "bottomright")


def _pct(v, whole: float) -> float:
    """px from a number or "6%" of `whole`."""
    if isinstance(v, str) and v.strip().endswith("%"):
        return float(v.strip()[:-1]) / 100 * whole
    return float(v)


def _logo(p, th, scene=None):
    """A brand logo as an image layer, at one of the brand's placements or a corner: {"use": "logo", "logo": "mark",
    "placement": "bottomright"}. On a dark background it takes the logo's onDark file when it has one."""
    W, H = (scene.w, scene.h) if scene is not None else (1920, 1080)
    logos = group(th, "logos")
    entries = {k: v for k, v in logos.items() if isinstance(v, dict)}
    name = p.get("logo") or ("full" if "full" in entries else next(iter(entries), None))
    if p.get("src"):
        src = p["src"]
    elif name in entries:
        e = entries[name]
        dark = p.get("dark")
        if dark is None:
            dark = C.luminance(C.parse(th.get("bg", th["void"]))) < 0.18
        src = e.get("onDark" if dark else "onLight") or e["file"]
    else:
        raise ValueError(f"layer '{p['id']}': logo is {name!r}, but " + (
            f"the brand's logos are: {', '.join(entries)}" if entries else
            "the scene's brand has no logos (add \"logos\": {\"full\": {\"file\": \"logos/full.svg\"}} to "
            "brand.json), and no src is given"))
    places = group(th, "placements")
    where = p.get("placement", "bottomright")
    if where in places and isinstance(places[where], dict):
        place = places[where]
    elif where in PLACES:
        place = {"anchor": where}
    else:
        raise ValueError(f"layer '{p['id']}': placement is {where!r}; use one of the brand's placements "
                         f"({', '.join(places) or 'none'}) or a corner or edge: {', '.join(PLACES)}")
    anchor = place.get("anchor", "bottomright")
    margin = place.get("margin", th.get("safeArea", "5%"))
    h = round(_pct(p.get("h", place.get("height", "8%")), H), 3)
    fx = 0.5 if anchor in ("top", "bottom", "center") else 1.0 if anchor.endswith("right") else 0.0
    fy = 0.5 if anchor in ("left", "right", "center") else 1.0 if anchor.startswith("bottom") else 0.0
    mx, my = _pct(margin, W), _pct(margin, H)
    x = mx + fx * (W - 2 * mx)
    y = my + fy * (H - 2 * my)
    clear = logos.get("clearSpace")
    if scene is not None and isinstance(clear, (int, float)) and clear > 0:
        scene.logos[p["id"]] = float(clear)  # check keeps other layers out of this share of the logo's box
    least = (logos.get("minHeight") if isinstance(logos.get("minHeight"), (int, float)) else 0) * H / 1080
    if scene is not None and h < least:
        scene.note("warning", "logo-small", p["id"], f"the logo is {h:g} px tall; the brand asks for at least "
                   f"{least:g} px at this size", f"set h to {least:g} or more, or a bigger placement height")
    out = {"type": "image", "src": src, "h": h, "fit": "contain", "anchor": anchor,
           "x": round(p.get("x", x), 3) if isinstance(p.get("x", x), (int, float)) else p["x"],
           "y": round(p.get("y", y), 3) if isinstance(p.get("y", y), (int, float)) else p["y"]}
    if p.get("w") is not None:
        out["w"] = _pct(p["w"], W)
    if p.get("opacity", place.get("opacity")) is not None:
        out["opacity"] = p.get("opacity", place.get("opacity"))
    return out


def _bug(p, th, scene=None):
    """The brand's bug: its logo at its placement and opacity for the whole video, unless timed."""
    bug = group(th, "bug")
    q = Params({**{k: bug[k] for k in ("logo", "placement", "opacity", "h") if k in bug},
                **{k: v for k, v in p.items() if v is not None}})
    q.setdefault("logo", "mark" if "mark" in group(th, "logos") else None)
    q.setdefault("opacity", 0.85)
    return _logo(q, th, scene)


def _bookend_logo(p, th, scene, h):
    """The brand logo centred at the group's origin, or None when the brand has no logos (then the name shows)."""
    logos = {k: v for k, v in group(th, "logos").items() if isinstance(v, dict)}
    if not logos and not p.get("src"):
        return None
    W = scene.w if scene is not None else 1920
    q = {"id": f"{p['id']}-logo", "logo": p.get("logo"), "src": p.get("src"), "placement": "center", "h": h,
         "w": round(W * 0.8, 3)}  # contained in that box, so a wide logo fits a tall frame
    out = _logo({k: v for k, v in q.items() if v is not None}, th, scene)
    out.update(id=q["id"], x=0, y=0)
    return out


def _sting(p, th, scene=None):
    """A brand intro: the logo (or the brand's name) comes in, a tagline follows, both leave before the end.
    Read from the brand's "intro"; "intro": true on a scene puts it before the video (see BRANDS)."""
    p = {**{k: v for k, v in group(th, "intro").items() if k not in ("scene", "dur")}, **p}
    H = scene.h if scene is not None else 1080
    h = _pct(p.get("h", "22%"), H)
    enter = p.get("enter", "pop")
    mark = _bookend_logo(p, th, scene, h)
    kids = []
    if mark is not None:
        kids.append({**mark, "enter": {"name": enter, "dur": 0.7} if isinstance(enter, str) else enter,
                     "exit": {"name": "fade", "dur": 0.4}})
        below = h / 2
    else:
        name = p.get("name") or th.get("brandName") or ""
        kids.append({"id": f"{p['id']}-name", "type": "text", "text": name or "Brand", "textStyle": "title",
                     "fill": th["paper"], "maxWidth": "86%", "enter": {"name": "rise+blur", "dur": 0.7},
                     "exit": {"name": "fade", "dur": 0.4}})
        below = H * 0.07
    if p.get("tagline"):
        kids.append({"id": f"{p['id']}-tagline", "type": "text", "text": p["tagline"], "textStyle": "caption",
                     "fill": th["muted"], "y": round(below + H * 0.02 + float((text_style("caption", th) or {})
                                                                           .get("size", 28)), 3),
                     "at": 0.35, "enter": "rise",
                     "exit": {"name": "fade", "dur": 0.3}, "maxWidth": "86%"})
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "children": kids}


def _end_card(p, th, scene=None):
    """A brand outro: logo, a title, a subtitle and a link line, staggered in, held to the end.
    Read from the brand's "outro"; "outro": true on a scene puts it after the video (see BRANDS)."""
    p = {**{k: v for k, v in group(th, "outro").items() if k not in ("scene", "dur")}, **p}
    H = scene.h if scene is not None else 1080
    h = _pct(p.get("h", "16%"), H)
    mark = _bookend_logo(p, th, scene, h)
    kids, top, bottom = [], 0.0, 0.0
    if mark is not None:
        kids.append({**mark, "y": 0, "enter": {"name": "rise", "dur": 0.6}})
        top, bottom = -h / 2, h / 2 + H * 0.03  # a gap under the logo
    at = 0.2
    for key, style, fill in (("title", "heading", th["paper"]), ("subtitle", "body", th["muted"]),
                             ("url", "label", th["accent"])):
        if p.get(key):
            size = float((text_style(style, th) or {}).get("size", 40))
            y = bottom + size * 0.7 if kids else 0.0
            if not kids:
                top = -size * 0.6
            kids.append({"id": f"{p['id']}-{key}", "type": "text", "text": p[key], "textStyle": style, "fill": fill,
                         "y": round(y, 3), "at": at, "enter": "rise", "maxWidth": "86%"})
            bottom = y + size * 0.6
            at += 0.2
    if not kids:
        kids.append({"id": f"{p['id']}-name", "type": "text", "text": th.get("brandName") or "Thanks for watching",
                     "textStyle": "title", "fill": th["paper"], "enter": "rise", "maxWidth": "86%"})
    shift = -(top + bottom) / 2  # centre what is there on the frame
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", f"50%{shift:+.1f}"), "children": kids}


# Parameters a builder reads only in some branches (so a single call may not touch them).
PARAMS = {"lowerThird": ("title", "subtitle", "x", "y", "accent", "size", "plate", "color", "subColor"),
          "titleCard": ("title", "subtitle", "kicker", "size", "accent", "color", "x", "y"),
          "callout": ("x", "y", "r", "label", "side", "accent", "size", "boxColor"),
          "focus": ("x", "y", "w", "h", "dim", "accent", "radius"),
          "progress": ("color", "h", "y"),
          "kinetic": ("words", "every", "size", "motion", "color", "x", "y"),
          "barChart": ("data", "labels", "x", "y", "w", "h", "max", "gap", "grow", "stagger", "size", "valueSize",
                       "color", "colors", "highlight", "accent", "values", "format"),
          "lineChart": ("data", "labels", "x", "y", "w", "h", "min", "max", "grow", "size", "color", "strokeWidth",
                        "r", "dots", "values", "format"),
          "pieChart": ("data", "labels", "x", "y", "r", "inner", "grow", "size", "colors", "format"),
          "logo": ("logo", "placement", "src", "dark", "h", "w", "x", "y", "opacity"),
          "bug": ("logo", "placement", "src", "dark", "h", "w", "x", "y", "opacity"),
          "sting": ("logo", "src", "h", "tagline", "name", "enter", "x", "y"),
          "endCard": ("logo", "src", "h", "title", "subtitle", "url", "x", "y")}

# Parameters a builder computes with, so text there would fail inside it.
NUMBER_PARAMS = {"callout": ("r",), "lowerThird": ("size",),
                 "barChart": ("w", "h", "max", "gap", "grow", "stagger", "size", "valueSize"),
                 "lineChart": ("w", "h", "min", "max", "grow", "size", "strokeWidth", "r"),
                 "pieChart": ("r", "inner", "grow", "size")}

BUILTIN = {"lowerThird": _lower_third, "titleCard": _title_card, "callout": _callout, "focus": _focus,
           "progress": _progress, "kinetic": _kinetic, "barChart": _bar_chart, "lineChart": _line_chart,
           "pieChart": _pie_chart, "logo": _logo, "bug": _bug, "sting": _sting, "endCard": _end_card,
           "avatar": lambda p, th, scene: _avatar(p, th, scene)}
NEEDS_SCENE = {"logo", "bug", "sting", "endCard", "avatar"}  # builders that read the canvas size or files
PARAMS["avatar"] = ("avatar", "say", "x", "y", "w", "h", "expressions", "gaze", "gestures", "idle", "seed")
NUMBER_PARAMS["avatar"] = ("w", "h")


def _avatar(p, th, scene):
    from .avatar import AvatarError, component
    try:
        return component(p, th, scene)
    except AvatarError as e:
        raise ValueError(f"layer '{p['id']}': {e}") from None


COMPONENT_DOCS = {
    "avatar": "avatar (the folder holding avatar.json), x, y [\"50%\"] (its middle), h or w (its size; its own "
              "without), say (a sound id, {\"text\": ...} or a list of them), expressions, gaze, gestures, idle, seed (see AVATARS)",
    "lowerThird": "title, subtitle, x [\"6%\"], y [\"78%\"] (top-left of its accent bar), accent, size [54], "
                  "color, subColor, plate (box colour, or false for none)",
    "titleCard": "title, subtitle, kicker (small line above), x, y [\"50%\"] (centre), size [128], accent, color",
    "callout": "x, y (centre of the ring), r [60], label, side [right] (left|right), accent, size, boxColor",
    "focus": "x, y, w, h (top-left and size of the region to highlight), dim [0.55], accent, radius",
    "progress": "color, h [8], y [\"100%\"] (a bar along the bottom edge that fills over the whole scene)",
    "kinetic": "words [list], every [0.6] (seconds per word), size [180], motion [pop], color, x, y [\"50%\"] (centre)",
    "barChart": "data ([12, 30] or [[\"Q1\", 12], ...] or [{\"label\": \"Q1\", \"value\": 12}, ...]), labels, x, y "
                "[\"50%\"] (centre), w [1200], h [600], max, gap [0.3], grow [1] (seconds a bar grows), "
                "stagger [0.12], color, colors (one a bar, in turn), highlight (bar number from 1, label, or a list), "
                "accent, values [true], format [\"{:,.0f}\"], size [30], valueSize [40]",
    "lineChart": "data (as barChart's; below 0 is fine), labels, x, y [\"50%\"] (centre), w [1200], h [600], min, "
                 "max, grow [1.5] (seconds the line draws on), color, strokeWidth [6], r [9] (dots), dots [true], "
                 "values [true], format [\"{:,.0f}\"], size [30]",
    "pieChart": "data (as barChart's), labels, x, y [\"50%\"] (centre), r [240], inner [0] (0..0.95: a hole for a "
                "doughnut), grow [1.2] (seconds to sweep round), colors, format [\"{:.0f}%\"], size [30]",
    "logo": "logo [full] (a name in the brand's logos), placement [bottomright] (a brand placement, or topleft|top|"
            "topright|left|center|right|bottomleft|bottom|bottomright inside the safe area), h [\"8%\"] (px or % of "
            "the height), w, x, y, opacity, dark (true takes the logo's onDark file; default: from the background), "
            "src (a file instead of a brand logo)",
    "sting": "the brand's intro (its \"intro\" part): logo [full, else the first] or the brand's name, tagline, h "
             "[\"22%\"], enter [pop], x, y [\"50%\"]. \"intro\": true on a scene renders it before the video",
    "endCard": "the brand's outro (its \"outro\" part): logo, title, subtitle, url, h [\"16%\"], x, y. \"outro\": "
               "true on a scene renders it after the video",
    "bug": "the brand's bug (its logo [mark], placement and opacity [0.85]) for the whole video unless timed; "
           "takes logo's parameters",
}
_PASS = ("id", "at", "dur", "until", "enter", "exit", "emphasis", "opacity", "scale", "rotate", "hidden", "z",
         "rotateX", "rotateY", "fixed", "sound")

# The parameters a brand's "components" part may set for each built-in: looks, never layout or timing.
STYLE_PARAMS = {"lowerThird": ("accent", "plate", "color", "subColor"), "titleCard": ("accent", "color"),
                "callout": ("accent", "boxColor"), "focus": ("accent",), "progress": ("color",),
                "kinetic": ("color", "motion"), "barChart": ("color", "accent"), "lineChart": ("color",),
                "pieChart": ("colors",)}
# The brand's "charts" entry each chart parameter defaults to.
CHART_PARAMS = {"barChart": {"color": "color", "accent": "highlight"}, "lineChart": {"color": "highlight"},
                "pieChart": {"colors": "series"}}
# Each built-in's text parts (layer id after the instance id, digits dropped) and the text style whose font they take
# in a brand. A brand's own text style, or "<part>Text" in its components, sets their weight, tracking and case too.
TEXT_PARTS = {"lowerThird": {"title": "heading", "sub": "body"},
              "titleCard": {"kicker": "caption", "title": "title", "sub": "body"},
              "callout": {"label": "body"}, "kinetic": {"w": "title"},
              "barChart": {"value": "heading", "label": "caption"}, "lineChart": {"value": "heading", "label": "caption"},
              "pieChart": {"label": "caption"}}


def _themed(v, theme: dict):
    """A brand value with palette role names ("surface") and "$name" read from the theme."""
    if isinstance(v, str):
        name = v[1:] if v.startswith("$") else v
        return theme.get(name, v) if name in theme and (v.startswith("$") or not v.startswith("#")) else v
    if isinstance(v, list):
        return [_themed(x, theme) for x in v]
    return v


def _brand_params(name: str, params: dict, theme: dict) -> None:
    """Fill the built-in's look parameters the instance leaves out from the brand's components and charts parts."""
    own = theme.get(f"components.{name}") or {}
    charts = group(theme, "charts")
    for k in STYLE_PARAMS.get(name, ()):
        if params.get(k) is not None:
            continue
        if k in own:
            params[k] = _themed(own[k], theme)
        elif CHART_PARAMS.get(name, {}).get(k) in charts:
            params[k] = _themed(charts[CHART_PARAMS[name][k]], theme)
    if name == "lowerThird" and params.get("plate") is None and re.fullmatch(r"#[0-9a-fA-F]{6}", theme["bg"]):
        params["plate"] = theme["bg"] + "d0"


def _brand_text(name: str, out: dict, pid: str, theme: dict) -> None:
    """Give the built-in's text parts their brand text style: the font always; weight, tracking and case when the
    brand sets that style or names one for the part."""
    parts, own = TEXT_PARTS.get(name, {}), theme.get(f"components.{name}") or {}
    brand_styles = group(theme, "text")

    def walk(kids):
        for c in kids or []:
            if c.get("type") in ("text", "counter") and str(c.get("id", "")).startswith(pid + "-"):
                part = re.sub(r"\d+$", "", c["id"][len(pid) + 1:])
                chosen = own.get(part + "Text")
                style = text_style(chosen or parts.get(part), theme)
                if style:
                    full = chosen is not None or parts.get(part) in brand_styles
                    c.update({k: v for k, v in style.items() if k == "font" or (full and k != "size")})
            walk(c.get("children"))

    walk(out.get("children"))


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
        return _sequence(raw, scene)
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
        branded = scene is not None and getattr(scene, "brand", None) is not None
        if branded:
            _brand_params(name, params, theme)
        try:
            out = BUILTIN[name](params, theme, scene) if name in NEEDS_SCENE else BUILTIN[name](params, theme)
            if branded:
                _brand_text(name, out, params["id"], theme)
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
_ENTERS = {"fade": "fade", "wipe": "wipe-right", "slide": "left", "zoom": "zoom", "blur": "blur", "iris": "iris",
           "luma": "luma", "glitch": "glitch"}
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


def _joining(kind: str, overlap: float, tr: dict, width: float) -> tuple[dict, dict | None]:
    """A transition's motions: the enter of the clip that comes in, and the exit of the one that goes out, if it has
    one. A push moves both clips a frame width to the left, side by side; a whip does too, faster and blurred across.
    In a glitch the outgoing clip breaks up over the first half, and the incoming one comes in broken up and settles."""
    if kind in ("push", "whip"):
        smear = round(width * 0.04, 2)
        move = ({"name": "left", "dur": overlap, "ease": "inOutCubic", "fade": False} if kind == "push" else
                {"name": "whip", "dur": overlap, "ease": "inOutQuart", "fade": False, "amount": smear})
        # Side by side, overlapping so the background does not show between their antialiased edges: by 2 px, which
        # holds down to half size, and a whip by 4, since its blur stays inside each clip from a pixel in (fade false).
        dist = width - (2 if kind == "push" else 4)
        return dict(move, dist=dist), dict(move, dist=-dist)
    enter = {"name": _ENTERS.get(kind, "fade"), "dur": overlap, "ease": "inOutSine", "dist": 120}
    if kind == "luma":
        enter.update({k: tr[k] for k in ("map", "soft") if tr.get(k) is not None})
    if kind == "glitch":
        enter = {"name": "glitch", "dur": overlap, "ease": "inQuart"}
        return enter, dict(enter, fade=False, seed=1)
    return enter, None


def _sequence(raw: dict, scene=None) -> dict:
    """Cut list -> a group of back-to-back video layers, timed within the sequence. The group has the sequence's
    id and ends with its last clip, so "cut.out" works as a cue. Its other layer keys (opacity, blur, fx, enter, ...)
    apply to the whole cut; fit, volume, w, h, x, y and anchor go to each clip.
    "transition": {"type": "fade", "dur": 0.4} overlaps the clips: the incoming clip enters over the outgoing one,
    and their sound crossfades over the same time (the outgoing clip's fadeOut, the incoming one's fadeIn); push, whip
    and glitch move the outgoing clip too (_joining). A cut never overlaps. The group's "sequence" key records each
    join it timed, so the scene can check that every transition is shorter than the clips it joins."""
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
    from .scene import _num
    base = raw.get("id", "seq")
    width = float(scene.w if scene is not None else 1920)
    try:
        width = _num(raw.get("w"), width)
    except (TypeError, ValueError):
        pass  # the scene notes a w it cannot read
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
                enter, leave = _joining(kind, overlap, tr, width)
                c.setdefault("enter", enter)
                if leave:
                    prev.setdefault("exit", leave)
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
