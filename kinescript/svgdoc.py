"""An svg file read as layers. The svg layer becomes a group of path layers, one for each shape in the file, with
the file's transforms, fills, strokes, opacity and gradients, named from the file's ids, so each piece draws on,
enters and moves as any layer does. Also the size an svg is drawn at, for image layers too."""
from __future__ import annotations

import colorsys
import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import skia

from . import svgpath

SVG_NS = "{http://www.w3.org/2000/svg}"
_LENGTH = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(px|pt|pc|in|cm|mm|em|ex|%)?\s*", re.I)
_PX = {"px": 1, "pt": 4 / 3, "pc": 16, "in": 96, "cm": 96 / 2.54, "mm": 96 / 25.4, "em": 16, "ex": 8}
_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_CSS = """
aliceblue:f0f8ff antiquewhite:faebd7 aqua:00ffff aquamarine:7fffd4 azure:f0ffff beige:f5f5dc bisque:ffe4c4
black:000000 blanchedalmond:ffebcd blue:0000ff blueviolet:8a2be2 brown:a52a2a burlywood:deb887 cadetblue:5f9ea0
chartreuse:7fff00 chocolate:d2691e coral:ff7f50 cornflowerblue:6495ed cornsilk:fff8dc crimson:dc143c cyan:00ffff
darkblue:00008b darkcyan:008b8b darkgoldenrod:b8860b darkgray:a9a9a9 darkgreen:006400 darkgrey:a9a9a9
darkkhaki:bdb76b darkmagenta:8b008b darkolivegreen:556b2f darkorange:ff8c00 darkorchid:9932cc darkred:8b0000
darksalmon:e9967a darkseagreen:8fbc8f darkslateblue:483d8b darkslategray:2f4f4f darkslategrey:2f4f4f
darkturquoise:00ced1 darkviolet:9400d3 deeppink:ff1493 deepskyblue:00bfff dimgray:696969 dimgrey:696969
dodgerblue:1e90ff firebrick:b22222 floralwhite:fffaf0 forestgreen:228b22 fuchsia:ff00ff gainsboro:dcdcdc
ghostwhite:f8f8ff gold:ffd700 goldenrod:daa520 gray:808080 green:008000 greenyellow:adff2f grey:808080
honeydew:f0fff0 hotpink:ff69b4 indianred:cd5c5c indigo:4b0082 ivory:fffff0 khaki:f0e68c lavender:e6e6fa
lavenderblush:fff0f5 lawngreen:7cfc00 lemonchiffon:fffacd lightblue:add8e6 lightcoral:f08080 lightcyan:e0ffff
lightgoldenrodyellow:fafad2 lightgray:d3d3d3 lightgreen:90ee90 lightgrey:d3d3d3 lightpink:ffb6c1
lightsalmon:ffa07a lightseagreen:20b2aa lightskyblue:87cefa lightslategray:778899 lightslategrey:778899
lightsteelblue:b0c4de lightyellow:ffffe0 lime:00ff00 limegreen:32cd32 linen:faf0e6 magenta:ff00ff maroon:800000
mediumaquamarine:66cdaa mediumblue:0000cd mediumorchid:ba55d3 mediumpurple:9370db mediumseagreen:3cb371
mediumslateblue:7b68ee mediumspringgreen:00fa9a mediumturquoise:48d1cc mediumvioletred:c71585
midnightblue:191970 mintcream:f5fffa mistyrose:ffe4e1 moccasin:ffe4b5 navajowhite:ffdead navy:000080
oldlace:fdf5e6 olive:808000 olivedrab:6b8e23 orange:ffa500 orangered:ff4500 orchid:da70d6 palegoldenrod:eee8aa
palegreen:98fb98 paleturquoise:afeeee palevioletred:db7093 papayawhip:ffefd5 peachpuff:ffdab9 peru:cd853f
pink:ffc0cb plum:dda0dd powderblue:b0e0e6 purple:800080 rebeccapurple:663399 red:ff0000 rosybrown:bc8f8f
royalblue:4169e1 saddlebrown:8b4513 salmon:fa8072 sandybrown:f4a460 seagreen:2e8b57 seashell:fff5ee
sienna:a0522d silver:c0c0c0 skyblue:87ceeb slateblue:6a5acd slategray:708090 slategrey:708090 snow:fffafa
springgreen:00ff7f steelblue:4682b4 tan:d2b48c teal:008080 thistle:d8bfd8 tomato:ff6347 turquoise:40e0d0
violet:ee82ee wheat:f5deb3 white:ffffff whitesmoke:f5f5f5 yellow:ffff00 yellowgreen:9acd32
"""
CSS_COLOURS = dict(x.split(":") for x in _CSS.split())  # SVG's names are CSS's: red is #ff0000
_TURN = {"deg": 1.0, "grad": 0.9, "rad": 180 / math.pi, "turn": 360.0}  # an hsl hue's units, in degrees

# What the svg layer leaves out, by kind, with the fix for each: elements it finds anywhere in the file, and
# attributes and paints on the shapes it draws.
SKIPPED = {"text": "turn text into paths (Inkscape: Path > Object to Path; Illustrator: Type > Create Outlines)",
           "style": "write fill and stroke on the elements instead of in a <style> sheet (Illustrator: Styling: "
                    "Presentation Attributes)",
           "image": "show a picture in an image layer of its own",
           "clip-path": "an image layer with the same src draws clip paths, masks, filters and patterns",
           "mask": "an image layer with the same src draws clip paths, masks, filters and patterns",
           "filter": "an image layer with the same src draws clip paths, masks, filters and patterns",
           "pattern": "an image layer with the same src draws clip paths, masks, filters and patterns",
           "marker": "turn the stroke and its markers into paths (Inkscape: Path > Stroke to Path)",
           "animation": "an svg's own animation does not play: animate the pieces instead",
           "other": "draw it with paths and shapes"}
_ELEMENT_KIND = {"text": "text", "tspan": "text", "textPath": "text", "style": "style", "image": "image",
                 "foreignObject": "other", "animate": "animation", "animateTransform": "animation",
                 "animateMotion": "animation", "set": "animation"}
_SHOWN = {"tspan": "<text>", "textPath": "<text>", "animateTransform": "<animate>", "animateMotion": "<animate>",
          "set": "<animate>"}
_QUIET = {"defs", "title", "desc", "metadata", "linearGradient", "radialGradient", "stop", "clipPath", "mask",
          "filter", "pattern", "marker", "symbol", "style", "script", "font", "font-face", "cursor", "view"}
_INHERITED = ("fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity", "stroke-linecap",
              "stroke-linejoin", "stroke-dasharray", "stroke-dashoffset", "color", "visibility")
_SHAPES = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
CAP, JOIN = "round", "round"  # a path layer's own, which the svg's butt and miter replace


def length(v) -> float | None:
    """An svg width or height in px (96 to the inch), or None for one missing, a percentage or not above 0."""
    m = _LENGTH.fullmatch(v or "")
    if m is None or m.group(2) == "%":
        return None
    x = float(m.group(1)) * _PX[(m.group(2) or "px").lower()]
    return x if 0 < x < math.inf else None


def view_box(el: ET.Element) -> list[float] | None:
    try:
        vb = [float(x) for x in re.split(r"[\s,]+", (el.get("viewBox") or "").strip())]
    except ValueError:
        return None
    return vb if len(vb) == 4 and vb[2] > 0 and vb[3] > 0 else None


def size(root: ET.Element) -> tuple[float, float]:
    """The size an svg is drawn at, as a browser shows it in an <img>: its width and height; the viewBox's shape for
    a missing one; the viewBox's size for both; 300 x 150 for what is still missing."""
    w, h = length(root.get("width")), length(root.get("height"))
    vb = view_box(root)
    if not (w and h) and vb:
        aspect = vb[2] / vb[3]
        return (w, w / aspect) if w else (h * aspect, h) if h else (vb[2], vb[3])
    return w or 300.0, h or 150.0


def local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _svg_element(el: ET.Element) -> bool:
    """An element of svg's own: in its namespace, or in none (a file without xmlns); not an editor's (Inkscape's
    sodipodi:namedview, say)."""
    return isinstance(el.tag, str) and (el.tag.startswith(SVG_NS) or not el.tag.startswith("{"))


def _matrix(a, b, c, d, e, f) -> skia.Matrix:
    return skia.Matrix.MakeAll(a, c, e, b, d, f, 0, 0, 1)


def transform(v: str | None) -> skia.Matrix:
    """An svg transform list: translate, scale, rotate (about a point too), skewX, skewY and matrix, applied in
    turn as written."""
    m = skia.Matrix()
    for name, args in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", v or ""):
        n = [float(x) for x in _NUM.findall(args)]
        if name == "matrix" and len(n) == 6:
            t = _matrix(*n)
        elif name == "translate" and n:
            t = skia.Matrix.Translate(n[0], n[1] if len(n) > 1 else 0)
        elif name == "scale" and n:
            t = skia.Matrix.Scale(n[0], n[1] if len(n) > 1 else n[0])
        elif name == "rotate" and n:
            t = skia.Matrix.RotateDeg(n[0], skia.Point(n[1], n[2])) if len(n) == 3 else skia.Matrix.RotateDeg(n[0])
        elif name == "skewX" and n:
            t = _matrix(1, 0, math.tan(math.radians(n[0])), 1, 0, 0)
        elif name == "skewY" and n:
            t = _matrix(1, math.tan(math.radians(n[0])), 0, 1, 0, 0)
        else:
            continue
        m = skia.Matrix.Concat(m, t)
    return m


def fit(vb: list[float], w: float, h: float, par: str | None) -> tuple[skia.Matrix, bool]:
    """The matrix taking viewBox vb into 0, 0, w, h by preserveAspectRatio par, and whether it overflows (slice)."""
    words = (par or "").split()
    align = words[0] if words else "xMidYMid"
    sx, sy = w / vb[2], h / vb[3]
    if align == "none":
        return skia.Matrix.Concat(skia.Matrix.Scale(sx, sy), skia.Matrix.Translate(-vb[0], -vb[1])), False
    cut = len(words) > 1 and words[1] == "slice"
    s = max(sx, sy) if cut else min(sx, sy)
    where = {"Min": 0.0, "Mid": 0.5, "Max": 1.0}
    ax, ay = where.get(align[1:4], 0.5), where.get(align[5:8], 0.5)
    tx, ty = (w - vb[2] * s) * ax - vb[0] * s, (h - vb[3] * s) * ay - vb[1] * s
    return _matrix(s, 0, 0, s, tx, ty), cut


def colour(v: str | None, current: str | None, alpha: float = 1.0) -> str | None:
    """An svg colour as #rrggbb, or #rrggbbaa below full opacity; None for none, transparent or one not read."""
    v = (v or "").strip()
    if v == "currentColor":
        v = (current or "black").strip()
    low = v.lower()
    rgb: list[float] | None = None
    if low in CSS_COLOURS:
        rgb = [int(CSS_COLOURS[low][k:k + 2], 16) / 255 for k in (0, 2, 4)]
    elif re.fullmatch(r"#[0-9a-fA-F]{3,4}|#[0-9a-fA-F]{6}|#[0-9a-fA-F]{8}", v):
        hx = v[1:] if len(v) > 5 else "".join(ch * 2 for ch in v[1:])
        rgb = [int(hx[k:k + 2], 16) / 255 for k in (0, 2, 4)]
        if len(hx) == 8:
            alpha *= int(hx[6:8], 16) / 255
    elif (m := re.fullmatch(r"(rgba?|hsla?)\s*\((.*)\)", low)) is not None:
        parts = [p for p in re.split(r"[\s,/]+", m.group(2).strip()) if p]
        hue = re.fullmatch(r"(.+?)(deg|grad|rad|turn)", parts[0]) if parts and m.group(1)[0] == "h" else None
        try:
            if hue:
                parts[0] = repr(float(hue.group(1)) * _TURN[hue.group(2)])
            nums = [float(p.rstrip("%")) / (100 if p.endswith("%") else 1) for p in parts]
        except ValueError:
            return None
        if len(nums) < 3:
            return None
        if m.group(1).startswith("rgb"):
            rgb = [n if p.endswith("%") else n / 255 for n, p in zip(nums[:3], parts)]
        else:
            rgb = list(colorsys.hls_to_rgb(nums[0] / 360 % 1, nums[2], nums[1]))
        if len(nums) > 3:
            alpha *= nums[3]
    if rgb is None:
        return None
    out = "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in rgb)
    a = round(min(1.0, max(0.0, alpha)) * 255)
    return out if a >= 255 else out + f"{a:02x}"


def _n(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path_data(p: skia.Path) -> str:
    """skia path p as svg path data: M, L, Q, C and Z, conics as quads."""
    out = []
    it = skia.Path.Iter(p, False)
    V = skia.Path.Verb
    while True:
        verb, pts = it.next()
        if verb == V.kDone_Verb:
            break
        if verb == V.kMove_Verb:
            out.append(f"M{_n(pts[0].x())} {_n(pts[0].y())}")
        elif verb == V.kLine_Verb:
            out.append(f"L{_n(pts[1].x())} {_n(pts[1].y())}")
        elif verb == V.kQuad_Verb:
            out.append(f"Q{_n(pts[1].x())} {_n(pts[1].y())} {_n(pts[2].x())} {_n(pts[2].y())}")
        elif verb == V.kConic_Verb:
            q = skia.Path.ConvertConicToQuads(pts[0], pts[1], pts[2], it.conicWeight(), 2)
            out += [f"Q{_n(q[k].x())} {_n(q[k].y())} {_n(q[k + 1].x())} {_n(q[k + 1].y())}"
                    for k in range(1, len(q) - 1, 2)]
        elif verb == V.kCubic_Verb:
            out.append("C" + " ".join(f"{_n(pt.x())} {_n(pt.y())}" for pt in pts[1:4]))
        elif verb == V.kClose_Verb:
            out.append("Z")
    return " ".join(out)


def _num(v, default: float = 0.0, ref: float = 1.0) -> float:
    """An svg number or length; a percentage of ref."""
    m = _LENGTH.fullmatch(v or "")
    if m is None:
        return default
    x = float(m.group(1))
    unit = (m.group(2) or "px").lower()
    return x / 100 * ref if unit == "%" else x * _PX[unit]


class Doc:
    """One svg file being turned into layers for the svg layer `lid`."""

    def __init__(self, root: ET.Element, lid: str):
        self.root, self.lid = root, lid
        self.ids = {el.get("id"): el for el in root.iter() if el.get("id")}
        vb = view_box(root)
        self.vw, self.vh = (vb[2], vb[3]) if vb else size(root)  # what percentages are of, in its user units
        self.skipped: dict[str, list[str]] = {}  # kind -> the names shown for it
        self.pieces: dict[str, dict] = {}  # the file's id -> its layer
        self.named: set[str] = set()
        self.unnamed = 0
        for el in root.iter():
            name = local(el.tag)
            if name in _ELEMENT_KIND and _svg_element(el):
                self.skip(_ELEMENT_KIND[name], _SHOWN.get(name, f"<{name}>"))

    def skip(self, kind: str, shown: str | None = None):
        names = self.skipped.setdefault(kind, [])
        if (shown or kind) not in names:
            names.append(shown or kind)

    def name(self, sid: str | None) -> str:
        """A layer id: the svg layer's id and the file's id, or a number for a piece without one; unique."""
        base = f"{self.lid}-{re.sub(r'[^A-Za-z0-9_-]', '-', sid)}" if sid else None
        if base is None:
            self.unnamed += 1
            base = f"{self.lid}-{self.unnamed}"
        out, k = base, 2
        while out in self.named:
            out, k = f"{base}-{k}", k + 1
        self.named.add(out)
        return out

    def style(self, el: ET.Element, parent: dict) -> dict:
        """el's presentation: its attributes, then its style attribute, over what it inherits from parent."""
        own = {k: el.get(k) for k in _INHERITED + ("opacity", "display") if el.get(k) is not None}
        for decl in (el.get("style") or "").split(";"):
            k, _, v = decl.partition(":")
            if v.strip():
                own[k.strip()] = v.strip().removesuffix("!important").strip()
        st = {k: v for k, v in parent.items() if k in _INHERITED}
        st.update({k: v.strip() for k, v in own.items() if v.strip() != "inherit"})
        return st

    def walk(self, el: ET.Element, m: skia.Matrix, parent: dict, out: list, named: bool = True, depth: int = 0):
        """el's layers appended to out as (layer, bounds) in the svg layer's space; m takes el's user space there."""
        name = local(el.tag)
        if not _svg_element(el) or name in _QUIET or name in _ELEMENT_KIND or depth > 32:
            return
        st = self.style(el, parent)
        if st.get("display") == "none":
            return
        m = skia.Matrix.Concat(m, transform(el.get("transform")))
        for k in ("clip-path", "mask", "filter"):
            if (el.get(k) or st.get(k) or "none") != "none":
                self.skip(k)
        if any(el.get(k) for k in ("marker-start", "marker-mid", "marker-end", "marker")):
            self.skip("marker")
        sid = el.get("id") if named else None
        if name in ("g", "a", "switch", "svg", "use"):
            kids: list = []
            if name == "use":
                ref = self.ids.get((el.get("href") or el.get("{http://www.w3.org/1999/xlink}href") or "")[1:])
                if ref is None:
                    return
                m = skia.Matrix.Concat(m, skia.Matrix.Translate(_num(el.get("x")), _num(el.get("y"))))
                if local(ref.tag) == "symbol":  # a viewport of its own, as a nested svg
                    vb = view_box(ref)
                    if vb:
                        w, h = length(el.get("width")) or vb[2], length(el.get("height")) or vb[3]
                        m = skia.Matrix.Concat(m, fit(vb, w, h, ref.get("preserveAspectRatio"))[0])
                    for k in ref:
                        self.walk(k, m, self.style(ref, st), kids, False, depth + 1)
                else:
                    self.walk(ref, m, st, kids, False, depth + 1)
            else:
                if name == "svg":
                    m = skia.Matrix.Concat(m, skia.Matrix.Translate(_num(el.get("x")), _num(el.get("y"))))
                    vb = view_box(el)
                    if vb:
                        w, h = length(el.get("width")) or vb[2], length(el.get("height")) or vb[3]
                        m = skia.Matrix.Concat(m, fit(vb, w, h, el.get("preserveAspectRatio"))[0])
                for k in el if name != "switch" else [k for k in el if _svg_element(k)][:1]:
                    self.walk(k, m, st, kids, named, depth + 1)
            if not kids:
                return
            opacity = min(1.0, max(0.0, _num(st.get("opacity"), 1.0)))
            if not sid and opacity >= 1:
                out += kids
                return
            if len(kids) == 1 and opacity >= 1 and name == "use":  # a use of one shape: the use's id names it
                layer, b = kids[0]
                self._rename(layer, sid)
                out.append((layer, b))
                return
            b = skia.Rect.MakeLTRB(*kids[0][1])
            for _, kb in kids[1:]:
                b.join(kb)
            cx, cy = b.centerX(), b.centerY()
            for layer, _ in kids:
                layer["x"], layer["y"] = _r(layer["x"] - cx), _r(layer["y"] - cy)
            group = {"id": self.name(sid), "type": "group", "x": _r(cx), "y": _r(cy),
                     "children": [layer for layer, _ in kids]}
            if opacity < 1:
                group["opacity"] = _r(opacity)
            if sid:
                self.pieces[sid] = group
            out.append((group, b))
            return
        if name not in _SHAPES:
            self.skip("other", f"<{name}>")
            return
        p = self.shape(el, name)
        if p is None or st.get("visibility") in ("hidden", "collapse"):
            return
        piece = self.piece(p, m, st)
        if piece is None:
            return
        layer, b = piece
        layer["id"] = self.name(sid)
        if sid:
            self.pieces[sid] = layer
        out.append((layer, b))

    def _rename(self, layer: dict, sid: str | None):
        if sid:
            self.named.discard(layer["id"])
            layer["id"] = self.name(sid)
            self.pieces[sid] = layer

    def shape(self, el: ET.Element, name: str) -> skia.Path | None:
        """A shape element's outline in its own user space."""
        g = lambda k, ref=1.0: _num(el.get(k), 0.0, ref)  # noqa: E731
        p = skia.Path()
        if name == "path":
            try:
                p = svgpath.parse(el.get("d") or "")
            except (ValueError, IndexError):
                return None
        elif name == "rect":
            x, y, w, h = g("x", self.vw), g("y", self.vh), g("width", self.vw), g("height", self.vh)
            if w <= 0 or h <= 0:
                return None
            rx, ry = el.get("rx"), el.get("ry")
            rx, ry = _num(rx if rx is not None else ry, 0.0, self.vw), _num(ry if ry is not None else rx, 0.0, self.vh)
            rx, ry = min(max(rx, 0), w / 2), min(max(ry, 0), h / 2)
            if rx > 0 and ry > 0:
                p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), rx, ry))
            else:
                p.addRect(skia.Rect.MakeXYWH(x, y, w, h))
        elif name in ("circle", "ellipse"):
            cx, cy = g("cx", self.vw), g("cy", self.vh)
            rx = g("r", math.hypot(self.vw, self.vh) / math.sqrt(2)) if name == "circle" else g("rx", self.vw)
            ry = rx if name == "circle" else g("ry", self.vh)
            if rx <= 0 or ry <= 0:
                return None
            p.addOval(skia.Rect(cx - rx, cy - ry, cx + rx, cy + ry))
        elif name == "line":
            p.moveTo(g("x1", self.vw), g("y1", self.vh))
            p.lineTo(g("x2", self.vw), g("y2", self.vh))
        else:
            n = [float(x) for x in _NUM.findall(el.get("points") or "")]
            if len(n) < 4:
                return None
            p.moveTo(n[0], n[1])
            for k in range(2, len(n) - 1, 2):
                p.lineTo(n[k], n[k + 1])
            if name == "polygon":
                p.close()
        return p if p.countVerbs() else None

    def piece(self, p: skia.Path, m: skia.Matrix, st: dict) -> tuple[dict, skia.Rect] | None:
        """A path layer for outline p (in user space, taken to the layer's by m) painted by st, centred on its
        middle so it turns and scales about it, with its bounds; None when it draws nothing."""
        user = p.computeTightBounds()
        q = skia.Path(p)
        q.transform(m)
        b = q.computeTightBounds()
        cx, cy = b.centerX(), b.centerY()
        q.offset(-cx, -cy)
        box = skia.Rect.MakeXYWH(b.left() - cx, b.top() - cy, b.width(), b.height())
        cur = st.get("color")
        fill = self.paint(st.get("fill", "black"), _num(st.get("fill-opacity"), 1.0), cur, user, m, cx, cy, box)
        scale = math.sqrt(abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY()))
        width = _num(st.get("stroke-width"), 1.0, math.hypot(self.vw, self.vh) / math.sqrt(2)) * scale
        stroke = self.paint(st.get("stroke", "none"), _num(st.get("stroke-opacity"), 1.0), cur, user, m, cx, cy, box)
        if width <= 0:
            stroke = None
        if fill is None and stroke is None:
            return None
        layer = {"id": "", "type": "path", "x": _r(cx), "y": _r(cy), "d": path_data(q)}
        if fill is not None:
            layer["fill"] = fill
            if st.get("fill-rule") == "evenodd":
                layer["fillRule"] = "evenodd"
        if stroke is not None:
            layer["stroke"], layer["strokeWidth"] = stroke, _r(width)
            cap, join = st.get("stroke-linecap", "butt"), st.get("stroke-linejoin", "miter")
            if cap != CAP and cap in ("butt", "round", "square"):
                layer["cap"] = cap
            if join != JOIN and join in ("miter", "round", "bevel"):
                layer["join"] = join
            dash = [_num(x) * scale for x in re.split(r"[\s,]+", st.get("stroke-dasharray", "none").strip())
                    if _NUM.fullmatch(x.rstrip("px"))]
            if dash and sum(dash) > 0 and min(dash) >= 0:
                layer["dash"] = [_r(x) for x in (dash * 2 if len(dash) % 2 else dash)]
                off = _num(st.get("stroke-dashoffset")) * scale
                if off:
                    layer["dashOffset"] = _r(off)
        opacity = min(1.0, max(0.0, _num(st.get("opacity"), 1.0)))
        if opacity < 1:
            layer["opacity"] = _r(opacity)
        if b.isEmpty():  # a straight line: its stroke's width keeps it a box
            b = b.makeOutset(width / 2, width / 2)
        return layer, b

    def paint(self, v: str | None, alpha: float, cur, user: skia.Rect, m: skia.Matrix, cx: float, cy: float,
              box: skia.Rect):
        """A fill or stroke: a colour, a gradient over the piece's box, or None."""
        v = (v or "").strip()
        ref = re.match(r"url\(\s*['\"]?#([^)'\"]+)['\"]?\s*\)\s*(.*)", v)
        if ref is None:
            return colour(v, cur, alpha)
        grad = self.ids.get(ref.group(1))
        if grad is None or local(grad.tag) not in ("linearGradient", "radialGradient"):
            if grad is not None:
                self.skip("pattern")
            return colour(ref.group(2), cur, alpha) if ref.group(2) else None
        return self.gradient(grad, alpha, user, m, cx, cy, box)

    def gradient(self, grad: ET.Element, alpha: float, user: skia.Rect, m: skia.Matrix, cx: float, cy: float,
                 box: skia.Rect):
        """An svg gradient as a layer gradient: its points taken from the file's units to the piece's box, 0 to 1."""
        chain, el = [], grad
        while el is not None and el not in chain and len(chain) < 16:
            chain.append(el)
            href = el.get("href") or el.get("{http://www.w3.org/1999/xlink}href") or ""
            el = self.ids.get(href[1:]) if href.startswith("#") else None
        attr = lambda k: next((e.get(k) for e in chain if e.get(k) is not None), None)  # noqa: E731
        stops, last = [], 0.0
        for e in next(([s for s in e if local(s.tag) == "stop"] for e in chain
                       if any(local(s.tag) == "stop" for s in e)), []):
            st = {k: e.get(k) for k in ("stop-color", "stop-opacity", "color") if e.get(k) is not None}
            for decl in (e.get("style") or "").split(";"):
                k, _, val = decl.partition(":")
                if val.strip():
                    st[k.strip()] = val.strip()
            off = (e.get("offset") or "0").strip()
            o = float(off.rstrip("%")) / (100 if off.endswith("%") else 1) if _NUM.fullmatch(off.rstrip("%")) else 0.0
            last = max(last, min(1.0, max(0.0, o)))  # offsets never go back
            c = colour(st.get("stop-color", "black"), st.get("color"), alpha * _num(st.get("stop-opacity"), 1.0))
            stops.append([_r(last), c or "#00000000"])
        if not stops:
            return None
        if len(stops) == 1 or box.width() < 1e-6 or box.height() < 1e-6:
            return stops[-1][1] if len(stops) == 1 else stops[0][1]
        obb = attr("gradientUnits") != "userSpaceOnUse"
        to = transform(attr("gradientTransform"))
        if obb:  # 0..1 of the shape's own box
            to = skia.Matrix.Concat(_matrix(user.width(), 0, 0, user.height(), user.left(), user.top()), to)
        to = skia.Matrix.Concat(skia.Matrix.Concat(skia.Matrix.Translate(-cx, -cy), m), to)
        rw, rh = (1.0, 1.0) if obb else (self.vw, self.vh)

        def frac(x: float, y: float) -> list[float]:
            pt = to.mapXY(x, y)
            return [_r((pt.x() - box.left()) / box.width(), 4), _r((pt.y() - box.top()) / box.height(), 4)]
        if local(grad.tag) == "linearGradient":
            a = frac(_num(attr("x1"), 0.0, rw), _num(attr("y1"), 0.0, rh))
            b = frac(_num(attr("x2"), rw, rw), _num(attr("y2"), 0.0, rh))
            return {"linear": [a, b], "stops": stops}
        gx, gy = _num(attr("cx"), 0.5 * rw, rw), _num(attr("cy"), 0.5 * rh, rh)
        r = _num(attr("r"), 0.5 * (rw if obb else math.hypot(rw, rh) / math.sqrt(2)),
                 rw if obb else math.hypot(rw, rh) / math.sqrt(2))
        c, edge = to.mapXY(gx, gy), to.mapXY(gx + r, gy)
        across = math.hypot(edge.x() - c.x(), edge.y() - c.y()) / max(box.width(), box.height())
        return {"radial": frac(gx, gy) + [_r(across, 4)], "stops": stops}


def _r(x: float, places: int = 3) -> float:
    v = round(float(x), places)
    return 0.0 if v == 0 else v


def read(src: str | Path) -> ET.Element:
    """The root of svg file src; ValueError naming the file for one that cannot be read or is not an svg."""
    try:
        root = ET.fromstring(Path(src).read_bytes())
    except ET.ParseError as e:
        raise ValueError(f"cannot read svg {src}: {e}") from None
    if local(root.tag) != "svg":
        raise ValueError(f"cannot read svg {src}: it is <{local(root.tag)}>, not <svg>")
    return root


def layer(raw: dict, src: str | Path) -> tuple[dict, dict[str, list[str]], list[str]]:
    """The svg layer raw, reading src, as a group of path layers: (group, skipped, unknown parts). Its box is its w
    and h (one keeps the file's shape; neither, the file's own size), anchored as an image's is (at its centre once
    it has an x); the file's viewBox fits into it by its preserveAspectRatio. `skipped` is what was left out, kind
    -> the names shown for it, in SKIPPED's order; unknown parts are parts keys that name no element of the file.
    ValueError for a file that is not an svg."""
    root = read(src)
    lid = str(raw.get("id"))
    doc = Doc(root, lid)
    iw, ih = doc.vw, doc.vh
    w, h = raw.get("w"), raw.get("h")
    if w is not None and h is None:
        h = w * ih / iw
    elif h is not None and w is None:
        w = h * iw / ih
    w, h = float(w if w is not None else iw), float(h if h is not None else ih)
    anchor = raw.get("anchor") or ("topleft" if raw.get("x") is None else "center")
    from .render import ANCHORS
    ax, ay = ANCHORS.get(anchor, (0.5, 0.5))
    m, cut = fit(view_box(root) or [0.0, 0.0, iw, ih], w, h, root.get("preserveAspectRatio"))
    m = skia.Matrix.Concat(skia.Matrix.Translate(-ax * w, -ay * h), m)
    out: list = []
    st = doc.style(root, {})
    for el in root:
        doc.walk(el, m, st, out)
    kids = [k for k, _ in out]
    parts = raw.get("parts") or {}
    for key, part in parts.items():
        if key == "*":
            for k in kids:
                _every(k, part)
    for key, part in parts.items():
        if key != "*" and key in doc.pieces:
            doc.pieces[key].update(copy.deepcopy(part))
    group = {k: v for k, v in raw.items() if k not in ("type", "src", "w", "h", "parts", "anchor")}
    group.update(type="group", w=_r(w), h=_r(h), anchor=anchor, children=kids)
    if cut and "clip" not in raw:
        group["clip"] = True
    skipped = {k: doc.skipped[k] for k in SKIPPED if k in doc.skipped}
    return group, skipped, [k for k in parts if k != "*" and k not in doc.pieces]


def _every(layer: dict, part: dict):
    """part's keys on every shape under layer."""
    if layer["type"] == "group":
        for k in layer["children"]:
            _every(k, part)
    else:
        layer.update(copy.deepcopy(part))  # each its own, as loading may change a layer's keys
