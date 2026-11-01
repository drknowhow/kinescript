"""Colour parsing and perceptual (OKLab) interpolation."""
from __future__ import annotations

import math
import re
from functools import lru_cache

RGBA = tuple[float, float, float, float]

# The names Kinescript knows. They are iOS-style system colours, not the CSS ones: "red" is #ff3b30 and "blue" is
# #0a84ff. Names are read in any case ("Red", "RED").
NAMED = {
    "black": "#000000", "white": "#ffffff", "transparent": "#00000000", "red": "#ff3b30",
    "green": "#34c759", "blue": "#0a84ff", "yellow": "#ffd60a", "orange": "#ff9f0a",
    "gray": "#8e8e93", "grey": "#8e8e93", "purple": "#bf5af2", "pink": "#ff375f", "cyan": "#64d2ff",
}
FORMATS = ("#rrggbb, #rgb, #rrggbbaa (aa is opacity), rgb(255, 99, 63), rgba(255, 99, 63, 0.5), or one of the "
           "names " + ", ".join(NAMED) + " (iOS-style colours: red is #ff3b30, not CSS red)")

_FUNC = re.compile(r"([a-zA-Z]+)\s*\((.*)\)", re.S)


def _read(c: str) -> tuple[RGBA, str | None]:
    """The colour, and a note when a channel was outside its range and was clamped. ValueError says what is wrong."""
    s = c.strip()
    s = NAMED.get(s.lower(), s)
    if s.startswith("#"):
        h = s[1:]
        bad = next((ch for ch in h if ch not in "0123456789abcdefABCDEF"), None)
        if bad is not None:
            raise ValueError(f"'{c}' has '{bad}', which is not a hex digit (0-9, a-f)")
        if len(h) not in (3, 4, 6, 8):
            raise ValueError(f"'{c}' has {len(h)} hex digits; a hex colour has 3, 4, 6 or 8 (#rgb, #rgba, #rrggbb, "
                             "#rrggbbaa)")
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            h += "ff"
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4, 6)), None  # type: ignore[return-value]
    m = _FUNC.fullmatch(s)
    if m and m.group(1).lower() in ("rgb", "rgba"):
        parts = [p for p in re.split(r"[\s,/]+", m.group(2).strip()) if p]
        if len(parts) not in (3, 4):
            raise ValueError(f"'{c}' has {len(parts)} value{'s' * (len(parts) != 1)}; {m.group(1).lower()}() takes "
                             "red, green and blue (0-255) and an optional alpha (0-1)")
        vals = []
        for i, p in enumerate(parts):
            pct = p.endswith("%")
            try:
                v = float(p[:-1] if pct else p)
            except ValueError:
                raise ValueError(f"'{c}': '{p}' is not a number") from None
            if not math.isfinite(v):
                raise ValueError(f"'{c}': '{p}' is not a finite number")
            vals.append(v / 100 * (255 if i < 3 else 1) if pct else v)
        rgb, a = vals[:3], vals[3] if len(vals) == 4 else 1.0
        cr, ca = [max(0.0, min(255.0, v)) for v in rgb], max(0.0, min(1.0, a))
        note = None
        if cr != rgb or ca != a:
            note = (f"'{c}' has a value outside its range (red, green and blue run from 0 to 255, alpha from 0 to 1); "
                    f"it is read as rgba({', '.join(f'{v:g}' for v in cr)}, {ca:g})")
        return (cr[0] / 255, cr[1] / 255, cr[2] / 255, ca), note
    if m:
        raise ValueError(f"'{c}': {m.group(1)}() is not a colour Kinescript reads")
    raise ValueError(f"'{c}' is not a colour name Kinescript knows" if re.fullmatch(r"[A-Za-z]+", s)
                     else f"'{c}' is not a colour")


@lru_cache(maxsize=4096)
def _cached(c: str) -> tuple[RGBA, str | None]:
    return _read(c)


def parse(c) -> RGBA:
    """'#ff633f', 'rgb(255, 99, 63)', 'Red' -> (r, g, b, a) in 0..1. Channels outside their range are clamped.
    ValueError says what is wrong and which forms are read."""
    if not isinstance(c, str):
        raise ValueError(f"{c!r} is not a colour; write {FORMATS}")
    try:
        return _cached(c)[0]
    except ValueError as e:
        raise ValueError(f"{e}. Write {FORMATS}") from None


def problem(v) -> str | None:
    """Why `v` is not a colour, in one sentence without the list of forms; None when it is one."""
    if not isinstance(v, str):
        return f"{v!r} is not a colour"
    try:
        _cached(v)
    except ValueError as e:
        return str(e)
    return None


def clamped(v) -> str | None:
    """A note when colour `v` has a channel outside its range (it is clamped); None otherwise."""
    try:
        return _cached(v)[1] if isinstance(v, str) else None
    except ValueError:
        return None


def is_color(v) -> bool:
    return isinstance(v, str) and problem(v) is None


_ARGB: dict[tuple, int] = {}


def to_argb_int(c: RGBA, alpha_mul: float = 1.0) -> int:
    key = (c, alpha_mul)
    try:
        return _ARGB[key]
    except KeyError:
        pass
    except TypeError:  # a list
        return _argb_int(c, alpha_mul)
    if len(_ARGB) >= 65536:  # colours that animate make a new key on most frames
        _ARGB.clear()
    v = _ARGB[key] = _argb_int(c, alpha_mul)
    return v


def _argb_int(c: RGBA, alpha_mul: float) -> int:
    r, g, b, a = (max(0.0, min(1.0, x)) for x in c)  # a computed colour can overshoot (an outBack ease)
    a = max(0.0, min(1.0, a * alpha_mul))
    return (round(a * 255) << 24) | (round(r * 255) << 16) | (round(g * 255) << 8) | round(b * 255)


def _lin(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _gam(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055


def _to_oklab(c: RGBA):
    r, g, b = (_lin(max(0.0, min(1.0, v))) for v in c[:3])  # a negative channel has a complex cube root
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _from_oklab(L: float, A: float, B: float):
    l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3
    s = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3
    return (_gam(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
            _gam(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
            _gam(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))


def mix(a: RGBA, b: RGBA, t: float) -> RGBA:
    """Interpolate in OKLab so mid-tones stay vivid instead of going muddy."""
    la, lb = _to_oklab(a), _to_oklab(b)
    r, g, bl = _from_oklab(*(x + (y - x) * t for x, y in zip(la, lb)))
    return (r, g, bl, a[3] + (b[3] - a[3]) * t)


class Gradient:
    """A gradient object made into numbers, so one that animates mixes from key to key: its kind (linear, radial,
    sweep, or angle for a linear one given by an angle), its points as shares of the box, its angle in degrees, and
    its stops' positions and colours."""
    __slots__ = ("kind", "geo", "angle", "pos", "cols")

    def __init__(self, kind: str, geo: tuple, angle: float, pos: list[float], cols: list[RGBA]):
        self.kind, self.geo, self.angle, self.pos, self.cols = kind, geo, angle, pos, cols

    @classmethod
    def of(cls, spec: dict) -> "Gradient":
        stops = spec.get("stops") or ["#000", "#fff"]
        if isinstance(stops[0], str):  # spread evenly, as Skia spreads stops without positions
            pos, cols = [k / max(1, len(stops) - 1) for k in range(len(stops))], [parse(s) for s in stops]
        else:
            pos, cols = [float(s[0]) for s in stops], [parse(s[1]) for s in stops]
        angle = spec.get("angle")
        if "radial" in spec:
            kind, geo = "radial", ((spec["radial"] + [0.5, 0.5, 0.6])[:3] if isinstance(spec["radial"], list)
                                    else (0.5, 0.5, 0.6))
        elif "sweep" in spec:
            kind, geo = "sweep", (spec["sweep"] + [0.5, 0.5])[:2] if isinstance(spec["sweep"], list) else (0.5, 0.5)
        elif "linear" not in spec and angle is not None:
            kind, geo = "angle", ()
        else:
            (x1, y1), (x2, y2) = spec.get("linear", [[0, 0], [1, 1]])
            kind, geo = "linear", (x1, y1, x2, y2)
        return cls(kind, tuple(float(v) for v in geo), float(angle or 0), pos, cols)

    def mix(self, other: "Gradient", t: float) -> "Gradient":
        """The gradient t of the way to `other`; one of another kind or stop count is not mixed but cut to at t 1."""
        if other.kind != self.kind or len(other.cols) != len(self.cols):
            return other if t >= 1 else self
        return Gradient(self.kind, tuple(a + (b - a) * t for a, b in zip(self.geo, other.geo)),
                        self.angle + (other.angle - self.angle) * t,
                        [a + (b - a) * t for a, b in zip(self.pos, other.pos)],
                        [mix(a, b, t) for a, b in zip(self.cols, other.cols)])

    def spec(self) -> dict:
        """As a gradient object again, with every stop's position and parsed colour."""
        out = {"stops": [[p, c] for p, c in zip(self.pos, self.cols)], "angle": self.angle}
        if self.kind == "linear":
            out["linear"] = [list(self.geo[:2]), list(self.geo[2:])]
        elif self.kind != "angle":
            out[self.kind] = list(self.geo)
        return out


def luminance(c: RGBA) -> float:
    r, g, b = (_lin(v) for v in c[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)
