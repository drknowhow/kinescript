"""Colour parsing and perceptual (OKLab) interpolation."""
from __future__ import annotations

import re
from functools import lru_cache

RGBA = tuple[float, float, float, float]

NAMED = {
    "black": "#000000", "white": "#ffffff", "transparent": "#00000000", "red": "#ff3b30",
    "green": "#34c759", "blue": "#0a84ff", "yellow": "#ffd60a", "orange": "#ff9f0a",
    "gray": "#8e8e93", "grey": "#8e8e93", "purple": "#bf5af2", "pink": "#ff375f", "cyan": "#64d2ff",
}


def is_color(v) -> bool:
    return isinstance(v, str) and (v.startswith("#") or v.startswith("rgb") or v in NAMED)


@lru_cache(maxsize=4096)
def parse(c: str) -> RGBA:
    c = NAMED.get(c, c).strip()
    if c.startswith("#"):
        h = c[1:]
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            h += "ff"
        if len(h) != 8:
            raise ValueError(f"bad colour {c!r}")
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4, 6))  # type: ignore[return-value]
    m = re.fullmatch(r"rgba?\(([^)]*)\)", c)
    if m:
        parts = [float(p) for p in m.group(1).replace("/", ",").split(",") if p.strip()]
        r, g, b = (p / 255 for p in parts[:3])
        a = parts[3] if len(parts) > 3 else 1.0
        return (r, g, b, a)
    raise ValueError(f"bad colour {c!r} (use #rrggbb, #rrggbbaa, rgb(), rgba() or a name)")


def to_argb_int(c: RGBA, alpha_mul: float = 1.0) -> int:
    r, g, b, a = c
    a = max(0.0, min(1.0, a * alpha_mul))
    return (round(a * 255) << 24) | (round(r * 255) << 16) | (round(g * 255) << 8) | round(b * 255)


def _lin(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def _gam(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055


def _to_oklab(c: RGBA):
    r, g, b = (_lin(v) for v in c[:3])
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


def luminance(c: RGBA) -> float:
    r, g, b = (_lin(v) for v in c[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)
