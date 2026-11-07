"""A brand palette proposed from a logo: kine brand import.

The logo is drawn by the renderer (so .svg, .png, .jpg and .webp read alike), its opaque pixels are grouped into
colours in OKLab, and the colours are given roles: the most present saturated colour is the accent, a second hue
the accent2, and the neutrals are built around the accent's hue. Each colour is then moved in lightness until it
reads against its palette's background as check asks (text 4.5:1, muted and accents 3:1). It is a proposal: the
wizard shows it and the person applies it, or not.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

from . import brand as B
from . import color as C

SIDE = 256  # px the logo is drawn at to read its colours
NEUTRAL = 0.05  # OKLab chroma below which a colour counts as grey
# Lightness (OKLab L) and chroma of the neutral roles in each palette; their hue is the accent's.
NEUTRALS = {"dark": {"bg": (0.17, 0.012), "surface": (0.27, 0.014), "text": (0.96, 0.006), "muted": (0.72, 0.012)},
            "light": {"bg": (0.965, 0.008), "surface": (0.905, 0.012), "text": (0.2, 0.012), "muted": (0.5, 0.014)}}
NEEDS = {"text": 4.5, "muted": 3.0, "accent": 3.0, "accent2": 3.0}


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(v * 255))):02x}" for v in rgb[:3])


def _lab(hexc: str):
    return C._to_oklab(C.parse(hexc))


def _from(L, a, b) -> str:
    return _hex(C._from_oklab(L, a, b))


def _ratio(x: str, y: str) -> float:
    return C.contrast(C.luminance(C.parse(x)), C.luminance(C.parse(y)))


def logo_colours(path: str | Path, most: int = 8) -> list[dict]:
    """The colours of a logo's opaque pixels, largest share first: [{"hex", "share"}]."""
    import numpy as np
    import skia

    from .output import still
    src = Path(path).resolve()
    if not src.is_file():
        raise ValueError(f"no logo file at {src}")
    if src.suffix.lower() not in B.LOGO_EXTS:
        raise ValueError(f"{src.name}: a logo is one of {', '.join(B.LOGO_EXTS)}")
    scene = {"canvas": {"w": SIDE, "h": SIDE, "fps": 1, "duration": 1, "bg": "transparent"},
             "layers": [{"id": "logo", "type": "image", "src": str(src), "w": SIDE, "h": SIDE, "fit": "contain",
                         "x": "50%", "y": "50%"}]}
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="kine_import_")
    os.close(fd)
    try:
        still(scene, 0, tmp)
        img = skia.Image.open(tmp)
        px = img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
    finally:
        Path(tmp).unlink(missing_ok=True)
    px = px.reshape(-1, 4).astype(np.float64) / 255.0
    px = px[px[:, 3] > 0.5][:, :3]
    if not len(px):
        raise ValueError(f"{src.name} has no opaque pixels to read colours from")
    # bucket in sRGB at 5 bits a channel, then merge buckets that are close in OKLab, largest first
    keys, counts = np.unique(np.round(px * 31).astype(np.int32), axis=0, return_counts=True)
    order = np.argsort(-counts)
    total = float(counts.sum())
    clusters: list[list] = []  # [L, a, b, weight]
    for i in order:
        lab = C._to_oklab(tuple(keys[i] / 31.0))
        w = float(counts[i])
        for c in clusters:
            if math.dist(lab, c[:3]) < 0.07:
                t = c[3] + w
                c[:3] = [(c[j] * c[3] + lab[j] * w) / t for j in range(3)]
                c[3] = t
                break
        else:
            clusters.append([*lab, w])
    clusters.sort(key=lambda c: -c[3])
    return [{"hex": _from(*c[:3]), "share": round(c[3] / total, 3)} for c in clusters if c[3] / total >= 0.01][:most]


def _hue(lab) -> float:
    return math.degrees(math.atan2(lab[2], lab[1])) % 360


def _fit(hexc: str, bg: str, need: float, dark: bool) -> str:
    """The colour, lighter on a dark background or darker on a light one, until it reads at `need`:1."""
    L, a, b = _lab(hexc)
    for _ in range(100):
        if _ratio(hexc, bg) >= need:
            return hexc
        L = min(1.0, L + 0.01) if dark else max(0.0, L - 0.01)
        hexc = _from(L, a, b)
    return hexc


def propose(colours: list[dict], name: str | None = None) -> dict:
    """Palettes (dark and light), chart series and the roles chosen, from a logo's colours."""
    notes = []
    labs = [(c, _lab(c["hex"])) for c in colours]
    chroma = [(c, lab, math.hypot(lab[1], lab[2])) for c, lab in labs]
    vivid = sorted([x for x in chroma if x[2] >= NEUTRAL], key=lambda x: -(x[0]["share"] * (0.4 + x[2] * 4)))
    if vivid:
        accent = vivid[0]
        second = next((x for x in vivid[1:] if abs((_hue(x[1]) - _hue(accent[1]) + 180) % 360 - 180) > 25), None)
    else:
        accent, second = None, None
        notes.append("the logo has no saturated colour; the accent stays the Kinescript one (#ff633f): pick one")
    acc = accent[0]["hex"] if accent else "#ff633f"
    acc2 = second[0]["hex"] if second else None
    if accent and not second:
        notes.append("the logo has one hue; accent2 is left out, so it follows the accent")
    _, ha, hb = _lab(acc)
    hn = math.hypot(ha, hb) or 1.0
    ua, ub = ha / hn, hb / hn  # the accent's hue, to tint the greys
    palettes = {}
    for variant, roles in NEUTRALS.items():
        pal = {r: _from(L, ua * c, ub * c) for r, (L, c) in roles.items()}
        dark = variant == "dark"
        pal["accent"] = _fit(acc, pal["bg"], NEEDS["accent"], dark)
        if acc2:
            pal["accent2"] = _fit(acc2, pal["bg"], NEEDS["accent2"], dark)
        for r in ("text", "muted"):
            pal[r] = _fit(pal[r], pal["bg"], NEEDS[r], dark)
            if r == "text":
                pal[r] = _fit(pal[r], pal["surface"], NEEDS[r], dark)
        palettes[variant] = pal
        if pal["accent"] != acc:
            notes.append(f"{variant}: the accent {acc} is moved to {pal['accent']} to read 3:1 on its background")
    series = ["accent", *(["accent2"] if acc2 else []), "text", "muted"]
    return {"name": name, "colours": colours, "roles": {"accent": acc, "accent2": acc2},
            "palettes": palettes, "charts": {"series": series}, "notes": notes}
