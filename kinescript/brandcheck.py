"""kine check's brand rules: a logo's clear space on every sampled frame, and with --brand-strict, colours and fonts
outside the brand and text outside the brand's safe area."""
from __future__ import annotations

import math

from . import brand as B
from . import color as C

DELTA_E = 3.0  # CIE76 distance below which two colours read as the same


def _lab(rgb) -> tuple[float, float, float]:
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb[:3]]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
    f = [t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116 for t in (x, y, z)]
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def delta_e(a, b) -> float:
    return math.dist(_lab(a), _lab(b))


def safe_share(scene) -> float | None:
    """The brand's safe area as a share of the canvas, or None when it sets none."""
    v = scene.theme.get("safeArea") if scene.brand is not None else None
    if isinstance(v, str) and v.strip().endswith("%"):
        return float(v.strip()[:-1]) / 100
    return v if isinstance(v, (int, float)) and 0 <= v < 0.5 else None


def clear_space(scene, boxes: dict, T: float, add) -> None:
    """Warn when a layer comes into a logo's clear space: its box grown by the brand's clearSpace x its height."""
    W, H = scene.w, scene.h
    for lid, share in scene.logos.items():
        b = boxes.get(lid)
        if not b or b["alpha"] < 0.3:
            continue
        l, t, r, bt = b["rect"]
        pad = share * (bt - t)
        zone = (l - pad, t - pad, r + pad, bt + pad)
        for oid, o in boxes.items():
            if oid == lid or o["alpha"] < 0.3 or o["type"] == "group" or oid in scene.mattes:
                continue
            ol, ot, orr, ob = o["rect"]
            if (orr - ol) * (ob - ot) >= 0.8 * W * H:  # a background
                continue
            if ol <= zone[0] and ot <= zone[1] and orr >= zone[2] and ob >= zone[3]:  # a card the logo sits on
                continue
            if min(orr, zone[2]) > max(ol, zone[0]) and min(ob, zone[3]) > max(ot, zone[1]):
                add("warning", "logo-clear-space", f"{lid}+{oid}", T, f"'{oid}' comes inside the clear space the "
                    f"brand keeps round its logo ({share:g} of the logo's height on each side)",
                    f"move '{oid}' away from the logo, or show them at different times")


def _colours(scene) -> dict[str, tuple]:
    """The brand's colours in the scene's variant: role name -> RGBA."""
    out = {}
    for k in (*B.ROLES, *(f"charts.series[{i}]" for i in range(len(scene.theme.get("charts.series") or [])))):
        v = scene.theme.get(k)
        if k.startswith("charts.series["):
            v = scene.theme["charts.series"][int(k[14:-1])]
            v = scene.theme.get(v, v) if isinstance(v, str) and v in B.ROLES else v
        if isinstance(v, str) and C.problem(v) is None:
            out[k] = C.parse(v)
    return out


def strict(scene, add) -> None:
    """Colours and fonts the brand does not have, once per layer."""
    if scene.brand is None:
        return
    brand = _colours(scene)
    faces = {f.lower() for f in (scene.theme.get("font"), scene.theme.get("fontDisplay"), scene.theme.get("mono"))
             if isinstance(f, str)} | {f.lower() for f in scene.brand.fonts()}
    for L in scene.all_layers():
        raw = L.raw
        found = [(k, raw.get(k)) for k in ("fill", "stroke", "emph")]
        for k in ("box", "shadow"):
            if isinstance(raw.get(k), dict):
                found.append((f"{k}.{'fill' if k == 'box' else 'color'}", raw[k].get("fill" if k == "box" else "color")))
        for k, v in found:
            if not isinstance(v, str) or v.startswith("=") or C.problem(v) is not None or C.parse(v)[3] == 0:
                continue
            c = C.parse(v)
            if k == "shadow.color" and max(c[:3]) < 0.05:  # a black shadow is no brand colour, and none is needed
                continue
            name, d = min(((n, delta_e(c, x)) for n, x in brand.items()), key=lambda p: p[1])
            if d > DELTA_E:
                add("warning", "off-brand-colour", L.id, L.t_in, f"{k} {v} is not a brand colour; the nearest is "
                    f"{name} ({_hex(brand[name])}, ΔE {d:.0f})", f"use \"${name}\" if it is meant to be that",
                    until=L.t_out)
                break
        if L.type in ("text", "counter", "captions"):
            fam = raw.get("font") or scene.theme.get("font")
            if isinstance(fam, str) and fam.lower() not in faces:
                add("warning", "off-brand-font", L.id, L.t_in, f"font '{fam}' is not one of the brand's faces "
                    f"({', '.join(sorted(faces))})", "leave font out, or name the face with a textStyle",
                    until=L.t_out)


def _hex(c) -> str:
    return "#" + "".join(f"{round(v * 255):02x}" for v in c[:3])
