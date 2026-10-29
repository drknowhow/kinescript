"""SVG path data ('M10 10 L 90 90 Q ...') to skia.Path. skia-python ships no parser."""
from __future__ import annotations

import re

import skia

_TOK = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def parse(d: str) -> skia.Path:
    toks = _TOK.findall(d)
    path = skia.Path()
    i, cmd = 0, None
    cx = cy = sx = sy = 0.0
    last_ctrl: tuple[float, float] | None = None
    last_cmd = ""
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                path.close()
                cx, cy = sx, sy
                last_cmd = "Z"
                continue
        if cmd is None:
            raise ValueError(f"path data must start with a command: {d[:40]!r}")
        n = _ARGS[cmd.upper()]
        vals = [float(v) for v in toks[i:i + n]]
        if len(vals) < n:
            raise ValueError(f"path command {cmd} needs {n} numbers")
        i += n
        rel = cmd.islower()
        U = cmd.upper()
        ox, oy = (cx, cy) if rel else (0.0, 0.0)
        if U == "M":
            cx, cy = vals[0] + ox, vals[1] + oy
            path.moveTo(cx, cy)
            sx, sy = cx, cy
            cmd = "l" if rel else "L"  # extra pairs after M are line-tos
        elif U == "L":
            cx, cy = vals[0] + ox, vals[1] + oy
            path.lineTo(cx, cy)
        elif U == "H":
            cx = vals[0] + (cx if rel else 0)
            path.lineTo(cx, cy)
        elif U == "V":
            cy = vals[0] + (cy if rel else 0)
            path.lineTo(cx, cy)
        elif U == "C":
            x1, y1, x2, y2, x, y = vals
            path.cubicTo(x1 + ox, y1 + oy, x2 + ox, y2 + oy, x + ox, y + oy)
            last_ctrl = (x2 + ox, y2 + oy)
            cx, cy = x + ox, y + oy
        elif U == "S":
            x2, y2, x, y = vals
            x1, y1 = (2 * cx - last_ctrl[0], 2 * cy - last_ctrl[1]) if last_ctrl and last_cmd in "CS" else (cx, cy)
            path.cubicTo(x1, y1, x2 + ox, y2 + oy, x + ox, y + oy)
            last_ctrl = (x2 + ox, y2 + oy)
            cx, cy = x + ox, y + oy
        elif U == "Q":
            x1, y1, x, y = vals
            path.quadTo(x1 + ox, y1 + oy, x + ox, y + oy)
            last_ctrl = (x1 + ox, y1 + oy)
            cx, cy = x + ox, y + oy
        elif U == "T":
            x, y = vals
            x1, y1 = (2 * cx - last_ctrl[0], 2 * cy - last_ctrl[1]) if last_ctrl and last_cmd in "QT" else (cx, cy)
            path.quadTo(x1, y1, x + ox, y + oy)
            last_ctrl = (x1, y1)
            cx, cy = x + ox, y + oy
        elif U == "A":
            rx, ry, rot, large, sweep, x, y = vals
            path.arcTo(rx, ry, rot, skia.Path.ArcSize.kLarge_ArcSize if large else skia.Path.ArcSize.kSmall_ArcSize,
                       skia.PathDirection.kCW if sweep else skia.PathDirection.kCCW, x + ox, y + oy)
            cx, cy = x + ox, y + oy
        last_cmd = U
    return path
