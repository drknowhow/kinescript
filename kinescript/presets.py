"""Enter/exit motions. "enter": "rise", "rise+blur", or {"name": "pop", "dur": 0.4, "ease": "outBack"}.

Each preset maps progress p (0 = absent, 1 = fully present) onto modifiers that the
renderer composes with the layer's own properties.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import ease as E


@dataclass
class Mods:
    alpha: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    rotate: float = 0.0
    blur: float = 0.0
    reveal: float = 1.0      # typewriter / draw-on / wipe amount
    wipe: str = ""           # direction when a wipe is active
    track: float = 0.0       # extra letter spacing (px)

    def combine(self, o: "Mods") -> "Mods":
        return Mods(self.alpha * o.alpha, self.dx + o.dx, self.dy + o.dy, self.scale * o.scale,
                    self.rotate + o.rotate, self.blur + o.blur, min(self.reveal, o.reveal),
                    o.wipe or self.wipe, self.track + o.track)


IDENT = Mods()


def _fade(p, o):
    return Mods(alpha=p)


def _move(dx, dy):
    def f(p, o):
        dist = o.get("dist", 48)
        return Mods(alpha=min(1.0, p * 1.6), dx=(1 - p) * dist * dx, dy=(1 - p) * dist * dy)
    return f


def _pop(p, o):
    return Mods(alpha=min(1.0, p * 2), scale=o.get("from", 0.4) + (1 - o.get("from", 0.4)) * p)


def _zoom(p, o):
    return Mods(alpha=p, scale=o.get("from", 1.25) + (1 - o.get("from", 1.25)) * p)


def _blur(p, o):
    return Mods(alpha=p, blur=(1 - p) * o.get("amount", 24))


def _spin(p, o):
    return Mods(alpha=p, rotate=(1 - p) * o.get("deg", -90), scale=0.6 + 0.4 * p)


def _wipe(direction):
    return lambda p, o: Mods(reveal=p, wipe=direction)


def _reveal(p, o):
    return Mods(reveal=p)


def _snap(p, o):
    """Hard cut in: visible within the first fifth, settling from a slight overscale. No blur."""
    return Mods(alpha=1.0 if p >= 0.2 else p / 0.2, scale=1 + (1 - p) * o.get("amount", 0.06))


def _track(p, o):
    return Mods(alpha=p, track=(1 - p) * o.get("amount", 40))


PRESETS = {
    "none": lambda p, o: IDENT, "fade": _fade,
    "rise": _move(0, 1), "drop": _move(0, -1), "left": _move(1, 0), "right": _move(-1, 0),
    "slide-up": _move(0, 1), "slide-down": _move(0, -1),
    "pop": _pop, "zoom": _zoom, "blur": _blur, "spin": _spin, "track": _track,
    "wipe": _wipe("right"), "wipe-right": _wipe("right"), "wipe-left": _wipe("left"),
    "wipe-up": _wipe("up"), "wipe-down": _wipe("down"),
    "type": _reveal, "draw": _reveal, "snap": _snap,
}
DEFAULT_EASE = {"pop": "outBack", "spin": "outBack", "type": "linear", "draw": "inOutCubic", "snap": "outQuart"}


# Names other tools use for the same motions. Keys are lower case without dashes.
ALIASES = {"typewriter": "type", "typing": "type", "slide": "left", "scale": "pop", "grow": "pop", "bounce": "pop",
           "appear": "snap", "cut": "snap", "dissolve": "fade", "opacity": "fade", "up": "rise", "riseup": "rise",
           "down": "drop", "fall": "drop", "blurin": "blur", "focus": "blur", "rotate": "spin", "reveal": "wipe",
           "write": "draw", "stroke": "draw", "letterspacing": "track", "tracking": "track"}


def canonical(name: str) -> str | None:
    """'fadeIn' -> 'fade', 'slideUp' -> 'slide-up', 'typewriter' -> 'type'; None when nothing fits."""
    if name in PRESETS:
        return name
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "-", str(name).strip()).lower().replace("_", "-").replace(" ", "-")
    for cand in (s, re.sub(r"-?(in|out)$", "", s)):
        if cand in PRESETS:
            return cand
        sq = cand.replace("-", "")
        if sq in ALIASES:
            return ALIASES[sq]
        for p in PRESETS:
            if p.replace("-", "") == sq:
                return p
    return None


@dataclass
class Motion:
    names: list[str]
    dur: float = 0.5
    ease: str = ""
    stagger: float = 0.0
    opts: dict = field(default_factory=dict)

    def mods(self, p: float) -> Mods:
        out = IDENT
        for n in self.names:
            e = E.get(self.ease or DEFAULT_EASE.get(n, "outCubic"))
            out = out.combine(PRESETS[n](e(max(0.0, min(1.0, p))), self.opts))
        return out


def parse(spec, default_dur: float = 0.5) -> Motion | None:
    if not spec:
        return None
    if isinstance(spec, str):
        spec = {"name": spec}
    names = []
    for n in (n.strip() for n in str(spec.get("name", "fade")).split("+") if n.strip()):
        c = canonical(n)
        if c is None:
            from .schema import unknown
            raise ValueError(unknown("motion", n, PRESETS))
        names.append(c)
    opts = {k: v for k, v in spec.items() if k not in ("name", "dur", "ease", "stagger")}
    return Motion(names, float(spec.get("dur", default_dur)), spec.get("ease", ""), float(spec.get("stagger", 0)), opts)
