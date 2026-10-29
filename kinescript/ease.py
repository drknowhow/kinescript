"""Easing curves. Every curve maps [0,1] -> roughly [0,1] (back/elastic overshoot)."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable

Ease = Callable[[float], float]


def _out(f: Ease) -> Ease:
    return lambda t: 1.0 - f(1.0 - t)


def _inout(f: Ease) -> Ease:
    return lambda t: f(2 * t) / 2 if t < 0.5 else 1 - f(2 - 2 * t) / 2


def _back(t: float, s: float = 1.70158) -> float:
    return t * t * ((s + 1) * t - s)


def _elastic(t: float) -> float:
    if t in (0.0, 1.0):
        return t
    return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * (2 * math.pi) / 3)


def _bounce_out(t: float) -> float:
    n, d = 7.5625, 2.75
    if t < 1 / d:
        return n * t * t
    if t < 2 / d:
        t -= 1.5 / d
        return n * t * t + 0.75
    if t < 2.5 / d:
        t -= 2.25 / d
        return n * t * t + 0.9375
    t -= 2.625 / d
    return n * t * t + 0.984375


_BASE: dict[str, Ease] = {
    "sine": lambda t: 1 - math.cos(t * math.pi / 2),
    "quad": lambda t: t * t,
    "cubic": lambda t: t ** 3,
    "quart": lambda t: t ** 4,
    "quint": lambda t: t ** 5,
    "expo": lambda t: 0.0 if t == 0 else 2 ** (10 * t - 10),
    "circ": lambda t: 1 - math.sqrt(max(0.0, 1 - t * t)),
    "back": _back,
    "elastic": _elastic,
    "bounce": lambda t: 1 - _bounce_out(1 - t),
}

EASES: dict[str, Ease] = {"linear": lambda t: t, "hold": lambda t: 0.0 if t < 1 else 1.0}
for _n, _f in _BASE.items():
    _c = _n.capitalize()
    EASES["in" + _c] = _f
    EASES["out" + _c] = _out(_f)
    EASES["inOut" + _c] = _inout(_f)
# Short aliases agents reach for first.
EASES.update({"in": EASES["inCubic"], "out": EASES["outCubic"], "inOut": EASES["inOutCubic"],
              "smooth": lambda t: t * t * (3 - 2 * t)})


def _bezier(x1: float, y1: float, x2: float, y2: float) -> Ease:
    def sample(a1: float, a2: float, u: float) -> float:
        return 3 * a1 * u * (1 - u) ** 2 + 3 * a2 * u * u * (1 - u) + u ** 3

    def f(x: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2
            if sample(x1, x2, mid) < x:
                lo = mid
            else:
                hi = mid
        return sample(y1, y2, (lo + hi) / 2)
    return f


def _spring(stiffness: float = 170, damping: float = 26, mass: float = 1) -> Ease:
    """Critically/under-damped spring normalised so it settles at t=1."""
    w0 = math.sqrt(stiffness / mass)
    zeta = damping / (2 * math.sqrt(stiffness * mass))
    span = max(0.3, -math.log(0.001) / max(zeta * w0, 1e-3))

    def raw(t: float) -> float:
        s = t * span
        if zeta < 1:
            wd = w0 * math.sqrt(1 - zeta * zeta)
            return 1 - math.exp(-zeta * w0 * s) * (math.cos(wd * s) + zeta * w0 / wd * math.sin(wd * s))
        return 1 - math.exp(-w0 * s) * (1 + w0 * s)

    end = raw(1.0)
    # Spread the residual so the curve lands exactly on 1 and the last frame does not jump.
    return lambda t: 1.0 if t >= 1 else raw(t) + (1 - end) * t


@lru_cache(maxsize=256)
def get(name: str | None) -> Ease:
    """Resolve an ease by name: 'outCubic', 'spring', 'spring(120,12)', 'bezier(.2,.8,.2,1)', 'steps(4)'."""
    if not name:
        return EASES["outCubic"]
    if name in EASES:
        return EASES[name]
    m = re.fullmatch(r"(\w+)\(([^)]*)\)", name.replace(" ", ""))
    if m:
        kind, args = m.group(1), [float(a) for a in m.group(2).split(",") if a]
        if kind in ("bezier", "cubicBezier") and len(args) == 4:
            return _bezier(*args)
        if kind == "spring":
            return _spring(*args)
        if kind == "steps" and args:
            n = max(1, int(args[0]))
            return lambda t: min(1.0, math.floor(t * n) / n) if t < 1 else 1.0
    if name == "spring":
        return _spring()
    raise ValueError(f"unknown ease '{name}'. Known: {', '.join(sorted(EASES))}, spring(k,d), bezier(a,b,c,d), steps(n)")
