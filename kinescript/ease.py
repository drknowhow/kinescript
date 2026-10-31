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


FUNCTIONS = "spring(stiffness, damping, mass), bezier(x1, y1, x2, y2), steps(n)"


def _args(name: str, kind: str, body: str) -> list[float]:
    out = []
    for a in body.split(","):
        if not a.strip():
            continue
        try:
            v = float(a)
        except ValueError:
            raise ValueError(f"ease '{name}': '{a.strip()}' is not a number") from None
        if not math.isfinite(v):
            raise ValueError(f"ease '{name}': {a.strip()} is not a finite number")
        out.append(v)
    return out


def _function(name: str, kind: str, args: list[float]) -> Ease:
    """bezier(...), spring(...) and steps(...), each checked so a bad one is an error that says why, not a crash."""
    if kind == "bezier":
        if len(args) != 4:
            raise ValueError(f"ease '{name}' has {len(args)} number{'s' * (len(args) != 1)}; bezier takes 4: x1, y1, "
                             "x2, y2, as in CSS cubic-bezier, with x1 and x2 from 0 to 1")
        if not (0 <= args[0] <= 1 and 0 <= args[2] <= 1):
            raise ValueError(f"ease '{name}': x1 and x2 (the 1st and 3rd numbers) must be from 0 to 1")
        return _bezier(*args)
    if kind == "spring":
        if len(args) > 3:
            raise ValueError(f"ease '{name}' has {len(args)} numbers; spring takes up to 3: stiffness, damping, mass")
        full = args + [170.0, 26.0, 1.0][len(args):]
        if min(full) <= 0:  # no damping never settles, and no stiffness or mass divides by zero
            raise ValueError(f"ease '{name}': stiffness, damping and mass must each be above 0, e.g. spring(170, 26)")
        return _spring(*full)
    if len(args) != 1 or args[0] < 1 or args[0] != int(args[0]):
        raise ValueError(f"ease '{name}': steps takes one whole number of steps, 1 or more, like steps(4)")
    n = int(args[0])
    return lambda t: min(1.0, math.floor(t * n) / n) if t < 1 else 1.0


def get(name: str | None) -> Ease:
    """Resolve an ease by name: 'outCubic', 'spring', 'spring(120,12)', 'bezier(.2,.8,.2,1)', 'steps(4)'.
    ValueError says what is wrong with a name that is not an ease."""
    if name is None or name == "":
        return EASES["outCubic"]
    if not isinstance(name, str):
        raise ValueError(f"ease {name!r} is not a name; an ease is text such as \"outCubic\" or \"spring(170, 26)\"")
    return _get(name)


@lru_cache(maxsize=256)
def _get(name: str) -> Ease:
    if name in EASES:
        return EASES[name]
    m = re.fullmatch(r"(\w+)\((.*)\)", name.strip(), re.S)
    if m and m.group(1) in ("bezier", "cubicBezier", "spring", "steps"):
        kind = "bezier" if "ezier" in m.group(1) else m.group(1)
        return _function(name, kind, _args(name, kind, m.group(2)))
    if name == "spring":
        return _spring()
    c = canonical(name)
    if c and c != name:
        return _get(c)
    from .schema import unknown
    raise ValueError(unknown("ease", name, list(EASES) + ["spring"]) + ", " + FUNCTIONS)


def problem(name) -> str | None:
    """Why `name` is not an ease, or None when it is one."""
    try:
        get(name)
    except ValueError as e:
        return str(e)
    return None


CSS = {"ease": "inOutSine", "easein": "inCubic", "easeout": "outCubic", "easeinout": "inOutCubic",
       "linear": "linear", "cubicbezier": "bezier"}


def canonical(name: str) -> str | None:
    """'easeOutCubic', 'ease-out-cubic', 'EaseInOut', CSS 'ease-in' -> Kinescript names; None when nothing fits."""
    s = re.sub(r"[-_ ]", "", str(name)).lower()
    m = re.fullmatch(r"(\w+?)(\(.*\))", s)
    if m and m.group(1) in ("cubicbezier", "bezier", "spring", "steps"):
        return ("bezier" if "bezier" in m.group(1) else m.group(1)) + m.group(2)
    if s in CSS:
        return CSS[s]
    if s.startswith("ease"):
        s = s[4:]
    low = {k.lower(): k for k in EASES}
    return low.get(s)
