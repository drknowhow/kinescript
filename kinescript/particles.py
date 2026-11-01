"""The particles layer: an emitter whose particles each take their birth, start and path from their index and the
seed, with gravity and drag in closed form, so a frame needs no frame before it. Emitter.state gives where they are
at a time; Renderer._draw_particles draws them in one drawAtlas call."""
import math
import zlib

import numpy as np

from . import color as C
from .props import Ctx, compile_prop, key_time, length

SHAPES = ("circle", "square", "rect", "triangle", "star", "ring", "soft", "streak")
EMITTERS = ("point", "line", "rect", "circle", "ring")
RANGES = ("life", "speed", "angle", "size", "spin", "drag", "flip")  # a number, or [min, max] each particle draws from
CURVES = ("sizeOverLife", "opacityOverLife")  # keyframes over 0..1 of a particle's life
# What a particles layer reads itself; the scene compiles none of it as layer props.
OWN = {"preset", "rate", "burst", "gravity", "turbulence", "align", "emitter", "seed", "prewarm", "shape", "fill",
       "fillOverLife", *RANGES, *CURVES}
DEFAULTS = {"life": 2.0, "speed": 100.0, "angle": [0, 360], "size": 8.0, "spin": 0.0, "drag": 0.0, "flip": 0.0}
GRID = 240  # samples a second of a rate that animates
MOST = 200_000  # particles a layer may have alive at once
FLIPS = 8  # widths a flipping particle is drawn at, from edge-on to face-on
TURBULENCE_HZ = 0.5  # how often turbulence sways a particle when its speed is not given

PRESETS = {
    "confetti": {"burst": 150, "life": [3, 4.5], "speed": [900, 2000], "angle": [-118, -62], "gravity": 600,
                 "drag": 3, "size": [14, 22], "shape": "rect", "spin": [-400, 400], "flip": [0.8, 2.2],
                 "turbulence": 30, "fill": ["$accent", "$accent2", "$paper", "#4a7dff", "#ffd166"],
                 "opacityOverLife": [[0, 1], [0.85, 1], [1, 0]]},
    "sparks": {"rate": 90, "life": [0.5, 1.1], "speed": [350, 900], "angle": [-160, -20], "gravity": 1100,
               "drag": 0.8, "size": [16, 30], "shape": "streak", "align": True,
               "fillOverLife": [[0, "#fff6d8"], [0.35, "#ffc24a"], [1, "#ff4a1c"]],
               "sizeOverLife": [[0, 1], [1, 0.4]], "opacityOverLife": [[0, 1], [0.7, 0.9], [1, 0]]},
    "snow": {"rate": 60, "prewarm": True, "life": 18, "speed": [70, 100], "angle": [80, 100], "size": [4, 12],
             "shape": "soft", "turbulence": {"amount": 40, "speed": 0.25}, "fill": "#ffffff",
             "opacityOverLife": [[0, 0], [0.04, 0.9], [0.9, 0.9], [1, 0]], "emitter": "line", "w": "120%",
             "y": -30},
    "dust": {"rate": 12, "prewarm": True, "life": [7, 11], "speed": [4, 16], "size": [3, 8], "shape": "soft",
             "turbulence": {"amount": 30, "speed": 0.15}, "fill": "#ffffff",
             "opacityOverLife": [[0, 0], [0.3, 0.45], [0.7, 0.45], [1, 0]], "emitter": "rect", "w": "100%",
             "h": "100%"},
    "bubbles": {"rate": 5, "prewarm": True, "life": [8.5, 11], "speed": [90, 130], "angle": [-96, -84],
                "size": [12, 40], "shape": "ring", "turbulence": {"amount": 18, "speed": 0.7},
                "fill": "#ffffff", "sizeOverLife": [[0, 0.4], [0.12, 1], [1, 1.15]],
                "opacityOverLife": [[0, 0], [0.08, 0.8], [0.85, 0.8], [1, 0]], "emitter": "line", "w": "100%",
                "y": "100%+40"},
}

_M64 = (1 << 64) - 1
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _u(key: int, k: np.ndarray, ch: int) -> np.ndarray:
    """A number in [0, 1) for each index in k, from the stream's key and the channel (splitmix64): the same on every
    machine and at every frame, so a particle keeps its draw for life."""
    x = k.astype(np.uint64) * _GOLD + np.uint64((key * 0x632BE59BD9B4E019 + ch * 0xD1B54A32D192ED03 + 1) & _M64)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _themed(v, theme: dict):
    if isinstance(v, str) and v.startswith("$"):
        return theme.get(v[1:], v)
    if isinstance(v, list):
        return [_themed(x, theme) for x in v]
    return v


def expand(raw: dict, theme: dict) -> dict:
    """The layer with its preset's keys under its own, and its emitter in the middle of the frame unless placed."""
    out = {"x": "50%", "y": "50%"}
    out.update({k: _themed(v, theme) for k, v in PRESETS.get(raw.get("preset"), {}).items()})
    out.update(raw)
    return out


def _range(v) -> tuple[float, float]:
    if isinstance(v, (list, tuple)):
        return float(v[0]), float(v[-1])
    return float(v), float(v)


def _curve(v) -> tuple[np.ndarray, np.ndarray] | None:
    """Keyframes over 0..1 of a life ([[0, 1], [1, 0]]), {"from": 1, "to": 0} or a number, as np.interp's points."""
    if v is None:
        return None
    if isinstance(v, dict):
        v = [[0, v.get("from", v.get("to", 1))], [1, v.get("to", 1)]]
    if not isinstance(v, list):
        v = [[0, v]]
    pts = sorted((float(p), float(x)) for p, x in v)
    return np.array([p for p, _ in pts]), np.array([x for _, x in pts])


def _colours(v) -> tuple[np.ndarray, np.ndarray]:
    """Colours as 0xRRGGBB and their alphas."""
    cols = [C.parse(c) for c in (v if isinstance(v, list) else [v])]
    rgb = np.array([C.to_argb_int(c) & 0xFFFFFF for c in cols], dtype=np.uint32)
    return rgb, np.array([c[3] for c in cols])


def _over_life(v, steps: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """fillOverLife as 0xRRGGBB and alpha at `steps` points through a life, mixed key to key as colour keys are."""
    if isinstance(v, dict):
        v = [[0, v.get("from", v.get("to"))], [1, v.get("to")]]
    keys = sorted(((float(p), C.parse(c)) for p, c in v), key=lambda k: k[0])
    rgb, alpha = np.zeros(steps, dtype=np.uint32), np.zeros(steps)
    for j in range(steps):
        p = j / (steps - 1)
        if p <= keys[0][0]:
            col = keys[0][1]
        elif p >= keys[-1][0]:
            col = keys[-1][1]
        else:
            i = next(i for i in range(1, len(keys)) if keys[i][0] >= p)
            (p0, a), (p1, b) = keys[i - 1], keys[i]
            col = C.mix(a, b, (p - p0) / (p1 - p0) if p1 > p0 else 1.0)
        rgb[j], alpha[j] = C.to_argb_int(col) & 0xFFFFFF, col[3]
    return rgb, alpha


def _drag(k, age):
    """How far a unit start velocity and a unit pull carry a particle in `age` seconds against drag k (1/s):
    f = (1 - e^-k*age) / k and h = (age - f) / k, which are age and age^2 / 2 without drag."""
    if np.ndim(k) == 0 and k <= 0:
        return age, 0.5 * age * age
    k = np.maximum(k, 1e-6)
    f = -np.expm1(-k * age) / k
    return f, (age - f) / k


class Emitter:
    """A particles layer's particles at any time of the layer, from its keys alone."""

    def __init__(self, raw: dict, ctx: Ctx, lid: str):
        self.fps = ctx.fps
        get = lambda k: raw.get(k, DEFAULTS.get(k))  # noqa: E731
        self.ranges = {k: _range(get(k)) for k in RANGES}
        rate = raw.get("rate", 0)
        self.rate = float(rate) if isinstance(rate, (int, float)) else None
        self.rate_prop = None if self.rate is not None else compile_prop("rate", rate, ctx)
        burst = raw.get("burst")
        self.bursts = [] if not burst else [[0, burst]] if isinstance(burst, (int, float)) else burst
        g = raw.get("gravity", 0)
        self.gravity = (float(g[0]), float(g[1])) if isinstance(g, (list, tuple)) else (0.0, float(g))
        tb = raw.get("turbulence", 0)
        self.turbulence = ((float(tb.get("amount", 0)), float(tb.get("speed", TURBULENCE_HZ)))
                           if isinstance(tb, dict) else (float(tb), TURBULENCE_HZ))
        self.align = raw.get("align") is True
        pw = raw.get("prewarm")
        self.prewarm = self.ranges["life"][1] if pw is True else float(pw) if isinstance(pw, (int, float)) else 0.0
        self.kind = raw.get("emitter", "point")
        self.w = float(length(raw.get("w", ctx.w), "w", ctx))
        self.h = float(length(raw.get("h", ctx.h), "h", ctx))
        self.r = float(length(raw.get("r", 100), "r", ctx))
        seed = raw.get("seed")
        self.seed = int(seed) if isinstance(seed, (int, float)) else zlib.crc32(lid.encode())
        self.size_curve = _curve(raw.get("sizeOverLife"))
        self.opacity_curve = _curve(raw.get("opacityOverLife"))
        self.lut = _over_life(raw["fillOverLife"]) if raw.get("fillOverLife") is not None else None
        self.fill = _colours(raw.get("fill", "#ffffff"))
        self.shape = raw.get("shape") if raw.get("text") is None and raw.get("src") is None else None
        self.flips = self.ranges["flip"][1] > 0
        self.spins = self.ranges["spin"] != (0.0, 0.0) or self.flips
        self._cum: dict[float, tuple] = {}  # layer length -> (grid, emitted by each grid time) for a rate that animates

    def most(self) -> float:
        """The largest size a particle reaches, in px."""
        top = self.ranges["size"][1]
        return top * (float(self.size_curve[1].max()) if self.size_curve is not None else 1.0)

    def _cumulative(self, d: float, t_in: float) -> tuple[np.ndarray, np.ndarray]:
        hit = self._cum.get(d)
        if hit is None:
            P = self.prewarm
            n = int(math.ceil((d + P) * GRID)) + 1
            grid = np.arange(n) / GRID
            rate = np.array([max(0.0, float(self.rate_prop.at(max(0.0, s - P), d, t_in + max(0.0, s - P))))
                             for s in grid])
            cum = np.concatenate(([0.0], np.cumsum((rate[1:] + rate[:-1]) * (0.5 / GRID))))
            hit = self._cum[d] = (grid, cum)
        return hit

    def _born(self, t: float, d: float, t_in: float) -> tuple[np.ndarray, np.ndarray]:
        """The rate's particles born by t and maybe still alive: their indices and birth times."""
        P, lmax = self.prewarm, self.ranges["life"][1]
        now, end = t + P, d + P  # on the emitter's clock, which starts prewarm seconds before the layer
        if self.rate is not None:
            if self.rate <= 0:
                return np.empty(0, np.int64), np.empty(0)
            hi, lo = self.rate * min(now, end), self.rate * max(now - lmax, 0.0)
        else:
            grid, cum = self._cumulative(d, t_in)
            hi, lo = float(np.interp(min(now, end), grid, cum)), float(np.interp(max(now - lmax, 0.0), grid, cum))
        k0, k1 = max(int(math.floor(lo)) - 1, 0), int(math.floor(hi))
        if k1 < k0:
            return np.empty(0, np.int64), np.empty(0)
        if k1 - k0 > 4 * MOST:
            raise ValueError(f"rate gives {k1 - k0:,} particles alive at once; a particles layer has {MOST:,} at most")
        k = np.arange(k0, k1 + 1, dtype=np.int64)
        q = k + _u(self.seed * 1000003, k, 0)  # each birth falls somewhere in its own slot
        keep = q <= hi
        k, q = k[keep], q[keep]
        if self.rate is not None:
            b = q / self.rate
        else:
            i = np.clip(np.searchsorted(cum, q, side="left"), 1, len(cum) - 1)
            c0, c1 = cum[i - 1], cum[i]
            b = grid[i - 1] + (q - c0) / np.where(c1 > c0, c1 - c0, 1.0) / GRID
        return k, b - P

    def _draw(self, name: str, key: int, k: np.ndarray, ch: int):
        lo, hi = self.ranges[name]
        return lo if lo == hi else lo + (hi - lo) * _u(key, k, ch)

    def state(self, t: float, d: float, t_in: float = 0.0) -> dict | None:
        """The particles alive at the layer's time t (d: the layer's length; t_in: when it starts in the scene), in
        the layer's space, oldest first: x, y, size (px), rot (radians), argb (uint32), cell (which flip each is drawn
        at, or None) and birth (s). None when none are alive."""
        streams = []
        k, b = self._born(t, d, t_in)
        if len(k):
            streams.append((self.seed * 1000003, k, b))
        lmax = self.ranges["life"][1]
        for j, (when, n) in enumerate(self.bursts, 1):
            tb = key_time(when, d, self.fps)
            if 0 <= t - tb < lmax and n >= 1 and tb <= d:
                n = int(n)
                if n > MOST:
                    raise ValueError(f"a burst of {n:,} particles; a particles layer has {MOST:,} at most")
                streams.append((self.seed * 1000003 + j, np.arange(n, dtype=np.int64), np.full(n, float(tb))))
        parts = [p for p in (self._alive(key, k, b, t) for key, k, b in streams) if p is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        out = {name: np.concatenate([p[name] for p in parts]) if parts[0][name] is not None else None
               for name in parts[0]}
        order = np.argsort(out["birth"], kind="stable")
        return {name: v[order] if v is not None else None for name, v in out.items()}

    def _alive(self, key: int, k: np.ndarray, b: np.ndarray, t: float) -> dict | None:
        age = t - b
        life = self._draw("life", key, k, 1)
        live = age < life
        if not live.all():
            k, b, age = k[live], b[live], age[live]
            if np.ndim(life):
                life = life[live]
        n = len(k)
        if not n:
            return None
        p = age / life
        ang = np.radians(self._draw("angle", key, k, 3))
        speed = self._draw("speed", key, k, 2)
        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
        f, h = _drag(self._draw("drag", key, k, 7), age)
        gx, gy = self.gravity
        if self.kind == "point":
            ox = oy = 0.0
        elif self.kind == "line":
            ox, oy = (_u(key, k, 10) - 0.5) * self.w, 0.0
        elif self.kind == "rect":
            ox, oy = (_u(key, k, 10) - 0.5) * self.w, (_u(key, k, 11) - 0.5) * self.h
        else:  # circle (anywhere inside, evenly) or ring (on its edge)
            a = _u(key, k, 10) * (2 * math.pi)
            rad = self.r * (np.sqrt(_u(key, k, 11)) if self.kind == "circle" else 1.0)
            ox, oy = rad * np.cos(a), rad * np.sin(a)
        x = ox + vx * f + gx * h
        y = oy + vy * f + gy * h
        amount, hz = self.turbulence
        if amount > 0:  # two sways each particle has its own pace and phase for, still at birth
            wx = (2 * math.pi * hz) * (0.6 + 0.8 * _u(key, k, 12))
            wy = (2 * math.pi * hz) * (0.6 + 0.8 * _u(key, k, 13))
            px, py = (2 * math.pi) * _u(key, k, 14), (2 * math.pi) * _u(key, k, 15)
            x = x + amount * (np.sin(wx * age + px) - np.sin(px))
            y = y + amount * (np.sin(wy * age + py) - np.sin(py))
        if self.align:  # along its velocity: the start velocity worn down by drag, plus the pull, plus the sway
            e = np.exp(-np.maximum(self._draw("drag", key, k, 7), 0.0) * age)
            dx, dy = vx * e + gx * f, vy * e + gy * f
            if amount > 0:
                dx = dx + amount * wx * np.cos(wx * age + px)
                dy = dy + amount * wy * np.cos(wy * age + py)
            rot = np.arctan2(dy, dx)
        elif self.spins:
            rot = np.radians(360 * _u(key, k, 17) + self._draw("spin", key, k, 5) * age)
        else:
            rot = np.zeros(n)
        size = self._draw("size", key, k, 4) * np.ones(n)
        if self.size_curve is not None:
            size = size * np.interp(p, *self.size_curve)
        cell = None
        if self.flips:
            turn = np.abs(np.cos((2 * math.pi) * (self._draw("flip", key, k, 8) * age + _u(key, k, 16))))
            cell = np.rint(turn * (FLIPS - 1)).astype(np.int64)
        if self.lut is not None:
            j = np.rint(p * 255).astype(np.int64)
            rgb, alpha = self.lut[0][j], self.lut[1][j]
        else:
            rgb, alpha = self.fill
            if len(rgb) == 1:
                rgb, alpha = rgb[0], float(alpha[0])
            else:
                j = k % len(rgb)
                rgb, alpha = rgb[j], alpha[j]
        if self.opacity_curve is not None:
            alpha = alpha * np.interp(p, *self.opacity_curve)
        a8 = np.clip(np.rint(np.broadcast_to(alpha, (n,)) * 255), 0, 255).astype(np.uint32)
        return {"x": np.broadcast_to(x, (n,)).astype(np.float64), "y": np.broadcast_to(y, (n,)).astype(np.float64),
                "size": size, "rot": rot, "argb": (a8 << np.uint32(24)) | rgb, "cell": cell, "birth": b}
