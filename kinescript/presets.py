"""Enter/exit motions. "enter": "rise", "rise+blur", or {"name": "pop", "dur": 0.4, "ease": "outBack"}.

Each preset maps progress p (0 = absent, 1 = fully present) onto modifiers that the
renderer composes with the layer's own properties. Emphasis motions play in the middle of a
layer's life instead: "emphasis": [{"name": "pulse", "at": 1.2, "repeat": 2}], each cycle
starting and ending at rest.
"""
from __future__ import annotations

import math
import random
import re
import zlib
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
    rx: float = 0.0          # added to rotateX, rotateY (degrees) and z (px)
    ry: float = 0.0
    dz: float = 0.0
    sx: float = 1.0          # a squash or stretch, and a turn (degrees), about pivot
    sy: float = 1.0
    hinge: float = 0.0
    pivot: str = ""          # where sx, sy and hinge act: "" the middle of the box, "top" or "bottom" its edge
    smear: float = 0.0       # blur across (px), on top of blur
    glitch: float = 0.0      # how far the colours split and slices jump, 0..1
    scramble: float = 0.0    # the share of letters not settled yet, 0..1
    count: int = 0           # strips of blinds
    step: int = 0            # which random pick glitch and scramble show, with seed
    seed: int = 0

    def combine(self, o: "Mods") -> "Mods":
        return Mods(self.alpha * o.alpha, self.dx + o.dx, self.dy + o.dy, self.scale * o.scale,
                    self.rotate + o.rotate, self.blur + o.blur, min(self.reveal, o.reveal),
                    o.wipe or self.wipe, self.track + o.track, self.rx + o.rx, self.ry + o.ry, self.dz + o.dz,
                    self.sx * o.sx, self.sy * o.sy, self.hinge + o.hinge, o.pivot or self.pivot,
                    self.smear + o.smear, max(self.glitch, o.glitch), max(self.scramble, o.scramble),
                    o.count or self.count, o.step or self.step, o.seed or self.seed)


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


def _flip(p, o):
    return Mods(alpha=min(1.0, p * 4), ry=(1 - p) * o.get("deg", 90))


def _flip_up(p, o):
    return Mods(alpha=min(1.0, p * 4), rx=(1 - p) * o.get("deg", 90))


def _tumble(p, o):
    return Mods(alpha=min(1.0, p * 2), rx=(1 - p) * -70, ry=(1 - p) * o.get("deg", 160),
                dz=(1 - p) * -o.get("dist", 400))


def _push(p, o):
    return Mods(alpha=p, dz=(1 - p) * -o.get("dist", 600))


def fall(p: float, bounces: int, contact: float = 0.0) -> tuple[float, float, float]:
    """A dropped thing at p of its motion: its height (1 where it starts, 0 on the ground), its speed (1 as it first
    lands) and how far it is squashed (0..1). It falls under gravity and bounces `bounces` times, each half as fast
    as the one before, so a quarter as high; with contact, each landing holds it on the ground for that share of the
    fall's time (times the landing's speed) while it squashes and recovers, the last one included."""
    segs = [("fall", 1.0, 1.0)]
    for k in range(bounces + 1):
        v = 0.5 ** k  # how fast it lands the k+1th time
        if contact > 0:
            segs.append(("land", contact * v, v))
        if k < bounces:
            segs.append(("air", 1.0 * v, v / 2))  # up and down again at half the speed: 2 * v / 2 of the fall's time
    tau = max(0.0, min(1.0, p)) * sum(d for _, d, _ in segs)
    for kind, d, v in segs:
        if tau <= d:
            break
        tau -= d
    if kind == "fall":
        return 1 - tau * tau, tau, 0.0
    if kind == "air":  # thrown up at v (in landing speeds: 2v in heights over the fall's time)
        return max(0.0, 2 * v * tau - tau * tau), abs(v - tau), 0.0
    return 0.0, 0.0, v * math.sin(math.pi * min(1.0, tau / d))


def _land(p, o):
    h, _, _ = fall(p, int(o.get("bounces", 3)))
    return Mods(alpha=min(1.0, p * 8), dy=-h * o.get("dist", 160))


def _squash(p, o):
    h, v, q = fall(p, int(o.get("bounces", 2)), 0.35)
    sy = (1 + 0.25 * v) * (1 - o.get("amount", 0.35) * q)  # long as it falls, flat as it lands
    return Mods(alpha=min(1.0, p * 8), dy=-h * o.get("dist", 160), sx=1 / max(sy, 0.05), sy=sy, pivot="bottom")


def _elastic(p, o):
    f = o.get("from", 0.0)
    return Mods(alpha=min(1.0, max(0.0, p) * 4), scale=f + (1 - f) * p)


def _swing(p, o):
    """Hung from its top edge, it swings down into place: each swing smaller, at rest when the motion ends."""
    calm = (1 - p) ** 2 * (1 + 2 * p)  # 1 at the start, 0 at the end, still at both
    return Mods(alpha=min(1.0, p * 5), hinge=o.get("deg", 60) * calm * math.cos(3 * math.pi * p), pivot="top")


def _stretch(p, o):
    return Mods(alpha=min(1.0, max(0.0, p) * 4), sx=max(0.0, p), sy=1 + 0.4 * (1 - p))


def _scramble(p, o):
    return Mods(alpha=min(1.0, p * 10), scramble=1 - p, step=int(p * 16), seed=int(o.get("seed", 0)))


def _glitch(p, o):
    return Mods(alpha=min(1.0, p * 3), glitch=1 - p, step=int(p * 12), seed=int(o.get("seed", 0)))


def _whip(p, o):
    return Mods(alpha=min(1.0, p * 3), dx=(1 - p) * o.get("dist", 400), smear=(1 - p) * o.get("amount", 48))


def _shown(kind):
    return lambda p, o: Mods(reveal=p, wipe=kind, count=int(o.get("count", 6)) if kind == "blinds" else 0)


def slices(key: str, step: int, n: int = 6) -> list[tuple[float, float, float]]:
    """The n bands glitch cuts a layer into at a step, top to bottom, as (top, bottom, shift): top and bottom as shares
    of its height, shift from -1 to 1 as a share of the farthest a band jumps, 0 for about half the bands."""
    rnd = random.Random(zlib.crc32(f"{key}/{step}".encode()))
    cuts = [0.0] + sorted(rnd.random() for _ in range(n - 1)) + [1.0]
    return [(a, b, rnd.uniform(-1, 1) if rnd.random() < 0.5 else 0.0) for a, b in zip(cuts, cuts[1:])]


_POOLS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz", "0123456789")


def scrambled(text: str, first: int, s: float, base: int, n: int, key: str, step: int, own: str) -> str:
    """text, whose first character is number `first` of the text, with the share s of its letters and digits not yet
    settled shown as others: a letter of the same case, a digit, or one of `own` (the text's own characters) for
    letters outside A to Z. Each settles in turn from number `base` of `n`, a little early or late by key, and shows
    a new character at each step."""
    out = []
    for j, ch in enumerate(text):
        if not ch.isalnum():
            out.append(ch)
            continue
        i = first + j - base
        late = (zlib.crc32(f"{key}/{i}".encode()) & 0xFFFF) / 0x10000
        if 1 - s >= (i + late) / max(n, 1):
            out.append(ch)
            continue
        pool = (_POOLS[0] if "A" <= ch <= "Z" else _POOLS[1] if "a" <= ch <= "z" else _POOLS[2] if "0" <= ch <= "9"
                else own or ch)
        out.append(pool[zlib.crc32(f"{key}/{i}/{step}".encode()) % len(pool)])
    return "".join(out)


PRESETS = {
    "none": lambda p, o: IDENT, "fade": _fade,
    "rise": _move(0, 1), "drop": _move(0, -1), "left": _move(1, 0), "right": _move(-1, 0),
    "slide-up": _move(0, 1), "slide-down": _move(0, -1),
    "pop": _pop, "zoom": _zoom, "blur": _blur, "spin": _spin, "track": _track,
    "wipe": _wipe("right"), "wipe-right": _wipe("right"), "wipe-left": _wipe("left"),
    "wipe-up": _wipe("up"), "wipe-down": _wipe("down"),
    "type": _reveal, "draw": _reveal, "snap": _snap,
    "flip": _flip, "flip-up": _flip_up, "tumble": _tumble, "push": _push,
    "land": _land, "squash": _squash, "elastic": _elastic, "swing": _swing, "stretch": _stretch,
    "iris": _shown("iris"), "clock": _shown("clock"), "blinds": _shown("blinds"), "rise-mask": _shown("rise"),
    "scramble": _scramble, "glitch": _glitch, "whip": _whip,
}
# Motions that turn or move a layer in depth; a scene using one gets a camera.
DEPTH = {"flip", "flip-up", "tumble", "push"}
DEFAULT_EASE = {"pop": "outBack", "spin": "outBack", "type": "linear", "draw": "inOutCubic", "snap": "outQuart",
                "flip": "outBack", "flip-up": "outBack", "land": "linear", "squash": "linear", "swing": "linear",
                "elastic": "outElastic", "stretch": "outBack", "scramble": "linear", "whip": "outExpo",
                "rise-mask": "outQuart"}
# Motions that take longer than dur's 0.5 s unless given one: bounces and swings need the time to settle.
DEFAULT_DUR = {"land": 1.0, "squash": 1.0, "swing": 1.2, "elastic": 0.9, "scramble": 0.8}


# Names other tools use for the same motions. Keys are lower case without dashes.
ALIASES = {"typewriter": "type", "typing": "type", "slide": "left", "scale": "pop", "grow": "pop", "bounce": "pop",
           "appear": "snap", "cut": "snap", "dissolve": "fade", "opacity": "fade", "up": "rise", "riseup": "rise",
           "down": "drop", "fall": "drop", "blurin": "blur", "focus": "blur", "rotate": "spin", "reveal": "wipe",
           "write": "draw", "stroke": "draw", "build": "draw", "assemble": "draw", "letterspacing": "track", "tracking": "track",
           "flipy": "flip", "flipx": "flip-up", "rotatey": "flip", "rotatex": "flip-up", "dolly": "push",
           "fly": "push", "dropbounce": "land", "bouncedrop": "land", "squashstretch": "squash",
           "spring": "elastic", "springy": "elastic", "pendulum": "swing", "hinge": "swing", "circle": "iris",
           "radial": "clock", "radialwipe": "clock", "clockwipe": "clock", "venetian": "blinds", "shutters": "blinds",
           "mask": "rise-mask", "masked": "rise-mask", "linereveal": "rise-mask", "risemask": "rise-mask",
           "decode": "scramble", "shuffle": "scramble", "rgbsplit": "glitch", "chromatic": "glitch",
           "swish": "whip", "whippan": "whip", "smear": "whip"}


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
    order: str = "start"
    seed: int = 0

    def mods(self, p: float) -> Mods:
        out = IDENT
        for n in self.names:
            e = E.get(self.ease or DEFAULT_EASE.get(n, "outCubic"))
            out = out.combine(PRESETS[n](e(max(0.0, min(1.0, p))), self.opts))
        return out


ORDERS = ("start", "end", "center", "edges", "random")


def turns(n: int, order: str = "start", seed: int = 0) -> list[float]:
    """When each of n staggered units starts, in staggers: start goes first to last, end last to first, center from
    the middle out and edges from both ends in (units as far from the middle go together), random in the shuffled
    order seed picks."""
    if order == "end":
        return [float(n - 1 - k) for k in range(n)]
    if order in ("center", "edges"):
        far = [abs(k - (n - 1) / 2) for k in range(n)]
        lo, hi = min(far, default=0.0), max(far, default=0.0)
        return [f - lo if order == "center" else hi - f for f in far]
    if order == "random":
        out = [0.0] * n
        for j, k in enumerate(random.Random(seed).sample(range(n), n)):
            out[k] = float(j)
        return out
    return [float(k) for k in range(n)]


# Motion options that are numbers, and what each one is.
NUMBERS = {"dur": "seconds, 0 or more", "stagger": "seconds between units, 0 or more", "dist": "px moved",
           "amount": "px (blur, track, whip), a scale step (snap, pulse, heartbeat, wobble, tada), how far squash "
                     "flattens or how far flash dims", "deg": "degrees",
           "from": "the start scale", "bounces": "a whole number from 0 to 20",
           "count": "a whole number from 1 to 100"}


# ---- emphasis: each maps u, how far through one cycle (0..1), onto modifiers that are at rest at both ends

def _beat(u: float, a: float, b: float) -> float:
    """A swell from a to b: 0 outside them, 1 half way, still at both ends."""
    return math.sin(math.pi * (u - a) / (b - a)) ** 2 if a <= u < b else 0.0


def _pulse(u, o):
    return Mods(scale=1 + o.get("amount", 0.12) * _beat(u, 0, 1))


def _heartbeat(u, o):
    return Mods(scale=1 + o.get("amount", 0.15) * (_beat(u, 0, 0.28) + 0.6 * _beat(u, 0.28, 0.56)))


def _shake(u, o):
    return Mods(dx=o.get("dist", 12) * math.sin(8 * math.pi * u) * (1 - u))


def _wiggle(u, o):
    return Mods(rotate=o.get("deg", 8) * math.sin(6 * math.pi * u) * (1 - u))


def _wobble(u, o):
    w = o.get("amount", 0.15) * math.sin(4 * math.pi * u) * (1 - u)  # a jelly: wide and short, then narrow and tall
    return Mods(sx=1 + w, sy=1 - w)


def _float(u, o):
    return Mods(dy=-o.get("dist", 10) * (1 - math.cos(2 * math.pi * u)) / 2, rotate=o.get("deg", 2) * math.sin(
        2 * math.pi * u))


def _bob(u, o):
    return Mods(dy=-o.get("dist", 14) * math.sin(math.pi * u))


def _flash(u, o):
    return Mods(alpha=1 - o.get("amount", 1.0) * (1 - math.cos(4 * math.pi * u)) / 2)


def _tada(u, o):
    """Smaller and turned back a little, then larger and rocking, then at rest."""
    a, deg = o.get("amount", 0.1), o.get("deg", 4)
    if u < 0.2:
        s = math.sin(math.pi * u / 0.2)
        return Mods(scale=1 - a * s, rotate=-deg * s)
    s = math.sin(math.pi * (u - 0.2) / 0.8)
    return Mods(scale=1 + a * s, rotate=deg * math.sin(10 * math.pi * (u - 0.2) / 0.8) * s)


def _turn(u, o):
    return Mods(rotate=o.get("deg", 360) * u)


EMPHASES = {"pulse": _pulse, "heartbeat": _heartbeat, "shake": _shake, "wiggle": _wiggle, "wobble": _wobble,
            "float": _float, "bob": _bob, "flash": _flash, "tada": _tada, "spin": _turn}
# One cycle's seconds unless given a dur.
EMPHASIS_DUR = {"pulse": 0.5, "heartbeat": 0.9, "shake": 0.6, "wiggle": 0.6, "wobble": 0.8, "float": 3.0, "bob": 0.6,
                "flash": 0.8, "tada": 1.0, "spin": 1.0}
# Names other tools use for them. Keys are lower case without dashes.
EMPHASIS_ALIASES = {"throb": "pulse", "breathe": "pulse", "grow": "pulse", "beat": "heartbeat", "heart": "heartbeat",
                    "jiggle": "wiggle", "rock": "wiggle", "jello": "wobble", "jelly": "wobble",
                    "rubberband": "wobble", "hover": "float", "drift": "float", "hop": "bob", "bounce": "bob",
                    "blink": "flash", "headshake": "shake", "vibrate": "shake", "rotate": "spin", "turn": "spin",
                    "celebrate": "tada"}
EMPHASIS_KEYS = ("name", "at", "dur", "repeat", "loop", "ease", "stagger", "order", "seed")


def canonical_emphasis(name: str) -> str | None:
    """'headShake' -> 'shake', 'heart-beat' -> 'heartbeat'; None when nothing fits."""
    s = re.sub(r"[-_ ]", "", str(name).strip().lower())
    return s if s in EMPHASES else EMPHASIS_ALIASES.get(s)


@dataclass
class Emphasis:
    names: list[str]
    at: float = 0.0          # seconds into the layer when it starts
    dur: float = 0.5         # one cycle
    repeat: float = 1.0      # cycles, inf with loop
    ease: str = ""
    stagger: float = 0.0     # on a split text, each unit this long after the one before
    opts: dict = field(default_factory=dict)
    order: str = "start"
    seed: int = 0

    @property
    def end(self) -> float:
        return self.at + self.dur * self.repeat

    def mods(self, t: float) -> Mods:
        """Its modifiers t seconds into the layer: none before it starts or after its last cycle."""
        k = (t - self.at) / self.dur
        if not 0 < k < self.repeat:
            return IDENT
        u = k - math.floor(k)
        out = IDENT
        for n in self.names:
            e = E.get(self.ease or ("inOutCubic" if n == "spin" and self.repeat != math.inf else "linear"))
            out = out.combine(EMPHASES[n](e(u), self.opts))
        return out


def emphasis_problem(key: str, v) -> str | None:
    """Why an emphasis entry's value cannot be used, or None."""
    try:
        num = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    except OverflowError:  # an int past the float range
        num = False
    if key == "dur" and v is not None and not (num and v > 0):
        return f"dur is {v!r}; it takes the seconds one cycle lasts, more than 0"
    if key == "at" and v is not None and not (num and v >= 0):
        return f"at is {v!r}; it takes the seconds after the layer starts, 0 or more"
    if key == "repeat" and v is not None and not (num and v > 0):
        return f"repeat is {v!r}; it takes how many cycles to play, more than 0"
    if key == "loop" and v is not None and not isinstance(v, bool):
        return f"loop is {v!r}; it takes true (play until the layer ends) or false"
    return None if key in ("dur", "at", "repeat", "loop") else option_problem(key, v)


def parse_emphasis(spec) -> list[Emphasis]:
    """"pulse", {"name": "pulse", "at": 1.2, "repeat": 2}, or a list of them."""
    if not spec:
        return []
    out = []
    for k, item in enumerate(spec if isinstance(spec, list) else [spec]):
        where = f"emphasis[{k}]" if isinstance(spec, list) else "emphasis"
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be a name like \"pulse\" or an object "
                             f"{{\"name\": \"pulse\", \"at\": 1.2}}, got {item!r}")
        names = []
        for n in (n.strip() for n in str(item.get("name", "pulse")).split("+") if n.strip()):
            c = canonical_emphasis(n)
            if c is None:
                from .schema import unknown
                raise ValueError(f"{where}: " + unknown("emphasis", n, EMPHASES))
            names.append(c)
        for key, v in item.items():
            why = emphasis_problem(key, v)
            if why:
                raise ValueError(f"{where}: {why}")
        dur = item.get("dur")
        if dur is None:
            dur = max(EMPHASIS_DUR[n] for n in names) if names else 0.5
        opts = {key: v for key, v in item.items() if key not in EMPHASIS_KEYS and v is not None}
        out.append(Emphasis(names, float(item.get("at") or 0), float(dur),
                            math.inf if item.get("loop") is True else float(item.get("repeat") or 1),
                            item.get("ease") or "", float(item.get("stagger") or 0), opts,
                            item.get("order") or "start", int(item.get("seed") or 0)))
    return out


def option_problem(key: str, v) -> str | None:
    """Why a motion option's value cannot be used, or None."""
    if key == "ease":
        return E.problem(v) if v not in (None, "") else None
    if key == "order":
        return None if v is None or v in ORDERS else f"order is {v!r}; it takes {', '.join(ORDERS)}"
    if key == "seed" and v is not None and not (isinstance(v, int) and not isinstance(v, bool)):
        return f"seed is {v!r}; it takes a whole number"
    if key not in NUMBERS or v is None:  # null is the default
        return None
    try:
        ok = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
    except OverflowError:  # an int past the float range
        ok = False
    if ok and key in ("bounces", "count"):
        lo, hi = (0, 20) if key == "bounces" else (1, 100)
        ok = float(v).is_integer() and lo <= v <= hi
    if not ok or (key in ("dur", "stagger") and v < 0):
        return f"{key} is {v!r}; it takes a number: {NUMBERS[key]}"
    return None


def parse(spec, default_dur: float = 0.5, key: str = "motion") -> Motion | None:
    if not spec:
        return None
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"{key} must be a motion name like \"rise\" or an object "
                         f"{{\"name\": \"rise\", \"dur\": 0.5}}, "
                         f"got {spec!r}")
    names = []
    for n in (n.strip() for n in str(spec.get("name", "fade")).split("+") if n.strip()):
        c = canonical(n)
        if c is None:
            from .schema import unknown
            raise ValueError(f"{key}: " + unknown("motion", n, PRESETS))
        names.append(c)
    for k, v in spec.items():
        why = option_problem(k, v)
        if why:
            raise ValueError(f"{key}: {why}")
    opts = {k: v for k, v in spec.items() if k not in ("name", "dur", "ease", "stagger", "order")
            and v is not None}
    dur, stagger = spec.get("dur"), spec.get("stagger")
    if dur is None:
        dur = max(DEFAULT_DUR.get(n, default_dur) for n in names) if names else default_dur
    return Motion(names, float(dur), spec.get("ease") or "",
                  float(stagger or 0), opts, spec.get("order") or "start", int(spec.get("seed") or 0))
