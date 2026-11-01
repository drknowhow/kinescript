"""Animatable property values.

A property is one of:
    42 / "#ffb000" / "50%" / "50%-120"     constant (lengths relative to the canvas)
    [[0, 0], [0.6, 1, "outBack"], [-0.4, 1], ["end", 0]]
                                           keyframes: [time, value, ease into this key].
                                           Times are seconds from the layer's start;
                                           negative = from its end; "40%" = of its length.
    {"from": 0, "to": 1, "dur": 0.6, "delay": 0, "ease": "outCubic"}   one tween
    "=540 + sin(t*3)*40"                    expression of t (local s), d (layer length),
                                           p (t/d), T (scene s), i (unit index)
"""
from __future__ import annotations

import ast
import colorsys
import math
import re
from bisect import bisect_right
from typing import Any, Callable

from . import color as C
from . import ease as E

W_AXIS = {"x", "w", "left", "cx", "dx", "maxWidth", "x1", "x2"}
H_AXIS = {"y", "h", "top", "cy", "dy", "y1", "y2"}
NOT_ANIMATABLE = {"points", "trim", "src", "text", "id", "type", "cues", "children", "layers",
                  "d", "font", "split", "enter", "exit", "stops", "data", "words", "format",
                  "fit", "align", "anchor", "blend", "clip", "loop", "mode", "cap", "join"}

_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"  # 2, 2.5, .5, 1e3
_LEN = re.compile(rf"^\s*(-?{_NUM})\s*(%|vw|vh|vmin|px)?\s*(?:([+-])\s*({_NUM})\s*(px)?)?\s*$")
# A keyframe time: seconds ("0.5", "0.5s"), frames ("12f"), "500ms", a share of the layer ("40%"), or "end" with an
# optional offset ("end-0.2", "end-10f").
KEY_TIME = re.compile(r"\s*(-?\d+(\.\d+)?(s|f|ms)?|\d+(\.\d+)?%|end([+-]\d+(\.\d+)?(s|f|ms)?)?)\s*")
KEY_TIMES = ('0.5 (seconds from the layer start; negative counts from its end), "12f", "500ms", "40%", "end" or '
             '"end-0.2"')


def noise(x: float, seed: float = 0.0) -> float:
    """Smooth 1-D value noise in [-1, 1]. Pure function of x, so any frame renders alone."""
    def h(n: int) -> float:
        n = (n * 374761393 + int(seed * 668265263)) & 0xFFFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
        return ((n ^ (n >> 16)) & 0xFFFF) / 32767.5 - 1.0
    i = math.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def _hex(c) -> str:
    out = "#" + "".join(f"{round(max(0.0, min(1.0, float(v))) * 255):02x}" for v in c[:3])
    return out + (f"{round(max(0.0, min(1.0, float(c[3]))) * 255):02x}" if len(c) > 3 and c[3] < 1 else "")


def hsl(h: float, s: float = 0.7, l: float = 0.55) -> str:  # noqa: E741 - the name everyone knows
    """A colour from its hue in degrees (0 red, 120 green, 240 blue, and round again), saturation and lightness
    from 0 to 1, as #rrggbb."""
    return _hex(colorsys.hls_to_rgb((float(h) / 360.0) % 1.0, max(0.0, min(1.0, float(l))),
                                    max(0.0, min(1.0, float(s)))))


def mix(a, b, t: float) -> str:
    """The colour t of the way from colour a to colour b (t from 0 to 1), mixed as keyframes mix a fill."""
    return _hex(C.mix(C.parse(a), C.parse(b), max(0.0, min(1.0, float(t)))))


EXPR_ENV: dict[str, Any] = {k: getattr(math, k) for k in (
    "sin", "cos", "tan", "pi", "sqrt", "floor", "ceil", "exp", "log", "atan2", "hypot", "tau")}
EXPR_ENV.update(abs=abs, min=min, max=max, round=round, noise=noise, hsl=hsl, mix=mix,
                clamp=lambda v, a=0.0, b=1.0: max(a, min(b, v)),
                lerp=lambda a, b, t: a + (b - a) * t,
                wiggle=lambda freq, amp, t, seed=0: noise(t * freq, seed) * amp,
                ease=lambda name, t: E.get(name)(max(0.0, min(1.0, t))))


class Ctx:
    """Reference box for % lengths (the canvas, or a group's w/h)."""
    def __init__(self, w: float, h: float, fps: float):
        self.w, self.h, self.fps = w, h, fps


def length(v, name: str, ctx: Ctx):
    if not isinstance(v, str) or C.is_color(v):
        return v
    m = _LEN.match(v)
    if not m:
        return v
    num, unit, sign, add, _ = m.groups()
    x = float(num)
    if unit in ("%", "vw", "vh", "vmin"):
        if unit == "vw" or (unit == "%" and name in W_AXIS):
            ref = ctx.w
        elif unit == "vh" or (unit == "%" and name in H_AXIS):
            ref = ctx.h
        else:
            ref = min(ctx.w, ctx.h)
        x = x / 100 * ref
    if sign:
        x += float(add) if sign == "+" else -float(add)
    return x


def lerp(a, b, t: float):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) == 4:  # parsed colours
        return C.mix(a, b, t)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return [lerp(x, y, t) for x, y in zip(a, b)]
    return b if t >= 1 else a


def _norm(v):
    """Colours become RGBA tuples once, so draw code never re-parses them."""
    return C.parse(v) if C.is_color(v) else v


def _is_time(k) -> bool:
    if isinstance(k, (int, float)):
        return not isinstance(k, bool)
    return isinstance(k, str) and bool(KEY_TIME.fullmatch(k))


def is_keyframes(v) -> bool:
    return (isinstance(v, list) and len(v) >= 1
            and all(isinstance(k, (list, tuple)) and 2 <= len(k) <= 3 and _is_time(k[0]) for k in v))


class Prop:
    __slots__ = ("static", "value", "fn", "keys")

    def __init__(self, static: bool, value=None, fn: Callable | None = None, keys: Callable | None = None):
        self.static, self.value, self.fn, self.keys = static, value, fn, keys

    def at(self, t: float, d: float, T: float = 0.0, i: int = 0):
        return self.value if self.static else self.fn(t, d, T, i)

    def held(self, t0: float, t1: float, d: float) -> bool:
        """Whether the value is the same at every time in [t0, t1]: a constant, or keyframes before their first key or
        after their last. An expression may follow anything, so it never holds."""
        if self.static:
            return True
        if self.keys is None:
            return False
        pts = self.keys(d)
        return t1 <= pts[0][0] or t0 > pts[-1][0]


def _seconds(s: str, fps: float) -> float:
    """'0.5', '0.5s', '12f' or '500ms' in seconds."""
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    return float(s.rstrip("sf")) / (fps if s.endswith("f") else 1)


def key_time(k, d: float, fps: float) -> float:
    if isinstance(k, (int, float)):
        return d + k if k < 0 else float(k)
    s = k.strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100 * d
    if s.startswith("end"):
        return d + _seconds(s[3:].strip() or "+0", fps)
    v = _seconds(s, fps)
    return d + v if v < 0 else v


_SAFE_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
               ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List, ast.operator, ast.unaryop,
               ast.boolop, ast.cmpop)


def _sequence(node) -> bool:
    """A string, list or tuple, or arithmetic on one: '[0] * 10**9' would build it in memory."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, (ast.List, ast.Tuple)):
        return True
    if isinstance(node, ast.IfExp):
        return _sequence(node.body) or _sequence(node.orelse)
    if isinstance(node, ast.BoolOp):
        return any(_sequence(v) for v in node.values)
    if isinstance(node, ast.BinOp):
        return _sequence(node.left) or _sequence(node.right)
    return False


def _pow(a, b):
    """'**' in expressions. Python's integer power is exact, so 9**9**9 would run for hours; as floats it
    overflows at once."""
    return float(a) ** b


class _FloatPow(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.copy_location(ast.Call(ast.Name("_pow", ast.Load()), [node.left, node.right], []), node)
        return node


_GLOBALS = {"__builtins__": {}, "_pow": _pow}
ARGS = ("t", "d", "p", "T", "i")  # what an expression function takes, in order (expr_fn)


def compile_expr(src: str, name: str, allowed: set[str], args: tuple[str, ...] = ()):
    """Compile a scene expression. Only arithmetic, comparisons and calls to the whitelisted
    helpers are accepted (no attributes, subscripts or lambdas), because scene files may come
    from other agents and are evaluated with eval(). With args, the code evaluates to a function
    of those names."""
    try:
        return _compile_expr(src, name, allowed, args)
    except (RecursionError, MemoryError):  # "-" * 50000 + "1", or a sum of 5000 terms
        raise ValueError(f"'{name}': the expression is nested too deeply to read ({len(src)} characters); "
                         "write it shorter, or split it across keyframes") from None


def expr_fn(src: str, name: str, env: dict) -> Callable:
    """The expression as a function of (t, d, p, T, i) whose globals are env's helpers, so a call builds no dict."""
    return eval(compile_expr(src, name, set(env) | set(ARGS), ARGS), dict(env, **_GLOBALS))


def _compile_expr(src: str, name: str, allowed: set[str], args: tuple[str, ...] = ()):
    tree = ast.parse(src, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"'{name}': '{type(node).__name__}' is not allowed in expressions")
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ValueError(f"'{name}': unknown name '{node.id}' in expression. "
                             f"Available: {', '.join(sorted(allowed))}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError(f"'{name}': only direct helper calls are allowed")
        if isinstance(node, ast.BinOp) and (_sequence(node.left) or _sequence(node.right)):
            raise ValueError(f"'{name}': arithmetic works on numbers, not on text or lists")
    tree = _FloatPow().visit(tree)
    if args:
        params = ast.arguments(posonlyargs=[], args=[ast.arg(a) for a in args], kwonlyargs=[], kw_defaults=[],
                               defaults=[])
        tree = ast.Expression(ast.Lambda(params, tree.body))
    return compile(ast.fix_missing_locations(tree), f"<{name}>", "eval")


def _finite(v) -> bool:
    try:
        return math.isfinite(v)
    except OverflowError:  # an int past the float range
        return False


def _show(v) -> str:
    try:
        return f"{float(v):g}"
    except OverflowError:
        return f"a number with {len(str(abs(v)))} digits"


def _tween_secs(raw: dict, key: str, default: float, name: str) -> float:
    v = raw.get(key)
    if v is None:  # left out, or null
        return float(default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not _finite(v):
        raise ValueError(f"{name} tween: {key} is {v!r}; it takes a number of seconds ({default:g} when left out)")
    return float(v)


def compile_prop(name: str, raw, ctx: Ctx) -> Prop:
    if name in NOT_ANIMATABLE:
        return Prop(True, raw)
    if isinstance(raw, str) and raw.startswith("="):
        fn = expr_fn(raw[1:], name, dict(EXPR_ENV, W=ctx.w, H=ctx.h, fps=ctx.fps))

        def fx(t, d, T, i, fn=fn):
            try:
                v = fn(t, d, t / d if d else 0.0, T, i)
            except OverflowError:  # 10.0**400, exp(1000), floor(inf)
                raise ValueError("gives a number too large for a float, not a finite number") from None
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not _finite(v):
                raise ValueError(f"gives {_show(v)}, not a finite number")  # inf blanks a layer; 10**400 is no float
            return _norm(v)
        return Prop(False, raw, fn=fx)  # value keeps the source: Scene.check_expressions finds expressions by it
    if isinstance(raw, dict) and ("to" in raw or "keys" in raw):
        if "keys" in raw:
            return compile_prop(name, raw["keys"], ctx)
        delay, dur = (_tween_secs(raw, k, default, name) for k, default in (("delay", 0), ("dur", 0.6)))
        keys = [[delay, raw.get("from", raw["to"])], [delay + dur, raw["to"], raw.get("ease", "outCubic")]]
        return compile_prop(name, keys, ctx)
    if is_keyframes(raw):
        keys = [(k[0], _norm(length(k[1], name, ctx)), E.get(k[2] if len(k) > 2 else "inOutCubic")) for k in raw]
        fps = ctx.fps
        cache: dict[float, tuple[list, list[float]]] = {}  # layer length -> (keys in time order, their times)

        def sorted_keys(d):
            hit = cache.get(d)
            if hit is None:
                pts = sorted(((key_time(k, d, fps), v, e) for k, v, e in keys), key=lambda x: x[0])
                hit = cache[d] = (pts, [p[0] for p in pts])
            return hit

        def fk(t, d, T, i):
            pts, times = sorted_keys(d)
            if t <= times[0]:
                return pts[0][1]
            j = bisect_right(times, t)  # the first key after t; keys at equal times leave the later one in force
            if j == len(pts):
                return pts[-1][1]
            t0, v0, _ = pts[j - 1]
            t1, v1, e1 = pts[j]
            return lerp(v0, v1, e1((t - t0) / (t1 - t0)))
        return Prop(False, fn=fk, keys=lambda d: sorted_keys(d)[0])
    return Prop(True, _norm(length(raw, name, ctx)))


PLACE = ("x", "y", "z", "rotateX", "rotateY", "rotate", "scale")  # where a copy of a model is (Copies)
PLACE_DEFAULT = {k: (1.0 if k == "scale" else 0.0) for k in PLACE}
_MOVES = {"t", "p", "T"}  # an expression reading none of these gives each copy the same value at every frame


class Copies:
    """A model's "repeat": `n` copies of its mesh, each placed by x, y, z, rotateX, rotateY, rotate and scale and
    coloured by fill. A key is a number (fill: a colour), a list taken in turn (copy i takes item i, round again),
    or an expression over i (the copy, from 0), n and the layer's t, d, p and T. Keys that do not change over time
    are worked out once, so a still grid of 1000 copies costs nothing a frame."""
    __slots__ = ("n", "keys", "env", "still")

    def __init__(self, raw: dict, ctx: Ctx):
        self.n = int(raw.get("count", 1))
        self.env = dict(EXPR_ENV, W=ctx.w, H=ctx.h, fps=ctx.fps, n=self.n)
        self.keys: dict[str, tuple] = {}  # key -> ("expr", code, moves) | ("list", values) | ("const", value)
        self.still: dict[str, list] = {}
        for k in (*PLACE, "fill"):
            v = raw.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.startswith("="):
                try:
                    code = compile_expr(v[1:], f"repeat {k}", set(self.env) | {"t", "d", "p", "T", "i"})
                except SyntaxError as e:
                    raise ValueError(f"repeat {k}: '{v}' is not an expression Kinescript reads ({e.msg})") from None
                self.keys[k] = ("expr", code, bool(_MOVES & set(code.co_names)))
            elif isinstance(v, list):
                self.keys[k] = ("list", [self._value(k, x, j) for j, x in enumerate(v)])
            else:
                self.keys[k] = ("const", self._value(k, v, 0))

    @staticmethod
    def _value(k: str, v, i: int):
        """v as copy i's value of key k: a finite number, or for fill a colour as (r, g, b, a)."""
        if k == "fill":
            if isinstance(v, tuple) and len(v) == 4:
                return v
            if not C.is_color(v):
                raise ValueError(f"repeat fill gives {_show(v)} for copy {i}, not a colour")
            return C.parse(v)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not _finite(v):
            raise ValueError(f"repeat {k} gives {_show(v)} for copy {i}, not a finite number")
        return float(v)

    def _column(self, k: str, t: float, d: float, T: float) -> list:
        kind, *spec = self.keys[k]
        if kind == "const":
            return [spec[0]] * self.n
        if kind == "list":
            return [spec[0][i % len(spec[0])] for i in range(self.n)]
        loc = dict(self.env, t=t, d=d, p=(t / d if d else 0.0), T=T)
        out = []
        for i in range(self.n):
            loc["i"] = i
            try:
                v = eval(spec[0], _GLOBALS, loc)
            except OverflowError:
                raise ValueError(f"repeat {k} gives a number too large for a float for copy {i}") from None
            except (ArithmeticError, ValueError, TypeError) as e:
                raise ValueError(f"repeat {k} failed for copy {i}: {type(e).__name__}: {e}") from None
            out.append(self._value(k, v, i))
        return out

    def at(self, t: float, d: float, T: float = 0.0) -> tuple[dict[str, list], list | None]:
        """Each copy's placement at the layer's time t (scene time T), as a list per key it sets, and each copy's
        fill as (r, g, b, a), or None when it sets none."""
        cols = {}
        for k, spec in self.keys.items():
            if spec[0] == "expr" and spec[2]:
                cols[k] = self._column(k, t, d, T)
            else:
                if k not in self.still:
                    self.still[k] = self._column(k, t, d, T)
                cols[k] = self.still[k]
        return {k: v for k, v in cols.items() if k != "fill"}, cols.get("fill")

    def moves(self) -> bool:
        """Whether a key changes over time."""
        return any(s[0] == "expr" and s[2] for s in self.keys.values())
