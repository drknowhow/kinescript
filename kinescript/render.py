"""Frame-pure renderer: frame n is a function of the scene and n alone."""
from __future__ import annotations

import ctypes
import functools
import math
import os
import re
import sys
import time
import unicodedata
from collections import deque
from itertools import repeat
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import skia

from . import color as C
from . import media, render3d, space, stage, svgpath
from . import morph as M
from . import particles as PT
from . import presets as P
from .presets import IDENT, Mods
from .props import COPY_PAINT, length, noise
from .scene import Layer, Scene, SceneError, uses_depth

ANCHORS = {"center": (0.5, 0.5), "topleft": (0, 0), "top": (0.5, 0), "topright": (1, 0), "left": (0, 0.5),
           "right": (1, 0.5), "bottomleft": (0, 1), "bottom": (0.5, 1), "bottomright": (1, 1)}
FONT_FALLBACK = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
FONT_DIR = Path(__file__).parent / "fonts"
# Space Grotesk and IBM Plex Mono ship with the package (SIL OFL) so default text matches on every machine.
_GROTESK = {400: "SpaceGrotesk-Regular.ttf", 500: "SpaceGrotesk-Medium.ttf", 600: "SpaceGrotesk-SemiBold.ttf",
            700: "SpaceGrotesk-Bold.ttf"}
_PLEX = {400: "IBMPlexMono-Regular.ttf", 500: "IBMPlexMono-Medium.ttf", 600: "IBMPlexMono-SemiBold.ttf",
         700: "IBMPlexMono-Bold.ttf"}
BUNDLED = {"space grotesk": _GROTESK, "sans": _GROTESK, "ibm plex mono": _PLEX, "mono": _PLEX}
COMBINE = {"union": skia.PathOp.kUnion_PathOp, "subtract": skia.PathOp.kDifference_PathOp,
           "intersect": skia.PathOp.kIntersect_PathOp, "xor": skia.PathOp.kXOR_PathOp}
BLENDS = {"normal": skia.BlendMode.kSrcOver, "multiply": skia.BlendMode.kMultiply, "screen": skia.BlendMode.kScreen,
          "overlay": skia.BlendMode.kOverlay, "add": skia.BlendMode.kPlus, "lighten": skia.BlendMode.kLighten,
          "darken": skia.BlendMode.kDarken, "softLight": skia.BlendMode.kSoftLight,
          "difference": skia.BlendMode.kDifference, "color": skia.BlendMode.kColor,
          "hardLight": skia.BlendMode.kHardLight, "colorDodge": skia.BlendMode.kColorDodge,
          "colorBurn": skia.BlendMode.kColorBurn, "exclusion": skia.BlendMode.kExclusion, "hue": skia.BlendMode.kHue,
          "saturation": skia.BlendMode.kSaturation, "luminosity": skia.BlendMode.kLuminosity}
SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kNone)
# Video frames arrive at source size and upscale here. Catmull-Rom interpolates (a 1:1 draw is
# exact); Mitchell, used before, blurred every native-size frame (mean error 14.8 vs 4.7 on a sharp clip).
CUBIC = skia.SamplingOptions(skia.CubicResampler.CatmullRom())
# Seconds ahead that a render drawing forward opens the decoders of the clips coming on (Renderer._prime).
AHEAD = 1.0
_FMT = re.compile(r"\{:([^{}]*)\}")
_GL: dict = {}

# A frame as I420 (BT.709, limited range), computed on the GPU so the readback and the pipe to the
# encoder carry 12 bits a pixel instead of 32, and ffmpeg converts nothing. Each output RGBA pixel
# holds the next four bytes of the I420 buffer (Y plane, then U, then V), so one readPixels of a
# (w/4) x (1.5h) surface is the whole buffer. Chroma is the mean of each 2x2 block.
I420_SKSL = """
uniform shader src;
uniform float W;
uniform float H;
float3 px(float2 q) { return src.eval(q).rgb; }
float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
float byte_at(float b) {
    float wh = W * H;
    if (b < wh) {
        float y = floor(b / W);
        return (16.0 + 219.0 * luma(px(float2(b - y * W + 0.5, y + 0.5)))) / 255.0;
    }
    float k = b - wh;
    float q = wh * 0.25;
    float plane = floor(k / q);
    k -= plane * q;
    float cw = W * 0.5;
    float cy = floor(k / cw);
    float2 o = float2((k - cy * cw) * 2.0, cy * 2.0);
    float3 c = (px(o + float2(0.5, 0.5)) + px(o + float2(1.5, 0.5)) + px(o + float2(0.5, 1.5))
                + px(o + float2(1.5, 1.5))) * 0.25;
    float e = luma(c);
    float d = plane < 0.5 ? (c.b - e) / 1.8556 : (c.r - e) / 1.5748;
    return (128.0 + 224.0 * d) / 255.0;
}
half4 main(float2 p) {
    float b = (floor(p.y) * (W * 0.25) + floor(p.x)) * 4.0;
    return half4(byte_at(b), byte_at(b + 1.0), byte_at(b + 2.0), byte_at(b + 3.0));
}
"""

# Motion blur on the GPU: each sub-frame is drawn into its own surface and one pass averages them, so
# only the result is read back. It is the integer mean (sum // n) of the 8-bit sub-frames, the same
# bytes as averaging their readbacks in numpy; the +0.5 keeps floor() correct when the GPU's division
# lands just under an exact multiple. The pass samples one texture per sub-frame, so up to BLUR_GPU of
# them; more are averaged in numpy.
BLUR_GPU = 16


def _rrect(box, r) -> skia.RRect:
    """box rounded by r: one radius, or four as CSS orders them, [top-left, top-right, bottom-right, bottom-left];
    four that do not fit are scaled down together, as CSS scales them."""
    rect = skia.Rect.MakeXYWH(*box)
    if isinstance(r, (list, tuple)):
        rr = skia.RRect()
        rr.setRectRadii(rect, [skia.Point(max(0.0, float(v)), max(0.0, float(v))) for v in r])
        return rr
    return skia.RRect.MakeRectXY(rect, r, r)


def _outline(paint: skia.Paint, path: skia.Path) -> skia.Path:
    """The area path covers when drawn with paint: a stroke's outline, dashes and all."""
    out = skia.Path()
    paint.getFillPath(path, out)
    return out


def _blur_sksl(n: int) -> str:
    adds = "".join(f"t += floor(float4(s{k}.eval(p)) * 255 + 0.5);" for k in range(n))
    return ("".join(f"uniform shader s{k};" for k in range(n))
            + f"float4 main(float2 p) {{ float4 t = float4(0); {adds} return floor((t + 0.5) / {n}.0) / 255; }}")


# Premultiplied to straight alpha with the integer rounding blurred frames have always had (_straighten).
# A layer warped (Renderer._warped): each pixel taken from up to `amount` of the layer's own px away, in the direction
# the noise's red and green give; m takes the layer's px to the frame's.
WARP_SKSL = """
uniform shader layer;
uniform shader noise;
uniform float4 m;
uniform float amount;
half4 main(float2 p) {
    half4 n = noise.eval(p);
    float2 v = n.a > 0 ? (float2(n.rg) / n.a - 0.5) * 2 * amount : float2(0);
    return layer.eval(p + float2(m.x * v.x + m.z * v.y, m.y * v.x + m.w * v.y));
}
"""

STRAIGHT_SKSL = """
uniform shader src;
float4 main(float2 p) {
    float4 c = floor(float4(src.eval(p)) * 255 + 0.5);
    float3 rgb = c.a > 0 ? min(floor((c.rgb * 255 + floor(c.a * 0.5) + 0.5) / c.a), float3(255)) : float3(0);
    return float4(rgb, c.a) / 255;
}
"""


def _straighten(buf: np.ndarray):
    """Premultiplied RGBA to straight, in place: (c * 255 + a // 2) // a. Alpha 0 becomes black, alpha 255
    is already straight, so only the partly transparent pixels are computed."""
    a = buf[..., 3]
    buf[a == 0] = 0
    part = (a > 0) & (a < 255)
    px = buf[part]
    al = px[:, 3:4].astype(np.uint16)
    px[:, :3] = np.minimum((px[:, :3].astype(np.uint16) * 255 + al // 2) // al, 255)
    buf[part] = px


# An async readback hands its callback the pixels as a PyCapsule around a pointer, valid only during the call.
# Own prototypes, not ctypes.pythonapi's shared ones, whose argtypes other code may set differently.
_cap_name = ctypes.PYFUNCTYPE(ctypes.c_char_p, ctypes.py_object)(("PyCapsule_GetName", ctypes.pythonapi))
_cap_ptr = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi))


def _read_back(job: list, rows: int, res) -> None:
    """Callback of Surface.asyncRescaleAndReadPixels: copy the pixels into job[0] (rows bytes a row) and mark
    job done. An exception is kept for the caller: one raised here would unwind through Skia's C++."""
    try:
        if res is None:
            raise RuntimeError("the GPU readback of a frame failed")
        cap = res.data(0)
        h = job[0].size // rows
        src = (ctypes.c_uint8 * (res.rowBytes(0) * h)).from_address(_cap_ptr(cap, _cap_name(cap)))
        np.copyto(job[0].reshape(h, rows), np.ctypeslib.as_array(src).reshape(h, -1)[:, :rows])
    except BaseException as e:  # noqa: BLE001
        job[2] = e
    job[1] = True


# Layer types drawn from their props, enter and exit alone, which Renderer._holds reads. Footage is one: a sub-frame
# shows its frame's picture (_draw_video). Shaders, captions and models follow the clock in other ways.
HOLDS = {"rect", "circle", "ellipse", "polygon", "path", "line", "arrow", "image", "video", "text", "counter", "group"}
SHAPED = {"rect", "circle", "ellipse", "polygon", "path", "line"}  # drawn by _shape alone

BACKENDS = ("auto", "gpu", "cpu")


def backend_setting() -> str:
    """KINE_BACKEND: auto (the default, the GPU when there is one), gpu (the GPU or an error) or cpu (the raster
    backend). Any other value is an error; KINE_BACKEND=bogus used to mean the GPU."""
    raw = os.environ.get("KINE_BACKEND", "")
    v = raw.strip().lower() or "auto"
    if v not in BACKENDS:
        raise ValueError(f"KINE_BACKEND={raw!r} is not a backend: use cpu, gpu (the GPU, or an error when there is "
                         "none) or auto (the default: the GPU when there is one, else the CPU)")
    return v


def gpu_context():
    """One hidden OpenGL context per process, or None when there is no usable GPU.
    KINE_BACKEND=cpu forces the raster backend; KINE_BACKEND=gpu makes no GPU an error."""
    want = backend_setting()
    if "ctx" not in _GL:
        _GL["ctx"] = None if want == "cpu" else _make_gl()
    if want == "gpu" and _GL["ctx"] is None:
        raise RuntimeError("KINE_BACKEND=gpu, but no OpenGL context could be made on this machine; set it to auto "
                           "or cpu to draw on the CPU")
    return _GL["ctx"]


def _make_gl():
    try:
        import glfw
        if not glfw.init():
            return None
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(16, 16, "kinescript", None, None)
        if not win:
            return None
        glfw.make_context_current(win)
        _GL["win"] = win
        return skia.GrDirectContext.MakeGL()
    except (ImportError, RuntimeError, OSError):
        return None


# ---- text the font cannot draw alone ----------------------------------------------------------------
# A character the layer's font lacks is drawn with the system font that has it (emoji, CJK, Hebrew, Arabic...).
# Runs that need shaping (another font, right-to-left letters, combining marks, emoji sequences) go through
# HarfBuzz (TextBlob.MakeFromShapedText), which joins Arabic letters, places marks and puts right-to-left letters
# in reading order. Plain text keeps drawString, so it draws exactly as before. Words are then ordered by a
# simplified Unicode bidi algorithm (_bidi_order), word by word.
_FALLBACK: dict = {}  # (weight, width, slant, code point) -> typeface or None
_MIRROR = str.maketrans("()[]{}<>«»‹›", ")(][}{><»«›‹")
_COVERS: dict = {}  # (typeface id, code point) -> whether it has a glyph


def _fallback(style: skia.FontStyle, cp: int, emoji: bool = False):
    """The system's choice of typeface for code point cp in this style (a colour emoji font when `emoji`), or None
    when no installed font has it."""
    key = (style.weight(), style.width(), int(style.slant()), cp, emoji)
    if key not in _FALLBACK:
        try:
            _FALLBACK[key] = skia.FontMgr.RefDefault().matchFamilyStyleCharacter(
                "", style, ["und-Zsye"] if emoji else [], cp)
        except (TypeError, AttributeError, RuntimeError):  # a skia-python without system fallback
            _FALLBACK[key] = None
    return _FALLBACK[key]


def _emoji(cluster: str) -> bool:
    """Whether a cluster asks for emoji presentation: emoji code points, or the emoji variation selector."""
    return "\ufe0f" in cluster or "\u200d" in cluster or any(ord(ch) >= 0x1F000 for ch in cluster)


def _covers(tf: skia.Typeface, cp: int) -> bool:
    key = (tf.uniqueID(), cp)
    hit = _COVERS.get(key)
    if hit is None:
        hit = _COVERS[key] = bool(tf.unicharToGlyph(cp))
    return hit


def _face(tf: skia.Typeface, cluster: str) -> skia.Typeface:
    """The typeface a cluster is drawn in: tf, or the system font that has what tf lacks."""
    missing = next((ch for ch in cluster if not _ignorable(ch) and not _covers(tf, ord(ch))), None)
    if missing is None:
        return tf
    style = tf.fontStyle()
    return _fallback(style, ord(missing), _emoji(cluster)) or _fallback(style, ord(missing)) or tf


def _ignorable(ch: str) -> bool:
    """Drawn as nothing (joiners, variation selectors, direction marks), so it needs no glyph of its own."""
    cp = ord(ch)
    return unicodedata.category(ch) == "Cf" or 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF


def _extends(ch: str, prev: str) -> bool:
    """Whether ch belongs to the cluster `prev` (a base letter or emoji and what modifies it)."""
    cp = ord(ch)
    if unicodedata.category(ch) in ("Mn", "Me", "Mc") or _ignorable(ch) or prev.endswith("\u200d"):
        return True
    if 0x1F3FB <= cp <= 0x1F3FF:  # skin tone
        return True
    return 0x1F1E6 <= cp <= 0x1F1FF and len(prev) == 1 and 0x1F1E6 <= ord(prev) <= 0x1F1FF  # flag pair


def clusters(s: str) -> list[str]:
    """s cut into what a reader sees as one character: é written as e + accent, a flag, a family emoji."""
    out: list[str] = []
    for ch in s:
        if out and _extends(ch, out[-1]):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _shapes(ch: str) -> bool:
    """Whether ch needs shaping even in a font that has it: marks, right-to-left letters, joiners, emoji parts."""
    cp = ord(ch)
    return (unicodedata.category(ch) in ("Mn", "Me", "Mc") or unicodedata.bidirectional(ch) in ("R", "AL", "AN")
            or _ignorable(ch) or 0x1F3FB <= cp <= 0x1F3FF or 0x1F1E6 <= cp <= 0x1F1FF)


def _strong(s: str) -> str | None:
    """'R' or 'L' for the first letter with a direction in s; None when s has none (digits, punctuation)."""
    for ch in s:
        b = unicodedata.bidirectional(ch)
        if b in ("R", "AL"):
            return "R"
        if b == "L":
            return "L"
    return None


def joining(s: str) -> bool:
    """Whether s has letters that join their neighbours (Arabic, Syriac, N'Ko, Mongolian...)."""
    return any(unicodedata.bidirectional(ch) == "AL" or 0x1800 <= ord(ch) <= 0x18AF for ch in s)


def _bidi_order(words: list[str], rtl: bool) -> tuple[list[int], list[bool], list[tuple]]:
    """The left-to-right order of a line's words, whether each reads right to left, and what is around each (rtl,
    direction before it, direction after it: see _lay_pieces), by the Unicode bidi algorithm applied to whole
    words: numbers after left-to-right text take its direction, neutral words take the direction around them, and
    runs at a higher level are reversed (rules W7, N1-N2, I1-I2, L2)."""
    n = len(words)
    edge = "R" if rtl else "L"
    types, prev = [], edge
    for w in words:
        s = _strong(w)
        if s is None and any(unicodedata.bidirectional(ch) in ("EN", "AN") for ch in w):
            types.append("L" if prev == "L" else "EN")
        else:
            types.append(s or "N")
            prev = s or prev
    for i, t in enumerate(types):
        if t == "N":
            before = next((types[j] for j in range(i - 1, -1, -1) if types[j] != "N"), edge)
            after = next((types[j] for j in range(i + 1, n) if types[j] != "N"), edge)
            b, a = ("R" if x in ("R", "EN") else "L" for x in (before, after))
            types[i] = b if a == b else edge
    side = ["R" if t in ("R", "EN") else "L" for t in types]
    last = [_strong(w[::-1]) or d for w, d in zip(words, side)]
    ctx = [(rtl, last[i - 1] if i else edge, side[i + 1] if i + 1 < n else edge) for i in range(n)]
    return (*_visual(types, rtl), ctx)


def _visual(types: list[str], rtl: bool) -> tuple[list[int], list[bool]]:
    """Left-to-right order of items resolved to L, R or EN, in a right-to-left context or not, and whether each
    reads right to left: their levels (I1, I2), then each run at a level or higher reversed (L2)."""
    n = len(types)
    base = 1 if rtl else 0
    levels = [base + {"L": base, "R": 1 - base, "EN": 2 - base}[t] for t in types]
    order = list(range(n))
    for lv in range(max(levels, default=0), 0, -1):
        i = 0
        while i < n:
            if levels[order[i]] >= lv:
                j = i
                while j < n and levels[order[j]] >= lv:
                    j += 1
                order[i:j] = order[i:j][::-1]
                i = j
            else:
                i += 1
    return order, [lv % 2 == 1 for lv in levels]


_ICU = {"tried": False}


def _shape(text: str, font: skia.Font, ltr: bool):
    """text shaped by HarfBuzz into a blob whose origin is the top of the line. The first call makes Skia look for
    ICU data beside the Python executable, which a venv on Windows does not have, and print a line about it to
    stderr; the shaping used here does not need ICU, so that line is kept off the console."""
    if _ICU["tried"]:
        return skia.TextBlob.MakeFromShapedText(text, font, ltr)
    _ICU["tried"] = True
    try:
        saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return skia.TextBlob.MakeFromShapedText(text, font, ltr)
    try:
        sys.stderr.flush()
        os.dup2(null, 2)
        return skia.TextBlob.MakeFromShapedText(text, font, ltr)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)


def _text_font(tf: skia.Typeface, size: float) -> skia.Font:
    f = skia.Font(tf, size)
    f.setSubpixel(True)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    return f


@dataclass
class Piece:
    """Part of a run drawn with one font: by drawString (blob None) or as a shaped blob."""
    text: str
    font: skia.Font
    blob: object
    width: float
    drop: float  # a shaped blob's origin is the top of its line: draw it this far above the baseline


@dataclass
class Run:
    text: str
    x: float
    line: int
    emph: bool
    word: int
    char0: int
    rtl: bool = False  # reads right to left: its pieces go right to left
    ctx: tuple | None = None  # (paragraph reads right to left, direction before, after): see _lay_pieces


@dataclass
class Layout:
    runs: list
    lines: list           # widths
    w: float
    h: float
    lh: float
    baseline0: float
    nchars: int
    nwords: int
    units: list | None = None  # _draw_text's units, worked out on first use


def color_matrix(fx: dict) -> list[float] | None:
    """4x5 row-major colour matrix from brightness/contrast/saturate/hue/exposure/temperature/tint/invert/gray."""
    if not fx:
        return None
    m = np.eye(5)

    def apply(mat):
        nonlocal m
        m = mat @ m
    if fx.get("exposure"):
        k = 2 ** float(fx["exposure"])
        apply(np.diag([k, k, k, 1, 1]))
    if fx.get("brightness"):
        b = float(fx["brightness"])
        e = np.eye(5); e[:3, 4] = b
        apply(e)
    if fx.get("contrast") not in (None, 1):
        c = float(fx["contrast"])
        e = np.diag([c, c, c, 1, 1]); e[:3, 4] = 0.5 * (1 - c)
        apply(e)
    s = fx.get("saturate", fx.get("saturation"))
    if fx.get("gray"):
        s = 1 - float(fx["gray"])
    if s not in (None, 1):
        s = float(s)
        lr, lg, lb = 0.2126, 0.7152, 0.0722
        e = np.eye(5)
        e[:3, :3] = [[lr * (1 - s) + s, lg * (1 - s), lb * (1 - s)],
                     [lr * (1 - s), lg * (1 - s) + s, lb * (1 - s)],
                     [lr * (1 - s), lg * (1 - s), lb * (1 - s) + s]]
        apply(e)
    if fx.get("hue"):
        a = math.radians(float(fx["hue"]))
        cs, sn = math.cos(a), math.sin(a)
        e = np.eye(5)
        e[:3, :3] = [[0.213 + cs * 0.787 - sn * 0.213, 0.715 - cs * 0.715 - sn * 0.715, 0.072 - cs * 0.072 + sn * 0.928],
                     [0.213 - cs * 0.213 + sn * 0.143, 0.715 + cs * 0.285 + sn * 0.140, 0.072 - cs * 0.072 - sn * 0.283],
                     [0.213 - cs * 0.213 - sn * 0.787, 0.715 - cs * 0.715 + sn * 0.715, 0.072 + cs * 0.928 + sn * 0.072]]
        apply(e)
    if fx.get("temperature") or fx.get("tint"):
        k, g = float(fx.get("temperature", 0)), float(fx.get("tint", 0))
        apply(np.diag([1 + 0.12 * k, 1 + 0.08 * g, 1 - 0.12 * k, 1, 1]))
    if fx.get("invert"):
        e = np.diag([-1, -1, -1, 1, 1]).astype(float); e[:3, 4] = 1
        apply(e)
    return [float(v) for v in m[:4, :].flatten()]


def loop_end(L: Layer, duration: float) -> float:
    """Where a looping clip goes back to its trim start: the trim end, or the end of the file. Before, a trim end
    was ignored and [1, 3] looped 1 s to the end of the file."""
    trim = L.raw.get("trim") or [0]
    return min(duration, float(trim[1])) if len(trim) > 1 and trim[1] is not None else duration


def _lineage(L: Layer | None):
    """L and the layers it sits in."""
    while L is not None:
        yield L
        L = L.parent


def _see_through(bg) -> bool:
    """Whether the canvas can end up less than opaque: no background, or a partly transparent colour or gradient."""
    if bg in (None, "transparent", "none"):
        return True
    cols = [s if isinstance(s, str) else s[1] for s in bg.get("stops", [])] if isinstance(bg, dict) else [bg]
    try:
        return any(C.parse(c)[3] < 1 for c in cols if isinstance(c, str))
    except ValueError:
        return False


def _argb(c, alpha: float = 1.0) -> int:
    if c is None:
        return 0
    if isinstance(c, str):
        c = C.parse(c)
    return C.to_argb_int(c, alpha)


def declare_uniforms(src: str, uniforms: dict) -> str:
    """Add `uniform` lines for built-ins (iTime, iResolution, iProgress) and for entries of the layer's
    "uniforms" that the source uses without declaring, so a shader needs only its main()."""
    decl = []
    wanted = {"iTime": "float", "iResolution": "float2", "iProgress": "float"}
    for name, v in uniforms.items():
        n = len(v) if isinstance(v, (list, tuple)) else 4 if C.is_color(v) else 1
        wanted[name] = "float" if n == 1 else f"float{n}"
    for name, typ in wanted.items():
        if re.search(rf"\b{name}\b", src) and not re.search(rf"\buniform\s+\w+\s+{name}\b", src):
            decl.append(f"uniform {typ} {name};")
    return "\n".join(decl + [glsl_to_sksl(src)]) if decl else glsl_to_sksl(src)


def glsl_to_sksl(src: str) -> str:
    """Accept the two shapes models usually write instead of SkSL's `half4 main(float2 p)`:
    Shadertoy's mainImage(out vec4 fragColor, in vec2 fragCoord), and GLSL's void main() with
    gl_FragCoord / gl_FragColor (or a vec4 main() that returns the colour)."""
    if re.search(r"\b(half4|float4|vec4)\s+main\s*\(\s*(float2|vec2)", src):
        return src
    if re.search(r"\bvoid\s+mainImage\s*\(", src):
        return src + "\nhalf4 main(float2 p) { vec4 c = vec4(0); mainImage(c, p); return half4(c); }"
    m = re.search(r"\b(void|vec4|half4|float4)\s+main\s*\(\s*(void)?\s*\)\s*\{", src)
    if not m:
        return src
    depth, end = 1, m.end()
    while end < len(src) and depth:
        depth += {"{": 1, "}": -1}.get(src[end], 0)
        end += 1
    body = src[m.end():end - 1].replace("gl_FragCoord", "_kpos").replace("gl_FragColor", "_kcol")
    if m.group(1) == "void":
        body = "vec4 _kcol = vec4(0);" + body + "return half4(_kcol);"
    return src[:m.start()] + "half4 main(float2 _kpos) {" + body + "}" + src[end:]


# Largest frame side. 16384 is the biggest texture most GPUs take, and a 16384-pixel square is 1 GB of RGBA;
# --scale 100 on 1080p asked for 192000x108000 and failed inside Skia.
MAX_SIDE = 16384
# What the CPU takes to blur one pixel of an offscreen layer with this Skia wheel: 0.44 s for a 1080p layer.
# Renderer.filtered counts the pixels a frame's offscreen effects blur, or do as much work as blurring would.
CPU_BLUR_S = 0.44 / (1920 * 1080)
# What the CPU takes to draw a particle, and each pixel of the square its sprite covers: 5,000 particles of 1 px took
# 1.6 ms a frame, of 12 px 53 ms and of 24 px 197 ms. Renderer.particle_s adds them up for the frame.
CPU_PARTICLE_S, CPU_PARTICLE_PX_S = 0.3e-6, 70e-9


def frame_size(w: int, h: int, scale) -> tuple[int, int]:
    """The size a w x h canvas renders at `scale`; ValueError when that is not a frame that can be drawn."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a number above 0, got {scale!r}")
    fw, fh = max(2, round(w * scale)), max(2, round(h * scale))
    if max(fw, fh) > MAX_SIDE:
        raise ValueError(f"scale {scale:g} makes a {fw}x{fh} frame; the largest side is {MAX_SIDE} px")
    return fw, fh


class Renderer:
    def __init__(self, scene: Scene, track: bool = False, scale: float = 1.0, backend: str = "auto",
                 even: bool = False, alpha: bool = True):
        """alpha=False: the output keeps no alpha (.mp4, .gif), so frames come premultiplied, which is what is
        see-through shown over black."""
        self.s = scene
        self.scale = scale
        self.w, self.h = frame_size(scene.w, scene.h, scale)
        if even:  # 4:2:0 video needs even sizes
            self.w, self.h = self.w - self.w % 2, self.h - self.h % 2
        self.info = skia.ImageInfo.Make(self.w, self.h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        ctx = gpu_context() if backend in ("auto", "gpu") else None
        if backend == "gpu" and ctx is None:
            raise RuntimeError("GPU backend requested but no OpenGL context could be created")
        self.surface = skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, self.info) if ctx else None
        self.backend = "gpu" if self.surface is not None else "cpu"
        self.gr = ctx if self.backend == "gpu" else None
        if self.surface is None:
            self.surface = skia.Surface.MakeRaster(self.info)
        self.buf = np.empty((self.h, self.w, 4), np.uint8)
        # Encoders take straight (unpremultiplied) RGBA; reading premultiplied pixels into a transparent
        # .mov/.webm darkens every soft edge. Opaque frames are identical either way.
        self.transparent = alpha and _see_through(scene.bg)
        self.read_info = self.info.makeAlphaType(skia.kUnpremul_AlphaType) if self.transparent else self.info
        self.track = track
        # False: video layers are placed (box, stroke) but no frame is decoded, for checks that read boxes, not pixels.
        self.footage = True
        self.boxes: dict[str, dict] = {}
        self._tf: dict = {}
        self._layouts: dict = {}
        self._fonts: dict = {}
        self._shaped: dict = {}  # (typeface, size, text, rtl) -> how _draw_pieces draws it (_pieces)
        self._paths: dict = {}
        self._roughs: dict = {}  # layer id: (seed, the path it wobbled, the wobbled path), the last one made
        # While a combining group draws its children: into its space, and the outlines they give instead of drawing
        self._capture: tuple[skia.Matrix, list] | None = None
        self._matte: Layer | None = None  # the mask being drawn (_masked): layers drawn while it is are not seen as such
        self._cutting: Layer | None = None  # the masked layer being recorded, drawn as itself inside _masked
        self._warping: Layer | None = None  # the displaced layer being recorded, drawn as itself inside _warped
        self._looking: Layer | None = None  # the layer with effects being recorded (_looked), drawn without its layer
        self._look: tuple | None = None  # the paint of that layer and the matrix it applies in, once it is drawn
        self._warp_effect = None
        self._combos: dict = {}  # a combining group's id: (the outlines it merged, the merged path), the last one
        self._morphs: dict = {}  # (layer id, key): that key and the next matched point for point (morph.match)
        self._morphed: dict = {}  # layer id: (how far through its keys, the path there), the last one drawn
        self._routes: dict[str, tuple] = {}  # an along's path by layer (data) or route id: (path, measure, length)
        self._group_T: dict[str, float] = {}  # a group with a clock: the scene time it is drawn at, for its kids
        self._solid: dict[int, skia.Paint] = {}  # an antialiased fill paint for each colour (_fill_paint)
        self._shaders: dict = {}
        self.stop_at: str | None = None
        self.stop_after: str | None = None
        self._stopped = False
        self._m3d: np.ndarray | None = None  # world-to-canvas matrix of the layer being drawn, in 3D scenes
        self.culled: set[str] = set()  # layers skipped this frame because they cross the camera plane
        self.unlit: set[str] = set()  # models drawn this frame with the key light behind them
        self.fogged: set[str] = set()  # layers drawn this frame wholly hidden by fog
        self.inside: set[str] = set()  # models the camera is inside this frame
        self.swim: set[str] = set()  # textured models whose triangles stayed too large to map the image this frame
        self.unfocused = False  # the camera's focus plane was nearer or farther than every layer this frame
        self.slow_dof = False  # a stage blurred per pixel on the CPU this frame
        self.filtered = 0.0  # px blurred by offscreen effects this frame, or their like in work (CPU_BLUR_S)
        self._dof = None  # (focus distance, aperture, z 0's distance) while the camera blurs (render3d.defocus)
        self._focused = None  # _dof while drawing a layer the camera moves, else None
        self._span = [math.inf, 0.0]  # the least and most w of what the camera moves, drawn this frame
        self._lit: dict = {}  # light objects with their tweening keys compiled (render3d.at)
        self._cam: np.ndarray | None = None  # the camera matrix of the top-level layer being drawn
        self._lens: np.ndarray | None = None  # camera space to canvas, for lighting models
        self._models: dict = {}  # merged parts of each model layer
        self._solids: dict = {}  # outlines and extruded meshes of text and path models
        self._outer = None  # the 3D matrix of the layer being drawn's parent (models put their floor in it)
        self._msurf = None  # offscreen that models draw into, antialiased, before compositing
        self._mspare: dict = {}  # more offscreens like it, for the passes of textured models
        self._stages = None  # surfaces of groups with "sort": "pixel" (stage.Stage)
        self._stage = None  # the stage whose child is being drawn
        self._dev = 1.0  # stage px per device px of the canvas being drawn on (models draw in device px)
        self.tris = 0  # model triangles drawn this frame
        self._subframes = None  # motion blur on the GPU: (count, averaging effect, a surface per sub-frame)
        self._under = None  # motion blur on the GPU: the layers every sub-frame shares (_held)
        self._straight = None  # (effect, surface) straightening blurred transparent frames on the GPU
        self._frame_t: float | None = None  # while drawing a motion-blur sub-frame, the time of its frame
        self._blended: dict[str, bool] = {}  # group id -> whether a layer inside it has a blend mode
        self._ink: skia.Rect | None = None  # track: device bounds of the strokes the layer being drawn made
        self._alpha = 1.0  # the opacity of the layer being drawn when its paint carries it (_one_draw), else 1
        self._cf: skia.ColorFilter | None = None  # likewise its fog
        self._copy: tuple | None = None  # while a layer's copies draw (_copies): (the layer, the copy's own colours)
        self._ops: list | None = None  # while a shape's copies are made ready: what _shape would draw
        self._copying = 0  # how many layers' copies are drawing, one inside another: their boxes join (_note)
        self._tinted: dict[tuple, tuple] = {}  # (layer id, fill or stroke) -> its copies' still colours (_tints)
        self._sprite: dict[str, tuple] = {}  # particles layer id -> its sprite (_sprites)
        self._emitter: dict[str, tuple | None] = {}  # particles layer id -> where its emitter is over time (_emitted)
        self.particle_s = 0.0  # the CPU's time for the particles drawn this frame (CPU_PARTICLE_S)
        cm = color_matrix({k: v for k, v in scene.grade.items() if k not in ("vignette", "grain")})
        self.grade_filter = skia.ColorFilters.Matrix(cm) if cm else None
        self._videos = [L for L in scene.all_layers() if L.type == "video"]
        self._on: float | None = None  # the frame time the last draw showed, to tell a render drawing forward
        self._clips: dict = {}  # video textures: (model id, src, timing) -> (model layer, decoder)
        self._insets: set[str] = set()  # the groups being drawn as textures (_surface)
        # kine look --wire and --depth: keep each model's drawn triangle edges (frame px, (n, 2, 2)) and each
        # stage's depth image (frame px, 1/w over stage.QMAX, 0 where nothing is) as the frame draws.
        self.wire = self.depth_view = False
        self.wires: list[np.ndarray] = []
        self.depths: list[np.ndarray] = []

    # ---- public -------------------------------------------------------------------------
    def frame(self, n: int) -> np.ndarray:
        """RGBA uint8 array (h, w, 4) for frame n. The array is reused between calls."""
        if self._compose(n) > 1 and self.transparent:
            # A blurred frame keeps the integer rounding it has always been straightened with.
            if self.backend == "gpu":
                self._straightened().readPixels(self.info, self.buf, self.w * 4)
            else:
                self.surface.readPixels(self.info, self.buf, self.w * 4)
                _straighten(self.buf)
        else:
            self.surface.readPixels(self.read_info, self.buf, self.w * 4)
        return self.buf

    def _compose(self, n: int) -> int:
        """Draw frame n into the surface: with motion blur, the mean of its sub-frames. Returns how many."""
        mb = self.s.motion_blur
        samples = int(mb.get("samples", 0)) if mb else 0
        T = n / self.s.fps
        if samples < 2:
            self.draw(T)
            return 1
        shutter = float(mb.get("shutter", 0.5)) / self.s.fps
        # Inside the scene: half of frame 0's samples fell before 0 s, where nothing is drawn, so it came out half dark.
        # Each layer also keeps its sub-frames inside its own time (see _layer).
        last = (self.s.frames - 1) / self.s.fps
        times = [min(last, max(0.0, T + (k / (samples - 1) - 0.5) * shutter)) for k in range(samples)]
        held = self._held(times, T)
        if held == len(self.s.layers):  # nothing moves while the shutter is open: every sub-frame is the mean
            self.draw(times[0], frame_t=T)
        elif self.backend == "gpu" and samples <= BLUR_GPU:
            self._blur_gpu(times, T, held)
        else:
            self._blur_cpu(times, T, held)
        # Grain and vignette go on once, on the blurred frame: grain drawn on every sub-frame was averaged
        # down to about 70% of its strength.
        c = self.surface.getCanvas()
        c.resetMatrix()
        if self.scale != 1:
            c.scale(self.scale, self.scale)
        self._finish(c, T)
        return samples

    def _blur_gpu(self, times: list[float], T: float, held: int):
        n = len(times)
        ctx = self.surface.recordingContext()
        if self._subframes is None or self._subframes[0] != n:
            self._subframes = (n, skia.RuntimeEffect.MakeForShader(_blur_sksl(n)),
                               [skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, self.info) for _ in range(n)])
        _, eff, surfs = self._subframes
        main, under = self.surface, None
        try:
            if held:  # the layers every sub-frame shares, drawn once in a surface of their own
                if self._under is None:
                    self._under = skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, self.info)
                self.surface = self._under  # draw() draws into self.surface
                self.draw(times[0], frame_t=T, held=held)
                under = self._under.makeImageSnapshot()
            for t, surf in zip(times, surfs):
                self.surface = surf
                self.draw(t, frame_t=T, held=held, under=under)
        finally:
            self.surface = main
        del under
        imgs = [s.makeImageSnapshot() for s in surfs]
        b = skia.RuntimeEffectBuilder(eff)
        for k, img in enumerate(imgs):
            b.setChild(f"s{k}", img.makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, skia.SamplingOptions()))
        c = main.getCanvas()
        c.resetMatrix()
        c.drawPaint(skia.Paint(Shader=b.makeShader(), BlendMode=skia.BlendMode.kSrc))
        del imgs, b  # as in frame_i420: a live snapshot makes the next draw into its surface copy it first

    def _straightened(self) -> skia.Surface:
        """A surface holding this surface's frame with straight alpha (STRAIGHT_SKSL)."""
        if self._straight is None:
            self._straight = (skia.RuntimeEffect.MakeForShader(STRAIGHT_SKSL), skia.Surface.MakeRenderTarget(
                self.surface.recordingContext(), skia.Budgeted.kNo, self.info))
        eff, surf = self._straight
        img = self.surface.makeImageSnapshot()
        b = skia.RuntimeEffectBuilder(eff)
        b.setChild("src", img.makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, skia.SamplingOptions()))
        surf.getCanvas().drawPaint(skia.Paint(Shader=b.makeShader(), BlendMode=skia.BlendMode.kSrc))
        del img, b
        return surf

    def _blur_cpu(self, times: list[float], T: float, held: int):
        # 257 sub-frames of 255 still fit in 16 bits: half the memory traffic of 32-bit sums, same mean.
        acc = np.zeros((self.h, self.w, 4), np.uint16 if len(times) <= 257 else np.uint32)
        under = None
        if held:
            self.draw(times[0], frame_t=T, held=held)
            under = self.surface.makeImageSnapshot()
        for t in times:
            self.draw(t, frame_t=T, held=held, under=under)
            self.surface.readPixels(self.info, self.buf, self.w * 4)
            np.add(acc, self.buf, out=acc)
        del under
        np.floor_divide(acc, len(times), out=acc)
        np.copyto(self.buf, acc, casting="unsafe")
        self.surface.writePixels(skia.Pixmap(self.info, self.buf, self.w * 4))  # stills save the surface

    def _held(self, times: list[float], F: float) -> int:
        """How many of the scene's layers, from the bottom, draw the same on every sub-frame of the frame at F. They
        are drawn once and the sub-frames start from them: a title that has landed is not drawn again for each sample
        of a dot moving over it. Layers above a moving one stay in the sub-frames, which blend over it."""
        if self.track or self.s.is3d:  # boxes come from each draw; a camera moves everything
            return 0
        lo, hi = min(times), max(times)
        k = 0
        for L in self.s.layers:
            if L.id not in self.s.mattes and not self._holds(L, lo, hi, F):  # a mask holds where its layer does
                break
            k += 1
        return k

    def _holds(self, L: Layer, lo: float, hi: float, F: float) -> bool:
        """Whether L draws the same at every sub-frame time in [lo, hi] of the frame at F, by what _layer reads: its
        props, its enter and exit, and its children's."""
        if not L.visible(F):
            return True
        if L.type not in HOLDS or L.clock is not None or (L.copies is not None and L.copies.moves()):
            return False
        end = math.nextafter(L.t_out, -math.inf)
        lo, hi = min(max(lo, L.t_in), end), min(max(hi, L.t_in), end)  # as _layer clamps a sub-frame
        a, b, d = lo - L.t_in, hi - L.t_in, L.dur
        if L.enter:
            done = L.enter.dur
            if L.type == "group" and L.enter.stagger and L.children:
                done += max(k.turn for k in L.children)
            if L.type in ("text", "counter") and L.raw.get("split"):
                if L.type == "counter":  # its units come from a value that may change
                    return False
                # Units enter `stagger` apart (0.04 s unless set, as _draw_text has it); a unit is at least a character.
                done += (L.enter.stagger or 0.04) * len(self._text_of(L, a, lo, None))
            if a < done:
                return False
        if L.exit and d - b < L.exit.dur:
            return False
        route = self.s.by_id[L.along[0][1:]] if L.along and L.along[0].startswith("#") else None
        if route is not None and not all(p.static for p in route.props.values()):
            return False  # it rides a path that moves
        if L.mask is not None and not self._holds(L.mask[0], lo, hi, F):
            return False
        if L.warp is not None and float(L.warp.get("speed", 1)):
            return False  # its waves slide
        return all(p.held(a, b, d) for p in L.props.values()) and all(self._holds(k, lo, hi, F) for k in L.children)

    @property
    def yuv(self) -> bool:
        """Whether frame_i420 runs on the GPU: an opaque frame with width a multiple of 4."""
        return self.backend == "gpu" and not self.transparent and self.w % 4 == 0 and self.h % 2 == 0

    def frame_i420(self, n: int) -> np.ndarray:
        """Frame n as an I420 buffer (BT.709, limited range), reused between calls. Needs self.yuv."""
        _, info, surf, buf = self._to_i420(n)
        surf.readPixels(info, buf, self.w)
        return buf

    def frames_i420(self, frames):
        """frame_i420 of each of frames, in order; each buffer is valid until the next one is asked for. A frame's
        readback is queued on the GPU and collected once the next frame is drawn, so the GPU draws and reads back
        one frame while Python draws the next. A blocking readPixels waited for both; its copy alone took 1.6 ms of
        a 2.8 ms frame of a simple 1080p scene. The bytes are readPixels' bytes."""
        pending: deque = deque()  # [buffer, done, error] of each frame whose readback is in flight, oldest first
        bufs = None
        try:
            for k, n in enumerate(frames):
                _, info, surf, buf = self._to_i420(n)
                if bufs is None:
                    bufs = (buf, np.empty_like(buf))
                job = [bufs[k % 2], False, None]  # the other buffer is the one last handed out
                done = functools.partial(_read_back, job, self.w)
                surf.asyncRescaleAndReadPixels(info, skia.IRect.MakeWH(info.width(), info.height()),
                                               skia.Surface.RescaleGamma.kSrc, done)
                self.gr.submit()
                pending.append(job)
                if len(pending) > 1:
                    yield self._collect(pending.popleft())
            while pending:
                yield self._collect(pending.popleft())
        finally:
            if pending:  # stopped early: no callback may write into these buffers later
                self.gr.submit(skia.GrSyncCpu.kYes)
                self.gr.checkAsyncWorkCompletion()

    def _collect(self, job: list) -> np.ndarray:
        """Wait for job's readback. Usually done already: callbacks run in the submit after the next frame's draw.
        A blocking submit would also wait for that next frame; sleep(0) hands the core to an encoder thread."""
        while not job[1]:
            time.sleep(0)
            self.gr.checkAsyncWorkCompletion()
        if job[2] is not None:
            raise job[2]
        return job[0]

    def _to_i420(self, n: int) -> tuple:
        """Draw frame n and convert it into the I420 surface, not yet read back. Returns self._i420."""
        if not hasattr(self, "_i420"):
            info = skia.ImageInfo.Make(self.w // 4, self.h * 3 // 2, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
            surf = skia.Surface.MakeRenderTarget(self.surface.recordingContext(), skia.Budgeted.kNo, info)
            self._i420 = (skia.RuntimeEffect.MakeForShader(I420_SKSL), info, surf,
                          np.empty(self.w * self.h * 3 // 2, np.uint8))
        eff, info, surf, buf = self._i420
        self._compose(n)
        img = self.surface.makeImageSnapshot()
        b = skia.RuntimeEffectBuilder(eff)
        b.setChild("src", img.makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, skia.SamplingOptions()))
        b.setUniform("W", float(self.w))
        b.setUniform("H", float(self.h))
        surf.getCanvas().drawPaint(skia.Paint(Shader=b.makeShader(), BlendMode=skia.BlendMode.kSrc))
        # Drop the snapshot before the next draw: while it lives, drawing to the surface copies the
        # whole texture first (copy-on-write), which cost ~80 ms a frame.
        del img, b
        return self._i420

    def snapshot(self) -> skia.Image:
        return self.surface.makeImageSnapshot()

    def draw(self, T: float, stop_at: str | None = None, frame_t: float | None = None, stop_after: str | None = None,
             held: int = 0, under: skia.Image | None = None):
        """Draw the scene at T. stop_at / stop_after: stop before / after drawing that layer (no grade finish).
        frame_t: T is a motion-blur sub-frame of the frame at frame_t, so only the layers on at frame_t draw, each at
        a time inside its own range, and grain and vignette are left to the caller.
        held: the background and the first `held` layers draw the same on every sub-frame (_held). Without `under`,
        only they draw, ungraded; `under` is an image of that, which the sub-frame starts from in their place."""
        s = self.s
        self.boxes = {}
        self.culled, self.unlit, self.fogged, self.inside, self.swim = set(), set(), set(), set(), set()
        self.unfocused = self.slow_dof = False
        self.filtered = 0.0
        self._dof, self._span = None, [math.inf, 0.0]
        self.tris = 0
        self.particle_s = 0.0
        self.wires, self.depths = [], []
        self.stop_at, self.stop_after, self._stopped = stop_at, stop_after, False
        self._frame_t = frame_t
        c = self.surface.getCanvas()
        c.resetMatrix()
        c.clear(skia.ColorTRANSPARENT)
        if self.scale != 1:
            c.scale(self.scale, self.scale)
        layers = s.layers if not held else s.layers[held:] if under is not None else s.layers[:held]
        if self.grade_filter:
            # The background goes inside the graded layer, so the grade covers the whole frame and a blend mode at
            # the top level works on the background: in a layer that started empty, a multiply drew as its plain fill.
            # The held layers draw ungraded, but in a layer all the same: a draw comes out a level apart in a few
            # pixels in a layer and straight on a surface (an upscaled clip did).
            c.saveLayer(None, skia.Paint(ColorFilter=self.grade_filter) if not held or under is not None else None)
        if under is None:
            self._background(c)
        else:  # pixel for pixel, into the graded layer, as if the held layers had been drawn here
            c.save()
            c.resetMatrix()
            c.drawImage(under, 0, 0, skia.SamplingOptions(), skia.Paint(BlendMode=skia.BlendMode.kSrc))
            c.restore()
        on = T if frame_t is None else frame_t
        if self.footage and self._videos:
            last, self._on = self._on, on
            self._prime(on, AHEAD if last is not None and 0 < on - last <= AHEAD else 0.0)
        cams = self._cameras(T) if s.is3d else None
        for layer in layers:
            if self._stopped:
                break
            m = None
            if cams is not None:
                fixed = layer.raw.get("fixed", layer.type == "captions")
                if s.camera is not None and not fixed:
                    m = cams[0]
                elif layer.id in s.depth:
                    m = cams[1]
            self._m3d = self._cam = m
            self._focused = self._dof if m is not None and m is cams[0] and s.camera is not None else None
            if m is None:
                self._layer(c, layer, T, 1.0)
            else:
                c.save()
                c.concat(space.to_skia(m))
                self._layer(c, layer, T, 1.0)
                c.restore()
        self._m3d = self._focused = None
        if self._dof is not None and self._span[1] > 0:
            near = self._dof[0] / self._dof[2]
            self.unfocused = near < self._span[0] * 0.99 or near > self._span[1] * 1.01
        if self.grade_filter:
            c.restore()
        if not (stop_at or stop_after) and frame_t is None:
            self._finish(c, T)
        # A clip's decoder holds an ffmpeg process and its prefetched frames; once the clip is off screen,
        # free them, or a cut list keeps every clip it has played open to the end (60 cuts: 1.6 GB).
        for L in self._videos:
            if L.video is not None and L.video.proc is not None and not L.t_in <= on < L.t_out:
                L.video.close()
            if L.ahead is not None and not (L.t_in <= on + AHEAD and on < L.t_out):  # as _prime reckons it
                L.ahead[1].close()
                L.ahead = None
        for L, v in self._clips.values():
            if v.proc is not None and not L.t_in <= on < L.t_out:
                v.close()

    def _prime(self, on: float, ahead: float):
        """Open the decoder of each clip shown at frame time `on`, and of each coming on within `ahead` seconds, where
        it will first read, before any is waited on. ffmpeg takes about 0.12 s to give a 720p H.264 clip's first
        frame: opened one by one as each clip drew, the nine clips on video-grid's first frame waited 1 s, and the 60
        cuts of sequence-60 6.5 s of an 11 s render. A clip not drawn yet gets its decoder in L.ahead, which it takes
        when it draws if its box asks for the same size: a clip decodes at the size its first frame drawn gives."""
        for L in self._videos:
            T = max(on, L.t_in)
            if T > on + ahead or not all(p.visible(T) for p in _lineage(L)):
                continue
            t, v = T - L.t_in, L.video
            if v is None:
                g = lambda name, default=None: L.get(name, t, T, 0, default)  # noqa: E731
                size = self._decode_size(L, *self._box(L, t, T, g)[2:])
                if L.ahead is None or L.ahead[0] != size:
                    L.ahead = (size, self._decoder(L, size))
                v = L.ahead[1]
            elif T > on:
                continue  # opened for an earlier frame drawn: a render that went back
            v.prime(self._src_time(L, t, v.info))

    # ---- scene-level ---------------------------------------------------------------------
    def _cameras(self, T: float):
        """(world, fixed) camera matrices at T. Fixed layers share the lens but not the moves."""
        s, cam = self.s, self.s.camera
        fov = float(cam.get("fov", T, T, default=space.DEFAULT_FOV)) if cam else space.DEFAULT_FOV
        fixed = space.camera(s.w, s.h, fov)
        self._lens = space.lens(s.w, s.h, fov)
        if cam is None:
            return fixed, fixed
        g = lambda k: float(cam.get(k, T, T))  # noqa: E731
        target = self._aim(cam, "target", T, lambda v: self._spot(v, T)) if "target" in cam.props else None
        sway = self._shake(cam, T, fov) if "shake" in cam.props else None
        m = space.camera(s.w, s.h, fov, g("x"), g("y"), g("z"), g("rotateX"), g("rotateY"), g("rotate"), target, sway)
        aperture = float(cam.get("aperture", T, T, 0, 0.0))
        if aperture > 0:
            f0 = space.focus(s.h, fov)
            back = f0 + g("z")  # how far the camera is from what it looks at

            def far(v):
                """A focus value as a distance from the camera: a z (px toward the camera from what it looks at),
                or a layer id, the distance of that layer's middle."""
                if isinstance(v, str) and v in s.by_id:
                    return float((m @ np.append(self._spot(v, T), 1.0))[3]) * f0
                return back - float(length(v, "z", s.ctx))
            near = self._aim(cam, "focus", T, far) if "focus" in cam.props else back
            self._dof = (max(float(near), 1.0), aperture, f0)
        return m, fixed

    def _aim(self, cam: Layer, key: str, T: float, place):
        """The camera's target (a point in canvas px) or focus (a distance from the camera) at T: `place` turns one
        of its values, a layer id among them, into that. Keyframes were compiled with each key's index for its value
        (Scene._aim), so here the two keys either side are placed and mixed: a camera can pan or pull focus from a
        point to a layer, or from one moving layer to another."""
        p, d = cam.props[key], cam.dur
        if p.keys is None:
            return place(p.at(T, d, T))
        pts, raw = p.keys(d), cam.raw[key]
        if T <= pts[0][0]:
            return place(raw[pts[0][1]][1])
        for (t0, i0, _), (t1, i1, e1) in zip(pts, pts[1:]):
            if T < t1:
                a, b = place(raw[i0][1]), place(raw[i1][1])
                return a + (b - a) * e1((T - t0) / (t1 - t0) if t1 > t0 else 1.0)
        return place(raw[pts[-1][1]][1])

    def _reach(self, w: np.ndarray):
        """Widen the range of w drawn this frame, which kine check holds the focus plane to."""
        self._span[0], self._span[1] = min(self._span[0], float(w.min())), max(self._span[1], float(w.max()))

    def _spot(self, v, T: float) -> np.ndarray:
        """A target as a point in canvas px: [x, y] or [x, y, z] (lengths as a layer's x, y and z take them), or a
        layer id: the middle of that layer where its keys and its parents put it at T. Its enter and exit motions
        are left out, so a layer that rises in rises on screen rather than taking the camera with it."""
        if not isinstance(v, str):
            v = list(v) + [0.0] * (3 - len(v))
            return np.array([float(length(x, k, self.s.ctx)) for x, k in zip(v, ("x", "y", "z"))])
        L = self.s.by_id[v]
        m = np.eye(4)
        for A in reversed(list(_lineage(L))):
            t = min(max(T - A.t_in, 0.0), A.dur)
            g = lambda name, default=None, A=A, t=t: A.get(name, t, T, 0, default)  # noqa: E731
            sc = float(g("scale"))
            m = (m @ space.translate(float(g("x")), float(g("y"))) @ space.tilt(float(g("z")), float(g("rotateX")),
                 float(g("rotateY"))) @ space.rotate_z(float(g("rotate")))
                 @ space.scale(sc * float(g("scaleX")), sc * float(g("scaleY"))))
        box = self._box(L, t, T, g)
        mid = (0.0, 0.0) if box is None else (box[0] + box[2] / 2, box[1] + box[3] / 2)
        return (m @ np.array([mid[0], mid[1], 0.0, 1.0]))[:3]

    def _shake(self, cam: Layer, T: float, fov: float):
        """The camera's handheld turn at T, in degrees (rx, ry, rz): smooth noise of the scene time, `speed` wobbles
        a second, that moves the view up to about `amount` px, with a little roll. A pure function of T."""
        amount = float(cam.get("shake", T, T))
        if amount <= 0:
            return None
        spec = cam.raw["shake"] if isinstance(cam.raw["shake"], dict) else {}
        speed, seed = float(spec.get("speed", 1.5)), float(spec.get("seed", 0)) * 7
        a = amount / space.focus(self.s.h, fov)  # radians that move z 0 by amount px
        roll = 0.5 * amount / math.hypot(self.s.w / 2, self.s.h / 2)  # the corners move by half of it

        def wave(k):
            u = T * speed
            return 0.7 * noise(u, seed + k) + 0.3 * noise(u * 2.3 + 17.0, seed + k + 3)
        return tuple(math.degrees(r * wave(k)) for k, r in enumerate((a, a, roll)))

    def _far(self, L: Layer, T: float) -> float:
        """Sort key that puts the child farthest from the camera first."""
        t = T - L.t_in
        p = self._m3d @ np.array([float(L.get("x", t, T)), float(L.get("y", t, T)), float(L.get("z", t, T)), 1.0])
        return -p[3]

    def _background(self, c: skia.Canvas):
        bg = self.s.bg
        if isinstance(bg, dict):
            p = skia.Paint(Shader=self._gradient(bg, 0, 0, self.s.w, self.s.h, 1.0))
            c.drawRect(skia.Rect(0, 0, self.s.w, self.s.h), p)
        elif bg:
            c.clear(_argb(bg))

    def _finish(self, c: skia.Canvas, T: float):
        g = self.s.grade
        W, H = self.s.w, self.s.h
        if not (g.get("vignette") or g.get("grain")):
            return
        if self.transparent:
            # The grade goes on a copy of the frame that is put back only where the frame already has alpha
            # (src-atop), so it darkens and grains what is drawn and leaves the empty canvas empty. Drawn
            # straight onto a transparent canvas, the vignette made the corners half-opaque black and the
            # grain gave every pixel some alpha.
            # The record holds a bare pointer to the paint: keep the paint in a name, or it is freed before
            # saveLayer reads it (a segfault on Linux).
            atop = skia.Paint(BlendMode=skia.BlendMode.kSrcATop)
            rec = skia.Canvas.SaveLayerRec(None, atop, skia.Canvas.kInitWithPrevious_SaveLayerFlag)
            c.saveLayer(rec)
        if g.get("vignette"):
            v = float(g["vignette"])
            sh = skia.GradientShader.MakeRadial(skia.Point(W / 2, H / 2), math.hypot(W, H) / 2,
                                                [_argb("#000000", 0), _argb("#000000", 0), _argb("#000000", min(1, v))],
                                                [0.0, 0.55, 1.0])
            c.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=sh))
        if g.get("grain"):
            tiles = media.grain_tiles()
            n = int(T * self.s.fps + 1e-6)  # n / fps * fps can land just under n (frame 123 at 30 fps)
            tile = tiles[n % len(tiles)]
            mtx = skia.Matrix.Translate((n * 37) % 256, (n * 91) % 256)
            p = skia.Paint(Shader=tile.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, SAMPLING, mtx),
                           Alphaf=min(1.0, float(g["grain"])), BlendMode=skia.BlendMode.kOverlay)
            c.drawRect(skia.Rect(0, 0, W, H), p)
        if self.transparent:
            c.restore()

    # ---- layers --------------------------------------------------------------------------
    def _layer(self, c: skia.Canvas, L: Layer, T: float, parent_alpha: float):
        if self.stop_at == L.id:
            self._stopped = True
            return
        if L.id in self.s.surfaces:  # a model's texture: drawn there (_surface), not here
            return
        if L.id in self.s.mattes and L is not self._matte:  # a mask: drawn where it cuts the layer it masks
            return
        if L.mask is not None and self._capture is None and L is not self._cutting:  # a combining group takes it uncut
            self._masked(c, L, T, parent_alpha)
            return
        if self._capture is not None and L.type in ("image", "video", "shader", "model", "captions", "particles"):
            return  # a combining group merges shapes and text only
        F = self._frame_t
        if not L.visible(T if F is None else F):
            return
        if F is not None:
            # A motion-blur sub-frame: the layer is drawn on the frames it is on and nowhere else, and its
            # sub-frames stay inside its own time. Sub-frames straddling its first frame fell before it starts,
            # so that frame came out at half strength, and its sub-frames reached into the frame after it ends.
            T = min(max(T, L.t_in), math.nextafter(L.t_out, -math.inf))
        t = T - L.t_in
        d = L.dur
        g = lambda name, default=None: L.get(name, t, T, 0, default)  # noqa: E731
        if L.warp is not None and self._capture is None and L is not self._warping:
            self._warped(c, L, T, t, g, parent_alpha)
            return
        if L.looks is not None and self._capture is None and L is not self._looking:
            self._looked(c, L, T, parent_alpha)
            return
        split = L.raw.get("split") if L.type in ("text", "counter") else None
        mods = IDENT
        if L.enter and not split and t < L.enter.dur and not (L.type == "group" and L.enter.stagger):
            mods = L.enter.mods(t / L.enter.dur)
        if L.turn is not None:  # its group's enter, from its turn, on the group's own time
            m = L.parent.enter
            p = (self._group_T.get(L.parent.id, T) - L.parent.t_in - L.turn) / m.dur if m.dur else 1.0
            if p < 1:
                mods = mods.combine(m.mods(p))
        if L.exit and d - t < L.exit.dur:
            mods = mods.combine(L.exit.mods((d - t) / L.exit.dur))
        alpha = max(0.0, min(1.0, float(g("opacity")) * mods.alpha))
        if alpha <= 0.002:
            return
        box = self._box(L, t, T, g)
        x, y, rot, sc, sx, sy, blur = L.placed(t, T)
        x, y = float(x) + mods.dx, float(y) + mods.dy
        rot = float(rot) + mods.rotate
        if L.along is not None:
            ax, ay, turn = self._along(L, t, T)
            x, y = x + ax, y + ay
            if L.along[1]:
                rot += turn
        sc = float(sc) * mods.scale
        sx, sy = sc * float(sx), sc * float(sy)
        outer, tilt, facing = self._m3d, None, True
        defocus = 0.0  # the blur the camera's depth of field gives this layer, in its own px
        fg, fogf = None, 0.0  # the fog, and how much of it covers this layer
        if outer is not None:
            z, rx, ry = float(g("z")) + mods.dz, float(g("rotateX")) + mods.rx, float(g("rotateY")) + mods.ry
            m = outer @ space.translate(x, y)
            if z or rx or ry:
                tilt = space.tilt(z, rx, ry)
                m = m @ tilt
            m = m @ space.rotate_z(rot) @ space.scale(sx, sy)
            if box is not None:
                xy, w = space.project(m, box)
                if w.min() < space.NEAR:
                    self.culled.add(L.id)
                    return
                facing = space.facing(xy)
                if not facing and L.raw.get("backface") == "hidden":
                    return
                if self._focused is not None:
                    self._reach(w)
                    if self._stage is None and not L.children:
                        # Skia scales a layer's blur as the layer is scaled on screen: in its own px, the blur is
                        # the one wanted on screen over that scale (the root of its area's).
                        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = xy
                        area = abs((x0 - x2) * (y1 - y3) - (x1 - x3) * (y0 - y2)) / 2
                        defocus = float(render3d.defocus(self._focused, w.mean())) / max(
                            math.sqrt(area / ((box[2] or 1.0) * (box[3] or 1.0))), 0.05)
                if self.s.fog is not None and self._stage is None and not L.children and L.type != "model":
                    # A flat layer is fogged as a whole, by the depth of its middle; a model by each vertex's.
                    fg = render3d.fog(self, T)
                    fogf = 0.0 if fg is None else float(render3d.fogged(fg, w.mean()))
                    if fogf >= 0.999:
                        self.fogged.add(L.id)
            elif L.type == "model" and self._focused is not None and self._stage is None:
                # A model blurs whole, by the depth of its middle, scaled as its own px are there.
                o = m @ np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]).T
                if o[3].min() > space.NEAR:
                    (ax, ay), (bx, by), (cx, cy) = (o[:2] / o[3]).T
                    sc = math.sqrt(abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)))
                    defocus = float(render3d.defocus(self._focused, o[3, 0])) / max(sc, 0.05)
            self._m3d, self._outer = m, outer
            if self._stage is not None and self._stage.child is L:
                self._stage.m = m
        c.save()
        c.translate(x, y)
        if tilt is not None:
            c.concat(space.to_skia(tilt))
        if rot:
            c.rotate(rot)
        if sx != 1 or sy != 1:
            c.scale(sx, sy)
        blur = float(blur) + mods.blur
        if defocus > 0.05:
            blur = math.hypot(blur, defocus)
        shadow = L.raw.get("shadow")
        fx = L.raw.get("fx")
        # A masked or displaced layer blends once it is cut (_masked) or warped (_warped).
        blend = None if L is self._cutting or L is self._warping else L.raw.get("blend")
        effects = blur > 0.05 or shadow or fx or (blend and blend != "normal") or L.looks
        cf = None
        if fogf > 0.002:
            if not effects and self._one_draw(L):
                cf = render3d.fog_filter(fg, fogf)  # in the paint of its one draw, like its opacity
            else:
                effects = True
        layered = alpha < 0.999 or effects
        paint_alpha = 1.0
        if layered and L.children and not effects and self._blends_inside(L):
            # A fading group whose children blend: they keep blending with what is behind the group, and the
            # group mixes from that backdrop to its full result, so blend does not turn into normal (and the
            # frame jump) as opacity drops below 1.
            layered = 2
            self._fade_layers(c, alpha)
        elif layered and not effects and self._one_draw(L):
            # Opacity alone on a layer drawn in one call: the paint carries it, to within 3 levels. An offscreen layer
            # for each fading shape cost far more than the shape: 600 fading circles took 26 ms a frame on the GPU
            # and 45 s on the CPU, now 8 and 15 ms.
            layered, paint_alpha = False, alpha
        elif layered:
            paint = skia.Paint(Alphaf=alpha)
            filt, work, reach = self._looks(L, g, t) if L.looks else (None, 0.0, 0.0)
            if blur > 0.05:
                filt = skia.ImageFilters.Blur(blur, blur, skia.TileMode.kDecal, filt)
                work, reach = work + 1, max(reach, 3 * blur)
            if shadow:
                sh = shadow if isinstance(shadow, dict) else {}
                sig = float(sh.get("blur", 16)) / 2
                dx, dy = float(sh.get("x", 0)), float(sh.get("y", 6))
                filt = skia.ImageFilters.DropShadow(dx, dy, sig, sig, _argb(sh.get("color", "#00000099")), filt)
                work, reach = work + 1.2, max(reach, 3 * sig + max(abs(dx), abs(dy)))  # a blur, moved and merged
            if filt:
                paint.setImageFilter(filt)
            if fx:
                cm = color_matrix(fx)
                if cm:
                    paint.setColorFilter(skia.ColorFilters.Matrix(cm))
            if fogf > 0.002:
                mist, inner = render3d.fog_filter(fg, fogf), paint.getColorFilter()
                paint.setColorFilter(skia.ColorFilters.Compose(mist, inner) if inner else mist)
            if blend:
                paint.setBlendMode(BLENDS.get(blend, skia.BlendMode.kSrcOver))
            if L is self._looking:  # _looked opens the layer, as large as what is drawn now
                self._look, layered = (paint, c.getTotalMatrix(), work, reach), False
            else:
                if work:  # a layer the size of the clip, which is the frame unless a group clips it
                    clip = c.getDeviceClipBounds()
                    self.filtered += work * clip.width() * clip.height()
                c.saveLayer(None, paint)
        if box is not None:
            bx, by, bw, bh = box
            if mods.wipe and mods.reveal < 1:
                if L.copies is not None and L.type != "model":  # its copies wipe on as one
                    bx, by, bw, bh = self._copies_box(L, t, T, box)
                r = max(0.0, mods.reveal)
                clip = {"right": (bx, by, bx + bw * r, by + bh), "left": (bx + bw * (1 - r), by, bx + bw, by + bh),
                        "down": (bx, by, bx + bw, by + bh * r), "up": (bx, by + bh * (1 - r), bx + bw, by + bh)}[mods.wipe]
                c.clipRect(skia.Rect(*clip), doAntiAlias=True)
        reveal = mods.reveal if not mods.wipe else 1.0
        self._ink, self._alpha, self._cf, self._t = None, paint_alpha, cf, t
        if L.copies is not None and L.type != "model":  # a model's copies are merged into its mesh (render3d)
            self._copies(c, L, t, T, g, box, reveal, mods, alpha, parent_alpha, facing)
        else:
            self._body(c, L, t, T, g, box, reveal, mods, alpha, parent_alpha, facing)
        if layered == 2:
            c.restore()
        if layered:
            c.restore()
        c.restore()
        self._m3d = outer
        if self.stop_after == L.id:
            self._stopped = True

    def _body(self, c, L: Layer, t, T, g, box, reveal, mods, alpha, parent_alpha, facing):
        """L drawn on c in the space _layer set up, its box noted (track), then its children if it is a group."""
        drawn = getattr(self, "_draw_" + L.type)(c, L, t, T, g, box, reveal, mods)  # a model gives its bounds
        if self.track and (box is not None or drawn):
            self._note(c, L, g, box, drawn, alpha, parent_alpha, facing)
        if L.type == "group":
            kids = L.children
            frame_t = self._frame_t
            if L.clock is not None:  # the children run on the group's clock, a motion-blur sub-frame's frame too
                self._group_T[L.id] = T
                T = self._inner(L, t, T)
                if frame_t is not None:
                    self._frame_t = self._inner(L, frame_t - L.t_in, frame_t)
            if self._m3d is not None and L.raw.get("sort") == "depth":
                kids = sorted(kids, key=lambda k: self._far(k, T))
            if self._m3d is not None and L.raw.get("sort") == "pixel" and self._stage is None:
                stage.stage(self).draw(self, c, kids, T, parent_alpha * alpha, L.raw.get("shadows"))
            else:
                outer_capture = self._capture
                if L.raw.get("combine"):
                    inv = skia.Matrix()
                    self._capture = (inv, []) if c.getTotalMatrix().invert(inv) else None
                for child in kids:
                    if self._stopped:
                        break
                    self._layer(c, child, T, parent_alpha * alpha)
                if L.raw.get("combine"):
                    self._combine(c, L, t, g, box, reveal, outer_capture)
            self._frame_t = frame_t

    def _note(self, c, L: Layer, g, box, drawn, alpha, parent_alpha, facing, rect: skia.Rect | None = None):
        """For boxes (track): where L landed on the frame, its anchor, opacity and scale; rect, when given, is where
        it landed. Copies of it (_copies) join their boxes into one."""
        m = c.getTotalMatrix()
        if rect is not None:
            r = rect
        else:
            r = skia.Rect(*drawn) if drawn else m.mapRect(skia.Rect.MakeXYWH(*box))
            if self._ink is not None:  # a stroke reaches half its width past the shape, an arrow's head further
                r.join(self._ink)
        if self._copying and L.id in self.boxes:  # another copy of it: one box holds them all
            r.join(skia.Rect(*self.boxes[L.id]["rect"]))
        sc = 1.0 if drawn else math.sqrt(abs(m.getScaleX() * m.getScaleY())) or abs(m.getSkewX())
        o = m.mapXY(0, 0)  # its anchor, where x and y put it
        info = {"rect": [round(r.left(), 1), round(r.top(), 1), round(r.right(), 1), round(r.bottom(), 1)],
                "pos": [round(o.x(), 1), round(o.y(), 1)], "alpha": round(alpha * parent_alpha, 3),
                "type": L.type, "scale": round(sc, 4)}
        if L.type in ("text", "counter", "captions"):
            # The colour _draw_text uses, so checks see default-coloured text too.
            fill = (g("fill") if "fill" in L.props else None if isinstance(L.raw.get("fill"), dict)
                    else C.parse(self.s.theme.get("paper", "#f2f0e8")))
            info.update(size=float(g("size", 64)), fill=list(fill) if isinstance(fill, tuple) else None,
                        text=self._last_text if hasattr(self, "_last_text") else "")
        if not facing:
            info["mirrored"] = True
        if self._matte is not None:
            info["mask"] = True
        self.boxes[L.id] = info

    def _copies(self, c, L: Layer, t, T, g, box, reveal, mods, alpha, parent_alpha, facing):
        """L's "repeat": L drawn once for each copy, moved by the copy's x, y, rotate and scale in L's space and in
        the copy's own fill, stroke and opacity. A shape's path and paints are made once and drawn for every copy in
        one loop, recoloured. Anything else is recorded once and replayed for each copy while the copies keep L's
        colours, and drawn once for each copy when they do not."""
        cols = L.copy_columns(t, T)
        n = L.copies.n
        zero = [0.0] * n
        xs, ys, rs, ss = cols.get("x", zero), cols.get("y", zero), cols.get("rotate"), cols.get("scale")
        fades = cols.get("opacity")
        own = {k: cols[k] for k in COPY_PAINT if k in cols}
        if L.type in SHAPED and self._capture is None and not (L.type == "rect" and L.raw.get("dimOutside")):
            self._copy_shapes(c, L, t, T, g, box, reveal, mods, xs, ys, rs, ss, fades, own)
            seen = [i for i in range(n) if fades is None or fades[i] > 0.002]
            if self.track and box is not None and seen:
                base, inv = c.getTotalMatrix(), skia.Matrix()
                if base.invert(inv):
                    local = skia.Rect.MakeXYWH(*box)
                    if self._ink is not None:  # its stroke, in L's space
                        local.join(inv.mapRect(self._ink))
                    r = self._spread(base, local, seen, xs, ys, rs, ss)
                    if r is not None:
                        self._note(c, L, g, box, None, alpha, parent_alpha, facing, r)
            return
        a0, cf = self._alpha, self._cf
        if not own and self._m3d is None and self._capture is None and not self.track and self.stop_at is None \
                and self.stop_after is None:
            # Recorded once, where L lands, as far round as any copy reaches into the frame, and replayed moved by
            # each copy's place: in the frame's space, L's matrix times the copy's times L's inverse.
            base, inv = c.getTotalMatrix(), skia.Matrix()
            clip = c.getDeviceClipBounds()
            live = [i for i in range(n) if (fades is None or fades[i] > 0.002) and (ss is None or ss[i])]
            if not live or clip.isEmpty() or not base.invert(inv):
                return
            reach = self._spread(base, inv.mapRect(skia.Rect.Make(clip)), live, xs, ys, rs, ss, inverse=True)
            if reach is None:
                return
            reach.join(skia.Rect.Make(clip))
            rec = skia.PictureRecorder()
            rc = rec.beginRecording(reach, skia.RTreeFactory()())
            rc.setMatrix(c.getLocalToDevice())
            work = self.filtered
            self._body(rc, L, t, T, g, box, reveal, mods, alpha, parent_alpha, facing)
            self.filtered += (self.filtered - work) * (len(live) - 1)
            pic = rec.finishRecordingAsPicture()
            if pic.cullRect().isEmpty():
                return
            for i in live:
                a = 1.0 if fades is None else min(1.0, fades[i])
                m = skia.Matrix.Translate(xs[i], ys[i])
                if rs is not None and rs[i]:
                    m.preRotate(rs[i])
                if ss is not None and ss[i] != 1:
                    m.preScale(ss[i], ss[i])
                c.save()
                c.setMatrix(skia.Matrix.Concat(base, skia.Matrix.Concat(m, inv)))
                c.drawPicture(pic, None, skia.Paint(Alphaf=a) if a < 0.999 else None)
                c.restore()
            return
        one = self._one_draw(L, True)  # a copy's opacity goes in the paint of its one draw
        was = self._copy
        self._copying += 1
        try:
            for i in range(n):
                if self._stopped:
                    break
                a = 1.0 if fades is None else min(1.0, max(0.0, fades[i]))
                if a <= 0.002:
                    continue
                c.save()
                c.translate(xs[i], ys[i])
                if rs is not None and rs[i]:
                    c.rotate(rs[i])
                if ss is not None and ss[i] != 1:
                    c.scale(ss[i], ss[i])
                self._copy = (L, {k: v[i] for k, v in own.items()}) if own else None
                self._ink, self._alpha, self._cf, self._t = None, a0 * a if one else a0, cf, t
                if a < 0.999 and not one:  # it fades in a layer as large as it draws
                    pic = self._record(c, lambda rc: self._body(rc, L, t, T, g, box, reveal, mods, alpha,
                                                                parent_alpha, facing))
                    if pic is not None:
                        c.save()
                        c.resetMatrix()  # the picture was recorded where it lands on this canvas
                        c.drawPicture(pic, None, skia.Paint(Alphaf=a))
                        c.restore()
                else:
                    self._body(c, L, t, T, g, box, reveal, mods, alpha, parent_alpha, facing)
                c.restore()
        finally:
            self._copy = was
            self._copying -= 1

    def _copy_shapes(self, c, L: Layer, t, T, g, box, reveal, mods, xs, ys, rs, ss, fades, own: dict):
        """A shape's copies: the paths and paints _shape draws, made once with the first copy's colours, then drawn
        for each copy in one loop, each paint recoloured with the copy's own fill or stroke. A lone fill takes a
        copy's opacity in its paint; a fill and a stroke fade together in a layer of the copy's own."""
        was, self._copy = self._copy, (L, {k: v[0] for k, v in own.items()}) if own else None
        self._ops = []
        try:
            getattr(self, "_draw_" + L.type)(c, L, t, T, g, box, reveal, mods)
        finally:
            ops, self._ops, self._copy = self._ops, None, was
        base, inv = c.getTotalMatrix(), skia.Matrix()
        if not ops or not base.invert(inv):
            return
        one = len(ops) == 1 and ops[0][2] == "fill"
        kept = []  # (path, paint, each copy's colour as the paint takes it or None, the paint's alpha)
        for path, paint, role, mul, m in ops:
            if m != base:  # drawn moved within L's space (an anchored path)
                path = skia.Path(path)
                path.transform(skia.Matrix.Concat(inv, m))
            p = skia.Paint(paint)  # a kept solid paint is shared; a copy's own colour made it solid
            col = own.get(role)
            kept.append((path, p, None if col is None else self._tints(L, role, col, mul, fades if one else None),
                         p.getAlphaf()))
        save, restore, translate, rotate, scale, draw = c.save, c.restore, c.translate, c.rotate, c.scale, c.drawPath
        if fades is None and rs is None and ss is None and len(kept) == 1:  # copies that only move, in a loop of its own
            path, p, ints, _ = kept[0]
            tint = p.setColor
            for x, y, v in zip(xs, ys, ints or repeat(None)):
                save()
                translate(x, y)
                if v is not None:
                    tint(v)
                draw(path, p)
                restore()
            return
        bounds = None
        dev = math.sqrt(abs(base.getScaleX() * base.getScaleY() - base.getSkewX() * base.getSkewY()))
        for i in range(len(xs)):
            a = 1.0 if fades is None else min(1.0, max(0.0, fades[i]))
            if a <= 0.002:
                continue
            save()
            translate(xs[i], ys[i])
            if rs is not None and rs[i]:
                rotate(rs[i])
            k = 1.0
            if ss is not None and ss[i] != 1:
                k = ss[i]
                scale(k, k)
            alone = a < 0.999 and not one
            if alone:
                if bounds is None:
                    bounds = skia.Rect.MakeEmpty()
                    for path, p, *_ in kept:
                        bounds.join(p.computeFastBounds(path.getBounds()))
                pad = 1.5 / max(dev * abs(k), 1e-3)  # a pixel and a half, past the edge's antialiasing
                c.saveLayer(bounds.makeOutset(pad, pad), skia.Paint(Alphaf=a))
            for path, p, ints, alpha in kept:
                if ints is not None:
                    p.setColor(ints[i])
                elif one and fades is not None:
                    p.setAlphaf(alpha * a)
                draw(path, p)
            if alone:
                restore()
            restore()

    def _tints(self, L: Layer, role: str, col: list, mul: float, fades: list | None) -> list[int]:
        """Each copy's own fill or stroke as a paint takes it, at mul of its alpha (and its opacity, from fades). A
        still column is the same list from frame to frame (Copies), so its colours are kept while mul holds."""
        argb = C.to_argb_int
        if fades is not None:
            return [argb(v, mul * min(1.0, max(0.0, a))) for v, a in zip(col, fades)]
        hit = self._tinted.get((L.id, role))
        if hit is None or hit[0] is not col or hit[1] != mul:
            hit = self._tinted[(L.id, role)] = (col, mul, [argb(v, mul) for v in col])
        return hit[2]

    def _copies_box(self, L: Layer, t, T, box) -> tuple:
        """The box round all of L's copies (_copies), in L's space."""
        cols, n = L.copy_columns(t, T), L.copies.n
        zero = [0.0] * n
        r = self._spread(skia.Matrix(), skia.Rect.MakeXYWH(*box), range(n), cols.get("x", zero), cols.get("y", zero),
                         cols.get("rotate"), cols.get("scale"))
        return box if r is None else (r.left(), r.top(), r.width(), r.height())

    @staticmethod
    def _spread(base: skia.Matrix, rect: skia.Rect, which: list, xs, ys, rs, ss,
                inverse: bool = False) -> skia.Rect | None:
        """Where rect, in a layer's space, lands on the device (by base) moved as each copy in `which` moves it; with
        inverse, moved back by each one's move: what of the layer each one brings to rect."""
        q = np.array([[rect.left(), rect.top()], [rect.right(), rect.top()], [rect.right(), rect.bottom()],
                      [rect.left(), rect.bottom()]])
        w = np.asarray(which)
        x, y = np.asarray(xs, float)[w, None], np.asarray(ys, float)[w, None]
        ang = np.radians(np.asarray(rs, float)[w, None]) if rs is not None else np.zeros((len(w), 1))
        s = np.asarray(ss, float)[w, None] if ss is not None else np.ones((len(w), 1))
        cs, sn = np.cos(ang), np.sin(ang)
        if inverse:
            dx, dy = q[:, 0] - x, q[:, 1] - y
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = (dx * cs + dy * sn) / s, (dy * cs - dx * sn) / s
        else:
            px, py = q[:, 0] * s, q[:, 1] * s
            u, v = px * cs - py * sn + x, px * sn + py * cs + y
        m = base.get9()
        W = m[6] * u + m[7] * v + m[8]
        ok = np.isfinite(u) & np.isfinite(v) & (W > 1e-9)
        if not ok.any():
            return None
        X, Y = ((m[0] * u + m[1] * v + m[2]) / W)[ok], ((m[3] * u + m[4] * v + m[5]) / W)[ok]
        return skia.Rect(float(X.min()), float(Y.min()), float(X.max()), float(Y.max()))

    def _looks(self, L: Layer, g, t: float) -> tuple[skia.ImageFilter | None, float, float]:
        """L's innerShadow, outline and glow, in that order, as one image filter on its offscreen layer, so both
        backends draw them: the layer shaded inside its edge, outlined, and over a blurred copy of itself (in the
        glow's colour, if it has one). Their sizes are in the layer's own px. Also their work, in blurs of the layer
        (CPU_BLUR_S), and how far past the layer they reach, in its px."""
        F, CF, looks = skia.ImageFilters, skia.ColorFilters, L.looks
        f = None  # the layer as drawn
        work = reach = 0.0
        s = looks.get("innerShadow")
        sig = float(g("innerShadow blur", s.get("blur", 12))) / 2 if s else 0.0
        if s:
            r, gg, b, a = C.parse(s.get("color", "#00000080"))
            # The shadow's colour where the layer is not, moved and blurred, kept where the layer is.
            gap = F.ColorFilter(CF.Matrix([0, 0, 0, 0, r, 0, 0, 0, 0, gg, 0, 0, 0, 0, b, 0, 0, 0, -a, a]), f)
            gap = F.Offset(float(g("innerShadow x", s.get("x", 0))), float(g("innerShadow y", s.get("y", 4))), gap)
            if sig > 0.05:
                gap = F.Blur(sig, sig, skia.TileMode.kDecal, gap)
                work += 1
            f = F.Merge([f, F.Xfermode(skia.BlendMode.kSrcIn, f, gap)])
        o = looks.get("outline")
        w = float(g("outline width", o.get("width", 4))) if o else 0.0
        if w > 0.01:
            # The layer copied round circles of radius w, w - 1.5, ... down to a pixel or so, the copies at most 1.5 px
            # apart: a round outline with no gaps at thin strokes. Skia's dilate filter is square, and took 135 ms a
            # pass on the CPU for a 220 px word; the 21 copies of a 3 px outline take 13 ms.
            copies = []
            for r in [r for r in (w - 1.5 * k for k in range(int(w / 1.5) + 1)) if r > 0.4] or [w]:
                n = max(8, math.ceil(2 * math.pi * r / 1.5))
                copies += [F.Offset(r * math.cos(2 * math.pi * j / n), r * math.sin(2 * math.pi * j / n), f)
                           for j in range(n)]
            ring = F.ColorFilter(CF.Blend(_argb(o.get("color", "#000000")), skia.BlendMode.kSrcIn), F.Merge(copies))
            f = F.Merge([ring, f])
            work, reach = work + len(copies) / 20, max(reach, w)  # 20 copies take about as long as a blur
        gl = looks.get("glow")
        size = float(g("glow size", gl.get("size", 24))) if gl else 0.0
        strength = float(g("glow amount", gl.get("amount", 1))) if gl else 0.0
        if size > 0.05 and strength > 0.002:
            halo = F.Blur(size / 2, size / 2, skia.TileMode.kDecal, f)
            if gl.get("color") is not None:
                halo = F.ColorFilter(CF.Blend(_argb(gl["color"]), skia.BlendMode.kSrcIn), halo)
            if strength != 1:
                halo = F.ColorFilter(CF.Matrix([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, strength, 0]), halo)
            f = F.Merge([halo, f])
            work, reach = work + 1, max(reach, 1.5 * size)
        return f, work, reach

    def _looked(self, c, L: Layer, T: float, parent_alpha: float):
        """L with its glow, outline or inner shadow, in an offscreen layer only as large as what L draws. A layer the
        size of the frame, as blur and shadow have, made the CPU dilate and blur all of it: 3.4 s a 1080p frame for an
        outlined line of text. L is recorded first, without its layer; the layer then takes the same paint, in L's own
        space, so the effects keep their sizes and order."""
        was = self._looking, self._look
        self._looking, self._look = L, None
        try:
            body = self._record(c, lambda rc: self._layer(rc, L, T, parent_alpha))
            look = self._look
        finally:
            self._looking, self._look = was
        if body is None or look is None:
            return
        paint, m, work, reach = look
        cull = body.cullRect()
        out = reach * math.sqrt(abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY()))  # on the frame
        self.filtered += work * (cull.width() + 2 * out) * (cull.height() + 2 * out)
        inv = skia.Matrix()
        # What L drew, in its own space: the layer's filters spread it from there.
        bounds = inv.mapRect(body.cullRect()) if not m.hasPerspective() and m.invert(inv) else None
        c.save()
        c.setMatrix(m)
        c.saveLayer(bounds, paint)
        c.resetMatrix()  # the picture was recorded where it lands on this canvas
        c.drawPicture(body)
        c.restore()
        c.restore()

    def _warped(self, c, L: Layer, T: float, t: float, g, parent_alpha: float):
        """L displaced: drawn into an image of its own, at the frame's resolution, then drawn from it with each pixel
        taken from up to `amount` px away, in a direction fractal noise picks. The waves are `scale` px across in the
        layer's own px and slide `speed` of them a second; the image is sampled between its pixels, so bent edges stay
        smooth (Skia's displacement filter takes the nearest pixel, which left them in steps)."""
        d = L.warp
        amount = max(0.0, float(g("displace amount", d.get("amount", 8))))
        m = c.getTotalMatrix()
        a, b, cc, dd = m.getScaleX(), m.getSkewX(), m.getSkewY(), m.getScaleY()
        reach = math.ceil(amount * max(math.hypot(a, cc), math.hypot(b, dd)))  # how far a pixel moves, on the frame
        was, self._warping = self._warping, L
        try:  # what lies just outside the frame can be moved into it
            body = self._record(c, lambda rc: self._layer(rc, L, T, parent_alpha), reach)
        finally:
            self._warping = was
        if body is None:
            return
        blend = None if L is self._cutting else L.raw.get("blend")  # a masked layer blends once cut (_masked)
        mode = BLENDS[blend] if blend not in (None, "normal") else skia.BlendMode.kSrcOver
        c.save()
        c.resetMatrix()  # the picture was recorded where it lands on this canvas
        if reach < 1:
            c.drawPicture(body, None, skia.Paint(BlendMode=mode))
            c.restore()
            return
        src = body.cullRect().roundOut()
        dst = skia.IRect.MakeLTRB(src.left() - reach, src.top() - reach, src.right() + reach, src.bottom() + reach)
        if not dst.intersect(c.getDeviceClipBounds()):
            c.restore()
            return
        self.filtered += dst.width() * dst.height()  # drawing it again through the shader is about one blur's work
        surf = self.surface.makeSurface(self.info.makeWH(src.width(), src.height()))
        sc = surf.getCanvas()
        sc.clear(skia.ColorTRANSPARENT)
        sc.translate(-src.left(), -src.top())
        sc.drawPicture(body)
        img = surf.makeImageSnapshot()
        scale = float(d.get("scale", 60))
        shift = float(d.get("speed", 1)) * t * scale  # the waves slide diagonally
        noise = skia.PerlinNoiseShader.MakeFractalNoise(1 / scale, 1 / scale, 1, int(d.get("seed", 0)))
        if self._warp_effect is None:
            self._warp_effect = skia.RuntimeEffect.MakeForShader(WARP_SKSL)
        fx = skia.RuntimeEffectBuilder(self._warp_effect)
        fx.setChild("layer", img.makeShader(skia.TileMode.kDecal, skia.TileMode.kDecal, SAMPLING,
                                            skia.Matrix.Translate(src.left(), src.top())))
        fx.setChild("noise", noise.makeWithLocalMatrix(skia.Matrix.Concat(m, skia.Matrix.Translate(-shift, -0.6 * shift))))
        fx.setUniform("m", [a, cc, b, dd])
        fx.setUniform("amount", amount)
        # Paint(Shader=...): skia-python's setShader copies a shader that holds an image, 1.6 ms for a line of text.
        c.drawRect(skia.Rect.Make(dst), skia.Paint(Shader=fx.makeShader(), BlendMode=mode))
        c.restore()

    def _masked(self, c, L: Layer, T: float, parent_alpha: float):
        """L cut by its mask: the mask layer, drawn beside it on its own clock, keeps L where the mask is opaque
        (luma: bright), or takes it away there for an invert; a feather blurs the mask's edge by that many px. Both are
        recorded first, so the layers that cut are only as large as what they draw."""
        M, mode, feather = L.mask
        was, self._cutting = self._cutting, L
        try:
            body = self._record(c, lambda rc: self._layer(rc, L, T, parent_alpha))
        finally:
            self._cutting = was
        if body is None and not self.track:  # check sees where the mask is even when the layer draws nothing
            return
        m = c.getTotalMatrix()
        sigma = feather / 2 * math.sqrt(abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY()))
        was, self._matte = self._matte, M
        try:
            cut = self._record(c, lambda rc: self._layer(rc, M, T, parent_alpha))
        finally:
            self._matte = was
        if body is None:
            return
        area = body.cullRect()
        invert = mode.endswith("Invert")
        blend = L.raw.get("blend")
        mix = skia.Paint(BlendMode=BLENDS[blend]) if blend not in (None, "normal") else None
        if cut is None:  # no mask drawn now: an invert keeps all of the layer, any other mode none of it
            if invert:
                c.save()
                c.resetMatrix()
                c.drawPicture(body, None, mix)
                c.restore()
            return
        reach = cut.cullRect().makeOutset(3 * sigma, 3 * sigma)  # the mask, and its blur
        if not invert and not (area.left() < reach.right() and reach.left() < area.right()
                               and area.top() < reach.bottom() and reach.top() < area.bottom()):
            return  # nothing is left outside the mask
        # Both layers hold all of the layer and all of the mask, in whole pixels and one more each way: what a
        # layer's bounds cut through is antialiased otherwise (by up to 56 levels for a circle on the CPU). They hold
        # no more: 50 masks took 2.1 s a 1080p frame on the CPU in layers from the frame's corner, 0.15 s in these.
        # The CPU antialiases a circle or rounded rect by where it is in its layer, so one in a layer that moves with
        # its mask can shift by up to 14 levels at its edge.
        area.join(reach)
        area = skia.Rect.Make(area.makeOutset(1, 1).roundOut())
        paint = skia.Paint(BlendMode=skia.BlendMode.kDstOut if invert else skia.BlendMode.kDstIn)
        if mode.startswith("luma"):
            paint.setColorFilter(skia.LumaColorFilter.Make())  # premultiplied: half-clear white keeps half
        if sigma > 0:
            paint.setImageFilter(skia.ImageFilters.Blur(sigma, sigma))
        self.filtered += (0.4 + (sigma > 0)) * area.width() * area.height()  # two layers to clear and cut, a blur
        c.save()
        c.resetMatrix()  # the pictures were recorded where they land on this canvas
        c.saveLayer(area, mix)
        c.drawPicture(body)
        c.saveLayer(area, paint)
        c.drawPicture(cut)
        c.restore()
        c.restore()
        c.restore()

    @staticmethod
    def _record(c, draw, pad: int = 0) -> skia.Picture | None:
        """What draw(canvas) draws on c, recorded where it lands on c's device with its bounds, and pad px past c's
        clip; None if nothing."""
        clip = c.getDeviceClipBounds()
        if clip.isEmpty():
            return None
        rec = skia.PictureRecorder()
        rc = rec.beginRecording(skia.Rect.Make(clip).makeOutset(pad, pad), skia.RTreeFactory()())
        rc.setMatrix(c.getLocalToDevice())
        draw(rc)
        pic = rec.finishRecordingAsPicture()
        return None if pic.cullRect().isEmpty() else pic

    def _combine(self, c, L: Layer, t: float, g, box, reveal, outer):
        """Merge the outlines L's children gave (_take), first to last by L's combine, and draw the merged shape with
        L's fill and stroke; inside another combining group, give it to that one. Each merge so far is kept: a frame
        merges again from the first child whose outline changed, so children that hold still are merged once."""
        taken, self._capture = self._capture, outer
        if taken is None or self._stopped:
            return
        paths, op = taken[1], COMBINE[L.raw["combine"]]
        old, folds = self._combos.get(L.id, ([], []))
        k = 0
        while k < min(len(old), len(paths)) and old[k] == paths[k]:
            k += 1
        folds = folds[:k]
        for p in paths[k:]:
            folds.append((skia.Op(folds[-1], p, op) or folds[-1]) if folds else skia.Path(p))
        self._combos[L.id] = (paths, folds)
        out = folds[-1] if folds else skia.Path()
        if out.isEmpty():
            return
        if box is None:
            b = out.computeTightBounds()
            box = (b.left(), b.top(), b.width(), b.height())
        self._ink, self._alpha, self._cf, self._t = None, 1.0, None, t
        self._shape(c, L, g, out, box, reveal)

    def _take(self, c, path: skia.Path):
        """For a combining group: path, drawn in the canvas's current space, put in the group's space."""
        inv, paths = self._capture
        out = skia.Path()
        path.transform(skia.Matrix.Concat(inv, c.getTotalMatrix()), out)
        paths.append(out)

    @staticmethod
    def _letters(pcs: list[Piece] | None, s: str, x: float, y: float, font: skia.Font) -> skia.Path:
        """The outlines of s's letters where _draw_pieces draws them; a shaped piece letter by letter, unjoined."""
        out = skia.Path()
        for text, f, width in [(s, font, 0.0)] if pcs is None else [(p.text, p.font, p.width) for p in pcs]:
            glyphs = f.textToGlyphs(text)
            for gid, gx in zip(glyphs, f.getXPos(glyphs)):
                gp = f.getPath(gid)
                if gp is not None:
                    out.addPath(gp, x + gx, y)
            x += width
        return out

    @staticmethod
    def _inner(L: Layer, t: float, T: float) -> float:
        """The scene time group L's children see at its time t: timeRemap's seconds, or t at its speed, from its start,
        and round again every loop seconds."""
        speed, loop = L.clock
        u = float(L.get("timeRemap", t, T)) if "timeRemap" in L.props else t * speed
        if loop > 0:
            u %= loop
        return L.t_in + u

    def _blends_inside(self, L: Layer) -> bool:
        hit = self._blended.get(L.id)
        if hit is None:
            hit = self._blended[L.id] = any(k.raw.get("blend") not in (None, "normal") or self._blends_inside(k)
                                            for k in L.children)
        return hit

    @staticmethod
    def _one_draw(L: Layer, copy: bool = False) -> bool:
        """Whether the layer draws in a single call, so its opacity can go in that call's paint. Two draws that
        overlap, faded one by one, darken where they meet: a stroke over a fill, an arrow's head on its shaft, a
        rect's dimOutside, a stroke around a picture, glyphs that touch, copies (repeat). So does a stroke alone once
        it is thin enough to draw as a hairline, at its joints. Strokes, groups, text, shaders (whose output an
        offscreen layer clamps before it fades) and models keep the offscreen layer. With `copy`: whether one of its
        copies draws in a single call, its own stroke counting as the layer's."""
        ty, raw = L.type, L.raw
        if L.children or (L.copies is not None and not copy):
            return False
        own = copy and "stroke" in L.copies.keys  # a copy's own stroke
        if ty in ("rect", "circle", "ellipse", "path", "polygon"):  # the fill (_shape)
            return raw.get("stroke") is None and not own and not (ty == "rect" and raw.get("dimOutside"))
        if ty in ("image", "video"):  # the picture (_content)
            return raw.get("stroke") is None and not own
        return False

    @staticmethod
    def _fade_layers(c: skia.Canvas, alpha: float):
        """Push the two layers that draw a group at `alpha` as backdrop * (1 - alpha) + composite * alpha, where
        composite is the group drawn over the backdrop at full opacity. The outer layer holds the backdrop scaled
        by 1 - alpha and replaces the backdrop when it is restored; the inner one starts as the backdrop again, the
        children draw into it, and it adds itself at alpha. Both are half-float, so scaling the backdrop down and
        back up loses less than half a level. When the children blend normally this is exactly what an isolated
        group at alpha gives, over any backdrop; restore both layers when the children are drawn."""
        def scale_alpha(k):
            return skia.ImageFilters.ColorFilter(skia.ColorFilters.Matrix(
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, k, 0]))
        k = 1.0 - alpha
        # The records hold bare pointers to the paints and filters: keep them in names until saveLayer has read them.
        keep, down = skia.Paint(BlendMode=skia.BlendMode.kSrc), scale_alpha(k)
        c.saveLayer(skia.Canvas.SaveLayerRec(None, keep, down, skia.Canvas.kF16ColorType))
        add, up = skia.Paint(BlendMode=skia.BlendMode.kPlus, Alphaf=alpha), scale_alpha(1.0 / k)
        c.saveLayer(skia.Canvas.SaveLayerRec(None, add, up, skia.Canvas.kF16ColorType))

    def _box(self, L: Layer, t, T, g):
        """Local content rect (x, y, w, h) after anchoring, or None for unbounded layers."""
        ty = L.type
        if ty == "group":
            if L.raw.get("w") is None and L.raw.get("h") is None:
                return None
            w, h = float(g("w", self.s.w)), float(g("h", self.s.h))
            ax, ay = ANCHORS.get(L.raw.get("anchor", "topleft"), (0, 0))
            return (-ax * w, -ay * h, w, h)
        if ty in ("text", "counter"):
            lay = self._text_layout(L, t, T, g, self._text_of(L, t, T, g))
            pad = self._pad(L)
            w, h = lay.w + pad[1] * 2, lay.h + pad[0] * 2
        elif ty in ("captions", "model", "particles"):
            return None  # a model's bounds come from its projected mesh (render3d.draw), particles' from where they are
        elif ty in ("circle",):
            w = h = float(g("r", 50)) * 2
        elif ty in ("path", "line", "arrow", "polygon"):
            p = self._path(L, t, T, g)
            b = p.computeTightBounds()
            if "anchor" not in L.raw:
                return (b.left(), b.top(), b.width(), b.height())
            w, h = b.width(), b.height()
            ax, ay = ANCHORS.get(L.raw["anchor"], (0.5, 0.5))
            return (-ax * w, -ay * h, w, h)
        elif ty == "image":
            img = media.image(L.src)
            w, h = float(g("w", img.width())), float(g("h", img.height()))
            if L.raw.get("w") is not None and L.raw.get("h") is None:
                h = w * img.height() / img.width()
            elif L.raw.get("h") is not None and L.raw.get("w") is None:
                w = h * img.width() / img.height()
        elif ty in ("video", "shader"):
            w, h = float(g("w", self.s.w)), float(g("h", self.s.h))
        else:
            w, h = float(g("w", 200)), float(g("h", 120))
        default_anchor = "topleft" if ty in ("video", "shader") or (ty == "image" and L.raw.get("x") is None) else "center"
        ax, ay = ANCHORS.get(L.raw.get("anchor", default_anchor), (0.5, 0.5))
        return (-ax * w, -ay * h, w, h)

    # ---- paint helpers --------------------------------------------------------------------
    def _gradient(self, spec: dict, x, y, w, h, alpha: float):
        stops = spec.get("stops", ["#000", "#fff"])
        colors = [_argb(s if isinstance(s, str) else s[1], alpha) for s in stops]
        pos = None
        if stops and not isinstance(stops[0], str):
            pos = [float(s[0]) for s in stops]
        if "radial" in spec:
            cx, cy, r = (spec["radial"] + [0.5, 0.5, 0.6])[:3] if isinstance(spec["radial"], list) else (0.5, 0.5, 0.6)
            return skia.GradientShader.MakeRadial(skia.Point(x + cx * w, y + cy * h), max(1.0, r * max(w, h)), colors, pos)
        if "sweep" in spec:  # round its middle from the top, clockwise, turned by angle (CSS conic-gradient's start)
            cx, cy = (spec["sweep"] + [0.5, 0.5])[:2] if isinstance(spec["sweep"], list) else (0.5, 0.5)
            px, py = x + cx * w, y + cy * h
            return skia.GradientShader.MakeSweep(px, py, colors, pos, localMatrix=skia.Matrix.RotateDeg(
                float(spec.get("angle") or 0) - 90, skia.Point(px, py)))
        if "linear" not in spec and spec.get("angle") is not None:  # CSS's: 0 runs up, 90 to the right, corner to corner
            a = math.radians(float(spec["angle"]))
            dx, dy = math.sin(a), -math.cos(a)
            half = (abs(w * dx) + abs(h * dy)) / 2
            cx, cy = x + w / 2, y + h / 2
            return skia.GradientShader.MakeLinear([skia.Point(cx - dx * half, cy - dy * half),
                                                   skia.Point(cx + dx * half, cy + dy * half)], colors, pos)
        a, b = spec.get("linear", [[0, 0], [1, 1]])
        return skia.GradientShader.MakeLinear([skia.Point(x + a[0] * w, y + a[1] * h), skia.Point(x + b[0] * w, y + b[1] * h)],
                                              colors, pos)

    def _own(self, L: Layer, key: str):
        """A copy's own fill or stroke, as (r, g, b, a), while L's copies draw (_copies); else None."""
        cp = self._copy
        return None if cp is None or cp[0] is not L else cp[1].get(key)

    def _fill_paint(self, L: Layer, g, box, alpha=1.0, key="fill", default=None):
        own = self._own(L, key)
        raw = L.raw.get(key, default) if own is None else own
        if raw is None:
            return None
        if isinstance(raw, dict) and key not in L.props:  # a gradient; a colour tween is compiled into props
            p = skia.Paint(AntiAlias=True)
            bx, by, bw, bh = box
            p.setShader(self._gradient(raw, bx, by, bw, bh, alpha))
        else:
            col = (own if own is not None else g(key) if key in L.props else C.parse(raw) if isinstance(raw, str)
                   else raw)
            if isinstance(col, C.Gradient):  # one that animates
                p = skia.Paint(AntiAlias=True)
                p.setShader(self._gradient(col.spec(), *box, alpha))
                return p
            argb = _argb(col, alpha)
            p = self._solid.get(argb)
            if p is None:
                if len(self._solid) >= 4096:  # colours that animate make a new paint on most frames
                    self._solid.clear()
                p = self._solid[argb] = skia.Paint(AntiAlias=True, Color=argb)
        return p

    def _stroke_paint(self, L: Layer, g, box, alpha=1.0):
        own = self._own(L, "stroke")
        raw = L.raw.get("stroke") if own is None else own
        if raw is None:
            return None
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=float(g("strokeWidth", 4)))
        if isinstance(raw, dict) and "stroke" not in L.props:  # a gradient
            p.setShader(self._gradient(raw, *box, alpha))
        else:
            col = own if own is not None else g("stroke") if "stroke" in L.props else C.parse(raw)
            if isinstance(col, C.Gradient):  # one that animates
                p.setShader(self._gradient(col.spec(), *box, alpha))
            else:
                p.setColor(_argb(col, alpha))
        cap = L.raw.get("cap", "round" if L.type in ("line", "arrow", "path") else "butt")
        p.setStrokeCap({"round": skia.Paint.kRound_Cap, "square": skia.Paint.kSquare_Cap}.get(cap, skia.Paint.kButt_Cap))
        p.setStrokeJoin({"round": skia.Paint.kRound_Join, "bevel": skia.Paint.kBevel_Join}.get(L.raw.get("join", "round"), skia.Paint.kMiter_Join))
        return p

    @staticmethod
    def _trim(path: skia.Path, amount: float, start: float = 0.0) -> skia.Path:
        if amount >= 0.999 and start <= 0:
            return path
        out = skia.Path()
        meas = skia.PathMeasure(path, False)
        while True:
            n = meas.getLength()
            meas.getSegment(n * start, n * max(start, min(1.0, amount)), out, True)
            if not meas.nextContour():
                break
        return out

    @staticmethod
    def _dash(L: Layer, g, paint: skia.Paint, path: skia.Path | None = None, start: float = 0.0):
        """Mark out paint's stroke in L's dash, slid back along it by dashOffset as SVG slides one. A stroke that
        starts `start` of the way along `path` keeps its dashes where they lie on the whole path."""
        on = [float(d) for d in L.raw["dash"]]
        if len(on) % 2:  # an odd list repeats, as in SVG
            on += on
        phase = float(g("dashOffset", 0.0))
        if start > 0 and path is not None:
            phase += start * skia.PathMeasure(path, False).getLength()
        paint.setPathEffect(skia.DashPathEffect.Make(on, phase % sum(on)))

    @staticmethod
    def _radius(g):
        """radius as one number, or as the list of four corners it may be."""
        r = g("radius", 0)
        return [float(v) for v in r] if isinstance(r, (list, tuple)) else float(r)

    def _note_ink(self, c, path: skia.Path, paint: skia.Paint):
        """For boxes (track): add the device bounds of `path` stroked with `paint` to the layer's box."""
        if not self.track:
            return
        out = skia.Path()
        paint.getFillPath(path, out)
        r = c.getTotalMatrix().mapRect(out.computeTightBounds())
        if self._ink is None:
            self._ink = r
        else:
            self._ink.join(r)

    def _rough(self, L: Layer, path: skia.Path) -> skia.Path:
        """path wobbled as by hand: points `length` px apart moved up to `amount` px, the same for the same seed. boil
        picks a new seed that many times a second of the layer's time, so the frame stays a function of its time."""
        amount, length, seed, boil = L.rough
        if boil:
            seed += int(math.floor(self._t * boil + 1e-9))
        hit = self._roughs.get(L.id)
        if hit is not None and hit[0] == seed and hit[1] == path:  # the same path object, so the GPU's work is kept
            return hit[2]
        out = skia.Path()
        # As a hairline: a fill paint makes Skia measure an open path as closed, adding a stroke back to its start
        skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=0,
                   PathEffect=skia.DiscretePathEffect.Make(length, amount, seed)).getFillPath(path, out)
        self._roughs[L.id] = (seed, skia.Path(path), out)
        return out

    def _shape(self, c, L, g, path: skia.Path, box, reveal):
        """Fill then stroke. 'draw' (0..1) trims the stroke and fades the fill with it."""
        if L.rough is not None:
            path = self._rough(L, path)
        amt = max(0.0, min(1.0, float(g("draw")) * reveal))
        has_stroke = L.raw.get("stroke") is not None or self._own(L, "stroke") is not None
        fp = self._fill_paint(L, g, box, alpha=amt * self._alpha,
                              default=None if has_stroke else self.s.theme.get("paper", "#f2f0e8"))
        if self._capture is not None:  # a combining group takes its fill's outline, or its stroke's
            if fp is not None:
                if amt > 0:
                    self._take(c, path)
            elif (sp := self._stroke_paint(L, g, box)) is not None and amt > 0.0005:
                start = float(g("drawStart", 0.0))
                if "dash" in L.raw:
                    self._dash(L, g, sp, path, start)
                self._take(c, _outline(sp, self._trim(path, amt, start)))
            return
        if fp is not None and self._cf is not None:
            fp = skia.Paint(fp)  # a kept solid paint is shared
            fp.setColorFilter(self._cf)
        if fp is not None and amt > 0:
            if self._ops is not None:  # its copies draw it (_copy_shapes)
                self._ops.append((path, fp, "fill", amt * self._alpha, c.getTotalMatrix()))
            else:
                c.drawPath(path, fp)
        sp = self._stroke_paint(L, g, box)
        if sp is not None:
            self._note_ink(c, path, sp)  # the whole stroke, as the box is the whole shape while it draws on
            if amt > 0.0005:
                start = float(g("drawStart", 0.0))
                if "dash" in L.raw:
                    self._dash(L, g, sp, path, start)
                if self._ops is not None:
                    self._ops.append((self._trim(path, amt, start), sp, "stroke", 1.0, c.getTotalMatrix()))
                else:
                    c.drawPath(self._trim(path, amt, start), sp)

    # ---- layer types ----------------------------------------------------------------------
    def _draw_group(self, c, L, t, T, g, box, reveal, mods):
        if L.raw.get("combine"):  # its fill and stroke draw what its children merge into (_combine)
            return
        if box is not None and L.raw.get("clip"):
            c.clipRRect(_rrect(box, self._radius(g)), doAntiAlias=True)
        fp = self._fill_paint(L, g, box) if box is not None else None
        if fp is not None:
            c.drawRRect(_rrect(box, self._radius(g)), fp)

    def _draw_rect(self, c, L, t, T, g, box, reveal, mods):
        bx, by, bw, bh = box
        r = self._radius(g)
        r = r if isinstance(r, list) else min(r, bw / 2, bh / 2)
        dim = float(g("dimOutside", 0))
        if dim > 0:
            hole = skia.Path()
            hole.addRect(skia.Rect(-40000, -40000, 40000, 40000))
            hole.addRRect(_rrect(box, r))
            hole.setFillType(skia.PathFillType.kEvenOdd)
            c.drawPath(hole, skia.Paint(AntiAlias=True, Color=_argb("#000000", dim * reveal)))
        path = skia.Path()
        path.addRRect(_rrect(box, r))
        self._shape(c, L, g, path, box, reveal)

    def _draw_circle(self, c, L, t, T, g, box, reveal, mods):
        path = skia.Path()
        path.addOval(skia.Rect.MakeXYWH(*box), skia.PathDirection.kCW, 3)  # start at the top
        self._shape(c, L, g, path, box, reveal)

    def _draw_ellipse(self, c, L, t, T, g, box, reveal, mods):
        self._draw_circle(c, L, t, T, g, box, reveal, mods)

    def _along(self, L: Layer, t: float, T: float) -> tuple[float, float, float]:
        """Where L's along puts its anchor at t, as an offset from its x and y, and the path's direction there in
        degrees. The path's first contour is ridden: an open one stops at its ends, a closed one goes round again."""
        d = L.along[0]
        route = self.s.by_id[d[1:]] if d.startswith("#") else None
        if route is None:
            hit = self._routes.get(L.id)
            if hit is None:
                path = svgpath.parse(d)
                meas = skia.PathMeasure(path, False)
                hit = self._routes[L.id] = (path, meas, meas.getLength())
        else:
            rt = T - route.t_in
            rg = lambda name, default=None: route.get(name, rt, T, 0, default)  # noqa: E731
            path = self._path(route, rt, T, rg)
            hit = self._routes.get(route.id)
            if hit is None or hit[0] is not path:
                meas = skia.PathMeasure(path, False)
                hit = self._routes[route.id] = (path, meas, meas.getLength())
        path, meas, n = hit
        p = float(L.get("along", t, T))
        p = p % 1.0 if meas.isClosed() else min(max(p, 0.0), 1.0)
        pt = meas.getPosTan(p * n) if n > 0 else None
        if not pt:
            return 0.0, 0.0, 0.0
        pos, tan = pt
        px, py, tx, ty = pos.x(), pos.y(), tan.x(), tan.y()
        if route is not None:  # where the route draws it: _draw_path's anchor shift, then the route's own placing
            if "anchor" in route.raw:
                box, b = self._box(route, rt, T, rg), path.computeTightBounds()
                px, py = px + box[0] - b.left(), py + box[1] - b.top()
            x, y, rot, sc, sx, sy, _ = route.placed(rt, T)
            sx, sy = float(sc) * float(sx), float(sc) * float(sy)
            px, py, tx, ty = px * sx, py * sy, tx * sx, ty * sy
            if rot:
                co, si = math.cos(math.radians(float(rot))), math.sin(math.radians(float(rot)))
                px, py, tx, ty = px * co - py * si, px * si + py * co, tx * co - ty * si, tx * si + ty * co
            px, py = px + float(x), py + float(y)
        return px, py, math.degrees(math.atan2(ty, tx))

    def _morph(self, L: Layer, g) -> skia.Path:
        """A morphing path's d where its keys have got to: a key's own path on it, between two keys their matched
        pair mixed (an ease that overshoots carries on past the key)."""
        keys = L.morph
        u = float(g("morph"))
        i = min(max(int(math.floor(u)), 0), len(keys) - 2)
        if len(keys) == 1 or u == i:
            return keys[max(0, i)]
        if u == i + 1:
            return keys[i + 1]
        last = self._morphed.get(L.id)
        if last is not None and last[0] == u:
            return last[1]
        pair = self._morphs.get((L.id, i))
        if pair is None:
            pair = self._morphs[(L.id, i)] = M.match(keys[i], keys[i + 1])
        out = skia.Path()
        pair[0].interpolate(pair[1], 1.0 - (u - i), out)
        self._morphed[L.id] = (u, out)
        return out

    def _path(self, L: Layer, t, T, g) -> skia.Path:
        ty = L.type
        if L.morph is not None:
            p = self._morph(L, g)
            return self._corner(p, g) if "corner" in L.props else p
        static = all(L.props[k].static for k in ("x1", "y1", "x2", "y2", "bend", "r", "sides", "inner") if k in L.props)
        rounds = "corner" in L.props and not L.props["corner"].static  # a corner that changes: kept unrounded
        key = (L.id, "unrounded") if rounds else L.id
        if static and key in self._paths:
            return self._corner(self._paths[key], g) if rounds else self._paths[key]
        p = skia.Path()
        if ty == "path" and L.raw.get("d"):
            p = svgpath.parse(L.raw["d"])
        elif ty in ("path", "line"):
            pts = L.raw.get("points") or [[0, 0], [100, 0]]
            if L.raw.get("smooth") and len(pts) > 2:
                p.moveTo(*pts[0])
                for i in range(len(pts) - 1):
                    p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(len(pts) - 1, i + 2)]
                    p.cubicTo(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
                              p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6, p2[0], p2[1])
            else:
                p.moveTo(*pts[0])
                for q in pts[1:]:
                    p.lineTo(*q)
            if L.raw.get("closed"):
                p.close()
        elif ty == "arrow":
            x1, y1 = float(g("x1", 0)), float(g("y1", 0))
            x2, y2 = float(g("x2", 200)), float(g("y2", 0))
            bend = float(g("bend", 0.0))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = -(y2 - y1), (x2 - x1)
            p.moveTo(x1, y1)
            p.quadTo(mx + nx * bend, my + ny * bend, x2, y2)
        elif ty == "polygon":
            n = max(3, int(g("sides", 6)))
            r = float(g("r", 80))
            inner = g("inner")
            pts = []
            steps = n * 2 if inner else n
            for k in range(steps):
                a = -math.pi / 2 + k * 2 * math.pi / steps
                rr = r * (float(inner) if inner and k % 2 else 1)
                pts.append((math.cos(a) * rr, math.sin(a) * rr))
            p.moveTo(*pts[0])
            for q in pts[1:]:
                p.lineTo(*q)
            p.close()
        if static and rounds:
            self._paths[key] = p
            return self._corner(p, g)
        if "corner" in L.props:
            p = self._corner(p, g)
        if static:
            self._paths[key] = p
        return p

    @staticmethod
    def _corner(p: skia.Path, g) -> skia.Path:
        """p with every corner rounded by corner px, so draw and a rider on it follow the rounded line."""
        corner = float(g("corner", 0))
        if corner <= 0:
            return p
        out = skia.Path()
        skia.Paint(PathEffect=skia.CornerPathEffect.Make(corner)).getFillPath(p, out)
        return out

    def _draw_path(self, c, L, t, T, g, box, reveal, mods):
        path = self._path(L, t, T, g)
        c.save()  # the box is in the layer's space: an anchored path's shift stays in here
        if "anchor" in L.raw:
            b = path.computeTightBounds()
            c.translate(box[0] - b.left(), box[1] - b.top())
        self._shape(c, L, g, path, box, reveal)
        c.restore()

    _draw_polygon = _draw_path

    def _draw_line(self, c, L, t, T, g, box, reveal, mods):
        if L.raw.get("stroke") is None:
            L.raw["stroke"] = self.s.theme.get("paper", "#f2f0e8")
        self._draw_path(c, L, t, T, g, box, reveal, mods)

    def _draw_arrow(self, c, L, t, T, g, box, reveal, mods):
        if L.raw.get("stroke") is None:
            L.raw["stroke"] = self.s.theme.get("paper", "#f2f0e8")
        path = self._path(L, t, T, g)
        amt = float(g("draw")) * reveal
        sp = self._stroke_paint(L, g, box)
        head = float(g("head", 22))

        def head_at(k):
            meas = skia.PathMeasure(path, False)
            pt = meas.getPosTan(meas.getLength() * min(1.0, k))
            if not pt:
                return None
            pos, tan = pt
            ang = math.atan2(tan.y(), tan.x())
            hp = skia.Path()
            hp.moveTo(pos.x(), pos.y())
            for s in (2.6, -2.6):
                hp.moveTo(pos.x(), pos.y())
                hp.lineTo(pos.x() + head * math.cos(ang + s), pos.y() + head * math.sin(ang + s))
            return hp
        if self.track:  # the whole arrow and its head at the tip, as for any shape that draws on
            self._note_ink(c, path, sp)
            full = head_at(1.0)
            if full is not None:
                self._note_ink(c, full, sp)
        if amt <= 0.0005:
            return
        hp = head_at(amt)  # on the smooth line, so a rough arrow still points where it goes
        if L.rough is not None:
            path = self._rough(L, path)
            hp = None if hp is None else self._rough(L, hp)
        shaft = sp
        if "dash" in L.raw:  # the head stays whole
            shaft = skia.Paint(sp)
            self._dash(L, g, shaft)
        if self._capture is not None:  # a combining group takes the outline of its stroke
            out = _outline(shaft, self._trim(path, amt))
            if hp is not None:
                out.addPath(_outline(sp, hp))
            self._take(c, out)
            return
        c.drawPath(self._trim(path, amt), shaft)
        if hp is not None:
            c.drawPath(hp, sp)

    def _draw_image(self, c, L, t, T, g, box, reveal, mods):
        img = media.image(L.src)
        self._content(c, L, g, box, img, img.width(), img.height())

    def _content(self, c, L, g, box, img, iw, ih):
        bx, by, bw, bh = box
        zoom = float(g("zoom"))
        fx, fy = (L.raw.get("focus") or [0.5, 0.5]) if not isinstance(L.raw.get("focus"), str) else (0.5, 0.5)
        if "focusX" in L.props:
            fx = float(g("focusX"))
        if "focusY" in L.props:
            fy = float(g("focusY"))
        fit = L.raw.get("fit", "cover")
        r = self._radius(g)
        c.save()
        if zoom != 1 or r or fit == "cover":
            c.clipRRect(_rrect(box, r), doAntiAlias=True)
        if fit == "stretch" or L.type == "video":
            cw, ch = bw, bh
        else:
            k = (max if fit == "cover" else min)(bw / iw, bh / ih)
            cw, ch = iw * k, ih * k
        cx0 = bx + (bw - cw) / 2
        cy0 = by + (bh - ch) / 2
        px, py = bx + fx * bw, by + fy * bh
        dst = skia.Rect.MakeXYWH(px + (cx0 - px) * zoom, py + (cy0 - py) * zoom, cw * zoom, ch * zoom)
        if img is not None:
            c.drawImageRect(img, dst, CUBIC if L.type == "video" else SAMPLING,
                            skia.Paint(AntiAlias=True, Alphaf=self._alpha, ColorFilter=self._cf))
        c.restore()
        sp = self._stroke_paint(L, g, box)
        if sp:
            rr = _rrect(box, r)
            if self.track:
                edge = skia.Path()
                edge.addRRect(rr)
                self._note_ink(c, edge, sp)
            if "dash" in L.raw:
                self._dash(L, g, sp)
            c.drawRRect(rr, sp)

    def _draw_video(self, c, L, t, T, g, box, reveal, mods):
        bx, by, bw, bh = box
        if L.video is None:
            L.video = self._decoder(L, self._decode_size(L, bw, bh))
        if not self.footage:  # its box and stroke only; made above all the same, so its decode size does not change
            self._content(c, L, g, box, None, bw, bh)
            return
        if self._frame_t is not None:
            # A motion-blur sub-frame shows its frame's picture (footage carries its own blur). Sub-frame times lie
            # between the decoder's output frames, so each would reopen it; they used to round to whichever was near.
            t = min(max(self._frame_t, L.t_in), math.nextafter(L.t_out, -math.inf)) - L.t_in
        img = L.video.at(self._src_time(L, t, L.video.info))
        if img is not None:
            self._content(c, L, g, box, img, img.width(), img.height())

    def _surface(self, gid: str, T: float) -> skia.Image | None:
        """Group gid drawn at scene time T on an image of its own, for a model's texture: its box (w x h, or the
        canvas's size) fills the image. Its layers draw as in a scene without a camera, those that move in depth seen
        as a camera of the image's size sees them, without fog or depth of field; their boxes are not the frame's.
        None for a group that is already being drawn, which would show itself."""
        G = self.s.by_id[gid]
        if gid in self._insets:
            return None
        t = T - G.t_in
        w, h = (max(1.0, float(G.get(k, t, T, 0, d))) for k, d in (("w", self.s.w), ("h", self.s.h)))
        size = (max(1, min(4096, round(w * self.scale))), max(1, min(4096, round(h * self.scale))))
        surf = self._models.get(("surface", gid))
        if surf is None or (surf.width(), surf.height()) != size:
            info = skia.ImageInfo.MakeN32Premul(*size)
            surf = (self.gr and skia.Surface.MakeRenderTarget(self.gr, skia.Budgeted.kNo, info)) or \
                skia.Surface.MakeRaster(info)
            self._models[("surface", gid)] = surf
        c = surf.getCanvas()
        c.resetMatrix()
        c.clear(skia.ColorTRANSPARENT)
        c.scale(size[0] / w, size[1] / h)
        ax, ay = ANCHORS.get(G.raw.get("anchor", "topleft"), (0, 0))
        cam = space.camera(w, h, x=-ax * w, y=-ay * h)  # looking at the middle of the box
        kept = (self.boxes, self._m3d, self._cam, self._focused, self._stage, self._dev, self._lens, self._outer)
        self.boxes, self._focused, self._stage, self._dev, self._lens = {}, None, None, 1.0, space.lens(w, h)
        self._insets.add(gid)
        try:
            for child in G.children:
                if self._stopped:
                    break
                m = cam if uses_depth(child) else None
                self._m3d = self._cam = m
                c.save()
                if m is None:
                    c.translate(ax * w, ay * h)
                else:
                    c.concat(space.to_skia(m))
                self._layer(c, child, T, 1.0)
                c.restore()
        finally:
            self._insets.discard(gid)
            self.boxes, self._m3d, self._cam, self._focused, self._stage, self._dev, self._lens, self._outer = kept
        return surf.makeImageSnapshot()

    def _clip(self, L: Layer, spec: dict, t: float) -> skia.Image | None:
        """The frame video texture spec ({"src", "loop", "trim", "speed"}) on model L shows at L's own time t, the
        whole frame at the clip's own size, up to 2048 px across."""
        if self._frame_t is not None:  # a motion-blur sub-frame shows its frame's picture, as a video layer does
            t = min(max(self._frame_t, L.t_in), math.nextafter(L.t_out, -math.inf)) - L.t_in
        timing = SimpleNamespace(raw={k: spec[k] for k in ("loop", "trim", "speed") if k in spec})
        key = (L.id, spec["src"], repr(timing.raw))
        hit = self._clips.get(key)
        if hit is None:
            src = self.s.path(spec["src"])
            info = media.probe(src)
            k = min(1.0, self.scale, 2048 / max(info["w"], info["h"], 1))
            hit = self._clips[key] = (L, media.VideoSource(
                src, round(info["w"] * k), round(info["h"] * k), self.s.fps, float(spec.get("speed", 1)), "stretch",
                ctx=gpu_context() if self.backend == "gpu" else None))
        v = hit[1]
        return v.at(self._src_time(timing, t, v.info))

    def _decode_size(self, L: Layer, bw: float, bh: float) -> tuple[int, int]:
        """The size video layer L decodes at in a bw x bh box."""
        zmax = 1.0
        zp = L.props.get("zoom")
        if zp is not None:
            zmax = max(float(zp.at(L.dur * k / 8, L.dur)) for k in range(9))
        info = media.probe(L.src)
        # Decode at no more than the source's own resolution: piping upscaled RGBA (8 MB a frame at
        # 1080p) costs more than the decode, and the GPU upscales it at draw time anyway.
        rw, rh = info["w"] / max(bw, 1), info["h"] / max(bh, 1)
        native = (min if L.raw.get("fit", "cover") == "cover" else max)(rw, rh) if info["w"] else 1.0
        k = min(max(1.0, zmax) * self.scale, native)
        return round(bw * k), round(bh * k)

    def _decoder(self, L: Layer, size: tuple[int, int]) -> media.VideoSource:
        """A decoder for video layer L at `size`: the one _prime opened for it at that size, or a new one."""
        ahead, L.ahead = L.ahead, None
        if ahead is not None:
            if ahead[0] == size:
                return ahead[1]
            ahead[1].close()
        return media.VideoSource(L.src, *size, self.s.fps, float(L.raw.get("speed", 1)), L.raw.get("fit", "cover"),
                                 ctx=gpu_context() if self.backend == "gpu" else None)

    def _src_time(self, L: Layer, t: float, info: dict) -> float:
        """The source time video layer L shows at its own time t; info is its clip's probe."""
        speed = float(L.raw.get("speed", 1))
        trim0 = float((L.raw.get("trim") or [0])[0])
        src_t = trim0 + t * speed
        end = loop_end(L, info["duration"])  # the trim end, or the end of the file
        if L.raw.get("loop") and end - trim0 > 0:
            src_t = trim0 + (t * speed) % (end - trim0)
        # Up to the time whose nearest frame (VideoSource) lies before the end: a dur longer than the trimmed clip
        # holds the frame it ends on, the same frame for every frame of the hold. Before, trim [2, 4] with dur 6
        # played on to 6.9 s of the file.
        return min(src_t, max(trim0, end - 0.5 / (info["fps"] or self.s.fps)))

    # ---- shader layer ---------------------------------------------------------------------
    def _draw_shader(self, c, L, t, T, g, box, reveal, mods):
        eff, defaults = self._effect(L)
        bx, by, bw, bh = box
        b = skia.RuntimeShaderBuilder(eff)
        names = {u.name for u in eff.uniforms()}
        vals = dict(defaults, **(L.raw.get("uniforms") or {}))
        for name in names:
            if name == "iTime":
                b.setUniform(name, float(t))
            elif name == "iProgress":
                b.setUniform(name, float(t / L.dur if L.dur else 0))
            elif name == "iResolution":
                b.setUniform(name, [float(bw), float(bh)])
            elif name in vals:
                v = L.get("u_" + name, t, T) if ("u_" + name) in L.props else vals[name]
                if isinstance(v, str):
                    v = C.parse(v)
                b.setUniform(name, [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v))
        mtx = skia.Matrix.Translate(bx, by)
        c.drawRect(skia.Rect.MakeXYWH(*box), skia.Paint(Shader=b.makeShader(mtx)))

    def _draw_model(self, c, L, t, T, g, box, reveal, mods):
        return render3d.draw(self, c, L, g, reveal, t, T)

    def _draw_particles(self, c, L, t, T, g, box, reveal, mods):
        """The particles alive at t, each its sprite turned, sized and tinted, in one drawAtlas call; with track, the
        frame rect they cover."""
        try:
            st = L.particles.state(t, L.dur, L.t_in)
        except ValueError as e:
            raise SceneError(f"layer '{L.id}': {e}", layer=L.id) from e
        if st is None:
            return None
        x, y, size = st["x"], st["y"], st["size"]
        moved = self._emitted(L, t, T, mods, st["birth"])
        if moved is not None:
            x, y = x + moved[0], y + moved[1]
        img, cells, side, mid = self._sprites(L)
        k = size / side
        rot = st["rot"]
        scos, ssin = k * np.cos(rot), k * np.sin(rot)
        tx, ty = x - (scos - ssin) * mid, y - (ssin + scos) * mid
        rs = skia.RSXform
        xf = [rs(a, b, e, f) for a, b, e, f in zip(scos.tolist(), ssin.tolist(), tx.tolist(), ty.tolist())]
        cell = st["cell"]
        tex = [cells[0]] * len(xf) if cell is None else [cells[j] for j in cell.tolist()]
        c.drawAtlas(img, xf, tex, st["argb"].tolist(), skia.BlendMode.kModulate, SAMPLING)
        m = c.getTotalMatrix()
        quad = float(np.dot(k, k)) * (2 * mid) ** 2 * abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY())
        self.particle_s += len(xf) * CPU_PARTICLE_S + quad * CPU_PARTICLE_PX_S
        if not self.track:
            return None
        half = size * (0.5 if L.particles.shape in ("circle", "soft", "ring") else 0.71)
        r = m.mapRect(skia.Rect(float((x - half).min()), float((y - half).min()), float((x + half).max()),
                                float((y + half).max())))
        return (r.left(), r.top(), r.right(), r.bottom())

    def _emitted(self, L: Layer, t: float, T: float, mods, births: np.ndarray):
        """How far L's emitter has moved since each particle left it, in L's own space, so that particles stay where
        they were born while x, y or along move the layer; None while it stays put."""
        hit = self._emitter.get(L.id, False)
        if hit is False:
            hit = None
            if any(k in L.props and not L.props[k].static for k in ("x", "y")) or L.along is not None:
                off = T - t
                grid = np.minimum(np.arange(int(math.ceil(L.dur * PT.GRID)) + 1) / PT.GRID, L.dur)
                spots = np.array([self._spot_of(L, s, s + off) for s in grid.tolist()])
                hit = (grid, spots[:, 0], spots[:, 1])
            self._emitter[L.id] = hit
        if hit is None:
            return None
        grid, xs, ys = hit
        ex, ey = self._spot_of(L, t, T)
        dx, dy = np.interp(births, grid, xs) - ex, np.interp(births, grid, ys) - ey
        _, _, rot, sc, sx, sy, _ = L.placed(t, T)
        sc = float(sc) * mods.scale
        sx, sy = sc * float(sx), sc * float(sy)
        if abs(sx) < 1e-9 or abs(sy) < 1e-9:
            return None
        a = -math.radians(float(rot) + mods.rotate)
        cs, sn = math.cos(a), math.sin(a)
        return (cs * dx - sn * dy) / sx, (sn * dx + cs * dy) / sy

    def _spot_of(self, L: Layer, t: float, T: float) -> tuple[float, float]:
        """Where x, y and along put L's anchor at t, in its parent's space."""
        x, y = float(L.get("x", t, T)), float(L.get("y", t, T))
        if L.along is not None:
            ax, ay, _ = self._along(L, t, T)
            x, y = x + ax, y + ay
        return x, y

    def _sprites(self, L: Layer) -> tuple:
        """L's particle drawn white on clear, once: (the image, a cell rect for each width it flips through, the
        particle's size in the image's px, the middle of a cell). The image holds the particle at its largest size
        on the frame, so it is never drawn larger than it was made."""
        hit = self._sprite.get(L.id)
        if hit is not None:
            return hit
        em = L.particles
        side = int(min(256, max(8, math.ceil(em.most() * self.scale))))
        cell = side + 4
        n = PT.FLIPS if em.flips else 1
        surf = skia.Surface(cell * n, cell)
        c = surf.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        draw = self._sprite_of(L, side)
        for j in range(n):
            c.save()
            c.translate(cell * j + cell / 2, cell / 2)
            if n > 1:
                c.scale(max(0.08, j / (n - 1)), 1.0)
            draw(c)
            c.restore()
        img = surf.makeImageSnapshot()
        if self.gr is not None:
            img = img.makeTextureImage(self.gr) or img
        hit = self._sprite[L.id] = (img, [skia.Rect.MakeXYWH(cell * j, 0, cell, cell) for j in range(n)], side,
                                    cell / 2)
        return hit

    def _sprite_of(self, L: Layer, s: float):
        """A function drawing L's particle in white, `s` px across its longer side, centred on the origin."""
        paint = skia.Paint(AntiAlias=True, Color=skia.ColorWHITE)
        h = s / 2
        if L.raw.get("src") is not None:
            img = media.image(L.src)
            k = s / max(img.width(), img.height())
            w, ht = img.width() * k, img.height() * k
            return lambda c: c.drawImageRect(img, skia.Rect.MakeXYWH(-w / 2, -ht / 2, w, ht), SAMPLING, paint)
        if L.raw.get("text") is not None:
            txt = str(L.raw["text"])
            g = lambda k, d=None: L.raw.get(k, d)  # noqa: E731
            font = self._font(L, g, s)
            m = font.getMetrics()
            font = self._font(L, g, s * s / max(self._measure(font, txt), m.fDescent - m.fAscent, 1e-6))
            m, w = font.getMetrics(), self._measure(font, txt)
            pcs = self._pieces(font, txt)
            return lambda c: self._draw_pieces(c, pcs, txt, -w / 2, -(m.fAscent + m.fDescent) / 2, font, paint)
        shape = L.particles.shape or "circle"
        if shape == "circle":
            return lambda c: c.drawCircle(0, 0, h, paint)
        if shape == "square":
            return lambda c: c.drawRect(skia.Rect(-h, -h, h, h), paint)
        if shape == "rect":
            return lambda c: c.drawRect(skia.Rect(-h, -h / 2, h, h / 2), paint)
        if shape == "ring":
            ring = skia.Paint(AntiAlias=True, Color=skia.ColorWHITE, Style=skia.Paint.kStroke_Style, StrokeWidth=s / 8)
            return lambda c: c.drawCircle(0, 0, h - s / 16, ring)
        if shape == "soft":
            soft = skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeRadial(
                skia.Point(0, 0), h, [0xFFFFFFFF, 0x99FFFFFF, 0x33FFFFFF, 0x00FFFFFF], [0.0, 0.35, 0.7, 1.0]))
            return lambda c: c.drawCircle(0, 0, h, soft)
        if shape == "streak":  # fading from the tail to its head, which leads (+x) when it turns with its path
            streak = skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeLinear(
                [skia.Point(-h, 0), skia.Point(h, 0)], [0x00FFFFFF, 0xFFFFFFFF]))
            return lambda c: c.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-h, -s / 10, h, s / 10), s / 10, s / 10),
                                         streak)
        path = skia.Path()
        points = ([(h * math.cos(a), h * math.sin(a)) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)]
                  if shape == "triangle" else
                  [((h if j % 2 == 0 else h * 0.45) * math.cos(-math.pi / 2 + j * math.pi / 5),
                    (h if j % 2 == 0 else h * 0.45) * math.sin(-math.pi / 2 + j * math.pi / 5)) for j in range(10)])
        path.addPoly([skia.Point(px, py) for px, py in points], True)
        return lambda c: c.drawPath(path, paint)

    def _model_surface(self):
        """The offscreen models are drawn in at their place in the frame, and its supersampling factor: 2x, which
        `check` and `--boxes` skip (they take 4x MSAA on the GPU instead, else none).
        It is the size of the frame from the start. One that grew to fit the largest model so far made a
        frame depend on the frames before it: on the GPU, edges moved by up to 49 levels.
        Rendering does not use MSAA on the GPU, although it costs no more: the same frame drawn twice came out
        different by one level at a few hundred edge pixels. Supersampling is repeatable and took the same time
        (examples/model.json 19 ms a frame against 20 at 1920x1080)."""
        if self._msurf is None:
            ctx = self.surface.recordingContext()
            ss = 1 if self.track else 2
            info = skia.ImageInfo.Make(self.w * ss, self.h * ss, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
            surf = skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, info, 4 if self.track else 0) if ctx else None
            self._msurf = surf or skia.Surface.MakeRaster(info), ss
        return self._msurf

    def _model_spare(self, k: int) -> skia.Surface:
        """Another surface like _model_surface's, the k-th, for a pass of a textured model."""
        if k not in self._mspare:
            surf = self._model_surface()[0]
            info, ctx = surf.imageInfo(), surf.recordingContext()
            self._mspare[k] = (ctx and skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, info,
                                                                     4 if self.track else 0)) or surf.makeSurface(info)
        return self._mspare[k]

    def _effect(self, L: Layer):
        hit = self._shaders.get(L.id)
        if hit:
            return hit
        from . import shaders
        defaults: dict = {}
        src = L.raw.get("sksl")
        if not src:
            name = L.raw.get("preset", "mesh")
            if name not in shaders.PRESETS:
                raise ValueError(f"layer '{L.id}': unknown shader preset '{name}'. Known: {', '.join(shaders.PRESETS)}")
            src, defaults = shaders.PRESETS[name]
        else:
            src = declare_uniforms(src, L.raw.get("uniforms") or {})
        try:
            eff = skia.RuntimeEffect.MakeForShader(src)
        except RuntimeError as e:
            errs = [x.strip() for x in str(e).splitlines() if "error" in x.lower()] or [str(e).strip()]
            more = f" (and {len(errs) - 1} more, often caused by the first)" if len(errs) > 1 else ""
            raise ValueError(f"layer '{L.id}': SkSL error: {errs[0]}{more}. A shader is "
                             "half4 main(float2 p) { ... return half4(r, g, b, 1); } with p in pixels") from e
        if eff is None:
            raise ValueError(f"layer '{L.id}': SkSL did not compile")
        self._shaders[L.id] = (eff, defaults)
        return eff, defaults

    # ---- text -----------------------------------------------------------------------------
    def _typeface(self, family: str | None, weight: int, italic: bool) -> skia.Typeface:
        key = (family, weight, italic)
        tf = self._tf.get(key)
        if tf is None:
            style = skia.FontStyle(int(weight), 5, skia.FontStyle.kItalic_Slant if italic else skia.FontStyle.kUpright_Slant)
            bundled = BUNDLED.get((family or "").lower())
            if bundled:
                best = min(bundled, key=lambda w: abs(w - int(weight)))
                tf = skia.Typeface.MakeFromFile(str(FONT_DIR / bundled[best]))
            elif family and (family.lower().endswith((".ttf", ".otf", ".ttc"))):
                tf = skia.Typeface.MakeFromFile(self.s.path(family))
            else:
                for fam in ([family] if family else []) + FONT_FALLBACK:
                    tf = skia.Typeface(fam, style)
                    if tf and tf.getFamilyName().lower() == fam.lower():
                        break
            self._tf[key] = tf
        return tf

    def _pad(self, L: Layer):
        box = L.raw.get("box")
        if not box:
            return (0.0, 0.0)
        pad = box.get("pad", [12, 20]) if isinstance(box, dict) else [12, 20]
        return (float(pad), float(pad)) if isinstance(pad, (int, float)) else (float(pad[0]), float(pad[1]))

    def _text_of(self, L: Layer, t, T, g) -> str:
        if L.type == "counter":
            v = float(g("value", 0))
            fmt = L.raw.get("format", "{:,.0f}")
            txt = _FMT.sub(lambda m: format(v, m.group(1)), fmt)
        else:
            txt = str(L.raw.get("text", ""))
        case = L.raw.get("case") or L.raw.get("textCase")
        if case == "upper":
            txt = txt.upper()
        elif case == "lower":
            txt = txt.lower()
        self._last_text = txt.replace("**", "")
        return txt

    def _font(self, L: Layer, g, size: float) -> skia.Font:
        fam = L.raw.get("font") or self.s.theme.get("font")
        key = (fam, int(g("weight", 600)), bool(L.raw.get("italic")), size)
        f = self._fonts.get(key)
        if f is None:
            f = self._fonts[key] = _text_font(self._typeface(*key[:3]), size)
        return f

    def _pieces(self, font: skia.Font, s: str, rtl: bool = False, ctx: tuple | None = None) -> list[Piece] | None:
        """How to draw s in `font`: None when drawString draws it as it is (every character in the font, nothing
        to shape or reorder), else its pieces from left to right. rtl: s is (part of) a right-to-left word; ctx: s
        is a whole word, with what is around it (from _bidi_order)."""
        tf = font.getTypeface()
        key = (tf.uniqueID(), font.getSize(), s, rtl, ctx)
        if key in self._shaped:
            return self._shaped[key]
        base, before, after = ctx or (rtl, None, None)
        turns = (base or "R" in (before, after)) and any(unicodedata.bidirectional(ch) != "L" for ch in s)
        out = None
        if turns or not all(font.textToGlyphs(s)) or any(map(_shapes, s)):
            out = self._lay_pieces(font, s, base, before, after)
        if len(self._shaped) > 4000:
            self._shaped.clear()
        self._shaped[key] = out
        return out

    def _lay_pieces(self, font: skia.Font, s: str, rtl: bool, before: str | None = None,
                    after: str | None = None) -> list[Piece]:
        """s as pieces from left to right. Each cluster takes the layer's font, or the system font that has it. Only
        what needs it is shaped: a cluster of several code points (a letter and its marks, an emoji sequence, a
        flag) and a run of joining letters (an Arabic word); neither holds a place to break a line, so the shaper
        never wraps one. Right-to-left letters are put in reading order here, word direction from _bidi_order.
        rtl: the text around reads right to left; before, after: the direction of the text on either side, 'R' or
        'L' (default: rtl's), which punctuation at the ends of s takes when both agree."""
        tf, size = font.getTypeface(), font.getSize()
        units: list[list] = []  # [text, typeface, shaped, direction (R, L or N)]
        for cl in clusters(s):
            use = _face(tf, cl)
            d = _strong(cl) or "N"
            joins = joining(cl)
            if joins and units and units[-1][1] is use and units[-1][2] == "join":
                units[-1][0] += cl
                continue
            units.append([cl, use, "join" if joins else "shape" if len(cl) > 1 else "plain", d])
        # Reading order to left-to-right order: digits read left to right; a neutral (punctuation, emoji) takes the
        # direction of the letters or digits on both sides when they agree (past the ends of s, `before` and
        # `after`), else the direction around (rtl). Then runs at a higher level are reversed (_visual).
        bidi = [{unicodedata.bidirectional(ch) for ch in u[0]} for u in units]
        kinds = [u[3] if u[3] != "N" or not b & {"EN", "AN"} else "EN" for u, b in zip(units, bidi)]
        edge = "R" if rtl else "L"
        before, after = before or edge, after or edge
        for i, k in enumerate(kinds):  # 1,5 and 12% and $3 are one number (rules W4, W5)
            near = {kinds[i - 1] if i else None, kinds[i + 1] if i + 1 < len(kinds) else None}
            if k == "N" and bidi[i] <= {"ES", "CS", "ET"} and "EN" in near and (
                    bidi[i] <= {"ET"} or kinds[i - 1:i] == ["EN"] == kinds[i + 1:i + 2]):
                kinds[i] = "EN"
        strong = before
        for i, k in enumerate(kinds):  # digits after left-to-right letters are left to right (W7)
            if k in ("R", "L"):
                strong = k
            elif k == "EN" and strong == "L":
                kinds[i] = "L"
        for i, k in enumerate(kinds):  # neutrals (N1, N2): digits count as right to left here
            if k == "N":
                b = next((kinds[j] for j in range(i - 1, -1, -1) if kinds[j] != "N"), before)
                a = next((kinds[j] for j in range(i + 1, len(kinds)) if kinds[j] != "N"), after)
                b, a = ("R" if x in ("R", "EN") else "L" for x in (b, a))
                kinds[i] = b if a == b else edge
        order, odd = _visual(kinds, rtl)
        out: list[Piece] = []
        for i in order:
            text, use, kind, _ = units[i]
            f = font if use is tf else self._fallback_font(use, size)
            if kind == "plain":
                if odd[i]:  # a bracket in right-to-left text opens the other way: ( is drawn as )
                    text = text.translate(_MIRROR)
                if out and out[-1].blob is None and out[-1].font is f:
                    out[-1].text += text
                    out[-1].width = f.measureText(out[-1].text)
                else:
                    out.append(Piece(text, f, None, f.measureText(text), 0.0))
                continue
            blob = _shape(text, f, _strong(text) != "R")
            width = sum(sum(f.getWidths(list(run.fGlyphIndices))) for run in blob) if blob is not None else 0.0
            out.append(Piece(text, f, blob, width, f.getMetrics().fAscent))
        return out

    def _fallback_font(self, tf: skia.Typeface, size: float) -> skia.Font:
        key = (tf.uniqueID(), size)
        f = self._fonts.get(key)
        if f is None:
            f = self._fonts[key] = _text_font(tf, size)
        return f

    def _measure(self, font: skia.Font, s: str, rtl: bool = False, ctx: tuple | None = None) -> float:
        pcs = self._pieces(font, s, rtl, ctx)
        return font.measureText(s) if pcs is None else sum(p.width for p in pcs)

    @staticmethod
    def _draw_pieces(c: skia.Canvas, pcs: list[Piece] | None, s: str, x: float, y: float, font: skia.Font,
                     paint: skia.Paint):
        """s at (x, y), drawn as its pieces say (_pieces(font, s, ...); None: as it is)."""
        if pcs is None:
            c.drawString(s, x, y, font, paint)
            return
        for p in pcs:
            if p.blob is None:
                c.drawString(p.text, x, y, p.font, paint)
            else:
                c.drawTextBlob(p.blob, x, y + p.drop, paint)
            x += p.width

    def undrawable(self, L: Layer) -> tuple[list[str], list[str], list[str]]:
        """Characters of a text, counter, captions or 3D text layer that no installed font has (they draw as empty
        boxes), and those 3D text cannot draw as written: colour emoji (the first character of each), whose font has
        no outline for them, and with letterSpacing (each character on its own) marks, right-to-left letters and
        emoji parts."""
        fam = L.raw.get("font") or self.s.theme.get("font")
        tf = self._typeface(fam, int(L.get("weight", 0.0, 0.0, 0, 600) or 600), bool(L.raw.get("italic")))
        if L.type == "captions":
            text = " ".join(str(q[2]) for q in L.raw.get("cues") or [] if len(q) > 2)
        elif L.type == "counter":
            text = _FMT.sub("0123456789,.-", str(L.raw.get("format", "{:,.0f}")))
        else:
            text = str(L.raw.get("text", ""))
        text = text.replace("**", "")
        missing, flat = [], []
        spaced = L.raw.get("letterSpacing") not in (None, 0)
        for ch in dict.fromkeys(text):
            if ch in "\n" or _ignorable(ch):
                continue
            if not _covers(tf, ord(ch)) and _fallback(tf.fontStyle(), ord(ch)) is None:
                missing.append(ch)
            elif L.type == "model" and spaced and _shapes(ch):
                flat.append(ch)
        emoji: list[str] = []
        for cl in dict.fromkeys(clusters(text) if L.type == "model" else []):
            ch = next((c for c in cl if not _ignorable(c)), " ")
            if not ch.isspace() and ch not in missing and ch not in emoji:
                face = _face(tf, cl)
                if skia.Font(face, 12).getPath(face.unicharToGlyph(ord(ch))) is None:
                    emoji.append(ch)
        return missing, emoji, [ch for ch in flat if ch not in emoji]

    def _text_layout(self, L: Layer, t, T, g, txt: str, size: float | None = None, max_w=None) -> Layout:
        size = float(size if size is not None else g("size", 64))
        ls = float(g("letterSpacing", 0))
        mw = max_w if max_w is not None else g("maxWidth")
        lhm = float(g("lineHeight", 1.18))
        # Every input that changes the layout belongs in the key, or a frame depends on which frames came before.
        key = (L.id, txt, size, ls, mw, lhm, int(g("weight", 600)))
        lay = self._layouts.get(key)
        if lay:
            return lay
        font = self._font(L, g, size)
        meas = lambda s, rtl=False: self._measure(font, s, rtl) + ls * len(s)  # noqa: E731
        split = L.raw.get("split")
        apart = split == "chars" or ls  # each letter drawn on its own

        def wmeas(w: str) -> float:
            """A word's width as drawn: letter by letter when they are drawn apart and the word needs more than
            drawString (Arabic letters unjoined are wider)."""
            if apart and self._pieces(font, w) is not None:
                return sum(meas(ch) for ch in clusters(w))
            return meas(w)
        space = meas(" ")
        lines: list[list] = []
        widths: list[float] = []
        wi = 0
        for para in txt.split("\n"):
            rtl_para = _strong(para.replace("**", "")) == "R"
            first = len(lines)
            emph = False
            words = []
            for tok in para.split(" "):
                if tok == "":
                    continue
                # Each ** turns emphasis on or off, inside a word too: g**P**u colours only the P.
                segs, start = [], 0  # (start in the word, text, emphasised)
                for j, part in enumerate(tok.split("**")):
                    if j:
                        emph = not emph
                    if part:
                        segs.append((start, part, emph))
                        start += len(part)
                words.append(("".join(p for _, p, _ in segs), segs or [(0, "", emph)]))
            indent = (len(para) - len(para.lstrip(" "))) * space
            cur, cw = [], indent
            for w_, segs in words:
                ww = wmeas(w_)
                if cur and mw and cw + space + ww > float(mw):
                    lines.append(cur)
                    widths.append(cw)
                    cur, cw = [], 0.0
                x = cw + (space if cur else 0.0)
                cur.append((w_, segs, x, wi, False, None))
                wi += 1
                cw = x + ww
            lines.append(cur)
            widths.append(cw)
            for li in range(first, len(lines)):  # words in reading order, placed in the order they are seen
                ln = lines[li]
                if ln and (rtl_para or any(_strong(e[0]) == "R" for e in ln)):
                    order, rtls, around = _bidi_order([e[0] for e in ln], rtl_para)
                    pos, placed = ln[0][2], list(ln)
                    for k in order:
                        w_, segs, _, idx, _, _ = ln[k]
                        placed[k] = (w_, segs, pos, idx, rtls[k], around[k])
                        pos += wmeas(w_) + space
                    lines[li] = placed
        m = font.getMetrics()
        lh = size * lhm
        baseline0 = (lh - (m.fDescent - m.fAscent)) / 2 - m.fAscent
        W = max(widths) if widths else 0.0
        align = L.raw.get("align", "center")
        k = {"left": 0.0, "center": 0.5, "right": 1.0}.get(align, 0.5)
        runs = []
        ci = 0
        for li, (ln, lw) in enumerate(zip(lines, widths)):
            off = (W - lw) * k
            for w_, segs, x, idx, rtl, ctx in ln:
                # A right-to-left word's letters and emphasis runs go from its right edge leftwards.
                ww = wmeas(w_) if rtl else 0.0
                if apart:
                    cx = 0.0
                    for _, part, e_ in segs:
                        for ch in clusters(part):  # an accent or an emoji sequence stays with its letter
                            wc = meas(ch)
                            runs.append(Run(ch, off + x + (ww - cx - wc if rtl else cx), li, e_, idx, ci, rtl))
                            cx += wc
                            ci += len(ch)
                elif len(segs) == 1:
                    runs.append(Run(w_, off + x, li, segs[0][2], idx, ci, rtl, ctx))
                    ci += len(w_)
                else:  # one run per emphasis change, placed where it falls in the whole word
                    for s0, part, e_ in segs:
                        px = ww - meas(w_[:s0]) - meas(part) if rtl else meas(w_[:s0])
                        runs.append(Run(part, off + x + px, li, e_, idx, ci + s0, rtl))
                    ci += len(w_)
                ci += 1  # the space
        lay = Layout(runs, widths, W, lh * len(lines), lh, baseline0, ci, wi)
        if len(self._layouts) > 2000:
            self._layouts.clear()
        self._layouts[key] = lay
        return lay

    def _draw_counter(self, c, L, t, T, g, box, reveal, mods):
        self._draw_text(c, L, t, T, g, box, reveal, mods)

    def _blur_bounds(self, c: skia.Canvas, sigma: float, right: float, bottom: float, size: float,
                     spaint: skia.Paint | None) -> skia.Rect | None:
        """Bounds for the layer that blurs a text unit whose advances and lines end at (right, bottom), or None.
        Without bounds the layer is the canvas: on the CPU a blurred 1080p layer took 370 ms, for one letter. It is
        bounded at the right and bottom only, so it starts where an unbounded layer does: bounded on all sides, the
        letters in it sat at other coordinates, and those on a subpixel boundary moved a quarter of a pixel (63
        levels). On the GPU any bounds change the blur itself, by up to 9 levels, so it stays unbounded there."""
        m = c.getTotalMatrix()
        if self.backend != "cpu" or not (m.isScaleTranslate() and m.getScaleX() > 0 and m.getScaleY() > 0):
            return None
        # Glyphs reach less than an em past their advances and line, a stroke its width, the blur 3 sigma.
        out = 3 * sigma + size + (spaint.getStrokeWidth() if spaint is not None else 0.0)
        far = -1e6  # device pixels: left of and above any layer
        return skia.Rect.MakeLTRB((far - m.getTranslateX()) / m.getScaleX(), (far - m.getTranslateY()) / m.getScaleY(),
                                  right + out, bottom + out)

    def _units(self, lay: Layout, font: skia.Font, split) -> list:
        """The runs grouped into the units a split enter moves one by one, in order: (runs, left, right) with each
        run as (run, its pieces). They are the same on every frame the layout is, so they are kept with it."""
        if lay.units is None:
            units: dict[int, list] = {}
            for r in lay.runs:
                u = r.char0 if split == "chars" else r.word if split == "words" else r.line if split == "lines" else 0
                units.setdefault(u, []).append((r, self._pieces(font, r.text, r.rtl, r.ctx)))
            lay.units = [(runs, min(r.x for r, _ in runs),
                          max(r.x + self._measure(font, r.text, r.rtl, r.ctx) for r, _ in runs))
                         for _, runs in sorted(units.items())]
        return lay.units

    def _draw_text(self, c, L, t, T, g, box, reveal, mods, txt=None, layout=None, active=None, cue_t=None):
        txt = txt if txt is not None else self._text_of(L, t, T, g)
        lay = layout or self._text_layout(L, t, T, g, txt)
        size = float(g("size", 64))
        font = self._font(L, g, size)
        bx, by, bw, bh = box
        pad = self._pad(L)
        boxspec = L.raw.get("box")
        if boxspec and self._capture is None:
            bs = boxspec if isinstance(boxspec, dict) else {}
            r = float(bs.get("radius", 10))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r),
                        skia.Paint(AntiAlias=True, Color=_argb(bs.get("fill", "#000000b0"))))
        ox, oy = bx + pad[1], by + pad[0]
        fill = self._own(L, "fill") or (g("fill") if "fill" in L.props else L.raw["fill"]
                                        if isinstance(L.raw.get("fill"), dict)
                                        else C.parse(self.s.theme.get("paper", "#f2f0e8")))
        gpaint = None  # a gradient fill, across the text as a shape's goes across its box; emphasis stays a colour
        if isinstance(fill, (dict, C.Gradient)):
            gpaint = skia.Paint(AntiAlias=True, Shader=self._gradient(fill if isinstance(fill, dict) else fill.spec(),
                                                                      ox, oy, lay.w, lay.h, 1.0))
            fill = (1.0, 1.0, 1.0, 1.0)  # only its alpha counts under the shader
        emph = g("emph") if "emph" in L.props else C.parse(self.s.theme.get("accent", "#ff633f"))
        act = g("active") if "active" in L.props else emph
        fpaint = skia.Paint(AntiAlias=True)
        spaint = self._stroke_paint(L, g, box)
        split = L.raw.get("split")
        enter = L.enter if split else None
        stagger = (enter.stagger if enter and enter.stagger else 0.04) if split else 0
        units = self._units(lay, font, split)
        turn = None  # each unit's turn in the stagger, when it is not its place in the text
        if enter is not None and enter.order != "start":
            turn = P.turns(len(units), enter.order, enter.seed)
        typing = enter is None and L.enter and "type" in L.enter.names and t < L.enter.dur
        visible_chars = lay.nchars * reveal if (reveal < 1 or typing) else None
        track = mods.track
        c00 = lay.runs[0].char0 if lay.runs else 0
        unit_keys = {k for k in ("unitX", "unitY", "unitRotate", "unitScale", "unitOpacity") if k in L.props}
        cols, pop = (fill, emph, act), active is not None and L.raw.get("mode") == "pop"  # captions only
        painted = None  # (which colour, alpha) fpaint holds
        for ui, (runs, x0, x1) in enumerate(units):
            um = IDENT
            if enter is not None:
                p = (t - (ui if turn is None else turn[ui]) * stagger) / enter.dur if enter.dur else 1.0
                if p < 1:
                    um = enter.mods(p)
            if um.alpha <= 0.002:
                continue
            ux = um.dx + (float(L.get("unitX", t, T, ui)) if "unitX" in unit_keys else 0)
            uy = um.dy + (float(L.get("unitY", t, T, ui)) if "unitY" in unit_keys else 0)
            ua = um.alpha * (float(L.get("unitOpacity", t, T, ui)) if "unitOpacity" in unit_keys else 1)
            urot = um.rotate + (float(L.get("unitRotate", t, T, ui)) if "unitRotate" in unit_keys else 0)
            usc = um.scale * (float(L.get("unitScale", t, T, ui)) if "unitScale" in unit_keys else 1)
            moved = ux or uy or urot or usc != 1 or um.rx or um.ry or um.dz
            blurred = um.blur > 0.1
            if moved:  # a unit drawn where it lies needs no save and restore: most of them, once they are in
                c.save()
                ucx, ucy = ox + (x0 + x1) / 2, oy + runs[0][0].line * lay.lh + lay.lh / 2
                c.translate(ucx + ux, ucy + uy)
                if um.rx or um.ry or um.dz:
                    c.concat(space.to_skia(space.tilt(um.dz, um.rx, um.ry)))
                if urot:
                    c.rotate(urot)
                if usc != 1:
                    c.scale(usc, usc)
                c.translate(-ucx, -ucy)
            if blurred:
                bounds = self._blur_bounds(c, um.blur, ox + x1 + max(0.0, track * (runs[-1][0].char0 - c00)),
                                           oy + (runs[-1][0].line + 1) * lay.lh, size, spaint)
                area = skia.Rect.Make(c.getDeviceClipBounds())  # the layer: the clip, as far as the bounds reach
                if bounds is None or area.intersect(c.getTotalMatrix().mapRect(bounds)):
                    self.filtered += area.width() * area.height()
                c.saveLayer(bounds, skia.Paint(ImageFilter=skia.ImageFilters.Blur(um.blur, um.blur)))
            for r, pcs in runs:
                text = r.text
                if visible_chars is not None:
                    if r.char0 >= visible_chars:
                        continue
                    text = text[: max(0, int(visible_chars - r.char0))]
                k = 0
                if active is not None:
                    if r.word <= active:
                        k = 2
                    elif pop:
                        continue
                elif r.emph:
                    k = 1
                if painted != (k, ua):
                    fpaint.setColor(_argb(cols[k], ua))
                    if gpaint is not None:
                        gpaint.setAlphaf(ua)
                    painted = (k, ua)
                ink = gpaint if gpaint is not None and k == 0 else fpaint
                y = oy + r.line * lay.lh + lay.baseline0
                x = ox + r.x + track * (r.char0 - c00) if track else ox + r.x
                if text != r.text:  # typed so far: a right-to-left word grows from its right edge
                    if r.rtl:
                        x += self._measure(font, r.text, True, r.ctx) - self._measure(font, text, True, r.ctx)
                    pcs = self._pieces(font, text, r.rtl, r.ctx)
                if self._capture is not None:  # a combining group takes the letters' outlines
                    self._take(c, self._letters(pcs, text, x, y, font))
                    continue
                if spaint is not None:
                    self._draw_pieces(c, pcs, text, x, y, font, spaint)
                self._draw_pieces(c, pcs, text, x, y, font, ink)
            if blurred:
                c.restore()
            if moved:
                c.restore()
        if visible_chars is not None and L.raw.get("caret") and (int(T * 2) % 2 == 0 or reveal < 1) and (
                self._capture is None):
            last = None
            for r in lay.runs:
                if r.char0 < visible_chars:
                    last = r
            if last is not None:
                vis = last.text[: max(0, int(visible_chars - last.char0))]
                cw = max(3, size * 0.06)
                if last.rtl:  # left of what is typed so far
                    cx = (ox + last.x + self._measure(font, last.text, True, last.ctx)
                          - self._measure(font, vis, True, last.ctx) - 4 - cw)
                else:
                    cx = ox + last.x + self._measure(font, vis) + 4
                cy = oy + last.line * lay.lh
                c.drawRect(skia.Rect.MakeXYWH(cx, cy + lay.lh * 0.15, cw, lay.lh * 0.7),
                           skia.Paint(Color=_argb(emph)))

    def _draw_captions(self, c, L, t, T, g, box, reveal, mods):
        cues = L.raw.get("cues") or []
        cue = next((q for q in cues if q[0] <= t < q[1]), None)
        if cue is None:
            self._last_text = ""
            return
        s0, s1, text = cue[:3]
        mw = g("maxWidth") or self.s.w * 0.8
        lay = self._text_layout(L, t, T, g, text, max_w=mw)
        pad = self._pad(L) if L.raw.get("box") is not None else (0.0, 0.0)
        if L.raw.get("box") is None:
            L.raw["box"] = {"fill": "#000000a8", "pad": [14, 24], "radius": 12}
            pad = self._pad(L)
        w, h = lay.w + pad[1] * 2, lay.h + pad[0] * 2
        cbox = (-w / 2, -h / 2, w, h)
        ct = t - s0
        lines = set(text.split("\n"))
        # Cues that overlap are stacked into segments (components._stack); a line that goes on into the next
        # segment does not fade out and back in where they meet.
        goes_on = [any(abs(q[k] - edge) < 1e-6 and lines & set(q[2].split("\n")) for q in cues)
                   for k, edge in ((1, s0), (0, s1))]
        a = min(1.0, 1.0 if goes_on[0] else ct / 0.12, 1.0 if goes_on[1] else (s1 - t) / 0.12)
        active = None
        if L.raw.get("mode") in ("karaoke", "pop") and len(cue) > 3:  # word times from the voice
            first_word = lay.runs[0].word if lay.runs else 0
            active = first_word + sum(1 for ws, _ in cue[3] if ws <= t) - 1
        elif L.raw.get("mode") in ("karaoke", "pop"):
            words = text.replace("**", "").split()
            total = sum(len(wd) for wd in words) or 1
            frac = ct / max(0.05, (s1 - s0) * 0.92)
            acc, active = 0, -1
            first_word = lay.runs[0].word if lay.runs else 0
            for k, wd in enumerate(words):
                if acc / total <= frac:
                    active = first_word + k
                acc += len(wd)
        c.save()
        if a < 1:
            c.saveLayer(None, skia.Paint(Alphaf=max(0.0, a)))
        self._draw_text(c, L, t, T, g, cbox, 1.0, IDENT, txt=text, layout=lay, active=active)
        if self.track:
            m = c.getTotalMatrix()
            r = m.mapRect(skia.Rect.MakeXYWH(*cbox))
            self.boxes[L.id] = {"rect": [round(r.left(), 1), round(r.top(), 1), round(r.right(), 1), round(r.bottom(), 1)],
                                "alpha": round(a, 3), "type": "captions", "size": float(g("size", 64)),
                                "fill": list(f) if isinstance(f := g("fill") if "fill" in L.props else None, tuple)
                                else None, "text": text, "scale": 1.0}
        if a < 1:
            c.restore()
        c.restore()
