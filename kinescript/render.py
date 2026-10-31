"""Frame-pure renderer: frame n is a function of the scene and n alone."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skia

from . import color as C
from . import media, render3d, space, svgpath
from .presets import IDENT, Mods
from .scene import Layer, Scene

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
BLENDS = {"normal": skia.BlendMode.kSrcOver, "multiply": skia.BlendMode.kMultiply, "screen": skia.BlendMode.kScreen,
          "overlay": skia.BlendMode.kOverlay, "add": skia.BlendMode.kPlus, "lighten": skia.BlendMode.kLighten,
          "darken": skia.BlendMode.kDarken, "softLight": skia.BlendMode.kSoftLight,
          "difference": skia.BlendMode.kDifference, "color": skia.BlendMode.kColor}
SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kNone)
# Video frames arrive at source size and upscale here. Catmull-Rom interpolates (a 1:1 draw is
# exact); Mitchell, used before, blurred every native-size frame (mean error 14.8 vs 4.7 on a sharp clip).
CUBIC = skia.SamplingOptions(skia.CubicResampler.CatmullRom())
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


def _blur_sksl(n: int) -> str:
    adds = "".join(f"t += floor(float4(s{k}.eval(p)) * 255 + 0.5);" for k in range(n))
    return ("".join(f"uniform shader s{k};" for k in range(n))
            + f"float4 main(float2 p) {{ float4 t = float4(0); {adds} return floor((t + 0.5) / {n}.0) / 255; }}")


# Premultiplied to straight alpha with the integer rounding blurred frames have always had (_straighten).
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


def gpu_context():
    """One hidden OpenGL context per process, or None when there is no usable GPU.
    Set KINE_BACKEND=cpu to force the raster backend."""
    if "ctx" in _GL:
        return _GL["ctx"]
    _GL["ctx"] = None
    if os.environ.get("KINE_BACKEND", "").lower() == "cpu":
        return None
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
        _GL["ctx"] = skia.GrDirectContext.MakeGL()
    except (ImportError, RuntimeError, OSError):
        _GL["ctx"] = None
    return _GL["ctx"]


@dataclass
class Run:
    text: str
    x: float
    line: int
    emph: bool
    word: int
    char0: int


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


class Renderer:
    def __init__(self, scene: Scene, track: bool = False, scale: float = 1.0, backend: str = "auto",
                 even: bool = False):
        self.s = scene
        self.scale = scale
        self.w, self.h = max(2, round(scene.w * scale)), max(2, round(scene.h * scale))
        if even:  # 4:2:0 video needs even sizes
            self.w, self.h = self.w - self.w % 2, self.h - self.h % 2
        self.info = skia.ImageInfo.Make(self.w, self.h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        ctx = gpu_context() if backend in ("auto", "gpu") else None
        if backend == "gpu" and ctx is None:
            raise RuntimeError("GPU backend requested but no OpenGL context could be created")
        self.surface = skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, self.info) if ctx else None
        self.backend = "gpu" if self.surface is not None else "cpu"
        if self.surface is None:
            self.surface = skia.Surface.MakeRaster(self.info)
        self.buf = np.empty((self.h, self.w, 4), np.uint8)
        # Encoders take straight (unpremultiplied) RGBA; reading premultiplied pixels into a transparent
        # .mov/.webm darkens every soft edge. Opaque frames are identical either way.
        self.transparent = scene.bg in (None, "transparent", "none")
        self.read_info = self.info.makeAlphaType(skia.kUnpremul_AlphaType) if self.transparent else self.info
        self.track = track
        self.boxes: dict[str, dict] = {}
        self._tf: dict = {}
        self._layouts: dict = {}
        self._fonts: dict = {}
        self._paths: dict = {}
        self._shaders: dict = {}
        self.stop_at: str | None = None
        self._stopped = False
        self._m3d: np.ndarray | None = None  # world-to-canvas matrix of the layer being drawn, in 3D scenes
        self.culled: set[str] = set()  # layers skipped this frame because they cross the camera plane
        self._lens: np.ndarray | None = None  # camera space to canvas, for lighting models
        self._models: dict = {}  # merged parts of each model layer
        self._solids: dict = {}  # outlines and extruded meshes of text and path models
        self._outer = None  # the 3D matrix of the layer being drawn's parent (models put their floor in it)
        self._msurf = None  # offscreen that models draw into, antialiased, before compositing
        self.tris = 0  # model triangles drawn this frame
        self._subframes = None  # motion blur on the GPU: (count, averaging effect, a surface per sub-frame)
        self._straight = None  # (effect, surface) straightening blurred transparent frames on the GPU
        cm = color_matrix({k: v for k, v in scene.grade.items() if k not in ("vignette", "grain")})
        self.grade_filter = skia.ColorFilters.Matrix(cm) if cm else None
        self._videos = [L for L in scene.all_layers() if L.type == "video"]

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
        times = [T + (k / (samples - 1) - 0.5) * shutter for k in range(samples)]
        if self.backend == "gpu" and samples <= BLUR_GPU:
            self._blur_gpu(times)
        else:
            self._blur_cpu(times)
        return samples

    def _blur_gpu(self, times: list[float]):
        n = len(times)
        if self._subframes is None or self._subframes[0] != n:
            ctx = self.surface.recordingContext()
            self._subframes = (n, skia.RuntimeEffect.MakeForShader(_blur_sksl(n)),
                               [skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, self.info) for _ in range(n)])
        _, eff, surfs = self._subframes
        main = self.surface
        try:
            for t, surf in zip(times, surfs):
                self.surface = surf  # draw() draws into self.surface
                self.draw(t)
        finally:
            self.surface = main
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

    def _blur_cpu(self, times: list[float]):
        # 257 sub-frames of 255 still fit in 16 bits: half the memory traffic of 32-bit sums, same mean.
        acc = np.zeros((self.h, self.w, 4), np.uint16 if len(times) <= 257 else np.uint32)
        for t in times:
            self.draw(t)
            self.surface.readPixels(self.info, self.buf, self.w * 4)
            np.add(acc, self.buf, out=acc)
        np.floor_divide(acc, len(times), out=acc)
        np.copyto(self.buf, acc, casting="unsafe")
        self.surface.writePixels(skia.Pixmap(self.info, self.buf, self.w * 4))  # stills save the surface

    @property
    def yuv(self) -> bool:
        """Whether frame_i420 runs on the GPU: an opaque frame with width a multiple of 4."""
        return self.backend == "gpu" and not self.transparent and self.w % 4 == 0 and self.h % 2 == 0

    def frame_i420(self, n: int) -> np.ndarray:
        """Frame n as an I420 buffer (BT.709, limited range), reused between calls. Needs self.yuv."""
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
        surf.readPixels(info, buf, self.w)
        return buf

    def snapshot(self) -> skia.Image:
        return self.surface.makeImageSnapshot()

    def draw(self, T: float, stop_at: str | None = None):
        s = self.s
        self.boxes = {}
        self.culled = set()
        self.tris = 0
        self.stop_at, self._stopped = stop_at, False
        c = self.surface.getCanvas()
        c.resetMatrix()
        c.clear(skia.ColorTRANSPARENT)
        if self.scale != 1:
            c.scale(self.scale, self.scale)
        self._background(c)
        if self.grade_filter:
            c.saveLayer(None, skia.Paint(ColorFilter=self.grade_filter))
        cams = self._cameras(T) if s.is3d else None
        for layer in s.layers:
            if self._stopped:
                break
            m = None
            if cams is not None:
                fixed = layer.raw.get("fixed", layer.type == "captions")
                if s.camera is not None and not fixed:
                    m = cams[0]
                elif layer.id in s.depth:
                    m = cams[1]
            self._m3d = m
            if m is None:
                self._layer(c, layer, T, 1.0)
            else:
                c.save()
                c.concat(space.to_skia(m))
                self._layer(c, layer, T, 1.0)
                c.restore()
        self._m3d = None
        if self.grade_filter:
            c.restore()
        if not stop_at:
            self._finish(c, T)
        # A clip's decoder holds an ffmpeg process and its prefetched frames; once the clip is off screen,
        # free them, or a cut list keeps every clip it has played open to the end (60 cuts: 1.6 GB).
        for L in self._videos:
            if L.video is not None and L.video.proc is not None and not L.t_in <= T < L.t_out:
                L.video.close()

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
        return space.camera(s.w, s.h, fov, g("x"), g("y"), g("z"), g("rotateX"), g("rotateY"), g("rotate")), fixed

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
        if not L.visible(T):
            return
        t = T - L.t_in
        d = L.dur
        g = lambda name, default=None: L.get(name, t, T, 0, default)  # noqa: E731
        split = L.raw.get("split") if L.type in ("text", "counter") else None
        mods = IDENT
        if L.enter and not split and t < L.enter.dur:
            mods = L.enter.mods(t / L.enter.dur)
        if L.exit and d - t < L.exit.dur:
            mods = mods.combine(L.exit.mods((d - t) / L.exit.dur))
        alpha = max(0.0, min(1.0, float(g("opacity")) * mods.alpha))
        if alpha <= 0.002:
            return
        box = self._box(L, t, T, g)
        x, y = float(g("x")) + mods.dx, float(g("y")) + mods.dy
        rot = float(g("rotate")) + mods.rotate
        sc = float(g("scale")) * mods.scale
        sx, sy = sc * float(g("scaleX")), sc * float(g("scaleY"))
        outer, tilt, facing = self._m3d, None, True
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
            self._m3d, self._outer = m, outer
        c.save()
        c.translate(x, y)
        if tilt is not None:
            c.concat(space.to_skia(tilt))
        if rot:
            c.rotate(rot)
        if sx != 1 or sy != 1:
            c.scale(sx, sy)
        blur = float(g("blur")) + mods.blur
        shadow = L.raw.get("shadow")
        fx = L.raw.get("fx")
        blend = L.raw.get("blend")
        layered = alpha < 0.999 or blur > 0.05 or shadow or fx or (blend and blend != "normal")
        if layered:
            paint = skia.Paint(Alphaf=alpha)
            filt = None
            if blur > 0.05:
                filt = skia.ImageFilters.Blur(blur, blur)
            if shadow:
                sh = shadow if isinstance(shadow, dict) else {}
                sig = float(sh.get("blur", 16)) / 2
                filt = skia.ImageFilters.DropShadow(float(sh.get("x", 0)), float(sh.get("y", 6)), sig, sig,
                                                    _argb(sh.get("color", "#00000099")), filt)
            if filt:
                paint.setImageFilter(filt)
            if fx:
                cm = color_matrix(fx)
                if cm:
                    paint.setColorFilter(skia.ColorFilters.Matrix(cm))
            if blend:
                paint.setBlendMode(BLENDS.get(blend, skia.BlendMode.kSrcOver))
            c.saveLayer(None, paint)
        if box is not None:
            bx, by, bw, bh = box
            if mods.wipe and mods.reveal < 1:
                r = max(0.0, mods.reveal)
                clip = {"right": (bx, by, bx + bw * r, by + bh), "left": (bx + bw * (1 - r), by, bx + bw, by + bh),
                        "down": (bx, by, bx + bw, by + bh * r), "up": (bx, by + bh * (1 - r), bx + bw, by + bh)}[mods.wipe]
                c.clipRect(skia.Rect(*clip), doAntiAlias=True)
        reveal = mods.reveal if not mods.wipe else 1.0
        draw = getattr(self, "_draw_" + L.type)
        drawn = draw(c, L, t, T, g, box, reveal, mods)  # a model returns its projected bounds
        if self.track and (box is not None or drawn):
            m = c.getTotalMatrix()
            r = skia.Rect(*drawn) if drawn else m.mapRect(skia.Rect.MakeXYWH(*box))
            sc = 1.0 if drawn else math.sqrt(abs(m.getScaleX() * m.getScaleY())) or abs(m.getSkewX())
            info = {"rect": [round(r.left(), 1), round(r.top(), 1), round(r.right(), 1), round(r.bottom(), 1)],
                    "alpha": round(alpha * parent_alpha, 3), "type": L.type, "scale": round(sc, 4)}
            if L.type in ("text", "counter", "captions"):
                # The colour _draw_text uses, so checks see default-coloured text too.
                fill = g("fill") if "fill" in L.props else C.parse(self.s.theme.get("paper", "#f2f0e8"))
                info.update(size=float(g("size", 64)), fill=list(fill) if isinstance(fill, tuple) else None,
                            text=self._last_text if hasattr(self, "_last_text") else "")
            if not facing:
                info["mirrored"] = True
            self.boxes[L.id] = info
        if L.type == "group":
            kids = L.children
            if self._m3d is not None and L.raw.get("sort") == "depth":
                kids = sorted(kids, key=lambda k: self._far(k, T))
            for child in kids:
                if self._stopped:
                    break
                self._layer(c, child, T, parent_alpha * alpha)
        if layered:
            c.restore()
        c.restore()
        self._m3d = outer

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
        elif ty in ("captions", "model"):
            return None  # a model's bounds come from its projected mesh (render3d.draw)
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
        a, b = spec.get("linear", [[0, 0], [1, 1]])
        return skia.GradientShader.MakeLinear([skia.Point(x + a[0] * w, y + a[1] * h), skia.Point(x + b[0] * w, y + b[1] * h)],
                                              colors, pos)

    def _fill_paint(self, L: Layer, g, box, alpha=1.0, key="fill", default=None):
        raw = L.raw.get(key, default)
        if raw is None:
            return None
        p = skia.Paint(AntiAlias=True)
        if isinstance(raw, dict):
            bx, by, bw, bh = box
            p.setShader(self._gradient(raw, bx, by, bw, bh, alpha))
        else:
            col = g(key) if key in L.props else C.parse(raw) if isinstance(raw, str) else raw
            p.setColor(_argb(col, alpha))
        return p

    def _stroke_paint(self, L: Layer, g, alpha=1.0):
        if L.raw.get("stroke") is None:
            return None
        col = g("stroke") if "stroke" in L.props else C.parse(L.raw["stroke"])
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=float(g("strokeWidth", 4)))
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

    def _shape(self, c, L, g, path: skia.Path, box, reveal):
        """Fill then stroke. 'draw' (0..1) trims the stroke and fades the fill with it."""
        amt = max(0.0, min(1.0, float(g("draw")) * reveal))
        has_stroke = L.raw.get("stroke") is not None
        fp = self._fill_paint(L, g, box, alpha=amt, default=None if has_stroke else self.s.theme.get("paper", "#f2f0e8"))
        if fp is not None and amt > 0:
            c.drawPath(path, fp)
        sp = self._stroke_paint(L, g)
        if sp is not None and amt > 0.0005:
            c.drawPath(self._trim(path, amt, float(g("drawStart", 0.0))), sp)

    # ---- layer types ----------------------------------------------------------------------
    def _draw_group(self, c, L, t, T, g, box, reveal, mods):
        if box is not None and L.raw.get("clip"):
            r = float(g("radius", 0))
            rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r)
            c.clipRRect(rr, doAntiAlias=True)
        fp = self._fill_paint(L, g, box) if box is not None else None
        if fp is not None:
            r = float(g("radius", 0))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r), fp)

    def _draw_rect(self, c, L, t, T, g, box, reveal, mods):
        bx, by, bw, bh = box
        r = min(float(g("radius", 0)), bw / 2, bh / 2)
        dim = float(g("dimOutside", 0))
        if dim > 0:
            hole = skia.Path()
            hole.addRect(skia.Rect(-40000, -40000, 40000, 40000))
            hole.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r))
            hole.setFillType(skia.PathFillType.kEvenOdd)
            c.drawPath(hole, skia.Paint(AntiAlias=True, Color=_argb("#000000", dim * reveal)))
        path = skia.Path()
        path.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r))
        self._shape(c, L, g, path, box, reveal)

    def _draw_circle(self, c, L, t, T, g, box, reveal, mods):
        path = skia.Path()
        path.addOval(skia.Rect.MakeXYWH(*box), skia.PathDirection.kCW, 3)  # start at the top
        self._shape(c, L, g, path, box, reveal)

    def _draw_ellipse(self, c, L, t, T, g, box, reveal, mods):
        self._draw_circle(c, L, t, T, g, box, reveal, mods)

    def _path(self, L: Layer, t, T, g) -> skia.Path:
        ty = L.type
        static = all(L.props[k].static for k in ("x1", "y1", "x2", "y2", "bend", "r", "sides", "inner") if k in L.props)
        key = L.id
        if static and key in self._paths:
            return self._paths[key]
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
        if static:
            self._paths[key] = p
        return p

    def _draw_path(self, c, L, t, T, g, box, reveal, mods):
        path = self._path(L, t, T, g)
        if "anchor" in L.raw:
            b = path.computeTightBounds()
            c.translate(box[0] - b.left(), box[1] - b.top())
        self._shape(c, L, g, path, box, reveal)

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
        sp = self._stroke_paint(L, g)
        if amt <= 0.0005:
            return
        c.drawPath(self._trim(path, amt), sp)
        meas = skia.PathMeasure(path, False)
        pt = meas.getPosTan(meas.getLength() * min(1.0, amt))
        if pt:
            pos, tan = pt
            head = float(g("head", 22))
            ang = math.atan2(tan.y(), tan.x())
            hp = skia.Path()
            hp.moveTo(pos.x(), pos.y())
            for s in (2.6, -2.6):
                hp.moveTo(pos.x(), pos.y())
                hp.lineTo(pos.x() + head * math.cos(ang + s), pos.y() + head * math.sin(ang + s))
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
        r = float(g("radius", 0))
        c.save()
        if zoom != 1 or r or fit == "cover":
            c.clipRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r), doAntiAlias=True)
        if fit == "stretch" or L.type == "video":
            cw, ch = bw, bh
        else:
            k = (max if fit == "cover" else min)(bw / iw, bh / ih)
            cw, ch = iw * k, ih * k
        cx0 = bx + (bw - cw) / 2
        cy0 = by + (bh - ch) / 2
        px, py = bx + fx * bw, by + fy * bh
        dst = skia.Rect.MakeXYWH(px + (cx0 - px) * zoom, py + (cy0 - py) * zoom, cw * zoom, ch * zoom)
        c.drawImageRect(img, dst, CUBIC if L.type == "video" else SAMPLING, skia.Paint(AntiAlias=True))
        c.restore()
        sp = self._stroke_paint(L, g)
        if sp:
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r), sp)

    def _draw_video(self, c, L, t, T, g, box, reveal, mods):
        bx, by, bw, bh = box
        speed = float(L.raw.get("speed", 1))
        if L.video is None:
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
            L.video = media.VideoSource(L.src, round(bw * k), round(bh * k), self.s.fps, speed, L.raw.get("fit", "cover"),
                                        ctx=gpu_context() if self.backend == "gpu" else None)
        trim0 = float((L.raw.get("trim") or [0])[0])
        src_t = trim0 + t * speed
        if L.raw.get("loop"):
            span = L.video.info["duration"] - trim0
            if span > 0:
                src_t = trim0 + (t * speed) % span
        img = L.video.at(src_t)
        if img is not None:
            self._content(c, L, g, box, img, img.width(), img.height())

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
        return render3d.draw(self, c, L, g, reveal)

    def _model_surface(self):
        """The offscreen models are drawn in at their place in the frame, and its supersampling factor:
        4x MSAA on the GPU (no measurable cost), else 2x supersampling, which `check` and `--boxes` skip.
        It is the size of the frame from the start. One that grew to fit the largest model so far made a
        frame depend on the frames before it: on the GPU, edges moved by up to 49 levels."""
        if self._msurf is None:
            ctx = self.surface.recordingContext()
            ss = 1 if self.backend == "gpu" or self.track else 2
            info = skia.ImageInfo.Make(self.w * ss, self.h * ss, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
            surf = skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, info, 4) if ctx else None
            if surf is None and ctx:  # no MSAA here: supersample on the GPU instead
                ss, info = 2, info.makeWH(self.w * 2, self.h * 2)
                surf = skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, info)
            self._msurf = surf or skia.Surface.MakeRaster(info), ss
        return self._msurf

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
            f = self._fonts[key] = skia.Font(self._typeface(*key[:3]), size)
            f.setSubpixel(True)
            f.setEdging(skia.Font.Edging.kAntiAlias)
        return f

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
        meas = lambda s: font.measureText(s) + ls * len(s)  # noqa: E731
        space = meas(" ")
        lines: list[list] = []
        widths: list[float] = []
        wi = 0
        for para in txt.split("\n"):
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
                ww = meas(w_)
                if cur and mw and cw + space + ww > float(mw):
                    lines.append(cur)
                    widths.append(cw)
                    cur, cw = [], 0.0
                x = cw + (space if cur else 0.0)
                cur.append((w_, segs, x, wi))
                wi += 1
                cw = x + ww
            lines.append(cur)
            widths.append(cw)
        m = font.getMetrics()
        lh = size * lhm
        baseline0 = (lh - (m.fDescent - m.fAscent)) / 2 - m.fAscent
        W = max(widths) if widths else 0.0
        align = L.raw.get("align", "center")
        k = {"left": 0.0, "center": 0.5, "right": 1.0}.get(align, 0.5)
        runs = []
        ci = 0
        split = L.raw.get("split")
        for li, (ln, lw) in enumerate(zip(lines, widths)):
            off = (W - lw) * k
            for w_, segs, x, idx in ln:
                if split == "chars" or ls:
                    cx = x
                    for _, part, e_ in segs:
                        for ch in part:
                            runs.append(Run(ch, off + cx, li, e_, idx, ci))
                            cx += meas(ch)
                            ci += 1
                elif len(segs) == 1:
                    runs.append(Run(w_, off + x, li, segs[0][2], idx, ci))
                    ci += len(w_)
                else:  # one run per emphasis change, placed where it falls in the whole word
                    for s0, part, e_ in segs:
                        runs.append(Run(part, off + x + meas(w_[:s0]), li, e_, idx, ci + s0))
                    ci += len(w_)
                ci += 1  # the space
        lay = Layout(runs, widths, W, lh * len(lines), lh, baseline0, ci, wi)
        if len(self._layouts) > 2000:
            self._layouts.clear()
        self._layouts[key] = lay
        return lay

    def _draw_counter(self, c, L, t, T, g, box, reveal, mods):
        self._draw_text(c, L, t, T, g, box, reveal, mods)

    def _draw_text(self, c, L, t, T, g, box, reveal, mods, txt=None, layout=None, active=None, cue_t=None):
        txt = txt if txt is not None else self._text_of(L, t, T, g)
        lay = layout or self._text_layout(L, t, T, g, txt)
        size = float(g("size", 64))
        font = self._font(L, g, size)
        bx, by, bw, bh = box
        pad = self._pad(L)
        boxspec = L.raw.get("box")
        if boxspec:
            bs = boxspec if isinstance(boxspec, dict) else {}
            r = float(bs.get("radius", 10))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(*box), r, r),
                        skia.Paint(AntiAlias=True, Color=_argb(bs.get("fill", "#000000b0"))))
        ox, oy = bx + pad[1], by + pad[0]
        fill = g("fill") if "fill" in L.props else C.parse(self.s.theme.get("paper", "#f2f0e8"))
        emph = g("emph") if "emph" in L.props else C.parse(self.s.theme.get("accent", "#ff633f"))
        act = g("active") if "active" in L.props else emph
        fpaint = skia.Paint(AntiAlias=True)
        spaint = self._stroke_paint(L, g)
        split = L.raw.get("split")
        enter = L.enter if split else None
        stagger = (enter.stagger if enter and enter.stagger else 0.04) if split else 0
        typing = enter is None and L.enter and "type" in L.enter.names and t < L.enter.dur
        visible_chars = lay.nchars * reveal if (reveal < 1 or typing) else None
        track = mods.track
        unit_keys = [k for k in ("unitX", "unitY", "unitRotate", "unitScale", "unitOpacity") if k in L.props]
        units: dict[int, list[Run]] = {}
        for r in lay.runs:
            u = r.char0 if split == "chars" else r.word if split == "words" else r.line if split == "lines" else 0
            units.setdefault(u, []).append(r)
        order = sorted(units)
        for ui, u in enumerate(order):
            runs = units[u]
            um = IDENT
            if enter is not None:
                p = (t - ui * stagger) / enter.dur if enter.dur else 1.0
                if p < 1:
                    um = enter.mods(p)
            if um.alpha <= 0.002:
                continue
            ux = um.dx + (float(L.get("unitX", t, T, ui)) if "unitX" in unit_keys else 0)
            uy = um.dy + (float(L.get("unitY", t, T, ui)) if "unitY" in unit_keys else 0)
            ua = um.alpha * (float(L.get("unitOpacity", t, T, ui)) if "unitOpacity" in unit_keys else 1)
            urot = um.rotate + (float(L.get("unitRotate", t, T, ui)) if "unitRotate" in unit_keys else 0)
            usc = um.scale * (float(L.get("unitScale", t, T, ui)) if "unitScale" in unit_keys else 1)
            x0 = min(r.x for r in runs)
            x1 = max(r.x + font.measureText(r.text) for r in runs)
            ucx, ucy = ox + (x0 + x1) / 2, oy + runs[0].line * lay.lh + lay.lh / 2
            c.save()
            moved = ux or uy or urot or usc != 1 or um.rx or um.ry or um.dz
            if moved:
                c.translate(ucx + ux, ucy + uy)
                if um.rx or um.ry or um.dz:
                    c.concat(space.to_skia(space.tilt(um.dz, um.rx, um.ry)))
                if urot:
                    c.rotate(urot)
                if usc != 1:
                    c.scale(usc, usc)
                c.translate(-ucx, -ucy)
            if um.blur > 0.1:
                c.saveLayer(None, skia.Paint(ImageFilter=skia.ImageFilters.Blur(um.blur, um.blur)))
            for r in runs:
                text = r.text
                if visible_chars is not None:
                    if r.char0 >= visible_chars:
                        continue
                    text = text[: max(0, int(visible_chars - r.char0))]
                col = fill
                if active is not None:
                    col = act if r.word <= active else fill
                    if L.raw.get("mode") == "pop" and r.word > active:
                        continue
                elif r.emph:
                    col = emph
                fpaint.setColor(_argb(col, ua))
                y = oy + r.line * lay.lh + lay.baseline0
                x = ox + r.x + track * (r.char0 - lay.runs[0].char0) if track else ox + r.x
                if spaint is not None:
                    c.drawString(text, x, y, font, spaint)
                c.drawString(text, x, y, font, fpaint)
            if typing and L.raw.get("caret", True) and visible_chars is not None and runs and u == order[-1]:
                pass
            if um.blur > 0.1:
                c.restore()
            c.restore()
        if visible_chars is not None and L.raw.get("caret") and (int(T * 2) % 2 == 0 or reveal < 1):
            last = None
            for r in lay.runs:
                if r.char0 < visible_chars:
                    last = r
            if last is not None:
                vis = last.text[: max(0, int(visible_chars - last.char0))]
                cx = ox + last.x + font.measureText(vis) + 4
                cy = oy + last.line * lay.lh
                c.drawRect(skia.Rect.MakeXYWH(cx, cy + lay.lh * 0.15, max(3, size * 0.06), lay.lh * 0.7),
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
        a = min(1.0, ct / 0.12, (s1 - t) / 0.12)
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
                                "fill": list(g("fill")) if "fill" in L.props else None, "text": text, "scale": 1.0}
        if a < 1:
            c.restore()
        c.restore()
