"""Groups with "sort": "pixel" (stages): each child draws on its own with an image of its depth, and at each pixel
the nearer child is on top. Skia's runtime shaders can't read the fragment position, and drawVertices gives a shader
one pair of texture coordinates, so there is no depth test inside one draw; between children there is.

Depth is 1/w (w from the camera's projection), which is linear across a triangle on screen, stored over QMAX so that
nearer is larger and 0 is where nothing is. The GPU keeps it in 24 bits of an RGBA8 image: its float targets are
F16, about 1.5 px of depth at z 0. The CPU keeps float32."""
from __future__ import annotations

import numpy as np
import skia

from . import space

QMAX = 1.0 / space.NEAR  # the largest 1/w a drawn point has
CHILDREN = 16  # check warns past this many children: each one is drawn and merged on its own

_PACK = """
half4 main(float2 uv) {
    float d = floor(clamp(uv.x, 0.0, 1.0) * 16777215.0 + 0.5);
    float hi = floor(d / 65536.0);
    float mid = floor((d - hi * 65536.0) / 256.0);
    return half4(hi / 255.0, mid / 255.0, (d - hi * 65536.0 - mid * 256.0) / 255.0, 1.0);
}
"""
_DEP = """
float dep(half4 v) {
    return floor(v.r * 255.0 + 0.5) * 65536.0 + floor(v.g * 255.0 + 0.5) * 256.0 + floor(v.b * 255.0 + 0.5);
}
"""
_NEAR = _DEP + """
uniform shader kidd;
uniform shader accd;
half4 main(float2 p) {
    half4 d = kidd.eval(p);
    bool near = d.a > 0.0 && dep(d) >= dep(accd.eval(p));
    RESULT
}
"""
_KID = "uniform shader kid;"
_SKSL = {
    "pack": _PACK,
    # the child's depth, kept only where it drew (drawn with kDstIn)
    "mask": _KID + "half4 main(float2 p) { return kid.eval(p).a > 0.0 ? half4(1.0) : half4(0.0); }",
    # its colour over what is behind it (kSrcOver), and under what is in front of it (kDstOver)
    "over": _KID + _NEAR.replace("RESULT", "return near ? kid.eval(p) : half4(0.0);"),
    "under": _KID + _NEAR.replace("RESULT", "return near ? half4(0.0) : kid.eval(p);"),
    "depth": _NEAR.replace("RESULT", "return near ? d : accd.eval(p);"),
}
_EFFECTS: dict[str, skia.RuntimeEffect] = {}


def _effect(name: str) -> skia.RuntimeEffect:
    if name not in _EFFECTS:
        _EFFECTS[name] = skia.RuntimeEffect.MakeForShader(_SKSL[name])
    return _EFFECTS[name]


def _points(xy: np.ndarray) -> list:
    return list(map(skia.Point.Make, xy[:, 0].tolist(), xy[:, 1].tolist()))


class Stage:
    """The surfaces one renderer's stages draw into, frame-sized at the models' supersampling from the start, so a
    frame does not depend on the frames drawn before it."""

    def __init__(self, r):
        self.ss = r._model_surface()[1]
        self.w, self.h = r.w * self.ss, r.h * self.ss
        self.k = r.scale * self.ss  # canvas px to stage px
        self.gpu = r.backend == "gpu"
        self.child = None  # the child being drawn
        self.m = None  # its world-to-canvas matrix, set by Renderer._layer once it draws
        info = skia.ImageInfo.Make(self.w, self.h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        if self.gpu:
            ctx = r.surface.recordingContext()
            self.kid, self.kidd, self.acc, self.accd = (
                skia.Surface.MakeRenderTarget(ctx, skia.Budgeted.kNo, info, 0) for _ in range(4))
            self.paint = skia.Paint(Shader=_effect("pack").makeShader(None), BlendMode=skia.BlendMode.kSrc)
        else:
            self.scr = np.zeros((self.h, self.w), np.uint32)  # RGBA8888 pixels, alpha in the top byte
            self.dep = np.zeros((self.h, self.w, 4), np.float32)  # depth in channel 0
            self.out = np.zeros((self.h, self.w), np.uint32)
            self.outd = np.zeros((self.h, self.w), np.float32)
            self.kid = skia.Surface.MakeRasterDirect(info, self.scr, self.w * 4)
            self.kidd = skia.Surface.MakeRasterDirect(info.makeColorType(skia.kRGBA_F32_ColorType), self.dep,
                                                      self.w * 16)
            grad = skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(1, 0)],
                                                  [skia.Color4f(0, 0, 0, 1), skia.Color4f(1, 1, 1, 1)])
            self.paint = skia.Paint(Shader=grad, BlendMode=skia.BlendMode.kSrc)
            self.box = None  # (y0, y1, x0, x1) of out that holds pixels
            self.dirty = None  # of dep, written since the last child

    # ---- drawing a stage ----------------------------------------------------------------------------------------
    def draw(self, r, c: skia.Canvas, kids, T: float, alpha: float):
        """Draw kids, the children of a stage, onto c: each on its own, then merged by depth."""
        if self.gpu:
            self.acc.getCanvas().clear(skia.ColorTRANSPARENT)
            self.accd.getCanvas().clear(skia.ColorTRANSPARENT)
        else:
            self._zero(self.box, self.out, self.outd)
            self.box = None
        here = c.getLocalToDevice()
        r._stage, r._dev = self, self.ss
        try:
            for child in kids:
                if r._stopped:
                    break
                self.child, self.m = child, None
                cv = self.kid.getCanvas()  # on the GPU, a merge trades the child's colour surface for another
                cv.save()
                cv.resetMatrix()
                cv.scale(self.ss, self.ss)
                cv.concat(here)
                r._layer(cv, child, T, alpha)
                cv.restore()
                box = None if self.gpu else self._extent()
                if self.m is not None and (self.gpu or box is not None):
                    self._merge(child, box)
                self._clear(box)
        finally:
            r._stage, r._dev, self.child = None, 1.0, None
        self._finish(c)

    def depth(self, xy: np.ndarray, q: np.ndarray, tris: np.ndarray):
        """Draw the depth of triangles tris, whose corners are at xy (stage px) with 1/w q, in order: each one
        replaces what is under it, so give them far to near."""
        if not len(tris):
            return
        uv = np.stack([q / QMAX, np.zeros(len(q))], 1)
        cv = self.kidd.getCanvas()
        for k in range(0, len(tris), 20000):  # Skia indexes vertices with 16 bits
            t = tris[k:k + 20000]
            keep = np.zeros(len(xy), bool)
            keep[t] = True
            sel = np.flatnonzero(keep)
            idx = (np.cumsum(keep) - 1)[t]
            v = skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, _points(xy[sel]), _points(uv[sel]),
                                       None, memoryview(idx.astype(np.uint16).ravel()))
            cv.drawVertices(v, self.paint, skia.BlendMode.kDst)
        if not self.gpu:
            at = xy[tris.ravel()]
            (x0, y0), (x1, y1) = np.floor(at.min(0)).astype(int), np.ceil(at.max(0)).astype(int)
            self.dirty = _union(self.dirty, _clip((y0 - 1, y1 + 1, x0 - 1, x1 + 1), self.h, self.w))

    def plane(self, m: np.ndarray, corners: np.ndarray, bias: float = 1.0):
        """Draw the depth of the plane z 0 of world-to-canvas matrix m over the quad `corners` (4 x 2 stage px:
        top left, top right, bottom left, bottom right). 1/w is linear on screen across a plane, so the corners
        give it exactly. bias > 1 puts it a little nearer (a shadow on a floor that is also a child)."""
        A = m[np.ix_([0, 1, 3], [0, 1, 3])]
        try:
            row = np.linalg.inv(A)[2]
        except np.linalg.LinAlgError:
            return  # seen edge on: no area to cover
        q = (corners / self.k) @ row[:2] + row[2]
        self.depth(corners, q * bias, np.array([[0, 1, 2], [1, 3, 2]]))

    # ---- merging ------------------------------------------------------------------------------------------------
    def _extent(self):
        """(y0, y1, x0, x1) of the pixels the child drew on the CPU, or None."""
        rows = np.flatnonzero(self.scr.any(1))
        if not len(rows):
            return None
        cols = np.flatnonzero(self.scr[rows[0]:rows[-1] + 1].any(0))
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1

    def _merge(self, child, box):
        """Merge the child just drawn into the stage; box is where it drew on the CPU (the whole stage on the GPU).
        A model drew its depth as it drew; anything else lies in its own plane."""
        if child.type != "model":
            y0, y1, x0, x1 = (0, self.h, 0, self.w) if box is None else box
            self.plane(self.m, np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float))
        if self.gpu:
            self._merge_gpu()
        else:
            self._merge_cpu(box)

    def _merge_gpu(self):
        """No pass draws into a surface it reads, so Skia never copies one to keep an image of it: the child's depth
        is cut to where it drew, its colour goes over what is behind it and under what is in front, and the nearer
        depth goes into the child's colour surface, which then trades places with the stage's depth."""
        B = skia.BlendMode
        self._pass("mask", self.kidd, B.kDstIn, kid=self.kid)
        self._pass("over", self.acc, B.kSrcOver, kid=self.kid, kidd=self.kidd, accd=self.accd)
        self._pass("under", self.acc, B.kDstOver, kid=self.kid, kidd=self.kidd, accd=self.accd)
        self._pass("depth", self.kid, B.kSrc, kidd=self.kidd, accd=self.accd)
        self.kid, self.accd = self.accd, self.kid

    def _pass(self, name: str, dst: skia.Surface, mode: skia.BlendMode, **reads: skia.Surface):
        b = skia.RuntimeShaderBuilder(_effect(name))
        for key, s in reads.items():
            b.setChild(key, s.makeImageSnapshot().makeShader(skia.SamplingOptions()))
        dst.getCanvas().drawRect(skia.Rect(0, 0, self.w, self.h), skia.Paint(Shader=b.makeShader(), BlendMode=mode))

    def _merge_cpu(self, box):
        y0, y1, x0, x1 = box
        A, B = self.out[y0:y1, x0:x1], self.scr[y0:y1, x0:x1]
        DA, DB = self.outd[y0:y1, x0:x1], self.dep[y0:y1, x0:x1, 0]
        aa, ba = A >> 24, B >> 24
        near = (ba > 0) & (DB > 0) & (DB >= DA)
        top = np.where(near, B, A)
        # Where the one on top lets some of the other through, mix them as the GPU's source-over does.
        ys, xs = np.nonzero((aa > 0) & (ba > 0) & (np.where(near, ba, aa) < 255))
        if len(ys):
            n = near[ys, xs]
            t = _rgba(np.where(n, B[ys, xs], A[ys, xs]))
            u = _rgba(np.where(n, A[ys, xs], B[ys, xs]))
            mix = t + (u * (255 - t[:, 3:]) + 127) // 255
            top[ys, xs] = (mix[:, 0] | mix[:, 1] << 8 | mix[:, 2] << 16 | mix[:, 3] << 24).astype(np.uint32)
        A[...] = top
        np.copyto(DA, DB, where=near)
        self.box = _union(self.box, box)

    def _clear(self, box):
        if self.gpu:
            self.kid.getCanvas().clear(skia.ColorTRANSPARENT)
            self.kidd.getCanvas().clear(skia.ColorTRANSPARENT)
            return
        self._zero(box, self.scr)
        self._zero(self.dirty, self.dep)
        self.dirty = None

    @staticmethod
    def _zero(box, *arrays):
        if box is not None:
            y0, y1, x0, x1 = box
            for a in arrays:
                a[y0:y1, x0:x1] = 0

    def _finish(self, c: skia.Canvas):
        """Lay the merged children on c at the frame's resolution."""
        sampling = skia.SamplingOptions(skia.FilterMode.kLinear) if self.ss > 1 else skia.SamplingOptions()
        ss = self.ss
        if self.gpu:
            img, src, x0, y0 = self.acc.makeImageSnapshot(), skia.Rect(0, 0, self.w, self.h), 0, 0
            x1, y1 = self.w, self.h
        else:
            if self.box is None:
                return
            y0, y1, x0, x1 = self.box
            # Whole frame pixels, so the downsample lands each one where the GPU's does.
            y0, x0 = y0 - y0 % ss, x0 - x0 % ss
            y1, x1 = min(self.h, y1 + -y1 % ss), min(self.w, x1 + -x1 % ss)
            px = np.ascontiguousarray(self.out[y0:y1, x0:x1]).view(np.uint8).reshape(y1 - y0, x1 - x0, 4)
            img = skia.Image.fromarray(px, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
            src = skia.Rect(0, 0, x1 - x0, y1 - y0)
        c.save()
        c.resetMatrix()
        c.drawImageRect(img, src, skia.Rect(x0 / ss, y0 / ss, x1 / ss, y1 / ss), sampling)
        c.restore()


def _rgba(px: np.ndarray) -> np.ndarray:
    return np.stack([px & 255, px >> 8 & 255, px >> 16 & 255, px >> 24], 1).astype(np.uint32)


def _clip(box, h: int, w: int):
    y0, y1, x0, x1 = box
    y0, y1, x0, x1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    return (y0, y1, x0, x1) if y0 < y1 and x0 < x1 else None


def _union(a, b):
    if a is None or b is None:
        return a or b
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def stage(r) -> Stage:
    """The renderer's stage surfaces, made on first use."""
    if r._stages is None:
        r._stages = Stage(r)
    return r._stages
