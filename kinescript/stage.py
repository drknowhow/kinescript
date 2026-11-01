"""Groups with "sort": "pixel" (stages): each child draws on its own with an image of its depth, and at each pixel
the nearer child is on top. Skia's runtime shaders can't read the fragment position, and drawVertices gives a shader
one pair of texture coordinates, so there is no depth test inside one draw; between children there is.

Depth is 1/w (w from the camera's projection), which is linear across a triangle on screen, stored over QMAX so that
nearer is larger and 0 is where nothing is. The GPU keeps it in 24 bits of an RGBA8 image: its float targets are
F16, about 1.5 px of depth at z 0. The CPU keeps float32."""
from __future__ import annotations

import numpy as np
import skia

from . import render3d, space

QMAX = 1.0 / space.NEAR  # the largest 1/w a drawn point has
CHILDREN = 16  # check warns past this many children: each one is drawn and merged on its own
SUN = 1024  # the side of a stage's shadow map, in texels

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
# The depth at a frame pixel: the nearest of its samples, so its edges go with it.
_NEAREST = """max(max(dep(accd.eval(p + float2(-0.25, -0.25))), dep(accd.eval(p + float2(0.25, -0.25)))),
                  max(dep(accd.eval(p + float2(-0.25, 0.25))), dep(accd.eval(p + float2(0.25, 0.25)))))"""
# The merged stage at 1x, each pixel mixed toward the fog by how far away its depth puts it: dist = k / depth.
_FOG = _DEP + """
uniform shader acc;
uniform shader accd;
uniform half3 fogCol;
uniform float4 fog;  // near, far - near, amount, k
half4 main(float2 p) {
    half4 c = acc.eval(p);
    float d = NEAREST;
    if (d <= 0.0) {
        return c;
    }
    half f = half(clamp((fog.w / d - fog.x) / fog.y, 0.0, 1.0) * fog.z);
    return half4(c.rgb * (1.0 - f) + fogCol * (f * c.a), c.a);
}
""".replace("NEAREST", _NEAREST)
# The merged stage at 1x, darkened where the shadow map has something nearer the key light: each pixel's place in
# camera space from its depth, then across, up and toward the light, compared with the map where it falls and, if
# that is in shadow, at 9 taps round it for a soft edge. A surface turned from the light, which its depths and its
# neighbours' show, takes no shadow: it has no key light to lose, and tested against itself in the map an edge-on
# face shaded itself in stripes.
_SUN = _DEP + """
uniform shader acc;
uniform shader accd;
uniform shader sun;
uniform float4 cam;  // canvas px per frame px, half the canvas width and height, the camera's distance from z 0
uniform float3 su;
uniform float3 sv;
uniform float3 sl;
uniform float4 box;  // the map's corner across and up, its texels per px, the spacing of its taps in texels
uniform float4 dz;  // the map's least depth toward the light and its range, the bias, opacity
uniform float kq;  // w is kq over the packed depth
uniform float sub;  // frame px per depth pixel
float3 place(float2 p, float d) {
    float w = kq / d;
    return float3((p.x * cam.x - cam.y) * w, (p.y * cam.x - cam.z) * w, cam.w * (1.0 - w));
}
// From the point at p (depth d, place P) to the one along e, on the side that goes on most smoothly: the same
// surface rather than the gap to another. 0 when neither side has depth.
float3 along(float2 p, float2 e, float d, float3 P) {
    float a = dep(accd.eval(p + e));
    float b = dep(accd.eval(p - e));
    if (a > 0.0 && (b <= 0.0 || abs(a - d) <= abs(b - d))) {
        return place(p + e, a) - P;
    }
    if (b > 0.0) {
        return P - place(p - e, b);
    }
    return float3(0.0);
}
half4 main(float2 p) {
    half4 c = acc.eval(p);
    float d = NEAREST;
    float2 o = p - float2(0.5 - 0.5 * sub);  // the middle of the frame pixel's first depth pixel
    float e = dep(accd.eval(o));
    if (d <= 0.0 || e <= 0.0) {
        return c;
    }
    float3 Q = place(o, e);
    if (dot(cross(along(o, float2(sub, 0.0), e, Q), along(o, float2(0.0, sub), e, Q)), sl) <= 0.0) {
        return c;
    }
    float3 P = place(p, d);
    float2 t = float2(dot(P, su) - box.x, dot(P, sv) - box.y) * box.z;
    float toward = dot(P, sl) + dz.z;
    float m = dep(sun.eval(t)) / 16777215.0;
    if (m <= 0.0 || toward >= dz.x + (m - 0.001) / 0.998 * dz.y) {
        return c;
    }
    float n = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            m = dep(sun.eval(t + float2(float(i), float(j)) * box.w)) / 16777215.0;
            if (m > 0.0 && toward < dz.x + (m - 0.001) / 0.998 * dz.y) {
                n += 1.0;
            }
        }
    }
    return half4(c.rgb * half(1.0 - dz.w * n / 9.0), c.a);
}
""".replace("NEAREST", _NEAREST)
_SKSL = {
    "pack": _PACK,
    # the child's depth, kept only where it drew (drawn with kDstIn)
    "mask": _KID + "half4 main(float2 p) { return kid.eval(p).a > 0.0 ? half4(1.0) : half4(0.0); }",
    # its colour over what is behind it (kSrcOver), and under what is in front of it (kDstOver)
    "over": _KID + _NEAR.replace("RESULT", "return near ? kid.eval(p) : half4(0.0);"),
    "under": _KID + _NEAR.replace("RESULT", "return near ? half4(0.0) : kid.eval(p);"),
    "depth": _NEAR.replace("RESULT", "return near ? d : accd.eval(p);"),
    "fog": _FOG,
    "sun": _SUN,
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
        self.casters = None  # (camera-space corners, triangles) of the models that cast shadows, when they are on
        self.sun = self.sund = None  # the shadow map, made on first use; on the CPU, its pixels
        self.m = None  # its world-to-canvas matrix, set by Renderer._layer once it draws
        info = skia.ImageInfo.Make(self.w, self.h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        if self.gpu:
            ctx = self.ctx = r.surface.recordingContext()
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
    def draw(self, r, c: skia.Canvas, kids, T: float, alpha: float, shadows=None):
        """Draw kids, the children of a stage, onto c: each on its own, then merged by depth, with the shadows of its
        models from the key light if shadows is true or {"opacity", "soft"}."""
        self.casters = [] if shadows else None
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
        self._finish(c, render3d.fog(r, T), self._sun(r, T, shadows) if self.casters else None)

    def depth(self, xy: np.ndarray, q: np.ndarray, tris: np.ndarray):
        """Draw the depth of triangles tris, whose corners are at xy (stage px) with 1/w q, in order: each one
        replaces what is under it, so give them far to near."""
        if not len(tris):
            return
        self._values(self.kidd.getCanvas(), xy, q / QMAX, tris)
        if not self.gpu:
            at = xy[tris.ravel()]
            (x0, y0), (x1, y1) = np.floor(at.min(0)).astype(int), np.ceil(at.max(0)).astype(int)
            self.dirty = _union(self.dirty, _clip((y0 - 1, y1 + 1, x0 - 1, x1 + 1), self.h, self.w))

    def _values(self, cv: skia.Canvas, xy: np.ndarray, vals: np.ndarray, tris: np.ndarray):
        """Draw triangles tris, corners at xy, each point holding its value (0..1) as a depth image holds it."""
        uv = np.stack([vals, np.zeros(len(vals))], 1)
        for k in range(0, len(tris), 20000):  # Skia indexes vertices with 16 bits
            t = tris[k:k + 20000]
            keep = np.zeros(len(xy), bool)
            keep[t] = True
            sel = np.flatnonzero(keep)
            idx = (np.cumsum(keep) - 1)[t]
            v = skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, _points(xy[sel]), _points(uv[sel]),
                                       None, memoryview(idx.astype(np.uint16).ravel()))
            cv.drawVertices(v, self.paint, skia.BlendMode.kDst)

    # ---- shadows ------------------------------------------------------------------------------------------------
    def cast(self, P: np.ndarray, tris: np.ndarray):
        """Keep a model's triangles, corners P in camera space, to cast shadows if the stage has them."""
        if self.casters is not None and len(tris):
            self.casters.append((P, tris))

    def _sun(self, r, T: float, spec) -> dict | None:
        """Draw the shadow map: how near the key light each triangle the models cast is, seen along the light
        (orthographic) and fitted round them, nearer drawn over farther. Returns what _finish needs, or None."""
        key, amount = render3d.sun(r, T)
        if amount <= 0:
            return None
        sl = np.asarray(key, float) / np.linalg.norm(key)
        su = np.cross([0.0, 1.0, 0.0] if abs(sl[1]) < 0.9 else [1.0, 0.0, 0.0], sl)
        su /= np.linalg.norm(su)
        sv = np.cross(sl, su)
        P = np.concatenate([p for p, _ in self.casters])
        start = np.cumsum([0] + [len(p) for p, _ in self.casters[:-1]])
        tris = np.concatenate([t + o for (_, t), o in zip(self.casters, start)])
        # Only the faces turned from the light: a lit surface then lies well in front of what the map holds, and
        # does not shadow itself in stripes (a turned face is unlit by its shading anyway).
        e1, e2 = P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]]
        tris = tris[np.cross(e1, e2) @ sl <= 0]  # corners run so the cross product points out of the model
        if not len(tris):
            return None
        a, b, d = P @ su, P @ sv, P @ sl
        per = (SUN - 4) / max(float(np.ptp(a)), float(np.ptp(b)), 1.0)  # texels per px
        a0, b0, d0, dr = a.min() - 2 / per, b.min() - 2 / per, d.min(), max(float(np.ptp(d)), 1.0)
        if self.sun is None:
            info = skia.ImageInfo.Make(SUN, SUN, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
            if self.gpu:
                self.sun = skia.Surface.MakeRenderTarget(self.ctx, skia.Budgeted.kNo, info, 0)
            else:
                self.sund = np.zeros((SUN, SUN, 4), np.float32)
                self.sun = skia.Surface.MakeRasterDirect(info.makeColorType(skia.kRGBA_F32_ColorType), self.sund,
                                                         SUN * 16)
        cv = self.sun.getCanvas()
        cv.clear(skia.ColorTRANSPARENT)
        tris = tris[np.argsort(d[tris].mean(1), kind="stable")]  # farthest from the light first
        self._values(cv, np.stack([(a - a0) * per, (b - b0) * per], 1), 0.001 + 0.998 * (d - d0) / dr, tris)
        spec = spec if isinstance(spec, dict) else {}
        soft = max(0.0, float(spec.get("soft", 4.0)))
        return {"su": su, "sv": sv, "sl": sl, "a0": a0, "b0": b0, "per": per, "tap": soft * per / 2, "d0": d0,
                "dr": dr, "bias": 2 / per + 0.5 * soft + 1.0, "opacity": float(spec.get("opacity", 0.5)),
                "cam": [1 / r.scale, r.s.w / 2, r.s.h / 2, -1.0 / r._lens[3, 2]]}

    def _sun_gpu(self, colour: skia.Shader, depth: skia.Shader, sun: dict) -> skia.Shader:
        b = skia.RuntimeShaderBuilder(_effect("sun"))
        b.setChild("acc", colour)
        b.setChild("accd", depth)
        b.setChild("sun", self.sun.makeImageSnapshot().makeShader(skia.TileMode.kDecal, skia.TileMode.kDecal,
                                                                  skia.SamplingOptions()))
        b.setUniform("cam", sun["cam"])
        for k in ("su", "sv", "sl"):
            b.setUniform(k, sun[k].tolist())
        b.setUniform("box", [sun["a0"], sun["b0"], sun["per"], sun["tap"]])
        b.setUniform("dz", [sun["d0"], sun["dr"], sun["bias"], sun["opacity"]])
        b.setUniform("kq", 16777215.0 / QMAX)
        b.setUniform("sub", 1.0 / self.ss)
        return b.makeShader()

    def _sun_cpu(self, px: np.ndarray, depth: np.ndarray, y0: int, x0: int, sun: dict):
        """Darken premultiplied RGBA pixels px, whose top left is stage pixel (x0, y0), where the shadow map hides
        them from the key light, as the GPU's pass does: once a frame pixel, from the nearest of its depths, and only
        as far as each test needs. Over the whole stage at 2x, every test took 2 s a CPU frame at 1080p."""
        ss = self.ss
        h, w = depth.shape[0] // ss, depth.shape[1] // ss
        near = depth.reshape(h, ss, w, ss).max((1, 3)).ravel()
        hit = np.flatnonzero(near > 0)
        if not len(hit):
            return
        rows, cols = np.divmod(hit, w)
        cx, cy, focus = (np.float32(v) for v in sun["cam"][1:])
        k = np.float32(1 / self.k)

        def place(sy, sx, q):
            """Camera space of stage px (sx, sy) from the stage's top left, at depth q."""
            wv = np.float32(1 / QMAX) / np.where(q > 0, q, 1)
            P = np.empty((len(q), 3), np.float32)
            P[:, 0] = ((sx + np.float32(x0)) * k - cx) * wv
            P[:, 1] = ((sy + np.float32(y0)) * k - cy) * wv
            P[:, 2] = focus * (1 - wv)
            return P
        half = np.float32(ss / 2)
        P = place(rows * np.float32(ss) + half, cols * np.float32(ss) + half, near[hit])
        sl, per = sun["sl"].astype(np.float32), np.float32(sun["per"])
        ta = (P @ sun["su"].astype(np.float32) - np.float32(sun["a0"])) * per
        tb = (P @ sun["sv"].astype(np.float32) - np.float32(sun["b0"])) * per
        toward = P @ sl + np.float32(sun["bias"])
        lit, tap = self.sund[..., 0], np.float32(sun["tap"])

        def shadowed(sel, da, db):
            ia, ib = np.floor(ta[sel] + da).astype(np.int32), np.floor(tb[sel] + db).astype(np.int32)
            inside = (ia >= 0) & (ia < SUN) & (ib >= 0) & (ib < SUN)
            m = np.where(inside, lit[np.clip(ib, 0, SUN - 1), np.clip(ia, 0, SUN - 1)], 0)
            return (m > 0) & (toward[sel] < sun["d0"] + (m - 0.001) / 0.998 * sun["dr"])
        sel = np.flatnonzero(shadowed(slice(None), 0, 0))
        if not len(sel):
            return
        # Facing the light: from the depth of each pixel's first sample and its neighbours'.
        D = np.pad(depth, 1).ravel()  # no depth round the edge
        wide = depth.shape[1] + 2
        sy, sx = rows[sel] * ss, cols[sel] * ss
        at = (sy + 1) * wide + sx + 1
        e = D[at]
        Q = place(sy + np.float32(0.5), sx + np.float32(0.5), e)

        def along(dy, dx):
            a, b = D[at + dy * wide + dx], D[at - dy * wide - dx]
            fore = (a > 0) & ((b <= 0) | (np.abs(a - e) <= np.abs(b - e)))
            back = ~fore & (b > 0)
            ahead = place(sy + np.float32(0.5 + dy), sx + np.float32(0.5 + dx), a) - Q
            behind = Q - place(sy + np.float32(0.5 - dy), sx + np.float32(0.5 - dx), b)
            return np.where(fore[:, None], ahead, np.where(back[:, None], behind, 0))
        sel = sel[(e > 0) & (np.cross(along(0, 1), along(1, 0)) @ sl > 0)]
        if not len(sel):
            return
        n = sum(shadowed(sel, i * tap, j * tap).astype(np.float32) for i in (-1, 0, 1) for j in (-1, 0, 1))
        blocks = px.reshape(h, ss, w, ss, 4)
        r, c = rows[sel], cols[sel]
        dark = (1 - sun["opacity"] * n / 9)[:, None, None, None]
        blocks[r, :, c, :, :3] = np.rint(blocks[r, :, c, :, :3] * dark).astype(np.uint8)

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

    def _finish(self, c: skia.Canvas, fg=None, sun=None):
        """Lay the merged children on c at the frame's resolution, darkened where shadows fall (sun, from _sun) and
        through fog fg, when there are any."""
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
            if sun is not None:
                self._sun_cpu(px, self.outd[y0:y1, x0:x1], y0, x0, sun)
            if fg is not None:
                self._fog_cpu(px, self.outd[y0:y1, x0:x1], fg)
            img = skia.Image.fromarray(px, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
            src = skia.Rect(0, 0, x1 - x0, y1 - y0)
        c.save()
        c.resetMatrix()
        dst = skia.Rect(x0 / ss, y0 / ss, x1 / ss, y1 / ss)
        if self.gpu and (fg is not None or sun is not None):
            to = skia.Matrix.Scale(1 / ss, 1 / ss)
            colour = img.makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, sampling, to)
            depth = self.accd.makeImageSnapshot().makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp,
                                                             skia.SamplingOptions(), to)
            if sun is not None:
                colour = self._sun_gpu(colour, depth, sun)
            if fg is not None:
                b = skia.RuntimeShaderBuilder(_effect("fog"))
                b.setChild("acc", colour)
                b.setChild("accd", depth)
                col, near, far, amt, focus = fg
                b.setUniform("fogCol", col)
                b.setUniform("fog", [near, max(far - near, 1e-6), amt, focus * 16777215.0 / QMAX])
                colour = b.makeShader()
            c.drawRect(dst, skia.Paint(Shader=colour))
        else:
            c.drawImageRect(img, src, dst, sampling)
        c.restore()

    @staticmethod
    def _fog_cpu(px: np.ndarray, depth: np.ndarray, fg: tuple):
        """Mix premultiplied RGBA pixels px toward the fog by their depths (q / QMAX, 0 where nothing is). In float32
        and only where something is: over a whole stage at 1080p, float64 took 455 ms a frame and this 95."""
        hit = np.flatnonzero(depth > 0)
        if not len(hit):
            return
        col, near, far, amt, focus = fg
        f = np.clip((np.float32(focus / QMAX) / depth.ravel()[hit] - near) / max(far - near, 1e-6), 0, 1) * amt
        some = f > 0.002
        if not some.any():
            return
        idx, f = hit[some], f[some][:, None]
        flat = px.reshape(-1, 4)
        p = flat[idx]
        a = p[:, 3:] * np.float32(1 / 255)
        flat[idx, :3] = np.rint(p[:, :3] * (1 - f) + np.asarray(col, np.float32) * 255 * f * a).astype(np.uint8)


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
