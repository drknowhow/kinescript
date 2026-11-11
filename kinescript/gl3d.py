"""Model layers on OpenGL, through moderngl on the GL context Skia's GPU backend already made.

render3d draws a model with numpy and one Skia drawVertices call: it sorts the triangles far to near, because Skia has
no depth test, and hands every vertex to Skia as a Python object. Here a model's mesh goes to the GPU once, in buffers
kept while the mesh stays the same, and each frame is two draws:
- the triangles, through a depth buffer, so parts that cross meet cleanly, shaded per pixel by the same formula as
  render3d's SkSL; this pass also leaves each pixel's camera-space normal and its 1/w;
- a pass over the model's pixels that draws its edge lines from what the first pass left (where the depth steps,
  where the normal turns past the crease angle, where the model ends), and packs the depth a stage merges by.
Skia then draws the result where render3d would have drawn its own image, so the rest of a frame is unchanged.

A model draws here when the backend is the GPU, moderngl is installed (pip install "kinescript[gl]"), and it uses
nothing this path does not have: textures (including a model file's), transparency, explode and edges: "all" draw on
the Skia path, as do `kine check` (it measures the frame on that path) and --wire. KINE_3D=skia turns this path off;
KINE_3D=gl makes it an error when it cannot run.

The two paths differ where they are built differently: edge lines are found per pixel here and along the mesh's edges
there, and triangles that cross the camera plane are clipped here where render3d drops them."""
from __future__ import annotations

import math
import os
import weakref

import numpy as np
import skia

from . import space

MATS = 64  # materials one draw takes: schema.COPY_SLOTS, the most a model with copies has
STEP = 6.0  # canvas px a depth must step between pixels a line's width apart for a line to be drawn there
SETTINGS = ("auto", "skia", "gl")
_GL_TEXTURE_2D, _GL_RGBA8 = 0x0DE1, 0x8058

_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in float in_mat;
uniform mat4 u_m;   // model space to canvas px (x and y over w), w from the camera's lens
uniform mat3 u_n;   // model normals into camera space
uniform vec2 u_px;  // canvas px to clip space: 2 * scale * supersampling over the target's width and height
uniform vec4 u_fog; // near, far - near, amount, the distance of z 0 from the camera; amount 0: none
noperspective out vec2 v_n2;  // across the screen, as Skia interpolates render3d's normal
noperspective out float v_fog;
out vec3 v_n;
flat out int v_mat;
void main() {
    vec4 h = u_m * vec4(in_pos, 1.0);
    gl_Position = vec4(h.x * u_px.x - h.w, h.w - h.y * u_px.y, h.w - 2.0 * NEAR, h.w);
    v_n = u_n * in_nrm;
    v_n2 = normalize(v_n).xy;
    v_mat = int(in_mat + 0.5);
    v_fog = u_fog.z > 0.0 ? floor(clamp((h.w * u_fog.w - u_fog.x) / u_fog.y, 0.0, 1.0) * u_fog.z * 255.0 + 0.5)
                            / 255.0 : 0.0;
}
"""
# render3d._SKSL with _MORE and _ENV_MAIN, for every material at once.
_FRAG = """
#version 330
noperspective in vec2 v_n2;
noperspective in float v_fog;
in vec3 v_n;
flat in int v_mat;
uniform vec4 u_base[MATS];  // fill, bands
uniform vec4 u_surf[MATS];  // specular amount, exponent, metal, rim
uniform vec4 u_glow[MATS];  // emission, glow
uniform int u_flags;        // 1: a material glows, 2: one gives off light, 4: an environment
uniform vec3 keyDir;
uniform vec3 fillDir;
uniform vec3 keyCol;
uniform vec4 amt;
uniform int u_more;
uniform vec3 u_dir[3];
uniform vec3 u_col[3];
uniform vec3 envTop;
uniform vec3 envHor;
uniform vec3 envBot;
uniform vec3 envDown;
uniform float envAmt;
uniform vec3 u_fogCol;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_geo;
void main() {
    vec4 b = u_base[v_mat];
    vec4 p = u_surf[v_mat];
    vec2 n2 = v_n2;
    vec3 n = normalize(vec3(n2, sqrt(max(0.0, 1.0 - dot(n2, n2)))));
    float d = max(dot(n, keyDir), 0.0);
    float f = max(dot(n, fillDir), 0.0);
    float s = pow(max(dot(n, normalize(keyDir + vec3(0.0, 0.0, 1.0))), 0.0), p.y) * p.x;
    if (b.a > 1.5) {
        d = ceil(d * b.a - 0.25) / b.a;
        f = ceil(f * b.a - 0.25) / b.a;
        s = step(0.45, s) * p.x;
    }
    vec3 diffuse = b.rgb * (amt.z + amt.x * d * keyCol + amt.y * f * vec3(0.85, 0.9, 1.0)) * (1.0 - 0.75 * p.z);
    vec3 spec = mix(vec3(1.0), b.rgb, p.z) * s * amt.x * keyCol;
    for (int j = 0; j < u_more; j++) {
        float dj = max(dot(n, u_dir[j]), 0.0);
        float sj = pow(max(dot(n, normalize(u_dir[j] + vec3(0.0, 0.0, 1.0))), 0.0), p.y) * p.x;
        if (b.a > 1.5) {
            dj = ceil(dj * b.a - 0.25) / b.a;
            sj = step(0.45, sj) * p.x;
        }
        diffuse += b.rgb * dj * u_col[j] * (1.0 - 0.75 * p.z);
        spec += mix(vec3(1.0), b.rgb, p.z) * sj * u_col[j];
    }
    float rim = pow(1.0 - n.z, 3.0) * p.w * amt.w;
    vec3 lit = diffuse + spec + rim * keyCol;
    if ((u_flags & 4) != 0) {
        vec3 e = vec3(2.0 * n.z * n.x, 2.0 * n.z * n.y, 2.0 * n.z * n.z - 1.0);
        float up = -dot(e, envDown);
        vec3 sky = up > 0.0 ? mix(envHor, envTop, sqrt(up)) : mix(envHor, envBot, sqrt(-up));
        float sh = clamp((p.x - 0.4 * p.z) / 0.6, 0.0, 1.0);
        float fr = 0.04 + 0.96 * pow(1.0 - n.z, 5.0);
        lit += mix((envTop + 2.0 * envHor + envBot) * 0.25, sky, sh) * (p.z * b.rgb + (1.0 - p.z) * sh * fr) * envAmt;
    }
    if ((u_flags & 1) != 0) {
        lit = max(lit, b.rgb * u_glow[v_mat].a);
    }
    if ((u_flags & 2) != 0) {
        lit += u_glow[v_mat].rgb;
    }
    o_col = vec4(mix(clamp(lit, 0.0, 1.0), u_fogCol, v_fog), 1.0);
    o_geo = vec4(normalize(v_n), NEAR * gl_FragCoord.w);
}
"""
_QUAD = """
#version 330
in vec2 in_xy;
void main() { gl_Position = vec4(in_xy, 0.0, 1.0); }
"""
# Each pixel against taps half a line's width away in 16 directions (and a quarter in 8): a line where the model
# ends, where the depth steps (1/w is linear across a flat face, so its second difference is 0 there) or where the
# normal turns past the crease angle. A line pixel takes the nearest depth round it, so a line over a part behind is
# merged in front of that part.
_EDGE = """
#version 330
uniform sampler2D t_col;
uniform sampler2D t_geo;
uniform ivec4 u_box;  // the texels the first pass cleared: x0, y0, x1, y1 (ends excluded), from the bottom left
uniform float u_r;    // half the line width in texels; 0: no lines
uniform float u_cos;  // the cosine of the least crease angle drawn
uniform float u_gap;  // a line where 1/w's second difference passes this times its square
uniform vec3 u_ink;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_dep;
vec4 geo(ivec2 p) {
    if (any(lessThan(p, u_box.xy)) || any(greaterThanEqual(p, u_box.zw))) {
        return vec4(0.0);
    }
    return texelFetch(t_geo, p, 0);
}
vec4 pack(float q) {  // stage.py's _PACK
    float d = floor(clamp(q, 0.0, 1.0) * 16777215.0 + 0.5);
    float hi = floor(d / 65536.0);
    float mid = floor((d - hi * 65536.0) / 256.0);
    return vec4(hi / 255.0, mid / 255.0, (d - hi * 65536.0 - mid * 256.0) / 255.0, 1.0);
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 g0 = geo(p);
    float q = g0.w;
    bool line = false;
    float near = q;
    if (u_r > 0.0) {
        for (int k = 0; k < 12; k++) {
            float a = k < 8 ? 0.39269908 * float(k) : 0.78539816 * float(k - 8);
            ivec2 o = ivec2(round(vec2(cos(a), sin(a)) * (k < 8 ? u_r : 0.5 * u_r)));
            if (o == ivec2(0)) {
                continue;
            }
            vec4 ga = geo(p + o);
            vec4 gb = geo(p - o);
            near = max(near, max(ga.w, gb.w));
            if (q <= 0.0) {
                line = line || ga.w > 0.0 || gb.w > 0.0;
            } else if (ga.w <= 0.0 || gb.w <= 0.0 || abs(ga.w + gb.w - 2.0 * q) > u_gap * q * q
                       || dot(ga.xyz, g0.xyz) < u_cos || dot(gb.xyz, g0.xyz) < u_cos) {
                line = true;
            }
        }
    }
    if (line) {
        o_col = vec4(u_ink, 1.0);
        o_dep = pack(near);
    } else {
        o_col = texelFetch(t_col, p, 0);
        o_dep = q > 0.0 ? pack(q) : vec4(0.0);
    }
}
"""

_STATE: dict = {}  # the moderngl context and its programs, made once per process: None when it cannot be
_TARGETS: dict = {}  # render targets by size, the last two kept


def setting() -> str:
    """KINE_3D: auto (the default: OpenGL for the models it can draw, when it can run), skia or gl (as auto, but an
    error when OpenGL cannot run)."""
    raw = os.environ.get("KINE_3D", "")
    v = raw.strip().lower() or "auto"
    if v not in SETTINGS:
        raise ValueError(f"KINE_3D={raw!r} is not a setting: use auto (the default: OpenGL for the model layers it "
                         "can draw, on the GPU backend), skia (never OpenGL) or gl (as auto, but an error when OpenGL "
                         "cannot run)")
    return v


def _state():
    """The moderngl context on the current GL context, with this module's programs, or None without moderngl."""
    if "ctx" not in _STATE:
        try:
            import moderngl
        except ImportError:
            _STATE["ctx"] = None
            return None
        try:  # an OpenGL older than 3.3, or a driver that will not compile these, leaves models on Skia
            ctx = moderngl.create_context(require=330)
            near = f"{space.NEAR:.6f}"
            shade = ctx.program(vertex_shader=_VERT.replace("NEAR", near),
                                fragment_shader=_FRAG.replace("MATS", str(MATS)).replace("NEAR", near))
            edge = ctx.program(vertex_shader=_QUAD, fragment_shader=_EDGE)
        except Exception as e:  # noqa: BLE001 - moderngl raises its own Error and plain exceptions alike
            _STATE.update(ctx=None, error=f"{type(e).__name__}: {e}")
            return None
        edge["t_col"].value, edge["t_geo"].value = 0, 1
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4"))
        # Skia binds sampler objects of its own; these replace them on the two units read, so a mipmap filter left
        # on a unit cannot make a texture incomplete (texelFetch reads it as 0 then).
        nearest = ctx.sampler(filter=(moderngl.NEAREST, moderngl.NEAREST), repeat_x=False, repeat_y=False)
        _STATE.update(ctx=ctx, mgl=moderngl, shade=shade, edge=edge, sampler=nearest,
                      quad=ctx.vertex_array(edge, [(quad, "2f", "in_xy")]))
    return _STATE["ctx"]


def ready() -> bool:
    """Whether models can draw on OpenGL in this process: KINE_3D allows it and moderngl runs on the GL context
    gpu_context made (call that first)."""
    return setting() != "skia" and _state() is not None


def available(r) -> bool:
    """Whether this renderer draws its models on OpenGL when they allow it."""
    want = setting()
    if want == "skia" or r.wire:
        return False
    ok = r.backend == "gpu" and r.gr is not None and _state() is not None
    if want == "gl" and not ok:
        why = f" ({_STATE['error']})" if _STATE.get("error") else ""
        raise RuntimeError(f"KINE_3D=gl, but OpenGL models need the GPU backend and moderngl "
                           f"(pip install \"kinescript[gl]\"){why}; set KINE_3D to auto or skia to draw them with Skia")
    return ok


def fits(mats, texs, glows, explode: float, edging) -> bool:
    """Whether a model with these materials, textures, emission textures, explode and edges draws here."""
    return (len(mats) <= MATS and explode <= 0 and all(tx is None for tx in texs) and all(g is None for g in glows)
            and all(mt[8] >= 1 for mt in mats) and (edging is None or edging[2] <= 1.0))


def _mat(a: np.ndarray) -> bytes:
    """Matrix a as GLSL reads one: by columns."""
    return np.ascontiguousarray(np.asarray(a, "f4").T).tobytes()


def _targets(w: int, h: int):
    """Frame-sized render targets: the first pass's colour, normal and 1/w (float) and depth buffer, and the second
    pass's colour and packed depth, made once for each size."""
    hit = _TARGETS.pop((w, h), None)
    if hit is None:
        ctx = _STATE["ctx"]
        col, geo, out, outd = (ctx.texture((w, h), 4), ctx.texture((w, h), 4, dtype="f4"), ctx.texture((w, h), 4),
                               ctx.texture((w, h), 4))
        depth = ctx.depth_renderbuffer((w, h))
        hit = {"col": col, "geo": geo, "out": out, "outd": outd, "depth": depth,
               "first": ctx.framebuffer([col, geo], depth), "second": ctx.framebuffer([out, outd])}
        while len(_TARGETS) >= 2:
            for v in _TARGETS.pop(next(iter(_TARGETS))).values():
                v.release()
    _TARGETS[(w, h)] = hit
    return hit


class _Buffers:
    """One mesh's vertices and triangles on the GPU, and its triangles in build order once a draw needs them."""

    def __init__(self, m):
        ctx, shade = _STATE["ctx"], _STATE["shade"]
        self.key = (m.P, m.N, m.M, m.T)
        v = np.concatenate([m.P, m.N, m.M[:, None]], 1).astype("f4")
        self.vbo = ctx.buffer(v.tobytes())
        self.ibo = ctx.buffer(np.ascontiguousarray(m.T, "i4").tobytes())
        self.vao = ctx.vertex_array(shade, [(self.vbo, "3f 3f 1f", "in_pos", "in_nrm", "in_mat")], self.ibo)
        self.built = None  # (order, vao, ibo): the triangles in that build order

    def holds(self, m) -> bool:
        return all(a is b for a, b in zip(self.key, (m.P, m.N, m.M, m.T)))

    def moved(self, m) -> bool:
        """Take mesh m's vertices if it has these triangles (the same array: moving copies keep theirs)."""
        if m.T is not self.key[3] or len(m.P) * 28 != self.vbo.size:
            return False
        self.vbo.write(np.concatenate([m.P, m.N, m.M[:, None]], 1).astype("f4").tobytes())
        self.key = (m.P, m.N, m.M, m.T)
        return True

    def in_order(self, m, order: np.ndarray):
        """A vertex array of m's triangles in build order `order` (render3d keeps one while the mesh holds)."""
        if self.built is None or self.built[0] is not order:
            if self.built is not None:
                for o in self.built[1:]:
                    o.release()
            ctx = _STATE["ctx"]
            ibo = ctx.buffer(np.ascontiguousarray(m.T[order], "i4").tobytes())
            self.built = (order, ctx.vertex_array(_STATE["shade"], [(self.vbo, "3f 3f 1f", "in_pos", "in_nrm",
                                                                      "in_mat")], ibo), ibo)
        return self.built[1]

    def release(self):
        for o in (self.vao, self.ibo, self.vbo, *(self.built or ())[1:]):
            o.release()


def _release(kept: dict):
    for b in kept.values():
        b.release()
    kept.clear()


def _buffers(r, L, m) -> _Buffers:
    kept = r._gl_meshes
    if kept is None:
        kept = r._gl_meshes = {}
        weakref.finalize(r, _release, kept)
    hit = kept.get(L.id)
    if hit is None or not (hit.holds(m) or hit.moved(m)):
        if hit is not None:
            hit.release()
        hit = kept[L.id] = _Buffers(m)
    return hit


def _set(prog, name: str, value):
    """Set a uniform the compiler kept (it drops those a program never reads)."""
    u = prog.get(name, None)
    if u is not None:
        if isinstance(value, bytes):
            u.write(value)
        else:
            u.value = value


def _uniforms(r, mats, lights, fg, view: np.ndarray, ss: int, W: int, H: int):
    lt, more, env = lights
    sh = _STATE["shade"]
    _set(sh, "u_m", _mat(r._m3d))
    _set(sh, "u_n", _mat(np.linalg.inv(view[:3, :3]).T))
    _set(sh, "u_px", (2.0 * r.scale * ss / W, 2.0 * r.scale * ss / H))
    base, surf, glow = (np.zeros((MATS, 4), "f4") for _ in range(3))
    for i, mt in enumerate(mats):
        base[i] = (*mt[0], mt[1])
        surf[i] = mt[2:6]
        glow[i] = (*mt[7], mt[6])
    _set(sh, "u_base", base.tobytes())
    _set(sh, "u_surf", surf.tobytes())
    _set(sh, "u_glow", glow.tobytes())
    flags = (1 if any(mt[6] for mt in mats) else 0) | (2 if any(any(mt[7]) for mt in mats) else 0) | \
        (4 if env is not None else 0)
    _set(sh, "u_flags", flags)
    for name, v in zip(("keyDir", "fillDir", "keyCol", "amt"), lt):
        _set(sh, name, tuple(float(x) for x in v))
    dirs, cols = np.zeros((3, 3), "f4"), np.zeros((3, 3), "f4")
    for j, (d, col) in enumerate(more[:3]):
        dirs[j], cols[j] = d, col
    _set(sh, "u_more", len(more[:3]))
    _set(sh, "u_dir", dirs.tobytes())
    _set(sh, "u_col", cols.tobytes())
    if env is not None:
        for name, v in zip(("envTop", "envHor", "envBot", "envDown"), env[:4]):
            _set(sh, name, tuple(float(x) for x in v))
        _set(sh, "envAmt", float(env[4]))
    if fg is None:
        _set(sh, "u_fog", (0.0, 1.0, 0.0, 1.0))
    else:
        col, near, far, amount, focus = fg
        _set(sh, "u_fog", (near, max(far - near, 1e-6), amount, focus))
        _set(sh, "u_fogCol", tuple(col))


def draw(r, c: skia.Canvas, L, m, mats, lights, fg, edging, amt: float, order, tracked: bool = False):
    """Draw model L (mesh m, materials mats, lights (key, more, env), fog fg, edges as render3d._edging gives them,
    the first amt of its triangles in build order `order`) on canvas c as render3d.draw would. Returns its projected
    bounds in device pixels, or None. tracked: render3d.draw has worked out what a tracked render reports (the
    model's box, shadow casters, focus reach, and whether the camera is inside it, cuts it or the fog hides it), so
    only the pixels are drawn here."""
    from . import render3d
    T = m.T if amt >= 1.0 else m.T[order[:math.ceil(amt * len(m.T))]]
    P = m.P
    r.on_gl.add(L.id)
    if not len(T):
        return None
    view = np.linalg.inv(r._lens) @ r._m3d
    lo3, hi3 = render3d._fixed(r, L, m, "box", lambda: (P.min(0), P.max(0)))
    if not tracked:
        eye = np.linalg.solve(view, [0.0, 0.0, -1.0 / r._lens[3, 2], 1.0])[:3]
        if (eye > lo3).all() and (eye < hi3).all() and render3d._within(eye, P, T):
            r.inside.add(L.id)
        if r._stage is not None and r._stage.casters is not None:
            r._stage.cast(P @ view[:3, :3].T + view[:3, 3], T)
    # The box's corners bound where the mesh can be on screen while all of them are ahead of the camera (w is
    # linear, and a box seen in perspective covers its corners' outline). These bounds place the passes; check and
    # --boxes take the model's box from render3d.draw.
    box = np.array([[(lo3, hi3)[i >> k & 1][k] for k in range(3)] for i in range(8)])
    hom = box @ r._m3d[:3, :3].T + r._m3d[:3, 3]
    w = box @ r._m3d[3, :3] + r._m3d[3, 3]
    ahead = w.min() >= space.NEAR
    if not ahead or fg is not None or (r._focused is not None and not tracked):  # these need each vertex's w
        wv = P @ r._m3d[3, :3] + r._m3d[3, 3]
        if amt < 1.0:
            wv = wv[np.unique(T)]
        if wv.max() < space.NEAR:
            return None
        if wv.min() < space.NEAR and not tracked:
            r.culled.add(L.id)  # GL clips what crosses the camera plane; render3d leaves those triangles out
        if r._focused is not None and not tracked:
            r._reach(wv)
        if fg is not None:
            fv = render3d.fogged(fg, wv)
            if fv.min() >= 0.999 and not tracked:
                r.fogged.add(L.id)
            if fv.max() <= 0.002:
                fg = None
    half = 0.0 if edging is None else edging[0] / 2 * r.scale  # device px a line reaches past the mesh
    if ahead:
        X, Y = hom[:, 0] / w * r.scale, hom[:, 1] / w * r.scale
        bounds = [float(X.min()), float(Y.min()), float(X.max()), float(Y.max())]
        x0, y0 = max(0, math.floor(bounds[0] - half) - 2), max(0, math.floor(bounds[1] - half) - 2)
        x1, y1 = min(r.w, math.ceil(bounds[2] + half) + 2), min(r.h, math.ceil(bounds[3] + half) + 2)
    else:  # a clipped triangle can reach anywhere on the frame
        bounds = [0.0, 0.0, float(r.w), float(r.h)]
        x0, y0, x1, y1 = 0, 0, r.w, r.h
    if x1 <= x0 or y1 <= y0:
        return bounds
    # A tracked render (check, --boxes) keeps its Skia models and stages at 1x with multisampling; these targets
    # have none, so they stay at 2x and are scaled down as they are drawn.
    ss = max(2, r._model_surface()[1])
    W, H = r.w * ss, r.h * ss
    mgl, ctx = _STATE["mgl"], _STATE["ctx"]
    tg = _targets(W, H)
    buf = _buffers(r, L, m)
    rad = half * ss  # texels
    reach = math.ceil(rad) + 1
    bx0, bx1 = max(0, x0 * ss - reach), min(W, x1 * ss + reach)  # the first pass's box: the second's, and its taps
    by0, by1 = max(0, H - y1 * ss - reach), min(H, H - y0 * ss + reach)
    _uniforms(r, mats, lights, fg, view, ss, W, H)
    first = tg["first"]
    first.viewport = (0, 0, W, H)
    first.scissor = (bx0, by0, bx1 - bx0, by1 - by0)
    first.use()
    first.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    ctx.enable_only(mgl.DEPTH_TEST | mgl.CULL_FACE)
    ctx.depth_func = "<"
    ctx.front_face = "cw"  # canvas y runs down: render3d's faces toward the camera wind clockwise on the target
    ctx.cull_face = "back"
    vao = buf.vao if amt >= 1.0 else buf.in_order(m, order)
    vao.render(mgl.TRIANGLES, vertices=3 * len(T))
    ctx.enable_only(0)
    edge = _STATE["edge"]
    _set(edge, "u_box", (bx0, by0, bx1, by1))
    _set(edge, "u_r", float(rad) if edging is not None else 0.0)
    if edging is not None:
        focus = -1.0 / r._lens[3, 2]
        _set(edge, "u_cos", float(edging[2]))
        _set(edge, "u_gap", STEP / (focus * space.NEAR))
        _set(edge, "u_ink", tuple(float(v) for v in edging[1]))
    second = tg["second"]
    second.scissor = None
    second.viewport = (x0 * ss, H - y1 * ss, (x1 - x0) * ss, (y1 - y0) * ss)
    second.use()
    tg["col"].use(0)
    tg["geo"].use(1)
    _STATE["sampler"].use(0)
    _STATE["sampler"].use(1)
    _STATE["quad"].render(mgl.TRIANGLE_STRIP)
    r.gr.resetContext()
    region = skia.Rect(x0 * ss, y0 * ss, x1 * ss, y1 * ss)
    if L.raw.get("ground") not in (None, False):
        render3d._ground(r, c, L.raw["ground"], P, P if amt >= 1.0 else P[T.ravel()], fg, lights[0][0])
    if r._stage is not None and r._stage.child is L:
        dep = _image(r, tg["outd"], W, H)
        d = r._dev  # the stage's px per device px: ss in a render, where this copies each texel
        r._stage.kidd.getCanvas().drawImageRect(dep, region, skia.Rect(x0 * d, y0 * d, x1 * d, y1 * d),
                                                skia.SamplingOptions(), skia.Paint(BlendMode=skia.BlendMode.kSrc))
    img = _image(r, tg["out"], W, H)
    c.save()
    c.resetMatrix()
    c.scale(r._dev, r._dev)
    sampling = skia.SamplingOptions(skia.FilterMode.kLinear) if ss > r._dev else skia.SamplingOptions()
    c.drawImageRect(img, region, skia.Rect(x0, y0, x1, y1), sampling, None)
    c.restore()
    # The next model draws into the same textures: Skia's reads of them go to the GPU first.
    r.gr.flushAndSubmit(skia.GrSyncCpu.kNo)
    return bounds


def _image(r, tex, w: int, h: int) -> skia.Image:
    """Texture tex as a Skia image, its first row at the bottom as GL keeps it."""
    bt = skia.GrBackendTexture(w, h, skia.GrMipmapped.kNo, skia.GrGLTextureInfo(_GL_TEXTURE_2D, tex.glo, _GL_RGBA8))
    return skia.Image.MakeFromTexture(r.gr, bt, skia.kBottomLeft_GrSurfaceOrigin, skia.kRGBA_8888_ColorType,
                                      skia.kPremul_AlphaType)
