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
nothing this path does not have: explode, edges: "all" and a see-through model with edge lines draw on the Skia path,
as does --wire. KINE_3D=skia turns this path off; KINE_3D=gl makes it an error when it cannot run.

A model's see-through materials are left out of the first draw and drawn after it, over what is there: the triangles
render3d would draw (those turned away too, lit from inside), sorted far to near each frame, tested against the depth
buffer and not written to it. Edge lines are found from the nearest surface, so the lines of what shows through a
see-through part would be lost: such a model stays on the Skia path.

A model with textures (its own or a model file's, and emission textures) draws with a second program, which reads
each material's picture at every pixel. skia-python does not say which GL texture an image of Skia's is in, so each
image is copied once, by Skia, into a texture made here, and mipmaps are built for it (_Pictures). A video's or a
group's next frame is copied into the texture the last one had. A textured model of more than TEX_MATS materials
or UNITS pictures draws on the Skia path.

A model's copies ("repeat") that move are placed here, one mesh drawn once for each from a matrix, an offset and a
material slot of its own (draw's `copies`), where render3d merges them into one mesh each frame. That is for a model
all of whose points nothing else needs: see render3d._placing.

A stage's shadow map is drawn here too whenever models can be (sun_map): what casts, through a depth buffer along
the key light. A model drawn here casts from the buffers it drew from; one Skia drew has its triangles sent.

The two paths differ where they are built differently: edge lines are found per pixel here and along the mesh's edges
there, triangles that cross the camera plane are clipped here where render3d drops them, the shadow map keeps the
nearest caster at each texel here where Skia draws whole triangles far to near, and a picture is placed at each
pixel here where render3d spreads it evenly over triangles it cuts until that is within a pixel of right."""
from __future__ import annotations

import math
import os
import weakref

import numpy as np
import skia

from . import space

MATS = 64  # materials one draw takes: schema.COPY_SLOTS, the most a model with copies has
TEX_MATS = 32  # materials a textured model's draw takes: each has four more uniforms, and OpenGL 3.3 promises 1024
UNITS = 16  # pictures a textured model's draw takes: the texture units OpenGL 3.3 promises a fragment shader
STEP = 6.0  # canvas px a depth must step between pixels a line's width apart for a line to be drawn there
SETTINGS = ("auto", "skia", "gl")
_GL_TEXTURE_2D, _GL_RGBA8 = 0x0DE1, 0x8058

_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_nrm;
in float in_mat;@decl
uniform mat4 u_m;   // model space to canvas px (x and y over w), w from the camera's lens
uniform mat3 u_n;   // model normals into camera space
uniform vec2 u_px;  // canvas px to clip space: 2 * scale * supersampling over the target's width and height
uniform vec4 u_fog; // near, far - near, amount, the distance of z 0 from the camera; amount 0: none
noperspective out vec2 v_n2;  // across the screen, as Skia interpolates render3d's normal
noperspective out float v_fog;
out vec3 v_n;
flat out int v_mat;
void main() {
    vec4 h = u_m * vec4(@pos, 1.0);
    gl_Position = vec4(h.x * u_px.x - h.w, h.w - h.y * u_px.y, h.w - 2.0 * NEAR, h.w);
    v_n = u_n * @nrm;
    v_n2 = normalize(v_n).xy;
    v_mat = @mat;@set
    v_fog = u_fog.z > 0.0 ? floor(clamp((h.w * u_fog.w - u_fog.x) / u_fog.y, 0.0, 1.0) * u_fog.z * 255.0 + 0.5)
                            / 255.0 : 0.0;
}
"""
# render3d._SKSL with _MORE and _ENV_MAIN, for every material at once. The @ marks are filled by _PLAIN, which gives
# that, or by _TEXTURED.
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
uniform vec3 u_fogCol;@decl
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_geo;
void main() {
    vec4 b = u_base[v_mat];
    vec4 p = u_surf[v_mat];
    vec2 n2 = gl_FrontFacing ? v_n2 : -v_n2;  // a see-through face turned away is lit from inside
    vec3 n = normalize(vec3(n2, sqrt(max(0.0, 1.0 - dot(n2, n2)))));
    float d = max(dot(n, keyDir), 0.0);
    float f = max(dot(n, fillDir), 0.0);
    float s = pow(max(dot(n, normalize(keyDir + vec3(0.0, 0.0, 1.0))), 0.0), p.y) * p.x;
    if (b.a > 1.5) {
        d = ceil(d * b.a - 0.25) / b.a;
        f = ceil(f * b.a - 0.25) / b.a;
        s = step(0.45, s) * p.x;
    }
    vec3 diffuse = @fill(amt.z + amt.x * d * keyCol + amt.y * f * vec3(0.85, 0.9, 1.0)) * (1.0 - 0.75 * p.z);
    vec3 spec = mix(vec3(1.0), b.rgb, p.z) * s * amt.x * keyCol;
    for (int j = 0; j < u_more; j++) {
        float dj = max(dot(n, u_dir[j]), 0.0);
        float sj = pow(max(dot(n, normalize(u_dir[j] + vec3(0.0, 0.0, 1.0))), 0.0), p.y) * p.x;
        if (b.a > 1.5) {
            dj = ceil(dj * b.a - 0.25) / b.a;
            sj = step(0.45, sj) * p.x;
        }
        diffuse += @filldj * u_col[j] * (1.0 - 0.75 * p.z);
        spec += mix(vec3(1.0), b.rgb, p.z) * sj * u_col[j];
    }
    float rim = pow(1.0 - n.z, 3.0) * p.w * amt.w;
    vec3 @sum + rim * keyCol;
    if ((u_flags & 4) != 0) {
        vec3 e = vec3(2.0 * n.z * n.x, 2.0 * n.z * n.y, 2.0 * n.z * n.z - 1.0);
        float up = -dot(e, envDown);
        vec3 sky = up > 0.0 ? mix(envHor, envTop, sqrt(up)) : mix(envHor, envBot, sqrt(-up));
        float sh = clamp((p.x - 0.4 * p.z) / 0.6, 0.0, 1.0);
        float fr = 0.04 + 0.96 * pow(1.0 - n.z, 5.0);
        @env += mix((envTop + 2.0 * envHor + envBot) * 0.25, sky, sh) * (p.z * b.rgb + (1.0 - p.z) * sh * fr) * envAmt;
    }
@show    o_col = vec4(mix(clamp(lit, 0.0, 1.0), u_fogCol, v_fog), 1.0);
    o_geo = vec4(normalize(gl_FrontFacing ? v_n : -v_n), NEAR * gl_FragCoord.w);
@last}
"""
# What fills the marks in _VERT and _FRAG for a model without textures: the light on each material's fill. Its
# see-through materials are left out of the first draw (u_pass 0) and drawn after it, far to near: over what is
# there, the colour times the opacity (u_pass 1), then, for a stage, as the nearest surface (u_pass 2).
_PLAIN = {"decl": ("""
uniform float u_alpha[MATS];  // each material's opacity: 1 - its transparency
uniform int u_pass;
flat out float v_a;""", "\nuniform int u_pass;\nflat in float v_a;"), "set": """
    v_a = u_alpha[v_mat];
    if (u_pass == 0 && v_a < 1.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }""", "pos": "in_pos", "nrm": "in_nrm", "mat": "int(in_mat + 0.5)", "fill": "b.rgb * ",
          "sum": "lit = diffuse + spec", "env": "lit", "show": """
    if ((u_flags & 1) != 0) {
        lit = max(lit, b.rgb * u_glow[v_mat].a);
    }
    if ((u_flags & 2) != 0) {
        lit += u_glow[v_mat].rgb;
    }
""".lstrip("\n"), "last": """
    if (u_pass == 1) {
        o_col *= v_a;
    } else if (u_pass == 2) {
        o_col = o_geo;
    }
""".lstrip("\n")}
# And for one with textures, as render3d._textured combines its passes: each material's picture (its texture times
# its tint, over its fill) times the light on a white surface, plus the highlights, rim and reflection, plus what the
# material gives off (its emission texture over its emission colour). A picture is read where the texture
# coordinates put it in perspective, at the mipmap level their change across the screen picks.
_TEXTURED = {"decl": ("\nin vec2 in_uv;\nout vec2 v_uv;", """
in vec2 v_uv;
uniform vec4 u_look[MATS];   // each material's tint, and the unit its texture is on (-1: none)
uniform vec4 u_uv[MATS];     // its texture's repeat across and down, then its offset
uniform vec4 u_shine[MATS];  // the same for its emission texture
uniform vec4 u_suv[MATS];
uniform sampler2D u_tex[UNITS];
uniform vec2 u_fold[UNITS];  // 1 where a unit's picture mirrors as it repeats, across and down
vec4 picture(float unit, vec4 uv, vec2 dx, vec2 dy) {
    int k = int(unit + 0.5);
    vec2 q = v_uv * uv.xy - uv.zw;
    q = mix(q, 1.0 - abs(mod(q, 2.0) - 1.0), u_fold[k]);
    dx *= uv.xy * 0.70710678;  // half a mipmap level sharper, as Skia reads its mipmaps
    dy *= uv.xy * 0.70710678;
PICK
}""".rstrip("\n")), "set": "\n    v_uv = in_uv;", "pos": "in_pos", "nrm": "in_nrm", "mat": "int(in_mat + 0.5)",
             "fill": "", "sum": "add = spec", "env": "add", "show": """
    if ((u_flags & 1) != 0) {
        diffuse = max(diffuse, vec3(u_glow[v_mat].a));
    }
    vec2 dx = dFdx(v_uv);
    vec2 dy = dFdy(v_uv);
    vec3 pic = b.rgb;
    if (u_look[v_mat].a >= 0.0) {
        vec4 c = picture(u_look[v_mat].a, u_uv[v_mat], dx, dy);
        pic = c.rgb * u_look[v_mat].rgb + pic * (1.0 - c.a);
    }
    vec3 lit = pic * min(diffuse, 2.0) + add;
    if ((u_flags & 2) != 0) {
        vec3 e = u_glow[v_mat].rgb;
        if (u_shine[v_mat].a >= 0.0) {
            vec4 c = picture(u_shine[v_mat].a, u_suv[v_mat], dx, dy);
            e = c.rgb * u_shine[v_mat].rgb + e * (1.0 - c.a);
        }
        lit += e;
    }
""".lstrip("\n"), "last": ""}
# What places a model's copies, in either program: each copy's matrix (by columns) and offset, and the slot it takes
# among those of a material whose fill the copies set. u_slot holds each material's first slot, and 1 where the
# copies take their own after it.
_COPIED = {"pos": "(mat3(i_a, i_b, i_c) * in_pos + i_off)", "nrm": "(mat3(i_a, i_b, i_c) * in_nrm)",
           "mat": "int(u_slot[int(in_mat + 0.5)].x + u_slot[int(in_mat + 0.5)].y * i_q + 0.5)"}
_COPIED_DECL = """
in vec3 i_a;
in vec3 i_b;
in vec3 i_c;
in vec3 i_off;
in float i_q;
uniform vec2 u_slot[MATS];"""
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

# A stage's shadow map: what casts, seen along the key light. u_m takes a vertex to the map's clip space across and
# up, and to how near the light it is (0.001 farthest, 0.999 nearest, as stage._sun stores it) in z; the depth test
# keeps the nearest at each texel.
_SUN_VERT = """
#version 330
in vec3 in_pos;
uniform mat4 u_m;
out float v_d;
void main() {
    vec4 h = u_m * vec4(in_pos, 1.0);
    v_d = h.z;
    gl_Position = vec4(h.xy, 1.0 - 2.0 * h.z, 1.0);
}
"""
_SUN_FRAG = """
#version 330
in float v_d;
out vec4 o_col;
void main() {  // stage.py's _PACK
    float d = floor(clamp(v_d, 0.0, 1.0) * 16777215.0 + 0.5);
    float hi = floor(d / 65536.0);
    float mid = floor((d - hi * 65536.0) / 256.0);
    o_col = vec4(hi / 255.0, mid / 255.0, (d - hi * 65536.0 - mid * 256.0) / 255.0, 1.0);
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


def _program(ctx, fills: dict, mats: int):
    """The shading program with the marks of _VERT and _FRAG filled from fills, for that many materials."""
    pick = "".join(f"    if (k == {i}) {{ return textureGrad(u_tex[{i}], q, dx, dy); }}\n" for i in range(UNITS - 1))
    pick += f"    return textureGrad(u_tex[{UNITS - 1}], q, dx, dy);"
    out = []
    for src, decl in zip((_VERT, _FRAG), fills["decl"]):
        for mark, text in {**fills, "decl": decl}.items():
            src = src.replace("@" + mark, text)
        out.append(src.replace("PICK", pick).replace("MATS", str(mats)).replace("UNITS", str(UNITS))
                   .replace("NEAR", f"{space.NEAR:.6f}"))
    return ctx.program(vertex_shader=out[0], fragment_shader=out[1])


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
            shade = _program(ctx, _PLAIN, MATS)
            edge = ctx.program(vertex_shader=_QUAD, fragment_shader=_EDGE)
            sun = ctx.program(vertex_shader=_SUN_VERT, fragment_shader=_SUN_FRAG)
        except Exception as e:  # noqa: BLE001 - moderngl raises its own Error and plain exceptions alike
            _STATE.update(ctx=None, error=f"{type(e).__name__}: {e}")
            return None
        try:  # a driver that will not take this one leaves the models with textures on Skia
            tex = _program(ctx, _TEXTURED, TEX_MATS)
            tex["u_tex"].value = list(range(UNITS))
        except Exception:  # noqa: BLE001
            tex = None
        edge["t_col"].value, edge["t_geo"].value = 0, 1
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4"))
        # Skia binds sampler objects of its own; these replace them on the two units read, so a mipmap filter left
        # on a unit cannot make a texture incomplete (texelFetch reads it as 0 then).
        nearest = ctx.sampler(filter=(moderngl.NEAREST, moderngl.NEAREST), repeat_x=False, repeat_y=False)
        _STATE.update(ctx=ctx, mgl=moderngl, shade=shade, tex=tex, edge=edge, sun=sun, sampler=nearest, tiles={},
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
    if explode > 0 or (edging is not None and edging[2] > 1.0):
        return False
    shown = [tx for tx in (*texs, *glows) if tx is not None]
    if not shown:
        # Edge lines are found from the nearest surface: those of what shows through a see-through one would go.
        return len(mats) <= MATS and (edging is None or all(mt[8] >= 1 for mt in mats))
    return (_STATE.get("tex") is not None and len(mats) <= TEX_MATS
            and len({(id(tx[0]), tx[4]) for tx in shown}) <= UNITS)


def copying(textured: bool) -> bool:
    """Whether a model's copies can be placed here (draw's `copies`): the driver took the program that does it,
    which is made the first time one is asked for."""
    name = "tex" if textured else "shade"
    if name + "_copies" not in _STATE:
        fills, prog = _TEXTURED if textured else _PLAIN, None
        if _STATE.get(name) is not None:
            try:
                prog = _program(_STATE["ctx"], {**fills, **_COPIED, "decl": (fills["decl"][0] + _COPIED_DECL,
                                                                             fills["decl"][1])},
                                TEX_MATS if textured else MATS)
                if textured:
                    prog["u_tex"].value = list(range(UNITS))
            except Exception:  # noqa: BLE001 - then the copies are merged, as they were
                prog = None
        _STATE[name + "_copies"] = prog
    return _STATE[name + "_copies"] is not None


def _mat(a: np.ndarray) -> bytes:
    """Matrix a as GLSL reads one: by columns."""
    return np.ascontiguousarray(np.asarray(a, "f4").T).tobytes()


def _targets(w: int, h: int):
    """Frame-sized render targets: the first pass's colour, normal and 1/w (float) and depth buffer, and the second
    pass's colour and packed depth, made once for each size. tint and near are the first pass's colour and its
    normal and 1/w, each alone with the depth buffer: what see-through triangles are drawn into."""
    hit = _TARGETS.pop((w, h), None)
    if hit is None:
        ctx = _STATE["ctx"]
        col, geo, out, outd = (ctx.texture((w, h), 4), ctx.texture((w, h), 4, dtype="f4"), ctx.texture((w, h), 4),
                               ctx.texture((w, h), 4))
        depth = ctx.depth_renderbuffer((w, h))
        hit = {"col": col, "geo": geo, "out": out, "outd": outd, "depth": depth,
               "first": ctx.framebuffer([col, geo], depth), "second": ctx.framebuffer([out, outd]),
               "tint": ctx.framebuffer([col], depth), "near": ctx.framebuffer([geo], depth)}
        hit["tint"].depth_mask = False  # see-through triangles hide nothing: they are drawn far to near
        while len(_TARGETS) >= 2:
            for v in _TARGETS.pop(next(iter(_TARGETS))).values():
                v.release()
    _TARGETS[(w, h)] = hit
    return hit


class _Buffers:
    """One mesh's vertices and triangles on the GPU, its triangles in build order once a draw needs them, and its
    texture coordinates once a textured draw does."""

    def __init__(self, m):
        ctx = _STATE["ctx"]
        self.key = (m.P, m.N, m.M, m.T)
        v = np.concatenate([m.P, m.N, m.M[:, None]], 1).astype("f4")
        self.vbo = ctx.buffer(v.tobytes())
        self.ibo = ctx.buffer(np.ascontiguousarray(m.T, "i4").tobytes())
        self.built = None  # (order, ibo): the triangles in that build order
        self.uv = None  # the texture coordinates: a mesh's stay with its triangles
        self.picked = None  # a buffer for the triangles a frame picks (some)
        self.placed = None  # a buffer for where a frame puts the mesh's copies (copies)
        self.arrays = {}  # (program, whether in build order): a vertex array
        self.dead = False

    def casts(self, m, order) -> bool:
        """Whether these still hold mesh m, and its triangles in build order `order` if that is not None: what a
        model that cast a shadow drew."""
        in_order = order is None or (self.built is not None and self.built[0] is order)
        return not self.dead and self.holds(m) and in_order

    def holds(self, m) -> bool:
        return all(a is b for a, b in zip(self.key, (m.P, m.N, m.M, m.T)))

    def moved(self, m) -> bool:
        """Take mesh m's vertices if it has these triangles (the same array: moving copies keep theirs)."""
        if m.T is not self.key[3] or len(m.P) * 28 != self.vbo.size:
            return False
        self.vbo.write(np.concatenate([m.P, m.N, m.M[:, None]], 1).astype("f4").tobytes())
        self.key = (m.P, m.N, m.M, m.T)
        return True

    def _drop(self, which):
        for key in [k for k in self.arrays if which(k)]:
            self.arrays.pop(key).release()

    def array(self, name: str, m, order=None):
        """A vertex array for program `name` (shade, tex or sun) over m's triangles: all of them, or in build order
        `order` (render3d keeps one while the mesh holds)."""
        ctx = _STATE["ctx"]
        if order is not None and (self.built is None or self.built[0] is not order):
            self._drop(lambda k: k[1])
            if self.built is not None:
                self.built[1].release()
            self.built = (order, ctx.buffer(np.ascontiguousarray(m.T[order], "i4").tobytes()))
        key = (name, order is not None)
        if key not in self.arrays:
            content = [(self.vbo, "3f 16x", "in_pos")] if name == "sun" else self._content(name, m)
            self.arrays[key] = ctx.vertex_array(_STATE[name], content, self.ibo if order is None else self.built[1])
        return self.arrays[key]

    def _content(self, name: str, m) -> list:
        """What a vertex array for program `name` (shade or tex) reads of each vertex."""
        content = [(self.vbo, "3f 3f 1f", "in_pos", "in_nrm", "in_mat")]
        if name == "tex":
            if self.uv is None:
                # A mesh without texture coordinates shows the first texel of each picture, as it does on Skia.
                uv = np.zeros((len(m.P), 2), "f4") if m.U is None else np.ascontiguousarray(m.U, "f4")
                self.uv = _STATE["ctx"].buffer(uv.tobytes())
            content.append((self.uv, "2f", "in_uv"))
        return content

    def copies(self, name: str, m, data: np.ndarray):
        """A vertex array for the program that places copies for `name` (shade or tex) over all of m's triangles,
        once for each row of data: a copy's matrix by columns, its offset and its slot (13 numbers), until the
        next call."""
        ctx, raw = _STATE["ctx"], np.ascontiguousarray(data, "f4").tobytes()
        if self.placed is None or self.placed.size < len(raw):
            self._drop(lambda k: k[1] == "copies")
            if self.placed is not None:
                self.placed.release()
            self.placed = ctx.buffer(reserve=len(raw), dynamic=True)
        self.placed.orphan()
        self.placed.write(raw)
        key = (name, "copies")
        if key not in self.arrays:
            content = self._content(name, m) + [(self.placed, "3f 3f 3f 3f 1f/i", "i_a", "i_b", "i_c", "i_off", "i_q")]
            self.arrays[key] = ctx.vertex_array(_STATE[name + "_copies"], content, self.ibo)
        return self.arrays[key]

    def some(self, name: str, tris: np.ndarray):
        """A vertex array for program `name` over these triangles of the mesh, until the next call."""
        data = np.ascontiguousarray(tris, "i4").tobytes()
        if self.picked is None:
            self.picked = _STATE["ctx"].buffer(reserve=self.ibo.size, dynamic=True)
        self.picked.orphan()
        self.picked.write(data)
        key = (name, "some")
        if key not in self.arrays:
            self.arrays[key] = _STATE["ctx"].vertex_array(_STATE[name], [(self.vbo, "3f 3f 1f", "in_pos", "in_nrm",
                                                                         "in_mat")], self.picked)
        return self.arrays[key]

    def release(self):
        for o in (*self.arrays.values(), self.ibo, self.vbo, *(self.built or ())[1:],
                  *filter(None, [self.uv, self.picked, self.placed])):
            o.release()
        self.arrays = {}
        self.dead = True


class _Pictures:
    """The images a renderer's models show, each copied once into a texture of this module's with mipmaps:
    skia-python does not say which texture an image of Skia's is in. A copy is kept while some material shows its
    image; a material's next image (a video's or a group's next frame) takes the texture of the last."""

    def __init__(self):
        self.shown = {}  # (layer id, material, "look" or "glow"): the image that material showed last
        self.copies = {}  # id(image): [the image, its texture, how many materials show it]
        self.spare = []  # textures no material shows

    def take(self, slot: tuple, img: skia.Image):
        """The texture for img, which the material `slot` shows, and whether img is still to be copied into it."""
        was = self.shown.get(slot)
        if was is img:
            return self.copies[id(img)][1], False
        self.shown[slot] = img
        if was is not None:
            old = self.copies[id(was)]
            old[2] -= 1
            if not old[2]:
                self.spare.append(self.copies.pop(id(was))[1])
        hit = self.copies.get(id(img))
        if hit is not None:
            hit[2] += 1
            return hit[1], False
        size = (img.width(), img.height())
        tex = next((t for t in self.spare if t.size == size), None)
        if tex is None:
            tex = _STATE["ctx"].texture(size, 4)
        else:
            self.spare.remove(tex)
        for t in self.spare[:-4]:  # a few are kept: a video's frames take turns in one
            self.spare.remove(t)
            t.release()
        self.copies[id(img)] = [img, tex, 1]
        return tex, True

    def release(self):
        for t in (*(hit[1] for hit in self.copies.values()), *self.spare):
            t.release()
        self.shown, self.copies, self.spare = {}, {}, []


def _pictures(r, L, texs, glows) -> tuple[list, list]:
    """The pictures model L's materials show, ready to draw: [(texture, tiling)], one for each unit, and each
    material's unit for its texture and for its emission texture (-1: none). Images not copied yet are, by Skia."""
    kept = r._gl_pictures
    if kept is None:
        kept = r._gl_pictures = _Pictures()
        weakref.finalize(r, kept.release)
    units, where, fresh = {}, [], []
    for kind, txs in (("look", texs), ("glow", glows)):
        where.append([-1] * len(txs))
        for i, tx in enumerate(txs):
            if tx is None:
                continue
            tex, new = kept.take((L.id, i, kind), tx[0])
            if new:
                bt = skia.GrBackendTexture(*tex.size, skia.GrMipmapped.kNo,
                                           skia.GrGLTextureInfo(_GL_TEXTURE_2D, tex.glo, _GL_RGBA8))
                surf = skia.Surface.MakeFromBackendTexture(r.gr, bt, skia.kTopLeft_GrSurfaceOrigin, 0,
                                                           skia.kRGBA_8888_ColorType, None)
                surf.getCanvas().drawImage(tx[0], 0, 0, skia.SamplingOptions(),
                                           skia.Paint(BlendMode=skia.BlendMode.kSrc))
                fresh.append((surf, tex))
            where[-1][i] = units.setdefault((tex.glo, tx[4]), (len(units), tex, tx[4]))[0]
    if fresh:
        r.gr.resetContext()  # buffers and textures made since Skia last drew are bound where it left its own
        r.gr.flushAndSubmit(skia.GrSyncCpu.kNo)
        for _, tex in fresh:
            tex.build_mipmaps()
    return [(tex, tile) for _, tex, tile in units.values()], where


def _tiled(tile: tuple):
    """A sampler that repeats across and down where the tiling says "repeat": a mirrored picture is folded by the
    program, and a clamped one stays on its edge."""
    key = (tile[0] == "repeat", tile[1] == "repeat")
    if key not in _STATE["tiles"]:
        mgl = _STATE["mgl"]
        _STATE["tiles"][key] = _STATE["ctx"].sampler(filter=(mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR), repeat_x=key[0],
                                                     repeat_y=key[1])
    return _STATE["tiles"][key]


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


def _uniforms(sh, n: int, r, mats, lights, fg, view: np.ndarray, ss: int, W: int, H: int, shines: bool = False):
    """Set program sh (which takes n materials) for these materials, lights and fog. shines: a material has an
    emission texture."""
    lt, more, env = lights
    _set(sh, "u_m", _mat(r._m3d))
    _set(sh, "u_n", _mat(np.linalg.inv(view[:3, :3]).T))
    _set(sh, "u_px", (2.0 * r.scale * ss / W, 2.0 * r.scale * ss / H))
    base, surf, glow = (np.zeros((n, 4), "f4") for _ in range(3))
    for i, mt in enumerate(mats):
        base[i] = (*mt[0], mt[1])
        surf[i] = mt[2:6]
        glow[i] = (*mt[7], mt[6])
    _set(sh, "u_base", base.tobytes())
    _set(sh, "u_surf", surf.tobytes())
    _set(sh, "u_glow", glow.tobytes())
    flags = (1 if any(mt[6] for mt in mats) else 0) | (2 if shines or any(any(mt[7]) for mt in mats) else 0) | \
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


def _see_through(r, L, m, T: np.ndarray, ids, mats) -> np.ndarray:
    """The triangles among T (triangles ids of mesh m) that see-through materials draw, as render3d.draw picks and
    orders them: those turned to the camera, and those turned away (which show through the near ones) unless they
    are the back of a face drawn back to back, far to near."""
    from . import render3d
    clear = np.array([mt[8] < 1 for mt in mats])[m.M[T[:, 0]]]
    if not clear.any():
        return T[:0]
    hom = m.P @ r._m3d[[0, 1, 3], :3].T + r._m3d[[0, 1, 3], 3]  # x and y times w, and w
    front = np.linalg.det(hom[T]) > 0  # the sign of the area on screen, also for a triangle the camera plane cuts
    face = render3d._topology(r, L, m)[1][ids]
    seen = np.zeros(int(face.max()) + 2, bool)
    seen[face[front]] = True
    near, far = np.flatnonzero(clear & front), np.flatnonzero(clear & ~front & ~seen[face])
    pick = T[np.r_[near, far]]
    return pick[np.argsort(-hom[:, 2][pick].sum(1), kind="stable")]


def _corners(lo, hi) -> np.ndarray:
    """The eight corners of the box from lo to hi."""
    return np.array([[(lo, hi)[i >> k & 1][k] for k in range(3)] for i in range(8)])


def _in_a_copy(eye: np.ndarray, P: np.ndarray, T: np.ndarray, lo, hi, copies: tuple) -> bool:
    """Whether the point eye is inside the mesh (P, T, in the box lo to hi) as its copies place it: inside an odd
    number of them, as render3d._within finds for copies merged into one mesh when each is closed."""
    from .render3d import _within
    turn, size, off = copies[:3]
    ok = size != 0
    e = ((eye - off[ok])[:, None, :] @ turn[ok])[:, 0, :] / size[ok][:, None]  # eye in each copy's own space
    near = e[((e > lo) & (e < hi)).all(1)]
    return sum(_within(q, P, T) for q in near) % 2 == 1


def _depths(P: np.ndarray, copies: tuple, row: np.ndarray, k: float) -> np.ndarray:
    """The least and the greatest of row . p + k over the points of every copy of the mesh with points P."""
    turn, size, off = copies[:3]
    along = P @ (row @ turn).T  # each point (down) along each copy's (across) own direction for row
    lo, hi = along.min(0) * size, along.max(0) * size
    base = off @ row + k
    return np.array([(np.minimum(lo, hi) + base).min(), (np.maximum(lo, hi) + base).max()])


def _shown(sh, looks, units: list, where: list):
    """Set the textured program for the pictures of _pictures: each material's tint, unit, repeat and offset."""
    for txs, at, tint, uv in zip(looks, where, ("u_look", "u_shine"), ("u_uv", "u_suv")):
        a, b = np.zeros((TEX_MATS, 4), "f4"), np.zeros((TEX_MATS, 4), "f4")
        a[:, 3] = -1.0
        for i, tx in enumerate(txs):
            if tx is not None:
                a[i] = (*(tx[3][:3] if tx[3] is not None else (1.0, 1.0, 1.0)), at[i])
                b[i] = (*tx[1], *tx[2])
        _set(sh, tint, a.tobytes())
        _set(sh, uv, b.tobytes())
    fold = np.zeros((UNITS, 2), "f4")
    for k, (tex, tile) in enumerate(units):
        fold[k] = (tile[0] == "mirror", tile[1] == "mirror")
    _set(sh, "u_fold", fold.tobytes())


def draw(r, c: skia.Canvas, L, m, mats, lights, fg, edging, amt: float, order, tracked: bool = False, looks=None,
         copies=None):
    """Draw model L (mesh m, materials mats, lights (key, more, env), fog fg, edges as render3d._edging gives them,
    the first amt of its triangles in build order `order`) on canvas c as render3d.draw would. Returns its projected
    bounds in device pixels, or None. tracked: render3d.draw has worked out what a tracked render reports (the
    model's box, shadow casters, focus reach, and whether the camera is inside it, cuts it or the fog hides it), so
    only the pixels are drawn here. looks: each material's texture and each one's emission texture, as
    render3d.model gives them, for a model that has any. copies: m is one copy of the model, drawn whole once for
    each of these (render3d._placing): each one's turn (a matrix, n of them), size and offset, then each material's
    slots (or None: its own) and each copy's among them (or None). The bounds are then of a box round the copies'
    boxes, which can be wider than the model."""
    from . import render3d
    T = m.T if amt >= 1.0 else m.T[order[:math.ceil(amt * len(m.T))]]
    P = m.P
    r.on_gl.add(L.id)
    if not len(T):
        return None
    view = np.linalg.inv(r._lens) @ r._m3d
    lo3, hi3 = own = render3d._fixed(r, L, m, "box", lambda: (P.min(0), P.max(0)))
    if copies is not None:
        each = _corners(*own) @ (copies[0] * copies[1][:, None, None]).transpose(0, 2, 1) + copies[2][:, None, :]
        lo3, hi3 = each.min((0, 1)), each.max((0, 1))
    if not tracked:
        eye = np.linalg.solve(view, [0.0, 0.0, -1.0 / r._lens[3, 2], 1.0])[:3]
        if (eye > lo3).all() and (eye < hi3).all() and (
                render3d._within(eye, P, T) if copies is None else _in_a_copy(eye, P, T, *own, copies)):
            r.inside.add(L.id)
        if r._stage is not None and r._stage.casters is not None:
            # The shadow map draws these triangles from the buffers this draw fills, while they still hold them.
            r._stage.cast(P @ view[:3, :3].T + view[:3, 3], T,
                          (_buffers(r, L, m), m, None if amt >= 1.0 else order, view))
    # The box's corners bound where the mesh can be on screen while all of them are ahead of the camera (w is
    # linear, and a box seen in perspective covers its corners' outline). These bounds place the passes; check and
    # --boxes take the model's box from render3d.draw.
    box = _corners(lo3, hi3)
    hom = box @ r._m3d[:3, :3].T + r._m3d[:3, 3]
    w = box @ r._m3d[3, :3] + r._m3d[3, 3]
    ahead = w.min() >= space.NEAR
    if not ahead or fg is not None or (r._focused is not None and not tracked):  # these need each vertex's w
        if copies is not None:
            wv = _depths(P, copies, r._m3d[3, :3], r._m3d[3, 3])
        else:
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
    glass = T[:0]  # the triangles of see-through materials: a textured model draws solid, as on Skia
    name, most = ("shade", MATS) if looks is None else ("tex", TEX_MATS)
    sh = _STATE[name if copies is None else name + "_copies"]
    if looks is None:
        units = []
        _uniforms(sh, MATS, r, mats, lights, fg, view, ss, W, H)
        _set(sh, "u_alpha", np.array([mt[8] for mt in mats] + [1.0] * (MATS - len(mats)), "f4").tobytes())
        _set(sh, "u_pass", 0)
        if copies is None:  # copies placed here have no see-through material
            glass = _see_through(r, L, m, T, slice(None) if amt >= 1.0 else order[:len(T)], mats)
    else:
        units, where = _pictures(r, L, *looks)
        _uniforms(sh, TEX_MATS, r, mats, lights, fg, view, ss, W, H, any(g is not None for g in looks[1]))
        _shown(sh, looks, units, where)
    if copies is not None:
        turn, size, off, slots, pick = copies
        slot = np.zeros((most, 2), "f4")
        slot[:, 0] = np.arange(most)
        if slots is not None:
            slot[:len(slots), 0], slot[:len(slots), 1] = slots[:, 0], slots[:, -1] != slots[:, 0]
        _set(sh, "u_slot", slot.tobytes())
        placed = np.concatenate([(turn * size[:, None, None]).transpose(0, 2, 1).reshape(-1, 9), off,
                                 (np.zeros(len(off)) if pick is None else pick)[:, None]], 1)
    first = tg["first"]
    first.viewport = (0, 0, W, H)
    first.scissor = (bx0, by0, bx1 - bx0, by1 - by0)
    first.use()
    first.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    ctx.enable_only(mgl.DEPTH_TEST | mgl.CULL_FACE)
    ctx.depth_func = "<"
    ctx.front_face = "cw"  # canvas y runs down: render3d's faces toward the camera wind clockwise on the target
    ctx.cull_face = "back"
    for k, (tex, tile) in enumerate(units):
        tex.use(k)
        _tiled(tile).use(k)
    if copies is not None:
        buf.copies(name, m, placed).render(mgl.TRIANGLES, vertices=3 * len(T), instances=len(placed))
    else:
        buf.array(name, m, None if amt >= 1.0 else order).render(mgl.TRIANGLES, vertices=3 * len(T))
    if len(glass):
        some = buf.some("shade", glass)
        staged = r._stage is not None and r._stage.child is L  # a stage merges by the nearest surface's depth
        for name, flags in (("tint", mgl.DEPTH_TEST | mgl.BLEND), ("near", mgl.DEPTH_TEST))[:2 if staged else 1]:
            tg[name].viewport = first.viewport
            tg[name].use()
            ctx.enable_only(flags)
            ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
            _set(sh, "u_pass", 1 if name == "tint" else 2)
            some.render(mgl.TRIANGLES, vertices=glass.size)
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
        render3d._ground(r, c, L.raw["ground"], P, P if amt >= 1.0 else P[T.ravel()], fg, lights[0][0], L.id)
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


def _image(r, tex, w: int, h: int, origin=skia.kBottomLeft_GrSurfaceOrigin) -> skia.Image:
    """Texture tex as a Skia image, its first row at the bottom as GL keeps it (or at the top, with that origin)."""
    bt = skia.GrBackendTexture(w, h, skia.GrMipmapped.kNo, skia.GrGLTextureInfo(_GL_TEXTURE_2D, tex.glo, _GL_RGBA8))
    return skia.Image.MakeFromTexture(r.gr, bt, origin, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)


def _loose(P: np.ndarray, tris: np.ndarray):
    """A vertex array for the shadow map's program over corners P and triangles tris, which no model's buffers
    hold: sent through two buffers kept for that."""
    ctx = _STATE["ctx"]
    v, i = np.ascontiguousarray(P, "f4").tobytes(), np.ascontiguousarray(tris, "i4").tobytes()
    kept = _STATE.get("loose")
    if kept is None or kept[0].size < len(v) or kept[1].size < len(i):
        if kept is not None:
            for o in kept[::-1]:
                o.release()
        vbo, ibo = (ctx.buffer(reserve=max(len(b) * 3 // 2, 1 << 16), dynamic=True) for b in (v, i))
        kept = _STATE["loose"] = (vbo, ibo, ctx.vertex_array(_STATE["sun"], [(vbo, "3f", "in_pos")], ibo))
    else:
        kept[0].orphan()
        kept[1].orphan()
    kept[0].write(v)
    kept[1].write(i)
    return kept[2]


def sun_map(r, casters: list, to_map: np.ndarray, side: int) -> skia.Image:
    """A stage's shadow map, `side` texels square: the triangles of casters (as Stage.cast keeps them) that are
    turned from the key light, each texel holding the nearest of them to the light, packed as stage.py packs a
    depth. to_map takes camera space to the map: texels across it and from its first row to its last, and nearness
    (0.001 to 0.999) in z. A caster drawn here comes from the buffers its draw filled, while they hold what it drew;
    any other is sent as it is."""
    mgl, ctx, sun = _STATE["mgl"], _STATE["ctx"], _STATE["sun"]
    r.gr.flushAndSubmit(skia.GrSyncCpu.kNo)  # a stage drawn before this one may still be to read the last map
    tg = _STATE.get("sun_target")
    if tg is None or tg[0].size != (side, side):
        tex, depth = ctx.texture((side, side), 4), ctx.depth_renderbuffer((side, side))
        tg = _STATE["sun_target"] = (tex, depth, ctx.framebuffer([tex], depth))
    fbo = tg[2]
    fbo.viewport = (0, 0, side, side)
    fbo.scissor = None
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    ctx.enable_only(mgl.DEPTH_TEST | mgl.CULL_FACE)
    ctx.depth_func = "<"
    # Across and up the map with the light behind the viewer, a triangle turned from the light runs clockwise.
    ctx.front_face = "cw"
    ctx.cull_face = "back"
    clip = np.diag([2.0 / side, 2.0 / side, 1.0, 1.0]) @ to_map
    clip[:2, 3] -= 1.0
    loose, n = [], 0
    for P, tris, on in casters:
        if on is not None and on[0].casts(on[1], on[2]):
            buf, m, order, view = on
            _set(sun, "u_m", _mat(clip @ view))
            buf.array("sun", m, order).render(mgl.TRIANGLES, vertices=3 * len(tris))
        else:
            loose.append((P, tris + n))
            n += len(P)
    if loose:
        vao = _loose(np.concatenate([p for p, _ in loose]), np.concatenate([t for _, t in loose]))
        _set(sun, "u_m", _mat(clip))
        vao.render(mgl.TRIANGLES, vertices=3 * sum(len(t) for _, t in loose))
    ctx.enable_only(0)
    r.gr.resetContext()
    return _image(r, tg[0], side, side, skia.kTopLeft_GrSurfaceOrigin)
