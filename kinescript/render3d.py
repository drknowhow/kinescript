"""Model layers, drawn each frame by numpy and one drawVertices call: transform, drop triangles that
face away or cross the camera plane, sort the rest far to near (painter's algorithm), and shade per
pixel in SkSL. The view-space normal rides in each vertex's texture coordinate with the material
index packed into it, so every part of a model draws in one call and hides the others correctly."""
from __future__ import annotations

import math

import numpy as np
import skia

from . import color as C
from . import media, mesh, models, schema, space, svgpath
from .props import compile_prop

MATERIAL = ("fill", "shine", "metal", "rim", "bands", "glow", "texture", "transparency")
MAX_MATERIALS = 8
# Lights are fixed to the camera by default, like a studio key light on the camera rig: turning a model
# changes its shading, orbiting the camera around it does not. "follow": "world" fixes one in the scene instead.
LIGHTS = {"studio": {"key": 0.85, "fill": 0.25, "ambient": 0.22, "rim": 0.35, "from": "topleft"},
          "soft": {"key": 0.55, "fill": 0.4, "ambient": 0.38, "rim": 0.2, "from": "top"},
          "dramatic": {"key": 1.05, "fill": 0.06, "ambient": 0.06, "rim": 0.75, "from": "left"},
          "flat": {"key": 0.0, "fill": 0.0, "ambient": 1.0, "rim": 0.0, "from": "front"}}
FROM = {"topleft": (-0.5, -0.6, 0.62), "top": (0.0, -0.75, 0.66), "topright": (0.5, -0.6, 0.62),
        "left": (-0.85, -0.25, 0.46), "right": (0.85, -0.25, 0.46), "front": (0.0, 0.0, 1.0),
        "bottomleft": (-0.5, 0.6, 0.62), "bottom": (0.0, 0.75, 0.66), "bottomright": (0.5, 0.6, 0.62)}
MORE_LIGHTS = 3  # "lights" past the key, fill and rim
# Environments that shiny and metal models reflect: the colour above, at and below the horizon.
ENVS = {"studio": {"top": "#f6f3ec", "horizon": "#7d828c", "bottom": "#23252b"},
        "sunset": {"top": "#33407a", "horizon": "#ff9b5e", "bottom": "#2a1c26"},
        "night": {"top": "#0a0f24", "horizon": "#34477a", "bottom": "#050608"},
        "chrome": {"top": "#ffffff", "horizon": "#15171c", "bottom": "#9a9da3"}}
FIXED = {"preset", "from", "follow", "type"}  # light keys that do not tween
DOF_MAX = 48.0  # the most a layer or a stage pixel blurs out of focus (sigma, px)
SWIM = 1.0  # px a textured triangle's image may lie from where perspective puts it before the triangle is cut
TILE = {"repeat": skia.TileMode.kRepeat, "mirror": skia.TileMode.kMirror, "clamp": skia.TileMode.kClamp}
_DOUBLE = skia.ColorFilters.Matrix([2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0])
_HALF = skia.ColorFilters.Matrix([0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0])
_SKSL = """
DECLARE
uniform half3 keyDir;
uniform half3 fillDir;
uniform half3 keyCol;
uniform half4 amt;
half4 main(float2 uv) {
    float m = floor(uv.x * 0.5);
    half4 b = base0;
    half4 p = surf0;
    LOOKUP
    half2 n2 = half2((uv.x - 2.0 * m - 0.5) / 0.45, uv.y);
    half3 n = normalize(half3(n2, sqrt(max(0.0, 1.0 - dot(n2, n2)))));
    half d = max(dot(n, keyDir), 0.0);
    half f = max(dot(n, fillDir), 0.0);
    half s = pow(max(dot(n, normalize(keyDir + half3(0.0, 0.0, 1.0))), 0.0), p.y) * p.x;
    if (b.a > 1.5) {
        d = ceil(d * b.a - 0.25) / b.a;
        f = ceil(f * b.a - 0.25) / b.a;
        s = step(0.45, s) * p.x;
    }
    half3 diffuse = b.rgb * (amt.z + amt.x * d * keyCol + amt.y * f * half3(0.85, 0.9, 1.0)) * (1.0 - 0.75 * p.z);
    half3 spec = mix(half3(1.0), b.rgb, p.z) * s * amt.x * keyCol;MOREENV
    half rim = pow(1.0 - n.z, 3.0) * p.w * amt.w;
    return half4(saturate(LIT), 1.0);
}
"""
# Light j of "lights", shaded as the key is (without a rim) and added to diffuse and spec.
_MORE = """
    half dJ = max(dot(n, dirJ), 0.0);
    half sJ = pow(max(dot(n, normalize(dirJ + half3(0.0, 0.0, 1.0))), 0.0), p.y) * p.x;
    if (b.a > 1.5) {
        dJ = ceil(dJ * b.a - 0.25) / b.a;
        sJ = step(0.45, sJ) * p.x;
    }
    diffuse += b.rgb * dJ * colJ * (1.0 - 0.75 * p.z);
    spec += mix(half3(1.0), b.rgb, p.z) * sJ * colJ;"""
# The environment seen in the surface: the view (straight down -z) reflected off it, how far up it looks in the
# world, and that sky colour, blurred toward the mean as shine falls. Metal reflects it tinted by its fill, anything
# else by its shine and a Fresnel term, so a matte model reflects nothing.
_ENV = """
uniform half3 envTop;
uniform half3 envHor;
uniform half3 envBot;
uniform half3 envDown;
uniform half envAmt;
"""
_ENV_MAIN = """
    half3 e = half3(2.0 * n.z * n.x, 2.0 * n.z * n.y, 2.0 * n.z * n.z - 1.0);
    half up = -dot(e, envDown);
    half3 sky = up > 0.0 ? mix(envHor, envTop, sqrt(up)) : mix(envHor, envBot, sqrt(-up));
    half sh = saturate((p.x - 0.4 * p.z) / 0.6);
    half fr = 0.04 + 0.96 * pow(1.0 - n.z, 5.0);
    half3 refl = mix((envTop + 2.0 * envHor + envBot) * 0.25, sky, sh) * (p.z * b.rgb + (1.0 - p.z) * sh * fr) * envAmt;"""
# A textured model's picture (each texture over its fill) times twice the light pass (half the light on a white
# surface) plus the add pass (highlights, rim and reflection), under its fog: the passes _effect's parts draw.
_COMBINE = """
uniform shader base;
uniform shader light;
uniform shader add;
uniform shader fog;
half4 main(float2 p) {
    half4 t = base.eval(p);
    half4 f = fog.eval(p);
    return half4(saturate(t.rgb * light.eval(p).rgb * 2.0 + add.eval(p).rgb) * (1.0 - f.a) + f.rgb, t.a);
}
"""
# The same with the light a model file's materials give off themselves (emission), added before the fog.
_COMBINE_EMIT = _COMBINE.replace("uniform shader fog;", "uniform shader fog;\nuniform shader emit;").replace(
    "+ add.eval(p).rgb)", "+ add.eval(p).rgb + emit.eval(p).rgb)")
# A material's picture: its texture, times its tint, over its fill.
_LOOK = """
uniform shader image;
uniform half4 fill;
uniform half3 tint;
half4 main(float2 p) {
    half4 c = image.eval(p);
    return half4(c.rgb * tint, c.a) + fill * (1.0 - c.a);
}
"""
# Every material's picture in one shader: material k's coordinates come (k + 0.5) strides to the right (_picked).
_PICK = """
uniform float stride;
{children}
half4 main(float2 p) {{
    float k = floor(p.x / stride);
    float2 q = float2(p.x - (k + 0.5) * stride, p.y);
{branches}
}}
"""
_EFFECTS: dict[tuple, skia.RuntimeEffect] = {}


def _effect(n: int, more: int = 0, glow: bool = False, env: bool = False, part: str | None = None,
            emit: bool = False, edge: bool = False, clear: bool = False) -> skia.RuntimeEffect:
    """The shading effect for n materials, one pair of uniforms each (base colour + bands, then
    specular amount, exponent, metal and rim), a glow each if any glows, `more` lights, and an
    environment to reflect. Without these the source is what it was before they existed, so those
    models draw the same bytes. part "light" is half the light that falls on a white surface (or the glow, if
    more), and "add" what is added over the surface's colour, for a textured model (_COMBINE). emit: an emission
    colour each, added to the light of an untextured model (a textured one draws it as a pass of its own). edge:
    slot n is the model's edge lines (draw), drawn in their colour, unlit. clear: an opacity each (1 - its
    transparency), the colour premultiplied by it, for an untextured model."""
    if (n, more, glow, env, part, emit, edge, clear) not in _EFFECTS:
        decl = "\n".join(f"uniform half4 base{i};\nuniform half4 surf{i};" + (f"\nuniform half glow{i};" if glow else "")
                         + (f"\nuniform half3 emit{i};" if emit else "")
                         + (f"\nuniform half alpha{i};" if clear else "") for i in range(n))
        decl += "".join(f"\nuniform half3 dir{j};\nuniform half3 col{j};" for j in range(more)) + (_ENV if env else "")
        g = ("; g = glow{0}" if glow else "") + ("; e = emit{0}" if emit else "") + ("; a = alpha{0}" if clear else "")
        pick = " ".join(f"if (m == {i}.0) {{ b = base{i}; p = surf{i}{g.format(i)}; }}" for i in range(1, n))
        if emit:
            pick = "half3 e = emit0;\n    " + pick
        if clear:
            pick = "half a = alpha0;\n    " + pick
        if glow:
            pick = "half g = glow0;\n    " + pick
        if edge:
            decl += "\nuniform half3 edgeCol;" if part is None else ""
            line = {None: "edgeCol", "light": "half3(0.5)", "add": "half3(0.0)"}[part]
            pick = f"if (m == {n}.0) {{ return half4({line}, 1.0); }}\n    " + pick
        lit = "diffuse + spec + rim * keyCol" + (" + refl" if env else "")
        lit = f"max({lit}, b.rgb * g)" if glow else lit
        lit = f"{lit} + e" if emit and part is None else lit
        src = _SKSL.replace("saturate(LIT), 1.0)", "saturate(LIT) * a, a)") if clear and part is None else _SKSL
        if part == "light":
            src = src.replace("half3 diffuse = b.rgb * ", "half3 diffuse = ")
            lit = "max(diffuse, half3(g)) * 0.5" if glow else "diffuse * 0.5"
        elif part == "add":
            lit = "spec + rim * keyCol" + (" + refl" if env else "")
        more_src = "".join(_MORE.replace("J", str(j)) for j in range(more))
        if part == "light":
            more_src = more_src.replace("diffuse += b.rgb * ", "diffuse += ")
        src = (src.replace("DECLARE", decl).replace("LOOKUP", pick).replace("MORE", more_src)
               .replace("ENV", _ENV_MAIN if env else "").replace("LIT", lit))
        _EFFECTS[n, more, glow, env, part, emit, edge, clear] = skia.RuntimeEffect.MakeForShader(src)
    return _EFFECTS[n, more, glow, env, part, emit, edge, clear]


def light(spec) -> dict:
    """A light preset name or {key, fill, ambient, rim, from, color, ...} on top of the studio preset."""
    if isinstance(spec, dict):
        out = dict(LIGHTS["studio"], color="#fff4e6", **LIGHTS.get(spec.get("preset"), {}))
        out.update({k: v for k, v in spec.items() if k != "preset"})
        return out
    return dict(LIGHTS.get(spec or "studio", LIGHTS["studio"]), color="#fff4e6")


def at(cache: dict, ctx, spec, t: float, d: float, T: float):
    """A light object with each key that tweens (keyframes, a tween or an expression) at local time t of d.
    Compiled once per object and kept in cache."""
    if not isinstance(spec, dict):
        return spec
    hit = cache.get(id(spec))
    if hit is None or hit[0] is not spec:
        hit = cache[id(spec)] = (spec, {k: compile_prop(k, v, ctx) for k, v in spec.items()
                                        if k not in FIXED and not isinstance(v, (int, float))})
    return dict(spec, **{k: p.at(t, d, T) for k, p in hit[1].items()})


def _unit(v) -> list[float]:
    v = np.asarray(v, float)
    return (v / (np.linalg.norm(v) or 1.0)).tolist()


def _rgb(v) -> list[float]:
    return list((v if isinstance(v, tuple) else C.parse(v))[:3])


def toward(lt: dict) -> list[float]:
    """Unit vector toward a light, in the space it follows: its `from`, with `azimuth` (degrees round from the
    front, toward the right) and `elevation` (degrees up) in place of those parts of it."""
    v = _unit(FROM.get(lt.get("from"), FROM["topleft"]))
    az, el = lt.get("azimuth"), lt.get("elevation")
    if az is None and el is None:
        return v
    az = math.radians(float(az)) if az is not None else math.atan2(v[0], v[2])
    el = math.radians(float(el)) if el is not None else math.asin(max(-1.0, min(1.0, -v[1])))
    return [math.cos(el) * math.sin(az), -math.sin(el), math.cos(el) * math.cos(az)]


def _lights(lt: dict, R=None) -> tuple:
    """Key and fill directions in camera space, key colour, and the four amounts, as the shader takes them.
    R turns world directions into camera space, for a key that follows the world."""
    key = toward(lt)
    if lt.get("follow") == "world" and R is not None:
        key = _unit(R @ key)
    return (key, _unit((-key[0], 0.35, 0.8)), _rgb(lt.get("color", "#fff4e6")),
            [float(lt.get(k, 0)) for k in ("key", "fill", "ambient", "rim")])


def _more(specs, R, centre, w: float, h: float) -> list[tuple]:
    """The extra lights as (direction in camera space, colour times intensity). A point light lights the
    whole model from where its centre is, fading to half at `range` px."""
    out = []
    for s in specs[:MORE_LIGHTS]:
        col = np.asarray(_rgb(s.get("color", "#ffffff"))) * float(s.get("intensity", 0.6))
        if s.get("type") == "point":
            v = np.array([float(s.get("x", w / 2)), float(s.get("y", h / 2)), float(s.get("z", 600.0))]) - centre
            dist = float(np.linalg.norm(v)) or 1.0
            d = R @ (v / dist)
            col = col / (1.0 + (dist / max(float(s.get("range", 1000.0)), 1e-6)) ** 2)
        else:
            d = np.asarray(toward(s))
            if s.get("follow") == "world":
                d = R @ d
        out.append((_unit(d), col.tolist()))
    return out


def environment(spec) -> dict | None:
    """An environment preset name or {top, horizon, bottom, amount} on top of a preset (studio), or None."""
    if isinstance(spec, str):
        spec = {"preset": spec}
    if not isinstance(spec, dict):
        return None
    return dict(ENVS.get(spec.get("preset"), ENVS["studio"]), amount=1.0,
                **{k: v for k, v in spec.items() if k != "preset"})


def sun(r, T: float) -> tuple[list[float], float]:
    """The scene's key light at scene time T: its direction in camera space and its amount."""
    lt = light(at(r._lit, r.s.ctx, r.s.light, T, r.s.duration, T))
    cam = r._cam if r._cam is not None else r._lens
    key, _, _, amounts = _lights(lt, (np.linalg.inv(r._lens) @ cam)[:3, :3])
    return key, amounts[0]


def fog(r, T: float) -> tuple | None:
    """The scene's fog at scene time T as (colour, near, far, amount, focus), near and far in px from the camera
    and focus the distance of the canvas plane (z 0) from it, or None. A group drawn as a texture has none."""
    if r.s.fog is None or r._lens is None or r._insets:
        return None
    f = at(r._lit, r.s.ctx, r.s.fog, T, r.s.duration, T)
    focus = -1.0 / r._lens[3, 2]
    return (_rgb(f.get("color", _fog_colour(r.s.bg))), float(f.get("near", focus)), float(f.get("far", 4 * focus)),
            float(f.get("amount", 1.0)), focus)


def _fog_colour(bg):
    """Fog's default colour: the canvas background's, or its gradient's first stop."""
    if isinstance(bg, dict):
        stops = bg.get("stops") or ["#000000"]
        bg = stops[0][1] if isinstance(stops[0], (list, tuple)) else stops[0]
    return bg if C.is_color(bg) else "#000000"


def fogged(fg: tuple, w):
    """How much of fog fg covers points whose camera projection gives them w: 0 (clear) up to its amount."""
    col, near, far, amt, focus = fg
    return np.clip((np.asarray(w, float) * focus - near) / max(far - near, 1e-6), 0.0, 1.0) * amt


def defocus(dof: tuple, w):
    """The blur (sigma, frame px) of points whose camera projection gives them w, under depth of field dof =
    (focus distance, aperture, z 0's distance): aperture px per 1000 px from the focus plane near it, growing
    without end toward the camera and to aperture times the focus distance over 1000 far away, as a lens does.
    Never past DOF_MAX."""
    near, ap, focus = dof
    d = np.asarray(w, float) * focus
    return np.minimum(ap / 1000.0 * np.abs(d - near) * near / np.maximum(d, 1.0), DOF_MAX)


def fog_filter(fg: tuple, f: float) -> skia.ColorFilter:
    """A colour filter that mixes a layer f of the way to the fog's colour."""
    r_, g_, b_ = fg[0]
    k = 1.0 - f
    return skia.ColorFilters.Matrix([k, 0, 0, 0, r_ * f, 0, k, 0, 0, g_ * f, 0, 0, k, 0, b_ * f, 0, 0, 0, 1, 0])


def _env(e: dict | None, R) -> tuple | None:
    """The environment as the shader takes it: top, horizon and bottom colours, the world's down in camera
    space, and the amount."""
    if e is None:
        return None
    down = _unit(R[:, 1])
    return _rgb(e["top"]), _rgb(e["horizon"]), _rgb(e["bottom"]), down, float(e.get("amount", 1.0))


def _shade(nv, M, mats, key, fill, col, amt, more=(), env=None, fog=None, split=False, edge=None):
    """The shader's lighting, per vertex, as 0xAARRGGBB: the raster backend's path. split: instead, half each of
    the two parts a textured model draws (_effect's "light" and "add"; _textured). Vertices past the materials
    are edge lines (draw), in the colour edge. Each vertex takes its material's opacity (1 - its transparency)
    as its alpha."""
    line = M >= len(mats)
    M = np.minimum(M, len(mats) - 1)
    base = np.array([(*m[0], m[1]) for m in mats])[M]
    sp, ex, metal, rim = np.array([m[2:6] for m in mats])[M].T
    n = np.c_[nv[:, :2], np.sqrt(np.clip(1 - (nv[:, :2] ** 2).sum(1), 0, 1))]
    d, f = np.clip(n @ key, 0, None), np.clip(n @ fill, 0, None)
    s = np.clip(n @ _unit(np.add(key, (0, 0, 1))), 0, None) ** ex * sp
    b = base[:, 3]
    toon = b > 1.5
    d = np.where(toon, np.ceil(d * b - 0.25) / np.maximum(b, 1), d)
    f = np.where(toon, np.ceil(f * b - 0.25) / np.maximum(b, 1), f)
    s = np.where(toon, (s >= 0.45) * sp, s)
    col = np.asarray(col)
    light = amt[2] + amt[0] * d[:, None] * col + amt[1] * f[:, None] * (0.85, 0.9, 1.0)
    glow = np.array([m[6] for m in mats])[M]
    if split:
        dim = (1 - 0.75 * metal)[:, None]
        tint = 1 - metal[:, None] + metal[:, None] * base[:, :3]
        lit, add = light * dim, tint * (s * amt[0])[:, None] * col
        for dj, cj in more:
            dd, sj = _another(n, dj, ex, sp, toon, b)
            lit += dd[:, None] * cj * dim
            add += tint * sj[:, None] * cj
        add += ((1 - n[:, 2]) ** 3 * rim * amt[3])[:, None] * col
        if env is not None:
            add += _reflect(n, sp, metal, base, env)
        if glow.any():
            lit = np.maximum(lit, glow[:, None])
        lit[line], add[line] = 1.0, 0.0
        return _argb(lit * 0.5), _argb(add * 0.5)
    rgb = base[:, :3] * light * (1 - 0.75 * metal)[:, None]
    rgb += (1 - metal[:, None] + metal[:, None] * base[:, :3]) * (s * amt[0])[:, None] * col
    for dj, cj in more:
        dd, sj = _another(n, dj, ex, sp, toon, b)
        rgb += base[:, :3] * (dd[:, None] * cj) * (1 - 0.75 * metal)[:, None]
        rgb += (1 - metal[:, None] + metal[:, None] * base[:, :3]) * sj[:, None] * cj
    rgb += ((1 - n[:, 2]) ** 3 * rim * amt[3])[:, None] * col
    if env is not None:
        rgb += _reflect(n, sp, metal, base, env)
    if glow.any():  # at least its fill times glow, however it is lit
        rgb = np.maximum(rgb, base[:, :3] * glow[:, None])
    emit = np.array([m[7] for m in mats])[M]
    if emit.any():  # the light it gives off itself
        rgb += emit
    if line.any():
        rgb[line] = edge
    if fog is not None:  # (how much of each vertex the fog covers, its colour)
        rgb = np.clip(rgb, 0, 1) * (1 - fog[0][:, None]) + np.asarray(fog[1]) * fog[0][:, None]
    alpha = np.array([m[8] for m in mats])[M]
    if (alpha < 1).any():
        alpha[line] = 1.0
        return _argb(rgb, alpha)
    return _argb(rgb)


def _another(n, dj, ex, sp, toon, b) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse and specular of one of "lights" at each vertex (normals n), as the key's."""
    dd = np.clip(n @ dj, 0, None)
    sj = np.clip(n @ _unit(np.add(dj, (0, 0, 1))), 0, None) ** ex * sp
    dd = np.where(toon, np.ceil(dd * b - 0.25) / np.maximum(b, 1), dd)
    sj = np.where(toon, (sj >= 0.45) * sp, sj)
    return dd, sj


def _reflect(n, sp, metal, base, env) -> np.ndarray:
    """The environment each vertex reflects (_ENV_MAIN)."""
    top, hor, bot, down, k = (np.asarray(v) for v in env)
    nz = n[:, 2:]
    e = np.c_[2 * nz * n[:, :2], 2 * nz ** 2 - 1]
    up = -(e @ down)[:, None]
    sky = np.where(up > 0, hor + (top - hor) * np.sqrt(np.clip(up, 0, None)),
                   hor + (bot - hor) * np.sqrt(np.clip(-up, 0, None)))
    sh = np.clip((sp - 0.4 * metal) / 0.6, 0, 1)[:, None]
    fr = 0.04 + 0.96 * (1 - nz) ** 5
    mean = (top + 2 * hor + bot) * 0.25
    return (mean + (sky - mean) * sh) * (metal[:, None] * base[:, :3] + (1 - metal[:, None]) * sh * fr) * k


def _argb(rgb: np.ndarray, alpha=None) -> np.ndarray:
    c = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint32)
    a = 0xFF000000 if alpha is None else np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint32) << 24
    return a | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]


def _texture(r, spec, L=None, t: float = 0.0, T: float = 0.0) -> tuple | None:
    """A texture as drawn on model L at its time t (scene time T): its image (mipmapped, and on the GPU when
    drawing there), repeat and offset across and down, tint (or None) and tiling; None for none. spec is a file
    name or "#groupId", {"src", "repeat", "offset"} with a video's "loop", "trim" and "speed", or a model file's
    {"image", "tint", "tile"}. A video shows its frame at t and a group is drawn at T, each frame anew."""
    if not spec:
        return None
    if isinstance(spec, str):
        spec = {"src": spec}
    kind = schema.texture_kind(spec["src"]) if "src" in spec else "image"
    if kind == "image":
        key = ("texture", spec["src"]) if "src" in spec else ("texture", id(spec["image"]))
        img = r._models.get(key)
        if img is None:
            img = r._models[key] = _mipped(r, media.image(r.s.path(spec["src"])) if "src" in spec else spec["image"])
    elif kind == "video" and not r.footage:  # boxes only: a clear picture, drawn as a texture all the same
        img = _CLEAR
    else:
        img = r._clip(L, spec, t) if kind == "video" else r._surface(spec["src"][1:], T)
        if img is None:  # no frame decoded, or a group drawn inside itself
            img = _CLEAR
        img = _mipped(r, img)
    pair = lambda v: (float(v), float(v)) if isinstance(v, (int, float)) else (float(v[0]), float(v[1]))  # noqa: E731
    return img, pair(spec.get("repeat", 1)), pair(spec.get("offset", 0)), spec.get("tint"), \
        tuple(spec.get("tile", ("repeat", "repeat")))


_CLEAR = skia.Image.fromarray(np.zeros((2, 2, 4), np.uint8), skia.kRGBA_8888_ColorType)


def _mipped(r, img: skia.Image) -> skia.Image:
    """img with mipmaps, on the GPU when drawing there."""
    return img.makeTextureImage(r.gr, skia.GrMipmapped.kYes) if r.gr is not None else img.withDefaultMipmaps()


def _material(get, emit=None) -> tuple:
    """A material as the shaders take it; emit is the light a model file's material gives off, not a layer key.
    Last, its opacity: 1 - its transparency."""
    col = get("fill")
    col = C.parse(col) if isinstance(col, str) else col
    shine, metal = float(get("shine", 0.4)), float(get("metal", 0.0))
    return (tuple(col[:3]), float(get("bands", 0)), 0.6 * shine + 0.4 * metal, 6 + 90 * shine * shine, metal,
            float(get("rim", 0.5)), float(get("glow", 0.0)), tuple(emit or (0.0, 0.0, 0.0)),
            1.0 - min(1.0, max(0.0, float(get("transparency", 0.0)))))


def _text_path(r, num, raw) -> tuple[skia.Path, float]:
    """Outline of the text, centred on the origin (lines on their middle, capitals on theirs), and its size."""
    from .render import _covers, _fallback, _ignorable  # render imports this module
    size = float(num("size", 120))
    fam = raw.get("font") or r.s.theme.get("font")
    font = skia.Font(r._typeface(fam, int(num("weight", 600)), bool(raw.get("italic"))), size)
    ls, lh = float(num("letterSpacing", 0)), size * float(num("lineHeight", 1.18))
    lines = str(raw.get("text", "")).split("\n")
    path = skia.Path()
    for i, line in enumerate(lines):
        glyphs = font.textToGlyphs(line)
        if all(glyphs):
            fonts = [font] * len(glyphs)
            xs = np.array(font.getXPos(glyphs)) + ls * np.arange(len(glyphs))
            width = font.measureText(line) + ls * max(0, len(glyphs) - 1)
        else:  # a letter the font lacks comes from the system font that has it (CJK, emoji), unshaped
            fonts, glyphs, adv = [], [], []
            for ch in line:
                if _ignorable(ch):
                    continue
                f = font
                if not _covers(font.getTypeface(), ord(ch)):
                    fb = _fallback(font.getTypeface().fontStyle(), ord(ch))
                    f = skia.Font(fb, size) if fb is not None else font
                gid = f.unicharToGlyph(ord(ch))
                fonts.append(f)
                glyphs.append(gid)
                adv.append(f.getWidths([gid])[0])
            xs = np.concatenate([[0.0], np.cumsum(adv)[:-1]]) + ls * np.arange(len(glyphs)) if adv else np.zeros(0)
            width = float(sum(adv)) + ls * max(0, len(glyphs) - 1)
        for f, gid, x in zip(fonts, glyphs, xs):
            gp = f.getPath(gid)
            if gp is not None:
                path.addPath(gp, float(x) - width / 2, i * lh)
    path.offset(0, font.getMetrics().fCapHeight / 2 - (len(lines) - 1) * lh / 2)
    return path, size


def _svg_path(num, raw) -> tuple[skia.Path, float]:
    """The `d` outline (or its stroke's, with round ends, given strokeWidth) centred on the origin, scaled to
    fit w and/or h when given, and its size."""
    path = svgpath.parse(str(raw.get("d", "")))
    sw = float(num("strokeWidth", 0) or 0)
    if sw > 0:
        outline = skia.Path()
        skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=sw, StrokeCap=skia.Paint.kRound_Cap,
                   StrokeJoin=skia.Paint.kRound_Join).getFillPath(path, outline)
        path = outline
    b = path.computeTightBounds()
    w, h = num("w", None), num("h", None)
    fits = [float(v) / s for v, s in ((w, b.width()), (h, b.height())) if v is not None and s > 0]
    s = min(fits) if fits else 1.0
    fit = skia.Matrix()
    fit.setScaleTranslate(s, s, -b.centerX() * s, -b.centerY() * s)
    path.transform(fit)
    return path, max(b.width(), b.height()) * s


def _extruded(r, shape: str, num, raw, detail: str) -> mesh.Mesh:
    """A text or path model's solid. Outlines are cached per text, font and size, solids per depth and bevel,
    so a changing counter or an animated depth costs one rebuild per distinct value."""
    k = mesh.DETAIL.get(detail, 1.0)
    if shape == "text":
        # Built from the rounded values in the key, not from whichever exact value was drawn first: otherwise a
        # frame depended on the frames drawn before it (renders split across workers differed).
        snap = {n: round(float(num(n, d)), 2) for n, d in (("size", 120), ("letterSpacing", 0), ("lineHeight", 1.18))}
        key = (shape, str(raw.get("text", "")), raw.get("font") or r.s.theme.get("font"), int(num("weight", 600)),
               bool(raw.get("italic")), *snap.values(), k)
        exact = num

        def num(n, d=None):
            return snap[n] if n in snap else exact(n, d)
    else:
        key = (shape, str(raw.get("d", "")), num("w", None), num("h", None), num("strokeWidth", 0), k)
    if len(r._solids) > 64:
        r._solids.clear()
    flat = r._solids.get(key)
    if flat is None:
        path, extent = _text_path(r, num, raw) if shape == "text" else _svg_path(num, raw)
        seg = max(1.0, min(10.0, extent * 0.04)) / k  # length of the pieces curves are cut into, in px
        flat = r._solids[key] = (*mesh.shape2d(mesh.contours(path, seg)), seg, extent)
    Q, starts, caps, seg, extent = flat
    depth, bevel = round(float(num("depth", round(extent * 0.2))), 2), round(float(num("bevel", 0)), 2)
    full = key + (depth, bevel)
    m = r._solids.get(full)
    if m is None:
        rows = max(1, min(4, round(depth / (3 * seg))))
        # Caps are cut into triangles up to 6 pieces long: one long cap triangle sorts by its middle and
        # can be drawn under walls it should cover (the hole of an O seen at an angle).
        m = r._solids[full] = (mesh.extrude(Q, starts, caps, depth, bevel, rows, 6 * seg) if len(caps)
                               else mesh.Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int32),
                                              np.zeros(0, np.int32)))
    return m


def _shape(r, shape: str, num, raw, detail: str, slot=None) -> mesh.Mesh:
    """One shape's mesh at its size, centred on the origin. num(key, default) reads its numbers, raw its text.
    With a slot, a primitive is kept there while its size holds: placing it again each frame copied the mesh
    and made every _fixed result be worked out again."""
    if shape in mesh.EXTRUDED:
        return _extruded(r, shape, num, raw, detail)
    s3, tube = mesh.size(shape, num)
    return _kept(r, slot, (shape, detail, tube, s3), lambda: mesh.placed(mesh.unit(shape, detail, tube), s3))


def _kept(r, slot, key, make):
    """make(), kept in r._models[slot] until key changes (no slot: not kept)."""
    hit = r._models.get(slot) if slot is not None else None
    if hit is None or hit[0] != key:
        hit = (key, make())
        if slot is not None:
            r._models[slot] = hit
    return hit[1]


def _parts(r, L, t: float = 0.0, T: float = 0.0):
    """The model's parts merged: mesh, and for each material slot the part keys that override the layer, at the
    layer's time t (scene time T). Parts whose keys are all plain values merge once; when one moves or changes
    size, the mesh is merged again on the frames where its values change, and is kept while they hold. Parts with
    the same material keys share a slot, whose values are read at t."""
    hit = r._models.get(L.id)
    raw = L.raw.get("parts") or []
    if hit is None:
        slots, keys, first = [], [{}], [None]  # each part's slot; each slot's keys and the part that set them
        for i, p in enumerate(raw):
            own = {k: p[k] for k in MATERIAL if k in p}
            if own not in keys:
                keys.append(own)
                first.append(i)
            slots.append(keys.index(own))
        if len(keys) > MAX_MATERIALS:
            raise ValueError(f"layer '{L.id}': {len(keys)} different materials; a model takes at most {MAX_MATERIALS}")
        props = L.parts if len(L.parts) == len(raw) else [{} for _ in raw]
        moving = [i for i, pp in enumerate(props) if any(not v.static for k, v in pp.items() if k not in MATERIAL)]
        tinted = [s for s, i in enumerate(first) if i is not None and any(
            not v.static for k, v in props[i].items() if k in MATERIAL)]
        still = None if moving or not raw else _merge(r, L, raw, slots)
        hit = r._models[L.id] = (still, keys, slots, first, props, moving, tinted)
    still, keys, slots, first, props, moving, tinted = hit
    if moving:
        vals = [dict(p, **{k: L.part(i, k, t, T) for k in props[i] if k not in MATERIAL})
                if i in moving else p for i, p in enumerate(raw)]
        held = tuple(_hashable(vals[i]) for i in moving)
        still = _kept(r, ("parts", L.id), held, lambda: _merge(r, L, vals, slots))
    if tinted:
        keys = [dict(o, **{k: L.part(first[s], k, t, T) for k in o if k != "texture"})
                if s in tinted else o for s, o in enumerate(keys)]
    return still, keys


def _merge(r, L, parts: list, slots: list) -> mesh.Mesh:
    """Parts with their values, placed in the model and merged into one mesh. A primitive's mesh at its size is
    kept for each part, so moving a part places the same mesh again."""
    detail = L.raw.get("detail", "medium")
    meshes = []
    for i, p in enumerate(parts):
        m = _shape(r, p.get("shape", "cube"), lambda k, d, p=p: p.get(k, d), p, detail, ("part", L.id, i))
        meshes.append(mesh.placed(m, (1.0, 1.0, 1.0), mesh.part_matrix(p)))
    return mesh.merge(meshes, slots)


def _hashable(v):
    """A value as a key: lists and dicts as tuples."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _imported(r, L, taken: int):
    """The layer's model file: its mesh with its materials in the slots after the first `taken`, those
    materials less the keys the layer sets itself (a layer fill recolours the whole model), and its size."""
    hit = r._models.get(("src", L.id))
    if hit is None:
        imp = models.load(L.src)
        M, mats = models.slots(imp, MAX_MATERIALS - taken)
        own = [{k: v for k, v in m.items() if k not in L.raw} for m in mats]
        m = imp.mesh._replace(M=np.where(M < 0, 0, M + taken).astype(np.int32))
        hit = r._models[("src", L.id)] = (m, own, imp.size, imp)
    return hit


def playing(L, imp, t: float) -> tuple[int, float] | None:
    """The animation of the layer's model file it plays at its time t, and the time into it: "animation" names
    it, or is {"name", "speed", "loop", "start"}, on the layer's clock. A clip loops unless loop is false, which
    holds its ends. None when there is none to play (kine check reports a name the file does not have)."""
    spec = L.raw.get("animation")
    if spec is None or spec is False or imp.motion is None:
        return None
    spec = spec if isinstance(spec, dict) else {"name": spec}
    i = models.clip_index(imp, spec.get("name", 0))
    if i is None:
        return None
    length = imp.motion.clips[i][2]
    at = float(spec.get("start", 0)) + t * float(spec.get("speed", 1))
    if length <= 0:
        return i, 0.0
    return i, at % length if spec.get("loop", True) else min(max(at, 0.0), length)


def _fit(g, size) -> float:
    """Scale of a model file: it fits inside w and h keeping its proportions, or its longest side is 300 px."""
    s = [float(v) / e for v, e in ((g("w", None), size[0]), (g("h", None), size[1])) if v is not None and e > 1e-9]
    return min(s) if s else 300.0 / max(float(size.max()), 1e-9)


def model(r, L, g, t: float = 0.0, T: float = 0.0) -> tuple[mesh.Mesh, list[tuple], list, list]:
    """The model's mesh in its own space at its time t (scene time T), its materials, and each material's texture
    and emission texture (a model file's) or None. "texture": false on the layer turns both off."""
    parts, keys = _parts(r, L, t, T)
    shape = L.raw.get("shape")
    meshes = [] if parts is None else [parts]
    if L.src:
        m, own, size, imp = _imported(r, L, len(keys))
        keys = keys + own
        fit = _fit(g, size)
        play = playing(L, imp, t)
        if play is None:
            meshes.insert(0, _kept(r, ("fit", L.id), fit, lambda: m._replace(P=m.P * fit)))
        else:  # posed at this time: a new mesh, kept while the time into the clip holds (its ends, a pause)
            def posed():
                P, N = models.pose(imp, *play)
                return m._replace(P=P * fit, N=N)
            meshes.insert(0, _kept(r, ("fit", L.id), (fit, play), posed))
    elif shape:
        meshes.insert(0, _shape(r, shape, g, L.raw, L.raw.get("detail", "medium"), ("shape", L.id)))
    if len(meshes) == 1:
        m = meshes[0]
    else:  # the same pieces as last frame: the same mesh (compared by identity, so the pieces are held)
        hit = r._models.get(("cat", L.id))
        if hit is None or len(hit[0]) != len(meshes) or any(a is not b for a, b in zip(hit[0], meshes)):
            hit = r._models[("cat", L.id)] = (meshes, mesh.cat(meshes))
        m = hit[1]
    accent = r.s.theme.get("accent", "#ff633f")
    root = lambda k, d=None: g(k, accent if k == "fill" and d is None else d)  # noqa: E731
    mats = [_material(lambda k, d=None, o=o: o.get(k, root(k, d)), o.get("emit")) for o in keys]
    glows = [_texture(r, o.get("emitTexture")) if L.raw.get("texture") is not False else None for o in keys]
    return m, mats, [_texture(r, o.get("texture", L.raw.get("texture")), L, t, T) for o in keys], glows


def _inner(p: np.ndarray) -> np.ndarray:
    """Which 2D points lie well inside the octagon through the points farthest along x, y and the two
    diagonals. None of those can be the farthest point in any direction."""
    x, y = p[:, 0], p[:, 1]
    corners = p[[int(np.argmax(v)) for v in (x, x + y, y, y - x, -x, -x - y, -y, x - y)]]
    edge = np.roll(corners, -1, axis=0) - corners
    inward = np.stack([-edge[:, 1], edge[:, 0]])  # each edge's normal into the octagon, unnormalised
    span = float(np.ptp(p, axis=0).max()) or 1.0
    # A repeated corner leaves an edge of length 0, whose side test is 0 for every point: all are kept.
    return (p @ inward - (corners * inward.T).sum(1) > 1e-9 * span * span).all(1)


def _hull(p: np.ndarray) -> np.ndarray:
    """Convex outline of 2D points, from the farthest point in each of 48 directions, in order around it.
    Points that cannot be farthest are dropped first: of the rocket's 8,513 vertices 674 are left, and its
    outline takes 0.44 ms instead of 1.8."""
    p = p[~_inner(p)]
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    h = p[np.unique(np.argmax(p @ np.stack([np.cos(a), np.sin(a)]), axis=0))]
    mid = h.mean(0)
    return h[np.argsort(np.arctan2(h[:, 1] - mid[1], h[:, 0] - mid[0]))]


def _within(eye: np.ndarray, P: np.ndarray, T: np.ndarray) -> bool:
    """Whether the point eye is inside the mesh (P, T): two of three rays from it, each a little off an axis so it
    misses edges, cross the surface an odd number of times. A camera in a torus's hole is outside it."""
    a = P[T[:, 0]] - eye
    e1, e2 = P[T[:, 1]] - eye - a, P[T[:, 2]] - eye - a
    odd = 0
    for d in np.eye(3) + [0.0011, 0.0023, 0.0037]:
        pv = np.cross(d, e2)
        det = (e1 * pv).sum(1)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        u = -(a * pv).sum(1) * inv
        qv = np.cross(-a, e1)
        v = (qv @ d) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & ((qv * e2).sum(1) * inv > 0)
        odd += int(hit.sum()) % 2
    return odd >= 2


def _ground(r, c: skia.Canvas, spec, rest: np.ndarray, pts: np.ndarray, fg=None, key=None):
    """A soft shadow under the model, as if lit from straight above (or with "from": "light", along key, the key
    light's direction in camera space): the footprint of its points on a level floor, darkest where it touches
    and wider and fainter as it rises. The floor lies in the layer's parent space, at the model's lowest point
    before it turns (`rest`), unless `y` places it."""
    spec = spec if isinstance(spec, dict) else {}
    local = np.linalg.solve(r._outer, r._m3d)  # the layer's own transform
    Q = pts @ local[:3, :3].T + local[:3, 3]
    low = float(Q[:, 1].max())  # y runs down
    floor = float(spec["y"]) if "y" in spec else local[1, 3] + rest[:, 1].max() * np.linalg.norm(local[:3, 1])
    lift = max(0.0, floor - low)
    xz = Q[:, [0, 2]]
    span = float(np.ptp(xz, axis=0).max()) or 1.0
    if spec.get("from") == "light" and key is not None:
        up = np.linalg.solve((np.linalg.inv(r._lens) @ r._outer)[:3, :3], key)  # toward the light, parent space
        if up[1] >= 0:
            return  # lit from below the floor
        step = up[[0, 2]] / -up[1]  # how far along the floor the shadow moves per px of height (y runs down)
        step *= min(1.0, 5.0 / (np.linalg.norm(step) or 1.0))  # a light under 11 degrees up casts it 5 heights
        xz = xz - (floor - Q[:, 1])[:, None] * step
    op, soft = float(spec.get("opacity", 0.5)), float(spec.get("blur", 0.08 * span))
    # A wide soft shadow of the whole footprint, and a tight dark one of the part within reach of the floor.
    touch = xz[Q[:, 1] > low - 0.04 * span]
    shapes = [(_hull(p), sigma, alpha) for p, sigma, alpha in
              ((xz, soft + 0.3 * lift, 0.55 * op / (1 + 4 * lift / span)),
               (touch, 0.25 * soft + 0.5 * lift, op * math.exp(-6 * lift / span))) if alpha >= 0.004 and len(p) >= 3]
    if not shapes:
        return
    lo = np.min([h.min(0) - 3 * s for h, s, _ in shapes], axis=0)
    hi = np.max([h.max(0) + 3 * s for h, s, _ in shapes], axis=0)
    # Floor coordinates (x, z) to the parent's space; the unused third column keeps the matrix invertible.
    to = r._outer @ np.array([[1.0, 0, 0, 0], [0, 0, -1, floor], [0, 1, 0, 0], [0, 0, 0, 1]])
    box = np.array([[lo[0], lo[1], 0, 1], [hi[0], lo[1], 0, 1], [lo[0], hi[1], 0, 1], [hi[0], hi[1], 0, 1]])
    if ((box @ to.T)[:, 3] < space.NEAR).any():
        return  # the floor passes behind the camera
    clear = 1.0 if fg is None else 1.0 - float(fogged(fg, (box @ to.T)[:, 3].mean()))  # fog thins the shadow
    # Blurred seen from above at a few floor px per pixel, then laid on the floor: a blur drawn under the
    # camera's perspective added 36 ms to a CPU frame of examples/import.json, this adds about 3.
    px = max(1.0, soft / 4)
    w, h = (int(math.ceil(v)) for v in (hi - lo) / px)
    surf = skia.Surface(max(1, w), max(1, h))
    cv = surf.getCanvas()
    cv.scale(1 / px, 1 / px)
    cv.translate(-lo[0], -lo[1])
    for hull, sigma, alpha in shapes:
        path = skia.Path()
        path.addPoly([skia.Point(*q) for q in hull.tolist()], True)
        a = round(255 * min(1.0, alpha) * clear)
        cv.drawPath(path, skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(a, 0, 0, 0),
                                     MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma)))
    c.save()
    c.resetMatrix()
    c.scale(r.scale * r._dev, r.scale * r._dev)
    c.concat(space.to_skia(to))
    c.drawImageRect(surf.makeImageSnapshot(), skia.Rect(*lo, lo[0] + w * px, lo[1] + h * px),
                    skia.SamplingOptions(skia.FilterMode.kLinear))
    c.restore()
    st = r._stage
    if st is not None and st.child.type == "model":
        # In a stage the shadow lies on the floor, a little nearer than a floor layer in the same plane.
        corners = box @ to.T
        st.plane(to, corners[:, :2] / corners[:, 3:] * st.k, bias=1 + 1e-5)


def _fixed(r, L, m, what: str, make):
    """Work on a model's mesh that no frame changes, done once while the layer keeps the same mesh. Merged
    parts do; a primitive whose size animates gets a new mesh each frame, and this is worked out again."""
    hit = r._models.get((what, L.id))
    if hit is None or hit[0] is not m.P or hit[1] is not m.T:
        hit = r._models[(what, L.id)] = (m.P, m.T, make())
    return hit[2]


def _faces(m) -> tuple:
    """Each triangle's unit normal and explode jitter (a fixed per-face hash), and the mesh's extent."""
    P, T = m.P, m.T
    a, b, cc = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, cc - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    jitter = 0.5 + 0.5 * np.modf(np.abs(np.sin(np.arange(len(T)) * 12.9898)) * 43758.5453)[0]
    return fn, jitter, float(np.ptp(P, axis=0).max())


def _compact(T: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The vertices (of n) that triangles T use, in order, and T renumbered to them: what np.unique with
    return_inverse gives, without its sort."""
    keep = np.zeros(n, bool)
    keep[T] = True
    return np.flatnonzero(keep), (np.cumsum(keep) - 1)[T]


def _picked(looks: list, uv: np.ndarray, M: np.ndarray) -> tuple | None:
    """One shader for the pictures looks (one per material) and coordinates uv (each vertex's, in pixels of its
    material's image, M its material) moved to pick them: material k's go (k + 0.5) strides right, a stride being
    a power of two at least twice as wide as any of them and 2 px more. A model whose materials take turns in the
    sort is then one draw, not one per run of a material: a textured glTF can of two materials took 15,000 a frame.
    None for more than 32 materials, or strides too long for float32 to place a coordinate within 1/32 px."""
    n = len(looks)
    stride = 2.0 ** math.ceil(math.log2(2 * float(np.abs(uv[:, 0]).max(initial=0.0)) + 2))
    if n > 32 or n * stride > 2 ** 18:
        return None
    if ("pick", n) not in _EFFECTS:
        _EFFECTS[("pick", n)] = skia.RuntimeEffect.MakeForShader(_PICK.format(
            children="\n".join(f"uniform shader look{i};" for i in range(n)),
            branches="".join(f"    if (k < {i}.5) {{ return look{i}.eval(q); }}\n" for i in range(n - 1)) +
            f"    return look{n - 1}.eval(q);"))
    b = skia.RuntimeShaderBuilder(_EFFECTS[("pick", n)])
    b.setUniform("stride", stride)
    for i, look in enumerate(looks):
        b.setChild(f"look{i}", look)
    return b.makeShader(), uv + np.stack([(M + 0.5) * stride, np.zeros(len(M))], 1)


def _path_bytes(word: int, xy: np.ndarray) -> bytes:
    """Points xy as SkPath::writeToMemory lays out a move and then lines: four int32s (version and fill type,
    then the point, conic weight and verb counts), the points as float32 pairs, a byte per verb, zeros to 4."""
    n = len(xy)
    verbs = np.zeros(n + -n % 4, np.uint8)
    verbs[1:n] = 1  # kMove, then kLine
    return np.array([word, n, 0, n], np.int32).tobytes() + xy.astype(np.float32).tobytes() + verbs.tobytes()


def _path_word() -> int | None:
    """The first word this Skia writes for a path, if _path_bytes lays out a path exactly as Skia does; else None."""
    xy = np.array([[1.5, -2.25], [3.0, 4.0], [1e6, 0.1]])
    p = skia.Path()
    p.moveTo(*xy[0])
    for q in xy[1:]:
        p.lineTo(*q)
    raw = p.writeToMemory()
    word = int(np.frombuffer(raw[:4], np.int32)[0])
    return word if _path_bytes(word, xy) == raw else None


_WORD = _path_word()


def _points(xy: np.ndarray) -> list:
    """skia.Points for the rows of xy, the only form drawVertices takes. Skia makes them while reading a path
    back from bytes numpy writes, in two thirds of the time of a Point.Make call per row, freeing included
    (models-30 makes 110,000 a frame on the GPU). The same float32 values either way; Point.Make if this
    Skia lays paths out differently."""
    if _WORD is not None:
        p, buf = skia.Path(), _path_bytes(_WORD, xy)
        if p.readFromMemory(buf) == len(buf):
            return p.getPoints(len(xy))
    return list(map(skia.Point.Make, xy[:, 0].tolist(), xy[:, 1].tolist()))


def _paint(r, L, mats, lt, more, env, part: str | None = None, edge=None) -> skia.Paint:
    """The GPU's shading paint for these materials, lights and environment, kept while they hold. part: one of
    _effect's parts, for a textured model. edge: the colour of its edge lines, if it has any."""
    glow = any(mt[6] for mt in mats)
    emit = part is None and any(any(mt[7]) for mt in mats)
    clear = part is None and any(mt[8] < 1 for mt in mats)

    def make():
        bld = skia.RuntimeShaderBuilder(_effect(len(mats), len(more), glow, env is not None, part, emit,
                                                edge is not None, clear))
        if edge is not None and part is None:
            bld.setUniform("edgeCol", list(edge))
        for i, mt in enumerate(mats):
            bld.setUniform(f"base{i}", [*mt[0], mt[1]])
            bld.setUniform(f"surf{i}", list(mt[2:6]))
            if glow:
                bld.setUniform(f"glow{i}", mt[6])
            if emit:
                bld.setUniform(f"emit{i}", list(mt[7]))
            if clear:
                bld.setUniform(f"alpha{i}", mt[8])
        for name, v in zip(("keyDir", "fillDir", "keyCol", "amt"), lt):
            bld.setUniform(name, v)
        for j, (d, col) in enumerate(more):
            bld.setUniform(f"dir{j}", d)
            bld.setUniform(f"col{j}", col)
        for name, v in zip(("envTop", "envHor", "envBot", "envDown", "envAmt"), env or ()):
            bld.setUniform(name, v)
        return skia.Paint(Shader=bld.makeShader())
    key = (tuple(mats), tuple(map(tuple, lt)), tuple((tuple(d), tuple(c)) for d, c in more),
           None if env is None else tuple(tuple(v) if isinstance(v, list) else v for v in env),
           None if edge is None else tuple(edge))
    return _kept(r, ("paint", L.id) if part is None else ("paint", L.id, part), key, make)


def _lighting(r, L, m, t: float, T: float) -> tuple:
    """The model's light, extra lights and environment at this frame, as _paint and _shade take them. One set on
    the layer tweens on the layer's time, the scene's on the scene's."""
    def get(key, spec):
        if key in L.raw:
            return at(r._lit, r.s.ctx, spec, t, L.dur, T)
        return at(r._lit, r.s.ctx, spec, T, r.s.duration, T)
    lt = light(get("light", L.raw.get("light", r.s.light)))
    specs = L.raw["lights"] if "lights" in L.raw else r.s.lights
    specs = [get("lights", s) for s in ([specs] if isinstance(specs, dict) else specs or []) if isinstance(s, dict)]
    env = environment(get("env", L.raw.get("env", r.s.env)))
    R = centre = None
    cam = r._cam if r._cam is not None else r._lens
    if lt.get("follow") == "world" or specs or env is not None:
        R = (np.linalg.inv(r._lens) @ cam)[:3, :3]
    if any(s.get("type") == "point" for s in specs):
        mid = _fixed(r, L, m, "centre", lambda: (m.P.min(0) + m.P.max(0)) / 2 if len(m.P) else np.zeros(3))
        world = np.linalg.solve(cam, r._m3d) @ np.append(mid, 1.0)
        centre = world[:3] / (world[3] or 1.0)
    out = _lights(lt, R)
    if out[0][2] < -0.5 and out[3][0] > 0:
        r.unlit.add(L.id)
    return out, _more(specs, R, centre, r.s.w, r.s.h), _env(env, R)


def _vertices(cv: skia.Canvas, q: np.ndarray, T: np.ndarray, uv, colors, paint: skia.Paint, mode):
    """Triangles T of device points q, with texture coordinates uv and colours (either None), drawn with paint,
    mode combining the colours with its shader. Skia indexes vertices with 16 bits, and converts each one it is
    given: draw in sorted chunks, each with only the vertices its triangles use (one chunk uses them all).
    Indices go as a memoryview of uint16, which Skia reads 30% faster than a list and 4 times as fast as a numpy
    array. A few triangles of many vertices take only theirs."""
    for k in range(0, len(T), 20000):
        if len(T) <= 20000 and 3 * len(T) >= len(q):
            sel, idx = slice(None), T
        elif len(T) <= 20000:
            sel = np.unique(T)
            idx = np.searchsorted(sel, T)
        else:
            sel, idx = _compact(T[k:k + 20000], len(q))
        tex = None if uv is None else _points(uv[sel])
        verts = skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, _points(q[sel]), tex,
                                       None if colors is None else colors[sel].tolist(),
                                       memoryview(idx.astype(np.uint16).ravel()))
        cv.drawVertices(verts, paint, mode)


def _unswim(r, L, P, N, M, U, T, texs) -> tuple:
    """The textured triangles of T cut until perspective moves no part of their image more than SWIM px: Skia
    spreads an image evenly across a triangle on screen, where perspective squeezes its far end, so the image of
    a large triangle slid across it as it turned. An edge is cut where its middle is that far from where the image
    belongs (its length on screen times the gap in depth over twice the sum); cuts on shared edges are shared, so
    no crack opens. Returns P, N, M, U and T grown to match; r.swim names the model if some edge is still too
    long after nine rounds (a floor running to just in front of the camera took eight)."""
    A, o, q, c = r._m3d[:2, :3], r._m3d[:2, 3], r._m3d[3, :3], r._m3d[3, 3]
    textured = np.array([tx is not None for tx in texs])

    def long(P, extra, a, b):
        wa, wb = P[a] @ q + c, P[b] @ q + c
        gap = (P[a] @ A.T + o) / wa[:, None] - (P[b] @ A.T + o) / wb[:, None]
        return (np.hypot(gap[:, 0], gap[:, 1]) * r.scale * np.abs(wa - wb) / (2 * (wa + wb)) > SWIM) & \
            textured[extra[1][a]]

    e = T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    if not long(P, (N, M), e[:, 0], e[:, 1]).any():
        return P, N, M, U, T
    P, T, N, M, U = mesh._refine(P, T, 0.0, outline=True, extra=(N, M, U), long=long, rounds=9)
    e = T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    if long(P, (N, M), e[:, 0], e[:, 1]).any():
        r.swim.add(L.id)
    return P, N, M, U, T


def _look(r, L, i: int, fill, tx, slot: str = "look") -> skia.Shader:
    """Material i's picture: its texture tx times its tint over its fill, kept while they hold. A runtime shader:
    skia-python's Shaders.Blend took 1.5 ms with a small image and 99 ms with a 1080p video frame on the GPU."""
    def make():
        if tx is None:
            return skia.Shaders.Color(skia.Color4f(*fill, 1.0))
        img, _, _, tint, tile = tx
        if "look" not in _EFFECTS:
            _EFFECTS["look"] = skia.RuntimeEffect.MakeForShader(_LOOK)
        b = skia.RuntimeShaderBuilder(_EFFECTS["look"])
        b.setChild("image", img.makeShader(TILE[tile[0]], TILE[tile[1]],
                                           skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)))
        b.setUniform("fill", [float(v) for v in fill] + [1.0])
        b.setUniform("tint", [float(v) for v in (tint if tint is not None else (1.0, 1.0, 1.0))])
        return b.makeShader()
    # The image itself in the key, not its id: a video's or group's frame goes once drawn, and a later frame's
    # image could take its id.
    return _kept(r, (slot, L.id, i), (tuple(fill), None if tx is None else (tx[0], tx[3], tx[4])), make)


def _textured(r, L, cv: skia.Canvas, region: skia.Rect, q, T, nv, M, U, mats, texs, glows, lit, fv, fg,
              edge=None) -> bool:
    """A textured model's sorted triangles T on cv (cleared in region): its picture, each material's texture over
    its fill, times twice half the light on a white surface, plus its highlights, rim and reflection, under its
    fog. Each is drawn on its own surface with the same triangles and they are combined per pixel: drawn one over
    another, a triangle behind would be lit again by the one in front. On the GPU the light is per pixel. On the
    CPU it is per vertex, as for any model; the picture is drawn times its light straight onto cv, all at half
    strength, which saved two copies of the region, and True says the model is to be drawn twice as bright.
    The light a model file's materials give off (glows: emission textures) is one more such pass, added. Edge
    lines (in the colour edge, slot len(mats)) are their colour in the picture, lit at 1 and given nothing."""
    lt, more, env = lit

    def pictures(txs, under, slot):
        """Each material's picture and its texture coordinates in pixels of its image."""
        looks, sizes = [], np.zeros((len(mats) + 1, 6))  # repeat, offset and size of each image
        for i, (mt, tx) in enumerate(zip(mats, txs)):
            looks.append(_look(r, L, i, under(mt), tx, slot))
            if tx is not None:
                img, rep, off = tx[:3]
                sizes[i] = (*rep, *off, img.width(), img.height())
        if edge is not None:
            looks.append(skia.Shaders.Color(skia.Color4f(*(edge if slot == "look" else (0.0, 0.0, 0.0)), 1.0)))
        z = sizes[M]
        return looks, (U * z[:, :2] - z[:, 2:4]) * z[:, 4:]

    looks, uv = pictures(texs, lambda mt: mt[0], "look")
    shines = None  # emission: each material's emission texture over its emission colour (black without)
    if any(gl is not None for gl in glows) or any(any(mt[7]) for mt in mats):
        shines = pictures(glows, lambda mt: mt[7], "glow")
    slot = M[T[:, 0]]
    cuts = np.flatnonzero(np.diff(slot)) + 1
    runs = list(zip(np.r_[0, cuts], np.r_[cuts, len(T)]))  # triangles in sort order that share a material

    def apart(k: int, paint, colors=None, coords=None, mode=skia.BlendMode.kDst, by_material=False, looks=looks):
        surf = r._model_spare(k)
        c = surf.getCanvas()
        c.save()
        c.clipRect(region)
        c.clear(skia.ColorTRANSPARENT)
        one = _picked(looks, coords, M) if by_material and len(runs) > 1 else None
        if one is not None:
            _vertices(c, q, T, one[1], colors, skia.Paint(Shader=one[0]), mode)
        for a, b in [] if one is not None else runs if by_material else [(0, len(T))]:
            _vertices(c, q, T[a:b], coords, colors, paint if paint is not None else
                      skia.Paint(Shader=looks[slot[a]]), mode)
        c.restore()
        return surf.makeImageSnapshot()

    gpu, fog = r.backend == "gpu", None
    if fv is not None:  # each vertex's fog as its colour (at half strength on the CPU), the share as alpha
        a = np.round(fv * 255).astype(np.uint32)
        cr, cg, cb = (round(v * (255 if gpu else 127.5)) for v in fg[0])
        fog = apart(3, skia.Paint(BlendMode=skia.BlendMode.kSrc), (a << 24) | (cr << 16) | (cg << 8) | cb)
    cv.save()
    cv.clipRect(region)
    if gpu:
        normal = np.stack([2.0 * M + 0.5 + 0.45 * nv[:, 0], nv[:, 1]], 1)  # as for any model (draw)
        parts = {"base": apart(0, None, coords=uv, by_material=True),
                 "light": apart(1, _paint(r, L, mats, lt, more, env, "light", edge), coords=normal),
                 "add": apart(2, _paint(r, L, mats, lt, more, env, "add", edge), coords=normal)}
        if shines is not None:
            parts["emit"] = apart(4, None, coords=shines[1], by_material=True, looks=shines[0])
        b = skia.RuntimeShaderBuilder(_combined(shines is not None))
        for name, img in parts.items():
            b.setChild(name, img.makeShader(skia.SamplingOptions()))
        b.setChild("fog", fog.makeShader(skia.SamplingOptions()) if fog is not None else
                   skia.Shaders.Color(skia.ColorTRANSPARENT))
        cv.drawPaint(skia.Paint(Shader=b.makeShader(), BlendMode=skia.BlendMode.kSrc))
    else:
        light, add = _shade(nv, M, mats, *lt, more, env, split=True, edge=edge)
        one = _picked(looks, uv, M) if len(runs) > 1 else None
        if one is not None:
            _vertices(cv, q, T, one[1], light, skia.Paint(Shader=one[0]), skia.BlendMode.kModulate)
        for a, b in [] if one is not None else runs:
            _vertices(cv, q, T[a:b], uv, light, skia.Paint(Shader=looks[slot[a]]), skia.BlendMode.kModulate)
        glow = None if shines is None else apart(4, None, coords=shines[1], by_material=True, looks=shines[0])
        for img, mode, cf in ((apart(2, skia.Paint(), add), skia.BlendMode.kPlus, None),
                              (glow, skia.BlendMode.kPlus, _HALF), (fog, skia.BlendMode.kSrcOver, None)):
            if img is not None:
                cv.drawImage(img, 0, 0, skia.SamplingOptions(), skia.Paint(BlendMode=mode, ColorFilter=cf))
    cv.restore()
    return not gpu


def _edging(r, L, t: float, T: float) -> tuple | None:
    """The layer's edge lines at its time t (scene time T): (width in px, colour, cosine of the least crease
    angle drawn, 2 for every edge), or None. "all" draws every edge, true the defaults."""
    v = L.raw.get("edges")
    if v is None or v is False:
        return None
    v = {"angle": 0} if v == "all" else at(r._lit, r.s.ctx, v, t, L.dur, T) if isinstance(v, dict) else {}
    width, angle = float(v.get("width", 2.0)), float(v.get("angle", 30.0))
    if width <= 0:
        return None
    return (width, _rgb(v.get("color", r.s.theme.get("ink", "#0a0c10"))),
            math.cos(math.radians(angle)) if angle > 0 else 2.0)


def _lines(r, L, m, T, ids, front, w, cos: float, exact: bool) -> tuple:
    """The edges of triangles T (the mesh's triangles ids) to draw a line along, where `front` says which face
    the camera: creases whose faces turn more than the angle (cos, its cosine) and the outline, where a face
    toward the camera meets one away from it or none. Returns each line's two vertices, taken from a face toward
    the camera (an exploded face takes its lines with it), and its place in the sort: after every face at either
    end, so a line is hidden only by what is in front of it. exact: by the sort's own key; else by each face's
    nearest corner, for triangles _unswim is to cut."""
    weld, face, first, ends, sides, same = _topology(r, L, m)

    def creases():  # the cosine of the angle between the faces either side of each edge
        a, b, c = (m.P[m.T[first, k]] for k in range(3))
        n = np.cross(b - a, c - a)
        n = np.r_[n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30), np.zeros((1, 3))]
        return (n[sides[:, 0]] * n[sides[:, 1]]).sum(1) * np.where(same, -1.0, 1.0)
    dots = _fixed(r, L, m, "creases", creases)
    rows = np.flatnonzero(front)
    fid = face[ids[rows]]
    seen = np.zeros(len(first) + 1, bool)  # each face toward the camera; the last, past a border, never is
    seen[fid] = True
    seen[-1] = False
    row = np.zeros(len(first) + 1, np.int64)
    row[fid] = rows
    fa, fb = seen[sides[:, 0]], seen[sides[:, 1]]
    e = np.flatnonzero((fa | fb) & ((dots < cos) | (fa != fb)))
    rw = row[np.where(fa[e], sides[e, 0], sides[e, 1])]
    pts = weld[m.T[ids[rw]]]  # the points of each line's face, to find its two ends among its corners
    va = T[rw, np.argmax(pts == ends[e, :1], 1)]
    vb = T[rw, np.argmax(pts == ends[e, 1:], 1)]
    ww = w[T[rows]]
    top = np.full(int(weld.max()) + 1, -np.inf)
    np.maximum.at(top, weld[m.T[ids[rows]]].ravel(), np.repeat(-ww.sum(1) if exact else -3 * ww.min(1), 3))
    return va, vb, np.maximum(top[ends[e, 0]], top[ends[e, 1]])


def wire(xy: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Each edge of triangles T once, as its two ends at points xy: what kine look --wire draws."""
    e = np.sort(T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).astype(np.int64)
    e = np.unique(e[:, 0] * len(xy) + e[:, 1])
    return np.stack([xy[e // len(xy)], xy[e % len(xy)]], 1).astype(np.float32)


def _topology(r, L, m) -> tuple:
    """mesh.edges of the model's mesh, kept while its triangles are (a size that animates keeps them). A model
    file's animation moves points it welds, so a posed mesh's are worked out each time."""
    hit = r._models.get(("edges", L.id))
    if hit is None or hit[0] is not m.T or hit[1] != len(m.P) or (L.raw.get("animation") not in (None, False)
                                                                  and hit[3] is not m.P):
        hit = r._models[("edges", L.id)] = (m.T, len(m.P), mesh.edges(m.P, m.T), m.P)
    return hit[2]


def _quads(a: np.ndarray, b: np.ndarray, half: float, wa, wb) -> tuple[np.ndarray, np.ndarray]:
    """Lines from points a to b, half px either side and past each end (square caps, so lines meeting at a
    corner close it), as four corners each, and each corner's depth (that of the end it is at)."""
    d = b - a
    ln = np.hypot(d[:, 0], d[:, 1])[:, None]
    d = np.where(ln > 1e-9, d / np.maximum(ln, 1e-9), [1.0, 0.0]) * half  # an edge end on: a square
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    xy = np.stack([a - d + n, b + d + n, b + d - n, a - d - n], 1).reshape(-1, 2)
    return xy, np.stack([wa, wb, wb, wa], 1).ravel()


def _combined(emit: bool = False) -> skia.RuntimeEffect:
    key = "combine emit" if emit else "combine"
    if key not in _EFFECTS:
        _EFFECTS[key] = skia.RuntimeEffect.MakeForShader(_COMBINE_EMIT if emit else _COMBINE)
    return _EFFECTS[key]


def draw(r, c: skia.Canvas, L, g, reveal: float, t: float = 0.0, scene_t: float = 0.0):
    """Draw model layer L on canvas c at its time t (scene_t on the scene's clock). Returns its projected bounds
    in device pixels, or None."""
    m, mats, texs, glows = model(r, L, g, t, scene_t)
    lt, more, env = _lighting(r, L, m, t, scene_t)
    fg = fog(r, scene_t) if r._stage is None else None  # a stage fogs each pixel it shows
    P, N, T, M = m.P, m.N, m.T, m.M
    U = None  # texture coordinates, when any material has a texture (a file without them: check says no-uvs)
    if any(tx is not None for tx in texs + glows):
        U = m.U if m.U is not None else np.zeros((len(P), 2))
    amt = max(0.0, min(1.0, float(g("draw")) * reveal))
    tid = slice(None)  # each triangle's index in the whole mesh
    if amt < 1.0:
        tid = _fixed(r, L, m, "order", lambda: mesh.build_order(m))[:math.ceil(amt * len(T))]
        T = T[tid]
    edging = _edging(r, L, t, scene_t)
    ids = np.arange(len(T)) if isinstance(tid, slice) else tid  # the same, kept through the culls for the edges
    ex = float(g("explode", 0.0))
    if ex > 0 and len(T):
        # Each face flies out along its own normal, some further than others (a fixed per-face hash).
        fn, jitter, extent = _fixed(r, L, m, "explode", lambda: _faces(m))
        reach = ex * extent * 0.6 * jitter[tid]
        P = (P[T] + (fn[tid] * reach[:, None])[:, None, :]).reshape(-1, 3)
        N, M = N[T].reshape(-1, 3), M[T].reshape(-1)
        U = None if U is None else U[T].reshape(-1, 2)
        T = np.arange(len(P), dtype=np.int32).reshape(-1, 3)
    foot = P if amt >= 1.0 or ex > 0 else P[T.ravel()]  # what casts the ground shadow: what is drawn
    view = np.linalg.inv(r._lens) @ r._m3d  # the layer matrix without the lens: into camera space
    eye = np.linalg.solve(view, [0.0, 0.0, -1.0 / r._lens[3, 2], 1.0])[:3]  # the camera, in the model's space
    if len(T) and (eye > P.min(0)).all() and (eye < P.max(0)).all() and _within(eye, P, T):
        r.inside.add(L.id)
    if r._stage is not None and r._stage.casters is not None:  # every triangle drawn casts, facing the camera or not
        r._stage.cast(P @ view[:3, :3].T + view[:3, 3], T)
    hom = P @ r._m3d[:3, :3].T + r._m3d[:3, 3]
    w = P @ r._m3d[3, :3] + r._m3d[3, 3]
    if r._focused is not None and len(w):
        r._reach(w[T.ravel()] if len(T) else w)
    if len(w) and w.min() < space.NEAR:  # no vertex that near, no triangle crossing
        near = (w[T] < space.NEAR).any(1)
        if near.any():
            r.culled.add(L.id)
            T, ids = T[~near], ids[~near]
    d = np.where(np.abs(w) < 1e-9, 1e-9, w)
    X, Y = hom[:, 0] / d, hom[:, 1] / d
    x, y = X[T], Y[T]  # each triangle's corners
    front = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0]) > 0
    lines = None if edging is None or not front.any() else _lines(r, L, m, T, ids, front, w, edging[2], U is None)
    clear = np.array([mt[8] < 1 for mt in mats]) if U is None else np.zeros(len(mats), bool)
    if clear.any():
        # A see-through face turned away shows through the near ones: drawn too, lit from inside, unless it is the
        # back of a face drawn back to back (a plane), which would show twice. A copy of its corners with the
        # normals turned round, so the near faces sharing them keep theirs.
        face = _topology(r, L, m)[1]
        seen = np.zeros(int(face.max()) + 2, bool)
        seen[face[ids[front]]] = True
        back = np.flatnonzero(~front & clear[M[T[:, 0]]] & ~seen[face[ids]])
        idx, n = T[back].ravel(), len(P)
        N, M = np.r_[N, -N[idx]], np.r_[M, M[idx]]
        X, Y, w = np.r_[X, X[idx]], np.r_[Y, Y[idx]], np.r_[w, w[idx]]
        T = np.r_[T[front], (n + np.arange(len(idx), dtype=T.dtype)).reshape(-1, 3)]
    else:
        T = T[front]
    if not len(T):
        return None
    if U is not None:
        P, N, M, U, T = _unswim(r, L, P, N, M, U, T, [tx if tx is not None else gl for tx, gl in zip(texs, glows)])
        hom = P @ r._m3d[:3, :3].T + r._m3d[:3, 3]
        w = P @ r._m3d[3, :3] + r._m3d[3, 3]
        d = np.where(np.abs(w) < 1e-9, 1e-9, w)
        X, Y = hom[:, 0] / d, hom[:, 1] / d
    key = -w[T].sum(1)
    order = np.argsort(key, kind="stable")
    T, key = T[order], key[order]
    r.tris += len(T)
    # From here on, only the vertices of the triangles drawn: about half of them face away.
    used, T = _compact(T, len(X))
    N, M = N[used], M[used]
    U = None if U is None else U[used]
    nv = N @ np.linalg.inv(view[:3, :3])  # normals into camera space
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    edge, wv, faces = None, w[used], T
    if lines is not None and len(lines[0]):
        # Each line a quad of four vertices of its own, in the sort after every face at its ends.
        va, vb, after = lines
        n0, k = len(used), len(va)
        xy, wq = _quads(np.stack([X[va], Y[va]], 1), np.stack([X[vb], Y[vb]], 1), edging[0] / 2, w[va], w[vb])
        X, Y, wv = np.r_[X[used], xy[:, 0]], np.r_[Y[used], xy[:, 1]], np.r_[wv, wq]
        nv = np.r_[nv, np.tile([0.0, 0.0, 1.0], (4 * k, 1))]
        M = np.r_[M, np.full(4 * k, len(mats), M.dtype)]
        U = None if U is None else np.r_[U, np.zeros((4 * k, 2))]
        quads = (n0 + 4 * np.arange(k))[:, None, None] + np.array([[0, 1, 2], [0, 2, 3]])
        T = np.insert(T, np.repeat(np.searchsorted(key, after, side="right"), 2), quads.reshape(-1, 3), axis=0)
        edge = edging[1]
    else:
        X, Y = X[used], Y[used]
    X, Y = X * r.scale, Y * r.scale  # device pixels
    if r.wire and not r._insets:
        r.wires.append(wire(np.stack([X, Y], 1), faces))
    lo, hi = (X.min(), Y.min()), (X.max(), Y.max())
    bounds = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
    x0, y0 = max(0, math.floor(lo[0]) - 2), max(0, math.floor(lo[1]) - 2)
    x1, y1 = min(r.w, math.ceil(hi[0]) + 2), min(r.h, math.ceil(hi[1]) + 2)
    if x1 <= x0 or y1 <= y0:
        return bounds
    surf, ss = r._model_surface()
    region = skia.Rect(x0 * ss, y0 * ss, x1 * ss, y1 * ss)
    cv = surf.getCanvas()
    cv.save()
    cv.clipRect(region)
    cv.clear(skia.ColorTRANSPARENT)
    cv.restore()
    q = np.stack([X, Y], 1) * ss
    fv = None if fg is None else fogged(fg, wv)
    if fv is not None and fv.min() >= 0.999:
        r.fogged.add(L.id)
    if fv is not None and fv.max() <= 0.002:
        fv = None
    mode = skia.BlendMode.kDst  # vertex colours as given, when there are any
    halved = False  # drawn at half strength (_textured)
    if U is not None:
        halved = _textured(r, L, cv, region, q, T, nv, M, U, mats, texs, glows, (lt, more, env), fv, fg, edge)
    elif r.backend == "gpu":
        paint, colors = _paint(r, L, mats, lt, more, env, edge=edge), None
        uv = np.stack([2.0 * M + 0.5 + 0.45 * nv[:, 0], nv[:, 1]], 1)  # the normal, with the material slot packed in
        if fv is not None:
            # Each vertex's fog as its colour (fog colour, alpha the share); kDstATop gives that plus the shading
            # times what the fog leaves.
            a = np.round(fv * 255).astype(np.uint32)
            cr, cg, cb = (round(v * 255) for v in fg[0])
            colors, mode = (a << 24) | (cr << 16) | (cg << 8) | cb, skia.BlendMode.kDstATop
    else:
        # The raster backend runs SkSL slowly: examples/model.json spent 275 ms a frame in drawVertices
        # at 2x supersampling. Shading each vertex in numpy, the whole frame takes 72 ms.
        uv = None
        paint, colors = skia.Paint(), _shade(nv, M, mats, *lt, more, env, None if fv is None else (fv, fg[0]),
                                             edge=edge)
    if U is None:
        _vertices(cv, q, T, uv, colors, paint, mode)
    img = surf.makeImageSnapshot()
    if L.raw.get("ground") not in (None, False):
        _ground(r, c, L.raw["ground"], m.P, foot, fg, lt[0])
    if r._stage is not None and r._stage.child is L:
        r._stage.depth(q, 1.0 / wv, T)
    c.save()
    c.resetMatrix()
    c.scale(r._dev, r._dev)
    sampling = skia.SamplingOptions(skia.FilterMode.kLinear) if ss > r._dev else skia.SamplingOptions()
    c.drawImageRect(img, region, skia.Rect(x0, y0, x1, y1), sampling, skia.Paint(ColorFilter=_DOUBLE) if halved
                    else None)
    c.restore()
    del img  # a live snapshot makes the next draw into the surface copy it first
    return bounds
