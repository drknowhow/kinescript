"""Model layers, drawn each frame by numpy and one drawVertices call: transform, drop triangles that
face away or cross the camera plane, sort the rest far to near (painter's algorithm), and shade per
pixel in SkSL. The view-space normal rides in each vertex's texture coordinate with the material
index packed into it, so every part of a model draws in one call and hides the others correctly."""
from __future__ import annotations

import math

import numpy as np
import skia

from . import color as C
from . import mesh, space

MATERIAL = ("fill", "shine", "metal", "rim", "bands")
MAX_MATERIALS = 8
# Lights are fixed to the camera, like a studio key light on the camera rig: turning a model
# changes its shading, orbiting the camera around it does not.
LIGHTS = {"studio": {"key": 0.85, "fill": 0.25, "ambient": 0.22, "rim": 0.35, "from": "topleft"},
          "soft": {"key": 0.55, "fill": 0.4, "ambient": 0.38, "rim": 0.2, "from": "top"},
          "dramatic": {"key": 1.05, "fill": 0.06, "ambient": 0.06, "rim": 0.75, "from": "left"},
          "flat": {"key": 0.0, "fill": 0.0, "ambient": 1.0, "rim": 0.0, "from": "front"}}
FROM = {"topleft": (-0.5, -0.6, 0.62), "top": (0.0, -0.75, 0.66), "topright": (0.5, -0.6, 0.62),
        "left": (-0.85, -0.25, 0.46), "right": (0.85, -0.25, 0.46), "front": (0.0, 0.0, 1.0),
        "bottomleft": (-0.5, 0.6, 0.62), "bottom": (0.0, 0.75, 0.66), "bottomright": (0.5, 0.6, 0.62)}
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
    half3 spec = mix(half3(1.0), b.rgb, p.z) * s * amt.x * keyCol;
    half rim = pow(1.0 - n.z, 3.0) * p.w * amt.w;
    return half4(saturate(diffuse + spec + rim * keyCol), 1.0);
}
"""
_EFFECTS: dict[int, skia.RuntimeEffect] = {}


def _effect(n: int) -> skia.RuntimeEffect:
    """The shading effect for n materials, one pair of uniforms each (base colour + bands, then
    specular amount, exponent, metal and rim)."""
    if n not in _EFFECTS:
        decl = "\n".join(f"uniform half4 base{i};\nuniform half4 surf{i};" for i in range(n))
        pick = " ".join(f"if (m == {i}.0) {{ b = base{i}; p = surf{i}; }}" for i in range(1, n))
        _EFFECTS[n] = skia.RuntimeEffect.MakeForShader(_SKSL.replace("DECLARE", decl).replace("LOOKUP", pick))
    return _EFFECTS[n]


def light(spec) -> dict:
    """A light preset name or {key, fill, ambient, rim, from, color} on top of the studio preset."""
    if isinstance(spec, dict):
        out = dict(LIGHTS["studio"], color="#fff4e6", **LIGHTS.get(spec.get("preset"), {}))
        out.update({k: v for k, v in spec.items() if k != "preset"})
        return out
    return dict(LIGHTS.get(spec or "studio", LIGHTS["studio"]), color="#fff4e6")


def _unit(v) -> list[float]:
    v = np.asarray(v, float)
    return (v / (np.linalg.norm(v) or 1.0)).tolist()


def _lights(lt: dict) -> tuple:
    """Key and fill directions in camera space, key colour, and the four amounts, as the shader takes them."""
    key = _unit(FROM.get(lt.get("from"), FROM["topleft"]))
    return (key, _unit((-key[0], 0.35, 0.8)), list(C.parse(lt.get("color", "#fff4e6"))[:3]),
            [float(lt.get(k, 0)) for k in ("key", "fill", "ambient", "rim")])


def _shade(nv, M, mats, key, fill, col, amt) -> np.ndarray:
    """The shader's lighting, per vertex, as 0xAARRGGBB: the raster backend's path."""
    base = np.array([(*m[0], m[1]) for m in mats])[M]
    sp, ex, metal, rim = np.array([m[2:] for m in mats])[M].T
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
    rgb = base[:, :3] * light * (1 - 0.75 * metal)[:, None]
    rgb += (1 - metal[:, None] + metal[:, None] * base[:, :3]) * (s * amt[0])[:, None] * col
    rgb += ((1 - n[:, 2]) ** 3 * rim * amt[3])[:, None] * col
    c = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint32)
    return 0xFF000000 | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]


def _material(get) -> tuple:
    col = get("fill")
    col = C.parse(col) if isinstance(col, str) else col
    shine, metal = float(get("shine", 0.4)), float(get("metal", 0.0))
    return (tuple(col[:3]), float(get("bands", 0)), 0.6 * shine + 0.4 * metal, 6 + 90 * shine * shine, metal,
            float(get("rim", 0.5)))


def _parts(r, L):
    """The model's parts merged once: mesh, and for each material slot the part keys that override the layer."""
    hit = r._models.get(L.id)
    if hit is None:
        detail = L.raw.get("detail", "medium")
        meshes, slots, keys = [], [], [{}]
        for p in L.raw.get("parts") or []:
            shape = p.get("shape", "cube")
            s3, tube = mesh.size(shape, lambda k, d: p.get(k, d))
            own = {k: p[k] for k in MATERIAL if k in p}
            if own not in keys:
                keys.append(own)
            meshes.append(mesh.placed(mesh.unit(shape, detail, tube), s3, mesh.part_matrix(p)))
            slots.append(keys.index(own))
        if len(keys) > MAX_MATERIALS:
            raise ValueError(f"layer '{L.id}': {len(keys)} different materials; a model takes at most {MAX_MATERIALS}")
        hit = r._models[L.id] = (mesh.merge(meshes, slots) if meshes else None, keys)
    return hit


def model(r, L, g) -> tuple[mesh.Mesh, list[tuple]]:
    """The model's mesh in its own space at this frame, and its materials."""
    parts, keys = _parts(r, L)
    shape = L.raw.get("shape")
    meshes = [] if parts is None else [parts]
    if shape:
        s3, tube = mesh.size(shape, g)
        meshes.insert(0, mesh.placed(mesh.unit(shape, L.raw.get("detail", "medium"), tube), s3))
    m = meshes[0] if len(meshes) == 1 else mesh.cat(meshes)
    accent = r.s.theme.get("accent", "#ff633f")
    root = lambda k, d=None: g(k, accent if k == "fill" and d is None else d)  # noqa: E731
    mats = [_material(lambda k, d=None, o=o: o.get(k, root(k, d))) for o in keys]
    return m, mats


def draw(r, c: skia.Canvas, L, g, reveal: float):
    """Draw model layer L on canvas c. Returns its projected bounds in device pixels, or None."""
    m, mats = model(r, L, g)
    P, N, T, M = m
    amt = max(0.0, min(1.0, float(g("draw")) * reveal))
    tid = np.arange(len(T))  # each triangle's index in the whole mesh
    if amt < 1.0:
        tid = mesh.build_order(m)[:math.ceil(amt * len(T))]
        T = T[tid]
    ex = float(g("explode", 0.0))
    if ex > 0 and len(T):
        # Each face flies out along its own normal, some further than others (a fixed per-face hash).
        a, b, cc = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        fn = np.cross(b - a, cc - a)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True)
        jitter = 0.5 + 0.5 * np.modf(np.abs(np.sin(tid * 12.9898)) * 43758.5453)[0]
        reach = ex * float(np.ptp(P, axis=0).max()) * 0.6 * jitter
        P = (P[T] + (fn * reach[:, None])[:, None, :]).reshape(-1, 3)
        N, M = N[T].reshape(-1, 3), M[T].reshape(-1)
        T = np.arange(len(P), dtype=np.int32).reshape(-1, 3)
    hom = P @ r._m3d[:3, :3].T + r._m3d[:3, 3]
    w = P @ r._m3d[3, :3] + r._m3d[3, 3]
    near = (w[T] < space.NEAR).any(1)
    if near.any():
        r.culled.add(L.id)
        T = T[~near]
    xy = hom[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)[:, None]
    a, b, cc = xy[T[:, 0]], xy[T[:, 1]], xy[T[:, 2]]
    T = T[(b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0]) > 0]
    if not len(T):
        return None
    T = T[np.argsort(-w[T].sum(1), kind="stable")]
    r.tris += len(T)
    # Normals into camera space: the layer matrix without the lens.
    view = np.linalg.inv(r._lens) @ r._m3d
    nv = N @ np.linalg.inv(view[:3, :3])
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    dev = xy * r.scale
    lo, hi = dev[T.ravel()].min(0), dev[T.ravel()].max(0)
    bounds = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
    x0, y0 = max(0, math.floor(lo[0]) - 2), max(0, math.floor(lo[1]) - 2)
    x1, y1 = min(r.w, math.ceil(hi[0]) + 2), min(r.h, math.ceil(hi[1]) + 2)
    if x1 <= x0 or y1 <= y0:
        return bounds
    surf, ss = r._model_surface(x1 - x0, y1 - y0)
    region = skia.Rect(0, 0, (x1 - x0) * ss, (y1 - y0) * ss)
    cv = surf.getCanvas()
    cv.save()
    cv.clipRect(region)
    cv.clear(skia.ColorTRANSPARENT)
    cv.restore()
    q = (dev - (x0, y0)) * ss
    lt = _lights(light(L.raw.get("light", r.s.light)))
    if r.backend == "gpu":
        bld = skia.RuntimeShaderBuilder(_effect(len(mats)))
        for i, mt in enumerate(mats):
            bld.setUniform(f"base{i}", [*mt[0], mt[1]])
            bld.setUniform(f"surf{i}", list(mt[2:]))
        for name, v in zip(("keyDir", "fillDir", "keyCol", "amt"), lt):
            bld.setUniform(name, v)
        paint, colors = skia.Paint(Shader=bld.makeShader()), None
        u, v = 2.0 * M + 0.5 + 0.45 * nv[:, 0], nv[:, 1]  # the normal, with the material slot packed in
    else:
        # The raster backend runs SkSL slowly: examples/model.json spent 275 ms a frame in drawVertices
        # at 2x supersampling. Shading each vertex in numpy, the whole frame takes 72 ms.
        paint, colors = skia.Paint(), _shade(nv, M, mats, *lt)
    # Skia indexes vertices with 16 bits, and converts each one it is given (about 0.3 us): draw
    # in sorted chunks, each with only the vertices its triangles use.
    for k in range(0, len(T), 20000):
        used, idx = np.unique(T[k:k + 20000].ravel(), return_inverse=True)
        pos = list(map(skia.Point, q[used, 0].tolist(), q[used, 1].tolist()))
        tex = None if colors is not None else list(map(skia.Point, u[used].tolist(), v[used].tolist()))
        verts = skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, pos, tex,
                                       None if colors is None else colors[used].tolist(), idx)
        cv.drawVertices(verts, paint, skia.BlendMode.kDst)  # kDst: vertex colours as given, when there are any
    img = surf.makeImageSnapshot()
    c.save()
    c.resetMatrix()
    sampling = skia.SamplingOptions(skia.FilterMode.kLinear) if ss > 1 else skia.SamplingOptions()
    c.drawImageRect(img, region, skia.Rect(x0, y0, x1, y1), sampling)
    c.restore()
    del img  # a live snapshot makes the next draw into the surface copy it first
    return bounds
