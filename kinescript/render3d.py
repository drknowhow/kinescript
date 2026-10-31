"""Model layers, drawn each frame by numpy and one drawVertices call: transform, drop triangles that
face away or cross the camera plane, sort the rest far to near (painter's algorithm), and shade per
pixel in SkSL. The view-space normal rides in each vertex's texture coordinate with the material
index packed into it, so every part of a model draws in one call and hides the others correctly."""
from __future__ import annotations

import math

import numpy as np
import skia

from . import color as C
from . import mesh, models, space, svgpath

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


def _shape(r, shape: str, num, raw, detail: str) -> mesh.Mesh:
    """One shape's mesh at its size, centred on the origin. num(key, default) reads its numbers, raw its text."""
    if shape in mesh.EXTRUDED:
        return _extruded(r, shape, num, raw, detail)
    s3, tube = mesh.size(shape, num)
    return mesh.placed(mesh.unit(shape, detail, tube), s3)


def _parts(r, L):
    """The model's parts merged once: mesh, and for each material slot the part keys that override the layer."""
    hit = r._models.get(L.id)
    if hit is None:
        detail = L.raw.get("detail", "medium")
        meshes, slots, keys = [], [], [{}]
        for p in L.raw.get("parts") or []:
            own = {k: p[k] for k in MATERIAL if k in p}
            if own not in keys:
                keys.append(own)
            m = _shape(r, p.get("shape", "cube"), lambda k, d, p=p: p.get(k, d), p, detail)
            meshes.append(mesh.placed(m, (1.0, 1.0, 1.0), mesh.part_matrix(p)))
            slots.append(keys.index(own))
        if len(keys) > MAX_MATERIALS:
            raise ValueError(f"layer '{L.id}': {len(keys)} different materials; a model takes at most {MAX_MATERIALS}")
        hit = r._models[L.id] = (mesh.merge(meshes, slots) if meshes else None, keys)
    return hit


def _imported(r, L, taken: int):
    """The layer's model file: its mesh with its materials in the slots after the first `taken`, those
    materials less the keys the layer sets itself (a layer fill recolours the whole model), and its size."""
    hit = r._models.get(("src", L.id))
    if hit is None:
        imp = models.load(L.src)
        M, mats = models.slots(imp, MAX_MATERIALS - taken)
        own = [{k: v for k, v in m.items() if k not in L.raw} for m in mats]
        m = imp.mesh._replace(M=np.where(M < 0, 0, M + taken).astype(np.int32))
        hit = r._models[("src", L.id)] = (m, own, imp.size)
    return hit


def _fit(g, size) -> float:
    """Scale of a model file: it fits inside w and h keeping its proportions, or its longest side is 300 px."""
    s = [float(v) / e for v, e in ((g("w", None), size[0]), (g("h", None), size[1])) if v is not None and e > 1e-9]
    return min(s) if s else 300.0 / max(float(size.max()), 1e-9)


def model(r, L, g) -> tuple[mesh.Mesh, list[tuple]]:
    """The model's mesh in its own space at this frame, and its materials."""
    parts, keys = _parts(r, L)
    shape = L.raw.get("shape")
    meshes = [] if parts is None else [parts]
    if L.src:
        m, own, size = _imported(r, L, len(keys))
        keys = keys + own
        meshes.insert(0, m._replace(P=m.P * _fit(g, size)))
    elif shape:
        meshes.insert(0, _shape(r, shape, g, L.raw, L.raw.get("detail", "medium")))
    m = meshes[0] if len(meshes) == 1 else mesh.cat(meshes)
    accent = r.s.theme.get("accent", "#ff633f")
    root = lambda k, d=None: g(k, accent if k == "fill" and d is None else d)  # noqa: E731
    mats = [_material(lambda k, d=None, o=o: o.get(k, root(k, d))) for o in keys]
    return m, mats


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


def _ground(r, c: skia.Canvas, spec, rest: np.ndarray, pts: np.ndarray):
    """A soft shadow under the model, as if lit from straight above: the footprint of its points on a level
    floor, darkest where it touches and wider and fainter as it rises. The floor lies in the layer's parent
    space, at the model's lowest point before it turns (`rest`), unless `y` places it."""
    spec = spec if isinstance(spec, dict) else {}
    local = np.linalg.solve(r._outer, r._m3d)  # the layer's own transform
    Q = pts @ local[:3, :3].T + local[:3, 3]
    low = float(Q[:, 1].max())  # y runs down
    floor = float(spec["y"]) if "y" in spec else local[1, 3] + rest[:, 1].max() * np.linalg.norm(local[:3, 1])
    lift = max(0.0, floor - low)
    xz = Q[:, [0, 2]]
    span = float(np.ptp(xz, axis=0).max()) or 1.0
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
        cv.drawPath(path, skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(round(255 * min(1.0, alpha)), 0, 0, 0),
                                     MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma)))
    c.save()
    c.resetMatrix()
    c.scale(r.scale, r.scale)
    c.concat(space.to_skia(to))
    c.drawImageRect(surf.makeImageSnapshot(), skia.Rect(*lo, lo[0] + w * px, lo[1] + h * px),
                    skia.SamplingOptions(skia.FilterMode.kLinear))
    c.restore()


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


def draw(r, c: skia.Canvas, L, g, reveal: float):
    """Draw model layer L on canvas c. Returns its projected bounds in device pixels, or None."""
    m, mats = model(r, L, g)
    P, N, T, M = m.P, m.N, m.T, m.M
    amt = max(0.0, min(1.0, float(g("draw")) * reveal))
    tid = np.arange(len(T))  # each triangle's index in the whole mesh
    if amt < 1.0:
        tid = _fixed(r, L, m, "order", lambda: mesh.build_order(m))[:math.ceil(amt * len(T))]
        T = T[tid]
    ex = float(g("explode", 0.0))
    if ex > 0 and len(T):
        # Each face flies out along its own normal, some further than others (a fixed per-face hash).
        fn, jitter, extent = _fixed(r, L, m, "explode", lambda: _faces(m))
        reach = ex * extent * 0.6 * jitter[tid]
        P = (P[T] + (fn[tid] * reach[:, None])[:, None, :]).reshape(-1, 3)
        N, M = N[T].reshape(-1, 3), M[T].reshape(-1)
        T = np.arange(len(P), dtype=np.int32).reshape(-1, 3)
    foot = P if amt >= 1.0 or ex > 0 else P[T.ravel()]  # what casts the ground shadow: what is drawn
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
    # From here on, only the vertices of the triangles drawn: about half of them face away.
    used, T = _compact(T, len(P))
    xy, N, M = xy[used], N[used], M[used]
    # Normals into camera space: the layer matrix without the lens.
    view = np.linalg.inv(r._lens) @ r._m3d
    nv = N @ np.linalg.inv(view[:3, :3])
    nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    dev = xy * r.scale
    lo, hi = dev.min(0), dev.max(0)
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
    q = dev * ss
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
    # in sorted chunks, each with only the vertices its triangles use. Indices go as a list, which
    # Skia reads 4 times as fast as a numpy array.
    for k in range(0, len(T), 20000):
        sel, idx = _compact(T[k:k + 20000], len(q))
        pos = list(map(skia.Point.Make, q[sel, 0].tolist(), q[sel, 1].tolist()))  # Make: 13% faster
        tex = None if colors is not None else list(map(skia.Point.Make, u[sel].tolist(), v[sel].tolist()))
        verts = skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, pos, tex,
                                       None if colors is None else colors[sel].tolist(), idx.ravel().tolist())
        cv.drawVertices(verts, paint, skia.BlendMode.kDst)  # kDst: vertex colours as given, when there are any
    img = surf.makeImageSnapshot()
    if L.raw.get("ground") not in (None, False):
        _ground(r, c, L.raw["ground"], m.P, foot)
    c.save()
    c.resetMatrix()
    sampling = skia.SamplingOptions(skia.FilterMode.kLinear) if ss > 1 else skia.SamplingOptions()
    c.drawImageRect(img, region, skia.Rect(x0, y0, x1, y1), sampling)
    c.restore()
    del img  # a live snapshot makes the next draw into the surface copy it first
    return bounds
