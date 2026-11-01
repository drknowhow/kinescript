"""Imported models: glTF 2.0 (.glb, .gltf) and Wavefront .obj files read with numpy into a Mesh and its
materials. Node transforms, skins and morph target weights apply; a glTF file's animations are kept with it, and
pose() moves the mesh to a time of one of them. A glTF base colour texture or an .obj map_Kd is drawn as a
texture, with its mean colour as the material's fill for when it is not. Files are y up; the mesh comes out y down
like space.py."""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

import numpy as np
import skia

from .mesh import Mesh, _refine

EXTENSIONS = (".glb", ".gltf", ".obj")
_TYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_COMPRESSED = ("KHR_draco_mesh_compression", "EXT_meshopt_compression")
_WRAP = {33071: "clamp", 33648: "mirror", 10497: "repeat"}  # glTF sampler wrapS and wrapT
_PATHS = ("translation", "rotation", "scale")  # what an animation channel sets on a node, besides weights
EDGE = 1 / 24  # longest triangle edge kept, as a share of the model's longest side
_CACHE: dict = {}


class Imported(NamedTuple):
    mesh: Mesh  # centred on its bounding box, in file units; M indexes `materials`, -1 takes the layer's own
    materials: list  # {"fill": (r, g, b), "shine": s, "metal": m}, and "texture": {"image", "tint", "tile"}
    size: np.ndarray  # (3,) bounding box
    motion: Motion | None = None  # a glTF file's animations and what posing the mesh again takes
    about: dict | None = None  # what kine probe lists: animations, skins, morph targets, textures, UV sets


class Place(NamedTuple):
    """One glTF primitive as its node draws it, kept to pose it again."""
    P: np.ndarray  # positions in the mesh's own space
    N: np.ndarray | None  # normals, or None for flat faces worked out from the posed triangles
    dP: np.ndarray | None  # (targets, n, 3) morph target offsets of positions
    dN: np.ndarray | None  # and of normals
    joints: np.ndarray | None  # (n, 4) joints of its skin, None for a primitive its node carries
    weights: np.ndarray | None  # (n, 4) their weights, adding up to 1
    T: np.ndarray  # its triangles, wound as drawn
    double: bool  # double-sided: drawn again facing the other way
    node: int | None  # the node it hangs from (None in a file without nodes)
    skin: int | None
    w0: np.ndarray  # morph target weights when no animation sets them


class Motion(NamedTuple):
    """A glTF file's node tree and animations, and how its mesh was cut, centred and cleaned, so pose() can
    build the same mesh from the file's primitives moved."""
    parent: list  # each node's parent, -1 for a root
    order: list  # the nodes, each after its parent
    local: list  # each node's matrix when nothing animates it
    trs: list  # each node's [translation, rotation, scale], None for a node given as a matrix
    skins: list  # (joint nodes, inverse bind matrices (J, 4, 4))
    places: list  # one Place per primitive drawn, in the order the mesh holds them
    clips: list  # (name or None, first key time, duration, [(node, path, interpolation, times, values)])
    used: np.ndarray  # the vertices the mesh kept
    trail: list  # the edges _refine cut, round by round
    centre: np.ndarray  # the middle of the bounding box, taken off the points


def load(path: str) -> Imported:
    """The model in `path`, parsed once per file version (path, modification time, size)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit is None:
        ext = p.suffix.lower()
        try:
            if ext not in EXTENSIONS:
                raise ValueError(f"models are read from {', '.join(EXTENSIONS)} files")
            hit = _finish(*(_obj(p) if ext == ".obj" else _gltf(p)))
        except Exception as e:  # noqa: BLE001 - a malformed file fails in many ways; all of them name the file
            why = e if isinstance(e, ValueError) else f"could not be read ({type(e).__name__}: {e})"
            raise ValueError(f"{p.name}: {why}") from e
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = hit
    return hit


def slots(imp: Imported, n: int) -> tuple[np.ndarray, list]:
    """Material of each vertex and the materials, at most n of them. When a file has more, the ones on the
    most vertices stay and every other one takes the kept colour nearest its own."""
    M, mats = imp.mesh.M, imp.materials
    if len(mats) <= n:
        return M, mats
    if n <= 0:
        return np.full_like(M, -1), []
    keep = np.argsort(-np.bincount(M[M >= 0], minlength=len(mats)), kind="stable")[:n]
    cols = np.array([m["fill"] for m in mats])
    near = keep[np.argmin(((cols[:, None] - cols[keep][None]) ** 2).sum(-1), axis=1)]
    near[keep] = keep
    lut = np.full(len(mats), -1)
    lut[keep] = np.arange(n)
    return np.where(M >= 0, lut[near][np.maximum(M, 0)], -1), [mats[k] for k in keep]


def _flat(P: np.ndarray, T: np.ndarray, U: np.ndarray | None = None) -> tuple:
    """Each triangle its own three vertices with its face normal (files without normals)."""
    P = P[T].reshape(-1, 3)
    fn = np.cross(P[1::3] - P[0::3], P[2::3] - P[0::3])
    return P, np.repeat(fn, 3, axis=0), np.arange(len(P)).reshape(-1, 3), None if U is None else U[T].reshape(-1, 2)


def _finish(pieces: list, mats: list, rig: dict | None = None, about: dict | None = None) -> Imported:
    """One mesh from (P, N, T, M, U) pieces in file space (U None without texture coordinates): turned y down,
    cleaned of degenerate triangles and unused vertices, centred, and with repeated materials merged. `rig`, a
    glTF file's node tree and clips, becomes the Motion that poses it again."""
    if not pieces:
        raise ValueError("no triangle meshes found")
    base = np.cumsum([0] + [len(p[0]) for p in pieces[:-1]])
    P = np.concatenate([p[0] for p in pieces]) * (1.0, -1.0, 1.0)
    N = np.concatenate([p[1] for p in pieces]) * (1.0, -1.0, 1.0)
    T = np.concatenate([p[2] + b for p, b in zip(pieces, base)])[:, [0, 2, 1]]  # y flipped: wind back
    M = np.concatenate([p[3] for p in pieces])
    U = None
    if any(p[4] is not None for p in pieces):
        U = np.concatenate([np.zeros((len(p[0]), 2)) if p[4] is None else p[4] for p in pieces])
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    T = T[area > 1e-12 * max(float(np.ptp(P, axis=0).max()), 1e-9) ** 2]
    if not len(T):
        raise ValueError("no triangle meshes found")
    used, inv = np.unique(T, return_inverse=True)
    P, N, M, T = P[used], N[used], M[used], inv.reshape(-1, 3)
    extra = (N, M) if U is None else (N, M, U[used])
    lo, hi = P.min(0), P.max(0)
    # Long triangles sort by their middle and can be drawn over parts in front of them (the painter's
    # algorithm): cut every edge down to 1/24 of the model's size. Already fine meshes are unchanged.
    trail: list = []
    P, T, N, M, *U = _refine(P, T, float((hi - lo).max()) * EDGE, outline=True, extra=extra, trail=trail)
    U = U[0] if U else None
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    T = T.astype(np.int32)
    seen: dict = {}
    looks: dict = {}  # each merged material's texture
    remap = []
    for m in mats:
        t, e = m.get("texture"), m.get("emitTexture")
        k = seen.setdefault((tuple(np.round(m["fill"], 3)), m["shine"], m["metal"],
                             None if t is None else (id(t["image"]), t["tint"], t["tile"]), m.get("emit"),
                             None if e is None else (id(e["image"]), e["tint"], e["tile"]),
                             m.get("transparency")), len(seen))
        looks.setdefault(k, (t, e))
        remap.append(k)
    remap = np.array(remap + [-1], int)
    M = remap[np.where(M >= 0, M, len(mats))]
    kept = np.unique(M[M >= 0])  # materials no remaining triangle uses take no slot
    lut = np.full(len(seen) + 1, -1)
    lut[kept] = np.arange(len(kept))
    keys = list(seen)
    merged = [dict(fill=tuple(map(float, keys[k][0])), shine=keys[k][1], metal=keys[k][2],
                   **({} if looks[k][0] is None else {"texture": looks[k][0]}),
                   **({} if keys[k][4] is None else {"emit": keys[k][4]}),
                   **({} if looks[k][1] is None else {"emitTexture": looks[k][1]}),
                   **({} if keys[k][6] is None else {"transparency": keys[k][6]})) for k in kept]
    centre = (lo + hi) / 2
    about = dict(about or {})
    about.setdefault("uvSets", [] if U is None else ["vt"])
    about["textures"] = len({id(m[k]["image"]) for m in merged for k in ("texture", "emitTexture") if k in m})
    motion = None if rig is None else Motion(**rig, used=used, trail=trail, centre=centre)
    return Imported(Mesh(P - centre, N, T, lut[M].astype(np.int32), U), merged, hi - lo, motion, about)


def clip_names(imp: Imported) -> list[str]:
    """The file's animations by name, an unnamed one by its number."""
    return [c[0] or str(i) for i, c in enumerate(imp.motion.clips if imp.motion else [])]


def clip_index(imp: Imported, name) -> int | None:
    """Which animation `name` picks: one named so (case aside, when only one matches), a number counting from 0,
    or true for the first; None for none."""
    names = clip_names(imp)
    if isinstance(name, bool):
        return 0 if name and names else None
    if isinstance(name, int):
        return name if 0 <= name < len(names) else None
    if name in names:
        return names.index(name)
    low = [n.lower() for n in names]
    s = str(name).lower()
    return low.index(s) if low.count(s) == 1 else None


def pose(imp: Imported, clip: int, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The mesh's points and normals, as imp.mesh holds them, t seconds into animation `clip` (clamped to it):
    its channels set node translations, rotations, scales and morph target weights, skins follow their joints, and
    the triangles are cut as when the file was read, so a vertex moves with the ends of the edge it was cut from."""
    mo = imp.motion
    _, start, _, chans = mo.clips[clip]
    trs = list(mo.trs)
    weights: dict = {}
    for node, path, how, times, vals in chans:
        v = _sample(times, vals, how, start + t, path == "rotation")
        if path == "weights":
            weights[node] = v
        else:
            trs[node] = [v if k == path else x for k, x in zip(_PATHS, trs[node])]
    local = [m if x is mo.trs[i] else _trs(*x) for i, (m, x) in enumerate(zip(mo.local, trs))]
    G = _worlds(mo.parent, mo.order, local)
    bones = [np.stack([G[j] for j in J]) @ ibm if J else None for J, ibm in mo.skins]
    got = []
    for pc in mo.places:
        w = pc.w0
        if pc.node in weights:
            v = weights[pc.node][:len(w)]
            w = np.r_[v, w[len(v):]]
        got.append(_place(pc, np.eye(4) if pc.node is None else G[pc.node],
                          None if pc.skin is None else bones[pc.skin], w))
    P = np.concatenate([p for p, _ in got]) * (1.0, -1.0, 1.0)
    N = np.concatenate([n for _, n in got]) * (1.0, -1.0, 1.0)
    P, N = P[mo.used], N[mo.used]
    for a, b in mo.trail:  # the cuts _finish made, on the moved points
        P = np.concatenate([P, (P[a] + P[b]) / 2])
        N = np.concatenate([N, (N[a] + N[b]) / 2])
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return P - mo.centre, N


def _sample(times: np.ndarray, vals: np.ndarray, how: str, t: float, turn: bool) -> np.ndarray:
    """An animation channel's value at time t: held before its first key and after its last, STEP holding each
    key, LINEAR blending (rotations along the sphere), CUBICSPLINE through the keys with the file's tangents.
    `vals` is (keys, width), or (keys, 3, width) for CUBICSPLINE: in-tangent, value, out-tangent."""
    cubic = how == "CUBICSPLINE"
    v = vals[:, 1] if cubic else vals
    if len(times) == 1 or t <= times[0]:
        out = v[0]
    elif t >= times[-1]:
        out = v[-1]
    else:
        i = int(np.searchsorted(times, t, side="right")) - 1
        dt = float(times[i + 1] - times[i])
        u = (t - float(times[i])) / dt if dt > 0 else 0.0
        if how == "STEP":
            out = v[i]
        elif cubic:
            u2, u3 = u * u, u * u * u
            out = ((2 * u3 - 3 * u2 + 1) * v[i] + (u3 - 2 * u2 + u) * dt * vals[i, 2]
                   + (3 * u2 - 2 * u3) * v[i + 1] + (u3 - u2) * dt * vals[i + 1, 0])
        elif turn:
            out = _slerp(v[i], v[i + 1], u)
        else:
            out = v[i] * (1 - u) + v[i + 1] * u
    return out / max(float(np.linalg.norm(out)), 1e-12) if turn else out


def _slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """The rotation u of the way from quaternion a to b, the short way round."""
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:  # nearly the same: a straight blend, normalised by the caller
        return a + (b - a) * u
    th = math.acos(d)
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)


def _worlds(parent: list, order: list, local: list) -> list:
    """Each node's matrix in the file's space: its parents' and its own, in turn."""
    G = [np.eye(4)] * len(local)
    for i in order:
        G[i] = (np.eye(4) if parent[i] < 0 else G[parent[i]]) @ local[i]
    return G


def _place(pc: Place, world: np.ndarray, bones: np.ndarray | None, w: np.ndarray) -> tuple:
    """One primitive's points and normals in file space: its morph targets at weights w, then its skin's joints
    (bones: each joint's matrix times its inverse bind matrix; the node's own matrix is not used, as glTF says)
    or its node's matrix."""
    P, N = pc.P, pc.N
    if pc.dP is not None and np.any(w):
        P = P + np.tensordot(w, pc.dP, 1)
        if N is not None and pc.dN is not None:
            N = N + np.tensordot(w, pc.dN, 1)
    if bones is not None:
        B = sum(pc.weights[:, k, None, None] * bones[pc.joints[:, k], :3] for k in range(4))  # (n, 3, 4)
        P = np.einsum("nij,nj->ni", B[:, :, :3], P) + B[:, :, 3]
        if N is not None:  # each vertex's matrix's cofactors carry its normal
            c0, c1, c2 = B[:, :, 0], B[:, :, 1], B[:, :, 2]
            N = N[:, :1] * np.cross(c1, c2) + N[:, 1:2] * np.cross(c2, c0) + N[:, 2:] * np.cross(c0, c1)
    else:
        A = world[:3, :3]
        P = P @ A.T + world[:3, 3]
        if N is not None:
            N = N @ np.linalg.inv(A)
    if N is None:
        P, N = _flat(P, pc.T)[:2]
    if pc.double:
        P, N = np.r_[P, P], np.r_[N, -N]
    return P, N


# ---- glTF -----------------------------------------------------------------------------------------------

def _rest(nd: dict) -> list:
    return [nd.get("translation", (0.0, 0.0, 0.0)), nd.get("rotation", (0.0, 0.0, 0.0, 1.0)),
            nd.get("scale", (1.0, 1.0, 1.0))]


def _trs(t, r, s) -> np.ndarray:
    x, y, z, w = r
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = R * np.asarray(s, float)
    m[:3, 3] = t
    return m


def _node(nd: dict) -> np.ndarray:
    if "matrix" in nd:
        return np.array(nd["matrix"], float).reshape(4, 4).T  # stored column by column
    return _trs(*_rest(nd))


def _gltf(p: Path) -> tuple[list, list]:
    data, chunks = p.read_bytes(), []
    if data[:4] == b"glTF":
        version, total = struct.unpack_from("<II", data, 4)
        if version != 2:
            raise ValueError(f"glTF version {version}; only glTF 2.0 is read")
        off, doc = 12, None
        while off + 8 <= min(total, len(data)):
            n, kind = struct.unpack_from("<II", data, off)
            if kind == 0x4E4F534A:
                doc = json.loads(data[off + 8:off + 8 + n])
            elif kind == 0x004E4942:
                chunks.append(data[off + 8:off + 8 + n])
            off += 8 + n
        if doc is None:
            raise ValueError("no JSON chunk; not a glTF binary")
    else:
        doc = json.loads(data)
    if not str(doc.get("asset", {}).get("version", "2")).startswith("2"):
        raise ValueError(f"glTF version {doc['asset']['version']}; only glTF 2.0 is read")
    squeezed = [e for e in doc.get("extensionsRequired", []) if e in _COMPRESSED]
    if squeezed:
        raise ValueError(f"meshes are compressed with {squeezed[0]}, which Kinescript does not read; "
                         "export it again without mesh compression")
    bufs: dict = {}

    def uri(u: str) -> bytes:
        if u.startswith("data:"):
            return base64.b64decode(u.split(",", 1)[1])
        return (p.parent / unquote(u)).read_bytes()

    def buffer(i: int) -> bytes:
        if i not in bufs:
            b = doc["buffers"][i]
            bufs[i] = uri(b["uri"]) if "uri" in b else chunks[0]
        return bufs[i]

    def view(i: int) -> tuple[bytes, int, int | None]:
        v = doc["bufferViews"][i]
        return buffer(v["buffer"]), v.get("byteOffset", 0), v.get("byteStride")

    def accessor(i: int) -> np.ndarray:
        a = doc["accessors"][i]
        dt, k, n = np.dtype(_TYPES[a["componentType"]]), _WIDTH[a["type"]], a["count"]

        def read(at: dict, dt: np.dtype, n: int, k: int) -> np.ndarray:
            buf, off, stride = view(at["bufferView"])
            return np.ndarray((n, k), dt, buf, off + at.get("byteOffset", 0),
                              (stride or k * dt.itemsize, dt.itemsize)).astype(float)

        out = read(a, dt, n, k) if "bufferView" in a else np.zeros((n, k))
        sp = a.get("sparse")
        if sp:  # the values that differ, at their indices (morph targets are often stored so)
            m = sp["count"]
            idx = read(sp["indices"], np.dtype(_TYPES[sp["indices"]["componentType"]]), m, 1).ravel().astype(np.int64)
            out[idx] = read(sp["values"], dt, m, k)
        if a.get("normalized") and dt.kind in "iu":
            out = np.maximum(out / np.iinfo(dt).max, -1.0)
        return out

    pictures: dict = {}

    def texture(t: dict | None) -> tuple | None:
        """A texture's image, its mean colour (linear) and how it tiles across and down, or None when it cannot
        be read."""
        try:
            tex = doc["textures"][t["index"]]
            src = tex["source"]
            if src not in pictures:
                im = doc["images"][src]
                if "bufferView" in im:
                    buf, off, _ = view(im["bufferView"])
                    raw = buf[off:off + doc["bufferViews"][im["bufferView"]]["byteLength"]]
                else:
                    raw = uri(im["uri"])
                img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(raw))
                px = img.toarray(colorType=skia.kRGBA_8888_ColorType)
                px = px[::max(1, px.shape[0] // 64), ::max(1, px.shape[1] // 64), :3] / 255.0
                pictures[src] = (img, (px.reshape(-1, 3) ** 2.2).mean(0))
            sampler = doc["samplers"][tex["sampler"]] if "sampler" in tex else {}
            return (*pictures[src], tuple(_WRAP.get(sampler.get(k, 10497), "repeat") for k in ("wrapS", "wrapT")))
        except (KeyError, IndexError, TypeError, AttributeError, OSError):
            return None

    mats, sets = [], []  # each material, and which TEXCOORD_n its texture reads
    for mt in doc.get("materials", []):
        pbr = mt.get("pbrMetallicRoughness", {})
        factor = np.clip(np.array(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))[:3], float), 0, 1)
        got = texture(pbr.get("baseColorTexture"))
        col = factor if got is None else factor * got[1]
        metal, rough = float(pbr.get("metallicFactor", 1.0)), float(pbr.get("roughnessFactor", 1.0))
        # Rough metal looks like plain paint here; only smooth metal gets the tinted highlight.
        mats.append({"fill": tuple(np.clip(col, 0, 1) ** (1 / 2.2)), "shine": round(1 - rough, 3),
                     "metal": round(metal * (1 - rough), 3)})
        alpha = float(np.clip(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))[3:4] or [1.0], 0, 1)[0])
        if mt.get("alphaMode") == "BLEND" and alpha < 1:  # see-through: the base colour's alpha
            mats[-1]["transparency"] = round(1 - alpha, 3)
        if got is not None:  # the image times the base colour factor, as the file asks
            tint = tuple(float(v) for v in np.round(factor ** (1 / 2.2), 4))
            mats[-1]["texture"] = {"image": got[0], "tint": None if min(tint) >= 1 else tint, "tile": got[2]}
        # The light it gives off: emissiveFactor (linear, times KHR_materials_emissive_strength) times
        # emissiveTexture. "emit" is that colour as drawn, with the texture's mean in it; "emitTexture" the image
        # and the factor as its tint.
        strength = ((mt.get("extensions") or {}).get("KHR_materials_emissive_strength") or {}).get(
            "emissiveStrength", 1.0)
        emit = np.clip(np.array(mt.get("emissiveFactor", (0.0, 0.0, 0.0))[:3], float), 0, None) * float(strength)
        if emit.max() > 0:
            lit = texture(mt.get("emissiveTexture"))
            col = emit if lit is None else emit * lit[1]
            mats[-1]["emit"] = tuple(float(v) for v in np.round(col ** (1 / 2.2), 4))
            if lit is not None:
                tint = tuple(float(v) for v in np.round(emit ** (1 / 2.2), 4))
                mats[-1]["emitTexture"] = {"image": lit[0], "tint": None if tint == (1.0, 1.0, 1.0) else tint,
                                           "tile": lit[2]}
        sets.append(int((pbr.get("baseColorTexture") or mt.get("emissiveTexture") or {}).get("texCoord", 0)))
    nodes, meshes = doc.get("nodes", []), doc.get("meshes", [])
    parent = [-1] * len(nodes)
    for i, nd in enumerate(nodes):
        for k in nd.get("children", []):
            parent[k] = i
    order, seen = [], set()  # every node after its parent
    for i in [i for i in range(len(nodes)) if parent[i] < 0]:
        todo = [i]
        while todo:
            j = todo.pop()
            if j not in seen:
                seen.add(j)
                order.append(j)
                todo.extend(nodes[j].get("children", []))
    local = [_node(nd) for nd in nodes]
    rest = _worlds(parent, order, local)
    skins = []
    for sk in doc.get("skins", []):
        J = [int(j) for j in sk.get("joints", [])]
        ibm = (accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)  # column by column
               if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(J), 1, 1)))
        skins.append((J, ibm[:len(J)]))
    bones = [np.stack([rest[j] for j in J]) @ ibm if J else None for J, ibm in skins]
    pieces, places, sets_read, crowded, most = [], [], set(), 0, 0

    def emit(mi: int, world: np.ndarray, node: int | None):
        nonlocal crowded, most
        A = world[:3, :3]
        nd = nodes[node] if node is not None else {}
        skin = nd.get("skin")
        skin = skin if isinstance(skin, int) and 0 <= skin < len(skins) and skins[skin][0] else None
        for prim in meshes[mi].get("primitives", []):
            mode, at = prim.get("mode", 4), prim.get("attributes", {})
            if mode not in (4, 5, 6) or "POSITION" not in at:
                continue  # points and lines have no surface
            P = accessor(at["POSITION"])[:, :3]
            idx = accessor(prim["indices"]).ravel().astype(np.int64) if "indices" in prim else np.arange(len(P))
            if mode == 4:
                T = idx[:len(idx) // 3 * 3].reshape(-1, 3)
            else:
                i = np.arange(max(0, len(idx) - 2))
                T = (np.stack([idx[i], idx[i + 1 + i % 2], idx[i + 2 - i % 2]], 1) if mode == 5
                     else np.stack([np.full(len(i), idx[0]), idx[i + 1], idx[i + 2]], 1))
            joints = weights = None
            if skin is not None and "JOINTS_0" in at and "WEIGHTS_0" in at:
                weights = np.zeros((len(P), 4))
                got = accessor(at["WEIGHTS_0"])[:, :4]
                weights[:, :got.shape[1]] = got
                if "WEIGHTS_1" in at:  # joints past the fourth are dropped
                    crowded += int((accessor(at["WEIGHTS_1"]) > 0).any(1).sum())
                total = weights.sum(1, keepdims=True)
                weights = np.where(total > 0, weights / np.where(total > 0, total, 1.0), [1.0, 0.0, 0.0, 0.0])
                joints = np.zeros((len(P), 4), np.int64)
                got = accessor(at["JOINTS_0"])[:, :4].astype(np.int64)
                joints[:, :got.shape[1]] = np.clip(got, 0, len(skins[skin][0]) - 1)
            elif np.linalg.det(A) < 0:
                T = T[:, [0, 2, 1]]  # a mirroring node turns its faces inside out; turn them back
            targets = [t for t in prim.get("targets", []) if isinstance(t, dict)]
            most = max(most, len(targets))
            dP = dN = None
            if targets:
                dP = np.stack([accessor(t["POSITION"])[:, :3] if "POSITION" in t else np.zeros((len(P), 3))
                               for t in targets])
                if "NORMAL" in at:
                    dN = np.stack([accessor(t["NORMAL"])[:, :3] if "NORMAL" in t else np.zeros((len(P), 3))
                                   for t in targets])
            w0 = np.zeros(len(targets))
            given = list(nd.get("weights", meshes[mi].get("weights", [])))[:len(targets)]
            w0[:len(given)] = given
            mat = prim.get("material", -1)
            uv = f"TEXCOORD_{sets[mat] if mat >= 0 else 0}"
            sets_read.update(k for k in at if k.startswith("TEXCOORD_"))
            U = accessor(at[uv])[:, :2] if uv in at else None
            N = accessor(at["NORMAL"])[:, :3] if "NORMAL" in at else None
            double = bool(mat >= 0 and doc["materials"][mat].get("doubleSided"))
            pc = Place(P, N, dP, dN, joints, weights, T, double, node, skin, w0)
            places.append(pc)
            P, N = _place(pc, world, None if skin is None else bones[skin], w0)
            if pc.N is None:  # each triangle its own three vertices, with its face's normal
                U = None if U is None else U[T].reshape(-1, 2)
                T = np.arange(3 * len(T)).reshape(-1, 3)
            if double:
                T = np.r_[T, T[:, [0, 2, 1]] + len(P) // 2]
                U = None if U is None else np.r_[U, U]
            pieces.append((P, N, T, np.full(len(P), mat), U))

    if doc.get("scenes"):
        roots = doc["scenes"][doc.get("scene", 0)].get("nodes", [])
    else:
        kids = {k for nd in nodes for k in nd.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in kids]
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        i, up = stack.pop()
        world = up @ local[i]
        if "mesh" in nodes[i]:
            emit(nodes[i]["mesh"], world, i)
        stack.extend((k, world) for k in nodes[i].get("children", []))
    if not nodes:
        for mi in range(len(meshes)):
            emit(mi, np.eye(4), None)
    clips = []
    for an in doc.get("animations", []):
        chans = []
        for ch in an.get("channels", []):
            node, path = ch.get("target", {}).get("node"), ch.get("target", {}).get("path")
            if not (isinstance(node, int) and 0 <= node < len(nodes)) or path not in (*_PATHS, "weights") \
                    or (path != "weights" and "matrix" in nodes[node]):
                continue  # a node given as a matrix does not animate; KHR_animation_pointer is not read
            sm = an["samplers"][ch["sampler"]]
            how = sm.get("interpolation", "LINEAR")
            times, vals = accessor(sm["input"]).ravel(), accessor(sm["output"])
            per = len(times) * (3 if how == "CUBICSPLINE" else 1)
            width = {"translation": 3, "rotation": 4, "scale": 3}.get(path) or (vals.size // per if per else 0)
            if not len(times) or not width or vals.size != per * width:
                continue
            vals = vals.reshape(len(times), 3, width) if how == "CUBICSPLINE" else vals.reshape(len(times), width)
            chans.append((node, path, how, times, vals))
        lo = min((float(c[3][0]) for c in chans), default=0.0)
        hi = max((float(c[3][-1]) for c in chans), default=0.0)
        clips.append((an.get("name") or None, lo, hi - lo, chans))
    about = {"animations": [{"name": n or str(i), "duration": round(d, 3)} for i, (n, _, d, _) in enumerate(clips)],
             "skins": [{"name": sk.get("name") or str(i), "joints": len(skins[i][0])}
                       for i, sk in enumerate(doc.get("skins", []))],
             "morphTargets": most, "uvSets": sorted(sets_read), "crowded": crowded}
    rig = None
    if clips:
        rig = dict(parent=parent, order=order, local=local, skins=skins,
                   trs=[None if "matrix" in nd else _rest(nd) for nd in nodes], places=places, clips=clips)
    return pieces, mats, rig, about


# ---- OBJ ------------------------------------------------------------------------------------------------

def _mtl(path: Path) -> dict:
    """Materials of an .mtl file: Kd is the colour, Ns (0..1000) the shine, map_Kd a texture drawn as it is, Ke
    the light it gives off and map_Ke an image of that (times Ke, when the file gives one), d (1 solid) or Tr
    (0 solid) how see-through it is."""
    out: dict = {}
    cur = None
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            cur = out.setdefault(" ".join(t[1:]), {"fill": (0.8, 0.8, 0.8), "shine": 0.0, "metal": 0.0})
        elif cur is not None and t[0] == "Kd" and len(t) >= 4:
            cur["fill"] = tuple(min(1.0, max(0.0, float(x))) for x in t[1:4])
        elif cur is not None and t[0] == "Ns" and len(t) >= 2:
            cur["shine"] = round(min(1.0, max(0.0, float(t[1]) / 1000)) ** 0.5, 3)
        elif cur is not None and t[0] in ("d", "Tr") and len(t) >= 2:
            v = min(1.0, max(0.0, float(t[-1])))
            if (1 - v if t[0] == "d" else v) > 0:
                cur["transparency"] = round(1 - v if t[0] == "d" else v, 3)
        elif cur is not None and t[0] == "Ke" and len(t) >= 4:
            cur["_ke"] = tuple(max(0.0, float(x)) for x in t[1:4])
        elif cur is not None and t[0] in ("map_Kd", "map_Ke") and len(t) >= 2:
            f = path.parent / t[-1].replace("\\", "/")  # options such as -s come first; the file is last
            img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(f.read_bytes())) if f.is_file() else None
            if img is not None:
                cur["texture" if t[0] == "map_Kd" else "_map_ke"] = {"image": img, "tint": None,
                                                                    "tile": ("repeat", "repeat")}
    for cur in out.values():
        ke, lit = cur.pop("_ke", None), cur.pop("_map_ke", None)
        if lit is not None and (ke is None or max(ke) > 0):
            px = lit["image"].toarray(colorType=skia.kRGBA_8888_ColorType)
            mean = px[::max(1, px.shape[0] // 64), ::max(1, px.shape[1] // 64), :3].reshape(-1, 3).mean(0) / 255
            cur["emitTexture"] = dict(lit, tint=None if ke is None or ke == (1.0, 1.0, 1.0) else ke)
            cur["emit"] = tuple(float(v) for v in np.round(mean * (ke or (1.0, 1.0, 1.0)), 4))
        elif ke is not None and max(ke) > 0:
            cur["emit"] = ke
    return out


def _obj(p: Path) -> tuple[list, list]:
    V, VN, VT, tris, nrm, uvs, fm, fg = [], [], [], [], [], [], [], []
    lib: dict = {}
    names: list = []
    mat, grp = -1, 0

    def ix(s: str, n: int) -> int:
        i = int(s)
        return i - 1 if i > 0 else n + i

    for line in p.read_text(errors="replace").splitlines():
        t = line.split("#", 1)[0].split()
        if not t:
            continue
        if t[0] == "v":
            V.append(t[1:4])
        elif t[0] == "vn":
            VN.append(t[1:4])
        elif t[0] == "vt":
            VT.append((t[1:3] + ["0"])[:2])
        elif t[0] == "f" and len(t) >= 4:
            c = [tok.split("/") for tok in t[1:]]
            vi = [ix(x[0], len(V)) for x in c]
            ni = [ix(x[2], len(VN)) if len(x) > 2 and x[2] else -1 for x in c]
            ti = [ix(x[1], len(VT)) if len(x) > 1 and x[1] else -1 for x in c]
            for k in range(1, len(c) - 1):  # a polygon as a fan
                tris.append((vi[0], vi[k], vi[k + 1]))
                nrm.append((ni[0], ni[k], ni[k + 1]))
                uvs.append((ti[0], ti[k], ti[k + 1]))
                fm.append(mat)
                fg.append(grp)
        elif t[0] == "usemtl":
            name = " ".join(t[1:])
            if name not in names:
                names.append(name)
            mat = names.index(name)
        elif t[0] == "mtllib":
            lib.update(_mtl(p.parent / " ".join(t[1:])))
        elif t[0] == "s" and len(t) > 1:
            grp = 0 if t[1].lower() in ("off", "0") else int(t[1]) if t[1].isdigit() else 1
    if not tris:
        return [], []
    V = np.array(V, float)
    cv, cn, cu = np.array(tris).ravel(), np.array(nrm).ravel(), np.array(uvs).ravel()
    known = np.array([n in lib for n in names] + [False])
    fm = np.array(fm)
    fm = np.where(known[fm], fm, -1)  # usemtl of a material no .mtl defines: the layer's own
    cm, ct = np.repeat(fm, 3), np.repeat(np.arange(len(tris)), 3)
    if len(VN) and (cn >= 0).all():
        key = np.stack([cv, cn, cm, cu], 1)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        N = np.array(VN, float)[cn[first]]
    else:
        # No normals in the file: flat faces, smooth across the vertices of each smoothing group ("s 1").
        cg = np.repeat(np.array(fg), 3)
        key = np.stack([cv, np.where(cg > 0, cg, -1 - ct), cm, cu], 1)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        a, b, c = V[cv[0::3]], V[cv[1::3]], V[cv[2::3]]
        N = np.zeros((len(first), 3))
        np.add.at(N, inv.ravel(), np.repeat(np.cross(b - a, c - a), 3, axis=0))
    inv = inv.ravel()
    mats = [lib[n] if n in lib else {"fill": (0.8, 0.8, 0.8), "shine": 0.0, "metal": 0.0} for n in names]
    U = None
    if len(VT) and (cu >= 0).any():  # .obj images start at the bottom; these coordinates start at the top
        U = np.where(cu[first, None] >= 0, np.array(VT, float)[np.maximum(cu[first], 0)] * (1, -1) + (0, 1), 0.0)
    return [(V[cv[first]], N, inv.reshape(-1, 3), cm[first], U)], mats
