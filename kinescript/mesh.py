"""Meshes for model layers. Each primitive is built once at unit size with numpy and scaled per frame,
so sizes animate for free. Coordinates follow space.py: x right, y down, z toward the viewer."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import skia

from . import space

SHAPES = ("cube", "sphere", "cylinder", "cone", "torus", "plane", "capsule")
EXTRUDED = ("text", "path")
SWEPT = ("lathe", "tube")  # built at their size: a profile turned round the y axis, a circle along a path
DETAIL = {"low": 0.5, "medium": 1.0, "high": 2.0}


class Mesh(NamedTuple):
    P: np.ndarray  # (n, 3) positions
    N: np.ndarray  # (n, 3) unit normals
    T: np.ndarray  # (m, 3) triangles, wound so a face toward the camera is clockwise on screen (y down)
    M: np.ndarray  # (n,) material index of each vertex
    U: np.ndarray | None = None  # (n, 2) texture coordinates: x right and y down across the image, 1 a whole image


def _grid(nu: int, nv: int, base: int = 0) -> np.ndarray:
    """Two triangles per cell of a (nu + 1) x (nv + 1) vertex grid stored row by row in u."""
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a = base + i * (nv + 1) + j
    b, c, d = a + nv + 1, a + nv + 2, a + 1
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])


def _fan(n: int, centre: int, ring: int) -> np.ndarray:
    k = np.arange(n)
    return np.stack([np.full(n, centre), ring + k, ring + k + 1], -1)


def _orient(P, N, T, U=None) -> Mesh:
    """Wind every triangle so its face normal agrees with its vertex normals; drop degenerate ones."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    keep = np.linalg.norm(fn, axis=1) > 1e-9
    T, fn = T[keep], fn[keep]
    flip = (fn * (N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]])).sum(1) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    return Mesh(P.astype(float), N / np.linalg.norm(N, axis=1, keepdims=True), T.astype(np.int32),
                np.zeros(len(P), np.int32), None if U is None else np.asarray(U, float))


def cat(meshes) -> Mesh:
    base = np.cumsum([0] + [len(m.P) for m in meshes[:-1]])
    U = None
    if any(m.U is not None for m in meshes):  # a mesh without coordinates takes the image's corner
        U = np.concatenate([np.zeros((len(m.P), 2)) if m.U is None else m.U for m in meshes])
    return Mesh(np.concatenate([m.P for m in meshes]), np.concatenate([m.N for m in meshes]),
                np.concatenate([m.T + b for m, b in zip(meshes, base)]), np.concatenate([m.M for m in meshes]), U)


def _surface(fn, nu: int, nv: int):
    """Vertices of a parametric surface over u, v in [0, 1] (both ends included), and their u, v."""
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
    P, N = fn(u.ravel(), v.ravel())
    return P, N, _grid(nu, nv), np.stack([u.ravel(), v.ravel()], -1)


def _wrapped(P, N, T, uv) -> Mesh:
    """A surface wrapped round the y axis (u the angle from +x toward +z, v from top to bottom): the image runs
    left to right across the front as the camera sees it, its middle facing the camera, its edges meeting behind."""
    return _orient(P, N, T, np.stack([0.75 - uv[:, 0], uv[:, 1]], -1))


def _facing(P: np.ndarray, n, half: float = 0.5) -> np.ndarray:
    """Texture coordinates of points P on a flat face with normal n (an axis), as seen from in front of the face:
    upright on a side, on top and underneath with the image's top away from the camera. P spans -half..half."""
    n = np.asarray(n, float)
    down = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.5 else np.array([0.0, 0.0, -n[1]])
    right = np.cross(down, n)
    return np.stack([P @ right, P @ down], -1) / (2 * half) + 0.5


def _disc(y: float, n: int, normal_y: float):
    a = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.stack([np.cos(a), np.full(n + 1, y), np.sin(a)], -1)
    P = np.concatenate([[[0, y, 0]], ring])
    return P, np.tile([0, normal_y, 0], (n + 2, 1)), _fan(n, 0, 1), _facing(P, (0, normal_y, 0), 1.0)


def _band(n: int, arc: int, q: float) -> np.ndarray:
    """Coordinates across a rounded box's face from -0.5 to 0.5: n pieces over the flat middle, and arc pieces
    into each rounded edge q wide, spaced so each turns the surface by the same angle."""
    edge = (0.5 - q) + q * np.tan(np.linspace(0, np.pi / 4, arc + 1))
    c = np.concatenate([-edge[::-1], np.linspace(-(0.5 - q), 0.5 - q, n + 1)[1:-1], edge])
    return c[np.r_[True, np.diff(c) > 1e-9]]


def _revolve(R, Y, NR, NY, V, seg: int) -> Mesh:
    """A surface turned round the y axis in seg pieces: profile points at radius R and height Y, their normals
    (NR, NY) in the profile's plane, and V how far down the image each is."""
    th = np.linspace(0, 2 * np.pi, seg + 1)
    c, s, n = np.cos(th)[:, None], np.sin(th)[:, None], len(R)
    P = np.stack([c * R, np.broadcast_to(Y, (seg + 1, n)), s * R], -1).reshape(-1, 3)
    N = np.stack([c * NR, np.broadcast_to(NY, (seg + 1, n)), s * NR], -1).reshape(-1, 3)
    uv = np.stack([np.repeat(th / (2 * np.pi), n), np.tile(V, seg + 1)], -1)
    return _wrapped(P, N, _grid(seg, n - 1), uv)


@lru_cache(maxsize=64)
def unit(shape: str, detail: str = "medium", form=0.35) -> Mesh:
    """`shape` at unit size: cube and plane span -0.5..0.5, round shapes have radius 1 and height 1. `form` is a
    share of the size: the torus's tube radius, the cube's rounding along x, y and z (0: square edges), or the
    capsule's cap height."""
    k = DETAIL.get(detail, 1.0)
    seg = max(6, round(24 * k))
    if shape == "cube" and isinstance(form, tuple):
        # Each face's grid pushed out onto a box with ellipsoid corners; scaled to w, h and depth they are round.
        n, arc, q = max(1, round(6 * k)), max(1, round(seg / 8)), np.asarray(form, float)
        faces = []
        for i, (a, b, c) in enumerate((a, b, c) for a in range(3) for b, c in [((a + 1) % 3, (a + 2) % 3),
                                                                            ((a + 2) % 3, (a + 1) % 3)]):
            u, v = np.meshgrid(_band(n, arc, q[b]), _band(n, arc, q[c]), indexing="ij")
            F = np.zeros((u.size, 3))
            F[:, a], F[:, b], F[:, c] = 0.5 if i % 2 == 0 else -0.5, u.ravel(), v.ravel()
            inner = np.clip(F, q - 0.5, 0.5 - q)
            d = (F - inner) / q
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            face = np.zeros(3)
            face[a] = F[0, a] * 2
            faces.append(_orient(inner + d * q, d / q, _grid(u.shape[0] - 1, u.shape[1] - 1), _facing(F, face)))
        return cat(faces)
    if shape in ("cube", "plane"):
        n = max(1, round(6 * k))
        faces = []
        axes = [(2, 0, 1), (2, 1, 0)] if shape == "plane" else [(a, b, c) for a in range(3) for b, c in
                                                                 [((a + 1) % 3, (a + 2) % 3), ((a + 2) % 3, (a + 1) % 3)]]
        for i, (a, b, c) in enumerate(axes):
            sign = 1.0 if i % 2 == 0 else -1.0
            u, v = np.meshgrid(np.linspace(-0.5, 0.5, n + 1), np.linspace(-0.5, 0.5, n + 1), indexing="ij")
            P = np.zeros((u.size, 3))
            P[:, a], P[:, b], P[:, c] = (0.0 if shape == "plane" else 0.5 * sign), u.ravel(), v.ravel()
            N = np.zeros_like(P)
            N[:, a] = sign
            faces.append(_orient(P, N, _grid(n, n), _facing(P, N[0])))
        return cat(faces)
    if shape == "sphere":
        def fn(u, v):
            th, ph = u * 2 * np.pi, v * np.pi
            P = np.stack([np.sin(ph) * np.cos(th), -np.cos(ph), np.sin(ph) * np.sin(th)], -1)
            return P, P
        return _wrapped(*_surface(fn, 2 * seg, seg))
    if shape == "torus":
        def fn(u, v):
            th, ph = u * 2 * np.pi, v * 2 * np.pi
            N = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)
            P = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1) + form * N
            return P, N
        return _orient(*_surface(fn, round(2.6 * seg), seg))  # the image once round the ring and round the tube
    if shape in ("cylinder", "cone"):
        cone = shape == "cone"

        def fn(u, v):
            th = u * 2 * np.pi
            r = v if cone else np.ones_like(v)
            P = np.stack([r * np.cos(th), v - 0.5, r * np.sin(th)], -1)
            N = np.stack([np.cos(th), np.full_like(th, -1.0 if cone else 0.0), np.sin(th)], -1)
            return P, N
        side = _wrapped(*_surface(fn, 2 * seg, max(1, seg // 6)))
        caps = [_orient(*_disc(0.5, 2 * seg, 1.0))] + ([] if cone else [_orient(*_disc(-0.5, 2 * seg, -1.0))])
        return cat([side] + caps)
    if shape == "capsule":
        # Half ellipsoids form high on a cylinder; scaled to r and h they are half spheres.
        q, ph = min(0.5, max(float(form), 0.001)), np.linspace(0, np.pi / 2, max(2, round(seg / 4)) + 1)
        body = np.linspace(q - 0.5, 0.5 - q, max(1, seg // 6) + 1)[1:-1] if q < 0.5 else np.zeros(0)
        R = np.concatenate([np.sin(ph), np.ones(len(body)), np.sin(ph[::-1])])
        Y = np.concatenate([q - 0.5 - q * np.cos(ph), body, 0.5 - q + q * np.cos(ph[::-1])])
        NY = np.concatenate([-np.cos(ph) / q, np.zeros(len(body)), np.cos(ph[::-1]) / q])
        return _revolve(R, Y, R, NY, Y + 0.5, 2 * seg)
    raise ValueError(f"unknown shape '{shape}'. Known: {', '.join(SHAPES)}")


def size(shape: str, get) -> tuple[tuple[float, float, float], float]:
    """(x, y, z) scale of the unit shape and the torus tube ratio, from w/h/depth/r/tube via get(key, default)."""
    if shape in ("cube", "plane"):
        w = float(get("w", 200))
        s3 = (w, float(get("h", w)), float(get("depth", w)) if shape == "cube" else 1.0)
        rr = min(float(get("r", 0) or 0), min(s3) / 2) if shape == "cube" else 0.0  # its edges' rounding
        return s3, (tuple(max(round(rr / v, 3), 0.001) for v in s3) if rr > 0 else 0.0)
    if shape == "capsule":
        r = float(get("r", 60))
        h = max(float(get("h", 240)), 2 * r)
        return (r, h, r), round(r / h, 3) if h > 0 else 0.5
    r = float(get("r", {"sphere": 100, "torus": 110, "cylinder": 80, "cone": 90}[shape]))
    if shape == "torus":
        return (r, r, r), round(min(0.95, max(0.02, float(get("tube", r * 0.35)) / (r or 1))), 2)
    return ((r, r, r) if shape == "sphere" else (r, float(get("h", 200)), r)), 0.35


def placed(m: Mesh, scale3, matrix: np.ndarray | None = None) -> Mesh:
    """`m` scaled by scale3, then moved by a 4x4 matrix (normals by its inverse transpose)."""
    s = np.asarray(scale3, float)
    P, N = m.P * s, m.N / s
    if matrix is not None:
        P = P @ matrix[:3, :3].T + matrix[:3, 3]
        N = N @ np.linalg.inv(matrix[:3, :3])
    return m._replace(P=P, N=N / np.linalg.norm(N, axis=1, keepdims=True))


def part_matrix(p: dict) -> np.ndarray:
    """A part's placement inside its model, in the same order as a layer transform."""
    f = lambda k, d=0.0: float(p.get(k, d))  # noqa: E731
    sc = f("scale", 1.0)
    return (space.translate(f("x"), f("y")) @ space.tilt(f("z"), f("rotateX"), f("rotateY"))
            @ space.rotate_z(f("rotate")) @ np.diag([sc, sc, sc, 1.0]))


def merge(meshes: list[Mesh], materials: list[int]) -> Mesh:
    """One mesh from several, each vertex tagged with its mesh's material slot."""
    return cat([m._replace(M=np.full(len(m.P), k, np.int32)) for m, k in zip(meshes, materials)])


def edges(P: np.ndarray, T: np.ndarray) -> tuple:
    """Each edge of triangles T once, for drawing lines along them. Corners at the same place are one point, so
    a cube's faces, whose corners are kept apart for their normals, still meet; triangles on the same three
    points are one face, so a face drawn back to back (a double-sided plane) counts once. Returns each vertex's
    point, each triangle's face (-1 for one with two corners at a point), each face's first triangle, and per
    edge its two points, the faces either side (the second is the face count past a border) and whether the
    second face runs along the edge the way the first does (a mesh wound against itself there). An edge three or
    more faces meet at (where a cut part's face meets the faces of the part round it) has its first face both
    sides, so it draws no line: which two of them face each other is not known."""
    span = float(np.ptp(P, axis=0).max()) if len(P) else 0.0
    q = np.round((P - P.min(0)) / (span or 1.0) * 1e6).astype(np.int64) if len(P) else np.zeros((0, 3), np.int64)
    _, weld = np.unique((q[:, 0] * 1000003 + q[:, 1]) * 1000003 + q[:, 2], return_inverse=True)
    weld = weld.ravel()
    W = weld[T]
    _, first, face = np.unique(np.sort(W, 1), axis=0, return_index=True, return_inverse=True)
    face = face.ravel()
    ok = (W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 2] != W[:, 0])
    face[~ok] = -1
    g = np.flatnonzero(ok[first])  # the faces with three points
    Wf = W[first[g]]
    a, b, f = Wf.ravel(), Wf[:, [1, 2, 0]].ravel(), np.repeat(g, 3)
    n = int(weld.max()) + 1 if len(weld) else 1
    key = np.minimum(a, b) * n + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    key, a, b, f = key[o], a[o], b[o], f[o]
    s = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, np.int64)
    count = np.r_[s[1:], len(key)] - s
    two = count >= 2  # a second face along this edge
    nxt = np.minimum(s + 1, max(len(key) - 1, 0))
    sides = np.stack([f[s], np.where(count > 2, f[s], np.where(two, f[nxt], len(first)))], 1)
    return weld, face, first, np.stack([a[s], b[s]], 1), sides, (count == 2) & (a[nxt] == a[s])


def tiled(e: tuple, n: int, nt: int) -> tuple:
    """edges() of n copies of one mesh of nt triangles merged in order (cat), from the one's
    edges() e: each copy's points, faces and edges numbered after the copy before's, borders past the last face."""
    weld, face, first, ends, sides, same = e
    nw, nf = (int(weld.max()) + 1 if len(weld) else 1), len(first)
    k = np.arange(n)[:, None]
    two = sides[None] + (k * nf)[:, :, None]
    two[..., 1] = np.where(sides[None, :, 1] == nf, n * nf, two[..., 1])
    return ((weld[None] + k * nw).ravel(), np.where(face[None] < 0, -1, face[None] + k * nf).ravel(),
            (first[None] + k * nt).ravel(), (ends[None] + (k * nw)[:, :, None]).reshape(-1, 2),
            two.reshape(-1, 2), np.tile(same, n))


def build_order(m: Mesh) -> np.ndarray:
    """Triangle indices bottom to top, the order `draw` and the build motion reveal them in."""
    return np.argsort(-m.P[m.T, 1].mean(1), kind="stable")


# The side a cut comes in from: the axis across it (0 x, 1 y, 2 z) and 1 from the axis's low end, -1 its high one.
# y runs down and z toward the camera.
SIDES = {"front": (2, -1.0), "back": (2, 1.0), "left": (0, 1.0), "right": (0, -1.0), "top": (1, 1.0),
         "bottom": (1, -1.0)}


def cut(m: Mesh, axis: int, side: float, at: float, cap: int | None = None) -> tuple[Mesh, int]:
    """Mesh m with one end taken off by a plane across `axis` (0 x, 1 y, 2 z), and the opening closed. The plane
    is `at` of the mesh's size along the axis in from its low end (side 1) or its high end (side -1), and what
    lies between that end and the plane goes. Triangles across the plane are cut along it. The cap fills each
    closed line the plane traces round a part, less the lines inside it, so a part inside another shows its own
    cut face and a hollow stays open. It is flat and faces the end taken off. The outermost faces take material
    slot `cap` (each part's own when cap is None), and a part inside another its own. A surface that does not
    close (a plane, an open lathe line) gets no cap.
    Returns the mesh and how many triangles of the cap end it."""
    if not len(m.T) or at <= 0:
        return m, 0
    lo, hi = float(m.P[:, axis].min()), float(m.P[:, axis].max())
    if hi - lo <= 1e-9:
        return m, 0
    c = lo + at * (hi - lo) if side > 0 else hi - at * (hi - lo)
    f = (m.P[:, axis] - c) * side  # above 0: kept
    f = np.where(np.abs(f) < 1e-7 * float(np.ptp(m.P, axis=0).max()), 0.0, f)  # on the plane: kept, and cut at
    keep = f >= 0
    k = keep[m.T].sum(1)
    cross = np.flatnonzero((k == 1) | (k == 2))
    whole = m.T[k == 3]
    if not len(cross):
        return _used(m, whole), 0
    one = k[cross] == 1  # one corner kept (a smaller triangle), else two (a four-sided piece, two triangles)
    Tc = m.T[cross]
    odd = np.where(one, np.argmax(keep[Tc], 1), np.argmin(keep[Tc], 1))
    o, p, q = np.take_along_axis(Tc, (odd[:, None] + np.arange(3)) % 3, 1).T  # odd corner first, wound as before
    n0, n = len(m.P), len(cross)
    key = np.r_[np.minimum(o, p) * n0 + np.maximum(o, p), np.minimum(o, q) * n0 + np.maximum(o, q)]
    uk, inv = np.unique(key.astype(np.int64), return_inverse=True)  # one point per edge the plane crosses
    inv = inv.ravel()
    a, b = uk // n0, uk % n0
    at_end = (f[a] == 0) | (f[b] == 0)  # crossed at a corner on the plane: that corner
    idx = np.where(f[a] == 0, a, b).astype(np.int64)
    a, b = a[~at_end], b[~at_end]
    idx[~at_end] = n0 + np.arange(len(a))
    s = (f[a] / (f[a] - f[b]))[:, None]
    nn = m.N[a] + s * (m.N[b] - m.N[a])
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
    P, N = np.r_[m.P, m.P[a] + s * (m.P[b] - m.P[a])], np.r_[m.N, nn]
    M = np.r_[m.M, m.M[np.where(keep[a], a, b)]]
    U = None if m.U is None else np.r_[m.U, m.U[a] + s * (m.U[b] - m.U[a])]
    A, B = idx[inv[:n]], idx[inv[n:]]
    T = np.concatenate([whole, np.stack([o, A, B], 1)[one], np.stack([A, p, q], 1)[~one],
                        np.stack([A, q, B], 1)[~one]]).astype(np.int32)
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 2] != T[:, 0])]  # none of no size, at plane corners
    # The line the plane traces, each piece run the way the border of the surface left runs.
    lines = np.where(one[:, None], np.stack([A, B], 1), np.stack([B, A], 1))
    Pc, Tcap, Mc = _caps(P, lines, m.M[np.where(one, o, p)], axis, side, c, cap)
    if len(Tcap):
        nc = np.zeros(3)
        nc[axis] = -side
        T = np.r_[T, Tcap + len(P)].astype(np.int32)
        P, N = np.r_[P, Pc], np.r_[N, np.tile(nc, (len(Pc), 1))]
        M = np.r_[M, Mc]
        U = None if U is None else np.r_[U, np.zeros((len(Pc), 2))]
    return _used(Mesh(P, N, T, M.astype(np.int32), U), T), len(Tcap)


def _used(m: Mesh, T: np.ndarray) -> Mesh:
    """m with triangles T and only the points they use."""
    used, back = np.unique(T, return_inverse=True)
    return Mesh(m.P[used], m.N[used], back.reshape(-1, 3).astype(np.int32), m.M[used],
                None if m.U is None else m.U[used])


def _caps(P: np.ndarray, lines: np.ndarray, mats: np.ndarray, axis: int, side: float, c: float,
          slot: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The flat faces closing a cut (see cut): points, triangles into them and each point's material. lines are
    the pieces of the plane's trace as pairs of indices into P, each run the way the border of the surface left
    runs, and mats each piece's material."""
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int32))
    if not len(lines):
        return empty
    u, v = (axis + 1) % 3, (axis + 2) % 3  # the plane's own axes, in turn after the cut one, so area is signed
    span = float(np.ptp(P, axis=0).max()) or 1.0
    q = np.round((P[lines.ravel()] - P.min(0)) / span * 1e6).astype(np.int64)
    ids = np.unique(q, axis=0, return_inverse=True)[1].ravel().reshape(-1, 2)  # ends at one place are one point
    good = ids[:, 0] != ids[:, 1]
    ids, lines, mats = ids[good], lines[good], mats[good]
    start: dict[int, int] = {}
    for i, s0 in enumerate(ids[:, 0].tolist()):
        start.setdefault(s0, i)
    seen, loops = np.zeros(len(ids), bool), []
    for i in range(len(ids)):
        run, j = [], i
        while j is not None and not seen[j]:
            seen[j] = True
            run.append(j)
            j = start.get(int(ids[j, 1]))
        if j == i and len(run) >= 3:  # back where it began: a closed line
            loops.append(run)
    rings = [(P[lines[run, 0]][:, [u, v]], int(mats[run[0]])) for run in loops]
    rings = [(g, mt) for g, mt in rings if abs(_area(g)) > 1e-9 * span * span]
    if not rings:
        return empty
    lo, hi = [g.min(0) for g, _ in rings], [g.max(0) for g, _ in rings]
    inside = [[j for j in range(len(rings)) if j != i and (lo[i] <= lo[j]).all() and (hi[j] <= hi[i]).all()
               and _contains(rings[i][0], rings[j][0][0])] for i in range(len(rings))]  # the lines within line i
    pts, tris, mat, base = [], [], [], 0
    for i, (g, mt) in enumerate(rings):
        if side * _area(g) <= 0:  # run the other way round: a hollow's inner wall, whose inside stays open
            continue
        holes = [j for j in inside[i] if not any(j in inside[k] for k in inside[i])]
        Q, _, tri = shape2d([g] + [rings[j][0] for j in holes])
        if not len(tri):
            continue
        Q, tri = _refine(Q, tri, span / 24)  # small pieces, as an extruded face's, for the painter's sort
        X = np.zeros((len(Q), 3))
        X[:, u], X[:, v], X[:, axis] = Q[:, 0], Q[:, 1], c
        a, b, d = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
        tri = tri.copy()
        flip = np.cross(b - a, d - a)[:, axis] * side > 0  # each wound to face the end taken off
        tri[flip] = tri[flip][:, [0, 2, 1]]
        pts.append(X)
        tris.append(tri + base)
        outer = not any(i in inside[j] for j in range(len(rings)))
        mat.append(np.full(len(Q), slot if slot is not None and outer else mt, np.int32))
        base += len(Q)
    if not pts:
        return empty
    return np.concatenate(pts), np.concatenate(tris), np.concatenate(mat)


# ---- extruded outlines (text and SVG paths) -----------------------------------------------------------

def _flatten(verb, p: np.ndarray, seg: float, w: float) -> np.ndarray:
    """Points along one path segment, after its start point. Curves are cut into pieces about `seg` long,
    lines into pieces up to 3 * seg, so walls stay short enough for the painter's sort."""
    if verb == skia.Path.kLine_Verb:
        n = max(1, math.ceil(float(np.hypot(*(p[1] - p[0]))) / (3 * seg)))
    else:
        n = max(1, min(64, math.ceil(float(np.hypot(*np.diff(p, axis=0).T).sum()) / seg)))
    t = np.linspace(0, 1, n + 1)[1:, None]
    s = 1 - t
    if verb == skia.Path.kLine_Verb:
        return p[0] + t * (p[1] - p[0])
    if verb == skia.Path.kQuad_Verb:
        return s * s * p[0] + 2 * s * t * p[1] + t * t * p[2]
    if verb == skia.Path.kConic_Verb:
        return (s * s * p[0] + 2 * w * s * t * p[1] + t * t * p[2]) / (s * s + 2 * w * s * t + t * t)
    return s ** 3 * p[0] + 3 * s * s * t * p[1] + 3 * s * t * t * p[2] + t ** 3 * p[3]


def _area(q: np.ndarray) -> float:
    """Signed area; positive when the outline runs clockwise on screen (y down)."""
    return 0.5 * float((q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]).sum())


def _ring(pts) -> np.ndarray | None:
    q = np.asarray(pts, float)
    q = q[np.any(np.abs(q - np.roll(q, 1, 0)) > 1e-6, axis=1)]  # drop repeated points, the closing one too
    return q if len(q) >= 3 and abs(_area(q)) > 1e-6 else None


def contours(path: skia.Path, seg: float) -> list[np.ndarray]:
    """The closed outlines of `path` as (n, 2) arrays, with overlaps removed."""
    try:
        path = skia.Simplify(path)
    except Exception:  # Simplify gives up on some degenerate paths; use the path as given
        pass
    out, cur = [], []
    it = skia.Path.Iter(path, True)
    while True:
        verb, pts = it.next()
        if verb in (skia.Path.kMove_Verb, skia.Path.kClose_Verb, skia.Path.kDone_Verb):
            q = _ring(cur) if len(cur) > 2 else None
            if q is not None:
                out.append(q)
            cur = [(pts[0].fX, pts[0].fY)] if verb == skia.Path.kMove_Verb else []
            if verb == skia.Path.kDone_Verb:
                return out
            continue
        cur.extend(_flatten(verb, np.array([(q.fX, q.fY) for q in pts]), seg, it.conicWeight()))


def _contains(q: np.ndarray, p) -> bool:
    """Whether point p is inside the closed outline q (even-odd)."""
    x, y = q[:, 0], q[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = ((y > p[1]) != (y2 > p[1])) & (p[0] < x + (p[1] - y) * (x2 - x) / (y2 - y))
    return bool(hit.sum() % 2)


def _in_tri(p: np.ndarray, a, b, c) -> np.ndarray:
    """Which points p (k, 2) lie inside or on triangle abc, either winding."""
    d1 = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
    d2 = (c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0])
    d3 = (a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0])
    return ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))


def _bridge(Q: np.ndarray, ring: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Join `hole` into `ring` (index arrays into Q) with a cut from the hole's rightmost point to a ring point
    it can see (Eberly, "Triangulation by ear clipping"). The ring runs clockwise on screen, the hole the other way."""
    m = hole[np.argmax(Q[hole, 0])]
    mx, my = Q[m]
    a, b = Q[ring], Q[np.roll(ring, -1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = a[:, 0] + (my - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    ok = ((a[:, 1] > my) != (b[:, 1] > my)) & (x >= mx)
    if not ok.any():
        return ring
    e = np.flatnonzero(ok)[np.argmin(x[ok])]
    k = e if a[e, 0] > b[e, 0] else (e + 1) % len(ring)  # the end of the hit edge further along the ray
    prv, nxt = Q[np.roll(ring, 1)], Q[np.roll(ring, -1)]
    reflex = (a[:, 0] - prv[:, 0]) * (nxt[:, 1] - prv[:, 1]) - (a[:, 1] - prv[:, 1]) * (nxt[:, 0] - prv[:, 0]) < 0
    block = reflex & _in_tri(a, (mx, my), (x[e], my), Q[ring[k]])
    block[k] = False
    if block.any():  # a reflex corner hides that point: cut to the corner closest in angle to the ray instead
        cand = np.flatnonzero(block)
        d = a[cand] - (mx, my)
        dist = np.hypot(d[:, 0], d[:, 1])
        k = cand[np.lexsort((dist, np.abs(d[:, 1]) / np.maximum(dist, 1e-12)))[0]]
    j = int(np.flatnonzero(hole == m)[0])
    return np.concatenate([ring[:k + 1], hole[j:], hole[:j], [m, ring[k]], ring[k + 1:]])


def _earclip(Q: np.ndarray) -> np.ndarray:
    """Triangles filling the simple polygon Q (n, 2), which runs clockwise on screen. (m, 3) indices into Q."""
    n = len(Q)
    prv, nxt = np.roll(np.arange(n), 1), np.roll(np.arange(n), -1)
    alive = np.ones(n, bool)
    tiny = 1e-9 * float(np.ptp(Q, axis=0).max()) ** 2

    def cross(a, b, c):
        return (Q[b, 0] - Q[a, 0]) * (Q[c, 1] - Q[a, 1]) - (Q[b, 1] - Q[a, 1]) * (Q[c, 0] - Q[a, 0])

    def blocked(a, b, c):
        p = Q[alive]
        same = (p == Q[a]).all(1) | (p == Q[b]).all(1) | (p == Q[c]).all(1)  # the corners, and bridge copies
        return bool((_in_tri(p, Q[a], Q[b], Q[c]) & ~same).any())

    out, i, left, miss = [], 0, n, 0
    while left > 3:
        a, c = prv[i], nxt[i]
        if cross(a, i, c) > tiny and not blocked(a, i, c):
            out.append((a, i, c))
        elif miss < left:
            i, miss = c, miss + 1
            continue
        else:  # no clean ear in a whole lap (degenerate outline): drop a flat corner, else the most convex one
            ids = np.flatnonzero(alive)
            cr = cross(prv[ids], ids, nxt[ids])
            flat = np.abs(cr) <= tiny
            i = int(ids[np.argmax(flat)] if flat.any() else ids[np.argmax(cr)])
            a, c = prv[i], nxt[i]
            if not flat.any():
                out.append((a, i, c))
        alive[i] = False
        nxt[a], prv[c] = c, a
        i, left, miss = c, left - 1, 0
    i = int(np.flatnonzero(alive)[0])
    out.append((prv[i], i, nxt[i]))
    return np.array(out, np.int32).reshape(-1, 3)


def shape2d(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outlines as one point array Q, wound for extrusion (outer outlines clockwise on screen, holes the other
    way), the index where each outline starts, and the triangles (into Q) that fill them."""
    lo, hi = [q.min(0) for q in rings], [q.max(0) for q in rings]
    within = [[j for j in range(len(rings)) if j != i and (lo[j] <= q[0]).all() and (q[0] <= hi[j]).all()
               and _contains(rings[j], q[0])] for i, q in enumerate(rings)]
    rings = [q if (len(w) % 2 == 0) == (_area(q) > 0) else q[::-1] for q, w in zip(rings, within)]
    starts = np.cumsum([0] + [len(q) for q in rings[:-1]]).astype(np.int64)
    Q = np.concatenate(rings) if rings else np.zeros((0, 2))
    tris = []
    for i, w in enumerate(within):
        if len(w) % 2:
            continue
        ring = np.arange(starts[i], starts[i] + len(rings[i]))
        holes = [np.arange(starts[h], starts[h] + len(rings[h])) for h, wh in enumerate(within)
                 if len(wh) == len(w) + 1 and i in wh]
        for h in sorted(holes, key=lambda h: -Q[h, 0].max()):
            ring = _bridge(Q, ring, h)
        tris.append(ring[_earclip(Q[ring])])
    return Q, starts, (np.concatenate(tris) if tris else np.zeros((0, 3), np.int64))


def _refine(P: np.ndarray, T: np.ndarray, most: float, outline: bool = False, extra: tuple = (), long=None,
            rounds: int = 12, trail: list | None = None) -> tuple:
    """Split edges longer than `most` at their midpoints until none is left, so a surface is made of small
    triangles the painter's sort can order against nearby walls and parts. Returns P, T and the per-vertex
    arrays in `extra` grown to match (a new vertex takes the mean of its edge's ends, or the first end's
    integer). `outline=False` never splits edges on the outline, so a cap still meets its walls.
    long(P, extra, a, b), when given, says which edges (from vertices a to vertices b) are too long instead.
    `trail`, when given, gets each round's cut edges (a, b): the new vertices of that round are their midpoints,
    in order, so the same cuts can be made again on the vertices moved somewhere else."""
    extra = list(extra)
    for _ in range(rounds):
        e = np.sort(T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)  # edges ab, bc, ca of each triangle
        # As np.unique(e, axis=0), in the same order, in a fifth of the time: one int64 per edge.
        keys, inv, count = np.unique(e[:, 0].astype(np.int64) * len(P) + e[:, 1], return_inverse=True,
                                     return_counts=True)
        edges = np.stack([keys // len(P), keys % len(P)], 1)
        a, b = edges[:, 0], edges[:, 1]
        cut = ((count == 2) | outline) & (np.linalg.norm(P[a] - P[b], axis=1) > most if long is None
                                          else long(P, extra, a, b))
        if not cut.any():
            break
        mid = np.full(len(edges), -1)
        mid[cut] = len(P) + np.arange(cut.sum())
        a, b = edges[cut, 0], edges[cut, 1]
        if trail is not None:
            trail.append((a, b))
        P = np.concatenate([P, (P[a] + P[b]) / 2])
        extra = [np.concatenate([x, x[a] if x.dtype.kind in "iu" else (x[a] + x[b]) / 2]) for x in extra]
        M = mid[inv.ravel()].reshape(-1, 3)
        n = (M >= 0).sum(1)
        # Turn each triangle so one cut is on ab, or two cuts are on ab and bc; winding is unchanged.
        r = np.where(n == 1, np.argmax(M >= 0, 1), np.where(n == 2, (np.argmin(M >= 0, 1) + 1) % 3, 0))
        turn = (np.arange(3)[None] + r[:, None]) % 3
        (a, b, c), (ab, bc, ca) = np.take_along_axis(T, turn, 1).T, np.take_along_axis(M, turn, 1).T
        tri = lambda *v, k: np.stack(v, 1)[n == k]  # noqa: E731
        T = np.concatenate([tri(a, b, c, k=0), tri(a, ab, c, k=1), tri(ab, b, c, k=1),
                            tri(ab, b, bc, k=2), tri(a, ab, bc, k=2), tri(a, bc, c, k=2),
                            tri(a, ab, ca, k=3), tri(ab, b, bc, k=3), tri(ca, bc, c, k=3), tri(ab, bc, ca, k=3)])
    return (P, T, *extra)


def extrude(Q: np.ndarray, starts: np.ndarray, caps: np.ndarray, depth: float, bevel: float, rows: int,
            edge: float) -> Mesh:
    """A solid from shape2d's outlines, spanning z = -depth/2..depth/2. The walls are cut into `rows` bands
    and shade smoothly where outline pieces meet at under about 45 degrees; `bevel` chamfers the edges at
    45 degrees; no cap edge inside the outline is longer than `edge`."""
    n = len(Q)
    size = np.diff(np.append(starts, n))
    first, count = np.repeat(starts, size), np.repeat(size, size)
    idx = np.arange(n)
    nxt, prv = first + (idx - first + 1) % count, first + (idx - first - 1) % count
    d = Q[nxt] - Q
    en = np.stack([d[:, 1], -d[:, 0]], 1)
    en /= np.linalg.norm(en, axis=1, keepdims=True)  # outward normal of the edge from point i to the next
    ep = en[prv]
    dot = (en * ep).sum(1)
    vn = en + ep
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    smooth = (dot > 0.7)[:, None]
    n0, n1 = np.where(smooth, vn, en), np.where(smooth[nxt], vn[nxt], en)  # normal at each end of each edge
    b = max(0.0, min(bevel, depth * 0.45))
    zf = depth / 2 - b
    pieces = []

    def band(rings, z, nz):
        """A strip of quads along every edge, through one ring of points per height in z."""
        P = np.empty((n, 2, len(z), 3))
        N = np.empty_like(P)
        for j, r in enumerate(rings):
            P[:, 0, j, :2], P[:, 1, j, :2] = r, r[nxt]
        P[..., 2] = z
        N[:, 0, :, :2], N[:, 1, :, :2] = n0[:, None], n1[:, None]
        N[..., 2] = nz
        T = _grid(1, len(z) - 1)[None] + (np.arange(n) * 2 * len(z))[:, None, None]
        pieces.append((P.reshape(-1, 3), N.reshape(-1, 3), T.reshape(-1, 3)))

    band([Q] * (rows + 1), np.linspace(zf, -zf, rows + 1), 0.0)
    face = Q
    if b > 0:
        face = Q - b * (en + ep) / np.maximum(1 + dot, 0.5)[:, None]  # outline moved in along its mitred normals
        for s in (1.0, -1.0):
            band([face, Q], np.array([s * depth / 2, s * zf]), s)
    face, caps = _refine(face, caps, edge)
    for s in (1.0, -1.0):
        P = np.concatenate([face, np.full((len(face), 1), s * depth / 2)], 1)
        pieces.append((P, np.tile([0.0, 0.0, s], (len(face), 1)), caps))
    base = np.cumsum([0] + [len(p[0]) for p in pieces[:-1]])
    P = np.concatenate([p[0] for p in pieces])
    lo, hi = Q.min(0), Q.max(0)
    U = (P[:, :2] - lo) / np.maximum(hi - lo, 1e-9)  # the image across the outline's box, seen from the front
    return _orient(P, np.concatenate([p[1] for p in pieces]),
                   np.concatenate([p[2] + o for p, o in zip(pieces, base)]).astype(np.int64), U)


# ---- turned and swept shapes (lathe, tube) -------------------------------------------------------------

_EMPTY = Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int32), np.zeros(0, np.int32))


def polylines(path: skia.Path, seg: float) -> list[tuple[np.ndarray, bool]]:
    """Each piece of `path` as drawn, as (n, 2) points and whether it closes: curves cut into pieces about `seg`
    long, lines into pieces up to 3 * seg."""
    out, cur = [], []
    it = skia.Path.Iter(path, False)
    while True:
        verb, pts = it.next()
        if verb in (skia.Path.kMove_Verb, skia.Path.kClose_Verb, skia.Path.kDone_Verb):
            q = _plain(np.asarray(cur, float).reshape(-1, 2))
            closed = verb == skia.Path.kClose_Verb
            if closed and len(q) > 2 and np.allclose(q[0], q[-1], atol=1e-6):
                q = q[:-1]  # the line back to the start that closing drew
            if len(q) >= 2:
                out.append((q, closed and len(q) >= 3))
            cur = [(pts[0].fX, pts[0].fY)] if verb == skia.Path.kMove_Verb else []
            if verb == skia.Path.kDone_Verb:
                return out
            continue
        cur.extend(_flatten(verb, np.array([(q.fX, q.fY) for q in pts]), seg, it.conicWeight()))


def _plain(q: np.ndarray) -> np.ndarray:
    """Points without the ones that repeat the point before."""
    return q[np.r_[True, np.any(np.abs(np.diff(q, axis=0)) > 1e-6, axis=1)]] if len(q) else q


def lathe(lines: list[tuple[np.ndarray, bool]], seg: int, piece: float, crease: float = 40.0) -> Mesh:
    """Profile lines turned round the y axis in seg pieces. Each line is (n, 2) points [r, y], r out from the axis
    and y down, and whether it closes. Pieces longer than `piece` are cut, for the painter's sort; where the
    profile bends by more than `crease` degrees each side keeps its own normal. A line that is open and does not
    start and end on the axis has two sides, as a plane does."""
    meshes, cos = [], math.cos(math.radians(crease))
    for q, closed in lines:
        q = np.column_stack([np.maximum(q[:, 0], 0.0), q[:, 1]])
        loop = np.r_[q, q[:1]] if closed else q
        d = np.diff(loop, axis=0)
        cuts = np.maximum(1, np.ceil(np.hypot(*d.T) / piece)).astype(int)
        pts = _plain(np.concatenate([loop[i] + np.linspace(0, 1, k, endpoint=False)[:, None] * d[i]
                                     for i, k in enumerate(cuts)] + [loop[-1:]]))
        if len(pts) < 2:
            continue
        # Outward is to the right going down the outside: turn round a line drawn the other way, judged by the
        # outline it makes closed along the axis.
        if _area(pts if closed else np.r_[pts, [[0.0, pts[-1, 1]], [0.0, pts[0, 1]]]]) < 0:
            pts = pts[::-1]
        d = np.diff(pts, axis=0)
        en = np.stack([d[:, 1], -d[:, 0]], 1) / np.hypot(*d.T)[:, None]  # each piece's outward normal
        rows = [(pts[0], en[0])]
        for i in range(1, len(pts)):
            a = en[i - 1]
            if i == len(pts) - 1 and not closed:
                rows.append((pts[i], a))
                continue
            b = en[i % len(en)]
            if a @ b >= cos:  # a gentle bend: one normal between the two
                m = (a + b) / np.linalg.norm(a + b)
                rows.append((pts[i], m))
                if i == len(pts) - 1:  # where a closed line meets its start
                    rows[0] = (pts[0], m)
            elif i < len(pts) - 1:
                rows += [(pts[i], a), (pts[i], b)]
            else:
                rows.append((pts[i], a))
        P = np.array([p for p, _ in rows])
        N = np.array([n for _, n in rows])
        V = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        V /= max(float(V[-1]), 1e-9)
        meshes.append(_revolve(P[:, 0], P[:, 1], N[:, 0], N[:, 1], V, seg))
        if not closed and max(pts[0, 0], pts[-1, 0]) > 1e-6:
            meshes.append(_revolve(P[:, 0], P[:, 1], -N[:, 0], -N[:, 1], V, seg))
    return cat(meshes) if meshes else _EMPTY


# A tube's lines in space when it has no d, read in this order, and the defaults of a walk's and rays' keys.
ROUTES = ("points", "walk", "rays")
WALK = {"steps": 60, "step": 10, "seed": 0, "within": 0, "turn": 30}
RAYS = {"count": 12, "length": 100, "inner": 0, "seed": 0}


def tube(lines: list[tuple[np.ndarray, bool]], r: float, around: int, grow: float = 1.0,
         together: bool = False) -> Mesh:
    """A round tube of radius r along lines (each (n, 2) points in the xy plane, y down, or (n, 3) in space, and
    whether it closes), in `around` pieces round, with round ends where a line ends. Its rings are carried along
    each line without twisting (_carried). grow, from 0 to 1, draws the first share of the lines' whole length,
    the lines in turn, ending round where it stops; with together, that share of every line at once."""
    lines = [(np.c_[q, np.zeros(len(q))] if q.shape[1] == 2 else np.asarray(q, float), c) for q, c in lines]
    lens = [np.linalg.norm(np.diff(np.r_[q, q[:1]] if c else q, axis=0), axis=1) for q, c in lines]
    total = sum(float(n.sum()) for n in lens)
    g = max(0.0, min(1.0, grow))
    left = g * total
    al = np.linspace(0, 2 * np.pi, around + 1)
    ca, sa = np.cos(al)[None, :, None], np.sin(al)[None, :, None]
    meshes, before = [], 0.0
    for (q, closed), n in zip(lines, lens):
        start = before
        if together:
            left = g * float(n.sum())
            before += float(n.sum())
        if left <= 1e-9 or total <= 0:
            if together:
                continue
            break
        pts, s = (np.r_[q, q[:1]] if closed else q), np.r_[0.0, np.cumsum(n)]
        if left < s[-1] - 1e-9:  # it stops on this line
            k = int(np.searchsorted(s, left))
            f = (left - s[k - 1]) / max(float(n[k - 1]), 1e-12)
            pts, s, closed = np.r_[pts[:k], [pts[k - 1] + f * (pts[k] - pts[k - 1])]], np.r_[s[:k], left], False
        if not together:
            left -= s[-1]
        keep = np.r_[True, np.diff(s) > 1e-9]
        pts, s = pts[keep], s[keep] + start
        if not together:
            before = s[-1]
        if len(pts) < 2:
            continue
        d = np.diff(pts, axis=0)
        d /= np.linalg.norm(d, axis=1)[:, None]
        t_in, t_out = (np.r_[d[-1:], d], np.r_[d, d[:1]]) if closed else (np.r_[d[:1], d], np.r_[d, d[-1:]])
        t = t_in + t_out
        back = np.linalg.norm(t, axis=1) < 1e-6  # doubling straight back
        t[back] = t_out[back]
        t /= np.linalg.norm(t, axis=1)[:, None]
        miter = 1.0 / np.maximum((t * t_out).sum(1), 0.3)  # a bend keeps the tube's width across it
        side, up = _carried(pts, t, closed)
        bend = t_out - t_in  # the way the line turns, across it: the ring stretches that way at a bend
        bl = np.linalg.norm(bend, axis=1, keepdims=True)
        bend = np.where(bl > 1e-9, bend / np.maximum(bl, 1e-12), 0.0)[:, None]
        radial = ca * side[:, None] + sa * up[:, None]
        stretch = (miter[:, None, None] - 1) * (radial * bend).sum(-1, keepdims=True) * bend
        P = [pts[:, None] + r * (radial + stretch)]
        N, S = [radial], [np.repeat(s[:, None], around + 1, 1)]
        if not closed:  # round ends: rings shrinking to a point past each end
            ph = np.linspace(0, np.pi / 2, max(2, around // 4) + 1)[1:, None, None]
            for end, sign in ((0, -1.0), (-1, 1.0)):
                f = ph if sign > 0 else ph[::-1]
                rad = radial[end][None]
                ring = (pts[end] + sign * t[end] * r * np.sin(f) + r * np.cos(f) * rad,
                        np.cos(f) * rad + sign * np.sin(f) * t[end], np.full((len(f), around + 1), s[end]))
                for rows, part in zip((P, N, S), ring):
                    rows.insert(0 if sign < 0 else len(rows), part)
        P, N, S = np.concatenate(P), np.concatenate(N), np.concatenate(S)
        U = np.stack([S / max(total, 1e-9), np.broadcast_to(al / (2 * np.pi), S.shape)], -1)
        meshes.append(_orient(P.reshape(-1, 3), N.reshape(-1, 3), _grid(len(P) - 1, around), U.reshape(-1, 2)))
    return cat(meshes) if meshes else _EMPTY


def _carried(pts: np.ndarray, t: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """A tube's ring directions at each point of a line with tangents t: side, then up = t x side, carried from
    point to point without turning about the line (double reflection: Wang et al., "Computation of rotation
    minimizing frames", 2008). It starts with side across the line in the xy plane, so a line in that plane keeps
    up along z. Round a closed line the frame comes back turned; that turn is spread along it so the ends meet."""
    n = len(t)
    side = np.zeros((n, 3))
    s0 = np.cross([0.0, 0.0, 1.0] if abs(t[0, 2]) < 0.9 else [1.0, 0.0, 0.0], t[0])
    side[0] = s0 / np.linalg.norm(s0)
    for i in range(n - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-18:
            side[i + 1] = side[i]
            continue
        rl = side[i] - (2 / c1) * float(v1 @ side[i]) * v1
        tl = t[i] - (2 / c1) * float(v1 @ t[i]) * v1
        v2 = t[i + 1] - tl
        c2 = float(v2 @ v2)
        side[i + 1] = rl if c2 < 1e-18 else rl - (2 / c2) * float(v2 @ rl) * v2
    if closed and n > 2:
        a = math.atan2(float(np.cross(side[-1], side[0]) @ t[0]), float(side[-1] @ side[0]))
        if abs(a) > 1e-9:
            run = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            turn = (a * run / max(run[-1], 1e-12))[:, None]
            side = side * np.cos(turn) + np.cross(t, side) * np.sin(turn)
    side -= (side * t).sum(1, keepdims=True) * t
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
    return side, np.cross(t, side)


def smooth(q: np.ndarray, closed: bool, seg: float) -> np.ndarray:
    """Points about seg apart on a Catmull-Rom curve through the points q (n, 3), round to the start if closed."""
    if len(q) < 3:
        return q
    ext = np.r_[q[-1:], q, q[:2]] if closed else np.r_[2 * q[:1] - q[1:2], q, 2 * q[-1:] - q[-2:-1]]
    out = []
    for i in range(len(q) if closed else len(q) - 1):
        p0, p1, p2, p3 = ext[i:i + 4]
        k = max(1, math.ceil(float(np.linalg.norm(p2 - p1)) / seg))
        u = (np.arange(k) / k)[:, None]
        out.append(0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3))
    if not closed:
        out.append(q[-1:])
    return np.concatenate(out)


def walk(steps: int, step: float, seed: int, within: float, turn: float, start) -> np.ndarray:
    """(steps + 1, 3) points of a seeded random walk from start: each step `step` long and turned up to `turn`
    degrees from the last, kept within `within` of the start (0: anywhere) by turning back off that ball."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    o = np.asarray(start, float)
    p, d = o.copy(), rng.normal(size=3)
    d /= np.linalg.norm(d)
    pts = [p.copy()]
    for _ in range(int(steps)):
        a = rng.normal(size=3)
        a -= (a @ d) * d
        a /= max(float(np.linalg.norm(a)), 1e-12)
        ang = math.radians(turn) * float(rng.random())
        d = d * math.cos(ang) + a * math.sin(ang)
        if within > 0:
            out = p + d * step - o
            if np.linalg.norm(out) > within and d @ out > 0:  # it would leave: turn back off the wall
                nrm = out / np.linalg.norm(out)
                d = d - 2 * (d @ nrm) * nrm
        p = p + d * step
        if within > 0 and np.linalg.norm(p - o) > within:
            p = o + (p - o) * (within / np.linalg.norm(p - o))
        pts.append(p.copy())
    return np.array(pts)


def rays(count: int, length, inner: float, seed: int, start) -> list[np.ndarray]:
    """count straight lines out from start, spread evenly in every direction (a Fibonacci sphere turned by the
    seed), each from `inner` out to inner + its length: a number, or [least, most] picked by the seed."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    i = np.arange(count) + 0.5
    phi, th = np.arccos(1 - 2 * i / count), math.pi * (1 + 5 ** 0.5) * i
    dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    q, rr = np.linalg.qr(rng.normal(size=(3, 3)))
    dirs = dirs @ (q * np.sign(np.diag(rr))).T
    lo, hi = (length, length) if isinstance(length, (int, float)) else (length[0], length[1])
    ln = lo + (hi - lo) * rng.random(count)
    o = np.asarray(start, float)
    return [np.stack([o + d * inner, o + d * (inner + float(n))]) for d, n in zip(dirs, ln)]
