"""Meshes for model layers. Each primitive is built once at unit size with numpy and scaled per frame,
so sizes animate for free. Coordinates follow space.py: x right, y down, z toward the viewer."""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np

from . import space

SHAPES = ("cube", "sphere", "cylinder", "cone", "torus", "plane")
DETAIL = {"low": 0.5, "medium": 1.0, "high": 2.0}


class Mesh(NamedTuple):
    P: np.ndarray  # (n, 3) positions
    N: np.ndarray  # (n, 3) unit normals
    T: np.ndarray  # (m, 3) triangles, wound so a face toward the camera is clockwise on screen (y down)
    M: np.ndarray  # (n,) material index of each vertex


def _grid(nu: int, nv: int, base: int = 0) -> np.ndarray:
    """Two triangles per cell of a (nu + 1) x (nv + 1) vertex grid stored row by row in u."""
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a = base + i * (nv + 1) + j
    b, c, d = a + nv + 1, a + nv + 2, a + 1
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])


def _fan(n: int, centre: int, ring: int) -> np.ndarray:
    k = np.arange(n)
    return np.stack([np.full(n, centre), ring + k, ring + k + 1], -1)


def _orient(P, N, T) -> Mesh:
    """Wind every triangle so its face normal agrees with its vertex normals; drop degenerate ones."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    keep = np.linalg.norm(fn, axis=1) > 1e-9
    T, fn = T[keep], fn[keep]
    flip = (fn * (N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]])).sum(1) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    return Mesh(P.astype(float), N / np.linalg.norm(N, axis=1, keepdims=True), T.astype(np.int32),
                np.zeros(len(P), np.int32))


def cat(meshes) -> Mesh:
    base = np.cumsum([0] + [len(m.P) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.P for m in meshes]), np.concatenate([m.N for m in meshes]),
                np.concatenate([m.T + b for m, b in zip(meshes, base)]), np.concatenate([m.M for m in meshes]))


def _surface(fn, nu: int, nv: int):
    """Vertices of a parametric surface over u, v in [0, 1] (both ends included)."""
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
    P, N = fn(u.ravel(), v.ravel())
    return P, N, _grid(nu, nv)


def _disc(y: float, n: int, normal_y: float):
    a = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.stack([np.cos(a), np.full(n + 1, y), np.sin(a)], -1)
    P = np.concatenate([[[0, y, 0]], ring])
    return P, np.tile([0, normal_y, 0], (n + 2, 1)), _fan(n, 0, 1)


@lru_cache(maxsize=64)
def unit(shape: str, detail: str = "medium", tube: float = 0.35) -> Mesh:
    """`shape` at unit size: cube and plane span -0.5..0.5, round shapes have radius 1 and height 1."""
    k = DETAIL.get(detail, 1.0)
    seg = max(6, round(24 * k))
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
            faces.append(_orient(P, N, _grid(n, n)))
        return cat(faces)
    if shape == "sphere":
        def fn(u, v):
            th, ph = u * 2 * np.pi, v * np.pi
            P = np.stack([np.sin(ph) * np.cos(th), -np.cos(ph), np.sin(ph) * np.sin(th)], -1)
            return P, P
        return _orient(*_surface(fn, 2 * seg, seg))
    if shape == "torus":
        def fn(u, v):
            th, ph = u * 2 * np.pi, v * 2 * np.pi
            N = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)
            P = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1) + tube * N
            return P, N
        return _orient(*_surface(fn, round(2.6 * seg), seg))
    if shape in ("cylinder", "cone"):
        cone = shape == "cone"

        def fn(u, v):
            th = u * 2 * np.pi
            r = v if cone else np.ones_like(v)
            P = np.stack([r * np.cos(th), v - 0.5, r * np.sin(th)], -1)
            N = np.stack([np.cos(th), np.full_like(th, -1.0 if cone else 0.0), np.sin(th)], -1)
            return P, N
        side = _orient(*_surface(fn, 2 * seg, max(1, seg // 6)))
        caps = [_orient(*_disc(0.5, 2 * seg, 1.0))] + ([] if cone else [_orient(*_disc(-0.5, 2 * seg, -1.0))])
        return cat([side] + caps)
    raise ValueError(f"unknown shape '{shape}'. Known: {', '.join(SHAPES)}")


def size(shape: str, get) -> tuple[tuple[float, float, float], float]:
    """(x, y, z) scale of the unit shape and the torus tube ratio, from w/h/depth/r/tube via get(key, default)."""
    if shape in ("cube", "plane"):
        w = float(get("w", 200))
        return (w, float(get("h", w)), float(get("depth", w)) if shape == "cube" else 1.0), 0.35
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
    return Mesh(P, N / np.linalg.norm(N, axis=1, keepdims=True), m.T, m.M)


def part_matrix(p: dict) -> np.ndarray:
    """A part's placement inside its model, in the same order as a layer transform."""
    f = lambda k, d=0.0: float(p.get(k, d))  # noqa: E731
    sc = f("scale", 1.0)
    return (space.translate(f("x"), f("y")) @ space.tilt(f("z"), f("rotateX"), f("rotateY"))
            @ space.rotate_z(f("rotate")) @ np.diag([sc, sc, sc, 1.0]))


def merge(meshes: list[Mesh], materials: list[int]) -> Mesh:
    """One mesh from several, each vertex tagged with its mesh's material slot."""
    return cat([m._replace(M=np.full(len(m.P), k, np.int32)) for m, k in zip(meshes, materials)])


def build_order(m: Mesh) -> np.ndarray:
    """Triangle indices bottom to top, the order `draw` and the build motion reveal them in."""
    return np.argsort(-m.P[m.T, 1].mean(1), kind="stable")
