"""Depth for layers: 4x4 matrices in canvas pixels with x right, y down and z toward the viewer, the
convention CSS transforms use, and the camera that projects them onto the canvas."""
from __future__ import annotations

import math

import numpy as np
import skia

DEFAULT_FOV = 40.0
# A point is drawn only while its w (distance from the camera over the focus distance) is above
# this: closer than 1/20 of the way, a layer is magnified 20x and soon passes behind the lens.
NEAR = 0.05


def translate(x: float, y: float, z: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = x, y, z
    return m


def scale(sx: float, sy: float) -> np.ndarray:
    return np.diag([sx, sy, 1.0, 1.0])


def rotate_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], float)


def rotate_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], float)


def rotate_z(deg: float) -> np.ndarray:
    """Clockwise on screen, like Canvas.rotate."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def tilt(z: float, rx: float, ry: float) -> np.ndarray:
    """The depth part of a layer transform, applied between its translate and its rotate."""
    return translate(0, 0, z) @ rotate_x(rx) @ rotate_y(ry)


def camera(w: float, h: float, fov: float = DEFAULT_FOV, x: float = 0.0, y: float = 0.0, z: float = 0.0,
           rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> np.ndarray:
    """World to canvas. The default camera looks at the canvas centre from the distance at which
    `fov` (vertical, degrees) spans the canvas height, so z=0 maps to itself. x, y pan the camera
    (and the point it orbits); z moves it back; ry orbits it right, rx up (looking down), rz rolls it."""
    orbit = rotate_z(-rz) @ rotate_x(-rx) @ rotate_y(-ry)
    return lens(w, h, fov) @ translate(0, 0, -z) @ orbit @ translate(-w / 2 - x, -h / 2 - y)


def lens(w: float, h: float, fov: float = DEFAULT_FOV) -> np.ndarray:
    """The projection part of `camera`: camera space (origin at the canvas centre) to canvas."""
    focus = (h / 2) / math.tan(math.radians(min(max(fov, 1.0), 179.0)) / 2)
    persp = np.eye(4)
    persp[3, 2] = -1.0 / focus
    return translate(w / 2, h / 2) @ persp


def to_skia(m: np.ndarray) -> skia.M44:
    return skia.M44.Rows(*(skia.V4(*row) for row in m.tolist()))


def project(m: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """Canvas positions and w of a local rect's corners (TL, TR, BR, BL) at z=0."""
    bx, by, bw, bh = box
    bw, bh = bw or 1.0, bh or 1.0
    pts = np.array([[bx, by, 0, 1], [bx + bw, by, 0, 1], [bx + bw, by + bh, 0, 1], [bx, by + bh, 0, 1]], float)
    out = pts @ m.T
    w = out[:, 3]
    return out[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)[:, None], w


def facing(xy: np.ndarray) -> bool:
    """True while the rect whose projected corners are `xy` shows its front."""
    (x0, y0), (x1, y1), _, (x3, y3) = xy
    return (x1 - x0) * (y3 - y0) - (y1 - y0) * (x3 - x0) >= 0
