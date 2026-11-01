"""Morphing between any two paths. The pair is matched once: each of its contours is resampled by length to the same
number of points, its start turned to where the points move least, and a contour with no partner grows from (or
shrinks into) the middle of the nearest one. The two matched paths have the same verbs, so a frame between them is
one Path.interpolate."""
from __future__ import annotations

import math

import numpy as np
import skia

SPACING = 1.5  # px between the points a frame between two keys is drawn through
FEWEST, MOST = 16, 2048  # points on one contour


class Contour:
    """One contour of a path, on its own, with its length, whether it closes, and its turn and middle (from 64 points
    along it: the turn is the signed area, positive clockwise on screen)."""
    __slots__ = ("path", "length", "closed", "area", "middle")

    def __init__(self, path: skia.Path, length: float, closed: bool):
        self.path, self.length, self.closed = path, length, closed
        pts = sample(self, 64, closed)
        x, y = pts[:, 0], pts[:, 1]
        self.area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) if closed else 0.0
        self.middle = pts.mean(axis=0)


def contours(path: skia.Path) -> list[Contour]:
    out = []
    meas = skia.PathMeasure(path, False)
    while True:
        n = meas.getLength()
        if n > 1e-6:
            one = skia.Path()
            meas.getSegment(0, n, one, True)
            if meas.isClosed():
                one.close()
            out.append(Contour(one, n, meas.isClosed()))
        if not meas.nextContour():
            break
    return out


def sample(c: Contour, n: int, closed: bool) -> np.ndarray:
    """n points spread along c by length: a closed one round to where it started, an open one end to end."""
    meas = skia.PathMeasure(c.path, False)
    total = meas.getLength()
    step = total / n if closed else total / max(1, n - 1)
    pts = np.empty((n, 2))
    last = (0.0, 0.0)
    for k in range(n):
        hit = meas.getPosTan(min(k * step, total))
        if hit:
            last = (hit[0].x(), hit[0].y())
        pts[k] = last
    return pts


def _turned(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """b rolled so that its points move least to a's: tried every 64th of the way round, then near the best."""
    n = len(a)

    def cost(k: int) -> float:
        return float(((a - np.roll(b, -k, axis=0)) ** 2).sum())
    step = max(1, n // 64)
    best = min(range(0, n, step), key=cost)
    best = min(range(best - step, best + step + 1), key=cost)
    return np.roll(b, -(best % n), axis=0)


def _pairs(ca: list[Contour], cb: list[Contour]) -> list[tuple[Contour | None, Contour | None, Contour]]:
    """(a, b, nearest): contours paired largest with largest, then each one left over with no partner (None) and
    the contour on the other side it grows from or shrinks into."""
    size = lambda c: (abs(c.area), c.length)  # noqa: E731
    ra, rb = sorted(ca, key=size, reverse=True), sorted(cb, key=size, reverse=True)
    n = min(len(ra), len(rb))
    out = [(a, b, b) for a, b in zip(ra[:n], rb[:n])]

    def nearest(c: Contour, side: list[Contour]) -> Contour:  # with nothing on the other side, its own middle
        return min(side, key=lambda o: float(((o.middle - c.middle) ** 2).sum())) if side else c
    out += [(a, None, nearest(a, rb)) for a in ra[n:]]
    out += [(None, b, nearest(b, ra)) for b in rb[n:]]
    return out


def match(a: skia.Path, b: skia.Path) -> tuple[skia.Path, skia.Path]:
    """a and b drawn with the same verbs: the pair that a frame between them interpolates."""
    ca, cb = contours(a), contours(b)
    if not ca and not cb:
        return a, a
    # If the largest shapes turn opposite ways, every contour of b is turned round, so the shapes keep their turns
    # to one another (a hole stays a hole) and b's do not cross over a's.
    flip = bool(ca and cb) and max(ca, key=lambda c: abs(c.area)).area * max(cb, key=lambda c: abs(c.area)).area < 0
    pa, pb = skia.Path(), skia.Path()
    for x, y, near in _pairs(ca, cb):
        n = min(MOST, max(FEWEST, math.ceil(max(c.length for c in (x, y) if c is not None) / SPACING)))
        closed = (x is None or x.closed) and (y is None or y.closed)
        if x is None:
            xs, ys = np.repeat(near.middle[None], n, axis=0), sample(y, n, closed)
        elif y is None:
            xs, ys = sample(x, n, closed), np.repeat(near.middle[None], n, axis=0)
        else:
            xs, ys = sample(x, n, closed), sample(y, n, closed)
        if flip and y is not None:
            ys = ys[::-1].copy()
        if closed and x is not None and y is not None:
            ys = _turned(xs, ys)
        elif x is not None and y is not None and ((xs - ys[::-1]) ** 2).sum() < ((xs - ys) ** 2).sum():
            ys = ys[::-1].copy()
        pa.addPoly([skia.Point(float(p[0]), float(p[1])) for p in xs], closed)
        pb.addPoly([skia.Point(float(p[0]), float(p[1])) for p in ys], closed)
    pa.setFillType(a.getFillType())
    pb.setFillType(a.getFillType())
    return pa, pb
