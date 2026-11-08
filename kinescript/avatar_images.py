"""Avatars from generated pictures: a base picture of a character and variants of it that change only the mouth or
the eyes, turned into an avatar folder that talks and blinks.

    source/base.png         the character, mouth at rest (shape X), eyes open, on a flat background or with alpha
    source/mouth-A.png ...  the same picture with only the mouth changed (A, C and D at least; X is the base)
    source/blink.png        the same picture with the eyes closed
    source/source.json      optional: name, voice, files, mouthBox, eyesBox, background

The character is never cut into limbs. The base becomes the avatar's one body part, and each variant gives a small
patch, feathered at its edge, that lies over the base where the mouth or the eyes are. Image models move a picture a
little between variants, so each one is lined up with the base first (shift and scale); a variant that also changed
something outside its patch is reported by name (avatar-source-drift), to be made again. Heads do not turn on their
own: gestures move the whole figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import mouth as M

SOURCE = "source.json"
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")
LOOKS = ("left", "right", "up", "down")  # eye states a gaze picks (avatar.LOOKS)
RESERVED = {"open", "closed", *LOOKS}  # eye states an expression can't be named
PROMPTS = "PROMPTS.md"
WORK = 512  # the longest side, in pixels, that lining up and finding regions work at
SCALES = np.linspace(0.96, 1.04, 9)
DRIFT_WARN, DRIFT_ERROR = 0.02, 0.08  # share of the figure outside a patch that a variant changed
MOUTHS = {
    "A": "lips pressed together, as for P, B and M",
    "B": "lips slightly apart, teeth together, as for S and EE",
    "C": "mouth open, as for EH in 'bed'",
    "D": "mouth wide open, jaw dropped, as for AA in 'father'",
    "E": "lips rounded, slightly open, as for AW in 'law'",
    "F": "lips puckered into a small circle, as for OO and W",
    "G": "upper teeth resting on the lower lip, as for F and V",
    "H": "mouth open with the tongue tip touching behind the upper teeth, as for L",
}


def _issue(level: str, code: str, message: str, fix: str = "") -> dict:
    return {"level": level, "code": code, "message": message, "fix": fix}


def read_image(path: Path) -> np.ndarray:
    """An image file as straight RGBA floats from 0 to 1, shape (h, w, 4)."""
    import skia
    data = Path(path).read_bytes()
    img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(data))
    if img is None:
        raise ValueError(f"cannot decode {path} (png, jpg and webp are supported)")
    px = img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
    return px.astype(np.float32) / 255.0


def write_image(px: np.ndarray, path: Path) -> None:
    from .output import save_png
    save_png(np.clip(np.rint(px * 255), 0, 255).astype(np.uint8), path)


# --- sampling -------------------------------------------------------------------------------------------------

def _sample(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """img at the points (sx, sy), bilinear, edges held."""
    h, w = img.shape[:2]
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _warp(img: np.ndarray, scale: float, shift: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    """The picture whose point p is img's point c + scale * (p + shift - c), c the centre: size (w, h)."""
    w, h = size
    ih, iw = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    kx, ky = iw / w, ih / h  # a variant of another size is stretched to the base's first
    sx = (cx + scale * (xx + shift[0] - cx)) * kx
    sy = (cy + scale * (yy + shift[1] - cy)) * ky
    return _sample(img, sx, sy)


def _small(img: np.ndarray, f: int) -> np.ndarray:
    """img shrunk by a whole factor f, averaging f x f blocks."""
    if f <= 1:
        return img
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    x = img[:h, :w]
    return x.reshape(h // f, f, w // f, f, *x.shape[2:]).mean(axis=(1, 3))


def _grey(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    """A box blur of radius r on a 2D array, edges held."""
    if r <= 0:
        return a
    k = 2 * r + 1
    for axis in (0, 1):
        p = np.pad(a, [(r, r) if i == axis else (0, 0) for i in range(2)], mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        a = ((np.take(c, range(k, c.shape[axis]), axis=axis) - np.take(c, range(0, c.shape[axis] - k), axis=axis))
             / k).astype(np.float32)
    return a


def _grow(seed: np.ndarray, allowed: np.ndarray, limit: int = 100000) -> np.ndarray:
    """The pixels of `allowed` connected (4-way) to `seed`."""
    reach = seed & allowed
    for _ in range(limit):
        n = reach.copy()
        n[1:] |= reach[:-1]
        n[:-1] |= reach[1:]
        n[:, 1:] |= reach[:, :-1]
        n[:, :-1] |= reach[:, 1:]
        n &= allowed
        if np.array_equal(n, reach):
            break
        reach = n
    return reach


# --- background -----------------------------------------------------------------------------------------------

def key(img: np.ndarray, background="auto") -> tuple[np.ndarray, list[dict]]:
    """The base picture with its background made transparent. A picture with its own alpha keeps it; otherwise the
    background is the colour of its border (or `background`, a hex colour), removed where it joins the border, so
    the same colour inside the figure (white teeth on white) stays."""
    issues: list[dict] = []
    if background == "alpha" or (background == "auto" and img[..., 3].min() < 0.98):
        return img.copy(), issues
    rgb = img[..., :3]
    # the top and sides: a head-and-shoulders picture runs off its bottom edge
    border = np.concatenate([rgb[:2].reshape(-1, 3), rgb[:, :2].reshape(-1, 3), rgb[:, -2:].reshape(-1, 3)])
    if isinstance(background, str) and background.startswith("#"):
        hx = background.lstrip("#")
        bg = np.array([int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)
    else:
        bg = np.median(border, axis=0).astype(np.float32)
        if float(np.mean(np.linalg.norm(border - bg, axis=1) > 0.12)) > 0.4:
            issues.append(_issue("error", "avatar-source-background",
                                 "the base picture's border is not one flat colour, so its background can't be keyed",
                                 "make the base on a plain background (such as #00FF00), or give it an alpha channel"))
            return img.copy(), issues
    lo, hi = 0.08, 0.22
    dist = np.linalg.norm(rgb - bg, axis=2)
    near = dist < hi
    f = max(1, max(img.shape[:2]) // WORK)
    small = _small(near.astype(np.float32), f) > 0.5
    seed = np.zeros_like(small)
    seed[0], seed[-1], seed[:, 0], seed[:, -1] = small[0], small[-1], small[:, 0], small[:, -1]
    reach = _grow(seed, small)
    reach[1:] |= reach[:-1].copy()  # one low pixel wider, then back to full size
    reach[:-1] |= reach[1:].copy()
    reach[:, 1:] |= reach[:, :-1].copy()
    reach[:, :-1] |= reach[:, 1:].copy()
    full = np.repeat(np.repeat(reach, f, 0), f, 1)
    full = np.pad(full, ((0, img.shape[0] - full.shape[0]), (0, img.shape[1] - full.shape[1])), mode="edge")
    zone = full & near
    alpha = np.where(zone, np.clip((dist - lo) / (hi - lo), 0, 1), 1.0).astype(np.float32)
    a = alpha[..., None]
    unmixed = np.where(a > 0.01, (rgb - (1 - a) * bg) / np.maximum(a, 0.01), rgb)  # take the background's tint out
    out = np.concatenate([np.clip(unmixed, 0, 1), alpha[..., None] * img[..., 3:4]], axis=2).astype(np.float32)
    if float(alpha.mean()) > 0.97:
        issues.append(_issue("warning", "avatar-source-background",
                             "almost none of the base picture was keyed as background",
                             "check the base has a plain border, or give source.json a \"background\" colour"))
    return out, issues


# --- lining up ------------------------------------------------------------------------------------------------

def _hann(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _phase(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """The shift d with b(p + d) ~ a(p), to a tenth of a pixel."""
    win = _hann(*a.shape)
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = np.conj(A) * B
    R /= np.maximum(np.abs(R), 1e-9)
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    h, w = r.shape

    def fine(c, m, n, along):
        v0, v1, v2 = (r[(c - 1) % n, m], r[c, m], r[(c + 1) % n, m]) if along == 0 else \
                     (r[m, (c - 1) % n], r[m, c], r[m, (c + 1) % n])
        d = v0 - 2 * v1 + v2
        return c + (0.5 * (v0 - v2) / d if abs(d) > 1e-12 else 0.0)

    dy = fine(py, px, h, 0)
    dx = fine(px, py, w, 1)
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dx), float(dy)


def align(base: np.ndarray, var: np.ndarray) -> tuple[float, tuple[float, float], np.ndarray]:
    """var lined up with base: (scale, shift in base pixels, the lined-up picture at base's size)."""
    h, w = base.shape[:2]
    f = max(1, max(h, w) // WORK)
    a = _small(_grey(base), f)
    sh, sw = a.shape
    v = _warp(var, 1.0, (0.0, 0.0), (w, h))  # at base size first
    vs = _small(_grey(v), f)
    best = None
    for s in SCALES:
        ws = _warp(vs[..., None], float(s), (0.0, 0.0), (sw, sh))[..., 0]
        d = _phase(a, ws)
        moved = _warp(vs[..., None], float(s), d, (sw, sh))[..., 0]
        err = float(np.mean(np.abs(moved - a)))
        if best is None or err < best[0]:
            best = (err, float(s), d)
    _, s, (dx, dy) = best
    shift = (dx * f, dy * f)
    return s, shift, _warp(v, s, shift, (w, h))


# --- regions --------------------------------------------------------------------------------------------------

def _diff(base: np.ndarray, var: np.ndarray, f: int) -> np.ndarray:
    d = np.abs(base[..., :3] - var[..., :3]).mean(axis=2)
    return _blur(_small(d, f), 2)


def region(base: np.ndarray, var: np.ndarray) -> tuple[list[int] | None, np.ndarray]:
    """Where a lined-up variant differs from the base most: the box [x0, y0, x1, y1] round the blob that holds the
    biggest change, in base pixels, and the work-size difference map."""
    h, w = base.shape[:2]
    f = max(1, max(h, w) // WORK)
    d = _diff(base, var, f)
    peak = float(d.max())
    if peak < 0.04:
        return None, d
    blob = d > max(0.035, 0.3 * peak)
    left, got = d.copy(), np.zeros_like(blob)
    for _ in range(6):  # each blob whose change is near the biggest: both eyes, not only one
        top = float(left.max())
        if top < 0.5 * peak:
            break
        seed = np.zeros_like(blob)
        seed[np.unravel_index(int(np.argmax(left)), left.shape)] = True
        one = _grow(seed, blob)
        got |= one
        left[one | ~blob] = 0
    ys, xs = np.nonzero(got)
    return [int(xs.min() * f), int(ys.min() * f), int((xs.max() + 1) * f), int((ys.max() + 1) * f)], d


def _union(boxes: list[list[int]]) -> list[int] | None:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _pad(box: list[int], size: tuple[int, int], k: float = 0.25, least: int = 8) -> list[int]:
    """box grown on every side by k of its larger side (at least `least` px), so the patch's feather lies outside
    what changed."""
    w, h = size
    p = max(least, int(max(box[2] - box[0], box[3] - box[1]) * k))
    return [max(0, box[0] - p), max(0, box[1] - p), min(w, box[2] + p), min(h, box[3] + p)]


def drift(d: np.ndarray, box: list[int], figure: np.ndarray, f: int) -> float:
    """The share of the figure, outside box, that a variant changed."""
    out = figure.copy()
    x0, y0, x1, y1 = (int(v // f) for v in box)
    out[y0:y1 + 1, x0:x1 + 1] = False
    n = int(out.sum())
    return float(((d > 0.1) & out).sum() / n) if n else 0.0


# --- patches --------------------------------------------------------------------------------------------------

def _ramp(n: int, f: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    e = np.minimum(i + 0.5, n - 0.5 - i) / max(f, 1)
    t = np.clip(e, 0, 1)
    return t * t * (3 - 2 * t)


def patch(base_keyed: np.ndarray, var: np.ndarray, box: list[int]) -> np.ndarray:
    """A full-size picture, clear except box, where it shows var feathered into the base at its edge, its colour
    matched to the base's on the feather."""
    h, w = base_keyed.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    f = max(3, min(int(0.15 * max(bw, bh)), bw // 3, bh // 3))
    m = np.outer(_ramp(bh, f), _ramp(bw, f))
    src = var[y0:y1, x0:x1, :3].copy()
    ref = base_keyed[y0:y1, x0:x1, :3]
    ring = (m > 0) & (m < 1) & (base_keyed[y0:y1, x0:x1, 3] > 0.5)
    if ring.sum() > 16:
        src = np.clip(src + (ref[ring].mean(axis=0) - src[ring].mean(axis=0)), 0, 1)
    out = np.zeros((h, w, 4), np.float32)
    out[y0:y1, x0:x1, :3] = src
    out[y0:y1, x0:x1, 3] = m * base_keyed[y0:y1, x0:x1, 3]
    return out


# --- the folder -----------------------------------------------------------------------------------------------

def _files(src: Path, spec: dict) -> dict[str, Path]:
    """The source pictures by role: base, blink, mouth-A ... mouth-H, look-left ... look-down, expr-NAME."""
    given = spec.get("files") or {}
    out = {}
    exprs = sorted({q.stem for q in src.iterdir() if q.is_file() and q.stem.startswith("expr-")
                    and q.suffix.lower() in IMAGE_EXTS} | {r for r in given if r.startswith("expr-")})
    for role in ["base", "blink", *[f"mouth-{s}" for s in M.SHAPES if s != "X"], *[f"look-{d}" for d in LOOKS],
                 *exprs]:
        if role in given:
            out[role] = src / given[role]
            continue
        for ext in IMAGE_EXTS:
            if (src / f"{role}{ext}").is_file():
                out[role] = src / f"{role}{ext}"
                break
    return out


def _gestures(w: float, h: float) -> dict:
    return {
        "nod": {"dur": 0.6, "body": {"y": [[0, 0], [0.15, round(0.012 * h, 1)], [0.3, 0], [0.45, round(0.007 * h, 1)],
                                           [0.6, 0]], "rotate": [[0, 0], [0.15, 0.8], [0.3, 0]]}},
        "shake": {"dur": 0.8, "body": {"rotate": [[0, 0], [0.15, -1.6], [0.35, 1.6], [0.55, -1], [0.8, 0]]}},
        "tilt": {"dur": 1.0, "body": {"rotate": [[0, 0], [0.3, -2.5], [0.7, -2.5], [1.0, 0]]}},
        "lean": {"dur": 1.2, "body": {"y": [[0, 0], [0.3, round(0.02 * h, 1)], [0.9, round(0.02 * h, 1)], [1.2, 0]],
                                      "rotate": [[0, 0], [0.3, 1.5], [0.9, 1.5], [1.2, 0]]}},
    }


def init(src: str | Path) -> dict:
    """Write source.json and PROMPTS.md into src (made if missing), keeping files already there."""
    src = Path(src)
    src.mkdir(parents=True, exist_ok=True)
    wrote, kept = [], []
    spec = {"name": src.name.title(), "voice": {"backend": "kokoro", "speaker": "af_heart"}, "background": "auto"}
    lines = ["# Pictures for this avatar", "",
             "Make base.png first. Make every other picture from it, as an edit of that image (or with it as the "
             "reference), at the same size and framing. Change only what the line says; keep the pose, hair, "
             "clothes, lighting and background exactly the same.", "",
             "- base.png: the character facing the camera, head and shoulders, mouth gently closed and relaxed, eyes "
             "open looking at the camera, on a plain flat #00FF00 background (or a transparent one).",
             "- blink.png: the same picture with both eyes fully closed.",
             *[f"- mouth-{s}.png: the same picture, {d}." for s, d in MOUTHS.items()], "",
             "mouth-A, mouth-C and mouth-D are required; the others fall back to the nearest of those.", "",
             "Optional, for gaze and expressions (eyes and brows only; speech owns the mouth):",
             *[f"- look-{d}.png: the same picture with only the eyes looking {d}." for d in LOOKS],
             "- expr-NAME.png (such as expr-happy.png, expr-surprise.png, expr-think.png): the same picture with "
             "only the eyes and brows changed to show that feeling; keep the mouth closed and relaxed.",
             "Then run: kine avatar from-images " + src.as_posix()]
    for name, text in ((SOURCE, json.dumps(spec, indent=2) + "\n"), (PROMPTS, "\n".join(lines) + "\n")):
        p = src / name
        if p.exists():
            kept.append(str(p))
        else:
            p.write_text(text, encoding="utf-8")
            wrote.append(str(p))
    return {"ok": True, "wrote": wrote, "kept": kept}


def from_images(src: str | Path, out: str | Path | None = None, sheet: bool = True) -> dict:
    """Build an avatar folder at out (default src/avatar) from the pictures in src. Returns the report kine prints:
    ok, out, size, boxes, each variant's scale, shift and drift, issues, and the pose sheet's path."""
    from . import avatar as A
    src = Path(src)
    out = Path(out) if out else src / "avatar"
    issues: list[dict] = []
    spec = {}
    if (src / SOURCE).is_file():
        try:
            spec = json.loads((src / SOURCE).read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            return {"ok": False, "issues": [_issue("error", "avatar-source", f"{src / SOURCE} is not JSON: {e}")]}
    files = _files(src, spec)
    if "base" not in files:
        return {"ok": False, "issues": [_issue("error", "avatar-source", f"no base picture in {src}",
                                               f"add base.png, or run kine avatar from-images {src} --init")]}
    for p in files.values():
        if not p.is_file():
            issues.append(_issue("error", "avatar-source", f"'{p}' is not a file"))
    missing = [s for s in M.REQUIRED if s != "X" and f"mouth-{s}" not in files]
    if missing:
        issues.append(_issue("error", "avatar-missing-shape", "no picture for mouth shape"
                             f"{'s' * (len(missing) > 1)} {', '.join(missing)}",
                             "add " + ", ".join(f"mouth-{s}.png" for s in missing) + " (see PROMPTS.md)"))
    if "blink" not in files:
        issues.append(_issue("error", "avatar-source", "no blink picture", "add blink.png: the base with eyes closed"))
    if any(i["level"] == "error" for i in issues):
        return {"ok": False, "issues": issues}

    raw = read_image(files["base"])
    h, w = raw.shape[:2]
    base, got = key(raw, spec.get("background", "auto"))
    issues += got
    if any(i["level"] == "error" for i in issues):
        return {"ok": False, "issues": issues}
    f = max(1, max(h, w) // WORK)
    figure = _small(base[..., 3], f) > 0.5

    lined, report, boxes = {}, {}, {}
    for role in [r for r in files if r != "base"]:
        s, shift, v = align(raw, read_image(files[role]))
        lined[role] = v
        report[role] = {"scale": round(s, 4), "shift": [round(shift[0], 1), round(shift[1], 1)]}
        box, d = region(raw, v)
        boxes[role] = (box, d)
        if box is None:
            issues.append(_issue("warning", "avatar-source-same", f"{files[role].name} looks the same as the base",
                                 "make it again so that the mouth or eyes change as PROMPTS.md says"))

    def chosen(name: str, roles: list[str]) -> list[int] | None:
        given = spec.get(f"{name}Box")
        if given:
            x, y, bw, bh = (int(v) for v in given)
            return [x, y, x + bw, y + bh]
        found = [boxes[r][0] for r in roles if r in boxes and boxes[r][0]]
        if not found:
            return None
        mid = [float(np.median([b[i] for b in found])) for i in range(4)]  # a variant that changed more can't widen it
        agree = [b for b in found if mid[0] <= (b[0] + b[2]) / 2 <= mid[2] and mid[1] <= (b[1] + b[3]) / 2 <= mid[3]]
        return _pad(_union(agree or found), (w, h))

    mouths = [r for r in files if r.startswith("mouth-")]
    mbox = chosen("mouth", mouths)
    looks = [r for r in files if r.startswith("look-")]
    ebox = chosen("eyes", ["blink", *looks])
    for name, box in (("mouth", mbox), ("eyes", ebox)):
        if box is None:
            issues.append(_issue("error", "avatar-source-region", f"could not find where the {name} changes",
                                 f"give source.json \"{name}Box\": [x, y, w, h] in the base's pixels"))
    if mbox and ebox and not (mbox[2] <= ebox[0] or ebox[2] <= mbox[0] or mbox[3] <= ebox[1] or ebox[3] <= mbox[1]):
        issues.append(_issue("warning", "avatar-source-region", "the mouth and eye patches overlap",
                             "a variant changed more than its feature; check the sheet, or give mouthBox and eyesBox"))
    exprs = {r: r[5:] for r in files if r.startswith("expr-")}
    for role, name in exprs.items():
        if name in RESERVED or not name:
            issues.append(_issue("error", "avatar-source", f"{files[role].name}: an expression can't be called "
                                 f"'{name}'", "those names are the eyes' own drawings; pick another name"))
    if any(i["level"] == "error" for i in issues):
        return {"ok": False, "issues": issues, "files": report}

    own = {}  # each expression's patch: the eyes' box and what it changed above the mouth
    for role in exprs:
        found = boxes[role][0]
        box = _pad(_union([found, ebox]) if found else ebox, (w, h), k=0.1)
        box[3] = max(box[1] + 4, min(box[3], mbox[1]))
        own[role] = box

    for role in report:
        box = own.get(role) or (ebox if role == "blink" or role.startswith("look-") else mbox)
        share = drift(boxes[role][1], box, figure, f)
        report[role]["drift"] = round(share, 4)
        if share > DRIFT_WARN:
            issues.append(_issue("error" if share > DRIFT_ERROR else "warning", "avatar-source-drift",
                                 f"{files[role].name} also changed {share:.0%} of the figure outside its patch",
                                 "make it again from the base, changing only the "
                                 + ("mouth" if role.startswith("mouth-") else "eyes and brows"
                                    if role.startswith("expr-") else "eyes")))

    parts_dir = out / "parts"
    write_image(base, parts_dir / "body.png")
    write_image(patch(base, raw, ebox), parts_dir / "eyes-open.png")
    write_image(patch(base, lined["blink"], ebox), parts_dir / "eyes-closed.png")
    states = {"open": "parts/eyes-open.png", "closed": "parts/eyes-closed.png"}
    for role in looks:
        write_image(patch(base, lined[role], ebox), parts_dir / f"eyes-{role[5:]}.png")
        states[role[5:]] = f"parts/eyes-{role[5:]}.png"
    for role, name in exprs.items():
        write_image(patch(base, lined[role], own[role]), parts_dir / f"eyes-{name}.png")
        states[name] = f"parts/eyes-{name}.png"
    shapes = {"X": "parts/mouth-X.png"}
    write_image(patch(base, raw, mbox), parts_dir / "mouth-X.png")
    for role in mouths:
        s = role[-1]
        write_image(patch(base, lined[role], mbox), parts_dir / f"mouth-{s}.png")
        shapes[s] = f"parts/mouth-{s}.png"
    centre = lambda b: [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)]  # noqa: E731
    av = {"kinescriptAvatar": A.VERSION, "name": spec.get("name") or src.name.title(),
          "description": spec.get("description", f"made by kine avatar from-images from {src.name}"),
          "size": [w, h],
          "parts": {"body": {"src": "parts/body.png", "pivot": [w / 2, float(h)]},
                    "eyes": {"parent": "body", "pivot": centre(ebox),
                             "states": states},
                    "mouth": {"parent": "body", "pivot": centre(mbox),
                              "shapes": {s: shapes[s] for s in M.SHAPES if s in shapes}}},
          **({"expressions": {name: {"eyes": {"state": name}} for name in exprs.values()}} if exprs else {}),
          "gestures": _gestures(w, h),
          "idle": {"blink": [2.5, 5], "breath": 0.008, "saccade": [0.8, 2.5]},
          "voice": spec.get("voice", {"backend": "kokoro", "speaker": "af_heart"})}
    (out / A.FILE).write_text(json.dumps(av, indent=2) + "\n", encoding="utf-8")
    checked = A.validate(out)
    issues += checked
    ok = not any(i["level"] == "error" for i in issues)
    result = {"ok": ok, "out": str(out), "size": [w, h],
              "boxes": {"mouth": [mbox[0], mbox[1], mbox[2] - mbox[0], mbox[3] - mbox[1]],
                        "eyes": [ebox[0], ebox[1], ebox[2] - ebox[0], ebox[3] - ebox[1]]},
              "files": report, "issues": issues}
    if sheet and not any(i["level"] == "error" for i in checked):
        result["sheet"] = A.sheet(out, str(out / "avatar_sheet.png"))["out"]
    return result
