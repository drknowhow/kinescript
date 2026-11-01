"""Agent-side verification: contact sheets, layout lint and frame probes.

An agent cannot watch a video. These turn a scene into things it can read: one image of
many timestamps, a JSON list of problems with time ranges, and the exact box of every layer.
"""
from __future__ import annotations

import bisect
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import skia

from . import color as C
from . import listen, media, models, stage
from .props import HOLD
from .render import CPU_BLUR_S, FONT_DIR, Renderer, _argb, joining, loop_end
from .scene import Scene, SceneError, load

# Model triangles one frame may draw before check warns. 100,000 took 71 ms a frame at 1080p on
# the GPU in the 3D spike (2026-10-29), against about 15 ms at 20,000.
TRIANGLE_BUDGET = 60_000
SLOW_EFFECTS = 0.25  # s a frame on the CPU for offscreen effects, past which check warns slow-effects
# Layers check warns about when more than half of one stays off the canvas for MEDIA_OFF_FOR seconds; a slide or
# a transition passes through faster.
MEDIA = ("video", "image", "shader")
MEDIA_OFF_FOR = 1.0
# Seconds a text of one or two words needs on screen; shorter, and it is a flash nobody reads. Quick kinetic type
# (the kinetic component's default is 0.6 s a word) stays above it.
FLASH = 0.4


MOVES = ("x", "y", "z", "rotate", "rotateX", "rotateY", "scale", "scaleX", "scaleY")  # keys whose stops stop-and-go finds


def _at_rest(ease, end: bool) -> bool:
    """Whether the ease's speed is near zero at its start (or end): next to 1% of a linear ease's."""
    return (1.0 - ease(0.999) if end else ease(0.001)) < 1e-5


def stops(layer) -> list[tuple[str, float]]:
    """(key, time) of each inner keyframe where a moving key of the layer comes to rest between two moves the same
    way, because the ease into the key slows to it and the ease out of it starts from rest."""
    out = []
    for k in MOVES:
        p = layer.props.get(k)
        if p is None or p.keys is None or isinstance(layer.raw.get(k), dict) and layer.raw[k].get("smooth"):
            continue
        pts = p.keys(layer.dur)
        for (t0, v0, _), (t1, v1, e1), (t2, v2, e2) in zip(pts, pts[1:], pts[2:]):
            if not (isinstance(v1, (int, float)) and t0 < t1 < t2 and (v1 - v0) * (v2 - v1) > 0) or HOLD in (e1, e2):
                continue
            if _at_rest(e1, True) and _at_rest(e2, False):
                out.append((k, t1))
                break
    return out


STROBE = 40 / 1080  # px a frame, as a share of the canvas's short side, past which a move shows as copies
STROBE_RUN = 5  # frames running that make it a move: not a cut, nor the first frames of a quick ease out


def strobes(scene: Scene, r: Renderer) -> list[tuple[str, float, float]]:
    """(layer id, time, px a frame) where a layer's anchor, moved by its x, y and along, goes further than STROBE each
    frame for STROBE_RUN frames running. Motion blur smears such a move, so a scene with it has none."""
    if int((scene.motion_blur or {}).get("samples", 0)) >= 2:
        return []
    fps, limit, out = scene.fps, STROBE * min(scene.w, scene.h), []
    for L in scene.all_layers():
        if L.along is None and all(L.props[k].static for k in ("x", "y") if k in L.props):
            continue
        prev, run, fast = None, 0, 0.0
        try:
            for f in range(math.ceil(L.t_in * fps), min(scene.frames, math.ceil(L.t_out * fps))):
                T = f / fps
                t = T - L.t_in
                x, y = L.placed(t, T)[:2]
                x, y = float(x), float(y)
                if L.along is not None:
                    ax, ay, _ = r._along(L, t, T)
                    x, y = x + ax, y + ay
                step = 0.0 if prev is None else math.hypot(x - prev[0], y - prev[1])
                prev = (x, y)
                if step <= limit:
                    run, fast = 0, 0.0
                    continue
                run, fast = run + 1, max(fast, step)
                if run == STROBE_RUN:
                    out.append((L.id, (f - STROBE_RUN) / fps, fast))
                    break
        except (SceneError, ArithmeticError, ValueError, TypeError):
            continue  # an expression that fails is reported where the frame is drawn
    return out


def _times(scene: Scene, count: int, times=None) -> list[float]:
    if times:  # as still reads them: negative counts from the end, and the scene's ends clamp
        from .output import frame_at
        return [frame_at(scene, t) / scene.fps for t in times]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count must be a whole number of frames, 1 or more, got {count!r}")
    step = scene.duration / count
    return [min(scene.duration - 1 / scene.fps, step * (k + 0.5)) for k in range(count)]


def contact_sheet(src, out: str, count: int = 12, cols: int | None = None, width: int = 480,
                  times=None, boxes: bool = False, audio: bool = False, wire: bool = False,
                  depth: bool = False) -> dict:
    """One PNG of `count` evenly spaced frames, each labelled with its time and frame number.
    boxes=True outlines every visible layer with its id. audio=True adds the sound under the frames, one lane per
    role and one for the mix, and returns its measurements. wire=True draws every model's drawn triangles over
    it; depth=True dims the frame and shows each stage's depth image on it, nearer brighter."""
    from .output import IMAGE, at_note, out_path, save_image
    scene = load(src)
    try:
        ts = _times(scene, count, times)
        notes = [m for m in (at_note(scene, t) for t in times or []) if m]  # a time outside the scene, clamped
        if cols is not None and (isinstance(cols, bool) or not isinstance(cols, int) or cols < 1):
            raise ValueError(f"cols must be a whole number, 1 or more, got {cols!r}")
        if isinstance(width, bool) or not isinstance(width, int) or not 16 <= width <= 4096:
            raise ValueError(f"width must be a whole number of pixels from 16 to 4096, got {width!r}")
        cols = min(cols or 4, len(ts))
        out = str(out_path(scene, out, IMAGE))
    except ValueError:
        scene.close()
        raise
    rows = math.ceil(len(ts) / cols)
    scale = width / scene.w
    r = Renderer(scene, track=boxes, scale=scale)
    r.wire, r.depth_view = wire, depth
    staged = 0  # frames in which a stage drew its depth
    tw, th = r.w, r.h
    gap, label_h = 8, 26
    rep = listen.report(scene) if audio else None
    grid_h = rows * (th + label_h + gap) + gap
    sheet = skia.Surface(cols * (tw + gap) + gap, grid_h + listen.strip_height(rep))
    c = sheet.getCanvas()
    c.clear(_argb("#0a0c10"))
    mono = skia.Typeface.MakeFromFile(str(FONT_DIR / "IBMPlexMono-Medium.ttf"))
    font = skia.Font(mono, 15)
    small = skia.Font(mono, 11)
    for k, T in enumerate(ts):
        n = int(round(T * scene.fps))
        arr = r.frame(n)
        img = skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType)
        x = gap + (k % cols) * (tw + gap)
        y = gap + (k // cols) * (th + label_h + gap)
        c.drawImage(img, x, y + label_h)
        if depth:
            staged += bool(r.depths)
            _draw_depth(c, r.depths, x, y + label_h, tw, th)
        if wire and r.wires:
            _draw_wire(c, r.wires, x, y + label_h, tw, th)
        c.drawString(f"{T:6.2f}s  f{n}", x + 2, y + 18, font, skia.Paint(AntiAlias=True, Color=_argb("#f2f0e8")))
        if boxes:
            _draw_boxes(c, scene, r.boxes, x, y + label_h, tw, th, small)
    if rep is not None:
        listen.strip(c, rep, listen.issues(scene, rep), gap, grid_h, cols * (tw + gap) - gap, ts, font, small)
    scene.close()
    save_image(sheet.makeImageSnapshot(), Path(out))
    info = {"out": out, "times": [round(t, 3) for t in ts], "grid": [cols, rows], "backend": r.backend}
    if audio:
        info["audio"] = listen.summary(rep)
    if depth and not staged:
        notes.append('no stage drew in these frames: --depth shows the depth of groups with "sort": "pixel"')
    if notes:
        info["notes"] = notes
    return info


def _draw_wire(c, wires: list, ox: float, oy: float, tw: float, th: float):
    """Every model's drawn triangle edges, as hairlines over its thumbnail."""
    from .render3d import _points
    paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0, Color=_argb("#b9f45dc0"))
    c.save()
    c.clipRect(skia.Rect.MakeXYWH(ox, oy, tw, th))
    c.translate(ox, oy)
    for segs in wires:
        if len(segs):
            c.drawPoints(skia.Canvas.kLines_PointMode, _points(segs.reshape(-1, 2).astype(float)), paint)
    c.restore()


def _draw_depth(c, depths: list, ox: float, oy: float, tw: float, th: float):
    """The thumbnail dimmed, and the stages' depth over it: nearer is brighter, over the range this frame holds."""
    c.drawRect(skia.Rect.MakeXYWH(ox, oy, tw, th), skia.Paint(Color=_argb("#0a0c10c0")))
    if not depths:
        return
    d = np.max(np.stack([a[:th, :tw] for a in depths]), axis=0)
    seen = d > 0
    if not seen.any():
        return
    lo, hi = float(d[seen].min()), float(d[seen].max())
    v = np.where(seen, 40 + 215 * (d - lo) / max(hi - lo, 1e-12), 0).astype(np.uint8)
    px = np.zeros(d.shape + (4,), np.uint8)
    px[..., 0] = px[..., 1] = px[..., 2] = v
    px[..., 3] = seen * 255
    c.drawImage(skia.Image.fromarray(px, skia.kRGBA_8888_ColorType), ox, oy)


def _behind(boxes: dict, lid: str, a: list, w: int, h: int) -> str | None:
    """The shape or media layer drawn before `lid` that covers most of its box, if any.
    `boxes` is in draw order, so every entry before `lid` is underneath it. Layers that fill the
    canvas are skipped: under a background the colour is wrong, not the placement."""
    best, area = None, 0.0
    for other, b in boxes.items():
        if other == lid:
            break
        r = b["rect"]
        if (b["type"] in ("text", "counter", "captions", "group") or b["alpha"] < 0.5
                or (r[0] <= 0 and r[1] <= 0 and r[2] >= w and r[3] >= h)):
            continue
        ix, iy = min(a[2], r[2]) - max(a[0], r[0]), min(a[3], r[3]) - max(a[1], r[1])
        if ix > 0 and iy > 0 and ix * iy > area:
            best, area = other, ix * iy
    return best


def _pixels(r: Renderer) -> np.ndarray:
    """What the renderer's surface holds now, premultiplied."""
    buf = np.empty((r.h, r.w, 4), np.uint8)
    r.surface.readPixels(r.info, buf, r.w * 4)
    return buf


def _combined(layer) -> bool:
    """Whether a combining group merges the layer into its shape, so it is not drawn as itself."""
    p = layer.parent
    while p is not None and not p.raw.get("combine"):
        p = p.parent
    return p is not None


def _over(boxes: dict, lid: str, a: list) -> str | None:
    """The layer drawn after `lid` whose box covers most of `lid`'s box, if any covers it at all."""
    best, area, after = None, 0.0, False
    for other, b in boxes.items():
        if other == lid:
            after = True
            continue
        if not after or b["alpha"] < 0.5 or b.get("mask"):
            continue
        r = b["rect"]
        ix, iy = min(a[2], r[2]) - max(a[0], r[0]), min(a[3], r[3]) - max(a[1], r[1])
        if ix > 0 and iy > 0 and ix * iy > area:
            best, area = other, ix * iy
    return best


# A text layer's pixels count as shown where the frame with it differs from the frame without it by more than this.
SHOWS = 8


def _covered(r: Renderer, layer, T: float, box, full: np.ndarray, under: np.ndarray) -> bool:
    """Whether text `layer` draws something of its own at T (the frame up to and including it differs from the frame
    up to it) and yet none of that is left in the finished frame `full` (the frame without it is the same).
    `under` is the frame drawn up to the layer; `box` its box as slices."""
    r.draw(T, stop_after=layer.id)
    ink = (np.abs(_pixels(r)[box].astype(np.int16) - under[box]).max(axis=2) > SHOWS).sum()
    if ink < 4:
        return False  # it does not stand out from what is behind it: contrast reports that
    was = layer.raw.get("hidden")
    layer.raw["hidden"] = True
    try:
        r.draw(T)
    finally:
        if was is None:
            del layer.raw["hidden"]
        else:
            layer.raw["hidden"] = was
    shown = (np.abs(full[box].astype(np.int16) - _pixels(r)[box]).max(axis=2) > SHOWS).sum()
    return shown <= ink * 0.02


def _draw_boxes(c, scene: Scene, boxes: dict, ox: float, oy: float, tw: float, th: float, font):
    """Outline every visible layer; label only top-level layers, on a dark tab just outside the box,
    so labels neither cover the text they name nor pile up for every child of a component."""
    line = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1, Color=_argb("#b9f45d"))
    tab = skia.Paint(AntiAlias=True, Color=_argb("#0a0c10d8"))
    ink = skia.Paint(AntiAlias=True, Color=_argb("#b9f45d"))
    taken: list[skia.Rect] = []
    c.save()
    c.clipRect(skia.Rect.MakeXYWH(ox, oy, tw, th))
    for lid, b in boxes.items():
        if b["alpha"] < 0.05:
            continue
        l, t_, rr, bb = b["rect"]  # already in thumbnail pixels (the renderer is scaled)
        c.drawRect(skia.Rect(ox + l, oy + t_, ox + rr, oy + bb), line)
        layer = scene.by_id.get(lid)
        if layer is not None and layer.parent is not None:
            continue
        w = font.measureText(lid) + 6
        lx = min(max(ox, ox + l), ox + tw - w)
        ly = oy + t_ - 14 if t_ - 14 >= 0 else oy + bb  # above the box, or below it at the top edge
        rect = skia.Rect.MakeXYWH(lx, ly, w, 14)
        if any(rect.intersect(o) for o in taken):
            continue  # skip a label that would sit on another; the outline still shows
        taken.append(rect)
        c.drawRect(rect, tab)
        c.drawString(lid, lx + 3, ly + 11, font, ink)
    c.restore()


def probe(src, at: float | None = None) -> dict:
    """Resolved timeline; with `at`, also every visible layer's screen box at that time."""
    scene = load(src)
    out = {"ok": True, **scene.summary()}  # as a media file's probe says it
    if at is not None:
        try:
            t = float(at)
        except (TypeError, ValueError):
            t = math.nan
        if not math.isfinite(t):
            scene.close()
            raise ValueError(f"at must be a time in seconds, got {at!r}")
        r = Renderer(scene, track=True)
        r.footage = False  # boxes only
        r.draw(t)
        out["at"] = t
        out["visible"] = r.boxes
    scene.close()
    return out


def _chars(chars: list[str], most: int = 8) -> str:
    """'😀' (U+1F600), 'ש' (U+05E9) and 3 more."""
    named = [f"'{ch}' (U+{ord(ch):04X})" for ch in chars[:most]]
    more = f" and {len(chars) - most} more" if len(chars) > most else ""
    return (", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 and not more else ", ".join(named)) + more
def probe_media(src: str) -> dict:
    """What `kine probe` says about a file that is not a scene: {"ok", "src", "kind": video|audio|image|model,
    ...}. A media file gives its duration, displayed w and h, fps, audio, alpha and rotation; a model its triangles,
    vertices, materials, size, animations (each with its duration in seconds), skins (with their joint counts),
    morph targets, textures and texture coordinate sets. Raises MediaError or ValueError, naming the file, for one
    it cannot read."""
    if Path(src).suffix.lower() in models.EXTENSIONS:
        imp = models.load(src)
        about = imp.about or {}
        out = {"ok": True, "src": src, "kind": "model", "triangles": int(len(imp.mesh.T)),
               "vertices": int(len(imp.mesh.P)), "materials": len(imp.materials),
               "size": [round(float(v), 3) for v in imp.size], "animations": about.get("animations", []),
               "skins": about.get("skins", []), "morphTargets": about.get("morphTargets", 0),
               "textures": about.get("textures", 0), "uvSets": about.get("uvSets", [])}
        if about.get("crowded"):
            out["note"] = (f"{about['crowded']} vertices are moved by more than four joints; each follows the four "
                           "in JOINTS_0")
        return out
    info = media.probe(src)
    kind = "image" if info["image"] else "video" if info["video"] else "audio" if info["audio"] else "other"
    out = {"ok": True, "src": src, "kind": kind, **info}
    if info["codec"] and not info["video"] and not info["image"]:
        out["note"] = (f"it lists a {info['codec']} video stream but holds no frame of it, so a video layer shows "
                       "nothing; a cut made with -c copy that starts between key frames does this: cut it again "
                       "without -c copy")
    return out


def media_issues(scene: Scene, files_only: bool = False) -> list[dict]:
    """Warnings about the clips the video layers play: a still image, a file whose picture stops before its probed
    end (cut short or damaged), alpha this ffmpeg cannot decode, and (unless files_only) a dur that runs past the
    trimmed clip or the file, so its last frame holds. A file with no picture is an error when the scene loads."""
    out, ends = [], {}
    layers = [L for L in scene.all_layers() if L.type == "video" and L.src and Path(L.src).is_file()]

    def warn(code, L, T0, T1, msg, fix):
        out.append({"level": "warning", "code": code, "layer": L.id, "from": round(T0, 2), "to": round(T1, 2),
                    "message": msg, "fix": fix})

    plays = []
    for L in layers:
        try:
            info = media.probe(L.src)
        except media.MediaError:
            continue  # reported when the scene loads
        name, shown = L.raw.get("src", L.src), min(L.t_out, scene.duration)
        if shown <= L.t_in:
            continue
        if info["image"]:
            warn("media-still", L, L.t_in, shown, f"video src {name} is a still image, which a video layer plays as "
                 "a clip one frame long", "use an image layer (\"type\": \"image\") for a still")
            continue
        if not info["video"]:
            continue  # a no-picture error when the scene loads
        if info["alpha"] and info["codec"] in media.ALPHA_DECODERS and not media.alpha_decoder(info):
            warn("media-alpha-lost", L, L.t_in, shown, f"this ffmpeg has no {media.ALPHA_DECODERS[info['codec']]} "
                 f"decoder, so {name} is shown without its transparency",
                 "install an ffmpeg built with libvpx, or use a .mov (ProRes 4444) with alpha")
        trim0 = float((L.raw.get("trim") or [0])[0])
        speed = float(L.raw.get("speed", 1))
        end = loop_end(L, info["duration"])  # the trim end, or the end of the file
        loop = bool(L.raw.get("loop"))
        need = end if loop else min(end, trim0 + (shown - L.t_in) * speed)  # the latest source time it shows
        plays.append((L, info, name, trim0, speed, end, need, shown))
        over = (trim0 + (shown - L.t_in) * speed - end) / speed
        if not files_only and not loop and info["duration"] and end > trim0 and over > 1.5 / scene.fps:
            T = L.t_in + (end - trim0) / speed
            clip = (f"the trimmed clip ({trim0:g} s to {end:g} s of {name})" if end < info["duration"] - 1e-3
                    else f"the end of {name} ({info['duration']:.2f} s long)")
            warn("clip-hold", L, T, shown, f"the layer runs {over:.2f} s past {clip}, so its last frame holds from "
                 f"{T:.2f} s", "set \"loop\": true to repeat the clip, shorten dur, or leave dur out to end with "
                 "the clip")
    # Where each file's picture really ends: a decode of its last second, files side by side.
    files = sorted({p[0].src for p in plays})
    if files:
        with ThreadPoolExecutor(min(4, len(files))) as pool:
            ends = dict(zip(files, pool.map(media.decoded_end, files)))
    for L, info, name, trim0, speed, end, need, shown in plays:
        got = ends.get(L.src)
        if got is None or got >= need - max(0.15, 2.5 / (info["fps"] or scene.fps)):
            continue
        T = max(L.t_in, min(shown, L.t_in + (got - trim0) / speed))
        warn("media-truncated", L, T, shown, f"{name} has pictures only up to {got:.2f} s of its "
             f"{info['duration']:.2f} s, so the layer holds its last frame from {T:.2f} s",
             f"the file was probably cut short or damaged: export or copy it again; if its picture really ends "
             f"early, trim it to [{trim0:g}, {got:.2f}]")
    return out


def media_warnings_later(scene: Scene):
    """Start media_issues(scene, files_only=True) in a thread, so a render reports a cut-short or picture-less clip
    without waiting for it before its first frame. Returns a function that adds the warnings to a result dict."""
    pool = ThreadPoolExecutor(1)
    fut = pool.submit(media_issues, scene, True)
    pool.shutdown(wait=False)

    def add(res: dict) -> dict:
        try:
            found = fut.result()
        except (RuntimeError, OSError, ValueError):
            found = []
        if found:
            res["warnings"] = found
        return res
    return add


def _words(text: str) -> int:
    return len([w for w in text.replace("**", "").split() if any(ch.isalnum() for ch in w)])


def _plain(text) -> str:
    return " ".join(str(text).replace("**", "").split())


def _on_screen(t0: float, t1: float, spans: list[tuple[float, float]], slack: float) -> float:
    """How long the same words stay on screen around [t0, t1], joining the spans of layers that hand over to
    each other within `slack` seconds."""
    grown = True
    while grown:
        grown = False
        for s0, s1 in spans:
            if s0 <= t1 + slack and s1 >= t0 - slack and (s0 < t0 or s1 > t1):
                t0, t1, grown = min(t0, s0), max(t1, s1), True
    return t1 - t0


def check(src, step: float = 0.25, safe: float = 0.05) -> dict:
    """Lint a scene. Returns {"ok", "errors", "warnings", "issues": [...]} where each issue names
    the layer, the time range it was seen in, and a suggested fix."""
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
        raise ValueError(f"step must be a number of seconds above 0, got {step!r}")
    try:
        scene = load(src)
    except SceneError as e:
        return _result(e.issues)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        from .media import explain  # names ffmpeg or ffprobe when one is missing
        return _result([{"level": "error", "code": "load", "layer": None, "message": explain(e), "fix": ""}])
    except (TypeError, AttributeError, KeyError, IndexError, ArithmeticError) as e:  # a value of a shape not checked
        return _result([{"level": "error", "code": "load", "layer": None, "message": f"{type(e).__name__}: {e}",
                          "fix": "a key in the scene probably has the wrong type of value (a list for an object, …)"}])
    scene.check_expressions()
    W, H = scene.w, scene.h
    odd = []
    if W % 2 or H % 2:
        # 4:2:0 video needs even sizes: the renderer drops the last column or row rather than invent one.
        odd = [{"level": "info", "code": "odd-size", "layer": None, "from": 0, "to": round(scene.duration, 2),
                "message": f"the canvas is {W}x{H}; .mp4, .webm and .gif need an even width and height, so they come "
                           f"out {W - W % 2}x{H - H % 2}, without the last {'column' if W % 2 else 'row'}"
                           f"{' and row' if W % 2 and H % 2 else ''}. PNG frames, stills and .mov keep {W}x{H}",
                "fix": f"make canvas w and h even, e.g. {W + W % 2}x{H + H % 2}"}]
    pool = ThreadPoolExecutor(2)
    heard = pool.submit(_heard, scene)  # the mix renders and is measured while the frames are checked
    clips = pool.submit(media_issues, scene)  # and the clips' last seconds are decoded
    pool.shutdown(wait=False)
    r = Renderer(scene, track=True, scale=1.0)
    # The samples read boxes, not pixels: a clip decoded at each of them reopened ffmpeg nearly every time.
    r.footage = False
    found: dict[tuple, dict] = {}
    off: dict[str, list] = {}  # media layer (or its sequence) id -> (time, box, share off, layer) when mostly off
    cuts: dict[str, list] = {}  # masked layer id -> [times it is drawn at, times its mask overlaps it at]
    shown: dict[str, int] = {}  # other layer id -> samples it is visible at
    gone: dict[str, list] = {}  # other layer id -> times it is visible but entirely off the canvas
    slow: list[tuple[float, float]] = []  # (time, s its offscreen effects take on the CPU) past SLOW_EFFECTS

    def add(level, code, lid, T, msg, fix="", until=None):
        key = (code, lid)
        it = found.get(key)
        if it is None:
            found[key] = {"level": level, "code": code, "layer": lid, "from": round(T, 2),
                          "to": round(T if until is None else until, 2), "message": msg, "fix": fix}
        else:
            it["to"] = round(T, 2)

    def broke(e, T):
        """check's result when the frame at T cannot be drawn."""
        heard.cancel() or heard.result()
        clips.cancel() or clips.result()
        scene.close()
        lid = (getattr(e, "issues", None) or SceneError(str(e)).issues)[0]["layer"]  # from "layer 'x': ..."
        return _result(scene.issues + [{"level": "error", "code": "render", "layer": lid, "from": round(T, 2),
                                        "to": round(T, 2), "message": f"{type(e).__name__}: {e}", "fix": ""}],
                       duration=scene.duration)

    step = max(step, 1 / scene.fps)  # finer than a frame draws the same frames again
    n = max(1, int(scene.duration / step))
    grid = [min(scene.duration - 1e-3, k * step) for k in range(n + 1)]
    # A layer on screen for less than a step can fall between two samples and go unchecked: look at its middle.
    extra = set()
    for L in scene.all_layers():
        end, k = min(L.t_out, scene.duration), bisect.bisect_left(grid, L.t_in)
        if L.t_in < end and not (k < len(grid) and grid[k] < end):
            extra.add((L.t_in + end) / 2)
    for T in sorted(set(grid) | extra):
        try:
            r.draw(T)
        except (ValueError, RuntimeError, OSError) as e:
            return broke(e, T)
        for lid in r.inside:
            add("warning", "camera-inside-model", lid, T, "the camera is inside this model, so it sees only the "
                "model's inside, which is not drawn", "move the camera back (a higher camera z) or the model away")
        for lid in r.swim:
            add("warning", "texture-swim", lid, T, "part of this model comes so near the camera that its texture is "
                "drawn out of place there: Skia spreads an image evenly across each triangle, and these stay too "
                "large on screen after cutting", "keep the model farther from the camera")
        for lid in r.culled - r.inside:
            msg = ("the triangles of this model that cross the camera plane are not drawn"
                   if scene.by_id[lid].type == "model" else "layer crosses the camera plane, so it is not drawn")
            add("warning", "behind-camera", lid, T, msg, "lower its z, or move the camera back (a higher camera z)")
        for lid in r.fogged:
            add("warning", "fog-hides", lid, T, "this layer lies past the fog's far distance, so it shows only the "
                "fog's colour", "bring it nearer, or move fog far back")
        for lid in r.unlit:
            add("warning", "light-behind", lid, T, "the key light is behind this model as the camera sees it, so "
                "its front shows only ambient and rim light", "turn the light toward the camera: an azimuth between "
                "-90 and 90, or \"follow\": \"camera\"")
        if r.unfocused:
            add("warning", "focus-on-nothing", "camera", T, "the camera's focus plane is nearer or farther than every "
                "layer it moves, so nothing is sharp", "focus on a layer id, or a z between the layers'")
        if r.slow_dof:
            add("warning", "slow-depth-of-field", "camera", T, "a stage blurs per pixel for depth of field, which on "
                "the CPU takes about five times as long as the stage (0.8 s a 960x540 frame of examples/focus.json, "
                "against 0.17)",
                "render on a GPU, or leave aperture off while drafting")
        if r.backend == "cpu" and r.filtered * CPU_BLUR_S > SLOW_EFFECTS:
            slow.append((T, r.filtered * CPU_BLUR_S))
        if r.tris > TRIANGLE_BUDGET:
            add("warning", "triangle-budget", None, T, f"{r.tris:,} model triangles in one frame; renders slow "
                f"down past {TRIANGLE_BUDGET:,} (about 70 ms a frame at 100,000 on a GPU)",
                "use \"detail\": \"low\" or \"medium\", or fewer parts")
        texts = []
        for lid, b in r.boxes.items():
            layer = scene.by_id.get(lid)
            if layer is not None and layer.mask is not None and not layer.mask[1].endswith("Invert"):
                seen = cuts.setdefault(lid, [[], []])
                seen[0].append(T)
                mb, a = r.boxes.get(layer.mask[0].id), b["rect"]
                if mb is not None and min(a[2], mb["rect"][2]) > max(a[0], mb["rect"][0]) and \
                        min(a[3], mb["rect"][3]) > max(a[1], mb["rect"][1]):
                    seen[1].append(T)
            if b["alpha"] < 0.3 or b.get("mask"):
                continue
            l, t, rr, bb = b["rect"]
            up = layer
            allow = False
            while up is not None and not allow:  # set on a group or sequence, it covers what is inside
                allow, up = bool(up.raw.get("allowOffscreen")), up.parent
            if b["type"] in ("text", "counter", "captions"):
                texts.append((lid, b))
                if b.get("mirrored"):
                    add("warning", "mirrored-text", lid, T, "text faces away from the camera, so it reads mirrored",
                        "keep rotateX and rotateY within ±90°, or set \"backface\": \"hidden\" and add a back side")
                if (l < -1 or t < -1 or rr > W + 1 or bb > H + 1) and not allow:
                    add("error", "text-offscreen", lid, T, f"text box {b['rect']} leaves the {W}x{H} canvas",
                        "move it inward, lower size, or set maxWidth")
                elif (l < W * safe or t < H * safe or rr > W * (1 - safe) or bb > H * (1 - safe)) and not allow:
                    add("warning", "safe-area", lid, T, f"text is inside the outer {int(safe * 100)}% margin",
                        "keep titles within the title-safe area")
                px = b["size"] * b.get("scale", 1)
                if px < H / 45 and b["alpha"] > 0.9:
                    add("warning", "tiny-text", lid, T, f"text renders at ~{px:.0f}px, small for {H}p video",
                        f"use at least {math.ceil(H / 45)}px")
            elif b["type"] in MEDIA:
                area = (rr - l) * (bb - t)
                seen = max(0.0, min(rr, W) - max(l, 0.0)) * max(0.0, min(bb, H) - max(t, 0.0))
                if area > 0 and seen < area / 2 and not allow and layer is not None and T not in extra:
                    seq = layer.parent if layer.parent is not None and layer.parent.raw.get("span") else None
                    off.setdefault(seq.id if seq else lid, []).append((T, b["rect"], 1 - seen / area, layer))
            elif b["type"] != "group" and not allow:
                shown[lid] = shown.get(lid, 0) + 1
                if rr < 0 or bb < 0 or l > W or t > H:
                    gone.setdefault(lid, []).append(T)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b2 = texts[i][1]["rect"], texts[j][1]["rect"]
                ix = min(a[2], b2[2]) - max(a[0], b2[0])
                iy = min(a[3], b2[3]) - max(a[1], b2[1])
                if ix > 4 and iy > 4:
                    area = min((a[2] - a[0]) * (a[3] - a[1]), (b2[2] - b2[0]) * (b2[3] - b2[1])) or 1
                    if ix * iy / area > 0.15:
                        add("warning", "text-overlap", f"{texts[i][0]}+{texts[j][0]}", T,
                            "two text layers overlap", "offset one of them or separate them in time")
    for lid, (drawn, met) in cuts.items():
        if not met:
            m = scene.by_id[lid].mask[0].id
            add("warning", "mask-misses", lid, drawn[0], f"its mask '{m}' never overlaps it, so it is never seen",
                f"move '{m}' over '{lid}' (it stands in the same group, by its own x and y), or check the mask's "
                "times", until=drawn[-1])
    for lid, ts in gone.items():
        # A shape that sweeps across the canvas starts or ends off it: note one that is off whenever it is on, or
        # for MEDIA_OFF_FOR seconds.
        if len(ts) == shown[lid] or len(ts) * step >= MEDIA_OFF_FOR:
            add("info", "offscreen", lid, ts[0], "layer is visible but entirely off canvas", "check x/y", until=ts[-1])
    for lid, seen in off.items():
        if len({s[0] for s in seen}) * step < MEDIA_OFF_FOR:  # frames, once each where a sequence's clips overlap
            continue
        T, rect, share, layer = seen[0]
        ty = layer.type
        fix = ("a video or shader is placed by its top-left corner unless anchor says otherwise: set \"anchor\": "
               "\"center\" to centre it on x and y, or move it inward"
               if ty in ("video", "shader") and "anchor" not in layer.raw else
               "move it inward, or set allowOffscreen if it is meant to be cut off")
        add("warning", "media-offscreen", lid, T, f"{ty} box {rect} is {share:.0%} off the {W}x{H} canvas", fix,
            until=seen[-1][0])
    # Text under a later layer, and contrast: render the frame without the text layer and its successors, compare
    # luminance under its box. These read pixels, footage included.
    r.footage = True
    for layer in scene.all_layers():
        if layer.type not in ("text", "counter") or layer.id in scene.mattes:
            continue
        T = min(layer.t_out - 1e-3, layer.t_in + min(layer.dur * 0.6, 1.5))
        try:
            r.draw(T)  # the clips on screen are decoded here; the draws below at T show the same frames
        except (ValueError, RuntimeError, OSError) as e:
            return broke(e, T)
        boxes = r.boxes  # draw() makes a new dict each time
        b = boxes.get(layer.id)
        if not b or b["alpha"] < 0.5 or _combined(layer):
            continue
        l, t, rr, bb = (int(max(0, v)) for v in b["rect"])
        rr, bb = min(r.w, rr), min(r.h, bb)
        if rr - l < 2 or bb - t < 2:
            continue
        full = _pixels(r)
        r.draw(T, stop_at=layer.id)
        buf = _pixels(r)
        over = _over(boxes, layer.id, b["rect"])
        if over and _covered(r, layer, T, (slice(t, bb), slice(l, rr)), full, buf):
            add("warning", "text-hidden", layer.id, T, f"text is covered by '{over}', a later layer drawn on top of "
                f"it: none of it shows at {T:.2f}s",
                f"move '{layer.id}' after '{over}' in the layer list (later layers draw on top), or move one of them")
            continue
        if not b.get("fill") or layer.raw.get("box"):
            continue
        patch = buf[t:bb, l:rr, :3].reshape(-1, 3).astype(np.float32) / 255
        lin = np.where(patch <= 0.04045, patch / 12.92, ((patch + 0.055) / 1.055) ** 2.4)
        lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        fg = C.luminance(tuple(b["fill"]))
        worst = min(C.contrast(fg, float(np.percentile(lum, 10))), C.contrast(fg, float(np.percentile(lum, 90))))
        if worst < 3.0:
            under = _behind(r.boxes, layer.id, b["rect"], r.w, r.h)
            add("warning", "contrast", layer.id, T, f"contrast against what is behind it is {worst:.1f}:1 (want 4.5:1)"
                + (f"; it overlaps '{under}'" if under else ""),
                f"move it clear of '{under}', or add a box or a shadow" if under
                else "add a box, a shadow, or change the colour")
    if slow:
        add("warning", "slow-effects", None, slow[0][0], "the blurs, shadows, glows, outlines, inner shadows, mask "
            f"feathers and displace drawn offscreen take about {max(s for _, s in slow):.1f} s a frame on the CPU. "
            "Skia's CPU blur takes 0.44 s for a 1080p layer, and a layer's blur or shadow blurs one the size of the frame",
            "render on a GPU; draft with --scale 0.5 (a quarter of the pixels); or blur fewer layers",
            until=slow[-1][0])
    for lid, T, px in strobes(scene, r):
        add("warning", "strobe", lid, T, f"moves {px:.0f} px a frame, fast enough to show as separate copies of "
            "itself instead of one moving thing", "add \"motionBlur\": {\"samples\": 8} to the scene, or slow the move")
    for layer in scene.all_layers():
        for k, t in stops(layer):
            add("warning", "stop-and-go", layer.id, layer.t_in + t,
                f"{k} comes to a stop at its keyframe at {t:g}s between two moves the same way: the ease into the key "
                "slows to it and the ease out of it starts from rest",
                f"pass through the key: \"{k}\": {{\"keys\": [...], \"smooth\": true}}; or a linear ease on the key")
        n = len(layer.children)
        if layer.type == "group" and layer.raw.get("sort") == "pixel" and n > stage.CHILDREN:
            add("warning", "stage-cost", layer.id, layer.t_in,
                f"a stage draws and merges each of its {n} children on its own, so frames slow down past "
                f"{stage.CHILDREN} children", "keep layers that cross nothing outside the stage, or merge touching "
                "models as parts of one", until=layer.t_out)
    # Characters that cannot be drawn as written.
    for layer in scene.all_layers():
        if not (layer.type in ("text", "counter", "captions") or layer.type == "model" and layer.raw.get("shape") == "text"):
            continue
        missing, emoji, flat = r.undrawable(layer)
        if missing:
            them = "it" if len(missing) == 1 else "them"
            add("warning", "missing-glyph", layer.id, layer.t_in,
                f"no installed font has {_chars(missing)}, so "
                + ("it draws as an empty box" if len(missing) == 1 else "they draw as empty boxes"),
                f"install a font that has {them} and name it in \"font\", or take {them} out", until=layer.t_out)
        if emoji:
            add("warning", "unshaped-3d-text", layer.id, layer.t_in,
                f"colour emoji have no outline to push out in depth, so 3D text leaves out {_chars(emoji)}",
                "put emoji in a text or image layer beside the model", until=layer.t_out)
        if flat:
            add("warning", "unshaped-3d-text", layer.id, layer.t_in,
                f"letterSpacing draws each character of 3D text on its own, so {_chars(flat)} "
                f"{'comes' if len(flat) == 1 else 'come'} out unjoined, in the wrong order or with misplaced marks",
                "take letterSpacing out: without it 3D text is shaped as a text layer is", until=layer.t_out)
        text = str(layer.raw.get("text", ""))
        spaced = layer.raw.get("letterSpacing") not in (None, 0)
        if layer.type == "text" and (layer.raw.get("split") == "chars" or spaced) and joining(text):
            add("warning", "unjoined-letters", layer.id, layer.t_in,
                f"{'split chars' if layer.raw.get('split') == 'chars' else 'letterSpacing'} draws each letter on its "
                "own, so the Arabic-script letters of this text do not join",
                "use split words or lines, and no letterSpacing, for Arabic-script text", until=layer.t_out)
    # Reading time: about 0.4 s to find the text and 0.3 s a word; one or two words need FLASH. Durations are whole
    # frames, so a layer within half a frame of the need passes (4 words for 1.6 s were flagged for 12 ms).
    # A text that takes over from the same words (a chip's highlighted state after its plain one) counts the time
    # the words were already on screen: site/demos/voices.json's "[laughs]" chip was flagged as a 0.34 s flash.
    shown: dict[str, list[tuple[float, float]]] = {}
    for layer in scene.all_layers():
        if layer.type == "text":
            shown.setdefault(_plain(layer.raw.get("text", "")), []).append((layer.t_in, layer.t_out))
    for layer in scene.all_layers():
        if layer.type == "text":
            words = _words(str(layer.raw.get("text", "")))
            need = 0.4 + 0.3 * words if words >= 3 else FLASH
            on = _on_screen(layer.t_in, layer.t_out, shown[_plain(layer.raw.get("text", ""))], 0.5 / scene.fps)
            if words and on + 0.5 / scene.fps < need:
                add("warning", "read-time", layer.id, layer.t_in,
                    f"{words} word{'s' if words > 1 else ''} on screen for {on:.2f}s; viewers need ~{need:.1f}s"
                    + ("" if words >= 3 else ": shorter is a flash nobody reads"), "extend dur or cut words",
                    until=layer.t_out)
        if layer.type == "captions" and not layer.raw.get("voice"):  # a voice's captions keep the pace it speaks at
            for s0, s1, txt, *_ in layer.raw.get("cues", []):
                words = _words(txt)
                if words and (s1 - s0) < 0.25 + words / 4.5:
                    add("warning", "read-time", layer.id, layer.t_in + s0,
                        f"caption '{txt[:30]}' is {s1 - s0:.2f}s for {words} words", "split or lengthen the cue",
                        until=layer.t_in + s1)
                    break
    sound = heard.result()
    try:
        clip_issues = clips.result()
    except (RuntimeError, OSError, ValueError) as e:
        clip_issues = [{"level": "info", "code": "media-unchecked", "layer": None, "from": 0,
                        "to": round(scene.duration, 2), "message": f"the clips could not be decoded to check them: {e}",
                        "fix": ""}]
    scene.close()
    return _result(scene.issues + odd + list(found.values()) + sound + clip_issues, duration=scene.duration)


def _heard(scene: Scene) -> list[dict]:
    """check's sound issues. A mix that ffmpeg cannot make is an error, since a render with sound stops on the same
    mix; a mix that was made but could not be measured is an info note."""
    try:
        rep = listen.report(scene)
        return listen.issues(scene, rep) + _unheard(scene, rep)
    except Exception as e:  # noqa: BLE001 - check reports, it does not crash: a TypeError here took check down
        failed = str(e).startswith("audio mix failed")
        return [{"level": "error" if failed else "info", "code": "audio-mix" if failed else "audio-unmeasured",
                 "layer": None, "from": 0, "to": round(scene.duration, 2),
                 "message": (f"the sound cannot be mixed, so a render with sound stops too: {e}" if failed else
                             f"the sound could not be measured: {type(e).__name__}: {e}"), "fix": ""}]


def _unheard(scene: Scene, rep: dict | None) -> list[dict]:
    """Sounds that play nothing anyone hears: one that starts after the scene ends (or ends before it starts), and one
    whose gain, volume, fades and duck keep it under listen.SILENT the whole time (volume 0, gain -90). A file that is
    silent itself is audio-silent, from listen.issues."""
    out, D, timed = [], scene.duration, set()

    def add(sid, t0, t1, msg, fix):
        out.append({"level": "warning", "code": "audio-unheard", "layer": sid, "from": round(t0, 2),
                    "to": round(t1, 2), "message": msg, "fix": fix})

    for s in scene.audio:
        name = f"the sound of '{s.layer}'" if s.layer else f"'{s.id}'"
        if s.t_in >= D - 1e-6:
            timed.add(s.id)
            add(s.layer or s.id, D, D, f"{name} starts at {s.t_in:.2f}s, when the scene ({D:.2f}s) has ended, so it is "
                "never heard", "start it earlier, or make the scene longer")
        elif s.t_out <= 1e-6:
            timed.add(s.id)
            add(s.layer or s.id, 0, 0, f"{name} ends at {s.t_out:.2f}s, before the scene starts, so it is never heard",
                "start it later")
    by_id = {s.id: s for s in scene.audio}
    boost = max(0.0, (rep or {}).get("correction") or 0.0)  # the managed mix raises a quiet mix by up to 12 dB
    for s in (rep or {}).get("sounds", []):
        if s["id"] in timed or not s["info"]["peak"] > listen.SILENT:
            continue
        top = float(np.max(s["env"], initial=-np.inf)) + boost
        if top > listen.SILENT:
            continue
        snd = by_id.get(s["id"])
        raw = snd.raw if snd is not None else {}
        why = ("its volume is 0" if raw.get("volume") == 0 and raw.get("gain") is None else
               f"its gain is {raw['gain']:g} dB" if isinstance(raw.get("gain"), (int, float)) else
               f"with its gain and fades it stays under {listen.SILENT:.0f} dBFS")
        sid = snd.layer if snd is not None and snd.layer else s["id"]
        add(sid, s["in"], min(s["out"], D), f"'{sid}' is never heard: {why}",
            "raise its volume or gain, or remove it")
    return out


_RANK = {"error": 0, "warning": 1, "info": 2}


def _result(issues: list[dict], duration: float | None = None) -> dict:
    """Errors first, then warnings, then info notes. `ok` is true when there are no errors."""
    issues = sorted(issues, key=lambda i: (_RANK[i["level"]], i.get("from", -1)))
    errors = sum(1 for i in issues if i["level"] == "error")
    out = {"ok": errors == 0, "errors": errors, "warnings": sum(1 for i in issues if i["level"] == "warning"),
           "issues": issues}
    if duration is not None:
        out["duration"] = round(duration, 3)
    return out
