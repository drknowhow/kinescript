"""Agent-side verification: contact sheets, layout lint and frame probes.

An agent cannot watch a video. These turn a scene into things it can read: one image of
many timestamps, a JSON list of problems with time ranges, and the exact box of every layer.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import skia

from . import color as C
from . import listen
from .render import FONT_DIR, Renderer, _argb
from .scene import Scene, SceneError, load

# Model triangles one frame may draw before check warns. 100,000 took 71 ms a frame at 1080p on
# the GPU in the 3D spike (2026-10-29), against about 15 ms at 20,000.
TRIANGLE_BUDGET = 60_000


def _times(scene: Scene, count: int, times=None) -> list[float]:
    if times:
        return [float(t) for t in times]
    count = max(1, count)
    step = scene.duration / count
    return [min(scene.duration - 1 / scene.fps, step * (k + 0.5)) for k in range(count)]


def contact_sheet(src, out: str, count: int = 12, cols: int | None = None, width: int = 480,
                  times=None, boxes: bool = False, audio: bool = False) -> dict:
    """One PNG of `count` evenly spaced frames, each labelled with its time and frame number.
    boxes=True outlines every visible layer with its id. audio=True adds the sound under the frames, one lane per
    role and one for the mix, and returns its measurements."""
    scene = load(src)
    ts = _times(scene, count, times)
    cols = cols or min(len(ts), 4)
    rows = math.ceil(len(ts) / cols)
    scale = width / scene.w
    r = Renderer(scene, track=boxes, scale=scale)
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
        c.drawString(f"{T:6.2f}s  f{n}", x + 2, y + 18, font, skia.Paint(AntiAlias=True, Color=_argb("#f2f0e8")))
        if boxes:
            _draw_boxes(c, scene, r.boxes, x, y + label_h, tw, th, small)
    if rep is not None:
        listen.strip(c, rep, listen.issues(scene, rep), gap, grid_h, cols * (tw + gap) - gap, ts, font, small)
    sheet.makeImageSnapshot().save(out, skia.kPNG)
    scene.close()
    info = {"out": out, "times": [round(t, 3) for t in ts], "grid": [cols, rows], "backend": r.backend}
    if audio:
        info["audio"] = listen.summary(rep)
    return info


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
    out = scene.summary()
    if at is not None:
        r = Renderer(scene, track=True)
        r.draw(float(at))
        out["at"] = float(at)
        out["visible"] = r.boxes
    scene.close()
    return out


def _words(text: str) -> int:
    return len([w for w in text.replace("**", "").split() if any(ch.isalnum() for ch in w)])


def check(src, step: float = 0.25, safe: float = 0.05) -> dict:
    """Lint a scene. Returns {"ok", "errors", "warnings", "issues": [...]} where each issue names
    the layer, the time range it was seen in, and a suggested fix."""
    try:
        scene = load(src)
    except SceneError as e:
        return _result(e.issues)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        return _result([{"level": "error", "code": "load", "layer": None, "message": str(e), "fix": ""}])
    W, H = scene.w, scene.h
    pool = ThreadPoolExecutor(1)
    heard = pool.submit(_heard, scene)  # the mix renders and is measured while the frames are checked
    pool.shutdown(wait=False)
    r = Renderer(scene, track=True, scale=1.0)
    found: dict[tuple, dict] = {}

    def add(level, code, lid, T, msg, fix="", until=None):
        key = (code, lid)
        it = found.get(key)
        if it is None:
            found[key] = {"level": level, "code": code, "layer": lid, "from": round(T, 2),
                          "to": round(T if until is None else until, 2), "message": msg, "fix": fix}
        else:
            it["to"] = round(T, 2)

    n = max(1, int(scene.duration / step))
    for k in range(n + 1):
        T = min(scene.duration - 1e-3, k * step)
        try:
            r.draw(T)
        except (ValueError, RuntimeError, OSError) as e:
            heard.cancel() or heard.result()
            scene.close()
            return _result(scene.issues + [{"level": "error", "code": "render", "layer": None, "from": round(T, 2),
                                            "to": round(T, 2), "message": f"{type(e).__name__}: {e}", "fix": ""}],
                           duration=scene.duration)
        for lid in r.culled:
            msg = ("the triangles of this model that cross the camera plane are not drawn"
                   if scene.by_id[lid].type == "model" else "layer crosses the camera plane, so it is not drawn")
            add("warning", "behind-camera", lid, T, msg, "lower its z, or move the camera back (a higher camera z)")
        if r.tris > TRIANGLE_BUDGET:
            add("warning", "triangle-budget", None, T, f"{r.tris:,} model triangles in one frame; renders slow "
                f"down past {TRIANGLE_BUDGET:,} (about 70 ms a frame at 100,000 on a GPU)",
                "use \"detail\": \"low\" or \"medium\", or fewer parts")
        texts = []
        for lid, b in r.boxes.items():
            if b["alpha"] < 0.3:
                continue
            l, t, rr, bb = b["rect"]
            layer = scene.by_id.get(lid)
            allow = layer is not None and layer.raw.get("allowOffscreen")
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
            elif b["type"] not in ("video", "group") and not allow:
                if rr < 0 or bb < 0 or l > W or t > H:
                    add("info", "offscreen", lid, T, "layer is visible but entirely off canvas", "check x/y")
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
    # Contrast: render the frame without the text layer and its successors, compare luminance under its box.
    for layer in scene.all_layers():
        if layer.type not in ("text", "counter"):
            continue
        T = min(layer.t_out - 1e-3, layer.t_in + min(layer.dur * 0.6, 1.5))
        r.draw(T)
        b = r.boxes.get(layer.id)
        if not b or b["alpha"] < 0.5 or not b.get("fill") or layer.raw.get("box"):
            continue
        r.draw(T, stop_at=layer.id)
        buf = np.empty((r.h, r.w, 4), np.uint8)
        r.surface.readPixels(r.info, buf, r.w * 4)
        l, t, rr, bb = (int(max(0, v)) for v in b["rect"])
        rr, bb = min(r.w, rr), min(r.h, bb)
        if rr - l < 2 or bb - t < 2:
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
    # Reading time.
    for layer in scene.all_layers():
        if layer.type == "text":
            words = _words(str(layer.raw.get("text", "")))
            need = 0.4 + words / 3.3
            if words >= 3 and layer.dur < need:
                add("warning", "read-time", layer.id, layer.t_in,
                    f"{words} words on screen for {layer.dur:.2f}s; viewers need ~{need:.1f}s", "extend dur or cut words",
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
    scene.close()
    return _result(scene.issues + list(found.values()) + sound, duration=scene.duration)


def _heard(scene: Scene) -> list[dict]:
    """check's sound issues; an info note when the mix cannot be made."""
    try:
        return listen.issues(scene, listen.report(scene))
    except (RuntimeError, OSError, ValueError) as e:
        return [{"level": "info", "code": "audio-unmeasured", "layer": None, "from": 0, "to": round(scene.duration, 2),
                 "message": f"the sound could not be measured: {e}", "fix": ""}]


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
