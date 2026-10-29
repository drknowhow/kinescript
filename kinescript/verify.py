"""Agent-side verification: contact sheets, layout lint and frame probes.

An agent cannot watch a video. These turn a scene into things it can read: one image of
many timestamps, a JSON list of problems with time ranges, and the exact box of every layer.
"""
from __future__ import annotations

import math

import numpy as np
import skia

from . import color as C
from .render import FONT_DIR, Renderer, _argb
from .scene import Scene, load


def _times(scene: Scene, count: int, times=None) -> list[float]:
    if times:
        return [float(t) for t in times]
    count = max(1, count)
    step = scene.duration / count
    return [min(scene.duration - 1 / scene.fps, step * (k + 0.5)) for k in range(count)]


def contact_sheet(src, out: str, count: int = 12, cols: int | None = None, width: int = 480,
                  times=None, boxes: bool = False) -> dict:
    """One PNG of `count` evenly spaced frames, each labelled with its time and frame number.
    boxes=True outlines every visible layer with its id."""
    scene = load(src)
    ts = _times(scene, count, times)
    cols = cols or min(len(ts), 4)
    rows = math.ceil(len(ts) / cols)
    scale = width / scene.w
    r = Renderer(scene, track=boxes, scale=scale)
    tw, th = r.w, r.h
    gap, label_h = 8, 26
    sheet = skia.Surface(cols * (tw + gap) + gap, rows * (th + label_h + gap) + gap)
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
            for lid, b in r.boxes.items():
                if b["alpha"] < 0.05:
                    continue
                l, t_, rr, bb = b["rect"]  # already in thumbnail pixels (the renderer is scaled)
                p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1, Color=_argb("#b9f45d"))
                c.drawRect(skia.Rect(x + l, y + label_h + t_, x + rr, y + label_h + bb), p)
                c.drawString(lid, x + l + 2, y + label_h + t_ + 12, small,
                             skia.Paint(AntiAlias=True, Color=_argb("#b9f45d")))
    sheet.makeImageSnapshot().save(out, skia.kPNG)
    scene.close()
    return {"out": out, "times": [round(t, 3) for t in ts], "grid": [cols, rows], "backend": r.backend}


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
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return {"ok": False, "errors": 1, "warnings": 0,
                "issues": [{"level": "error", "code": "load", "layer": None, "message": str(e)}]}
    W, H = scene.w, scene.h
    r = Renderer(scene, track=True, scale=1.0)
    found: dict[tuple, dict] = {}

    def add(level, code, lid, T, msg, fix=""):
        key = (code, lid)
        it = found.get(key)
        if it is None:
            found[key] = {"level": level, "code": code, "layer": lid, "from": round(T, 2), "to": round(T, 2),
                          "message": msg, "fix": fix}
        else:
            it["to"] = round(T, 2)

    n = max(1, int(scene.duration / step))
    for k in range(n + 1):
        T = min(scene.duration - 1e-3, k * step)
        try:
            r.draw(T)
        except (ValueError, RuntimeError, OSError) as e:
            scene.close()
            return {"ok": False, "errors": 1, "warnings": 0, "duration": round(scene.duration, 3),
                    "issues": [{"level": "error", "code": "render", "layer": None, "from": round(T, 2),
                                "to": round(T, 2), "message": f"{type(e).__name__}: {e}"}]}
        texts = []
        for lid, b in r.boxes.items():
            if b["alpha"] < 0.3:
                continue
            l, t, rr, bb = b["rect"]
            layer = scene.by_id.get(lid)
            allow = layer is not None and layer.raw.get("allowOffscreen")
            if b["type"] in ("text", "counter", "captions"):
                texts.append((lid, b))
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
            add("warning", "contrast", layer.id, T, f"contrast against what is behind it is {worst:.1f}:1 (want 4.5:1)",
                "add a box, a shadow, or change the colour")
    # Reading time.
    for layer in scene.all_layers():
        if layer.type == "text":
            words = _words(str(layer.raw.get("text", "")))
            need = 0.4 + words / 3.3
            if words >= 3 and layer.dur < need:
                add("warning", "read-time", layer.id, layer.t_in,
                    f"{words} words on screen for {layer.dur:.2f}s; viewers need ~{need:.1f}s", "extend dur or cut words")
        if layer.type == "captions":
            for s0, s1, txt in layer.raw.get("cues", []):
                words = _words(txt)
                if words and (s1 - s0) < 0.25 + words / 4.5:
                    add("warning", "read-time", layer.id, layer.t_in + s0,
                        f"caption '{txt[:30]}' is {s1 - s0:.2f}s for {words} words", "split or lengthen the cue")
                    break
    scene.close()
    issues = sorted(found.values(), key=lambda i: ({"error": 0, "warning": 1, "info": 2}[i["level"]], i["from"]))
    errors = sum(1 for i in issues if i["level"] == "error")
    return {"ok": errors == 0, "errors": errors, "warnings": sum(1 for i in issues if i["level"] == "warning"),
            "duration": round(scene.duration, 3), "issues": issues}
