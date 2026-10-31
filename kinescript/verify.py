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
from . import listen
from .render import FONT_DIR, Renderer, _argb
from .scene import Scene, SceneError, load

# Model triangles one frame may draw before check warns. 100,000 took 71 ms a frame at 1080p on
# the GPU in the 3D spike (2026-10-29), against about 15 ms at 20,000.
TRIANGLE_BUDGET = 60_000
# Layers check warns about when more than half of one stays off the canvas for MEDIA_OFF_FOR seconds; a slide or
# a transition passes through faster.
MEDIA = ("video", "image", "shader")
MEDIA_OFF_FOR = 1.0


def _times(scene: Scene, count: int, times=None) -> list[float]:
    if times:  # as still reads them: negative counts from the end, and the scene's ends clamp
        from .output import frame_at
        return [frame_at(scene, t) / scene.fps for t in times]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count must be a whole number of frames, 1 or more, got {count!r}")
    step = scene.duration / count
    return [min(scene.duration - 1 / scene.fps, step * (k + 0.5)) for k in range(count)]


def contact_sheet(src, out: str, count: int = 12, cols: int | None = None, width: int = 480,
                  times=None, boxes: bool = False, audio: bool = False) -> dict:
    """One PNG of `count` evenly spaced frames, each labelled with its time and frame number.
    boxes=True outlines every visible layer with its id. audio=True adds the sound under the frames, one lane per
    role and one for the mix, and returns its measurements."""
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
    scene.close()
    save_image(sheet.makeImageSnapshot(), Path(out))
    info = {"out": out, "times": [round(t, 3) for t in ts], "grid": [cols, rows], "backend": r.backend}
    if audio:
        info["audio"] = listen.summary(rep)
    if notes:
        info["notes"] = notes
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
        try:
            t = float(at)
        except (TypeError, ValueError):
            t = math.nan
        if not math.isfinite(t):
            scene.close()
            raise ValueError(f"at must be a time in seconds, got {at!r}")
        r = Renderer(scene, track=True)
        r.draw(t)
        out["at"] = t
        out["visible"] = r.boxes
    scene.close()
    return out


def _words(text: str) -> int:
    return len([w for w in text.replace("**", "").split() if any(ch.isalnum() for ch in w)])


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
    pool = ThreadPoolExecutor(1)
    heard = pool.submit(_heard, scene)  # the mix renders and is measured while the frames are checked
    pool.shutdown(wait=False)
    r = Renderer(scene, track=True, scale=1.0)
    found: dict[tuple, dict] = {}
    off: dict[str, list] = {}  # media layer (or its sequence) id -> (time, box, share off, layer) when mostly off

    def add(level, code, lid, T, msg, fix="", until=None):
        key = (code, lid)
        it = found.get(key)
        if it is None:
            found[key] = {"level": level, "code": code, "layer": lid, "from": round(T, 2),
                          "to": round(T if until is None else until, 2), "message": msg, "fix": fix}
        else:
            it["to"] = round(T, 2)

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
            heard.cancel() or heard.result()
            scene.close()
            lid = (getattr(e, "issues", None) or SceneError(str(e)).issues)[0]["layer"]  # from "layer 'x': ..."
            return _result(scene.issues + [{"level": "error", "code": "render", "layer": lid, "from": round(T, 2),
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
            layer = up = scene.by_id.get(lid)
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
