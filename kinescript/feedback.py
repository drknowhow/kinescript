"""Review notes on a scene, kept in a `<scene>.feedback.json` file beside it.

A note pins a comment to a time and a point (x, y as fractions of the frame) and, when the point
lands on one, a layer id. The editor writes notes; an agent lists and resolves them through the
`kine_feedback` MCP tool. Each note records a hash of the scene it was made against, so a note on
a scene that has since changed can be flagged stale.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

VERSION = 1
KINDS = ("pin", "box", "draw", "prop")


class FeedbackError(ValueError):
    pass


def sidecar(scene_path: str | os.PathLike) -> Path:
    p = Path(scene_path)
    return p.with_name(p.stem + ".feedback.json")


def scene_hash(scene_path: str | os.PathLike) -> str:
    return hashlib.sha1(Path(scene_path).read_bytes()).hexdigest()[:12]


def load(scene_path: str | os.PathLike) -> dict:
    p = sidecar(scene_path)
    if not p.is_file():
        return {"version": VERSION, "notes": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise FeedbackError(f"{p.name} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("notes"), list):
        raise FeedbackError(f"{p.name} has no notes list")
    return data


def _save(scene_path, data: dict) -> None:
    p = sidecar(scene_path)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _frac(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        raise FeedbackError(f"{name} must be a number from 0 to 1")
    return float(v)


def marked_still(sc, note: dict, scale: float = 0.5) -> str:
    """Path of a PNG of the frame at the note's time with its point ringed."""
    import skia

    from .output import still
    out = str(Path(tempfile.gettempdir()) / f"kine_note_{os.getpid()}_{note['id']}.png")
    still(sc, note["t"], out, scale)
    if "x" in note or note.get("strokes"):
        img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(Path(out).read_bytes()))
        surf = skia.Surface(img.width(), img.height())
        with surf as canvas:
            canvas.drawImage(img, 0, 0)
            w, h = img.width(), img.height()
            for width, color in ((6, 0xFF000000), (3, 0xFFFFD400)):
                paint = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, Color=color,
                                   StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
                if "x" in note:
                    canvas.drawCircle(note["x"] * w, note["y"] * h, max(w, h) * 0.025, paint)
                for line in note.get("strokes", []):
                    path = skia.Path()
                    path.moveTo(line[0][0] * w, line[0][1] * h)
                    for px, py in line[1:]:
                        path.lineTo(px * w, py * h)
                    if len(line) == 1:
                        path.lineTo(line[0][0] * w + 0.1, line[0][1] * h)
                    canvas.drawPath(path, paint)
        surf.makeImageSnapshot().save(out, skia.kPNG)
    return out


MAX_STROKES, MAX_POINTS = 20, 500


def _strokes(v) -> list:
    if not isinstance(v, list) or not v or len(v) > MAX_STROKES:
        raise FeedbackError(f"strokes must be 1 to {MAX_STROKES} lines")
    out = []
    for line in v:
        if not isinstance(line, list) or not 1 <= len(line) <= MAX_POINTS:
            raise FeedbackError(f"a stroke has 1 to {MAX_POINTS} points")
        out.append([[_frac("stroke x", p[0]), _frac("stroke y", p[1])] if isinstance(p, list) and len(p) == 2
                    else _frac("stroke", None) for p in line])
    return out


def add(scene_path, t: float, text: str, x: float | None = None, y: float | None = None,
        layer: str | None = None, kind: str = "pin", extra: dict | None = None,
        sound: str | None = None, word: str | None = None, strokes=None) -> dict:
    """A note on a time, optionally at a point, a layer, a sound (and one of its words) or drawn strokes."""
    if strokes is not None:
        kind = "draw"
    if kind not in KINDS:
        raise FeedbackError(f"kind must be one of {', '.join(KINDS)}")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
        raise FeedbackError("t must be a time in seconds, 0 or more")
    if not isinstance(text, str) or not text.strip():
        raise FeedbackError("text is empty")
    data = load(scene_path)
    used = {n.get("id") for n in data["notes"]}
    n = 1
    while f"n{n}" in used:
        n += 1
    note = {"id": f"n{n}", "t": float(t), "kind": kind, "text": text.strip(), "status": "open",
            "scene": scene_hash(scene_path), "created": int(time.time()), "replies": []}
    if x is not None or y is not None:
        note["x"], note["y"] = _frac("x", x), _frac("y", y)
    if layer:
        note["layer"] = layer
    if sound:
        note["sound"] = str(sound)
    if word:
        note["word"] = str(word)
    if strokes is not None:
        note["strokes"] = _strokes(strokes)
    if extra:
        note.update({k: v for k, v in extra.items() if k not in note})
    data["version"] = VERSION
    data["notes"].append(note)
    _save(scene_path, data)
    return note


def _find(data: dict, note_id: str) -> dict:
    for n in data["notes"]:
        if n.get("id") == note_id:
            return n
    raise FeedbackError(f"no note {note_id}")


def listing(scene_path, status: str | None = "open") -> list[dict]:
    """Notes in time order, each with `stale` set when the scene changed after it was made."""
    if status not in (None, "open", "resolved"):
        raise FeedbackError("status must be open or resolved")
    now = scene_hash(scene_path)
    out = [dict(n, stale=n.get("scene") != now) for n in load(scene_path)["notes"]
           if status is None or n.get("status") == status]
    return sorted(out, key=lambda n: n.get("t", 0))


def get(scene_path, note_id: str) -> dict:
    n = _find(load(scene_path), note_id)
    return dict(n, stale=n.get("scene") != scene_hash(scene_path))


def reply(scene_path, note_id: str, text: str, by: str = "agent") -> dict:
    if not isinstance(text, str) or not text.strip():
        raise FeedbackError("text is empty")
    data = load(scene_path)
    n = _find(data, note_id)
    n["replies"].append({"by": by, "text": text.strip(), "at": int(time.time())})
    _save(scene_path, data)
    return n


def reopen(scene_path, note_id: str) -> dict:
    data = load(scene_path)
    n = _find(data, note_id)
    n["status"] = "open"
    _save(scene_path, data)
    return n


def resolve(scene_path, note_id: str, text: str | None = None, by: str = "agent") -> dict:
    data = load(scene_path)
    n = _find(data, note_id)
    n["status"] = "resolved"
    if text and text.strip():
        n["replies"].append({"by": by, "text": text.strip(), "at": int(time.time())})
    _save(scene_path, data)
    return n
