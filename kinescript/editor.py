"""`kine edit`: a local web page to scrub a scene, pin notes on frames, and leave them for an agent.

Frames and layer hits come from the same render path as `kine still` and `kine probe`, loaded fresh
per request, so an agent's edit to the scene file shows on the next frame. Notes go to the scene's
`<scene>.feedback.json` (see feedback.py).
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import feedback

MAX_BODY = 1 << 20
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
_draw_lock = threading.Lock()  # one Renderer draws at a time


class BadRequest(ValueError):
    pass


def _num(q: dict, key: str, default=None, lo=-math.inf, hi=math.inf) -> float:
    raw = q.get(key, [None])[0]
    if raw is None:
        if default is None:
            raise BadRequest(f"missing {key}")
        return default
    try:
        v = float(raw)
    except ValueError:
        raise BadRequest(f"{key} must be a number") from None
    if not math.isfinite(v):
        raise BadRequest(f"{key} must be a number")
    return min(max(v, lo), hi)


def _host_name(header: str) -> str:
    if header.startswith("["):
        return header.split("]")[0] + "]"
    return header.rsplit(":", 1)[0] if ":" in header else header


COMMON = ("at", "dur")
CONTROLS = {"text": ("text", "x", "y", "size", "fill") + COMMON, "counter": ("x", "y", "size", "fill") + COMMON,
            "circle": ("x", "y", "r", "fill") + COMMON, "rect": ("x", "y", "w", "h", "fill") + COMMON,
            "image": ("x", "y", "w", "h") + COMMON, "video": ("x", "y", "w", "h") + COMMON,
            "path": ("stroke",) + COMMON, "line": ("stroke",) + COMMON, "arrow": ("stroke",) + COMMON}
EDITABLE = {k for keys in CONTROLS.values() for k in keys}


def _walk(layers):
    for raw in layers if isinstance(layers, list) else []:
        if isinstance(raw, dict):
            yield raw
            yield from _walk(raw.get("children") or raw.get("layers"))


def _raw_layers(doc) -> dict:
    return {r["id"]: r for r in _walk(doc.get("layers")) if isinstance(r.get("id"), str)}


def scene_info(path: Path) -> dict:
    from .schema import loads
    from .verify import probe
    with _draw_lock:
        info = probe(str(path))
    c = info["canvas"]
    raws = _raw_layers(loads(path.read_text(encoding="utf-8"))[0])
    layers = []
    for row in info["layers"]:
        raw = raws.get(row["id"], {})
        keys = CONTROLS.get(row["type"], ("x", "y") + COMMON)
        layers.append({"id": row["id"], "type": row["type"], "in": row["in"], "out": row["out"], "parent": row["parent"],
                       "controls": [{"key": k, "value": raw.get(k), "editable": isinstance(raw.get(k), (int, float, str))
                                     and not isinstance(raw.get(k), bool) or k not in raw} for k in keys]})
    video = path.with_suffix(".mp4")
    return {"name": path.name, "w": c["w"], "h": c["h"], "fps": c["fps"], "duration": c["duration"],
            "hash": feedback.scene_hash(path), "video": video.name if video.is_file() else None, "layers": layers}


def set_props(path: Path, layer: str, changes: dict, base_hash: str, t: float = 0) -> dict:
    """Write properties of one layer into the scene file in one save, and log them as one note for the agent.

    Refuses when the file changed since the page loaded it, holds comments or other non-strict JSON
    (rewriting would drop them), or the result would not load as a scene."""
    from .scene import Scene, SceneError
    if not isinstance(changes, dict) or not changes:
        raise BadRequest("nothing to change")
    for key, value in changes.items():
        if key not in EDITABLE:
            raise BadRequest(f"{key} cannot be edited here")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise BadRequest("value must be a number or text")
    if feedback.scene_hash(path) != base_hash:
        raise BadRequest("the scene changed since this page loaded it; reload")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise BadRequest("this scene has comments or loose JSON, so the editor will not rewrite it; leave a note instead") from None
    raw = _raw_layers(doc).get(layer)
    if raw is None:
        raise BadRequest(f"no layer {layer}")
    before = {}
    for key, value in changes.items():
        if key in raw and not isinstance(raw[key], (int, float, str)):
            raise BadRequest(f"{layer}.{key} is animated or computed; leave a note instead")
        before[key] = raw.get(key)
        raw[key] = value
    try:
        Scene(doc, str(path.parent)).close()
    except (SceneError, ValueError) as e:
        raise BadRequest(f"that value breaks the scene: {e}") from None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    text = ", ".join(f"{layer}.{k}: {before[k]!r} -> {v!r}" for k, v in changes.items())
    feedback.add(path, t, text, layer=layer, kind="prop",
                 extra={"change": {k: {"from": before[k], "to": v} for k, v in changes.items()}})
    return scene_info(path)


def frame_png(path: Path, t: float, scale: float) -> bytes:
    from .output import still
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="kine_edit_")
    os.close(fd)
    try:
        with _draw_lock:
            still(str(path), t, tmp, scale)
        return Path(tmp).read_bytes()
    finally:
        Path(tmp).unlink(missing_ok=True)


def hit(path: Path, x: float, y: float, t: float) -> dict:
    """The topmost layer whose box holds the point (x, y as fractions of the frame) at time t."""
    from .verify import probe
    with _draw_lock:
        info = probe(str(path), t)
    c = info["canvas"]
    px, py = x * c["w"], y * c["h"]
    found = None
    for lid, b in info["visible"].items():  # boxes are recorded in draw order, so the last match is on top
        left, top, right, bottom = b["rect"]
        if left <= px <= right and top <= py <= bottom:
            found = lid
    boxes = {lid: [b["rect"][0] / c["w"], b["rect"][1] / c["h"], b["rect"][2] / c["w"], b["rect"][3] / c["h"]]
             for lid, b in info["visible"].items()}
    pos = {lid: b["pos"] for lid, b in info["visible"].items()}
    return {"layer": found, "layers": list(info["visible"]), "boxes": boxes, "pos": pos}


def make_handler(path: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "kine-edit"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def _guard(self, fn):
            if _host_name(self.headers.get("Host") or "") not in LOCAL_HOSTS:  # blocks DNS rebinding
                return self._json({"error": "bad host"}, 403)
            try:
                fn()
            except (BadRequest, feedback.FeedbackError) as e:
                self._json({"error": str(e)}, 400)
            except FileNotFoundError:
                self._json({"error": f"{path.name} is gone"}, 404)
            except Exception as e:  # the scene may be mid-edit; the page shows the message
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_GET(self):
            self._guard(self._get)

        def do_POST(self):
            self._guard(self._post)

        def _get(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                html = resources.files("kinescript.editor_ui").joinpath("index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/scene":
                return self._json(scene_info(path))
            if u.path == "/frame":
                png = frame_png(path, _num(q, "t", lo=0), _num(q, "scale", 0.5, 0.1, 1.0))
                return self._send(200, png, "image/png")
            if u.path == "/hit":
                return self._json(hit(path, _num(q, "x", lo=0, hi=1), _num(q, "y", lo=0, hi=1), _num(q, "t", lo=0)))
            if u.path == "/notes":
                status = q.get("status", ["open"])[0]
                return self._json({"notes": feedback.listing(path, None if status == "all" else status)})
            if u.path == "/video":
                return self._video()
            self._json({"error": "not found"}, 404)

        def _video(self):
            vid = path.with_suffix(".mp4")
            if not vid.is_file():
                return self._json({"error": "no render yet"}, 404)
            size = vid.stat().st_size
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if m and (m[1] or m[2]):
                if m[1]:
                    start = int(m[1])
                    end = int(m[2]) if m[2] else end
                else:
                    start = max(size - int(m[2]), 0)
                end = min(end, size - 1)
                if start > end:
                    return self._send(416, b"", "video/mp4", {"Content-Range": f"bytes */{size}"})
            else:
                m = None
            with vid.open("rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if m:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._send(206 if m else 200, body, "video/mp4", extra)

        def _post(self):
            if urlparse(self.path).path not in ("/notes", "/prop"):
                return self._json({"error": "not found"}, 404)
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise BadRequest("send application/json")
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                raise BadRequest("body is empty or too large")
            try:
                body = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                raise BadRequest("body is not valid JSON") from None
            if not isinstance(body, dict):
                raise BadRequest("body must be an object")
            action = "prop" if urlparse(self.path).path == "/prop" else body.get("action", "add")
            if action == "prop":
                changes = body.get("changes") or {body.get("key"): body.get("value")}
                return self._json(set_props(path, body.get("layer"), changes, body.get("hash"), body.get("t") or 0))
            if action == "add":
                note = feedback.add(path, body.get("t"), body.get("text"), body.get("x"), body.get("y"),
                                    body.get("layer"), body.get("kind", "pin"))
                return self._json({"note": note}, 201)
            if action == "resolve":
                return self._json({"note": feedback.resolve(path, body.get("id"), body.get("text"), by="user")})
            if action == "reopen":
                return self._json({"note": feedback.reopen(path, body.get("id"))})
            raise BadRequest("action must be add, resolve or reopen")

    return Handler


def serve(scene: str, port: int = 8765, open_browser: bool = True) -> None:
    path = Path(scene).resolve()
    if not path.is_file():
        raise ValueError(f"no scene file at {path}")
    scene_info(path)  # fail early on a scene that does not load
    os.environ["KINE_BACKEND"] = "cpu"  # the GPU context belongs to the thread that made it; requests run on many
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(path))
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"editing {path.name} at {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
