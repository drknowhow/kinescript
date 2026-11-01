"""Minimal MCP server (JSON-RPC 2.0 over stdio, newline-delimited). No dependencies.

Tools take `scene` as a path to a .json file or as the scene object itself.
Look and still return the image inline, so an agent with vision sees the frames directly.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from . import __version__

_SCENE = {"description": "Path to a scene .json, or the scene object itself", "type": ["string", "object"]}
TOOLS = [
    {"name": "kine_docs", "description": "Kinescript scene reference: layer types, properties, timing, motions, "
     "components, eases. Read once before writing a scene.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "kine_check", "description": "Lint a scene: load errors, text off canvas or in the safe margin, overlapping "
     "text, low contrast, tiny text, too little reading time, and sound (a voice cut off or under the music, voices "
     "overlapping, an effect late on its cue, silence, loudness and peaks). Returns issues with time ranges and fixes.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}}, "required": ["scene"]}},
    {"name": "kine_look", "description": "Contact sheet: `count` frames in one labelled image. boxes=true outlines each "
     "layer with its id. wire=true draws every model's drawn triangles; depth=true shows each stage's depth image, "
     "nearer brighter. audio=true draws the sound under the frames on the same time axis (a lane per role and the "
     "mix, with levels, the duck and issues in red) and returns its measurements. times are read as kine_still reads "
     "at; one outside the scene draws its first or last frame, with a note. Use it to review motion, layout "
     "and sound without watching video.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"},
                                                      "count": {"type": "integer", "default": 12},
                                                      "times": {"type": "array", "items": {"type": "number"}},
                                                      "boxes": {"type": "boolean", "default": False},
                                                      "wire": {"type": "boolean", "default": False},
                                                      "depth": {"type": "boolean", "default": False},
                                                      "audio": {"type": "boolean", "default": False},
                                                      "width": {"type": "integer", "default": 400}}, "required": ["scene"]}},
    {"name": "kine_still", "description": "One full-size frame at time `at` (seconds), returned as an image, then "
     "its frame number, time and size (with notes when `at` is outside the scene).",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"},
                                                      "at": {"type": ["number", "string"], "description":
                                                             "seconds; negative counts from the end; \"end\" is "
                                                             "the last frame"},
                                                      "scale": {"type": "number", "default": 0.5}},
                     "required": ["scene", "at"]}},
    {"name": "kine_probe", "description": "Resolved timeline (every layer's in/out) and, with `at`, every visible "
     "layer's screen box, alpha and text at that time. Given a media file instead of a scene, its duration, size, "
     "fps and audio.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}, "at": {"type": "number"}},
                     "required": ["scene"]}},
    {"name": "kine_voice", "description": "Speak each sound's `text` into its file (voice/<id>.wav unless it has a "
     "src) and write its word times, so layers can cue from words (\"vo1:render\") and captions can follow it. Only "
     "sounds whose text, speaker, rate, backend or options changed are spoken again, unless force. A sound's own "
     "\"backend\" picks its voice plugin; backend here is the one for sounds without. Tags such as [laughs] are "
     "acted by backends that can and listed under \"ignored\" by the rest. Run it before check and render.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"},
                                                      "only": {"type": "array", "items": {"type": "string"}},
                                                      "force": {"type": "boolean", "default": False},
                                                      "backend": {"type": "string"}}, "required": ["scene"]}},
    {"name": "kine_render", "description": "Render to a file (.mp4, .mov ProRes 4444 alpha, .webm VP9 alpha, .gif, or a "
     ".png sequence; .wav/.m4a/.mp3/.flac for the sound alone). quality draft|standard|high; scale <1 for previews; "
     "from/to (s) for a range: every frame on screen from `from` until `to`. The result's notes say when `to` was "
     "past the end.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}, "out": {"type": "string"},
                                                      "quality": {"type": "string", "enum": ["draft", "standard", "high"]},
                                                      "scale": {"type": "number"}, "from": {"type": "number"},
                                                      "to": {"type": "number"}}, "required": ["scene", "out"]}},
]


def _need(args: dict, key: str):
    if args.get(key) is None:
        raise ValueError(f"missing argument \"{key}\"")
    return args[key]


def _scene(args: dict):
    from .scene import Scene, load
    sc = _need(args, "scene")
    if isinstance(sc, dict):
        return Scene(sc, args.get("base") or os.getcwd())
    if not isinstance(sc, str):
        raise ValueError("scene must be a path to a .json file or the scene object itself")
    return load(sc)


def _image(path: str) -> dict:
    """The image inline; the temporary file is removed."""
    p = Path(path)
    try:
        data = p.read_bytes()
    finally:
        p.unlink(missing_ok=True)
    return {"type": "image", "mimeType": "image/png", "data": base64.b64encode(data).decode()}


def _text(obj) -> dict:
    return {"type": "text", "text": obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False)}


def call(name: str, args: dict) -> list[dict]:
    if name == "kine_docs":
        from .docs import reference
        return [_text(reference())]
    if name == "kine_check":
        from .scene import SceneError
        from .verify import _result, check
        try:
            sc = _scene(args)
        except SceneError as e:  # the issues, as `kine check` gives them, not a one-line error
            return [_text(_result(e.issues))]
        return [_text(check(sc))]
    if name == "kine_probe":
        from .verify import probe, probe_media
        sc = _need(args, "scene")
        if isinstance(sc, str) and Path(sc).suffix.lower() != ".json" and Path(sc).is_file():
            return [_text(probe_media(sc))]
        return [_text(probe(_scene(args), args.get("at")))]
    if name == "kine_look":
        from .verify import contact_sheet
        out = str(Path(tempfile.gettempdir()) / f"kine_look_{os.getpid()}.png")
        info = contact_sheet(_scene(args), out, int(args.get("count", 12)), None, int(args.get("width", 400)),
                             args.get("times"), bool(args.get("boxes")), bool(args.get("audio")),
                             wire=bool(args.get("wire")), depth=bool(args.get("depth")))
        del info["out"]  # a temporary file, gone once read
        return [_image(out), _text(info)]
    if name == "kine_still":
        from .output import still
        out = str(Path(tempfile.gettempdir()) / f"kine_still_{os.getpid()}.png")
        info = still(_scene(args), _need(args, "at"), out, float(args.get("scale", 0.5)))
        del info["out"]  # a temporary file, gone once read
        return [_image(out), _text(info)]  # which frame it drew, and a note when `at` was outside the scene
    if name == "kine_voice":
        from .voice import generate
        return [_text(generate(_need(args, "scene"), args.get("only"), bool(args.get("force")), args.get("backend"),
                               args.get("base") or os.getcwd()))]
    if name == "kine_render":
        from .output import render
        sc, out = _need(args, "scene"), _need(args, "out")
        if isinstance(sc, dict):
            sc = _scene(args)
        if not isinstance(out, str) or not out.strip():
            raise ValueError("out must be a file path")
        out = os.path.join(os.getcwd(), out)  # the result says where a relative out landed: the server's folder
        res = render(sc, out, args.get("quality", "standard"), None, float(args.get("scale", 1.0)),
                     args.get("from"), args.get("to"))
        return [_text(res)]
    raise ValueError(f"unknown tool {name}")


def _reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def serve():
    try:
        _serve()
    except OSError:  # the client closed its end of the pipe (BrokenPipeError, or EINVAL on Windows)
        sys.stdout = open(os.devnull, "w")  # so the interpreter's final flush stays quiet


def _serve():
    try:  # clients send UTF-8; Windows read stdin in its code page, so "café" arrived as "cafÃ©"
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _reply(None, error={"code": -32700, "message": "parse error"})
            continue
        if not isinstance(req, dict):  # a number, a string or a batch: each of these ended the server
            _reply(None, error={"code": -32600, "message": "invalid request: send one JSON-RPC object per line"})
            continue
        try:
            _handle(req)
        except Exception as e:  # noqa: BLE001 - one bad request must not end the server
            if req.get("id") is not None:
                _reply(req["id"], error={"code": -32603, "message": f"internal error: {e}"})


def _handle(req: dict):
    method, id_ = req.get("method"), req.get("id")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    if id_ is None:
        return  # notification
    if method == "initialize":
        _reply(id_, {"protocolVersion": params.get("protocolVersion", "2024-11-05"),
                     "capabilities": {"tools": {}},
                     "serverInfo": {"name": "kinescript", "version": __version__},
                     "instructions": "Kinescript renders motion graphics and video edits from JSON scenes. "
                                     "Call kine_docs first, then write a scene (kine_voice it if a sound has "
                                     "text), kine_check it, kine_look at it, "
                                     "and kine_render it."})
    elif method == "tools/list":
        _reply(id_, {"tools": TOOLS})
    elif method == "tools/call":
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            _reply(id_, error={"code": -32602, "message": "arguments must be an object"})
            return
        try:
            _reply(id_, {"content": call(params.get("name"), args)})
        except (ValueError, RuntimeError, OSError, KeyError) as e:
            from .media import explain  # which of ffmpeg and ffprobe is missing, not "[WinError 2]"
            _reply(id_, {"content": [_text(f"error: {explain(e)}")], "isError": True})
        except Exception as e:  # noqa: BLE001 - report unexpected failures to the client instead of dying
            from .cli import ISSUES_URL
            _reply(id_, {"content": [_text(f"internal error: {type(e).__name__}: {e}. This is a bug in Kinescript; "
                                           f"please report it at {ISSUES_URL} with the call and the scene.\n"
                                           f"{traceback.format_exc()[-1500:]}")], "isError": True})
    elif method == "ping":
        _reply(id_, {})
    else:
        _reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
