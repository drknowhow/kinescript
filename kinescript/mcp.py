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
     "text, low contrast, tiny text, too little reading time. Returns issues with time ranges and fixes.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}}, "required": ["scene"]}},
    {"name": "kine_look", "description": "Contact sheet: `count` frames in one labelled image. boxes=true outlines each "
     "layer with its id. Use it to review motion and layout without watching video.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"},
                                                      "count": {"type": "integer", "default": 12},
                                                      "times": {"type": "array", "items": {"type": "number"}},
                                                      "boxes": {"type": "boolean", "default": False},
                                                      "width": {"type": "integer", "default": 400}}, "required": ["scene"]}},
    {"name": "kine_still", "description": "One full-size frame at time `at` (seconds), returned as an image.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}, "at": {"type": "number"},
                                                      "scale": {"type": "number", "default": 0.5}},
                     "required": ["scene", "at"]}},
    {"name": "kine_probe", "description": "Resolved timeline (every layer's in/out) and, with `at`, every visible "
     "layer's screen box, alpha and text at that time.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}, "at": {"type": "number"}},
                     "required": ["scene"]}},
    {"name": "kine_render", "description": "Render to a file (.mp4, .mov ProRes 4444 alpha, .webm VP9 alpha, .gif, or a "
     ".png sequence; .wav/.m4a/.mp3/.flac for the sound alone). quality draft|standard|high; scale <1 for previews; "
     "from/to (s) for a range.",
     "inputSchema": {"type": "object", "properties": {"scene": _SCENE, "base": {"type": "string"}, "out": {"type": "string"},
                                                      "quality": {"type": "string", "enum": ["draft", "standard", "high"]},
                                                      "scale": {"type": "number"}, "from": {"type": "number"},
                                                      "to": {"type": "number"}}, "required": ["scene", "out"]}},
]


def _scene(args: dict):
    from .scene import Scene, load
    sc = args["scene"]
    if isinstance(sc, dict):
        return Scene(sc, args.get("base") or os.getcwd())
    return load(sc)


def _image(path: str) -> dict:
    return {"type": "image", "mimeType": "image/png", "data": base64.b64encode(Path(path).read_bytes()).decode()}


def _text(obj) -> dict:
    return {"type": "text", "text": obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False)}


def call(name: str, args: dict) -> list[dict]:
    if name == "kine_docs":
        from .docs import reference
        return [_text(reference())]
    if name == "kine_check":
        from .verify import check
        return [_text(check(_scene(args)))]
    if name == "kine_probe":
        from .verify import probe
        return [_text(probe(_scene(args), args.get("at")))]
    if name == "kine_look":
        from .verify import contact_sheet
        out = str(Path(tempfile.gettempdir()) / f"kine_look_{os.getpid()}.png")
        info = contact_sheet(_scene(args), out, int(args.get("count", 12)), None, int(args.get("width", 400)),
                             args.get("times"), bool(args.get("boxes")))
        return [_image(out), _text(info)]
    if name == "kine_still":
        from .output import still
        out = str(Path(tempfile.gettempdir()) / f"kine_still_{os.getpid()}.png")
        still(_scene(args), float(args["at"]), out, float(args.get("scale", 0.5)))
        return [_image(out)]
    if name == "kine_render":
        from .output import render
        sc = args["scene"]
        if isinstance(sc, dict):
            sc = _scene(args)
        res = render(sc, args["out"], args.get("quality", "standard"), None, float(args.get("scale", 1.0)),
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
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _reply(None, error={"code": -32700, "message": "parse error"})
            continue
        method, id_ = req.get("method"), req.get("id")
        if id_ is None:
            continue  # notification
        if method == "initialize":
            _reply(id_, {"protocolVersion": req.get("params", {}).get("protocolVersion", "2024-11-05"),
                         "capabilities": {"tools": {}},
                         "serverInfo": {"name": "kinescript", "version": __version__},
                         "instructions": "Kinescript renders motion graphics and video edits from JSON scenes. "
                                         "Call kine_docs first, then write a scene, kine_check it, kine_look at it, "
                                         "and kine_render it."})
        elif method == "tools/list":
            _reply(id_, {"tools": TOOLS})
        elif method == "tools/call":
            p = req.get("params", {})
            try:
                _reply(id_, {"content": call(p.get("name"), p.get("arguments") or {})})
            except (ValueError, RuntimeError, FileNotFoundError, KeyError) as e:
                _reply(id_, {"content": [_text(f"error: {e}")], "isError": True})
            except Exception as e:  # noqa: BLE001 - report unexpected failures to the client instead of dying
                _reply(id_, {"content": [_text(f"internal error: {e}\n{traceback.format_exc()[-1500:]}")],
                             "isError": True})
        elif method == "ping":
            _reply(id_, {})
        else:
            _reply(id_, error={"code": -32601, "message": f"method not found: {method}"})
