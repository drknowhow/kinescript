"""kine: command line for Kinescript."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__

STARTER = {
    "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": "#0a0c10"},
    "grade": {"vignette": 0.3},
    "layers": [
        {"id": "bg", "type": "shader", "preset": "mesh", "opacity": 0.35},
        {"use": "titleCard", "id": "title", "title": "Your **title**", "subtitle": "a line of context",
         "at": 0.3, "dur": 3.5},
        {"use": "lowerThird", "id": "who", "title": "Name Surname", "subtitle": "Role", "at": "title.out+0.2", "dur": 3},
    ],
}


def _emit(obj):
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kine", description="Kinescript: motion graphics and video edits as data.")
    ap.add_argument("--version", action="version", version=f"kinescript {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a scene to .mp4/.mov/.webm/.gif or a .png sequence, or its sound "
                                      "alone to .wav/.m4a/.mp3/.flac")
    r.add_argument("scene")
    r.add_argument("-o", "--out")
    r.add_argument("-q", "--quality", choices=["draft", "standard", "high"], default="standard")
    r.add_argument("--scale", type=float, default=1.0, help="resolution multiplier, e.g. 0.5 for a fast preview")
    r.add_argument("--from", dest="start", type=float)
    r.add_argument("--to", dest="end", type=float)
    r.add_argument("-j", "--workers", type=int)
    r.add_argument("--transparent", action="store_true", help="drop the background (use .mov or .webm)")

    s = sub.add_parser("still", help="render one frame to PNG/JPG")
    s.add_argument("scene")
    s.add_argument("--at", required=True, help='seconds; negative counts from the end; "end" is the last frame')
    s.add_argument("-o", "--out")
    s.add_argument("--scale", type=float, default=1.0)

    lk = sub.add_parser("look", help="contact sheet: many frames in one image, for review")
    lk.add_argument("scene")
    lk.add_argument("-n", "--count", type=int, default=12)
    lk.add_argument("--cols", type=int)
    lk.add_argument("--width", type=int, default=480, help="width of each thumbnail")
    lk.add_argument("--at", type=float, nargs="*", help="explicit times instead of even spacing")
    lk.add_argument("--boxes", action="store_true", help="outline every layer with its id")
    lk.add_argument("-o", "--out")

    ck = sub.add_parser("check", help="lint layout, timing, contrast and reading time; prints JSON")
    ck.add_argument("scene")
    ck.add_argument("--step", type=float, default=0.25)

    pr = sub.add_parser("probe", help="resolved timeline and layer boxes at --at; or, for a media file, "
                                      "its duration, size, fps and audio")
    pr.add_argument("scene")
    pr.add_argument("--at", type=float)

    sub.add_parser("docs", help="print the scene reference")
    nw = sub.add_parser("new", help="write a starter scene")
    nw.add_argument("path", nargs="?", default="scene.json")

    w = sub.add_parser("watch", help="re-render a contact sheet whenever the scene file changes")
    w.add_argument("scene")
    w.add_argument("-n", "--count", type=int, default=12)

    sub.add_parser("mcp", help="run the MCP server on stdio")
    sub.add_parser("doctor", help="report ffmpeg, GPU and font availability")

    a = ap.parse_args(argv)
    try:
        return _run(a)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        _emit({"ok": False, "error": str(e)})
        return 2


def _default_out(scene: str, suffix: str) -> str:
    p = Path(scene)
    return str(p.with_name(p.stem + suffix))


def _run(a) -> int:
    if a.cmd == "docs":
        from .docs import reference
        sys.stdout.write(reference())
        return 0
    if a.cmd == "new":
        p = Path(a.path)
        if p.exists():
            raise ValueError(f"{p} exists; pick another name")
        p.write_text(json.dumps(STARTER, indent=2), encoding="utf-8")
        _emit({"ok": True, "out": str(p)})
        return 0
    if a.cmd == "mcp":
        from .mcp import serve
        serve()
        return 0
    if a.cmd == "doctor":
        from . import media
        from .render import gpu_context, Renderer
        from .scene import Scene
        info = {"version": __version__, "ffmpeg": media.FFMPEG, "gpu": gpu_context() is not None}
        r = Renderer(Scene({"canvas": {"w": 64, "h": 64}, "layers": [{"type": "text", "text": "kine"}]}))
        info["font"] = r._typeface(r.s.theme["font"], 600, False).getFamilyName()
        _emit(info)
        return 0
    if a.cmd == "render":
        from . import output
        from .output import render
        output.WARM = 0  # one render, then the process exits: shut the workers down inside it, as timed
        out = a.out or _default_out(a.scene, ".mp4")
        _emit(dict(ok=True, **render(a.scene, out, a.quality, a.workers, a.scale, a.start, a.end, a.transparent)))
        return 0
    if a.cmd == "still":
        from .output import still
        at = str(a.at)
        out = a.out or _default_out(a.scene, f"_{at.lstrip('-')}{'s' if at != 'end' else ''}.png")
        _emit(dict(ok=True, **still(a.scene, at, out, a.scale)))
        return 0
    if a.cmd == "look":
        from .verify import contact_sheet
        out = a.out or _default_out(a.scene, "_sheet.png")
        _emit(dict(ok=True, **contact_sheet(a.scene, out, a.count, a.cols, a.width, a.at, a.boxes)))
        return 0
    if a.cmd == "check":
        from .verify import check
        res = check(a.scene, a.step)
        _emit(res)
        return 0 if res["ok"] else 1
    if a.cmd == "probe":
        from .verify import probe
        if Path(a.scene).suffix.lower() != ".json" and Path(a.scene).exists():
            from . import media
            _emit(dict(ok=True, src=a.scene, **media.probe(a.scene)))
            return 0
        _emit(probe(a.scene, a.at))
        return 0
    if a.cmd == "watch":
        from .verify import contact_sheet
        p = Path(a.scene)
        last = 0.0
        out = _default_out(a.scene, "_sheet.png")
        print(f"watching {p} -> {out} (Ctrl+C to stop)", file=sys.stderr)
        while True:
            m = p.stat().st_mtime
            if m != last:
                last = m
                try:
                    contact_sheet(a.scene, out, a.count)
                    print(f"{time.strftime('%H:%M:%S')} updated", file=sys.stderr)
                except (ValueError, RuntimeError) as e:
                    print(f"{time.strftime('%H:%M:%S')} error: {e}", file=sys.stderr)
            time.sleep(0.5)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
