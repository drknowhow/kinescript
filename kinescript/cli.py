"""kine: command line for Kinescript."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
    r.add_argument("--from", dest="start", type=float, help="seconds: start with the frame on screen then")
    r.add_argument("--to", dest="end", type=float, help="seconds: end with the last frame that starts before then "
                                                          "(past the end of the scene: the end, with a note)")
    r.add_argument("-j", "--workers", type=int, help="render processes, at most one per frame and one per CPU "
                                                     "(by default kine picks)")
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
    lk.add_argument("--at", nargs="*", help="explicit times instead of even spacing, read as still reads --at")
    lk.add_argument("--boxes", action="store_true", help="outline every layer with its id")
    lk.add_argument("--wire", action="store_true", help="draw every model's drawn triangles over the frames")
    lk.add_argument("--depth", action="store_true", help="show each stage's depth image, nearer brighter, over "
                    "the frames dimmed")
    lk.add_argument("--audio", action="store_true", help="draw the sound under the frames (a lane per role and the "
                    "mix, with levels, the duck and issues) and print its measurements")
    lk.add_argument("--trails", action="store_true", help="draw each moving layer's path over the frames, a dot a "
                    "frame, so their spacing shows speed, eases and stops")
    lk.add_argument("--onion", type=int, default=0, metavar="N", help="draw the N frames before each frame over it, "
                    "fainter the older they are")
    lk.add_argument("-o", "--out")

    ck = sub.add_parser("check", help="lint layout, timing, contrast, reading time and sound; prints JSON")
    ck.add_argument("scene")
    ck.add_argument("--step", type=float, default=0.25)

    pr = sub.add_parser("probe", help="resolved timeline and layer boxes at --at; or, for a media file, "
                                      "its duration, size, fps and audio")
    pr.add_argument("scene")
    when = pr.add_mutually_exclusive_group()
    when.add_argument("--at", type=float)
    when.add_argument("--curve", metavar="ID.KEY", help="a layer's key at every frame and how fast it changes, where "
                      "it holds still and its fastest frame, such as title.x")

    v = sub.add_parser("voice", help="speak each sound's text into its file and time its words (only the ones "
                                     "whose text changed, unless --force)")
    v.add_argument("scene")
    v.add_argument("--only", nargs="+", metavar="ID", help="speak just these sounds")
    v.add_argument("--force", action="store_true", help="speak them again even if their text did not change")
    v.add_argument("--backend", help="which installed voice backend to use, when there are several")

    sub.add_parser("docs", help="print the scene reference")
    nw = sub.add_parser("new", help="write a starter scene")
    nw.add_argument("path", nargs="?", default="scene.json")

    w = sub.add_parser("watch", help="re-render a contact sheet whenever the scene file changes")
    w.add_argument("scene")
    w.add_argument("-n", "--count", type=int, default=12)

    sub.add_parser("mcp", help="run the MCP server on stdio")
    sub.add_parser("doctor", help="report ffmpeg, GPU and font availability")

    a = ap.parse_args(argv)
    # UTF-8 even into a Windows pipe: its default code page can't hold a CJK layer id, and the failed write after
    # a render had succeeded was reported as the render failing.
    for f in (sys.stdout, sys.stderr):
        try:
            f.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # replaced by something that is not a text stream
            pass
    try:
        return _run(a)
    except (ValueError, RuntimeError, OSError) as e:
        _emit({"ok": False, "error": _message(e)})
        return BAD
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return STOPPED
    except Exception as e:  # noqa: BLE001 - a bug: one line, not a traceback with the code a lint error has
        _emit({"ok": False, "error": crash_message(e)})
        if os.environ.get("KINE_DEBUG"):
            traceback.print_exc()
        return CRASH


# Exit codes, as kine docs lists them. A crash exited 1, the code `check` uses for a scene with errors.
OK, ISSUES, BAD, CRASH, STOPPED = 0, 1, 2, 3, 130
ISSUES_URL = "https://github.com/drknowhow/kinescript/issues"


def _message(e: BaseException) -> str:
    if isinstance(e, FileNotFoundError):
        from .media import explain  # which of ffmpeg and ffprobe is missing, not "[WinError 2]"
        return explain(e)
    return str(e)


def crash_message(e: BaseException) -> str:
    """One line for an error Kinescript did not expect: what it was, where, and where to report it."""
    tb = traceback.extract_tb(e.__traceback__)
    where = next((f for f in reversed(tb) if Path(f.filename).parent.name == "kinescript"), tb[-1] if tb else None)
    at = f" at {Path(where.filename).name}:{where.lineno}" if where else ""
    return (f"internal error: {type(e).__name__}: {e}{at}. This is a bug in Kinescript; please report it at "
            f"{ISSUES_URL} with the command and the scene (KINE_DEBUG=1 prints the traceback)")


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
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(STARTER, indent=2), encoding="utf-8")
        _emit({"ok": True, "out": str(p)})
        return 0
    if a.cmd == "mcp":
        from .mcp import serve
        serve()
        return 0
    if a.cmd == "doctor":
        from . import media
        from .render import backend_setting, gpu_context, Renderer
        from .scene import Scene
        import shutil
        tools = {name: shutil.which(path) for name, path in (("ffmpeg", media.FFMPEG), ("ffprobe", media.FFPROBE))}
        info = {"version": __version__, **tools, "backend": backend_setting(), "gpu": gpu_context() is not None}
        r = Renderer(Scene({"canvas": {"w": 64, "h": 64}, "layers": [{"type": "text", "text": "kine"}]}))
        info["font"] = r._typeface(r.s.theme["font"], 600, False).getFamilyName()
        missing = media.missing_tools()
        if missing:
            info["error"] = (f"{' and '.join(missing)} not found on the PATH; Kinescript needs both to read and write "
                             f"media: {media.INSTALL_FFMPEG}")
        _emit(info)
        return ISSUES if missing else OK
    if a.cmd == "render":
        from . import output
        from .output import render
        output.WARM = 0  # one render, then the process exits: shut the workers down inside it, as timed
        out = a.out or _default_out(a.scene, ".mp4")
        _emit(dict(ok=True, **render(a.scene, out, a.quality, a.workers, a.scale, a.start, a.end, a.transparent)))
        return 0
    if a.cmd == "still":
        from .output import still
        at = str(a.at).strip()  # --at -1 is 1 s before the end: its own name, not the same file as --at 1
        tag = "end" if at.lower() == "end" else f"end{at}s" if at.startswith("-") else f"{at}s"
        out = a.out or _default_out(a.scene, f"_{tag}.png")
        _emit(dict(ok=True, **still(a.scene, at, out, a.scale)))
        return 0
    if a.cmd == "look":
        from .verify import contact_sheet
        out = a.out or _default_out(a.scene, "_sheet.png")
        _emit(dict(ok=True, **contact_sheet(a.scene, out, a.count, a.cols, a.width, a.at, a.boxes, a.audio,
                                            wire=a.wire, depth=a.depth, trails=a.trails, onion=a.onion)))
        return 0
    if a.cmd == "voice":
        from .voice import generate
        _emit(generate(a.scene, a.only, a.force, a.backend))
        return 0
    if a.cmd == "check":
        from .verify import check
        res = check(a.scene, a.step)
        _emit(res)
        return OK if res["ok"] else ISSUES
    if a.cmd == "probe":
        from .verify import probe, probe_media
        if Path(a.scene).suffix.lower() != ".json" and Path(a.scene).exists():
            _emit(probe_media(a.scene))
            return 0
        if a.curve:
            from .verify import curve
            _emit(curve(a.scene, a.curve))
            return 0
        _emit(probe(a.scene, a.at))
        return 0
    if a.cmd == "watch":
        from .verify import contact_sheet
        p = Path(a.scene)
        if not p.is_file():
            raise ValueError(f"no scene file at {p}")
        last = 0.0
        out = _default_out(a.scene, "_sheet.png")
        print(f"watching {p} -> {out} (Ctrl+C to stop)", file=sys.stderr)
        try:
            while True:
                try:
                    m = p.stat().st_mtime
                except OSError:  # gone for a moment: some editors save by deleting and renaming
                    m = last
                if m != last:
                    last = m
                    try:
                        contact_sheet(a.scene, out, a.count)
                        print(f"{time.strftime('%H:%M:%S')} updated", file=sys.stderr)
                    except PermissionError:  # held open by the editor or a virus scanner: read it again
                        last = 0.0
                    except (ValueError, RuntimeError, OSError) as e:
                        print(f"{time.strftime('%H:%M:%S')} error: {_message(e)}", file=sys.stderr)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("stopped", file=sys.stderr)
            return OK
    return BAD


if __name__ == "__main__":
    raise SystemExit(main())
