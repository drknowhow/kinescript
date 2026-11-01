"""Footage and image access through ffmpeg/ffprobe pipes."""
from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import shutil
import queue
import subprocess
import threading
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import skia

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
INSTALL_FFMPEG = ("install ffmpeg, which includes ffprobe (winget install Gyan.FFmpeg, brew install ffmpeg or "
                  "sudo apt install ffmpeg), open a new terminal, and run kine doctor to check")


def missing_tools() -> list[str]:
    """Which of ffmpeg and ffprobe cannot be found."""
    return [name for name, path in (("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)) if not shutil.which(path)]


def explain(e: BaseException) -> str:
    """The message for an error: when a program could not be started and ffmpeg or ffprobe is not on the PATH, which
    one and how to install it. Starting a program that is not there said only "[WinError 2] The system cannot find
    the file specified", without naming it."""
    if isinstance(e, FileNotFoundError) and e.filename in (None, FFMPEG, FFPROBE):
        gone = missing_tools()
        if gone:
            return (f"{' and '.join(gone)} {'is' if len(gone) == 1 else 'are'} not on the PATH, and Kinescript runs "
                    f"{'it' if len(gone) == 1 else 'them'} to read and write media: {INSTALL_FFMPEG}")
    return str(e)


class MediaError(RuntimeError):
    pass


def _version(src: str) -> tuple:
    """(path, modification time, size): a cache key that changes when the file does, so the MCP server and
    warm render workers read an edited file again."""
    try:
        st = os.stat(src)
    except OSError:
        return src, 0, -1
    return src, st.st_mtime_ns, st.st_size


def _ffmpeg_says(*args) -> str:
    return subprocess.run([FFMPEG, "-hide_banner", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", creationflags=_NOWIN).stdout


@lru_cache(maxsize=1)
def ffmpeg_version() -> int:
    """The major version of the ffmpeg in use, 0 when unknown."""
    m = re.search(r"version n?(\d+)\.", _ffmpeg_says("-version"))
    return int(m.group(1)) if m else 0


@lru_cache(maxsize=1)
def capabilities() -> dict:
    """{'version': major version (0 when unknown), 'encoders': set of names} of the ffmpeg in use."""
    encoders = {p[1] for p in (line.split() for line in _ffmpeg_says("-encoders").splitlines()) if len(p) > 1}
    return {"version": ffmpeg_version(), "encoders": encoders}


def filter_script(path: str) -> list[str]:
    """ffmpeg arguments that read a filter graph from a file, which has no command-line length limit.
    ffmpeg 7 renamed -filter_complex_script to -/filter_complex; the old name still works there.
    Only the version is asked for: a one-worker render waits for it before its first frame."""
    return ["-/filter_complex" if ffmpeg_version() >= 7 else "-filter_complex_script", path]


def probe(src: str) -> dict:
    """{'duration', 'w', 'h', 'fps', 'audio', 'video', 'image', 'codec', 'alpha', 'rotation', 'pix_fmt',
    'color_space'} for a media file, probed once per file version. w and h are the size the picture is shown at
    (after its rotation); duration is the length from its first frame or sample."""
    return _probe(*_version(src))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# Pixel formats with an alpha channel. VP8/VP9 keep theirs in a side stream that the stream's alpha_mode tag
# announces, while ffprobe still says yuv420p.
_ALPHA_FMTS = ("yuva", "rgba", "bgra", "argb", "abgr", "gbrap", "ya8", "ya16", "rgba64", "bgra64")
# The decoders that read that side stream; ffmpeg's own vp8/vp9 decoders drop it.
ALPHA_DECODERS = {"vp9": "libvpx-vp9", "vp8": "libvpx"}


@lru_cache(maxsize=128)
def _probe(src: str, mtime: int, size: int) -> dict:
    if not Path(src).exists():
        raise MediaError(f"media not found: {src}")
    if Path(src).is_dir():
        raise MediaError(f"{src} is a folder, not a media file: give the path of a file in it")
    # ffprobe writes tags and paths as UTF-8; decoded as cp1252 (the Windows default) a Cyrillic or Japanese
    # title tag raised UnicodeDecodeError.
    out = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", src],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=_NOWIN)
    if out.returncode:
        raise MediaError(f"ffprobe failed on {src}: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout)
    fmt = info.get("format", {})
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    dur = _num(fmt.get("duration")) or _num((v or {}).get("duration"))
    start, name = _num(fmt.get("start_time")), str(fmt.get("format_name", ""))
    if start > 0 and name.split(",")[0] in ("matroska", "nut"):
        # These report where the file ends, not how long it is: footage that starts at 5 s and runs 7 s gave
        # 12 s, and its layer held the last frame for 5 s and then went black.
        dur = max(0.0, dur - start)
    fps = 0.0
    if v and v.get("avg_frame_rate", "0/0") != "0/0":
        n, d = v["avg_frame_rate"].split("/")
        fps = _num(n) / (_num(d) or 1)
    # A still: an image demuxer (png_pipe, jpeg_pipe, ...) or a one-frame GIF. ffprobe gives these 25 fps.
    image = bool(v) and (name.endswith("_pipe") or name == "image2"
                         or (name == "gif" and int(_num(v.get("nb_frames"))) <= 1))
    if image:
        fps, dur = 0.0, 0.0
    w, h = int((v or {}).get("width", 0)), int((v or {}).get("height", 0))
    # A phone clip shot upright is stored sideways with a rotation; ffmpeg turns it upright when decoding.
    rot = 0.0
    for sd in (v or {}).get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = _num(sd["rotation"])
    tags = {str(k).lower(): str(t) for k, t in ((v or {}).get("tags") or {}).items()}
    rot = int(round(rot or _num(tags.get("rotate")))) % 360
    if rot in (90, 270):
        w, h = h, w
    pf, codec = (v or {}).get("pix_fmt", ""), (v or {}).get("codec_name", "")
    alpha = bool(v) and (pf.startswith(_ALPHA_FMTS) or (tags.get("alpha_mode") == "1" and codec in ALPHA_DECODERS))
    # video: moving pictures to decode. A TS cut with -c copy can declare a video stream and hold no frame of it.
    # start: the file's own time of its first frame or sample, which source seconds count from.
    return {"duration": dur, "w": w, "h": h, "fps": fps, "audio": a, "video": bool(v) and w > 0 and h > 0 and not image,
            "image": image, "codec": codec, "alpha": alpha, "rotation": rot, "start": start,
            "pix_fmt": pf, "color_space": (v or {}).get("color_space", "")}


@lru_cache(maxsize=1)
def decoders() -> set:
    """The names of the video decoders the ffmpeg in use has."""
    return {p[1] for p in (line.split() for line in _ffmpeg_says("-decoders").splitlines())
            if len(p) > 1 and p[0].startswith("V")}


def alpha_decoder(info: dict) -> str | None:
    """The decoder that keeps a VP8/VP9 clip's alpha, when it has one and this ffmpeg has that decoder."""
    name = ALPHA_DECODERS.get(info.get("codec", "")) if info.get("alpha") else None
    return name if name and name in decoders() else None


def decoded_end(src: str) -> float | None:
    """Where the picture of a clip really ends (seconds from its start); None when it has no picture. A file cut
    short or damaged ends before its probed duration."""
    return _decoded_end(*_version(src))


def _ends(src: str, what: str, args: list) -> tuple[list[float], float]:
    """(the end time of each packet or frame (`what`) of the first video stream that ffprobe reads with `args`,
    the file's start time)."""
    fields = "pts_time,dts_time,duration_time" if what == "packet" else "best_effort_timestamp_time,duration_time"
    out = subprocess.run([FFPROBE, "-v", "quiet", *args, "-select_streams", "v:0", "-show_entries",
                          f"{what}={fields}:format=start_time", "-of", "csv", src],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=_NOWIN)
    ends, start = [], 0.0
    for line in out.stdout.splitlines():
        head, *vals = line.split(",")
        p = [_num(x) if x.strip() not in ("", "N/A") else None for x in vals]
        if head == "format":
            start = p[0] or 0.0 if p else 0.0
        elif head == what and p:
            t = next((x for x in p[:-1] if x is not None), None)
            if t is not None:
                ends.append(t + (p[-1] or 0.0))
    return ends, start


@lru_cache(maxsize=64)
def _decoded_end(src: str, mtime: int, size: int) -> float | None:
    info = _probe(src, mtime, size)
    if not info["video"]:
        return None
    dur = info["duration"]
    # The packets of its last second first, read without decoding them (a few hundredths of a second): a file
    # that was cut short stops giving packets where its data stops.
    start = _num(subprocess.run([FFPROBE, "-v", "quiet", "-show_entries", "format=start_time", "-of", "csv=p=0", src],
                                capture_output=True, text=True, creationflags=_NOWIN).stdout.strip())
    tail = ["-read_intervals", f"{start + max(0.0, dur - 1.0):.3f}%"]  # -read_intervals counts from 0, not start
    ends, start = _ends(src, "packet", tail)
    if ends and max(ends) - start >= dur - 0.25:
        return max(ends) - start
    # Short: decode on from the key frame before its last second, to the last frame that decodes.
    ends, start = _ends(src, "frame", tail)
    if not ends:  # it stops well before that: find its last key frame, and decode on from there
        keys, start = _ends(src, "frame", ["-skip_frame", "nokey"])
        ends = (_ends(src, "frame", ["-read_intervals", f"{max(keys) - 0.001:.3f}%"])[0] or keys) if keys else []
    return max(0.0, max(ends) - start) if ends else 0.0


def yuv_matrix(info: dict):
    """Skia's YUV colour space for a probed clip: its tag, or, untagged, the rule players use
    (BT.709 from 1280 wide or above 576 lines, else BT.601, both counted as stored, before any rotation).
    All limited range."""
    cs = info.get("color_space") or ""
    if cs == "bt709":
        return skia.kRec709_YUVColorSpace
    if cs in ("smpte170m", "bt470bg", "fcc"):
        return skia.kRec601_YUVColorSpace
    if cs.startswith("bt2020"):
        return skia.kBT2020_YUVColorSpace
    w, h = info.get("w", 0), info.get("h", 0)
    if info.get("rotation") in (90, 270):
        w, h = h, w
    return skia.kRec709_YUVColorSpace if w >= 1280 or h > 576 else skia.kRec601_YUVColorSpace


def image(src: str) -> skia.Image:
    """The decoded image, once per file version. An svg comes as pixels: drawn at its own size times the whole number
    that takes its longer side to 512 px or more (at most 4096), for a model's texture or a particle's sprite."""
    return _image(*_version(src))


@lru_cache(maxsize=64)
def _image(src: str, mtime: int, size: int) -> skia.Image:
    if src.lower().endswith(".svg"):
        v = vector(src)
        longest = max(v.w, v.h)
        k = min(max(1, math.ceil(512 / longest)), 4096 / longest)
        surf = skia.Surface.MakeRaster(skia.ImageInfo.MakeN32Premul(max(1, round(v.w * k)), max(1, round(v.h * k))))
        c = surf.getCanvas()
        c.scale(surf.width() / v.w, surf.height() / v.h)
        c.drawPicture(v.picture)
        return surf.makeImageSnapshot()
    data = Path(src).read_bytes() if Path(src).exists() else None
    if data is None:
        raise MediaError(f"image not found: {src}")
    img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(data))
    if img is None:
        raise MediaError(f"cannot decode image {src} (png, jpg, webp, gif, bmp and svg are supported)")
    return img


def still(src: str) -> skia.Image | Vector:
    """What an image layer draws from src: its vector drawing for an svg, else the decoded image."""
    return vector(src) if src.lower().endswith(".svg") else image(src)


class Vector:
    """An svg file as a vector drawing: its picture, drawn at the file's own size w x h, and `undrawn`, the kinds of
    element in it that skia's svg reader leaves out, in SVG_FIX's order."""
    __slots__ = ("picture", "w", "h", "undrawn")

    def __init__(self, picture: skia.Picture, w: float, h: float, undrawn: tuple):
        self.picture, self.w, self.h, self.undrawn = picture, w, h, undrawn

    def width(self) -> float:
        return self.w

    def height(self) -> float:
        return self.h


# Elements skia's svg reader does not draw, by kind; each kind's fix is in SVG_FIX. The reader draws shapes, paths,
# strokes, gradients, patterns, clip paths, masks, opacity, transforms, <use> of a shape and most filters.
SVG_UNDRAWN = {"text": "text", "tspan": "text", "textPath": "text", "style": "style", "symbol": "symbol",
               "image": "image", "marker": "marker", "switch": "switch", "foreignObject": "foreignObject",
               "feDropShadow": "feDropShadow", "animate": "animation", "animateTransform": "animation",
               "animateMotion": "animation", "set": "animation"}
SVG_FIX = {"text": "turn text into paths (Inkscape: Path > Object to Path; Illustrator: Type > Create Outlines)",
           "style": "write fill and stroke on the elements instead of in a <style> sheet (Illustrator: Styling: "
                    "Presentation Attributes)",
           "symbol": "put the shapes of each <symbol> where it is used",
           "image": "show a picture in an image layer of its own",
           "marker": "turn the stroke and its markers into paths (Inkscape: Path > Stroke to Path)",
           "switch": "keep the one branch of the <switch> you want",
           "foreignObject": "draw it with svg shapes",
           "feDropShadow": "drop the filter and give the layer a shadow",
           "animation": "an svg's own animation does not play: animate the layer instead"}
_LENGTH = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(px|pt|pc|in|cm|mm|em|ex|%)?\s*", re.I)
_PX = {"px": 1, "pt": 4 / 3, "pc": 16, "in": 96, "cm": 96 / 2.54, "mm": 96 / 25.4, "em": 16, "ex": 8}
_START_TAG = re.compile(rb"<[A-Za-z][^>]*>")
_HREF = re.compile(rb"(\s)href(\s*=)")


def _svg_length(v) -> float | None:
    """An svg width or height in px (96 to the inch), or None for one missing, a percentage or not above 0."""
    m = _LENGTH.fullmatch(v or "")
    if m is None or m.group(2) == "%":
        return None
    x = float(m.group(1)) * _PX[(m.group(2) or "px").lower()]
    return x if 0 < x < math.inf else None


def _svg_size(root: ET.Element) -> tuple[float, float]:
    """The size an svg is drawn at, as a browser shows it in an <img>: its width and height; the viewBox's shape for
    a missing one; the viewBox's size for both; 300 x 150 for what is still missing."""
    w, h = _svg_length(root.get("width")), _svg_length(root.get("height"))
    try:
        vb = [float(x) for x in re.split(r"[\s,]+", (root.get("viewBox") or "").strip())]
    except ValueError:
        vb = []
    if not (w and h) and len(vb) == 4 and vb[2] > 0 and vb[3] > 0:
        aspect = vb[2] / vb[3]
        return (w, w / aspect) if w else (h * aspect, h) if h else (vb[2], vb[3])
    return w or 300.0, h or 150.0


def _xlink(tag: re.Match) -> bytes:
    t = tag.group(0)
    return t if b"xlink:href" in t else _HREF.sub(rb"\1xlink:href\2", t, count=1)


def vector(src: str) -> Vector:
    """The svg file src as a vector drawing, once per file version."""
    return _vector(*_version(src))


@lru_cache(maxsize=64)
def _vector(src: str, mtime: int, size: int) -> Vector:
    data = Path(src).read_bytes() if Path(src).exists() else None
    if data is None:
        raise MediaError(f"svg not found: {src}")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise MediaError(f"cannot read svg {src}: {e}") from None
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise MediaError(f"cannot read svg {src}: it is <{root.tag.rsplit('}', 1)[-1]}>, not <svg>")
    found = {SVG_UNDRAWN.get(el.tag.rsplit("}", 1)[-1]) for el in root.iter() if isinstance(el.tag, str)}
    undrawn = tuple(k for k in SVG_FIX if k in found)
    w, h = _svg_size(root)
    # The reader follows xlink:href only; SVG 2 writes href (a <use>, a gradient taking another's stops).
    dom = skia.SVGDOM.MakeFromStream(skia.MemoryStream(skia.Data.MakeWithCopy(_START_TAG.sub(_xlink, data))))
    if dom is None:
        raise MediaError(f"cannot read svg {src}")
    dom.setContainerSize(skia.Size(w, h))
    rec = skia.PictureRecorder()
    c = rec.beginRecording(skia.Rect.MakeWH(w, h))
    c.clipRect(skia.Rect.MakeWH(w, h))  # what lies outside its viewport is not shown, as in a browser
    dom.render(c)
    return Vector(rec.finishRecordingAsPicture(), w, h, undrawn)


FRAME_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".svg")  # the files a folder of frames is read for
SEQUENCE_FPS = 12  # frames a second of a sequence or sheet without fps: hand-drawn animation's usual rate
_NUMBER = re.compile(r"%0?\d*d|#+")  # the number in a pattern: run_%03d.png (as ffmpeg writes it) or run_####.png


class Frames(NamedTuple):
    """An image layer's frame-by-frame animation: `files` in order, a missing number holding the file before it, or
    None for a sheet, whose `count` frames are its cells of `cols` x `rows`, row by row from the top left."""
    files: list | None
    cols: int
    rows: int
    count: int

    def at(self, t: float, fps: float, loop) -> int:
        """The frame shown at the layer's own time t: frame k from k / fps seconds. loop True starts again after the
        last, "pingpong" plays back to the first and on, and False holds the last."""
        k, n = math.floor(t * fps + 1e-6), self.count
        if loop == "pingpong" and n > 1:
            k %= 2 * n - 2
            return k if k < n else 2 * n - 2 - k
        return k % n if loop else min(max(k, 0), n - 1)


def _natural(name: str) -> list:
    """A sort key that puts run_2 before run_10."""
    return [(0, int(s), "") if s.isdigit() else (1, 0, s.lower()) for s in re.split(r"(\d+)", name) if s]


def sequence(spec: str) -> tuple[list[str], list[int]]:
    """The files of an image sequence and the numbers missing from it. `spec` is a pattern with the number in it
    (run_%03d.png or run_####.png), a glob (run_*.png) or a folder of images. When the names differ only in their
    number, frame k is the lowest number plus k, and a missing number shows the file before it, so the others keep
    their times; otherwise the files go in name order, numbers counted as numbers."""
    p = Path(spec)
    parts = _NUMBER.split(p.name)
    if p.is_dir():
        found = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in FRAME_EXTS]
    elif len(parts) > 1:  # the number is the first %d or # run; the rest must match as written
        rx = re.compile(re.escape(parts[0]) + r"(\d+)" + "".join(map(re.escape, parts[1:])), re.I)
        found = [x for x in p.parent.iterdir() if x.is_file() and rx.fullmatch(x.name)] if p.parent.is_dir() else []
    elif any(ch in p.name for ch in "*?["):
        found = [x for x in p.parent.iterdir() if x.is_file() and fnmatch.fnmatch(x.name.lower(), p.name.lower())
                 ] if p.parent.is_dir() else []
    else:
        return ([str(p)] if p.is_file() else []), []
    if len(parts) > 1 and not p.is_dir():
        nums = {x: int(rx.fullmatch(x.name).group(1)) for x in found}
    else:  # numbered when the names differ only in their last number: run_1.png, run_2.png
        shapes = {re.sub(r"\d+(\D*)$", r"#\1", x.name.lower()) for x in found}
        nums = ({x: int(re.findall(r"\d+", x.name)[-1]) for x in found}
                if len(shapes) == 1 and "#" in next(iter(shapes)) else {})
    if found and len(nums) == len(found) and len(set(nums.values())) == len(found):
        by = {n: str(x) for x, n in nums.items()}
        files, missing, last = [], [], by[min(by)]
        for n in range(min(by), max(by) + 1):
            if n in by:
                last = by[n]
            else:
                missing.append(n)
            files.append(last)
        return files, missing
    return [str(x) for x in sorted(found, key=lambda x: _natural(x.name))], []


class VideoSource:
    """Delivers frames of one clip, already resampled to the scene fps and sized to the
    layer box, so a sequential render reads each frame exactly once. Out-of-order access
    (contact sheets, probes) reopens the pipe at the requested time.
    A source time shows the source frame nearest it, the earlier one when two are equally near, whether the pipe
    was opened there or read on to it: a frame is a function of its time, not of where the render started."""

    def __init__(self, src: str, w: int, h: int, fps: float, speed: float = 1.0, fit: str = "cover", ctx=None):
        """ctx: a GPU context. With one, YUV clips without alpha arrive as yuv420p (1.5 bytes a pixel
        instead of 4; ffmpeg converts nothing) and become RGB on the GPU at draw time."""
        self.src, self.w, self.h, self.fps, self.speed, self.fit = src, max(2, w), max(2, h), fps, speed, fit
        self.info = probe(src)
        pf = self.info["pix_fmt"]
        self.ctx = ctx if (ctx is not None and fit != "contain" and pf.startswith(("yuv", "nv", "p0"))
                           and not self.info["alpha"]) else None
        # A VP8/VP9 clip with alpha (Kinescript's own transparent .webm, for one) is read by libvpx, which keeps it.
        self.decoder = alpha_decoder(self.info)
        self.end: float | None = None  # the time of the last frame, once a read has run out of picture before it
        # A frame up to this far after a time is the one shown there: the nearest, the earlier of two equally near.
        # The microsecond keeps an exact tie on the same side however the decoder's arithmetic rounds it.
        self.near = 0.5 / (self.info["fps"] or fps) - 1e-6
        if self.ctx is not None:  # 4:2:0 planes need even sizes
            self.w, self.h = max(2, self.w - self.w % 2), max(2, self.h - self.h % 2)
            self.yuv_info = skia.YUVAInfo((self.w, self.h), skia.YUVAInfo.kY_U_V, skia.YUVAInfo.k420,
                                          yuv_matrix(self.info))
        # While the pipe is open, a list that holds the ffmpeg process once the reader thread has started it.
        self.proc: list[subprocess.Popen] | None = None
        self.t0 = 0.0
        self.index = -1
        self.frame: skia.Image | None = None
        self.nbytes = self.w * self.h * 3 // 2 if self.ctx is not None else self.w * self.h * 4

    def _vf(self) -> str:
        w, h = self.w, self.h
        # Output frame k is source time t0 + k * speed / fps and shows the last frame at or before that time plus
        # self.near, from any t0. Times are the file's own (-copyts), in nanoseconds, measured from t0, and fps
        # rounds up. Before, the grid started at the first frame decoded (PTS-STARTPTS), up to half a source frame
        # from t0, and fps rounded to the nearest output frame: a 30 fps clip at 24 fps showed frame 8 at 0.25 s
        # played from 0 and frame 7 opened there. eof_action=pass keeps a last frame that ends between output frames.
        # float(): a numpy time (a frame number from np.linspace) printed as np.float64(...), which ffmpeg cannot
        # parse; the pipe gave no frame, and the clip showed its last frame, found by decoding the file to its end.
        off = float((self.info["start"] + self.t0 + self.near) * 1e9)
        timing = (f"settb=1/1000000000,setpts=(PTS-({off!r}))/{self.speed!r},"
                  f"fps={self.fps}:start_time=0:round=up:eof_action=pass")
        if self.ctx is not None:  # cover or stretch, kept in YUV and limited range
            fit = ":force_original_aspect_ratio=increase" if self.fit == "cover" else ""
            crop = f",crop={w}:{h}" if self.fit == "cover" else ""
            return f"{timing},scale={w}:{h}{fit}:out_range=tv{crop},format=yuv420p"
        # The same matrix the YUV path uses, so GPU and raster renders of a clip agree.
        m = {skia.kRec709_YUVColorSpace: "bt709", skia.kRec601_YUVColorSpace: "bt601",
             skia.kBT2020_YUVColorSpace: "bt2020"}[yuv_matrix(self.info)]
        cm = f":in_color_matrix={m}" if self.info.get("pix_fmt", "").startswith(("yuv", "nv", "p0")) else ""
        if self.fit == "contain":
            size = (f"scale={w}:{h}{cm}:force_original_aspect_ratio=decrease,format=rgba,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x00000000")
        elif self.fit == "stretch":
            size = f"scale={w}:{h}{cm},format=rgba"
        else:
            size = f"scale={w}:{h}{cm}:force_original_aspect_ratio=increase,crop={w}:{h},format=rgba"
        return f"{timing},{size}"

    def _open(self, t: float):
        self.close()
        self.t0 = max(0.0, t)
        # -ss drops the frames before it, and the frame t0 shows can lie half a source frame before t0: seek a frame
        # and a half early, which also leaves room for timestamps that wander (phone clips).
        ss = max(0.0, self.t0 - 1.5 / max(self.info["fps"] or self.fps, 1))
        dec = ["-c:v", self.decoder] if self.decoder else []
        cmd = [FFMPEG, "-v", "error", *dec, "-copyts", "-ss", f"{ss:.6f}", "-i", self.src, "-an", "-vf", self._vf(),
               "-f", "rawvideo", "-pix_fmt", "yuv420p" if self.ctx is not None else "rgba", "-"]
        self.index = -1
        # A reader thread starts ffmpeg and keeps up to PREFETCH decoded frames waiting, so ffmpeg decodes the next
        # frames while this process draws the current one instead of both taking turns.
        self.queue = q = queue.Queue(maxsize=self.PREFETCH)
        self.stopping, self.reading = stop, reading = threading.Event(), threading.Event()
        self.proc = started = []
        threading.Thread(target=self._pump, args=(cmd, self.nbytes, q, stop, reading, started), daemon=True).start()

    PREFETCH = 6

    @staticmethod
    def _pump(cmd, nbytes, q, stop, reading, started):
        # Started and ended here, not in _open() and close(): on Windows ffmpeg takes ~6 ms to start and ~9 ms to go
        # once killed, which held up the frame that opened or closed a clip.
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 2,
                                    creationflags=_NOWIN)
        except OSError as e:  # raised where the frame is read (_next)
            q.put(e)
            return
        started.append(proc)
        try:
            while not stop.is_set():
                buf = proc.stdout.read(nbytes)
                item = buf if len(buf) == nbytes else None
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if item is None:
                    return
                # A pipe opened before its clip shows (prime) holds one frame until the clip reads it, not PREFETCH:
                # cut lists open a few ahead, each up to 50 MB of 1080p RGBA.
                while not (stop.is_set() or reading.wait(0.1)):
                    pass
        except OSError:  # killed by close()
            return
        finally:
            proc.kill()
            proc.stdout.close()  # type: ignore[union-attr]
            proc.wait()

    def _next(self) -> bytes | None:
        """The next frame's bytes, None at the end of the stream."""
        self.reading.set()
        buf = self.queue.get()
        if buf is None or isinstance(buf, OSError):
            self.queue.put(buf)  # stay at end of stream for later calls
            if buf is not None:
                raise buf
        return buf

    def _read(self) -> bool:
        buf = self._next()
        if buf is None:
            return False
        self._show(buf)
        self.index += 1
        return True

    def _show(self, buf: bytes):
        # The image is made when at() returns it: a frame read past on the way to a later one (a contact sheet,
        # a render that skips frames) cost an upload of its own.
        self._buf, self.frame = buf, None

    def _image(self):
        buf = self._buf
        if self.frame is not None or buf is None:
            return
        # No copy: the image (or the planes) wrap the bytes object, which self._buf keeps alive.
        if self.ctx is not None:
            w, h, b = self.w, self.h, np.frombuffer(buf, np.uint8)
            planes = (b[:w * h].reshape(h, w), b[w * h:w * h * 5 // 4].reshape(h // 2, w // 2),
                      b[w * h * 5 // 4:].reshape(h // 2, w // 2))
            pms = [skia.Pixmap(skia.ImageInfo.Make(a.shape[1], a.shape[0], skia.kGray_8_ColorType,
                                                   skia.kOpaque_AlphaType), a, a.shape[1]) for a in planes]
            self.frame = skia.Image.MakeFromYUVAPixmaps(
                self.ctx, skia.YUVAPixmaps.FromExternalPixmaps(self.yuv_info, pms + [skia.Pixmap()]))
        else:
            arr = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)
            self.frame = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType, copy=False)

    def _when(self, src_t: float) -> float:
        """The source time at() shows for src_t: inside the file, and not past a last frame already found."""
        dur = self.info["duration"]
        src_t = max(0.0, min(src_t, dur) if dur else src_t)
        return src_t if self.end is None else min(src_t, self.end)

    def _opens(self, src_t: float) -> bool:
        """Whether showing src_t (from _when) needs the pipe opened there."""
        k = (src_t - self.t0) / self.speed * self.fps
        want = round(k)
        # A time between the pipe's output frames (a held last frame, a loop starting over) opens at that time:
        # read on to the output frame next to it, it would show that frame's picture instead of its own.
        return (self.proc is None or want < self.index or want > self.index + int(self.fps * 2)
                or abs(k - want) > 1e-6)

    def prime(self, src_t: float):
        """Open the pipe where at(src_t) would, without waiting for a frame. ffmpeg takes about 0.1 s to give a
        clip's first frame; a decoder opened early starts while other work goes on (Renderer._prime)."""
        src_t = self._when(src_t)
        if self._opens(src_t):
            self._open(src_t)

    def at(self, src_t: float) -> skia.Image | None:
        """Frame showing source time src_t (seconds into the file). Past the last frame, the last frame: what a
        clip played up to that time holds once its picture runs out."""
        self.prime(src_t)
        src_t = self._when(src_t)
        want = round((src_t - self.t0) / self.speed * self.fps)
        while self.index < want:
            if not self._read():
                if self.index < 0:  # opened at src_t, here or by prime(), and there is no picture there
                    self._last(src_t)
                break
        self._image()
        return self.frame

    def _last(self, t: float):
        """Opened at t, past the last frame (the file runs on in sound only, or was cut short): find the last frame
        before t by reading on from further and further back. Only the last one becomes an image. Later times past
        it then map to it, so they do not search again."""
        for back in (1.0, 4.0, 16.0, t):
            start = max(0.0, t - back)
            self._open(start)
            last, n = None, 0
            while (buf := self._next()) is not None:
                last, n = buf, n + 1
            if last is not None:
                self._show(last)
                self.index = n - 1
                self.end = start + self.index * self.speed / self.fps
                return
            if start == 0.0:
                return

    def close(self):
        if self.proc is not None:
            self.stopping.set()  # the reader stops after its next read, which the kill ends at once
            for p in self.proc:
                try:
                    p.kill()
                except OSError:
                    pass
            self.proc = None
            # The prefetched frames and the last one shown are 3-8 MB each at 1080p; at() reopens and refills.
            self.queue = self.frame = self._buf = None


@lru_cache(maxsize=4)
def grain_tiles(seed: int = 7, size: int = 256, count: int = 6) -> list[skia.Image]:
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(count):
        g = rng.integers(0, 256, (size, size), dtype=np.uint8)
        arr = np.dstack([g, g, g, np.full_like(g, 255)])
        tiles.append(skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType))
    return tiles
