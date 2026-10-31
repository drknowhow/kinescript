"""Footage and image access through ffmpeg/ffprobe pipes."""
from __future__ import annotations

import json
import os
import re
import shutil
import queue
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

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
    return {"duration": dur, "w": w, "h": h, "fps": fps, "audio": a, "video": bool(v) and w > 0 and h > 0 and not image,
            "image": image, "codec": codec, "alpha": alpha, "rotation": rot,
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
    """The decoded image, once per file version."""
    return _image(*_version(src))


@lru_cache(maxsize=64)
def _image(src: str, mtime: int, size: int) -> skia.Image:
    data = Path(src).read_bytes() if Path(src).exists() else None
    if data is None:
        raise MediaError(f"image not found: {src}")
    img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(data))
    if img is None:
        raise MediaError(f"cannot decode image {src} (png, jpg, webp, gif, bmp are supported)")
    return img


class VideoSource:
    """Delivers frames of one clip, already resampled to the scene fps and sized to the
    layer box, so a sequential render reads each frame exactly once. Out-of-order access
    (contact sheets, probes) reopens the pipe at the requested time."""

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
        self.held_at: float | None = None  # the time held() opened at, while the pipe still shows that frame
        if self.ctx is not None:  # 4:2:0 planes need even sizes
            self.w, self.h = max(2, self.w - self.w % 2), max(2, self.h - self.h % 2)
            self.yuv_info = skia.YUVAInfo((self.w, self.h), skia.YUVAInfo.kY_U_V, skia.YUVAInfo.k420,
                                          yuv_matrix(self.info))
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0
        self.index = -1
        self.frame: skia.Image | None = None
        self.nbytes = self.w * self.h * 3 // 2 if self.ctx is not None else self.w * self.h * 4

    def _vf(self) -> str:
        w, h = self.w, self.h
        # eof_action=pass: resampled to another fps, the last frame of the file was dropped when the end of the file
        # fell before the next output frame's time, so a clip opened near its end held the frame before its last one.
        timing = f"setpts=(PTS-STARTPTS)/{self.speed},fps={self.fps}:eof_action=pass"
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
        # Seek half a source frame early: -ss drops the frames before it, and 41/30 written as 1.3667 lies past
        # frame 41's own time, so a clip opened there started a frame late. The first frame is now the nearest one.
        # Rounded down, never up: a 60 fps scene asks for times exactly half a 30 fps frame past one, and 0.96667
        # written as 0.9667 dropped that frame too, where a clip played through holds it for that output frame.
        ss = int(max(0.0, self.t0 - 0.5 / max(self.info["fps"] or self.fps, 1)) * 1e4) / 1e4
        dec = ["-c:v", self.decoder] if self.decoder else []
        cmd = [FFMPEG, "-v", "error", *dec, "-ss", f"{ss:.4f}", "-i", self.src, "-an", "-vf", self._vf(),
               "-f", "rawvideo", "-pix_fmt", "yuv420p" if self.ctx is not None else "rgba", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=self.nbytes * 2, creationflags=_NOWIN)
        self.index = -1
        # A reader thread keeps up to PREFETCH decoded frames waiting, so ffmpeg decodes the next
        # frames while this process draws the current one instead of both taking turns.
        self.queue = q = queue.Queue(maxsize=self.PREFETCH)
        self.stopping = stop = threading.Event()
        threading.Thread(target=self._pump, args=(self.proc, q, stop, self.nbytes), daemon=True).start()

    PREFETCH = 6

    @staticmethod
    def _pump(proc, q, stop, nbytes):
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
        except (OSError, ValueError):  # the pipe was closed by close()
            return

    def _next(self) -> bytes | None:
        """The next frame's bytes, None at the end of the stream."""
        buf = self.queue.get()
        if buf is None:
            self.queue.put(None)  # stay at end of stream for later calls
        return buf

    def _read(self) -> bool:
        buf = self._next()
        if buf is None:
            return False
        self._show(buf)
        self.index += 1
        return True

    def _show(self, buf: bytes):
        # No copy: the image (or the planes) wrap the bytes object, which self._buf keeps alive.
        self._buf = buf
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

    def at(self, src_t: float) -> skia.Image | None:
        """Frame showing source time src_t (seconds into the file). Past the last frame, the last frame: what a
        clip played up to that time holds once its picture runs out."""
        self.held_at = None
        dur = self.info["duration"]
        src_t = max(0.0, min(src_t, dur) if dur else src_t)
        if self.end is not None:
            src_t = min(src_t, self.end)
        want = round((src_t - self.t0) / self.speed * self.fps)
        if self.proc is None or want < self.index or want > self.index + int(self.fps * 2):
            self._open(src_t)
            if not self._read():
                self._last(src_t)
            return self.frame
        while self.index < want:
            if not self._read():
                break
        return self.frame

    def held(self, src_t: float) -> skia.Image | None:
        """The frame a clip that has played out holds, src_t being the time of its last frame: always the frame a
        fresh open at src_t shows. A pipe that reached src_t by reading on from an earlier time can land on the
        source frame either side of it when the scene's fps is not the clip's, so a hold read that way could change
        partway through, where one render chunk ends and the next begins. Opened once; later calls reuse it."""
        if self.held_at != src_t or self.proc is None:
            self.close()
            self.at(src_t)
            self.held_at = src_t
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
        if self.proc:
            self.stopping.set()
            try:
                self.proc.stdout.close()  # type: ignore[union-attr]
                self.proc.kill()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
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
