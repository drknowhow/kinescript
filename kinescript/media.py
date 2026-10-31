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
    """{'duration', 'w', 'h', 'fps', 'audio'} for a media file, probed once per file version."""
    return _probe(*_version(src))


@lru_cache(maxsize=128)
def _probe(src: str, mtime: int, size: int) -> dict:
    if not Path(src).exists():
        raise MediaError(f"media not found: {src}")
    # ffprobe writes tags and paths as UTF-8; decoded as cp1252 (the Windows default) a Cyrillic or Japanese
    # title tag raised UnicodeDecodeError.
    out = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", src],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=_NOWIN)
    if out.returncode:
        raise MediaError(f"ffprobe failed on {src}: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    dur = float(info.get("format", {}).get("duration") or (v or {}).get("duration") or 0)
    fps = 0.0
    if v and v.get("avg_frame_rate", "0/0") != "0/0":
        n, d = v["avg_frame_rate"].split("/")
        fps = float(n) / float(d or 1)
    return {"duration": dur, "w": int((v or {}).get("width", 0)), "h": int((v or {}).get("height", 0)),
            "fps": fps, "audio": a, "image": bool(v) and fps == 0 and dur == 0,
            "pix_fmt": (v or {}).get("pix_fmt", ""), "color_space": (v or {}).get("color_space", "")}


def yuv_matrix(info: dict):
    """Skia's YUV colour space for a probed clip: its tag, or, untagged, the rule players use
    (BT.709 from 1280 wide or above 576 lines, else BT.601). All limited range."""
    cs = info.get("color_space") or ""
    if cs == "bt709":
        return skia.kRec709_YUVColorSpace
    if cs in ("smpte170m", "bt470bg", "fcc"):
        return skia.kRec601_YUVColorSpace
    if cs.startswith("bt2020"):
        return skia.kBT2020_YUVColorSpace
    return skia.kRec709_YUVColorSpace if info.get("w", 0) >= 1280 or info.get("h", 0) > 576 else skia.kRec601_YUVColorSpace


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
                           and not pf.startswith("yuva")) else None
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
        if self.ctx is not None:  # cover or stretch, kept in YUV and limited range
            fit = ":force_original_aspect_ratio=increase" if self.fit == "cover" else ""
            crop = f",crop={w}:{h}" if self.fit == "cover" else ""
            return f"setpts=(PTS-STARTPTS)/{self.speed},fps={self.fps},scale={w}:{h}{fit}:out_range=tv{crop},format=yuv420p"
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
        return f"setpts=(PTS-STARTPTS)/{self.speed},fps={self.fps},{size}"

    def _open(self, t: float):
        self.close()
        self.t0 = max(0.0, t)
        # Seek half a source frame early: -ss drops the frames before it, and 41/30 written as 1.3667 lies past
        # frame 41's own time, so a clip opened there started a frame late. The first frame is now the nearest one.
        ss = max(0.0, self.t0 - 0.5 / max(self.info["fps"] or self.fps, 1))
        cmd = [FFMPEG, "-v", "error", "-ss", f"{ss:.4f}", "-i", self.src, "-an", "-vf", self._vf(),
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

    def _read(self) -> bool:
        buf = self.queue.get()
        if buf is None:
            self.queue.put(None)  # stay at end of stream for later calls
            return False
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
        self.index += 1
        return True

    def at(self, src_t: float) -> skia.Image | None:
        """Frame showing source time src_t (seconds into the file)."""
        dur = self.info["duration"]
        if dur:
            src_t = min(src_t, max(0.0, dur - 1.5 / max(self.info["fps"] or self.fps, 1)))
        want = round((src_t - self.t0) / self.speed * self.fps)
        if self.proc is None or want < self.index or want > self.index + int(self.fps * 2):
            self._open(src_t)
            want = 0
        while self.index < want:
            if not self._read():
                break
        return self.frame

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
