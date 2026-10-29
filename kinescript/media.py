"""Footage and image access through ffmpeg/ffprobe pipes."""
from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import skia

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class MediaError(RuntimeError):
    pass


@lru_cache(maxsize=128)
def probe(src: str) -> dict:
    """{'duration', 'w', 'h', 'fps', 'audio'} for a media file."""
    if not Path(src).exists():
        raise MediaError(f"media not found: {src}")
    out = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", src],
                         capture_output=True, text=True, creationflags=_NOWIN)
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
            "fps": fps, "audio": a, "image": bool(v) and fps == 0 and dur == 0}


@lru_cache(maxsize=64)
def image(src: str) -> skia.Image:
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

    def __init__(self, src: str, w: int, h: int, fps: float, speed: float = 1.0, fit: str = "cover"):
        self.src, self.w, self.h, self.fps, self.speed, self.fit = src, max(2, w), max(2, h), fps, speed, fit
        self.info = probe(src)
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0
        self.index = -1
        self.frame: skia.Image | None = None
        self.nbytes = self.w * self.h * 4

    def _vf(self) -> str:
        w, h = self.w, self.h
        if self.fit == "contain":
            size = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,format=rgba,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x00000000")
        elif self.fit == "stretch":
            size = f"scale={w}:{h},format=rgba"
        else:
            size = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=rgba"
        return f"setpts=(PTS-STARTPTS)/{self.speed},fps={self.fps},{size}"

    def _open(self, t: float):
        self.close()
        self.t0 = max(0.0, t)
        cmd = [FFMPEG, "-v", "error", "-ss", f"{self.t0:.4f}", "-i", self.src, "-an", "-vf", self._vf(),
               "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=self.nbytes * 2, creationflags=_NOWIN)
        self.index = -1

    def _read(self) -> bool:
        assert self.proc and self.proc.stdout
        buf = self.proc.stdout.read(self.nbytes)
        if len(buf) < self.nbytes:
            return False
        arr = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)
        self.frame = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType)
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
            try:
                self.proc.stdout.close()  # type: ignore[union-attr]
                self.proc.kill()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.proc = None


@lru_cache(maxsize=4)
def grain_tiles(seed: int = 7, size: int = 256, count: int = 6) -> list[skia.Image]:
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(count):
        g = rng.integers(0, 256, (size, size), dtype=np.uint8)
        arr = np.dstack([g, g, g, np.full_like(g, 255)])
        tiles.append(skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType))
    return tiles
