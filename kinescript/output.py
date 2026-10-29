"""Encoding: frames piped to ffmpeg, split across worker processes, audio mixed at the end."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import media
from .render import Renderer
from .scene import Scene, load

QUALITY = {  # crf, x264 preset
    "draft": (30, "ultrafast"),
    "standard": (20, "veryfast"),
    "high": (16, "medium"),
}


def _video_args(fmt: str, quality: str, fps: float, threads: int = 0) -> list[str]:
    crf, preset = QUALITY.get(quality, QUALITY["standard"])
    # Each worker runs its own encoder; left to auto-threading, eight of them oversubscribe the CPU.
    thr = ["-threads", str(threads)] if threads else []
    if fmt == "mov":  # ProRes 4444 with alpha, for compositing in an NLE
        return ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0", *thr]
    if fmt == "webm":
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", str(crf + 10),
                "-row-mt", "1", "-deadline", "realtime" if quality == "draft" else "good", "-cpu-used", "4", *thr]
    if fmt == "gif":  # intermediate for the palette pass; a near-lossless one keeps grain and doubles the GIF
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", *thr]
    # x264's own GOP defaults (keyint 250, 3 B-frames, scene-cut keyframes): same PSNR as -g 2s -bf 2
    # in 10-12% fewer bytes on the examples.
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", *thr]


# HD video is BT.709. Untagged files are read as 709 by browsers and players, but ffmpeg's default
# RGB->YUV matrix is BT.601, which shifts saturated colours; so convert with 709 and say so.
# The tags go on the frames (setparams): ffmpeg 8 drops -color_primaries / -color_trc output options
# when the frames themselves say "unknown".
TAG_709 = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
TO_709 = "scale=out_color_matrix=bt709:out_range=tv"


def _encoder(path: str, w: int, h: int, fps: float, fmt: str, quality: str, threads: int = 0, yuv: bool = False):
    """yuv: frames arrive as BT.709 I420 (Renderer.frame_i420) and pass to the encoder unconverted."""
    pix, vf = ("yuv420p", TAG_709) if yuv else ("rgba", f"{TO_709},{TAG_709}")
    cmd = [media.FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-", "-vf", vf, *_video_args(fmt, quality, fps, threads), "-an", path]
    # stderr goes to a file: a pipe nobody reads until the end can fill and stall the encoder.
    log = tempfile.TemporaryFile()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log, creationflags=media._NOWIN), log


def _render_range(args) -> dict:
    spec, base, start, end, path, fmt, quality, scale, transparent, threads = args
    scene = Scene(spec, base)
    if transparent:
        scene.bg = None
    r = Renderer(scene, scale=scale, even=fmt in ("mp4", "webm", "gif"))
    yuv = r.yuv and fmt in ("mp4", "gif")
    enc, log = _encoder(path, r.w, r.h, scene.fps, fmt, quality, threads, yuv)
    grab = r.frame_i420 if yuv else r.frame
    t0 = time.perf_counter()
    try:
        for n in range(start, end):
            enc.stdin.write(grab(n).data)  # type: ignore[union-attr]
    except BrokenPipeError:
        pass  # ffmpeg quit early; its own message is reported below
    finally:
        scene.close()
        try:
            enc.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass
        enc.wait()
        log.seek(0)
        err = log.read().decode(errors="replace")
        log.close()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg failed on frames {start}-{end}: {err.strip()[:400]}")
    return {"frames": end - start, "seconds": time.perf_counter() - t0}


def _audio_inputs(scene: Scene) -> list[dict]:
    """Every audible source as {src, at, trim0, trim1, gain_db, fade_in, fade_out, speed}."""
    out = []
    for layer in scene.all_layers():
        if layer.type != "video" or not layer.src:
            continue
        vol = layer.raw.get("volume", 1)
        if not vol or not media.probe(layer.src)["audio"]:
            continue
        trim = layer.raw.get("trim") or [0, None]
        speed = float(layer.raw.get("speed", 1))
        fi = layer.enter.dur if layer.enter else 0
        fo = layer.exit.dur if layer.exit else 0
        out.append(dict(src=layer.src, at=layer.t_in, trim0=float(trim[0]), dur=layer.dur * speed, gain=float(vol),
                        fade_in=fi, fade_out=fo, speed=speed))
    for a in scene.audio:
        src = scene.path(a["src"])
        at = scene.timeline._time(a.get("at", 0), "audio", 0.0)
        trim = a.get("trim") or [0, None]
        total = media.probe(src)["duration"]
        end = trim[1] if len(trim) > 1 and trim[1] is not None else total
        dur = min(end - trim[0], scene.duration - at)
        gain = 10 ** (float(a.get("gain", 0)) / 20) if "gain" in a else float(a.get("volume", 1))
        out.append(dict(src=src, at=at, trim0=float(trim[0]), dur=max(0.01, dur), gain=gain,
                        fade_in=float(a.get("fadeIn", 0)), fade_out=float(a.get("fadeOut", 0)), speed=1.0))
    return out


def _copy(video_in: list[str], out: str):
    """Write the video input (one segment, or the concat list of several) to out without re-encoding."""
    if video_in[0] == "-i":
        shutil.move(video_in[1], out)
        return
    res = subprocess.run([media.FFMPEG, "-y", "-v", "error", *video_in, "-c", "copy", out], capture_output=True,
                         text=True, creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"concat failed: {res.stderr.strip()[:400]}")


def _mux_audio(scene: Scene, video_in: list[str], out: str):
    """Mix the scene's audio under the video in one pass; the video stream is copied, joining the
    segments on the way when video_in is a concat list."""
    ins = _audio_inputs(scene)
    if not ins:
        _copy(video_in, out)
        return
    cmd = [media.FFMPEG, "-y", "-v", "error", *video_in]
    chains = []
    for k, a in enumerate(ins):
        cmd += ["-i", a["src"]]
        f = [f"atrim=start={a['trim0']:.4f}:duration={a['dur']:.4f}", "asetpts=PTS-STARTPTS"]
        if a["speed"] != 1:
            f.append(f"atempo={max(0.5, min(2.0, a['speed'])):.4f}")
        if a["fade_in"]:
            f.append(f"afade=t=in:d={a['fade_in']:.3f}")
        if a["fade_out"]:
            f.append(f"afade=t=out:st={max(0, a['dur'] / a['speed'] - a['fade_out']):.3f}:d={a['fade_out']:.3f}")
        f.append(f"volume={a['gain']:.4f}")
        ms = int(round(a["at"] * 1000))
        f.append(f"adelay={ms}|{ms}")
        chains.append(f"[{k + 1}:a]{','.join(f)}[a{k}]")
    mix = "".join(f"[a{k}]" for k in range(len(ins)))
    graph = ";".join(chains) + f";{mix}amix=inputs={len(ins)}:normalize=0:dropout_transition=0,alimiter=limit=0.95,atrim=duration={scene.duration:.4f}[aout]"
    cmd += ["-filter_complex", graph, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out]
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"audio mix failed: {res.stderr.strip()[:400]}")


def render(src, out: str, quality: str = "standard", workers: int | None = None, scale: float = 1.0,
           start: float | None = None, end: float | None = None, transparent: bool = False) -> dict:
    """Render a scene to a file. Format from the extension: .mp4, .mov (ProRes 4444 alpha), .webm (VP9 alpha),
    .gif, or a directory / pattern ending in .png for a frame sequence."""
    scene = load(src)
    spec, base = scene.spec, str(scene.base)
    fps = scene.fps
    f0 = 0 if start is None else max(0, int(round(start * fps)))
    f1 = scene.frames if end is None else min(scene.frames, int(round(end * fps)))
    if f1 <= f0:
        raise ValueError("nothing to render: end is before start")
    out_p = Path(out)
    ext = out_p.suffix.lower()
    t0 = time.perf_counter()
    if ext == ".png" or out.endswith(("/", "\\")):
        return _render_pngs(scene, out_p, f0, f1, scale, transparent)
    fmt = {".mov": "mov", ".webm": "webm", ".gif": "gif"}.get(ext, "mp4")
    transparent = transparent or fmt in ("mov", "webm") and spec.get("canvas", {}).get("bg") in (None, "transparent")
    if fmt == "gif":  # render no larger than the GIF will be (long side GIF_SIDE), not full size then shrink
        scale = min(scale, GIF_SIDE / max(scene.w, scene.h))
    n = f1 - f0
    cpu = os.cpu_count() or 4
    workers = workers or _workers(scene, n, cpu, fmt)
    threads = max(1, cpu // workers)
    tmp = Path(tempfile.mkdtemp(prefix="kine-"))
    try:
        seg_ext = ".mov" if fmt == "mov" else ".webm" if fmt == "webm" else ".mp4"
        bounds = [f0 + n * k // workers for k in range(workers + 1)]
        jobs = [(spec, base, bounds[k], bounds[k + 1], str(tmp / f"seg{k:03d}{seg_ext}"), fmt, quality, scale,
                 transparent, threads)
                for k in range(workers) if bounds[k + 1] > bounds[k]]
        if len(jobs) == 1:
            stats = [_render_range(jobs[0])]
        else:
            with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
                stats = list(ex.map(_render_range, jobs))
        if len(jobs) == 1:
            video_in = ["-i", jobs[0][4]]
        else:
            listing = tmp / "list.txt"
            listing.write_text("".join(f"file '{j[4]}'\n" for j in jobs))
            video_in = ["-f", "concat", "-safe", "0", "-i", str(listing)]
        out_p.parent.mkdir(parents=True, exist_ok=True)
        final = str(tmp / f"final{seg_ext}")
        if ext == ".gif":
            _to_gif(video_in, str(out_p), fps)
        elif fmt == "mp4" and start is None and end is None:
            _mux_audio(scene, video_in, final)
        else:
            _copy(video_in, final)
        if ext != ".gif":
            shutil.move(final, str(out_p))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        scene.close()
    wall = time.perf_counter() - t0
    w, h = max(2, round(scene.w * scale)), max(2, round(scene.h * scale))
    if fmt in ("mp4", "webm", "gif"):
        w, h = w - w % 2, h - h % 2
    return {"out": str(out_p), "w": w, "h": h, "fps": min(fps, GIF_FPS) if fmt == "gif" else fps, "frames": n,
            "seconds": round(n / fps, 3), "wall": round(wall, 2), "fps_rendered": round(n / wall, 1),
            "workers": len(jobs), "size_mb": round(out_p.stat().st_size / 1e6, 2) if out_p.exists() else None}


def _workers(scene: Scene, frames: int, cpu: int, fmt: str = "mp4") -> int:
    """Render processes for a render. On the GPU every process opens its own GL context on the same
    GPU and its own video decoders, so past a few they slow each other down. Measured on a 20-thread
    CPU with one GPU (2026-10-29): 1038 vector frames were fastest with 4-6 workers (3.2 s against
    4.8 s with one); 126 frames, and 312 frames with two video layers, were fastest with one. The
    raster (CPU) backend is bound by the CPU and keeps one worker per 1.5 s of frames. VP9 and
    ProRes encode slower than the GPU draws, so .webm/.mov take up to 4 workers, one per 60 frames
    (1038-frame promo as .webm: 18.7 s with one worker, 10.9 s with four, 11.3 s with eight)."""
    from .render import gpu_context
    if fmt in ("webm", "mov"):
        return max(1, min(4, frames // 60))
    if gpu_context() is None:
        return max(1, min(cpu - 1, 8, frames // int(scene.fps * 1.5) or 1))
    video = any(L.type in ("video", "sequence") for L in scene.all_layers())
    return max(1, min(4, frames // (500 if video else 250)))


GIF_SIDE, GIF_FPS = 720, 15  # a 6 s portrait GIF at full width and 20 fps ran past 10 MB


def _to_gif(video_in: list[str], out: str, fps: float):
    gfps = min(fps, GIF_FPS)
    # 128 colours with ordered dither and changed-rectangle updates: about half the size of error
    # diffusion on moving gradients (6.6 -> 3.5 MB on a 6 s portrait clip) with no visible loss.
    vf = (f"scale=in_color_matrix=bt709:in_range=tv,format=rgb24,fps={gfps},split[a][b];[a]palettegen=stats_mode=diff:max_colors=128[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    res = subprocess.run([media.FFMPEG, "-y", "-v", "error", *video_in, "-vf", vf, out], capture_output=True, text=True,
                         creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"gif failed: {res.stderr.strip()[:400]}")


def _render_pngs(scene: Scene, out: Path, f0: int, f1: int, scale: float, transparent: bool) -> dict:
    import skia
    if transparent:
        scene.bg = None
    folder = out if out.suffix.lower() != ".png" else out.parent
    stem = "frame" if out.suffix.lower() != ".png" else out.stem
    folder.mkdir(parents=True, exist_ok=True)
    r = Renderer(scene, scale=scale)
    t0 = time.perf_counter()
    for n in range(f0, f1):
        r.frame(n)
        r.snapshot().save(str(folder / f"{stem}_{n:05d}.png"), skia.kPNG)
    scene.close()
    return {"out": str(folder), "frames": f1 - f0, "wall": round(time.perf_counter() - t0, 2)}


def still(src, at, out: str, scale: float = 1.0) -> dict:
    """One frame to PNG/JPG. `at` is seconds, negative seconds from the end, or "end" (the last frame)."""
    import skia
    scene = load(src)
    if str(at).strip().lower() == "end":
        n = scene.frames - 1
    else:
        t = float(at)
        n = int(round((scene.duration + t if t < 0 else t) * scene.fps))
    n = max(0, min(scene.frames - 1, n))
    r = Renderer(scene, scale=scale)
    r.frame(n)
    r.snapshot().save(out, skia.kPNG if out.lower().endswith(".png") else skia.kJPEG)
    scene.close()
    return {"out": out, "frame": n, "at": round(n / scene.fps, 3), "w": r.w, "h": r.h}


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
