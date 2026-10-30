"""Encoding: frames piped to ffmpeg, split across worker processes, audio mixed alongside them."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from . import audio as A
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
                        fade_in=fi, fade_out=fo, speed=speed, loop=None, pan=0.0, role=None, curve=None,
                        onset=False))
    for s in scene.audio:
        a = s.raw
        trim = a.get("trim") or [0, None]
        loop = None
        if a.get("loop"):  # the trimmed segment repeats
            total = media.probe(s.src)["duration"]
            end = trim[1] if len(trim) > 1 and trim[1] is not None else total
            loop = max(0.01, min(float(end), total) - float(trim[0]))
        g = a.get("gain")
        curve = g if g is not None and not isinstance(g, (int, float)) else None  # keyframes or an expression
        gain = 10 ** (float(g) / 20) if curve is None and g is not None else float(a.get("volume", 1))
        out.append(dict(src=s.src, at=s.t_in, trim0=float(trim[0]), dur=max(0.01, min(s.t_out, scene.duration) - s.t_in),
                        gain=gain, fade_in=float(a.get("fadeIn", 0)), fade_out=float(a.get("fadeOut", 0)), speed=1.0,
                        loop=loop, pan=float(a.get("pan", 0)), role=s.role, curve=curve,
                        onset=s.role == "sfx" and not a.get("trim") and not loop))
    return out


def _codec(ext: str) -> list[str] | None:
    """Audio encoder arguments for a file extension; None when this ffmpeg cannot write it."""
    if ext in (".opus", ".ogg", ".webm"):
        enc = media.capabilities()["encoders"]
        if "libopus" in enc:
            return ["-c:a", "libopus", "-b:a", "160k", "-ar", "48000"]
        if "opus" in enc:  # ffmpeg's own Opus encoder is marked experimental
            return ["-c:a", "opus", "-strict", "-2", "-b:a", "160k", "-ar", "48000"]
        return None
    return {".wav": ["-c:a", "pcm_s16le"], ".flac": ["-c:a", "flac"],
            ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"]}.get(ext, ["-c:a", "aac", "-b:a", "192k"])


def _pan(p: float) -> str:
    """Constant-power pan: -1 left, 0 unchanged, 1 right."""
    th = (p + 1) * math.pi / 4
    return f"aformat=channel_layouts=stereo,pan=stereo|c0={math.sqrt(2) * math.cos(th):.4f}*c0|c1={math.sqrt(2) * math.sin(th):.4f}*c1"


def _copy(video_in: list[str], out: str):
    """Write the video input (one segment, or the concat list of several) to out without re-encoding."""
    if video_in[0] == "-i":
        shutil.move(video_in[1], out)
        return
    res = subprocess.run([media.FFMPEG, "-y", "-v", "error", *video_in, "-c", "copy", out], capture_output=True,
                         text=True, creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"concat failed: {res.stderr.strip()[:400]}")


_CURVE_IN = ["-f", "f32le", "-ar", str(A.RATE), "-ac", "1", "-i"]
_CURVE = "aresample=48000,aformat=channel_layouts=stereo"


def _graph(ins: list[dict], managed: bool, curves: dict[int, str], duck: str | None, tail: str) -> tuple[list[str], str]:
    """ffmpeg inputs and the filter graph that mixes `ins` into [aout]. curves: sound index -> its gain curve file;
    duck: the curve file the sounds marked "duck" are multiplied by after they are mixed; tail: filters after the
    mix. managed: every sound is converted to 48 kHz stereo first, so curves and buses line up."""
    args = [x for a in ins for x in ("-i", a["src"])]
    n = len(ins)
    chains = []

    def curve(path: str, label: str):
        nonlocal n
        args.extend([*_CURVE_IN, path])
        chains.append(f"[{n}:a]{_CURVE}[{label}]")
        n += 1

    for k, a in enumerate(ins):
        if a["loop"]:  # 48 kHz first, so aloop's size in samples is known
            f = [f"atrim=start={a['trim0']:.4f}:duration={a['loop']:.4f}", "asetpts=PTS-STARTPTS", "aresample=48000",
                 f"aloop=loop=-1:size={round(a['loop'] * 48000)}", f"atrim=duration={a['dur']:.4f}"]
        else:
            f = [f"atrim=start={a['trim0']:.4f}:duration={a['dur']:.4f}", "asetpts=PTS-STARTPTS"]
        if managed:
            f.append("aformat=sample_rates=48000:channel_layouts=stereo")
        if a["speed"] != 1:
            f.append(f"atempo={max(0.5, min(2.0, a['speed'])):.4f}")
        if a["fade_in"]:
            f.append(f"afade=t=in:d={a['fade_in']:.3f}")
        if a["fade_out"]:
            f.append(f"afade=t=out:st={max(0, a['dur'] / a['speed'] - a['fade_out']):.3f}:d={a['fade_out']:.3f}")
        f.append(f"volume={a['gain']:.4f}")
        head = f"[{k}:a]"
        if k in curves:
            chains.append(head + ",".join(f) + f"[p{k}]")
            curve(curves[k], f"c{k}")
            head, f = f"[p{k}][c{k}]amultiply", []
        if a["pan"]:
            f.append(_pan(a["pan"]))
        ms = int(round(a["at"] * 1000))
        f.append(f"adelay={ms}|{ms}")
        if managed:  # adelay leaves its leading silence without timestamps, and the atrims after the mix need them
            f.append("asetpts=N/SR/TB")
        chains.append(head + ("," if head.endswith("amultiply") else "") + ",".join(f) + f"[a{k}]")
    ducked = [f"[a{k}]" for k, a in enumerate(ins) if duck and a.get("duck")]
    mix = [f"[a{k}]" for k, a in enumerate(ins) if f"[a{k}]" not in ducked]
    if ducked:
        chains.append(f"{''.join(ducked)}amix=inputs={len(ducked)}:normalize=0:dropout_transition=0,apad[bus]")
        curve(duck, "dk")
        chains.append("[bus][dk]amultiply[ducked]")
        mix.append("[ducked]")
    return args, ";".join(chains) + f";{''.join(mix)}amix=inputs={len(mix)}:normalize=0:dropout_transition=0,{tail}[aout]"


def _cut(start: float, end: float | None, duration: float) -> str:
    return "" if not start and end is None else \
        f"atrim=start={start:.4f}:end={duration if end is None else end:.4f},asetpts=PTS-STARTPTS"


def _start_audio(scene: Scene, out: str, start: float = 0.0, end: float | None = None):
    """Start mixing the scene's audio to `out` (encoder from its extension) so it runs while the frames render;
    it took 0.7 s of the launch video's 3.8 s when it ran after them. start/end cut a range out of the full mix,
    so fades crossing them stay as they are. Returns (process, log, out), or None if the scene is silent. A scene
    with roles gets the managed mix (audio.py), in a thread; any other mix is one ffmpeg process."""
    ins = _audio_inputs(scene)
    if not ins:
        return None
    log = tempfile.TemporaryFile()  # as in _encoder: an unread stderr pipe can stall ffmpeg
    if any(a["role"] for a in ins):
        return _Mix(lambda job: _managed(scene, ins, out, start, end, job), log), log, out
    curves = {}
    for k, a in enumerate(ins):
        if a["curve"] is not None:
            curves[k] = f"{out}.gain{k}.f32"
            A.gain_curve(a["curve"], a["dur"], a["at"], scene.fps).astype("<f4").tofile(curves[k])
    # Padded to the full length: the mux keeps the shorter stream, so a mix that ends early would cut the video.
    cut = _cut(start, end, scene.duration) or f"atrim=duration={scene.duration:.4f}"
    args, graph = _graph(ins, False, curves, None, f"alimiter=limit=0.95,apad,{cut}")
    script = out + ".graph"
    Path(script).write_text(graph, encoding="utf-8")
    cmd = [media.FFMPEG, "-y", "-v", "error", *args, *media.filter_script(script), "-map", "[aout]",
           *_codec(Path(out).suffix.lower()), out]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stderr=log, creationflags=media._NOWIN), log, out


def _managed(scene: Scene, ins: list[dict], out: str, start: float, end: float | None, job: "_Mix"):
    """Level each sound for its role, duck, mix to a float master, measure it, then correct its loudness, limit
    its peaks and encode."""
    cfg = A.settings(scene.spec)
    duck_cfg, tmp = cfg["duck"], Path(out).parent
    voices, curves = [], {}
    spans = [(a["src"], a["trim0"], a["trim0"] + (a["loop"] or a["dur"] * a["speed"])) if a["role"] else None
             for a in ins]
    todo = list(dict.fromkeys(s for s in spans if s))  # each file and span measured once, a few at a time
    with ThreadPoolExecutor(min(4, len(todo))) as ex:
        infos = dict(zip(todo, ex.map(lambda s: A.analyze(*s), todo)))
    for k, a in enumerate(ins):
        if a["role"]:
            info = infos[spans[k]]
            if a["onset"] and info["lead"] > 0.005:  # the effect's first sound lands on its cue
                a["trim0"] += info["lead"]
            a["gain"] *= 10 ** (A.level(a["role"], info, cfg["levels"]) / 20)
            if duck_cfg and a["role"] in duck_cfg["by"]:
                voices.append((a["at"], a["at"] + a["dur"] / a["speed"], info["rms"], 20 * math.log10(max(a["gain"], 1e-6))))
        a["duck"] = bool(duck_cfg) and a["role"] in duck_cfg["under"]
        if a["curve"] is not None:
            curves[k] = str(tmp / f"gain{k}.f32")
            A.gain_curve(a["curve"], a["dur"], a["at"], scene.fps).astype("<f4").tofile(curves[k])
    duck = None
    if voices and any(a["duck"] for a in ins):
        duck = str(tmp / "duck.f32")
        A.duck_curve(voices, scene.duration, duck_cfg).astype("<f4").tofile(duck)
    master, script = str(tmp / "master.wav"), tmp / "master.graph"
    args, graph = _graph(ins, True, curves, duck, f"apad,atrim=duration={scene.duration:.4f}")
    script.write_text(graph, encoding="utf-8")
    job.run([media.FFMPEG, "-y", "-v", "error", *args, *media.filter_script(str(script)), "-map", "[aout]",
             "-c:a", "pcm_f32le", master])
    measured = A.loudness(job.run([media.FFMPEG, "-hide_banner", "-nostats", "-i", master, "-af", "ebur128=peak=true",
                                   "-f", "null", "-"]))
    # alimiter limits sample peaks; half a dB under the ceiling leaves room for peaks between samples.
    limit = max(0.0625, 10 ** ((cfg["peak"] - 0.5) / 20))
    af = f"volume={A.correction(measured, cfg['loudness']):.2f}dB,alimiter=limit={limit:.4f}:level=0:latency=1"
    cut = _cut(start, end, scene.duration)
    job.run([media.FFMPEG, "-y", "-v", "error", "-i", master, "-af", af + ("," + cut if cut else ""),
             *_codec(Path(out).suffix.lower()), out])


class _Mix:
    """The managed mix's ffmpeg steps, run in a thread so the frames render meanwhile. It answers poll, wait and
    kill like the Popen of the one-process mix, so render() waits for it and stops it the same way."""

    def __init__(self, steps, log):
        self.returncode = None
        self._log, self._proc, self._stop = log, None, threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self._thread.start()

    def _run(self, steps):
        try:
            steps(self)
            code = 0
        except Exception as e:  # noqa: BLE001 - any failure is reported like a failed ffmpeg, never as success
            self._log.write((str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}").encode())
            code = 1
        self.returncode = code

    def run(self, cmd: list[str]) -> str:
        """Run one ffmpeg step and return its log; raises once kill() has been called."""
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("stopped")
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, creationflags=media._NOWIN)
        err = self._proc.communicate()[1].decode(errors="replace")
        if self._proc.returncode:
            raise RuntimeError(err.strip()[-400:] or f"ffmpeg exited with {self._proc.returncode}")
        return err

    def poll(self):
        return self.returncode

    def wait(self):
        self._thread.join()
        return self.returncode

    def kill(self):
        with self._lock:
            self._stop.set()
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()


def _wait_audio(audio) -> str:
    """Wait for a mix started by _start_audio and return its file."""
    proc, log, path = audio
    proc.wait()
    log.seek(0)
    err = log.read().decode(errors="replace")
    log.close()
    if proc.returncode:
        raise RuntimeError(f"audio mix failed: {err.strip()[:400]}")
    return path


def _mux(video_in: list[str], audio, out: str):
    """Put the mixed audio under the video, both copied; the segments are joined on the way when video_in
    is a concat list."""
    path = _wait_audio(audio)
    res = subprocess.run([media.FFMPEG, "-y", "-v", "error", *video_in, "-i", path, "-map", "0:v", "-map", "1:a",
                          "-c", "copy", "-shortest", out], capture_output=True, text=True, creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"audio mux failed: {res.stderr.strip()[:400]}")


def render(src, out: str, quality: str = "standard", workers: int | None = None, scale: float = 1.0,
           start: float | None = None, end: float | None = None, transparent: bool = False) -> dict:
    """Render a scene to a file. Format from the extension: .mp4, .mov (ProRes 4444 alpha), .webm (VP9 alpha),
    .gif, or a directory / pattern ending in .png for a frame sequence."""
    scene = load(src)
    spec, base = scene.spec, str(scene.base.resolve())  # warm workers keep the directory they started in
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
    if ext in AUDIO_ONLY:
        return _render_audio(scene, out_p, f0 / fps, f1 / fps, start is None and end is None, t0)
    fmt = {".mov": "mov", ".webm": "webm", ".gif": "gif"}.get(ext, "mp4")
    transparent = transparent or fmt in ("mov", "webm") and spec.get("canvas", {}).get("bg") in (None, "transparent")
    if fmt == "gif":  # render no larger than the GIF will be (long side GIF_SIDE), not full size then shrink
        scale = min(scale, GIF_SIDE / max(scene.w, scene.h))
    n = f1 - f0
    cpu = os.cpu_count() or 4
    workers = workers or _workers(scene, n, cpu, fmt)
    threads = max(1, cpu // workers)
    tmp = Path(tempfile.mkdtemp(prefix="kine-"))
    audio = None
    a_ext = {"mp4": ".m4a", "mov": ".wav", "webm": ".opus"}.get(fmt)
    mix = a_ext is not None and _codec(a_ext) is not None
    a_args = (str(tmp / f"audio{a_ext}"), 0.0, None) if start is None and end is None else \
        (str(tmp / f"audio{a_ext}"), f0 / fps, f1 / fps)
    try:
        seg_ext = ".mov" if fmt == "mov" else ".webm" if fmt == "webm" else ".mp4"
        bounds = [f0 + n * k // workers for k in range(workers + 1)]
        jobs = [(spec, base, bounds[k], bounds[k + 1], str(tmp / f"seg{k:03d}{seg_ext}"), fmt, quality, scale,
                 transparent, threads)
                for k in range(workers) if bounds[k + 1] > bounds[k]]
        if len(jobs) == 1:
            audio = _start_audio(scene, *a_args) if mix else None
            stats = [_render_range(jobs[0])]
        else:
            ex, failed = _pool(len(jobs)), True
            try:
                pending = ex.map(_render_range, jobs)  # submitted now; the audio mix starts behind them
                audio = _start_audio(scene, *a_args) if mix else None
                stats = list(pending)
                failed = False
            finally:
                _rest(failed)
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
        elif audio:
            _mux(video_in, audio, final)
        else:
            _copy(video_in, final)
        if ext != ".gif":
            shutil.move(final, str(out_p))
    finally:
        if audio and audio[0].poll() is None:  # the frames failed while the mix was still running
            audio[0].kill()
            audio[0].wait()
        if audio:
            audio[1].close()
        shutil.rmtree(tmp, ignore_errors=True)
        scene.close()
    wall = time.perf_counter() - t0
    w, h = max(2, round(scene.w * scale)), max(2, round(scene.h * scale))
    if fmt in ("mp4", "webm", "gif"):
        w, h = w - w % 2, h - h % 2
    return {"out": str(out_p), "w": w, "h": h, "fps": min(fps, GIF_FPS) if fmt == "gif" else fps, "frames": n,
            "seconds": round(n / fps, 3), "wall": round(wall, 2), "fps_rendered": round(n / wall, 1),
            "workers": len(jobs), "size_mb": round(out_p.stat().st_size / 1e6, 2) if out_p.exists() else None,
            "audio": bool(audio)}


# Render processes stay up for WARM seconds after a render, so the next one in the same process (the MCP
# server, a script rendering several scenes) skips starting them: each imports Skia and opens a GL context.
# Repeat renders of examples/model.json took 1.7 s instead of 2.45 s, the launch video 2.2 s instead of
# 2.6 s (2026-10-30). What they cache is keyed on file versions (media, models) or on parameters (meshes,
# eases, colours), so nothing they keep goes stale. 0 shuts them down at the end of each render.
WARM = 120.0
_POOL: dict = {"busy": 0, "gen": 0}
_POOL_LOCK = threading.Lock()


def _pool(n: int) -> ProcessPoolExecutor:
    """The warm pool, new when it has fewer than n processes or KINE_BACKEND changed since it started."""
    key = os.environ.get("KINE_BACKEND", "")
    with _POOL_LOCK:
        if "ex" in _POOL and (_POOL["key"] != key or _POOL["size"] < n):
            _drop()
        if "ex" not in _POOL:
            _POOL.update(ex=ProcessPoolExecutor(max_workers=n), key=key, size=n)
        _POOL["busy"] += 1
        _POOL["gen"] += 1
        return _POOL["ex"]


def _rest(failed: bool):
    """After a render: a failed pool shuts down now (its other jobs finish first), an idle one after WARM s."""
    with _POOL_LOCK:
        _POOL["busy"] -= 1
        if failed or WARM <= 0:
            _drop()
            return
        gen = _POOL["gen"]
    t = threading.Timer(WARM, _idle, (gen,))
    t.daemon = True
    t.start()


def _idle(gen: int):
    with _POOL_LOCK:
        if _POOL["gen"] == gen and not _POOL["busy"]:
            _drop()


def _drop():
    ex = _POOL.pop("ex", None)
    if ex is not None:
        ex.shutdown(wait=True, cancel_futures=True)


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
    layers = list(scene.all_layers())
    if any(L.type == "model" for L in layers):
        # Model frames spend their time in numpy and in building Skia's vertex lists, not on the GPU,
        # so they spread over processes: examples/model.json (180 frames, ~11k triangles drawn a
        # frame) took 4.2 s with one worker, 2.3 s with six and 2.2 s with eight (2026-10-30).
        return max(1, min(6, frames // 30))
    video = any(L.type in ("video", "sequence") for L in layers)
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


AUDIO_ONLY = (".wav", ".m4a", ".mp3", ".flac", ".opus")


def _render_audio(scene: Scene, out: Path, start: float, end: float, full: bool, t0: float) -> dict:
    """The mix alone, without rendering a frame."""
    if _codec(out.suffix.lower()) is None:
        scene.close()
        raise RuntimeError(f"this ffmpeg has no encoder for {out.suffix} files; use .wav or .m4a")
    tmp = Path(tempfile.mkdtemp(prefix="kine-"))
    try:
        audio = _start_audio(scene, str(tmp / f"mix{out.suffix.lower()}"), *((0.0, None) if full else (start, end)))
        if audio is None:
            raise ValueError("the scene has no audio")
        path = _wait_audio(audio)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, str(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        scene.close()
    return {"out": str(out), "seconds": round(end - start, 3), "wall": round(time.perf_counter() - t0, 2),
            "size_mb": round(out.stat().st_size / 1e6, 2)}


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
