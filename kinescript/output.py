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
from .render import Renderer, frame_size, loop_end
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
    r = Renderer(scene, scale=scale, even=fmt in ("mp4", "webm", "gif"), alpha=fmt not in ("mp4", "gif"))
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
    """Every audible source as {src, at, trim0, dur, play, gain, fade_in, fade_out, speed, loop, ...}. dur is the
    source seconds the sound spans (its time in the scene times its speed); play is how much of that the source has
    up to its trim end, so a clip trimmed to [2, 4] and given 6 s is silent after its 2 s, unless it loops. A sound
    that ends before the scene starts is left out."""
    out = []
    for layer in scene.all_layers():
        if layer.type != "video" or not layer.src:
            continue
        vol = layer.raw.get("volume", 1)
        up, hidden = layer, False
        while up is not None and not hidden:  # hidden, or inside a hidden group: no picture and no sound
            hidden, up = bool(up.raw.get("hidden")), up.parent
        if hidden or not vol or not media.probe(layer.src)["audio"] or layer.t_out <= 0:
            continue
        trim0 = float((layer.raw.get("trim") or [0])[0])
        speed = float(layer.raw.get("speed", 1))
        # Its sound fades in over its enter and out over its exit, unless fadeIn or fadeOut says otherwise; a
        # sequence sets both, so the sound of two clips crossfades with the picture.
        fi = float(layer.raw.get("fadeIn", layer.enter.dur if layer.enter else 0))
        fo = float(layer.raw.get("fadeOut", layer.exit.dur if layer.exit else 0))
        end = loop_end(layer, media.probe(layer.src)["duration"]) or None  # the trim end, or the end of the file
        loop = None
        if layer.raw.get("loop"):  # the trimmed part repeats, as the picture does (render._draw_video)
            loop = max(0.01, (end or 0) - trim0)
        out.append(_sound(src=layer.src, at=layer.t_in, trim0=trim0, end=end, dur=layer.dur * speed, gain=float(vol),
                          fade_in=fi, fade_out=fo, speed=speed, loop=loop, pan=0.0, role=None, curve=None,
                          onset=False, id=layer.id))
    for s in scene.audio:
        a = s.raw
        if s.t_out <= 0:
            continue
        trim = a.get("trim") or [0, None]
        total = media.probe(s.src)["duration"]
        end = min(float(trim[1]), total or math.inf) if len(trim) > 1 and trim[1] is not None else total or None
        loop = max(0.01, (end or 0) - float(trim[0])) if a.get("loop") else None  # the trimmed segment repeats
        g = a.get("gain")
        curve = g if g is not None and not isinstance(g, (int, float)) else None  # keyframes or an expression
        gain = 10 ** (float(g) / 20) if curve is None and g is not None else float(a.get("volume", 1))
        out.append(_sound(src=s.src, at=s.t_in, trim0=float(trim[0]), end=end,
                          dur=max(0.01, min(s.t_out, scene.duration) - s.t_in), gain=gain,
                          fade_in=float(a.get("fadeIn", 0)), fade_out=float(a.get("fadeOut", 0)), speed=1.0,
                          loop=loop, pan=float(a.get("pan", 0)), role=s.role, curve=curve,
                          onset=s.role == "sfx" and not a.get("trim") and not loop, id=s.id))
    return out


def _sound(end: float | None, **a) -> dict:
    """One mix input: `a` plus play, the source seconds read from trim0, which stop at `end` (the trim end, or the
    end of the file) unless the sound loops. It played on past the trim end before: [2, 4] given 6 s read to 8 s."""
    a["play"] = a["dur"] if a["loop"] or end is None else max(0.0, min(a["dur"], end - a["trim0"]))
    return a


def _tempo(speed: float) -> list[str]:
    """atempo steps whose product is `speed`. One atempo goes from 0.5 to 2, and the mix clamped to that, so a clip
    at speed 4 sounded at twice its speed and fell behind its picture; 4 is now 2 x 2 and 0.1 is 0.5 x 0.5 x 0.5 x
    0.8, and the sound keeps pace at any speed."""
    steps, s = [], float(speed)
    while s > 2.0 + 1e-9:
        steps.append(2.0)
        s /= 2.0
    while s < 0.5 - 1e-9:
        steps.append(0.5)
        s /= 0.5
    if abs(s - 1.0) > 1e-9:
        steps.append(s)
    return [f"atempo={x:.6f}" for x in steps]


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
                         text=True, encoding="utf-8", errors="replace", creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"concat failed: {res.stderr.strip()[:400]}")


_CURVE_IN = ["-f", "f32le", "-ar", str(A.RATE), "-ac", "1", "-i"]
_CURVE = "aresample=48000,aformat=channel_layouts=stereo"


def _pad(duration: float) -> str:
    """Silence after the mix up to `duration`. Counted in samples: with an endless apad, the atrim after it judged
    the end by timestamps, and when ffmpeg 8 started the inputs in an unlucky order those were wrong and the mix
    never ended, filling the disk."""
    return f"asetpts=N/SR/TB,apad=whole_dur={duration:.4f}"


def _graph(ins: list[dict], managed: bool, curves: dict[int, str], duck: str | None, tail: str,
           pad: str) -> tuple[list[str], str]:
    """ffmpeg inputs and the filter graph that mixes `ins` into [aout]. curves: sound index -> its gain curve file;
    duck: the curve file the sounds marked "duck" are multiplied by after they are mixed; tail: filters after the
    mix; pad: what makes the ducked bus last the whole scene. managed: every sound is converted to 48 kHz stereo
    first, so curves and buses line up."""
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
            f = [f"atrim=start={a['trim0']:.4f}:duration={a.get('play', a['dur']):.4f}", "asetpts=PTS-STARTPTS"]
        if managed:
            f.append("aformat=sample_rates=48000:channel_layouts=stereo")
        f.extend(_tempo(a["speed"]))
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
        if a["at"] < 0:  # it started before the scene: the scene hears it from there (adelay takes no negative delay)
            f.extend([f"atrim=start={-a['at']:.4f}", "asetpts=PTS-STARTPTS"])
        if a["pan"]:
            f.append(_pan(a["pan"]))
        ms = int(round(max(0.0, a["at"]) * 1000))
        f.extend([f"adelay={ms}|{ms}", "asetpts=N/SR/TB"])  # adelay's leading silence has no timestamps
        chains.append(head + ("," if head.endswith("amultiply") else "") + ",".join(f) + f"[a{k}]")
    ducked = [f"[a{k}]" for k, a in enumerate(ins) if duck and a.get("duck")]
    mix = [f"[a{k}]" for k, a in enumerate(ins) if f"[a{k}]" not in ducked]
    if ducked:
        chains.append(f"{''.join(ducked)}amix=inputs={len(ducked)}:normalize=0:dropout_transition=0,{pad}[bus]")
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
    pad = _pad(scene.duration)
    args, graph = _graph(ins, False, curves, None, f"alimiter=limit=0.95,{pad},{cut}", pad)
    script = out + ".graph"
    Path(script).write_text(graph, encoding="utf-8")
    cmd = [media.FFMPEG, "-y", "-v", "error", *args, *media.filter_script(script), "-map", "[aout]",
           *_codec(Path(out).suffix.lower()), out]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stderr=log, creationflags=media._NOWIN), log, out


def measure(ins: list[dict]) -> list[dict]:
    """audio.analyze for the part of each source a sound plays, each file and span once, a few at a time."""
    # The source seconds that play: dur is already in source seconds (it was multiplied by speed a second time).
    spans = [(a["src"], a["trim0"], a["trim0"] + (a["loop"] or a.get("play", a["dur"]))) for a in ins]
    todo = list(dict.fromkeys(spans))
    with ThreadPoolExecutor(max(1, min(4, len(todo)))) as ex:
        infos = dict(zip(todo, ex.map(lambda s: A.analyze(*s), todo)))
    return [infos[s] for s in spans]


def plan(scene: Scene, ins: list[dict]) -> dict:
    """The managed mix worked out before ffmpeg runs, changing `ins` in place: each sound with a role set to its
    level (gain), an effect moved past its leading silence (trim0, and shift in seconds) and the sounds under the
    duck marked. Returns the settings, each source's measurements and the duck curve (None without ducking)."""
    cfg = A.settings(scene.spec)
    duck_cfg, voices = cfg["duck"], []
    infos = measure(ins)
    for a, info in zip(ins, infos):
        a["shift"] = 0.0
        if a["role"]:
            if a["onset"] and info["lead"] > 0.005:  # the effect's first sound lands on its cue
                a["trim0"] += info["lead"]
                a["shift"] = info["lead"]
            a["gain"] *= 10 ** (A.level(a["role"], info, cfg["levels"]) / 20)
            if duck_cfg and a["role"] in duck_cfg["by"]:
                voices.append((a["at"], a["at"] + a["dur"] / a["speed"], info["rms"], 20 * math.log10(max(a["gain"], 1e-6))))
        a["duck"] = bool(duck_cfg) and a["role"] in duck_cfg["under"]
    duck = A.duck_curve(voices, scene.duration, duck_cfg) if voices and any(a["duck"] for a in ins) else None
    return {"cfg": cfg, "info": infos, "duck": duck, "ins": ins}


def _managed(scene: Scene, ins: list[dict], out: str, start: float, end: float | None, job: "_Mix") -> dict:
    """Level each sound for its role, duck, mix to a float master, measure it, then correct its loudness, limit
    its peaks and encode. Returns the plan, with the master's measurements and its correction in dB."""
    p = plan(scene, ins)
    cfg, tmp, curves = p["cfg"], Path(out).parent, {}
    for k, a in enumerate(ins):
        if a["curve"] is not None:
            curves[k] = str(tmp / f"gain{k}.f32")
            A.gain_curve(a["curve"], a["dur"], a["at"], scene.fps).astype("<f4").tofile(curves[k])
    duck = None
    if p["duck"] is not None:
        duck = str(tmp / "duck.f32")
        p["duck"].astype("<f4").tofile(duck)
    master, script = str(tmp / "master.wav"), tmp / "master.graph"
    pad = _pad(scene.duration)
    args, graph = _graph(ins, True, curves, duck, f"{pad},atrim=duration={scene.duration:.4f}", pad)
    script.write_text(graph, encoding="utf-8")
    job.run([media.FFMPEG, "-y", "-v", "error", *args, *media.filter_script(str(script)), "-map", "[aout]",
             "-c:a", "pcm_f32le", master])
    measured = A.loudness(job.run([media.FFMPEG, "-hide_banner", "-nostats", "-i", master, "-af", "ebur128=peak=true",
                                   "-f", "null", "-"]))
    # alimiter limits sample peaks; half a dB under the ceiling leaves room for peaks between samples.
    limit = max(0.0625, 10 ** ((cfg["peak"] - 0.5) / 20))
    p["master"], p["correction"] = measured, A.correction(measured, cfg["loudness"])
    af = f"volume={p['correction']:.2f}dB,alimiter=limit={limit:.4f}:level=0:latency=1"
    cut = _cut(start, end, scene.duration)
    job.run([media.FFMPEG, "-y", "-v", "error", "-i", master, "-af", af + ("," + cut if cut else ""),
             *_codec(Path(out).suffix.lower()), out])
    return p


class _Mix:
    """The managed mix's ffmpeg steps, run in a thread so the frames render meanwhile. It answers poll, wait and
    kill like the Popen of the one-process mix, so render() waits for it and stops it the same way."""

    def __init__(self, steps, log):
        self.returncode = self.result = None
        self._log, self._proc, self._stop = log, None, threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=(steps,), daemon=True)
        self._thread.start()

    def _run(self, steps):
        try:
            self.result = steps(self)
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


def _mux(video_in: list[str], audio, out: str, seconds: float):
    """Put the mixed audio under the video, both copied; the segments are joined on the way when video_in
    is a concat list. The file ends with the video (`seconds` long): -shortest dropped the last frame whenever the
    audio came out a few samples shorter, as it does when the frame count rounds the duration up."""
    path = _wait_audio(audio)
    res = subprocess.run([media.FFMPEG, "-y", "-v", "error", *video_in, "-i", path, "-map", "0:v", "-map", "1:a",
                          "-c", "copy", "-t", f"{seconds:.6f}", out], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", creationflags=media._NOWIN)
    if res.returncode:
        raise RuntimeError(f"audio mux failed: {res.stderr.strip()[:400]}")


VIDEO = {".mp4": "mp4", ".mov": "mov", ".webm": "webm", ".gif": "gif"}
IMAGE = (".png", ".jpg", ".jpeg")


def out_path(scene: Scene, out, exts) -> Path:
    """`out` as a Path, or ValueError unless it has one of `exts`, is not a folder, and is not a file the scene
    reads. An unknown extension used to get an MP4 (or JPEG) under that name, a folder got the render's temp
    name inside it, and a render to the scene's own footage replaced the footage."""
    s = str(out if out is not None else "").strip()
    if not s:
        raise ValueError("no output file given")
    p = Path(s)
    if p.is_dir():
        raise ValueError(f"{s} is a folder; give a file name such as {p / ('out' + exts[0])}")
    if p.suffix.lower() not in exts:
        raise ValueError(f"can't write {p.suffix or 'a file with no extension'}: use {', '.join(exts)}")
    reads = {Path(f).resolve() for f in [L.src for L in scene.all_layers()] + [a.src for a in scene.audio] if f}
    if p.resolve() in reads:
        raise ValueError(f"{s} is a file this scene reads; writing there would overwrite it")
    writable(p)
    return p


def writable(p: Path, folder: bool = False) -> None:
    """ValueError unless `p` (a file, or with `folder` a folder of frames) can be written. A read-only -o drew every
    frame first and then failed with [Errno 13]: 2.4 s for a 30 s scene that renders in 2.3 s."""
    if p.is_file():
        try:
            with open(p, "r+b"):  # opens to write without changing it; the read-only flag or a lock refuses
                pass
        except PermissionError:
            raise ValueError(f"{p} can't be written: it is read-only or another program has it open") from None
        return
    d = p if folder else p.parent
    while not d.is_dir() and d != d.parent:  # render makes the folders that do not exist yet
        d = d.parent
    try:
        tempfile.TemporaryFile(dir=d).close()
    except PermissionError:
        raise ValueError(f"can't write in {d}: this account may not create files there") from None
    except OSError:
        pass  # not a permission problem (a drive that is gone, say): the write itself reports it


def _seconds(name: str, v) -> float | None:
    if v is None:
        return None
    try:
        t = float(v)
    except (TypeError, ValueError):
        t = math.nan
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"{name} must be 0 or more seconds, got {v!r}")
    return t


def _scale(scene: Scene, scale) -> None:
    """ValueError unless `scale` draws the scene at 2 px a side or more (and no larger than frame_size allows).
    Below that every side was drawn at 2 px: --scale 0.0001 made a 2x2 video and said nothing."""
    frame_size(scene.w, scene.h, scale)
    if scale < 1 and min(scene.w * scale, scene.h * scale) < 1.5:
        raise ValueError(f"scale {scale:g} makes a {scene.w * scale:.3g} x {scene.h * scale:.3g} px frame of this "
                         f"{scene.w}x{scene.h} scene; each side must be 2 px or more")


def _frames(scene: Scene, start: float | None, end: float | None, notes: list[str]) -> tuple[int, int]:
    """The frames from `start` to `end` seconds: every frame on screen in that time, from the one showing at `start`
    to the last one that starts before `end`. Rounding `start` to the nearest frame made --from 2.99 on a 3 s scene
    round to 3 s and render nothing. A `to` past the end stops at the end, with a note saying so."""
    fps, frames = scene.fps, scene.frames
    f0 = 0 if start is None else math.floor(start * fps + 1e-6)  # within a millionth of a frame of a boundary: on it
    f1 = frames if end is None else math.ceil(end * fps - 1e-6)
    if f0 >= frames:
        raise ValueError(f"from {start:g}s is past the end of the {scene.duration:g}s scene; its last frame starts at "
                         f"{(frames - 1) / fps:.3f}s")
    if f1 <= f0:
        raise ValueError(f"nothing to render: from {start or 0:g}s to {end:g}s holds no frame; to must be after from")
    if f1 > frames:
        notes.append(f"to {end:g}s is past the end of the {scene.duration:g}s scene, so the render stops at the end "
                     f"({frames / fps:.3f}s)")
        f1 = frames
    return f0, f1


def _cap_workers(asked: int, frames: int, cpu: int, notes: list[str]) -> int:
    """`asked` render processes, but no more than one per frame or per CPU: -j 100 on 60 frames started 60."""
    most = max(1, min(frames, cpu))
    if asked > most:
        notes.append(f"workers {asked} is more than this render can use; it used {most} (at most one per frame and "
                     f"one per CPU: {frames} frames, {cpu} CPUs)")
        return most
    return asked


def _noted(res: dict, notes: list[str]) -> dict:
    if notes:
        res["notes"] = notes
    return res


def render(src, out: str, quality: str = "standard", workers: int | None = None, scale: float = 1.0,
           start: float | None = None, end: float | None = None, transparent: bool = False) -> dict:
    """Render a scene to a file. Format from the extension: .mp4, .mov (ProRes 4444 alpha), .webm (VP9 alpha),
    .gif, or a directory / pattern ending in .png for a frame sequence. The result's "notes" say where an argument
    was changed: a `to` past the end, or more workers than the render can use."""
    from .render import backend_setting
    if quality not in QUALITY:
        raise ValueError(f"quality must be one of {', '.join(QUALITY)}, got {quality!r}")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 0):
        raise ValueError(f"workers must be a whole number, 1 or more (0 picks for you), got {workers!r}")
    start, end = _seconds("from", start), _seconds("to", end)
    backend_setting()  # a bad KINE_BACKEND is an error here, not in each worker
    scene = load(src)
    notes: list[str] = []
    try:
        _scale(scene, scale)
        out = str(out if out is not None else "")
        folder = out.endswith(("/", "\\"))  # a folder for a PNG sequence
        out_p = Path(out) if folder else out_path(scene, out, (*VIDEO, ".png", *AUDIO_ONLY))
        if folder:
            writable(out_p, folder=True)
        f0, f1 = _frames(scene, start, end, notes)
    except ValueError:
        scene.close()
        raise
    spec, base = scene.spec, str(scene.base.resolve())  # warm workers keep the directory they started in
    fps = scene.fps
    ext = out_p.suffix.lower()
    t0 = time.perf_counter()
    from .verify import media_warnings_later
    warned = media_warnings_later(scene)  # a clip cut short or with no picture is reported with the result
    if ext == ".png" or folder:
        return warned(_noted(_render_pngs(scene, out_p, f0, f1, scale, transparent), notes))
    if ext in AUDIO_ONLY:
        whole = start is None and end is None
        return warned(_noted(_render_audio(scene, out_p, f0 / fps, f1 / fps, whole, t0), notes))
    fmt = VIDEO[ext]
    if transparent and fmt in ("mp4", "gif"):
        scene.close()
        raise ValueError(f"{ext} keeps no alpha; render a transparent video to .mov or .webm, or frames to .png")
    transparent = transparent or fmt in ("mov", "webm") and spec.get("canvas", {}).get("bg") in (None, "transparent")
    if fmt == "gif":  # render no larger than the GIF will be (long side GIF_SIDE), not full size then shrink
        scale = min(scale, GIF_SIDE / max(scene.w, scene.h))
    n = f1 - f0
    gfps = gif_rate(n, fps) if fmt == "gif" else fps
    cpu = os.cpu_count() or 4
    workers = _cap_workers(workers, n, cpu, notes) if workers else _workers(scene, n, cpu, fmt)
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
            q = "'\\''"  # a quote inside a quoted concat path, as when the temp dir is under C:\Users\O'Brien
            listing.write_text("".join(f"file '{j[4].replace(chr(39), q)}'\n" for j in jobs), encoding="utf-8")
            video_in = ["-f", "concat", "-safe", "0", "-i", str(listing)]
        out_p.parent.mkdir(parents=True, exist_ok=True)
        final = str(tmp / f"final{seg_ext}")
        if ext == ".gif":
            _to_gif(video_in, str(out_p), gfps)
        elif audio:
            _mux(video_in, audio, final, n / fps)
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
    done = {"out": str(out_p), "w": w, "h": h, "fps": gfps if fmt == "gif" else fps,
            "frames": gif_frames(n, fps, gfps) if fmt == "gif" else n,  # was the scene's frames for a GIF
            "seconds": round(n / fps, 3), "wall": round(wall, 2), "fps_rendered": round(n / wall, 1),
            "workers": len(jobs), "size_mb": round(out_p.stat().st_size / 1e6, 2) if out_p.exists() else None,
            "audio": bool(audio)}
    return warned(_noted(done, notes))


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
            _POOL.update(ex=ProcessPoolExecutor(max_workers=n, initializer=_orphan_exit), key=key, size=n)
        _POOL["busy"] += 1
        _POOL["gen"] += 1
        return _POOL["ex"]


def _orphan_exit():
    """In a render worker: exit when the process that started it is gone. The warm pool's idle timer runs in that
    process, so a killed MCP server left its workers running until someone killed them."""
    import multiprocessing
    from multiprocessing.connection import wait
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch():
        wait([parent.sentinel])
        os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


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


def gif_rate(frames: int, fps: float) -> float:
    """The GIF's frame rate: GIF_FPS at most, or the scene's own when GIF_FPS would leave no frame (one frame of a
    60 fps scene; ffmpeg then failed with "Error writing trailer")."""
    return min(fps, GIF_FPS) if gif_frames(frames, fps, min(fps, GIF_FPS)) else fps


def gif_frames(frames: int, fps: float, gfps: float) -> int:
    """How many frames ffmpeg's fps filter keeps of `frames` at `fps` when it resamples them to `gfps` (it rounds
    each output time to the nearest input frame). Measured against ffprobe for 1 to 39, 61, 89 to 91 and 151 frames
    at 12, 15, 23.976, 24, 25, 29.97, 30, 50 and 60 fps: all equal."""
    return math.floor(frames * gfps / fps + 0.5)


def _to_gif(video_in: list[str], out: str, gfps: float):
    # 128 colours with ordered dither and changed-rectangle updates: about half the size of error
    # diffusion on moving gradients (6.6 -> 3.5 MB on a 6 s portrait clip) with no visible loss.
    vf = (f"scale=in_color_matrix=bt709:in_range=tv,format=rgb24,fps={gfps},split[a][b];[a]palettegen=stats_mode=diff:max_colors=128[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    res = subprocess.run([media.FFMPEG, "-y", "-v", "error", *video_in, "-vf", vf, out], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", creationflags=media._NOWIN)
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
    if transparent:
        scene.bg = None
    folder = out if out.suffix.lower() != ".png" else out.parent
    stem = "frame" if out.suffix.lower() != ".png" else out.stem
    try:  # a printf pattern such as seq_%04d.png numbers the files itself
        stem % 0
        name = stem.__mod__
    except (TypeError, ValueError):
        name = f"{stem}_{{:05d}}".format
    t0 = time.perf_counter()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        r = Renderer(scene, scale=scale)
        for n in range(f0, f1):
            r.frame(n)
            save_image(r.snapshot(), folder / f"{name(n)}.png")
    finally:
        scene.close()
    return {"out": str(folder), "frames": f1 - f0, "wall": round(time.perf_counter() - t0, 2)}


def still(src, at, out: str, scale: float = 1.0) -> dict:
    """One frame to PNG/JPG. `at` is seconds, negative seconds from the end, or "end" (the last frame)."""
    scene = load(src)
    try:
        n = frame_at(scene, at)
        note = at_note(scene, at)
        _scale(scene, scale)
        p = out_path(scene, out, IMAGE)
        r = Renderer(scene, scale=scale)
        r.frame(n)
        save_image(r.snapshot(), p)
    finally:
        scene.close()
    return _noted({"out": str(p), "frame": n, "at": round(n / scene.fps, 3), "w": r.w, "h": r.h},
                  [note] if note else [])


def save_image(img, path: Path):
    """PNG or JPEG by the extension, parent folders made. Skia's own save returns None whether or not it wrote
    the file, so a still into a folder that did not exist said ok and left nothing."""
    import skia
    data = img.encodeToData(skia.kPNG if path.suffix.lower() == ".png" else skia.kJPEG, 100)
    if data is None:
        raise RuntimeError(f"could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(data))


def frame_at(scene: Scene, at) -> int:
    """The frame `at` names: seconds, negative seconds from the end, or "end" (the last frame); clamped to the scene."""
    if str(at).strip().lower() == "end":
        return scene.frames - 1
    try:
        t = float(at)
    except (TypeError, ValueError):
        t = math.nan
    if not math.isfinite(t):
        raise ValueError(f"at must be seconds, negative seconds from the end, or \"end\"; got {at!r}")
    n = int(round((scene.duration + t if t < 0 else t) * scene.fps))
    return max(0, min(scene.frames - 1, n))


def at_note(scene: Scene, at) -> str | None:
    """What frame_at did with a time outside the scene, for the result to say; None for a time inside it.
    still --at 99 and look --at 99 drew the last frame and said nothing."""
    if str(at).strip().lower() == "end":
        return None
    t = float(at)
    s = scene.duration + t if t < 0 else t
    if s > scene.duration + 1e-9:
        return (f"at {t:g}s is past the end of the {scene.duration:g}s scene, so it drew the last frame "
                f"({(scene.frames - 1) / scene.fps:.3f}s)")
    if s < -1e-9:
        return f"at {t:g}s is before the start of the {scene.duration:g}s scene, so it drew the first frame (0s)"
    return None


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
