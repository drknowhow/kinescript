"""The managed mix, used when any sound in a scene has a role: each sound is set to its role's level, music and
ambience dip while a voice speaks, and the whole mix is brought to one loudness under a peak ceiling.

Every gain here is worked out from the files and the scene, never by listening at render time, so a scene always
mixes the same way and the numbers can be reported to an agent that cannot hear.
"""
from __future__ import annotations

import math
import re
import subprocess
from functools import lru_cache

import numpy as np

from . import media
from .props import Ctx, compile_prop

ROLES = ("voice", "music", "sfx", "ambience")
# voice, music and ambience in LUFS; sfx as a sample peak in dBFS, since a short effect has no useful loudness.
LEVELS = {"voice": -18.0, "music": -16.0, "ambience": -30.0, "sfx": -6.0}
DUCK = {"by": ["voice"], "under": ["music", "ambience"], "depth": 14.0, "attack": 0.08, "hold": 0.15, "release": 0.4}
LOUDNESS, PEAK = -16.0, -1.5  # the whole mix: LUFS, and dBTP
RATE = 1000  # gain curve samples per second
_AN = 8000  # analysis sample rate
_WIN = 0.02  # RMS window, seconds
SPEECH = -45.0  # dBFS: a voice window louder than this, after levelling, ducks the music


def settings(spec: dict) -> dict:
    """The scene's "mix" block over the defaults."""
    m = spec.get("mix") or {}
    duck = m.get("duck", {})
    if duck is not False:
        duck = dict(DUCK, **(duck if isinstance(duck, dict) else {}))
        for k in ("by", "under"):
            duck[k] = [duck[k]] if isinstance(duck[k], str) else list(duck[k])
    return {"loudness": float(m.get("loudness", LOUDNESS)), "peak": float(m.get("peak", PEAK)),
            "levels": dict(LEVELS, **(m.get("levels") or {})), "duck": duck or None}


def analyze(src: str, start: float = 0.0, end: float | None = None) -> dict:
    """Loudness and shape of src between start and end (seconds in the file), measured once per file version:
    lufs, peak (sample, dBFS), true_peak (dBTP), power (mean, dBFS), rms (dBFS per 20 ms window) and lead
    (seconds of silence before the first sound)."""
    return _analyze(*media._version(src), round(float(start), 4), None if end is None else round(float(end), 4))


@lru_cache(maxsize=256)
def _analyze(src: str, mtime: int, size: int, start: float, end: float | None) -> dict:
    return measure(src, start, end)


def measure(src: str, start: float = 0.0, end: float | None = None) -> dict:
    """analyze without the cache, for a file that is made once and measured once, like a finished mix."""
    cut = f"atrim=start={start}" + ("" if end is None else f":end={end}") + ",asetpts=PTS-STARTPTS,"
    res = subprocess.run([media.FFMPEG, "-hide_banner", "-nostats", "-i", src, "-af",
                          cut + f"ebur128=peak=true+sample,aresample={_AN},aformat=sample_fmts=flt:channel_layouts=mono",
                          "-f", "f32le", "-"], capture_output=True, creationflags=media._NOWIN)
    if res.returncode:
        raise media.MediaError(f"could not measure {src}: {res.stderr.decode(errors='replace').strip()[-300:]}")
    info = loudness(res.stderr.decode(errors="replace"))
    pcm = np.frombuffer(res.stdout, np.float32).astype(np.float64)
    w = int(_AN * _WIN)
    n = len(pcm) // w
    info["rms"] = 10 * np.log10(np.mean(pcm[: n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    info["power"] = float(10 * np.log10(np.mean(pcm ** 2) + 1e-12)) if len(pcm) else -math.inf
    floor = 10 ** (min(-60.0, info["peak"] - 50) / 20) if math.isfinite(info["peak"]) else 1.0
    loud = np.abs(pcm) > floor
    info["lead"] = float(np.argmax(loud)) / _AN if loud.any() else 0.0
    return info


def loudness(log: str) -> dict:
    """lufs, peak, true_peak and lra (loudness range, LU) from the summary ffmpeg's ebur128 filter writes to its
    log, and short_max, the loudest 3 s, from the lines it writes every 0.1 s."""
    at = log.rfind("Summary:")
    s = log[at:]

    def num(pattern: str) -> float:
        m = re.search(pattern, s)
        return float(m.group(1)) if m else -math.inf
    short = [float(x) for x in re.findall(r"\bS:\s*(-?[\d.]+)", log[:at])]
    return {"lufs": num(r"I:\s+(-?[\d.]+|-inf) LUFS"), "peak": num(r"Sample peak:\s+Peak:\s+(-?[\d.]+|-inf)"),
            "true_peak": num(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf)"), "lra": num(r"LRA:\s+(-?[\d.]+) LU"),
            "short_max": max((x for x in short if x > -70), default=-math.inf)}


def level(role: str, info: dict, levels: dict) -> float:
    """dB that sets a sound to its role's level, between -40 and +24; 0 for a silent file."""
    if role == "sfx":
        ref = info["peak"]
    else:  # under 0.4 s ebur128 reports -70 LUFS: use the mean power instead
        ref = info["lufs"] if info["lufs"] > -69.9 else info["power"]
    if not math.isfinite(ref) or ref < -90:
        return 0.0
    return max(-40.0, min(24.0, levels[role] - ref))


def correction(measured: dict, target: float) -> float:
    """dB that brings the whole mix to the target loudness, between -20 and +12; 0 for a silent mix."""
    i = measured["lufs"]
    return max(-20.0, min(12.0, target - i)) if i > -69.9 else 0.0


def duck_curve(voices: list[tuple[float, float, np.ndarray, float]], duration: float, cfg: dict) -> np.ndarray:
    """Linear gain at RATE over the scene, plus a spare second, for the sounds that duck. It is 1, and -depth dB
    while a voice speaks: the fall starts `attack` seconds before the voice, so the music is already down for the
    first syllable; pauses shorter than `hold` stay down; the rise takes `release` seconds.
    voices: (start, end, RMS in dBFS per 20 ms window from the start, gain in dB applied to the voice)."""
    n = int(math.ceil(duration * RATE)) + RATE
    active = np.zeros(n, bool)
    per = int(RATE * _WIN)
    for t0, t1, rms, gain in voices:
        on = np.repeat(rms + gain > SPEECH, per)
        a0 = int(round(t0 * RATE))
        if a0 < 0:  # a voice that started before the scene: the part before 0 s is not in it
            on, a0 = on[-a0:], 0
        a, b = a0, min(n, int(round(t1 * RATE)))
        m = min(b - a, len(on))
        if m > 0:
            active[a:a + m] |= on[:m]
    hold = int(cfg["hold"] * RATE)
    if hold:
        active = np.convolve(active, np.ones(hold + 1), "full")[:n] > 0
    depth = float(cfg["depth"])
    d, idx = np.where(active, depth, 0.0), np.arange(n)
    up = depth / max(1.0, cfg["release"] * RATE)
    d = np.maximum.accumulate(d + up * idx) - up * idx  # after each stretch, rise over `release`
    down = depth / max(1.0, cfg["attack"] * RATE)
    d = (np.maximum.accumulate(d[::-1] + down * idx) - down * idx)[::-1]  # before each, fall over `attack`
    return 10 ** (-np.clip(d, 0.0, depth) / 20)


def gain_curve(raw, dur: float, start: float, fps: float) -> np.ndarray:
    """A sound's keyframed or expression gain (dB, timed from the sound's start) as linear gain at RATE, plus a
    spare second. Sampled at 100 Hz and interpolated: gain moves slowly next to the audio."""
    prop = compile_prop("gain", raw, Ctx(1, 1, fps))
    ts = np.linspace(0.0, dur, max(2, int(math.ceil(dur * 100)) + 1))
    db = np.array([float(prop.at(t, dur, start + t, 0)) for t in ts])
    n = int(math.ceil(dur * RATE)) + RATE
    return 10 ** (np.interp(np.arange(n) / RATE, ts, db) / 20)
