"""Evidence about the sound, for an agent that cannot hear it.

report() renders a scene's mix once and measures it: each sound's level, gain and place, the duck, and the finished
mix's loudness and peaks. issues() turns that into check's problems, each with a time range and a fix, and strip()
draws it as lanes under a contact sheet, on the same time axis as the frames above.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import skia

from . import audio as A
from . import media
from . import output as O
from .render import _argb
from .scene import Scene

WIN = A._WIN  # seconds per envelope window
GAP = 10.0  # LU a voice should sit above the music and ambience under it
SILENT = -60.0  # dBFS: a sound whose loudest sample is quieter plays nothing
LATE = 0.05  # seconds of silence before an effect's first sound that a viewer notices
LANES = (("voice", "#b9f45d"), ("music", "#5db9f4"), ("ambience", "#a08cf4"), ("sfx", "#f4b95d"), (None, "#9aa0a8"))
LANE_H, LABEL_W, AXIS_H = 44, 96, 22
_CACHE: dict[str, dict] = {}  # check, look and render in one process (the MCP server) measure a mix once


def report(scene: Scene) -> dict | None:
    """The scene's mix rendered and measured; None when the scene is silent."""
    ins = O._audio_inputs(scene)
    if not ins:
        return None
    key = repr((ins, [media._version(a["src"]) for a in ins], scene.spec.get("mix"), round(scene.duration, 4)))
    if key in _CACHE:
        return _CACHE[key]
    tmp = Path(tempfile.mkdtemp(prefix="kine-"))
    try:
        mix = O._start_audio(scene, str(tmp / "mix.wav"))
        master = A.measure(O._wait_audio(mix))
        plan = getattr(mix[0], "result", None)  # the managed mix hands back what it worked out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if plan is None:  # mixed as given: the same inputs, measured here
        cfg, infos, duck, correction = A.settings(scene.spec), O.measure(ins), None, None
        for a in ins:
            a["shift"], a["duck"] = 0.0, False
    else:
        ins, cfg, infos, duck, correction = plan["ins"], plan["cfg"], plan["info"], plan["duck"], plan["correction"]
    n = int(math.ceil(scene.duration / WIN))
    sounds = []
    for a, info in zip(ins, infos):
        g = _gain_db(a, duck, n, scene.fps)
        sounds.append({"id": a["id"], "role": a["role"], "src": a["src"], "in": a["at"],
                       "out": a["at"] + a["dur"] / a["speed"], "lufs": info["lufs"], "peak": info["peak"],
                       "lead": info["lead"], "gain": 20 * math.log10(max(a["gain"], 1e-6)), "shift": a["shift"],
                       "ducked": bool(a["duck"]), "trim0": a["trim0"], "onset": a["onset"], "info": info, "g": g,
                       "env": _place(a, info, n) + g})
    rep = {"managed": plan is not None, "sounds": sounds, "duration": scene.duration, "target": cfg["loudness"],
           "ceiling": cfg["peak"], "correction": correction, "master": master, "duck": duck,
           "depth": cfg["duck"]["depth"] if cfg["duck"] else 0.0, "ducking": bool(cfg["duck"])}
    if len(_CACHE) >= 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = rep
    return rep


def _place(a: dict, info: dict, n: int) -> np.ndarray:
    """A sound's RMS (dBFS per window, before gain) where it plays on the scene's time axis; -inf elsewhere."""
    out = np.full(n, -np.inf)
    rms = info["rms"][int(round(a["shift"] / WIN)):]  # measured from the trim before an effect was moved
    m = int(round(a["dur"] / a["speed"] / WIN))
    if not len(rms) or m <= 0:
        return out
    src = (np.arange(m) * a["speed"]).astype(int)
    if a["loop"]:
        src %= len(rms)
    idx = np.arange(m) + int(round(a["at"] / WIN))
    keep = (src < len(rms)) & (idx >= 0) & (idx < n)
    out[idx[keep]] = rms[src[keep]]
    return out


def _gain_db(a: dict, duck, n: int, fps: float) -> np.ndarray:
    """Everything the mix does to a sound's level, in dB per window on the scene's time axis: gain (with its role's
    level), fades, gain keyframes and the duck. -inf where the sound does not play."""
    out = np.full(n, -np.inf)
    length = a["dur"] / a["speed"]
    t = (np.arange(max(0, int(round(length / WIN)))) + 0.5) * WIN
    g = np.full(len(t), 20 * math.log10(max(a["gain"], 1e-6)))
    if a["fade_in"]:
        g += 20 * np.log10(np.clip(t / a["fade_in"], 1e-3, 1))
    if a["fade_out"]:
        g += 20 * np.log10(np.clip((length - t) / a["fade_out"], 1e-3, 1))
    if a["curve"] is not None:
        c = A.gain_curve(a["curve"], a["dur"], a["at"], fps)
        g += 20 * np.log10(np.maximum(c[np.minimum((t * A.RATE).astype(int), len(c) - 1)], 1e-6))
    idx = np.arange(len(t)) + int(round(a["at"] / WIN))
    if a["duck"] and duck is not None:
        g += 20 * np.log10(np.maximum(duck[np.minimum((idx * WIN * A.RATE).astype(int), len(duck) - 1)], 1e-6))
    keep = (idx >= 0) & (idx < n)
    out[idx[keep]] = g[keep]
    return out


def _loud(info: dict) -> float:
    """A source's loudness in LUFS; its mean power when it is too short for ebur128 (under 0.4 s)."""
    return info["lufs"] if info["lufs"] > -69.9 else info["power"]


def _sum(envs: list[np.ndarray]) -> np.ndarray:
    """Envelopes in dB added as power."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(sum(10 ** (e / 10) for e in envs))


def _speech(rms: np.ndarray) -> tuple[float, float] | None:
    """Seconds from the start of `rms` to the first and past the last window with sound in it."""
    loud = np.flatnonzero((rms > SILENT) & (rms > np.max(rms, initial=-np.inf) - 35))
    return (loud[0] * WIN, (loud[-1] + 1) * WIN) if len(loud) else None


def _runs(mask: np.ndarray) -> list[list[float]]:
    """[start, end] seconds of each stretch where mask is true."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [[round(a * WIN, 2), round(b * WIN, 2)] for a, b in zip(edges[::2], edges[1::2])]


def issues(scene: Scene, rep: dict | None) -> list[dict]:
    """check's audio rules. Each issue names the sound (or None for the whole mix), a time range and a fix."""
    if rep is None:
        return []
    out: list[dict] = []

    def add(level, code, sid, t0, t1, msg, fix):
        out.append({"level": level, "code": code, "layer": sid, "from": round(t0, 2), "to": round(t1, 2),
                    "message": msg, "fix": fix})

    D, by_id = rep["duration"], {s.id: s for s in scene.audio}
    voices = [s for s in rep["sounds"] if s["role"] == "voice"]
    for s in rep["sounds"]:
        info = s["info"]
        if not info["peak"] > SILENT:
            add("warning", "audio-silent", s["id"], s["in"], s["out"],
                f"'{Path(s['src']).name}' plays silence: nothing in it is louder than {SILENT:.0f} dBFS",
                "check src and trim")
        elif s["role"] == "sfx" and not s["onset"] and info["lead"] > LATE:
            add("warning", "sfx-late", s["id"], s["in"], s["in"] + info["lead"],
                f"its first sound comes {info['lead']:.2f}s after its cue",
                f"start its trim at {s['trim0'] + info['lead']:.2f}, or remove trim and loop so it starts at its "
                "first sound")
    spans = []
    for s in voices:
        snd = by_id.get(s["id"])
        sp = _speech(s["info"]["rms"])
        if sp:
            spans.append((s["in"] + sp[0], min(s["in"] + sp[1], s["out"]), s["id"]))
        if snd is None or snd.natural is None or s["in"] + snd.natural - s["out"] < 0.05:
            continue
        whole = _speech(A.analyze(s["src"], s["trim0"], s["trim0"] + snd.natural)["rms"])
        if whole and s["in"] + whole[1] > s["out"] + 0.02:
            by_scene = snd.t_out > D - 1e-3
            add("warning", "audio-cut", s["id"], s["out"], s["in"] + whole[1],
                f"the voice is cut at {s['out']:.2f}s; it speaks until {s['in'] + whole[1]:.2f}s",
                "end the scene later (\"duration\": \"auto\" fits every voice) or start the voice earlier" if by_scene
                else "lengthen its dur or until, or remove them so it plays to its end")
    for i, (a0, a1, ai) in enumerate(spans):
        for b0, b1, bi in spans[i + 1:]:
            if min(a1, b1) - max(a0, b0) > 0.05:
                first, second = (ai, bi) if a0 <= b0 else (bi, ai)
                add("warning", "voice-overlap", f"{ai}+{bi}", max(a0, b0), min(a1, b1), "two voices speak at once",
                    f"start '{second}' after '{first}' ends, for example \"at\": \"{first}.out+0.2\"")
    under = [s for s in rep["sounds"] if s["role"] in ("music", "ambience")]
    for v in voices:
        talk = v["env"] > max(A.SPEECH, np.max(v["env"], initial=-np.inf) - 30)
        if talk.sum() * WIN < 0.3:
            continue
        levels = [_loud(u["info"]) + float(np.median(u["g"][talk & np.isfinite(u["g"])])) for u in under
                  if (talk & np.isfinite(u["g"])).sum() * WIN >= 0.3]
        if not levels:
            continue
        gap = _loud(v["info"]) + float(np.median(v["g"][talk])) - 10 * math.log10(sum(10 ** (x / 10) for x in levels))
        if gap < GAP:
            idx = np.flatnonzero(talk)
            add("warning", "voice-masked", v["id"], idx[0] * WIN, (idx[-1] + 1) * WIN,
                (f"music and ambience are only {gap:.0f} LU under the voice" if gap >= 0.5 else
                 f"music and ambience are {-gap:.0f} LU louder than the voice" if gap <= -0.5 else
                 "music and ambience are as loud as the voice")
                + f" while it speaks; with less than {GAP:.0f} LU between them the words get hard to follow",
                "remove \"duck\": false from mix" if not rep["ducking"] else
                f"lower their gain, or duck deeper: \"mix\": {{\"duck\": {{\"depth\": {rep['depth'] + GAP - gap:.0f}}}}}")
    m = rep["master"]
    loudest = int(np.argmax(m["rms"])) * WIN if len(m["rms"]) else 0.0
    if rep["managed"]:
        if m["lufs"] > -69.9 and abs(m["lufs"] - rep["target"]) > 2:
            add("warning", "audio-loudness", None, 0, D, f"the mix is {m['lufs']:.1f} LUFS against a target of "
                f"{rep['target']:g}", "raise the gain of the quietest sounds, or set a target it can reach: "
                f"\"mix\": {{\"loudness\": {math.floor(m['lufs']):d}}}")
        if m["true_peak"] > rep["ceiling"] + 0.5:
            add("warning", "audio-peak", None, max(0.0, loudest - 0.1), min(D, loudest + 0.1),
                f"the mix peaks at {m['true_peak']:.1f} dBTP, over its {rep['ceiling']:g} dBTP ceiling",
                "lower the gain of the loudest sound there")
    else:
        if m["lufs"] > -69.9 and abs(m["lufs"] - A.LOUDNESS) > 6:
            add("info", "audio-loudness", None, 0, D, f"the mix is {m['lufs']:.1f} LUFS; video platforms play "
                "at about -14 to -16", "give the sounds a role (voice, music, sfx or ambience) and the mix is set to "
                "-16 LUFS")
        if m["peak"] > 20 * math.log10(0.95) - 0.15:  # at the ceiling of the mix's limiter (alimiter limit=0.95)
            add("warning", "audio-peak", None, max(0.0, loudest - 0.1), min(D, loudest + 0.1),
                f"the mix peaks at {m['peak']:.1f} dBFS, the limiter's ceiling, so its loudest parts are squashed",
                "lower the gain of the loudest sounds, or give the sounds roles to have them levelled")
    return out


def summary(rep: dict | None) -> dict | None:
    """The report as JSON: per sound, its loudness in and after gain; for the mix, loudness, peaks and range."""
    if rep is None:
        return None

    def r(v, k=1):
        return None if v is None or not math.isfinite(v) else round(float(v), k)
    m = rep["master"]
    return {"managed": rep["managed"],
            "sounds": [{"id": s["id"], "role": s["role"], "src": Path(s["src"]).name, "in": r(s["in"], 2),
                        "out": r(s["out"], 2), "lufs": r(s["lufs"]), "peak": r(s["peak"]), "gain": r(s["gain"]),
                        "lufs_out": r(s["lufs"] + s["gain"]) if s["lufs"] > -69.9 else None,
                        "peak_out": r(s["peak"] + s["gain"]), "lead": r(s["lead"], 3), "shift": r(s["shift"], 3),
                        "ducked": s["ducked"]} for s in rep["sounds"]],
            "mix": {"lufs": r(m["lufs"]), "true_peak": r(m["true_peak"]), "peak": r(m["peak"]), "lra": r(m["lra"]),
                    "short_max": r(m["short_max"]), "correction": r(rep["correction"]),
                    "target": rep["target"] if rep["managed"] else None, "ceiling": rep["ceiling"] if rep["managed"] else None},
            "duck": None if rep["duck"] is None else {"depth": rep["depth"], "spans": _runs(
                20 * np.log10(np.maximum(rep["duck"][::int(A.RATE * WIN)], 1e-6)) < -1)}}


def _lanes(rep: dict) -> list[tuple]:
    return [(role, color, [s for s in rep["sounds"] if s["role"] == role]) for role, color in LANES
            if any(s["role"] == role for s in rep["sounds"])]


def strip_height(rep: dict | None) -> int:
    return 0 if rep is None else AXIS_H + LANE_H * (len(_lanes(rep)) + 1) + 8


def strip(c, rep: dict, found: list[dict], x: float, y: float, w: float, ts: list[float], font, small):
    """Lanes at x, y, w wide: one per role and one for the mix, each an RMS envelope from -60 to 0 dBFS on the
    scene's time axis, with each sound's span and id, the duck, the times of the frames above, and issues:
    problems in red, suggestions outlined."""
    D = rep["duration"]
    x0, x1 = x + LABEL_W, x + w
    cols = max(1, int(x1 - x0))

    def px(t):
        return x0 + max(0.0, min(D, t)) / D * (x1 - x0)

    def paint(col, **kw):
        return skia.Paint(AntiAlias=True, Color=_argb(col), **kw)

    def columns(values: np.ndarray, rate: float, worst) -> np.ndarray:
        """One value per pixel column from samples at `rate` a second: worst (np.maximum or np.minimum) of the
        samples in the column, or the sample under it when the column is narrower than a sample."""
        if not len(values):
            return np.full(cols, -np.inf)
        starts = np.minimum((np.arange(cols) / cols * D * rate).astype(int), len(values) - 1)
        return worst.reduceat(values, starts)

    lanes = _lanes(rep)
    rows = [(role or "other", col, snds, _sum([s["env"] for s in snds]) + (rep["correction"] or 0))
            for role, col, snds in lanes]
    rows.append(("mix", "#f2f0e8", [], rep["master"]["rms"]))
    top = y + AXIS_H
    bottom = top + LANE_H * len(rows)
    step = next(s for s in (0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if D / s <= 14)
    for k in range(int(D / step) + 1):
        t = k * step
        c.drawLine(px(t), top - 4, px(t), top, paint("#6b7079"))
        if px(t) + small.measureText(f"{t:g}s") + 2 <= x1:
            c.drawString(f"{t:g}s", px(t) + 2, top - 7, small, paint("#9aa0a8"))
    for T in ts:  # the frames above, so a sound can be matched to what is on screen
        c.drawLine(px(T), top, px(T), bottom, paint("#ffffff26", StrokeWidth=1))
    for i, (name, col, snds, env) in enumerate(rows):
        ly = top + i * LANE_H
        c.drawRect(skia.Rect(x0, ly + 1, x1, ly + LANE_H - 1), paint("#14171d"))
        c.drawString(name, x + 4, ly + 17, font, paint(col))
        h = LANE_H - 8
        lvl = np.clip((columns(env, 1 / WIN, np.maximum) + 60) / 60, 0, 1) * h
        path = skia.Path()
        base = ly + LANE_H - 4
        path.moveTo(x0, base)
        for k in range(cols):
            path.lineTo(x0 + k, base - lvl[k])
        path.lineTo(x1, base)
        path.close()
        c.drawPath(path, paint(col + "b0"))
        free = [x0, x0]  # where the next id can start on each of two text rows, so sounds that start together both read
        for s in sorted(snds, key=lambda s: s["in"]):
            a, b = px(s["in"]), max(px(s["in"]) + 2, px(s["out"]))
            c.drawRect(skia.Rect(a, ly + 2, b, ly + 5), paint(col))
            tw = small.measureText(s["id"]) + 4
            row = next((r for r in (0, 1) if free[r] <= a), None)
            if tw < b - a and row is not None:
                c.drawString(s["id"], a + 2, ly + 15 + 11 * row, small, paint("#f2f0e8"))
                free[row] = a + tw
        if rep["duck"] is not None and any(s["ducked"] for s in snds):
            dc = columns(20 * np.log10(np.maximum(rep["duck"], 1e-6)), A.RATE, np.minimum)
            line = skia.Path()
            line.moveTo(x0, ly + 4 - dc[0] / 60 * h)
            for k in range(1, cols):
                line.lineTo(x0 + k, ly + 4 - dc[k] / 60 * h)
            c.drawPath(line, paint("#f2f0e8", Style=skia.Paint.kStroke_Style, StrokeWidth=1,
                                   PathEffect=skia.DashPathEffect.Make([4, 3], 0)))
            c.drawString(f"duck -{rep['depth']:g} dB", x + 4, ly + 30, small, paint("#9aa0a8"))
        if name == "mix":
            m = rep["master"]
            c.drawString(f"{m['lufs']:.1f} LUFS", x + 4, ly + 29, small, paint("#9aa0a8"))
            c.drawString(f"{m['true_peak']:.1f} dBTP", x + 4, ly + 40, small, paint("#9aa0a8"))
    lane_of = {s["id"]: i for i, (_, _, snds, _) in enumerate(rows) for s in snds}
    taken: dict[int, list[float]] = {}  # per lane, where the next code can start on each of three rows up from its foot
    for it in sorted(found, key=lambda it: it["from"]):
        i = lane_of.get(str(it["layer"]).split("+")[0], len(rows) - 1)
        ly = top + i * LANE_H
        a = px(it["from"])
        b = max(a + 3, px(it["to"]))
        note = it["level"] == "info"  # a suggestion is outlined; a problem is filled red
        box = skia.Rect(a, ly + 1, b, ly + LANE_H - 1)
        if note:
            c.drawRect(box.makeInset(0.5, 0.5), paint("#9aa0a8", Style=skia.Paint.kStroke_Style, StrokeWidth=1))
        else:
            c.drawRect(box, paint("#ff3b3b55"))
        free = taken.setdefault(i, [x0, x0, x0])
        row = next((r for r in range(3) if free[r] <= a), min(range(3), key=lambda r: free[r]))
        tx, ty = max(a, free[row]), ly + LANE_H - 13 * row
        tw = small.measureText(it["code"]) + 4
        free[row] = tx + tw + 2
        c.drawRect(skia.Rect(tx + 1, ty - 16, tx + tw + 1, ty - 3), paint("#0a0c10d0"))
        c.drawString(it["code"], tx + 3, ty - 6, small, paint("#c8ccd2" if note else "#ff6b6b"))
