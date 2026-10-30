"""Kokoro voices for `kine voice`: speak a sound's text to a WAV and time each word it says.

kine voice calls synth({"text", "speaker", "rate", "out"}) through the "kinescript.voice" entry point, or this
package runs as `python -m kinescript_kokoro` for KINE_VOICE_CMD, when Kokoro lives in another Python.
"""
from __future__ import annotations

import os
import re
import wave
from importlib.metadata import PackageNotFoundError, version

REPO = "hexgrad/Kokoro-82M"
SPEAKER = "af_heart"
RATE = 24000  # Kokoro's sample rate
_pipes: dict = {}  # a KPipeline per language, loaded once per process


def words_from_tokens(tokens, offset: float = 0.0) -> list[dict]:
    """Kokoro's tokens -> [{"word", "start", "end"}] split where the text has spaces, as the text reads.
    Punctuation joins its word ("scene" + "." -> "scene.") without stretching its time over the pause after it;
    a word with no time of its own (Kokoro gives none to some) takes the end of the word before it."""
    out, cur = [], None
    for tok in tokens:
        if cur is None:
            cur = {"word": "", "start": None, "end": None}
        cur["word"] += tok.text
        if re.search(r"\w", tok.text) and tok.start_ts is not None and tok.end_ts is not None:
            if cur["start"] is None:
                cur["start"] = tok.start_ts + offset
            cur["end"] = tok.end_ts + offset
        if tok.whitespace:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    timed, last = [], offset
    for w in out:
        if not re.search(r"\w", w["word"]):  # punctuation on its own ("—"): nothing is said
            continue
        if w["start"] is None:
            w["start"] = w["end"] = last
        timed.append({"word": w["word"].strip(), "start": round(w["start"], 4), "end": round(w["end"], 4)})
        last = w["end"]
    return timed


def _device() -> str:
    """cuda when this torch was built for the GPU (a GPU newer than the build fails at the first sound), else cpu.
    KOKORO_DEVICE picks one."""
    if os.environ.get("KOKORO_DEVICE"):
        return os.environ["KOKORO_DEVICE"]
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


def _pipe(lang: str):
    if lang not in _pipes:
        from kokoro import KPipeline
        _pipes[lang] = KPipeline(lang_code=lang, repo_id=REPO, device=_device())
    return _pipes[lang]


def _version() -> str:
    def v(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return "?"
    return f"kokoro {v('kokoro')}, kinescript-kokoro {v('kinescript-kokoro')}"


def synth(req: dict) -> dict:
    """Speak req["text"] with req["speaker"] (a Kokoro voice, af_heart unless named; its first letter is the
    language) at req["rate"] (1 = normal) into the WAV req["out"]. Word times come for English voices only."""
    import numpy as np
    speaker = req.get("speaker") or SPEAKER
    pipe = _pipe(speaker[0])
    chunks, words, at = [], [], 0
    for r in pipe(req["text"], voice=speaker, speed=float(req.get("rate") or 1)):
        if r.audio is None:
            continue
        audio = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)
        words += words_from_tokens(r.tokens or [], at / RATE)  # a chunk's times start at that chunk
        chunks.append(audio)
        at += len(audio)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    with wave.open(req["out"], "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return {"words": words, "speaker": speaker, "version": _version(), "seconds": round(at / RATE, 3)}
