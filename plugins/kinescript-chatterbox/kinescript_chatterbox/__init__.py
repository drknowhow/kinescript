"""Chatterbox voices for `kine voice`: Resemble AI's open models speak a sound's text, Turbo with tags such as
[laugh], and a wav2vec2 alignment times each word (Chatterbox gives no word times of its own).

kine voice calls synth(req) through the "kinescript.voice" entry point, or this package runs as
`python -m kinescript_chatterbox` for KINE_VOICE_CMD_CHATTERBOX, when Chatterbox lives in another Python.
"""
from __future__ import annotations

import os
import re
import wave
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .align import TAG, align

MODELS = ("turbo", "standard")  # turbo acts on tags; standard takes exaggeration and cfg_weight instead
TURBO_TAGS: set[str] = set()  # filled from Turbo's tokenizer when it loads
OPTIONS = {"turbo": {"temperature", "top_p", "top_k", "repetition_penalty"},
           "standard": {"exaggeration", "cfg_weight", "temperature", "min_p", "top_p", "repetition_penalty"}}
CHUNK = 300  # characters per generate call: Chatterbox reads long text better a few sentences at a time
_models: dict = {}


def _device() -> str:
    """cuda when this torch was built for the GPU, else cpu. CHATTERBOX_DEVICE picks one."""
    if os.environ.get("CHATTERBOX_DEVICE"):
        return os.environ["CHATTERBOX_DEVICE"]
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


def _model(kind: str):
    """The model and its own voice, loaded once per process."""
    if kind not in _models:
        if kind == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model
        m = Model.from_pretrained(device=_device())
        if kind == "turbo":
            TURBO_TAGS.update(t for t in m.tokenizer.get_added_vocab() if TAG.fullmatch(t))
            # chatterbox-tts 0.1.7 scales the clip by a NumPy float64, which makes float64 under NumPy 2 (Python
            # 3.13 and later), and its tokenizer then fails on a clip to copy; keep the clip float32
            norm = m.norm_loudness
            m.norm_loudness = lambda wav, sr, **kw: norm(wav, sr, **kw).astype("float32", copy=False)
        _models[kind] = (m, m.conds)
    return _models[kind]


def _forms(word: str) -> set[str]:
    return {word} | {word[:-len(s)] for s in ("ing", "es", "s") if word.endswith(s) and len(word) - len(s) >= 3}


def turbo_tag(tag: str) -> str | None:
    """The Turbo tag for one written as another backend writes it ([laughs] -> [laugh], [whispers] ->
    [whispering], [clears throat] -> [clear throat]), or None."""
    t = tag.lower()
    if t in TURBO_TAGS:
        return t
    words = t[1:-1].split()
    for known in sorted(TURBO_TAGS):
        kw = known[1:-1].split()
        if len(kw) == len(words) and all(_forms(a) & _forms(b) for a, b in zip(words, kw)):
            return known
    return None


def chunks(text: str, size: int = CHUNK) -> list[str]:
    """The text in pieces of whole sentences, each at most about `size` characters."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if out and len(out[-1]) + 1 + len(sentence) <= size:
            out[-1] += " " + sentence
        elif sentence:
            out.append(sentence)
    return out


def _version(kind: str) -> str:
    def v(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return "?"
    return f"chatterbox-tts {v('chatterbox-tts')} {kind}, kinescript-chatterbox {v('kinescript-chatterbox')}"


def synth(req: dict) -> dict:
    """Speak req["tagged"] (Turbo) or req["text"] (standard) into the WAV req["out"], in the voice of the clip
    req["speaker"] names (5 s or more, relative to req["base"]) or the model's own voice, and time each word."""
    import numpy as np
    import torch
    opts = dict(req.get("options") or {})
    kind = str(opts.pop("model", "turbo"))
    if kind not in MODELS:
        raise ValueError(f"Chatterbox model '{kind}': use one of {', '.join(MODELS)}")
    seed = int(opts.pop("seed", 0))
    m, own = _model(kind)
    ignored = [f"options.{k}" for k in opts if k not in OPTIONS[kind]]
    if float(req.get("rate") or 1) != 1:
        ignored.append(f"rate {req['rate']}: Chatterbox speaks at one speed")
    text = (req.get("tagged") or req["text"]) if kind == "turbo" else req["text"]
    if kind == "turbo":  # a tag Turbo does not know would be read out as a word
        ignored += [t for t in TAG.findall(text) if not turbo_tag(t)]
        text = re.sub(r"\s+", " ", TAG.sub(lambda t: turbo_tag(t.group(0)) or "", text)).strip()
    speaker = req.get("speaker")
    if speaker:
        ref = Path(speaker)
        ref = ref if ref.is_absolute() else Path(req.get("base") or ".") / ref
        if not ref.exists():
            raise FileNotFoundError(f"Chatterbox clones the voice of a clip of 5 s or more; {ref} does not exist")
        m.prepare_conditionals(str(ref), **({"exaggeration": opts["exaggeration"]} if "exaggeration" in opts else {}))
    else:
        m.conds = own
    kw = {k: v for k, v in opts.items() if k in OPTIONS[kind]}
    audio, words, untimed, at = [], [], [], 0
    for k, piece in enumerate(chunks(text)):
        torch.manual_seed(seed + k)  # the same text says the same thing on the next run
        wav = m.generate(piece, **kw).squeeze(0).numpy().astype(np.float32)
        got = align(wav, m.sr, piece, device=_device(), offset=at / m.sr)
        said = " ".join(TAG.sub(" ", piece).split())
        if not got and said:  # the sound is too short to hold the text: it was cut off
            untimed.append(said)
        words += got
        audio.append(wav)
        at += len(wav)
    out = np.concatenate(audio) if audio else np.zeros(0, np.float32)
    with wave.open(req["out"], "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(m.sr)
        f.writeframes((np.clip(out, -1, 1) * 32767).astype("<i2").tobytes())
    got = {"words": words, "speaker": speaker or f"{kind} built-in", "version": _version(kind),
           "seconds": round(at / m.sr, 3), "tags": kind == "turbo", "ignored": ignored}
    if untimed:
        got["note"] = ("too short for its text, so these words have no times (try another seed): "
                       + " / ".join(untimed))
    return got
