"""ElevenLabs voices for `kine voice`: ElevenLabs' API speaks a sound's text, Eleven v4 and v3 with audio tags such
as [whispers] or [laughs], and times each character it says; those times become the words.

It needs an API key in ELEVENLABS_API_KEY, and every line it speaks uses characters of that account's quota.
kine voice calls synth(req) through the "kinescript.voice" entry point; the package has no dependencies, so it
installs next to Kinescript in any Python it runs on.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
import uuid
import wave
from importlib.metadata import PackageNotFoundError, version

API = "https://api.elevenlabs.io"
MODEL = "eleven_v4"
SPEAKER = "JBFqnCBsd6RMkjVDRZzb"  # George, the voice in ElevenLabs' own examples
RATE = 24000  # pcm_24000: raw 16-bit mono, on every plan
TAG = re.compile(r"(?<!\[)\[(?!\[)[^\[\]()]+\](?![\](])")  # as kinescript.voice.TAG
SETTINGS = {"stability", "similarity_boost", "style", "use_speaker_boost"}  # voice_settings
BODY = {"seed", "language_code", "apply_text_normalization"}
SPEED = (0.7, 1.2)  # what voice_settings.speed takes


def acts_on_tags(model: str) -> bool:
    return model.startswith(("eleven_v3", "eleven_v4"))


def _key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
    if not key:
        raise RuntimeError("set ELEVENLABS_API_KEY to an ElevenLabs API key (elevenlabs.io > Developers > API keys)")
    return key


def _call(path: str, key: str, body: bytes | None = None, ctype: str = "application/json") -> dict:
    req = urllib.request.Request(API + path, data=body, method="POST" if body is not None else "GET",
                                 headers={"xi-api-key": key, "Content-Type": ctype, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"ElevenLabs {path.split('?')[0]} answered {e.code}: {detail}") from None


def voice_id(name: str, key: str) -> str:
    """A voice id as given, or the id of the account's voice with that name (\"George\")."""
    if re.fullmatch(r"[A-Za-z0-9]{20}", name):
        return name
    voices = _call("/v1/voices", key).get("voices") or []
    for v in voices:
        if v.get("name", "").split(" - ")[0].strip().lower() == name.strip().lower():
            return v["voice_id"]
    names = ", ".join(sorted(v.get("name", "").split(" - ")[0] for v in voices)[:30])
    raise ValueError(f"no ElevenLabs voice named '{name}' in this account; its voices: {names or 'none'}")


def words_from_chars(chars: list[str], starts: list[float], ends: list[float], offset: float = 0.0) -> list[dict]:
    """ElevenLabs' character times -> [{"word", "start", "end"}], split where the text has spaces, as the text
    reads. Tags are not words; a word's time runs from its first letter to its last, so punctuation and the pause
    after it are not stretched in."""
    text = "".join(chars)
    at, hidden = 0, [False] * len(chars)
    spans = [m.span() for m in TAG.finditer(text)]
    for k, c in enumerate(chars):
        hidden[k] = any(a <= at < b for a, b in spans)
        at += len(c)
    out, cur = [], None
    for c, s, e, h in zip(chars, starts, ends, hidden):
        if h or c.isspace():
            if cur and cur["word"]:
                out.append(cur)
            cur = None
            continue
        cur = cur or {"word": "", "start": None, "end": None}
        cur["word"] += c
        if re.search(r"\w", c):
            cur["start"] = s if cur["start"] is None else cur["start"]
            cur["end"] = e
    if cur and cur["word"]:
        out.append(cur)
    timed, last = [], offset
    for w in out:
        if not re.search(r"\w", w["word"]):  # punctuation on its own ("—"): nothing is said
            continue
        start, end = (w["start"] + offset, w["end"] + offset) if w["start"] is not None else (last, last)
        timed.append({"word": w["word"], "start": round(start, 4), "end": round(end, 4)})
        last = end
    return timed


def _wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm)


def _forced(path: str, text: str, key: str) -> list[dict]:
    """Words timed by ElevenLabs' forced alignment, for a model that returned no character times."""
    b = uuid.uuid4().hex
    with open(path, "rb") as f:
        audio = f.read()
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n{text}\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + audio + f"\r\n--{b}--\r\n".encode()
    got = _call("/v1/forced-alignment", key, body, f"multipart/form-data; boundary={b}")
    return [{"word": w["text"].strip(), "start": round(w["start"], 4), "end": round(w["end"], 4)}
            for w in got.get("words") or [] if re.search(r"\w", w.get("text", ""))]


def _version(model: str) -> str:
    try:
        own = version("kinescript-elevenlabs")
    except PackageNotFoundError:
        own = "?"
    return f"{model}, kinescript-elevenlabs {own}"


def synth(req: dict) -> dict:
    """Speak req["tagged"] (or req["text"] on a model without tags) with the voice req["speaker"] names (an id or
    a name in the account, George unless named) into the WAV req["out"], and time each word."""
    key = _key()
    opts = dict(req.get("options") or {})
    model = str(opts.pop("model", MODEL))
    tags = acts_on_tags(model)
    text = (req.get("tagged") or req["text"]) if tags else req["text"]
    speaker = req.get("speaker") or SPEAKER
    ignored = [f"options.{k}" for k in opts if k not in SETTINGS | BODY]
    settings = {k: opts[k] for k in SETTINGS if k in opts}
    rate = float(req.get("rate") or 1)
    if rate != 1:
        if SPEED[0] <= rate <= SPEED[1]:
            settings["speed"] = rate
        else:
            ignored.append(f"rate {req['rate']}: ElevenLabs speaks at {SPEED[0]} to {SPEED[1]}")
    body = {"text": text, "model_id": model, **{k: opts[k] for k in BODY if k in opts}}
    if settings:
        body["voice_settings"] = settings
    vid = voice_id(speaker, key)
    got = _call(f"/v1/text-to-speech/{vid}/with-timestamps?output_format=pcm_{RATE}", key, json.dumps(body).encode())
    pcm = base64.b64decode(got["audio_base64"])
    _wav(req["out"], pcm)
    timing = got.get("alignment") or {}
    words = words_from_chars(timing.get("characters") or [], timing.get("character_start_times_seconds") or [],
                             timing.get("character_end_times_seconds") or [])
    if not words or not any(w["end"] for w in words):
        words = _forced(req["out"], TAG.sub("", text), key)
    return {"words": words, "speaker": speaker, "version": _version(model), "seconds": round(len(pcm) / 2 / RATE, 3),
            "tags": tags, "ignored": ignored}
