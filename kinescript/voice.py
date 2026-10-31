"""`kine voice`: speak each sound's `text` into its file and time its words. Rendering never does this.

A sound with text is spoken by a backend:

    {"id": "vo1", "role": "voice", "text": "Write a scene. [[check]]Check it.", "speaker": "af_heart"}

writes voice/vo1.wav (or its src) and voice/vo1.words.json. [[name]] gives the next word that id, so a cue can
say "vo1:check" whatever the word is. A tag such as [laughs] or [whispers] tells a backend that acts on tags how
to say what follows; it is not a word, and other backends never see it. A sound is spoken again only when its text,
speaker, rate, backend or options changed since its file was made (or with force); `kine check` warns when they
changed and the file was not remade.

A backend is a function registered under the entry-point group "kinescript.voice" (kinescript-kokoro is one) that
takes {"text", "speaker", "rate", "out"}, writes a WAV to out, and returns {"words": [{"word", "start", "end"}],
"speaker", "version", "seconds"} (seconds, the file's length, is optional). The request also carries "tagged" (the
text with its tags), "options" (the sound's options, as given) and "base" (the scene's folder, for files the options
or speaker name). A backend that acts on tags returns "tags": true, anything it could not do in "ignored", and
words it could not time in "note".
A sound picks its backend with "backend"; KINE_VOICE_CMD_<NAME> runs that backend in another Python, and
KINE_VOICE_CMD runs the default one (or the backend whose module it runs, python -m kinescript_<name>). A command
reads a JSON list of those requests on stdin and prints the list of results. Scene text never reaches a shell.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from . import words as W

GROUP = "kinescript.voice"
KOKORO = "kinescript-kokoro"
MARK = re.compile(r"\[\[\s*(\w+)\s*\]\]")
# [laughs], [clears throat]: in single brackets, and neither a mark ([[name]]) nor a pronunciation ([word](/ph/)).
TAG = re.compile(r"(?<!\[)\[(?!\[)[^\[\]()]+\](?![\](])")


def default_src(sid: str) -> str:
    return f"voice/{sid}.wav"


def fingerprint(a: dict) -> str:
    """What the file was made from: text, speaker, rate, and the backend and options when the sound names them.
    A change in any means it must be spoken again."""
    spec = {"text": a.get("text"), "speaker": a.get("speaker"), "rate": a.get("rate", 1)}
    spec.update({k: a[k] for k in ("backend", "options") if a.get(k)})  # absent: the same hash as before they existed
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def tags(text: str) -> list[str]:
    return TAG.findall(text)


def untag(text: str) -> str:
    """The text without its tags: what a backend that does not act on tags says, and the words marks count."""
    return re.sub(r"\s+", " ", TAG.sub("", text)).strip()


def split_marks(text: str) -> tuple[str, dict[int, str]]:
    """'Go [[x]]now' -> ('Go now', {1: 'x'}): the text to speak, and the word index each mark names. Tags stay in
    the text and are not counted as words."""
    out, marks = [], {}
    for k, part in enumerate(MARK.split(text)):
        if k % 2:
            n = len(untag(" ".join(out)).split())
            if n in marks:
                raise ValueError(f"two marks, [[{marks[n]}]] and [[{part}]], name the same word")
            marks[n] = part
        else:
            out.append(part)
    names = list(marks.values())
    if len(set(names)) < len(names):
        raise ValueError(f"the mark [[{next(m for m in names if names.count(m) > 1)}]] is used twice")
    return re.sub(r"\s+", " ", " ".join(out)).strip(), marks


def place_marks(text: str, marks: dict[int, str], said: list[dict]) -> list[dict]:
    """Give each marked word of `text` its id in `said` (the words the backend timed), matching words in order,
    since a backend may split or join them (punctuation, numbers read out)."""
    if not marks:
        return said
    tokens = [W.slug(t) for t in text.split()]
    heard = [W.slug(w["word"]) for w in said]
    at, j = [], 0
    for t in tokens:
        k = next((i for i in range(j, min(len(heard), j + 8)) if heard[i] == t and t), None)
        at.append(k)
        if k is not None:
            j = k + 1
    out = [dict(w) for w in said]
    for n, name in marks.items():
        k = at[n] if n < len(at) else None
        if k is None and said:  # read differently than written: the word at the same place in the line
            k = min(len(said) - 1, round(n * len(said) / max(1, len(tokens))))
        if k is not None:
            out[k]["id"] = name
    return out


def _spec(scene, base) -> tuple[dict, Path]:
    """The scene with its aliases read (a scene that does not load yet, because its voices are missing, still reads)."""
    from . import schema
    if isinstance(scene, dict):
        spec, base = scene, Path(base or ".")
    else:
        p = Path(scene)
        if not p.exists():
            raise FileNotFoundError(f"scene file not found: {p}")
        spec, _ = schema.loads(p.read_text(encoding="utf-8-sig"))
        base = p.parent
    return schema.normalize(spec)[0], base


def voices(spec: dict) -> list[tuple[str, dict]]:
    """The audio entries with text, with their ids (as the scene gives them)."""
    return [(str(a.get("id") or f"audio{i + 1}"), a) for i, a in enumerate(spec.get("audio") or [])
            if isinstance(a, dict) and isinstance(a.get("text"), str) and a["text"].strip()]


def paths(sid: str, a: dict, base: Path) -> tuple[Path, Path]:
    """Where a sound with text is spoken to, and where its words go."""
    src = Path(a.get("src") or default_src(sid))
    src = src if src.is_absolute() else base / src
    words = a.get("words")
    return src, (base / words if isinstance(words, str) and not Path(words).is_absolute()
                 else Path(words) if isinstance(words, str) else W.default_path(str(src)))


def made(words_file: Path) -> dict | None:
    """What a words file says its sound was made from, or None."""
    try:
        rec = json.loads(Path(words_file).read_text(encoding="utf-8")).get("made")
    except (OSError, ValueError, AttributeError):
        return None
    return rec if isinstance(rec, dict) else None


def backends() -> dict:
    from importlib.metadata import entry_points
    return {e.name: e for e in entry_points(group=GROUP)}


def _argv(cmd: str) -> list[str]:
    if cmd.startswith("["):
        return [str(x) for x in json.loads(cmd)]
    return [t.strip('"') for t in shlex.split(cmd, posix=os.name != "nt")]


def _env_name(name: str) -> str:
    return "KINE_VOICE_CMD_" + re.sub(r"\W", "_", name).upper()


def _command(name: str | None = None) -> tuple[list[str], str] | None:
    """The command that runs backend `name` in another Python, and the variable it came from: KINE_VOICE_CMD_<NAME>,
    else KINE_VOICE_CMD when no backend is named or it runs that backend's module (-m kinescript_<name>)."""
    own = os.environ.get(_env_name(name), "").strip() if name else ""
    if own:
        return _argv(own), _env_name(name)
    cmd = os.environ.get("KINE_VOICE_CMD", "").strip()
    if cmd and (not name or f"kinescript_{name}" in _argv(cmd)):
        return _argv(cmd), "KINE_VOICE_CMD"
    return None


def _speaker(name: str | None):
    """The function that speaks a list of requests, and its name."""
    found = backends()
    cmd = _command(name)
    if cmd and (not name or name not in found or cmd[1] != "KINE_VOICE_CMD"):
        argv, var = cmd

        def run(reqs):
            res = subprocess.run(argv, input=json.dumps(reqs), capture_output=True, text=True, encoding="utf-8")
            if res.returncode:
                raise RuntimeError(f"{var} failed ({res.returncode}): {res.stderr.strip()[-600:]}")
            return json.loads(res.stdout)
        return run, name or Path(argv[-1]).name
    if name:
        if name not in found:
            raise ValueError(f"no voice backend '{name}': pip install kinescript-{name}, or set {_env_name(name)} to a "
                             f"command that runs it in another Python; installed: {', '.join(found) or 'none'}")
        key = name
    elif len(found) == 1:
        key = next(iter(found))
    elif found:
        raise ValueError(f"several voice backends are installed ({', '.join(found)}); pick one with --backend")
    else:
        raise ValueError(f"no voice backend is installed: pip install {KOKORO} (Python 3.10 to 3.12), "
                         "or set KINE_VOICE_CMD to a command that speaks")
    fn = found[key].load()
    return (lambda reqs: [fn(r) for r in reqs]), key


def generate(scene, only=None, force: bool = False, backend: str | None = None, base=None) -> dict:
    """Speak every sound with text whose file is missing or out of date; return what was made and what was fresh."""
    spec, base = _spec(scene, base)
    todo, out = [], []
    wanted = set(only) if only else None
    entries = voices(spec)
    if wanted and wanted - {sid for sid, _ in entries}:
        raise ValueError(f"no sound with text has id {', '.join(sorted(wanted - {s for s, _ in entries}))}; "
                         f"sounds with text: {', '.join(s for s, _ in entries) or 'none'}")
    for sid, a in entries:
        if wanted and sid not in wanted:
            continue
        if isinstance(a.get("words"), list):
            raise ValueError(f"'{sid}' has text and a words list; remove the list, kine voice writes its words file")
        src, words_file = paths(sid, a, base)
        text, marks = split_marks(a["text"])
        rec = made(words_file)
        if not force and src.exists() and rec and rec.get("hash") == fingerprint(a):
            out.append({"id": sid, "src": str(a.get("src") or default_src(sid)), "status": "fresh"})
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        todo.append((sid, a, src, words_file, text, marks))
    groups: dict = {}  # a sound's own backend, else the one asked for, else the default
    for job in todo:
        groups.setdefault(job[1].get("backend") or backend, []).append(job)
    for want, jobs in groups.items():
        speak, name = _speaker(want)
        t0 = time.perf_counter()
        results = speak([{"text": untag(text), "tagged": text, "speaker": a.get("speaker"), "rate": a.get("rate", 1),
                          "options": a.get("options") or {}, "out": str(src), "base": str(Path(base).resolve())}
                         for _, a, src, _, text, _ in jobs])
        took = (time.perf_counter() - t0) / len(jobs)
        for (sid, a, src, words_file, text, marks), res in zip(jobs, results):
            if not src.exists():
                raise RuntimeError(f"the voice backend {name} did not write {src} for '{sid}'")
            timed = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in W.read(res.get("words") or [])]
            said = place_marks(untag(text), marks, timed)
            W.read(said)  # the ids are valid and unique before the file is written
            rec = {"hash": fingerprint(a), "backend": name, "version": res.get("version"),
                   "speaker": res.get("speaker") or a.get("speaker")}
            words_file.write_text(json.dumps({"words": said, "made": rec}, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
            secs = res.get("seconds") or (said[-1]["end"] if said else 0.0)
            item = {"id": sid, "src": str(a.get("src") or default_src(sid)), "status": "made", "words": len(said),
                    "seconds": round(float(secs), 2), "took": round(took, 2)}
            ignored = list(res.get("ignored") or []) + ([] if res.get("tags") else tags(text))
            if ignored:
                item["ignored"] = ignored  # said without these: the backend cannot do them
            if res.get("note"):
                item["note"] = str(res["note"])
            elif not said and untag(text).strip():
                item["note"] = f"{name} gave no word times, so word cues and voice captions of '{sid}' do not work"
            out.append(item)
    order = {sid: k for k, (sid, _) in enumerate(entries)}
    return {"ok": True, "voices": sorted(out, key=lambda v: order[v["id"]])}
