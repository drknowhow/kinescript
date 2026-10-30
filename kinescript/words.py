"""Word times for a voice: what lets a layer start on a word ("vo1:render") and captions follow the voice.

A words file sits next to the sound, with .words.json in place of its extension (voice/vo1.wav ->
voice/vo1.words.json), or an audio entry names one with "words" (a path, or the list itself):

    {"words": [{"word": "Write", "start": 0.12, "end": 0.41}, {"word": "it.", "start": 0.45, "end": 0.6}]}

Times are seconds in the sound file. Whisper's JSON ({"segments": [{"words": [...]}]}) reads as it is.
A word's id is its own "id", or the word in lower case without punctuation, with _2, _3 for repeats:
"it." -> it, a second "it" -> it_2. Words are read once, when the scene loads, like a file's length.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WIDTH = 42  # characters on a caption line
LEAD = 0.1  # seconds a caption appears before its first word
HOLD = 0.5  # seconds it stays after its last word, unless the next one starts sooner
PAUSE = 0.6  # a silence this long between words starts a new caption


class WordsError(ValueError):
    pass


def default_path(src: str) -> Path:
    return Path(src).with_suffix(".words.json")


def slug(word: str) -> str:
    """'Don't' -> 'dont', 'check-in,' -> 'check_in': the form a cue can name."""
    return re.sub(r"\W+", "_", re.sub(r"['’]", "", word.lower())).strip("_")


def read(spec) -> list[dict]:
    """Words from a file path or a list, as [{"id", "word", "start", "end"}] in time order.
    Raises WordsError naming the first word that cannot be read."""
    if isinstance(spec, (str, Path)):
        p = Path(spec)
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            raise WordsError(f"words file not found: {p}") from None
        except (OSError, json.JSONDecodeError) as e:
            raise WordsError(f"cannot read the words file {p}: {e}") from None
    else:
        data = spec
    if isinstance(data, dict):
        data = data.get("words", [w for seg in data.get("segments") or [] for w in seg.get("words") or []])
    if not isinstance(data, list):
        raise WordsError('words must be a list like [{"word": "Hello", "start": 0.1, "end": 0.4}]')
    out = []
    for k, w in enumerate(data):
        text = w.get("word", w.get("text")) if isinstance(w, dict) else None
        s, e = (w.get("start"), w.get("end")) if isinstance(w, dict) else (None, None)
        if not isinstance(text, str) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (s, e)):
            raise WordsError(f"word #{k} is {w!r}; each word is {{\"word\": \"Hello\", \"start\": 0.1, \"end\": 0.4}} "
                             "in seconds of the sound file")
        if e < s:
            raise WordsError(f"word #{k} '{text.strip()}' ends at {e:g}s, before it starts at {s:g}s")
        wid = w.get("id")
        if wid is not None and not re.fullmatch(r"\w+", str(wid)):
            raise WordsError(f"word #{k} has id {wid!r}; an id is letters, digits and _ only")
        out.append({"id": None if wid is None else str(wid), "word": text.strip(), "start": float(s), "end": float(e)})
    out.sort(key=lambda w: w["start"])
    taken = {w["id"] for w in out if w["id"]}
    if len(taken) < sum(1 for w in out if w["id"]):
        raise WordsError("two words have the same id; every id must be different")
    count: dict[str, int] = {}
    for w in out:
        b = slug(w["word"])
        if w["id"] is not None or not b:
            continue
        n = count.get(b, 0) + 1
        wid = b if n == 1 else f"{b}_{n}"
        while wid in taken:  # an explicit id already has this name
            n += 1
            wid = f"{b}_{n}"
        count[b] = n
        w["id"] = wid
        taken.add(wid)
    return out


def cues(words: list[dict], trim0: float = 0.0, trim1: float | None = None) -> list[list]:
    """Caption cues [start, end, text, [[start, end] per word]] in seconds from when the sound starts playing.
    A line ends at a sentence end, at a comma once it is half full, before a pause, or before WIDTH characters."""
    kept = [(w["word"], w["start"] - trim0, w["end"] - trim0) for w in words
            if w["word"] and w["start"] >= trim0 and (trim1 is None or w["start"] < trim1)]
    limit = None if trim1 is None else trim1 - trim0
    lines: list[list] = []
    cur: list = []
    for word, s, e in kept:
        if cur and (len(" ".join(x[0] for x in cur)) + 1 + len(word) > WIDTH or s - cur[-1][2] >= PAUSE):
            lines.append(cur)
            cur = []
        cur.append((word, s, e if limit is None else min(e, limit)))
        n = len(" ".join(x[0] for x in cur))
        if word[-1] in ".!?;:" or (word[-1] == "," and n >= WIDTH // 2):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out: list[list] = []
    for i, line in enumerate(lines):
        start = max(0.0, line[0][1] - LEAD, out[-1][1] if out else 0.0)
        nxt = lines[i + 1][0][1] - LEAD if i + 1 < len(lines) else None
        end = line[-1][2] + HOLD if nxt is None else min(nxt, line[-1][2] + HOLD)
        if limit is not None:
            end = min(end, limit)
        out.append([round(start, 4), round(max(end, start + 0.05), 4), " ".join(x[0] for x in line),
                    [[round(s, 4), round(e, 4)] for _, s, e in line]])
    return out
