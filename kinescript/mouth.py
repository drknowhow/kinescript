"""Mouth tracks: which mouth shape a talking avatar shows when, worked out once from a voice's word times.

A track is a list of [shape, start, end] in seconds of the sound, with no gaps from 0 to the last word's end; after
the end the mouth rests. The shapes are the nine Rhubarb Lip Sync uses, so mouth art drawn for that set fits:

    X rest          A closed: P B M        B teeth: most consonants, EE    C open: EH AE
    D wide: AA      E rounded: AO ER R     F puckered: OO OW W            G lip under teeth: F V    H tongue: L

`kine voice` writes a track next to each words file (voice/vo1.words.json -> voice/vo1.mouth.json). Its phonemes
come from the backend when it returns "phones" ([{"ipa" or "arpabet", "start", "end"}], one sound or a word's
sounds each), else from spelling
rules applied to each word and spread over the word's time. The rules are English and approximate; the track says
which source it used and which words the rules could not read.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

VERSION = 1
SHAPES = "XABCDEFGH"
REQUIRED = "XACD"  # an avatar draws these; the others fall back
FALLBACK = {"B": "C", "E": "D", "F": "D", "G": "A", "H": "C"}
CLOSE = 0.15  # a silence between words at least this long closes the mouth
MIN = 0.05  # shorter shapes are merged into a neighbour, except closures between sounds, kept at this length

# ARPAbet phonemes -> shapes. A diphthong is two shapes, the first held longer.
ARPA = {
    **dict.fromkeys(("P", "B", "M"), "A"),
    **dict.fromkeys(("T", "D", "K", "G", "N", "NG", "S", "Z", "SH", "ZH", "CH", "JH", "TH", "DH", "Y", "HH",
                     "IY", "IH"), "B"),
    **dict.fromkeys(("EH", "AE", "AH"), "C"),
    "AA": "D",
    **dict.fromkeys(("AO", "ER", "R"), "E"),
    **dict.fromkeys(("UW", "UH", "OW", "W"), "F"),
    **dict.fromkeys(("F", "V"), "G"),
    "L": "H",
    "EY": "CB", "AY": "DB", "AW": "DF", "OY": "EB",
}
# IPA (as misaki writes it, including its one-letter diphthongs A I O W Y) -> shapes.
IPA = {
    **dict.fromkeys("pbm", "A"),
    **dict.fromkeys("tdkgɡnŋszʃʒθðjhʔɾçxi", "B"), "ɪ": "B", "ᵻ": "B",
    **dict.fromkeys("ɛæəʌeɐ", "C"),
    **dict.fromkeys("ɑa", "D"),
    **dict.fromkeys("ɔɝɚɹr", "E"),
    **dict.fromkeys("uʊow", "F"),
    **dict.fromkeys("fv", "G"),
    "l": "H",
    "A": "CB", "I": "DB", "O": "EF", "W": "DF", "Y": "EB",
}
WEIGHT = {"A": 0.5, "B": 0.6, "G": 0.6, "H": 0.7, "C": 1.0, "D": 1.0, "E": 0.9, "F": 0.9}

# Spelling rules, longest first: letters -> ARPAbet. "_" marks the start of a word.
_RULES = sorted({
    "tch": "CH", "tion": "SH AH N", "sion": "ZH AH N", "igh": "AY", "_kn": "N", "_wr": "R", "_wh": "W",
    "ch": "CH", "sh": "SH", "th": "TH", "ph": "F", "ck": "K", "ng": "NG", "qu": "K W", "gh": "",
    "oo": "UW", "ee": "IY", "ea": "IY", "ai": "EY", "ay": "EY", "ey": "EY", "oa": "OW", "ow": "OW", "ou": "AW",
    "oi": "OY", "oy": "OY", "au": "AO", "aw": "AO", "ie": "IY", "ue": "UW", "ew": "UW",
    "er": "ER", "ir": "ER", "ur": "ER", "ar": "AA R", "or": "AO R",
    "a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH",
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G", "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T", "v": "V", "w": "W", "x": "K S", "z": "Z",
}.items(), key=lambda kv: -len(kv[0].lstrip("_")))
# Short words the rules read wrongly, and they are the most common ones.
_WORDS = {"the": "DH AH", "a": "AH", "i": "AY", "to": "T UW", "do": "D UW", "you": "Y UW", "of": "AH V",
          "one": "W AH N", "are": "AA R", "was": "W AH Z", "what": "W AH T", "said": "S EH D", "have": "HH AE V",
          "give": "G IH V", "is": "IH Z", "his": "HH IH Z", "as": "AE Z", "has": "HH AE Z", "be": "B IY",
          "he": "HH IY", "me": "M IY", "we": "W IY", "she": "SH IY", "no": "N OW", "go": "G OW", "so": "S OW",
          "who": "HH UW", "two": "T UW", "there": "DH EH R", "where": "W EH R", "come": "K AH M",
          "some": "S AH M", "done": "D AH N", "does": "D AH Z", "our": "AW ER", "your": "Y AO R"}
_LONG = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}
_VOWELS = set("aeiou")


def path_for(words_file) -> Path:
    """voice/vo1.words.json -> voice/vo1.mouth.json"""
    p = Path(words_file)
    name = p.name[: -len(".words.json")] if p.name.endswith(".words.json") else p.stem
    return p.with_name(name + ".mouth.json")


def spell(word: str) -> list[str] | None:
    """ARPAbet phonemes for an English word from its spelling, or None when it has no letters to read."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return None
    if w in _WORDS:
        return _WORDS[w].split()
    long_at = None  # a vowel, consonant, final e ("make", "note") makes the vowel long and the e silent
    if len(w) > 2 and w[-1] == "e" and w[-2] not in _VOWELS and w[-3] in _VOWELS:
        long_at, w = len(w) - 3, w[:-1]
    out, k = [], 0
    while k < len(w):
        if k == long_at:
            out.append(_LONG[w[k]])
            k += 1
            continue
        if w[k] == "y":  # a consonant at the start of a word, a vowel elsewhere
            out.append("Y" if k == 0 else "IY")
            k += 1
            continue
        if w[k] in "cg" and k + 1 < len(w) and w[k + 1] in "eiy":  # soft c and g ("sc" as in scene is one S)
            if not (w[k] == "c" and k and w[k - 1] == "s"):
                out.append("S" if w[k] == "c" else "JH")
            k += 1
            continue
        for letters, phones in _RULES:
            at_start = letters.startswith("_")
            letters = letters.lstrip("_")
            if (not at_start or k == 0) and w.startswith(letters, k):
                out += phones.split()
                k += len(letters)
                break
    return [ph for j, ph in enumerate(out) if not j or ph != out[j - 1]]  # a doubled letter is one sound


def _shapes(phone: dict) -> str:
    """The shapes a timed phone entry makes: {"ipa": "hɛlO"} (any number of sounds) or {"arpabet": "HH AH0"}
    (stress digits ignored); "" for sounds not known."""
    if "arpabet" in phone:
        return "".join(ARPA.get(re.sub(r"[0-9]", "", p).upper(), "") for p in str(phone["arpabet"]).split())
    return "".join(IPA.get(ch, "") for ch in re.sub(r"[ˈˌː]", "", str(phone.get("ipa", ""))))


def _spread(shapes: str, start: float, end: float) -> list[list]:
    """Shapes over [start, end], each taking time by its weight."""
    if not shapes:
        return []
    total = sum(WEIGHT.get(s, 0.8) for s in shapes)
    out, t = [], start
    for s in shapes:
        dt = (end - start) * WEIGHT.get(s, 0.8) / total
        out.append([s, t, t + dt])
        t += dt
    out[-1][2] = end
    return out


def _tidy(segs: list[list], end: float) -> list[list]:
    """Fill gaps (closing on long ones), merge repeats and slivers, and cover 0 to end."""
    out: list[list] = []
    t = 0.0
    for s, a, b in sorted(segs, key=lambda x: x[1]):
        a, b = max(a, t), max(b, t)
        if b <= a:
            continue
        if a - t >= CLOSE or not out:
            out.append(["X", t, a])
        elif a > t:
            out[-1][2] = a  # a short gap holds the shape before it
        out.append([s, a, b])
        t = b
    if not out:
        return [["X", 0.0, end]] if end > 0 else []
    out = [x for x in out if x[2] > x[1]]
    k = 0
    while k < len(out):
        s, a, b = out[k]
        if k and out[k - 1][0] == s:
            out[k - 1][2] = b
            del out[k]
            continue
        if b - a < MIN and len(out) > 1 and s != "X":  # a silence stays, however short
            need = MIN - (b - a)
            room = [j for j in (k + 1, k - 1) if 0 <= j < len(out) and out[j][2] - out[j][1] > MIN + need]
            if s == "A" and room:  # a closure (P B M) shows however short: it takes time from a neighbour
                j = room[0]
                if j > k:
                    out[j][1] += need
                    out[k][2] = out[j][1]
                else:
                    out[j][2] -= need
                    out[k][1] = out[j][2]
                k += 1
                continue
            before = k and out[k - 1][0] != "X"
            after = k < len(out) - 1 and out[k + 1][0] != "X"
            if before or after:  # into the sound beside it, not into a silence
                if before:
                    out[k - 1][2] = b
                else:
                    out[k + 1][1] = a
                del out[k]
                k = max(0, k - 1)
                continue
        k += 1
    return [[s, round(a, 4), round(b, 4)] for s, a, b in out]


def track(words: list[dict], phones: list[dict] | None = None) -> dict:
    """{"mouth": [[shape, start, end], ...], "source": "phones" | "rules", "unknown": [words]} from word times,
    and the backend's timed phonemes when it gave them."""
    end = max((w["end"] for w in words), default=0.0)
    if phones:
        segs = []
        for p in phones:
            segs += _spread(_shapes(p), float(p["start"]), float(p["end"]))
        end = max(end, max(float(p["end"]) for p in phones))
        return {"mouth": _tidy(segs, end), "source": "phones", "unknown": []}
    segs, unknown = [], []
    for w in words:
        ph = spell(w["word"])
        if ph is None:
            unknown.append(w["word"])
            ph = ["AH"]  # said somehow (a number, a symbol): an open mouth over its time
        segs += _spread("".join(ARPA.get(p, "") for p in ph), w["start"], w["end"])
    return {"mouth": _tidy(segs, end), "source": "rules", "unknown": unknown}


def levels(rms_db, win: float) -> dict:
    """A track from a sound's loudness alone (dBFS per window of `win` seconds), for recorded speech with no word
    times: quiet closes the mouth, louder opens it wider. It cannot tell one sound from another."""
    rms = [float(v) for v in rms_db]
    peak = max(rms, default=-120.0)
    n = max(1, math.ceil(MIN / win - 1e-9))  # windows per step, so no shape is shorter than MIN
    segs = []
    for k in range(0, len(rms), n):
        v = max(rms[k:k + n])
        s = "X" if v < peak - 35 else "A" if v < peak - 24 else "B" if v < peak - 16 else "C" if v < peak - 8 else "D"
        segs.append([s, k * win, min(k + n, len(rms)) * win])
    talk = [x for x in segs if x[0] != "X"]
    return {"mouth": _tidy(talk, len(rms) * win), "source": "levels", "unknown": []}


def key(words: list[dict], phones: list[dict] | None) -> str:
    """What a track was made from; a different key means it must be made again."""
    spec = {"v": VERSION, "words": [[w["word"], w["start"], w["end"]] for w in words], "phones": phones or None}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def stored(words_file) -> tuple[list[dict], list[dict] | None]:
    """The words and phones a words file holds."""
    from . import words as W
    raw = json.loads(Path(words_file).read_text(encoding="utf-8"))
    phones = raw.get("phones") if isinstance(raw, dict) else None
    return W.read(words_file), phones if isinstance(phones, list) and phones else None


def fresh(words_file) -> bool:
    """Whether the track next to a words file was made from what that file holds now."""
    try:
        made = json.loads(path_for(words_file).read_text(encoding="utf-8")).get("made") or {}
        return made.get("hash") == key(*stored(words_file))
    except (OSError, ValueError, AttributeError):
        return False


def write(words_file) -> dict:
    """Make the track for a words file and write it beside it; returns the track."""
    words, phones = stored(words_file)
    got = track(words, phones)
    rec = {"hash": key(words, phones), "version": VERSION, "source": got["source"]}
    body = {"mouth": got["mouth"], "made": rec, **({"unknown": got["unknown"]} if got["unknown"] else {})}
    path_for(words_file).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return got
