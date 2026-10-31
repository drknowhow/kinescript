"""Word times for speech whose words are known: wav2vec2 gives each 20 ms of sound the odds of every letter, and a
CTC forced alignment finds the path through the text's letters that fits those odds best.

A number is aligned as it is read ("250" as TWO HUNDRED FIFTY, "2026" as TWENTY TWENTY SIX), so its words claim
their own sound. A tag ([laugh]), and a word with nothing left to align, become a wildcard that any sound can
fill, so a laugh does not stretch the words around it.
"""
from __future__ import annotations

import re

import numpy as np

MODEL = "facebook/wav2vec2-base-960h"
SR = 16000
HOP = 320  # input samples per output frame: 20 ms
TAG = re.compile(r"(?<!\[)\[(?!\[)[^\[\]()]+\](?![\](])")  # as kinescript.voice.TAG
WILD, MAYBE = -1, -2  # a wildcard that must take a frame (a word with no letters), and one that may take none (a tag)
PENALTY = 1.0  # what a wildcard frame costs (nats), so the text's own letters win wherever they fit
_model: dict = {}


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SYMBOLS = {"%": "percent", "&": "and", "+": "plus", "$": "dollars"}
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth"}


def letters(word: str) -> str:
    return re.sub(r"[^A-Z']", "", word.upper().replace("’", "'"))


def _say(n: int) -> list[str]:
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [ONES[n // 100], "hundred"] + (_say(n % 100) if n % 100 else [])
    size, name = next((s, w) for s, w in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")) if n >= s)
    return _say(n // size) + [name] + (_say(n % size) if n % size else [])


def number(digits: str) -> list[str]:
    """How a number is read: 250 two hundred fifty, 3.5 three point five, 1.2.0 one point two point zero, 2026 (a
    year) twenty twenty six."""
    whole, *rest = digits.replace(",", "").split(".")
    n = int(whole)
    if not rest and len(whole) == 4 and 1100 <= n <= 2099 and n % 100 and not 2000 <= n <= 2009:
        return _say(n // 100) + (["oh"] if n % 100 < 10 else []) + _say(n % 100)
    if len(rest) == 1:
        return _say(n) + ["point"] + [ONES[int(d)] for d in rest[0]]
    return _say(n) + [w for part in rest for w in ["point"] + _say(int(part))]


def ordinal(word: str) -> str:
    """first, twentieth, hundredth: the ordinal of a number word."""
    if word in ORDINALS:
        return ORDINALS[word]
    return word[:-1] + "ieth" if word.endswith("y") else word + "th"


def spoken(word: str) -> list[str]:
    """The words a written word is read as, in capitals: "250" -> [TWO, HUNDRED, FIFTY], "Render." -> [RENDER]."""
    if not re.search(r"\d", word):
        return [letters(word)]
    out = []
    for part in re.findall(r"\d+(?:[.,]\d+)*|[A-Za-z']+|[%&+$]", word):
        if part.lower() in ("st", "nd", "rd", "th") and out and out[-1] in ONES + TENS + ["hundred", "thousand"]:
            out[-1] = ordinal(out[-1])  # 21st: twenty first
        else:
            out += number(part) if part[0].isdigit() else [SYMBOLS.get(part, part)]
    if "$" in word:  # $5: five dollars
        out = [w for w in out if w != "dollars"] + ["dollars"]
    return [letters(w) for w in out]


def units(tagged: str, vocab: dict) -> tuple[list[str], list[tuple[int, int | None]]]:
    """The words of the text as written, and what is aligned: (vocabulary id, or WILD/MAYBE; the word's index, or
    None for a tag or a word gap)."""
    words, seq, gap = [], [], vocab["|"]
    for piece in re.split(f"({TAG.pattern})", tagged):
        if TAG.fullmatch(piece):
            seq.append((MAYBE, None))
            continue
        for w in piece.split():
            if not re.search(r"\w", w):  # punctuation on its own ("—"): nothing is said
                continue
            if seq and seq[-1][0] != MAYBE:
                seq.append((gap, None))
            ids = []
            for k, part in enumerate(spoken(w)):
                ids += ([gap] if k else []) + [vocab[c] for c in part if c in vocab]
            # nothing left to align: any sound
            seq += [(i, len(words)) for i in ids] if set(ids) - {gap} else [(WILD, len(words))]
            words.append(w)
    return words, seq


def ctc_path(logp: np.ndarray, seq: list[tuple[int, int | None]], blank: int) -> np.ndarray:
    """The best CTC path through seq for frames of log odds logp (frames x vocabulary): the index into seq that
    each frame says, or -1 for a blank."""
    T = logp.shape[0]
    labels = [blank]
    for lab, _ in seq:
        labels += [lab, blank]
    S = len(labels)
    speech = np.delete(logp, blank, axis=1).max(axis=1) - PENALTY  # a wildcard scores as the likeliest letter
    lab = np.array(labels)
    emit = np.where(lab[None, :] < 0, speech[:, None], logp[:, np.clip(lab, 0, None)])
    # a step of 2 skips the blank between two different letters; around a tag it skips the tag itself
    skip = np.zeros(S, bool)
    for s in range(2, S):
        if lab[s] != blank and lab[s] != lab[s - 2]:
            skip[s] = True
        if lab[s] == blank and lab[s - 1] == MAYBE:
            skip[s] = True
    neg = -np.inf
    score = np.full(S, neg)
    score[0], score[1] = emit[0, 0], emit[0, 1]
    if S > 3 and lab[1] == MAYBE:
        score[2] = emit[0, 2]
    back = np.zeros((T, S), np.int8)
    for t in range(1, T):
        one = np.concatenate([[neg], score[:-1]])
        two = np.where(skip, np.concatenate([[neg, neg], score[:-2]]), neg)
        stack = np.stack([score, one, two])
        back[t] = stack.argmax(axis=0)
        score = stack.max(axis=0) + emit[t]
    ends = [S - 1, S - 2] + ([S - 3, S - 4] if S > 4 and lab[S - 2] == MAYBE else [])
    s = max(ends, key=lambda e: score[e])
    if not np.isfinite(score[s]):
        raise ValueError("the sound is too short for its text")
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])  # int8 minus a Python int stays int8, which overflows past 127 states
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


def times(path: np.ndarray, seq: list, words: list[str], offset: float = 0.0) -> list[dict]:
    """[{"word", "start", "end"}] from the frames each word's letters took; a word that took none (the alignment
    could not place it) takes the end of the word before."""
    owner = np.array([seq[i][1] if i >= 0 and seq[i][1] is not None else -1 for i in path])
    out, last = [], offset
    for k, w in enumerate(words):
        frames = np.flatnonzero(owner == k)
        if len(frames):
            start, end = offset + frames[0] * HOP / SR, offset + (frames[-1] + 1) * HOP / SR
        else:
            start = end = last
        out.append({"word": w, "start": round(start, 3), "end": round(end, 3)})
        last = end
    return out


def _load(device: str):
    if device not in _model:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(device).eval()
        _model[device] = (model, Wav2Vec2CTCTokenizer.from_pretrained(MODEL).get_vocab())
    return _model[device]


def align(audio: np.ndarray, sr: int, tagged: str, device: str = "cpu", offset: float = 0.0) -> list[dict]:
    """The words of `tagged` (tags left out) with the times `audio` (mono floats at sr) says them."""
    import librosa
    import torch
    model, vocab = _load(device)
    words, seq = units(tagged, vocab)
    if not words:
        return []
    x = librosa.resample(np.asarray(audio, np.float32), orig_sr=sr, target_sr=SR) if sr != SR else audio
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x.astype(np.float32))[None].to(device)).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
    try:
        return times(ctc_path(logp, seq, vocab["<pad>"]), seq, words, offset)
    except ValueError:  # too short to hold every letter: no honest times
        return []
