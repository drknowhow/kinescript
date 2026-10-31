"""Cue-graph timing: start/end times written as references to other layers.

    "at": 1.5               seconds
    "at": "45f"             frames
    "at": "0:02.5"          m:ss
    "at": "title.out+0.2"   0.2 s after layer 'title' ends
    "at": "prev.in+10f"     10 frames after the previous layer starts
    "at": "drop"            a named marker from scene "markers"
    "at": "vo1:render.out"  when sound 'vo1' finishes saying "render" (from its words file)

Change one duration and everything cued from it moves with it.
"""
from __future__ import annotations

import math
import re

_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"  # 2, 2.5, .5, 1e3
_TERM = re.compile(rf"\s*([+-])\s*({_NUM})\s*(s|f|ms)?\s*")
# A cue name: letters (any script), digits, _ and -, as in "1st", "título" or "vo1:render".
_BASE = re.compile(r"^\s*([^\W][\w-]*(?::\w+)?)(?:\.(in|out|mid))?")
# Times past this many seconds (11.6 days) are typos: their frame counts lose precision, and 10**400 is no float.
LIMIT = 1e6
# Words a cue reads before any id, so a layer or audio id cannot be one of them. ("end" is special only as the
# whole of "until": "end"; a layer called end is still cued as "end.out".)
KEYWORDS = ("start", "prev")


def finite(v) -> bool:
    """A number of seconds Kinescript can use: finite and within LIMIT (10**400 is too large for a float)."""
    return abs(v) <= LIMIT and math.isfinite(v)


def show(v) -> str:
    """A value for a message, short even for 10**400 or 1e308."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return repr(v)
    try:
        f = float(v)
    except OverflowError:
        return f"a number with {len(str(abs(v)))} digits"
    return f"{f:.6g}" if math.isfinite(f) else str(f)


def secs(t: float) -> str:
    """A time for a message: '2.500s', or '1e+308 s' rather than 309 digits."""
    return f"{t:.3f}s" if abs(t) < LIMIT else f"{t:.4g} s"


class TimingError(ValueError):
    def __init__(self, message: str, layer: str | None = None):
        super().__init__(message)
        self.layer = layer


FORMS = ('a number of seconds (1.5), "45f" (frames), "0:02.5", "title.out+0.2" (0.2 s after layer title ends), '
         '"title.in", "title.mid", "prev.out", a marker name, or "vo1:hello" (when sound vo1 says hello)')


def parse_clock(s: str, fps: float) -> float | None:
    s = s.strip()
    m = re.fullmatch(rf"(-?{_NUM})\s*(s|f|ms)?", s)
    if m:
        v = float(m.group(1))
        return v / fps if m.group(2) == "f" else v / 1000 if m.group(2) == "ms" else v
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def split_ref(expr: str) -> tuple[str | None, str, str]:
    """'title.out+0.2' -> ('title', 'out', '+0.2'); 'vo1:go.out' -> ('vo1:go', 'out', '').
    Plain clocks return (None, '', expr)."""
    m = _BASE.match(expr)
    if not m or parse_clock(expr, 30) is not None:
        return None, "", expr
    return m.group(1), m.group(2) or "", expr[m.end():]


def offset(rest: str, fps: float) -> float:
    total, pos = 0.0, 0
    while pos < len(rest):
        m = _TERM.match(rest, pos)
        if not m or m.end() == pos:
            raise TimingError(f"cannot parse time offset {rest!r}")
        v = float(m.group(2))
        v = v / fps if m.group(3) == "f" else v / 1000 if m.group(3) == "ms" else v
        total += v if m.group(1) == "+" else -v
        pos = m.end()
    return total


class Timeline:
    """Resolves in/out for every layer, following references depth-first."""

    def __init__(self, fps: float, markers: dict[str, float] | None = None, words: dict[str, dict] | None = None):
        """`words`: per sound id, {"ids": {word id: (start, end)} in file seconds, or None when it has no words file,
        "trim0" and "trim1" (the part of the file that plays) and "file" (where a words file would go)."""
        self.fps = fps
        self.markers = {k: self._marker(k, v) for k, v in (markers or {}).items()}
        self.words = words or {}
        self.nodes: dict[str, dict] = {}
        self.order: list[str] = []
        self.resolved: dict[str, tuple[float, float | None]] = {}
        self._starts: dict[str, float] = {}
        self._stack: list[str] = []

    def add(self, lid: str, at, dur=None, until=None, natural: float | None = None,
            prev: str | None = None, parent: str | None = None, default_end: str | None = None,
            span: bool = False, counts: bool = True):
        """`span`: with no dur or until, end when the last child ends (a sequence's group).
        `counts`: its end can set an "auto" scene duration (a music bed longer than the video does not)."""
        if lid in self.nodes:
            raise TimingError(f"duplicate id '{lid}'; every layer and audio id must be unique", lid)
        self.nodes[lid] = dict(at=at, dur=dur, until=until, natural=natural, prev=prev,
                               parent=parent, default_end=default_end, span=span, counts=counts)
        self.order.append(lid)

    def _marker(self, name: str, v) -> float:
        t = parse_clock(v, self.fps) if isinstance(v, str) else v
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not finite(t):
            raise TimingError(f"marker '{name}' is {show(v)}; a marker is a time: seconds like 2.5, \"45f\" or "
                              "\"0:02.5\"")
        return float(t)

    def _time(self, v, owner: str, base: float) -> float:
        if v is None:
            return base
        if isinstance(v, bool) or (not isinstance(v, (int, float, str))):
            raise TimingError(f"cannot read the time {v!r}. Write {FORMS}", owner)
        if isinstance(v, (int, float)):
            if not finite(v):
                raise TimingError(f"the time {show(v)} is not a finite number of seconds up to {LIMIT:g}", owner)
            return base + float(v)
        s = str(v)
        clk = parse_clock(s, self.fps)
        if clk is not None:
            return base + clk
        name, edge, rest = split_ref(s)
        if name is None:
            self._uncueable(s, owner)
            raise TimingError(f"cannot read the time {s!r}. Write {FORMS}", owner)
        name, _, word = name.partition(":")
        if name == "prev":
            name = self.nodes[owner]["prev"] if owner in self.nodes else None
            if name is None:
                raise TimingError("'prev' used on the first layer, which has no previous layer", owner)
        if word:
            t = self._word(name, word, edge, owner)
        elif name in self.markers and not edge:
            t = self.markers[name]
        elif name in self.markers and name not in self.nodes:
            raise TimingError(f"'{name}' in {s!r} is a marker, one moment in time, so it has no .{edge}. Write "
                              f"\"{name}\" or \"{name}+0.5\"", owner)
        elif name == "start":
            t = 0.0
        elif name in self.nodes:
            if edge in ("", "in"):
                t = self.start(name)
            else:
                t_in, t_out = self.resolve(name)
                if t_out is None:
                    raise TimingError(f"refers to '{name}.{edge}', but '{name}' runs to the scene end, so it has "
                                      f"no end time. Give '{name}' a dur or until", owner)
                t = t_out if edge == "out" else (t_in + t_out) / 2
        else:
            from .schema import suggest
            self._uncueable(s, owner)
            options = list(self.nodes) + list(self.markers)
            close = suggest(name, options)
            known = ", ".join(options[:16])
            if " " in s.strip():
                raise TimingError(f"cannot read the time {s!r}. Write {FORMS}", owner)
            raise TimingError(f"unknown reference '{name}' in {s!r}." + (f" Did you mean '{close}'?" if close else "")
                              + f" Layer ids and markers: {known}", owner)
        try:
            return t + offset(rest, self.fps)
        except TimingError as e:
            raise TimingError(f"{e} in {s!r}. Write {FORMS}", owner) from None

    def _uncueable(self, s: str, owner: str):
        """A cue that starts with an id a cue cannot name (one with a space, a dot or other punctuation) says so."""
        s = s.strip()
        for lid in sorted(list(self.nodes) + list(self.markers), key=len, reverse=True):
            if s.startswith(lid) and _BASE.fullmatch(lid) is None and not s[len(lid):len(lid) + 1].isalnum():
                raise TimingError(f"{s!r} names '{lid}', but a cue cannot name an id with spaces, dots or other "
                                  "punctuation. Give it an id of letters, digits, - and _ only", owner)

    def _word(self, sid: str, word: str, edge: str, owner: str) -> float:
        """When sound `sid` says `word`: its start in the scene, plus the word's time in the part of the file that plays."""
        from .schema import suggest
        w = self.words.get(sid)
        if w is None:
            close = suggest(sid, self.words)
            raise TimingError(f"refers to the word '{word}' of '{sid}', which is not an audio id."
                              + (f" Did you mean '{close}'?" if close else "")
                              + (f" Audio ids: {', '.join(list(self.words)[:16])}" if self.words else ""), owner)
        if w["ids"] is None:
            raise TimingError(f"refers to the word '{word}' of '{sid}', but '{sid}' has no word times. Add {w['file']} "
                              "(kine voice writes it for a sound with text; for a recording, a speech-to-text tool "
                              f"that times words, such as Whisper, writes one) or give '{sid}' a \"words\" list", owner)
        if word not in w["ids"]:
            close = suggest(word, w["ids"])
            raise TimingError(f"'{sid}' has no word '{word}'." + (f" Did you mean '{close}'?" if close else "")
                              + f" Its words: {', '.join(list(w['ids'])[:24])}", owner)
        ws, we = w["ids"][word]
        if ws < w["trim0"] or (w["trim1"] is not None and ws >= w["trim1"]):
            raise TimingError(f"the word '{word}' is at {ws:.2f}s in the file of '{sid}', outside the part its trim "
                              "plays", owner)
        return self.start(sid) - w["trim0"] + (ws if edge in ("", "in") else we if edge == "out" else (ws + we) / 2)

    def _enter(self, lid: str):
        if lid in self._stack:
            raise TimingError("timing cycle: " + " -> ".join(self._stack + [lid])
                              + ". A layer cannot start or end relative to itself", lid)
        self._stack.append(lid)

    def start(self, lid: str) -> float:
        """When a layer starts. Worked out apart from its end, so a group can end relative to its children."""
        if lid in self._starts:
            return self._starts[lid]
        self._enter(lid)
        n = self.nodes[lid]
        t = self._time(n["at"], lid, self.start(n["parent"]) if n["parent"] else 0.0)
        self._stack.pop()
        self._starts[lid] = t
        return t

    def resolve(self, lid: str) -> tuple[float, float | None]:
        if lid in self.resolved:
            return self.resolved[lid]
        t_in = self.start(lid)
        self._enter(lid)
        n = self.nodes[lid]
        base = self.start(n["parent"]) if n["parent"] else 0.0
        until = n["until"]
        if n["dur"] is not None:
            d = n["dur"]
            if isinstance(d, str):
                clk = parse_clock(d, self.fps)
                if clk is None:
                    raise TimingError(f"dur {d!r} is not a length of time; use seconds like 2.5, or \"45f\"", lid)
                d = clk
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not finite(d):
                raise TimingError(f"dur {show(d)} is not a length of time; use seconds like 2.5, or \"45f\"", lid)
            t_out: float | None = t_in + float(d)
        elif until is not None and until != "end":
            t_out = self._time(until, lid, base)
        elif n["natural"] is not None and until != "end":
            t_out = t_in + n["natural"]
        elif n["span"] and until != "end":
            outs = [self.resolve(k)[1] for k in self.order if self.nodes[k]["parent"] == lid]
            ends = [o for o in outs if o is not None]
            t_out = max(ends) if ends and len(ends) == len(outs) else None  # an open child leaves it open
        elif n["default_end"] is not None and until != "end":
            # A parent still working out its end (one that ends with its children) leaves this one open.
            t_out = None if n["default_end"] in self._stack else self.resolve(n["default_end"])[1]
        else:
            t_out = None  # open: runs to the scene end
        if abs(t_in) > LIMIT or (t_out is not None and abs(t_out) > LIMIT):
            raise TimingError(f"{'starts' if abs(t_in) > LIMIT else 'ends'} at "
                              f"{secs(t_in if abs(t_in) > LIMIT else t_out)}, past {LIMIT:g} seconds", lid)
        if t_out is not None and t_out <= t_in:
            raise TimingError(f"ends at {secs(t_out)}, which is not after its start at {secs(t_in)}", lid)
        self._stack.pop()
        self.resolved[lid] = (t_in, t_out)
        return t_in, t_out

    def solve(self, duration=None) -> float:
        for lid in self.order:
            self.resolve(lid)
        if duration not in (None, "auto"):
            end = self._time(duration, "canvas", 0.0)
            if end <= 0 or end > LIMIT:
                raise TimingError(f"canvas.duration {duration!r} comes to {secs(end)}; the scene must last more than "
                                  f"0 s (and at most {LIMIT:g} s)", "canvas")
        else:
            ends = [o for lid, (_, o) in self.resolved.items() if o is not None and self.nodes[lid]["counts"]]
            end = max(ends) if ends else 5.0
            if end <= 0:
                raise TimingError(f"with canvas.duration \"auto\" the scene would end at {secs(end)}, because every "
                                  "timed layer ends by then. Give canvas.duration in seconds, or start the layers "
                                  "at 0 or later", "canvas")
        for lid, (i, o) in list(self.resolved.items()):
            if o is None:
                self.resolved[lid] = (i, max(end, i + 1 / self.fps))
        return end
