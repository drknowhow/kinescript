"""Cue-graph timing: start/end times written as references to other layers.

    "at": 1.5               seconds
    "at": "45f"             frames
    "at": "0:02.5"          m:ss
    "at": "title.out+0.2"   0.2 s after layer 'title' ends
    "at": "prev.in+10f"     10 frames after the previous layer starts
    "at": "drop"            a named marker from scene "markers"

Change one duration and everything cued from it moves with it.
"""
from __future__ import annotations

import re

_TERM = re.compile(r"\s*([+-])\s*(\d+(?:\.\d+)?)\s*(s|f|ms)?\s*")
_BASE = re.compile(r"^\s*([A-Za-z_][\w-]*)(?:\.(in|out|mid))?")


class TimingError(ValueError):
    def __init__(self, message: str, layer: str | None = None):
        super().__init__(message)
        self.layer = layer


FORMS = ('a number of seconds (1.5), "45f" (frames), "0:02.5", "title.out+0.2" (0.2 s after layer title ends), '
         '"title.in", "title.mid", "prev.out", or a marker name')


def parse_clock(s: str, fps: float) -> float | None:
    s = s.strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(s|f|ms)?", s)
    if m:
        v = float(m.group(1))
        return v / fps if m.group(2) == "f" else v / 1000 if m.group(2) == "ms" else v
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def split_ref(expr: str) -> tuple[str | None, str, str]:
    """'title.out+0.2' -> ('title', 'out', '+0.2'). Plain clocks return (None, '', expr)."""
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

    def __init__(self, fps: float, markers: dict[str, float] | None = None):
        self.fps = fps
        self.markers = {k: float(v) for k, v in (markers or {}).items()}
        self.nodes: dict[str, dict] = {}
        self.order: list[str] = []
        self.resolved: dict[str, tuple[float, float | None]] = {}
        self._stack: list[str] = []

    def add(self, lid: str, at, dur=None, until=None, natural: float | None = None,
            prev: str | None = None, parent: str | None = None, default_end: str | None = None):
        if lid in self.nodes:
            raise TimingError(f"duplicate layer id '{lid}'; every id must be unique", lid)
        self.nodes[lid] = dict(at=at, dur=dur, until=until, natural=natural, prev=prev,
                               parent=parent, default_end=default_end)
        self.order.append(lid)

    def _time(self, v, owner: str, base: float) -> float:
        if v is None:
            return base
        if isinstance(v, (int, float)):
            return base + float(v)
        s = str(v)
        clk = parse_clock(s, self.fps)
        if clk is not None:
            return base + clk
        name, edge, rest = split_ref(s)
        if name is None:
            raise TimingError(f"cannot read the time {s!r}. Write {FORMS}", owner)
        if name == "prev":
            name = self.nodes[owner]["prev"] if owner in self.nodes else None
            if name is None:
                raise TimingError("'prev' used on the first layer, which has no previous layer", owner)
        if name in self.markers and not edge:
            t = self.markers[name]
        elif name == "start":
            t = 0.0
        elif name in self.nodes:
            t_in, t_out = self.resolve(name)
            if edge in ("", "in"):
                t = t_in
            else:
                if t_out is None:
                    raise TimingError(f"refers to '{name}.{edge}', but '{name}' runs to the scene end, so it has "
                                      f"no end time. Give '{name}' a dur or until", owner)
                t = t_out if edge == "out" else (t_in + t_out) / 2
        else:
            from .schema import suggest
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

    def resolve(self, lid: str) -> tuple[float, float | None]:
        if lid in self.resolved:
            return self.resolved[lid]
        if lid in self._stack:
            raise TimingError("timing cycle: " + " -> ".join(self._stack + [lid])
                              + ". A layer cannot start or end relative to itself", lid)
        self._stack.append(lid)
        n = self.nodes[lid]
        base = self.resolve(n["parent"])[0] if n["parent"] else 0.0
        t_in = self._time(n["at"], lid, base)
        until = n["until"]
        if n["dur"] is not None:
            d = n["dur"]
            if isinstance(d, str):
                clk = parse_clock(d, self.fps)
                if clk is None:
                    raise TimingError(f"dur {d!r} is not a length of time; use seconds like 2.5, or \"45f\"", lid)
                d = clk
            t_out: float | None = t_in + float(d)
        elif until is not None and until != "end":
            t_out = self._time(until, lid, base)
        elif n["natural"] is not None and until != "end":
            t_out = t_in + n["natural"]
        elif n["default_end"] is not None and until != "end":
            t_out = self.resolve(n["default_end"])[1]
        else:
            t_out = None  # open: runs to the scene end
        if t_out is not None and t_out <= t_in:
            raise TimingError(f"ends at {t_out:.3f}s, which is not after its start at {t_in:.3f}s", lid)
        self._stack.pop()
        self.resolved[lid] = (t_in, t_out)
        return t_in, t_out

    def solve(self, duration=None) -> float:
        for lid in self.order:
            self.resolve(lid)
        if duration not in (None, "auto"):
            end = self._time(duration, "canvas", 0.0)
        else:
            ends = [o for _, o in self.resolved.values() if o is not None]
            end = max(ends) if ends else 5.0
        for lid, (i, o) in list(self.resolved.items()):
            if o is None:
                self.resolved[lid] = (i, max(end, i + 1 / self.fps))
        return end
