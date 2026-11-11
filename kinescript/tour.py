"""use: "tour": a list of stops made into what a guided explainer needs. Each stop flies the camera to its focus (a
layer, or a point in a layer's own space), holds while its voice line plays, and shows a caption card and a label
pinned to the focus. The stops are timed one after another by their lines, so the camera keys are written once
the timeline is solved (Scene.late), from where each stop landed."""
from __future__ import annotations

import bisect
import math

from . import space

PARAMS = {"stops", "at", "move", "hold", "pause", "speaker", "rate", "ease", "drift", "cards", "accent", "fov", "dim"}
STOP = {"focus", "zoom", "orbit", "say", "title", "body", "note", "hold"}
WORDS_A_SECOND = 2.6  # a voice line's length before kine voice has spoken it
DEFAULTS = {"move": 1.2, "hold": 2.5, "pause": 0.4, "ease": "inOutCubic", "drift": 4.0, "cards": True, "dim": 0.0}


class TourError(ValueError):
    pass


def tours(layers) -> list[dict]:
    """The scene's tour layers: at the top level only, where they place the camera."""
    return [L for L in layers or [] if isinstance(L, dict) and L.get("use") == "tour"]


def say_sounds(spec: dict) -> dict:
    """The scene with each tour stop's say written as a voice sound in "audio" (id "<tour>-s<n>-v"), so kine voice
    speaks it and its length times the stop."""
    found = tours(spec.get("layers"))
    if not any(isinstance(st, dict) and st.get("say") for L in found for st in L.get("stops") or []):
        return spec
    import copy
    spec = copy.deepcopy(spec)
    audio = spec.setdefault("audio", [])
    taken = {str(a.get("id")) for a in audio if isinstance(a, dict)}
    for n, L in enumerate(tours(spec.get("layers")), 1):
        tid = str(L.setdefault("id", f"tour{n}"))
        o = dict(DEFAULTS, **{k: v for k, v in L.items() if k in PARAMS and v is not None})
        for i, stop in enumerate(L.get("stops") or [], 1):
            if not isinstance(stop, dict) or not isinstance(stop.get("say"), str) or f"{tid}-s{i}-v" in taken:
                continue
            a = {"id": f"{tid}-s{i}-v", "role": "voice", "text": stop["say"], "at": f"{tid}-s{i}.in+{_lead(o, i)}"}
            for k in ("speaker", "rate"):
                if o.get(k) is not None:
                    a[k] = o[k]
            audio.append(a)
    return spec


def _lead(o: dict, i: int) -> float:
    """Seconds into stop i before it lands: the camera's move, none for the first stop, which starts there."""
    return float(o["move"]) if i > 1 else 0.0


def expand(raw: dict, scene) -> list[dict]:
    """A tour layer -> top-level layers: a group for each stop, timed after the one before and as long as its move
    and its voice line (say_sounds put the lines in audio), holding its card; and a callout pinned to its focus for
    its note. Registers the camera keys with scene.late."""
    tid = str(raw.get("id") or "tour")
    unknown = sorted(set(raw) - PARAMS - {"use", "id", "type"})
    if unknown:
        raise TourError(f"layer '{tid}': tour has no parameter '{unknown[0]}'. Its parameters: "
                        f"{', '.join(sorted(PARAMS))}")
    stops = raw.get("stops")
    if not isinstance(stops, list) or not stops or not all(isinstance(s, dict) for s in stops):
        raise TourError(f"layer '{tid}': tour stops is a list of one or more objects like "
                        '{"focus": "cell", "say": "This is a cell.", "title": "The cell"}')
    o = dict(DEFAULTS, **{k: v for k, v in raw.items() if k in PARAMS and v is not None})
    for k in ("move", "hold", "pause", "drift"):
        if isinstance(o[k], bool) or not isinstance(o[k], (int, float)) or not 0 <= o[k] < 1e4:
            raise TourError(f"layer '{tid}': tour {k} is {o[k]!r}; it is seconds (drift: degrees), 0 or more")
    if isinstance(o["dim"], bool) or not isinstance(o["dim"], (int, float)) or not 0 <= o["dim"] <= 1:
        raise TourError(f"layer '{tid}': tour dim is {o['dim']!r}; it is how far other models fade, 0 to 1")
    layers, prev = [], None
    for i, stop in enumerate(stops, 1):
        _check_stop(tid, i, stop)
        sid = f"{tid}-s{i}"
        line = _line(stop, o, tid, i, scene)
        hold = float(stop.get("hold", o["hold"]))
        dur = _lead(o, i) + (line + o["pause"] if line is not None else hold)
        at = f"{prev}.out" if prev else raw.get("at", 0)
        kids = []
        if o["cards"] and (stop.get("title") or stop.get("body")):
            card = {"use": "lowerThird", "id": f"{sid}-card", "title": str(stop.get("title") or ""),
                    "at": _lead(o, i)}
            if stop.get("body"):
                card["subtitle"] = str(stop["body"])
            if o.get("accent"):
                card["accent"] = o["accent"]
            kids.append(card)
        # On the screen, not in the space the camera flies through: the card stays put as the camera moves.
        layers.append({"id": sid, "type": "group", "at": at, "dur": round(dur, 4), "fixed": True, "children": kids})
        if stop.get("note"):
            layers.append({"use": "callout", "id": f"{sid}-note", "pin": _pin(stop["focus"]), "r": 26,
                           "label": str(stop["note"]), "size": 30, "at": f"{sid}.in+{_lead(o, i)}",
                           "until": f"{sid}.out", **({"accent": o["accent"]} if o.get("accent") else {})})
        prev = sid
    scene.late.append(lambda s: _camera(s, tid, stops, o))
    return layers


def _check_stop(tid: str, i: int, stop: dict):
    where = f"layer '{tid}': tour stop {i}"
    extra = sorted(set(stop) - STOP)
    if extra:
        raise TourError(f"{where} has no key '{extra[0]}'. A stop takes {', '.join(sorted(STOP))}")
    f = stop.get("focus")
    if not (isinstance(f, str) and f.strip()) and not (isinstance(f, dict) and isinstance(f.get("layer"), str)):
        raise TourError(f"{where}: focus is a layer id, or {{\"layer\": \"cell\", \"at\": [x, y, z]}}")
    if isinstance(f, dict):
        at = f.get("at")
        if at is not None and not (isinstance(at, list) and len(at) in (2, 3) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in at)):
            raise TourError(f"{where}: focus at is a point [x, y, z] in px from the layer's middle")
    for k, least in (("zoom", 1e-3), ("hold", 0.0)):
        v = stop.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not least <= v < 1e4):
            raise TourError(f"{where}: {k} is {v!r}; it is a number" + (" above 0" if k == "zoom" else ", 0 or more"))
    v = stop.get("orbit")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not -720 <= v <= 720):
        raise TourError(f"{where}: orbit is {v!r}; it is degrees the camera swings round the focus")
    if stop.get("say") is not None and not (isinstance(stop["say"], str) and stop["say"].strip()):
        raise TourError(f"{where}: say is the words the voice speaks, as text")


def _pin(focus) -> dict:
    return {"layer": focus} if isinstance(focus, str) else {"layer": focus["layer"], "at": list(focus.get("at") or [])
                                                           or [0, 0, 0]}


def _line(stop: dict, o: dict, tid: str, i: int, scene) -> float | None:
    """How long a stop's voice line lasts: its file's length once kine voice has spoken it, else an estimate."""
    say = stop.get("say")
    if say is None:
        return None
    from . import voice as V
    from .scene import _natural
    src = scene.path(V.default_src(f"{tid}-s{i}-v")) if scene is not None else None
    n = _natural({"text": say}, src) if src is not None else None
    if n:
        return float(n)
    words = len([w for w in str(say).replace("[[", " [[").split() if not w.startswith("[[")])
    return max(0.6, words / WORDS_A_SECOND / float(o.get("rate") or 1.0))


def _camera(scene, tid: str, stops: list, o: dict):
    """The camera's target, z and rotateY keys from where each stop group landed: it flies to each focus over
    `move` s, eased, closes in to the stop's zoom, and swings round it by orbit (plus a little drift) while it
    holds."""
    cam = scene.camera
    fov = float(o.get("fov") or (cam.raw.get("fov") if cam is not None and isinstance(cam.raw.get("fov"), (int, float))
                                 else space.DEFAULT_FOV))
    from .schema import suggest
    lost = [(i, s["focus"] if isinstance(s["focus"], str) else s["focus"]["layer"]) for i, s in enumerate(stops, 1)]
    lost = [(i, lid) for i, lid in lost if lid not in scene.by_id]
    for i, lid in lost:
        close = suggest(lid, list(scene.by_id))
        scene.note("error", "tour-focus", tid, f"tour stop {i}: focus '{lid}' names no layer",
                   (f"did you mean '{close}'? " if close else "") + "give the layer to fly to that id")
    if lost:
        return
    f = space.focus(scene.h, fov)
    ease = o["ease"]
    target, z, ry, spans = [], [], [], []
    turn = 0.0
    for i, stop in enumerate(stops, 1):
        G = scene.by_id[f"{tid}-s{i}"]
        t0, t1 = G.t_in, G.t_out
        arrive = t0 + (min(o["move"], (t1 - t0) / 2) if i > 1 else 0.0)
        spans.append((t0, arrive))
        aim = _pin(stop["focus"])
        aim = aim["layer"] if isinstance(stop["focus"], str) else aim
        zoom = float(stop.get("zoom", 1.0))
        back = round(f * (1 / zoom - 1), 3)  # camera z that shows the focus zoom times as large
        if i > 1 and t0 > target[-1][0] + 1e-6:  # hold where the last stop left it until this one starts
            target.append([t0, target[-1][1]])
            z.append([t0, z[-1][1]])
        if i > 1 and arrive > t0 + 1e-6:
            target.append([arrive, aim, ease])
            z.append([arrive, back, ease])
        elif i > 1:  # no move: cut to it
            target[-1], z[-1] = [t0, aim], [t0, back]
        else:
            target.append([arrive, aim])
            z.append([arrive, back])
        if not ry or arrive > ry[-1][0] + 1e-6:
            ry.append([arrive, round(turn, 3)])
        turn += float(stop.get("orbit", 0.0)) + o["drift"] * (1 if i % 2 else -1)
        if t1 > ry[-1][0] + 1e-6:
            ry.append([t1, round(turn, 3), "inOutSine"])
    raw = dict(cam.raw if cam is not None else {}, target=target, z=z, rotateY=ry)
    camera = scene._camera(raw)
    camera.t_out = scene.duration
    scene.camera = camera
    if o["dim"]:
        _dim(scene, stops, spans, 1.0 - float(o["dim"]))


def _dim(scene, stops: list, spans: list, low: float):
    """Each model layer a stop does not focus (itself or a group round it) fades to `low` of its opacity while the
    camera flies there, and back as a stop focuses it: a function of scene time, so frames stay pure."""
    from .props import Prop
    focus = [s["focus"] if isinstance(s["focus"], str) else s["focus"]["layer"] for s in stops]
    for L in scene.by_id.values():
        if L.type != "model":
            continue
        line, P = set(), L
        while P is not None:
            line.add(P.id)
            P = P.parent
        keys, prev = [], None
        for (t0, arrive), f in zip(spans, focus):
            v = 1.0 if f in line else low
            if prev is not None:
                keys.append((t0, prev))
            keys.append((arrive, v))
            prev = v
        if all(v == 1.0 for _, v in keys):
            continue
        L.props["opacity"] = Prop(False, fn=_faded(L.props.get("opacity"), keys))


def _faded(old, keys: list):
    times = [k[0] for k in keys]

    def fn(t, d, T, i=0, *n):
        base = 1.0 if old is None else (old.value if old.static else old.fn(t, d, T, i, *n))
        j = bisect.bisect_right(times, T)
        if j == 0:
            return base * keys[0][1]
        if j == len(keys):
            return base * keys[-1][1]
        (a, va), (b, vb) = keys[j - 1], keys[j]
        u = (T - a) / (b - a) if b > a else 1.0
        return base * (va + (vb - va) * u * u * (3 - 2 * u))
    return fn
