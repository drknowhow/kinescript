"""3D avatars: an avatar.json whose "model" is a glTF file. Its mouth shapes, blink and expressions are morph target
weights by name, so the avatar is one model layer whose "morphs" this module works out.

    {"kinescriptAvatar": 1, "name": "Ada",
     "model": {"src": "ada.glb", "animation": "Idle"},
     "mouth": {"A": {"mouthClose": 1}, "C": {"jawOpen": 0.4}, "D": {"jawOpen": 0.9}},
     "blink": {"eyeBlinkLeft": 1, "eyeBlinkRight": 1},
     "expressions": {"smile": {"morphs": {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}}}}

A mouth shape left out falls back as a drawn avatar's does (B to C, E and F to D, ...), and X, at rest, is every
mouth morph at 0 unless given. Keys cannot read cues (when a line starts depends on the scene's timing), so the
component leaves each morph at 0 and Scene runs bake() once the timing is solved: it samples the mouth, blinks and
expressions at every frame of the avatar and writes them as keys on the layer's clock. A frame stays a pure function
of the scene and its number.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

from . import mouth as M

KEYS = {"src", "animation"}  # what "model" takes
BLEND = 0.06  # seconds the mouth takes to move from one shape to the next
CLOSE = 0.04  # seconds the lids take to close (and open) around a blink's BLINK seconds shut
LOOK = (25.0, 12.0)  # degrees the model turns for a full gaze aside and up or down, unless "look" says
TURN = 0.3  # seconds a turn to a new gaze takes
GESTURE_KEYS = {"clip", "speed"}


def is_model(spec: dict) -> bool:
    return isinstance(spec, dict) and "model" in spec


def _issue(level: str, code: str, message: str, fix: str = "") -> dict:
    return {"level": level, "code": code, "message": message, "fix": fix}


def _weights(where: str, v, names: list[str] | None, out: list) -> None:
    """A map of morph target names to weights, each 0 to 1 and a name the model file has."""
    if not isinstance(v, dict):
        out.append(_issue("error", "avatar-morph", f"{where} must be an object of morph targets and weights, "
                                                    'such as {"jawOpen": 0.8}'))
        return
    for n, w in v.items():
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 <= w <= 1:
            out.append(_issue("error", "avatar-morph", f"{where}.{n} is {w!r}; a morph weight is 0 to 1"))
        if names is not None and n not in names:
            from .schema import suggest
            close = suggest(n, names) if names else None
            out.append(_issue("error", "avatar-morph", f"{where}: the model has no morph target '{n}'",
                              f"did you mean '{close}'?" if close else
                              "its morph targets: " + (", ".join(names) or "none") + " (kine probe lists them)"))


def check(folder: Path, spec: dict, file) -> tuple[list[dict], Path | None, dict]:
    """The problems with a 3D avatar's spec, its model file, and the mouth shape each shape shows (a fallback)."""
    from . import models
    out: list[dict] = []
    mo = spec.get("model")
    if not isinstance(mo, dict) or not isinstance(mo.get("src"), str):
        out.append(_issue("error", "avatar-model", 'model must be {"src": "face.glb"}, a glTF file in the folder'))
        return out, None, {}
    for k in sorted(set(mo) - KEYS):
        out.append(_issue("warning", "avatar-key", f"model has an unknown key '{k}'", "known: animation, src"))
    if "parts" in spec:
        out.append(_issue("error", "avatar-model", "an avatar has parts (2D) or a model (3D), not both",
                          "drop parts, or drop model"))
    src = file(folder, mo["src"], "model.src", out, (".glb", ".gltf"))
    names, imp = None, None
    if src is not None:
        try:
            imp = models.load(str(src))
            names = models.morph_names(imp)
            clip = mo.get("animation")
            if clip is not None and models.clip_index(imp, clip) is None:
                out.append(_issue("error", "avatar-model", f"model.animation {clip!r} is not in {src.name}",
                                  "its animations: " + (", ".join(models.clip_names(imp)) or "none")))
        except (ValueError, OSError) as e:
            imp = None
            out.append(_issue("error", "avatar-model", f"model.src: {e}"))
    mouth = spec.get("mouth")
    falls: dict = {}
    if mouth is None:
        out.append(_issue("info", "avatar-no-mouth", "no mouth map, so the avatar does not talk",
                          'give "mouth": {"A": {...}, "C": {...}, "D": {...}}, each shape its morph weights'))
    elif not isinstance(mouth, dict):
        out.append(_issue("error", "avatar-morph", "mouth must be an object of mouth shapes and their morphs"))
    else:
        for s, v in mouth.items():
            if s not in M.SHAPES:
                out.append(_issue("error", "avatar-morph", f"mouth.{s} is not a mouth shape",
                                  f"the shapes are {', '.join(M.SHAPES)}"))
            _weights(f"mouth.{s}", v, names, out)
        missing = [s for s in M.REQUIRED if s != "X" and s not in mouth]
        if missing:
            out.append(_issue("error", "avatar-missing-shape", f"mouth lacks shape{'s' * (len(missing) > 1)} "
                              f"{', '.join(missing)}", "map at least A (closed), C (open) and D (wide)"))
        for s in M.SHAPES[1:]:  # X, at rest, is every mouth morph at 0 unless mapped
            if s not in mouth:
                to = M.FALLBACK.get(s, "X")
                falls[s] = to if to in mouth else "X"
    if "blink" in spec:
        _weights("blink", spec["blink"], names, out)
    for name, e in (spec.get("expressions") or {}).items() if isinstance(spec.get("expressions"), dict) else ():
        if not isinstance(e, dict) or set(e) != {"morphs"}:
            out.append(_issue("error", "avatar-expression", f"expression '{name}' of a 3D avatar is "
                                                            '{"morphs": {"name": weight}}'))
        else:
            _weights(f"expression '{name}'.morphs", e["morphs"], names, out)
    gests = spec.get("gestures") or {}
    if not isinstance(gests, dict):
        out.append(_issue("error", "avatar-gesture", "gestures must be an object of named clips"))
        gests = {}
    for name, g in gests.items():
        if not isinstance(g, dict) or not isinstance(g.get("clip"), (str, int)) or set(g) - GESTURE_KEYS:
            out.append(_issue("error", "avatar-gesture", f"gesture '{name}' of a 3D avatar is "
                                                         '{"clip": "Wave"}, a clip of the model, with speed'))
        elif imp is not None and models.clip_index(imp, g["clip"]) is None:
            out.append(_issue("error", "avatar-gesture", f"gesture '{name}': {g['clip']!r} is not a clip of "
                              f"{src.name}", "its animations: " + (", ".join(models.clip_names(imp)) or "none")))
    look = spec.get("look")
    if look is not None and not (isinstance(look, list) and len(look) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 90 for v in look)):
        out.append(_issue("error", "avatar-look", "look is [degrees aside, degrees up or down] the model turns for "
                                                  "a full gaze, each 0 to 90, such as [25, 12]"))
    return out, src, falls


def shape(av, s: str) -> dict:
    """The morph weights mouth shape s shows (a fallback's when s is not mapped); X is every mouth morph at 0."""
    mouth = av.spec.get("mouth") or {}
    s = s if s in mouth else av.fallbacks.get(s, "X")
    return mouth.get(s, {})


def used(av) -> list[str]:
    """Every morph target the avatar sets, in the order it names them."""
    names: dict = {}
    for v in (av.spec.get("mouth") or {}).values():
        names.update(dict.fromkeys(v))
    names.update(dict.fromkeys(av.spec.get("blink") or {}))
    for e in (av.spec.get("expressions") or {}).values():
        names.update(dict.fromkeys(e["morphs"]))
    return list(names)


def layer(av, lid: str, w: float, h: float, morphs: dict | None = None) -> dict:
    """The avatar's model layer, fitted inside w by h about its (0, 0)."""
    mo = av.spec["model"]
    L = {"id": lid, "type": "model", "src": av.model.as_posix(), "x": 0, "y": 0, "w": w, "h": h,
         "morphs": {n: (morphs or {}).get(n, 0) for n in used(av)}}
    if mo.get("animation") is not None:
        L["animation"] = mo["animation"]
    return L


def still(av, lid: str, pose: dict, k: float = 1.0) -> dict:
    """The avatar in one still pose, k times its size, a group with its (0, 0) at the avatar box's top left: pose
    takes "mouth" (a shape), "eyes" ("closed" for the blink) and "morphs" (an expression's)."""
    w, h = av.size[0] * k, av.size[1] * k
    m = dict(shape(av, pose.get("mouth", "X")))
    if pose.get("eyes") == "closed":
        m.update(av.spec.get("blink") or {})
    m.update(pose.get("morphs") or {})
    L = layer(av, f"{lid}-model", w, h, m)
    L.update(x=w / 2, y=h / 2)
    if pose.get("clip"):
        name, at = pose["clip"]
        L["animation"] = {"name": name, "start": at, "speed": 0}
    if pose.get("look"):
        L.update(_turn(av, *pose["look"]))
    return {"id": lid, "type": "group", "children": [L]}


def _turn(av, gx: float, gy: float) -> dict:
    """The model's rotateY and rotateX that look at [gx, gy]."""
    look = av.spec.get("look") or LOOK
    return {"rotateY": round(gx * look[0], 4), "rotateX": round(-gy * look[1], 4)}


def poses(av) -> list[tuple[str, dict]]:
    out = [("neutral", {})]
    if av.spec.get("mouth"):
        out += [(f"mouth {s}" + (f" (as {av.fallbacks[s]})" if s in av.fallbacks else ""), {"mouth": s})
                for s in M.SHAPES]
    if av.spec.get("blink"):
        out.append(("eyes closed", {"eyes": "closed"}))
    out += [("look left", {"look": [-1, 0]}), ("look right", {"look": [1, 0]})]
    out += [(f"expression {n}", {"morphs": e["morphs"]}) for n, e in (av.spec.get("expressions") or {}).items()]
    from . import models
    imp = models.load(str(av.model))
    for n, g in (av.spec.get("gestures") or {}).items():
        i = models.clip_index(imp, g["clip"])
        t = round(imp.motion.clips[i][2] / 2, 3)
        out.append((f"gesture {n} at {t:g}s", {"clip": (g["clip"], t)}))
    return out


def component(av, p: dict, scene, say: list, exprs: list, idle: dict, k: float) -> dict:
    """The avatar as a group holding its model layer (breathing in a group of its own when idle.breath is on), and
    the bake that sets its morphs once the scene's timing is solved."""
    from . import avatar as A
    lid = p["id"]
    w, h = av.size
    model = layer(av, f"{lid}-model", w * k, h * k)
    gests = A._entries(p, "gestures", av.spec.get("gestures") or {})
    if gests:
        model["clips"] = [{"name": av.spec["gestures"][g["name"]]["clip"], "at": g.get("at", 0),
                           **({"speed": av.spec["gestures"][g["name"]]["speed"]}
                              if "speed" in av.spec["gestures"][g["name"]] else {})} for g in gests]
    gazes = []
    for i, g in enumerate(A._entries(p, "gaze", None)):
        try:
            gazes.append((g.get("at", 0), *A._gaze(g.get("target", "camera"), p, av, k, scene, f"gaze[{i}]")))
        except ValueError as e:
            scene.note("error", "avatar-gaze-target", lid, str(e), "")
    tracks = []
    for sid in say:
        got, t0, words_file = A._track(scene, sid)
        A._notes(scene, lid, sid, got, words_file, av)
        tracks.append((sid, got["mouth"], t0))
    blinks = A._blink_times(lid, idle, scene) if idle["blink"] and av.spec.get("blink") else []
    late = getattr(scene, "late", None)
    if late is not None:
        late.append(lambda s: bake(s, av, lid, model["id"], tracks, exprs, blinks, gazes))
    body = model
    if idle["breath"]:
        body = {"id": f"{lid}-breath", "type": "group", "at": f"{lid}.in",
                "scale": f"=1+{idle['breath']:g}*(0.5-0.5*cos(tau*t/4))", "children": [model]}
    return {"type": "group", "x": p.get("x", "50%"), "y": p.get("y", "50%"), "children": [body]}


def _segments(scene, lid: str, tracks: list) -> list[tuple[float, str]]:
    """When each mouth shape starts, in scene seconds: each line's track from when its sound starts, until the
    next line starts, then at rest."""
    out: list[tuple[float, str]] = []
    for sid, track, t0 in tracks:
        s = scene.timeline._time(f"{sid}.in", lid, 0.0)
        out = [x for x in out if x[0] < s]
        out += [(s + max(0.0, a - t0), sh) for sh, a, _ in track]
        if track:
            out.append((s + max(0.0, track[-1][2] - t0), "X"))
    return out


def _mouth_at(av, segs: list, starts: list, T: float) -> dict:
    j = bisect_right(starts, T) - 1
    if j < 0:
        return shape(av, "X")
    cur = shape(av, segs[j][1])
    f = min(1.0, (T - segs[j][0]) / BLEND)
    if f >= 1:
        return cur
    prev = shape(av, segs[j - 1][1]) if j > 0 else shape(av, "X")
    return {n: prev.get(n, 0) + (cur.get(n, 0) - prev.get(n, 0)) * f for n in set(prev) | set(cur)}


def _envelope(T: float, at: float, dur, ramp: float) -> float:
    if T <= at:
        return 0.0
    up = min(1.0, (T - at) / ramp)
    if dur is None:
        return up
    return max(0.0, min(up, (at + dur - T) / ramp))


def _lids(T: float, blinks: list) -> float:
    from .avatar import BLINK
    j = bisect_right(blinks, T + CLOSE) - 1
    if j < 0:
        return 0.0
    b = blinks[j]
    if T < b:
        return 1 - (b - T) / CLOSE
    return max(0.0, min(1.0, 1 - (T - b - BLINK) / CLOSE))


def _gazing(T: float, turns: list) -> tuple[float, float]:
    """Where the avatar looks at scene time T: each gaze eased in over TURN from where the one before left it."""
    gx = gy = 0.0
    for at, x, y in turns:
        if T <= at:
            break
        f = min(1.0, (T - at) / TURN)
        f = f * f * (3 - 2 * f)
        gx, gy = gx + (x - gx) * f, gy + (y - gy) * f
    return gx, gy


def bake(scene, av, lid: str, mid: str, tracks: list, exprs: list, blinks: list, gazes: list = ()) -> None:
    """Set the model layer's morphs: the mouth, blinks and expressions summed (at most 1) at every frame of the
    layer, as keys on its clock that change only where they must."""
    from .avatar import QUIET, RAMP
    from .props import compile_prop
    L = scene.by_id.get(mid)
    if L is None:
        return
    try:
        segs = _segments(scene, lid, tracks)
        base = scene.timeline.start(lid)
        ex = [(scene.timeline._time(e.get("at", 0), lid, base), e.get("dur"), float(e.get("weight", 1)),
               av.spec["expressions"][e["name"]]["morphs"]) for e in exprs]
        lines = [scene.timeline._time(f"{sid}.in", lid, 0.0) for sid, _, _ in tracks]
        turns = sorted((scene.timeline._time(at, lid, base), gx, gy) for at, gx, gy in gazes)
    except ValueError as e:
        scene.note("error", "avatar-time", lid, str(e), "")
        return
    from .avatar import BLINK
    shut = [base + b for b in blinks
            if not any(s - BLINK - CLOSE < base + b < s + QUIET for s in lines)]  # not as a line starts
    starts = [s for s, _ in segs]
    blink = av.spec.get("blink") or {}
    fps, names = scene.fps, used(av)
    end = L.t_out if L.t_out is not None else scene.duration
    f0, f1 = round(L.t_in * fps), round(min(end, scene.duration) * fps)
    keys: dict[str, list] = {"w_" + n: [] for n in names}
    if turns:
        keys.update(rotateY=[], rotateX=[])
    for f in range(f0, f1 + 1):
        T = f / fps
        v = dict.fromkeys(names, 0.0)
        for n, w in _mouth_at(av, segs, starts, T).items():
            v[n] += w
        lids = _lids(T, shut) if shut else 0.0
        for n, w in blink.items():
            v[n] += w * lids
        for at, dur, wt, m in ex:
            env = _envelope(T, at, dur, min(RAMP, dur / 2) if dur else RAMP)
            for n, w in m.items():
                v[n] += env * wt * w
        t = round(T - L.t_in, 6)
        for n in names:
            keys["w_" + n].append([t, round(min(1.0, max(0.0, v[n])), 4), "linear"])
        if turns:
            for a, d in _turn(av, *_gazing(T, turns)).items():
                keys[a].append([t, d, "linear"])
    for n, ks in keys.items():
        kept = [ks[0]] if ks else [[0, 0.0]]
        for p, a, b in zip(ks, ks[1:], ks[2:]):  # drop a key on the line between its neighbours (frames are evenly
            if abs(a[1] - (p[1] + b[1]) / 2) > 1e-6:  # spaced, so a run of such keys is one straight line)
                kept.append(a)
        if len(ks) > 1:
            kept.append(ks[-1])
        if n.startswith("w_"):
            L.raw["morphs"][n[2:]] = kept
        L.props[n] = compile_prop(n, kept if len(kept) > 1 else kept[0][1], L.ctx or scene.ctx)
