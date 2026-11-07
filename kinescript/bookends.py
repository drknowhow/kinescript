"""A brand's intro and outro around a video: "intro": true / "outro": true on a scene.

Each is a scene of its own, made at the video's size and frame rate from the brand: the sting or endCard
built-in with the brand's "intro"/"outro" parameters, or a scene file in the brand folder ("intro": {"scene":
"intro.json"}). kine render renders it as a segment before or after the scene's frames and joins them, so the
scene's own times, cues, markers and sound stay as written; check, look, still and probe see the scene alone.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import brand as B

PARTS = tuple(B.BOOKENDS)


def wanted(spec: dict) -> list[str]:
    """Which of intro and outro a scene asks for."""
    return [k for k in PARTS if spec.get(k)]


def segment(scene, part: str) -> tuple[dict, str] | None:
    """The (spec, base folder) of the scene's intro or outro, or None when it does not ask for one.
    ValueError says why one cannot be made."""
    ask = scene.spec.get(part)
    if not ask:
        return None
    if scene.brand is None:
        raise ValueError(f"\"{part}\": true takes the scene's brand's {part}, and the scene names no brand; add "
                         "\"brand\": \"brand\"")
    builtin, default_dur = B.BOOKENDS[part]
    conf = dict(scene.brand.spec.get(part) or {})
    if isinstance(ask, dict):
        conf.update(ask)
    dur = conf.pop("dur", default_dur)
    folder = scene.brand.folder.resolve()
    canvas = {"w": scene.w, "h": scene.h, "fps": scene.fps}
    variant = scene.spec.get("variant")
    if "scene" in conf:
        p = B.inside(folder, conf.pop("scene"))
        try:
            spec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{part}.scene {p.name}: {e}") from None
        if not isinstance(spec, dict):
            raise ValueError(f"{part}.scene {p.name} must hold a scene object")
        spec = copy.deepcopy(spec)
        spec["canvas"] = {**(spec.get("canvas") or {}), **canvas}
        spec["canvas"].setdefault("duration", dur)
        spec["brand"] = str(folder)
        if variant is not None:
            spec.setdefault("variant", variant)
        for k in PARTS:
            spec.pop(k, None)  # an intro of the intro would never end
        return spec, str(p.parent)
    spec = {"canvas": {**canvas, "duration": dur}, "brand": str(folder),
            "layers": [{"use": builtin, "id": part, **conf}]}
    if variant is not None:
        spec["variant"] = variant
    return spec, str(folder)


def segments(scene) -> dict[str, tuple[dict, str]]:
    return {k: seg for k in PARTS if (seg := segment(scene, k)) is not None}


def durations(scene) -> dict[str, float]:
    """Seconds each bookend the scene asks for lasts (for probe)."""
    from .scene import Scene
    out = {}
    for k, (spec, base) in segments(scene).items():
        s = Scene(spec, base)
        try:
            out[k] = round(s.duration, 3)
        finally:
            s.close()
    return out


def check(scene) -> list[dict]:
    """The issues of the scene's intro and outro, each named after its part (bad-intro when one cannot be made)."""
    from .scene import Scene, SceneError
    out = []
    for k in wanted(scene.spec):
        try:
            seg = segment(scene, k)
            s = Scene(*seg)
        except (SceneError, ValueError) as e:
            issues = getattr(e, "issues", None) or [{"level": "error", "code": f"bad-{k}", "layer": k,
                                                    "message": str(e), "fix": ""}]
            out += [{**i, "message": f"{k}: {i['message']}"} for i in issues]
            continue
        try:
            out += [{**i, "message": f"{k}: {i['message']}"} for i in s.issues if i.get("level") == "error"]
        finally:
            s.close()
    return out
