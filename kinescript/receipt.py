"""Render receipts: what a video was made from, written next to it, and a check of a video against its receipt.

kine render writes OUT.receipt.json beside each video or sound file: the Kinescript and ffmpeg that made it, the
render's settings, and a SHA-256 of the output, the scene and every file the scene reads (footage, images, sounds
and their word times, fonts, models and the files a model names, an avatar's folder, the brand's folder). Nothing
in it changes from one render to the next unless something it records did, so a receipt kept with the scene diffs
cleanly, and `kine receipt OUT` says whether the file, the scene or any input changed since.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

SUFFIX = ".receipt.json"
FORMAT = 1
_PACKAGE = Path(__file__).resolve().parent
# Keys whose value can name a folder rather than a file; a folder they name is read whole. Elsewhere a string
# that names a folder (".", "media/") is left alone, so a scene never records its own project.
_FOLDERS = {"avatar", "frames"}
_SIBLINGS = (".words.json", ".mouth.json")  # a sound's word times and mouth track, read beside it


def path_for(out) -> Path:
    """Where the receipt of `out` goes: out.mp4 -> out.mp4.receipt.json."""
    p = Path(out)
    return p.with_name(p.name + SUFFIX)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rel(p: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(p, base)).as_posix()
    except ValueError:  # another drive on Windows
        return p.as_posix()


def _model_files(p: Path) -> list[Path]:
    """The files a model file names: a .gltf's buffers and images, an .obj's .mtl and the maps in it."""
    out: list[Path] = []
    try:
        if p.suffix.lower() == ".gltf":
            doc = json.loads(p.read_text(encoding="utf-8"))
            for item in (doc.get("buffers") or []) + (doc.get("images") or []):
                u = item.get("uri") if isinstance(item, dict) else None
                if isinstance(u, str) and not u.startswith("data:"):
                    out.append(p.parent / u)
        elif p.suffix.lower() in (".obj", ".mtl"):
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                t = line.split(None, 1)
                if len(t) == 2 and (t[0] == "mtllib" or t[0].startswith("map_")):
                    name = t[1].split()[-1]  # a map line can carry options before the file
                    q = p.parent / name
                    out.append(q)
                    if t[0] == "mtllib":
                        out += _model_files(q)
    except (OSError, ValueError, AttributeError):
        pass  # the scene's own checks report a model it cannot read
    return out


def inputs(scene, skip=()) -> list[Path]:
    """Every file the scene reads, resolved, sorted, without the scene file and files in `skip`. Files inside the
    Kinescript package (its fonts, the built-in avatars) are the version's, and the receipt names the version."""
    base = Path(scene.base)
    found: set[Path] = set()

    def add(p, folder=False):
        try:
            q = Path(p)
            q = (q if q.is_absolute() else base / q).resolve()
        except (OSError, ValueError, TypeError):
            return
        if q.is_file():
            found.add(q)
        elif folder and q.is_dir() and q != base.resolve() and q not in base.resolve().parents:
            found.update(x.resolve() for x in q.rglob("*") if x.is_file())

    def walk(v, key=None):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x, key)
        elif isinstance(v, str) and len(v) < 1024 and "\n" not in v and re.search(r"[./\\]", v) \
                and not v.startswith(("=", "#", "$", "kine:", "http:", "https:", "data:")):
            add(v, key in _FOLDERS)

    walk(scene.spec)
    walk(scene.theme)
    for L in scene.all_layers():
        walk(L.raw)
        if getattr(L, "src", None):
            add(L.src)
        seq = getattr(L, "seq", None)
        for f in getattr(seq, "files", None) or ():
            add(f)
    for s in scene.audio:
        walk(s.raw)
        if s.src:
            add(s.src)
    if getattr(scene, "brand", None) is not None:
        add(scene.brand.folder, folder=True)
    for p in list(found):
        for suffix in _SIBLINGS:
            add(p.with_suffix(suffix))
        if p.suffix.lower() in (".gltf", ".obj"):
            for q in _model_files(p):
                add(q)
    drop = {Path(x).resolve() for x in skip}
    if getattr(scene, "file", None):
        drop.add(Path(scene.file).resolve())
    return sorted(p for p in found if p not in drop and _PACKAGE not in p.parents)


def _tools() -> dict:
    import platform

    import skia

    from . import media
    try:
        first = media._ffmpeg_says("-version").splitlines()[0]
        ff = re.sub(r"^ffmpeg version\s+", "", first).split(" Copyright")[0].strip() or None
    except (OSError, IndexError, AttributeError):
        ff = None
    return {"ffmpeg": ff, "skia-python": getattr(skia, "__version__", None), "python": platform.python_version()}


def _scene_entry(scene, where: Path) -> dict:
    f = getattr(scene, "file", None)
    if f and Path(f).is_file():
        return {"file": _rel(Path(f).resolve(), where), "sha256": sha256(f)}
    # A scene given as a dict or a JSON string has no file: its parsed JSON is hashed, keys sorted.
    text = json.dumps(scene.source if getattr(scene, "source", None) is not None else scene.spec, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return {"file": None, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def write(scene, result: dict, settings: dict) -> str:
    """Write the receipt of a finished render (`result` is what render returned) and return its path."""
    from . import __version__
    from .render import backend_setting
    out = Path(result["out"]).resolve()
    rp = path_for(out)
    base = Path(scene.base).resolve()
    doc = {
        "receipt": FORMAT,
        "kinescript": __version__,
        "out": {"file": out.name, "sha256": sha256(out), "bytes": out.stat().st_size,
                **{k: result[k] for k in ("w", "h", "fps", "frames", "seconds", "audio", "poster") if k in result}},
        "render": {**settings, "backend": backend_setting()},
        "scene": _scene_entry(scene, rp.parent),
        "inputs": {_rel(p, base): {"sha256": sha256(p), "bytes": p.stat().st_size}
                   for p in inputs(scene, skip=(out, rp))},
        "tools": _tools(),
    }
    rp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(rp)


def check(target, receipt=None) -> dict:
    """Compare a rendered file with its receipt. `target` is the video (its receipt is beside it) or the receipt
    itself. ok is true when the output, the scene and every input are as recorded and the scene reads no file the
    receipt does not list; a different Kinescript or ffmpeg is a note, since it may draw the same scene differently."""
    from . import __version__
    t = Path(target)
    rp = Path(receipt) if receipt else t if t.name.endswith(SUFFIX) else path_for(t)
    if not rp.is_file():
        raise ValueError(f"no receipt at {rp}; kine render writes one beside each video or sound file")
    try:
        doc = json.loads(rp.read_text(encoding="utf-8"))
        rec_out, rec_scene, rec_inputs = doc["out"], doc["scene"], doc["inputs"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{rp} is not a Kinescript receipt ({e})") from None
    where = rp.parent
    out_p = t if t != rp else where / rec_out["file"]
    res: dict = {"ok": True, "receipt": str(rp), "out": _state(out_p, rec_out.get("sha256")), "scene": None,
                 "inputs": {"changed": [], "missing": [], "added": [], "same": 0},
                 "kinescript": {"receipt": doc.get("kinescript"), "installed": __version__}, "notes": []}
    scene_file = where / rec_scene["file"] if rec_scene.get("file") else None
    if scene_file is None:
        res["scene"] = "unknown"
        res["notes"].append("the scene was given as data, not a file, so only its inputs are compared")
    else:
        res["scene"] = _state(scene_file, rec_scene.get("sha256"))
    base = None
    if scene_file is not None and scene_file.is_file():
        from .scene import SceneError, load
        try:
            sc = load(str(scene_file))
        except (SceneError, ValueError, OSError) as e:
            res["notes"].append(f"the scene does not load now, so files it reads that the receipt lacks are not "
                                f"looked for: {e}")
        else:
            try:
                base = Path(sc.base).resolve()
                now = {_rel(p, base) for p in inputs(sc, skip=(out_p, rp))}
                res["inputs"]["added"] = sorted(now - set(rec_inputs))
            finally:
                sc.close()
        base = base or scene_file.resolve().parent
    else:
        base = where.resolve()
    for name, rec in sorted(rec_inputs.items()):
        state = _state(base / name, rec.get("sha256") if isinstance(rec, dict) else None)
        if state == "same":
            res["inputs"]["same"] += 1
        else:
            res["inputs"][state].append(name)
    if doc.get("kinescript") != __version__:
        res["notes"].append(f"rendered by Kinescript {doc.get('kinescript')}; this is {__version__}, which may draw "
                            "the same scene differently")
    i = res["inputs"]
    res["ok"] = res["out"] == "same" and res["scene"] in ("same", "unknown") and not (
        i["changed"] or i["missing"] or i["added"])
    return res


def _state(p: Path, want) -> str:
    if not p.is_file():
        return "missing"
    return "same" if want and sha256(p) == want else "changed"
