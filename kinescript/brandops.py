"""The operations on brands that the CLI (kine brand), the MCP tool (kine_brand) and the brand wizard all call, so the
same change made through any of them writes the same files.

A brand lives in the library (~/.kinescript/brands/<id>, or $KINE_BRANDS/<id>) and is copied into a project's brand/
folder by use(), which records the copy's digest so a later use() can tell local edits from the library's.
Every write takes the revision (digest) the caller last read and refuses when the brand changed since.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from xml.etree import ElementTree

from . import brand as B

SOURCE = ".kinescript-source.json"  # in a project's copy: the library id and the digest it was copied at
KINDS = {"fonts": (B.FONT_EXTS, 20 << 20), "logos": (B.LOGO_EXTS, 10 << 20), "audio": (B.AUDIO_EXTS, 50 << 20)}
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")

DEFAULT = {
    "kinescriptBrand": B.VERSION, "id": "kinescript", "name": "Kinescript", "default": "dark",
    "palettes": {
        "dark": {"bg": "#0a0c10", "surface": "#242832", "text": "#f2f0e8", "muted": "#9299a8", "accent": "#ff633f",
                 "accent2": "#b9f45d"},
        "light": {"bg": "#f2f0e8", "surface": "#e2dfd4", "text": "#0a0c10", "muted": "#5b6272", "accent": "#c8411f",
                  "accent2": "#3d6b12"},
    },
    "fonts": {"display": {"family": "Space Grotesk"}, "body": {"family": "Space Grotesk"},
              "mono": {"family": "IBM Plex Mono"}},
    "charts": {"series": ["accent", "accent2", "text", "muted"], "highlight": "accent"},
    "logos": {"clearSpace": 0.25, "minHeight": 32},
    "placements": {"corner": {"anchor": "bottomright", "margin": "5%", "height": "7%"}},
    "bug": {"placement": "corner", "opacity": 0.85},
    "motion": {"enter": "rise", "exit": "fade"},
}


class BrandOpError(ValueError):
    """A brand operation refused, with the reason a person or agent can act on."""


def library() -> Path:
    return Path(os.environ.get("KINE_BRANDS") or Path.home() / ".kinescript" / "brands")


def locate(ref: str | Path) -> Path:
    """A brand folder: `ref` when it is a folder holding brand.json (or a path), else the library brand of that id."""
    p = Path(ref)
    if (p / B.FILE).is_file() or p.is_absolute() or len(p.parts) > 1:
        return p
    return library() / str(ref)


def _check_id(bid: str) -> str:
    if not isinstance(bid, str) or not _ID.fullmatch(bid):
        raise BrandOpError(f"brand id {bid!r}: use lower-case letters, digits and dashes, such as \"acme\"")
    return bid


def _rev(folder: Path) -> str:
    return B.digest(folder)


def _guard(folder: Path, base_rev: str | None) -> None:
    if not (folder / B.FILE).is_file():
        raise BrandOpError(f"no brand at {folder}: run kine brand init, or name a folder holding {B.FILE}")
    if base_rev is not None and base_rev != _rev(folder):
        raise BrandOpError("the brand changed since you read it (its rev differs); read it again and redo the change")


def _write_json(path: Path, obj) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _errors(issues: list[dict]) -> list[dict]:
    return [i for i in issues if i["level"] == "error"]


def show(ref) -> dict:
    """The brand as stored, its rev, where it is, and its issues."""
    folder = locate(ref)
    _guard(folder, None)
    files = sorted(str(p.relative_to(folder)).replace("\\", "/") for kind in KINDS if (folder / kind).is_dir()
                   for p in (folder / kind).iterdir() if p.is_file())
    out = {"path": str(folder.resolve()), "rev": _rev(folder), "brand": B.read(folder), "files": files,
           "issues": B.validate(folder)}
    src = folder / SOURCE
    if src.is_file():
        out["source"] = json.loads(src.read_text(encoding="utf-8"))
        out["edited"] = out["source"].get("digest") != out["rev"]
    return out


def list_library() -> list[dict]:
    root = library()
    out = []
    for d in sorted(root.iterdir()) if root.is_dir() else ():
        if (d / B.FILE).is_file():
            try:
                spec = B.read(d)
            except ValueError as e:
                out.append({"id": d.name, "path": str(d), "error": str(e)})
                continue
            out.append({"id": d.name, "name": spec.get("name", d.name), "path": str(d), "rev": _rev(d),
                        "variants": list(spec.get("palettes") or {})})
    return out


def init(bid: str, folder: str | Path | None = None, source: str | Path | None = None) -> dict:
    """A new brand `bid` in the library (or `folder`), copied from `source` (an id or folder) or the default."""
    _check_id(bid)
    dest = Path(folder) if folder else library() / bid
    if (dest / B.FILE).exists():
        raise BrandOpError(f"a brand is already at {dest}; pick another id, or edit that one")
    if source is not None:
        src = locate(source)
        _guard(src, None)
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(SOURCE, ".tmp-*"))
        spec = B.read(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        spec = copy.deepcopy(DEFAULT)
    spec["id"] = bid
    spec["name"] = "Kinescript" if bid == "kinescript" else bid.replace("-", " ").title()
    _write_json(dest / B.FILE, spec)
    return show(dest)


def use(ref, project: str | Path = ".", force: bool = False) -> dict:
    """Copy a brand into `project`/brand and record its digest. Refuses to overwrite a copy that was edited in the
    project since it was copied, unless force."""
    src = locate(ref)
    _guard(src, None)
    dest = Path(project) / "brand"
    if (dest / B.FILE).is_file() and not force:
        recorded = (json.loads((dest / SOURCE).read_text(encoding="utf-8")).get("digest")
                    if (dest / SOURCE).is_file() else None)
        if recorded != _rev(dest):
            raise BrandOpError(f"{dest} has changes made in this project since it was copied; use force to replace "
                               "them with the library's brand, or copy them to the library first")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(SOURCE, ".tmp-*"))
    _write_json(dest / SOURCE, {"id": B.read(src).get("id", src.name), "from": str(src.resolve()),
                                "digest": _rev(dest)})
    return show(dest)


def _split(path: str) -> list[str]:
    """"palettes.dark.accent" or "/palettes/dark/accent" as keys; numbers index lists."""
    parts = [p for p in re.split(r"[./]", path.strip()) if p]
    if not parts:
        raise BrandOpError("path is empty; name a part such as \"palettes.dark.accent\"")
    return parts


def set_value(ref, path: str, value, base_rev: str | None = None) -> dict:
    """Set (or with value None, remove) one value in brand.json. Refuses a change that makes the brand unusable."""
    folder = locate(ref)
    _guard(folder, base_rev)
    spec = B.read(folder)
    keys = _split(path)
    node = spec
    for k in keys[:-1]:
        if isinstance(node, list):
            node = node[int(k)]
        else:
            node = node.setdefault(k, {})
        if not isinstance(node, (dict, list)):
            raise BrandOpError(f"{path}: '{k}' holds {node!r}, not an object")
    last = keys[-1]
    if isinstance(node, list):
        i = int(last)
        if value is None:
            node.pop(i)
        else:
            node[i:i + 1] = [value]
    elif value is None:
        node.pop(last, None)
    else:
        node[last] = value
    before = {(i["code"], i["message"]) for i in _errors(B.validate(folder))}
    after = [i for i in _errors(B.validate(folder, spec)) if (i["code"], i["message"]) not in before]
    if after:
        raise BrandOpError(f"{path}: " + "; ".join(i["message"] for i in after))
    _write_json(folder / B.FILE, spec)
    return show(folder)


def replace(ref, spec: dict, base_rev: str | None = None) -> dict:
    """Write a whole brand.json at once (the wizard's raw editor). Refuses one that adds errors, as set_value does."""
    folder = locate(ref)
    _guard(folder, base_rev)
    if not isinstance(spec, dict):
        raise BrandOpError("the brand must be a JSON object")
    before = {(i["code"], i["message"]) for i in _errors(B.validate(folder))}
    after = [i for i in _errors(B.validate(folder, spec)) if (i["code"], i["message"]) not in before]
    if after:
        raise BrandOpError("; ".join(i["message"] for i in after))
    _write_json(folder / B.FILE, spec)
    return show(folder)


def _sniff(kind: str, name: str, data: bytes) -> None:
    ext = Path(name).suffix.lower()
    exts, cap = KINDS[kind]
    if ext not in exts:
        raise BrandOpError(f"{name}: {kind} take {', '.join(exts)} files")
    if not data:
        raise BrandOpError(f"{name} is empty")
    if len(data) > cap:
        raise BrandOpError(f"{name} is {len(data) >> 20} MB; {kind} take up to {cap >> 20} MB")
    if kind == "fonts":
        ok = data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf")
    elif kind == "audio":
        ok = {".wav": data[:4] == b"RIFF" and data[8:12] == b"WAVE",
              ".mp3": data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0),
              ".flac": data[:4] == b"fLaC", ".ogg": data[:4] == b"OggS", ".opus": data[:4] == b"OggS",
              ".m4a": data[4:8] == b"ftyp",
              ".aac": len(data) > 1 and data[0] == 0xFF and data[1] & 0xF0 == 0xF0}.get(ext, False)
    elif ext == ".png":
        ok = data[:8] == b"\x89PNG\r\n\x1a\n"
    elif ext in (".jpg", ".jpeg"):
        ok = data[:3] == b"\xff\xd8\xff"
    elif ext == ".webp":
        ok = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    else:
        ok = _safe_svg(name, data)
    if not ok:
        raise BrandOpError(f"{name} is not a {ext} file (its first bytes do not match)")


def _safe_svg(name: str, data: bytes) -> bool:
    if b"<!ENTITY" in data or b"<!DOCTYPE" in data:
        raise BrandOpError(f"{name}: an svg with a DOCTYPE or entities is refused; save it again without them")
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as e:
        raise BrandOpError(f"{name} is not valid svg: {e}") from None
    if not root.tag.endswith("svg"):
        return False
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1].lower()
        if tag in ("script", "foreignobject"):
            raise BrandOpError(f"{name} has a <{tag}>; a logo svg may hold only drawing")
        for k, v in el.attrib.items():
            key = k.rsplit("}", 1)[-1].lower()
            if key.startswith("on"):
                raise BrandOpError(f"{name} has an {key} handler; a logo svg may hold only drawing")
            if key == "href" and not v.startswith("#"):
                raise BrandOpError(f"{name} links to {v!r}; a logo svg may link only inside itself")
    return True


def add_file(ref, kind: str, name: str, data: bytes, base_rev: str | None = None, replace: bool = False) -> dict:
    """Store a font or logo file in the brand (fonts/<name> or logos/<name>), after checking it is that kind of file.
    It does not name the file in brand.json; set_value does."""
    if kind not in KINDS:
        raise BrandOpError(f"kind is {kind!r}; it is fonts or logos")
    folder = locate(ref)
    _guard(folder, base_rev)
    base = str(name)
    if not base or "/" in base or "\\" in base or base.startswith(".") or ":" in base:
        raise BrandOpError(f"name {name!r}: give a plain file name such as \"acme.svg\"")
    _sniff(kind, base, data)
    dest = folder / kind / base
    if dest.exists() and not replace:
        raise BrandOpError(f"{kind}/{base} exists; pass replace to overwrite it")
    dest.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return {**show(folder), "file": f"{kind}/{base}"}


def remove_file(ref, rel: str, base_rev: str | None = None) -> dict:
    """Delete a font or logo file the brand no longer names."""
    folder = locate(ref)
    _guard(folder, base_rev)
    rel = str(rel).replace("\\", "/")
    parts = rel.split("/")
    if len(parts) != 2 or parts[0] not in KINDS or parts[1] in ("", ".", ".."):
        raise BrandOpError(f"{rel}: only files under fonts/, logos/ or audio/ can be removed, such as logos/acme.svg")
    try:
        p = B.inside(folder, rel)
    except ValueError as e:
        raise BrandOpError(str(e)) from None
    used = [w for w, r, _ in B._files(B.read(folder)) if r == rel]
    if used:
        raise BrandOpError(f"{rel} is still named by {', '.join(used)}; change that first")
    if not p.is_file():
        raise BrandOpError(f"no file {rel} in the brand")
    p.unlink()
    return show(folder)


ACTIONS = ("list", "show", "init", "use", "set", "replace", "add_file", "remove_file", "add_avatar", "check", "preview",
           "import")


def add_avatar(ref, name: str, copy: bool = False, base_rev: str | None = None) -> dict:
    """Name an avatar in the brand: the starter (kine:host), or with copy the starter's files in avatars/<name> to
    redraw. A scene then uses it as "avatar": "brand:<name>"."""
    from . import avatar as A
    folder = locate(ref)
    _guard(folder, base_rev)
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise BrandOpError(f"avatar name {name!r}: lower-case letters, digits and -, such as host")
    spec = B.read(folder)
    avs = spec.get("avatars") if isinstance(spec.get("avatars"), dict) else {}
    if name in avs:
        raise BrandOpError(f"the brand already has an avatar '{name}'; pick another name, or set avatars.{name}")
    src = "kine:host"
    if copy:
        try:
            A.new(folder / "avatars" / name)
        except ValueError as e:
            raise BrandOpError(str(e)) from None
        src = f"avatars/{name}"
    spec["avatars"] = {**avs, name: {"src": src}}
    _write_json(folder / B.FILE, spec)
    return show(folder)


def _need(args: dict, key: str):
    v = args.get(key)
    if v is None or v == "":
        raise BrandOpError(f"'{key}' is needed for this action")
    return v


def run(action: str, args: dict) -> dict:
    """One brand operation by name, as kine brand and kine_brand give them; a BrandOpError says why one refused."""
    if action == "list":
        return {"library": str(library()), "brands": list_library()}
    if action == "show":
        return show(_need(args, "brand"))
    if action == "init":
        return init(_need(args, "id"), args.get("dir"), args.get("from"))
    if action == "use":
        return use(_need(args, "brand"), args.get("project") or ".", bool(args.get("force")))
    if action == "set":
        if "value" not in args:
            raise BrandOpError("'value' is needed (null removes the entry)")
        return set_value(_need(args, "brand"), _need(args, "path"), args["value"], args.get("base_rev"))
    if action == "replace":
        spec = args.get("spec")
        if spec is None and args.get("file"):
            spec = json.loads(Path(args["file"]).read_text(encoding="utf-8"))
        if spec is None:
            raise BrandOpError("'spec' (the whole brand.json) is needed")
        return replace(_need(args, "brand"), spec, args.get("base_rev"))
    if action == "add_file":
        import base64
        src = args.get("file")
        if src:
            data, name = Path(src).read_bytes(), args.get("name") or Path(src).name
        else:
            data, name = base64.b64decode(_need(args, "data_base64")), _need(args, "name")
        return add_file(_need(args, "brand"), _need(args, "kind"), name, data, args.get("base_rev"),
                        bool(args.get("replace")))
    if action == "remove_file":
        return remove_file(_need(args, "brand"), _need(args, "file"), args.get("base_rev"))
    if action == "add_avatar":
        return add_avatar(_need(args, "brand"), _need(args, "name"), bool(args.get("copy")), args.get("base_rev"))
    if action == "check":
        issues = B.validate(locate(_need(args, "brand")))
        return {"ok": not _errors(issues), "issues": issues}
    if action == "preview":
        return preview(_need(args, "brand"), _need(args, "out"), args.get("variant"))
    if action == "import":
        return import_palette(_need(args, "brand"), args.get("logo"), args.get("file"), bool(args.get("apply")),
                              args.get("base_rev"))
    raise BrandOpError(f"action is {action!r}; it is one of: {', '.join(ACTIONS)}")


def import_palette(ref, logo: str | None = None, file: str | None = None, apply: bool = False,
                   base_rev: str | None = None) -> dict:
    """Palettes proposed from a logo: `logo` a file in the brand (logos/mark.svg), or `file` a path. With apply, they
    (and the chart series) are written to the brand, replacing its palettes; otherwise nothing changes."""
    from . import palette
    folder = locate(ref)
    _guard(folder, base_rev)
    if logo:
        try:
            src = B.inside(folder, logo)
        except ValueError as e:
            raise BrandOpError(str(e)) from None
    elif file:
        src = Path(file)
    else:
        raise BrandOpError("name the logo: 'logo' for a file in the brand (logos/mark.svg), or 'file' for a path")
    try:
        found = palette.propose(palette.logo_colours(src), B.read(folder).get("name"))
    except ValueError as e:
        raise BrandOpError(str(e)) from None
    if not apply:
        return {"proposal": found, "applied": False}
    spec = B.read(folder)
    spec["palettes"] = found["palettes"]
    if spec.get("default") not in found["palettes"]:
        spec["default"] = "dark"
    spec.setdefault("charts", {})
    if isinstance(spec["charts"], dict):
        spec["charts"]["series"] = found["charts"]["series"]
    out = replace(folder, spec)
    return {**out, "proposal": found, "applied": True}


def sample_scene(folder: Path, variant: str | None = None, w: int = 1280, h: int = 720) -> dict:
    """A short scene that shows the brand's palette, text styles, a chart, a lower third and its logos."""
    spec = B.read(folder)
    entries = [k for k, v in (spec.get("logos") or {}).items() if isinstance(v, dict)]
    name = str(spec.get("name") or spec.get("id") or "Brand")
    swatches = [{"id": f"sw-{r}", "type": "rect", "x": 80 + i * 120, "y": 90, "w": 100, "h": 60, "anchor": "topleft",
                 "radius": 8, "fill": f"${r}", "stroke": "$muted", "strokeWidth": 1}
                for i, r in enumerate(("bg", "surface", "text", "muted", "accent", "accent2"))]
    layers = swatches + [
        {"id": "title", "type": "text", "text": name, "textStyle": "title", "size": max(28, min(64, 900 // len(name))), "x": 80, "y": 200, "anchor": "topleft", "align": "left", "fill": "$text", "maxWidth": 640},
        {"id": "body", "type": "text", "text": "Body text in the brand's body face.", "textStyle": "body", "size": 28,
         "x": 80, "y": 300, "anchor": "topleft", "align": "left", "fill": "$text"},
        {"id": "caption", "type": "text", "text": "Caption style", "textStyle": "caption", "size": 20, "x": 80,
         "y": 350, "anchor": "topleft", "align": "left", "fill": "$muted"},
        {"use": "barChart", "id": "chart", "data": [["A", 3], ["B", 5], ["C", 4]], "highlight": "B", "w": 420,
         "h": 220, "size": 20, "x": 960, "y": 300, "grow": 0.01, "stagger": 0},
        {"use": "lowerThird", "id": "lt", "title": "Name Surname", "subtitle": "Role", "size": 40, "x": 80, "y": 470},
    ]
    if entries:
        layers.append({"use": "logo", "id": "logo", "logo": "full" if "full" in entries else entries[0],
                       "placement": "topright", "h": 60})
    out = {"canvas": {"w": w, "h": h, "fps": 30, "duration": 2}, "brand": str(folder.resolve()), "layers": layers}
    if variant:
        out["variant"] = variant
    return out


GALLERY = ("sample", "palette", "type", "titleCard", "lowerThird", "callout", "barChart", "lineChart", "pieChart",
           "logo", "bug", "sting", "endCard")
_DATA = [["North", 42], ["South", 61], ["East", 35], ["West", 50]]


def _note(text: str) -> dict:
    return {"id": "note", "type": "text", "text": text, "size": 40, "x": 960, "y": 540, "fill": "$muted",
            "maxWidth": 1400}


def avatar_scene(folder: Path, name: str, variant: str | None = None) -> dict:
    """A 1920x1080 scene of the brand's avatar `name` in the brand's palette: at rest, talking, blinking, and its
    first expressions and gestures."""
    from . import avatar as A
    try:
        where, _ = B.avatar(folder, name)
        av = A.load(where)
    except ValueError as e:
        raise BrandOpError(str(e)) from None
    theme, _ = B.resolve(B.load(folder), variant)
    poses = [p for p in A.poses(av) if p[0] in ("neutral", "mouth D", "eyes closed")
             or p[0].startswith(("expression", "gesture"))][:6]
    w, h = av.size
    k = 760 / h
    layers = []
    for i, (label, pose) in enumerate(poses):
        x0 = 30 + i * 310
        g = A.placed(av, f"p{i}", pose, theme, x0 + 155 - w * k / 2, 90, k)
        label = label.split(" at ")[0].replace("expression ", "").replace("gesture ", "")
        layers += [g, {"id": f"p{i}-label", "type": "text", "text": label, "x": x0 + 155, "y": 920, "size": 30,
                       "fill": "$muted", "textStyle": "label"}]
    return {"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 3}, "brand": str(folder.resolve()),
            **({"variant": variant} if variant else {}), "layers": layers}


def gallery_scene(folder: Path, what: str, variant: str | None = None) -> dict:
    """A 1920x1080 scene showing one part of the brand (one of GALLERY), drawn at 2.5 s."""
    if what == "sample":
        return sample_scene(folder, variant)  # 1280x720: gallery_png draws it at the same size as the rest
    if what.startswith("avatar:"):
        return avatar_scene(folder, what[7:], variant)
    if what not in GALLERY:
        raise BrandOpError(f"what is {what!r}; it is one of: {', '.join(GALLERY)}")
    from . import components
    spec = B.read(folder)
    logos = [k for k, v in (spec.get("logos") or {}).items() if isinstance(v, dict)]
    if what == "palette":
        roles = B.ROLES
        layers = []
        for i, r in enumerate(roles):
            x, y = 160 + (i % 4) * 420, 200 + (i // 4) * 400
            layers += [{"id": f"sw-{r}", "type": "rect", "x": x, "y": y, "w": 360, "h": 220, "anchor": "topleft",
                        "radius": 16, "fill": f"${r}", "stroke": "$muted", "strokeWidth": 2},
                       {"id": f"lb-{r}", "type": "text", "text": r, "textStyle": "label", "size": 40, "x": x, "y": y + 250,
                        "anchor": "topleft", "align": "left", "fill": "$text"}]
    elif what == "type":
        theme, _ = B.resolve(B.load(folder), variant)
        layers, y = [], 70
        styles = {n: components.text_style(n, theme) for n in components.text_styles(theme)}
        for name, st in sorted(styles.items(), key=lambda kv: -float(kv[1].get("size", 40))):
            size = min(float(st.get("size", 40)), 120)
            layers.append({"id": f"ts-{name}", "type": "text", "text": f"{name}: Brand in motion", "textStyle": name,
                           "size": size, "x": 120, "y": y, "anchor": "topleft", "align": "left", "fill": "$text",
                           "maxWidth": 1680})
            y += size * 1.35 + 24
    elif what == "titleCard":
        layers = [{"use": "titleCard", "id": "tc", "kicker": "Chapter one", "title": "A brand title",
                   "subtitle": "And the line under it"}]
    elif what == "lowerThird":
        layers = [{"use": "lowerThird", "id": "lt", "title": "Name Surname", "subtitle": "Role, Company"}]
    elif what == "callout":
        layers = [{"id": "dot", "type": "circle", "x": 760, "y": 540, "r": 24, "fill": "$surface"},
                  {"use": "callout", "id": "co", "x": 760, "y": 540, "label": "A callout"}]
    elif what == "barChart":
        layers = [{"use": "barChart", "id": "bc", "data": _DATA, "highlight": "South", "w": 1300, "h": 640}]
    elif what == "lineChart":
        layers = [{"use": "lineChart", "id": "lc", "data": _DATA, "w": 1300, "h": 640}]
    elif what == "pieChart":
        layers = [{"use": "pieChart", "id": "pc", "data": _DATA, "r": 320}]
    elif what == "sting":
        layers = [{"use": "sting", "id": "sting"}]
    elif what == "endCard":
        outro = spec.get("outro") if isinstance(spec.get("outro"), dict) else {}
        sample = {} if any(k in outro for k in ("title", "subtitle", "url")) else             {"title": "Thanks for watching", "subtitle": spec.get("name") or "Your brand"}
        layers = [{"use": "endCard", "id": "end", **sample}]
    elif not logos:
        layers = [_note("No logos yet: add a logo file, then name it in the brand's logos.")]
    elif what == "logo":
        places = list(spec.get("placements") or {}) or ["bottomright"]
        layers = [{"use": "logo", "id": f"logo-{p}", "logo": "full" if "full" in logos else logos[0], "placement": p}
                  for p in places]
    else:
        layers = [_note("The bug stays on screen for the whole video."), {"use": "bug", "id": "bug"}]
    out = {"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 3}, "brand": str(folder.resolve()),
           "layers": layers}
    if variant:
        out["variant"] = variant
    return out


def gallery_png(ref, what: str, variant: str | None = None, scale: float = 0.5) -> bytes:
    """gallery_scene drawn to PNG bytes."""
    from .output import still
    folder = locate(ref)
    _guard(folder, None)
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="kine_brand_")
    os.close(fd)
    try:
        still(gallery_scene(folder, what, variant), 1.5 if what == "sample" else 2.5, tmp,
              scale * 1.5 if what == "sample" else scale)
        return Path(tmp).read_bytes()
    finally:
        Path(tmp).unlink(missing_ok=True)


def contrast(ref) -> dict:
    """For each palette: every role against bg and surface, with the ratio a declared pair needs; and the chart
    series against bg (3:1)."""
    folder = locate(ref)
    spec = B.read(folder)
    need = {(a, b): r for a, b, r in B.PAIRS}
    out = {}
    for name, pal in (spec.get("palettes") or {}).items():
        if not isinstance(pal, dict):
            continue

        def col(role, pal=pal):
            seen = role
            while seen not in pal and seen in B.FALLBACK:
                seen = B.FALLBACK[seen]
            return pal.get(seen)
        pairs = []
        for fg in B.ROLES:
            for bg in ("bg", "surface"):
                a, b = col(fg), col(bg)
                if fg == bg or not (isinstance(a, str) and isinstance(b, str)):
                    continue
                try:
                    ratio = round(B._ratio(a, b), 2)
                except ValueError:
                    continue
                req = need.get((fg, bg))
                pairs.append({"fg": fg, "bg": bg, "ratio": ratio, "needs": req,
                              "ok": None if req is None else ratio >= req})
        series = []
        for c in (spec.get("charts") or {}).get("series") or []:
            v = col(c) if isinstance(c, str) and not c.startswith("#") else c
            try:
                ratio = round(B._ratio(v, col("bg")), 2)
            except (ValueError, TypeError, AttributeError):
                continue
            series.append({"color": c, "value": v, "ratio": ratio, "ok": ratio >= 3.0})
        out[name] = {"pairs": pairs, "series": series}
    return out


def compare(ref) -> dict | None:
    """For a project copy: each top-level part as the library has it or not (same|changed|project-only|
    library-only), and which parts the brand locks. None when it is not a copy or the library brand is gone."""
    folder = locate(ref)
    src = folder / SOURCE
    if not src.is_file():
        return None
    lib = Path(json.loads(src.read_text(encoding="utf-8")).get("from") or "")
    if not (lib / B.FILE).is_file():
        return None
    mine, theirs = B.read(folder), B.read(lib)
    parts = {}
    for k in sorted(set(mine) | set(theirs)):
        parts[k] = ("project-only" if k not in theirs else "library-only" if k not in mine
                    else "same" if mine[k] == theirs[k] else "changed")
    return {"library": str(lib), "libraryRev": _rev(lib), "parts": parts, "locked": list(mine.get("lock") or [])}


def preview(ref, out: str | Path, variant: str | None = None, at: float = 1.5) -> dict:
    """A still of sample_scene in the brand, written to `out` (.png or .jpg)."""
    from .output import still
    folder = locate(ref)
    _guard(folder, None)
    return still(sample_scene(folder, variant), at, str(out))
