"""Brands: a folder holding brand.json (with its fonts/ and logos/) that scenes name with "brand", so many videos
share one set of colours, fonts, text styles, chart colours and logos.

A brand is read, checked and flattened into theme values once, when a scene loads; frames only ever see the result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import color as C

VERSION = 1
FILE = "brand.json"
ROLES = ("bg", "surface", "text", "muted", "accent", "accent2", "positive", "negative")
REQUIRED = ("bg", "text", "accent")
FALLBACK = {"surface": "bg", "muted": "text", "accent2": "accent", "positive": "accent", "negative": "accent"}
# The theme names scenes and built-ins already use, each filled from a palette role.
ALIASES = {"void": "bg", "ink": "bg", "sheet": "text", "paper": "text", "track": "surface", "cue": "accent",
           "key": "accent2", "annotation": "muted"}
FACES = ("display", "body", "mono")
# Parts whose entries become "$part.name" theme values (and "$part" for the whole object).
GROUPS = ("text", "charts", "logos", "placements", "bug", "motion", "components", "intro", "outro")
BOOKENDS = {"intro": ("sting", 2.0), "outro": ("endCard", 3.0)}  # the built-in that draws each, its default dur
PARTS = {"kinescriptBrand", "id", "name", "description", "palettes", "default", "fonts", "safeArea", "lock", *GROUPS}
LOCKABLE = ("palettes", "fonts", *GROUPS, "safeArea")
LOGO_FILES = ("file", "onDark", "onLight")
FONT_EXTS = (".ttf", ".otf", ".ttc")
LOGO_EXTS = (".svg", ".png", ".jpg", ".jpeg", ".webp")


class BrandError(ValueError):
    """A brand that cannot be used. `issues` lists every problem found."""

    def __init__(self, folder: Path, issues: list[dict]):
        errors = [i for i in issues if i["level"] == "error"]
        super().__init__(f"brand {folder}: " + "; ".join(i["message"] for i in errors))
        self.issues = issues


@dataclass(frozen=True)
class Brand:
    folder: Path
    spec: dict
    digest: str
    issues: list = field(default_factory=list)  # warnings and info from validate

    @property
    def lock(self) -> set[str]:
        return set(self.spec.get("lock") or [])

    def fonts(self) -> dict[str, list[tuple[str, int, bool]]]:
        """Each face family with files: [(absolute path, weight, italic)]."""
        out: dict[str, list[tuple[str, int, bool]]] = {}
        for face in (self.spec.get("fonts") or {}).values():
            if isinstance(face, dict) and face.get("files"):
                out.setdefault(face["family"], []).extend(
                    (str(inside(self.folder, f["file"])), int(f.get("weight", 400)), bool(f.get("italic", False)))
                    for f in face["files"])
        return out


def inside(folder: Path, rel) -> Path:
    """`rel` resolved in `folder`. ValueError when it is absolute or leads out of the folder (.. or a symlink)."""
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(f"{rel!r} is not a file name")
    p = Path(rel)
    if p.is_absolute() or p.drive:
        raise ValueError(f"'{rel}' is an absolute path; name a file inside the brand folder, such as \"fonts/a.ttf\"")
    root = folder.resolve()
    full = (root / p).resolve()
    if full != root and root not in full.parents:
        raise ValueError(f"'{rel}' leads out of the brand folder")
    return full


def _files(spec: dict) -> list[tuple[str, str, tuple]]:
    """Every file the brand names: (where, relative path, extensions it may have)."""
    out = []
    for name, face in (spec.get("fonts") or {}).items() if isinstance(spec.get("fonts"), dict) else ():
        files = face.get("files") if isinstance(face, dict) else None
        for i, f in enumerate(files if isinstance(files, list) else []):
            if isinstance(f, dict):
                out.append((f"fonts.{name}.files[{i}]", f.get("file"), FONT_EXTS))
    logos = spec.get("logos") if isinstance(spec.get("logos"), dict) else {}
    for name, logo in logos.items():
        if isinstance(logo, dict):
            out += [(f"logos.{name}.{k}", logo[k], LOGO_EXTS) for k in LOGO_FILES if k in logo]
    for part in BOOKENDS:
        if isinstance(spec.get(part), dict) and "scene" in spec[part]:
            out.append((f"{part}.scene", spec[part]["scene"], (".json",)))
    return out


def read(folder: str | Path) -> dict:
    """brand.json's object. ValueError when it is missing or not a JSON object."""
    f = Path(folder) / FILE
    if not f.is_file():
        raise ValueError(f"no {FILE} in {folder}")
    try:
        spec = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{FILE} is not valid JSON: {e}") from None
    if not isinstance(spec, dict):
        raise ValueError(f"{FILE} must hold an object")
    return spec


def digest(folder: str | Path, spec: dict | None = None) -> str:
    """sha256 over brand.json and every file it names (in name order), so any change to the brand changes it."""
    folder = Path(folder)
    spec = read(folder) if spec is None else spec
    h = hashlib.sha256((folder / FILE).read_bytes())
    for rel in sorted({r for _, r, _ in _files(spec) if isinstance(r, str)}):
        try:
            p = inside(folder, rel)
        except ValueError:
            continue
        h.update(rel.encode() + b"\0")
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()


def _issue(level: str, code: str, message: str, fix: str = "") -> dict:
    return {"level": level, "code": code, "layer": "brand", "message": message, "fix": fix}


def validate(folder: str | Path, spec: dict | None = None) -> list[dict]:
    """Every problem with the brand in `folder`, as check issues (level, code, message, fix)."""
    folder = Path(folder)
    out: list[dict] = []
    if spec is None:
        try:
            spec = read(folder)
        except ValueError as e:
            return [_issue("error", "brand-file", str(e), f"run kine brand init, or point \"brand\" at a folder "
                                                          f"holding {FILE}")]
    ver = spec.get("kinescriptBrand")
    if not isinstance(ver, int) or isinstance(ver, bool):
        out.append(_issue("error", "brand-version", f"{FILE} has no \"kinescriptBrand\" version number",
                          f"add \"kinescriptBrand\": {VERSION}"))
    elif ver > VERSION:
        out.append(_issue("error", "brand-version", f"the brand is version {ver}; this Kinescript reads up to "
                                                    f"{VERSION}", "upgrade Kinescript"))
    for k in sorted(set(spec) - PARTS):
        if not str(k).startswith(("_", "//")):
            out.append(_issue("warning", "brand-key", f"{FILE} has an unknown part '{k}'",
                              "known parts: " + ", ".join(sorted(PARTS))))
    pals = spec.get("palettes")
    if not isinstance(pals, dict) or not pals:
        out.append(_issue("error", "brand-palette", "the brand has no palettes",
                          "add \"palettes\": {\"dark\": {\"bg\": \"#0a0c10\", \"text\": \"#f2f0e8\", "
                          "\"accent\": \"#ff633f\"}}"))
        pals = {}
    for name, pal in pals.items():
        if not isinstance(pal, dict):
            out.append(_issue("error", "brand-palette", f"palettes.{name} must be an object of role colours"))
            continue
        for role in REQUIRED:
            if role not in pal:
                out.append(_issue("error", "brand-palette", f"palettes.{name} has no '{role}'",
                                  f"every palette sets {', '.join(REQUIRED)}"))
        for role, v in pal.items():
            if role not in ROLES:
                out.append(_issue("warning", "brand-palette", f"palettes.{name}.{role} is not a palette role",
                                  "roles: " + ", ".join(ROLES)))
            elif (why := C.problem(v)) is not None:
                out.append(_issue("error", "bad-colour", f"palettes.{name}.{role}: {why}", "use #rrggbb"))
    default = spec.get("default")
    if default is not None and default not in pals:
        out.append(_issue("error", "brand-variant", f"default is '{default}', but there is no palettes.{default}",
                          "name one of: " + ", ".join(pals)))
    fonts = spec.get("fonts", {})
    if not isinstance(fonts, dict):
        out.append(_issue("error", "brand-font", "fonts must be an object of display, body and mono faces"))
        fonts = {}
    for name, face in fonts.items():
        if name not in FACES:
            out.append(_issue("warning", "brand-font", f"fonts.{name} is not a face", "faces: " + ", ".join(FACES)))
        if not isinstance(face, dict) or not isinstance(face.get("family"), str):
            out.append(_issue("error", "brand-font", f"fonts.{name} needs a \"family\" name"))
            continue
        files = face.get("files", [])
        if not isinstance(files, list) or not all(isinstance(f, dict) and "file" in f for f in files):
            out.append(_issue("error", "brand-font", f"fonts.{name}.files must be a list like "
                                                     "[{\"file\": \"fonts/A-Bold.ttf\", \"weight\": 700}]"))
        elif files and not face.get("license"):
            out.append(_issue("warning", "brand-font-license", f"fonts.{name} ships font files but names no license",
                              "add \"license\": \"OFL\" (or the font's own), so people know they may use it"))
    for g in GROUPS:
        if g in spec and not isinstance(spec[g], dict):
            out.append(_issue("error", "brand-key", f"{g} must be an object"))
    for part in BOOKENDS:
        dur = spec.get(part, {}).get("dur") if isinstance(spec.get(part), dict) else None
        if dur is not None and (isinstance(dur, bool) or not isinstance(dur, (int, float)) or not 0 < dur <= 60):
            out.append(_issue("error", "brand-key", f"{part}.dur is {dur!r}; it is seconds, above 0 and at most 60"))
    lock = spec.get("lock", [])
    if not isinstance(lock, list):
        out.append(_issue("error", "brand-lock", "lock must be a list such as [\"palettes\", \"fonts\"]"))
    else:
        for part in lock:
            if part not in LOCKABLE:
                out.append(_issue("error", "brand-lock", f"lock names '{part}', which cannot be locked",
                                  "lockable: " + ", ".join(LOCKABLE)))
    out += _contrast(pals)
    charts = spec.get("charts") if isinstance(spec.get("charts"), dict) else {}
    series = charts.get("series") if isinstance(charts.get("series"), list) else []
    for name, pal in pals.items():
        bg = pal.get("bg") if isinstance(pal, dict) else None
        for i, c in enumerate(series):
            while isinstance(c, str) and c in ROLES:  # a role, or the one it falls back to
                c = pal.get(c) or FALLBACK.get(c, "bg")
                if c == "bg":
                    break
            if C.problem(bg) is None and C.problem(c) is None and (r := _ratio(c, bg)) < 3.0:
                out.append(_issue("warning", "brand-chart-contrast", f"charts.series[{i}] {c} is {r:.1f}:1 against "
                                  f"palettes.{name}.bg; chart colours want 3:1 to be told from the background",
                                  "darken or lighten it, or use another variant for charts"))
    for where, rel, exts in _files(spec):
        try:
            p = inside(folder, rel)
        except ValueError as e:
            out.append(_issue("error", "brand-path", f"{where}: {e}"))
            continue
        if p.suffix.lower() not in exts:
            out.append(_issue("error", "brand-file", f"{where} is '{rel}'; it must be one of {', '.join(exts)}"))
        elif not p.is_file():
            out.append(_issue("error", "missing-media", f"{where}: no file at {p}", "add the file, or fix the name"))
    return out


# Pairs a brand promises to draw together, and the contrast each needs (WCAG: 4.5 for text, 3 for large type and
# graphics).
PAIRS = (("text", "bg", 4.5), ("text", "surface", 4.5), ("muted", "bg", 3.0), ("accent", "bg", 3.0))


def _ratio(a: str, b: str) -> float:
    return C.contrast(C.luminance(C.parse(a)), C.luminance(C.parse(b)))


def _contrast(pals: dict) -> list[dict]:
    out = []
    for name, pal in pals.items():
        if not isinstance(pal, dict):
            continue
        for fg, bg, want in PAIRS:
            a, b = pal.get(fg), pal.get(bg)
            if a is None or b is None or C.problem(a) or C.problem(b):
                continue
            if (r := _ratio(a, b)) < want:
                out.append(_issue("warning", "brand-contrast", f"palettes.{name}: {fg} {a} on {bg} {b} is {r:.1f}:1; "
                                  f"{fg} on {bg} wants {want:g}:1", f"darken or lighten {fg} or {bg}"))
    return out


def load(folder: str | Path) -> Brand:
    """The brand in `folder`, checked. BrandError lists its errors."""
    folder = Path(folder)
    issues = validate(folder)
    if any(i["level"] == "error" for i in issues):
        raise BrandError(folder, issues)
    spec = read(folder)
    return Brand(folder.resolve(), spec, digest(folder, spec), issues)


def variants(brand: Brand) -> list[str]:
    return list(brand.spec["palettes"])


def resolve(brand: Brand, variant: str | None = None) -> tuple[dict, dict[str, str]]:
    """The theme values the brand gives in `variant` (its default, else its first palette), and which brand part
    each theme name comes from (for lock). ValueError for a variant the brand does not have."""
    pals = brand.spec["palettes"]
    name = variant or brand.spec.get("default") or next(iter(pals))
    if name not in pals:
        raise ValueError(f"variant is '{name}', but the brand's palettes are: {', '.join(pals)}")
    pal = dict(pals[name])
    for role, other in FALLBACK.items():
        pal.setdefault(role, pal.get(other))
    theme: dict = {}
    owner: dict[str, str] = {}

    def put(k, v, part):
        theme[k] = v
        owner[k] = part

    for role in ROLES:
        put(role, pal[role], "palettes")
    for alias, role in ALIASES.items():
        put(alias, pal[role], "palettes")
    put("variant", name, "palettes")
    if isinstance(brand.spec.get("name"), str):
        put("brandName", brand.spec["name"], "name")
    fonts = brand.spec.get("fonts") or {}
    fam = {f: fonts[f]["family"] for f in FACES if f in fonts}
    if fam:
        put("font", fam.get("body") or fam.get("display"), "fonts")
        put("fontDisplay", fam.get("display") or fam.get("body"), "fonts")
    if "mono" in fam:
        put("mono", fam["mono"], "fonts")
    for g in GROUPS:
        part = brand.spec.get(g)
        if not part:
            continue
        if g == "logos":
            part = {k: _logo_paths(brand.folder, v) for k, v in part.items()}
        for k, v in part.items():  # dotted only: "text" alone is the palette role
            put(f"{g}.{k}", v, g)
    if "safeArea" in brand.spec:
        put("safeArea", brand.spec["safeArea"], "safeArea")
    return theme, owner


def _logo_paths(folder: Path, logo):
    """A logo entry with its files as absolute paths, so layers in any scene folder find them."""
    if not isinstance(logo, dict):
        return logo
    return {k: (str(inside(folder, v)) if k in LOGO_FILES else v) for k, v in logo.items()}
