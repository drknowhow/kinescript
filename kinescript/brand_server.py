"""`kine brand wizard`: a local web page to see, make and change brands.

Every change goes through brandops, the same operations kine brand and kine_brand run, so a change made on the page
writes the same files an agent's call would. The page names a brand as a library id, or "@project" for the project's
brand/ copy; it cannot reach any other folder.
"""
from __future__ import annotations

import base64
import os
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import brand as B
from . import brandops as ops
from .localhttp import BadRequest, LocalHandler

PROJECT = "@project"
ANCHORS = ["topleft", "top", "topright", "left", "center", "right", "bottomleft", "bottom", "bottomright"]
WRITES = ("init", "use", "set", "replace", "add_file", "remove_file", "add_avatar", "import")  # import writes only with apply
_draw_lock = threading.Lock()  # one Renderer draws at a time
FILE_TYPES = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
              ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg"}


def folder_of(ref: str, project: Path) -> Path:
    """The brand folder a page's ref names: the project's copy, or a library brand by id."""
    if ref == PROJECT:
        return project / "brand"
    if not isinstance(ref, str) or not ops._ID.fullmatch(ref):
        raise BadRequest(f"brand {ref!r}: name a library brand by its id, or {PROJECT}")
    return ops.library() / ref


def state(project: Path) -> dict:
    """The library and whether the project has a brand copy."""
    copy = project / "brand"
    return {"library": str(ops.library()), "brands": ops.list_library(), "project": str(project),
            "projectBrand": (copy / B.FILE).is_file(), "gallery": list(ops.GALLERY), "roles": list(B.ROLES),
            "required": list(B.REQUIRED), "fallback": B.FALLBACK, "faces": list(B.FACES),
            "lockable": list(B.LOCKABLE), "anchors": ANCHORS,
            "audioRoles": list(B.AUDIO_ROLES), "idleKeys": ["blink", "breath", "saccade"]}


def brand_info(ref: str, project: Path) -> dict:
    """What the page shows for one brand: show(), the contrast table, the copy-vs-library view, the text styles."""
    from . import components
    folder = folder_of(ref, project)
    out = ops.show(folder)
    out["ref"] = ref
    out["contrast"] = ops.contrast(folder)
    out["compare"] = ops.compare(folder)
    styles = {}
    try:
        brand = B.load(folder)
        out["variants"] = B.variants(brand)
        theme, _ = B.resolve(brand, None)
        styles = {n: components.text_style(n, theme) for n in components.text_styles(theme)}
    except ValueError:
        out["variants"] = list((out["brand"].get("palettes") or {}))
    out["textStyles"] = styles
    return out


def operate(body: dict, project: Path) -> dict:
    """One write from the page, through brandops.run with the page's ref turned into a folder."""
    action = body.get("action")
    if action not in WRITES:
        raise BadRequest(f"action is {action!r}; it is one of: {', '.join(WRITES)}")
    drop = ("dir", "project") if action == "remove_file" else ("file", "dir", "project")  # no local paths
    args = {k: v for k, v in body.items() if k not in drop}
    if action == "init":
        if args.get("from"):
            args["from"] = str(folder_of(args["from"], project))
        out = ops.run("init", args)
        return {**out, "ref": args["id"]}
    folder = folder_of(body.get("ref"), project)
    args["brand"] = str(folder)
    if action == "use":
        if body.get("ref") == PROJECT:
            raise BadRequest("use copies a library brand into the project; pick one from the library")
        args["project"] = str(project)
        return {**ops.run("use", args), "ref": PROJECT}
    if action == "add_file" and not args.get("data_base64"):
        raise BadRequest("send the file as data_base64 with its name")
    return {**ops.run(action, args), "ref": body.get("ref")}


def gallery(ref: str, what: str, variant: str | None, project: Path) -> bytes:
    with _draw_lock:
        return ops.gallery_png(folder_of(ref, project), what, variant or None)


def make_handler(project: Path, start: str | None = None):
    class Handler(LocalHandler):
        server_version = "kine-brand"
        client_errors = (BadRequest, ops.BrandOpError)
        max_body = 32 << 20  # a 20 MB font as base64

        def _get(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                html = resources.files("kinescript.brand_ui").joinpath("index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json({**state(project), "start": start})
            if u.path == "/api/brand":
                return self._json(brand_info(q.get("ref", ""), project))
            if u.path == "/api/gallery":
                png = gallery(q.get("ref", ""), q.get("what", "sample"), q.get("variant"), project)
                return self._send(200, png, "image/png")
            if u.path == "/api/file":
                return self._file(q.get("ref", ""), q.get("path", ""))
            self._json({"error": "not found"}, 404)

        def _file(self, ref, rel):
            folder = folder_of(ref, project)
            if not rel.startswith(("logos/", "audio/")):
                raise BadRequest("only logos and sounds are served")
            try:
                p = B.inside(folder, rel)
            except ValueError as e:
                raise BadRequest(str(e)) from None
            ctype = FILE_TYPES.get(p.suffix.lower())
            if ctype is None or not p.is_file():
                return self._json({"error": f"no logo {rel}"}, 404)
            self._send(200, p.read_bytes(), ctype, {"Content-Security-Policy": "sandbox; default-src 'none'; "
                                                    "style-src 'unsafe-inline'", "X-Content-Type-Options": "nosniff"})

        def _post(self):
            if urlparse(self.path).path != "/api/op":
                return self._json({"error": "not found"}, 404)
            body = self._body()
            if body.get("action") == "add_file" and body.get("data_base64"):
                try:
                    base64.b64decode(body["data_base64"], validate=True)
                except ValueError:
                    raise BadRequest("data_base64 is not base64") from None
            self._json(operate(body, project))

    return Handler


def serve(start: str | None = None, project: str = ".", port: int = 8766, open_browser: bool = True) -> None:
    proj = Path(project).resolve()
    if not proj.is_dir():
        raise ValueError(f"no project folder at {proj}")
    if start and start != PROJECT:
        folder_of(start, proj)
    os.environ["KINE_BACKEND"] = "cpu"  # the GPU context belongs to the thread that made it; requests run on many
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(proj, start))
    url = f"http://127.0.0.1:{srv.server_port}/"
    print(f"brand wizard at {url} (library {ops.library()}, project {proj}) (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
