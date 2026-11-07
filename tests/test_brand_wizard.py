"""kine brand wizard: the page's routes, locked to this machine, writing through the same brandops as kine_brand."""
import base64
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from kinescript import brand as B
from kinescript import brand_server, brandops as ops
from kinescript.mcp import call

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10"><rect width="20" height="10" fill="#f00"/></svg>'


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    monkeypatch.setenv("KINE_BRANDS", str(tmp_path / "library"))
    project = tmp_path / "proj"
    project.mkdir()
    ops.init("acme")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), brand_server.make_handler(project, "acme"))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield httpd.server_port, project, tmp_path / "library"
    httpd.shutdown()
    httpd.server_close()


def req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    out = (r.status, r.read(), dict(r.getheaders()))
    c.close()
    return out


def jget(port, path):
    code, body, _ = req(port, "GET", path)
    return code, json.loads(body)


def post(port, body):
    code, data, _ = req(port, "POST", "/api/op", body)
    return code, json.loads(data)


def test_page_state_and_brand(srv):
    port, _, _ = srv
    code, html, _ = req(port, "GET", "/")
    assert code == 200 and b"Kinescript brands" in html
    code, st = jget(port, "/api/state")
    assert code == 200 and st["start"] == "acme" and [b["id"] for b in st["brands"]] == ["acme"]
    assert st["projectBrand"] is False and "palette" in st["gallery"]
    code, info = jget(port, "/api/brand?ref=acme")
    assert code == 200 and info["brand"]["id"] == "acme" and info["issues"] == []
    assert {"dark", "light"} <= set(info["contrast"]) and info["variants"] == ["dark", "light"]
    pair = next(p for p in info["contrast"]["dark"]["pairs"] if p["fg"] == "text" and p["bg"] == "bg")
    assert pair["needs"] == 4.5 and pair["ok"] is True
    assert "title" in info["textStyles"]


def test_gallery_png(srv):
    port, _, _ = srv
    code, png, h = req(port, "GET", "/api/gallery?ref=acme&what=palette&variant=light")
    assert code == 200 and png[:4] == b"\x89PNG" and h["Content-Type"] == "image/png"
    code, out = jget(port, "/api/gallery?ref=acme&what=nope")
    assert code == 400 and "one of" in out["error"]


def test_page_edit_writes_what_kine_brand_writes(srv):
    port, _, lib = srv
    ops.init("twin")
    rev = jget(port, "/api/brand?ref=acme")[1]["rev"]
    code, out = post(port, {"action": "set", "ref": "acme", "base_rev": rev, "path": "palettes.dark.accent",
                            "value": "#3366ff"})
    assert code == 200 and out["brand"]["palettes"]["dark"]["accent"] == "#3366ff"
    call("kine_brand", {"action": "set", "brand": "twin", "path": "palettes.dark.accent", "value": "#3366ff"})
    data = base64.b64encode(SVG).decode()
    post(port, {"action": "add_file", "ref": "acme", "kind": "logos", "name": "m.svg", "data_base64": data})
    call("kine_brand", {"action": "add_file", "brand": "twin", "kind": "logos", "name": "m.svg", "data_base64": data})
    a = (lib / "acme" / B.FILE).read_text(encoding="utf-8").replace('"acme"', "").replace('"Acme"', "")
    b = (lib / "twin" / B.FILE).read_text(encoding="utf-8").replace('"twin"', "").replace('"Twin"', "")
    assert a == b
    assert (lib / "acme" / "logos" / "m.svg").read_bytes() == (lib / "twin" / "logos" / "m.svg").read_bytes()


def test_stale_rev_and_breaking_change_are_400(srv):
    port, _, lib = srv
    rev = jget(port, "/api/brand?ref=acme")[1]["rev"]
    post(port, {"action": "set", "ref": "acme", "base_rev": rev, "path": "name", "value": "Acme Inc"})
    code, out = post(port, {"action": "set", "ref": "acme", "base_rev": rev, "path": "name", "value": "Acme Ltd"})
    assert code == 400 and "changed since" in out["error"]
    code, out = post(port, {"action": "replace", "ref": "acme", "spec": {"kinescriptBrand": 1, "palettes": {}}})
    assert code == 400
    assert B.read(lib / "acme")["name"] == "Acme Inc"


def test_use_init_and_project_ref(srv):
    port, project, _ = srv
    code, out = post(port, {"action": "use", "ref": "acme"})
    assert code == 200 and out["ref"] == "@project" and (project / "brand" / B.FILE).is_file()
    assert jget(port, "/api/state")[1]["projectBrand"] is True
    code, out = post(port, {"action": "set", "ref": "@project", "path": "palettes.dark.accent", "value": "#00aa00"})
    assert code == 200 and out["edited"] is True
    info = jget(port, "/api/brand?ref=@project")[1]
    assert info["compare"]["parts"]["palettes"] == "changed" and info["compare"]["parts"]["fonts"] == "same"
    code, out = post(port, {"action": "use", "ref": "acme"})
    assert code == 400 and "changes made in this project" in out["error"]
    code, out = post(port, {"action": "use", "ref": "acme", "force": True})
    assert code == 200 and out["edited"] is False
    code, out = post(port, {"action": "init", "id": "acme-copy", "from": "@project"})
    assert code == 200 and out["ref"] == "acme-copy" and out["brand"]["name"] == "Acme Copy"


def test_page_cannot_reach_other_folders(srv, tmp_path):
    port, _, _ = srv
    other = tmp_path / "elsewhere"
    ops.init("x", folder=other)
    for ref in (str(other), "../elsewhere", "C:/Windows"):
        code, out = post(port, {"action": "set", "ref": ref, "path": "name", "value": "X"})
        assert code == 400 and "library brand" in out["error"]
    code, out = post(port, {"action": "replace", "ref": "acme", "file": str(other / B.FILE)})
    assert code == 400  # no local paths from the page
    code, out = post(port, {"action": "add_file", "ref": "acme", "kind": "logos", "file": str(other / B.FILE)})
    assert code == 400
    code, out = post(port, {"action": "list"})
    assert code == 400 and "one of" in out["error"]
    code, out = jget(port, "/api/file?ref=acme&path=brand.json")
    assert code == 400
    code, out = jget(port, "/api/file?ref=acme&path=logos/../brand.json")
    assert code in (400, 404)


def test_logo_file_is_served_sandboxed(srv):
    port, _, _ = srv
    ops.add_file("acme", "logos", "m.svg", SVG)
    code, body, h = req(port, "GET", "/api/file?ref=acme&path=logos/m.svg")
    assert code == 200 and body == SVG and h["Content-Type"] == "image/svg+xml"
    assert "sandbox" in h["Content-Security-Policy"]


def test_foreign_host_and_bad_bodies(srv):
    port, _, _ = srv
    code, _, _ = req(port, "GET", "/api/state", headers={"Host": "evil.example"})
    assert code == 403
    code, _, _ = req(port, "POST", "/api/op", body={"action": "set"}, headers={"Content-Type": "text/plain"})
    assert code == 400
    code, out = post(port, {"action": "add_file", "ref": "acme", "kind": "logos", "name": "a.svg",
                            "data_base64": "%%%"})
    assert code == 400 and "base64" in out["error"]


def test_the_avatars_step_adds_the_starter_or_a_copy_and_draws_it(srv):
    port, _, lib = srv
    code, html, _ = req(port, "GET", "/")
    assert b'data-tab="Avatars"' in html
    code, out = post(port, {"action": "add_avatar", "ref": "acme", "name": "host"})
    assert code == 200 and out["brand"]["avatars"] == {"host": {"src": "kine:host"}}
    code, out = post(port, {"action": "add_avatar", "ref": "acme", "name": "guide", "copy": True})
    assert code == 200 and out["brand"]["avatars"]["guide"] == {"src": "avatars/guide"} and out["issues"] == []
    assert (lib / "acme" / "avatars" / "guide" / "parts" / "mouth-X.svg").is_file()
    code, out = post(port, {"action": "add_avatar", "ref": "acme", "name": "host"})
    assert code == 400 and "already has an avatar 'host'" in out["error"]
    code, out = post(port, {"action": "set", "ref": "acme", "path": "avatars.host.voice.speaker", "value": "am_adam"})
    assert code == 200 and out["brand"]["avatars"]["host"]["voice"] == {"speaker": "am_adam"}
    code, png, h = req(port, "GET", "/api/gallery?ref=acme&what=avatar:guide&variant=light")
    assert code == 200 and png[:4] == b"\x89PNG"
    code, out = jget(port, "/api/gallery?ref=acme&what=avatar:nobody")
    assert code == 400 and "no avatar 'nobody'" in out["error"]


def test_kine_brand_add_avatar(tmp_path, monkeypatch, capsys):
    from kinescript.cli import main
    monkeypatch.setenv("KINE_BRANDS", str(tmp_path / "library"))
    ops.init("acme")
    assert main(["brand", "add-avatar", "acme", "host", "--copy"]) == 0
    assert json.loads(capsys.readouterr().out)["brand"]["avatars"] == {"host": {"src": "avatars/host"}}
    assert main(["brand", "add-avatar", "acme", "Bad Name"]) != 0
