"""Brand library operations: kine brand (CLI), kine_brand (MCP) and brandops agree, and refuse what they should."""
import json
import struct
import zlib
from pathlib import Path

import pytest

from kinescript import brand as B
from kinescript import brandops as ops
from kinescript.cli import main
from kinescript.mcp import call

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="#f00"/></svg>'


def png(w=4, h=4) -> bytes:
    raw = b"".join(b"\x00" + b"\xff\x00\x00\xff" * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "library"
    monkeypatch.setenv("KINE_BRANDS", str(root))
    return root


def cli(capsys, *argv) -> tuple[int, dict]:
    code = main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def mcp(**args) -> dict:
    return json.loads(call("kine_brand", args)[-1]["text"])


def snapshot(folder: Path) -> dict:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_default_brand_is_clean(lib):
    out = ops.init("kinescript")
    assert out["brand"]["name"] == "Kinescript"
    assert out["issues"] == []
    assert B.load(lib / "kinescript").spec["palettes"]["dark"]["accent"] == "#ff633f"


def test_init_list_show(lib):
    ops.init("acme")
    ops.init("acme-two", source="acme")
    ids = [b["id"] for b in ops.list_library()]
    assert ids == ["acme", "acme-two"]
    assert ops.show("acme-two")["brand"]["name"] == "Acme Two"
    with pytest.raises(ops.BrandOpError, match="already"):
        ops.init("acme")
    with pytest.raises(ops.BrandOpError, match="lower-case"):
        ops.init("Acme Co")


def test_cli_and_mcp_write_the_same_files(lib, tmp_path, capsys):
    logo = tmp_path / "mark.svg"
    logo.write_bytes(SVG)
    steps = [("init", {"id": "{id}"}),
             ("set", {"brand": "{id}", "path": "palettes.dark.accent", "value": "#3366ff"}),
             ("set", {"brand": "{id}", "path": "palettes.light", "value": None}),
             ("add_file", {"brand": "{id}", "kind": "logos", "file": str(logo)}),
             ("set", {"brand": "{id}", "path": "logos.mark", "value": {"file": "logos/mark.svg"}})]

    def fill(d, bid):
        return {k: (v.replace("{id}", bid) if isinstance(v, str) else v) for k, v in d.items()}

    for action, args in steps:
        a = fill(args, "via-mcp")
        mcp(action=action, **a)
        b = fill(args, "via-cli")
        argv = ["brand", action.replace("_", "-"), b.get("brand") or b["id"]]
        if action == "set":
            argv += [b["path"], json.dumps(b["value"])]
        if action == "add_file":
            argv += [b["file"], "--kind", b["kind"]]
        code, out = cli(capsys, *argv)
        assert code == 0 and out["ok"], out
    one, two = snapshot(lib / "via-mcp"), snapshot(lib / "via-cli")
    assert one.keys() == two.keys()
    for k in one:
        assert one[k].replace(b"via-mcp", b"").replace(b"Via Mcp", b"") == \
            two[k].replace(b"via-cli", b"").replace(b"Via Cli", b""), k
    assert "light" not in B.read(lib / "via-cli")["palettes"]


def test_cli_set_takes_plain_text(lib, capsys):
    ops.init("acme")
    code, out = cli(capsys, "brand", "set", "acme", "palettes.dark.accent", "#123456")
    assert code == 0 and out["brand"]["palettes"]["dark"]["accent"] == "#123456"


def test_stale_rev_refused(lib):
    rev = ops.init("acme")["rev"]
    ops.set_value("acme", "name", "Acme Inc", base_rev=rev)
    with pytest.raises(ops.BrandOpError, match="changed since"):
        ops.set_value("acme", "name", "Acme Ltd", base_rev=rev)
    with pytest.raises(ops.BrandOpError, match="changed since"):
        ops.add_file("acme", "logos", "a.svg", SVG, base_rev=rev)
    assert B.read(lib / "acme")["name"] == "Acme Inc"


def test_breaking_change_refused(lib):
    ops.init("acme")
    before = (lib / "acme" / B.FILE).read_bytes()
    with pytest.raises(ops.BrandOpError):
        ops.set_value("acme", "palettes.dark.accent", None)  # a required role
    with pytest.raises(ops.BrandOpError):
        ops.set_value("acme", "kinescriptBrand", 99)
    assert (lib / "acme" / B.FILE).read_bytes() == before


def test_use_copies_and_guards_local_edits(lib, tmp_path):
    ops.init("acme")
    project = tmp_path / "proj"
    project.mkdir()
    out = ops.use("acme", project)
    assert out["source"]["id"] == "acme" and out["edited"] is False
    ops.set_value("acme", "palettes.dark.accent", "#3366ff")
    ops.use("acme", project)  # no local edits: updates the copy
    assert B.read(project / "brand")["palettes"]["dark"]["accent"] == "#3366ff"
    ops.set_value(str(project / "brand"), "palettes.dark.accent", "#00aa00")
    assert ops.show(str(project / "brand"))["edited"] is True
    with pytest.raises(ops.BrandOpError, match="changes made in this project"):
        ops.use("acme", project)
    assert B.read(project / "brand")["palettes"]["dark"]["accent"] == "#00aa00"
    ops.use("acme", project, force=True)
    assert B.read(project / "brand")["palettes"]["dark"]["accent"] == "#3366ff"


def test_used_brand_renders_in_a_scene(lib, tmp_path):
    from kinescript.scene import Scene
    ops.init("acme")
    ops.use("acme", tmp_path)
    sc = Scene({"canvas": {"w": 64, "h": 36, "fps": 10, "duration": 1}, "brand": "brand",
                "layers": [{"id": "d", "type": "circle", "x": 32, "y": 18, "r": 4, "fill": "$accent"}]}, str(tmp_path))
    assert sc.theme["accent"] == "#ff633f"


@pytest.mark.parametrize("kind,name,data,match", [
    ("fonts", "x.ttf", b"<html>not a font</html>", "first bytes"),
    ("fonts", "x.exe", b"\x00\x01\x00\x00", "take"),
    ("logos", "x.png", b"GIF89a....", "first bytes"),
    ("logos", "x.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "script"),
    ("logos", "x.svg", b'<svg xmlns="http://www.w3.org/2000/svg" onload="x()"/>', "onload"),
    ("logos", "x.svg", b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:l="http://www.w3.org/1999/xlink">'
                       b'<image l:href="http://evil/x.png"/></svg>', "links"),
    ("logos", "x.svg", b'<!DOCTYPE svg [<!ENTITY a "b">]><svg xmlns="http://www.w3.org/2000/svg"/>', "DOCTYPE"),
    ("logos", "../x.svg", SVG, "plain file name"),
    ("logos", "x.svg", b"", "empty"),
])
def test_upload_sniffing(lib, kind, name, data, match):
    ops.init("acme")
    with pytest.raises(ops.BrandOpError, match=match):
        ops.add_file("acme", kind, name, data)
    assert not (lib / "acme" / kind).exists() or not any((lib / "acme" / kind).iterdir())


def test_add_and_remove_files(lib):
    ops.init("acme")
    out = ops.add_file("acme", "logos", "full.png", png())
    assert out["file"] == "logos/full.png" and "logos/full.png" in out["files"]
    ops.add_file("acme", "fonts", "Acme.ttf", b"\x00\x01\x00\x00" + b"\x00" * 60)
    with pytest.raises(ops.BrandOpError, match="exists"):
        ops.add_file("acme", "logos", "full.png", png())
    ops.add_file("acme", "logos", "full.png", png(8, 8), replace=True)
    ops.set_value("acme", "logos.full", {"file": "logos/full.png"})
    with pytest.raises(ops.BrandOpError, match="still named"):
        ops.remove_file("acme", "logos/full.png")
    ops.set_value("acme", "logos.full", None)
    assert "logos/full.png" not in ops.remove_file("acme", "logos/full.png")["files"]
    with pytest.raises(ops.BrandOpError, match="only files under"):
        ops.remove_file("acme", "brand.json")
    with pytest.raises(ops.BrandOpError):
        ops.remove_file("acme", "fonts/../brand.json")
    assert (lib / "acme" / B.FILE).is_file()


def test_mcp_add_file_from_base64(lib):
    import base64
    ops.init("acme")
    out = mcp(action="add_file", brand="acme", kind="logos", name="m.svg", data_base64=base64.b64encode(SVG).decode())
    assert out["file"] == "logos/m.svg"


def test_errors_reach_the_caller(lib, capsys):
    code, out = cli(capsys, "brand", "show", "nope")
    assert code == 2 and "no brand" in out["error"]
    with pytest.raises(ValueError, match="'path' is needed"):
        call("kine_brand", {"action": "set", "brand": "x", "value": 1})


def test_check_action(lib, capsys):
    ops.init("acme")
    code, out = cli(capsys, "brand", "check", "acme")
    assert code == 0 and out["ok"] is True


def test_preview(lib, tmp_path):
    ops.init("acme")
    out = tmp_path / "p.png"
    res = call("kine_brand", {"action": "preview", "brand": "acme", "out": str(out)})
    assert res[0]["type"] == "image" and out.is_file() and out.read_bytes()[:4] == b"\x89PNG"
