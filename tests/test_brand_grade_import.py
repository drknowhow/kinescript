"""A brand's grade defaults, and palettes proposed from a logo (kine brand import)."""
import json
import shutil
from pathlib import Path

import pytest

from kinescript import brand as B
from kinescript import brandops as ops
from kinescript import palette
from kinescript.cli import main
from kinescript.mcp import call
from kinescript.scene import Scene, SceneError

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "brand"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    shutil.copytree(EXAMPLE, tmp_path / "brand")
    return tmp_path


def put(folder: Path, **parts):
    spec = B.read(folder)
    spec.update(parts)
    (folder / "brand.json").write_text(json.dumps(spec), encoding="utf-8")


def scene(**top) -> dict:
    return {"canvas": {"w": 320, "h": 180, "fps": 10, "duration": 1}, "brand": "brand",
            "layers": [{"id": "t", "type": "rect", "w": 40, "h": 40, "x": 160, "y": 90, "fill": "$accent"}], **top}


def test_grade_comes_from_the_brand_under_the_scene(proj):
    put(proj / "brand", grade={"temperature": 0.1, "contrast": 1.05})
    assert Scene(scene(), str(proj)).grade == {"temperature": 0.1, "contrast": 1.05}
    s = Scene(scene(grade={"contrast": 1.2, "vignette": 0.3}), str(proj))
    assert s.grade == {"temperature": 0.1, "contrast": 1.2, "vignette": 0.3}


def test_a_locked_grade_refuses_a_change(proj):
    spec = B.read(proj / "brand")
    put(proj / "brand", grade={"temperature": 0.1}, lock=[*spec.get("lock", []), "grade"])
    Scene(scene(grade={"temperature": 0.1}), str(proj))  # the same value is no change
    with pytest.raises(SceneError) as e:
        Scene(scene(grade={"temperature": 0.3}), str(proj))
    assert any(i["code"] == "brand-locked" and "grade.temperature" in i["message"] for i in e.value.issues)


def test_grade_is_validated(proj):
    spec = B.read(proj / "brand")
    spec["grade"] = {"temprature": 0.1}
    assert any("grade.temprature" in i["message"] for i in B.validate(proj / "brand", spec))
    spec["grade"] = {"contrast": "lots"}
    assert any("grade" in i["message"] for i in B.validate(proj / "brand", spec) if i["level"] == "error")


def test_logo_colours_and_a_readable_proposal(proj):
    cols = palette.logo_colours(proj / "brand" / "logos" / "full.svg")
    assert cols and all(c["hex"].startswith("#") and 0 < c["share"] <= 1 for c in cols)
    found = palette.propose(cols, "Tidewater")
    assert set(found["palettes"]) == {"dark", "light"} and found["roles"]["accent"]
    for name, pal in found["palettes"].items():
        assert palette._ratio(pal["text"], pal["bg"]) >= 4.5, name
        assert palette._ratio(pal["accent"], pal["bg"]) >= 3.0, name
    spec = {**B.read(proj / "brand"), "palettes": found["palettes"], "charts": found["charts"]}
    assert [i for i in B.validate(proj / "brand", spec) if i["level"] == "error"] == []


def test_a_grey_logo_keeps_the_default_accent(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    svg = tmp_path / "grey.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" '
                   'fill="#777777"/></svg>', encoding="utf-8")
    found = palette.propose(palette.logo_colours(svg))
    assert found["roles"]["accent"] == "#ff633f" and any("no saturated" in n for n in found["notes"])


def test_import_proposes_then_applies(proj):
    before = B.read(proj / "brand")
    out = ops.import_palette(str(proj / "brand"), logo="logos/full.svg")
    assert not out["applied"] and B.read(proj / "brand") == before
    out = ops.import_palette(str(proj / "brand"), logo="logos/full.svg", apply=True)
    after = B.read(proj / "brand")
    assert out["applied"] and after["palettes"] == out["proposal"]["palettes"]
    assert after["charts"]["series"] == out["proposal"]["charts"]["series"]
    with pytest.raises(ops.BrandOpError, match="leads out"):
        ops.import_palette(str(proj / "brand"), logo="../outside.svg")
    with pytest.raises(ops.BrandOpError, match="name the logo"):
        ops.import_palette(str(proj / "brand"))


def test_cli_and_mcp_import(proj, capsys):
    code = main(["brand", "import", str(proj / "brand"), "logos/full.svg"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["proposal"]["palettes"]["dark"]["accent"]
    res = json.loads(call("kine_brand", {"action": "import", "brand": str(proj / "brand"),
                                         "logo": "logos/full.svg"})[-1]["text"])
    assert res["proposal"] == out["proposal"]
    code = main(["brand", "import", str(proj / "brand"), str(proj / "brand" / "logos" / "mark.svg"), "--apply"])
    assert code == 0 and json.loads(capsys.readouterr().out)["applied"]


def test_wizard_import_names_a_brand_logo_only(proj, monkeypatch):
    from kinescript import brand_server
    monkeypatch.setenv("KINE_BRANDS", str(proj / "library"))
    before = B.read(proj / "brand")
    out = brand_server.operate({"action": "import", "ref": "@project", "logo": "logos/mark.svg"}, proj)
    assert not out["applied"] and B.read(proj / "brand") == before
    with pytest.raises(ops.BrandOpError, match="name the logo"):  # a local path from the page is dropped
        brand_server.operate({"action": "import", "ref": "@project", "file": str(proj / "brand" / "logos" / "mark.svg")},
                             proj)
    out = brand_server.operate({"action": "import", "ref": "@project", "logo": "logos/mark.svg", "apply": True}, proj)
    assert out["applied"] and B.read(proj / "brand")["palettes"] == out["proposal"]["palettes"]
