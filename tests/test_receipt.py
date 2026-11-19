"""Render receipts (OUT.receipt.json) and kine receipt, and the scene's "kinescript" version requirement."""
import json
import struct
import wave

import pytest

import kinescript as k
from kinescript import cli
from kinescript import receipt as R
from kinescript.scene import SceneError, load
from kinescript.schema import requirement, satisfies


def tone(path, seconds=0.5):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"".join(struct.pack("<h", (i % 40 - 20) * 500) for i in range(int(8000 * seconds))))


def project(tmp_path, **extra):
    """A scene file that reads a sound and its word times, and a font, from beside it."""
    (tmp_path / "sfx").mkdir()
    tone(tmp_path / "sfx" / "beep.wav")
    (tmp_path / "sfx" / "beep.words.json").write_text(json.dumps({"words": [{"word": "beep", "start": 0, "end": 0.4}]}))
    spec = dict({"canvas": {"w": 160, "h": 90, "fps": 10, "duration": 1, "bg": "#101010"},
                 "audio": [{"id": "b", "src": "sfx/beep.wav", "at": 0.1}],
                 "layers": [{"id": "t", "type": "text", "text": "hi.there", "x": 80, "y": 45, "size": 20}]}, **extra)
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(spec))
    return scene


def test_a_render_writes_a_receipt_of_what_it_was_made_from(tmp_path):
    scene = project(tmp_path)
    res = k.render(str(scene), str(tmp_path / "out" / "v.mp4"), quality="draft")
    rp = tmp_path / "out" / "v.mp4.receipt.json"
    assert res["receipt"] == str(rp.resolve())
    doc = json.loads(rp.read_text())
    assert doc["kinescript"] == k.__version__
    assert doc["out"]["file"] == "v.mp4" and doc["out"]["sha256"] == R.sha256(tmp_path / "out" / "v.mp4")
    assert doc["out"]["frames"] == 10 and doc["out"]["audio"] is True
    assert doc["scene"] == {"file": "../scene.json", "sha256": R.sha256(scene)}
    assert sorted(doc["inputs"]) == ["sfx/beep.wav", "sfx/beep.words.json"]  # not the text "hi.there"
    assert doc["render"]["quality"] == "draft" and doc["render"]["from"] is None
    assert "rendered" not in json.dumps(doc)  # no time stamp: an unchanged render writes the same receipt
    assert R.check(tmp_path / "out" / "v.mp4")["ok"]


def test_receipt_check_names_what_changed(tmp_path):
    scene = project(tmp_path)
    out = tmp_path / "v.mp4"
    k.render(str(scene), str(out), quality="draft")
    tone(tmp_path / "sfx" / "beep.wav", seconds=0.7)
    res = R.check(out)
    assert not res["ok"] and res["out"] == "same" and res["scene"] == "same"
    assert res["inputs"]["changed"] == ["sfx/beep.wav"] and res["inputs"]["same"] == 1

    scene.write_text(scene.read_text().replace("hi.there", "bye"))
    res = R.check(str(out) + ".receipt.json")  # the receipt itself works as the argument too
    assert res["scene"] == "changed" and res["out"] == "same"


def test_a_scene_that_reads_a_new_file_is_not_ok(tmp_path):
    scene = project(tmp_path)
    out = tmp_path / "v.wav"
    k.render(str(scene), str(out))
    tone(tmp_path / "sfx" / "boop.wav")
    spec = json.loads(scene.read_text())
    spec["audio"].append({"id": "c", "src": "sfx/boop.wav", "at": 0.5})
    scene.write_text(json.dumps(spec))
    res = R.check(out)
    assert res["inputs"]["added"] == ["sfx/boop.wav"] and not res["ok"]


def test_a_changed_or_missing_output_is_reported(tmp_path):
    scene = project(tmp_path)
    out = tmp_path / "v.mp4"
    k.render(str(scene), str(out), quality="draft")
    out.write_bytes(out.read_bytes() + b"\0")
    assert R.check(out)["out"] == "changed"
    out.unlink()
    res = R.check(str(out) + ".receipt.json")
    assert res["out"] == "missing" and not res["ok"]


def test_no_receipt_for_frames_or_when_asked_not_to(tmp_path):
    scene = project(tmp_path)
    res = k.render(str(scene), str(tmp_path / "v.mp4"), quality="draft", receipt=False)
    assert "receipt" not in res and not (tmp_path / "v.mp4.receipt.json").exists()
    res = k.render(str(scene), str(tmp_path / "frames") + "/", scale=0.5)
    assert "receipt" not in res and not list(tmp_path.glob("*.receipt.json"))


def test_a_scene_given_as_data_is_hashed_as_data(tmp_path):
    spec = {"canvas": {"w": 64, "h": 64, "fps": 10, "duration": 0.5}, "layers": [{"type": "rect", "w": 10, "h": 10}]}
    k.render(spec, str(tmp_path / "a.mp4"), quality="draft")
    k.render(json.dumps(spec), str(tmp_path / "b.mp4"), quality="draft")
    a, b = (json.loads((tmp_path / f"{x}.mp4.receipt.json").read_text()) for x in "ab")
    assert a["scene"]["file"] is None and a["scene"]["sha256"] == b["scene"]["sha256"]
    res = R.check(tmp_path / "a.mp4")
    assert res["ok"] and res["scene"] == "unknown"


def test_kine_receipt_exit_code(capsys, tmp_path):
    scene = project(tmp_path)
    out = tmp_path / "v.mp4"
    assert cli.main(["render", str(scene), "-o", str(out), "-q", "draft"]) == cli.OK
    capsys.readouterr()
    assert cli.main(["receipt", str(out)]) == cli.OK
    assert json.loads(capsys.readouterr().out)["ok"]
    scene.write_text(scene.read_text().replace("hi.there", "bye"))
    assert cli.main(["receipt", str(out)]) == cli.ISSUES
    assert json.loads(capsys.readouterr().out)["scene"] == "changed"
    assert cli.main(["receipt", str(tmp_path / "nothing.mp4")]) == cli.BAD
    assert "no receipt" in capsys.readouterr().out
    assert cli.main(["render", str(scene), "-o", str(tmp_path / "w.mp4"), "-q", "draft", "--no-receipt"]) == cli.OK
    assert not (tmp_path / "w.mp4.receipt.json").exists()


def test_model_and_brand_files_are_inputs(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "box.obj").write_text("mtllib box.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (tmp_path / "m" / "box.mtl").write_text("newmtl a\nKd 1 0 0\nmap_Kd -s 1 1 1 tex.png\n")
    (tmp_path / "m" / "tex.png").write_bytes(b"")
    spec = {"canvas": {"w": 64, "h": 64, "duration": 1}, "layers": [{"id": "m", "type": "model", "src": "m/box.obj"}]}
    sc = k.Scene(spec, tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in R.inputs(sc)}
    assert names == {"m/box.obj", "m/box.mtl", "m/tex.png"}


@pytest.mark.parametrize("text, ok", [
    ("0.1", True), (">=0.1", True), (">=99", False), ("<99", True), ("<1", False), (">=1, <99", True),
    ("~=1.0", False), ("!=" + k.__version__, False), ("==" + k.__version__, True), ("==" + k.__version__ + ".0", True),
])
def test_the_scene_can_name_the_kinescript_it_needs(text, ok):
    spec = {"canvas": {"w": 64, "h": 64}, "layers": [{"type": "rect", "w": 10, "h": 10}], "kinescript": text}
    if ok:
        load(spec)
        return
    with pytest.raises(SceneError) as e:
        load(spec)
    first = e.value.issues[0]
    assert first["code"] == "kinescript-version" and k.__version__ in first["message"]
    assert first["fix"].startswith('pip install "kinescript')


def test_a_too_old_kinescript_is_the_first_error_before_unknown_keys():
    spec = {"canvas": {"w": 64, "h": 64}, "kinescript": ">=99", "layers": [{"type": "rect", "w": 10, "glimmer": 3}]}
    with pytest.raises(SceneError) as e:
        load(spec)
    codes = [i["code"] for i in e.value.issues if i["level"] == "error"]
    assert codes[0] == "kinescript-version" and "unknown-key" in codes


@pytest.mark.parametrize("bad, says", [(2.6, "a number loses digits"), ("2.x", "not a version clause"),
                                       ("~=3", "two parts"), (">=", "not a version clause")])
def test_a_requirement_that_cannot_be_read_is_an_error(bad, says):
    with pytest.raises(SceneError) as e:
        load({"canvas": {"w": 64, "h": 64}, "layers": [], "kinescript": bad})
    assert e.value.issues[0]["code"] == "bad-kinescript" and says in e.value.issues[0]["message"]


def test_requirement_reading():
    assert requirement(">=2.6,<3") == [(">=", (2, 6)), ("<", (3,))]
    assert requirement("~=2.6") == [(">=", (2, 6)), ("<", (3,))]
    assert requirement("~=2.6.1") == [(">=", (2, 6, 1)), ("<", (2, 7))]
    assert satisfies("2.6.0.dev1", requirement("2.6")) and not satisfies("2.5.9", requirement("2.6"))
    assert satisfies("2.10", requirement(">2.9"))  # numbers, not text: 2.10 is after 2.9
