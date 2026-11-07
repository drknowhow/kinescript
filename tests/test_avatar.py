import json

import pytest

from kinescript import avatar as A

W, H = 200, 300


def _svg(body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}</svg>'


PARTS = {
    "body.svg": '<rect id="shirt" x="40" y="160" width="120" height="140" rx="30" fill="#3060c0"/>',
    "head.svg": '<circle id="skin" cx="100" cy="100" r="60" fill="#e8c4a0"/>',
    "eyes-open.svg": '<circle cx="80" cy="90" r="6"/><circle cx="120" cy="90" r="6"/>',
    "eyes-closed.svg": '<rect x="74" y="89" width="12" height="2"/><rect x="114" y="89" width="12" height="2"/>',
    "mouth-X.svg": '<rect x="85" y="128" width="30" height="3"/>',
    "mouth-A.svg": '<rect x="85" y="129" width="30" height="2"/>',
    "mouth-C.svg": '<ellipse cx="100" cy="130" rx="15" ry="7"/>',
    "mouth-D.svg": '<ellipse cx="100" cy="132" rx="16" ry="12"/>',
}


def make(folder, **over):
    """A small, valid avatar in `folder`; keyword arguments replace top-level keys of its avatar.json."""
    (folder / "parts").mkdir(parents=True, exist_ok=True)
    for name, body in PARTS.items():
        (folder / "parts" / name).write_text(_svg(body), encoding="utf-8")
    spec = {
        "kinescriptAvatar": 1, "name": "Test", "size": [W, H],
        "parts": {
            "body": {"src": "parts/body.svg", "pivot": [100, 280]},
            "head": {"src": "parts/head.svg", "parent": "body", "pivot": [100, 160]},
            "eyes": {"parent": "head", "states": {"open": "parts/eyes-open.svg", "closed": "parts/eyes-closed.svg"},
                     "look": [6, 3]},
            "mouth": {"parent": "head", "shapes": {s: f"parts/mouth-{s}.svg" for s in "XACD"}},
        },
        "colors": {"skin": "#e8c4a0", "shirt": "accent"},
        "expressions": {"smile": {"head": {"rotate": -3}, "mouth": "C"}},
        "gestures": {"nod": {"dur": 0.5, "head": {"rotate": [[0, 0], [0.2, 6], [0.5, 0]]}}},
        "idle": {"blink": [2.5, 5], "breath": 0.012, "saccade": [0.8, 2.5]},
        "voice": {"backend": "kokoro", "speaker": "am_michael"},
    }
    spec.update(over)
    (folder / "avatar.json").write_text(json.dumps(spec), encoding="utf-8")
    return folder


def codes(folder):
    return [i["code"] for i in A.validate(folder) if i["level"] == "error"]


def test_a_valid_avatar_loads_with_its_mouth_eyes_and_fallbacks(tmp_path):
    av = A.load(make(tmp_path))
    assert list(av.parts) == ["body", "head", "eyes", "mouth"]
    assert av.mouth.name == "mouth" and [p.name for p in av.eyes] == ["eyes"]
    assert av.parts["head"].parent == "body" and av.parts["head"].pivot == (100.0, 160.0)
    assert av.fallbacks == {"B": "C", "E": "D", "F": "D", "G": "A", "H": "C"}
    assert av.shape("C") == "C" and av.shape("F") == "D"
    assert A.validate(tmp_path) == []


def test_missing_required_mouth_shapes_are_an_error_that_names_them(tmp_path):
    make(tmp_path)
    spec = json.loads((tmp_path / "avatar.json").read_text())
    spec["parts"]["mouth"]["shapes"] = {"X": "parts/mouth-X.svg", "C": "parts/mouth-C.svg"}
    (tmp_path / "avatar.json").write_text(json.dumps(spec))
    issue = next(i for i in A.validate(tmp_path) if i["code"] == "avatar-missing-shape")
    assert "lacks shapes A, D" in issue["message"]
    with pytest.raises(A.AvatarError, match="lacks shapes A, D"):
        A.load(tmp_path)


@pytest.mark.parametrize("change, code", [
    (lambda s: s["parts"]["head"].update(parent="hat"), "avatar-parent"),
    (lambda s: s["parts"]["body"].update(parent="head"), "avatar-parent"),
    (lambda s: s["parts"]["body"].update(src="parts/nope.svg"), "avatar-file"),
    (lambda s: s["parts"]["body"].update(src="../outside.svg"), "avatar-file"),
    (lambda s: s["parts"]["body"].update(states={"open": "parts/body.svg"}), "avatar-part"),
    (lambda s: s["parts"]["mouth"]["shapes"].update(Q="parts/mouth-X.svg"), "avatar-shape"),
    (lambda s: s["parts"]["head"].update(pivot=[1]), "avatar-pivot"),
    (lambda s: s["parts"]["head"].update(look=[2, 2]), "avatar-look"),
    (lambda s: s.update(size=[0, 10]), "avatar-size"),
    (lambda s: s["colors"].update(shirt="sky"), "avatar-color"),
    (lambda s: s["expressions"].update(frown={"hat": {"y": 2}}), "avatar-pose"),
    (lambda s: s["expressions"].update(frown={"head": {"tilt": 2}}), "avatar-pose"),
    (lambda s: s["gestures"].update(wave={"head": {"rotate": [[0, 1]]}}), "avatar-gesture"),
    (lambda s: s["gestures"]["nod"]["head"].update(rotate=[[0, 0], [0.9, 1]]), "avatar-gesture"),
    (lambda s: s["idle"].update(blink=[5, 2]), "avatar-idle"),
    (lambda s: s.update(voice={"pitch": 2}), "avatar-voice"),
    (lambda s: s.update(mouthMode="blend"), "avatar-mouth-mode"),
    (lambda s: s.pop("kinescriptAvatar"), "avatar-version"),
])
def test_each_mistake_is_reported_with_its_code(tmp_path, change, code):
    make(tmp_path)
    spec = json.loads((tmp_path / "avatar.json").read_text())
    change(spec)
    (tmp_path / "avatar.json").write_text(json.dumps(spec))
    assert code in codes(tmp_path)


def test_an_avatar_without_eyes_or_mouth_is_valid_but_says_so(tmp_path):
    make(tmp_path, parts={"body": {"src": "parts/body.svg"}})
    got = {i["code"]: i["level"] for i in A.validate(tmp_path)}
    assert got["avatar-no-blink"] == "info" and got["avatar-no-mouth"] == "info"
    assert "avatar-pose" in got  # the expressions still move a head it no longer has


def test_kine_avatar_check_prints_what_the_avatar_has(tmp_path, capsys):
    from kinescript.cli import main
    make(tmp_path)
    assert main(["avatar", "check", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["shapes"] == ["A", "C", "D", "X"] and out["gestures"] == ["nod"]
    (tmp_path / "avatar.json").write_text("{}")
    assert main(["avatar", "check", str(tmp_path)]) == 1
    assert "avatar-version" in [i["code"] for i in json.loads(capsys.readouterr().out)["issues"]]
