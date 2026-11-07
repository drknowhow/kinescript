"""3D avatars: an avatar.json whose model is a glTF file, its mouth, blink and expressions morph target weights."""
import json

import numpy as np
import pytest

import kinescript as k
from kinescript import avatar as A
from kinescript import mesh
from kinescript.render import Renderer
from test_animation import Gltf
from test_avatar import _wav

NAMES = ["jawOpen", "mouthClose", "smile", "blink"]


def _face(folder, **over):
    """A plane with four morph targets, each moving its own corner of it, and an avatar.json for it."""
    folder.mkdir(parents=True, exist_ok=True)
    plane = mesh.placed(mesh.unit("plane"), (20, 20, 1))
    g, targets = Gltf(), []
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        d = np.zeros_like(plane.P)
        d[(np.sign(plane.P[:, 0]) == sx) & (np.sign(plane.P[:, 1]) == sy), 2] = 5
        targets.append({"POSITION": g.add(d * (1, -1, 1))})
    m = g.mesh(plane, targets=targets)
    g.doc["meshes"][m]["extras"] = {"targetNames": NAMES}
    g.node(mesh=m)
    g.save(folder / "face.gltf")
    spec = {"kinescriptAvatar": 1, "name": "Ada", "model": {"src": "face.gltf"},
            "mouth": {"A": {"mouthClose": 1}, "C": {"jawOpen": 0.5}, "D": {"jawOpen": 1}},
            "blink": {"blink": 1}, "expressions": {"smile": {"morphs": {"smile": 0.8}}}}
    spec.update(over)
    (folder / "avatar.json").write_text(json.dumps(spec))
    return folder


def _scene(tmp_path, **layer):
    _face(tmp_path / "ada")
    _wav(tmp_path / "vo.wav")
    words = [{"word": "Make", "start": 0.5, "end": 0.85}, {"word": "move", "start": 1.5, "end": 1.9}]
    L = {"id": "ada", "use": "avatar", "avatar": "ada", "h": 200, "say": "vo", "idle": False}
    L.update(layer)
    return {"canvas": {"w": 320, "h": 240, "fps": 30, "duration": 4},
            "audio": [{"id": "vo", "src": "vo.wav", "at": 1.0, "words": words}], "layers": [L]}


def _w(sc, name, T):
    L = sc.by_id["ada-model"]
    return round(float(L.get("w_" + name, T - L.t_in, T)), 3)


def test_a_3d_avatar_checks_and_names_morphs_its_model_lacks(tmp_path):
    assert A.summary(_face(tmp_path / "a"))["morphs"] == ["mouthClose", "jawOpen", "blink", "smile"]
    bad = _face(tmp_path / "b", mouth={"A": {"mouthClsoe": 1}, "D": {"jawOpen": 2}}, gestures={"nod": {"dur": 1}})
    got = {(i["code"], i["message"]) for i in A.validate(bad)}
    assert ("avatar-morph", "mouth.A: the model has no morph target 'mouthClsoe'") in got
    assert ("avatar-morph", "mouth.D.jawOpen is 2; a morph weight is 0 to 1") in got
    assert any(c == "avatar-missing-shape" and "C" in m for c, m in got)
    assert any(c == "avatar-gesture" for c, _ in got)


def test_the_mouth_follows_the_words_as_morph_weights(tmp_path):
    sc = k.Scene(_scene(tmp_path), tmp_path)
    assert _w(sc, "jawOpen", 0.8) == 0 and _w(sc, "mouthClose", 0.8) == 0  # before the sound: at rest
    assert max(_w(sc, "mouthClose", f / 30) for f in range(45, 50)) > 0.9  # the M of "Make", 0.5 s in
    assert _w(sc, "jawOpen", 1.7) == 0.5  # C
    assert _w(sc, "jawOpen", 2.2) == 0 and _w(sc, "jawOpen", 3.5) == 0  # the pause, and after


def test_expressions_and_blinks_set_their_morphs(tmp_path):
    sc = k.Scene(_scene(tmp_path, expressions=[{"name": "smile", "at": 0.5, "dur": 1}]), tmp_path)
    assert _w(sc, "smile", 0.4) == 0 and _w(sc, "smile", 1.0) == 0.8 and _w(sc, "smile", 1.7) == 0
    on = k.Scene(_scene(tmp_path, idle={"breath": False}), tmp_path)
    shut = [f for f in range(120) if _w(on, "blink", f / 30) == 1]
    assert shut and _w(on, "blink", 0) == 0
    again = k.Scene(_scene(tmp_path, idle={"breath": False}), tmp_path)
    assert shut == [f for f in range(120) if _w(again, "blink", f / 30) == 1]  # seeded: the same every time
    other = k.Scene(_scene(tmp_path, idle={"breath": False}, seed=3), tmp_path)
    assert shut != [f for f in range(120) if _w(other, "blink", f / 30) == 1]


def test_a_3d_avatar_draws_and_its_mouth_moves_the_mesh(tmp_path):
    spec = _scene(tmp_path)

    r = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    rest, talking = r.frame(15).copy(), r.frame(51).copy()  # 0.5 s, and 1.7 s: the jaw half open
    assert rest.std() > 0 and np.abs(rest.astype(int) - talking.astype(int)).max() > 20


def test_the_sheet_shows_a_3d_avatars_poses(tmp_path):
    av = A.load(_face(tmp_path / "ada"))
    names = [n for n, _ in A.poses(av)]
    assert names[0] == "neutral" and "mouth D" in names and "eyes closed" in names and "expression smile" in names
    assert "mouth B (as C)" in names
    got = A.sheet(tmp_path / "ada", str(tmp_path / "sheet.png"), width=120)
    assert got["poses"] == names


def test_a_3d_avatar_refuses_gaze_and_gestures_for_now(tmp_path):
    with pytest.raises(ValueError, match="does not take gaze or gestures yet"):
        k.Scene(_scene(tmp_path, gaze=[{"target": "left"}]), tmp_path)
