import json
from pathlib import Path

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


def test_a_pose_becomes_groups_that_turn_about_each_pivot(tmp_path):
    av = A.load(make(tmp_path))
    g = A.layers(av, "a", {"parts": {"head": {"rotate": 6}}, "mouth": "F", "eyes": "closed", "look": [1, 0]})
    body = g["children"][0]
    assert (body["id"], body["x"], body["y"]) == ("a-body", 100, 280)
    art, head = body["children"]
    assert art["type"] == "svg" and (art["x"], art["y"]) == (0, -130)  # the drawing's middle, from the pivot
    assert art["parts"] == {"shirt": {"fill": "#ff633f"}}  # the accent role, from the built-in theme
    assert (head["x"], head["y"], head["rotate"]) == (0, -120, 6)
    eyes, mouth = head["children"][1:]
    assert [L.get("opacity", 1) for L in eyes["children"]] == [0, 1]  # closed
    assert eyes["children"][1]["x"] == 100 + 6  # no pivot, so at the middle; looking right by its whole reach
    assert [L["id"] for L in mouth["children"] if L.get("opacity", 1)] == ["a-mouth-D"]  # F falls back to D


def test_parts_listed_before_their_parent_draw_behind_it(tmp_path):
    make(tmp_path)
    spec = json.loads((tmp_path / "avatar.json").read_text())
    spec["parts"] = {"arm": {"src": "parts/head.svg", "parent": "body", "pivot": [60, 180]}, **spec["parts"]}
    (tmp_path / "avatar.json").write_text(json.dumps(spec))
    body = A.layers(A.load(tmp_path), "a")["children"][0]
    assert [c["id"] for c in body["children"]] == ["a-arm", "a-body-art", "a-head"]


def test_a_brand_palette_fills_the_role_colours(tmp_path):
    av = A.load(make(tmp_path))
    assert A.colors(av, {"accent": "#123456"}) == {"skin": "#e8c4a0", "shirt": "#123456"}


def test_gesture_peak_and_sampling():
    nod = {"dur": 0.5, "head": {"rotate": [[0, 0], [0.2, 6], [0.5, 0]]}}
    assert A.peak(nod) == 0.2
    assert A.sample(nod["head"]["rotate"], 0.35) == pytest.approx(3) and A.sample([[0, 1], [1, 2]], 5) == 2


def test_kine_look_on_an_avatar_folder_draws_every_pose(tmp_path, capsys):
    from kinescript.cli import main
    make(tmp_path)
    assert main(["look", str(tmp_path), "--width", "200"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["poses"][:3] == ["neutral", "mouth X", "mouth A"] and "mouth F (as D)" in out["poses"]
    assert out["poses"][-2:] == ["expression smile", "gesture nod at 0.2s"]
    assert (tmp_path / "avatar_sheet.png").stat().st_size > 1000 and out["w"] == 600


def test_the_sheet_scene_draws_every_part_without_problems(tmp_path):
    from kinescript.verify import check
    scene, _ = A.sheet_scene(A.load(make(tmp_path)), 200, 3)
    got = check(scene)
    reading = {"tiny-text", "read-time", "safe-area"}  # about video labels, not a still review sheet
    assert got["ok"] and [i for i in got["issues"] if i["level"] != "info" and i["code"] not in reading] == []


def _wav(path, seconds=3.0):
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\0\0" * int(8000 * seconds))


def _talking(tmp_path, say, audio=(), at=0.5):
    make(tmp_path / "host", voice={"speaker": "am_michael"})
    return {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 4}, "audio": list(audio),
            "layers": [{"id": "host", "use": "avatar", "avatar": "host", "h": 150, "at": at, "say": say}]}


def test_say_lines_become_voice_sounds_with_the_avatars_voice(tmp_path):
    spec = _talking(tmp_path, [{"text": "One"}, {"text": "Two", "speaker": "af_heart"}, "intro"],
                    audio=[{"id": "intro", "src": "intro.wav"}])
    got = A.say_sounds(spec, tmp_path)
    assert spec["layers"][0]["say"][0] == {"text": "One"}  # the scene itself is left as written
    one, two = got["audio"][1:]
    assert one == {"id": "host-say1", "role": "voice", "speaker": "am_michael", "text": "One", "at": "host.in"}
    assert two["speaker"] == "af_heart" and two["at"] == "host-say1.out+0.4"
    assert got["layers"][0]["say"] == ["host-say1", "host-say2", "intro"]


@pytest.mark.parametrize("say, message", [
    ({"txt": "x"}, "each say is a sound id"),
    ({"text": "x", "pitch": 2}, "not pitch"),
    ({"text": "x", "id": "intro"}, "already a sound id"),
])
def test_say_mistakes_are_named(tmp_path, say, message):
    spec = _talking(tmp_path, say, audio=[{"id": "intro", "src": "intro.wav"}])
    with pytest.raises(ValueError, match=message):
        A.say_sounds(spec, tmp_path)


def _mouth_at(spec, base, t):
    from kinescript.verify import probe
    f = base / "s.json"
    f.write_text(json.dumps(spec))
    return sorted({i.split("-")[2] for i, v in probe(str(f), t)["visible"].items()
                   if i.startswith("host-mouth-") and v["alpha"] > 0})


def test_the_mouth_follows_the_words_of_a_recorded_sound(tmp_path):
    _wav(tmp_path / "vo.wav")
    words = [{"word": "Make", "start": 0.5, "end": 0.85}, {"word": "move", "start": 1.5, "end": 1.9}]
    spec = _talking(tmp_path, "vo", audio=[{"id": "vo", "src": "vo.wav", "at": 1.0, "words": words}])
    assert _mouth_at(spec, tmp_path, 0.8) == ["X"]  # before the sound: at rest
    assert _mouth_at(spec, tmp_path, 1.52) == ["A"]  # the M of "Make", 0.5 s into the sound
    assert _mouth_at(spec, tmp_path, 1.7) == ["C"]
    assert _mouth_at(spec, tmp_path, 2.2) == ["X"]  # the pause between the words closes it
    assert _mouth_at(spec, tmp_path, 3.5) == ["X"]


def test_a_line_spoken_by_kine_voice_moves_the_mouth(tmp_path, monkeypatch):
    from kinescript import voice

    def speak(req):
        _wav(Path(req["out"]))
        said = req["text"].split()
        return {"words": [{"word": w, "start": 0.5 + 0.4 * i, "end": 0.85 + 0.4 * i} for i, w in enumerate(said)],
                "speaker": req["speaker"], "version": "1"}
    monkeypatch.delenv("KINE_VOICE_CMD", raising=False)
    monkeypatch.setattr(voice, "backends", lambda: {"fake": type("E", (), {"load": lambda self: speak})()})
    spec = _talking(tmp_path, {"text": "Make it move"})
    (tmp_path / "s.json").write_text(json.dumps(spec))
    made = voice.generate(str(tmp_path / "s.json"))["voices"]
    assert made[0]["id"] == "host-say" and made[0]["status"] == "made"
    words = json.loads((tmp_path / "voice" / "host-say.words.json").read_text())
    assert words["made"]["speaker"] == "am_michael"  # the avatar's voice
    assert _mouth_at(spec, tmp_path, 0.5 + 0.52) == ["A"]  # the line starts with the avatar, at 0.5


def test_an_avatar_layer_without_a_folder_is_an_error(tmp_path):
    spec = _talking(tmp_path, "vo", audio=[{"id": "vo", "src": "vo.wav", "words": []}])
    spec["layers"][0]["avatar"] = "nobody"
    _wav(tmp_path / "vo.wav")
    with pytest.raises(Exception, match="layer 'host'.*no avatar.json"):
        import kinescript as k
        k.Scene(spec, tmp_path)


def test_avatar_frames_are_a_pure_function_of_the_frame_number(tmp_path):
    import numpy as np
    import kinescript as k
    from kinescript.render import Renderer
    _wav(tmp_path / "vo.wav")
    words = [{"word": "Make", "start": 0.1, "end": 0.45}, {"word": "move", "start": 0.6, "end": 0.95}]
    spec = _talking(tmp_path, "vo", audio=[{"id": "vo", "src": "vo.wav", "words": words}], at=0)
    r = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 30)]
    shuffled = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    for n in (23, 4, 17, 9):
        assert np.array_equal(shuffled.frame(n), sequential[n])


def test_each_line_of_a_say_list_moves_the_mouth_in_turn(tmp_path):
    _wav(tmp_path / "a.wav", 1.0)
    _wav(tmp_path / "b.wav", 1.0)
    spec = _talking(tmp_path, ["a", "b"], audio=[
        {"id": "a", "src": "a.wav", "at": 1.0, "words": [{"word": "ma", "start": 0.2, "end": 0.6}]},
        {"id": "b", "src": "b.wav", "at": 2.5, "words": [{"word": "fa", "start": 0.2, "end": 0.6}]}])
    assert _mouth_at(spec, tmp_path, 1.22) == ["A"]
    assert _mouth_at(spec, tmp_path, 2.0) == ["X"]  # between the lines
    assert _mouth_at(spec, tmp_path, 2.72) == ["A"]  # the F of "fa" shows its fallback, A
    assert _mouth_at(spec, tmp_path, 2.9) == ["C"]


def _moving(tmp_path, **over):
    make(tmp_path / "host", gestures={"nod": {"dur": 0.5, "head": {"rotate": [[0, 0], [0.2, 6], [0.5, 0]]}},
                                      "tilt": {"dur": 1.0, "head": {"rotate": [[0, 0], [0.5, -8], [1.0, 0]]}}})
    host = {"id": "host", "use": "avatar", "avatar": "host", "h": 150, "at": 0.5, "idle": False, **over}
    return {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 4}, "layers": [host]}


def _at(spec, base, t, lid):
    from kinescript.verify import probe
    f = base / "s.json"
    f.write_text(json.dumps(spec))
    return probe(str(f), t)["visible"].get(lid)


def _turn(spec, base, t):
    """How far the head's art is turned at t, from the width of its box (a turned box is wider)."""
    r = _at(spec, base, t, "host-rig-head-art")["rect"]
    return round(r[2] - r[0], 1)


def test_an_expression_comes_on_at_its_time_and_goes_after_its_dur(tmp_path):
    spec = _moving(tmp_path, expressions=[{"name": "smile", "at": 1, "dur": 1}])
    still = _turn(spec, tmp_path, 1.2)
    assert _turn(spec, tmp_path, 1.45) == still  # it starts 1 s after the avatar, at 1.5
    assert _turn(spec, tmp_path, 2.0) > still
    assert _turn(spec, tmp_path, 2.6) == still
    assert _mouth_at(spec, tmp_path, 2.0) == ["C"]  # the smile's mouth, while nothing is said
    assert _mouth_at(spec, tmp_path, 2.6) == ["X"]


def test_half_an_expression_turns_half_as_far(tmp_path):
    full = _moving(tmp_path, expressions=[{"name": "smile", "at": 0}])
    half = _moving(tmp_path, expressions=[{"name": "smile", "at": 0, "weight": 0.5}])
    still = _turn(_moving(tmp_path), tmp_path, 2)
    a, b = _turn(full, tmp_path, 2) - still, _turn(half, tmp_path, 2) - still
    assert a > 0 and b == pytest.approx(a / 2, abs=0.3)


def test_speech_owns_the_mouth_while_an_expression_shows_its_mouth_at_rest(tmp_path):
    _wav(tmp_path / "vo.wav")
    words = [{"word": "Make", "start": 0.5, "end": 0.85}, {"word": "move", "start": 1.5, "end": 1.9}]
    spec = _moving(tmp_path, say="vo", expressions=[{"name": "smile", "at": "vo.in", "dur": 3}])
    spec["audio"] = [{"id": "vo", "src": "vo.wav", "at": 1.0, "words": words}]
    assert _mouth_at(spec, tmp_path, 0.8) == ["X"]
    assert _mouth_at(spec, tmp_path, 1.2) == ["C"]  # the smile, before the first word
    assert _mouth_at(spec, tmp_path, 1.52) == ["A"]  # the M of "Make": speech wins
    assert _mouth_at(spec, tmp_path, 2.2) == ["C"]  # the pause between words rests on the smile


def test_a_gesture_plays_its_keys_from_its_cue(tmp_path):
    spec = _moving(tmp_path, gestures=[{"name": "nod", "at": "host.in+2"}])
    still = _turn(spec, tmp_path, 2.4)
    assert _turn(spec, tmp_path, 2.7) > still  # its peak, 0.2 s into it
    assert _turn(spec, tmp_path, 3.1) == still


def test_two_gestures_on_one_part_at_once_are_an_error(tmp_path):
    import kinescript as k
    spec = _moving(tmp_path, gestures=[{"name": "tilt", "at": 1}, {"name": "nod", "at": 1.5}])
    with pytest.raises(Exception, match=r"gestures 'tilt' \(1.50s to 2.50s\) and 'nod' .* both move 'head'"):
        k.Scene(spec, tmp_path)
    spec = _moving(tmp_path, gestures=[{"name": "tilt", "at": 1}, {"name": "nod", "at": 2}])
    k.Scene(spec, tmp_path)  # one after the other


def test_gaze_moves_the_eyes_by_their_reach_and_back(tmp_path):
    spec = _moving(tmp_path, gaze=[{"at": 0.5, "target": "right"}, {"at": 2.5, "target": "camera"}])
    x = lambda t: _at(spec, tmp_path, t, "host-rig-eyes-open-1")["pos"][0]
    ahead = x(0.9)
    assert x(1.3) == pytest.approx(ahead + 6 * 0.5)  # look reach 6, at half size
    assert x(3.3) == pytest.approx(ahead)


def test_gaze_at_a_layer_looks_its_way(tmp_path):
    spec = _moving(tmp_path, x=100, gaze=[{"at": 0, "target": "#chart"}])
    spec["layers"].append({"id": "chart", "type": "rect", "x": 300, "y": 20, "w": 10, "h": 10})
    from kinescript import avatar as A2
    import kinescript as k
    gx, gy = A2._gaze("#chart", spec["layers"][0], A.load(tmp_path / "host"), 0.5, k.Scene(spec, tmp_path), "")
    assert gx > 0.9 and gy < 0  # right and up: the head's pivot is at y 95, the chart at 20
    x = _at(spec, tmp_path, 1.0, "host-rig-eyes-open-1")["pos"][0]
    assert x == pytest.approx(_at(_moving(tmp_path, x=100), tmp_path, 1.0, "host-rig-eyes-open-1")["pos"][0] + 3 * gx, abs=0.1)


@pytest.mark.parametrize("over, message", [
    ({"gestures": [{"name": "wave", "at": 1}]}, "names gesture 'wave', which the avatar does not have. Its gestures: "
                                                "nod, tilt"),
    ({"expressions": [{"name": "smile", "when": 1}]}, r"expressions\[0\] takes at, dur, name, weight; not when"),
    ({"expressions": [{"name": "smile", "dur": 0}]}, "dur is 0; it takes a number above 0"),
    ({"gaze": [{"at": 0, "target": "sky"}]}, "target is 'sky'"),
    ({"gaze": [{"at": 0, "target": "#nobody"}]}, "no layer at the top of the scene has id 'nobody'"),
    ({"gestures": "nod"}, r"gestures\[0\] must be an object"),
])
def test_motion_mistakes_are_named(tmp_path, over, message):
    import kinescript as k
    with pytest.raises(Exception, match=message):
        k.Scene(_moving(tmp_path, **over), tmp_path)


def test_a_moving_avatar_renders_the_same_frames_in_any_order(tmp_path):
    import numpy as np
    import kinescript as k
    from kinescript.render import Renderer
    spec = _moving(tmp_path, at=0, expressions=[{"name": "smile", "at": 0.2, "dur": 0.4}],
                   gaze=[{"at": 0.1, "target": "left"}], gestures=[{"name": "nod", "at": 0.3}])
    r = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 30)]
    assert not np.array_equal(sequential[0], sequential[12])
    shuffled = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    for n in (23, 4, 17, 9, 12):
        assert np.array_equal(shuffled.frame(n), sequential[n])


def _issues(spec, base):
    import kinescript as k
    return {i["code"]: i for i in k.Scene(spec, base).issues if i["code"].startswith(("mouth-", "avatar-"))}


def test_check_says_when_the_mouth_only_follows_loudness(tmp_path):
    _wav(tmp_path / "vo.wav")
    got = _issues(_talking(tmp_path, "vo", audio=[{"id": "vo", "src": "vo.wav"}]), tmp_path)
    assert got["mouth-approximate"]["level"] == "info" and "only how loud 'vo' is" in got["mouth-approximate"]["message"]


def test_check_names_words_the_rules_cannot_read_and_shapes_the_avatar_lacks(tmp_path):
    _wav(tmp_path / "vo.wav")
    words = [{"word": "35", "start": 0.1, "end": 0.5}, {"word": "fee", "start": 0.6, "end": 1.0}]
    got = _issues(_talking(tmp_path, "vo", audio=[{"id": "vo", "src": "vo.wav", "words": words}]), tmp_path)
    assert "cannot read 1 word of 'vo' (35)" in got["mouth-unknown-words"]["message"]
    assert "G (1x, shown as A)" in got["avatar-missing-shape"]["message"]  # the F of "fee"
    assert "mouth-approximate" not in got


def test_check_says_a_stored_mouth_track_is_older_than_its_words(tmp_path):
    from kinescript import mouth as M
    _wav(tmp_path / "vo.wav")
    wf = tmp_path / "vo.words.json"
    wf.write_text(json.dumps([{"word": "Make", "start": 0.1, "end": 0.5}]))
    M.write(wf)
    spec = _talking(tmp_path, "vo", audio=[{"id": "vo", "src": "vo.wav", "words": "vo.words.json"}])
    assert "mouth-stale" not in _issues(spec, tmp_path)
    wf.write_text(json.dumps([{"word": "Move", "start": 0.1, "end": 0.5}]))
    assert "vo.mouth.json was made from other words" in _issues(spec, tmp_path)["mouth-stale"]["message"]


def test_a_gaze_target_mistake_has_its_own_code(tmp_path):
    import kinescript as k
    with pytest.raises(k.SceneError) as e:
        k.Scene(_moving(tmp_path, gaze=[{"at": 0, "target": "#nobody"}]), tmp_path)
    assert [i["code"] for i in e.value.issues if i["level"] == "error"] == ["avatar-gaze-target"]


def _living(tmp_path, dur=12, **over):
    spec = _moving(tmp_path, at=0, **over)
    spec["layers"][0].pop("idle") if "idle" not in over else None
    spec["canvas"]["duration"] = dur
    return spec


def _blink_times(spec, base):
    import kinescript as k
    keys = k.Scene(spec, base).by_id["host-rig-eyes-closed"].raw["opacity"]
    return [t for t, v, _ in keys if v == 1]


def test_the_avatar_blinks_at_seeded_times_within_its_gaps(tmp_path):
    spec = _living(tmp_path)
    times = _blink_times(spec, tmp_path)
    assert 2 <= len(times) <= 8 and times == _blink_times(spec, tmp_path)
    assert all(b - a >= 2.5 or b - a < 0.3 for a, b in zip(times, times[1:]))  # a gap, or a double blink
    assert _blink_times(_living(tmp_path, seed=7), tmp_path) != times
    b = times[0]
    assert _at(spec, tmp_path, b + 0.05, "host-rig-eyes-closed") and not _at(spec, tmp_path, b + 0.05, "host-rig-eyes-open")
    assert _at(spec, tmp_path, b + 0.2, "host-rig-eyes-open")


def test_no_blink_shows_as_a_line_starts_or_at_a_gesture_peak(tmp_path):
    b = _blink_times(_living(tmp_path), tmp_path)[0]
    _wav(tmp_path / "vo.wav", 1.0)
    spec = _living(tmp_path, say="vo", gestures=[{"name": "nod", "at": round(b - 0.2, 3)}])  # its peak at b
    spec["audio"] = [{"id": "vo", "src": "vo.wav", "at": round(b - 0.05, 3), "words": []}]
    assert _blink_times(spec, tmp_path)[0] == b  # the same blink, kept from showing
    assert not _at(spec, tmp_path, b + 0.05, "host-rig-eyes-closed")
    assert _at(spec, tmp_path, b + 0.05, "host-rig-eyes-open-quiet1") or _at(spec, tmp_path, b + 0.05,
                                                                             "host-rig-eyes-open-quiet2")


def test_breath_swells_the_body_and_idle_false_keeps_it_still(tmp_path):
    y = lambda spec, t: _at(spec, tmp_path, t, "host-rig-head-art")["rect"][1]
    spec = _living(tmp_path, gaze=[{"at": 0, "target": "camera"}])
    assert y(spec, 2.0) < y(spec, 0.0)  # half a breath in, the head is higher
    still = _living(tmp_path, idle={"breath": False})
    assert y(still, 2.0) == y(still, 0.0)


def test_eyes_glance_when_idle_and_explicit_gaze_stops_it(tmp_path):
    import kinescript as k
    assert "host-glance-eyes" in k.Scene(_living(tmp_path), tmp_path).by_id
    assert "host-glance-eyes" not in k.Scene(_living(tmp_path, gaze=[{"at": 0, "target": "left"}]), tmp_path).by_id
    spec = _living(tmp_path)
    make(tmp_path / "host", idle=False)
    s = k.Scene(spec, tmp_path)
    assert "host-glance-eyes" not in s.by_id and "host-breath-body" not in s.by_id
    assert not isinstance(s.by_id["host-rig-eyes-closed"].raw.get("opacity"), list)


def test_a_wrong_idle_on_the_layer_is_named(tmp_path):
    import kinescript as k
    with pytest.raises(Exception, match="idle is false, or an object of blink, breath and saccade"):
        k.Scene(_living(tmp_path, idle={"yawn": True}), tmp_path)


def test_a_living_avatar_renders_the_same_frames_in_any_order_and_any_chunk(tmp_path):
    import numpy as np
    import kinescript as k
    from kinescript.render import Renderer
    spec = _living(tmp_path, dur=6)
    frames = [n * 6 for n in range(30)]
    sequential = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    want = {n: sequential.frame(n).copy() for n in frames}
    assert any(not np.array_equal(want[0], v) for v in want.values())
    for chunk in (frames[15:][::-1], frames[:15][::3]):  # fresh renderers, as parallel chunks start
        r = Renderer(k.Scene(spec, tmp_path), backend="cpu")
        for n in chunk:
            assert np.array_equal(r.frame(n), want[n])


def test_blinks_keep_their_times_inside_an_expression_on_the_head(tmp_path):
    spec = _living(tmp_path, expressions=[{"name": "smile", "at": 0}])
    b = _blink_times(spec, tmp_path)[0]
    assert _at(spec, tmp_path, b + 0.05, "host-rig-eyes-closed")
    assert not _at(spec, tmp_path, b - 0.3, "host-rig-eyes-closed")


def test_the_starter_avatar_is_valid_and_draws_every_mouth_shape():
    folder = A.where("kine:host")
    assert A.starters() == ["host"] and A.validate(folder) == []
    av = A.load(folder)
    assert sorted(av.mouth.files) == sorted("XABCDEFGH") and av.fallbacks == {}
    roles = {k: v for k, v in av.spec["colors"].items() if v in A.B.ROLES}
    assert set(roles) == {"line", "shirt", "hair", "brow"}  # these follow the palette; the face keeps its own
    assert av.spec["voice"]["backend"] == "kokoro"


def test_the_starter_files_are_what_make_host_draws():
    import runpy
    made = runpy.run_path(str(Path(__file__).parents[1] / "brand" / "make_host.py"))["files"]()
    for rel, text in made.items():
        assert (A.BUILT_IN / "host" / rel).read_text(encoding="utf-8") == text, f"run python brand/make_host.py ({rel})"


def test_the_starter_takes_the_palette_it_is_drawn_with():
    av = A.load(A.where("kine:host"))
    light = A.colors(av, {"text": "#101010", "bg": "#fafafa", "accent": "#2255ff", "muted": "#777777"})
    assert light["shirt"] == "#2255ff" and light["line"] == "#777777"
    assert light["skin"] == "#F2E8DC" and light["feature"] == "#1E2229"  # never the palette's text and bg


def test_a_scene_can_use_the_starter_by_name_with_its_voice(tmp_path):
    import kinescript as k
    spec = {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 2},
            "layers": [{"id": "host", "use": "avatar", "avatar": "kine:host", "h": 160, "say": {"text": "Hi"}}]}
    assert A.say_sounds(spec, tmp_path)["audio"][0]["speaker"] == "af_heart"
    spec["layers"][0].pop("say")
    k.Scene(spec, tmp_path)
    spec["layers"][0]["avatar"] = "kine:nobody"
    with pytest.raises(Exception, match="no built-in avatar 'kine:nobody'; built in: kine:host"):
        k.Scene(spec, tmp_path)


def test_kine_avatar_new_copies_the_starter_to_edit(tmp_path, capsys):
    from kinescript.cli import main
    assert main(["avatar", "new", str(tmp_path / "mine")]) == 0
    assert A.validate(tmp_path / "mine") == [] and (tmp_path / "mine" / "parts" / "mouth-H.svg").exists()
    assert main(["avatar", "new", str(tmp_path / "mine")]) != 0
    assert "already holds an avatar" in capsys.readouterr().out
    assert main(["avatar", "check", "kine:host"]) == 0
    assert main(["look", "kine:host", "-o", str(tmp_path / "host.png")]) == 0 and (tmp_path / "host.png").exists()


def _branded(tmp_path, avatars, **layer):
    from test_brand import make_brand
    folder = make_brand(tmp_path / "brand", avatars=avatars)
    host = {"id": "host", "use": "avatar", "avatar": "brand:host", "h": 160, **layer}
    return folder, {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 2}, "brand": "brand", "layers": [host]}


def test_a_brand_names_its_avatar_with_the_brands_voice_and_palette(tmp_path):
    import kinescript as k
    folder, spec = _branded(tmp_path, {"host": {"src": "kine:host", "voice": {"speaker": "am_michael"}}},
                            say={"text": "Hi"})
    assert A.say_sounds(spec, tmp_path)["audio"][0]["speaker"] == "am_michael"  # the brand's over the avatar's
    spec["layers"][0].pop("say")
    s = k.Scene(spec, tmp_path)
    fills = {str(L.raw.get("fill")).lower() for i, L in s.by_id.items() if i.startswith("host-rig-body-art")}
    assert "#00a3e0" in fills  # the shirt takes the brand's accent
    from kinescript import brand as Bl
    assert Bl.validate(folder) == []


def test_a_brand_avatar_folder_lives_in_the_brand_and_is_checked(tmp_path):
    import kinescript as k
    from kinescript import brand as Bl
    folder, spec = _branded(tmp_path, {"host": "avatars/host"})
    assert "avatars.host.src: no file at" in " ".join(i["message"] for i in Bl.validate(folder))
    make(folder / "avatars" / "host")
    assert Bl.validate(folder) == []
    k.Scene(spec, tmp_path)
    make(folder / "avatars" / "host", size=[0, 0])
    assert any(i["code"] == "brand-avatar" and "size" in i["message"] for i in Bl.validate(folder))


def test_a_brand_avatars_idle_sits_between_the_avatars_and_the_layers(tmp_path):
    import kinescript as k
    _, spec = _branded(tmp_path, {"host": {"src": "kine:host", "idle": {"saccade": False}}})
    assert "host-glance-eyes" not in k.Scene(spec, tmp_path).by_id
    spec["layers"][0]["idle"] = {"saccade": [1, 2]}
    assert "host-glance-eyes" in k.Scene(spec, tmp_path).by_id


@pytest.mark.parametrize("avatars, name, message", [
    ({"host": "kine:host"}, "brand:guest", "has no avatar 'guest'; its avatars: host"),
    ({"host": {"src": "kine:host", "voice": {"pitch": 2}}}, None, "avatars.host.voice takes"),
    ({"host": {"voice": {}}}, None, "avatars.host needs src"),
    ({"host": "../outside"}, None, "leads out of the brand folder"),
])
def test_brand_avatar_mistakes_are_named(tmp_path, avatars, name, message):
    import kinescript as k
    from kinescript import brand as Bl
    folder, spec = _branded(tmp_path, avatars)
    if name:
        spec["layers"][0]["avatar"] = name
        with pytest.raises(Exception, match=message):
            k.Scene(spec, tmp_path)
    else:
        assert message in " ".join(i["message"] for i in Bl.validate(folder))


def test_a_brand_avatar_in_a_scene_without_a_brand_is_named(tmp_path):
    import kinescript as k
    _, spec = _branded(tmp_path, {"host": "kine:host"})
    spec.pop("brand")
    with pytest.raises(Exception, match="'brand:host' is a brand's avatar, and the scene names no brand"):
        k.Scene(spec, tmp_path)


@pytest.mark.parametrize("over, message", [
    ({"expressions": {"x": {"head": {"state": "open"}}}}, "which has none"),
    ({"expressions": {"x": {"eyes": {"state": "wink"}}}}, "has no state 'wink'"),
    ({"gestures": {"x": {"dur": 1, "eyes": {"state": "closed"}}}}, "an expression can pick a state"),
])
def test_state_mistakes_are_named(tmp_path, over, message):
    issues = A.validate(make(tmp_path, **over))
    assert any(message in i["message"] + i["fix"] for i in issues if i["level"] == "error"), issues


def test_an_expression_state_shows_that_drawing_in_a_still(tmp_path):
    make(tmp_path, expressions={"sleepy": {"eyes": {"state": "closed"}}})
    av = A.load(tmp_path)
    g = A.layers(av, "a", {"parts": {"eyes": {"state": "closed"}}})
    eyes = A._find(g, "a-eyes")
    shown = {c["id"]: c.get("opacity", 1) for c in eyes["children"]}
    assert shown == {"a-eyes-open": 0, "a-eyes-closed": 1}


def _sparkly(tmp_path, **over):
    """The test avatar with an effect part, sparkle, that the expression joy and the gesture pop show."""
    folder = tmp_path / "host"
    make(folder, expressions={"joy": {"sparkle": {"scale": 1.2}}},
         gestures={"pop": {"dur": 0.6, "sparkle": {"y": [[0, 0], [0.3, -10], [0.6, 0]]}}})
    (folder / "parts" / "sparkle.svg").write_text(_svg('<path d="M160 40 l6 14 14 6 -14 6 -6 14 -6 -14 -14 -6 '
                                                       '14 -6z" fill="#b9f45d"/>'), encoding="utf-8")
    spec = json.loads((folder / "avatar.json").read_text())
    spec["parts"]["sparkle"] = {"src": "parts/sparkle.svg", "parent": "body", "pivot": [160, 60], "effect": True}
    spec.update(over)
    (folder / "avatar.json").write_text(json.dumps(spec), encoding="utf-8")
    return folder


def _shows(spec, base, t):
    from kinescript.verify import probe
    f = base / "s.json"
    f.write_text(json.dumps(spec))
    seen = probe(str(f), t)["visible"]
    return sorted(k for k, v in seen.items() if k.startswith("host-rig-sparkle-art") and v["type"] == "group")


def test_an_effect_part_shows_only_while_an_expression_or_gesture_names_it(tmp_path):
    spec = _moving(tmp_path, expressions=[{"name": "joy", "at": 1, "dur": 1}], gestures=[{"name": "pop", "at": 2.5}])
    assert not codes(_sparkly(tmp_path))  # after _moving, which writes the folder without the effect
    assert _shows(spec, tmp_path, 1.2) == []
    assert _shows(spec, tmp_path, 2.0) == ["host-rig-sparkle-art-host-expr1-sparkle"]
    assert _shows(spec, tmp_path, 2.7) == []
    assert _shows(spec, tmp_path, 3.25) == ["host-rig-sparkle-art-host-gesture1-sparkle"]
    assert _shows(spec, tmp_path, 3.8) == []


def test_an_effect_shows_in_its_poses_on_the_sheet_and_not_at_rest(tmp_path):
    av = A.load(_sparkly(tmp_path))
    hidden = lambda pose: A._find(A.layers(av, "a", pose), "a-sparkle-art").get("opacity", 1) == 0  # noqa: E731
    assert hidden({}) and hidden({"mouth": "D"})
    names = dict(A.poses(av))
    assert not hidden(names["expression joy"]) and not hidden(names["gesture pop at 0.3s"])


@pytest.mark.parametrize("change, message", [
    (lambda p: p["sparkle"].update(effect="yes"), "effect is true or false"),
    (lambda p: p.update(sparkle={"states": {"open": "parts/sparkle.svg"}, "parent": "body", "effect": True}),
     "only a part with src"),
    (lambda p: p["mouth"].update(parent="sparkle"), "an effect holds no other parts"),
])
def test_effect_mistakes_are_named(tmp_path, change, message):
    folder = _sparkly(tmp_path)
    spec = json.loads((folder / "avatar.json").read_text())
    change(spec["parts"])
    (folder / "avatar.json").write_text(json.dumps(spec), encoding="utf-8")
    issues = A.validate(folder)
    assert any(message in i["message"] + i["fix"] for i in issues if i["level"] == "error"), issues


def test_a_part_named_like_a_sheet_cell_still_draws_on_the_sheet(tmp_path):
    spec = json.loads((make(tmp_path) / "avatar.json").read_text())
    spec["parts"]["label"] = spec["parts"].pop("head")
    for p in spec["parts"].values():
        if p.get("parent") == "head":
            p["parent"] = "label"
    spec["expressions"], spec["gestures"] = {}, {}
    (tmp_path / "avatar.json").write_text(json.dumps(spec), encoding="utf-8")
    assert not codes(tmp_path)
    assert A.sheet(tmp_path, str(tmp_path / "sheet.png"))["out"]


@pytest.mark.parametrize("name", ["cue", "maya", "rin", "slate"])
def test_the_example_avatars_are_valid(name):
    folder = Path(__file__).parent.parent / "examples" / "avatars" / name
    assert [i for i in A.validate(folder) if i["level"] == "error"] == []


def test_the_drawn_example_avatars_match_their_generator(tmp_path, monkeypatch):
    import importlib.util
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("make_mascots", root / "brand" / "make_mascots.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    mod.cue()
    mod.slate()
    for made in tmp_path.rglob("*"):
        if made.is_file():
            kept = root / "examples" / "avatars" / made.relative_to(tmp_path)
            assert kept.read_text(encoding="utf-8") == made.read_text(encoding="utf-8"), kept
