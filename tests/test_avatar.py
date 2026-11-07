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
    host = {"id": "host", "use": "avatar", "avatar": "host", "h": 150, "at": 0.5, **over}
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
