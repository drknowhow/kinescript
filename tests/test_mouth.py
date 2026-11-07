import json

import pytest

from kinescript import mouth as M
from kinescript import voice


def _gapless(track, end):
    assert track[0][1] == 0.0 and track[-1][2] == pytest.approx(end)
    for (_, _, b), (_, a, _) in zip(track, track[1:]):
        assert a == pytest.approx(b)
    assert all(s in M.SHAPES for s, _, _ in track)


def test_spelling_rules_read_common_patterns():
    assert M.spell("make") == ["M", "EY", "K"]
    assert M.spell("cell") == ["S", "EH", "L"]
    assert M.spell("scene") == ["S", "IY", "N"]
    assert M.spell("Night,") == ["N", "AY", "T"]
    assert M.spell("you") == ["Y", "UW"]
    assert M.spell("35") is None


def test_track_covers_the_line_and_closes_only_on_long_pauses():
    words = [{"word": "Make", "start": 0.2, "end": 0.5}, {"word": "it", "start": 0.55, "end": 0.7},
             {"word": "move.", "start": 1.2, "end": 1.6}]
    got = M.track(words)
    assert got["source"] == "rules" and got["unknown"] == []
    t = got["mouth"]
    _gapless(t, 1.6)
    assert t[0] == ["X", 0.0, 0.2]
    assert ["X", 0.7, 1.2] in t  # 0.5 s between "it" and "move" closes the mouth
    assert not any(s == "X" and a == pytest.approx(0.5) for s, a, _ in t)  # 0.05 s between words does not
    assert t[1][0] == "A"  # M
    assert M.track(words) == got


def test_words_the_rules_cannot_read_are_reported_and_still_move_the_mouth():
    got = M.track([{"word": "35", "start": 0.0, "end": 0.4}])
    assert got["unknown"] == ["35"] and got["mouth"] == [["C", 0.0, 0.4]]


def test_short_shapes_merge_but_a_closure_between_sounds_is_kept():
    for s, a, b in M.track([{"word": "ambient", "start": 0.0, "end": 0.3}])["mouth"]:
        assert b - a >= M.MIN - 1e-9 or s == "X"
    t = M._tidy([["D", 0.0, 0.2], ["A", 0.2, 0.22], ["D", 0.22, 0.4]], 0.4)
    assert [s for s, _, _ in t] == ["D", "A", "D"] and t[1][2] - t[1][1] == pytest.approx(M.MIN)
    _gapless(t, 0.4)


def test_backend_phones_in_either_alphabet_make_the_track():
    phones = [{"ipa": "hˈɛlO", "start": 0.1, "end": 0.5}, {"arpabet": "W ER1 L D", "start": 0.55, "end": 0.9}]
    got = M.track([], phones)
    assert got["source"] == "phones"
    _gapless(got["mouth"], 0.9)
    assert [s for s, _, _ in got["mouth"]][:4] == ["X", "B", "C", "H"]


def test_loudness_alone_opens_with_level_and_never_flickers():
    rms = [-80] * 5 + [-12, -10, -11, -30, -30, -30] + [-80] * 20 + [-11] * 6
    got = M.levels(rms, 0.02)
    assert got["source"] == "levels"
    t = got["mouth"]
    assert t[0][0] == "X" and "D" in [s for s, _, _ in t]
    assert any(s == "X" and b - a > 0.3 for s, a, b in t)
    assert all(b - a >= M.MIN - 1e-9 for s, a, b in t if s != "X")


def _scene(text):
    return {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 2}, "layers": [],
            "audio": [{"id": "vo1", "role": "voice", "text": text}]}


@pytest.fixture
def speak(monkeypatch):
    said = []

    def fn(req):
        said.append(req["text"])
        open(req["out"], "wb").write(b"RIFF")
        words = [{"word": w, "start": 0.3 * k, "end": 0.3 * k + 0.25} for k, w in enumerate(req["text"].split())]
        return {"words": words, "speaker": "x", "version": "1", "phones": fn.phones}
    fn.phones = None
    monkeypatch.delenv("KINE_VOICE_CMD", raising=False)
    monkeypatch.setattr(voice, "backends", lambda: {"fake": type("E", (), {"load": lambda self: fn})()})
    return fn, said


def test_kine_voice_writes_a_mouth_track_and_remakes_it_without_speaking_again(speak, tmp_path):
    fn, said = speak
    spec = _scene("Make it move.")
    voice.generate(spec, base=tmp_path)
    track = tmp_path / "voice" / "vo1.mouth.json"
    kept = json.loads(track.read_text(encoding="utf-8"))
    assert kept["made"]["source"] == "rules" and kept["mouth"][0][0] == "A"  # the M of "Make", at 0
    assert M.fresh(tmp_path / "voice" / "vo1.words.json")
    track.unlink()
    assert voice.generate(spec, base=tmp_path)["voices"][0]["status"] == "fresh" and len(said) == 1
    assert track.exists()


def test_kine_voice_keeps_backend_phones_and_refuses_ones_it_cannot_read(speak, tmp_path):
    fn, _ = speak
    fn.phones = [{"arpabet": "M EY1 K", "start": 0.0, "end": 0.25}]
    voice.generate(_scene("Make"), base=tmp_path)
    words = json.loads((tmp_path / "voice" / "vo1.words.json").read_text(encoding="utf-8"))
    assert words["phones"] == fn.phones
    assert json.loads((tmp_path / "voice" / "vo1.mouth.json").read_text(encoding="utf-8"))["made"]["source"] == "phones"
    fn.phones = [{"sound": "m", "start": 0, "end": 1}]
    with pytest.raises(RuntimeError, match="returned phones it cannot use at item 0"):
        voice.generate(_scene("Make"), base=tmp_path, force=True)


def test_probe_mouth_reports_the_track_and_whether_its_file_is_current(speak, tmp_path):
    from kinescript.verify import mouth
    f = tmp_path / "s.json"
    f.write_text(json.dumps(_scene("Make it move.")), encoding="utf-8")
    with pytest.raises(ValueError, match="no words and no sound file yet; run kine voice"):
        mouth(str(f), "vo1")
    voice.generate(str(f))
    got = mouth(str(f), "vo1")
    assert got["fresh"] is True and got["source"] == "rules" and got["shapes"]["A"] >= 1
    assert got["mouth"] == json.loads((tmp_path / "voice" / "vo1.mouth.json").read_text(encoding="utf-8"))["mouth"]
    with pytest.raises(ValueError, match="no sound has id vo2; sounds: vo1"):
        mouth(str(f), "vo2")
