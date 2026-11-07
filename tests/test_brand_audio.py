"""A brand's sound defaults: voice speaker and backend, gains and fades by role, the mix, the music bed, and a
sonic logo on the intro. kine voice and the scene read sounds the same way."""
import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from kinescript import bookends, voice
from kinescript import brand as B
from kinescript import brandops as ops
from kinescript.output import render
from kinescript.scene import Scene, SceneError
from kinescript.verify import check

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "brand"


def tone(path: Path, secs=1.0, hz=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / 48000)))
                               for i in range(int(48000 * secs))))


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    shutil.copytree(EXAMPLE, tmp_path / "brand")
    tone(tmp_path / "brand" / "audio" / "bed.wav", 2.0, 220)
    tone(tmp_path / "brand" / "audio" / "logo.wav", 0.5, 880)
    spec = B.read(tmp_path / "brand")
    spec["audio"] = {"voice": {"speaker": "bf_emma", "backend": "kokoro", "rate": 1.1},
                     "music": {"src": "audio/bed.wav", "gain": -4, "fadeIn": 0.5},
                     "sfx": {"gain": -3}, "mix": {"loudness": -14, "duck": {"depth": 10}}}
    spec["intro"]["sound"] = "audio/logo.wav"
    (tmp_path / "brand" / "brand.json").write_text(json.dumps(spec), encoding="utf-8")
    return tmp_path


def scene(proj, audio=(), **top) -> dict:
    spec = {"canvas": {"w": 320, "h": 180, "fps": 10, "duration": 1}, "brand": "brand",
            "layers": [{"id": "t", "type": "rect", "w": 40, "h": 40, "x": 160, "y": 90, "fill": "$accent"}],
            "audio": list(audio), **top}
    (proj / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_brand_validates_its_audio(proj):
    assert [i for i in B.validate(proj / "brand") if i["level"] == "error"] == []
    spec = B.read(proj / "brand")
    spec["audio"]["music"]["src"] = "audio/none.wav"
    spec["audio"]["choir"] = {}
    spec["audio"]["sfx"]["speaker"] = "x"
    issues = B.validate(proj / "brand", spec)
    assert any(i["code"] == "missing-media" and "audio.music.src" in i["message"] for i in issues)
    assert any("audio.choir" in i["message"] for i in issues)
    assert any("audio.sfx.speaker" in i["message"] for i in issues)


def test_sounds_take_their_roles_defaults(proj):
    tone(proj / "hit.wav", 0.3)
    s = Scene(scene(proj, [{"id": "vo", "role": "voice", "text": "Hello there", "src": "hit.wav"},
                           {"id": "vo2", "role": "voice", "text": "Hi", "speaker": "am_adam", "src": "hit.wav"},
                           {"id": "hit", "role": "sfx", "src": "hit.wav"},
                           {"id": "loud", "role": "sfx", "src": "hit.wav", "gain": 2}]), str(proj))
    by = {a["id"]: a for a in s.spec["audio"]}
    assert by["vo"]["speaker"] == "bf_emma" and by["vo"]["backend"] == "kokoro" and by["vo"]["rate"] == 1.1
    assert by["vo2"]["speaker"] == "am_adam" and by["vo2"]["rate"] == 1.1
    assert by["hit"]["gain"] == -3 and by["loud"]["gain"] == 2
    assert s.spec["mix"] == {"loudness": -14, "duck": {"depth": 10}}
    s = Scene(scene(proj, mix={"duck": {"attack": 0.1}, "peak": -1}), str(proj))
    assert s.spec["mix"] == {"loudness": -14, "duck": {"depth": 10, "attack": 0.1}, "peak": -1}


def test_kine_voice_reads_the_same_defaults_so_a_new_speaker_is_stale(proj):
    scene(proj, [{"id": "vo", "role": "voice", "text": "Hello there"}])
    spec, base = voice._spec(str(proj / "s.json"), None)
    (sid, a), = voice.voices(spec)
    assert a["speaker"] == "bf_emma"
    codes = {i["code"] for i in check(str(proj / "s.json"))["issues"]}
    assert "voice-missing" in codes
    src, words = voice.paths(sid, a, base)
    tone(src, 0.6)
    words.write_text(json.dumps({"words": [{"word": "Hello", "start": 0.0, "end": 0.3}],
                                 "made": {"hash": voice.fingerprint(a)}}), encoding="utf-8")
    codes = {i["code"] for i in check(str(proj / "s.json"))["issues"]}
    assert "voice-missing" not in codes and "voice-stale" not in codes
    ops.set_value(str(proj / "brand"), "audio.voice.speaker", "af_heart")
    codes = {i["code"] for i in check(str(proj / "s.json"))["issues"]}
    assert "voice-stale" in codes


def test_music_bed(proj):
    s = Scene(scene(proj, music=True), str(proj))
    bed, = [a for a in s.spec["audio"] if a["id"] == "music"]
    assert Path(bed["src"]) == (proj / "brand" / "audio" / "bed.wav").resolve()
    assert bed["loop"] is True and bed["until"] == "end" and bed["gain"] == -4 and bed["fadeIn"] == 0.5
    s = Scene(scene(proj, music={"gain": -12}), str(proj))
    assert next(a for a in s.spec["audio"] if a["id"] == "music")["gain"] == -12
    tone(proj / "mine.wav", 0.5)
    s = Scene(scene(proj, [{"id": "music", "role": "music", "src": "mine.wav"}], music=True), str(proj))
    assert [a["src"] for a in s.spec["audio"]] == ["mine.wav"]  # the scene's own music stays
    res = check(str(proj / "s.json"))
    assert res["errors"] == 0
    scene(proj, music=True)
    res = render(str(proj / "s.json"), str(proj / "out.mp4"), workers=1)
    assert res["audio"]


def test_music_without_a_bed_is_an_error(proj):
    spec = B.read(proj / "brand")
    del spec["audio"]["music"]
    (proj / "brand" / "brand.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(SceneError) as e:
        Scene(scene(proj, music=True), str(proj))
    assert any(i["code"] == "brand-audio" for i in e.value.issues)


def test_sonic_logo_plays_on_the_intro(proj):
    s = Scene(scene(proj, intro=True), str(proj))
    seg, _ = bookends.segment(s, "intro")
    snd, = seg["audio"]
    assert snd["role"] == "sfx" and Path(snd["src"]) == (proj / "brand" / "audio" / "logo.wav").resolve()
    intro = Scene(*bookends.segment(s, "intro"))
    assert intro.spec["audio"][0]["gain"] == -3  # the brand's sfx default reaches it too
    res = render(str(proj / "s.json"), str(proj / "out.mp4"), workers=1)
    assert res["audio"] and res["intro"]["seconds"] == 2.0
    spec = B.read(proj / "brand")
    spec["intro"]["sound"] = "../outside.wav"
    assert any(i["code"] == "brand-path" for i in B.validate(proj / "brand", spec))


def test_audio_upload_is_sniffed(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BRANDS", str(tmp_path / "lib"))
    ops.init("acme")
    tone(tmp_path / "a.wav", 0.1)
    out = ops.add_file("acme", "audio", "a.wav", (tmp_path / "a.wav").read_bytes())
    assert out["file"] == "audio/a.wav"
    with pytest.raises(ops.BrandOpError, match="first bytes"):
        ops.add_file("acme", "audio", "b.mp3", b"<html>not sound</html>")
    with pytest.raises(ops.BrandOpError, match="take"):
        ops.add_file("acme", "audio", "c.exe", b"RIFF0000WAVE")
    ops.set_value("acme", "audio", {"music": {"src": "audio/a.wav"}})
    with pytest.raises(ops.BrandOpError, match="still named"):
        ops.remove_file("acme", "audio/a.wav")
