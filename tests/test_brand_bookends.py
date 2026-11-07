"""A brand's intro and outro: "intro": true / "outro": true on a scene, rendered as segments around it."""
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from kinescript import bookends
from kinescript import brand as B
from kinescript.output import render, still
from kinescript.scene import Scene
from kinescript.verify import check, probe

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "brand"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    shutil.copytree(EXAMPLE, tmp_path / "brand")
    return tmp_path


def scene(proj, **top) -> dict:
    spec = {"canvas": {"w": 320, "h": 180, "fps": 10, "duration": 1}, "brand": "brand",
            "layers": [{"id": "t", "type": "rect", "w": 40, "h": 40, "x": 160, "y": 90, "fill": "$accent"}], **top}
    (proj / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_segments_take_the_video_size_and_the_brand_parameters(proj):
    s = Scene(scene(proj, intro=True, outro={"title": "See you"}, variant="dark"), str(proj))
    segs = bookends.segments(s)
    intro, base = segs["intro"]
    assert intro["canvas"] == {"w": 320, "h": 180, "fps": 10.0, "duration": 2}
    assert intro["variant"] == "dark" and intro["layers"][0]["use"] == "sting"
    assert intro["layers"][0]["tagline"] == "Across the sound since 1952"
    outro = segs["outro"][0]
    assert outro["layers"][0]["title"] == "See you" and outro["layers"][0]["url"] == "tidewater.example"
    assert outro["canvas"]["duration"] == 3
    assert bookends.durations(s) == {"intro": 2.0, "outro": 3.0}


def test_no_bookends_unless_asked(proj):
    s = Scene(scene(proj), str(proj))
    assert bookends.segments(s) == {} and "bookends" not in probe(str(proj / "s.json"))


def test_probe_and_check(proj):
    scene(proj, intro=True)
    assert probe(str(proj / "s.json"))["bookends"]["intro"] == 2.0
    assert check(str(proj / "s.json"))["errors"] == 0
    scene(proj, intro={"logo": "nope"})
    msgs = [i["message"] for i in check(str(proj / "s.json"))["issues"] if i["level"] == "error"]
    assert any(m.startswith("intro:") and "nope" in m for m in msgs)
    spec = scene(proj, outro=True)
    spec.pop("brand")
    (proj / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    msgs = [i["message"] for i in check(str(proj / "s.json"))["issues"] if i["level"] == "error"]
    assert any("names no brand" in m for m in msgs)


def test_bad_values_are_refused(proj):
    scene(proj, intro="yes")
    assert any(i["code"] == "bad-value" and i["layer"] == "intro" for i in check(str(proj / "s.json"))["issues"])
    spec = B.read(proj / "brand")
    spec["intro"]["dur"] = 0
    assert any("intro.dur" in i["message"] for i in B.validate(proj / "brand", spec))


def test_intro_from_a_scene_file_in_the_brand(proj):
    (proj / "brand" / "intro.json").write_text(json.dumps({
        "canvas": {"w": 1920, "h": 1080, "fps": 60, "duration": 1.5}, "intro": True,
        "layers": [{"id": "dot", "type": "circle", "r": 20, "x": "50%", "y": "50%", "fill": "$accent"}]}),
        encoding="utf-8")
    spec = B.read(proj / "brand")
    spec["intro"] = {"scene": "intro.json"}
    (proj / "brand" / "brand.json").write_text(json.dumps(spec), encoding="utf-8")
    s = Scene(scene(proj, intro=True), str(proj))
    seg, base = bookends.segment(s, "intro")
    assert seg["canvas"] == {"w": 320, "h": 180, "fps": 10.0, "duration": 1.5}
    assert "intro" not in seg and Path(base) == (proj / "brand").resolve()
    assert bookends.durations(s)["intro"] == 1.5
    spec["intro"] = {"scene": "../s.json"}
    issues = B.validate(proj / "brand", spec)
    assert any(i["code"] == "brand-path" for i in issues)


def test_sting_and_end_card_draw_as_components(proj):
    spec = {"canvas": {"w": 320, "h": 180, "fps": 10, "duration": 2}, "brand": "brand", "variant": "dark",
            "layers": [{"use": "sting", "id": "s", "dur": 1}, {"use": "endCard", "id": "e", "at": 1, "title": "Bye"}]}
    s = Scene(spec, str(proj))
    assert s.by_id["s-logo"].raw["type"] == "image" and s.by_id["s-tagline"].raw["text"].startswith("Across")
    assert s.by_id["e-title"].raw["text"] == "Bye"
    for t in (0.5, 1.8):
        still(s, t, str(proj / f"f{t}.png"))
    # no logos: the brand's name
    brand = B.read(proj / "brand")
    brand.pop("logos")
    brand.pop("bug")
    (proj / "brand" / "brand.json").write_text(json.dumps(brand), encoding="utf-8")
    s = Scene({**spec, "layers": [{"use": "sting", "id": "s"}]}, str(proj))
    assert s.by_id["s-name"].raw["text"] == brand["name"]


def tone(path: Path, secs=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 48000)))
                               for i in range(int(48000 * secs))))


def ffprobe(path) -> dict:
    from kinescript import media
    out = subprocess.run([media.FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,nb_frames,duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_render_joins_intro_video_and_outro_with_the_sound_in_place(proj):
    tone(proj / "tone.wav")
    scene(proj, intro=True, outro=True, audio=[{"id": "tone", "src": "tone.wav"}])
    res = render(str(proj / "s.json"), str(proj / "out.mp4"), workers=1)
    assert res["frames"] == 20 + 10 + 30 and res["seconds"] == 6.0
    assert res["intro"] == {"seconds": 2.0, "frames": 20} and res["outro"]["frames"] == 30 and res["audio"]
    st = ffprobe(proj / "out.mp4")
    assert int(st["video"]["nb_frames"]) == 60 and abs(float(st["audio"]["duration"]) - 6.0) < 0.06
    from kinescript import media
    log = subprocess.run([media.FFMPEG, "-v", "info", "-i", str(proj / "out.mp4"), "-af",
                          "silencedetect=n=-40dB:d=0.2", "-f", "null", "-"], capture_output=True, text=True).stderr
    ends = [float(x.split("silence_end: ")[1].split()[0]) for x in log.splitlines() if "silence_end" in x]
    assert abs(ends[0] - 2.0) < 0.05  # the tone starts with the scene, after the intro


def test_render_without_sound_and_to_gif(proj):
    scene(proj, outro=True)
    res = render(str(proj / "s.json"), str(proj / "out.mp4"), workers=1)
    assert res["frames"] == 40 and not res["audio"] and "audio" not in ffprobe(proj / "out.mp4")
    res = render(str(proj / "s.json"), str(proj / "out.gif"), workers=1)
    assert res["seconds"] == 4.0 and (proj / "out.gif").stat().st_size > 0


def test_a_frame_range_leaves_them_out(proj):
    scene(proj, intro=True)
    res = render(str(proj / "s.json"), str(proj / "part.mp4"), start=0, end=0.5, workers=1)
    assert res["frames"] == 5 and any("left out of a frame range" in n for n in res["notes"])
