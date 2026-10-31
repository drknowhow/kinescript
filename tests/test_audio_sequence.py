"""Sound, sequences and captions: the mix keeps pace with the picture, transitions and caption files are checked, and
check names the sounds nobody will hear."""
import json
import subprocess

import numpy as np
import pytest

import kinescript as k
from kinescript import media, output
from kinescript.components import load_cues
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 64, "h": 36, "fps": 30}, **canvas), "layers": layers}


def _ff(*args):
    subprocess.run([media.FFMPEG, "-y", "-v", "error", *map(str, args)], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    """lo.mp4 and hi.mp4: 2 s clips with a 330 Hz and an 880 Hz tone. steps.mp4: 4 s whose tone is 200 Hz in its
    first second, 400 in its second, 600, 800, so the frequency says which second of the file is playing.
    mute.mp4: picture only. tone.wav: 1 s of 1 kHz. still.png: an image."""
    d = tmp_path_factory.mktemp("av")
    pic = ["-f", "lavfi", "-i", "testsrc2=size=64x36:rate=30"]
    for name, expr, dur in (("lo", "0.5*sin(2*PI*330*t)", 2), ("hi", "0.5*sin(2*PI*880*t)", 2),
                            ("steps", "0.5*sin(2*PI*(200+200*floor(t))*t)", 4)):
        _ff(*pic, "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={dur}", "-t", dur, "-pix_fmt", "yuv420p",
            "-shortest", d / f"{name}.mp4")
    _ff(*pic, "-t", 1, "-pix_fmt", "yuv420p", d / "mute.mp4")
    _ff("-f", "lavfi", "-i", "sine=frequency=1000:duration=1", d / "tone.wav")
    _ff("-f", "lavfi", "-i", "color=c=red:size=16x16", "-frames:v", 1, d / "still.png")
    return {p.stem: str(p) for p in d.iterdir()}


def _pcm(path):
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a", "-ac", "1", "-ar", "8000", "-f",
                          "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def _rms(pcm, t0, t1):
    return float(np.sqrt(np.mean(pcm[int(t0 * 8000):int(t1 * 8000)] ** 2)))


def _band(pcm, t0, t1, hz):
    """Level in dB of one frequency between t0 and t1."""
    seg = pcm[int(t0 * 8000):int(t1 * 8000)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return 20 * np.log10(spec[np.argmin(np.abs(np.fft.rfftfreq(len(seg), 1 / 8000) - hz))] + 1e-12)


def _pitch(pcm, t0, t1):
    seg = pcm[int(t0 * 8000):int(t1 * 8000)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(np.fft.rfftfreq(len(seg), 1 / 8000)[np.argmax(spec)])


def _mix(spec, tmp_path, name="o.wav"):
    out = tmp_path / name
    output.render(spec, str(out))
    return _pcm(out)


def _issues(spec):
    return {i["code"]: i for i in check(spec)["issues"]}


def _errors(spec):
    with pytest.raises(k.SceneError) as e:
        k.Scene(spec).close()
    return [(i["code"], i["message"]) for i in e.value.issues if i["level"] == "error"]


# ---- C10: speed ---------------------------------------------------------------------------------------------------
def test_tempo_steps_multiply_to_any_speed_within_what_atempo_takes():
    for speed in (0.01, 0.1, 0.25, 0.5, 0.8, 1.5, 2, 4, 10, 37.5):
        steps = [float(s.split("=")[1]) for s in output._tempo(speed)]
        assert all(0.5 <= s <= 2 for s in steps) and np.prod(steps) == pytest.approx(speed, rel=1e-5)
    assert output._tempo(1) == []


def test_a_videos_sound_keeps_pace_with_its_picture_at_any_speed(media_dir, tmp_path):
    vid = {"id": "v", "type": "video", "src": media_dir["steps"], "w": 64, "h": 36}
    pcm = _mix(scene([dict(vid, speed=4)]), tmp_path)  # 4 s of file in 1 s
    assert len(pcm) == pytest.approx(8000, abs=100)
    # 0.8-0.95 s is the file's fourth second (800 Hz). atempo was clamped to 2, so it played the second (400 Hz).
    assert _pitch(pcm, 0.8, 0.95) == pytest.approx(800, abs=20)
    pcm = _mix(scene([dict(vid, speed=0.25, trim=[1, 1.5])]), tmp_path)  # half a second of file over 2 s
    assert len(pcm) == pytest.approx(16000, abs=100)
    assert _pitch(pcm, 1.6, 1.9) == pytest.approx(400, abs=20) and _rms(pcm, 1.6, 1.9) > 0.1  # was silent after 1 s


@pytest.mark.parametrize("speed", [0, -1, "2x"])
def test_a_speed_of_zero_or_less_is_an_error(media_dir, speed):
    spec = scene([{"id": "v", "type": "video", "src": media_dir["steps"], "speed": speed}])
    assert not check(spec)["ok"]


# ---- C13 and A20: transitions longer than their clips, sounds before 0 s -------------------------------------------
@pytest.mark.parametrize("dur", [1, 2, 100])
def test_a_transition_must_be_shorter_than_the_clips_it_joins(media_dir, tmp_path, dur):
    spec = scene([{"id": "cut", "type": "sequence", "clips": [{"src": media_dir["lo"], "trim": [0, 1]},
                                                              {"src": media_dir["hi"], "trim": [0, 1]}],
                   "transition": {"type": "fade", "dur": dur}}])
    got = _errors(spec)
    assert got and got[0][0] == "bad-transition" and "'cut-1' (1.00s) and 'cut-2' (1.00s)" in got[0][1], got
    r = check(spec)
    assert not r["ok"] and r["issues"][0]["code"] == "bad-transition"  # check and render agree: it was clean
    with pytest.raises(k.SceneError, match="not shorter than both clips"):  # was ffmpeg's "Delay must be non negative"
        output.render(spec, str(tmp_path / "o.mp4"))


def test_a_sound_that_starts_before_the_scene_plays_from_its_part_at_zero(media_dir, tmp_path):
    spec = dict(scene([{"type": "rect"}], duration=1.5), audio=[{"src": media_dir["tone"], "at": -0.5}])
    pcm = _mix(spec, tmp_path)  # was ffmpeg's "Delay must be non negative number"
    assert _rms(pcm, 0.05, 0.45) > 0.05 and _rms(pcm, 0.55, 1.45) < 1e-3
    vid = {"id": "v", "type": "video", "src": media_dir["lo"], "at": -1.5, "w": 64, "h": 36}
    pcm = _mix(scene([vid], duration=1), tmp_path, "v.wav")
    assert _rms(pcm, 0.05, 0.45) > 0.1 and _rms(pcm, 0.55, 0.95) < 1e-3
    assert _issues(dict(scene([{"type": "rect"}], duration=1), audio=[{"src": media_dir["tone"], "at": -3}])
                   )["audio-unheard"]["message"].startswith("'audio1' ends at -2.00s, before the scene starts")


# ---- C17: crossfades and fades ---------------------------------------------------------------------------------------
def test_a_sequence_fade_crossfades_the_sound_of_its_clips(media_dir, tmp_path):
    spec = scene([{"id": "cut", "type": "sequence", "clips": [{"src": media_dir["lo"]}, {"src": media_dir["hi"]}],
                   "transition": {"type": "fade", "dur": 1}}])
    s = k.Scene(spec)
    assert (s.by_id["cut-1"].raw["fadeOut"], s.by_id["cut-2"].raw["fadeIn"], s.duration) == (1, 1, 3)
    s.close()
    pcm = _mix(spec, tmp_path)
    lo = [_band(pcm, t, t + 0.2, 330) for t in (0.4, 1.1, 1.7, 2.3)]
    hi = [_band(pcm, t, t + 0.2, 880) for t in (0.4, 1.1, 1.7, 2.3)]
    assert lo[1] - lo[2] > 6 and lo[3] < lo[0] - 40  # the outgoing clip fades out; it stayed level, then stopped
    assert hi[2] - hi[1] > 6 and hi[0] < hi[3] - 40  # and the incoming one fades in


def test_a_videos_sound_fades_with_its_enter_and_exit_unless_fadeIn_or_fadeOut_say_otherwise(media_dir, tmp_path):
    vid = {"id": "v", "type": "video", "src": media_dir["lo"], "w": 64, "h": 36, "enter": {"name": "rise", "dur": 1},
           "exit": {"name": "fade", "dur": 0.5}}
    pcm = _mix(scene([vid]), tmp_path)
    assert _rms(pcm, 0.1, 0.2) < _rms(pcm, 1.2, 1.4) / 4 and _rms(pcm, 1.9, 1.98) < _rms(pcm, 1.2, 1.4) / 4
    pcm = _mix(scene([dict(vid, fadeIn=0, fadeOut=0)]), tmp_path, "flat.wav")
    level = _rms(pcm, 0.8, 1.4)  # an AAC tone wobbles by about 10 % from one tenth of a second to the next
    assert _rms(pcm, 0.02, 0.3) == pytest.approx(level, rel=0.15)
    assert _rms(pcm, 1.7, 1.98) == pytest.approx(level, rel=0.15)


# ---- C22: sources with no sound, or no picture --------------------------------------------------------------------
@pytest.mark.parametrize("key, says", [("mute", "a video with no sound track"), ("still", "an image")])
def test_a_sound_from_a_file_with_no_sound_is_a_check_error(media_dir, key, says):
    for spec in (dict(scene([{"type": "rect"}], duration=1), audio=[{"id": "a", "src": media_dir[key]}]),
                 scene([{"id": "r", "type": "rect", "sound": media_dir[key]}], duration=1)):
        r = check(spec)  # was clean, then the render stopped on ffmpeg's "Stream specifier ':a' ... matches no streams"
        assert not r["ok"] and r["issues"][0]["code"] == "no-audio" and says in r["issues"][0]["message"], r


def test_a_video_layer_of_a_sound_file_is_a_check_error(media_dir):
    r = check(scene([{"id": "v", "type": "video", "src": media_dir["tone"], "dur": 1}]))
    assert not r["ok"] and r["issues"][0]["code"] == "no-picture", r  # it drew nothing, with a clean check


# ---- C9, C22, C26: values of the wrong type or range --------------------------------------------------------------
@pytest.mark.parametrize("bad, says", [
    ({"volume": [[0, 0], [2, 1]]}, "volume is [[0, 0], [2, 1]]"),  # took check itself down with a TypeError
    ({"volume": -1}, "volume is -1"),  # flipped the sound's phase
    ({"volume": None}, "volume is None"),  # muted it
    ({"volume": "0.5"}, "volume is '0.5'"),
    ({"fadeIn": -1}, "fadeIn is -1"),
    ({"trim": [1, 2, 3]}, "trim must be [start, end]"),
])
def test_a_video_volume_and_fades_are_numbers_from_zero(media_dir, bad, says):
    got = _errors(scene([dict({"id": "v", "type": "video", "src": media_dir["lo"]}, **bad)]))
    assert any(says in m for _, m in got), got


@pytest.mark.parametrize("bad, says", [
    ({"volume": "x"}, "volume is 'x'"), ({"fadeIn": "1s"}, "fadeIn is '1s'"), ({"fadeIn": -1}, "fadeIn is -1"),
    ({"volume": None}, "volume is None"), ({"trim": [None, 1]}, "trim must be [start, end]"),
    ({"src": ["a.wav", "b.wav"]}, "src is ['a.wav', 'b.wav']"),
])
def test_wrong_audio_values_are_errors_before_the_mix(media_dir, bad, says):
    a = dict({"id": "a", "src": media_dir["tone"]}, **bad)
    got = _errors(dict(scene([{"type": "rect"}], duration=1), audio=[a]))
    assert any(says in m for _, m in got), got


def test_media_src_is_one_file(media_dir, tmp_path):
    got = _errors(scene([{"id": "i", "type": "image", "src": 5}]))
    assert got == [("bad-value", "src is 5; it is one file name, as a string")]
    folder = tmp_path / "pics.png"
    folder.mkdir()
    got = _errors(scene([{"id": "i", "type": "image", "src": str(folder)}]))  # was a PermissionError mid-render
    assert got[0][0] == "media-folder" and "is a folder, not a file" in got[0][1]


# ---- C23 and A20: sequences -----------------------------------------------------------------------------------------
def _seq(media_dir, **kw):
    clips = kw.pop("clips", [{"src": media_dir["lo"]}, {"src": media_dir["hi"]}])
    return scene([dict({"id": "cut", "type": "sequence", "clips": clips}, **kw)])


@pytest.mark.parametrize("kw, code, says", [
    ({"transition": {"type": "spin", "dur": 0.5}}, "bad-value", "transition type: unknown value 'spin'"),
    ({"transition": {"type": None, "dur": 0.5}}, "bad-value", "transition type is None"),
    ({"transition": {"type": 3, "dur": 0.5}}, "bad-value", "transition type is 3"),
    ({"transition": {"type": "fade", "dur": "abc"}}, "bad-value", "transition dur is 'abc'"),
    ({"transition": {"type": "fade", "dur": -0.5}}, "bad-value", "transition dur is -0.5"),
    ({"transition": "fade"}, "bad-value", "transition is 'fade'; it is an object"),
    ({"clips": []}, "missing-key", "a sequence needs 'clips'"),
    ({"clips": None}, "missing-key", "a sequence needs 'clips'"),
    ({"clips": "a.mp4"}, "bad-value", "clips must be a list"),  # iterated its characters
    ({"clips": {"src": "a.mp4"}}, "bad-value", "clips must be a list"),  # iterated its keys
    ({"clips": [{"trim": [0, 1]}]}, "missing-key", "a video layer needs 'src'"),  # a 1-frame layer, no error
    ({"clips": [5]}, "bad-value", "clips[0] is 5"),
    ({"clips": ["photo.jpg"]}, "missing-key", "clips[0] 'photo.jpg' is an image"),
])
def test_sequence_values_are_checked(media_dir, kw, code, says):
    spec = _seq(media_dir, **kw)
    errs = [i for i in check(spec)["issues"] if i["level"] == "error"]
    assert errs and errs[0]["code"] == code and says in errs[0]["message"], errs


def test_a_sequence_without_clips_key_is_an_error(media_dir):
    spec = scene([{"id": "cut", "type": "sequence"}])  # was 5 s of nothing
    assert _errors(spec)[0] == ("missing-key", "a sequence needs 'clips', a list of at least one clip")


def test_transition_types_dur_default_and_cut(media_dir):
    def starts(**kw):
        got, s = _issues(_seq(media_dir, **kw)), k.Scene(_seq(media_dir, **kw))
        out = (round(s.by_id["cut-2"].t_in, 3), s.by_id["cut-2"].raw.get("enter"), got)
        s.close()
        return out
    t, enter, got = starts(transition={"type": "dissolve", "dur": 0.4})
    assert t == 1.6 and enter["name"] == "fade" and got["renamed-value"]["level"] == "info"
    t, enter, got = starts(transition={"type": "wipe"})  # no dur: half a second, as the docs say; it was a hard cut
    assert t == 1.5 and enter["name"] == "wipe-right" and enter["dur"] == 0.5
    t, enter, got = starts(transition={"type": "cut", "dur": 0.4})  # a cut overlapped the clips by its dur
    assert t == 2.0 and enter is None and got["ignored-key"]["level"] == "warning"
    t, enter, got = starts(transition={"type": "none"})
    assert t == 2.0 and enter is None


def test_an_image_clip_needs_a_dur_and_then_holds_for_it(media_dir):
    errs = [i for i in check(_seq(media_dir, clips=[{"src": media_dir["lo"]}, {"src": media_dir["still"]}]))["issues"]
            if i["level"] == "error"]
    assert errs[0]["message"].startswith("clips[1]: an image clip needs 'dur'")  # it was a 0.04 s flash
    s = k.Scene(_seq(media_dir, clips=[{"src": media_dir["lo"]}, {"src": media_dir["still"], "dur": 1.5}],
                     transition={"type": "fade", "dur": 0.5}))
    assert (s.by_id["cut-2"].type, s.by_id["cut-2"].t_in, s.duration) == ("image", 1.5, 3.0)
    s.close()


# ---- C26: trim end ---------------------------------------------------------------------------------------------------
def test_sound_stops_at_the_trim_end_unless_the_clip_loops(media_dir, tmp_path):
    vid = {"id": "v", "type": "video", "src": media_dir["steps"], "w": 64, "h": 36, "trim": [1, 2], "dur": 3}
    pcm = _mix(scene([vid]), tmp_path)  # read on to 4 s of the file before
    assert _pitch(pcm, 0.2, 0.8) == pytest.approx(400, abs=20) and _rms(pcm, 1.2, 2.9) < 1e-3
    pcm = _mix(scene([dict(vid, loop=True)]), tmp_path, "loop.wav")
    assert _pitch(pcm, 1.2, 1.8) == pytest.approx(400, abs=20) and _rms(pcm, 2.2, 2.8) > 0.1
    spec = dict(scene([{"type": "rect"}], duration=2), audio=[{"src": media_dir["steps"], "trim": [0, 0.5], "dur": 2}])
    pcm = _mix(spec, tmp_path, "a.wav")
    assert _rms(pcm, 0.1, 0.4) > 0.1 and _rms(pcm, 0.6, 1.9) < 1e-3


# ---- C25: sounds that are never heard -------------------------------------------------------------------------------
@pytest.mark.parametrize("extra, says", [
    ({"volume": 0}, "its volume is 0"), ({"gain": -90}, "its gain is -90 dB"),
    ({"at": 10}, "starts at 10.00s, when the scene (2.00s) has ended"),
    ({"role": "sfx", "gain": -100}, "its gain is -100 dB"),
])
def test_check_warns_about_a_sound_nobody_hears(media_dir, extra, says):
    spec = dict(scene([{"type": "rect"}], duration=2), audio=[dict({"id": "a", "src": media_dir["tone"]}, **extra)])
    got = _issues(spec)
    assert got["audio-unheard"]["level"] == "warning" and says in got["audio-unheard"]["message"], got
    spec["audio"] = [{"id": "a", "src": media_dir["tone"], "gain": -20}]
    assert "audio-unheard" not in _issues(spec)


def test_check_reports_a_mix_ffmpeg_cannot_make_as_an_error(media_dir, monkeypatch):
    from kinescript import listen
    spec = dict(scene([{"type": "rect"}], duration=1), audio=[{"src": media_dir["tone"]}])

    def fails(kind):
        def report(_):
            raise kind
        monkeypatch.setattr(listen, "report", report)
        return _issues(spec)
    assert fails(RuntimeError("audio mix failed: boom"))["audio-mix"]["level"] == "error"
    assert fails(TypeError("odd"))["audio-unmeasured"]["level"] == "info"  # a TypeError here took check down


# ---- C24: caption files ---------------------------------------------------------------------------------------------
def _caps(tmp_path, data: bytes, name="c.srt"):
    (tmp_path / name).write_bytes(data)
    spec = scene([{"id": "cc", "type": "captions", "src": name}], w=1280, h=720, duration=4)
    (tmp_path / "s.json").write_text(json.dumps(spec))
    return str(tmp_path / "s.json")


def _cues(tmp_path, data: bytes, name="c.srt"):
    notes = []
    (tmp_path / name).write_bytes(data)
    return load_cues({}, str(tmp_path / name), notes), notes


def test_caption_text_reads_as_written(tmp_path):
    srt = "1\n00:00:01,000 --> 00:00:02,000\n<i>Fish</i> &amp; chips &lt;3 {\\an8}top\n"
    assert _cues(tmp_path, srt.encode())[0] == [[1.0, 2.0, "Fish & chips <3 top"]]  # entities were shown as written
    cues, notes = _cues(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nCafé\n".encode("cp1252"))
    assert cues[0][2] == "Café" and notes[0][1] == "caption-encoding" and "Windows-1252" in notes[0][2]
    cues, notes = _cues(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\n日本\n".encode("utf-16"))
    assert cues[0][2] == "日本" and "UTF-16" in notes[0][2]  # both were "'utf-8' codec can't decode byte"
    cues, _ = _cues(tmp_path, b"1\r00:00:01,000 --> 00:00:02,000\rOld Mac\r\r2\r00:00:03,000 --> 00:00:04,000\rTwo\r")
    assert [c[2] for c in cues] == ["Old Mac", "Two"]
    cues, notes = _cues(tmp_path, b"1\n00:00:01,500 -> 00:00:02,500\nSingle arrow\n")
    assert cues == [[1.5, 2.5, "Single arrow"]] and notes[0][1] == "caption-arrow"
    vtt = b"WEBVTT\n\nNOTE a comment\n\nintro\n00:00.500 --> 00:01.500 align:start\n<v Ann>Hello</v>\n"
    assert _cues(tmp_path, vtt, "c.vtt")[0] == [[0.5, 1.5, "Hello"]]


def test_overlapping_cues_show_together(tmp_path):
    srt = b"1\n00:00:00,500 --> 00:00:02,500\nA\n\n2\n00:00:01,500 --> 00:00:03,000\nB\n"
    assert _cues(tmp_path, srt)[0] == [[0.5, 1.5, "A"], [1.5, 2.5, "A\nB"], [2.5, 3.0, "B"]]  # B showed from 2.5 s
    s = k.load(_caps(tmp_path, srt))
    assert s.by_id["cc"].raw["cues"] == [[0.5, 1.5, "A"], [1.5, 2.5, "A\nB"], [2.5, 3.0, "B"]]
    s.close()


def test_a_caption_line_that_goes_on_does_not_blink_where_stacked_cues_meet():
    from kinescript.render import Renderer
    spec = scene([{"id": "cc", "type": "captions", "cues": [[0, 2, "A"], [1, 3, "B"]], "fill": "#000000",
                   "box": {"fill": "#ffffff", "pad": [4, 8]}}], w=320, h=180, bg="#000000")
    r = Renderer(k.Scene(spec), backend="cpu")

    def white(n):
        return int((r.frame(n)[..., :3].min(axis=2) > 240).sum())
    full = white(45)
    assert full > 500
    # 0.03 s before "A" ends, "B" goes on: its box faded to 28 % and back in, so no pixel was white
    assert white(59) == pytest.approx(full, rel=0.5) and white(30) == pytest.approx(full, rel=0.5)
    assert white(89) == 0  # the last cue still fades out at its end


@pytest.mark.parametrize("data, says", [
    (b"1\n00:00:xx,500 --> 00:00:01,500\nBad\n", "captions src c.srt line 2: '00:00:xx,500' is not a time"),
    (b"1\n00:00:01,500 --> \nNo end\n", "captions src c.srt line 2: the cue '00:00:01,500 -->' has no end time"),
    (b"1\n00:00:03,000 --> 00:00:01,000\nBackwards\n", "line 2: the cue ends at 1.000s, which is not after its start"),
])
def test_a_caption_file_with_a_broken_cue_names_the_file_and_line(tmp_path, data, says):
    r = check(_caps(tmp_path, data))
    assert not r["ok"] and r["issues"][0]["code"] == "bad-captions" and says in r["issues"][0]["message"], r


@pytest.mark.parametrize("data", [b"", b"just some text\nno cues\n", b"\xff\xfe1\x00\n\x00"])
def test_a_caption_file_with_no_cues_warns(tmp_path, data):
    r = check(_caps(tmp_path, data))
    assert r["ok"] and r["issues"][0]["code"] == "captions-empty", r  # nothing was drawn, with no word


@pytest.mark.parametrize("cues, says", [
    ([[1, 2]], "cues[0] is [1, 2]"), ([[2, 1, "x"]], "cues[0] ends at 1s"), ([["a", 2, "x"]], 'cues[0] is ["a", 2, "x"]'),
    ([{"at": 1, "text": "x"}], 'cues[0] is {"at": 1, "text": "x"}'),
])
def test_inline_cues_are_checked(cues, says):
    got = _errors(scene([{"id": "cc", "type": "captions", "cues": cues}]))
    assert got[0][0] == "bad-captions" and says in got[0][1], got


# ---- C9: output errors from the command line -------------------------------------------------------------------------
def test_a_render_that_cannot_write_its_file_is_an_error_not_a_traceback(tmp_path, capsys):
    import os
    import stat
    from kinescript.cli import main
    (tmp_path / "s.json").write_text(json.dumps(scene([{"type": "rect"}], duration=0.2)))
    ro = tmp_path / "ro.mp4"
    ro.write_bytes(b"old")
    os.chmod(ro, stat.S_IREAD)
    try:
        rc = main(["render", str(tmp_path / "s.json"), "-o", str(ro), "-q", "draft"])
    finally:
        os.chmod(ro, stat.S_IREAD | stat.S_IWRITE)
    res = json.loads(capsys.readouterr().out)
    if os.name == "nt":  # root on Linux CI writes through a read-only bit
        assert rc == 2 and res["ok"] is False and "ro.mp4" in res["error"], res
    (tmp_path / "hd.json").write_text(json.dumps(scene([{"type": "rect"}], w=1280, h=720, duration=0.2)))
    rc = main(["render", str(tmp_path / "hd.json"), "-o", str(tmp_path / "o.mp4"), "--scale", "100"])
    assert rc == 2 and "largest side" in json.loads(capsys.readouterr().out)["error"]
