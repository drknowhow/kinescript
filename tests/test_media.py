"""Footage decoding and probing: which source frame each output frame shows, alpha, rotation, start offsets,
files cut short, and what `kine probe` says about media, models and scenes. The clips are made here with ffmpeg."""
import json
import math
import subprocess
import time

import numpy as np
import pytest

import kinescript as k
from kinescript import media, output
from kinescript.render import Renderer
from kinescript.verify import check, probe, probe_media

# Frame n of an index clip shows n in binary as eight bars (bit 0 on the left), so a frame reads back exactly.
INDEX = "format=gray,geq=lum='if(bitand(N\\,pow(2\\,floor(X/16)))\\,255\\,0)'"


def _ff(*args):
    subprocess.run([media.FFMPEG, "-y", "-v", "error", *map(str, args)], check=True)


def _index_clip(path, seconds=2, *out_args):
    _ff("-f", "lavfi", "-i", f"color=c=black:s=128x16:r=30:d={seconds},{INDEX}", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-qp", "0", *out_args, path)
    return str(path)


@pytest.fixture(scope="session")
def idx(tmp_path_factory):
    """Two seconds, 60 frames at 30 fps."""
    return _index_clip(tmp_path_factory.mktemp("idx") / "idx.mp4")


def _shown(frame) -> int:
    """The index an index clip stretched over the whole frame shows."""
    h, w = frame.shape[:2]
    return sum(1 << b for b in range(8) if frame[h // 2, int((b + 0.5) * w / 8), 0] > 127)


def _spec(src, duration=None, fps=30, **layer):
    vid = dict({"id": "v", "type": "video", "src": src, "x": 0, "y": 0, "anchor": "topleft", "w": 128, "h": 16,
                "fit": "stretch"}, **layer)
    canvas = {"w": 128, "h": 16, "fps": fps, "bg": "#000000"}
    if duration:
        canvas["duration"] = duration
    return {"canvas": canvas, "layers": [vid]}


def _played(spec) -> list[int]:
    """The index each frame shows, rendered in order."""
    s = k.Scene(spec)
    r = Renderer(s, backend="cpu")
    out = [_shown(r.frame(n)) for n in range(s.frames)]
    s.close()
    return out


def _cold(spec, n) -> int:
    """The index frame n shows, rendered on its own (a worker starting there, a still)."""
    s = k.Scene(spec)
    got = _shown(Renderer(s, backend="cpu").frame(n))
    s.close()
    return got


def test_a_clip_played_to_its_end_shows_its_last_frame(idx):
    # The source time was held 1.5 frames short of the end, so frame 59 showed frame 58 again.
    spec = _spec(idx)
    assert _played(spec) == list(range(60))
    assert _cold(spec, 59) == 59
    assert _played(_spec(idx, speed=2)) == list(range(0, 60, 2))


def test_a_dur_longer_than_the_trim_holds_the_last_frame_of_the_trim(idx, tmp_path):
    # trim [0.5, 1] with dur 1 played on past the trim end to 1.5 s of the file, picture and sound.
    spec = _spec(idx, trim=[0.5, 1.0], dur=1.0)
    assert _played(spec) == list(range(15, 30)) + [29] * 15
    assert _cold(spec, 22) == 29
    at60 = _spec(idx, fps=60, trim=[0.5, 1.0], dur=1.0)  # the clip's own last frame, not a time between two
    assert _played(at60)[30:] == [29] * 30 and _cold(at60, 45) == 29
    sound = str(tmp_path / "sound.mp4")
    _ff("-f", "lavfi", "-i", "testsrc2=s=64x36:r=30:d=2", "-f", "lavfi", "-i", "sine=d=2", "-pix_fmt", "yuv420p",
        "-shortest", sound)
    s = k.Scene(_spec(sound, trim=[0.5, 1.0], dur=1.0))
    assert output._audio_inputs(s)[0]["play"] == pytest.approx(0.5)  # was 1.0: the sound played on past the trim
    s.close()
    issues = [i for i in check(spec)["issues"] if i["code"] == "clip-hold"]
    assert len(issues) == 1 and issues[0]["level"] == "warning" and issues[0]["from"] == 0.5, issues
    assert "trimmed clip" in issues[0]["message"] and "loop" in issues[0]["fix"]
    looped = _spec(idx, trim=[0.5, 1.0], dur=1.0, loop=True)
    assert _played(looped) == list(range(15, 30)) * 2
    assert not [i for i in check(looped)["issues"] if i["code"] == "clip-hold"]


def test_a_resampled_clip_holds_one_frame_however_the_render_reaches_it(idx, tmp_path):
    # At 24 fps a 30 fps clip's hold read on from an earlier frame landed on frame 28 or 29 depending on where the
    # render started, so the held picture could change where one render chunk ends and the next begins.
    spec = _spec(idx, fps=24, trim=[0.5, 1.0], dur=1.5)
    for start in range(0, 14):
        s = k.Scene(spec)
        r = Renderer(s, backend="cpu")
        got = [_shown(r.frame(n)) for n in range(start, 36)]
        s.close()
        assert got[max(0, 12 - start):] == [29] * (36 - max(12, start)), (start, got)
    # Half speed at 60 fps, the last frame's time is 0.9917 s, within half a source frame of the trim end: a clip
    # opened there shows the nearest frame, 30, past the end, so the time asked for stops short of that.
    assert _cold(_spec(idx, fps=60, speed=0.5, trim=[0, 1.0]), 119) == 29
    # Picture 2 s, sound 3 s: opened past the picture at 24 fps, the clip held frame 58, a frame short of its end.
    longer = str(tmp_path / "longer.mp4")
    _ff("-f", "lavfi", "-i", f"color=c=black:s=128x16:r=30:d=2,{INDEX}", "-f", "lavfi", "-i", "sine=d=3",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", 0, longer)
    assert [_cold(_spec(longer, fps=24), n) for n in (60, 70)] == [59, 59]


@pytest.mark.parametrize("fps, layer", [(24, {}), (25, {}), (60, {}), (30, {"speed": 0.5}), (30, {"speed": 1.5}),
                                        (30, {"speed": 2}), (24, {"trim": [0.43, 1.9]})])
def test_a_resampled_clip_shows_the_same_frame_however_the_render_reaches_it(idx, fps, layer):
    # Each frame shows the source frame nearest its source time, the earlier of two equally near. Opened mid-clip,
    # the decoder's frame grid started at the first frame it decoded and rounded to the nearest output frame: at
    # 24 fps frame 6 (0.25 s, halfway between source frames 7 and 8) showed 8 played from 0 and 7 on its own.
    spec = _spec(idx, fps=fps, **layer)
    trim0, end = layer.get("trim", [0, 2.0])
    speed = layer.get("speed", 1)
    played = _played(spec)
    assert played == [min(math.ceil((trim0 + n * speed / fps) * 30 - 0.5 - 1e-9), round(end * 30) - 1)
                      for n in range(len(played))]
    for n in range(1, len(played), 3):
        assert _cold(spec, n) == played[n], n


def test_a_numpy_frame_number_shows_the_same_frame(idx):
    # bench/stages.py draws frame numbers from np.linspace. The source time came out a numpy float, written into
    # ffmpeg's filter as np.float64(...): the pipe gave no frame, so a clip opened mid-way showed its last frame,
    # found by decoding the file to its end every frame (the edit stage drew in 390 ms instead of 8).
    s = k.Scene(_spec(idx))
    r = Renderer(s, backend="cpu")
    got = [_shown(r.frame(n)) for n in np.array([17, 23, 40])]
    s.close()
    assert got == [17, 23, 40]


def test_reading_on_to_a_frame_makes_an_image_of_that_frame_only(idx, monkeypatch):
    # A render that skips frames (bench/stages.py, contact sheets) reads past them; each was made an image, on the GPU
    # an upload of its planes, and dropped: 2.4 ms of the edit stage's 8.4 at the median.
    made, reads = [], []
    image, read = media.VideoSource._image, media.VideoSource._read
    monkeypatch.setattr(media.VideoSource, "_image", lambda self: made.append(self.frame is None) or image(self))
    monkeypatch.setattr(media.VideoSource, "_read", lambda self: reads.append(1) or read(self))
    v = media.VideoSource(idx, 128, 16, 30)
    assert _shown(v.at(0.0).toarray()) == 0 and _shown(v.at(40 / 30).toarray()) == 40
    v.close()
    assert len(reads) == 41 and made == [True, True]


def _cuts(src, trims, duration=None):
    canvas = {"w": 128, "h": 16, "fps": 30, "bg": "#000000"}
    if duration:
        canvas["duration"] = duration
    return {"canvas": canvas, "layers": [{"id": "cut", "type": "sequence",
                                          "clips": [{"src": src, "trim": tr} for tr in trims]}]}


def test_a_cut_list_opens_each_clip_before_its_first_frame(idx, monkeypatch):
    # ffmpeg takes ~0.12 s to a clip's first frame, and a render waited that long at every cut (sequence-60: 6.5 s of
    # 11). Drawing forward, a clip's decoder now opens up to render.AHEAD before it comes on, where it will first read;
    # the frames are the ones a decoder opened at the cut, or cold, shows.
    spec = _cuts(idx, [[1.0, 1.5], [0.2, 0.7], [1.4, 1.9]])
    opened = []
    real = media.VideoSource._open
    monkeypatch.setattr(media.VideoSource, "_open", lambda self, t: opened.append(round(t, 3)) or real(self, t))
    s = k.Scene(spec)
    r = Renderer(s, backend="cpu")
    second = s.by_id["cut-2"]
    shown, procs = [], []
    for n in range(s.frames):
        shown.append(_shown(r.frame(n)))
        if n == 14:  # the frame before cut-2 comes on: its decoder is open and has read nothing
            assert second.video is None and second.ahead is not None
            ahead = second.ahead[1]
            assert ahead.proc is not None and ahead.index == -1 and ahead.t0 == pytest.approx(0.2)
        if n == 15:
            assert second.video is ahead and second.ahead is None  # the one opened ahead, not a second one
        for L in s.all_layers():
            if L.video is not None and L.video.proc is not None:
                procs.extend(L.video.proc)
    assert shown == list(range(30, 45)) + list(range(6, 21)) + list(range(42, 57))
    assert opened == [1.0, 0.2, 1.4]  # one open a clip, each where the clip starts
    assert [L.id for L in s.all_layers() if L.video is not None and L.video.proc is not None] == ["cut-3"]
    assert all(L.ahead is None for L in s.all_layers())
    s.close()
    for n in (15, 16, 31):
        assert _cold(spec, n) == shown[n], n
    # Closing ends the ffmpeg processes, not only the pipes: their reader threads kill and reap them.
    deadline = time.monotonic() + 5
    while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert procs and all(p.poll() is not None for p in procs)


def test_the_clips_a_frame_shows_open_before_any_is_waited_on(idx, monkeypatch):
    # Nine clips on video-grid's first frame each opened and waited for its first frame in turn: 1 s.
    layers = [{"id": f"v{i}", "type": "video", "src": idx, "x": 64 * i, "y": 0, "anchor": "topleft", "w": 64, "h": 16,
               "trim": [0.5 * i, 2]} for i in range(2)]
    s = k.Scene({"canvas": {"w": 128, "h": 16, "fps": 30}, "layers": layers})
    live = []
    real = media.VideoSource._next

    def counted(self):
        decoders = [d for L in s.all_layers() for d in (L.video, L.ahead and L.ahead[1]) if d]
        live.append(sum(d.proc is not None for d in decoders))
        return real(self)

    monkeypatch.setattr(media.VideoSource, "_next", counted)
    Renderer(s, backend="cpu").frame(9)
    s.close()
    assert live[0] == 2


def test_a_clip_opened_ahead_past_its_picture_holds_its_last_frame(tmp_path):
    # Picture 2 s, sound 3 s: a clip trimmed to start at 2.5 s has no frame there. Opened ahead, its decoder gives no
    # frame, and the clip shows the file's last one, as a decoder opened at the cut does.
    longer = str(tmp_path / "longer.mp4")
    _ff("-f", "lavfi", "-i", f"color=c=black:s=128x16:r=30:d=2,{INDEX}", "-f", "lavfi", "-i", "sine=d=3",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", 0, longer)
    spec = _cuts(longer, [[0.5, 1.0], [2.5, 3.0]])
    assert _played(spec) == list(range(15, 30)) + [59] * 15
    assert _cold(spec, 20) == 59


def test_render_workers_split_a_resampled_clip_without_a_seam(idx, tmp_path):
    # The second worker opens the clip at frame 19 and reads on from there. Its decoder's frame grid began at the
    # first frame it decoded, so at frame 21 it showed source frame 40 where a single worker shows 39.
    spec = _spec(idx, fps=24, trim=[0.43, 2.0])

    def shown(workers):
        out = str(tmp_path / f"w{workers}.mp4")
        assert output.render(spec, out, quality="high", workers=workers)["workers"] == workers
        raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", out, "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                             capture_output=True, check=True).stdout
        return [_shown(f) for f in np.frombuffer(raw, np.uint8).reshape(-1, 16, 128, 4)]

    one = shown(1)
    assert one == _played(spec) and len(one) == 38
    assert shown(2) == one


def test_a_dur_longer_than_the_file_holds_its_last_frame_and_check_says_so(idx):
    spec = _spec(idx, dur=3)
    assert _played(spec)[55:] == [55, 56, 57, 58] + [59] * 31
    assert _cold(spec, 80) == 59
    issues = [i for i in check(spec)["issues"] if i["code"] == "clip-hold"]
    assert len(issues) == 1 and issues[0]["from"] == 2.0 and issues[0]["to"] == 3.0, issues
    assert "past the end of" in issues[0]["message"]
    assert not [i for i in check(_spec(idx))["issues"] if i["code"] == "clip-hold"]


@pytest.mark.parametrize("ext", [".mkv", ".mp4"])
def test_footage_that_starts_late_lasts_as_long_as_it_plays(tmp_path, ext):
    # An mkv whose first frame is at 5 s reported its end (6 s) as its length: 5 s of held frame, then black.
    src = _index_clip(tmp_path / f"late{ext}", 1, "-output_ts_offset", 5)
    assert media.probe(src)["duration"] == pytest.approx(1.0, abs=0.05)
    s = k.Scene(_spec(src))
    assert s.by_id["v"].t_out == pytest.approx(1.0) and s.duration == pytest.approx(1.0)
    s.close()
    assert _played(_spec(src)) == list(range(30))


def test_rotated_footage_is_probed_upright_and_decoded_at_full_size(tmp_path):
    flat = _index_clip(tmp_path / "flat.mp4", 0.2)
    phone = str(tmp_path / "phone.mp4")
    _ff("-display_rotation", 90, "-i", flat, "-c", "copy", phone)  # stored 128x16, shown 16x128
    info = media.probe(phone)
    assert (info["w"], info["h"]) == (16, 128) and info["rotation"] in (90, 270), info
    assert media.yuv_matrix(info) == media.yuv_matrix(media.probe(flat))  # chosen on the stored size
    spec = {"canvas": {"w": 16, "h": 128, "fps": 30}, "layers": [{"id": "v", "type": "video", "src": phone,
                                                                 "x": 0, "y": 0, "anchor": "topleft"}]}
    s = k.Scene(spec)
    Renderer(s, backend="cpu").frame(0)
    # Sized from the stored 128x16, a 16x128 box was decoded at 16 x 2 px and scaled up: soft.
    assert (s.by_id["v"].video.w, s.by_id["v"].video.h) == (16, 128)
    s.close()


def test_a_transparent_webm_keeps_its_alpha_as_footage(tmp_path):
    if "libvpx-vp9" not in media.capabilities()["encoders"] or "libvpx-vp9" not in media.decoders():
        pytest.skip("this ffmpeg has no libvpx-vp9")
    box = {"type": "rect", "x": 32, "y": 0, "anchor": "topleft", "w": 32, "h": 36, "fill": "#0000ff", "opacity": 0.3}
    clip = str(tmp_path / "over.webm")
    output.render({"canvas": {"w": 64, "h": 36, "fps": 30, "bg": "transparent", "duration": 0.2}, "layers": [box]},
                  clip, quality="draft", workers=1)
    spec = {"canvas": {"w": 64, "h": 36, "fps": 30, "bg": "#336699", "duration": 0.2},
            "layers": [{"type": "video", "src": clip, "x": 0, "y": 0, "anchor": "topleft", "w": 64, "h": 36,
                        "fit": "stretch"}]}
    f = Renderer(k.Scene(spec), backend="cpu").frame(3)
    assert np.abs(f[18, 8, :3].astype(int) - (51, 102, 153)).max() <= 2  # was (0, 0, 0): the alpha was dropped
    assert np.abs(f[18, 48, :3].astype(int) - (36, 71, 184)).max() <= 3  # 30% blue over the background
    info = media.probe(clip)
    assert info["alpha"] and info["codec"] == "vp9"
    # The GPU's YUV path has no alpha plane, so such a clip never takes it.
    assert media.VideoSource(clip, 64, 36, 30, ctx="a gpu").ctx is None


def test_check_says_when_this_ffmpeg_cannot_decode_a_webms_alpha(tmp_path, monkeypatch):
    clip = str(tmp_path / "a.webm")
    _ff("-f", "lavfi", "-i", "color=c=red@0.5:s=64x36:r=30:d=0.2,format=yuva420p", "-c:v", "libvpx-vp9",
        "-deadline", "realtime", clip)
    if not media.probe(clip)["alpha"]:
        pytest.skip("this ffmpeg wrote no alpha")
    monkeypatch.setattr(media, "decoders", lambda: set())
    spec = {"canvas": {"w": 64, "h": 36, "fps": 30, "duration": 0.2}, "layers": [{"type": "video", "src": clip}]}
    lost = [i for i in check(spec)["issues"] if i["code"] == "media-alpha-lost"]
    assert len(lost) == 1 and "a.webm" in lost[0]["message"] and "libvpx" in lost[0]["fix"], lost


def test_a_file_cut_short_is_reported_and_holds_the_same_frame_however_it_is_reached(idx, tmp_path):
    whole = str(tmp_path / "whole.mp4")
    _ff("-f", "lavfi", "-i", "testsrc2=s=128x16:r=30:d=2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", 10,
        "-movflags", "+faststart", whole)
    data = open(whole, "rb").read()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[:len(data) // 2])
    assert media.probe(str(cut))["duration"] == pytest.approx(2.0)  # the index still promises 2 s
    end = media.decoded_end(str(cut))
    assert 0.1 < end < 1.8 and media.decoded_end(whole) == pytest.approx(2.0, abs=0.04)
    spec = _spec(str(cut))
    s = k.Scene(spec)
    r = Renderer(s, backend="cpu")
    seq = [r.frame(n).copy() for n in range(60)]
    s.close()
    held = int(end * 30) + 1  # from here on the last picture holds, whichever way the frame is reached
    assert all(np.array_equal(f, seq[-1]) for f in seq[held:]) and not np.array_equal(seq[held - 3], seq[-1])
    for n in (59, held + 2):
        s = k.Scene(spec)
        assert np.array_equal(Renderer(s, backend="cpu").frame(n), seq[-1]), n
        s.close()
    issues = [i for i in check(spec)["issues"] if i["code"] == "media-truncated"]
    assert len(issues) == 1 and issues[0]["level"] == "warning" and issues[0]["from"] < 1.8, issues
    assert "cut.mp4" in issues[0]["message"] and "again" in issues[0]["fix"]
    res = output.render(spec, str(tmp_path / "o.mp4"), quality="draft", workers=1)
    assert [w["code"] for w in res.get("warnings", [])] == ["media-truncated"]
    assert "warnings" not in output.render(_spec(idx), str(tmp_path / "p.mp4"), quality="draft", workers=1)


def test_a_ts_cut_with_copy_and_no_key_frame_has_no_picture(tmp_path):
    src = tmp_path / "src.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=64x36:r=30:d=4", "-f", "lavfi", "-i", "sine=d=4", "-c:v", "libx264",
        "-g", 300, "-pix_fmt", "yuv420p", "-shortest", src)
    cut = str(tmp_path / "cut.ts")
    _ff("-i", src, "-ss", 1.4, "-t", 2, "-c", "copy", cut)  # no key frame inside: the video stream stays empty
    spec = {"canvas": {"w": 64, "h": 36, "fps": 30, "duration": 1}, "layers": [{"id": "v", "type": "video", "src": cut}]}
    none = [i for i in check(spec)["issues"] if i["code"] == "no-picture"]  # it said the file was sound only
    assert len(none) == 1 and none[0]["level"] == "error" and none[0]["layer"] == "v", none
    assert "cut.ts" in none[0]["message"] and "-c copy" in none[0]["fix"], none
    info = probe_media(cut)  # was w 0, h 0, fps 0 and nothing else
    assert info["video"] is False and info["kind"] == "audio" and "-c copy" in info["note"], info


def test_probe_names_stills_models_and_scenes(tmp_path, capsys):
    from kinescript.cli import main
    png = str(tmp_path / "still.png")
    _ff("-f", "lavfi", "-i", "color=c=red:s=32x16", "-frames:v", 1, png)
    info = probe_media(png)  # ffprobe gives a still 25 fps, so it was reported as image: false
    assert info["image"] and info["fps"] == 0 and info["kind"] == "image" and not info["video"], info
    obj = tmp_path / "tri.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert main(["probe", str(obj)]) == 0  # was ffprobe's "Invalid data found"
    got = json.loads(capsys.readouterr().out)
    assert got["ok"] and got["kind"] == "model" and got["triangles"] >= 1 and got["vertices"] >= 3, got
    assert probe({"canvas": {"w": 64, "h": 36}, "layers": [{"type": "rect"}]})["ok"] is True
