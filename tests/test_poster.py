"""canvas.poster: the frame a rendered video opens on, so a player or feed that shows frame 0 before it plays does
not show an empty background."""
import subprocess

import numpy as np
import pytest

import kinescript as k
from kinescript import media
from kinescript.scene import SceneError, load
from kinescript.verify import check


def scene(poster=None, at=1.0, **extra):
    """2 s on black; a red card from `at` to the end."""
    canvas = {"w": 320, "h": 180, "fps": 30, "bg": "#000000", "duration": 2}
    if poster is not None:
        canvas["poster"] = poster
    return dict({"canvas": canvas, "layers": [{"id": "card", "type": "rect", "x": 160, "y": 90, "w": 320, "h": 180,
                                               "fill": "#ff0000", "at": at}]}, **extra)


def frames(path, n):
    """The first n frames of a video as (n, h, w, 3) RGB."""
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(path), "-frames:v", str(n), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(n, 180, 320, 3)


def red(px) -> bool:
    return px[..., 0].mean() > 200 and px[..., 1].mean() < 40


def black(px) -> bool:
    return px.mean() < 20


@pytest.mark.parametrize("workers", [1, 3])
def test_a_video_opens_on_the_poster_frame_and_keeps_the_rest_in_place(tmp_path, workers):
    out = tmp_path / "o.mp4"
    res = k.render(scene("card.in+0.5"), str(out), quality="draft", workers=workers)
    assert res["poster"] == {"at": 1.5, "frame": 45} and res["frames"] == 60
    f = frames(out, 31)
    assert red(f[0])  # the poster frame, though the card is not on until 1 s
    assert all(black(f[n]) for n in range(1, 30)) and red(f[30])  # every other frame where it was


def test_no_poster_and_a_scene_without_one_keep_frame_0(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    res = k.render(scene("card.in+0.5"), str(a), quality="draft", poster=False)
    assert "poster" not in res and black(frames(a, 1)[0])
    res = k.render(scene(), str(b), quality="draft")
    assert "poster" not in res and black(frames(b, 1)[0])


def test_a_render_from_after_0s_and_a_png_sequence_keep_their_first_frame(tmp_path):
    out = tmp_path / "part.mp4"
    res = k.render(scene("card.in+0.5"), str(out), quality="draft", start=0.5, end=1.5)
    assert "poster" not in res and black(frames(out, 1)[0])
    k.render(scene("card.in+0.5"), str(tmp_path / "f") + "/", end=0.1)
    pngs = sorted((tmp_path / "f").iterdir())
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(pngs[0]), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    assert black(np.frombuffer(raw, np.uint8))


def test_poster_takes_seconds_clocks_markers_and_cues():
    for poster, frame in ((1.5, 45), ("45f", 45), ("0:01.5", 45), ("drop", 36), ("card.in+0.5", 45), ("card.out", 59),
                          (0, 0)):
        sc = load(scene(poster, markers={"drop": 1.2}))
        assert sc.poster_frame == frame, poster
        sc.close()


@pytest.mark.parametrize("poster, words", [(2.5, "outside the 2s scene"), ("nope.in", "unknown reference 'nope'"),
                                           (-1, "0 or more seconds"), (True, "it is a time")])
def test_a_bad_poster_is_an_error_that_says_why(poster, words):
    with pytest.raises(SceneError) as e:
        load(scene(poster))
    assert words in str(e.value)


def test_a_top_level_poster_or_thumbnail_is_read_as_canvas_poster():
    spec = scene()
    spec["poster"] = 1.5
    sc = load(spec)
    assert sc.poster_frame == 45 and any(i["code"] == "renamed-key" for i in sc.issues)
    sc.close()
    spec = scene()
    spec["canvas"]["thumbnail"] = "card.in"
    sc = load(spec)
    assert sc.poster_frame == 30
    sc.close()


def test_still_at_poster_draws_the_poster_frame(tmp_path):
    res = k.still(scene("card.in+0.5"), "poster", str(tmp_path / "p.png"))
    assert res["frame"] == 45 and "notes" not in res
    with pytest.raises(ValueError, match="canvas.poster"):
        k.still(scene(), "poster", str(tmp_path / "q.png"))


def codes(res):
    return {i["code"] for i in res["issues"]}


def test_check_warns_of_a_blank_first_frame_until_a_poster_is_set():
    assert "blank-first-frame" in codes(check(scene()))
    assert not {"blank-first-frame", "blank-poster"} & codes(check(scene("card.in+0.5")))
    assert not {"blank-first-frame", "blank-poster"} & codes(check(scene(at=0)))  # on screen from 0 s
    faded = scene(at=0)
    faded["layers"][0]["enter"] = {"name": "fade", "dur": 0.5}
    assert "blank-first-frame" in codes(check(faded))


def test_check_warns_of_a_poster_that_shows_nothing():
    res = check(scene(0.5))
    blank = [i for i in res["issues"] if i["code"] == "blank-poster"]
    assert blank and blank[0]["level"] == "warning" and blank[0]["from"] == 0.5
    assert "blank-first-frame" not in codes(res)
