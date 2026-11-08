import json

import numpy as np
import pytest

from kinescript import avatar as A
from kinescript import avatar_images as I

W, H = 320, 480
C = ((W - 1) / 2, (H - 1) / 2)
GREEN = (0.0, 1.0, 0.0)
SKIN, SHIRT, DARK, WHITE = (0.85, 0.65, 0.5), (0.2, 0.3, 0.7), (0.1, 0.05, 0.05), (1.0, 1.0, 1.0)


def _ellipse(xx, yy, cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def draw(mouth=4.0, eyes=9.0, scale=1.0, shift=(0.0, 0.0), bg=GREEN, hair=False, teeth=False, gaze=0.0, brows=False):
    """A figure on a flat background: shirt, face, two eyes and a mouth whose height is `mouth`. The figure is
    scaled by `scale` about the centre and moved by `shift`, as an image model moves a variant."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fx = C[0] + (xx - C[0] - shift[0]) / scale  # the figure's own point shown at each pixel
    fy = C[1] + (yy - C[1] - shift[1]) / scale
    img = np.zeros((H, W, 4), np.float32)
    img[..., :3], img[..., 3] = bg, 1
    paint = [(_ellipse(fx, fy, 160, 470, 130, 120), SHIRT), (_ellipse(fx, fy, 160, 220, 95, 115), SKIN),
             (_ellipse(fx, fy, 125 + gaze, 200, 12, eyes), DARK), (_ellipse(fx, fy, 195 + gaze, 200, 12, eyes), DARK),
             (_ellipse(fx, fy, 160, 285, 30, mouth), DARK)]
    if teeth:
        paint.append((_ellipse(fx, fy, 160, 282, 22, 3), WHITE))
    if brows:
        paint += [((abs(fy - 172) < 4) & (abs(fx - 125) < 16), DARK), ((abs(fy - 172) < 4) & (abs(fx - 195) < 16), DARK)]
    if hair:
        paint.append(((fy > 100) & (fy < 150) & (abs(fx - 160) < 80), (0.3, 0.2, 0.1)))
    for mask, colour in paint:
        img[mask, :3] = colour
    return img


def source(folder, **variants):
    """A source folder: base, blink and mouths A, C, D, each a little moved; keyword arguments replace a file."""
    folder.mkdir(parents=True, exist_ok=True)
    pics = {"base": draw(), "blink": draw(eyes=1.5, shift=(2, -1)),
            "mouth-A": draw(mouth=2.0, shift=(-2, 1), scale=1.01), "mouth-C": draw(mouth=12, shift=(3, 2)),
            "mouth-D": draw(mouth=20, scale=0.99), **variants}
    for name, px in pics.items():
        I.write_image(px, folder / f"{name}.png")
    return folder


def test_keying_removes_the_background_but_keeps_the_same_colour_inside_the_figure():
    img = draw(bg=WHITE, teeth=True)
    out, issues = I.key(img)
    assert issues == []
    assert out[5, 5, 3] == 0 and out[240, 160, 3] == 1  # the corner is clear, the face solid
    assert out[282, 160, 3] == 1 and np.allclose(out[282, 160, :3], WHITE)  # white teeth on white stay


def test_a_picture_with_its_own_alpha_keeps_it():
    img = draw()
    img[:10, :, 3] = 0
    out, _ = I.key(img)
    assert np.array_equal(out, img)


def test_a_busy_border_cannot_be_keyed():
    img = draw()
    img[:, :40, :3] = np.random.default_rng(1).random((H, 40, 3))
    _, issues = I.key(img)
    assert [i["code"] for i in issues] == ["avatar-source-background"] and issues[0]["level"] == "error"


def test_a_variant_is_lined_up_with_the_base():
    s, shift, lined = I.align(draw(), draw(mouth=12, scale=1.02, shift=(4, -3)))
    assert s == pytest.approx(1.02, abs=0.006)
    assert shift[0] == pytest.approx(4 / 1.02, abs=1.5) and shift[1] == pytest.approx(-3 / 1.02, abs=1.5)
    _, d = I.region(draw(), lined)
    assert float(np.mean(d > 0.1)) < 0.02  # only the mouth differs once lined up


def test_regions_cover_both_eyes_and_the_mouth():
    base = draw()
    eyes, _ = I.region(base, draw(eyes=1.5))
    mouth, _ = I.region(base, draw(mouth=20))
    assert eyes[0] <= 113 and eyes[2] >= 207 and eyes[1] <= 195 and eyes[3] >= 205
    assert mouth[0] <= 135 and mouth[2] >= 185 and mouth[1] <= 270 and mouth[3] >= 300
    assert mouth[1] > eyes[3]


def test_images_become_an_avatar_that_talks_and_blinks(tmp_path):
    got = I.from_images(source(tmp_path / "src"))
    assert got["ok"], got["issues"]
    out = tmp_path / "src" / "avatar"
    assert A.validate(out) == [] or all(i["level"] != "error" for i in A.validate(out))
    av = A.load(out)
    assert sorted(av.mouth.files) == ["A", "C", "D", "X"] and [p.name for p in av.eyes] == ["eyes"]
    assert av.size == (W, H) and set(av.spec["gestures"]) == {"nod", "shake", "tilt", "lean"}
    for role in ("blink", "mouth-A", "mouth-C", "mouth-D"):
        assert got["files"][role]["drift"] < I.DRIFT_WARN, role
    x, y, w, h = got["boxes"]["mouth"]
    assert x < 160 < x + w and y < 285 < y + h
    assert (out / "avatar_sheet.png").is_file() and got["sheet"]
    body = I.read_image(out / "parts" / "body.png")
    assert body[5, 5, 3] == 0 and body[240, 160, 3] == 1


def test_a_mouth_patch_shows_its_shape_where_the_mouth_is(tmp_path):
    from kinescript.output import still
    I.from_images(source(tmp_path / "src"))
    av = A.load(tmp_path / "src" / "avatar")
    for shape, dark in (("X", False), ("D", True)):
        scene = {"canvas": {"w": W, "h": H, "bg": "#ffffff"}, "layers": [A.layers(av, "a", {"mouth": shape})]}
        still(scene, 0, str(tmp_path / f"{shape}.png"))
        px = I.read_image(tmp_path / f"{shape}.png")
        assert (px[297, 160, :3].max() < 0.3) == dark, shape  # below the closed mouth's line, inside the open one


def test_a_variant_that_changed_more_than_its_mouth_is_named(tmp_path):
    got = I.from_images(source(tmp_path / "src", **{"mouth-C": draw(mouth=12, hair=True)}))
    drifted = [i for i in got["issues"] if i["code"] == "avatar-source-drift"]
    assert len(drifted) == 1 and "mouth-C.png" in drifted[0]["message"] and drifted[0]["level"] == "error"
    assert not got["ok"] and (tmp_path / "src" / "avatar" / "avatar.json").is_file()  # written, for the sheet


def test_missing_pictures_are_named(tmp_path):
    src = source(tmp_path / "src")
    (src / "mouth-D.png").unlink()
    (src / "blink.png").unlink()
    got = I.from_images(src)
    assert not got["ok"]
    assert {i["code"] for i in got["issues"]} == {"avatar-missing-shape", "avatar-source"}
    assert "mouth-D.png" in json.dumps(got["issues"])


def test_source_json_sets_the_name_voice_and_boxes(tmp_path):
    src = source(tmp_path / "src")
    (src / "source.json").write_text(json.dumps({"name": "Pat", "voice": {"backend": "kokoro", "speaker": "am_adam"},
                                                 "mouthBox": [120, 260, 80, 50]}))
    got = I.from_images(src, tmp_path / "pat")
    assert got["ok"] and got["boxes"]["mouth"] == [120, 260, 80, 50]
    spec = A.read(tmp_path / "pat")
    assert spec["name"] == "Pat" and spec["voice"]["speaker"] == "am_adam"


def test_init_writes_the_prompts_and_keeps_existing_files(tmp_path):
    got = I.init(tmp_path / "new")
    assert len(got["wrote"]) == 2
    text = (tmp_path / "new" / "PROMPTS.md").read_text()
    assert "mouth-D.png" in text and "#00FF00" in text
    assert I.init(tmp_path / "new")["kept"] and not I.init(tmp_path / "new")["wrote"]


def test_kine_avatar_from_images_prints_the_report(tmp_path, capsys):
    from kinescript.cli import main
    src = source(tmp_path / "src")
    assert main(["avatar", "from-images", str(src), "--out", str(tmp_path / "a")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["out"] == str(tmp_path / "a") and "mouth-C" in out["files"]
    assert main(["avatar", "from-images", str(tmp_path / "fresh"), "--init"]) == 0
    assert (tmp_path / "fresh" / "source.json").is_file()


def test_noise_and_a_small_tone_change_in_every_variant_are_not_drift(tmp_path):
    rng = np.random.default_rng(7)

    def rough(px, tone):
        px = px.copy()
        px[..., :3] = np.clip(px[..., :3] + tone + rng.normal(0, 0.02, px[..., :3].shape), 0, 1)
        return px

    got = I.from_images(source(tmp_path / "src", blink=rough(draw(eyes=1.5, shift=(2, -1)), 0.03),
                               **{"mouth-A": rough(draw(mouth=2.0, shift=(-2, 1)), -0.02),
                                  "mouth-C": rough(draw(mouth=12, shift=(3, 2)), 0.03),
                                  "mouth-D": rough(draw(mouth=20, scale=0.99), 0.02)}))
    assert got["ok"], got["issues"]
    assert not [i for i in got["issues"] if i["code"] == "avatar-source-drift"]
    x, y, w, h = got["boxes"]["mouth"]
    assert x < 160 < x + w and y < 285 < y + h and h < 120


def test_look_and_expression_pictures_become_eye_drawings_a_scene_picks(tmp_path):
    from kinescript.output import still
    src = source(tmp_path / "src", **{"look-left": draw(gaze=-10, shift=(1, 1)), "expr-happy": draw(brows=True)})
    got = I.from_images(src)
    assert got["ok"], got["issues"]
    av = A.load(tmp_path / "src" / "avatar")
    eyes = av.parts["eyes"]
    assert set(eyes.files) == {"open", "closed", "left", "happy"}
    assert av.spec["expressions"] == {"happy": {"eyes": {"state": "happy"}}}
    names = [n for n, _ in A.poses(av)]
    assert "look left" in names and "expression happy" in names
    scene = {"canvas": {"w": W, "h": H, "fps": 10, "duration": 3, "bg": "#ffffff"},
             "layers": [{"id": "a", "use": "avatar", "avatar": str(tmp_path / "src" / "avatar"), "x": W / 2,
                         "y": H / 2, "h": H, "idle": False,
                         "expressions": [{"name": "happy", "at": 0.5, "dur": 0.5}],
                         "gaze": [{"at": 1.5, "target": "left"}, {"at": 2.5, "target": "camera"}]}]}

    def dark(t, x, y):
        still(scene, t, str(tmp_path / "f.png"))
        return I.read_image(tmp_path / "f.png")[y, x, :3].max() < 0.3

    assert [dark(t, 125, 172) for t in (0.2, 0.75, 1.2)] == [False, True, False]  # the brows come and go
    assert [dark(t, 112, 200) for t in (1.2, 1.8, 2.8)] == [False, True, False]  # the eyes look left, then back


def test_an_expression_name_that_is_an_eye_drawing_is_refused(tmp_path):
    got = I.from_images(source(tmp_path / "src", **{"expr-left": draw(brows=True)}))
    assert not got["ok"] and "expr-left.png" in json.dumps(got["issues"])


def test_an_avatar_that_swaps_drawings_renders_the_same_frames_in_any_order(tmp_path):
    import kinescript as k
    from kinescript.render import Renderer
    src = source(tmp_path / "src", **{"look-left": draw(gaze=-10), "expr-happy": draw(brows=True)})
    I.from_images(src)
    spec = {"canvas": {"w": 160, "h": 240, "fps": 10, "duration": 3},
            "layers": [{"id": "a", "use": "avatar", "avatar": str(src / "avatar"), "h": 240,
                        "expressions": [{"name": "happy", "at": 0.4, "dur": 0.6}],
                        "gaze": [{"at": 1.2, "target": "left"}, {"at": 2.0, "target": "camera"}],
                        "gestures": [{"name": "nod", "at": 0.5}]}]}
    r = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(30)]
    shuffled = Renderer(k.Scene(spec, tmp_path), backend="cpu")
    for n in (25, 6, 14, 3, 19):
        assert np.array_equal(shuffled.frame(n), sequential[n])
