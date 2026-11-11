"""use: "tour": stops made into stop groups timed one after another, voice lines in audio, cards, pinned notes, and
camera keys that fly to each focus and frame it; frames stay pure; and the checks."""
import wave

import numpy as np
import pytest

import kinescript as k
from kinescript import space, voice
from kinescript.render import Renderer
from kinescript.verify import check

BEAD = [60, -40, 70]


def scene(layers, **top):
    return dict({"canvas": {"w": 640, "h": 360, "fps": 30, "duration": "auto", "bg": "#ffffff"}, "layers": layers},
                **top)


def cell():
    return {"id": "cell", "type": "model", "x": 320, "y": 180, "light": "flat", "rotateY": "=t * 20",
            "parts": [{"shape": "sphere", "r": 50, "fill": "#dddddd"},
                      {"shape": "sphere", "r": 8, "fill": "#ff0000", "x": BEAD[0], "y": BEAD[1], "z": BEAD[2]}]}


def tour(**kw):
    return dict({"use": "tour", "id": "t", "move": 1, "hold": 1.5, "cards": False, "drift": 0, "stops": [
        {"focus": "cell"},
        {"focus": {"layer": "cell", "at": BEAD}, "zoom": 2.5, "title": "Bead", "note": "Bead"},
        {"focus": "cell", "orbit": 30}]}, **kw)


def centroid(img, rgb):
    ys, xs = np.nonzero(np.abs(img[..., :3].astype(int) - rgb).max(-1) < 40)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def test_stops_follow_one_another_and_the_camera_lands_on_each_focus():
    s = k.Scene(scene([cell(), tour()]))
    groups = [s.by_id[f"t-s{i}"] for i in (1, 2, 3)]
    assert [(round(g.t_in, 3), round(g.t_out, 3)) for g in groups] == [(0, 1.5), (1.5, 4.0), (4.0, 6.5)]
    assert s.duration == pytest.approx(6.5, abs=0.05)
    r = Renderer(s, backend="cpu")
    # Held at stop 2, the bead is in the middle of the frame and 2.5 times as large as at zoom 1.
    img = r.frame(round(3.5 * 30)).copy()
    assert np.abs(centroid(img, (255, 0, 0)) - (320, 180)).max() < 3
    # Stop 1 and 3 frame the cell's middle.
    for t in (0.5, 6.0):
        grey = centroid(r.frame(round(t * 30)).copy(), (221, 221, 221))
        assert np.abs(grey - (320, 180)).max() < 6, t
    # The note rides the bead, and frames are a pure function of their number.
    assert "t-s2-note" in s.by_id
    for n in (20, 60, 110, 180):
        assert np.array_equal(Renderer(k.Scene(scene([cell(), tour()])), backend="cpu").frame(n),
                              Renderer(s, backend="cpu").frame(n))


def silence(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0\0" * int(24000 * seconds))


def test_a_spoken_stop_lasts_as_long_as_its_line(tmp_path):
    spec = scene([cell(), tour(stops=[{"focus": "cell", "say": "One two three four five six seven eight nine ten."},
                                      {"focus": "cell", "say": "Short."}])])
    # kine voice speaks the tour's lines, under the ids the stops cue them by.
    assert [sid for sid, _ in voice.voices(voice._spec(spec, tmp_path)[0])] == ["t-s1-v", "t-s2-v"]
    # Not spoken yet, check says so and nothing else.
    assert {i["code"] for i in check(spec)["issues"] if i["level"] == "error"} <= {"voice-missing"}
    silence(tmp_path / "voice" / "t-s1-v.wav", 2.0)
    silence(tmp_path / "voice" / "t-s2-v.wav", 0.8)
    s = k.Scene(spec, base_dir=tmp_path)
    a, b = s.by_id["t-s1"], s.by_id["t-s2"]
    assert a.t_out - a.t_in == pytest.approx(2.0 + 0.4, abs=1 / 30)  # its line, then the pause
    assert b.t_in == pytest.approx(a.t_out, abs=1 / 30)
    assert b.t_out - b.t_in == pytest.approx(1.0 + 0.8 + 0.4, abs=1 / 30)  # the move, the line, the pause
    v2 = next(x for x in s.audio if x.id == "t-s2-v")
    assert v2.t_in == pytest.approx(b.t_in + 1.0, abs=1 / 30)  # it speaks once the camera lands


def test_tour_is_checked():
    def codes(**kw):
        return [(i["level"], i["code"]) for i in check(scene([cell(), tour(**kw)]))["issues"]]
    assert not [c for c in codes() if c[0] == "error"]
    for bad in ({"stops": []}, {"stops": [{"focus": "nucleus"}]}, {"stops": [{"say": "hi"}]},
                {"stops": [{"focus": "cell", "zoom": 0}]}, {"stops": [{"focus": "cell", "spin": 3}]},
                {"move": -1}, {"speed": 2}):
        assert any(lv == "error" for lv, _ in codes(**bad)), bad
    nested = scene([cell(), {"id": "g", "type": "group", "children": [tour()]}])
    assert any(i["level"] == "error" for i in check(nested)["issues"])


def test_dim_fades_the_models_a_stop_does_not_focus():
    def ball(lid, x, fill):
        return {"id": lid, "type": "model", "x": x, "y": 180, "light": "flat", "shape": "sphere", "r": 40, "fill": fill}
    spec = scene([ball("a", 220, "#ff0000"), ball("b", 420, "#0000ff"),
                  {"use": "tour", "id": "t", "move": 1, "hold": 1.5, "cards": False, "drift": 0, "dim": 0.8,
                   "stops": [{"focus": "a"}, {"focus": "b"}]}])
    s = k.Scene(spec)
    a, b = s.by_id["a"], s.by_id["b"]
    op = lambda L, T: L.get("opacity", T - L.t_in, T)  # noqa: E731
    assert (op(a, 0.5), op(b, 0.5)) == pytest.approx((1.0, 0.2))  # stop 1 focuses a
    assert (op(a, 3.0), op(b, 3.0)) == pytest.approx((0.2, 1.0))  # and stop 2, b, once the camera is there
    assert 0.2 < op(b, 2.0) < 1.0  # on the way
    kept = k.Scene(dict(spec, layers=spec["layers"][:2] + [dict(spec["layers"][2], stops=[
        {"focus": "a", "keep": ["b"]}, {"focus": "b"}])]))
    assert kept.by_id["b"].get("opacity", 0.5, 0.5) == pytest.approx(1.0)  # kept bright with a
    assert {i["code"] for i in check(dict(spec, layers=spec["layers"][:2] + [dict(spec["layers"][2], stops=[
        {"focus": "a", "keep": ["bee"]}])]))["issues"] if i["level"] == "error"} == {"tour-focus"}
    r = Renderer(s, backend="cpu")
    blue = lambda n: int((np.abs(r.frame(n)[..., :3].astype(int) - (0, 0, 255)).max(-1) < 40).sum())  # noqa: E731
    assert blue(15) == 0 and blue(90) > 1000
    assert np.array_equal(Renderer(k.Scene(spec), backend="cpu").frame(60), r.frame(60))
    bad = [(i["level"], i["code"]) for i in check(dict(spec, layers=spec["layers"][:2] + [
        dict(spec["layers"][2], dim=2)]))["issues"]]
    assert any(lv == "error" for lv, _ in bad)
