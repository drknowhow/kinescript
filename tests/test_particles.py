"""The particles layer: an emitter whose particles take their birth, start and path from their index and the seed, in
closed form, drawn from one sprite in one drawAtlas call."""
import math

import numpy as np
import pytest
import skia

import kinescript as k
from kinescript import particles as PT
from kinescript.props import Ctx
from kinescript.render import Renderer, _GL
from kinescript import verify
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def emitter(**raw):
    return PT.Emitter(raw, Ctx(1920, 1080, 30), "p")


def codes(spec):
    try:
        s = k.Scene(spec)
        return [i["code"] for i in s.issues], s
    except k.SceneError as e:
        return [i["code"] for i in e.issues], None


def _pixels(spec, T=0.0, backend="cpu", track=False):
    try:
        r = Renderer(k.Scene(spec), backend=backend, track=track)
    except RuntimeError:
        pytest.skip("no GPU")
    r.draw(T)
    return r.snapshot().toarray()[..., :3].astype(int), r


def test_a_particle_flies_by_gravity_and_drag_in_closed_form():
    st = emitter(burst=1, life=10, speed=200, angle=0, gravity=100).state(1.0, 10)
    assert st["x"][0] == pytest.approx(200) and st["y"][0] == pytest.approx(50)  # v t and g t^2 / 2
    dragged = emitter(burst=1, life=10, speed=[200, 200], angle=-30, gravity=[40, 100], drag=2)
    x, y, vx, vy, dt = 0.0, 0.0, 200 * math.cos(math.radians(-30)), 200 * math.sin(math.radians(-30)), 1e-5
    for _ in range(int(round(1.5 / dt))):  # what it integrates step by step
        vx, vy = vx + (40 - 2 * vx) * dt, vy + (100 - 2 * vy) * dt
        x, y = x + vx * dt, y + vy * dt
    st = dragged.state(1.5, 10)
    assert st["x"][0] == pytest.approx(x, abs=0.01) and st["y"][0] == pytest.approx(y, abs=0.01)


def test_a_rate_keeps_rate_times_life_alive_and_a_keyframed_rate_emits_its_integral():
    st = emitter(rate=100, life=1).state(5.0, 10)
    assert len(st["x"]) == 100
    assert np.all(np.diff(st["birth"]) > 0) and 4.0 <= st["birth"][0] and st["birth"][-1] <= 5.0  # oldest first
    tapered = emitter(rate=[[0, 100], [1, 0]], life=100).state(2.0, 3)  # an eased ramp from 100 to 0 over 1 s
    assert 49 <= len(tapered["x"]) <= 51 and tapered["birth"].max() <= 1.0
    assert emitter(rate=100, life=1).state(0.0, 10) is None  # nothing is born before the layer starts
    warm = emitter(rate=50, life=2, prewarm=True).state(0.0, 10)  # as if it had run for a life already
    assert len(warm["x"]) == 100 and warm["birth"].min() < -1.9


def test_bursts_leave_at_their_times_and_the_emitter_stops_with_the_layer():
    e = emitter(burst=[[0, 10], [1, 5]], life=100)
    assert len(e.state(0.5, 4)["x"]) == 10 and len(e.state(1.5, 4)["x"]) == 15
    assert list(e.state(1.5, 4)["birth"]) == [0.0] * 10 + [1.0] * 5
    assert len(emitter(rate=10, life=100).state(9.0, 4)["x"]) == 40  # born over the layer's 4 s only


def test_a_particle_keeps_its_draws_whatever_frame_is_asked_first():
    raw = dict(rate=200, life=[0.5, 2], speed=[50, 400], angle=[0, 360], size=[2, 9], spin=[-90, 90], flip=[1, 3],
               turbulence=20, fill=["#ff0000", "#00ff00"], sizeOverLife=[[0, 1], [1, 0]], seed=7)
    a, b = emitter(**raw), emitter(**raw)
    a.state(0.3, 5)
    late, fresh = a.state(1.7, 5), b.state(1.7, 5)
    for name in ("x", "y", "size", "rot", "argb", "cell", "birth"):
        assert np.array_equal(late[name], fresh[name]), name
    other = emitter(**dict(raw, seed=8)).state(1.7, 5)
    assert not np.array_equal(other["x"][:50], late["x"][:50])
    # Without a seed, each layer draws its own from its id.
    ids = [PT.Emitter(dict(rate=50), Ctx(320, 180, 30), lid).state(1.0, 2)["x"] for lid in ("a", "b")]
    assert not np.array_equal(ids[0], ids[1])


def test_colours_go_in_turn_or_over_life_and_opacity_over_life():
    st = emitter(burst=6, life=10, fill=["#ff0000", "#00ff00", "#0000ff"]).state(1.0, 10)
    assert [int(v) & 0xFFFFFF for v in st["argb"]] == [0xFF0000, 0x00FF00, 0x0000FF] * 2
    e = emitter(burst=1, life=2, fillOverLife=[[0, "#ffffff"], [1, "#ff0000"]], opacityOverLife=[[0, 1], [1, 0]])
    start, end = int(e.state(0.0, 4)["argb"][0]), int(e.state(1.99, 4)["argb"][0])
    assert start == 0xFFFFFFFF and (end >> 16) & 255 == 255 and (end >> 8) & 255 <= 16 and end & 255 <= 16
    assert end >> 24 <= 3
    assert abs((int(e.state(1.0, 4)["argb"][0]) >> 24) - 128) <= 1


def test_turning_spinning_flipping_and_aligning():
    spun = emitter(burst=3, life=10, spin=90)
    assert np.allclose(spun.state(2.0, 10)["rot"] - spun.state(1.0, 10)["rot"], math.pi / 2)
    still = emitter(burst=3, life=10).state(1.0, 10)
    assert np.all(still["rot"] == 0) and still["cell"] is None
    aligned = emitter(burst=1, life=10, speed=100, angle=30, align=True).state(1.0, 10)
    assert aligned["rot"][0] == pytest.approx(math.radians(30))
    falling = emitter(burst=1, life=10, speed=100, angle=0, gravity=1000, align=True).state(1.0, 10)
    assert falling["rot"][0] == pytest.approx(math.atan2(1000, 100))  # along its path, pulled down
    flipping = emitter(burst=50, life=10, flip=2)
    cells = np.concatenate([flipping.state(t, 10)["cell"] for t in (0.1, 0.2, 0.3)])
    assert cells.min() >= 0 and cells.max() <= PT.FLIPS - 1 and len(set(cells.tolist())) > 3


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_each_shape_draws_its_sprite_sized_and_tinted(backend):
    def dot(shape, at=(160, 90)):
        layer = {"id": "p", "type": "particles", "burst": 1, "speed": 0, "life": 10, "size": 60, "shape": shape,
                 "x": 160, "y": 90, "fill": "#ffffff"}
        img, _ = _pixels(scene([layer]), 0.1, backend)
        return img
    circle = dot("circle")
    assert circle[90, 160].min() > 240 and circle[90, 186].min() > 240 and circle[118, 188].max() == 0
    assert dot("square")[116, 186].min() > 240
    rect = dot("rect")
    assert rect[102, 160].min() > 240 and rect[110, 160].max() == 0  # twice as wide as it is tall
    ring = dot("ring")
    assert ring[90, 160].max() == 0 and ring[90, 187].min() > 200
    soft = dot("soft")
    assert soft[90, 160].min() > 200 and soft[90, 186].max() < 60
    streak = dot("streak")
    assert streak[90, 185].min() > 150 and streak[90, 135].max() < 40 and streak[100, 160].max() == 0
    for shape in ("star", "triangle"):
        assert dot(shape)[90, 160].min() > 240, shape
    tinted, _ = _pixels(scene([{"id": "p", "type": "particles", "burst": 1, "speed": 0, "life": 10, "size": 40,
                                "x": 160, "y": 90, "fill": "#ff633f"}]), 0.1, backend)
    assert abs(tinted[90, 160] - [255, 99, 63]).max() <= 1


def test_a_character_and_an_image_are_particles_too(tmp_path):
    img, _ = _pixels(scene([{"id": "p", "type": "particles", "burst": 1, "speed": 0, "life": 10, "size": 80,
                             "text": "H", "x": 160, "y": 90, "fill": "#00ff00"}]), 0.1)
    lit = np.argwhere(img[..., 1] > 200)
    assert len(lit) > 200 and img[..., 0].max() == 0
    assert 120 <= lit[:, 1].min() and lit[:, 1].max() <= 200 and 50 <= lit[:, 0].min() and lit[:, 0].max() <= 130
    surf = skia.Surface(8, 4)
    surf.getCanvas().clear(skia.Color(255, 0, 0))
    surf.makeImageSnapshot().save(str(tmp_path / "chip.png"))
    spec = scene([{"id": "p", "type": "particles", "burst": 1, "speed": 0, "life": 10, "size": 80,
                   "src": "chip.png", "x": 160, "y": 90}])
    s = k.Scene(spec, base_dir=tmp_path)
    r = Renderer(s, backend="cpu")
    r.draw(0.1)
    chip = r.snapshot().toarray()[..., :3].astype(int)
    assert tuple(chip[90, 160]) == (255, 0, 0) and tuple(chip[90, 195]) == (255, 0, 0)  # 80 wide, 40 tall, untinted
    assert chip[115, 160].max() == 0


def test_particles_stay_where_they_were_born_while_the_emitter_moves():
    layer = {"id": "p", "type": "particles", "burst": [[0, 1], [0.99, 1]], "speed": 0, "life": 10, "size": 20,
             "x": [[0, 60], [1, 260, "linear"]], "y": 90, "fill": "#ffffff"}
    img, _ = _pixels(scene([layer]), 1.0)
    assert img[90, 60].min() > 240 and img[90, 258].min() > 240 and img[90, 160].max() == 0
    turned, _ = _pixels(scene([dict(layer, rotate=90, scale=2)]), 1.0)  # in its parent's space, whatever turns it
    assert turned[90, 60].min() > 240 and turned[90, 258].min() > 240 and turned[90, 160].max() == 0


def test_the_box_holds_every_particle_and_opacity_fades_them_together():
    layer = {"id": "p", "type": "particles", "burst": 40, "speed": [40, 90], "life": 10, "size": 6, "x": 160,
             "y": 90, "fill": "#ffffff"}
    img, r = _pixels(scene([layer]), 1.0, track=True)
    x0, y0, x1, y1 = r.boxes["p"]["rect"]
    lit = np.argwhere(img.max(axis=2) > 0)
    assert x0 <= lit[:, 1].min() and lit[:, 1].max() <= x1 and y0 <= lit[:, 0].min() and lit[:, 0].max() <= y1
    assert 40 <= x1 - x0 <= 200 and r.boxes["p"]["type"] == "particles"
    one = {"id": "p", "type": "particles", "burst": 1, "speed": 0, "life": 10, "size": 40, "x": 160, "y": 90,
           "fill": "#ffffff"}
    half, _ = _pixels(scene([dict(one, opacity=0.5)]), 0.1)
    assert abs(half[90, 160, 0] - 128) <= 1


def test_frames_are_pure_with_particles():
    spec = scene([{"id": "s", "type": "particles", "preset": "sparks", "x": "=60+t*100", "y": 60},
                  {"id": "c", "type": "particles", "preset": "confetti", "y": 150, "turbulence": 10},
                  {"id": "d", "type": "particles", "preset": "dust", "flip": [1, 2]}])
    r = Renderer(k.Scene(spec), backend="cpu")
    ahead = r.frame(40).copy()
    r.frame(12)
    assert np.array_equal(r.frame(40), ahead)
    assert np.array_equal(Renderer(k.Scene(spec), backend="cpu").frame(40), ahead)


def test_presets_fill_in_what_the_layer_leaves_out():
    for name in PT.PRESETS:
        got, s = codes(scene([{"id": "p", "type": "particles", "preset": name}]))
        assert s is not None, (name, got)
        e = s.by_id["p"].particles
        assert e.state(1.5, 2) is not None, name
    _, s = codes(scene([{"id": "p", "type": "particles", "preset": "snow", "size": 30, "y": 100}]))
    raw = s.by_id["p"].raw
    assert raw["size"] == 30 and raw["y"] == 100 and raw["emitter"] == "line" and raw["shape"] == "soft"
    assert s.by_id["p"].particles.w == pytest.approx(320 * 1.2)  # "120%" of the canvas


def test_schema_reads_particle_keys_and_catches_bad_ones():
    got, s = codes(scene([{"id": "p", "type": "confetti", "lifetime": [1, 2], "colors": ["#fff", "#f00"],
                           "emitter": {"type": "box", "width": 200, "h": 50}, "shape": "★", "angle": [-100, -80]}]))
    assert s is not None and "renamed-key" in got and "renamed-value" in got
    raw = s.by_id["p"].raw
    assert (raw["preset"], raw["life"], raw["fill"], raw["emitter"], raw["w"], raw["h"], raw["text"]) == (
        "confetti", [1, 2], ["#fff", "#f00"], "rect", 200, 50, "★")
    assert raw["angle"] == [-100, -80] and "rotate" not in raw  # a particle's angle is its direction
    _, s = codes(scene([{"id": "p", "type": "emitter", "rate": 10, "shape": "spark.png"}]))
    assert s is None  # the file is missing, but it was read as src
    got, _ = codes(scene([{"id": "p", "type": "particles", "rate": 10, "shape": "spark.png"}]))
    assert "media-missing" in got
    for bad, code in (({"life": "long"}, "bad-value"), ({"size": [1, 2, 3]}, "bad-value"), ({"life": 0}, "bad-value"),
                      ({"burst": [[0, -1]]}, "bad-value"), ({"gravity": "down"}, "bad-value"),
                      ({"emitter": "cone"}, "bad-value"), ({"shape": "hexagon"}, "bad-value"),
                      ({"sizeOverLife": [[2, 1]]}, "bad-value"), ({"fill": {"stops": []}}, "bad-value"),
                      ({"fillOverLife": [[0, 3]]}, "bad-value"), ({"w": [[0, 10], [1, 20]]}, "bad-value"),
                      ({"rate": 10000, "life": 30}, "too-many-particles"), ({"preset": "fireworks"}, "bad-value"),
                      ({"lifes": 2}, "unknown-key")):
        got, _ = codes(scene([dict({"id": "p", "type": "particles", "rate": 10}, **bad)]))
        assert code in got, (bad, got)
    got, _ = codes(scene([{"id": "p", "type": "particles", "life": 2}]))
    assert "missing-key" in got  # neither rate, burst nor a preset: nothing is emitted


def test_check_warns_when_particles_would_be_slow_on_the_cpu(monkeypatch):
    heavy = scene([{"id": "p", "type": "particles", "rate": 2000, "life": 1, "size": 20, "speed": [0, 200]}],
                  w=1920, h=1080, duration=1)
    light = scene([{"id": "p", "type": "particles", "burst": 20, "size": 6}], w=1920, h=1080, duration=1)
    monkeypatch.setitem(_GL, "ctx", None)  # as on a machine with no GPU
    monkeypatch.setattr(verify, "SLOW_PARTICLES", 0.02)  # 2,000 of 20 px take about 80 ms, 20 of 6 px next to none
    warned = [i for i in check(heavy)["issues"] if i["code"] == "particle-cost"]
    assert warned and "53 ms" in warned[0]["message"], warned
    assert not any(i["code"] == "particle-cost" for i in check(light)["issues"])
