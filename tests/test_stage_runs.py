"""Runs of a stage's models merged on OpenGL (gl3d's _JOIN and _WASH, stage.Stage._merge_run): which children join a
run, and that the frame is the one the stage makes when it merges each model on its own. The tests need the GPU
backend and moderngl, and skip without them (CI runs on the CPU). With them, a driver that does not take the
programs that merge a run fails the tests that expect one: every OpenGL 3.3 driver should take them."""
import numpy as np
import pytest

import kinescript as k
from kinescript import gl3d, stage
from kinescript.render import Renderer
from test_gl3d import ball, flock, glass, lit, lit_scene, renderer, ring, scene

TURN = [[0, 0], [2, 300]]  # a turn over the scene's two seconds
PAPER = np.array([0xf4, 0xef, 0xe6], float)


def staged(kids, **canvas):
    return scene([{"id": "st", "type": "group", "sort": "pixel", "children": kids}], **canvas)


def trio(mid=None, **more):
    """A ring, a ball that crosses it and a second ball in front of both, with `mid` between the first two."""
    return [ring(r=62, tube=20, rotateY=TURN), *([mid] if mid else []), ball(r=34, z=5, **more),
            ball(id="near", x=205, y=100, r=24, z=25, fill="#3aa76d", rotateX=TURN)]


def drawn(spec, monkeypatch, frames=(0,), joined=True, depth=False):
    """Frames of spec with its models on OpenGL, what the stage merged for the last of them in order ("run" for a
    run of models, or the id of a child merged on its own), and the renderer. joined=False: with every model merged
    on its own, as a stage merged them before it had runs."""
    seen = []
    one, run = stage.Stage._merge, stage.Stage._merge_run
    with monkeypatch.context() as mp:
        if not joined:
            mp.setattr(gl3d, "joining", lambda: False)
        mp.setattr(stage.Stage, "_merge", lambda self, child, box: (seen.append(child.id), one(self, child, box))[1])
        mp.setattr(stage.Stage, "_merge_run", lambda self, r: (seen.append("run"), run(self, r))[1])
        r = renderer(spec, mp)
        r.depth_view = depth
        out = []
        for n in frames:
            seen.clear()
            out.append(r.frame(n).copy())
    return np.stack(out), list(seen), r


def both(spec, monkeypatch, frames=(0, 9, 31, 52), **kw):
    """The frames with runs, checked against those without, and what the stage merged."""
    a, seen, r = drawn(spec, monkeypatch, frames, **kw)
    b, alone, r0 = drawn(spec, monkeypatch, frames, joined=False, **kw)
    assert "run" not in alone
    assert np.array_equal(a, b), (int((a != b).any(-1).sum()), int(np.abs(a.astype(int) - b).max()))
    if kw.get("depth"):
        assert len(r.depths) == len(r0.depths) > 0 and all(np.array_equal(x, y) for x, y in zip(r.depths, r0.depths))
    return a, seen, alone


def test_models_in_a_row_are_merged_as_one_child(monkeypatch):
    """Three models that cross and cover one another, turning: one run, and the frames of three merges."""
    a, seen, alone = both(staged(trio()), monkeypatch, depth=True)
    assert seen == ["run"] and alone == ["ring", "ball", "near"]
    assert (np.abs(a[0, ..., :3] - PAPER).max(-1) > 30).mean() > 0.1  # and they are in the frame


MIDDLES = {
    "a flat layer": ({"id": "mid", "type": "rect", "x": 160, "y": 120, "w": 60, "h": 150, "fill": "#33323a",
                      "rotateY": 60}, False),
    "a see-through model": (glass(id="mid", r=34, tube=13, x=125), False),
    "a model with a ground shadow": (ball(id="mid", r=20, x=105, z=-30, ground=True), False),
    "a model that fades": (ball(id="mid", r=20, x=105, z=40, opacity=0.5), False),
    "a blurred model": (ball(id="mid", r=20, x=105, z=40, blur=3), False),
    "a washed and blurred model": (ball(id="mid", r=20, x=105, z=40, wash=0.5, blur=3), False),
    "a washed model that fades": (ball(id="mid", r=20, x=105, z=40, wash=0.5, opacity=0.5), False),
    "a washed model with a ground shadow": (ball(id="mid", r=20, x=105, z=-30, wash=0.5, ground=True), False),
    "a washed see-through model": (glass(id="mid", r=34, tube=13, x=125, wash=0.5), False),
    "a washed model": (ball(id="mid", r=20, x=105, z=40, wash=0.5), True),
    "a model of copies": (flock(id="mid"), True),
    "a model that draws nothing": (ball(id="mid", r=20, x=105, z=40, draw=0), True),
    "a model out of the frame": (ball(id="mid", r=20, x=900), True),
}


@pytest.mark.parametrize("name", MIDDLES)
def test_a_run_ends_at_a_child_that_cannot_join_it(name, monkeypatch):
    """Between two models: a child the stage has to merge on its own ends the run before it, and the models after
    it start another; one that can join, or that draws nothing, leaves one run."""
    mid, joins = MIDDLES[name]
    a, seen, alone = both(staged(trio(mid)), monkeypatch)
    assert seen == (["run"] if joins else ["run", "mid", "run"]), name
    if mid.get("draw") != 0 and mid["x"] < 320:  # and it is drawn
        without = drawn(staged(trio()), monkeypatch, (0, 9, 31, 52))[0]
        assert (np.abs(a.astype(int) - without).max(-1) > 20).mean() > 0.002, name


@pytest.mark.parametrize("order", [("far", "close"), ("close", "far")])
def test_what_shows_through_a_child_in_front_of_a_run_is_the_first_model_that_drew_there(order, monkeypatch):
    """A see-through sheet in front, merged before two balls that overlap: through it shows the ball merged first,
    the nearer or not, which is not the one that shows beside the sheet."""
    balls = {"far": ball(id="far", r=55, x=140, z=-40, fill="#d0392b", edges=None),
             "close": ball(id="close", r=55, x=185, z=30, fill="#2b6cd0", edges=None)}
    sheet = {"id": "sheet", "type": "rect", "x": 160, "y": 90, "w": 200, "h": 70, "z": 120, "fill": "#ffffff",
             "opacity": 0.5}
    a, seen, alone = both(staged([sheet] + [balls[k] for k in order]), monkeypatch, frames=(0,))
    assert seen == ["sheet", "run"]
    px = a[0, ..., :3].astype(int)
    under, beside = px[95, 168], px[140, 168]  # where the balls overlap: behind the sheet, and below it
    assert beside[2] > beside[0] + 40  # the near ball, blue
    first_is_red = order[0] == "far"
    assert (under[0] > under[2] + 15) == first_is_red and (under[2] > under[0] + 15) != first_is_red, (under, order)


def inside(mask):
    """mask without its rim: the pixels whose eight neighbours are in it too."""
    m = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            m &= np.roll(mask, (dy, dx), (0, 1))
    return m


@pytest.mark.parametrize("joined", [True, False])
@pytest.mark.parametrize("wash", [0.25, 0.6, 1.0])
def test_a_washed_model_of_a_stage_is_mixed_toward_the_paper(wash, joined, monkeypatch):
    """In a run or merged on its own, each of its pixels is the unwashed one mixed toward the canvas background."""
    kids = [ring(r=62, tube=20), ball(r=50, z=30, wash=wash)]
    plain = drawn(staged([ring(r=62, tube=20), ball(r=50, z=30)]), monkeypatch)[0][0, ..., :3].astype(float)
    a, seen, _ = drawn(staged(kids), monkeypatch, joined=joined)
    assert seen == (["run"] if joined else ["ring", "ball"])
    lone = drawn(staged([ring(r=62, tube=20)]), monkeypatch)[0][0, ..., :3].astype(float)
    mask = inside(np.abs(plain - lone).max(-1) > 0)  # where the ball draws: the ring is not washed
    assert mask.mean() > 0.05
    want = plain * (1 - wash) + PAPER * wash
    off = np.abs(want - a[0, ..., :3])[mask]
    # A frame pixel is four of the stage's, each rounded after it is mixed.
    assert off.max() <= 1.5 and off.mean() < 0.5, (float(off.max()), float(off.mean()))
    assert np.abs(a[0, ..., :3] - plain)[mask].mean() > 5  # and it is washed


@pytest.mark.parametrize("mode", ["cpu", "skia"])
def test_a_washed_model_of_a_stage_is_mixed_toward_the_paper_where_skia_draws_it(mode, monkeypatch):
    """On the CPU, and on the GPU with the models kept on Skia: the layer that washes it is opened for it."""
    def frame(spec):
        if mode == "cpu":
            return Renderer(k.Scene(spec), backend="cpu").frame(0)[..., :3].astype(float)
        return renderer(spec, monkeypatch, "skia").frame(0)[..., :3].astype(float)
    plain, lone = frame(staged([ring(r=62, tube=20), ball(r=50, z=30)])), frame(staged([ring(r=62, tube=20)]))
    a = frame(staged([ring(r=62, tube=20), ball(r=50, z=30, wash=0.6)]))
    mask = inside(np.abs(plain - lone).max(-1) > 0)
    assert mask.mean() > 0.05
    off = np.abs(plain * 0.4 + PAPER * 0.6 - a)[mask]
    assert off.max() <= 1.5 and off.mean() < 0.5, (float(off.max()), float(off.mean()))
    assert np.abs(a - plain)[mask].mean() > 5


def test_a_wash_that_changes_and_a_model_washed_among_others_draw_as_merged_one_by_one(monkeypatch):
    kids = [ring(r=62, tube=20, wash=[[0, 0], [1, 0.9], [2, 0.1]]), ball(r=34, z=5),
            ball(id="near", x=205, r=24, z=25, wash=[[0, 1], [2, 0]])]
    a, seen, alone = both(staged(kids), monkeypatch, frames=(0, 14, 30, 45, 59))
    assert seen == ["run"] and alone == ["ring", "ball", "near"]


def test_a_lit_stage_and_two_stages_merge_their_runs_as_one_by_one(monkeypatch):
    """The models still cast: the shadow map is drawn from what each run's models left."""
    kids = [ring(x=-30, y=-50, r=30, tube=10, rotateY=TURN), ball(x=30, y=-40, r=22)]
    a, seen, alone = both(lit_scene([lit(kids)]), monkeypatch)
    assert seen == ["run", "floor160"]
    dark = drawn(lit_scene([lit(kids, shadows=None)]), monkeypatch, (0,))[0]
    assert (np.abs(a[0].astype(int) - dark[0]).max(-1) > 20).mean() > 0.002  # a shadow falls
    left = [dict(kid, id=kid["id"] + "l") for kid in kids]
    a, seen, alone = both(lit_scene([lit(left, x=80, w=150), lit(kids, x=240, w=150)]), monkeypatch)
    assert seen == ["run", "floor80", "run", "floor240"]  # each stage merges a run of its own


def test_a_frame_with_runs_is_the_same_whatever_was_drawn_before_it(monkeypatch):
    """A run starts from nothing: not from the run of the frame before, nor of a stage of another size."""
    moving = dict(ball(id="mid", r=20, z=40, wash=0.4), x=[[0, 40], [2, 280]])
    spec = staged(trio(moving))
    want = drawn(spec, monkeypatch, (40,))[0][0]
    a, _, r = drawn(spec, monkeypatch, (0, 59, 12, 40))
    assert np.array_equal(a[-1], want)
    other = renderer(staged([ring(), ball()], w=480, h=360), monkeypatch)
    other.frame(3)
    assert np.array_equal(r.frame(40), want)


def test_a_run_whose_stage_fails_holds_no_targets(monkeypatch):
    spec = staged(trio())
    want = drawn(spec, monkeypatch)[0][0]
    r = renderer(spec, monkeypatch)
    with monkeypatch.context() as mp:
        mp.setattr(stage.Stage, "_merge_run", lambda self, r: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            r.frame(0)
    assert "held" not in gl3d._STATE
    assert np.array_equal(r.frame(0), want)


def test_the_targets_of_a_run_are_kept_while_other_sizes_are_asked_for(monkeypatch):
    r = renderer(staged(trio()), monkeypatch)
    r.frame(0)
    size = next(reversed(gl3d._TARGETS))
    gl3d._STATE["held"] = size
    try:
        for other in ((64, 48), (80, 48), (96, 48)):
            gl3d._targets(*other)
        assert size in gl3d._TARGETS and len(gl3d._TARGETS) == 3
    finally:
        gl3d.let_go()
    gl3d._targets(112, 48)
    r.gr.resetContext()  # Skia has not seen the targets made here
    assert size not in gl3d._TARGETS and len(gl3d._TARGETS) == 2
