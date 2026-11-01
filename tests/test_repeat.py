"""A model's "repeat": copies of its mesh, each placed and coloured by a number, a list or an expression of i, merged
into one mesh that sorts as one model."""
import numpy as np
import pytest

import kinescript as k
from kinescript import mesh, props, render3d
from kinescript.render import Renderer
from kinescript.verify import check
from test_animation import Gltf, box


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def _model(repeat, **layer):
    return scene([dict({"id": "m", "type": "model", "shape": "cube", "w": 30, "h": 30, "depth": 2, "x": 160, "y": 90,
                        "light": "flat", "repeat": repeat}, **layer)])


def _shot(spec, T=0.0, backend="cpu"):
    try:
        r = Renderer(k.Scene(spec), backend=backend, track=True)
    except RuntimeError:
        pytest.skip("no GPU")
    r.draw(T)
    return r


def _pixels(r):
    return r.snapshot().toarray()[..., :3].astype(int)


def test_copies_are_placed_by_numbers_lists_and_expressions_of_i():
    c = props.Copies({"count": 4, "x": "=i * 10", "y": [1, 2], "z": 5, "scale": "=n"}, props.Ctx(320, 180, 30))
    cols, fills = c.at(0.0, 2.0)
    assert cols == {"x": [0, 10, 20, 30], "y": [1, 2, 1, 2], "z": [5] * 4, "scale": [4] * 4} and fills is None
    assert not c.moves()
    assert c.at(1.0, 2.0)[0]["x"] is cols["x"]  # worked out once
    moving = props.Copies({"count": 2, "x": "=i + t"}, props.Ctx(320, 180, 30))
    assert moving.moves() and moving.at(0.5, 2.0)[0]["x"] == [0.5, 1.5]
    colours = props.Copies({"count": 3, "fill": ["red", "#00ff00"]}, props.Ctx(320, 180, 30)).at(0, 1)[1]
    assert colours[0] == colours[2] != colours[1] and colours[1] == (0.0, 1.0, 0.0, 1.0)


def test_colour_helpers_make_colours_expressions_can_return():
    assert props.hsl(0, 1, 0.5) == "#ff0000" and props.hsl(240, 1, 0.5) == "#0000ff" and props.hsl(360, 1, 0.5) == \
        "#ff0000"
    assert props.mix("#000000", "#ffffff", 0) == "#000000" and props.mix("red", "blue", 1) == "#0a84ff"
    grey = props.mix("#000000", "#ffffff", 0.5)
    assert grey[1:3] == grey[3:5] == grey[5:7] and 0x30 < int(grey[1:3], 16) < 0xd0
    s = k.Scene(scene([{"id": "r", "type": "rect", "w": 10, "h": 10, "fill": "=mix('#000000', '#ff0000', p)"}]))
    half = s.by_id["r"].get("fill", 1.0)  # a layer's fill can use them too
    assert 0.2 < half[0] < 0.8 and half[1] < 0.1


def test_copies_merge_into_one_mesh_where_each_is_put():
    r = _shot(_model({"count": 3, "x": "=(i - 1) * 80"}))
    m, base = r._models[("copies", "m")][2], r._models[("copies", "m")][0]
    assert len(m.P) == 3 * len(base.P) and len(m.T) == 3 * len(base.T)
    for i, dx in enumerate((-80, 0, 80)):
        part = m.P[i * len(base.P):(i + 1) * len(base.P)]
        assert np.allclose(part, base.P + (dx, 0, 0))
    img = _pixels(r)
    for x in (80, 160, 240):  # three cubes in a row, gaps between them
        assert img[90, x].max() > 100, x
    assert img[90, 120].max() < 20 and img[90, 200].max() < 20


def test_copies_turn_and_scale_as_parts_do():
    copy = {"count": 2, "x": [-60, 60], "rotate": [0, 45], "rotateY": [0, 30], "rotateX": [10, 0], "scale": [1, 1.5],
            "z": [0, 20]}
    parts = [{"shape": "cube", "w": 30, "h": 30, "depth": 2, "x": -60, "rotateX": 10},
             {"shape": "cube", "w": 30, "h": 30, "depth": 2, "x": 60, "rotate": 45, "rotateY": 30, "scale": 1.5,
              "z": 20}]
    a = _shot(_model(copy))._models[("copies", "m")][2]
    b = mesh.merge([mesh.placed(mesh.placed(mesh.unit("cube"), (30, 30, 2)), (1, 1, 1), mesh.part_matrix(p))
                    for p in parts], [0, 0])
    assert np.allclose(a.P, b.P) and np.allclose(a.N, b.N, atol=1e-9) and np.array_equal(a.T, b.T)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_each_copy_takes_its_fill_and_parts_with_their_own_keep_it(backend):
    r = _shot(_model({"count": 3, "x": "=(i - 1) * 80", "fill": ["#ff0000", "#00ff00", "#0000ff"]}), backend=backend)
    img = _pixels(r)
    for x, ch in ((80, 0), (160, 1), (240, 2)):
        assert img[90, x].argmax() == ch and img[90, x].max() > 200, (x, img[90, x])
    spec = scene([{"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat",
                   "parts": [{"shape": "cube", "w": 20, "h": 20, "depth": 2, "y": -15},
                             {"shape": "cube", "w": 20, "h": 20, "depth": 2, "y": 15, "fill": "#ffffff"}],
                   "repeat": {"count": 2, "x": [-60, 60], "fill": ["#ff0000", "#0000ff"]}}])
    img = _pixels(_shot(spec, backend=backend))
    assert img[75, 100].argmax() == 0 and img[75, 220].argmax() == 2  # the part without a fill takes the copy's
    assert img[105, 100].min() > 200 and img[105, 220].min() > 200  # the white part stays white


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_more_fills_than_a_model_draws_take_the_nearest_kept(backend):
    spec = _model({"count": 200, "x": "=(i % 20 - 9.5) * 14", "y": "=(floor(i / 20) - 4.5) * 14",
                   "fill": "=hsl(i * 1.8, 0.8, 0.5)"}, w=10, h=10)
    r = _shot(spec, backend=backend)
    m = r._models[("copies", "m")][2]
    assert m.M.max() + 1 == render3d.COPY_SLOTS
    img = _pixels(r)
    first, last = img[90 - 63, 160 - 133], img[90 + 63, 160 + 133]  # copies 0 and 199: hues 0 and 358, reds
    assert first.argmax() == last.argmax() == 0 and first.max() > 150 and last.max() > 150, (first, last)
    middle = img[90 + 7, 160 + 7]  # copy 110: hue 198, a blue
    assert middle.argmax() == 2 and middle[0] < 100, middle
    colours, q = render3d._palette([(1.0, 0, 0, 1)] * 3 + [(0, 0, 1.0, 1), (0.9, 0, 0.1, 1)], 2)
    assert colours == [(1.0, 0, 0, 1), (0.9, 0, 0.1, 1)] and q.tolist() == [0, 0, 0, 1, 1]
    rep = check(spec)
    assert [i["code"] for i in rep["issues"]] == ["repeat-colours"], rep["issues"]


def test_moving_copies_keep_their_triangles_and_held_ones_their_mesh():
    spec = _model({"count": 5, "x": "=(i - 2) * 50", "y": "=sin(t * 3 + i) * 30", "fill": "=hsl(i * 60)"})
    r = _shot(spec, 0.0)
    first = r._models[("copies", "m")][2]
    r.draw(0.5)
    moved = r._models[("copies", "m")][2]
    assert moved is not first and moved.T is first.T and not np.allclose(moved.P, first.P)
    still = _shot(_model({"count": 5, "x": "=(i - 2) * 50"}), 0.0)
    held = still._models[("copies", "m")][2]
    still.draw(1.0)
    assert still._models[("copies", "m")][2] is held


def test_copies_edges_are_the_ones_of_each_copy():
    one = mesh.placed(mesh.unit("cube"), (30, 30, 30))
    e = mesh.edges(one.P, one.T)
    apart = mesh.cat([one._replace(P=one.P + (100 * i, 0, 0)) for i in range(3)])
    direct, tiled = mesh.edges(apart.P, apart.T), mesh.tiled(e, 3, len(one.T))
    for a, b in zip(direct, tiled):
        assert a.shape == b.shape
    pairs = lambda E: sorted(map(tuple, np.round(apart.P[np.argmax(E[0][None] == E[3][:, :1], 1)], 6).tolist()))  # noqa
    assert pairs(direct) == pairs(tiled)
    assert np.array_equal(tiled[0], direct[0]) and np.array_equal(tiled[1], direct[1])
    touching = _model({"count": 2, "x": [-15, 15]}, edges=True, w=30, h=30, depth=30)  # faces against each other
    r = _shot(touching)
    assert r._models[("edges", "m", "copies")][1][3].shape[0] == 2 * len(e[3])


def test_repeat_frames_are_pure_and_their_expressions_are_checked():
    spec = _model({"count": 30, "x": "=(i % 6 - 2.5) * 40", "y": "=(floor(i / 6) - 2) * 30 + sin(t * 4 + i) * 8",
                   "rotateY": "=t * 90 + i * 12", "fill": "=hsl(i * 12 + t * 60)"}, edges=True, w=20, h=20, depth=20)
    spec["canvas"]["duration"] = 1
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 20)]
    alone = Renderer(k.Scene(spec), backend="cpu").frame(13).copy()
    assert np.array_equal(sequential[13], alone)
    bad = check(_model({"count": 4, "x": "=1 / (i - 2)"}))
    hit = [i for i in bad["issues"] if i["code"] == "bad-expression"]
    assert hit and "repeat x failed for copy 2" in hit[0]["message"], bad["issues"]
    late = check(_model({"count": 4, "y": "=1 / (t - 1)"}))
    assert any(i["code"] == "bad-expression" and "t=1.00s" in i["message"] for i in late["issues"]), late["issues"]
    colour = check(_model({"count": 2, "fill": "=i * 2"}))
    assert any("not a colour" in i["message"] for i in colour["issues"]), colour["issues"]


def test_repeat_keys_are_read_the_way_agents_write_them():
    s = k.Scene(scene([{"id": "a", "type": "model", "shape": "cube",
                        "copies": {"n": 3.0, "colours": ["red", "blue"], "rotation": "=i * 10"}}]))
    assert s.by_id["a"].raw["repeat"] == {"count": 3, "fill": ["red", "blue"], "rotate": "=i * 10"}
    for bad, words in (({"x": "=i"}, "needs a count"), ({"count": 0}, "count is 0"),
                       ({"count": 20_000}, "from 1 to 10,000"), ({"count": 2, "x": "left"}, "repeat x is"),
                       ({"count": 2, "fill": [1, 2]}, "repeat fill is"), ({"count": 2, "fill": "#zz0000"}, "repeat fill"),
                       (12, "repeat is 12"), ({"count": 2, "spin": 3}, "unknown key 'spin'"),
                       ({"count": 2, "x": "=i +"}, "repeat x")):
        with pytest.raises(k.SceneError, match=words):
            k.Scene(scene([{"id": "a", "type": "model", "shape": "cube", "repeat": bad}]))
    with pytest.raises(k.SceneError, match="unknown key 'repeat'"):
        k.Scene(scene([{"id": "t", "type": "text", "text": "Hi", "repeat": {"count": 2}}]))


def test_a_model_files_materials_take_the_copies_fill(tmp_path):
    g = Gltf()
    g.doc["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}]
    g.node(mesh=g.mesh(box(10, 10, 0.2), material=0))
    src = g.save(tmp_path / "plate.gltf")
    spec = scene([{"id": "m", "type": "model", "src": src, "w": 60, "x": 160, "y": 90, "light": "flat",
                   "repeat": {"count": 2, "x": [-60, 60], "fill": ["#ff0000", "#0000ff"]}}])
    img = _pixels(_shot(spec))
    assert img[90, 100].argmax() == 0 and img[90, 220].argmax() == 2, (img[90, 100], img[90, 220])
