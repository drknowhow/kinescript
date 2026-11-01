"""glTF animation: node channels, morph target weights and skins, played on a model layer's clock."""
import base64
import json
import math

import numpy as np
import pytest

import kinescript as k
from kinescript import mesh, models
from kinescript.render import Renderer
from kinescript.verify import check, probe_media

TYPES = {5126: "<f4", 5125: "<u4", 5123: "<u2"}
KINDS = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}


class Gltf:
    """A glTF file built in memory: add() stores an array as an accessor, the rest is the document itself."""

    def __init__(self):
        self.blob, self.doc = bytearray(), {"asset": {"version": "2.0"}, "accessors": [], "bufferViews": [],
                                            "meshes": [], "nodes": [], "scene": 0, "scenes": [{"nodes": []}]}

    def add(self, arr, ctype=5126) -> int:
        arr = np.asarray(arr, TYPES[ctype])
        arr = arr.reshape(len(arr), -1)
        b = arr.tobytes()
        self.doc["bufferViews"].append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(b)})
        self.blob.extend(b + b"\0" * (-len(b) % 4))
        self.doc["accessors"].append({"bufferView": len(self.doc["bufferViews"]) - 1, "componentType": ctype,
                                      "count": len(arr), "type": KINDS[arr.shape[1]]})
        return len(self.doc["accessors"]) - 1

    def mesh(self, m, **prim) -> int:
        """A mesh of kinescript mesh m, stored y up and counter-clockwise, with more primitive keys."""
        at = {"POSITION": self.add(m.P * (1, -1, 1)), "NORMAL": self.add(m.N * (1, -1, 1))}
        at.update(prim.pop("attributes", {}))
        self.doc["meshes"].append({"primitives": [dict(attributes=at, indices=self.add(m.T[:, [0, 2, 1]].ravel(), 5125),
                                                       **prim)]})
        return len(self.doc["meshes"]) - 1

    def node(self, root=True, **nd) -> int:
        self.doc["nodes"].append(nd)
        if root:
            self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]) - 1)
        return len(self.doc["nodes"]) - 1

    def clip(self, name, *channels):
        """channels: (node, path, times, values[, interpolation])."""
        samplers, chans = [], []
        for node, path, times, values, *how in channels:
            samplers.append({"input": self.add(times), "output": self.add(values),
                             "interpolation": how[0] if how else "LINEAR"})
            chans.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
        self.doc.setdefault("animations", []).append({"name": name, "samplers": samplers, "channels": chans})

    def save(self, path) -> str:
        self.doc["buffers"] = [{"byteLength": len(self.blob), "uri": "data:application/octet-stream;base64,"
                                + base64.b64encode(bytes(self.blob)).decode()}]
        path.write_text(json.dumps(self.doc))
        return str(path)


def turn(axis, degrees):
    """A quaternion [x, y, z, w]."""
    h = math.radians(degrees) / 2
    return [*(np.array(axis, float) * math.sin(h)), math.cos(h)]


def file_space(imp, P):
    """Posed points as the file holds them: y up, not centred."""
    return (P + imp.motion.centre) * (1, -1, 1)


def box(w=10, h=10, d=10):
    return mesh.placed(mesh.unit("cube"), (w, h, d))


def test_node_channels_move_turn_and_scale_the_mesh(tmp_path):
    g = Gltf()
    n = g.node(mesh=g.mesh(box(20, 4, 4)))
    g.clip("go", (n, "translation", [0, 1], [[0, 0, 0], [30, 0, 0]]),
           (n, "rotation", [0, 2], [turn([0, 1, 0], 0), turn([0, 1, 0], 90)]),
           (n, "scale", [0, 1], [[1, 1, 1], [2, 2, 2]], "STEP"))
    imp = models.load(g.save(tmp_path / "m.gltf"))
    rest = file_space(imp, imp.mesh.P)
    for t, x, deg, s in ((0.5, 15, 22.5, 1), (1.5, 30, 67.5, 2)):  # translation held past its last key
        P, N = models.pose(imp, 0, t)
        q = turn([0, 1, 0], deg)
        R = models._trs([0, 0, 0], q, [s, s, s])[:3, :3]
        assert np.allclose(file_space(imp, P), rest @ R.T + [x, 0, 0], atol=1e-4), t
        assert np.allclose(N * (1, -1, 1), imp.mesh.N * (1, -1, 1) @ (R / s).T, atol=1e-6)
    P, N = models.pose(imp, 0, 0)  # every channel at the node's own values: the mesh as read
    assert np.allclose(P, imp.mesh.P, atol=1e-5) and np.allclose(N, imp.mesh.N, atol=1e-6)


def test_cubic_spline_keys_follow_their_tangents(tmp_path):
    g = Gltf()
    n = g.node(mesh=g.mesh(box()))
    # Keys at 0 and 1 s, each [in-tangent, value, out-tangent]: x leaves 0 at 10 px/s and arrives at 10 level.
    g.clip("ease", (n, "translation", [0, 1], [[0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0]],
                    "CUBICSPLINE"))
    imp = models.load(g.save(tmp_path / "m.gltf"))
    P, _ = models.pose(imp, 0, 0.5)
    # h00 0 + h10 * 10 + h01 * 10 + h11 * 0 at u = 0.5: 0.125 * 10 + 0.5 * 10
    assert (P - imp.mesh.P)[:, 0] == pytest.approx(6.25, abs=1e-4)


@pytest.mark.parametrize("sparse", [False, True])
def test_morph_target_weights_rest_at_the_files_and_animate(tmp_path, sparse):
    plane = mesh.placed(mesh.unit("plane"), (20, 20, 1))
    lift = np.zeros_like(plane.P)
    lift[plane.P[:, 0] > 0, 2] = 10  # the right edge comes 10 toward the viewer
    g = Gltf()
    if sparse:  # only the moved vertices stored, at their indices, as exporters often write targets
        idx = np.flatnonzero(lift[:, 2])
        at = [g.doc["accessors"][g.add(a, c)]["bufferView"]
              for a, c in ((idx, 5125), (lift[idx] * (1, -1, 1), 5126))]
        g.doc["accessors"].append({"count": len(lift), "componentType": 5126, "type": "VEC3", "sparse": {
            "count": len(idx), "indices": {"bufferView": at[0], "componentType": 5125}, "values": {"bufferView": at[1]}}})
        target = len(g.doc["accessors"]) - 1
    else:
        target = g.add(lift * (1, -1, 1))
    m = g.mesh(plane, targets=[{"POSITION": target}])
    g.doc["meshes"][m]["weights"] = [0.2]
    n = g.node(mesh=m)
    g.clip("lift", (n, "weights", [0, 1], [0, 1]))
    imp = models.load(g.save(tmp_path / "m.gltf"))
    assert np.ptp(imp.mesh.P[:, 2]) == pytest.approx(2, abs=1e-4)  # at the mesh's own weight
    assert np.ptp(models.pose(imp, 0, 0.5)[0][:, 2]) == pytest.approx(5, abs=1e-4)
    assert probe_media(str(tmp_path / "m.gltf"))["morphTargets"] == 1


def _arm(g, crowd=False, mesh_scale=None):
    """A tube 100 tall and 40 wide on two joints: the lower half follows the root, the upper half the elbow at
    y 50, as the file holds it (y up)."""
    tube = mesh.placed(mesh.unit("cylinder"), (20, 100, 20))
    tube = tube._replace(P=tube.P + (0, -50, 0))  # y 0 to 100 in the file (y down here)
    up = -tube.P[:, 1]
    joints = np.where(up[:, None] >= 50, [[1, 0, 0, 0]], [[0, 0, 0, 0]])
    weights = np.tile([1.0, 0, 0, 0], (len(up), 1))
    at = {"JOINTS_0": g.add(joints, 5123), "WEIGHTS_0": g.add(weights)}
    if crowd:
        at |= {"JOINTS_1": g.add(np.zeros_like(joints), 5123), "WEIGHTS_1": g.add(np.where(up[:, None] > 99,
                                                                                          [[0.1, 0, 0, 0]], 0))}
    root = g.node(children=[1])
    elbow = g.node(root=False, translation=[0, 50, 0])
    ibm = np.stack([np.eye(4), np.array(models._trs([0, -50, 0], [0, 0, 0, 1], [1, 1, 1]))])
    g.doc["skins"] = [{"joints": [root, elbow], "inverseBindMatrices": g.add(ibm.transpose(0, 2, 1).reshape(2, 16))}]
    nd = {"mesh": g.mesh(tube, attributes=at), "skin": 0}
    if mesh_scale:
        nd["scale"] = mesh_scale  # glTF ignores a skinned mesh node's own transform
    g.node(**nd)
    g.clip("bend", (elbow, "rotation", [0, 1], [turn([0, 0, 1], 0), turn([0, 0, 1], 90)]))
    return elbow


def test_a_skin_bends_its_vertices_with_their_joints(tmp_path):
    g = Gltf()
    _arm(g, mesh_scale=[3, 3, 3])
    imp = models.load(g.save(tmp_path / "arm.gltf"))
    assert imp.size.tolist() == pytest.approx([40, 100, 40], abs=1e-3)  # the node's scale is not applied
    rest = file_space(imp, imp.mesh.P)
    P, N = models.pose(imp, 0, 0)
    assert np.allclose(P, imp.mesh.P, atol=1e-4) and np.allclose(N, imp.mesh.N, atol=1e-6)
    bent = file_space(imp, models.pose(imp, 0, 1)[0])
    top, low = rest[:, 1] > 99.9, rest[:, 1] < 0.1
    # Turned 90 degrees about z round the elbow: the top end lies along -x, 50 out, at the elbow's height.
    assert np.allclose(bent[top, 0], -50, atol=1e-3)
    assert np.allclose(bent[top, 1], 50 + rest[top, 0], atol=1e-3)
    assert np.allclose(bent[low], rest[low], atol=1e-4)  # the lower end stays with the root
    half = file_space(imp, models.pose(imp, 0, 0.5)[0])
    c = math.cos(math.pi / 4)  # 45 degrees: halfway along the sphere
    assert np.allclose(half[top, 0], (rest[top, 0] - 50) * c, atol=1e-3)
    assert np.allclose(half[top, 1], 50 + (rest[top, 0] + 50) * c, atol=1e-3)


def _shot(spec, T):
    r = Renderer(k.Scene(spec), backend="cpu", track=True)
    r.draw(T)
    return r


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 2}, **canvas), "layers": layers}


def _slide(tmp_path):
    g = Gltf()
    n = g.node(mesh=g.mesh(box(10, 10, 0.2)))  # thin, so perspective does not widen it as it moves
    g.clip("idle", (n, "rotation", [0, 1], [turn([0, 1, 0], 0), turn([0, 1, 0], 0)]))
    g.clip("slide", (n, "translation", [0, 1], [[0, 0, 0], [20, 0, 0]]))
    return g.save(tmp_path / "slide.gltf")


def test_a_model_layer_plays_a_clip_on_its_clock(tmp_path):
    src = _slide(tmp_path)
    base = {"id": "m", "type": "model", "src": src, "x": 100, "y": 90, "h": 40}

    def left(T, **kw):
        return _shot(scene([dict(base, **kw)]), T).boxes["m"]["rect"][0]

    still = left(0.5)
    assert left(0.5, animation="slide") == pytest.approx(still + 40, abs=2)  # 10 of 20 units, 4 px a unit
    assert left(1.25, animation="slide") == pytest.approx(still + 20, abs=2)  # it loops
    held = {"name": "slide", "loop": False}
    assert left(1.25, animation=held) == pytest.approx(still + 80, abs=2)
    assert left(0.25, animation=dict(held, speed=2, start=0.25)) == pytest.approx(still + 60, abs=2)
    assert left(0.5, animation=1) == left(0.5, animation="Slide")  # by number from 0, or name in any case
    assert left(0.5, animation=True) == pytest.approx(still, abs=1)  # the first: idle


def test_animated_model_frames_are_pure_functions_of_their_index(tmp_path):
    g = Gltf()
    _arm(g)
    src = g.save(tmp_path / "arm.gltf")
    spec = scene([{"id": "m", "type": "model", "src": src, "x": 160, "y": 90, "h": 150, "edges": True,
                   "animation": {"name": "bend", "speed": 0.7}}], duration=1)
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 20)]
    alone = Renderer(k.Scene(spec), backend="cpu").frame(13).copy()
    assert np.array_equal(sequential[13], alone)
    assert not np.array_equal(sequential[0], sequential[13])


def test_check_names_a_files_clips_and_warns_of_dropped_joints(tmp_path):
    src = _slide(tmp_path)
    r = check(scene([{"id": "m", "type": "model", "src": src, "animation": "slid"}]))
    hit = [i for i in r["issues"] if i["code"] == "unknown-animation"]
    assert not r["ok"] and hit and "'idle', 'slide'" in hit[0]["message"] and "'slide'" in hit[0]["fix"], r["issues"]
    still = str(tmp_path / "still.gltf")
    g = Gltf()
    g.node(mesh=g.mesh(box()))
    g.save(tmp_path / "still.gltf")
    hit = [i for i in check(scene([{"id": "m", "src": still, "animation": 0}]))["issues"]
           if i["code"] == "unknown-animation"]
    assert hit and "no animations" in hit[0]["message"]
    g = Gltf()
    _arm(g, crowd=True)
    crowded = g.save(tmp_path / "crowd.gltf")
    hit = [i for i in check(scene([{"id": "m", "src": crowded}]))["issues"] if i["code"] == "skin-influences"]
    assert hit and hit[0]["level"] == "warning", hit


def test_probe_lists_a_files_animations_skins_and_morph_targets(tmp_path):
    g = Gltf()
    _arm(g)
    info = probe_media(g.save(tmp_path / "arm.gltf"))
    assert info["animations"] == [{"name": "bend", "duration": 1.0}]
    assert info["skins"] == [{"name": "0", "joints": 2}] and info["morphTargets"] == 0
    assert info["uvSets"] == [] and info["textures"] == 0


def test_the_sample_arm_bends_on_its_joints():
    from pathlib import Path
    arm = Path(__file__).resolve().parents[1] / "examples" / "media" / "arm.glb"
    if not arm.exists():
        pytest.skip("run examples/make_media.py first")
    imp = models.load(str(arm))
    assert models.clip_names(imp) == ["wave", "curl"] and imp.about["skins"] == [{"name": "arm", "joints": 3}]
    rest = imp.mesh.P
    P, _ = models.pose(imp, models.clip_index(imp, "curl"), 1.5)
    top = rest[:, 1] < rest[:, 1].min() + 1  # y runs down: the tip
    assert (P[top, 0] - rest[top, 0]).mean() < -50  # curled over to the left
    assert np.allclose(P[rest[:, 1] > rest[:, 1].max() - 1], rest[rest[:, 1] > rest[:, 1].max() - 1], atol=1e-4)


def test_animation_keys_are_read_the_way_agents_write_them(tmp_path):
    src = _slide(tmp_path)
    s = k.Scene(scene([{"id": "a", "src": src, "animation": {"clip": "slide", "rate": 2, "repeat": "false"}},
                       {"id": "b", "type": "model", "file": src, "animation": {"speed": 0.5}}]))
    assert s.by_id["a"].raw["animation"] == {"name": "slide", "speed": 2, "loop": False}
    assert s.by_id["b"].raw["animation"] == {"speed": 0.5}  # a clip object, not a wrapper of layer keys
    for bad, words in (({"name": "slide", "speed": "fast"}, "speed"), (-1, "animation is -1"),
                       ({"name": [1]}, "name")):
        with pytest.raises(k.SceneError, match=words):
            k.Scene(scene([{"id": "c", "src": src, "animation": bad}]))
    t = k.Scene(scene([{"id": "t", "type": "text", "text": "Hi", "animation": {"name": "fade"}},
                       {"id": "u", "type": "model", "shape": "cube", "animation": {"name": "rise"}}]))
    assert t.by_id["t"].raw.get("enter") == {"name": "fade"}  # elsewhere it is still the entrance
    assert t.by_id["u"].raw.get("enter") == {"name": "rise"}  # and on a model with no file
