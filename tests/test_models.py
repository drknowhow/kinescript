"""Imported models (glTF and OBJ files) and the ground shadow under model layers."""
import base64
import json
import struct

import numpy as np
import pytest

import kinescript as k
from kinescript import mesh, models
from kinescript.render import Renderer
from kinescript.verify import check


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 1}, **canvas), "layers": layers}


def write_gltf(path, prims, nodes=None, roots=None, materials=(), binary=False, extra=None, images=()):
    """A glTF file of `prims` [(P, T, N or None, material or None[, texture coordinates])], stored y up, faces
    counter-clockwise, with `images` (PNG bytes) stored in its buffer."""
    blob, views, accessors, meshes = bytearray(), [], [], []

    def view(b: bytes) -> int:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(b)})
        blob.extend(b + b"\0" * (-len(b) % 4))
        return len(views) - 1

    def add(arr, ctype, kind):
        accessors.append({"bufferView": view(arr.tobytes()), "componentType": ctype, "count": len(arr), "type": kind})
        return len(accessors) - 1

    pictures = [{"bufferView": view(png), "mimeType": "image/png"} for png in images]
    for P, T, N, mat, *uv in prims:
        at = {"POSITION": add(np.asarray(P, "<f4"), 5126, "VEC3")}
        if uv:
            at["TEXCOORD_0"] = add(np.asarray(uv[0], "<f4"), 5126, "VEC2")
        if N is not None:
            at["NORMAL"] = add(np.asarray(N, "<f4"), 5126, "VEC3")
        prim = {"attributes": at, "indices": add(np.asarray(T, "<u4").ravel(), 5125, "SCALAR")}
        if mat is not None:
            prim["material"] = mat
        meshes.append({"primitives": [prim]})
    nodes = nodes or [{"mesh": i} for i in range(len(meshes))]
    doc = {"asset": {"version": "2.0"}, "meshes": meshes, "accessors": accessors, "bufferViews": views,
           "materials": list(materials), "nodes": nodes, "scene": 0,
           **({"images": pictures, "textures": [{"source": i} for i in range(len(pictures))]} if pictures else {}),
           "scenes": [{"nodes": roots if roots is not None else list(range(len(nodes)))}], **(extra or {})}
    if binary:
        doc["buffers"] = [{"byteLength": len(blob)}]
        js = json.dumps(doc).encode()
        js += b" " * (-len(js) % 4)
        body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
        path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    else:
        doc["buffers"] = [{"byteLength": len(blob),
                           "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]
        path.write_text(json.dumps(doc))
    return str(path)


def file_space(m):
    """A kinescript mesh as a file holds it: y up, faces counter-clockwise."""
    return m.P * (1, -1, 1), m.T[:, [0, 2, 1]], m.N * (1, -1, 1)


def outward(m):
    """Per triangle: its winding's normal agrees with its vertex normals, and (for a convex solid centred on
    the origin) points away from the centre."""
    a, b, c = m.P[m.T[:, 0]], m.P[m.T[:, 1]], m.P[m.T[:, 2]]
    fn = np.cross(b - a, c - a)
    return ((fn * m.N[m.T].sum(1)).sum(1) > 0) & ((fn * (a + b + c)).sum(1) > 0)


RED = {"pbrMetallicRoughness": {"baseColorFactor": [1.0, (99 / 255) ** 2.2, (63 / 255) ** 2.2, 1.0],
                                "metallicFactor": 0.0, "roughnessFactor": 0.5}}


@pytest.mark.parametrize("binary", [False, True])
def test_gltf_is_read_y_down_with_faces_outward_and_its_colour(tmp_path, binary):
    cone = mesh.placed(mesh.unit("cone"), (50, 100, 50))  # tip up, which is -y here
    P, T, N = file_space(cone)
    imp = models.load(write_gltf(tmp_path / ("m.glb" if binary else "m.gltf"), [(P, T, N, 0)], materials=[RED],
                                 binary=binary))
    assert imp.mesh.P[imp.mesh.P[:, 1].argmin()] == pytest.approx([0, -50, 0], abs=1e-4)  # tip still up
    assert outward(imp.mesh).all()
    assert imp.size.tolist() == pytest.approx([100, 100, 100], abs=0.1)
    assert imp.materials == [{"fill": pytest.approx((1.0, 99 / 255, 63 / 255), abs=2e-3), "shine": 0.5, "metal": 0.0}]


def test_node_tree_applies_and_a_mirroring_node_keeps_faces_outward(tmp_path):
    box = mesh.placed(mesh.unit("cube"), (100, 20, 20))
    P, T, N = file_space(box)
    turn = [0, np.sin(np.pi / 4), 0, np.cos(np.pi / 4)]  # 90 degrees about y: the long side runs along z
    nodes = [{"children": [1], "rotation": turn}, {"children": [2], "scale": [-1, 1, 1]}, {"mesh": 0}]
    imp = models.load(write_gltf(tmp_path / "m.gltf", [(P, T, None, None)], nodes=nodes, roots=[0]))
    assert imp.size.tolist() == pytest.approx([20, 20, 100])
    assert outward(imp.mesh).all()
    assert set(imp.mesh.M.tolist()) == {-1}  # no material in the file: the layer's own
    a, b, c = (imp.mesh.P[imp.mesh.T[:, i]] for i in range(3))
    fn = np.cross(b - a, c - a)
    flat = fn / np.linalg.norm(fn, axis=1, keepdims=True)
    assert np.allclose(imp.mesh.N[imp.mesh.T[:, 0]], flat, atol=1e-6)  # no normals in the file: flat faces


def test_long_triangles_are_cut_for_the_depth_sort(tmp_path):
    quad = np.array([[-100, -100, 0], [100, -100, 0], [100, 100, 0], [-100, 100, 0]], float)
    imp = models.load(write_gltf(tmp_path / "m.gltf", [(quad, [[0, 1, 2], [0, 2, 3]], None, None)]))
    P, T = imp.mesh.P, imp.mesh.T
    edges = np.linalg.norm(P[T] - P[np.roll(T, 1, axis=1)], axis=2)
    assert len(T) > 2 and edges.max() <= 200 / 24 + 1e-6
    assert outward(imp.mesh._replace(P=P + (0, 0, 1))).all()  # cut pieces keep the face's winding


def test_obj_reads_polygons_negative_indices_mtl_colours_and_smoothing(tmp_path):
    (tmp_path / "m.mtl").write_text("newmtl paint\nKd 1 0.5 0\nNs 250\n")
    v = "".join(f"v {x} {y} {z}\n" for x, y, z in [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                                                   (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)])
    faces = ["1 2 3 4", "8 7 6 5", "4 3 7 8", "5 6 2 1", "2 6 7 3", "5 1 4 8"]  # counter-clockwise from outside
    neg = [" ".join(str(int(i) - 9) for i in f.split()) for f in faces[3:]]  # -1 is the last vertex
    (tmp_path / "m.obj").write_text("mtllib m.mtl\n" + v + "usemtl paint\n" + "".join(f"f {f}\n" for f in faces[:3])
                                    + "usemtl unknown\ns 1\n" + "".join(f"f {f}\n" for f in neg))
    imp = models.load(str(tmp_path / "m.obj"))
    m = imp.mesh
    assert outward(m).all()
    assert imp.size.tolist() == pytest.approx([2, 2, 2])
    assert imp.materials == [{"fill": (1.0, 0.5, 0.0), "shine": 0.5, "metal": 0.0}]
    unknown, paint = np.bincount(m.M[m.T[:, 0]] + 1)  # usemtl of an unknown material: the layer's own
    assert unknown == paint > 0  # three quads each
    smooth = m.N[m.M == -1]
    assert np.abs(smooth).max(1).min() < 0.99  # "s 1" corners average the faces around them; flat ones do not
    assert np.abs(m.N[m.M == 0]).max(1).min() == pytest.approx(1.0)


def test_more_than_eight_materials_merge_into_the_nearest_kept_colour():
    cols = [(i / 9, 0.0, 0.0) for i in range(10)]
    P = np.zeros((30, 3))
    M = np.repeat(np.arange(10), 3)
    M[:3] = 9  # colour 0 is unused, colour 9 is on the most vertices
    imp = models.Imported(mesh.Mesh(P, P, np.arange(30).reshape(-1, 3), M),
                          [{"fill": c, "shine": 0.4, "metal": 0.0} for c in cols], np.ones(3))
    slot, kept = models.slots(imp, 7)
    assert len(kept) == 7 and kept[0]["fill"] == cols[9]
    got = np.array([kept[s]["fill"][0] for s in slot])
    assert np.abs(got - np.array([cols[i][0] for i in M])).max() <= 1 / 9 + 1e-9
    assert (models.slots(imp, 0)[0] == -1).all()


def _shot(spec, T=0.0, track=False):
    r = Renderer(k.Scene(spec), backend="cpu", track=track)
    r.draw(T)
    return r, r.snapshot().toarray()[..., :3].astype(int)


def test_model_src_fits_its_box_and_a_layer_fill_recolours_it(tmp_path):
    P, T, N = file_space(mesh.placed(mesh.unit("cube"), (10, 5, 0.2)))  # thin, so depth does not enlarge it
    src = write_gltf(tmp_path / "m.glb", [(P, T, N, 0)], materials=[RED], binary=True)
    layer = {"id": "m", "type": "model", "src": src, "x": 160, "y": 90, "w": 200, "h": 60}
    r, img = _shot(scene([layer], bg="#000000"), track=True)
    x0, y0, x1, y1 = r.boxes["m"]["rect"]
    assert (x1 - x0, y1 - y0) == (pytest.approx(120, abs=3), pytest.approx(60, abs=3))  # h is the tighter fit
    lit = img[img.sum(2) > 30]
    assert (lit[:, 0] > lit[:, 2] + 40).mean() > 0.95  # the file's red
    _, img = _shot(scene([dict(layer, fill="#2050ff")], bg="#000000"))
    lit = img[img.sum(2) > 30]
    assert (lit[:, 2] > lit[:, 0] + 40).mean() > 0.95
    alone = {"id": "m", "src": src, "x": 160, "y": 90}  # the type comes from the file's extension
    r, _ = _shot(scene([alone], w=640, h=480), track=True)
    x0, y0, x1, y1 = r.boxes["m"]["rect"]
    assert x1 - x0 == pytest.approx(300, abs=4)  # no w or h: the longest side is 300 px


def test_unreadable_compressed_or_missing_model_files_are_reported_by_check(tmp_path):
    P, T, N = file_space(mesh.placed(mesh.unit("cube"), (10, 10, 10)))
    draco = write_gltf(tmp_path / "d.gltf", [(P, T, N, None)],
                       extra={"extensionsRequired": ["KHR_draco_mesh_compression"]})
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"glTF" + struct.pack("<II", 2, 40) + b"\0" * 8)
    for src, code, words in ((draco, "bad-model", "KHR_draco_mesh_compression"), (str(bad), "bad-model", "bad.glb"),
                             (str(tmp_path / "gone.glb"), "media-missing", "gone.glb")):
        r = check(scene([{"id": "m", "type": "model", "src": src}]))
        hit = [i for i in r["issues"] if i["code"] == code]
        assert not r["ok"] and hit and words in hit[0]["message"], r["issues"]


def _floor_darkness(img, y0, y1):
    """How much darker than the white background the band of rows y0..y1 gets, summed."""
    return float((255 * 3 - img[y0:y1].sum(2)).sum())


def test_ground_shadow_darkens_the_floor_under_the_model_and_fades_as_it_rises():
    ball = {"id": "b", "type": "model", "shape": "sphere", "r": 40, "x": 160, "y": 90, "fill": "#ffffff"}
    spec = lambda **kw: dict(scene([dict(ball, **kw)], bg="#ffffff"), camera={"rotateX": 25})  # noqa: E731
    _, plain = _shot(spec())
    _, touching = _shot(spec(ground={"y": 130}))
    _, raised = _shot(spec(ground={"y": 130}, y=40))
    _, rest = _shot(spec(ground=True))  # floor at the ball's lowest point: the same as y 130
    rows = (136, 175)  # below the ball, where only the shadow can be
    assert _floor_darkness(plain, *rows) == 0
    assert _floor_darkness(touching, *rows) > 2000
    assert _floor_darkness(raised, *rows) < 0.6 * _floor_darkness(touching, *rows)
    assert np.array_equal(rest, touching)


def test_imported_model_frames_are_pure_functions_of_their_index(tmp_path):
    P, T, N = file_space(mesh.placed(mesh.unit("cylinder"), (40, 80, 40)))
    src = write_gltf(tmp_path / "m.glb", [(P, T, N, 0)], materials=[RED], binary=True)
    spec = scene([{"id": "m", "type": "model", "src": src, "x": 160, "y": 80, "h": 120, "rotateY": "=t*200",
                   "ground": {"opacity": 0.7}, "enter": {"name": "build", "dur": 0.5}}], duration=1)
    r = Renderer(k.Scene(spec), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(0, 20)]
    alone = Renderer(k.Scene(spec), backend="cpu").frame(13).copy()
    assert np.array_equal(sequential[13], alone)


def test_model_surface_is_the_frame_size_whatever_was_drawn_before():
    # It used to grow to fit the largest model so far, and on the GPU the size it had moved model edges in
    # later frames by up to 49 levels, so a frame depended on the frames rendered before it.
    spec = scene([{"id": "m", "type": "model", "shape": "sphere", "r": 10, "x": 160, "y": 90,
                   "scale": [[0, 1], [1, 8]]}], duration=1)
    for backend in ("auto", "cpu"):
        r = Renderer(k.Scene(spec), backend=backend)
        sizes = []
        for n in (0, 29, 0):
            r.frame(n)
            surf, ss = r._model_surface()
            sizes.append((surf.width(), surf.height()))
        assert sizes == [(r.w * ss, r.h * ss)] * 3, backend


def test_model_file_keys_are_read_the_way_agents_write_them(tmp_path):
    src = str(tmp_path / "m.obj")
    (tmp_path / "m.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    s = k.Scene(scene([{"id": "a", "type": "model", "shape": src}, {"id": "b", "type": "3d", "file": src},
                       {"id": "c", "type": "model", "src": src, "floor": {"alpha": 0.3}}]))
    assert [s.by_id[i].src for i in "abc"] == [src] * 3
    assert s.by_id["c"].raw["ground"] == {"opacity": 0.3}
    assert "renamed-key" in [i["code"] for i in s.issues]
    with pytest.raises(k.SceneError, match="ground"):
        k.Scene(scene([{"id": "d", "type": "model", "src": src, "ground": "yes"}]))
