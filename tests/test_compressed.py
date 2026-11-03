"""Compressed glTF files: buffer views packed by meshopt (EXT_meshopt_compression, as gltfpack -c writes them),
Draco-compressed meshes (KHR_draco_mesh_compression, read with the draco extra), and the KHR_texture_transform
gltfpack adds for the texture coordinates it stores quantized."""
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from kinescript import meshopt, models
from kinescript.verify import check
from test_models import scene, write_gltf
from test_textures import BLUE, GREEN, RED, YELLOW, frame, near, quadrants

VECTORS = {  # written by meshoptimizer 1.3.0's encoder (npm meshoptimizer) from the sources below
    "v0": ("a00000000000000000afaa55550000ff8080404040402020202020202020101010101010101010101010101010f888888888"
           "8888881088888888888888888444444444444444444444444444444444444444444444444444444444444444eaaaaaaa04aa"
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa05010000308000007e8000000080000000000000000000"
           "0000ffffffff003fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040"
           "d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd0"
           "4040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf40"
           "3fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3f"
           "cf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf3f3fcf403fd04040d08f0090cf"
           "3f3fcf403fd04040d08f00905555555503f03f00fffffffffffffc0fc03fffffffffffffffffff03f00fc0fffffffffffffc"
           "03f03fffffffffffffffffff00fc0fc0ffffffffffff3f03f00fffffffffffffffffc0fc03f0ffffffffffffff3f00fc0fff"
           "ffffffffffffffc03f03f0ffffffffffffff0fc0fc03fffffffffffffff03f00fcffffffffffffffff0fc03f03ffffffffff"
           "fffff00fc0fcffffffffffffffff03f03f00fffffffffffffc0fc03fffffffffffffffffff03f00fc0ffffffffffff00013f"
           "000000ffffff01880000000180000000000001ff000000cf3f3fcf01fc000000ffffff000000000000000000000000000000"
           "000000000000000000000000000000c8c0"),
    "v1": ("a11a79acaa5555f8ff8080404040402020202020202020101010101010101010101010101010f88888888888888810888888"
           "88888888888444444444444444444444444444444444444444444444444444444444444444eaaaaaaa04aaaaaaaaaaaaaaaa"
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa0100000006007e0155555555015015005405401501500540540150150054"
           "0540150150054054015015005405401501500540540150150054054015015005405401501500540540150150054000c07040"
           "c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c0"
           "7040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070"
           "c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c0"
           "4070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090"
           "d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d0900090d0c04070c0c07040c0d090"
           "0090555555550414504105141041450410514104145041051410414504105141041450410514104145041051410414504105"
           "1410414504105141041450410514104145041051bffb00ffffff00ffffff0200020001010100d0c040700100000100000000"
           "00000000000000000000000000000000c8c00112"),
    "tri16": ("e1fe1e100e100e100e100e100efe1e1b0e1a0e190e180e170eff1e170e170e170e170e170eff1e170e170e170e170e170eff"
              "1e170e170e170e170e170eff1e170e170e170e170e170efefffffffffffefffffffffffefffffffffffeff0f0ebf026f0b0e"
              "6f0b0e6f0b0e6f0b0e00ff151926ff232c17ff2b100cff335631ff26270dff1e1ef835028f1944ff063520ff01101df5090b"
              "ff0b3aff10434eff2102108229ff172c09ff28511a9f14ff0b3a00007687566778a9866589689801690000"),
    "tri32": ("e1fe1e100e100e100e100e100efe1e1b0e1a0e190e180e170eff1e170e170e170e170e170eff1e170e170e170e170e170eff"
              "1e170e170e170e170e170eff1e170e170e170e170e170efefffffffffffefffffffffffefffffffffffeff0f0ebf026f0b0e"
              "6f0b0e6f0b0e6f0b0e00ff151926ff232c17ff2b100cff335631ff26270dff1e1ef835028f1944ff063520ff01101df5090b"
              "ff0b3aff10434eff2102108229ff172c09ff28511a9f14ff0b3a00007687566778a9866589689801690000"),
    "grid": ("a000007f0040ffa0605050df40ffa0605050df40ffa0605050df40ffa0605050df40ffa0605050df40ffa0605050df40ffa0"
             "6050c000000050153083c20f8283828382083c20f08382838283c20f08838283820000150003000c40ff003000c0a0600300"
             "0c005050050003000082000000800000153ccf33cc8080ff8080ff8080fff33ccf338080ff8080ff8080ff80ccf33ccf80ff"
             "8080ff8080ff808055363d8f63807f807f807fd8f63d8f807f807f807f8063d8f63d7f807f807f8080000000000000000000"
             "0000000000000000000000000000000000000000000000000000"),
    "seq32": ("d1000418161c02160418161c02160418161c02160418161c02160418161c02160418161c02120418161c02160418161c0216"
              "0418161c02160418161c02160418161c02160418161c02120418161c02160418161c02160418161c02160418161c02160418"
              "161c02160418161c02120418161c02160418161c02160418161c02160418161c02160418161c02160418161c02120418161c"
              "02160418161c02160418161c02160418161c02160418161c02160418161c02120418161c02160418161c02160418161c0216"
              "0418161c02160418161c02160418161c0201050526324c46582e56201866a901634d4f1b3d3d082c04749b0116060c6a4002"
              "203a12160c22746e9901081543052153336963591351163440164630740000000000"),
    "oct": ("a0011fffff000e5baa30bc669b4e2006013fffcf001b1f1a62173acb76472d00000000000000000000000000000000000000"
            "0000000000000000000000a9357f00"),
    "oct16": ("a0013fffff001fe430aaf31f6628ec0472020003a2c674200000013fffcf00300355187a57cb9d79e7020112614b60310000"
              "0000000000000000000000000000000000000000000000000000000087fa5e03ff070000"),
    "quat": ("a0013fffff00d1a3c2b6ef5d299a27d0f6020137ff6664f60000121513013fffff00a2ac93b0b6b145560fbae30115ffd100"
             "0e0d100a05013fffff002f0815e8b580deb681cc8302035cdedf4b6200001001008b00000300000000000000000000000000"
             "000000000000000000000000bb0334fdef00fc07"),
    "exp": ("a0013fffff0030301f60803f3f7f6f303001002f00000807010020000001011fa00005060100af5800201d000001011fa000"
            "0506013cffff00bfbf80ffffffff40bfbf013fffff001614272c049c0e03171416010008000001011fa00005060000000000"
            "00000000000000000000000000000090fffffc000000fc80c1fffc"),
}


def _pairs():
    i = np.arange(260)
    return np.stack([i * 0.5, i * i % 13 - 6.25], 1).astype("<f4")


def _grid():
    i = np.arange(49)
    return np.stack([i % 7 * 10, i // 7 * 10, i * 3 % 5], 1).astype("<f4")


def _indices(extra=True):
    idx = [v for y in range(6) for x in range(6) for a in [y * 7 + x] for v in (a, a + 1, a + 7, a + 1, a + 8, a + 7)]
    if extra:
        idx += [v for i in range(20) for v in (i * 37 % 49, (i * 23 + 1) % 49, (i * 41 + 2) % 49)]
    return np.array(idx)


def _turned(idx):
    """Triangles each from its lowest index, keeping their winding: the codec may start a triangle elsewhere."""
    t = np.asarray(idx, np.int64).reshape(-1, 3)
    first = t.argmin(1)
    return t[np.arange(len(t))[:, None], (first[:, None] + np.arange(3)) % 3]


def _decode(name, count, size, mode="ATTRIBUTES", filter="NONE"):
    data = bytes.fromhex(VECTORS[name])
    return meshopt.decode({"buffer": 0, "byteLength": len(data), "byteStride": size, "count": count, "mode": mode,
                           "filter": filter}, data)


def test_meshopt_views_decode_to_what_was_encoded():
    assert _decode("v0", 260, 8) == _pairs().tobytes()  # two blocks, of 256 and 4 elements
    assert _decode("v1", 260, 8) == _pairs().tobytes()  # codec version 1
    assert _decode("grid", 49, 12) == _grid().tobytes()
    idx = _indices()
    assert np.array_equal(_turned(np.frombuffer(_decode("tri16", len(idx), 2, "TRIANGLES"), "<u2")), _turned(idx))
    assert np.array_equal(_turned(np.frombuffer(_decode("tri32", len(idx), 4, "TRIANGLES"), "<u4")), _turned(idx))
    assert np.array_equal(np.frombuffer(_decode("seq32", len(idx), 4, "INDICES"), "<u4"), idx)


def test_meshopt_filters_give_back_unit_vectors_quaternions_and_floats():
    k = np.arange(12)
    unit = np.stack([k - 5.5, 3 - k % 4, k % 3 - 1], 1)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    got = np.frombuffer(_decode("oct", 12, 4, filter="OCTAHEDRAL"), "<i1").reshape(12, 4)[:, :3] / 127
    assert np.abs(got - unit).max() < 0.02
    got = np.frombuffer(_decode("oct16", 12, 8, filter="OCTAHEDRAL"), "<i2").reshape(12, 4)[:, :3] / 32767
    assert np.abs(got - unit).max() < 0.002
    quat = np.stack([k - 5.5, k % 5 - 2, np.full(12, 1.5), k % 3 - 0.5], 1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    got = np.frombuffer(_decode("quat", 12, 8, filter="QUATERNION"), "<i2").reshape(12, 4) / 32767
    assert (np.abs((got * quat).sum(1)) > 0.9999).all()  # q and -q are the same turn
    flt = np.stack([k * 1.5 - 7, k * k * 0.001, -1000 + k * 170], 1)
    got = np.frombuffer(_decode("exp", 12, 12, filter="EXPONENTIAL"), "<f4").reshape(12, 3)
    assert (np.abs(got - flt) <= np.abs(flt).max(1, keepdims=True) * 2.0 ** -13).all()  # one exponent per row


@pytest.mark.parametrize("name, cut, mode, words", [
    ("tri16", 100, "TRIANGLES", "ends early"), ("seq32", 120, "INDICES", "ends early"),
    ("v0", 0, "POINTS", "not ATTRIBUTES")])
def test_broken_meshopt_data_is_named(name, cut, mode, words):
    raw = bytes.fromhex(VECTORS[name])
    raw = raw[:cut] if cut else raw
    with pytest.raises(ValueError, match=words):
        meshopt.decode({"byteLength": len(raw), "byteStride": 4, "count": len(_indices()), "mode": mode}, raw)
    with pytest.raises(ValueError, match="not a0 or a1"):
        meshopt.decode({"byteLength": 4, "byteStride": 8, "count": 2, "mode": "ATTRIBUTES"}, b"\xff\0\0\0")


def _uri(b: bytes) -> str:
    return "data:application/octet-stream;base64," + base64.b64encode(b).decode()


def _file(path, doc):
    doc = dict({"asset": {"version": "2.0"}, "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}, **doc)
    path.write_text(json.dumps(doc))
    return str(path)


def _soup(path):
    """A glTF file's triangles as read, before they are cut for the depth sort: each its three corners from the
    lowest, in sorted order."""
    (P, _, T, _, _), = models._gltf(Path(path))[0]
    tri = P[T]
    first = np.array([min(range(3), key=lambda k: tuple(t[k])) for t in tri])
    tri = tri[np.arange(len(tri))[:, None], (first[:, None] + np.arange(3)) % 3].reshape(len(tri), 9)
    return tri[np.lexsort(tri.T[::-1])]


@pytest.mark.parametrize("name", meshopt.NAMES)
def test_a_gltf_with_meshopt_views_reads_as_it_would_uncompressed(tmp_path, name):
    P, idx = _grid(), _indices()
    plain = write_gltf(tmp_path / "plain.gltf", [(P, idx.reshape(-1, 3), None, None)])
    pos, tri = bytes.fromhex(VECTORS["grid"]), bytes.fromhex(VECTORS["tri16"])
    at = len(pos) + -len(pos) % 4
    packed = _file(tmp_path / "packed.gltf", {
        "extensionsUsed": [name], "extensionsRequired": [name],
        "buffers": [{"byteLength": at + len(tri), "uri": _uri(pos + b"\0" * (at - len(pos)) + tri)},
                    {"byteLength": 588 + 2 * len(idx), "extensions": {name: {"fallback": True}}}],
        "bufferViews": [
            {"buffer": 1, "byteLength": 588, "byteStride": 12, "extensions": {name: {
                "buffer": 0, "byteLength": len(pos), "byteStride": 12, "count": 49, "mode": "ATTRIBUTES"}}},
            {"buffer": 1, "byteOffset": 588, "byteLength": 2 * len(idx), "extensions": {name: {
                "buffer": 0, "byteOffset": at, "byteLength": len(tri), "byteStride": 2, "count": len(idx),
                "mode": "TRIANGLES"}}}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 49, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5123, "count": len(idx), "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}]})
    assert np.array_equal(_soup(packed), _soup(plain))
    res = check(scene([{"id": "m", "type": "model", "src": packed}]))
    assert not [i for i in res["issues"] if i["code"] == "bad-model"], res["issues"]


def _draco_file(tmp_path, DracoPy):
    """The grid with texture coordinates, Draco-compressed, and the same file uncompressed."""
    P, T = _grid().astype(float), _indices(extra=False).reshape(-1, 3)
    uv = P[:, :2] / 60
    raw = DracoPy.encode(P, T.astype(np.uint32), tex_coord=uv, preserve_order=True)
    uid = {a["attribute_type"]: a["unique_id"] for a in DracoPy.decode(raw).attributes}  # 0 position, 3 uv
    draco = "KHR_draco_mesh_compression"
    src = _file(tmp_path / "draco.gltf", {
        "extensionsUsed": [draco], "extensionsRequired": [draco],
        "buffers": [{"byteLength": len(raw), "uri": _uri(raw)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(raw)}],
        "accessors": [{"componentType": 5126, "count": 49, "type": "VEC3"},
                      {"componentType": 5126, "count": 49, "type": "VEC2"},
                      {"componentType": 5125, "count": T.size, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "extensions": {
            draco: {"bufferView": 0, "attributes": {"POSITION": uid[0], "TEXCOORD_0": uid[3]}}}}]}]})
    return src, write_gltf(tmp_path / "plain.gltf", [(P, T, None, None, uv)])


def test_draco_meshes_are_read_with_the_draco_extra(tmp_path):
    DracoPy = pytest.importorskip("DracoPy")
    src, plain = _draco_file(tmp_path, DracoPy)
    got, want = models.load(src), models.load(plain)
    assert np.array_equal(got.mesh.T, want.mesh.T)
    assert np.abs(got.mesh.P - want.mesh.P).max() < 0.02  # Draco keeps positions to 14 bits
    assert np.abs(got.mesh.U - want.mesh.U).max() < 0.002


def test_without_dracopy_a_draco_file_names_the_extra(tmp_path, monkeypatch):
    DracoPy = pytest.importorskip("DracoPy")
    src, _ = _draco_file(tmp_path, DracoPy)
    monkeypatch.setitem(sys.modules, "DracoPy", None)  # import DracoPy now fails
    res = check(scene([{"id": "m", "type": "model", "src": src}]))
    bad = [i for i in res["issues"] if i["code"] == "bad-model"]
    assert bad and "draco.gltf" in bad[0]["message"] and "kinescript[draco]" in bad[0]["message"]


def test_texture_transform_moves_the_texture_coordinates(tmp_path):
    """gltfpack stores texture coordinates quantized and gives their scale and offset back in
    KHR_texture_transform. Here they are also turned a quarter: stored (s, t) reads as (4 t + 0.25, 0.5 - 2 s)."""
    png = open(quadrants(tmp_path / "q.png"), "rb").read()
    P = [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    stored = np.stack([(0.5 - uv[:, 1]) / 2, (uv[:, 0] - 0.25) / 4], 1)
    move = {"offset": [0.25, 0.5], "rotation": math.pi / 2, "scale": [2, 4]}
    mat = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "extensions": {"KHR_texture_transform": move}},
                                    "metallicFactor": 0, "roughnessFactor": 1}}
    src = write_gltf(tmp_path / "quad.glb", [(P, [[0, 3, 2], [0, 2, 1]], [[0, 0, 1]] * 4, 0, stored)],
                     materials=[mat], binary=True, images=[png])
    U = models.load(src).mesh.U
    assert np.allclose(U.min(0), [0, 0]) and np.allclose(U.max(0), [1, 1])
    f = frame(scene([{"id": "m", "type": "model", "src": src, "w": 160, "x": 160, "y": 90, "light": "flat"}]))
    assert near(f[50, 120], RED) and near(f[50, 200], GREEN) and near(f[130, 120], BLUE) and near(f[130, 200], YELLOW)
