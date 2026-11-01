"""Textures on models: coordinates on every shape, image files on layers and parts, a model file's own textures,
and the image staying put under perspective."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript import mesh
from kinescript.render import Renderer
from kinescript.verify import check
from test_models import write_gltf

RED, GREEN, BLUE, YELLOW = (230, 40, 40), (40, 200, 60), (40, 70, 230), (240, 220, 40)


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 30, "duration": 1, "bg": "#000000"}, **canvas),
            "layers": layers}


def quadrants(path, size=64, clear=False) -> str:
    """An image red at the top left, green at the top right, blue at the bottom left and yellow at the bottom
    right (or clear there)."""
    h = size // 2
    px = np.zeros((size, size, 4), np.uint8)
    px[..., 3] = 255
    px[:h, :h, :3], px[:h, h:, :3], px[h:, :h, :3], px[h:, h:, :3] = RED, GREEN, BLUE, YELLOW
    if clear:
        px[h:, h:] = 0
    skia.Image.fromarray(px, skia.kRGBA_8888_ColorType).save(str(path))
    return str(path)


def frame(spec, n=0, backend="cpu"):
    return Renderer(k.Scene(spec), backend=backend).frame(n)[..., :3].astype(int)


def card(texture, **kw):
    """A flat-lit plane facing the camera over x 80..240, y 10..170."""
    return dict({"id": "card", "type": "model", "shape": "plane", "w": 160, "x": 160, "y": 90, "light": "flat",
                 "texture": texture}, **kw)


def near(px, colour, tol=8):
    return bool(np.abs(np.asarray(px) - colour).max() <= tol)


def test_every_shape_has_texture_coordinates_and_faces_read_upright():
    for shape in mesh.SHAPES:
        m = mesh.unit(shape)
        assert m.U is not None and m.U.shape == (len(m.P), 2), shape
    cube = mesh.unit("cube")
    front = cube.N[:, 2] > 0.5
    top_left = front & np.isclose(cube.P[:, 0], -0.5) & np.isclose(cube.P[:, 1], -0.5)
    assert np.allclose(cube.U[top_left], [0, 0])  # the image's top left at the face's top left
    top = cube.N[:, 1] < -0.5  # y is down: this face looks up
    far_left = top & np.isclose(cube.P[:, 0], -0.5) & np.isclose(cube.P[:, 2], -0.5)
    assert np.allclose(cube.U[far_left], [0, 0])  # on top, the image's top is away from the camera
    ball = mesh.unit("sphere")
    front = np.argmax(ball.P[:, 2] - np.abs(ball.P[:, 1]))  # on the equator, facing the camera
    assert np.allclose(ball.U[front], [0.5, 0.5])
    text = mesh.extrude(*mesh.shape2d([np.array([[0, 0], [40, 0], [40, 20], [0, 20]], float)]), 10, 0, 1, 50)
    assert np.allclose(text.U.min(0), 0) and np.allclose(text.U.max(0), 1)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_texture_shows_on_a_model_the_right_way_up(tmp_path, backend):
    img = quadrants(tmp_path / "q.png")
    try:
        f = frame(scene([card(img)]), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    assert near(f[50, 120], RED) and near(f[50, 200], GREEN) and near(f[130, 120], BLUE) and near(f[130, 200], YELLOW)


def test_repeat_and_offset_move_the_image_and_fill_shows_through_clear_pixels(tmp_path):
    img = quadrants(tmp_path / "q.png")
    f = frame(scene([card({"src": img, "repeat": 2})]))
    assert near(f[30, 100], RED) and near(f[30, 140], GREEN) and near(f[30, 180], RED) and near(f[70, 100], BLUE)
    f = frame(scene([card({"src": img, "offset": [0.5, 0]})]))  # half an image to the right
    assert near(f[50, 120], GREEN) and near(f[50, 200], RED)
    f = frame(scene([card(quadrants(tmp_path / "c.png", clear=True), fill="#ff00ff")]))
    assert near(f[130, 200], (255, 0, 255)) and near(f[50, 120], RED)


def test_a_part_takes_its_own_texture_and_the_others_their_fill(tmp_path):
    img = quadrants(tmp_path / "q.png")
    model = {"id": "m", "type": "model", "x": 160, "y": 90, "light": "flat", "fill": "#ffffff", "parts": [
        {"shape": "plane", "w": 100, "x": -60, "texture": img}, {"shape": "plane", "w": 100, "x": 60}]}
    f = frame(scene([model]))
    assert near(f[60, 80], RED) and near(f[120, 120], YELLOW) and near(f[90, 220], (255, 255, 255))
    model = dict(model, texture=img, parts=[{"shape": "plane", "w": 100, "x": -60},  # the layer's texture
                                            {"shape": "plane", "w": 100, "x": 60, "texture": False}])  # its fill
    f = frame(scene([model]))
    assert near(f[60, 80], RED) and near(f[120, 120], YELLOW) and near(f[90, 220], (255, 255, 255))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_perspective_does_not_slide_the_image(tmp_path, backend):
    """A plane tipped far back, with few triangles: the line between the image's halves stays on the plane's
    middle, which is where the layer is, as Skia's even spread across each triangle would not put it."""
    px = np.zeros((64, 64, 4), np.uint8)
    px[..., 3], px[:32, :, :3] = 255, 255
    skia.Image.fromarray(px, skia.kRGBA_8888_ColorType).save(str(tmp_path / "half.png"))
    spec = scene([card(str(tmp_path / "half.png"), w=360, detail="low", rotateX=75)])
    try:
        col = frame(spec, backend=backend)[:, 160, 0]
    except RuntimeError:
        pytest.skip("no GPU")
    middle = np.flatnonzero(col > 128).max() + 0.5  # the white half is the far one, above
    assert abs(middle - 90) <= 1.0, middle  # 3.5 px off without cutting the triangles


def test_a_model_files_own_texture_and_coordinates_are_drawn(tmp_path):
    png = open(quadrants(tmp_path / "q.png"), "rb").read()
    P = [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]  # y up: top left, top right, bottom right, bottom left
    uv = [[0, 0], [1, 0], [1, 1], [0, 1]]  # glTF images start at the top left
    mat = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1}}
    src = write_gltf(tmp_path / "quad.glb", [(P, [[0, 3, 2], [0, 2, 1]], [[0, 0, 1]] * 4, 0, uv)],
                     materials=[mat], binary=True, images=[png])
    layer = {"id": "m", "type": "model", "src": src, "w": 160, "x": 160, "y": 90, "light": "flat"}
    f = frame(scene([layer]))
    assert near(f[50, 120], RED) and near(f[50, 200], GREEN) and near(f[130, 120], BLUE) and near(f[130, 200], YELLOW)
    f = frame(scene([dict(layer, texture=False)]))  # its mean colour, as before textures were drawn
    assert np.abs(f[50, 120] - f[130, 200]).max() <= 2 and f[50, 120].min() > 60
    (tmp_path / "quad.obj").write_text("mtllib quad.mtl\nv -1 1 0\nv 1 1 0\nv 1 -1 0\nv -1 -1 0\n"
                                       "vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nusemtl q\nf 1/1 4/4 3/3 2/2\n")
    (tmp_path / "quad.mtl").write_text("newmtl q\nKd 1 1 1\nmap_Kd q.png\n")  # .obj images start at the bottom
    f = frame(scene([dict(layer, src=str(tmp_path / "quad.obj"))]))
    assert near(f[50, 120], RED) and near(f[50, 200], GREEN) and near(f[130, 120], BLUE) and near(f[130, 200], YELLOW)


def test_textured_frames_are_pure_functions_of_their_index(tmp_path):
    img = quadrants(tmp_path / "q.png")
    spec = scene([card(img, rotateY=[[0, -60], ["end", 60]], rotateX=-30, light="studio", shine=0.8)])
    spec["fog"] = {"near": 0, "far": 3000}
    r = Renderer(k.Scene(spec), backend="cpu")
    seq = [r.frame(n).copy() for n in range(0, 30, 6)]
    for n, want in zip(range(24, -1, -6), seq[::-1]):
        assert (Renderer(k.Scene(spec), backend="cpu").frame(n) == want).all()


def test_texture_keys_files_and_coordinates_are_checked(tmp_path):
    img = quadrants(tmp_path / "q.png")

    def codes(layer):
        return [(i["level"], i["code"]) for i in check(scene([layer]))["issues"]]
    assert codes(card(img)) == []
    renamed = {k: v for k, v in card(None).items() if k != "texture"}
    assert ("info", "renamed-key") in codes(dict(renamed, map={"image": img, "tiles": 3}))
    for bad in ({"src": img, "repeat": 0}, {"src": img, "offset": "left"}, {"repeat": 2}, "clip.mp4", 3):
        assert any(lv == "error" for lv, _ in codes(card(bad))), bad
    assert ("error", "media-missing") in codes(card(str(tmp_path / "gone.png")))
    assert ("error", "media-missing") in codes({"id": "m", "type": "model", "parts": [
        {"shape": "cube", "texture": str(tmp_path / "gone.png")}]})
    issues = check(scene([card(img, w=3000, rotateX=85, y=150, z=150)]))["issues"]  # a floor past the camera
    assert {"texture-swim", "behind-camera"} <= {i["code"] for i in issues}
    plain = write_gltf(tmp_path / "plain.glb", [([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], None, None)],
                       binary=True)
    assert ("warning", "no-uvs") in codes({"id": "m", "type": "model", "src": plain, "texture": img})
