"""SVG files in image layers: drawn as vectors at any size, sized by their width, height and viewBox as a browser
sizes them, and as pixels where a texture, a sheet's cell or a particle needs an image. kine check names the elements
skia's svg reader leaves out."""
import numpy as np
import pytest

import kinescript as k
from kinescript import media
from kinescript.render import Renderer
from kinescript.verify import probe_media

SVG = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" {}>{}</svg>'
HALVES = '<rect width="2" height="2" fill="#ff0000"/><rect x="2" width="2" height="2" fill="#0000ff"/>'


def svg(path, body, attrs='width="40" height="20" viewBox="0 0 4 2"'):
    path.write_text(SVG.format(attrs, body))
    return str(path)


def scene(layers, **canvas):
    return {"canvas": dict({"w": 640, "h": 360, "fps": 10, "duration": 1, "bg": "#000000"}, **canvas),
            "layers": layers}


def frame(spec, base, backend="cpu", i=0, track=False):
    try:
        r = Renderer(k.Scene(spec, base_dir=base), backend=backend, track=track)
    except RuntimeError:
        pytest.skip("no GPU")
    return r.frame(i)[:, :, :3].astype(int), r


def codes(spec, base):
    try:
        s = k.Scene(spec, base_dir=base)
        return s.issues, s
    except k.SceneError as e:
        return e.issues, None


def test_its_size_is_its_width_and_height_else_its_viewbox(tmp_path):
    cases = [('width="2in" height="1in"', (192, 96)),  # 96 px to the inch
             ('width="30mm" height="12pt"', (30 * 96 / 25.4, 16)),
             ('viewBox="0 0 50 25"', (50, 25)),
             ('width="100" viewBox="0,0,50,25"', (100, 50)),  # the missing one from the viewBox's shape
             ('height="100px" viewBox="0 0 50 25"', (200, 100)),
             ('width="100%" height="100%" viewBox="0 0 50 25"', (50, 25)),  # a percentage is no size
             ('width="0" height="20" viewBox="0 0 50 25"', (40, 20)),
             ('width="-10" height="20" viewBox="0 0 50 25"', (40, 20)),
             ('', (300, 150))]  # as a browser shows an svg with neither
    for i, (attrs, want) in enumerate(cases):
        v = media.vector(svg(tmp_path / f"s{i}.svg", HALVES, attrs))
        assert (v.width(), v.height()) == pytest.approx(want), attrs


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_an_svg_draws_as_vectors_sharp_at_any_size(tmp_path, backend):
    svg(tmp_path / "halves.svg", HALVES)
    img, r = frame(scene([{"id": "v", "type": "image", "src": "halves.svg", "x": 320, "y": 180}]), tmp_path, backend,
                   track=True)
    assert r.boxes["v"]["rect"] == [300, 170, 340, 190]  # its own size, 40 x 20
    assert img[180, 310].tolist() == [255, 0, 0] and img[180, 330].tolist() == [0, 0, 255]
    img, _ = frame(scene([{"id": "v", "type": "image", "src": "halves.svg", "x": 320, "y": 180, "w": 600}]), tmp_path,
                   backend)
    row = img[180, 30:610]
    red, blue = (row == [255, 0, 0]).all(axis=1), (row == [0, 0, 255]).all(axis=1)
    assert red[:280].all() and blue[300:].all()
    assert (~(red | blue)).sum() <= 1  # 15 times its size, and the edge between the halves is a pixel at most
    flat = svg(tmp_path / "flat.svg", HALVES, 'width="40" height="20" viewBox="0 0 4 2" preserveAspectRatio="none"')
    img, _ = frame(scene([{"id": "v", "type": "image", "src": flat, "x": 320, "y": 180, "w": 200, "h": 200,
                           "fit": "stretch"}]), tmp_path, backend)
    assert img[85, 300].tolist() == [255, 0, 0] and img[275, 340].tolist() == [0, 0, 255]  # stretched both ways


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_faded_svg_fades_as_one_drawing(tmp_path, backend):
    svg(tmp_path / "two.svg", '<rect width="30" height="20" fill="#ff0000"/><rect x="10" width="30" height="20" '
        'fill="#ff0000"/>')
    img, _ = frame(scene([{"id": "v", "type": "image", "src": "two.svg", "x": 320, "y": 180, "w": 400,
                           "opacity": 0.5}]), tmp_path, backend)
    alone, both = img[180, 150], img[180, 320]
    assert 120 <= alone[0] <= 136 and abs(int(both[0]) - int(alone[0])) <= 1  # the overlap no darker


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_what_lies_outside_its_viewport_is_not_shown(tmp_path, backend):
    svg(tmp_path / "spill.svg", '<rect x="-50" y="-50" width="200" height="200" fill="#ffffff"/>',
        'width="20" height="20"')
    img, _ = frame(scene([{"id": "v", "type": "image", "src": "spill.svg", "x": 320, "y": 180, "w": 300, "h": 100,
                           "fit": "contain"}]), tmp_path, backend)
    assert img[180, 320].tolist() == [255, 255, 255] and img[180, 230].tolist() == [0, 0, 0]


def test_svg_2_href_is_followed(tmp_path):
    # skia's reader follows xlink:href only; SVG 2 writes href on a <use> and on a gradient taking another's stops
    svg(tmp_path / "use.svg", '<defs><rect id="r" width="2" height="2" fill="#00ff00"/>'
        '<linearGradient id="g0"><stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#ff0000"/>'
        '</linearGradient><linearGradient id="g" href="#g0"/></defs>'
        '<use href="#r"/><rect x="2" width="2" height="2" fill="url(#g)"/>')
    img, _ = frame(scene([{"id": "v", "type": "image", "src": "use.svg", "x": 320, "y": 180, "w": 400}]), tmp_path)
    assert img[180, 220].tolist() == [0, 255, 0] and img[180, 420].tolist() == [255, 0, 0]


def test_check_names_the_elements_that_are_not_drawn(tmp_path):
    svg(tmp_path / "text.svg", HALVES + '<style>rect{fill:red}</style><text x="0" y="2">A<tspan>B</tspan></text>'
        '<animate attributeName="x" to="5" dur="1s"/>')
    issues, s = codes(scene([{"id": "v", "type": "image", "src": "text.svg"}]), tmp_path)
    (w,) = [i for i in issues if i["code"] == "svg-unsupported"]
    assert s is not None and w["level"] == "warning" and w["layer"] == "v"
    assert "text.svg has <text>, <style>, <animate>" in w["message"] and "<tspan>" not in w["message"]
    assert w["fix"].startswith("turn text into paths") and "does not play" in w["fix"]
    assert probe_media(str(tmp_path / "text.svg"))["undrawn"] == ["text", "style", "animation"]
    svg(tmp_path / "clean.svg", HALVES)
    issues, _ = codes(scene([{"id": "v", "type": "image", "src": "clean.svg"}]), tmp_path)
    assert "svg-unsupported" not in [i["code"] for i in issues]
    about = probe_media(str(tmp_path / "clean.svg"))
    assert (about["kind"], about["w"], about["h"], about["undrawn"]) == ("image", 40, 20, []) and "note" not in about


def test_an_svg_that_cannot_be_read_fails_at_load(tmp_path):
    (tmp_path / "bad.svg").write_text("not svg")
    (tmp_path / "html.svg").write_text("<html><body/></html>")
    for name in ("bad.svg", "html.svg"):
        issues, s = codes(scene([{"id": "v", "type": "image", "src": name}]), tmp_path)
        (e,) = [i for i in issues if i["code"] == "bad-media"]
        assert s is None and e["message"].startswith("image src cannot read svg") and name in e["message"]
    issues, _ = codes(scene([{"id": "m", "type": "model", "shape": "cube", "texture": "bad.svg"}]), tmp_path)
    assert [i["message"][:24] for i in issues if i["code"] == "bad-media"] == ["texture cannot read svg "]
    with pytest.raises(media.MediaError, match="it is <html>, not <svg>"):
        media.vector(str(tmp_path / "html.svg"))


def test_as_pixels_it_is_drawn_at_a_whole_multiple_of_its_size(tmp_path):
    img = media.image(svg(tmp_path / "small.svg", HALVES))
    assert (img.width(), img.height()) == (520, 260)  # 13 times: its longer side to 512 or more
    a = img.toarray()
    assert a[130, 100, 3] == a[130, 420, 3] == 255 and a[130, 100, :3].tolist() != a[130, 420, :3].tolist()
    big = media.image(svg(tmp_path / "big.svg", HALVES, 'width="8000" height="100" viewBox="0 0 4 2"'))
    assert (big.width(), big.height()) == (4096, 51)  # no more than 4096 across
    same = media.image(svg(tmp_path / "even.svg", HALVES, 'width="600" height="300" viewBox="0 0 4 2"'))
    assert (same.width(), same.height()) == (600, 300)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_sheet_frames_texture_and_sprite_take_svg(tmp_path, backend):
    cells = "".join(f'<rect x="{i * 2}" width="2" height="2" fill="{c}"/>'
                    for i, c in enumerate(("#ff0000", "#00ff00", "#0000ff")))
    svg(tmp_path / "strip.svg", cells, 'width="60" height="20" viewBox="0 0 6 2"')
    spec = scene([{"id": "s", "sheet": {"src": "strip.svg", "cols": 3}, "fps": 10, "x": 320, "y": 180, "w": 200}])
    got = [frame(spec, tmp_path, backend, i)[0][180, 320].tolist() for i in range(4)]
    assert got == [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]]  # a cell at a time, its neighbours cut off
    spec["layers"][0]["fit"] = "contain"  # no box clip
    img, _ = frame(spec, tmp_path, backend, 1)
    assert img[180, 225].tolist() == [0, 255, 0] and img[180, 415].tolist() == [0, 255, 0]
    assert img[180, 200].tolist() == [0, 0, 0] and img[180, 440].tolist() == [0, 0, 0]
    (tmp_path / "run").mkdir()
    for i, c in enumerate(("#ff0000", "#0000ff")):
        svg(tmp_path / "run" / f"run_{i + 1}.svg", f'<rect width="4" height="2" fill="{c}"/>')
    spec = scene([{"id": "f", "type": "image", "frames": "run", "fps": 10, "x": 320, "y": 180}])
    assert [frame(spec, tmp_path, backend, i)[0][180, 320].tolist() for i in range(2)] == [[255, 0, 0], [0, 0, 255]]
    spec = scene([{"id": "p", "type": "particles", "src": "halves.svg", "burst": 1, "speed": 0, "size": 80,
                   "x": 320, "y": 180}])
    svg(tmp_path / "halves.svg", HALVES)
    img, _ = frame(spec, tmp_path, backend)
    assert img[175:185, 290:350].max() > 200  # the sprite, from the svg's pixels
    spec = scene([{"id": "m", "type": "model", "shape": "plane", "texture": "halves.svg", "w": 300, "x": 320,
                   "y": 180, "light": "flat"}])
    img, _ = frame(spec, tmp_path, backend)
    assert np.abs(img[180, 250] - [255, 0, 0]).max() <= 4 and np.abs(img[180, 390] - [0, 0, 255]).max() <= 4


def test_a_sequence_clip_with_an_svg_is_an_image(tmp_path):
    svg(tmp_path / "card.svg", HALVES)
    s = k.Scene(scene([{"id": "cut", "type": "sequence", "clips": [{"src": "card.svg", "dur": 1}]}]),
                base_dir=tmp_path)
    assert s.by_id["cut-1"].type == "image"
