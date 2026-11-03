"""The svg layer: an svg file as a group of path layers, one for each shape in it, named from the file's ids, so each
piece draws on, enters and moves as any layer does. kine check names what it leaves out and parts that name no piece.
A group whose layers stand still is recorded once into a picture and played back, to the same bytes."""
import numpy as np
import pytest

import kinescript as k
from kinescript import svgdoc, svgpath
from kinescript.render import Renderer

SVG = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" {}>{}</svg>'
BOX = 'width="100" height="100" viewBox="0 0 100 100"'
# Most of what the svg layer reads: transforms, a nested viewport, use, gradients in both units and through href,
# strokes with caps, joins and dashes, opacity on a group, an even-odd hole, colours written four ways, arcs.
MIX = """
<defs>
  <linearGradient id="lg" x1="0" y1="0" x2="400" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="red"/><stop offset="100%" style="stop-color:rgb(0,0,255);stop-opacity:0.8"/>
  </linearGradient>
  <linearGradient id="lg2" xlink:href="#lg" x1="0" y1="0" x2="0" y2="1" gradientUnits="objectBoundingBox"/>
  <radialGradient id="rg" cx="0.4" cy="0.4" r="0.6"><stop offset="0" stop-color="#fff"/>
    <stop offset="1" stop-color="#36c"/></radialGradient>
  <rect id="tile" width="18" height="18" rx="3" fill="url(#rg)"/>
</defs>
<g transform="translate(20 20) rotate(10)" fill="url(#lg)" stroke="navy" stroke-width="4">
  <rect width="160" height="60" rx="12"/>
  <ellipse cx="260" cy="40" rx="50" ry="30" fill="#3c3"/>
</g>
<path d="M20 160 h100 v30 h-100 z M45 168 h50 v14 h-50 z" fill="#fc0" fill-rule="evenodd"/>
<polyline points="160,190 200,140 240,190 280,140" fill="none" stroke="currentColor" color="teal" stroke-width="6"
          stroke-linecap="square" stroke-linejoin="bevel"/>
<circle cx="350" cy="160" r="30" fill="url(#lg2)" opacity="0.7"/>
<line x1="300" y1="110" x2="390" y2="110" stroke="black" stroke-width="3" stroke-dasharray="10 5"/>
<g opacity="0.5"><rect x="300" y="120" width="40" height="40" fill="purple"/>
  <rect x="320" y="130" width="40" height="40" fill="purple"/></g>
<path d="M150 100 a20 20 0 1 0 40 0a20 20 0 1 0-40 0z" fill="#e33"/>
<g transform="matrix(1 0 0.3 1 0 0) scale(1.2)"><polygon points="250,70 270,90 230,90" fill="#f2a516"/></g>
<svg x="120" y="20" width="60" height="30" viewBox="0 0 20 10"><circle cx="10" cy="5" r="5" fill="#a0f"/></svg>
<use href="#tile" x="10" y="100"/><use href="#tile" x="34" y="100" transform="rotate(15 43 109)"/>
"""


def svg(path, body, attrs=BOX):
    path.write_text(SVG.format(attrs, body))
    return str(path)


def scene(layers, **canvas):
    return {"canvas": dict({"w": 400, "h": 300, "fps": 10, "duration": 1, "bg": "#000000"}, **canvas),
            "layers": layers}


def renderer(spec, base, backend="cpu", track=False):
    try:
        return Renderer(k.Scene(spec, base_dir=base), backend=backend, track=track)
    except RuntimeError:
        pytest.skip("no GPU")


def rgb(r, i=0):
    return r.frame(i)[:, :, :3].astype(int)


def issues(spec, base):
    try:
        return k.Scene(spec, base_dir=base).issues
    except k.SceneError as e:
        return e.issues


def ids(L):
    return [c.id for c in L.children]


def test_each_shape_becomes_a_path_layer_named_from_its_id(tmp_path):
    svg(tmp_path / "a.svg", '<rect id="box" x="10" y="10" width="30" height="20" fill="red"/>'
        '<circle cx="70" cy="70" r="10" fill="#00f"/><g id="pair"><rect x="50" y="10" width="10" height="10" '
        'fill="lime"/><rect x="70" y="10" width="10" height="10" fill="lime"/></g><rect id="x y" width="1" '
        'height="1" fill="#fff"/><rect id="x-y" width="1" height="1" fill="#fff"/>')
    s = k.Scene(scene([{"id": "s", "type": "svg", "src": "a.svg"}]), base_dir=tmp_path)
    (L,) = s.layers
    assert L.type == "group" and ids(L) == ["s-box", "s-1", "s-pair", "s-x-y", "s-x-y-2"]
    assert ids(L.children[2]) == ["s-2", "s-3"]
    box = L.children[0]
    assert box.type == "path" and box.raw["fill"] == "#ff0000"  # svg's red, not the theme's
    img = rgb(Renderer(s, backend="cpu"))
    assert img[20, 25].tolist() == [255, 0, 0] and img[70, 70].tolist() == [0, 0, 255]
    assert img[15, 55].tolist() == [0, 255, 0] and img[15, 75].tolist() == [0, 255, 0]
    assert img[50, 50].tolist() == [0, 0, 0]


def test_it_draws_as_the_image_layer_draws_the_same_file(tmp_path):
    svg(tmp_path / "mix.svg", MIX, 'width="200" height="100" viewBox="0 0 400 200"')
    r = renderer(scene([{"id": "a", "type": "svg", "src": "mix.svg", "x": 200, "y": 100, "w": 380},
                        {"id": "b", "type": "image", "src": "mix.svg", "x": 200, "y": 300, "w": 380}], h=400), tmp_path)
    assert not r.s.issues
    img = rgb(r)
    d = np.abs(img[:200] - img[200:]).max(axis=2)
    assert d.mean() < 0.4 and (d > 64).sum() < 60, (d.mean(), (d > 64).sum())


def test_svg_colours():
    c = svgdoc.colour
    assert c("red", None) == "#ff0000" and c("RebeccaPurple", None) == "#663399"
    assert c("#abc", None) == "#aabbcc" and c("#abcd", None) == "#aabbccdd" and c("#AaBbCc80", None) == "#aabbcc80"
    assert c("rgb(255, 0, 0)", None) == "#ff0000" and c("rgb(100%,0%,0%)", None) == "#ff0000"
    assert c("rgba(0,0,255,0.5)", None) == "#0000ff80" and c("rgb(0 0 255 / 50%)", None) == "#0000ff80"
    assert c("hsl(120, 100%, 50%)", None) == "#00ff00" and c("hsla(240deg 100% 50% / 0.5)", None) == "#0000ff80"
    assert c("currentColor", "blue") == "#0000ff" and c("currentColor", None) == "#000000"
    assert c("red", None, 0.5) == "#ff000080"  # its fill-opacity
    assert c("none", None) is None and c("transparent", None) is None and c("rgb(1,2)", None) is None


def test_strokes_keep_the_files_width_caps_joins_and_dashes(tmp_path):
    svg(tmp_path / "s.svg", '<g transform="scale(2)"><path id="p" d="M10 10 L40 10" stroke="#fff" stroke-width="3" '
        'stroke-dasharray="4 2 1" stroke-dashoffset="1" fill="none"/><path id="q" d="M10 20 L40 20 L40 40" '
        'stroke="#fff" '
        'stroke-linecap="round" stroke-linejoin="round" fill="none"/></g>')
    (L,) = k.Scene(scene([{"id": "s", "type": "svg", "src": "s.svg"}]), base_dir=tmp_path).layers
    p, q = (c.raw for c in L.children)
    assert "fill" not in p and p["strokeWidth"] == 6 and p["cap"] == "butt" and p["join"] == "miter"
    assert p["dash"] == [8, 4, 2, 8, 4, 2] and p["dashOffset"] == 2  # an odd list repeats, both scaled with it
    assert "cap" not in q and "join" not in q and q["strokeWidth"] == 2  # a path layer's own are round


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_an_even_odd_fill_leaves_holes(tmp_path, backend):
    rings = ('<path id="{}" d="M{} 10 h40 v40 h-40 z M{} 20 h20 v20 h-20 z" fill="#fff" fill-rule="{}"/>')
    svg(tmp_path / "e.svg", rings.format("eo", 10, 20, "evenodd") + rings.format("nz", 60, 70, "nonzero"))
    r = renderer(scene([{"id": "s", "type": "svg", "src": "e.svg"}]), tmp_path, backend)
    eo, nz = (c.raw for c in r.s.layers[0].children)
    assert eo["fillRule"] == "evenodd" and "fillRule" not in nz
    img = rgb(r)
    assert img[30, 30].tolist() == [0, 0, 0] and img[15, 15].tolist() == [255, 255, 255]  # a hole in the middle
    assert img[30, 80].tolist() == [255, 255, 255]  # both squares wind the same way: no hole
    spec = scene([{"id": "p", "type": "path", "d": "M10 10 h40 v40 h-40 z M20 20 h20 v20 h-20 z", "fill": "#fff",
                   "fillRule": "even-odd"}])
    img = rgb(renderer(spec, tmp_path, backend))
    assert img[30, 30].tolist() == [0, 0, 0] and img[15, 15].tolist() == [255, 255, 255]
    spec["layers"][0]["fillRule"] = "winding"
    assert [(i["code"], i["message"]) for i in issues(spec, tmp_path)] == [
        ("renamed-value", "fillRule 'winding' read as 'nonzero'")]


def test_gradients_map_to_the_box_of_each_piece(tmp_path):
    svg(tmp_path / "g.svg", '<defs><linearGradient id="a"><stop offset="0" stop-color="#f00"/><stop offset="1" '
        'stop-color="#00f"/></linearGradient><linearGradient id="u" gradientUnits="userSpaceOnUse" x1="0" x2="100" '
        'href="#a"/><radialGradient id="r" r="0.5"><stop offset="20%" stop-color="#fff"/><stop offset="0.1" '
        'stop-color="#000"/></radialGradient><linearGradient id="t" href="#a" gradientTransform="rotate(90 .5 .5)"/>'
        '</defs><rect id="obb" x="0" y="0" width="50" height="20" '
        'fill="url(#a)"/><rect id="user" x="50" y="40" width="50" height="20" fill="url(#u)"/><rect id="rad" x="0" '
        'y="60" width="40" height="40" fill="url(#r)"/><rect id="miss" width="5" height="5" '
        'fill="url(#nothing) #0f0"/><rect id="turned" x="60" width="10" height="30" fill="url(#t)"/>')
    r = renderer(scene([{"id": "s", "type": "svg", "src": "g.svg"}]), tmp_path)
    obb, user, rad, miss, turned = (c.raw["fill"] for c in r.s.layers[0].children)
    assert obb == {"linear": [[0, 0], [1, 0]], "stops": [[0, "#ff0000"], [1, "#0000ff"]]}
    assert user["linear"] == [[-1, -2], [1, -2]]  # in the file's units
    assert user["stops"] == obb["stops"]  # its stops by href
    assert rad == {"radial": [0.5, 0.5, 0.5], "stops": [[0.2, "#ffffff"], [0.2, "#000000"]]}  # offsets never go back
    assert miss == "#00ff00"  # the colour after a url that leads nowhere
    assert turned["linear"] == [[1, 0], [1, 1]]  # its gradientTransform turns it top to bottom
    img = rgb(r)
    assert img[10, 2, 0] > 240 and img[10, 47, 2] > 240  # red at the left of its own box, blue at the right
    assert img[50, 52].tolist()[0] in range(100, 156)  # the user-space gradient is half way there


def test_parts_change_pieces_by_id_and_star_changes_every_shape(tmp_path):
    svg(tmp_path / "a.svg", '<rect id="box" width="30" height="20" fill="red"/><g id="pair"><rect x="50" '
        'width="10" height="10"/><rect x="70" width="10" height="10"/></g>')
    spec = scene([{"id": "s", "type": "svg", "src": "a.svg", "parts": {
        "box": {"fill": "#00ff00", "draw": [[0, 0], [1, 1]]}, "*": {"stroke": "#ffffff", "strokeWidth": 2},
        "pair": {"enter": {"name": "fade", "stagger": 0.1}}}}])
    s = k.Scene(spec, base_dir=tmp_path)
    assert not s.issues
    box, pair = s.layers[0].children
    assert box.raw["fill"] == "#00ff00" and not box.props["draw"].static
    assert all(c.raw["stroke"] == "#ffffff" and c.raw["strokeWidth"] == 2 for c in [box, *pair.children])
    assert pair.enter is not None and [c.turn for c in pair.children] == pytest.approx([0, 0.1])


def test_check_reports_bad_svg_layers(tmp_path):
    svg(tmp_path / "a.svg", '<rect id="box" width="30" height="20"/><rect id="bar" width="3" height="2"/>')
    (tmp_path / "not.svg").write_text("<html><body/></html>")
    (tmp_path / "broken.svg").write_text("<svg")

    def got(**layer):
        return [(i["level"], i["code"]) for i in issues(scene([dict({"id": "s", "type": "svg"}, **layer)]), tmp_path)]
    assert got(src="a.svg") == []
    assert ("error", "unknown-part") in got(src="a.svg", parts={"bxo": {"fill": "#fff"}})
    fix = [i["fix"] for i in issues(scene([{"id": "s", "type": "svg", "src": "a.svg", "parts": {"bxo": {}}}]),
                                    tmp_path) if i["code"] == "unknown-part"]
    assert fix and "did you mean 'box'?" in fix[0] and "bar" in fix[0]
    assert ("error", "bad-key") in got(src="a.svg", parts={"box": {"d": "M0 0 L1 1"}})
    assert ("error", "bad-value") in got(src="a.svg", parts={"box": {"opacity": "lots"}})
    assert ("error", "bad-value") in got(src="a.svg", parts=["box"])
    assert ("error", "bad-value") in got(src="a.png")
    assert ("error", "bad-value") in got(src="a.svg", w=0)
    assert ("error", "bad-value") in got(src="a.svg", w=[[0, 10], [1, 20]])
    assert ("error", "missing-key") in got()
    assert ("error", "media-missing") in got(src="gone.svg")
    assert ("error", "bad-media") in got(src="not.svg")
    assert ("error", "bad-media") in got(src="broken.svg")


def test_check_names_what_the_svg_layer_leaves_out(tmp_path):
    svg(tmp_path / "a.svg", '<style>rect{fill:red}</style><defs><clipPath id="c"><rect width="5" height="5"/>'
        '</clipPath><pattern id="p" width="4" height="4"/></defs><text x="5" y="5">Hi<tspan>!</tspan></text>'
        '<image href="x.png" width="5" height="5"/><rect width="9" height="9" clip-path="url(#c)"/>'
        '<rect width="9" height="9" fill="url(#p)"/><path d="M0 0 L9 9" stroke="#fff" marker-end="url(#m)"/>'
        '<circle r="3"><animate attributeName="r" to="5" dur="1s"/></circle><foreignObject/>')
    warn = [i for i in issues(scene([{"id": "s", "type": "svg", "src": "a.svg"}]), tmp_path)
            if i["code"] == "svg-unsupported"]
    assert len(warn) == 1 and warn[0]["level"] == "warning"
    for name in ("<text>", "<style>", "<image>", "clip-path", "pattern", "marker", "<animate>", "<foreignObject>"):
        assert name in warn[0]["message"], name
    assert "Object to Path" in warn[0]["fix"] and warn[0]["fix"].count("an image layer with the same src") == 1


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_its_box_is_w_and_h_and_the_viewbox_fits_inside(tmp_path, backend):
    svg(tmp_path / "h.svg", '<rect width="2" height="2" fill="#f00"/><rect x="2" width="2" height="2" fill="#00f"/>',
        'viewBox="0 0 4 2"')
    r = renderer(scene([{"id": "s", "type": "svg", "src": "h.svg", "w": 200}]), tmp_path, backend, track=True)
    img = rgb(r)
    assert r.boxes["s"]["rect"] == [0, 0, 200, 100]  # the file's shape, from its top left without an x
    assert r.boxes["s-2"]["rect"] == [100, 0, 200, 100]  # boxes draw the pieces one by one
    assert img[50, 98].tolist() == [255, 0, 0] and img[50, 102].tolist() == [0, 0, 255]
    r = renderer(scene([{"id": "s", "type": "svg", "src": "h.svg", "x": 200, "y": 150, "w": 100, "h": 100}]),
                 tmp_path, backend, track=True)
    img = rgb(r)
    assert r.boxes["s"]["rect"] == [150, 100, 250, 200]  # at its middle once it has an x
    assert r.boxes["s-1"]["rect"] == [150, 125, 200, 175]
    assert img[150, 160].tolist() == [255, 0, 0] and img[110, 200].tolist() == [0, 0, 0]  # met: bars above, below
    svg(tmp_path / "c.svg", '<rect width="2" height="2" fill="#f00"/><rect x="2" width="2" height="2" fill="#00f"/>',
        'viewBox="0 0 4 2" preserveAspectRatio="xMinYMid slice"')
    r = renderer(scene([{"id": "s", "type": "svg", "src": "c.svg", "x": 200, "y": 150, "w": 100, "h": 100}]),
                 tmp_path, backend)
    img = rgb(r)
    assert r.s.layers[0].raw["clip"] is True
    assert img[150, 190].tolist() == [255, 0, 0] and img[150, 240].tolist() == [255, 0, 0]  # the red half fills it
    assert img[150, 260].tolist() == [0, 0, 0]  # and the blue one is cut off


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_drawing_standing_still_plays_back_from_one_picture(tmp_path, backend):
    # a faded stroke alone: a picture would fold its offscreen layer into the stroke's paint
    svg(tmp_path / "mix.svg", MIX + '<path id="faded" d="M10 10 L390 190" stroke="#fff" stroke-width="0.7" '
        'fill="none" opacity="0.5"/>', 'width="200" height="100" viewBox="0 0 400 200"')

    def frames(parts, still=True):
        r = renderer(scene([{"id": "s", "type": "svg", "src": "mix.svg", "x": 200, "y": 150, "w": 390,
                             "rotate": [[0, 0], [1, 20]], "parts": parts}]), tmp_path, backend)
        if not still:
            r._still = lambda *a: None
        return [r.frame(i).copy() for i in range(10)], r._rests.get("s")  # frame() reuses its buffer
    plain, rest = frames({})
    assert rest[:2] == [0, 1] and rest[2] is not None  # the group turns; what is in it stands still
    assert all(np.array_equal(a, b) for a, b in zip(plain, frames({}, still=False)[0]))
    popped, rest = frames({"faded": {"enter": {"name": "pop", "dur": 0.4}}})
    assert rest[0] == pytest.approx(0.4) and rest[2] is not None  # recorded once the pop is over
    assert all(np.array_equal(a, b) for a, b in zip(popped, frames({"faded": {"enter": {"name": "pop", "dur": 0.4}}},
                                                                     still=False)[0]))
    assert not np.array_equal(popped[1], plain[1]) and np.array_equal(popped[9], plain[9])


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_motion_blur_draws_the_same_with_the_picture(tmp_path, backend):
    spec = scene([{"id": "g", "type": "group", "x": [[0, 40], [1, 360]], "children": [
        {"id": "a", "type": "rect", "x": 0, "y": 150, "w": 30, "h": 30},
        {"id": "b", "type": "circle", "x": 0, "y": 100, "r": 15, "at": 0.53}]}], fps=10)
    spec["motionBlur"] = {"samples": 8, "shutter": 1}
    a, b = renderer(spec, tmp_path, backend), renderer(spec, tmp_path, backend)
    b._still = lambda *x: None
    for i in range(10):  # the frame at 0.5 s has sub-frames after b comes on at 0.53 s, while b is not on it yet
        assert np.array_equal(a.frame(i), b.frame(i)), i


def test_a_group_with_a_layer_that_changes_is_drawn_layer_by_layer(tmp_path):
    def rest(child, **group):
        b = {k: v for k, v in dict({"id": "b"}, **child).items() if v is not None}
        r = renderer(scene([dict({"id": "g", "type": "group", "children": [
            {"id": "a", "type": "rect", "x": 50, "y": 50, "w": 20, "h": 20}, b]}, **group)]), tmp_path)
        r.frame(5)
        return r._rests.get("g")
    still = {"type": "circle", "x": 90, "y": 50, "r": 10}
    assert rest(still)[2] is not None
    assert rest(still, x=[[0, 0], [1, 100]], opacity=0.5)[2] is not None  # the group itself may move and fade
    for change in ({"x": [[0, 80], [1, 90]]}, {"fill": "=hsl(t, 0.5, 0.5)"}, {"blur": 4}, {"shadow": True},
                   {"blend": "screen"}, {"fx": {"saturation": 0}},
                   {"type": "group", "combine": "union", "children": [still], "r": None}, {"rough": 2},
                   {"type": "text", "text": "hi", "r": None}, {"repeat": {"count": 3, "x": "=i * 30"}}):
        assert rest(dict(still, **change)) == [], change
    assert not rest(still, combine="union")
    assert rest(dict(still, at=0.5))[:2] == [0.5, 1]  # from when it comes on
    assert rest(dict(still, exit={"name": "fade", "dur": 0.2}))[1] == pytest.approx(0.8)  # until its exit starts
    assert rest(still, enter={"name": "fade", "stagger": 0.3, "dur": 0.2})[0] == pytest.approx(0.5)  # its last turn


def test_arc_flags_may_run_into_the_numbers_after_them():
    def same(a, b):
        return svgdoc.path_data(svgpath.parse(a)) == svgdoc.path_data(svgpath.parse(b))
    assert same("M0 0 a10 10 0 0110 10", "M0 0 a10 10 0 0 1 10 10")
    assert same("M0 0 A5 5 0 1010 0", "M0 0 A5 5 0 1 0 10 0")
    assert same("M0 0 a5 5 0 01-5 5", "M0 0 a5 5 0 0 1 -5 5")
    assert same("M0 0 a5 5 0 11.5 5", "M0 0 a5 5 0 1 1 .5 5")  # flags 1 and 1, then x .5
    assert not same("M0 0 a10 10 0 0110 10", "M0 0 a10 10 0 0 0 110 10")
