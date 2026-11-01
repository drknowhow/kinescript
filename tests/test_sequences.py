"""Image sequences: an image layer playing numbered files (frames) or the cells of a sheet, at fps, looping, holding
its last frame or playing back and forth, its frame a function of the layer's own time."""
import numpy as np
import pytest
import skia

import kinescript as k
from kinescript import media
from kinescript.render import Renderer

RED, GREEN, BLUE, YELLOW = (230, 40, 40), (40, 200, 60), (40, 80, 230), (240, 200, 40)


def scene(layers, **canvas):
    return {"canvas": dict({"w": 320, "h": 180, "fps": 24, "duration": 2, "bg": "#000000"}, **canvas),
            "layers": layers}


def _png(path, colours, cols=None, cell=(64, 64)):
    """Coloured cells in a grid, cols across, saved as a .png."""
    cols = cols or len(colours)
    rows = -(-len(colours) // cols)
    s = skia.Surface(cell[0] * cols, cell[1] * rows)
    c = s.getCanvas()
    for i, col in enumerate(colours):
        c.drawRect(skia.Rect.MakeXYWH(i % cols * cell[0], i // cols * cell[1], *cell),
                   skia.Paint(Color=skia.Color(*col)))
    s.makeImageSnapshot().save(str(path))


def _frames(spec, tmp_path, backend="cpu", n=range(10), at=(160, 90), track=False):
    try:
        r = Renderer(k.Scene(spec, base_dir=tmp_path), backend=backend, track=track)
    except RuntimeError:
        pytest.skip("no GPU")
    return [tuple(int(v) for v in r.frame(i)[at[1], at[0], :3]) for i in n], r


def codes(spec, base):
    try:
        s = k.Scene(spec, base_dir=base)
        return [i["code"] for i in s.issues], s
    except k.SceneError as e:
        return [i["code"] for i in e.issues], None


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_sheet_plays_its_cells_row_by_row_at_its_fps(tmp_path, backend):
    _png(tmp_path / "walk.png", [RED, GREEN, BLUE, YELLOW], cols=2)
    spec = scene([{"id": "w", "type": "image", "sheet": {"src": "walk.png", "cols": 2, "rows": 2}, "x": 160, "y": 90,
                   "w": 100}])
    got, _ = _frames(spec, tmp_path, backend)
    assert got == [RED, RED, GREEN, GREEN, BLUE, BLUE, YELLOW, YELLOW, RED, RED]  # 12 fps in a 24 fps scene
    spec["layers"][0].update(fps=24, sheet={"src": "walk.png", "cols": 2, "rows": 2, "count": 3})
    got, _ = _frames(spec, tmp_path, backend, n=range(5))
    assert got == [RED, GREEN, BLUE, RED, GREEN]  # count leaves the last cell out


def test_loop_false_holds_the_last_frame_and_pingpong_plays_back(tmp_path):
    q = media.Frames(None, 4, 1, 4)
    assert [q.at(i / 12, 12, True) for i in range(9)] == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert [q.at(i / 12, 12, False) for i in range(6)] == [0, 1, 2, 3, 3, 3]
    assert [q.at(i / 12, 12, "pingpong") for i in range(9)] == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert media.Frames(None, 1, 1, 1).at(5.0, 12, "pingpong") == 0
    assert [q.at(i / 30, 10, True) for i in range(7)] == [0, 0, 0, 1, 1, 1, 2]
    assert q.at(9 / 30 - 0.1, 10, True) == 2  # 0.2 s in, though 9 / 30 - 0.1 is a hair under 0.2
    _png(tmp_path / "strip.png", [RED, GREEN, BLUE])
    spec = scene([{"id": "s", "sheet": {"src": "strip.png", "cols": 3}, "fps": 24, "loop": "yoyo", "x": 160, "y": 90}])
    got, _ = _frames(spec, tmp_path, n=range(6))
    assert got == [RED, GREEN, BLUE, GREEN, RED, GREEN]  # the type is read from sheet; yoyo is pingpong
    spec["layers"][0]["loop"] = False
    got, _ = _frames(spec, tmp_path, n=range(5))
    assert got == [RED, GREEN, BLUE, BLUE, BLUE]


def test_frames_read_a_pattern_a_glob_a_folder_or_a_list(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for i, col in enumerate([RED, GREEN, BLUE], start=1):
        _png(run / f"run_{i:03d}.png", [col], cell=(40, 30))
    _png(run / "run_010.png", [YELLOW], cell=(40, 30))
    names = lambda files: [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in files]  # noqa: E731
    for spec in ("run_%03d.png", "run_###.png", "run_*.png"):
        files, gaps = media.sequence(str(run / spec))
        assert names(files) == ["run_001.png", "run_002.png"] + ["run_003.png"] * 7 + ["run_010.png"], spec
        assert gaps == [4, 5, 6, 7, 8, 9]
    assert names(media.sequence(str(run))[0]) == names(media.sequence(str(run / "run_%03d.png"))[0])
    (run / "notes.txt").write_text("not a frame")
    (run / "cover.png").write_bytes((run / "run_001.png").read_bytes())
    files, gaps = media.sequence(str(run))  # names that differ in more than their number go in name order
    assert names(files) == ["cover.png", "run_001.png", "run_002.png", "run_003.png", "run_010.png"] and not gaps
    spec = scene([{"id": "f", "type": "image", "frames": ["run/run_002.png", "run/run_010.png"], "fps": 24,
                   "x": 160, "y": 90}])
    got, _ = _frames(spec, tmp_path, n=range(3))
    assert got == [GREEN, YELLOW, GREEN]


def test_a_missing_number_holds_the_frame_before_and_check_warns(tmp_path):
    for i, col in [(1, RED), (2, GREEN), (4, YELLOW)]:
        _png(tmp_path / f"f{i}.png", [col], cell=(40, 30))
    spec = scene([{"id": "f", "type": "image", "frames": "f%d.png", "fps": 24, "x": 160, "y": 90}])
    found, s = codes(spec, tmp_path)
    assert "missing-frames" in found
    note = next(i for i in s.issues if i["code"] == "missing-frames")
    assert "skip number 3" in note["message"] and note["level"] == "warning"
    got, _ = _frames(spec, tmp_path, n=range(5))
    assert got == [RED, GREEN, GREEN, YELLOW, RED]


def test_frames_that_match_nothing_fail_naming_the_files_there(tmp_path):
    _png(tmp_path / "walk_01.png", [RED])
    found, _ = codes(scene([{"id": "f", "type": "image", "frames": "run_%02d.png"}]), tmp_path)
    assert found == ["media-missing"]
    with pytest.raises(k.SceneError) as e:
        k.Scene(scene([{"id": "f", "type": "image", "frames": "run_%02d.png"}]), base_dir=tmp_path)
    assert "walk_01.png" in str(e.value.issues[0]["fix"])
    found, _ = codes(scene([{"id": "f", "type": "image", "frames": ["walk_01.png", "walk_02.png"]}]), tmp_path)
    assert found == ["media-missing"]
    found, _ = codes(scene([{"id": "f", "type": "image", "sheet": {"src": "gone.png", "cols": 2}}]), tmp_path)
    assert found == ["media-missing"]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_sheet_cell_does_not_take_in_its_neighbours_pixels(tmp_path, backend):
    _png(tmp_path / "rb.png", [RED, BLUE], cell=(8, 8))
    spec = scene([{"id": "s", "type": "image", "sheet": {"src": "rb.png", "cols": 2}, "x": 160, "y": 90, "w": 160,
                   "fit": "stretch", "loop": False}])
    try:
        r = Renderer(k.Scene(spec, base_dir=tmp_path), backend=backend)
    except RuntimeError:
        pytest.skip("no GPU")
    px = r.frame(0)[90, 80:240, :3].astype(int)  # the box is x 80 to 240: cell 0, red, eight times its size
    assert px[:, 2].max() <= BLUE[2] * 0.02 + RED[2] and (px[-1] == RED).all()


def test_pixelated_scales_an_image_up_with_hard_edges(tmp_path):
    _png(tmp_path / "bw.png", [(0, 0, 0), (255, 255, 255)], cell=(1, 1))
    layer = {"id": "p", "type": "image", "src": "bw.png", "x": 160, "y": 90, "w": 200, "h": 100, "fit": "stretch"}
    row = lambda raw: Renderer(k.Scene(scene([raw]), base_dir=tmp_path),  # noqa: E731
                               backend="cpu").frame(0)[90, 60:260, 0].astype(int)
    soft, hard = row(layer), row(dict(layer, pixelated=True))
    assert (hard[:100] == 0).all() and (hard[100:] == 255).all()
    assert ((soft > 20) & (soft < 235)).sum() > 50  # linear filtering spreads the step over the middle


def test_the_box_is_a_frame_or_a_cell_in_size(tmp_path):
    _png(tmp_path / "grid.png", [RED] * 8, cols=4, cell=(30, 20))
    _png(tmp_path / "a1.png", [RED], cell=(50, 40))
    spec = scene([{"id": "s", "type": "image", "sheet": {"src": "grid.png", "cols": 4, "rows": 2}, "x": 100, "y": 90},
                  {"id": "w", "type": "image", "sheet": {"src": "grid.png", "cols": 4, "rows": 2}, "x": 100, "y": 90,
                   "w": 60},
                  {"id": "f", "type": "image", "frames": "a%d.png", "x": 200, "y": 90}])
    _, r = _frames(spec, tmp_path, n=[0], track=True)
    size = lambda b: (b["rect"][2] - b["rect"][0], b["rect"][3] - b["rect"][1])  # noqa: E731
    assert size(r.boxes["s"]) == (30, 20) and size(r.boxes["w"]) == (60, 40) and size(r.boxes["f"]) == (50, 40)


def test_a_motion_blur_sub_frame_shows_its_frames_drawing(tmp_path):
    _png(tmp_path / "rg.png", [RED, GREEN])
    spec = dict(scene([{"id": "s", "sheet": {"src": "rg.png", "cols": 2}, "fps": 24, "x": 160, "y": 90,
                        "w": 120}]), motionBlur={"samples": 6, "shutter": 1})
    got, _ = _frames(spec, tmp_path, n=range(4))
    assert got == [RED, GREEN, RED, GREEN]  # not a mix of the two drawings


def test_an_image_sequence_frame_is_a_function_of_its_index(tmp_path):
    _png(tmp_path / "walk.png", [RED, GREEN, BLUE, YELLOW], cols=2)
    for i, col in enumerate([BLUE, YELLOW, RED], start=1):
        _png(tmp_path / f"run{i}.png", [col], cell=(40, 30))
    spec = scene([{"id": "w", "sheet": {"src": "walk.png", "cols": 2, "rows": 2}, "x": "=80+t*40", "y": 90,
                   "loop": "pingpong", "at": 0.3},
                  {"id": "r", "frames": "run*.png", "fps": 7, "x": 240, "y": 90, "loop": False}])
    r = Renderer(k.Scene(spec, base_dir=tmp_path), backend="cpu")
    sequential = [r.frame(n).copy() for n in range(30)]
    alone = Renderer(k.Scene(spec, base_dir=tmp_path), backend="cpu").frame(23)
    assert np.array_equal(sequential[23], alone)


def test_schema_reads_sequence_keys_and_catches_bad_ones(tmp_path):
    _png(tmp_path / "walk.png", [RED, GREEN, BLUE], cell=(30, 20))
    def one(**raw):
        return codes(scene([dict({"id": "s", "type": "image"}, **raw)]), tmp_path)
    found, s = one(src="walk.png", sheet={"cols": 3}, loop="ping-pong", pixelated="yes")
    assert "renamed-key" in found and s.by_id["s"].raw["sheet"]["src"] == "walk.png"
    assert s.by_id["s"].raw["loop"] == "pingpong" and s.by_id["s"].raw["pixelated"] is True
    assert one(spriteSheet={"src": "walk.png", "columns": 3}, frameRate=8)[1].by_id["s"].raw["fps"] == 8
    assert "bad-value" in one(sheet={"src": "walk.png", "cols": 3}, fps=0)[0]
    assert "bad-value" in one(sheet={"src": "walk.png", "cols": 3}, loop="sometimes")[0]
    assert "bad-value" in one(sheet={"src": "walk.png", "cols": 2.5})[0]
    assert "bad-value" in one(sheet={"src": "walk.png", "cols": 3, "count": 4})[0]
    assert "missing-key" in one(sheet={"src": "walk.png"})[0]
    assert "missing-key" in one(sheet={"cols": 3})[0]
    assert "bad-value" in one(frames=5)[0]
    assert "duplicate-key" in one(src="walk.png", frames="walk*.png")[0]
    assert "missing-key" in one()[0]
    assert "ignored-key" in one(src="walk.png", fps=12)[0]
    assert "sheet-grid" in one(sheet={"src": "walk.png", "cols": 4})[0]
    assert one(sheet={"src": "walk.png", "cols": 3}, fps=8, loop=False)[0] == []
