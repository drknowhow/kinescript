"""Kinescript Sans ships with the package: a scene that names it draws with its own files."""

import hashlib

from kinescript.render import BUNDLED, FONT_DIR

SHA = {"KinescriptSans-Regular.ttf": "a298f414b014e79496640397be482103b98fe399d8a2738ca9b9a78c1ab31144",
       "KinescriptSans-Bold.ttf": "51aad98ddda11dcc8e0e9dd9052ffbad16cd20f891b584c24255330209b71da4"}


def test_the_files_are_the_signed_release():
    # YepType release kinescript-sans 0.1.0, bundle sha256:eb49f672...
    for name, sha in SHA.items():
        assert hashlib.sha256((FONT_DIR / name).read_bytes()).hexdigest() == sha
    assert (FONT_DIR / "KinescriptSans-OFL.txt").read_text(encoding="utf-8").startswith(
        "Copyright 2026 The Kinescript Sans Project Authors")


def test_scenes_that_name_it_draw_with_its_files(tmp_path):
    # "Kinescript Sans" at 700 draws exactly what the Bold file does by path, at 400 what Regular
    # does; the two weights differ, so a scene really gets the weight it asks for.
    import shutil

    import numpy as np

    import kinescript as k
    from kinescript.render import Renderer
    from test_brand import scene_at

    def frame(sub, font, weight):
        d = tmp_path / sub
        d.mkdir()
        for name in SHA:
            shutil.copy(FONT_DIR / name, d / name)
        layer = {"id": "t", "type": "text", "text": "Fox 0123 ffi", "x": "50%", "y": "50%", "size": 48,
                 "weight": weight, "font": font, "fill": "#ffffff"}
        return Renderer(k.load(str(scene_at(d, [layer]))), backend="cpu").frame(0).copy()

    bold, regular = frame("b", "Kinescript Sans", 700), frame("r", "Kinescript Sans", 400)
    assert np.array_equal(bold, frame("bp", "KinescriptSans-Bold.ttf", 700))
    assert np.array_equal(regular, frame("rp", "KinescriptSans-Regular.ttf", 400))
    assert not np.array_equal(bold, regular)
    assert set(BUNDLED["kinescript sans"]) == {400, 700}
