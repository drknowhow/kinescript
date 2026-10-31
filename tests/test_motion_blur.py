"""Motion blur is the integer mean of the sub-frames: the same bytes from the GPU pass and from numpy, in video
frames, in the I420 path and in stills, and when the layers that hold still are drawn once."""
import subprocess
import sys

import numpy as np
import pytest
import skia

import kinescript as k
from kinescript.render import BLUR_GPU, Renderer


def _spec(samples, bg="#101418"):
    return {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 1, "bg": bg},
            "motionBlur": {"samples": samples, "shutter": 0.5},
            "layers": [{"id": "c", "type": "circle", "r": 24, "x": "=40+t*400", "y": 90, "fill": "#ff5a36"},
                       {"id": "t", "type": "text", "text": "Blur", "x": "=280-t*300", "y": 40, "size": 36,
                        "fill": "#3ddbd9", "opacity": 0.7, "rotate": "=t*90"}]}


def _renderer(spec, backend, even=False):
    try:
        return Renderer(k.Scene(spec), backend=backend, even=even)
    except RuntimeError:
        pytest.skip("no GPU")


def _mean(r, n, samples, shutter=0.5):
    """Motion blur as it was first written: sum the sub-frames' readbacks in 32 bits, floor-divide, straighten."""
    T, sh = n / r.s.fps, shutter / r.s.fps
    acc = np.zeros((r.h, r.w, 4), np.uint32)
    for j in range(samples):
        r.draw(T + (j / (samples - 1) - 0.5) * sh)
        r.surface.readPixels(r.info, r.buf, r.w * 4)
        acc += r.buf
    out = (acc // samples).astype(np.uint8)
    if r.transparent:
        a = out[..., 3:4].astype(np.uint16)
        rgb = np.where(a > 0, (out[..., :3].astype(np.uint16) * 255 + a // 2) // np.maximum(a, 1), 0)
        out[..., :3] = np.minimum(rgb, 255).astype(np.uint8)
    return out


@pytest.mark.parametrize("backend,samples", [("gpu", 2), ("gpu", 6), ("gpu", 7), ("gpu", BLUR_GPU),
                                             ("gpu", BLUR_GPU + 1), ("cpu", 6), ("cpu", 260)])
@pytest.mark.parametrize("bg", ["#101418", "transparent"])
def test_blurred_frame_is_the_integer_mean_of_its_subframes(backend, samples, bg):
    r = _renderer(_spec(samples, bg), backend)
    for n in (3, 17):
        got = r.frame(n).copy()
        assert np.array_equal(got, _mean(r, n, samples)), n


def test_blurred_frame_is_pure_function_of_index():
    spec = _spec(6)
    r = _renderer(spec, "gpu")
    sequential = [r.frame(n).copy() for n in range(12, 20)]
    assert np.array_equal(sequential[5], _renderer(spec, "gpu").frame(17))


def test_blurred_mp4_frames_take_the_i420_path():
    r = _renderer(_spec(6), "gpu", even=True)
    assert r.yuv
    got = r.frame_i420(17).astype(int)
    f = _mean(r, 17, 6)[..., :3].astype(np.float64) / 255
    m = [0.2126, 0.7152, 0.0722]
    c = (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) / 4
    planes = [16 + 219 * (f @ m), 128 + 224 * (c[..., 2] - c @ m) / 1.8556, 128 + 224 * (c[..., 0] - c @ m) / 1.5748]
    want = np.concatenate([np.clip(np.round(p), 0, 255).astype(np.uint8).ravel() for p in planes]).astype(int)
    assert np.abs(got - want).max() <= 1


def _png(data: bytes) -> np.ndarray:
    """A PNG's pixels as stored: straight RGBA, not premultiplied on the way through."""
    codec = skia.Codec.MakeFromData(data)
    w, h = codec.getInfo().width(), codec.getInfo().height()
    px = np.empty((h, w, 4), np.uint8)
    info = skia.ImageInfo.Make(w, h, skia.kRGBA_8888_ColorType, skia.kUnpremul_AlphaType)
    assert codec.getPixels(info, px, w * 4) == skia.Codec.kSuccess
    return px


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
@pytest.mark.parametrize("bg", ["#101418", "transparent"])
def test_stills_and_png_frames_save_the_blurred_frame(tmp_path, monkeypatch, backend, bg):
    if backend == "cpu":
        monkeypatch.setenv("KINE_BACKEND", "cpu")
        monkeypatch.setattr(sys.modules["kinescript.render"], "_GL", {})
    spec = _spec(6, bg)
    want = _renderer(spec, "auto").frame(17).copy()
    k.still(spec, 17 / 30, str(tmp_path / "s.png"))
    k.render(spec, str(tmp_path / "seq") + "/", start=17 / 30, end=18 / 30)
    for png in (tmp_path / "s.png", tmp_path / "seq" / "frame_00017.png"):
        data = png.read_bytes()
        assert data[25] == 6  # RGBA: a transparent frame keeps its alpha, an opaque one is written as before
        assert np.array_equal(_png(data), want), png  # transparent: Skia's encoder rounded some colours 1 off


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_a_transparent_png_is_within_a_level_of_what_skias_encoder_wrote(backend):
    # PNGs were Skia's encoding of the premultiplied surface; they are now the straight frame, as a .mov or .webm
    # gets it. Straightening a blurred frame rounds its colour a level apart from Skia's encoder in a few pixels.
    from kinescript import output
    r = _renderer(_spec(6, "transparent"), backend)
    for n in (3, 17):
        ours = _png(output._png_end(output._png_start(r.frame(n).copy())))
        before = _png(bytes(r.snapshot().encodeToData(skia.kPNG, 100)))
        d = np.abs(ours.astype(int) - before.astype(int))
        assert d[..., 3].max() == 0 and d.max() <= 1, n


# ---- layers that hold still while the shutter is open are drawn once ------------------------------------------

def _held_spec(bg, samples):
    return {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 1.2, "bg": bg},
            "grade": {"contrast": 1.2, "saturate": 0.6, "vignette": 0.3, "grain": 0.1},
            "motionBlur": {"samples": samples, "shutter": 0.5},
            "layers": [
                {"id": "card", "type": "rect", "x": 160, "y": 90, "w": 220, "h": 110, "fill": "#2a3140",
                 "radius": 12, "enter": {"name": "pop", "dur": 0.2}, "exit": {"name": "fade", "dur": 0.2}},
                {"id": "word", "type": "text", "text": "Held", "x": 160, "y": 64, "size": 44, "split": "chars",
                 "enter": {"name": "rise", "dur": 0.2, "stagger": 0.05}, "opacity": 0.9},
                {"id": "bar", "type": "rect", "x": [[0, 70], [0.5, 250]], "y": 150, "w": 40, "h": 10,
                 "fill": "#ff5a36", "blend": "multiply"},
                {"id": "g", "type": "group", "children": [
                    {"id": "ring", "type": "circle", "r": 14, "x": 100, "y": 118, "stroke": "#3ddbd9",
                     "draw": [[0.2, 0], [0.6, 1]]},
                    {"id": "label", "type": "text", "text": "on", "x": 220, "y": 118, "size": 20, "at": 0.5}]},
                {"id": "dot", "type": "circle", "r": 10, "x": "=40+t*240", "y": 100, "fill": "#ffffff",
                 "blend": "screen", "at": 0.8}]}


@pytest.mark.parametrize("backend,samples,scale", [("gpu", 6, 1.0), ("gpu", BLUR_GPU + 1, 0.5), ("cpu", 6, 0.5)])
@pytest.mark.parametrize("bg", ["#101418", "transparent"])
def test_layers_that_hold_still_are_drawn_once_with_the_same_bytes(backend, samples, scale, bg):
    # The background and the bottom layers that do not move while the shutter is open are drawn once and every
    # sub-frame starts from them; with nothing moving the frame is drawn once. Either way the bytes are the mean of
    # every layer drawn in every sub-frame: enter, exit, split units, keyframes, draw-on, a group, a late child, a
    # blend over the held layers, the grade, grain and alpha.
    held = _as_every_layer_draws_it(_held_spec(bg, samples), backend, scale)
    top = len(_held_spec(bg, samples)["layers"])
    assert 0 in held and top in held and any(0 < h < top for h in held), held


def _as_every_layer_draws_it(spec, backend, scale=1.0):
    """Every frame of spec is the same bytes as with every layer drawn in every sub-frame. Returns how many layers
    each frame held."""
    try:
        r, every = (Renderer(k.Scene(spec), backend=backend, scale=scale) for _ in range(2))
    except RuntimeError:
        pytest.skip("no GPU")
    every._held = lambda times, F: 0  # every layer in every sub-frame, as before
    held, real = [], r._held
    r._held = lambda times, F: held.append(real(times, F)) or held[-1]
    for n in range(r.s.frames):
        assert np.array_equal(r.frame(n), every.frame(n)), n
    r.s.close()
    every.s.close()
    return held


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    from kinescript import media
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run([media.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-t", "2",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return str(p)


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_footage_that_stays_put_is_drawn_once_with_the_same_bytes(clip, backend):
    # Each sub-frame shows its frame's picture, so a clip holds once its punch-in has landed. Under a grade the held
    # clip is drawn in a layer as the graded frame draws it: upscaled straight onto a surface, it came out a level or
    # two apart in a few hundred pixels.
    spec = {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 1.2}, "grade": {"contrast": 1.2},
            "motionBlur": {"samples": 6, "shutter": 0.5},
            "layers": [{"id": "v", "type": "video", "src": clip, "x": 0, "y": 0, "anchor": "topleft", "w": 320,
                        "h": 180, "zoom": [[0, 1], [0.5, 1.3]]},
                       {"id": "dot", "type": "circle", "r": 12, "x": "=40+t*200", "y": 90, "fill": "#ffffff"}]}
    held = _as_every_layer_draws_it(spec, backend, 1.0 if backend == "gpu" else 0.5)
    assert 0 in held and 1 in held, held


def test_a_prop_holds_outside_its_keyframes_only():
    from kinescript.props import Ctx, compile_prop
    ctx = Ctx(100, 100, 30)
    keys = compile_prop("x", [[0.5, 0], [1, 100]], ctx)
    assert keys.held(0, 0.5, 2) and keys.held(1.01, 2, 2)
    assert not keys.held(0.4, 0.6, 2) and not keys.held(0.9, 1.1, 2) and not keys.held(1, 1.2, 2)
    assert compile_prop("x", 40, ctx).held(0, 2, 2)
    assert not compile_prop("x", "=40", ctx).held(0, 0, 2)  # an expression may follow anything
    tail = compile_prop("x", [[-0.5, 0], ["end", 100]], ctx)  # keys at 1.5 s and 2 s of a 2 s layer
    assert tail.held(0, 1.5, 2) and not tail.held(0, 1.6, 2) and tail.held(0, 1.6, 4)
