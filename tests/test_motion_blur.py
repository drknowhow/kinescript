"""Motion blur is the integer mean of the sub-frames: the same bytes from the GPU pass and from numpy, in video
frames, in the I420 path and in stills."""
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


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_stills_and_png_frames_save_the_blurred_frame(tmp_path, monkeypatch, backend):
    if backend == "cpu":
        monkeypatch.setenv("KINE_BACKEND", "cpu")
        monkeypatch.setattr(sys.modules["kinescript.render"], "_GL", {})
    spec = _spec(6)
    want = _renderer(spec, "auto").frame(17).copy()
    k.still(_spec(6), 17 / 30, str(tmp_path / "s.png"))
    k.render(_spec(6), str(tmp_path / "seq") + "/", start=17 / 30, end=18 / 30)
    for png in (tmp_path / "s.png", tmp_path / "seq" / "frame_00017.png"):
        got = skia.Image.open(str(png)).toarray(colorType=skia.kRGBA_8888_ColorType)
        assert np.array_equal(got, want), png
