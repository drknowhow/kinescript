"""Colours must survive render -> MP4 -> decode, on both the GPU I420 path and the RGBA path."""
import json
import subprocess

import numpy as np
import pytest

import kinescript as k
from kinescript import media
from kinescript.render import Renderer, _read_back

SWATCHES = ["#ff0000", "#00ff00", "#0000ff", "#ff633f", "#808080", "#f2f0e8"]


def _scene(tmp_path):
    layers = [{"id": f"s{i}", "type": "rect", "x": 160 * i, "y": 0, "anchor": "topleft", "w": 160, "h": 360,
               "fill": c} for i, c in enumerate(SWATCHES)]
    f = tmp_path / "swatch.json"
    f.write_text(json.dumps({"canvas": {"w": 960, "h": 360, "fps": 30, "duration": 0.2, "bg": "#000000"},
                             "layers": layers}))
    return f


def _ref_i420(rgba):
    f = rgba[..., :3].astype(np.float64) / 255
    m = [0.2126, 0.7152, 0.0722]
    c = (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) / 4
    planes = [16 + 219 * (f @ m), 128 + 224 * (c[..., 2] - c @ m) / 1.8556, 128 + 224 * (c[..., 0] - c @ m) / 1.5748]
    return np.concatenate([np.clip(np.round(p), 0, 255).astype(np.uint8).ravel() for p in planes])


def test_gpu_i420_matches_the_bt709_formula():
    s = k.load("marketing/promo.json")
    r = Renderer(s, even=True)
    if not r.yuv:
        s.close()
        pytest.skip("GPU I420 path needs the GPU backend")
    for n in (0, 45, 120):
        got = r.frame_i420(n).astype(int)
        r.surface.readPixels(r.info, r.buf, r.w * 4)
        assert np.abs(got - _ref_i420(r.buf).astype(int)).max() <= 1, n
    s.close()


def test_gpu_i420_stream_is_the_blocking_reads_in_order():
    """frames_i420 reads a frame back while the next one draws: the same bytes as frame_i420, in order, also when
    the stream stops early, and a failed readback is an error, not a frame."""
    a, b = k.load("examples/hello.json"), k.load("examples/hello.json")
    ra, rb = Renderer(a, even=True), Renderer(b, even=True)
    if not ra.yuv:
        a.close()
        b.close()
        pytest.skip("GPU I420 path needs the GPU backend")
    ns = [0, 40, 41, 125, 3, 3]
    want = [ra.frame_i420(n).copy() for n in ns]
    same = [np.array_equal(px, w) for px, w in zip(rb.frames_i420(ns), want, strict=True)]
    assert same == [True] * len(ns)
    stream = rb.frames_i420(ns)
    assert np.array_equal(next(stream), want[0])
    stream.close()  # frame 40's readback is still in flight
    assert np.array_equal(rb.frame_i420(41), want[2])
    job = [np.zeros(8, np.uint8), False, None]
    _read_back(job, 4, None)
    assert job[1] and isinstance(job[2], RuntimeError)
    with pytest.raises(RuntimeError, match="readback"):
        rb._collect(job)
    a.close()
    b.close()


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_mp4_is_tagged_bt709_and_colours_round_trip(tmp_path, monkeypatch, backend):
    if backend == "cpu":
        monkeypatch.setenv("KINE_BACKEND", "cpu")
        import sys
        monkeypatch.setattr(sys.modules["kinescript.render"], "_GL", {})
    out = tmp_path / "s.mp4"
    k.render(str(_scene(tmp_path)), str(out), quality="high", workers=1)
    probe = subprocess.run([media.FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v",
                            "-show_entries", "stream=color_space,color_primaries,color_transfer,color_range",
                            "-of", "json", str(out)], capture_output=True, text=True)
    tags = json.loads(probe.stdout)["streams"][0]
    assert tags == {"color_range": "tv", "color_space": "bt709", "color_transfer": "bt709",
                    "color_primaries": "bt709"}, tags
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(out), "-frames:v", "1", "-vf",
                          "scale=in_color_matrix=bt709:in_range=tv,format=rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(360, 960, 3)
    for i, c in enumerate(SWATCHES):
        want = np.array([int(c[j:j + 2], 16) for j in (1, 3, 5)])
        got = img[180, 160 * i + 80].astype(int)
        assert np.abs(got - want).max() <= 4, (c, got.tolist())


def _clip(path, w, h, mode):
    """A 0.5 s clip of the swatches, encoded the way players will read it: "tagged" BT.709,
    "hd" untagged but BT.709 (players assume 709 for HD), "sd" untagged BT.601."""
    cells = "".join(f"drawbox=x={w * i // 6}:y=0:w={w // 6 + 1}:h={h}:color={c}:t=fill,"
                    for i, c in enumerate(SWATCHES))
    vf = "format=rgb24," + cells + {
        "tagged": "scale=out_color_matrix=bt709:out_range=tv,"
                  "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv,",
        "hd": "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=colorspace=unknown,",
        "sd": "scale=out_color_matrix=bt601:out_range=tv,format=yuv420p,setparams=colorspace=unknown,"}[mode]
    subprocess.run([media.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={w}x{h}:d=0.5:r=30",
                    "-vf", vf + "format=yuv420p", "-c:v", "libx264", "-crf", "1", str(path)], check=True)
    assert (media.probe(str(path))["color_space"] == "bt709") == (mode == "tagged")


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
@pytest.mark.parametrize("w,h,mode", [(1280, 720, "tagged"), (1280, 720, "hd"), (640, 360, "sd")])
def test_video_layer_colours_match_how_players_decode_the_clip(tmp_path, monkeypatch, backend, w, h, mode):
    if backend == "cpu":
        import sys
        monkeypatch.setenv("KINE_BACKEND", "cpu")
        monkeypatch.setattr(sys.modules["kinescript.render"], "_GL", {})
    clip = tmp_path / "clip.mp4"
    _clip(clip, w, h, mode)
    spec = {"canvas": {"w": w, "h": h, "fps": 30, "duration": 0.4},
            "layers": [{"id": "v", "type": "video", "src": str(clip).replace("\\", "/"), "w": w, "h": h,
                        "anchor": "topleft"}]}
    f = tmp_path / "v.json"
    f.write_text(json.dumps(spec))
    s = k.load(str(f))
    r = Renderer(s, even=True)
    img = r.frame(3)
    for i, c in enumerate(SWATCHES):
        want = np.array([int(c[j:j + 2], 16) for j in (1, 3, 5)])
        got = img[h // 2, w * i // 6 + w // 12, :3].astype(int)
        assert np.abs(got - want).max() <= 4, (backend, c, got.tolist())
    s.close()
