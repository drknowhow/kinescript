"""Where render time goes, per workload: scene load, drawing (GPU work finished), readback, writing to the
encoder pipe, and the whole export. Stage times overlap in a real render (ffmpeg encodes while the next
frame draws), so they are not added up; the export wall time is the end-to-end number.

    python bench/stages.py                 # every workload
    python bench/stages.py hello promo     # some of them
    python bench/stages.py --runs 5        # exports per workload (median reported)

Numbers quoted in README.md, site/ and the promo come from this script; say which machine ran it."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import skia

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kinescript as k  # noqa: E402
from kinescript import output  # noqa: E402
from kinescript.render import Renderer, gpu_context  # noqa: E402


def _blur(path: str, transparent: bool = False) -> dict:
    spec = json.loads((ROOT / path).read_text())
    spec["motionBlur"] = {"samples": 8, "shutter": 0.5}
    if transparent:
        spec["canvas"]["bg"] = "transparent"
    return spec


WORKLOADS = {  # name: (scene file or spec, output extension)
    "hello": ("examples/hello.json", ".mp4"),
    "promo": ("marketing/promo.json", ".mp4"),
    "edit": ("examples/edit.json", ".mp4"),
    "model": ("examples/model.json", ".mp4"),
    "import": ("examples/import.json", ".mp4"),
    "blur": (lambda: _blur("examples/hello.json"), ".mp4"),
    "blur-alpha": (lambda: _blur("examples/hello.json", True), ".mov"),
}


def _scene(src) -> k.Scene:
    if callable(src):
        return k.Scene(src(), base_dir=ROOT / "examples")
    return k.load(str(ROOT / src))


def _ms(xs) -> str:
    xs = sorted(xs)
    return f"{1000 * statistics.median(xs):7.2f} / {1000 * xs[int(0.95 * (len(xs) - 1))]:7.2f}"


def stages(name: str, src, ext: str, frames: int) -> dict:
    t = time.perf_counter()
    scene = _scene(src)
    load = time.perf_counter() - t
    fmt = {".mov": "mov", ".webm": "webm"}.get(ext, "mp4")
    r = Renderer(scene, even=fmt == "mp4")
    yuv = r.yuv and fmt == "mp4"
    ctx = gpu_context() if r.backend == "gpu" else None
    enc, log = output._encoder(str(Path(tempfile.gettempdir()) / f"kine-bench{ext}"), r.w, r.h, scene.fps, fmt,
                               "standard", 0, yuv)
    grab = r.frame_i420 if yuv else r.frame
    picks = np.linspace(0, scene.frames - 1, min(frames, scene.frames)).round().astype(int)
    draw, frame, pipe = [], [], []
    mb = int(scene.motion_blur.get("samples", 0)) if scene.motion_blur else 0
    for i, n in enumerate([picks[0]] * 3 + list(picks)):  # three warm-up frames
        t0 = time.perf_counter()
        for s in range(max(1, mb)):  # drawing alone, waiting until the GPU has finished
            r.draw(n / scene.fps + s * 1e-4)
            if ctx:
                ctx.flushAndSubmit(skia.GrSyncCpu.kYes)
        t1 = time.perf_counter()
        buf = grab(n)  # what an export does per frame: draw, convert, read back
        t2 = time.perf_counter()
        enc.stdin.write(buf.data)
        t3 = time.perf_counter()
        if i >= 3:
            draw.append(t1 - t0)
            frame.append(t2 - t1)
            pipe.append(t3 - t2)
    enc.stdin.close()
    enc.wait()
    log.close()
    scene.close()
    return {"load": load, "draw": draw, "frame": frame, "pipe": pipe, "backend": r.backend, "yuv": yuv}


def export(src, ext: str, runs: int) -> dict:
    walls, info = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(runs):
            info = output.render(_scene(src), str(Path(tmp) / f"out{ext}"), quality="standard")
            walls.append(info["wall"])
    return {"wall": statistics.median(walls), "walls": walls, "workers": info["workers"], "frames": info["frames"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(WORKLOADS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=60, help="frames sampled for the stage times")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()
    git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout
    print(f"# {git.strip()} · {platform.processor() or platform.machine()} · Python {platform.python_version()} · "
          f"skia {skia.__version__} · numpy {np.__version__}")
    print("ms per frame, median / p95, one process. draw = drawing until the GPU finishes; frame = draw +\n"
          "conversion + readback, as an export grabs it; pipe = blocked writing to ffmpeg. export = wall seconds.")
    print(f"{'workload':11s} {'backend':8s} {'load s':>7s}  {'draw':>17s}  {'frame':>17s}  {'pipe':>17s}  "
          f"{'export s':>9s} {'frames':>6s} {'workers':>7s}")
    for name in a.names:
        src, ext = WORKLOADS[name]
        st = stages(name, src, ext, a.frames)
        ex = {"wall": float("nan"), "frames": 0, "workers": 0} if a.no_export else export(src, ext, a.runs)
        print(f"{name:11s} {st['backend'] + ('/i420' if st['yuv'] else ''):8s} {st['load']:7.2f}  {_ms(st['draw'])}  "
              f"{_ms(st['frame'])}  {_ms(st['pipe'])}  {ex['wall']:9.2f} {ex['frames']:6d} {ex['workers']:7d}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
