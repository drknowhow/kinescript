"""Generate the sample footage and music the examples use (no third-party media)."""
import subprocess
from pathlib import Path

from kinescript.media import FFMPEG

out = Path(__file__).parent / "media"
out.mkdir(exist_ok=True)
jobs = {
    "fractal.mp4": ["-f", "lavfi", "-i", "mandelbrot=size=1280x720:rate=30:end_scale=0.02",
                    "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", "6"],
    "cells.mp4": ["-f", "lavfi", "-i", "cellauto=rule=110:size=1280x720:rate=30,scale=1280:720:flags=neighbor,"
                  "colorchannelmixer=rr=0.2:gg=0.85:bb=0.85",
                  "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000", "-t", "6"],
    "pad.m4a": ["-f", "lavfi", "-i", "aevalsrc=0.12*sin(2*PI*110*t)*(0.6+0.4*sin(2*PI*0.25*t))"
                "+0.08*sin(2*PI*164.8*t)+0.06*sin(2*PI*220*t)+0.04*sin(2*PI*277.2*t):s=48000", "-t", "45"],
}
for name, args in jobs.items():
    tail = ["-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"] if name.endswith(".mp4") else ["-c:a", "aac"]
    subprocess.run([FFMPEG, "-y", "-v", "error", *args, *tail, str(out / name)], check=True)
    print("wrote", out / name)
