"""tomp4.py RUN.out OUT.mp4: palettized 320x200 frames from play.html -> 35 fps H.264, 1600x1200 nearest."""
import base64
import json
import subprocess
import sys

import numpy as np

text = open(sys.argv[1]).read()
run = json.loads(text[text.find("{"):])
if "error" in run:
    sys.exit(run["error"])

palette = np.array([[c & 255, (c >> 8) & 255, (c >> 16) & 255] for c in run["palette"]], np.uint8)
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x200", "-r", "35", "-i", "-",
     "-vf", "scale=1600:1200:flags=neighbor", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
     "-pix_fmt", "yuv420p", sys.argv[2]],
    stdin=subprocess.PIPE)
for frame in run["frames"]:
    indices = np.frombuffer(base64.b64decode(frame), np.uint8).reshape(200, 320)
    ffmpeg.stdin.write(palette[indices].tobytes())
ffmpeg.stdin.close()
ffmpeg.wait()

print(sys.argv[2], len(run["frames"]), "frames",
      {k: run.get(k) for k in ["map", "result", "tics", "kills", "totalKills", "items", "health"]})
