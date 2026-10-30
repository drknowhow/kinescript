"""Generate the sample footage, music, voice and model the examples use (no third-party media)."""
import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np

from kinescript import mesh
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

# A stand-in voice: a hum for each word, at the times its words file gives, so word cues can be tried
# without a speech engine.
WORDS = [("Write", 0.20, 0.55), ("a", 0.62, 0.72), ("scene.", 0.78, 1.30), ("Check", 1.80, 2.15), ("it.", 2.22, 2.50),
         ("Render", 3.00, 3.45), ("it.", 3.52, 3.85)]
hum = "+".join(f"between(t,{s},{e})*pow(sin(PI*(t-{s})/{e - s:.2f}),2)*sin(2*PI*{180 + 15 * k}*t)"
               for k, (_, s, e) in enumerate(WORDS))
subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc='0.35*({hum})':s=48000:d=4.2",
                str(out / "voice.wav")], check=True)
(out / "voice.words.json").write_text(json.dumps({"words": [{"word": w, "start": s, "end": e} for w, s, e in WORDS]},
                                                 indent=1))
print("wrote", out / "voice.wav", "and its words", out / "voice.words.json")


def rocket(path: Path):
    """A glTF binary built from primitives, laid out the way modelling tools write them: shared meshes,
    a node tree with translation, rotation and scale, and metallic-roughness materials."""
    blob, views, accessors, meshes = bytearray(), [], [], []

    def add(arr, ctype, kind, target):
        b = arr.tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(b), "target": target})
        blob.extend(b + b"\0" * (-len(b) % 4))
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
        accessors.append(acc)
        return len(accessors) - 1

    geometry = {}
    for shape in ("cylinder", "cone", "cube", "torus"):
        m = mesh.unit(shape, "medium", 0.08)  # y down; files are y up with faces wound the other way
        geometry[shape] = {"POSITION": add((m.P * (1, -1, 1)).astype("<f4"), 5126, "VEC3", 34962),
                           "NORMAL": add((m.N * (1, -1, 1)).astype("<f4"), 5126, "VEC3", 34962)}, \
            add(m.T[:, [0, 2, 1]].astype("<u2").ravel(), 5123, "SCALAR", 34963)

    def lin(hex_):  # baseColorFactor is linear
        return [(int(hex_[i:i + 2], 16) / 255) ** 2.2 for i in (1, 3, 5)] + [1.0]

    materials = [{"name": name, "pbrMetallicRoughness": {"baseColorFactor": lin(col), "metallicFactor": metal,
                                                         "roughnessFactor": rough}}
                 for name, col, metal, rough in (("hull", "#242832", 0.0, 0.4), ("paint", "#ff633f", 0.0, 0.35),
                                                 ("band", "#b9f45d", 0.0, 0.3), ("steel", "#8a8f98", 1.0, 0.3))]
    for name, shape, mat in (("body", "cylinder", 0), ("nose", "cone", 1), ("fin", "cube", 1), ("band", "torus", 2),
                             ("nozzle", "cylinder", 3)):
        attrs, idx = geometry[shape]
        meshes.append({"name": name, "primitives": [{"attributes": attrs, "indices": idx, "material": mat}]})
    nodes = [{"name": "rocket", "children": [1, 2, 3, 4, 5, 6, 7, 8]},
             {"name": "body", "mesh": 0, "scale": [40, 200, 40]},
             {"name": "nose", "mesh": 1, "translation": [0, 145, 0], "scale": [40, 90, 40]},
             {"name": "band", "mesh": 3, "translation": [0, 40, 0], "rotation": [math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
              "scale": [43.5, 43.5, 43.5]},  # the torus lies flat around the body, touching it
             {"name": "nozzle", "mesh": 4, "translation": [0, -112, 0], "scale": [26, 24, 26]}]
    for k in range(4):  # each fin (nodes 9 to 12) hangs off a pivot (5 to 8) turned about the rocket's axis
        a = math.radians(45 + 90 * k)
        nodes.append({"name": f"fin pivot {k}", "rotation": [0, math.sin(a / 2), 0, math.cos(a / 2)],
                      "children": [9 + k]})
    for k in range(4):
        nodes.append({"name": f"fin {k}", "mesh": 2, "translation": [0, -70, 60], "scale": [6, 80, 40]})
    doc = {"asset": {"version": "2.0", "generator": "kinescript examples/make_media.py"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes, "materials": materials,
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    print("wrote", path)


rocket(out / "rocket.glb")
