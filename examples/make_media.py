"""Generate the sample footage, music, voice, images and models the examples use (no third-party media)."""
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
    # a click after 50 ms of silence, which the mix skips so the click lands on its cue
    "click.wav": ["-f", "lavfi", "-i", "aevalsrc=if(gte(t\\,0.05)\\,0.8*exp(-60*(t-0.05))*sin(2*PI*1800*t)\\,0):s=48000:d=0.3"],
}
for name, args in jobs.items():
    tail = (["-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"] if name.endswith(".mp4")
            else [] if name.endswith(".wav") else ["-c:a", "aac"])
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


def textures():
    """Images for examples/textures.json, made with numpy and Skia: crate boards, a can label, a globe's map (all
    the way round, top to bottom) and floor tiles."""
    import skia

    def save(px, name):
        skia.Image.fromarray(np.ascontiguousarray(np.clip(px, 0, 255).astype(np.uint8)),
                             skia.kRGB_888x_ColorType).save(str(out / name))
        print("wrote", out / name)

    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:512, 0:512] / 512.0
    board = np.floor(y * 4)  # four boards, each with its own shade and grain
    grain = np.sin((x * 40 + np.sin(y * 9 + board * 2.1) * 1.6 + board * 5) * 3) * 0.5 + 0.5
    shade = 0.82 + 0.18 * rng.random(4)[board.astype(int)]
    wood = np.stack([196, 142, 88], -1) * (shade * (0.82 + 0.18 * grain))[..., None]
    gap = (np.abs(y * 4 - np.round(y * 4)) * 128 < 3)[..., None]
    rim = ((np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)) * 512) < 34)[..., None]
    crate = np.where(rim, wood * 0.72, np.where(gap, wood * 0.35, wood))
    for cx, cy in ((17, 17), (495, 17), (17, 495), (495, 495)):  # a nail in each corner
        crate[(np.hypot(x * 512 - cx, y * 512 - cy) < 6)] = (70, 72, 78)
    save(np.dstack([crate, np.full((512, 512), 255)]), "crate.png")

    surf = skia.Surface(1024, 512)
    c = surf.getCanvas()
    c.clear(skia.Color(0x24, 0x28, 0x32))
    c.drawRect(skia.Rect(0, 150, 1024, 362), skia.Paint(Color=skia.Color(0xFF, 0x63, 0x3F)))
    c.drawRect(skia.Rect(0, 372, 1024, 392), skia.Paint(Color=skia.Color(0xB9, 0xF4, 0x5D)))
    font = skia.Font(skia.Typeface("Arial", skia.FontStyle.Bold()), 150)
    for left in (40, 552):  # twice round the can, so a word faces the camera either way
        c.drawString("KINE", left, 315, font, skia.Paint(Color=skia.Color(0xF2, 0xF0, 0xE8), AntiAlias=True))
    small = skia.Font(skia.Typeface("Arial"), 28)
    for left in (60, 572):
        c.drawString("JSON SCENES IN, VIDEO OUT", left, 110, small, skia.Paint(Color=skia.Color(0x92, 0x99, 0xA8),
                                                                                AntiAlias=True))
    surf.makeImageSnapshot().save(str(out / "label.png"))
    print("wrote", out / "label.png")

    lat, lon = np.mgrid[0:512, 0:1024] / np.array([512.0, 1024.0])[:, None, None]
    th, ph = lon * 2 * np.pi, lat * np.pi
    land = sum(np.sin(th * f + rng.random() * 6.3) * np.sin(ph * g + rng.random() * 6.3) / (f + g)
               for f, g in ((1, 2), (2, 3), (3, 1), (4, 5), (6, 4), (8, 7)))  # smooth all the way round
    sea = np.stack([18 + 30 * lat, 60 + 40 * lat, 120 + 40 * (1 - lat)], -1)
    green = np.stack([90 + 60 * land, 140 + 40 * land, 70 + 0 * land], -1)
    ice = (np.abs(lat - 0.5) > 0.43)[..., None]
    globe = np.where(ice, 236, np.where((land > 0.08)[..., None], green, sea))
    save(np.dstack([globe, np.full(lat.shape, 255)]), "globe.png")

    t = np.mgrid[0:256, 0:256] / 256.0
    check = ((np.floor(t[0] * 2) + np.floor(t[1] * 2)) % 2)[..., None]
    tiles = np.where(check > 0, (56, 62, 76), (44, 49, 61)) + rng.normal(0, 2.5, (256, 256, 1))
    grout = ((np.abs(t * 2 - np.round(t * 2)) * 128 < 2).any(0))[..., None]
    save(np.dstack([np.where(grout, (24, 27, 34), tiles), np.full((256, 256), 255)]), "tiles.png")


textures()


def can(path: Path):
    """A drinks can as a glTF binary for examples/product.json: the body's base colour texture is label.png,
    embedded in the file and read through TEXCOORD_0, under a steel lid with a rolled rim and a ring pull."""
    blob, views, accessors = bytearray(), [], []

    def view(b: bytes, target=None) -> int:
        views.append(dict({"buffer": 0, "byteOffset": len(blob), "byteLength": len(b)},
                          **({"target": target} if target else {})))
        blob.extend(b + b"\0" * (-len(b) % 4))
        return len(views) - 1

    def add(arr, ctype, kind, target):
        acc = {"bufferView": view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
        accessors.append(acc)
        return len(accessors) - 1

    def prim(shape, material, detail, tube=0.35, uv=False):
        m = mesh.unit(shape, detail, tube)  # y down; files are y up with faces wound the other way
        attrs = {"POSITION": add((m.P * (1, -1, 1)).astype("<f4"), 5126, "VEC3", 34962),
                 "NORMAL": add((m.N * (1, -1, 1)).astype("<f4"), 5126, "VEC3", 34962)}
        if uv:  # glTF coordinates, like the mesh's, run right and down across the image
            attrs["TEXCOORD_0"] = add(m.U.astype("<f4"), 5126, "VEC2", 34962)
        return {"name": shape, "primitives": [{"attributes": attrs, "material": material,
                                               "indices": add(m.T[:, [0, 2, 1]].astype("<u2").ravel(), 5123,
                                                              "SCALAR", 34963)}]}

    meshes = [prim("cylinder", 0, "high", uv=True), prim("cylinder", 1, "medium"), prim("torus", 1, "medium", 0.08),
              prim("torus", 1, "low", 0.25)]
    flat = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]  # a torus laid flat round the can's axis
    nodes = [{"name": "can", "children": [1, 2, 3, 4]},
             {"name": "body", "mesh": 0, "scale": [33, 120, 33]},
             {"name": "lid", "mesh": 1, "translation": [0, 61, 0], "scale": [31, 2, 31]},
             {"name": "top rim", "mesh": 2, "translation": [0, 61, 0], "rotation": flat, "scale": [31.5, 31.5, 31.5]},
             {"name": "ring pull", "mesh": 3, "translation": [0, 62.6, 10], "rotation": flat, "scale": [7, 7, 2]}]
    image = view((out / "label.png").read_bytes())
    doc = {"asset": {"version": "2.0", "generator": "kinescript examples/make_media.py"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes,
           "materials": [{"name": "label", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                                    "metallicFactor": 0.0, "roughnessFactor": 0.3}},
                         {"name": "steel", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.62, 0.66, 1.0],
                                                                    "metallicFactor": 1.0, "roughnessFactor": 0.25}}],
           "textures": [{"source": 0, "sampler": 0}], "samplers": [{}],
           "images": [{"bufferView": image, "mimeType": "image/png"}],
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    print("wrote", path)


can(out / "can.glb")
