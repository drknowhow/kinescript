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


def flipbook():
    """Frames for examples/flipbook.json: a walk cycle as a sheet of eight 32 px cells, drawn without anti-aliasing to
    be scaled up pixelated, and a flame as twelve numbered files, flame/flame_01.png to flame_12.png. Both loop: frame
    i is drawn at phase i / n of a turn."""
    import skia

    cell, n = 32, 8
    surf = skia.Surface(cell * n, cell)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    pen = lambda hexa, w: skia.Paint(Color=skia.Color(*bytes.fromhex(hexa[1:])), StrokeWidth=w,  # noqa: E731
                                     Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kSquare_Cap)
    fill = lambda hexa: skia.Paint(Color=skia.Color(*bytes.fromhex(hexa[1:])))  # noqa: E731
    for i in range(n):
        a = i / n * 2 * math.pi
        x0, bob = i * cell + 16, round(abs(math.sin(a)))  # up on the passing pose
        for k, leg in enumerate((math.sin(a), -math.sin(a))):  # the far leg darker
            ang = 0.7 * leg
            c.drawLine(x0, 20 - bob, round(x0 + 10 * math.sin(ang)), round(30 - 10 * (1 - math.cos(ang))) - bob,
                       pen("#9299a8" if k else "#f2f0e8", 3))
        c.drawRect(skia.Rect(x0 - 3, 11 - bob, x0 + 4, 21 - bob), fill("#ff633f"))  # the body
        c.drawLine(x0, 13 - bob, round(x0 - 6 * math.sin(a)), 19 - bob, pen("#f2f0e8", 2))  # the near arm swings
        c.drawRect(skia.Rect(x0 - 4, 2 - bob, x0 + 5, 11 - bob), fill("#f2f0e8"))  # the head
        c.drawRect(skia.Rect(x0 + 1, 4 - bob, x0 + 5, 7 - bob), fill("#b9f45d"))  # its visor, facing right
    surf.makeImageSnapshot().save(str(out / "walk.png"))
    print("wrote", out / "walk.png")

    flame = out / "flame"
    flame.mkdir(exist_ok=True)
    for i in range(12):
        a = i / 12 * 2 * math.pi
        surf = skia.Surface(160, 240)
        c = surf.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        for scale, colour, lean in ((1.0, "#ff633f", 0.0), (0.62, "#ffb347", 0.8), (0.32, "#f2f0e8", 1.6)):
            r = 52 * scale
            tip_x = 80 + (18 * math.sin(a + lean) + 7 * math.sin(2 * a + 1 + lean)) * scale
            tip_y = 228 - r - (130 + 14 * math.sin(3 * a + 0.5 + lean)) * scale
            p = skia.Path()
            p.moveTo(tip_x, tip_y)
            p.cubicTo(tip_x - 6 * scale, tip_y + 60 * scale, 80 - r, 228 - r * 1.4, 80 - r, 228 - r)
            p.arcTo(skia.Rect(80 - r, 228 - 2 * r, 80 + r, 228), 180, -180, False)
            p.cubicTo(80 + r, 228 - r * 1.4, tip_x + 10 * scale, tip_y + 50 * scale, tip_x, tip_y)
            c.drawPath(p, skia.Paint(AntiAlias=True, Color=skia.Color(*bytes.fromhex(colour[1:]))))
        surf.makeImageSnapshot().save(str(flame / f"flame_{i + 1:02d}.png"))
    print("wrote", flame / "flame_01.png", "to flame_12.png")


flipbook()


def vectors():
    """SVG drawings for examples/vector.json and svg.json: an emblem with only a viewBox 120 units across, so its own
    size is 120 px, its pieces named (badge, ring, star, and dots, a <use> of one dot each, SVG 2 href); and a spinner
    as a sheet of eight 48 px cells, the lit dot one step further round in each."""
    star = " ".join(f"{'M' if i == 0 else 'L'}{60 + r * math.sin(i * math.pi / 5):.2f} "
                    f"{62 - r * math.cos(i * math.pi / 5):.2f}" for i, r in enumerate([30, 12.5] * 5)) + " Z"
    dots = "".join(f'<use href="#dot" x="{60 + 50 * math.cos(i * math.pi / 6):.2f}" '
                   f'y="{60 + 50 * math.sin(i * math.pi / 6):.2f}"/>' for i in range(12))
    (out / "emblem.svg").write_text(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">
  <defs>
    <radialGradient id="disc" cx="0.35" cy="0.3" r="0.8">
      <stop offset="0" stop-color="#ff9a78"/><stop offset="1" stop-color="#ff633f"/>
    </radialGradient>
    <circle id="dot" r="2.4" fill="#f2f0e8"/>
  </defs>
  <circle id="badge" cx="60" cy="60" r="44" fill="url(#disc)"/>
  <circle id="ring" cx="60" cy="60" r="38" fill="none" stroke="#0a0c10" stroke-width="1.2" stroke-dasharray="3 2.5"
          opacity="0.45"/>
  <path id="star" d="{star}" fill="#f2f0e8" stroke="#0a0c10" stroke-width="2" stroke-linejoin="round"/>
  <g id="dots">{dots}</g>
</svg>
""")
    cells = "".join(f'  <circle cx="{k * 48 + 24 + 16 * math.sin(i * math.pi / 4):.2f}" '
                    f'cy="{24 - 16 * math.cos(i * math.pi / 4):.2f}" r="{4.5 if i == k else 3}" '
                    f'fill="{"#b9f45d" if i == k else "#9299a8"}"/>\n' for k in range(8) for i in range(8))
    (out / "spinner.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="384" height="48" '
                                     f'viewBox="0 0 384 48">\n{cells}</svg>\n')
    print("wrote", out / "emblem.svg", "and", out / "spinner.svg")


vectors()


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


def arm(path: Path):
    """A skinned, animated glTF binary: a capped tube 150 tall on three joints (at heights 0, 50 and 100), each
    vertex weighted to the one or two joints it is between, with two clips: "wave", a 2 s loop of LINEAR
    rotation keys, and "curl", 1.5 s from straight to curled through CUBICSPLINE keys that ease in and out."""
    rings, seg, r, h = 31, 32, 14.0, 150.0
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    rim = np.stack([np.cos(a), np.zeros(seg), np.sin(a)], 1)
    y = np.linspace(0, h, rings)
    P = [(rim[None] * r + y[:, None, None] * [0, 1, 0]).reshape(-1, 3)]
    N = [np.tile(rim, (rings, 1))]
    i, j = np.arange(rings - 1)[:, None] * seg, np.arange(seg)[None]
    a0, a1 = i + j, i + (j + 1) % seg
    T = [np.stack([a0, a0 + seg, a1], -1).reshape(-1, 3), np.stack([a1, a0 + seg, a1 + seg], -1).reshape(-1, 3)]
    for end, up in ((0.0, -1.0), (h, 1.0)):  # flat caps, faces outward (counter-clockwise seen from outside)
        base = sum(len(p) for p in P)
        P.append(np.r_[[[0, end, 0]], rim * r + [0, end, 0]])
        N.append(np.tile([0, up, 0], (seg + 1, 1)))
        k = np.arange(seg)
        T.append(np.stack([np.full(seg, base), base + 1 + (k + 1) % seg if up > 0 else base + 1 + k,
                           base + 1 + k if up > 0 else base + 1 + (k + 1) % seg], 1))
    P, N, T = np.concatenate(P), np.concatenate(N), np.concatenate(T)
    # The lower joint of each pair gives way to the upper one over 30 px round where they meet.
    s = lambda b: (lambda u: u * u * (3 - 2 * u))(np.clip((P[:, 1] - b + 15) / 30, 0, 1))  # noqa: E731
    w1, w2 = s(50), s(100)
    weights = np.stack([1 - w1, w1 - w2, w2, np.zeros(len(P))], 1)
    joints = np.tile([0, 1, 2, 0], (len(P), 1))
    blob, views, accessors = bytearray(), [], []

    def add(arr, ctype, kind):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes})
        blob.extend(arr.tobytes() + b"\0" * (-arr.nbytes % 4))
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
        accessors.append(acc)
        return len(accessors) - 1

    def quat(deg):  # about z, toward the viewer: the tube bends to the left
        return [0.0, 0.0, math.sin(math.radians(deg) / 2), math.cos(math.radians(deg) / 2)]

    attrs = {"POSITION": add(P.astype("<f4"), 5126, "VEC3"), "NORMAL": add(N.astype("<f4"), 5126, "VEC3"),
             "JOINTS_0": add(joints.astype("<u1"), 5121, "VEC4"), "WEIGHTS_0": add(weights.astype("<f4"), 5126, "VEC4")}
    ibm = np.stack([np.eye(4) - np.pad([[0, 0, 0, 0], [0, 0, 0, b], [0, 0, 0, 0]], ((0, 1), (0, 0)))
                    for b in (0, 50, 100)])  # each joint's bind matrix inverted: down by its height
    samplers, channels = [], []
    clips = []
    t = np.arange(17) / 8  # every 1/8 s over the loop
    for node, angle in ((2, lambda t: 35 * np.sin(np.pi * t)), (3, lambda t: 50 * np.sin(np.pi * t - 0.9))):
        samplers.append({"input": add(t.astype("<f4"), 5126, "SCALAR"),
                         "output": add(np.array([quat(d) for d in angle(t)], "<f4"), 5126, "VEC4")})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": "rotation"}})
    clips.append({"name": "wave", "samplers": samplers, "channels": channels})
    samplers, channels = [], []
    for node, deg in ((2, 70), (3, 85)):  # in-tangent, value, out-tangent per key; flat tangents ease
        keys = np.array([[0] * 4, quat(0), [0] * 4, [0] * 4, quat(deg), [0] * 4], "<f4")
        samplers.append({"input": add(np.array([0, 1.5], "<f4"), 5126, "SCALAR"),
                         "output": add(keys, 5126, "VEC4"), "interpolation": "CUBICSPLINE"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": "rotation"}})
    clips.append({"name": "curl", "samplers": samplers, "channels": channels})
    lin = [(v / 255) ** 2.2 for v in (0xff, 0x63, 0x3f)]  # the brand accent; baseColorFactor is linear
    doc = {"asset": {"version": "2.0", "generator": "kinescript examples/make_media.py"}, "scene": 0,
           "scenes": [{"nodes": [0, 4]}],
           "nodes": [{"name": "rig", "children": [1]}, {"name": "base", "children": [2]},
                     {"name": "elbow", "translation": [0, 50, 0], "children": [3]},
                     {"name": "wrist", "translation": [0, 50, 0]}, {"name": "tube", "mesh": 0, "skin": 0}],
           "skins": [{"name": "arm", "joints": [1, 2, 3],
                      "inverseBindMatrices": add(ibm.transpose(0, 2, 1).reshape(3, 16).astype("<f4"), 5126,
                                                 "MAT4")}],
           "meshes": [{"name": "tube", "primitives": [{"attributes": attrs, "material": 0,
                                                        "indices": add(T.astype("<u2").ravel(), 5123, "SCALAR")}]}],
           "materials": [{"name": "accent", "pbrMetallicRoughness": {"baseColorFactor": [*lin, 1.0],
                                                                     "metallicFactor": 0.0, "roughnessFactor": 0.4}}],
           "animations": clips, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    print("wrote", path)


def figure(path: Path):
    """A skinned, animated glTF binary: a blocky figure 182 tall facing the viewer (+z), a torso, head, arms and
    legs on 11 joints, each limb bending at its elbow or knee, with two loops: "walk" (1 s: legs and arms swing,
    knees bend in the swing, the body bobs twice) and "wave" (2 s: the right arm raised, its forearm waving)."""
    bones = [("hips", None, (0, 92)), ("chest", 0, (0, 128)), ("head", 1, (0, 156)), ("shoulderL", 1, (-25, 150)),
             ("elbowL", 3, (-25, 121)), ("shoulderR", 1, (25, 150)), ("elbowR", 5, (25, 121)),
             ("hipL", 0, (-10, 90)), ("kneeL", 7, (-10, 47)), ("hipR", 0, (10, 90)), ("kneeR", 9, (10, 47))]
    at = {name: k for k, (name, _, _) in enumerate(bones)}
    ease = lambda u: (lambda c: c * c * (3 - 2 * c))(np.clip(u, 0, 1))  # noqa: E731

    def tube(x, y0, y1, r, rings=24, seg=24):
        """A capped tube from y0 up to y1 at x: points, normals and triangles counter-clockwise from outside."""
        a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        rim = np.stack([np.cos(a), np.zeros(seg), np.sin(a)], 1)
        y = np.linspace(y0, y1, rings)
        P = [(rim[None] * r + y[:, None, None] * [0, 1, 0]).reshape(-1, 3) + [x, 0, 0]]
        N = [np.tile(rim, (rings, 1))]
        i, j = np.arange(rings - 1)[:, None] * seg, np.arange(seg)[None]
        a0, a1 = i + j, i + (j + 1) % seg
        T = [np.stack([a0, a0 + seg, a1], -1).reshape(-1, 3), np.stack([a1, a0 + seg, a1 + seg], -1).reshape(-1, 3)]
        for end, up in ((y0, -1.0), (y1, 1.0)):
            base = sum(len(p) for p in P)
            P.append(np.r_[[[x, end, 0]], rim * r + [x, end, 0]])
            N.append(np.tile([0, up, 0], (seg + 1, 1)))
            k = np.arange(seg)
            T.append(np.stack([np.full(seg, base), base + 1 + (k + 1) % seg if up > 0 else base + 1 + k,
                               base + 1 + k if up > 0 else base + 1 + (k + 1) % seg], 1))
        return np.concatenate(P), np.concatenate(N), np.concatenate(T)

    def ball(c, r):
        sp = mesh.unit("sphere")  # y down, wound clockwise on screen: turn both round for glTF
        return sp.P * [1, -1, 1] * r + c, sp.N * [1, -1, 1], sp.T[:, [0, 2, 1]]

    def limb(x, top, bend, bottom, r, upper, lower):
        """A tube from top down to bottom, the upper joint giving way to the lower over 16 round bend."""
        P, N, T = tube(x, bottom, top, r)
        w = ease((bend + 8 - P[:, 1]) / 16)
        return P, N, T, np.stack([1 - w, w], 1), [at[upper], at[lower]]

    pieces = {"suit": [limb(0, 152, 118, 86, 17, "chest", "hips"),
                       limb(-25, 152, 121, 92, 6, "shoulderL", "elbowL"),
                       limb(25, 152, 121, 92, 6, "shoulderR", "elbowR")],
              "legs": [limb(-10, 86, 47, 2, 7.5, "hipL", "kneeL"), limb(10, 86, 47, 2, 7.5, "hipR", "kneeR")],
              "head": [(*ball([0, 170, 0], 14), np.ones((1, 2)) * [1, 0], [at["head"], at["head"]])]}
    blob, views, accessors = bytearray(), [], []

    def add(arr, ctype, kind):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes})
        blob.extend(arr.tobytes() + b"\0" * (-arr.nbytes % 4))
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
        accessors.append(acc)
        return len(accessors) - 1

    prims = []
    for m, name in enumerate(pieces):
        Ps, Ns, Ts, Js, Ws, base = [], [], [], [], [], 0
        for P, N, T, w, js in pieces[name]:
            Ps.append(P)
            Ns.append(N)
            Ts.append(T + base)
            Js.append(np.tile([js[0], js[1], 0, 0], (len(P), 1)))
            Ws.append(np.c_[np.broadcast_to(w, (len(P), 2)), np.zeros((len(P), 2))])
            base += len(P)
        attrs = {"POSITION": add(np.concatenate(Ps).astype("<f4"), 5126, "VEC3"),
                 "NORMAL": add(np.concatenate(Ns).astype("<f4"), 5126, "VEC3"),
                 "JOINTS_0": add(np.concatenate(Js).astype("<u1"), 5121, "VEC4"),
                 "WEIGHTS_0": add(np.concatenate(Ws).astype("<f4"), 5126, "VEC4")}
        prims.append({"attributes": attrs, "material": m,
                      "indices": add(np.concatenate(Ts).astype("<u2").ravel(), 5123, "SCALAR")})
    nodes = []
    for name, parent, (x, y) in bones:
        px, py = (0, 0) if parent is None else bones[parent][2]
        kids = [k for k, b in enumerate(bones) if b[1] == at[name]]
        nodes.append({"name": name, "translation": [x - px, y - py, 0], **({"children": kids} if kids else {})})
    nodes.append({"name": "body", "mesh": 0, "skin": 0})
    ibm = np.tile(np.eye(4), (len(bones), 1, 1))
    ibm[:, 0, 3], ibm[:, 1, 3] = [-b[2][0] for b in bones], [-b[2][1] for b in bones]

    def turn(axis, deg):  # quaternions about an axis, one per angle
        h = np.radians(np.asarray(deg, float)) / 2
        return np.c_[np.outer(np.sin(h), axis), np.cos(h)]

    X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    s = lambda t, f=1, ph=0.0: np.sin(2 * np.pi * f * t + ph)  # noqa: E731
    walk_t, wave_t = np.arange(17) / 16, np.arange(33) / 16
    clips = {"walk": (walk_t, [("hipL", X, -26 * s(walk_t)), ("hipR", X, 26 * s(walk_t)),
                               ("kneeL", X, 40 * np.maximum(0, s(walk_t, ph=np.pi / 2))),
                               ("kneeR", X, 40 * np.maximum(0, -s(walk_t, ph=np.pi / 2))),
                               ("shoulderL", X, 20 * s(walk_t)), ("shoulderR", X, -20 * s(walk_t)),
                               ("elbowL", X, -16 - 6 * s(walk_t)), ("elbowR", X, -16 + 6 * s(walk_t)),
                               ("chest", Y, 7 * s(walk_t)), ("hips", None, 92 + 2.5 * s(walk_t, 2, np.pi / 2))]),
             "wave": (wave_t, [("shoulderR", Z, 145 + 5 * s(wave_t, 0.5)), ("elbowR", Z, 25 * s(wave_t, 1.5)),
                               ("shoulderL", Z, -4 - 3 * s(wave_t, 0.5)), ("chest", Z, 3 * s(wave_t, 0.5)),
                               ("hips", None, 92 + s(wave_t, 0.5))])}
    anims = []
    for clip, (t, chans) in clips.items():
        samplers, channels = [], []
        for name, axis, v in chans:
            values = turn(axis, v) if axis else np.c_[np.zeros(len(t)), v, np.zeros(len(t))]
            samplers.append({"input": add(t.astype("<f4"), 5126, "SCALAR"),
                             "output": add(values.astype("<f4"), 5126, "VEC4" if axis else "VEC3")})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": at[name], "path": "rotation" if axis else "translation"}})
        anims.append({"name": clip, "samplers": samplers, "channels": channels})
    lin = lambda h: [(int(h[k:k + 2], 16) / 255) ** 2.2 for k in (1, 3, 5)] + [1.0]  # noqa: E731
    doc = {"asset": {"version": "2.0", "generator": "kinescript examples/make_media.py"}, "scene": 0,
           "scenes": [{"nodes": [0, len(bones)]}], "nodes": nodes,
           "skins": [{"name": "figure", "joints": list(range(len(bones))),
                      "inverseBindMatrices": add(ibm.transpose(0, 2, 1).reshape(-1, 16).astype("<f4"), 5126, "MAT4")}],
           "meshes": [{"name": "body", "primitives": prims}],
           "materials": [{"name": name, "pbrMetallicRoughness": {"baseColorFactor": lin(c), "metallicFactor": 0.0,
                                                                 "roughnessFactor": rough}}
                         for name, c, rough in (("suit", "#ff633f", 0.45), ("legs", "#2b3242", 0.6),
                                                ("head", "#f2f0e8", 0.35))],
           "animations": anims, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    print("wrote", path)


can(out / "can.glb")
arm(out / "arm.glb")
figure(out / "figure.glb")
