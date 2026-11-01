"""Stress scenes: each one pushes one dimension (layer count, nesting, text, expressions, keyframes, shaders,
resolution, length, frame rate, motion blur, alpha, footage, cuts, sound, models, captions, odd sizes, output
formats) well past what the examples do. Every scene is generated here, checked, rendered and its output probed,
each in its own process so a crash or a leak in one cannot spoil the next.

    python bench/stress.py                       # every scene
    python bench/stress.py layers-2000 uhd-4k    # some of them
    python bench/stress.py --list                # names and what each one stresses
    python bench/stress.py --keep out/stress     # keep the scene JSON and the rendered files
    python bench/stress.py --json results.json   # also write the rows as JSON

Columns: check = errors/warnings from kine check; frames ok = the output holds exactly the frames the scene
has (ffprobe packet count, or PNG files); peak MB = the largest working set of this process plus the render
processes it started (ffmpeg not counted), summed, so it is an upper bound on memory in use at once.
Wall times depend on the machine and on what else runs on it; say which machine ran them when quoting."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "examples" / "media"
WORK = Path(tempfile.gettempdir()) / "kinescript-stress"  # where scenes that need their own files write them
WORDS = ("motion", "written", "checked", "rendered", "frames", "scene", "layer", "cue", "agent", "render")


def _m(name: str) -> str:
    return (MEDIA / name).as_posix()


def _scene(layers: list, w: int = 1920, h: int = 1080, fps: float = 30, dur: float | str = 4, bg="#0e0f13",
           **top) -> dict:
    return {"canvas": {"w": w, "h": h, "fps": fps, "duration": dur, "bg": bg}, **top, "layers": layers}


def _sentence(n: int, words: int) -> str:
    return " ".join(WORDS[(n * 7 + k * 3) % len(WORDS)] for k in range(words))


def layers_2000() -> dict:
    out = []
    for n in range(2000):
        x, y = 40 + (n % 50) * 37, 60 + (n // 50) * 24
        L = {"id": f"s{n}", "type": "circle" if n % 2 else "rect", "x": x, "y": [[0, y], [4, y + 30, "inOutSine"]],
             "fill": f"#{(n * 37) % 256:02x}{(n * 91) % 256:02x}c8", "at": round(n * 0.001, 3), "enter": "fade"}
        L.update({"r": 9} if n % 2 else {"w": 18, "h": 14, "radius": 3})
        out.append(L)
    return _scene(out)


def groups_deep() -> dict:
    stacks = []
    for s in range(10):
        node = {"id": f"leaf{s}", "type": "text", "text": f"depth {s}", "size": 28}
        for d in range(30):
            node = {"id": f"g{s}-{d}", "type": "group", "rotate": f"=sin(t + {d})*3", "scale": 0.99,
                    "children": [node, {"id": f"r{s}-{d}", "type": "rect", "w": 260 - d * 6, "h": 160 - d * 4,
                                        "stroke": "#4a7dff", "strokeWidth": 1}]}
            if d % 5 == 0:
                node.update({"w": 280, "h": 180, "clip": True, "radius": 12})
        node.update({"x": 200 + (s % 5) * 380, "y": 300 + (s // 5) * 480})
        stacks.append(node)
    return _scene(stacks)


def text_split() -> dict:
    return _scene([{"id": f"t{n}", "type": "text", "text": _sentence(n, 14).replace("scene", "**scene**"),
                    "x": "50%", "y": 30 + n * 26, "size": 22, "split": "chars", "at": n * 0.02,
                    "enter": {"name": "rise+blur", "dur": 0.5, "stagger": 0.01, "dist": 20},
                    "unitY": "=sin(T*4 + i*0.3)*3"} for n in range(40)])


def text_wrap() -> dict:  # each paragraph wraps to about 20 lines
    return _scene([{"id": f"p{n}", "type": "text", "text": _sentence(n, 200), "x": 40 + (n % 4) * 470,
                    "y": 60 + (n // 4) * 200, "anchor": "topleft", "size": 9, "maxWidth": 440, "at": n * 0.1,
                    "enter": {"name": "type", "dur": 2.5}, "caret": True} for n in range(20)])


def expressions() -> dict:
    return _scene([{"id": f"e{n}", "type": "circle", "r": f"=6 + 3*sin(t*3 + {n})",
                    "x": f"=960 + cos(t*{1 + n % 7} + {n})*{100 + n}", "y": f"=540 + sin(t*{1 + n % 5} + {n})*{50 + n % 400}",
                    "opacity": f"=0.5 + 0.5*sin(t + {n})", "fill": "$accent"} for n in range(600)])


def keys_dense() -> dict:
    out = []
    for n in range(300):
        keys = [[round(k * 4 / 120, 4), 100 + (n * 13 + k * 29) % 1700] for k in range(121)]
        out.append({"id": f"k{n}", "type": "rect", "w": 16, "h": 16, "x": keys,
                    "y": [[round(k * 4 / 120, 4), 80 + (n * 3 + k * 17) % 900, "inOutCubic"] for k in range(121)],
                    "rotate": [[round(k * 4 / 120, 4), k * 9] for k in range(121)],
                    "fill": [[round(k * 4 / 30, 4), ("#ff5a36", "#4a7dff", "#e8e2d0")[(n + k) % 3]] for k in range(31)]})
    return _scene(out)


def along_100() -> dict:  # 60 darts on 6 paths that draw on, and 40 bars round closed paths of their own
    out = []
    for n in range(6):
        y = 140 + n * 150
        out.append({"id": f"route{n}", "type": "path", "d": f"M80 {y} C560 {y - 220} 1360 {y + 220} 1840 {y}",
                    "stroke": "#2c3440", "strokeWidth": 4, "draw": [[0, 0], [2, 1]]})
        out += [{"id": f"dart{n}-{k}", "type": "polygon", "sides": 3, "r": 12, "rotate": 90, "fill": "$accent",
                 "along": {"d": f"#route{n}", "p": [[k * 0.15, 0], [k * 0.15 + 3, 1, "linear"]], "orient": True}}
                for k in range(10)]
    for n in range(40):
        cx, cy = 140 + (n % 10) * 182, 1000 - (n // 10) * 40
        out.append({"id": f"bar{n}", "type": "rect", "w": 18, "h": 6, "fill": "#e8e2d0",
                    "along": {"d": f"M{cx - 60} {cy} a60 18 0 1 0 120 0 a60 18 0 1 0 -120 0 Z",
                              "p": f"=t * 0.4 + {n / 40}", "orient": True}})
    return _scene(out)


def strokes_300() -> dict:  # 100 dashed paths that march and draw on, 100 stars whose corners round, 100 tabs
    out = []
    for n in range(100):
        x, y = 60 + (n % 10) * 186, 50 + (n // 10) * 34
        out.append({"id": f"dash{n}", "type": "path", "d": f"M{x} {y} q40 -24 80 0 t80 0", "stroke": "#e8e2d0",
                    "strokeWidth": 3, "dash": [10, 6], "dashOffset": f"=t * -48 + {n}", "draw": [[0, 0], [2, 1]]})
        out.append({"id": f"star{n}", "type": "polygon", "x": x + 40, "y": 420 + (n // 10) * 34, "sides": 5,
                    "r": 15, "inner": 0.5, "fill": "$accent", "corner": [[0, 0], [4, 6, "linear"]]})
        out.append({"id": f"tab{n}", "type": "rect", "x": x + 110, "y": 420 + (n // 10) * 34, "w": 60, "h": 24,
                    "fill": "#4a7dff", "radius": [[0, [12, 12, 0, 0]], [4, [0, 0, 12, 12]]]})
    return _scene(out)


def rough_200() -> dict:  # 100 outlined cards boiling 8 times a second, and 100 rough lines that draw on
    out = []
    for n in range(100):
        x, y = 100 + (n % 10) * 180, 70 + (n // 10) * 46
        out.append({"id": f"card{n}", "type": "rect", "x": x, "y": y, "w": 120, "h": 34, "radius": 6,
                    "fill": "#ffd25a", "stroke": "#14171c", "strokeWidth": 3,
                    "rough": {"amount": 2.5, "seed": n, "boil": 8}})
        out.append({"id": f"scrawl{n}", "type": "line", "points": [[x - 60, y + 560], [x, y + 540], [x + 60, y + 560]],
                    "stroke": "#e8e2d0", "strokeWidth": 3, "rough": {"amount": 2, "length": 8, "seed": n},
                    "draw": [[0, 0], [3, 1]]})
    return _scene(out)


def combine_60() -> dict:  # 60 badges with a word cut out, half of them with a hole moving through as well
    out = []
    for n in range(60):
        kids = [{"id": f"plate{n}", "type": "rect", "w": 240, "h": 96, "radius": 24},
                {"id": f"word{n}", "type": "text", "text": WORDS[n % len(WORDS)].upper(), "size": 40, "weight": 800}]
        if n % 2:
            kids.append({"id": f"hole{n}", "type": "circle", "r": 22, "y": 30,
                         "x": [[0, -110], [4, 110, "linear"]]})
        out.append({"id": f"badge{n}", "type": "group", "x": 260 + (n % 6) * 280, "y": 110 + (n // 6) * 96,
                    "combine": "subtract", "fill": "$accent", "children": kids})
    return _scene(out)


def morph_200() -> dict:  # 200 paths morphing triangle, circle, star and back, a tenth of a second apart
    tri, disc = "M0 -30 L30 22 L-30 22 Z", "M0 -30 A30 30 0 1 1 -0.1 -30 Z"
    star = "M0 -32 L8 -10 L30 -10 L12 4 L19 26 L0 13 L-19 26 L-12 4 L-30 -10 L-8 -10 Z"
    out = []
    for n in range(200):
        a = round((n % 20) * 0.05 + (n // 20) * 0.02, 3)
        out.append({"id": f"m{n}", "type": "path", "x": 150 + (n % 20) * 85, "y": 130 + (n // 20) * 88,
                    "fill": ("#ff5a36", "#4a7dff", "#e8e2d0")[n % 3],
                    "d": [[a, tri], [a + 1, disc], [a + 2, star], [a + 3, tri]]})
    return _scene(out)


def masks_50() -> dict:  # 50 masked layers: 25 gradient cards seen through words, 25 through a moving lens
    out = []
    for n in range(50):
        x, y = 190 + (n % 5) * 385, 100 + (n // 5) * 98
        if n % 2:
            out.append({"id": f"lens{n}", "type": "circle", "r": 40, "y": y,
                        "x": [[0, x - 150], [4, x + 150, "inOutSine"]]})
            card = {"id": f"card{n}", "type": "rect", "x": x, "y": y, "w": 340, "h": 84, "radius": 12,
                    "mask": {"layer": f"lens{n}", "feather": 16},
                    "fill": {"angle": 90, "stops": ["#ff5a36", "#4a7dff"]}}
        else:
            out.append({"id": f"word{n}", "type": "text", "text": WORDS[n % len(WORDS)].upper(), "x": x, "y": y,
                        "size": 64, "weight": 800})
            card = {"id": f"card{n}", "type": "rect", "x": x, "y": y, "w": 340, "h": 84, "mask": f"word{n}",
                    "fill": [[0, {"angle": 0, "stops": ["#ffd166", "#ef476f"]}],
                             [4, {"angle": 360, "stops": ["#ffd166", "#ef476f"]}, "linear"]]}
        out.append(card)
    return _scene(out)


def glow_text() -> dict:  # a glowing title over lines that glow, are outlined, shaded inside and displaced
    out = [{"id": "title", "type": "text", "text": "GLOW", "x": "50%", "y": 260, "size": 220, "weight": 800,
            "fill": "#ffe1d6", "glow": {"color": "#ff633f", "size": [[0, 16], [2, 48, "inOutSine"], [4, 16, "inOutSine"]]},
            "outline": {"width": 3, "color": "#ff633f"}}]
    for n in range(12):
        x, y, kind = 330 + (n % 3) * 630, 520 + (n // 3) * 140, ("glow", "outline", "innerShadow", "displace")[n % 4]
        layer = {"id": f"line{n}", "type": "text", "text": _sentence(n, 3).upper(), "x": x, "y": y, "size": 56,
                 "weight": 800, "fill": "#e8ecf2"}
        if kind == "innerShadow":
            layer = {"id": f"card{n}", "type": "rect", "x": x, "y": y, "w": 520, "h": 100, "radius": 18,
                     "fill": "#2b3240", "innerShadow": {"y": 6, "blur": 14, "color": "#000000c0"}}
        else:
            layer[kind] = {"glow": {"size": 20}, "outline": {"width": 4, "color": "#4a7dff"},
                           "displace": {"amount": 6, "scale": 40, "speed": 1, "seed": n}}[kind]
        out.append(layer)
    return _scene(out)


def _drawing_svg(n: int = 500) -> str:
    """An svg of n shapes in a grid, 1920x1080: circles, outlined rounded rects, hexagons filled with one of three
    gradients, open curves and turned squares in groups of their own. Written to WORK; returns its path."""
    pal = ("#ff633f", "#b9f45d", "#f2f0e8", "#9299a8", "#7b5cff", "#242832")
    out = []
    for i in range(n):
        x, y, col = 40 + (i % 25) * 75, 40 + (i // 25) * 50, pal[i % len(pal)]
        kind = i % 5
        if kind == 0:
            out.append(f'<circle cx="{x}" cy="{y}" r="{12 + i % 9}" fill="{col}"/>')
        elif kind == 1:
            out.append(f'<rect x="{x - 15}" y="{y - 12}" width="30" height="24" rx="5" fill="{col}" '
                       f'stroke="#0a0c10" stroke-width="2"/>')
        elif kind == 2:
            pts = " ".join(f"{x + 16 * math.cos(a * math.pi / 3):.1f},{y + 16 * math.sin(a * math.pi / 3):.1f}"
                           for a in range(6))
            out.append(f'<polygon points="{pts}" fill="url(#g{i % 3})"/>')
        elif kind == 3:
            out.append(f'<path d="M{x - 18} {y} C {x - 6} {y - 20} {x + 6} {y + 20} {x + 18} {y}" fill="none" '
                       f'stroke="{col}" stroke-width="3"/>')
        else:
            out.append(f'<g transform="translate({x} {y}) rotate({i * 7})"><rect x="-10" y="-10" width="20" '
                       f'height="20" fill="{col}"/></g>')
    grads = "".join(f'<linearGradient id="g{j}"><stop offset="0" stop-color="{pal[j]}"/>'
                    f'<stop offset="1" stop-color="{pal[j + 2]}"/></linearGradient>' for j in range(3))
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"drawing{n}.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">'
                    f'<defs>{grads}</defs>{"".join(out)}</svg>', encoding="utf-8")
    return path.as_posix()


def svg_500() -> dict:  # an svg layer of 500 pieces standing still: drawn once into a picture, then played back
    return _scene([{"id": "drawing", "type": "svg", "src": _drawing_svg(), "w": 1920, "h": 1080}])


def _example(name: str) -> dict:
    return json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def logo() -> dict:
    return _example("logo")


def kinetic() -> dict:
    return _example("kinetic")


def explainer() -> dict:
    return _example("explainer")


def counters() -> dict:
    return _scene([{"id": f"c{n}", "type": "counter", "x": 120 + (n % 10) * 180, "y": 80 + (n // 10) * 100,
                    "size": 40, "value": {"from": 0, "to": 10 ** (1 + n % 6), "dur": 3.5}, "format": "{:,.0f}"}
                   for n in range(100)])


def shaders() -> dict:
    blends = ("normal", "screen", "add", "overlay")
    return _scene([{"id": p, "type": "shader", "preset": p, "opacity": 1 if b == "normal" else 0.6, "blend": b}
                   for p, b in zip(("mesh", "aurora", "grid", "score"), blends)],
                  grade={"vignette": 0.5, "grain": 0.08, "contrast": 1.1})


def uhd_4k() -> dict:
    return _scene([{"id": "bg", "type": "shader", "preset": "aurora"},
                {"id": "title", "type": "text", "text": "Four **thousand** pixels", "x": "50%", "y": "42%",
                 "size": 280, "weight": 800, "split": "chars", "enter": {"name": "rise+blur", "dur": 0.8, "stagger": 0.03}},
                {"id": "sub", "type": "text", "text": _sentence(1, 12), "x": "50%", "y": "60%", "size": 80,
                 "at": 0.5, "enter": "rise"},
                {"id": "ring", "type": "circle", "x": "50%", "y": "50%", "r": 900, "stroke": "$accent",
                 "strokeWidth": 12, "fill": "transparent", "draw": [[0, 0], [3, 1, "inOutCubic"]], "shadow": {"blur": 30}}],
               w=3840, h=2160, grade={"vignette": 0.4, "grain": 0.05})


def long_120s() -> dict:
    return _scene([{"id": "bg", "type": "rect", "x": "50%", "y": "50%", "w": 1920, "h": 1080, "fill": "#14161d"},
                   *({"id": f"o{n}", "type": "circle", "r": 30, "fill": "$accent2",
                      "x": f"=960 + cos(t*0.{n + 3} + {n})*700", "y": f"=540 + sin(t*0.{n + 2})*400"} for n in range(6)),
                   {"id": "clock", "type": "counter", "x": "50%", "y": "50%", "size": 120,
                    "value": "=t", "format": "{:.2f}"}], dur=120)


def _hello(**canvas) -> dict:
    spec = json.loads((ROOT / "examples" / "hello.json").read_text(encoding="utf-8"))
    spec["canvas"].update(canvas)
    return spec


def fps_120() -> dict:
    return _hello(fps=120)


def fps_23976() -> dict:
    return _hello(fps=23.976)


def blur_alpha() -> dict:
    spec = _hello(bg="transparent")
    spec["motionBlur"] = {"samples": 16, "shutter": 0.5}
    return spec


def alpha_webm() -> dict:
    return _hello(bg="transparent")


def video_grid() -> dict:
    out = []
    for n in range(9):
        src = ("cells.mp4", "fractal.mp4")[n % 2]
        L = {"id": f"v{n}", "type": "video", "src": _m(src), "x": 320 + (n % 3) * 640, "y": 180 + (n // 3) * 360,
             "w": 620, "h": 345, "radius": 18, "fit": "cover", "trim": [0.2 * n, 0.2 * n + 3], "loop": True,
             "speed": (0.5, 1, 1.5, 2)[n % 4], "volume": 0.2}
        if n % 3 == 0:
            L.update({"zoom": [[0, 1], [6, 1.6, "inOutSine"]], "focus": [0.3, 0.6]})
        out.append(L)
    return _scene(out, dur=8)


def sequence_60() -> dict:
    clips = [{"src": _m(("cells.mp4", "fractal.mp4")[n % 2]), "trim": [0.1 * (n % 30), 0.1 * (n % 30) + 0.6]}
             for n in range(60)]
    return _scene([{"id": "cut", "type": "sequence", "clips": clips, "transition": {"type": "slide", "dur": 0.2}},
                   {"id": "note", "type": "text", "text": "60 cuts", "x": "50%", "y": "88%", "size": 48, "fixed": True}],
                  dur="auto")


def audio_40() -> dict:
    audio = [{"id": "bed", "role": "music", "src": _m("pad.m4a"), "loop": True, "until": "end", "fadeIn": 0.5,
              "fadeOut": 1},
             {"id": "vo", "role": "voice", "src": _m("voice.wav"), "at": 1.0}]
    audio += [{"role": "sfx", "src": _m("click.wav"), "at": round(0.25 + n * 0.2, 2), "pan": round(math.sin(n), 2),
               "gain": -6 - n % 6} for n in range(40)]
    return _scene([{"id": f"b{n}", "type": "rect", "x": 100 + n * 43, "y": "50%", "w": 30, "h": 200, "fill": "$accent",
                    "at": round(0.25 + n * 0.2, 2), "dur": 0.2} for n in range(40)],
                  dur=9, audio=audio, mix={"loudness": -14, "peak": -1, "duck": {"depth": 10}})


def models_30() -> dict:
    shapes = ("cube", "sphere", "cylinder", "cone", "torus")
    out = [{"id": f"m{n}", "type": "model", "shape": shapes[n % 5], "detail": "high", "r": 60, "w": 110, "h": 120,
            "tube": 22, "x": 160 + (n % 6) * 320, "y": 150 + (n // 6) * 200, "z": -100 * (n % 4),
            "rotateX": [[0, 10 * n], [4, 10 * n + 180]], "rotateY": [[0, 0], [4, 270]],
            "fill": ("#ff5a36", "#4a7dff", "#e8e2d0")[n % 3], "shine": 0.6, "metal": 0.3} for n in range(30)]
    out += [{"id": f"rocket{n}", "type": "model", "src": _m("rocket.glb"), "x": 480 + n * 480, "y": "85%", "scale": 0.6,
             "rotateY": [[0, 0], [4, 360]]} for n in range(3)]
    return _scene(out, camera={"rotateY": [[0, -10], ["end", 10]]}, light="studio")


def stage_16() -> dict:  # 12 spinning rings and 4 turning cards that pass through them, each merged on its own
    kids = [{"id": f"ring{n}", "type": "model", "shape": "torus", "r": 120, "tube": 22, "detail": "high",
             "x": -750 + (n % 6) * 300, "y": -160 + (n // 6) * 320, "rotateY": 90 * (n % 2),
             "rotateX": [[0, 0], [4, 90 + 30 * n]], "fill": ("#ff5a36", "#4a7dff", "#e8e2d0")[n % 3], "shine": 0.6}
            for n in range(12)]
    kids += [{"id": f"card{n}", "type": "rect", "x": -600 + n * 400, "w": 260, "h": 700, "radius": 24,
              "fill": "#f2f0e8", "opacity": 0.9, "rotateY": [[0, -40 + 20 * n], [4, 40 - 20 * n]]} for n in range(4)]
    return _scene([{"id": "stage", "type": "group", "x": "50%", "y": "50%", "sort": "pixel", "children": kids}],
                  camera={"rotateY": [[0, -14], ["end", 14]], "rotateX": [[0, 10], ["end", -6]]}, light="studio")


def _globe_obj(lat: int = 160, lon: int = 320) -> str:
    """A sphere of 2 x lat x lon triangles (102,400 by default) with normals and coordinates, as an .obj whose .mtl
    maps globe.png onto it. Written to WORK; returns its path."""
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy(MEDIA / "globe.png", WORK / "globe.png")
    (WORK / "globe100k.mtl").write_text("newmtl map\nKd 1 1 1\nNs 300\nmap_Kd globe.png\n", encoding="utf-8")
    lines = ["mtllib globe100k.mtl", "usemtl map"]
    for i in range(lat + 1):
        th = math.pi * i / lat
        for j in range(lon + 1):
            ph = 2 * math.pi * j / lon
            x, y, z = math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)
            lines += [f"v {x:.6f} {y:.6f} {z:.6f}", f"vn {x:.6f} {y:.6f} {z:.6f}", f"vt {j / lon:.6f} {1 - i / lat:.6f}"]
    for i in range(lat):
        for j in range(lon):
            a, b = i * (lon + 1) + j + 1, (i + 1) * (lon + 1) + j + 1
            lines += [f"f {a}/{a}/{a} {b + 1}/{b + 1}/{b + 1} {b}/{b}/{b}",
                      f"f {a}/{a}/{a} {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1}"]
    path = WORK / "globe100k.obj"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.as_posix()


def textured_100k() -> dict:  # one textured model of 102,400 triangles, turning, cut for its texture every frame
    return _scene([{"id": "globe", "type": "model", "src": _globe_obj(), "x": "50%", "y": "50%", "h": 820,
                    "rotateY": [[0, 0], [4, -180]], "rotate": 18, "shine": 0.5}],
                  light={"preset": "studio", "follow": "world", "azimuth": -35, "elevation": 30}, env="studio")


def dof_stage() -> dict:  # a stage with a floor taking shadows, models and cards in depth, and focus pulls
    shapes = ("cube", "sphere", "cylinder", "cone", "torus")
    kids = [{"id": "floor", "type": "rect", "y": 200, "w": 3000, "h": 2400, "radius": 40, "rotateX": 90,
             "fill": "#2b3242"}]
    kids += [{"id": f"m{n}", "type": "model", "shape": shapes[n % 5], "detail": "high", "r": 90, "w": 170, "h": 200,
              "tube": 30, "x": -640 + (n % 5) * 320, "y": 60, "z": 350 - (n // 5) * 800,
              "rotateY": [[0, 30 * n], [4, 30 * n + 120]], "fill": ("#ff633f", "#4a7dff", "#e8e2d4")[n % 3],
              "shine": 0.6} for n in range(10)]
    kids += [{"id": f"card{n}", "type": "rect", "x": -500 + n * 500, "y": -180, "z": -1400, "w": 400, "h": 260,
              "radius": 24, "fill": "#363e51"} for n in range(3)]
    return _scene([{"id": "stage", "type": "group", "x": "50%", "y": "50%", "sort": "pixel",
                    "shadows": {"opacity": 0.45, "soft": 3}, "children": kids}],
                  camera={"rotateX": 16, "rotateY": [[0, -10], ["end", 10]], "aperture": 30,
                          "focus": [[0.3, "m2"], [1.8, "m7", "inOutSine"], [2.6, "m7"], [3.8, "card1", "inOutSine"]]},
                  light={"preset": "studio", "follow": "world", "azimuth": -40, "elevation": 35})


def _snake_gltf(joints: int = 24, rings: int = 289, seg: int = 40) -> str:
    """A tube 600 long on a chain of `joints` joints, each vertex weighted to the two it lies between, with a 2 s
    "slither" clip that bends each joint a little later than the one before (33 LINEAR rotation keys a joint):
    rings x seg vertices, 11,560 and 23,040 triangles by default. Written to WORK as a .gltf; returns its path."""
    import base64
    import numpy as np
    length, r, step = 600.0, 18.0, 600.0 / joints
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    rim = np.stack([np.cos(a), np.zeros(seg), np.sin(a)], 1)
    y = np.linspace(0, length, rings)
    P = (rim[None] * r + y[:, None, None] * [0, 1, 0]).reshape(-1, 3)
    N = np.tile(rim, (rings, 1))
    i, j = np.arange(rings - 1)[:, None] * seg, np.arange(seg)[None]
    a0, a1 = i + j, i + (j + 1) % seg
    T = np.r_[np.stack([a0, a0 + seg, a1], -1).reshape(-1, 3), np.stack([a1, a0 + seg, a1 + seg], -1).reshape(-1, 3)]
    u = P[:, 1] / step
    j0 = np.minimum(np.floor(u), joints - 1).astype(int)
    f = np.clip(u - j0, 0, 1)
    f = np.where(j0 + 1 < joints, f * f * (3 - 2 * f), 0.0)
    jw = np.stack([j0, np.minimum(j0 + 1, joints - 1), np.zeros_like(j0), np.zeros_like(j0)], 1)
    ww = np.stack([1 - f, f, np.zeros_like(f), np.zeros_like(f)], 1)
    blob, views, accessors = bytearray(), [], []

    def add(arr, ctype, kind):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes})
        blob.extend(arr.tobytes() + b"\0" * (-arr.nbytes % 4))
        acc = {"bufferView": len(views) - 1, "componentType": ctype, "count": len(arr), "type": kind}
        if kind == "VEC3":
            acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
        accessors.append(acc)
        return len(accessors) - 1

    attrs = {"POSITION": add(P.astype("<f4"), 5126, "VEC3"), "NORMAL": add(N.astype("<f4"), 5126, "VEC3"),
             "JOINTS_0": add(jw.astype("<u1"), 5121, "VEC4"), "WEIGHTS_0": add(ww.astype("<f4"), 5126, "VEC4")}
    ibm = np.tile(np.eye(4), (joints, 1, 1))
    ibm[:, 1, 3] = -np.arange(joints) * step
    axis = np.array([0.35, 0.0, 1.0]) / np.hypot(0.35, 1.0)
    t = np.arange(33) / 16
    samplers, channels = [], []
    for n in range(1, joints):
        half = np.radians(14 * np.sin(np.pi * t - 0.5 * n)) / 2
        q = np.c_[axis[None] * np.sin(half)[:, None], np.cos(half)]
        samplers.append({"input": add(t.astype("<f4"), 5126, "SCALAR"), "output": add(q.astype("<f4"), 5126, "VEC4")})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": n, "path": "rotation"}})
    nodes = [{"name": f"j{n}", **({"translation": [0, step, 0]} if n else {}),
              **({"children": [n + 1]} if n + 1 < joints else {})} for n in range(joints)]
    nodes.append({"name": "tube", "mesh": 0, "skin": 0})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, joints]}], "nodes": nodes,
           "skins": [{"joints": list(range(joints)),
                      "inverseBindMatrices": add(ibm.transpose(0, 2, 1).reshape(-1, 16).astype("<f4"), 5126, "MAT4")}],
           "meshes": [{"primitives": [{"attributes": attrs, "material": 0,
                                       "indices": add(T.astype("<u4").ravel(), 5125, "SCALAR")}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.18, 0.55, 1.0, 1.0], "metallicFactor": 0.2,
                                                   "roughnessFactor": 0.35}}],
           "animations": [{"name": "slither", "samplers": samplers, "channels": channels}],
           "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(blob),
                         "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]}
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "snake.gltf"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path.as_posix()


def skinned() -> dict:  # 6 skinned tubes of 24 joints and 23,040 triangles each and 12 arms, posed every frame
    src = _snake_gltf()
    out = [{"id": f"snake{n}", "type": "model", "src": src, "h": 640, "x": 260 + n * 280, "y": "42%",
            "rotateY": 30 * n, "animation": {"name": "slither", "start": 0.3 * n, "speed": 0.8 + 0.1 * n}}
           for n in range(6)]
    out += [{"id": f"arm{n}", "type": "model", "src": _m("arm.glb"), "h": 180, "x": 120 + n * 152, "y": "86%",
             "animation": {"name": ("wave", "curl")[n % 2], "start": 0.1 * n}} for n in range(12)]
    return _scene(out, light="studio", camera={"rotateY": [[0, -8], ["end", 8]]})


def repeat_1000() -> dict:  # one model of 1000 cubes, each moved, turned and coloured by expressions every frame
    return _scene([{"id": "grid", "type": "model", "shape": "cube", "w": 26, "detail": "low", "x": "50%", "y": "55%",
                    "edges": {"width": 1, "angle": 40}, "rotateY": [[0, -20], ["end", 20]],
                    "repeat": {"count": 1000, "x": "=(i % 40 - 19.5) * 40", "z": "=(floor(i / 40) - 12) * 40",
                               "y": "=sin(t * 3 + (i % 40) * 0.3 + floor(i / 40) * 0.2) * 40",
                               "rotateY": "=t * 90 + i * 7",
                               "fill": "=hsl((i % 40) * 9 + t * 45, 0.7, 0.55)"}}],
                  camera={"rotateX": 32}, light="studio")


def repeat_10k() -> dict:  # one circle's 10,000 copies, each moved by two expressions every frame
    return _scene([{"id": "swarm", "type": "circle", "r": 5, "x": "50%", "y": "50%",
                    "repeat": {"count": 10000,
                               "x": "=cos(i * 2.39996 + t * (0.2 + (i % 7) * 0.05)) * sqrt(i) * 5.2",
                               "y": "=sin(i * 2.39996 + t * (0.2 + (i % 7) * 0.05)) * sqrt(i) * 5.2",
                               "fill": "=hsl(i * 0.036, 0.7, 0.6)"}}])


def particles_5k() -> dict:  # 5,000 alive from the first frame: a fountain of stars spun, shrunk and faded over life
    return _scene([{"id": "fountain", "type": "particles", "rate": 2500, "life": 2, "prewarm": True, "x": "50%",
                    "y": "92%", "speed": [600, 1100], "angle": [-115, -65], "gravity": 700, "drag": 0.4,
                    "size": [10, 22], "shape": "star", "spin": [-360, 360], "turbulence": 20,
                    "fill": ["#ff633f", "#b9f45d", "#f2f0e8", "#4a7dff", "#ffd166"],
                    "sizeOverLife": [[0, 0.4], [0.1, 1], [1, 0.6]], "opacityOverLife": [[0, 1], [0.8, 1], [1, 0]]}])


def captions_600() -> dict:
    cues = [[round(n * 0.05, 2), round(n * 0.05 + 0.05, 2), _sentence(n, 3)] for n in range(600)]
    return _scene([{"id": "cc", "type": "captions", "mode": "karaoke", "cues": cues},
                   {"id": "pop", "type": "captions", "mode": "pop", "y": "20%", "cues": cues[::3]}], dur=30)


def odd_size() -> dict:
    return _hello(w=1365, h=767)


def gif_out() -> dict:
    return _hello()


def png_seq() -> dict:
    return _hello(duration=1)


def still_8k() -> dict:
    s = uhd_4k()
    s["canvas"].update(w=7680, h=4320)
    return s


# name: (builder, output extension or "png/" or "still", what it stresses)
SCENES = {
    "layers-2000": (layers_2000, ".mp4", "2000 shapes, keyframed y, staggered fades: per-layer CPU cost"),
    "groups-deep": (groups_deep, ".mp4", "10 stacks of 30 nested groups with clips and expressions"),
    "text-split": (text_split, ".mp4", "40 lines split per char, blur-rise stagger, per-unit wave"),
    "text-wrap": (text_wrap, ".mp4", "20 wrapped 200-word paragraphs typed on with a caret"),
    "expressions": (expressions, ".mp4", "600 layers, 4 expressions each"),
    "keys-dense": (keys_dense, ".mp4", "300 layers x 121 keyframes on x, y, rotate, plus colour keys"),
    "counters": (counters, ".mp4", "100 counters formatting every frame"),
    "along-100": (along_100, ".mp4", "100 layers riding paths: 60 on 6 path layers that draw on, 40 round closed paths"),
    "strokes-300": (strokes_300, ".mp4", "100 marching dashed paths that draw on, 100 stars rounding, 100 four-corner tabs"),
    "rough-200": (rough_200, ".mp4", "100 outlined cards boiling 8 times a second, 100 rough lines drawing on"),
    "combine-60": (combine_60, ".mp4", "60 badges with a word cut out, 30 with a hole moving through them too"),
    "morph-200": (morph_200, ".mp4", "200 paths morphing triangle, circle, star and back"),
    "masks-50": (masks_50, ".mp4", "50 masked cards: 25 seen through words, 25 through a moving lens"),
    "glow-text": (glow_text, ".mp4", "a 220 px title glowing and outlined over 12 lines with glow, outline, inner "
                                      "shadow or displace"),
    "svg-500": (svg_500, ".mp4", "an svg layer of 500 shapes, gradients and groups standing still"),
    "logo": (logo, ".mp4", "examples/logo.json: the mark draws on and morphs, the word slides out through a mask"),
    "kinetic": (kinetic, ".mp4", "examples/kinetic.json: stagger orders, a combined badge, a mask, marching dashes"),
    "explainer": (explainer, ".mp4", "examples/explainer.json: boiling rough boxes, a dot riding a route, a lens mask"),
    "shaders": (shaders, ".mp4", "4 full-frame SkSL presets blended, grain and vignette"),
    "uhd-4k": (uhd_4k, ".mp4", "3840x2160: shader, split text, stroke draw, shadow, grade"),
    "long-120s": (long_120s, ".mp4", "3600 frames: worker split, segment joins"),
    "fps-120": (fps_120, ".mp4", "hello.json at 120 fps"),
    "fps-23.976": (fps_23976, ".mp4", "hello.json at 23.976 fps"),
    "blur-alpha": (blur_alpha, ".mov", "16-sample motion blur on a transparent canvas to ProRes 4444"),
    "alpha-webm": (alpha_webm, ".webm", "transparent canvas to VP9 with alpha"),
    "video-grid": (video_grid, ".mp4", "9 decoded clips at once: trims, speeds, loops, punch-in"),
    "sequence-60": (sequence_60, ".mp4", "60 cuts with transitions: decoder reopen per clip"),
    "audio-40": (audio_40, ".mp4", "music bed, voice, 40 panned clicks, ducking, loudness target"),
    "models-30": (models_30, ".mp4", "30 lit primitives at high detail and 3 glTF rockets"),
    "stage-16": (stage_16, ".mp4", "a stage: 12 rings and 4 cards passing through them, merged by depth per pixel"),
    "textured-100k": (textured_100k, ".mp4", "a textured .obj of 102,400 triangles turning under world light and env"),
    "dof-stage": (dof_stage, ".mp4", "a stage of 10 models and 3 cards with floor shadows and focus pulls"),
    "skinned": (skinned, ".mp4", "6 skinned glTF tubes of 24 joints and 23k triangles and 12 arms, posed every frame"),
    "repeat-1000": (repeat_1000, ".mp4", "one model of 1000 cubes moved, turned and coloured by expressions, with edges"),
    "repeat-10k": (repeat_10k, ".mp4", "one circle repeated 10,000 times, each copy moved by expressions every frame"),
    "particles-5k": (particles_5k, ".mp4", "a fountain keeping 5,000 stars alive, spun, shrunk and faded over life"),
    "captions-600": (captions_600, ".mp4", "600 karaoke cues and 200 pop cues in 30 s"),
    "odd-size": (odd_size, ".mp4", "1365x767 canvas into H.264 (even rounding)"),
    "gif": (gif_out, ".gif", "hello.json to GIF"),
    "png-seq": (png_seq, "png/", "1 s of hello.json as a PNG sequence"),
    "still-8k": (still_8k, "still", "one 7680x4320 frame"),
}


def build(name: str) -> dict:
    return SCENES[name][0]()


# --- memory: peak working set of this process and its children (the warm render pool) --------------------------

def _peaks_windows() -> int:
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                   "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                   "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.Process32First.argtypes = k32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    me, pids = os.getpid(), []
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    e = Entry()
    e.dwSize = ctypes.sizeof(Entry)
    ok = k32.Process32First(snap, ctypes.byref(e))
    while ok:
        if e.th32ParentProcessID == me and e.szExeFile.lower().startswith(b"python"):
            pids.append(e.th32ProcessID)
        ok = k32.Process32Next(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    total = 0
    for pid in [me] + pids:
        h = k32.OpenProcess(0x1000 | 0x10, False, pid)
        if not h:
            continue
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        if k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            total += c.PeakWorkingSetSize
        k32.CloseHandle(h)
    return total


def _peaks_linux() -> int:
    me, total = os.getpid(), 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            status = (d / "status").read_text()
        except OSError:
            continue
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        if int(d.name) == me or int(fields.get("PPid", "0").strip()) == me:
            total += int(fields.get("VmHWM", "0 kB").split()[0]) * 1024
    return total


def peak_mb() -> float | None:
    try:
        if sys.platform == "win32":
            return round(_peaks_windows() / 2 ** 20)
        if Path("/proc/self/status").exists():
            return round(_peaks_linux() / 2 ** 20)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)  # macOS: bytes, this process
    except Exception:  # noqa: BLE001 - memory is a nice-to-have column, never a reason to fail a scene
        return None


# --- one scene, in its own process --------------------------------------------------------------------------------

def _packets(path: Path) -> int | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return None


def one(name: str, work: Path, scale: float) -> dict:
    global WORK
    WORK = work
    sys.path.insert(0, str(ROOT))
    import kinescript as k
    from kinescript import output, verify

    row: dict = {"scene": name, "ok": False}
    stage = "build"
    try:
        path = work / f"{name}.json"
        path.write_text(json.dumps(build(name), indent=1), encoding="utf-8")
        stage = "load"
        t = time.perf_counter()
        scene = k.load(str(path))
        row.update(load_s=round(time.perf_counter() - t, 3), frames=scene.frames, w=scene.w, h=scene.h)
        scene.close()
        stage = "check"
        t = time.perf_counter()
        rep = verify.check(str(path))
        row.update(check_s=round(time.perf_counter() - t, 2), errors=rep["errors"], warnings=rep["warnings"],
                   codes=sorted({i["code"] for i in rep["issues"] if i["level"] in ("error", "warning")}))
        stage = "render"
        ext = SCENES[name][1]
        if ext == "still":
            out = work / f"{name}.png"
            t = time.perf_counter()
            output.still(str(path), 0.5 * scene.frames / scene.fps, str(out), scale=scale)
            row.update(render_s=round(time.perf_counter() - t, 2), frames_ok=out.exists())
        elif ext == "png/":
            out = work / f"{name}-png" / "frame.png"
            info = output.render(str(path), str(out), scale=scale)
            got = len(list(out.parent.glob("*.png")))
            row.update(render_s=info["wall"], workers=info.get("workers"), frames_ok=got == scene.frames)
        else:
            out = work / f"{name}{ext}"
            info = output.render(str(path), str(out), scale=scale)
            got = _packets(out) if ext != ".gif" else None
            row.update(render_s=info["wall"], fps=info["fps_rendered"], workers=info["workers"],
                       size_mb=info["size_mb"], audio=info["audio"],
                       frames_ok=None if got is None else got == scene.frames, frames_out=got)
        row["ok"] = True
    except Exception as e:  # noqa: BLE001 - the row records the failure; the next scene still runs
        row.update(stage=stage, error=f"{type(e).__name__}: {e}"[:300], trace=traceback.format_exc()[-1500:])
    row["peak_mb"] = peak_mb()
    return row


# --- the table ----------------------------------------------------------------------------------------------------

def _fmt(row: dict) -> str:
    def v(key, spec="", missing="-"):
        x = row.get(key)
        return missing if x is None else format(x, spec)

    check = f"{row['errors']}/{row['warnings']}" if "errors" in row else "-"
    ok = {True: "yes", False: "NO", None: "-"}[row.get("frames_ok")]
    status = "ok" if row["ok"] else f"FAIL at {row.get('stage')}: {row.get('error', '')[:70]}"
    return (f"{row['scene']:13s} {v('frames', 'd'):>6s} {check:>6s} {v('check_s', '.2f'):>7s} {v('render_s', '.2f'):>8s} "
            f"{v('fps', '.0f'):>6s} {v('workers', 'd'):>4s} {v('peak_mb', '.0f'):>7s} {v('size_mb', '.2f'):>7s} "
            f"{ok:>6s}  {status}")


def main():
    ap = argparse.ArgumentParser(description="Generated stress scenes: check, render and probe each one.")
    ap.add_argument("names", nargs="*", default=list(SCENES))
    ap.add_argument("--list", action="store_true", help="name and purpose of every scene")
    ap.add_argument("--keep", metavar="DIR", help="write the scenes and renders here instead of a temporary folder")
    ap.add_argument("--json", metavar="PATH", help="also write the rows as JSON")
    ap.add_argument("--scale", type=float, default=1.0, help="render scale (1 = the scene's own size)")
    ap.add_argument("--timeout", type=float, default=900, help="seconds before a scene counts as hung")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--work", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(a.one, Path(a.work), a.scale)))
        return
    if a.list:
        for name, (_, ext, what) in SCENES.items():
            print(f"{name:13s} {ext:6s} {what}")
        return
    unknown = [n for n in a.names if n not in SCENES]
    if unknown:
        sys.exit(f"unknown scene: {', '.join(unknown)} (see --list)")
    if not (MEDIA / "cells.mp4").exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_media.py")], check=True)
    git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout
    print(f"# {git.strip()} · {platform.processor() or platform.machine()} · {os.cpu_count()} threads · "
          f"Python {platform.python_version()} · scale {a.scale}")
    print(f"{'scene':13s} {'frames':>6s} {'check':>6s} {'check s':>7s} {'render s':>8s} {'fps':>6s} {'wrk':>4s} "
          f"{'peak MB':>7s} {'out MB':>7s} {'frames':>6s}  status")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(a.keep or tmp)
        work.mkdir(parents=True, exist_ok=True)
        for name in a.names:
            cmd = [sys.executable, __file__, "--one", name, "--work", str(work), "--scale", str(a.scale)]
            t = time.perf_counter()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                row = json.loads(lines[-1]) if lines else {
                    "scene": name, "ok": False, "stage": "process", "error": f"exit {r.returncode}: {r.stderr[-300:]}"}
            except subprocess.TimeoutExpired:
                row = {"scene": name, "ok": False, "stage": "process", "error": f"timed out after {a.timeout:.0f} s"}
            row["total_s"] = round(time.perf_counter() - t, 2)
            rows.append(row)
            print(_fmt(row))
            sys.stdout.flush()
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    failed = [r["scene"] for r in rows if not r["ok"] or r.get("frames_ok") is False]
    if failed:
        print(f"\n{len(failed)} of {len(rows)} failed: {', '.join(failed)}")
        for r in rows:
            if r.get("trace"):
                print(f"\n--- {r['scene']} ({r['stage']})\n{r['trace']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
