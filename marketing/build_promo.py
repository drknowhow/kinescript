"""Builds marketing/promo.json, the launch video, which is itself a Kinescript scene.

    python marketing/build_promo.py && kine check marketing/promo.json && kine render marketing/promo.json -o docs/promo.mp4
"""
import json
from pathlib import Path

INK, PAPER, CUE, SIGNAL, MUTED, PANEL = "#0e0f13", "#f3efe6", "#ff5a36", "#3ddbd9", "#8b8f99", "#16181f"
MONO = "Cascadia Code"


def heading(id_, text, sub=None):
    out = [{"id": id_, "type": "text", "text": text, "x": 160, "y": 170, "anchor": "left", "align": "left",
            "size": 66, "weight": 800, "enter": {"name": "rise", "dur": 0.5}}]
    if sub:
        out.append({"id": id_ + "-sub", "type": "text", "text": sub, "x": 162, "y": 245, "anchor": "left",
                    "align": "left", "size": 34, "weight": 400, "fill": MUTED, "at": 0.25, "enter": "rise"})
    return out


def logo(prefix, cx, cy, at=0.0, size=120):
    k = size / 120
    dot_x = cx - 300 * k
    parts = [{"id": f"{prefix}-dot", "type": "circle", "x": dot_x, "y": cy, "r": 30 * k, "fill": CUE, "at": at,
              "enter": {"name": "pop", "dur": 0.5}}]
    for i, (dy, length) in enumerate(((-20, 64), (0, 104), (20, 52))):
        parts.append({"id": f"{prefix}-trail{i}", "type": "line", "points": [[0, 0], [-length * k, 0]],
                      "x": dot_x - 46 * k, "y": cy + dy * k, "stroke": PAPER, "strokeWidth": 9 * k,
                      "at": at + 0.3 + i * 0.06, "enter": {"name": "draw", "dur": 0.45}})
    parts.append({"id": f"{prefix}-word", "type": "text", "text": "kinescript", "x": dot_x + 62 * k, "y": cy + 4 * k,
                  "anchor": "left", "align": "left", "size": size, "weight": 800, "split": "chars", "at": at + 0.4,
                  "enter": {"name": "rise+blur", "dur": 0.5, "stagger": 0.03, "dist": 40}})
    return parts


def section(id_, at, dur, children):
    return {"id": id_, "type": "group", "at": at, "dur": dur, "exit": {"name": "fade", "dur": 0.35},
            "children": children}


s1 = section("s1", 0, 3.6, logo("logo", 960, 500) + [
    {"id": "tagline", "type": "text", "text": "Video you can **diff.**", "x": 960, "y": 650, "size": 48,
     "weight": 500, "fill": MUTED, "at": 1.4, "enter": "rise"}])

code = ('{\n  "type": "text",\n  "text": **"Hello, world"**,\n  "at": **"logo.out+0.2"**,\n'
        '  "enter": **"rise+blur"**\n}')
s2 = section("s2", "s1.out", 6.8, heading("h2", "Write it as data.", "One JSON scene. No timeline UI, no hidden state.") + [
    {"id": "code-label", "type": "text", "text": "scene.json", "x": 162, "y": 330, "anchor": "left", "align": "left",
     "font": MONO, "size": 26, "fill": MUTED, "at": 0.3, "enter": "fade"},
    {"id": "code-panel", "type": "rect", "x": 160, "y": 360, "w": 780, "h": 440, "anchor": "topleft", "radius": 18,
     "fill": PANEL, "stroke": "#2a2d38", "strokeWidth": 2, "at": 0.3, "enter": "fade"},
    {"id": "code", "type": "text", "text": code, "x": 200, "y": 400, "anchor": "topleft", "align": "left",
     "font": MONO, "size": 34, "weight": 400, "lineHeight": 1.55, "emph": SIGNAL, "caret": True, "at": 0.6,
     "enter": {"name": "type", "dur": 2.4}},
    {"id": "code-arrow", "type": "arrow", "x1": 975, "y1": 580, "x2": 1060, "y2": 580, "stroke": CUE,
     "strokeWidth": 6, "head": 18, "at": 3.1, "enter": {"name": "draw", "dur": 0.4}},
    {"id": "result-label", "type": "text", "text": "rendered", "x": 1102, "y": 330, "anchor": "left", "align": "left",
     "font": MONO, "size": 26, "fill": MUTED, "at": 3.1, "enter": "fade"},
    {"id": "result", "type": "group", "x": 1100, "y": 360, "w": 660, "h": 440, "clip": True, "radius": 18,
     "fill": "#0b0c10", "at": 3.1, "enter": "fade", "children": [
         {"id": "result-bg", "type": "shader", "preset": "aurora", "w": 660, "h": 440, "opacity": 0.55},
         {"id": "result-text", "type": "text", "text": "Hello, world", "x": 330, "y": 220, "size": 72, "weight": 800,
          "split": "chars", "at": 0.4, "enter": {"name": "rise+blur", "dur": 0.55, "stagger": 0.035, "dist": 50}}]},
])

grow = "ease('inOutCubic', (t - 1.6) / 1.1)"
ticks = "M260 380 H1700 " + " ".join(f"M{260 + i * 120} 370 V390" for i in range(13))
s3 = section("s3", "s2.out", 6.4, heading("h3", "Time is a reference.", "Lengthen one layer and everything cued after it moves.") + [
    {"id": "ruler", "type": "path", "d": ticks, "stroke": "#3a3d48", "strokeWidth": 2, "at": 0.3,
     "enter": {"name": "draw", "dur": 0.6}},
    {"id": "bar-a", "type": "rect", "x": 260, "y": 470, "h": 76, "w": f"=300 + 240*{grow}", "anchor": "left",
     "radius": 12, "fill": CUE, "at": 0.4, "enter": {"name": "wipe-right", "dur": 0.4}},
    {"id": "bar-a-label", "type": "text", "text": "intro", "x": 288, "y": 470, "anchor": "left", "align": "left",
     "font": MONO, "size": 30, "weight": 600, "fill": INK, "at": 0.6, "enter": "fade"},
    {"id": "bar-b", "type": "rect", "x": f"=584 + 240*{grow}", "y": 580, "h": 76, "w": 520, "anchor": "left",
     "radius": 12, "fill": SIGNAL, "at": 0.6, "enter": {"name": "wipe-right", "dur": 0.4}},
    {"id": "bar-b-label", "type": "text", "text": 'title  "at": "intro.out"', "x": f"=612 + 240*{grow}", "y": 580,
     "anchor": "left", "align": "left", "font": MONO, "size": 30, "weight": 600, "fill": INK, "at": 0.8, "enter": "fade"},
    {"id": "bar-c", "type": "rect", "x": f"=1128 + 240*{grow}", "y": 690, "h": 76, "w": 430, "anchor": "left",
     "radius": 12, "fill": PAPER, "at": 0.8, "enter": {"name": "wipe-right", "dur": 0.4}},
    {"id": "bar-c-label", "type": "text", "text": 'cta  "at": "title.out"', "x": f"=1156 + 240*{grow}", "y": 690,
     "anchor": "left", "align": "left", "font": MONO, "size": 30, "weight": 600, "fill": INK, "at": 1.0, "enter": "fade"},
    {"id": "grow-note", "type": "text", "text": '"dur": 2.5  →  "dur": 4.5', "x": 260, "y": 860, "anchor": "left",
     "align": "left", "font": MONO, "size": 32, "fill": CUE, "at": 1.6, "enter": "rise"},
])

tiles = []
for i in range(8):
    col, row = i % 4, i // 4
    x, y = 260 + col * 360, 340 + row * 230
    stops = [[CUE, "#5a1f14"], [SIGNAL, "#103a3a"], ["#2a2d38", PANEL], ["#ffd60a", "#4a3b05"]][(i + row) % 4]
    tiles.append({"id": f"tile{i}", "type": "rect", "x": x, "y": y, "w": 330, "h": 186, "anchor": "topleft",
                  "radius": 10, "fill": {"linear": [[0, 0], [1, 1]], "stops": stops}, "at": 0.5 + i * 0.07,
                  "enter": {"name": "pop", "dur": 0.35, "from": 0.85}})
    tiles.append({"id": f"tile{i}-tc", "type": "text", "text": f"{0.6 + i * 1.3:.2f}s", "x": x + 12, "y": y + 22,
                  "anchor": "left", "align": "left", "font": MONO, "size": 24, "fill": PAPER, "at": 0.6 + i * 0.07,
                  "box": {"fill": "#0e0f13c0", "pad": [4, 8], "radius": 5}, "enter": "fade"})
s4 = section("s4", "s3.out", 7.0, heading("h4", "Checked before it renders.", "kine look puts many frames in one image. kine check reads them for you.") + tiles + [
    {"id": "flag", "type": "rect", "x": 1340, "y": 570, "w": 330, "h": 186, "anchor": "topleft", "radius": 10,
     "stroke": CUE, "strokeWidth": 6, "at": 1.8, "enter": {"name": "draw", "dur": 0.5}},
    {"id": "flag-chip", "type": "text", "text": "contrast 2.0:1 on lt-title  →  add a box", "x": 1670, "y": 830,
     "anchor": "right", "align": "right", "font": MONO, "size": 28, "weight": 600, "fill": INK,
     "box": {"fill": CUE, "pad": [10, 18], "radius": 8}, "at": 1.9, "dur": 2.6, "enter": "pop",
     "exit": {"name": "fade", "dur": 0.2}},
    {"id": "fixed-chip", "type": "text", "text": "kine check  →  0 errors, 0 warnings", "x": 1670, "y": 830,
     "anchor": "right", "align": "right", "font": MONO, "size": 28, "weight": 600, "fill": INK,
     "box": {"fill": SIGNAL, "pad": [10, 18], "radius": 8}, "at": "flag-chip.out", "enter": "pop"},
])

s5 = section("s5", "s4.out", 5.6, heading("h5", "Fast where it counts.", "GPU drawing, frames split across processes, one ffmpeg pass.") + [
    {"id": "fps", "type": "counter", "value": [[0, 0], [1.6, "{FPS}", "outCubic"]], "format": "{:.0f}", "x": 160,
     "y": 520, "anchor": "left", "align": "left", "size": 260, "weight": 900, "fill": CUE, "at": 0.3, "enter": "fade"},
    {"id": "fps-label", "type": "text", "text": "frames per second, 1080p,\nencoded with audio", "x": 700, "y": 520,
     "anchor": "left", "align": "left", "size": 40, "weight": 500, "fill": PAPER, "at": 0.6, "enter": "rise"},
    {"id": "ms", "type": "counter", "value": [[0, 0], [1.4, "{MS}", "outCubic"]], "format": "{:.0f} ms", "x": 160,
     "y": 790, "anchor": "left", "align": "left", "size": 120, "weight": 800, "fill": SIGNAL, "at": 1.0, "enter": "fade"},
    {"id": "ms-label", "type": "text", "text": "to draw one frame on the GPU", "x": 700, "y": 790, "anchor": "left",
     "align": "left", "size": 40, "weight": 500, "fill": PAPER, "at": 1.2, "enter": "rise"},
])

s6 = section("s6", "s5.out", 5.0, logo("end", 960, 460, at=0.1) + [
    {"id": "cmd", "type": "text", "text": "kine render scene.json", "x": 960, "y": 640, "font": MONO, "size": 40,
     "fill": PAPER, "box": {"fill": PANEL, "pad": [16, 28], "radius": 12}, "at": 1.0, "enter": "rise"},
    {"id": "verbs", "type": "text", "text": "write  ·  check  ·  look  ·  render", "x": 960, "y": 760, "size": 34,
     "weight": 500, "fill": MUTED, "at": 1.4, "enter": "fade"},
])
s6.pop("exit")

promo = {
    "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": INK},
    "grade": {"vignette": 0.4, "grain": 0.035},
    "audio": [{"src": "../examples/media/pad.m4a", "gain": -3, "fadeIn": 1.5, "fadeOut": 2.5}],
    "layers": [{"id": "bg", "type": "shader", "preset": "mesh", "opacity": 0.12,
                "uniforms": {"c1": CUE, "c2": SIGNAL, "c3": INK, "c4": PANEL, "speed": 0.35}},
               s1, s2, s3, s4, s5, s6],
}


def build(fps_value: float = 65, ms_value: float = 6) -> str:
    text = json.dumps(promo, indent=1, ensure_ascii=False)
    text = text.replace('"{FPS}"', str(round(fps_value))).replace('"{MS}"', str(round(ms_value)))
    out = Path(__file__).with_name("promo.json")
    out.write_text(text, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import sys
    print(build(*(float(a) for a in sys.argv[1:3])))
