"""Builds marketing/promo.json, the launch video, which is itself a Kinescript scene.

    python marketing/build_promo.py 160 5     # measured fps and frame ms, shown in the video
    kine check marketing/promo.json && kine render marketing/promo.json -o docs/promo.mp4 -q high
    kine render marketing/promo.json -o docs/hero.gif --scale 0.5 --to 10.5
    kine still marketing/promo.json --at 3 -o docs/poster.png
    kine still marketing/og.json --at 0 -o docs/og.png

Motion follows brand/BRAND.md ("score-build"): timing bars land first, cues snap onto them,
interpolation paths draw between them. Hard cuts, short overshoots, no blur entrances.
"""
import json
from pathlib import Path

VOID, SHEET, TRACK, CUE, KEY, NOTE = "#0a0c10", "#f2f0e8", "#242832", "#ff633f", "#b9f45d", "#9299a8"
PANEL = "#12151b"
MONO = "IBM Plex Mono"
ART = "../brand/art/"

# The logo mark, in its 200x200 design box (same geometry as brand/make_logo.py).
ARMS = [(150, 36), (117, 36), (117, 100), (84, 100), (117, 100), (117, 164), (150, 164)]
WORD_W = 599  # width of "kinescript" in Space Grotesk Bold at 128 px with -3 tracking

SNAP = {"name": "snap", "dur": 0.3}
HEAD = {"name": "left", "dur": 0.32, "dist": 28, "ease": "outQuart"}


def heading(id_, text, sub=None):
    out = [{"id": id_, "type": "text", "text": text, "x": 160, "y": 170, "anchor": "left", "align": "left",
            "size": 66, "weight": 700, "enter": HEAD}]
    if sub:
        out.append({"id": id_ + "-sub", "type": "text", "text": sub, "x": 162, "y": 245, "anchor": "left",
                    "align": "left", "size": 34, "weight": 400, "fill": NOTE, "at": 0.2,
                    "enter": {"name": "fade", "dur": 0.25}})
    return out


def logo(prefix, cx, cy, at=0.0, size=120):
    """The lockup, built the way the brand moves: bar, then the eased arms draw, then the cue snaps."""
    k = size / 128
    x0, y0 = cx - (196 + WORD_W) * k / 2, cy - 100 * k
    P = lambda x, y: f"{x0 + x * k:.1f} {y0 + y * k:.1f}"
    a = ARMS
    d = f"M{P(*a[0])} C {P(*a[1])} {P(*a[2])} {P(*a[3])} C {P(*a[4])} {P(*a[5])} {P(*a[6])}"
    return [
        {"id": f"{prefix}-bar", "type": "path", "d": f"M{P(52, 36)} L{P(52, 164)}", "stroke": SHEET,
         "strokeWidth": 18 * k, "at": at, "enter": {"name": "draw", "dur": 0.3, "ease": "outQuart"}},
        {"id": f"{prefix}-arms", "type": "path", "d": d, "stroke": SHEET, "strokeWidth": 18 * k, "at": at + 0.22,
         "enter": {"name": "draw", "dur": 0.5}},
        {"id": f"{prefix}-cue", "type": "circle", "x": x0 + 84 * k, "y": y0 + 100 * k, "r": 17 * k, "fill": CUE,
         "at": at + 0.5, "enter": {"name": "pop", "dur": 0.28, "from": 0.2}},
        {"id": f"{prefix}-word", "type": "text", "text": "kinescript", "x": x0 + 196 * k, "y": y0 + 114 * k,
         "anchor": "left", "align": "left", "size": size, "weight": 700, "letterSpacing": -3 * k,
         "split": "chars", "at": at + 0.62, "enter": {"name": "snap", "dur": 0.22, "stagger": 0.028}},
    ]


def section(id_, at, dur, children):
    return {"id": id_, "type": "group", "at": at, "dur": dur, "exit": {"name": "fade", "dur": 0.22},
            "children": children}


s1 = section("s1", 0, 3.8, logo("logo", 960, 500) + [
    {"id": "tagline", "type": "text", "text": "Video you can **diff.**", "x": 960, "y": 680, "size": 46,
     "weight": 500, "fill": NOTE, "emph": CUE, "at": 1.5, "enter": {"name": "fade", "dur": 0.3}}])

code = ('{\n  "type": "text",\n  "text": **"Hello, world"**,\n  "at": **"logo.out+0.2"**,\n'
        '  "enter": **"snap"**\n}')
s2 = section("s2", "s1.out", 6.8, heading("h2", "Write it as data.", "One JSON scene. No timeline UI, no hidden state.") + [
    {"id": "code-label", "type": "text", "text": "scene.json", "x": 162, "y": 330, "anchor": "left", "align": "left",
     "font": MONO, "size": 26, "fill": NOTE, "at": 0.3, "enter": "fade"},
    {"id": "code-panel", "type": "rect", "x": 160, "y": 360, "w": 780, "h": 440, "anchor": "topleft", "radius": 14,
     "fill": PANEL, "stroke": TRACK, "strokeWidth": 2, "at": 0.3, "enter": {"name": "wipe-down", "dur": 0.3}},
    {"id": "code", "type": "text", "text": code, "x": 200, "y": 400, "anchor": "topleft", "align": "left",
     "font": MONO, "size": 34, "weight": 400, "lineHeight": 1.55, "emph": KEY, "caret": True, "at": 0.6,
     "enter": {"name": "type", "dur": 2.4}},
    {"id": "code-arrow", "type": "arrow", "x1": 975, "y1": 580, "x2": 1060, "y2": 580, "stroke": CUE,
     "strokeWidth": 6, "head": 18, "at": 3.1, "enter": {"name": "draw", "dur": 0.3}},
    {"id": "result-label", "type": "text", "text": "rendered", "x": 1102, "y": 330, "anchor": "left", "align": "left",
     "font": MONO, "size": 26, "fill": NOTE, "at": 3.1, "enter": "fade"},
    {"id": "result", "type": "group", "x": 1100, "y": 360, "w": 660, "h": 440, "clip": True, "radius": 14,
     "fill": VOID, "at": 3.1, "enter": SNAP, "children": [
         {"id": "result-bg", "type": "image", "src": ART + "transition.jpg", "x": 330, "y": 220, "w": 660, "h": 440,
          "fit": "cover", "opacity": 0.7, "zoom": [[0, 1.0], ["end", 1.08]]},
         {"id": "result-text", "type": "text", "text": "Hello, world", "x": 330, "y": 220, "size": 72, "weight": 700,
          "box": {"fill": "#0a0c10d0", "pad": [10, 22], "radius": 10},
          "split": "chars", "at": 0.4, "enter": {"name": "snap", "dur": 0.22, "stagger": 0.03}}]},
])

grow = "ease('inOutCubic', (t - 1.6) / 1.1)"
ticks = "M260 380 H1700 " + " ".join(f"M{260 + i * 120} {368 if i % 4 == 0 else 372} V390" for i in range(13))
bars = []
for key, y, x, w, fill, label, at in (("a", 470, 260, "=300 + 240*" + grow, CUE, "intro", 0.4),
                                      ("b", 580, "=584 + 240*" + grow, 520, KEY, 'title  "at": "intro.out"', 0.6),
                                      ("c", 690, "=1128 + 240*" + grow, 430, SHEET, 'cta  "at": "title.out"', 0.8)):
    bars += [
        {"id": f"bar-{key}", "type": "rect", "x": x, "y": y, "h": 76, "w": w, "anchor": "left", "radius": 8,
         "fill": fill, "at": at, "enter": {"name": "wipe-right", "dur": 0.3, "ease": "outQuart"}},
        {"id": f"bar-{key}-label", "type": "text", "text": label, "font": MONO, "size": 30, "weight": 600,
         "x": x + 28 if key == "a" else f"=28 + {x[1:]}", "y": y, "anchor": "left", "align": "left", "fill": VOID,
         "at": at + 0.2, "enter": "fade"},
    ]
s3 = section("s3", "s2.out", 6.4, heading("h3", "Time is a reference.", "Lengthen one layer and everything cued after it moves.") + [
    {"id": "ruler", "type": "path", "d": ticks, "stroke": TRACK, "strokeWidth": 2, "cap": "butt", "at": 0.25,
     "enter": {"name": "draw", "dur": 0.4, "ease": "outQuart"}},
] + bars + [
    {"id": "grow-note", "type": "text", "text": '"dur": 2.5  →  "dur": 4.5', "x": 260, "y": 860, "anchor": "left",
     "align": "left", "font": MONO, "size": 32, "fill": CUE, "at": 1.6, "enter": SNAP},
])

tiles = []
plates = ["hero", "transition", "og", "hero", "transition", "og", "hero", "transition"]
focus = [[0.72, 0.3], [0.62, 0.45], [0.8, 0.45], [0.6, 0.75], [0.3, 0.2], [0.7, 0.6], [0.85, 0.55], [0.45, 0.8]]
for i in range(8):
    col, row = i % 4, i // 4
    x, y = 260 + col * 360, 340 + row * 230
    tiles.append({"id": f"tile{i}", "type": "image", "src": ART + plates[i] + ".jpg", "x": x, "y": y, "w": 330,
                  "h": 186, "anchor": "topleft", "fit": "cover", "zoom": 1.6 + 0.15 * (i % 3), "focus": focus[i],
                  "radius": 8, "stroke": TRACK, "strokeWidth": 2, "at": 0.5 + i * 0.06, "enter": SNAP})
    tiles.append({"id": f"tile{i}-tc", "type": "text", "text": f"{0.6 + i * 1.3:.2f}s", "x": x + 12, "y": y + 22,
                  "anchor": "left", "align": "left", "font": MONO, "size": 24, "fill": SHEET, "at": 0.6 + i * 0.06,
                  "box": {"fill": "#0a0c10d0", "pad": [4, 8], "radius": 4}, "enter": "fade"})
s4 = section("s4", "s3.out", 7.0, heading("h4", "Checked before it renders.", "kine look puts many frames in one image. kine check reads them for you.") + tiles + [
    {"id": "flag", "type": "rect", "x": 1340, "y": 570, "w": 330, "h": 186, "anchor": "topleft", "radius": 8,
     "stroke": CUE, "strokeWidth": 6, "at": 1.8, "enter": {"name": "draw", "dur": 0.4}},
    {"id": "flag-chip", "type": "text", "text": "contrast 2.0:1 on lt-title  →  add a box", "x": 1670, "y": 830,
     "anchor": "right", "align": "right", "font": MONO, "size": 28, "weight": 600, "fill": VOID,
     "box": {"fill": CUE, "pad": [10, 18], "radius": 6}, "at": 1.9, "dur": 2.6, "enter": SNAP,
     "exit": {"name": "fade", "dur": 0.15}},
    {"id": "fixed-chip", "type": "text", "text": "kine check  →  0 errors, 0 warnings", "x": 1670, "y": 830,
     "anchor": "right", "align": "right", "font": MONO, "size": 28, "weight": 600, "fill": VOID,
     "box": {"fill": KEY, "pad": [10, 18], "radius": 6}, "at": "flag-chip.out", "enter": SNAP},
])

s5 = section("s5", "s4.out", 5.6, heading("h5", "Fast where it counts.", "GPU drawing, frames split across processes, one ffmpeg pass.") + [
    {"id": "fps", "type": "counter", "value": [[0, 0], [1.4, "{FPS}", "outQuart"]], "format": "{:.0f}", "x": 160,
     "y": 520, "anchor": "left", "align": "left", "size": 260, "weight": 700, "fill": CUE, "at": 0.3, "enter": SNAP},
    {"id": "fps-label", "type": "text", "text": "frames per second, 1080p,\nencoded with audio", "x": 700, "y": 520,
     "anchor": "left", "align": "left", "size": 40, "weight": 500, "fill": SHEET, "at": 0.6, "enter": HEAD},
    {"id": "ms", "type": "counter", "value": [[0, 0], [1.2, "{MS}", "outQuart"]], "format": "{:.0f} ms", "x": 160,
     "y": 790, "anchor": "left", "align": "left", "size": 120, "weight": 700, "fill": KEY, "at": 1.0, "enter": SNAP},
    {"id": "ms-label", "type": "text", "text": "to draw one frame on the GPU", "x": 700, "y": 790, "anchor": "left",
     "align": "left", "size": 40, "weight": 500, "fill": SHEET, "at": 1.2, "enter": HEAD},
])

s6 = section("s6", "s5.out", 5.0, logo("end", 960, 440, at=0.1) + [
    {"id": "cmd", "type": "text", "text": "pip install kinescript", "x": 960, "y": 640, "font": MONO, "size": 40,
     "fill": SHEET, "box": {"fill": PANEL, "pad": [16, 28], "radius": 10}, "at": 1.0, "enter": SNAP},
    {"id": "verbs", "type": "text", "text": "write  ·  check  ·  look  ·  render", "x": 960, "y": 760, "size": 34,
     "weight": 500, "fill": NOTE, "at": 1.4, "enter": "fade"},
])
s6.pop("exit")

promo = {
    "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": VOID},
    "grade": {"vignette": 0.3, "grain": 0.03},
    "audio": [{"src": "../examples/media/pad.m4a", "gain": -3, "fadeIn": 1.5, "fadeOut": 2.5}],
    "layers": [{"id": "bg", "type": "shader", "preset": "score", "opacity": 0.8,
                "uniforms": {"base": VOID, "line": TRACK, "mark": NOTE}},
               s1, s2, s3, s4, s5, s6],
}


def build(fps_value: float = 65, ms_value: float = 6) -> str:
    text = json.dumps(promo, indent=1, ensure_ascii=False)
    text = text.replace('"{FPS}"', str(round(fps_value))).replace('"{MS}"', str(round(ms_value)))
    out = Path(__file__).with_name("promo.json")
    out.write_text(text, encoding="utf-8")
    return str(out)


def build_og() -> str:
    """marketing/og.json: the 1200x630 link card (render with kine still --at 0)."""
    layers = [{"id": "art", "type": "image", "src": ART + "og.jpg", "x": 600, "y": 315, "w": 1200, "h": 630,
               "fit": "cover"}]
    k = 76 / 128  # bar edge (box x 43) lines up with the tagline at x 90
    for layer in logo("logo", 90 + ((196 + WORD_W) / 2 - 43) * k, 270, size=76):
        layer.pop("enter")
        layer["at"] = 0
        layers.append(layer)
    layers += [
        {"id": "tagline", "type": "text", "text": "Video you can **diff.**", "x": 90, "y": 380, "anchor": "left",
         "align": "left", "size": 44, "weight": 600, "emph": CUE},
        {"id": "sub", "type": "text", "text": "Motion graphics and edits as JSON.\nChecked by agents, rendered on the GPU.",
         "x": 90, "y": 470, "anchor": "left", "align": "left", "size": 24, "fill": NOTE, "lineHeight": 1.4},
    ]
    # Five seconds only so the read-time check treats it like a card someone actually reads.
    scene = {"canvas": {"w": 1200, "h": 630, "fps": 30, "bg": VOID, "duration": 5}, "layers": layers}
    out = Path(__file__).with_name("og.json")
    out.write_text(json.dumps(scene, indent=1, ensure_ascii=False), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import sys
    print(build(*(float(a) for a in sys.argv[1:3])))
    print(build_og())
