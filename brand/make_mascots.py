"""Draws the example avatars Cue (a robot with a screen for a face) and Slate (a clapperboard whose clapper is its
jaw) as SVG parts, into examples/avatars/cue and examples/avatars/slate. Run from the repository root:

    python brand/make_mascots.py

Colours are palette roles in avatar.json (each svg piece's id names one), so a brand recolours both.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

W, H = 600, 750
OUT = Path("examples/avatars")


def svg(*pieces: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            + "".join(f"  {p}\n" for p in pieces) + "</svg>\n")


def path(pid: str, d: str, **attrs) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<path id="{pid}" d="{d}" fill="#000"{extra}/>'


def rr(x: float, y: float, w: float, h: float, r: float) -> str:
    """A rounded rectangle as path data."""
    r = min(r, w / 2, h / 2)
    return (f"M{x + r:g} {y:g} H{x + w - r:g} A{r:g} {r:g} 0 0 1 {x + w:g} {y + r:g} V{y + h - r:g} "
            f"A{r:g} {r:g} 0 0 1 {x + w - r:g} {y + h:g} H{x + r:g} A{r:g} {r:g} 0 0 1 {x:g} {y + h - r:g} "
            f"V{y + r:g} A{r:g} {r:g} 0 0 1 {x + r:g} {y:g} Z")


def circle(cx: float, cy: float, r: float) -> str:
    return f"M{cx - r:g} {cy:g} A{r:g} {r:g} 0 1 0 {cx + r:g} {cy:g} A{r:g} {r:g} 0 1 0 {cx - r:g} {cy:g} Z"


def ring(cx: float, cy: float, r: float, t: float) -> str:
    return circle(cx, cy, r) + " " + circle(cx, cy, r - t)


def poly(points) -> str:
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points) + " Z"


def band(points, t: float) -> str:
    """A line through points, t thick, as a filled shape (so a palette role can colour it)."""
    top = [(x, y - t / 2) for x, y in points]
    bottom = [(x, y + t / 2) for x, y in reversed(points)]
    return poly(top + bottom)


def arc_eye(cx: float, cy: float, w: float, up: bool = True) -> str:
    """A happy ^ eye: a thick arc."""
    s = -1 if up else 1
    return (f"M{cx - w / 2:g} {cy:g} Q{cx:g} {cy + s * w * 0.9:g} {cx + w / 2:g} {cy:g} "
            f"L{cx + w / 2 - 12:g} {cy:g} Q{cx:g} {cy + s * w * 0.55:g} {cx - w / 2 + 12:g} {cy:g} Z")


def diamond(cx: float, cy: float, r: float) -> str:
    return poly([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)])


def sparkle(cx: float, cy: float, r: float) -> str:
    k = r * 0.22
    return (f"M{cx:g} {cy - r:g} Q{cx + k:g} {cy - k:g} {cx + r:g} {cy:g} Q{cx + k:g} {cy + k:g} {cx:g} {cy + r:g} "
            f"Q{cx - k:g} {cy + k:g} {cx - r:g} {cy:g} Q{cx - k:g} {cy - k:g} {cx:g} {cy - r:g} Z")


def stroke(x1, y1, x2, y2, t) -> str:
    """A straight line with round ends, t thick, as a filled shape."""
    a = math.atan2(y2 - y1, x2 - x1)
    nx, ny = -math.sin(a) * t / 2, math.cos(a) * t / 2
    r = t / 2
    return (f"M{x1 + nx:.1f} {y1 + ny:.1f} L{x2 + nx:.1f} {y2 + ny:.1f} A{r:g} {r:g} 0 0 0 {x2 - nx:.1f} {y2 - ny:.1f} "
            f"L{x1 - nx:.1f} {y1 - ny:.1f} A{r:g} {r:g} 0 0 0 {x1 + nx:.1f} {y1 + ny:.1f} Z")


def write(folder: Path, files: dict[str, str], spec: dict) -> None:
    (folder / "parts").mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / "parts" / name).write_text(text, encoding="utf-8")
    (folder / "avatar.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


# --- Cue ----------------------------------------------------------------------------------------------------------

def wave(amp: float, width: float, cycles: float, jag: bool = False) -> str:
    """Cue's mouth: a waveform across the screen, amp pixels high."""
    x0, y0, n = 300 - width / 2, 408, 48
    pts = []
    for i in range(n + 1):
        u = i / n
        env = math.sin(math.pi * u)  # quiet at the ends, loud in the middle
        s = math.sin(2 * math.pi * cycles * u)
        if jag:
            s = 1 if s > 0 else -1
        pts.append((x0 + width * u, y0 - amp * env * s))
    return band(pts, 10)


CUE_MOUTHS = {"X": (0, 72, 1), "A": (0, 48, 1), "B": (7, 84, 4), "C": (16, 96, 2.5), "D": (30, 108, 2),
              "E": (22, 84, 1.5), "F": (12, 48, 1.5), "G": (12, 84, 4, True), "H": (20, 96, 3)}


def cue() -> None:
    eye = lambda h: path("eye", rr(222, 315 - h / 2, 42, h, 21) + " " + rr(336, 315 - h / 2, 42, h, 21))  # noqa: E731
    files = {
        "body.svg": svg(path("neck", rr(270, 470, 60, 80, 12)),
                        path("torso", "M180 528 H420 A54 54 0 0 1 474 582 V750 H126 V582 A54 54 0 0 1 180 528 Z"),
                        path("chest", circle(300, 618, 18))),
        "arm-left.svg": svg(path("arm", rr(84, 560, 54, 176, 27))),
        "arm-right.svg": svg(path("arm", rr(462, 560, 54, 176, 27))),
        "head.svg": svg(path("stalk", rr(294, 110, 12, 84, 6)),
                        path("shell", rr(114, 186, 372, 312, 90)),
                        path("screen", rr(150, 222, 300, 240, 60))),
        "antenna.svg": svg(path("antenna", circle(300, 102, 24))),
        "eyes-open.svg": svg(eye(66)),
        "eyes-closed.svg": svg(eye(12)),
        "eyes-happy.svg": svg(path("eye", arc_eye(243, 330, 54) + " " + arc_eye(357, 330, 54))),
        "eyes-wow.svg": svg(path("eye", circle(243, 312, 30) + " " + circle(357, 312, 30))),
        "eyes-error.svg": svg(path("eye", " ".join(stroke(cx - 20, 295, cx + 20, 335, 12) + " "
                                                   + stroke(cx - 20, 335, cx + 20, 295, 12) for cx in (243, 357)))),
        "fx-sparks.svg": svg(path("spark", " ".join(diamond(x, y, r) for x, y, r in
                                                    ((510, 168, 22), (84, 210, 16), (528, 300, 12), (66, 330, 10),
                                                     (450, 96, 10))))),
        "fx-glow.svg": svg(path("glow", ring(300, 102, 54, 9), fill_opacity="0.55")),
        "fx-alert.svg": svg(path("alert", rr(512, 150, 22, 70, 11) + " " + circle(523, 246, 13))),
    }
    for s, (amp, width, cycles, *jag) in CUE_MOUTHS.items():
        files[f"mouth-{s}.svg"] = svg(path("wave", wave(amp, width, cycles, bool(jag))))
    spec = {
        "kinescriptAvatar": 1, "name": "Cue",
        "description": "Kinescript's mascot: a small robot with a screen for a face and the Cue dot as its antenna. "
                       "Its mouth is a waveform. Drawn by brand/make_mascots.py.",
        "size": [W, H],
        "parts": {
            "armL": {"src": "parts/arm-left.svg", "parent": "body", "pivot": [111, 580]},
            "armR": {"src": "parts/arm-right.svg", "parent": "body", "pivot": [489, 580]},
            "body": {"src": "parts/body.svg", "pivot": [300, 750]},
            "head": {"src": "parts/head.svg", "parent": "body", "pivot": [300, 498]},
            "antenna": {"src": "parts/antenna.svg", "parent": "head", "pivot": [300, 102]},
            "eyes": {"parent": "head", "pivot": [300, 315], "look": [18, 10],
                     "states": {k: f"parts/eyes-{k}.svg" for k in ("open", "closed", "happy", "wow", "error")}},
            "mouth": {"parent": "head", "pivot": [300, 408], "shapes": {s: f"parts/mouth-{s}.svg" for s in CUE_MOUTHS}},
            "fx-sparks": {"src": "parts/fx-sparks.svg", "parent": "head", "pivot": [300, 240], "effect": True},
            "fx-glow": {"src": "parts/fx-glow.svg", "parent": "head", "pivot": [300, 102], "effect": True},
            "fx-alert": {"src": "parts/fx-alert.svg", "parent": "head", "pivot": [523, 200], "effect": True},
        },
        "colors": {"neck": "muted", "torso": "surface", "chest": "accent", "arm": "muted", "stalk": "muted",
                   "shell": "text", "screen": "bg", "antenna": "accent", "eye": "text", "wave": "accent2",
                   "spark": "accent2", "glow": "accent", "alert": "accent"},
        "expressions": {
            "happy": {"eyes": {"state": "happy"}, "head": {"rotate": -5}, "fx-sparks": {}},
            "wow": {"eyes": {"state": "wow"}, "head": {"y": -6}, "mouth": "E"},
            "error": {"eyes": {"state": "error"}, "head": {"rotate": 5, "y": 6}, "fx-alert": {}},
            "think": {"head": {"rotate": 6}, "eyes": {"y": -8}},
        },
        "gestures": {
            "nod": {"dur": 0.6, "head": {"y": [[0, 0], [0.15, 10], [0.3, 0], [0.45, 6], [0.6, 0]]}},
            "shake": {"dur": 0.8, "head": {"rotate": [[0, 0], [0.15, -6], [0.35, 6], [0.55, -4], [0.8, 0]]}},
            "wave": {"dur": 1.4, "armR": {"rotate": [[0, 0], [0.3, -150], [0.55, -125], [0.8, -150], [1.05, -125],
                                                     [1.4, 0]]}},
            "ping": {"dur": 0.8, "antenna": {"scale": [[0, 1], [0.15, 1.35], [0.4, 1], [0.8, 1]]},
                     "fx-glow": {"scale": [[0, 0.6], [0.4, 1.3], [0.8, 1.6]], "opacity": [[0, 0], [0.1, 1], [0.8, 0]]}},
            "bounce": {"dur": 0.7, "body": {"y": [[0, 0], [0.15, 10], [0.35, -26], [0.55, 0], [0.62, 5], [0.7, 0]]}},
        },
        "idle": {"blink": [2.5, 5], "breath": 0.01, "saccade": [0.8, 2.5]},
        "voice": {"backend": "kokoro", "speaker": "am_puck"},
    }
    write(OUT / "cue", files, spec)


# --- Slate --------------------------------------------------------------------------------------------------------

def stripes(x0: float, y0: float, w: float, h: float, n: int = 6) -> str:
    """Slanted black stripes across a clapper bar."""
    step = w / n
    return " ".join(poly([(x0 + i * step + 18, y0), (x0 + i * step + 18 + step / 2, y0),
                          (x0 + i * step + step / 2 - 6, y0 + h), (x0 + i * step - 6, y0 + h)])
                    for i in range(n))


def letters_cut(x: float, y: float, s: float) -> str:
    """The word CUT in blocky strokes, s tall, its top left at x, y."""
    t = s * 0.2
    w = s * 0.72
    c = (f"M{x + w:g} {y:g} H{x + t * 0.9:g} A{t * 0.9:g} {t * 0.9:g} 0 0 0 {x:g} {y + t * 0.9:g} V{y + s - t * 0.9:g} "
         f"A{t * 0.9:g} {t * 0.9:g} 0 0 0 {x + t * 0.9:g} {y + s:g} H{x + w:g} V{y + s - t:g} H{x + t:g} V{y + t:g} "
         f"H{x + w:g} Z")
    ux = x + w + s * 0.18
    u = (f"M{ux:g} {y:g} H{ux + t:g} V{y + s - t:g} H{ux + w - t:g} V{y:g} H{ux + w:g} V{y + s - t * 0.9:g} "
         f"A{t * 0.9:g} {t * 0.9:g} 0 0 1 {ux + w - t * 0.9:g} {y + s:g} H{ux + t * 0.9:g} "
         f"A{t * 0.9:g} {t * 0.9:g} 0 0 1 {ux:g} {y + s - t * 0.9:g} Z")
    tx = ux + w + s * 0.18
    tt = f"M{tx:g} {y:g} H{tx + w:g} V{y + t:g} H{tx + w / 2 + t / 2:g} V{y + s:g} H{tx + w / 2 - t / 2:g} V{y + t:g} H{tx:g} Z"
    return " ".join((c, u, tt))


SLATE_MOUTHS = {"X": -8, "A": 0, "B": -11, "C": -16, "D": -28, "E": -21, "F": -9, "G": -5, "H": -14}


def slate() -> None:
    eyes = lambda r, py: [path("white", circle(234, 408, r) + " " + circle(366, 408, r)),  # noqa: E731
                          path("pupil", circle(240, 408 + py, r * 0.43) + " " + circle(372, 408 + py, r * 0.43))]
    files = {
        "body.svg": svg(path("leg", " ".join((stroke(222, 600, 222, 708, 18), stroke(222, 708, 186, 708, 18),
                                              stroke(378, 600, 378, 708, 18), stroke(378, 708, 414, 708, 18))))),
        "board.svg": svg(path("board", rr(120, 288, 360, 312, 30)),
                         path("edge", rr(120, 288, 360, 312, 30) + " " + rr(129, 297, 342, 294, 22),
                              fill_rule="evenodd"),
                         path("bar", rr(120, 288, 360, 42, 8)),
                         path("stripes", stripes(126, 288, 348, 42))),
        "label-open.svg": svg(path("label", " ".join((rr(156, 516, 96, 14, 7), rr(156, 546, 132, 14, 7),
                                                      rr(336, 516, 108, 14, 7), rr(366, 546, 78, 14, 7)))),
                              path("dot", circle(438, 570, 12))),
        "label-cut.svg": svg(path("cut", letters_cut(201, 505, 66))),
        "eyes-open.svg": svg(*eyes(42, 6)),
        "eyes-closed.svg": svg(path("lid", rr(198, 404, 72, 12, 6) + " " + rr(330, 404, 72, 12, 6))),
        "eyes-happy.svg": svg(path("lid", arc_eye(234, 420, 78) + " " + arc_eye(366, 420, 78))),
        "eyes-squeeze.svg": svg(path("lid", " ".join((stroke(204, 384, 258, 408, 14), stroke(258, 408, 204, 432, 14),
                                                      stroke(396, 384, 342, 408, 14), stroke(342, 408, 396, 432, 14))))),
        "eyes-wow.svg": svg(*eyes(52, 0)),
        "fx-lines.svg": svg(path("line", " ".join((stroke(96, 228, 54, 198, 14), stroke(90, 270, 42, 270, 14),
                                                   stroke(504, 228, 546, 198, 14), stroke(510, 270, 558, 270, 14))))),
        "fx-stars.svg": svg(path("star", " ".join(sparkle(x, y, r) for x, y, r in
                                                  ((84, 384, 30), (522, 336, 24), (540, 486, 16), (72, 516, 14)))))
    }
    for s, a in SLATE_MOUTHS.items():
        files[f"mouth-{s}.svg"] = svg(f'<g transform="rotate({a} 132 288)">',
                                      path("clap", rr(120, 246, 360, 42, 8)),
                                      path("clapstripes", stripes(126, 246, 348, 42)), "</g>")
    spec = {
        "kinescriptAvatar": 1, "name": "Slate",
        "description": "A clapperboard with a face; the clapper is its jaw, so every word is a little clap. "
                       "Drawn by brand/make_mascots.py.",
        "size": [W, H],
        "parts": {
            "body": {"src": "parts/body.svg", "pivot": [300, 708]},
            "board": {"src": "parts/board.svg", "parent": "body", "pivot": [300, 600]},
            "label": {"parent": "board", "pivot": [300, 540],
                      "states": {"open": "parts/label-open.svg", "cut": "parts/label-cut.svg"}},
            "eyes": {"parent": "board", "pivot": [300, 408], "look": [16, 8],
                     "states": {k: f"parts/eyes-{k}.svg" for k in ("open", "closed", "happy", "squeeze", "wow")}},
            "mouth": {"parent": "board", "pivot": [132, 288], "shapes": {s: f"parts/mouth-{s}.svg" for s in SLATE_MOUTHS}},
            "fx-lines": {"src": "parts/fx-lines.svg", "parent": "board", "pivot": [300, 240], "effect": True},
            "fx-stars": {"src": "parts/fx-stars.svg", "parent": "board", "pivot": [300, 430], "effect": True},
        },
        "colors": {"leg": "text", "board": "#12151B", "edge": "text", "bar": "text", "stripes": "bg",
                   "label": "muted", "dot": "accent", "cut": "accent", "white": "text", "pupil": "bg", "lid": "text",
                   "clap": "text", "clapstripes": "bg", "line": "accent", "star": "accent2"},
        "expressions": {
            "cut": {"label": {"state": "cut"}, "eyes": {"state": "squeeze"}, "mouth": "A", "fx-lines": {}},
            "happy": {"eyes": {"state": "happy"}, "board": {"rotate": -4}, "fx-stars": {}},
            "wow": {"eyes": {"state": "wow"}, "board": {"y": -10}, "mouth": "D"},
            "think": {"board": {"rotate": 5}, "eyes": {"y": -6}},
        },
        "gestures": {
            "snap": {"dur": 0.6, "board": {"scale": [[0, 1], [0.1, 1.06], [0.25, 0.98], [0.6, 1]]},
                     "fx-lines": {"scale": [[0, 0.8], [0.15, 1.1], [0.6, 1.2]],
                                  "opacity": [[0, 0], [0.08, 1], [0.45, 1], [0.6, 0]]}},
            "hop": {"dur": 0.8, "body": {"y": [[0, 0], [0.15, 12], [0.4, -60], [0.65, 0], [0.72, 8], [0.8, 0]],
                                         "rotate": [[0, 0], [0.4, -4], [0.65, 0]]}},
            "nod": {"dur": 0.6, "board": {"rotate": [[0, 0], [0.15, 3], [0.3, 0], [0.45, 2], [0.6, 0]],
                                          "y": [[0, 0], [0.15, 8], [0.3, 0]]}},
            "shake": {"dur": 0.8, "board": {"rotate": [[0, 0], [0.15, -5], [0.35, 5], [0.55, -3], [0.8, 0]]}},
        },
        "idle": {"blink": [2.5, 5], "breath": 0.01, "saccade": [0.8, 2.5]},
        "voice": {"backend": "kokoro", "speaker": "bm_george"},
    }
    write(OUT / "slate", files, spec)


if __name__ == "__main__":
    cue()
    slate()
    print("wrote", OUT / "cue", "and", OUT / "slate")
