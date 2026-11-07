"""Writes the starter avatar, kine:host, into kinescript/avatars/host/ (avatar.json and parts/*.svg).

A plain, friendly figure drawn in code: a round head with ears and a tuft of hair, two eyes, brows, a mouth in all
nine shapes, a torso and two arms. Every colour is a palette role (the face and hands are text, the features bg, the
shirt accent, hair and brows muted), so the host takes any brand's light and dark palettes. Each file draws the
whole 600 x 900 avatar box, in place, with one element per colour id. Run it after changing the drawing:

    python brand/make_host.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "kinescript" / "avatars" / "host"
W, H = 600, 900
MX, MY = 300, 478  # the middle of the mouth


def svg(*elements: str) -> str:
    body = "".join(f"\n  {e}" for e in elements)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{body}\n</svg>\n'


def path(pid: str, d: str) -> str:
    return f'<path id="{pid}" d="{d}" fill="#000"/>'


def ellipse(cx, cy, rx, ry) -> str:
    """An ellipse as path commands, so several can share one path (and one colour id)."""
    return (f"M{cx - rx:g} {cy:g} A{rx:g} {ry:g} 0 1 0 {cx + rx:g} {cy:g} "
            f"A{rx:g} {ry:g} 0 1 0 {cx - rx:g} {cy:g} Z")


def rrect(x, y, w, h, r) -> str:
    return (f"M{x + r:g} {y:g} H{x + w - r:g} A{r:g} {r:g} 0 0 1 {x + w:g} {y + r:g} V{y + h - r:g} "
            f"A{r:g} {r:g} 0 0 1 {x + w - r:g} {y + h:g} H{x + r:g} A{r:g} {r:g} 0 0 1 {x:g} {y + h - r:g} "
            f"V{y + r:g} A{r:g} {r:g} 0 0 1 {x + r:g} {y:g} Z")


def mouth(shape: str) -> str:
    """One mouth shape, by the Rhubarb letters kine probe --mouth uses, centred on MX, MY."""
    x, y = MX, MY
    smile = f"M{x - 38} {y - 6} Q{x} {y + 22} {x + 38} {y - 6} Q{x} {y + 10} {x - 38} {y - 6} Z"
    shapes = {
        "X": [path("mouth", smile)],  # at rest: a closed smile
        "A": [path("mouth", rrect(x - 30, y - 4, 60, 9, 4.5))],  # M, B, P: lips pressed
        "B": [path("mouth", rrect(x - 34, y - 10, 68, 22, 11)),  # K, S, T: a little open, teeth showing
              path("teeth", rrect(x - 26, y - 10, 52, 8, 3))],
        "C": [path("mouth", ellipse(x, y + 2, 32, 20)),  # EH, AE: open
              path("teeth", rrect(x - 22, y - 18, 44, 9, 4))],
        "D": [path("mouth", ellipse(x, y + 6, 38, 32)),  # AA: wide open
              path("teeth", rrect(x - 24, y - 26, 48, 9, 4)),
              path("tongue", ellipse(x, y + 26, 22, 10))],
        "E": [path("mouth", ellipse(x, y + 4, 24, 22))],  # AO, ER: rounded
        "F": [path("mouth", ellipse(x, y + 2, 14, 15))],  # UW, OW, W: puckered
        "G": [path("mouth", rrect(x - 30, y - 6, 60, 18, 8)),  # F, V: teeth on the lip
              path("teeth", rrect(x - 24, y - 6, 48, 10, 4))],
        "H": [path("mouth", ellipse(x, y + 2, 30, 20)),  # L: the tongue up
              path("tongue", ellipse(x, y - 6, 16, 8))],
    }
    return svg(*shapes[shape])


PARTS = {
    "body.svg": svg(path("skin", rrect(272, 520, 56, 70, 18)),  # the neck
                    path("shirt", "M190 900 V676 C190 612 228 582 300 582 C372 582 410 612 410 676 V900 Z")),
    "arm-left.svg": svg(path("shirt", rrect(130, 616, 54, 180, 27)), path("skin", ellipse(157, 812, 28, 28))),
    "arm-right.svg": svg(path("shirt", rrect(416, 616, 54, 180, 27)), path("skin", ellipse(443, 812, 28, 28))),
    "head.svg": svg(path("skin", ellipse(300, 400, 150, 150) + " " + ellipse(154, 412, 30, 34) + " " +
                         ellipse(446, 412, 30, 34)),
                    path("hair", "M178 318 C196 238 260 214 312 220 C372 226 418 262 424 318 "
                                 "C392 282 346 268 300 272 C254 276 214 288 178 318 Z")),
    "brows.svg": svg(path("brow", rrect(218, 330, 56, 12, 6) + " " + rrect(326, 330, 56, 12, 6))),
    "eyes-open.svg": svg(path("feature", ellipse(246, 392, 17, 23) + " " + ellipse(354, 392, 17, 23)),
                         path("shine", ellipse(252, 383, 5, 5) + " " + ellipse(360, 383, 5, 5))),
    "eyes-closed.svg": svg(path("feature", "M228 394 Q246 406 264 394 Q246 400 228 394 Z "
                                           "M336 394 Q354 406 372 394 Q354 400 336 394 Z")),
    **{f"mouth-{s}.svg": mouth(s) for s in "XABCDEFGH"},
}

MANIFEST = {
    "kinescriptAvatar": 1,
    "name": "Host",
    "description": "Kinescript's starter avatar: a plain, friendly figure whose colours follow the palette.",
    "size": [W, H],
    "parts": {
        "body": {"src": "parts/body.svg", "pivot": [300, 900]},
        "head": {"src": "parts/head.svg", "parent": "body", "pivot": [300, 548]},
        "brows": {"src": "parts/brows.svg", "parent": "head", "pivot": [300, 336]},
        "eyes": {"parent": "head", "pivot": [300, 392], "look": [12, 7],
                 "states": {"open": "parts/eyes-open.svg", "closed": "parts/eyes-closed.svg"}},
        "mouth": {"parent": "head", "pivot": [MX, MY], "shapes": {s: f"parts/mouth-{s}.svg" for s in "XABCDEFGH"}},
        "armL": {"src": "parts/arm-left.svg", "parent": "body", "pivot": [157, 626]},
        "armR": {"src": "parts/arm-right.svg", "parent": "body", "pivot": [443, 626]},
    },
    "colors": {"skin": "text", "shirt": "accent", "hair": "muted", "brow": "muted", "feature": "bg", "shine": "text",
               "mouth": "bg", "teeth": "text", "tongue": "accent"},
    "expressions": {
        "smile": {"head": {"rotate": -3}, "brows": {"y": -8}},
        "happy": {"head": {"rotate": -4}, "brows": {"y": -12}, "mouth": "C"},
        "think": {"head": {"rotate": 7}, "brows": {"y": 6, "rotate": -4}},
        "surprise": {"brows": {"y": -24}, "mouth": "E"},
        "sad": {"head": {"rotate": 4, "y": 6}, "brows": {"y": 8}},
    },
    "gestures": {
        "nod": {"dur": 0.6, "head": {"y": [[0, 0], [0.15, 12], [0.3, 0], [0.45, 7], [0.6, 0]],
                                     "rotate": [[0, 0], [0.15, 2], [0.3, 0]]}},
        "shake": {"dur": 0.8, "head": {"rotate": [[0, 0], [0.15, -7], [0.35, 7], [0.55, -5], [0.8, 0]]}},
        "tilt": {"dur": 1.0, "head": {"rotate": [[0, 0], [0.3, -10], [0.7, -10], [1.0, 0]]}},
        "wave": {"dur": 1.4, "armR": {"rotate": [[0, 0], [0.3, -150], [0.55, -125], [0.8, -150], [1.05, -125],
                                                 [1.4, 0]]}},
        "point": {"dur": 1.2, "armR": {"rotate": [[0, 0], [0.3, -100], [0.9, -100], [1.2, 0]]}},
        "shrug": {"dur": 0.9, "armL": {"y": [[0, 0], [0.3, -20], [0.6, -20], [0.9, 0]],
                                       "rotate": [[0, 0], [0.3, 6], [0.6, 6], [0.9, 0]]},
                  "armR": {"y": [[0, 0], [0.3, -20], [0.6, -20], [0.9, 0]],
                           "rotate": [[0, 0], [0.3, -6], [0.6, -6], [0.9, 0]]},
                  "head": {"y": [[0, 0], [0.3, 10], [0.6, 10], [0.9, 0]]}},
    },
    "idle": {"blink": [2.5, 5.0], "breath": 0.012, "saccade": [0.8, 2.5]},
    "voice": {"backend": "kokoro", "speaker": "af_heart"},
}


def files() -> dict[str, str]:
    """Every file of the avatar, by its path inside the folder."""
    return {"avatar.json": json.dumps(MANIFEST, indent=2) + "\n", **{f"parts/{n}": s for n, s in PARTS.items()}}


def main():
    for rel, text in files().items():
        f = OUT / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(files())} files to {OUT}")


if __name__ == "__main__":
    main()
