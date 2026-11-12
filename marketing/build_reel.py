"""Builds marketing/reel.json, the 32-second reel at the top of the project site: one scene that shows what Kinescript
draws, cut to music, with a line of text naming each thing and the key that does it.

    python marketing/build_reel.py            # writes marketing/reel.json
    kine voice marketing/reel.json            # speaks the host's one line (Kokoro plugin or KINE_VOICE_CMD)
    kine check marketing/reel.json && kine look marketing/reel.json --audio
    python marketing/build_site.py reel       # site/media/reel.mp4 and its poster

It plays muted on the page, so the picture carries it: each section holds one capability, mostly drawn live in this
scene (the 3D stage, the cutaway, the type, the charts, the host). Three parts play renders from site/demos instead:
Maya and Rin, whose avatars are built from pictures; the brand section, because a brand applies to a whole scene,
so two palettes are two renders; and the footage section, which is about cutting existing video. Sections start on the bars of the 120 bpm bed
from build_hero.py and change on a hard cut with a short whoosh. Motion follows brand/BRAND.md ("score-build").
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hero import snd  # noqa: E402
from build_promo import CUE, HEAD, KEY, MONO, NOTE, PANEL, SHEET, SNAP, TRACK, VOID, logo  # noqa: E402

HERE = Path(__file__).resolve().parent
MEDIA = "../examples/media/"
DEMOS = "../site/demos/"

# Section starts in seconds: whole bars of the 120 bpm bed (2 s), or half bars.
CUTS = {"open": 0, "stage": 3, "cell": 7, "type": 10, "charts": 13, "talk": 16, "brands": 21, "edit": 24,
        "end": 27}
END = 32


def span(name):
    names = list(CUTS)
    i = names.index(name)
    return CUTS[name], (CUTS[names[i + 1]] if i + 1 < len(names) else END)


def section(name, children, fixed=True, whoosh=True):
    """A section's group. Every section after the first starts on a hard cut with a short whoosh just before it."""
    t0, t1 = span(name)
    out = {"id": name, "type": "group", "at": t0, "dur": t1 - t0, "fixed": fixed, "children": children}
    if whoosh and t0 > 0:
        out["sound"] = snd("whoosh", at="in-0.15", gain=-14, pan=0.2)
    return out


def stage3d(name, children, head, key):
    """A 3D section: the set moves with the camera, its caption (a group of its own) does not."""
    return [section(name, children, fixed=False), section(name, caption(name, head, key), whoosh=False) | {"id": f"{name}-cap"}]


def caption(prefix, head, key):
    """The line that names what the section shows, and under it the key that does it (values in Keyframe)."""
    return [
        {"id": f"{prefix}-head", "type": "text", "text": head, "x": 160, "y": 150, "anchor": "left", "align": "left",
         "size": 60, "weight": 700, "at": 0.15, "enter": HEAD, "fixed": True},
        {"id": f"{prefix}-key", "type": "text", "text": key, "x": 162, "y": 214, "anchor": "left", "align": "left",
         "font": MONO, "size": 30, "weight": 500, "fill": NOTE, "emph": KEY, "at": 0.35,
         "enter": {"name": "fade", "dur": 0.25}, "fixed": True},
    ]


# ---- sections ---------------------------------------------------------------------------------------------

def s_open():
    return section("open", [
        *logo("lock", 960, 500, at=0.2, size=150),
        {"id": "tag", "type": "text", "text": "Video you can **diff.**", "x": 960, "y": 690, "size": 52, "weight": 500,
         "emph": KEY, "at": 1.25, "enter": SNAP, "sound": snd("pop", gain=-6)},
    ])


def s_stage():
    return stage3d("stage", [
        {"id": "stage-set", "type": "group", "x": "50%", "y": "54%", "sort": "pixel",
         "shadows": {"opacity": 0.55, "soft": 4}, "children": [
             {"id": "floor", "type": "rect", "y": 261, "z": -1400, "w": 5200, "h": 4000, "rotateX": 90, "fill": "#1d2129"},
             {"id": "table", "type": "model", "shape": "cylinder", "r": 300, "h": 24, "y": 249, "fill": "#e8e2d4",
              "shine": 0.25, "rotateY": [[0, 0], ["end", 160]]},
             {"id": "can", "type": "model", "src": MEDIA + "can.glb", "h": 470, "rotateY": [[0, -40], ["end", 130]],
              "enter": {"name": "build", "dur": 0.9}},
             {"id": "can-l", "type": "model", "src": MEDIA + "can.glb", "h": 470, "x": -620, "y": 26, "z": -1900,
              "rotateY": 40},
             {"id": "can-r", "type": "model", "src": MEDIA + "can.glb", "h": 470, "x": 900, "y": 26, "z": -2200,
              "rotateY": -70}]},
    ], "Lit 3D, with shadows.", '{"type": "model", "src": "**can.glb**"}')


def s_cell():
    return stage3d("cell", [
        {"id": "cellm", "type": "model", "x": "50%", "y": "56%", "rotateX": -24,
         "rotateY": [[0, -70], ["end", -34, "outCubic"]], "bands": 3,
         "edges": {"width": [[0, 0], [0.6, 2.5]], "color": "#20202a", "angle": 50},
         "cut": {"from": "front", "at": [[0.4, 0], [1.6, 0.5, "inOutCubic"]], "cap": "#fbe3c8"},
         "enter": {"name": "build", "dur": 0.6}, "sound": snd("whoosh", at="in+0.4", gain=-8),
         "parts": [{"shape": "sphere", "r": 270, "fill": "#e88a6a"},
                   {"shape": "sphere", "r": 105, "fill": "#7a5cc8", "x": 50, "y": -25},
                   {"shape": "torus", "r": 78, "tube": 24, "fill": "#3aa37a", "x": -115, "y": 105, "rotateX": 60},
                   {"shape": "capsule", "r": 26, "h": 120, "fill": "#e9c46a", "x": 120, "y": 120, "rotate": 40}]},
    ], "Cut it open.", '"cut": {"from": "front", "at": **0.5**}')


def s_type():
    return section("type", [
        {"id": "word", "type": "text", "text": "Every letter on a cue.", "x": 960, "y": 560, "size": 150, "weight": 700,
         "letterSpacing": -3, "split": "chars", "at": 0.2,
         "enter": {"name": "snap", "dur": 0.22, "stagger": 0.03},
         "unitY": "=sin(T*4 + i*0.55) * 10",
         "unitFill": "=mix('#f2f0e8', '#b9f45d', sweep(i, n, clamp((t-1.4)/1.2, 0, 1), 3))"},
        {"id": "word-rule", "type": "line", "points": [[-640, 0], [640, 0]], "x": 960, "y": 660, "stroke": TRACK,
         "strokeWidth": 6, "at": 0.1, "enter": {"name": "draw", "dur": 0.35, "ease": "outQuart"}},
        *caption("type", "Type that moves.", '"split": "**chars**"'),
    ])


def s_charts():
    return section("charts", [
        {"use": "barChart", "id": "bars", "x": 640, "y": 640, "w": 760, "h": 480, "size": 28, "valueSize": 34,
         "data": [["Q1", 42], ["Q2", 58], ["Q3", 91], ["Q4", 64]], "highlight": "Q3", "at": 0.3},
        {"use": "lineChart", "id": "line", "x": 1380, "y": 640, "w": 640, "h": 400, "size": 28, "color": "$key",
         "data": [["Mon", 12], ["Tue", 19], ["Wed", 15], ["Thu", 27], ["Fri", 34]], "at": 0.6},
        *caption("charts", "Charts from numbers.", '"data": [["Q1", **42**], ["Q2", **58**], ...]'),
    ])


def avatar_tile(id_, src, start, y, at):
    return {"id": id_, "type": "video", "src": DEMOS + src, "trim": [start, start + 2.6], "volume": 0,
            "anchor": "center", "x": 1400, "y": y, "w": 640, "h": 360, "radius": 14,
            "shadow": {"y": 12, "blur": 30, "color": "#00000088"}, "at": at,
            "enter": {"name": "pop", "dur": 0.35, "from": 0.85}, "sound": snd("pop", gain=-10)}


def s_host():
    return section("talk", [
        {"id": "host", "use": "avatar", "avatar": "kine:host", "x": "30%", "y": "60%", "h": 860, "at": 0.1,
         "say": {"text": "Write a [[line]]line, and I [[say]]say it.", "speaker": "am_michael", "at": "talk.in+0.6"},
         "expressions": [{"name": "smile", "at": "host-say.in", "dur": 1.2}, {"name": "happy", "at": "host-say:say"}],
         "gaze": [{"at": 0, "target": "camera"}],
         "gestures": [{"name": "wave", "at": "host-say.in"}, {"name": "nod", "at": "host-say:say"}]},
        {"id": "host-cc", "type": "captions", "voice": "host-say", "mode": "karaoke", "active": KEY, "size": 44,
         "x": "30%", "y": "90%", "maxWidth": 760, "box": {"fill": "#0a0c10d8", "pad": [12, 22], "radius": 10}},
        # the same rig drives presenters built from pictures: Maya (realistic) and Rin (anime), from avatars.html
        avatar_tile("maya", "avatar-maya.mp4", 4.9, 430, 2.5),
        avatar_tile("rin", "avatar-rin.mp4", 4.7, 810, 2.75),
        *caption("host", "Characters that talk.", '"say": {"text": "**Write a line**"}'),
    ])


def s_brands():
    tile = lambda id_, src, x, at: {  # noqa: E731
        "id": id_, "type": "video", "src": DEMOS + src, "trim": [1.2 + at, 4.5], "volume": 0, "anchor": "center",
        "x": x, "y": 600, "w": 768, "h": 432, "radius": 14, "shadow": {"y": 12, "blur": 30, "color": "#00000088"},
        "at": at, "enter": {"name": "pop", "dur": 0.35, "from": 0.85}, "sound": snd("pop", gain=-8)}
    return section("brands", [
        tile("light", "brand-tidewater.mp4", 540, 0.3),
        tile("dark", "brand-tidewater-dark.mp4", 1380, 0.55),
        *caption("brands", "One scene, every palette.", '"brand": "tidewater", "variant": "**dark**"'),
    ])


def s_edit():
    return section("edit", [
        {"id": "cut", "type": "sequence", "anchor": "center", "x": 960, "y": 640, "w": 1120, "h": 630,
         "clips": [{"src": DEMOS + "import.mp4", "trim": [0.8, 2.45], "volume": 0},
                   {"src": DEMOS + "materials.mp4", "trim": [2.2, 3.85], "volume": 0}],
         "transition": {"type": "zoom", "dur": 0.4}, "at": 0.15},
        {"id": "edit-cc", "type": "captions", "mode": "karaoke", "active": KEY, "size": 40, "y": "86%",
         "box": {"fill": "#0a0c10d8", "pad": [10, 20], "radius": 10},
         "cues": [[0.4, 1.6, "Trim, cut and caption"], [1.6, 2.9, "in the same file"]]},
        *caption("edit", "Cut footage, captioned.", '"trim": [**0.8**, **2.45**]'),
    ])


CODE = [
    '{',
    '  "canvas": {"w": 1920, "h": 1080},',
    '  "layers": [',
    '    {"type": "text", "text": "Hello",',
    '     "enter": "**snap**"}',
    '  ]',
    '}',
]


def s_end():
    lines = [{"id": f"code-{i}", "type": "text", "text": line, "x": 220, "y": 400 + 52 * i, "anchor": "left",
              "align": "left", "font": MONO, "size": 34, "weight": 500, "emph": KEY, "at": 0.35 + 0.22 * i,
              "enter": {"name": "type", "dur": 0.2}, "sound": snd("tick", gain=-14)} for i, line in enumerate(CODE)]
    return section("end", [
        {"id": "panel", "type": "rect", "x": 640, "y": 556, "w": 900, "h": 520, "radius": 18, "fill": PANEL,
         "stroke": TRACK, "strokeWidth": 2, "enter": {"name": "wipe-right", "dur": 0.35, "ease": "outQuart"}},
        {"id": "file", "type": "text", "text": "scene.json", "x": 220, "y": 336, "anchor": "left", "align": "left",
         "font": MONO, "size": 26, "weight": 500, "fill": NOTE, "at": 0.25, "enter": "fade"},
        *lines,
        *logo("end-lock", 1480, 470, at=1.9, size=96),
        {"id": "pip", "type": "text", "text": "pip install kinescript", "x": 1480, "y": 620, "font": MONO, "size": 38,
         "weight": 500, "fill": KEY, "at": 2.6, "enter": {"name": "type", "dur": 0.6}, "caret": True,
         "sound": snd("hit", gain=-4)},
        {"id": "confetti", "type": "particles", "preset": "confetti", "x": 1480, "y": 470, "at": 2.6, "dur": 2.3,
         "burst": 70, "rate": 0, "fill": [SHEET, KEY, NOTE, CUE]},
        *caption("end", "Write it. Check it. Render it.", "kine render **scene.json**"),
    ])


def camera():
    """One camera for the scene: still for the flat sections, moving through the two 3D ones."""
    def keys(segs, rest=0.0):
        out, last = [[0, rest]], rest
        for t0, t1, a, b in segs:
            out += [[t0 - 0.01, last], [t0, a], [t1 - 0.01, b, "inOutSine"]]
            last = b
        return out + [[END, last]]
    s0, s1 = span("stage")
    c0, c1 = span("cell")
    return {"rotateX": keys([(s0, s1, 14, 10), (c0, c1, 0, 0)]),
            "rotateY": keys([(s0, s1, -26, 18), (c0, c1, 0, 0)]),
            "y": keys([(s0, s1, -30, -10), (c0, c1, 0, 0)]),
            "z": keys([(s0, s1, 260, 40), (c0, c1, 160, 0)])}


def scene():
    spec = {
        "canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": END, "bg": VOID, "poster": CUTS["stage"] + 2.2},
        "grade": {"vignette": 0.3, "grain": 0.03},
        "camera": camera(),
        "light": {"preset": "studio", "follow": "world", "azimuth": -40, "elevation": 42},
        "env": "studio",
        "audio": [{"id": "bed", "role": "music", "src": "sound/bed.m4a", "until": "end", "fadeIn": 0.5,
                   "fadeOut": 2.0}],
        "layers": [{"id": "bg", "type": "shader", "preset": "score", "opacity": 0.8, "fixed": True,
                    "uniforms": {"base": VOID, "line": TRACK, "mark": NOTE}},
                   s_open(), *s_stage(), *s_cell(), s_type(), s_charts(), s_host(), s_brands(), s_edit(), s_end()],
    }
    return json.dumps(spec, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    out = HERE / "reel.json"
    out.write_text(scene() + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(HERE.parent)}")
