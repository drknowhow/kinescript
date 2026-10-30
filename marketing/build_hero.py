"""Builds marketing/hero.json, the narrated video at the top of the project site, and the sounds it plays.

    python marketing/build_hero.py 120 9.6    # measured render speed shown in it: frames/s, and seconds for all of it
    kine voice marketing/hero.json            # speaks the six lines (Kokoro plugin or KINE_VOICE_CMD)
    kine check marketing/hero.json && kine look marketing/hero.json --audio
    python marketing/build_site.py hero       # site/media/hero.mp4 and its poster

Every section starts when its line of voice-over starts, and the layers inside it start on the words the voice
says ("at": "vo2:shapes"), so speaking a line again re-times the picture. Effects are layer sounds, so they move
with the layer that makes them. The music bed (120 bpm, the beat the score shader steps on), the effects and the
voice are files in marketing/sound/ and marketing/voice/; this script writes the first two, kine voice the third.
Motion follows brand/BRAND.md ("score-build").
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_promo import ART, CUE, HEAD, KEY, MONO, NOTE, PANEL, SHEET, SNAP, TRACK, VOID, heading, logo  # noqa: E402

HERE = Path(__file__).resolve().parent
FONTS = HERE.parent / "kinescript" / "fonts"
SOUND = HERE / "sound"
DEMOS = "../site/demos/"
GLOW = {"x": 0, "y": 0, "blur": 22, "color": CUE + "88"}
RATE = 48000

LINES = {
    "vo1": "This is Kinescript. Video you can [[diff]]diff.",
    "vo2": "A video is one JSON file: [[text]]text, [[shapes]]shapes, [[footage]]footage, [[depth]]depth, and "
           "[[sound]]sound, all written as data.",
    "vo3": "Time is a reference. Make one layer longer, and everything cued after it [[moves]]moves.",
    "vo4": "The voice starts as a line of text, and its words are cues. The music [[dips]]dips under it, and each "
           "effect [[lands]]lands on its cue.",
    "vo5": "Before it renders, it checks every frame, and the [[mix]]mix, and names each [[problem]]problem with "
           "its fix.",
    "vo6": "Then it renders on the GPU, with the sound mixed in. [[write]]Write it. [[check]]Check it. "
           "[[render]]Render it.",
}
GAP = {"vo2": 0.9, "vo3": 0.9, "vo4": 1.4, "vo5": 0.9, "vo6": 2.6}  # seconds of music before each line


def section(id_, at, until, children):
    return {"id": id_, "type": "group", "at": at, "until": until, "exit": {"name": "fade", "dur": 0.22},
            "children": children}


def width(text, size, face="SpaceGrotesk-Medium"):
    import skia
    return skia.Font(skia.Typeface.MakeFromFile(str(FONTS / f"{face}.ttf")), size).measureText(text)


def spoken(vid, default=7.0):
    """How long a line lasts, from the words file kine voice wrote (a fixed guess before it has)."""
    f = HERE / "voice" / f"{vid}.words.json"
    return round(max(w["end"] for w in json.loads(f.read_text())["words"]), 2) if f.exists() else default


def snd(name, **kw):
    return {"src": f"sound/{name}.wav", **kw} if kw else f"sound/{name}.wav"


def mono(id_, text, x, y, at, size=30, **kw):
    return {"id": id_, "type": "text", "text": text, "x": x, "y": y, "anchor": "left", "align": "left", "font": MONO,
            "size": size, "weight": 500, "emph": KEY, "at": at, "enter": SNAP, **kw}


def wave_path(x0, x1, mid, amp, step, seed, words=True):
    """Vertical strokes like a waveform: speech comes in bursts with gaps, music is steadier."""
    rng = np.random.default_rng(seed)
    out, x = [], x0
    level = 0.0
    while x <= x1:
        if words:
            level = 0.15 if rng.random() < 0.14 else min(1.0, max(0.3, level + rng.normal(0, 0.25)) if level > 0.2 else 0.7)
        else:
            level = 0.55 + 0.25 * np.sin(x / 90) + rng.normal(0, 0.08)
        h = max(4.0, amp * level * (0.7 + 0.3 * rng.random()))
        out.append(f"M{x:.0f} {mid - h:.0f} V{mid + h:.0f}")
        x += step
    return " ".join(out)


# ---- sections ---------------------------------------------------------------------------------------------

def s1():
    tag_x, tag_y = 960, 860
    left = tag_x - width("Video you can diff.", 48) / 2 + width("Video you can ", 48)
    return section("s1", 0, "vo2.in-0.4", [
        {"id": "mark", "type": "model", "x": "50%", "y": "33%", "rotateY": [[0, -70], [2.2, 0, "outCubic"]],
         "rotateX": [[0, 25], [2.2, 0, "outCubic"]], "enter": {"name": "build", "dur": 1.2},
         "sound": snd("whoosh", gain=-4),
         "parts": [
             {"shape": "path", "d": "M52 36L52 164M150 36C117 36 117 100 84 100C117 100 117 164 150 164",
              "strokeWidth": 18, "h": 300, "depth": 44, "bevel": 4, "fill": SHEET, "shine": 0.5},
             {"shape": "sphere", "r": 35, "x": -35, "z": 58, "fill": CUE, "shine": 0.8}]},
        {"id": "word", "type": "model", "shape": "text", "text": "kinescript", "x": "50%", "y": "62%", "size": 150,
         "weight": 700, "letterSpacing": -3, "depth": 36, "bevel": 3, "fill": SHEET, "shine": 0.5, "at": 0.6,
         "rotateX": [[0, 70], [2, 0, "outCubic"]], "enter": {"name": "build", "dur": 1.2},
         "sound": snd("hit", at="in+1.1", gain=-3)},
        {"id": "tagline", "type": "text", "text": "Video you can **diff.**", "x": tag_x, "y": tag_y, "size": 48,
         "weight": 500, "fill": NOTE, "emph": CUE, "at": "vo1:video", "enter": {"name": "fade", "dur": 0.3}},
        {"id": "tag-hit", "type": "rect", "x": round(left), "y": tag_y + 36, "w": round(width("diff", 48)), "h": 5,
         "anchor": "left", "fill": CUE, "shadow": GLOW, "at": "vo1:diff",
         "enter": {"name": "wipe-right", "dur": 0.3, "ease": "outQuart"}, "sound": snd("hit")},
    ])


CODE = [  # (line, the word it lands on, the demo clip the panel cuts to, the part of that clip)
    ('  {"type": **"text"**, "text": "Hello"},', "text", "dolly", [0.6, 9.0]),
    ('  {"type": **"rect"**, "radius": 24},', "shapes", "card-flip", [2.4, 6.0]),
    ('  {"type": **"video"**, "src": "a.mp4"},', "footage", "edit", [1.0, 11.6]),
    ('  {"type": **"model"**, "shape": "torus"}', "depth", "materials", [2.2, 6.0]),
    ('"audio": [{"role": **"voice"**, ...}]', "sound", "voice", [6.2, 9.9]),
]


def s2():
    lines = [mono("code-0", '"layers": [', 200, 390, 0.5)]
    clips = []
    for i, (text, word, clip, trim) in enumerate(CODE):
        y = 390 + 52 * (i + 1) + (52 if word == "sound" else 0)
        lines.append(mono(f"code-{word}", text, 200, y, f"vo2:{word}", sound=snd("tick")))
        nxt = CODE[i + 1][1] if i + 1 < len(CODE) else None
        clips.append({"id": f"clip-{word}", "type": "video", "src": f"{DEMOS}{clip}.mp4", "x": 0, "y": 0,
                      "anchor": "topleft", "w": 760, "h": 428, "fit": "cover", "volume": 0,
                      "trim": trim, "at": f"vo2:{word}", **({"until": f"vo2:{nxt}"} if nxt else {})})
    lines.insert(5, mono("code-close", "],", 200, 390 + 52 * 5, "vo2:depth+0.15"))
    return section("s2", "s1.out", "vo3.in-0.4", heading("h2", "Write it as data.", "One JSON file holds the picture and the sound.") + [
        {"id": "code-label", "type": "text", "text": "scene.json", "x": 162, "y": 305, "anchor": "left", "align": "left",
         "font": MONO, "size": 26, "fill": NOTE, "at": 0.2, "enter": "fade"},
        {"id": "code-panel", "type": "rect", "x": 160, "y": 330, "w": 760, "h": 428, "anchor": "topleft", "radius": 14,
         "fill": PANEL, "stroke": TRACK, "strokeWidth": 2, "at": 0.2, "enter": {"name": "wipe-down", "dur": 0.3}},
        *lines,
        {"id": "out-label", "type": "text", "text": "rendered", "x": 1002, "y": 305, "anchor": "left", "align": "left",
         "font": MONO, "size": 26, "fill": NOTE, "at": 0.35, "enter": "fade"},
        {"id": "out", "type": "group", "x": 1000, "y": 330, "w": 760, "h": 428, "anchor": "topleft", "clip": True,
         "radius": 14, "fill": VOID, "at": 0.35, "enter": SNAP, "children": [
             {"id": "out-plate", "type": "image", "src": ART + "transition.jpg", "x": 380, "y": 214, "w": 760,
              "h": 428, "fit": "cover", "opacity": 0.7, "zoom": [[0, 1.0], ["end", 1.1]]},
             *clips]},
        {"id": "out-frame", "type": "rect", "x": 1000, "y": 330, "w": 760, "h": 428, "anchor": "topleft", "radius": 14,
         "stroke": TRACK, "strokeWidth": 2, "at": 0.35, "enter": SNAP},
    ])


def s3():
    ticks = "M260 380 H1700 " + " ".join(f"M{260 + i * 120} {368 if i % 4 == 0 else 372} V390" for i in range(13))
    # (key, y, x before, width before, fill, label): after the cue, a grows by 240 and b and c move 240 right
    bars = [("a", 470, 260, 300, CUE, "intro"), ("b", 580, 584, 520, KEY, 'title  "at": "intro.out"'),
            ("c", 690, 1128, 430, SHEET, 'cta  "at": "title.out"')]
    grow = {"dur": 1.0, "ease": "inOutCubic"}
    layers = [{"id": "ruler", "type": "path", "d": ticks, "stroke": TRACK, "strokeWidth": 2, "cap": "butt", "at": 0.25,
               "enter": {"name": "draw", "dur": 0.4, "ease": "outQuart"}}]
    for n, (key, y, x, w, fill, label) in enumerate(bars):
        at = 0.4 + 0.2 * n
        common = {"type": "rect", "y": y, "h": 76, "anchor": "left", "radius": 8, "fill": fill}
        text = {"type": "text", "text": label, "font": MONO, "size": 30, "weight": 600, "y": y, "anchor": "left",
                "align": "left", "fill": VOID}
        layers += [
            {"id": f"bar-{key}", **common, "x": x, "w": w, "at": at, "until": "vo3:moves",
             "enter": {"name": "wipe-right", "dur": 0.3, "ease": "outQuart"}},
            {"id": f"bar-{key}-label", **text, "x": x + 28, "at": at + 0.2, "until": "vo3:moves", "enter": "fade"},
        ]
        if key == "a":
            moved = {"w": {"from": w, "to": w + 240, **grow}, "x": x}
            label_x = x + 28
        else:
            moved = {"w": w, "x": {"from": x, "to": x + 240, **grow}}
            label_x = {"from": x + 28, "to": x + 268, **grow}
        layers += [
            {"id": f"bar-{key}-moved", **common, **moved, "at": "vo3:moves",
             **({"sound": snd("whoosh", gain=-6)} if key == "a" else {})},
            {"id": f"bar-{key}-moved-label", **text, "x": label_x, "at": "vo3:moves"},
        ]
    layers.append({"id": "grow-note", "type": "text", "text": '"dur": 2.5  →  "dur": 4.5', "x": 260, "y": 840,
                   "anchor": "left", "align": "left", "font": MONO, "size": 32, "fill": CUE, "at": "vo3:moves",
                   "enter": SNAP})
    return section("s3", "s2.out", "vo4.in-0.4",
                   heading("h3", "Time is a reference.", 'A layer starts from another: "at": "intro.out".') + layers)


def s4():
    x0, x1 = 330, 1760
    lanes = [("voice", 430), ("music", 575), ("sfx", 720)]
    layers = []
    for name, y in lanes:
        layers += [
            {"id": f"lane-{name}", "type": "path", "d": f"M{x0} {y} H{x1}", "stroke": TRACK, "strokeWidth": 2,
             "at": 0.2, "enter": {"name": "draw", "dur": 0.4, "ease": "outQuart"}},
            mono(f"lane-{name}-label", f'"{name}"', 160, y, 0.3, size=28, fill=NOTE, enter="fade"),
        ]
    marks = [700, 1110, 1520]
    music = wave_path(-(x1 - x0) / 2, (x1 - x0) / 2, 0, 40, 16, 5, words=False)  # around the group's centre
    layers += [  # the voice's wave is uncovered as it speaks (its length is read from the spoken file)
        {"id": "voice-wave", "type": "path", "d": wave_path(x0, x1, 430, 52, 16, 3), "stroke": SHEET,
         "strokeWidth": 7, "cap": "round", "at": "vo4",
         "enter": {"name": "wipe-right", "dur": spoken("vo4"), "ease": "linear"}},
        {"id": "music", "type": "group", "x": (x0 + x1) / 2, "y": 575, "at": 0.3, "until": "vo4:dips", "children": [
            {"id": "music-wave", "type": "path", "d": music, "stroke": KEY, "strokeWidth": 7, "cap": "round",
             "enter": {"name": "wipe-right", "dur": 0.5, "ease": "outQuart"}}]},
        {"id": "music-dip", "type": "group", "x": (x0 + x1) / 2, "y": 575, "at": "vo4:dips",
         "scaleY": [[0, 1], [0.45, 0.3, "outCubic"]], "children": [
             {"id": "music-dip-wave", "type": "path", "d": music, "stroke": KEY, "strokeWidth": 7, "cap": "round"}]},
        mono("dip-note", "-14 dB while the voice speaks", 1360, 505, "vo4:dips+0.2", size=26, fill=KEY),
    ]
    for i, x in enumerate(marks):
        at = f"vo4:lands+{0.2 * i:g}"
        layers += [
            {"id": f"cue-line-{i}", "type": "path", "d": f"M{x} 470 V700", "stroke": NOTE, "strokeWidth": 2,
             "at": at, "enter": {"name": "draw", "dur": 0.25, "ease": "outQuart"}},
            {"id": f"cue-{i}", "type": "polygon", "sides": 4, "r": 18, "x": x, "y": 720, "fill": CUE, "shadow": GLOW,
             "at": at, "enter": {"name": "pop", "dur": 0.28, "from": 0.2}, "sound": snd("pop" if i < 2 else "hit")},
        ]
    layers.append({"id": "playhead", "type": "rect", "x": f"={x0} + {x1 - x0} * p", "y": 575, "w": 3, "h": 380,
                   "fill": SHEET, "opacity": 0.55, "at": "vo4", "until": "vo4.out"})
    return section("s4", "s3.out", "vo5.in-0.4", heading(
        "h4", "Sound on the same timeline.", "A voice from a line of text. Music and effects that level themselves.") + layers)


TILES = ["card-flip", "dolly", "materials", "title3d", "edit", "voice", "import", "captions"]


def s5():
    tiles = []  # placed from the sheet's centre, which it tips back on
    for i, name in enumerate(TILES):
        col, row = i % 4, i // 4
        x, y = col * 360 - 705, row * 222 - 204
        tiles += [
            {"id": f"tile-{name}", "type": "image", "src": f"{DEMOS}{name}.jpg", "x": x, "y": y, "w": 330, "h": 186,
             "anchor": "topleft", "fit": "cover", "radius": 8, "stroke": TRACK, "strokeWidth": 2,
             "at": 0.35 + i * 0.05, "enter": SNAP},
            {"id": f"tile-{name}-tc", "type": "text", "text": f"{0.6 + i * 1.3:.2f}s", "x": x + 12, "y": y + 22,
             "anchor": "left", "align": "left", "font": MONO, "size": 24, "fill": SHEET, "at": 0.45 + i * 0.05,
             "box": {"fill": "#0a0c10d0", "pad": [4, 8], "radius": 4}, "enter": "fade"},
        ]
    flag_x, flag_y = 260 + 2 * 360, 330 + 222
    return section("s5", "s4.out", "vo6.in-0.4", heading("h5", "Checked before it renders.", "kine look puts the frames in one image. kine check reads them, and the mix.") + [
        {"id": "sheet", "type": "group", "x": 965, "y": 534, "rotateX": [[0, 32], [0.9, 0, "outCubic"]],
         "children": tiles, "sound": snd("whoosh", gain=-8)},
        {"id": "mix-wave", "type": "path", "d": wave_path(260, 1670, 800, 26, 12, 9), "stroke": NOTE, "strokeWidth": 5,
         "cap": "round", "at": 0.9, "enter": {"name": "wipe-right", "dur": 0.6, "ease": "outQuart"}},
        {"id": "mix-flag", "type": "rect", "x": 1180, "y": 800, "w": 260, "h": 76, "radius": 6, "fill": CUE + "40",
         "stroke": CUE, "strokeWidth": 3, "at": "vo5:mix", "enter": {"name": "wipe-right", "dur": 0.25}},
        {"id": "flag", "type": "rect", "x": flag_x, "y": flag_y, "w": 330, "h": 186, "anchor": "topleft", "radius": 8,
         "stroke": CUE, "strokeWidth": 6, "shadow": GLOW, "at": "vo5:problem",
         "enter": {"name": "draw", "dur": 0.4}, "sound": snd("pop")},
        mono("flag-chip", "contrast 1.4:1  →  add a box", 1670, 290, "vo5:problem+0.15", size=28,
             fill=VOID, anchor="right", align="right", weight=600, until="vo5.out+0.6",
             box={"fill": CUE, "pad": [10, 18], "radius": 6}, exit={"name": "fade", "dur": 0.15}),
        mono("fixed-chip", "0 errors, 0 warnings", 1670, 290, "flag-chip.out", size=28, fill=VOID,
             anchor="right", align="right", weight=600, box={"fill": KEY, "pad": [10, 18], "radius": 6},
             sound=snd("tick")),
    ])


def s6():
    fast = [
        {"id": "fps", "type": "counter", "value": [[0, 0], [1.4, "{FPS}", "outQuart"]], "format": "{:.0f}", "x": 160,
         "y": 470, "anchor": "left", "align": "left", "size": 240, "weight": 700, "fill": CUE, "at": 0.3, "enter": SNAP},
        {"id": "fps-label", "type": "text", "text": "frames per second,\n1080p, with its sound", "x": 700, "y": 470,
         "anchor": "left", "align": "left", "size": 40, "weight": 500, "fill": SHEET, "at": 0.6, "enter": HEAD},
        {"id": "secs", "type": "counter", "value": [[0, 0], [1.2, "{SECS}", "outQuart"]], "format": "{:.1f} s",
         "x": 160, "y": 720, "anchor": "left", "align": "left", "size": 120, "weight": 700, "fill": KEY, "at": 0.7,
         "enter": SNAP},
        {"id": "secs-label", "type": "text", "text": "to render all of this video", "x": 700, "y": 720, "anchor": "left",
         "align": "left", "size": 40, "weight": 500, "fill": SHEET, "at": 0.8, "enter": HEAD},
    ]
    fast_group = {"id": "fast", "type": "group", "until": "vo6:write-0.25", "exit": {"name": "fade", "dur": 0.2},
                  "children": heading("h6", "Fast where it counts.", "GPU drawing, frames split across processes, one ffmpeg pass.") + fast}
    verbs = []
    for word, x in (("write", 700), ("check", 960), ("render", 1220)):
        verbs.append({"id": f"verb-{word}", "type": "text", "text": word, "x": x, "y": 690, "size": 44, "weight": 600,
                      "fill": SHEET, "at": f"vo6:{word}", "enter": SNAP, "sound": snd("tick" if word != "render" else "hit")})
    end = {"id": "end", "type": "group", "at": "vo6:write", "children": logo("end", 960, 440, at=0.0) + verbs + [
        {"id": "verb-dot-1", "type": "circle", "x": 830, "y": 692, "r": 5, "fill": NOTE, "at": "vo6:check",
         "enter": {"name": "pop", "dur": 0.25, "from": 0.2}},
        {"id": "verb-dot-2", "type": "circle", "x": 1090, "y": 692, "r": 5, "fill": NOTE, "at": "vo6:render",
         "enter": {"name": "pop", "dur": 0.25, "from": 0.2}},
        {"id": "cmd", "type": "text", "text": "pip install kinescript", "x": 960, "y": 810, "font": MONO, "size": 40,
         "fill": SHEET, "box": {"fill": PANEL, "pad": [16, 28], "radius": 10}, "at": "vo6:render.out+0.3",
         "enter": SNAP}]}
    s = section("s6", "s5.out", "vo6.out+2.8", [fast_group, end])
    s.pop("exit")
    return s


def wipes():
    """The playhead line that sweeps across at every section change."""
    out = []
    for n in range(2, 7):
        out.append({"id": f"wipe-{n}", "type": "rect", "x": [[0, -40], [0.45, 1960, "inOutCubic"]], "y": 540,
                    "w": 4, "h": 1080, "fill": SHEET, "shadow": GLOW, "at": f"s{n}.in-0.2", "dur": 0.45,
                    "sound": snd("whoosh", gain=-10, pan=0.2)})
    return {"id": "wipes", "type": "group", "children": out}


def scene(fps_value=0, secs_value=0):
    audio = []
    for n, (vid, text) in enumerate(LINES.items()):
        at = 1.0 if n == 0 else f"vo{n}.out+{GAP[vid]:g}"
        audio.append({"id": vid, "role": "voice", "text": text, "speaker": "af_heart", "at": at})
    audio.append({"id": "bed", "role": "music", "src": "sound/bed.m4a", "loop": True, "until": "end",
                  "fadeIn": 1.5, "fadeOut": 2.5})
    captions = {"id": "cc", "type": "group", "children": [
        {"id": f"cc-{vid}", "type": "captions", "voice": vid, "mode": "karaoke", "active": KEY, "size": 38,
         "y": "90%", "box": {"fill": "#0a0c10d8", "pad": [10, 20], "radius": 10}}
        for vid in LINES if vid not in ("vo1",)]}
    spec = {
        "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": VOID},
        "grade": {"vignette": 0.3, "grain": 0.03},
        "camera": {"z": [[0, 260], [3.2, 0, "outCubic"]], "rotateY": [[0, 10], [3.2, 0, "outCubic"]],
                   "rotateX": [[0, -4], [3.2, 0, "outCubic"]]},
        "light": "studio",
        "audio": audio,
        "layers": [{"id": "bg", "type": "shader", "preset": "score", "opacity": 0.8, "fixed": True,
                    "uniforms": {"base": VOID, "line": TRACK, "mark": NOTE}},
                   s1(), s2(), s3(), s4(), s5(), s6(), wipes(), captions],
    }
    text = json.dumps(spec, indent=1, ensure_ascii=False)
    return text.replace('"{FPS}"', str(round(fps_value))).replace('"{SECS}"', f"{secs_value:.1f}")


# ---- sounds -----------------------------------------------------------------------------------------------

def _wav(path: Path, audio) -> None:
    a = np.asarray(audio, dtype=np.float64)
    a = a.reshape(len(a), -1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(a.shape[1])
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())


def _lowpass(x, cutoff):
    """One-pole low-pass whose cutoff (Hz, one value per sample) moves over time."""
    a = 1 - np.exp(-2 * np.pi * np.asarray(cutoff) / RATE)
    y, acc = np.empty_like(x), 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def effects() -> None:
    rng = np.random.default_rng(11)
    lead = 0.02  # silence before each sound: the mix skips it, so the attack lands on the cue

    def t_of(seconds):
        return np.arange(int(seconds * RATE)) / RATE - lead

    u = t_of(0.25)
    tick = np.where(u >= 0, np.sin(2 * np.pi * 2200 * u) * np.exp(-u * 90) + 0.6 * np.sin(2 * np.pi * 640 * u)
                    * np.exp(-u * 45) + 0.25 * rng.standard_normal(len(u)) * np.exp(-u * 400), 0)
    _wav(SOUND / "tick.wav", 0.8 * tick / np.abs(tick).max())

    u = t_of(0.3)
    f = 600 + 700 * np.exp(-np.maximum(u, 0) * 40)
    pop = np.where(u >= 0, np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-u * 22)
                   + 0.15 * rng.standard_normal(len(u)) * np.exp(-u * 300), 0)
    _wav(SOUND / "pop.wav", 0.8 * pop / np.abs(pop).max())

    u = t_of(0.9)
    f = 42 + 120 * np.exp(-np.maximum(u, 0) * 28)
    kick = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-u * 6)
    body = 0.35 * np.sin(2 * np.pi * 190 * u) * np.exp(-u * 16)
    snap = 0.5 * _lowpass(rng.standard_normal(len(u)), np.full(len(u), 5000)) * np.exp(-u * 70)
    hit = np.where(u >= 0, np.tanh(1.6 * (kick + body + snap)), 0)
    _wav(SOUND / "hit.wav", 0.9 * hit / np.abs(hit).max())

    n = int(0.75 * RATE)
    p = np.arange(n) / n
    shape = np.sin(np.pi * p ** 0.8) ** 2
    noise = rng.standard_normal((n, 2))
    sweep = 400 + 5000 * shape
    left, right = _lowpass(noise[:, 0], sweep), _lowpass(noise[:, 1], sweep)
    pan = p  # moves left to right
    whoosh = np.stack([left * shape * np.cos(pan * np.pi / 2), right * shape * np.sin(pan * np.pi / 2)], 1)
    _wav(SOUND / "whoosh.wav", 0.8 * whoosh / np.abs(whoosh).max())


def bed() -> None:
    """Eight bars at 120 bpm, the two beats a second the score shader steps at, that loop without a seam: every
    pitch makes a whole number of cycles over the loop, and every envelope wraps."""
    import subprocess
    import tempfile
    beat, bars = 0.5, 8
    length = bars * 4 * beat
    t = np.arange(int(length * RATE)) / RATE
    rng = np.random.default_rng(5)

    def hz(note):
        return round(440 * 2 ** ((note - 69) / 12) * length) / length

    def env(start, attack, decay, hold=None):
        u = (t - start) % length
        e = np.minimum(u / attack, 1) * np.exp(-u * decay)
        return e if hold is None else np.where(u < hold, e, 0)

    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [48, 55, 60, 64], [55, 59, 62, 67]]  # Am7 Fmaj7 C G
    roots = [45, 41, 48, 43]
    left, right = np.zeros_like(t), np.zeros_like(t)
    for bar in range(bars):
        k = bar % 4
        b0 = bar * 4 * beat
        e = env(b0, 0.35, 0.18, 4 * beat + 0.6)
        for n, note in enumerate(chords[k]):
            for det in (-0.12, 0.12):
                f = hz(note + det)
                tone = e * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * 0.07
                left += tone * (1.1 if det < 0 else 0.6)
                right += tone * (0.6 if det < 0 else 1.1)
        for s in range(8):  # bass on eighths, the off-beats softer
            f = hz(roots[k] - 12 + (12 if s in (3, 7) else 0))
            e = env(b0 + s * beat / 2, 0.004, 9.0, beat / 2)
            bass = (0.3 if s % 2 == 0 else 0.18) * e * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t))
            left += bass
            right += bass
        if bar >= 4:  # the second half adds a plucked line
            notes = [chords[k][1], chords[k][2], chords[k][3], chords[k][2] + 12]
            for s in range(8):
                f = hz(notes[s % 4] + 12)
                e = env(b0 + s * beat / 2, 0.003, 7.0, 1.0)
                pluck = 0.09 * e * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
                left += pluck * (0.9 if s % 2 else 0.5)
                right += pluck * (0.5 if s % 2 else 0.9)
    def add(start, sound, gl, gr):  # a short sound into the loop, wrapping past its end
        idx = (int(round(start * RATE)) + np.arange(len(sound))) % len(t)
        left[idx] += gl * sound
        right[idx] += gr * sound

    u = np.arange(int(0.4 * RATE)) / RATE
    kick = np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-u * 35)) / RATE) * np.exp(-u * 9)
    for b in range(bars * 4):  # kick on every beat, clap on 2 and 4, hats on the off-beats
        add(b * beat, kick, 0.35, 0.35)
        u = np.arange(int(0.08 * RATE)) / RATE
        hat = np.exp(-u * 70) * rng.standard_normal(len(u))
        add(b * beat + beat / 2, hat - _lowpass(hat, np.full(len(u), 6000)), 0.05, 0.04)
        if b % 2 == 1:
            u = np.arange(int(0.2 * RATE)) / RATE
            clap = np.exp(-u * 30) * rng.standard_normal(len(u))
            add(b * beat, _lowpass(clap, np.full(len(u), 3000)) - _lowpass(clap, np.full(len(u), 700)), 0.12, 0.12)
    mix = np.stack([left, right], 1)
    mix = np.tanh(1.2 * mix / np.abs(mix).max()) * 0.6
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "bed.wav"
        _wav(raw, mix)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:a", "aac", "-b:a", "160k",
                        str(SOUND / "bed.m4a")], check=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    SOUND.mkdir(exist_ok=True)
    if "--sound" in sys.argv or not (SOUND / "bed.m4a").exists():
        effects()
        bed()
        print(f"wrote {SOUND.relative_to(HERE.parent)}: bed.m4a, tick, pop, hit, whoosh")
    out = HERE / "hero.json"
    out.write_text(scene(*(float(a) for a in args[:2])), encoding="utf-8")
    print(out)
