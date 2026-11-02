"""Render the landing page's media (site/) with Kinescript, and write each demo's JSON and score into the page.

    python examples/make_media.py        # once: footage, images and the models the examples read
    python marketing/build_site.py       # everything
    python marketing/build_site.py dolly hero  # only these videos and posters ("look" is the contact sheets)
    python marketing/build_site.py sound       # only make the voice demo's bed and click again
    python marketing/build_site.py --page      # only rewrite the page fragments, no rendering

The voices of the voice demos (site/demos/voice/) and of the hero (marketing/voice/) are spoken by `kine voice` and
committed, so building the site does not need a speech engine unless a line changes; then install the plugin each
sound names (voices.json uses all three) or set KINE_VOICE_CMD_<NAME>. bench/voices.py times them. The voice
demo's music bed and click (site/demos/sound/) are synthesised here, the hero's sounds by marketing/build_hero.py,
and both are committed too.

Every video on the page is a scene rendered by `kine render -q high` at a fraction of its canvas size
(the demos at half scale: 960x540, or 540x960 for the vertical one; the hero at 1280x720). The hero plays the
demos' videos and posters, so it renders after them. ffmpeg then
moves the index to the front of each file (`-movflags +faststart`, no re-encode) so browsers start playing
before the whole file has arrived. Posters are `kine still` frames saved as JPEG.

site/index.html holds marked regions, <!--build:KIND:NAME--> ... <!--/build-->, that this script fills in:
json (the scene file, highlighted), score (top-level layers from `kine probe`, drawn as bars with a dot for
each child's start), diff, check and probe (real command output). Text outside the markers is hand-written.
"""
from __future__ import annotations

import difflib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinescript.verify import check, probe  # noqa: E402

SITE = ROOT / "site"
DEMOS = SITE / "demos"
MEDIA = SITE / "media"
PAGE = SITE / "index.html"

# name: (scene, poster time in seconds, render scale). Order matters: edit.json cuts other demos' videos.
DEMO = {
    "card-flip": ("site/demos/card-flip.json", 1.55, 0.5),
    "dolly": ("site/demos/dolly.json", 5.5, 0.5),
    "materials": ("site/demos/materials.json", 3.5, 0.5),
    "title3d": ("examples/title3d.json", 3.2, 0.5),
    "import": ("examples/import.json", 3.0, 0.5),
    "product": ("examples/product.json", 2.5, 0.5),
    "captions": ("site/demos/captions.json", 4.0, 0.5),
    "voice": ("site/demos/voice.json", 6.0, 0.5),
    "voices": ("site/demos/voices.json", 18.45, 0.5),
    "edit": ("site/demos/edit.json", 2.0, 0.5),
    "logo": ("examples/logo.json", 4.5, 0.5),
    "kinetic": ("examples/kinetic.json", 3.5, 0.5),
    "explainer": ("examples/explainer.json", 4.9, 0.5),
    "motions": ("examples/motions.json", 5.0, 0.5),
    "transitions": ("examples/transitions.json", 2.2, 0.5),
    "charts": ("examples/charts.json", 4.5, 0.5),
}
HERO = ("marketing/hero.json", 3.6, 2 / 3)
LOOK = "card-flip"          # the contact sheet shown in the agent section
LISTEN = "voice"            # the contact sheet with the sound under it, in the sound section
TRAILS = "explainer"        # the contact sheet with motion trails, in the agent section
CHECK = "site/demos/problems.json"
PROBE = ("card-flip", 1.5)
CUES = "site/demos/cues.json"
CUES_CHANGE = ('"dur": 2.5', '"dur": 4.5')

KINE = [sys.executable, "-m", "kinescript.cli"]


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}\n{p.stdout}\n{p.stderr}")
    return p.stdout


def render(scene: str, out: Path, scale: float, poster_at: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw, png = Path(tmp) / "raw.mp4", Path(tmp) / "poster.png"
        res = json.loads(run(*KINE, "render", scene, "-o", str(raw), "-q", "high", "--scale", f"{scale:.6f}"))
        run("ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c", "copy", "-movflags", "+faststart", str(out))
        run(*KINE, "still", scene, "--at", str(poster_at), "--scale", f"{scale:.6f}", "-o", str(png))
        run("ffmpeg", "-y", "-v", "error", "-i", str(png), "-q:v", "3", str(out.with_suffix(".jpg")))
    print(f"{out.relative_to(ROOT)}  {res['w']}x{res['h']}  {res['seconds']} s  {out.stat().st_size / 1e6:.2f} MB")


def look_sheet() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "look.png"
        run(*KINE, "look", DEMO[LOOK][0], "--boxes", "-n", "8", "--cols", "4", "--width", "480", "-o", str(png))
        run("ffmpeg", "-y", "-v", "error", "-i", str(png), "-q:v", "3", str(MEDIA / "look.jpg"))
        png = Path(tmp) / "listen.png"
        run(*KINE, "look", DEMO[LISTEN][0], "--audio", "-n", "8", "--cols", "4", "--width", "480", "-o", str(png))
        run("ffmpeg", "-y", "-v", "error", "-i", str(png), "-q:v", "3", str(MEDIA / "listen.jpg"))
        png = Path(tmp) / "trails.png"
        run(*KINE, "look", DEMO[TRAILS][0], "--trails", "-n", "8", "--cols", "4", "--width", "480", "-o", str(png))
        run("ffmpeg", "-y", "-v", "error", "-i", str(png), "-q:v", "3", str(MEDIA / "trails.jpg"))


def _wav(path: Path, audio, rate: int = 48000) -> None:
    import wave
    import numpy as np
    a = np.asarray(audio, dtype=np.float64)
    a = a.reshape(len(a), -1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(a.shape[1])
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())


def sound_assets() -> None:
    """The voice demo's music bed (four chords and a plucked arpeggio at 100 bpm that loops without a seam) and
    its click (a short tick after 20 ms of silence, which the mix skips so the tick lands on its cue)."""
    import numpy as np
    out = DEMOS / "sound"
    out.mkdir(exist_ok=True)
    rate, beat = 48000, 0.6
    chords = [[45, 52, 55, 60], [41, 48, 52, 57], [48, 55, 60, 64], [43, 50, 55, 59]]  # Am7 Fmaj7 C G (MIDI)
    length = len(chords) * 4 * beat
    t = np.arange(int(length * rate)) / rate

    def hz(note):  # a whole number of cycles over the loop, so its end meets its start
        return round(440 * 2 ** ((note - 69) / 12) * length) / length

    def env(start, attack, decay, hold):
        u = (t - start) % length
        return np.where(u < hold, np.minimum(u / attack, 1) * np.exp(-u * decay), 0)

    left, right = np.zeros_like(t), np.zeros_like(t)
    for k, chord in enumerate(chords):
        e = env(k * 4 * beat, 0.5, 0.25, 4 * beat + 0.8)
        for n, note in enumerate(chord):
            f = hz(note)
            tone = e * (np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t)) / len(chord)
            left += tone * (0.6 + 0.1 * n)
            right += tone * (0.9 - 0.1 * n)
        for s, note in enumerate([chord[1], chord[2], chord[3] + 12, chord[2], chord[3], chord[2] + 12, chord[3], chord[1] + 12]):
            f = hz(note + 12)
            e = env(k * 4 * beat + s * beat / 2, 0.005, 5.0, 1.2)
            pluck = 0.35 * e * (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t))
            left += pluck * (0.8 if s % 2 else 0.5)
            right += pluck * (0.5 if s % 2 else 0.8)
    bed = np.stack([left, right], 1)
    bed *= 0.5 / np.abs(bed).max()
    with tempfile.TemporaryDirectory() as tmp:
        _wav(Path(tmp) / "bed.wav", bed, rate)
        run("ffmpeg", "-y", "-v", "error", "-i", str(Path(tmp) / "bed.wav"), "-c:a", "aac", "-b:a", "128k",
            str(out / "bed.m4a"))
    u = np.arange(int(0.25 * rate)) / rate - 0.02
    noise = np.random.default_rng(7).standard_normal(len(u))
    tick = np.where(u >= 0, np.sin(2 * np.pi * 2200 * u) * np.exp(-u * 90) + 0.6 * np.sin(2 * np.pi * 640 * u)
                    * np.exp(-u * 45) + 0.25 * noise * np.exp(-u * 400), 0)
    _wav(out / "click.wav", 0.8 * tick / np.abs(tick).max(), rate)
    print(f"wrote {out.relative_to(ROOT)}: bed.m4a ({length:g} s loop), click.wav")


# ---- page fragments -------------------------------------------------------------------------------------

TOKEN = re.compile(r'"(?:\\.|[^"\\])*"(?=\s*:)|"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b')
TIME_KEYS = {'"at"', '"until"'}


def highlight(line: str, state: dict) -> str:
    """One line of JSON as HTML: keys, strings, numbers. A string given to "at" or "until" is a cue reference."""
    out, pos = [], 0
    for m in TOKEN.finditer(line):
        out.append(html.escape(line[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith('"') and line[m.end():].lstrip().startswith(":"):
            cls = "k"
            state["key"] = tok
        elif tok.startswith('"'):
            cls = "c" if state.get("key") in TIME_KEYS else "s"
            state["key"] = None
        else:
            cls = "n"
            state["key"] = None
        out.append(f'<span class="{cls}">{html.escape(tok)}</span>')
        pos = m.end()
    out.append(html.escape(line[pos:]))
    return "".join(out)


def json_block(path: Path) -> str:
    state: dict = {}
    lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")
    return "".join(f'<span class="ln">{highlight(l, state) or " "}</span>' for l in lines)


def diff_block(before: str, after: str) -> str:
    rows, state = [], {}
    for line in difflib.unified_diff(before.rstrip("\n").split("\n"), after.rstrip("\n").split("\n"), n=100,
                                     lineterm=""):
        if line.startswith(("---", "+++", "@@")):
            continue
        mark, body = line[0], line[1:]
        cls = {"-": " del", "+": " add"}.get(mark, "")
        rows.append(f'<span class="ln{cls}"><span class="mark">{mark if mark != " " else " "}</span>'
                    f'{highlight(body, state) or " "}</span>')
    return "".join(rows)


def score_block(summary: dict, caption: str, axis: float | None = None) -> str:
    """Top-level layers as bars on one time axis, with a dot where each of their children starts, then the scene's
    own sounds (not those of layers), with a dot where each word of a voice is said.
    `axis` (seconds) puts several scores on the same scale; by default the axis is the scene's length."""
    end = summary["canvas"]["duration"]
    dur = axis or end
    layers = summary["layers"]
    parent = {l["id"]: l["parent"] for l in layers}

    def root(i):
        while parent[i]:
            i = parent[i]
        return i

    cues: dict[str, set] = {}
    for l in layers:
        if l["parent"]:
            cues.setdefault(root(l["id"]), set()).add(round(l["in"], 3))
    rows = []
    for l in layers:
        if l["parent"]:
            continue
        a, b = l["in"], l["out"]
        span = max(b - a, 1e-6)
        dots = "".join(f'<i style="left:{(t - a) / span * 100:.2f}%"></i>'
                       for t in sorted(cues.get(l["id"], ())) if t > a + 1e-3)
        rows.append(f'<li><span class="sid">{html.escape(l["id"])}</span><span class="track">'
                    f'<span class="bar" data-in="{a:g}" data-out="{b:g}" title="{html.escape(l["id"])}: {a:g} s to {b:g} s" '
                    f'style="left:{a / dur * 100:.2f}%;width:{span / dur * 100:.2f}%">{dots}</span></span></li>')
    for s in summary.get("audio", []):
        if s["layer"]:
            continue
        a, b = s["in"], min(s["out"], end)
        span = max(b - a, 1e-6)
        dots = "".join(f'<i style="left:{(t - a) / span * 100:.2f}%"></i>' for _, t, _ in s.get("words", ()) if t > a + 1e-3)
        rows.append(f'<li class="snd"><span class="sid">{html.escape(s["id"])}</span><span class="track">'
                    f'<span class="bar" data-in="{a:g}" data-out="{b:g}" title="{html.escape(s["id"])} '
                    f'({s["role"] or "sound"}): {a:g} s to {b:g} s" '
                    f'style="left:{a / dur * 100:.2f}%;width:{span / dur * 100:.2f}%">{dots}</span></span></li>')
    return (f'<ol class="score" data-dur="{dur:g}" style="--dur:{dur:g}" aria-label="{html.escape(caption)}, '
            f'{end:g} s">' + "".join(rows) + '<li class="ruler" aria-hidden="true"><span class="sid"></span>'
            f'<span class="track"><span class="t0">0 s</span><span class="t1" style="left:{end / dur * 100:.2f}%">'
            f'{end:g} s</span></span></li></ol>')


def fragments() -> dict:
    out = {}
    for name, (scene, _, _) in DEMO.items():
        path = ROOT / scene
        if not scene.startswith("site/"):  # examples: keep a copy beside the video
            shutil.copyfile(path, DEMOS / path.name)
            path = DEMOS / path.name
        out["json", name] = json_block(path)
        summary = probe(str(ROOT / scene))
        out["score", name] = score_block(summary, f"Timeline of {path.name}")
    out["score", "hero"] = score_block(probe(str(ROOT / HERO[0])), f"Timeline of {HERO[0]}")

    before = (ROOT / CUES).read_text(encoding="utf-8")
    after = before.replace(*CUES_CHANGE, 1)
    assert after != before, "the cue change no longer applies to cues.json"
    out["diff", "cues"] = diff_block(before, after)
    sb, sa = probe(json.loads(before)), probe(json.loads(after))
    axis = max(sb["canvas"]["duration"], sa["canvas"]["duration"])
    out["score", "cues-before"] = score_block(sb, "Timeline before the change", axis)
    out["score", "cues-after"] = score_block(sa, "Timeline after the change", axis)

    out["json", "problems"] = json_block(ROOT / CHECK)
    res = check(str(ROOT / CHECK))
    out["check", "problems"] = html.escape(json.dumps(res, indent=2, ensure_ascii=False))

    name, at = PROBE
    p = probe(str(ROOT / DEMO[name][0]), at)
    keep = ("title", "card-1-back", "card-1-new", "card-2-old")
    vis = {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("rect", "alpha", "type", "scale", "text")}
           for k, v in p["visible"].items() if k in keep}
    for v in vis.values():
        v["rect"] = [round(x) for x in v["rect"]]
    shown = {"at": at, "visible": vis}
    text = json.dumps(shown, indent=2, ensure_ascii=False)
    text = re.sub(r"\[\s+([^\[\]]+?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    out["probe", name] = html.escape(text)
    return out


MARK = re.compile(r"(<!--build:(\w+):([\w-]+)-->)(.*?)(<!--/build-->)", re.S)


def write_page() -> None:
    frags = fragments()
    page = PAGE.read_text(encoding="utf-8")
    seen = set()

    def fill(m):
        key = (m.group(2), m.group(3))
        if key not in frags:
            print(f"warning: no fragment for {key}")
            return m.group(0)
        seen.add(key)
        return m.group(1) + frags[key] + m.group(5)

    page = MARK.sub(fill, page)
    for key in sorted(set(frags) - seen):
        print(f"note: fragment {key} has no marker in the page")
    PAGE.write_text(page, encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}")


def main(argv: list[str]) -> None:
    names = [a for a in argv if not a.startswith("-")]
    if "--page" not in argv:
        if not (DEMOS / "sound" / "bed.m4a").exists() or "sound" in names:
            sound_assets()
        for scene in (DEMO["voice"][0], DEMO["voices"][0], HERO[0]):  # speaks only the lines that changed since their files were made
            run(*KINE, "voice", scene)
        for name in names or [*DEMO, "hero", "look"]:
            if name == "sound":
                continue
            if name == "hero":
                scene, at, scale = HERO
                render(scene, MEDIA / "hero.mp4", scale, at)
            elif name == "look":
                look_sheet()
            else:
                scene, at, scale = DEMO[name]
                render(scene, DEMOS / f"{name}.mp4", scale, at)
    if not names or "--page" in argv:
        write_page()


if __name__ == "__main__":
    main(sys.argv[1:])
