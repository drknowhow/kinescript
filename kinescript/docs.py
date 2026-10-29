"""The scene reference printed by `kine docs` and served to agents by the MCP server."""
from __future__ import annotations

from .components import COMPONENT_DOCS
from .ease import EASES
from .presets import PRESETS
from .shaders import PRESETS as SHADERS

REFERENCE = """\
KINESCRIPT SCENE REFERENCE
A scene is JSON. Frame n depends only on the scene and n, so any frame renders alone.

{
  "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": "#0a0c10", "duration": "auto"},
  "theme":  {"accent": "#ff633f"},          // "$accent" anywhere = this value
  "markers": {"drop": 12.5},                 // named times: "at": "drop"
  "grade":  {"contrast": 1.05, "saturate": 1.1, "vignette": 0.35, "grain": 0.05},
  "motionBlur": {"samples": 6, "shutter": 0.5},
  "audio":  [{"src": "music.mp3", "at": 0, "gain": -8, "fadeIn": 1, "fadeOut": 2}],
  "components": {"name": {"params": {...}, "layers": [... "{{param}}" ...]}},
  "layers": [ ... drawn in order, later on top ... ]
}

TIME (at / dur / until)
  1.5 | "1.5s" | "45f" | "0:02.5" | "title.out+0.2" | "prev.in+10f" | "title.mid" | "drop" | "start"
  until: "end" runs to the scene end. No dur/until: runs to the end (video: its trimmed length).
  Children of a group use times relative to the group start.

LAYER (common)
  id, type, at, dur, until, x, y, anchor (center|topleft|top|...|bottomright),
  opacity, scale, scaleX, scaleY, rotate (deg), blur, blend (multiply|screen|overlay|add|...),
  shadow {x, y, blur, color}, fx {brightness, contrast, saturate, hue, exposure, temperature, tint, gray, invert},
  enter, exit, hidden, allowOffscreen
  Lengths: px, "50%" (of canvas or group w/h), "50%-120", "10vw", "5vh".

ANIMATING ANY NUMBER OR COLOUR
  constant         "x": 960
  keyframes        "x": [[0, 200], [0.8, 960, "outBack"], [-0.5, 960], ["end", 1700, "inCubic"]]
                   time = s from layer start; negative = from its end; "40%" = of its length
  tween            "opacity": {"from": 0, "to": 1, "dur": 0.4, "delay": 0.2, "ease": "outCubic"}
  expression       "y": "=540 + sin(t*3)*40"   names: t d p T i W H fps; helpers: sin cos tan sqrt
                   abs min max clamp lerp noise(x,seed) wiggle(freq,amp,t,seed) ease(name,t)
  Colours blend in OKLab, so midpoints stay vivid.

ENTER / EXIT
  "enter": "rise" | "rise+blur" | {"name": "pop", "dur": 0.4, "ease": "outBack", "dist": 60, "stagger": 0.03}
  motions: {motions}

TYPES
  rect       w h radius fill stroke strokeWidth draw(0..1 trims the stroke) dimOutside(0..1)
  circle     r fill stroke strokeWidth draw          ellipse  w h ...
  polygon    sides r inner(star ratio) ...           line     points [[x,y]...] smooth closed
  path       d (SVG path data) | points ...          arrow    x1 y1 x2 y2 bend head stroke strokeWidth draw
  text       text ("**word**" uses emph colour) size weight font(family or .ttf path) italic fill emph
             align maxWidth lineHeight letterSpacing stroke box{fill,pad,radius} case(upper|lower) caret
             split (chars|words|lines) + enter.stagger animates each unit
             unitX unitY unitRotate unitScale unitOpacity: per-unit expressions using i
  counter    value (animate it) format "{:,.0f}" prefix/suffix inside format: "${:,.2f}M"; text props
  image      src w h fit(cover|contain|stretch) zoom focus[fx,fy] focusX focusY radius stroke
  video      src trim[a,b] speed volume loop w h fit zoom focus radius   (audio is mixed in automatically)
  captions   src (.srt/.vtt) | cues [[start,end,"text"]...] mode(karaoke|pop) active size fill box maxWidth
  shader     preset ({shaders}) uniforms{name: value|colour} | sksl "<SkSL source>"
             built-in uniforms: iTime iResolution iProgress
  group      children [...] w h clip radius fill   (children positioned from the group origin)
  sequence   clips [{src, trim}...] transition {type: fade|wipe|slide|zoom|blur|cut, dur}
             (expands to back-to-back video layers)

COMPONENTS  {"use": "<name>", ...params, "at": ..., "dur": ...}
{components}

EASES  {eases}, spring, spring(stiffness,damping), bezier(x1,y1,x2,y2), steps(n)

WORKFLOW FOR AGENTS
  1. write the scene      2. kine check  (fix every error, read the warnings)
  3. kine look --boxes    (one image of many frames; look at it)
  4. kine render -q draft for timing, then the final render
"""


def reference() -> str:
    comps = "\n".join(f"  {k:<11}{v}" for k, v in COMPONENT_DOCS.items())
    eases = " ".join(sorted(k for k in EASES if k not in ("in", "out", "inOut")))
    return (REFERENCE.replace("{motions}", " ".join(sorted(PRESETS)))
            .replace("{shaders}", "|".join(SHADERS)).replace("{components}", comps).replace("{eases}", eases))
