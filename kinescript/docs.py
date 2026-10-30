"""The scene reference printed by `kine docs` and served to agents by the MCP server.

Written for small models as well as large ones: rules first, one complete example, every default
stated, then recipes to copy. Names of motions, eases, shaders and components are filled in from
the code, so they cannot drift; tests/test_docs.py renders every JSON example in this text.
"""
from __future__ import annotations

from .components import COMPONENT_DOCS
from .ease import EASES
from .presets import PRESETS
from .shaders import PRESETS as SHADERS

REFERENCE = """\
KINESCRIPT SCENE REFERENCE
A scene is one JSON object. Kinescript turns it into a video. Frame n depends only on the scene
and n, so any frame can be rendered alone.

RULES
  1. Write strict JSON: double quotes, no comments, no trailing commas.
  2. Every layer has a "type" and a unique "id".
  3. Times are in seconds. Sizes are in pixels. Colours are "#rrggbb" or "#rrggbbaa".
  4. x and y say where the layer's anchor goes. They default to 0, the TOP-LEFT corner of the
     canvas. For the centre write "x": "50%", "y": "50%". The anchor defaults to "center".
  5. Layers draw in list order: later layers are on top. Put backgrounds first.
  6. A layer without "dur" or "until" stays until the scene ends. Nothing fades out unless you
     give it an "exit", e.g. "exit": "fade".
  7. Use only the keys listed here. `kine check` rejects unknown keys and names the right one.
  8. After writing a scene, run `kine check scene.json` and fix every error before rendering.

A COMPLETE SCENE (copy it and change it)
{
  "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": "#0a0c10"},
  "layers": [
    {"id": "bg", "type": "shader", "preset": "score", "opacity": 0.6},
    {"id": "title", "type": "text", "text": "Launch **Day**", "x": "50%", "y": "45%", "size": 140,
     "weight": 800, "fill": "#f2f0e8", "emph": "#ff633f", "at": 0.3, "dur": 3,
     "split": "chars", "enter": {"name": "rise", "dur": 0.5, "stagger": 0.03}, "exit": "fade"},
    {"id": "sub", "type": "text", "text": "October 2026", "x": "50%", "y": "60%", "size": 48,
     "fill": "#9299a8", "at": "title.in+0.8", "until": "title.out", "enter": "fade", "exit": "fade"},
    {"id": "bar", "type": "rect", "x": "50%", "y": "68%", "w": 320, "h": 6, "fill": "#ff633f",
     "at": 1.2, "dur": 2, "enter": "wipe", "exit": "fade"}
  ]
}
It is 3.3 s long: with canvas duration "auto", the scene ends when the last timed layer ends.

SCENE (top level)
  canvas      {"w": 1920, "h": 1080, "fps": 30, "bg": "#0a0c10", "duration": "auto"}
              w and h 16..16384, fps 1..240. duration: seconds, or "auto" (the default: the
              latest end of any layer). Vertical video: "w": 1080, "h": 1920.
  layers      the list of layers.
  theme       {"accent": "#ff633f"}: then "$accent" anywhere means that value.
  markers     {"drop": 12.5}: named times, usable as "at": "drop".
  grade       whole-frame look: {"contrast": 1.05, "saturate": 1.1, "vignette": 0.35, "grain": 0.05}
  motionBlur  {"samples": 6, "shutter": 0.5}   (slower: renders 6 sub-frames per frame)
  camera      moves and turns the view in depth (see DEPTH AND CAMERA).
  light       how model layers are lit (see 3D MODELS).
  audio       the list of sounds: voice-over, music, effects (see SOUND).
  mix         loudness, role levels and ducking of the sound (see SOUND).
  components  your own reusable layer groups (see COMPONENTS).

TIME: "at" (start), "dur" (length), "until" (end)
  1.5               seconds
  "45f"             frames
  "0:02.5"          minutes:seconds
  "title.in"        when layer "title" starts; "title.out" when it ends; "title.mid" its middle
  "title.out+0.2"   0.2 s after title ends;  "title.in-10f" 10 frames before title starts
  "prev.out"        when the layer listed just before this one ends
  "drop"            a marker from "markers"
  "until": "end"    runs to the scene end
  "at" defaults to 0. Inside a group, times count from the group's start.
  A video without dur lasts as long as its trimmed clip.

KEYS EVERY LAYER HAS
  id, type, at, dur, until
  x, y          position of the anchor. A number (px), "50%" (of the canvas, or of the group's
                w and h), "50%-120", "10vw", "5vh", or an expression.
  anchor        center (default) | topleft | top | topright | left | right | bottomleft | bottom |
                bottomright. Video, shader, and image without x default to topleft.
  opacity       0..1 (default 1)
  scale, scaleX, scaleY   1 = normal size
  rotate        degrees, clockwise
  z, rotateX, rotateY, backface, fixed   depth: see DEPTH AND CAMERA
  blur          px
  blend         normal | multiply | screen | overlay | add | lighten | darken | softLight |
                difference | color
  shadow        {"x": 0, "y": 6, "blur": 16, "color": "#00000099"}   (those are the defaults)
  fx            {"brightness": 0, "contrast": 1, "saturate": 1, "hue": 0, "exposure": 0,
                 "temperature": 0, "tint": 0, "gray": 0, "invert": 0}. brightness adds to every
                channel (-1..1), so below 0 dark footage turns black; exposure is in stops and
                multiplies (-1 halves the light), which is the way to darken footage.
  enter, exit   a motion (see ENTER AND EXIT)
  hidden        true skips the layer (and its sound)
  sound         a file played when the layer starts, timed by the layer (see SOUND)
  allowOffscreen  true stops check warning when the layer leaves the canvas
  note          free text, ignored

LAYER TYPES (defaults in brackets)
  rect      w [200] h [120] radius [0] fill [#f2f0e8 unless stroke is set] stroke strokeWidth [4]
            dimOutside (0..1: darkens everything outside the rect, to highlight part of a video)
  circle    r [50] fill stroke strokeWidth
  ellipse   w h fill stroke strokeWidth
  polygon   sides [6] r [80] inner (0..1 makes a star) fill stroke strokeWidth
  line      points [[x, y], [x, y], ...] (measured from the layer's x, y) stroke strokeWidth
            smooth closed
  path      d (SVG path data, e.g. "M0 0 L100 0") or points; fill stroke strokeWidth
  arrow     x1 y1 x2 y2 bend [0] head [22] stroke strokeWidth
  text      text (centred on x, y: the anchor is center), size [64], weight [600] (100..900), font ["Space Grotesk"; also "IBM Plex Mono",
            any installed family, or a .ttf path], italic, fill [#f2f0e8], emph (colour of
            text written **like this**, also inside a word: g**P**u colours the P),
            align [center] (left|center|right), maxWidth (wraps lines), lineHeight [1.18],
            letterSpacing [0], stroke, strokeWidth,
            box {"fill": "#000000b0", "pad": [12, 20], "radius": 10} (a background behind the
            text; pad is [vertical, horizontal]), case (upper|lower),
            split (chars|words|lines: animate each unit, see ENTER AND EXIT),
            caret (true shows a cursor while typing),
            unitX unitY unitRotate unitScale unitOpacity (expressions per unit; i is its index)
  counter   value (animate it, e.g. {"from": 0, "to": 1250, "dur": 2}), format ["{:,.0f}"; also
            "${:,.2f}M" or "{:.0f}%"], and every text key
  image     src (every src, for image, video, captions and audio, is relative to the scene
            file), w, h (give one and the other keeps the
            aspect), fit [cover] (cover|contain|stretch), zoom [1] (zooms inside the frame),
            focus [[0.5, 0.5]] (the point zoom moves toward), radius, stroke, strokeWidth
  video     src, trim [start, end] (seconds in the source file), speed [1], volume [1] (0 mutes),
            loop, w, h [canvas size], fit, zoom, focus, radius. Its sound is mixed in.
  captions  cues [[start, end, "text"], ...] (seconds from the layer start) or src (.srt/.vtt),
            or voice: a sound's id, to caption what it says word by word (see SOUND),
            mode (karaoke|pop: highlight the current word), active (colour of that word), and
            the text keys. Defaults: x "50%", y "86%", size 46, weight 700, a dark box behind.
  shader    preset ({shaders}) with uniforms {"name": number or colour},
            mesh and score are subtle; aurora and grid are strong, so use opacity 0.2-0.4.
            Or sksl: your own SkSL "half4 main(float2 p) { ... return half4(r, g, b, a); }".
            p is the pixel position (0..w, 0..h); p / iResolution is 0..1. The colour is
            premultiplied: for a see-through colour return half4(rgb * a, a). iTime (seconds),
            iResolution, iProgress (0..1) and your uniforms are declared for you.
            Fills the canvas unless you give w and h.
  model     a lit 3D solid, 3D text, an extruded SVG path or a model file: shape, src or parts
            (see 3D MODELS).
  group     children [layers], w, h, clip (true cuts children off at w x h), radius, fill.
            Children are placed from the group's x, y and start at the group's "at".
  sequence  clips [{"src": "a.mp4", "trim": [0, 4]}, ...] and transition {"type": "fade",
            "dur": 0.4} (fade|wipe|slide|zoom|blur|cut). Becomes a group of back-to-back video layers
            named <id>-1, <id>-2, ... that ends with its last clip, so "<id>.out" is a cue. Each
            transition overlaps the clips it joins, so 3 s + 3 s with a 0.5 s fade lasts 5.5 s.
            fit, volume, w, h, x, y and anchor go to every clip; opacity, blur, fx, enter and the
            other layer keys apply to the whole cut.

DEPTH AND CAMERA (3D)
  Any layer can move and turn in depth. x is right, y is down and z is toward the viewer, in px,
  as in CSS. A scene that uses depth is seen through a camera; at z 0 everything keeps its size.
  z           px toward the viewer. Near layers grow; far ones (negative z) shrink and drift less
              when the camera moves.
  rotateX     degrees: positive tips the top edge away.
  rotateY     degrees: positive turns the right edge away. Past 90 you see the back, mirrored.
  backface    visible (default) | hidden: hidden skips the layer while its back faces the camera.
  fixed       true: a top-level layer ignores the camera's moves (captions are fixed).
  sort        on a group: "depth" draws its children far to near each frame, not in list order.
  camera      top level. Every key animates, and all default to 0 except fov:
              {"x": 0, "y": 0, "z": 0, "rotateX": 0, "rotateY": 0, "rotate": 0, "fov": 40}
              x and y pan (px). z moves the camera back; negative moves it in. rotateY orbits it
              right around the canvas centre, rotateX up over it (looking down), rotate rolls it.
              fov (5..150) sets how strong perspective looks; it does not change size at z 0.
  Motions: flip turns in on the vertical axis (deg [90]), flip-up on the horizontal axis, tumble
  spins in from depth, push comes forward from depth (dist [600]). With "split" each unit flips.
  Layers still draw in list order, so put far layers first, or use a group with "sort": "depth".
  A card that turns over: rotateY on the group, and each side hides its back:
    {"id": "card", "type": "group", "x": "50%", "y": "50%", "rotateY": [[0.5, 0], [1.5, 180]], "children": [
     {"id": "card-front", "type": "group", "anchor": "center", "w": 560, "h": 150, "radius": 28,
      "fill": "#f2f0e8", "backface": "hidden", "children": [
      {"id": "card-front-text", "type": "text", "text": "front", "size": 64, "fill": "#0a0c10"}]},
     {"id": "card-back", "type": "group", "anchor": "center", "w": 560, "h": 150, "radius": 28,
      "fill": "#ff633f", "rotateY": 180, "backface": "hidden", "children": [
      {"id": "card-back-text", "type": "text", "text": "back", "size": 64, "fill": "#0a0c10"}]}]}
  A whole scene: the camera pushes in and swings round layers at three depths; the note stays put:
    {"canvas": {"w": 1920, "h": 1080, "fps": 30, "duration": 4},
     "camera": {"z": [[0, 300], ["end", -100]], "rotateY": [[0, -12], ["end", 10]]},
     "layers": [
      {"id": "far", "type": "rect", "x": "50%", "y": "50%", "z": -600, "w": 1600, "h": 900, "radius": 40,
       "fill": "#151922"},
      {"id": "title", "type": "text", "text": "Depth", "x": "50%", "y": "45%", "size": 140, "weight": 800,
       "split": "chars", "enter": {"name": "flip", "stagger": 0.05}},
      {"id": "near", "type": "circle", "x": "78%", "y": "72%", "z": 250, "r": 60, "fill": "#ff633f"},
      {"id": "note", "type": "text", "text": "fixed to the screen", "x": "50%", "y": "88%", "size": 40,
       "fixed": true}]}

3D MODELS
  A model layer is a lit solid in the space of DEPTH AND CAMERA, so z, rotateX, rotateY, rotate,
  scale and the camera apply to it. x and y place its centre.
  shape     cube (w [200], h and depth [= w]) | sphere (r [100]) | cylinder (r [80], h [200]) |
            cone (r [90], h [200], tip up) | torus (r [110], tube [r x 0.35], facing you) |
            plane (w, h; both sides show) | text | path. Sizes are px and animate.
  text      shape "text" pushes letters out in depth: text ("\n" breaks lines), size [120],
            weight [600], font, italic, letterSpacing, lineHeight, depth [size x 0.2], bevel [0]
            (px of 45-degree edge). Centred on x, y. Each new text, size, depth or bevel value
            rebuilds the solid, so grow it with scale.
  path      shape "path": d (SVG path data, filled nonzero), w and/or h (fits inside them, keeping
            its shape), depth [a fifth of its larger side], bevel [0], strokeWidth (in d's units:
            extrudes the stroke, with round ends, for line icons).
  detail    low | medium (default) | high: how many triangles. A medium sphere has about 2,200.
  fill      one colour [theme accent]   shine 0..1 [0.4]   metal 0..1 [0]   rim 0..1 [0.5] (edge
            light)   bands [0] (2 or more: flat toon steps). These animate.
  parts     [{"shape": "sphere", "r": 30, "x": 0, "y": -120, "z": 0, "rotateX": 0, "rotateY": 0,
            "rotate": 0, "scale": 1, "fill": "#b9f45d", "shine": 0.9}, ...] placed from the
            model's centre, in plain numbers (parts do not animate; the model does). A part takes
            the model's material keys it does not set. A model may have shape (or src), parts, or both.
  src       a model file: .glb or .gltf (glTF 2.0), or .obj with its .mtl. Colours, shine and metal
            come from the file; a fill on the layer recolours all of it. It fits inside w and/or h
            keeping its shape, else its longest side is 300 px. Node transforms apply; animation,
            skins and textures do not (a texture gives its mean colour). Long triangles are cut to
            1/24 of its size for the depth sort. Draco- or meshopt-compressed files are not read.
  ground    true or {"opacity": 0.5, "blur": px, "y": px}: a soft shadow on a level floor under the
            model, as if lit from above, fainter as the model rises. The floor sits at the model's
            lowest point before it turns, or at y in the layer's parent space (canvas px at the top
            level): give y when the model moves up and down.
  explode   0..1: faces fly out along their normals.
  draw      0..1, or "enter": "build": triangles appear from the bottom up.
  light     top level, or on one model: studio (default) | soft | dramatic | flat (no shading),
            or {"preset": "studio", "key": 0.85, "fill": 0.25, "ambient": 0.22, "rim": 0.35,
            "from": "topleft", "color": "#fff4e6"}. from: topleft | top | topright | left | right |
            front | bottomleft | bottom | bottomright. Lights move with the camera.
  Triangles are sorted far to near within one model, so things that touch or pass through each
  other go in one model as parts. Where two surfaces cut through each other the seam is stepped,
  so let parts touch rather than cross. Separate layers draw in list order. check warns past
  60,000 triangles in a frame.
  A ring around a ball that builds in, turns, and flies apart:
    {"id": "emblem", "type": "model", "x": "50%", "y": "45%", "detail": "high",
     "rotateX": [[0, 64], ["end", 18]], "rotateY": [[0, -30], ["end", 70]],
     "enter": {"name": "build", "dur": 1.6}, "explode": [[3, 0], ["end", 1]],
     "parts": [{"shape": "torus", "r": 250, "tube": 70, "fill": "#ff633f", "metal": 0.45, "shine": 0.7},
               {"shape": "sphere", "r": 100, "fill": "#b9f45d", "shine": 0.9}]}
  A 3D title that builds in, turns, and settles:
    {"id": "title3d", "type": "model", "shape": "text", "text": "LAUNCH", "x": "50%", "y": "50%",
     "size": 240, "weight": 800, "depth": 70, "bevel": 5, "shine": 0.6,
     "rotateY": [[0, -40], [2.4, 0, "outCubic"]], "rotateX": [[0, 18], [2.4, 0, "outCubic"]],
     "enter": {"name": "build", "dur": 1.2}}
  A product turntable from a model file, on a soft floor shadow:
    {"id": "product", "type": "model", "src": "rocket.glb", "x": "50%", "y": "46%", "h": 600,
     "rotateY": [[0, 0], ["end", 360]], "ground": true, "enter": {"name": "build", "dur": 1.6}}

ANIMATING A VALUE (any number or colour key)
  constant     "x": 960
  tween        "opacity": {"from": 0, "to": 1, "dur": 0.6, "delay": 0, "ease": "outCubic"}
               (those are the defaults; delay counts from the layer start)
  keyframes    "x": [[0, 200], [0.8, 960, "outBack"], ["end", 1700]]
               Each keyframe is [time, value] or [time, value, ease into it]. Time is seconds from
               the layer start; a negative time counts back from its end; "40%" is a share of its
               length; "end" is its end. Between keyframes the default ease is inOutCubic.
               A flat list such as [0, 100] is NOT keyframes.
  expression   "y": "=540 + sin(t * 3) * 40"   (a string starting with "=")
               t = seconds since the layer started   d = the layer's length   p = t / d (0..1)
               T = seconds since the scene started   i = unit index (split text)   W, H, fps
               helpers: sin cos tan sqrt abs min max round floor ceil exp log atan2 hypot pi tau
                        clamp(v, lo, hi)  lerp(a, b, p)  noise(x, seed)  wiggle(freq, amp, t, seed)
                        ease("outCubic", p)
  Colours animate too: "fill": {"from": "#ff633f", "to": "#b9f45d", "dur": 1}.
  Gradient fill: "fill": {"linear": [[0, 0], [1, 1]], "stops": ["#ff633f", "#7b5cff"]}
                 or {"radial": [0.5, 0.5, 0.6], "stops": [...]}; points are 0..1 of the layer box.

ENTER AND EXIT
  "enter": "rise"   or   {"name": "rise", "dur": 0.5, "ease": "outCubic", "stagger": 0.03}
  Defaults: dur 0.5, ease outCubic (pop and spin: outBack; type: linear; draw: inOutCubic;
  snap: outQuart). An exit plays during the LAST dur seconds of the layer and ends with it.
  Combine with +: "rise+blur". Options: dist (px moved by rise, drop, left, right; default 48),
  from (start scale of pop [0.4] and zoom [1.25]), amount (blur px [24], track px [40]).
  With "split" on text, each unit enters in turn: unit k starts k x stagger seconds after the
  layer starts, so the whole reveal takes dur + (units - 1) x stagger.
  motions: {motions}
  rise moves up into place, drop moves down, left moves left, right moves right. type is a
  typewriter. draw draws a stroke or path on, and builds a model (build is the same). wipe reveals left to right. snap is a hard cut in
  with a small settle. flip, flip-up, tumble and push move in depth (see DEPTH AND CAMERA).
  none is no motion.

EASES  {eases}, spring, spring(stiffness, damping), bezier(x1, y1, x2, y2), steps(n)
  in* starts slow, out* ends slow (the most natural for entrances), inOut* does both.

COMPONENTS  {"use": "<name>", "id": "...", "at": ..., "dur": ..., ...parameters}
  Built in (parameters):
{components}
  x and y place the component as noted above. Components also accept enter, exit, opacity,
  scale, rotate, hidden, sound.
  Your own: "components": {"badge": {"params": {"label": "NEW"}, "layers": [
              {"type": "text", "text": "{{label}}", "size": 40}]}}
  then a layer {"use": "badge", "label": "BETA", "x": 200, "y": 100}.

SOUND
  Top-level "audio" lists the sounds. Each one:
    {"id": "vo1", "role": "voice", "src": "vo1.wav", "at": 0.5}
  src (relative to the scene file), id, role (voice|music|sfx|ambience), at, dur, until,
  trim [start, end] (seconds in the file), gain (dB: 0 = unchanged, -6 = about half as loud) or
  volume (1 = unchanged), fadeIn, fadeOut (seconds), loop (true repeats the trimmed file until the
  sound ends), pan (-1 left .. 1 right), text, speaker and rate (a voice spoken from text, see
  below). A sound lasts as long as its trimmed file; fadeOut ends
  where it ends, which is the scene end at the latest.
  A sound is a cue like a layer: "at": "vo1.out+0.3" starts something 0.3 s after the voice ends,
  and a sound can start at "title.in". "prev" is the sound listed before. A voice makes
  "duration": "auto" long enough to finish it; music, ambience and sfx do not.
  Music under the whole video:
    {"id": "bed", "role": "music", "src": "music.mp3", "gain": -10, "loop": true, "fadeIn": 1,
     "fadeOut": 2}
  A layer's sound moves when the layer moves:
    "sound": "pop.wav"   plays when the layer starts
    "sound": {"src": "whoosh.wav", "at": "out-0.1", "gain": -6}
  Its at is "in", "out" or "mid" with an offset, or seconds after the layer starts; it also takes
  trim, dur, gain, volume, fadeIn, fadeOut, pan and role (sfx unless you say).
  gain changes over time like any property, timed from the sound's start: "gain": [[0, -30], [2, 0]]
  fades a sound up over its first 2 s.
  Give sounds a role and the mix is levelled for you: voice -18 LUFS, music -16, ambience -30, and an
  sfx peaks at -6 dBFS; gain on a sound with a role is a change from that level. Music and ambience
  dip 14 dB while a voice speaks. An sfx starts past the silence at the start of its file, so its
  first sound lands on its cue. The whole mix ends at -16 LUFS with peaks under -1.5 dBTP. Sounds
  without a role keep the gain you give them, and a scene where no sound has a role is mixed as
  given. Change the targets with a top-level "mix", for example
  "mix": {"loudness": -14, "peak": -1, "levels": {"music": -20}, "duck": {"depth": 10}}.
  "duck": false turns the dip off; duck also takes by and under (lists of roles), and attack, hold
  and release in seconds (0.08, 0.15, 0.4).
  Cue from what a voice says. Its word times come from a words file next to it, vo1.words.json for
  vo1.wav, in seconds of the file:
    {"words": [{"word": "Write", "start": 0.12, "end": 0.41}, {"word": "render.", "start": 1.2, "end": 1.5}]}
  Whisper's JSON reads as it is; "words" on the sound names another file or holds the list. A word's
  id is the word in lower case without punctuation, with _2, _3 for repeats; kine probe lists the ids
  and when each is said. "at": "vo1:render" starts a layer as the voice says render, "vo1:render.out"
  as it finishes the word, and "vo1:render_2.in+0.1" works like any cue. Captions of the voice, each
  word lit as it is said:
    {"id": "cc", "type": "captions", "voice": "vo1", "mode": "karaoke"}
  A voice from text: write what it says and kine voice speaks it (see COMMANDS):
    {"id": "vo1", "role": "voice", "text": "Write a scene. Check it. [[go]]Render it.", "speaker": "af_heart"}
  It writes voice/vo1.wav (or the sound's src) and its words file, so cues and captions work as above.
  [[name]] gives the next word that id: "at": "vo1:go" starts as it says Render, whatever the line
  says later. speaker is the backend's voice (Kokoro: af_heart, af_bella, am_adam, bf_emma, bm_george),
  rate (0.5 to 2, default 1) how fast it speaks. kine check reports a sound with text that is not spoken
  yet (voice-missing) and one whose text, speaker or rate changed since it was spoken (voice-stale).
  You cannot hear the mix, so read it: kine check measures it and names each sound problem with its
  time range, and kine look --audio draws it under the frames, with each sound's loudness in the JSON.
  Sound is mixed into .mp4, .mov and .webm, and into --from/--to renders. -o mix.wav (or .m4a,
  .mp3, .flac) renders the sound alone, without drawing frames.

RECIPES (layers to paste into "layers")
  Title that types on:
    {"id": "hello", "type": "text", "text": "Hello world", "x": "50%", "y": "50%", "size": 120,
     "enter": {"name": "type", "dur": 1.2}, "caret": true}
  Words one after another, each replacing the last, centred at x, y:
    {"use": "kinetic", "id": "words", "words": ["Write", "Check", "Render"], "every": 0.6}
  Pill label:
    {"id": "pill", "type": "text", "text": "NEW", "x": "50%", "y": "50%", "size": 40,
     "fill": "#0a0c10", "box": {"fill": "#b9f45d", "pad": [10, 24], "radius": 999}}
  Bar of a bar chart growing up from a baseline at y 900, with its value riding on top and its
  label under the baseline. The value's y uses the same tween as the bar's height (same dur,
  same default ease), so it stays 40 px above the bar. The label is anchored at its top edge,
  20 px below the baseline. Keep text off the bar: light text over the orange fails the
  contrast check:
    {"id": "bar1", "type": "rect", "x": 600, "y": 900, "anchor": "bottom", "w": 120,
     "h": {"from": 0, "to": 400, "dur": 0.8}, "fill": "#ff633f", "at": 0.5}
    {"id": "bar1-value", "type": "counter", "x": 600, "y": {"from": 860, "to": 460, "dur": 0.8},
     "size": 48, "at": 0.5, "value": {"from": 0, "to": 80, "dur": 0.8}, "format": "{:.0f}"}
    {"id": "bar1-label", "type": "text", "text": "Q1", "x": 600, "y": 920, "anchor": "top",
     "size": 40, "at": 0.5}
  Slow zoom on a photo (Ken Burns):
    {"id": "photo", "type": "image", "src": "photo.jpg", "w": 1920, "h": 1080, "anchor": "topleft",
     "zoom": [[0, 1], ["end", 1.15, "linear"]], "focus": [0.6, 0.4]}
  Two clips with a crossfade:
    {"id": "cut", "type": "sequence", "clips": [{"src": "a.mp4", "trim": [0, 4]},
     {"src": "b.mp4", "trim": [2, 6]}], "transition": {"type": "fade", "dur": 0.5}}
  Full-screen gradient background:
    {"id": "bg", "type": "rect", "x": 0, "y": 0, "anchor": "topleft", "w": "100%", "h": "100%",
     "fill": {"linear": [[0, 0], [1, 1]], "stops": ["#1e3a8a", "#6d28d9"]}}
  Progress bar along the bottom for the whole video:
    {"use": "progress", "id": "progress", "color": "#ff633f", "h": 8}
  Music under everything: top-level "audio": [{"role": "music", "src": "music.mp3", "gain": -10,
    "loop": true, "fadeIn": 1, "fadeOut": 2}]

COMMANDS
  kine new scene.json             write a starter scene
  kine check scene.json           JSON list of problems. error: must fix. warning: should fix:
                                  text off the canvas or inside the outer 5% (safe-area),
                                  overlapping text, contrast under 3:1, or text shown for less
                                  than about 0.4 s + 0.3 s per word (read-time). With sound (it is
                                  mixed and measured, see SOUND): a voice cut off (audio-cut), two
                                  voices at once (voice-overlap), music under 10 LU below a voice
                                  (voice-masked), an sfx whose first sound is late (sfx-late), a
                                  silent sound, and the mix's loudness and peaks.
                                  info: a name it read for you, e.g. "'color' read as 'fill'".
  kine look scene.json --boxes    one PNG of 12 evenly spaced frames; --boxes outlines layers and
                                  labels the top-level ones; -n 16 frames, --at 1 2.5 4 exact
                                  times (use it to see a transition), --width 480, --cols 4, -o.
                                  --audio adds the sound under the frames on the same time axis:
                                  a lane per role and one for the mix, the duck, and issues in red;
                                  its JSON gives each sound's loudness and gain and the mix's.
  kine still scene.json --at 2.5  one frame as PNG (-o to name it); --at -1 is 1 s before the
                                  end; --at end is the last frame
  kine probe scene.json --at 2    every layer's and sound's start and end, and screen boxes at 2 s
  kine probe clip.mp4             a media file's duration, w, h, fps and whether it has audio
  kine render scene.json -o out.mp4 -q draft|standard|high
                                  standard is the default, draft the fastest, high the best.
                                  Also .mov or .webm (with --transparent for no background),
                                  .gif (longest side 720 px, 15 fps), or a folder/ for PNG frames.
                                  .wav, .m4a, .mp3 or .flac: the sound alone.
                                  --scale 0.5 halves the size; --from 2 --to 4 renders a part.
  kine voice scene.json           speaks each sound's text into its file and times its words; a sound
                                  is spoken again only when its text, speaker or rate changed
                                  (--force: all of them, --only vo1: one). Needs a voice backend,
                                  the Kokoro plugin (with none installed, kine voice prints the pip
                                  command), or KINE_VOICE_CMD set to a command that reads JSON
                                  requests on stdin. Rendering never speaks.
  kine watch scene.json           re-makes the contact sheet on every save
  kine doctor                     checks ffmpeg, the GPU and fonts
  kine mcp                        the same tools for MCP clients

WORKFLOW FOR AGENTS
  1. Write the scene (if a sound has text, kine voice).   2. kine check: fix every error, read the warnings, run it again.
  3. kine look --boxes, and look at the image (with sound, kine look --audio).
  4. kine render -q draft, then the final render.
"""


def reference() -> str:
    comps = "\n".join(f"    {k:<11}{v}" for k, v in COMPONENT_DOCS.items())
    eases = " ".join(sorted(k for k in EASES if k not in ("in", "out", "inOut")))
    return (REFERENCE.replace("{motions}", " ".join(sorted(PRESETS)))
            .replace("{shaders}", "|".join(SHADERS)).replace("{components}", comps).replace("{eases}", eases))
