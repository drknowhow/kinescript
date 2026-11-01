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
  2. Every layer has a "type" and a unique "id". An id that cues name is letters, digits, _ and -
     (no spaces or dots), and it cannot be start or prev: cues read those words first.
  3. Times are in seconds. Sizes are in pixels. Colours are "#rrggbb" or "#rrggbbaa" (aa is
     opacity), also #rgb, rgb(255, 99, 63) and rgba(255, 99, 63, 0.5). rgb values past 0..255 are
     clamped, with a warning. The colour names, in any case, are black, white, transparent, gray
     (grey), and the iOS-style red #ff3b30, green #34c759, blue #0a84ff, yellow #ffd60a, orange
     #ff9f0a, purple #bf5af2, pink #ff375f, cyan #64d2ff. They are not the CSS colours.
     null for an optional value means its default.
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
              latest end of any layer). The video has duration x fps frames, to the nearest
              frame, a half rounding up (2.45 s at 30 fps: 74). Vertical video: "w": 1080, "h": 1920.
              Keep w and h even: .mp4, .webm and .gif need it, so an odd size loses its last
              column or row there.
  layers      the list of layers.
  theme       {"accent": "#ff633f"}: then "$accent" anywhere means that value.
  markers     {"drop": 12.5}: named times, usable as "at": "drop". A marker is seconds or a clock
              ("45f", "0:12.5"); it has no .in or .out. If a marker and a layer share a name, "drop"
              reads the marker (check warns), "drop.out" the layer.
  grade       whole-frame look: {"contrast": 1.05, "saturate": 1.1, "vignette": 0.35, "grain": 0.05}
  motionBlur  {"samples": 6, "shutter": 0.5}   (slower: renders 6 sub-frames per frame). A layer
              blurs only inside its own time, so its first and last frames are whole; grain is
              added once, to the blended frame. Still layers below every moving one are drawn once
              a frame, not once a sub-frame, so list still layers first. Without it, check warns
              "strobe" where a layer moves more than 40 px a frame at 1080p for 5 frames running.
  camera      moves and turns the view in depth (see DEPTH AND CAMERA).
  fog         far layers fade into a colour (see DEPTH AND CAMERA).
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
  "at" defaults to 0. Inside a group, times count from the group's start. dur is above 0; give dur
  or until, not both (with both, until is ignored and check warns). Times are at most 1000000 s.
  check warns about a layer whose time lies wholly outside the scene or its group: it never shows.
  A video without dur lasts as long as its trimmed clip.
  A longer dur holds the clip's last frame (its sound stops at the trim end) and check warns
  (clip-hold); "loop": true repeats the clip instead.

KEYS EVERY LAYER HAS
  id, type, at, dur, until
  x, y          position of the anchor. A number (px), "50%" (of the canvas, or of the group's
                w and h), "50%-120", "10vw", "5vh", or an expression. "position": [x, y],
                [x, y, z] or {"x": ..., "y": ...} is read as x and y (and z); so are keyframes of
                points, [[0, [200, 540]], [2, [960, 300]]], and smooth keys or a tween of points.
  along         a path the layer rides: {"d": "M0 0 C300 -200 600 200 900 0", "p": [[0, 0], [2, 1]],
                "orient": true}. d is SVG path data, moved by x and y as a path layer's is, or the
                "#id" of a path, line, arrow or polygon in the same group, ridden where it is drawn
                (x and y then move the layer off it). p is the share of the path's length, a key
                like any other [from 0 at the layer's start to 1 at its end]; past 1 a closed path
                goes round again. orient turns the layer with the path, adding to rotate.
  anchor       center (default) | topleft | top | topright | left | right | bottomleft | bottom |
                bottomright. Video and shader default to topleft, even with x and y set; an image
                defaults to topleft without x and to center with it.
  opacity       0..1 (default 1). A group fades as one picture; layers in it with a blend still
                blend with what is under the group.
  scale, scaleX, scaleY   1 = normal size
  rotate        degrees, clockwise
  z, rotateX, rotateY, backface, fixed   depth: see DEPTH AND CAMERA
  blur          px
  blend         normal | multiply | screen | overlay | add | lighten | darken | softLight |
                hardLight | colorDodge | colorBurn | difference | exclusion | hue | saturation |
                color | luminosity (the last four take that of the layer and the rest from below)
  mask          the id of a layer beside it, in the same group, that it shows through: "mask": "lens",
                or {"layer": "lens", "mode": "alpha", "feather": 12}. mode alpha (default: shown where
                the mask is opaque), luma (where it is bright), alphaInvert or lumaInvert (where it is
                not). feather blurs the mask's edge, in px. The mask layer draws only as a mask, on its
                own clock, placed by its own x and y in the group, so it can move over a layer that
                holds still. Video inside a word is a video layer masked by a text layer.
  repeat        copies of the layer, up to 10,000: {"count": 60, "rotate": "=i * 6", "x": "=cos(i *
                0.1047) * 300", "y": "=sin(i * 0.1047) * 300", "fill": "=hsl(i * 6)", "opacity": 0.8}.
                A copy is the layer moved by its x, y, rotate and scale in the layer's own space, so
                copies turn and scale round the layer's x and y. fill and stroke recolour the layer's
                (on the layers that take them; a stroke a copy brings is strokeWidth [4] wide), and
                opacity fades each copy. A key is a number (fill, stroke: a colour), a list taken in
                turn (copy i takes item i, round again) or an expression of i (the copy, from 0), n
                (the count) and the layer's t, d, p and T; keys that read no time are worked out
                once. The layer's own keys, enter, exit, effects and mask act on all its copies as
                one picture. A shape's copies draw in one loop: 10,000 circles, each moved by two
                expressions, render at 1080p at about 40 fps on an RTX 5060 Ti. Captions do not
                repeat; a model's repeat is under 3D MODELS.
  shadow        {"x": 0, "y": 6, "blur": 16, "color": "#00000099"}   (those are the defaults)
  fx            {"brightness": 0, "contrast": 1, "saturate": 1, "hue": 0, "exposure": 0,
                 "temperature": 0, "tint": 0, "gray": 0, "invert": 0}. brightness adds to every
                channel (-1..1), so below 0 dark footage turns black; exposure is in stops and
                multiplies (-1 halves the light), which is the way to darken footage.
  glow          a blurred copy behind the layer: {"color": "#ff633f", "size": 24, "amount": 1}.
                No color: the layer's own colours. amount above 1 brightens it. "glow": 24 is the size.
  outline       a line round the layer's edge, outside it: {"width": 4, "color": "#000000"}
  innerShadow   a shadow inside the layer's edge: {"x": 0, "y": 4, "blur": 12, "color": "#00000080"}
  displace      waves that warp the layer: {"amount": 8, "scale": 60, "speed": 1, "seed": 0}. amount is
                how far in px, scale the size of the waves, speed how many waves a second they slide.
                The sizes of all four (glow size and amount, outline width, innerShadow x, y and
                blur, displace amount) animate like any key; each draws the layer offscreen, as blur does.
                On a model, glow and outline keep their 3D meanings (its material's glow, its edges):
                put the model in a group to glow or outline it. On the CPU, effects drawn offscreen are
                slow: blur and shadow blur a layer the size of the frame (0.44 s at 1080p), the rest one as
                large as the layer, and check warns slow-effects past 0.25 s a frame.
  enter, exit   a motion (see ENTER AND EXIT)
  hidden        true skips the layer (and its sound)
  sound         a file played when the layer starts, timed by the layer (see SOUND)
  allowOffscreen  true stops check warning when the layer leaves the canvas; on a group or
                  sequence it covers the layers inside
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
            d also takes keyframes of paths and morphs through them, any path into any other:
            "d": [[0, "M0 0 L100 0 L50 90 Z"], [1, "M50 0 A50 50 0 1 1 49.9 0 Z"]]. Each piece
            pairs with the one its size in the other key and starts where its points move least;
            a piece with no partner grows from, or shrinks into, the middle of the nearest one.
  arrow     x1 y1 x2 y2 bend [0] head [22] stroke strokeWidth
  strokes   on shapes, images and videos: dash [on, off, ...] px marks the stroke out as SVG's
            stroke-dasharray does; dashOffset slides it back along the stroke (animate it for
            marching dashes), and a dashed stroke still draws on, its dashes where they lie on
            the whole path. corner (px) on a path, line or polygon rounds every corner. radius on
            a rect, group, image or video takes four values, [top-left, top-right, bottom-right,
            bottom-left], as CSS's border-radius does; [24, 24, 0, 0] rounds the top corners.
  rough     on shapes: true, an amount in px [3], or {"amount": 3, "length": 12, "seed": 0, "boil": 0}.
            It wobbles the outline as if drawn by hand: points length px apart stray up to amount px,
            the same way for the same seed. boil draws it anew that many times a second (8 gives a
            hand-drawn flicker). Fill, stroke and draw follow the wobbled line.
  text      text (centred on x, y: the anchor is center), size [64], weight [600] (100..900), font ["Space Grotesk"; also "IBM Plex Mono",
            any installed family, or a .ttf path], italic, fill [#f2f0e8], emph (colour of
            text written **like this**, also inside a word: g**P**u colours the P),
            align [center] (left|center|right), maxWidth (wraps lines), lineHeight [1.18],
            letterSpacing [0], stroke, strokeWidth,
            box {"fill": "#000000b0", "pad": [12, 20], "radius": 10} (a background behind the
            text; pad is [vertical, horizontal]), case (upper|lower),
            split (chars|words|lines: animate each unit, see ENTER AND EXIT),
            caret (true shows a cursor while typing),
            unitX unitY unitRotate unitScale unitOpacity unitBlur unitFill (expressions per unit:
            i is its number from 0 and n how many there are; unitFill colours its plain letters,
            emph keeps its own). sweep(i, n, p, soft) is 1 for the units a sweep has passed, so
            "unitFill": "=mix('#f2f0e8', '#ff633f', sweep(i, n, p, 3))" runs a highlight across.
            textPath lays the letters along a path, each turned to it where its middle is: SVG
            path data in the layer's space (x and y move it), "#id" of a path, line, arrow or
            polygon in the same group (where that layer draws it), or {"d": ..., "offset": [0]
            (how far along, a share of the path's length; animate it to run the text along),
            "align": [start] (start|center|end: the part of the text that stands at offset)}.
            The first line stands on the path and the others below it. An open path leaves out
            the letters past its ends (check: text-off-path); a closed one takes them round
            again. anchor does nothing there, and unit keys move along and across the path.
            A character the font lacks (emoji, CJK, Hebrew, Arabic...) is drawn with an installed
            font that has it; check names any that no font has (missing-glyph). Arabic letters
            join and accents sit on their letter. Right-to-left words read right to left, and a
            line's words are put in order by the Unicode bidi rules applied word by word.
            split chars, letterSpacing and textPath draw each letter alone, so Arabic does not join
            there.
  counter   value (animate it, e.g. {"from": 0, "to": 1250, "dur": 2}), format ["{:,.0f}"; also
            "${:,.2f}M" or "{:.0f}%"], and every text key
  image     src (every src, for image, video, captions and audio, is relative to the scene
            file), w, h (give one and the other keeps the
            aspect), fit [cover] (cover|contain|stretch), zoom [1] (zooms inside the frame),
            focus [[0.5, 0.5]] (the point zoom moves toward), radius, stroke, strokeWidth,
            pixelated (true scales it with hard pixel edges, for pixel art).
            An .svg src draws as vectors, sharp at any size; its own size is its width and height,
            or its viewBox's. Text, <style> sheets, <symbol>, <image>, markers and an svg's own
            animation are left out (check warns svg-unsupported): turn text into paths when saving.
            Frame by frame: frames in place of src, a pattern with the frame number in it
            ("run/run_%03d.png" or "run/run_####.png"), a glob ("run/*.png"), a folder of images
            or a list of files; or sheet {"src": "walk.png", "cols": 8, "rows": 2, "count": 14},
            its cells played row by row from the top left. fps [12] is frames a second; loop
            [true] starts again after the last frame, false holds it, "pingpong" plays back to
            the first. A number missing from the files shows the frame before it (check warns
            missing-frames). The box is a frame's size, or a cell's. A file is decoded when it
            first shows and the last 64 stay decoded: a new 1080p PNG every frame took 34 ms a
            frame on an RTX 5060 Ti, one already decoded 4 ms.
  video     src, trim [start, end] (seconds in the source file), speed [1] (any number above 0),
            volume [1] (a number; 0 mutes), loop, w, h [canvas size], fit, zoom, focus, radius.
            Its sound is mixed in at the clip's speed and stops at the trim end unless loop is true.
            An enter or exit fades the sound over its dur too; fadeIn and fadeOut (seconds, 0 for
            none) set the sound's fades instead.
            Source seconds count from the file's first frame. A phone clip's rotation is applied.
            Alpha is kept from .mov (ProRes 4444) and .webm (VP8/VP9 with alpha, as --transparent
            writes it), so a transparent render can be footage in another scene.
  captions  cues [[start, end, "text"], ...] (seconds from the layer start) or src (.srt/.vtt, in
            UTF-8, UTF-16 or Windows-1252; tags are dropped, &amp; reads as &, and cues that
            overlap show together, the earlier one on top),
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
  particles an emitter: rate (particles a second; keyframes taper it) and burst (how many leave at
            once at the layer's start, or [[time, count], ...]), or a preset (confetti|sparks|snow|
            dust|bubbles) whose keys the layer's own replace: {"type": "particles", "preset":
            "confetti", "y": "70%"}. Each particle draws its own value from a number or [min, max] of
            life [2] (seconds), speed [100] (px a second), angle [[0, 360]] (degrees: 0 is right, 90
            down, so [-120, -60] sprays upward), size [8] (px across), spin [0] (degrees a second),
            drag [0] (2 loses most of the speed in a second) and flip [0] (turns over that many times
            a second, as confetti does). gravity [0] pulls in px a second each second, down, or
            [x, y]. turbulence (px, or {"amount": 30, "speed": 0.5}) sways each one. align turns
            each along its path. sizeOverLife and opacityOverLife are keyframes over 0..1 of a
            particle's life, [[0, 1], [1, 0]] shrinking or fading it out; fillOverLife takes colours.
            fill [#ffffff] is a colour or a list taken in turn. shape [circle] (circle|square|rect|
            triangle|star|ring|soft|streak), or text (a character such as "★") or src (an image),
            tinted by fill: white leaves them as they are. emitter [point] (point|line|rect|circle|
            ring) spreads births over w [canvas width] and h [canvas height], or r [100], round x
            and y [50%]. prewarm (true, or seconds) starts it as if it had been running; seed [from
            the id] gives another draw. Particles stay where they were born while x, y or along move
            the layer; rotate and scale turn and scale them all. A particle's path comes from its
            number and the seed, so any frame renders alone. 5,000 alive render at 1080p at 131 to
            159 fps on an RTX 5060 Ti and take about 0.1 s a frame on the CPU, where check warns
            particle-cost past 0.25 s.
  group     children [layers], w, h, clip (true cuts children off at w x h), radius, fill.
            Children are placed from the group's x, y and start at the group's "at".
            Its children's time: speed (2 runs them twice as fast), loop (seconds after which they
            start again, so a looping icon is one group with "loop": 1.2) and timeRemap (keyframes
            or an expression giving their time in seconds at the group's time t; a number freezes
            them there). Sounds keep the scene's time.
            combine (union|subtract|intersect|xor) merges its shapes and text into one shape, drawn
            with the group's fill, stroke, strokeWidth, draw, dash and rough instead of the
            children's own: "subtract" cuts every later child out of the first, so a word cut out of
            a badge is a group of a rect and a text. A filled child gives its fill's outline, a line
            or arrow its stroke's; images, videos, shaders, models and captions are left out. Text
            gives its letters' outlines, a shaped script's letter by letter. Children that hold
            still are merged once; moving ones each frame.
  sequence  clips [{"src": "a.mp4", "trim": [0, 4]}, ...] (an image clip needs dur) and transition
            {"type": "fade", "dur": 0.4} (fade|wipe|slide|zoom|blur|cut; dur defaults to 0.5, cut
            takes none, and no transition is a cut). Becomes a group of back-to-back layers named
            <id>-1, <id>-2, ... that ends with its last clip, so "<id>.out" is a cue. Each
            transition overlaps the clips it joins, so 3 s + 3 s with a 0.5 s fade lasts 5.5 s, and
            must be shorter than both. Their sound crossfades over it: the clip before fades out
            as the next fades in.
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
              "pixel" makes the group a stage: each child is drawn on its own and at each pixel the
              nearest one shows, so models and cards pass through each other with a clean edge. A
              model is as deep as its triangles; any other child lies flat in its own plane (a group
              child too). Each child costs a merge (check warns past 16), and a blend inside a stage
              has nothing under it. Stages do not nest.
  shadows     on a stage: true or {"opacity": 0.5, "soft": 3}. The stage's models cast shadows from
              the scene's key light onto every child, models and flat layers alike: a rect turned 90
              on rotateX is a floor that takes them. opacity is how much a shadow darkens, soft its
              edge in px. Flat layers do not cast, and a surface turned from the light takes none.
  camera      top level. Every key animates, and all default to 0 except fov:
              {"x": 0, "y": 0, "z": 0, "rotateX": 0, "rotateY": 0, "rotate": 0, "fov": 40}
              x and y pan (px). z moves the camera back; negative moves it in. rotateY orbits it
              right around the canvas centre, rotateX up over it (looking down), rotate rolls it.
              fov (5..150) sets how strong perspective looks; it does not change size at z 0.
              target: what the camera looks at and orbits in place of the canvas centre, framed as z 0
              is (in the middle, at its own size): a point [x, y, z] in px, or a layer id, which the
              camera follows (its keys and its parents', not its enter and exit motions). x and y still
              pan from it. Keyframes between points and ids pan across: [[0, "logo"], [2, "title"]].
              shake: px, or {"amount": 8, "speed": 1.5, "seed": 0}: a handheld wobble, smooth noise that
              turns the view by up to about amount px, speed times a second. amount animates.
              aperture and focus: depth of field. aperture is px of blur per 1000 px from the focus
              plane (12 is gentle, 40 strong); far layers blur up to aperture times the focus distance
              over 1000, near ones more, as a lens does, never past 48 px. focus is a z (px toward the
              camera from what it looks at; 0, the default, is the target or z 0) or a layer id, and
              keyframes between them pull focus: [[0, "cup"], [2, "title"]]. Both animate. A flat layer
              or a model blurs whole, by the depth of its middle; a stage blurs each pixel by its own
              depth, which on the CPU is slow (check warns: slow-depth-of-field).
              check warns (camera-inside-model) when the camera is inside a model, and
              (focus-on-nothing) when the focus plane is nearer or farther than every layer.
  Camera moves:
    orbit       "target": "hero", "rotateY": [[0, -40], ["end", 40]]
    crane       "y": [[0, 0], ["end", -300]] rises with the view; add "rotateX": [[0, 0], ["end", 30]] to
                swing up over the target, looking down
    push-in     "z": [[0, 0], ["end", -400]]
    dolly zoom  "fov": [[0, 20], ["end", 70]]: z 0 keeps its size while the far layers fall away, since
                fov already moves the camera to keep z 0 framed
    handheld    "shake": 6, on top of any of these
    rack focus  "aperture": 24, "focus": [[0.5, "cup"], [2, "title", "inOutSine"]]
  fog         top level: true, or {"color": "#0a0c10", "near": 1500, "far": 6000, "amount": 1}.
              Up to near px from the camera is clear; from there to far px everything mixes into
              color, up to amount. z 0 is (h / 2) / tan(fov / 2) px from the camera: 1484 at 1080p
              and fov 40. Defaults: color is the canvas bg (a gradient's first stop), near is z 0's
              distance and far 4 times it. A model mixes at each vertex, any other layer by the
              depth of its middle, and a stage at each pixel. All four keys animate. check warns
              (fog-hides) when a layer lies wholly past far.
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
  A ring through a panel, in a stage:
    {"id": "stage", "type": "group", "x": "50%", "y": "50%", "sort": "pixel", "children": [
     {"id": "ring", "type": "model", "shape": "torus", "r": 200, "tube": 40, "rotateY": 70,
      "rotateX": [[0, 0], ["end", 30]]},
     {"id": "panel", "type": "rect", "w": 560, "h": 320, "radius": 28, "fill": "#f2f0e8"}]}

3D MODELS
  A model layer is a lit solid in the space of DEPTH AND CAMERA, so z, rotateX, rotateY, rotate,
  scale and the camera apply to it. x and y place its centre.
  shape     cube (w [200], h and depth [= w], r [0] rounds its edges and corners) | sphere (r [100]) |
            cylinder (r [80], h [200]) | cone (r [90], h [200], tip up) | capsule (r [60], h [240] end
            to end: a cylinder with round ends) | torus (r [110], tube [r x 0.35], facing you) |
            plane (w, h; both sides show) | lathe | tube | text | path. Sizes are px and animate.
  lathe     shape "lathe" turns profile round the vertical axis: [[r, y], ...] points (r out from the
            axis, y down; closed when the last point is the first) or path data, its x the distance
            from the axis. Centred on y. A profile that ends off the axis (a cup with no lid) shows
            both sides; bends sharper than 40 degrees stay sharp. profile does not animate:
            "profile": [[0, -120], [70, -120], [40, 0], [90, 120], [0, 120]].
  tube      shape "tube" runs a round tube of radius r [12] along d (SVG path data, fitted to w and/or
            h as a path shape's), round at its ends: 3D lines and routes. draw grows it along its
            length (other shapes show bottom to top). Without d, "tube" is read as a cylinder.
  text      shape "text" pushes letters out in depth: text ("\n" breaks lines), size [120],
            weight [600], font, italic, letterSpacing, lineHeight, depth [size x 0.2], bevel [0]
            (px of 45-degree edge). Centred on x, y. Each new text, size, depth or bevel value
            rebuilds the solid, so grow it with scale. Lines are shaped as a text layer's are:
            Arabic letters join, marks sit on their letters, right-to-left runs read right to left.
            letterSpacing draws each character on its own, unshaped, and colour emoji have no
            outline, so they are left out (check warns: unshaped-3d-text).
  path      shape "path": d (SVG path data, filled nonzero), w and/or h (fits inside them, keeping
            its shape), depth [a fifth of its larger side], bevel [0], strokeWidth (in d's units:
            extrudes the stroke, with round ends, for line icons).
  detail    low | medium (default) | high: how many triangles. A medium sphere has about 2,200.
  fill      one colour [theme accent]   shine 0..1 [0.4]   metal 0..1 [0]   rim 0..1 [0.5] (edge
            light)   bands [0] (2 or more: flat toon steps)   glow 0..1 [0] (shows at least fill x
            glow however it is lit: neon, screens, lamps)   transparency 0..1 [0] (see-through: the far
            side and what is behind show, faces drawn far to near; a model with a texture draws solid,
            check warns). These animate. opacity fades the drawn model as a whole, as on any layer.
  texture   an image (png, jpg, webp, gif, bmp, svg) or video file drawn on the surface, "#groupId", or
            an object: {"src": "wood.jpg", "repeat": 2 or [across, down], "offset": [across, down]
            (in images, toward the right and down)}. A video plays on the model's clock, with "loop", "trim" and
            "speed" as on a video layer. "#groupId" draws that group on the surface at each frame's
            time, its box (its w and h, else the canvas) filling the image: a UI on a phone's screen.
            The group is drawn only there, its layers as in a scene without a camera. Once on each
            cube face and on a plane, upright from in front (on top and underneath, its top is the
            far edge); round a sphere, cylinder, cone, capsule or lathe with its middle facing you, and
            flat across a cylinder's or cone's ends; once round a torus and its tube; along a tube and
            round it; across the front and back of text and path shapes. The fill shows through clear pixels. A part takes the layer's texture unless it
            sets its own; false draws its fill. Triangles are cut where perspective would slide the
            image across them (check warns texture-swim where that is not enough, right in front of
            the camera). repeat and offset do not animate. A textured model takes about twice as
            long to draw.
  parts     [{"shape": "sphere", "r": 30, "x": 0, "y": -120, "z": 0, "rotateX": 0, "rotateY": 0,
            "rotate": 0, "scale": 1, "fill": "#b9f45d", "shine": 0.9}, ...] placed from the
            model's centre. Part keys take keyframes and expressions like layer keys, with i the
            part's number from 0: "rotate": "=t*90" spins one part. The parts merge into one mesh,
            merged again on the frames where a part moves or changes size. Parts with the same
            material keys share a material, its values read for the first of them. A part takes
            the model's material keys it does not set. A model may have shape (or src), parts, or both.
  repeat    {"count": 100, "x": "=(i % 10 - 4.5) * 50", "z": "=(floor(i / 10) - 4.5) * 50",
            "y": "=sin(t * 2 + i) * 20", "rotateX", "rotateY", "rotate", "scale", "fill": "=hsl(i * 3.6)"}
            draws count copies of the model (shape, parts or src, posed), up to 10,000, merged into
            one mesh that sorts as one model. Each key is a number (fill: a colour), a list taken in
            turn (copy i takes item i, round again) or an expression of i (the copy, from 0), n (the
            count) and the layer's t, d, p and T. A copy is placed from the model's centre, as a part
            is, and its fill recolours what the layer's fill would. Keys that read no time are worked
            out once. A model draws at most 64 colours: past that, each copy takes the nearest kept
            one (check warns repeat-colours). Copies cost what that many triangles cost: use "detail":
            "low" for many.
  src       a model file: .glb or .gltf (glTF 2.0), or .obj with its .mtl. Colours, shine and metal
            come from the file; a fill on the layer recolours all of it. It fits inside w and/or h
            keeping its shape, else its longest side is 300 px. Node transforms, skins (up to four
            joints a vertex; check warns skin-influences past that) and morph target weights apply,
            and animation plays the file's clips. A glTF base colour texture (with its TEXCOORD_n)
            or an .obj map_Kd is drawn as a texture; "texture": false draws its mean colour, and a
            texture on the layer replaces the file's (check warns no-uvs when the file has no
            texture coordinates).
            The light a material gives off (glTF emissiveFactor and emissiveTexture, .mtl Ke and
            map_Ke) is added to its colour however it is lit. A glTF material with alphaMode BLEND is
            see-through by its base colour's alpha, an .mtl one by d or Tr. Long triangles are cut to
            1/24 of its size for the depth sort. Compressed files are read: meshopt
            (EXT_meshopt_compression, gltfpack -c) and quantized ones as they are; Draco
            (KHR_draco_mesh_compression) with pip install "kinescript[draco]" (check names the
            extra when it is missing).
  animation a clip of the glTF file to play on the layer's clock: its name (kine probe lists them),
            its number from 0, true for the first, or {"name": "Walk", "speed": 1, "loop": true,
            "start": 0}. speed 2 plays twice as fast, -1 backwards; start is the seconds into the
            clip at the layer's start; loop false holds the last pose. Node translation, rotation
            and scale, morph target weights and skins move (LINEAR, STEP and CUBICSPLINE keys). The
            model keeps the size and centre of its rest pose, so a clip that walks it moves it. A
            name the file lacks is an error (unknown-animation) that lists the ones it has.
  ground    true or {"opacity": 0.5, "blur": px, "y": px, "from": "above"}: a soft shadow on a level
            floor under the model, as if lit from above, fainter as the model rises. The floor sits at
            the model's lowest point before it turns, or at y in the layer's parent space (canvas px
            at the top level): give y when the model moves up and down. "from": "light" casts it along
            the model's key light instead, so it moves as the light does (none when the light is
            below the floor). For shadows on other layers, use a stage's shadows.
  edges     lines along the model: true (creases and the outline, 2 px, theme ink), "all" (every
            triangle's edges: a wireframe with the lines behind it hidden), or {"width": 2, "color":
            "#0a0c10", "angle": 30}. A crease is drawn where the faces either side turn by more than
            angle degrees (0 draws every edge), the outline where a face toward the camera meets one
            turned away. Lines are width px wide whatever their depth, and hidden by whatever is in
            front of them. width, color and angle animate: "width": [[0, 0], [1, 2]] draws them in.
            Where parts of one model touch, a line along the seam can come out stepped, as the faces
            there do.
  explode   0..1: faces fly out along their normals.
  draw      0..1, or "enter": "build": triangles appear from the bottom up.
  light     top level, or on one model: studio (default) | soft | dramatic | flat (no shading),
            or {"preset": "studio", "key": 0.85, "fill": 0.25, "ambient": 0.22, "rim": 0.35,
            "from": "topleft", "color": "#fff4e6"}. from: topleft | top | topright | left | right |
            front | bottomleft | bottom | bottomright. azimuth (degrees round from the front, toward
            the right) and elevation (degrees up) replace those parts of from. Every amount, the
            angles and the colour animate: "azimuth": [[0, -60], ["end", 60]] sweeps the key across.
            follow: camera (default: the light moves with the camera, so orbiting does not change
            the shading) | world (it stays in the scene: orbit the camera for a turntable look).
            check warns (light-behind) when the key is behind a model as the camera sees it.
  lights    top level, or on one model: up to 3 more lights, [{"from": "right", "color":
            "#4a7dff", "intensity": 0.6}, {"type": "point", "x": 900, "y": 200, "z": 400,
            "range": 1000}]. A directional light takes from, azimuth, elevation and follow like the
            key; a point light sits at x, y, z in canvas px (z toward the viewer) and lights each
            model from where the model's centre is, at half strength range px away. These animate.
  env       top level, or on one model: what shiny and metal models reflect, fixed in the world:
            studio | sunset | night | chrome, or {"preset": "studio", "top": "#f6f3ec", "horizon":
            "#7d828c", "bottom": "#23252b", "amount": 1}. Metal reflects it tinted by its fill,
            anything else by its shine, most at grazing angles; shine 0 reflects a blur of it and a
            matte model none. Unset (the default), nothing is reflected. The colours and amount animate.
  Triangles are sorted far to near within one model, so where two parts of one model cut through
  each other the seam is stepped: let parts touch rather than cross. Separate layers draw in list
  order, except in a stage (a group with "sort": "pixel", see DEPTH AND CAMERA), where they meet
  at a clean edge wherever they cross: put things that cross in a stage as separate layers. A
  ground shadow in a stage lies on a floor layer in the same plane. check warns past 60,000
  triangles in a frame.
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
  A chrome logo reflecting a sunset as it turns:
    {"id": "chrome", "type": "model", "shape": "torus", "r": 200, "tube": 60, "x": "50%", "y": "50%",
     "detail": "high", "fill": "#d9dce1", "metal": 1, "shine": 0.95, "env": "sunset",
     "rotateX": [[0, 70], ["end", 20]], "rotateY": [[0, -30], ["end", 30]]}
  A planet whose key light sweeps round it, with a cool light from the right and a ring that glows:
    {"id": "planet", "type": "model", "x": "50%", "y": "50%", "detail": "high", "rotateY": [[0, -20], ["end", 20]],
     "light": {"preset": "dramatic", "azimuth": [[0, -70], ["end", 40]]},
     "lights": [{"from": "right", "color": "#4a7dff", "intensity": 0.5}],
     "parts": [{"shape": "sphere", "r": 150, "fill": "#e8e2d0", "shine": 0.8},
               {"shape": "torus", "r": 230, "tube": 12, "rotateX": 75, "fill": "#ff633f", "glow": 0.9}]}

ANIMATING A VALUE (any number or colour key)
  constant     "x": 960
  tween        "opacity": {"from": 0, "to": 1, "dur": 0.6, "delay": 0, "ease": "outCubic"}
               (those are the defaults, also when given as null; delay counts from the layer start)
  keyframes    "x": [[0, 200], [0.8, 960, "outBack"], ["end", 1700]]
               Each keyframe is [time, value] or [time, value, ease into it]. Time is seconds from
               the layer start ("500ms", "12f" also work); a negative time counts back from its
               end; "40%" is a share of its length; "end" is its end. Between keyframes the default
               ease is inOutCubic, which slows to a stop at every key.
               A flat list such as [0, 100] is NOT keyframes.
  smooth keys  "x": {"keys": [[0, 200], [1, 700], [2, 960], [3, 1700]], "smooth": true}
               Passes through each key without stopping: at rest only at the first and last key,
               at a turn, and around a "hold" key. Other eases on the keys are ignored. Numbers only.
  expression   "y": "=540 + sin(t * 3) * 40"   (a string starting with "=")
               t = seconds since the layer started   d = the layer's length   p = t / d (0..1)
               T = seconds since the scene started   i = unit index (split text)   W, H, fps
               n = how many units (in unit keys only)
               helpers: sin cos tan sqrt abs min max round floor ceil exp log atan2 hypot pi tau
                        clamp(v, lo, hi)  lerp(a, b, p)  noise(x, seed)  wiggle(freq, amp, t, seed)
                        ease("outCubic", p)  hsl(hue, sat, light)  mix(colour, colour, p)
                        sweep(i, n, p, soft)
               hsl takes a hue in degrees and saturation and lightness from 0 to 1; hsl and mix give
               a colour, so "fill": "=mix('#ff633f', '#7b5cff', p)" works. sweep is 1 for the units
               a sweep across n of them has passed at p, 0 for those it has not reached, and eased
               between over soft [1] units at its front; it starts before unit 0 and ends past n - 1.
  Colours animate too: "fill": {"from": "#ff633f", "to": "#b9f45d", "dur": 1}.
  Gradient fill: "fill": {"linear": [[0, 0], [1, 1]], "stops": ["#ff633f", "#7b5cff"]}
                 or {"radial": [0.5, 0.5, 0.6], "stops": [...]}; points are 0..1 of the layer box.
                 {"angle": 90, "stops": [...]} runs corner to corner as CSS's does (0 up, 90 right);
                 {"sweep": [0.5, 0.5], "stops": [...]} goes round its middle from the top, clockwise,
                 turned by angle. Text and strokes take gradients too. A gradient animates into one
                 of the same kind with as many stops, by keyframes or a tween:
                 "fill": {"from": {"angle": 0, "stops": [...]}, "to": {"angle": 180, "stops": [...]}}

ENTER AND EXIT
  "enter": "rise"   or   {"name": "rise", "dur": 0.5, "ease": "outCubic", "stagger": 0.03}
  Defaults: dur 0.5, ease outCubic (pop and spin: outBack; type: linear; draw: inOutCubic;
  snap: outQuart). An exit plays during the LAST dur seconds of the layer and ends with it.
  Combine with +: "rise+blur". Options: dist (px moved by rise, drop, left, right; default 48),
  from (start scale of pop [0.4] and zoom [1.25]), amount (blur px [24], track px [40]).
  With "split" on text, each unit enters in turn: unit k starts k x stagger seconds after the
  layer starts, so the whole reveal takes dur + (units - 1) x stagger. On a group, an enter
  with a stagger, such as {"name": "rise", "stagger": 0.08}, enters its children in turn the
  same way, and the group itself does not move.
  order sets who goes first: start (default, in the order listed), end, center (from the
  middle out), edges (from both ends in) or random (seed, a whole number, picks the shuffle).
  motions: {motions}
  rise moves up into place, drop moves down, left moves left, right moves right. type is a
  typewriter. draw draws a stroke or path on, and builds a model (build is the same). wipe reveals left to right. snap is a hard cut in
  with a small settle. flip, flip-up, tumble and push move in depth (see DEPTH AND CAMERA).
  none is no motion.

EASES  {eases}, spring, spring(stiffness, damping), bezier(x1, y1, x2, y2), steps(n)
  in* starts slow, out* ends slow (the most natural for entrances), inOut* does both.
  spring(stiffness, damping, mass) takes numbers above 0 [170, 26, 1]; bezier takes four numbers
  with x1 and x2 from 0 to 1, as in CSS; steps takes a whole number, 1 or more.

COMPONENTS  {"use": "<name>", "id": "...", "at": ..., "dur": ..., ...parameters}
  Built in (parameters):
{components}
  x and y place the component as noted above. Components also accept enter, exit, opacity,
  scale, rotate, hidden, sound.
  Your own: "components": {"badge": {"params": {"label": "NEW"}, "layers": [
              {"type": "text", "text": "{{label}}", "size": 40}]}}
  then a layer {"use": "badge", "label": "BETA", "x": 200, "y": 100}. Its layers' ids, nested ones too,
  get the instance id in front ("b1-title" for "title" in {"use": "badge", "id": "b1"}).
  params holds the defaults; a parameter given as null takes its default. A {{placeholder}}
  ({{ label }} with spaces works too) with no value and no default is an error that names it.

SOUND
  Top-level "audio" lists the sounds. Each one:
    {"id": "vo1", "role": "voice", "src": "vo1.wav", "at": 0.5}
  src (relative to the scene file), id, role (voice|music|sfx|ambience), at, dur, until,
  trim [start, end] (seconds in the file), gain (dB: 0 = unchanged, -6 = about half as loud) or
  volume (1 = unchanged), fadeIn, fadeOut (seconds), loop (true repeats the trimmed file until the
  sound ends), pan (-1 left .. 1 right), text, speaker and rate (a voice spoken from text, see
  below). A sound lasts as long as its trimmed file; a dur past that is silence unless loop is
  true. fadeOut ends where it ends, which is the scene end at the latest. A sound with a
  negative at plays from the part of it that is still going at 0 s. kine check says when a src
  has no sound in it (no-audio), and warns about a sound nobody will hear: volume 0, a gain
  that leaves it silent, or one that starts after the scene ends or ends before it starts
  (audio-unheard).
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
  rate (0.5 to 2, default 1) how fast it speaks. With Kokoro, a speaker of "af_heart,af_bella" blends
  the two voices, [word](/phonemes/) in text says that word with the IPA phonemes you give (for a name it
  says wrong), and [word](-1) or [word](+1) lowers or raises its stress; captions and word ids keep the
  word. kine check reports a sound with text that is not spoken yet (voice-missing) and one whose text,
  speaker, rate, backend or options changed since it was spoken (voice-stale).
  A tag in the text says how to speak what follows, for a backend that acts on tags: [whispers], [laughs],
  [sighs]. Tags are not words: marks and word ids skip them, a backend without tags says the line without
  them, and kine voice lists what a backend could not do under "ignored". A sound picks its backend with
  backend (kokoro, chatterbox, elevenlabs, each a plugin) and hands it settings with options:
    {"id": "vo2", "role": "voice", "text": "[whispers] Nobody knows. [laughs] Yet.", "backend": "elevenlabs",
     "speaker": "George", "options": {"stability": 0.3}}
  chatterbox takes [laugh], [chuckle], [sigh], [gasp], [cough], [clear throat], [whispering], [angry],
  [happy], [sarcastic], [surprised], [crying] and a few more ([laughs] reads as [laugh]), but acts them
  only some of the time ([laugh] made a sound in 6 of 9 test takes, [whispering] whispered in 0 of 7),
  so use elevenlabs when a line needs its tags heard; its speaker
  is a clip of 5 s or more whose voice it copies, and "options": {"model": "standard", "exaggeration": 0.8}
  trades the tags for stronger emotion. elevenlabs needs an ELEVENLABS_API_KEY and uses the account's
  quota; its speaker is a voice name or id, its options model (eleven_v4), stability, style and
  similarity_boost. kokoro has no tags.
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
                                  overlapping text, contrast under 3:1, text a later layer
                                  covers (text-hidden), text shown for less than about 0.4 s +
                                  0.3 s per word, or one or two words for under 0.4 s
                                  (read-time), a character no installed font has
                                  (missing-glyph), Arabic drawn letter by letter
                                  (unjoined-letters), colour emoji or spaced-out letters
                                  in 3D text (unshaped-3d-text), an svg with parts that are
                                  not drawn, such as text (svg-unsupported), or a video, image
                                  or shader more than half off the canvas for 1 s or more
                                  (media-offscreen).
                                  With sound (it is
                                  mixed and measured, see SOUND): a voice cut off (audio-cut), two
                                  voices at once (voice-overlap), music under 10 LU below a voice
                                  (voice-masked), an sfx whose first sound is late (sfx-late), a
                                  silent sound, a sound nobody hears (audio-unheard), and the
                                  mix's loudness and peaks. With footage: a dur past the end of
                                  its clip, so the last frame holds (clip-hold), a file whose
                                  picture stops before its length, cut short or damaged
                                  (media-truncated), a still image as a video (media-still), and
                                  a .webm whose alpha this ffmpeg cannot read (media-alpha-lost).
                                  info: a name it read for you, e.g. "'color' read as 'fill'",
                                  or an odd canvas size (odd-size).
  kine look scene.json --boxes    one PNG of 12 evenly spaced frames; --boxes outlines layers and
                                  labels the top-level ones; -n 16 frames, --at 1 2.5 4 exact
                                  times, read as still reads them (use it to see a transition;
                                  a time outside the scene draws its first or last frame and
                                  the JSON's "notes" say so), --width 480 (16 to 4096), --cols 4,
                                  -o (.png or .jpg). --wire draws every model's triangles over the
                                  frames; --depth dims them and shows each stage's depth image,
                                  nearer brighter. --trails draws each moving layer's path, a dot
                                  a frame, so their spacing shows its speed, eases and stops;
                                  --onion 3 draws the 3 frames before each one over it, fainter.
                                  --audio adds the sound under the frames on the same time axis:
                                  a lane per role and one for the mix, the duck, and issues in red;
                                  its JSON gives each sound's loudness and gain and the mix's.
  kine still scene.json --at 2.5  one frame as PNG (-o to name it, .png or .jpg); --at -1 is 1 s
                                  before the end; --at end is the last frame; a time past the end
                                  draws the last frame, with a note. --scale 0.5 halves the size.
  kine probe scene.json --at 2    every layer's and sound's start and end, and at 2 s each visible
                                  layer's screen box and anchor (pos)
  kine probe scene.json --curve title.x
                                  title's x at every frame, through eases, expressions and group
                                  clocks, and its speed a second (signed; a distance for position,
                                  x and y as a point, or a colour), where it holds still and its
                                  fastest frame. An enter, exit or along moves a layer as well.
  kine probe clip.mp4             a media file's kind (video, audio or image), duration, w and h as
                                  shown (after rotation), fps, audio, alpha and rotation; a note
                                  when it lists a video stream with no frames in it. A model file
                                  (.glb, .gltf, .obj): its triangles, vertices, materials, size,
                                  animations with their durations, skins, morph targets, textures
                                  and texture coordinate sets. An .svg: the w and h it is drawn
                                  at and the kinds of element left out of it (undrawn).
  kine render scene.json -o out.mp4 -q draft|standard|high
                                  standard is the default, draft the fastest, high the best.
                                  Also .mov or .webm (with --transparent for no background),
                                  .gif (longest side 720 px, 15 fps), or a folder/ for PNG frames
                                  (frames/f_%04d.png names them). .wav, .m4a, .mp3, .flac or
                                  .opus: the sound alone. Any other extension is an error, and so
                                  is -o naming a file the scene reads, or a file or folder this
                                  account can't write (found before the first frame is drawn).
                                  render, still and look make the folders in -o that do not
                                  exist yet.
                                  --scale 0.5 halves the size (each side must stay 2 px or more).
                                  --from 2 --to 4 renders a part: every frame on screen from 2 s
                                  until 4 s; a --to past the end stops at the end, with a note.
                                  -j 4 runs 4 render processes, at most one per frame and one
                                  per CPU (more is cut, with a note); without -j, or -j 0, kine
                                  picks. A GIF's "frames" is the number of frames in the GIF.
                                  A frame is the same whichever frames come before it, except
                                  that on the GPU text drawn in depth (a camera, z, rotateX or
                                  rotateY) can differ by a level or two at a few edge pixels
                                  (the GPU's glyph cache); KINE_BACKEND=cpu renders the same
                                  bytes every time.
                                  Its JSON lists warnings for footage cut short, with no picture or
                                  with alpha this ffmpeg cannot read, as check gives them.
  kine voice scene.json           speaks each sound's text into its file and times its words; a sound
                                  is spoken again only when its text, speaker, rate, backend or
                                  options changed (--force: all of them, --only vo1: one). Needs a
                                  voice backend plugin (with none installed, kine voice prints the
                                  pip command); --backend picks one for sounds without their own.
                                  KINE_VOICE_CMD (or KINE_VOICE_CMD_CHATTERBOX for one backend) runs
                                  a backend in another Python: it reads the requests as a JSON list
                                  on stdin and prints a JSON list of results, one per request.
                                  Rendering never speaks.
  kine watch scene.json           re-makes the contact sheet on every save
  kine doctor                     checks ffmpeg and ffprobe, the GPU and fonts; exits 1 when
                                  ffmpeg or ffprobe is missing, and says how to install it
  kine mcp                        the same tools for MCP clients
  Commands answer in JSON on stdout. "notes" in an answer say where an argument was changed:
  a time past the end, or more workers than a render can use. Exit status:
    0    done
    1    kine check found errors (warnings alone exit 0), or kine doctor found ffmpeg or
         ffprobe missing
    2    a bad argument, scene or file, or a tool that failed; "error" says what was wrong
    3    a bug in Kinescript; "error" says what and where. Please report it with the command
         and the scene (KINE_DEBUG=1 also prints the traceback)
    130  stopped with Ctrl+C
  KINE_BACKEND=cpu draws on the CPU; gpu insists on the GPU (an error when there is none);
  auto, the default, uses the GPU when there is one. Any other value is an error.
  ffmpeg and ffprobe must be on the PATH: kine names the one it cannot find.

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
