<p align="center"><img src="https://raw.githubusercontent.com/drknowhow/kinescript/main/brand/logo-dark.png" width="460" alt="kinescript"></p>

<p align="center"><b>Video you can diff.</b><br>Motion graphics, video edits and voice-over written as JSON, checked by agents, rendered on the GPU.</p>

<p align="center"><a href="https://drknowhow.github.io/kinescript/">Documentation</a> · <a href="https://github.com/drknowhow/kinescript">Source</a> · <a href="https://github.com/drknowhow/kinescript/blob/main/CHANGELOG.md">Changelog</a> · <a href="https://github.com/drknowhow/kinescript/blob/main/docs/promo.mp4">Launch video</a></p>

<p align="center"><img src="https://raw.githubusercontent.com/drknowhow/kinescript/main/docs/hero.gif" width="720" alt="The Kinescript launch video, rendered by Kinescript"></p>

Kinescript turns a JSON scene into video. Every layer, keyframe, cut and sound is a line of data, so a scene can
be generated, reviewed in a pull request, and changed by a script or an agent. A voice-over can start as a line of
text. The renderer is a pure function of the frame number: any frame renders on its own, which makes previews
instant and lets a render split across processes.

```bash
pip install kinescript
kine doctor
```

It needs Python 3.10 or newer and ffmpeg on your `PATH` ([details](https://github.com/drknowhow/kinescript#install)).

The [launch video](https://github.com/drknowhow/kinescript/blob/main/docs/promo.mp4) is a Kinescript scene,
[`marketing/promo.json`](https://github.com/drknowhow/kinescript/blob/main/marketing/promo.json). It is 35
seconds of 1080p and renders, with audio, in about 3.3 seconds on the development machine. The
[project site](https://drknowhow.github.io/kinescript/) shows more demos, each next to the scene file it was
rendered from.

## What it does

- **Cue timing.** Start and end times can name other layers: `"at": "title.out+0.2"`. Lengthen one layer and
  everything cued after it moves. Cycles and missing references are reported with the ids involved.
- **Animate anything.** Numbers and colours take constants, keyframes (`[[0, 0], [0.6, 1, "outBack"]]`, with
  negative times counted from the layer's end), a tween object, or an expression such as `"=540 + sin(t*3)*40"`.
  Colours interpolate in OKLab.
- **Text that moves per letter.** `"split": "chars"` (or words, or lines) with a staggered enter motion, a
  typewriter reveal with a caret, `**emphasis**` markup, and per-unit expressions (`"unitY": "=sin(T*4 + i)*12"`
  for a wave).
- **Footage.** Video layers with trim, speed, loop, cover/contain fit, punch-in `zoom` around a `focus` point, and
  rounded picture-in-picture. A `sequence` turns a cut list into clips with fade, wipe, slide, zoom or blur
  transitions. Clip audio and music beds are mixed automatically.
- **Sound on the same timeline.** Voice-over, music and effects are cues like layers, a voice's words are cues
  too, a voice can be spoken from text, and sounds with a role are levelled and ducked for you
  ([Sound](https://github.com/drknowhow/kinescript#sound)).
- **Captions** from `.srt`/`.vtt`, inline cues or a voice's word times, with karaoke and pop modes.
- **Shaders.** SkSL layers with `mesh`, `aurora`, `grid` and `score` presets, or your own source.
- **Finishing.** Per-layer blur, shadow, blend modes and colour adjustments; a scene grade with vignette and film
  grain; optional motion blur.
- **Components.** `lowerThird`, `titleCard`, `callout`, `focus` (dim everything but a region), `progress`,
  `kinetic`, plus your own with `{{param}}` placeholders.
- **Outputs.** H.264 `.mp4`, ProRes 4444 `.mov` and VP9 `.webm` with alpha for compositing, `.gif`, PNG sequences.

## 3D

Depth and models are drawn by the same Skia renderer as everything else, with no extra dependency. Scenes that use
no depth draw exactly as before.

- **Camera and depth on any layer.** Every layer takes `z`, `rotateX`, `rotateY` and `backface`. A top-level
  `camera` pans, dollies, orbits and rolls with perspective; `fixed` layers ignore its moves, and a group with
  `"sort": "depth"` draws its children far to near. The `flip`, `flip-up`, `tumble` and `push` motions work on
  whole layers and per letter with `split`. A two-sided card is a group with `"backface": "hidden"` on each side.
- **Stages.** A group with `"sort": "pixel"` draws each child with an image of its depth and shows the nearest at
  each pixel, so models and cards pass through each other with a clean edge, and a model's `ground` shadow lies
  on a floor layer.
- **Model layers.** `"type": "model"` draws a lit `cube`, `sphere`, `cylinder`, `cone`, `torus` or `plane`, or
  several shapes merged as `parts` so their triangles are depth-sorted together. Materials take `fill`, `shine`,
  `metal`, `rim`, `bands` (toon steps) and `glow`. `light`, for the scene or one model, takes `studio`, `soft`,
  `dramatic` or `flat`, or your own amounts, and every amount, its `azimuth`, `elevation` and colour animate. It
  follows the camera, or with `"follow": "world"` stays in the scene as the camera orbits. `lights` adds up to three
  more, directional or point. `explode` sends faces outward and the `build` motion assembles a model.
- **Extruded text and SVG paths.** `"shape": "text"` extrudes letters in any font and `"shape": "path"` extrudes
  SVG path data, filled, or stroked with `strokeWidth` for line icons. Both take a `depth` and a 45-degree `bevel`.
- **Model files.** `src` reads `.glb` and `.gltf` (glTF 2.0) and `.obj` with its `.mtl`: node transforms, base
  colours, and metallic and roughness as `metal` and `shine`. `ground` puts a soft shadow under any model.

A 3D title that builds in and turns to face the camera:

```json
{"id": "title3d", "type": "model", "shape": "text", "text": "LAUNCH",
 "x": "50%", "y": "50%", "size": 240, "weight": 800, "depth": 70, "bevel": 5,
 "rotateY": [[0, -40], [2.4, 0, "outCubic"]], "rotateX": [[0, 18], [2.4, 0, "outCubic"]],
 "shine": 0.6, "enter": {"name": "build", "dur": 1.2}}
```

## Sound

Sounds are a top-level `audio` list, timed like layers and mixed by ffmpeg while the frames render.

- **Cues both ways.** `"at": "vo1.out+0.3"` starts a layer 0.3 s after the voice ends, and a sound can start at
  `"title.in"`. A layer's `"sound": "pop.wav"` plays when it appears and moves when it moves. A voice stretches an
  `auto` duration so it is never cut off.
- **Words are cues.** A words file next to a voice (`vo1.words.json`, or Whisper's JSON as it is) times each
  word: `"at": "vo1:render"` starts a layer as the voice says "render", and `{"type": "captions", "voice": "vo1"}`
  captions it, lighting each word as it is said.
- **Voice-over from text.** Write what a voice says in `text` and `kine voice` speaks it through a plugin, writing
  the file and its word times: [Kokoro](https://github.com/drknowhow/kinescript/tree/main/plugins/kinescript-kokoro)
  (local, fast),
  [Chatterbox](https://github.com/drknowhow/kinescript/tree/main/plugins/kinescript-chatterbox) (local,
  cloned voices) or [ElevenLabs](https://github.com/drknowhow/kinescript/tree/main/plugins/kinescript-elevenlabs)
  (cloud, needs an API key), picked per sound with `backend`. `[[go]]` in the text names the next
  word, so a cue to `vo1:go` survives a rewrite of the line. Only a line whose text, speaker, rate, backend or
  options changed is spoken again, and rendering never calls a speech engine.
- **Delivery tags.** `[whispers]`, `[laughs]` or `[sighs]` in the text tell ElevenLabs how to say what follows.
  Chatterbox takes them too but acts them only some of the time: in test takes it made a sound where `[laughs]`
  goes in six of nine and whispered in none of seven. Tags are not words, so cues, marks and captions skip them,
  and Kokoro says the line without them.
- **A mix without listening.** Give a sound a `role` and it is set to that role's level: voice -18 LUFS, music
  -16, ambience -30, effects peaking at -6 dBFS. Music and ambience dip 14 dB while a voice speaks, an effect's
  leading silence is skipped so it lands on its cue, and the whole mix ends at -16 LUFS with peaks under
  -1.5 dBTP. A top-level `mix` changes the targets. Sounds without a role are mixed as given.
- **Checked and drawn.** `kine check` mixes and measures the sound and reports a voice cut off or masked by the
  music, an effect that lands late, silence, and a mix off its targets. `kine look --audio` draws the sound under
  the frames on the same time axis.
- **Everywhere.** Gain takes keyframes and expressions like any property; sounds loop and pan; the mix goes into
  MP4, MOV, WebM and `--from/--to` renders, and `-o mix.wav` renders the sound alone.

A narrated beat with a click on a marked word, over music that ducks under the voice:

```json
"audio": [
  {"id": "vo", "role": "voice", "at": 0.6, "text": "Each word it says is a [[hit]]cue."},
  {"id": "bed", "role": "music", "src": "bed.m4a", "loop": true, "until": "end", "fadeOut": 1.5}
],
"layers": [
  {"id": "cue", "type": "text", "text": "Cue", "size": 200, "at": "vo:hit", "sound": "click.wav", "enter": "pop"},
  {"id": "cc", "type": "captions", "voice": "vo", "mode": "karaoke"}
]
```

`kine voice scene.json` speaks it once. The
[voice demo](https://drknowhow.github.io/kinescript/#sound) on the project site plays the result with sound, next
to its scene file. Install the plugin for the voice you want next to Kinescript:

```bash
pip install kinescript-kokoro       # local and fast; Python 3.10 to 3.12
pip install kinescript-chatterbox   # local, copies a voice from a clip, tags hit or miss; installs PyTorch
pip install kinescript-elevenlabs   # ElevenLabs' API, acts on tags; needs ELEVENLABS_API_KEY
```

A plugin can also live in another Python, for Kokoro on Python 3.13 or newer or to keep PyTorch out of
Kinescript's environment: point `KINE_VOICE_CMD`, or `KINE_VOICE_CMD_<NAME>` for one backend, at it. Each plugin's
README has the details:
[Kokoro](https://github.com/drknowhow/kinescript/tree/main/plugins/kinescript-kokoro),
[Chatterbox](https://github.com/drknowhow/kinescript/tree/main/plugins/kinescript-chatterbox),
[ElevenLabs](https://github.com/drknowhow/kinescript/tree/main/plugins/kinescript-elevenlabs).

## Built for agents

An agent cannot watch a video, so Kinescript gives it things it can read:

| Command | What the agent gets |
|---|---|
| `kine docs` | the full scene reference, short enough to keep in context |
| `kine check scene.json` | every problem at once as JSON, each with its layer, time range and a fix |
| `kine look scene.json --boxes` | one PNG of many labelled frames, with every layer outlined and the top-level ones named |
| `kine look scene.json --audio` | the same frames with the sound under them: a lane per role and the mix, the duck and issues, plus each sound's measured loudness |
| `kine probe scene.json --at 3.2` | resolved in/out of every layer and each visible layer's screen box |
| `kine still` / `kine render` | a frame, a range (`--from/--to`), a draft (`-q draft --scale 0.5`) or the final file |
| `kine watch scene.json` | the contact sheet, made again every time the file is saved |
| `kine voice scene.json` | each sound with text spoken into its file, with its word times, skipping what has not changed |

`kine check` reports unknown keys and bad values, missing media and out-of-range trims, text off the canvas or
in the safe margin, overlapping text, contrast under 3:1 against what is actually behind the text, tiny text and
too little reading time. When the scene has sound it mixes and measures it, without drawing a frame, while the
frames are checked, and reports a voice cut off by the scene end, two voices at once, music less than 10 LU under a
voice, an effect whose first sound comes late, a silent sound, and a mix off its loudness target or at its peak
ceiling. An unknown key names the intended key first. Common names from other tools (`color`,
`fontSize`, `duration`, `position`) are read and reported as info notes. It exits with status 1 while there are
errors.

The same tools are available over MCP (`kine mcp`) as `kine_docs`, `kine_check`, `kine_look`, `kine_still`,
`kine_probe`, `kine_render` and `kine_voice`. They take a scene as a file path or as the JSON object itself, and `kine_look`
and `kine_still` return images inline.

```json
{
  "mcpServers": {
    "kinescript": {"command": "kine", "args": ["mcp"]}
  }
}
```

## Install

Requires Python 3.10+ and ffmpeg on `PATH` (for example `winget install Gyan.FFmpeg`, `brew install ffmpeg` or
`sudo apt install ffmpeg`). On Linux, `skia-python` also loads the system's EGL, GL and fontconfig libraries, even
with `KINE_BACKEND=cpu`. A bare container or server needs `sudo apt install libegl1 libgl1 libfontconfig1`, or
importing Kinescript fails with `libEGL.so.1: cannot open shared object file`.

```bash
pip install kinescript
kine doctor        # ffmpeg path, GPU availability, default font
```

From a checkout: `pip install -e .`

Dependencies are `skia-python`, `numpy` and `glfw`. Space Grotesk and IBM Plex Mono (both SIL Open Font License)
ship with the package, so default text looks the same on every machine.

## Quick start

```bash
kine new scene.json
kine check scene.json
kine look scene.json --boxes
kine render scene.json -o scene.mp4
```

```json
{
  "canvas": {"w": 1920, "h": 1080, "fps": 30, "bg": "#0a0c10"},
  "grade": {"vignette": 0.35, "grain": 0.04},
  "layers": [
    {"id": "dot", "type": "circle", "x": "50%", "y": "50%", "r": 26, "fill": "$accent",
     "dur": 1.1, "enter": {"name": "pop", "dur": 0.4}, "exit": "fade"},
    {"id": "title", "type": "text", "text": "Motion, **written.**", "x": "50%",
     "y": "46%", "size": 140, "weight": 700, "split": "chars", "at": "dot.out-0.1",
     "dur": 3.2, "enter": {"name": "snap", "stagger": 0.03}, "exit": "fade"},
    {"id": "sub", "type": "text", "text": "scenes as data", "x": "50%", "y": "62%",
     "size": 40, "fill": "$muted", "at": "title.in+0.6", "until": "title.out",
     "enter": "rise"}
  ]
}
```

More scenes in [`examples/`](https://github.com/drknowhow/kinescript/tree/main/examples):

- [`hello.json`](https://github.com/drknowhow/kinescript/blob/main/examples/hello.json): a title animation, 4.2 s.
- [`edit.json`](https://github.com/drknowhow/kinescript/blob/main/examples/edit.json): a cut list with a
  punch-in, a transition, a lower third, a callout, karaoke captions and a music bed, 10.4 s.
- [`3d.json`](https://github.com/drknowhow/kinescript/blob/main/examples/3d.json): a camera move past layers
  at several depths and a card that turns over, 6 s.
- [`model.json`](https://github.com/drknowhow/kinescript/blob/main/examples/model.json): lit primitives and a
  ring around a ball that builds in and flies apart, 6 s.
- [`title3d.json`](https://github.com/drknowhow/kinescript/blob/main/examples/title3d.json): the brand mark as an
  extruded SVG path and the wordmark as extruded text, 5 s.
- [`import.json`](https://github.com/drknowhow/kinescript/blob/main/examples/import.json): a glTF rocket on a
  ground shadow, 6 s.
- [`stage.json`](https://github.com/drknowhow/kinescript/blob/main/examples/stage.json): a ring through a panel
  and a ball that comes through it, on a floor, in a stage, 6 s.
- [`lights.json`](https://github.com/drknowhow/kinescript/blob/main/examples/lights.json): a ringed planet under a
  key light that stays in the scene as the camera orbits, a blue light from the side, a warm point light below and
  a glowing ring, 6 s.
- [`voiceover.json`](https://github.com/drknowhow/kinescript/blob/main/examples/voiceover.json): titles cued from
  the words of a voice-over, karaoke captions of it, and a music bed that dips under it, 5.4 s.
- [`narrated.json`](https://github.com/drknowhow/kinescript/blob/main/examples/narrated.json): the same idea
  with a voice written as text, a marked word and a click on it. Run `kine voice examples/narrated.json` first.

`edit.json`, `import.json`, `voiceover.json` and `narrated.json` read sample media: run
`python examples/make_media.py` first to generate it.

## Performance

Measured on the development machine (Intel Core Ultra 7 265, 20 threads, NVIDIA RTX 5060 Ti, Windows 11),
2026-10-31:

| Workload | Result |
|---|---|
| Drawing one 1080p frame of the launch video on the GPU, converted to I420 and read back as a render does | median 2.0 ms, p95 3.3 ms (all 1038 frames) |
| Launch video, 34.6 s at 1080p30, `-q high`, with audio | 3.6 s wall, about 290 frames/s (median of 5 runs) |
| `examples/edit.json`, 10.4 s with two decoded video clips | 1.95 s wall, 160 frames/s (median of 5 runs) |

[`python bench/stages.py`](https://github.com/drknowhow/kinescript/blob/main/bench/stages.py) splits the time
per frame into drawing, conversion and readback, and writing to ffmpeg for several workloads, and times whole
renders. [`python bench/stress.py`](https://github.com/drknowhow/kinescript/blob/main/bench/stress.py) generates
23 scenes that each push one thing far past the examples (2000 layers, 60 cuts, 4K, 120 s, 16-sample motion blur
with alpha, 600 caption cues and more), then checks, renders and probes each one in its own process and reports
time, peak memory and whether the output holds every frame.

Rendering uses Skia on the GPU through a hidden OpenGL context, one per worker process. Frames are split into
chunks, encoded in parallel and joined without re-encoding, while another ffmpeg process mixes the audio.
Within one process, such as the MCP server, the workers stay warm for two minutes between renders.

## Limits

- Without a usable GPU, Kinescript falls back to Skia's CPU rasteriser. With the Windows `skia-python` wheel this is
  very slow (about 0.65 s per 1080p frame for the launch video); use `--scale 0.5` for previews on such machines.
  `KINE_BACKEND=cpu` forces this path.
- 3D is drawn by Skia with a depth sort, not a GL pipeline: models have no image textures, skeletal animation or
  shadows cast on each other. Separate layers hide each other per pixel only inside a stage, and parts of one
  model that cut through each other meet at a stepped seam. Draco- and meshopt-compressed model files are not read.
- Audio is mixing and levelling (trim, gain over time, fades, speed, loop, pan, role levels, ducking, a loudness
  target), with no effects such as EQ, compression of single sounds or reverb.
- Kinescript does not speak by itself: `kine voice` needs a plugin or a command. The Kokoro plugin times words
  for English voices only and needs Python 3.10 to 3.12; the Chatterbox plugin speaks and times English only;
  the ElevenLabs plugin needs an API key and a network connection.
- Video is decoded and resampled by ffmpeg to the scene frame rate. Jumping backwards or far ahead reopens the
  decoder, which is slower than sequential playback.
- Text layout is single-script left-to-right with greedy wrapping. No shaping for complex scripts or bidi.
- Expressions are limited to arithmetic and the listed helpers; attribute access and other names are rejected.

## Project layout

```
kinescript/   engine: timing, props, scene, components, render (Skia), space, render3d,
              mesh and models (3D), output (encoding and the audio mix),
              verify (check, look, probe), schema, docs, mcp, cli
examples/     example scenes and the generator for sample media
plugins/      voices for kine voice, each installed on its own: kinescript-kokoro, kinescript-chatterbox,
              kinescript-elevenlabs
marketing/    the launch video as a scene and its builder; build_site.py renders the site
site/         the project site: demo scenes next to the videos rendered from them
brand/        logo files, generated art and the brand guide
bench/        per-stage timing of frames and whole renders
content/      launch post and social copy
tests/        pytest suite
```

## License

MIT, see [`LICENSE`](https://github.com/drknowhow/kinescript/blob/main/LICENSE). The bundled fonts are under the
SIL Open Font License 1.1; their notices are in
[`kinescript/fonts/`](https://github.com/drknowhow/kinescript/tree/main/kinescript/fonts).
