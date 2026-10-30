# Changelog

## Unreleased

- Sounds are timeline cues. An audio entry takes an `id` and a `role` (voice, music, sfx, ambience) and is timed like
  a layer, with `dur`, `until` and `prev`; layers cue from it (`"at": "vo1.out+0.3"`) and it from layers. A voice
  stretches `"duration": "auto"` to finish; music, ambience and sfx do not, as before.
- A layer's `sound` (`"sound": "pop.wav"`, or `{"src", "at": "out-0.1", "gain", ...}`) plays when the layer starts,
  moves when the layer moves, is skipped when the layer is hidden, and passes through components and sequences.
- `loop` repeats a sound's trimmed file until it ends; `pan` places it from -1 (left) to 1 (right).
- `.mov` (PCM), `.webm` (Opus) and `--from/--to` renders carry the mix. `-o mix.wav` (or `.m4a`, `.mp3`, `.flac`)
  renders the sound alone without drawing frames. `kine probe` lists every sound's start and end.
- A managed mix for scenes whose sounds have roles. Each sound is measured once (loudness, peak, leading silence)
  and set to its role's level: voice -18 LUFS, music -16, ambience -30, sfx peaking at -6 dBFS; `gain` on it is a
  change from that level. Music and ambience dip 14 dB while a voice speaks (0.08 s attack, 0.15 s hold, 0.4 s
  release), worked out from the voice's levels, so the same scene always mixes the same way. An sfx starts past
  its leading silence, so its first sound lands on its cue. The mix is measured and brought to -16 LUFS, then
  limited to -1.5 dBTP. A top-level `mix` changes the targets: `{"loudness", "peak", "levels", "duck"}`, and
  `"duck": false` turns ducking off. Scenes where no sound has a role mix as before.
- `gain` changes over time like any property: keyframes (`[[0, -30], [2, 0]]`), tweens or `=` expressions, in dB
  from the sound's start. `kine check` reports a gain that does not give a number.
- `kine check` hears the sound. It mixes and measures the scene's audio, without drawing a frame and while the
  frames are being checked (warm check time is unchanged for `examples/edit.json` and the promo), and reports a
  voice cut off (`audio-cut`), two voices at once (`voice-overlap`), music and ambience less than 10 LU under a voice
  (`voice-masked`), an sfx whose first sound comes more than 50 ms after its cue (`sfx-late`), a sound that plays
  silence (`audio-silent`), and a mix off its loudness target or over its peak ceiling (`audio-loudness`,
  `audio-peak`). Each issue has the sound's id, a time range and a fix.
- `kine look --audio` (MCP `kine_look` with `audio`) draws the sound under the contact sheet on the frames' time
  axis: a lane per role and one for the mix, each sound's span and id, the duck, and the issues in red. Its JSON
  gives each sound's loudness before and after gain, its leading silence and any shift, and the mix's loudness,
  true peak, loudness range and loudest 3 s.
- Cues from words. A sound's word times come from a words file next to it (`vo1.words.json` for `vo1.wav`), from
  a path or list in its `words` key, or from Whisper's JSON as it is. `"at": "vo1:render"` starts a layer when the
  voice says "render"; `.in`, `.out`, `.mid` and offsets work as on any cue, and repeats are `render_2`, `render_3`.
  Times follow the sound's start and trim. `kine probe` lists each sound's word ids and when they are said, and a
  cue to a word that is not there names the closest one.
- Captions of a voice: `{"type": "captions", "voice": "vo1"}` makes a cue a line from its words, starting when the
  voice starts, and karaoke and pop light each word as it is said instead of guessing from its length.
- `kine look --audio` marks each word a voice says in its lane.
- `examples/voiceover.json`, with a stand-in voice and its words file from `examples/make_media.py`.
- Fixed: a scene whose sounds all ended before it did rendered a video cut short at the end of the last sound. The
  mix is now padded to the scene length. Scenes whose sound already reached the end render the same audio, bit for
  bit.

## 1.1.1 (2026-10-30)

- A project site with demos, each video beside the scene file it was rendered from:
  https://drknowhow.github.io/kinescript/. The README, which PyPI shows as the project description, covers 3D,
  the MCP tools and every example, and its links work from PyPI.
- Fixed: a model frame could differ depending on which frames the same worker had drawn before it; on the GPU,
  edges moved by up to 49 levels. Models now draw into an offscreen the size of the frame from the start.
- 3D models draw faster, with the same pixels: work on a mesh that no frame changes is done once, only the
  vertices of triangles facing the camera are converted for Skia, and the ground shadow's outline skips
  points that cannot be on it. On an RTX 5060 Ti with 6 workers, `examples/model.json` exports in 1.5 s instead
  of 1.7 (one frame draws in a median 10 ms instead of 14.8) and `examples/import.json` in 0.84 s instead of
  1.1 (9 ms instead of 12.9).

Fixes from the first outside test, a promo reel built from `kine docs` alone (`examples/gpu-doom`, #11):

- A sequence is a group that ends with its last clip, so its own id is a cue: `"until": "cut.out"` instead of naming
  the last clip. `opacity`, `blur`, `fx`, `enter` and the other layer keys on a sequence apply to the whole cut, and
  `kine check` accepts them. Clip times are unchanged. A group can also end relative to one of its children.
- `**emphasis**` can start and stop inside a word: `g**P**u` colours only the P. Punctuation written outside the
  stars, as in `**word**,`, is no longer coloured.
- On Linux, importing Kinescript without the system EGL and GL libraries now names the packages to install, and the
  README lists them.
- `kine docs` says that `fx.brightness` adds to each channel and `exposure` is the key that darkens footage.

## 1.1.0 (2026-10-30)

First release on PyPI: `pip install kinescript`.

- Fixed: `kine doctor` failed with "the scene has no layers".
- Depth: every layer takes `z`, `rotateX`, `rotateY` and `backface`; a top-level `camera` pans, dollies,
  orbits and rolls with perspective, and `fixed` layers ignore its moves; a group with `"sort": "depth"` draws
  its children far to near. New motions `flip`, `flip-up`, `tumble` and `push`, also per letter with `split`.
  Built on Skia's 4x4 matrices, so no new dependency, and scenes that use no depth draw exactly as before.
  `examples/3d.json` (180 frames at 1080p) renders in 0.93 s on the GPU.
- 3D models: a `model` layer draws a lit solid, a `cube`, `sphere`, `cylinder`, `cone`, `torus` or `plane`,
  or `parts` merged into one object whose triangles are sorted together. Materials take `fill`, `shine`,
  `metal`, `rim` and `bands` (toon steps); a top-level or per-model `light` takes `studio`, `soft`, `dramatic`,
  `flat` or its own amounts. `explode` sends faces outward and the `build` motion assembles a model. The GPU
  shades per pixel in SkSL with 4x MSAA; the CPU shades per vertex with 2x supersampling. Still no new
  dependency. `examples/model.json` (180 frames at 1080p, up to 10,800 triangles drawn a frame) renders in
  2.5 s on the GPU with 6 workers; one frame takes a median 18 ms on the GPU and 72 ms on the CPU.
- 3D text and paths: `"shape": "text"` extrudes letters in any font and `"shape": "path"` extrudes SVG path
  data, filled, or stroked with `strokeWidth` for line icons; `depth` and a 45-degree `bevel` in px. Holes
  (o, a, 8) stay open, curved sides shade smoothly, and both work as `parts`. Outlines are cached per text and
  font. `examples/title3d.json` (the brand mark and wordmark, 150 frames at 1080p, up to 5,718 triangles drawn
  a frame) renders in 1.34 s on the GPU with 5 workers and 2.87 s on the CPU.
- Model files: `src` on a model layer reads `.glb` and `.gltf` (glTF 2.0) and `.obj` with its `.mtl`, in numpy:
  node transforms, base colours (a texture gives its mean colour), metallic and roughness as `metal` and
  `shine`, flat or smoothing-group normals when a file has none. A layer `fill` recolours the whole model; it
  fits inside `w`/`h`. Long triangles are cut to 1/24 of the model's size so the depth sort holds. Unreadable
  or Draco-compressed files fail at `kine check` with the file named.
- `ground` puts a soft shadow under any model on a level floor, fading as the model rises.
  `examples/import.json` (a 15,264-triangle glTF rocket, 180 frames at 1080p, up to 7,410 triangles drawn a
  frame) renders in 1.74 s on the GPU with 6 workers and 3.04 s on the CPU. `examples/make_media.py` writes
  the rocket.
- `bench/stages.py` times each frame's drawing, conversion and readback, and pipe writes to ffmpeg, plus whole
  renders, for vector, video-edit, model and motion-blur workloads. The README, site and launch-video numbers
  are re-measured: the launch video renders in 4.0 s (was quoted as 6.4 s), `examples/edit.json` in 2.3 s
  (was 6.1 s).
- Motion blur averages its sub-frames on the GPU in one pass and reads back only the result, with the same
  bytes as before. Blurred MP4 frames now take the I420 path. `examples/hello.json` with 8 samples: 8.4 ms a
  frame instead of 111 ms, 1.46 s to render instead of 14.78 s; with a transparent background as `.mov`,
  10.6 ms instead of 191 ms, 2.56 s instead of 13.92 s. Past 16 samples, or on the CPU, the numpy average uses
  16-bit sums.
- Fixed: stills and PNG sequences with motion blur saved the last sub-frame instead of the blurred frame.
- The audio mix runs in its own ffmpeg process while the frames render, and the end of a render only joins
  the streams by copying them. The output is byte-identical. The launch video at `-q high` takes 3.44 s instead
  of 3.99 s, and `examples/edit.json` 2.09 s instead of 2.34 s (medians of 5 runs). A writer thread overlapping
  the pipe to ffmpeg was tried first and made renders 4-5% slower, because drawing holds the GIL.
- Render workers stay warm for two minutes between renders in the same process (the MCP server, a script),
  so the next render skips starting them. Rendering again in one process: `examples/model.json` 1.7 s instead
  of 2.45 s, `examples/import.json` 1.08 s instead of 1.65 s, the launch video 2.2 s instead of 2.6 s.
  `kine render` renders once and exits, so it shuts its workers down as before.
- Fixed: images and probed media were cached by path alone, so the MCP server and `kine watch` kept showing a
  file's old contents after it was replaced. The caches now key on the file's modification time and size.
- `kine check` warns past 60,000 model triangles in one frame.
- `z` used to be ignored with a note; it is now depth in px toward the viewer.
- `kine check` warns when text faces away from the camera (mirrored) and when a layer crosses the camera plane.
- `kine check` validates every key against the schema, names the intended key first, and lists
  every problem at once with its layer and field. Synonyms from other tools, CSS gradients,
  Shadertoy and GLSL shaders, comments and trailing commas in JSON are read and noted.
- Missing media and trims past the end of a clip fail before rendering, with nearby file names.
- `kine docs` rewritten for small models: rules, a complete scene, every default, recipes.
- Fixed: a text frame could depend on earlier frames when weight was animated.
- Fixed: transparent .mov/.webm had dark fringes (premultiplied pixels sent as straight alpha).
- Fixed: odd output sizes crashed MP4 encoding.
- Fixed: MP4 colours. Frames were converted with BT.601 and left untagged, which players read as
  BT.709, so pure green played back as #00d700. Output is now BT.709 and tagged as such.
- Fixed: untagged HD source clips were decoded as BT.601; they now follow the player rule
  (BT.709 from 1280 wide or above 576 lines).
- Fixed: video layers were softened by Mitchell resampling even at native size; now Catmull-Rom.
- Frames reach the encoder as I420 converted on the GPU (3.1 MB a 1080p frame instead of 8.3 MB),
  and clips decode as yuv420p and become RGB on the GPU.
- One render process for short scenes and video scenes on the GPU, up to four for long ones;
  the audio mix and the segment join are one pass; x264 keeps its own GOP defaults.
  Against the previous commit: renders of the three examples are 29-38% faster, and files are
  16-34% smaller except hello at -q high (+19% at equal PSNR).
- Fixed: the contrast check skipped text and counters that used the default colour.
- A contrast warning names the shape behind the text and says to move the text clear of it.
- The bar chart recipe adds a label under the baseline, anchored at its top edge.
- Video decodes at source size on a prefetch thread (single worker on `edit.json`: 34 to 60 fps).
- GIFs render at a 720 px long side, 15 fps, 128 colours: on examples/hello.json, 0.78 MB in 1.1 s against 1.22 MB in 2.4 s before.
- `kine still --at end` and negative times; `kine probe clip.mp4`; render results include w and h.
- `look --boxes` labels only top-level layers, outside their boxes.

## 1.0.0 (2026-10-29)

First release.

- Scenes as JSON: layers, keyframes, tweens, AST-checked expressions, OKLab colour blending.
- Cue timing: `"at": "title.out+0.2"`, markers, group-relative times, cycle and missing-id errors.
- Layer types: shapes, SVG paths, arrows, text with per-unit motion, counters, images, video with trim, speed,
  loop and punch-in zoom, captions from SRT/VTT, SkSL shaders, groups and cut-list sequences.
- Motion presets including `snap` for hard cuts; shader presets `mesh`, `aurora`, `grid` and `score`.
- Components: `lowerThird`, `titleCard`, `callout`, `focus`, `progress`, `kinetic`, and user components.
- GPU rendering through Skia with a CPU fallback; frames split across processes and joined without re-encoding.
- Outputs: MP4, ProRes 4444 and VP9 with alpha, GIF, PNG sequences, with an audio mix.
- Agent tools: `kine check`, `kine look`, `kine probe`, `kine docs`, and an MCP server (`kine mcp`).
- Bundled fonts: Space Grotesk and IBM Plex Mono (SIL OFL 1.1).
