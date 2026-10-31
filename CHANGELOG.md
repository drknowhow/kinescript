# Changelog

## Unreleased

- A video clip's decoder is closed once the clip is off screen. Before, every clip a render had played kept its
  ffmpeg process and six prefetched frames to the end: a 60-cut sequence peaked at 1.66 GB and ran 60 decoders at
  once; it now peaks at 270 MB.
- `grade` vignette and grain no longer paint into the alpha channel of a transparent `.mov`, `.webm` or PNG. The
  vignette made the corners about half-opaque black and the grain gave every pixel some alpha, which also made
  files huge (4 s of `hello.json` as VP9 with alpha: 93 MB, now 0.46 MB). Drawn layers are graded as before.
- Film grain changes on every frame. Frame 123 at 30 fps (and about 2% of frames at most rates) repeated the
  previous frame's grain, because the frame number was recovered from the time by truncation.
- Colour tweens work on `fill`, `stroke`, `emph` and `active`, as the docs say: `"fill": {"from": "#f00", "to":
  "#00f", "dur": 1}` drew a flat grey (it was read as a gradient), and on `stroke` it stopped the render with
  `unhashable type: 'dict'`. A CSS gradient on `stroke` also stopped the render; it now draws.
- `kine check` evaluates expressions at every frame their layer is on, not only at its start, middle and end, so
  `"x": "=100+1/(t-0.7)"` is reported (`bad-expression`, with the frame) instead of failing the render at 0.7 s.
  An expression that still fails while drawing names its layer, property and time.
- `**` in an expression is a float power: `9**9**9` made `check` run for hours computing an exact integer; it is
  now an `OverflowError` at once. Arithmetic on text or lists (`[0]*999999999`) is refused when the scene loads.
- `"type": "Text"` (any capitalisation of a layer type) is that layer type. `Text` was read as a 3D model with
  shape text, because model shape names were matched first.
- A scene component whose group lists its layers under `"layers"` (not `"children"`) can be used more than once.
- Values of the wrong type are errors from the schema instead of Python errors later: `theme`, `markers`, `grade`
  and `components` that are not objects, `motionBlur` that is not an object (a number is read as its `samples`),
  `audio` that is a number, an audio `volume`, `fadeIn`, `fadeOut` or `gain` that is not a number (or keyframes,
  for gain), an infinite `canvas.duration`, a `canvas.bg` that is a number, a sequence's `clips` or `transition`
  that is not a list or object, shader `uniforms` that are not an object, a path `d` that is not a string and
  `points` that are not `[x, y]` pairs. `null` for `canvas.w`, `h` or `fps` means the default. Anything left that
  fails while loading is a `load` issue in `kine check`, not a traceback.
- A video clip opened partway through (a still, `kine look`, the start of each render worker's chunk) shows the
  same frame as one played through to that point. Its seek time was rounded up past the frame's own time, so
  ffmpeg could skip that frame and the whole chunk ran one source frame ahead.
- A render with sound keeps its last frame. The mux stopped with the shorter stream, and the sound ends at the
  scene's duration, which can come before the last frame ends: a 0.49 s scene at 30 fps has 15 frames and lost
  the 15th. The mux now cuts at the video's own length.
- A layer that starts or ends within a millionth of a frame of a frame boundary starts or ends on it. `"at": 2.2,
  "dur": 8.9` ends at 11.100000000000001 s, so at 30 fps it also showed on frame 333, and a layer placed at its
  `out` started a frame late, on 334.
- The `aurora` shader draws correctly on the CPU backend. It raised a negative number to a power, which SkSL
  leaves undefined, and Skia's raster backend drew the bands wrong.
- An mp4 or GIF of a scene with a transparent background shows it over black. The frames kept straight colour
  and lost the alpha, so a 50% red layer over nothing came out at full red; it is now half red. Asking for
  `transparent` with mp4 or GIF is an error, since neither keeps alpha.
- A `.mov` or `.webm` whose background colour is partly transparent (`#ff000080`) is written with straight
  alpha. Only a fully transparent background switched to straight alpha before, so the background's red read back
  as 128 instead of 255.
- Frame 0 with `motionBlur` is no longer dimmed. Half its samples came from before the scene starts, where
  nothing is drawn, so a white layer was 127 on frame 0; samples now stay inside the scene.
- 3D text with an animated `size`, `depth` or `bevel` draws the same whichever frames came before it. The outline
  was cached under the size rounded to 0.01 but built from the exact size of the first frame that used that
  key, so a render split across workers could differ by a few pixels.
- A hidden video layer, or a video in a hidden group, is silent. A looped video with a `trim` end loops only the
  trimmed part, picture and sound; it used to play on to the end of the file before looping.
- `trim` and `speed` are checked when the scene loads: a negative start, a start that is not a number, a `trim`
  that is not a list, and a `speed` of 0 or less are errors (`bad-trim`, `bad-value`). Before, a speed of 0 or a
  `trim` of `3` crashed the load and a negative start played the clip's sound early.
- Media whose tags are not ASCII (a Cyrillic or Japanese title) probes on Windows, where ffprobe's output was
  read in the system code page. A temp folder with `'` in its path no longer breaks joining the workers' chunks.
- `kine` writes UTF-8 when its output goes to a pipe or a file on Windows. A layer id or text in Chinese, Japanese
  or emoji could not be written in the console's code page, so a render that had finished was reported as failed
  (rc 2, a `'charmap'` error), and a name like `café` came out as a byte that is not UTF-8.
- The MCP server reads its input as UTF-8 (a scene path or text with `é` arrived as `Ã©`) and keeps serving after
  a line that is valid JSON but not a request, such as `123` or a batch, or an `initialize` with `"params": null`.
  Each of those ended the server.
- Bad arguments get an error in the JSON instead of a traceback or a wrong file:
  - `--scale` must be above 0 and give a frame no larger than 16384 px a side (`0` made a 2x2 video, `100` failed
    inside Skia).
  - `--at`, `--from` and `--to` must be finite; `--from` and `--to` must not be negative (`--from -1` rendered
    everything). `check --step` must be above 0 (`0` divided by zero).
  - `look` takes `-n` and `--cols` of 1 or more and `--width` from 16 to 4096; `--cols` past the frame count is
    cut to it (it made a sheet 1000 thumbnails wide). `look --at` reads times as `still` does: negative from the
    end, `end` for the last frame, clamped to the scene.
  - A scene path that is a folder or not UTF-8 text says so.
- Output paths are checked before anything renders. `render`, `still` and `look` refuse an extension they cannot
  write (`.mkv` got an MP4 inside, `.webp` a JPEG), a folder (a render left its temp name, `final.mp4`, inside
  it), and a file the scene reads (a render to its own footage replaced it; `-o scene.json` replaced the scene).
  `still` and `look` make missing folders; before, they said `ok` and wrote nothing. `look -o x.jpg` writes a
  JPEG.
- A PNG sequence named with a pattern, such as `-o frames/f_%04d.png`, numbers its files with it instead of
  writing `f_%04d_00000.png`.
- `still --at -1` names its file `<scene>_end-1s.png`. It was named like `--at 1`, so one overwrote the other.
- `check` looks at every layer at least once. A layer shorter than `--step` (a 0.15 s flash of text) could fall
  between two samples and go unchecked; it is now drawn at its middle.
- `watch` keeps going when the scene file disappears for a moment or is locked while an editor saves it, and
  Ctrl+C stops it without a traceback.
- Over MCP, `kine_check` on a scene that fails to load returns its issues as `kine check` does, not a one-line
  error. `kine_still` takes `"at": "end"`. `kine_probe` of a media file returns its duration, size and fps, as
  the command does. A missing argument is named. `kine_render` reports `out` as a full path and rejects an unknown
  `quality`. The temporary images behind `kine_look` and `kine_still` are deleted after they are sent.
- Render workers kept warm by the MCP server exit when the server is killed. Before, they ran until someone
  killed them.
- `kine doctor` names ffmpeg or ffprobe when one is not on the PATH and exits 1. `kine new` makes missing folders.

## 1.3.1 (2026-10-31)

- A scene's own component can hold a group and still be used more than once: the ids of nested layers now get the
  instance id in front too, as top-level ones did. Before, a second use stopped with `duplicate layer id`.
- The site's sound section has a demo of the three voice backends speaking the same line,
  `site/demos/voices.json`, with the time each took. `bench/voices.py` measures those times.
- `kine docs`, the README and the site no longer say Chatterbox acts on tags without qualification. In the demo it
  spoke `[whispers]` in its normal voice and left a silent pause for `[laughs]`; in test takes it made a sound
  where the laugh goes in six of nine and whispered in none of seven. ElevenLabs acted both.
- `kine check` warns (`media-offscreen`) when a video, image or shader layer is more than half off the canvas for
  a second or more, and for a video or shader without an `anchor` it says why: they are placed by their top-left
  corner even when `x` and `y` are set. A sequence is reported once, under its own id. Found in #11, where a clip
  set to the middle of the canvas landed mostly off it and `check` said nothing.
- `allowOffscreen` on a group or sequence now covers the layers inside it.
- `kine docs` says plainly which layers default to `topleft`: video and shader always, an image only without `x`.

## 1.3.0 (2026-10-30)

Voices that act. A line's text can say how to speak it, each sound picks its own voice backend, and two new
backends speak it: Chatterbox on your own machine and ElevenLabs' API.

- The Kokoro voice plugin is on PyPI: `pip install kinescript-kokoro` (Python 3.10 to 3.12). Pushing a tag such
  as `kokoro-v0.1.0` publishes it from `plugins/kinescript-kokoro` (`.github/workflows/release-kokoro.yml`), after
  it has spoken a line on CI. With no voice backend installed, `kine voice` prints that command.
- The project site's hero is a narrated tour, `marketing/hero.json`.
- `kine docs` and the Kokoro plugin's README cover Kokoro's pronunciation fixes: `[word](/phonemes/)` says a word
  with the IPA phonemes given, `[word](-1)` and `[word](+1)` change its stress, and `"speaker": "af_heart,af_bella"`
  blends two voices.
- Delivery tags. `[whispers]`, `[laughs]` or `[sighs]` in a voice's text tell a backend that acts on tags how to say
  what follows. Tags are not words: `[[marks]]`, word ids and captions skip them, a backend without tags gets the
  line without them, and `kine voice` lists what a backend could not do under `ignored`.
- A sound picks its voice backend with `backend` and passes it settings with `options`; both join what decides
  whether a line is spoken again, and a sound without them keeps its old fingerprint, so no existing voice turns
  stale. `KINE_VOICE_CMD_<NAME>` runs one backend in another Python, and `KINE_VOICE_CMD` also answers for the
  backend whose module it runs. Backends get the scene's folder as `base`, for files their options name.
- `kinescript-chatterbox`: Resemble AI's Chatterbox on your own machine. Turbo acts on tags such as `[laugh]` and
  `[sigh]` (`[laughs]` reads as `[laugh]`), the standard model takes `exaggeration`, and either copies the voice of
  a clip of 5 s or more. Chatterbox gives no word times, so the plugin finds them with a wav2vec2 forced alignment
  that reads numbers as they are said and lets a tag such as a laugh fill its own place.
- `kinescript-elevenlabs`: ElevenLabs' API (Eleven v4 by default, with audio tags), any voice in the account by name
  or id, voice settings through `options`, and word times from ElevenLabs' character times. It has no
  dependencies and needs `ELEVENLABS_API_KEY`.
- What a voice backend prints goes to stderr. It used to land in `kine voice`'s JSON and in `kine mcp`'s replies,
  which Chatterbox's loading messages broke.
- Pushing a tag such as `chatterbox-v0.1.0` or `elevenlabs-v0.1.0` publishes that plugin to PyPI
  (`.github/workflows/release-plugins.yml`) after it has spoken a line on CI: Chatterbox with its own models on the
  CPU, ElevenLabs against a stand-in for its API.

## 1.2.0 (2026-10-30)

Sound. Voice-over, music and effects sit on the same timeline as the picture, are levelled and checked without
listening, and a voice can be spoken from a line of text in the scene.

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
- Voice-over from text. A sound's `text` (with `speaker` and `rate`) is what a voice says, and `kine voice
  scene.json` (MCP `kine_voice`) speaks it into its `src`, `voice/<id>.wav` unless it names one, and writes its
  words file. `[[go]]` in the text gives the next word the id `go`, so `"at": "vo1:go"` stays on that word when
  the line is rewritten. A sound is spoken again only when its text, speaker or rate changed (`--force` speaks
  all, `--only vo1` one). `kine check` reports a sound with text that is not spoken yet (`voice-missing`, an
  error) and one whose text changed since (`voice-stale`). Rendering never speaks.
- Voice backends are plugins in the `kinescript.voice` entry-point group, or a command named by `KINE_VOICE_CMD`
  for a backend in another Python, which reads the requests as JSON on stdin; scene text never goes through a
  shell. `plugins/kinescript-kokoro` is the first backend: Kokoro voices, with the time of each word for English.
- Captions that follow a voice no longer get `read-time` warnings; they keep the pace the voice speaks at.
- `examples/narrated.json`: a voice written as text, a marked word and a click on it.
- The Kokoro plugin installs from this repository:
  `pip install "kinescript-kokoro @ git+https://github.com/drknowhow/kinescript#subdirectory=plugins/kinescript-kokoro"`.
  It is not on PyPI, and `kine voice` prints this command when no backend is installed.
- The project site has a sound section: a demo whose voice Kokoro spoke from the scene's text, next to that scene,
  and the `kine look --audio` sheet of it. The timelines under the site's videos show their sounds. The README has
  a Sound section.
- `kine probe` names the file a voice spoken from text plays (`voice/<id>.wav`) instead of none.
- Fixed: theme names such as `"$muted"` were not replaced inside a scene's own components.
- Before a mix, ffmpeg is asked only for its version, not its encoder list as well, so a one-worker render starts
  drawing about 25 ms sooner. `examples/edit.json` renders in 2.20 s, as 1.1.1 does (2.19 s) on the same machine
  the same day.
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
