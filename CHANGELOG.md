# Changelog

## Unreleased

- Model layers draw in two thirds of the time, with the same pixels. Of the 2.5 ms a model took each frame,
  1.5 ms went on making two `skia.Point`s per vertex, handing them and the indices to Skia, and freeing them;
  Skia now makes the points while reading back a path numpy writes, indices go as a uint16 memoryview, and a
  primitive's mesh and the GPU's shading paint are kept while they hold. The models-30 stress scene drew a frame
  in 79 ms on the GPU and now in 52 (224 and 186 on the CPU), and rendered at 36 fps on 4 workers and now at 50;
  `examples/model.json` in `bench/stages.py` draws in 7.8 ms at the median and 31.5 at p95, from 10.7 and 41.6.
- PNG sequences, PNG stills and `kine look` sheets are written by Kinescript instead of Skia's PNG encoder, which
  took 337 ms on a 1080p frame: rows take the Sub filter and deflate at zlib level 5 in bands that every core
  compresses at once, while the next frame draws. One second of `examples/hello.json` as PNG frames renders in
  0.49 s instead of 11.7 s, a 3840x2160 still in 0.11 s instead of 1.59 s, a 7680x4320 one in 0.31 s instead of
  4.76 s, and a sheet in 0.11 s instead of 0.32 s. Files are within a few percent of their old size (1080p frames
  4% larger, 4K 6% smaller), still 8-bit RGBA, and decode to the same pixels; the one exception is a transparent
  frame with motion blur, whose colours are now the ones its `.mov` and `.webm` get, a level apart in a few pixels.
- `kine check` places clips without decoding them. Its samples need where layers are, not what they show, yet each
  one decoded the clips on screen, nearly always in a new ffmpeg: the 60-cut stress scene started 169 ffmpeg
  processes and took 17.5 s on the GPU, and `examples/edit.json` took 6.9 s. Only the contrast and covered-text
  checks, which read pixels, decode now: 2.8 s with 36 processes (most of them the sound check's), and 1.2 s. The
  results are the same. `kine probe --at` no longer decodes either.
- A fading shape or picture draws faster: a rect, circle, ellipse, path or polygon with a fill and no stroke, or an
  image or video without a stroke, whose only effect is its opacity now carries that opacity in its paint instead
  of drawing into an offscreen layer of its own. The `expressions` stress scene (600 fading circles) draws a frame
  in a median 8.2 ms instead of 26.3 on the GPU and in about 15 ms instead of 45 s on the CPU; `layers-2000` in
  21.0 ms instead of 24.2 on the GPU and 24-46 ms instead of 35 s on the CPU. Pixels move by at most 3 levels.
  Strokes, text, groups, shaders and models keep the offscreen layer: a stroke over a fill, or a hairline's joints,
  would darken where the two draws overlap.
- An output that can't be written (a read-only file, one another program holds, or a folder this account can't
  write in) is an error before the first frame. A 30 s 1080p render to a read-only `.mp4` drew every frame and
  failed after 2.4 s with `[Errno 13] Permission denied`; it now stops at once and says why.
- A video's sound keeps pace with its picture at any `speed`. ffmpeg's `atempo` takes 0.5 to 2, so at `"speed": 4`
  the picture showed the file's fourth second while the sound played its second, and at 0.25 the sound ran out
  halfway and the rest was silent. The tempo is now made of steps within that range. `kine check` also measured
  such a sound over the wrong stretch of the file.
- A video's sound stops at its trim end unless `loop` is true. `"trim": [1, 2], "dur": 3` played the file's
  second second and then read on to the end of the file; it is now silent after 1 s. An `audio` entry whose `dur`
  runs past its trimmed file is silent after it in the same way.
- A sequence `fade` crossfades its clips' sound: the clip before fades out over the transition while the next
  fades in. The outgoing clip played at full level to its last frame and stopped. An `enter` or `exit` still fades
  a video's sound over its dur; the new video keys `fadeIn` and `fadeOut` set the sound's fades instead (0 for
  none).
- A sequence transition must be shorter than both clips it joins. A 1 s fade between two 1 s clips (or a 100 s
  one) passed `kine check` and then stopped the render with ffmpeg's `Delay must be non negative number`; both
  now report `bad-transition`, naming the clips and their lengths. A sound with a negative `at`, or a video that
  starts before 0 s, plays from the part still going at 0 s; it stopped the render with the same ffmpeg error.
- Sequence `transition` and `clips` are checked. An unknown `type` (`"spin"`), a `type` that is not a string, a
  `dur` that is not a number above 0, `clips` missing or empty (that was 5 s of nothing), a clip without `src`
  (it became a one-frame layer) and an image clip without `dur` (a 0.04 s flash) are errors. A transition without
  `dur` lasts 0.5 s, as the docs said; it was a hard cut. `cut` with a `dur` warns and ignores it; it overlapped
  the clips by that much. `dissolve` and `crossfade` read as `fade` and `none` as `cut`, with a note. A clip whose
  `src` is an image is an image layer that holds for its `dur`.
- `kine check` reports a sound it cannot play. A `src` with no sound in it (a video without a sound track, an
  image) is a `no-audio` error that says which; check was clean and the render stopped on ffmpeg's `Stream
  specifier ':a' ... matches no streams`. A video layer whose `src` is only sound is a `no-picture` error; it drew
  nothing. A `src` that is a folder is a `media-folder` error; it was a PermissionError partway through a render.
- `kine check` warns about a sound nobody will hear (`audio-unheard`): `volume` 0, a `gain` that keeps its loudest
  moment under -60 dBFS (`"gain": -90`), or a sound that starts after the scene ends or ends before it starts. A mix that
  ffmpeg cannot make is an `audio-mix` error in check, since the render would stop on it; any other failure while
  measuring the sound is an `audio-unmeasured` note instead of a crash.
- A video's `volume` must be a number from 0: keyframes took `kine check` itself down with a `TypeError`, `-1`
  flipped the sound's phase and `null` muted it. `null` for an audio `volume`, `fadeIn` or `fadeOut`, an audio
  `trim` with a `null` start, and a `src` that is not one string (a list of files, a number) are errors too.
- Caption files: a UTF-16 or Windows-1252 `.srt` loads (both stopped with `'utf-8' codec can't decode byte`; the
  encoding it used is a note), `&amp;` reads as `&`, `{\an8}` override tags are dropped, and a file with old Mac
  line ends (`\r`) or a `->` arrow is read. Cues that overlap show together, the earlier one on top, without a
  blink where one ends; the later cue used to wait until the earlier one ended. A cue with a bad time, no end time
  or an end before its start is a `bad-captions` error naming the file and line (they were Python errors or were
  dropped), and a file with no cues is a `captions-empty` warning; nothing was drawn, with no word. Inline `cues`
  are checked the same way.
- A bad ease is a check error that names the layer and field and says which forms work. `"ease": 5` and
  `["outCubic"]` stopped loading with `AttributeError` or `unhashable type`, `spring(0, 0, 0)` divided by zero and
  `bezier(1, 2)` or `steps()` recursed until Python gave up. spring takes numbers above 0, bezier four numbers with
  x1 and x2 from 0 to 1, steps a whole number from 1.
- Motion options are checked: `"enter": {"name": "rise", "dist": "far"}` (also `amount`, `from`, `deg`, a list as
  `ease`) stopped the render with `can't multiply sequence`; it is now `enter: dist is 'far'; it takes a number:
  px moved`. `null` for a motion option, a tween's `dur` or `delay`, `anchor`, an `fx` value, or a layer's number
  or colour key means its default (a null tween `dur` used to stop the load).
- Wrong types that crashed while drawing or were silently ignored are errors on their layer: `fx` that is not an
  object of numbers (`5`, `"warm"`, NaN), `shadow` or `box` that is not true or an object, `box.pad` `[]` or
  `"big"`, `font` that is not text, and `anchor`, `blend`, `split` or `align` that is a number, list or object.
- `text` given as a number is read as its digits (`42` shows "42", with a note); a list or object is an error
  instead of its Python repr on screen, and text with half of a surrogate pair (`"a\ud800b"`) is an error instead
  of a crash in Skia.
- Times are checked: `at`, `dur` or `until` of `true`, NaN, Infinity, `10**400`, a list or an object are errors
  (true was read as 1, NaN hid the layer, 10**400 crashed), times past 1,000,000 s are refused, and a huge time
  prints as `1e+308`, not 300 digits. `canvas.duration` of `"0"` or `"a.out-5"`, and an `"auto"` duration that
  would end at or before 0, are errors (they gave scenes of 0 or -4 s).
- `dur` and `until` on one layer warn that `until` is ignored. A layer whose time lies wholly outside the scene or
  its group (`"at": -1, "dur": 0.5`, or `"at": 5` in a 2 s scene) warns that it is never shown.
- A layer or audio id `start` or `prev` is an error: a cue reads those words first, so `"at": "start.out"` never
  meant the layer. A marker with the same name as a layer warns that a bare `"at": "m"` reads the marker.
- Markers take clocks (`"drop": "0:01.5"`, `"45f"`); a marker that is not a time is an error naming it, and
  `"drop.out"` on a marker says a marker has no `.out` (it said `unknown reference 'drop'. Did you mean 'drop'?`).
- Cues can name ids that start with a digit or use letters beyond ASCII (`"1st.out"`, `"título.in"`); a cue on an
  id with a space or dot says that id cannot be cued instead of `cannot read the time`.
- Numbers in text may have exponents and leading dots: `"x": "1e3"`, `".5"` and times like `"1e1"` were "not a
  number". Keyframe times take `"500ms"`, and a time that is not one lists the forms that are.
- An expression whose value overflows a float (`"=10**400"`, `"=exp(1000)"`) is a `bad-expression` error with
  its layer and field instead of an `OverflowError` while drawing, and an expression nested too deeply to parse
  (a 5000-term sum) is an error instead of `maximum recursion depth exceeded`. A colour expression that gives
  something that is not a colour is found by `kine check`, not the render.
- Colours: `rgb()` channels past 0..255 (and alpha past 0..1) are clamped with a `colour-clamped` warning;
  `rgb(300, 0, 0)` drew as dark red (44, 0, 0) by wrapping around, and a negative channel crashed Skia or gave
  complex numbers when mixed. `rgb(1, 2, 3, 4, 5)` is an error instead of ignoring the extras. A bad colour says
  what is wrong (`'#zzzzzz' has 'z', which is not a hex digit`, `has 5 hex digits`, `rgb() takes red, green and
  blue`) instead of `invalid literal for int() with base 16`, and `canvas.bg` errors name `canvas.bg`.
- Colour names are matched in any case (`"RED"` was an error) and messages list the 13 names Kinescript has
  instead of promising "a CSS colour name": they are iOS-style, so `red` is #ff3b30 and `blue` #0a84ff.
- A `$theme` colour used inside `box`, `shadow`, a gradient, `canvas.bg` or a `light` is checked after it is
  filled in, so a theme entry that is not a colour (or a `$name` the theme does not set) is an error on that
  field instead of a failure while drawing.
- Gradient objects are checked: `stops` with fewer than 2 colours, the string `"red"`, a mix of plain and
  `[position, colour]` stops, a bad colour, or a malformed `linear` or `radial` is an error. They drew nothing,
  raised `length of colors must be 2 or more`, or an IndexError. A stop position outside 0 to 1 (CSS `150%`)
  warns that it is drawn at the end.
- `"position": [x, y, z]` is read as x, y and z. A position that cannot be read (`"10,20"`, `[1]`) is an error;
  it was deleted while a note said it had been read, so the layer sat at 0, 0.
- A negative `radius`, `r`, `blur`, `strokeWidth` or `size`, polygon `sides` below 3, and a group with both
  `children` and `layers` (which dropped `layers`) are errors.
- Component placeholders: `{{ title }}` with spaces is filled in, and a placeholder with no value and no default
  (`{{titel}}` misspelt, or a parameter left out) is a `missing-param` error that names the component, the
  placeholder and the parameters that exist. They were drawn as the literal text `{{titel}}`. A parameter given
  as `null` takes its default (it drew "None").
- Scene components of the wrong shape (a list, a string, `layers` that is not a list of objects) and components
  that use themselves, directly or through another, are errors instead of `AttributeError` or `maximum recursion
  depth exceeded`. A built-in component parameter of the wrong type (`kinetic` `every: "fast"` or `words: 5`,
  `callout` `r: "big"`) names the component, the parameter and the parameters it takes. `until` on a `kinetic`
  (or any built-in) now ends it there; the builder's own `dur` won.
- A model layer's `light` object (`{"preset": "dramatic", "from": "left"}`) is read as the docs describe. With
  `from` in it, it was taken for a tween and rejected with `light tween: unknown key 'preset'`.
- A font family that is not installed warns that the text falls back to another font (it fell back silently),
  and a missing `.ttf` file warns too.
- Values that failed while drawing are check errors on their layer: a shader uniform that is text but not a
  colour, `sksl` that is not text, a counter `format` such as `"{:s}"`, path data `d` that does not parse
  (`"M0 0 A 1 2"`), an image `focus` that is not `[x, y]`, and caption `cues` that are not `[start, end, text]`.
- Check issues name their layer: issues raised while expanding a component, a duplicate id, a layer that is not an
  object (`layers[2]`) and errors while drawing had `layer: null`; issues outside layers read `canvas.bg: ...` or
  `markers: ...` instead of `layer 'canvas': ...`.
- Text draws characters its font lacks with an installed font that has them. Emoji, Chinese, Japanese, Korean,
  Hebrew and Arabic in a `text`, `counter` or `captions` layer drew as empty boxes in the default font; they now
  draw, Arabic letters join, accents typed as separate marks sit on their letter, emoji sequences (a family, a
  skin tone) are one picture, and right-to-left words read right to left, put in order on the line by the Unicode
  bidi rules applied word by word (`"مرحبا بالعالم 2026!"` reads as a browser shows it). 3D text also falls back
  to an installed font for each character, unshaped.
- `kine check` warns about text it cannot draw as written: a character no installed font has (`missing-glyph`,
  naming the layer and each character), Arabic letters drawn one by one by `split: "chars"` or `letterSpacing`
  (`unjoined-letters`), and right-to-left letters, separate accents or emoji sequences in 3D text
  (`unshaped-3d-text`).
- A group's opacity no longer changes what a `blend` inside it does. At group opacity 0.999 an orange `multiply`
  child over grey drew (128,64,0); at 0.9989 it drew (255,128,0), blending with nothing, so a fade on such a group
  jumped. The group now fades evenly from the blended picture to what is under it.
- A top-level `blend` in a scene with a `grade` blends with the background. Under any grade, even
  `{"contrast": 1.0001}`, a grey `multiply` rect over a `#3080ff` background drew plain grey (128,128,128) instead
  of (24,64,128), because the layers were graded in a layer of their own that started empty. The background is now
  graded with them, so `contrast`, `saturate` and the other colour keys change it too: the whole frame, as the docs
  say. A `#0b0706` background under `"contrast": 1.04` is now `#060201`.
- With `motionBlur`, a layer's first and last frames are drawn at full strength, and it no longer shows faintly
  one frame after it ends. A red layer on frames 30 to 44 drew (255,127,127) at frames 30 and 45; now red at 30
  and 44 and nothing at 45.
- `grade` grain is as strong with `motionBlur` as without. It was added to every sub-frame and partly averaged
  away (pixel spread 5.86 instead of 8.35 with 8 samples); it is now added once to the blended frame.
- Model layers on the GPU are repeatable: the same frame drawn twice differed by one level at a few hundred edge
  pixels (4x MSAA). Models are now supersampled 2x on the GPU too, at the same speed.
- Screen boxes (`kine probe`, `look --boxes`, `check`) include the stroke: a rect, circle, line, path, polygon or
  arrow with a stroke had a box half a stroke short on every side, and an arrow's box missed its head. An
  anchored path's box was also shifted from where it drew.
- `kine check` warns when text is covered by a later opaque layer (`text-hidden`, naming the layer on top), and
  when one or two words are on screen for under 0.4 s (`read-time`): a 0.2 s one-word flash passed. A text
  within half a frame of its reading time passes (4 words for 1.6 s were flagged for 12 ms), and a text that
  takes over from the same words, such as a chip's highlighted state, counts the time they were already shown.
- `kine check` no longer notes a shape that sweeps across the canvas as `offscreen` because it starts or ends off
  it; it notes one that is off whenever it is on, or for 1 s or more.
- `kine check` notes an odd canvas size (`odd-size`): `.mp4`, `.webm` and `.gif` need even sizes, so a 321x181
  scene comes out 320x180 there, without its last column and row. PNG frames, stills and `.mov` keep 321x181.
- A `.webm` with alpha keeps it as footage, as a `.mov` with alpha already did. VP9 alpha (Kinescript's own
  `--transparent` `.webm` among them) was decoded without it: the clear parts came out black instead of the
  background, and a 30% blue box fully blue. Such clips are now read with libvpx. `kine check` warns
  (`media-alpha-lost`) when this ffmpeg has no libvpx decoder.
- A phone clip stored sideways with a rotation is decoded at its full size. Its size was read before the rotation,
  so a 1920x1080 clip shown upright in a 1080x1920 canvas was decoded at 56% and came out soft. `kine probe` gives
  the size as shown (360x640 for such a clip, not 640x360) and the `rotation`.
- A video file cut short or damaged is reported. A half-copied mp4 that promises 4 s froze on its frame at 2.0 s
  without a word; `kine check` now warns (`media-truncated`) with where its picture stops, and `kine render` lists
  the same warning in its JSON. The frame it holds is the same in order, in a still, or from a worker starting there.
- Footage whose first frame is not at 0 s lasts as long as it plays. An `.mkv` starting at 5 s and running 7 s gave
  a 12 s layer: 5 s of held frame, then black.
- A clip that plays to its end shows its final frame. A 210-frame clip showed frame 208 twice and never frame 209.
- A video `dur` longer than its trimmed clip holds the clip's last frame. With `trim` [2, 4] and `dur` 6 the
  picture played on to 6.9 s of the file. `kine check` warns (`clip-hold`) when a `dur` runs past the trim end
  or the end of the file; past the file it froze the last frame silently. The held frame is the same however a
  render reaches it: a 30 fps clip in a 24 fps scene held frame 28 or 29 depending on where a render worker
  started, and opened past its picture it held the frame before its last.
- A clip at another frame rate or `speed` shows the same frame however a render reaches it: in order, in a still,
  or from a render worker starting partway through. A 30 fps clip in a 24 fps scene showed source frame 3 at frame
  2 played from the start and frame 2 rendered on its own (one frame in four differed; at `speed` 1.5, every other
  one), and a worker starting at frame 11 showed source frame 17 at frame 13 where one render showed 16. Each frame
  now shows the source frame nearest its time, the earlier of two equally near, from any start, and footage shows
  its frame's picture on every motion-blur sub-frame. `kine probe` of a media file also gives its `start`, the
  file's own time of its first frame, which source seconds count from.
- `kine probe` of a PNG says `image: true` with fps 0; ffprobe gives a still 25 fps, so it said `image: false`. A
  model file (`.glb`, `.gltf`, `.obj`) gives its triangles, vertices, materials and size instead of an ffprobe
  error. A media probe names its `kind` (video, audio, image or model), and a scene's probe has `ok` as a media
  file's does.
- A file that lists a video stream with no frames in it, such as a `.ts` cut with `-c copy` between key frames:
  `kine probe` says so in a `note` (it gave w 0, h 0 and fps 0 and nothing more), and the `no-picture` error for a
  video layer that plays it says to cut it again without `-c copy` instead of calling the file sound only. A still
  image as a video layer is a warning (`media-still`).
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
  now an error at once (`gives a number too large for a float`). Arithmetic on text or lists (`[0]*999999999`) is
  refused when the scene loads.
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
- A time or range outside the scene is reported instead of clamped in silence. `render --to 99` on a 2 s scene
  renders to the end and its JSON has a note saying so; `look --at 99` and `still --at 99` draw the last frame,
  and `--at -100` the first, each with a note. Over MCP, `kine_still` also returns the frame number, time and
  size it drew, with the same notes.
- `render --from` starts at the frame on screen at that time. `--from 2.99` on a 3 s scene was rounded to 3 s and
  said there was nothing to render; it now renders the last frame. A range takes every frame on screen from
  `--from` until `--to`, so `--from 1.01 --to 1.02` renders the one frame showing then. An empty range says `to
  must be after from`, and a `--from` past the end gives the time the last frame starts.
- A GIF's `frames` in the render's JSON is the number of frames in the GIF: 30 for 2 s at 30 fps, which the GIF
  plays at 15 fps, not the scene's 60. A GIF of one frame of a 60 fps scene failed with `Error writing trailer`;
  it now holds that frame.
- A scene's frame count rounds half a frame up. 2.55 s at 30 fps (76.5 frames) gave 76 while 2.45 s (73.5) gave
  74; they are now 77 and 74.
- `--scale` that makes a side of the frame smaller than 2 px is an error. `still --scale 0.0001` and `render
  --scale 0.001` drew a 2x2 image or video and said nothing.
- `render -j` uses at most one process per frame and one per CPU, and its JSON has a note when it cut the number.
  `-j 100` on a 60-frame scene started 60 processes.
- When ffmpeg or ffprobe is not on the PATH, `render`, `still`, `look`, `probe`, `check` and the MCP tools say
  which one is missing and how to install it. They said only `[WinError 2] The system cannot find the file
  specified`.
- `KINE_BACKEND` takes `cpu`, `gpu` or `auto` (the default). Any other value is an error that lists them;
  `KINE_BACKEND=bogus` was taken to mean the GPU. `gpu` is an error on a machine with no GPU instead of falling
  back to the CPU.
- `kine voice` names `KINE_VOICE_CMD` (or `KINE_VOICE_CMD_<NAME>`) when its command fails. A program that is not
  there said only `[WinError 2] The system cannot find the file specified`, and a command that printed something
  other than JSON said `Expecting value: line 1 column 1 (char 0)`. A command that prints the wrong number of
  results, and a variable that is not a command (an unclosed JSON list), are also named.
- An unexpected error in `kine` exits 3 with one line that says what failed and where, and asks for a bug report
  (`KINE_DEBUG=1` also prints the traceback). It exited 1 with a traceback, the code `check` uses for a scene with
  errors. Ctrl+C exits 130 without a traceback. `kine docs` lists the exit codes: 0 done, 1 `check` found errors
  or `doctor` found ffmpeg missing, 2 a bad argument, scene or file, 3 a bug.

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
