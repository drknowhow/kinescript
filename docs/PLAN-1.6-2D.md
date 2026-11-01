# Kinescript 1.6: 2D plan

Proposal, 2026-11-01; the decisions at the end were agreed the same day. Numbers marked "target" are budgets to
measure against. The others were measured on the development machine on 2026-11-01 (RTX 5060 Ti, skia-python 144.1,
Python 3.14) or come from the 2026-10-31 stress report.

## Where 2D stands in 1.5.1

- Shapes: rect, circle, ellipse, polygon (and stars), line, path (SVG path data or points), arrow. Fill and stroke
  take a colour or a linear or radial gradient; `draw` and `drawStart` trim a stroke on.
- Text with `split` units, per-unit expressions (`unitX`, `unitY`, `unitRotate`, `unitScale`, `unitOpacity`),
  `**emphasis**`, a box, typing with a caret, fallback fonts, Arabic joining and bidi order. Counters and captions.
- Images and video with fit, zoom and focus; SkSL shaders; groups that clip to their rounded rect.
- Values: constants, tweens, keyframes with an ease per key, and AST-checked expressions. 33 named eases plus
  `spring`, `bezier` and `steps`.
- 24 enter and exit motions, combined with `+`, staggered across split text.
- Look: 10 blend modes, blur, drop shadow, an `fx` colour matrix, a whole-frame grade, motion blur by sub-frames.
- Components, built in and your own.

What a motion designer reaches for and does not find:

- Keyframes ease each segment on its own, `inOutCubic` unless set, so a layer with three or more keys stops dead at
  every inner key. There are no curved motion paths: x and y animate separately. `position` keyframes such as
  `[[0, [100, 300]], [2, [540, 60]]]` fail with `x is [0, [100, 300]]`, because `schema.py:789` reads any two-item
  list as `[x, y]`.
- `d` does not animate (it is in `scene.STRUCT`), so shapes cannot morph.
- No dashes, no rounded corners on paths, one radius for all four corners of a rect, no boolean shape operations.
- No masks or track mattes. A group clips only to its rounded rect, and `wipe` clips to the layer's box.
- No outer glow, outline, inner shadow or displacement. Skia has seven blend modes Kinescript does not name:
  hardLight, colorDodge, colorBurn, exclusion, hue, saturation and luminosity.
- `repeat` exists only on models. No particles, no image sequences or sprite sheets.
- Stagger works only on split text, in reading order. A group cannot stagger its children.
- No time remapping: `speed` and `loop` are for video only.
- No text on a path, and per-unit colour is not a key.
- No SVG files: `image` reads png, jpg, webp, gif and bmp, and `"type": "svg"` is an alias for `path`.
- Motions happen only on entry and exit. There is nothing for the middle of a layer's life (a pulse, a shake, an
  idle float).
- Gradients do not animate and have no sweep (conic) form.

Bugs found while reviewing:

- A gradient `fill` on a text layer passes `kine check` and draws in the theme's paper colour: `_draw_text` only ever
  calls `fpaint.setColor`.
- The `position` keyframes above.
- README.md:335 still says text has no shaping for complex scripts or bidi, which 1.4 added.

## Where the frame time goes

One renderer on the GPU, 1080p, frames 50 to 89 drawn and read back, no encoding:

| scene | what it is | ms a frame |
|---|---|---|
| `layers-2000` | 2000 shapes, keyframed y, staggered fades | 26.4 |
| `expressions` | 600 layers, 4 expressions each | 14.4 |
| `keys-dense` | 300 layers, 121 keyframes on x, y and rotate, colour keys | 14.8 |
| `text-split` | 40 lines split per character, blur-rise stagger, per-unit wave | 24.4 |

cProfile over frames 10 to 49 says the 2D cost is mostly Python, not drawing:

- `layers-2000`: about six `Layer.get` calls per layer per frame, and a new `skia.Paint` plus a colour conversion
  (`color.to_argb_int`) for every fill. `drawPath` took 0.06 s of the 1.5 s spent under `Renderer._layer`.
- `keys-dense`: each keyframed value scans its keys from the first one (`fk` in `props.compile_prop`), 7 µs a call
  with 121 keys.
- `expressions`: each evaluation copies the whole helper dict (`dict(env, t=t, ...)`) before `eval`.
- `text-split`: about 3,050 save/restore pairs and 950 `saveLayer` calls a frame, one blur layer for each moving
  letter.

So 1.6 starts by making the per-layer path cheaper, and every new feature gets a stress scene and a budget before
it merges.

## What Skia 144.1 already has

Present in the installed wheel: `skia.Op` (path union, difference, intersect, xor), `DashPathEffect`,
`CornerPathEffect`, `DiscretePathEffect`, `Path1DPathEffect`, `Path.interpolate` and `isInterpolatable`,
`GradientShader.MakeSweep` and `MakeTwoPointConical`, `SVGDOM`, `Canvas.drawAtlas` with `RSXform`,
`TextBlob.MakeFromRSXform`, `PictureRecorder` and `Canvas.drawPicture`, `ImageFilters.Dilate`, `Erode` and
`DisplacementMap`, and `PerlinNoiseShader`. Only their names were checked. The first PR of each phase tries the
calls it needs on both backends before building on them.

## Rules every change keeps

- A frame is a pure function of (scene, frame number). Particles, rough-line boil, scramble text and random stagger
  orders take a `seed` and compute each frame from it, with nothing carried between frames.
  `tests/test_core.py::test_frame_is_pure_function_of_index` gets a case for each.
- Scenes that use none of the new keys draw the same bytes as 1.5.1, Phase 0's speedups included. `bench/same.py`
  adds `marketing/promo.json`, `examples/edit.json` and the 2D stress scenes to its list and runs on every PR, GPU
  and CPU.
- No new required dependency. SVG parsing uses `xml.etree`.
- Every feature works on both backends. Where the CPU takes a cheaper path, `kine docs` says so.
- Every new key is validated in `schema.py` with errors that name the layer and field, documented in `docs.py`, and
  gets a `kine check` rule for its most likely mistake.
- Every feature gets a stress scene in `bench/stress.py` and a budget. A feature that misses its budget on the CPU
  gets a `kine check` warning, as `slow-depth-of-field` does.
- Before a visual change is called done: `kine check`, then `kine look --boxes`, and look at the image.

## Phases

Each bullet is about one PR. Phase 0 goes first so new features are measured against the faster base. Phases 1 to
3 can run in parallel after it; 4 to 6 build on 2 and 3.

### Phase 0: a cheaper frame

- Keyframes find their segment with `bisect` over times kept per layer length, in place of the scan from the first
  key. Same segment, same value, same bytes. Target: `keys-dense` 11 ms or less.
- Expressions compile to a function of (t, d, p, T, i) with the helpers as its globals, so no dict is built per
  call. The arithmetic is unchanged. Target: `expressions` 11 ms or less.
- A static layer works out its constant values, transform and paints once: `_fill_paint` and `_stroke_paint` keep
  a paint for a constant colour, and `to_argb_int` keeps the integer for a parsed colour. Target: `layers-2000`
  18 ms or less.
- Groups whose props, children and media are all constant are recorded once into an `SkPicture` per scale and
  replayed with `drawPicture`. It stays only if `bench/same.py` passes on both backends and it measures faster.
  Phase 5's SVG pieces and Phase 4's repeated groups replay from it.
- `bench/same.py` also draws `examples/edit.json`, `marketing/promo.json` and the 2D stress scenes, and the 2D
  stress scenes that measure later phases (`morph-200`, `masks-50`, `glow-text`, `repeat-10k`, `particles-5k`,
  `svg-500`, `along-100`) arrive with the feature each one measures.

Status (2026-11-01): merged (#84) without the picture cache, which moves to Phase 5, where SVG pieces give it
constant groups to replay. Frame times on the GPU, frames 50 to 89: `layers-2000` 16.8 ms, `expressions` 9.7 ms,
`keys-dense` 10.2 ms. Draw time per frame against 1.5.1, alternating runs: 15.8 to 12.5, 7.3 to 5.4 and 9.1 to
5.8 ms. `bench/same.py` finds no frame that differs on either backend.

### Phase 1: motion that flows

- `"smooth": true` on the keyframes object (`{"keys": [...], "smooth": true}`, a form `compile_prop` already
  reads) passes through every key with no stop: a monotone cubic through the values, so it does not overshoot
  them, at rest at the first and last keys. `"hold"` as a key's ease keeps its value until the next key.
- `position` keyframes: `"position": [[0, [100, 300]], [1, [500, 120]], [2, [900, 300]]]` moves the anchor through
  the points, and with `smooth` along a curve through them. The `schema.py:789` misread becomes this.
- `along` moves a layer along a path: `"along": {"d": "M0 0 C 300 -200 600 200 900 0", "p": [[0, 0], [2, 1]],
  "orient": true}`. `p` is the share of the path's length (eased like any key), `orient` turns the layer to the
  path's direction, and `d` may be `"#routeId"` to follow a path layer, so a dashed route draws on while a dot rides
  it. x and y move the path, as they move a path layer, and move a rider off a path layer it follows.
- Stagger on groups and in any order. A group's enter with a stagger enters each child in turn instead of the group
  as one picture: `"enter": {"name": "rise", "stagger": 0.08}`. `order` sets who goes first, for groups and split
  text: `start` (default), `end`, `center`, `edges`, `random` (with `seed`).
- Time on groups: `speed`, `loop` (seconds to repeat) and `time` (keyframes or an expression giving the children's
  time, so a constant freezes them). A looping icon becomes one group with `"loop": 1.2`.
- Check rules: `stop-and-go` (an inner key stops a layer between two moves the same way: add `smooth`) and `strobe`
  (a layer moves more than about 40 px a frame at 1080p without `motionBlur`, naming the time).

Status (2026-11-01): smooth keys, position keyframes and `stop-and-go` merged (#85); `along` and `strobe` (#86);
group stagger and `order`, then group clocks, in the two PRs after. Changes from the bullets above: smooth keys are
at rest at the first and last key whatever their eases. x and y move an `along` path as they move a path layer, so
a path's data rides the same route in both. `strobe` needs five fast frames running, since three flagged the first
frames of the docs' outCubic bar recipe. The remap key is `timeRemap`, because `time` already reads as `at`.
`along-100` draws in 5.6 ms a frame on the GPU, 0.7 ms more than the same layers standing still.

### Phase 2: vector shapes

- Morphing: `d` takes keyframes, `"d": [[0, "M0 0 L100 0 L50 90 Z"], [1, "M50 0 A50 50 0 1 1 49.9 0 Z"]]`. Any two
  paths morph: each contour is resampled by length to the same number of cubic segments, the start point is turned
  to the one that moves least, and a missing contour grows from the centre of the one it pairs with. The matched
  points are kept per key pair, so a frame only interpolates arrays. Check: `morph-contours` (info) when the
  contour counts differ. Target: `morph-200` (200 paths morphing at once) under 8 ms a frame on the GPU.
- Strokes: `dash` ([on, off, ...] px) with `dashOffset`, which animates for marching dashes and works with `draw`;
  `corner` (px) rounds every corner of a path, polygon or line; rect `radius` takes four values [tl, tr, br, bl].
- `rough`: a hand-drawn wobble, `{"amount": 3, "seed": 1, "boil": 8}`, where `boil` re-draws it that many times a
  second for the drawn-by-hand flicker. Built on `DiscretePathEffect` with its seed, so frames stay pure.
- `combine` on a group: `union`, `subtract`, `intersect` or `xor` merges its shape children (text glyph outlines
  included) into one path drawn with the group's fill and stroke, so a word cut out of a badge is one group.
  Children whose values change are combined each frame, constant ones once.
- Gradients: the text gradient bug is fixed, and text stroke takes gradients too. A `sweep` gradient
  (`{"sweep": [0.5, 0.5], "stops": [...]}`), `angle` in degrees for a linear one, and gradients that animate: a fill
  keyframed between two gradient objects with the same number of stops mixes their points and colours.

Status (2026-11-01): gradients (#89), then strokes, `rough`, `combine` and morphing in the PRs after. Changes from the bullets above: a gradient
animates into one of the same kind as well as the same stop count, since a linear and a sweep gradient have no
points in common to mix. `dash` marks out the strokes of images and videos too, and `radius` takes four values on
groups, images and videos as well as rects. `strokes-300` draws in 6.2 ms a frame on the GPU, 3.2 ms more than its
layers without dash, corner and four radii: about 1.2 ms for each 100 marching dashes, 1.2 ms for each 100
corners that animate (a constant corner is cached and costs 0.1 ms) and 0.5 ms for each 100 animated radii.
`rough` takes a `length` too, the spacing of the points that stray; `rough-200` draws in 4.1 ms, 1.6 ms more than
without it, since a layer keeps its last wobbled path while its seed and outline stay the same. `combine` keeps
each partial merge and starts again from the first child that changed, so a hole moving through a badge with a
word cut out costs one path operation a frame; `combine-60` draws in 5.9 ms, 4.0 ms more than its layers drawn as
themselves. A combined group draws with `dash` and `rough` as a shape does. Morphing resamples each piece to
points 1.5 px apart rather than to cubic segments, draws each key as written on its own time, and mixes a pair
with Skia's Path.interpolate; `morph-200` draws in 2.2 ms a frame on the GPU against the 8 ms target.

### Phase 3: masks, mattes and effects

- `mask` on any layer: `"mask": "lens"` or `{"layer": "lens", "mode": "alpha", "feather": 12}`, mode `alpha`,
  `luma`, `alphaInvert` or `lumaInvert`. The mask layer draws only as a mask, as a model's texture group draws only on
  the model, and animates on its own clock in the masked layer's parent space. Drawn as a `saveLayer` bounded by the
  layer's box, the content, then the mask with `kDstIn` (or `kDstOut` to invert; luma through a colour filter).
  Video inside a word is a video layer masked by a text layer. Check rules: `mask-unknown`, `mask-misses` (the mask
  never overlaps the layer at any sampled time). Target: `masks-50` under 10 ms a frame on the GPU.
- Blend modes: hardLight, colorDodge, colorBurn, exclusion, hue, saturation, luminosity.
- Layer effects, each an image filter on the layer's offscreen so both backends draw them: `glow` (`{"color",
  "size", "amount"}`, a blurred, tinted copy of the layer's shape), `outline` (`{"width", "color"}`, by `Dilate`),
  `innerShadow`, and `displace` (`{"amount", "scale", "speed", "seed"}`, a Perlin-noise warp whose noise slides with
  time). Target: a glowing 1080p title under 1 ms on the GPU (`glow-text`).
- Check rule: `slow-effects` on the CPU when masks and effects together pass a measured per-frame cost.

Status (2026-11-01): masks in their own PR. Changes from the bullet above: the mask layer stands beside the layer it
masks, in the same group (check says `bad-mask` otherwise), so its x and y mean what they say. A saveLayer bounded by
the masked layer's box alone was not enough: the layer and its mask are recorded as pictures, and both layers hold
all of both, in whole pixels, since Skia's CPU antialiasing of a circle changes by up to 56 levels where a layer's
bounds cut through it. Those bounds matter on the CPU: `masks-50` took 2.1 s a frame in layers from the frame's
corner and 0.16 s in these (without feathers). On the GPU it draws in 5.4 ms, 3.4 ms more than unmasked. A feather
is Skia's blur filter, which is slow on the CPU (15 ms for a 400x150 layer), as a layer's `blur` already is: the
`slow-effects` rule covers both.

Status (2026-11-01), effects: in their own PR. `displace` is not an image filter: Skia's DisplacementMap takes the
nearest pixel, which left bent edges in steps, so the layer is drawn into an image and redrawn through a runtime
shader that samples between pixels. `outline` is copies of the layer round circles instead of `Dilate`, which is
square and took 135 ms a pass on the CPU for a 220 px word. All four draw in an offscreen layer only as large as what
the layer draws: in a layer the size of the frame, which `blur` and `shadow` still use, the CPU took 16.5 s for a
`glow-text` frame, against 0.31 s now. A glowing 220 px title adds 0.6 ms a 1080p frame on the GPU (target: under
1 ms). A model keeps `glow` and `outline` as its material's glow and its edges. `blur` and `shadow` keep their
frame-sized layers in 1.6, since bounding them changes pixels; the CPU blurs a 1080p layer in 0.44 s, which
`slow-effects` reports.

Status (2026-11-01), `slow-effects`: in its own PR. The renderer counts, each frame, the pixels its offscreen effects
blur (an outline's copies, a mask's two layers and displace count as the share of a blur they take), at 0.44 s for a
1080p layer; `check` warns past 0.25 s a frame when it draws on the CPU. On the stress scenes and on layers with
blur or shadow, the estimate came within 2% below to 35% above the measured time.

### Phase 4: many things at once

- `repeat` on 2D layers (shapes, text, images, groups) with the model's keys plus `opacity` and `stroke`, through
  the same `Copies` class: `{"count": 60, "rotate": "=i * 6", "x": "=cos(i * 0.1047) * 300", "fill":
  "=hsl(i * 6)"}`. Copies draw in one loop under the layer's transform with kept paints; a constant group draws
  once into a picture and replays per copy. Up to 10,000. Target: `repeat-10k` (10,000 circles moving) 30 fps on
  the GPU.
- `particles` layer: an emitter with `rate` or `burst`, `life`, `speed`, `angle`, `gravity`, `drag`, `size` and
  `opacity` over life (keyframes over 0..1 of a particle's life), `spin`, `fill` (a list taken in turn), `shape`
  (circle, rect, star, a character, or an image), `emitter` (point, line, rect or circle) and `seed`. Each particle's
  birth, start and path come from its index and the seed, with ballistic motion and drag in closed form and noise for
  turbulence, so frame n needs no frame before it. Computed in numpy, drawn with `drawAtlas` from one sprite per
  shape and colour. Presets: `confetti`, `sparks`, `snow`, `dust`, `bubbles`. Target: `particles-5k` (5,000 alive)
  30 fps on the GPU. Check: `particle-cost` on the CPU.
- Image sequences: `image` with `frames` (`"frames/run_%03d.png"` or a folder) or `sheet` (`{"src": "run.png",
  "cols": 8, "rows": 2}`), `fps` and `loop`, for frame-by-frame animation. Check: `missing-frames`.

### Phase 5: type and vector art

- Text on a path: `textPath` (`{"d": "...", "offset": [[0, 0], [2, 1]], "align": "start"}`) lays the letters along a
  curve, each turned to it, through `TextBlob.MakeFromRSXform`. Split units and their keys still apply.
- Per-unit keys `unitFill` and `unitBlur`, and an expression helper `sweep(i, n, p, soft)` that is 1 for units the
  sweep has passed and 0 for the rest, with a soft edge, so a highlight running across a line is one expression:
  `"unitFill": "=mix('#f2f0e8', '#ff633f', sweep(i, n, p, 3))"`. `n` (the unit count) joins `i` in unit expressions.
- SVG files. An `image` with an `.svg` src draws through `SVGDOM` as vectors, sharp at any zoom. An `svg` layer
  (`{"type": "svg", "src": "logo.svg"}`) turns the file's path, rect, circle, ellipse, line, polyline and polygon
  elements, `g` transforms, fill, stroke, opacity and linear and radial gradients into child layers named from the
  file's ids, so `draw`, a staggered enter, morphing and masks work on each piece. What it skips (CSS in `<style>`,
  filters, `<text>`, clip paths) is listed by `kine check` as `svg-unsupported`. Constant pieces replay from a
  picture (Phase 0). Target: `svg-500` (a 500-element drawing, static) under 2 ms a frame on the GPU.

### Phase 6: motion library

- Enter and exit motions: a drop with real bounces, `elastic`, `squash` (squash and stretch on landing), `swing`,
  `iris` (a circle opening from the middle), `clock` (a radial wipe), `blinds`, `rise-mask` (each line rises from
  behind a cut line, the usual title reveal), `scramble` (text settles from random characters, seeded), `glitch`
  (RGB split and slice offsets), `whip` (a fast slide with directional blur), `stretch`. The bouncing drop is
  `land` (decision 2).
- `emphasis`: motions in the middle of a layer's life, `[{"name": "pulse", "at": 1.2, "dur": 0.4, "repeat": 2}]`, or
  `"loop": true` for idle ones. Names: `pulse`, `shake`, `wiggle`, `wobble`, `float`, `bob`, `heartbeat`, `flash`,
  `tada`, `spin`. They use the same `Mods` as enter and exit, so they combine with them.
- Sequence transitions: `iris`, `push` (both clips move), `whip`, `luma` (wipe by a gradient's or an image's
  brightness, with `soft`) and `glitch`.
- Optional, if the phase has room: chart components `bar-chart`, `line-chart` and `pie-chart`, built from today's
  layers (bars grow, values count up, the line draws on, slices sweep). The bar recipe in `kine docs` is what they
  replace.

### Phase 7: tooling and release

- `kine look --trails` draws each moving layer's anchor path over its tiles, a dot per frame, so spacing shows speed
  and an agent can see arcs, eases and stops without watching. `--onion 3` ghosts the three frames before each tile.
  Status (2026-11-01): in its own PR. The renderer records each tracked layer's anchor (`pos`, also in `probe`);
  the trails come from a pass over every frame at a quarter of the thumbnails' scale, since anchors come from the
  matrices, which on the CPU took a stress frame from 24 to 38 ms down to 4 to 8.
- `kine probe scene.json --curve title.x` samples a value and its speed at every frame, with stops and the fastest
  frame, as JSON.
- Examples: `logo.json` (an SVG logo draws on, morphs into a wordmark and glows), `kinetic.json` (stagger orders,
  `rise-mask`, `scramble`, text on a path), `explainer.json` (rough shapes, a dashed route with a dot riding it,
  masks), `confetti.json` (a burst over a counter).
- The 1.6 stress report: every Phase 0 stress scene and the four 2D scenes above, GPU and CPU, frame time and peak
  memory, as 1.5's table.
- README's 2D section and its limits (the stale shaping line fixed), `kine docs`, site/, CHANGELOG with measured
  numbers, version 1.6.0. A 2D shot in the launch video is optional and means re-measuring the promo.

## Out of scope

- Bones, puppet pins and mesh warps for 2D characters.
- Lottie or After Effects import and export.
- Expressions that read another layer's values (`layer("logo").x`). Groups, `along` with a path id and the camera's
  `target` cover the common cases; a cross-layer reference needs cycle checks and rules for enter and exit offsets.
- Physics with state: collisions, cloth, fluids.

## Risks

- Morphs between very different shapes can twist or cross themselves. The start-point turn and the contour pairing
  limit it; `kine look --at` on mid-morph frames is in the tests, and `morph-contours` warns about the case most
  likely to look wrong.
- Masks, glow and outline each add an offscreen layer. On the GPU that is cheap. On the Windows CPU wheel, which
  rasterises about 30 times slower than Linux (1.5 Phase 0), it is not: every layer is bounded to the layer's box,
  and `slow-effects` warns.
- The SVG subset can grow without end. It stops at the listed elements, and `svg-unsupported` names the rest.
- Byte identity through Phase 0. Replayed pictures may not match direct drawing on the GPU; the picture cache stays
  only if `bench/same.py` passes.
- `kine docs` is what small models read, and it is already about 670 lines. Each phase's docs stay short, and if it
  passes about 900 lines, `kine docs` splits into topics (`kine docs 2d`).
- Names that already mean something: `svg` is an alias for `path` until Phase 5, and `bounce` for `pop`
  (decision 2).

## Decisions (agreed 2026-11-01)

1. Release split as 1.5 did it: 1.6.0 ships Phases 0 to 3 and 7; Phases 4 to 6 go in if they pass their checks by
   then, otherwise in 1.6.1.
2. `"type": "svg"` (today read as `path`, with an info note) becomes the SVG layer, since today it only renames.
   `bounce` stays an alias for `pop`, and the bouncing drop is `land`.
3. Speedups that change pixels, such as blurring split-text letters with a mask filter on the paint instead of a
   layer each (fewer of `text-split`'s 950 layers a frame), are not in 1.6. Byte identity stays strict, and a
   pixel-changing speedup only arrives behind a new key.
