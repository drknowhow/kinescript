# Kinescript 1.5: 3D plan

Proposal, 2026-10-31. Released as 1.5.0 on 2026-11-01 with Phases 0 to 4 and 7 (#57 to #73); Phases 5 and 6
move to 1.5.1 (decision 3). Each phase's status says what was built and how it differs from its bullets. Numbers
marked "target" are budgets to measure against; the others come from the 1.4 CHANGELOG or from the spike below.

## Where 3D stands in 1.4

- Depth on any layer (`z`, `rotateX`, `rotateY`, `backface`, `fixed`), a top-level `camera` that pans, dollies
  and orbits around the canvas centre, and groups with `"sort": "depth"` (`space.py`, `Renderer._cameras`,
  `Renderer._layer`).
- `model` layers: six primitives, extruded text and SVG paths, `parts`, and glTF and OBJ files (`mesh.py`,
  `models.py`). `render3d.draw` sorts each model's triangles far to near and draws them with `drawVertices` into a
  frame-sized offscreen at 2x supersampling. The GPU shades per pixel in SkSL, with the normal and the material
  slot packed into the two texture coordinates; the CPU shades per vertex in numpy.
- One light rig, fixed to the camera: key, fill, ambient and rim from one of nine directions. It does not animate:
  `Scene.light` is the raw value, where `camera` is compiled like a layer.
- `ground`: a blurred footprint straight below the model.

README.md:267 lists the limits users run into: no image textures, no skeletal animation, no shadows cast on other
objects, separate model layers that don't hide each other per pixel, and no Draco or meshopt files. The code adds
three more: lights can't move, `parts` can't animate, and 3D text isn't shaped (`kine check` warns
`unshaped-3d-text`).

## What the spike settled

A short test against skia-python 144.1 on the raster backend found:

- A `kRGBA_F32` surface with `BlendMode.kDarken` keeps the minimum per pixel. Triangles drawn with their depth in
  the texture coordinate and a one-line SkSL shader leave the nearest depth at each pixel, in any draw order.
- A runtime shader can't read `sk_FragCoord` (SkSL reports "unknown identifier"), and `drawVertices` gives the
  shader one pair of texture coordinates. Vertex colours go to the blend mode, not to the shader.
- A canvas-sized `drawPaint` with image children works, and there the shader's coordinate is the device pixel.
- `TextBlob.Iter` and `Font.getPath` exist, so shaped glyph runs can be turned into outlines.

So a depth test inside one `drawVertices` call isn't possible with public SkSL, and triangles inside one model
keep the depth sort. Per-pixel visibility between layers is possible: each layer draws a colour image and a depth
image, and a compositing pass keeps the nearer pixel. 1.5 is built on that, with no new required dependency.

Phase 0 then measured the GPU side (RTX 5060 Ti, skia-python 144.1), below.

## Rules every change keeps

- A frame is a pure function of (scene, frame number). Skinned poses, glTF animations, camera shake and repeat
  expressions are sampled at the frame's time, and caches key on values, as `render3d._kept` does.
  `tests/test_core.py::test_frame_is_pure_function_of_index` gets a case for each new feature.
- Scenes that use none of the new keys draw the same bytes as 1.4. `bench/same.py` draws five frames each of
  `examples/3d.json`, `model.json`, `title3d.json`, `import.json` and `hello.json` at the branch and at `main`, GPU
  and CPU, and every PR runs it.
- No new required dependency. Draco is the one candidate, as an optional extra (decision 2).
- Every feature works on both backends. Where the CPU takes a cheaper path, such as per-vertex instead of
  per-pixel, `kine docs` says so, as it does for shading now.
- Every new key is validated in `schema.py` with errors that name the layer and field, documented in `docs.py`
  for small models, and gets a `kine check` rule for its most likely mistake.
- Before a visual change is called done: `kine check`, then `kine look --boxes`, and look at the image. Numbers in
  the CHANGELOG, README and site are measured.

## Phases

Each bullet is about one PR. Phases 2 and 3 can run in parallel once Phase 1 is in.

### Phase 0: GPU spike (done, 2026-10-31)

- Float targets: the GL context makes F16 render targets but not F32. F16 keeps 1.2345678 as 1.234375, about
  1.5 px of depth at z 0, too coarse for surfaces that touch. Depth is packed into 24 bits of an RGBA8 image by an
  SkSL shader instead: the largest error over a ramp was 6.5e-8 of the range, 0.0001 px over a 2000 px deep
  stage. Each model writes its depth in its own sort order with `kSrc`. The CPU writes float32 through a linear
  gradient shader into an F32 raster surface (error 5.9e-8).
- Compositing on the GPU at 3840x2160 (1080p at 2x): 0.21 to 0.49 ms a layer for 2 to 16 layers. At 7680x4320
  (4K at 2x) it took 28 ms a layer, but other processes held 14.8 of the card's 16.3 GB at the time, so that
  number needs a quiet GPU.
- Compositing on the CPU at 3840x2160: numpy over the whole frame took 469 ms a layer in float32 and 60 ms with
  uint32 pixel selects; limited to where two layers overlap (8% of the frame in the test), 4.8 ms.
- Edges: rendering already supersamples 2x without MSAA, and a model's colour and depth come from the same
  triangles at that size, so their coverage matches and the downsample antialiases the merged result. No depth
  growing was needed.
- Found on the way: the Windows skia-python 144.1 wheel rasterises about 30 times slower than the Linux one on the
  same machine (a translucent 1080p rect: 55.6 ms against 1.6 in Docker; an AA circle: 13.9 against 0.48). Only an
  unscaled image drawn onto a BGRA8888 surface skips the slow path (1.4 ms against 58.7 into RGBA8888). Moving
  Kinescript's CPU surfaces to BGRA sped up CPU frames 1.0 to 2.1 times on Windows (model.json the most), shifted
  pixels by up to 3 levels, and on Linux made translucent fills twice as slow, so it was dropped.

### Phase 1: per-pixel stage

- `"sort": "pixel"` on a group makes it a stage. Each child draws a colour image and a depth image. Models take
  depth from the minimum pass. Flat layers take it from their quad: 1/w is linear across a plane on screen, so
  four corners give exact depth. The stage merges children into two accumulators one at a time, so memory doesn't
  grow with the number of children; one F16 surface at 4K with 2x supersampling is 265 MB.
- Models that pass through each other, and cards that pass through models, now meet at a clean line. Inside one
  model the triangle sort stays, so the docs' advice changes from "put touching things in one model as parts" to
  "put things that cross in a stage as separate layers".
- Check rules: `stage-cost` (more than 16 children), `coplanar-in-stage` (two surfaces at the same depth flicker).
- `examples/stage.json`: two interlocking rings, a sphere passing through a card, a title in depth.
- `stage-16` stress scene in `bench/stress.py`.

Status (2026-10-31): merged (#57). Changes from the bullets above:
- The GPU keeps depth in 24 bits of RGBA8 (no F32 targets), so a stage holds four frame-sized RGBA8 images at 2x,
  33 MB each at 1080p. The merge runs four passes, none drawing into an image it reads: drawing into a surface with
  a live snapshot makes Skia copy it, and with those copies 6 workers rendered `examples/stage.json` in 5.8 s
  instead of 1.9 (14.2 GB of the card's 16.3 GB held by other programs).
- `coplanar-in-stage` is not built: ties go to the later child, so equal depths don't flicker between frames.
- `examples/stage.json` is a floor, a turning panel, a ring through it and a ball with a ground shadow.
- `stage-16` in `bench/stress.py`: 120 frames in 2.88 s (42 fps) against models-30's 49 fps.
- CPU stages are slow on Windows: `examples/stage.json` takes 22.7 s against 6.6 with the group as `"order"`. The
  Windows wheel's slow rasteriser (see Phase 0) draws the merged stage back at 1x in about 68 ms a frame.

### Phase 2: light, shadow, fog

- `light` compiles like `camera`, so every amount tweens, and `from` also takes `azimuth` and `elevation` in
  degrees. A light sweeping across a 3D logo is one keyframed key.
- `"follow": "camera"` (the default, as now) or `"world"`. With world lights, orbiting the camera around a product
  changes its shading, as on a real turntable.
- `lights`: up to three more, each `directional` (`from`) or `point` (`x`, `y`, `z` in scene px, with falloff),
  with `color` and `intensity`. The SkSL effect gets uniforms per light; the CPU shades them per vertex.
- `env`: reflections on shiny and metal materials from a procedural environment, `studio`, `sunset`, `night`,
  `chrome` or `{"top", "horizon", "bottom"}` colours, looked up by the reflected normal.
- Material `glow` (0..1): emissive colour that light doesn't change.
- `ground` takes `"from": "light"` to project the footprint along the key light instead of straight down, so the
  shadow moves with the light. The default stays straight down, with the same bytes.
- `"shadows": true` on a stage: the stage draws a depth image of its models from the key light (orthographic),
  and the compositing pass darkens pixels the light doesn't reach, on models and on flat layers. A rect turned 90
  degrees on `rotateX` becomes a floor that takes shadows.
- `fog` at the top level: `{"color", "near", "far"}` in px from the camera. Flat layers are tinted as a whole;
  models per pixel on the GPU and per vertex on the CPU; inside a stage, per pixel from the depth image.
- Check rules: `light-behind` (the key light sits behind what the camera sees, so a model shows only ambient),
  `fog-hides` (a layer lies wholly past `far`).

Status (2026-10-31): merged as four PRs: lights (#58), `env` (#59), `fog` (#60) and shadows (#61). Lights:
`light` tweens, `azimuth`, `elevation`, `follow`, `lights` (directional and point), material `glow` and
`light-behind`. Changes from the bullets above:
- The GPU shader gets no fragment position, so a point light lights a whole model from where its centre is.
- `glow` is a floor, max(lit, fill x glow), not an addition: added to a lit surface it washed the colour to white.
- glTF `emissiveFactor` is not read yet.
`env` (branch `env-reflections`): built. The reflected view assumes the camera looks straight down -z at every
pixel (the shader has no position), and roughness is a blend toward the environment's mean colour as shine falls.
`fog` (branch `fog`): built, with `amount` and defaults (the canvas bg colour, near at z 0's distance, far 4
times it). Changes from the bullet above: on the GPU a model's fog is per vertex too, carried in vertex colours and
blended with `kDstATop` over the shading, since the shader has no position. A stage fogs at 1x on the GPU (one pass
reading the merged colour and depth) and at 2x on the CPU, in float32 over the pixels that hold something.
Shadows (branch `shadows`): `ground` `"from": "light"` and stage `shadows` are built. Changes from the bullets
above: the map holds the faces turned from the light (second-depth shadow mapping), and a pixel whose surface turns
from the light, judged from its neighbours' depths, takes no shadow; together they keep surfaces from shading
themselves in stripes. Flat layers receive but do not cast. `shadows` takes `opacity` and `soft`, which do not
animate yet. That completes Phase 2.

### Phase 3: camera

- `target`: a point `[x, y, z]` or a layer id. `rotateX` and `rotateY` orbit the target instead of the canvas
  centre; a layer id makes the camera follow that layer.
- `shake`: `{"amount": px, "speed": hz, "seed": n}`, smooth noise of the frame's time, for handheld moves.
- Depth of field: `focus` (z in px, or a layer id) and `aperture` (px of blur per 1000 px from the focus plane).
  Flat layers blur whole, by their depth, through the existing `blur`. In a stage the blur is per pixel: four
  blurred copies and a per-pixel choice by depth. A canvas-sized blur took 370 ms at 1080p on the CPU in the 1.4
  split-text work, so `kine check` warns when a CPU render uses per-pixel depth of field.
- `kine docs` recipes for orbit, crane, push-in and dolly zoom. Dolly zoom needs no new key: `fov` already keeps
  the size at z 0, so animating `fov` alone does it.
- Check rules: `camera-inside-model`, `focus-on-nothing` (the focus plane is nearer or farther than every layer).

Status (2026-10-31): camera moves (#62, merged): `target`, `shake`, the `kine docs` recipes and
`camera-inside-model`. Depth of field (#63): `aperture`, `focus`, `focus-on-nothing` and
`slow-depth-of-field`; a stage blurs by the render's scale since #73. Changes from the bullets above:
x and y still pan from the target. A layer target follows the layer's keys and its parents', not its enter and exit
motions, and keyframes pan between points and layer ids. `shake` turns the camera on its own spot, so near and far
layers move alike, with a roll that moves the corners half as far; `amount` animates, `speed` and `seed` do not.
`camera-inside-model` counts crossings of three rays from the camera, so a camera in a ring's hole is outside it.
The blur is a lens's, `aperture` / 1000 x |D - Df| x Df / D for distances D from the camera and Df to the focus
plane, at most 48 px: in proportion near the plane as planned, but bounded far behind it. `focus` as a z is
measured from the target. A stage mixes the sharp frame and four copies (1/8, 1/4, 1/2 and all of its most blur),
each copy holding only what is at least half as blurred as it; a plain mix let a sharp layer glow onto a blurred
one in front of it. On the CPU the stage blurs at halved sizes and mixes in numpy (an SkSL mix took 900 ms at
1080p); a 1080p frame of `examples/focus.json` takes 3.1 s against 0.58 s without depth of field, 21 ms on the GPU.

### Phase 4: textures and surface looks

- Texture coordinates: every primitive makes them (cube per face, sphere and cylinder wrapped, plane 0..1,
  extruded text and paths planar on the caps), and glTF `TEXCOORD_0` is read.
- `texture` on a model or part: an image or video file, or `"#groupId"` to put a rendered group on the surface,
  such as an animated UI on a phone's screen or footage on a curved panel. `repeat` and `offset` adjust it. The
  group renders at the frame's own time, so frames stay pure.
- glTF `baseColorTexture` is drawn as a texture instead of being averaged to one colour; `emissiveTexture` drives
  `glow`.
- Drawing: the texture coordinates now carry the texture, so they can't carry the normal. Textured models draw
  diffuse light, specular and texture as separate passes of the same sorted triangles and combine them (the GPU
  per pixel; the CPU per-vertex light times a per-pixel texture, which raster Skia samples quickly). Skia maps an
  image affinely across each triangle, so textured triangles are split by their size on screen to stop the image
  sliding.
- `edges`: `{"width", "color", "angle"}` draws creases sharper than `angle` and silhouettes, for technical and
  toon looks; `"edges": "all"` is a wireframe. Edge quads join the sorted triangle list just in front of their
  faces, so hidden lines stay hidden without a depth buffer.
- `opacity` on models, drawn in sort order. Order-independent transparency stays out of scope.
- Check rules: `no-uvs` (a texture on a mesh without coordinates), `texture-missing`, `texture-swim` (a textured
  triangle still too large on screen).

Status (2026-11-01): image textures merged (#64): coordinates on every shape, `texture` with `repeat` and
`offset` on layers and parts, glTF `TEXCOORD_n` and `baseColorTexture`, the passes, the cutting, `no-uvs` and
`texture-swim`. Video and `"#groupId"` textures merged (#65). Emission merged (#66). `edges` merged (#67): corners at the same place are welded into points, so each edge is found once with the face on
either side; a face drawn back to back counts once. Each frame, an edge whose faces turn by more than `angle`, or
where a face toward the camera meets one turned away or none, gets a quad `width` px wide with square ends. Its four
vertices join the sorted triangles after every face at either end (by each face's nearest corner on a textured
model, whose triangles are cut later), in an extra material slot drawn in the line's colour, unlit. A line taken
from a face toward the camera moves with it when the model explodes. `"edges": true` uses 2 px, theme `ink` and
30 degrees; `"all"` is angle 0. In a stage the quads write their depth too. See-through models merged (#68) as a
material key `transparency` (layer, part, and glTF `alphaMode` `BLEND` or `.mtl` `d` and `Tr`) instead of the
layer's `opacity`: `opacity` and the enter and exit fades fade every layer as a flat image, and keyframed fades of
models would have started showing their far side. A see-through face turned away is drawn too, lit from inside,
unless it is the back of a face drawn back to back; the colour is premultiplied by the opacity in the shader (GPU)
or the vertex colour's alpha (CPU), and the sorted triangles blend far to near. A textured model draws solid
(`ignored-key`): its passes are combined per pixel, nearest face only. Emission:
glTF `emissiveFactor` (with `KHR_materials_emissive_strength`) and `emissiveTexture`, and `.mtl` `Ke` and
`map_Ke` are an emission colour per material, added in the shader (GPU) or per vertex (CPU) of an untextured
model, and drawn as one more pass of a textured one. It is not a layer key: `glow` stays the way to light a
model of your own. Video and group textures: a video texture takes `loop`, `trim` and
`speed` as a video layer does and plays on the model's clock. A group named by a texture is drawn only on the
model, its box (its `w` and `h`, else the canvas) filling the image, its layers as in a scene without a camera
(those that move in depth seen by a camera of the image's size, with no fog or depth of field). `kine check`
adds `unknown-texture` and `bad-texture` (a layer that is not a group, or a group the model is in). Each
material's picture is now a small runtime shader: skia-python's `Shaders.Blend` took 99 ms a call with a 1080p
video frame on the GPU, and the same bytes come out for image textures. Changes from the bullets above: a missing or unreadable texture is `media-missing` or
`bad-media`, as for other files, not `texture-missing`. `.obj` files' `vt` and `map_Kd` are read too.
The pictures of a model's materials are drawn in one call a pass, a shader picking each material's by an offset
on its texture coordinates (#71): one call a run of a material in the sort
took 48 s a frame on a glTF can whose label and steel alternate in depth.
`"texture": false` draws the fill (on a part, instead of the layer's texture; on a model file, its mean colour).
Cylinder and cone ends take the image flat. On the GPU the picture, the light on a white surface and the added
light (highlights, rim, reflection) are drawn on three surfaces and combined per pixel; on the CPU the picture is
drawn times its per-vertex light at half strength and doubled when it is composited, which saved two copies of the
region. Triangles are cut on shared edges until the image is within 1 px of where perspective puts it, for up to
nine rounds: a floor reaching to just in front of the camera took eight.

### Phase 5: motion from files, parts and repeats

- glTF animation: node translation, rotation and scale channels (LINEAR, STEP, CUBICSPLINE), morph target weights,
  and skins (`JOINTS_0`, `WEIGHTS_0`, inverse bind matrices; linear blend skinning in numpy). The layer key
  `animation` takes a clip name or `{"name", "speed", "loop", "start"}` on the layer's clock.
  Built on branch `gltf-animation`, with the probe, check and make_media bullets below: `models.pose` sets the
  clip's channels on the node tree, blends morph targets and skins on the file's own primitives, then replays the
  edge cuts `_refine` made when the file was read (it now records them), so the posed mesh has the same triangles
  and sorts the same way. Skins and default morph weights apply at rest too. The model keeps its rest size and
  centre. Posing `arm.glb` (1,314 vertices) takes 0.36 ms; `bench/same.py` finds no frame that differs.
- `parts` animate: part keys take keyframes and expressions like layer keys. The merged mesh is rebuilt only on
  frames where a part moves.
- `repeat`: `{"count": n, "x": "...", "y": "...", "z": "...", "rotateY": "...", "scale": "...", "fill": "..."}`,
  with expressions over `i`, `n` and `t` in the existing AST-checked expression engine. The copies merge into one
  mesh and sort together, so a waving grid of 400 cubes is one layer.
- `kine probe` on a model file lists its animations with durations, skins, morph targets, textures and texture
  coordinate sets.
- Check rules: `unknown-animation` (naming the clips the file has), `skin-influences` (a vertex with more than
  four joints; the extra weights are dropped).
- `examples/make_media.py` writes a skinned, animated glTF (a tube bending on three joints) and a textured one, so
  tests and examples need no downloaded assets.

### Phase 6: shapes, 3D text, compressed files

- `cube` takes `radius` for a rounded box. New shapes: `capsule`; `lathe`, where `profile` (`[[r, y], ...]` or
  path data) turns around the vertical axis; and `tube`, a round tube along path data `d`, where `draw` grows it
  along its length for 3D lines and routes.
- Shaped 3D text: glyph runs from `TextBlob.MakeFromShapedText` (HarfBuzz, which 2D text already uses), read with
  `TextBlob.Iter`, outlines from `Font.getPath`. Arabic letters join, marks sit on their letters and right-to-left
  runs read right to left. `unshaped-3d-text` narrows to colour emoji, which have no outline to extrude.
- `EXT_meshopt_compression` decoded in numpy. Draco per decision 2.

### Phase 7: tooling and release

- `kine look --depth` shows a stage's depth image; `kine look --wire` draws triangle edges over the frame.
  Merged (#69): the renderer keeps each model's drawn triangle edges and each stage's merged depth (the nearest of
  a frame pixel's samples) while `look` asks for them.
- Stress scenes `stage-16`, `textured-100k`, `skinned`, `repeat-1000` and `dof-stage`, each with frame time and
  peak memory in the 1.5 report.
  Merged (#70): `textured-100k` (an .obj globe of 102,400 triangles with normals,
  coordinates and a map_Kd image, turning under world light and `env`) and `dof-stage` (a stage of 10 models and
  3 cards on a floor with shadows, focus pulled from the near row to the far row to the cards). `skinned` and
  `repeat-1000` wait for Phase 5. Results in `docs/stress-test-1.5/`, 120 frames at 1080p each, render wall time
  and peak memory of the render processes (development machine, 2026-10-31):

  | scene | GPU, 4 workers | CPU, 2 workers |
  |---|---|---|
  | `models-30` | 2.60 s, 46 fps, 737 MB | 12.5 s, 10 fps, 294 MB |
  | `stage-16` | 2.90 s, 41 fps, 736 MB | 42.2 s, 3 fps, 718 MB |
  | `textured-100k` | 3.15 s, 38 fps, 999 MB | 41.9 s, 3 fps, 600 MB |
  | `dof-stage` | 2.66 s, 45 fps, 1000 MB | 209.5 s, 0.6 fps, 1883 MB |

  `kine check` of `dof-stage` takes 1.0 s on the GPU and 49.8 s on the CPU, where it warns
  `slow-depth-of-field`.
- Examples: `stage.json`, `product.json` (a textured glTF turntable under world lights with `env`, depth of field
  and a floor that takes its shadow), `character.json`, `cubes.json`.
  `product.json` merged (#72): `examples/make_media.py` writes `can.glb`, a can
  whose body takes `label.png` as an embedded base colour texture, under a steel lid, rim and ring pull. It turns on
  a table in a stage with shadows and depth of field, two more cans out of focus behind it. 180 frames render in
  16.0 s on the GPU (6 workers). `character.json` and `cubes.json` wait for Phase 5.
- README 3D section and its limitations line, `kine docs`, site/, CHANGELOG with measured numbers, version 1.5.0.
  A 3D shot in the launch video is optional and means re-measuring the promo.
  Built on branch `release-1.5.0`: the README's model files and limits, the site's 3D section rewritten for 1.5
  with a product turntable demo, version 1.5.0. The launch video is unchanged, so its numbers stand.

## Out of scope

- A depth test inside one model, which the SkSL limits above rule out.
- Ray tracing, global illumination, ambient occlusion, physics, HDR image lighting, order-independent
  transparency.
- A second GL pipeline (moderngl or PyOpenGL). It would give a depth buffer and shadow maps directly, but it adds
  a dependency and a second renderer that the CPU path would have to match.

## Risks

- Stage cost grows with children and resolution. The two accumulators, `stage-cost` and the Phase 0 gate bound it.
- CPU time for per-pixel fog, shadows and depth of field: 1080p at 2x is 8.3 million pixels a pass in numpy.
  Phase 0 measures it; a feature that misses its budget gets a CPU warning in `kine check`.
- Textures sliding under affine mapping, and the triangles that splitting adds.
- Byte identity for 1.4 scenes. The hash test fails on the first PR that breaks it.

## Decisions (agreed 2026-10-31)

1. Per-pixel visibility only inside `"sort": "pixel"` groups, so 1.4 scenes keep their bytes.
2. Draco as an optional extra, `pip install kinescript[draco]` (DracoPy), keeping today's error that names the file
   when the extra is missing.
3. 1.5.0 ships with Phases 0 to 4 and 7. Phases 5 and 6 go in if they pass their checks by then, otherwise in
   1.5.1.

Byte identity is checked with `bench/same.py`, which draws sampled frames at this tree and at a git revision on
both backends, instead of committed frame hashes: CI renders on Linux's CPU backend, whose bytes differ from a
Windows machine's.
