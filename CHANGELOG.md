# Changelog

## Unreleased

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
