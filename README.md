<p align="center"><img src="brand/logo-dark.png" width="460" alt="kinescript"></p>

<p align="center"><b>Video you can diff.</b><br>Motion graphics and video edits written as JSON, checked by agents, rendered on the GPU.</p>

<p align="center"><img src="docs/hero.gif" width="720" alt="The Kinescript launch video, rendered by Kinescript"></p>

Kinescript turns a JSON scene into video. Every layer, keyframe and cut is a line of data, so a scene can be
generated, reviewed in a pull request, and changed by a script or an agent. The renderer is a pure function of
the frame number: any frame renders on its own, which makes previews instant and lets a render split across
processes.

The launch video in `docs/promo.mp4` is a Kinescript scene (`marketing/promo.json`). It is 35 seconds of 1080p
and renders, with audio, in about 6.4 seconds on the development machine.

## What it does

- **Cue timing.** Start and end times can name other layers: `"at": "title.out+0.2"`. Lengthen one layer and
  everything cued after it moves. Cycles and missing references are reported with the ids involved.
- **Animate anything.** Numbers and colours take constants, keyframes (`[[0, 0], [0.6, 1, "outBack"]]`, with
  negative times counted from the layer's end), a tween object, or an expression such as `"=540 + sin(t*3)*40"`.
  Colours interpolate in OKLab.
- **Text that moves per letter.** `"split": "chars"` with a staggered enter motion, typewriter reveal with a caret,
  `**emphasis**` markup, per-unit expressions (`"unitY": "=sin(T*4 + i)*12"` for a wave).
- **Footage.** Video layers with trim, speed, loop, cover/contain fit, punch-in `zoom` around a `focus` point, and
  rounded picture-in-picture. A `sequence` turns a cut list into clips with fade, wipe, slide, zoom or blur
  transitions. Clip audio and music beds are mixed automatically.
- **Captions** from `.srt`/`.vtt` or inline cues, with karaoke and pop modes.
- **Depth.** Any layer takes `z`, `rotateX` and `rotateY`; a scene `camera` pans, dollies and orbits with real
  perspective; `flip`, `flip-up`, `tumble` and `push` motions; two-sided cards with `"backface": "hidden"`.
- **3D models.** Lit cubes, spheres, cylinders, cones, tori and planes, or several merged as `parts`, with
  shine, metal, rim light and toon bands; light presets; `explode` and a `build` motion.
- **Shaders.** SkSL layers with `mesh`, `aurora`, `grid` and `score` presets, or your own source.
- **Finishing.** Per-layer blur, shadow, blend modes and colour adjustments; a scene grade with vignette and film
  grain; optional motion blur.
- **Components.** `lowerThird`, `titleCard`, `callout`, `focus` (dim everything but a region), `progress`,
  `kinetic`, plus your own with `{{param}}` placeholders.
- **Outputs.** H.264 `.mp4`, ProRes 4444 `.mov` and VP9 `.webm` with alpha for compositing, `.gif`, PNG sequences.

## Built for agents

An agent cannot watch a video, so Kinescript gives it things it can read:

| Command | What the agent gets |
|---|---|
| `kine docs` | the full scene reference, short enough to keep in context |
| `kine check scene.json` | JSON issues with layer ids and time ranges: unknown keys (with the intended name first), bad values, missing media and out-of-range trims, text off canvas or in the safe margin, overlapping text, contrast under 3:1 against what is actually behind it, tiny text, too little reading time. Every problem is listed at once. Common names from other tools (`color`, `fontSize`, `duration`, `position`) are read and reported as info notes. |
| `kine look scene.json --boxes` | one PNG of many labelled frames, with every layer outlined and named |
| `kine probe scene.json --at 3.2` | resolved in/out of every layer and each visible layer's screen box |
| `kine still` / `kine render` | a frame, a range (`--from/--to`), a draft (`-q draft --scale 0.5`) or the final file |

The same tools are available over MCP (`kine mcp`), and `kine_look` / `kine_still` return images inline.

```json
{
  "mcpServers": {
    "kinescript": {"command": "kine", "args": ["mcp"]}
  }
}
```

## Install

Requires Python 3.10+ and ffmpeg on `PATH`.

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
    {"id": "dot", "type": "circle", "x": "50%", "y": "50%", "r": 26, "fill": "$accent", "dur": 1.1,
     "enter": {"name": "pop", "dur": 0.4}, "exit": "fade"},
    {"id": "title", "type": "text", "text": "Motion, **written.**", "x": "50%", "y": "46%", "size": 140,
     "weight": 700, "split": "chars", "at": "dot.out-0.1", "dur": 3.2,
     "enter": {"name": "snap", "stagger": 0.03}, "exit": "fade"},
    {"id": "sub", "type": "text", "text": "scenes as data", "x": "50%", "y": "62%", "size": 40,
     "fill": "$muted", "at": "title.in+0.6", "until": "title.out", "enter": "rise"}
  ]
}
```

More in `examples/`: `hello.json` (title animation) and `edit.json` (a cut list with a punch-in, crossfade,
lower third, callout, karaoke captions and a music bed). Run `python examples/make_media.py` first to generate the
sample footage.

## Performance

Measured on the development machine (Intel Core Ultra 7 265, 20 threads, NVIDIA RTX 5060 Ti, Windows 11):

| Workload | Result |
|---|---|
| Drawing one 1080p frame of the launch video (GPU, including readback) | median 5.2 ms, p95 6.5 ms |
| Launch video, 34.6 s at 1080p30, `-q high`, with audio | 6.4 s wall, about 160 frames/s (median of 5 runs) |
| `examples/edit.json`, 10.4 s with two decoded video clips | 6.1 s wall, 51 frames/s |

Rendering uses Skia on the GPU through a hidden OpenGL context, one per worker process. Frames are split into
chunks, encoded in parallel and joined without re-encoding.

## Limits

- Without a usable GPU, Kinescript falls back to Skia's CPU rasteriser. With the Windows `skia-python` wheel this is
  very slow (about 0.65 s per 1080p frame for the launch video); use `--scale 0.5` for previews on such machines.
  `KINE_BACKEND=cpu` forces this path.
- 2D only. Audio is mixing (trim, gain, fades, speed), not editing or effects.
- Video is decoded and resampled by ffmpeg to the scene frame rate. Jumping backwards or far ahead reopens the
  decoder, which is slower than sequential playback.
- Text layout is single-script left-to-right with greedy wrapping. No shaping for complex scripts or bidi.
- Expressions are limited to arithmetic and the listed helpers; attribute access and other names are rejected.

## Project layout

```
kinescript/   engine: timing, props, presets, render, output, verify (check/look/probe), mcp, cli
examples/     scenes and the generator for sample media
marketing/    the launch video, as a scene and its builder script
brand/        logo files, generated art and the brand guide
content/      launch post and social copy
site/         landing page
tests/        pytest suite
```

## License

MIT, see `LICENSE`. The bundled fonts are under the SIL Open Font License 1.1; their notices are in
`kinescript/fonts/`.
