# Kinescript: video you can diff

Most video tools keep the edit inside the application. The timeline is a file only that program can read, and the
way to change it is to open the program and drag things. That works for a person at a desk. It does not work for
a script that needs to make forty variants, a reviewer who wants to see what changed, or an AI agent that cannot
see a timeline at all.

Kinescript keeps the edit in a JSON file. Every title, keyframe, cut and caption is a line of data. You render it
from the command line, and you can read the change in a pull request:

```diff
-    {"id": "intro", "type": "text", "text": "Q3 results", "dur": 2.5},
+    {"id": "intro", "type": "text", "text": "Q3 results", "dur": 4.5},
```

## Time is a reference

The part we use most is cue timing. A layer can start relative to another one:

```json
{"id": "title", "at": "intro.out+0.2"},
{"id": "cta",   "at": "title.out"}
```

Make the intro two seconds longer and the title and the call to action move with it. There is no list of absolute
timestamps to keep in sync. If a reference points at a layer that does not exist, or two layers wait on each
other, Kinescript says which ids are involved before it renders anything.

## It checks its own work

A person catches a white caption over a white sky by watching the video. An agent can't do that, so Kinescript
does part of the watching:

- `kine check` samples the scene and reports text that leaves the frame or the title-safe margin, text layers that
  overlap, text under 4.5:1 contrast against the pixels actually behind it, text too small for the resolution,
  and text that is not on screen long enough to read. Each issue comes with a time range and a suggested fix.
- `kine look` renders many frames into one labelled image, with every layer outlined and named if you ask. One
  look at that image replaces scrubbing through the timeline.
- `kine probe` returns where every layer is at a given time, in pixels.

While making the launch video, `kine check` flagged a lower third that was hard to read over bright footage
(2.0:1). The lowerThird component now has a backing plate by default, and the check comes back clean.

## Fast because frames are independent

The renderer is a function of the scene and the frame number. Nothing carries over from the previous frame, so a
render splits into chunks that run in parallel and get joined at the end without re-encoding. Drawing happens on
the GPU through Skia.

On the development machine (RTX 5060 Ti, Core Ultra 7 265), a frame of the launch video takes a median 5.2 ms to
draw, and the whole 35-second 1080p video renders with audio in about 6.4 seconds.

## What is in the box

Text with per-letter animation, typewriter reveals and emphasis markup. Shapes and SVG paths that draw on.
Footage with trims, speed changes, punch-in zooms and cross-transitions from a cut list. Captions from SRT with a
karaoke mode. SkSL shader backgrounds. Grading, vignette, grain and motion blur. Components for lower thirds,
title cards, callouts and progress bars. Output to MP4, ProRes 4444 or VP9 with alpha, GIF and PNG sequences.
And an MCP server, so an agent can call all of it and get frames back as images.

The launch video at the top of this post is itself a Kinescript scene. Its source is in the repository.

## Limits

It is 2D. Audio is mixing, not editing. Without a GPU the fallback renderer is slow on Windows. Text layout covers
left-to-right scripts without complex shaping. These are listed in the README with numbers.

## Try it

```bash
pip install kinescript
kine new scene.json
kine check scene.json && kine look scene.json && kine render scene.json
```
