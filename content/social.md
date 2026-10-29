# Social copy

Attach `docs/promo.mp4` (or `docs/hero.gif` where video is not supported) to the first post.

## Launch, short (Bluesky / X)

Kinescript: video you can diff.

Motion graphics and video edits as JSON. Times reference other layers ("at": "title.out+0.2"), a linter checks
contrast and reading time, and a 34 s 1080p video renders in 7 s on a GPU.

This video is a Kinescript scene.

## Launch, longer (LinkedIn)

I built a video renderer that an AI agent can use properly.

Kinescript scenes are JSON. Every title, keyframe and cut is a line you can diff and review. Start times can point
at other layers, so lengthening one section moves everything after it.

Agents can't watch video, so the tool checks for them: `kine check` reports text off the frame, overlapping text,
low contrast against the real background and captions too short to read, each with a time range. `kine look` puts
many frames into one image.

The renderer draws each frame independently on the GPU and splits the work across processes. The 34-second launch
video renders in about 7 seconds, and its source is in the repo.

## Follow-ups (one per post)

1. Cue timing. Show the diff of one `dur` change and the before/after contact sheets.
2. `kine check` catching a 2.0:1 lower third over bright footage, and the fix.
3. A karaoke caption pass over a talking-head clip from an SRT file.
4. `kine look --boxes`: what an agent sees when it reviews a scene.
5. Shader backgrounds in three lines of JSON: `{"type": "shader", "preset": "aurora"}`.
