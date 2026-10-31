# gPu(doom) promo

A 36-second reel for [gPu(doom)](https://austegard.com/games/doom/), the DOOM engine rewritten as WebGPU
compute shaders. It is the first outside test of Kinescript: a scene written by an agent against `kine docs`, checked
with `kine check` and `kine look --boxes`, and rendered on the CPU backend in a Linux container with no GPU.

- `promo.json`: the scene. A title, seven gameplay clips over a blurred copy of themselves, one caption per clip,
  and an end card.
- `stf*.png`: Freedoom's status-bar faces, read from `e1.wad` and scaled 10× with nearest-neighbour.
- `capture/`: how the footage was made. `capture.sh` runs the shipped engine headless with the autoplay bot from
  `oaustegard/experiments` (`trm-doom/test/play.html`), which returns every tic's 320×200 frame. `tomp4.py` turns
  them into 35 fps H.264 at 1600×1200, five times wide and six times tall, which is DOOM's 4:3 with whole pixels.
- `media/` is gitignored. Run `capture/capture.sh WORKDIR` to make it.

```bash
kine check examples/gpu-doom/promo.json
kine render examples/gpu-doom/promo.json -o gpu-doom-promo.mp4 -q high
```

The footage is the bot playing with its BFS planner, not the TRM model; the last caption says it is a bot.
The faces and footage are Freedoom art, © 2001-2024 Contributors to the Freedoom project. Its licence, with the
notice, conditions and disclaimer that go with every copy, is in [`FREEDOOM-COPYING.txt`](FREEDOOM-COPYING.txt).
