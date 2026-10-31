#!/bin/bash
# Capture gPu(doom) gameplay for the promo: every tic as a frame, played by the autoplay bot.
#   capture.sh WORKDIR    WORKDIR holds clones of oaustegard/experiments and oaustegard/oaustegard.github.io
# Needs Node with Playwright and a Chromium that runs WebGPU on SwiftShader. Writes media/E1M?.mp4 next to this
# script's parent. The harness steps the engine tic by tic, so the footage is smooth however slow the GPU is.
set -euo pipefail
work=${1:?usage: capture.sh WORKDIR}
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../media"; mkdir -p "$out"
cd "$work/experiments/trm-doom"
for spec in "E1M1 1400" "E1M2 1000" "E1M3 1000"; do
  set -- $spec
  NODE_PATH=$(npm root -g) timeout 900 node test/run_page.cjs test/play.html 880 "map=$1&tics=$2&frames=1&global" 2>&1 \
    | grep -v '^console: {' > "$out/$1.out"
  python3 "$here/tomp4.py" "$out/$1.out" "$out/$1.mp4"
  rm "$out/$1.out"
done
