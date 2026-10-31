"""How long each voice backend takes to speak a line: the wall time of `kine voice --force --only <id>`, which starts
the backend's Python, loads its model (or calls its API) and writes the WAV and its word times. Each run is a new
process, as on the command line, so model loading is included; first-run downloads are not (run it once before).

    python bench/voices.py                         # every sound with text in site/demos/voices.json
    python bench/voices.py --runs 5 vo-k vo-c      # some of them, median of 5
    python bench/voices.py --scene other.json

A backend in another Python needs its KINE_VOICE_CMD_<NAME>, and ElevenLabs needs ELEVENLABS_API_KEY; every run
spends that account's characters. The numbers in site/demos/voices.json come from this script; say which machine
ran it.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--scene", default="site/demos/voices.json")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    scene = ROOT / a.scene
    spec = json.loads(scene.read_text(encoding="utf-8"))
    sounds = {s["id"]: s for s in spec.get("audio", []) if s.get("text")}
    out = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(), "voices": {}}
    for sid in a.ids or list(sounds):
        times = []
        for _ in range(a.runs):
            t0 = time.perf_counter()
            subprocess.run([sys.executable, "-m", "kinescript.cli", "voice", str(scene), "--only", sid, "--force"],
                           cwd=ROOT, check=True, capture_output=True)
            times.append(time.perf_counter() - t0)
        src = scene.parent / (sounds[sid].get("src") or f"voice/{sid}.wav")
        with wave.open(str(src)) as w:
            speech = w.getnframes() / w.getframerate()
        out["voices"][sid] = {"backend": sounds[sid].get("backend"), "median_s": round(statistics.median(times), 2),
                              "runs_s": [round(t, 2) for t in times], "speech_s": round(speech, 2)}
        print(sid, out["voices"][sid], file=sys.stderr)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
