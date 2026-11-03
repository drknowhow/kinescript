"""Screenshots of `kine edit` for the README and the landing page, taken in headless Chrome.

    python examples/make_media.py              # once: the narrated example reads its voice and music
    python marketing/build_editor_shots.py     # docs/editor.jpg, site/media/editor.jpg, site/media/editor-shapes.jpg

Each shot opens a real editor on an example scene at a chosen time with a layer selected, as a person would see it.
It needs Chrome or Chromium: set KINE_CHROME to its path if it is not found.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import skia

ROOT = Path(__file__).resolve().parents[1]
SHOTS = [  # scene, time, layer to select, output
    ("examples/narrated.json", 1.7, "check", ROOT / "docs" / "editor.jpg"),
    ("examples/narrated.json", 1.7, "check", ROOT / "site" / "media" / "editor.jpg"),
    ("examples/explainer.json", 3.0, "route", ROOT / "site" / "media" / "editor-shapes.jpg"),
]


def chrome() -> str:
    for c in (os.environ.get("KINE_CHROME"), shutil.which("google-chrome"), shutil.which("chromium"),
              shutil.which("chrome"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if c and Path(c).is_file():
            return c
    sys.exit("no Chrome found: set KINE_CHROME")


def serve(scene: str):
    p = subprocess.Popen([sys.executable, "-m", "kinescript.cli", "edit", str(ROOT / scene), "--port", "0", "--no-open"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    line = p.stdout.readline()
    m = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
    if not m:
        p.kill()
        sys.exit(f"the editor did not start: {line}")
    return p, int(m[1])


def main() -> None:
    exe = chrome()
    tmp = Path(tempfile.mkdtemp())
    for scene, t, layer, out in SHOTS:
        proc, port = serve(scene)
        try:
            png = tmp / "shot.png"
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,1000",
                            "--virtual-time-budget=9000", f"--screenshot={png}",
                            f"http://127.0.0.1:{port}/?t={t}&layer={layer}"], check=True, capture_output=True, timeout=120)
        finally:
            proc.kill()
        img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(png.read_bytes()))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(bytes(img.encodeToData(skia.kJPEG, 86)))
        print(out.relative_to(ROOT), out.stat().st_size // 1024, "KB")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
