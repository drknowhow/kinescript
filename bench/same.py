"""Do frames come out as the same bytes as at another revision? Draws the same frames of each scene with this tree
and with a git revision (checked out in a temporary worktree), on the GPU and on the CPU, and compares them.

    python bench/same.py                          # the 3D examples and hello.json, against main
    python bench/same.py --rev 050f36b a.json     # other scenes, another revision

Exits 1 when any frame differs, and says by how many levels and in how many pixels."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["examples/3d.json", "examples/model.json", "examples/title3d.json", "examples/import.json",
          "examples/hello.json"]
SAMPLES = 5  # frames a scene, spread over its length

# Run by each tree's own Python path: writes every sampled frame of every scene to one .npz.
DRAW = """
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
import kinescript as k
from kinescript.render import Renderer
assert Path(k.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), k.__file__
out, frames = {}, int(sys.argv[3])
for path in sys.argv[4:]:
    s = k.load(path)
    r = Renderer(s)
    for n in sorted({round(i * (s.frames - 1) / max(1, frames - 1)) for i in range(frames)}):
        out[f"{path}@{n}"] = r.frame(n).copy()
    s.close()
np.savez(sys.argv[2], **out)
"""


def draw(tree: Path, scenes: list[str], backend: str, dest: Path) -> dict:
    env = dict(os.environ, KINE_BACKEND=backend)
    subprocess.run([sys.executable, "-c", DRAW, str(tree), str(dest), str(SAMPLES), *scenes], check=True, env=env,
                   cwd=ROOT)
    with np.load(dest) as z:
        return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", default=SCENES)
    ap.add_argument("--rev", default="main", help="git revision to compare with (default main)")
    ap.add_argument("--backend", choices=("gpu", "cpu", "both"), default="both")
    a = ap.parse_args()
    scenes = [str(Path(s).resolve()) for s in a.scenes]
    backends = ("gpu", "cpu") if a.backend == "both" else (a.backend,)
    differ = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        subprocess.run(["git", "worktree", "add", "--detach", str(base), a.rev], check=True, cwd=ROOT,
                       capture_output=True)
        try:
            for be in backends:
                old = draw(base, scenes, be, Path(tmp) / f"old-{be}.npz")
                new = draw(ROOT, scenes, be, Path(tmp) / f"new-{be}.npz")
                for key in old:
                    d = np.abs(old[key].astype(int) - new[key].astype(int))
                    name = Path(key.split("@")[0]).name + "@" + key.split("@")[1]
                    if d.any():
                        differ += 1
                        print(f"{be} {name}: differs by up to {d.max()} levels in {int(d.any(2).sum())} pixels")
                    else:
                        print(f"{be} {name}: same bytes")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(base)], cwd=ROOT, capture_output=True)
    print(json.dumps({"rev": a.rev, "frames_differ": differ}))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
