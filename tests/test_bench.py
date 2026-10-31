"""The stress scenes in bench/stress.py must keep loading as the schema changes, or the benchmark silently
measures a scene that no longer exists."""
import importlib.util
from pathlib import Path

import pytest

import kinescript as k

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("stress", ROOT / "bench" / "stress.py")
stress = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stress)


@pytest.mark.parametrize("name", list(stress.SCENES))
def test_stress_scene_loads(name):
    scene = k.load(stress.build(name))
    try:
        assert scene.frames > 0 and not [i for i in scene.issues if i["level"] == "error"]
    finally:
        scene.close()
