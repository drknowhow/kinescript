"""Kinescript: motion graphics and video edits written as data, checked and rendered by agents."""
try:
    import skia  # noqa: F401
except ImportError as e:  # on Linux the wheel loads the system's EGL and GL, even for the CPU backend
    if ".so" not in str(e):
        raise
    raise ImportError(f"{e}. skia-python needs system libraries; on Debian or Ubuntu: "
                      "sudo apt install libegl1 libgl1 libfontconfig1") from e
# Depth (z, rotateX, rotateY, a camera) builds its matrices with M44.Rows, which skia-python first binds in 144.1.
if not hasattr(skia.M44, "Rows"):
    raise ImportError(f"skia-python {getattr(skia, '__version__', '?')} is too old for Kinescript: "
                      'pip install -U "skia-python>=144.1"')
from .output import render, still
from .scene import Scene, SceneError, load

__version__ = "1.5.0"
__all__ = ["Scene", "SceneError", "load", "render", "still", "__version__"]
