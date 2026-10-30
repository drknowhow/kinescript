"""Kinescript: motion graphics and video edits written as data, checked and rendered by agents."""
from .output import render, still
from .scene import Scene, SceneError, load

__version__ = "1.1.0"
__all__ = ["Scene", "SceneError", "load", "render", "still", "__version__"]
