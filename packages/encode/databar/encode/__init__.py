"""Teaching package: the sending side (data -> images).

Technique: re-export the useful names here and list them in __all__, so callers
write `from databar.encode import Sender` and `__all__` documents the public API.
"""
from .sender import Sender, render

# Technique: __all__ defines what `from package import *` exposes — the intended surface.
__all__ = ['Sender', 'render']
