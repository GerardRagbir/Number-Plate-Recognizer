"""Resolve every Apple-framework symbol the package uses, once, on the calling thread.

pyobjc binds framework functions and constants lazily on first attribute access, and that first
access is not thread-safe: two pipelines starting together can race and one gets a KeyError.
Touching each ``Framework.symbol`` we reference up front makes all later lookups plain dict hits.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

_ALIASES = {"AVF": "AVFoundation", "SCK": "ScreenCaptureKit"}
_FRAMEWORKS = ("Quartz", "CoreMedia", "AVFoundation", "Vision", "CoreML", "ScreenCaptureKit", "Metal")
_REF = re.compile(r"\b(" + "|".join((*_FRAMEWORKS, *_ALIASES)) + r")\.([A-Za-z_]\w*)")
_done = False


def preload() -> int:
    global _done
    if _done:
        return 0
    n = 0
    for path in Path(__file__).resolve().parent.parent.rglob("*.py"):
        for mod, name in set(_REF.findall(path.read_text())):
            try:
                getattr(importlib.import_module(_ALIASES.get(mod, mod)), name)
                n += 1
            except Exception:  # a symbol missing on this macOS version is reported where it is used
                pass
    _done = True
    return n
