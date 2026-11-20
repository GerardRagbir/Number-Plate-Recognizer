"""Search facets: the vocabulary behind the Activity search bar.

A *facet* is one kind of thing a typed keyword can mean (a colour, a vehicle type, …). Each facet has
a short list of values, and each value a list of words that mean it. The dashboard turns a matching
word into a chip; the server filters on the facet's value. To teach the search a new category, add a
facet here and a filter for its key in ``Store.activity``.

Colours are deliberately coarse *families*. The cameras are uncalibrated, so "red" has to cover
everything that looks red-ish (maroon, crimson, rust, a red car at dusk), not one exact shade.
"""
from __future__ import annotations

import numpy as np

from .config import settings

# id, label, swatch shown on the chip, words that mean it
COLOURS = [
    ("red", "Red", "#d0463b", ["red", "maroon", "burgundy", "crimson", "scarlet", "rust", "brown", "wine"]),
    ("yellow", "Yellow / orange", "#e2b33c", ["yellow", "gold", "golden", "orange", "amber", "beige", "tan", "cream", "mustard"]),
    ("green", "Green", "#3f9b5a", ["green", "lime", "olive", "teal", "turquoise", "mint"]),
    ("blue", "Blue", "#3f78d0", ["blue", "navy", "azure", "cyan", "indigo"]),
    ("purple", "Purple", "#8a5bd0", ["purple", "violet", "lilac", "mauve"]),
    ("pink", "Pink", "#d86aa8", ["pink", "magenta", "fuchsia", "rose"]),
    ("white", "White", "#eef1f4", ["white", "ivory", "pearl"]),
    ("grey", "Grey / silver", "#8d98a5", ["grey", "gray", "silver", "gunmetal", "charcoal", "graphite"]),
    ("black", "Black", "#15181c", ["black", "dark"]),
]
TYPE_WORDS = {
    "car": ["car", "cars", "sedan", "suv", "hatchback", "taxi", "van", "pickup"],
    "truck": ["truck", "trucks", "lorry", "lorries"],
    "bus": ["bus", "buses", "coach", "maxi"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "scooter"],
    "bicycle": ["bicycle", "bicycles", "cycle", "cyclist", "pushbike"],
}

# hue bins of the stored colour signature (see vision.plates.colour_histogram): 12 hues of 30°
_HUE_FAMILY = ["red", "yellow", "green", "green", "green", "green", "blue", "blue", "blue", "purple", "pink", "red"]


def colour_family(hist) -> str | None:
    """Name the colour family of a vehicle from its 40-bin colour signature.

    Bins 0–35 are saturated pixels (12 hues × 3 saturation levels); 36–39 are unsaturated pixels
    from black to white. Windows, tyres and shadow put dark pixels on every vehicle, so the darkest
    bin is discounted before deciding between black, grey and white.
    """
    if hist is None:
        return None
    h = np.asarray(hist, dtype=np.float32)
    if h.size != 40 or not np.isfinite(h).all() or h.sum() <= 0:
        return None
    h = h / h.sum()
    chroma = h[:36].reshape(12, 3).sum(1)
    if chroma.sum() >= 0.25:
        score: dict[str, float] = {}
        for i, fam in enumerate(_HUE_FAMILY):
            score[fam] = score.get(fam, 0.0) + float(chroma[i])
        return max(score, key=score.get)
    black, dark, light, white = (float(v) for v in h[36:])
    black *= 0.55
    if white >= max(black, dark + light * 0.5):
        return "white"
    if black >= dark + light:
        return "black"
    return "grey"


def facets() -> list[dict]:
    types = [t for t in settings.vehicle_classes.values()]
    return [
        {"key": "colour", "label": "Colour", "param": "colour",
         "values": [{"id": i, "label": label, "swatch": sw, "words": words} for i, label, sw, words in COLOURS]},
        {"key": "type", "label": "Vehicle type", "param": "cls",
         "values": [{"id": t, "label": t.capitalize(), "words": TYPE_WORDS.get(t, [t])} for t in types]},
    ]
