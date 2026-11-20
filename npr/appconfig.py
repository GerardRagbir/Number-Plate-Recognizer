"""User-editable configuration: one global document plus an optional override per camera.

The global document lives in the ``kv`` table; a camera's lives in ``sources.opts["config"]``.
``effective()`` folds the two into what a pipeline actually runs with.
"""
from __future__ import annotations

import copy
import re

from .config import settings

CAMERA_TYPES = ("", "fixed", "ptz", "dome", "fisheye", "doorbell", "dashcam", "traffic", "other")

# sensitivity → fraction of the (downscaled) picture that must change to count as motion
MOTION_LEVELS = {"low": 0.006, "medium": 0.002, "high": 0.0006}


def object_types() -> list[str]:
    return list(settings.vehicle_classes.values())


def global_defaults() -> dict:
    return {
        "objects": {t: True for t in object_types()},  # False = blocked everywhere unless a camera allows it
        "motion": {"enabled": True, "sensitivity": "medium", "keepalive_s": 10.0, "hold_s": 2.0},
        "network": {"reconnect": True},
        "defaults": {"hfov": 70.0, "range_m": 60.0},
        "ui": {
            "toolbar_autohide": True,  # the overlay toolbar slides away until the pointer nears it
            # the live wall: "auto" fits the cameras and pages when there are more than per_page;
            # "groups" gives each camera group its own page; "custom" uses the pages listed here
            "wall": {"mode": "auto", "per_page": 4, "pages": []},
        },
    }


def camera_defaults() -> dict:
    return {
        "camera_type": "", "make_model": "", "notes": "",
        "profile": "",        # id of the entry in the camera profile database this was set up from
        "mount_height_m": None, "tilt_deg": None,
        "objects": {},        # type → "allow" | "block"; missing = follow the global setting
        "motion": "inherit",  # "inherit" | "on" | "off"
        "group": None,        # id of the camera group this belongs to
    }


def clean_group(doc: dict | None) -> dict:
    """A camera group: a name plus the same optional overrides a camera can carry."""
    doc = doc or {}
    motion = doc.get("motion")
    return {
        "id": int(doc.get("id") or 0),
        "name": (str(doc.get("name") or "").strip() or "Group")[:80],
        "objects": {t: v for t, v in (doc.get("objects") or {}).items()
                    if t in object_types() and v in ("allow", "block")},
        "motion": motion if motion in ("inherit", "on", "off") else "inherit",
    }


def _merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict) and k != "objects":
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


def clean_global(doc: dict | None) -> dict:
    out = _merge(global_defaults(), doc)
    types = object_types()
    out["objects"] = {t: bool((doc or {}).get("objects", {}).get(t, True)) for t in types}
    m = out["motion"]
    m["enabled"] = bool(m["enabled"])
    if m["sensitivity"] not in MOTION_LEVELS:
        m["sensitivity"] = "medium"
    m["keepalive_s"] = min(max(float(m["keepalive_s"] or 0), 0.0), 3600.0)
    m["hold_s"] = min(max(float(m["hold_s"] or 0), 0.0), 60.0)
    out["network"]["reconnect"] = bool(out["network"]["reconnect"])
    out["ui"]["toolbar_autohide"] = bool(out["ui"]["toolbar_autohide"])
    out["ui"]["wall"] = clean_wall(out["ui"].get("wall"))
    d = out["defaults"]
    d["hfov"] = min(max(float(d["hfov"] or 70), 5.0), 180.0)
    d["range_m"] = min(max(float(d["range_m"] or 60), 5.0), 2000.0)
    return out


WALL_SIZES = {"1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "focus": 6, "9": 9, "16": 16}  # layout → tiles


def clean_wall(doc: dict | None) -> dict:
    doc = doc if isinstance(doc, dict) else {}
    mode = doc.get("mode") if doc.get("mode") in ("auto", "groups", "custom") else "auto"
    try:
        per = int(doc.get("per_page") or 4)
    except (TypeError, ValueError):
        per = 4
    per = per if per in (1, 2, 3, 4, 6, 9, 16) else 4
    pages = []
    for p in (doc.get("pages") or [])[:24]:
        if not isinstance(p, dict):
            continue
        size = str(p.get("size")) if str(p.get("size")) in WALL_SIZES else "4"
        tiles = [t if isinstance(t, int) and not isinstance(t, bool) else None for t in (p.get("tiles") or [])]
        n = WALL_SIZES[size]
        pages.append({"name": (str(p.get("name") or "").strip() or f"Page {len(pages) + 1}")[:40], "size": size,
                      "tiles": (tiles + [None] * n)[:n]})
    return {"mode": mode, "per_page": per, "pages": pages}


def _num(v):
    if v in (None, ""):
        return None
    return float(v)


def clean_camera(doc: dict | None) -> dict:
    out = _merge(camera_defaults(), doc)
    if out["camera_type"] not in CAMERA_TYPES:
        out["camera_type"] = "other"
    for k in ("make_model", "notes", "profile"):
        out[k] = str(out[k] or "")[:500]
    out["mount_height_m"], out["tilt_deg"] = _num(out["mount_height_m"]), _num(out["tilt_deg"])
    out["objects"] = {t: v for t, v in (out["objects"] or {}).items()
                      if t in object_types() and v in ("allow", "block")}
    if out["motion"] not in ("inherit", "on", "off"):
        out["motion"] = "inherit"
    try:
        out["group"] = int(out["group"]) if out["group"] not in (None, "", 0) else None
    except (TypeError, ValueError):
        out["group"] = None
    return out


def effective(global_doc: dict | None, camera_doc: dict | None, group_doc: dict | None = None) -> dict:
    """What one pipeline should do. The most specific setting wins: camera, then its group, then global."""
    g, c = clean_global(global_doc), clean_camera(camera_doc)
    grp = clean_group(group_doc) if group_doc else {"objects": {}, "motion": "inherit"}
    allowed = set()
    for t in object_types():
        rule = c["objects"].get(t) or grp["objects"].get(t)
        if rule == "allow" or (rule is None and g["objects"][t]):
            allowed.add(t)
    choice = c["motion"] if c["motion"] != "inherit" else grp["motion"]
    motion = g["motion"]["enabled"] if choice == "inherit" else choice == "on"
    return {
        "allowed": allowed,
        "motion": motion,
        "motion_fraction": MOTION_LEVELS[g["motion"]["sensitivity"]],
        "keepalive_s": g["motion"]["keepalive_s"],
        "hold_s": g["motion"]["hold_s"],
        "reconnect": g["network"]["reconnect"],
    }


# ---- credentials in URLs ------------------------------------------------------------------
_CRED = re.compile(r"(?<=://)([^\s:/@]+):([^\s@/]+)@")
MASK = "***"


def mask_uri(text: str | None) -> str | None:
    """Hide the password of every ``scheme://user:password@`` in a string (URLs, errors, logs)."""
    return text if not text else _CRED.sub(lambda m: f"{m.group(1)}:{MASK}@", text)


def unmask_uri(new: str, old: str | None) -> str:
    """A URL edited in the dashboard still carries the mask: put the stored password back."""
    m_new, m_old = _CRED.search(new or ""), _CRED.search(old or "")
    if m_new and m_new.group(2) == MASK and m_old:
        return new[: m_new.start(2)] + m_old.group(2) + new[m_new.end(2):]
    return new
