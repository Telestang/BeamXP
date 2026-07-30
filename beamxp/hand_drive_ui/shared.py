from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from beamxp import hand_drive_core as core
from beamxp.model_preview import ModelPreview
from beamxp.plates import generator as plate_generator
from beamxp.plates.editor import PlateEditorDialog
from beamxp.plates.library import PlateLibraryDialog

try:  # GPU mesh preview; the box viewer remains the fallback
    from beamxp import mesh_preview
except Exception:
    mesh_preview = None


# This package is intended to sit in the directory that previously contained
# the monolithic GUI script. Keeping APP_ROOT one level above this package
# preserves the old resource lookup semantics.
APP_ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else APP_ROOT
)
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_ROOT))
SOURCE_ROOT_DIR = APP_ROOT.parent
BLENDER_PREVIEW_SCRIPT = RESOURCE_DIR / "blender_preview_backend.py"
APP_ICON_NAME = "assets/BeamXP_icon.ico"
BLENDER_CANDIDATES = (
    Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
)

def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def position_labels(
    position: tuple[float, float, float],
    variant_dependent: bool,
) -> tuple[str, str, str]:
    """x/y/z cells, marked when the part sits elsewhere on other trims.

    Without the marker the columns silently change meaning between rows: most
    parts have one position, but a part declared by mutually exclusive parts
    only has the one belonging to the trim on screen. All three cells carry
    the mark because it is the whole coordinate that is trim-specific --
    flagging x alone would imply x is the axis that moves, and usually it is
    not (the D-Series gooseneck hitch shifts along y)."""
    suffix = " *" if variant_dependent else ""
    return tuple(f"{fmt_float(value)}{suffix}" for value in position)


def yn_label(value: object) -> str:
    return "Y" if bool(value) else "N"


def mode_label(mode: str) -> str:
    return {
        core.MODE_SKIP: "Skip",
        core.MODE_MIRROR: "Mirror Aesthetic",
        core.MODE_MIRROR_STRUCTURAL: "Mirror Structural",
        core.MODE_TRANSLATE: "Translate",
    }.get(mode, "Skip")


def part_type_label(
    object_id: str,
    flexbody_meshes: set[str],
    prop_meshes: set[str],
) -> str:
    """JBeam rendering role across the currently selected variants."""
    is_flexbody = object_id in flexbody_meshes
    is_prop = object_id in prop_meshes
    if is_flexbody and is_prop:
        return "Flexbody + Prop"
    if is_flexbody:
        return "Flexbody"
    if is_prop:
        return "Prop"
    return "Unknown"


MODE_CYCLE_VALUES = [core.MODE_SKIP, core.MODE_MIRROR, core.MODE_MIRROR_STRUCTURAL, core.MODE_TRANSLATE]
MODE_VALUES_BY_LABEL = {mode_label(mode): mode for mode in MODE_CYCLE_VALUES}
MODE_HOTKEYS = {
    "q": core.MODE_SKIP,
    "w": core.MODE_MIRROR,
    "e": core.MODE_MIRROR_STRUCTURAL,
    "r": core.MODE_TRANSLATE,
}

BUILD_LABELS = {
    core.BUILD_OFF: "Off",
    core.BUILD_CONVERTED: "Converted",
    core.BUILD_ORIGINAL: "Plates Only",
    core.BUILD_BOTH: "Both",
}

# How long (in milliseconds) a part may sit on Mirror Structural before the
# source-part prompt commits it. Tweak this value to change the timeout.
STRUCTURAL_PROMPT_DELAY_MS = 300

def offset_label(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return fmt_float(abs(float(value)))
    except (TypeError, ValueError):
        return ""


def offset_display(mode: str, value: object, *, manual_delta: bool) -> str:
    if mode != core.MODE_TRANSLATE:
        return "N/A"
    explicit = offset_label(value)
    if explicit:
        return explicit
    return "Manual" if manual_delta else "Auto"


def existing_initial_dir(path: object, fallback: Path) -> str:
    candidate = Path(str(path)) if path else fallback
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.exists():
        return str(candidate)
    return str(fallback)


def app_icon_path() -> Path | None:
    for candidate in (
        RESOURCE_DIR / APP_ICON_NAME,
        THIS_DIR / APP_ICON_NAME,
        SOURCE_ROOT_DIR / APP_ICON_NAME,
    ):
        if candidate.exists():
            return candidate
    return None

__all__ = [
    "APP_ICON_NAME",
    "APP_ROOT",
    "BLENDER_CANDIDATES",
    "BLENDER_PREVIEW_SCRIPT",
    "BUILD_LABELS",
    "MODE_CYCLE_VALUES",
    "MODE_HOTKEYS",
    "MODE_VALUES_BY_LABEL",
    "RESOURCE_DIR",
    "SOURCE_ROOT_DIR",
    "STRUCTURAL_PROMPT_DELAY_MS",
    "THIS_DIR",
    "ModelPreview",
    "Path",
    "PlateEditorDialog",
    "PlateLibraryDialog",
    "ThreadPoolExecutor",
    "app_icon_path",
    "argparse",
    "core",
    "datetime",
    "existing_initial_dir",
    "filedialog",
    "fmt_float",
    "json",
    "mesh_preview",
    "messagebox",
    "mode_label",
    "offset_display",
    "offset_label",
    "part_type_label",
    "plate_generator",
    "position_labels",
    "queue",
    "re",
    "subprocess",
    "sys",
    "threading",
    "tk",
    "ttk",
    "yn_label",
]
