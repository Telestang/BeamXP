from __future__ import annotations

import os
import sys
from pathlib import Path

from beamxp import transform_helpers
from beamxp.plates import generator as plate_generator


def default_user_data_dir() -> Path:
    # Shared with plate_generator so the one-time BeamHDC -> BeamXP data
    # folder migration runs no matter which module resolves the path first.
    return plate_generator.default_user_data_dir()


def default_beamng_mods_dir() -> Path:
    local_appdata = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return local_appdata / "BeamNG" / "BeamNG.drive" / "current" / "mods"


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
SOURCE_ROOT_DIR = APP_DIR if getattr(sys, "frozen", False) else APP_DIR.parent
USER_DATA_DIR = Path(os.environ.get("BEAMXP_DATA_DIR") or os.environ.get("BEAMHDC_DATA_DIR") or default_user_data_dir())
WORKSPACE_DIR = USER_DATA_DIR
THIS_DIR = APP_DIR
PROJECTS_DIR = USER_DATA_DIR / "handedness_conversion_projects"
APP_SETTINGS_PATH = USER_DATA_DIR / "hand_drive_tool_settings.json"
TOOL_VERSION = 2

HAND_LHD = "LHD"
HAND_RHD = "RHD"
HAND_UNKNOWN = "Unknown"
HAND_AUTO = "Auto"
HAND_CHOICES = (HAND_AUTO, HAND_LHD, HAND_RHD, HAND_UNKNOWN)
ACTION_OPPOSITE = "Opposite"
ACTION_TO_RHD = "To RHD"
ACTION_TO_LHD = "To LHD"
ACTION_SKIP = "Skip"
MODE_SKIP = "skip"
MODE_MIRROR = "mirror"
MODE_MIRROR_STRUCTURAL = "mirrorStructural"
MODE_TRANSLATE = "translate"
MODE_CHOICES = (MODE_SKIP, MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_TRANSLATE)
BUILD_OFF = "off"
BUILD_CONVERTED = "converted"
BUILD_ORIGINAL = "original"
BUILD_BOTH = "both"
BUILD_CHOICES = (BUILD_OFF, BUILD_CONVERTED, BUILD_ORIGINAL, BUILD_BOTH)

# Meshes placed further than this from the vehicle origin are treated as
# deliberately hidden and are left out of previews. Keep in sync with
# FAR_LIMIT in beamxp.mesh_preview.
PREVIEW_FAR_LIMIT = 100.0
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEERING_NAME_EXCLUDES = (
    "airbag",
    "box",
    "button",
    "buttons",
    "cowl",
    "cover",
    "rack",
    "shaft",
    "stitch",
    "column",
)

NS = transform_helpers.NS
