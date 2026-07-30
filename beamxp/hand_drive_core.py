from __future__ import annotations
# pyright: reportUnsupportedDunderAll=false

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import re
import shutil
import sys
import textwrap
import types
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
from xml.etree import ElementTree as ET

import numpy as np

from beamxp import spatial_visibility_backend
from beamxp import transform_helpers
from beamxp.core import dae
from beamxp.core import cache as context_cache
from beamxp.core import mesh_resolution
from beamxp.core.beam_json import (
    add_missing_json_commas,
    display_name_for,
    info_path_for_config,
    json_line_needs_comma,
    load_info,
    load_pc,
    parse_beamng_json,
    strip_json_comments,
    zip_json_by_name as _zip_json_by_name,
)
from beamxp.core.constants import (
    ACTION_OPPOSITE,
    ACTION_SKIP,
    ACTION_TO_LHD,
    ACTION_TO_RHD,
    APP_DIR,
    APP_SETTINGS_PATH,
    BUILD_BOTH,
    BUILD_CHOICES,
    BUILD_CONVERTED,
    BUILD_OFF,
    BUILD_ORIGINAL,
    HAND_AUTO,
    HAND_CHOICES,
    HAND_LHD,
    HAND_RHD,
    HAND_UNKNOWN,
    MODE_CHOICES,
    MODE_MIRROR,
    MODE_MIRROR_STRUCTURAL,
    MODE_SKIP,
    MODE_TRANSLATE,
    NS,
    NUMBER_RE,
    PREVIEW_FAR_LIMIT,
    PROJECTS_DIR,
    SOURCE_ROOT_DIR,
    STEERING_NAME_EXCLUDES,
    THIS_DIR,
    TOOL_VERSION,
    USER_DATA_DIR,
    WORKSPACE_DIR,
    default_beamng_mods_dir,
    default_user_data_dir,
)
from beamxp.core.files import (
    beamng_game_common_zips,
    clean_dir,
    common_zip_candidates,
    direct_vehicle_files,
    fs_path,
    list_vehicle_files,
    load_jbeam_texts,
    make_zip,
    project_dir_for,
    read_json_file,
    safe_id,
    safe_project_segment,
    vehicle_ids_in_zip,
    vehicle_prefix,
    write_bytes_file,
    write_text_file,
    write_xml_tree,
    zip_member_path,
)
from beamxp.core.cache import (
    CONTEXT_CACHE_VERSION,
    HAND_DETECTION_CACHE_VERSION,
    clear_parts_cache,
    clear_variant_hands_cache,
    context_cache_fingerprint,
    context_cache_path,
    context_fingerprint_hash,
    load_cached_part_ids,
    load_cached_vehicle_context,
    parts_cache_path,
    save_cached_part_ids,
    save_vehicle_context_cache,
    selection_cache_key,
    variant_hands_cache_fingerprint,
    variant_hands_cache_path,
)
from beamxp.core.geometry import (
    PROP_VECTOR_RE,
    brg_rotation_matrix3,
    clamp_value,
    cross_product,
    euler_from_matrix3,
    euler_matrix3,
    euler_yzx_from_matrix3,
    identity_matrix,
    matrix3_from_axes,
    matrix3_from_matrix4,
    matrix4_flat,
    mirror_rotation_matrix_x,
    mirror_x_matrix4,
    multiply_matrix,
    multiply_matrix3,
    normalize_vector,
    prop_base_rotation_matrix3,
    prop_row_vector_objects,
    rotation_transpose_matrix3,
    rotation_transpose_matrix4,
    rotation_x_matrix,
    rotation_y_matrix,
    rotation_z_matrix,
    scale_matrix,
    sign_number,
    translation_matrix,
    vector_subtract,
)
from beamxp.core.models import (
    BakedMeshSpec,
    BuildResult,
    MeshPlacement,
    ResolvedMeshPosition,
    SharedBakeContext,
    SlotDef,
    VariantInfo,
    VehicleContext,
)
from beamxp.plates import generator as plate_generator


# ---------------------------------------------------------------------------
# Public orchestration / compatibility facade
# ---------------------------------------------------------------------------
# The implementation modules deliberately do not import one another.  This
# keeps the physical split free of circular-import failures while preserving
# the original function bodies unchanged.  Once all slices are imported, the
# original module namespace is wired into every slice so their existing global
# references resolve exactly as they did in the monolith.

from beamxp.hand_drive_parts import (
    mesh_data,
    jbeam_syntax,
    mesh_placement,
    preview_images,
    vehicle_context,
    configuration,
    part_resolution,
    handedness,
    rewriting,
    generation,
    build_pipeline,
    spatial_analysis,
)

_IMPLEMENTATION_MODULES = (
mesh_data,
jbeam_syntax,
mesh_placement,
preview_images,
vehicle_context,
configuration,
part_resolution,
handedness,
rewriting,
generation,
build_pipeline,
spatial_analysis,
)


def _wire_implementation_modules() -> dict[str, object]:
    public: dict[str, object] = {}
    owners: dict[str, str] = {}

    for module in _IMPLEMENTATION_MODULES:
        for name in module.__all__:
            if name in public:
                raise RuntimeError(
                    f"duplicate implementation symbol {name!r} in "
                    f"{owners[name]} and {module.__name__}"
                )
            public[name] = getattr(module, name)
            owners[name] = module.__name__

    # Inject only symbols originating in the former monolith.  Each slice
    # already imports its own standard-library and beamxp dependencies.
    for module in _IMPLEMENTATION_MODULES:
        module.__dict__.update(public)

    return public


_IMPLEMENTATION_EXPORTS = _wire_implementation_modules()
globals().update(_IMPLEMENTATION_EXPORTS)


class _HandDriveCoreFacade(types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in globals().get("_IMPLEMENTATION_EXPORTS", {}):
            for module in _IMPLEMENTATION_MODULES:
                if name in module.__dict__:
                    module.__dict__[name] = value


sys.modules[__name__].__class__ = _HandDriveCoreFacade

# Preserve the practical public surface of the old module: imported constants
# and helpers remain available, as do every function/class/constant that was
# defined by the monolith itself.
__all__ = sorted(
    name for name in globals()
    if not name.startswith("__")
    and name not in {"_IMPLEMENTATION_MODULES", "_IMPLEMENTATION_EXPORTS"}
)
