from __future__ import annotations

import multiprocessing

from beamxp.hand_drive_ui.app import HandDriveToolApp, main, parse_args, validate_source
from beamxp.hand_drive_ui.recommendation_common import (
    CABIN_NAME_PATTERNS,
    CENTRE_FUNCS,
    DRIVER_BAND,
    DRIVER_FUNCS,
    DRIVER_NAME_PATTERNS,
    INERT_NAME_PATTERNS,
    PAIR_MIN_OFFSET,
    TEXTURE_GLYPH_PATTERNS,
    _is_door_card_mesh_id,
    _is_seat_mesh_id,
    _side_pair_kind_for_mesh,
    assembly_func_role,
    mesh_center,
    prop_func_role,
    recommendation_matches,
    recommendation_text,
    slot_family,
    slot_role,
)
from beamxp.hand_drive_ui.recommendation_engine import build_mode_recommendations
from beamxp.hand_drive_ui.recommendation_evidence import vehicle_evidence
from beamxp.hand_drive_ui.recommendation_pairing import (
    _name_pair_candidate,
    handed_mesh_ids,
    resolve_side_twin,
)
from beamxp.hand_drive_ui.shared import *


__all__ = sorted(name for name in globals() if not name.startswith("__"))


if __name__ == "__main__":
    # Texture correction spreads its atlases over a process pool, and Windows
    # starts those by spawning this entry point again. In a PyInstaller build
    # that means re-running the executable, which without this would raise a
    # second copy of the whole tool per worker instead of a worker.
    multiprocessing.freeze_support()
    main()
