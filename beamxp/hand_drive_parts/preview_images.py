"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 2407-2575. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
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

def conversion_source_name(context: VehicleContext) -> str:
    source_name = original_source_name(context)
    if "custom" in source_name.lower():
        return "Custom"
    return f"Custom based on {source_name}"


def converted_description(base_description: object, target_hand: str) -> str:
    suffix = f"converted to {target_hand}"
    description = str(base_description or "").strip()
    if not description:
        return suffix[0].upper() + suffix[1:]
    lowered = description.lower()
    if lowered.endswith((f" - {suffix.lower()}", suffix.lower())):
        return description
    return f"{description} - {suffix}"


def source_preview_path(source_zip: Path, vehicle_path: str, config_name: str) -> str | None:
    prefix = f"{vehicle_path.rstrip('/')}/{config_name}".lower()
    preview_exts = (".jpg", ".jpeg", ".png", ".webp")
    with zipfile.ZipFile(source_zip) as zf:
        for name in zf.namelist():
            clean = name.replace("\\", "/")
            if (
                clean.lower().startswith(prefix)
                and Path(clean).suffix.lower() in preview_exts
                and clean.lower() == f"{prefix}{Path(clean).suffix.lower()}"
            ):
                return clean
    return None


# Generated config preview sticker tuning. Origin values are fractions of the
# preview image size, measured from the selected anchor corner. Positive X/Y
# offsets move inward from that corner. The sticker keeps its own aspect ratio.
XP_STICKER_ANCHOR = "top_left"  # top_left, top_right, bottom_left, bottom_right
XP_STICKER_ORIGIN_X_FRACTION = 0.02
XP_STICKER_ORIGIN_Y_FRACTION = 0.65
# 0.25 tuned against the 512px-wide HDC sticker; the XP sticker is 435px wide
# at the same height, so 0.25 * 435/512 keeps the on-screen badge size equal.
XP_STICKER_WIDTH_FRACTION = 0.2124


def xp_sticker_path() -> Path | None:
    """Locate the bundled XP sticker PNG. Checks the PyInstaller bundle dir
    first (frozen builds), then the source tree. Returns None if absent."""
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "xp_sticker.png")
        candidates.append(Path(meipass) / "assets" / "xp_sticker.png")
    candidates.append(APP_DIR / "xp_sticker.png")
    candidates.append(APP_DIR / "assets" / "xp_sticker.png")
    candidates.append(SOURCE_ROOT_DIR / "xp_sticker.png")
    candidates.append(SOURCE_ROOT_DIR / "assets" / "xp_sticker.png")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def xp_sticker_position(
    image_size: tuple[int, int],
    sticker_size: tuple[int, int],
) -> tuple[int, int]:
    image_w, image_h = image_size
    sticker_w, sticker_h = sticker_size
    offset_x = round(image_w * XP_STICKER_ORIGIN_X_FRACTION)
    offset_y = round(image_h * XP_STICKER_ORIGIN_Y_FRACTION)
    anchor = XP_STICKER_ANCHOR.lower()
    x = image_w - sticker_w - offset_x if "right" in anchor else offset_x
    y = image_h - sticker_h - offset_y if "bottom" in anchor else offset_y
    return max(0, min(image_w - sticker_w, x)), max(0, min(image_h - sticker_h, y))


def composite_xp_sticker(image):
    """Alpha-composite the XP sticker onto generated preview images. Returns
    a new RGBA image on success, or the input unchanged if the sticker asset is
    missing or anything fails -- a sticker problem must never discard the
    generated/mirrored preview it is decorating."""
    try:
        from PIL import Image

        sticker_path = xp_sticker_path()
        if sticker_path is None:
            return image
        base = image.convert("RGBA")
        with Image.open(sticker_path) as raw:
            sticker = raw.convert("RGBA")
        target_w = max(1, min(base.width, round(base.width * XP_STICKER_WIDTH_FRACTION)))
        scale = target_w / sticker.width
        target_h = max(1, min(base.height, round(sticker.height * scale)))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        sticker = sticker.resize((target_w, target_h), resample)
        base.alpha_composite(sticker, xp_sticker_position(base.size, sticker.size))
        return base
    except Exception:
        return image


def write_mirrored_preview(
    context: VehicleContext,
    output_vehicle_dir: Path,
    config_name: str,
    output_config: str,
) -> Path | None:
    preview_path = source_preview_path(context.source_zip, context.vehicle_path, config_name)
    if not preview_path:
        return None

    target = output_vehicle_dir / f"{output_config}{Path(preview_path).suffix.lower()}"
    with zipfile.ZipFile(context.source_zip) as zf:
        data = zf.read(preview_path)

    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as image:
            mirrored = ImageOps.mirror(image)
            # Final compositing step: brand the generated preview with the XP
            # sticker (top-left, alpha-blended). Never applied to stock originals.
            mirrored = composite_xp_sticker(mirrored)
            save_kwargs: dict[str, object] = {}
            image_format = image.format or Path(preview_path).suffix.lstrip(".").upper()
            if image_format.upper() in {"JPG", "JPEG"}:
                image_format = "JPEG"
                save_kwargs = {"quality": 92}
                if mirrored.mode in {"RGBA", "P"}:
                    mirrored = mirrored.convert("RGB")
            out = io.BytesIO()
            mirrored.save(out, format=image_format, **save_kwargs)
            write_bytes_file(target, out.getvalue())
    except Exception:
        # A copied preview is still better than leaving the generated config blank.
        write_bytes_file(target, data)
    return target


def write_stock_preview(
    context: VehicleContext,
    output_vehicle_dir: Path,
    config_name: str,
    output_config: str,
) -> Path | None:
    """Copy a source preview for a plates-only trim and add the XP marker."""
    preview_path = source_preview_path(context.source_zip, context.vehicle_path, config_name)
    if not preview_path:
        return None
    target = output_vehicle_dir / f"{output_config}{Path(preview_path).suffix.lower()}"
    with zipfile.ZipFile(context.source_zip) as zf:
        data = zf.read(preview_path)
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            branded = composite_xp_sticker(image)
            image_format = image.format or Path(preview_path).suffix.lstrip(".").upper()
            save_kwargs: dict[str, object] = {}
            if image_format.upper() in {"JPG", "JPEG"}:
                image_format = "JPEG"
                save_kwargs = {"quality": 92}
                if branded.mode in {"RGBA", "P"}:
                    branded = branded.convert("RGB")
            out = io.BytesIO()
            branded.save(out, format=image_format, **save_kwargs)
            write_bytes_file(target, out.getvalue())
    except Exception:
        write_bytes_file(target, data)
    return target

__all__ = ['conversion_source_name', 'converted_description', 'source_preview_path', 'XP_STICKER_ANCHOR', 'XP_STICKER_ORIGIN_X_FRACTION', 'XP_STICKER_ORIGIN_Y_FRACTION', 'XP_STICKER_WIDTH_FRACTION', 'xp_sticker_path', 'xp_sticker_position', 'composite_xp_sticker', 'write_mirrored_preview', 'write_stock_preview']
