"""Licence plate design + registration generation for exported conversions.

Generates BeamNG licence plate design mods (design JSON, background textures,
font sprite atlas + character layout, rear-plate parts) from the plate settings
stored in a conversion config, and applies them to the exported .pc files.

The user-facing model is deliberately constrained to three plate families:
EU (wide), US and JP (both 2:1). BeamNG's internal format ids ("52-11",
"30-15") never appear in the UI; they only exist in the generated assets.

Kept separate from the handedness transform logic on purpose: the only shared
touch point is build_batch() calling apply_to_build() on the unpacked output.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from beamxp import transform_helpers

PLATE_SIZE_EU = "EU"
PLATE_SIZE_US = "US"
PLATE_SIZE_JP = "JP"
PLATE_SIZES = (PLATE_SIZE_EU, PLATE_SIZE_US, PLATE_SIZE_JP)

PLATE_PART_AUTO = "auto"
PLATE_PART_WIDE = "52-11"
PLATE_PART_2_1 = "30-15"
PLATE_PART_NONE = "none"
PLATE_PART_CHOICES = (PLATE_PART_AUTO, PLATE_PART_WIDE, PLATE_PART_2_1, PLATE_PART_NONE)
PLATE_MESH_CHOICE_PREFIX = "mesh:"

PLATE_MODE_OFF = "off"
PLATE_MODE_CUSTOM = "custom"
PLATE_MODE_SET = "set"
PLATE_MODE_GENERAL = "general"
PLATE_MODE_TRIM = "trim"

BAND_NONE = "none"
BAND_EU = "eu"
BAND_CUSTOM = "custom"
BAND_CHOICES = (BAND_NONE, BAND_EU, BAND_CUSTOM)

# style -> (background colour, text colour) following the real JP plate classes
JP_STYLES = {
    "private": ("#f4f6f3", "#1d5c3c"),
    "kei": ("#f7c50c", "#141414"),
    "commercial": ("#1d5c3c", "#f4f6f3"),
    "kei commercial": ("#141414", "#f7c50c"),
}
JP_KANA_CHOICES = tuple("あいうえかきくけこさすせそたちつてとなにぬねのはひふほまみむめもやゆよらりるれろわ")
JP_REGION_CHOICES = ("品川", "横浜", "名古屋", "大阪", "なにわ", "神戸", "京都", "札幌", "仙台", "広島", "福岡", "沖縄", "富士山", "練馬")

# BeamNG internal plate format ids (never shown in the UI).
_FORMAT_WIDE = "52-11"
_FORMAT_2_1 = "30-15"
_REAR_FORMAT_WIDE = "bhdc-rear-wide"
_REAR_FORMAT_2_1 = "bhdc-rear-2-1"
_CANVAS = {
    _FORMAT_WIDE: (1024, 196),
    _FORMAT_2_1: (512, 256),
    _REAR_FORMAT_WIDE: (1024, 196),
    _REAR_FORMAT_2_1: (512, 256),
}
_DESIGN_SLOT = "licenseplate_design_2_1"
_REAR_MESH_WIDE = "licenseplate-bhdc-rear-wide"
_REAR_MESH_2_1 = "licenseplate-bhdc-rear-2-1"
_REAR_FALLBACK_MESHES = {_REAR_MESH_WIDE, _REAR_MESH_2_1}
_EU_BAND_FRACTION = 0.11
_EU_BLUE = "#003399"
_ATLAS_WIDTH = 1024
_ATLAS_PAD = 6
_CAP_TARGET = 100  # target capital letter height in atlas pixels
_ASSET_VERSION = 8  # bump to invalidate hashed asset folders
EMBOSS_MAX_UI = 2.0  # upper bound of the emboss slider in the UI
_EMBOSS_BLUR_RADIUS = 1.2
_EMBOSS_NORMAL_HEIGHT = 10.0

_LETTERS = string.ascii_uppercase
_DIGITS = string.digits
_ALNUM = _LETTERS + _DIGITS
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")
_CENTER_DOT_CHAR = "."


class PlateError(RuntimeError):
    """A plate configuration or asset problem the user can fix."""


# --------------------------------------------------------------------------
# Configuration model


def default_plate_config() -> dict[str, object]:
    return {
        "enabled": False,
        "size": PLATE_SIZE_EU,
        "font": {"source": "default", "path": ""},
        # User-supplied background images (any family). Scaled to cover the
        # plate canvas with the overflowing dimension centre-cropped. Each
        # side is independent: a side without an image uses its colour.
        "background": {"frontImage": "", "rearImage": ""},
        "border": {
            "enabled": False,
            "color": "#101010",
            "offset": 8,
            "thickness": 3,
            "cornerRadius": 10,
        },
        "eu": {
            "pattern": "@@## @@@",
            "frontColor": "#ffffff",
            "rearColor": "#ffffff",
            "textColor": "#101010",
            "textX": 0.0,
            "spacing": 0,
            "sideBand": BAND_NONE,
            "bandCode": "",
            "bandCodeColor": "#ffffff",
            "bandColor": _EU_BLUE,
            "bandImage": "",
            "bandFullImage": "",
        },
        "embossStrength": 1.0,
        "us": {
            "pattern": "@## @@@",
            "bgColor": "#ffffff",
            "bgImage": "",
            "textColor": "#1a3378",
            "textScale": 1.0,
            "textX": 0.0,
            "textY": 0.0,
            "spacing": 0,
        },
        "jp": {
            "pattern": "##-##",
            "style": "private",
            "region": "品川",
            "classification": "300",
            "kana": "さ",
        },
    }


def normalized_plate_config(raw: object) -> dict[str, object]:
    """Merge a stored plate config over the defaults so missing keys are safe."""
    merged = default_plate_config()
    if not isinstance(raw, dict):
        return merged
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        elif key in merged:
            merged[key] = value
    if merged.get("size") not in PLATE_SIZES:
        merged["size"] = PLATE_SIZE_EU
    # The US-only bgImage predates the shared front/rear background images;
    # migrate it so old configs keep rendering identically.
    legacy_bg = str(merged["us"].get("bgImage") or "")
    if legacy_bg and not str(merged["background"].get("frontImage") or ""):
        merged["background"]["frontImage"] = legacy_bg
    merged["us"]["bgImage"] = ""
    return merged


def default_plate_binding(*, variant: bool = False) -> dict[str, object]:
    return {
        "mode": PLATE_MODE_GENERAL if variant else PLATE_MODE_OFF,
        "setId": "",
        "sourceConfig": "",
        "customDefined": False,
        "config": default_plate_config(),
        "customConfig": default_plate_config(),
    }


def normalized_plate_binding(raw: object, *, variant: bool = False) -> dict[str, object]:
    """Normalize the XP plate binding model and migrate the old inline model."""
    allowed = {PLATE_MODE_GENERAL, PLATE_MODE_OFF, PLATE_MODE_CUSTOM, PLATE_MODE_SET, PLATE_MODE_TRIM}
    if isinstance(raw, dict) and str(raw.get("mode") or "") in allowed:
        mode = str(raw.get("mode"))
        if not variant and mode == PLATE_MODE_GENERAL:
            mode = PLATE_MODE_OFF
        config = normalized_plate_config(raw.get("config"))
        custom_source = raw.get("customConfig")
        if not isinstance(custom_source, dict) and mode == PLATE_MODE_CUSTOM:
            custom_source = raw.get("config")
        custom_config = normalized_plate_config(custom_source)
        if mode in {PLATE_MODE_CUSTOM, PLATE_MODE_SET}:
            config["enabled"] = True
        return {
            "mode": mode,
            "setId": str(raw.get("setId") or ""),
            "sourceConfig": str(raw.get("sourceConfig") or ""),
            "customDefined": bool(raw.get("customDefined")) or mode == PLATE_MODE_CUSTOM,
            "config": config,
            "customConfig": custom_config,
        }

    # Before the plate library, the general value was the config itself and
    # trim overrides were {mode: general/custom/off, config: ...}. The latter
    # was handled above; this branch migrates the former without losing edits.
    config = normalized_plate_config(raw)
    enabled = bool(config.get("enabled"))
    config["enabled"] = True
    return {
        "mode": PLATE_MODE_CUSTOM if enabled else (PLATE_MODE_GENERAL if variant else PLATE_MODE_OFF),
        "setId": "",
        "sourceConfig": "",
        "customDefined": enabled,
        "config": config,
        "customConfig": copy.deepcopy(config),
    }


def _user_data_dir() -> Path:
    return Path(os.environ.get("BEAMXP_DATA_DIR") or os.environ.get("BEAMHDC_DATA_DIR") or default_user_data_dir())


def plate_sets_dir() -> Path:
    return _user_data_dir() / "plates"


def default_plate_export_path() -> Path:
    return _user_data_dir() / "plate_exports" / "BeamXP_plates.zip"


def _safe_set_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "plate-set"


def plate_set_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    folder = plate_sets_dir()
    if not folder.is_dir():
        return records
    for path in sorted(folder.glob("*.json"), key=lambda item: item.name.lower()):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        set_id = str(raw.get("id") or path.stem)
        records.append({
            "id": set_id,
            "name": str(raw.get("name") or set_id),
            "config": normalized_plate_config(raw.get("config")),
        })
    return records


def plate_set_by_id(set_id: str) -> dict[str, object] | None:
    wanted = str(set_id)
    return next((record for record in plate_set_records() if record["id"] == wanted), None)


def unique_plate_set_id(name: str) -> str:
    base = _safe_set_id(name)
    used = {str(record["id"]) for record in plate_set_records()}
    candidate = base
    number = 2
    while candidate in used:
        candidate = f"{base}-{number}"
        number += 1
    return candidate


def save_plate_set(record: dict[str, object]) -> dict[str, object]:
    set_id = _safe_set_id(str(record.get("id") or record.get("name") or "plate-set"))
    normalized = {
        "id": set_id,
        "name": str(record.get("name") or set_id).strip() or set_id,
        "config": normalized_plate_config(record.get("config")),
    }
    normalized["config"]["enabled"] = True
    folder = plate_sets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{set_id}.json"
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized


def delete_plate_set(set_id: str) -> None:
    path = plate_sets_dir() / f"{_safe_set_id(set_id)}.json"
    if path.exists():
        path.unlink()


def variant_plate_mode(variant_settings: object) -> str:
    """How a trim relates to plate settings: general/set/custom/off."""
    if not isinstance(variant_settings, dict):
        return "general"
    override = variant_settings.get("plate")
    if not isinstance(override, dict):
        return "general"
    return str(override.get("mode") or PLATE_MODE_GENERAL)


def _resolved_binding_config(
    binding: dict[str, object],
    *,
    warnings: list[str] | None = None,
    label: str = "plate settings",
) -> tuple[dict[str, object] | None, str | None]:
    mode = str(binding.get("mode") or PLATE_MODE_OFF)
    if mode in {PLATE_MODE_OFF, PLATE_MODE_GENERAL}:
        return None, None
    if mode == PLATE_MODE_SET:
        set_id = str(binding.get("setId") or "")
        record = plate_set_by_id(set_id)
        if record is not None:
            config = normalized_plate_config(record.get("config"))
            config["enabled"] = True
            binding["config"] = copy.deepcopy(config)  # live value becomes the embedded fallback snapshot
            return config, set_id
        snapshot = normalized_plate_config(binding.get("config"))
        snapshot["enabled"] = True
        if warnings is not None:
            warnings.append(f"{label}: plate set '{set_id}' is missing; using its last saved snapshot")
        return snapshot, set_id or None
    config = normalized_plate_config(binding.get("customConfig"))
    config["enabled"] = True
    binding["config"] = copy.deepcopy(config)
    return config, None


def effective_plate_selection(
    conversion: dict[str, object],
    config_name: str,
    *,
    warnings: list[str] | None = None,
) -> tuple[dict[str, object] | None, str | None]:
    """Return (resolved config, stable set id) for one source trim."""
    general = normalized_plate_binding(conversion.get("plate"))
    conversion["plate"] = general
    variants = conversion.get("variants", {})
    settings = variants.get(config_name) if isinstance(variants, dict) else None
    if isinstance(settings, dict):
        override = normalized_plate_binding(settings.get("plate"), variant=True)
        settings["plate"] = override
    else:
        override = default_plate_binding(variant=True)
    if override["mode"] == PLATE_MODE_GENERAL:
        binding = general
    elif override["mode"] == PLATE_MODE_TRIM:
        source_config = str(override.get("sourceConfig") or "")
        source_settings = variants.get(source_config) if isinstance(variants, dict) else None
        if isinstance(source_settings, dict):
            source_binding = normalized_plate_binding(source_settings.get("plate"), variant=True)
            source_settings["plate"] = source_binding
            config = normalized_plate_config(source_binding.get("customConfig"))
            config["enabled"] = True
            override["config"] = copy.deepcopy(config)
            return config, None
        snapshot = normalized_plate_config(override.get("config"))
        snapshot["enabled"] = True
        if warnings is not None:
            warnings.append(
                f"{config_name}: custom plate source '{source_config}' is missing; using its last saved snapshot"
            )
        return snapshot, None
    else:
        binding = override
    return _resolved_binding_config(binding, warnings=warnings, label=config_name)


def effective_plate_config(conversion: dict[str, object], config_name: str) -> dict[str, object] | None:
    """The plate config that applies to one trim, or None when plates are off."""
    return effective_plate_selection(conversion, config_name)[0]


def active_section(cfg: dict[str, object]) -> dict[str, object]:
    """The per-family ("eu"/"us"/"jp") settings block for the active size."""
    section = cfg.get(str(cfg.get("size", PLATE_SIZE_EU)).lower(), {})
    return section if isinstance(section, dict) else {}


def active_pattern(cfg: dict[str, object]) -> str:
    return str(active_section(cfg).get("pattern") or "")


def plate_summary_label(conversion: dict[str, object]) -> str:
    binding = normalized_plate_binding(conversion.get("plate"))
    mode = str(binding.get("mode"))
    if mode == PLATE_MODE_OFF:
        return "Off"
    if mode == PLATE_MODE_SET:
        set_id = str(binding.get("setId") or "")
        record = plate_set_by_id(set_id)
        return str(record.get("name")) if record else f"Missing set: {set_id}"
    cfg = normalized_plate_config(binding.get("config"))
    return f"{cfg['size']}  ·  {active_pattern(cfg) or 'no pattern'}"


def emboss_strength(cfg: dict[str, object]) -> float:
    """The user's emboss setting clamped to the UI's 0..EMBOSS_MAX_UI scale."""
    try:
        value = float(cfg.get("embossStrength", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(EMBOSS_MAX_UI, value))


def _effective_emboss_strength(cfg: dict[str, object]) -> float:
    """Convert the UI's compact 0-2 scale into a stronger normal-map value."""
    strength = emboss_strength(cfg)
    return strength * (2.0 + strength)


def default_user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return _migrated_user_data_dir(base)


def _migrated_user_data_dir(base: Path) -> Path:
    """One-time BeamHDC -> BeamXP rebrand migration: move the legacy data
    folder to the new name the first time either module resolves the path.
    File contents are left untouched. If the move fails, keep using the
    legacy folder rather than silently starting with empty settings."""
    new_dir = base / "BeamXP"
    legacy_dir = base / "BeamHDC"
    if not new_dir.exists() and legacy_dir.is_dir():
        try:
            legacy_dir.rename(new_dir)
        except OSError:
            return legacy_dir
    return new_dir


def user_fonts_dir() -> Path:
    return _user_data_dir() / "fonts"


def ensure_user_fonts_dir() -> Path:
    path = user_fonts_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_font_files() -> list[Path]:
    folder = ensure_user_fonts_dir()
    try:
        return sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        return []


# --------------------------------------------------------------------------
# Registration patterns


def validate_pattern(pattern: str) -> str | None:
    text = str(pattern or "").strip()
    if not text:
        return "Enter a registration pattern (e.g. @@## @@@)."
    if len(text) > 14:
        return "Registration pattern is too long (max 14 characters)."
    for ch in text:
        if ch in '"\\':
            return "Registration pattern cannot contain quotes or backslashes."
        if not ch.isprintable():
            return "Registration pattern contains unprintable characters."
    return None


def generate_registration(pattern: str, rng: random.Random | None = None) -> str:
    pick = (rng or random).choice
    out = []
    for ch in str(pattern or "").strip():
        if ch == "@":
            out.append(pick(_LETTERS))
        elif ch == "#":
            out.append(pick(_DIGITS))
        elif ch == "~":
            out.append(pick(_ALNUM))
        else:
            out.append(ch.upper())
    return "".join(out)


def _split_two_line_text(text: str) -> tuple[str, str]:
    words = [word for word in str(text or "").strip().split() if word]
    if len(words) <= 1:
        compact = "".join(words) if words else str(text or "").strip()
        midpoint = max(1, (len(compact) + 1) // 2)
        return compact[:midpoint], compact[midpoint:]

    best_index = 1
    best_score: tuple[int, int] | None = None
    for idx in range(1, len(words)):
        left = " ".join(words[:idx])
        right = " ".join(words[idx:])
        score = (abs(len(left) - len(right)), -len(left))
        if best_score is None or score < best_score:
            best_index = idx
            best_score = score
    return " ".join(words[:best_index]), " ".join(words[best_index:])


def _pattern_glyphs(pattern: str) -> set[str]:
    # Designs are also usable on stock vehicles where BeamNG, rather than a
    # BeamXP registration pattern, supplies the text. Keep the inexpensive
    # universal glyph coverage in every atlas so punctuation never vanishes.
    glyphs = set(_ALNUM) | {" ", "-", "."}
    for ch in str(pattern or "").strip():
        if ch not in "@#~":
            glyphs.add(ch.upper())
    return glyphs


# --------------------------------------------------------------------------
# Fonts

_FONT_DIR_CANDIDATES = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
)
_DEFAULT_FONT_NAMES = (
    "bahnschrift.ttf",
    "arialbd.ttf",
    "arial.ttf",
    "calibrib.ttf",
    "tahomabd.ttf",
    "verdanab.ttf",
    "DejaVuSans-Bold.ttf",
    "Arial Bold.ttf",
    "Arial.ttf",
)
_JP_FONT_NAMES = ("YuGothB.ttc", "yugothb.ttc", "meiryob.ttc", "meiryo.ttc", "msgothic.ttc", "YuGothM.ttc")


def _find_font(names: tuple[str, ...]) -> Path | None:
    for folder in _FONT_DIR_CANDIDATES:
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def resolve_font_path(font_cfg: object) -> Path:
    source = "default"
    custom = ""
    library_name = ""
    if isinstance(font_cfg, dict):
        source = str(font_cfg.get("source") or "default")
        custom = str(font_cfg.get("path") or "")
        library_name = str(font_cfg.get("name") or "")
    if source == "library":
        name = library_name or (Path(custom).name if custom else "")
        if not name or Path(name).name != name:
            raise PlateError("Choose a font from the BeamXP fonts folder.")
        path = ensure_user_fonts_dir() / name
        if not path.is_file():
            raise PlateError(f"Plate font file not found in BeamXP fonts folder: {name}")
        return path
    if source == "custom":
        path = Path(custom)
        if not custom or not path.is_file():
            raise PlateError(f"Plate font file not found: {custom or '(none selected)'}")
        return path
    found = _find_font(_DEFAULT_FONT_NAMES)
    if found is None:
        raise PlateError("No default plate font found on this system; choose a TTF/OTF font file instead.")
    return found


def resolve_jp_font_path() -> Path | None:
    return _find_font(_JP_FONT_NAMES)


def _load_font(path: Path, size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(str(path), size=size)
    except Exception as exc:
        raise PlateError(f"Could not read font '{path.name}': {exc}") from exc


# --------------------------------------------------------------------------
# Font sprite atlas + character layout


@dataclass(frozen=True)
class _FontMetrics:
    """Vertical metrics of the plate font at its atlas pixel size."""

    font_px: int
    line_height: int
    base: int  # baseline offset from the tile top (= ascent)
    cap_height: float
    cap_top: int  # top of a capital's ink relative to the tile top


def _plate_font_metrics(font_path: Path):
    """Load the plate font at the size that puts capitals at _CAP_TARGET px."""
    probe = _load_font(font_path, 128)
    cap_box = probe.getbbox("H")
    if not cap_box or cap_box[3] <= cap_box[1]:
        raise PlateError(f"Font '{font_path.name}' has no usable uppercase glyphs.")
    font_px = max(16, round(128 * _CAP_TARGET / (cap_box[3] - cap_box[1])))
    font = _load_font(font_path, font_px)
    ascent, descent = font.getmetrics()
    cap_box = font.getbbox("H")
    metrics = _FontMetrics(
        font_px=font_px,
        line_height=max(1, ascent + descent),
        base=ascent,
        cap_height=float(cap_box[3] - cap_box[1]),
        cap_top=cap_box[1],
    )
    return font, metrics


@dataclass(frozen=True)
class FontAtlas:
    key: str
    layout: dict[str, object]
    image_d: object
    image_n: object
    image_s: object
    metrics: _FontMetrics


def _font_atlas_key(font_path: Path, glyphs: set[str], spacing: int, emboss_strength: float) -> str:
    digest = hashlib.sha1()
    digest.update(str(_ASSET_VERSION).encode())
    try:
        digest.update(hashlib.sha1(font_path.read_bytes()).digest())
    except OSError as exc:
        raise PlateError(f"Could not read font '{font_path}': {exc}") from exc
    digest.update("".join(sorted(glyphs)).encode("utf-8"))
    digest.update(str(int(spacing)).encode())
    digest.update(f"{emboss_strength:.3f}".encode())
    return digest.hexdigest()[:10]


def build_font_atlas(font_path: Path, glyphs: set[str], spacing: int = 0, emboss_strength: float = 3.0) -> FontAtlas:
    """Rasterise a TTF/OTF into a BeamNG plate font atlas + BMFont-style layout.

    Glyph tiles are full line height with the baseline at 'base', yoffset 0;
    the runtime plate generator then only needs xoffset/xadvance which come
    straight from the font metrics (plus the user's extra character spacing).
    """
    from PIL import Image, ImageDraw

    key = _font_atlas_key(font_path, glyphs, spacing, emboss_strength)
    font, metrics = _plate_font_metrics(font_path)
    line_height = metrics.line_height
    cap_height = metrics.cap_height

    entries = []
    for ch in sorted(glyphs):
        if ch == _CENTER_DOT_CHAR:
            advance = max(font.getlength(ch), cap_height * 0.32)
            entries.append({"ch": ch, "w": max(1, round(advance)), "l": 0, "advance": advance, "box": None, "centerDot": True})
            continue
        advance = font.getlength(ch)
        box = font.getbbox(ch)
        if ch == " " or not box or box[2] <= box[0]:
            entries.append({"ch": ch, "w": 0, "l": 0, "advance": advance, "box": None})
            continue
        left, _top, right, _bottom = box
        entries.append({"ch": ch, "w": right - left, "l": left, "advance": advance, "box": box})

    # simple shelf packing
    x = _ATLAS_PAD
    y = _ATLAS_PAD
    row_h = line_height + _ATLAS_PAD
    for entry in entries:
        if entry["w"] <= 0:
            entry["x"], entry["y"] = 0, 0
            continue
        if x + entry["w"] + _ATLAS_PAD > _ATLAS_WIDTH:
            x = _ATLAS_PAD
            y += row_h
        entry["x"], entry["y"] = x, y
        x += entry["w"] + _ATLAS_PAD
    used_height = y + row_h
    atlas_height = 1
    while atlas_height < used_height:
        atlas_height *= 2

    image_d = Image.new("RGBA", (_ATLAS_WIDTH, atlas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image_d)
    for entry in entries:
        if entry["w"] <= 0:
            continue
        if entry.get("centerDot"):
            diameter = max(4, round(cap_height * 0.16))
            left = entry["x"] + (entry["w"] - diameter) / 2
            top = entry["y"] + metrics.cap_top + (cap_height - diameter) / 2
            draw.ellipse((left, top, left + diameter, top + diameter), fill=(255, 255, 255, 255))
            continue
        # baseline sits at 'base' inside the tile: PIL's text origin is the
        # ascent top, so drawing at tile_y keeps ink at its natural offset.
        draw.text((entry["x"] - entry["l"], entry["y"]), entry["ch"], font=font, fill=(255, 255, 255, 255))

    image_s = _specular_from_alpha(image_d.getchannel("A"))
    image_n = _normal_from_alpha(image_d, emboss_strength)

    chars = []
    for entry in entries:
        chars.append({
            "id": str(ord(entry["ch"])),
            "x": str(entry["x"]),
            "y": str(entry["y"]),
            "width": str(entry["w"]),
            "height": str(line_height),
            "xoffset": str(int(entry["l"])),
            "yoffset": "0",
            "xadvance": str(int(round(entry["advance"])) + int(spacing)),
            "page": "0",
            "chnl": "15",
        })
    layout = {
        "info": {
            "face": font_path.stem,
            "size": str(-metrics.font_px),
            "bold": "0", "italic": "0", "charset": "", "unicode": "1",
            "stretchH": "100", "smooth": "1", "aa": "2",
            "padding": "0,0,0,0", "spacing": f"{_ATLAS_PAD},{_ATLAS_PAD}", "outline": "0",
        },
        "common": {
            "lineHeight": str(line_height),
            "base": str(metrics.base),
            "scaleW": str(_ATLAS_WIDTH),
            "scaleH": str(atlas_height),
            "pages": "1", "packed": "0",
            "alphaChnl": "1", "redChnl": "0", "greenChnl": "0", "blueChnl": "0",
        },
        "pages": [{"id": "0", "file": "platefont_d.png"}],
        "chars": {"count": str(len(chars)), "char": chars},
    }
    return FontAtlas(key, layout, image_d, image_n, image_s, metrics)


def _specular_from_alpha(alpha):
    """A flat light-grey specular map wherever the coverage mask has ink."""
    from PIL import Image

    grey = alpha.point(lambda _v: 210)
    return Image.merge("RGBA", (grey, grey, grey, alpha))


def _normal_from_alpha(image, emboss_strength: float):
    """Derive an embossed tangent-space normal map from glyph coverage."""
    import numpy as np
    from PIL import Image, ImageFilter

    height_map = image.getchannel("A").filter(ImageFilter.GaussianBlur(_EMBOSS_BLUR_RADIUS))
    field = (
        np.asarray(height_map, dtype=np.float32)
        / 255.0
        * (_EMBOSS_NORMAL_HEIGHT * max(0.0, float(emboss_strength)))
    )
    gy, gx = np.gradient(field)
    nz = np.ones_like(field)
    length = np.sqrt(gx * gx + gy * gy + nz * nz)
    encode = lambda component: ((component / length) * 0.5 + 0.5) * 255.0
    rgb = np.stack([encode(-gx), encode(gy), encode(nz)], axis=-1).astype(np.uint8)
    out = Image.fromarray(rgb, "RGB").convert("RGBA")
    out.putalpha(image.getchannel("A"))
    return out


# --------------------------------------------------------------------------
# Plate backgrounds


def _open_user_image(path_text: str, purpose: str):
    from PIL import Image

    path = Path(str(path_text))
    if not path.is_file():
        raise PlateError(f"{purpose} image not found: {path_text}")
    try:
        with Image.open(path) as raw:
            return raw.convert("RGBA")
    except Exception as exc:
        raise PlateError(f"Could not read {purpose.lower()} image '{path.name}': {exc}") from exc


def _bounded_int(value: object, default: int, low: int, high: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _border_cfg(cfg: dict[str, object]) -> dict[str, object]:
    raw = cfg.get("border", {})
    return raw if isinstance(raw, dict) else {}


def _corner_radius(cfg: dict[str, object], size: tuple[int, int]) -> int:
    border = _border_cfg(cfg)
    return _bounded_int(border.get("cornerRadius"), 10, 0, min(size) // 2)


def _border_enabled(cfg: dict[str, object]) -> bool:
    border = _border_cfg(cfg)
    return bool(border.get("enabled"))


def _active_side_band_width(cfg: dict[str, object], size: tuple[int, int]) -> int:
    if str(cfg.get("size")) != PLATE_SIZE_EU:
        return 0
    eu = cfg.get("eu", {})
    if not isinstance(eu, dict) or str(eu.get("sideBand") or BAND_NONE) == BAND_NONE:
        return 0
    return max(1, round(size[0] * _EU_BAND_FRACTION))


def _border_geometry(cfg: dict[str, object], size: tuple[int, int]) -> tuple[tuple[float, float, float, float], int, float] | None:
    if not _border_enabled(cfg):
        return None
    border = _border_cfg(cfg)
    max_dim = max(1, min(size))
    offset = _bounded_int(border.get("offset"), 8, 0, max_dim // 3)
    thickness = _bounded_int(border.get("thickness"), 3, 1, max_dim // 4)
    inset = offset + thickness / 2
    left = _active_side_band_width(cfg, size) + inset
    top = inset
    right = size[0] - 1 - inset
    bottom = size[1] - 1 - inset
    if left >= right or top >= bottom:
        return None
    # The radius only shapes the border line; the plate texture corners stay
    # square because the vanilla plate meshes already round the corners.
    radius = min(_corner_radius(cfg, size), (right - left) / 2, (bottom - top) / 2)
    return (left, top, right, bottom), thickness, radius


def _border_alpha(cfg: dict[str, object], size: tuple[int, int]):
    from PIL import Image, ImageDraw

    geometry = _border_geometry(cfg, size)
    if geometry is None:
        return None
    box, thickness, radius = geometry
    alpha = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.rounded_rectangle(
        box,
        radius=radius,
        outline=255,
        width=thickness,
    )
    return alpha


def _decorate_plate_background(cfg: dict[str, object], plate):
    from PIL import ImageDraw

    size = plate.size
    geometry = _border_geometry(cfg, size)
    if geometry is not None:
        box, thickness, radius = geometry
        draw = ImageDraw.Draw(plate)
        draw.rounded_rectangle(
            box,
            radius=radius,
            outline=str(_border_cfg(cfg).get("color") or "#101010"),
            width=thickness,
        )
    plate.putalpha(255)
    return plate


def _render_border_mode_images(cfg: dict[str, object], size: tuple[int, int], emboss_strength: float):
    from PIL import Image

    alpha = _border_alpha(cfg, size)
    if alpha is None:
        return None
    border = Image.new("RGBA", size, (255, 255, 255, 0))
    border.putalpha(alpha)
    return _normal_from_alpha(border, emboss_strength), _specular_from_alpha(alpha)


def _draw_star(draw, center: tuple[float, float], radius: float, fill) -> None:
    import math

    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.42
        points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
    draw.polygon(points, fill=fill)


def _fit_inside(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    src_w, src_h = max(1, src[0]), max(1, src[1])
    dst_w, dst_h = max(1, dst[0]), max(1, dst[1])
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


def _band_base_color(cfg_eu: dict[str, object], kind: str) -> str:
    return _EU_BLUE if kind == BAND_EU else str(cfg_eu.get("bandColor") or _EU_BLUE)


def _render_side_band_art(
    cfg_eu: dict[str, object],
    size: tuple[int, int],
    font_path: Path,
    kind: str,
    code_color: str,
):
    import math

    from PIL import Image, ImageDraw, ImageOps

    band_w, height = size
    full_image_path = str(cfg_eu.get("bandFullImage") or "")

    if full_image_path:
        band = ImageOps.fit(_open_user_image(full_image_path, "Side band image"), (band_w, height))
        draw = ImageDraw.Draw(band)
        return _draw_band_code(cfg_eu, band, draw, font_path, code_color)

    band = Image.new("RGBA", (band_w, height), _band_base_color(cfg_eu, kind))
    draw = ImageDraw.Draw(band)
    if kind == BAND_EU:
        ring_center = (band_w / 2, height * 0.34)
        ring_radius = height * 0.20
        for i in range(12):
            angle = -math.pi / 2 + i * math.pi / 6
            star_center = (ring_center[0] + ring_radius * math.cos(angle), ring_center[1] + ring_radius * math.sin(angle))
            _draw_star(draw, star_center, height * 0.045, "#f7d117")
    else:
        emblem_path = str(cfg_eu.get("bandImage") or "")
        if emblem_path:
            emblem = _open_user_image(emblem_path, "Side band emblem")
            box = (round(band_w * 0.82), round(height * 0.42))
            emblem.thumbnail(box)
            band.alpha_composite(emblem, (round((band_w - emblem.width) / 2), round(height * 0.34 - emblem.height / 2)))

    return _draw_band_code(cfg_eu, band, draw, font_path, code_color)


def _draw_band_code(cfg_eu: dict[str, object], band, draw, font_path: Path, code_color: str):
    """Draw the country/code text on the lower part of the side band."""
    band_w, height = band.size
    code = str(cfg_eu.get("bandCode") or "").strip().upper()[:3]
    if code:
        code_font = _load_font(font_path, max(10, round(height * 0.26)))
        left, top, right, bottom = code_font.getbbox(code)
        draw.text(
            ((band_w - (right - left)) / 2 - left, height * 0.80 - (bottom - top) / 2 - top),
            code, font=code_font, fill=code_color,
        )
    return band


def _render_side_band(cfg_eu: dict[str, object], size: tuple[int, int], font_path: Path, code_color: str):
    """The optional EU/custom side band, composited onto the plain background."""
    from PIL import Image

    width, height = size
    band_w = max(1, round(width * _EU_BAND_FRACTION))
    kind = str(cfg_eu.get("sideBand") or BAND_NONE)
    if kind == BAND_NONE:
        return None

    target = Image.new("RGBA", (band_w, height), _band_base_color(cfg_eu, kind))
    canonical_band = (max(1, round(_CANVAS[_FORMAT_WIDE][0] * _EU_BAND_FRACTION)), _CANVAS[_FORMAT_WIDE][1])
    art_w, art_h = _fit_inside(canonical_band, (band_w, height))
    art = _render_side_band_art(cfg_eu, (art_w, art_h), font_path, kind, code_color)
    target.alpha_composite(art, (round((band_w - art_w) / 2), round((height - art_h) / 2)))
    return target


def _user_background(cfg: dict[str, object], size: tuple[int, int], *, rear: bool = False):
    """The user's front/rear background image scaled to cover `size` (aspect
    preserved, overflowing dimension centre-cropped), or None when that side
    has no image. Sides are independent so an image-free side keeps its
    solid colour."""
    from PIL import ImageOps

    background = cfg.get("background", {})
    if not isinstance(background, dict):
        return None
    path = str(background.get("rearImage" if rear else "frontImage") or "")
    if not path:
        return None
    side = "Rear background" if rear else "Front background"
    return ImageOps.fit(_open_user_image(path, side), size)


def _render_eu_background(cfg_eu: dict[str, object], color: str, size: tuple[int, int], font_path: Path, base=None):
    from PIL import Image

    plate = base if base is not None else Image.new("RGBA", size, str(color))
    band = _render_side_band(cfg_eu, size, font_path, str(color))
    if band is not None:
        plate.alpha_composite(band, (0, 0))
    return plate


def _render_us_background(cfg_us: dict[str, object], size: tuple[int, int], base=None):
    from PIL import Image

    if base is not None:
        return base
    return Image.new("RGBA", size, str(cfg_us.get("bgColor") or "#ffffff"))


def _jp_style(cfg_jp: dict[str, object]) -> tuple[str, str]:
    return JP_STYLES.get(str(cfg_jp.get("style") or "private"), JP_STYLES["private"])


def _render_jp_background(cfg_jp: dict[str, object], size: tuple[int, int], plate_font_path: Path, base=None):
    """JP plate background with the region/classification/kana fields baked in.

    Only the main serial number is live registration text in-game; the other
    logical fields are static per design, which matches how the BeamNG plate
    pipeline renders a single text string.
    """
    from PIL import Image, ImageDraw

    width, height = size
    bg_color, text_color = _jp_style(cfg_jp)
    plate = base if base is not None else Image.new("RGBA", size, bg_color)
    draw = ImageDraw.Draw(plate)

    region = str(cfg_jp.get("region") or "").strip()
    classification = str(cfg_jp.get("classification") or "").strip()[:3]
    kana = str(cfg_jp.get("kana") or "").strip()[:1]

    jp_font_path = resolve_jp_font_path()

    def field_font(text: str, px: int):
        if any(ord(ch) > 0x7F for ch in text):
            if jp_font_path is None:
                raise PlateError(
                    "No Japanese-capable font found on this system; use ASCII text for the region/kana fields."
                )
            return _load_font(jp_font_path, px)
        return _load_font(plate_font_path, px)

    top_line = " ".join(part for part in (region, classification) if part)
    if top_line:
        font = field_font(top_line, max(10, round(height * 0.26)))
        left, top, right, bottom = font.getbbox(top_line)
        draw.text(
            (width * 0.52 - (right - left) / 2 - left, height * 0.185 - (bottom - top) / 2 - top),
            top_line, font=font, fill=text_color,
        )
    if kana:
        font = field_font(kana, max(10, round(height * 0.34)))
        left, top, right, bottom = font.getbbox(kana)
        draw.text(
            (width * 0.115 - (right - left) / 2 - left, height * 0.62 - (bottom - top) / 2 - top),
            kana, font=font, fill=text_color,
        )
    return plate


# --------------------------------------------------------------------------
# Design text metrics


def _text_y_fraction(metrics: _FontMetrics, canvas_h: int, scale: float, center_frac: float) -> float:
    """Solve the design JSON text.y so the caps line lands on center_frac.

    Mirrors the game's plate generator placement:
      lineY = H*ty - lineHeight/2 ; tile top = lineY - (lineHeight - base)
      baseline = tile top + base*scale
    """
    lh, base, cap = metrics.line_height, metrics.base, metrics.cap_height
    target = center_frac * canvas_h
    return (target + (lh - base) + lh * 0.5 - (base - cap * 0.5) * scale) / canvas_h


def _family_text_params(cfg: dict[str, object], fmt: str, metrics: _FontMetrics) -> dict[str, object]:
    """text block (x/y/scale/color/limit) for one format of the active family."""
    size_family = str(cfg.get("size"))
    width, height = _CANVAS[fmt]
    wide = width >= 1024
    limit = max(12, len(active_pattern(cfg)))
    if size_family == PLATE_SIZE_EU:
        eu = cfg["eu"]
        # User offset stacks on the band compensation so 0 always means
        # "centred in the visible plate face".
        tx = (
            0.5
            + max(-0.4, min(0.4, float(eu.get("textX") or 0.0)))
            + (_EU_BAND_FRACTION / 2 if str(eu.get("sideBand")) != BAND_NONE else 0.0)
        )
        color = str(eu.get("textColor") or "#101010")
        if not wide:
            # A wide EU registration on a 2:1 plate wraps onto two lines.
            scale = (height * 0.31) / metrics.cap_height
            line = {
                "x": round(tx, 4),
                "scale": round(scale, 4),
                "maxWidth": round(0.84 if str(eu.get("sideBand")) == BAND_NONE else 0.78, 4),
            }
            return {
                "layout": "two-line",
                "x": round(tx, 4),
                "y": round(_text_y_fraction(metrics, height, scale, 0.5), 4),
                "scale": round(scale, 4),
                "color": color,
                "limit": limit,
                "lines": [
                    {**line, "y": round(_text_y_fraction(metrics, height, scale, 0.34), 4)},
                    {**line, "y": round(_text_y_fraction(metrics, height, scale, 0.70), 4)},
                ],
            }
        scale = (height * 0.60) / metrics.cap_height
        cy = 0.5
    elif size_family == PLATE_SIZE_US:
        us = cfg["us"]
        user_scale = max(0.3, min(2.5, float(us.get("textScale") or 1.0)))
        scale = (height * (0.52 if wide else 0.30)) / metrics.cap_height * user_scale
        tx = 0.5 + max(-0.4, min(0.4, float(us.get("textX") or 0.0)))
        cy = 0.5 + max(-0.4, min(0.4, float(us.get("textY") or 0.0)))
        color = str(us.get("textColor") or "#101010")
    else:
        jp = cfg["jp"]
        scale = (height * (0.50 if wide else 0.38)) / metrics.cap_height
        tx = 0.58
        cy = 0.62
        _bg, color = _jp_style(jp)
    return {
        "x": round(tx, 4),
        "y": round(_text_y_fraction(metrics, height, scale, cy), 4),
        "scale": round(scale, 4),
        "color": color,
        "limit": limit,
    }


def _active_spacing(cfg: dict[str, object]) -> int:
    return _bounded_int(active_section(cfg).get("spacing"), 0, -20, 60)


def _render_background_for(cfg: dict[str, object], fmt: str, font_path: Path, *, rear: bool = False):
    size = _CANVAS[fmt]
    family = str(cfg.get("size"))
    base = _user_background(cfg, size, rear=rear)
    if family == PLATE_SIZE_EU:
        eu = cfg["eu"]
        color = str((eu.get("rearColor") if rear else eu.get("frontColor")) or "#ffffff")
        return _decorate_plate_background(cfg, _render_eu_background(eu, color, size, font_path, base=base))
    if family == PLATE_SIZE_US:
        return _decorate_plate_background(cfg, _render_us_background(cfg["us"], size, base=base))
    return _decorate_plate_background(cfg, _render_jp_background(cfg["jp"], size, font_path, base=base))


def _rear_texture_differs(cfg: dict[str, object]) -> bool:
    """Whether rear formats need their own background texture."""
    background = cfg.get("background", {})
    background = background if isinstance(background, dict) else {}
    front_image = str(background.get("frontImage") or "")
    rear_image = str(background.get("rearImage") or "")
    if front_image or rear_image:
        # An image overrides that side's colour, so any mismatch (including
        # image on one side only) means the rear texture differs.
        return front_image != rear_image
    if str(cfg.get("size")) != PLATE_SIZE_EU:
        return False
    eu = cfg["eu"]
    front = str(eu.get("frontColor") or "#ffffff").lower()
    rear = str(eu.get("rearColor") or "#ffffff").lower()
    return front != rear


# --------------------------------------------------------------------------
# Validation


def validate_plate_config(cfg: object) -> list[str]:
    """User-fixable problems with a plate config; empty list when buildable."""
    cfg = normalized_plate_config(cfg)
    errors: list[str] = []
    pattern_error = validate_pattern(active_pattern(cfg))
    if pattern_error:
        errors.append(pattern_error)
    try:
        font_path = resolve_font_path(cfg.get("font"))
        _load_font(font_path, 32)
    except PlateError as exc:
        errors.append(str(exc))
        font_path = None
    family = str(cfg.get("size"))
    try:
        if str(cfg["background"].get("frontImage") or ""):
            _open_user_image(str(cfg["background"]["frontImage"]), "Front background")
        if str(cfg["background"].get("rearImage") or ""):
            _open_user_image(str(cfg["background"]["rearImage"]), "Rear background")
        if family == PLATE_SIZE_EU and str(cfg["eu"].get("sideBand")) != BAND_NONE and str(cfg["eu"].get("bandFullImage") or ""):
            _open_user_image(str(cfg["eu"]["bandFullImage"]), "Side band image")
        if family == PLATE_SIZE_EU and str(cfg["eu"].get("sideBand")) == BAND_CUSTOM and str(cfg["eu"].get("bandImage") or ""):
            _open_user_image(str(cfg["eu"]["bandImage"]), "Side band emblem")
        if family == PLATE_SIZE_JP and font_path is not None:
            _render_jp_background(cfg["jp"], (256, 128), font_path)
    except PlateError as exc:
        errors.append(str(exc))
    return errors


# --------------------------------------------------------------------------
# Preview rendering (UI helper; approximates the in-game compositor)


def render_plate_preview(
    cfg: object,
    registration: str | None = None,
    *,
    fmt: str | None = None,
    rear: bool = False,
):
    """A PIL image of the primary plate format for the current settings."""
    cfg = normalized_plate_config(cfg)
    family = str(cfg.get("size"))
    if fmt in {_REAR_FORMAT_WIDE, _FORMAT_WIDE}:
        fmt = _FORMAT_WIDE
    elif fmt in {_REAR_FORMAT_2_1, _FORMAT_2_1}:
        fmt = _FORMAT_2_1
    else:
        fmt = _FORMAT_WIDE if family == PLATE_SIZE_EU else _FORMAT_2_1
    font_path = resolve_font_path(cfg.get("font"))
    if registration is None:
        registration = generate_registration(active_pattern(cfg))

    plate = _render_background_for(cfg, fmt, font_path, rear=rear)
    # Same maths as the generated design JSON, just rendered directly.
    font, metrics = _plate_font_metrics(font_path)
    params = _family_text_params(cfg, fmt, metrics)
    spacing = _active_spacing(cfg)

    from PIL import ImageDraw

    draw = ImageDraw.Draw(plate)
    width, height = plate.size
    text = registration[: int(params["limit"])]

    def draw_line(line: str, line_params: dict[str, object]) -> None:
        line_scale = float(line_params.get("scale", params.get("scale", 1.0)))

        def char_advance(ch: str) -> float:
            if ch == _CENTER_DOT_CHAR:
                return max(font.getlength(ch), metrics.cap_height * 0.32)
            return font.getlength(ch)

        advances = [(char_advance(ch) + spacing) * line_scale + 2 for ch in line]
        total = sum(advances)
        max_width = float(line_params.get("maxWidth", params.get("maxWidth", 0)) or 0)
        if max_width and total > width * max_width and total > 0:
            line_scale *= (width * max_width) / total
            advances = [(char_advance(ch) + spacing) * line_scale + 2 for ch in line]
            total = sum(advances)
        draw_font = _load_font(font_path, max(8, round(metrics.font_px * line_scale)))
        x = width * float(line_params.get("x", params.get("x", 0.5))) - total / 2
        cap_box2 = draw_font.getbbox("H")
        caps_center_offset = cap_box2[1] + (cap_box2[3] - cap_box2[1]) / 2
        lh, base, cap = metrics.line_height, metrics.base, metrics.cap_height
        target_center = float(line_params.get("y", params.get("y", 0.5))) * height - (
            (lh - base) + lh * 0.5 - (base - cap * 0.5) * line_scale
        )
        y = target_center - caps_center_offset
        for ch, advance in zip(line, advances):
            if ch == _CENTER_DOT_CHAR:
                diameter = max(2, round((cap_box2[3] - cap_box2[1]) * 0.16))
                center_x = x + advance / 2
                center_y = y + caps_center_offset
                draw.ellipse(
                    (
                        center_x - diameter / 2,
                        center_y - diameter / 2,
                        center_x + diameter / 2,
                        center_y + diameter / 2,
                    ),
                    fill=str(params["color"]),
                )
            else:
                draw.text((x, y), ch, font=draw_font, fill=str(params["color"]))
            x += advance

    if params.get("layout") == "two-line":
        pieces = _split_two_line_text(text)
        line_params = params.get("lines")
        lines = line_params if isinstance(line_params, list) else []
        draw_line(pieces[0], lines[0] if len(lines) > 0 and isinstance(lines[0], dict) else params)
        draw_line(pieces[1], lines[1] if len(lines) > 1 and isinstance(lines[1], dict) else params)
    else:
        draw_line(text, params)
    return plate


# --------------------------------------------------------------------------
# Asset emission


def _design_key(cfg: dict[str, object], font_key: str) -> str:
    hashable = json.dumps({k: v for k, v in cfg.items() if k != "enabled"}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(f"{_ASSET_VERSION}|{font_key}|{hashable}".encode("utf-8"))
    return digest.hexdigest()[:10]


def _save_png(image, path: Path) -> None:
    import io

    from beamxp import hand_drive_core as core

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    core.write_bytes_file(path, buffer.getvalue())  # fs_path handles long paths


def _diffuse_fill_style(cfg: dict[str, object], *, rear: bool = False) -> str:
    family = str(cfg.get("size"))
    if family == PLATE_SIZE_EU:
        eu = cfg["eu"]
        return str((eu.get("rearColor") if rear else eu.get("frontColor")) or "#ffffff")
    if family == PLATE_SIZE_US:
        return str(cfg["us"].get("bgColor") or "#ffffff")
    bg_color, _text_color = _jp_style(cfg["jp"])
    return bg_color


def _plate_generator_html() -> str:
    return r"""<!doctype html>
<html>
<body>
<style>
body {
  margin: 0;
  padding: 0;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  overflow: hidden;
}
#canvas {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
}
</style>
<canvas id="canvas" width="512" height="256"></canvas>
<script>
function localUrl(path) {
  if (!path) return "";
  if (path.indexOf("://") !== -1) return path;
  return "local://local/" + path;
}

function init(mode, text, design) {
  text = (text || "").toUpperCase();
  if (!design || !design.data || !design.data[mode]) {
    finish();
    return;
  }

  var data = design.data;
  var modeData = data[mode];
  var canvas = document.getElementById("canvas");
  canvas.width = data.size.x;
  canvas.height = data.size.y;
  var ctx = canvas.getContext("2d");
  ctx.fillStyle = modeData.fillStyle || "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  var textParams = data.text || {};
  if (textParams.limit) {
    text = text.substring(0, textParams.limit);
  }

  var font = data.characterLayout;
  if (font && font.chars && font.chars.char) {
    font.charMap = {};
    for (var i = 0; i < font.chars.count; i++) {
      for (var key in font.chars.char[i]) {
        font.chars.char[i][key] = parseInt(font.chars.char[i][key]);
      }
      font.charMap[parseInt(font.chars.char[i].id)] = font.chars.char[i];
    }
  }

  var pending = 0;
  var background = null;
  var sprites = null;

  function loadImage(path, callback) {
    if (!path) {
      callback(null);
      return;
    }
    pending++;
    var img = new Image();
    img.addEventListener("load", function() {
      pending--;
      callback(img);
      renderIfReady();
    }, false);
    img.addEventListener("error", function() {
      pending--;
      callback(null);
      renderIfReady();
    }, false);
    img.src = localUrl(path);
  }

  function renderIfReady() {
    if (pending > 0) return;

    if (background) {
      ctx.drawImage(background, 0, 0, background.naturalWidth, background.naturalHeight, 0, 0, canvas.width, canvas.height);
    }

    if (sprites && font && font.charMap) {
      var tintSprites = sprites;
      if (mode === "diffuse") {
        tintSprites = document.createElement("canvas");
        tintSprites.width = sprites.width;
        tintSprites.height = sprites.height;
        var tintCtx = tintSprites.getContext("2d");
        tintCtx.fillStyle = textParams.color || "black";
        tintCtx.fillRect(0, 0, tintSprites.width, tintSprites.height);
        tintCtx.globalCompositeOperation = "multiply";
        tintCtx.drawImage(sprites, 0, 0);
        tintCtx.globalCompositeOperation = "destination-in";
        tintCtx.drawImage(sprites, 0, 0);
      }

      var lineHeight = parseInt(font.common.lineHeight);
      var lineBase = parseInt(font.common.base);

      function splitTwoLine(raw) {
        var source = (raw || "").replace(/^\s+|\s+$/g, "");
        var rawWords = source.split(/\s+/);
        var words = [];
        for (var wi = 0; wi < rawWords.length; wi++) {
          if (rawWords[wi]) words.push(rawWords[wi]);
        }
        if (words.length <= 1) {
          var compact = words.length ? words[0] : source;
          var midpoint = Math.max(1, Math.ceil(compact.length / 2));
          return [compact.substring(0, midpoint), compact.substring(midpoint)];
        }
        var bestIndex = 1;
        var bestScore = 999999;
        for (var si = 1; si < words.length; si++) {
          var left = words.slice(0, si).join(" ");
          var right = words.slice(si).join(" ");
          var score = Math.abs(left.length - right.length) * 100 - left.length;
          if (score < bestScore) {
            bestIndex = si;
            bestScore = score;
          }
        }
        return [words.slice(0, bestIndex).join(" "), words.slice(bestIndex).join(" ")];
      }

      function measureLine(line, scale) {
        var width = 0;
        for (var ci = 0; ci < line.length; ci++) {
          var code = line.charCodeAt(ci);
          if (font.charMap[code] === undefined) continue;
          width += font.charMap[code].xadvance * scale + 2;
        }
        return width;
      }

      function drawLine(line, lineParams) {
        if (!line) return;
        lineParams = lineParams || textParams;
        var scale = lineParams.scale || textParams.scale || 1;
        var maxWidth = lineParams.maxWidth || textParams.maxWidth || 0;
        if (maxWidth) {
          var measured = measureLine(line, scale);
          var available = canvas.width * maxWidth;
          if (measured > available && measured > 0) {
            scale = scale * available / measured;
          }
        }
        var textWidth = measureLine(line, scale);
        var x = canvas.width * (lineParams.x || textParams.x || 0.5) - textWidth * 0.5;
        var y = canvas.height * (lineParams.y || textParams.y || 0.5) - lineHeight * 0.5;
        for (var ti = 0; ti < line.length; ti++) {
          var ch = line.charCodeAt(ti);
          if (font.charMap[ch] === undefined) continue;
          var glyph = font.charMap[ch];
          ctx.drawImage(
            tintSprites,
            glyph.x, glyph.y, glyph.width, glyph.height,
            x + glyph.xoffset * scale,
            y - (lineHeight - lineBase) - glyph.yoffset * scale,
            glyph.width * scale,
            glyph.height * scale
          );
          x += glyph.xadvance * scale + 2;
        }
      }

      if (textParams.layout === "two-line") {
        var pieces = splitTwoLine(text);
        var lineParams = textParams.lines || [];
        drawLine(pieces[0], lineParams[0] || textParams);
        drawLine(pieces[1], lineParams[1] || textParams);
      } else {
        drawLine(text, textParams);
      }
    }

    finish();
  }

  function finish() {
    if (typeof beamng !== "undefined") {
      beamng.uiUpdate();
      beamng.uiDestroy();
    }
  }

  loadImage(modeData.backgroundImg, function(img) { background = img; });
  loadImage(modeData.spriteImg, function(img) { sprites = img; });
  renderIfReady();
}
</script>
</body>
</html>
"""


def _mode_blocks(
    bg_rel: str | None,
    font_rel: str,
    *,
    fill_style: str,
    bump_bg_rel: str | None = None,
    specular_bg_rel: str | None = None,
) -> dict[str, object]:
    diffuse: dict[str, object] = {"spriteImg": f"{font_rel}/platefont_d.png", "fillStyle": fill_style}
    if bg_rel:
        diffuse["backgroundImg"] = bg_rel
    bump = {"spriteImg": f"{font_rel}/platefont_n.png", "fillStyle": "rgb(127,127,255)"}
    if bump_bg_rel:
        bump["backgroundImg"] = bump_bg_rel
    specular = {"spriteImg": f"{font_rel}/platefont_s.png", "fillStyle": "rgb(233,233,233)"}
    if specular_bg_rel:
        specular["backgroundImg"] = specular_bg_rel
    return {
        "diffuse": diffuse,
        "bump": bump,
        "specular": specular,
    }


class _DesignOutput:
    def __init__(self, key: str, part_id: str, rear_formats: dict[str, str], *, rear_differs: bool = False):
        self.key = key
        self.part_id = part_id
        self.rear_formats = rear_formats  # vanilla fmt -> rear fmt id
        self.rear_differs = rear_differs  # rear texture differs from the front


def _emit_design(
    cfg: dict[str, object],
    output_root: Path,
    prefix: str,
    cache: dict[str, _DesignOutput],
    *,
    set_id: str | None = None,
) -> _DesignOutput:
    from beamxp import hand_drive_core as core

    font_path = resolve_font_path(cfg.get("font"))
    glyphs = _pattern_glyphs(active_pattern(cfg))
    spacing = _active_spacing(cfg)
    emboss_strength = _effective_emboss_strength(cfg)
    font_key = _font_atlas_key(font_path, glyphs, spacing, emboss_strength)
    design_key = _design_key(cfg, font_key)
    cache_key = f"{design_key}:{set_id or ''}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    plates_root = output_root / "vehicles" / "common" / "licenseplates" / prefix
    font_rel = f"vehicles/common/licenseplates/{prefix}/font_{font_key}"
    font_dir = plates_root / f"font_{font_key}"
    generator_rel = f"vehicles/common/licenseplates/{prefix}/bhdc-licenseplate.html"
    core.write_text_file(plates_root / "bhdc-licenseplate.html", _plate_generator_html(), encoding="utf-8")
    if not (font_dir / "platefont.json").exists():
        atlas = build_font_atlas(font_path, glyphs, spacing, emboss_strength)
        _save_png(atlas.image_d, font_dir / "platefont_d.png")
        _save_png(atlas.image_n, font_dir / "platefont_n.png")
        _save_png(atlas.image_s, font_dir / "platefont_s.png")
        core.write_text_file(font_dir / "platefont.json", json.dumps(atlas.layout, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        atlas = build_font_atlas(font_path, glyphs, spacing, emboss_strength)

    design_rel = f"vehicles/common/licenseplates/{prefix}/design_{design_key}"
    design_dir = plates_root / f"design_{design_key}"
    # Every design defines the bhdc rear formats, mirroring the front when the
    # rear does not differ. XP vehicles clone their rear plate parts onto these
    # formats, so a design without them would leave the rear plate on the plain
    # white fallback texture as soon as it is selected in-game.
    rear_differs = _rear_texture_differs(cfg)
    rear_formats = {_FORMAT_WIDE: _REAR_FORMAT_WIDE, _FORMAT_2_1: _REAR_FORMAT_2_1}

    formats: dict[str, object] = {}
    for fmt in (_FORMAT_WIDE, _FORMAT_2_1):
        border_maps = _render_border_mode_images(cfg, _CANVAS[fmt], emboss_strength)
        border_n_rel = None
        border_s_rel = None
        if border_maps is not None:
            border_n_name = f"border_{fmt.replace('-', '_')}_n.png"
            border_s_name = f"border_{fmt.replace('-', '_')}_s.png"
            _save_png(border_maps[0], design_dir / border_n_name)
            _save_png(border_maps[1], design_dir / border_s_name)
            border_n_rel = f"{design_rel}/{border_n_name}"
            border_s_rel = f"{design_rel}/{border_s_name}"

        def format_block(bg_stem: str, *, rear: bool) -> dict[str, object]:
            """One format entry: background texture on disk + design JSON block."""
            bg_name = f"bg_{bg_stem.replace('-', '_')}_d.png"
            _save_png(_render_background_for(cfg, fmt, font_path, rear=rear), design_dir / bg_name)
            block: dict[str, object] = {
                "size": {"x": _CANVAS[fmt][0], "y": _CANVAS[fmt][1]},
                "text": _family_text_params(cfg, fmt, atlas.metrics),
                "characterLayout": f"{font_rel}/platefont.json",
                "generator": generator_rel,
            }
            block.update(_mode_blocks(
                f"{design_rel}/{bg_name}",
                font_rel,
                fill_style=_diffuse_fill_style(cfg, rear=rear),
                bump_bg_rel=border_n_rel,
                specular_bg_rel=border_s_rel,
            ))
            return block

        formats[fmt] = format_block(fmt, rear=False)
        rear_fmt = rear_formats.get(fmt)
        if rear_fmt:
            formats[rear_fmt] = format_block(rear_fmt, rear=True)

    design_json = {
        "name": "BeamXP Custom" if set_id is None else f"BeamXP {cfg.get('size')} plate design",
        "version": 2,
        "data": {"format": formats},
    }
    core.write_text_file(design_dir / "licensePlate.json", json.dumps(design_json, indent=2, ensure_ascii=False), encoding="utf-8")

    part_id = f"bhdc_plateset_{_safe_set_id(set_id).replace('-', '_')}" if set_id else f"bhdc_plate_design_{design_key}"
    out = _DesignOutput(design_key, part_id, rear_formats, rear_differs=rear_differs)
    out.design_json_rel = f"{design_rel}/licensePlate.json"
    cache[cache_key] = out
    return out


def _design_part_body(out: _DesignOutput, size_label: str, *, custom: bool = False) -> str:
    return json.dumps({
        "information": {
            "authors": "BeamXP",
            "name": "BeamXP Custom" if custom else f"BeamXP {size_label} Plate Design",
            "value": 0,
        },
        "slotType": _DESIGN_SLOT,
        "licenseplate_path": out.design_json_rel,
    }, indent=4)


# --------------------------------------------------------------------------
# Rear plate part cloning (EU front/rear colour split)

_REAR_NAME_RE = re.compile(r"(?:^|[_\-])(?:r|rear)(?:[_\-]|$)", re.IGNORECASE)
_FRONT_NAME_RE = re.compile(r"(?:^|[_\-])(?:f|front)(?:[_\-]|$)", re.IGNORECASE)


def _looks_rear(slot_type: str, part_id: str, part_body: str) -> bool:
    for name in (slot_type, part_id):
        if _REAR_NAME_RE.search(name):
            return True
        if _FRONT_NAME_RE.search(name):
            return False
    # fall back to the flexbody's global position: +Y is rearwards in BeamNG
    match = re.search(r'"pos"\s*:\s*\{[^}]*"y"\s*:\s*([-+0-9.eE]+)', part_body)
    if match:
        try:
            return float(match.group(1)) > 0.2
        except ValueError:
            return False
    return False


def _clone_rear_plate_part(
    context,
    part_id: str,
    part_body: str,
    new_part_id: str,
    clone_sources: dict[str, _RearMeshClone],
) -> tuple[str, str, list[str]] | None:
    """Clone a rear plate part onto the BeamXP rear plate mesh/format.

    Returns (new_body, rear_format, output_meshes) or None when the part carries no
    recognisable plate mesh.
    """
    meshes = transform_helpers.extract_part_mesh_names(part_body)
    plate_meshes = sorted(m for m in meshes if "licenseplate" in m.lower())
    if not plate_meshes:
        return None
    wide = _part_uses_wide_plate(part_body)
    rear_fmt = _REAR_FORMAT_WIDE if wide else _REAR_FORMAT_2_1
    fallback_mesh = _REAR_MESH_WIDE if wide else _REAR_MESH_2_1
    mesh_map: dict[str, str] = {}
    for mesh in plate_meshes:
        mesh_map[mesh] = _rear_clone_for_mesh(context, mesh, clone_sources) or fallback_mesh

    body = transform_helpers.replace_first(part_body, f'"{part_id}"', f'"{new_part_id}"')
    body = _rewrite_part_mesh_names(body, mesh_map)
    # Real mesh clones already carry the vehicle's intended plate plane. Only
    # nudge the old generic fallback quad, where the source mesh could not be
    # cloned and the replacement may otherwise sit inside the mount.
    if set(mesh_map.values()).issubset(_REAR_FALLBACK_MESHES):
        body = _offset_plate_y(body, 0.01)
    if '"licenseplateFormat"' in body:
        body = re.sub(r'("licenseplateFormat"\s*:\s*)"[^"]*"', rf'\g<1>"{rear_fmt}"', body, count=1)
    else:
        brace = body.find("{")
        if brace < 0:
            return None
        body = body[: brace + 1] + f'\n    "licenseplateFormat": "{rear_fmt}",' + body[brace + 1 :]
    return body, rear_fmt, sorted(set(mesh_map.values()))


def _rewrite_part_mesh_names(part_body: str, mesh_map: dict[str, str]) -> str:
    if not mesh_map:
        return part_body

    body = transform_helpers.replace_array_region(
        part_body,
        "flexbodies",
        lambda array_text: transform_helpers.rewrite_flexbody_meshes(array_text, mesh_map),
    )

    def rewrite_props(array_text: str) -> str:
        out = array_text
        for old_mesh, new_mesh in sorted(mesh_map.items(), key=lambda item: len(item[0]), reverse=True):
            out = re.sub(
                rf'(\[\s*"((?:[^"\\]|\\.)*)"\s*(?:,\s*|\s+))"{re.escape(old_mesh)}"(?=\s*(?:,|"))',
                rf'\1"{new_mesh}"',
                out,
            )
        return out

    return transform_helpers.replace_array_region(body, "props", rewrite_props)


_REAR_QUADS = {
    _REAR_MESH_WIDE: (0.52, 0.11),
    _REAR_MESH_2_1: (0.30, 0.15),
}


@dataclass(frozen=True)
class _RearMeshClone:
    source_zip: Path
    dae_path: str
    source_node_id: str
    source_mesh: str
    output_mesh: str


@dataclass(frozen=True)
class PlatePartOption:
    """One user-facing physical plate choice for a trim side."""

    value: str
    label: str
    part_id: str | None
    mesh: str | None
    format: str | None
    is_default: bool = False


@dataclass(frozen=True)
class _ModelPlatePart:
    part_id: str
    name: str
    body: str
    filename: str
    slot_types: tuple[str, ...]
    rear: bool
    format: str
    meshes: tuple[str, ...]


@dataclass(frozen=True)
class _ConfigPlateSlot:
    slot_type: str
    current_part: str
    current_body: str
    rear: bool


@dataclass(frozen=True)
class _VanillaPlateMesh:
    mesh: str
    format: str


@dataclass(frozen=True)
class _ResolvedPlatePart:
    part: _ModelPlatePart
    mesh: str
    format: str


_JBEAM_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_HIDDEN_PLATE_COORD = 1000.0


def _plate_positions(part_body: str) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    for match in re.finditer(r'"pos"\s*:\s*\{(?P<body>[^}]*)\}', part_body):
        body = match.group("body")
        coords: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            axis_match = re.search(rf'"{axis}"\s*:\s*({_JBEAM_NUMBER})', body)
            if axis_match:
                try:
                    coords[axis] = float(axis_match.group(1))
                except ValueError:
                    pass
        if all(axis in coords for axis in ("x", "y", "z")):
            positions.append((coords["x"], coords["y"], coords["z"]))
    return positions


def _plate_part_is_hidden(part_body: str) -> bool:
    positions = _plate_positions(part_body)
    return bool(positions) and all(max(abs(value) for value in pos) > _HIDDEN_PLATE_COORD for pos in positions)


def _part_plate_width(part_body: str) -> bool | None:
    if re.search(r'"licenseplateFormat"\s*:\s*"52-11"', part_body):
        return True
    if re.search(r'"licenseplateFormat"\s*:\s*"30-15"', part_body):
        return False
    if "52-11" in part_body:
        return True
    if "30-15" in part_body:
        return False
    return None


def _part_uses_wide_plate(part_body: str) -> bool:
    return _part_plate_width(part_body) is True


def _model_plate_part_catalog(context) -> list[_ModelPlatePart]:
    """Return every physical plate part defined by the loaded vehicle model.

    Shared BeamNG plate *meshes* are intentionally allowed, but shared/common
    JBeam parts are not: their slot types and placement groups belong to other
    models.  Parts from any slot family in this model remain eligible so a
    donor can be cloned into the trim's active plate slot at build time.
    """
    from beamxp import hand_drive_core as core

    cached = getattr(context, "_bhdc_model_plate_parts", None)
    if isinstance(cached, list):
        return cached

    vehicle_prefix = str(context.vehicle_path).replace("\\", "/").rstrip("/").lower() + "/"
    catalog: list[_ModelPlatePart] = []
    for part_id, (body, filename) in context.part_body_index.items():
        filename_text = str(filename).replace("\\", "/")
        filename_lower = filename_text.lower()
        # Hand-built/test contexts use short filenames; real BeamNG paths are
        # constrained to the selected vehicle rather than vehicles/common or
        # another vehicle archive.
        if filename_lower.startswith("vehicles/") and not filename_lower.startswith(vehicle_prefix):
            continue
        slot_types = tuple(
            sorted(
                slot
                for slot in transform_helpers.extract_part_slot_types(body)
                if "licenseplate" in slot.lower() and "design" not in slot.lower()
            )
        )
        if not slot_types:
            continue
        meshes = tuple(
            sorted(
                mesh
                for mesh in transform_helpers.extract_part_mesh_names(body)
                if "licenseplate" in mesh.lower()
            )
        )
        if not meshes and "licenseplateformat" not in body.lower():
            continue
        if _plate_part_is_hidden(body):
            continue
        width = _part_plate_width(body)
        fmt = _FORMAT_WIDE if width is True else _FORMAT_2_1
        name = core.part_information_name(body) or str(part_id)
        for rear in sorted({_looks_rear(slot, str(part_id), body) for slot in slot_types}):
            side_slots = tuple(
                slot for slot in slot_types if _looks_rear(slot, str(part_id), body) == rear
            )
            if not side_slots:
                continue
            catalog.append(
                _ModelPlatePart(
                    part_id=str(part_id),
                    name=name,
                    body=body,
                    filename=filename_text,
                    slot_types=side_slots,
                    rear=rear,
                    format=fmt,
                    meshes=meshes,
                )
            )
    catalog.sort(key=lambda part: (part.rear, part.name.lower(), part.part_id.lower()))
    setattr(context, "_bhdc_model_plate_parts", catalog)
    return catalog


_VANILLA_PLATE_MESH_RE = re.compile(
    r"^licenseplate(?:-52-11(?:-(?:r\d+(?:_\d+)?|b\d+))?)?$",
    re.IGNORECASE,
)


def _vanilla_plate_mesh_sort_key(item: _VanillaPlateMesh) -> tuple[object, ...]:
    if item.format == _FORMAT_2_1:
        return (0, 0, item.mesh.lower())
    suffix = item.mesh.lower().partition("licenseplate-52-11-")[2]
    if not suffix:
        return (1, 0, item.mesh.lower())
    if suffix.startswith("r"):
        try:
            radius = float(suffix[1:].replace("_", "."))
        except ValueError:
            radius = 0.0
        return (1, 1, -radius, item.mesh.lower())
    return (1, 2, suffix)


def _vanilla_plate_mesh_catalog(context) -> list[_VanillaPlateMesh]:
    """Discover the shared physical plate meshes from BeamNG's common.zip."""
    from beamxp import hand_drive_core as core

    cached = getattr(context, "_bhdc_vanilla_plate_meshes", None)
    if isinstance(cached, list):
        return cached

    formats_by_mesh: dict[str, str] = {}
    for candidate_zip in core.common_zip_candidates(context.source_zip):
        for dae_path in core.common_dae_paths(candidate_zip):
            path_lower = dae_path.lower()
            if "licenseplate" not in path_lower and not path_lower.endswith("/empty.dae"):
                continue
            try:
                objects = core.list_dae_objects_for_file(candidate_zip, dae_path)
            except Exception:
                continue
            for object_id, obj in objects.items():
                for mesh in (str(object_id), str(obj.name)):
                    if not _VANILLA_PLATE_MESH_RE.fullmatch(mesh):
                        continue
                    formats_by_mesh.setdefault(
                        mesh,
                        _FORMAT_WIDE if mesh.lower().startswith("licenseplate-52-11") else _FORMAT_2_1,
                    )
                    # Rear-colour splitting clones the selected geometry into
                    # a BeamXP-owned DAE/material.  Register newly discovered
                    # common meshes so that path can preserve every curvature,
                    # including shapes this model did not originally use.
                    context.objects.setdefault(mesh, obj)

    # A community model can ship a physical plate mesh outside common.zip.
    # Keep those model-local options available alongside the shared catalogue.
    for part in _model_plate_part_catalog(context):
        for mesh in part.meshes:
            formats_by_mesh.setdefault(mesh, part.format)

    catalog = [
        _VanillaPlateMesh(mesh=mesh, format=fmt)
        for mesh, fmt in formats_by_mesh.items()
    ]
    catalog.sort(key=_vanilla_plate_mesh_sort_key)
    setattr(context, "_bhdc_vanilla_plate_meshes", catalog)
    return catalog


def _plate_slots_by_side_for_config(context, config_name: str) -> dict[str, list[_ConfigPlateSlot]]:
    """Find the active front/rear plate slots and their stock selections.

    Unlike the old detector this retains explicitly empty slots, which is how
    trims such as the Sunburst's US configurations express "no front plate".
    """
    from beamxp import hand_drive_core as core

    cache = getattr(context, "_bhdc_plate_slots_by_config", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(context, "_bhdc_plate_slots_by_config", cache)
    cached = cache.get(config_name)
    if isinstance(cached, dict):
        return cached

    variant = context.variants.get(config_name)
    if variant is None:
        result = {"front": [], "rear": []}
        cache[config_name] = result
        return result

    pc = core.load_pc(context.source_zip, variant.pc_path)
    explicit = {
        str(slot_type): str(part_id or "")
        for slot_type, part_id in dict(pc.get("parts", {})).items()
    }
    explicit_plate_slots = {
        slot: part
        for slot, part in explicit.items()
        if "licenseplate" in slot.lower() and "design" not in slot.lower()
    }
    if explicit_plate_slots:
        # Vanilla .pc files normally spell out both physical plate slots,
        # including an empty string for no plate.  This path avoids resolving
        # the trim's complete slot graph merely to rediscover those two keys.
        current_by_slot = explicit_plate_slots
    else:
        selected = core.selected_parts_for_config(context, config_name)
        selected_by_slot = selected.get("selected_by_slot", {})
        selected_by_slot = selected_by_slot if isinstance(selected_by_slot, dict) else {}

        current_by_slot: dict[str, str] = {}
        # Capture empty defaults declared by the active body/bumper as well as
        # non-empty defaults already resolved by selected_parts_for_config().
        for selected_part in sorted(str(part) for part in selected.get("parts", set())):
            found = core.part_body_for_context(context, selected_part)
            if found is None:
                continue
            for slot_def in core.extract_slot_defs(found[0]):
                current_by_slot.setdefault(slot_def.slot_type, slot_def.default_part)
        current_by_slot.update({str(slot): str(part or "") for slot, part in selected_by_slot.items()})

    catalog = _model_plate_part_catalog(context)
    result: dict[str, list[_ConfigPlateSlot]] = {"front": [], "rear": []}
    for slot_type, current_part in sorted(current_by_slot.items()):
        slot_lower = slot_type.lower()
        if "licenseplate" not in slot_lower or "design" in slot_lower:
            continue
        found = core.part_body_for_context(context, current_part) if current_part else None
        current_body = found[0] if found is not None else ""
        rear = _looks_rear(slot_type, current_part, current_body)
        # Generic/empty slot names have no body position to inspect.  Let the
        # model catalogue decide when all parts for that slot agree on a side.
        if not current_part and not _REAR_NAME_RE.search(slot_type) and not _FRONT_NAME_RE.search(slot_type):
            catalog_sides = {part.rear for part in catalog if slot_type in part.slot_types}
            if len(catalog_sides) == 1:
                rear = catalog_sides.pop()
        side = "rear" if rear else "front"
        result[side].append(_ConfigPlateSlot(slot_type, current_part, current_body, rear))
    cache[config_name] = result
    return result


def _iter_selected_plate_parts(
    context,
    config_name: str,
    parts_override: dict[str, object] | None = None,
    generated_part_bodies: dict[str, str] | None = None,
):
    """Yield (slot_type, part_id, part_body) for a config's licence plate
    parts, skipping the design slot and parts without a resolvable body."""
    from beamxp import hand_drive_core as core

    selected = core.selected_parts_for_config(context, config_name)
    selected_by_slot = selected.get("selected_by_slot", {}) if isinstance(selected, dict) else {}
    if not isinstance(selected_by_slot, dict):
        return
    resolved_slots = dict(selected_by_slot)
    if isinstance(parts_override, dict):
        for slot_type, part_id in parts_override.items():
            slot_text = str(slot_type)
            part_text = str(part_id or "")
            if "licenseplate" in slot_text.lower() or "licenseplate" in part_text.lower():
                resolved_slots[slot_text] = part_text
    for slot_type, part_id in sorted(resolved_slots.items()):
        slot_text = str(slot_type)
        part_text = str(part_id)
        if not part_text:
            continue
        slot_lower = slot_text.lower()
        part_lower = part_text.lower()
        if "design" in slot_lower or "design" in part_lower:
            continue
        if "licenseplate" not in slot_lower and "licenseplate" not in part_lower:
            continue
        found = core.part_body_for_context(context, part_text)
        body = found[0] if found is not None else None
        if body is None and isinstance(generated_part_bodies, dict):
            body = generated_part_bodies.get(part_text)
        if body is None:
            continue
        yield slot_text, part_text, body


def _plate_slots_for_side(context, config_name: str, side: str) -> list[tuple[str, str, str]]:
    side_key = "rear" if str(side).lower() == "rear" else "front"
    return [
        (slot.slot_type, slot.current_part, slot.current_body)
        for slot in _plate_slots_by_side_for_config(context, config_name)[side_key]
    ]


def _plate_mesh_label(mesh: str, fmt: str) -> str:
    if fmt == _FORMAT_2_1:
        return "US/JP"
    suffix = mesh.lower().partition("licenseplate-52-11-")[2]
    if not suffix:
        return "EU Flat"
    elif suffix.startswith("r"):
        return f"EU R{suffix[1:].replace('_', '.')}"
    elif suffix.startswith("b"):
        return f"EU {suffix[1:]}°"
    return f"EU {suffix.upper()}"


def _model_plate_part_by_id(context, part_id: str, *, rear: bool) -> _ModelPlatePart | None:
    return next(
        (part for part in _model_plate_part_catalog(context) if part.part_id == part_id and part.rear == rear),
        None,
    )


def _plate_part_base_label(part: _ModelPlatePart) -> str:
    return _plate_mesh_label(part.meshes[0], part.format) if part.meshes else part.name


def plate_part_choices_for_config(context, config_name: str, side: str) -> list[PlatePartOption]:
    """Shared vanilla meshes, applied through this model's plate JBeam parts."""
    rear = str(side).lower() == "rear"
    slots = _plate_slots_for_side(context, config_name, side)
    model_catalog = [part for part in _model_plate_part_catalog(context) if part.rear == rear]
    raw_labels = [_plate_part_base_label(part) for part in model_catalog]
    duplicate_labels = {label for label in raw_labels if raw_labels.count(label) > 1}

    def display(part: _ModelPlatePart) -> str:
        label = _plate_part_base_label(part)
        return f"{label} [{part.part_id}]" if label in duplicate_labels else label

    default_ids = list(dict.fromkeys(part_id for _slot, part_id, _body in slots if part_id))
    if not slots:
        default_label = "No plate slot (default)"
        default_format = None
        default_part = None
        default_mesh = None
    elif not default_ids:
        default_label = "None (default)"
        default_format = None
        default_part = None
        default_mesh = None
    elif len(default_ids) == 1:
        default_part = default_ids[0]
        record = _model_plate_part_by_id(context, default_part, rear=rear)
        if record is not None:
            default_format = record.format
            default_mesh = record.meshes[0] if record.meshes else None
            default_label = (
                f"{_plate_mesh_label(default_mesh, default_format)} (default)"
                if default_mesh
                else f"{display(record)} (default)"
            )
        else:
            body = next((body for _slot, part_id, body in slots if part_id == default_part), "")
            default_label = f"{default_part} (default)"
            default_format = _FORMAT_WIDE if _part_uses_wide_plate(body) else _FORMAT_2_1
            default_mesh = next(
                (
                    mesh
                    for mesh in transform_helpers.extract_part_mesh_names(body)
                    if "licenseplate" in mesh.lower()
                ),
                None,
            )
            if default_mesh:
                default_label = f"{_plate_mesh_label(default_mesh, default_format)} (default)"
    else:
        default_part = None
        default_label = "Current plate combination (default)"
        default_format = None
        default_mesh = None

    choices = [
        PlatePartOption(
            value=PLATE_PART_AUTO,
            label=default_label,
            part_id=default_part,
            mesh=default_mesh,
            format=default_format,
            is_default=True,
        )
    ]
    default_meshes = {
        mesh
        for _slot, part_id, body in slots
        if part_id
        for mesh in transform_helpers.extract_part_mesh_names(body)
        if "licenseplate" in mesh.lower()
    }
    if slots and model_catalog:
        for plate_mesh in _vanilla_plate_mesh_catalog(context):
            if plate_mesh.mesh in default_meshes:
                continue
            choices.append(
                PlatePartOption(
                    value=f"{PLATE_MESH_CHOICE_PREFIX}{plate_mesh.mesh}",
                    label=_plate_mesh_label(plate_mesh.mesh, plate_mesh.format),
                    part_id=None,
                    mesh=plate_mesh.mesh,
                    format=plate_mesh.format,
                )
            )
    if slots and default_ids:
        choices.append(
            PlatePartOption(
                value=PLATE_PART_NONE,
                label="None",
                part_id=None,
                mesh=None,
                format=None,
            )
        )
    return choices


def plate_part_options_for_config(context, config_name: str, side: str) -> list[str]:
    return [choice.value for choice in plate_part_choices_for_config(context, config_name, side)]


def plate_part_label_for_config(context, config_name: str, side: str, value: object) -> str:
    normalized = normalized_plate_part_choice(value)
    choices = plate_part_choices_for_config(context, config_name, side)
    for choice in choices:
        if choice.value == normalized:
            return choice.label
    rear = str(side).lower() == "rear"
    record = _model_plate_part_by_id(context, normalized, rear=rear)
    if record is not None:
        default_choice = choices[0] if choices else None
        if default_choice is not None and default_choice.part_id == normalized:
            return default_choice.label
        return _plate_part_base_label(record)
    if normalized in {PLATE_PART_2_1, PLATE_PART_WIDE}:
        slots = _plate_slots_for_side(context, config_name, side)
        if slots:
            slot_type, current_part, _body = slots[0]
            candidate = _best_plate_candidate(context, slot_type, current_part, normalized, rear=rear)
            if candidate is not None:
                candidate_record = _model_plate_part_by_id(context, candidate, rear=rear)
                if candidate_record is not None:
                    default_choice = choices[0] if choices else None
                    if default_choice is not None and default_choice.part_id == candidate:
                        return default_choice.label
                    return _plate_part_base_label(candidate_record)
        return "EU wide (legacy)" if normalized == PLATE_PART_WIDE else "US 2:1 (legacy)"
    return f"Missing plate: {normalized}"


def normalized_plate_part_choice(value: object) -> str:
    if not isinstance(value, str):
        return PLATE_PART_AUTO
    text = value.strip()
    if not text or text == "default":
        return PLATE_PART_AUTO
    return text


def variant_has_plate_changes(conversion: dict[str, object], config_name: str, context=None) -> bool:
    variants = conversion.get("variants", {})
    settings = variants.get(config_name) if isinstance(variants, dict) else None
    front = normalized_plate_part_choice(settings.get("frontPlate")) if isinstance(settings, dict) else PLATE_PART_AUTO
    rear = normalized_plate_part_choice(settings.get("rearPlate")) if isinstance(settings, dict) else PLATE_PART_AUTO
    if effective_plate_config(conversion, config_name) is not None:
        return True
    if context is None:
        return front != PLATE_PART_AUTO or rear != PLATE_PART_AUTO
    return _plate_part_choice_changes(context, config_name, "front", front) or _plate_part_choice_changes(
        context, config_name, "rear", rear
    )


def _plate_part_choice_changes(context, config_name: str, side: str, value: str) -> bool:
    choice = normalized_plate_part_choice(value)
    if choice == PLATE_PART_AUTO:
        return False
    slots = _plate_slots_for_side(context, config_name, side)
    if not slots:
        return False
    if choice == PLATE_PART_NONE:
        return any(current_part for _slot, current_part, _body in slots)
    rear = str(side).lower() == "rear"
    for slot_type, current_part, _body in slots:
        candidate = _requested_plate_part(context, slot_type, current_part, choice, rear=rear)
        if candidate is None:
            return True
        current_meshes = {
            mesh
            for mesh in transform_helpers.extract_part_mesh_names(_body)
            if "licenseplate" in mesh.lower()
        }
        if (
            candidate.part.part_id != current_part
            or slot_type not in candidate.part.slot_types
            or candidate.mesh not in current_meshes
        ):
            return True
    return False


def _best_plate_candidate(
    context,
    slot_type: str,
    current_part: str,
    wanted_format: str,
    *,
    rear: bool,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    current_prefix = _plate_part_prefix(current_part)
    for part in _model_plate_part_catalog(context):
        if part.rear != rear or part.format != wanted_format:
            continue
        score = 0
        if slot_type in part.slot_types:
            score += 100
        if current_prefix and _plate_part_prefix(part.part_id) == current_prefix:
            score += 20
        if "wide" in part.part_id.lower() and wanted_format == _FORMAT_WIDE:
            score += 3
        candidates.append((-score, part.part_id))
    return min(candidates)[1] if candidates else None


def _requested_plate_part(
    context,
    slot_type: str,
    current_part: str,
    choice: str,
    *,
    rear: bool,
) -> _ResolvedPlatePart | None:
    selected_mesh: str | None = None
    selected_format: str | None = None
    if choice.startswith(PLATE_MESH_CHOICE_PREFIX):
        selected_mesh = choice[len(PLATE_MESH_CHOICE_PREFIX) :]
        mesh_record = next(
            (record for record in _vanilla_plate_mesh_catalog(context) if record.mesh == selected_mesh),
            None,
        )
        if mesh_record is None:
            return None
        selected_format = mesh_record.format
        exact = [
            part
            for part in _model_plate_part_catalog(context)
            if part.rear == rear and selected_mesh in part.meshes
        ]
        if exact:
            exact.sort(
                key=lambda part: (
                    0 if slot_type in part.slot_types else 1,
                    0 if part.part_id == current_part else 1,
                    part.part_id,
                )
            )
            return _ResolvedPlatePart(exact[0], selected_mesh, selected_format)

        # The mesh is global but unused by this model.  Reuse the closest
        # model-specific placement and swap only its mesh/format.
        donor_id = _best_plate_candidate(
            context,
            slot_type,
            current_part,
            selected_format,
            rear=rear,
        )
        donor = _model_plate_part_by_id(context, donor_id or current_part, rear=rear)
        if donor is None:
            donor = next(
                (part for part in _model_plate_part_catalog(context) if part.rear == rear),
                None,
            )
        return _ResolvedPlatePart(donor, selected_mesh, selected_format) if donor is not None else None

    if choice in {PLATE_PART_2_1, PLATE_PART_WIDE}:
        part_id = _best_plate_candidate(context, slot_type, current_part, choice, rear=rear)
        part = _model_plate_part_by_id(context, part_id or "", rear=rear)
    else:
        part = _model_plate_part_by_id(context, choice, rear=rear)
    if part is None:
        return None
    mesh = part.meshes[0] if part.meshes else ""
    return _ResolvedPlatePart(part, mesh, part.format) if mesh else None


def _plate_part_alias_id(part_id: str, slot_type: str, mesh: str) -> str:
    from beamxp import hand_drive_core as core

    digest = hashlib.sha1(f"{slot_type}|{mesh}".encode("utf-8", errors="replace")).hexdigest()[:8]
    return core.safe_id(f"bhdc_plate_{part_id}_{digest}")


def _clone_plate_part_for_slot(
    part: _ModelPlatePart,
    slot_type: str,
    new_part_id: str,
    mesh: str,
    fmt: str,
) -> str:
    body = transform_helpers.replace_first(part.body, f'"{part.part_id}"', f'"{new_part_id}"')
    body = _force_slot_type(body, slot_type)
    body = _rewrite_part_mesh_names(body, {source: mesh for source in part.meshes})
    return _force_licenseplate_format(body, fmt)


def _apply_plate_part_choices(
    context,
    config_name: str,
    parts: dict[str, object],
    front_choice: str,
    rear_choice: str,
    generated_part_bodies: dict[str, str],
    summary: dict[str, object],
) -> bool:
    changed = False
    for side, choice in (("front", front_choice), ("rear", rear_choice)):
        choice = normalized_plate_part_choice(choice)
        if choice == PLATE_PART_AUTO:
            continue
        is_rear = side == "rear"
        slots = _plate_slots_for_side(context, config_name, side)
        if not slots:
            summary["warnings"].append(f"{config_name}: no {side} licence plate slot found")
            continue
        for slot_type, current_part, _body in slots:
            if choice == PLATE_PART_NONE:
                parts[slot_type] = ""
                changed = changed or bool(current_part)
                continue
            candidate = _requested_plate_part(
                context,
                slot_type,
                current_part,
                choice,
                rear=is_rear,
            )
            if candidate is None:
                summary["warnings"].append(
                    f"{config_name}: model plate part '{choice}' is unavailable for the {side}; keeping {current_part or 'None'}"
                )
                continue
            donor = candidate.part
            selected_part = donor.part_id
            if slot_type not in donor.slot_types or candidate.mesh not in donor.meshes:
                selected_part = _plate_part_alias_id(donor.part_id, slot_type, candidate.mesh)
                if selected_part not in generated_part_bodies:
                    generated_part_bodies[selected_part] = _clone_plate_part_for_slot(
                        donor,
                        slot_type,
                        selected_part,
                        candidate.mesh,
                        candidate.format,
                    )
                    summary["physicalPartsCloned"] = int(summary.get("physicalPartsCloned", 0)) + 1
            parts[slot_type] = selected_part
            changed = changed or selected_part != current_part
    return changed


def preview_pc_with_plate_parts(
    context,
    conversion: dict[str, object],
    config_name: str,
) -> tuple[dict[str, object], dict[str, str]]:
    """Resolve the physical plate selections without writing a build.

    The ModernGL preview uses the returned PC for both converted and original
    layouts, so its layout toggle never silently restores the source plates.
    Generated alias JBeam bodies are returned alongside it for part traversal.
    Plate-design textures and rear-colour clones do not alter preview geometry
    and are therefore intentionally omitted here.
    """
    from beamxp import hand_drive_core as core

    variant = context.variants[config_name]
    pc = core.load_pc(context.source_zip, variant.pc_path)
    parts = dict(pc.get("parts", {}))
    variants = conversion.get("variants", {})
    settings = variants.get(config_name, {}) if isinstance(variants, dict) else {}
    if not isinstance(settings, dict):
        settings = {}
    generated_part_bodies: dict[str, str] = {}
    summary: dict[str, object] = {"warnings": [], "physicalPartsCloned": 0}
    _apply_plate_part_choices(
        context,
        config_name,
        parts,
        normalized_plate_part_choice(settings.get("frontPlate")),
        normalized_plate_part_choice(settings.get("rearPlate")),
        generated_part_bodies,
        summary,
    )
    pc["parts"] = parts
    return pc, generated_part_bodies


def preview_format_for_config(
    context,
    config_name: str,
    side: str = "front",
    choice: object = PLATE_PART_AUTO,
) -> str | None:
    """Return the chosen physical plate format used by one trim side."""
    if context is None or not config_name:
        return None
    normalized = normalized_plate_part_choice(choice)
    if normalized == PLATE_PART_NONE:
        return None
    want_rear = str(side).lower() == "rear"
    slots = _plate_slots_for_side(context, config_name, side)
    if normalized in {PLATE_PART_2_1, PLATE_PART_WIDE}:
        return normalized
    if normalized.startswith(PLATE_MESH_CHOICE_PREFIX):
        selected_mesh = normalized[len(PLATE_MESH_CHOICE_PREFIX) :]
        record = next(
            (item for item in _vanilla_plate_mesh_catalog(context) if item.mesh == selected_mesh),
            None,
        )
        return record.format if record is not None else None
    if normalized != PLATE_PART_AUTO:
        record = _model_plate_part_by_id(context, normalized, rear=want_rear)
        return record.format if record is not None else None

    try:
        plate_parts = [slot for slot in slots if slot[1]]
    except Exception:
        return None
    for slot_text, part_text, part_body in plate_parts:
        if want_rear and _plate_part_is_hidden(part_body):
            donor = _visible_rear_plate_donor(context, slot_text, part_text, part_body)
            if donor is not None:
                _donor_id, part_body = donor
        fmt = _FORMAT_WIDE if _part_plate_width(part_body) else _FORMAT_2_1
        return fmt
    return None


def _plate_part_prefix(part_id: str) -> str:
    lowered = part_id.lower()
    idx = lowered.find("licenseplate")
    if idx >= 0:
        return lowered[:idx].strip("_-")
    return lowered.split("_", 1)[0]


def _force_slot_type(part_body: str, slot_type: str) -> str:
    encoded = json.dumps(slot_type)
    return re.sub(
        r'("slotType"\s*:\s*)(?:"(?:[^"\\]|\\.)*"|\[[^\]]*\])',
        lambda match: match.group(1) + encoded,
        part_body,
        count=1,
    )


def _force_licenseplate_format(part_body: str, fmt: str) -> str:
    encoded = json.dumps(fmt)
    if re.search(r'"licenseplateFormat"\s*:', part_body):
        return re.sub(
            r'("licenseplateFormat"\s*:\s*)"(?:[^"\\]|\\.)*"',
            lambda match: match.group(1) + encoded,
            part_body,
            count=1,
        )
    brace = part_body.find("{")
    if brace < 0:
        return part_body
    return part_body[: brace + 1] + f'\n    "licenseplateFormat": {encoded},' + part_body[brace + 1 :]


def _offset_plate_y(part_body: str, amount: float) -> str:
    def replace_pos(match: re.Match[str]) -> str:
        body = match.group("body")

        def replace_y(y_match: re.Match[str]) -> str:
            try:
                value = float(y_match.group(1)) + amount
            except ValueError:
                return y_match.group(0)
            return f'"y":{value:.5g}'

        body = re.sub(r'"y"\s*:\s*(' + _JBEAM_NUMBER + r')', replace_y, body, count=1)
        return '"pos":{' + body + "}"

    return re.sub(r'"pos"\s*:\s*\{(?P<body>[^}]*)\}', replace_pos, part_body)


def _rear_clone_mesh_name(context, source_mesh: str, obj) -> str:
    from beamxp import hand_drive_core as core

    source_zip = obj.dae_source_zip or context.source_zip
    digest = hashlib.sha1(
        f"{source_zip}|{obj.dae_path}|{obj.id}|{source_mesh}".encode("utf-8", errors="replace")
    ).hexdigest()[:8]
    return core.safe_id(f"bhdc_rear_{source_mesh}_{digest}")


def _rear_clone_for_mesh(
    context,
    source_mesh: str,
    clone_sources: dict[str, _RearMeshClone],
) -> str | None:
    obj = context.objects.get(source_mesh)
    if obj is None or not obj.dae_path:
        return None
    output_mesh = _rear_clone_mesh_name(context, source_mesh, obj)
    if output_mesh not in clone_sources:
        clone_sources[output_mesh] = _RearMeshClone(
            source_zip=obj.dae_source_zip or context.source_zip,
            dae_path=obj.dae_path,
            source_node_id=obj.id,
            source_mesh=source_mesh,
            output_mesh=output_mesh,
        )
    return output_mesh


def _namespace_tag(tag: str) -> str:
    return f"{{{transform_helpers.NS['c']}}}{tag}"


def _clear_children(elem: ET.Element) -> None:
    for child in list(elem):
        elem.remove(child)


def _find_or_create_library(root: ET.Element, tag: str) -> ET.Element:
    found = root.find(f"c:{tag}", transform_helpers.NS)
    if found is not None:
        return found
    created = ET.Element(_namespace_tag(tag))
    root.append(created)
    return created


def _append_plate_effect(library_effects: ET.Element, output_mesh: str) -> None:
    effect = ET.SubElement(library_effects, _namespace_tag("effect"), {"id": f"{output_mesh}-effect"})
    profile = ET.SubElement(effect, _namespace_tag("profile_COMMON"))
    technique = ET.SubElement(profile, _namespace_tag("technique"), {"sid": "common"})
    lambert = ET.SubElement(technique, _namespace_tag("lambert"))
    diffuse = ET.SubElement(lambert, _namespace_tag("diffuse"))
    color = ET.SubElement(diffuse, _namespace_tag("color"))
    color.text = "0.8 0.8 0.8 1"


def _append_plate_material(library_materials: ET.Element, output_mesh: str) -> None:
    material = ET.SubElement(
        library_materials,
        _namespace_tag("material"),
        {"id": f"{output_mesh}-material", "name": output_mesh},
    )
    ET.SubElement(material, _namespace_tag("instance_effect"), {"url": f"#{output_mesh}-effect"})


def _set_geometry_material(geometry: ET.Element, material_id: str) -> None:
    for tag in ("triangles", "polylist", "polygons"):
        for primitive in geometry.findall(f".//c:{tag}", transform_helpers.NS):
            primitive.set("material", material_id)


def _set_instance_materials(node: ET.Element, material_id: str) -> None:
    for instance_material in node.findall(".//c:instance_material", transform_helpers.NS):
        instance_material.set("symbol", material_id)
        instance_material.set("target", f"#{material_id}")


def _rear_clone_dae_name(source_zip: Path, dae_path: str) -> str:
    digest = hashlib.sha1(f"{source_zip}|{dae_path}".encode("utf-8", errors="replace")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(dae_path).stem).strip("._-") or "source"
    return f"bhdc_plate_rear_{stem}_{digest}.dae"


def _write_rear_clone_daes(
    context,
    output_vehicle_dir: Path,
    clone_sources: dict[str, _RearMeshClone],
    summary: dict[str, object],
) -> None:
    from beamxp import hand_drive_core as core

    grouped: dict[tuple[Path, str], list[_RearMeshClone]] = {}
    for clone in clone_sources.values():
        grouped.setdefault((clone.source_zip, clone.dae_path), []).append(clone)

    for (source_zip, dae_path), clones in sorted(grouped.items(), key=lambda item: (str(item[0][0]).lower(), item[0][1].lower())):
        tree = core.parse_dae(source_zip, dae_path)
        root = tree.getroot()
        library_geometries = root.find("c:library_geometries", transform_helpers.NS)
        library_visual_scenes = root.find("c:library_visual_scenes", transform_helpers.NS)
        if library_geometries is None or library_visual_scenes is None:
            summary["warnings"].append(f"Could not clone rear plate mesh from {dae_path}: DAE has no geometry scene")
            continue
        library_materials = _find_or_create_library(root, "library_materials")
        library_effects = _find_or_create_library(root, "library_effects")

        geometries_by_id = {
            geom.get("id"): geom
            for geom in library_geometries.findall("c:geometry", transform_helpers.NS)
            if geom.get("id")
        }
        cloned_nodes: list[ET.Element] = []
        cloned_geometries: dict[str, ET.Element] = {}
        material_meshes: set[str] = set()

        for clone in sorted(clones, key=lambda item: item.output_mesh):
            source_node = core.find_dae_node(root, clone.source_node_id)
            if source_node is None:
                summary["warnings"].append(
                    f"Could not clone rear plate mesh '{clone.source_mesh}': node missing in {Path(dae_path).name}"
                )
                continue
            new_node = copy.deepcopy(source_node)
            new_node.set("id", clone.output_mesh)
            new_node.set("name", clone.output_mesh)
            material_id = f"{clone.output_mesh}-material"
            _set_instance_materials(new_node, material_id)

            for inst in new_node.findall(".//c:instance_geometry", transform_helpers.NS):
                url = inst.get("url", "")
                if not url.startswith("#"):
                    continue
                old_geom_id = url[1:]
                old_geom = geometries_by_id.get(old_geom_id)
                if old_geom is None:
                    continue
                new_geom_id = core.safe_id(f"{old_geom_id}_{clone.output_mesh}")
                if new_geom_id not in cloned_geometries:
                    new_geom = transform_helpers.copied_geometry(old_geom, new_geom_id)
                    _set_geometry_material(new_geom, material_id)
                    cloned_geometries[new_geom_id] = new_geom
                inst.set("url", f"#{new_geom_id}")
                if inst.get("name"):
                    inst.set("name", clone.output_mesh)

            if new_node.findall(".//c:instance_geometry", transform_helpers.NS):
                cloned_nodes.append(new_node)
                material_meshes.add(clone.output_mesh)

        if not cloned_nodes or not cloned_geometries:
            continue

        _clear_children(library_geometries)
        for geometry in cloned_geometries.values():
            library_geometries.append(geometry)
        _clear_children(library_materials)
        _clear_children(library_effects)
        for output_mesh in sorted(material_meshes):
            _append_plate_effect(library_effects, output_mesh)
            _append_plate_material(library_materials, output_mesh)
        for visual_scene in library_visual_scenes.findall("c:visual_scene", transform_helpers.NS):
            _clear_children(visual_scene)
            for node in cloned_nodes:
                visual_scene.append(node)

        core.write_xml_tree(tree, output_vehicle_dir / _rear_clone_dae_name(source_zip, dae_path))


def _rear_plate_dae(mesh_names: list[str]) -> str:
    """A minimal COLLADA file holding centred plate quads (normal -Y, Z up),
    matching the orientation/UV conventions of the vanilla plate meshes."""
    geoms = []
    nodes = []
    materials = []
    effects = []
    for mesh in mesh_names:
        w, h = _REAR_QUADS[mesh]
        x, z = w / 2, h / 2
        positions = f"-{x} 0 -{z} {x} 0 -{z} {x} 0 {z} -{x} 0 {z}"
        effects.append(
            f'<effect id="{mesh}-effect"><profile_COMMON><technique sid="common"><lambert>'
            f'<diffuse><color>0.8 0.8 0.8 1</color></diffuse>'
            f"</lambert></technique></profile_COMMON></effect>"
        )
        materials.append(f'<material id="{mesh}-material" name="{mesh}"><instance_effect url="#{mesh}-effect"/></material>')
        uv_source = (
            f'<source id="{mesh}-uv0"><float_array id="{mesh}-uv0-array" count="8">0 0 1 0 1 1 0 1</float_array>'
            f'<technique_common><accessor source="#{mesh}-uv0-array" count="4" stride="2">'
            f'<param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>'
            f'<source id="{mesh}-uv1"><float_array id="{mesh}-uv1-array" count="8">0 0 1 0 1 1 0 1</float_array>'
            f'<technique_common><accessor source="#{mesh}-uv1-array" count="4" stride="2">'
            f'<param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>'
        )
        geoms.append(
            f'<geometry id="{mesh}-mesh" name="{mesh}"><mesh>'
            f'<source id="{mesh}-pos"><float_array id="{mesh}-pos-array" count="12">{positions}</float_array>'
            f'<technique_common><accessor source="#{mesh}-pos-array" count="4" stride="3">'
            f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>'
            f'<source id="{mesh}-nrm"><float_array id="{mesh}-nrm-array" count="3">0 -1 0</float_array>'
            f'<technique_common><accessor source="#{mesh}-nrm-array" count="1" stride="3">'
            f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>'
            f"{uv_source}"
            f'<vertices id="{mesh}-verts"><input semantic="POSITION" source="#{mesh}-pos"/></vertices>'
            f'<triangles material="{mesh}-material" count="2">'
            f'<input semantic="VERTEX" source="#{mesh}-verts" offset="0"/>'
            f'<input semantic="NORMAL" source="#{mesh}-nrm" offset="1"/>'
            f'<input semantic="TEXCOORD" source="#{mesh}-uv0" offset="2" set="0"/>'
            f'<input semantic="TEXCOORD" source="#{mesh}-uv1" offset="2" set="1"/>'
            f"<p>0 0 0 1 0 1 2 0 2 0 0 0 2 0 2 3 0 3</p></triangles></mesh></geometry>"
        )
        nodes.append(
            f'<node id="{mesh}" name="{mesh}" type="NODE">'
            f'<matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
            f'<instance_geometry url="#{mesh}-mesh" name="{mesh}"><bind_material><technique_common>'
            f'<instance_material symbol="{mesh}-material" target="#{mesh}-material">'
            f'<bind_vertex_input semantic="{mesh}-uv0" input_semantic="TEXCOORD" input_set="0"/>'
            f'<bind_vertex_input semantic="{mesh}-uv1" input_semantic="TEXCOORD" input_set="1"/>'
            f"</instance_material>"
            f"</technique_common></bind_material></instance_geometry></node>"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
        "<asset><contributor><authoring_tool>BeamXP</authoring_tool></contributor>"
        "<unit name=\"meter\" meter=\"1\"/><up_axis>Z_UP</up_axis></asset>"
        f"<library_effects>{''.join(effects)}</library_effects>"
        f"<library_materials>{''.join(materials)}</library_materials>"
        f"<library_geometries>{''.join(geoms)}</library_geometries>"
        f'<library_visual_scenes><visual_scene id="Scene" name="Scene">{"".join(nodes)}</visual_scene></library_visual_scenes>'
        '<scene><instance_visual_scene url="#Scene"/></scene></COLLADA>\n'
    )


def _rear_material_entry(mesh: str, fmt: str, normal_strength: float) -> dict[str, object]:
    tag = f"@licenseplate-{fmt}"
    return {
        "name": mesh,
        "mapTo": mesh,
        "class": "Material",
        "Stages": [
            {
                "baseColorMap": tag,
                "normalMap": f"{tag}-normal",
                "normalMapStrength": round(normal_strength, 3),
                "normalMapUseUV": 1,
                "roughnessFactor": 0.8,
                "roughnessMap": f"{tag}-specular",
            },
            {}, {}, {},
        ],
        "activeLayers": 1,
        "dynamicCubemap": True,
        "version": 1.5,
    }


# --------------------------------------------------------------------------
# Vanilla design compatibility for the cloned rear plates
#
# The cloned rear parts request the custom formats _REAR_FORMAT_WIDE/_2_1.
# The game only generates plate textures for formats defined by the active
# design's licensePlate.json, so switching to a vanilla design would leave the
# rear meshes untextured ("[NO TEXTURE]" in setPlateText, core/vehicles.lua).
# Two layers fix that:
#  1. licensePlate-default-<fmt>.json fallback files - the game's documented
#     per-format fallback - so the rear always gets at least the stock generic
#     plate look, even if our extension is not loaded.
#  2. A small GE Lua extension hooked on onLicensePlateChanged that re-renders
#     the rear texture tags from the active design's matching *front* format,
#     so rear plates mirror the front under any vanilla design.

_REAR_TO_FRONT = {_REAR_FORMAT_WIDE: _FORMAT_WIDE, _REAR_FORMAT_2_1: _FORMAT_2_1}
_REAR_COMPAT_EXTENSION = "bhdc_rearPlates"

# Copies of the game's own default plate blocks (licensePlate-default.json and
# licensePlate-default-52-11.json) so the static fallback matches the stock
# look; all referenced assets ship with the base game.
_GAME_DEFAULT_FONT_DIR = "vehicles/common/licenseplates/default"
_REAR_FALLBACK_BLOCKS = {
    _REAR_FORMAT_WIDE: {
        "size": {"x": 1024, "y": 196},
        "text": {"x": 0.50, "y": 0.45, "scale": 1.7, "color": "black", "limit": 10},
        "diffuse": {
            "spriteImg": f"{_GAME_DEFAULT_FONT_DIR}/platefont_d.png",
            "backgroundImg": f"{_GAME_DEFAULT_FONT_DIR}/license_plate_generic_new_wide_d.png",
            "fillStyle": "white",
        },
        "bump": {
            "spriteImg": f"{_GAME_DEFAULT_FONT_DIR}/platefont_n.png",
            "backgroundImg": "vehicles/common/licenseplates/driver_training/license_plate_german_new_wide_n.png",
            "fillStyle": "rgb(127,127,255)",
        },
        "specular": {
            "spriteImg": f"{_GAME_DEFAULT_FONT_DIR}/platefont_s.png",
            "fillStyle": "rgb(233,233,233)",
        },
    },
    _REAR_FORMAT_2_1: {
        "size": {"x": 512, "y": 256},
        "text": {"x": 0.5, "y": 0.5, "scale": 1, "color": "black", "limit": 8},
        "diffuse": {
            "spriteImg": f"{_GAME_DEFAULT_FONT_DIR}/platefont_d.png",
            "fillStyle": "white",
        },
        "bump": {
            "spriteImg": f"{_GAME_DEFAULT_FONT_DIR}/platefont_n.png",
            "backgroundImg": f"{_GAME_DEFAULT_FONT_DIR}/licenseplate-default_n.png",
            "fillStyle": "rgb(0,0,255)",
        },
        "specular": {
            "spriteImg": f"{_GAME_DEFAULT_FONT_DIR}/platefont_s.png",
            "fillStyle": "rgb(233,233,233)",
        },
    },
}


def _rear_fallback_design(rear_fmt: str) -> dict[str, object]:
    """A licensePlate-default-<rear_fmt>.json body for the game's fallback path."""
    return {
        "name": "BeamHDC rear plate fallback",
        "version": 2,
        "data": {"format": {rear_fmt: _REAR_FALLBACK_BLOCKS[rear_fmt]}},
    }


def _rear_mod_script_lua() -> str:
    return (
        "-- Generated by BeamXP. Keeps the cloned rear licence plates textured\n"
        "-- when a vanilla plate design is selected (see lua/ge/extensions/"
        f"{_REAR_COMPAT_EXTENSION}.lua).\n"
        f'setExtensionUnloadMode("{_REAR_COMPAT_EXTENSION}", "manual")\n'
    )


def _rear_plates_extension_lua() -> str:
    """GE extension: mirror the front plate texture onto the bhdc rear formats
    whenever the active design does not define them itself.

    Keep this file byte-identical across BeamHDC mods: several installed
    conversions may each ship a copy at the same virtual path, and the game
    mounts whichever it finds first.
    """
    return r"""-- Generated by BeamHDC (shared across BeamHDC conversions - keep in sync).
--
-- BeamHDC conversions clone rear licence plate parts onto the custom plate
-- formats "bhdc-rear-wide"/"bhdc-rear-2-1" so front and rear plates can carry
-- different textures. Vanilla plate designs do not define those formats, so
-- selecting one would leave the rear plates untextured. setPlateText()
-- (lua/ge/extensions/core/vehicles.lua) fires onLicensePlateChanged after
-- generating plate textures; this extension re-renders the rear texture tags
-- from the active design's matching front format so the rear mirrors the
-- front. Static fallback files under vehicles/common/licenseplates/default/
-- cover the case where this extension is not loaded.
local M = {}

local REAR_TO_FRONT = {
  ["bhdc-rear-wide"] = "52-11",
  ["bhdc-rear-2-1"] = "30-15",
}

-- setPlateText tags the 30-15 textures as "default" rather than by format id
local FRONT_TAG_PREFIX = {
  ["52-11"] = "@licenseplate-52-11",
  ["30-15"] = "@licenseplate-default",
}

local function designFormatTable(designPath)
  if type(designPath) ~= "string" or designPath == "" or not FS:fileExists(designPath) then
    return nil
  end
  local design = jsonReadFile(designPath)
  if not design then return nil end
  if design.version == 1 then
    return { ["30-15"] = design.data }
  end
  return design.data and design.data.format or nil
end

local function frontBlockFor(formatTable, frontFmt)
  local block = formatTable and formatTable[frontFmt]
  if block then return block end
  -- mirror the game's own per-format fallback for designs that lack the
  -- front format too (e.g. version-1 designs asked for a wide plate)
  local fallbackPath = "vehicles/common/licenseplates/default/licensePlate-default-" .. frontFmt .. ".json"
  if not FS:fileExists(fallbackPath) then return nil end
  local fallback = jsonReadFile(fallbackPath)
  return fallback and fallback.data and fallback.data.format and fallback.data.format[frontFmt] or nil
end

local function skipGeneration(veh)
  return settings.getValue("SkipGenerateLicencePlate")
    or veh:getDynDataFieldbyName("licenseNoGen", 0)
    or headless_mode
end

-- mirrors the texture generation half of setPlateText for a single tag
local function renderTag(veh, tag, txt, block, frontFmt)
  local data = deepcopy(block)
  if data.characterLayout then
    if FS:fileExists(data.characterLayout) then
      data.characterLayout = jsonReadFile(data.characterLayout)
    end
  else
    data.characterLayout = jsonReadFile("vehicles/common/licenseplates/default/platefont.json")
  end
  if data.generator then
    if FS:fileExists(data.generator) then
      data.generator = "local://local/" .. data.generator
    end
  else
    data.generator = "local://local/vehicles/common/licenseplates/default/licenseplate-default.html"
  end
  data.format = frontFmt
  local designData = { data = data }
  local modes = { diffuse = "", bump = "-normal", specular = "-specular" }
  for mode, suffix in pairs(modes) do
    veh:createUITexture(tag .. suffix, data.generator, data.size.x, data.size.y, UI_TEXTURE_USAGE_AUTOMATIC, 1)
    veh:queueJSUITexture(tag .. suffix, 'init("' .. mode .. '","' .. txt .. '", ' .. jsonEncode(designData) .. ');')
  end
end

local function onLicensePlateChanged(plateText, vehId, designPath, formats)
  if type(formats) ~= "table" then return end
  local veh = vehId and getObjectByID(vehId) or nil
  if not veh then return end

  local formatTable = nil
  local formatTableLoaded = false
  for _, fmt in ipairs(formats) do
    local frontFmt = REAR_TO_FRONT[fmt]
    if frontFmt then
      if not formatTableLoaded then
        formatTable = designFormatTable(designPath)
        formatTableLoaded = true
      end
      if not (formatTable and formatTable[fmt]) then
        local rearTag = "@licenseplate-" .. fmt
        if skipGeneration(veh) then
          local premade = "/vehicles/common/licenseplates/premade" .. FRONT_TAG_PREFIX[frontFmt]
          veh:setTaggedTexture(rearTag, premade .. ".dds")
          veh:setTaggedTexture(rearTag .. "-normal", premade .. "-normal.dds")
          veh:setTaggedTexture(rearTag .. "-specular", premade .. "-specular.dds")
        else
          local block = frontBlockFor(formatTable, frontFmt)
          if block then
            local txt = type(plateText) == "string" and plateText or ""
            renderTag(veh, rearTag, txt, block, frontFmt)
          end
        end
      end
    end
  end
end

M.onLicensePlateChanged = onLicensePlateChanged

return M
"""


def _write_rear_design_compat(output_root: Path, prefix: str, rear_formats_used: set[str]) -> None:
    """Ship the vanilla-design compatibility files alongside the rear clones."""
    from beamxp import hand_drive_core as core

    fallback_dir = output_root / "vehicles" / "common" / "licenseplates" / "default"
    for rear_fmt in sorted(rear_formats_used):
        core.write_text_file(
            fallback_dir / f"licensePlate-default-{rear_fmt}.json",
            json.dumps(_rear_fallback_design(rear_fmt), indent=2),
            encoding="utf-8",
        )
    core.write_text_file(
        output_root / "lua" / "ge" / "extensions" / f"{_REAR_COMPAT_EXTENSION}.lua",
        _rear_plates_extension_lua(),
        encoding="utf-8",
    )
    # modScript path is namespaced per vehicle so BeamXP mods never shadow
    # each other's scripts; the extension file itself is identical in all of
    # them, so shadowing there is harmless.
    core.write_text_file(
        output_root / "scripts" / prefix / "modScript.lua",
        _rear_mod_script_lua(),
        encoding="utf-8",
    )


def _visible_rear_plate_donor(
    context,
    slot_type: str,
    selected_part_id: str,
    selected_body: str,
) -> tuple[str, str] | None:
    """Find a usable rear plate part when the selected one is a hidden proxy.

    Some community mods keep the real BeamNG license plate flexbody far outside
    the vehicle and use a bumper-baked plate face instead. If that hidden proxy
    is selected, cloning it preserves the offscreen position. Prefer a sibling
    rear plate part with the same plate format and a plausible vehicle-space
    position, then force it into the active slot.
    """
    from beamxp import transform_helpers

    selected_wide = _part_uses_wide_plate(selected_body)
    selected_prefix = _plate_part_prefix(selected_part_id)
    candidates: list[tuple[int, str, str]] = []
    for part_id, (body, _filename) in context.part_body_index.items():
        part_text = str(part_id)
        if part_text == selected_part_id:
            continue
        meshes = transform_helpers.extract_part_mesh_names(body)
        if not any("licenseplate" in mesh.lower() for mesh in meshes):
            continue
        if _part_uses_wide_plate(body) != selected_wide:
            continue
        if _plate_part_is_hidden(body):
            continue
        slot_types = transform_helpers.extract_part_slot_types(body)
        slot_text = " ".join(slot_types)
        if not _looks_rear(slot_text, part_text, body):
            continue

        score = 0
        if slot_type in slot_types:
            score += 100
        if _plate_part_prefix(part_text) == selected_prefix:
            score += 50
        if selected_part_id.lower().replace("_eu", "") in part_text.lower():
            score += 20
        if "wide" in part_text.lower():
            score += 5
        candidates.append((score, part_text, body))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _score, part_id, body = candidates[0]
    return part_id, _force_slot_type(body, slot_type)


# --------------------------------------------------------------------------
# Build integration


def apply_to_build(
    context,
    conversion: dict[str, object],
    output_root: Path,
    output_vehicle_dir: Path,
    output_plans: list[dict[str, object]],
) -> dict[str, object]:
    """Generate plate assets for the built configs and update the written .pc
    files with a per-config random registration and the generated design."""
    from beamxp import hand_drive_core as core

    summary: dict[str, object] = {
        "configsUpdated": 0,
        "designs": 0,
        "rearPartsCloned": 0,
        "physicalPartsCloned": 0,
        "physicalSelectionsUpdated": 0,
        "warnings": [],
    }
    plans: list[tuple[str, str, dict[str, object] | None, str | None, str, str]] = []
    variants = conversion.get("variants", {})
    for plan in output_plans:
        config_name = str(plan["source"])
        output_config = str(plan["output"])
        cfg, set_id = effective_plate_selection(
            conversion,
            config_name,
            warnings=summary["warnings"],
        )
        settings = variants.get(config_name) if isinstance(variants, dict) else None
        front_choice = normalized_plate_part_choice(settings.get("frontPlate")) if isinstance(settings, dict) else PLATE_PART_AUTO
        rear_choice = normalized_plate_part_choice(settings.get("rearPlate")) if isinstance(settings, dict) else PLATE_PART_AUTO
        if cfg is None and front_choice == PLATE_PART_AUTO and rear_choice == PLATE_PART_AUTO:
            continue
        if cfg is None:
            plans.append((config_name, output_config, None, None, front_choice, rear_choice))
            continue
        errors = validate_plate_config(cfg)
        if errors:
            raise PlateError(
                f"Licence plate settings for '{config_name}' are not buildable:\n- " + "\n- ".join(errors)
            )
        plans.append((config_name, output_config, cfg, set_id, front_choice, rear_choice))
    if not plans:
        return summary

    prefix = f"bhdc_{core.safe_id(context.vehicle_id)}"
    design_cache: dict[str, _DesignOutput] = {}
    part_bodies: dict[str, str] = {}
    rear_meshes_used: dict[str, tuple[str, float]] = {}  # mesh -> (rear format, max normal strength)
    rear_mesh_clone_sources: dict[str, _RearMeshClone] = {}
    rng = random.Random()

    for config_name, output_config, cfg, set_id, front_choice, rear_choice in plans:
        pc_path = output_vehicle_dir / f"{output_config}.pc"
        if not pc_path.is_file():
            summary["warnings"].append(f"{config_name}: expected config file missing ({pc_path.name})")
            continue
        pc = core.load_beamng_json_file(pc_path)
        parts = dict(pc.get("parts", {}))
        physical_changed = _apply_plate_part_choices(
            context,
            config_name,
            parts,
            front_choice,
            rear_choice,
            part_bodies,
            summary,
        )
        if physical_changed:
            summary["physicalSelectionsUpdated"] = int(summary["physicalSelectionsUpdated"]) + 1

        if cfg is not None:
            design = _emit_design(cfg, output_root, prefix, design_cache, set_id=set_id)
            if set_id:
                record = plate_set_by_id(set_id)
                design_label = str(record.get("name")) if record is not None else set_id
            else:
                design_label = str(cfg.get("size"))
            part_bodies.setdefault(
                design.part_id,
                _design_part_body(design, design_label, custom=set_id is None),
            )
            parts[_DESIGN_SLOT] = design.part_id

            if design.rear_formats and rear_choice != PLATE_PART_NONE:
                # Clone the rear part even when this design's rear matches its
                # front, so a rear-differing design selected in-game later
                # still gets its own rear texture.
                cloned = _clone_rear_parts_for_config(
                    context,
                    config_name,
                    cfg,
                    design,
                    part_bodies,
                    rear_meshes_used,
                    rear_mesh_clone_sources,
                    parts,
                    summary,
                )
                if not cloned and design.rear_differs:
                    summary["warnings"].append(
                        f"{config_name}: no rear plate part found; rear plate keeps the front colour"
                    )

        pc["parts"] = parts
        if cfg is not None:
            pc["licenseName"] = generate_registration(active_pattern(cfg), rng)
        core.write_text_file(pc_path, json.dumps(pc, indent=2), encoding="utf-8")
        summary["configsUpdated"] += 1

    if rear_meshes_used:
        fallback_meshes = sorted(mesh for mesh in rear_meshes_used if mesh in _REAR_FALLBACK_MESHES)
        if fallback_meshes:
            core.write_text_file(output_vehicle_dir / "bhdc_plate_rear.dae", _rear_plate_dae(fallback_meshes), encoding="utf-8")
        if rear_mesh_clone_sources:
            _write_rear_clone_daes(context, output_vehicle_dir, rear_mesh_clone_sources, summary)
        materials = {
            mesh: _rear_material_entry(mesh, fmt, normal_strength)
            for mesh, (fmt, normal_strength) in sorted(rear_meshes_used.items())
        }
        core.write_text_file(output_vehicle_dir / "bhdc_plate_rear.materials.json", json.dumps(materials, indent=2), encoding="utf-8")
        rear_formats_used = {fmt for fmt, _strength in rear_meshes_used.values()}
        _write_rear_design_compat(output_root, prefix, rear_formats_used)
        summary["rearVanillaFallbacks"] = sorted(rear_formats_used)

    if part_bodies:
        jbeam_dir = output_vehicle_dir / "jbeam"
        jbeam_dir.mkdir(parents=True, exist_ok=True)
        body_text = ",\n".join(f'"{part_id}": {body}' if not body.lstrip().startswith(f'"{part_id}"') else body
                               for part_id, body in sorted(part_bodies.items()))
        contents = "{\n// Generated physical plate parts and designs (BeamXP).\n" + body_text + "\n}\n"
        core.write_text_file(jbeam_dir / "bhdc_licenseplates.jbeam", contents, encoding="utf-8")

    summary["designs"] = len(design_cache)
    return summary


def export_plate_sets(records: list[dict[str, object]], target_zip: Path) -> dict[str, object]:
    """Write selected reusable designs as one universal BeamXP plates mod."""
    from beamxp import hand_drive_core as core

    if not records:
        raise PlateError("Select at least one plate set to export")
    output_root = target_zip.parent / "BeamXP_plates_unpacked"
    core.clean_dir(output_root)
    cache: dict[str, _DesignOutput] = {}
    part_bodies: dict[str, str] = {}
    exported: list[str] = []
    for record in records:
        set_id = str(record.get("id") or "")
        if not set_id:
            continue
        cfg = normalized_plate_config(record.get("config"))
        cfg["enabled"] = True
        errors = validate_plate_config(cfg)
        if errors:
            raise PlateError(f"Plate set '{record.get('name') or set_id}' is not buildable:\n- " + "\n- ".join(errors))
        design = _emit_design(cfg, output_root, "bhdc_plates", cache, set_id=set_id)
        part_bodies[design.part_id] = _design_part_body(design, str(record.get("name") or cfg.get("size")))
        exported.append(set_id)

    if not part_bodies:
        raise PlateError("No valid plate sets were selected")
    common_dir = output_root / "vehicles" / "common" / "licenseplates"
    common_dir.mkdir(parents=True, exist_ok=True)
    body_text = ",\n".join(
        f'"{part_id}": {body}' for part_id, body in sorted(part_bodies.items())
    )
    core.write_text_file(
        common_dir / "bhdc_plate_sets.jbeam",
        "{\n// Reusable licence plate designs generated by BeamXP.\n" + body_text + "\n}\n",
        encoding="utf-8",
    )
    mod_info = output_root / "mod_info"
    mod_info.mkdir(parents=True, exist_ok=True)
    core.write_text_file(
        mod_info / "info.json",
        json.dumps({
            "name": "BeamXP Plate Sets",
            "version": "1.0.0",
            "authors": "BeamXP",
            "description": "Reusable BeamXP licence plate designs for all supported vehicles.",
        }, indent=2),
        encoding="utf-8",
    )
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    core.make_zip(output_root, target_zip)
    return {"zip": target_zip, "setIds": exported, "designs": len(part_bodies)}


def export_all_plate_sets(target_zip: Path | None = None) -> dict[str, object] | None:
    """Export every library plate set as the universal plates mod, or None
    when the library is empty."""
    records = plate_set_records()
    if not records:
        return None
    return export_plate_sets(records, target_zip or default_plate_export_path())


def _clone_rear_parts_for_config(
    context,
    config_name: str,
    cfg: dict[str, object],
    design: _DesignOutput,
    part_bodies: dict[str, str],
    rear_meshes_used: dict[str, tuple[str, float]],
    rear_mesh_clone_sources: dict[str, _RearMeshClone],
    parts: dict[str, object],
    summary: dict[str, object],
) -> bool:
    from beamxp import hand_drive_core as core

    cloned_any = False
    for slot_text, part_text, part_body in _iter_selected_plate_parts(
        context,
        config_name,
        parts,
        part_bodies,
    ):
        if not _looks_rear(slot_text, part_text, part_body):
            continue
        new_part_id = f"bhdc_rear_{core.safe_id(part_text)}"
        if new_part_id not in part_bodies:
            clone_source_id = part_text
            clone_source_body = part_body
            if _plate_part_is_hidden(part_body):
                donor = _visible_rear_plate_donor(context, slot_text, part_text, part_body)
                if donor is None:
                    continue
                clone_source_id, clone_source_body = donor
            cloned = _clone_rear_plate_part(context, clone_source_id, clone_source_body, new_part_id, rear_mesh_clone_sources)
            if cloned is None:
                continue
            body, rear_fmt, output_meshes = cloned
            part_bodies[new_part_id] = body
            normal_strength = _effective_emboss_strength(cfg)
            for rear_mesh in output_meshes:
                existing = rear_meshes_used.get(rear_mesh)
                rear_meshes_used[rear_mesh] = (
                    rear_fmt,
                    max(normal_strength, existing[1] if existing else 0.0),
                )
            summary["rearPartsCloned"] = int(summary.get("rearPartsCloned", 0)) + 1
        parts[slot_text] = new_part_id
        cloned_any = True
    return cloned_any
