"""Enumerate selectable vehicles across the game's vehicles folder and mods folder.

The tool used to require the user to find a specific zip. This walks the two
folders they already configure and lists what the in-game selector would list,
so the Model dropdown reads "ETK 800-Series" rather than "etk800".

Grouping follows the engine (see :func:`beamxp.core.files.vehicle_catalog_entries_in_zip`):
one listing per selector *tile*, so vivace contributes Vivace, Ardente and
Tograc while etk800 contributes one entry holding all 29 trims.

A mod that replaces or extends a stock vehicle is listed alongside the stock
one rather than hiding it, since either may be the intended conversion source.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

from beamxp.core.files import VehicleCatalogEntry, vehicle_catalog_entries_in_zip

# lua/ge/extensions/ui/vehicleSelector/tileGrouping.lua maps the model's Type
# onto a selector section. Car and Truck are the "Cars and Trucks" section.
CAR_TRUCK_TYPES = frozenset({"Car", "Truck"})
# Automation exports are a section of their own in game. They are drivable and
# convertible, so they are offered behind a toggle rather than hidden outright.
AUTOMATION_TYPES = frozenset({"Automation"})

# build_batch writes this into every mod it produces. Our own output must never
# be offered as a conversion source: it is already converted.
TOOL_BUILD_MARKER = "handedness_conversion/conversion.json"

# vehicles/common holds shared parts, not a vehicle.
RESERVED_VEHICLE_IDS = frozenset({"common"})


@dataclass(frozen=True)
class VehicleListing:
    source_zip: Path
    entry: VehicleCatalogEntry
    is_mod: bool
    display_name: str
    vehicle_type: str
    # Where the thumbnail lives, which is not always source_zip: a mod that
    # extends a stock vehicle borrows the stock preview.
    preview_zip: Path
    preview_member: str
    # 0 when this vehicle comes from a single mod; otherwise its 1-based place
    # among the mods offering the same vehicle, so the labels stay distinct.
    mod_index: int = 0
    # Appended when the name alone is still ambiguous. Stock ships two distinct
    # vehicles both called "Ibishu Pessima" (midsize 1996-2000 and pessima
    # 1988-1991), which the game also shows under one name.
    disambiguator: str = ""

    @property
    def vehicle_id(self) -> str:
        return self.entry.vehicle_id

    def label(self) -> str:
        text = self.display_name
        if self.is_mod:
            text += f" [mod] #{self.mod_index}" if self.mod_index else " [mod]"
        if self.disambiguator:
            text += f" ({self.disambiguator})"
        return text


def _iter_zips(folder: Path) -> list[Path]:
    if not folder or not folder.is_dir():
        return []
    try:
        # Mods can sit in subfolders (mods/repo/, mods/unpacked/ siblings).
        return sorted(folder.rglob("*.zip"))
    except OSError:
        return []


def _is_tool_build(source_zip: Path) -> bool:
    try:
        with zipfile.ZipFile(source_zip) as archive:
            return any(
                name.replace("\\", "/") == TOOL_BUILD_MARKER for name in archive.namelist()
            )
    except (OSError, zipfile.BadZipFile):
        return False


def scan_folder(folder: Path, *, is_mod: bool) -> list[tuple[Path, VehicleCatalogEntry]]:
    """Every (zip, catalog entry) pair in the folder that could be converted."""
    found: list[tuple[Path, VehicleCatalogEntry]] = []
    for source_zip in _iter_zips(folder):
        if is_mod and _is_tool_build(source_zip):
            continue
        try:
            entries = vehicle_catalog_entries_in_zip(source_zip)
        except (OSError, zipfile.BadZipFile):
            continue  # an unreadable or partially-downloaded mod must not stop the scan
        for entry in entries:
            if entry.source_vehicle_id.lower() in RESERVED_VEHICLE_IDS:
                continue
            # A zip that ships parts for a vehicle but no trim of its own (a
            # plate or livery overlay) has nothing to convert.
            if entry.config_count == 0:
                continue
            found.append((source_zip, entry))
    return found


def scan_vehicle_inventory(
    game_vehicles_folder: Path | None,
    mods_folder: Path | None,
    *,
    include_automation: bool = False,
) -> list[VehicleListing]:
    """Selectable vehicles from both folders, sorted by display name."""
    wanted = set(CAR_TRUCK_TYPES)
    if include_automation:
        wanted |= AUTOMATION_TYPES

    stock_pairs = scan_folder(Path(game_vehicles_folder), is_mod=False) if game_vehicles_folder else []
    mod_pairs = scan_folder(Path(mods_folder), is_mod=True) if mods_folder else []

    # A mod extending a stock vehicle usually ships no model info.json, so its
    # name, type and preview are inherited from the vehicle it extends.
    stock_by_vehicle: dict[str, tuple[Path, VehicleCatalogEntry]] = {}
    for source_zip, entry in stock_pairs:
        stock_by_vehicle.setdefault(entry.vehicle_id.lower(), (source_zip, entry))

    listings: list[VehicleListing] = []
    for source_zip, entry in stock_pairs:
        if entry.vehicle_type not in wanted:
            continue
        listings.append(
            VehicleListing(
                source_zip=source_zip,
                entry=entry,
                is_mod=False,
                display_name=entry.display_name or entry.vehicle_id,
                vehicle_type=entry.vehicle_type,
                preview_zip=source_zip,
                preview_member=entry.preview_member,
            )
        )

    for source_zip, entry in mod_pairs:
        base = stock_by_vehicle.get(entry.vehicle_id.lower())
        display_name = entry.display_name
        vehicle_type = entry.vehicle_type
        preview_zip, preview_member = source_zip, entry.preview_member
        if base is not None:
            base_zip, base_entry = base
            display_name = display_name or base_entry.display_name
            vehicle_type = vehicle_type or base_entry.vehicle_type
            if not preview_member:
                preview_zip, preview_member = base_zip, base_entry.preview_member
        if vehicle_type not in wanted:
            continue
        listings.append(
            VehicleListing(
                source_zip=source_zip,
                entry=entry,
                is_mod=True,
                display_name=display_name or entry.vehicle_id,
                vehicle_type=vehicle_type,
                preview_zip=preview_zip,
                preview_member=preview_member,
            )
        )

    listings = _disambiguate_labels(listings)
    return sorted(
        listings,
        key=lambda item: (item.display_name.lower(), item.is_mod, item.mod_index, item.vehicle_id.lower()),
    )


def _disambiguate_labels(listings: list[VehicleListing]) -> list[VehicleListing]:
    """Make every label distinct, so the dropdown never shows two of one name.

    Numbers the mods when several offer the same vehicle, then falls back to the
    vehicle id for any label still shared -- which is how two different stock
    vehicles that carry the same name are told apart.
    """
    # Stable order so a given folder always yields the same numbering.
    ordered = sorted(listings, key=lambda item: (item.display_name.lower(), str(item.source_zip)))

    mods_per_vehicle: dict[str, int] = {}
    for listing in ordered:
        if listing.is_mod:
            key = listing.vehicle_id.lower()
            mods_per_vehicle[key] = mods_per_vehicle.get(key, 0) + 1

    seen: dict[str, int] = {}
    numbered: list[VehicleListing] = []
    for listing in ordered:
        if listing.is_mod and mods_per_vehicle.get(listing.vehicle_id.lower(), 0) > 1:
            key = listing.vehicle_id.lower()
            seen[key] = seen.get(key, 0) + 1
            listing = replace(listing, mod_index=seen[key])
        numbered.append(listing)

    label_counts: dict[str, int] = {}
    for listing in numbered:
        label_counts[listing.label()] = label_counts.get(listing.label(), 0) + 1
    return [
        replace(listing, disambiguator=listing.vehicle_id)
        if label_counts.get(listing.label(), 0) > 1
        else listing
        for listing in numbered
    ]


def read_preview_image_bytes(listing: VehicleListing) -> bytes | None:
    """The tile's preview image, or None when no usable preview was found."""
    if not listing.preview_member:
        return None
    try:
        with zipfile.ZipFile(listing.preview_zip) as archive:
            return archive.read(listing.preview_member)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


__all__ = [
    "AUTOMATION_TYPES",
    "CAR_TRUCK_TYPES",
    "RESERVED_VEHICLE_IDS",
    "TOOL_BUILD_MARKER",
    "VehicleListing",
    "read_preview_image_bytes",
    "scan_folder",
    "scan_vehicle_inventory",
]
