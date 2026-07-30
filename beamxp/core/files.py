from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from beamxp.core.constants import APP_SETTINGS_PATH, PROJECTS_DIR


@dataclass(frozen=True)
class VehicleCatalogEntry:
    vehicle_id: str
    source_vehicle_id: str
    config_names: tuple[str, ...] = ()


def _direct_zip_files(names: list[str], vehicle_id: str, suffix: str) -> list[str]:
    prefix = f"vehicles/{vehicle_id}/"
    wanted = suffix.lower()
    out: list[str] = []
    for name in names:
        clean = name.replace("\\", "/")
        if not clean.startswith(prefix) or Path(clean).suffix.lower() != wanted:
            continue
        if "/" in clean[len(prefix) :]:
            continue
        out.append(clean)
    return sorted(out)


def _beamng_catalog_group_key(info: dict[str, object]) -> str:
    for key in ("vehicleSelectorSubGroup", "vehicleSelectorSubCluster"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _selector_model_key(selector_key: str, source_vehicle_id: str) -> str:
    pattern = rf"^vehiclesData\.{re.escape(source_vehicle_id)}(?:\.([^.]+))?\.vehicleSelectorSub(?:Group|Cluster)$"
    match = re.match(pattern, selector_key)
    if not match:
        return ""
    return str(match.group(1) or source_vehicle_id)


def _shared_config_prefix(config_names: list[str]) -> str:
    if not config_names:
        return ""
    split_names = [name.split("_") for name in config_names]
    prefix: list[str] = []
    for parts in zip(*split_names):
        first = parts[0]
        if all(part == first for part in parts):
            prefix.append(first)
        else:
            break
    return "_".join(prefix)


def _catalog_vehicle_id(
    source_vehicle_id: str,
    selector_key: str,
    config_names: list[str],
) -> str:
    model_key = _selector_model_key(selector_key, source_vehicle_id)
    if not model_key:
        return source_vehicle_id
    if model_key == source_vehicle_id:
        return source_vehicle_id
    shared_prefix = _shared_config_prefix(config_names)
    if shared_prefix and model_key.startswith(f"{shared_prefix}_"):
        return shared_prefix
    return model_key


def vehicle_catalog_entries_in_zip(source_zip: Path) -> list[VehicleCatalogEntry]:
    vehicles: dict[str, set[str]] = {}
    with zipfile.ZipFile(source_zip) as zf:
        names = [name.replace("\\", "/") for name in zf.namelist()]
        names_set = set(names)
        for name in names:
            match = re.match(r"vehicles/([^/]+)/(.+)", name, re.IGNORECASE)
            if not match:
                continue
            vehicle_id, rest = match.groups()
            suffix = Path(rest).suffix.lower()
            if suffix in {".dae", ".pc", ".jbeam"}:
                vehicles.setdefault(vehicle_id, set()).add(suffix)

        entries: list[VehicleCatalogEntry] = []
        for source_vehicle_id in sorted(vehicles):
            suffixes = vehicles[source_vehicle_id]
            if ".dae" not in suffixes or (".pc" not in suffixes and ".jbeam" not in suffixes):
                continue

            pc_paths = _direct_zip_files(names, source_vehicle_id, ".pc")
            grouped: dict[str, list[str]] = {}
            ungrouped: list[str] = []
            if pc_paths:
                from beamxp.core.beam_json import load_info

                for pc_path in pc_paths:
                    config_name = Path(pc_path).stem
                    info_candidates = (
                        f"vehicles/{source_vehicle_id}/info_{config_name}.json",
                        f"vehicles/{source_vehicle_id}/{config_name}.json",
                    )
                    info_path = next((candidate for candidate in info_candidates if candidate in names_set), None)
                    group_key = ""
                    if info_path is not None:
                        try:
                            group_key = _beamng_catalog_group_key(load_info(source_zip, info_path))
                        except Exception:
                            group_key = ""
                    if group_key:
                        grouped.setdefault(group_key, []).append(config_name)
                    else:
                        ungrouped.append(config_name)

            if not grouped:
                entries.append(VehicleCatalogEntry(source_vehicle_id, source_vehicle_id))
                continue

            used_ids: set[str] = set()
            for group_key, config_names in sorted(grouped.items()):
                vehicle_id = safe_project_segment(
                    _catalog_vehicle_id(source_vehicle_id, group_key, sorted(config_names))
                )
                if vehicle_id in used_ids:
                    vehicle_id = safe_project_segment(f"{source_vehicle_id}_{vehicle_id}")
                used_ids.add(vehicle_id)
                entries.append(
                    VehicleCatalogEntry(
                        vehicle_id=vehicle_id,
                        source_vehicle_id=source_vehicle_id,
                        config_names=tuple(sorted(config_names)),
                    )
                )
            if ungrouped:
                vehicle_id = source_vehicle_id
                if vehicle_id in used_ids:
                    vehicle_id = safe_project_segment(f"{source_vehicle_id}_other")
                entries.append(
                    VehicleCatalogEntry(
                        vehicle_id=vehicle_id,
                        source_vehicle_id=source_vehicle_id,
                        config_names=tuple(sorted(ungrouped)),
                    )
                )
        return sorted(entries, key=lambda entry: (entry.vehicle_id.lower(), entry.source_vehicle_id.lower()))


def vehicle_catalog_entry_for_id(source_zip: Path, vehicle_id: str) -> VehicleCatalogEntry | None:
    wanted = str(vehicle_id).lower()
    return next(
        (
            entry
            for entry in vehicle_catalog_entries_in_zip(source_zip)
            if entry.vehicle_id.lower() == wanted
        ),
        None,
    )


def vehicle_ids_in_zip(source_zip: Path) -> list[str]:
    return [entry.vehicle_id for entry in vehicle_catalog_entries_in_zip(source_zip)]


def safe_project_segment(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "vehicle"


def project_dir_for(source_zip: Path, vehicle_id: str) -> Path:
    source_segment = safe_project_segment(source_zip.stem)
    vehicle_segment = safe_project_segment(vehicle_id)
    if source_segment.lower() == vehicle_segment.lower():
        return PROJECTS_DIR / vehicle_segment
    return PROJECTS_DIR / f"{source_segment}_{vehicle_segment}"


def fs_path(path: Path) -> str:
    text = str(path.resolve(strict=False))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def write_text_file(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(fs_path(path), "w", encoding=encoding) as fh:
        fh.write(text)


def write_bytes_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(fs_path(path), "wb") as fh:
        fh.write(data)


def write_xml_tree(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(fs_path(path), "wb") as fh:
        tree.write(fh, encoding="utf-8", xml_declaration=True)


def read_json_file(path: Path) -> dict[str, object]:
    with open(fs_path(path), encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def vehicle_prefix(vehicle_id: str) -> str:
    return f"vehicles/{vehicle_id}"


def list_vehicle_files(source_zip: Path, vehicle_id: str, suffix: str) -> list[str]:
    prefix = f"{vehicle_prefix(vehicle_id)}/"
    wanted = suffix.lower()
    with zipfile.ZipFile(source_zip) as zf:
        return sorted(
            name.replace("\\", "/")
            for name in zf.namelist()
            if name.replace("\\", "/").startswith(prefix)
            and Path(name).suffix.lower() == wanted
        )


def direct_vehicle_files(source_zip: Path, vehicle_id: str, suffix: str) -> list[str]:
    prefix = f"{vehicle_prefix(vehicle_id)}/"
    wanted = suffix.lower()
    with zipfile.ZipFile(source_zip) as zf:
        out = []
        for name in zf.namelist():
            clean = name.replace("\\", "/")
            if not clean.startswith(prefix) or Path(clean).suffix.lower() != wanted:
                continue
            if "/" in clean[len(prefix) :]:
                continue
            out.append(clean)
    return sorted(out)


_game_common_zips_cache: list[Path] | None = None


def beamng_game_common_zips() -> list[Path]:
    """The game install's content/vehicles/common.zip."""
    global _game_common_zips_cache
    if _game_common_zips_cache is not None:
        return _game_common_zips_cache
    found: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        try:
            resolved = str(candidate.resolve(strict=False)).lower()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        if candidate.is_file():
            found.append(candidate)

    try:
        raw = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        folders = [raw.get("lastVehicleZipFolder")]
        recents = raw.get("recentVehicles")
        if isinstance(recents, list):
            for entry in recents:
                if isinstance(entry, dict) and entry.get("zip"):
                    folders.append(str(Path(str(entry["zip"])).parent))
        for folder in folders:
            if folder:
                add(Path(str(folder)) / "common.zip")
    except Exception:
        pass

    try:
        steam_roots: list[Path] = []
        if sys.platform == "win32":
            import winreg

            for hive, key, value_name in (
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as handle:
                        value, _kind = winreg.QueryValueEx(handle, value_name)
                    steam_roots.append(Path(str(value)))
                except OSError:
                    continue
        for root in list(steam_roots):
            try:
                text = (root / "steamapps" / "libraryfolders.vdf").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            for match in re.finditer(r'"path"\s+"((?:[^"\\]|\\.)*)"', text):
                steam_roots.append(Path(match.group(1).replace("\\\\", "\\")))
        for root in steam_roots:
            add(root / "steamapps" / "common" / "BeamNG.drive" / "content" / "vehicles" / "common.zip")
    except Exception:
        pass

    _game_common_zips_cache = found
    return found


def common_zip_candidates(source_zip: Path) -> list[Path]:
    candidates = [source_zip]
    resolved: set[str] = {str(source_zip.resolve(strict=False)).lower()}

    def add(candidate: Path) -> None:
        key = str(candidate.resolve(strict=False)).lower()
        if key not in resolved:
            resolved.add(key)
            candidates.append(candidate)

    sibling_common = source_zip.parent / "common.zip"
    if sibling_common.exists():
        add(sibling_common)
    for game_common in beamng_game_common_zips():
        add(game_common)
    return candidates


def load_jbeam_texts(source_zip: Path, vehicle_id: str) -> dict[str, str]:
    prefix = f"{vehicle_prefix(vehicle_id)}/"
    with zipfile.ZipFile(source_zip) as zf:
        return {
            name.replace("\\", "/"): zf.read(name).decode("utf-8", errors="replace")
            for name in zf.namelist()
            if name.replace("\\", "/").startswith(prefix)
            and name.lower().endswith(".jbeam")
        }


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(fs_path(path))
    path.mkdir(parents=True, exist_ok=True)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def zip_member_path(value: str) -> Path:
    return Path(*[part for part in value.replace("\\", "/").split("/") if part])


def make_zip(src: Path, target: Path) -> None:
    if os.path.exists(fs_path(target)):
        os.remove(fs_path(target))
    src_path = fs_path(src)
    with zipfile.ZipFile(fs_path(target), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                archive_name = os.path.relpath(file_path, src_path).replace(os.sep, "/")
                zf.write(file_path, archive_name)
