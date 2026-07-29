from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from beamxp.core.constants import APP_SETTINGS_PATH, PROJECTS_DIR


def vehicle_ids_in_zip(source_zip: Path) -> list[str]:
    vehicles: dict[str, set[str]] = {}
    with zipfile.ZipFile(source_zip) as zf:
        for name in zf.namelist():
            match = re.match(r"vehicles/([^/]+)/(.+)", name.replace("\\", "/"), re.IGNORECASE)
            if not match:
                continue
            vehicle_id, rest = match.groups()
            suffix = Path(rest).suffix.lower()
            if suffix in {".dae", ".pc", ".jbeam"}:
                vehicles.setdefault(vehicle_id, set()).add(suffix)
    return sorted(
        vehicle_id
        for vehicle_id, suffixes in vehicles.items()
        if ".dae" in suffixes and (".pc" in suffixes or ".jbeam" in suffixes)
    )


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
