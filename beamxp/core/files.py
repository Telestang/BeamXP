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
    # Presentation, resolved the way the in-game selector resolves it. Empty
    # when the zip carries no model info.json (older mods), in which case
    # callers fall back to the vehicle id.
    display_name: str = ""
    vehicle_type: str = ""
    preview_member: str = ""
    # Configs this tile owns. Zero means the zip contributes parts to a vehicle
    # without shipping any trim of its own (a plate or bodykit overlay).
    config_count: int = 0


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


_selector_translations_cache: dict[str, dict[str, str]] = {}


def _flatten_translations(node: object, prefix: str, out: dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_translations(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, str):
        out[prefix] = node


def _selector_translations(source_zip: Path) -> dict[str, str]:
    """The game's en-US vehicle strings, keyed by localisation path.

    The selector clusters on the *translated* subCluster, not the raw key, which
    is why vivace's two Tograc keys form one tile. Without the game install we
    fall back to the raw key, exactly as the engine does for a missing string.
    """
    roots: list[Path] = []
    # <game>/content/vehicles/<vehicle>.zip -> <game>
    if len(source_zip.parents) >= 3:
        roots.append(source_zip.parents[2])
    for common_zip in beamng_game_common_zips():
        if len(common_zip.parents) >= 3:
            roots.append(common_zip.parents[2])

    for root in roots:
        path = root / "locales" / "translations" / "en-US" / "vehiclesGenerated.translation.json"
        cache_key = str(path).lower()
        if cache_key in _selector_translations_cache:
            cached = _selector_translations_cache[cache_key]
            if cached:
                return cached
            continue
        table: dict[str, str] = {}
        try:
            _flatten_translations(json.loads(path.read_text(encoding="utf-8")), "", table)
        except Exception:
            table = {}
        _selector_translations_cache[cache_key] = table
        if table:
            return table
    return {}


def _translate_selector_key(value: object, translations: dict[str, str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    translated = translations.get(text)
    if translated:
        return translated
    # No game install to translate against (a mod zip opened standalone). Reduce
    # the localisation path to its distinctive segment so the derived id stays
    # readable: vehiclesData.vivace.tograc_110_M.vehicleSelectorSubCluster ->
    # tograc_110_M. Two keys that share a translated name stay separate here,
    # which splits a tile the game would merge; the ids remain stable either way.
    if text.startswith("vehiclesData."):
        segments = [segment for segment in text.split(".") if segment]
        if segments and segments[-1].startswith("vehicleSelectorSub"):
            segments = segments[:-1]
        if segments:
            return segments[-1]
    return text


def _config_cluster_name(
    model_info: dict[str, object],
    config_info: dict[str, object],
    translations: dict[str, str],
) -> str:
    """The selector tile a config belongs to, or "" when the model does not split.

    Mirrors the engine's clusterModeFunctions['model'] in
    lua/ge/extensions/ui/vehicleSelector/tileClustering.lua:

        config.model_key .. (config.useSubCluster and " " .. subCluster or "")

    ``useSubCluster`` is a model-level opt-in a config may override
    (core/vehicles.lua), and only md_series, us_semi and vivace set it in stock
    content. ``vehicleSelectorSubGroup`` is deliberately ignored: the engine uses
    it purely as a heading inside a tile ("Sedan", "Wagon", "Custom"), and its
    values are shared across unrelated vehicles.
    """
    use_sub_cluster = config_info.get("useSubCluster")
    if use_sub_cluster is None:
        use_sub_cluster = model_info.get("useSubCluster")
    if not use_sub_cluster:
        return ""
    raw = config_info.get("vehicleSelectorSubCluster") or model_info.get("vehicleSelectorSubCluster")
    return _translate_selector_key(raw, translations)


def _tile_display_name(
    model_info: dict[str, object],
    source_vehicle_id: str,
    cluster_name: str,
    translations: dict[str, str],
) -> str:
    """The label the in-game selector puts on a tile.

    createTileFromModel builds ``brand .. " " .. (subCluster or model.Name)``,
    which is why vivace's tiles read "Cherrier Ardente"/"Cherrier Tograc" rather
    than the model's own name ("Cherrier FCV").
    """
    brand = str(model_info.get("Brand") or "").strip()
    if cluster_name:
        name = cluster_name
    else:
        name = _translate_selector_key(model_info.get("Name"), translations)
    name = name.strip()
    if not brand and not name:
        # No model identity in this zip at all. Reported as empty rather than as
        # the vehicle id so a caller can tell "unnamed" from "named after its
        # folder" -- a mod extending a stock vehicle inherits the stock name.
        return ""
    return f"{brand} {name or source_vehicle_id}".strip()


def _tile_preview_member(
    names_set: set[str],
    model_info: dict[str, object],
    source_vehicle_id: str,
    cluster_configs: list[str],
    default_config_for_cluster: str,
    holds_model_default: bool,
) -> str:
    """Zip member holding the tile's preview image.

    getClusteredItemsStats uses the model preview (or ``/vehicles/<key>/
    default.jpg``) for the tile that owns the model's ``default_pc``, and only
    reaches for a config image on the other sub-cluster tiles -- preferring the
    one flagged ``isDefaultForSubCluster``, then the first by name.
    """
    model_candidates: list[str] = []
    model_preview = str(model_info.get("preview") or "").strip().lstrip("/")
    if model_preview:
        model_candidates.append(model_preview)
    # Stock ships default.jpg; mods commonly ship default.png instead.
    model_candidates.append(f"vehicles/{source_vehicle_id}/default.jpg")
    model_candidates.append(f"vehicles/{source_vehicle_id}/default.png")

    config_candidates: list[str] = []
    for config_name in (default_config_for_cluster, *cluster_configs):
        if config_name:
            config_candidates.append(f"vehicles/{source_vehicle_id}/{config_name}.jpg")
            config_candidates.append(f"vehicles/{source_vehicle_id}/{config_name}.png")

    if holds_model_default:
        # The model's own tile prefers the model image; config images are only a
        # last resort for mods that ship no default.jpg/png at all.
        candidates = model_candidates + config_candidates
    else:
        candidates = config_candidates + model_candidates
    return next((candidate for candidate in candidates if candidate in names_set), "")


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

            from beamxp.core.beam_json import load_info

            model_info: dict[str, object] = {}
            model_info_path = f"vehicles/{source_vehicle_id}/info.json"
            if model_info_path in names_set:
                try:
                    model_info = load_info(source_zip, model_info_path)
                except Exception:
                    model_info = {}
            vehicle_type = str(model_info.get("Type") or "").strip()
            translations = _selector_translations(source_zip)

            pc_paths = _direct_zip_files(names, source_vehicle_id, ".pc")
            config_names_all = [Path(pc_path).stem for pc_path in pc_paths]
            clusters: dict[str, list[str]] = {}
            cluster_defaults: dict[str, str] = {}
            default_pc = str(model_info.get("default_pc") or "")

            def _config_info_member(config_name: str) -> str | None:
                for candidate in (
                    f"vehicles/{source_vehicle_id}/info_{config_name}.json",
                    f"vehicles/{source_vehicle_id}/{config_name}.json",
                ):
                    if candidate in names_set:
                        return candidate
                return None

            # useSubCluster is a model-level opt-in that a config may override.
            # Parsing every config's info json to find out costs seconds across a
            # folder of 120+ zips, so when the model stays silent, look for the
            # key in the raw bytes first and only parse if some config sets it.
            splits = bool(model_info.get("useSubCluster"))
            if not splits:
                for config_name in config_names_all:
                    member = _config_info_member(config_name)
                    if member is None:
                        continue
                    try:
                        if b"useSubCluster" in zf.read(member):
                            splits = True
                            break
                    except (KeyError, OSError):
                        continue

            if splits:
                for config_name in config_names_all:
                    member = _config_info_member(config_name)
                    config_info: dict[str, object] = {}
                    if member is not None:
                        try:
                            config_info = load_info(source_zip, member)
                        except Exception:
                            config_info = {}
                    cluster = _config_cluster_name(model_info, config_info, translations)
                    clusters.setdefault(cluster, []).append(config_name)
                    if config_info.get("isDefaultForSubCluster"):
                        cluster_defaults.setdefault(cluster, config_name)
            elif config_names_all:
                clusters[""] = list(config_names_all)

            # One tile per vehicle unless the model opted into sub-clusters, so a
            # vehicle keeps every trim in a single entry (and a single project).
            if len(clusters) < 2:
                only_configs = next(iter(clusters.values()), [])
                entries.append(
                    VehicleCatalogEntry(
                        vehicle_id=source_vehicle_id,
                        source_vehicle_id=source_vehicle_id,
                        display_name=_tile_display_name(
                            model_info, source_vehicle_id, "", translations
                        ),
                        vehicle_type=vehicle_type,
                        preview_member=_tile_preview_member(
                            names_set,
                            model_info,
                            source_vehicle_id,
                            only_configs,
                            default_pc,
                            holds_model_default=True,
                        ),
                        config_count=len(only_configs),
                    )
                )
                continue

            # The cluster holding the model's own default config keeps the plain
            # vehicle id, so the primary tile's project path never moves.
            primary = next(
                (name for name, configs in sorted(clusters.items()) if default_pc in configs),
                max(sorted(clusters), key=lambda name: len(clusters[name])),
            )
            used_ids: set[str] = set()
            for cluster_name, config_names in sorted(clusters.items()):
                if cluster_name == primary:
                    vehicle_id = safe_project_segment(source_vehicle_id)
                else:
                    vehicle_id = safe_project_segment(f"{source_vehicle_id}_{cluster_name}".lower())
                if vehicle_id.lower() in used_ids:
                    suffix = 2
                    while f"{vehicle_id}_{suffix}".lower() in used_ids:
                        suffix += 1
                    vehicle_id = f"{vehicle_id}_{suffix}"
                used_ids.add(vehicle_id.lower())
                entries.append(
                    VehicleCatalogEntry(
                        vehicle_id=vehicle_id,
                        source_vehicle_id=source_vehicle_id,
                        config_names=tuple(sorted(config_names)),
                        display_name=_tile_display_name(
                            model_info, source_vehicle_id, cluster_name, translations
                        ),
                        vehicle_type=vehicle_type,
                        preview_member=_tile_preview_member(
                            names_set,
                            model_info,
                            source_vehicle_id,
                            sorted(config_names),
                            cluster_defaults.get(cluster_name, ""),
                            holds_model_default=default_pc in config_names,
                        ),
                        config_count=len(config_names),
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
