from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field, fields as dataclass_fields, replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np

import beamng_transform_helpers as transform_helpers
import plate_generator
import spatial_visibility_backend


def default_user_data_dir() -> Path:
    # Shared with plate_generator so the one-time BeamHDC -> BeamXP data
    # folder migration runs no matter which module resolves the path first.
    return plate_generator.default_user_data_dir()


def default_beamng_mods_dir() -> Path:
    local_appdata = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return local_appdata / "BeamNG" / "BeamNG.drive" / "current" / "mods"


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
SOURCE_ROOT_DIR = APP_DIR.parent if APP_DIR.name == "beamng-hand-drive-converter" else APP_DIR
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
# deliberately hidden (mods "remove" unwanted meshes by offsetting them
# thousands of km away) and are left out of previews so they cannot wreck
# the camera framing. Keep in sync with FAR_LIMIT in mesh_preview.py.
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


@dataclass(frozen=True)
class DaeObject:
    id: str
    name: str
    dae_path: str
    x: float
    y: float
    z: float
    geometry_ids: tuple[str, ...]
    dae_source_zip: Path | None = None


@dataclass(frozen=True)
class MeshPlacement:
    position: tuple[float, float, float]
    matrix: list[list[float]]


@dataclass(frozen=True)
class ResolvedMeshPosition:
    """Where a mesh sits in ONE configuration.

    Several placements within a single config are simultaneous instances (a
    wheel at four corners), so position is their average -- one DaeObject
    cannot represent four. matrices are the flexbody row matrices for that
    config, empty when the mesh is placed only as a prop."""

    position: tuple[float, float, float]
    matrices: tuple[tuple[tuple[float, ...], ...], ...] = ()


@dataclass(frozen=True)
class SlotDef:
    slot_type: str
    default_part: str
    options: str | None = None


@dataclass(frozen=True)
class BakedMeshSpec:
    configured_mesh: str
    source_mesh: str
    output_mesh: str
    target_hand: str
    mode: str
    placement_matrix: list[list[float]]
    bake_transform_into_dae: bool
    is_prop: bool = False


@dataclass
class SharedBakeContext:
    context: "VehicleContext"
    config_name: str
    target_hand: str
    source_part_id: str
    object_modes: dict[str, str]
    structural_sources: dict[str, str]
    translate_magnitudes: dict[str, float]
    baked_specs: list[BakedMeshSpec]


@dataclass(frozen=True)
class VariantInfo:
    name: str
    pc_path: str
    info_path: str | None
    display_name: str


@dataclass
class VehicleContext:
    source_zip: Path
    vehicle_id: str
    vehicle_path: str
    dae_paths: list[str]
    variants: dict[str, VariantInfo]
    objects: dict[str, DaeObject]
    preview_by_id: dict[str, dict[str, object]]
    jbeam_texts: dict[str, str]
    node_positions: dict[str, tuple[float, float, float]]
    project_dir: Path
    part_body_index: dict[str, tuple[str, str]] = field(default_factory=dict)
    jbeam_positioned_flexbodies: set[str] = field(default_factory=set)
    # Raw DAE node-matrix translations (mesh pivots), captured before mesh
    # positions get resolved/averaged. Props anchor their mesh pivot in
    # vehicle space, so hand conversion must transform pivot positions.
    mesh_pivots: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Authored geometry centre per mesh (world space, node translation
    # included), captured before apply_resolved_mesh_positions overwrites
    # preview_by_id. A flexbody row that authors no pos of its own renders at
    # this centre minus the node's own translation (see
    # flexbody_row_needs_node_translation) -- the node offset is a leftover
    # export artefact the game does not apply, not real placement.
    mesh_authored_centers: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # Meshes whose resolved position differs between trims -- i.e. declared by
    # parts that can never coexist. The single position on DaeObject is only a
    # representative for these; ask resolved_mesh_positions_for_config for the
    # value that is true in a given trim.
    variant_dependent_meshes: set[str] = field(default_factory=set)
    selected_parts_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    # Per-config resolved positions; rebuilt on demand, never pickled (it is
    # trims x meshes and would dwarf the rest of the cache).
    resolved_positions_cache: dict[str, dict[str, ResolvedMeshPosition]] = field(default_factory=dict)
    mesh_roles_cache: dict[str, tuple[set[str], set[str], set[str]]] = field(default_factory=dict)
    selected_node_positions_cache: dict[str, dict[str, tuple[float, float, float]]] = field(default_factory=dict)
    part_array_cache: dict[tuple[str, str], str | None] = field(default_factory=dict)
    variant_hands_cache: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class BuildResult:
    unpacked_dir: Path
    package_zip: Path | None
    installed_zip: Path | None
    generated_configs: list[str]
    generated_daes: list[Path]
    skipped_variants: dict[str, str]
    plate_summary: dict[str, object] = field(default_factory=dict)
    installed_plates_zip: Path | None = None


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


def parse_dae(source_zip: Path, dae_path: str) -> ET.ElementTree:
    with zipfile.ZipFile(source_zip) as zf:
        with zf.open(dae_path) as fh:
            return ET.parse(fh)


def dae_objects_from_tree(
    tree: ET.ElementTree,
    dae_path: str,
    *,
    dae_source_zip: Path | None = None,
) -> dict[str, DaeObject]:
    objects: dict[str, DaeObject] = {}
    for node in tree.getroot().findall(".//c:node", NS):
        object_id = node.get("id")
        if not object_id:
            continue
        instance_geometries = node.findall(".//c:instance_geometry", NS)
        if not instance_geometries:
            continue
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = transform_helpers.parse_matrix(matrix_elem.text)
        geometry_ids = tuple(
            inst.get("url", "")[1:]
            for inst in instance_geometries
            if inst.get("url", "").startswith("#")
        )
        obj = DaeObject(
            id=object_id,
            name=(node.get("name") or object_id).strip(),
            dae_path=dae_path,
            x=matrix[0][3],
            y=matrix[1][3],
            z=matrix[2][3],
            geometry_ids=geometry_ids,
            dae_source_zip=dae_source_zip,
        )
        for alias in dae_node_aliases(node):
            objects.setdefault(alias, obj)
    return objects


def dae_node_aliases(node: ET.Element) -> list[str]:
    aliases: list[str] = []
    for value in (node.get("id"), node.get("name")):
        if not value:
            continue
        for alias in (value, value.strip()):
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


def find_dae_node(root: ET.Element, object_id: str) -> ET.Element | None:
    node = root.find(f".//c:node[@id='{object_id}']", NS)
    if node is not None:
        return node
    for candidate in root.findall(".//c:node", NS):
        if object_id in dae_node_aliases(candidate):
            return candidate
    return None


def list_dae_objects_for_file(source_zip: Path, dae_path: str) -> dict[str, DaeObject]:
    tree = parse_dae(source_zip, dae_path)
    return dae_objects_from_tree(tree, dae_path, dae_source_zip=source_zip)


def list_dae_objects_for_path(path: Path) -> dict[str, DaeObject]:
    return dae_objects_from_tree(ET.parse(path), str(path), dae_source_zip=None)


_game_common_zips_cache: list[Path] | None = None


def beamng_game_common_zips() -> list[Path]:
    """The game install's content/vehicles/common.zip. Mod zips live in the
    user's mods folder with no sibling common.zip, yet routinely reference
    vanilla wheels/tires/props from vehicles/common - without this lookup
    those parts have no jbeam bodies and no DAE geometry. Candidates come from
    the app settings' recently opened game folders and from Steam's install
    metadata (registry + libraryfolders.vdf). Cached per process."""
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

    # Folders vehicles were opened from before; the game's content folder is
    # recorded here as soon as any vanilla vehicle has been opened.
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

    # Steam installs, including secondary library folders on other drives.
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


def common_dae_paths(source_zip: Path) -> list[str]:
    try:
        with zipfile.ZipFile(source_zip) as zf:
            return sorted(
                name.replace("\\", "/")
                for name in zf.namelist()
                if name.replace("\\", "/").lower().startswith("vehicles/common/")
                and name.lower().endswith(".dae")
            )
    except Exception:
        return []


def referenced_mesh_names(part_body_index: dict[str, tuple[str, str]]) -> set[str]:
    meshes: set[str] = set()
    for part_body, _filename in part_body_index.values():
        meshes.update(transform_helpers.extract_part_mesh_names(part_body))
    return meshes


DAE_ALIAS_ATTR_RE = re.compile(rb'(?:id|name)="([^"]*)"')


def dae_alias_candidates(data: bytes) -> set[str]:
    """Every id/name attribute value in a raw DAE, plus stripped forms.

    A superset of what dae_node_aliases can key an object by, so using it to
    skip files is safe: it can only ever over-select. Matching on attributes
    rather than searching for each wanted mesh name matters a great deal --
    a combined ``mesh1|mesh2|...`` regex over a 680 MB common.zip is O(bytes
    x alternatives) in Python's re (no Aho-Corasick) and measured 211s on
    pickup, versus 0.4s for one attribute pass plus a set intersection.
    """
    names: set[str] = set()
    for raw in DAE_ALIAS_ATTR_RE.findall(data):
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        names.add(value)
        stripped = value.strip()
        if stripped:
            names.add(stripped)
    return names


def load_common_dae_objects(
    source_zip: Path,
    wanted_meshes: set[str],
    existing_objects: dict[str, DaeObject],
) -> tuple[dict[str, DaeObject], dict[str, dict[str, object]], list[str]]:
    still_missing = wanted_meshes - set(existing_objects)
    if not still_missing:
        return {}, {}, []

    found_objects: dict[str, DaeObject] = {}
    found_previews: dict[str, dict[str, object]] = {}
    found_paths: set[str] = set()

    for candidate_zip in common_zip_candidates(source_zip):
        if not still_missing:
            break
        paths = common_dae_paths(candidate_zip)
        if not paths:
            continue
        try:
            zf = zipfile.ZipFile(candidate_zip)
        except Exception:
            continue
        # One handle for the whole zip: reopening per DAE re-reads the
        # central directory of a multi-hundred-MB archive every time.
        with zf:
            for dae_path in paths:
                if not still_missing:
                    break
                try:
                    data = zf.read(dae_path)
                except Exception:
                    continue
                if still_missing.isdisjoint(dae_alias_candidates(data)):
                    continue

                try:
                    tree = ET.ElementTree(ET.fromstring(data))
                    file_objects = dae_objects_from_tree(
                        tree, dae_path, dae_source_zip=candidate_zip
                    )
                except Exception:
                    continue
                matched_ids = sorted(
                    object_id for object_id in file_objects if object_id in still_missing
                )
                if not matched_ids:
                    continue

                try:
                    file_previews = preview_data_from_tree(tree)
                except Exception:
                    file_previews = {}
                for object_id in matched_ids:
                    found_objects.setdefault(object_id, file_objects[object_id])
                    if object_id in file_previews:
                        found_previews.setdefault(object_id, file_previews[object_id])
                    still_missing.discard(object_id)
                found_paths.add(dae_path)

    return found_objects, found_previews, sorted(found_paths)


def geometry_position_points(geometry: ET.Element) -> list[tuple[float, float, float]]:
    return transform_helpers.geometry_position_points(geometry)


def preview_data_for_file(
    source_zip: Path,
    dae_path: str,
    max_points_per_object: int = 350,
) -> dict[str, dict[str, object]]:
    return preview_data_from_tree(
        parse_dae(source_zip, dae_path),
        max_points_per_object=max_points_per_object,
    )


def preview_data_from_tree(
    tree: ET.ElementTree,
    max_points_per_object: int = 350,
) -> dict[str, dict[str, object]]:
    """Preview payload from an already-parsed DAE.

    Split out so callers that also need dae_objects_from_tree can parse the
    file once instead of once per helper; a common DAE is tens of MB of XML.
    """
    root = tree.getroot()
    library_geometries = root.find("c:library_geometries", NS)
    if library_geometries is None:
        return {}

    geometries_by_id = {
        geom.get("id"): geom
        for geom in library_geometries.findall("c:geometry", NS)
        if geom.get("id")
    }
    local_points_by_geometry = {
        geom_id: geometry_position_points(geom)
        for geom_id, geom in geometries_by_id.items()
    }

    preview: dict[str, dict[str, object]] = {}
    for node in root.findall(".//c:node", NS):
        object_id = node.get("id")
        if not object_id:
            continue
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = transform_helpers.parse_matrix(matrix_elem.text)

        object_points: list[tuple[float, float, float]] = []
        geometry_ids: list[str] = []
        for inst in node.findall(".//c:instance_geometry", NS):
            url = inst.get("url", "")
            if not url.startswith("#"):
                continue
            geometry_id = url[1:]
            geometry_ids.append(geometry_id)
            local_points = local_points_by_geometry.get(geometry_id, [])
            object_points.extend(transform_helpers.transform_point(matrix, point) for point in local_points)

        if not object_points:
            continue
        # Material bindings feed the spatial classifier (glass panes bound the
        # cabin; emissive surfaces are displays that need a texture flip).
        material_symbols = sorted({
            re.sub(r"-material$", "", symbol).lower()
            for inst_mat in node.findall(".//c:instance_material", NS)
            for symbol in (inst_mat.get("symbol") or inst_mat.get("target", "").lstrip("#"),)
            if symbol
        })
        bounds = transform_helpers.bounds_from_points(object_points)
        min_point, max_point = bounds
        center = (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        )
        item = {
            "bounds": bounds,
            "center": center,
            "sample_points": transform_helpers.sample_points(object_points, max_points_per_object),
            "geometry_ids": geometry_ids,
            "materials": tuple(material_symbols),
        }
        for alias in dae_node_aliases(node):
            preview.setdefault(alias, item)
    return preview


def surface_triangles_from_tree(
    tree: ET.ElementTree,
) -> dict[str, np.ndarray]:
    """Filled triangle surfaces for each object in an already-parsed DAE."""
    root = tree.getroot()
    library_geometries = root.find("c:library_geometries", NS)
    if library_geometries is None:
        return {}

    local_by_geometry: dict[str, np.ndarray] = {}
    for geometry in library_geometries.findall("c:geometry", NS):
        geometry_id = geometry.get("id")
        if not geometry_id:
            continue
        triangles = transform_helpers.geometry_surface_triangles(geometry)
        local_by_geometry[geometry_id] = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))

    surfaces: dict[str, np.ndarray] = {}
    for node in root.findall(".//c:node", NS):
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = np.asarray(transform_helpers.parse_matrix(matrix_elem.text), dtype=float)
        chunks: list[np.ndarray] = []
        for instance in node.findall(".//c:instance_geometry", NS):
            url = instance.get("url", "")
            if not url.startswith("#"):
                continue
            local = local_by_geometry.get(url[1:])
            if local is None or len(local) == 0:
                continue
            flat = local.reshape((-1, 3))
            homogeneous = np.concatenate(
                [flat, np.ones((len(flat), 1), dtype=float)], axis=1
            )
            chunks.append((homogeneous @ matrix.T)[:, :3].reshape((-1, 3, 3)))
        if not chunks:
            continue
        triangles = np.concatenate(chunks)
        for alias in dae_node_aliases(node):
            surfaces.setdefault(alias, triangles)
    return surfaces


def dae_source_index(
    context: VehicleContext,
) -> tuple[
    dict[str, tuple[str, str]],
    dict[tuple[str, str], tuple[str, ...]],
    dict[tuple[str, str], tuple[Path, str]],
]:
    """Stable DAE source keys and file membership for every context object.

    Shared accessory packs make source paths significant, but resolving the
    same zip path inside every per-object/per-trim cache lookup is extremely
    expensive on Windows.  Build the immutable index once, resolving each
    distinct zip only once.
    """
    cached = getattr(context, "_dae_source_index", None)
    if cached is not None:
        return cached

    resolved_zips: dict[str, str] = {}
    keys: dict[str, tuple[str, str]] = {}
    members: dict[tuple[str, str], list[str]] = {}
    files: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id, obj in context.objects.items():
        source_zip = Path(obj.dae_source_zip or context.source_zip)
        raw_zip = str(source_zip)
        zip_key = resolved_zips.get(raw_zip)
        if zip_key is None:
            try:
                zip_key = str(source_zip.resolve(strict=False)).lower()
            except OSError:
                zip_key = raw_zip.lower()
            resolved_zips[raw_zip] = zip_key
        dae_path = obj.dae_path.replace("\\", "/")
        key = (zip_key, dae_path)
        keys[object_id] = key
        members.setdefault(key, []).append(object_id)
        files.setdefault(key, (source_zip, obj.dae_path))

    result = (
        keys,
        {key: tuple(object_ids) for key, object_ids in members.items()},
        files,
    )
    context._dae_source_index = result
    return result


def full_surface_triangles_for_ids(
    context: VehicleContext,
    ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """DAE triangle surfaces aligned to the representative placed previews.

    Each source file is parsed once for surfaces, including shared accessory
    packs referenced through ``dae_source_zip``.  Missing or malformed
    primitives yield an empty surface so callers can safely retain the older
    point-shell result as their fallback.
    """
    surfaces = getattr(context, "_surface_triangles", None)
    if surfaces is None:
        surfaces = {}
        context._surface_triangles = surfaces
    authored_surfaces = getattr(context, "_authored_surface_triangles", None)
    if authored_surfaces is None:
        authored_surfaces = {}
        context._authored_surface_triangles = authored_surfaces
    parsed_files = getattr(context, "_surface_triangle_files", None)
    if parsed_files is None:
        parsed_files = set()
        context._surface_triangle_files = parsed_files

    requested = {str(object_id) for object_id in ids}
    source_keys, source_members, source_files = dae_source_index(context)

    by_file: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id in requested:
        if object_id in surfaces:
            continue
        obj = context.objects.get(object_id)
        if obj is None or not obj.dae_path:
            continue
        key = source_keys.get(object_id)
        if key is not None:
            by_file.setdefault(key, source_files[key])

    for key, (source_zip, dae_path) in by_file.items():
        if key in parsed_files:
            continue
        parsed_files.add(key)
        try:
            tree = parse_dae(source_zip, dae_path)
            file_surfaces = surface_triangles_from_tree(tree)
            authored_centers = getattr(context, "_authored_full_centers", {})
            needs_preview = any(
                object_id not in authored_centers
                for object_id in source_members.get(key, ())
            )
            file_preview = (
                preview_data_from_tree(tree, max_points_per_object=sys.maxsize)
                if needs_preview else {}
            )
        except Exception:
            file_surfaces = {}
            file_preview = {}

        clouds = getattr(context, "_full_clouds", None)
        if clouds is None:
            clouds = {}
            context._full_clouds = clouds
        authored_clouds = getattr(context, "_authored_full_clouds", None)
        if authored_clouds is None:
            authored_clouds = {}
            context._authored_full_clouds = authored_clouds
        authored_centers = getattr(context, "_authored_full_centers", None)
        if authored_centers is None:
            authored_centers = {}
            context._authored_full_centers = authored_centers
        full_files = getattr(context, "_full_cloud_files", None)
        if full_files is None:
            full_files = set()
            context._full_cloud_files = full_files
        full_files.add(key)

        for object_id in source_members.get(key, ()):
            triangles = file_surfaces.get(object_id)
            if triangles is None or len(triangles) == 0:
                surfaces.setdefault(object_id, np.empty((0, 3, 3), dtype=float))
            authored = file_preview.get(object_id)
            placed = context.preview_by_id.get(object_id)
            if authored is not None:
                points = np.asarray(authored.get("sample_points", ()), dtype=float)
                authored_center = np.asarray(authored.get("center"), dtype=float)
                if len(points):
                    authored_clouds[object_id] = points
                    if authored_center.shape == (3,):
                        authored_centers[object_id] = authored_center
                    if placed is not None:
                        placed_center = np.asarray(placed.get("center"), dtype=float)
                        if authored_center.shape == (3,) and placed_center.shape == (3,):
                            points = points + (placed_center - authored_center)
                    clouds[object_id] = points
            if triangles is None or len(triangles) == 0:
                continue
            authored_surfaces[object_id] = triangles
            authored_center = authored_centers.get(object_id)
            if authored_center is not None and placed is not None:
                authored_center = np.asarray(authored_center, dtype=float)
                placed_center = np.asarray(placed.get("center"), dtype=float)
                if authored_center.shape == (3,) and placed_center.shape == (3,):
                    triangles = triangles + (placed_center - authored_center)
            surfaces[object_id] = triangles

    return {
        object_id: surfaces.get(object_id, np.empty((0, 3, 3), dtype=float))
        for object_id in requested
    }


def full_vertex_clouds_for_ids(
    context: VehicleContext,
    ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Uncapped DAE vertex clouds, aligned to the placed preview centres.

    Preview clouds are deliberately capped for interactive work, but striding
    and truncating a vertex buffer can split exact mirror pairs.  Self-
    symmetry therefore needs every authored vertex.  Each source DAE is
    parsed at most once, including DAEs supplied by ``dae_source_zip``; any
    unreadable mesh falls back to its preview cloud so failure favours a
    benign extra mirror rather than a missed asymmetric part.

    JBeam-placed shared meshes are commonly authored at the origin.  Their
    full clouds are translated onto the representative preview bbox centre.
    Per-trim x translation is applied by the classifier, which has the trim's
    resolved preview in hand.
    """
    clouds = getattr(context, "_full_clouds", None)
    if clouds is None:
        clouds = {}
        context._full_clouds = clouds
    authored_clouds = getattr(context, "_authored_full_clouds", None)
    if authored_clouds is None:
        authored_clouds = {}
        context._authored_full_clouds = authored_clouds
    authored_centers = getattr(context, "_authored_full_centers", None)
    if authored_centers is None:
        authored_centers = {}
        context._authored_full_centers = authored_centers
    parsed_files = getattr(context, "_full_cloud_files", None)
    if parsed_files is None:
        parsed_files = set()
        context._full_cloud_files = parsed_files

    requested = {str(object_id) for object_id in ids}
    source_keys, source_members, source_files = dae_source_index(context)

    by_file: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id in requested:
        if object_id in clouds:
            continue
        obj = context.objects.get(object_id)
        if obj is None or not obj.dae_path:
            continue
        key = source_keys.get(object_id)
        if key is not None:
            by_file.setdefault(key, source_files[key])

    for key, (source_zip, dae_path) in by_file.items():
        if key in parsed_files:
            continue
        parsed_files.add(key)
        try:
            full_preview = preview_data_from_tree(
                parse_dae(source_zip, dae_path),
                max_points_per_object=sys.maxsize,
            )
        except Exception:
            full_preview = {}

        # Cache every context object backed by this DAE.  A later trim can ask
        # about a different mesh in the same file without forcing a reparse.
        for object_id in source_members.get(key, ()):
            authored = full_preview.get(object_id)
            placed = context.preview_by_id.get(object_id)
            if authored is None or placed is None:
                continue
            points = np.asarray(authored.get("sample_points", ()), dtype=float)
            if len(points) == 0:
                continue
            authored_clouds[object_id] = points
            authored_center = np.asarray(authored.get("center"), dtype=float)
            if authored_center.shape == (3,):
                authored_centers[object_id] = authored_center
            placed_center = np.asarray(placed.get("center"), dtype=float)
            if authored_center.shape == (3,) and placed_center.shape == (3,):
                points = points + (placed_center - authored_center)
            clouds[object_id] = points

    result: dict[str, np.ndarray] = {}
    for object_id in requested:
        points = clouds.get(object_id)
        if points is None:
            preview = context.preview_by_id.get(object_id, {})
            points = np.asarray(preview.get("sample_points", ()), dtype=float)
            clouds[object_id] = points
        result[object_id] = points
    return result


def vertex_cloud_for_resolved_placement(
    context: VehicleContext,
    object_id: str,
    resolved: ResolvedMeshPosition,
    max_points: int = 350,
) -> np.ndarray | None:
    """Rebuild one flexbody cloud from its authored DAE and trim matrices.

    Representative previews can contain a different instance count from the
    current trim.  Apply every real placement and return their sampled union;
    this avoids both phantom twins in single-seat trims and fictitious averaged
    positions in two-seat/four-wheel trims.
    """
    if not resolved.matrices:
        return None
    authored = getattr(context, "_authored_full_clouds", {}).get(object_id)
    if authored is None:
        full_vertex_clouds_for_ids(context, (object_id,))
        authored = getattr(context, "_authored_full_clouds", {}).get(object_id)
    if authored is None or len(authored) == 0:
        return None
    homogeneous = np.concatenate(
        [np.asarray(authored, dtype=float), np.ones((len(authored), 1), dtype=float)],
        axis=1,
    )
    chunks = [
        (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3]
        for matrix in resolved.matrices
    ]
    points = np.concatenate(chunks)
    if object_id not in context.jbeam_positioned_flexbodies:
        pivot = context.mesh_pivots.get(object_id)
        if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
            points = points - np.asarray(pivot, dtype=float)
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride][:max_points]
    return points


def surface_triangles_for_resolved_placement(
    context: VehicleContext,
    object_id: str,
    resolved: ResolvedMeshPosition,
) -> np.ndarray | None:
    """Rebuild one object's authored surface at all of its trim matrices."""
    if not resolved.matrices:
        return None
    authored = getattr(context, "_authored_surface_triangles", {}).get(object_id)
    if authored is None:
        full_surface_triangles_for_ids(context, (object_id,))
        authored = getattr(context, "_authored_surface_triangles", {}).get(object_id)
    if authored is None or len(authored) == 0:
        return None
    flat = np.asarray(authored, dtype=float).reshape((-1, 3))
    homogeneous = np.concatenate(
        [flat, np.ones((len(flat), 1), dtype=float)], axis=1
    )
    triangles = np.concatenate([
        (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3].reshape((-1, 3, 3))
        for matrix in resolved.matrices
    ])
    if object_id not in context.jbeam_positioned_flexbodies:
        pivot = context.mesh_pivots.get(object_id)
        if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
            triangles = triangles - np.asarray(pivot, dtype=float)
    return triangles


def load_jbeam_texts(source_zip: Path, vehicle_id: str) -> dict[str, str]:
    prefix = f"{vehicle_prefix(vehicle_id)}/"
    with zipfile.ZipFile(source_zip) as zf:
        return {
            name.replace("\\", "/"): zf.read(name).decode("utf-8", errors="replace")
            for name in zf.namelist()
            if name.replace("\\", "/").startswith(prefix)
            and name.lower().endswith(".jbeam")
        }


def brg_rotation_matrix3(brg_rad: Iterable[float]) -> list[list[float]]:
    """Engine getBaseRotationGlobal() euler (radians) -> rest rotation 3x3.

    This is the prop's ENGINE rest rotation. The engine stores the base
    orientation as a quaternion (always a proper rotation, det +1); for rows
    without an authored baseRotationGlobal its value does NOT equal the
    analytic frame*baseRotation composition, so dumped values are the only
    exact source for those rows.

    Convention: the engine's euler works in row-vector (transposed) matrix
    convention, so the column-vector rest matrix is the TRANSPOSE of
    Ry(y)*Rz(z)*Rx(x). Ground truth: grp_steerwheel_hub dumps x=+70deg and
    its -z-authored quick-release boss must point forward-down along the
    steering column (Rx(-70)); the wheel face likewise. All other validated
    rows (identity, 180deg flips, symmetric discs) are sign-invariant."""
    x, y, z = (math.degrees(float(v)) for v in brg_rad)
    matrix = identity_matrix()
    for next_matrix in (rotation_y_matrix(y), rotation_z_matrix(z), rotation_x_matrix(x)):
        matrix = multiply_matrix(matrix, next_matrix)
    return rotation_transpose_matrix3(matrix3_from_matrix4(matrix))


def rotation_transpose_matrix3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def prop_rest_rotation_override(
    row: str,
    node_positions: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[list[list[float]] | None, str]:
    """Resolve a prop row's REST rotation and report the source.

    The engine renders props from NODE-LOCAL geometry: the DAE node's
    ROTATION is discarded at load (its translation becomes the pivot, scale
    is kept) and baseRotationGlobal - authored or engine-computed - IS the
    rest rotation applied to that local geometry. Renderers must therefore
    pair these rotations with derotated node transforms (see
    mesh_preview/blender backend). Ground truth: bx_steer's node carries
    Rx(~73.5deg) over flat-authored verts and dumps brg 71.6deg; the
    sunburst2 driveshaft node carries Rx(180) and authors brg -176.75 (the
    3.25deg difference is the deliberate driveline angle). Meshes with
    rotation-free nodes (grp_*, steeringwheels) are unaffected.

    Resolution order: authored baseRotationGlobal (the field is the rest
    rotation verbatim), then the analytic engine model. None means the caller
    falls back to prop_row_global_rotation_matrix, only approximate for rows
    without authored brg."""
    authored = vector_from_row(row, "baseRotationGlobal")
    if authored is not None:
        return (
            brg_rotation_matrix3([math.radians(float(v)) for v in authored]),
            "authored-brg",
        )
    if node_positions is not None:
        rotation = prop_engine_rest_rotation(row, node_positions)
        if rotation is not None:
            return rotation, "analytic-engine"
    return None, "analytic"


def prop_engine_rest_rotation(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> list[list[float]] | None:
    """ENGINE-EXACT rest rotation for prop rows without authored brg.

    rest = F * B^T, where F is the RIGHT-handed triad frame
      x = norm(idX - idRef), y = norm((idY - idRef) x x), z = x x y
    and B is prop_base_rotation_matrix3 (the meshs.lua "-X -Z +Y" euler) -
    transposed because the engine composes in row-vector convention like
    every other euler here. Returns None on degenerate triads."""
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    ref_pos = node_positions.get(strings[2])
    x_pos = node_positions.get(strings[3])
    y_pos = node_positions.get(strings[4])
    if ref_pos is None or x_pos is None or y_pos is None:
        return None
    axis_x = normalize_vector(vector_subtract(x_pos, ref_pos))
    seed = normalize_vector(vector_subtract(y_pos, ref_pos))
    axis_y = normalize_vector(cross_product(seed, axis_x))
    if axis_x == (0.0, 0.0, 0.0) or axis_y == (0.0, 0.0, 0.0):
        return None
    axis_z = cross_product(axis_x, axis_y)
    frame = matrix3_from_axes(axis_x, axis_y, axis_z)
    vectors = prop_row_vector_objects(row)
    base = vectors[0] if vectors else (0.0, 0.0, 0.0)
    return multiply_matrix3(
        frame,
        rotation_transpose_matrix3(prop_base_rotation_matrix3(base)),
    )


def load_common_jbeam_texts(source_zip: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for candidate_zip in common_zip_candidates(source_zip):
        try:
            with zipfile.ZipFile(candidate_zip) as zf:
                for name in zf.namelist():
                    norm = name.replace("\\", "/")
                    if (
                        norm.lower().startswith("vehicles/common/")
                        and norm.lower().endswith(".jbeam")
                        and norm not in texts
                    ):
                        texts[norm] = zf.read(name).decode("utf-8", errors="replace")
        except Exception:
            continue
    return texts


def slot_demand_types(part_body: str) -> set[str]:
    demanded: set[str] = set()
    slots = transform_helpers.extract_named_array(part_body, "slots")
    if slots:
        for row in iter_active_top_level_rows(slots):
            values = split_top_level_values(row)
            if len(values) < 2:
                continue
            slot_type = quoted_string_value(values[0])
            if slot_type and slot_type not in {"type", "name"}:
                demanded.add(slot_type)
    slots2 = transform_helpers.extract_named_array(part_body, "slots2")
    if slots2:
        for row in iter_active_top_level_rows(slots2):
            values = split_top_level_values(row)
            if len(values) < 4:
                continue
            name = quoted_string_value(values[0])
            if not name or name in {"name", "type"}:
                continue
            demanded.add(name)
            demanded.update(re.findall(r'"((?:[^"\\]|\\.)*)"', values[1]))
    return demanded


def reachable_common_part_index(
    vehicle_part_index: dict[str, tuple[str, str]],
    common_part_index: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Parts defined under vehicles/common are only indexed when the vehicle's
    slot graph can actually pull them in, keeping the part inventory focused."""
    parts_by_slot_type: dict[str, list[str]] = {}
    for part_id, (body, _filename) in common_part_index.items():
        for slot_type in transform_helpers.extract_part_slot_types(body):
            parts_by_slot_type.setdefault(slot_type, []).append(part_id)

    demanded: set[str] = set()
    reachable: dict[str, tuple[str, str]] = {}
    pending = [body for body, _filename in vehicle_part_index.values()]
    while pending:
        body = pending.pop()
        # sorted(): slot_demand_types returns a set, and Python randomises str
        # hashing per process, so unsorted iteration made this dict's insertion
        # order vary run to run. That order reaches collect_flexbody_mesh_placements
        # and therefore which points sample_points keeps, making previews (and the
        # context cache) irreproducible between runs over identical input.
        for slot_type in sorted(slot_demand_types(body)):
            if slot_type in demanded:
                continue
            demanded.add(slot_type)
            for part_id in parts_by_slot_type.get(slot_type, []):
                if part_id in reachable or part_id in vehicle_part_index:
                    continue
                entry = common_part_index[part_id]
                reachable[part_id] = entry
                pending.append(entry[0])
    return reachable


def extract_node_positions_from_array(array_text: str) -> dict[str, tuple[float, float, float]]:
    node_re = re.compile(
        rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
        rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})',
        re.MULTILINE,
    )
    nodes: dict[str, tuple[float, float, float]] = {}
    for match in node_re.finditer(array_text):
        node_id = match.group("id")
        if node_id in {"id", "type", "mesh", "func"}:
            continue
        nodes[node_id] = (
            float(match.group("x")),
            float(match.group("y")),
            float(match.group("z")),
        )
    return nodes


def build_node_position_index(jbeam_texts: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    nodes: dict[str, tuple[float, float, float]] = {}
    pattern = re.compile(r'"nodes"\s*:[\s,]*\[')
    for text in jbeam_texts.values():
        for match in pattern.finditer(text):
            bracket = text.rfind("[", match.start(), match.end())
            if bracket < 0:
                continue
            try:
                end = transform_helpers.find_matching(text, bracket, "[", "]")
            except Exception:
                continue
            for node_id, position in extract_node_positions_from_array(text[bracket:end]).items():
                nodes.setdefault(node_id, position)
    return nodes


def build_part_body_index(jbeam_texts: dict[str, str]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    # [\s,]* tolerates the stray comma stock jbeam ships between the colon
    # and the brace ("bluebuck_bumper_F":, {...}); the game accepts it.
    key_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:[\s,]*\{')
    for filename, text in jbeam_texts.items():
        for match in key_pattern.finditer(text):
            part_id = match.group(1)
            if part_id in index:
                continue
            brace = text.find("{", match.start(), match.end())
            if brace < 0:
                continue
            try:
                end = transform_helpers.find_matching(text, brace, "{", "}")
            except Exception:
                continue
            body = text[match.start() : end]
            if '"slotType"' not in body:
                continue
            index[part_id] = (body, filename)
    return index


TOP_LEVEL_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def iter_top_level_rows(array_text: str) -> list[str]:
    rows: list[str] = []
    idx = 1 if array_text.startswith("[") else 0
    length = len(array_text)
    while idx < length:
        ch = array_text[idx]
        if ch == '"':
            match = TOP_LEVEL_STRING_RE.match(array_text, idx)
            idx = match.end() if match else idx + 1
            continue
        if ch == "[":
            try:
                end = transform_helpers.find_matching(array_text, idx, "[", "]")
            except ValueError:
                # Stock jbeam ships the odd row whose quotes/brackets never
                # balance (usually inside a commented-out line); skip the
                # bracket instead of failing the whole array.
                idx += 1
                continue
            rows.append(array_text[idx:end])
            idx = end
            continue
        idx += 1
    return rows


def iter_active_top_level_rows(array_text: str) -> list[str]:
    """Like iter_top_level_rows, but skips rows that are commented out
    (``//`` line comments and ``/* */`` block comments), matching what the
    game's jbeam parser actually loads. Used by the preview payloads and by
    slot resolution (commented-out slot rows must not select parts); the
    build path keeps iter_top_level_rows so commented text is preserved
    verbatim in rewritten jbeam."""
    rows: list[str] = []
    idx = 1 if array_text.startswith("[") else 0
    length = len(array_text)
    while idx < length:
        ch = array_text[idx]
        if ch == "/" and array_text.startswith("//", idx):
            newline = array_text.find("\n", idx)
            idx = length if newline < 0 else newline + 1
            continue
        if ch == "/" and array_text.startswith("/*", idx):
            close = array_text.find("*/", idx + 2)
            idx = length if close < 0 else close + 2
            continue
        if ch == '"':
            match = TOP_LEVEL_STRING_RE.match(array_text, idx)
            idx = match.end() if match else idx + 1
            continue
        if ch == "[":
            try:
                end = transform_helpers.find_matching(array_text, idx, "[", "]")
            except ValueError:
                idx += 1
                continue
            rows.append(array_text[idx:end])
            idx = end
            continue
        idx += 1
    return rows


def split_top_level_values(row: str) -> list[str]:
    text = row.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    values: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "[{":
            depth += 1
            continue
        if ch in "]}":
            depth -= 1
            continue
        if ch == "," and depth == 0:
            values.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        values.append(tail)
    return values


def quoted_string_value(value: str) -> str | None:
    match = re.match(r'\s*"((?:[^"\\]|\\.)*)"', value)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return match.group(1)


def trailing_options_object(values: list[str]) -> str | None:
    if not values:
        return None
    value = values[-1].strip()
    if value.startswith("{") and value.endswith("}"):
        return value
    return None


def extract_slot_defs(part_body: str) -> list[SlotDef]:
    out: list[SlotDef] = []
    seen: set[str] = set()

    slots = transform_helpers.extract_named_array(part_body, "slots")
    if slots:
        for row in iter_active_top_level_rows(slots):
            values = split_top_level_values(row)
            if len(values) < 2:
                continue
            slot_type = quoted_string_value(values[0])
            default_part = quoted_string_value(values[1])
            if not slot_type or slot_type in {"type", "name"} or default_part is None:
                continue
            out.append(SlotDef(slot_type, default_part, trailing_options_object(values)))
            seen.add(slot_type)

    slots2 = transform_helpers.extract_named_array(part_body, "slots2")
    if slots2:
        for row in iter_active_top_level_rows(slots2):
            values = split_top_level_values(row)
            if len(values) < 4:
                continue
            slot_type = quoted_string_value(values[0])
            default_part = quoted_string_value(values[3])
            if not slot_type or slot_type in {"type", "name"} or default_part is None or slot_type in seen:
                continue
            out.append(SlotDef(slot_type, default_part, trailing_options_object(values)))
            seen.add(slot_type)

    return out


def vector_from_row(row: str, key: str) -> tuple[float, float, float] | None:
    """A row's "pos"/"rot"/"scale" vector, one component at a time.

    Some stock content makes a component a jbeam variable expression instead
    of a literal -- e.g. the D-Series heavy hub's spacer:
    "pos":{"x":"$=-$trackoffset_F-0.885", "y":-1.463, "z":0.46}. Reading x/y/z
    independently via object_number_property (which already falls back to
    approximate_expression_number, the same machinery node_transform_ops uses
    for nodeOffset/nodeMove) means a literal y and z are recovered even when x
    is an expression. The previous all-three-or-nothing regex discarded the
    whole vector in that case -- not just the unparseable x, but the entirely
    literal y and z along with it -- and any node_translation_offset that
    reads this vector's sign lost the L/R distinction it exists to make."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', row)
    if match is None:
        return None
    brace = row.rfind("{", match.start(), match.end())
    try:
        end = transform_helpers.find_matching(row, brace, "{", "}")
    except ValueError:
        return None
    object_text = row[brace:end]
    x = object_number_property(object_text, "x")
    y = object_number_property(object_text, "y")
    z = object_number_property(object_text, "z")
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def cross_product(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def normalize_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (value[0] / length, value[1] / length, value[2] / length)


def prop_row_position(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    inherited_options: Iterable[str] = (),
) -> tuple[float, float, float] | None:
    global_translation = vector_from_row(row, "baseTranslationGlobal")
    if global_translation is not None:
        return pos_after_node_transforms(row, global_translation, inherited_options)

    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    func, mesh, id_ref = strings[:3]
    if func == "func" or mesh == "mesh":
        return None
    ref_pos = node_positions.get(id_ref)
    if ref_pos is None:
        return None
    local_translation = vector_from_row(row, "baseTranslation") or (0.0, 0.0, 0.0)
    if len(strings) < 5:
        return ref_pos
    id_x, id_y = strings[3], strings[4]
    x_pos = node_positions.get(id_x)
    y_pos = node_positions.get(id_y)
    if x_pos is None or y_pos is None:
        position = (
            ref_pos[0] + local_translation[0],
            ref_pos[1] + local_translation[1],
            ref_pos[2] + local_translation[2],
        )
        return pos_after_node_transforms(row, position, inherited_options)

    axis_x = vector_subtract(x_pos, ref_pos)
    axis_y = vector_subtract(y_pos, ref_pos)
    axis_z = normalize_vector(cross_product(axis_y, axis_x))
    position = (
        ref_pos[0] + axis_x[0] * local_translation[0] + axis_y[0] * local_translation[1] + axis_z[0] * local_translation[2],
        ref_pos[1] + axis_x[1] * local_translation[0] + axis_y[1] * local_translation[1] + axis_z[1] * local_translation[2],
        ref_pos[2] + axis_x[2] * local_translation[0] + axis_y[2] * local_translation[1] + axis_z[2] * local_translation[2],
    )
    return pos_after_node_transforms(row, position, inherited_options)


def part_information_name(part_body: str) -> str | None:
    info = transform_helpers.extract_keyed_object(part_body, "information")
    if not info:
        return None
    match = re.search(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"', info)
    return match.group(1) if match else None


def collect_prop_only_objects(
    jbeam_texts: dict[str, str],
    node_positions: dict[str, tuple[float, float, float]],
    existing_objects: dict[str, DaeObject],
    part_body_index: dict[str, tuple[str, str]],
) -> tuple[dict[str, DaeObject], dict[str, dict[str, object]]]:
    positions: dict[str, list[tuple[float, float, float]]] = {}
    labels: dict[str, str] = {}
    for part_body, _filename in part_body_index.values():
        props = transform_helpers.extract_named_array(part_body, "props")
        if not props:
            continue
        part_name = part_information_name(part_body)
        for row in iter_top_level_rows(props):
            strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
            if len(strings) < 2:
                continue
            func, mesh = strings[:2]
            if func == "func" or mesh == "mesh" or mesh in existing_objects:
                continue
            position = prop_row_position(row, node_positions)
            if position is not None:
                positions.setdefault(mesh, []).append(position)
                if part_name:
                    labels.setdefault(mesh, part_name)

    objects: dict[str, DaeObject] = {}
    previews: dict[str, dict[str, object]] = {}
    for mesh, mesh_positions in positions.items():
        if not mesh_positions:
            continue
        x = sum(pos[0] for pos in mesh_positions) / len(mesh_positions)
        y = sum(pos[1] for pos in mesh_positions) / len(mesh_positions)
        z = sum(pos[2] for pos in mesh_positions) / len(mesh_positions)
        objects[mesh] = DaeObject(
            id=mesh,
            name=labels.get(mesh, mesh),
            dae_path="",
            x=x,
            y=y,
            z=z,
            geometry_ids=(),
        )
        pad = 0.035
        previews[mesh] = {
            "bounds": ((x - pad, y - pad, z - pad), (x + pad, y + pad, z + pad)),
            "center": (x, y, z),
            "sample_points": [(x, y, z)],
            "geometry_ids": [],
        }
    return objects, previews


def flexbody_row_mesh(row: str) -> str | None:
    match = re.match(r'\s*\[\s*"((?:[^"\\]|\\.)*)"', row)
    if match is None:
        return None
    mesh = match.group(1)
    return None if mesh == "mesh" else mesh


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a[row][idx] * b[idx][col] for idx in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def translation_matrix(values: tuple[float, float, float]) -> list[list[float]]:
    out = identity_matrix()
    out[0][3], out[1][3], out[2][3] = values
    return out


def scale_matrix(values: tuple[float, float, float]) -> list[list[float]]:
    out = identity_matrix()
    out[0][0], out[1][1], out[2][2] = values
    return out


def mirror_x_matrix4() -> list[list[float]]:
    out = identity_matrix()
    out[0][0] = -1.0
    return out


def rotation_x_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z_matrix(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


PROP_VECTOR_RE = re.compile(
    rf'\{{\s*"x"\s*:\s*(?P<x>{NUMBER_RE})\s*,?\s*'
    rf'"y"\s*:\s*(?P<y>{NUMBER_RE})\s*,\s*"z"\s*:\s*(?P<z>{NUMBER_RE})\s*\}}'
)


def prop_row_vector_objects(row: str) -> list[tuple[float, float, float]]:
    return [
        (float(match.group("x")), float(match.group("y")), float(match.group("z")))
        for match in PROP_VECTOR_RE.finditer(row)
    ]


def matrix3_from_matrix4(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for col in range(3)] for row in range(3)]


def multiply_matrix3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(a[row][idx] * b[idx][col] for idx in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def euler_matrix3(degrees: tuple[float, float, float]) -> list[list[float]]:
    matrix = identity_matrix()
    for next_matrix in (
        rotation_z_matrix(degrees[2]),
        rotation_y_matrix(degrees[1]),
        rotation_x_matrix(degrees[0]),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix3_from_matrix4(matrix)


def prop_base_rotation_matrix3(degrees: tuple[float, float, float]) -> list[list[float]]:
    # Game prop baseRotation euler order is -X -Z +Y intrinsic
    # (lua/common/jbeam/sections/meshs.lua), i.e. B = Rx(-x)*Rz(-z)*Ry(+y).
    matrix = identity_matrix()
    for next_matrix in (
        rotation_x_matrix(-degrees[0]),
        rotation_z_matrix(-degrees[2]),
        rotation_y_matrix(degrees[1]),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix3_from_matrix4(matrix)


def euler_yzx_from_matrix3(matrix: list[list[float]]) -> tuple[float, float, float]:
    # Decompose M = Ry(y)*Rz(z)*Rx(x) — the game's baseRotationGlobal order
    # ("YZX intrinsic" per lua/common/jbeam/sections/meshs.lua).
    sz = max(-1.0, min(1.0, matrix[1][0]))
    z = math.asin(sz)
    cz = math.cos(z)
    if abs(cz) > 1e-8:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        y = math.atan2(-matrix[2][0], matrix[0][0])
    else:
        x = math.atan2(matrix[2][1], matrix[2][2])
        y = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def matrix3_from_axes(
    axis_x: tuple[float, float, float],
    axis_y: tuple[float, float, float],
    axis_z: tuple[float, float, float],
) -> list[list[float]]:
    return [
        [axis_x[0], axis_y[0], axis_z[0]],
        [axis_x[1], axis_y[1], axis_z[1]],
        [axis_x[2], axis_y[2], axis_z[2]],
    ]


def mirror_rotation_matrix_x(matrix: list[list[float]]) -> list[list[float]]:
    out = [row[:] for row in matrix]
    for col in range(3):
        out[0][col] *= -1
    for row in range(3):
        out[row][0] *= -1
    return out


def euler_from_matrix3(matrix: list[list[float]]) -> tuple[float, float, float]:
    sy = max(-1.0, min(1.0, -matrix[2][0]))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-8:
        x = math.atan2(matrix[2][1], matrix[2][2])
        z = math.atan2(matrix[1][0], matrix[0][0])
    else:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


NODE_TRANSFORM_KEY_RE = re.compile(r'"(?P<key>node(?:Rotate|Offset|Move)(?P<index>\d*)?)"\s*:')


def sign_number(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def approximate_expression_number(value: str) -> float | None:
    text = value.strip()
    if text.startswith("$="):
        text = text[2:]
    try:
        return float(text)
    except ValueError:
        pass
    constants = [float(match.group(0)) for match in re.finditer(NUMBER_RE, text)]
    if not constants:
        return None
    return sum(constants)


def object_number_property(object_text: str, key: str) -> float | None:
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*(?P<value>{NUMBER_RE}|"(?:[^"\\]|\\.)*")',
        object_text,
    )
    if not match:
        return None
    raw = match.group("value")
    if raw.startswith('"'):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = raw.strip('"')
        return approximate_expression_number(str(decoded))
    return float(raw)


def node_transform_kind(key: str) -> tuple[str, int] | None:
    for prefix in ("nodeRotate", "nodeOffset", "nodeMove"):
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix and not suffix.isdigit():
                return None
            return prefix, int(suffix or 0)
    return None


def node_transform_ops(texts: Iterable[str]) -> dict[tuple[str, int], dict[str, float]]:
    ops: dict[tuple[str, int], dict[str, float]] = {}
    for text in texts:
        for match in NODE_TRANSFORM_KEY_RE.finditer(text):
            parsed_key = node_transform_kind(match.group("key"))
            if parsed_key is None:
                continue
            idx = match.end()
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text) or text[idx] != "{":
                ops.pop(parsed_key, None)
                continue
            try:
                end = transform_helpers.find_matching(text, idx, "{", "}")
            except ValueError:
                ops.pop(parsed_key, None)
                continue
            object_text = text[idx:end]
            x = object_number_property(object_text, "x")
            y = object_number_property(object_text, "y")
            z = object_number_property(object_text, "z")
            if x is None and y is None and z is None:
                ops.pop(parsed_key, None)
                continue
            op = {"x": x or 0.0, "y": y or 0.0, "z": z or 0.0}
            for pivot_key in ("px", "py", "pz"):
                pivot_value = object_number_property(object_text, pivot_key)
                if pivot_value is not None:
                    op[pivot_key] = pivot_value
            ops[parsed_key] = op
    return ops


def node_op_indices(ops: dict[tuple[str, int], dict[str, float]]) -> range:
    if not ops:
        return range(0)
    indices = [idx for _kind, idx in ops]
    return range(min(indices), max(indices) + 1)


def has_node_rotations(ops: dict[tuple[str, int], dict[str, float]]) -> bool:
    return any(kind == "nodeRotate" for kind, _idx in ops)


def node_translation_offset(
    ops: dict[tuple[str, int], dict[str, float]],
    pos_x_sign: int,
) -> tuple[float, float, float]:
    x = y = z = 0.0
    for idx in node_op_indices(ops):
        offset = ops.get(("nodeOffset", idx))
        if offset is not None:
            x += pos_x_sign * offset["x"]
            y += offset["y"]
            z += offset["z"]
        move = ops.get(("nodeMove", idx))
        if move is not None:
            x += move["x"]
            y += move["y"]
            z += move["z"]
    return x, y, z


def matrix4_from_matrix3(rotation: list[list[float]]) -> list[list[float]]:
    matrix = identity_matrix()
    for row in range(3):
        for col in range(3):
            matrix[row][col] = rotation[row][col]
    return matrix


def inverse_affine_matrix(matrix: list[list[float]]) -> list[list[float]]:
    inverse3 = transform_helpers.inverse_3x3(matrix)
    tx, ty, tz = matrix[0][3], matrix[1][3], matrix[2][3]
    out = identity_matrix()
    for row in range(3):
        for col in range(3):
            out[row][col] = inverse3[row][col]
        out[row][3] = -(inverse3[row][0] * tx + inverse3[row][1] * ty + inverse3[row][2] * tz)
    return out


def node_transform_matrix(
    ops: dict[tuple[str, int], dict[str, float]],
    pos_x: float,
) -> list[list[float]]:
    matrix = identity_matrix()
    pos_x_sign = sign_number(pos_x)
    for idx in node_op_indices(ops):
        rotation = ops.get(("nodeRotate", idx))
        if rotation is not None:
            rotation_matrix = matrix4_from_matrix3(
                euler_matrix3((-rotation["x"], -rotation["y"], -rotation["z"]))
            )
            if any(key in rotation for key in ("px", "py", "pz")):
                pivot = (
                    rotation.get("px", 0.0),
                    rotation.get("py", 0.0),
                    rotation.get("pz", 0.0),
                )
                rotation_matrix = multiply_matrix(
                    multiply_matrix(translation_matrix(pivot), rotation_matrix),
                    translation_matrix((-pivot[0], -pivot[1], -pivot[2])),
                )
            matrix = multiply_matrix(matrix, rotation_matrix)

        offset = ops.get(("nodeOffset", idx))
        if offset is not None:
            matrix = multiply_matrix(
                matrix,
                translation_matrix((pos_x_sign * offset["x"], offset["y"], offset["z"])),
            )

        move = ops.get(("nodeMove", idx))
        if move is not None:
            matrix = multiply_matrix(matrix, translation_matrix((move["x"], move["y"], move["z"])))
    return matrix


def node_transform_source_texts(
    row: str,
    inherited_options: Iterable[str] = (),
) -> list[str]:
    return [text for text in [*inherited_options, row] if text]


def pos_after_node_transforms(
    row: str,
    position: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
) -> tuple[float, float, float]:
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options))
    if not ops:
        return position
    if not has_node_rotations(ops):
        dx, dy, dz = node_translation_offset(ops, sign_number(position[0]))
        return position[0] + dx, position[1] + dy, position[2] + dz
    return transform_helpers.transform_point(node_transform_matrix(ops, position[0]), position)


def pos_before_node_transforms(
    row: str,
    position: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
) -> tuple[float, float, float]:
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options))
    if not ops:
        return position

    if not has_node_rotations(ops):
        fallback = position
        for pos_x_sign in (1, 0, -1):
            dx, dy, dz = node_translation_offset(ops, pos_x_sign)
            candidate = (position[0] - dx, position[1] - dy, position[2] - dz)
            fallback = candidate
            if sign_number(candidate[0]) == pos_x_sign:
                return candidate
        return fallback

    fallback = position
    for pos_x_sign in (1, 0, -1):
        matrix = node_transform_matrix(ops, float(pos_x_sign))
        candidate = transform_helpers.transform_point(inverse_affine_matrix(matrix), position)
        fallback = candidate
        if sign_number(candidate[0]) == pos_x_sign:
            return candidate
    return fallback


def pos_rot_before_node_transforms(
    row: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options))
    if not ops:
        return position, rotation
    if not has_node_rotations(ops):
        return pos_before_node_transforms(row, position, inherited_options), rotation

    fallback_pos = position
    fallback_rot = rotation
    for pos_x_sign in (1, 0, -1):
        inverse = inverse_affine_matrix(node_transform_matrix(ops, float(pos_x_sign)))
        candidate_pos = transform_helpers.transform_point(inverse, position)
        inverse_rotation = matrix3_from_matrix4(inverse)
        neg_rotation = euler_matrix3((-rotation[0], -rotation[1], -rotation[2]))
        candidate_neg_rotation = multiply_matrix3(neg_rotation, inverse_rotation)
        euler = euler_from_matrix3(candidate_neg_rotation)
        candidate_rot = (-euler[0], -euler[1], -euler[2])
        fallback_pos, fallback_rot = candidate_pos, candidate_rot
        if sign_number(candidate_pos[0]) == pos_x_sign:
            return candidate_pos, candidate_rot
    return fallback_pos, fallback_rot


def prop_row_global_rotation_matrix(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> list[list[float]] | None:
    global_rotation = vector_from_row(row, "baseRotationGlobal")
    if global_rotation is not None:
        # baseRotationGlobal euler order is YZX intrinsic: Ry(y)*Rz(z)*Rx(x)
        matrix = identity_matrix()
        for next_matrix in (
            rotation_y_matrix(global_rotation[1]),
            rotation_z_matrix(global_rotation[2]),
            rotation_x_matrix(global_rotation[0]),
        ):
            matrix = multiply_matrix(matrix, next_matrix)
        return matrix3_from_matrix4(matrix)

    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    vectors = prop_row_vector_objects(row)
    if not vectors:
        return None

    ref_pos = node_positions.get(strings[2])
    x_pos = node_positions.get(strings[3])
    y_pos = node_positions.get(strings[4])
    if ref_pos is None or x_pos is None or y_pos is None:
        return None

    axis_x = normalize_vector(vector_subtract(x_pos, ref_pos))
    axis_y_seed = normalize_vector(vector_subtract(y_pos, ref_pos))
    axis_z = normalize_vector(cross_product(axis_y_seed, axis_x))
    if axis_x == (0.0, 0.0, 0.0) or axis_z == (0.0, 0.0, 0.0):
        return None
    axis_y = normalize_vector(cross_product(axis_x, axis_z))
    if axis_y == (0.0, 0.0, 0.0):
        return None

    frame = matrix3_from_axes(axis_x, axis_y, axis_z)
    return multiply_matrix3(frame, prop_base_rotation_matrix3(vectors[0]))


def mirrored_prop_global_rotation(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    rotation = prop_row_global_rotation_matrix(row, node_positions)
    if rotation is None:
        return None
    return euler_yzx_from_matrix3(mirror_rotation_matrix_x(rotation))


def prop_frame_axes(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...] | None:
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    ref_pos = node_positions.get(strings[2])
    x_pos = node_positions.get(strings[3])
    y_pos = node_positions.get(strings[4])
    if ref_pos is None or x_pos is None or y_pos is None:
        return None
    axis_x = normalize_vector(vector_subtract(x_pos, ref_pos))
    seed = normalize_vector(vector_subtract(y_pos, ref_pos))
    axis_z = normalize_vector(cross_product(seed, axis_x))
    if axis_x == (0.0, 0.0, 0.0) or axis_z == (0.0, 0.0, 0.0):
        return None
    axis_y = normalize_vector(cross_product(axis_x, axis_z))
    if axis_y == (0.0, 0.0, 0.0):
        return None
    return ref_pos, axis_x, axis_y, axis_z


def prop_row_pivot_position(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    pivot: tuple[float, float, float] | None,
    inherited_options: Iterable[str] = (),
) -> tuple[float, float, float] | None:
    """World rest position of the prop mesh's DAE pivot.

    Engine rule, verified against an in-game dump of getBaseTranslationGlobal()
    for every prop of the stock sunburst2 rally config:
      1. row has baseTranslationGlobal -> that value verbatim;
      2. row has baseTranslation      -> refNode + normalizedFrame * baseTranslation
                                         (the mesh pivot contributes nothing);
      3. neither                      -> the mesh's authored DAE pivot (identity rest).
    Hand conversion must mirror/translate this position.
    """
    global_translation = vector_from_row(row, "baseTranslationGlobal")
    if global_translation is not None:
        return pos_after_node_transforms(row, global_translation, inherited_options)

    base_translation = vector_from_row(row, "baseTranslation")
    if base_translation is None:
        if pivot is None:
            return None
        return pos_after_node_transforms(row, pivot, inherited_options)

    frame = prop_frame_axes(row, node_positions)
    if frame is None:
        return None
    ref_pos, axis_x, axis_y, axis_z = frame
    return (
        ref_pos[0] + axis_x[0] * base_translation[0] + axis_y[0] * base_translation[1] + axis_z[0] * base_translation[2],
        ref_pos[1] + axis_x[1] * base_translation[0] + axis_y[1] * base_translation[1] + axis_z[1] * base_translation[2],
        ref_pos[2] + axis_x[2] * base_translation[0] + axis_y[2] * base_translation[1] + axis_z[2] * base_translation[2],
    )


def matrix4_with_rotation_translation(
    rotation: list[list[float]] | None,
    position: tuple[float, float, float],
) -> list[list[float]]:
    matrix = identity_matrix()
    if rotation is not None:
        for row in range(3):
            for col in range(3):
                matrix[row][col] = rotation[row][col]
    matrix[0][3], matrix[1][3], matrix[2][3] = position
    return matrix


def prop_row_source_matrix(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    inherited_options: Iterable[str] = (),
    rotation_override: list[list[float]] | None = None,
) -> list[list[float]] | None:
    position = prop_row_position(row, node_positions, inherited_options)
    if position is None:
        return None
    rotation = rotation_override
    if rotation is None:
        rotation = prop_row_global_rotation_matrix(row, node_positions)
    return matrix4_with_rotation_translation(rotation, position)


def flexbody_row_matrix(row: str) -> list[list[float]]:
    pos = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
    rot = vector_from_row(row, "rot") or (0.0, 0.0, 0.0)
    scale = vector_from_row(row, "scale") or (1.0, 1.0, 1.0)
    matrix = translation_matrix(pos)
    # Game flexbody rot euler is "+Z +X +Y intrinsic" (meshs.lua) with the
    # sequence listed innermost-first: Z is applied to the mesh first, then X,
    # then Y, i.e. v = pos + Ry*Rx*Rz*(scale*v). Ground truth: the sunburst2
    # boot spare (rot x:75 z:90) must lie flat under its authored-in-place
    # strap (axis (0,.26,.97)), not stand vertically (axis y); the offroad
    # swing-mount spare (z:90 only) pins the signs as positive. Single-axis
    # rows are order-insensitive, which is why this stayed hidden.
    for next_matrix in (
        rotation_y_matrix(rot[1]),
        rotation_x_matrix(rot[0]),
        rotation_z_matrix(rot[2]),
        scale_matrix(scale),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix


def flexbody_row_source_matrix(
    row: str,
    inherited_options: Iterable[str] = (),
) -> list[list[float]]:
    matrix = flexbody_row_matrix(row)
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options))
    if not ops:
        return matrix
    if not has_node_rotations(ops):
        pos = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
        dx, dy, dz = node_translation_offset(ops, sign_number(pos[0]))
        return multiply_matrix(translation_matrix((dx, dy, dz)), matrix)
    pos = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
    return multiply_matrix(node_transform_matrix(ops, pos[0]), matrix)


def is_shared_dae_object(context: VehicleContext, object_id: str) -> bool:
    obj = context.objects.get(object_id)
    if obj is None or not obj.dae_path:
        return False
    dae_path = obj.dae_path.replace("\\", "/").lower()
    return dae_path.startswith("vehicles/common/") or (
        obj.dae_source_zip is not None
        and obj.dae_source_zip.resolve(strict=False) != context.source_zip.resolve(strict=False)
    )


def baked_mesh_output_name(
    source_mesh: str,
    target_hand: str,
    config_name: str,
    source_part_id: str,
    index: int,
) -> str:
    return safe_id(
        f"{source_mesh}{suffix_for_hand(target_hand)}__{config_name}__{source_part_id}__{index:04d}"
    )


def add_baked_shared_mesh(
    bake_context: SharedBakeContext,
    mesh: str,
    placement_matrix: list[list[float]],
    bake_transform_into_dae: bool,
    is_prop: bool = False,
) -> str | None:
    source_mesh = bake_context.structural_sources.get(mesh, mesh)
    mode = bake_context.object_modes.get(mesh)
    if mode not in {MODE_TRANSLATE, MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
        return None
    # Mirrored props always need a per-row baked copy: their node-triad frame is
    # left-handed, so the mirrored orientation is a reflection that
    # baseRotationGlobal (an euler rotation) cannot express. Vehicle-local
    # meshes on other paths keep using the shared "_to_rhd" copies.
    if not is_shared_dae_object(bake_context.context, source_mesh) and not (
        is_prop and mode == MODE_MIRROR
    ):
        return None
    source_obj = bake_context.context.objects.get(source_mesh)
    if source_obj is None or not source_obj.dae_path:
        return None
    output_mesh = baked_mesh_output_name(
        mesh,
        bake_context.target_hand,
        bake_context.config_name,
        bake_context.source_part_id,
        len(bake_context.baked_specs),
    )
    bake_context.baked_specs.append(
        BakedMeshSpec(
            configured_mesh=mesh,
            source_mesh=source_mesh,
            output_mesh=output_mesh,
            target_hand=bake_context.target_hand,
            mode=mode,
            placement_matrix=placement_matrix,
            bake_transform_into_dae=bake_transform_into_dae,
            is_prop=is_prop,
        )
    )
    return output_mesh


def rotation_transpose_matrix4(matrix: list[list[float]]) -> list[list[float]]:
    out = identity_matrix()
    for row in range(3):
        for col in range(3):
            out[row][col] = matrix[col][row]
    return out


def baked_dae_matrix(
    source_node_matrix: list[list[float]],
    spec: BakedMeshSpec,
    translate_magnitudes: dict[str, float],
) -> list[list[float]]:
    mirror = mirror_x_matrix4()

    if not spec.bake_transform_into_dae:
        if spec.mode == MODE_TRANSLATE:
            # Position handled in jbeam (flexbody pos / prop baseTranslationGlobal);
            # the DAE copy stays identical to the source.
            return source_node_matrix
        if spec.is_prop:
            # Prop mirror: jbeam keeps the node-frame anchoring (baseTranslationGlobal moves
            # the anchor to the mirrored position) so the reflection must be baked into the
            # mesh in prop-model space. With world = T(A)*R*(M*g) and the mirrored anchor
            # T(S*A), we need M' = (R^T*S*R)*M*S paired with locally mirrored geometry S*g:
            # T(S*A)*R*M'*(S*g) = S*(T(A)*R*M*g).
            rotation = rotation_transpose_matrix4(spec.placement_matrix)
            reflection = multiply_matrix(
                multiply_matrix(rotation, mirror),
                matrix4_with_rotation_translation(matrix3_from_matrix4(spec.placement_matrix), (0.0, 0.0, 0.0)),
            )
            return multiply_matrix(multiply_matrix(reflection, source_node_matrix), mirror)
        # Flexbody mirror with the pos/rot mirrored in the jbeam row (P' = S*P*S):
        # the DAE copy must supply the world-mirrored mesh in DAE space, i.e.
        # node matrix S*M*S with locally mirrored geometry S*g, so that
        # P'*(S*M*S)*(S*g) = S*(P*M*g).
        return multiply_matrix(multiply_matrix(mirror, source_node_matrix), mirror)

    placement_inverse = inverse_affine_matrix(spec.placement_matrix)
    if spec.mode == MODE_TRANSLATE:
        delta = signed_delta_for_target(
            spec.target_hand,
            translate_magnitudes.get(spec.configured_mesh, 0.0),
        )
        return multiply_matrix(
            multiply_matrix(
                multiply_matrix(placement_inverse, translation_matrix((delta, 0.0, 0.0))),
                spec.placement_matrix,
            ),
            source_node_matrix,
        )
    return multiply_matrix(
        multiply_matrix(
            multiply_matrix(
                multiply_matrix(placement_inverse, mirror),
                spec.placement_matrix,
            ),
            source_node_matrix,
        ),
        mirror,
    )


def collect_prop_mesh_positions(
    node_positions: dict[str, tuple[float, float, float]],
    part_body_index: dict[str, tuple[str, str]],
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    positions: dict[str, list[tuple[float, float, float]]] = {}
    for part_body, _filename in part_body_index.values():
        props = transform_helpers.extract_named_array(part_body, "props")
        if not props:
            continue
        for row in iter_top_level_rows(props):
            strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
            if len(strings) < 2:
                continue
            func, mesh = strings[:2]
            if func == "func" or mesh == "mesh":
                continue
            pivot = (mesh_pivots or {}).get(mesh)
            position = prop_row_pivot_position(row, node_positions, pivot)
            if position is not None:
                positions.setdefault(mesh, []).append(position)
    return positions


def collect_flexbody_mesh_placements(
    objects: dict[str, DaeObject],
    part_body_index: dict[str, tuple[str, str]],
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[dict[str, list[MeshPlacement]], set[str]]:
    """Every flexbody placement in the whole part index, ignoring trims.

    Placements from parts that can never coexist are all present here, so this
    must NOT be used to decide where a mesh sits -- see
    resolved_mesh_positions_for_config for that. It stays because the returned
    positioned-mesh set (meshes any jbeam row gives an explicit pos) is a
    whole-vehicle property the build relies on.

    Positions are measured from the authored pivot: passing objects whose
    coordinates have already been resolved would compound the placement onto
    an already-placed mesh."""
    placements: dict[str, list[MeshPlacement]] = {}
    positioned_meshes: set[str] = set()
    for part_body, _filename in part_body_index.values():
        flexbodies = transform_helpers.extract_named_array(part_body, "flexbodies")
        if not flexbodies:
            continue
        for row in iter_top_level_rows(flexbodies):
            mesh = flexbody_row_mesh(row)
            if not mesh or mesh not in objects:
                continue
            if (
                vector_from_row(row, "pos") is None
                and vector_from_row(row, "rot") is None
                and vector_from_row(row, "scale") is None
            ):
                continue
            obj = objects[mesh]
            pivot = (mesh_pivots or {}).get(mesh, (obj.x, obj.y, obj.z))
            matrix = flexbody_row_matrix(row)
            position = transform_helpers.transform_point(matrix, pivot)
            placements.setdefault(mesh, []).append(MeshPlacement(position=position, matrix=matrix))
            if vector_from_row(row, "pos") is not None:
                positioned_meshes.add(mesh)
    return placements, positioned_meshes


def is_far_placement(position: tuple[float, float, float]) -> bool:
    """Whether a row parks the mesh so far out it is really being hidden.

    Same threshold the preview payload uses to drop instances, so both agree
    on what counts as present in a configuration."""
    return math.hypot(position[0], position[1], position[2]) > PREVIEW_FAR_LIMIT


def average_position(positions: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return (
        sum(position[0] for position in positions) / len(positions),
        sum(position[1] for position in positions) / len(positions),
        sum(position[2] for position in positions) / len(positions),
    )


def moved_object(obj: DaeObject, position: tuple[float, float, float]) -> DaeObject:
    return DaeObject(
        id=obj.id,
        name=obj.name,
        dae_path=obj.dae_path,
        x=position[0],
        y=position[1],
        z=position[2],
        geometry_ids=obj.geometry_ids,
        dae_source_zip=obj.dae_source_zip,
    )


def bounds_corners(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    min_point, max_point = bounds
    return [
        (x, y, z)
        for x in (min_point[0], max_point[0])
        for y in (min_point[1], max_point[1])
        for z in (min_point[2], max_point[2])
    ]


def transform_preview_points(
    preview: dict[str, object],
    matrices: list[list[list[float]]],
) -> dict[str, object]:
    raw_points = list(preview.get("sample_points", []))
    if not raw_points and "center" in preview:
        raw_points = [preview["center"]]
    if "bounds" in preview:
        raw_points.extend(bounds_corners(preview["bounds"]))
    points = [
        transform_helpers.transform_point(matrix, point)
        for matrix in matrices
        for point in raw_points
    ]
    if not points:
        return preview
    bounds = transform_helpers.bounds_from_points(points)
    min_point, max_point = bounds
    return {
        **preview,
        "bounds": bounds,
        "center": (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        ),
        "sample_points": transform_helpers.sample_points(points, 350),
    }


def translate_preview_points(
    preview: dict[str, object],
    delta: tuple[float, float, float],
) -> dict[str, object]:
    matrix = translation_matrix(delta)
    return transform_preview_points(preview, [matrix])


def apply_resolved_mesh_positions(
    objects: dict[str, DaeObject],
    preview_by_id: dict[str, dict[str, object]],
    resolved: dict[str, ResolvedMeshPosition],
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
    jbeam_positioned_flexbodies: set[str] | None = None,
) -> None:
    """Move each mesh to its representative position (see
    representative_mesh_positions). Flexbody previews are transformed by the
    representative trim's row matrices so rotation/scale survive; prop previews
    are translated by the delta, matching how the engine places each kind.

    A flexbody mesh whose row authors no pos of its own additionally sheds the
    DAE node translation, because that is what the renderers do with it (see
    flexbody_row_needs_node_translation / mesh_preview.build_scene). Without
    this the preview boxes and the drawn geometry disagree by the node
    translation on exactly those meshes -- e.g. etk800's manual shifter."""
    positioned = jbeam_positioned_flexbodies or set()
    for mesh, entry in resolved.items():
        obj = objects.get(mesh)
        if obj is None:
            continue
        objects[mesh] = moved_object(obj, entry.position)

        preview = preview_by_id.get(mesh)
        if preview is None:
            continue
        if entry.matrices:
            preview = transform_preview_points(preview, list(entry.matrices))
            if mesh not in positioned:
                pivot = (mesh_pivots or {}).get(mesh)
                if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
                    preview = translate_preview_points(
                        preview, (-pivot[0], -pivot[1], -pivot[2])
                    )
            preview_by_id[mesh] = preview
        else:
            preview_by_id[mesh] = translate_preview_points(
                preview,
                (
                    entry.position[0] - obj.x,
                    entry.position[1] - obj.y,
                    entry.position[2] - obj.z,
                ),
            )


def strip_json_comments(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    line_comment = False
    block_comment = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if line_comment:
            if ch in "\r\n":
                line_comment = False
                out.append(ch)
            idx += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                idx += 2
            else:
                out.append("\n" if ch in "\r\n" else " ")
                idx += 1
            continue
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            idx += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            idx += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            idx += 2
            continue
        out.append(ch)
        if ch == '"':
            in_string = True
        idx += 1
    return "".join(out)


def json_line_needs_comma(current: str, next_line: str) -> bool:
    current = current.strip()
    next_line = next_line.strip()
    if not current or not next_line:
        return False
    if current.endswith(",") or current.endswith("{") or current.endswith("["):
        return False
    if next_line[0] in "]}":
        return False
    if not re.search(r'(?:"|-?\d+(?:\.\d*)?|\.\d+|true|false|null|\]|\})$', current):
        return False
    return bool(re.match(r'(?:"|\{|\[|-?\d+(?:\.\d*)?|\.\d+|true|false|null)', next_line))


def add_missing_json_commas(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for index, line in enumerate(lines):
        next_line = ""
        for candidate in lines[index + 1 :]:
            if candidate.strip():
                next_line = candidate
                break
        if next_line and json_line_needs_comma(line, next_line):
            line_ending = ""
            content = line
            if content.endswith("\r\n"):
                content, line_ending = content[:-2], "\r\n"
            elif content.endswith("\n"):
                content, line_ending = content[:-1], "\n"
            line = content + "," + line_ending
        out.append(line)
    return "".join(out)


def parse_beamng_json(text: str, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_error:
        cleaned = strip_json_comments(text.lstrip("\ufeff"))
        cleaned = re.sub(r",\s*,+", ",", cleaned)
        cleaned = add_missing_json_commas(cleaned)
        cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as second_error:
            try:
                parsed, end = json.JSONDecoder().raw_decode(cleaned)
                remainder = cleaned[end:].strip()
                if remainder.strip("}").strip():
                    raise second_error
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Could not parse BeamNG config {label}: {second_error}. "
                    f"Initial strict JSON error was: {first_error}"
                ) from second_error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"BeamNG config {label} did not parse to an object")
    return parsed


def load_pc(source_zip: Path, pc_path: str) -> dict[str, object]:
    with zipfile.ZipFile(source_zip) as zf:
        return parse_beamng_json(
            zf.read(pc_path).decode("utf-8", errors="replace"),
            label=pc_path,
        )


def load_info(source_zip: Path, info_path: str) -> dict[str, object]:
    with zipfile.ZipFile(source_zip) as zf:
        return parse_beamng_json(
            zf.read(info_path).decode("utf-8", errors="replace"),
            label=info_path,
        )


def info_path_for_config(source_zip: Path, vehicle_id: str, config_name: str) -> str | None:
    candidates = [
        f"{vehicle_prefix(vehicle_id)}/info_{config_name}.json",
        f"{vehicle_prefix(vehicle_id)}/{config_name}.json",
    ]
    with zipfile.ZipFile(source_zip) as zf:
        names = {name.replace("\\", "/") for name in zf.namelist()}
    return next((candidate for candidate in candidates if candidate in names), None)


def display_name_for(source_zip: Path, info_path: str | None, config_name: str) -> str:
    if info_path is None:
        return config_name
    try:
        info = load_info(source_zip, info_path)
    except Exception:
        return config_name
    for key in ("Configuration", "Name", "name", "configuration"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return config_name


def _zip_json_by_name(source_zip: Path, wanted_name: str) -> dict[str, object]:
    wanted = wanted_name.replace("\\", "/").lower()
    with zipfile.ZipFile(source_zip) as zf:
        actual = next(
            (
                name
                for name in zf.namelist()
                if name.replace("\\", "/").lower() == wanted
            ),
            None,
        )
        if not actual:
            return {}
        try:
            parsed = parse_beamng_json(
                zf.read(actual).decode("utf-8", errors="replace"),
                label=actual,
            )
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def original_source_name(context: VehicleContext) -> str:
    source_posix = context.source_zip.as_posix().lower()
    if "/content/vehicles/" in source_posix:
        return "BeamNG - Official"

    mod_info = _zip_json_by_name(context.source_zip, "mod_info/info.json")
    for key in ("source", "Source", "name", "Name", "title", "Title"):
        value = mod_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    authors = mod_info.get("authors") or mod_info.get("Authors") or mod_info.get("author")
    if isinstance(authors, str) and authors.strip():
        return authors.strip()
    if isinstance(authors, list):
        joined = ", ".join(str(author).strip() for author in authors if str(author).strip())
        if joined:
            return joined

    return "Custom"


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
    if lowered.endswith(f" - {suffix.lower()}") or lowered.endswith(suffix.lower()):
        return description
    return f"{description} - {suffix}"


def source_preview_path(source_zip: Path, vehicle_path: str, config_name: str) -> str | None:
    prefix = f"{vehicle_path.rstrip('/')}/{config_name}".lower()
    preview_exts = (".jpg", ".jpeg", ".png", ".webp")
    with zipfile.ZipFile(source_zip) as zf:
        for name in zf.namelist():
            clean = name.replace("\\", "/")
            if clean.lower().startswith(prefix) and Path(clean).suffix.lower() in preview_exts:
                if clean.lower() == f"{prefix}{Path(clean).suffix.lower()}":
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
    candidates.append(APP_DIR / "xp_sticker.png")
    candidates.append(SOURCE_ROOT_DIR / "xp_sticker.png")
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


def hand_from_text(text: str) -> str:
    lowered = text.lower()
    rhd_tokens = ("rhd", "right hand drive", "right-hand drive", "right hand-drive", "jdm", "uk")
    lhd_tokens = ("lhd", "left hand drive", "left-hand drive", "left hand-drive")
    if any(token in lowered for token in rhd_tokens):
        return HAND_RHD
    if any(token in lowered for token in lhd_tokens):
        return HAND_LHD
    return HAND_UNKNOWN


# Bump whenever context-building logic changes in a way that affects cached
# VehicleContext content (parsing, pivots, common indexing, ...). Structural
# dataclass changes are caught automatically via the field-name fingerprint.
CONTEXT_CACHE_VERSION = 5  # 5: bare flexbody rows resolve position without the node's own translation


def context_cache_path(source_zip: Path, vehicle_id: str) -> Path:
    return project_dir_for(source_zip, vehicle_id) / "context.cache"


def context_cache_fingerprint(source_zip: Path) -> tuple:
    parts: list[tuple] = [
        ("cacheVersion", CONTEXT_CACHE_VERSION),
        ("contextFields", tuple(f.name for f in dataclass_fields(VehicleContext))),
        ("objectFields", tuple(f.name for f in dataclass_fields(DaeObject))),
    ]
    for candidate in common_zip_candidates(Path(source_zip)):
        try:
            stat = Path(candidate).stat()
            parts.append((str(candidate), stat.st_size, stat.st_mtime_ns))
        except OSError:
            parts.append((str(candidate), None, None))
    return tuple(parts)


def load_cached_vehicle_context(source_zip: Path, vehicle_id: str) -> VehicleContext | None:
    path = context_cache_path(source_zip, vehicle_id)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != context_cache_fingerprint(source_zip):
        return None
    context = payload.get("context")
    if not isinstance(context, VehicleContext):
        return None
    context.project_dir = project_dir_for(source_zip, vehicle_id)
    context.source_zip = Path(source_zip)
    context.vehicle_id = vehicle_id
    context.selected_parts_cache = {}
    context.mesh_roles_cache = {}
    context.selected_node_positions_cache = {}
    context.part_array_cache = {}
    context.variant_hands_cache = {}
    context.resolved_positions_cache = {}
    return context


def save_vehicle_context_cache(context: VehicleContext) -> Path | None:
    path = context_cache_path(context.source_zip, context.vehicle_id)
    try:
        payload = {
            "fingerprint": context_cache_fingerprint(context.source_zip),
            "context": dataclass_replace(
                context,
                selected_parts_cache={},
                mesh_roles_cache={},
                selected_node_positions_cache={},
                part_array_cache={},
                variant_hands_cache={},
                resolved_positions_cache={},
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        return path
    except Exception:
        return None


def context_fingerprint_hash(source_zip: Path) -> str:
    payload = json.dumps(context_cache_fingerprint(Path(source_zip)), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def parts_cache_path(context: VehicleContext) -> Path:
    return context.project_dir / "parts_cache.json"


def selection_cache_key(selected: Iterable[str]) -> str:
    return "|".join(sorted(str(name) for name in selected))


def load_cached_part_ids(context: VehicleContext, selected: Iterable[str]) -> list[str] | None:
    """Resolved used-part ids for a variant selection, persisted across
    sessions. Valid only while the source/common zips are unchanged (same
    fingerprint as the context cache)."""
    path = parts_cache_path(context)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("fingerprint") != context_fingerprint_hash(context.source_zip):
        return None
    selections = data.get("selections")
    if not isinstance(selections, dict):
        return None
    ids = selections.get(selection_cache_key(selected))
    if not isinstance(ids, list):
        return None
    return [str(part_id) for part_id in ids]


def save_cached_part_ids(
    context: VehicleContext,
    selected: Iterable[str],
    part_ids: Iterable[str],
    max_entries: int = 8,
) -> None:
    path = parts_cache_path(context)
    fingerprint = context_fingerprint_hash(context.source_zip)
    selections: dict[str, list[str]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(data, dict)
            and data.get("fingerprint") == fingerprint
            and isinstance(data.get("selections"), dict)
        ):
            selections = {str(k): list(v) for k, v in data["selections"].items() if isinstance(v, list)}
    except Exception:
        pass
    key = selection_cache_key(selected)
    selections.pop(key, None)
    selections[key] = [str(part_id) for part_id in part_ids]
    while len(selections) > max_entries:
        selections.pop(next(iter(selections)))
    try:
        write_text_file(path, json.dumps({"fingerprint": fingerprint, "selections": selections}, indent=1))
    except Exception:
        pass


def clear_parts_cache(context: VehicleContext) -> None:
    try:
        parts_cache_path(context).unlink(missing_ok=True)
    except OSError:
        pass


HAND_DETECTION_CACHE_VERSION = 1


def variant_hands_cache_path(context: VehicleContext) -> Path:
    return context.project_dir / "variant_hands_cache.json"


def variant_hand_detection_signature(conversion: dict[str, object]) -> tuple[str, ...]:
    """Inputs from a conversion that can change stock-hand detection."""
    return tuple(sorted(selected_steering_refs(conversion)))


def variant_hands_cache_key(conversion: dict[str, object]) -> str:
    payload = json.dumps(variant_hand_detection_signature(conversion), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def variant_hands_cache_fingerprint(context: VehicleContext) -> str:
    payload = f"{HAND_DETECTION_CACHE_VERSION}:{context_fingerprint_hash(context.source_zip)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalized_cached_variant_hands(
    context: VehicleContext,
    value: object,
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(config_name): str(hand)
        for config_name, hand in value.items()
        if config_name in context.variants and hand in {HAND_LHD, HAND_RHD, HAND_UNKNOWN}
    }


def load_cached_variant_hands(
    context: VehicleContext,
    conversion: dict[str, object],
) -> dict[str, str] | None:
    """Load detected stock handedness for this steering-reference selection.

    Results are kept in memory and persisted across sessions. The source/common
    zip fingerprint and detection version prevent stale model data being reused.
    """
    key = variant_hands_cache_key(conversion)
    memory = _normalized_cached_variant_hands(context, context.variant_hands_cache.get(key))
    if memory:
        return dict(memory)

    path = variant_hands_cache_path(context)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("fingerprint") != variant_hands_cache_fingerprint(context):
        return None
    detections = data.get("detections")
    if not isinstance(detections, dict):
        return None
    hands = _normalized_cached_variant_hands(context, detections.get(key))
    if not hands:
        return None
    context.variant_hands_cache[key] = hands
    return dict(hands)


def save_cached_variant_hands(
    context: VehicleContext,
    conversion: dict[str, object],
    hands: dict[str, str],
    max_entries: int = 8,
) -> None:
    normalized = _normalized_cached_variant_hands(context, hands)
    if not normalized:
        return
    key = variant_hands_cache_key(conversion)
    context.variant_hands_cache[key] = dict(normalized)
    path = variant_hands_cache_path(context)
    fingerprint = variant_hands_cache_fingerprint(context)
    detections: dict[str, dict[str, str]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(data, dict)
            and data.get("fingerprint") == fingerprint
            and isinstance(data.get("detections"), dict)
        ):
            detections = {
                str(saved_key): _normalized_cached_variant_hands(context, saved_hands)
                for saved_key, saved_hands in data["detections"].items()
                if isinstance(saved_hands, dict)
            }
    except Exception:
        pass
    detections.pop(key, None)
    detections[key] = normalized
    while len(detections) > max_entries:
        detections.pop(next(iter(detections)))
    try:
        write_text_file(path, json.dumps({"fingerprint": fingerprint, "detections": detections}, indent=1))
    except Exception:
        pass


def clear_variant_hands_cache(context: VehicleContext) -> None:
    context.variant_hands_cache = {}
    try:
        variant_hands_cache_path(context).unlink(missing_ok=True)
    except OSError:
        pass


def load_vehicle_context(
    source_zip: Path,
    vehicle_id: str | None = None,
    *,
    use_cache: bool = True,
) -> VehicleContext:
    source_zip = Path(source_zip)
    vehicle_ids = vehicle_ids_in_zip(source_zip)
    if not vehicle_ids:
        raise RuntimeError(f"No BeamNG vehicles with DAE/PC/JBeam files found in {source_zip}")
    selected_vehicle_id = vehicle_id or vehicle_ids[0]
    if selected_vehicle_id not in vehicle_ids:
        raise RuntimeError(f"Vehicle {selected_vehicle_id!r} not found in {source_zip}")

    if use_cache:
        cached = load_cached_vehicle_context(source_zip, selected_vehicle_id)
        if cached is not None:
            cached.loaded_from_cache = True
            return cached

    dae_paths = direct_vehicle_files(source_zip, selected_vehicle_id, ".dae")
    if not dae_paths:
        dae_paths = list_vehicle_files(source_zip, selected_vehicle_id, ".dae")
    if not dae_paths:
        raise RuntimeError(f"No DAE files found for vehicles/{selected_vehicle_id}")

    objects: dict[str, DaeObject] = {}
    preview_by_id: dict[str, dict[str, object]] = {}
    for dae_path in dae_paths:
        # Parse once and feed both helpers; each used to re-parse the file.
        tree = parse_dae(source_zip, dae_path)
        for object_id, obj in dae_objects_from_tree(
            tree, dae_path, dae_source_zip=source_zip
        ).items():
            objects.setdefault(object_id, obj)
        for object_id, preview in preview_data_from_tree(tree).items():
            preview_by_id.setdefault(object_id, preview)

    variants: dict[str, VariantInfo] = {}
    for pc_path in direct_vehicle_files(source_zip, selected_vehicle_id, ".pc"):
        config_name = Path(pc_path).stem
        info_path = info_path_for_config(source_zip, selected_vehicle_id, config_name)
        display_name = display_name_for(source_zip, info_path, config_name)
        variants[config_name] = VariantInfo(
            name=config_name,
            pc_path=pc_path,
            info_path=info_path,
            display_name=display_name,
        )

    jbeam_texts = load_jbeam_texts(source_zip, selected_vehicle_id)
    part_body_index = build_part_body_index(jbeam_texts)
    common_jbeam_texts = load_common_jbeam_texts(source_zip)
    if common_jbeam_texts:
        common_part_index = build_part_body_index(common_jbeam_texts)
        reachable_common = reachable_common_part_index(part_body_index, common_part_index)
        if reachable_common:
            part_body_index.update(reachable_common)
            for _body, filename in reachable_common.values():
                jbeam_texts.setdefault(filename, common_jbeam_texts[filename])
    node_positions = build_node_position_index(jbeam_texts)
    common_objects, common_previews, common_daes = load_common_dae_objects(
        source_zip,
        referenced_mesh_names(part_body_index),
        objects,
    )
    objects.update(common_objects)
    preview_by_id.update(common_previews)
    mesh_pivots = {
        object_id: (obj.x, obj.y, obj.z)
        for object_id, obj in objects.items()
        if obj.dae_path
    }
    prop_objects, prop_previews = collect_prop_only_objects(
        jbeam_texts,
        node_positions,
        objects,
        part_body_index,
    )
    objects.update(prop_objects)
    preview_by_id.update(prop_previews)
    # Snapshot authored centres before apply_resolved_mesh_positions rewrites
    # preview_by_id below -- resolving a bare flexbody row's position needs
    # the ORIGINAL geometry centre, not the representative-shifted one.
    mesh_authored_centers = {
        object_id: tuple(preview["center"])
        for object_id, preview in preview_by_id.items()
        if isinstance(preview, dict) and "center" in preview
    }
    # Only the positioned-mesh set is wanted here; the placements themselves
    # span parts that cannot coexist, so positions come from the per-config
    # resolution below instead.
    _placements, positioned_flexbodies = collect_flexbody_mesh_placements(
        objects, part_body_index, mesh_pivots
    )
    project_dir = project_dir_for(source_zip, selected_vehicle_id)

    context = VehicleContext(
        source_zip=source_zip,
        vehicle_id=selected_vehicle_id,
        vehicle_path=vehicle_prefix(selected_vehicle_id),
        dae_paths=dae_paths + [path for path in common_daes if path not in dae_paths],
        variants=variants,
        objects=objects,
        preview_by_id=preview_by_id,
        jbeam_texts=jbeam_texts,
        node_positions=node_positions,
        project_dir=project_dir,
        part_body_index=part_body_index,
        jbeam_positioned_flexbodies=positioned_flexbodies,
        mesh_pivots=mesh_pivots,
        mesh_authored_centers=mesh_authored_centers,
    )
    # Resolving positions needs a finished context (variants, part index,
    # pivots), so it runs here rather than inline above.
    representative, variant_dependent = representative_mesh_positions(context)
    apply_resolved_mesh_positions(
        context.objects,
        context.preview_by_id,
        representative,
        context.mesh_pivots,
        context.jbeam_positioned_flexbodies,
    )
    context.variant_dependent_meshes = variant_dependent
    save_vehicle_context_cache(context)
    context.loaded_from_cache = False
    return context


def median_value(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def steering_ref_score(object_id: str, obj: DaeObject) -> int:
    lowered = f"{object_id} {obj.name}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if "steer" not in compact:
        return 0

    score = 5
    if "wheel" in compact or "swheel" in compact:
        score += 25
    if abs(obj.x) > 0.05:
        score += 10
    if any(token in lowered for token in STEERING_NAME_EXCLUDES):
        score -= 25
    return score


def is_default_steering_ref(object_id: str, obj: DaeObject) -> bool:
    # 15 = "steer" in the name + off-center placement with no excluded token;
    # vehicles like the etk800 name their wheels plain "steer"/"steer_01a"
    # without a "wheel" token, so demanding the wheel bonus finds nothing.
    return abs(obj.x) > 0.05 and steering_ref_score(object_id, obj) >= 15


def vehicle_prefix_rank(context: VehicleContext, object_id: str) -> int:
    """Vehicle-named meshes (etk800_steer) outrank shared-library wheels
    (steer_01a, ...): the prefixed mesh is the vehicle's own default fitment
    while the rest are optional customisation parts."""
    return 0 if object_id.lower().startswith(f"{context.vehicle_id.lower()}_") else 1


def keep_single_steering_ref(context: VehicleContext, parts: dict[str, object]) -> None:
    """The tool works with exactly ONE steering reference (the GUI enforces
    this on click); when several part settings are flagged, keep the
    best-scoring one (same ordering as likely_steering_ref_ids) and clear
    the rest in place."""
    refs = [
        object_id
        for object_id, settings in parts.items()
        if isinstance(settings, dict) and settings.get("steeringRef")
    ]
    if len(refs) <= 1:
        return

    def rank(object_id: str) -> tuple[int, int, int, float, str]:
        obj = context.objects.get(object_id)
        if obj is None:
            return (1, 0, 0, 0.0, object_id)
        return (
            0,
            -steering_ref_score(object_id, obj),
            vehicle_prefix_rank(context, object_id),
            -abs(obj.x),
            object_id,
        )

    best = min(refs, key=rank)
    for object_id in refs:
        if object_id != best:
            parts[object_id]["steeringRef"] = False


def ensure_default_steering_ref(context: VehicleContext, parts: dict[str, object]) -> None:
    """Re-run steering-ref auto-detection when no part carries the flag, so a
    save written without one (older tool versions, cleared by hand) recovers
    the default on load instead of silencing detection forever."""
    for settings in parts.values():
        if isinstance(settings, dict) and settings.get("steeringRef"):
            return
    for object_id, settings in parts.items():
        if not isinstance(settings, dict):
            continue
        obj = context.objects.get(object_id)
        if obj is not None and is_default_steering_ref(object_id, obj):
            settings["steeringRef"] = True
    keep_single_steering_ref(context, parts)


def likely_steering_ref_ids(
    context: VehicleContext,
    used_meshes: set[str] | None = None,
) -> list[str]:
    candidates = used_meshes if used_meshes is not None else set(context.objects)
    scored: list[tuple[int, int, float, str]] = []
    for object_id in candidates:
        obj = context.objects.get(object_id)
        if obj is None:
            continue
        if abs(obj.x) <= 0.05:
            continue
        score = steering_ref_score(object_id, obj)
        if score >= 15:
            scored.append((score, vehicle_prefix_rank(context, object_id), abs(obj.x), object_id))
    scored.sort(key=lambda item: (-item[0], item[1], -item[2], item[3]))
    return [object_id for _score, _prefix, _abs_x, object_id in scored]


def estimated_vehicle_center_x(
    context: VehicleContext,
    used_meshes: set[str],
    steering_ids: set[str],
) -> float:
    object_xs = [
        context.objects[object_id].x
        for object_id in used_meshes
        if object_id in context.objects and object_id not in steering_ids
    ]
    object_center = median_value(object_xs)
    if object_center is not None and len(object_xs) >= 8:
        return object_center

    node_center = median_value([position[0] for position in context.node_positions.values()])
    if node_center is not None:
        return node_center

    return object_center if object_center is not None else 0.0


def hand_from_steering_positions(
    context: VehicleContext,
    steering_ids: list[str],
    used_meshes: set[str] | None = None,
) -> str:
    existing_ids = [object_id for object_id in steering_ids if object_id in context.objects]
    if not existing_ids:
        return HAND_UNKNOWN
    mesh_scope = used_meshes if used_meshes is not None else set(context.objects)
    center_x = estimated_vehicle_center_x(context, mesh_scope, set(existing_ids))
    offsets = [
        context.objects[object_id].x - center_x
        for object_id in existing_ids
        if abs(context.objects[object_id].x - center_x) > 0.01
    ]
    if not offsets:
        return HAND_UNKNOWN
    left_count = sum(1 for offset in offsets if offset > 0)
    right_count = sum(1 for offset in offsets if offset < 0)
    if left_count and not right_count:
        return HAND_LHD
    if right_count and not left_count:
        return HAND_RHD
    average_offset = sum(offsets) / len(offsets)
    if average_offset > 0.05 and left_count > right_count:
        return HAND_LHD
    if average_offset < -0.05 and right_count > left_count:
        return HAND_RHD
    return HAND_UNKNOWN


def hand_from_offsets(offsets: list[float]) -> str:
    offsets = [offset for offset in offsets if abs(offset) > 0.01]
    if not offsets:
        return HAND_UNKNOWN
    left_count = sum(1 for offset in offsets if offset > 0)
    right_count = sum(1 for offset in offsets if offset < 0)
    if left_count and not right_count:
        return HAND_LHD
    if right_count and not left_count:
        return HAND_RHD
    average_offset = sum(offsets) / len(offsets)
    if average_offset > 0.05 and left_count > right_count:
        return HAND_LHD
    if average_offset < -0.05 and right_count > left_count:
        return HAND_RHD
    return HAND_UNKNOWN


def stock_steering_ref_score(object_id: str, obj: DaeObject) -> int:
    lowered = f"{object_id} {obj.name}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if "steer" not in compact:
        return 0

    score = 10
    if "wheel" in compact or "swheel" in compact:
        score += 25
    if "airbag" in compact:
        score += 10
    stock_excludes = tuple(token for token in STEERING_NAME_EXCLUDES if token != "airbag") + ("boot",)
    if any(token in lowered for token in stock_excludes):
        score -= 25
    return score


def likely_stock_steering_ref_ids(
    context: VehicleContext,
    used_meshes: set[str],
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for object_id in used_meshes:
        obj = context.objects.get(object_id)
        if obj is None:
            continue
        score = stock_steering_ref_score(object_id, obj)
        if score >= 10:
            scored.append((score, object_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [object_id for _score, object_id in scored]


def flexbody_row_needs_node_translation(context: VehicleContext, mesh: str) -> bool:
    """Whether mesh's own DAE node translation is real placement (add it) or a
    leftover export artefact the game ignores (drop it).

    The tell is whether ANY flexbody row placing this mesh authors its own
    "pos": if it does, the mesh is a reusable template the row positions
    (the D-Series gooseneck hitch is one part's pos:{y:0.325} away from
    another's pos:{y:-0.03}), and the node's translation is a second, real
    contribution that must be added on top -- dropping it put the hitch 3.85 m
    from the bed. A BARE row (mesh + material groups, no pos/rot/scale at all,
    e.g. etk800's manual shifter knob and boot) means the DAE vertices are
    already authored at their final position; the node still carries a
    translation, but it is a Blender-export artefact the game does not apply,
    and adding it moves the mesh across the vehicle (the shifter rendered on
    the passenger side, mirrored from the steering wheel). This is the exact
    signal jbeam_positioned_flexbodies already tracks for the same reason on
    the build side (generate_daes), just reused here for preview/detection."""
    return mesh in context.jbeam_positioned_flexbodies


def flexbody_mesh_reference_point(
    context: VehicleContext,
    mesh: str,
    obj: DaeObject,
) -> tuple[float, float, float]:
    """The point a flexbody row's own matrix should be applied to.

    KEEP: the node's authored translation (mesh_pivots) -- real placement.
    DROP: that translation backed out of the authored geometry centre, i.e.
    where the mesh sits once the node's redundant offset is removed. Falls
    back to the node translation if no authored centre was captured (e.g. a
    mesh with no geometry), which is at worst today's behaviour."""
    if flexbody_row_needs_node_translation(context, mesh):
        return context.mesh_pivots.get(mesh, (obj.x, obj.y, obj.z))
    center = context.mesh_authored_centers.get(mesh)
    pivot = context.mesh_pivots.get(mesh)
    if center is None or pivot is None:
        return context.mesh_pivots.get(mesh, (obj.x, obj.y, obj.z))
    return (center[0] - pivot[0], center[1] - pivot[1], center[2] - pivot[2])


def selected_flexbody_mesh_placements(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[MeshPlacement]]:
    """Flexbody placements for the parts ONE config actually selects.

    Unlike collect_flexbody_mesh_placements this never mixes parts that cannot
    coexist. Positions are measured from the authored pivot (or, for a mesh
    whose row authors no pos of its own, the authored geometry centre -- see
    flexbody_mesh_reference_point), so it is safe to call after DaeObject
    coordinates have been resolved."""
    selected = selected_parts_for_config(context, config_name)
    placements: dict[str, list[MeshPlacement]] = {}
    part_slot_options = selected.get("part_slot_options", {})
    for part_id in sorted(str(item) for item in selected.get("parts", set())):
        flexbodies = part_named_array_for_context(context, part_id, "flexbodies")
        if not flexbodies:
            continue
        inherited_options = ()
        if isinstance(part_slot_options, dict):
            raw_options = part_slot_options.get(part_id, ())
            if isinstance(raw_options, (list, tuple)):
                inherited_options = tuple(str(item) for item in raw_options if item)
        for row in iter_active_top_level_rows(flexbodies):
            mesh = flexbody_row_mesh(row)
            if mesh not in mesh_ids:
                continue
            obj = context.objects.get(mesh)
            if obj is None:
                continue
            pivot = flexbody_mesh_reference_point(context, mesh, obj)
            matrix = flexbody_row_source_matrix(row, inherited_options)
            placements.setdefault(mesh, []).append(
                MeshPlacement(
                    position=transform_helpers.transform_point(matrix, pivot),
                    matrix=matrix,
                )
            )
    return placements


def selected_flexbody_mesh_positions(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[tuple[float, float, float]]]:
    return {
        mesh: [placement.position for placement in placements]
        for mesh, placements in selected_flexbody_mesh_placements(
            context, config_name, mesh_ids
        ).items()
    }


def resolved_mesh_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, ResolvedMeshPosition]:
    """Where each mesh of one trim actually sits.

    This is the honest answer the averaged DaeObject position cannot give: a
    mesh declared by several mutually exclusive parts (the D-Series gooseneck
    hitch sits in five, at two different offsets) resolves here to the offset
    of the part THIS trim selects."""
    cached = context.resolved_positions_cache.get(config_name)
    if cached is not None:
        return cached

    used = used_meshes_for_config(context, config_name)
    flex = selected_flexbody_mesh_placements(context, config_name, used)
    props = selected_prop_mesh_positions(context, config_name, used)

    resolved: dict[str, ResolvedMeshPosition] = {}
    for mesh in used:
        # jbeam hides a part it does not want by parking it kilometres away
        # (astrah stows a spare licence plate at y=-4.5e6). Those rows render
        # nothing, so averaging them in would drag the mesh off the vehicle --
        # the preview payload discards them on the same threshold.
        flex_placements = [
            placement
            for placement in flex.get(mesh, [])
            if not is_far_placement(placement.position)
        ]
        points = [placement.position for placement in flex_placements]
        points.extend(
            position for position in props.get(mesh, []) if not is_far_placement(position)
        )
        if not points:
            continue
        resolved[mesh] = ResolvedMeshPosition(
            position=average_position(points),
            matrices=tuple(
                tuple(tuple(row) for row in placement.matrix)
                for placement in flex_placements
            ),
        )
    context.resolved_positions_cache[config_name] = resolved
    return resolved


def preview_entries_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, dict[str, object]]:
    """preview_by_id shifted from the representative onto one trim.

    context.preview_by_id is baked once with the representative placement, so
    for a variant-dependent mesh its box sits where that mesh lands on some
    OTHER trim (the D-Series gooseneck hitch box is 0.61 m out on the long
    bed). Shifting by the difference between the two resolved positions is
    exact whenever the trims differ by translation, which is the case that
    produces a visible offset."""
    resolved = resolved_mesh_positions_for_config(context, config_name)
    entries: dict[str, dict[str, object]] = dict(context.preview_by_id)
    for mesh, entry in resolved.items():
        preview = context.preview_by_id.get(mesh)
        obj = context.objects.get(mesh)
        if preview is None or obj is None:
            continue
        delta = (
            entry.position[0] - obj.x,
            entry.position[1] - obj.y,
            entry.position[2] - obj.z,
        )
        if max(abs(value) for value in delta) < 1e-9:
            continue
        entries[mesh] = translate_preview_points(preview, delta)
    return entries


def representative_mesh_positions(
    context: VehicleContext,
) -> tuple[dict[str, ResolvedMeshPosition], set[str]]:
    """One position per mesh for callers with no trim in hand, plus the set of
    meshes for which that position is only a representative.

    The representative is the position the mesh holds in the MOST trims, ties
    broken by the alphabetically-first trim so it never depends on dict order.
    A mesh placed identically everywhere -- the overwhelming majority, and
    every steering reference measured so far -- resolves to exactly that
    position, which is what the old whole-index average produced too.

    Deliberately NOT the authored DAE pivot: shared-library meshes such as
    grp_steerwheel_hub are authored at the origin and positioned entirely by
    their jbeam row, so pivots would report x=0 and collapse the conversion
    delta computed from the steering reference."""
    grouped: dict[str, dict[tuple[float, ...], list[str]]] = {}
    entries: dict[str, dict[tuple[float, ...], ResolvedMeshPosition]] = {}
    for config_name in sorted(context.variants):
        for mesh, entry in resolved_mesh_positions_for_config(context, config_name).items():
            key = tuple(round(value, 6) for value in entry.position)
            grouped.setdefault(mesh, {}).setdefault(key, []).append(config_name)
            entries.setdefault(mesh, {}).setdefault(key, entry)

    representative: dict[str, ResolvedMeshPosition] = {}
    variant_dependent: set[str] = set()
    for mesh, groups in grouped.items():
        if len(groups) > 1:
            variant_dependent.add(mesh)
        winner = min(groups, key=lambda key: (-len(groups[key]), min(groups[key])))
        representative[mesh] = entries[mesh][winner]
    return representative, variant_dependent


def stock_steering_positions_for_config(
    context: VehicleContext,
    config_name: str,
    steering_ids: list[str],
) -> list[tuple[float, float, float]]:
    wanted = {object_id for object_id in steering_ids if object_id in context.objects}
    if not wanted:
        return []
    positions_by_mesh = selected_prop_mesh_positions(context, config_name, wanted)
    flex_positions = selected_flexbody_mesh_positions(context, config_name, wanted - set(positions_by_mesh))
    positions: list[tuple[float, float, float]] = []
    for mesh in steering_ids:
        positions.extend(positions_by_mesh.get(mesh, ()))
        positions.extend(flex_positions.get(mesh, ()))
    return positions


def selected_vehicle_center_x(context: VehicleContext, config_name: str, used_meshes: set[str]) -> float:
    selected_nodes = selected_node_positions_for_config(context, config_name)
    node_center = median_value([position[0] for position in selected_nodes.values()])
    if node_center is not None:
        return node_center
    return estimated_vehicle_center_x(context, used_meshes, set())


def hand_from_stock_steering_for_variant(
    context: VehicleContext,
    config_name: str,
    steering_ids: list[str],
    used_meshes: set[str],
) -> str:
    positions = stock_steering_positions_for_config(context, config_name, steering_ids)
    if not positions:
        return HAND_UNKNOWN
    center_x = selected_vehicle_center_x(context, config_name, used_meshes)
    return hand_from_offsets([position[0] - center_x for position in positions])


def used_meshes_for_config(context: VehicleContext, config_name: str) -> set[str]:
    return set(mesh_roles_for_config(context, config_name)[2])


def default_part_settings(context: VehicleContext) -> dict[str, dict[str, object]]:
    settings: dict[str, dict[str, object]] = {}
    for object_id, obj in sorted(context.objects.items()):
        settings[object_id] = {
            "mode": MODE_SKIP,
            "mirrorSource": None,
            "translateOffset": None,
            "textureFlip": False,
            "steeringRef": is_default_steering_ref(object_id, obj),
            "viewerVisible": True,
            "viewerSolo": False,
        }
    # Vehicles often index several confident candidates (steering wheel
    # variants, columns, ...); auto-detect must flag only the best one.
    keep_single_steering_ref(context, settings)
    return settings


def default_variant_settings(context: VehicleContext) -> dict[str, dict[str, object]]:
    return {
        name: {
            "selected": False,
            "build": BUILD_OFF,
            "sourceHandOverride": HAND_AUTO,
            "frontPlate": plate_generator.PLATE_PART_AUTO,
            "rearPlate": plate_generator.PLATE_PART_AUTO,
        }
        for name in sorted(context.variants)
    }


def variant_build_mode(settings: object) -> str:
    """Return the output mode for one source trim.

    ``selected`` is retained as a compatibility mirror for pre-XP projects;
    old saves therefore migrate to Converted/Off without changing behaviour.
    """
    if not isinstance(settings, dict):
        return BUILD_OFF
    mode = str(settings.get("build") or "").lower()
    if mode in BUILD_CHOICES:
        return mode
    return BUILD_CONVERTED if settings.get("selected") else BUILD_OFF


def set_variant_build_mode(settings: dict[str, object], mode: str) -> None:
    normalized = mode if mode in BUILD_CHOICES else BUILD_OFF
    settings["build"] = normalized
    settings["selected"] = normalized != BUILD_OFF


def base_conversion_config(context: VehicleContext) -> dict[str, object]:
    return {
        "toolVersion": TOOL_VERSION,
        "source": {
            "fileName": context.source_zip.name,
            "sourcePath": str(context.source_zip),
            "vehicleId": context.vehicle_id,
            "daeFiles": context.dae_paths,
            "configs": sorted(context.variants),
        },
        "variants": default_variant_settings(context),
        "parts": default_part_settings(context),
        "plate": plate_generator.default_plate_binding(),
        "delta": {
            "manual": False,
            "magnitude": None,
            "steeringRefs": [
                object_id
                for object_id, part in default_part_settings(context).items()
                if part.get("steeringRef")
            ],
        },
    }


def conversion_path(context: VehicleContext) -> Path:
    return context.project_dir / "conversion.json"


def load_or_create_conversion(context: VehicleContext) -> tuple[dict[str, object], bool]:
    path = conversion_path(context)
    if path.exists():
        data = read_json_file(path)
        source = data.get("source", {})
        if not isinstance(source, dict) or source.get("vehicleId") in {None, context.vehicle_id}:
            return merge_with_current_inventory(context, data), True
    return base_conversion_config(context), False


def merge_with_current_inventory(context: VehicleContext, data: dict[str, object]) -> dict[str, object]:
    merged = base_conversion_config(context)
    old_variants = data.get("variants", {})
    if isinstance(old_variants, dict):
        for name, settings in old_variants.items():
            if name in merged["variants"] and isinstance(settings, dict):
                merged["variants"][name].update(
                    {
                        key: settings[key]
                        for key in (
                            "selected",
                            "build",
                            "sourceHandOverride",
                            "plate",
                            "frontPlate",
                            "rearPlate",
                        )
                        if key in settings
                    }
                )
                if "build" in settings:
                    migrated_build = variant_build_mode(settings)
                elif "selected" in settings:
                    migrated_build = BUILD_CONVERTED if settings.get("selected") else BUILD_OFF
                else:
                    migrated_build = variant_build_mode(merged["variants"][name])
                set_variant_build_mode(merged["variants"][name], migrated_build)
                merged["variants"][name]["plate"] = plate_generator.normalized_plate_binding(
                    merged["variants"][name].get("plate"), variant=True
                )

    if isinstance(data.get("plate"), dict):
        merged["plate"] = plate_generator.normalized_plate_binding(data["plate"])

    old_parts = data.get("parts", {})
    if isinstance(old_parts, dict):
        # The save's steering-ref choice wins over the auto-detected default:
        # clear the default flag(s) whenever the save carries a usable ref, so
        # the two can never combine into multiple refs.
        saved_has_ref = any(
            isinstance(settings, dict)
            and settings.get("steeringRef")
            and object_id in merged["parts"]
            for object_id, settings in old_parts.items()
        )
        if saved_has_ref:
            for settings in merged["parts"].values():
                settings["steeringRef"] = False
        for object_id, settings in old_parts.items():
            if object_id in merged["parts"] and isinstance(settings, dict):
                merged["parts"][object_id].update(
                    {
                        key: settings[key]
                        for key in (
                            "mode",
                            "mirrorSource",
                            "translateOffset",
                            "textureFlip",
                            "steeringRef",
                            "viewerVisible",
                            "viewerSolo",
                        )
                        if key in settings
                    }
                )
        # Old saves written before single-ref enforcement may carry several.
        keep_single_steering_ref(context, merged["parts"])
    # A save without any ref (older tool, different detection rules) must not
    # pin detection off forever: re-run it whenever the merge ends up empty.
    ensure_default_steering_ref(context, merged["parts"])

    old_delta = data.get("delta", {})
    if isinstance(old_delta, dict):
        merged["delta"].update(
            {
                key: old_delta[key]
                for key in ("manual", "magnitude")
                if key in old_delta
            }
        )
    return merged


def import_matching_conversion(
    context: VehicleContext,
    current: dict[str, object],
    imported: dict[str, object],
) -> tuple[dict[str, object], dict[str, int]]:
    out = merge_with_current_inventory(context, current)
    imported_variants = imported.get("variants", {})
    imported_parts = imported.get("parts", {})
    counts = {
        "variantImported": 0,
        "variantSkipped": 0,
        "partImported": 0,
        "partSkipped": 0,
    }

    if isinstance(imported_variants, dict):
        for name, settings in imported_variants.items():
            if name in out["variants"] and isinstance(settings, dict):
                out["variants"][name].update(
                    {
                        key: settings[key]
                        for key in (
                            "selected",
                            "build",
                            "sourceHandOverride",
                            "plate",
                            "frontPlate",
                            "rearPlate",
                        )
                        if key in settings
                    }
                )
                if "build" in settings:
                    imported_build = variant_build_mode(settings)
                elif "selected" in settings:
                    imported_build = BUILD_CONVERTED if settings.get("selected") else BUILD_OFF
                else:
                    imported_build = variant_build_mode(out["variants"][name])
                set_variant_build_mode(out["variants"][name], imported_build)
                out["variants"][name]["plate"] = plate_generator.normalized_plate_binding(
                    out["variants"][name].get("plate"), variant=True
                )
                counts["variantImported"] += 1
            else:
                counts["variantSkipped"] += 1

    if isinstance(imported_parts, dict):
        # Same single-ref rule as merge_with_current_inventory: an imported
        # steering ref replaces the current one instead of joining it.
        imported_has_ref = any(
            isinstance(settings, dict)
            and settings.get("steeringRef")
            and object_id in out["parts"]
            for object_id, settings in imported_parts.items()
        )
        if imported_has_ref:
            for settings in out["parts"].values():
                settings["steeringRef"] = False
        for object_id, settings in imported_parts.items():
            if object_id in out["parts"] and isinstance(settings, dict):
                out["parts"][object_id].update(
                    {
                        key: settings[key]
                        for key in (
                            "mode",
                            "mirrorSource",
                            "translateOffset",
                            "textureFlip",
                            "steeringRef",
                            "viewerVisible",
                            "viewerSolo",
                        )
                        if key in settings
                    }
                )
                counts["partImported"] += 1
            else:
                counts["partSkipped"] += 1
        keep_single_steering_ref(context, out["parts"])
    ensure_default_steering_ref(context, out["parts"])
    if isinstance(imported.get("plate"), dict):
        out["plate"] = plate_generator.normalized_plate_binding(imported["plate"])
    return out, counts


def save_conversion(context: VehicleContext, conversion: dict[str, object]) -> Path:
    context.project_dir.mkdir(parents=True, exist_ok=True)
    conversion["toolVersion"] = TOOL_VERSION
    conversion["plate"] = plate_generator.normalized_plate_binding(conversion.get("plate"))
    variants = conversion.get("variants", {})
    if isinstance(variants, dict):
        for settings in variants.values():
            if not isinstance(settings, dict):
                continue
            set_variant_build_mode(settings, variant_build_mode(settings))
            settings["plate"] = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
            settings["frontPlate"] = plate_generator.normalized_plate_part_choice(settings.get("frontPlate"))
            settings["rearPlate"] = plate_generator.normalized_plate_part_choice(settings.get("rearPlate"))
    delta = conversion.setdefault("delta", {})
    if isinstance(delta, dict):
        delta["steeringRefs"] = selected_steering_refs(conversion)
    conversion["source"] = {
        "fileName": context.source_zip.name,
        "sourcePath": str(context.source_zip),
        "vehicleId": context.vehicle_id,
        "daeFiles": context.dae_paths,
        "configs": sorted(context.variants),
    }
    path = conversion_path(context)
    write_text_file(path, json.dumps(conversion, indent=2), encoding="utf-8")
    return path


def load_app_settings() -> dict[str, object]:
    default_mods = default_beamng_mods_dir()
    data: dict[str, object] = {}
    if APP_SETTINGS_PATH.exists():
        try:
            data = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    return {
        "modsFolder": data.get("modsFolder") or str(default_mods),
        "blenderExecutable": data.get("blenderExecutable") or "",
        "lastVehicleZipPath": data.get("lastVehicleZipPath") or "",
        "lastVehicleId": data.get("lastVehicleId") or "",
        "lastVehicleZipFolder": data.get("lastVehicleZipFolder") or str(default_mods),
        "lastModsFolder": data.get("lastModsFolder") or str(default_mods),
        "lastBlenderFolder": data.get("lastBlenderFolder") or r"C:\Program Files",
        "previewOutputByVehicle": data.get("previewOutputByVehicle")
        if isinstance(data.get("previewOutputByVehicle"), dict)
        else {},
        "recentVehicles": data.get("recentVehicles")
        if isinstance(data.get("recentVehicles"), list)
        else [],
    }


def save_app_settings(settings: dict[str, object]) -> None:
    write_text_file(APP_SETTINGS_PATH, json.dumps(settings, indent=2), encoding="utf-8")


def find_part_body(
    part_id: str,
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    if part_body_index is not None:
        found = part_body_index.get(part_id)
        if found is not None:
            return found
    for name, text in jbeam_texts.items():
        body = transform_helpers.extract_keyed_object(text, part_id)
        if body is not None and '"slotType"' in body:
            return body, name
    return None


def part_body_for_context(context: VehicleContext, part_id: str) -> tuple[str, str] | None:
    return find_part_body(part_id, context.jbeam_texts, context.part_body_index)


def part_named_array_for_context(context: VehicleContext, part_id: str, array_key: str) -> str | None:
    cache_key = (part_id, array_key)
    if cache_key in context.part_array_cache:
        return context.part_array_cache[cache_key]
    found = part_body_for_context(context, part_id)
    if found is None:
        context.part_array_cache[cache_key] = None
        return None
    array_text = transform_helpers.extract_named_array(found[0], array_key)
    context.part_array_cache[cache_key] = array_text
    return array_text


def resolve_selected_parts(
    pc: dict[str, object],
    jbeam_texts: dict[str, str],
    *,
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    explicit_parts = {
        str(slot_type): str(part_id)
        for slot_type, part_id in dict(pc.get("parts", {})).items()
    }
    main_part = str(pc.get("mainPartName") or vehicle_id)
    selected: set[str] = set()
    missing_parts: set[str] = set()
    queue: list[tuple[str, tuple[str, ...]]] = [(main_part, tuple())]
    selected_by_slot: dict[str, str] = {"main": main_part}
    part_slot_options: dict[str, tuple[str, ...]] = {main_part: tuple()}
    for slot_type, part_id in explicit_parts.items():
        if not part_id:
            continue
        selected_by_slot[slot_type] = part_id

    def process_queue() -> None:
        while queue:
            part_id, inherited_options = queue.pop(0)
            if not part_id or part_id in selected:
                continue
            selected.add(part_id)
            part_slot_options.setdefault(part_id, inherited_options)

            found = find_part_body(part_id, jbeam_texts, part_body_index)
            if found is None:
                missing_parts.add(part_id)
                continue
            part_body, _filename = found

            for slot_def in extract_slot_defs(part_body):
                chosen = explicit_parts.get(slot_def.slot_type, slot_def.default_part)
                if not chosen:
                    continue
                selected_by_slot[slot_def.slot_type] = chosen
                child_options = list(inherited_options)
                if slot_def.options:
                    child_options.append(slot_def.options)
                child_options_tuple = tuple(child_options)
                part_slot_options.setdefault(chosen, child_options_tuple)
                if chosen not in selected:
                    queue.append((chosen, child_options_tuple))

    process_queue()
    for part_id in explicit_parts.values():
        if part_id and part_id not in selected:
            part_slot_options.setdefault(part_id, tuple())
            queue.append((part_id, tuple()))
    process_queue()

    return {
        "main_part": main_part,
        "parts": selected,
        "selected_by_slot": selected_by_slot,
        "part_slot_options": part_slot_options,
        "missing_parts": missing_parts,
    }


def selected_parts_for_config(context: VehicleContext, config_name: str) -> dict[str, object]:
    cached = context.selected_parts_cache.get(config_name)
    if cached is not None:
        return cached
    variant = context.variants[config_name]
    pc = load_pc(context.source_zip, variant.pc_path)
    selected = resolve_selected_parts(
        pc,
        context.jbeam_texts,
        vehicle_id=context.vehicle_id,
        part_body_index=context.part_body_index,
    )
    context.selected_parts_cache[config_name] = selected
    return selected


def selected_node_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, tuple[float, float, float]]:
    cached = context.selected_node_positions_cache.get(config_name)
    if cached is not None:
        return cached

    selected = selected_parts_for_config(context, config_name)
    nodes: dict[str, tuple[float, float, float]] = {}
    part_slot_options = selected.get("part_slot_options", {})
    for part_id in sorted(str(item) for item in selected.get("parts", set())):
        node_array = part_named_array_for_context(context, part_id, "nodes")
        if not node_array:
            continue
        inherited_options = ()
        if isinstance(part_slot_options, dict):
            raw_options = part_slot_options.get(part_id, ())
            if isinstance(raw_options, (list, tuple)):
                inherited_options = tuple(str(item) for item in raw_options if item)
        node_re = re.compile(
            rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
            rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})'
        )
        for row in iter_top_level_rows(node_array):
            match = node_re.match(row)
            if match is None:
                continue
            node_id = match.group("id")
            if node_id in {"id", "type", "mesh", "func"}:
                continue
            position = (
                float(match.group("x")),
                float(match.group("y")),
                float(match.group("z")),
            )
            nodes.setdefault(node_id, pos_after_node_transforms(row, position, inherited_options))

    context.selected_node_positions_cache[config_name] = nodes
    return nodes


def selected_node_positions_for_parts(
    selected: dict[str, object],
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[float, float, float]]:
    nodes: dict[str, tuple[float, float, float]] = {}
    part_slot_options = selected.get("part_slot_options", {})
    for part_id in sorted(str(item) for item in selected.get("parts", set())):
        found = find_part_body(part_id, jbeam_texts, part_body_index)
        if found is None:
            continue
        part_body, _filename = found
        node_array = transform_helpers.extract_named_array(part_body, "nodes")
        if not node_array:
            continue
        inherited_options = ()
        if isinstance(part_slot_options, dict):
            raw_options = part_slot_options.get(part_id, ())
            if isinstance(raw_options, (list, tuple)):
                inherited_options = tuple(str(item) for item in raw_options if item)
        node_re = re.compile(
            rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
            rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})'
        )
        for row in iter_top_level_rows(node_array):
            match = node_re.match(row)
            if match is None:
                continue
            node_id = match.group("id")
            if node_id in {"id", "type", "mesh", "func"}:
                continue
            position = (
                float(match.group("x")),
                float(match.group("y")),
                float(match.group("z")),
            )
            nodes.setdefault(node_id, pos_after_node_transforms(row, position, inherited_options))
    return nodes


def prop_row_mesh(row: str) -> str | None:
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 2:
        return None
    func, mesh = strings[:2]
    if func == "func" or mesh == "mesh":
        return None
    return mesh


def prop_row_nodes_present(row: str, node_positions: dict[str, tuple[float, float, float]]) -> bool:
    """Whether the row's idRef/idX/idY nodes all exist in node_positions.

    The engine only spawns a prop when its reference nodes exist in the
    assembled vehicle; pass the SELECTED parts' node positions to reproduce
    that (a global all-files node index would also resolve dormant rows,
    e.g. the manual handbrake mount in a sequential-shifter config)."""
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return True
    return all(node_id in node_positions for node_id in strings[2:5])


def selected_prop_mesh_positions(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[tuple[float, float, float]]]:
    selected = selected_parts_for_config(context, config_name)
    node_positions = selected_node_positions_for_config(context, config_name)
    positions: dict[str, list[tuple[float, float, float]]] = {}
    part_slot_options = selected.get("part_slot_options", {})
    for part_id in sorted(str(item) for item in selected.get("parts", set())):
        props = part_named_array_for_context(context, part_id, "props")
        if not props:
            continue
        inherited_options = ()
        if isinstance(part_slot_options, dict):
            raw_options = part_slot_options.get(part_id, ())
            if isinstance(raw_options, (list, tuple)):
                inherited_options = tuple(str(item) for item in raw_options if item)
        for row in iter_top_level_rows(props):
            mesh = prop_row_mesh(row)
            if mesh not in mesh_ids:
                continue
            pivot = context.mesh_pivots.get(mesh)
            position = prop_row_pivot_position(row, node_positions, pivot, inherited_options)
            if position is not None:
                positions.setdefault(mesh, []).append(position)
    return positions


def mesh_roles_for_config(context: VehicleContext, config_name: str) -> tuple[set[str], set[str], set[str]]:
    cached = context.mesh_roles_cache.get(config_name)
    if cached is not None:
        return cached

    flexbody_meshes: set[str] = set()
    prop_meshes: set[str] = set()
    all_meshes: set[str] = set()
    selected = selected_parts_for_config(context, config_name)
    for part_id in selected["parts"]:
        part_id = str(part_id)
        flexbodies = part_named_array_for_context(context, part_id, "flexbodies")
        if flexbodies:
            for mesh in re.findall(r'\[\s*"((?:[^"\\]|\\.)*)"\s*(?=,|\[|\{)', flexbodies):
                if mesh and mesh != "mesh":
                    flexbody_meshes.add(mesh)
                    all_meshes.add(mesh)
        props = part_named_array_for_context(context, part_id, "props")
        if props:
            for _full, _func, mesh in transform_helpers.PROP_FUNC_MESH_RE.findall(props):
                if mesh and mesh != "mesh":
                    prop_meshes.add(mesh)
                    all_meshes.add(mesh)

    roles = (flexbody_meshes, prop_meshes, all_meshes)
    context.mesh_roles_cache[config_name] = roles
    return roles


def selected_mesh_roles(
    context: VehicleContext,
    selected_configs: list[str],
) -> tuple[set[str], set[str], set[str]]:
    flexbody_meshes: set[str] = set()
    prop_meshes: set[str] = set()
    all_meshes: set[str] = set()
    for config_name in selected_configs:
        config_flex, config_props, config_all = mesh_roles_for_config(context, config_name)
        flexbody_meshes.update(config_flex)
        prop_meshes.update(config_props)
        all_meshes.update(config_all)
    return flexbody_meshes, prop_meshes, all_meshes


def active_part_modes(conversion: dict[str, object]) -> dict[str, str]:
    parts = conversion.get("parts", {})
    modes: dict[str, str] = {}
    if not isinstance(parts, dict):
        return modes
    for object_id, settings in parts.items():
        if not isinstance(settings, dict):
            continue
        mode = str(settings.get("mode", MODE_SKIP))
        if mode in MODE_CHOICES and mode != MODE_SKIP:
            modes[str(object_id)] = mode
    return modes


def texture_flip_mesh_ids(
    conversion: dict[str, object],
    object_modes: dict[str, str],
) -> set[str]:
    """Mirrored parts whose texture must keep its left/right reading (nav
    screens, decals with text): their DAE copies get the TEXCOORD S axis
    reflected alongside the geometric mirror. Mirror Aesthetic only —
    Mirror Structural swaps in an opposite-side mesh that already carries
    its own correct mapping."""
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return set()
    return {
        object_id
        for object_id, mode in object_modes.items()
        if mode == MODE_MIRROR
        and isinstance(parts.get(object_id), dict)
        and bool(parts[object_id].get("textureFlip"))
    }


def structural_mirror_source_for_settings(
    context: VehicleContext,
    object_id: str,
    settings: object,
) -> str | None:
    if not isinstance(settings, dict):
        return None
    source_id = str(settings.get("mirrorSource") or "")
    source_obj = context.objects.get(source_id)
    if source_id and source_id != object_id and source_obj is not None and source_obj.dae_path:
        return source_id
    return None


def structural_mirror_sources(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str] | None = None,
) -> dict[str, str]:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return {}
    wanted = object_modes or active_part_modes(conversion)
    sources: dict[str, str] = {}
    for object_id, mode in wanted.items():
        if mode != MODE_MIRROR_STRUCTURAL:
            continue
        settings = parts.get(object_id, {})
        source_id = structural_mirror_source_for_settings(context, object_id, settings)
        if source_id is not None:
            sources[object_id] = source_id
    return sources


def fallback_structural_part_modes(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str] | None = None,
    *,
    selected_configs: Iterable[str] = (),
) -> dict[str, str]:
    modes = dict(object_modes or active_part_modes(conversion))
    if not modes:
        return modes
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return modes
    for object_id, mode in list(modes.items()):
        if mode != MODE_MIRROR_STRUCTURAL:
            continue
        source_id = structural_mirror_source_for_settings(context, object_id, parts.get(object_id, {}))
        if source_id is None:
            modes[object_id] = MODE_MIRROR
    return modes


def selected_steering_refs(conversion: dict[str, object]) -> list[str]:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return []
    return [
        str(object_id)
        for object_id, settings in parts.items()
        if isinstance(settings, dict) and settings.get("steeringRef")
    ]


def auto_delta_source_refs(context: VehicleContext, conversion: dict[str, object]) -> list[str]:
    """Steering-ref parts that actually contribute to the auto delta (indexed
    objects with a usable off-center X). Empty means the auto delta falls back
    to its default of 0."""
    return [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects and abs(context.objects[object_id].x) > 0.05
    ]


STEERING_PROP_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def steering_column_axis_offsets(context: VehicleContext) -> list[float]:
    """|x| of each steering column's rotation centre -- the idRef node of a
    ``func:"steering"`` prop row, the point the wheel animation spins around.

    A delta fallback only. It is NOT the primary signal because the rotation
    centre is only as trustworthy as the mod authored it: the sheik_yaris
    places its wheel on the right (x=-0.33) but leaves int_strw on the left
    (x=+0.37), so trusting the node there would mirror the wheel the wrong
    way. Where the wheel's own geometry gives a usable offset that wins; this
    covers the case where it does not, e.g. a wheel mesh authored at the
    origin and placed entirely by its prop row."""
    offsets: list[float] = []
    seen: set[str] = set()
    for body, _filename in context.part_body_index.values():
        props = transform_helpers.extract_named_array(body, "props")
        if not props or '"steering"' not in props:
            continue
        for row in iter_active_top_level_rows(props):
            strings = STEERING_PROP_STR_RE.findall(row)
            if len(strings) < 5 or strings[0] != "steering":
                continue
            idref = strings[2]
            if idref in seen:
                continue
            seen.add(idref)
            node = context.node_positions.get(idref)
            if node is not None and abs(node[0]) > 0.05:
                offsets.append(abs(node[0]))
    return offsets


def auto_delta_magnitude(context: VehicleContext, conversion: dict[str, object]) -> float:
    offsets = [
        abs(context.objects[object_id].x)
        for object_id in auto_delta_source_refs(context, conversion)
    ]
    offset = median_value(offsets)
    if offset is None:
        # No steering wheel geometry far enough off-centre to measure; fall
        # back to the steering column's rotation centre. Never overrides usable
        # wheel geometry, so validated conversions are unaffected.
        offset = median_value(steering_column_axis_offsets(context))
    if offset is None:
        return 0.0
    return offset * 2.0


def delta_magnitude(context: VehicleContext, conversion: dict[str, object]) -> float:
    delta = conversion.get("delta", {})
    if isinstance(delta, dict) and delta.get("manual"):
        try:
            return abs(float(delta.get("magnitude") or 0.0))
        except (TypeError, ValueError):
            return 0.0
    return auto_delta_magnitude(context, conversion)


def part_translate_magnitude(
    context: VehicleContext,
    conversion: dict[str, object],
    object_id: str,
) -> float:
    parts = conversion.get("parts", {})
    settings = parts.get(object_id, {}) if isinstance(parts, dict) else {}
    if isinstance(settings, dict):
        raw = settings.get("translateOffset")
        if raw not in (None, ""):
            try:
                return abs(float(raw))
            except (TypeError, ValueError):
                return 0.0
    return delta_magnitude(context, conversion)


def part_translate_magnitudes(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str],
) -> dict[str, float]:
    return {
        object_id: part_translate_magnitude(context, conversion, object_id)
        for object_id, mode in object_modes.items()
        if mode == MODE_TRANSLATE
    }


def detect_hand_for_variant(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str:
    used_meshes = used_meshes_for_config(context, config_name)
    explicit_used_refs = [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects and object_id in used_meshes
    ]
    stock_hand = hand_from_stock_steering_for_variant(
        context,
        config_name,
        explicit_used_refs,
        used_meshes,
    )
    if stock_hand != HAND_UNKNOWN:
        return stock_hand

    stock_hand = hand_from_stock_steering_for_variant(
        context,
        config_name,
        likely_stock_steering_ref_ids(context, used_meshes),
        used_meshes,
    )
    if stock_hand != HAND_UNKNOWN:
        return stock_hand

    variant = context.variants[config_name]
    metadata_hand = hand_from_text(f"{variant.name} {variant.display_name}")
    if metadata_hand != HAND_UNKNOWN:
        return metadata_hand

    explicit_refs = [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects
    ]
    global_hand = hand_from_steering_positions(context, explicit_refs)
    if global_hand != HAND_UNKNOWN:
        return global_hand

    global_hand = hand_from_steering_positions(context, likely_steering_ref_ids(context))
    if global_hand != HAND_UNKNOWN:
        return global_hand

    steering_ids = [
        object_id
        for object_id in explicit_refs
        if object_id in context.objects and object_id in used_meshes
    ]
    if not steering_ids:
        steering_ids = likely_steering_ref_ids(context, used_meshes)
    return hand_from_steering_positions(context, steering_ids, used_meshes)


def detect_hands_for_variants(
    context: VehicleContext,
    conversion: dict[str, object],
) -> dict[str, str]:
    """Return all stock-hand detections, filling only cache misses."""
    hands = load_cached_variant_hands(context, conversion) or {}
    changed = False
    for config_name in sorted(context.variants):
        if config_name in hands:
            continue
        hands[config_name] = detect_hand_for_variant(context, conversion, config_name)
        changed = True
    if changed:
        save_cached_variant_hands(context, conversion, hands)
    return hands


def cached_hand_for_variant(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str:
    hands = load_cached_variant_hands(context, conversion) or {}
    hand = hands.get(config_name)
    if hand is not None:
        return hand
    hand = detect_hand_for_variant(context, conversion, config_name)
    hands[config_name] = hand
    save_cached_variant_hands(context, conversion, hands)
    return hand


def effective_source_hand(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str:
    variant_settings = dict(conversion.get("variants", {}).get(config_name, {}))
    override = str(variant_settings.get("sourceHandOverride", HAND_AUTO))
    if override in {HAND_LHD, HAND_RHD, HAND_UNKNOWN}:
        return override
    return cached_hand_for_variant(context, conversion, config_name)


def target_hand_for(
    source_hand: str,
    action: str,
) -> str | None:
    if action == ACTION_SKIP:
        return None
    if action == ACTION_OPPOSITE:
        if source_hand == HAND_LHD:
            return HAND_RHD
        if source_hand == HAND_RHD:
            return HAND_LHD
        return None
    if action == ACTION_TO_RHD:
        return None if source_hand == HAND_RHD else HAND_RHD
    if action == ACTION_TO_LHD:
        return None if source_hand == HAND_LHD else HAND_LHD
    return None


def suffix_for_hand(hand: str) -> str:
    return "_to_rhd" if hand == HAND_RHD else "_to_lhd"


def signed_delta_for_target(hand: str, magnitude: float) -> float:
    return -abs(magnitude) if hand == HAND_RHD else abs(magnitude)


def generated_mesh_name(source_mesh: str, target_hand: str) -> str:
    return f"{source_mesh}{suffix_for_hand(target_hand)}"


def generated_part_name(source_part: str, target_hand: str) -> str:
    return f"{source_part}{suffix_for_hand(target_hand)}"


def generated_variant_part_name(source_part: str, target_hand: str, config_name: str) -> str:
    return f"{generated_part_name(source_part, target_hand)}__{safe_id(config_name)}"


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(fs_path(path))
    path.mkdir(parents=True, exist_ok=True)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def zip_member_path(value: str) -> Path:
    return Path(*[part for part in value.replace("\\", "/").split("/") if part])


def generated_dae_output_path(
    output_root: Path,
    output_vehicle_dir: Path,
    context: VehicleContext,
    dae_path: str,
) -> Path:
    source_rel = zip_member_path(dae_path)
    vehicle_rel = zip_member_path(context.vehicle_path)
    try:
        local_rel = source_rel.relative_to(vehicle_rel)
        target_parent = output_vehicle_dir / local_rel.parent
        return target_parent / f"{local_rel.stem}_handdrive{local_rel.suffix}"
    except ValueError:
        pass

    flattened_source = source_rel.with_suffix("")
    if (
        len(source_rel.parts) >= 2
        and source_rel.parts[0].lower() == "vehicles"
        and source_rel.parts[1].lower() == "common"
    ):
        flattened_source = Path("common", *source_rel.parts[2:]).with_suffix("")
    target_stem = safe_id("_".join(flattened_source.parts))
    return output_vehicle_dir / f"{target_stem}_handdrive{source_rel.suffix}"


def source_object_position(
    context: VehicleContext,
    object_id: str,
    config_name: str | None = None,
) -> tuple[float, float, float]:
    """Where a mesh sits, in the given trim when one is known.

    The DaeObject coordinate is only a representative across trims (see
    representative_mesh_positions), so build callers pass their config: a mesh
    declared by mutually exclusive parts sits somewhere different in each, and
    writing the representative into one trim's jbeam would misplace it."""
    if config_name is not None:
        resolved = resolved_mesh_positions_for_config(context, config_name).get(object_id)
        if resolved is not None:
            return resolved.position
    obj = context.objects[object_id]
    return (obj.x, obj.y, obj.z)


def target_object_position(
    context: VehicleContext,
    object_id: str,
    signed_delta: float,
    config_name: str | None = None,
) -> tuple[float, float, float]:
    x, y, z = source_object_position(context, object_id, config_name)
    return (x + signed_delta, y, z)


def mirrored_object_position(
    context: VehicleContext,
    object_id: str,
    config_name: str | None = None,
) -> tuple[float, float, float]:
    x, y, z = source_object_position(context, object_id, config_name)
    return (-x, y, z)


def format_inline_vector(key: str, values: tuple[float, float, float]) -> str:
    x, y, z = values
    return (
        f'"{key}":{{"x":{transform_helpers.format_num(x)},'
        f'"y":{transform_helpers.format_num(y)},"z":{transform_helpers.format_num(z)}}}'
    )


def vector_pattern(key: str) -> re.Pattern[str]:
    return re.compile(
        rf'"{re.escape(key)}"\s*:\s*\{{\s*"x"\s*:\s*(?P<x>{NUMBER_RE})\s*,'
        rf'\s*"y"\s*:\s*(?P<y>{NUMBER_RE})\s*,\s*"z"\s*:\s*(?P<z>{NUMBER_RE})\s*\}}'
    )


def replace_inline_vector(row: str, key: str, values: tuple[float, float, float]) -> str:
    return vector_pattern(key).sub(format_inline_vector(key, values), row, count=1)


def insert_inline_vector_near_key(line: str, preferred_key: str, replacement: str) -> str | None:
    match = re.search(rf'"{re.escape(preferred_key)}"\s*:', line)
    if match is None:
        return None
    object_start = line.rfind("{", 0, match.start())
    if object_start < 0:
        return None
    try:
        object_end = transform_helpers.find_matching(line, object_start, "{", "}")
    except ValueError:
        return None
    object_close = object_end - 1
    return line[:object_close] + f",{replacement}" + line[object_close:]


def replace_or_append_inline_vector(
    line: str,
    key: str,
    values: tuple[float, float, float],
    *,
    preferred_key: str | None = None,
) -> str:
    existing = vector_pattern(key)
    if existing.search(line):
        return existing.sub(format_inline_vector(key, values), line, count=1)

    replacement = format_inline_vector(key, values)
    if preferred_key is not None:
        updated = insert_inline_vector_near_key(line, preferred_key, replacement)
        if updated is not None:
            return updated

    insert_at = line.rfind("]")
    if insert_at < 0:
        raise RuntimeError(f"Could not append {key} to prop row: {line}")
    return line[:insert_at] + f", {{{replacement}}}" + line[insert_at:]


def transform_flexbody_row(
    row: str,
    action: str,
    delta_x: float = 0.0,
    inherited_options: Iterable[str] = (),
) -> str:
    if action == "translate":
        pos = vector_from_row(row, "pos")
        if pos is None:
            return row
        source_pos = pos_after_node_transforms(row, pos, inherited_options)
        target_pos = (source_pos[0] + delta_x, source_pos[1], source_pos[2])
        return replace_inline_vector(row, "pos", pos_before_node_transforms(row, target_pos, inherited_options))

    if action == "mirror":
        out = row
        pos = vector_from_row(out, "pos")
        if pos is not None:
            source_pos = pos_after_node_transforms(out, pos, inherited_options)
            target_pos = (-source_pos[0], source_pos[1], source_pos[2])
        else:
            target_pos = None
        rot = vector_from_row(out, "rot")
        if rot is not None:
            target_rot = (rot[0], -rot[1], -rot[2])
            if target_pos is not None:
                jbeam_pos, jbeam_rot = pos_rot_before_node_transforms(
                    out,
                    target_pos,
                    target_rot,
                    inherited_options,
                )
                out = replace_inline_vector(out, "pos", jbeam_pos)
                out = replace_inline_vector(out, "rot", jbeam_rot)
            else:
                out = replace_inline_vector(out, "rot", target_rot)
        elif target_pos is not None:
            out = replace_inline_vector(out, "pos", pos_before_node_transforms(out, target_pos, inherited_options))
        return out

    return row


def flexbody_row_can_carry_transform(row: str, action: str) -> bool:
    if action == "translate":
        return vector_from_row(row, "pos") is not None
    if action == "mirror":
        return vector_from_row(row, "pos") is not None or vector_from_row(row, "rot") is not None
    return False


def rewrite_flexbody_meshes_with_transforms(
    array_text: str,
    mesh_map: dict[str, str],
    row_transforms: dict[str, tuple[str, float]],
    inherited_options: Iterable[str] = (),
    shared_bake: SharedBakeContext | None = None,
) -> str:
    spans: list[tuple[int, int, str]] = []
    idx = 1 if array_text.startswith("[") else 0
    while idx < len(array_text):
        if array_text[idx] == "[":
            end = transform_helpers.find_matching(array_text, idx, "[", "]")
            spans.append((idx, end, array_text[idx:end]))
            idx = end
            continue
        idx += 1

    if not spans:
        return rewrite_flexbody_meshes(array_text, mesh_map)

    out: list[str] = []
    cursor = 0
    for start, end, row in spans:
        out.append(array_text[cursor:start])
        mesh = flexbody_row_mesh(row)
        new_row = row
        baked_mesh = None
        row_transform: tuple[str, float] | None = row_transforms.get(mesh) if mesh else None
        bake_transform_into_dae = True
        if row_transform is not None:
            bake_transform_into_dae = not flexbody_row_can_carry_transform(row, row_transform[0])
        if mesh and mesh in mesh_map and shared_bake is not None:
            baked_mesh = add_baked_shared_mesh(
                shared_bake,
                mesh,
                flexbody_row_source_matrix(row, inherited_options),
                bake_transform_into_dae,
            )
        if row_transform is not None and (baked_mesh is None or not bake_transform_into_dae):
            action, delta_x = row_transform
            new_row = transform_flexbody_row(new_row, action, delta_x, inherited_options)
        if baked_mesh is not None:
            new_row = re.sub(
                rf'(\[\s*)"{re.escape(mesh)}"(?=\s*(?:,|\[|\{{))',
                rf'\1"{baked_mesh}"',
                new_row,
                count=1,
            )
        elif mesh in mesh_map:
            new_row = re.sub(
                rf'(\[\s*)"{re.escape(mesh)}"(?=\s*(?:,|\[|\{{))',
                rf'\1"{mesh_map[mesh]}"',
                new_row,
                count=1,
            )
        out.append(new_row)
        cursor = end
    out.append(array_text[cursor:])
    return "".join(out)


def replace_or_append_prop_translation_global(
    line: str,
    values: tuple[float, float, float],
) -> str:
    existing_translation_property = vector_pattern("baseTranslationGlobal")
    if existing_translation_property.search(line):
        return existing_translation_property.sub(format_inline_vector("baseTranslationGlobal", values), line, count=1)
    existing_translation_property = vector_pattern("baseTranslation")
    if existing_translation_property.search(line):
        return existing_translation_property.sub(format_inline_vector("baseTranslationGlobal", values), line, count=1)
    return replace_or_append_inline_vector(line, "baseTranslationGlobal", values)


def replace_or_append_prop_rotation_global(
    line: str,
    values: tuple[float, float, float],
) -> str:
    return replace_or_append_inline_vector(
        line,
        "baseRotationGlobal",
        values,
        preferred_key="baseTranslationGlobal",
    )


def rewrite_flexbody_meshes(array_text: str, mesh_map: dict[str, str]) -> str:
    return transform_helpers.rewrite_flexbody_meshes(array_text, mesh_map)


def rewrite_prop_meshes_with_globals(
    array_text: str,
    mesh_map: dict[str, str],
    prop_global_positions: dict[str, tuple[float, float, float]],
    prop_row_transforms: dict[str, tuple[str, float]],
    node_positions: dict[str, tuple[float, float, float]],
    inherited_options: Iterable[str] = (),
    shared_bake: SharedBakeContext | None = None,
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    out_lines: list[str] = []
    for line in array_text.splitlines(keepends=True):
        line_ending = ""
        content = line
        if content.endswith("\r\n"):
            content, line_ending = content[:-2], "\r\n"
        elif content.endswith("\n"):
            content, line_ending = content[:-1], "\n"

        matched_old_mesh: str | None = None
        baked_mesh: str | None = None
        for old_mesh, new_mesh in sorted(mesh_map.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = rf'(\[\s*"((?:[^"\\]|\\.)*)"\s*(?:,\s*|\s+))"{re.escape(old_mesh)}"(?=\s*(?:,|"))'
            if re.search(pattern, content) is not None:
                matched_old_mesh = old_mesh
                if shared_bake is not None:
                    # Mirror bakes reflect across the frame's x-axis via
                    # D = R^T*S*R, so R must be the ENGINE's rest rotation:
                    # authored baseRotationGlobal, or the analytic engine
                    # model for rows without authored brg.
                    rest_rotation, _source = prop_rest_rotation_override(content, node_positions)
                    placement_matrix = prop_row_source_matrix(
                        content, node_positions, inherited_options, rest_rotation
                    )
                    if placement_matrix is not None:
                        baked_mesh = add_baked_shared_mesh(
                            shared_bake,
                            old_mesh,
                            placement_matrix,
                            False,
                            is_prop=True,
                        )
                replacement_mesh = baked_mesh or new_mesh
                content = re.sub(
                    pattern,
                    rf'\1"{replacement_mesh}"',
                    content,
                    count=1,
                )
                break

        row_position = None
        if matched_old_mesh in prop_row_transforms:
            pivot = (mesh_pivots or {}).get(matched_old_mesh)
            row_position = prop_row_pivot_position(content, node_positions, pivot, inherited_options)
        if matched_old_mesh in prop_row_transforms and row_position is not None:
            action, delta_x = prop_row_transforms[matched_old_mesh]
            if action == "translate":
                target_position = (row_position[0] + delta_x, row_position[1], row_position[2])
            elif action == "mirror":
                target_position = (-row_position[0], row_position[1], row_position[2])
            else:
                target_position = None
            if target_position is not None:
                jbeam_position = pos_before_node_transforms(content, target_position, inherited_options)
                content = replace_or_append_prop_translation_global(content, jbeam_position)
                if action == "mirror" and baked_mesh is None:
                    # Vehicle-local prop meshes carry the mirrored orientation via
                    # baseRotationGlobal. Baked shared copies instead have the frame-aligned
                    # reflection baked into the DAE (see baked_dae_matrix), so their rows
                    # must keep the original rotation fields untouched.
                    mirrored_rotation = mirrored_prop_global_rotation(content, node_positions)
                    if mirrored_rotation is not None:
                        _jbeam_position, jbeam_rotation = pos_rot_before_node_transforms(
                            content,
                            target_position,
                            mirrored_rotation,
                            inherited_options,
                        )
                        content = replace_or_append_prop_rotation_global(content, jbeam_rotation)
        elif matched_old_mesh in prop_row_transforms:
            pass
        elif matched_old_mesh in prop_global_positions:
            jbeam_position = pos_before_node_transforms(
                content,
                prop_global_positions[matched_old_mesh],
                inherited_options,
            )
            content = replace_or_append_prop_translation_global(
                content,
                jbeam_position,
            )
        out_lines.append(content + line_ending)
    return "".join(out_lines)


def swap_token_pair(value: str, left: str, right: str) -> str:
    left_marker = "\0LEFT_SIDE_TOKEN\0"
    right_marker = "\0RIGHT_SIDE_TOKEN\0"
    return value.replace(left, left_marker).replace(right, right_marker).replace(
        left_marker,
        right,
    ).replace(right_marker, left)


def mirror_lateral_node_id(value: str) -> str:
    token_pairs = (
        ("_FL", "_FR"),
        ("_FRONTLEFT", "_FRONTRIGHT"),
        ("_FrontLeft", "_FrontRight"),
        ("_frontLeft", "_frontRight"),
        ("_frontleft", "_frontright"),
        ("_RL", "_RR"),
        ("_REARLEFT", "_REARRIGHT"),
        ("_RearLeft", "_RearRight"),
        ("_rearLeft", "_rearRight"),
        ("_rearleft", "_rearright"),
        ("_LEFT", "_RIGHT"),
        ("_Left", "_Right"),
        ("_left", "_right"),
        ("_L", "_R"),
        ("_l", "_r"),
        ("-FL", "-FR"),
        ("-fl", "-fr"),
        ("-RL", "-RR"),
        ("-rl", "-rr"),
        ("-LEFT", "-RIGHT"),
        ("-Left", "-Right"),
        ("-left", "-right"),
        ("-L", "-R"),
        ("-l", "-r"),
        (".FL", ".FR"),
        (".fl", ".fr"),
        (".RL", ".RR"),
        (".rl", ".rr"),
        (".LEFT", ".RIGHT"),
        (".Left", ".Right"),
        (".left", ".right"),
        (".L", ".R"),
        (".l", ".r"),
    )
    for left, right in token_pairs:
        if left in value or right in value:
            return swap_token_pair(value, left, right)

    if value.endswith("ll"):
        return value[:-2] + "rr"
    if value.endswith("rr"):
        return value[:-2] + "ll"
    if value.endswith("l"):
        return value[:-1] + "r"
    if value.endswith("r"):
        return value[:-1] + "l"
    return value


def build_node_mirror_map(
    node_positions: dict[str, tuple[float, float, float]],
) -> dict[str, str]:
    mirror_map: dict[str, str] = {}
    items = list(node_positions.items())
    for node_id, (x, y, z) in items:
        if abs(x) < 1e-5:
            mirror_map[node_id] = node_id
            continue
        best: tuple[float, str] | None = None
        for candidate_id, (cx, cy, cz) in items:
            if candidate_id == node_id:
                continue
            same_side_penalty = 10.0 if x * cx > 0 and abs(x) > 0.02 and abs(cx) > 0.02 else 0.0
            score = same_side_penalty + abs(cx + x) * 4.0 + abs(cy - y) + abs(cz - z)
            if best is None or score < best[0]:
                best = (score, candidate_id)
        if best is not None and best[0] <= 0.18:
            mirror_map[node_id] = best[1]
    return mirror_map


def mirror_camera_reference(value: str, node_mirror_map: dict[str, str]) -> str:
    mapped = node_mirror_map.get(value)
    if mapped:
        return mapped
    return mirror_lateral_node_id(value)


def rewrite_internal_camera_line(
    content: str,
    node_mirror_map: dict[str, str],
) -> tuple[str, bool]:
    row_re = re.compile(
        rf'^(?P<prefix>\s*\[\s*"(?P<row_type>(?:[^"\\]|\\.)*)"\s*,\s*)'
        rf'(?P<x>{NUMBER_RE})'
        rf'(?P<rest>\s*,\s*{NUMBER_RE}\s*,\s*{NUMBER_RE}.*)$'
    )
    match = row_re.match(content)
    if match is None:
        return content, False

    x_value = -float(match.group("x"))
    if abs(x_value) < 1e-9:
        x_value = 0.0
    rest = match.group("rest")
    option_start = rest.find("{")
    if option_start >= 0:
        id_span = rest[:option_start]
        options = rest[option_start:]
    else:
        id_span = rest
        options = ""
    id_span = re.sub(
        r'"((?:[^"\\]|\\.)*)"',
        lambda item: f'"{mirror_camera_reference(item.group(1), node_mirror_map)}"',
        id_span,
    )
    return f"{match.group('prefix')}{transform_helpers.format_num(x_value)}{id_span}{options}", True


# The engine derives asymmetric first-person behavior (look-back direction,
# which side the head sticks out of the window) from the "driver"/"dash" row's
# rightHandCamera flag (lua/ge/extensions/core/cameraModes/driver.lua), so a
# mirror conversion must flip it alongside the x coordinate and node ids.
# Vanilla LHD/RHD pairs (bx, covet, miramar) differ by exactly these three edits.
CAMERA_HAND_FLAG_RE = re.compile(r'("rightHand(?:Camera|Door)"\s*:\s*)(true|false)')
# indent restricted to [ \t] so the match can't start on a masked comment line
# above the row and swallow its newline into the captured indent
CAMERA_DRIVER_ROW_RE = re.compile(r'^([ \t]*)\[[ \t]*"(?:dash|driver)"', re.MULTILINE)


def rewrite_internal_cameras(
    array_text: str,
    node_mirror_map: dict[str, str],
) -> str:
    out_lines: list[str] = []
    for line in array_text.splitlines(keepends=True):
        line_ending = ""
        content = line
        if content.endswith("\r\n"):
            content, line_ending = content[:-2], "\r\n"
        elif content.endswith("\n"):
            content, line_ending = content[:-1], "\n"
        rewritten, _changed = rewrite_internal_camera_line(content, node_mirror_map)
        out_lines.append(rewritten + line_ending)
    out = "".join(out_lines)
    masked = transform_helpers.mask_comments_preserve_offsets(out)
    flag_matches = list(CAMERA_HAND_FLAG_RE.finditer(masked))
    if flag_matches:
        for match in reversed(flag_matches):
            flipped = "false" if match.group(2) == "true" else "true"
            out = out[: match.start(2)] + flipped + out[match.end(2) :]
    else:
        driver_row = CAMERA_DRIVER_ROW_RE.search(masked)
        if driver_row is not None:
            indent = driver_row.group(1)
            newline = "\r\n" if "\r\n" in out else "\n"
            out = (
                out[: driver_row.start()]
                + f'{indent}{{"rightHandCamera":true}},{newline}'
                + out[driver_row.start() :]
            )
    return out


def part_has_transformable_internal_camera(
    part_body: str,
    node_mirror_map: dict[str, str],
) -> bool:
    cameras = transform_helpers.extract_named_array(part_body, "camerasInternal")
    if not cameras:
        return False
    return rewrite_internal_cameras(cameras, node_mirror_map) != cameras


def clone_part_for_target(
    part_body: str,
    source_part_id: str,
    target_hand: str,
    new_part_id: str | None,
    mesh_map: dict[str, str],
    flexbody_row_transforms: dict[str, tuple[str, float]],
    prop_global_positions: dict[str, tuple[float, float, float]],
    prop_row_transforms: dict[str, tuple[str, float]],
    node_positions: dict[str, tuple[float, float, float]],
    node_mirror_map: dict[str, str],
    inherited_options: Iterable[str] = (),
    shared_bake: SharedBakeContext | None = None,
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    new_part_id = new_part_id or generated_part_name(source_part_id, target_hand)
    out = transform_helpers.replace_first(part_body, f'"{source_part_id}"', f'"{new_part_id}"')
    out = transform_helpers.replace_array_region(
        out,
        "flexbodies",
        lambda text: rewrite_flexbody_meshes_with_transforms(
            text,
            mesh_map,
            flexbody_row_transforms,
            inherited_options,
            shared_bake,
        ),
    )
    out = transform_helpers.replace_array_region(
        out,
        "props",
        lambda text: rewrite_prop_meshes_with_globals(
            text,
            mesh_map,
            prop_global_positions,
            prop_row_transforms,
            node_positions,
            inherited_options,
            shared_bake,
            mesh_pivots,
        ),
    )
    out = transform_helpers.replace_array_region(
        out,
        "camerasInternal",
        lambda text: rewrite_internal_cameras(text, node_mirror_map),
    )
    out = re.sub(
        r'("name"\s*:\s*")([^"]*)(")',
        lambda match: f"{match.group(1)}{append_hand_label(match.group(2), target_hand)}{match.group(3)}",
        out,
        count=1,
    )
    return out


def generate_daes(
    context: VehicleContext,
    output_root: Path,
    output_vehicle_dir: Path,
    object_modes: dict[str, str],
    structural_sources: dict[str, str],
    target_hands: set[str],
    translate_magnitudes: dict[str, float],
    translated_prop_meshes: set[str],
    translated_flexbody_meshes: set[str],
    jbeam_positioned_flexbodies: set[str],
    baked_shared_specs: list[BakedMeshSpec],
    texture_flip_ids: set[str] | None = None,
) -> list[Path]:
    texture_flip_ids = texture_flip_ids or set()
    generated: list[Path] = []
    objects_by_dae: dict[tuple[Path, str], list[tuple[str, str]]] = {}
    for object_id in object_modes:
        source_id = structural_sources.get(object_id, object_id)
        source_obj = context.objects.get(source_id)
        if source_obj is None or not source_obj.dae_path:
            continue
        dae_source_zip = source_obj.dae_source_zip or context.source_zip
        objects_by_dae.setdefault((dae_source_zip, source_obj.dae_path), []).append((object_id, source_id))

    baked_by_dae: dict[tuple[Path, str], list[BakedMeshSpec]] = {}
    for spec in baked_shared_specs:
        source_obj = context.objects.get(spec.source_mesh)
        if source_obj is None or not source_obj.dae_path:
            continue
        dae_source_zip = source_obj.dae_source_zip or context.source_zip
        baked_by_dae.setdefault((dae_source_zip, source_obj.dae_path), []).append(spec)

    dae_keys = sorted(
        set(objects_by_dae) | set(baked_by_dae),
        key=lambda item: (str(item[0]).lower(), item[1].lower()),
    )
    for dae_source_zip, dae_path in dae_keys:
        object_pairs = objects_by_dae.get((dae_source_zip, dae_path), [])
        baked_specs = baked_by_dae.get((dae_source_zip, dae_path), [])
        tree = parse_dae(dae_source_zip, dae_path)
        root = tree.getroot()
        library_geometries = root.find("c:library_geometries", NS)
        library_visual_scenes = root.find("c:library_visual_scenes", NS)
        if library_geometries is None or library_visual_scenes is None:
            continue
        geometries_by_id = {
            geom.get("id"): geom
            for geom in library_geometries.findall("c:geometry", NS)
            if geom.get("id")
        }
        selected_nodes: list[ET.Element] = []
        generated_geometry: dict[str, ET.Element] = {}

        for object_id, source_id in sorted(object_pairs):
            mode = object_modes[object_id]
            source_obj = context.objects.get(source_id)
            source_node_id = source_obj.id if source_obj is not None else source_id
            source_node = find_dae_node(root, source_node_id)
            if source_node is None:
                continue
            for target_hand in sorted(target_hands):
                suffix = suffix_for_hand(target_hand)
                signed_delta = signed_delta_for_target(
                    target_hand,
                    translate_magnitudes.get(object_id, 0.0),
                )
                new_name = generated_mesh_name(object_id, target_hand)
                new_node = copy.deepcopy(source_node)
                new_node.set("id", new_name)
                new_node.set("name", new_name)

                matrix_elem = new_node.find("c:matrix", NS)
                parsed_matrix = None
                translate_delta = None
                if matrix_elem is not None and matrix_elem.text:
                    parsed_matrix = transform_helpers.parse_matrix(matrix_elem.text)
                    if mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
                        matrix_elem.text = transform_helpers.format_matrix(transform_helpers.mirror_matrix_x(parsed_matrix))
                    elif mode == MODE_TRANSLATE:
                        if object_id in translated_prop_meshes:
                            pass
                        elif (
                            object_id in translated_flexbody_meshes
                            and object_id in jbeam_positioned_flexbodies
                        ):
                            pass
                        elif (
                            object_id in translated_flexbody_meshes
                            and object_id not in jbeam_positioned_flexbodies
                        ):
                            translate_delta = transform_helpers.local_delta_for_world_translation(
                                parsed_matrix,
                                (signed_delta, 0.0, 0.0),
                            )
                        else:
                            matrix_elem.text = transform_helpers.format_matrix(
                                transform_helpers.translate_matrix_x(parsed_matrix, signed_delta)
                            )

                for inst in new_node.findall(".//c:instance_geometry", NS):
                    url = inst.get("url", "")
                    if not url.startswith("#"):
                        continue
                    old_geom_id = url[1:]
                    old_geom = geometries_by_id.get(old_geom_id)
                    if old_geom is None:
                        continue
                    new_geom_id = safe_id(f"{old_geom_id}{suffix}_{object_id}")
                    if new_geom_id not in generated_geometry:
                        if mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
                            generated_geometry[new_geom_id] = transform_helpers.mirrored_geometry(
                                old_geom,
                                new_geom_id,
                                flip_texture=object_id in texture_flip_ids,
                            )
                        elif (
                            mode == MODE_TRANSLATE
                            and object_id in translated_flexbody_meshes
                            and object_id not in jbeam_positioned_flexbodies
                        ):
                            if translate_delta is None:
                                raise RuntimeError(f"Missing translated geometry delta for {object_id}")
                            generated_geometry[new_geom_id] = transform_helpers.translated_geometry(
                                old_geom,
                                new_geom_id,
                                translate_delta,
                            )
                        else:
                            generated_geometry[new_geom_id] = transform_helpers.copied_geometry(old_geom, new_geom_id)
                    inst.set("url", f"#{new_geom_id}")
                    if inst.get("name"):
                        inst.set("name", new_name)

                selected_nodes.append(new_node)

        for spec in baked_specs:
            source_obj = context.objects.get(spec.source_mesh)
            if source_obj is None:
                continue
            source_node = find_dae_node(root, source_obj.id)
            if source_node is None:
                continue
            new_node = copy.deepcopy(source_node)
            new_node.set("id", spec.output_mesh)
            new_node.set("name", spec.output_mesh)

            matrix_elem = new_node.find("c:matrix", NS)
            if matrix_elem is not None and matrix_elem.text:
                source_node_matrix = transform_helpers.parse_matrix(matrix_elem.text)
                matrix_elem.text = transform_helpers.format_matrix(
                    baked_dae_matrix(source_node_matrix, spec, translate_magnitudes)
                )

            for inst in new_node.findall(".//c:instance_geometry", NS):
                url = inst.get("url", "")
                if not url.startswith("#"):
                    continue
                old_geom_id = url[1:]
                old_geom = geometries_by_id.get(old_geom_id)
                if old_geom is None:
                    continue
                new_geom_id = safe_id(f"{old_geom_id}_{spec.output_mesh}")
                if new_geom_id not in generated_geometry:
                    if spec.mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
                        generated_geometry[new_geom_id] = transform_helpers.mirrored_geometry(
                            old_geom,
                            new_geom_id,
                            flip_texture=spec.configured_mesh in texture_flip_ids,
                        )
                    else:
                        generated_geometry[new_geom_id] = transform_helpers.copied_geometry(old_geom, new_geom_id)
                inst.set("url", f"#{new_geom_id}")
                if inst.get("name"):
                    inst.set("name", spec.output_mesh)

            selected_nodes.append(new_node)

        if not selected_nodes:
            continue
        for child in list(library_geometries):
            library_geometries.remove(child)
        for geom in generated_geometry.values():
            library_geometries.append(geom)
        for visual_scene in library_visual_scenes.findall("c:visual_scene", NS):
            for child in list(visual_scene):
                visual_scene.remove(child)
            for node in selected_nodes:
                visual_scene.append(node)

        target = generated_dae_output_path(output_root, output_vehicle_dir, context, dae_path)
        write_xml_tree(tree, target)
        generated.append(target)

    return generated


def variant_output_name(config_name: str, target_hand: str) -> str:
    suffix = "_rhd" if target_hand == HAND_RHD else "_lhd"
    if config_name.lower().endswith(suffix):
        return config_name
    return f"{config_name}{suffix}"


def original_plate_output_name(config_name: str) -> str:
    return config_name if config_name.lower().endswith("_plates") else f"{config_name}_plates"


def append_hand_label(name: object, target_hand: str) -> str:
    text = str(name or "").strip()
    if not text:
        return target_hand
    if re.search(rf"(?:\s|\(){re.escape(target_hand)}\)?$", text, re.IGNORECASE):
        return text
    return f"{text} {target_hand}"


def write_generated_jbeam_and_configs(
    context: VehicleContext,
    output_vehicle_dir: Path,
    conversion: dict[str, object],
    object_modes: dict[str, str],
    structural_sources: dict[str, str],
    node_mirror_map: dict[str, str],
    variant_targets: dict[str, str],
    translate_magnitudes: dict[str, float],
    translated_prop_meshes: set[str],
    translated_flexbody_meshes: set[str],
    mirrored_prop_meshes: set[str],
    structural_prop_meshes: set[str],
    baked_shared_specs: list[BakedMeshSpec],
) -> list[str]:
    cloned_bodies: list[str] = []
    cloned_part_ids: set[str] = set()
    generated_configs: list[str] = []

    for config_name, target_hand in sorted(variant_targets.items()):
        variant = context.variants[config_name]
        pc = load_pc(context.source_zip, variant.pc_path)
        selected = selected_parts_for_config(context, config_name)
        selected_node_positions = selected_node_positions_for_config(context, config_name)
        prop_node_positions = dict(context.node_positions)
        prop_node_positions.update(selected_node_positions)
        selected_by_slot = selected.get("selected_by_slot", {})
        part_slot_options = selected.get("part_slot_options", {})
        slot_updates: dict[str, str] = {}
        main_update: str | None = None
        suffix = suffix_for_hand(target_hand)

        for source_part_id in sorted(selected["parts"]):
            found = part_body_for_context(context, str(source_part_id))
            if found is None:
                continue
            part_body, _filename = found
            part_meshes = transform_helpers.extract_part_mesh_names(part_body)
            mesh_hits = sorted(mesh for mesh in part_meshes if mesh in object_modes)
            camera_hit = part_has_transformable_internal_camera(part_body, node_mirror_map)
            if not mesh_hits and not camera_hit:
                continue

            new_part_id = generated_variant_part_name(str(source_part_id), target_hand, config_name)
            if str(source_part_id) == selected["main_part"]:
                main_update = new_part_id
            selected_slot_types = []
            if isinstance(selected_by_slot, dict):
                selected_slot_types = [
                    str(slot_type)
                    for slot_type, part_id in selected_by_slot.items()
                    if slot_type != "main" and str(part_id) == str(source_part_id)
                ]
            slot_types = selected_slot_types or transform_helpers.extract_part_slot_types(part_body)
            for slot_type in slot_types:
                slot_updates[slot_type] = new_part_id

            if new_part_id in cloned_part_ids:
                continue
            cloned_part_ids.add(new_part_id)

            mesh_map = {}
            for mesh in mesh_hits:
                source_mesh = structural_sources.get(mesh, mesh)
                source_obj = context.objects.get(source_mesh)
                mesh_map[mesh] = f"{mesh}{suffix}" if source_obj is not None and source_obj.dae_path else mesh
            flexbody_row_transforms: dict[str, tuple[str, float]] = {}
            for mesh in mesh_hits:
                if object_modes.get(mesh) == MODE_TRANSLATE and mesh in translated_flexbody_meshes:
                    flexbody_row_transforms[mesh] = (
                        "translate",
                        signed_delta_for_target(target_hand, translate_magnitudes.get(mesh, 0.0)),
                    )
                elif object_modes.get(mesh) in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
                    # Structural rows must carry the mirror in the jbeam pos/rot
                    # like plain mirror rows: the engine drops the DAE node
                    # translation for flexbodies, so a side-swap baked into the
                    # copy's node matrix never reaches the screen.
                    flexbody_row_transforms[mesh] = ("mirror", 0.0)
            prop_row_transforms: dict[str, tuple[str, float]] = {}
            for mesh in mesh_hits:
                if object_modes.get(mesh) == MODE_TRANSLATE and mesh in translated_prop_meshes:
                    prop_row_transforms[mesh] = (
                        "translate",
                        signed_delta_for_target(target_hand, translate_magnitudes.get(mesh, 0.0)),
                    )
                elif object_modes.get(mesh) == MODE_MIRROR and mesh in mirrored_prop_meshes:
                    prop_row_transforms[mesh] = ("mirror", 0.0)
            # config_name: these positions are written into ONE trim's jbeam,
            # so they must be that trim's, not the cross-trim representative.
            prop_globals = {
                mesh: target_object_position(
                    context,
                    mesh,
                    signed_delta_for_target(target_hand, translate_magnitudes.get(mesh, 0.0)),
                    config_name,
                )
                for mesh in mesh_hits
                if mesh in translated_prop_meshes and object_modes.get(mesh) == MODE_TRANSLATE
            }
            prop_globals.update(
                {
                    mesh: mirrored_object_position(context, mesh, config_name)
                    for mesh in mesh_hits
                    if mesh in mirrored_prop_meshes and object_modes.get(mesh) == MODE_MIRROR
                }
            )
            # Structural-mirror props reach rewrite_prop_meshes_with_globals
            # through prop_globals rather than prop_row_transforms, so this is
            # the one build path that positions a mesh from a stored coordinate
            # rather than the row it is rewriting.
            prop_globals.update(
                {
                    mesh: mirrored_object_position(
                        context, structural_sources[mesh], config_name
                    )
                    for mesh in mesh_hits
                    if mesh in structural_prop_meshes
                    and object_modes.get(mesh) == MODE_MIRROR_STRUCTURAL
                    and mesh in structural_sources
                }
            )
            inherited_options = ()
            if isinstance(part_slot_options, dict):
                raw_options = part_slot_options.get(str(source_part_id), ())
                if isinstance(raw_options, (list, tuple)):
                    inherited_options = tuple(str(item) for item in raw_options if item)
            shared_bake = SharedBakeContext(
                context=context,
                config_name=config_name,
                target_hand=target_hand,
                source_part_id=str(source_part_id),
                object_modes=object_modes,
                structural_sources=structural_sources,
                translate_magnitudes=translate_magnitudes,
                baked_specs=baked_shared_specs,
            )
            cloned_bodies.append(
                clone_part_for_target(
                    part_body,
                    str(source_part_id),
                    target_hand,
                    new_part_id,
                    mesh_map,
                    flexbody_row_transforms,
                    prop_globals,
                    prop_row_transforms,
                    prop_node_positions,
                    node_mirror_map,
                    inherited_options,
                    shared_bake,
                    context.mesh_pivots,
                )
            )

        if main_update:
            pc["mainPartName"] = main_update
        parts = dict(pc.get("parts", {}))
        parts.update(slot_updates)
        pc["parts"] = parts
        output_config = variant_output_name(config_name, target_hand)
        pc["licenseName"] = append_hand_label(pc.get("licenseName") or context.vehicle_id, target_hand)
        output_vehicle_dir.mkdir(parents=True, exist_ok=True)
        write_text_file(output_vehicle_dir / f"{output_config}.pc", json.dumps(pc, indent=2), encoding="utf-8")

        info = {}
        if variant.info_path:
            try:
                info = load_info(context.source_zip, variant.info_path)
            except Exception:
                info = {}
        existing_name = str(info.get("Configuration") or info.get("Name") or variant.display_name)
        existing_description = info.get("Description") or info.get("description") or ""
        converted_name = append_hand_label(existing_name, target_hand)
        info["Configuration"] = converted_name
        info["Name"] = converted_name
        info["Description"] = converted_description(existing_description, target_hand)
        info["Config Type"] = "Custom"
        info["Source"] = conversion_source_name(context)
        write_text_file(
            output_vehicle_dir / f"info_{output_config}.json",
            json.dumps(info, indent=2),
            encoding="utf-8",
        )
        write_mirrored_preview(context, output_vehicle_dir, config_name, output_config)
        generated_configs.append(output_config)

    if cloned_bodies:
        jbeam_dir = output_vehicle_dir / "jbeam"
        jbeam_dir.mkdir(parents=True, exist_ok=True)
        contents = textwrap.dedent(
            f"""\
            {{
            // Generated visual hand-drive conversion parts.
            // Source: {context.source_zip.name}
            {','.join(cloned_bodies)}
            }}
            """
        )
        write_text_file(jbeam_dir / "handdrive_visual_conversion.jbeam", contents, encoding="utf-8")
    return generated_configs


def write_original_plate_configs(
    context: VehicleContext,
    output_vehicle_dir: Path,
    conversion: dict[str, object],
    config_names: Iterable[str],
) -> list[str]:
    """Copy stock trims as new configs for the plates-only build path."""
    generated: list[str] = []
    output_vehicle_dir.mkdir(parents=True, exist_ok=True)
    for config_name in sorted(set(config_names)):
        variant = context.variants[config_name]
        pc = load_pc(context.source_zip, variant.pc_path)
        output_config = original_plate_output_name(config_name)
        write_text_file(output_vehicle_dir / f"{output_config}.pc", json.dumps(pc, indent=2), encoding="utf-8")

        info: dict[str, object] = {}
        if variant.info_path:
            try:
                info = load_info(context.source_zip, variant.info_path)
            except Exception:
                info = {}
        existing_name = str(info.get("Configuration") or info.get("Name") or variant.display_name)
        plates_name = existing_name if existing_name.lower().endswith(" plates") else f"{existing_name} Plates"
        info["Configuration"] = plates_name
        info["Name"] = plates_name
        description = str(info.get("Description") or info.get("description") or "").strip()
        info["Description"] = f"{description} - BeamXP plate configuration" if description else "BeamXP plate configuration"
        info["Config Type"] = "Custom"
        info["Source"] = conversion_source_name(context)
        write_text_file(output_vehicle_dir / f"info_{output_config}.json", json.dumps(info, indent=2), encoding="utf-8")
        write_stock_preview(context, output_vehicle_dir, config_name, output_config)
        generated.append(output_config)
    return generated


def variant_target_hand(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str | None:
    variants = conversion.get("variants", {})
    settings = variants.get(config_name) if isinstance(variants, dict) else None
    if variant_build_mode(settings) not in {BUILD_CONVERTED, BUILD_BOTH}:
        return None
    return target_hand_for(effective_source_hand(context, conversion, config_name), ACTION_OPPOSITE)


def output_config_sources(
    context: VehicleContext,
    conversion: dict[str, object],
) -> dict[str, str]:
    plans, _skipped = selected_output_plans(context, conversion)
    return {str(plan["output"]): str(plan["source"]) for plan in plans}


def load_beamng_json_file(path: Path) -> dict[str, object]:
    return parse_beamng_json(path.read_text(encoding="utf-8", errors="replace"), label=str(path))


def prop_row_world_matrix(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    pivot: tuple[float, float, float] | None,
    inherited_options: Iterable[str] = (),
    rotation_override: list[list[float]] | None = None,
) -> list[list[float]] | None:
    """Affine map from DAE-world coordinates to vehicle space for a prop row
    at rest: W = T(anchor) * R * T(-pivot), per the engine model verified
    against in-game dumps.

    rotation_override supplies the resolved engine rest rotation (authored
    baseRotationGlobal or analytic engine model)."""
    anchor = prop_row_pivot_position(row, node_positions, pivot, inherited_options)
    if anchor is None:
        return None
    rotation = rotation_override
    if rotation is None:
        rotation = prop_row_global_rotation_matrix(row, node_positions)
    matrix = matrix4_with_rotation_translation(rotation, anchor)
    t = pivot or (0.0, 0.0, 0.0)
    return multiply_matrix(matrix, translation_matrix((-t[0], -t[1], -t[2])))


def matrix4_flat(matrix: list[list[float]]) -> list[float]:
    return [matrix[row][col] for row in range(4) for col in range(4)]


def preview_node_names(obj: DaeObject) -> list[str]:
    node_names = [obj.id]
    for alias in (obj.name, obj.id.strip("_ ")):
        if alias and alias not in node_names:
            node_names.append(alias)
    return node_names


def extract_preview_dae(zip_path: Path, dae_path: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{Path(zip_path).stem}__{dae_path}")
    if not stem.lower().endswith(".dae"):
        stem += ".dae"
    target = cache_dir / stem
    try:
        if target.exists() and target.stat().st_mtime_ns >= Path(zip_path).stat().st_mtime_ns:
            return target
    except OSError:
        pass
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(dae_path)
    target.write_bytes(data)
    return target


def output_vehicle_preview_payload(
    context: VehicleContext,
    conversion: dict[str, object],
    output_name: str,
    output_root: Path,
    generated_daes: Iterable[Path],
    run_dir: Path,
) -> dict[str, object]:
    """Blender preview payload for one generated output config.

    This follows the validated build artifacts instead of re-simulating the
    conversion directly from the source vehicle: it reads the selected output
    ``.pc``, resolves that part tree against source + generated JBeam, and
    instances the exact mesh names referenced by those final rows.
    """
    output_sources = output_config_sources(context, conversion)
    config_name = output_sources.get(output_name)
    if config_name is None:
        raise RuntimeError(f"Unknown generated output config {output_name!r}")

    output_vehicle_dir = output_root / context.vehicle_path
    pc_path = output_vehicle_dir / f"{output_name}.pc"
    if not pc_path.exists():
        raise RuntimeError(f"Generated output config does not exist: {pc_path}")

    output_jbeam_texts: dict[str, str] = {}
    if output_vehicle_dir.exists():
        for path in sorted(output_vehicle_dir.rglob("*.jbeam")):
            output_jbeam_texts[str(path.relative_to(output_vehicle_dir)).replace("\\", "/")] = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

    combined_jbeam_texts = dict(context.jbeam_texts)
    combined_jbeam_texts.update({f"generated/{name}": text for name, text in output_jbeam_texts.items()})
    combined_part_index = dict(context.part_body_index)
    combined_part_index.update(build_part_body_index(output_jbeam_texts))

    output_objects: dict[str, DaeObject] = {}
    output_pivots: dict[str, tuple[float, float, float]] = {}
    for dae_path in sorted((Path(path) for path in generated_daes), key=lambda item: str(item).lower()):
        if not dae_path.exists():
            continue
        for alias, obj in list_dae_objects_for_path(dae_path).items():
            output_objects.setdefault(alias, obj)
            output_pivots.setdefault(alias, (obj.x, obj.y, obj.z))

    dae_index: dict[tuple[str, str, str], int] = {}
    dae_entries: list[dict[str, object]] = []
    instances: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    used_output_paths = {str(Path(path)) for path in generated_daes}
    cache_dir = context.project_dir / "blender_preview" / "dae_cache"

    def dae_ref(obj: DaeObject) -> int:
        if obj.dae_source_zip is None:
            path = Path(obj.dae_path)
            key = ("file", str(path), "")
            entry = {"path": str(path), "dae_path": str(path)}
        else:
            zip_path = obj.dae_source_zip or context.source_zip
            key = ("zip", str(zip_path), obj.dae_path)
            entry = {
                "zip": str(zip_path),
                "dae_path": obj.dae_path,
                "path": str(extract_preview_dae(zip_path, obj.dae_path, cache_dir)),
            }
        index = dae_index.get(key)
        if index is None:
            index = len(dae_entries)
            dae_index[key] = index
            dae_entries.append(entry)
        return index

    selected = resolve_selected_parts(
        load_beamng_json_file(pc_path),
        combined_jbeam_texts,
        vehicle_id=context.vehicle_id,
        part_body_index=combined_part_index,
    )
    selected_nodes = selected_node_positions_for_parts(
        selected,
        combined_jbeam_texts,
        combined_part_index,
    )
    node_positions = build_node_position_index(combined_jbeam_texts)
    node_positions.update(selected_nodes)
    part_slot_options = selected.get("part_slot_options", {})

    rotation_counts: dict[str, int] = {}

    for part_id in sorted(str(part) for part in selected.get("parts", set())):
        found = find_part_body(part_id, combined_jbeam_texts, combined_part_index)
        if found is None:
            skipped.setdefault(part_id, "part body not found")
            continue
        part_body, _filename = found
        raw_options = part_slot_options.get(part_id, ()) if isinstance(part_slot_options, dict) else ()
        opts = tuple(str(item) for item in raw_options if item) if isinstance(raw_options, (list, tuple)) else ()
        for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
            array_text = transform_helpers.extract_named_array(part_body, array_key)
            if not array_text:
                continue
            for row in iter_active_top_level_rows(array_text):
                mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                if not mesh or mesh in ("SPOTLIGHT", "POINTLIGHT"):
                    continue
                obj = output_objects.get(mesh) or context.objects.get(mesh)
                if obj is None or not obj.dae_path:
                    skipped.setdefault(mesh, "no DAE geometry indexed")
                    continue
                if kind == "prop" and not prop_row_nodes_present(row, selected_nodes):
                    # the engine only spawns a prop when its idRef/idX/idY nodes
                    # exist in the assembled config; other rows (e.g. the manual
                    # vs sequential handbrake mounts) stay dormant
                    skipped.setdefault(mesh, "inactive row (prop nodes not in this config)")
                    continue
                rotation_source = None
                if kind == "flex":
                    world = flexbody_row_source_matrix(row, opts)
                else:
                    pivot = output_pivots.get(mesh) or context.mesh_pivots.get(mesh)
                    rotation_override, rotation_source = prop_rest_rotation_override(row, node_positions)
                    rotation_counts[rotation_source] = rotation_counts.get(rotation_source, 0) + 1
                    world = prop_row_world_matrix(row, node_positions, pivot, opts, rotation_override)
                if world is None:
                    skipped.setdefault(mesh, "placement unresolved (inactive row?)")
                    continue
                is_generated = obj.dae_source_zip is None and str(Path(obj.dae_path)) in used_output_paths
                instances.append(
                    {
                        "dae": dae_ref(obj),
                        "node": obj.id,
                        "node_names": preview_node_names(obj),
                        "mesh": mesh,
                        "part": part_id,
                        "kind": kind,
                        "mode": "output" if is_generated else MODE_SKIP,
                        "matrix": matrix4_flat(world),
                        # Our own generated geometry always means what its node
                        # says; source geometry follows the row-authorship rule
                        # (see flexbody_row_needs_node_translation).
                        "keep_node_translation": is_generated
                        or (kind == "flex" and flexbody_row_needs_node_translation(context, mesh)),
                        **({"rotation_source": rotation_source} if rotation_source else {}),
                    }
                )

    return {
        "preview_kind": "generated_output",
        "vehicle_id": context.vehicle_id,
        "config_name": config_name,
        "output_name": output_name,
        "target_hand": None
        if output_name == original_plate_output_name(config_name)
        else variant_target_hand(context, conversion, config_name),
        "output_root": str(output_root),
        "dae_files": dae_entries,
        "instances": instances,
        "skipped_meshes": skipped,
        "rotation_calibration": rotation_counts,
        "show_unchanged": True,
    }


def full_vehicle_preview_payload(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
    run_dir: Path,
    extra_meshes: Iterable[str] = (),
) -> dict[str, object]:
    """Full-vehicle Blender preview of one config after conversion.

    Every flexbody/prop row of the resolved part tree gets a final world
    matrix computed with the same engine-verified functions the build uses;
    geometry is referenced from the ORIGINAL DAE files by node name, so the
    preview needs no build output and no generated meshes. Mirrored rows use
    negative-determinant matrices (fine for preview rendering).

    extra_meshes are object ids NOT used by this config that should still be
    included (the GUI passes selected-but-inactive parts so they can be shown
    temporarily); their instances carry \"extra\": True."""
    if config_name not in context.variants:
        raise RuntimeError(f"Unknown config {config_name!r}")
    target_hand = variant_target_hand(context, conversion, config_name)
    object_modes = fallback_structural_part_modes(
        context,
        conversion,
        active_part_modes(conversion),
        selected_configs=(config_name,),
    )
    structural = structural_mirror_sources(context, conversion, object_modes)
    preview_pc, generated_plate_parts = plate_generator.preview_pc_with_plate_parts(
        context,
        conversion,
        config_name,
    )
    preview_part_index = dict(context.part_body_index)
    preview_part_index.update(
        {
            part_id: (body, "bhdc_preview_licenseplates.jbeam")
            for part_id, body in generated_plate_parts.items()
        }
    )
    selected = resolve_selected_parts(
        preview_pc,
        context.jbeam_texts,
        vehicle_id=context.vehicle_id,
        part_body_index=preview_part_index,
    )
    part_slot_options = selected.get("part_slot_options", {})
    selected_nodes = selected_node_positions_for_config(context, config_name)
    node_positions = dict(context.node_positions)
    node_positions.update(selected_nodes)
    mirror = mirror_x_matrix4()
    convertible = {MODE_TRANSLATE, MODE_MIRROR, MODE_MIRROR_STRUCTURAL}
    rotation_counts: dict[str, int] = {}

    dae_index: dict[tuple[str, str], int] = {}
    dae_entries: list[dict[str, object]] = []
    instances: list[dict[str, object]] = []
    skipped: dict[str, str] = {}

    def dae_ref(obj: DaeObject) -> int:
        zip_path = obj.dae_source_zip or context.source_zip
        key = (str(zip_path), obj.dae_path)
        index = dae_index.get(key)
        if index is None:
            index = len(dae_entries)
            dae_index[key] = index
            dae_entries.append({"zip": str(zip_path), "dae_path": obj.dae_path})
        return index

    def final_matrix(mesh: str, mode: str, world: list[list[float]]) -> list[list[float]]:
        if target_hand is None or mode not in convertible:
            return world
        if mode == MODE_TRANSLATE:
            delta = signed_delta_for_target(
                target_hand,
                part_translate_magnitude(context, conversion, mesh),
            )
            return multiply_matrix(translation_matrix((delta, 0.0, 0.0)), world)
        return multiply_matrix(mirror, world)

    def preview_part_array(part_id: str, array_key: str) -> str | None:
        body = generated_plate_parts.get(part_id)
        if body is not None:
            return transform_helpers.extract_named_array(body, array_key)
        return part_named_array_for_context(context, part_id, array_key)

    for part_id in sorted(str(part) for part in selected.get("parts", set())):
        raw_options = part_slot_options.get(part_id, ()) if isinstance(part_slot_options, dict) else ()
        opts = tuple(str(item) for item in raw_options if item) if isinstance(raw_options, (list, tuple)) else ()
        for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
            array_text = preview_part_array(part_id, array_key)
            if not array_text:
                continue
            for row in iter_active_top_level_rows(array_text):
                mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                if not mesh or mesh in ("SPOTLIGHT", "POINTLIGHT"):
                    continue
                mode = object_modes.get(mesh, MODE_SKIP)
                geometry_mesh = structural.get(mesh, mesh) if mode == MODE_MIRROR_STRUCTURAL else mesh
                obj = context.objects.get(geometry_mesh)
                if obj is None or not obj.dae_path:
                    skipped.setdefault(mesh, "no DAE geometry indexed")
                    continue
                if kind == "prop" and not prop_row_nodes_present(row, selected_nodes):
                    skipped.setdefault(mesh, "inactive row (prop nodes not in this config)")
                    continue
                if kind == "flex":
                    world = flexbody_row_source_matrix(row, opts)
                else:
                    pivot = context.mesh_pivots.get(mesh)
                    rotation_override, rotation_source = prop_rest_rotation_override(row, node_positions)
                    rotation_counts[rotation_source] = rotation_counts.get(rotation_source, 0) + 1
                    world = prop_row_world_matrix(row, node_positions, pivot, opts, rotation_override)
                if world is None:
                    # matches the engine: prop rows whose reference nodes are
                    # absent from this config never render
                    skipped.setdefault(mesh, "placement unresolved (inactive row?)")
                    continue
                if math.hypot(world[0][3], world[1][3], world[2][3]) > PREVIEW_FAR_LIMIT:
                    skipped.setdefault(mesh, "placed far outside the vehicle (hidden by its jbeam)")
                    continue
                instances.append(
                    {
                        "dae": dae_ref(obj),
                        "node": obj.id,
                        "node_names": preview_node_names(obj),
                        "mesh": mesh,
                        "part": part_id,
                        "kind": kind,
                        "mode": mode if target_hand is not None else MODE_SKIP,
                        "matrix": matrix4_flat(final_matrix(mesh, mode, world)),
                        "stock_matrix": matrix4_flat(world),
                        "keep_node_translation": (
                            kind != "flex" or flexbody_row_needs_node_translation(context, geometry_mesh)
                        ),
                    }
                )

    # Temporarily-shown parts that are NOT in this config's part tree: find
    # each mesh's flexbody/prop row in any part of the vehicle and place it
    # with the same helpers, falling back to its indexed position when the
    # row cannot be resolved outside its own config.
    extra_wanted = {str(mesh) for mesh in extra_meshes or ()}
    extra_wanted -= {str(inst["mesh"]) for inst in instances}
    extra_rows: dict[str, tuple[str, str, str]] = {}
    if extra_wanted:
        for extra_part_id in context.part_body_index:
            if len(extra_rows) == len(extra_wanted):
                break
            for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
                array_text = part_named_array_for_context(context, extra_part_id, array_key)
                if not array_text:
                    continue
                for row in iter_active_top_level_rows(array_text):
                    mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                    if mesh in extra_wanted and mesh not in extra_rows:
                        extra_rows[mesh] = (extra_part_id, kind, row)
    for mesh in sorted(extra_wanted):
        mode = object_modes.get(mesh, MODE_SKIP)
        geometry_mesh = structural.get(mesh, mesh) if mode == MODE_MIRROR_STRUCTURAL else mesh
        obj = context.objects.get(geometry_mesh)
        if obj is None or not obj.dae_path:
            skipped.setdefault(mesh, "no DAE geometry indexed")
            continue
        extra_part_id, kind, row = extra_rows.get(mesh, ("", "flex", ""))
        world = None
        if row:
            if kind == "flex":
                world = flexbody_row_source_matrix(row)
            else:
                rotation_override, _source = prop_rest_rotation_override(row, node_positions)
                world = prop_row_world_matrix(
                    row, node_positions, context.mesh_pivots.get(mesh), (), rotation_override
                )
        if world is None:
            world = translation_matrix(context.mesh_pivots.get(mesh) or (obj.x, obj.y, obj.z))
        if math.hypot(world[0][3], world[1][3], world[2][3]) > PREVIEW_FAR_LIMIT:
            skipped.setdefault(mesh, "placed far outside the vehicle (hidden by its jbeam)")
            continue
        instances.append(
            {
                "dae": dae_ref(obj),
                "node": obj.id,
                "node_names": preview_node_names(obj),
                "mesh": mesh,
                "part": extra_part_id,
                "kind": kind,
                "mode": mode if target_hand is not None else MODE_SKIP,
                "matrix": matrix4_flat(final_matrix(mesh, mode, world)),
                "stock_matrix": matrix4_flat(world),
                "keep_node_translation": (
                    kind != "flex" or flexbody_row_needs_node_translation(context, geometry_mesh)
                ),
                "extra": True,
            }
        )

    cache_dir = context.project_dir / "blender_preview" / "dae_cache"
    for entry in dae_entries:
        entry["path"] = str(extract_preview_dae(Path(str(entry["zip"])), str(entry["dae_path"]), cache_dir))

    output_name = config_name if target_hand is None else variant_output_name(config_name, target_hand)
    return {
        "vehicle_id": context.vehicle_id,
        "config_name": config_name,
        "output_name": output_name,
        "target_hand": target_hand,
        "dae_files": dae_entries,
        "instances": instances,
        "skipped_meshes": skipped,
        "rotation_calibration": rotation_counts,
    }


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


def package_name_for_context(context: VehicleContext) -> str:
    return f"{context.source_zip.stem}_XP_conversion.zip"


def write_mod_info(root: Path, context: VehicleContext) -> None:
    mod_info = root / "mod_info"
    mod_info.mkdir(parents=True, exist_ok=True)
    source_name = conversion_source_name(context)
    info = {
        "name": f"{context.vehicle_id} BeamXP Conversion",
        "version": "0.2.1",
        "authors": source_name,
        "description": (
            f"Generated BeamXP handedness and/or plate configuration overlay for {context.vehicle_id}. "
            f"Depends on {context.source_zip.name}."
        ),
        "source": source_name,
    }
    write_text_file(mod_info / "info.json", json.dumps(info, indent=2), encoding="utf-8")


def selected_variant_targets(
    context: VehicleContext,
    conversion: dict[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    targets: dict[str, str] = {}
    skipped: dict[str, str] = {}
    variants = conversion.get("variants", {})
    if not isinstance(variants, dict):
        return targets, skipped
    for config_name, settings in variants.items():
        if config_name not in context.variants or not isinstance(settings, dict):
            continue
        if variant_build_mode(settings) not in {BUILD_CONVERTED, BUILD_BOTH}:
            continue
        source_hand = effective_source_hand(context, conversion, config_name)
        target = target_hand_for(source_hand, ACTION_OPPOSITE)
        if target is None:
            skipped[config_name] = f"No opposite target for source hand {source_hand}"
        else:
            targets[config_name] = target
    return targets, skipped


def selected_output_plans(
    context: VehicleContext,
    conversion: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Expand each trim row into zero, one, or two generated configs."""
    targets, skipped = selected_variant_targets(context, conversion)
    plans: list[dict[str, object]] = []
    variants = conversion.get("variants", {})
    if not isinstance(variants, dict):
        return plans, skipped
    for config_name, settings in sorted(variants.items()):
        if config_name not in context.variants or not isinstance(settings, dict):
            continue
        mode = variant_build_mode(settings)
        if mode in {BUILD_CONVERTED, BUILD_BOTH} and config_name in targets:
            target = targets[config_name]
            plans.append({
                "source": config_name,
                "kind": BUILD_CONVERTED,
                "targetHand": target,
                "output": variant_output_name(config_name, target),
            })
        if mode in {BUILD_ORIGINAL, BUILD_BOTH}:
            plans.append({
                "source": config_name,
                "kind": BUILD_ORIGINAL,
                "targetHand": None,
                "output": original_plate_output_name(config_name),
            })
    return plans, skipped


def build_batch(
    context: VehicleContext,
    conversion: dict[str, object],
    *,
    write_zip: bool = True,
    install: bool = False,
    mods_folder: Path | None = None,
) -> BuildResult:
    output_plans, skipped = selected_output_plans(context, conversion)
    if not output_plans:
        raise RuntimeError("No trim outputs are selected")
    variant_targets, skipped = selected_variant_targets(context, conversion)
    original_configs = [
        str(plan["source"])
        for plan in output_plans
        if plan["kind"] == BUILD_ORIGINAL
    ]
    no_op_originals = [
        config_name
        for config_name in original_configs
        if not plate_generator.variant_has_plate_changes(conversion, config_name, context)
    ]
    if no_op_originals:
        raise RuntimeError(
            "Plates Only output has no plate changes for: "
            + ", ".join(no_op_originals[:8])
            + ("..." if len(no_op_originals) > 8 else "")
            + ". Choose a plate design, a different physical plate, or None for at least one side."
        )

    object_modes: dict[str, str] = {}
    structural_sources: dict[str, str] = {}
    node_mirror_map: dict[str, str] = {}
    translated_prop_meshes: set[str] = set()
    mirrored_prop_meshes: set[str] = set()
    structural_prop_meshes: set[str] = set()
    translated_flexbody_meshes: set[str] = set()
    translate_magnitudes: dict[str, float] = {}
    texture_flip_ids: set[str] = set()
    if variant_targets:
        object_modes = active_part_modes(conversion)
        if not object_modes:
            raise RuntimeError("Converted outputs require at least one Mirror Aesthetic, Mirror Structural, or Translate part")
        selected_configs = sorted(variant_targets)
        flexbody_meshes, prop_meshes, all_meshes = selected_mesh_roles(context, selected_configs)
        if all_meshes:
            object_modes = {mesh: mode for mesh, mode in object_modes.items() if mesh in all_meshes}
        if not object_modes:
            raise RuntimeError(
                "No Mirror Aesthetic, Mirror Structural, or Translate parts are used by the converted trims"
            )
        object_modes = fallback_structural_part_modes(
            context,
            conversion,
            object_modes,
            selected_configs=selected_configs,
        )
        structural_sources = structural_mirror_sources(context, conversion, object_modes)
        texture_flip_ids = texture_flip_mesh_ids(conversion, object_modes)
        node_mirror_map = build_node_mirror_map(context.node_positions)
        translated_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_TRANSLATE and mesh in prop_meshes
        }
        mirrored_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_MIRROR and mesh in prop_meshes
        }
        structural_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_MIRROR_STRUCTURAL and mesh in prop_meshes
        }
        translated_flexbody_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode == MODE_TRANSLATE and mesh in flexbody_meshes and mesh not in translated_prop_meshes
        }
        translate_magnitudes = part_translate_magnitudes(context, conversion, object_modes)
        zero_translate = sorted(
            object_id
            for object_id, mode in object_modes.items()
            if mode == MODE_TRANSLATE and translate_magnitudes.get(object_id, 0.0) <= 0
        )
        if zero_translate:
            raise RuntimeError(
                "Delta X magnitude is zero for translated part(s): "
                + ", ".join(zero_translate[:8])
                + ("..." if len(zero_translate) > 8 else "")
                + ". Select a steering reference, enter a global manual delta, or set per-part offsets."
            )

    output_root = context.project_dir / "unpacked_output"
    build_dir = context.project_dir / "build"
    clean_dir(output_root)
    build_dir.mkdir(parents=True, exist_ok=True)
    output_vehicle_dir = output_root / context.vehicle_path

    baked_shared_specs: list[BakedMeshSpec] = []
    generated_configs: list[str] = []
    generated_daes: list[Path] = []
    if variant_targets:
        generated_configs.extend(write_generated_jbeam_and_configs(
            context,
            output_vehicle_dir,
            conversion,
            object_modes,
            structural_sources,
            node_mirror_map,
            variant_targets,
            translate_magnitudes,
            translated_prop_meshes,
            translated_flexbody_meshes,
            mirrored_prop_meshes,
            structural_prop_meshes,
            baked_shared_specs,
        ))
        generated_daes = generate_daes(
            context,
            output_root,
            output_vehicle_dir,
            object_modes,
            structural_sources,
            set(variant_targets.values()),
            translate_magnitudes,
            translated_prop_meshes,
            translated_flexbody_meshes,
            context.jbeam_positioned_flexbodies,
            baked_shared_specs,
            texture_flip_ids,
        )
    generated_configs.extend(write_original_plate_configs(
        context,
        output_vehicle_dir,
        conversion,
        original_configs,
    ))
    generated_configs.sort()
    write_mod_info(output_root, context)
    # Licence plates are generated as a separate pass over the written output
    # so plate logic stays fully decoupled from the handedness transforms.
    try:
        plate_summary = plate_generator.apply_to_build(
            context,
            conversion,
            output_root,
            output_vehicle_dir,
            output_plans,
        )
    except plate_generator.PlateError as exc:
        raise RuntimeError(str(exc)) from exc
    embedded_dir = output_root / "handedness_conversion"
    embedded_dir.mkdir(parents=True, exist_ok=True)
    delta = conversion.setdefault("delta", {})
    if isinstance(delta, dict):
        delta["steeringRefs"] = selected_steering_refs(conversion)
    embedded = copy.deepcopy(conversion)
    embedded["builtAt"] = datetime.now().isoformat(timespec="seconds")
    embedded["build"] = {
        "generatedConfigs": generated_configs,
        "outputs": output_plans,
        "targetHands": variant_targets,
        "deltaMagnitude": delta_magnitude(context, conversion),
        "translateMagnitudes": translate_magnitudes,
        "mirroredPropMeshes": sorted(mirrored_prop_meshes),
        "textureFlipMeshes": sorted(texture_flip_ids),
        "structuralMirrorSources": structural_sources,
        "structuralPropMeshes": sorted(structural_prop_meshes),
        "bakedSharedMeshCount": len(baked_shared_specs),
        "cameraNodeMirrorCount": len(node_mirror_map),
        "plates": plate_summary,
    }
    write_text_file(embedded_dir / "conversion.json", json.dumps(embedded, indent=2), encoding="utf-8")

    package_zip = None
    installed_zip = None
    installed_plates_zip = None
    if write_zip:
        package_zip = build_dir / package_name_for_context(context)
        make_zip(output_root, package_zip)
    if install:
        if package_zip is None:
            raise RuntimeError("Install requires zip build")
        if mods_folder is None:
            raise RuntimeError("Install requested without a mods folder")
        mods_folder.mkdir(parents=True, exist_ok=True)
        installed_zip = mods_folder / package_zip.name
        shutil.copy2(package_zip, installed_zip)
        # Refresh the universal plates mod alongside the vehicle so every
        # library design stays selectable on any vehicle, not just the sets
        # bound to this build. A broken library set must not fail the build.
        try:
            plates_mod = plate_generator.export_all_plate_sets()
        except plate_generator.PlateError as exc:
            plate_summary.setdefault("warnings", []).append(f"plates library mod not refreshed: {exc}")
        else:
            if plates_mod is not None:
                plates_zip = Path(plates_mod["zip"])
                installed_plates_zip = mods_folder / plates_zip.name
                shutil.copy2(plates_zip, installed_plates_zip)
                plate_summary["libraryModDesigns"] = plates_mod["designs"]

    save_conversion(context, conversion)
    return BuildResult(
        unpacked_dir=output_root,
        package_zip=package_zip,
        installed_zip=installed_zip,
        generated_configs=generated_configs,
        generated_daes=generated_daes,
        skipped_variants=skipped,
        plate_summary=plate_summary,
        installed_plates_zip=installed_plates_zip,
    )


# ---------------------------------------------------------------------------
# Spatial geometry for Recommend Modes
#
# The classifier in beamng_hand_drive_tool.build_mode_recommendations reasons
# from the driver's viewpoint: an eye point parsed from the jbeam internal
# camera, a nearest-surface shell swept around it, and per-mesh evidence
# (visibility, backing, symmetry, twin geometry). The pure geometry lives
# here; the tool module only orchestrates.

SPATIAL_BIN_DEG = 6.0


@dataclass(frozen=True)
class DriverFrame:
    """The driver's viewpoint: everything the classifier measures is relative
    to this frame, so it transfers across cabins (saloon, truck, buggy)."""

    eye: tuple[float, float, float]
    forward: tuple[float, float, float]
    side: int  # +1 driver sits at +x of the centreline, -1 at -x
    center_x: float
    wheel_id: str | None
    wheel_center: tuple[float, float, float] | None
    source: str  # "camera+wheel" | "camera" | "wheel"


def iter_named_array_texts(text: str, key: str) -> list[str]:
    """Bracket-matched bodies of every '"key": [...]' array in a jbeam text.

    Raw-text variant of part_named_array_for_context for callers that have no
    part id in hand (the internal camera can sit in any part of any file)."""
    bodies: list[str] = []
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*\[', text):
        depth = 0
        i = match.end() - 1
        start = i
        in_str = False
        escaped = False
        while i < len(text):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies.append(text[start:i + 1])
    return bodies


def parse_internal_camera_positions(
    jbeam_texts: dict[str, str],
    node_positions: dict[str, tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Eye positions of every "dash"/"driver" camerasInternal row.

    Coordinates are literal numbers on every vanilla vehicle measured so far;
    a quoted value is resolved through node_positions component-wise as a
    fallback, and rows that resolve nothing are dropped rather than guessed."""
    rows: list[tuple[float, float, float]] = []
    for text in jbeam_texts.values():
        for array_text in iter_named_array_texts(text, "camerasInternal"):
            for row in iter_active_top_level_rows(array_text):
                values = split_top_level_values(row)
                if not values:
                    continue
                kind = quoted_string_value(values[0])
                if kind not in ("dash", "driver"):
                    continue
                position: list[float] = []
                ok = True
                for value in values[1:4]:
                    value = value.strip()
                    quoted = quoted_string_value(value)
                    if quoted is not None:
                        node = node_positions.get(quoted)
                        if node is None:
                            ok = False
                            break
                        position.append(float(node[len(position)]))
                        continue
                    try:
                        position.append(float(value))
                    except ValueError:
                        ok = False
                        break
                if ok and len(position) == 3:
                    rows.append((position[0], position[1], position[2]))
    return rows


def driver_frame_for_context(
    context: VehicleContext,
    object_ids: Iterable[str] | None = None,
) -> DriverFrame | None:
    """Anchor the classifier on the driver's eye.

    Camera rows give the eye; the steering wheel corroborates it and fixes the
    forward direction. Camera without wheel keeps BeamNG's -y-forward
    convention; wheel without camera estimates the eye behind/above the rim
    (degraded); neither means the spatial frame is untrustworthy and callers
    must emit nothing rather than guess."""
    candidates = list(object_ids) if object_ids is not None else list(context.objects)
    camera_rows = parse_internal_camera_positions(context.jbeam_texts, context.node_positions)
    eye: tuple[float, float, float] | None = None
    if camera_rows:
        arr = np.array(camera_rows, dtype=float)
        eye = tuple(float(v) for v in np.median(arr, axis=0))

    best: tuple[int, int, str] | None = None
    for object_id in candidates:
        obj = context.objects.get(object_id)
        if obj is None or object_id not in context.preview_by_id:
            continue
        if is_default_steering_ref(object_id, obj):
            rank = (
                steering_ref_score(object_id, obj),
                vehicle_prefix_rank(context, object_id),
            )
            if best is None or (rank[0], -rank[1]) > (best[0], -best[1]):
                best = (rank[0], rank[1], object_id)
    wheel_id = best[2] if best else None
    wheel_center: tuple[float, float, float] | None = None
    if wheel_id is not None:
        points = np.array(context.preview_by_id[wheel_id]["sample_points"], dtype=float)
        if eye is not None:
            # Wheel meshes may include the whole steering shaft; keep the rim
            # (near and below the eye) so the centre is the hub, not the rack.
            distances = np.linalg.norm(points - np.array(eye), axis=1)
            mask = (distances < 1.3) & (points[:, 2] < eye[2] + 0.25)
            if int(mask.sum()) >= 10:
                points = points[mask]
        wheel_center = tuple(float(v) for v in np.median(points, axis=0))

    if eye is None and wheel_center is None:
        return None
    center_x = median_value([p[0] for p in context.node_positions.values()]) or 0.0
    if eye is None:
        eye = (wheel_center[0], wheel_center[1] + 0.60, wheel_center[2] + 0.35)
        source = "wheel"
        forward = (0.0, -1.0, 0.0)
    elif wheel_center is None:
        source = "camera"
        forward = (0.0, -1.0, 0.0)
    else:
        source = "camera+wheel"
        fxy = np.array([wheel_center[0] - eye[0], wheel_center[1] - eye[1]])
        norm = float(np.linalg.norm(fxy))
        forward = (fxy[0] / norm, fxy[1] / norm, 0.0) if norm > 0.05 else (0.0, -1.0, 0.0)
    side = 1 if (eye[0] - center_x) >= 0 else -1
    return DriverFrame(
        eye=eye,
        forward=(float(forward[0]), float(forward[1]), 0.0),
        side=side,
        center_x=float(center_x),
        wheel_id=wheel_id,
        wheel_center=wheel_center,
        source=source,
    )


def spatial_spherical_bins(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-angle (elevation, azimuth) bin index and range per direction."""
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    step = math.radians(SPATIAL_BIN_DEG)
    n_azimuth = int(2 * math.pi / step) + 1
    i_el = ((elevation + math.pi / 2) / step).astype(np.int64)
    i_az = ((azimuth + math.pi) / step).astype(np.int64)
    return i_el * n_azimuth + i_az, radii


def visibility_scan(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    transparent_ids: set[str],
    forward: tuple[float, float, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Nearest-surface shell around the eye plus per-mesh evidence.

    Per mesh:
      vf     -- fraction of points on/inside the 360-degree shell
      front_vf -- fraction both on/inside the shell and in the forward
                  180-degree hemisphere (equals vf when forward is omitted)
      backed -- fraction with ANY other mesh somewhere behind (not outermost)
      lined  -- fraction with another mesh CLOSE behind, 3-30 cm (a lining
                layer: door card over skin); own-mesh thickness never counts
      depth  -- mean distance points sit behind the shell (occlusion depth)
      min_r  -- nearest point to the eye

    The shell is per-bin nearest with a range-scaled tolerance; deliberately
    NOT dilated -- on ~350-point clouds dilation over-occludes far worse than
    leaks under-occlude, and the vetoes downstream absorb the leaks."""
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for index, (object_id, points) in enumerate(points_by_id.items()):
        ids.append(object_id)
        chunks.append(points)
        owners.append(np.full(len(points), index))
    all_points = np.concatenate(chunks)
    owner = np.concatenate(owners)
    eye_vectors = all_points - np.array(eye)
    bins, radii = spatial_spherical_bins(eye_vectors)
    opaque = np.array([ids[o] not in transparent_ids for o in owner])
    gpu_shell = spatial_visibility_backend.gpu_visibility_shell(
        eye_vectors, bins, radii, owner, opaque, forward
    )
    if gpu_shell is not None:
        visible, front_visible, backed, lined_flag, depth = gpu_shell
        stats: dict[str, dict[str, float]] = {}
        for index, object_id in enumerate(ids):
            mask = owner == index
            count = int(mask.sum())
            stats[object_id] = {
                "n": count,
                "vf": float(visible[mask].sum()) / count,
                "front_vf": float(front_visible[mask].sum()) / count,
                "backed": float(backed[mask].sum()) / count,
                "lined": float(lined_flag[mask].sum()) / count,
                "depth": float(depth[mask].mean()),
                "min_r": float(radii[mask].min()),
            }
        return stats
    n_bins = int(bins.max()) + 2
    field = np.full(n_bins, np.inf)
    np.minimum.at(field, bins[opaque], radii[opaque])

    far1 = np.full(n_bins, -np.inf)
    far1_owner = np.full(n_bins, -1)
    far2 = np.full(n_bins, -np.inf)
    far2_owner = np.full(n_bins, -1)
    bins_o = bins[opaque]
    radii_o = radii[opaque]
    owner_o = owner[opaque]
    order = np.lexsort((radii_o, bins_o))
    sorted_bins = bins_o[order]
    sorted_radii = radii_o[order]
    sorted_owner = owner_o[order]
    is_last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
    far1[sorted_bins[is_last]] = sorted_radii[is_last]
    far1_owner[sorted_bins[is_last]] = sorted_owner[is_last]
    second_idx = np.flatnonzero(is_last) - 1
    second_idx = second_idx[second_idx >= 0]
    is_second = np.zeros(len(sorted_bins), dtype=bool)
    is_second[second_idx] = True
    if len(sorted_bins):
        is_second &= np.r_[sorted_bins[:-1] == sorted_bins[1:], False]
    far2[sorted_bins[is_second]] = sorted_radii[is_second]
    far2_owner[sorted_bins[is_second]] = sorted_owner[is_second]

    # close backing (lined): another mesh within (0.03, 0.30] behind, same bin
    lined_flag = np.zeros(len(all_points), dtype=bool)
    segment_start = np.flatnonzero(np.r_[True, sorted_bins[1:] != sorted_bins[:-1]])
    segment_end = np.r_[segment_start[1:], len(sorted_bins)]
    opaque_idx = np.flatnonzero(opaque)
    for seg_a, seg_b in zip(segment_start, segment_end):
        if seg_b - seg_a < 2:
            continue
        seg_radii = sorted_radii[seg_a:seg_b]
        seg_owner = sorted_owner[seg_a:seg_b]
        for i in range(seg_b - seg_a):
            r_i = seg_radii[i]
            lo = np.searchsorted(seg_radii, r_i + 0.03, side="right")
            hi = np.searchsorted(seg_radii, r_i + 0.30, side="right")
            if lo < hi and np.any(seg_owner[lo:hi] != seg_owner[i]):
                lined_flag[opaque_idx[order[seg_a + i]]] = True

    tolerance = 0.05 + 0.04 * radii
    visible = radii <= field[bins] + tolerance
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((eye_vectors @ forward_v) >= 0.0)
    depth = np.maximum(0.0, radii - field[bins])
    behind1 = (far1[bins] > radii + 0.05) & (far1_owner[bins] != owner)
    behind2 = (far2[bins] > radii + 0.05) & (far2_owner[bins] != owner)
    backed = behind1 | behind2

    stats: dict[str, dict[str, float]] = {}
    for index, object_id in enumerate(ids):
        mask = owner == index
        count = int(mask.sum())
        stats[object_id] = {
            "n": count,
            "vf": float(visible[mask].sum()) / count,
            "front_vf": float(front_visible[mask].sum()) / count,
            "backed": float(backed[mask].sum()) / count,
            "lined": float(lined_flag[mask].sum()) / count,
            "depth": float(depth[mask].mean()),
            "min_r": float(radii[mask].min()),
        }
    return stats


def surface_visibility_stats(
    points: np.ndarray,
    eye: tuple[float, float, float],
    triangles_by_id: dict[str, np.ndarray] | np.ndarray,
    transparent_ids: set[str],
    forward: tuple[float, float, float] | None = None,
    endpoint_gap: float = 0.002,
) -> dict[str, float] | None:
    """Exact point visibility against filled mesh triangles.

    Rays run from the eye to each sampled point.  A point is hidden when any
    opaque triangle intersects the open segment, including an earlier surface
    of its own mesh.  Intersections within ``endpoint_gap`` of the sampled
    point are ignored so that the point's incident face does not hide itself.
    Callers may supply per-object arrays or one scene compiled once for the
    trim; the classifier uses the latter to avoid hundreds of tiny NumPy
    kernels per candidate.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    origin = np.asarray(eye, dtype=float)
    directions = points - origin
    distances = np.linalg.norm(directions, axis=1)
    valid_rays = distances > 1e-9
    blocked = np.zeros(len(points), dtype=bool)
    has_surface = False

    # Moller-Trumbore, vectorised over all still-live rays and bounded chunks
    # of triangles.  Parameters use the unnormalised eye->point vector, so an
    # intersection lies on the segment precisely when 0 < t < 1.
    if isinstance(triangles_by_id, np.ndarray):
        surface_items = (("", triangles_by_id),)
    else:
        surface_items = triangles_by_id.items()
    for object_id, object_triangles in surface_items:
        if object_id in transparent_ids:
            continue
        triangles = np.asarray(object_triangles, dtype=float).reshape((-1, 3, 3))
        if len(triangles) == 0:
            continue
        has_surface = True
        for start in range(0, len(triangles), 1024):
            active = np.flatnonzero(valid_rays & ~blocked)
            if len(active) == 0:
                break
            tri = triangles[start : start + 1024]
            vertex0 = tri[:, 0]
            edge1 = tri[:, 1] - vertex0
            edge2 = tri[:, 2] - vertex0
            # Degenerate export faces cannot occlude anything and amplify
            # floating-point noise in the determinant.
            nondegenerate = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-12
            if not bool(nondegenerate.any()):
                continue
            vertex0 = vertex0[nondegenerate]
            edge1 = edge1[nondegenerate]
            edge2 = edge2[nondegenerate]
            ray_d = directions[active]
            pvec = np.cross(ray_d[:, None, :], edge2[None, :, :])
            det = np.einsum("tj,rtj->rt", edge1, pvec)
            determinant_ok = np.abs(det) > 1e-10
            inverse_det = np.zeros_like(det)
            np.divide(1.0, det, out=inverse_det, where=determinant_ok)
            tvec = origin[None, :] - vertex0
            u = np.einsum("tj,rtj->rt", tvec, pvec) * inverse_det
            qvec = np.cross(tvec, edge1)
            v = np.einsum("rj,tj->rt", ray_d, qvec) * inverse_det
            t_num = np.einsum("tj,tj->t", edge2, qvec)
            t = t_num[None, :] * inverse_det
            maximum_t = 1.0 - endpoint_gap / distances[active]
            hit = (
                determinant_ok
                & (u >= -1e-9)
                & (v >= -1e-9)
                & (u + v <= 1.0 + 1e-9)
                & (t > 1e-7)
                & (t < maximum_t[:, None])
            )
            blocked[active[np.any(hit, axis=1)]] = True
        if bool((valid_rays & ~blocked).sum()) == 0:
            break

    if not has_surface:
        return None
    visible = valid_rays & ~blocked
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((directions @ forward_v) >= 0.0)
    return {
        "vf": float(visible.sum()) / len(points),
        "front_vf": float(front_visible.sum()) / len(points),
        "blocked": float(blocked.sum()) / len(points),
    }


def surface_visibility_stats_batch(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    triangles: np.ndarray,
    forward: tuple[float, float, float] | None = None,
    endpoint_gap: float = 0.002,
    max_cpu_workers: int = 4,
) -> dict[str, dict[str, float] | None]:
    """Exact visibility for many meshes with one compiled triangle scene.

    The GPU path concatenates every candidate's rays into one compute dispatch
    and uploads the scene once.  It implements the same Moller-Trumbore/open-
    segment test as :func:`surface_visibility_stats`.  Machines without an
    OpenGL 4.3 compute context retain the previous bounded CPU thread pool.
    Set ``BEAMXP_SPATIAL_BACKEND=cpu`` to force that reference path.
    """
    usable = {
        object_id: np.asarray(points, dtype=float).reshape((-1, 3))
        for object_id, points in points_by_id.items()
        if len(points)
    }
    if not usable:
        return {}
    scene = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
    if not len(scene):
        return {object_id: None for object_id in usable}

    ids = list(usable)
    counts = [len(usable[object_id]) for object_id in ids]
    all_points = np.concatenate([usable[object_id] for object_id in ids])
    blocked = spatial_visibility_backend.gpu_blocked_rays(
        all_points, eye, scene, endpoint_gap
    )
    if blocked is None:
        workers = max(1, min(max_cpu_workers, len(ids)))
        if workers == 1:
            return {
                object_id: surface_visibility_stats(
                    usable[object_id], eye, scene, set(), forward, endpoint_gap
                )
                for object_id in ids
            }
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="spatial-rays"
        ) as executor:
            futures = {
                object_id: executor.submit(
                    surface_visibility_stats,
                    usable[object_id],
                    eye,
                    scene,
                    set(),
                    forward,
                    endpoint_gap,
                )
                for object_id in ids
            }
            return {
                object_id: futures[object_id].result()
                for object_id in ids
            }

    origin = np.asarray(eye, dtype=float)
    directions = all_points - origin
    distances = np.linalg.norm(directions, axis=1)
    valid = distances > 1e-9
    visible = valid & ~blocked
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((directions @ forward_v) >= 0.0)

    results: dict[str, dict[str, float] | None] = {}
    offset = 0
    for object_id, count in zip(ids, counts):
        item = slice(offset, offset + count)
        results[object_id] = {
            "vf": float(visible[item].sum()) / count,
            "front_vf": float(front_visible[item].sum()) / count,
            "blocked": float(blocked[item].sum()) / count,
        }
        offset += count
    return results


def directional_verdict_backing(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    foreground_ids: Iterable[str],
    verdict_by_id: dict[str, str],
    min_gap: float = 0.03,
) -> dict[str, dict[str, float]]:
    """Fractions of a mesh's sightline points backed by transformed classes.

    For each 6-degree direction bin, the farthest point belonging to a mesh
    in ``verdict_by_id`` supplies both its range and verdict class.  A
    foreground point is backed when that transformed geometry continues at
    least ``min_gap`` farther along the same eye ray.  Distances are otherwise
    unbounded: this is a line-of-sight relationship, not a contact test.
    """
    transformed = [
        object_id for object_id in verdict_by_id
        if object_id in points_by_id and len(points_by_id[object_id])
    ]
    foreground = [
        object_id for object_id in foreground_ids
        if object_id in points_by_id and len(points_by_id[object_id])
    ]
    result = {object_id: {} for object_id in foreground}
    if not transformed or not foreground:
        return result

    class_names = sorted(set(verdict_by_id[object_id] for object_id in transformed))
    class_index = {name: index for index, name in enumerate(class_names)}
    chunks = [points_by_id[object_id] for object_id in transformed]
    classes = np.concatenate([
        np.full(len(points_by_id[object_id]), class_index[verdict_by_id[object_id]])
        for object_id in transformed
    ])
    all_points = np.concatenate(chunks)
    bins, radii = spatial_spherical_bins(all_points - np.asarray(eye, dtype=float))
    order = np.lexsort((radii, bins))
    sorted_bins = bins[order]
    is_last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
    winners = order[is_last]
    n_bins = int(bins.max()) + 2
    far_range = np.full(n_bins, -np.inf)
    far_class = np.full(n_bins, -1, dtype=np.int64)
    far_range[bins[winners]] = radii[winners]
    far_class[bins[winners]] = classes[winners]

    eye_v = np.asarray(eye, dtype=float)
    for object_id in foreground:
        points = points_by_id[object_id]
        point_bins, point_ranges = spatial_spherical_bins(points - eye_v)
        in_field = point_bins < n_bins
        backed = np.zeros(len(points), dtype=bool)
        backed[in_field] = far_range[point_bins[in_field]] > point_ranges[in_field] + min_gap
        count = len(points)
        for name, index in class_index.items():
            matched = backed & in_field
            matched[in_field] &= far_class[point_bins[in_field]] == index
            hits = int(matched.sum())
            if hits:
                result[object_id][name] = float(hits) / count
    return result


def floor_height_from_shell(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    forward: tuple[float, float, float],
    transparent_ids: set[str],
) -> float | None:
    """Cabin floor height read off the shell in the forward-down (footwell)
    sector. Straight-down rays end on the seat cushion, so they are excluded."""
    chunks = [pts for oid, pts in points_by_id.items() if oid not in transparent_ids]
    if not chunks:
        return None
    all_points = np.concatenate(chunks)
    vectors = all_points - np.array(eye)
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    horizontal = vectors[:, :2] / np.maximum(np.linalg.norm(vectors[:, :2], axis=1), 1e-9)[:, None]
    forwardness = horizontal @ np.array(forward[:2])
    footwell = (
        (elevation < math.radians(-35.0))
        & (elevation > math.radians(-75.0))
        & (forwardness > 0.5)
    )
    if int(footwell.sum()) < 20:
        return None
    bins, seg_radii = spatial_spherical_bins(vectors[footwell])
    heights = all_points[footwell][:, 2]
    n_bins = int(bins.max()) + 2
    nearest_field = np.full(n_bins, np.inf)
    np.minimum.at(nearest_field, bins, seg_radii)
    nearest = seg_radii <= nearest_field[bins] + 0.02
    return float(np.percentile(heights[nearest], 20))


def glass_beyond_fractions(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    glass_ids: set[str],
    check_ids: Iterable[str],
) -> dict[str, float]:
    """Fraction of each mesh's points beyond a large planar glass pane within
    the pane's angular footprint: outside the glasshouse (wipers beyond the
    windscreen, the truck bed beyond the rear window). Small panes (gauge
    lenses) are not cabin boundaries and are ignored."""
    result = {object_id: 0.0 for object_id in check_ids}
    eye_v = np.array(eye)
    step = math.radians(SPATIAL_BIN_DEG)
    n_azimuth = int(2 * math.pi / step) + 1
    panes: list[tuple[np.ndarray, np.ndarray, set]] = []
    for glass_id in glass_ids:
        points = points_by_id.get(glass_id)
        if points is None or len(points) < 10:
            continue
        if float(np.linalg.norm(np.ptp(points, axis=0))) < 0.75:
            continue  # instrument lens, not a window
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[2]
        rms = float(np.sqrt(((centered @ normal) ** 2).mean()))
        if rms > 0.05:
            continue  # not planar enough to bound anything
        if (centroid - eye_v) @ normal < 0:
            normal = -normal  # orient away from the eye
        pane_bins, _ = spatial_spherical_bins(points - eye_v)
        footprint = set()
        for b in set(pane_bins.tolist()):
            for d_el in (-2, -1, 0, 1, 2):
                for d_az in (-2, -1, 0, 1, 2):
                    footprint.add(b + d_el * n_azimuth + d_az)
        panes.append((centroid, normal, footprint))
    if not panes:
        return result
    for object_id in result:
        points = points_by_id.get(object_id)
        if points is None or len(points) == 0:
            continue
        point_bins, _ = spatial_spherical_bins(points - eye_v)
        beyond = np.zeros(len(points), dtype=bool)
        for centroid, normal, footprint in panes:
            in_footprint = np.fromiter(
                (b in footprint for b in point_bins.tolist()), bool, len(points)
            )
            distance = (points - centroid) @ normal
            beyond |= in_footprint & (distance > 0.05)
        result[object_id] = float(beyond.mean())
    return result


def cloud_symmetry_residual(points: np.ndarray, center_x: float) -> float:
    """Normalised Chamfer residual of the cloud against its own x-reflection.

    Small means reflecting the mesh across the centreline is a visual no-op;
    large means the mesh is one-sided. Normalised by the bbox diagonal so a
    4 cm button and a 2 m dashboard are judged on the same scale."""
    reflected = points.copy()
    reflected[:, 0] = 2 * center_x - reflected[:, 0]
    squared = ((reflected[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    nearest = np.sqrt(squared.min(axis=1))
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    return float(np.median(nearest)) / max(diagonal, 0.05)


def reflected_orphan_stats(
    points: np.ndarray,
    center_x: float,
    exact_tol: float = 1e-4,
    coarse_tol: float = 0.02,
) -> tuple[int, float]:
    """Count vertices missing an x-reflected twin at two tolerances.

    ``orphans`` uses a 0.1 mm default tolerance to absorb DAE float dust while
    retaining a zero-threshold decision: only a mesh whose every vertex has a
    reflected partner is a symmetry no-op.  ``coarse_fraction`` is diagnostic
    evidence for confidence grading, never the skip decision.

    A spatial hash keeps each pass linear for ordinary mesh density.  Actual
    Euclidean distances are checked inside neighbouring cells, so quantisation
    cannot turn a near cell into a false match.
    """
    cloud = np.asarray(points, dtype=float)
    if len(cloud) == 0:
        return 0, 0.0
    if exact_tol <= 0.0 or coarse_tol <= 0.0:
        raise ValueError("symmetry tolerance must be positive")

    gpu_stats = spatial_visibility_backend.gpu_reflected_orphan_stats(
        cloud,
        float(center_x),
        float(exact_tol),
        float(coarse_tol),
    )
    if gpu_stats is not None:
        return gpu_stats

    reflected = cloud.copy()
    reflected[:, 0] = 2.0 * float(center_x) - reflected[:, 0]

    def orphan_count(tol: float) -> int:
        cells: dict[tuple[int, int, int], list[np.ndarray]] = {}
        indices = np.floor(cloud / tol).astype(np.int64)
        for index, cell in enumerate(indices):
            cells.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(cloud[index])
        tol2 = tol * tol
        count = 0
        for query in reflected:
            base = np.floor(query / tol).astype(np.int64)
            matched = False
            for dx in (-1, 0, 1):
                if matched:
                    break
                for dy in (-1, 0, 1):
                    if matched:
                        break
                    for dz in (-1, 0, 1):
                        candidates = cells.get(
                            (int(base[0] + dx), int(base[1] + dy), int(base[2] + dz))
                        )
                        if candidates is None:
                            continue
                        if any(float(np.dot(point - query, point - query)) <= tol2 for point in candidates):
                            matched = True
                            break
            if not matched:
                count += 1
        return count

    exact_orphans = orphan_count(float(exact_tol))
    coarse_orphans = orphan_count(float(coarse_tol))
    return exact_orphans, float(coarse_orphans) / len(cloud)


def mirror_pair_residual(points_a: np.ndarray, points_b: np.ndarray, center_x: float) -> float:
    """Symmetric normalised Chamfer distance between reflect(A) and B: how
    well B is A's geometric twin across the centreline."""
    reflected = points_a.copy()
    reflected[:, 0] = 2 * center_x - reflected[:, 0]

    def chamfer(u: np.ndarray, v: np.ndarray) -> float:
        squared = ((u[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        return float(np.median(np.sqrt(squared.min(axis=1))))

    diagonal = float(np.linalg.norm(np.ptp(np.concatenate([points_a, points_b]), axis=0)))
    return (chamfer(reflected, points_b) + chamfer(points_b, reflected)) / 2.0 / max(diagonal, 0.05)


def principal_extent_sds(points: np.ndarray) -> np.ndarray:
    """Ascending standard deviations along the cloud's principal axes
    (thickness, width, length): the planarity/elongation fingerprint."""
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))


def material_flags_for_context(context: VehicleContext) -> dict[str, dict[str, bool]]:
    """Material name -> {"emissive", "glass"} from every *.materials.json in
    the vehicle zip and the game's common zips.

    emissive = any stage carries an emissiveMap (a lit display image).
    glass    = translucent and NOT emissive (a window; translucent+emissive is
               a screen). Cached on the context; missing zips degrade to {}."""
    cached = getattr(context, "_material_flags", None)
    if cached is not None:
        return cached
    flags: dict[str, dict[str, bool]] = {}
    zips = [context.source_zip] + common_zip_candidates(context.source_zip)
    for zip_path in zips:
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception:
            continue
        with zf:
            for name in zf.namelist():
                if not name.replace("\\", "/").endswith(".materials.json"):
                    continue
                try:
                    data = parse_beamng_json(
                        zf.read(name).decode("utf-8", errors="replace"), label=name
                    )
                except Exception:
                    continue
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    material_name = str(value.get("mapTo") or value.get("name") or key)
                    stages = value.get("Stages") or []
                    emissive = any(
                        isinstance(stage, dict) and "emissiveMap" in stage for stage in stages
                    )
                    glass = bool(value.get("translucent")) and not emissive
                    flags[material_name.lower()] = {"emissive": emissive, "glass": glass}
    context._material_flags = flags
    return flags


def mesh_material_symbols(context: VehicleContext) -> dict[str, tuple[str, ...]]:
    """Mesh id -> material symbols bound in the DAE.

    New contexts carry them in preview_by_id (preview_data_from_tree); older
    cached contexts fall back to a one-off parse of the vehicle's own DAEs.
    Shared-library meshes without symbols simply get no material evidence."""
    cached = getattr(context, "_material_symbols", None)
    if cached is not None:
        return cached
    symbols: dict[str, tuple[str, ...]] = {}
    missing = False
    for object_id, item in context.preview_by_id.items():
        materials = item.get("materials")
        if materials is None:
            missing = True
        elif materials:
            symbols[object_id] = tuple(materials)
    if missing and not symbols:
        try:
            with zipfile.ZipFile(context.source_zip) as zf:
                for dae_path in context.dae_paths:
                    try:
                        tree = ET.parse(io.BytesIO(zf.read(dae_path)))
                    except Exception:
                        continue
                    for node in tree.getroot().findall(".//c:node", NS):
                        node_symbols = {
                            re.sub(r"-material$", "", symbol).lower()
                            for inst in node.findall(".//c:instance_material", NS)
                            for symbol in (inst.get("symbol") or inst.get("target", "").lstrip("#"),)
                            if symbol
                        }
                        if node_symbols:
                            for alias in dae_node_aliases(node):
                                symbols.setdefault(alias, tuple(sorted(node_symbols)))
        except Exception:
            pass
    context._material_symbols = symbols
    return symbols
