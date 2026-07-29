from __future__ import annotations

import copy
import csv
import json
import math
import os
import queue
import shutil
import threading
import time
import traceback
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_NAME = "BeamXP Vehicle ZIP Mesh Transform POC"
PRIMITIVE_TAGS = {"triangles", "polylist", "polygons", "trifans", "tristrips"}
TRANSFORM_TAGS = {"matrix", "translate", "rotate", "scale", "lookat", "skew"}
DIALOG_DIRECTORY_KEYS = frozenset({"source", "export"})


def application_settings_path() -> Path:
    """Return a per-user settings path without requiring installation privileges."""
    if os.name == "nt":
        base_value = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base_value:
            return Path(base_value) / "BeamXP" / "mesh_transform_poc_settings.json"
    else:
        xdg_value = os.environ.get("XDG_CONFIG_HOME")
        if xdg_value:
            return Path(xdg_value) / "BeamXP" / "mesh_transform_poc_settings.json"
    return Path.home() / ".config" / "BeamXP" / "mesh_transform_poc_settings.json"


def load_dialog_directories(path: Path | None = None) -> dict[str, str]:
    """Load only known dialog-directory values; malformed settings are ignored."""
    settings_path = path or application_settings_path()
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    raw = payload.get("dialog_directories", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if key in DIALOG_DIRECTORY_KEYS and isinstance(value, str) and value.strip()
    }


def save_dialog_directories(
    directories: dict[str, str],
    path: Path | None = None,
) -> None:
    """Persist dialog directories atomically; failure never blocks mesh work."""
    settings_path = path or application_settings_path()
    payload = {
        "dialog_directories": {
            key: value
            for key, value in directories.items()
            if key in DIALOG_DIRECTORY_KEYS and isinstance(value, str) and value.strip()
        }
    }
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = settings_path.with_name(settings_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(settings_path)
    except OSError:
        # Remembering a folder is a convenience feature; never fail a conversion
        # because a profile directory is read-only or temporarily unavailable.
        return


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class GeometryInstance:
    geometry_id: str


@dataclass(slots=True)
class DaePart:
    key: str
    label: str
    node_id: str
    node_name: str
    matrix: np.ndarray
    instances: tuple[GeometryInstance, ...]


@dataclass(slots=True, frozen=True)
class ArchiveMaterialPreviewLayer:
    base_colour_reference: str
    base_colour_factor: tuple[float, float, float, float]
    opacity_reference: str


@dataclass(slots=True)
class PrimitiveTriangles:
    attributes: dict[str, str]
    input_attributes: tuple[dict[str, str], ...]
    rows: np.ndarray  # (triangle, corner, input stride)


@dataclass(slots=True)
class RawGeometry:
    vertices: np.ndarray
    triangles: np.ndarray
    primitives: tuple[PrimitiveTriangles, ...]
    face_sources: tuple[tuple[int, int], ...]  # primitive index, local triangle index


@dataclass(slots=True, frozen=True)
class SourceFaceRef:
    instance_index: int
    geometry_id: str
    primitive_index: int
    triangle_index: int


@dataclass(slots=True)
class LoadedDae:
    path: Path
    tree: ET.ElementTree
    namespace: str
    unit_scale: float
    parts: list[DaePart]
    geometries: dict[str, ET.Element]
    geometry_cache: dict[str, RawGeometry] = field(default_factory=dict)
    topology_cache: dict[str, "Topology"] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ArchiveMaterialRecord:
    key: str
    name: str
    map_to: str
    materials_member: str
    base_colour_reference: str
    preview_layers: tuple[ArchiveMaterialPreviewLayer, ...] = ()

    @property
    def aliases(self) -> tuple[str, ...]:
        values = (self.key, self.name, self.map_to)
        return tuple(dict.fromkeys(value for value in values if value))


@dataclass(slots=True, frozen=True)
class ArchiveTextureBinding:
    dae_material: str
    material_key: str
    materials_member: str
    texture_reference: str
    texture_member: str
    preview_layers: tuple[ArchiveMaterialPreviewLayer, ...] = ()


@dataclass(slots=True)
class VehicleArchive:
    path: Path
    members: tuple[str, ...]
    member_by_lower: dict[str, str]
    member_sizes: dict[str, int]
    dae_members: tuple[str, ...]
    materials: tuple[ArchiveMaterialRecord, ...]
    workspace: Path
    material_errors: tuple[str, ...] = ()


@dataclass(slots=True)
class Topology:
    vertices: np.ndarray
    triangles: np.ndarray
    source_faces: tuple[SourceFaceRef, ...]
    face_normals: np.ndarray
    face_areas: np.ndarray
    edge_faces: dict[tuple[int, int], tuple[int, ...]]
    edge_angles: dict[tuple[int, int], float]
    boundary_edges: set[tuple[int, int]]
    nonmanifold_edges: set[tuple[int, int]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    diagonal: float


@dataclass(slots=True)
class RegionBoundary:
    edges: tuple[tuple[int, int], ...]
    loops: tuple[tuple[int, ...], ...]
    perimeter: float
    crease_edges: int
    mesh_edges: int
    cut_edges: int
    nonmanifold_edges: int
    closed: bool


@dataclass(slots=True)
class ActiveRegion:
    region_index: int
    island_index: int
    faces: tuple[int, ...]
    area: float
    area_perimeter_ratio: float
    boundary: RegionBoundary
    role: str
    eligible: bool

    @property
    def triangle_count(self) -> int:
        return len(self.faces)


@dataclass(slots=True)
class SymmetryMeasurement:
    centroid: tuple[float, float, float]
    plane_normal: tuple[float, float, float]
    post_reflection_variance: float
    rms_error: float
    max_error: float
    initial_plane_normal: tuple[float, float, float]
    initial_post_reflection_variance: float
    initial_rms_error: float
    initial_max_error: float
    mirror_plane_y_tilt_degrees: float
    rigid_y_rotation_correction_degrees: float
    tilt_search_applied: bool
    whole_loop_sample_count: int
    sibling_pair_count: int
    mirror_crossings: int
    passed: bool


@dataclass(slots=True, frozen=True)
class ChildAdoption:
    island_index: int
    faces: tuple[int, ...]
    area: float
    area_ratio: float
    projected_overlap: float
    median_surface_gap: float
    p90_surface_gap: float
    adoption_mode: str = "eager_island"
    source_candidate_id: int | None = None

    @property
    def triangle_count(self) -> int:
        return len(self.faces)


@dataclass(slots=True)
class AcceptedCandidate:
    key: tuple[int, int]  # island index, local candidate number
    candidate_id: int
    island_index: int
    accepted_level: int
    accepted_angle: float
    faces: tuple[int, ...]
    area: float
    area_perimeter_ratio: float
    boundary_edges: tuple[tuple[int, int], ...]
    boundary_loops: tuple[tuple[int, ...], ...]
    perimeter: float
    measurement: SymmetryMeasurement
    host_faces: tuple[int, ...] = ()
    adoptions: tuple[ChildAdoption, ...] = ()

    @property
    def triangle_count(self) -> int:
        return len(self.faces)


@dataclass(slots=True)
class IslandCandidate:
    key: tuple[int, int]
    island_index: int
    accepted_level: int
    accepted_angle: float
    faces: tuple[int, ...]
    area: float
    area_perimeter_ratio: float
    boundary_edges: tuple[tuple[int, int], ...]
    boundary_loops: tuple[tuple[int, ...], ...]
    perimeter: float
    measurement: SymmetryMeasurement
    host_faces: tuple[int, ...] = ()
    adoptions: tuple[ChildAdoption, ...] = ()


@dataclass(slots=True)
class IslandLevelResult:
    level_index: int
    threshold: float
    recursion_passes: int
    tested_candidates: int
    accepted_keys: tuple[tuple[int, int], ...]
    active_faces: tuple[int, ...]
    active_regions: tuple[ActiveRegion, ...]


@dataclass(slots=True)
class IslandSweepResult:
    island_index: int
    accepted: tuple[IslandCandidate, ...]
    remaining_faces: tuple[int, ...]
    levels: tuple[IslandLevelResult, ...]


@dataclass(slots=True)
class SweepLevelResult:
    level_index: int
    crease_angle: float
    recursion_passes: int
    tested_candidates: int
    accepted_this_level: tuple[int, ...]
    accepted_cumulative: tuple[int, ...]
    active_regions: list[ActiveRegion]
    active_face_labels: np.ndarray
    candidate_face_labels: np.ndarray


@dataclass(slots=True)
class SymmetrySweepResult:
    part: DaePart
    topology: Topology
    crease_max: float
    crease_min: float
    threshold_steps: int
    thresholds: tuple[float, ...]
    min_region_faces: int
    min_area_perimeter_ratio_metres: float
    symmetry_tolerance_metres: float
    direct_symmetry_tolerance_metres: float
    sample_spacing_metres: float
    island_faces: tuple[tuple[int, ...], ...]
    main_island_index: int
    candidates: list[AcceptedCandidate]
    remaining_faces: tuple[int, ...]
    levels: list[SweepLevelResult]
    adopted_island_count: int
    late_adopted_candidate_count: int
    adopted_child_count: int
    adopted_triangle_count: int
    processing_seconds: float
    topology_seconds: float
    sweep_seconds: float


# ---------------------------------------------------------------------------
# COLLADA parsing
# ---------------------------------------------------------------------------


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def namespace_uri(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag[1:].split("}", 1)[0]
    return ""


def qname(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name




def _normalise_archive_member(member: str) -> str:
    """Return a stable POSIX member path and reject traversal components."""
    cleaned = member.replace("\\", "/").lstrip("/")
    path = PurePosixPath(cleaned)
    if not cleaned or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe or invalid ZIP member path: {member!r}")
    return path.as_posix()


def _human_size(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.1f} {suffix}" if suffix != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def _material_texture_reference(stage: dict[str, object]) -> str:
    for key in ("baseColorMap", "colorMap", "diffuseMap"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip() and not value.lstrip().startswith("@"):
            return value.strip()
    return ""


def _material_base_colour_reference(material: dict[str, object]) -> str:
    stages = material.get("Stages")
    if not isinstance(stages, list):
        return ""
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        reference = _material_texture_reference(stage)
        if reference:
            return reference
    return ""


def _material_base_colour_factor(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        red, green, blue = (float(value[index]) for index in range(3))
        alpha = float(value[3]) if len(value) > 3 else 1.0
    except (TypeError, ValueError):
        return None
    return (
        min(max(red, 0.0), 1.0),
        min(max(green, 0.0), 1.0),
        min(max(blue, 0.0), 1.0),
        min(max(alpha, 0.0), 1.0),
    )


def _material_preview_layers(material: dict[str, object]) -> tuple[ArchiveMaterialPreviewLayer, ...]:
    stages = material.get("Stages")
    if not isinstance(stages, list):
        return ()
    try:
        active_layers = int(material.get("activeLayers") or 0)
    except (TypeError, ValueError):
        active_layers = 0
    if active_layers <= 1:
        active_layers = len(stages)

    layers: list[ArchiveMaterialPreviewLayer] = []
    for stage in stages[1:active_layers]:
        if not isinstance(stage, dict):
            continue
        base_colour = _material_texture_reference(stage)
        opacity = stage.get("opacityMap")
        if not (
            base_colour
            and isinstance(opacity, str)
            and opacity.strip()
            and not opacity.lstrip().startswith("@")
        ):
            continue
        factor = _material_base_colour_factor(stage.get("baseColorFactor"))
        layers.append(
            ArchiveMaterialPreviewLayer(
                base_colour_reference=base_colour,
                base_colour_factor=factor or (1.0, 1.0, 1.0, 1.0),
                opacity_reference=opacity.strip(),
            )
        )
    return tuple(layers)


def scan_vehicle_archive(path: Path, workspace: Path | None = None) -> VehicleArchive:
    """Index a BeamNG vehicle/mod ZIP without extracting the whole archive."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a readable ZIP archive: {path}")

    workspace = workspace or Path(tempfile.mkdtemp(prefix="beamxp_vehicle_zip_"))
    members: list[str] = []
    sizes: dict[str, int] = {}
    member_by_lower: dict[str, str] = {}
    materials: list[ArchiveMaterialRecord] = []
    errors: list[str] = []

    with zipfile.ZipFile(path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            try:
                member = _normalise_archive_member(info.filename)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            members.append(member)
            sizes[member] = int(info.file_size)
            member_by_lower.setdefault(member.lower(), member)

        for member in members:
            if not member.lower().endswith(".materials.json"):
                continue
            try:
                raw = archive.read(member).decode("utf-8-sig")
                document = json.loads(raw)
            except Exception as exc:
                errors.append(f"{member}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(document, dict):
                continue
            for key, value in document.items():
                if not isinstance(value, dict):
                    continue
                base_colour = _material_base_colour_reference(value)
                if not base_colour:
                    continue
                name = str(value.get("name") or "").strip()
                map_to = str(value.get("mapTo") or "").strip()
                materials.append(
                    ArchiveMaterialRecord(
                        key=str(key).strip(),
                        name=name,
                        map_to=map_to,
                        materials_member=member,
                        base_colour_reference=base_colour,
                        preview_layers=_material_preview_layers(value),
                    )
                )

    dae_members = [member for member in members if member.lower().endswith(".dae")]
    dae_members.sort(
        key=lambda member: (
            -sizes.get(member, 0),
            member.count("/"),
            member.lower(),
        )
    )
    if not dae_members:
        raise ValueError("The archive contains no .dae files.")

    return VehicleArchive(
        path=path,
        members=tuple(members),
        member_by_lower=member_by_lower,
        member_sizes=sizes,
        dae_members=tuple(dae_members),
        materials=tuple(materials),
        workspace=workspace,
        material_errors=tuple(errors),
    )


def extract_archive_member(archive: VehicleArchive, member: str) -> Path:
    """Extract one indexed member into the archive's temporary workspace."""
    normalised = _normalise_archive_member(member)
    actual = archive.member_by_lower.get(normalised.lower())
    if actual is None:
        raise FileNotFoundError(f"ZIP member not found: {member}")
    target = archive.workspace.joinpath(*PurePosixPath(actual).parts)
    target.resolve().relative_to(archive.workspace.resolve())
    if target.is_file() and target.stat().st_size == archive.member_sizes.get(actual, -1):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive.path, "r") as source, source.open(actual, "r") as handle:
        with target.open("wb") as output:
            shutil.copyfileobj(handle, output, length=1024 * 1024)
    return target


def _texture_reference_candidates(reference: str, materials_member: str) -> tuple[str, ...]:
    reference = reference.replace("\\", "/").strip()
    if not reference or reference.startswith("@"):
        return ()
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.lstrip("/")
        try:
            value = _normalise_archive_member(value)
        except ValueError:
            return
        if value not in candidates:
            candidates.append(value)

    add(reference)
    if not reference.startswith("/"):
        parent = PurePosixPath(materials_member).parent
        add((parent / reference).as_posix())

    expanded = list(candidates)
    for candidate in expanded:
        path = PurePosixPath(candidate)
        suffix = path.suffix.lower()
        if suffix in {".png", ".dds", ".tga", ".jpg", ".jpeg"}:
            stem = candidate[: -len(path.suffix)]
            for replacement in (".dds", ".png", ".tga", ".jpg", ".jpeg"):
                add(stem + replacement)
    return tuple(candidates)


def resolve_archive_texture_member(
    archive: VehicleArchive,
    reference: str,
    materials_member: str,
) -> str | None:
    """Resolve a BeamNG virtual texture path, including PNG-to-DDS aliases."""
    candidates = _texture_reference_candidates(reference, materials_member)
    for candidate in candidates:
        exact = archive.member_by_lower.get(candidate.lower())
        if exact is not None:
            return exact

    # Official archives and downloaded mods may wrap vehicles/ in one extra root
    # directory. A suffix match preserves the BeamNG virtual path in that case.
    for candidate in candidates:
        suffix = "/" + candidate.lower()
        matches = [member for member in archive.members if ("/" + member.lower()).endswith(suffix)]
        if matches:
            matches.sort(key=lambda value: (value.count("/"), len(value), value.lower()))
            return matches[0]
    return None


def _blend_archive_preview_texture(
    base_texture: Path,
    archive: VehicleArchive,
    binding: ArchiveTextureBinding,
) -> Path:
    """Bake simple BeamNG layered material stages into one Blender texture."""
    if not binding.preview_layers:
        return base_texture
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Baking layered BeamNG materials for Blender requires Pillow. "
            "Install it with: python -m pip install Pillow"
        ) from exc

    with Image.open(base_texture) as base_image:
        base_rgba = base_image.convert("RGBA")
    base_array = np.asarray(base_rgba, dtype=np.float32) / 255.0
    composite = base_array[:, :, :3].copy()
    alpha_channel = np.asarray(base_rgba, dtype=np.uint8)[:, :, 3]

    applied_layers: list[ArchiveMaterialPreviewLayer] = []
    for layer in binding.preview_layers:
        layer_member = resolve_archive_texture_member(
            archive,
            layer.base_colour_reference,
            binding.materials_member,
        )
        opacity_member = resolve_archive_texture_member(
            archive,
            layer.opacity_reference,
            binding.materials_member,
        )
        if layer_member is None or opacity_member is None:
            continue

        layer_texture = extract_archive_member(archive, layer_member)
        opacity_texture = extract_archive_member(archive, opacity_member)
        with Image.open(layer_texture) as layer_image:
            layer_rgba = layer_image.convert("RGBA")
            if layer_rgba.size != base_rgba.size:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                layer_rgba = layer_rgba.resize(base_rgba.size, resampling)
        with Image.open(opacity_texture) as opacity_image:
            opacity_luma = opacity_image.convert("L")
            if opacity_luma.size != base_rgba.size:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                opacity_luma = opacity_luma.resize(base_rgba.size, resampling)

        layer_array = np.asarray(layer_rgba, dtype=np.float32)[:, :, :3] / 255.0
        mask = np.asarray(opacity_luma, dtype=np.float32)[:, :, None] / 255.0
        factor = np.asarray(layer.base_colour_factor[:3], dtype=np.float32)
        mask *= float(layer.base_colour_factor[3])
        composite = composite * (1.0 - mask) + (layer_array * factor) * mask
        applied_layers.append(layer)

    if not applied_layers:
        return base_texture

    output_rgb = np.clip(composite * 255.0, 0.0, 255.0).astype(np.uint8)
    output_array = np.dstack([output_rgb, alpha_channel])
    output_dir = archive.workspace / "__beamxp_preview_textures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{safe_name(binding.material_key)}__beamxp_preview.png"
    Image.fromarray(output_array).save(output)
    return output


def _normalise_material_alias(value: str) -> str:
    value = value.strip().lstrip("#").lower()
    for suffix in ("-material", "_material"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def _material_alias_lookup_keys(value: str) -> tuple[str, ...]:
    """Return exact and BeamNG skin-variant lookup keys for one material alias."""
    normalised = _normalise_material_alias(value)
    if not normalised:
        return ()
    keys = [normalised]
    skin_marker = next(
        (
            index
            for index in range(len(normalised))
            if normalised.startswith(".skin", index)
            and (index + len(".skin") == len(normalised) or normalised[index + len(".skin")] in "._")
        ),
        -1,
    )
    if skin_marker > 0:
        keys.append(normalised[:skin_marker])
    return tuple(dict.fromkeys(keys))


def _material_alias_match_rank(
    record: ArchiveMaterialRecord,
    normalised_dae_material: str,
) -> int:
    for alias in record.aliases:
        normalised = _normalise_material_alias(alias)
        if normalised == normalised_dae_material:
            return 0
    for alias in record.aliases:
        if normalised_dae_material in _material_alias_lookup_keys(alias):
            return 1
    return 2


def _preview_layer_signature(
    layers: tuple[ArchiveMaterialPreviewLayer, ...],
) -> tuple[tuple[str, tuple[float, float, float, float], str], ...]:
    return tuple(
        (
            layer.base_colour_reference.lower(),
            layer.base_colour_factor,
            layer.opacity_reference.lower(),
        )
        for layer in layers
    )


def material_names_for_part(loaded: LoadedDae, part: DaePart) -> tuple[str, ...]:
    """Return COLLADA material names/symbols bound to the selected scene node."""
    root = loaded.tree.getroot()
    namespace = loaded.namespace
    target = _find_target_node(root, namespace, part)
    materials_library = root.find(f"./{qname(namespace, 'library_materials')}")
    material_names: dict[str, str] = {}
    if materials_library is not None:
        for material in materials_library.findall(qname(namespace, "material")):
            material_id = (material.get("id") or "").strip()
            material_name = (material.get("name") or material_id).strip()
            if material_id:
                material_names[material_id] = material_name

    names: list[str] = []
    for instance_geometry in target.findall(qname(namespace, "instance_geometry")):
        for instance_material in instance_geometry.iter(qname(namespace, "instance_material")):
            symbol = (instance_material.get("symbol") or "").strip()
            target_id = (instance_material.get("target") or "").strip().lstrip("#")
            for value in (material_names.get(target_id, ""), target_id, symbol):
                value = value.strip()
                if value and value not in names:
                    names.append(value)
    return tuple(names)


def archive_texture_choices_for_part(
    archive: VehicleArchive,
    loaded: LoadedDae,
    part: DaePart,
) -> tuple[ArchiveTextureBinding, ...]:
    """Return every materials-JSON base-colour asset valid for one DAE part.

    A choice is admitted only when a material record's key, name or ``mapTo``
    alias matches a COLLADA material actually bound to the selected part, and
    the referenced texture resolves to a real archive member.  No filename-only
    guessing is used.
    """
    requested = material_names_for_part(loaded, part)
    records_by_alias: dict[str, list[ArchiveMaterialRecord]] = defaultdict(list)
    for record in archive.materials:
        record_keys: set[str] = set()
        for alias in record.aliases:
            record_keys.update(_material_alias_lookup_keys(alias))
        for key in record_keys:
            records_by_alias[key].append(record)

    choices: list[ArchiveTextureBinding] = []
    used_members: set[
        tuple[str, str, tuple[tuple[str, tuple[float, float, float, float], str], ...]]
    ] = set()
    for dae_material in requested:
        normalised_dae_material = _normalise_material_alias(dae_material)
        records = sorted(
            records_by_alias.get(normalised_dae_material, ()),
            key=lambda record: (
                _material_alias_match_rank(record, normalised_dae_material),
                0 if _normalise_material_alias(record.map_to) == normalised_dae_material else 1,
                record.materials_member.count("/"),
                record.materials_member.lower(),
                record.key.lower(),
            ),
        )
        for record in records:
            texture_member = resolve_archive_texture_member(
                archive,
                record.base_colour_reference,
                record.materials_member,
            )
            if texture_member is None:
                continue
            signature = (
                normalised_dae_material,
                texture_member.lower(),
                _preview_layer_signature(record.preview_layers),
            )
            if signature in used_members:
                continue
            used_members.add(signature)
            choices.append(
                ArchiveTextureBinding(
                    dae_material=dae_material,
                    material_key=record.key,
                    materials_member=record.materials_member,
                    texture_reference=record.base_colour_reference,
                    texture_member=texture_member,
                    preview_layers=record.preview_layers,
                )
            )
    return tuple(choices)


def archive_texture_bindings_for_part(
    archive: VehicleArchive,
    loaded: LoadedDae,
    part: DaePart,
) -> tuple[ArchiveTextureBinding, ...]:
    """Return the first materials-JSON choice for each material on the part."""
    defaults: list[ArchiveTextureBinding] = []
    seen_materials: set[str] = set()
    for binding in archive_texture_choices_for_part(archive, loaded, part):
        key = _normalise_material_alias(binding.dae_material)
        if key in seen_materials:
            continue
        seen_materials.add(key)
        defaults.append(binding)
    return tuple(defaults)


def parse_float_list(text: str | None) -> list[float]:
    return [float(value) for value in (text or "").split()]


def parse_int_list(text: str | None) -> list[int]:
    return [int(value) for value in (text or "").split()]


def parse_matrix(text: str | None) -> np.ndarray:
    values = parse_float_list(text)
    if len(values) != 16:
        return np.eye(4, dtype=float)
    # Preserve the matrix layout used by the Blender-authored source DAE and
    # by the pre-refactor implementation.  Its XML values are consumed in
    # ordinary row-major sequence by the rest of this program.
    return np.asarray(values, dtype=float).reshape((4, 4))


def rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-15:
        return np.eye(4, dtype=float)
    x, y, z = axis / norm
    angle = math.radians(angle_degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    one = 1.0 - c
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=float,
    )
    return out


def node_local_matrix(node: ET.Element, namespace: str) -> np.ndarray:
    result = np.eye(4, dtype=float)
    for child in list(node):
        tag = local_name(child.tag)
        values = parse_float_list(child.text)
        transform: np.ndarray | None = None
        if tag == "matrix" and len(values) == 16:
            transform = parse_matrix(child.text)
        elif tag == "translate" and len(values) >= 3:
            transform = np.eye(4, dtype=float)
            transform[:3, 3] = values[:3]
        elif tag == "scale" and len(values) >= 3:
            transform = np.diag([values[0], values[1], values[2], 1.0])
        elif tag == "rotate" and len(values) >= 4:
            transform = rotation_matrix(np.asarray(values[:3], dtype=float), values[3])
        if transform is not None:
            result = result @ transform
    return result


def dae_unit_scale(root: ET.Element, namespace: str) -> float:
    unit = root.find(f"./{qname(namespace, 'asset')}/{qname(namespace, 'unit')}")
    if unit is None:
        return 1.0
    try:
        value = float(unit.get("meter", "1"))
    except ValueError:
        return 1.0
    return value if value > 0.0 else 1.0


def load_dae(path: Path) -> LoadedDae:
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = namespace_uri(root)
    geometry_tag = qname(namespace, "geometry")
    geometries = {
        geometry.get("id", ""): geometry
        for geometry in root.iter(geometry_tag)
        if geometry.get("id")
    }

    parts: list[DaePart] = []
    used_keys: set[str] = set()
    node_tag = qname(namespace, "node")
    instance_geometry_tag = qname(namespace, "instance_geometry")
    visual_scenes = root.findall(
        f"./{qname(namespace, 'library_visual_scenes')}/{qname(namespace, 'visual_scene')}"
    )

    def visit(node: ET.Element, parent_matrix: np.ndarray, path_bits: tuple[str, ...]) -> None:
        local = node_local_matrix(node, namespace)
        world = parent_matrix @ local
        node_id = (node.get("id") or "").strip()
        node_name = (node.get("name") or node_id or "unnamed").strip()
        instances: list[GeometryInstance] = []
        for instance in node.findall(instance_geometry_tag):
            url = instance.get("url", "")
            if url.startswith("#") and url[1:] in geometries:
                instances.append(GeometryInstance(url[1:]))

        current_path = (*path_bits, node_name)
        if instances:
            base_key = node_id or "/".join(current_path)
            key = base_key
            suffix = 2
            while key in used_keys:
                key = f"{base_key}#{suffix}"
                suffix += 1
            used_keys.add(key)
            geometry_names = ", ".join(instance.geometry_id for instance in instances)
            label = (
                f"{node_name}  [{node_id}]  —  {geometry_names}"
                if node_id and node_name != node_id
                else f"{node_name}  —  {geometry_names}"
            )
            parts.append(
                DaePart(
                    key=key,
                    label=label,
                    node_id=node_id,
                    node_name=node_name,
                    matrix=world,
                    instances=tuple(instances),
                )
            )

        for child in node.findall(node_tag):
            visit(child, world, current_path)

    for visual_scene in visual_scenes:
        for node in visual_scene.findall(node_tag):
            visit(node, np.eye(4, dtype=float), ())

    if not parts:
        for node in root.iter(node_tag):
            instances: list[GeometryInstance] = []
            for instance in node.findall(instance_geometry_tag):
                url = instance.get("url", "")
                if url.startswith("#") and url[1:] in geometries:
                    instances.append(GeometryInstance(url[1:]))
            if not instances:
                continue
            node_id = (node.get("id") or "").strip()
            node_name = (node.get("name") or node_id or "unnamed").strip()
            key = node_id or node_name
            while key in used_keys:
                key += "#"
            used_keys.add(key)
            parts.append(
                DaePart(
                    key=key,
                    label=f"{node_name}  —  {', '.join(i.geometry_id for i in instances)}",
                    node_id=node_id,
                    node_name=node_name,
                    matrix=node_local_matrix(node, namespace),
                    instances=tuple(instances),
                )
            )

    parts.sort(key=lambda part: (part.node_name.lower(), part.node_id.lower(), part.key.lower()))
    return LoadedDae(
        path=path,
        tree=tree,
        namespace=namespace,
        unit_scale=dae_unit_scale(root, namespace),
        parts=parts,
        geometries=geometries,
    )


def source_float_matrix(source: ET.Element, namespace: str) -> np.ndarray:
    float_array = source.find(qname(namespace, "float_array"))
    accessor = source.find(
        f"./{qname(namespace, 'technique_common')}/{qname(namespace, 'accessor')}"
    )
    if float_array is None or accessor is None:
        return np.empty((0, 3), dtype=float)
    values = np.asarray(parse_float_list(float_array.text), dtype=float)
    stride = max(1, int(accessor.get("stride", "1")))
    offset = max(0, int(accessor.get("offset", "0")))
    count = int(accessor.get("count", str(max(0, (len(values) - offset) // stride))))
    usable = values[offset : offset + count * stride]
    if len(usable) < count * stride:
        count = len(usable) // stride
        usable = usable[: count * stride]
    return usable.reshape((count, stride)) if count else np.empty((0, stride), dtype=float)


def _triangulate_rows(rows: np.ndarray) -> list[np.ndarray]:
    if len(rows) < 3:
        return []
    return [np.stack((rows[0], rows[index], rows[index + 1])) for index in range(1, len(rows) - 1)]


def parse_geometry(loaded: LoadedDae, geometry_id: str) -> RawGeometry:
    cached = loaded.geometry_cache.get(geometry_id)
    if cached is not None:
        return cached

    geometry = loaded.geometries[geometry_id]
    namespace = loaded.namespace
    mesh = geometry.find(qname(namespace, "mesh"))
    if mesh is None:
        result = RawGeometry(
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=np.int64),
            (),
            (),
        )
        loaded.geometry_cache[geometry_id] = result
        return result

    sources = {
        source.get("id", ""): source
        for source in mesh.findall(qname(namespace, "source"))
        if source.get("id")
    }
    position_sources: dict[str, np.ndarray] = {}
    for source_id, source in sources.items():
        matrix = source_float_matrix(source, namespace)
        if matrix.shape[1] >= 3:
            position_sources[source_id] = matrix[:, :3].copy()

    vertices_to_position: dict[str, str] = {}
    for vertices in mesh.findall(qname(namespace, "vertices")):
        vertices_id = vertices.get("id", "")
        for input_element in vertices.findall(qname(namespace, "input")):
            if input_element.get("semantic") == "POSITION":
                source_url = input_element.get("source", "")
                if source_url.startswith("#"):
                    vertices_to_position[vertices_id] = source_url[1:]

    source_offsets: dict[str, int] = {}
    combined_vertices: list[np.ndarray] = []
    combined_triangles: list[tuple[int, int, int]] = []
    primitives: list[PrimitiveTriangles] = []
    face_sources: list[tuple[int, int]] = []

    def ensure_source(source_id: str) -> int:
        if source_id not in source_offsets:
            source_offsets[source_id] = sum(len(chunk) for chunk in combined_vertices)
            combined_vertices.append(position_sources.get(source_id, np.empty((0, 3), dtype=float)))
        return source_offsets[source_id]

    for primitive in list(mesh):
        tag = local_name(primitive.tag)
        if tag not in PRIMITIVE_TAGS:
            continue
        inputs = primitive.findall(qname(namespace, "input"))
        if not inputs:
            continue
        stride = max(int(input_element.get("offset", "0")) for input_element in inputs) + 1
        position_offset: int | None = None
        position_source_id: str | None = None
        for input_element in inputs:
            semantic = input_element.get("semantic")
            source_url = input_element.get("source", "")
            if not source_url.startswith("#"):
                continue
            source_id = source_url[1:]
            if semantic == "VERTEX":
                position_source_id = vertices_to_position.get(source_id)
                position_offset = int(input_element.get("offset", "0"))
                break
            if semantic == "POSITION":
                position_source_id = source_id
                position_offset = int(input_element.get("offset", "0"))
                break
        if position_source_id is None or position_offset is None:
            continue
        positions = position_sources.get(position_source_id)
        if positions is None or len(positions) == 0:
            continue
        vertex_base = ensure_source(position_source_id)

        def rows_from_p(p_element: ET.Element) -> np.ndarray:
            values = np.asarray(parse_int_list(p_element.text), dtype=np.int64)
            if len(values) < stride:
                return np.empty((0, stride), dtype=np.int64)
            return values[: len(values) - (len(values) % stride)].reshape((-1, stride))

        triangle_rows: list[np.ndarray] = []
        if tag == "triangles":
            for p_element in primitive.findall(qname(namespace, "p")):
                rows = rows_from_p(p_element)
                count = len(rows) - (len(rows) % 3)
                for start in range(0, count, 3):
                    triangle_rows.append(rows[start : start + 3].copy())
        elif tag == "polylist":
            p_element = primitive.find(qname(namespace, "p"))
            vcount_element = primitive.find(qname(namespace, "vcount"))
            if p_element is not None and vcount_element is not None:
                rows = rows_from_p(p_element)
                cursor = 0
                for count in parse_int_list(vcount_element.text):
                    triangle_rows.extend(_triangulate_rows(rows[cursor : cursor + count]))
                    cursor += count
        elif tag in {"polygons", "trifans"}:
            for p_element in primitive.findall(qname(namespace, "p")):
                triangle_rows.extend(_triangulate_rows(rows_from_p(p_element)))
        elif tag == "tristrips":
            for p_element in primitive.findall(qname(namespace, "p")):
                rows = rows_from_p(p_element)
                for index in range(len(rows) - 2):
                    if index % 2:
                        triangle_rows.append(np.stack((rows[index + 1], rows[index], rows[index + 2])))
                    else:
                        triangle_rows.append(np.stack((rows[index], rows[index + 1], rows[index + 2])))

        if not triangle_rows:
            continue
        primitive_index = len(primitives)
        primitive_array = np.asarray(triangle_rows, dtype=np.int64).reshape((-1, 3, stride))
        primitives.append(
            PrimitiveTriangles(
                attributes={key: value for key, value in primitive.attrib.items() if key != "count"},
                input_attributes=tuple(dict(input_element.attrib) for input_element in inputs),
                rows=primitive_array,
            )
        )
        for local_triangle, rows in enumerate(primitive_array):
            combined_triangles.append(
                tuple(int(rows[corner, position_offset]) + vertex_base for corner in range(3))
            )
            face_sources.append((primitive_index, local_triangle))

    vertices_array = (
        np.concatenate(combined_vertices, axis=0)
        if combined_vertices
        else np.empty((0, 3), dtype=float)
    )
    triangles_array = np.asarray(combined_triangles, dtype=np.int64).reshape((-1, 3))
    result = RawGeometry(vertices_array, triangles_array, tuple(primitives), tuple(face_sources))
    loaded.geometry_cache[geometry_id] = result
    return result


def weld_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    source_faces: list[SourceFaceRef],
) -> tuple[np.ndarray, np.ndarray, list[SourceFaceRef]]:
    if len(vertices) == 0:
        return vertices, triangles, source_faces
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    tolerance = max(diagonal * 1e-9, 1e-11)
    inverse = 1.0 / tolerance
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    representatives: list[np.ndarray] = []
    remap = np.empty(len(vertices), dtype=np.int64)

    for old_index, point in enumerate(vertices):
        key = tuple(int(round(value * inverse)) for value in point)
        found: int | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if float(np.linalg.norm(representatives[candidate] - point)) <= tolerance:
                            found = candidate
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            found = len(representatives)
            representatives.append(point.copy())
            buckets[key].append(found)
        remap[old_index] = found

    welded = remap[triangles]
    valid = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 2] != welded[:, 0])
    )
    return (
        np.asarray(representatives, dtype=float),
        welded[valid],
        [source for source, keep in zip(source_faces, valid) if bool(keep)],
    )


def mesh_for_part(
    loaded: LoadedDae, part: DaePart
) -> tuple[np.ndarray, np.ndarray, list[SourceFaceRef]]:
    vertex_chunks: list[np.ndarray] = []
    triangle_chunks: list[np.ndarray] = []
    source_faces: list[SourceFaceRef] = []
    vertex_base = 0
    for instance_index, instance in enumerate(part.instances):
        raw = parse_geometry(loaded, instance.geometry_id)
        if len(raw.vertices) == 0 or len(raw.triangles) == 0:
            continue
        homogeneous = np.concatenate(
            [raw.vertices, np.ones((len(raw.vertices), 1), dtype=float)], axis=1
        )
        transformed = (homogeneous @ part.matrix.T)[:, :3] * loaded.unit_scale
        vertex_chunks.append(transformed)
        triangle_chunks.append(raw.triangles + vertex_base)
        source_faces.extend(
            SourceFaceRef(instance_index, instance.geometry_id, primitive_index, triangle_index)
            for primitive_index, triangle_index in raw.face_sources
        )
        vertex_base += len(raw.vertices)
    if not vertex_chunks:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=np.int64),
            [],
        )
    return weld_vertices(
        np.concatenate(vertex_chunks, axis=0),
        np.concatenate(triangle_chunks, axis=0),
        source_faces,
    )


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def canonical_edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def build_topology(
    vertices: np.ndarray,
    triangles: np.ndarray,
    source_faces: list[SourceFaceRef],
) -> Topology:
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("The selected DAE part contains no triangle surface geometry.")

    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    double_areas = np.linalg.norm(cross, axis=1)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    valid = double_areas > max(diagonal * diagonal * 1e-16, 1e-24)
    triangles = triangles[valid]
    cross = cross[valid]
    double_areas = double_areas[valid]
    source_faces = [source for source, keep in zip(source_faces, valid) if bool(keep)]
    if len(triangles) == 0:
        raise ValueError("Every triangle in the selected part is degenerate.")

    face_normals = cross / double_areas[:, None]
    face_areas = double_areas * 0.5
    edge_faces_lists: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, triangle in enumerate(triangles):
        a, b, c = (int(value) for value in triangle)
        edge_faces_lists[canonical_edge(a, b)].append(face_index)
        edge_faces_lists[canonical_edge(b, c)].append(face_index)
        edge_faces_lists[canonical_edge(c, a)].append(face_index)
    edge_faces = {edge: tuple(faces) for edge, faces in edge_faces_lists.items()}
    boundary_edges = {edge for edge, faces in edge_faces.items() if len(faces) == 1}
    nonmanifold_edges = {edge for edge, faces in edge_faces.items() if len(faces) > 2}
    edge_angles: dict[tuple[int, int], float] = {}
    for edge, faces in edge_faces.items():
        if len(faces) == 2:
            dot = float(np.clip(np.dot(face_normals[faces[0]], face_normals[faces[1]]), -1.0, 1.0))
            edge_angles[edge] = math.degrees(math.acos(dot))

    return Topology(
        vertices=vertices,
        triangles=triangles,
        source_faces=tuple(source_faces),
        face_normals=face_normals,
        face_areas=face_areas,
        edge_faces=edge_faces,
        edge_angles=edge_angles,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        diagonal=max(diagonal, 1e-12),
    )


def topology_for_part(loaded: LoadedDae, part: DaePart) -> Topology:
    cached = loaded.topology_cache.get(part.key)
    if cached is not None:
        return cached
    topology = build_topology(*mesh_for_part(loaded, part))
    loaded.topology_cache[part.key] = topology
    return topology


def geometric_face_islands(topology: Topology) -> list[tuple[int, ...]]:
    adjacency: list[set[int]] = [set() for _ in range(len(topology.triangles))]
    for faces in topology.edge_faces.values():
        if len(faces) < 2:
            continue
        anchor = faces[0]
        for other in faces[1:]:
            adjacency[anchor].add(other)
            adjacency[other].add(anchor)
    islands: list[tuple[int, ...]] = []
    remaining = set(range(len(topology.triangles)))
    while remaining:
        seed = next(iter(remaining))
        found = {seed}
        pending = [seed]
        while pending:
            face = pending.pop()
            for neighbour in adjacency[face]:
                if neighbour not in found:
                    found.add(neighbour)
                    pending.append(neighbour)
        remaining.difference_update(found)
        islands.append(tuple(sorted(found)))
    islands.sort(
        key=lambda faces: (
            -float(topology.face_areas[list(faces)].sum()),
            -len(faces),
            faces[0],
        )
    )
    return islands


def inclusive_thresholds(maximum: float, minimum: float, steps: int) -> tuple[float, ...]:
    if steps <= 1:
        return (float(maximum),)
    return tuple(float(value) for value in np.linspace(maximum, minimum, steps))


# ---------------------------------------------------------------------------
# Perimeter extraction and deterministic symmetry test
# ---------------------------------------------------------------------------


def ordered_boundary_loops(
    edges: Iterable[tuple[int, int]],
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    edge_set = {canonical_edge(first, second) for first, second in edges}
    if not edge_set:
        return (), True
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edge_set:
        adjacency[first].add(second)
        adjacency[second].add(first)
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        return (), False

    unused = set(edge_set)
    loops: list[tuple[int, ...]] = []
    while unused:
        first_edge = min(unused)
        start, following = first_edge
        loop = [start]
        previous = start
        current = following
        unused.discard(first_edge)
        for _ in range(len(edge_set) + 2):
            if current == start:
                break
            loop.append(current)
            candidates = sorted(adjacency[current] - {previous})
            if len(candidates) != 1:
                return (), False
            next_vertex = candidates[0]
            edge = canonical_edge(current, next_vertex)
            if edge not in unused and next_vertex != start:
                return (), False
            unused.discard(edge)
            previous, current = current, next_vertex
        else:
            return (), False
        if current != start or len(loop) < 3:
            return (), False
        loops.append(tuple(loop))
    loops.sort(key=lambda loop: (-len(loop), loop[0]))
    return tuple(loops), True


@dataclass(slots=True)
class ClosedPolyline:
    points: np.ndarray
    lengths: np.ndarray
    cumulative: np.ndarray
    total_length: float


def closed_polyline(vertices: np.ndarray, loop: tuple[int, ...]) -> ClosedPolyline:
    points = vertices[np.asarray(loop, dtype=np.int64)]
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return ClosedPolyline(points, lengths, cumulative, float(lengths.sum()))


def sample_closed_polyline(polyline: ClosedPolyline, distances: np.ndarray) -> np.ndarray:
    if len(distances) == 0:
        return np.empty((0, 3), dtype=float)
    if polyline.total_length <= 1e-15 or len(polyline.points) == 0:
        return np.repeat(polyline.points[:1], len(distances), axis=0)
    targets = np.mod(np.asarray(distances, dtype=float), polyline.total_length)
    indices = np.searchsorted(polyline.cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(polyline.points) - 1)
    local = targets - polyline.cumulative[indices]
    fractions = np.divide(
        local,
        polyline.lengths[indices],
        out=np.zeros_like(local),
        where=polyline.lengths[indices] > 0,
    )
    following = polyline.points[(indices + 1) % len(polyline.points)]
    return polyline.points[indices] + (following - polyline.points[indices]) * fractions[:, None]


def resample_closed_loop_at_rate(polyline: ClosedPolyline, spacing: float) -> np.ndarray:
    if polyline.total_length <= 1e-15:
        return np.repeat(polyline.points[:1], 1, axis=0)
    count = max(8, int(math.ceil(polyline.total_length / max(spacing, 1e-12))))
    targets = np.linspace(0.0, polyline.total_length, count, endpoint=False)
    return sample_closed_polyline(polyline, targets)


def mirror_plane_crossings(
    polyline: ClosedPolyline,
    centroid: np.ndarray,
    normal: np.ndarray,
    spacing: float,
) -> tuple[float, ...] | None:
    """Return exact arc-length positions where a closed loop crosses the plane.

    Vertices lying on the plane are de-duplicated. A perimeter segment lying in
    the mirror plane is ambiguous for the opposite-direction sibling walk and is
    rejected rather than guessed through.
    """
    if polyline.total_length <= 1e-15:
        return ()
    signed = (polyline.points - centroid) @ normal
    distance_epsilon = max(1e-10, spacing * 1e-6, polyline.total_length * 1e-10)
    arc_epsilon = max(1e-10, spacing * 1e-5, polyline.total_length * 1e-10)
    crossings: list[float] = []
    count = len(polyline.points)
    for index in range(count):
        following = (index + 1) % count
        first_distance = float(signed[index])
        second_distance = float(signed[following])
        first_on = abs(first_distance) <= distance_epsilon
        second_on = abs(second_distance) <= distance_epsilon
        if first_on and second_on and polyline.lengths[index] > distance_epsilon:
            return None
        if first_on:
            crossings.append(float(polyline.cumulative[index]))
            continue
        if first_distance * second_distance < 0.0:
            fraction = first_distance / (first_distance - second_distance)
            crossings.append(
                float(polyline.cumulative[index] + fraction * polyline.lengths[index])
            )

    if not crossings:
        return ()
    positions = sorted(value % polyline.total_length for value in crossings)
    merged: list[float] = []
    for value in positions:
        if not merged or value - merged[-1] > arc_epsilon:
            merged.append(value)
        else:
            merged[-1] = 0.5 * (merged[-1] + value)
    if len(merged) > 1 and polyline.total_length - merged[-1] + merged[0] <= arc_epsilon:
        wrapped = ((merged[-1] - polyline.total_length) + merged[0]) * 0.5
        merged[0] = wrapped % polyline.total_length
        merged.pop()
        merged.sort()
    return tuple(merged)


def _failed_symmetry_measurement(
    centroid: np.ndarray,
    plane_normal: np.ndarray,
    whole_samples: int,
    crossings: int,
) -> SymmetryMeasurement:
    normal_tuple = tuple(float(value) for value in plane_normal)
    return SymmetryMeasurement(
        centroid=tuple(float(value) for value in centroid),
        plane_normal=normal_tuple,
        post_reflection_variance=float("inf"),
        rms_error=float("inf"),
        max_error=float("inf"),
        initial_plane_normal=normal_tuple,
        initial_post_reflection_variance=float("inf"),
        initial_rms_error=float("inf"),
        initial_max_error=float("inf"),
        mirror_plane_y_tilt_degrees=0.0,
        rigid_y_rotation_correction_degrees=0.0,
        tilt_search_applied=False,
        whole_loop_sample_count=whole_samples,
        sibling_pair_count=0,
        mirror_crossings=crossings,
        passed=False,
    )

def measure_perimeter_symmetry(
    vertices: np.ndarray,
    loops: tuple[tuple[int, ...], ...],
    sample_spacing: float,
    rms_tolerance: float,
    direct_rms_tolerance: float,
) -> SymmetryMeasurement:
    """Measure perimeter symmetry and optionally correct a shallow Y-axis tilt.

    The ordinary deterministic test is unchanged:

    1. Resample each complete closed perimeter at a fixed physical spacing.
    2. Use the samples for the centroid and PCA perimeter-plane normal.
    3. Cross that normal with global Z to obtain the initial mirror-plane normal,
       with the existing near-horizontal fallback search.
    4. Split each loop at exact mirror-plane crossings, traverse opposite halves
       in opposite directions, and calculate the full-3D sibling RMS residual.

    Candidates below ``direct_rms_tolerance`` use the initial plane unchanged.
    Candidates between ``direct_rms_tolerance`` and ``rms_tolerance`` receive a
    small bounded search in which the initial mirror-plane normal is rotated
    about global Y. The plane always passes through the original centroid. The
    best plane normal is stored on the measurement, so the existing composition
    of global-X reflection with local-plane reflection automatically applies the
    corresponding double-angle rigid correction to the exported submesh.

    Candidates above ``rms_tolerance`` are rejected and are never rescued by the
    tilt search.
    """
    polylines = [closed_polyline(vertices, loop) for loop in loops]
    whole_sample_chunks = [
        resample_closed_loop_at_rate(polyline, sample_spacing)
        for polyline in polylines
        if polyline.total_length > 1e-15
    ]

    if not whole_sample_chunks:
        origin = np.zeros(3, dtype=float)
        return _failed_symmetry_measurement(
            origin,
            np.array([1.0, 0.0, 0.0], dtype=float),
            0,
            0,
        )

    whole_samples = np.concatenate(whole_sample_chunks, axis=0)
    centroid = whole_samples.mean(axis=0)
    centred = whole_samples - centroid
    covariance = centred.T @ centred / max(1, len(whole_samples))
    _, eigenvectors = np.linalg.eigh(covariance)
    surface_normal = eigenvectors[:, 0]

    def make_measurement(
        mirror_normal: np.ndarray,
        squared_residuals: np.ndarray,
        sibling_pairs: int,
        crossing_total: int,
    ) -> SymmetryMeasurement:
        post_variance = float(np.mean(squared_residuals))
        rms_error = float(math.sqrt(post_variance))
        max_error = float(math.sqrt(float(squared_residuals.max(initial=0.0))))
        normal_tuple = tuple(float(value) for value in mirror_normal)
        return SymmetryMeasurement(
            centroid=tuple(float(value) for value in centroid),
            plane_normal=normal_tuple,
            post_reflection_variance=post_variance,
            rms_error=rms_error,
            max_error=max_error,
            initial_plane_normal=normal_tuple,
            initial_post_reflection_variance=post_variance,
            initial_rms_error=rms_error,
            initial_max_error=max_error,
            mirror_plane_y_tilt_degrees=0.0,
            rigid_y_rotation_correction_degrees=0.0,
            tilt_search_applied=False,
            whole_loop_sample_count=len(whole_samples),
            sibling_pair_count=sibling_pairs,
            mirror_crossings=crossing_total,
            passed=False,
        )

    def evaluate_mirror_normal(
        proposed_normal: np.ndarray,
    ) -> SymmetryMeasurement:
        """Evaluate one mirror plane through the fixed perimeter centroid."""
        mirror_normal = np.asarray(proposed_normal, dtype=float)
        normal_length = float(np.linalg.norm(mirror_normal))

        if normal_length <= 1e-12:
            return _failed_symmetry_measurement(
                centroid,
                np.array([1.0, 0.0, 0.0], dtype=float),
                len(whole_samples),
                0,
            )

        mirror_normal = mirror_normal / normal_length
        squared_residual_chunks: list[np.ndarray] = []
        sibling_pairs = 0
        crossing_total = 0

        for polyline in polylines:
            crossings = mirror_plane_crossings(
                polyline,
                centroid,
                mirror_normal,
                sample_spacing,
            )

            if crossings is None:
                return _failed_symmetry_measurement(
                    centroid,
                    mirror_normal,
                    len(whole_samples),
                    crossing_total,
                )

            crossing_total += len(crossings)
            if len(crossings) != 2:
                return _failed_symmetry_measurement(
                    centroid,
                    mirror_normal,
                    len(whole_samples),
                    crossing_total,
                )

            first, second = sorted(crossings)
            forward_length = second - first
            reverse_length = polyline.total_length - forward_length
            longest_half = max(forward_length, reverse_length)

            interval_count = max(
                1,
                int(math.ceil(longest_half / max(sample_spacing, 1e-12))),
            )
            travelled = np.linspace(0.0, longest_half, interval_count + 1)

            forward_distances = first + np.minimum(travelled, forward_length)
            reverse_distances = first - np.minimum(travelled, reverse_length)

            forward_points = sample_closed_polyline(polyline, forward_distances)
            reverse_points = sample_closed_polyline(polyline, reverse_distances)

            reflected = forward_points - 2.0 * (
                (forward_points - centroid) @ mirror_normal
            )[:, None] * mirror_normal[None, :]

            differences = reflected - reverse_points
            squared_residual_chunks.append(
                np.einsum("ij,ij->i", differences, differences)
            )
            sibling_pairs += len(differences)

        if not squared_residual_chunks:
            return _failed_symmetry_measurement(
                centroid,
                mirror_normal,
                len(whole_samples),
                crossing_total,
            )

        return make_measurement(
            mirror_normal,
            np.concatenate(squared_residual_chunks),
            sibling_pairs,
            crossing_total,
        )

    def result_is_better(
        candidate_result: SymmetryMeasurement,
        candidate_angle_degrees: float,
        best_result: SymmetryMeasurement | None,
        best_angle_degrees: float,
    ) -> bool:
        if best_result is None:
            return True
        candidate_variance = candidate_result.post_reflection_variance
        best_variance = best_result.post_reflection_variance
        if candidate_variance < best_variance - 1e-18:
            return True
        if candidate_variance > best_variance + 1e-18:
            return False
        if abs(candidate_angle_degrees) < abs(best_angle_degrees) - 1e-12:
            return True
        if abs(candidate_angle_degrees) > abs(best_angle_degrees) + 1e-12:
            return False
        return candidate_angle_degrees < best_angle_degrees

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    analytical_normal = np.cross(surface_normal, z_axis)
    analytical_length = float(np.linalg.norm(analytical_normal))
    search_trigger = math.sin(math.radians(5.0))

    if analytical_length > search_trigger:
        initial_result = evaluate_mirror_normal(
            analytical_normal / analytical_length
        )
    else:
        # Near-horizontal fallback: search all distinct vertical mirror planes.
        initial_result: SymmetryMeasurement | None = None
        best_horizontal_angle = 0.0
        best_horizontal_normal = np.array([1.0, 0.0, 0.0], dtype=float)

        for degrees in range(180):
            angle = math.radians(float(degrees))
            candidate_normal = np.array(
                [math.cos(angle), math.sin(angle), 0.0],
                dtype=float,
            )
            candidate_result = evaluate_mirror_normal(candidate_normal)
            if result_is_better(
                candidate_result,
                float(degrees),
                initial_result,
                math.degrees(best_horizontal_angle),
            ):
                initial_result = candidate_result
                best_horizontal_angle = angle
                best_horizontal_normal = candidate_normal

        for offset_tenths in range(-10, 11):
            angle = (
                best_horizontal_angle + math.radians(offset_tenths / 10.0)
            ) % math.pi
            candidate_normal = np.array(
                [math.cos(angle), math.sin(angle), 0.0],
                dtype=float,
            )
            candidate_result = evaluate_mirror_normal(candidate_normal)
            candidate_degrees = math.degrees(angle)
            if result_is_better(
                candidate_result,
                candidate_degrees,
                initial_result,
                math.degrees(best_horizontal_angle),
            ):
                initial_result = candidate_result
                best_horizontal_angle = angle
                best_horizontal_normal = candidate_normal

        assert initial_result is not None
        # Normalise the sign deterministically for stable reports/transforms.
        if float(np.dot(np.asarray(initial_result.plane_normal), best_horizontal_normal)) < 0.0:
            best_horizontal_normal = -best_horizontal_normal
        initial_result = evaluate_mirror_normal(best_horizontal_normal)

    initial_plane_normal = tuple(initial_result.plane_normal)
    initial_variance = initial_result.post_reflection_variance
    initial_rms = initial_result.rms_error
    initial_max = initial_result.max_error

    def finalise(
        result: SymmetryMeasurement,
        *,
        passed: bool,
        tilt_applied: bool,
        tilt_degrees: float,
    ) -> SymmetryMeasurement:
        return SymmetryMeasurement(
            centroid=result.centroid,
            plane_normal=result.plane_normal,
            post_reflection_variance=result.post_reflection_variance,
            rms_error=result.rms_error,
            max_error=result.max_error,
            initial_plane_normal=initial_plane_normal,
            initial_post_reflection_variance=initial_variance,
            initial_rms_error=initial_rms,
            initial_max_error=initial_max,
            mirror_plane_y_tilt_degrees=float(tilt_degrees),
            rigid_y_rotation_correction_degrees=float(2.0 * tilt_degrees),
            tilt_search_applied=tilt_applied,
            whole_loop_sample_count=result.whole_loop_sample_count,
            sibling_pair_count=result.sibling_pair_count,
            mirror_crossings=result.mirror_crossings,
            passed=passed,
        )

    # The outer threshold remains the acceptance gate. The Y-tilt search only
    # corrects already acceptable, nearly symmetric candidates.
    if not math.isfinite(initial_rms) or initial_rms > rms_tolerance:
        return finalise(
            initial_result,
            passed=False,
            tilt_applied=False,
            tilt_degrees=0.0,
        )

    if initial_rms <= direct_rms_tolerance:
        return finalise(
            initial_result,
            passed=True,
            tilt_applied=False,
            tilt_degrees=0.0,
        )

    base_normal = np.asarray(initial_result.plane_normal, dtype=float)
    base_normal /= max(float(np.linalg.norm(base_normal)), 1e-15)

    def rotate_normal_about_y(angle_degrees: float) -> np.ndarray:
        angle = math.radians(angle_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        x, y, z = (float(value) for value in base_normal)
        return np.array(
            [
                cosine * x + sine * z,
                y,
                -sine * x + cosine * z,
            ],
            dtype=float,
        )

    # This is deliberately a small local search around the deterministic plane:
    # dashboard-following submeshes should need only a shallow correction.
    max_tilt_degrees = 6.0
    coarse_step_degrees = 0.25
    fine_step_degrees = 0.02

    best_result = initial_result
    best_tilt_degrees = 0.0

    coarse_angles = np.arange(
        -max_tilt_degrees,
        max_tilt_degrees + coarse_step_degrees * 0.5,
        coarse_step_degrees,
    )
    for angle_degrees in coarse_angles:
        candidate_result = evaluate_mirror_normal(
            rotate_normal_about_y(float(angle_degrees))
        )
        if result_is_better(
            candidate_result,
            float(angle_degrees),
            best_result,
            best_tilt_degrees,
        ):
            best_result = candidate_result
            best_tilt_degrees = float(angle_degrees)

    fine_start = max(-max_tilt_degrees, best_tilt_degrees - coarse_step_degrees)
    fine_end = min(max_tilt_degrees, best_tilt_degrees + coarse_step_degrees)
    fine_angles = np.arange(
        fine_start,
        fine_end + fine_step_degrees * 0.5,
        fine_step_degrees,
    )
    for angle_degrees in fine_angles:
        candidate_result = evaluate_mirror_normal(
            rotate_normal_about_y(float(angle_degrees))
        )
        if result_is_better(
            candidate_result,
            float(angle_degrees),
            best_result,
            best_tilt_degrees,
        ):
            best_result = candidate_result
            best_tilt_degrees = float(angle_degrees)

    return finalise(
        best_result,
        passed=True,
        tilt_applied=True,
        tilt_degrees=best_tilt_degrees,
    )


# ---------------------------------------------------------------------------
# Recursive coarse-to-fine sweep, sequential by disconnected island
# ---------------------------------------------------------------------------


def _group_edges_by_island(
    topology: Topology,
    islands: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[tuple[int, int], tuple[int, ...], float | None], ...], ...]:
    """Assign every topology edge to its disconnected geometric island once."""
    face_island = np.full(len(topology.triangles), -1, dtype=np.int64)
    for island_index, faces in enumerate(islands):
        face_island[np.asarray(faces, dtype=np.int64)] = island_index

    grouped: list[list[tuple[tuple[int, int], tuple[int, ...], float | None]]] = [
        [] for _ in islands
    ]
    for edge, faces in topology.edge_faces.items():
        if not faces:
            continue
        island_index = int(face_island[faces[0]])
        if island_index < 0:
            continue
        if any(int(face_island[face]) != island_index for face in faces[1:]):
            raise ValueError("An edge unexpectedly spans disconnected geometric islands.")
        grouped[island_index].append((edge, faces, topology.edge_angles.get(edge)))

    return tuple(tuple(entries) for entries in grouped)


def _segment_active_faces(
    active: set[int],
    threshold: float,
    island_edges: tuple[tuple[tuple[int, int], tuple[int, ...], float | None], ...],
    face_areas: np.ndarray,
) -> list[tuple[int, ...]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for _edge, faces, angle in island_edges:
        if len(faces) != 2 or angle is None:
            continue
        first, second = faces
        if first in active and second in active and angle < threshold:
            adjacency[first].append(second)
            adjacency[second].append(first)

    remaining = set(active)
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = next(iter(remaining))
        found = {seed}
        pending = [seed]
        while pending:
            face = pending.pop()
            for neighbour in adjacency.get(face, ()):
                if neighbour not in found:
                    found.add(neighbour)
                    pending.append(neighbour)
        remaining.difference_update(found)
        components.append(tuple(sorted(found)))

    components.sort(
        key=lambda faces: (-float(face_areas[list(faces)].sum()), -len(faces), faces[0])
    )
    return components


def _region_boundary(
    topology: Topology,
    faces: tuple[int, ...],
    active: set[int],
    threshold: float,
    edge_lookup: dict[tuple[int, int], tuple[tuple[int, ...], float | None]],
) -> RegionBoundary:
    face_set = set(faces)
    candidate_edges: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = (int(value) for value in topology.triangles[face])
        candidate_edges.update((canonical_edge(a, b), canonical_edge(b, c), canonical_edge(c, a)))

    boundary: set[tuple[int, int]] = set()
    crease_count = mesh_count = cut_count = nonmanifold_count = 0
    for edge in candidate_edges:
        adjacent, angle = edge_lookup[edge]
        inside = sum(face in face_set for face in adjacent)
        if inside == len(adjacent) and len(adjacent) == 2:
            continue
        if len(adjacent) == 1:
            boundary.add(edge)
            mesh_count += 1
        elif len(adjacent) > 2:
            boundary.add(edge)
            nonmanifold_count += 1
        elif len(adjacent) == 2:
            other = adjacent[0] if adjacent[1] in face_set else adjacent[1]
            boundary.add(edge)
            if other not in active:
                cut_count += 1
            elif angle is not None and angle >= threshold:
                crease_count += 1
            else:
                cut_count += 1

    loops, closed = ordered_boundary_loops(boundary)
    perimeter = sum(
        float(np.linalg.norm(topology.vertices[first] - topology.vertices[second]))
        for first, second in boundary
    )
    return RegionBoundary(
        edges=tuple(sorted(boundary)),
        loops=loops,
        perimeter=perimeter,
        crease_edges=crease_count,
        mesh_edges=mesh_count,
        cut_edges=cut_count,
        nonmanifold_edges=nonmanifold_count,
        closed=closed,
    )


def _classify_region(
    topology: Topology,
    island_index: int,
    is_main: bool,
    faces: tuple[int, ...],
    active: set[int],
    threshold: float,
    fallback_carrier_faces: set[int],
    min_area_perimeter_ratio: float,
    edge_lookup: dict[tuple[int, int], tuple[tuple[int, ...], float | None]],
) -> ActiveRegion:
    boundary = _region_boundary(topology, faces, active, threshold, edge_lookup)
    area = float(topology.face_areas[list(faces)].sum())
    area_perimeter_ratio = area / boundary.perimeter if boundary.perimeter > 1e-15 else 0.0
    role = "candidate"
    eligible = True
    if len(faces) == 0:
        role, eligible = "empty", False
    elif not boundary.edges:
        role, eligible = "no perimeter", False
    elif not boundary.closed or boundary.nonmanifold_edges:
        role, eligible = "open/rejected", False
    elif boundary.cut_edges:
        role, eligible = "cut boundary", False
    elif is_main and (boundary.mesh_edges > 0 or bool(set(faces) & fallback_carrier_faces)):
        role, eligible = "main carrier", False
    elif area_perimeter_ratio < min_area_perimeter_ratio:
        role, eligible = "thin/rejected", False
    return ActiveRegion(
        region_index=0,
        island_index=island_index,
        faces=faces,
        area=area,
        area_perimeter_ratio=area_perimeter_ratio,
        boundary=boundary,
        role=role,
        eligible=eligible,
    )


@dataclass(slots=True, frozen=True)
class _IslandGeometry:
    area: float
    points: np.ndarray
    samples: np.ndarray


def _surface_points_and_samples(
    topology: Topology,
    faces: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    face_indices = np.fromiter(faces, dtype=np.int64)
    if len(face_indices) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    triangles = topology.triangles[face_indices]
    triangle_points = topology.vertices[triangles]
    vertex_indices = np.unique(triangles.reshape(-1))
    points = topology.vertices[vertex_indices]
    samples = np.concatenate(
        (
            points,
            triangle_points.mean(axis=1),
            (triangle_points[:, 0] + triangle_points[:, 1]) * 0.5,
            (triangle_points[:, 1] + triangle_points[:, 2]) * 0.5,
            (triangle_points[:, 2] + triangle_points[:, 0]) * 0.5,
        ),
        axis=0,
    )
    return points, samples


def _nearest_sample_distances(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(samples) == 0:
        return np.full(len(points), np.inf, dtype=float)
    chunks: list[np.ndarray] = []
    # Chunking avoids allocating a potentially large child×host×XYZ array.
    for start in range(0, len(points), 256):
        delta = points[start : start + 256, None, :] - samples[None, :, :]
        chunks.append(np.sqrt(np.min(np.einsum("ijk,ijk->ij", delta, delta), axis=1)))
    return np.concatenate(chunks)


def _host_projection_frame(
    host_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Build the host facade frame once for all prospective children."""
    if len(host_points) < 3:
        return None
    centroid = host_points.mean(axis=0)
    centred = host_points - centroid
    covariance = centred.T @ centred / max(1, len(host_points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    planarity_rms = math.sqrt(max(float(eigenvalues[0]), 0.0))
    in_plane_axes = eigenvectors[:, 1:]
    host_projected = centred @ in_plane_axes
    return (
        centroid,
        in_plane_axes,
        host_projected.min(axis=0) - 0.003,
        host_projected.max(axis=0) + 0.003,
        planarity_rms,
    )


def _projected_child_overlap(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    child_points: np.ndarray,
) -> float:
    """Return the proportion of the child's PCA-plane bbox over the host.

    The two broad in-plane PCA axes define the facade footprint. The smallest
    PCA axis is used only as a planarity guard; final attachment evidence still
    comes from actual sampled host-surface distance.
    """
    if len(child_points) == 0:
        return 0.0
    centroid, in_plane_axes, host_min, host_max, _planarity_rms = frame
    child_projected = (child_points - centroid) @ in_plane_axes
    child_min = child_projected.min(axis=0)
    child_max = child_projected.max(axis=0)
    child_span = np.maximum(child_max - child_min, 1e-9)
    overlap_span = np.maximum(
        np.minimum(host_max, child_max) - np.maximum(host_min, child_min),
        0.0,
    )
    return float(np.prod(overlap_span / child_span))


def _find_eager_child_adoptions(
    topology: Topology,
    host_faces: tuple[int, ...],
    host_area: float,
    host_island_index: int,
    islands: tuple[tuple[int, ...], ...],
    island_geometry: tuple[_IslandGeometry, ...],
    active_by_island: list[set[int]],
    island_candidate_counts: list[int],
) -> tuple[ChildAdoption, ...]:
    """Claim untouched whole islands strongly supported by a symmetric host.

    This first POC intentionally handles one direct generation only. It favours
    precision over recall because an adopted island is skipped by later symmetry
    testing and inherits the host transform.
    """
    host_points, host_samples = _surface_points_and_samples(topology, host_faces)
    if len(host_points) < 3 or len(host_samples) == 0 or host_area <= 1e-15:
        return ()
    frame = _host_projection_frame(host_points)
    if frame is None or frame[4] > 0.006:
        return ()

    adopted: list[ChildAdoption] = []
    for child_offset, child_faces in enumerate(islands):
        child_island_index = child_offset + 1
        if child_island_index in {1, host_island_index}:
            continue
        # Adoption is only allowed before any part of the island has been
        # accepted or removed. This makes the skip semantics deterministic.
        if island_candidate_counts[child_offset] != 0:
            continue
        if active_by_island[child_offset] != set(child_faces):
            continue

        child = island_geometry[child_offset]
        area_ratio = child.area / host_area
        if area_ratio > 0.8:
            continue

        projected_overlap = _projected_child_overlap(frame, child.points)
        if projected_overlap < 0.8:
            continue

        distances = _nearest_sample_distances(child.samples, host_samples)
        median_gap = float(np.median(distances))
        p90_gap = float(np.percentile(distances, 90.0))
        if median_gap > 0.015 or p90_gap > 0.025:
            continue

        adopted.append(
            ChildAdoption(
                island_index=child_island_index,
                faces=child_faces,
                area=child.area,
                area_ratio=area_ratio,
                projected_overlap=projected_overlap,
                median_surface_gap=median_gap,
                p90_surface_gap=p90_gap,
            )
        )
        active_by_island[child_offset].clear()

    adopted.sort(key=lambda item: item.island_index)
    return tuple(adopted)


def _late_candidate_adoption(
    topology: Topology,
    host: IslandCandidate,
    child: IslandCandidate,
    source_candidate_id: int,
) -> ChildAdoption | None:
    """Return a late-host adoption when a later, larger facade supports an earlier root.

    Eager adoption cannot see a parent that only separates from a connected carrier at
    a lower crease threshold. This conservative recovery uses the same physical tests
    after the sweep, but only allows a later accepted, larger candidate from another
    geometric island to absorb the earlier candidate.
    """
    if host.accepted_level <= child.accepted_level:
        return None
    if host.island_index == child.island_index:
        return None
    if host.area <= 1e-15 or child.area >= host.area:
        return None

    area_ratio = child.area / host.area
    if area_ratio > 0.8:
        return None

    host_points, host_samples = _surface_points_and_samples(topology, host.host_faces)
    child_points, child_samples = _surface_points_and_samples(topology, child.faces)
    if len(host_points) < 3 or len(host_samples) == 0 or len(child_points) == 0:
        return None
    frame = _host_projection_frame(host_points)
    if frame is None or frame[4] > 0.006:
        return None

    projected_overlap = _projected_child_overlap(frame, child_points)
    if projected_overlap < 0.8:
        return None

    distances = _nearest_sample_distances(child_samples, host_samples)
    median_gap = float(np.median(distances))
    p90_gap = float(np.percentile(distances, 90.0))
    if median_gap > 0.015 or p90_gap > 0.025:
        return None

    return ChildAdoption(
        island_index=child.island_index,
        faces=child.host_faces,
        area=child.area,
        area_ratio=area_ratio,
        projected_overlap=projected_overlap,
        median_surface_gap=median_gap,
        p90_surface_gap=p90_gap,
        adoption_mode="late_candidate",
        source_candidate_id=source_candidate_id,
    )


def _resolve_late_host_adoptions(
    topology: Topology,
    candidates: list[IslandCandidate],
    premerge_id_by_key: dict[tuple[int, int], int],
) -> tuple[list[IslandCandidate], dict[tuple[int, int], tuple[int, int]]]:
    """Merge early roots into later-discovered supported hosts.

    Each child chooses its strongest qualifying host. Area strictly increases along
    every edge, so the resulting ownership graph is acyclic. Merges are applied from
    small to large, allowing a recovered assembly to be absorbed again if an even
    larger parent is found later.
    """
    candidate_by_key = {candidate.key: candidate for candidate in candidates}
    parent_for: dict[tuple[int, int], tuple[int, int]] = {}
    adoption_for: dict[tuple[int, int], ChildAdoption] = {}

    for child in candidates:
        best: tuple[tuple[float, float, float, float], IslandCandidate, ChildAdoption] | None = None
        for host in candidates:
            if host.key == child.key:
                continue
            adoption = _late_candidate_adoption(
                topology, host, child, premerge_id_by_key[child.key]
            )
            if adoption is None:
                continue
            score = (
                adoption.projected_overlap,
                -adoption.p90_surface_gap,
                -adoption.median_surface_gap,
                host.area,
            )
            if best is None or score > best[0]:
                best = (score, host, adoption)
        if best is not None:
            parent_for[child.key] = best[1].key
            adoption_for[child.key] = best[2]

    # Merge the smallest children first. A host that is itself later absorbed already
    # contains all of its descendants by the time it moves to the next parent.
    for child_key in sorted(parent_for, key=lambda key: candidate_by_key[key].area):
        host_key = parent_for[child_key]
        child = candidate_by_key[child_key]
        host = candidate_by_key[host_key]
        late_adoption = adoption_for[child_key]
        host.faces = tuple(sorted(set(host.faces) | set(child.faces)))
        host.adoptions = tuple(
            sorted(
                (*host.adoptions, late_adoption, *child.adoptions),
                key=lambda item: (item.adoption_mode, item.island_index, item.faces[0]),
            )
        )

    survivors = [candidate for candidate in candidates if candidate.key not in parent_for]

    def ultimate_parent(key: tuple[int, int]) -> tuple[int, int]:
        while key in parent_for:
            key = parent_for[key]
        return key

    absorbed_to_root = {key: ultimate_parent(key) for key in parent_for}
    return survivors, absorbed_to_root


def _sweep_one_island(
    topology: Topology,
    island_index: int,
    island_faces: tuple[int, ...],
    island_edges: tuple[tuple[tuple[int, int], tuple[int, ...], float | None], ...],
    is_main: bool,
    thresholds: tuple[float, ...],
    min_faces: int,
    min_area_perimeter_ratio: float,
    rms_tolerance: float,
    direct_rms_tolerance: float,
    sample_spacing: float,
) -> IslandSweepResult:
    edge_lookup = {edge: (faces, angle) for edge, faces, angle in island_edges}
    original_boundary_faces = {
        face
        for _edge, faces, _angle in island_edges
        if len(faces) == 1
        for face in faces
    }

    active = set(island_faces)
    accepted: list[IslandCandidate] = []
    level_results: list[IslandLevelResult] = []
    local_candidate_number = 0

    for level_index, threshold in enumerate(thresholds, start=1):
        tested_signatures: set[tuple[int, ...]] = set()
        accepted_this_level: list[tuple[int, int]] = []
        tested_count = 0
        recursion_passes = 0

        while active:
            recursion_passes += 1
            components = _segment_active_faces(
                active, threshold, island_edges, topology.face_areas
            )
            if not components:
                break
            if is_main and not original_boundary_faces:
                fallback_carrier_faces = set(components[0])
            else:
                fallback_carrier_faces = original_boundary_faces

            candidate_regions: list[ActiveRegion] = []
            for component in components:
                if len(component) < min_faces:
                    continue
                region = _classify_region(
                    topology,
                    island_index,
                    is_main,
                    component,
                    active,
                    threshold,
                    fallback_carrier_faces,
                    min_area_perimeter_ratio,
                    edge_lookup,
                )
                if region.eligible and component not in tested_signatures:
                    candidate_regions.append(region)
            candidate_regions.sort(key=lambda region: (-region.area, -len(region.faces), region.faces[0]))
            if not candidate_regions:
                break

            passed_regions: list[tuple[ActiveRegion, SymmetryMeasurement]] = []
            for region in candidate_regions:
                tested_signatures.add(region.faces)
                tested_count += 1
                measurement = measure_perimeter_symmetry(
                    topology.vertices,
                    region.boundary.loops,
                    sample_spacing,
                    rms_tolerance,
                    direct_rms_tolerance,
                )
                if measurement.passed:
                    passed_regions.append((region, measurement))
            if not passed_regions:
                break

            for region, measurement in passed_regions:
                local_candidate_number += 1
                key = (island_index, local_candidate_number)
                accepted.append(
                    IslandCandidate(
                        key=key,
                        island_index=island_index,
                        accepted_level=level_index,
                        accepted_angle=threshold,
                        faces=region.faces,
                        area=region.area,
                        area_perimeter_ratio=region.area_perimeter_ratio,
                        boundary_edges=region.boundary.edges,
                        boundary_loops=region.boundary.loops,
                        perimeter=region.boundary.perimeter,
                        measurement=measurement,
                    )
                )
                accepted_this_level.append(key)
                active.difference_update(region.faces)

        components = (
            _segment_active_faces(active, threshold, island_edges, topology.face_areas)
            if active
            else []
        )
        if is_main and components and not original_boundary_faces:
            fallback_carrier_faces = set(components[0])
        else:
            fallback_carrier_faces = original_boundary_faces
        display_regions: list[ActiveRegion] = []
        for component in components:
            region = _classify_region(
                topology,
                island_index,
                is_main,
                component,
                active,
                threshold,
                fallback_carrier_faces,
                min_area_perimeter_ratio,
                edge_lookup,
            )
            display_regions.append(region)
        for index, region in enumerate(display_regions, start=1):
            region.region_index = index

        level_results.append(
            IslandLevelResult(
                level_index=level_index,
                threshold=threshold,
                recursion_passes=recursion_passes,
                tested_candidates=tested_count,
                accepted_keys=tuple(accepted_this_level),
                active_faces=tuple(sorted(active)),
                active_regions=tuple(display_regions),
            )
        )

    return IslandSweepResult(
        island_index=island_index,
        accepted=tuple(accepted),
        remaining_faces=tuple(sorted(active)),
        levels=tuple(level_results),
    )


def analyse_symmetry_sweep(
    loaded: LoadedDae,
    part: DaePart,
    crease_max: float,
    crease_min: float,
    threshold_steps: int,
    min_region_faces: int,
    min_area_perimeter_ratio_metres: float,
    symmetry_tolerance_metres: float,
    direct_symmetry_tolerance_metres: float,
    sample_spacing_metres: float,
) -> SymmetrySweepResult:
    processing_started = time.perf_counter()
    topology_started = time.perf_counter()
    topology = topology_for_part(loaded, part)
    topology_seconds = time.perf_counter() - topology_started

    sweep_started = time.perf_counter()
    thresholds = inclusive_thresholds(crease_max, crease_min, threshold_steps)
    islands = tuple(geometric_face_islands(topology))
    island_edge_groups = _group_edges_by_island(topology, islands)
    edge_lookups = [
        {edge: (faces, angle) for edge, faces, angle in island_edges}
        for island_edges in island_edge_groups
    ]
    original_boundary_faces = [
        {
            face
            for _edge, faces, _angle in island_edges
            if len(faces) == 1
            for face in faces
        }
        for island_edges in island_edge_groups
    ]
    island_geometry_list: list[_IslandGeometry] = []
    for faces in islands:
        points, samples = _surface_points_and_samples(topology, faces)
        island_geometry_list.append(
            _IslandGeometry(
                area=float(topology.face_areas[list(faces)].sum()),
                points=points,
                samples=samples,
            )
        )
    island_geometry = tuple(island_geometry_list)

    active_by_island = [set(faces) for faces in islands]
    island_candidate_counts = [0 for _ in islands]
    accepted_candidates: list[IslandCandidate] = []
    raw_levels: list[tuple[int, float, int, int, tuple[tuple[int, int], ...], list[ActiveRegion]]] = []

    for level_index, threshold in enumerate(thresholds, start=1):
        tested_signatures: list[set[tuple[int, ...]]] = [set() for _ in islands]
        accepted_this_level: list[tuple[int, int]] = []
        tested_count = 0
        recursion_passes = 0

        while any(active_by_island):
            recursion_passes += 1
            candidate_regions: list[ActiveRegion] = []
            for island_offset, active in enumerate(active_by_island):
                if not active:
                    continue
                island_index = island_offset + 1
                components = _segment_active_faces(
                    active,
                    threshold,
                    island_edge_groups[island_offset],
                    topology.face_areas,
                )
                if island_index == 1 and components and not original_boundary_faces[island_offset]:
                    fallback_carrier_faces = set(components[0])
                else:
                    fallback_carrier_faces = original_boundary_faces[island_offset]
                for component in components:
                    if len(component) < min_region_faces:
                        continue
                    region = _classify_region(
                        topology,
                        island_index,
                        island_index == 1,
                        component,
                        active,
                        threshold,
                        fallback_carrier_faces,
                        min_area_perimeter_ratio_metres,
                        edge_lookups[island_offset],
                    )
                    if (
                        region.eligible
                        and component not in tested_signatures[island_offset]
                    ):
                        candidate_regions.append(region)

            candidate_regions.sort(
                key=lambda region: (-region.area, -len(region.faces), region.island_index, region.faces[0])
            )
            if not candidate_regions:
                break

            passed_any = False
            for region in candidate_regions:
                island_offset = region.island_index - 1
                active = active_by_island[island_offset]
                if not set(region.faces).issubset(active):
                    continue
                if region.faces in tested_signatures[island_offset]:
                    continue
                tested_signatures[island_offset].add(region.faces)
                tested_count += 1
                measurement = measure_perimeter_symmetry(
                    topology.vertices,
                    region.boundary.loops,
                    sample_spacing_metres,
                    symmetry_tolerance_metres,
                    direct_symmetry_tolerance_metres,
                )
                if not measurement.passed:
                    continue

                island_candidate_counts[island_offset] += 1
                key = (region.island_index, island_candidate_counts[island_offset])
                adoptions = _find_eager_child_adoptions(
                    topology,
                    region.faces,
                    region.area,
                    region.island_index,
                    islands,
                    island_geometry,
                    active_by_island,
                    island_candidate_counts,
                )
                all_faces = tuple(
                    sorted(
                        (*region.faces, *(face for adoption in adoptions for face in adoption.faces))
                    )
                )
                accepted_candidates.append(
                    IslandCandidate(
                        key=key,
                        island_index=region.island_index,
                        accepted_level=level_index,
                        accepted_angle=threshold,
                        faces=all_faces,
                        area=region.area,
                        area_perimeter_ratio=region.area_perimeter_ratio,
                        boundary_edges=region.boundary.edges,
                        boundary_loops=region.boundary.loops,
                        perimeter=region.boundary.perimeter,
                        measurement=measurement,
                        host_faces=region.faces,
                        adoptions=adoptions,
                    )
                )
                accepted_this_level.append(key)
                active.difference_update(region.faces)
                passed_any = True

            if not passed_any:
                break

        active_regions: list[ActiveRegion] = []
        for island_offset, active in enumerate(active_by_island):
            if not active:
                continue
            island_index = island_offset + 1
            components = _segment_active_faces(
                active,
                threshold,
                island_edge_groups[island_offset],
                topology.face_areas,
            )
            if island_index == 1 and components and not original_boundary_faces[island_offset]:
                fallback_carrier_faces = set(components[0])
            else:
                fallback_carrier_faces = original_boundary_faces[island_offset]
            for component in components:
                active_regions.append(
                    _classify_region(
                        topology,
                        island_index,
                        island_index == 1,
                        component,
                        active,
                        threshold,
                        fallback_carrier_faces,
                        min_area_perimeter_ratio_metres,
                        edge_lookups[island_offset],
                    )
                )
        active_regions.sort(key=lambda region: (-region.area, region.island_index, region.faces[0]))
        for region_index, region in enumerate(active_regions, start=1):
            region.region_index = region_index
        raw_levels.append(
            (
                level_index,
                threshold,
                recursion_passes,
                tested_count,
                tuple(accepted_this_level),
                active_regions,
            )
        )

    accepted_candidates.sort(
        key=lambda candidate: (
            candidate.accepted_level,
            -candidate.area,
            candidate.island_index,
            candidate.host_faces[0],
        )
    )
    premerge_faces_by_key = {candidate.key: candidate.faces for candidate in accepted_candidates}
    premerge_id_by_key = {
        candidate.key: index
        for index, candidate in enumerate(accepted_candidates, start=1)
    }
    accepted_candidates, absorbed_key_to_host_key = _resolve_late_host_adoptions(
        topology, accepted_candidates, premerge_id_by_key
    )
    accepted_candidates.sort(
        key=lambda candidate: (
            candidate.accepted_level,
            -candidate.area,
            candidate.island_index,
            candidate.host_faces[0],
        )
    )
    key_to_id = {
        candidate.key: index
        for index, candidate in enumerate(accepted_candidates, start=1)
    }

    def surviving_key(key: tuple[int, int]) -> tuple[int, int]:
        return absorbed_key_to_host_key.get(key, key)
    candidates = [
        AcceptedCandidate(
            key=candidate.key,
            candidate_id=key_to_id[candidate.key],
            island_index=candidate.island_index,
            accepted_level=candidate.accepted_level,
            accepted_angle=candidate.accepted_angle,
            faces=candidate.faces,
            area=candidate.area,
            area_perimeter_ratio=candidate.area_perimeter_ratio,
            boundary_edges=candidate.boundary_edges,
            boundary_loops=candidate.boundary_loops,
            perimeter=candidate.perimeter,
            measurement=candidate.measurement,
            host_faces=candidate.host_faces,
            adoptions=candidate.adoptions,
        )
        for candidate in accepted_candidates
    ]
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}

    levels: list[SweepLevelResult] = []
    cumulative_ids: list[int] = []
    for (
        level_index,
        threshold,
        recursion_passes,
        tested_count,
        accepted_keys,
        active_regions,
    ) in raw_levels:
        accepted_ids = tuple(
            sorted({key_to_id[surviving_key(key)] for key in accepted_keys})
        )
        cumulative_ids.extend(accepted_ids)
        active_labels = np.full(len(topology.triangles), -1, dtype=np.int64)
        for region in active_regions:
            active_labels[list(region.faces)] = region.region_index - 1
        candidate_labels = np.full(len(topology.triangles), -1, dtype=np.int64)
        cumulative_keys = [
            key
            for previous in raw_levels[:level_index]
            for key in previous[4]
        ]
        for historical_key in cumulative_keys:
            candidate_id = key_to_id[surviving_key(historical_key)]
            candidate_labels[list(premerge_faces_by_key[historical_key])] = candidate_id - 1
        levels.append(
            SweepLevelResult(
                level_index=level_index,
                crease_angle=threshold,
                recursion_passes=recursion_passes,
                tested_candidates=tested_count,
                accepted_this_level=accepted_ids,
                accepted_cumulative=tuple(sorted(set(cumulative_ids))),
                active_regions=active_regions,
                active_face_labels=active_labels,
                candidate_face_labels=candidate_labels,
            )
        )

    remaining = tuple(sorted(face for active in active_by_island for face in active))
    adopted_island_count = sum(
        adoption.adoption_mode == "eager_island"
        for candidate in candidates
        for adoption in candidate.adoptions
    )
    late_adopted_candidate_count = sum(
        adoption.adoption_mode == "late_candidate"
        for candidate in candidates
        for adoption in candidate.adoptions
    )
    adopted_child_count = adopted_island_count + late_adopted_candidate_count
    adopted_triangle_count = sum(
        adoption.triangle_count
        for candidate in candidates
        for adoption in candidate.adoptions
    )
    sweep_seconds = time.perf_counter() - sweep_started
    processing_seconds = time.perf_counter() - processing_started
    return SymmetrySweepResult(
        part=part,
        topology=topology,
        crease_max=crease_max,
        crease_min=crease_min,
        threshold_steps=threshold_steps,
        thresholds=thresholds,
        min_region_faces=min_region_faces,
        min_area_perimeter_ratio_metres=min_area_perimeter_ratio_metres,
        symmetry_tolerance_metres=symmetry_tolerance_metres,
        direct_symmetry_tolerance_metres=direct_symmetry_tolerance_metres,
        sample_spacing_metres=sample_spacing_metres,
        island_faces=islands,
        main_island_index=1,
        candidates=candidates,
        remaining_faces=remaining,
        levels=levels,
        adopted_island_count=adopted_island_count,
        late_adopted_candidate_count=late_adopted_candidate_count,
        adopted_child_count=adopted_child_count,
        adopted_triangle_count=adopted_triangle_count,
        processing_seconds=processing_seconds,
        topology_seconds=topology_seconds,
        sweep_seconds=sweep_seconds,
    )


# ---------------------------------------------------------------------------
# DAE splitting export
# ---------------------------------------------------------------------------


def safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return cleaned.strip("._-") or "part"


def _unique_id(base: str, used: set[str]) -> str:
    candidate = safe_name(base)
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    result = f"{candidate}_{suffix}"
    used.add(result)
    return result


def _find_target_node(root: ET.Element, namespace: str, part: DaePart) -> ET.Element:
    node_tag = qname(namespace, "node")
    if part.node_id:
        for node in root.iter(node_tag):
            if node.get("id") == part.node_id:
                return node
    wanted = [instance.geometry_id for instance in part.instances]
    for node in root.iter(node_tag):
        if (node.get("name") or "").strip() != part.node_name:
            continue
        found = [
            instance.get("url", "")[1:]
            for instance in node.findall(qname(namespace, "instance_geometry"))
            if instance.get("url", "").startswith("#")
        ]
        if found == wanted:
            return node
    raise ValueError("Could not locate the selected part node in the copied DAE tree.")


def _subset_source_element(
    original: ET.Element,
    selected_indices: list[int],
    new_source_id: str,
    used_ids: set[str],
    namespace: str,
) -> ET.Element:
    source = copy.deepcopy(original)
    source.set("id", new_source_id)
    float_array = source.find(qname(namespace, "float_array"))
    accessor = source.find(
        f"./{qname(namespace, 'technique_common')}/{qname(namespace, 'accessor')}"
    )
    if float_array is None or accessor is None or not float_array.text:
        # Rare non-float source: retain its payload but still make every nested ID
        # unique so the generated DAE remains schema-safe.
        for element in source.iter():
            nested_id = element.get("id")
            if nested_id and element is not source:
                replacement = _unique_id(f"{new_source_id}_{local_name(element.tag)}", used_ids)
                element.set("id", replacement)
        return source

    values = np.asarray(parse_float_list(float_array.text), dtype=float)
    stride = max(1, int(accessor.get("stride", "1")))
    offset = max(0, int(accessor.get("offset", "0")))
    rows: list[np.ndarray] = []
    for index in selected_indices:
        start = offset + index * stride
        end = start + stride
        if 0 <= start and end <= len(values):
            rows.append(values[start:end])
    compact = np.concatenate(rows) if rows else np.empty(0, dtype=float)
    float_id = _unique_id(f"{new_source_id}_array", used_ids)
    float_array.set("id", float_id)
    float_array.set("count", str(len(compact)))
    float_array.text = " ".join(f"{float(value):.9g}" for value in compact)
    accessor.set("source", f"#{float_id}")
    accessor.set("count", str(len(rows)))
    accessor.set("offset", "0")
    return source


def _transform_compact_source_xyz(
    source: ET.Element,
    namespace: str,
    matrix: np.ndarray,
    *,
    point: bool,
    normalise: bool = False,
) -> None:
    """Transform the XYZ columns of a compacted COLLADA float source in place."""
    float_array = source.find(qname(namespace, "float_array"))
    accessor = source.find(
        f"./{qname(namespace, 'technique_common')}/{qname(namespace, 'accessor')}"
    )
    if float_array is None or accessor is None or not float_array.text:
        return
    stride = max(1, int(accessor.get("stride", "1")))
    count = max(0, int(accessor.get("count", "0")))
    if stride < 3 or count == 0:
        return
    values = np.asarray(parse_float_list(float_array.text), dtype=float)
    usable = min(len(values), count * stride)
    if usable < count * stride:
        count = usable // stride
    if count == 0:
        return
    rows = values[: count * stride].reshape((count, stride)).copy()
    xyz = rows[:, :3]
    if point:
        homogeneous = np.concatenate(
            [xyz, np.ones((len(xyz), 1), dtype=float)], axis=1
        )
        transformed = (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3]
    else:
        linear = np.asarray(matrix, dtype=float)[:3, :3]
        transformed = xyz @ linear.T
    if normalise:
        lengths = np.linalg.norm(transformed, axis=1)
        valid = lengths > 1e-15
        transformed[valid] /= lengths[valid, None]
    rows[:, :3] = transformed
    values[: count * stride] = rows.reshape(-1)
    float_array.text = " ".join(f"{float(value):.9g}" for value in values)


def _subset_geometry(
    original: ET.Element,
    raw: RawGeometry,
    selected_by_primitive: dict[int, set[int]],
    new_id: str,
    new_name: str,
    namespace: str,
    used_ids: set[str],
    *,
    reverse_winding: bool = False,
    position_transform: np.ndarray | None = None,
    normal_transform: np.ndarray | None = None,
) -> ET.Element:
    original_mesh = original.find(qname(namespace, "mesh"))
    if original_mesh is None:
        geometry = copy.deepcopy(original)
        geometry.set("id", new_id)
        geometry.set("name", new_name)
        return geometry

    source_elements = {
        element.get("id", ""): element
        for element in original_mesh.findall(qname(namespace, "source"))
        if element.get("id")
    }
    vertices_elements = {
        element.get("id", ""): element
        for element in original_mesh.findall(qname(namespace, "vertices"))
        if element.get("id")
    }
    position_source_ids = {
        input_element.get("source", "")[1:]
        for vertices in vertices_elements.values()
        for input_element in vertices.findall(qname(namespace, "input"))
        if input_element.get("semantic") == "POSITION"
        and input_element.get("source", "").startswith("#")
    }
    normal_source_ids = {
        attributes.get("source", "")[1:]
        for primitive in raw.primitives
        for attributes in primitive.input_attributes
        if attributes.get("semantic") == "NORMAL"
        and attributes.get("source", "").startswith("#")
    }

    # Collect the source indices actually used by the selected triangles. VERTEX
    # indices address a <vertices> element, whose child inputs share that index.
    indices_by_target: dict[str, set[int]] = defaultdict(set)
    chosen_rows: dict[int, np.ndarray] = {}
    for primitive_index, primitive in enumerate(raw.primitives):
        chosen = sorted(selected_by_primitive.get(primitive_index, set()))
        if not chosen:
            continue
        rows = primitive.rows[np.asarray(chosen, dtype=np.int64)].copy()
        chosen_rows[primitive_index] = rows
        for attributes in primitive.input_attributes:
            source_url = attributes.get("source", "")
            if not source_url.startswith("#"):
                continue
            target = source_url[1:]
            offset = int(attributes.get("offset", "0"))
            indices_by_target[target].update(int(value) for value in rows[:, :, offset].reshape(-1))

    # A VERTEX target implies the same selected indices for every source wired
    # through that <vertices> element (POSITION in normal BeamNG exports).
    for vertices_id, indices in list(indices_by_target.items()):
        vertices = vertices_elements.get(vertices_id)
        if vertices is None:
            continue
        for input_element in vertices.findall(qname(namespace, "input")):
            source_url = input_element.get("source", "")
            if source_url.startswith("#"):
                indices_by_target[source_url[1:]].update(indices)

    source_id_map: dict[str, str] = {}
    source_index_maps: dict[str, dict[int, int]] = {}
    compact_sources: list[ET.Element] = []
    for source_id, indices in sorted(indices_by_target.items()):
        original_source = source_elements.get(source_id)
        if original_source is None:
            continue
        ordered = sorted(indices)
        compact_id = _unique_id(f"{new_id}_{safe_name(source_id)}", used_ids)
        source_id_map[source_id] = compact_id
        source_index_maps[source_id] = {old: new for new, old in enumerate(ordered)}
        compact_source = _subset_source_element(
            original_source,
            ordered,
            compact_id,
            used_ids,
            namespace,
        )
        if position_transform is not None and source_id in position_source_ids:
            _transform_compact_source_xyz(
                compact_source, namespace, position_transform, point=True
            )
        if normal_transform is not None and source_id in normal_source_ids:
            _transform_compact_source_xyz(
                compact_source,
                namespace,
                normal_transform,
                point=False,
                normalise=True,
            )
        compact_sources.append(compact_source)

    vertices_id_map: dict[str, str] = {}
    compact_vertices: list[ET.Element] = []
    for vertices_id, original_vertices in vertices_elements.items():
        if vertices_id not in indices_by_target:
            continue
        compact_id = _unique_id(f"{new_id}_{safe_name(vertices_id)}", used_ids)
        vertices_id_map[vertices_id] = compact_id
        vertices = copy.deepcopy(original_vertices)
        vertices.set("id", compact_id)
        for input_element in vertices.findall(qname(namespace, "input")):
            source_url = input_element.get("source", "")
            if source_url.startswith("#") and source_url[1:] in source_id_map:
                input_element.set("source", f"#{source_id_map[source_url[1:]]}")
        compact_vertices.append(vertices)

    geometry = ET.Element(qname(namespace, "geometry"), dict(original.attrib))
    geometry.set("id", new_id)
    geometry.set("name", new_name)
    mesh = ET.SubElement(geometry, qname(namespace, "mesh"))
    for source in compact_sources:
        mesh.append(source)
    for vertices in compact_vertices:
        mesh.append(vertices)

    for primitive_index, primitive in enumerate(raw.primitives):
        rows = chosen_rows.get(primitive_index)
        if rows is None:
            continue
        remapped = rows.copy()
        if reverse_winding:
            # Reverse complete COLLADA corner records together. Position, normal,
            # UV, colour and tangent indices therefore remain associated with the
            # same physical corner while the reflected node keeps front faces
            # facing outwards.
            remapped = remapped[:, ::-1, :]
        input_attributes: list[dict[str, str]] = []
        index_map_by_offset: dict[int, dict[int, int]] = {}

        for attributes in primitive.input_attributes:
            attributes = dict(attributes)
            source_url = attributes.get("source", "")
            offset = int(attributes.get("offset", "0"))
            index_map: dict[int, int] | None = None

            if source_url.startswith("#"):
                target = source_url[1:]
                if target in vertices_id_map:
                    attributes["source"] = f"#{vertices_id_map[target]}"
                    # A VERTEX input indexes the sources referenced by <vertices>.
                    vertices = vertices_elements[target]
                    source_targets = [
                        child.get("source", "")[1:]
                        for child in vertices.findall(qname(namespace, "input"))
                        if child.get("source", "").startswith("#")
                    ]
                    if source_targets:
                        index_map = source_index_maps[source_targets[0]]
                elif target in source_id_map:
                    attributes["source"] = f"#{source_id_map[target]}"
                    index_map = source_index_maps[target]

            if index_map is not None:
                existing = index_map_by_offset.get(offset)
                if existing is None:
                    index_map_by_offset[offset] = index_map
                elif existing != index_map:
                    raise ValueError(
                        "COLLADA inputs sharing one offset use incompatible index domains."
                    )
            input_attributes.append(attributes)

        # Multiple COLLADA inputs may intentionally share an offset. Remap that
        # index stream once; remapping it once per input would remap compacted
        # indices a second time and can raise a spurious KeyError.
        for offset, index_map in index_map_by_offset.items():
            values = remapped[:, :, offset]
            ordered_pairs = sorted(index_map.items())
            old_indices = np.fromiter(
                (old for old, _ in ordered_pairs), dtype=np.int64, count=len(ordered_pairs)
            )
            new_indices = np.fromiter(
                (new for _, new in ordered_pairs), dtype=np.int64, count=len(ordered_pairs)
            )
            positions = np.searchsorted(old_indices, values)
            in_range = positions < len(old_indices)
            clipped = np.minimum(positions, max(len(old_indices) - 1, 0))
            valid = in_range & (old_indices[clipped] == values)
            if not np.all(valid):
                missing = np.unique(values[~valid])
                preview = ", ".join(str(int(value)) for value in missing[:8])
                suffix = "…" if len(missing) > 8 else ""
                raise ValueError(
                    "Primitive references source indices absent from the compacted source: "
                    f"{preview}{suffix}"
                )
            remapped[:, :, offset] = new_indices[positions]

        triangles = ET.SubElement(mesh, qname(namespace, "triangles"), dict(primitive.attributes))
        triangles.set("count", str(len(remapped)))
        for attributes in input_attributes:
            ET.SubElement(triangles, qname(namespace, "input"), attributes)
        p = ET.SubElement(triangles, qname(namespace, "p"))
        p.text = " ".join(str(int(value)) for value in remapped.reshape(-1))

    # Preserve geometry-level extras, which may contain exporter metadata, but
    # never copy old mesh sources/IDs into the new geometry.
    for child in list(original):
        if local_name(child.tag) != "mesh":
            geometry.append(copy.deepcopy(child))
    return geometry


def _format_matrix(matrix: np.ndarray) -> str:
    # Keep export layout compatible with the known-good pre-refactor output.
    values = np.asarray(matrix, dtype=float).reshape(-1)
    return " ".join("0" if abs(float(value)) < 1e-12 else f"{float(value):.10g}" for value in values)


def _global_x_reflection() -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = -1.0
    return matrix


def _plane_reflection_in_dae_units(
    centroid_metres: tuple[float, float, float],
    normal_world: tuple[float, float, float],
    unit_scale: float,
) -> np.ndarray:
    """Reflection in the detected vertical symmetry plane.

    Symmetry measurements are stored in world metres, whereas a COLLADA node
    matrix operates in the DAE's authored units before the <unit meter="...">
    scale is applied. The rotation/reflection block is dimensionless; only the
    translation needs conversion back to DAE units.
    """
    normal = np.asarray(normal_world, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("Detected symmetry plane has a zero normal.")
    normal /= norm
    centroid = np.asarray(centroid_metres, dtype=float)
    linear = np.eye(3, dtype=float) - 2.0 * np.outer(normal, normal)
    translation_metres = 2.0 * normal * float(np.dot(normal, centroid))
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation_metres / max(float(unit_scale), 1e-15)
    return matrix


def _candidate_rigid_transform(
    measurement: SymmetryMeasurement,
    unit_scale: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return the proper rigid transform replacing a geometric X reflection.

    For a candidate S symmetric about its own detected plane L, L(S) = S.
    Therefore the globally mirrored placement G(S) can be produced without
    mirroring its UV/text-bearing geometry by applying the composition G L:

        G(S) = (G L)(S)

    The composition has determinant +1. With the ordinary vertical symmetry
    plane it is a rotation about Z plus an XY translation. A borderline candidate
    may use the small corrected Y-tilted plane found by the RMS search, in which
    case the same exact matrix composition supplies the corresponding 3-D rigid
    correction without Euler-angle reconstruction.
    """
    local_reflection = _plane_reflection_in_dae_units(
        measurement.centroid,
        measurement.plane_normal,
        unit_scale,
    )
    transform = _global_x_reflection() @ local_reflection
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-7:
        raise ValueError(f"Candidate transform is not a proper rigid transform (det={determinant:.9g}).")
    yaw_degrees = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    translation_metres = transform[:3, 3] * float(unit_scale)
    return transform, rotation, yaw_degrees, translation_metres


def _new_preview_node(
    namespace: str,
    node_id: str,
    node_name: str,
    matrix: np.ndarray,
) -> ET.Element:
    node = ET.Element(qname(namespace, "node"), {"id": node_id, "name": node_name, "type": "NODE"})
    matrix_element = ET.SubElement(node, qname(namespace, "matrix"), {"sid": "transform"})
    matrix_element.text = _format_matrix(matrix)
    return node


def _prepare_preview_scene(root: ET.Element, namespace: str, used_ids: set[str]) -> ET.Element:
    """Replace the source visual scenes with one static selected-part preview."""
    visual_library = root.find(f"./{qname(namespace, 'library_visual_scenes')}")
    if visual_library is None:
        visual_library = ET.SubElement(root, qname(namespace, "library_visual_scenes"))
    else:
        for child in list(visual_library):
            visual_library.remove(child)
    scene_id = _unique_id("BeamXP_Transform_Preview_Scene", used_ids)
    visual_scene = ET.SubElement(
        visual_library,
        qname(namespace, "visual_scene"),
        {"id": scene_id, "name": scene_id},
    )

    scene = root.find(f"./{qname(namespace, 'scene')}")
    if scene is None:
        scene = ET.SubElement(root, qname(namespace, "scene"))
    else:
        for child in list(scene):
            scene.remove(child)
    ET.SubElement(scene, qname(namespace, "instance_visual_scene"), {"url": f"#{scene_id}"})
    return visual_scene


def _copy_or_convert_blender_texture(texture_path: Path, output_dae: Path) -> tuple[Path, str]:
    """Place a Blender-readable base-colour image beside the preview DAE.

    DDS is converted to PNG when Pillow is available. This is deliberately an
    export-only convenience: the source texture is never modified, and the
    generated geometry/UV data is unchanged. Other image formats are copied as-is.
    """
    if not texture_path.is_file():
        raise FileNotFoundError(f"Blender preview texture not found: {texture_path}")

    output_dae.parent.mkdir(parents=True, exist_ok=True)
    if texture_path.suffix.lower() == ".dds":
        target = output_dae.parent / f"{texture_path.stem}.png"
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Exporting a DDS texture for Blender requires Pillow. "
                "Install it with: python -m pip install Pillow"
            ) from exc
        with Image.open(texture_path) as image:
            image.save(target, format="PNG")
        return target, "DDS converted to PNG with Pillow"

    target = output_dae.parent / texture_path.name
    try:
        same_file = texture_path.resolve() == target.resolve()
    except OSError:
        same_file = False
    if not same_file:
        shutil.copy2(texture_path, target)
    return target, "image copied beside DAE"


def _wire_blender_base_colour(
    root: ET.Element,
    namespace: str,
    generated_geometries: Iterable[ET.Element],
    texture_path: Path,
    output_dae: Path,
) -> dict[str, object]:
    """Repair missing COLLADA image records for the selected-part materials.

    BeamNG DAEs often retain a valid effect/sampler/UV binding but omit the
    <image> record expected by general-purpose COLLADA importers. Resolve the
    image IDs referenced by materials actually used by the exported geometries,
    then point the single unresolved base-colour image at a local Blender-safe
    file. Existing resolved image bindings are never overwritten.
    """
    used_symbols = {
        primitive.get("material", "")
        for geometry in generated_geometries
        for primitive in geometry.iter()
        if local_name(primitive.tag) in PRIMITIVE_TAGS and primitive.get("material")
    }
    if not used_symbols:
        raise ValueError("The generated preview geometry has no material symbols to texture.")

    materials_library = root.find(f"./{qname(namespace, 'library_materials')}")
    effects_library = root.find(f"./{qname(namespace, 'library_effects')}")
    if materials_library is None or effects_library is None:
        raise ValueError("The source DAE has no COLLADA material/effect libraries.")

    material_by_id = {
        material.get("id", ""): material
        for material in materials_library.findall(qname(namespace, "material"))
        if material.get("id")
    }
    effect_by_id = {
        effect.get("id", ""): effect
        for effect in effects_library.findall(qname(namespace, "effect"))
        if effect.get("id")
    }

    referenced_images: dict[str, set[str]] = defaultdict(set)
    used_effects: set[str] = set()
    targets_by_symbol = _material_targets_by_symbol(root, namespace)
    for symbol in sorted(used_symbols):
        material, _aliases = _resolve_collada_material_for_symbol(
            symbol, material_by_id, targets_by_symbol
        )
        if material is None:
            continue
        instance_effect = material.find(qname(namespace, "instance_effect"))
        url = instance_effect.get("url", "") if instance_effect is not None else ""
        if not url.startswith("#"):
            continue
        effect_id = url[1:]
        effect = effect_by_id.get(effect_id)
        if effect is None:
            continue
        used_effects.add(effect_id)
        for surface in effect.iter(qname(namespace, "surface")):
            init_from = surface.find(qname(namespace, "init_from"))
            image_id = (init_from.text or "").strip() if init_from is not None else ""
            if image_id:
                referenced_images[image_id].add(symbol)

    images_library = root.find(f"./{qname(namespace, 'library_images')}")
    existing_image_ids: set[str] = set()
    if images_library is not None:
        existing_image_ids = {
            image.get("id", "")
            for image in images_library.findall(qname(namespace, "image"))
            if image.get("id")
        }

    unresolved = sorted(set(referenced_images) - existing_image_ids)
    if not unresolved:
        raise ValueError(
            "No unresolved image binding was found in the exported part's materials; "
            "refusing to replace an existing texture automatically."
        )
    if len(unresolved) != 1:
        details = ", ".join(unresolved)
        raise ValueError(
            "The exported part references multiple unresolved COLLADA images "
            f"({details}). This POC accepts one base-colour texture at a time."
        )

    local_texture, preparation = _copy_or_convert_blender_texture(texture_path, output_dae)
    if images_library is None:
        images_library = ET.Element(qname(namespace, "library_images"))
        children = list(root)
        insert_at = next(
            (
                index
                for index, child in enumerate(children)
                if local_name(child.tag) in {"library_effects", "library_materials", "library_geometries"}
            ),
            len(children),
        )
        root.insert(insert_at, images_library)

    image_id = unresolved[0]
    image = ET.SubElement(
        images_library,
        qname(namespace, "image"),
        {"id": image_id, "name": image_id},
    )
    init_from = ET.SubElement(image, qname(namespace, "init_from"))
    init_from.text = local_texture.name

    return {
        "enabled": True,
        "source_texture": str(texture_path),
        "output_texture": str(local_texture),
        "output_texture_reference": local_texture.name,
        "preparation": preparation,
        "material_symbols": sorted(used_symbols),
        "effect_ids": sorted(used_effects),
        "repaired_image_id": image_id,
        "uv_data_modified": False,
    }




def _material_targets_by_symbol(root: ET.Element, namespace: str) -> dict[str, tuple[str, ...]]:
    """Map COLLADA primitive material symbols to bound library material IDs.

    A primitive's ``material`` attribute is a symbol scoped to an
    ``instance_geometry``. It is often, but not necessarily, identical to the
    target ``library_materials/material`` ID. Vehicle DAEs from different
    exporters use both forms, so texture wiring must follow ``instance_material``
    rather than assuming the symbol is a material ID.
    """
    targets: dict[str, list[str]] = defaultdict(list)
    for instance_material in root.iter(qname(namespace, "instance_material")):
        symbol = (instance_material.get("symbol") or "").strip()
        target = (instance_material.get("target") or "").strip().lstrip("#")
        if symbol and target and target not in targets[symbol]:
            targets[symbol].append(target)
    return {symbol: tuple(values) for symbol, values in targets.items()}


def _resolve_collada_material_for_symbol(
    symbol: str,
    material_by_id: dict[str, ET.Element],
    targets_by_symbol: dict[str, tuple[str, ...]],
) -> tuple[ET.Element | None, tuple[str, ...]]:
    """Resolve a primitive symbol to its material and return useful aliases."""
    candidate_ids: list[str] = []
    for value in (symbol, *targets_by_symbol.get(symbol, ())):
        if value and value not in candidate_ids:
            candidate_ids.append(value)

    material: ET.Element | None = None
    for material_id in candidate_ids:
        material = material_by_id.get(material_id)
        if material is not None:
            break

    if material is None:
        normalised = _normalise_material_alias(symbol)
        for candidate in material_by_id.values():
            aliases = (candidate.get("id", ""), candidate.get("name", ""))
            if any(_normalise_material_alias(alias) == normalised for alias in aliases if alias):
                material = candidate
                break

    aliases: list[str] = []
    for value in (
        symbol,
        *candidate_ids,
        material.get("id", "") if material is not None else "",
        material.get("name", "") if material is not None else "",
    ):
        value = (value or "").strip()
        if value and value not in aliases:
            aliases.append(value)
    return material, tuple(aliases)


def _effect_base_colour_image_ids(effect: ET.Element, namespace: str) -> tuple[str, ...]:
    """Resolve image IDs used by the effect's diffuse/base-colour texture."""
    newparams = {
        element.get("sid", ""): element
        for element in effect.iter(qname(namespace, "newparam"))
        if element.get("sid")
    }
    sampler_to_surface: dict[str, str] = {}
    surface_to_image: dict[str, str] = {}
    for sid, element in newparams.items():
        sampler = element.find(qname(namespace, "sampler2D"))
        if sampler is not None:
            source = sampler.find(qname(namespace, "source"))
            if source is not None and (source.text or "").strip():
                sampler_to_surface[sid] = (source.text or "").strip()
        surface = element.find(qname(namespace, "surface"))
        if surface is not None:
            init_from = surface.find(qname(namespace, "init_from"))
            if init_from is not None and (init_from.text or "").strip():
                surface_to_image[sid] = (init_from.text or "").strip()

    image_ids: list[str] = []
    for channel_name in ("diffuse", "ambient", "emission"):
        for channel in effect.iter(qname(namespace, channel_name)):
            texture = channel.find(qname(namespace, "texture"))
            sampler_sid = texture.get("texture", "") if texture is not None else ""
            surface_sid = sampler_to_surface.get(sampler_sid, sampler_sid)
            image_id = surface_to_image.get(surface_sid, surface_sid)
            if image_id and image_id not in image_ids:
                image_ids.append(image_id)
        if image_ids:
            break

    if not image_ids:
        for image_id in surface_to_image.values():
            if image_id and image_id not in image_ids:
                image_ids.append(image_id)
    return tuple(image_ids)


def _texcoord_semantics_by_symbol(
    root: ET.Element, namespace: str
) -> dict[str, str]:
    """Return the TEXCOORD bind semantic used by each material symbol."""
    semantics: dict[str, str] = {}
    for instance_material in root.iter(qname(namespace, "instance_material")):
        symbol = (instance_material.get("symbol") or "").strip()
        if not symbol or symbol in semantics:
            continue
        bindings = list(instance_material.findall(qname(namespace, "bind_vertex_input")))
        preferred = next(
            (
                binding
                for binding in bindings
                if (binding.get("input_semantic") or "").upper() == "TEXCOORD"
                and (binding.get("input_set") or "0") == "0"
            ),
            None,
        )
        if preferred is None:
            preferred = next(
                (
                    binding
                    for binding in bindings
                    if (binding.get("input_semantic") or "").upper() == "TEXCOORD"
                ),
                None,
            )
        semantic = (preferred.get("semantic") or "").strip() if preferred is not None else ""
        semantics[symbol] = semantic or "UVChannel_1"
    return semantics


def _ensure_effect_base_colour_image_chain(
    root: ET.Element,
    effect: ET.Element,
    material: ET.Element,
    symbol: str,
    namespace: str,
    texcoord_semantic: str,
    image_by_id: dict[str, ET.Element],
) -> tuple[tuple[str, ...], bool]:
    """Return an effect's base-colour image IDs, creating a chain when absent.

    Many BeamNG DAEs rely on the external ``*.materials.json`` definitions and
    therefore contain only a diffuse colour in ``profile_COMMON``. Blender
    cannot infer the JSON material, so the preview exporter must create the
    ordinary COLLADA image -> surface -> sampler -> diffuse texture chain.
    """
    existing = _effect_base_colour_image_ids(effect, namespace)
    if existing:
        return existing, False

    profile = effect.find(qname(namespace, "profile_COMMON"))
    if profile is None:
        profile = ET.SubElement(effect, qname(namespace, "profile_COMMON"))

    material_base = safe_name(
        material.get("id") or material.get("name") or symbol or "beamxp_material"
    )
    existing_image_ids = set(image_by_id)
    image_id = f"{material_base}__beamxp_base_colour"
    suffix = 2
    while image_id in existing_image_ids:
        image_id = f"{material_base}__beamxp_base_colour_{suffix}"
        suffix += 1

    existing_sids = {
        element.get("sid", "")
        for element in profile.findall(qname(namespace, "newparam"))
        if element.get("sid")
    }
    surface_sid = f"{material_base}__beamxp_surface"
    sampler_sid = f"{material_base}__beamxp_sampler"
    suffix = 2
    while surface_sid in existing_sids or sampler_sid in existing_sids:
        surface_sid = f"{material_base}__beamxp_surface_{suffix}"
        sampler_sid = f"{material_base}__beamxp_sampler_{suffix}"
        suffix += 1


    technique = profile.find(qname(namespace, "technique"))
    insertion_index = list(profile).index(technique) if technique is not None else len(profile)

    surface_param = ET.Element(qname(namespace, "newparam"), {"sid": surface_sid})
    surface = ET.SubElement(surface_param, qname(namespace, "surface"), {"type": "2D"})
    ET.SubElement(surface, qname(namespace, "init_from")).text = image_id
    profile.insert(insertion_index, surface_param)
    insertion_index += 1

    sampler_param = ET.Element(qname(namespace, "newparam"), {"sid": sampler_sid})
    sampler = ET.SubElement(sampler_param, qname(namespace, "sampler2D"))
    ET.SubElement(sampler, qname(namespace, "source")).text = surface_sid
    profile.insert(insertion_index, sampler_param)

    if technique is None:
        technique = ET.SubElement(profile, qname(namespace, "technique"), {"sid": "common"})

    shader = next(
        (technique.find(qname(namespace, name)) for name in ("lambert", "phong", "blinn", "constant")
         if technique.find(qname(namespace, name)) is not None),
        None,
    )
    if shader is None:
        shader = ET.SubElement(technique, qname(namespace, "lambert"))

    diffuse = shader.find(qname(namespace, "diffuse"))
    if diffuse is None:
        diffuse = ET.SubElement(shader, qname(namespace, "diffuse"))
    for child in list(diffuse):
        diffuse.remove(child)
    ET.SubElement(
        diffuse,
        qname(namespace, "texture"),
        {"texture": sampler_sid, "texcoord": texcoord_semantic or "UVChannel_1"},
    )
    return (image_id,), True


def _wire_blender_base_colours(
    root: ET.Element,
    namespace: str,
    generated_geometries: Iterable[ET.Element],
    texture_paths_by_material: dict[str, Path],
    output_dae: Path,
) -> dict[str, object]:
    """Wire one or more archive-resolved material textures for Blender preview."""
    alias_to_path = {
        _normalise_material_alias(alias): path
        for alias, path in texture_paths_by_material.items()
        if alias and path is not None
    }
    if not alias_to_path:
        return {"enabled": False, "reason": "no resolved archive textures"}

    used_symbols = {
        primitive.get("material", "")
        for geometry in generated_geometries
        for primitive in geometry.iter()
        if local_name(primitive.tag) in PRIMITIVE_TAGS and primitive.get("material")
    }
    materials_library = root.find(f"./{qname(namespace, 'library_materials')}")
    effects_library = root.find(f"./{qname(namespace, 'library_effects')}")
    if materials_library is None or effects_library is None:
        raise ValueError("The source DAE has no COLLADA material/effect libraries.")

    material_by_id = {
        material.get("id", ""): material
        for material in materials_library.findall(qname(namespace, "material"))
        if material.get("id")
    }
    effect_by_id = {
        effect.get("id", ""): effect
        for effect in effects_library.findall(qname(namespace, "effect"))
        if effect.get("id")
    }
    images_library = root.find(f"./{qname(namespace, 'library_images')}")
    if images_library is None:
        images_library = ET.Element(qname(namespace, "library_images"))
        children = list(root)
        insert_at = next(
            (
                index
                for index, child in enumerate(children)
                if local_name(child.tag) in {"library_effects", "library_materials", "library_geometries"}
            ),
            len(children),
        )
        root.insert(insert_at, images_library)

    image_by_id = {
        image.get("id", ""): image
        for image in images_library.findall(qname(namespace, "image"))
        if image.get("id")
    }
    targets_by_symbol = _material_targets_by_symbol(root, namespace)
    texcoord_by_symbol = _texcoord_semantics_by_symbol(root, namespace)
    wired: list[dict[str, object]] = []
    unresolved_symbols: list[str] = []
    unresolved_reasons: dict[str, str] = {}

    for symbol in sorted(used_symbols):
        material, aliases = _resolve_collada_material_for_symbol(
            symbol, material_by_id, targets_by_symbol
        )
        if material is None:
            unresolved_symbols.append(symbol)
            unresolved_reasons[symbol] = "no bound COLLADA material"
            continue
        texture_path = next(
            (
                alias_to_path[_normalise_material_alias(alias)]
                for alias in aliases
                if _normalise_material_alias(alias) in alias_to_path
            ),
            None,
        )
        if texture_path is None:
            unresolved_symbols.append(symbol)
            unresolved_reasons[symbol] = (
                "archive texture aliases did not match " + ", ".join(aliases)
            )
            continue

        instance_effect = material.find(qname(namespace, "instance_effect"))
        effect_url = instance_effect.get("url", "") if instance_effect is not None else ""
        effect = effect_by_id.get(effect_url.lstrip("#")) if effect_url.startswith("#") else None
        if effect is None:
            unresolved_symbols.append(symbol)
            unresolved_reasons[symbol] = "bound material has no resolvable COLLADA effect"
            continue
        image_ids, created_effect_chain = _ensure_effect_base_colour_image_chain(
            root,
            effect,
            material,
            symbol,
            namespace,
            texcoord_by_symbol.get(symbol, "UVChannel_1"),
            image_by_id,
        )

        # A materials-JSON trim selected in the UI is authoritative. Once its
        # archive member has resolved, conversion failure must be explicit rather
        # than silently exporting an untextured preview.
        local_texture, preparation = _copy_or_convert_blender_texture(texture_path, output_dae)
        for image_id in image_ids:
            image = image_by_id.get(image_id)
            if image is None:
                image = ET.SubElement(
                    images_library,
                    qname(namespace, "image"),
                    {"id": image_id, "name": image_id},
                )
                image_by_id[image_id] = image
            init_from = image.find(qname(namespace, "init_from"))
            if init_from is None:
                init_from = ET.SubElement(image, qname(namespace, "init_from"))
            init_from.text = local_texture.name
        wired.append(
            {
                "material_symbol": symbol,
                "material_name": material.get("name") or material.get("id"),
                "effect_id": effect.get("id"),
                "image_ids": list(image_ids),
                "source_texture": str(texture_path),
                "output_texture": str(local_texture),
                "output_texture_reference": local_texture.name,
                "preparation": preparation,
                "created_effect_texture_chain": created_effect_chain,
                "texcoord_semantic": texcoord_by_symbol.get(symbol, "UVChannel_1"),
            }
        )

    if not wired:
        details = "; ".join(
            f"{symbol}: {unresolved_reasons.get(symbol, 'unresolved')}"
            for symbol in sorted(set(unresolved_symbols))
        ) or "no used material symbols"
        raise ValueError(
            "A materials-JSON texture was selected, but it could not be bound to "
            f"the exported COLLADA material ({details})."
        )
    return {
        "enabled": True,
        "mode": "vehicle ZIP automatic material resolution",
        "wired_materials": wired,
        "unresolved_material_symbols": sorted(set(unresolved_symbols)),
        "uv_data_modified": False,
    }


def export_transformed_part_dae(
    loaded: LoadedDae,
    result: SymmetrySweepResult,
    output_path: Path,
    blender_base_colour: Path | None = None,
    blender_base_colours: dict[str, Path] | None = None,
) -> dict[str, object]:
    """Export only the selected part, already converted to its opposite hand.

    Residual/unclassified triangles are geometrically mirrored about vehicle X=0
    by baking the reflection into their positions and normal vectors, with reversed
    triangle winding. The carrier node itself retains a positive determinant so
    Blender does not need to interpret custom normals under a negative object scale.
    Each perimeter-symmetric candidate remains an independent mesh and receives
    the equivalent proper rotation+translation G*L, where L is reflection in the
    candidate's own symmetry plane. No stitching or node snapping is performed.
    """
    tree = copy.deepcopy(loaded.tree)
    root = tree.getroot()
    namespace = loaded.namespace
    if namespace:
        ET.register_namespace("", namespace)

    target = _find_target_node(root, namespace, result.part)
    original_instances = list(target.findall(qname(namespace, "instance_geometry")))
    if len(original_instances) != len(result.part.instances):
        raise ValueError(
            "The selected node's instance_geometry count changed between analysis and export."
        )

    library = root.find(f"./{qname(namespace, 'library_geometries')}")
    if library is None:
        raise ValueError("DAE has no library_geometries element.")
    geometry_lookup = {
        geometry.get("id", ""): geometry
        for geometry in library.findall(qname(namespace, "geometry"))
        if geometry.get("id")
    }
    used_ids = {element.get("id") for element in root.iter() if element.get("id")}
    used_ids.discard(None)

    candidate_for_face = np.full(len(result.topology.triangles), -1, dtype=np.int64)
    for candidate in result.candidates:
        candidate_for_face[list(candidate.faces)] = candidate.candidate_id

    accepted_by_source: dict[tuple[int, int, int], int] = {}
    for face_index, candidate_id in enumerate(candidate_for_face):
        if candidate_id < 0:
            continue
        source = result.topology.source_faces[face_index]
        accepted_by_source[(source.instance_index, source.primitive_index, source.triangle_index)] = int(candidate_id)

    residual_selection: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    candidate_selection: dict[int, dict[int, dict[int, set[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for instance_index, instance in enumerate(result.part.instances):
        raw = parse_geometry(loaded, instance.geometry_id)
        for primitive_index, primitive in enumerate(raw.primitives):
            for triangle_index in range(len(primitive.rows)):
                candidate_id = accepted_by_source.get((instance_index, primitive_index, triangle_index))
                if candidate_id is None:
                    residual_selection[instance_index][primitive_index].add(triangle_index)
                else:
                    candidate_selection[candidate_id][instance_index][primitive_index].add(triangle_index)

    # Generate compact geometries before clearing the original geometry library.
    generated_geometries: list[ET.Element] = []
    generated_geometry_ids: list[str] = []
    base_node_name = result.part.node_name or result.part.node_id or "beamxp_part"

    mirror_world = _global_x_reflection()
    # Bake the world-X reflection into carrier-local geometry. Keeping the original
    # positive-determinant part matrix avoids Blender's inconsistent custom-normal
    # handling on negatively scaled COLLADA nodes. P @ B = G @ P, hence
    # B = inv(P) @ G @ P.
    part_matrix = np.asarray(result.part.matrix, dtype=float)
    carrier_local_reflection = np.linalg.inv(part_matrix) @ mirror_world @ part_matrix
    carrier_normal_transform = np.linalg.inv(carrier_local_reflection[:3, :3]).T
    carrier_matrix = part_matrix.copy()
    carrier_id = _unique_id(f"{base_node_name}__beamxp_mirrored_carrier", used_ids)  # type: ignore[arg-type]
    carrier_node = _new_preview_node(namespace, carrier_id, carrier_id, carrier_matrix)
    residual_triangle_count = 0
    residual_geometry_ids: list[str] = []
    for instance_index, original_instance in enumerate(original_instances):
        instance_info = result.part.instances[instance_index]
        raw = parse_geometry(loaded, instance_info.geometry_id)
        selected = residual_selection.get(instance_index, {})
        triangle_count = sum(len(indices) for indices in selected.values())
        if triangle_count == 0:
            continue
        original_geometry = geometry_lookup[instance_info.geometry_id]
        geometry_id = _unique_id(
            f"{instance_info.geometry_id}__beamxp_mirrored_carrier", used_ids  # type: ignore[arg-type]
        )
        geometry = _subset_geometry(
            original_geometry,
            raw,
            selected,
            geometry_id,
            f"{original_geometry.get('name') or instance_info.geometry_id} BeamXP mirrored carrier",
            namespace,
            used_ids,
            reverse_winding=True,
            position_transform=carrier_local_reflection,
            normal_transform=carrier_normal_transform,
        )
        generated_geometries.append(geometry)
        generated_geometry_ids.append(geometry_id)
        residual_geometry_ids.append(geometry_id)
        residual_triangle_count += triangle_count
        instance_geometry = copy.deepcopy(original_instance)
        instance_geometry.set("url", f"#{geometry_id}")
        carrier_node.append(instance_geometry)

    preview_scene = _prepare_preview_scene(root, namespace, used_ids)  # type: ignore[arg-type]
    if residual_triangle_count:
        preview_scene.append(carrier_node)

    candidate_by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    generated_candidates: list[dict[str, object]] = []
    for candidate_id in sorted(candidate_selection):
        candidate = candidate_by_id[candidate_id]
        proper_transform, rotation, yaw_degrees, translation_metres = _candidate_rigid_transform(
            candidate.measurement,
            loaded.unit_scale,
        )
        node_matrix = proper_transform @ result.part.matrix
        node_id = _unique_id(f"{base_node_name}__beamxp_rigid_{candidate_id:03d}", used_ids)  # type: ignore[arg-type]
        node = _new_preview_node(namespace, node_id, node_id, node_matrix)
        instance_geometry_ids: list[str] = []
        triangle_count = 0
        for instance_index, primitive_selection in sorted(candidate_selection[candidate_id].items()):
            original_instance = original_instances[instance_index]
            instance_info = result.part.instances[instance_index]
            raw = parse_geometry(loaded, instance_info.geometry_id)
            original_geometry = geometry_lookup[instance_info.geometry_id]
            geometry_id = _unique_id(
                f"{instance_info.geometry_id}__beamxp_rigid_{candidate_id:03d}", used_ids  # type: ignore[arg-type]
            )
            geometry = _subset_geometry(
                original_geometry,
                raw,
                primitive_selection,
                geometry_id,
                f"{original_geometry.get('name') or instance_info.geometry_id} BeamXP rigid {candidate_id:03d}",
                namespace,
                used_ids,
                reverse_winding=False,
            )
            generated_geometries.append(geometry)
            generated_geometry_ids.append(geometry_id)
            instance_geometry_ids.append(geometry_id)
            triangle_count += sum(len(indices) for indices in primitive_selection.values())
            instance_geometry = copy.deepcopy(original_instance)
            instance_geometry.set("url", f"#{geometry_id}")
            node.append(instance_geometry)
        preview_scene.append(node)
        generated_candidates.append(
            {
                "candidate_id": candidate_id,
                "node_id": node_id,
                "geometry_ids": instance_geometry_ids,
                "triangle_count": triangle_count,
                "accepted_at_degrees": candidate.accepted_angle,
                "symmetry_plane_centroid_world_metres": list(candidate.measurement.centroid),
                "initial_symmetry_plane_normal_world": list(candidate.measurement.initial_plane_normal),
                "symmetry_plane_normal_world": list(candidate.measurement.plane_normal),
                "initial_rms_error_millimetres": candidate.measurement.initial_rms_error * 1000.0,
                "corrected_rms_error_millimetres": candidate.measurement.rms_error * 1000.0,
                "mirror_plane_y_tilt_degrees": candidate.measurement.mirror_plane_y_tilt_degrees,
                "nominal_rigid_y_rotation_correction_degrees": candidate.measurement.rigid_y_rotation_correction_degrees,
                "tilt_search_applied": candidate.measurement.tilt_search_applied,
                "rigid_yaw_degrees": yaw_degrees,
                "rigid_translation_world_metres": [float(value) for value in translation_metres],
                "rigid_rotation_matrix_world": rotation.tolist(),
                "node_matrix_dae_units": node_matrix.tolist(),
            }
        )

    # The preview DAE contains only the generated selected-part geometries. Keep
    # material/effect/image libraries from the source because cloned bindings may
    # reference them, but discard unrelated source meshes and static scene nodes.
    for child in list(library):
        library.remove(child)
    for geometry in generated_geometries:
        library.append(geometry)

    # Controllers/animations can retain references to geometry that was pruned;
    # this output is intentionally a static transform preview.
    removable_libraries = {
        "library_animations",
        "library_animation_clips",
        "library_controllers",
        "library_nodes",
        "library_cameras",
        "library_lights",
        "library_force_fields",
        "library_physics_materials",
        "library_physics_models",
        "library_physics_scenes",
    }
    for child in list(root):
        if local_name(child.tag) in removable_libraries:
            root.remove(child)

    blender_texture_info: dict[str, object] | None = None
    if blender_base_colours:
        blender_texture_info = _wire_blender_base_colours(
            root,
            namespace,
            generated_geometries,
            blender_base_colours,
            output_path,
        )
    elif blender_base_colour is not None:
        blender_texture_info = _wire_blender_base_colour(
            root,
            namespace,
            generated_geometries,
            blender_base_colour,
            output_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "output_dae": str(output_path),
        "selected_node": result.part.node_id or result.part.node_name,
        "scope": "selected part only; meshes remain separate; no stitching",
        "global_mirror_plane": "world X = 0",
        "carrier": {
            "node_id": carrier_id if residual_triangle_count else None,
            "geometry_ids": residual_geometry_ids,
            "triangle_count": residual_triangle_count,
            "node_matrix_dae_units": carrier_matrix.tolist(),
            "node_linear_determinant": float(np.linalg.det(carrier_matrix[:3, :3])),
            "reflection_baked_into_geometry": True,
            "carrier_local_reflection_matrix": carrier_local_reflection.tolist(),
            "positions_transformed": True,
            "normals_transformed_and_normalised": True,
            "winding_reversed": True,
        },
        "rigid_symmetric_nodes": generated_candidates,
        "generated_geometry_ids": generated_geometry_ids,
        "blender_texture": blender_texture_info,
    }


# ---------------------------------------------------------------------------
# Diagnostic exports
# ---------------------------------------------------------------------------


def _candidate_transform_metres(candidate: AcceptedCandidate) -> np.ndarray:
    normal = np.asarray(candidate.measurement.plane_normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    centroid = np.asarray(candidate.measurement.centroid, dtype=float)
    local = np.eye(4, dtype=float)
    local[:3, :3] = np.eye(3, dtype=float) - 2.0 * np.outer(normal, normal)
    local[:3, 3] = 2.0 * normal * float(np.dot(normal, centroid))
    return _global_x_reflection() @ local


def write_transformed_result_obj(path: Path, result: SymmetrySweepResult) -> None:
    """Small geometry-only companion preview; COLLADA remains authoritative."""
    candidate_by_face = np.full(len(result.topology.triangles), -1, dtype=np.int64)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    for candidate in result.candidates:
        candidate_by_face[list(candidate.faces)] = candidate.candidate_id

    grouped_faces: dict[int, list[int]] = defaultdict(list)
    for face_index, candidate_id in enumerate(candidate_by_face):
        grouped_faces[int(candidate_id)].append(face_index)

    vertex_offset = 1
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# BeamXP transformed selected-part preview\n")
        for candidate_id in sorted(grouped_faces):
            face_indices = grouped_faces[candidate_id]
            if not face_indices:
                continue
            if candidate_id < 0:
                group = "mirrored_carrier"
                transform = _global_x_reflection()
            else:
                group = f"rigid_symmetric_{candidate_id:03d}"
                transform = _candidate_transform_metres(candidate_by_id[candidate_id])
            handle.write(f"g {group}\n")
            original_vertices = sorted(
                {int(vertex) for face in face_indices for vertex in result.topology.triangles[face]}
            )
            remap = {old: vertex_offset + index for index, old in enumerate(original_vertices)}
            points = result.topology.vertices[np.asarray(original_vertices, dtype=np.int64)]
            homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=float)], axis=1)
            transformed = (homogeneous @ transform.T)[:, :3]
            for x, y, z in transformed:
                handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for face_index in face_indices:
                triangle = [int(value) for value in result.topology.triangles[face_index]]
                if candidate_id < 0:
                    triangle.reverse()
                handle.write("f " + " ".join(str(remap[value]) for value in triangle) + "\n")
            vertex_offset += len(original_vertices)


def _candidate_json(candidate: AcceptedCandidate) -> dict[str, object]:
    measurement = candidate.measurement
    return {
        "candidate_id": candidate.candidate_id,
        "island_index": candidate.island_index,
        "accepted_level": candidate.accepted_level,
        "accepted_crease_angle_degrees": candidate.accepted_angle,
        "triangle_count": candidate.triangle_count,
        "host_triangle_count": len(candidate.host_faces),
        "adopted_triangle_count": sum(item.triangle_count for item in candidate.adoptions),
        "adopted_child_count": len(candidate.adoptions),
        "area_square_metres": candidate.area,
        "perimeter_metres": candidate.perimeter,
        "area_perimeter_ratio_metres": candidate.area_perimeter_ratio,
        "area_perimeter_ratio_millimetres": candidate.area_perimeter_ratio * 1000.0,
        "boundary_edge_count": len(candidate.boundary_edges),
        "boundary_loop_count": len(candidate.boundary_loops),
        "symmetry": {
            "centroid_world_metres": list(measurement.centroid),
            "initial_plane_normal_world": list(measurement.initial_plane_normal),
            "final_plane_normal_world": list(measurement.plane_normal),
            "tilt_search_applied": measurement.tilt_search_applied,
            "mirror_plane_y_tilt_degrees": measurement.mirror_plane_y_tilt_degrees,
            "nominal_rigid_y_rotation_correction_degrees": measurement.rigid_y_rotation_correction_degrees,
            "initial_post_reflection_variance_square_metres": measurement.initial_post_reflection_variance,
            "initial_post_reflection_variance_square_millimetres": measurement.initial_post_reflection_variance * 1_000_000.0,
            "initial_rms_error_metres": measurement.initial_rms_error,
            "initial_rms_error_millimetres": measurement.initial_rms_error * 1000.0,
            "initial_max_error_metres": measurement.initial_max_error,
            "initial_max_error_millimetres": measurement.initial_max_error * 1000.0,
            "whole_loop_sample_count": measurement.whole_loop_sample_count,
            "sibling_pair_count": measurement.sibling_pair_count,
            "mirror_plane_crossings": measurement.mirror_crossings,
            "post_reflection_variance_square_metres": measurement.post_reflection_variance,
            "post_reflection_variance_square_millimetres": measurement.post_reflection_variance * 1_000_000.0,
            "rms_error_metres": measurement.rms_error,
            "rms_error_millimetres": measurement.rms_error * 1000.0,
            "max_error_metres": measurement.max_error,
            "max_error_millimetres": measurement.max_error * 1000.0,
        },
        "adopted_children": [
            {
                "island_index": adoption.island_index,
                "adoption_mode": adoption.adoption_mode,
                "source_candidate_id_before_merge": adoption.source_candidate_id,
                "triangle_count": adoption.triangle_count,
                "area_square_metres": adoption.area,
                "child_to_host_area_ratio": adoption.area_ratio,
                "projected_footprint_overlap": adoption.projected_overlap,
                "median_surface_gap_metres": adoption.median_surface_gap,
                "median_surface_gap_millimetres": adoption.median_surface_gap * 1000.0,
                "p90_surface_gap_metres": adoption.p90_surface_gap,
                "p90_surface_gap_millimetres": adoption.p90_surface_gap * 1000.0,
                "face_indices": list(adoption.faces),
            }
            for adoption in candidate.adoptions
        ],
        "host_face_indices": list(candidate.host_faces),
        "face_indices": list(candidate.faces),
        "boundary_edges": [list(edge) for edge in candidate.boundary_edges],
    }


def write_report(
    path: Path,
    loaded: LoadedDae,
    result: SymmetrySweepResult,
    dae_export: dict[str, object] | None = None,
) -> None:
    report = {
        "application": APP_NAME,
        "source_dae": str(loaded.path),
        "part": {
            "key": result.part.key,
            "node_id": result.part.node_id,
            "node_name": result.part.node_name,
            "geometry_ids": [instance.geometry_id for instance in result.part.instances],
        },
        "parameters": {
            "maximum_crease_angle_degrees": result.crease_max,
            "minimum_crease_angle_degrees": result.crease_min,
            "threshold_steps_inclusive": result.threshold_steps,
            "thresholds_descending": list(result.thresholds),
            "minimum_region_faces": result.min_region_faces,
            "minimum_area_perimeter_ratio_metres": result.min_area_perimeter_ratio_metres,
            "minimum_area_perimeter_ratio_millimetres": result.min_area_perimeter_ratio_metres * 1000.0,
            "symmetry_rms_tolerance_metres": result.symmetry_tolerance_metres,
            "symmetry_rms_tolerance_millimetres": result.symmetry_tolerance_metres * 1000.0,
            "direct_symmetry_rms_tolerance_metres": result.direct_symmetry_tolerance_metres,
            "direct_symmetry_rms_tolerance_millimetres": result.direct_symmetry_tolerance_metres * 1000.0,
            "perimeter_sample_spacing_metres": result.sample_spacing_metres,
            "perimeter_sample_spacing_millimetres": result.sample_spacing_metres * 1000.0,
        },
        "method": {
            "segmentation": "at each global threshold, active adjacent triangles join when their dihedral angle is below the threshold",
            "recursion": "passing host faces and confidently adopted whole child islands are removed globally, then the same threshold is segmented again until no further host passes",
            "main_island": "largest surface-area island; regions touching its original mesh perimeter are never tested",
            "shape_filter": "before symmetry testing, candidate surface area divided by total closed-boundary perimeter must meet the configured absolute minimum; this characteristic length rejects thin bands",
            "symmetry_scope": "closed perimeter samples only; interior triangles, normals, materials and UVs are ignored",
            "plane": "first pass resamples complete closed perimeters at a fixed physical spacing; PCA gives the best-fit perimeter-plane normal; crossing it with global Z gives the deterministic vertical mirror-plane normal through the sample centroid",
            "comparison": "second pass inserts exact mirror-plane crossings, walks opposite half-perimeters in opposite directions, resamples sibling positions at equal travelled arc distance, reflects one half and averages the full 3-D squared sibling residuals",
            "acceptance": "candidate A/P characteristic length must meet the configured minimum and the initial post-reflection RMS must not exceed the outer tolerance; candidates below the stricter direct threshold use the deterministic plane unchanged",
            "borderline_tilt_correction": "accepted candidates between the direct and outer RMS thresholds sweep the original mirror-plane normal about global Y over +/-6 degrees, refine the local minimum, and store the corrected plane; composing that reflection with the global X reflection applies the corresponding double-angle rigid correction",
            "eager_child_adoption": "after a host passes, untouched whole islands are adopted before symmetry testing when child area is at most 80% of host area, host PCA thickness RMS is at most 6 mm, at least 80% of the child projected footprint lies over the host, median sampled surface gap is at most 15 mm, and the 90th percentile gap is at most 25 mm; only one direct child generation is used",
            "late_host_recovery": "after the sweep, a larger candidate discovered at a later threshold may absorb an earlier accepted candidate from another geometric island using the same area, planarity, footprint-overlap and sampled-gap tests; this recovers parents that only separate from the main carrier at low crease angles",
        },
        "timing": {
            "processing_seconds": result.processing_seconds,
            "topology_seconds": result.topology_seconds,
            "sweep_seconds": result.sweep_seconds,
        },
        "mesh": {
            "vertices": len(result.topology.vertices),
            "triangles": len(result.topology.triangles),
            "islands": len(result.island_faces),
            "symmetric_candidates": len(result.candidates),
            "separated_triangles": sum(candidate.triangle_count for candidate in result.candidates),
            "adopted_children": result.adopted_child_count,
            "eager_adopted_islands": result.adopted_island_count,
            "late_adopted_candidates": result.late_adopted_candidate_count,
            "adopted_triangles": result.adopted_triangle_count,
            "remaining_mirror_triangles": len(result.remaining_faces),
        },
        "candidates": [_candidate_json(candidate) for candidate in result.candidates],
        "levels": [
            {
                "level": level.level_index,
                "crease_angle_degrees": level.crease_angle,
                "recursion_passes_across_islands": level.recursion_passes,
                "tested_candidates": level.tested_candidates,
                "accepted_this_level": list(level.accepted_this_level),
                "accepted_cumulative": list(level.accepted_cumulative),
                "remaining_active_regions": len(level.active_regions),
                "remaining_active_faces": int(np.count_nonzero(level.active_face_labels >= 0)),
            }
            for level in result.levels
        ],
        "dae_export": dae_export,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------


def colour_for_candidate(candidate_id: int) -> str:
    # Stable high-contrast palette generated arithmetically without dependencies.
    hue = (candidate_id * 0.61803398875) % 1.0
    sector = int(hue * 6)
    fraction = hue * 6 - sector
    value = 230
    low = 85
    middle = round(low + (value - low) * fraction)
    options = [
        (value, middle, low),
        (middle, value, low),
        (low, value, middle),
        (low, middle, value),
        (middle, low, value),
        (value, low, middle),
    ]
    red, green, blue = options[sector % 6]
    return f"#{red:02x}{green:02x}{blue:02x}"


class SymmetrySweepProbeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1450x900")
        self.minsize(1120, 720)

        self.loaded: LoadedDae | None = None
        self.result: SymmetrySweepResult | None = None
        self.archive: VehicleArchive | None = None
        self.archive_dae_member: str | None = None
        self.archive_texture_bindings: tuple[ArchiveTextureBinding, ...] = ()
        self.archive_texture_choices: tuple[ArchiveTextureBinding, ...] = ()
        self.trim_texture_by_label: dict[str, ArchiveTextureBinding] = {}
        self.current_level_index = 0
        self.selected_candidate_id: int | None = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.dialog_settings_path = application_settings_path()
        self.dialog_directories = load_dialog_directories(self.dialog_settings_path)

        self.dae_path_var = tk.StringVar()
        self.archive_dae_var = tk.StringVar()
        self.auto_texture_var = tk.StringVar(value="No ZIP material resolution active.")
        self.part_var = tk.StringVar()
        self.trim_texture_var = tk.StringVar()
        self.search_var = tk.StringVar()
        self.crease_max_var = tk.StringVar(value="180")
        self.crease_min_var = tk.StringVar(value="15")
        self.steps_var = tk.StringVar(value="20")
        self.min_faces_var = tk.StringVar(value="10")
        self.min_area_perimeter_var = tk.StringVar(value="4.5")
        self.tolerance_var = tk.StringVar(value="1")
        self.direct_tolerance_var = tk.StringVar(value="0.5")
        self.spacing_var = tk.StringVar(value="2.0")
        self.threshold_var = tk.StringVar()
        self.projection_var = tk.StringVar(value="Front (X–Z)")
        self.show_boundaries_var = tk.BooleanVar(value=True)
        self.show_wireframe_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Select a BeamNG vehicle ZIP or COLLADA DAE to begin.")
        self.stats_var = tk.StringVar()

        self.part_by_label: dict[str, DaePart] = {}
        self.all_part_labels: list[str] = []
        self.archive_dae_by_label: dict[str, str] = {}
        self._build_ui()
        self.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(2, weight=1)

        file_frame = ttk.LabelFrame(outer, text="Vehicle source", padding=8)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        file_frame.columnconfigure(1, weight=1)
        ttk.Label(file_frame, text="ZIP or DAE").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(file_frame, textvariable=self.dae_path_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(file_frame, text="Browse…", command=self._browse).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(file_frame, text="Load", command=self._load_entry).grid(row=0, column=3, padx=(8, 0))

        ttk.Label(file_frame, text="DAE in archive").grid(row=1, column=0, padx=(0, 8), pady=(6, 0))
        self.archive_dae_combo = ttk.Combobox(
            file_frame, textvariable=self.archive_dae_var, state="disabled"
        )
        self.archive_dae_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.archive_dae_combo.bind("<<ComboboxSelected>>", self._select_archive_dae)
        self.archive_dae_button = ttk.Button(
            file_frame, text="Load selected DAE", command=self._load_selected_archive_dae, state="disabled"
        )
        self.archive_dae_button.grid(row=1, column=2, columnspan=2, padx=(8, 0), pady=(6, 0), sticky="ew")

        controls = ttk.LabelFrame(outer, text="Part, crease sweep and perimeter symmetry", padding=8)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 10))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Filter parts").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.search_var).grid(row=0, column=1, sticky="ew")
        self.search_var.trace_add("write", lambda *_: self._filter_parts())
        ttk.Label(controls, text="Part").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.part_combo = ttk.Combobox(controls, textvariable=self.part_var, state="readonly")
        self.part_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.part_combo.bind("<<ComboboxSelected>>", self._part_changed)

        ttk.Label(controls, text="Trim texture").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.trim_texture_combo = ttk.Combobox(
            controls, textvariable=self.trim_texture_var, state="disabled"
        )
        self.trim_texture_combo.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        self.trim_texture_combo.bind("<<ComboboxSelected>>", self._trim_texture_changed)
        ttk.Label(controls, textvariable=self.auto_texture_var).grid(
            row=3, column=1, sticky="w", pady=(3, 0)
        )

        params = ttk.Frame(controls)
        params.grid(row=0, column=2, rowspan=4, padx=(16, 0))
        labels = (
            ("Max angle (°)", self.crease_max_var, 7),
            ("Min angle (°)", self.crease_min_var, 7),
            ("Steps", self.steps_var, 6),
            ("Minimum faces", self.min_faces_var, 7),
            ("Min A/P (mm)", self.min_area_perimeter_var, 7),
            ("RMS tol. (mm)", self.tolerance_var, 7),
            ("Direct RMS (mm)", self.direct_tolerance_var, 7),
            ("Sample spacing (mm)", self.spacing_var, 7),
        )
        for index, (label, variable, width) in enumerate(labels):
            row = index // 4
            column = (index % 4) * 2
            ttk.Label(params, text=label).grid(row=row, column=column, sticky="w", pady=(7 if row else 0, 0))
            ttk.Entry(params, textvariable=variable, width=width).grid(
                row=row, column=column + 1, padx=(5, 12), pady=(7 if row else 0, 0)
            )
        self.run_button = ttk.Button(params, text="Run symmetry sweep", command=self._start_analysis)
        self.run_button.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(7, 0))

        left = ttk.Frame(outer)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, textvariable=self.stats_var, justify="left").grid(row=0, column=0, sticky="ew", pady=(0, 6))

        columns = (
            "id", "island", "angle", "faces", "children", "loops",
            "initial_rms", "rms", "ytilt", "max", "ap", "area",
        )
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=20)
        headings = {
            "id": "ID", "island": "Island", "angle": "Accepted °", "faces": "Faces",
            "children": "Children",
            "loops": "Loops", "initial_rms": "Initial RMS", "rms": "Final RMS",
            "ytilt": "Plane Y°", "max": "Max mm", "ap": "A/P mm", "area": "Area m²",
        }
        widths = {
            "id": 38, "island": 48, "angle": 72, "faces": 58, "children": 58, "loops": 48,
            "initial_rms": 76, "rms": 70, "ytilt": 66, "max": 70, "ap": 70, "area": 78,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<<TreeviewSelect>>", self._select_candidate)

        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.export_button = ttk.Button(buttons, text="Export transformed part DAE…", command=self._export, state="disabled")
        self.export_button.pack(side="left")
        ttk.Button(buttons, text="Clear selection", command=self._clear_selection).pack(side="left", padx=(8, 0))

        preview = ttk.LabelFrame(outer, text="Recursive sweep preview", padding=6)
        preview.grid(row=2, column=1, sticky="nsew")
        preview.rowconfigure(1, weight=1)
        preview.columnconfigure(0, weight=1)
        preview_controls = ttk.Frame(preview)
        preview_controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(preview_controls, text="Threshold").pack(side="left")
        self.threshold_combo = ttk.Combobox(preview_controls, textvariable=self.threshold_var, state="readonly", width=18)
        self.threshold_combo.pack(side="left", padx=(6, 14))
        self.threshold_combo.bind("<<ComboboxSelected>>", self._select_threshold)
        ttk.Label(preview_controls, text="View").pack(side="left")
        projection = ttk.Combobox(
            preview_controls,
            textvariable=self.projection_var,
            state="readonly",
            width=15,
            values=("Front (X–Z)", "Top (X–Y)", "Side (Y–Z)"),
        )
        projection.pack(side="left", padx=(6, 14))
        projection.bind("<<ComboboxSelected>>", lambda _event: self._draw())
        ttk.Checkbutton(preview_controls, text="Boundaries", variable=self.show_boundaries_var, command=self._draw).pack(side="left")
        ttk.Checkbutton(preview_controls, text="Wireframe", variable=self.show_wireframe_var, command=self._draw).pack(side="left", padx=(8, 0))
        self.canvas = tk.Canvas(preview, background="#171a1f", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw())

        status = ttk.Frame(outer)
        status.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1)

    def _dialog_initial_directory(
        self,
        key: str,
        fallback: Path | None = None,
    ) -> str:
        """Return the most useful existing directory for one file dialog."""
        candidates: list[Path] = []
        stored = self.dialog_directories.get(key)
        if stored:
            candidates.append(Path(stored).expanduser())
        if fallback is not None:
            candidates.append(fallback.expanduser())
        candidates.extend((Path.home(), Path.cwd()))
        for candidate in candidates:
            try:
                if candidate.is_dir():
                    return str(candidate.resolve())
            except OSError:
                continue
        return str(Path.cwd())

    def _remember_dialog_directory(self, key: str, selected_path: Path) -> None:
        if key not in DIALOG_DIRECTORY_KEYS:
            return
        directory = selected_path if selected_path.is_dir() else selected_path.parent
        try:
            directory = directory.expanduser().resolve()
        except OSError:
            directory = directory.expanduser()
        if not directory.is_dir():
            return
        self.dialog_directories[key] = str(directory)
        save_dialog_directories(self.dialog_directories, self.dialog_settings_path)

    def _source_directory_fallback(self) -> Path | None:
        value = self.dae_path_var.get().strip().strip('"')
        if value:
            path = Path(value).expanduser()
            return path if path.is_dir() else path.parent
        if self.loaded is not None:
            return self.loaded.path.parent
        return None

    def _browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select BeamNG vehicle ZIP or COLLADA DAE",
            initialdir=self._dialog_initial_directory("source"),
            filetypes=(
                ("BeamNG vehicle archives and DAE", "*.zip *.dae"),
                ("ZIP archives", "*.zip"),
                ("COLLADA DAE", "*.dae"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            selected_path = Path(selected)
            self._remember_dialog_directory("source", selected_path)
            self.dae_path_var.set(selected)
            self._start_source_load(selected_path)

    def _load_entry(self) -> None:
        value = self.dae_path_var.get().strip().strip('"')
        if not value:
            messagebox.showerror(APP_NAME, "Choose a vehicle ZIP or DAE first.")
            return
        self._start_source_load(Path(value))

    def _cleanup_archive(self) -> None:
        archive = self.archive
        self.archive = None
        self.archive_dae_member = None
        self.archive_texture_bindings = ()
        self.archive_texture_choices = ()
        self.trim_texture_by_label.clear()
        self.trim_texture_var.set("")
        self.trim_texture_combo.configure(values=(), state="disabled")
        self.archive_dae_by_label.clear()
        self.archive_dae_var.set("")
        self.archive_dae_combo.configure(values=(), state="disabled")
        self.archive_dae_button.configure(state="disabled")
        self.auto_texture_var.set("No ZIP material resolution active.")
        if archive is not None:
            shutil.rmtree(archive.workspace, ignore_errors=True)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.status_var.set(text)
        if busy:
            self.progress.start(12)
            self.run_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
            self.archive_dae_button.configure(state="disabled")
        else:
            self.progress.stop()
            self.run_button.configure(state="normal")
            self.export_button.configure(state=("normal" if self.result else "disabled"))
            self.archive_dae_button.configure(state=("normal" if self.archive else "disabled"))

    def _reset_loaded_mesh(self) -> None:
        self.loaded = None
        self.result = None
        self.part_var.set("")
        self.part_combo.configure(values=())
        self.part_by_label.clear()
        self.all_part_labels.clear()
        self.archive_texture_bindings = ()
        self.archive_texture_choices = ()
        self.trim_texture_by_label.clear()
        self.trim_texture_var.set("")
        self.trim_texture_combo.configure(values=(), state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.canvas.delete("all")
        self.auto_texture_var.set(
            "Archive loaded; select a DAE and part." if self.archive else "No ZIP material resolution active."
        )

    def _start_source_load(self, path: Path) -> None:
        if not path.is_file():
            messagebox.showerror(APP_NAME, f"File not found:\n{path}")
            return
        if path.suffix.lower() == ".zip":
            self._start_archive_scan(path)
        elif path.suffix.lower() == ".dae":
            self._cleanup_archive()
            self._start_load(path, None)
        else:
            messagebox.showerror(APP_NAME, "Choose a .zip vehicle archive or .dae file.")

    def _start_archive_scan(self, path: Path) -> None:
        self._cleanup_archive()
        self._reset_loaded_mesh()
        self._set_busy(True, f"Indexing {path.name} without extracting it…")

        def work() -> None:
            workspace: Path | None = None
            try:
                workspace = Path(tempfile.mkdtemp(prefix="beamxp_vehicle_zip_"))
                archive = scan_vehicle_archive(path, workspace)
                self.worker_queue.put(("archive_scanned", archive))
            except Exception:
                if workspace is not None:
                    shutil.rmtree(workspace, ignore_errors=True)
                self.worker_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _select_archive_dae(self, _event: object = None) -> None:
        label = self.archive_dae_var.get()
        member = self.archive_dae_by_label.get(label)
        if member:
            self.status_var.set(f"Selected archive DAE: {member}")

    def _load_selected_archive_dae(self) -> None:
        label = self.archive_dae_var.get()
        member = self.archive_dae_by_label.get(label)
        if self.archive is None or member is None:
            messagebox.showerror(APP_NAME, "Choose a DAE from the archive.")
            return
        self._start_archive_dae_load(member)

    def _start_archive_dae_load(self, member: str) -> None:
        archive = self.archive
        if archive is None:
            return
        self._reset_loaded_mesh()
        self._set_busy(True, f"Extracting and loading {PurePosixPath(member).name}…")

        def work() -> None:
            try:
                dae_path = extract_archive_member(archive, member)
                loaded = load_dae(dae_path)
                self.worker_queue.put(("loaded", (loaded, member)))
            except Exception:
                self.worker_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _start_load(self, path: Path, archive_member: str | None = None) -> None:
        if not path.is_file():
            messagebox.showerror(APP_NAME, f"File not found:\n{path}")
            return
        self._reset_loaded_mesh()
        self._set_busy(True, f"Loading {path.name}…")

        def work() -> None:
            try:
                self.worker_queue.put(("loaded", (load_dae(path), archive_member)))
            except Exception:
                self.worker_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _filter_parts(self) -> None:
        query = self.search_var.get().strip().lower()
        labels = [label for label in self.all_part_labels if query in label.lower()]
        self.part_combo.configure(values=labels)
        changed = self.part_var.get() not in labels
        if changed:
            self.part_var.set(labels[0] if labels else "")
        if changed or labels:
            self._part_changed()

    def _part_changed(self, _event: object = None) -> None:
        self.archive_texture_bindings = ()
        self.archive_texture_choices = ()
        self.trim_texture_by_label.clear()
        self.trim_texture_var.set("")
        self.trim_texture_combo.configure(values=(), state="disabled")

        if self.archive is None or self.loaded is None:
            self.auto_texture_var.set("Trim textures are resolved from a vehicle ZIP's materials JSON.")
            return
        part = self.part_by_label.get(self.part_var.get())
        if part is None:
            self.auto_texture_var.set("Select a part to resolve its valid trim textures.")
            return
        try:
            choices = archive_texture_choices_for_part(self.archive, self.loaded, part)
            defaults = archive_texture_bindings_for_part(self.archive, self.loaded, part)
        except Exception as exc:
            self.auto_texture_var.set(f"Material resolution failed: {exc}")
            return

        self.archive_texture_choices = choices
        self.archive_texture_bindings = defaults
        dae_materials = material_names_for_part(self.loaded, part)
        if not choices:
            names = ", ".join(dae_materials) or "no bound COLLADA material"
            self.auto_texture_var.set(f"No materials-JSON base colour resolved for: {names}")
            return

        filename_counts: dict[str, int] = defaultdict(int)
        for binding in choices:
            filename_counts[PurePosixPath(binding.texture_member).name.lower()] += 1

        labels: list[str] = []
        for binding in choices:
            filename = PurePosixPath(binding.texture_member).name
            label = filename
            if filename_counts[filename.lower()] > 1 or len(dae_materials) > 1:
                label += f"  —  {binding.dae_material} / {binding.material_key}"
            if label in self.trim_texture_by_label:
                label += f"  [{binding.materials_member}]"
            base_label = label
            suffix = 2
            while label in self.trim_texture_by_label:
                label = f"{base_label} #{suffix}"
                suffix += 1
            self.trim_texture_by_label[label] = binding
            labels.append(label)

        self.trim_texture_combo.configure(values=labels, state="readonly")
        default_binding = defaults[0] if defaults else choices[0]
        default_label = next(
            (label for label, binding in self.trim_texture_by_label.items() if binding == default_binding),
            labels[0],
        )
        self.trim_texture_var.set(default_label)
        self._trim_texture_changed()

    def _trim_texture_changed(self, _event: object = None) -> None:
        binding = self.trim_texture_by_label.get(self.trim_texture_var.get())
        if binding is None:
            if self.archive_texture_choices:
                self.auto_texture_var.set("Choose a trim texture resolved from the selected part's materials JSON.")
            return
        self.auto_texture_var.set(
            f"{binding.dae_material} via {binding.material_key} in {binding.materials_member}"
        )

    def _parameters(self) -> tuple[float, float, int, int, float, float, float, float] | None:
        try:
            maximum = float(self.crease_max_var.get())
            minimum = float(self.crease_min_var.get())
            steps = int(self.steps_var.get())
            min_faces = int(self.min_faces_var.get())
            min_area_perimeter_mm = float(self.min_area_perimeter_var.get())
            tolerance_mm = float(self.tolerance_var.get())
            direct_tolerance_mm = float(self.direct_tolerance_var.get())
            spacing_mm = float(self.spacing_var.get())
        except ValueError:
            messagebox.showerror(APP_NAME, "All analysis parameters must be numeric.")
            return None
        if not (0.0 < minimum <= maximum <= 180.0):
            messagebox.showerror(APP_NAME, "Angles must satisfy 0 < minimum ≤ maximum ≤ 180.")
            return None
        if not (1 <= steps <= 100):
            messagebox.showerror(APP_NAME, "Steps must be between 1 and 100.")
            return None
        if min_faces < 1:
            messagebox.showerror(APP_NAME, "Minimum faces must be at least 1.")
            return None
        if not (0.0 <= min_area_perimeter_mm <= 1000.0):
            messagebox.showerror(APP_NAME, "Minimum area/perimeter must be between 0 and 1000 mm. Use 0 to disable it.")
            return None
        if not (0.001 <= tolerance_mm <= 100.0):
            messagebox.showerror(APP_NAME, "RMS symmetry tolerance must be between 0.001 mm and 100 mm.")
            return None
        if not (0.001 <= direct_tolerance_mm <= tolerance_mm):
            messagebox.showerror(
                APP_NAME,
                "Direct RMS tolerance must be at least 0.001 mm and no greater than the outer RMS tolerance.",
            )
            return None
        if not (0.05 <= spacing_mm <= 100.0):
            messagebox.showerror(APP_NAME, "Perimeter sample spacing must be between 0.05 mm and 100 mm.")
            return None
        return (
            maximum, minimum, steps, min_faces, min_area_perimeter_mm / 1000.0,
            tolerance_mm / 1000.0, direct_tolerance_mm / 1000.0,
            spacing_mm / 1000.0,
        )

    def _start_analysis(self) -> None:
        if self.loaded is None:
            messagebox.showerror(APP_NAME, "Load a DAE first.")
            return
        part = self.part_by_label.get(self.part_var.get())
        if part is None:
            messagebox.showerror(APP_NAME, "Choose a part.")
            return
        parameters = self._parameters()
        if parameters is None:
            return
        (
            maximum, minimum, steps, min_faces, min_area_perimeter,
            tolerance, direct_tolerance, spacing,
        ) = parameters
        self.result = None
        self.current_level_index = 0
        self.selected_candidate_id = None
        self.tree.delete(*self.tree.get_children())
        self.threshold_combo.configure(values=())
        self.canvas.delete("all")
        self._set_busy(True, f"Running recursive symmetry sweep on {part.node_name}…")

        def work() -> None:
            try:
                assert self.loaded is not None
                result = analyse_symmetry_sweep(
                    self.loaded,
                    part,
                    maximum,
                    minimum,
                    steps,
                    min_faces,
                    min_area_perimeter,
                    tolerance,
                    direct_tolerance,
                    spacing,
                )
                self.worker_queue.put(("analysed", result))
            except Exception:
                self.worker_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "archive_scanned":
                    self.archive = payload  # type: ignore[assignment]
                    assert self.archive is not None
                    self.archive_dae_by_label = {
                        f"{member}  [{_human_size(self.archive.member_sizes.get(member, 0))}]": member
                        for member in self.archive.dae_members
                    }
                    labels = list(self.archive_dae_by_label)
                    self.archive_dae_combo.configure(values=labels, state="readonly")
                    self.archive_dae_var.set(labels[0] if labels else "")
                    self.archive_dae_button.configure(state=("normal" if labels else "disabled"))
                    warning = (
                        f"; {len(self.archive.material_errors)} material files skipped"
                        if self.archive.material_errors else ""
                    )
                    self._set_busy(
                        False,
                        f"Indexed {self.archive.path.name}: {len(labels)} DAE files, "
                        f"{len(self.archive.materials)} textured materials{warning}.",
                    )
                    if labels:
                        self._start_archive_dae_load(self.archive_dae_by_label[labels[0]])
                elif kind == "loaded":
                    loaded, archive_member = payload  # type: ignore[misc]
                    self.loaded = loaded
                    self.archive_dae_member = archive_member
                    self.part_by_label = {part.label: part for part in self.loaded.parts}
                    self.all_part_labels = list(self.part_by_label)
                    self._filter_parts()
                    source = archive_member or str(self.loaded.path)
                    self._set_busy(
                        False,
                        f"Loaded {source}: {len(self.loaded.parts)} selectable parts.",
                    )
                elif kind == "analysed":
                    self.result = payload  # type: ignore[assignment]
                    self._populate_result()
                    assert self.result is not None
                    self._set_busy(
                        False,
                        f"Accepted {len(self.result.candidates)} symmetric candidates; "
                        f"adopted {self.result.adopted_child_count} child meshes; "
                        f"{len(self.result.remaining_faces)} triangles remain on the mirror geometry; "
                        f"processed in {self.result.processing_seconds:.3f} s.",
                    )
                elif kind == "error":
                    self._set_busy(False, "Operation failed.")
                    messagebox.showerror(APP_NAME, str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_worker)

    def _populate_result(self) -> None:
        assert self.result is not None
        labels = [f"{level.level_index}: {level.crease_angle:.6g}°" for level in self.result.levels]
        self.threshold_combo.configure(values=labels)
        self.current_level_index = 0
        self.threshold_var.set(labels[0] if labels else "")
        self.tree.delete(*self.tree.get_children())
        for candidate in self.result.candidates:
            self.tree.insert(
                "", "end", iid=str(candidate.candidate_id),
                values=(
                    candidate.candidate_id,
                    candidate.island_index,
                    f"{candidate.accepted_angle:.6g}",
                    candidate.triangle_count,
                    len(candidate.adoptions),
                    len(candidate.boundary_loops),
                    f"{candidate.measurement.initial_rms_error * 1000.0:.4f}",
                    f"{candidate.measurement.rms_error * 1000.0:.4f}",
                    f"{candidate.measurement.mirror_plane_y_tilt_degrees:.3f}",
                    f"{candidate.measurement.max_error * 1000.0:.4f}",
                    f"{candidate.area_perimeter_ratio * 1000.0:.4f}",
                    f"{candidate.area:.6g}",
                ),
            )
        self._refresh_level()

    def _current_level(self) -> SweepLevelResult | None:
        if self.result is None or not self.result.levels:
            return None
        return self.result.levels[self.current_level_index]

    def _select_threshold(self, _event: object = None) -> None:
        selected = self.threshold_combo.current()
        if selected >= 0:
            self.current_level_index = selected
            self._refresh_level()

    def _refresh_level(self) -> None:
        result = self.result
        level = self._current_level()
        if result is None or level is None:
            return
        self.stats_var.set(
            f"{len(result.topology.vertices):,} vertices   {len(result.topology.triangles):,} triangles   "
            f"{len(result.island_faces):,} islands\n"
            f"Level {level.level_index}/{len(result.levels)}   {level.crease_angle:.6g}°   "
            f"{level.tested_candidates:,} tested   {len(level.accepted_this_level):,} accepted here\n"
            f"{len(level.accepted_cumulative):,} accepted cumulatively   "
            f"{result.adopted_child_count:,} adopted children / {result.adopted_triangle_count:,} triangles "
            f"({result.adopted_island_count:,} eager, {result.late_adopted_candidate_count:,} late)   "
            f"{len(level.active_regions):,} residual regions\n"
            f"Processed in {result.processing_seconds:.3f} s   "
            f"topology {result.topology_seconds:.3f} s   sweep/adoption {result.sweep_seconds:.3f} s"
        )
        self._draw()

    def _select_candidate(self, _event: object = None) -> None:
        selected = self.tree.selection()
        self.selected_candidate_id = int(selected[0]) if selected else None
        self._draw()

    def _clear_selection(self) -> None:
        for item in self.tree.selection():
            self.tree.selection_remove(item)
        self.selected_candidate_id = None
        self._draw()

    def _projection_axes(self) -> tuple[tuple[int, int], int]:
        view = self.projection_var.get()
        if view.startswith("Top"):
            return (0, 1), 2
        if view.startswith("Side"):
            return (1, 2), 0
        return (0, 2), 1

    def _draw(self) -> None:
        self.canvas.delete("all")
        result = self.result
        level = self._current_level()
        if result is None or level is None or self.canvas.winfo_width() < 20:
            return
        topology = result.topology
        axes, depth_axis = self._projection_axes()
        projected = topology.vertices[:, list(axes)]
        minimum, maximum = projected.min(axis=0), projected.max(axis=0)
        extent = np.maximum(maximum - minimum, 1e-12)
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        margin = 24.0
        scale = min((width - 2 * margin) / extent[0], (height - 2 * margin) / extent[1])
        centre = (minimum + maximum) * 0.5
        screen = np.empty_like(projected)
        screen[:, 0] = (projected[:, 0] - centre[0]) * scale + width * 0.5
        screen[:, 1] = height * 0.5 - (projected[:, 1] - centre[1]) * scale
        depths = topology.vertices[topology.triangles][:, :, depth_axis].mean(axis=1)

        for face_index in np.argsort(depths):
            face_index = int(face_index)
            candidate_label = int(level.candidate_face_labels[face_index])
            active_label = int(level.active_face_labels[face_index])
            if candidate_label >= 0:
                candidate_id = candidate_label + 1
                fill = colour_for_candidate(candidate_id)
                selected = candidate_id == self.selected_candidate_id
            elif active_label >= 0:
                region = level.active_regions[active_label]
                if region.role == "main carrier":
                    fill = "#46515d"
                elif region.role == "thin/rejected":
                    fill = "#5a3939"
                else:
                    fill = "#303942"
                selected = False
            else:
                fill = "#24292f"
                selected = False
            coordinates: list[float] = []
            for vertex in topology.triangles[face_index]:
                point = screen[int(vertex)]
                coordinates.extend((float(point[0]), float(point[1])))
            self.canvas.create_polygon(
                *coordinates,
                fill=("#ffe18a" if selected else fill),
                outline=("#fff4c2" if selected else fill),
                width=(2 if selected else 1),
            )

        def point(vertex: int) -> tuple[float, float]:
            value = screen[vertex]
            return float(value[0]), float(value[1])

        if self.show_wireframe_var.get():
            for first, second in topology.edge_faces:
                self.canvas.create_line(*point(first), *point(second), fill="#343b44")
        if self.show_boundaries_var.get():
            boundary_edges: set[tuple[int, int]] = set()
            for region in level.active_regions:
                boundary_edges.update(region.boundary.edges)
            for candidate_id in level.accepted_cumulative:
                boundary_edges.update(result.candidates[candidate_id - 1].boundary_edges)
            for first, second in boundary_edges:
                self.canvas.create_line(*point(first), *point(second), fill="#111418", width=2)

    def _export(self) -> None:
        if self.loaded is None or self.result is None:
            return
        source_stem = PurePosixPath(self.archive_dae_member).stem if self.archive_dae_member else self.loaded.path.stem
        default = f"{safe_name(source_stem)}_{safe_name(self.result.part.node_name)}_transformed_preview.dae"
        selected = filedialog.asksaveasfilename(
            title="Save transformed selected-part COLLADA DAE",
            defaultextension=".dae",
            initialdir=self._dialog_initial_directory(
                "export",
                self._source_directory_fallback(),
            ),
            initialfile=default,
            filetypes=(("COLLADA DAE", "*.dae"),),
        )
        if not selected:
            return
        output = Path(selected)
        self._remember_dialog_directory("export", output)
        automatic_textures: dict[str, Path] = {}
        automatic_bindings: tuple[ArchiveTextureBinding, ...] = ()
        selected_trim = self.trim_texture_by_label.get(self.trim_texture_var.get())
        try:
            if self.archive is not None:
                selected_bindings = list(
                    archive_texture_bindings_for_part(
                        self.archive,
                        self.loaded,
                        self.result.part,
                    )
                )
                if selected_trim is not None:
                    selected_material = _normalise_material_alias(selected_trim.dae_material)
                    replaced = False
                    for index, binding in enumerate(selected_bindings):
                        if _normalise_material_alias(binding.dae_material) == selected_material:
                            selected_bindings[index] = selected_trim
                            replaced = True
                            break
                    if not replaced:
                        selected_bindings.append(selected_trim)
                automatic_bindings = tuple(selected_bindings)
                for binding in automatic_bindings:
                    extracted = extract_archive_member(self.archive, binding.texture_member)
                    extracted = _blend_archive_preview_texture(extracted, self.archive, binding)
                    automatic_textures[binding.dae_material] = extracted
                    automatic_textures[binding.material_key] = extracted
                if selected_trim is not None and not automatic_textures:
                    raise ValueError(
                        "The selected trim resolved from materials JSON but no archive texture was extracted."
                    )

            export_info = export_transformed_part_dae(
                self.loaded,
                self.result,
                output,
                blender_base_colours=(automatic_textures or None),
            )
            if self.archive is not None:
                export_info["source_archive"] = str(self.archive.path)
                export_info["source_dae_member"] = self.archive_dae_member
                export_info["selected_trim_texture"] = (
                    {
                        "dae_material": selected_trim.dae_material,
                        "material_key": selected_trim.material_key,
                        "materials_member": selected_trim.materials_member,
                        "texture_reference": selected_trim.texture_reference,
                        "texture_member": selected_trim.texture_member,
                        "preview_layers": [
                            {
                                "base_colour_reference": layer.base_colour_reference,
                                "base_colour_factor": list(layer.base_colour_factor),
                                "opacity_reference": layer.opacity_reference,
                            }
                            for layer in selected_trim.preview_layers
                        ],
                    }
                    if selected_trim is not None else None
                )
                export_info["archive_texture_bindings"] = [
                    {
                        "dae_material": binding.dae_material,
                        "material_key": binding.material_key,
                        "materials_member": binding.materials_member,
                        "texture_reference": binding.texture_reference,
                        "texture_member": binding.texture_member,
                        "preview_layers": [
                            {
                                "base_colour_reference": layer.base_colour_reference,
                                "base_colour_factor": list(layer.base_colour_factor),
                                "opacity_reference": layer.opacity_reference,
                            }
                            for layer in binding.preview_layers
                        ],
                    }
                    for binding in automatic_bindings
                ]
            report_path = output.with_suffix(".transform.json")
            obj_path = output.with_suffix(".transform.obj")
            write_report(report_path, self.loaded, self.result, export_info)
            write_transformed_result_obj(obj_path, self.result)
        except Exception:
            messagebox.showerror(APP_NAME, traceback.format_exc())
            return
        self.status_var.set(f"Exported transformed selected-part DAE to {output}")
        exported_lines = [
            str(output),
            str(output.with_suffix('.transform.json')),
            str(output.with_suffix('.transform.obj')),
        ]
        texture_info = export_info.get("blender_texture")
        warning_text = ""
        if isinstance(texture_info, dict):
            if texture_info.get("output_texture"):
                exported_lines.append(str(texture_info["output_texture"]))
            for item in texture_info.get("wired_materials", []):
                if isinstance(item, dict) and item.get("output_texture"):
                    value = str(item["output_texture"])
                    if value not in exported_lines:
                        exported_lines.append(value)
            if texture_info.get("enabled") is False:
                warning_text = (
                    "\n\nTexture preview warning:\n"
                    + str(texture_info.get("details") or texture_info.get("reason") or "Automatic texture wiring failed.")
                    + "\nThe transformed DAE was still exported normally."
                )
        messagebox.showinfo(APP_NAME, "Exported:\n" + "\n".join(exported_lines) + warning_text)

    def destroy(self) -> None:
        archive = self.archive
        self.archive = None
        if archive is not None:
            shutil.rmtree(archive.workspace, ignore_errors=True)
        super().destroy()


def main() -> None:
    app = SymmetrySweepProbeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
