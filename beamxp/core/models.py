from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beamxp.core.dae import DaeObject


@dataclass(frozen=True)
class MeshPlacement:
    position: tuple[float, float, float]
    matrix: list[list[float]]


@dataclass(frozen=True)
class ResolvedMeshPosition:
    """Where a mesh sits in one configuration.

    Several placements within a single config are simultaneous instances, so
    position is their average. matrices are the flexbody row matrices for that
    config, empty when the mesh is placed only as a prop.
    """

    position: tuple[float, float, float]
    matrices: tuple[tuple[tuple[float, ...], ...], ...] = ()


@dataclass(frozen=True)
class SlotDef:
    # slot_type is the slot identifier: the type for a slots(v1) row, the name
    # for a slots2 row. allow_types/deny_types drive jbeam partFitsSlot fitment.
    slot_type: str
    default_part: str
    options: str | None = None
    allow_types: tuple[str, ...] = ()
    deny_types: tuple[str, ...] = ()


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
    context: VehicleContext
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
    mesh_pivots: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    mesh_authored_centers: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    variant_dependent_meshes: set[str] = field(default_factory=set)
    selected_parts_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    resolved_positions_cache: dict[str, dict[str, ResolvedMeshPosition]] = field(default_factory=dict)
    mesh_roles_cache: dict[str, tuple[set[str], set[str], set[str]]] = field(default_factory=dict)
    selected_node_positions_cache: dict[str, dict[str, tuple[float, float, float]]] = field(default_factory=dict)
    part_array_cache: dict[tuple[str, str], str | None] = field(default_factory=dict)
    variant_hands_cache: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def source_vehicle_id(self) -> str:
        parts = [part for part in self.vehicle_path.replace("\\", "/").split("/") if part]
        if len(parts) >= 2 and parts[0].lower() == "vehicles":
            return parts[1]
        return self.vehicle_id


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
