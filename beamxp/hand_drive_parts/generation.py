"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 4885-5809. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import copy
import json
import math
import re
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from beamxp import transform_helpers
from beamxp.core.beam_json import (
    add_missing_json_commas,
    display_name_for,
    info_path_for_config,
    json_line_needs_comma,
    load_info,
    load_pc,
    parse_beamng_json,
    strip_json_comments,
    zip_json_by_name as _zip_json_by_name,
)
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
from beamxp.core.geometry import (
    PROP_VECTOR_RE,
    brg_rotation_matrix3,
    clamp_value,
    cross_product,
    euler_from_matrix3,
    euler_matrix3,
    euler_yzx_from_matrix3,
    identity_matrix,
    matrix3_from_axes,
    matrix3_from_matrix4,
    matrix4_flat,
    mirror_rotation_matrix_x,
    mirror_x_matrix4,
    multiply_matrix,
    multiply_matrix3,
    normalize_vector,
    prop_base_rotation_matrix3,
    prop_row_vector_objects,
    rotation_transpose_matrix3,
    rotation_transpose_matrix4,
    rotation_x_matrix,
    rotation_y_matrix,
    rotation_z_matrix,
    scale_matrix,
    sign_number,
    translation_matrix,
    vector_subtract,
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
from beamxp.plates import generator as plate_generator

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
    # Per-mesh nav-screen material scope: which UV island(s) to reflect. A
    # dedicated screen mesh maps to all its symbols (whole-mesh flip); a shared
    # mesh (nav screen + cluster) maps to only the screen island.
    nav_flip_scope = nav_screen_mesh_scope(context)
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
                                flip_materials=nav_flip_scope.get(object_id),
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
                            flip_materials=nav_flip_scope.get(spec.configured_mesh),
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
    variables: dict[str, float] | None = None,
) -> list[list[float]] | None:
    """Affine map from DAE-world coordinates to vehicle space for a prop row
    at rest: W = T(anchor) * R * T(-pivot), per the engine model verified
    against in-game dumps.

    rotation_override supplies the resolved engine rest rotation (authored
    baseRotationGlobal or analytic engine model)."""
    anchor = prop_row_pivot_position(row, node_positions, pivot, inherited_options, variables)
    if anchor is None:
        return None
    rotation = rotation_override
    if rotation is None:
        rotation = prop_row_global_rotation_matrix(row, node_positions)
    matrix = matrix4_with_rotation_translation(rotation, anchor)
    t = pivot or (0.0, 0.0, 0.0)
    return multiply_matrix(matrix, translation_matrix((-t[0], -t[1], -t[2])))


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
    rotation_counts: dict[str, int] = {}

    for part_instance in selected_part_instances(selected):
        part_id = str(part_instance.get("part_id") or "")
        instance_id = str(part_instance.get("instance_id") or "")
        slot_path = str(part_instance.get("slot_path") or "/")
        found = find_part_body(part_id, combined_jbeam_texts, combined_part_index)
        if found is None:
            skipped.setdefault(instance_id or part_id, "part body not found")
            continue
        part_body, _filename = found
        opts = part_instance_options(part_instance)
        variables = part_instance_variable_scope(selected, part_instance)
        for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
            array_text = transform_helpers.extract_named_array(part_body, array_key)
            if not array_text:
                continue
            for raw_row in iter_active_top_level_rows(array_text):
                row = resolve_jbeam_row_strings(raw_row, variables)
                mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                if not mesh or mesh in ("SPOTLIGHT", "POINTLIGHT"):
                    continue
                obj = output_objects.get(mesh) or context.objects.get(mesh)
                if obj is None or not obj.dae_path:
                    skipped.setdefault(mesh, "no DAE geometry indexed")
                    continue
                if kind == "prop" and not prop_row_nodes_present(row, selected_nodes):
                    skipped.setdefault(mesh, "inactive row (prop nodes not in this config)")
                    continue
                rotation_source = None
                if kind == "flex":
                    world = flexbody_row_source_matrix(row, opts, variables)
                else:
                    pivot = output_pivots.get(mesh) or context.mesh_pivots.get(mesh)
                    rotation_override, rotation_source = prop_rest_rotation_override(row, node_positions)
                    rotation_counts[rotation_source] = rotation_counts.get(rotation_source, 0) + 1
                    world = prop_row_world_matrix(
                        row, node_positions, pivot, opts, rotation_override, variables
                    )
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
                        "part_instance": instance_id,
                        "slot_path": slot_path,
                        "kind": kind,
                        "mode": "output" if is_generated else MODE_SKIP,
                        "matrix": matrix4_flat(world),
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
    selected_nodes = selected_node_positions_for_parts(
        selected,
        context.jbeam_texts,
        preview_part_index,
    )
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

    for part_instance in selected_part_instances(selected):
        part_id = str(part_instance.get("part_id") or "")
        instance_id = str(part_instance.get("instance_id") or "")
        slot_path = str(part_instance.get("slot_path") or "/")
        opts = part_instance_options(part_instance)
        variables = part_instance_variable_scope(selected, part_instance)
        for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
            array_text = preview_part_array(part_id, array_key)
            if not array_text:
                continue
            for raw_row in iter_active_top_level_rows(array_text):
                row = resolve_jbeam_row_strings(raw_row, variables)
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
                rotation_source = None
                if kind == "flex":
                    world = flexbody_row_source_matrix(row, opts, variables)
                else:
                    pivot = context.mesh_pivots.get(mesh)
                    rotation_override, rotation_source = prop_rest_rotation_override(row, node_positions)
                    rotation_counts[rotation_source] = rotation_counts.get(rotation_source, 0) + 1
                    world = prop_row_world_matrix(
                        row, node_positions, pivot, opts, rotation_override, variables
                    )
                if world is None:
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
                        "part_instance": instance_id,
                        "slot_path": slot_path,
                        "kind": kind,
                        "mode": mode if target_hand is not None else MODE_SKIP,
                        "matrix": matrix4_flat(final_matrix(mesh, mode, world)),
                        "stock_matrix": matrix4_flat(world),
                        "keep_node_translation": (
                            kind != "flex" or flexbody_row_needs_node_translation(context, geometry_mesh)
                        ),
                        **({"rotation_source": rotation_source} if rotation_source else {}),
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

__all__ = ['generate_daes', 'variant_output_name', 'original_plate_output_name', 'append_hand_label', 'write_generated_jbeam_and_configs', 'write_original_plate_configs', 'variant_target_hand', 'output_config_sources', 'load_beamng_json_file', 'prop_row_world_matrix', 'preview_node_names', 'extract_preview_dae', 'output_vehicle_preview_payload', 'full_vehicle_preview_payload']
