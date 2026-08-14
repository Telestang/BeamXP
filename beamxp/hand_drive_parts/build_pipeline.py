"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 5810-6096. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from beamxp import transform_helpers
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
    MODE_MIRROR_POSITION,
    MODE_REPLACE_SOURCE,
    MODE_SKIP,
    MODE_TRANSLATE,
    NS,
    NUMBER_RE,
    PART_TEXTURE_CORRECTION_KEY,
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
    vehicle_catalog_entry_for_id,
    vehicle_ids_in_zip,
    vehicle_prefix,
    write_bytes_file,
    write_text_file,
    write_xml_tree,
    zip_member_path,
)
from beamxp.core.beam_json import parse_beamng_json
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

MOD_AUTHOR = "Telestang - BeamXP"
MOD_VERSION = "0.2.1"


def package_stem_for_context(context: VehicleContext) -> str:
    """The one identity a conversion has: this vehicle, out of this zip.

    One zip routinely holds several vehicles -- vivace.zip carries the vivace,
    the ardente and the tograc -- and the tool converts one of them at a time.
    Named the way ``project_dir_for`` names the project it came from: the zip,
    plus the vehicle when they differ.
    """
    source_segment = safe_project_segment(context.source_zip.stem)
    vehicle_segment = safe_project_segment(context.vehicle_id)
    if source_segment.lower() == vehicle_segment.lower():
        return vehicle_segment
    return f"{source_segment}_{vehicle_segment}"


def package_name_for_context(context: VehicleContext) -> str:
    """The output mod's filename.

    Carries the vehicle so converting the next vehicle out of the same zip does
    not overwrite this one in the mods folder.
    """
    return f"{package_stem_for_context(context)}_XP_conversion.zip"


def mod_id_for_context(context: VehicleContext) -> str:
    """The mod's Unique ID, derived from what it is rather than assigned.

    The mod manager reads its manifest from ``mod_info/<id>/info.json`` and
    matches that folder against ``[0-9a-zA-Z]*``
    (``core/modmanager.lua``), so the id is alphanumeric and, like a repo
    tagid, upper case. There is no registry to draw a number from, so it is a
    digest of the conversion's own identity: the same vehicle out of the same
    zip always rebuilds to the same id, two vehicles out of one zip never
    share one, and the XP prefix keeps it clear of a real repo tagid.
    """
    digest = hashlib.blake2b(
        package_stem_for_context(context).lower().encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"XP{digest.upper()}"


def showcase_preview_for_build(
    context: VehicleContext,
    output_vehicle_dir: Path | None,
    generated_configs: Iterable[str],
    config_sources: dict[str, str] | None,
) -> Path | None:
    """The generated preview standing in for the whole build.

    The selector's tile image for this vehicle is already resolved the way the
    engine resolves it (``_tile_preview_member``); when that image is a config's
    own, the build's counterpart of that config is the same car, converted. A
    tile that shows the model image instead -- or a default that this build did
    not convert -- falls back to the first output by name, so the choice stays
    fixed for a given build rather than depending on dict order.
    """
    if output_vehicle_dir is None:
        return None
    outputs = sorted(str(name) for name in generated_configs if name)
    if not outputs:
        return None

    preferred: list[str] = []
    entry = vehicle_catalog_entry_for_id(context.source_zip, context.vehicle_id)
    tile_config = Path(entry.preview_member).stem if entry and entry.preview_member else ""
    if tile_config:
        sources = config_sources or {}
        preferred = [name for name in outputs if sources.get(name, name) == tile_config]

    for output_config in (*preferred, *outputs):
        for suffix in (".jpg", ".png", ".jpeg"):
            candidate = output_vehicle_dir / f"{output_config}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def mod_description_bbcode(
    context: VehicleContext,
    generated_configs: Iterable[str],
    image_path: str = "",
) -> str:
    """The mod's description page, as BBCode.

    ``repository-details.html`` binds the pane to ``modData.message``, which
    ``repository.js`` runs through ``Utils.parseBBCode``. The parser turns
    newlines into breaks and ``[img]`` into an ``<img>`` with that src verbatim
    (``ui-vue/src/services/content/bbcode_main.js``), so the build's own preview
    goes in the page by its in-game path. ``[url]`` is deliberately absent: the
    parser emits a Vue component for it that the Angular page cannot mount.
    """
    outputs = sorted(str(name) for name in generated_configs if name)
    lines: list[str] = []
    if image_path:
        lines.extend([f"[img]{image_path}[/img]", ""])
    lines.extend(
        [
            f"[b]{context.vehicle_id}[/b], hand-drive converted by BeamXP.",
            "",
            "This mod adds converted configurations alongside the vehicle they "
            "came from. Mirrored geometry, relocated interior parts, handed "
            "lighting and license plates are generated from the source vehicle, "
            "so the original stays installed and untouched.",
            "",
            f"[b]Requires:[/b] {context.source_zip.name} -- the configurations "
            "here reference its parts and meshes, and will not load without it.",
            "",
            f"[b]Configurations added ({len(outputs)}):[/b]",
            "[list]",
        ]
    )
    lines.extend(f"[*]{name}" for name in outputs)
    lines.extend(
        [
            "[/list]",
            "[i]Generated automatically. Conversion problems belong to the tool, "
            "not to the author of the source vehicle.[/i]",
        ]
    )
    return "\n".join(lines)


def write_mod_info(
    root: Path,
    context: VehicleContext,
    output_vehicle_dir: Path | None = None,
    generated_configs: Iterable[str] = (),
    config_sources: dict[str, str] | None = None,
) -> None:
    """Write the manifest the mod manager shows for this build.

    A mod the repository has never heard of is served to the UI by
    ``core_repository.requestModOffline``, which hands this file over as
    ``modData`` untouched -- so every field the mod pages bind has to be in it,
    spelled the way a repository entry spells it.

    Three of those spellings are not obvious:

    * the description page (``repository-details.html``) binds ``message``, not
      ``text``; ``text`` feeds only the older fallback panel, so both are set;
    * ``requestModOffline`` fills ``filesize`` from the file on disk, but only
      for a manifest that has a ``message`` -- so the size comes free, and
      guessing at it here would only get in the way;
    * ``last_update`` is multiplied by 1000 before the date filter sees it
      (``repository.js``), so it is epoch seconds. Anything else renders null.

    ``icon`` is set for the mod manager's own panel, which resolves it against
    this folder. The repository page cannot use it -- it rebuilds the icon as a
    beamng.com URL from ``path`` regardless -- which is why the description
    carries the preview itself.
    """
    mod_id = mod_id_for_context(context)
    mod_info = root / "mod_info" / mod_id
    mod_info.mkdir(parents=True, exist_ok=True)
    generated_configs = sorted(str(name) for name in generated_configs if name)
    source_name = conversion_source_name(context)
    title = f"{context.vehicle_id} BeamXP Conversion"
    tag_line = (
        f"Hand-drive conversion of {context.vehicle_id}, generated from "
        f"{context.source_zip.name}."
    )

    preview = showcase_preview_for_build(
        context, output_vehicle_dir, generated_configs, config_sources
    )
    thumbnail = f"preview{preview.suffix.lower()}" if preview is not None else ""
    if preview is not None:
        _atomic_copy_file(preview, mod_info / thumbnail)
    image_path = f"/mod_info/{mod_id}/{thumbnail}" if thumbnail else ""
    description = mod_description_bbcode(context, generated_configs, image_path)

    info: dict[str, object] = {
        "tagid": mod_id,
        "username": MOD_AUTHOR,
        "prefix_title": "BeamXP",
        "title": title,
        "tag_line": tag_line,
        "version_string": MOD_VERSION,
        "current_version_id": MOD_VERSION,
        "last_update": int(datetime.now(UTC).timestamp()),
        "category_title": "Configurations",
        # The panel hides its forum link for thread 1; this mod has no thread.
        "discussion_thread_id": 1,
        "filename": package_name_for_context(context),
        "path": f"{mod_id}/",
        "message": description,
        "text": description,
        # Kept from the pre-manifest layout: harmless to the mod manager, and
        # still the fields a human reads in the file itself.
        "name": title,
        "version": MOD_VERSION,
        "authors": MOD_AUTHOR,
        "description": description,
        "source": source_name,
    }
    if thumbnail:
        info["icon"] = thumbnail
        info["attachments"] = [
            {"filename": thumbnail, "thumb_filename": thumbnail, "title": title}
        ]

    write_text_file(mod_info / "info.json", json.dumps(info, indent=2), encoding="utf-8")


def selected_variant_targets(
    context: VehicleContext,
    conversion: dict[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    targets, skipped, _source_hands = _selected_variant_targets_and_source_hands(context, conversion)
    return targets, skipped


def _selected_variant_targets_and_source_hands(
    context: VehicleContext,
    conversion: dict[str, object],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    targets: dict[str, str] = {}
    skipped: dict[str, str] = {}
    source_hands: dict[str, str] = {}
    variants = conversion.get("variants", {})
    if not isinstance(variants, dict):
        return targets, skipped, source_hands
    for config_name, settings in variants.items():
        if config_name not in context.variants or not isinstance(settings, dict):
            continue
        if variant_build_mode(settings) not in {BUILD_CONVERTED, BUILD_BOTH}:
            continue
        source_hand = effective_source_hand(context, conversion, config_name)
        source_hands[config_name] = source_hand
        target = target_hand_for(source_hand, ACTION_OPPOSITE)
        if target is None:
            skipped[config_name] = f"No opposite target for source hand {source_hand}"
        else:
            targets[config_name] = target
    return targets, skipped, source_hands


def selected_output_plans(
    context: VehicleContext,
    conversion: dict[str, object],
    *,
    variant_targets: dict[str, str] | None = None,
    target_skipped: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Expand each trim row into zero, one, or two generated configs."""
    targets = variant_targets
    skipped = dict(target_skipped or {})
    if targets is None:
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


def split_authored_hand_drive_targets(
    context: VehicleContext,
    conversion: dict[str, object],
    variant_targets: dict[str, str],
    source_hands: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Resolve authored LHD/RHD swaps, keeping every variant in the build.

    An authored swap covers its own subtree only. The rest of the trim -- seats,
    pedals, mirrors, anything hung off the body rather than the dashboard --
    still needs the generated mirroring pass, so the two run together on the
    same variant instead of one displacing the other.
    """
    generated: dict[str, str] = dict(variant_targets)
    authored: dict[str, dict[str, object]] = {}
    source_hands = source_hands or {}
    for config_name, target_hand in sorted(variant_targets.items()):
        source_hand = source_hands.get(config_name)
        if source_hand is None:
            source_hand = effective_source_hand(context, conversion, config_name)
        group = find_hand_authored_opposite_group(
            context,
            config_name,
            source_hand,
            target_hand,
        )
        if group is not None:
            authored[config_name] = group
    return generated, authored


def generated_mesh_scope(
    context: VehicleContext,
    selected_configs: Iterable[str],
    authored_groups: dict[str, dict[str, object]],
    slot_pair_plans: dict[str, dict[str, object]] | None = None,
) -> set[str]:
    """Meshes the generated pass may transform.

    Per trim, everything the trim uses minus the parts an authored swap or a
    slot pair already resolves -- those land as stock opposite-side parts, and
    mirroring their meshes on top would undo the swap.
    """
    slot_pair_plans = slot_pair_plans or {}
    scope: set[str] = set()
    for config_name in selected_configs:
        covered = authored_group_meshes(context, authored_groups.get(config_name))
        covered |= authored_group_meshes(context, slot_pair_plans.get(config_name))
        scope.update(used_meshes_for_config(context, config_name) - covered)
    return scope


def relocation_meshes(
    context: VehicleContext,
    slot_pair_plans: dict[str, dict[str, object]],
) -> set[str]:
    """Meshes belonging to parts that must be rebuilt on the other side.

    A relocated part has no stock counterpart to inherit geometry from, so its
    meshes always need a mirrored bake regardless of the mode the user left on
    them -- the part is crossing the car either way.
    """
    meshes: set[str] = set()
    for plan in slot_pair_plans.values():
        for relocation in slot_pair_plan_relocations(plan):
            found = part_body_for_context(context, str(relocation.get("partId") or ""))
            if found is not None:
                meshes.update(part_mesh_names_for_context(context, str(relocation.get("partId") or "")))
    return {mesh for mesh in meshes if mesh in context.objects}


def _atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _atomic_move_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    source.replace(temporary)
    temporary.replace(target)


def _same_copied_file(source: Path, target: Path) -> bool:
    try:
        source_stat = source.stat()
        target_stat = target.stat()
    except OSError:
        return False
    return (
        source_stat.st_size == target_stat.st_size
        and source_stat.st_mtime_ns == target_stat.st_mtime_ns
    )


def _copy_file_if_changed(source: Path, target: Path) -> bool:
    if _same_copied_file(source, target):
        return False
    _atomic_copy_file(source, target)
    return True


def _texture_correction_output_stem(source_zip: Path, dae_path: str) -> str:
    normalised_dae_path = dae_path.replace("\\", "/")
    stem = f"{source_zip.stem}_{Path(normalised_dae_path).stem}"
    return safe_project_segment(stem) or "texture_correction"


def _matching_loaded_dae_parts(loaded: object, mesh_ids: Iterable[str]) -> tuple[list[object], list[str]]:
    wanted = {str(mesh_id) for mesh_id in mesh_ids}
    matched: list[object] = []
    matched_ids: set[str] = set()
    for part in getattr(loaded, "parts", ()) or ():
        aliases = {
            str(getattr(part, "key", "") or ""),
            str(getattr(part, "node_id", "") or ""),
            str(getattr(part, "node_name", "") or ""),
        }
        aliases.discard("")
        hits = aliases & wanted
        if not hits:
            continue
        matched.append(part)
        matched_ids.update(hits)
    return matched, sorted(wanted - matched_ids)


def texture_correction_asset_archives(context: VehicleContext) -> list[Path]:
    """Ordered BeamNG virtual-asset search path for texture correction.

    A marked mesh's DAE can bind materials/textures shipped by the vehicle, a
    sibling stock archive, or any stock game vehicle archive. The source archive
    stays first so mods override stock assets when paths collide.
    """
    paths: list[Path] = []
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
            paths.append(candidate)

    generated_package = package_name_for_context(context).lower()

    add(context.source_zip)
    if context.source_zip.parent.is_dir():
        for candidate in sorted(context.source_zip.parent.glob("*.zip"), key=lambda item: item.name.lower()):
            # Generated conversion assets must not become inputs to the next
            # build, or corrected files recursively turn into *_rhd_rhd jobs.
            if candidate.name.lower() == generated_package:
                continue
            add(candidate)
    for common_zip in beamng_game_common_zips():
        parent = common_zip.parent
        if parent.is_dir():
            for candidate in sorted(parent.glob("*.zip"), key=lambda item: item.name.lower()):
                add(candidate)
        else:
            add(common_zip)
    return paths


def export_texture_correction_artifacts(
    context: VehicleContext,
    artifact_root: Path,
    mesh_ids: Iterable[str],
    progress: Callable[[str], None] | None = None,
    shared_atlas_dependency_targets: dict[str, set[str]] | None = None,
    force_mirrored_dependency_ids: set[str] | None = None,
    texture_member_scope_by_source: dict[tuple[Path, str], set[str]] | None = None,
    bc7_quality: str | None = None,
) -> dict[str, object]:
    """Run the standalone atlas-correction exporter for marked meshes.

    ``artifact_root`` is a working directory, not part of the shipped mod. The
    exporter emits a full audit trail per job -- the corrected DDS, the PNG it
    was encoded from, debug and preview renders, the plan JSON, a per-mesh DAE
    and a Blender script -- and only the DDS is a game asset. Those are copied
    into the vehicle folder by _prepare_texture_correction_materials, so the
    rest has no business in the package: on the Ardente it was 1.01 GB of a
    1.20 GB zip, 842 MB of that in PNGs nothing loads.
    """
    selected = sorted({str(mesh_id) for mesh_id in mesh_ids if str(mesh_id) in context.objects})
    report: dict[str, object] = {
        "enabled": bool(selected),
        "requestedMeshes": selected,
        "jobs": [],
        "missing": [],
        "failures": [],
    }
    if not selected:
        return report
    if progress is not None:
        progress(f"Preparing texture correction for {len(selected)} mesh(es)...")

    try:
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            DEFAULT_CONFIG as DEFAULT_MSER_CONFIG,
            DEFAULT_RELIEF_DETECTION_CONFIG,
            DEFAULT_RHD_CONFIG,
            export_parts_preview,
            texture_bindings_for_parts,
        )
        from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
            extract_archive_member,
            load_dae,
            scan_vehicle_archive,
        )
    except Exception as exc:  # pragma: no cover - depends on optional packages in packaged builds
        raise RuntimeError(
            "Texture correction was requested, but the standalone texture tooling "
            f"could not be loaded: {type(exc).__name__}: {exc}"
        ) from exc
    texture_config = replace(
        DEFAULT_RHD_CONFIG,
        detect_on_normal_map=True,
        write_debug_overlays=False,
        # The encoder tier is the user's speed/quality choice for this build.
        bc7_profile=bc7_quality or DEFAULT_RHD_CONFIG.bc7_profile,
        # Nothing in a build reads the inspection PNG: materials are wired from
        # the DDS, and only a Blender preview ever opens the other copy.
        # Scintilla wrote 3.39 GB of them, against 0.74 GB of shipped DDS.
        write_preview_png=False,
    )

    by_source: dict[tuple[Path, str], list[str]] = {}
    for mesh_id in selected:
        obj = context.objects.get(mesh_id)
        if obj is None or not obj.dae_path:
            continue
        source_zip = obj.dae_source_zip or context.source_zip
        by_source.setdefault((source_zip, obj.dae_path), []).append(mesh_id)
    dependency_targets = shared_atlas_dependency_targets or {}
    forced_dependency_ids = force_mirrored_dependency_ids or set()
    scoped_members_by_source = texture_member_scope_by_source or {}
    dependencies_by_source: dict[tuple[Path, str], list[str]] = {}
    for mesh_id in dependency_targets:
        if mesh_id in selected:
            continue
        obj = context.objects.get(mesh_id)
        if obj is None or not obj.dae_path:
            continue
        source_zip = obj.dae_source_zip or context.source_zip
        dependencies_by_source.setdefault((source_zip, obj.dae_path), []).append(mesh_id)
    auto_included_targets: dict[str, set[str]] = {}
    deferred_forced_meshes: set[str] = set()
    deferred_forced_scope: dict[tuple[Path, str], set[str]] = {}

    output_dir = artifact_root
    workspace_root = context.project_dir / "build" / "texture_correction_workspace"
    clean_dir(workspace_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_cache: dict[Path, object] = {}

    for (source_zip, dae_member), source_meshes in sorted(by_source.items(), key=lambda item: (str(item[0][0]).lower(), item[0][1].lower())):
        try:
            archive = archive_cache.get(source_zip)
            if archive is None:
                archive_workspace = workspace_root / safe_project_segment(source_zip.stem)
                asset_archives = [
                    candidate
                    for candidate in texture_correction_asset_archives(context)
                    if candidate != source_zip
                ]
                archive = scan_vehicle_archive(
                    source_zip,
                    archive_workspace,
                    asset_archives=asset_archives,
                )
                archive_cache[source_zip] = archive
            loaded = load_dae(extract_archive_member(archive, dae_member))
        except Exception as exc:
            report["failures"].append(
                {
                    "sourceZip": str(source_zip),
                    "dae": dae_member,
                    "meshes": sorted(source_meshes),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if progress is not None:
                progress(f"Texture correction failed loading {Path(dae_member).name}")
            continue

        # Keep meshes from one DAE in one export so shared atlases are corrected
        # once with the full selected UV scope.
        wanted_meshes = sorted(source_meshes)
        job_dir = output_dir / _texture_correction_output_stem(source_zip, dae_member)
        clean_dir(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        log_lines: list[str] = []

        def log(line: object = "") -> None:
            log_lines.append(str(line))

        parts, missing = _matching_loaded_dae_parts(loaded, wanted_meshes)
        if missing:
            report["missing"].append(
                {
                    "sourceZip": str(source_zip),
                    "dae": dae_member,
                    "meshes": missing,
                }
            )
        if not parts:
            continue
        texture_parts = list(parts)
        texture_part_keys = {part.key for part in texture_parts}
        forced_texture_part_keys: set[str] = set()
        selected_texture_members = set(
            texture_bindings_for_parts(archive, loaded, parts)
        )
        requested_member_scope = scoped_members_by_source.get((source_zip, dae_member))
        if requested_member_scope is not None:
            selected_texture_members.intersection_update(requested_member_scope)
        for dependency_mesh in sorted(
            dependencies_by_source.get((source_zip, dae_member), ())
        ):
            dependency_parts, _dependency_missing = _matching_loaded_dae_parts(
                loaded, (dependency_mesh,)
            )
            if not dependency_parts:
                continue
            dependency_members = set(
                texture_bindings_for_parts(archive, loaded, dependency_parts)
            )
            shared_members = selected_texture_members.intersection(dependency_members)
            if not shared_members:
                continue
            if dependency_mesh in forced_dependency_ids:
                for dependency_part in dependency_parts:
                    if dependency_part.key not in texture_part_keys:
                        texture_parts.append(dependency_part)
                        texture_part_keys.add(dependency_part.key)
                    forced_texture_part_keys.add(dependency_part.key)
                deferred_forced_meshes.add(dependency_mesh)
                deferred_forced_scope.setdefault((source_zip, dae_member), set()).update(
                    shared_members
                )
                auto_included_targets.setdefault(dependency_mesh, set()).update(
                    dependency_targets.get(dependency_mesh, ())
                )
                continue
            parts.extend(dependency_parts)
            wanted_meshes.append(dependency_mesh)
            auto_included_targets.setdefault(dependency_mesh, set()).update(
                dependency_targets.get(dependency_mesh, ())
            )
        wanted_meshes = sorted(set(wanted_meshes))
        forced_part_keys = (
            set(wanted_meshes).intersection(forced_dependency_ids)
            | forced_texture_part_keys
        )
        matched_meshes = [mesh for mesh in wanted_meshes if mesh not in set(missing)]
        if progress is not None:
            progress(
                "Texture correction: "
                f"{len(parts)} mesh(es) from {Path(dae_member).name}..."
            )

        try:
            preview = export_parts_preview(
                archive,
                loaded,
                parts,
                job_dir,
                texture_config,
                DEFAULT_MSER_CONFIG,
                bake=False,
                relief_mser_config=DEFAULT_RELIEF_DETECTION_CONFIG,
                log=log,
                texture_member_scope=selected_texture_members,
                force_mirrored_part_keys=forced_part_keys,
                texture_part_scope=texture_parts,
            )
            (job_dir / "texture_correction.log").write_text(
                "\n".join(log_lines) + ("\n" if log_lines else ""),
                encoding="utf-8",
            )
            # A part with no correctable atlas is skipped by the exporter
            # rather than failing its whole DAE, so the meshes it stands for
            # are reported as failures on their own.
            skipped_aliases: set[str] = set()
            for entry in getattr(preview, "failed_parts", ()) or ():
                source_part = entry.get("source_part") if isinstance(entry, dict) else None
                if not isinstance(source_part, dict):
                    continue
                aliases = {
                    str(source_part.get(key) or "")
                    for key in ("key", "node_id", "node_name")
                } - {""}
                skipped_aliases.update(aliases)
                report["failures"].append(
                    {
                        "sourceZip": str(source_zip),
                        "dae": dae_member,
                        "meshes": sorted(mesh for mesh in matched_meshes if mesh in aliases),
                        "error": str(entry.get("error") or "unknown error"),
                    }
                )
            corrected_meshes = [
                mesh for mesh in matched_meshes if mesh not in skipped_aliases
            ]
            report["jobs"].append(
                {
                    "sourceZip": str(source_zip),
                    "dae": dae_member,
                    "meshes": corrected_meshes,
                    "selectedParts": [
                        {
                            "key": str(getattr(part, "key", "") or ""),
                            "nodeId": str(getattr(part, "node_id", "") or ""),
                            "nodeName": str(getattr(part, "node_name", "") or ""),
                        }
                        for part in parts
                    ],
                    "textureScopeParts": [
                        {
                            "key": str(getattr(part, "key", "") or ""),
                            "nodeId": str(getattr(part, "node_id", "") or ""),
                            "nodeName": str(getattr(part, "node_name", "") or ""),
                        }
                        for part in texture_parts
                    ],
                    "outputDirectory": str(job_dir),
                    "reportPath": str(preview.report_path) if preview.report_path is not None else None,
                    "daePaths": [str(path) for path in preview.dae_paths],
                    "textureCount": len(preview.textures),
                    "forceMirroredMeshes": sorted(forced_part_keys),
                    "seconds": round(float(preview.seconds), 6),
                }
            )
            if progress is not None:
                skipped = len(getattr(preview, "failed_parts", ()) or ())
                progress(
                    f"Texture correction finished {len(corrected_meshes)} mesh(es) "
                    f"in {float(preview.seconds):.1f}s"
                    + (f", skipped {skipped}" if skipped else "")
                )
        except Exception as exc:
            if log_lines:
                (job_dir / "texture_correction.log").write_text(
                    "\n".join(log_lines) + "\n",
                    encoding="utf-8",
                )
            report["failures"].append(
                {
                    "sourceZip": str(source_zip),
                    "dae": dae_member,
                    "meshes": matched_meshes,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if progress is not None:
                progress(
                    f"Texture correction failed for {Path(dae_member).name}"
                )

    if report["failures"] and not report["jobs"]:
        first = report["failures"][0]
        raise RuntimeError(
            "Texture correction failed for marked mesh(es): "
            f"{', '.join(first.get('meshes', []))}: {first.get('error')}"
        )
    if deferred_forced_meshes:
        forced_report = export_texture_correction_artifacts(
            context,
            artifact_root / "structural_mirror",
            deferred_forced_meshes,
            progress=progress,
            force_mirrored_dependency_ids=deferred_forced_meshes,
            texture_member_scope_by_source=deferred_forced_scope,
            bc7_quality=bc7_quality,
        )
        for key in ("jobs", "missing", "failures"):
            values = forced_report.get(key, [])
            if isinstance(values, list):
                report[key].extend(values)
    report["autoIncludedTargets"] = {
        source: sorted(targets)
        for source, targets in sorted(auto_included_targets.items())
    }
    summary_path = output_dir / "texture_correction.report.json"
    report["reportPath"] = str(summary_path)
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
ET.register_namespace("", COLLADA_NS)


def _collada_q(tag: str) -> str:
    return f"{{{COLLADA_NS}}}{tag}"


def _identity_matrix4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _parse_collada_matrix(node: ET.Element) -> list[list[float]]:
    matrix = node.find("c:matrix", NS)
    if matrix is None or not matrix.text:
        return _identity_matrix4()
    try:
        values = [float(value) for value in matrix.text.split()]
    except ValueError:
        return _identity_matrix4()
    if len(values) != 16:
        return _identity_matrix4()
    return [values[0:4], values[4:8], values[8:12], values[12:16]]


def _matrix_multiply4(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][col] for index in range(4)) for col in range(4)]
        for row in range(4)
    ]


def _collada_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    parents: dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parents[child] = parent
    return parents


def _collada_world_matrix(
    node: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> list[list[float]]:
    chain: list[ET.Element] = []
    current: ET.Element | None = node
    while current is not None:
        if current.tag == _collada_q("node"):
            chain.append(current)
        current = parents.get(current)
    matrix = _identity_matrix4()
    for item in reversed(chain):
        matrix = _matrix_multiply4(matrix, _parse_collada_matrix(item))
    return matrix


def _transform_point4(matrix: list[list[float]], point: list[float]) -> list[float]:
    x, y, z = point
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def _transform_direction4(matrix: list[list[float]], direction: list[float]) -> list[float]:
    x, y, z = direction
    transformed = [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    ]
    length = math.sqrt(sum(value * value for value in transformed))
    if length > 0.0:
        return [value / length for value in transformed]
    return transformed


def _format_collada_floats(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _bake_geometry_transform(geometry: ET.Element, matrix: list[list[float]]) -> None:
    for source in geometry.findall(".//c:source", NS):
        source_id = source.get("id", "").lower()
        if not (source_id.endswith("-positions") or source_id.endswith("-normals")):
            continue
        array = source.find("c:float_array", NS)
        if array is None or not array.text:
            continue
        try:
            values = [float(value) for value in array.text.split()]
        except ValueError:
            continue
        baked: list[float] = []
        transform = _transform_direction4 if source_id.endswith("-normals") else _transform_point4
        for index in range(0, len(values) - 2, 3):
            baked.extend(transform(matrix, values[index : index + 3]))
        array.text = _format_collada_floats(baked)


def _set_identity_node_transform(node: ET.Element) -> None:
    transform_tags = {
        _collada_q("matrix"),
        _collada_q("translate"),
        _collada_q("rotate"),
        _collada_q("scale"),
    }
    for child in list(node):
        if child.tag in transform_tags:
            node.remove(child)
    matrix = ET.Element(_collada_q("matrix"))
    matrix.text = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    node.insert(0, matrix)


def _normalise_material_alias(value: str) -> str:
    value = value.strip().lstrip("#").lower()
    for suffix in ("-material", "_material"):
        value = value.removesuffix(suffix)
    return value


def _material_source_for_beamng(job_dir: Path, relative_path: str) -> Path:
    source = job_dir / relative_path
    name = source.name
    candidates: list[Path] = []
    if name.lower().endswith(".preview.png"):
        candidates.append(source.with_name(name[: -len(".preview.png")] + ".dds"))
        candidates.append(source.with_name(name[: -len(".preview.png")] + ".png"))
    if source.suffix.lower() == ".png":
        candidates.append(source.with_suffix(".dds"))
    candidates.append(source)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".dds" and not _beamng_dds_size_supported(candidate):
            png = candidate.with_suffix(".png")
            if png.is_file():
                return png
            continue
        return candidate
    return source


def _beamng_dds_size_supported(path: Path) -> bool:
    """Reject compressed NPOT DDS files that BeamNG's texture loader rejects."""
    try:
        header = path.read_bytes()[:20]
    except OSError:
        return True
    if len(header) < 20 or header[:4] != b"DDS ":
        return True
    height = int.from_bytes(header[12:16], "little")
    width = int.from_bytes(header[16:20], "little")
    return (
        width > 0
        and height > 0
        and width & (width - 1) == 0
        and height & (height - 1) == 0
    )


def _material_skin_suffix_start(name: str) -> int:
    """Where BeamNG's skin-variant suffix begins in a material name, or -1.

    A skin is authored as ``<material>.<slotType>.<skinName>`` --
    ``scintilla_interior.skin_interior.luxe`` beside ``scintilla_interior`` --
    and a skin part is nothing but that pair of names:

        "scintilla_skin_interior_luxe": {
            "slotType" : "skin_interior",
            "skinName" : "luxe"
        }

    The engine rebinds by composing them onto whatever material the mesh
    actually binds, so the suffix has to survive on the end of any name we give
    that material or the rebinding has nothing to find.
    """
    for index in range(len(name)):
        if not name.startswith(".skin", index):
            continue
        after = index + len(".skin")
        if after == len(name) or name[after] in "._":
            return index
    return -1


def _split_material_skin_suffix(name: str) -> tuple[str, str]:
    """A material name as (base, skin suffix); the suffix is "" when plain."""
    index = _material_skin_suffix_start(name)
    if index <= 0:
        return name, ""
    return name[:index], name[index:]


def _texture_correction_material_name(
    alias: str,
    used: set[str],
    base_names: dict[str, str] | None = None,
) -> str:
    """The corrected material's name, with any skin suffix left on the end.

    A skin has to be named for the material it skins, so the correction suffix
    goes on the base and the skin suffix stays last:
    ``scintilla_interior_beamxp_tc.skin_interior.luxe``. Suffixing the whole
    alias instead gave ``scintilla_interior.skin_interior.luxe_beamxp_tc`` --
    a material no config could ever ask for, since the mesh binds
    ``scintilla_interior_beamxp_tc`` and the engine looks for that name plus
    the skin's slot and name. The prune pass then removed it as unreachable,
    quite correctly, and every mirrored mesh fell back to the base textures:
    ten of the scintilla's sixteen trims wore the wrong interior.

    ``base_names`` carries the corrected name already chosen for each base
    alias, so a skin lands on its own base rather than starting a fresh one.
    """
    base_alias, skin_suffix = _split_material_skin_suffix(_normalise_material_alias(alias))
    skin_suffix = safe_id(skin_suffix)
    if skin_suffix and base_names is not None:
        base_name = base_names.get(base_alias)
        if base_name is not None:
            candidate = f"{base_name}{skin_suffix}"
            used.add(candidate.lower())
            return candidate
    base = safe_id(f"{base_alias}_beamxp_tc") or "beamxp_texture_corrected"
    candidate = base
    counter = 2
    while f"{candidate}{skin_suffix}".lower() in used:
        candidate = f"{base}_{counter}"
        counter += 1
    if base_names is not None and not skin_suffix:
        # The newest base wins: where one alias corrects to two materials -- two
        # layouts sharing a name, which is what the _2 suffix is for -- the
        # skins named after it belong to the one just allocated.
        base_names[base_alias] = candidate
    candidate = f"{candidate}{skin_suffix}"
    used.add(candidate.lower())
    return candidate


def _vehicle_virtual_path(output_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(output_root)
    except ValueError:
        rel = path
    return "/" + rel.as_posix()


def _texture_reference_keys(value: str) -> tuple[str, ...]:
    value = value.replace("\\", "/").strip()
    if not value or value.startswith("@"):
        return ()
    path = value.lstrip("/")
    name = PurePosixPath(path).name.lower()
    stem = PurePosixPath(name).with_suffix("").as_posix().lower()
    keys = [path.lower(), name]
    if stem:
        keys.append(stem)
    return tuple(dict.fromkeys(keys))


def _register_texture_output(
    mapping: dict[str, str],
    source_reference: str,
    virtual_path: str,
) -> None:
    for key in _texture_reference_keys(source_reference):
        mapping.setdefault(key, virtual_path)


def _entry_corrected_texture_outputs(
    job_dir: Path,
    target_dir: Path,
    output_root: Path,
    material_name: str,
    entry: dict[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, str] = {}
    by_stage: dict[str, str] = {}
    output_maps = entry.get("outputMaps")
    if isinstance(output_maps, list):
        for item in output_maps:
            if not isinstance(item, dict):
                continue
            member = item.get("member")
            stage_key = item.get("stageKey")
            relative = item.get("dds") or item.get("png") or item.get("preview")
            if not isinstance(member, str) or not isinstance(relative, str):
                continue
            source = _material_source_for_beamng(job_dir, relative)
            if not source.is_file():
                continue
            destination = target_dir / f"{safe_id(material_name)}_{source.name}"
            shutil.copy2(source, destination)
            virtual_path = _vehicle_virtual_path(output_root, destination)
            _register_texture_output(by_source, member, virtual_path)
            if isinstance(stage_key, str):
                by_stage.setdefault(stage_key, virtual_path)

    maps = entry.get("maps", {})
    if isinstance(maps, dict):
        for stage_key, relative in maps.items():
            if not isinstance(stage_key, str) or not isinstance(relative, str):
                continue
            source = _material_source_for_beamng(job_dir, relative)
            if not source.is_file():
                continue
            destination = target_dir / f"{safe_id(material_name)}_{source.name}"
            shutil.copy2(source, destination)
            virtual_path = _vehicle_virtual_path(output_root, destination)
            by_stage.setdefault(stage_key, virtual_path)
            _register_texture_output(by_source, relative, virtual_path)
    return by_source, by_stage


def _stage_texture_reference(stage: dict[str, object]) -> str:
    for key in ("baseColorMap", "colorMap", "diffuseMap"):
        value = stage.get(key)
        if isinstance(value, str) and value.strip() and not value.lstrip().startswith("@"):
            return value
    return ""


def _source_material_score(
    material: dict[str, object],
    base_member: str,
) -> int:
    stages = material.get("Stages")
    if not isinstance(stages, list):
        return 0
    wanted = set(_texture_reference_keys(base_member))
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if any(key in wanted for key in _texture_reference_keys(_stage_texture_reference(stage))):
            return 2
    return 1


def _select_source_material(entry: dict[str, object]) -> dict[str, object] | None:
    materials = entry.get("sourceMaterials")
    if not isinstance(materials, list):
        return None
    output_maps = entry.get("outputMaps")
    base_member = ""
    if isinstance(output_maps, list):
        for item in output_maps:
            if isinstance(item, dict) and item.get("stageKey") == "baseColorMap":
                member = item.get("member")
                if isinstance(member, str):
                    base_member = member
                    break
    best: tuple[int, dict[str, object]] | None = None
    for item in materials:
        if not isinstance(item, dict):
            continue
        material = item.get("material")
        if not isinstance(material, dict):
            continue
        score = _source_material_score(material, base_member) if base_member else 1
        if best is None or score > best[0]:
            best = (score, material)
    return copy.deepcopy(best[1]) if best is not None else None


def _duplicate_safe_material_alias(value: str) -> str:
    alias = _normalise_material_alias(value)
    duplicate = re.match(r"^(?P<base>.+?)(?:[._]\d{3})$", alias)
    return duplicate.group("base") if duplicate is not None else alias


def _switch_base_aliases_for_source_aliases(
    source_aliases: Iterable[str],
    all_source_exact_keys: set[str],
) -> set[str]:
    bases: set[str] = set()
    for alias in source_aliases:
        key = _duplicate_safe_material_alias(alias)
        suffix = "_off"
        if key.endswith(suffix) and len(key) > len(suffix):
            bases.add(key[: -len(suffix)])
    return bases


def _entry_material_variants(
    entry: dict[str, object],
    aliases: list[str],
) -> list[tuple[list[str], dict[str, object] | None]]:
    source_materials = entry.get("sourceMaterials")
    if not isinstance(source_materials, list):
        return [(aliases, _select_source_material(entry))]

    variants: list[tuple[list[str], dict[str, object] | None]] = []
    seen_sources: set[tuple[str, ...]] = set()
    all_source_exact_keys = {
        _duplicate_safe_material_alias(alias)
        for item in source_materials
        if isinstance(item, dict)
        for alias in (
            item.get("aliases", [])
            if isinstance(item.get("aliases"), list)
            else [item.get("key")]
        )
        if isinstance(alias, str) and _duplicate_safe_material_alias(alias)
    }
    for item in source_materials:
        if not isinstance(item, dict):
            continue
        material = item.get("material")
        if not isinstance(material, dict):
            continue
        report_aliases = [
            str(alias)
            for alias in item.get("aliases", [])
            if isinstance(alias, str) and alias.strip()
        ]
        if not report_aliases:
            key = item.get("key")
            if isinstance(key, str) and key.strip():
                report_aliases = [key]
        source_exact_keys = {
            _duplicate_safe_material_alias(alias)
            for alias in report_aliases
            if _duplicate_safe_material_alias(alias)
        }
        switch_base_aliases = _switch_base_aliases_for_source_aliases(
            report_aliases,
            all_source_exact_keys,
        )
        if not source_exact_keys:
            continue
        source_signature = tuple(sorted(source_exact_keys))
        if source_signature in seen_sources:
            continue
        seen_sources.add(source_signature)

        variant_aliases: list[str] = []
        for alias in (*report_aliases, *aliases):
            alias_key = _duplicate_safe_material_alias(alias)
            if (
                alias_key in source_exact_keys
                or alias_key in switch_base_aliases
            ) and alias not in variant_aliases:
                variant_aliases.append(alias)
        variants.append((variant_aliases or report_aliases, copy.deepcopy(material)))

    return variants or [(aliases, _select_source_material(entry))]


def _retarget_material_document(
    material: dict[str, object],
    material_name: str,
    by_source: dict[str, str],
    _by_stage: dict[str, str],
) -> dict[str, object]:
    material["name"] = material_name
    material["mapTo"] = material_name
    stages = material.get("Stages")
    if not isinstance(stages, list):
        stages = [{}]
        material["Stages"] = stages
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for key, value in list(stage.items()):
            if not isinstance(value, str):
                continue
            replacement = next(
                (by_source[source_key] for source_key in _texture_reference_keys(value) if source_key in by_source),
                None,
            )
            if replacement is not None:
                stage[key] = replacement
    return material


def _fallback_texture_correction_material(
    material_name: str,
    copied_maps: dict[str, str],
) -> dict[str, object]:
    stage: dict[str, object] = {}
    for key in (
        "baseColorMap",
        "normalMap",
        "roughnessMap",
        "metallicMap",
        "ambientOcclusionMap",
        "opacityMap",
    ):
        if key in copied_maps:
            stage[key] = copied_maps[key]
    if "metallicMap" in stage:
        stage.setdefault("metallicFactor", 1)
    return {
        "name": material_name,
        "mapTo": material_name,
        "class": "Material",
        "Stages": [stage],
        "doubleSided": True,
        "dynamicCubemap": True,
        "materialTag0": "beamng",
        "materialTag1": "vehicle",
        "translucentBlendOp": "None",
        "version": 1.5,
    }


_DAE_MATERIAL_SYMBOL_RE = re.compile(r'<instance_material[^>]*\bsymbol="([^"]+)"')
_DAE_MATERIAL_ID_RE = re.compile(r'<material[^>]*\bid="([^"]+)"')


def _glowmap_material_references(text: str) -> set[str]:
    references: set[str] = set()

    def collect(glow_text: str) -> tuple[str, int]:
        for key, _start, _end, value_text in _top_level_jbeam_object_entries(glow_text):
            if key:
                references.add(key)
            for match in _GLOW_MATERIAL_STATE_RE.finditer(value_text):
                material = _decode_jbeam_string(f'"{match.group("material")}"')
                if material:
                    references.add(material.lstrip("@"))
        return glow_text, 0

    _replace_all_jbeam_object_regions(text, "glowMap", collect)
    return {_normalise_material_alias(reference) for reference in references}


def _materials_bound_by_generated_meshes(output_vehicle_dir: Path) -> set[str]:
    """Every material name the generated assets can reach."""
    bound: set[str] = set()
    for path in output_vehicle_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".dae":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            bound.update(_DAE_MATERIAL_SYMBOL_RE.findall(text))
            for value in _DAE_MATERIAL_ID_RE.findall(text):
                bound.add(value)
                if value.endswith("-material"):
                    bound.add(value[: -len("-material")])
        elif path.suffix.lower() == ".jbeam":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            bound.update(_glowmap_material_references(text))
    return bound


def prune_unused_texture_correction_assets(
    output_root: Path,
    output_vehicle_dir: Path,
) -> dict[str, object]:
    """Drop corrected materials and textures no generated mesh ended up binding.

    The exporter corrects every material alias it finds in a source DAE, but
    only the aliases the generated meshes turn out to use get bound into them,
    and which those are is not known until integration has run. Whatever is
    left over is a full texture set nothing can ever sample: on the Ardente the
    racing interior's alcantar and alumin variants are corrected and never
    bound, which is 69.9 MB of DDS in a 189.6 MB package.

    A skin variant is reachable without ever being bound: no mesh names
    ``..._beamxp_tc.skin_interior.luxe``, the engine composes it at runtime from
    the material the mesh does bind and the skin part the config selected. So a
    skin is kept exactly when the base it skins is kept, and dropped with it.

    Runs on the finished output rather than predicting up front, so it cannot
    be wrong about what is in use, and only ever removes files inside the
    generated vehicle folder.
    """
    bound = _materials_bound_by_generated_meshes(output_vehicle_dir)

    def reachable(name: str) -> bool:
        return name in bound or _split_material_skin_suffix(name)[0] in bound
    removed_materials: list[str] = []
    removed_files: list[str] = []
    freed = 0
    for material_file in sorted(output_vehicle_dir.rglob("beamxp_texture_correction.materials.json")):
        try:
            document = json.loads(material_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        keep = {name: body for name, body in document.items() if reachable(name)}
        drop = {name: body for name, body in document.items() if not reachable(name)}
        if not drop:
            continue

        def texture_paths(entry: object) -> set[str]:
            return set(re.findall(r'"([^"]*\.(?:dds|png|jpg|jpeg))"', json.dumps(entry)))

        still_used = set().union(*(texture_paths(b) for b in keep.values())) if keep else set()
        for name, body in drop.items():
            removed_materials.append(name)
            for reference in texture_paths(body) - still_used:
                candidate = output_root / reference.lstrip("/")
                # never reach outside the folder this build generated
                if not candidate.is_file() or output_vehicle_dir not in candidate.parents:
                    continue
                freed += candidate.stat().st_size
                removed_files.append(reference)
                candidate.unlink()
        if keep:
            write_text_file(material_file, json.dumps(keep, indent=2), encoding="utf-8")
        else:
            material_file.unlink()
    return {
        "removedMaterials": sorted(removed_materials),
        "removedTextures": sorted(removed_files),
        "bytesFreed": freed,
    }


class TextureCorrectionMaterials(dict[str, str]):
    """Alias -> corrected material, with the per-mesh overrides one atlas needs.

    Reads as the flat union it has always been, so reporting and the glowMap
    pass see every corrected material.  ``for_part`` narrows it to one source
    mesh, which matters only where a texture was corrected more than once: the
    LC500's two interiors both fill ``lc500_interior`` with ``coreSlot:true``
    and both paint ``lc500_screens``, but they disagree about every texel of
    screen.dds, so each needs its own answer under the same alias.  Retargeting
    both meshes off the flat map hands the second whichever correction was
    minted first, which is the wrong atlas rather than merely an uncorrected
    one.

    A mesh no scoped entry names keeps only the unscoped aliases -- it was not
    part of that texture's correction, so it stays on the shipped texture.
    """

    def __init__(
        self,
        shared: Mapping[str, str] | None = None,
        by_part: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.shared: dict[str, str] = dict(shared or {})
        self.by_part: dict[str, dict[str, str]] = {
            str(key): dict(value) for key, value in (by_part or {}).items()
        }
        super().__init__(self.shared)
        for scoped in self.by_part.values():
            for alias, material in scoped.items():
                self.setdefault(alias, material)

    def for_part(self, part_key: str) -> dict[str, str]:
        if not self.by_part:
            return dict(self)
        return {**self.shared, **self.by_part.get(part_key, {})}


def _prepare_texture_correction_materials(
    job_dir: Path,
    target_dir: Path,
    output_root: Path,
) -> TextureCorrectionMaterials:
    manifest = job_dir / "rhd_materials.json"
    if not manifest.is_file():
        return TextureCorrectionMaterials()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    entries = document.get("materials", []) if isinstance(document, dict) else []
    material_file = target_dir / "beamxp_texture_correction.materials.json"
    existing: dict[str, object] = {}
    if material_file.is_file():
        try:
            loaded = json.loads(material_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    shared: dict[str, str] = {}
    by_part: dict[str, dict[str, str]] = {}
    used_names = {str(key).lower() for key in existing}
    base_names: dict[str, str] = {}
    for key in existing:
        base_alias, skin_suffix = _split_material_skin_suffix(str(key).lower())
        if not skin_suffix and base_alias.endswith("_beamxp_tc"):
            base_names.setdefault(base_alias[: -len("_beamxp_tc")], str(key))

    # Skins are named for the base they skin, so every base has to be named
    # first -- a manifest lists them in whatever order the exporter found them.
    pending: list[tuple[list[str], dict[str, object] | None, dict[str, object]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        aliases = [str(alias) for alias in entry.get("aliases", []) if str(alias).strip()]
        maps = entry.get("maps", {})
        if not aliases or not isinstance(maps, dict):
            continue
        for variant_aliases, source_material in _entry_material_variants(entry, aliases):
            if variant_aliases:
                pending.append((variant_aliases, source_material, entry))

    def is_skin(item: tuple[list[str], dict[str, object] | None, dict[str, object]]) -> bool:
        return bool(_split_material_skin_suffix(_normalise_material_alias(item[0][0]))[1])

    for variant_aliases, source_material, entry in sorted(pending, key=is_skin):
        material_name = _texture_correction_material_name(
            variant_aliases[0], used_names, base_names
        )
        # ``partKeys`` is present only where the exporter corrected one texture
        # more than once, and names the meshes this copy was corrected for.
        part_keys = [
            str(key)
            for key in (entry.get("partKeys") or [])
            if isinstance(key, str) and key.strip()
        ]
        for alias in variant_aliases:
            normalised = _normalise_material_alias(alias)
            if part_keys:
                for part_key in part_keys:
                    by_part.setdefault(part_key, {}).setdefault(normalised, material_name)
            else:
                shared.setdefault(normalised, material_name)

        by_source, by_stage = _entry_corrected_texture_outputs(
            job_dir,
            target_dir,
            output_root,
            material_name,
            entry,
        )
        if source_material is not None:
            existing[material_name] = _retarget_material_document(
                source_material,
                material_name,
                by_source,
                by_stage,
            )
        else:
            existing[material_name] = _fallback_texture_correction_material(
                material_name,
                by_stage,
            )

    materials = TextureCorrectionMaterials(shared, by_part)
    if materials:
        material_file.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return materials


def _texture_correction_switch_base_aliases(job_dir: Path) -> set[str]:
    manifest = job_dir / "rhd_materials.json"
    if not manifest.is_file():
        return set()
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return set()
    entries = document.get("materials", []) if isinstance(document, dict) else []
    aliases: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("switchBaseAliases", []):
            if isinstance(alias, str) and alias.strip():
                aliases.add(_normalise_material_alias(alias))
    return aliases


def _retarget_collada_materials(root: ET.Element, alias_to_material: dict[str, str]) -> None:
    if not alias_to_material:
        return
    for primitive in root.findall(".//*[@material]", NS):
        material = primitive.get("material") or ""
        replacement = alias_to_material.get(_normalise_material_alias(material))
        if replacement:
            primitive.set("material", replacement)
    for binding in root.findall(".//c:instance_material", NS):
        aliases = [
            binding.get("symbol") or "",
            (binding.get("target") or "").lstrip("#"),
        ]
        replacement = next(
            (
                alias_to_material[_normalise_material_alias(alias)]
                for alias in aliases
                if _normalise_material_alias(alias) in alias_to_material
            ),
            None,
        )
        if replacement:
            binding.set("symbol", replacement)
            binding.set("target", f"#{replacement}-material")


def _collada_child(root: ET.Element, name: str) -> ET.Element | None:
    return root.find(f"./c:{name}", NS)


def _ensure_collada_library(root: ET.Element, name: str) -> ET.Element:
    existing = _collada_child(root, name)
    if existing is not None:
        return existing
    library = ET.Element(_collada_q(name))
    children = list(root)
    insert_at = next(
        (
            index
            for index, child in enumerate(children)
            if child.tag in {
                _collada_q("library_effects"),
                _collada_q("library_materials"),
                _collada_q("library_geometries"),
                _collada_q("library_visual_scenes"),
            }
        ),
        len(children),
    )
    root.insert(insert_at, library)
    return library


def _collada_material_aliases(material: ET.Element) -> set[str]:
    aliases = {
        material.get("id") or "",
        material.get("name") or "",
    }
    return {_normalise_material_alias(alias) for alias in aliases if alias}


def _clone_collada_library_entry(
    source_root: ET.Element,
    target_root: ET.Element,
    library_name: str,
    entry_id: str,
) -> None:
    target_library = _ensure_collada_library(target_root, library_name)
    if any(child.get("id") == entry_id for child in list(target_library)):
        return
    source_library = _collada_child(source_root, library_name)
    if source_library is None:
        return
    for source_entry in list(source_library):
        if source_entry.get("id") == entry_id:
            target_library.append(copy.deepcopy(source_entry))
            return


def _ensure_retargeted_collada_materials(
    target_root: ET.Element,
    source_root: ET.Element,
    alias_to_material: dict[str, str],
) -> None:
    if not alias_to_material:
        return
    target_materials = _ensure_collada_library(target_root, "library_materials")
    existing_ids = {
        material.get("id") or ""
        for material in target_materials.findall("c:material", NS)
    }
    source_materials = source_root.findall("./c:library_materials/c:material", NS)
    for source_material in source_materials:
        replacement = next(
            (
                alias_to_material[alias]
                for alias in _collada_material_aliases(source_material)
                if alias in alias_to_material
            ),
            None,
        )
        if not replacement:
            continue
        material_id = f"{replacement}-material"
        if material_id in existing_ids:
            continue
        material = copy.deepcopy(source_material)
        material.set("id", material_id)
        material.set("name", replacement)
        target_materials.append(material)
        existing_ids.add(material_id)

        instance_effect = material.find("c:instance_effect", NS)
        effect_url = instance_effect.get("url", "") if instance_effect is not None else ""
        if effect_url.startswith("#"):
            _clone_collada_library_entry(
                source_root,
                target_root,
                "library_effects",
                effect_url.lstrip("#"),
            )


def _append_texture_correction_dae(
    target_dae: Path,
    source_dae: Path,
    node_ids: set[str],
    alias_to_material: dict[str, str],
    superseded_nodes: set[str] | None = None,
) -> list[str]:
    """Append the per-material split meshes, dropping what they replace.

    ``superseded_nodes`` names the whole-mesh copies whose flexbody rows the
    caller is about to repoint at these splits. They are removed here, while
    the tree is open and before it is written, rather than left for a later
    pass -- a superseded node carries its geometry with it, and geometry is
    where the bytes are.
    """
    if not node_ids:
        return []
    target_tree = ET.parse(target_dae)
    target_root = target_tree.getroot()
    target_geometries = target_root.find(".//c:library_geometries", NS)
    target_scene = target_root.find(".//c:library_visual_scenes/c:visual_scene", NS)
    if target_geometries is None or target_scene is None:
        raise RuntimeError(f"{target_dae} is missing library_geometries or visual_scene")

    source_tree = ET.parse(source_dae)
    source_root = source_tree.getroot()
    _ensure_retargeted_collada_materials(target_root, source_root, alias_to_material)
    _retarget_collada_materials(source_root, alias_to_material)
    parents = _collada_parent_map(source_root)
    source_geometries = {
        geometry.get("id"): geometry
        for geometry in source_root.findall(".//c:geometry", NS)
        if geometry.get("id")
    }

    appended_nodes: list[ET.Element] = []
    appended_geometries: list[ET.Element] = []
    appended_geometry_ids: set[str] = set()
    for source_node in source_root.findall(".//c:node", NS):
        node_id = source_node.get("id") or ""
        if node_id not in node_ids:
            continue
        node_matrix = _collada_world_matrix(source_node, parents)
        node = copy.deepcopy(source_node)
        _set_identity_node_transform(node)
        appended_nodes.append(node)

        for instance in source_node.findall(".//c:instance_geometry", NS):
            geometry_id = (instance.get("url") or "").lstrip("#")
            if not geometry_id or geometry_id in appended_geometry_ids:
                continue
            source_geometry = source_geometries.get(geometry_id)
            if source_geometry is None:
                continue
            geometry = copy.deepcopy(source_geometry)
            _bake_geometry_transform(geometry, node_matrix)
            appended_geometries.append(geometry)
            appended_geometry_ids.add(geometry_id)

    if not appended_nodes:
        return []

    appended_node_ids = {node.get("id") for node in appended_nodes}
    for child in list(target_scene):
        if child.get("id") in appended_node_ids:
            target_scene.remove(child)
    for child in list(target_geometries):
        if child.get("id") in appended_geometry_ids:
            target_geometries.remove(child)
    for geometry in appended_geometries:
        target_geometries.append(geometry)
    for node in appended_nodes:
        target_scene.append(node)

    if superseded_nodes:
        for child in list(target_scene):
            if child.get("id") in superseded_nodes or child.get("name") in superseded_nodes:
                target_scene.remove(child)
        # A geometry may be shared, so only drop the ones nothing instances now.
        still_used = {
            (instance.get("url") or "").lstrip("#")
            for instance in target_scene.findall(".//c:instance_geometry", NS)
        }
        for child in list(target_geometries):
            if child.get("id") not in still_used:
                target_geometries.remove(child)

    write_xml_tree(target_tree, target_dae)
    return [str(node.get("id") or "") for node in appended_nodes]


def _node_material_symbols(dae_path: Path, node_ids: set[str]) -> dict[str, set[str]]:
    """The materials each named node's geometry actually paints with.

    Read off the geometry's own primitives, never the node's
    ``instance_material`` table: a split keeps the whole mesh's binding table on
    every piece, so asking the node says each piece paints with everything and
    tells the pieces apart not at all. The primitives are what the piece really
    carries.
    """
    if not node_ids:
        return {}
    try:
        root = ET.parse(dae_path).getroot()
    except (OSError, ET.ParseError):
        return {}
    geometry_materials: dict[str, set[str]] = {}
    for geometry in root.findall(".//c:geometry", NS):
        geometry_id = geometry.get("id")
        if not geometry_id:
            continue
        geometry_materials[geometry_id] = {
            _normalise_material_alias(symbol)
            for primitive in geometry.iter()
            if (symbol := primitive.get("material"))
        }
    found: dict[str, set[str]] = {}
    for node in root.findall(".//c:node", NS):
        node_id = node.get("id") or ""
        if node_id not in node_ids:
            continue
        symbols: set[str] = set()
        for instance in node.findall(".//c:instance_geometry", NS):
            symbols |= geometry_materials.get((instance.get("url") or "").lstrip("#"), set())
        found[node_id] = symbols
    return found


def _mirror_row_split_target(
    pieces: Iterable[str],
    piece_materials: dict[str, set[str]],
    corrected_materials: set[str],
) -> str:
    """Which split piece a ``mirrors`` row should follow, or "" when unclear.

    Splitting a mesh for texture correction renames it, and ``addMirror`` binds
    by mesh name (``lua/common/jbeam/sections/mirror.lua``), so a row left
    naming the whole mesh binds nothing at all and the glass stops reflecting --
    the same failure the rename fix cured, arriving by a later rename.

    A row names one mesh, so it has to be the piece holding the reflective
    surface rather than the housing. A mirror material is the reflection: it
    carries no base colour of its own, which is exactly why the texture
    correction never records it and never renames it, so the glass is the piece
    the correction did not touch. Where that does not pick out a single piece
    the row is left alone -- a mirrors row aimed at a dashboard would turn the
    dashboard into a mirror, which is worse than the reflection staying broken.
    """
    candidates = [
        piece
        for piece in pieces
        if (symbols := piece_materials.get(piece))
        and not symbols & corrected_materials
    ]
    return candidates[0] if len(candidates) == 1 else ""


def _retarget_texture_correction_generated_nodes(
    target_dae: Path,
    source_dae: Path,
    node_ids: set[str],
    alias_to_material: dict[str, str],
) -> list[str]:
    if not node_ids or not alias_to_material:
        return []
    target_tree = ET.parse(target_dae)
    target_root = target_tree.getroot()
    source_root = ET.parse(source_dae).getroot()
    _ensure_retargeted_collada_materials(target_root, source_root, alias_to_material)

    geometries = {
        geometry.get("id"): geometry
        for geometry in target_root.findall(".//c:geometry", NS)
        if geometry.get("id")
    }
    retargeted: list[str] = []
    for node in target_root.findall(".//c:node", NS):
        node_id = node.get("id") or ""
        if node_id not in node_ids:
            continue
        _retarget_collada_materials(node, alias_to_material)
        for instance in node.findall(".//c:instance_geometry", NS):
            geometry_id = (instance.get("url") or "").lstrip("#")
            geometry = geometries.get(geometry_id)
            if geometry is not None:
                _retarget_collada_materials(geometry, alias_to_material)
        retargeted.append(node_id)

    if retargeted:
        write_xml_tree(target_tree, target_dae)
    return retargeted


def _replace_first_flexbody_mesh(row: str, mesh: str) -> str:
    return re.sub(
        r'(\[\s*)"((?:[^"\\]|\\.)*)"',
        rf'\1"{mesh}"',
        row,
        count=1,
    )


def _expand_texture_correction_flexbody_array(
    array_text: str,
    replacements: dict[str, list[str]],
) -> tuple[str, int]:
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
        return array_text, 0

    out: list[str] = []
    cursor = 0
    changed = 0
    for start, end, row in spans:
        out.append(array_text[cursor:start])
        mesh = flexbody_row_mesh(row)
        split_meshes = replacements.get(mesh or "")
        if split_meshes:
            line_joiner = ",\n" if "\n" in array_text else ", "
            out.append(line_joiner.join(_replace_first_flexbody_mesh(row, split) for split in split_meshes))
            changed += 1
        else:
            out.append(row)
        cursor = end
    out.append(array_text[cursor:])
    return "".join(out), changed


_GLOW_MATERIAL_STATE_RE = re.compile(
    r'("(?P<state>off|on|on_intense)"\s*:\s*)"@?(?P<material>(?:[^"\\]|\\.)*)"',
)


def _normalise_jbeam_material_alias(value: str) -> str:
    return _normalise_material_alias(value.lstrip("@"))


def _decode_jbeam_string(token: str) -> str:
    try:
        loaded = json.loads(token)
    except ValueError:
        return token.strip('"')
    return str(loaded)


def _top_level_jbeam_object_entries(
    object_text: str,
) -> list[tuple[str, int, int, str]]:
    """Return top-level ``"key": value`` spans from a JBeam object body."""
    masked = transform_helpers.mask_comments_preserve_offsets(object_text)
    brace = masked.find("{")
    if brace < 0:
        return []
    try:
        close = transform_helpers.find_matching(masked, brace, "{", "}") - 1
    except ValueError:
        return []

    entries: list[tuple[str, int, int, str]] = []
    idx = brace + 1
    while idx < close:
        while idx < close and masked[idx] in " \t\r\n,":
            idx += 1
        if idx >= close:
            break
        if masked[idx] != '"':
            idx += 1
            continue
        key_start = idx
        idx += 1
        escape = False
        while idx < close:
            ch = masked[idx]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                idx += 1
                break
            idx += 1
        key_token = object_text[key_start:idx]
        key = _decode_jbeam_string(key_token)
        while idx < close and masked[idx] in " \t\r\n":
            idx += 1
        if idx >= close or masked[idx] != ":":
            continue
        idx += 1
        while idx < close and masked[idx] in " \t\r\n,":
            idx += 1
        value_start = idx
        if idx >= close:
            break
        try:
            if masked[idx] == "{":
                value_end = transform_helpers.find_matching(masked, idx, "{", "}")
            elif masked[idx] == "[":
                value_end = transform_helpers.find_matching(masked, idx, "[", "]")
            elif masked[idx] == '"':
                idx += 1
                escape = False
                while idx < close:
                    ch = masked[idx]
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        idx += 1
                        break
                    idx += 1
                value_end = idx
            else:
                while idx < close and masked[idx] not in ",\r\n}":
                    idx += 1
                value_end = idx
        except ValueError:
            break
        entries.append((key, key_start, value_end, object_text[value_start:value_end]))
        idx = value_end
    return entries


def _replace_glow_material_states(
    value_text: str,
    base_key: str,
    corrected_base: str,
    alias_to_material: dict[str, str],
) -> str:
    base_alias = _normalise_jbeam_material_alias(base_key)

    def replace(match: re.Match[str]) -> str:
        material = _decode_jbeam_string(f'"{match.group("material")}"')
        material_alias = _normalise_jbeam_material_alias(material)
        replacement = alias_to_material.get(material_alias)
        if replacement is None and corrected_base and material_alias == base_alias:
            replacement = corrected_base
        if replacement is None:
            return match.group(0)
        return f'{match.group(1)}"{replacement}"'

    return _GLOW_MATERIAL_STATE_RE.sub(replace, value_text)


def _retarget_unmapped_glowmap_state_entries(
    glow_text: str,
    entries: list[tuple[str, int, int, str]],
    alias_to_material: dict[str, str],
    switch_base_aliases: set[str],
) -> tuple[str, int]:
    if not alias_to_material:
        return glow_text, 0
    out: list[str] = []
    cursor = 0
    changed = 0
    for key, _start, end, value_text in entries:
        key_alias = _normalise_jbeam_material_alias(key)
        if key_alias in alias_to_material and key_alias not in switch_base_aliases:
            continue
        value_start = end - len(value_text)
        updated_value = _replace_glow_material_states(
            value_text,
            key,
            "",
            alias_to_material,
        )
        if updated_value == value_text:
            continue
        out.append(glow_text[cursor:value_start])
        out.append(updated_value)
        cursor = end
        changed += 1
    if not out:
        return glow_text, 0
    out.append(glow_text[cursor:])
    return "".join(out), changed


def _append_texture_correction_glowmap_entries(
    glow_text: str,
    material_alias_sets: Iterable[dict[str, str]],
    switch_base_aliases: set[str] | None = None,
) -> tuple[str, int]:
    entries = _top_level_jbeam_object_entries(glow_text)
    if not entries:
        return glow_text, 0
    switch_base_aliases = switch_base_aliases or set()
    combined_alias_to_material: dict[str, str] = {}
    for alias_to_material in material_alias_sets:
        for alias, material in alias_to_material.items():
            combined_alias_to_material.setdefault(alias, material)
    if not combined_alias_to_material:
        return glow_text, 0
    updated_glow_text, retargeted_entries = _retarget_unmapped_glowmap_state_entries(
        glow_text,
        entries,
        combined_alias_to_material,
        switch_base_aliases,
    )
    existing_keys = {_normalise_jbeam_material_alias(key) for key, *_ in entries}
    additions: list[str] = []
    for alias_to_material in material_alias_sets:
        if not alias_to_material:
            continue
        for key, _start, _end, value_text in entries:
            if _normalise_jbeam_material_alias(key) in switch_base_aliases:
                continue
            corrected_base = alias_to_material.get(_normalise_jbeam_material_alias(key))
            if not corrected_base:
                continue
            corrected_alias = _normalise_jbeam_material_alias(corrected_base)
            if corrected_alias in existing_keys:
                continue
            corrected_value = _replace_glow_material_states(
                value_text,
                key,
                corrected_base,
                combined_alias_to_material,
            )
            additions.append(f'{json.dumps(corrected_base)}:{corrected_value},')
            existing_keys.add(corrected_alias)
    if not additions:
        return updated_glow_text, retargeted_entries

    last_entry_start = entries[-1][1]
    line_start = glow_text.rfind("\n", 0, last_entry_start)
    indent = ""
    if line_start >= 0:
        indent = re.match(r"[ \t]*", glow_text[line_start + 1 : last_entry_start]).group(0)
    insertion = "".join(f"\n{indent}{entry}" for entry in additions)
    close = updated_glow_text.rfind("}")
    if close < 0:
        return updated_glow_text, retargeted_entries
    return (
        updated_glow_text[:close] + insertion + updated_glow_text[close:],
        retargeted_entries + len(additions),
    )


def _replace_all_jbeam_object_regions(text: str, key: str, transform) -> str:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:[\s,]*\{{')
    out: list[str] = []
    cursor = 0
    search_at = 0
    masked = transform_helpers.mask_comments_preserve_offsets(text)
    while True:
        match = pattern.search(masked, search_at)
        if match is None:
            break
        brace = masked.rfind("{", match.start(), match.end())
        try:
            end = transform_helpers.find_matching(masked, brace, "{", "}")
        except Exception:
            search_at = match.end()
            continue
        old = text[match.start() : end]
        new, _changed = transform(old)
        out.append(text[cursor : match.start()])
        out.append(new)
        cursor = end
        search_at = end
    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)


def _corrected_source_glowmap_entries(
    source_jbeam_texts: Iterable[str],
    material_alias_sets: Iterable[dict[str, str]],
    switch_base_aliases: set[str],
) -> dict[str, str]:
    combined: dict[str, str] = {}
    for alias_to_material in material_alias_sets:
        for alias, material in alias_to_material.items():
            combined.setdefault(_normalise_jbeam_material_alias(alias), material)
    if not combined:
        return {}

    corrected: dict[str, str] = {}

    def collect(glow_text: str) -> tuple[str, int]:
        for key, _start, _end, value_text in _top_level_jbeam_object_entries(glow_text):
            key_alias = _normalise_jbeam_material_alias(key)
            updated_value = _replace_glow_material_states(
                value_text,
                key,
                "",
                combined,
            )
            if updated_value == value_text and key_alias not in combined:
                continue
            output_key = key
            if key_alias in combined and key_alias not in switch_base_aliases:
                output_key = combined[key_alias]
                updated_value = _replace_glow_material_states(
                    value_text,
                    key,
                    output_key,
                    combined,
                )
            corrected.setdefault(output_key, updated_value)
        return glow_text, 0

    for source_text in source_jbeam_texts:
        _replace_all_jbeam_object_regions(source_text, "glowMap", collect)
    return corrected


def _upsert_glowmap_entries(
    glow_text: str,
    entries: dict[str, str],
) -> str:
    existing = _top_level_jbeam_object_entries(glow_text)
    out: list[str] = []
    cursor = 0
    consumed: set[str] = set()
    for key, key_start, end, value_text in existing:
        replacement = entries.get(key)
        if replacement is None:
            continue
        out.append(glow_text[cursor:key_start])
        out.append(f"{json.dumps(key)}:{replacement}")
        cursor = end
        consumed.add(key)
    if out:
        out.append(glow_text[cursor:])
        glow_text = "".join(out)

    missing = [(key, value) for key, value in entries.items() if key not in consumed]
    if not missing:
        return glow_text
    close = glow_text.rfind("}")
    if close < 0:
        return glow_text
    prefix = glow_text[:close].rstrip()
    separator = "" if prefix.endswith(("{", ",")) else ","
    additions = "".join(
        f"\n      {json.dumps(key)}:{value}," for key, value in missing
    )
    return prefix + separator + additions + "\n    " + glow_text[close:]


def _upsert_part_glowmap(
    text: str,
    part_name: str,
    entries: dict[str, str],
) -> tuple[str, bool]:
    if not entries:
        return text, False
    masked = transform_helpers.mask_comments_preserve_offsets(text)
    match = re.search(rf'"{re.escape(part_name)}"\s*:[\s,]*\{{', masked)
    if match is None:
        return text, False
    brace = masked.rfind("{", match.start(), match.end())
    try:
        end = transform_helpers.find_matching(masked, brace, "{", "}")
    except ValueError:
        return text, False
    part_body = text[brace:end]
    updated = _replace_all_jbeam_object_regions(
        part_body,
        "glowMap",
        lambda glow: (_upsert_glowmap_entries(glow, entries), len(entries)),
    )
    if updated == part_body:
        close = part_body.rfind("}")
        if close < 0:
            return text, False
        prefix = part_body[:close].rstrip()
        separator = "" if prefix.endswith(("{", ",")) else ","
        rows = "".join(
            f"\n      {json.dumps(key)}:{value}," for key, value in entries.items()
        )
        updated = (
            prefix
            + separator
            + "\n    \"glowMap\":{"
            + rows
            + "\n    },\n  "
            + part_body[close:]
        )
    return text[:brace] + updated + text[end:], updated != part_body


def _patch_texture_correction_jbeams(
    output_vehicle_dir: Path,
    replacements: dict[str, list[str]],
    material_alias_sets: Iterable[dict[str, str]] = (),
    switch_base_aliases: Iterable[str] = (),
    source_jbeam_texts: Iterable[str] = (),
    mirror_row_targets: dict[str, str] | None = None,
) -> dict[str, object]:
    patched_files: list[str] = []
    replaced_rows = 0
    mirror_rows = 0
    mirror_row_targets = mirror_row_targets or {}
    material_alias_sets = tuple(material_alias_sets)
    switch_base_aliases = set(switch_base_aliases)
    corrected_source_glow_entries = _corrected_source_glowmap_entries(
        source_jbeam_texts,
        material_alias_sets,
        switch_base_aliases,
    )
    if not replacements and not material_alias_sets:
        return {
            "files": patched_files,
            "replacedRows": replaced_rows,
            "mirrorRows": mirror_rows,
        }
    for path in sorted(output_vehicle_dir.rglob("*.jbeam")):
        original = path.read_text(encoding="utf-8")
        file_replacements = 0

        def replace_array(array_text: str) -> tuple[str, int]:
            nonlocal file_replacements
            new_text, changed = _expand_texture_correction_flexbody_array(array_text, replacements)
            file_replacements += changed
            return new_text, changed

        def replace_mirrors(array_text: str) -> tuple[str, int]:
            nonlocal mirror_rows
            new_text = rewrite_mirror_rows(array_text, mirror_row_targets, {})
            changed = 1 if new_text != array_text else 0
            mirror_rows += changed
            return new_text, changed

        updated = _replace_all_jbeam_array_regions(original, "flexbodies", replace_array)
        if mirror_row_targets:
            # The glass followed its mesh into the split; the row that binds it
            # has to follow too, or addMirror finds nothing to reflect into.
            updated = _replace_all_jbeam_array_regions(updated, "mirrors", replace_mirrors)
        if material_alias_sets:
            updated = _replace_all_jbeam_object_regions(
                updated,
                "glowMap",
                lambda text: _append_texture_correction_glowmap_entries(
                    text,
                    material_alias_sets,
                    switch_base_aliases,
                ),
            )
        if corrected_source_glow_entries:
            for part_name in replacements:
                updated, _changed = _upsert_part_glowmap(
                    updated,
                    part_name,
                    corrected_source_glow_entries,
                )
        if updated == original:
            continue
        path.write_text(updated, encoding="utf-8")
        patched_files.append(str(path))
        replaced_rows += file_replacements
    return {
        "files": patched_files,
        "replacedRows": replaced_rows,
        "mirrorRows": mirror_rows,
    }


def _replace_all_jbeam_array_regions(text: str, key: str, transform) -> str:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:[\s,]*\[')
    out: list[str] = []
    cursor = 0
    search_at = 0
    while True:
        match = pattern.search(text, search_at)
        if match is None:
            break
        bracket = text.rfind("[", match.start(), match.end())
        try:
            end = transform_helpers.find_matching(text, bracket, "[", "]")
        except Exception:
            search_at = match.end()
            continue
        old = text[bracket:end]
        new, _changed = transform(old)
        out.append(text[cursor:bracket])
        out.append(new)
        cursor = end
        search_at = end
    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)


_BEAM_NAVIGATOR_ROW_RE = re.compile(r'\[\s*"beamNavigator"\s*,\s*\{')
_JBEAM_STRING_FIELD_RE = r'("{field}"\s*:\s*)"(?:[^"\\]|\\.)*"'
_GENERATED_HAND_PART_RE = re.compile(r'(?:^|_)xp_(?:lhd|rhd)(?:_|$)', re.IGNORECASE)


def _runtime_alias(value: object) -> str:
    return str(value or "").strip().lstrip("@").lower()


def _replace_jbeam_string_field(object_text: str, field: str, value: str) -> str:
    pattern = re.compile(_JBEAM_STRING_FIELD_RE.format(field=re.escape(field)))
    replacement = lambda match: match.group(1) + json.dumps(value)
    updated, changed = pattern.subn(replacement, object_text, count=1)
    if changed:
        return updated
    close = object_text.rfind("}")
    if close < 0:
        return object_text
    prefix = object_text[:close].rstrip()
    separator = "" if prefix.endswith(("{", ",")) else ","
    return prefix + separator + f" {json.dumps(field)}:{json.dumps(value)}" + object_text[close:]


def _source_beam_navigator_objects(context: VehicleContext) -> dict[str, str]:
    controllers: dict[str, str] = {}
    for text in context.jbeam_texts.values():
        masked = transform_helpers.mask_comments_preserve_offsets(text)
        for match in _BEAM_NAVIGATOR_ROW_RE.finditer(masked):
            brace = masked.rfind("{", match.start(), match.end())
            try:
                end = transform_helpers.find_matching(masked, brace, "{", "}")
            except ValueError:
                continue
            object_text = text[brace:end]
            material_match = re.search(
                _JBEAM_STRING_FIELD_RE.format(field="screenMaterialName"),
                object_text,
            )
            if material_match is None:
                continue
            try:
                screen_material = json.loads(material_match.group(0).split(":", 1)[1])
            except Exception:
                continue
            alias = _runtime_alias(screen_material)
            if alias:
                controllers.setdefault(alias, object_text)
    return controllers


def _source_glow_entries_for_runtime_alias(
    context: VehicleContext,
    runtime_alias: str,
) -> dict[str, str]:
    entries: dict[str, str] = {}

    def collect(glow_text: str) -> tuple[str, int]:
        for key, _start, _end, value_text in _top_level_jbeam_object_entries(glow_text):
            references = {
                _runtime_alias(match.group(1))
                for match in re.finditer(r':\s*"(@?(?:[^"\\]|\\.)*)"', value_text)
            }
            if runtime_alias in references:
                entries.setdefault(key, value_text)
        return glow_text, 0

    for source_text in context.jbeam_texts.values():
        _replace_all_jbeam_object_regions(source_text, "glowMap", collect)
    return entries


def _replace_runtime_alias_in_glow_entry(
    value_text: str,
    source_alias: str,
    target_alias: str,
) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        if _runtime_alias(value) != source_alias:
            return match.group(0)
        return match.group(0)[: match.group(0).find('"')] + json.dumps(target_alias)

    return re.sub(r':\s*"(@?(?:[^"\\]|\\.)*)"', replace, value_text)


_CONTROLLER_LUA_MEMBER_RE = re.compile(r"(?:^|/)lua/controller/(.+)\.lua$", re.IGNORECASE)


def _source_controller_lua(context: VehicleContext) -> dict[str, tuple[str, str]]:
    """Controller file name -> (archive member, source text) for the mod's Lua.

    ``lua/vehicle/controller.lua`` loads a jbeam controller row through
    ``require("controller/" .. fileName)``, so a vehicle's own
    ``lua/controller`` directory is the name space its controllers live in and
    the row's ``fileName`` is the path within it.
    """
    try:
        archive = zipfile.ZipFile(context.source_zip, "r")
    except Exception:
        return {}
    controllers: dict[str, tuple[str, str]] = {}
    with archive:
        for member in archive.namelist():
            match = _CONTROLLER_LUA_MEMBER_RE.search(member)
            if match is None:
                continue
            try:
                text = archive.read(member).decode("utf-8", errors="replace")
            except Exception:
                continue
            controllers.setdefault(match.group(1), (member, text))
    return controllers


def _lua_owned_runtime_aliases(
    context: VehicleContext,
    runtime_aliases: Iterable[str],
) -> dict[str, str]:
    """Runtime texture alias -> the controller file name that hard-codes it.

    ``htmlTexture.create`` ends in ``obj:createWebView(tag, ...)`` and that tag
    is a single global key: the first vehicle to ask for it owns it, and a
    second vehicle asking for the same tag gets nothing.  That is why a
    conversion spawned beside its donor leaves one of the two live screens
    black, whichever spawned second.  A tag the controller's own Lua names
    cannot be moved from jbeam the way ``screenMaterialName`` can, so the
    conversion has to carry its own copy of that controller.
    """
    wanted = {alias for alias in runtime_aliases if alias}
    if not wanted:
        return {}
    owners: dict[str, str] = {}
    for file_name, (_member, text) in _source_controller_lua(context).items():
        for match in re.finditer(r'"@([^"\\]+)"', text):
            alias = _runtime_alias(match.group(1))
            if alias in wanted:
                owners.setdefault(alias, file_name)
    return owners


def _retag_controller_lua(text: str, source_alias: str, target_alias: str) -> str:
    """Point one controller's hard-coded runtime tag at the conversion's own."""
    pattern = re.compile(r'"@(' + re.escape(source_alias) + r')"', re.IGNORECASE)
    return pattern.sub('"@' + target_alias + '"', text)


_LOCAL_PAGE_RE = re.compile(r'"local://local/([^"]+\.html)"', re.IGNORECASE)


def _mirror_page_style(origin: float) -> str:
    """Reflect the rendered page about ``origin`` of its own width.

    Reflecting the page is the only correction a live screen can take: there is
    no image to rewrite, and no UV island to reflect either, because a
    texture-corrected mesh is rebuilt by the symmetry sweep and never sees the
    flip scope.  The axis has to be the middle of the window the quad actually
    samples, not the middle of the page -- the LC500's cluster reads
    u 0.275..0.785, so reflecting about the page instead slides the dial 12% of
    the quad's width to the left and tucks part of it behind the binnacle.
    """
    return (
        '<style id="beamxp-mirror">'
        f"html{{transform:scaleX(-1);transform-origin:{origin * 100:.4f}% 50%;}}"
        "</style>"
    )


def _mirrored_screen_page(page: str, origin: float = 0.5) -> str:
    """Return the page rendered left-to-right reversed about ``origin``."""
    if 'id="beamxp-mirror"' in page:
        return page
    style = _mirror_page_style(origin)
    match = re.search(r"</head\s*>", page, re.IGNORECASE)
    if match is not None:
        return page[: match.start()] + style + "\n" + page[match.start():]
    return style + "\n" + page


def _sampled_u_centre(context: VehicleContext, symbols: set[str]) -> float | None:
    """Middle of the horizontal texture window ``symbols`` sample, or None.

    A screen quad rarely reads the whole page: the LC500's cluster takes the
    middle half of it. The reflection has to turn that window about its own
    centre so the same pixels stay on the quad, exactly as the UV flip path
    reflects s within its own bounds rather than within 0..1.
    """
    if not symbols:
        return None
    wanted = {symbol.lower() for symbol in symbols}
    wanted |= {f"{symbol}-material" for symbol in wanted}
    lowest = highest = None
    try:
        archive = zipfile.ZipFile(context.source_zip, "r")
    except Exception:
        return None
    with archive:
        for dae_path in context.dae_paths:
            try:
                with archive.open(dae_path) as handle:
                    root = ET.parse(handle).getroot()
            except Exception:
                continue
            for mesh in root.iter(f"{{{NS['c']}}}mesh"):
                sources = {
                    source.get("id"): source
                    for source in mesh.findall("c:source", NS)
                }
                for primitive in mesh:
                    if primitive.tag.rpartition("}")[2] not in {"triangles", "polylist"}:
                        continue
                    if (primitive.get("material") or "").lower() not in wanted:
                        continue
                    inputs = primitive.findall("c:input", NS)
                    if not inputs:
                        continue
                    stride = max(int(n.get("offset", "0")) for n in inputs) + 1
                    texcoord = None
                    for node in inputs:
                        if node.get("semantic") != "TEXCOORD":
                            continue
                        if texcoord is None or int(node.get("set", "0")) < int(
                            texcoord.get("set", "0")
                        ):
                            texcoord = node
                    index_node = primitive.find("c:p", NS)
                    if texcoord is None or index_node is None:
                        continue
                    source = sources.get(texcoord.get("source", "")[1:])
                    if source is None:
                        continue
                    array = source.find("c:float_array", NS)
                    accessor = source.find("c:technique_common/c:accessor", NS)
                    if array is None or array.text is None:
                        continue
                    values = [float(v) for v in array.text.split()]
                    uv_stride = int(accessor.get("stride", "2")) if accessor is not None else 2
                    offset = int(texcoord.get("offset", "0"))
                    indices = [int(v) for v in (index_node.text or "").split()]
                    for start in range(offset, len(indices), stride):
                        u = values[indices[start] * uv_stride]
                        lowest = u if lowest is None else min(lowest, u)
                        highest = u if highest is None else max(highest, u)
            if lowest is not None:
                break
    if lowest is None or highest is None:
        return None
    return (lowest + highest) / 2.0


def _mirrored_controller_pages(
    context: VehicleContext,
    controller_text: str,
    suffix: str,
    origin: float = 0.5,
) -> tuple[str, dict[str, str]]:
    """Give a controller its own mirrored copy of each page it renders.

    The copy keeps the donor's directory so its relative fonts and images still
    resolve, and only the file name carries the conversion's id.
    """
    vehicle_root = str(context.vehicle_path).strip("/").lower()
    pages: dict[str, str] = {}
    updated = controller_text
    try:
        archive = zipfile.ZipFile(context.source_zip, "r")
    except Exception:
        return controller_text, {}
    with archive:
        members = {name.lower(): name for name in archive.namelist()}
        for match in _LOCAL_PAGE_RE.finditer(controller_text):
            reference = match.group(1)
            member = members.get(reference.lower())
            if member is None:
                continue
            relative = reference.lower().removeprefix(vehicle_root).strip("/")
            if not relative or relative == reference.lower():
                # Outside this vehicle's folder: not ours to copy or reflect.
                continue
            try:
                page = archive.read(member).decode("utf-8", errors="replace")
            except Exception:
                continue
            head, _, name = relative.rpartition("/")
            stem, _, extension = name.rpartition(".")
            target_name = f"{stem}_beamxp_{suffix}.{extension}"
            target_relative = f"{head}/{target_name}" if head else target_name
            pages[target_relative] = _mirrored_screen_page(page, origin)
            head_reference, _, _ = reference.rpartition("/")
            updated = updated.replace(
                match.group(0),
                '"local://local/'
                + (f"{head_reference}/{target_name}" if head_reference else target_name)
                + '"',
            )
    return updated, pages


def _source_materials_referencing_runtime_alias(
    context: VehicleContext,
    runtime_alias: str,
) -> tuple[str, ...]:
    """Material keys whose stages draw from one runtime texture."""
    try:
        archive = zipfile.ZipFile(context.source_zip, "r")
    except Exception:
        return ()
    keys: list[str] = []
    with archive:
        for member in archive.namelist():
            if not member.lower().endswith(".materials.json"):
                continue
            try:
                document = parse_beamng_json(
                    archive.read(member).decode("utf-8-sig", errors="replace"),
                    label=member,
                )
            except Exception:
                continue
            if not isinstance(document, dict):
                continue
            for key, material in document.items():
                if not isinstance(material, dict):
                    continue
                for stage in material.get("Stages", []):
                    if not isinstance(stage, dict):
                        continue
                    if any(
                        isinstance(value, str) and _runtime_alias(value) == runtime_alias
                        for value in stage.values()
                        if isinstance(value, str) and value.strip().startswith("@")
                    ):
                        alias = _runtime_alias(key)
                        if alias and alias not in keys:
                            keys.append(alias)
                        break
    return tuple(keys)


def _rename_controller_file(part_body: str, source: str, target: str) -> str:
    """Point a part's controller row at the conversion's copy of that file."""
    controller = transform_helpers.extract_named_array(part_body, "controller")
    if not controller:
        return part_body
    pattern = re.compile(r'(\[\s*)"' + re.escape(source) + r'"(?=\s*[,\]])')
    updated, changed = pattern.subn(
        lambda match: match.group(1) + json.dumps(target), controller
    )
    if not changed:
        return part_body
    return part_body.replace(controller, updated, 1)


def _rebind_part_glow_materials(
    part_body: str,
    renames: dict[str, str],
) -> tuple[str, set[str]]:
    """Point a part's own glow rows at the conversion's copies of a material.

    Returns the rewritten part and which trigger keys were touched, so the
    caller only has to fall back to the donor's row for a trigger this part
    does not carry.
    """
    glow = transform_helpers.extract_keyed_object(part_body, "glowMap")
    if not glow or not renames:
        return part_body, set()
    touched: set[str] = set()
    updated_entries: dict[str, str] = {}
    for key, _start, _end, value_text in _top_level_jbeam_object_entries(glow):
        rewritten = value_text
        for source, target in renames.items():
            rewritten = _replace_runtime_alias_in_glow_entry(rewritten, source, target)
        if rewritten != value_text:
            updated_entries[key] = rewritten
            touched.add(key)
    if not updated_entries:
        return part_body, set()
    return (
        part_body.replace(glow, _upsert_glowmap_entries(glow, updated_entries), 1),
        touched,
    )


def _patch_controller_owned_screen_parts(
    text: str,
    specs: list[dict[str, object]],
) -> tuple[str, int]:
    """Rewrite generated parts that load a controller owning a runtime tag."""
    key_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:[\s,]*\{')
    masked = transform_helpers.mask_comments_preserve_offsets(text)
    out: list[str] = []
    cursor = 0
    changed = 0
    for match in key_pattern.finditer(masked):
        if match.start() < cursor:
            continue
        brace = masked.find("{", match.start(), match.end())
        try:
            end = transform_helpers.find_matching(masked, brace, "{", "}")
        except ValueError:
            continue
        part_id = match.group(1)
        part_body = text[match.start():end]
        if (
            '"slotType"' not in part_body
            or _GENERATED_HAND_PART_RE.search(part_id) is None
        ):
            continue
        updated = part_body
        for spec in specs:
            renamed = _rename_controller_file(
                updated,
                str(spec["sourceController"]),
                str(spec["targetController"]),
            )
            if renamed == updated:
                continue
            # Rewriting the part's own rows rather than replacing them with the
            # donor's keeps whatever texture correction has already done to the
            # other states of the same trigger.
            updated, rebound = _rebind_part_glow_materials(
                renamed, dict(spec["materialRenames"])
            )
            missing = {
                key: value
                for key, value in dict(spec["glowEntries"]).items()
                if key not in rebound
            }
            updated = _upsert_part_glow_entries(updated, missing)
        if updated == part_body:
            continue
        out.append(text[cursor:match.start()])
        out.append(updated)
        cursor = end
        changed += 1
    if not out:
        return text, 0
    out.append(text[cursor:])
    return "".join(out), changed


def _clone_runtime_material_definition(
    context: VehicleContext,
    source_alias: str,
    target_alias: str,
    runtime_aliases: dict[str, str] | None = None,
) -> dict[str, object] | None:
    try:
        archive = zipfile.ZipFile(context.source_zip, "r")
    except Exception:
        return None
    with archive:
        for member in archive.namelist():
            if not member.lower().endswith(".materials.json"):
                continue
            try:
                document = parse_beamng_json(
                    archive.read(member).decode("utf-8-sig", errors="replace"),
                    label=member,
                )
            except Exception:
                continue
            for key, material in document.items():
                if not isinstance(material, dict):
                    continue
                aliases = {
                    _runtime_alias(key),
                    _runtime_alias(material.get("name")),
                    _runtime_alias(material.get("mapTo")),
                }
                if source_alias not in aliases:
                    continue
                rewrites = runtime_aliases or {source_alias: target_alias}
                cloned = copy.deepcopy(material)
                cloned["name"] = target_alias
                cloned["mapTo"] = target_alias
                for stage in cloned.get("Stages", []):
                    if not isinstance(stage, dict):
                        continue
                    for stage_key, stage_value in tuple(stage.items()):
                        if not (
                            isinstance(stage_value, str)
                            and stage_value.strip().startswith("@")
                        ):
                            continue
                        replacement = rewrites.get(_runtime_alias(stage_value))
                        if replacement:
                            stage[stage_key] = "@" + replacement
                return cloned
    return None


def _append_controller_row(part_body: str, row: str, runtime_alias: str) -> str:
    controller = transform_helpers.extract_named_array(part_body, "controller")
    if controller:
        if runtime_alias.lower() in controller.lower():
            return part_body
        close = controller.rfind("]")
        if close < 0:
            return part_body
        prefix = controller[:close].rstrip()
        separator = "" if prefix.endswith(("[", ",")) else ","
        updated = prefix + separator + "\n      " + row + ",\n    " + controller[close:]
        return part_body.replace(controller, updated, 1)

    close = part_body.rfind("}")
    if close < 0:
        return part_body
    prefix = part_body[:close].rstrip()
    separator = "" if prefix.endswith(("{", ",")) else ","
    addition = (
        separator
        + '\n    "controller": [\n      ["fileName"],\n      '
        + row
        + ",\n    ],\n  "
    )
    return prefix + addition + part_body[close:]


def _upsert_part_glow_entries(part_body: str, entries: dict[str, str]) -> str:
    if not entries:
        return part_body
    glow = transform_helpers.extract_keyed_object(part_body, "glowMap")
    if glow:
        updated = _upsert_glowmap_entries(glow, entries)
        return part_body.replace(glow, updated, 1)
    close = part_body.rfind("}")
    if close < 0:
        return part_body
    prefix = part_body[:close].rstrip()
    separator = "" if prefix.endswith(("{", ",")) else ","
    rows = "".join(f"\n      {json.dumps(key)}:{value}," for key, value in entries.items())
    return prefix + separator + '\n    "glowMap":{' + rows + "\n    },\n  " + part_body[close:]


def _patch_runtime_screen_parts(
    text: str,
    target_meshes: set[str],
    specs: list[dict[str, object]],
) -> tuple[str, int]:
    key_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:[\s,]*\{')
    masked = transform_helpers.mask_comments_preserve_offsets(text)
    out: list[str] = []
    cursor = 0
    changed = 0
    for match in key_pattern.finditer(masked):
        if match.start() < cursor:
            continue
        brace = masked.find("{", match.start(), match.end())
        try:
            end = transform_helpers.find_matching(masked, brace, "{", "}")
        except ValueError:
            continue
        part_id = match.group(1)
        part_body = text[match.start():end]
        if (
            '"slotType"' not in part_body
            or _GENERATED_HAND_PART_RE.search(part_id) is None
            or part_id not in target_meshes
            or not any(
            json.dumps(mesh) in part_body for mesh in target_meshes
            )
        ):
            continue
        updated = part_body
        for spec in specs:
            updated = _append_controller_row(
                updated,
                str(spec["controllerRow"]),
                str(spec["targetAlias"]),
            )
            # In place first: this runs after texture correction has rewritten
            # the other states of the same trigger, and replacing the row with
            # the donor's would hand those states back to the donor.
            updated, rebound = _rebind_part_glow_materials(
                updated,
                {str(spec["sourceAlias"]): str(spec["targetAlias"])},
            )
            updated = _upsert_part_glow_entries(
                updated,
                {
                    key: value
                    for key, value in dict(spec["glowEntries"]).items()
                    if key not in rebound
                },
            )
        if updated == part_body:
            continue
        out.append(text[cursor:match.start()])
        out.append(updated)
        cursor = end
        changed += 1
    if not out:
        return text, 0
    out.append(text[cursor:])
    return "".join(out), changed


def isolate_converted_runtime_screens(
    context: VehicleContext,
    output_vehicle_dir: Path,
    source_meshes: Iterable[str],
    target_hands: Iterable[str],
    reflected_geometry: bool = True,
) -> dict[str, object]:
    """Give switched HTML screens a conversion-owned webview/material key.

    Two stock configs can safely share their authored runtime identity. A
    converted mesh has different UV consumers and corrected overlay materials,
    so it must not recreate the stock vehicle's live texture under that key.
    """
    nav_scope = nav_screen_mesh_scope(context)
    selected_sources = set(source_meshes).intersection(nav_scope)
    target_meshes = {
        generated_mesh_name(source_mesh, hand)
        for source_mesh in selected_sources
        for hand in set(target_hands)
    }

    controllers = _source_beam_navigator_objects(context)
    suffix = mod_id_for_context(context).lower()
    material_definitions: dict[str, object] = {}
    specs: list[dict[str, object]] = []
    for source_alias, controller_object in controllers.items():
        source_entries = _source_glow_entries_for_runtime_alias(context, source_alias)
        # Direct-bound navigator materials do not need a glowMap override and
        # require COLLADA material retargeting, which is a separate path.
        if not source_entries:
            continue
        target_alias = f"{source_alias}_beamxp_{suffix}"
        material = _clone_runtime_material_definition(
            context, source_alias, target_alias
        )
        if material is None:
            continue
        controller = _replace_jbeam_string_field(
            controller_object, "screenMaterialName", "@" + target_alias
        )
        controller = _replace_jbeam_string_field(
            controller, "name", "beamxp_" + target_alias
        )
        glow_entries = {
            key: _replace_runtime_alias_in_glow_entry(
                value, source_alias, target_alias
            )
            for key, value in source_entries.items()
        }
        material_definitions[target_alias] = material
        specs.append(
            {
                "sourceAlias": source_alias,
                "targetAlias": target_alias,
                "controllerRow": '["beamNavigator", ' + controller + "]",
                "glowEntries": glow_entries,
            }
        )

    # Isolating the tag is needed whatever the conversion does to the geometry.
    # Reflecting the page is only right once the cabin is actually reflected --
    # a translate-mode conversion leaves the screen reading as authored.
    controller_specs, controller_materials, controller_lua, controller_pages = (
        _controller_owned_screen_specs(
            context, controllers, suffix, reflect_pages=reflected_geometry
        )
    )
    material_definitions.update(controller_materials)

    patched_files: list[str] = []
    for path in sorted(output_vehicle_dir.rglob("*.jbeam")):
        original = path.read_text(encoding="utf-8")
        updated = original
        if specs and target_meshes:
            updated, _changed = _patch_runtime_screen_parts(
                updated, target_meshes, specs
            )
        if controller_specs:
            updated, _changed = _patch_controller_owned_screen_parts(
                updated, controller_specs
            )
        if updated == original:
            continue
        path.write_text(updated, encoding="utf-8")
        patched_files.append(str(path))
    if not patched_files:
        return {"enabled": False, "materials": [], "jbeamFiles": []}

    written_lua: list[str] = []
    for file_name, text in controller_lua.items():
        path = output_vehicle_dir / "lua" / "controller" / f"{file_name}.lua"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written_lua.append(str(path))

    written_pages: list[str] = []
    for relative, text in controller_pages.items():
        path = output_vehicle_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written_pages.append(str(path))

    material_path = output_vehicle_dir / "beamxp_runtime_screens.materials.json"
    material_path.write_text(
        json.dumps(material_definitions, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "enabled": True,
        "materials": sorted(material_definitions),
        "jbeamFiles": patched_files,
        "controllerFiles": sorted(written_lua),
        "screenPages": sorted(written_pages),
    }


def _controller_owned_screen_specs(
    context: VehicleContext,
    navigator_controllers: dict[str, str],
    suffix: str,
    reflect_pages: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, str], dict[str, str]]:
    """Plan the isolation of runtime tags a mod's own controller hard-codes.

    The navigator path can move its tag because ``screenMaterialName`` is a
    jbeam field.  A controller that names its tag in Lua cannot, so the
    conversion ships a copy of that controller under its own file name, points
    the copy at its own tag, and re-binds every material that drew from the
    original.
    """
    aliases = _runtime_aliases_in_source_materials(context)
    owners = _lua_owned_runtime_aliases(
        context,
        (alias for alias in aliases if alias not in navigator_controllers),
    )
    if not owners:
        return [], {}, {}, {}

    lua_sources = _source_controller_lua(context)
    specs: list[dict[str, object]] = []
    materials: dict[str, object] = {}
    lua_files: dict[str, str] = {}
    page_files: dict[str, str] = {}
    for source_alias, source_controller in sorted(owners.items()):
        entry = lua_sources.get(source_controller)
        if entry is None:
            continue
        target_alias = f"{source_alias}_beamxp_{suffix}"
        head, _, stem = source_controller.rpartition("/")
        target_controller = f"{head}/{stem}_beamxp_{suffix}" if head else f"{stem}_beamxp_{suffix}"

        renames: dict[str, str] = {}
        for material_key in _source_materials_referencing_runtime_alias(
            context, source_alias
        ):
            target_material = f"{material_key}_beamxp_{suffix}"
            cloned = _clone_runtime_material_definition(
                context,
                material_key,
                target_material,
                {source_alias: target_alias},
            )
            if cloned is None:
                continue
            materials[target_material] = cloned
            renames[material_key] = target_material
        if not renames:
            continue

        glow_entries: dict[str, str] = {}
        for material_key, target_material in renames.items():
            for key, value in _source_glow_entries_for_runtime_alias(
                context, material_key
            ).items():
                glow_entries[key] = _replace_runtime_alias_in_glow_entry(
                    glow_entries.get(key, value), material_key, target_material
                )

        controller_text = _retag_controller_lua(entry[1], source_alias, target_alias)
        if reflect_pages:
            # The glow trigger keys are the symbols the DAE binds, which is what
            # tells us how much of the page this screen actually shows.
            origin = _sampled_u_centre(context, set(glow_entries))
            controller_text, pages = _mirrored_controller_pages(
                context,
                controller_text,
                suffix,
                0.5 if origin is None else origin,
            )
            page_files.update(pages)
        lua_files[target_controller] = controller_text
        specs.append(
            {
                "sourceAlias": source_alias,
                "targetAlias": target_alias,
                "sourceController": source_controller,
                "targetController": target_controller,
                "materialRenames": renames,
                "glowEntries": glow_entries,
            }
        )
    return specs, materials, lua_files, page_files


def _runtime_aliases_in_source_materials(context: VehicleContext) -> tuple[str, ...]:
    """Every ``@`` runtime texture the donor's materials draw from."""
    try:
        archive = zipfile.ZipFile(context.source_zip, "r")
    except Exception:
        return ()
    aliases: list[str] = []
    with archive:
        for member in archive.namelist():
            if not member.lower().endswith(".materials.json"):
                continue
            try:
                document = parse_beamng_json(
                    archive.read(member).decode("utf-8-sig", errors="replace"),
                    label=member,
                )
            except Exception:
                continue
            if not isinstance(document, dict):
                continue
            for material in document.values():
                if not isinstance(material, dict):
                    continue
                for stage in material.get("Stages", []):
                    if not isinstance(stage, dict):
                        continue
                    for value in stage.values():
                        if not (
                            isinstance(value, str) and value.strip().startswith("@")
                        ):
                            continue
                        alias = _runtime_alias(value)
                        if alias and alias not in aliases:
                            aliases.append(alias)
    return tuple(aliases)


def integrate_texture_correction_artifacts(
    context: VehicleContext,
    output_root: Path,
    output_vehicle_dir: Path,
    texture_correction_report: dict[str, object],
    target_hands: Iterable[str],
    *,
    texture_correction_targets: dict[str, set[str]] | None = None,
    structural_sources: dict[str, str] | None = None,
) -> dict[str, object]:
    jobs = texture_correction_report.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        return {"enabled": False, "daePatches": [], "jbeamPatch": {"files": [], "replacedRows": 0}}

    dae_patches: list[dict[str, object]] = []
    row_replacements: dict[str, list[str]] = {}
    mirror_row_targets: dict[str, str] = {}
    glow_material_alias_sets: list[dict[str, str]] = []
    glow_switch_base_aliases: set[str] = set()
    hands = sorted(set(target_hands))
    texture_correction_targets = texture_correction_targets or {}
    structural_sources = structural_sources or {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        dae_member = str(job.get("dae") or "")
        output_directory = Path(str(job.get("outputDirectory") or ""))
        report_path = Path(str(job.get("reportPath") or ""))
        if not dae_member or not output_directory.is_dir() or not report_path.is_file():
            continue
        target_dae = generated_dae_output_path(output_root, output_vehicle_dir, context, dae_member)
        if not target_dae.is_file():
            dae_patches.append({"dae": dae_member, "target": str(target_dae), "error": "generated DAE not found"})
            continue
        alias_to_material = _prepare_texture_correction_materials(
            output_directory,
            target_dae.parent,
            output_root,
        )
        switch_base_aliases = _texture_correction_switch_base_aliases(output_directory)
        glow_switch_base_aliases.update(switch_base_aliases)
        if alias_to_material:
            glow_material_alias_sets.append(dict(alias_to_material))
        detail = json.loads(report_path.read_text(encoding="utf-8"))
        for dae_export in detail.get("dae_exports", []):
            if not isinstance(dae_export, dict):
                continue
            source_part = dae_export.get("source_part")
            if not isinstance(source_part, dict):
                continue
            source_mesh = str(source_part.get("key") or source_part.get("node_id") or "")
            # Where one texture needed correcting twice, the alias means a
            # different corrected material per mesh; ask for this mesh's.
            part_alias_to_material = alias_to_material.for_part(source_mesh)
            collada_alias_to_material = {
                alias: material
                for alias, material in part_alias_to_material.items()
                if alias not in switch_base_aliases
            }
            rows = dae_export.get("generated_flexbody_rows", [])
            split_nodes = [
                str(row.get("node_id") or "")
                for row in rows
                if isinstance(row, dict) and row.get("node_id")
            ]
            source_dae = Path(str(dae_export.get("dae_path") or ""))
            target_meshes = sorted(texture_correction_targets.get(source_mesh, {source_mesh}))
            structural_target_meshes = [
                target_mesh
                for target_mesh in target_meshes
                if structural_sources.get(target_mesh) == source_mesh
            ]
            row_target_meshes = [
                target_mesh
                for target_mesh in target_meshes
                if target_mesh not in structural_target_meshes
            ]
            appended: list[str] = []
            if row_target_meshes:
                appended = _append_texture_correction_dae(
                    target_dae,
                    source_dae,
                    set(split_nodes),
                    collada_alias_to_material,
                    superseded_nodes={
                        generated_mesh_name(target_mesh, hand)
                        for target_mesh in row_target_meshes
                        for hand in hands
                    },
                )
                if appended:
                    # Read back off the DAE just written: the pieces carry the
                    # retargeted material names, which is what has to be
                    # compared against the ones the correction minted.
                    piece_materials = _node_material_symbols(target_dae, set(appended))
                    glass = _mirror_row_split_target(
                        appended,
                        piece_materials,
                        {
                            _normalise_material_alias(name)
                            for name in collada_alias_to_material.values()
                        },
                    )
                    for target_mesh in row_target_meshes:
                        for hand in hands:
                            name = generated_mesh_name(target_mesh, hand)
                            row_replacements[name] = appended
                            if glass:
                                mirror_row_targets[name] = glass
            retargeted: list[str] = []
            if structural_target_meshes:
                retargeted = _retarget_texture_correction_generated_nodes(
                    target_dae,
                    source_dae,
                    {
                        generated_mesh_name(target_mesh, hand)
                        for target_mesh in structural_target_meshes
                        for hand in hands
                    },
                    collada_alias_to_material,
                )
            dae_patches.append(
                {
                    "sourceMesh": source_mesh,
                    "sourceDae": str(source_dae),
                    "targetDae": str(target_dae),
                    "appendedNodes": appended,
                    "retargetedNodes": retargeted,
                    "materialAliases": sorted(part_alias_to_material),
                    "colladaMaterialAliases": sorted(collada_alias_to_material),
                    "switchBaseAliases": sorted(switch_base_aliases),
                }
            )

    jbeam_patch = _patch_texture_correction_jbeams(
        output_vehicle_dir,
        row_replacements,
        glow_material_alias_sets,
        glow_switch_base_aliases,
        context.jbeam_texts.values(),
        mirror_row_targets,
    )
    return {
        "enabled": True,
        "daePatches": dae_patches,
        "jbeamPatch": jbeam_patch,
        "rowReplacements": row_replacements,
        "mirrorRowTargets": mirror_row_targets,
    }


def build_batch(
    context: VehicleContext,
    conversion: dict[str, object],
    *,
    write_zip: bool = True,
    install: bool = False,
    mods_folder: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> BuildResult:
    def emit_progress(message: str) -> None:
        if progress is not None:
            progress(message)

    emit_progress("Preparing build plan...")
    variant_targets, skipped, source_hands = _selected_variant_targets_and_source_hands(
        context, conversion
    )
    output_plans, skipped = selected_output_plans(
        context,
        conversion,
        variant_targets=variant_targets,
        target_skipped=skipped,
    )
    if not output_plans:
        raise RuntimeError("No trim outputs are selected")
    generated_variant_targets, authored_groups = split_authored_hand_drive_targets(
        context, conversion, variant_targets, source_hands
    )
    slot_pair_plans = slot_pair_plans_for_variants(
        context, conversion, sorted(variant_targets)
    )
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
    mirror_position_prop_meshes: set[str] = set()
    mirror_position_flexbody_meshes: set[str] = set()
    translated_flexbody_meshes: set[str] = set()
    translate_magnitudes: dict[str, float] = {}
    texture_flip_ids: set[str] = set()
    texture_correction_ids: set[str] = set()
    texture_correction_targets: dict[str, set[str]] = {}
    texture_correction_source_ids: set[str] = set()
    shared_atlas_dependency_targets: dict[str, set[str]] = {}
    force_mirrored_dependency_ids: set[str] = set()
    if generated_variant_targets:
        object_modes = active_part_modes(conversion)
        # A vehicle whose every converted trim is fully covered by an authored
        # swap needs no transformed parts at all, so an empty mode set is only
        # an error when nothing authored is carrying the conversion.
        swap_driven = bool(authored_groups or slot_pair_plans)
        if not object_modes and not swap_driven:
            raise RuntimeError(
                "Converted outputs require at least one Move, Mirror Move, Mirror, "
                "Swap Mesh, Replace Source, or slot pair"
            )
        selected_configs = sorted(generated_variant_targets)
        flexbody_meshes, prop_meshes, _all_meshes = selected_mesh_roles(context, selected_configs)
        mesh_scope = generated_mesh_scope(
            context, selected_configs, authored_groups, slot_pair_plans
        )
        if mesh_scope:
            object_modes = {mesh: mode for mesh, mode in object_modes.items() if mesh in mesh_scope}
            texture_correction_ids = active_texture_correction_mesh_ids(conversion) & mesh_scope
        else:
            texture_correction_ids = active_texture_correction_mesh_ids(conversion)
        for mesh in relocation_meshes(context, slot_pair_plans):
            object_modes[mesh] = MODE_MIRROR
        if not object_modes and not swap_driven:
            raise RuntimeError(
                "No Move, Mirror Move, Mirror, Swap Mesh, or Replace Source entries are used "
                "by the converted trims, and no slot pair applies to them"
            )
        texture_flip_ids = texture_flip_mesh_ids(context, object_modes)
        object_modes = fallback_structural_part_modes(
            context,
            conversion,
            object_modes,
            selected_configs=selected_configs,
        )
        structural_sources = structural_mirror_sources(context, conversion, object_modes)
        for mesh in sorted(texture_correction_ids):
            source_mesh = structural_sources.get(mesh, mesh)
            texture_correction_targets.setdefault(source_mesh, set()).add(mesh)
        texture_correction_source_ids = set(texture_correction_targets)
        if texture_correction_source_ids:
            for target_mesh, mode in object_modes.items():
                if mode not in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
                    continue
                source_mesh = structural_sources.get(target_mesh, target_mesh)
                shared_atlas_dependency_targets.setdefault(source_mesh, set()).add(
                    target_mesh
                )
                if mode == MODE_MIRROR_STRUCTURAL:
                    force_mirrored_dependency_ids.add(source_mesh)
        node_mirror_map = build_node_mirror_map(context.node_positions)
        translated_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_TRANSLATE and mesh in prop_meshes
        }
        mirrored_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_MIRROR and mesh in prop_meshes
        }
        mirror_position_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_MIRROR_POSITION and mesh in prop_meshes
        }
        structural_prop_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode in {MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE} and mesh in prop_meshes
        }
        translated_flexbody_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode == MODE_TRANSLATE and mesh in flexbody_meshes and mesh not in translated_prop_meshes
        }
        mirror_position_flexbody_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode == MODE_MIRROR_POSITION and mesh in flexbody_meshes and mesh not in mirror_position_prop_meshes
        }
        translate_magnitudes = part_translate_magnitudes(context, conversion, object_modes)
        # Zero only -- a negative offset is a deliberate "move it the other
        # way" override, not a missing delta.
        zero_translate = sorted(
            object_id
            for object_id, mode in object_modes.items()
            if mode == MODE_TRANSLATE and translate_magnitudes.get(object_id, 0.0) == 0
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
    emit_progress("Preparing build output...")
    clean_dir(output_root)
    build_dir.mkdir(parents=True, exist_ok=True)
    output_vehicle_dir = output_root / context.vehicle_path

    baked_shared_specs: list[BakedMeshSpec] = []
    generated_configs: list[str] = []
    generated_daes: list[Path] = []
    texture_correction_report: dict[str, object] = {
        "enabled": False,
        "requestedMeshes": sorted(texture_correction_ids),
        "jobs": [],
        "missing": [],
        "failures": [],
    }
    runtime_screen_patch: dict[str, object] = {
        "enabled": False,
        "materials": [],
        "jbeamFiles": [],
    }
    if variant_targets:
        emit_progress("Writing generated JBeam and config files...")
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
            mirror_position_prop_meshes,
            mirror_position_flexbody_meshes,
            structural_prop_meshes,
            baked_shared_specs,
            authored_groups,
            slot_pair_plans,
        ))
    if generated_variant_targets and object_modes:
        emit_progress("Generating transformed vehicle meshes...")
        generated_daes = generate_daes(
            context,
            output_root,
            output_vehicle_dir,
            object_modes,
            structural_sources,
            set(generated_variant_targets.values()),
            translate_magnitudes,
            translated_prop_meshes,
            translated_flexbody_meshes,
            context.jbeam_positioned_flexbodies,
            baked_shared_specs,
            texture_flip_ids,
        )
    if generated_variant_targets and texture_correction_ids:
        emit_progress(f"Running texture correction for {len(texture_correction_ids)} mesh(es)...")
        texture_correction_report = export_texture_correction_artifacts(
            context,
            context.project_dir / "build" / "texture_correction",
            texture_correction_source_ids,
            progress=emit_progress,
            shared_atlas_dependency_targets=shared_atlas_dependency_targets,
            force_mirrored_dependency_ids=force_mirrored_dependency_ids,
            bc7_quality=texture_quality_setting(conversion),
        )
        auto_included = texture_correction_report.get("autoIncludedTargets", {})
        if isinstance(auto_included, dict):
            for source_mesh, targets in auto_included.items():
                if not isinstance(targets, list):
                    continue
                texture_correction_targets.setdefault(str(source_mesh), set()).update(
                    str(target) for target in targets
                )
        emit_progress("Integrating texture-corrected meshes...")
        texture_correction_report["integration"] = integrate_texture_correction_artifacts(
            context,
            output_root,
            output_vehicle_dir,
            texture_correction_report,
            set(generated_variant_targets.values()),
            texture_correction_targets=texture_correction_targets,
            structural_sources=structural_sources,
        )
        texture_correction_report["pruned"] = prune_unused_texture_correction_assets(
            output_root, output_vehicle_dir
        )
        report_path = texture_correction_report.get("reportPath")
        if isinstance(report_path, str) and report_path:
            Path(report_path).write_text(
                json.dumps(texture_correction_report, indent=2) + "\n",
                encoding="utf-8",
            )
    if generated_variant_targets and object_modes:
        # After correction, not before: correction rebuilds a switched trigger's
        # row from the donor's jbeam, so a runtime rebind applied first is
        # handed straight back to the donor's material.
        runtime_screen_patch = isolate_converted_runtime_screens(
            context,
            output_vehicle_dir,
            texture_flip_ids,
            set(generated_variant_targets.values()),
            reflected_geometry=any(
                mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}
                for mode in object_modes.values()
            ),
        )
    texture_correction_report["runtimeScreens"] = runtime_screen_patch
    emit_progress("Writing original config outputs...")
    generated_configs.extend(write_original_plate_configs(
        context,
        output_vehicle_dir,
        conversion,
        original_configs,
    ))
    generated_configs.sort()
    write_mod_info(
        output_root,
        context,
        output_vehicle_dir,
        generated_configs,
        output_config_sources(context, conversion),
    )
    # Licence plates are generated as a separate pass over the written output
    # so plate logic stays fully decoupled from the handedness transforms.
    try:
        emit_progress("Applying licence plate settings...")
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
    embedded["builtAt"] = datetime.now(UTC).isoformat(timespec="seconds")
    embedded["build"] = {
        "generatedConfigs": generated_configs,
        "outputs": output_plans,
        "targetHands": variant_targets,
        "generatedTargetHands": generated_variant_targets,
        "authoredHandDriveGroups": authored_groups,
        "slotPairPlans": slot_pair_plans,
        "deltaMagnitude": delta_magnitude(context, conversion),
        "translateMagnitudes": translate_magnitudes,
        "mirroredPropMeshes": sorted(mirrored_prop_meshes),
        "textureFlipMeshes": sorted(texture_flip_ids),
        "textureCorrectionMeshes": sorted(texture_correction_ids),
        "textureCorrection": texture_correction_report,
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
        emit_progress("Packaging XP conversion zip...")
        package_name = package_name_for_context(context)
        package_zip = build_dir / package_name
        if install:
            package_zip = build_dir / f"{package_name}.installing"
        make_zip(output_root, package_zip)
    if install:
        if package_zip is None:
            raise RuntimeError("Install requires zip build")
        if mods_folder is None:
            raise RuntimeError("Install requested without a mods folder")
        mods_folder.mkdir(parents=True, exist_ok=True)
        emit_progress("Installing XP conversion zip...")
        installed_zip = mods_folder / package_name_for_context(context)
        _atomic_move_file(package_zip, installed_zip)
        package_zip = installed_zip
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
                if _copy_file_if_changed(plates_zip, installed_plates_zip):
                    plate_summary["libraryModInstalled"] = True
                else:
                    plate_summary["libraryModInstalled"] = False
                plate_summary["libraryModDesigns"] = plates_mod["designs"]

    save_conversion(context, conversion)
    emit_progress("Build complete.")
    return BuildResult(
        unpacked_dir=output_root,
        package_zip=package_zip,
        installed_zip=installed_zip,
        generated_configs=generated_configs,
        generated_daes=generated_daes,
        skipped_variants=skipped,
        plate_summary=plate_summary,
        installed_plates_zip=installed_plates_zip,
        texture_correction=texture_correction_report,
    )

__all__ = ['generated_mesh_scope', 'relocation_meshes', 'MOD_AUTHOR', 'MOD_VERSION', 'package_stem_for_context', 'package_name_for_context', 'mod_id_for_context', 'showcase_preview_for_build', 'mod_description_bbcode', 'write_mod_info', 'selected_variant_targets', 'selected_output_plans', 'split_authored_hand_drive_targets', 'texture_correction_asset_archives', 'export_texture_correction_artifacts', 'prune_unused_texture_correction_assets', 'integrate_texture_correction_artifacts', 'build_batch']
