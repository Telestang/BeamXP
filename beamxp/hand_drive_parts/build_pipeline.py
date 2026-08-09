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
from collections.abc import Callable, Iterable
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

    add(context.source_zip)
    if context.source_zip.parent.is_dir():
        for candidate in sorted(context.source_zip.parent.glob("*.zip"), key=lambda item: item.name.lower()):
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
    texture_config = replace(DEFAULT_RHD_CONFIG, detect_on_normal_map=True)

    by_source: dict[tuple[Path, str], list[str]] = {}
    for mesh_id in selected:
        obj = context.objects.get(mesh_id)
        if obj is None or not obj.dae_path:
            continue
        source_zip = obj.dae_source_zip or context.source_zip
        by_source.setdefault((source_zip, obj.dae_path), []).append(mesh_id)

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
                    "outputDirectory": str(job_dir),
                    "reportPath": str(preview.report_path) if preview.report_path is not None else None,
                    "daePaths": [str(path) for path in preview.dae_paths],
                    "textureCount": len(preview.textures),
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
    if source.suffix.lower() == ".png":
        candidates.append(source.with_suffix(".dds"))
    candidates.append(source)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return source


def _texture_correction_material_name(alias: str, used: set[str]) -> str:
    base = safe_id(f"{_normalise_material_alias(alias)}_beamxp_tc") or "beamxp_texture_corrected"
    candidate = base
    counter = 2
    while candidate.lower() in used:
        candidate = f"{base}_{counter}"
        counter += 1
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


def _materials_bound_by_generated_meshes(output_vehicle_dir: Path) -> set[str]:
    """Every material name the generated COLLADA actually binds."""
    bound: set[str] = set()
    for path in output_vehicle_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".dae":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        bound.update(_DAE_MATERIAL_SYMBOL_RE.findall(text))
        for value in _DAE_MATERIAL_ID_RE.findall(text):
            bound.add(value)
            if value.endswith("-material"):
                bound.add(value[: -len("-material")])
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

    Runs on the finished output rather than predicting up front, so it cannot
    be wrong about what is in use, and only ever removes files inside the
    generated vehicle folder.
    """
    bound = _materials_bound_by_generated_meshes(output_vehicle_dir)
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
        keep = {name: body for name, body in document.items() if name in bound}
        drop = {name: body for name, body in document.items() if name not in bound}
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


def _prepare_texture_correction_materials(
    job_dir: Path,
    target_dir: Path,
    output_root: Path,
) -> dict[str, str]:
    manifest = job_dir / "rhd_materials.json"
    if not manifest.is_file():
        return {}
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

    alias_to_material: dict[str, str] = {}
    used_names = {str(key).lower() for key in existing}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        aliases = [str(alias) for alias in entry.get("aliases", []) if str(alias).strip()]
        maps = entry.get("maps", {})
        if not aliases or not isinstance(maps, dict):
            continue
        material_name = _texture_correction_material_name(aliases[0], used_names)
        for alias in aliases:
            alias_to_material.setdefault(_normalise_material_alias(alias), material_name)

        by_source, by_stage = _entry_corrected_texture_outputs(
            job_dir,
            target_dir,
            output_root,
            material_name,
            entry,
        )
        source_material = _select_source_material(entry)
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

    if alias_to_material:
        material_file.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return alias_to_material


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


def _patch_texture_correction_jbeams(
    output_vehicle_dir: Path,
    replacements: dict[str, list[str]],
) -> dict[str, object]:
    patched_files: list[str] = []
    replaced_rows = 0
    if not replacements:
        return {"files": patched_files, "replacedRows": replaced_rows}
    for path in sorted(output_vehicle_dir.rglob("*.jbeam")):
        original = path.read_text(encoding="utf-8")
        file_replacements = 0

        def replace_array(array_text: str) -> tuple[str, int]:
            nonlocal file_replacements
            new_text, changed = _expand_texture_correction_flexbody_array(array_text, replacements)
            file_replacements += changed
            return new_text, changed

        updated = _replace_all_jbeam_array_regions(original, "flexbodies", replace_array)
        if updated == original:
            continue
        path.write_text(updated, encoding="utf-8")
        patched_files.append(str(path))
        replaced_rows += file_replacements
    return {"files": patched_files, "replacedRows": replaced_rows}


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
        detail = json.loads(report_path.read_text(encoding="utf-8"))
        for dae_export in detail.get("dae_exports", []):
            if not isinstance(dae_export, dict):
                continue
            source_part = dae_export.get("source_part")
            if not isinstance(source_part, dict):
                continue
            source_mesh = str(source_part.get("key") or source_part.get("node_id") or "")
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
                    alias_to_material,
                    superseded_nodes={
                        generated_mesh_name(target_mesh, hand)
                        for target_mesh in row_target_meshes
                        for hand in hands
                    },
                )
                if appended:
                    for target_mesh in row_target_meshes:
                        for hand in hands:
                            row_replacements[generated_mesh_name(target_mesh, hand)] = appended
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
                    alias_to_material,
                )
            dae_patches.append(
                {
                    "sourceMesh": source_mesh,
                    "sourceDae": str(source_dae),
                    "targetDae": str(target_dae),
                    "appendedNodes": appended,
                    "retargetedNodes": retargeted,
                    "materialAliases": sorted(alias_to_material),
                }
            )

    jbeam_patch = _patch_texture_correction_jbeams(output_vehicle_dir, row_replacements)
    return {
        "enabled": True,
        "daePatches": dae_patches,
        "jbeamPatch": jbeam_patch,
        "rowReplacements": row_replacements,
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
