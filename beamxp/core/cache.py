from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Iterable
from dataclasses import fields as dataclass_fields
from dataclasses import replace as dataclass_replace
from pathlib import Path

from beamxp.core.constants import HAND_LHD, HAND_RHD, HAND_UNKNOWN
from beamxp.core.dae import DaeObject
from beamxp.core.files import common_zip_candidates, project_dir_for, write_text_file
from beamxp.core.models import VehicleContext

# Bump whenever context-building logic changes in a way that affects cached
# VehicleContext content. Structural dataclass changes are also caught by the
# field-name fingerprint.
#
# 12: BeamNG 0.39 support. Catalog grouping moved to the engine's own rule, so a
#     cached context can hold the wrong trim set entirely (etk800 cached 7 of its
#     29 variants); config/info reading moved to the SJSON parser, which recovers
#     names, slot options and component trees that the old reader silently
#     dropped. Neither shows up in the game-file fingerprint, so without this
#     bump an existing install keeps loading the stale context.
# 13: flexbodies whose node groups this trim leaves empty are excluded, so the
#     used-part sets and preview scenes cached under 12 list meshes that never
#     spawn (the BX race trims' door controls).
CONTEXT_CACHE_VERSION = 13
HAND_DETECTION_CACHE_VERSION = 1
# Bump when the mesh numbering or the flexbody/prop role split changes shape or
# meaning, so an existing project stops reusing what the old rule produced.
PART_TABLE_CACHE_VERSION = 1


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
    context.node_groups_cache = {}
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
                node_groups_cache={},
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
    """Resolved used-part ids for a variant selection, persisted across sessions."""
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
    if selected and not ids:
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


# ----- part table preparation -------------------------------------------
#
# Building the parts table costs about five seconds on the etk800 and none of
# it depends on the session: numbering every mesh instance walks all 29 trims
# resolving each one's part tree (3.6 s), and splitting meshes into flexbodies
# and props walks them again (1.4 s). Both answers are small -- a few kilobytes
# -- and both are settled by the vehicle's own files, so they are written
# beside the project and read back on the next load.


def part_table_cache_fingerprint(context: VehicleContext) -> str:
    return hashlib.sha1(
        f"{PART_TABLE_CACHE_VERSION}:{context_fingerprint_hash(context.source_zip)}".encode("utf-8")
    ).hexdigest()


def _part_table_cache_file(context: VehicleContext, name: str) -> Path | None:
    """Where a part-table cache lives, or None if this context has no project.

    Saving already tolerates an unwritable project dir; not having one at all
    is the same situation, so it reads and writes nothing rather than failing
    the caller that only wanted the answer.
    """
    project_dir = getattr(context, "project_dir", None)
    return Path(project_dir) / name if project_dir else None


def _load_part_table_cache(path: Path | None, context: VehicleContext, section: str) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict) or data.get("fingerprint") != part_table_cache_fingerprint(context):
        return {}
    entries = data.get(section)
    return entries if isinstance(entries, dict) else {}


def _save_part_table_cache(path: Path | None, context: VehicleContext, section: str, entries: dict) -> None:
    if path is None:
        return
    try:
        write_text_file(
            path,
            json.dumps(
                {"fingerprint": part_table_cache_fingerprint(context), section: entries},
                indent=1,
            ),
        )
    except Exception:
        pass


def mesh_numbering_cache_path(context: VehicleContext) -> Path | None:
    return _part_table_cache_file(context, "mesh_numbering_cache.json")


def mesh_numbering_cache_key(plan_fingerprint: str) -> str:
    return hashlib.sha1(str(plan_fingerprint).encode("utf-8")).hexdigest()


def load_cached_mesh_numbering(
    context: VehicleContext,
    plan_fingerprint: str,
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]] | None:
    """Mesh instance numbering for one set of slot/side pairings.

    Two maps: each mesh's number for each place it appears, and the places
    each instance ref covers in trim order. They come from one walk and are
    useless apart, so a half-written entry counts as a miss.
    """
    entries = _load_part_table_cache(mesh_numbering_cache_path(context), context, "numbering")
    entry = entries.get(mesh_numbering_cache_key(plan_fingerprint))
    if not isinstance(entry, dict):
        return None
    numbering = entry.get("byMesh")
    keys_by_ref = entry.get("byRef")
    if not isinstance(numbering, dict) or not isinstance(keys_by_ref, dict):
        return None
    return (
        {
            str(mesh_id): {str(key): int(ordinal) for key, ordinal in keys.items()}
            for mesh_id, keys in numbering.items()
            if isinstance(keys, dict)
        },
        {
            str(ref): [str(key) for key in keys]
            for ref, keys in keys_by_ref.items()
            if isinstance(keys, list)
        },
    )


def save_cached_mesh_numbering(
    context: VehicleContext,
    plan_fingerprint: str,
    index: tuple[dict[str, dict[str, int]], dict[str, list[str]]],
    max_entries: int = 4,
) -> None:
    numbering, keys_by_ref = index
    path = mesh_numbering_cache_path(context)
    entries = _load_part_table_cache(path, context, "numbering")
    key = mesh_numbering_cache_key(plan_fingerprint)
    entries.pop(key, None)
    entries[key] = {
        "byMesh": {
            str(mesh_id): {str(number_key): int(ordinal) for number_key, ordinal in keys.items()}
            for mesh_id, keys in numbering.items()
        },
        "byRef": {
            str(ref): [str(number_key) for number_key in keys]
            for ref, keys in keys_by_ref.items()
        },
    }
    while len(entries) > max_entries:
        entries.pop(next(iter(entries)))
    _save_part_table_cache(path, context, "numbering", entries)


def mesh_roles_cache_path(context: VehicleContext) -> Path | None:
    return _part_table_cache_file(context, "mesh_roles_cache.json")


def load_cached_mesh_roles(
    context: VehicleContext,
) -> dict[str, tuple[set[str], set[str], set[str]]]:
    """Every trim's flexbody/prop/all mesh split that has been worked out before.

    Returned for the caller to seed `context.mesh_roles_cache` with, which is
    what the resolver reads; a trim missing from here is simply resolved.
    """
    entries = _load_part_table_cache(mesh_roles_cache_path(context), context, "meshRoles")
    roles: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for config_name, value in entries.items():
        if not isinstance(value, list) or len(value) != 3:
            continue
        if not all(isinstance(group, list) for group in value):
            continue
        if config_name not in context.variants:
            continue
        roles[str(config_name)] = tuple({str(mesh) for mesh in group} for group in value)
    return roles


def save_cached_mesh_roles(
    context: VehicleContext,
    roles: dict[str, tuple[set[str], set[str], set[str]]],
) -> None:
    entries = {
        str(config_name): [sorted(str(mesh) for mesh in group) for group in value]
        for config_name, value in roles.items()
        if isinstance(value, tuple) and len(value) == 3
    }
    if not entries:
        return
    _save_part_table_cache(mesh_roles_cache_path(context), context, "meshRoles", entries)


def clear_part_table_caches(context: VehicleContext) -> None:
    for path in (mesh_numbering_cache_path(context), mesh_roles_cache_path(context)):
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def variant_hands_cache_path(context: VehicleContext) -> Path:
    return context.project_dir / "variant_hands_cache.json"


def variant_hands_cache_key(signature: Iterable[str]) -> str:
    payload = json.dumps(tuple(sorted(str(item) for item in signature)), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def variant_hands_cache_fingerprint(context: VehicleContext) -> str:
    payload = f"{HAND_DETECTION_CACHE_VERSION}:{context_fingerprint_hash(context.source_zip)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalized_cached_variant_hands(
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


def load_cached_variant_hands_by_key(
    context: VehicleContext,
    key: str,
) -> dict[str, str] | None:
    memory = normalized_cached_variant_hands(context, context.variant_hands_cache.get(key))
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
    hands = normalized_cached_variant_hands(context, detections.get(key))
    if not hands:
        return None
    context.variant_hands_cache[key] = hands
    return dict(hands)


def save_cached_variant_hands_by_key(
    context: VehicleContext,
    key: str,
    hands: dict[str, str],
    max_entries: int = 8,
) -> None:
    normalized = normalized_cached_variant_hands(context, hands)
    if not normalized:
        return
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
                str(saved_key): normalized_cached_variant_hands(context, saved_hands)
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
