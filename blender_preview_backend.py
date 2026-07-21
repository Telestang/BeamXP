"""Blender backend for the full-vehicle hand-drive preview.

The tool computes one final vehicle-space matrix per flexbody/prop row using
the same engine-verified transform code the build uses, and references
geometry from the ORIGINAL DAE files by node name. This script just realizes
that payload: import each DAE once into a hidden library, then instance each
row as a linked duplicate with its matrix applied.

Collections:
  Converted Vehicle - the vehicle as it will look in game after conversion
  Stock Reference   - the same rows at stock placement (hidden by default)
  Source Library    - raw DAE imports (hidden; holds the shared mesh data)
"""
from __future__ import annotations

import json
import math
import re
import sys
import traceback
from xml.etree import ElementTree as ET
from pathlib import Path

COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
ET.register_namespace("", COLLADA_NS)


def payload_path_from_args() -> Path:
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        args = []
    if not args:
        raise RuntimeError("Missing Blender preview payload path")
    return Path(args[0])


def strip_blender_suffix(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", name)


def ensure_collection(bpy, name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def move_object_to_collection(obj, collection) -> None:
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for old_collection in list(obj.users_collection):
        if old_collection != collection:
            old_collection.objects.unlink(obj)


def import_dae(bpy, path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(path)
    before = {obj.name for obj in bpy.data.objects}
    if hasattr(bpy.ops.wm, "collada_import"):
        bpy.ops.wm.collada_import(filepath=str(path))
    else:
        bpy.ops.import_scene.dae(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj.name not in before]


def xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def safe_alias_path(path: Path, index: int, target_dir: Path) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:48] or "dae"
    return target_dir / f"{index:03d}_{stem}.dae"


def prepare_dae_for_import(path: Path, aliases: dict[str, str], index: int, target_dir: Path) -> Path:
    if not aliases:
        return path
    target_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(path)
    changed = False
    for node in tree.getroot().iter():
        if xml_local_name(node.tag) != "node":
            continue
        alias = aliases.get(node.get("id") or "") or aliases.get(node.get("name") or "")
        if not alias:
            continue
        node.set("id", alias)
        node.set("name", alias)
        changed = True
    if not changed:
        return path
    target = safe_alias_path(path, index, target_dir)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target


def node_aliases_for_payload(payload: dict) -> list[dict[str, str]]:
    dae_files = payload.get("dae_files", [])
    aliases_by_dae: list[dict[str, str]] = [dict() for _entry in dae_files]
    for instance in payload.get("instances", []):
        try:
            dae_index = int(instance["dae"])
        except Exception:
            continue
        if dae_index < 0 or dae_index >= len(aliases_by_dae):
            continue
        node = str(instance.get("node") or "")
        if len(node) <= 50:
            continue
        aliases = aliases_by_dae[dae_index]
        alias = aliases.get(node)
        if alias is None:
            alias = f"rhdnode_{dae_index:03d}_{len(aliases):04d}"
            aliases[node] = alias
        instance["import_node"] = alias
    return aliases_by_dae


def build_node_map(imported: list) -> dict:
    mapping: dict = {}
    for obj in imported:
        names = [obj.name or ""]
        data = getattr(obj, "data", None)
        if data is not None and getattr(data, "name", None):
            names.append(data.name)
        for raw in names:
            base = strip_blender_suffix(raw)
            for candidate in (raw, base, raw.strip("_ "), base.strip("_ ")):
                if candidate:
                    mapping.setdefault(candidate, obj)
    return mapping


def mesh_objects_for_base(base) -> list:
    if base.type == "MESH":
        return [base]
    meshes = []
    stack = list(getattr(base, "children", ()))
    while stack:
        child = stack.pop(0)
        if child.type == "MESH":
            meshes.append(child)
        stack.extend(getattr(child, "children", ()))
    return meshes


def find_layer_collection(layer_collection, name: str):
    if layer_collection.collection.name == name:
        return layer_collection
    for child in layer_collection.children:
        found = find_layer_collection(child, name)
        if found is not None:
            return found
    return None


def main() -> None:
    import bpy
    from mathutils import Matrix

    payload = json.loads(payload_path_from_args().read_text(encoding="utf-8"))
    label = str(payload.get("output_name") or payload.get("config_name") or "preview")
    show_unchanged = bool(payload.get("show_unchanged"))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    library_name = "Source Library"
    library_col = ensure_collection(bpy, library_name)
    changed_col = ensure_collection(bpy, f"{label} · converted parts")
    unchanged_col = ensure_collection(bpy, f"{label} · unchanged parts")
    stock_col = ensure_collection(bpy, f"{label} · stock positions of converted parts")

    aliases_by_dae = node_aliases_for_payload(payload)
    import_cache_dir = payload_path_from_args().parent / "dae_import"
    node_maps = []
    for index, entry in enumerate(payload.get("dae_files", [])):
        import_path = prepare_dae_for_import(
            Path(entry["path"]),
            aliases_by_dae[index] if index < len(aliases_by_dae) else {},
            index,
            import_cache_dir,
        )
        imported = import_dae(bpy, import_path)
        node_maps.append(build_node_map(imported))
        for obj in imported:
            move_object_to_collection(obj, library_col)
            obj.name = f"src.{obj.name}"
        print(f"imported {len(imported)} object(s) from {import_path}")

    missing: list[str] = []
    placed = changed = 0
    for instance in payload.get("instances", []):
        mapping = node_maps[instance["dae"]]
        base = None
        candidates = []
        if instance.get("import_node"):
            candidates.append(str(instance["import_node"]))
        candidates.extend(instance.get("node_names") or [instance["node"]])
        for name in candidates:
            name = str(name)
            base = mapping.get(name) or mapping.get(strip_blender_suffix(name)) or mapping.get(name.strip("_ "))
            if base is not None:
                break
        base_meshes = mesh_objects_for_base(base) if base is not None else []
        if not base_meshes:
            missing.append(f"{instance['node']} (part {instance['part']})")
            continue
        # Props render from node-LOCAL geometry: the engine discards the DAE
        # node's rotation (baseRotationGlobal supplies the rest rotation),
        # keeps translation (pivot) and scale.
        #
        # Flexbodies keep the node's rotation/scale, and whether the world
        # translation applies depends on whether this row authors its own pos
        # elsewhere (see flexbody_row_needs_node_translation in core.py). A row
        # that does is a reusable template the row itself positions --
        # pickup_common.DAE's gooseneck hitch is one row's pos:{y:0.325} away
        # from another's pos:{y:-0.03} -- so the node's translation is a real,
        # additional contribution and must be applied. A BARE row (mesh +
        # material groups, nothing else) means the vertices are already
        # authored at their final position and the node's translation is a
        # leftover export artefact the game ignores; applying it anyway sinks
        # the astra's fog light below the road and puts etk800's manual
        # shifter on the passenger side. Mirrors mesh_preview.build_scene.
        rebase = Matrix.Identity(4)
        if base is not None:
            node_matrix = base.matrix_world
            if instance.get("kind") == "prop":
                loc, _rot, scale = node_matrix.decompose()
                derotated = Matrix.Translation(loc) @ Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
                rebase = derotated @ node_matrix.inverted()
            elif not instance.get("keep_node_translation", True):
                rebase = Matrix.Translation(-node_matrix.translation)
        is_converted = instance.get("mode") not in (None, "", "skip")
        targets = [(changed_col if is_converted else unchanged_col, "matrix", instance["mesh"])]
        if is_converted:
            targets.append((stock_col, "stock_matrix", f"stock.{instance['mesh']}"))
        for collection, matrix_key, name in targets:
            flat = instance.get(matrix_key)
            if not flat:
                continue
            world = Matrix((flat[0:4], flat[4:8], flat[8:12], flat[12:16]))
            for base_mesh in base_meshes:
                duplicate = base_mesh.copy()  # linked duplicate: shares mesh data
                duplicate.name = name if len(base_meshes) == 1 else f"{name}.{strip_blender_suffix(base_mesh.name)}"
                duplicate.matrix_world = world @ rebase @ base_mesh.matrix_world
                duplicate["rhd_part"] = instance["part"]
                duplicate["rhd_mode"] = instance["mode"]
                duplicate["rhd_kind"] = instance["kind"]
                collection.objects.link(duplicate)
        placed += 1
        changed += int(is_converted)

    # Generated-output previews show the whole selected config by default.
    # The older analytic preview keeps unchanged context hidden. The raw DAE
    # imports are excluded so they cannot be mistaken for preview content.
    unchanged_col.hide_viewport = not show_unchanged
    unchanged_col.hide_render = not show_unchanged
    stock_col.hide_viewport = True
    stock_col.hide_render = True
    library_layer = find_layer_collection(bpy.context.view_layer.layer_collection, library_name)
    if library_layer is not None:
        library_layer.exclude = True

    write_notes(bpy, payload, placed, missing)
    frame_meshes = [obj for obj in changed_col.objects if obj.type == "MESH"]
    if show_unchanged:
        frame_meshes.extend(obj for obj in unchanged_col.objects if obj.type == "MESH")
    frame_scene(bpy, frame_meshes)

    # The preview stays an unsaved Blender instance; the user saves manually
    # from Blender if they want to keep it.
    print(
        f"preview ready (unsaved) | instances placed: {placed} "
        f"({changed} converted, {placed - changed} unchanged) | missing nodes: {len(missing)}"
    )
    for item in missing:
        print(f"  missing: {item}")


def write_notes(bpy, payload: dict, placed: int, missing: list) -> None:
    label = str(payload.get("output_name") or payload.get("config_name") or "preview")
    lines = [
        "BeamXP Full Vehicle Preview",
        "",
        f"Vehicle: {payload.get('vehicle_id')}",
        f"Config: {payload.get('config_name')} -> {payload.get('output_name')} "
        f"(target hand: {payload.get('target_hand')})",
        f"Instances placed: {placed}",
        "",
        "Collections (this preview shows ONE config/variant):",
        f"- '{label} · converted parts' (visible): parts the conversion moves, at their",
        "  converted in-game positions.",
        f"- '{label} · unchanged parts' ({'visible' if payload.get('show_unchanged') else 'hidden'}): the rest of the vehicle for context;",
        "  parts whose mode is skip render exactly at stock position, including any",
        "  driver-side meshes you have not assigned a mode yet.",
        f"- '{label} · stock positions of converted parts' (hidden): A/B reference.",
        "- 'Source Library' is excluded from the view layer (raw DAE imports).",
        "",
        "Mirrored parts use negative-scale matrices; shading may show flipped",
        "normals on them, which is expected for a placement preview.",
        "",
    ]
    calibration = payload.get("rotation_calibration") or {}
    if calibration:
        summary = ", ".join(f"{source}: {count}" for source, count in sorted(calibration.items()))
        lines.append(f"Prop rest rotations: {summary}.")
        lines.append("")
    skipped = payload.get("skipped_meshes") or {}
    if skipped:
        lines.append("Rows skipped by the exporter:")
        lines.extend(f"- {mesh}: {reason}" for mesh, reason in sorted(skipped.items()))
        lines.append("")
    if missing:
        lines.append("Nodes not found after DAE import:")
        lines.extend(f"- {item}" for item in missing)
    text = bpy.data.texts.new("RHD Preview Notes")
    text.write("\n".join(lines))


def frame_scene(bpy, meshes: list) -> None:
    from mathutils import Vector

    bpy.context.scene.unit_settings.system = "METRIC"
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        pass

    light_data = bpy.data.lights.new("RHD Preview Key", type="AREA")
    light_obj = bpy.data.objects.new("RHD Preview Key", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (2.8, -4.0, 4.5)
    light_data.energy = 650
    light_data.size = 5

    camera_data = bpy.data.cameras.new("RHD Preview Camera")
    camera_obj = bpy.data.objects.new("RHD Preview Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj

    bounds = combined_bounds(meshes)
    if bounds is None:
        center = Vector((0.0, 0.0, 0.0))
        radius = 2.0
    else:
        min_corner, max_corner = bounds
        center = (min_corner + max_corner) * 0.5
        radius = max((max_corner - min_corner).length * 0.5, 0.25)

    distance = radius * 3.0
    camera_obj.location = center + Vector((distance * 0.55, -distance * 0.9, distance * 0.45))
    direction = center - camera_obj.location
    camera_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 32
    camera_data.clip_start = 0.01
    camera_data.clip_end = max(distance * 10, 100)

    if bpy.context.screen is not None:
        for area in bpy.context.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region_3d = area.spaces.active.region_3d
            region_3d.view_perspective = "CAMERA"


def combined_bounds(meshes: list):
    from mathutils import Vector

    min_corner = Vector((math.inf, math.inf, math.inf))
    max_corner = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
            found = True
    if not found:
        return None
    return min_corner, max_corner


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
