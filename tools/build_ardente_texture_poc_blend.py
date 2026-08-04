from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


TARGET_NAMES = {"ardente_dashboard_xp_rhd", "ardente_console_xp_rhd"}
STOCK_NAMES = {"ardente_dashboard", "ardente_console"}


def _import_collada(path: Path):
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    before = set(bpy.data.objects)
    bpy.ops.wm.collada_import(filepath=str(path))
    bpy.ops.object.select_all(action="DESELECT")
    return [obj for obj in bpy.data.objects if obj not in before]


def _keep_named(imported, names: set[str]):
    kept = []
    for obj in imported:
        base = obj.name.split(".", 1)[0]
        if base in names:
            kept.append(obj)
    return kept


def _delete_objects(objects):
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action="DESELECT")


def _image_node(tree, path: Path, colorspace: str, location):
    import bpy

    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.image.colorspace_settings.name = colorspace
    node.location = location
    return node


def _build_rhd_material(texture_dir: Path):
    import bpy

    mat = bpy.data.materials.new("ardente_interior_xp_rhd_inspection")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (640, 0)
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (320, 0)
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    base = _image_node(
        tree,
        texture_dir / "ardente_interior_b.color_rhd.png",
        "sRGB",
        (-520, 260),
    )
    ao = _image_node(
        tree,
        texture_dir / "ardente_interior_ao.data_rhd.png",
        "Non-Color",
        (-520, 60),
    )
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.location = (-160, 190)
    tree.links.new(base.outputs["Color"], mix.inputs[6])
    tree.links.new(ao.outputs["Color"], mix.inputs[7])
    tree.links.new(mix.outputs[2], shader.inputs["Base Color"])

    normal = _image_node(
        tree,
        texture_dir / "ardente_interior_nm.normal_rhd.preview.png",
        "Non-Color",
        (-520, -420),
    )
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.location = (-160, -420)
    tree.links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])

    roughness = _image_node(
        tree,
        texture_dir / "ardente_interior_r.data_rhd.png",
        "Non-Color",
        (-520, -130),
    )
    tree.links.new(roughness.outputs["Color"], shader.inputs["Roughness"])

    metallic = _image_node(
        tree,
        texture_dir / "ardente_interior_m.data_rhd.png",
        "Non-Color",
        (-520, -270),
    )
    tree.links.new(metallic.outputs["Color"], shader.inputs["Metallic"])
    return mat


def _build_reference_material():
    import bpy

    mat = bpy.data.materials.new("stock_lhd_reference_gray")
    mat.diffuse_color = (0.45, 0.48, 0.50, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.45, 0.48, 0.50, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.75
    return mat


def _put_in_collection(objects, name: str):
    import bpy

    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for parent in list(obj.users_collection):
            parent.objects.unlink(obj)
        collection.objects.link(obj)
    return collection


def _frame_scene(objects):
    import bpy
    from mathutils import Vector

    if not objects:
        return
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).length, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - 3.0, center.z + 3.0))
    light = bpy.context.object
    light.name = "inspection_area_light"
    light.data.energy = 500
    light.data.size = 4

    bpy.ops.object.camera_add(
        location=(center.x, center.y - radius * 1.4, center.z + radius * 0.55),
        rotation=(math.radians(68), 0, 0),
    )
    bpy.context.scene.camera = bpy.context.object


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handdrive-dae", required=True, type=Path)
    parser.add_argument("--stock-dae", required=True, type=Path)
    parser.add_argument("--texture-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)

    hand_imported = _import_collada(args.handdrive_dae)
    rhd = _keep_named(hand_imported, TARGET_NAMES)
    _delete_objects([obj for obj in hand_imported if obj not in rhd])

    stock_imported = _import_collada(args.stock_dae)
    stock = _keep_named(stock_imported, STOCK_NAMES)
    _delete_objects([obj for obj in stock_imported if obj not in stock])

    rhd_mat = _build_rhd_material(args.texture_dir)
    for obj in rhd:
        obj.data.materials.clear()
        obj.data.materials.append(rhd_mat)
        obj.name = obj.name.split(".", 1)[0] + "_generated_rhd"

    ref_mat = _build_reference_material()
    for obj in stock:
        obj.data.materials.clear()
        obj.data.materials.append(ref_mat)
        obj.location.x += 2.0
        obj.name = obj.name.split(".", 1)[0] + "_stock_lhd_reference"

    _put_in_collection(rhd, "Generated RHD texture POC")
    _put_in_collection(stock, "Stock LHD reference, offset +2m X")
    _frame_scene(rhd + stock)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"saved {args.output}")
    print("generated objects:", ", ".join(obj.name for obj in rhd))
    print("reference objects:", ", ".join(obj.name for obj in stock))
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(arguments))
