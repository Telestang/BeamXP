"""Build a combined Blender inspection file for texture-correction artifacts.

Run with Blender:

    blender --background --python tools/build_texture_correction_inspection_blend.py -- \
        --job-dir PATH_TO_TEXTURE_CORRECTION_JOB --save output.blend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def normalise_material_name(name: str) -> str:
    name = name.strip().lower()
    if len(name) > 4 and name[-4] == "." and name[-3:].isdigit():
        name = name[:-4]
    for suffix in ("-material", "_material"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def image_for(job_dir: Path, relative_path: str, colour_space: str):
    image = bpy.data.images.load(str(job_dir / relative_path), check_existing=True)
    image.colorspace_settings.name = colour_space
    return image


def wire_material(job_dir: Path, material, maps: dict[str, str]) -> list[str]:
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (260, 0)
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    def texture(path: str, colour_space: str, y: int):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image_for(job_dir, path, colour_space)
        node.location = (-420, y)
        return node

    wired: list[str] = []
    base = maps.get("baseColorMap")
    if base:
        colour = texture(base, "sRGB", 300)
        source = colour.outputs["Color"]
        occlusion = maps.get("ambientOcclusionMap")
        if occlusion:
            ao = texture(occlusion, "Non-Color", 60)
            mix = tree.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.location = (-120, 220)
            tree.links.new(colour.outputs["Color"], mix.inputs[6])
            tree.links.new(ao.outputs["Color"], mix.inputs[7])
            source = mix.outputs[2]
            wired.append("ambientOcclusionMap")
        tree.links.new(source, shader.inputs["Base Color"])
        wired.append("baseColorMap")

    normal = maps.get("normalMap")
    if normal:
        node = texture(normal, "Non-Color", -460)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.location = (-120, -460)
        tree.links.new(node.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
        wired.append("normalMap")

    for key, socket, y in (
        ("roughnessMap", "Roughness", -160),
        ("metallicMap", "Metallic", -310),
    ):
        path = maps.get(key)
        if path:
            node = texture(path, "Non-Color", y)
            tree.links.new(node.outputs["Color"], shader.inputs[socket])
            wired.append(key)
    return wired


def parse_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir.resolve()
    manifest_path = job_dir / "rhd_materials.json"
    dae_paths = sorted(job_dir.glob("*_rhd.dae"), key=lambda path: path.name.lower())
    if not dae_paths:
        raise SystemExit(f"No *_rhd.dae files found in {job_dir}")
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    for dae_path in dae_paths:
        print(f"importing {dae_path.name}")
        bpy.ops.wm.collada_import(filepath=str(dae_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_alias: dict[str, dict[str, object]] = {}
    for entry in manifest.get("materials", []):
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases", []):
            by_alias.setdefault(normalise_material_name(str(alias)), entry)

    matched = 0
    for material in bpy.data.materials:
        entry = by_alias.get(normalise_material_name(material.name))
        if not isinstance(entry, dict):
            continue
        maps = entry.get("maps")
        if not isinstance(maps, dict):
            continue
        matched += 1
        wired = wire_material(job_dir, material, {str(k): str(v) for k, v in maps.items()})
        print(f"wired {material.name} -> {', '.join(wired)}")
    print(f"{matched} of {len(bpy.data.materials)} material(s) matched corrected maps")

    bpy.ops.wm.save_as_mainfile(filepath=str(args.save.resolve()))
    print(f"saved {args.save}")


main()
