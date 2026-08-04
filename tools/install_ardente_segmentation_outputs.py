from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import zipfile
import math
from pathlib import Path
from xml.etree import ElementTree as ET


COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": COLLADA_NS}
ET.register_namespace("", COLLADA_NS)

PROJECT = Path(
    r"C:\Users\ashle\AppData\Local\BeamXP\handedness_conversion_projects"
    r"\vivace_vivace_ardente"
)
UNPACKED = PROJECT / "unpacked_output"
VEHICLE_ROOT = UNPACKED / "vehicles" / "vivace"
ARDENTE_ROOT = VEHICLE_ROOT / "ardente"
HANDDRIVE_DAE = ARDENTE_ROOT / "ardente_handdrive.dae"
HANDDRIVE_JBEAM = VEHICLE_ROOT / "jbeam" / "handdrive_visual_conversion.jbeam"
MATERIALS_JSON = ARDENTE_ROOT / "ardente_handdrive_texture_poc.materials.json"
BUILD_ZIP = PROJECT / "build" / "vivace_XP_conversion.zip"
INSTALL_ZIP = Path(
    r"C:\Users\ashle\AppData\Local\BeamNG\BeamNG.drive\current\mods"
    r"\vivace_XP_conversion.zip"
)
COLLADA_CACHE = Path(
    r"C:\Users\ashle\AppData\Local\BeamNG\BeamNG.drive\current\temp"
    r"\vehicles\vivace\ardente\ardente_handdrive.cdae"
)

OUTPUT_ROOT = Path("mesh_segmentation_transform") / "segmentation_outputs"
PARTS = {
    "dash": {
        "source_dir": OUTPUT_ROOT / "ardente_dash",
        "source_dae": "ardente_dashboard_rhd.dae",
        "node_prefix": "ardente_dashboard__beamxp_",
        "old_mesh": "ardente_dashboard_xp_rhd",
        "groups": '["ardente_dash"]',
        "material": "ardente_interior_xp_rhd_dash",
    },
    "console": {
        "source_dir": OUTPUT_ROOT / "ardente_console",
        "source_dae": "ardente_console_rhd.dae",
        "node_prefix": "ardente_console__beamxp_",
        "old_mesh": "ardente_console_xp_rhd",
        "groups": '["ardente_dash", "ardente_floor"]',
        "material": "ardente_interior_xp_rhd_console",
    },
}

TEXTURE_NAMES = [
    "ardente_interior_ao.data_rhd.dds",
    "ardente_interior_b.color_rhd.dds",
    "ardente_interior_carpet_o.data_rhd.dds",
    "ardente_interior_headliner_o.data_rhd.dds",
    "ardente_interior_leather_o.data_rhd.dds",
    "ardente_interior_m.data_rhd.dds",
    "ardente_interior_nm.normal_rhd.dds",
    "ardente_interior_r.data_rhd.dds",
]


def q(tag: str) -> str:
    return f"{{{COLLADA_NS}}}{tag}"


def remove_if(parent: ET.Element, predicate) -> int:
    removed = 0
    for child in list(parent):
        if predicate(child):
            parent.remove(child)
            removed += 1
    return removed


def materialise_geometry(root: ET.Element, material_name: str) -> None:
    for primitive in root.findall(".//*[@material]", NS):
        if primitive.get("material") == "ardente_interior-material":
            primitive.set("material", material_name)
    for binding in root.findall(".//c:instance_material", NS):
        if binding.get("symbol") == "ardente_interior-material":
            binding.set("symbol", material_name)
            binding.set("target", f"#{material_name}-material")


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def parse_matrix(node: ET.Element) -> list[list[float]]:
    matrix = node.find("c:matrix", NS)
    if matrix is None or not matrix.text:
        return identity_matrix()
    values = [float(value) for value in matrix.text.split()]
    if len(values) != 16:
        return identity_matrix()
    return [
        values[0:4],
        values[4:8],
        values[8:12],
        values[12:16],
    ]


def matrix_multiply(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    x, y, z = point
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def transform_direction(matrix: list[list[float]], direction: list[float]) -> list[float]:
    x, y, z = direction
    transformed = [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    ]
    length = math.sqrt(sum(value * value for value in transformed))
    if length > 0:
        return [value / length for value in transformed]
    return transformed


def parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    parents: dict[ET.Element, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parents[child] = parent
    return parents


def world_matrix(node: ET.Element, parents: dict[ET.Element, ET.Element]) -> list[list[float]]:
    chain: list[ET.Element] = []
    current: ET.Element | None = node
    while current is not None:
        if current.tag == q("node"):
            chain.append(current)
        current = parents.get(current)
    matrix = identity_matrix()
    for item in reversed(chain):
        matrix = matrix_multiply(matrix, parse_matrix(item))
    return matrix


def format_floats(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def bake_geometry_transform(geometry: ET.Element, matrix: list[list[float]]) -> None:
    for source in geometry.findall(".//c:source", NS):
        source_id = source.get("id", "").lower()
        if not (source_id.endswith("-positions") or source_id.endswith("-normals")):
            continue
        array = source.find("c:float_array", NS)
        if array is None or not array.text:
            continue
        values = [float(value) for value in array.text.split()]
        baked: list[float] = []
        transform = (
            transform_direction if source_id.endswith("-normals") else transform_point
        )
        for index in range(0, len(values), 3):
            baked.extend(transform(matrix, values[index : index + 3]))
        array.text = format_floats(baked)


def set_identity_node_transform(node: ET.Element) -> None:
    for child in list(node):
        if child.tag in {q("matrix"), q("translate"), q("rotate"), q("scale")}:
            node.remove(child)
    matrix = ET.Element(q("matrix"))
    matrix.text = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    node.insert(0, matrix)


def source_nodes_and_geometry(path: Path, node_prefix: str, material_name: str):
    source_root = ET.parse(path).getroot()
    materialise_geometry(source_root, material_name)
    parents = parent_map(source_root)

    source_geometries = {
        geometry.get("id"): geometry
        for geometry in source_root.findall(".//c:geometry", NS)
        if geometry.get("id")
    }
    nodes: list[ET.Element] = []
    geometries: list[ET.Element] = []
    seen_geometries: set[str] = set()
    for source_node in source_root.findall(".//c:node", NS):
        if not (source_node.get("id") or "").startswith(node_prefix):
            continue
        node_matrix = world_matrix(source_node, parents)
        node = copy.deepcopy(source_node)
        set_identity_node_transform(node)
        nodes.append(node)

        for instance in source_node.findall(".//c:instance_geometry", NS):
            geometry_id = (instance.get("url") or "").lstrip("#")
            if not geometry_id or geometry_id in seen_geometries:
                continue
            geometry = copy.deepcopy(source_geometries[geometry_id])
            bake_geometry_transform(geometry, node_matrix)
            geometries.append(geometry)
            seen_geometries.add(geometry_id)
    return nodes, geometries


def patch_dae() -> dict[str, list[str]]:
    tree = ET.parse(HANDDRIVE_DAE)
    root = tree.getroot()
    library_geometries = root.find(".//c:library_geometries", NS)
    visual_scene = root.find(".//c:library_visual_scenes/c:visual_scene", NS)
    if library_geometries is None or visual_scene is None:
        raise RuntimeError("handdrive DAE is missing library_geometries or visual_scene")

    inserted: dict[str, list[str]] = {}
    for key, part in PARTS.items():
        prefix = part["node_prefix"]
        material = part["material"]
        nodes, geometries = source_nodes_and_geometry(
            Path(part["source_dir"]) / part["source_dae"], prefix, material
        )
        if not nodes or not geometries:
            raise RuntimeError(f"no segmentation nodes/geometries found for {key}")

        remove_if(
            visual_scene,
            lambda child, p=prefix: (child.get("id") or "").startswith(p),
        )
        remove_if(
            library_geometries,
            lambda child, ids={g.get("id") for g in geometries}: child.get("id") in ids,
        )
        for geometry in geometries:
            library_geometries.append(geometry)
        for node in nodes:
            visual_scene.append(node)
        inserted[key] = [node.get("id", "") for node in nodes]

    tree.write(HANDDRIVE_DAE, encoding="utf-8", xml_declaration=True)
    return inserted


def flexbody_rows(meshes: list[str], groups: str) -> str:
    return "\n".join(f'         ["{mesh}", {groups}],' for mesh in meshes)


def patch_jbeam(inserted: dict[str, list[str]]) -> None:
    text = HANDDRIVE_JBEAM.read_text(encoding="utf-8")
    for key, part in PARTS.items():
        old_mesh = part["old_mesh"]
        groups = part["groups"]
        replacement = flexbody_rows(inserted[key], groups)
        prefixes = [
            f'         ["{old_mesh}", {groups}],',
            (
                f'         ["{old_mesh}", {groups}, [], '
                '{"materialOverride":[["ardente_interior_xp_rhd", "ardente_interior_xp_rhd"]]}],'
            ),
        ]
        replaced = 0
        for old in prefixes:
            count = text.count(old)
            if count:
                text = text.replace(old, replacement)
                replaced += count
        if replaced == 0:
            existing = sum(text.count(f'["{mesh}", {groups}],') for mesh in inserted[key])
            if existing:
                print(
                    f"kept {existing} existing split JBeam row(s) for {old_mesh}"
                )
                continue
            raise RuntimeError(f"did not find JBeam flexbody row for {old_mesh}")
        print(
            f"replaced {replaced} JBeam row(s) for {old_mesh} "
            f"with {len(inserted[key])} rows"
        )
    HANDDRIVE_JBEAM.write_text(text, encoding="utf-8")


def part_texture_name(original: str, suffix: str) -> str:
    path = Path(original)
    return f"{path.stem}_{suffix}{path.suffix}"


def copy_textures() -> dict[str, dict[str, str]]:
    copied: dict[str, dict[str, str]] = {}
    for key, part in PARTS.items():
        copied[key] = {}
        source_dir = Path(part["source_dir"])
        for name in TEXTURE_NAMES:
            src = source_dir / name
            dst_name = part_texture_name(name, key)
            dst = ARDENTE_ROOT / dst_name
            shutil.copy2(src, dst)
            copied[key][name] = dst_name
    return copied


def update_stage_paths(mat: dict, replacements: dict[str, str]) -> None:
    for stage in mat.get("Stages", []):
        for field, value in list(stage.items()):
            if not isinstance(value, str):
                continue
            filename = Path(value).name
            if filename in replacements:
                stage[field] = f"/vehicles/vivace/ardente/{replacements[filename]}"


def patch_materials(copied: dict[str, dict[str, str]]) -> None:
    data = json.loads(MATERIALS_JSON.read_text(encoding="utf-8"))
    template = data.get("ardente_interior_xp_rhd") or next(iter(data.values()))
    for key, part in PARTS.items():
        name = part["material"]
        mat = copy.deepcopy(template)
        mat["name"] = name
        mat["mapTo"] = name
        update_stage_paths(mat, copied[key])
        data[name] = mat
        # Keep skin variants available in case BeamNG switches these custom slots.
        for skin in ("interior_black", "interior_red", "interior_tan"):
            skin_name = f"{name}.skin_interior_ardente.{skin}"
            skin_mat = copy.deepcopy(mat)
            skin_mat["name"] = skin_name
            skin_mat["mapTo"] = skin_name
            data[skin_name] = skin_mat
    MATERIALS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def repack() -> None:
    tmp = BUILD_ZIP.with_suffix(".zip.tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(UNPACKED.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(UNPACKED).as_posix())
    os.replace(tmp, BUILD_ZIP)

    install_tmp = INSTALL_ZIP.with_suffix(".zip.tmp")
    if install_tmp.exists():
        install_tmp.unlink()
    shutil.copy2(BUILD_ZIP, install_tmp)
    os.replace(install_tmp, INSTALL_ZIP)
    if COLLADA_CACHE.exists():
        COLLADA_CACHE.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-repack", action="store_true")
    args = parser.parse_args()

    inserted = patch_dae()
    patch_jbeam(inserted)
    copied = copy_textures()
    patch_materials(copied)
    if not args.no_repack:
        repack()
    for key, meshes in inserted.items():
        print(f"{key}: {len(meshes)} mesh node(s)")
    print(f"updated {HANDDRIVE_DAE}")
    print(f"updated {HANDDRIVE_JBEAM}")
    print(f"updated {MATERIALS_JSON}")
    if not args.no_repack:
        print(f"installed {INSTALL_ZIP}")
        print(f"cache exists: {COLLADA_CACHE.exists()}")


if __name__ == "__main__":
    main()
