"""Experimental rear-window sticker export (2026-09-08)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from beamxp import transform_helpers
from beamxp.core.constants import BUILD_CONVERTED, SOURCE_ROOT_DIR
from beamxp.core.sjson import decode, encode_beamng_json

# 2026-09-08: No rear-pane role exists in meshs.lua. Use the author's part
# description as a conservative fallback, then require glass and pane geometry.
REAR_GLASS = re.compile(r"backlight|rear[ _-]*(?:wind(?:screen|shield|ow)|glass)|tailgate[ _-]*glass", re.IGNORECASE)
WIDTH = 0.24
# Preferred inset from the pane's lower/right extents. Keep this separate from
# COVERAGE_MARGIN: placement aesthetics should not make valid panes fail the
# geometry-coverage test. 4 cm moves the sticker ~1 cm further up/inboard than
# the original prototype without making it look centrally placed.
EDGE_INSET = 0.04
# Glass required around the sticker footprint when validating a placement.
COVERAGE_MARGIN = 0.03
# 2026-09-08: Outward offset in metres; increase to move away from the glass.
# Re-export the vehicle after tuning this value.
LIFT = 0.001


def sticker_asset() -> Path:
    roots = [Path(getattr(sys, "_MEIPASS", SOURCE_ROOT_DIR)), SOURCE_ROOT_DIR]
    for root in roots:
        path = root / "assets" / "beamxp_imported_by_sticker.png"
        if path.is_file():
            return path
    raise FileNotFoundError("Bundled BeamXP imported-by sticker PNG is missing")


def write_sticker_textures(asset: Path, output_vehicle_dir: Path, charts=()):
    # 2026-09-08: BeamNG engine::_loadTextureBitmap rejected the original
    # 1500x430 maps in beamng.log: each axis must be a power of two, >=16.
    # The texture cooker documents RGB8 colour and L8 opacity PNGs:
    # https://documentation.beamng.com/modding/materials/texture_cooker/
    # Mesh dimensions still use the source aspect; full-range UVs compensate
    # for the independently resized axes. Derive both maps from one resize.
    with Image.open(asset) as source:
        size = np.array([max(16, 1 << (axis - 1).bit_length()) for axis in source.size])
        offset = np.zeros(2, dtype=int)
        atlas_size = size.copy()
        if len(charts):
            # 2026-09-08: Full glass triangles extend beyond the artwork.
            # Fit their UVs inside a transparent atlas (including an 8-texel
            # gutter) so texture wrapping cannot repeat the sticker. Cap the
            # atlas at 4096 per axis for unusually coarse source panes.
            points = np.concatenate([chart.reshape(-1, 2) for chart in charts])
            low = np.minimum(points.min(axis=0), 0)
            high = np.maximum(points.max(axis=0), 1)
            while True:
                origin = np.floor(low * size).astype(int) - 8
                extent = np.ceil(high * size).astype(int) + 8 - origin
                atlas_size = np.array([1 << (int(axis) - 1).bit_length() for axis in extent])
                if atlas_size.max() <= 4096:
                    break
                size = np.maximum(size // 2, 1)
            offset = -origin
        artwork = source.convert("RGBA")
        if artwork.size != tuple(size):
            artwork = artwork.resize(tuple(size), Image.Resampling.LANCZOS)
        atlas = Image.new("RGBA", tuple(atlas_size))
        atlas.paste(artwork, (int(offset[0]), int(atlas_size[1] - offset[1] - size[1])))
        atlas.convert("RGB").save(output_vehicle_dir / "beamxp_imported_by.color.png")
        atlas.getchannel("A").save(output_vehicle_dir / "beamxp_imported_by_opacity.data.png")
    return size / atlas_size, offset / atlas_size


def _clip(triangles, uv, bounds):
    result, coordinates = [], []
    lo, hi = uv.min(axis=1), uv.max(axis=1)
    intersects = (hi[:, 0] >= bounds[0]) & (lo[:, 0] <= bounds[1]) & (hi[:, 1] >= bounds[2]) & (lo[:, 1] <= bounds[3])
    triangles, uv = triangles[intersects], uv[intersects]
    for triangle, chart in zip(triangles, uv):
        poly = [np.concatenate((p, t)) for p, t in zip(triangle, chart)]
        for axis, edge, sign in ((3, bounds[0], 1), (3, bounds[1], -1),
                                 (4, bounds[2], 1), (4, bounds[3], -1)):
            clipped = []
            for a, b in zip(poly, poly[1:] + poly[:1]):
                da, db = sign * (a[axis] - edge), sign * (b[axis] - edge)
                if da >= 0:
                    clipped.append(a)
                if (da >= 0) != (db >= 0):
                    clipped.append(a + (b - a) * da / (da - db))
            poly = clipped
            if not poly:
                break
        for i in range(1, len(poly) - 1):
            points = np.array([poly[0], poly[i], poly[i + 1]])
            if np.linalg.norm(np.cross(points[1, :3] - points[0, :3], points[2, :3] - points[0, :3])) < 1e-12:
                continue
            result.append(points[:, :3])
            coordinates.append(points[:, 3:])
    return np.asarray(result).reshape(-1, 3, 3), np.asarray(coordinates).reshape(-1, 3, 2)


def _area(uv):
    a, b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2


def _lower_edge_direction(uv, left, right):
    """Fit the pane's lower silhouette over one sticker-width interval."""
    # 2026-09-08: Intersect the actual projected edges, not a convex hull:
    # a hull bridges concavities and can give the wrong lower-edge angle.
    start = uv.reshape(-1, 2)
    end = np.roll(uv, -1, axis=1).reshape(-1, 2)
    delta = end - start
    nonvertical = np.abs(delta[:, 0]) > 1e-10
    start, delta = start[nonvertical], delta[nonvertical]
    if not len(start):
        return None
    samples = np.linspace(left, right, 17)
    fraction = (samples[:, None] - start[None, :, 0]) / delta[None, :, 0]
    hits = (fraction >= 0) & (fraction <= 1)
    heights = np.where(hits, start[None, :, 1] + fraction * delta[None, :, 1], np.inf).min(axis=1)
    if not np.isfinite(heights).all():
        return None
    centred = samples - samples.mean()
    slope = np.dot(centred, heights - heights.mean()) / np.dot(centred, centred)
    direction = np.array([1., slope])
    return direction / np.linalg.norm(direction)


def place_sticker(triangles: np.ndarray, aspect: float):
    """Fit artwork to a rear pane and retain its supporting glass triangles."""
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    if not len(triangles) or not np.isfinite(triangles).all():
        return None
    points = triangles.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    if low[0] > -0.2 or high[0] < 0.2 or high[2] - low[2] < 0.12:
        return None
    # 2026-09-08: BeamNG vehicle right is -X, rear is +Y. Fit slope in YZ
    # to establish a pane chart; the local lower silhouette sets the angle.
    _, _, axes = np.linalg.svd(points[:, 1:] - points[:, 1:].mean(axis=0), full_matrices=False)
    up = np.array([0., *axes[0]])
    if up[2] < 0:
        up = -up
    right = np.array([-1., 0., 0.])
    normal = np.cross(right, up)
    if normal[1] < 0.25:
        return None
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    outward = crosses @ normal > lengths * 0.5
    triangles = triangles[outward & (lengths > 1e-10)]
    if not len(triangles):
        return None
    uv = np.stack((triangles @ right, triangles @ up), axis=-1)
    lo, hi = uv.reshape(-1, 2).min(axis=0), uv.reshape(-1, 2).max(axis=0)
    width, height = WIDTH, WIDTH / aspect
    if (hi[0] - lo[0] < width + 2 * COVERAGE_MARGIN
            or hi[1] - lo[1] < height + 2 * COVERAGE_MARGIN):
        return None
    # 2026-09-08: Prefer the lower-right quarter, slightly inset from the pane
    # extents. An expanded rectangle must be fully covered; holes/duplicate
    # layers fail. EDGE_INSET controls appearance, while COVERAGE_MARGIN is only
    # the geometric safety border around the candidate sticker.
    target_right, target_bottom = hi[0] - EDGE_INSET, lo[1] + EDGE_INSET
    # Always test the preferred inset once, even on unusually shallow/narrow
    # panes where EDGE_INSET itself lies beyond the nominal quarter boundary.
    bottom_stop = max(lo[1] + (hi[1] - lo[1]) * .25, target_bottom + .005)
    edge_stop = min(hi[0] - (hi[0] - lo[0]) * .25, target_right - .005)
    positions = [(edge, bottom)
                 for bottom in np.arange(target_bottom, bottom_stop, .005)
                 for edge in np.arange(target_right, edge_stop, -.005)]
    positions.sort(key=lambda p: (p[0] - target_right) ** 2 + (p[1] - target_bottom) ** 2)
    directions = {}
    for edge, bottom in positions:
        if edge not in directions:
            directions[edge] = _lower_edge_direction(uv, edge - width, edge)
        along = directions[edge]
        if along is None:
            continue
        rotation = np.column_stack((along, [-along[1], along[0]]))
        centre = np.array([edge - width / 2, bottom + height / 2])
        aligned_uv = (uv - centre) @ rotation
        bounds = (-width / 2, width / 2, -height / 2, height / 2)
        expanded = (bounds[0] - COVERAGE_MARGIN, bounds[1] + COVERAGE_MARGIN,
                    bounds[2] - COVERAGE_MARGIN, bounds[3] + COVERAGE_MARGIN)
        _, coverage = _clip(triangles, aligned_uv, expanded)
        expected = ((width + 2 * COVERAGE_MARGIN)
                    * (height + 2 * COVERAGE_MARGIN))
        if abs(_area(coverage) - expected) > expected * 1e-5:
            continue
        _, chart = _clip(triangles, aligned_uv, bounds)
        if abs(_area(chart) - width * height) > width * height * 1e-5:
            continue
        # 2026-09-08: BeamNG maps each flexbody vertex independently to nodes.
        # Clipping introduces vertices with different locators from the glass;
        # sharing groups alone does not keep those surfaces together in game.
        # Preserve the source triangles; a padded opacity atlas cuts the outline.
        # https://documentation.beamng.com/modding/vehicle/vehicle-art/modeling/
        # https://documentation.beamng.com/modding/vehicle/sections/flexbodies/
        low, high = aligned_uv.min(axis=1), aligned_uv.max(axis=1)
        supporting = ((high[:, 0] >= bounds[0]) & (low[:, 0] <= bounds[1])
                      & (high[:, 1] >= bounds[2]) & (low[:, 1] <= bounds[3]))
        mesh = triangles[supporting] + normal * LIFT
        chart = (aligned_uv[supporting] - [bounds[0], bounds[2]]) / [width, height]
        return mesh, chart
    return None


def write_sticker_dae(path: Path, name: str, material: str, triangles, uv):
    ns = "http://www.collada.org/2005/11/COLLADASchema"

    def sub(parent, tag, text=None, **attrs):
        element = ET.SubElement(parent, f"{{{ns}}}{tag}", attrs)
        element.text = text
        return element

    root = ET.Element(f"{{{ns}}}COLLADA", version="1.4.1")
    asset = sub(root, "asset")
    sub(asset, "unit", name="meter", meter="1")
    sub(asset, "up_axis", "Z_UP")
    effects = sub(root, "library_effects")
    effect = sub(effects, "effect", id=f"{material}-fx")
    technique = sub(sub(effect, "profile_COMMON"), "technique", sid="common")
    sub(sub(sub(technique, "lambert"), "diffuse"), "color", "1 1 1 1")
    mat = sub(sub(root, "library_materials"), "material", id=material, name=material)
    sub(mat, "instance_effect", url=f"#{material}-fx")
    mesh = sub(sub(sub(root, "library_geometries"), "geometry", id=f"{name}-geo"), "mesh")
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normals = np.repeat(normals[:, None, :], 3, axis=1)
    for suffix, values, params in (("pos", triangles, "XYZ"), ("normal", normals, "XYZ"), ("uv", uv, "ST")):
        source_id = f"{name}-{suffix}"
        source = sub(mesh, "source", id=source_id)
        sub(source, "float_array", " ".join(f"{v:.9g}" for v in values.flat),
            id=f"{source_id}-array", count=str(values.size))
        accessor = sub(sub(source, "technique_common"), "accessor", source=f"#{source_id}-array",
                       count=str(values.size // len(params)), stride=str(len(params)))
        for param in params:
            sub(accessor, "param", name=param, type="float")
    sub(sub(mesh, "vertices", id=f"{name}-verts"), "input", semantic="POSITION", source=f"#{name}-pos")
    tris = sub(mesh, "triangles", count=str(len(triangles)), material=material)
    for semantic, suffix in (("VERTEX", "verts"), ("NORMAL", "normal"), ("TEXCOORD", "uv")):
        sub(tris, "input", semantic=semantic, source=f"#{name}-{suffix}", offset="0", **({"set": "0"} if semantic == "TEXCOORD" else {}))
    sub(tris, "p", " ".join(map(str, range(len(triangles) * 3))))
    scene = sub(sub(root, "library_visual_scenes"), "visual_scene", id="Scene")
    node = sub(scene, "node", id=name, name=name, type="NODE")
    sub(node, "matrix", "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")
    instance = sub(node, "instance_geometry", url=f"#{name}-geo")
    binding = sub(sub(sub(instance, "bind_material"), "technique_common"), "instance_material", symbol=material, target=f"#{material}")
    sub(binding, "bind_vertex_input", semantic="UVMap", input_semantic="TEXCOORD", input_set="0")
    sub(sub(root, "scene"), "instance_visual_scene", url="#Scene")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def apply_to_build(context, conversion, output_vehicle_dir, output_plans):
    from beamxp import hand_drive_core as core

    summary = {"enabled": conversion.get("rearWindowSticker") is True, "placed": [], "skipped": []}
    if not summary["enabled"]:
        return summary
    asset = sticker_asset()
    with Image.open(asset) as artwork:
        aspect = artwork.width / artwork.height
    texts = {str(p.relative_to(output_vehicle_dir)): p.read_text(encoding="utf-8")
             for p in sorted(output_vehicle_dir.rglob("*.jbeam"))}
    index = dict(context.part_body_index)
    index.update(core.build_part_body_index(texts))
    symbols = core.mesh_material_symbols(context)
    flags = core.material_flags_for_context(context)
    surfaces = {}
    clones = {}
    meshes = {}
    material = f"beamxp_{core.safe_id(context.vehicle_id)}_imported_by"
    for plan in output_plans:
        if plan["kind"] != BUILD_CONVERTED:
            continue
        config = str(plan["output"])
        pc_path = output_vehicle_dir / f"{config}.pc"
        pc = core.load_beamng_json_file(pc_path)
        reason = "No supported rear-glass pane fitted"
        if pc.get("partsTree"):
            summary["skipped"].append({"config": config, "reason": "Hierarchical partsTree is not supported by experimental sticker placement"})
            continue
        selected = core.resolve_selected_parts(pc, context.jbeam_texts, vehicle_id=context.source_vehicle_id, part_body_index=index)
        populated_groups = core.node_groups_for_selection(
            selected, lambda part, section: transform_helpers.extract_named_array(index.get(part, ("", ""))[0], section))
        candidates = []
        for instance in selected["part_instances"]:
            part = instance["part_id"]
            found = index.get(part)
            if not found:
                continue
            body = found[0]
            if not REAR_GLASS.search(core.part_information_name(body) or part):
                continue
            array = transform_helpers.extract_named_array(body, "flexbodies")
            for row in core.iter_jbeam_table_rows(array or "[]"):
                mesh = str(row.get("mesh", ""))
                if mesh not in context.objects or not row.get("[group]:"):
                    continue
                if not set(core.jbeam_group_names(row["[group]:"])) & populated_groups:
                    continue
                if not any(flags.get(s.lower().removesuffix("-material"), {}).get("glass") for s in symbols.get(mesh, ())):
                    continue
                # 2026-09-08: Keep the experimental path to authored-in-place
                # panes; variable/slot transforms need instance-aware baking.
                row_text = json.dumps(row)
                if any(key in row for key in ("pos", "rot", "scale", "nodeOffset", "nodeMove", "nodeRotate")) or any(core.NODE_TRANSFORM_KEY_RE.search(o) for o in instance["inherited_options"]) or "$" in row_text:
                    reason = "Rear pane has transforms or variables requiring manual placement"
                    continue
                obj = context.objects[mesh]
                key = (obj.dae_source_zip or context.source_zip, obj.dae_path)
                if key not in surfaces:
                    surfaces[key] = core.surface_triangles_from_tree(core.parse_dae(*key))
                triangles = surfaces[key].get(mesh)
                if triangles is None:
                    continue
                placement = place_sticker(triangles, aspect)
                if placement is not None:
                    candidates.append((instance, body, row, placement))
                else:
                    reason = "Rear pane has no clear lower-right area large enough for the sticker"
        if len(candidates) != 1:
            summary["skipped"].append({"config": config, "reason": "Ambiguous rear panes" if candidates else reason})
            continue
        instance, body, row, (triangles, uv) = candidates[0]
        digest = hashlib.sha256((body + str(row["mesh"])).encode()).hexdigest()[:12]
        name = f"{material}_{digest}"
        clone_id = f"{instance['part_id']}_beamxp_sticker_{digest}"
        if clone_id not in clones:
            # 2026-09-08: meshs.lua/processFlexbodies binds the sticker to the
            # glass groups; materials.lua/process maps its deformGroup to damage.
            options = {"deformGroup": row.get("deformGroup", ""),
                       "deformMaterialBase": material, "deformMaterialDamaged": material + "_gone",
                       "deformSound": "", "deformVolume": 0, "materialOverride": []}
            array = transform_helpers.extract_named_array(body, "flexbodies")
            header_start = array.index("[", 1)
            header_end = transform_helpers.find_matching(array, header_start, "[", "]")
            header = decode(array[header_start:header_end])
            values = {**row, **options, "mesh": name, "nonFlexMaterials": []}
            # 2026-09-08: tableSchema.lua/processJbeamTableRow only merges
            # inline options AFTER all header columns. A dict in column 2
            # becomes [group]: itself; links.lua/resolveGroupLinks then binds
            # the wrong nodes. The in-game probe confirmed the sticker was
            # driven by body nodes and sank into the correctly placed glass.
            sticker_row = [values.get(column, "") for column in header]
            sticker_row.append({key: value for key, value in values.items() if key not in header})
            updated_array = array[:header_end] + ",\n" + json.dumps(sticker_row) + array[header_end:]
            cloned = transform_helpers.replace_array_region(body, "flexbodies", lambda _, value=updated_array: value)
            cloned = re.sub(r'^"(?:[^"\\]|\\.)*"', json.dumps(clone_id), cloned, count=1)
            info = transform_helpers.extract_keyed_object(cloned, "information")
            if info:
                label = core.part_information_name(body) or instance["part_id"]
                renamed_info = re.sub(r'("name"\s*:\s*)"(?:[^"\\]|\\.)*"',
                                      lambda match, value=label: match[1] + json.dumps(value + " (BeamXP sticker)"), info, count=1)
                cloned = cloned.replace(info, renamed_info, 1)
            clones[clone_id] = cloned
            meshes[name] = (triangles, uv)
        if instance["slot_id"] == "main":
            pc["mainPartName"] = clone_id
        else:
            parts = pc.setdefault("parts", {})
            # 2026-09-08: resolve_selected_parts (and slotSystem.fillSlots_rec)
            # gives flat slot selections precedence over path selections.
            parts[str(instance["slot_id"])] = clone_id
        pc_path.write_text(encode_beamng_json(pc, indent=2) + "\n", encoding="utf-8")
        summary["placed"].append({"config": config, "sourceMesh": row["mesh"], "mesh": name, "part": clone_id, "widthMetres": WIDTH})
    if clones:
        (output_vehicle_dir / "beamxp_stickers.jbeam").write_text("{\n" + ",\n".join(clones.values()) + "\n}\n", encoding="utf-8")
        scale, offset = write_sticker_textures(asset, output_vehicle_dir, [uv for _, uv in meshes.values()])
        for name, (triangles, uv) in meshes.items():
            write_sticker_dae(output_vehicle_dir / f"{name}.dae", name, material, triangles, uv * scale + offset)
        # 2026-09-08: The sticker is an opaque cutout, not genuinely
        # translucent. Blended translucency lets the rear-glass render pass
        # composite over the sticker even when the sticker mesh is physically
        # in front and translucentZWrite is enabled. Alpha testing keeps normal
        # depth ordering and matches the nearly binary opacity mask.
        base = {"name": material, "mapTo": material, "class": "Material", "version": 1.5,
                "Stages": [{"colorMap": f"/{context.vehicle_path}/beamxp_imported_by.color.png",
                            "opacityMap": f"/{context.vehicle_path}/beamxp_imported_by_opacity.data.png",
                            "roughnessFactor": .65, "metallicFactor": 0}],
                "alphaTest": True, "alphaRef": 128, "doubleSided": True,
                "castShadows": False, "translucent": False}
        hidden = {"name": material + "_gone", "mapTo": material + "_gone", "class": "Material", "version": 1.5,
                  "Stages": [{"baseColorFactor": [1, 1, 1, 0], "opacityFactor": 0}],
                  "castShadows": False, "translucent": True, "translucentBlendOp": "LerpAlpha"}
        (output_vehicle_dir / "beamxp_stickers.materials.json").write_text(encode_beamng_json({material: base, material + "_gone": hidden}, indent=2) + "\n", encoding="utf-8")
    return summary