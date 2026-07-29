"""COLLADA (.dae) loading and parsing for the hand-drive tool.

The read-only DAE layer, split out of ``beamng_hand_drive_core`` so that mesh
discovery and geometry extraction can be read, tested and profiled without
pulling in the jbeam resolver, the build pipeline or the GUI.

Three groups of entry points:

  - Object discovery -- :func:`parse_dae`, :func:`dae_objects_from_tree` and the
    ``list_dae_objects_for_*`` wrappers turn a DAE into ``{alias: DaeObject}``,
    keyed by every id/name a jbeam row might reference.
  - Geometry extraction -- :func:`preview_data_from_tree` (sampled point
    clouds, bounds, material symbols) and :func:`surface_triangles_from_tree`
    (filled triangles) both work from an already-parsed tree, so a caller that
    needs several payloads parses the file once.  A stock common DAE is tens of
    megabytes of XML.
  - Shared-mesh lookup -- :func:`common_dae_paths` and
    :func:`load_common_dae_objects` resolve meshes that a vehicle references but
    does not ship, scanning ``vehicles/common/`` inside candidate zips.

Everything here is side-effect free and takes no ``VehicleContext``.  The
context-aware caching layer that sits on top of these functions
(``dae_source_index``, ``full_surface_triangles_for_ids``,
``full_vertex_clouds_for_ids`` and the ``*_for_resolved_placement`` rebuilders)
stays in ``beamng_hand_drive_core``, because it reads and mutates per-context
caches and needs ``ResolvedMeshPosition``.

Scale note: vertex and pivot coordinates returned from this module are already
multiplied by the asset's COLLADA ``<unit meter>`` factor, so callers always
receive metres.  See :func:`dae_unit_scale`.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np

from beamxp import transform_helpers

NS = transform_helpers.NS



@dataclass(frozen=True)
class DaeObject:
    id: str
    name: str
    dae_path: str
    x: float
    y: float
    z: float
    geometry_ids: tuple[str, ...]
    dae_source_zip: Path | None = None


def parse_dae(source_zip: Path, dae_path: str) -> ET.ElementTree:
    with zipfile.ZipFile(source_zip) as zf:
        with zf.open(dae_path) as fh:
            return ET.parse(fh)


def dae_unit_scale(root: ET.Element) -> float:
    """The COLLADA asset's <unit meter="..."> factor (metres per unit).

    Some stock geometry is authored in centimetres (e.g.
    common/tires/bolide_80s_tires.dae carries <unit meter="0.01"> with no scale
    in its node transforms), so its vertex coordinates are 100x too large unless
    this factor is applied -- which is what made the bolide's tyres render metres
    wide. DAEs authored in metres report meter="1" (a no-op), and DAEs that carry
    a scale in the node matrix instead (e.g. gavrilsteeringwheels.dae) report
    meter="1", so applying this on top of the node transform never double-scales.
    """
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "unit":
            try:
                value = float(elem.get("meter"))
            except (TypeError, ValueError):
                return 1.0
            return value if value > 0 else 1.0
        if tag == "library_geometries":
            break  # <unit> lives in <asset> near the top; stop once past it
    return 1.0


def dae_objects_from_tree(
    tree: ET.ElementTree,
    dae_path: str,
    *,
    dae_source_zip: Path | None = None,
) -> dict[str, DaeObject]:
    objects: dict[str, DaeObject] = {}
    unit = dae_unit_scale(tree.getroot())
    for node in tree.getroot().findall(".//c:node", NS):
        object_id = node.get("id")
        if not object_id:
            continue
        instance_geometries = node.findall(".//c:instance_geometry", NS)
        if not instance_geometries:
            continue
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = transform_helpers.parse_matrix(matrix_elem.text)
        geometry_ids = tuple(
            inst.get("url", "")[1:]
            for inst in instance_geometries
            if inst.get("url", "").startswith("#")
        )
        obj = DaeObject(
            id=object_id,
            name=(node.get("name") or object_id).strip(),
            dae_path=dae_path,
            x=matrix[0][3] * unit,
            y=matrix[1][3] * unit,
            z=matrix[2][3] * unit,
            geometry_ids=geometry_ids,
            dae_source_zip=dae_source_zip,
        )
        for alias in dae_node_aliases(node):
            objects.setdefault(alias, obj)
    return objects


def dae_node_aliases(node: ET.Element) -> list[str]:
    aliases: list[str] = []
    for value in (node.get("id"), node.get("name")):
        if not value:
            continue
        for alias in (value, value.strip()):
            if alias and alias not in aliases:
                aliases.append(alias)
    return aliases


def find_dae_node(root: ET.Element, object_id: str) -> ET.Element | None:
    node = root.find(f".//c:node[@id='{object_id}']", NS)
    if node is not None:
        return node
    for candidate in root.findall(".//c:node", NS):
        if object_id in dae_node_aliases(candidate):
            return candidate
    return None


def list_dae_objects_for_file(source_zip: Path, dae_path: str) -> dict[str, DaeObject]:
    tree = parse_dae(source_zip, dae_path)
    return dae_objects_from_tree(tree, dae_path, dae_source_zip=source_zip)


def list_dae_objects_for_path(path: Path) -> dict[str, DaeObject]:
    return dae_objects_from_tree(ET.parse(path), str(path), dae_source_zip=None)


def common_dae_paths(source_zip: Path) -> list[str]:
    try:
        with zipfile.ZipFile(source_zip) as zf:
            return sorted(
                name.replace("\\", "/")
                for name in zf.namelist()
                if name.replace("\\", "/").lower().startswith("vehicles/common/")
                and name.lower().endswith(".dae")
            )
    except Exception:
        return []


DAE_ALIAS_ATTR_RE = re.compile(rb'(?:id|name)="([^"]*)"')


def dae_alias_candidates(data: bytes) -> set[str]:
    """Every id/name attribute value in a raw DAE, plus stripped forms.

    A superset of what dae_node_aliases can key an object by, so using it to
    skip files is safe: it can only ever over-select. Matching on attributes
    rather than searching for each wanted mesh name matters a great deal --
    a combined ``mesh1|mesh2|...`` regex over a 680 MB common.zip is O(bytes
    x alternatives) in Python's re (no Aho-Corasick) and measured 211s on
    pickup, versus 0.4s for one attribute pass plus a set intersection.
    """
    names: set[str] = set()
    for raw in DAE_ALIAS_ATTR_RE.findall(data):
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        names.add(value)
        stripped = value.strip()
        if stripped:
            names.add(stripped)
    return names


def geometry_position_points(geometry: ET.Element) -> list[tuple[float, float, float]]:
    return transform_helpers.geometry_position_points(geometry)


def preview_data_for_file(
    source_zip: Path,
    dae_path: str,
    max_points_per_object: int = 350,
) -> dict[str, dict[str, object]]:
    return preview_data_from_tree(
        parse_dae(source_zip, dae_path),
        max_points_per_object=max_points_per_object,
    )


def preview_data_from_tree(
    tree: ET.ElementTree,
    max_points_per_object: int = 350,
) -> dict[str, dict[str, object]]:
    """Preview payload from an already-parsed DAE.

    Split out so callers that also need dae_objects_from_tree can parse the
    file once instead of once per helper; a common DAE is tens of MB of XML.
    """
    root = tree.getroot()
    library_geometries = root.find("c:library_geometries", NS)
    if library_geometries is None:
        return {}

    unit = dae_unit_scale(root)
    geometries_by_id = {
        geom.get("id"): geom
        for geom in library_geometries.findall("c:geometry", NS)
        if geom.get("id")
    }
    local_points_by_geometry = {
        geom_id: geometry_position_points(geom)
        for geom_id, geom in geometries_by_id.items()
    }

    preview: dict[str, dict[str, object]] = {}
    for node in root.findall(".//c:node", NS):
        object_id = node.get("id")
        if not object_id:
            continue
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = transform_helpers.parse_matrix(matrix_elem.text)

        object_points: list[tuple[float, float, float]] = []
        geometry_ids: list[str] = []
        for inst in node.findall(".//c:instance_geometry", NS):
            url = inst.get("url", "")
            if not url.startswith("#"):
                continue
            geometry_id = url[1:]
            geometry_ids.append(geometry_id)
            local_points = local_points_by_geometry.get(geometry_id, [])
            for point in local_points:
                wx, wy, wz = transform_helpers.transform_point(matrix, point)
                object_points.append((wx * unit, wy * unit, wz * unit))

        if not object_points:
            continue
        # Material bindings feed the spatial classifier (glass panes bound the
        # cabin; emissive surfaces are displays that need a texture flip).
        material_symbols = sorted({
            re.sub(r"-material$", "", symbol).lower()
            for inst_mat in node.findall(".//c:instance_material", NS)
            for symbol in (inst_mat.get("symbol") or inst_mat.get("target", "").lstrip("#"),)
            if symbol
        })
        bounds = transform_helpers.bounds_from_points(object_points)
        min_point, max_point = bounds
        center = (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        )
        item = {
            "bounds": bounds,
            "center": center,
            "sample_points": transform_helpers.sample_points(object_points, max_points_per_object),
            "geometry_ids": geometry_ids,
            "materials": tuple(material_symbols),
        }
        for alias in dae_node_aliases(node):
            preview.setdefault(alias, item)
    return preview


def surface_triangles_from_tree(
    tree: ET.ElementTree,
) -> dict[str, np.ndarray]:
    """Filled triangle surfaces for each object in an already-parsed DAE."""
    root = tree.getroot()
    library_geometries = root.find("c:library_geometries", NS)
    if library_geometries is None:
        return {}

    unit = dae_unit_scale(root)
    local_by_geometry: dict[str, np.ndarray] = {}
    for geometry in library_geometries.findall("c:geometry", NS):
        geometry_id = geometry.get("id")
        if not geometry_id:
            continue
        triangles = transform_helpers.geometry_surface_triangles(geometry)
        local_by_geometry[geometry_id] = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))

    surfaces: dict[str, np.ndarray] = {}
    for node in root.findall(".//c:node", NS):
        matrix_elem = node.find("c:matrix", NS)
        if matrix_elem is None or not matrix_elem.text:
            continue
        matrix = np.asarray(transform_helpers.parse_matrix(matrix_elem.text), dtype=float)
        chunks: list[np.ndarray] = []
        for instance in node.findall(".//c:instance_geometry", NS):
            url = instance.get("url", "")
            if not url.startswith("#"):
                continue
            local = local_by_geometry.get(url[1:])
            if local is None or len(local) == 0:
                continue
            flat = local.reshape((-1, 3))
            homogeneous = np.concatenate(
                [flat, np.ones((len(flat), 1), dtype=float)], axis=1
            )
            chunks.append(((homogeneous @ matrix.T)[:, :3] * unit).reshape((-1, 3, 3)))
        if not chunks:
            continue
        triangles = np.concatenate(chunks)
        for alias in dae_node_aliases(node):
            surfaces.setdefault(alias, triangles)
    return surfaces


def load_common_dae_objects(
    zip_candidates: Iterable[Path],
    wanted_meshes: set[str],
    existing_objects: dict[str, DaeObject],
) -> tuple[dict[str, DaeObject], dict[str, dict[str, object]], list[str]]:
    """Find meshes a vehicle references but does not ship.

    ``zip_candidates`` is the ordered search path -- the vehicle zip first,
    then any sibling and game ``common.zip``. It is passed in rather than
    derived here so this module stays independent of BeamNG install discovery.

    Returns the objects found, their preview payloads, and the DAE paths hit.
    """
    still_missing = wanted_meshes - set(existing_objects)
    if not still_missing:
        return {}, {}, []

    found_objects: dict[str, DaeObject] = {}
    found_previews: dict[str, dict[str, object]] = {}
    found_paths: set[str] = set()

    for candidate_zip in zip_candidates:
        if not still_missing:
            break
        paths = common_dae_paths(candidate_zip)
        if not paths:
            continue
        try:
            zf = zipfile.ZipFile(candidate_zip)
        except Exception:
            continue
        # One handle for the whole zip: reopening per DAE re-reads the
        # central directory of a multi-hundred-MB archive every time.
        with zf:
            for dae_path in paths:
                if not still_missing:
                    break
                try:
                    data = zf.read(dae_path)
                except Exception:
                    continue
                if still_missing.isdisjoint(dae_alias_candidates(data)):
                    continue

                try:
                    tree = ET.ElementTree(ET.fromstring(data))
                    file_objects = dae_objects_from_tree(
                        tree, dae_path, dae_source_zip=candidate_zip
                    )
                except Exception:
                    continue
                matched_ids = sorted(
                    object_id for object_id in file_objects if object_id in still_missing
                )
                if not matched_ids:
                    continue

                try:
                    file_previews = preview_data_from_tree(tree)
                except Exception:
                    file_previews = {}
                for object_id in matched_ids:
                    found_objects.setdefault(object_id, file_objects[object_id])
                    if object_id in file_previews:
                        found_previews.setdefault(object_id, file_previews[object_id])
                    still_missing.discard(object_id)
                found_paths.add(dae_path)

    return found_objects, found_previews, sorted(found_paths)
