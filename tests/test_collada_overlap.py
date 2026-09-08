from __future__ import annotations

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from beamxp import hand_drive_core as core
from beamxp.core.collada_overlap import duplicate_uv_keys, mesh_faces
from beamxp.core.dae import list_dae_objects_for_file
from beamxp.hand_drive_parts import build_pipeline

NS = core.NS
Q = "{http://www.collada.org/2005/11/COLLADASchema}"


def fixture() -> ET.Element:
    root = ET.Element(Q + "COLLADA")
    library = ET.SubElement(root, Q + "library_geometries")
    scenes = ET.SubElement(root, Q + "library_visual_scenes")
    scene = ET.SubElement(scenes, Q + "visual_scene", id="Scene")
    for name, offset in (("road", 3), ("race", 6)):
        geometry = ET.SubElement(library, Q + "geometry", id=name + "_geometry")
        mesh = ET.SubElement(geometry, Q + "mesh")
        positions = [0, 0, 0, 1, 0, 0, 0, 1, 0,
                     offset, 0, 0, offset + 1, 0, 0, offset, 1, 0]
        uv = [0, 0, .2, 0, 0, .2, .5, .5, .7, .5, .5, .7]
        for semantic, values, stride in (("pos", positions, 3), ("uv", uv, 2)):
            source_id = name + "_" + semantic
            source = ET.SubElement(mesh, Q + "source", id=source_id)
            array = ET.SubElement(source, Q + "float_array", id=source_id + "_array", count=str(len(values)))
            array.text = " ".join(map(str, values))
            technique = ET.SubElement(source, Q + "technique_common")
            ET.SubElement(technique, Q + "accessor", source="#" + source_id + "_array",
                          count="6", stride=str(stride))
        vertices = ET.SubElement(mesh, Q + "vertices", id=name + "_vertices")
        ET.SubElement(vertices, Q + "input", semantic="POSITION", source="#" + name + "_pos")
        triangles = ET.SubElement(mesh, Q + "triangles", material="interior-material", count="2")
        ET.SubElement(triangles, Q + "input", semantic="VERTEX", source="#" + name + "_vertices", offset="0")
        ET.SubElement(triangles, Q + "input", semantic="TEXCOORD", source="#" + name + "_uv", offset="1")
        ET.SubElement(triangles, Q + "p").text = "0 0 1 1 2 2 3 3 4 4 5 5"
        node = ET.SubElement(scene, Q + "node", id=name, name=name)
        ET.SubElement(node, Q + "matrix").text = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
        ET.SubElement(node, Q + "instance_geometry", url="#" + name + "_geometry")
    return root


def faces(root, name):
    node = next(n for n in root.findall(".//c:node", NS) if n.get("name") == name)
    return mesh_faces(root, node)


class ColladaOverlapTests(unittest.TestCase):
    def test_material_winding_and_reused_uvs_are_not_duplicates(self):
        original = fixture()
        road = faces(original, "road")
        race = faces(original, "race")
        self.assertEqual(len(duplicate_uv_keys(race, road)), 1)
        for change in ("material", "winding", "ambiguous_uv"):
            with self.subTest(change=change):
                modified = copy.deepcopy(race)
                if change == "material":
                    modified[0].material = "decal"
                elif change == "winding":
                    modified[0].xyz = modified[0].xyz[::-1]
                    modified[0].uv = modified[0].uv[::-1]
                else:
                    modified[1].uv_key = modified[0].uv_key
                self.assertEqual(duplicate_uv_keys(modified, road), set())

    def test_only_the_cofitted_mesh_loses_shared_faces_after_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            source = fixture()
            archive = tmp / "source.zip"
            member = "vehicles/car/car.dae"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(member, ET.tostring(source))
            context = SimpleNamespace(source_zip=archive, objects=list_dae_objects_for_file(archive, member))
            output = tmp / "output"
            output.mkdir()
            generated = copy.deepcopy(source)
            for node in generated.findall(".//c:node", NS):
                node.set("name", node.get("name") + "_xp_rhd")
                node.set("id", node.get("id") + "_xp_rhd")
            for geometry in generated.findall(".//c:geometry", NS):
                alias = geometry.get("id").split("_")[0]
                geometry.find("c:mesh/c:triangles", NS).set("material", "corrected_" + alias)
            # 2026-09-08: correction may move a formerly shared face rigidly.
            # Source correspondence must still remove that duplicate copy.
            positions = generated.find(".//c:source[@id='race_pos']/c:float_array", NS)
            positions.text = "0.01 0 0 1.01 0 0 0.01 1 0 6 0 0 7 0 0 6 1 0"
            dae = output / "car.dae"
            ET.ElementTree(generated).write(dae)
            header = ["mesh", "[group]:", "nonFlexMaterials"]
            document = {
                "road_part": {"flexbodies": [header, ["road_xp_rhd", ["dash"]], ["race_xp_rhd", ["dash"]]]},
                "race_part": {"flexbodies": [header, ["race_xp_rhd", ["dash"]]]},
                "separate_mounts": {"flexbodies": [header, ["road_xp_rhd", ["dash"]],
                                                   {"pos": {"x": 2}}, ["race_xp_rhd", ["dash"]]]},
            }
            for name, part in document.items():
                part["slotType"] = name
            jbeam = output / "parts.jbeam"
            jbeam.write_text(json.dumps(document))
            result = build_pipeline.deduplicate_generated_flexbodies(
                context, output, {"road": core.MODE_MIRROR, "race": core.MODE_MIRROR},
                {core.HAND_RHD}, {"corrected_road": "interior", "corrected_race": "interior"},
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["removedFaces"], 1)
            after = ET.parse(dae).getroot()
            unique = result[0]["replacement"]
            self.assertEqual(len(faces(after, unique)), 1)
            self.assertEqual(len(faces(after, "race_xp_rhd")), 2)
            self.assertEqual(faces(after, unique)[0].xyz.tolist(), faces(generated, "race_xp_rhd")[1].xyz.tolist())
            patched = core.parse_beamng_json(jbeam.read_text(), label="parts")
            self.assertEqual(patched["race_part"], document["race_part"])
            self.assertEqual(patched["separate_mounts"], document["separate_mounts"])
            self.assertEqual(patched["road_part"]["flexbodies"][2][0], unique)
            ids = [element.get("id") for element in after.iter() if element.get("id")]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(build_pipeline.deduplicate_generated_flexbodies(
                context, output, {"road": core.MODE_MIRROR, "race": core.MODE_MIRROR},
                {core.HAND_RHD}, {"corrected_road": "interior", "corrected_race": "interior"},
            ), [])


if __name__ == "__main__":
    unittest.main()
