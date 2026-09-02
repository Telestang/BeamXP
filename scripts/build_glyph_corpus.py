"""Extract a fixed texture corpus for glyph_lab, once, from the vehicle zips.

This is the one place glyph_lab leans on the existing code, and it leans hard
on purpose: resolving a material name to the COLLADA symbols a part actually
binds, and rasterising that part's UV islands, is subtle and already correct in
``beamxp_transform_sym_mesh_POC`` / ``annotate_texture_tuning_app``.
Reimplementing it here would be a bug farm for no gain, and nothing in the new
pipeline imports any of it -- only this script does, and only offline.

    python scripts/build_glyph_corpus.py selections.json corpus/
    python scripts/build_glyph_corpus.py --list-parts <vehicle.zip> <dae-member>

A selections file is a list of entries:

    [{"name": "etk800_dash",
      "zip":  "G:/SteamLibrary/.../etk800.zip",
      "dae":  "vehicles/etk800/etk800.dae",
      "part": "etk800_dash",
      "texture": "etk800_interior_black_b.color"}]

``part`` is the legacy single-part form.  ``parts`` selects an explicit set of
DAE part labels, each matched exactly or by prefix.  ``texture`` matches any
substring of the resolved texture member, case-insensitively.  Entries that
resolve to the same archive texture member share one corpus atlas: their mesh
geometry is combined under the first entry's name.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import ndimage

from glyph_lab.corpus import (
    MANIFEST_NAME,
    UvDomainGeometry,
    domain_labels_from_crops,
    write_texture,
)
# 2026-09-02: this ``material_symbols_for_binding`` is NOT the one the export
# path uses.  ``mirror_texture_for_rhd`` has its own, which prefers an exact
# ``dae_material`` match and only falls back to the looser alias union; this one
# takes the union outright.  Measured across the shipped selections they agree
# on eight of eleven bindings and differ on the Volvo, where this one also picks
# up ``v60_andronisk_int_buttons_off-material`` -- so that corpus texture's seam
# map covers the off-state geometry and the export's would not.  Left divergent
# rather than unified: both are defensible (a seam map wants every symbol
# painting the atlas; a layer correction wants the symbol it is correcting), and
# changing this one silently rebuilds the corpus that the detector was tuned
# against.  Reconcile deliberately, with a corpus rebuild and a re-judge.
from mesh_segmentation_transform.annotate_texture_tuning_app import (
    build_texture_source,
    load_normal_map,
    material_symbols_for_binding,
)
from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    archive_texture_choices_for_part,
    extract_archive_member,
    load_dae,
    scan_vehicle_archive,
)
from mesh_segmentation_transform.extract_uv_island_paths import (
    discontinuous_edge_extents,
    merge_overlapping_mask_crops,
    uv_island_masks,
)


@dataclass(frozen=True, slots=True)
class ResolvedSelection:
    """One explicit mesh selection after its texture binding is resolved."""

    entry: dict
    archive: object
    loaded: object
    part_bindings: tuple[tuple[object, object], ...]

    @property
    def texture_member(self) -> str:
        return self.part_bindings[0][1].texture_member


def pick_part(loaded, wanted: str):
    exact = [part for part in loaded.parts if part.label == wanted]
    if exact:
        return exact[0]
    prefixed = [part for part in loaded.parts if part.label.startswith(wanted)]
    if not prefixed:
        raise KeyError(
            f"no part matching {wanted!r}; try --list-parts to see what is there"
        )
    return prefixed[0]


def pick_parts(loaded, entry: dict):
    """Resolve explicit parts, one legacy part, or a contained-name selection."""
    explicit = tuple(str(value) for value in entry.get("parts", ()))
    if explicit:
        picked = tuple(pick_part(loaded, wanted) for wanted in explicit)
        unique = tuple(
            part for index, part in enumerate(picked)
            if all(part.label != earlier.label for earlier in picked[:index])
        )
        if len(unique) != len(explicit):
            raise ValueError(f"duplicate selected part in {explicit!r}")
        return unique
    contains = tuple(str(value).lower() for value in entry.get("part_contains", ()))
    if not contains:
        return (pick_part(loaded, entry["part"]),)
    parts = tuple(
        part for part in loaded.parts
        if any(term in part.node_name.lower() for term in contains)
    )
    if not parts:
        raise KeyError(f"no parts contain any of {contains!r}")
    return parts


def pick_binding(archive, loaded, parts, wanted: str):
    """Choose one texture binding that every selected mesh actually uses."""
    choices = tuple(
        archive_texture_choices_for_part(archive, loaded, part) for part in parts
    )
    matches = [
        choice for choice in choices[0]
        if wanted.lower() in choice.texture_member.lower()
    ]
    if not matches:
        available = ", ".join(sorted({c.texture_member for c in choices[0]})) or "none"
        raise KeyError(f"no texture matching {wanted!r}; first part has: {available}")
    for match in matches:
        member = match.texture_member.lower()
        if all(
            any(choice.texture_member.lower() == member for choice in group)
            for group in choices[1:]
        ):
            return match
    raise ValueError(
        f"{wanted!r} is not shared by every selected part: "
        f"{', '.join(part.node_name for part in parts)}"
    )


def resolve_selection(
    entry: dict,
    loaded_archives: dict[tuple[Path, str], tuple[object, object]],
) -> ResolvedSelection:
    """Resolve one selection without reloading a DAE another entry uses."""
    archive_key = (Path(entry["zip"]).resolve(), str(entry["dae"]))
    archive_and_dae = loaded_archives.get(archive_key)
    if archive_and_dae is None:
        workspace = Path(tempfile.mkdtemp(prefix="glyph_corpus_"))
        archive = scan_vehicle_archive(archive_key[0], workspace)
        loaded = load_dae(extract_archive_member(archive, archive_key[1]))
        archive_and_dae = archive, loaded
        loaded_archives[archive_key] = archive_and_dae
    archive, loaded = archive_and_dae
    parts = pick_parts(loaded, entry)
    part_bindings = tuple(
        (part, pick_binding(archive, loaded, (part,), entry["texture"]))
        for part in parts
    )
    member = part_bindings[0][1].texture_member.casefold()
    if any(binding.texture_member.casefold() != member
           for _part, binding in part_bindings[1:]):
        raise ValueError(
            f"{entry['texture']!r} resolves to different texture members for "
            f"{', '.join(part.node_name for part in parts)}"
        )
    return ResolvedSelection(entry, archive, loaded, part_bindings)


def texture_key(selection: ResolvedSelection) -> tuple[Path, str, str]:
    """The physical image backing a corpus texture, not a selection label."""
    return (
        Path(selection.entry["zip"]).resolve(),
        str(selection.entry["dae"]),
        selection.texture_member.casefold(),
    )


def mesh_seams(
    loaded,
    shape: tuple[int, int],
    part_bindings: tuple[tuple[object, object], ...],
) -> tuple[
    np.ndarray,
    tuple[tuple[np.ndarray, tuple[tuple[tuple[float, float], tuple[float, float]], ...]], ...],
    np.ndarray,
]:
    """Every edge the mesh does not carry through, tagged with its chart's size.

    2026-08-30: a material can be shared by unrelated meshes in the same DAE.
    The selected part's geometry IDs are therefore an inclusion boundary, not
    just a hint: a pedal face sharing a dashboard material must not produce a
    dashboard seam.
    """
    height, width = shape
    root = loaded.tree.getroot()
    seams = np.zeros((height, width), dtype=np.uint16)
    regions: list[tuple[np.ndarray, tuple[tuple[tuple[float, float], tuple[float, float]], ...]]] = []
    geometry_seams = np.zeros((height, width), dtype=bool)
    for part, binding in part_bindings:
        geometry_ids = frozenset(
            instance.geometry_id for instance in part.instances
        )
        if not geometry_ids:
            raise ValueError(f"selected part has no DAE geometry: {part.label}")
        # 2026-08-30: selected meshes can intentionally share an atlas.  Keep
        # their topology passes separate so coincident UV coordinates never
        # weld two meshes into one seam region.
        for symbol in material_symbols_for_binding(loaded, binding):
            try:
                extents, stats = discontinuous_edge_extents(
                    root, symbol, (width, height), collect_regions=True,
                    geometry_ids=geometry_ids,
                )
            except ValueError:
                continue
            for region, segments in zip(
                stats["seam_region_masks"], stats["seam_region_segments"], strict=True,
            ):
                regions.append((region, segments))
                geometry_seams |= region
            # Where two symbols' charts land on the same texel the smaller wins,
            # for the reason the extents exist: size the band for the more delicate
            # of the two features it separates.
            taken = extents > 0
            held = seams > 0
            seams = np.where(taken & held, np.minimum(seams, extents),
                             np.where(taken, extents, seams)).astype(np.uint16)
    return seams, tuple(regions), geometry_seams


def part_uv_islands(
    loaded,
    shape: tuple[int, int],
    part_bindings: tuple[tuple[object, object], ...],
):
    """Rasterise the selected DAE meshes' charts on their shared atlas."""
    height, width = shape
    islands = []
    root = loaded.tree.getroot()
    for part, binding in part_bindings:
        geometry_ids = frozenset(
            instance.geometry_id for instance in part.instances
        )
        if not geometry_ids:
            raise ValueError(f"selected part has no DAE geometry: {part.label}")
        for symbol in material_symbols_for_binding(loaded, binding):
            try:
                symbol_islands, _stats = uv_island_masks(
                    root, symbol, (width, height), geometry_ids,
                )
            except ValueError:
                continue
            islands.extend(symbol_islands)
    if not islands:
        raise ValueError("no UV islands for selected meshes")
    return tuple(islands)


def simple_uv_geometry(
    domains: np.ndarray,
    seam_regions: tuple[
        tuple[np.ndarray, tuple[tuple[tuple[float, float], tuple[float, float]], ...]],
        ...,
    ],
) -> tuple[UvDomainGeometry, ...]:
    """Persist raw simple-chart geometry for the Glyph Lab overlay.

    2026-08-30: a sidecar records authored mesh-seam coordinates, not a
    build-time classification.  Glyph Lab applies its tuning controls to each
    region live, preserving the mesh boundary as its evidence.
    """
    geometry: list[UvDomainGeometry] = []
    for region, source_segments in seam_regions:
        nearby = ndimage.binary_dilation(
            region, structure=np.ones((3, 3), dtype=bool),
        )
        labels = domains[nearby]
        labels = labels[labels > 0]
        if len(labels):
            values, counts = np.unique(labels, return_counts=True)
            label = int(values[np.argmax(counts)])
        else:
            # 2026-08-30: a wrapped source chart can have no in-tile label.
            label = 0
        geometry.append(UvDomainGeometry(
            label, "seam", source_segments=source_segments,
        ))
    return tuple(geometry)


def extract(selections: tuple[ResolvedSelection, ...], out_dir: Path) -> dict:
    """Write one texture atlas from every selected mesh bound to that image."""
    first = selections[0]
    archive, loaded = first.archive, first.loaded
    part_bindings = tuple(
        pair for selection in selections for pair in selection.part_bindings
    )
    binding = part_bindings[0][1]
    source = build_texture_source(archive, loaded, binding, load_normal=True)
    if source.normal_rgb is None:
        # load_normal=True defers nothing, but a material may simply have none.
        source.normal_rgb, _member, _note = load_normal_map(
            archive, binding,
            (source.image.shape[1], source.image.shape[0]),
        )

    crops = merge_overlapping_mask_crops(
        tuple(
            (island.x, island.y, island.mask)
            for island in part_uv_islands(
                loaded, source.image.shape[:2], part_bindings,
            ) if island.mask.any()
        )
    )
    domains = domain_labels_from_crops(list(crops), source.image.shape[:2])
    seams, seam_regions, geometry_seams = mesh_seams(
        loaded, source.image.shape[:2], part_bindings,
    )
    geometry = simple_uv_geometry(domains, seam_regions)
    record = write_texture(
        out_dir, first.entry["name"], source.image, domains, source.normal_rgb,
        seams,
        geometry,
        geometry_seams,
    )
    record["source"] = {
        "zip": Path(first.entry["zip"]).name,
        "dae": first.entry["dae"],
        "part": ", ".join(part.label for part, _binding in part_bindings),
        "texture": binding.texture_member,
        "material": ", ".join(dict.fromkeys(
            part_binding.dae_material for _part, part_binding in part_bindings
        )),
    }
    record["selected_meshes"] = len(part_bindings)
    record["topological_islands"] = len(crops)
    return record


def remove_texture_files(directory: Path, name: str) -> None:
    """Remove generated sidecars for an atlas superseded by texture sharing."""
    for suffix in (
        ".color.png", ".domains.png", ".normal.png", ".seams.png",
        ".geometry.json", ".geometry_seams.png",
    ):
        (directory / f"{name}{suffix}").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selections", nargs="?", type=Path)
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--list-parts", nargs=2, metavar=("ZIP", "DAE"))
    parser.add_argument(
        "--append", action="store_true",
        help="retain existing manifest entries while adding new selections",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="replace named existing entries while retaining every other manifest entry",
    )
    args = parser.parse_args()

    if args.list_parts:
        zip_path, dae_member = args.list_parts
        workspace = Path(tempfile.mkdtemp(prefix="glyph_corpus_"))
        archive = scan_vehicle_archive(Path(zip_path), workspace)
        loaded = load_dae(extract_archive_member(archive, dae_member))
        for part in loaded.parts:
            print(part.label)
        return

    if not args.selections or not args.out_dir:
        parser.error("selections and out_dir are required unless --list-parts")

    entries = json.loads(args.selections.read_text())
    manifest_path = args.out_dir / MANIFEST_NAME
    existing_records: list[dict] = []
    if (args.append or args.replace) and manifest_path.exists():
        existing_records = list(json.loads(manifest_path.read_text()).get("textures", []))
    loaded_archives: dict[tuple[Path, str], tuple[object, object]] = {}
    grouped: dict[tuple[Path, str, str], list[ResolvedSelection]] = {}
    failures = []
    for entry in entries:
        try:
            selection = resolve_selection(entry, loaded_archives)
        except Exception as exc:  # one bad selection must not lose the rest
            failures.append((entry.get("name", "?"), f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {entry.get('name', '?')}: {exc}")
            continue
        group = grouped.setdefault(texture_key(selection), [])
        if group:
            print(
                f"  share {entry['name']:<22} -> {group[0].entry['name']} "
                f"({selection.texture_member})"
            )
        group.append(selection)

    existing_names = {record["name"] for record in existing_records}
    records, replacements, stale_aliases = [], {}, set()
    for selections in grouped.values():
        name = selections[0].entry["name"]
        aliases = {
            selection.entry["name"] for selection in selections[1:]
            if selection.entry["name"] != name
        }
        try:
            if name in existing_names and not args.replace:
                raise ValueError(f"corpus already has a texture named {name!r}")
            occupied_aliases = aliases & existing_names
            if occupied_aliases and not args.replace:
                raise ValueError(
                    "shared texture still has old alias records: "
                    f"{', '.join(sorted(occupied_aliases))}"
                )
            record = extract(tuple(selections), args.out_dir)
        except Exception as exc:  # one bad texture must not lose the rest
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {name}: {exc}")
            continue
        stale_aliases.update(aliases & existing_names)
        if name in existing_names:
            replacements[name] = record
        else:
            records.append(record)
        existing_names.add(name)
        print(
            f"  {name:<28} {record['size'][0]}x{record['size'][1]}  "
            f"{record['domains']} domains  UV {record['uv_fraction']:.1%}"
            f"  {record['seam_texels']:,} seam texels"
            f"{'  +normal' if record['has_normal'] else ''}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_records = [
        replacements.get(record["name"], record)
        for record in existing_records if record["name"] not in stale_aliases
    ] + records
    manifest_path.write_text(
        json.dumps({"textures": combined_records}, indent=2)
    )
    for name in stale_aliases:
        remove_texture_files(args.out_dir, name)
    print(
        f"\n{len(records)} texture(s) added; {len(replacements)} replaced; "
        f"{len(stale_aliases)} shared alias(es) removed; "
        f"{len(combined_records)} total -> {manifest_path}"
    )
    for name, why in failures:
        print(f"  unresolved: {name}: {why}")


if __name__ == "__main__":
    main()
