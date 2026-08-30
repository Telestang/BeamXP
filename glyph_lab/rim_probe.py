"""Inspect every stage-6 rim candidate under the current cached settings.

Run ``python -m glyph_lab.rim_probe`` from the repository root.  The command
writes a contact sheet and CSV rather than changing detection: use it to find
features that separate edge bands/stitching from valid controls before a rim
rule is changed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glyph_lab.app import SECTIONS, load_settings
from glyph_lab.corpus import CorpusTexture, load_corpus
from glyph_lab.detect import STAGES, UNRIMMED, LabConfig, detect
from glyph_lab.probe import (
    RegionLabel,
    RegionObservation,
    RegionProbeCollector,
    load_region_labels,
    observation_key,
)

TILE_SIZE = 180
TILE_CAPTION = 34
TILES_ACROSS = 6


def cached_config() -> LabConfig:
    """The same typed, cached configuration the Lab UI would run with."""
    values = load_settings()
    sections: dict[str, object] = {}
    for _title, config_type, attribute in SECTIONS:
        defaults = config_type()
        overrides: dict[str, object] = {}
        for entry in fields(config_type):
            raw = values.get(entry.name, getattr(defaults, entry.name))
            default = getattr(defaults, entry.name)
            overrides[entry.name] = (
                bool(raw) if isinstance(default, bool)
                else int(raw) if isinstance(default, int)
                else float(raw)
            )
        sections[attribute] = replace(defaults, **overrides)
    # A probe has to reach its producer.  Advancing an earlier explicit stop
    # does not change the preceding stages or any cached tuning values.
    stop = int(sections["contrast"].stop_after_stage)
    rim_stage = STAGES.index(UNRIMMED) + 1
    if 0 < stop < rim_stage:
        sections["contrast"] = replace(sections["contrast"], stop_after_stage=rim_stage)
    return LabConfig(**sections)


def expected_decision(
    observation: RegionObservation, default: str,
    labels: dict[tuple[str, str, int, tuple[int, int, int, int]], RegionLabel],
) -> str:
    """Return the explicit human decision, or the sheet's unmarked default."""
    label = labels.get(observation_key(observation))
    return default if label is None else label.decision


def csv_rows(
    observations: list[RegionObservation], default: str | None = None,
    labels: dict[tuple[str, str, int, tuple[int, int, int, int]], RegionLabel] | None = None,
) -> list[dict[str, object]]:
    """Flatten generic observations while retaining arbitrary future features."""
    rows: list[dict[str, object]] = []
    for number, observation in enumerate(observations, start=1):
        row: dict[str, object] = {
            "candidate": number,
            "stage": observation.stage,
            "texture": observation.texture_name,
            "domain": observation.domain,
            "decision": "discard" if observation.discarded else "keep",
            **observation.features,
        }
        if default is not None and labels is not None:
            expected = expected_decision(observation, default, labels)
            row["expected"] = expected
            row["matches"] = (
                "" if expected == "indifferent"
                else observation.discarded == (expected == "discard")
            )
        rows.append(row)
    return rows


def write_csv(
    path: Path, observations: list[RegionObservation], default: str | None = None,
    labels: dict[tuple[str, str, int, tuple[int, int, int, int]], RegionLabel] | None = None,
) -> None:
    rows = csv_rows(observations, default, labels)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def candidate_tile(
    texture: CorpusTexture, observation: RegionObservation, number: int,
    expected: str | None = None, decision_label: str | None = None,
) -> Image.Image:
    """A source-colour crop with the candidate mask painted by its decision."""
    x, y, width, height = observation.region.bounds
    pad = 12
    row0, row1 = max(y - pad, 0), min(y + height + pad, texture.shape[0])
    column0, column1 = max(x - pad, 0), min(x + width + pad, texture.shape[1])
    values = texture.colour[row0:row1, column0:column1].copy()
    mask = np.zeros(values.shape[:2], dtype=bool)
    top, left = y - row0, x - column0
    mask[top:top + height, left:left + width] = observation.region.mask
    accent = np.array(
        (40, 205, 230) if decision_label is not None
        else (230, 55, 55) if observation.discarded else (50, 235, 80),
        dtype=np.uint8,
    )
    values[mask] = (values[mask].astype(np.uint16) + accent.astype(np.uint16)) // 2
    image = Image.fromarray(values)
    image.thumbnail((TILE_SIZE, TILE_SIZE - TILE_CAPTION), Image.Resampling.NEAREST)
    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "#181818")
    left = (TILE_SIZE - image.width) // 2
    top = (TILE_SIZE - TILE_CAPTION - image.height) // 2
    tile.paste(image, (left, top))
    draw = ImageDraw.Draw(tile)
    decision = decision_label or ("DISCARD" if observation.discarded else "KEEP")
    if expected is not None:
        decision = f"{decision} / {expected.upper()}"
    draw.text(
        (4, TILE_SIZE - TILE_CAPTION + 2), f"#{number} {decision}",
        fill=tuple(int(value) for value in accent),
    )
    draw.text(
        (4, TILE_SIZE - 16),
        f"{observation.texture_name} / {observation.domain}", fill="white",
    )
    return tile


def write_gallery(
    path: Path, observations: list[RegionObservation],
    textures: dict[str, CorpusTexture],
    default: str | None = None,
    labels: dict[tuple[str, str, int, tuple[int, int, int, int]], RegionLabel] | None = None,
    decision_label: str | None = None,
    numbers: dict[int, int] | None = None,
) -> None:
    ordered = gallery_order(observations)
    numbers = numbers or {
        id(observation): number
        for number, observation in enumerate(observations, start=1)
    }
    rows = max((len(ordered) + TILES_ACROSS - 1) // TILES_ACROSS, 1)
    sheet = Image.new("RGB", (TILES_ACROSS * TILE_SIZE, rows * TILE_SIZE), "#101010")
    for index, observation in enumerate(ordered):
        tile = candidate_tile(
            textures[observation.texture_name], observation, numbers[id(observation)],
            (expected_decision(observation, default, labels)
             if default is not None and labels is not None else None),
            decision_label,
        )
        sheet.paste(tile, ((index % TILES_ACROSS) * TILE_SIZE, (index // TILES_ACROSS) * TILE_SIZE))
    sheet.save(path)


def gallery_order(observations: list[RegionObservation]) -> list[RegionObservation]:
    """The stable visual order shared by a gallery and its colour annotation."""
    return sorted(
        observations,
        key=lambda item: (
            not item.discarded, item.texture_name,
            int(item.features["y"]), int(item.features["x"]),
        ),
    )


def run(
    corpus: Path, output: Path, names: list[str] | None = None,
    labels_path: Path | None = None,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    textures = load_corpus(corpus, names)
    config = cached_config()
    probe = RegionProbeCollector()
    for texture in textures:
        detect(texture, config, probe=probe)
    observations = [
        observation for observation in probe.observations
        if observation.stage == UNRIMMED
    ]
    default, labels = (
        load_region_labels(labels_path) if labels_path is not None else (None, None)
    )
    write_csv(output / "rim_candidates.csv", observations, default, labels)
    write_gallery(
        output / "rim_candidates.png", observations,
        {texture.name: texture for texture in textures},
        default, labels,
    )
    discarded = sum(observation.discarded for observation in observations)
    summary = f"{len(observations)} rim candidates; {discarded} discarded"
    if default is not None and labels is not None:
        expected = [expected_decision(item, default, labels) for item in observations]
        comparable = [
            (item, decision) for item, decision in zip(observations, expected)
            if decision != "indifferent"
        ]
        mismatches = sum(
            item.discarded != (decision == "discard")
            for item, decision in comparable
        )
        summary += (
            f"; {mismatches}/{len(comparable)} labelled mismatches"
            f" ({expected.count('indifferent')} indifferent)"
        )
    print(
        f"{summary}\n"
        f"{output / 'rim_candidates.csv'}\n{output / 'rim_candidates.png'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", type=Path, default=Path("glyph_corpus"))
    parser.add_argument("--output", type=Path, default=Path("rim_probe"))
    parser.add_argument("--texture", dest="names", action="append")
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    return run(args.corpus, args.output, args.names, args.labels)


if __name__ == "__main__":
    raise SystemExit(main())
