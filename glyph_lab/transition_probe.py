"""Present every post-rim region for material-transition labelling.

Run ``python -m glyph_lab.transition_probe`` from the repository root.  The
command records every region after the rim stage, including ones that the
current conservative rule does not consider eligible.  This makes the next
annotation an evidence set rather than a gallery of the current hypothesis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from glyph_lab.corpus import CorpusTexture, load_corpus
from glyph_lab.detect import UNTRANSITIONED, detect
from glyph_lab.probe import (
    RegionObservation,
    RegionProbeCollector,
    load_region_labels,
)
from glyph_lab.rim_probe import (
    cached_config,
    expected_decision,
    gallery_order,
    write_csv,
    write_gallery,
)

ANNOTATION_DISCARD = (136, 0, 21)
ANNOTATION_INDIFFERENT = (255, 127, 39)
ANNOTATION_MIN_TEXELS = 16


def collect(
    textures: list[CorpusTexture], min_span: int, max_fill: float,
) -> list[RegionObservation]:
    """Collect every post-rim region, including rule-ineligible examples."""
    config = cached_config()
    config = replace(config, contrast=replace(
        config.contrast,
        transition_min_span_px=min_span,
        transition_max_fill=max_fill,
    ))
    probe = RegionProbeCollector()
    for texture in textures:
        detect(texture, config, probe=probe)
    return [item for item in probe.observations if item.stage == UNTRANSITIONED]


def decisions_from_annotation(
    path: Path, observations: list[RegionObservation],
    numbers: dict[int, int], ignore: set[int],
) -> dict[int, str]:
    """Read exact-colour annotations from a contact sheet of this probe.

    2026-08-30: markers are read by RGB equality, rather than approximate
    colour, so artwork in a tile cannot silently become a human label.  The
    input must be the original-size contact sheet, not a resized chat preview.
    """
    image = np.asarray(Image.open(path).convert("RGB"))
    ordered = gallery_order(observations)
    expected_height = ((len(ordered) + 5) // 6) * 180
    if image.shape[:2] != (expected_height, 1080):
        raise ValueError(
            f"{path}: expected 1080x{expected_height} annotation, got "
            f"{image.shape[1]}x{image.shape[0]}"
        )
    discard = np.all(image == ANNOTATION_DISCARD, axis=2)
    indifferent = np.all(image == ANNOTATION_INDIFFERENT, axis=2)
    decisions: dict[int, str] = {}
    for index, observation in enumerate(ordered):
        row, column = divmod(index, 6)
        tile = np.s_[row * 180:(row + 1) * 180, column * 180:(column + 1) * 180]
        number = numbers[id(observation)]
        if number in ignore:
            decisions[number] = "indifferent"
        elif int(discard[tile].sum()) >= ANNOTATION_MIN_TEXELS:
            decisions[number] = "discard"
        elif int(indifferent[tile].sum()) >= ANNOTATION_MIN_TEXELS:
            decisions[number] = "indifferent"
        else:
            decisions[number] = "keep"
    return decisions


def write_annotation_labels(
    path: Path, observations: list[RegionObservation], numbers: dict[int, int],
    decisions: dict[int, str], ignore: set[int],
) -> None:
    """Store human labels using durable texture/domain/bounds identities."""
    labels = []
    for observation in observations:
        number = numbers[id(observation)]
        decision = decisions[number]
        if decision == "keep":
            continue
        labels.append({
            "stage": observation.stage,
            "texture": observation.texture_name,
            "domain": observation.domain,
            "bounds": list(observation.region.bounds),
            "decision": decision,
        })
    document = {
        "default": "keep",
        "notes": [
            "2026-08-30: labels parsed from exact-colour probe annotation.",
            f"Ignored display candidates: {sorted(ignore)} (stored as indifferent).",
        ],
        "labels": labels,
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def run(
    corpus: Path, output: Path, names: list[str] | None = None,
    min_span: int = 64, max_fill: float = 0.30,
    labels_path: Path | None = None, annotation: Path | None = None,
    ignore: set[int] | None = None,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    textures = load_corpus(corpus, names)
    observations = collect(textures, min_span, max_fill)
    sources = {texture.name: texture for texture in textures}
    numbers = {
        id(observation): number
        for number, observation in enumerate(observations, start=1)
    }
    ignored = set() if ignore is None else ignore
    if annotation is not None:
        decisions = decisions_from_annotation(annotation, observations, numbers, ignored)
        labels_path = output / "transition_probe_labels.json"
        write_annotation_labels(labels_path, observations, numbers, decisions, ignored)
    default, labels = (
        load_region_labels(labels_path) if labels_path is not None else (None, None)
    )
    write_csv(output / "transition_candidates.csv", observations, default, labels)
    write_gallery(
        output / "transition_candidates.png", observations, sources,
        default, labels, numbers=numbers,
    )
    for texture in textures:
        subset = [
            observation for observation in observations
            if observation.texture_name == texture.name
        ]
        if subset:
            write_gallery(
                output / f"transition_candidates_{texture.name}.png",
                subset, sources, default, labels, numbers=numbers,
            )
    summary = f"{len(observations)} material-transition candidates"
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
        f"{output / 'transition_candidates.csv'}\n"
        f"{output / 'transition_candidates.png'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", type=Path, default=Path("glyph_corpus"))
    parser.add_argument("--output", type=Path, default=Path("transition_probe"))
    parser.add_argument("--texture", dest="names", action="append")
    parser.add_argument("--min-span", type=int, default=64)
    parser.add_argument("--max-fill", type=float, default=0.30)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--ignore", type=int, action="append", default=[])
    args = parser.parse_args()
    return run(
        args.corpus, args.output, args.names, args.min_span, args.max_fill,
        args.labels, args.annotation, set(args.ignore),
    )


if __name__ == "__main__":
    raise SystemExit(main())
