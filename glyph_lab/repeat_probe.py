"""Compare an image-repeat proposal with the current stage-5 decision.

The normal repeat filter intentionally skips small regions before allocating a
correlation response.  This probe is for deciding whether a lower gate or a
smaller image tile is safe: its delta gallery contains only regions a proposal
would newly discard, so collateral is visible before tuning is adopted.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from glyph_lab.corpus import CorpusTexture, load_corpus
from glyph_lab.detect import (
    STAGES,
    UNCROWDED,
    UNREPEATED,
    build_repeat_texture_index,
    detect,
    repeat_texture_evidence,
)
from glyph_lab.probe import RegionObservation
from glyph_lab.regions import Region
from glyph_lab.rim_probe import cached_config, write_csv, write_gallery


def repeat_decision(
    region: Region, own: int, directions: int, min_span: int, min_area: int,
    min_repeats: int, context_directions: int,
) -> bool:
    """Whether one region passes the cheap gate and repeat evidence."""
    return (
        min(region.width, region.height) >= max(int(min_span), 0)
        and region.texels >= max(int(min_area), 0)
        and (own >= min_repeats or directions >= context_directions)
    )


def repeat_features(
    region: Region, own: int, directions: int, current_discarded: bool,
) -> dict[str, int | float | bool]:
    """Scalar evidence retained for a proposal and a later label set."""
    minor = min(region.width, region.height)
    major = max(region.width, region.height)
    return {
        "x": region.x,
        "y": region.y,
        "width_px": region.width,
        "height_px": region.height,
        "texels": region.texels,
        "minor_span_px": minor,
        "major_span_px": major,
        "aspect": major / max(minor, 1),
        "texture_repeats": own,
        "context_directions": directions,
        "current_discarded": current_discarded,
    }


def collect(
    textures: list[CorpusTexture], tile_px: int, min_span: int, min_area: int,
    context_directions: int | None = None,
) -> list[RegionObservation]:
    """Score every post-size region with the current and proposal settings."""
    config = cached_config()
    current = config.contrast
    proposal = replace(
        current, repeat_texture_tile_px=tile_px,
        repeat_texture_min_region_span_px=min_span,
        repeat_texture_min_region_area_px=min_area,
        repeat_texture_context_directions=(
            current.repeat_texture_context_directions
            if context_directions is None else context_directions
        ),
    )
    observations: list[RegionObservation] = []
    for texture in textures:
        # 2026-09-04: symbolic, because the number moved when the recovery
        # went below the crowd filter.  It was 4 and meant the crowd stage.
        before_repeat = replace(
            config,
            contrast=replace(
                current, stop_after_stage=STAGES.index(UNCROWDED) + 1,
            ),
        )
        regions = detect(texture, before_repeat).regions
        current_index = build_repeat_texture_index(texture.colour, current)
        proposal_index = (
            current_index if proposal.repeat_texture_tile_px == current.repeat_texture_tile_px
            else build_repeat_texture_index(texture.colour, proposal)
        )
        for region in regions:
            current_own, current_directions = repeat_texture_evidence(
                current_index, region.bounds, current
            )
            proposed_own, proposed_directions = repeat_texture_evidence(
                proposal_index, region.bounds, proposal
            )
            current_discarded = repeat_decision(
                region, current_own, current_directions,
                current.repeat_texture_min_region_span_px,
                current.repeat_texture_min_region_area_px,
                current.repeat_texture_min_repeats,
                current.repeat_texture_context_directions,
            )
            proposed_discarded = repeat_decision(
                region, proposed_own, proposed_directions,
                proposal.repeat_texture_min_region_span_px,
                proposal.repeat_texture_min_region_area_px,
                proposal.repeat_texture_min_repeats,
                proposal.repeat_texture_context_directions,
            )
            observations.append(RegionObservation(
                stage=UNREPEATED, texture_name=texture.name, domain=region.domain,
                region=region,
                features=repeat_features(
                    region, proposed_own, proposed_directions, current_discarded,
                ),
                discarded=proposed_discarded,
            ))
    return observations


def run(
    corpus: Path, output: Path, names: list[str] | None = None,
    tile_px: int = 24, min_span: int = 8, min_area: int = 0,
    context_directions: int | None = None,
) -> int:
    """Write all candidate decisions and the proposal-only removal delta."""
    output.mkdir(parents=True, exist_ok=True)
    textures = load_corpus(corpus, names)
    observations = collect(textures, tile_px, min_span, min_area, context_directions)
    sources = {texture.name: texture for texture in textures}
    numbers = {id(item): number for number, item in enumerate(observations, 1)}
    delta = [
        item for item in observations
        if item.discarded and not bool(item.features["current_discarded"])
    ]
    write_csv(output / "repeat_candidates.csv", observations)
    write_gallery(output / "repeat_candidates.png", observations, sources, numbers=numbers)
    write_csv(output / "repeat_delta.csv", delta)
    write_gallery(
        output / "repeat_delta.png", delta, sources,
        decision_label="NEW DISCARD", numbers=numbers,
    )
    print(
        f"{len(observations)} post-size regions; {len(delta)} newly discarded\n"
        f"{output / 'repeat_candidates.csv'}\n{output / 'repeat_delta.png'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", type=Path, default=Path("glyph_corpus"))
    parser.add_argument("--output", type=Path, default=Path("repeat_probe"))
    parser.add_argument("--texture", dest="names", action="append")
    parser.add_argument("--tile-px", type=int, default=24)
    parser.add_argument("--min-span", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--context-directions", type=int)
    args = parser.parse_args()
    return run(
        args.corpus, args.output, args.names,
        args.tile_px, args.min_span, args.min_area, args.context_directions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
