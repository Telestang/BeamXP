"""Reusable candidate observations for inspecting late-stage region filters.

The detector stays independent of any particular report/UI: a filter emits a
stage name, its region and scalar facts, and a caller decides whether to keep
them in memory, write a gallery, or compare two runs.  Rim-artifact analysis
is the first consumer, not a special debugging side channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from glyph_lab.regions import Region

ProbeValue = bool | int | float | str | None


@dataclass(frozen=True, slots=True)
class RegionObservation:
    """One candidate a late-stage filter considered."""

    stage: str
    texture_name: str
    domain: int
    region: Region
    features: dict[str, ProbeValue]
    discarded: bool


class RegionProbe(Protocol):
    """A sink that receives candidate observations from any region filter."""

    def observe(self, observation: RegionObservation) -> None:
        """Accept one complete observation."""


@dataclass(slots=True)
class RegionProbeCollector:
    """The small in-memory collector used by diagnostic commands and tests."""

    observations: list[RegionObservation] = field(default_factory=list)

    def observe(self, observation: RegionObservation) -> None:
        self.observations.append(observation)


@dataclass(frozen=True, slots=True)
class RegionLabel:
    """A human decision tied to one stable, observed region."""

    stage: str
    texture_name: str
    domain: int
    bounds: tuple[int, int, int, int]
    decision: str


def observation_key(
    observation: RegionObservation,
) -> tuple[str, str, int, tuple[int, int, int, int]]:
    """The durable identity for a labelled candidate, independent of display ID."""
    return (
        observation.stage, observation.texture_name, observation.domain,
        observation.region.bounds,
    )


def load_region_labels(path: Path) -> tuple[str, dict[tuple[str, str, int, tuple[int, int, int, int]], RegionLabel]]:
    """Read a generic probe label set and index it by observed-region identity.

    The sheet's display number is intentionally absent: ordering changes when
    a filter changes, whereas its source texture, domain and tight bounds do
    not.  A label file's default describes the unmarked tiles in that sheet.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    default = document.get("default", "keep")
    if default not in {"keep", "discard", "indifferent"}:
        raise ValueError(f"{path}: invalid default decision {default!r}")
    labels: dict[tuple[str, str, int, tuple[int, int, int, int]], RegionLabel] = {}
    for raw in document.get("labels", []):
        decision = raw.get("decision")
        bounds = raw.get("bounds")
        if decision not in {"keep", "discard", "indifferent"}:
            raise ValueError(f"{path}: invalid decision {decision!r}")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError(f"{path}: each label needs four bounds")
        label = RegionLabel(
            stage=str(raw["stage"]), texture_name=str(raw["texture"]),
            domain=int(raw["domain"]),
            bounds=tuple(int(value) for value in bounds), decision=decision,
        )
        key = (label.stage, label.texture_name, label.domain, label.bounds)
        if key in labels:
            raise ValueError(f"{path}: duplicate label for {key}")
        labels[key] = label
    return default, labels
