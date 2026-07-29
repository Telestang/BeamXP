from __future__ import annotations

from .shared import core
from .recommendation_common import (
    _spatial_entries_for_trim,
    _spatial_surfaces_for_trim,
)
from .recommendation_classifier import _classify_meshes_for_trim
from .recommendation_pairing import (
    _inherit_mounted_parts,
    _passenger_footwell_forced,
    _resolve_trim_pairs,
)


def build_mode_recommendations(
    context: core.VehicleContext,
    object_ids: list[str],
) -> list[dict[str, str]]:
    """Classify meshes for hand conversion from the driver's viewpoint.

    Batch model: the intrinsic class is a property of the mesh, not the trim,
    so each unique mesh is classified once (in the first trim that contains
    it) and memoised; later trims only re-solve low-confidence meshes whose
    position is trim-dependent, plus the inherently per-trim structural
    pairing. State is cached on the context so reopening the modal reuses
    every solved trim. No driver frame (no camera and no wheel) means no
    trustworthy spatial reasoning: the answer is no recommendations, never a
    name-based guess."""
    available = {o for o in object_ids if o in context.objects and o in context.preview_by_id}
    if not available:
        return []
    # A vehicle with no camera and no wheel anywhere is untrustworthy; bail
    # before any work. The frame is then recomputed per trim inside the loop so
    # each trim's driver camera carries that trim's cab nodeMove: multi-cab
    # vehicles (us_semi cabover vs conventional) and LHD/RHD splits (bx, covet)
    # get the correct driver-side eye instead of a meaningless average of both
    # cabs'/sides' cameras.
    if core.driver_frame_for_context(context) is None:
        return []

    state = getattr(context, "_spatial_recommendation_state", None)
    if state is None:
        state = {
            "memo": {}, "vetoed": set(), "hard_vetoed": set(),
            "scoped": set(), "pair_votes": {}, "trims_done": set(),
        }
        context._spatial_recommendation_state = state
    memo: dict[str, tuple[str, str, str, dict]] = state["memo"]
    vetoed: set[str] = state["vetoed"]
    hard_vetoed: set[str] = state.setdefault("hard_vetoed", set())
    scoped: set[str] = state.setdefault("scoped", set())
    pair_votes: dict[str, dict[str, int]] = state["pair_votes"]

    trims: list[str | None] = sorted(context.variants) if context.variants else [None]
    for trim in trims:
        present, entries_np = _spatial_entries_for_trim(context, trim, available)
        if not present:
            continue
        frame = core.driver_frame_for_context(context, config_name=trim)
        if frame is None:
            continue
        surface_np: dict[str, object] = {}
        todo = [
            o for o in present
            if o not in memo
            or (o in context.variant_dependent_meshes
                and (memo[o][0] == "none" or memo[o][2] == "low")
                and trim not in state["trims_done"])
        ]
        if todo:
            surface_np = _spatial_surfaces_for_trim(
                context, trim, present, entries_np
            )
            verdicts, newly_vetoed = _classify_meshes_for_trim(
                context, frame, present, entries_np, todo,
                hard_vetoed=hard_vetoed,
                scoped=scoped,
                surface_np=surface_np,
            )
            vetoed.update(newly_vetoed)
            for o in todo:
                verdict = verdicts.get(o, ("none", "", "med", {}))
                previous = memo.get(o)
                if previous is None or previous[0] == "none":
                    memo[o] = verdict
                elif previous[2] == "low" and verdict[0] != "none" and verdict[2] != "low":
                    memo[o] = verdict  # a trim resolved the borderline case
            forced = _passenger_footwell_forced(
                frame, present, entries_np, memo, hard_vetoed
            )
            forced_todo = [
                o for o in present
                if o in forced and memo.get(o, ("none",))[0] == "none"
            ]
            if forced_todo:
                if not surface_np:
                    surface_np = _spatial_surfaces_for_trim(
                        context, trim, present, entries_np
                    )
                forced_verdicts, forced_vetoed = _classify_meshes_for_trim(
                    context, frame, present, entries_np, forced_todo, forced,
                    hard_vetoed, scoped, surface_np,
                )
                vetoed.update(forced_vetoed)
                for o in forced_todo:
                    verdict = forced_verdicts.get(o, ("none", "", "med", {}))
                    if verdict[0] != "none":
                        memo[o] = verdict
                        vetoed.discard(o)
        if trim not in state["trims_done"]:
            _resolve_trim_pairs(
                context, frame, present, entries_np, memo, vetoed, pair_votes
            )
            _inherit_mounted_parts(
                context, frame, present, entries_np, memo, vetoed, pair_votes, scoped
            )
            # Contact inheritance may expose an off-centre L/R satellite pair
            # after the initial structural pass. Resolve those new pairables
            # now; lone satellites retain the normal aesthetic-Mirror fallback.
            _resolve_trim_pairs(
                context, frame, present, entries_np, memo, vetoed, pair_votes
            )
            state["trims_done"].add(trim)

    # Meshes no trim uses stay unclassified on purpose: the union of mutually
    # exclusive variants is not a cabin, and a part no config fits cannot be
    # converted anyway.

    recommendations: list[dict[str, str]] = []
    emitted_pairs: set[frozenset] = set()
    requested = set(object_ids)
    for object_id in sorted(requested & set(memo)):
        mode, reason, confidence, extra = memo[object_id]
        if mode in {"none", "functional_skip"}:
            continue
        if confidence == "low":
            reason = f"{reason} (low confidence)"
        if mode == "pairable":
            votes = pair_votes.get(object_id)
            twin = max(votes, key=lambda t: (votes[t], t)) if votes else None
            if twin is not None and twin in requested:
                key = frozenset((object_id, twin))
                if key in emitted_pairs:
                    continue
                emitted_pairs.add(key)
                # name the driver-side member so the modal reads naturally
                obj_a = context.objects.get(object_id)
                obj_b = context.objects.get(twin)
                first, second = object_id, twin
                if obj_a is not None and obj_b is not None:
                    if frame.side * obj_b.x > frame.side * obj_a.x:
                        first, second = twin, object_id
                recommendations.append({
                    "kind": "pair",
                    "object_id": first,
                    "source_id": second,
                    "mode": core.MODE_MIRROR_STRUCTURAL,
                    "reason": reason,
                    "confidence": confidence,
                })
            else:
                entry = {
                    "kind": "single",
                    "object_id": object_id,
                    "source_id": "",
                    "mode": core.MODE_MIRROR,
                    "reason": f"{reason}; twin absent in this trim",
                    "confidence": confidence,
                }
                recommendations.append(entry)
        else:
            entry = {
                "kind": "single",
                "object_id": object_id,
                "source_id": "",
                "mode": core.MODE_TRANSLATE if mode == "translate" else core.MODE_MIRROR,
                "reason": reason,
                "confidence": confidence,
            }
            recommendations.append(entry)

    mode_order = {
        core.MODE_TRANSLATE: 0,
        core.MODE_MIRROR: 1,
        core.MODE_MIRROR_STRUCTURAL: 2,
    }
    recommendations.sort(
        key=lambda item: (
            mode_order.get(item["mode"], 99),
            item["object_id"].lower(),
            item.get("source_id", "").lower(),
        )
    )
    return recommendations
