from __future__ import annotations

from .shared import *

# What share of the bar each build stage gets, in order. Eight equal stages
# was the obvious reading and a badly wrong one: measured over two recorded
# lc500 builds, texture correction is 86% of the wall clock and the other
# seven stages share the rest, so an even bar sat between 50% and 61% for
# ninety of a hundred and ten seconds and then jumped to the end.
#
# 2026-09-07: measured, not guessed -- 1.0 / 0.0 / 5.8 / 1.7 / 85.7 / 4.0 /
# 0.8 / 0.9 percent, one trim, with and without packaging. Rounded, and stage
# eight given more than it took here because that build zipped one small trim
# and an install copies far more. A build that skips texture correction skips
# most of the bar in one move, which the easing turns into a glide and which
# is the truth: the work really was not done.
STAGE_WEIGHTS: tuple[int, ...] = (1, 1, 6, 2, 80, 4, 1, 5)


def _stage_span(stage: int, stage_total: int) -> tuple[float, float]:
    """The (start, end) of one stage's share of the whole bar."""
    weights = STAGE_WEIGHTS if stage_total == len(STAGE_WEIGHTS) else (1,) * stage_total
    total = sum(weights) or 1
    return sum(weights[: stage - 1]) / total, sum(weights[:stage]) / total


# Where a stage's phases sit inside it, as (start, end) of that stage's share.
#
# Only stage five needs this, and it needs it badly: on an lc500 build it is
# 89 of the 114 seconds, and it runs three phases back to back, each counting
# something different. Masks count textures (34 of them, 30 s), corrections
# count groups (28 of them, 54 s), and the export around them counts nothing.
# Given the whole stage, each would drive the bar from its start to its end
# and the next would drag it back; given a slice, they hand over.
#
# 2026-09-07: measured off a recorded build, not guessed -- the phases are
# sequential, and the split is roughly where the time goes.
STAGE_PHASE_SPANS: dict[tuple[int, str], tuple[float, float]] = {
    (5, "build_domain_masks"): (0.05, 0.40),
    (5, "correct_textures"): (0.40, 0.92),
}


class WorkerHandlersMixin:
    """Dispatch and completion handling for background worker messages."""

    def _handle_worker_message(self, kind: str, payload: object) -> None:
        if kind == "parts_success":
            self._handle_parts_success(payload)
            return
        if kind == "vehicle_load_success":
            self._handle_vehicle_load_success(payload)
            return
        if kind == "vehicle_load_error":
            self._handle_vehicle_load_error(payload)
            return
        if kind == "inventory_scan_done":
            self._handle_inventory_scan_done(payload)
            return
        if kind == "inventory_scan_error":
            self._handle_inventory_scan_error(payload)
            return
        if kind == "recommendations_success":
            self._handle_recommendations_success(payload)
            return
        if kind == "recommendations_error":
            self._handle_recommendations_error(payload)
            return
        if kind == "mesh_scene_done":
            self._handle_mesh_scene_done(payload)
            return
        if kind == "part_table_success":
            self._handle_part_table_success(payload)
            return
        if kind == "variant_hands_done":
            self._handle_variant_hands_done(payload)
            return
        if kind == "build_progress":
            self._handle_build_progress(payload)
            return
        if kind == "status":
            self.status_var.set(str(payload))
            return

        self._set_busy(False)
        if kind == "build_success":
            result: core.BuildResult = payload
            plate_note = ""
            plates = result.plate_summary or {}
            if plates.get("configsUpdated"):
                plate_note = f"; plates on {plates['configsUpdated']} config(s)"
                warnings = plates.get("warnings") or []
                if warnings:
                    plate_note += f" ({len(warnings)} plate warning(s), see conversion.json)"
            if result.installed_plates_zip:
                action = "refreshed" if plates.get("libraryModInstalled", True) else "unchanged"
                plate_note += f"; plate library mod {action} ({plates.get('libraryModDesigns', 0)} design(s))"
            if result.installed_zip:
                self.status_var.set(
                    f"Built and installed {result.installed_zip}; "
                    f"{len(result.generated_configs)} config(s){plate_note}"
                )
            else:
                self.status_var.set(
                    f"Built {result.package_zip}; {len(result.generated_configs)} config(s){plate_note}"
                )
        elif kind == "preview_success":
            self.status_var.set(f"Blender preview launched: {payload}")
        else:
            self._show_error("Operation failed", str(payload))
            self.status_var.set("Operation failed")
        self._refresh_all()

    def _handle_build_progress(self, payload: object) -> None:
        """Show a build stage, and place it on the bar as part of the whole run.

        The bar reads the stage number, not the stage's own counts: eight
        stages each driving the bar from empty to full meant eight sweeps and
        seven snaps back to the start.
        """
        if not isinstance(payload, dict):
            return
        try:
            stage = int(payload.get("stage") or 0)
            stage_total = int(payload.get("stage_total") or 0)
        except (TypeError, ValueError):
            return
        message = str(payload.get("message") or "Working...")
        prefix = (
            f"Stage {stage}/{stage_total} — "
            if stage > 0 and stage_total > 0
            else ""
        )
        phase = str(payload.get("phase") or "")
        current = payload.get("current")
        total = payload.get("total")
        try:
            current = int(current) if current is not None else None
            total = int(total) if total is not None else None
        except (TypeError, ValueError):
            current = total = None

        if phase == "build_domain_masks" and total and current is not None:
            message = f"UV masks — {current}/{total} atlas files"
        elif phase == "correct_textures" and total and current is not None:
            remaining = int(payload.get("remaining_corrections") or 0)
            workers = int(payload.get("workers") or 1)
            message = (
                f"Texture correction — {current}/{total} groups complete; "
                f"{remaining} map corrections remaining ({workers} worker(s))"
            )

        self.status_var.set(prefix + message)
        if not hasattr(self, "busy_progress"):
            return
        if stage <= 0 or stage_total <= 0:
            self._sweep_progress()
            return
        # Where this stage has got to. A stage that reports nothing
        # contributes its own start, which the monotonic clamp then reads as
        # "stay put"; a phase with a slice of the stage reports inside it.
        within = 0.0
        if current is not None and total is not None and total > 0:
            within = max(0.0, min(1.0, current / total))
        phase_start, phase_end = STAGE_PHASE_SPANS.get((stage, phase), (0.0, 1.0))
        within = phase_start + within * (phase_end - phase_start)
        stage_start, stage_end = _stage_span(stage, stage_total)
        self._advance_progress(stage_start + within * (stage_end - stage_start))

    def _handle_parts_success(self, payload: object) -> None:
        seq, context, reset_view, selected, future = payload
        self.part_refresh_running = False
        should_apply = seq == self.part_refresh_seq and context is self.context
        try:
            result = future.result()
        except Exception as exc:
            if should_apply:
                self.resolved_part_ids = []
                self._refresh_parts(reset_view=reset_view)
                self.status_var.set(f"Part resolver failed: {exc}")
            self._schedule_pending_parts_refresh()
            return
        if not should_apply:
            self._schedule_pending_parts_refresh()
            return
        self.resolved_part_ids = result
        self._invalidate_slot_usage()
        core.save_cached_part_ids(context, selected, self.resolved_part_ids)
        self._refresh_parts(reset_view=reset_view)
        self._update_detail()
        if self.part_table_snapshot_key == self._part_table_snapshot_key():
            self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed")
        self._schedule_pending_parts_refresh()

    def _handle_part_table_success(self, payload: object) -> None:
        seq, context, key, future = payload
        self.part_table_running = False
        self.part_table_requested_key = None
        should_apply = (
            seq == self.part_table_seq
            and context is self.context
            and key == self._part_table_snapshot_key()
        )
        try:
            snapshot = future.result()
        except Exception as exc:
            if should_apply:
                self.status_var.set(f"Part table refresh failed: {exc}")
            else:
                self.part_table_pending = True
            self._schedule_pending_part_table_snapshot()
            return
        if not should_apply:
            # 2026-09-06: The key can change without a pending request (for
            # example when detected hands refresh the preview config choices).
            # A rejected result must arrange a table for the current state.
            self.part_table_pending = True
            self._schedule_pending_part_table_snapshot()
            return
        self.part_table_snapshot_key = key
        self.part_table_snapshot = snapshot
        reset_view = self.part_table_pending_reset
        self.part_table_pending_reset = False
        self._refresh_parts(reset_view=reset_view)
        self._update_detail()
        self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed")
        self._schedule_pending_part_table_snapshot()

    def _schedule_pending_parts_refresh(self) -> None:
        if not self.part_refresh_pending:
            return
        reset_view = self.part_refresh_pending_reset
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self._schedule_parts_refresh(reset_view=reset_view)
