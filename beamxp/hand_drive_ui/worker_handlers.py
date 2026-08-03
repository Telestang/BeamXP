from __future__ import annotations

from .shared import *


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
        if kind == "variant_hands_done":
            self._handle_variant_hands_done(payload)
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
                plate_note += f"; plate library mod refreshed ({plates.get('libraryModDesigns', 0)} design(s))"
            if result.installed_zip:
                self.status_var.set(
                    f"Built {result.package_zip} and installed {result.installed_zip}; "
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
        self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed")
        self._schedule_pending_parts_refresh()

    def _schedule_pending_parts_refresh(self) -> None:
        if not self.part_refresh_pending:
            return
        reset_view = self.part_refresh_pending_reset
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self._schedule_parts_refresh(reset_view=reset_view)
