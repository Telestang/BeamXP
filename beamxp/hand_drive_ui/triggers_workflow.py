"""The Triggers table: the transform for each interaction trigger box.

A trigger box is placed by three jbeam nodes (idRef/idX/idY) plus an offset in
the frame they build, so where it ends up after a conversion depends on what
those nodes are attached to. BeamXP can usually work that out, but not always:
on the scintilla six of sixteen boxes match no rung of the attribution ladder,
including the hood release, which therefore stayed in the left footwell
through a conversion.

Rows are keyed by the box's **authored position**, not by the part that
declares it. That is what keeps the table trim-proof: a switch authored at the
same place in the road dash and the race dash is one row with one answer, and
two boxes that merely share a name are "headlights #1" and "headlights #2".
"""

from __future__ import annotations

from .shared import *

TRIGGER_ROW_PREFIX = "trig"
TRIGGER_MODE_LABELS = {
    core.MODE_SKIP: "Skip",
    core.MODE_TRANSLATE: "Move",
    core.MODE_MIRROR: "Mirror",
}
TRIGGER_MODE_BY_LABEL = {label: mode for mode, label in TRIGGER_MODE_LABELS.items()}
TRIGGER_MODE_OPTIONS = list(TRIGGER_MODE_LABELS.values())


class TriggersWorkflowMixin:
    """Per-trigger transform configuration, keyed by authored position."""

    def _trigger_boxes(self) -> list[dict[str, object]]:
        """The distinct trigger boxes for this selection, cached.

        Enumerating them walks each selected trim's part bodies -- about 23 ms
        across the scintilla's sixteen trims -- and none of it depends on how
        any part is set, so it is keyed on the vehicle and the trims alone. A
        filter keystroke or a mode edit must never pay for it again.
        """
        if self.context is None:
            return []
        configs = tuple(self._selected_variant_names())
        if not configs:
            return []
        key = (id(self.context), configs)
        cached = getattr(self, "_trigger_box_cache", None)
        if isinstance(cached, tuple) and cached[0] == key:
            return cached[1]

        part_ids: set[str] = set()
        for config_name in configs:
            try:
                selected = core.selected_parts_for_config(self.context, config_name)
            except Exception:
                continue
            part_ids.update(str(part_id) for part_id in selected.get("parts", ()))

        positions = self.context.node_positions
        by_key: dict[tuple, dict[str, object]] = {}
        for part_id in sorted(part_ids):
            found = core.part_body_for_context(self.context, part_id)
            if found is None:
                continue
            part_body = found[0]
            # One read of the part serves the rows, the twinned set and the
            # authored positions.
            rows = list(core.iter_trigger_rows(part_body))
            if not rows:
                continue
            twinned = core.twinned_trigger_ids(entry[0] for entry in rows)
            refs = {entry[0]: (entry[4][0] if entry[4] else "") for entry in rows}
            for trigger_id, (anchor, centre) in core.authored_trigger_placements(
                part_body, positions
            ).items():
                # Keyed on the anchor -- the raw authored value, so a saved
                # answer survives corrections to how a shape is read -- but
                # shown at the shape's middle, which is where it really is.
                key_at = core.trigger_position_key(trigger_id, anchor)
                if key_at is None:
                    continue
                entry = by_key.setdefault(key_at, {
                    "trigger": trigger_id,
                    "at": key_at[1],
                    "centre": centre,
                    "parts": set(),
                    "twinned": False,
                    "ref": "",
                })
                entry["parts"].add(part_id)
                if trigger_id in twinned:
                    entry["twinned"] = True
                if not entry["ref"]:
                    entry["ref"] = refs.get(trigger_id, "")

        counts: dict[str, int] = {}
        for trigger_id, _at in by_key:
            counts[trigger_id] = counts.get(trigger_id, 0) + 1
        ordinals: dict[str, int] = {}
        boxes: list[dict[str, object]] = []
        for key_at in sorted(by_key, key=lambda item: (item[0].lower(), item[1])):
            trigger_id, at = key_at
            entry = by_key[key_at]
            if counts[trigger_id] > 1:
                ordinals[trigger_id] = ordinals.get(trigger_id, 0) + 1
                label = f"{trigger_id} #{ordinals[trigger_id]}"
            else:
                label = trigger_id
            boxes.append({
                "id": f"{TRIGGER_ROW_PREFIX}|{trigger_id}|{at[0]}|{at[1]}|{at[2]}",
                "key": key_at,
                "trigger": trigger_id,
                "label": label,
                "at": at,
                "centre": entry.get("centre") or at,
                "parts": sorted(entry["parts"]),
                "twinned": bool(entry["twinned"]),
                "ref": str(entry["ref"]),
            })
        self._trigger_box_cache = (key, boxes)
        return boxes

    def _auto_action(self, box: dict[str, object], node_transforms) -> str:
        """The action the build would give this box if left unanswered.

        This has to agree with the build, not merely with the attribution
        ladder: a twinned pair is left alone before attribution is consulted
        at all, so reporting the ladder's verdict for one would promise a
        mirror the build never performs. An empty string means the ladder
        found no owner, which is not the same as deciding to skip.
        """
        if box.get("twinned"):
            return "skip"
        hit = node_transforms.get(str(box.get("ref") or ""))
        return str(hit[0]) if hit is not None else ""

    @staticmethod
    def _auto_label(action: str, twinned: bool) -> str:
        if twinned:
            return "Skip, twinned pair"
        return {
            "translate": "Move",
            "mirror": "Mirror",
            "mirrorPosition": "Mirror Move",
            "skip": "Skip",
        }.get(action, "unattributed")

    def _trigger_rows(self) -> list[dict[str, object]]:
        """The boxes with the user's answer and the automatic verdict attached.

        Both are cheap to read off the cached scan, so they are layered on
        here rather than baked into it.
        """
        boxes = self._trigger_boxes()
        if not boxes:
            return []
        chosen = core.trigger_mode_map(self.conversion)
        offsets = core.trigger_offset_map(self.conversion)
        owners = self._trigger_auto_owners()
        node_transforms = owners[1] if owners else {}
        rows: list[dict[str, object]] = []
        for box in boxes:
            action = self._auto_action(box, node_transforms)
            rows.append({
                **box,
                "mode": chosen.get(box["key"]),
                "offset": offsets.get(box["key"]),
                "auto_action": action,
                "auto": self._auto_label(action, bool(box.get("twinned"))),
            })
        return rows

    def _trigger_actions(self) -> dict[tuple, str]:
        """(trigger id, position) -> the action each box will receive.

        The preview needs this to draw a box where it will end up, and the
        answer costs an attribution-ladder run, so it is taken from the rows
        the table has already resolved rather than worked out again.
        """
        mode_actions = {
            core.MODE_SKIP: "skip",
            core.MODE_TRANSLATE: "translate",
            core.MODE_MIRROR: "mirror",
        }
        actions: dict[tuple, str] = {}
        for row in self._trigger_rows():
            mode = row.get("mode")
            actions[row["key"]] = (
                mode_actions.get(str(mode), "skip")
                if mode
                else str(row.get("auto_action") or "")
            )
        return actions

    def _trigger_label(self, row: dict[str, object]) -> str:
        at = row.get("centre") or row["at"]
        return f"{row['label']}  ({at[0]:+.2f}, {at[1]:+.2f}, {at[2]:+.2f})"

    def _trigger_mode_label(self, row: dict[str, object]) -> str:
        mode = row.get("mode")
        if mode is None:
            return f"(auto: {row.get('auto') or 'unattributed'})"
        return TRIGGER_MODE_LABELS.get(str(mode), "Skip")

    def _trigger_offset_display(self, row: dict[str, object]) -> str:
        """The box's Move X cell.

        Open on the transforms that move the box -- the whole distance for a
        Move, a nudge over the reflection for a Mirror. An auto row is left as
        "N/A" even where the ladder resolved one of those, because what it
        travels is the owning mesh's, and overriding it here would quietly
        detach the box from the mesh it was attributed to; choose the transform
        outright and the column opens up.
        """
        mode = str(row.get("mode") or "")
        return offset_display(
            mode if mode in core.TRIGGER_OFFSET_MODES else core.MODE_SKIP,
            row.get("offset"),
            manual_delta=self.manual_delta_enabled.get(),
        )

    def _trigger_auto_owners(self):
        """The cheap half of the attribution ladder, cached per selection.

        Prop pivots and node-group claims only: the enclosing-bounds rung reads
        every mesh's vertices, far too much work for a table refresh.
        """
        if self.context is None:
            return None
        configs = tuple(self._selected_variant_names())
        if not configs:
            return None
        modes = core.active_part_modes(self.conversion)
        key = (id(self.context), configs, hash(frozenset(modes.items())))
        cached = getattr(self, "trigger_owner_cache", None)
        if isinstance(cached, tuple) and cached[0] == key:
            return cached[1]
        try:
            selected = core.selected_parts_for_config(self.context, configs[0])
            owners = core.trigger_owners_for_config(
                self.context,
                selected,
                modes,
                {},
                core.HAND_RHD,
                self.context.node_positions,
            )
        except Exception:
            owners = ([], {}, [])
        self.trigger_owner_cache = (key, owners)
        return owners

    def _invalidate_trigger_cache(self) -> None:
        self.trigger_owner_cache = None
        self._trigger_box_cache = None

    def _refresh_triggers(self) -> None:
        with timed_ui("_refresh_triggers"):
            if not hasattr(self, "trigger_tree"):
                return
            keep = set(self.trigger_tree.selection())
            for item in self.trigger_tree.get_children():
                self.trigger_tree.delete(item)
            self.trigger_rows_by_iid = {}
            if self.context is None:
                return
            query = self.trigger_filter_var.get().strip().lower()
            for index, row in enumerate(self._trigger_rows()):
                label = self._trigger_label(row)
                mode = self._trigger_mode_label(row)
                if query and query not in label.lower() and query not in mode.lower():
                    continue
                row_id = str(row["id"])
                self.trigger_rows_by_iid[row_id] = row
                self.trigger_tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    tags=self._row_tags(index),
                    values=(label, mode, self._trigger_offset_display(row)),
                )
            visible_keep = [item for item in keep if self.trigger_tree.exists(item)]
            if visible_keep:
                self.trigger_tree.selection_set(visible_keep)

    def _claim_selection(self, owner: str) -> None:
        """Give the selection to one table and take it from the other.

        A mesh and a trigger are different kinds of thing with different
        detail panels and different hotkeys, so holding both at once leaves
        it ambiguous which one a keystroke is about. Clearing a treeview
        fires <<TreeviewSelect>>, which lands back here, so the guard stops
        the two tables handing the selection to each other forever.
        """
        if getattr(self, "_selection_sync", False):
            return
        self._selection_sync = True
        try:
            if owner != "parts" and hasattr(self, "part_tree"):
                if self.part_tree.selection():
                    self.part_tree.selection_set([])
            if owner != "triggers" and hasattr(self, "trigger_tree"):
                if self.trigger_tree.selection():
                    self.trigger_tree.selection_set([])
        finally:
            self._selection_sync = False

    def _clear_all_selection(self) -> None:
        """Nothing selected in either table -- an empty click in the preview."""
        if getattr(self, "_selection_sync", False):
            return
        self._selection_sync = True
        try:
            if hasattr(self, "part_tree") and self.part_tree.selection():
                self.part_tree.selection_set([])
            if hasattr(self, "trigger_tree") and self.trigger_tree.selection():
                self.trigger_tree.selection_set([])
        finally:
            self._selection_sync = False
        self._refresh_trigger_highlight()
        self._update_detail()

    def _trigger_selection_changed(self) -> None:
        # Only a table that actually holds a selection may claim it. Deleting
        # a selected row fires this event with an empty selection, so claiming
        # unconditionally let a rebuild of this table steal the selection off
        # whatever the user had just clicked in the other one.
        if self.trigger_tree.selection():
            self._claim_selection("triggers")
        self._refresh_trigger_highlight()

    def _refresh_trigger_highlight(self) -> None:
        """Push the trigger selection to the preview without a full rebuild."""
        viewer = getattr(self, "viewer", None)
        if viewer is None or not getattr(self, "viewer_supports_scene", False):
            return
        if not hasattr(viewer, "set_selected_ids"):
            return
        viewer.set_selected_ids(
            set(self._selected_preview_ids()) | self._selected_trigger_scene_ids()
        )

    def _trigger_click(self, event: tk.Event) -> str | None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.trigger_tree, event):
            return None
        item = self.trigger_tree.identify_row(event.y)
        row = getattr(self, "trigger_rows_by_iid", {}).get(item)
        if not item or row is None:
            return None
        column = self.trigger_tree.identify_column(event.x)
        name = self._tree_column_name(self.trigger_tree, column)
        self.trigger_tree.focus(item)
        self.trigger_tree.selection_set([item])
        if name == "mode":
            current = TRIGGER_MODE_LABELS.get(str(row.get("mode") or ""), "")

            def commit(value: str) -> None:
                self._set_trigger_mode(row, TRIGGER_MODE_BY_LABEL.get(value, core.MODE_SKIP))

            self._edit_tree_combo(
                self.trigger_tree, item, column, TRIGGER_MODE_OPTIONS, current, commit
            )
        elif name == "offset":
            if str(row.get("mode") or "") not in core.TRIGGER_OFFSET_MODES:
                self.status_var.set("Move X only applies to a box set to Move or Mirror")
                return "break"
            self._edit_tree_entry(
                self.trigger_tree,
                item,
                column,
                offset_label(row.get("offset")),
                lambda value: self._set_trigger_offset(row, value),
            )
        return "break"

    def _set_trigger_mode(self, row: dict[str, object], mode: str) -> None:
        core.set_trigger_mode(self.conversion, str(row["trigger"]), row["at"], mode)
        self._refresh_triggers()
        self._select_trigger_row(str(row["id"]))
        self._update_detail()
        self.status_var.set(
            f"{row['label']} set to {TRIGGER_MODE_LABELS.get(mode, mode)}"
        )

    def _set_trigger_offset(self, row: dict[str, object], value: str) -> None:
        cleaned = value.strip()
        if cleaned and core.trigger_offset_value(cleaned) is None:
            self._show_error(
                "Invalid offset",
                "Move X must be blank or a non-zero number. Positive moves the box the way this "
                "trim converts; negative moves it the opposite way. To leave a box exactly where "
                "it is, set its transform to Skip.",
            )
            return
        core.set_trigger_offset(self.conversion, str(row["trigger"]), row["at"], cleaned or None)
        self._refresh_triggers()
        self._select_trigger_row(str(row["id"]))
        self._update_detail()
        self.status_var.set(
            f"{row['label']} moves {offset_label(cleaned)}" if cleaned
            else f"{row['label']} back to the conversion delta"
        )

    def _set_selected_mode_shortcut(self, event: tk.Event, mode: str) -> str | None:
        """Send a transform hotkey to whichever table holds the selection.

        Q/W/E/R/T/Y run along the parts dropdown; the triggers table answers
        the first three of them, since a box only takes Skip, Move or Mirror.
        Exactly one table holds the selection at a time, so the trigger table
        gets first refusal and the parts table takes everything it declines.
        """
        handled = self._set_selected_trigger_mode_shortcut(event, mode)
        if handled is not None:
            return handled
        return self._set_selected_part_mode_shortcut(event, mode)

    def _set_selected_trigger_mode_shortcut(self, _event: tk.Event, mode: str) -> str | None:
        """Apply a transform hotkey to the selected box, or decline it.

        None means "not mine": either something is being typed into, or the
        trigger table holds no selection, and the parts table should answer.
        The selection is read rather than the focus, because the table keeps a
        focused row after the parts table has taken the selection off it.
        """
        if not hasattr(self, "trigger_tree"):
            return None
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in TYPING_WIDGET_CLASSES:
            return None
        selection = self.trigger_tree.selection()
        row = getattr(self, "trigger_rows_by_iid", {}).get(selection[0]) if selection else None
        if row is None:
            return None
        if mode not in TRIGGER_MODE_LABELS:
            self.status_var.set("A trigger box takes only Skip (Q), Move (W) or Mirror (E)")
            return "break"
        self._set_trigger_mode(row, mode)
        return "break"

    def _selected_trigger_scene_ids(self) -> set[str]:
        """The scene group names of the currently selected trigger rows."""
        if not hasattr(self, "trigger_tree"):
            return set()
        return {
            str(item)
            for item in self.trigger_tree.selection()
            if item in getattr(self, "trigger_rows_by_iid", {})
        }

    def _select_trigger_row(self, row_id: str) -> None:
        if hasattr(self, "trigger_tree") and self.trigger_tree.exists(row_id):
            self.trigger_tree.selection_set([row_id])
            self.trigger_tree.focus(row_id)
            self.trigger_tree.see(row_id)

    def _selected_trigger_row(self) -> dict[str, object] | None:
        if not hasattr(self, "trigger_tree"):
            return None
        selection = self.trigger_tree.selection()
        item = str(selection[0]) if selection else str(self.trigger_tree.focus() or "")
        return getattr(self, "trigger_rows_by_iid", {}).get(item)

    def _reset_selected_trigger(self) -> None:
        """Hand one box back to the automatic attribution."""
        row = self._selected_trigger_row()
        if row is None:
            return
        core.clear_trigger_mode(self.conversion, str(row["trigger"]), row["at"])
        self._refresh_triggers()
        self._select_trigger_row(str(row["id"]))
        self._update_detail()
        self.status_var.set(f"{row['label']} back to the recommendation")

    def _clear_trigger_modes(self) -> None:
        core.clear_trigger_modes(self.conversion)
        self._refresh_triggers()
        self._update_detail()
        self.status_var.set("Cleared every trigger override")

    def _focus_trigger_table_shortcut(self, event: tk.Event) -> str | None:
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in {"Entry", "TEntry", "TCombobox", "Text"}:
            return None
        if hasattr(self, "trigger_tree"):
            self.trigger_tree.focus_set()
            children = self.trigger_tree.get_children()
            if children and not self.trigger_tree.selection():
                self.trigger_tree.selection_set([children[0]])
                self.trigger_tree.focus(children[0])
        return "break"
