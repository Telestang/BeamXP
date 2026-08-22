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
# Tk reports keyboard modifiers as bits in event.state.
SHIFT_MASK = 0x0001
CONTROL_MASK = 0x0004
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
        """The transform the build will give this box if left unanswered.

        This has to agree with the build, not merely with the attribution
        ladder: a twinned pair is left alone before attribution is consulted
        at all, so reporting the ladder's verdict for one would promise a
        mirror the build never performs.

        A box the ladder finds no owner for is reported as Skip, because that
        is what the build does with it -- ``part_has_relocatable_trigger``
        finds no transforming owner and leaves the box exactly where it was.
        Naming that outcome after the reason for it said nothing the user
        could act on and named a transform that does not exist.
        """
        if box.get("twinned"):
            return core.MODE_SKIP
        hit = node_transforms.get(str(box.get("ref") or ""))
        return str(hit[0]) if hit is not None else core.MODE_SKIP

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
            rows.append({
                **box,
                "mode": chosen.get(box["key"]),
                "offset": offsets.get(box["key"]),
                "auto_action": self._auto_action(box, node_transforms),
            })
        return rows

    @staticmethod
    def _effective_trigger_mode(row: dict[str, object]) -> str:
        """The transform this box will receive: the user's, or the prediction.

        The ladder reports its verdict using the same names the modes carry,
        so an unanswered row needs no translation -- and, since a prediction
        is only ever Skip, Move, Mirror or Mirror Move, the result is always a
        real transform rather than a placeholder for "no answer yet".
        """
        mode = row.get("mode")
        return str(mode) if mode is not None else str(row.get("auto_action") or core.MODE_SKIP)

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
        """The Trigger cell: the box's name alone.

        Rows are keyed by position, but the position is not shown: two boxes at
        different places are already told apart by the "#1"/"#2" the row label
        carries, and the coordinates only cost the column the width its names
        need. Click a row and the preview says where it is.
        """
        return str(row["label"])

    def _trigger_mode_label(self, row: dict[str, object]) -> str:
        """The Transform cell: what the box will do, chosen or predicted.

        A row the user has not answered reads as the plain transform it will
        get, not as an announcement that nobody has answered it. The table's
        job is to state what the build will do; disagreeing with it is what
        picking a transform is for.
        """
        return mode_label(self._effective_trigger_mode(row))

    def _trigger_offset_display(self, row: dict[str, object]) -> str:
        """The box's Move X cell.

        Open on the transforms that move the box -- the whole distance for a
        Move, a nudge over the reflection for a Mirror. A row left on the
        prediction reads "N/A" even where that prediction is one of those,
        because the distance it travels is the owning mesh's: this box has no
        Move X of its own to show, and writing one here would quietly detach it
        from the mesh it was matched to. Choose the transform outright -- which
        is what tells the build to stop following that mesh -- and the column
        opens up.
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

    @staticmethod
    def _toggle_tree_selection(tree, item: str) -> None:
        """Ctrl-click semantics for a row picked in the 3D preview: add it to
        the selection, or drop it if it was already there, so a mis-pick is
        undone by picking it again rather than by starting over."""
        current = list(tree.selection())
        if item in current:
            tree.selection_set([row for row in current if row != item])
        else:
            tree.selection_add(item)

    def _note_selection_modifier(self, event: tk.Event) -> None:
        """Record whether the click about to land is holding Ctrl or Shift.

        Bound ahead of the Treeview's own class binding, so the flag is set
        before <<TreeviewSelect>> fires and ``_claim_selection`` reads it.
        """
        try:
            state = int(getattr(event, "state", 0) or 0)
        except (TypeError, ValueError):
            state = 0
        self._selection_accumulating = bool(state & (SHIFT_MASK | CONTROL_MASK))

    def _claim_selection(self, owner: str, item: str = "") -> None:
        """Give the selection to the table being clicked, at click time.

        Deliberately driven by the click and NOT by <<TreeviewSelect>>. Tk
        *queues* that event rather than sending it: a handler runs after the
        call that changed the selection has returned, so any "we are only
        rebuilding" flag set around the change is already back off by the time
        the handler reads it. A rebuild's own re-selection is then
        indistinguishable from a click, and clearing from the handler took the
        other table down with every refresh. Deciding here removes the timing
        question rather than trying to win it.

        A plain click starts fresh, so the other table is cleared: a transform
        must never land on rows still selected in a table the user has stopped
        looking at. Ctrl- and Shift-click say "add to what I have", so both
        tables keep their rows -- which is the point, because a shifter's
        meshes and the boxes that label them want the same Move and the same
        Move X in one edit.

        A click that lands *inside* a selection this table already holds is an
        edit of that selection, not a new one, so it leaves the other table
        alone. That is what lets Move X be typed into one row of a gathered
        set without dropping the rest of it.
        """
        if getattr(self, "_selection_accumulating", False):
            return
        own = getattr(self, "part_tree" if owner == "parts" else "trigger_tree", None)
        if own is not None and item and own.exists(item) and item in own.selection():
            return
        other = getattr(self, "trigger_tree" if owner == "parts" else "part_tree", None)
        if other is not None and other.selection():
            other.selection_set([])

    def _clear_all_selection(self) -> None:
        """Nothing selected in either table -- an empty click in the preview.

        No re-entrancy guard needed: neither table's <<TreeviewSelect>> handler
        changes a selection any more, so clearing one cannot come back here.
        """
        if hasattr(self, "part_tree") and self.part_tree.selection():
            self.part_tree.selection_set([])
        if hasattr(self, "trigger_tree") and self.trigger_tree.selection():
            self.trigger_tree.selection_set([])
        self._refresh_trigger_highlight()
        self._update_detail()

    def _trigger_selection_changed(self) -> None:
        # Nothing here decides who owns the selection. Tk queues this event, so
        # it arrives indistinguishable from the one a rebuild's re-selection
        # produces; _claim_selection runs from the click instead.
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
        self._note_selection_modifier(event)
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.trigger_tree, event):
            return None
        item = self.trigger_tree.identify_row(event.y)
        row = getattr(self, "trigger_rows_by_iid", {}).get(item)
        if not item or row is None:
            return None
        self._claim_selection("triggers", item)
        column = self.trigger_tree.identify_column(event.x)
        name = self._tree_column_name(self.trigger_tree, column)
        if name not in {"mode", "offset"}:
            # The Trigger column is the one you gather rows by, so the click has
            # to reach the Treeview's own class binding -- that binding is what
            # gives Ctrl- and Shift-click their meaning. Returning "break" here
            # swallowed it and left the table single-select whatever its
            # selectmode said.
            return None
        self.trigger_tree.focus(item)
        # An editor opening on a row that is already part of a gathered
        # selection must not throw the rest of it away -- that edit is meant for
        # all of them. Only a click on an unselected row starts fresh.
        if item not in self.trigger_tree.selection():
            self.trigger_tree.selection_set([item])
        if name == "mode":
            # Opens on what the cell shows, prediction included, so the list
            # starts from the transform in force rather than from blank.
            current = TRIGGER_MODE_LABELS.get(self._effective_trigger_mode(row), "")

            def commit(value: str) -> None:
                self._set_trigger_mode(row, TRIGGER_MODE_BY_LABEL.get(value, core.MODE_SKIP))

            self._edit_tree_combo(
                self.trigger_tree, item, column, TRIGGER_MODE_OPTIONS, current, commit
            )
        elif name == "offset":
            if str(row.get("mode") or "") not in core.TRIGGER_OFFSET_MODES:
                self.status_var.set(
                    "This box travels with the mesh it was matched to. Pick its transform "
                    "yourself to give it a Move X of its own."
                    if row.get("mode") is None
                    else "Move X only applies to a box set to Move or Mirror"
                )
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
        """The Triggers dropdown. Speaks for the whole selection, as the parts
        one does; the row it was opened on is one of the rows it applies to."""
        self._apply_mode_to_selection(mode)

    def _set_trigger_offset(self, row: dict[str, object], value: str) -> None:
        self._apply_offset_to_selection(value)

    def _apply_offset_to_selection(self, value: str) -> None:
        """Set one Move X on everything selected, in both tables at once.

        Rows whose transform has no distance to take -- a Skip, or a box left
        on the prediction, which travels with the mesh it was matched to --
        are passed over rather than being quietly given a mode they were never
        set to, and counted in the message so a silent no-op cannot look like
        a change.
        """
        cleaned = value.strip()
        if cleaned:
            try:
                parsed = float(cleaned)
            except ValueError:
                parsed = None
            if parsed is None or parsed != parsed or parsed in (float("inf"), float("-inf")):
                self._show_error(
                    "Invalid offset",
                    "Move X must be blank or a number. Positive moves the way this trim converts; "
                    "negative moves it the opposite way.",
                )
                return
            if parsed == 0:
                # Stored on a mesh this fails the build later with "Delta X
                # magnitude is zero"; on a box it is silently no override. Say
                # so now instead, and point at the two things that mean it.
                self._show_error(
                    "Invalid offset",
                    "Move X cannot be zero. Leave it blank to use the conversion delta, or set "
                    "the transform to Skip to leave the part exactly where it is.",
                )
                return

        stored = float(cleaned) if cleaned else None
        targets, _rows = self._selected_part_mode_targets()
        meshes = [
            object_id
            for object_id in targets
            if str(self._get_part_setting(object_id, "mode", core.MODE_SKIP)) in OFFSET_MODES
        ]
        trigger_rows = self._selected_trigger_rows()
        boxes = [
            row for row in trigger_rows
            if str(row.get("mode") or "") in core.TRIGGER_OFFSET_MODES
        ]
        if not meshes and not boxes:
            self.status_var.set("Move X applies only to rows set to Move or Mirror")
            return

        for object_id in meshes:
            self._part_settings(object_id)["translateOffset"] = stored
        for row in boxes:
            core.set_trigger_offset(self.conversion, str(row["trigger"]), row["at"], stored)
        if boxes:
            self._refresh_triggers()
            self._reselect_trigger_rows(boxes)
        if meshes:
            self._refresh_parts()
            self._refresh_delta_label()
        self._update_detail()
        what = f"Move X {offset_label(cleaned)}" if cleaned else "Move X cleared"
        self.status_var.set(
            self._selection_change_message(
                what, len(meshes), len(boxes), len(trigger_rows) - len(boxes)
            )
        )

    def _selected_trigger_rows(self) -> list[dict[str, object]]:
        """Every selected trigger row, in table order."""
        if not hasattr(self, "trigger_tree"):
            return []
        by_iid = getattr(self, "trigger_rows_by_iid", {})
        return [row for item in self.trigger_tree.selection() if (row := by_iid.get(item))]

    def _set_selected_mode_shortcut(self, _event: tk.Event, mode: str) -> str | None:
        """The Q/W/E/R/T/Y entry point. Stands down for whatever is being typed
        into; the dropdowns reach ``_apply_mode_to_selection`` directly, since
        the combobox committing the edit is itself a typing widget."""
        focus = self.focus_get()
        # Only typing targets swallow the hotkeys; buttons and other focusable
        # widgets don't react to letter keys, so mode setting stays live.
        if focus is not None and focus.winfo_class() in TYPING_WIDGET_CLASSES:
            return None
        return self._apply_mode_to_selection(mode)

    def _apply_mode_to_selection(self, mode: str) -> str | None:
        """Set one transform on everything selected, in both tables at once.

        Q/W/E/R/T/Y run along the parts dropdown. The two tables no longer take
        turns: a Ctrl-click can gather a shifter's meshes and the trigger boxes
        that label them, and one keystroke moves the lot. Where the two
        disagree about a mode -- a box takes only Skip, Move or Mirror -- the
        rows that can take it still do, and the message says what stood out.
        """
        if self.context is None:
            return None
        targets, target_rows = self._selected_part_mode_targets()
        trigger_rows = self._selected_trigger_rows()
        if not targets and not trigger_rows:
            if hasattr(self, "part_tree") and any(
                self._part_child_override(row_id) for row_id in self.part_tree.selection()
            ):
                self.status_var.set(
                    "Effective source/replaced rows are controlled by their Replace Source parent"
                )
                return "break"
            return None

        # Swap Mesh and Replace Source are pairings rather than settings: each
        # needs a partner picked for the one row it is set on, so they stay
        # single-target and never reach a trigger box.
        if mode in {core.MODE_MIRROR_STRUCTURAL, core.MODE_REPLACE_SOURCE}:
            noun = "mesh" if mode == core.MODE_MIRROR_STRUCTURAL else "part"
            if len(targets) != 1 or trigger_rows:
                self.status_var.set(
                    f"Select one {noun} to set {mode_label(mode)}; it needs a source {noun}"
                )
                return "break"
            if mode == core.MODE_MIRROR_STRUCTURAL:
                self._set_part_mode(targets[0], mode)
            else:
                self._set_part_mode(targets[0], mode, row_id=target_rows[0])
            return "break"

        takeable = [row for row in trigger_rows if mode in TRIGGER_MODE_LABELS]
        if not targets and not takeable:
            self.status_var.set("A trigger box takes only Skip (Q), Move (W) or Mirror (E)")
            return "break"

        if targets:
            self._apply_part_modes(targets, mode)
        for row in takeable:
            core.set_trigger_mode(self.conversion, str(row["trigger"]), row["at"], mode)
        if takeable:
            self._refresh_triggers()
            self._reselect_trigger_rows(takeable)
        if targets:
            self._refresh_parts()
            self._refresh_delta_label()
        self._update_detail()
        self.status_var.set(
            self._selection_change_message(
                mode_label(mode), len(targets), len(takeable), len(trigger_rows) - len(takeable)
            )
        )
        return "break"

    @staticmethod
    def _selection_change_message(
        what: str,
        meshes: int,
        triggers: int,
        refused_triggers: int = 0,
    ) -> str:
        parts = []
        if meshes:
            parts.append(f"{meshes} mesh(es)")
        if triggers:
            parts.append(f"{triggers} trigger(s)")
        message = f"{what} on {' and '.join(parts)}" if parts else f"{what}: nothing to change"
        if refused_triggers:
            message += f"; {refused_triggers} trigger(s) cannot take it"
        return message

    def _reselect_trigger_rows(self, rows: list[dict[str, object]]) -> None:
        """Put the selection back after a refresh rebuilt the table's rows."""
        if not hasattr(self, "trigger_tree"):
            return
        wanted = [str(row["id"]) for row in rows if self.trigger_tree.exists(str(row["id"]))]
        if not wanted:
            return
        self.trigger_tree.selection_set(wanted)
        self.trigger_tree.focus(wanted[0])
        self.trigger_tree.see(wanted[0])

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

    def _toggle_triggers_visible(self) -> None:
        """Show or hide every trigger box in the preview.

        The boxes sit over the cabin geometry they label, so they are worth
        getting out of the way while a mesh underneath is being judged. Only
        the drawing changes: the rows, their transforms and the build are
        untouched, and a hidden box stops being pickable rather than becoming
        an invisible click target.
        """
        self.triggers_visible = not self.triggers_visible
        self._apply_trigger_visibility()
        self.status_var.set(
            "Trigger boxes shown" if self.triggers_visible else "Trigger boxes hidden"
        )

    def _apply_trigger_visibility(self) -> None:
        """Push the toggle to the button label and the viewer."""
        if hasattr(self, "toggle_triggers_button"):
            self.toggle_triggers_button.configure(
                text="Hide" if self.triggers_visible else "Show"
            )
        viewer = getattr(self, "viewer", None)
        # The box-viewer fallback draws no trigger boxes at all, so it has
        # nothing to switch off.
        if viewer is not None and hasattr(viewer, "set_triggers_visible"):
            viewer.set_triggers_visible(self.triggers_visible)

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
