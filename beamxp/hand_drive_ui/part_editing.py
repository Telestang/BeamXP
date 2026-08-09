from __future__ import annotations

from .shared import *


class PartEditingMixin:
    """Part-table interaction, inline editors, modes, offsets, visibility, and manual delta editing."""

    def _part_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.part_tree, event):
            return None
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if not item:
            return None
        if isinstance(getattr(self, "side_pair_pick_target", None), dict):
            self.part_tree.selection_set([item])
            self.part_tree.focus(item)
            self.part_tree.see(item)
            self._commit_side_pair_part_pick_from_row(item)
            return "break"
        object_id = self._part_row_mesh_id(item)
        name = self._tree_column_name(self.part_tree, column)
        child_override = self._part_child_override(item)
        if child_override and name in {"mode", "source", "children", "offset"}:
            self.status_var.set(self._part_override_status_label(child_override))
            return "break"
        if name == "visible":
            self._toggle_part_bool(object_id, "viewerVisible", default=True)
            return "break"
        if name == "solo":
            self._toggle_part_bool(object_id, "viewerSolo")
            return "break"
        if name == "mode":
            self.part_tree.focus(item)
            self.part_tree.selection_set(item)
            current = mode_label(str(self._get_part_setting(object_id, "mode", core.MODE_SKIP)))
            replace_label = mode_label(core.MODE_REPLACE_SOURCE)
            disabled = set()
            if not self._replace_source_candidate_ids(object_id, item):
                disabled.add(replace_label)
            self._edit_tree_combo(
                self.part_tree,
                item,
                column,
                [mode_label(mode) for mode in self._mode_values_for_part_row(item)],
                current,
                lambda value: self._set_part_row_mode_from_label(item, value),
                disabled_values=disabled,
                disabled_message="Replace Source needs another part that fits this slot",
            )
            return "break"
        if name == "textureCorrection":
            self._toggle_texture_correction(object_id)
            return "break"
        if name == "source":
            mode = self._get_part_setting(object_id, "mode", core.MODE_SKIP)
            if mode == core.MODE_MIRROR_STRUCTURAL:
                source_id = self._choose_structural_source(object_id)
            elif mode == core.MODE_REPLACE_SOURCE:
                candidates = self._replace_source_candidate_ids(object_id, item)
                existing = str(self._get_part_setting(object_id, "mirrorSource", "") or "")
                if existing and existing in self.context.objects and existing != object_id and existing not in candidates:
                    candidates.append(existing)
                if not candidates:
                    self.status_var.set("Replace Source needs another part that fits this slot")
                    return "break"
                current = self._swap_source_label(object_id)
                label_universe = [object_id, *candidates]
                value_by_label = {
                    self._part_option_label(candidate, label_universe): candidate
                    for candidate in candidates
                }
                label_by_value = {value: label for label, value in value_by_label.items()}
                self._edit_tree_combo(
                    self.part_tree,
                    item,
                    column,
                    list(value_by_label),
                    label_by_value.get(existing, current if current in value_by_label else next(iter(value_by_label))),
                    lambda value: self._set_replace_source(object_id, value_by_label[value]),
                )
                return "break"
            else:
                self.status_var.set("Source only applies to Swap Mesh or Replace Source")
                return "break"
            if source_id:
                if mode == core.MODE_MIRROR_STRUCTURAL:
                    self._set_structural_pair(object_id, source_id)
            return "break"
        if name == "children":
            self._toggle_include_children(object_id)
            return "break"
        if name == "offset":
            if self._get_part_setting(object_id, "mode", core.MODE_SKIP) != core.MODE_TRANSLATE:
                self.status_var.set("Offset X only applies to Move mode")
                return "break"
            self._edit_tree_entry(
                self.part_tree,
                item,
                column,
                offset_label(self._get_part_setting(object_id, "translateOffset", None)),
                lambda value: self._set_part_offset(object_id, value),
            )
            return "break"
        if name == "steering":
            self._set_single_steering_ref(object_id)
            return "break"
        # "active" is read-only, and #0/coords fall through to default row select.
        return None

    def _part_double_click(self, event: tk.Event) -> None:
        if not self._tree_body_click(self.part_tree, event):
            return None
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if not item:
            return
        object_id = self._part_row_mesh_id(item)
        name = self._tree_column_name(self.part_tree, column)
        child_override = self._part_child_override(item)
        if child_override and name in {"mode", "source", "children", "offset"}:
            self.status_var.set(self._part_override_status_label(child_override))
            return "break"
        if name == "mode":
            return "break"
        elif name == "offset":
            if self._get_part_setting(object_id, "mode", core.MODE_SKIP) != core.MODE_TRANSLATE:
                self.status_var.set("Offset X only applies to Move mode")
                return
            self._edit_tree_entry(
                self.part_tree,
                item,
                column,
                offset_label(self._get_part_setting(object_id, "translateOffset", None)),
                lambda value: self._set_part_offset(object_id, value),
            )

    def _set_part_mode_from_label(self, object_id: str, label: str) -> None:
        mode = MODE_VALUES_BY_LABEL.get(label)
        if mode is None:
            return
        self._set_part_mode(object_id, mode)
        if mode != core.MODE_MIRROR_STRUCTURAL:
            self.status_var.set(f"{self._part_display_name(object_id)}: {mode_label(mode)}")

    def _set_part_row_mode_from_label(self, row_id: str, label: str) -> None:
        object_id = self._part_row_mesh_id(row_id)
        mode = MODE_VALUES_BY_LABEL.get(label)
        if mode is None:
            return
        if mode == core.MODE_REPLACE_SOURCE and not self._replace_source_candidate_ids(object_id, row_id):
            self.status_var.set("Replace Source needs another part that fits this slot")
            return
        self._set_part_mode(object_id, mode, row_id=row_id)
        if mode not in {core.MODE_MIRROR_STRUCTURAL, core.MODE_REPLACE_SOURCE}:
            self.status_var.set(f"{self._part_display_name(object_id)}: {mode_label(mode)}")

    def _cancel_structural_prompt(self, object_id: str | None = None) -> None:
        if object_id is not None and self.structural_prompt_part_id != object_id:
            return
        if self.structural_prompt_after_id is not None:
            try:
                self.after_cancel(self.structural_prompt_after_id)
            except tk.TclError:
                pass
        self.structural_prompt_after_id = None
        self.structural_prompt_part_id = None
        self.structural_prompt_previous_mode = core.MODE_SKIP

    def _schedule_structural_prompt(self, object_id: str, previous_mode: str) -> None:
        self._cancel_structural_prompt()
        self.structural_prompt_part_id = object_id
        self.structural_prompt_previous_mode = (
            previous_mode if previous_mode in MODE_CYCLE_VALUES else core.MODE_SKIP
        )
        self.structural_prompt_after_id = self.after(
            STRUCTURAL_PROMPT_DELAY_MS,
            self._trigger_structural_prompt,
        )
        self.status_var.set(
            f"Swap Mesh selected for {self._part_display_name(object_id)}; choose a source mesh"
        )

    def _trigger_structural_prompt(self) -> None:
        object_id = self.structural_prompt_part_id
        previous_mode = self.structural_prompt_previous_mode
        if object_id is None or self.structural_prompt_open:
            return
        self._cancel_structural_prompt(object_id)
        if self.context is None:
            return
        settings = self._part_settings(object_id)
        if settings.get("mode") != core.MODE_MIRROR_STRUCTURAL or settings.get("mirrorSource"):
            return

        self.structural_prompt_open = True
        try:
            source_id = self._choose_structural_source(object_id)
        finally:
            self.structural_prompt_open = False

        settings = self._part_settings(object_id)
        if settings.get("mode") != core.MODE_MIRROR_STRUCTURAL:
            return
        if source_id:
            self._set_structural_pair(object_id, source_id)
            return

        restore_mode = previous_mode if previous_mode != core.MODE_MIRROR_STRUCTURAL else core.MODE_SKIP
        settings["mode"] = restore_mode
        settings["mirrorSource"] = None
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"Swap Mesh cancelled for {self._part_display_name(object_id)}; restored {mode_label(restore_mode)}"
        )

    def _edit_tree_combo(
        self,
        tree: ttk.Treeview,
        item: str,
        column: str,
        values: list[str],
        current: str,
        on_commit,
        *,
        disabled_values: set[str] | None = None,
        disabled_message: str = "",
    ) -> None:
        self._close_tree_combo_editor()
        if not tree.exists(item):
            return
        disabled_values = disabled_values or set()
        tree.see(item)
        bbox = tree.bbox(item, column)
        if not bbox:
            return
        x, y, width, height = bbox
        combo = ttk.Combobox(
            tree,
            values=values,
            state="readonly",
            exportselection=False,
            height=min(max(len(values), 1), 15),
        )
        combo.set(current)
        combo.place(x=x, y=y, width=width, height=height)
        tree.focus(item)
        tree.selection_set(item)
        self._tree_combo_editor = combo
        combo.focus_set()

        def commit(_event=None) -> None:
            if self._tree_combo_editor is not combo:
                return "break"
            value = combo.get()
            self._close_tree_combo_editor()
            if value in disabled_values:
                tree.focus_set()
                if disabled_message:
                    self.status_var.set(disabled_message)
                return "break"
            on_commit(value)
            return "break"

        def cancel(_event=None) -> str:
            if self._tree_combo_editor is combo:
                self._close_tree_combo_editor()
                tree.focus_set()
            return "break"

        def check_focus() -> None:
            self._tree_combo_focus_after_id = None
            if self._tree_combo_editor is not combo:
                return
            try:
                focus_path = str(self.tk.call("focus"))
                combo_path = str(combo)
                # The popdown list is a separate Tk window, but its Tcl path is
                # rooted below the combobox.  Keep checking until it unposts so
                # <<ComboboxSelected>> gets the first chance to commit.
                if focus_path.startswith(combo_path + ".") or combo.instate(("pressed",)):
                    self._tree_combo_focus_after_id = self.after(50, check_focus)
                    return
                # If the list was dismissed without a selection, ttk returns
                # focus to the combobox.  Treat that as cancellation; a real
                # selection has already emitted <<ComboboxSelected>> by now.
                if focus_path == combo_path:
                    self._close_tree_combo_editor()
                    return
            except tk.TclError:
                return
            self._close_tree_combo_editor()

        def focus_out(_event=None) -> None:
            if self._tree_combo_focus_after_id is not None:
                try:
                    self.after_cancel(self._tree_combo_focus_after_id)
                except tk.TclError:
                    pass
            self._tree_combo_focus_after_id = self.after(50, check_focus)

        def post_dropdown() -> None:
            if self._tree_combo_editor is not combo or not combo.winfo_exists():
                return
            combo.focus_set()
            # Drive the public mouse bindings instead of calling Tk's private
            # ttk::combobox::Post command.  This gives an in-cell DDL genuine
            # one-click behaviour while retaining native theme handling.
            arrow_x = max(1, combo.winfo_width() - 4)
            arrow_y = max(1, combo.winfo_height() // 2)
            combo.event_generate("<ButtonPress-1>", x=arrow_x, y=arrow_y)
            combo.event_generate("<ButtonRelease-1>", x=arrow_x, y=arrow_y)
            self.after(30, mark_disabled_items)

        def mark_disabled_items() -> None:
            if not disabled_values or self._tree_combo_editor is not combo:
                return
            try:
                popdown = str(combo.tk.call("ttk::combobox::PopdownWindow", combo))
                listbox = f"{popdown}.f.l"
                if not int(combo.tk.call("winfo", "exists", listbox)):
                    return
                for index, value in enumerate(values):
                    if value in disabled_values:
                        combo.tk.call(listbox, "itemconfigure", index, "-foreground", "#8a8a8a")
            except tk.TclError:
                return

        combo.bind("<<ComboboxSelected>>", commit)
        combo.bind("<Return>", commit)
        combo.bind("<KP_Enter>", commit)
        combo.bind("<Tab>", commit)
        combo.bind("<Escape>", cancel)
        combo.bind("<FocusOut>", focus_out)
        self.after_idle(post_dropdown)

    def _close_tree_combo_editor(self) -> None:
        if self._tree_combo_focus_after_id is not None:
            try:
                self.after_cancel(self._tree_combo_focus_after_id)
            except tk.TclError:
                pass
            self._tree_combo_focus_after_id = None
        combo = self._tree_combo_editor
        self._tree_combo_editor = None
        if combo is not None:
            try:
                combo.destroy()
            except tk.TclError:
                pass

    def _edit_tree_entry(
        self,
        tree: ttk.Treeview,
        item: str,
        column: str,
        current: str,
        on_commit,
    ) -> None:
        bbox = tree.bbox(item, column)
        if not bbox:
            return
        x, y, width, height = bbox
        entry = ttk.Entry(tree)
        entry.insert(0, current)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()
        entry.selection_range(0, tk.END)

        committed = {"done": False}

        def commit(_event=None) -> None:
            if committed["done"]:
                return
            committed["done"] = True
            value = entry.get()
            entry.destroy()
            on_commit(value)

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        def cancel(_event=None) -> None:
            committed["done"] = True
            entry.destroy()

        entry.bind("<Escape>", cancel)

    def _get_variant_setting(self, config_name: str, key: str, default: object) -> object:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if not isinstance(settings, dict):
            return default
        return settings.get(key, default)

    def _set_variant_setting(self, config_name: str, key: str, value: object) -> None:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            settings[key] = value
        self._refresh_variants()
        self._refresh_delta_label()
        self._update_detail()

    def _get_part_setting(self, object_id: str, key: str, default: object) -> object:
        object_id = self._part_row_mesh_id(object_id)
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if not isinstance(settings, dict):
            return default
        return settings.get(key, default)

    def _refresh_part_viewer_cells(self, object_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        parts = self.conversion.get("parts", {})
        if not isinstance(parts, dict):
            return
        for object_id in object_ids:
            mesh_id = self._part_row_mesh_id(object_id)
            row_ids = [
                row_id
                for row_id in self.part_tree.get_children()
                if row_id == object_id or self._part_row_mesh_id(row_id) == mesh_id
            ]
            if not row_ids:
                continue
            settings = parts.get(mesh_id)
            if not isinstance(settings, dict):
                settings = {}
            for row_id in row_ids:
                self.part_tree.set(row_id, "visible", yn_label(settings.get("viewerVisible", True)))
                self.part_tree.set(row_id, "solo", yn_label(settings.get("viewerSolo")))

    def _refresh_part_texture_correction_cells(
        self,
        object_ids: list[str] | tuple[str, ...] | set[str],
    ) -> None:
        parts = self.conversion.get("parts", {})
        if not isinstance(parts, dict):
            return
        for object_id in object_ids:
            mesh_id = self._part_row_mesh_id(object_id)
            row_ids = [
                row_id
                for row_id in self.part_tree.get_children()
                if row_id == object_id or self._part_row_mesh_id(row_id) == mesh_id
            ]
            settings = parts.get(mesh_id)
            enabled = bool(settings.get(core.PART_TEXTURE_CORRECTION_KEY)) if isinstance(settings, dict) else False
            for row_id in row_ids:
                self.part_tree.set(row_id, "textureCorrection", yn_label(enabled))

    def _toggle_part_bool(self, object_id: str, key: str, *, default: bool = False) -> None:
        object_id = self._part_row_mesh_id(object_id)
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if isinstance(settings, dict):
            settings[key] = not bool(settings.get(key, default))
        if key in {"viewerVisible", "viewerSolo"}:
            self._refresh_part_viewer_cells([object_id])
            self._refresh_viewer()
            self._update_detail()
            return
        if key == "steeringRef":
            self._refresh_variants()
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()

    def _toggle_texture_correction(self, object_id: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        settings = self._part_settings(object_id)
        enabled = not bool(settings.get(core.PART_TEXTURE_CORRECTION_KEY))
        settings[core.PART_TEXTURE_CORRECTION_KEY] = enabled
        self._refresh_part_texture_correction_cells([object_id])
        self._refresh_derived_output_summary()
        self._update_detail()
        self.status_var.set(
            f"{self._part_display_name(object_id)} texture correction {'enabled' if enabled else 'disabled'}"
        )

    def _toggle_include_children(self, object_id: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        settings = self._part_settings(object_id)
        if settings.get("mode") != core.MODE_REPLACE_SOURCE:
            self.status_var.set("Children only applies to Replace Source")
            return
        include_children = not bool(settings.get("includeChildren"))
        settings["includeChildren"] = include_children
        source_id = str(settings.get("mirrorSource") or "")
        source_settings = self._part_settings(source_id) if source_id else None
        if (
            source_settings is not None
            and source_settings.get("mode") == core.MODE_REPLACE_SOURCE
            and str(source_settings.get("mirrorSource") or "") == object_id
        ):
            source_settings["includeChildren"] = include_children
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"{self._part_display_name(object_id)} child parts "
            f"{'included' if include_children else 'not included'}"
        )

    def _set_single_steering_ref(self, object_id: str) -> None:
        if self.context is None:
            return
        object_id = self._part_row_mesh_id(object_id)
        parts = self.conversion.setdefault("parts", {})
        was_selected = bool(self._get_part_setting(object_id, "steeringRef", False))
        for part_id in list(parts):
            settings = parts.get(part_id)
            if isinstance(settings, dict):
                settings["steeringRef"] = False
        settings = self._part_settings(object_id)
        settings["steeringRef"] = not was_selected
        self._invalidate_variant_detection()
        self._refresh_variants()
        self._schedule_variant_detection()
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()
        if settings["steeringRef"]:
            self.status_var.set(f"Steering reference set: {self._part_display_name(object_id)}")
        else:
            self.status_var.set("Steering reference cleared")

    def _set_all_parts_visible(self, visible: bool) -> None:
        if self.context is None:
            return
        object_ids = [
            self._part_row_mesh_id(row_id)
            for row_id in self.current_part_ids
            if self._part_row_mesh_id(row_id) in self.context.objects
        ]
        object_ids = sorted(set(object_ids))
        if not object_ids:
            if self.resolved_part_ids:
                self.status_var.set("No displayed parts match the current filter")
            else:
                self.status_var.set("No used parts are loaded yet")
            return
        for object_id in object_ids:
            settings = self._part_settings(object_id)
            settings["viewerVisible"] = visible
            settings["viewerSolo"] = False
        self._refresh_part_viewer_cells(object_ids)
        self._refresh_viewer()
        self._update_detail()
        state = "visible" if visible else "hidden"
        scope = "displayed" if self.filter_var.get().strip() else "used"
        self.status_var.set(f"Set {len(object_ids)} {scope} part(s) {state}; cleared solo flags")

    def _show_active_parts_only(self) -> None:
        if self.context is None:
            return
        active_ids = self._preview_active_ids()
        object_ids = [
            object_id
            for object_id in self._preview_base_part_ids()
            if object_id in self.context.objects
        ]
        if not object_ids:
            self.status_var.set("No used meshes are loaded yet")
            return
        for object_id in object_ids:
            settings = self._part_settings(object_id)
            settings["viewerVisible"] = object_id in active_ids
            settings["viewerSolo"] = False
        self._refresh_parts()
        self._refresh_viewer()
        self._update_detail()
        self.status_var.set(f"Showing {len(active_ids & set(object_ids))} active mesh(es) for the previewed trim")

    def _clear_part_solo(self) -> None:
        if self.context is None:
            return
        cleared = 0
        for object_id in self._preview_base_part_ids():
            settings = self._part_settings(object_id)
            if settings.get("viewerSolo"):
                settings["viewerSolo"] = False
                cleared += 1
        self._refresh_parts()
        self._refresh_viewer()
        self._update_detail()
        self.status_var.set(f"Cleared {cleared} solo flag(s)")

    def _toggle_selected_parts_visibility_shortcut(self, event: tk.Event) -> str | None:
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in TYPING_WIDGET_CLASSES:
            return None
        if self.context is None:
            return None
        selected = [
            self._part_row_mesh_id(row_id)
            for row_id in self.part_tree.selection()
            if self.part_tree.exists(row_id) and self._part_row_mesh_id(row_id) in self.context.objects
        ]
        selected = sorted(set(selected))
        if not selected:
            return None
        for object_id in selected:
            settings = self._part_settings(object_id)
            settings["viewerVisible"] = not bool(settings.get("viewerVisible", True))
        self._refresh_part_viewer_cells(selected)
        self._refresh_viewer()
        self._update_detail()
        if len(selected) == 1:
            object_id = selected[0]
            visible = bool(self._get_part_setting(object_id, "viewerVisible", True))
            self.status_var.set(
                f"{self._part_display_name(object_id)} {'visible' if visible else 'hidden'}"
            )
        else:
            self.status_var.set(f"Toggled visibility for {len(selected)} selected part(s)")
        return "break"

    def _set_selected_part_mode_shortcut(self, _event: tk.Event, mode: str) -> str | None:
        focus = self.focus_get()
        # Only typing targets swallow the hotkeys; buttons and other focusable
        # widgets don't react to letter keys, so mode setting stays live.
        if focus is not None and focus.winfo_class() in TYPING_WIDGET_CLASSES:
            return None
        if self.context is None:
            return None
        target_rows = [
            row_id
            for row_id in self.part_tree.selection()
            if self.part_tree.exists(row_id)
            and self._part_row_mesh_id(row_id) in self.context.objects
            and not self._part_child_override(row_id)
        ]
        if not target_rows:
            item = self.part_tree.focus()
            if (
                item
                and self.part_tree.exists(item)
                and self._part_row_mesh_id(item) in self.context.objects
                and not self._part_child_override(item)
            ):
                target_rows = [item]
        targets = [self._part_row_mesh_id(row_id) for row_id in target_rows]
        targets = sorted(set(targets))
        if not targets:
            if any(self._part_child_override(row_id) for row_id in self.part_tree.selection()):
                self.status_var.set("Effective source/replaced rows are controlled by their Replace Source parent")
                return "break"
            return None
        if mode == core.MODE_MIRROR_STRUCTURAL:
            if len(targets) != 1:
                self.status_var.set("Select one mesh to set Swap Mesh; it needs a source mesh")
                return "break"
            self._set_part_mode(targets[0], mode)
            return "break"
        if mode == core.MODE_REPLACE_SOURCE:
            # Replace Source is a pairing, so it goes through _set_part_mode --
            # which picks the partner and refuses the mode when the slot holds
            # no other part -- rather than being written straight onto the row.
            if len(targets) != 1:
                self.status_var.set("Select one part to set Replace Source; it needs a source part")
                return "break"
            self._set_part_mode(targets[0], mode, row_id=target_rows[0])
            return "break"
        for object_id in targets:
            self._cancel_structural_prompt(object_id)
            self._apply_single_part_mode(object_id, mode)
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()
        if len(targets) == 1:
            self.status_var.set(f"{self._part_display_name(targets[0])}: {mode_label(mode)}")
        else:
            self.status_var.set(f"Set {mode_label(mode)} on {len(targets)} part(s)")
        return "break"

    def _part_settings(self, object_id: str) -> dict[str, object]:
        object_id = self._part_row_mesh_id(object_id)
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(
            object_id,
            core.default_part_setting(object_id),
        )
        if not isinstance(settings, dict):
            settings = {}
            parts[object_id] = settings
        return settings

    def _mode_values_for_part_row(self, row_id: str) -> list[str]:
        return list(MODE_CYCLE_VALUES)

    def _slot_def_for_part_row(self, row_id: str) -> core.SlotDef | None:
        if self.context is None:
            return None
        row = getattr(self, "part_instance_rows", {}).get(row_id)
        if not isinstance(row, dict):
            return None
        slot_id = str(row.get("slot_id") or "")
        slot_path = str(row.get("slot_path") or "")
        if not slot_id or slot_id == "main" or not slot_path:
            return None
        config_name = self._mesh_scene_config()
        if config_name is None:
            return None
        try:
            selected = core.selected_parts_for_config(self.context, config_name)
        except Exception:
            return None
        selected_by_path = selected.get("selected_by_path", {})
        if not isinstance(selected_by_path, dict):
            return None
        parent_path = "/"
        stripped = slot_path.strip("/")
        if "/" in stripped:
            parent_path = "/" + stripped.rsplit("/", 1)[0] + "/"
        parent_part = str(selected_by_path.get(parent_path) or "")
        found = core.part_body_for_context(self.context, parent_part) if parent_part else None
        if found is not None:
            for slot_def in core.extract_slot_defs(found[0]):
                if slot_def.slot_type == slot_id:
                    return slot_def
        return core.SlotDef(slot_id, str(row.get("part_id") or ""), allow_types=(slot_id,))

    def _replace_source_candidate_ids(self, object_id: str, row_id: str | None = None) -> list[str]:
        if self.context is None:
            return []
        row_id = row_id or next(
            (
                item
                for item in getattr(self, "part_instance_rows", {})
                if self._part_row_mesh_id(item) == object_id
            ),
            "",
        )
        slot_def = self._slot_def_for_part_row(row_id)
        row = getattr(self, "part_instance_rows", {}).get(row_id, {})
        current_part = str(row.get("part_id") or "")
        if slot_def is None:
            return []
        candidate_meshes: set[str] = set()
        seen: set[str] = set()
        for candidate_part, (candidate_body, _filename) in self.context.part_body_index.items():
            if candidate_part == current_part:
                continue
            if not core.part_fits_slot(
                core.transform_helpers.extract_part_slot_types(candidate_body),
                slot_def,
            ):
                continue
            for mesh_id in core.transform_helpers.extract_part_mesh_names(candidate_body):
                if mesh_id == object_id or mesh_id in seen:
                    continue
                obj = self.context.objects.get(mesh_id)
                if obj is None or not obj.dae_path:
                    continue
                candidate_meshes.add(mesh_id)
                seen.add(mesh_id)
        candidates = self._matching_replacement_meshes(object_id, candidate_meshes)
        candidates.sort(key=lambda item: self._part_display_name(item).lower())
        return candidates

    def _replacement_identity(self, object_id: str) -> str:
        text = object_id.lower()
        if self.context is not None:
            prefix = f"{self.context.vehicle_id.lower()}_"
            if text.startswith(prefix):
                text = text[len(prefix) :]
        text = re.sub(r"(^|[_\-.])(lhd|rhd|left|right|driver|passenger|fl|fr|rl|rr|l|r)(?=$|[_\-.])", r"\1", text)
        return re.sub(r"[^a-z0-9]+", "", text)

    def _matching_replacement_meshes(self, object_id: str, candidates: set[str]) -> list[str]:
        if not candidates:
            return []
        named = self._name_pair_candidate(object_id, sorted(candidates))
        if named:
            return [named]
        identity = self._replacement_identity(object_id)
        matches = [
            candidate
            for candidate in candidates
            if self._replacement_identity(candidate) == identity
        ]
        if matches:
            return matches
        return []

    def _choose_replace_source(self, object_id: str, row_id: str | None = None) -> str | None:
        if self.context is None:
            return None
        candidates = self._replace_source_candidate_ids(object_id, row_id)
        existing = str(self._get_part_setting(object_id, "mirrorSource", "") or "")
        if existing and existing in self.context.objects and existing != object_id and existing not in candidates:
            candidates.append(existing)
        if not candidates:
            self._show_error("Replace Source", "No alternative part with renderable meshes fits this slot.")
            return None

        label_universe = [object_id, *candidates]
        value_by_label = {
            self._part_option_label(candidate, label_universe): candidate
            for candidate in candidates
        }
        label_by_value = {value: label for label, value in value_by_label.items()}

        modal = tk.Toplevel(self)
        modal.title("Replace Source")
        modal.transient(self)
        modal.resizable(False, False)
        modal.columnconfigure(1, weight=1)

        ttk.Label(modal, text="Mesh").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(modal, text=self._part_option_label(object_id, label_universe)).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 10),
            pady=(10, 4),
        )
        ttk.Label(modal, text="Source").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        source_var = tk.StringVar()
        combo = ttk.Combobox(
            modal,
            textvariable=source_var,
            values=list(value_by_label),
            state="readonly",
            width=58,
            height=min(max(len(value_by_label), 1), 16),
        )
        combo.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)
        if existing and existing in label_by_value:
            source_var.set(label_by_value[existing])
        elif value_by_label:
            source_var.set(next(iter(value_by_label)))

        result: dict[str, str | None] = {"source": None}

        def commit() -> None:
            selected = value_by_label.get(source_var.get())
            if not selected:
                self._show_error("Replace Source", "Select a replacement source.", parent=modal)
                return
            result["source"] = selected
            modal.destroy()

        buttons = ttk.Frame(modal)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=10, pady=(6, 10))
        ttk.Button(buttons, text="Cancel", command=modal.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply", command=commit).pack(side="right", padx=(0, 6))

        modal.protocol("WM_DELETE_WINDOW", modal.destroy)
        modal.bind("<Escape>", lambda _event: modal.destroy())
        modal.bind("<Return>", lambda _event: commit())
        self._place_modal_on_app_monitor(modal)
        combo.focus_set()
        modal.grab_set()
        self.wait_window(modal)
        return result["source"]

    def _swap_source_label(self, object_id: str, settings: object | None = None) -> str:
        if not isinstance(settings, dict):
            settings = self._part_settings(object_id)
        if settings.get("mode") not in {core.MODE_MIRROR_STRUCTURAL, core.MODE_REPLACE_SOURCE}:
            return "N/A"
        source_id = str(settings.get("mirrorSource") or "")
        if not source_id:
            return "Choose..."
        return self._part_display_label(source_id)

    def _set_replace_source(self, object_id: str, source_id: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        source_id = self._part_row_mesh_id(source_id)
        existing_settings = self._part_settings(object_id)
        existing_source_settings = self._part_settings(source_id)
        already_replace_source = existing_settings.get("mode") == core.MODE_REPLACE_SOURCE
        include_children = (
            bool(existing_settings.get("includeChildren"))
            or bool(existing_source_settings.get("includeChildren"))
            or not already_replace_source
        )
        self._cancel_structural_prompt(object_id)
        self._cancel_structural_prompt(source_id)
        self._clear_structural_pair(object_id)
        self._clear_structural_pair(source_id)
        self._clear_replace_source_pair(object_id)
        self._clear_replace_source_pair(source_id)
        settings = self._part_settings(object_id)
        source_settings = self._part_settings(source_id)
        settings["mode"] = core.MODE_REPLACE_SOURCE
        settings["mirrorSource"] = source_id
        settings["includeChildren"] = include_children
        source_settings["mode"] = core.MODE_REPLACE_SOURCE
        source_settings["mirrorSource"] = object_id
        source_settings["includeChildren"] = include_children
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"Replace Source pair set: {self._part_display_name(object_id)} from "
            f"{self._part_display_name(source_id)}"
        )

    def _clear_replace_source_pair(self, object_id: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        settings = self._part_settings(object_id)
        source_id = str(settings.get("mirrorSource") or "")
        settings["mirrorSource"] = None
        settings["includeChildren"] = False
        if not source_id:
            return
        source_settings = self._part_settings(source_id)
        if (
            source_settings.get("mode") == core.MODE_REPLACE_SOURCE
            and str(source_settings.get("mirrorSource") or "") == object_id
        ):
            source_settings["mode"] = core.MODE_SKIP
            source_settings["mirrorSource"] = None
            source_settings["includeChildren"] = False

    def _clear_structural_pair(self, object_id: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        settings = self._part_settings(object_id)
        source_id = str(settings.get("mirrorSource") or "")
        settings["mirrorSource"] = None
        if not source_id:
            return
        source_settings = self._part_settings(source_id)
        if (
            source_settings.get("mode") == core.MODE_MIRROR_STRUCTURAL
            and str(source_settings.get("mirrorSource") or "") == object_id
        ):
            source_settings["mode"] = core.MODE_SKIP
            source_settings["mirrorSource"] = None

    def _set_structural_pair(self, object_id: str, source_id: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        source_id = self._part_row_mesh_id(source_id)
        self._cancel_structural_prompt(object_id)
        self._cancel_structural_prompt(source_id)
        self._clear_structural_pair(object_id)
        self._clear_structural_pair(source_id)
        self._clear_replace_source_pair(object_id)
        self._clear_replace_source_pair(source_id)
        settings = self._part_settings(object_id)
        source_settings = self._part_settings(source_id)
        settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        settings["mirrorSource"] = source_id
        settings["includeChildren"] = False
        source_settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        source_settings["mirrorSource"] = object_id
        source_settings["includeChildren"] = False
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"Swap Mesh pair set: {self._part_display_name(object_id)} <-> "
            f"{self._part_display_name(source_id)}"
        )

    def _set_part_mode(self, object_id: str, mode: str, *, row_id: str | None = None) -> None:
        object_id = self._part_row_mesh_id(object_id)
        current_mode = str(self._get_part_setting(object_id, "mode", core.MODE_SKIP))
        if mode == core.MODE_MIRROR_STRUCTURAL:
            settings = self._part_settings(object_id)
            if current_mode == core.MODE_MIRROR_STRUCTURAL:
                self._clear_structural_pair(object_id)
                settings = self._part_settings(object_id)
            if current_mode == core.MODE_REPLACE_SOURCE:
                self._clear_replace_source_pair(object_id)
                settings = self._part_settings(object_id)
            settings["mode"] = core.MODE_MIRROR_STRUCTURAL
            settings["mirrorSource"] = None
            settings["includeChildren"] = False
            self._refresh_parts()
            self._update_detail()
            self._schedule_structural_prompt(object_id, current_mode)
            return
        if mode == core.MODE_REPLACE_SOURCE:
            candidates = self._replace_source_candidate_ids(object_id, row_id)
            if not candidates:
                self.status_var.set("Replace Source needs another part that fits this slot")
                return
            self._set_replace_source(object_id, candidates[0])
            return
        self._cancel_structural_prompt(object_id)
        settings = self._part_settings(object_id)
        if settings.get("mode") == core.MODE_MIRROR_STRUCTURAL:
            self._clear_structural_pair(object_id)
            settings = self._part_settings(object_id)
        if settings.get("mode") == core.MODE_REPLACE_SOURCE:
            self._clear_replace_source_pair(object_id)
            settings = self._part_settings(object_id)
        settings["mode"] = mode
        settings["mirrorSource"] = None
        settings["includeChildren"] = False
        self._refresh_parts()
        self._update_detail()

    def _set_part_offset(self, object_id: str, value: str) -> None:
        object_id = self._part_row_mesh_id(object_id)
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if not isinstance(settings, dict):
            return
        cleaned = value.strip()
        if not cleaned:
            settings["translateOffset"] = None
        else:
            try:
                # Signed: positive follows the trim's own conversion direction,
                # negative walks the part back the other way.
                settings["translateOffset"] = float(cleaned)
            except ValueError:
                self._show_error(
                    "Invalid offset",
                    "Part offset must be blank or a number. Positive moves the part the way this "
                    "trim converts; negative moves it the opposite way.",
                )
                return
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()

    def _part_selection_changed(self) -> None:
        if self.part_tree.selection():
            self._claim_selection("parts")
        self._refresh_viewer()
        self._update_detail()

    def _on_preview_pick(self, object_id: object) -> None:
        """A part was clicked in the GPU preview. object_id is the picked mesh
        name (== a part_tree iid) or None for empty space. Setting the tree
        selection fires <<TreeviewSelect>> which refreshes the highlight+detail."""
        if self.context is None:
            return
        # A viewer click means the user is working with parts: pull keyboard
        # focus onto the table so the mode/visibility hotkeys apply directly.
        self.part_tree.focus_set()
        if not object_id:
            if isinstance(getattr(self, "side_pair_pick_target", None), dict):
                self._clear_side_pair_pick_target()
                self._refresh_slots()
                self.status_var.set("Equivalent part picker cancelled")
                return
            self._clear_all_selection()  # empty click -> deselect everything
            return
        object_id = str(object_id)
        # A trigger box is scene geometry with no part row behind it, so it
        # answers to the Triggers table instead.
        if mesh_preview is not None and object_id.startswith(
            f"{mesh_preview.TRIGGER_SCENE_PREFIX}|"
        ):
            self._select_trigger_row(object_id)
            row = getattr(self, "trigger_rows_by_iid", {}).get(object_id)
            if row is not None:
                self.trigger_tree.focus_set()
                self.status_var.set(
                    f"{row['label']} -- {self._trigger_mode_label(row)}"
                )
            return
        # The clicked part is rendered but may be filtered out of the table;
        # clear the filter so its row exists and can be selected. The filter_var
        # write-trace rebuilds the table synchronously.
        if not self.part_tree.exists(object_id) and self.filter_var.get().strip():
            self.filter_var.set("")
        if not self.part_tree.exists(object_id):
            for row_id in self.part_tree.get_children():
                if self._part_row_side_ref(row_id) == object_id or self._part_row_mesh_id(row_id) == object_id:
                    object_id = row_id
                    break
        if not self.part_tree.exists(object_id):
            scene = getattr(self.viewer, "scene", None) if getattr(self, "viewer", None) is not None else None
            pick_to_row = getattr(scene, "pick_to_row", {}) if scene is not None else {}
            mapped = str(pick_to_row.get(object_id, "")) if isinstance(pick_to_row, dict) else ""
            if mapped:
                if self.part_tree.exists(mapped):
                    object_id = mapped
                else:
                    for row_id in self.part_tree.get_children():
                        if self._part_row_mesh_id(row_id) == mapped:
                            object_id = row_id
                            break
        if self.part_tree.exists(object_id):
            if isinstance(getattr(self, "side_pair_pick_target", None), dict):
                self._commit_side_pair_part_pick_from_row(object_id)
                return
            self.part_tree.selection_set([object_id])
            self.part_tree.focus(object_id)
            self.part_tree.see(object_id)
        elif isinstance(getattr(self, "side_pair_pick_target", None), dict):
            self.status_var.set("Clicked preview mesh is not available in the Mesh Transforms table")

    def _manual_delta_toggled(self, *, refresh: bool = True) -> None:
        state = "normal" if self.manual_delta_enabled.get() else "disabled"
        self.manual_delta_entry.configure(state=state)
        delta = self.conversion.setdefault("delta", {})
        if isinstance(delta, dict):
            delta["manual"] = bool(self.manual_delta_enabled.get())
        if refresh:
            self._commit_delta_from_ui()

    def _commit_delta_from_ui(self) -> None:
        delta = self.conversion.setdefault("delta", {})
        if isinstance(delta, dict):
            delta["manual"] = bool(self.manual_delta_enabled.get())
            if self.manual_delta_enabled.get():
                text = self.manual_delta_var.get().strip()
                try:
                    delta["magnitude"] = abs(float(text)) if text else 0.0
                except ValueError:
                    self._show_error("Invalid delta", "Manual delta magnitude must be a number.")
                    return
        self._refresh_delta_label()
        self._refresh_parts()
        self._update_detail()
