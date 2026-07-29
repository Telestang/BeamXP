from __future__ import annotations

from .shared import *
from .recommendation_engine import build_mode_recommendations


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
        name = self._tree_column_name(self.part_tree, column)
        if name == "visible":
            self._toggle_part_bool(item, "viewerVisible", default=True)
            return "break"
        if name == "solo":
            self._toggle_part_bool(item, "viewerSolo")
            return "break"
        if name == "mode":
            self.part_tree.focus(item)
            self.part_tree.selection_set(item)
            current = mode_label(str(self._get_part_setting(item, "mode", core.MODE_SKIP)))
            self._edit_tree_combo(
                self.part_tree,
                item,
                column,
                [mode_label(mode) for mode in MODE_CYCLE_VALUES],
                current,
                lambda value: self._set_part_mode_from_label(item, value),
            )
            return "break"
        if name == "offset":
            if self._get_part_setting(item, "mode", core.MODE_SKIP) != core.MODE_TRANSLATE:
                self.status_var.set("Offset X only applies to Translate mode")
                return "break"
            self._edit_tree_entry(
                self.part_tree,
                item,
                column,
                offset_label(self._get_part_setting(item, "translateOffset", None)),
                lambda value: self._set_part_offset(item, value),
            )
            return "break"
        if name == "steering":
            self._set_single_steering_ref(item)
            return "break"
        # "active" is read-only, and #0/coords fall through to default row select.
        return None

    def _part_motion(self, event: tk.Event) -> None:
        if self.structural_prompt_part_id is None or self.structural_prompt_open:
            return
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if item != self.structural_prompt_part_id or self._tree_column_name(self.part_tree, column) != "mode":
            self._trigger_structural_prompt()

    def _part_leave(self, _event: tk.Event) -> None:
        if self.structural_prompt_part_id is not None and not self.structural_prompt_open:
            self._trigger_structural_prompt()

    def _part_double_click(self, event: tk.Event) -> None:
        if not self._tree_body_click(self.part_tree, event):
            return None
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if not item:
            return
        name = self._tree_column_name(self.part_tree, column)
        if name == "mode":
            return "break"
        elif name == "offset":
            if self._get_part_setting(item, "mode", core.MODE_SKIP) != core.MODE_TRANSLATE:
                self.status_var.set("Offset X only applies to Translate mode")
                return
            self._edit_tree_entry(
                self.part_tree,
                item,
                column,
                offset_label(self._get_part_setting(item, "translateOffset", None)),
                lambda value: self._set_part_offset(item, value),
            )

    def _set_part_mode_from_label(self, object_id: str, label: str) -> None:
        mode = MODE_VALUES_BY_LABEL.get(label)
        if mode is None:
            return
        self._set_part_mode(object_id, mode)
        if mode != core.MODE_MIRROR_STRUCTURAL:
            # Mirror Structural sets its own "choose a source" status message.
            self.status_var.set(f"{self._part_display_name(object_id)}: {mode_label(mode)}")

    def _cancel_structural_prompt(self, object_id: str | None = None) -> None:
        if object_id is not None and self.structural_prompt_part_id != object_id:
            return
        if self.structural_prompt_after_id is not None:
            try:
                self.after_cancel(self.structural_prompt_after_id)
            except Exception:
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
        self.structural_prompt_after_id = self.after(STRUCTURAL_PROMPT_DELAY_MS, self._trigger_structural_prompt)
        self.status_var.set(
            f"Mirror Structural selected for {self._part_display_name(object_id)}; choose a source to complete it"
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
            f"Mirror Structural cancelled for {self._part_display_name(object_id)}; restored {mode_label(restore_mode)}"
        )

    def _edit_tree_combo(
        self,
        tree: ttk.Treeview,
        item: str,
        column: str,
        values: list[str],
        current: str,
        on_commit,
    ) -> None:
        self._close_tree_combo_editor()
        if not tree.exists(item):
            return
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
            if not self.part_tree.exists(object_id):
                continue
            settings = parts.get(object_id)
            if not isinstance(settings, dict):
                settings = {}
            self.part_tree.set(object_id, "visible", yn_label(settings.get("viewerVisible", True)))
            self.part_tree.set(object_id, "solo", yn_label(settings.get("viewerSolo")))

    def _toggle_part_bool(self, object_id: str, key: str, *, default: bool = False) -> None:
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

    def _set_single_steering_ref(self, object_id: str) -> None:
        if self.context is None:
            return
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
            object_id
            for object_id in self.current_part_ids
            if object_id in self.context.objects
        ]
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

    def _toggle_selected_parts_visibility_shortcut(self, event: tk.Event) -> str | None:
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in {
            "Entry",
            "TEntry",
            "Text",
            "Combobox",
            "TCombobox",
            "Spinbox",
            "TSpinbox",
        }:
            return None
        if self.context is None:
            return None
        selected = [
            object_id
            for object_id in self.part_tree.selection()
            if self.part_tree.exists(object_id) and object_id in self.context.objects
        ]
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
        if focus is not None and focus.winfo_class() in {
            "Entry",
            "TEntry",
            "Text",
            "Combobox",
            "TCombobox",
            "Spinbox",
            "TSpinbox",
        }:
            return None
        if self.context is None:
            return None
        targets = [
            object_id
            for object_id in self.part_tree.selection()
            if self.part_tree.exists(object_id) and object_id in self.context.objects
        ]
        if not targets:
            item = self.part_tree.focus()
            if item and self.part_tree.exists(item) and item in self.context.objects:
                targets = [item]
        if not targets:
            return None
        if mode == core.MODE_MIRROR_STRUCTURAL:
            # The source-pair prompt is a per-part modal; only sensible one at a time.
            if len(targets) != 1:
                self.status_var.set("Select a single part to set Mirror Structural (it needs a source pair)")
                return "break"
            self._set_part_mode(targets[0], mode)
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
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(
            object_id,
            {
                "mode": core.MODE_SKIP,
                "mirrorSource": None,
                "translateOffset": None,
                "steeringRef": False,
                "viewerVisible": True,
                "viewerSolo": False,
            },
        )
        if not isinstance(settings, dict):
            settings = {}
            parts[object_id] = settings
        return settings

    def _clear_structural_pair(self, object_id: str) -> None:
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
        self._cancel_structural_prompt(object_id)
        self._cancel_structural_prompt(source_id)
        self._clear_structural_pair(object_id)
        self._clear_structural_pair(source_id)
        settings = self._part_settings(object_id)
        source_settings = self._part_settings(source_id)
        settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        settings["mirrorSource"] = source_id
        source_settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        source_settings["mirrorSource"] = object_id
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"Structural mirror pair set: {self._part_display_name(object_id)} <-> "
            f"{self._part_display_name(source_id)}"
        )

    def _set_part_mode(self, object_id: str, mode: str) -> None:
        current_mode = str(self._get_part_setting(object_id, "mode", core.MODE_SKIP))
        if mode == core.MODE_MIRROR_STRUCTURAL:
            settings = self._part_settings(object_id)
            if current_mode == core.MODE_MIRROR_STRUCTURAL:
                self._clear_structural_pair(object_id)
                settings = self._part_settings(object_id)
            settings["mode"] = core.MODE_MIRROR_STRUCTURAL
            settings["mirrorSource"] = None
            self._refresh_parts()
            self._update_detail()
            self._schedule_structural_prompt(object_id, current_mode)
            return
        self._cancel_structural_prompt(object_id)
        settings = self._part_settings(object_id)
        if settings.get("mode") == core.MODE_MIRROR_STRUCTURAL:
            self._clear_structural_pair(object_id)
            settings = self._part_settings(object_id)
        settings["mode"] = mode
        settings["mirrorSource"] = None
        self._refresh_parts()
        self._update_detail()

    def _set_part_offset(self, object_id: str, value: str) -> None:
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if not isinstance(settings, dict):
            return
        cleaned = value.strip()
        if not cleaned:
            settings["translateOffset"] = None
        else:
            try:
                settings["translateOffset"] = abs(float(cleaned))
            except ValueError:
                self._show_error("Invalid offset", "Part offset must be blank or a number.")
                return
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()

    def _part_selection_changed(self) -> None:
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
            if self.part_tree.selection():
                self.part_tree.selection_set([])  # empty click -> deselect
            return
        object_id = str(object_id)
        # The clicked part is rendered but may be filtered out of the table;
        # clear the filter so its row exists and can be selected. The filter_var
        # write-trace rebuilds the table synchronously.
        if not self.part_tree.exists(object_id) and self.filter_var.get().strip():
            self.filter_var.set("")
        if self.part_tree.exists(object_id):
            self.part_tree.selection_set([object_id])
            self.part_tree.focus(object_id)
            self.part_tree.see(object_id)

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
