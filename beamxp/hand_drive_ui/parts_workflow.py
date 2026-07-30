from __future__ import annotations

from .shared import *


class PartsWorkflowMixin:
    """Part inventory refresh, filtering, selection, viewer synchronisation, and variant interaction."""

    def _refresh_parts(self, *, reset_view: bool = False) -> None:
        if self.context is None:
            return
        query = self.filter_var.get().strip().lower()
        # The x/y/z columns read placed geometry, so make sure it matches the
        # trim on screen before the rows are built.
        self._refresh_box_preview()
        keep = set(self.part_tree.selection())
        previous_order = list(self.part_tree.get_children(""))
        for item in self.part_tree.get_children():
            self.part_tree.delete(item)

        parts = self.conversion.setdefault("parts", {})
        ids = self.resolved_part_ids
        active_ids = self._preview_active_ids()
        selected_variants = self._selected_variant_names()
        flexbody_meshes, prop_meshes, _all_meshes = core.selected_mesh_roles(
            self.context,
            selected_variants,
        )
        displayed: list[str] = []
        row_index = 0
        for object_id in ids:
            obj = self.context.objects.get(object_id)
            if obj is None:
                continue
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
                continue
            mode = str(settings.get("mode", core.MODE_SKIP))
            display_name = self._part_display_name(object_id)
            part_type = part_type_label(object_id, flexbody_meshes, prop_meshes)
            if (
                query
                and query not in object_id.lower()
                and query not in display_name.lower()
                and query not in mode
                and query not in part_type.lower()
            ):
                continue
            displayed.append(object_id)
            self.part_tree.insert(
                "",
                "end",
                iid=object_id,
                text=display_name,
                tags=self._row_tags(row_index),
                values=(
                    part_type,
                    yn_label(settings.get("viewerVisible", True)),
                    yn_label(settings.get("viewerSolo")),
                    yn_label(object_id in active_ids),
                    mode_label(mode),
                    offset_display(
                        mode,
                        settings.get("translateOffset"),
                        manual_delta=self.manual_delta_enabled.get(),
                    ),
                    yn_label(settings.get("steeringRef")),
                    *position_labels(*self._table_position(object_id)),
                ),
            )
            row_index += 1
        self.current_part_ids = displayed
        self._restore_tree_order(self.part_tree, previous_order)
        visible_keep = [item for item in keep if self.part_tree.exists(item)]
        if visible_keep:
            self.part_tree.selection_set(visible_keep)
        self._refresh_viewer(reset=reset_view)

    def _schedule_parts_refresh(self, *, reset_view: bool = False) -> None:
        self.part_refresh_pending_reset = self.part_refresh_pending_reset or reset_view
        if self.part_refresh_running:
            self.part_refresh_pending = True
            self.part_refresh_seq += 1
            return
        if self.part_refresh_after_id is not None:
            return
        self.part_refresh_after_id = self.after_idle(self._run_scheduled_parts_refresh)

    def _run_scheduled_parts_refresh(self) -> None:
        self.part_refresh_after_id = None
        reset_view = self.part_refresh_pending_reset
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self._start_parts_refresh(reset_view=reset_view)

    def _start_parts_refresh(self, *, reset_view: bool = False) -> None:
        self.part_refresh_after_id = None
        if self.context is None:
            self.resolved_part_ids = []
            self._refresh_parts(reset_view=reset_view)
            return
        selected = tuple(self._selected_variant_names())
        self.part_refresh_seq += 1
        seq = self.part_refresh_seq
        if not selected:
            self.resolved_part_ids = []
            self._refresh_parts(reset_view=reset_view)
            self.status_var.set("No trims selected; 0 used part(s) displayed")
            return
        context = self.context
        cached_ids = core.load_cached_part_ids(context, selected)
        if cached_ids is not None:
            self.resolved_part_ids = [part_id for part_id in cached_ids if part_id in context.objects]
            self._refresh_parts(reset_view=reset_view)
            self._update_detail()
            self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed (parts cache)")
            return
        self.status_var.set(f"Resolving used parts for {len(selected)} trim(s)...")
        self.part_refresh_running = True
        future = self.part_resolver.submit(self._resolve_part_ids_worker, context, selected)
        future.add_done_callback(
            lambda completed, current_seq=seq, current_context=context, should_reset=reset_view, current_selected=selected: self.worker_queue.put(
                ("parts_success", (current_seq, current_context, should_reset, current_selected, completed))
            )
        )

    @staticmethod
    def _resolve_part_ids_worker(
        context: core.VehicleContext,
        selected: tuple[str, ...],
    ) -> list[str]:
        _flex, _props, all_meshes = core.selected_mesh_roles(context, list(selected))
        return sorted(mesh for mesh in all_meshes if mesh in context.objects)

    def _selected_variant_names(self) -> list[str]:
        if self.context is None:
            return []
        variants = self.conversion.get("variants", {})
        if not isinstance(variants, dict):
            return []
        return [
            name
            for name, settings in variants.items()
            if name in self.context.variants
            and isinstance(settings, dict)
            and core.variant_build_mode(settings) != core.BUILD_OFF
        ]

    def _preview_base_part_ids(self) -> list[str]:
        if self.context is None:
            return []
        return [
            object_id
            for object_id in (self.resolved_part_ids or self.current_part_ids)
            if object_id in self.context.objects
        ]

    def _resolved_visible_ids(self) -> set[str]:
        """The set of parts actually present in the active preview / final
        visible output for the current variant selection: Solo (if any part is
        soloed) or per-part Visible toggles, over the resolved used-part set.
        Table selection deliberately has no effect here -- Visible/Solo have the
        final say over what the preview and the converted output contain."""
        if self.context is None:
            return set()
        parts = self.conversion.get("parts", {})
        base_ids = self._preview_base_part_ids()
        solo_ids = {
            object_id
            for object_id in base_ids
            if isinstance(parts, dict)
            and isinstance(parts.get(object_id), dict)
            and parts[object_id].get("viewerSolo")
        }
        if solo_ids:
            return solo_ids
        return {
            object_id
            for object_id in base_ids
            if not isinstance(parts, dict)
            or not isinstance(parts.get(object_id), dict)
            or parts[object_id].get("viewerVisible", True)
        }

    def _refresh_viewer(self, *, reset: bool = False) -> None:
        if self.viewer is None:
            return
        visible_ids = self._resolved_visible_ids()
        # Selected inactive parts are temporarily injected into the GPU scene
        # (scene.extra); show them while they stay selected. Intersecting with
        # the live selection hides a stale extra instantly after deselection,
        # before the scene rebuild that drops it has landed.
        scene = getattr(self.viewer, "scene", None)
        visible_ids |= set(getattr(scene, "extra", ()) or ()) & set(self.part_tree.selection())
        dimmed_ids = visible_ids - set(self.current_part_ids)
        self.viewer.set_visible_ids(list(visible_ids), reset=reset)
        if hasattr(self.viewer, "set_dimmed_ids"):
            self.viewer.set_dimmed_ids(dimmed_ids)
        # Selection only drives the highlight outline (skipped for hidden parts
        # in the renderer); it never adds a part to the visible set above.
        self.viewer.set_selected_ids(set(self.part_tree.selection()))

    def _preview_active_ids(self) -> set[str]:
        """Object ids present on the trim currently shown in the moderngl
        preview -- i.e. the config chosen in the Config dropdown. This
        indicates which parts the converted trim actually uses; it is NOT
        affected by the viewer Visible/Solo toggles (those only filter what is
        drawn). Ground truth is the built scene's mesh groups (keyed by object
        id, already excluding inactive/geometry-less rows for this config)."""
        scene = getattr(self.viewer, "scene", None) if self.viewer is not None else None
        groups = getattr(scene, "groups", None)
        if groups:
            # Temporarily-shown inactive parts (scene.extra) are in the scene
            # but not part of the previewed trim; they are never Active.
            return set(groups.keys()) - set(getattr(scene, "extra", ()) or ())
        # No GPU scene yet (box-viewer fallback, or the preview is still
        # building): resolve the previewed config's meshes directly. Roles are
        # cached per config on the context, so this stays cheap.
        if self.context is None:
            return set()
        config = self._mesh_scene_config()
        if config is None:
            return set()
        try:
            _flex, _props, all_meshes = core.selected_mesh_roles(self.context, [config])
        except Exception:
            return set()
        return {mesh for mesh in all_meshes if mesh in self.context.objects}

    def _selected_extra_preview_ids(self) -> list[str]:
        """Selected table parts NOT used by the previewed config. These get
        temporarily injected into the GPU scene so selecting an inactive part
        still shows it; deselecting removes it again. Active parts are never
        in this list, so their behaviour is unchanged."""
        if self.context is None or not hasattr(self, "part_tree"):
            return []
        config = self._mesh_scene_config()
        if config is None:
            return []
        try:
            _flex, _props, all_meshes = core.selected_mesh_roles(self.context, [config])
        except Exception:
            return []
        return sorted(
            object_id
            for object_id in self.part_tree.selection()
            if object_id in self.context.objects and object_id not in all_meshes
        )

    def _refresh_active_cells(self) -> None:
        """Update the parts table Active (Y/N) column for every displayed row to
        reflect the trim currently shown in the moderngl preview."""
        if not hasattr(self, "part_tree") or self.part_tree is None:
            return
        active_ids = self._preview_active_ids()
        for object_id in self.part_tree.get_children():
            self.part_tree.set(object_id, "active", yn_label(object_id in active_ids))

    def _refresh_delta_label(self) -> None:
        if self.context is None:
            self.auto_delta_var.set("")
            return
        auto = core.auto_delta_magnitude(self.context, self.conversion)
        source_refs = core.auto_delta_source_refs(self.context, self.conversion)
        if source_refs:
            names = ", ".join(self._part_display_name(object_id) for object_id in source_refs)
            source = f"found using {names}"
        else:
            # No steering ref selected (or the selected one has no usable
            # off-center X), so the auto delta is just its default.
            source = "no steering ref found"
        self.auto_delta_var.set(f"{fmt_float(auto)} ({source})")

    def _update_detail(self) -> None:
        if self.context is None:
            self.detail_var.set("")
            return
        # Every conversion mutation (mode, translate offset, structural pairing,
        # steering ref, manual delta, variant hand override) funnels through here
        # as its final UI step, so this is where we keep the GPU preview live.
        # _schedule_mesh_scene is snapshot-guarded: pure selection/visibility
        # changes leave the fingerprint unchanged and cost only a cheap compare.
        self._schedule_mesh_scene()
        selected_parts = self.part_tree.selection()
        if selected_parts:
            object_id = selected_parts[0]
            obj = self.context.objects.get(object_id)
            settings = self.conversion.get("parts", {}).get(object_id, {})
            if obj:
                display_name = self._part_display_name(object_id)
                mode = str(settings.get("mode", core.MODE_SKIP)) if isinstance(settings, dict) else core.MODE_SKIP
                part_offset = (
                    offset_display(
                        mode,
                        settings.get("translateOffset") if isinstance(settings, dict) else None,
                        manual_delta=self.manual_delta_enabled.get(),
                    )
                    if mode == core.MODE_TRANSLATE
                    else "N/A"
                )
                position, varies = self._table_position(object_id)
                self.detail_var.set(
                    f"{display_name}: {mode_label(mode)}, "
                    f"full id {object_id}, x {fmt_float(position[0])}, offset {part_offset}, "
                    f"dae {obj.dae_path}{self._variant_position_note(object_id) if varies else ''}"
                )
                return
        active = len(core.active_part_modes(self.conversion))
        selected_variants = len(self._selected_variant_names())
        self.detail_var.set(
            f"{len(self.current_part_ids)} displayed part(s), {active} transformed part setting(s), "
            f"{selected_variants} selected variant(s)"
        )

    def _set_all_variants_selected(self, selected: bool) -> None:
        if self.context is None:
            return
        variants = self.conversion.setdefault("variants", {})
        for config_name in self.context.variants:
            settings = variants.setdefault(config_name, {})
            if isinstance(settings, dict):
                core.set_variant_build_mode(settings, core.BUILD_CONVERTED if selected else core.BUILD_OFF)
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()
        self.status_var.set(
            f"{'All trims selected' if selected else 'All trims cleared'}; updating used parts..."
        )

    def _toggle_variant_selected(self, config_name: str) -> None:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            mode = core.variant_build_mode(settings)
            core.set_variant_build_mode(
                settings,
                core.BUILD_OFF if mode != core.BUILD_OFF else core.BUILD_CONVERTED,
            )
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()
        state = BUILD_LABELS[core.variant_build_mode(settings)] if isinstance(settings, dict) else "Off"
        self.status_var.set(f"{config_name} {state}; updating used parts...")

    def _variant_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.variant_tree, event):
            return None
        item = self.variant_tree.identify_row(event.y)
        column = self.variant_tree.identify_column(event.x)
        if not item or self.context is None:
            return
        name = self._tree_column_name(self.variant_tree, column)
        if name == "build":
            settings = self.conversion.setdefault("variants", {}).setdefault(item, {})
            current = BUILD_LABELS[core.variant_build_mode(settings)] if isinstance(settings, dict) else BUILD_LABELS[core.BUILD_OFF]
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                list(BUILD_LABELS.values()),
                current,
                lambda value: self._set_variant_build_label(item, value),
            )
            return "break"
        if name == "stock_hand":
            settings = self.conversion.get("variants", {}).get(item, {})
            if core.variant_build_mode(settings) not in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                return "break"
            labels, mapping = self._variant_stock_hand_choices(item, settings)
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                self._variant_stock_hand_label(item, settings),
                lambda value: self._set_variant_setting(item, "sourceHandOverride", mapping[value]),
            )
            return "break"
        if name == "plate":
            labels, mapping = self._variant_plate_choices(item)
            current_label = self._variant_plate_label(item, self.conversion.get("variants", {}).get(item, {}))
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                current_label,
                lambda value: self._set_variant_plate_choice(item, mapping[value]),
            )
            return "break"
        if name in {"front_plate", "rear_plate"}:
            side = "front" if name == "front_plate" else "rear"
            key = "frontPlate" if side == "front" else "rearPlate"
            choices = plate_generator.plate_part_choices_for_config(self.context, item, side)
            labels = [choice.label for choice in choices]
            values_by_label = {choice.label: choice.value for choice in choices}
            current_value = self._get_variant_setting(item, key, plate_generator.PLATE_PART_AUTO)
            current_label = plate_generator.plate_part_label_for_config(
                self.context,
                item,
                side,
                current_value,
            )
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                current_label,
                lambda value: self._set_variant_setting(item, key, values_by_label[value]),
            )
            return "break"
        if name is not None:
            # Clicking the descriptive cells keeps the old quick on/off action.
            self._toggle_variant_selected(item)
            return "break"
        return None

    def _variant_double_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.variant_tree, event):
            return None
        item = self.variant_tree.identify_row(event.y)
        column = self.variant_tree.identify_column(event.x)
        if not item:
            return
        name = self._tree_column_name(self.variant_tree, column)
        if name == "plate":
            self._open_plate_editor(item)
        elif name == "stock_hand":
            settings = self.conversion.get("variants", {}).get(item, {})
            if core.variant_build_mode(settings) not in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                return "break"
            labels, mapping = self._variant_stock_hand_choices(item, settings)
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                self._variant_stock_hand_label(item, settings),
                lambda value: self._set_variant_setting(item, "sourceHandOverride", mapping[value]),
            )
