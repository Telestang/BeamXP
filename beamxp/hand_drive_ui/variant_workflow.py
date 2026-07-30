from __future__ import annotations

from .shared import *


class VariantWorkflowMixin:
    """Variant table state, output selection, hand detection, and box-preview positioning."""

    def _refresh_variants(self) -> None:
        if self.context is None:
            return
        self._close_tree_combo_editor()
        keep = set(self.variant_tree.selection())
        previous_order = list(self.variant_tree.get_children(""))
        for item in self.variant_tree.get_children():
            self.variant_tree.delete(item)
        variants = self.conversion.setdefault("variants", {})
        row_index = 0
        for config_name, variant in sorted(self.context.variants.items()):
            settings = variants.setdefault(
                config_name,
                {
                    "selected": False,
                    "build": core.BUILD_OFF,
                    "sourceHandOverride": core.HAND_AUTO,
                    "frontPlate": plate_generator.PLATE_PART_AUTO,
                    "rearPlate": plate_generator.PLATE_PART_AUTO,
                },
            )
            if not isinstance(settings, dict):
                continue
            detected = self._detected_hand_for_ui(config_name)
            build_mode = core.variant_build_mode(settings)
            core.set_variant_build_mode(settings, build_mode)
            stock_hand = (
                self._variant_stock_hand_label(config_name, settings, detected)
                if build_mode in {core.BUILD_CONVERTED, core.BUILD_BOTH}
                else "—"
            )
            self.variant_tree.insert(
                "",
                "end",
                iid=config_name,
                tags=self._row_tags(row_index),
                values=(
                    BUILD_LABELS[build_mode],
                    config_name,
                    variant.display_name,
                    stock_hand,
                    self._variant_plate_label(config_name, settings),
                    plate_generator.plate_part_label_for_config(
                        self.context,
                        config_name,
                        "front",
                        settings.get("frontPlate"),
                    ),
                    plate_generator.plate_part_label_for_config(
                        self.context,
                        config_name,
                        "rear",
                        settings.get("rearPlate"),
                    ),
                ),
            )
            row_index += 1
        self._restore_tree_order(self.variant_tree, previous_order)
        visible_keep = [item for item in keep if self.variant_tree.exists(item)]
        if visible_keep:
            self.variant_tree.selection_set(visible_keep)
        self._refresh_plate_summary()
        self._refresh_preview_outputs()

    def _variant_plate_label(self, config_name: str, settings: dict[str, object]) -> str:
        mode = plate_generator.variant_plate_mode(settings)
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            return f"Custom ({config_name})"
        if mode == plate_generator.PLATE_MODE_TRIM:
            binding = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
            return f"Custom ({binding.get('sourceConfig') or 'missing'})"
        if mode == plate_generator.PLATE_MODE_OFF:
            return "Off"
        if mode == plate_generator.PLATE_MODE_SET:
            binding = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
            set_id = str(binding.get("setId") or "")
            record = plate_generator.plate_set_by_id(set_id)
            return f"Set: {record['name']}" if record else f"Missing set: {set_id}"
        return self._vehicle_plate_label()

    def _refresh_plate_summary(self) -> None:
        if hasattr(self, "plate_summary_var"):
            self.plate_summary_var.set(plate_generator.plate_summary_label(self.conversion))

    def _detected_hand_for_ui(self, config_name: str) -> str:
        return self.variant_detected_hands.get(config_name, "..." if not self.variant_detection_complete else core.HAND_UNKNOWN)

    def _variant_stock_hand_label(
        self,
        config_name: str,
        settings: dict[str, object],
        detected: str | None = None,
    ) -> str:
        detected = detected if detected is not None else self._detected_hand_for_ui(config_name)
        override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
        if override in {core.HAND_LHD, core.HAND_RHD, core.HAND_UNKNOWN} and override != detected:
            return override
        if detected == "...":
            return "Detecting..."
        return f"{detected} (default)"

    def _variant_stock_hand_choices(
        self,
        config_name: str,
        settings: dict[str, object],
    ) -> tuple[list[str], dict[str, str]]:
        detected = self._detected_hand_for_ui(config_name)
        default_label = "Detecting..." if detected == "..." else f"{detected} (default)"
        mapping = {default_label: core.HAND_AUTO}
        for hand in (core.HAND_LHD, core.HAND_RHD):
            if hand != detected:
                mapping[hand] = hand
        override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
        if override == core.HAND_UNKNOWN and core.HAND_UNKNOWN not in mapping:
            mapping[core.HAND_UNKNOWN] = core.HAND_UNKNOWN
        return list(mapping), mapping

    def _variant_output_name_for_ui(
        self,
        config_name: str,
        settings: dict[str, object],
        detected: str | None = None,
    ) -> str:
        detected = detected if detected is not None else self._detected_hand_for_ui(config_name)
        mode = core.variant_build_mode(settings)
        if mode == core.BUILD_OFF:
            return "skip"
        override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
        source = override if override != core.HAND_AUTO else detected
        outputs: list[str] = []
        if mode in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
            if source == "...":
                outputs.append("detecting")
            else:
                target = core.target_hand_for(source, core.ACTION_OPPOSITE)
                outputs.append("skip" if target is None else core.variant_output_name(config_name, target))
        if mode in {core.BUILD_ORIGINAL, core.BUILD_BOTH}:
            outputs.append(core.original_plate_output_name(config_name))
        return ", ".join(outputs)

    @staticmethod
    def _preview_config_label(output_name: str) -> str:
        return re.sub(r"_(?:rhd|lhd)$", "", output_name, flags=re.IGNORECASE)

    def _output_config_sources_for_ui(self) -> tuple[dict[str, str], dict[str, str]]:
        if self.context is None:
            return {}, {}
        variants = self.conversion.get("variants", {})
        if not isinstance(variants, dict):
            return {}, {}
        choices: dict[str, str] = {}
        outputs: dict[str, str] = {}

        def add_choice(config_name: str, output_name: str) -> None:
            label = self._preview_config_label(config_name)
            suffix = 2
            base = label
            while label in choices and choices.get(label) != config_name:
                label = f"{base} {suffix}"
                suffix += 1
            choices[label] = config_name
            outputs[label] = output_name

        for config_name, settings in variants.items():
            if config_name not in self.context.variants or not isinstance(settings, dict):
                continue
            detected = self._detected_hand_for_ui(config_name)
            mode = core.variant_build_mode(settings)
            if mode == core.BUILD_OFF:
                continue
            output_name = ""
            if mode in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
                source = override if override != core.HAND_AUTO else detected
                target = core.target_hand_for(source, core.ACTION_OPPOSITE)
                if target is not None:
                    output_name = core.variant_output_name(config_name, target)
            if not output_name and mode in {core.BUILD_ORIGINAL, core.BUILD_BOTH}:
                output_name = core.original_plate_output_name(config_name)
            if output_name:
                add_choice(config_name, output_name)
        return choices, outputs

    def _refresh_preview_outputs(self) -> None:
        if self.context is None or not hasattr(self, "preview_output_combo"):
            self.preview_output_to_config = {}
            self.preview_output_to_output = {}
            self.preview_output_var.set("")
            return
        current = self.preview_output_var.get()
        choices, outputs = self._output_config_sources_for_ui()
        self.preview_output_to_config = choices
        self.preview_output_to_output = outputs
        values = sorted(choices)
        self.preview_output_combo.configure(values=values)
        if current in choices:
            selected = current
        else:
            selected = self._cached_preview_output(choices, outputs)
            tree_selection = self.variant_tree.selection()
            if tree_selection:
                config_name = tree_selection[0]
                settings = self.conversion.get("variants", {}).get(config_name, {})
                output = (
                    self._variant_output_name_for_ui(config_name, settings)
                    if isinstance(settings, dict)
                    else ""
                )
                if not selected:
                    actual_outputs = {item.strip() for item in output.split(",") if item.strip()}
                    selected = next(
                        (label for label, actual in outputs.items() if actual in actual_outputs),
                        "",
                    )
            if not selected and values:
                selected = values[0]
        self.preview_output_var.set(selected)
        if self.worker_running or not values:
            self.preview_output_combo.configure(state="disabled")
        else:
            self.preview_output_combo.configure(state="readonly")

    def _preview_output_cache_key(self) -> str | None:
        if self.context is None:
            return None
        source = str(self.context.source_zip.resolve(strict=False))
        return f"{source}|{self.context.vehicle_id}"

    def _cached_preview_output(
        self,
        choices: dict[str, str],
        outputs: dict[str, str],
    ) -> str:
        key = self._preview_output_cache_key()
        cache = self.settings.setdefault("previewOutputByVehicle", {})
        if not key or not isinstance(cache, dict):
            return ""
        entry = cache.get(key)
        if isinstance(entry, dict):
            output = str(entry.get("output") or "")
            if output in choices:
                return output
            for label, actual_output in outputs.items():
                if actual_output == output:
                    return label
            config = str(entry.get("config") or "")
            if config:
                for label, source_config in choices.items():
                    if source_config == config:
                        return label
        elif isinstance(entry, str):
            if entry in choices:
                return entry
            for label, actual_output in outputs.items():
                if actual_output == entry:
                    return label
        return ""

    def _remember_preview_output(self, label: str | None = None) -> None:
        key = self._preview_output_cache_key()
        if key is None:
            return
        display_label = (label if label is not None else self.preview_output_var.get()).strip()
        config = self.preview_output_to_config.get(display_label)
        output = self.preview_output_to_output.get(display_label)
        if not display_label or not output or not config:
            return
        cache = self.settings.setdefault("previewOutputByVehicle", {})
        if not isinstance(cache, dict):
            cache = {}
            self.settings["previewOutputByVehicle"] = cache
        cache[key] = {"output": output, "config": config}
        core.save_app_settings(self.settings)

    def _preview_output_selected(self) -> None:
        self.preview_output_hover = None
        self._remember_preview_output()
        self._schedule_mesh_scene(immediate=True)
        # The x/y/z columns and the box viewer both show the previewed trim's
        # positions, so they have to follow the Config dropdown.
        self._refresh_box_preview()
        self._refresh_parts()

    def _selected_preview_output_name(self) -> str:
        label = (self.preview_output_hover or self.preview_output_var.get()).strip()
        return self.preview_output_to_output.get(label, "")

    def _wire_preview_output_popdown(self) -> None:
        """Hot-load trims while scrolling the Config dropdown. The ttk
        combobox popdown listbox is a plain Tcl widget with no Python wrapper;
        watch it via its <Map> event and poll the highlighted entry while it
        stays open."""
        combo = self.preview_output_combo
        try:
            popdown = str(combo.tk.call("ttk::combobox::PopdownWindow", combo))
            listbox = f"{popdown}.f.l"
            if not int(combo.tk.call("winfo", "exists", listbox)):
                return
            start = combo.register(self._start_preview_hover_watch)
            combo.tk.call("bind", listbox, "<Map>", f"+{start}")
        except tk.TclError:
            return
        self._preview_popdown_listbox = listbox

    def _start_preview_hover_watch(self) -> None:
        if self._preview_hover_after is not None:
            try:
                self.after_cancel(self._preview_hover_after)
            except Exception:
                pass
            self._preview_hover_after = None
        self._preview_hover_poll()

    def _preview_hover_poll(self) -> None:
        self._preview_hover_after = None
        combo = self.preview_output_combo
        listbox = self._preview_popdown_listbox
        mapped = False
        label = None
        if listbox is not None:
            try:
                mapped = bool(int(combo.tk.call("winfo", "ismapped", listbox)))
                if mapped:
                    selection = combo.tk.call(listbox, "curselection")
                    if selection:
                        index = selection[0] if isinstance(selection, (tuple, list)) else selection
                        label = str(combo.tk.call(listbox, "get", index))
            except tk.TclError:
                mapped = False
        if not mapped:
            self._end_preview_hover_watch()
            return
        if (
            label
            and label in self.preview_output_to_config
            and label != (self.preview_output_hover or self.preview_output_var.get())
        ):
            self.preview_output_hover = label
            self._schedule_mesh_scene(immediate=True)
        self._preview_hover_after = self.after(90, self._preview_hover_poll)

    def _end_preview_hover_watch(self) -> None:
        if self.preview_output_hover is None:
            return
        self.preview_output_hover = None
        # Confirming fires <<ComboboxSelected>> with the same trim already
        # loaded (snapshot-guarded no-op); after a cancel this restores the
        # preview of the actual selection.
        self._schedule_mesh_scene(immediate=True)

    def _variant_detection_signature(self) -> tuple[str, ...]:
        return core.variant_hand_detection_signature(self.conversion)

    def _invalidate_variant_detection(self) -> None:
        self.variant_detected_hands = {}
        self.variant_detection_complete = False
        self.mesh_scene_hash = None

    def _schedule_variant_detection(self) -> None:
        if self.context is None:
            return
        if self.variant_detection_complete:
            return
        if self.variant_detection_running:
            self.variant_detection_pending = True
            return
        self._start_variant_detection()

    def _start_variant_detection(self) -> None:
        if self.context is None:
            return
        self.variant_detection_running = True
        self.variant_detection_pending = False
        self.variant_detection_seq += 1
        seq = self.variant_detection_seq
        context = self.context
        signature = self._variant_detection_signature()
        conversion_copy = json.loads(json.dumps(self.conversion, default=str))
        future = self.variant_detector.submit(self._variant_detection_worker, context, conversion_copy)
        future.add_done_callback(
            lambda completed, current_seq=seq, current_context=context, current_signature=signature: self.worker_queue.put(
                ("variant_hands_done", (current_seq, current_context, current_signature, completed))
            )
        )

    @staticmethod
    def _variant_detection_worker(
        context: core.VehicleContext,
        conversion: dict[str, object],
    ) -> dict[str, str]:
        return core.detect_hands_for_variants(context, conversion)

    def _handle_variant_hands_done(self, payload: object) -> None:
        seq, context, signature, completed = payload
        self.variant_detection_running = False
        should_apply = (
            seq == self.variant_detection_seq
            and context is self.context
            and signature == self._variant_detection_signature()
        )
        try:
            detected = completed.result()
        except Exception as exc:
            if should_apply:
                self.variant_detected_hands = {}
                self.variant_detection_complete = True
                self._refresh_variants()
                self.status_var.set(f"Trim handedness detection failed: {exc}")
            if self.variant_detection_pending:
                self._schedule_variant_detection()
            return
        if should_apply:
            self.variant_detected_hands = {
                config_name: hand
                for config_name, hand in detected.items()
                if hand in {core.HAND_LHD, core.HAND_RHD, core.HAND_UNKNOWN}
            }
            self.variant_detection_complete = True
            self.mesh_scene_hash = None
            self._refresh_variants()
            self._schedule_mesh_scene(immediate=True)
        if self.variant_detection_pending:
            self._schedule_variant_detection()

    def _table_position(self, object_id: str) -> tuple[tuple[float, float, float], bool]:
        """Where the part's geometry actually sits, and whether that varies by trim.

        These are the drawn mesh's centre, not its DAE pivot. The pivot is
        meaningless for meshes authored in vehicle space with an identity node
        matrix -- a whole column of engine parts read 0,0,0 while rendering
        correctly. The box preview data is already the placed geometry for the
        trim on screen, so it is the same number the viewer draws."""
        if self.context is None:
            return ((0.0, 0.0, 0.0), False)
        obj = self.context.objects.get(object_id)
        if obj is None:
            return ((0.0, 0.0, 0.0), False)
        varies = object_id in self.context.variant_dependent_meshes
        entry = self.box_preview_by_id.get(object_id) or self.context.preview_by_id.get(object_id)
        centre = (entry or {}).get("center")
        if centre is not None:
            return (tuple(float(value) for value in centre), varies)
        # No geometry (prop-only rows with no mesh in the DAE): fall back to
        # the resolved pivot rather than showing nothing.
        config = self._mesh_scene_config()
        if config is not None and config in self.context.variants:
            try:
                resolved = core.resolved_mesh_positions_for_config(self.context, config).get(object_id)
            except Exception:
                resolved = None
            if resolved is not None:
                return (resolved.position, varies)
        return ((obj.x, obj.y, obj.z), varies)

    def _refresh_box_preview(self, *, force: bool = False) -> None:
        """Re-point the placed-geometry data at the previewed trim.

        Feeds the fallback box viewer and the table's x/y/z columns (the GPU
        scene builds its own geometry per config). Updated in place because
        ModelPreview holds the dict by reference, and skipped when the trim has
        not changed since it costs a pass over every mesh."""
        if self.context is None:
            self.box_preview_by_id.clear()
            self._box_preview_config = None
            return
        config = self._mesh_scene_config()
        if not force and config == self._box_preview_config and self.box_preview_by_id:
            return
        self._box_preview_config = config
        self.box_preview_by_id.clear()
        if config is None or config not in self.context.variants:
            self.box_preview_by_id.update(self.context.preview_by_id)
            return
        try:
            self.box_preview_by_id.update(
                core.preview_entries_for_config(self.context, config)
            )
        except Exception:
            self.box_preview_by_id.update(self.context.preview_by_id)

    def _variant_position_note(self, object_id: str) -> str:
        """Where else this part's geometry sits, for the marked (*) rows.

        Geometry centres, matching the x/y/z columns -- quoting pivots here
        would contradict them."""
        if self.context is None:
            return ""
        baked = (self.context.preview_by_id.get(object_id) or {}).get("center")
        obj = self.context.objects.get(object_id)
        if baked is None or obj is None:
            return ""
        representative = (obj.x, obj.y, obj.z)
        by_position: dict[tuple[float, ...], list[str]] = {}
        for config_name in sorted(self.context.variants):
            try:
                resolved = core.resolved_mesh_positions_for_config(self.context, config_name)
            except Exception:
                continue
            entry = resolved.get(object_id)
            if entry is None:
                continue
            # Same shift preview_entries_for_config applies, for one mesh:
            # building the whole per-config mapping here would copy every
            # mesh once per trim.
            key = tuple(
                round(float(baked[i]) + entry.position[i] - representative[i], 4)
                for i in range(3)
            )
            by_position.setdefault(key, []).append(config_name)
        if len(by_position) < 2:
            return ""
        shown = sorted(by_position.items(), key=lambda item: -len(item[1]))[:3]
        parts = [f"x {key[0]:.4f} y {key[1]:.4f} ({len(cfgs)} trims)" for key, cfgs in shown]
        more = "" if len(by_position) <= 3 else f", +{len(by_position) - 3} more"
        return f" | * varies by trim: {'; '.join(parts)}{more}"
