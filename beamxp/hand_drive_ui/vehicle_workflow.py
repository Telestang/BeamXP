from __future__ import annotations

from .shared import *


class VehicleWorkflowMixin:
    """Source selection, recent-model history, loading, viewer replacement, and global plate state."""

    def _open_zip_dialog(self) -> None:
        initial = existing_initial_dir(self.settings.get("lastVehicleZipFolder"), core.WORKSPACE_DIR)
        path = self._ask_open_filename(
            title="Open BeamNG vehicle zip",
            initialdir=initial,
            filetypes=(("Zip files", "*.zip"), ("All files", "*.*")),
        )
        if path:
            self._load_source_zip(Path(path))

    # ----- Model dropdown history -----------------------------------------

    def _recent_vehicle_entries(self) -> list[tuple[Path, str]]:
        """Persisted (zip, vehicle id) history, newest first, malformed rows
        dropped. Missing zips are kept so the history survives an unplugged
        drive; they are handled when actually selected."""
        recent = self.settings.get("recentVehicles")
        if not isinstance(recent, list):
            return []
        entries: list[tuple[Path, str]] = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            zip_str = str(item.get("zip") or "")
            vehicle_id = str(item.get("vehicleId") or "")
            if zip_str and vehicle_id:
                entries.append((Path(zip_str), vehicle_id))
        return entries

    def _record_recent_vehicle(self, source_zip: Path, vehicle_id: str) -> None:
        zip_str = str(source_zip)
        recent = self.settings.get("recentVehicles")
        if not isinstance(recent, list):
            recent = []
        deduped = [
            item
            for item in recent
            if isinstance(item, dict)
            and not (str(item.get("zip")) == zip_str and str(item.get("vehicleId")) == vehicle_id)
        ]
        deduped.insert(0, {"zip": zip_str, "vehicleId": vehicle_id})
        self.settings["recentVehicles"] = deduped

    def _prune_recent_vehicle(self, source_zip: Path, vehicle_id: str) -> None:
        zip_str = str(source_zip)
        recent = self.settings.get("recentVehicles")
        if not isinstance(recent, list):
            return
        self.settings["recentVehicles"] = [
            item
            for item in recent
            if isinstance(item, dict)
            and not (str(item.get("zip")) == zip_str and str(item.get("vehicleId")) == vehicle_id)
        ]
        core.save_app_settings(self.settings)

    @staticmethod
    def _model_history_label(zip_path: Path, vehicle_id: str, taken: dict[str, object]) -> str:
        label = f"{vehicle_id}  ({zip_path.stem})"
        base = label
        suffix = 2
        while label in taken:
            label = f"{base} #{suffix}"
            suffix += 1
        return label

    def _load_source_zip(self, source_zip: Path, vehicle_id: str | None = None) -> None:
        try:
            vehicle_ids = core.vehicle_ids_in_zip(source_zip)
            if not vehicle_ids:
                raise RuntimeError("No vehicles/<model>/ content with DAE/PC/JBeam files was found")
            self.source_zip = source_zip
            self.settings["lastVehicleZipFolder"] = str(source_zip.parent)
            core.save_app_settings(self.settings)
            self.vehicle_ids = vehicle_ids
            self.source_var.set(str(source_zip))
            selected_vehicle = vehicle_id if vehicle_id in vehicle_ids else vehicle_ids[0]
            self._rebuild_model_combo()
            self.vehicle_var.set(self._combo_label_for(source_zip, selected_vehicle))
            self.last_model_label = self.vehicle_var.get()
            self._show_model_preview(self.vehicle_var.get())
            self._load_selected_vehicle(vehicle_id=selected_vehicle)
        except Exception as exc:
            self._show_error("Open zip failed", str(exc))
            self.status_var.set("Open zip failed")

    @staticmethod
    def _same_zip(left: Path, right: Path) -> bool:
        """Compare zip paths the way the filesystem does.

        The settings file and the folder scan can disagree on case and
        separators for the same file, and a mismatch silently strands the
        loaded vehicle outside its own dropdown entry.
        """
        return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
            os.path.normpath(str(right))
        )

    def _combo_label_for(self, source_zip: Path, vehicle_id: str) -> str:
        """The dropdown label standing for (zip, vehicle), or the bare id."""
        for label, (zip_path, vid) in self.model_entries.items():
            if vid == vehicle_id and self._same_zip(zip_path, source_zip):
                return label
        return vehicle_id

    def _load_selected_vehicle(
        self, *, force_reload: bool = False, vehicle_id: str | None = None
    ) -> None:
        if self.source_zip is None:
            return
        if vehicle_id is None:
            # The combo shows display names ("ETK 800-Series"), so map back.
            label = self.vehicle_var.get()
            entry = self.model_entries.get(label)
            vehicle_id = entry[1] if entry else label
        if not vehicle_id:
            vehicle_id = self.vehicle_ids[0] if self.vehicle_ids else None
        if not vehicle_id:
            return
        self.vehicle_load_seq += 1
        seq = self.vehicle_load_seq
        if force_reload:
            self.status_var.set(f"Re-scanning vehicles/{vehicle_id} (ignoring cache)...")
        else:
            self.status_var.set(f"Loading vehicles/{vehicle_id}...")
        self._set_load_busy(True)
        worker = threading.Thread(
            target=self._vehicle_load_worker,
            args=(self.source_zip, vehicle_id, force_reload, seq),
            daemon=True,
        )
        worker.start()

    def _vehicle_load_worker(
        self,
        source_zip: Path,
        vehicle_id: str,
        force_reload: bool,
        seq: int,
    ) -> None:
        try:
            context = core.load_vehicle_context(source_zip, vehicle_id, use_cache=not force_reload)
            if force_reload:
                core.clear_parts_cache(context)
                core.clear_variant_hands_cache(context)
                core.clear_part_table_caches(context)
            conversion, loaded = core.load_or_create_conversion(context)
            self.worker_queue.put(
                ("vehicle_load_success", (seq, source_zip, vehicle_id, context, conversion, loaded))
            )
        except Exception as exc:
            self.worker_queue.put(("vehicle_load_error", (seq, exc)))

    def _set_load_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.open_button.configure(state=state)
        self.refresh_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.recommend_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.new_side_pair_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.remove_side_pair_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.clear_slot_pairs_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.reset_trigger_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.clear_triggers_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.toggle_triggers_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.show_all_parts_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.hide_all_parts_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.active_only_parts_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.clear_solo_parts_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.model_load_busy = busy
        self._update_model_combo_state()
        self._set_busy(busy)

    def _handle_vehicle_load_success(self, payload: object) -> None:
        seq, source_zip, vehicle_id, context, conversion, loaded = payload
        if seq != self.vehicle_load_seq:
            return
        self.context = context
        self.conversion = conversion
        self.preview_output_var.set("")
        cached_hands = core.load_cached_variant_hands(context, conversion) or {}
        self.variant_detected_hands = cached_hands
        self.variant_detection_complete = all(name in cached_hands for name in context.variants)
        self.variant_detection_pending = False
        self.settings["lastVehicleZipPath"] = str(source_zip)
        self.settings["lastVehicleId"] = vehicle_id
        self._record_recent_vehicle(source_zip, vehicle_id)
        core.save_app_settings(self.settings)
        self._rebuild_model_combo()
        self.vehicle_var.set(self._combo_label_for(source_zip, vehicle_id))
        self.last_model_label = self.vehicle_var.get()
        self._show_model_preview(self.vehicle_var.get())
        self.part_refresh_seq += 1
        self.resolved_part_ids = []
        self.current_part_ids = []
        self.part_columns_need_fit = True
        self.mesh_instance_numbering_key = None
        self.mesh_instance_numbering_cache = {}
        self.mesh_instance_keys_cache = {}
        self._clear_mesh_instance_cache()
        self.mesh_scene_hash = None
        self.mesh_scene_reset_pending = True
        self._set_load_busy(False)
        self._sync_delta_to_ui()
        self._sync_plate_to_ui()
        self._sync_texture_quality_to_ui()
        self._replace_viewer()
        self._refresh_all(reset_view=True)
        if not self.variant_detection_complete:
            self._schedule_variant_detection()
        self._schedule_mesh_scene(immediate=True)
        loaded_text = "loaded exact project config" if loaded else "new project config"
        self.project_var.set(f"Project: {context.project_dir} ({loaded_text})")
        from_cache = " (from cache)" if getattr(context, "loaded_from_cache", False) else ""
        self.status_var.set(
            f"Loaded {context.vehicle_id}{from_cache}: {len(context.variants)} variant(s), "
            f"{len(context.objects)} DAE object(s)"
        )

    def _handle_vehicle_load_error(self, payload: object) -> None:
        seq, exc = payload
        if seq != self.vehicle_load_seq:
            return
        self._set_load_busy(False)
        self._show_error("Load vehicle failed", str(exc))
        self.status_var.set("Load vehicle failed")

    def _replace_viewer(self) -> None:
        if self.viewer is not None and self.viewer_supports_scene:
            try:
                self.viewer.destroy()  # releases the GL context
            except Exception:
                pass
        for child in self.viewer_holder.winfo_children():
            child.destroy()
        self.viewer = None
        self.viewer_supports_scene = False
        if self.context is None:
            return
        if mesh_preview is not None:
            try:
                self.viewer = mesh_preview.MeshPreview(self.viewer_holder)
                self.viewer_supports_scene = True
                self.viewer.on_pick = self._on_preview_pick
                self.viewer.set_message("building preview...")
            except Exception as exc:
                print(f"[preview] GPU mesh preview unavailable ({exc}); using box preview")
                self.viewer = None
        if self.viewer is None:
            # The box viewer reads this dict live, so it is refreshed in place
            # whenever the previewed trim changes (see _refresh_box_preview).
            self._refresh_box_preview()
            self.viewer = ModelPreview(self.viewer_holder, self.box_preview_by_id)
        self.viewer.grid(row=0, column=0, sticky="nsew")
        # A fresh viewer starts with the boxes drawn, so carry the toggle over:
        # loading another vehicle must not put back what was hidden.
        self._apply_trigger_visibility()

    def _sync_delta_to_ui(self) -> None:
        delta = self.conversion.get("delta", {})
        if not isinstance(delta, dict):
            delta = {}
            self.conversion["delta"] = delta
        self.manual_delta_enabled.set(bool(delta.get("manual")))
        magnitude = delta.get("magnitude")
        self.manual_delta_var.set("" if magnitude in (None, "") else fmt_float(abs(float(magnitude))))
        self._manual_delta_toggled(refresh=False)

    def _sync_texture_quality_to_ui(self) -> None:
        self.conversion["textureQuality"] = core.texture_quality_setting(self.conversion)
        self._refresh_texture_quality_choices()

    def _refresh_texture_quality_choices(self) -> None:
        if not hasattr(self, "texture_quality_combo"):
            return
        self.texture_quality_to_tier = {
            core.TEXTURE_QUALITY_LABELS[tier]: tier
            for tier in core.BC7_QUALITY_TIERS
        }
        labels = list(self.texture_quality_to_tier)
        self.texture_quality_combo.configure(values=labels)
        current = core.texture_quality_setting(self.conversion)
        self.texture_quality_var.set(core.TEXTURE_QUALITY_LABELS[current])

    def _texture_quality_changed(self) -> None:
        tier = self.texture_quality_to_tier.get(
            self.texture_quality_var.get(), core.DEFAULT_BC7_QUALITY
        )
        self.conversion["textureQuality"] = tier
        self._refresh_texture_quality_choices()
        self.status_var.set(
            f"Texture quality: {core.TEXTURE_QUALITY_LABELS[tier]}"
        )

    def _sync_plate_to_ui(self) -> None:
        self.conversion["plate"] = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        self._refresh_plate_choices()
        self._refresh_plate_summary()

    def _refresh_plate_choices(self) -> None:
        if not hasattr(self, "plate_choice_combo"):
            return
        records = plate_generator.plate_set_records()
        custom_label = self._vehicle_custom_label()
        self.plate_choice_to_id = {"Off": "", custom_label: ""}
        values = ["Off", custom_label]
        for record in records:
            label = str(record["name"])
            if label in self.plate_choice_to_id:
                label = f"{label} ({record['id']})"
            values.append(label)
            self.plate_choice_to_id[label] = str(record["id"])
        self.plate_choice_combo.configure(values=values)
        binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        if binding["mode"] == plate_generator.PLATE_MODE_SET:
            set_id = str(binding.get("setId") or "")
            selected = next((label for label, value in self.plate_choice_to_id.items() if value == set_id), f"Missing set: {set_id}")
        elif binding["mode"] == plate_generator.PLATE_MODE_CUSTOM:
            selected = custom_label
        else:
            selected = "Off"
        self.plate_choice_var.set(selected)
        self.plate_configure_button.configure(state="disabled" if selected == "Off" else "normal")

    def _main_plate_choice_changed(self) -> None:
        label = self.plate_choice_var.get()
        custom_label = self._vehicle_custom_label()
        binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        if label == "Off":
            binding["mode"] = plate_generator.PLATE_MODE_OFF
            binding["setId"] = ""
        elif label == custom_label:
            binding["mode"] = plate_generator.PLATE_MODE_CUSTOM
            binding["setId"] = ""
            binding["customDefined"] = True
            binding["config"] = plate_generator.normalized_plate_config(binding.get("customConfig"))
        else:
            set_id = self.plate_choice_to_id.get(label, "")
            record = plate_generator.plate_set_by_id(set_id)
            if record is not None:
                binding["mode"] = plate_generator.PLATE_MODE_SET
                binding["setId"] = set_id
                binding["config"] = plate_generator.normalized_plate_config(record.get("config"))
        self.conversion["plate"] = binding
        self._refresh_plate_summary()
        self._refresh_plate_choices()
        self._refresh_variants()
        self._update_detail()
        self.status_var.set(f"Licence plates: {plate_generator.plate_summary_label(self.conversion)}")

    def _vehicle_custom_label(self) -> str:
        vehicle_id = self.context.vehicle_id if self.context is not None else "vehicle"
        return f"Custom ({vehicle_id})"

    def _vehicle_plate_label(self) -> str:
        binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        if binding["mode"] == plate_generator.PLATE_MODE_CUSTOM:
            return self._vehicle_custom_label()
        if binding["mode"] == plate_generator.PLATE_MODE_SET:
            set_id = str(binding.get("setId") or "")
            record = plate_generator.plate_set_by_id(set_id)
            return f"Set: {record['name']} (vehicle)" if record else f"Missing set: {set_id} (vehicle)"
        return "Off (vehicle)"

    def _refresh_all(self, *, reset_view: bool = False) -> None:
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=reset_view)
        self._refresh_delta_label()
        self._update_detail()
