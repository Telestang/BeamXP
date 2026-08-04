from __future__ import annotations

from .shared import *


class BuildAndPreviewMixin:
    """Settings persistence, build launch, Blender preview, mesh-scene generation, and worker polling."""

    def _browse_mods_folder(self) -> None:
        initial = existing_initial_dir(
            self.settings.get("lastModsFolder") or self.mods_folder_var.get(),
            core.WORKSPACE_DIR,
        )
        path = self._ask_directory(title="Select BeamNG mods folder", initialdir=initial)
        if path:
            self.mods_folder_var.set(path)
            self.settings["lastModsFolder"] = path
            self._save_app_settings_from_ui()

    def _browse_blender(self) -> None:
        initial = existing_initial_dir(
            self.settings.get("lastBlenderFolder") or self.blender_var.get(),
            Path(r"C:\Program Files"),
        )
        path = self._ask_open_filename(
            title="Select blender.exe",
            initialdir=initial,
            filetypes=(("Executable", "*.exe"), ("All files", "*.*")),
        )
        if path:
            self.blender_var.set(path)
            self.settings["lastBlenderFolder"] = str(Path(path).parent)
            self._save_app_settings_from_ui()

    def _save_app_settings_from_ui(self) -> None:
        mods_folder = self.mods_folder_var.get().strip()
        blender_exe = self.blender_var.get().strip()
        self.settings["modsFolder"] = mods_folder
        self.settings["blenderExecutable"] = blender_exe
        if mods_folder:
            self.settings["lastModsFolder"] = mods_folder
        if blender_exe:
            self.settings["lastBlenderFolder"] = str(Path(blender_exe).parent)
        core.save_app_settings(self.settings)

    def _save_config(self) -> None:
        if self.context is None:
            return
        try:
            self._commit_delta_from_ui()
            path = core.save_conversion(self.context, self.conversion)
            self._save_app_settings_from_ui()
            self.status_var.set(f"Saved config: {path}")
        except Exception as exc:
            self._show_error("Save failed", str(exc))

    def _import_config_dialog(self) -> None:
        if self.context is None:
            return
        path = self._ask_open_filename(
            title="Import conversion config",
            initialdir=str(core.PROJECTS_DIR),
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            imported = json.loads(Path(path).read_text(encoding="utf-8"))
            self.conversion, counts = core.import_matching_conversion(
                self.context,
                self.conversion,
                imported,
            )
            self._sync_delta_to_ui()
            self._invalidate_variant_detection()
            self._refresh_all(reset_view=True)
            self._schedule_variant_detection()
            self.status_var.set(
                "Imported matched settings: "
                f"{counts['variantImported']} variant(s), {counts['partImported']} part(s); "
                f"dropped {counts['variantSkipped']} variant(s), {counts['partSkipped']} part(s)"
            )
        except Exception as exc:
            self._show_error("Import failed", str(exc))

    def _set_busy(self, busy: bool) -> None:
        self.worker_running = busy
        state = "disabled" if busy else "normal"
        self.install_button.configure(state=state)
        self.blender_button.configure(state=state)
        if hasattr(self, "busy_progress"):
            if busy:
                self.busy_progress.grid()
                self.busy_progress.start(12)
            else:
                self.busy_progress.stop()
                self.busy_progress.grid_remove()
        if hasattr(self, "preview_output_combo"):
            if busy or not self.preview_output_to_config:
                self.preview_output_combo.configure(state="disabled")
            else:
                self.preview_output_combo.configure(state="readonly")

    def _start_build(self, *, install: bool) -> None:
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        if install and not self.mods_folder_var.get().strip():
            self._show_error("No mods folder", "Set a BeamNG mods folder before installing.")
            return
        self._commit_delta_from_ui()
        self._save_app_settings_from_ui()
        self._set_busy(True)
        self.status_var.set("Building XP conversion zip...")
        worker = threading.Thread(target=self._build_worker, args=(install,), daemon=True)
        worker.start()

    def _build_worker(self, install: bool) -> None:
        assert self.context is not None
        try:
            result = core.build_batch(
                self.context,
                self.conversion,
                write_zip=True,
                install=install,
                mods_folder=Path(self.mods_folder_var.get()) if install else None,
                progress=lambda message: self.worker_queue.put(("status", message)),
            )
            self.worker_queue.put(("build_success", result))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _start_blender_preview(self) -> None:
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        blender = self._resolve_blender()
        if blender is None:
            self._show_error("Blender not found", "Set the Blender executable path first.")
            return
        config_label = self.preview_output_var.get().strip()
        output_name = self.preview_output_to_output.get(config_label)
        if not output_name or config_label not in self.preview_output_to_config:
            self._show_error(
                "No config",
                "Select a buildable config in the Config dropdown.",
            )
            return
        self._commit_delta_from_ui()
        self._save_app_settings_from_ui()
        self._set_busy(True)
        self.status_var.set(f"Preparing Blender preview for {config_label}...")
        worker = threading.Thread(
            target=self._blender_preview_worker,
            args=(blender, output_name),
            daemon=True,
        )
        worker.start()

    def _resolve_blender(self) -> Path | None:
        configured = self.blender_var.get().strip()
        if configured and Path(configured).exists():
            return Path(configured)
        for candidate in BLENDER_CANDIDATES:
            if candidate.exists():
                self.blender_var.set(str(candidate))
                return candidate
        return None

    def _mesh_scene_config(self) -> str | None:
        # The dropdown's highlighted-but-unconfirmed entry wins while the
        # list is open, so trims hot-load as you scroll through them.
        label = (self.preview_output_hover or self.preview_output_var.get()).strip()
        config = self.preview_output_to_config.get(label)
        if config:
            return config
        selected = self._selected_variant_names()
        if selected:
            return selected[0]
        if self.context is not None and self.context.variants:
            return next(iter(self.context.variants))
        return None

    def _mesh_scene_snapshot(self) -> str | None:
        """Fingerprint of everything the 3D scene depends on. Viewer-only
        flags (visibility/solo) are excluded - those only filter the index
        buffer and never need a rebuild."""
        with timed_ui("_mesh_scene_snapshot"):
            config = self._mesh_scene_config()
            if config is None:
                return None
            conversion = json.loads(json.dumps(self.conversion, default=str))
            parts = conversion.get("parts")
            if isinstance(parts, dict):
                for settings in parts.values():
                    if isinstance(settings, dict):
                        settings.pop("viewerVisible", None)
                        settings.pop("viewerSolo", None)
            return json.dumps(
                {
                    "config": config,
                    "output": self._selected_preview_output_name(),
                    "conversion": conversion,
                    # Selected-but-inactive parts are injected into the scene, so
                    # the scene must rebuild when that set changes (and only then;
                    # selection moves between active parts leave it empty/equal).
                    "extra": self._selected_extra_preview_ids(),
                },
                sort_keys=True,
            )

    def _schedule_mesh_scene(self, *, immediate: bool = False) -> None:
        if self.context is None or not self.viewer_supports_scene:
            return
        snapshot = self._mesh_scene_snapshot()
        if snapshot is None:
            return
        if snapshot == self.mesh_scene_hash:
            return
        if self.mesh_scene_running:
            self.mesh_scene_pending = True
            return
        if self.mesh_scene_after is not None:
            return
        if immediate:
            self._start_mesh_scene()
        else:
            self.mesh_scene_after = self.after_idle(self._start_mesh_scene)

    def _start_mesh_scene(self) -> None:
        self.mesh_scene_after = None
        if self.context is None or not self.viewer_supports_scene or self.viewer is None:
            return
        snapshot = self._mesh_scene_snapshot()
        config = self._mesh_scene_config()
        if snapshot is None or config is None:
            return
        if snapshot == self.mesh_scene_hash:
            return
        self.mesh_scene_hash = snapshot
        self.mesh_scene_seq += 1
        seq = self.mesh_scene_seq
        context = self.context
        conversion_copy = json.loads(json.dumps(self.conversion, default=str))
        # The in-app preview represents one source trim, independently of how
        # many build outputs were requested. Always prepare both transformed
        # and original-layout vertex buffers; the viewer checkbox switches
        # between them while the resolved replacement plates remain the same.
        settings = conversion_copy.get("variants", {}).get(config, {})
        if isinstance(settings, dict):
            core.set_variant_build_mode(settings, core.BUILD_CONVERTED)
        self.viewer.set_message(f"building preview: {config}...")
        self.mesh_scene_running = True
        extra_meshes = tuple(self._selected_extra_preview_ids())
        future = self.part_resolver.submit(
            self._mesh_scene_worker, context, conversion_copy, config, extra_meshes
        )
        future.add_done_callback(
            lambda completed, current_seq=seq, current_snapshot=snapshot: self.worker_queue.put(
                ("mesh_scene_done", (current_seq, current_snapshot, completed))
            )
        )

    @staticmethod
    def _mesh_scene_worker(
        context: core.VehicleContext,
        conversion: dict[str, object],
        config_name: str,
        extra_meshes: tuple[str, ...] = (),
    ):
        payload = core.full_vehicle_preview_payload(
            context,
            conversion,
            config_name,
            context.project_dir / "blender_preview",
            extra_meshes=extra_meshes,
        )
        cache_dir = context.project_dir / "blender_preview" / "dae_cache" / "mesh_cache"
        return mesh_preview.build_scene(payload, cache_dir)

    def _handle_mesh_scene_done(self, payload: object) -> None:
        with timed_ui("_handle_mesh_scene_done"):
            seq, completed_snapshot, completed = payload
            self.mesh_scene_running = False
            should_apply = (
                seq == self.mesh_scene_seq
                and completed_snapshot == self._mesh_scene_snapshot()
                and self.viewer is not None
                and self.viewer_supports_scene
            )
            try:
                scene = completed.result()
            except Exception as exc:
                if should_apply and self.viewer is not None:
                    self.viewer.set_message(f"preview failed: {exc}")
                self._schedule_pending_mesh_scene()
                return
            if not should_apply:
                self._schedule_pending_mesh_scene()
                return
            assert self.viewer is not None
            reset_view = self.mesh_scene_reset_pending
            self.viewer.show_scene(scene, reset_view=False, apply_filters=False)
            self.mesh_scene_reset_pending = False
            # Replacement-source children can render as meshes that belong to the
            # replacement part rather than the original child row. Once the GPU
            # scene has its pick-to-row map, update only the preview-dependent
            # cells. A full parts-table rebuild here blocks Tk for large cars.
            self._refresh_preview_dependent_part_cells()
            self._refresh_viewer(reset=reset_view)
            self._schedule_pending_mesh_scene()

    def _schedule_pending_mesh_scene(self) -> None:
        if not self.mesh_scene_pending:
            return
        self.mesh_scene_pending = False
        self._schedule_mesh_scene(immediate=True)

    @staticmethod
    def _preview_needs_generated_output(
        context: core.VehicleContext,
        conversion: dict[str, object],
        config_name: str,
        output_name: str,
    ) -> bool:
        if output_name == core.original_plate_output_name(config_name):
            return True
        object_modes = core.active_part_modes(conversion)
        if not object_modes:
            return False
        _flex, _props, all_meshes = core.selected_mesh_roles(context, [config_name])
        return any(mesh in all_meshes for mesh in object_modes)

    def _blender_preview_worker(self, blender: Path, output_name: str) -> None:
        assert self.context is not None
        try:
            run_dir = self.context.project_dir / "blender_preview" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            output_sources = core.output_config_sources(self.context, self.conversion)
            config_name = output_sources.get(output_name)
            if config_name is None:
                raise RuntimeError(f"Unknown generated config {output_name!r}")
            if self._preview_needs_generated_output(self.context, self.conversion, config_name, output_name):
                result = core.build_batch(
                    self.context,
                    self.conversion,
                    write_zip=False,
                    install=False,
                    mods_folder=None,
                )
                if output_name not in result.generated_configs:
                    raise RuntimeError(f"Output {output_name!r} was not generated by the current settings")
                payload = core.output_vehicle_preview_payload(
                    self.context,
                    self.conversion,
                    output_name,
                    result.unpacked_dir,
                    result.generated_daes,
                    run_dir,
                )
            else:
                payload = core.full_vehicle_preview_payload(
                    self.context,
                    self.conversion,
                    config_name,
                    run_dir,
                )
                payload["output_name"] = output_name
                payload["show_unchanged"] = True
            payload_path = run_dir / "blender_preview_payload.json"
            payload_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            subprocess.Popen(
                [
                    str(blender),
                    "--python",
                    str(BLENDER_PREVIEW_SCRIPT),
                    "--",
                    str(payload_path),
                ],
                cwd=str(THIS_DIR),
            )
            self.worker_queue.put(("preview_success", payload_path))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _poll_worker_queue(self) -> None:
        handled = False
        while True:
            try:
                kind, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            handled = True
            self._handle_worker_message(kind, payload)
        self.after(40 if handled else 80, self._poll_worker_queue)
