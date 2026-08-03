from __future__ import annotations

import io

from beamxp.core import inventory as vehicle_inventory

from .shared import *

# Thumbnail box beside the Model dropdown. Previews are landscape, so the
# height is what actually constrains the top bar.
PREVIEW_THUMB_SIZE = (128, 72)
# Non-selectable spacer between pinned recents and the full list.
RECENT_SEPARATOR = "─" * 18


class VehicleBrowserMixin:
    """Folder-driven vehicle picker: folder buttons, the model dropdown built
    from those folders, and the preview thumbnail beside it."""

    # ----- configured folders ---------------------------------------------

    def _game_vehicles_folder(self) -> Path | None:
        raw = str(self.settings.get("gameVehiclesFolder") or "").strip()
        if raw:
            return Path(raw)
        found = core.default_beamng_vehicles_dir()
        if found is not None:
            # Remember the auto-detected folder so the button shows a path.
            self.settings["gameVehiclesFolder"] = str(found)
        return found

    def _mods_folder(self) -> Path | None:
        raw = self.mods_folder_var.get().strip()
        return Path(raw) if raw else None

    @staticmethod
    def _folder_button_text(prefix: str, folder: Path | None) -> str:
        """Enough tail of the path to tell two installs apart.

        The leaf alone is useless here -- every install ends in
        .../content/vehicles and .../current/mods.
        """
        if folder is None or not str(folder).strip():
            return f"{prefix}: set folder..."
        parts = [part for part in folder.parts if part not in ("/", "\\")]
        tail = "/".join(parts[-3:]) if len(parts) > 3 else str(folder)
        return f"{prefix}: {tail}"

    def _refresh_folder_buttons(self) -> None:
        game_folder = self._game_vehicles_folder()
        mods_folder = self._mods_folder()
        self.game_folder_button.configure(
            text=self._folder_button_text("Game vehicles", game_folder)
        )
        self.mods_folder_button.configure(text=self._folder_button_text("Mods", mods_folder))
        missing = [
            name
            for name, folder in (("game vehicles", game_folder), ("mods", mods_folder))
            if folder is None or not Path(folder).is_dir()
        ]
        self.folder_hint_var.set(
            f"Set the {' and '.join(missing)} folder to list vehicles" if missing else ""
        )

    def _browse_game_vehicles_folder(self) -> None:
        current = self._game_vehicles_folder()
        initial = existing_initial_dir(current, core.WORKSPACE_DIR)
        path = self._ask_directory(
            title="Select the BeamNG content/vehicles folder", initialdir=initial
        )
        if not path:
            return
        self.settings["gameVehiclesFolder"] = path
        core.save_app_settings(self.settings)
        self._refresh_folder_buttons()
        self._start_inventory_scan()

    def _browse_mods_folder_and_rescan(self) -> None:
        self._browse_mods_folder()
        self.settings["modsFolder"] = self.mods_folder_var.get().strip()
        core.save_app_settings(self.settings)
        self._refresh_folder_buttons()
        self._start_inventory_scan()

    def _on_include_automation_toggled(self) -> None:
        self.settings["includeAutomationVehicles"] = bool(self.include_automation_var.get())
        core.save_app_settings(self.settings)
        self._start_inventory_scan()

    # ----- scanning --------------------------------------------------------

    def _start_inventory_scan(self) -> None:
        """Rescan both folders off the UI thread; ~1s over 190 zips."""
        self.inventory_scan_seq += 1
        seq = self.inventory_scan_seq
        game_folder = self._game_vehicles_folder()
        mods_folder = self._mods_folder()
        include_automation = bool(self.include_automation_var.get())
        if game_folder is None and mods_folder is None:
            self.vehicle_listings = []
            self._rebuild_model_combo()
            return
        self.status_var.set("Scanning vehicle folders...")
        worker = threading.Thread(
            target=self._inventory_scan_worker,
            args=(game_folder, mods_folder, include_automation, seq),
            daemon=True,
        )
        worker.start()

    def _inventory_scan_worker(
        self,
        game_folder: Path | None,
        mods_folder: Path | None,
        include_automation: bool,
        seq: int,
    ) -> None:
        try:
            listings = vehicle_inventory.scan_vehicle_inventory(
                game_folder, mods_folder, include_automation=include_automation
            )
            self.worker_queue.put(("inventory_scan_done", (seq, listings)))
        except Exception as exc:  # noqa: BLE001 - a bad folder must not kill the app
            self.worker_queue.put(("inventory_scan_error", (seq, exc)))

    def _handle_inventory_scan_done(self, payload: object) -> None:
        seq, listings = payload
        if seq != self.inventory_scan_seq:
            return  # a newer scan is already in flight
        self.vehicle_listings = list(listings)
        self.preview_photo_cache.clear()
        self._rebuild_model_combo()
        mods = sum(1 for item in self.vehicle_listings if item.is_mod)
        self.status_var.set(
            f"{len(self.vehicle_listings)} vehicle(s) available "
            f"({len(self.vehicle_listings) - mods} stock, {mods} mod)"
        )

    def _handle_inventory_scan_error(self, payload: object) -> None:
        seq, exc = payload
        if seq != self.inventory_scan_seq:
            return
        self.vehicle_listings = []
        self._rebuild_model_combo()
        self.status_var.set(f"Vehicle folder scan failed: {exc}")

    # ----- dropdown --------------------------------------------------------

    def _rebuild_model_combo(self) -> None:
        """Model dropdown: recently converted vehicles pinned above the full
        folder-scanned list, plus anything from a directly opened zip that the
        folders do not cover."""
        entries: dict[str, tuple[Path, str]] = {}
        labels: dict[tuple[str, str], str] = {}
        for listing in self.vehicle_listings:
            label = listing.label()
            base = label
            suffix = 2
            while label in entries:
                label = f"{base} #{suffix}"
                suffix += 1
            entries[label] = (listing.source_zip, listing.vehicle_id)
            labels[(os.path.normcase(str(listing.source_zip)), listing.vehicle_id)] = label

        # A zip opened via Load Zip may sit outside both folders.
        if self.source_zip is not None:
            for vid in self.vehicle_ids:
                key = (os.path.normcase(str(self.source_zip)), vid)
                if key in labels:
                    continue
                label = self._model_history_label(self.source_zip, vid, entries)
                entries[label] = (self.source_zip, vid)
                labels[key] = label

        recent_labels: list[str] = []
        for zip_path, vid in self._recent_vehicle_entries():
            label = labels.get((os.path.normcase(str(zip_path)), vid))
            if label and label not in recent_labels:
                recent_labels.append(label)
            if len(recent_labels) >= 6:
                break

        # Recents are a shortcut, not a move: every vehicle still appears in the
        # sorted section below, so the list is predictable to scroll.
        rest = list(entries)
        values = recent_labels + ([RECENT_SEPARATOR] if recent_labels and rest else []) + rest

        self.model_entries = entries
        self.vehicle_combo.configure(values=values)
        self._update_model_combo_state()
        self._wire_model_popdown()
        self._resync_selected_label()
        self._show_model_preview(self.vehicle_var.get())

    def _resync_selected_label(self) -> None:
        """Re-resolve the shown label against the current entries.

        Startup opens the last vehicle before the folder scan finishes, so the
        box would otherwise keep the bare id ("etk800") it was set to when no
        listings existed yet.
        """
        if self.context is None or self.source_zip is None:
            return
        label = self._combo_label_for(self.source_zip, self.context.vehicle_id)
        if label and label != self.vehicle_var.get():
            self.vehicle_var.set(label)
            self.last_model_label = label

    def _update_model_combo_state(self) -> None:
        count = len(self.vehicle_combo.cget("values"))
        if self.model_load_busy or count < 1:
            self.vehicle_combo.configure(state="disabled")
        else:
            self.vehicle_combo.configure(state="readonly")

    def _on_model_selected(self) -> None:
        label = self.vehicle_var.get()
        if label == RECENT_SEPARATOR:
            # Purely decorative: put the previous selection back so the box
            # never sits showing the divider.
            self.vehicle_var.set(self.last_model_label or self._label_for_loaded_vehicle())
            self._show_model_preview(self.vehicle_var.get())
            return
        self.last_model_label = label
        self._end_model_hover_watch()
        entry = self.model_entries.get(label)
        if entry is None:
            self._load_selected_vehicle()
            return
        zip_path, vehicle_id = entry
        if self.source_zip is not None and self._same_zip(zip_path, self.source_zip):
            self.vehicle_var.set(label)
            self._load_selected_vehicle(vehicle_id=vehicle_id)
            return
        if not zip_path.exists():
            self._show_error(
                "Vehicle unavailable",
                f"This zip no longer exists:\n{zip_path}",
            )
            self._prune_recent_vehicle(zip_path, vehicle_id)
            if self.context is not None:
                self.vehicle_var.set(self._label_for_loaded_vehicle())
            self._rebuild_model_combo()
            return
        self._load_source_zip(zip_path, vehicle_id)

    def _label_for_loaded_vehicle(self) -> str:
        """The dropdown label for the vehicle currently loaded, if it has one."""
        if self.context is None or self.source_zip is None:
            return ""
        for label, (zip_path, vid) in self.model_entries.items():
            if vid == self.context.vehicle_id and self._same_zip(zip_path, self.source_zip):
                return label
        return self.context.vehicle_id

    # ----- preview thumbnail ----------------------------------------------

    def _listing_for_label(self, label: str) -> object | None:
        entry = self.model_entries.get(label)
        if entry is None:
            return None
        zip_path, vehicle_id = entry
        for listing in self.vehicle_listings:
            if listing.vehicle_id == vehicle_id and self._same_zip(listing.source_zip, zip_path):
                return listing
        return None

    def _show_model_preview(self, label: str) -> None:
        photo = self._preview_photo_for(label)
        if photo is None:
            self.model_preview_label.configure(image="", text="no\npreview")
            self.model_preview_photo = None
            return
        self.model_preview_label.configure(image=photo, text="")
        # Tk drops an image with no live Python reference.
        self.model_preview_photo = photo

    def _preview_photo_for(self, label: str):
        if not label or label == RECENT_SEPARATOR:
            return None
        if label in self.preview_photo_cache:
            return self.preview_photo_cache[label]
        listing = self._listing_for_label(label)
        photo = None
        if listing is not None:
            blob = vehicle_inventory.read_preview_image_bytes(listing)
            if blob:
                try:
                    from PIL import Image, ImageTk

                    image = Image.open(io.BytesIO(blob))
                    image.thumbnail(PREVIEW_THUMB_SIZE)
                    photo = ImageTk.PhotoImage(image.convert("RGB"))
                except Exception:
                    photo = None
        self.preview_photo_cache[label] = photo
        return photo

    # ----- highlight-before-commit hover ----------------------------------

    def _wire_model_popdown(self) -> None:
        """Update the thumbnail as the dropdown highlight moves, before the user
        commits. Mirrors the Config dropdown's trim hot-load: the popdown listbox
        is a bare Tcl widget, so watch its <Map> and poll while it is open."""
        if self._model_popdown_listbox is not None:
            return
        combo = self.vehicle_combo
        try:
            popdown = str(combo.tk.call("ttk::combobox::PopdownWindow", combo))
            listbox = f"{popdown}.f.l"
            if not int(combo.tk.call("winfo", "exists", listbox)):
                return
            start = combo.register(self._start_model_hover_watch)
            combo.tk.call("bind", listbox, "<Map>", f"+{start}")
        except tk.TclError:
            return
        self._model_popdown_listbox = listbox

    def _start_model_hover_watch(self) -> None:
        if self._model_hover_after is not None:
            try:
                self.after_cancel(self._model_hover_after)
            except Exception:
                pass
            self._model_hover_after = None
        self._model_hover_poll()

    def _model_hover_poll(self) -> None:
        self._model_hover_after = None
        combo = self.vehicle_combo
        listbox = self._model_popdown_listbox
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
            self._end_model_hover_watch()
            return
        if label and label != (self.model_preview_hover or self.vehicle_var.get()):
            self.model_preview_hover = label
            self._show_model_preview(label)
        self._model_hover_after = self.after(90, self._model_hover_poll)

    def _end_model_hover_watch(self) -> None:
        if self.model_preview_hover is None:
            return
        self.model_preview_hover = None
        # Restores the committed selection's image after a cancelled dropdown.
        self._show_model_preview(self.vehicle_var.get())
