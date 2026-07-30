from __future__ import annotations

from .shared import *


class PlatesWorkflowMixin:
    """Per-variant plate choices and plate editor/library integration."""

    def _variant_plate_choices(self, config_name: str) -> tuple[list[str], dict[str, tuple[str, str, str]]]:
        mapping: dict[str, tuple[str, str, str]] = {
            self._vehicle_plate_label(): (plate_generator.PLATE_MODE_GENERAL, "", ""),
            f"Custom ({config_name})": (plate_generator.PLATE_MODE_CUSTOM, "", config_name),
        }
        variants = self.conversion.get("variants", {})
        if isinstance(variants, dict):
            for source_name, source_settings in sorted(variants.items()):
                if source_name == config_name or not isinstance(source_settings, dict):
                    continue
                source_binding = plate_generator.normalized_plate_binding(
                    source_settings.get("plate"), variant=True
                )
                if not source_binding.get("customDefined"):
                    continue
                mapping[f"Custom ({source_name})"] = (
                    plate_generator.PLATE_MODE_TRIM,
                    "",
                    str(source_name),
                )
        for record in plate_generator.plate_set_records():
            label = f"Set: {record['name']}"
            if label in mapping:
                label = f"{label} ({record['id']})"
            mapping[label] = (plate_generator.PLATE_MODE_SET, str(record["id"]), "")
        mapping["Off"] = (plate_generator.PLATE_MODE_OFF, "", "")
        return list(mapping), mapping

    def _set_variant_build_label(self, config_name: str, label: str) -> None:
        mode = next((key for key, value in BUILD_LABELS.items() if value == label), core.BUILD_OFF)
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            core.set_variant_build_mode(settings, mode)
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()

    def _set_variant_plate_choice(self, config_name: str, choice: tuple[str, str, str]) -> None:
        mode, set_id, source_config = choice
        settings = self.conversion.setdefault("variants", {}).setdefault(config_name, {})
        if not isinstance(settings, dict):
            return
        binding = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
        previous_mode = str(binding.get("mode"))
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            if previous_mode != plate_generator.PLATE_MODE_CUSTOM and not binding.get("customDefined"):
                if previous_mode == plate_generator.PLATE_MODE_SET:
                    record = plate_generator.plate_set_by_id(str(binding.get("setId") or ""))
                    copy_config = record.get("config") if record is not None else binding.get("config")
                elif previous_mode == plate_generator.PLATE_MODE_TRIM:
                    referenced = str(binding.get("sourceConfig") or "")
                    referenced_settings = self.conversion.get("variants", {}).get(referenced, {})
                    referenced_binding = plate_generator.normalized_plate_binding(
                        referenced_settings.get("plate") if isinstance(referenced_settings, dict) else None,
                        variant=True,
                    )
                    copy_config = referenced_binding.get("customConfig")
                else:
                    general = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
                    copy_config = (
                        general.get("customConfig")
                        if general.get("mode") == plate_generator.PLATE_MODE_CUSTOM
                        else general.get("config")
                    )
                binding["customConfig"] = plate_generator.normalized_plate_config(copy_config)
            binding["config"] = plate_generator.normalized_plate_config(binding.get("customConfig"))
        binding["mode"] = mode
        binding["setId"] = set_id
        binding["sourceConfig"] = source_config
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            binding["customDefined"] = True
        elif mode == plate_generator.PLATE_MODE_TRIM:
            source_settings = self.conversion.get("variants", {}).get(source_config, {})
            source_binding = plate_generator.normalized_plate_binding(
                source_settings.get("plate") if isinstance(source_settings, dict) else None,
                variant=True,
            )
            binding["config"] = plate_generator.normalized_plate_config(source_binding.get("customConfig"))
        if mode == plate_generator.PLATE_MODE_SET:
            record = plate_generator.plate_set_by_id(set_id)
            if record is not None:
                binding["config"] = plate_generator.normalized_plate_config(record.get("config"))
        settings["plate"] = binding
        self._refresh_variants()
        self._update_detail()
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            self._open_plate_editor(config_name)

    def _open_plate_library(self) -> None:
        if self.plate_library_modal is not None and self.plate_library_modal.winfo_exists():
            self.plate_library_modal.lift()
            return
        self.plate_library_modal = PlateLibraryDialog(self)

    def _open_plate_editor(self, variant_name: str | None, *, set_id: str | None = None) -> None:
        if self.context is None and set_id is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        if self.plate_editor_modal is not None and self.plate_editor_modal.winfo_exists():
            self.plate_editor_modal.lift()
            return
        if set_id is None:
            if variant_name is None:
                binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
            else:
                settings = self.conversion.get("variants", {}).get(variant_name, {})
                binding = plate_generator.normalized_plate_binding(
                    settings.get("plate") if isinstance(settings, dict) else None,
                    variant=True,
                )
                if binding.get("mode") == plate_generator.PLATE_MODE_TRIM:
                    source_config = str(binding.get("sourceConfig") or "")
                    source_settings = self.conversion.get("variants", {}).get(source_config, {})
                    if not isinstance(source_settings, dict):
                        self._show_error(
                            "Missing custom plate settings",
                            f"Custom plate source '{source_config}' is no longer available. Choose a new plate option for this trim.",
                        )
                        return
                    variant_name = source_config
                    binding = plate_generator.normalized_plate_binding(source_settings.get("plate"), variant=True)
                elif binding.get("mode") == plate_generator.PLATE_MODE_GENERAL:
                    binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
                    variant_name = None
                    if binding.get("mode") == plate_generator.PLATE_MODE_OFF:
                        self._show_error(
                            "Plates are off",
                            "Choose a custom or library plate option before configuring it.",
                        )
                        return
            if binding.get("mode") == plate_generator.PLATE_MODE_SET:
                set_id = str(binding.get("setId") or "")
        if set_id is not None and plate_generator.plate_set_by_id(set_id) is None:
            self._show_error(
                "Missing plate set",
                f"Plate set '{set_id}' was deleted. The build can still use its saved snapshot; choose Custom to edit that snapshot.",
            )
            return
        self.plate_editor_modal = PlateEditorDialog(self, variant_name if set_id is None else None, set_id=set_id)

    def _plate_settings_applied(self) -> None:
        self._sync_plate_to_ui()
        self._refresh_variants()
        if self.plate_library_modal is not None and self.plate_library_modal.winfo_exists():
            self.plate_library_modal.refresh()
        self._update_detail()
        self.status_var.set(f"Licence plate settings updated ({plate_generator.plate_summary_label(self.conversion)})")
