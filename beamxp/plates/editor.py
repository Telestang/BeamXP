"""Licence plate editor dialog for the BeamXP tool.

Edits either the conversion's general plate settings or one trim's override
(general/custom/off), with a live preview rendered by plate_generator. All
plate asset generation lives in plate_generator; this module is UI only.
"""
from __future__ import annotations

import subprocess
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

from beamxp.plates import generator as plate_generator

if TYPE_CHECKING:
    from beamxp.hand_drive_tool import HandDriveToolApp

PLATE_MODE_LABELS = {
    "general": "Use vehicle plate setting",
    "custom": "Custom for this trim",
    "off": "Off",
}
PLATE_FONT_DEFAULT_LABEL = "Default (system DIN)"
PLATE_FONT_FOLDER_PREFIX = "Fonts folder: "
PLATE_FONT_CUSTOM_LABEL = "Custom font file..."
PLATE_FONT_LINKS = (
    {
        "name": "Mandatory",
        "region": "UK modern",
        "note": "Charles Wright-style plate lettering",
        "url": "https://www.dafont.com/mandatory.font",
    },
    {
        "name": "UK Number Plate",
        "region": "UK older/common",
        "note": "Simple UK plate approximation",
        "url": "https://www.dafont.com/uk-number-plate.font",
    },
    {
        "name": "Registration Plate UK",
        "region": "UK modern",
        "note": "Modern UK plate approximation",
        "url": "https://www.dafont.com/font-comment.php?file=registration_plate_uk",
    },
    {
        "name": "Din 1451",
        "region": "Germany/EU older",
        "note": "DIN-style German and European fallback",
        "url": "https://www.dafont.com/din-1451.font",
    },
    {
        "name": "Car-Go 2",
        "region": "Germany/EU",
        "note": "FE-Schrift-style lettering",
        "url": "https://www.dafont.com/cargo2.font",
    },
    {
        "name": "Kenteken",
        "region": "Netherlands",
        "note": "Dutch plate-style lettering",
        "url": "https://www.dafont.com/kenteken.font",
    },
    {
        "name": "Kenteken Smits",
        "region": "Netherlands classic",
        "note": "Older Dutch plate-style variants",
        "url": "https://www.dafont.com/kenteken-smits.font",
    },
    {
        "name": "License Plate USA",
        "region": "USA",
        "note": "Embossed US plate style",
        "url": "https://www.dafont.com/license-plate-usa.font",
    },
    {
        "name": "Licenz Plate",
        "region": "USA",
        "note": "Alternative North American plate style",
        "url": "https://www.dafont.com/licenz-plate.font",
    },
    {
        "name": "JDMGT-R34",
        "region": "Japan",
        "note": "English FontSpace page, JP/JDM plate recreation",
        "url": "https://www.fontspace.com/jdmgt-r34-font-f20080",
    },
    {
        "name": "TRM/FZ JP Plate Fonts",
        "region": "Japan",
        "note": "English GitHub resource folder with trm.ttf and fz.otf",
        "url": "https://github.com/ItsNickkk/license-plate-maker/tree/master/resources/font",
    },
)
PLATE_JP_STYLE_LABELS = {
    "private": "Private (white)",
    "kei": "Kei (yellow)",
    "commercial": "Commercial (green)",
    "kei commercial": "Kei commercial (black)",
}
PLATE_BAND_LABELS = {
    plate_generator.BAND_NONE: "None",
    plate_generator.BAND_EU: "EU",
    plate_generator.BAND_CUSTOM: "Custom",
}
PLATE_PREVIEW_WIDTH = 470
PLATE_PREVIEW_SIDE_FRONT = "Front"
PLATE_PREVIEW_SIDE_REAR = "Rear"


def _key_for_label(labels: dict[str, str], selected: str, default: str) -> str:
    """Reverse-map a display label from one of the *_LABELS tables to its key."""
    for key, label in labels.items():
        if label == selected:
            return key
    return default


class PlateEditorDialog(tk.Toplevel):
    """Editor for the general plate settings or a single trim's override."""

    def __init__(
        self,
        app: HandDriveToolApp,
        variant_name: str | None = None,
        *,
        set_id: str | None = None,
    ) -> None:
        super().__init__(app)
        self.app = app
        self.variant_name = variant_name
        self.set_id = set_id
        self._preview_photo = None
        self._preview_job: str | None = None
        self._registration: str | None = None
        self.vehicle_id = app.context.vehicle_id if app.context is not None else "vehicle"
        self.plate_mode_labels = {
            "general": app._vehicle_plate_label(),
            "custom": f"Custom ({variant_name})" if variant_name else f"Custom ({self.vehicle_id})",
            "off": "Off",
        }

        conversion = app.conversion
        general_binding = plate_generator.normalized_plate_binding(conversion.get("plate"))
        general_cfg = plate_generator.normalized_plate_config(general_binding.get("customConfig"))
        if set_id is not None:
            record = plate_generator.plate_set_by_id(set_id)
            if record is None:
                raise RuntimeError(f"Plate set '{set_id}' no longer exists")
            self.cfg = plate_generator.normalized_plate_config(record.get("config"))
            self.mode = "set"
            self.title(f"Licence Plates - {record['name']}")
        elif variant_name is None:
            self.mode = "general"
            self.cfg = general_cfg
            self.title("Licence Plates")
        else:
            settings = conversion.setdefault("variants", {}).setdefault(variant_name, {})
            self.mode = plate_generator.variant_plate_mode(settings)
            override = plate_generator.normalized_plate_binding(
                settings.get("plate") if isinstance(settings, dict) else None,
                variant=True,
            )
            stored = override.get("customConfig")
            # Start a fresh override from the general settings so "Custom"
            # begins as a copy the user can tweak.
            source_cfg = stored if self.mode == plate_generator.PLATE_MODE_CUSTOM and isinstance(stored, dict) else general_cfg
            self.cfg = plate_generator.normalized_plate_config(source_cfg)
            self.title(f"Licence Plates - {variant_name}")

        self.transient(app)
        self.minsize(760, 540)
        self.columnconfigure(0, weight=1)
        self._build_ui()
        self._fit_to_contents()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        app._place_modal_on_app_monitor(self)
        self.grab_set()
        self.focus_set()
        self._sync_control_states()
        self._schedule_preview()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        body = ttk.Frame(self, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)

        header = ttk.Frame(body)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        if self.set_id is not None:
            record = plate_generator.plate_set_by_id(self.set_id)
            name = str(record.get("name")) if record else self.set_id
            ttk.Label(
                header,
                text=f"Editing set '{name}' - changes apply to every conversion that references it",
            ).pack(side="left")
        elif self.variant_name is None:
            ttk.Label(header, text=f"Custom plate settings for {self.vehicle_id}").pack(side="left")
        else:
            ttk.Label(header, text="Plate settings for this trim").pack(side="left")
            self.variant_mode_var = tk.StringVar(
                value=self.plate_mode_labels.get(self.mode, self.plate_mode_labels["general"])
            )
            mode_combo = ttk.Combobox(
                header,
                textvariable=self.variant_mode_var,
                values=list(self.plate_mode_labels.values()),
                state="readonly",
                width=24,
            )
            mode_combo.pack(side="left", padx=(8, 0))
            mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_control_states())

        self._picker_vars: dict[tuple[str, str], list[tk.StringVar]] = {}
        self.controls_frame = ttk.Frame(body)
        self.controls_frame.grid(row=1, column=0, sticky="ew")
        self.controls_frame.columnconfigure(1, weight=1)

        ttk.Label(self.controls_frame, text="Plate type").grid(row=0, column=0, sticky="w")
        self.size_var = tk.StringVar(value=str(self.cfg.get("size")))
        size_combo = ttk.Combobox(
            self.controls_frame,
            textvariable=self.size_var,
            values=list(plate_generator.PLATE_SIZES),
            state="readonly",
            width=8,
        )
        size_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
        size_combo.bind("<<ComboboxSelected>>", lambda _event: self._size_changed())

        ttk.Label(self.controls_frame, text="Font").grid(row=1, column=0, sticky="w", pady=(6, 0))
        font_row = ttk.Frame(self.controls_frame)
        font_row.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        font_row.columnconfigure(1, weight=1)
        font_cfg = self.cfg.get("font", {})
        self.font_library_paths: dict[str, Path] = {}
        self._refresh_font_library_paths()
        font_values = self._font_combo_values(font_cfg)
        self.font_source_var = tk.StringVar(value=self._font_source_label(font_cfg))
        self.font_combo = ttk.Combobox(
            font_row,
            textvariable=self.font_source_var,
            values=font_values,
            state="readonly",
            width=28,
        )
        self.font_combo.grid(row=0, column=0, sticky="w")
        self.font_combo.bind("<<ComboboxSelected>>", lambda _event: self._font_source_changed())
        self.font_path_var = tk.StringVar(value=str(font_cfg.get("path") or "") if isinstance(font_cfg, dict) else "")
        self.font_path_label = ttk.Label(font_row, textvariable=self.font_path_var, anchor="w")
        self.font_path_label.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.font_browse_button = ttk.Button(font_row, text="Browse", command=self._browse_font)
        self.font_browse_button.grid(row=0, column=2, sticky="e", padx=(6, 0))
        self.font_folder_button = ttk.Button(font_row, text="Folder", command=self._open_font_folder)
        self.font_folder_button.grid(row=0, column=3, sticky="e", padx=(6, 0))
        self.font_refresh_button = ttk.Button(font_row, text="Refresh", command=self._refresh_font_combo)
        self.font_refresh_button.grid(row=0, column=4, sticky="e", padx=(6, 0))
        self.font_links_button = ttk.Button(font_row, text="Links...", command=self._open_font_links)
        self.font_links_button.grid(row=0, column=5, sticky="e", padx=(6, 0))

        ttk.Label(self.controls_frame, text="Registration pattern").grid(row=2, column=0, sticky="w", pady=(6, 0))
        pattern_row = ttk.Frame(self.controls_frame)
        pattern_row.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        pattern_row.columnconfigure(0, weight=1)
        self.pattern_var = tk.StringVar(value=plate_generator.active_pattern(self.cfg))
        pattern_entry = ttk.Entry(pattern_row, textvariable=self.pattern_var)
        pattern_entry.grid(row=0, column=0, sticky="ew")
        self.pattern_var.trace_add("write", lambda *_args: self._pattern_changed())
        ttk.Label(
            pattern_row,
            text="@ = letter, # = digit, ~ = letter or digit, . = centre dot",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(self.controls_frame, text="Emboss strength").grid(row=3, column=0, sticky="w", pady=(6, 0))
        emboss_row = ttk.Frame(self.controls_frame)
        emboss_row.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        emboss_row.columnconfigure(0, weight=1)
        self.emboss_value_var = tk.StringVar()
        initial_emboss = plate_generator.emboss_strength(self.cfg)
        self.emboss_var = tk.DoubleVar(value=initial_emboss)

        def emboss_changed(value: object) -> None:
            try:
                strength = max(0.0, min(plate_generator.EMBOSS_MAX_UI, float(value)))
            except (TypeError, ValueError):
                strength = 1.0
            self.cfg["embossStrength"] = round(strength, 2)
            self.emboss_value_var.set(f"{strength:.1f}x")
            self._schedule_preview()

        ttk.Scale(
            emboss_row,
            from_=0.0,
            to=plate_generator.EMBOSS_MAX_UI,
            variable=self.emboss_var,
            command=emboss_changed,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(emboss_row, textvariable=self.emboss_value_var, width=5).grid(row=0, column=1, sticky="e", padx=(8, 0))
        emboss_changed(initial_emboss)

        ttk.Label(self.controls_frame, text="Border").grid(row=4, column=0, sticky="w", pady=(6, 0))
        border_row = ttk.Frame(self.controls_frame)
        border_row.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        border_row.columnconfigure(9, weight=1)
        self.border_enabled_var = tk.BooleanVar(value=bool(self.cfg["border"].get("enabled")))
        ttk.Checkbutton(
            border_row,
            text="Enabled",
            variable=self.border_enabled_var,
            command=self._border_enabled_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(border_row, text="Colour").grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._color_button(border_row, "border", "color").grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Label(border_row, text="Offset").grid(row=0, column=3, sticky="w", padx=(10, 0))
        self._spin(border_row, "border", "offset", 0, 80, 1, integer=True).grid(row=0, column=4, sticky="w", padx=(4, 0))
        ttk.Label(border_row, text="Thickness").grid(row=0, column=5, sticky="w", padx=(10, 0))
        self._spin(border_row, "border", "thickness", 1, 40, 1, integer=True).grid(row=0, column=6, sticky="w", padx=(4, 0))
        ttk.Label(border_row, text="Corner radius").grid(row=0, column=7, sticky="w", padx=(10, 0))
        self._spin(border_row, "border", "cornerRadius", 0, 120, 1, integer=True).grid(row=0, column=8, sticky="w", padx=(4, 0))

        self.size_frames: dict[str, ttk.Frame] = {}
        holder = ttk.LabelFrame(body, text="Plate configuration", padding=8)
        holder.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        holder.columnconfigure(0, weight=1)
        self.size_holder = holder
        self._build_eu_frame(holder)
        self._build_us_frame(holder)
        self._build_jp_frame(holder)

        preview_frame = ttk.LabelFrame(body, text="Preview", padding=8)
        preview_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_header = ttk.Frame(preview_frame)
        preview_header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        preview_header.columnconfigure(4, weight=1)
        self.preview_side_var = tk.StringVar(value=self._initial_preview_side())
        ttk.Label(preview_header, text="Side").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            preview_header,
            text=PLATE_PREVIEW_SIDE_FRONT,
            value=PLATE_PREVIEW_SIDE_FRONT,
            variable=self.preview_side_var,
            command=lambda: self._schedule_preview(immediate=True),
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(
            preview_header,
            text=PLATE_PREVIEW_SIDE_REAR,
            value=PLATE_PREVIEW_SIDE_REAR,
            variable=self.preview_side_var,
            command=lambda: self._schedule_preview(immediate=True),
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.preview_format_var = tk.StringVar(value="")
        ttk.Label(preview_header, textvariable=self.preview_format_var).grid(row=0, column=3, sticky="w", padx=(14, 0))
        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.registration_var = tk.StringVar(value="")
        ttk.Label(preview_frame, text="Generated registration:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(preview_frame, textvariable=self.registration_var).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Button(preview_frame, text="Regenerate", command=self._regenerate_registration).grid(
            row=2, column=2, sticky="e", pady=(6, 0)
        )
        self.error_var = tk.StringVar(value="")
        error_label = ttk.Label(preview_frame, textvariable=self.error_var, foreground="#c04a3a", wraplength=520, justify="left")
        error_label.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        buttons.columnconfigure(0, weight=1)
        ttk.Button(buttons, text="Apply", command=self._apply).grid(row=0, column=1, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self._close).grid(row=0, column=2, sticky="e", padx=(6, 0))

        self._show_size_frame(str(self.cfg.get("size")))

    def _color_button(self, parent: ttk.Frame, section: str, key: str) -> tk.Button:
        def pick() -> None:
            from tkinter import colorchooser

            current = str(self.cfg[section].get(key) or "#ffffff")
            _rgb, hex_value = colorchooser.askcolor(color=current, parent=self)
            if hex_value:
                self.cfg[section][key] = hex_value
                button.configure(background=hex_value)
                self._schedule_preview()

        button = tk.Button(
            parent,
            width=4,
            relief="ridge",
            background=str(self.cfg[section].get(key) or "#ffffff"),
            command=pick,
        )
        return button

    def _spin(
        self,
        parent: ttk.Frame,
        section: str,
        key: str,
        low: float,
        high: float,
        step: float,
        *,
        integer: bool = False,
    ) -> ttk.Spinbox:
        var = tk.StringVar(value=str(self.cfg[section].get(key)))

        def commit(*_args: object) -> None:
            try:
                value = int(float(var.get())) if integer else float(var.get())
            except (TypeError, ValueError):
                return
            value = max(low, min(high, value))
            self.cfg[section][key] = value
            self._schedule_preview()

        var.trace_add("write", commit)
        spin = ttk.Spinbox(parent, textvariable=var, from_=low, to=high, increment=step, width=7)
        return spin

    def _config_text_var(self, section: str, key: str) -> tk.StringVar:
        """A StringVar that writes back to cfg[section][key] (stripped) and
        refreshes the preview on every edit."""
        var = tk.StringVar(value=str(self.cfg[section].get(key) or ""))

        def commit(*_args: object) -> None:
            self.cfg[section][key] = var.get().strip()
            self._schedule_preview()

        var.trace_add("write", commit)
        return var

    def _image_picker(self, parent: ttk.Frame, section: str, key: str, title: str) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.columnconfigure(0, weight=1)
        var = tk.StringVar(value=str(self.cfg[section].get(key) or ""))
        # The background pickers appear in every family frame; remember the
        # vars so switching frames can resync them from the shared config.
        self._picker_vars.setdefault((section, key), []).append(var)

        def browse() -> None:
            path = filedialog.askopenfilename(
                title=title,
                parent=self,
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")],
            )
            if path:
                var.set(path)
                self.cfg[section][key] = path
                self._schedule_preview()

        def clear() -> None:
            var.set("")
            self.cfg[section][key] = ""
            self._schedule_preview()

        ttk.Label(row, textvariable=var, anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="Browse", command=browse).grid(row=0, column=1, sticky="e", padx=(6, 0))
        ttk.Button(row, text="Clear", command=clear).grid(row=0, column=2, sticky="e", padx=(6, 0))
        return row

    def _build_eu_frame(self, holder: ttk.LabelFrame) -> None:
        frame = ttk.Frame(holder)
        frame.columnconfigure(2, weight=1)
        self.size_frames[plate_generator.PLATE_SIZE_EU] = frame

        ttk.Label(frame, text="Front background").grid(row=0, column=0, sticky="w")
        self._color_button(frame, "eu", "frontColor").grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._image_picker(frame, "background", "frontImage", "Choose front plate background image").grid(
            row=0, column=2, sticky="ew", padx=(10, 0)
        )
        ttk.Label(frame, text="Rear background").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._color_button(frame, "eu", "rearColor").grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        self._image_picker(frame, "background", "rearImage", "Choose rear plate background image").grid(
            row=1, column=2, sticky="ew", padx=(10, 0), pady=(4, 0)
        )
        ttk.Label(frame, text="Font colour").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._color_button(frame, "eu", "textColor").grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(frame, text="Character spacing").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self._spin(frame, "eu", "spacing", -20, 60, 1, integer=True).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(frame, text="Horizontal text offset").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self._spin(frame, "eu", "textX", -0.4, 0.4, 0.01).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        ttk.Label(frame, text="Side band").grid(row=6, column=0, sticky="w", pady=(8, 0))
        self.band_var = tk.StringVar(value=PLATE_BAND_LABELS.get(str(self.cfg["eu"].get("sideBand")), "None"))
        band_combo = ttk.Combobox(
            frame,
            textvariable=self.band_var,
            values=list(PLATE_BAND_LABELS.values()),
            state="readonly",
            width=10,
        )
        band_combo.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        band_combo.bind("<<ComboboxSelected>>", lambda _event: self._band_changed())

        # The band-specific controls live in a rebuildable child frame. Match
        # its label column to the parent grid so Country code/Band image line
        # up with Front background, Side band, etc. Give spare width to the
        # trailing column, not the entry column, so the example text stays
        # beside the country-code entry.
        frame.update_idletasks()
        label_column_width = max(
            (widget.winfo_reqwidth() for widget in frame.grid_slaves(column=0)),
            default=0,
        )
        self.band_detail = ttk.Frame(frame)
        self.band_detail.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.band_detail.columnconfigure(0, minsize=label_column_width)
        self.band_detail.columnconfigure(2, weight=1)
        self._rebuild_band_detail()

    def _rebuild_band_detail(self) -> None:
        for child in self.band_detail.winfo_children():
            child.destroy()
        band = str(self.cfg["eu"].get("sideBand"))
        if band == plate_generator.BAND_EU:
            ttk.Label(self.band_detail, text="Country code").grid(row=0, column=0, sticky="w")
            code_var = self._config_text_var("eu", "bandCode")
            ttk.Entry(self.band_detail, textvariable=code_var, width=6).grid(row=0, column=1, sticky="w", padx=(8, 0))
            ttk.Label(self.band_detail, text="e.g. D, F, NL").grid(row=0, column=2, sticky="w", padx=(8, 0))
            ttk.Label(self.band_detail, text="Band image").grid(row=1, column=0, sticky="w", pady=(4, 0))
            self._image_picker(self.band_detail, "eu", "bandFullImage", "Choose full side band image").grid(
                row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(4, 0)
            )
        elif band == plate_generator.BAND_CUSTOM:
            ttk.Label(self.band_detail, text="Band colour").grid(row=0, column=0, sticky="w")
            self._color_button(self.band_detail, "eu", "bandColor").grid(row=0, column=1, sticky="w", padx=(8, 0))
            ttk.Label(self.band_detail, text="Band image").grid(row=1, column=0, sticky="w", pady=(4, 0))
            self._image_picker(self.band_detail, "eu", "bandFullImage", "Choose full side band image").grid(
                row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(4, 0)
            )
            ttk.Label(self.band_detail, text="Code text").grid(row=2, column=0, sticky="w", pady=(4, 0))
            code_var = self._config_text_var("eu", "bandCode")
            ttk.Entry(self.band_detail, textvariable=code_var, width=6).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
            ttk.Label(self.band_detail, text="Emblem image").grid(row=3, column=0, sticky="w", pady=(4, 0))
            self._image_picker(self.band_detail, "eu", "bandImage", "Choose side band emblem image").grid(
                row=3, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(4, 0)
            )
        self._fit_to_contents()

    def _build_us_frame(self, holder: ttk.LabelFrame) -> None:
        frame = ttk.Frame(holder)
        frame.columnconfigure(2, weight=1)
        self.size_frames[plate_generator.PLATE_SIZE_US] = frame

        ttk.Label(frame, text="Front background").grid(row=0, column=0, sticky="w")
        self._color_button(frame, "us", "bgColor").grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._image_picker(frame, "background", "frontImage", "Choose front plate background image").grid(
            row=0, column=2, sticky="ew", padx=(10, 0)
        )
        ttk.Label(frame, text="Rear background").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._image_picker(frame, "background", "rearImage", "Choose rear plate background image").grid(
            row=1, column=2, sticky="ew", padx=(10, 0), pady=(4, 0)
        )
        ttk.Label(frame, text="Font colour").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._color_button(frame, "us", "textColor").grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(frame, text="Text scale").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self._spin(frame, "us", "textScale", 0.3, 2.5, 0.05).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(frame, text="Horizontal text offset").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self._spin(frame, "us", "textX", -0.4, 0.4, 0.01).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(frame, text="Vertical text offset").grid(row=6, column=0, sticky="w", pady=(4, 0))
        self._spin(frame, "us", "textY", -0.4, 0.4, 0.01).grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(frame, text="Character spacing").grid(row=7, column=0, sticky="w", pady=(4, 0))
        self._spin(frame, "us", "spacing", -20, 60, 1, integer=True).grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

    def _build_jp_frame(self, holder: ttk.LabelFrame) -> None:
        frame = ttk.Frame(holder)
        frame.columnconfigure(1, weight=1)
        self.size_frames[plate_generator.PLATE_SIZE_JP] = frame

        ttk.Label(frame, text="Front bg image").grid(row=0, column=0, sticky="w")
        self._image_picker(frame, "background", "frontImage", "Choose front plate background image").grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Label(frame, text="Rear bg image").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._image_picker(frame, "background", "rearImage", "Choose rear plate background image").grid(
            row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0)
        )
        ttk.Label(frame, text="Plate style").grid(row=3, column=0, sticky="w", pady=(6, 0))
        style_key = str(self.cfg["jp"].get("style") or "private")
        self.jp_style_var = tk.StringVar(value=PLATE_JP_STYLE_LABELS.get(style_key, PLATE_JP_STYLE_LABELS["private"]))
        style_combo = ttk.Combobox(
            frame,
            textvariable=self.jp_style_var,
            values=list(PLATE_JP_STYLE_LABELS.values()),
            state="readonly",
            width=22,
        )
        style_combo.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        def style_changed(_event: object = None) -> None:
            self.cfg["jp"]["style"] = _key_for_label(
                PLATE_JP_STYLE_LABELS, self.jp_style_var.get(), str(self.cfg["jp"].get("style") or "private")
            )
            self._schedule_preview()

        style_combo.bind("<<ComboboxSelected>>", style_changed)

        def text_field(row: int, label: str, key: str, values: tuple[str, ...] | None, width: int) -> None:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
            var = self._config_text_var("jp", key)
            if values:
                widget = ttk.Combobox(frame, textvariable=var, values=list(values), width=width)
            else:
                widget = ttk.Entry(frame, textvariable=var, width=width)
            widget.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        text_field(4, "Region", "region", plate_generator.JP_REGION_CHOICES, 12)
        text_field(5, "Classification", "classification", None, 6)
        text_field(6, "Kana", "kana", plate_generator.JP_KANA_CHOICES, 4)
        ttk.Label(
            frame,
            text="The registration pattern fills the main number (e.g. ##-##).",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # -- behaviour ----------------------------------------------------------

    def _fit_to_contents(self, *, center: bool = False) -> None:
        try:
            self.update_idletasks()
            work_left, work_top, work_right, work_bottom = self.app._current_monitor_work_area()
            max_width = max(760, work_right - work_left - 32)
            max_height = max(540, work_bottom - work_top - 32)
            width = min(max(760, self.winfo_reqwidth()), max_width)
            height = min(max(540, self.winfo_reqheight()), max_height)
            current_width = max(self.winfo_width(), 1)
            current_height = max(self.winfo_height(), 1)
            changed = abs(current_width - width) > 2 or abs(current_height - height) > 2
            if changed:
                self.geometry(f"{width}x{height}")
            if center and changed:
                self.app._place_modal_on_app_monitor(self)
        except tk.TclError:
            pass

    def _initial_preview_side(self) -> str:
        try:
            if str(self.cfg.get("size")) == plate_generator.PLATE_SIZE_EU:
                eu = self.cfg.get("eu", {})
                if isinstance(eu, dict) and str(eu.get("frontColor")).lower() != str(eu.get("rearColor")).lower():
                    return PLATE_PREVIEW_SIDE_REAR
        except Exception:
            pass
        return PLATE_PREVIEW_SIDE_FRONT

    def _preview_config_name(self) -> str | None:
        context = self.app.context
        if context is None:
            return None
        if self.variant_name is not None and self.variant_name in context.variants:
            return self.variant_name
        try:
            current = self.app._mesh_scene_config()
            if current in context.variants:
                return current
        except Exception:
            pass
        try:
            for name in self.app._selected_variant_names():
                if name in context.variants:
                    return name
        except Exception:
            pass
        return next(iter(sorted(context.variants)), None)

    def _preview_format(self) -> str | None:
        side = self.preview_side_var.get() if hasattr(self, "preview_side_var") else PLATE_PREVIEW_SIDE_FRONT
        config_name = self._preview_config_name()
        choice = plate_generator.PLATE_PART_AUTO
        if config_name:
            variants = self.app.conversion.get("variants", {})
            settings = variants.get(config_name) if isinstance(variants, dict) else None
            if isinstance(settings, dict):
                key = "rearPlate" if side == PLATE_PREVIEW_SIDE_REAR else "frontPlate"
                choice = settings.get(key, plate_generator.PLATE_PART_AUTO)
        fmt = plate_generator.preview_format_for_config(
            self.app.context,
            config_name or "",
            side.lower(),
            choice,
        )
        physical_label = ""
        if config_name and self.app.context is not None:
            physical_label = plate_generator.plate_part_label_for_config(
                self.app.context,
                config_name,
                side.lower(),
                choice,
            )
        label = f"Format: {fmt}" if fmt else "Format: no plate"
        if physical_label:
            label += f" — {physical_label}"
        if config_name:
            label += f" ({config_name})"
        self.preview_format_var.set(label)
        return fmt

    def _show_size_frame(self, size: str) -> None:
        for frame in self.size_frames.values():
            frame.grid_remove()
        frame = self.size_frames.get(size)
        if frame is not None:
            frame.grid(row=0, column=0, sticky="ew")
        self._fit_to_contents()

    def _size_changed(self) -> None:
        size = self.size_var.get()
        if size not in plate_generator.PLATE_SIZES:
            return
        self.cfg["size"] = size
        self.pattern_var.set(plate_generator.active_pattern(self.cfg))
        # Background pickers exist in every family frame over one shared
        # config section; refresh the incoming frame's copies.
        for (section, key), pickers in self._picker_vars.items():
            value = str(self.cfg.get(section, {}).get(key) or "")
            for var in pickers:
                if var.get() != value:
                    var.set(value)
        self._show_size_frame(size)
        self._registration = None
        self._schedule_preview()

    def _band_changed(self) -> None:
        self.cfg["eu"]["sideBand"] = _key_for_label(
            PLATE_BAND_LABELS, self.band_var.get(), str(self.cfg["eu"].get("sideBand") or plate_generator.BAND_NONE)
        )
        self._rebuild_band_detail()
        self._schedule_preview()

    def _pattern_changed(self) -> None:
        plate_generator.active_section(self.cfg)["pattern"] = self.pattern_var.get()
        self._registration = None
        self._schedule_preview()

    def _border_enabled_changed(self) -> None:
        self.cfg["border"]["enabled"] = bool(self.border_enabled_var.get())
        self._schedule_preview()

    def _refresh_font_library_paths(self) -> None:
        self.font_library_paths = {
            f"{PLATE_FONT_FOLDER_PREFIX}{path.name}": path
            for path in plate_generator.user_font_files()
        }

    def _font_source_label(self, font_cfg: object) -> str:
        if not isinstance(font_cfg, dict):
            return PLATE_FONT_DEFAULT_LABEL
        source = str(font_cfg.get("source") or "default")
        if source == "library":
            name = str(font_cfg.get("name") or "")
            if not name:
                path_text = str(font_cfg.get("path") or "")
                name = Path(path_text).name if path_text else ""
            return f"{PLATE_FONT_FOLDER_PREFIX}{name}" if name else PLATE_FONT_DEFAULT_LABEL
        if source == "custom":
            return PLATE_FONT_CUSTOM_LABEL
        return PLATE_FONT_DEFAULT_LABEL

    def _font_combo_values(self, font_cfg: object | None = None) -> list[str]:
        values = [PLATE_FONT_DEFAULT_LABEL, *self.font_library_paths.keys(), PLATE_FONT_CUSTOM_LABEL]
        if font_cfg is not None:
            selected = self._font_source_label(font_cfg)
            if selected.startswith(PLATE_FONT_FOLDER_PREFIX) and selected not in values:
                values.insert(-1, selected)
        return values

    def _refresh_font_combo(self) -> None:
        self._refresh_font_library_paths()
        self.font_combo.configure(values=self._font_combo_values(self.cfg.get("font")))
        self.font_source_var.set(self._font_source_label(self.cfg.get("font")))

    def _font_source_changed(self) -> None:
        selected = self.font_source_var.get()
        if selected == PLATE_FONT_DEFAULT_LABEL:
            self.cfg["font"] = {"source": "default", "path": ""}
            self.font_path_var.set("")
        elif selected == PLATE_FONT_CUSTOM_LABEL:
            current = self.cfg.get("font", {})
            current_source = str(current.get("source") or "") if isinstance(current, dict) else ""
            current_path = str(current.get("path") or "") if isinstance(current, dict) else ""
            if current_source != "custom" or not current_path:
                self.cfg["font"] = {"source": "custom", "path": ""}
                self.font_path_var.set("")
                self._browse_font()
                return
            self.cfg["font"] = {"source": "custom", "path": current_path}
            self.font_path_var.set(current_path)
        elif selected in self.font_library_paths:
            path = self.font_library_paths[selected]
            self.cfg["font"] = {"source": "library", "name": path.name, "path": str(path)}
            self.font_path_var.set(str(path))
        elif selected.startswith(PLATE_FONT_FOLDER_PREFIX):
            name = selected[len(PLATE_FONT_FOLDER_PREFIX):].strip()
            path = plate_generator.ensure_user_fonts_dir() / name
            self.cfg["font"] = {"source": "library", "name": name, "path": str(path)}
            self.font_path_var.set(str(path))
        else:
            self.cfg["font"] = {"source": "default", "path": ""}
            self.font_path_var.set("")
            self.font_source_var.set(PLATE_FONT_DEFAULT_LABEL)
        self._sync_control_states()
        self._schedule_preview()

    def _browse_font(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a TTF/OTF font",
            parent=self,
            filetypes=[("Fonts", "*.ttf *.otf *.ttc"), ("All files", "*.*")],
        )
        if not path:
            return
        self.cfg["font"] = {"source": "custom", "path": path}
        self.font_source_var.set(PLATE_FONT_CUSTOM_LABEL)
        self.font_path_var.set(path)
        self._sync_control_states()
        self._schedule_preview()

    def _open_font_folder(self) -> None:
        folder = plate_generator.ensure_user_fonts_dir()
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(folder)])
            else:
                webbrowser.open(folder.as_uri())
        except Exception as exc:
            messagebox.showerror("Fonts folder", f"Could not open fonts folder:\n{exc}", parent=self)
        self._refresh_font_combo()

    def _open_font_links(self) -> None:
        modal = tk.Toplevel(self)
        modal.title("External Plate Fonts")
        modal.transient(self)
        modal.geometry("840x460")
        modal.minsize(680, 360)
        modal.columnconfigure(0, weight=1)
        modal.rowconfigure(1, weight=1)

        fonts_dir = plate_generator.ensure_user_fonts_dir()
        ttk.Label(
            modal,
            text=f"Download/install from the source page, then drop .ttf/.otf/.ttc files into: {fonts_dir}",
            wraplength=680,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))

        columns = ("region", "note")
        tree_frame = ttk.Frame(modal)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", height=14)
        tree.heading("#0", text="Font")
        tree.heading("region", text="Region/style")
        tree.heading("note", text="Use")
        tree.column("#0", width=200, stretch=False)
        tree.column("region", width=170, stretch=False)
        tree.column("note", width=410, stretch=True)
        yscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

        urls: dict[str, str] = {}
        for index, item in enumerate(PLATE_FONT_LINKS):
            iid = str(index)
            urls[iid] = str(item["url"])
            tree.insert("", "end", iid=iid, text=str(item["name"]), values=(str(item["region"]), str(item["note"])))
        if urls:
            tree.selection_set("0")

        buttons = ttk.Frame(modal)
        buttons.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        buttons.columnconfigure(0, weight=1)

        def open_selected() -> None:
            selection = tree.selection()
            if selection:
                webbrowser.open(urls.get(selection[0], ""))

        def refresh_and_close() -> None:
            self._refresh_font_combo()
            modal.destroy()

        tree.bind("<Double-1>", lambda _event: open_selected())
        ttk.Button(buttons, text="Open fonts folder", command=self._open_font_folder).grid(row=0, column=0, sticky="w")
        ttk.Button(buttons, text="Open page", command=open_selected).grid(row=0, column=1, sticky="e")
        ttk.Button(buttons, text="Refresh", command=self._refresh_font_combo).grid(row=0, column=2, sticky="e", padx=(6, 0))
        ttk.Button(buttons, text="Close", command=refresh_and_close).grid(row=0, column=3, sticky="e", padx=(6, 0))
        self.app._place_modal_on_app_monitor(modal)

    def _editing_enabled(self) -> bool:
        if self.variant_name is None or self.set_id is not None:
            return True
        return self.variant_mode_var.get() == self.plate_mode_labels["custom"]

    def _sync_control_states(self) -> None:
        enabled = self._editing_enabled()
        state = "normal" if enabled else "disabled"

        def walk(widget: tk.Widget) -> None:
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                    walk(child)
                    continue
                try:
                    if isinstance(child, ttk.Combobox):
                        child.configure(state="readonly" if enabled else "disabled")
                    else:
                        child.configure(state=state)
                except tk.TclError:
                    pass

        walk(self.controls_frame)
        walk(self.size_holder)
        if enabled:
            self.font_browse_button.configure(
                state="normal" if self.font_source_var.get() == PLATE_FONT_CUSTOM_LABEL else "disabled"
            )
        # Editable combos (region/kana) should stay typeable, not readonly.
        jp_frame = self.size_frames.get(plate_generator.PLATE_SIZE_JP)
        if jp_frame is not None and enabled:
            for child in jp_frame.winfo_children():
                if isinstance(child, ttk.Combobox) and child.cget("textvariable") != str(self.jp_style_var):
                    child.configure(state="normal")

    def _regenerate_registration(self) -> None:
        self._registration = plate_generator.generate_registration(plate_generator.active_pattern(self.cfg))
        self._schedule_preview(immediate=True)

    def _schedule_preview(self, immediate: bool = False) -> None:
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except tk.TclError:
                pass
            self._preview_job = None
        delay = 10 if immediate else 250
        self._preview_job = self.after(delay, self._update_preview)

    def _update_preview(self) -> None:
        self._preview_job = None
        pattern = plate_generator.active_pattern(self.cfg)
        errors = plate_generator.validate_plate_config(self.cfg)
        preview_fmt = self._preview_format()
        preview_rear = self.preview_side_var.get() == PLATE_PREVIEW_SIDE_REAR
        if self._registration is None and not plate_generator.validate_pattern(pattern):
            self._registration = plate_generator.generate_registration(pattern)
        self.registration_var.set(self._registration or "")
        self.error_var.set("\n".join(errors))
        if errors:
            self.preview_label.configure(image="", text="Preview unavailable until the issues below are fixed.")
            self._preview_photo = None
            self._fit_to_contents(center=True)
            return
        try:
            image = plate_generator.render_plate_preview(self.cfg, self._registration, fmt=preview_fmt, rear=preview_rear)
        except plate_generator.PlateError as exc:
            self.preview_label.configure(image="", text=str(exc))
            self._preview_photo = None
            self._fit_to_contents(center=True)
            return
        except Exception as exc:  # keep the dialog alive on unexpected renderer errors
            self.preview_label.configure(image="", text=f"Preview failed: {exc}")
            self._preview_photo = None
            self._fit_to_contents(center=True)
            return
        try:
            from PIL import ImageTk

            ratio = PLATE_PREVIEW_WIDTH / image.width
            preview = image.resize((PLATE_PREVIEW_WIDTH, max(1, round(image.height * ratio))))
            self._preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview unavailable: {exc}")
            self._preview_photo = None
        self._fit_to_contents(center=True)

    # -- apply/close ---------------------------------------------------------

    def _apply(self) -> None:
        will_generate = True if self.variant_name is None else self._editing_enabled()
        if will_generate:
            errors = plate_generator.validate_plate_config(self.cfg)
            if errors:
                self.app._show_error("Licence plates", "Fix these before applying:\n- " + "\n- ".join(errors), parent=self)
                return
        if self.set_id is not None:
            record = plate_generator.plate_set_by_id(self.set_id)
            if record is None:
                self.app._show_error("Licence plates", f"Plate set '{self.set_id}' no longer exists", parent=self)
                return
            record["config"] = self.cfg
            plate_generator.save_plate_set(record)
        elif self.variant_name is None:
            binding = plate_generator.normalized_plate_binding(self.app.conversion.get("plate"))
            binding["mode"] = plate_generator.PLATE_MODE_CUSTOM
            binding["setId"] = ""
            binding["config"] = self.cfg
            binding["customConfig"] = self.cfg
            binding["customDefined"] = True
            self.app.conversion["plate"] = binding
        else:
            mode = _key_for_label(self.plate_mode_labels, self.variant_mode_var.get(), "general")
            settings = self.app.conversion.setdefault("variants", {}).setdefault(self.variant_name, {})
            if isinstance(settings, dict):
                existing = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
                settings["plate"] = {
                    "mode": mode,
                    "setId": "",
                    "sourceConfig": self.variant_name if mode == plate_generator.PLATE_MODE_CUSTOM else "",
                    "customDefined": bool(existing.get("customDefined")) or mode == plate_generator.PLATE_MODE_CUSTOM,
                    "config": self.cfg,
                    "customConfig": self.cfg
                    if mode == plate_generator.PLATE_MODE_CUSTOM
                    else existing.get("customConfig"),
                }
        self.app._plate_settings_applied()
        if self.set_id is not None:
            record = plate_generator.plate_set_by_id(self.set_id)
            self.app.status_var.set(f"Plate set '{record['name'] if record else self.set_id}' updated")
        self._close()

    def _close(self) -> None:
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except tk.TclError:
                pass
        self.app.plate_editor_modal = None
        self.destroy()
