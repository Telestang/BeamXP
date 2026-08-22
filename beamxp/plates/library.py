"""Reusable licence-plate set manager UI."""
from __future__ import annotations

import copy
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING

from beamxp.plates import generator as plate_generator

if TYPE_CHECKING:
    from beamxp.hand_drive_tool import HandDriveToolApp


class PlateLibraryDialog(tk.Toplevel):
    def __init__(self, app: HandDriveToolApp) -> None:
        super().__init__(app)
        self.app = app
        self.title("Licence Plate Library")
        self.transient(app)
        self.minsize(700, 360)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("name", "family", "pattern", "used")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for column, label, width in (
            ("name", "Name", 220),
            ("family", "Family", 80),
            ("pattern", "Pattern", 180),
            ("used", "Used here", 90),
        ):
            self.tree.heading(column, text=label, anchor="w")
            self.tree.column(column, width=width, anchor="w", stretch=column in {"name", "pattern"})
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<Double-1>", lambda _event: self._edit())

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for column in range(7):
            buttons.columnconfigure(column, weight=1 if column == 6 else 0)
        ttk.Button(buttons, text="New", command=self._new).grid(row=0, column=0)
        ttk.Button(buttons, text="Duplicate", command=self._duplicate).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(buttons, text="Rename", command=self._rename).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(buttons, text="Delete", command=self._delete).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(buttons, text="Edit...", command=self._edit).grid(row=0, column=4, padx=(6, 0))
        ttk.Button(buttons, text="Export plates mod...", command=self._export).grid(row=0, column=5, padx=(12, 0))
        ttk.Button(buttons, text="Close", command=self._close).grid(row=0, column=6, sticky="e")

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.refresh()
        app._place_modal_on_app_monitor(self)

    def _usage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        bindings: list[object] = [self.app.conversion.get("plate")]
        variants = self.app.conversion.get("variants", {})
        if isinstance(variants, dict):
            bindings.extend(settings.get("plate") for settings in variants.values() if isinstance(settings, dict))
        for raw in bindings:
            if not isinstance(raw, dict) or raw.get("mode") != plate_generator.PLATE_MODE_SET:
                continue
            set_id = str(raw.get("setId") or "")
            counts[set_id] = counts.get(set_id, 0) + 1
        return counts

    def refresh(self, select_id: str | None = None) -> None:
        previous = select_id or (self.tree.selection()[0] if self.tree.selection() else "")
        for item in self.tree.get_children():
            self.tree.delete(item)
        usage = self._usage_counts()
        for record in plate_generator.plate_set_records():
            cfg = plate_generator.normalized_plate_config(record.get("config"))
            self.tree.insert("", "end", iid=str(record["id"]), values=(
                record["name"],
                cfg["size"],
                plate_generator.active_pattern(cfg),
                usage.get(str(record["id"]), 0),
            ))
        if previous and self.tree.exists(previous):
            self.tree.selection_set(previous)
            self.tree.focus(previous)
        self.app._refresh_plate_choices()

    def _selected_record(self) -> dict[str, object] | None:
        selection = self.tree.selection()
        return plate_generator.plate_set_by_id(selection[0]) if selection else None

    def _new(self) -> None:
        name = simpledialog.askstring("New plate set", "Name:", parent=self)
        if not name or not name.strip():
            return
        set_id = plate_generator.unique_plate_set_id(name)
        cfg = plate_generator.default_plate_config()
        cfg["enabled"] = True
        plate_generator.save_plate_set({"id": set_id, "name": name.strip(), "config": cfg})
        self.refresh(set_id)
        self.app._open_plate_editor(None, set_id=set_id)

    def _duplicate(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        name = simpledialog.askstring(
            "Duplicate plate set",
            "Name:",
            initialvalue=f"{record['name']} Copy",
            parent=self,
        )
        if not name or not name.strip():
            return
        set_id = plate_generator.unique_plate_set_id(name)
        plate_generator.save_plate_set({"id": set_id, "name": name.strip(), "config": copy.deepcopy(record["config"])})
        self.refresh(set_id)

    def _rename(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        name = simpledialog.askstring("Rename plate set", "Name:", initialvalue=str(record["name"]), parent=self)
        if not name or not name.strip():
            return
        record["name"] = name.strip()
        plate_generator.save_plate_set(record)
        self.refresh(str(record["id"]))

    def _delete(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        if not messagebox.askyesno(
            "Delete plate set",
            f"Delete '{record['name']}'? Referencing conversions will use their last saved snapshot and show a build warning.",
            parent=self,
        ):
            return
        plate_generator.delete_plate_set(str(record["id"]))
        self.refresh()

    def _edit(self) -> None:
        record = self._selected_record()
        if record is not None:
            self.app._open_plate_editor(None, set_id=str(record["id"]))

    def _export(self) -> None:
        records = plate_generator.plate_set_records()
        if not records:
            messagebox.showinfo("Export plates mod", "Create a plate set first.", parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title("Export BeamXP plates mod")
        dialog.transient(self)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)
        ttk.Label(dialog, text="Select the reusable designs to include:", padding=(10, 10, 10, 4)).grid(row=0, column=0, sticky="w")
        names = tk.Listbox(dialog, selectmode="multiple", exportselection=False, height=min(12, len(records)))
        names.grid(row=1, column=0, sticky="nsew", padx=10)
        for index, record in enumerate(records):
            names.insert("end", str(record["name"]))
            names.selection_set(index)
        # Only pre-tick the install when there is a real folder to install
        # into: the mods setting carries BeamNG's standard path whether or not
        # the game is installed.
        mods_configured = self.app._folder_is_usable(self.app._mods_folder())
        install_var = tk.BooleanVar(value=mods_configured)
        ttk.Checkbutton(dialog, text="Install into the configured BeamNG mods folder", variable=install_var).grid(
            row=2, column=0, sticky="w", padx=10, pady=(8, 0)
        )
        buttons = ttk.Frame(dialog, padding=10)
        buttons.grid(row=3, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)

        def export() -> None:
            selected = [records[index] for index in names.curselection()]
            try:
                target = plate_generator.default_plate_export_path()
                result = plate_generator.export_plate_sets(selected, target)
                installed: Path | None = None
                if install_var.get():
                    mods = self.app._mods_folder()
                    if mods is None:
                        raise RuntimeError("Configure the BeamNG mods folder before installing")
                    mods.mkdir(parents=True, exist_ok=True)
                    installed = mods / target.name
                    shutil.copy2(target, installed)
                message = f"Exported {result['designs']} set(s) to {target}"
                if installed:
                    message += f"\nInstalled to {installed}"
                self.app.status_var.set(message.replace("\n", "; "))
                messagebox.showinfo("Export plates mod", message, parent=dialog)
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("Export plates mod", str(exc), parent=dialog)

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).grid(row=0, column=1)
        ttk.Button(buttons, text="Export", command=export).grid(row=0, column=2, padx=(6, 0))
        self.app._place_modal_on_app_monitor(dialog)

    def _close(self) -> None:
        self.app.plate_library_modal = None
        self.destroy()
