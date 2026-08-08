from __future__ import annotations

from .shared import *


class LayoutMixin:
    """Construction and responsive layout of the main Tk interface."""

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(10, 8, 10, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(4, weight=1)

        # The thumbnail sits in its own column beside both header rows rather
        # than under the buttons: it is as tall as the two rows together, so
        # giving it a column costs no height instead of adding a third row's
        # worth.
        self.model_preview_label = ttk.Label(
            top,
            text="no\npreview",
            anchor="center",
            justify="center",
            relief="sunken",
            width=12,
        )
        self.model_preview_label.grid(
            row=0, column=0, rowspan=2, sticky="wns", padx=(0, 10)
        )

        # Row 0 -- where vehicles are found. The folder buttons replace hunting
        # for a specific zip; Load Zip stays as the fallback for a mod that is
        # mistyped, unlisted, or deliberately kept outside the mods folder.
        sources = ttk.Frame(top)
        sources.grid(row=0, column=1, columnspan=4, sticky="ew")
        self.game_folder_button = ttk.Button(
            sources, text="Game vehicles: not set", command=self._browse_game_vehicles_folder
        )
        self.game_folder_button.grid(row=0, column=0, sticky="w")
        self.mods_folder_button = ttk.Button(
            sources, text="Mods: not set", command=self._browse_mods_folder_and_rescan
        )
        self.mods_folder_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.open_button = ttk.Button(sources, text="Load Zip...", command=self._open_zip_dialog)
        self.open_button.grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.refresh_button = ttk.Button(
            sources,
            text="Refresh",
            command=lambda: self._load_selected_vehicle(force_reload=True),
            state="disabled",
        )
        self.refresh_button.grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Checkbutton(
            sources,
            text="Include Automation",
            variable=self.include_automation_var,
            command=self._on_include_automation_toggled,
        ).grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Label(sources, textvariable=self.folder_hint_var, foreground="#a04000").grid(
            row=0, column=5, sticky="w", padx=(12, 0)
        )

        ttk.Button(top, text="Save Config", command=self._save_config).grid(row=0, column=5, sticky="e")
        ttk.Button(top, text="Import Config", command=self._import_config_dialog).grid(
            row=0, column=6, sticky="e", padx=(6, 0)
        )

        # Row 1 -- which vehicle, directly under the source buttons.  The
        # thumbnail beside it follows the dropdown highlight, so scrolling the
        # list previews each model before committing.
        ttk.Label(top, text="Model").grid(row=1, column=1, sticky="w", pady=(6, 0))
        self.vehicle_combo = ttk.Combobox(
            top, textvariable=self.vehicle_var, state="disabled", width=34
        )
        self.vehicle_combo.grid(row=1, column=2, sticky="w", padx=(6, 12), pady=(6, 0))
        self.vehicle_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_model_selected())
        ttk.Label(top, textvariable=self.source_var).grid(
            row=1, column=3, columnspan=4, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        ttk.Label(top, textvariable=self.project_var).grid(row=2, column=0, columnspan=7, sticky="ew", pady=(6, 0))

        # ttk.PanedWindow cannot change orientation after creation, so keep
        # one paned window per orientation and move the two panes between
        # them when the window aspect ratio flips (see _on_root_configure).
        # The pane frames are children of the toplevel so both paned windows
        # may manage them.
        self.main_paned_h = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned_v = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.main_orientation: str | None = None
        self.main_sash_pending = False

        left = self.tables_pane = ttk.Frame(self)
        left.columnconfigure(0, weight=1)
        # Three stacked tables: variants, slots, parts. Variants and slots are
        # short lists and keep their requested height; parts is the long one
        # and takes the slack.
        left.rowconfigure(1, weight=0)
        left.rowconfigure(3, weight=0)
        left.rowconfigure(5, weight=1)

        right = self.preview_pane = ttk.Frame(self)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_variant_panel(left)
        self._build_slot_panel(left)
        self._build_part_panel(left)
        self._build_right_panel(right)

        self._apply_main_orientation("landscape")
        self.bind("<Configure>", self._on_root_configure, add="+")

        bottom = ttk.Frame(self, padding=(10, 4, 10, 8))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.detail_var).grid(row=0, column=0, sticky="w")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if event.width <= 1 or event.height <= 1:
            return
        self._apply_main_orientation("portrait" if event.height > event.width else "landscape")
        if self.main_sash_pending:
            # Deferred, because the pane has not been given the new geometry
            # yet at the moment this event arrives -- reading its width here
            # returns the old one, or 1 before the window is first mapped.
            self.after_idle(self._centre_main_sash)

    def _apply_main_orientation(self, mode: str) -> None:
        if mode == self.main_orientation:
            return
        self.main_orientation = mode
        for paned in (self.main_paned_h, self.main_paned_v):
            for pane in paned.panes():
                paned.forget(pane)
            paned.grid_remove()
        if mode == "landscape":
            paned = self.main_paned_h
            # Even weights, so a window resize grows the tables and the
            # preview together rather than handing it all to one of them.
            paned.add(self.tables_pane, weight=1)
            paned.add(self.preview_pane, weight=1)
        else:
            paned = self.main_paned_v
            paned.add(self.tables_pane, weight=1)
            paned.add(self.preview_pane, weight=1)
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        self.main_sash_pending = True
        self.after_idle(self._centre_main_sash)

    def _main_paned(self) -> ttk.PanedWindow:
        return self.main_paned_h if self.main_orientation == "landscape" else self.main_paned_v

    def _centre_main_sash(self) -> None:
        """Split the window down the middle, once per orientation.

        Weights only divide up space a resize adds; the sash itself starts
        wherever the panes' requested sizes put it, and the tables ask for far
        more than they get, which left the preview a strip down the side. Half
        and half is the useful default -- but only a default: the sash is the
        user's to drag, so this fires while armed and not again.
        """
        if not self.main_sash_pending:
            return
        paned = self._main_paned()
        span = (
            paned.winfo_width()
            if self.main_orientation == "landscape"
            else paned.winfo_height()
        )
        # Before the window is mapped there is nothing to halve; the Configure
        # that comes with mapping (and with the maximise on start) lands here
        # again with a real size.
        if span <= 1:
            return
        self.main_sash_pending = False
        paned.sashpos(0, span // 2)

    def _build_variant_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(header, text="Variants").pack(side="left")
        ttk.Button(header, text="Clear Builds", command=lambda: self._set_all_variants_selected(False)).pack(
            side="right"
        )
        ttk.Button(header, text="Convert All", command=lambda: self._set_all_variants_selected(True)).pack(
            side="right",
            padx=(0, 6),
        )
        self.recommend_button = ttk.Button(
            header,
            text="Recommend Transforms",
            command=self._open_recommendations_modal,
            state="disabled",
        )
        self.recommend_button.pack(side="right", padx=(0, 6))

        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("build", "config", "display", "stock_hand", "plate", "front_plate", "rear_plate")
        self.variant_tree = ttk.Treeview(frame, columns=columns, show="headings", height=8, selectmode="browse")
        headings = {
            "build": "Build",
            "config": "Config",
            "display": "Display Name",
            "stock_hand": "Stock drive side",
            "plate": "Plates",
            "front_plate": "Front plate",
            "rear_plate": "Rear plate",
        }
        widths = {
            "build": 88,
            "config": 130,
            "display": 260,
            "stock_hand": 110,
            "plate": 120,
            "front_plate": 94,
            "rear_plate": 94,
        }
        for col in columns:
            self.variant_tree.heading(
                col,
                text=headings[col],
                anchor="w",
            )
            self.variant_tree.column(
                col,
                width=widths[col],
                minwidth=48,
                stretch=col == "display",
                anchor="w",
            )
        self._register_tree_headings(self.variant_tree, headings)
        yscroll = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=lambda *args: self._scroll_tree(self.variant_tree, "yview", *args),
        )
        xscroll = ttk.Scrollbar(
            frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: self._scroll_tree(self.variant_tree, "xview", *args),
        )
        self.variant_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.variant_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(self.variant_tree)
        self.variant_tree.bind("<Button-1>", self._variant_click)
        self.variant_tree.bind("<Double-1>", self._variant_double_click)
        self.variant_tree.bind("<MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Shift-MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Button-4>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Button-5>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Configure>", lambda _event: self._close_tree_combo_editor(), add="+")

    def _build_slot_panel(self, parent: ttk.Frame) -> None:
        """Equivalent Parts and Triggers, side by side.

        Both are two-column tables about where a thing ends up rather than how
        it is transformed, and neither needs the full table width, so they
        share one row instead of stacking and squeezing Parts Used.
        """
        row = ttk.Frame(parent)
        row.grid(row=2, column=0, rowspan=2, sticky="nsew", pady=(10, 0))
        row.columnconfigure(0, weight=1, uniform="sidetables")
        row.columnconfigure(1, weight=1, uniform="sidetables")
        row.rowconfigure(1, weight=1)
        self._build_side_pair_table(row, column=0)
        self._build_trigger_table(row, column=1)

    def _build_side_pair_table(self, parent: ttk.Frame, column: int) -> None:
        """Vehicle-level left/right equivalent part relationships."""
        holder = ttk.Frame(parent)
        holder.grid(row=0, column=column, rowspan=2, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(1, weight=1)
        parent = holder
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Equivalent Parts").grid(row=0, column=0, sticky="w")
        self.slot_filter_entry = ttk.Entry(header, textvariable=self.slot_filter_var)
        self.slot_filter_entry.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        self.new_side_pair_button = ttk.Button(
            header,
            text="New",
            command=self._add_default_side_pair,
            state="disabled",
        )
        self.new_side_pair_button.grid(row=0, column=2, sticky="e")
        self.remove_side_pair_button = ttk.Button(
            header,
            text="Remove",
            command=self._remove_selected_side_pair,
            state="disabled",
        )
        self.remove_side_pair_button.grid(row=0, column=4, sticky="e", padx=(6, 0))
        self.clear_slot_pairs_button = ttk.Button(
            header,
            text="Clear",
            command=self._clear_slot_pairs,
            state="disabled",
        )
        self.clear_slot_pairs_button.grid(row=0, column=5, sticky="e", padx=(6, 0))
        self.slot_filter_var.trace_add("write", lambda *_args: self._refresh_slots())

        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        columns = ("left", "right")
        self.slot_tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=8, selectmode="browse"
        )
        headings = {
            "left": "Left Part",
            "right": "Right Part",
        }
        widths = {
            "left": 360,
            "right": 360,
        }
        for col in columns:
            self.slot_tree.heading(col, text=headings[col], anchor="w")
            self.slot_tree.column(
                col,
                width=widths[col],
                minwidth=44,
                stretch=True,
                anchor="w",
            )
        self._register_tree_headings(self.slot_tree, headings)
        yscroll = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=lambda *args: self._scroll_tree(self.slot_tree, "yview", *args),
        )
        xscroll = ttk.Scrollbar(
            frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: self._scroll_tree(self.slot_tree, "xview", *args),
        )
        self.slot_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.slot_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(self.slot_tree)
        self.slot_tree.bind("<Button-1>", self._slot_click)
        self.slot_tree.bind("<Double-1>", self._slot_click)
        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>", "<Configure>"):
            self.slot_tree.bind(sequence, lambda _event: self._close_tree_combo_editor(), add="+")

    def _build_trigger_table(self, parent: ttk.Frame, column: int) -> None:
        """Which mesh each interaction trigger travels with."""
        holder = ttk.Frame(parent)
        holder.grid(row=0, column=column, rowspan=2, sticky="nsew", padx=(6, 0) if column else (0, 6))
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(1, weight=1)

        header = ttk.Frame(holder)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Triggers").grid(row=0, column=0, sticky="w")
        self.trigger_filter_entry = ttk.Entry(header, textvariable=self.trigger_filter_var)
        self.trigger_filter_entry.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        self.reset_trigger_button = ttk.Button(
            header,
            text="Auto",
            command=self._reset_selected_trigger,
            state="disabled",
        )
        self.reset_trigger_button.grid(row=0, column=2, sticky="e")
        self.clear_triggers_button = ttk.Button(
            header,
            text="Clear",
            command=self._clear_trigger_modes,
            state="disabled",
        )
        self.clear_triggers_button.grid(row=0, column=3, sticky="e", padx=(6, 0))
        self.trigger_filter_var.trace_add("write", lambda *_args: self._refresh_triggers())

        frame = ttk.Frame(holder)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        columns = ("trigger", "mode")
        self.trigger_tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=8, selectmode="browse"
        )
        headings = {"trigger": "Trigger", "mode": "Transform"}
        widths = {"trigger": 330, "mode": 130}
        for col in columns:
            self.trigger_tree.heading(col, text=headings[col], anchor="w")
            self.trigger_tree.column(
                col, width=widths[col], minwidth=44, stretch=col == "trigger", anchor="w"
            )
        self._register_tree_headings(self.trigger_tree, headings)
        yscroll = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=lambda *args: self._scroll_tree(self.trigger_tree, "yview", *args),
        )
        xscroll = ttk.Scrollbar(
            frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: self._scroll_tree(self.trigger_tree, "xview", *args),
        )
        self.trigger_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.trigger_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(self.trigger_tree)
        self.trigger_tree.bind("<Button-1>", self._trigger_click)
        self.trigger_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._trigger_selection_changed()
        )
        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>", "<Configure>"):
            self.trigger_tree.bind(sequence, lambda _event: self._close_tree_combo_editor(), add="+")

    def _build_part_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.grid(row=4, column=0, sticky="ew", pady=(10, 4))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Mesh Transforms").grid(row=0, column=0, sticky="w")
        self.part_filter_entry = ttk.Entry(header, textvariable=self.filter_var)
        self.part_filter_entry.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        self.part_filter_entry.insert(0, "")
        self.show_all_parts_button = ttk.Button(
            header,
            text="Show All",
            command=lambda: self._set_all_parts_visible(True),
            state="disabled",
        )
        self.show_all_parts_button.grid(row=0, column=3, sticky="e", padx=(6, 0))
        self.hide_all_parts_button = ttk.Button(
            header,
            text="Hide All",
            command=lambda: self._set_all_parts_visible(False),
            state="disabled",
        )
        self.hide_all_parts_button.grid(row=0, column=4, sticky="e", padx=(6, 0))
        self.active_only_parts_button = ttk.Button(
            header,
            text="Active Only",
            command=self._show_active_parts_only,
            state="disabled",
        )
        self.active_only_parts_button.grid(row=0, column=5, sticky="e", padx=(6, 0))
        self.clear_solo_parts_button = ttk.Button(
            header,
            text="Clear Solo",
            command=self._clear_part_solo,
            state="disabled",
        )
        self.clear_solo_parts_button.grid(row=0, column=6, sticky="e", padx=(6, 0))
        self.filter_var.trace_add("write", lambda *_args: self._refresh_parts())

        frame = ttk.Frame(parent)
        frame.grid(row=5, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        columns = (
            "parttype", "visible", "solo", "active", "mode", "textureCorrection",
            "source", "children", "offset", "steering", "x", "y", "z",
        )
        self.part_tree = ttk.Treeview(frame, columns=columns, show=("tree", "headings"), selectmode="extended")
        self.part_tree.heading("#0", text="Mesh", anchor="w")
        self.part_tree.column("#0", width=250, minwidth=150, stretch=True, anchor="w")
        headings = {
            "parttype": "Role",
            "mode": "Transform",
            "source": "Source",
            "children": "Children",
            "offset": "Move X",
            "steering": "Steering Ref",
            "visible": "Visible",
            "solo": "Solo",
            "active": "Active",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "textureCorrection": "Texture Fix",
        }
        widths = {
            "parttype": 112,
            "mode": 118,
            "textureCorrection": 88,
            "source": 170,
            "children": 76,
            "offset": 82,
            "steering": 96,
            "visible": 70,
            "solo": 60,
            "active": 64,
            "x": 82,
            "y": 82,
            "z": 82,
        }
        for col in columns:
            self.part_tree.heading(
                col,
                text=headings[col],
                anchor="w",
            )
            self.part_tree.column(
                col,
                width=widths[col],
                minwidth=50,
                stretch=False,
                anchor="center" if col in {"steering", "visible", "solo", "active", "children", "textureCorrection"} else "w",
            )
        self._register_tree_headings(self.part_tree, {"#0": "Mesh", **headings})
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.part_tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.part_tree.xview)
        self.part_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.part_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(self.part_tree)
        self.part_tree.bind("<<TreeviewSelect>>", lambda _event: self._part_selection_changed())
        self.part_tree.bind("<Button-1>", self._part_click)
        self.part_tree.bind("<Double-1>", self._part_double_click)
        self.part_tree.bind("<MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Shift-MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Button-4>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Button-5>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Configure>", lambda _event: self._close_tree_combo_editor(), add="+")

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        self.viewer_holder = ttk.Frame(parent)
        self.viewer_holder.grid(row=0, column=0, sticky="nsew")
        self.viewer_holder.columnconfigure(0, weight=1)
        self.viewer_holder.rowconfigure(0, weight=1)
        ttk.Label(self.viewer_holder, text="Load a vehicle zip to use the built-in part viewer").grid(row=0, column=0)

        controls = ttk.LabelFrame(parent, text="Build Settings", padding=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Config").grid(row=0, column=0, sticky="w")
        self.preview_output_combo = ttk.Combobox(
            controls,
            textvariable=self.preview_output_var,
            state="disabled",
            width=28,
        )
        self.preview_output_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0))
        self.preview_output_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._preview_output_selected(),
        )
        self._wire_preview_output_popdown()

        ttk.Label(controls, text="Auto delta X").grid(row=1, column=0, sticky="w", pady=(6, 0))
        delta_row = ttk.Frame(controls)
        delta_row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))
        delta_row.columnconfigure(0, weight=1)
        ttk.Label(delta_row, textvariable=self.auto_delta_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            delta_row,
            text="Manual magnitude",
            variable=self.manual_delta_enabled,
            command=self._manual_delta_toggled,
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.manual_delta_entry = ttk.Entry(delta_row, textvariable=self.manual_delta_var, width=12)
        self.manual_delta_entry.grid(row=0, column=2, sticky="e", padx=(6, 0))
        self.manual_delta_entry.bind("<FocusOut>", lambda _event: self._commit_delta_from_ui())
        self.manual_delta_entry.bind("<Return>", lambda _event: self._commit_delta_from_ui())

        ttk.Label(controls, text="Licence plates").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.plate_summary_var = tk.StringVar(value="Off")
        plate_row = ttk.Frame(controls)
        plate_row.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))
        plate_row.columnconfigure(0, weight=1)
        self.plate_choice_combo = ttk.Combobox(
            plate_row,
            textvariable=self.plate_choice_var,
            state="readonly",
        )
        self.plate_choice_combo.grid(row=0, column=0, sticky="ew")
        self.plate_choice_combo.bind("<<ComboboxSelected>>", lambda _event: self._main_plate_choice_changed())
        self.plate_configure_button = ttk.Button(plate_row, text="Configure...", command=lambda: self._open_plate_editor(None))
        self.plate_configure_button.grid(row=0, column=1, sticky="e", padx=(6, 0))
        ttk.Button(plate_row, text="Library...", command=self._open_plate_library).grid(
            row=0, column=2, sticky="e", padx=(6, 0)
        )

        # The mods folder moved to the top bar as a folder button: it now picks
        # which vehicles are listed, not just where builds install to.
        ttk.Label(controls, text="Blender exe").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.blender_var).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(controls, text="Browse", command=self._browse_blender).grid(row=3, column=2, sticky="e", padx=(6, 0), pady=(6, 0))

        ttk.Label(controls, text="Derived").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Label(controls, textvariable=self.derived_output_var, wraplength=620).grid(
            row=4,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)
        self.install_button = ttk.Button(buttons, text="Build + Install", command=lambda: self._start_build(install=True))
        self.install_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.blender_button = ttk.Button(buttons, text="Blender Preview", command=self._start_blender_preview)
        self.blender_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.busy_progress = ttk.Progressbar(buttons, mode="indeterminate")
        self.busy_progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.busy_progress.grid_remove()
