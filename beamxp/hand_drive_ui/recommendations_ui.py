from __future__ import annotations

from .recommendation_engine import build_mode_recommendations
from .shared import *


class RecommendationsUIMixin:
    """The recommendation review/apply workflow."""

    def _part_option_label(
        self,
        object_id: str,
        object_ids: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        display = self._part_display_label(object_id, object_ids)
        if display == object_id:
            return object_id
        return f"{display} ({object_id})"

    def _name_pair_candidate(self, object_id: str, candidates: list[str]) -> str | None:
        lower_to_id = {candidate.lower(): candidate for candidate in candidates}
        pairs = (
            ("_FL", "_FR"),
            ("_FR", "_FL"),
            ("_RL", "_RR"),
            ("_RR", "_RL"),
            ("_left", "_right"),
            ("_right", "_left"),
            ("_lhd", "_rhd"),
            ("_rhd", "_lhd"),
            ("_driver", "_passenger"),
            ("_passenger", "_driver"),
            ("_L", "_R"),
            ("_R", "_L"),
            ("-L", "-R"),
            ("-R", "-L"),
            (".L", ".R"),
            (".R", ".L"),
        )
        lowered = object_id.lower()
        for old, new in pairs:
            old_lower = old.lower()
            if old_lower not in lowered:
                continue
            candidate_lower = lowered.replace(old_lower, new.lower(), 1)
            if candidate_lower in lower_to_id:
                return lower_to_id[candidate_lower]
        return None

    def _geometry_pair_candidate(self, object_id: str, candidates: list[str]) -> str | None:
        if self.context is None or object_id not in self.context.objects:
            return None
        obj = self.context.objects[object_id]
        best: tuple[float, str] | None = None
        for candidate in candidates:
            if candidate == object_id:
                continue
            other = self.context.objects.get(candidate)
            if other is None:
                continue
            if abs(obj.x) > 0.02 and abs(other.x) > 0.02 and obj.x * other.x > 0:
                continue
            score = (
                abs(obj.x + other.x) * 4.0
                + abs(obj.y - other.y)
                + abs(obj.z - other.z)
                + (0.0 if obj.dae_path == other.dae_path else 0.5)
            )
            if best is None or score < best[0]:
                best = (score, candidate)
        return best[1] if best is not None else None

    def _structural_candidate_ids(self, object_id: str) -> list[str]:
        if self.context is None:
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for candidate in self.resolved_part_ids or self.current_part_ids:
            candidate = self._part_row_mesh_id(candidate)
            if candidate == object_id or candidate in seen:
                continue
            obj = self.context.objects.get(candidate)
            if obj is None or not obj.dae_path:
                continue
            candidates.append(candidate)
            seen.add(candidate)
        candidates.sort(key=lambda item: self._part_display_name(item).lower())
        return candidates

    def _suggest_structural_source(self, object_id: str, candidates: list[str]) -> str | None:
        return self._name_pair_candidate(object_id, candidates) or self._geometry_pair_candidate(
            object_id,
            candidates,
        )

    def _choose_structural_source(self, object_id: str) -> str | None:
        if self.context is None:
            return None
        candidates = self._structural_candidate_ids(object_id)
        existing = str(self._get_part_setting(object_id, "mirrorSource", "") or "")
        if existing and existing in self.context.objects and existing != object_id and existing not in candidates:
            candidates.append(existing)
        suggested = self._suggest_structural_source(object_id, candidates)
        if suggested and suggested not in candidates:
            candidates.append(suggested)
        if not candidates:
            self._show_error("Swap Mesh", "No other used mesh is available to mirror from.")
            return None

        label_universe = [object_id, *candidates]
        value_by_label = {
            self._part_option_label(candidate, label_universe): candidate
            for candidate in candidates
        }
        label_by_value = {value: label for label, value in value_by_label.items()}

        modal = tk.Toplevel(self)
        modal.title("Swap Mesh")
        modal.transient(self)
        modal.resizable(False, False)
        modal.columnconfigure(1, weight=1)

        ttk.Label(modal, text="Mesh").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(modal, text=self._part_option_label(object_id, label_universe)).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 10),
            pady=(10, 4),
        )
        ttk.Label(modal, text="Mirror From").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        source_var = tk.StringVar()
        combo = ttk.Combobox(
            modal,
            textvariable=source_var,
            values=list(value_by_label),
            state="readonly",
            width=52,
            height=min(max(len(value_by_label), 1), 16),
        )
        combo.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)

        suggestion_text = (
            f"Suggested: {self._part_option_label(suggested, label_universe)}"
            if suggested
            else "Suggested: no obvious pair found"
        )
        ttk.Label(modal, text=suggestion_text).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(0, 8),
        )

        if existing and existing in label_by_value:
            source_var.set(label_by_value[existing])
        elif suggested and suggested in label_by_value:
            source_var.set(label_by_value[suggested])

        result: dict[str, str | None] = {"source": None}

        def commit() -> None:
            selected = value_by_label.get(source_var.get())
            if not selected:
                self._show_error("Swap Mesh", "Select a source mesh to mirror from.", parent=modal)
                return
            if selected == object_id:
                self._show_error("Swap Mesh", "A mesh cannot swap from itself.", parent=modal)
                return
            result["source"] = selected
            modal.destroy()

        buttons = ttk.Frame(modal)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", padx=10, pady=(0, 10))
        ttk.Button(buttons, text="Cancel", command=modal.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply", command=commit).pack(side="right", padx=(0, 6))

        modal.protocol("WM_DELETE_WINDOW", modal.destroy)
        modal.bind("<Escape>", lambda _event: modal.destroy())
        modal.bind("<Return>", lambda _event: commit())
        self._place_modal_on_app_monitor(modal)
        combo.focus_set()
        modal.grab_set()
        self.wait_window(modal)
        return result["source"]

    def _open_recommendations_modal(self) -> None:
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        object_ids = list(self.resolved_part_ids or self.current_part_ids)
        if not object_ids:
            self._show_error(
                "No parts",
                "Select one or more variants and wait for the used-parts list to finish loading.",
            )
            return

        if self.recommendation_modal is not None and self.recommendation_modal.winfo_exists():
            self.recommendation_modal.lift()
            return

        self.recommendation_seq += 1
        seq = self.recommendation_seq
        self.recommendation_rows = {}

        modal = tk.Toplevel(self)
        self.recommendation_modal = modal
        modal.title("Recommended Mesh Transforms")
        modal.transient(self)
        modal.geometry("1150x560")
        modal.minsize(820, 420)
        modal.columnconfigure(0, weight=1)
        modal.rowconfigure(2, weight=1)

        top = ttk.Frame(modal, padding=(10, 10, 10, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(3, weight=1)
        select_all_button = ttk.Button(top, text="Select All", command=lambda: self._set_all_recommendations(True))
        clear_all_button = ttk.Button(top, text="Clear All", command=lambda: self._set_all_recommendations(False))
        self.apply_recommendations_button = ttk.Button(
            top,
            text="Apply Selected",
            command=self._apply_selected_recommendations,
            state="disabled",
        )
        select_all_button.grid(row=0, column=0, sticky="w")
        clear_all_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.apply_recommendations_button.grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Button(top, text="Close", command=modal.destroy).grid(row=0, column=4, sticky="e")

        self.recommendation_status_var = tk.StringVar(value="Finding recommendations...")
        ttk.Label(modal, textvariable=self.recommendation_status_var, padding=(10, 0, 10, 4)).grid(
            row=1,
            column=0,
            sticky="ew",
        )

        frame = ttk.Frame(modal, padding=(10, 0, 10, 10))
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("apply", "mode", "part", "source", "equivalent", "current", "reason")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.recommendation_tree = tree
        headings = {
            "apply": "Selected",
            "mode": "Transform",
            "part": "Mesh",
            "source": "Swap Source",
            "equivalent": "Equivalent Parts",
            "current": "Current",
            "reason": "Reason",
        }
        widths = {
            "apply": 54,
            "mode": 132,
            "part": 290,
            "source": 250,
            "equivalent": 110,
            "current": 190,
            "reason": 220,
        }
        for column in columns:
            tree.heading(
                column,
                text=headings[column],
                anchor="w",
            )
            tree.column(
                column,
                width=widths[column],
                minwidth=50,
                stretch=column in {"part", "reason"},
                anchor="center" if column in {"apply", "equivalent"} else "w",
            )
        self._register_tree_headings(tree, headings)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(tree)
        tree.bind("<Button-1>", self._recommendation_click)

        select_all_button.configure(state="disabled")
        clear_all_button.configure(state="disabled")
        self.recommendation_select_all_button = select_all_button
        self.recommendation_clear_all_button = clear_all_button

        def closed() -> None:
            self.recommendation_seq += 1
            self.recommendation_modal = None
            self._tree_sort.pop(tree, None)
            self._tree_heading_text.pop(tree, None)
            self.recommendation_tree = None
            self.recommendation_rows = {}
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", closed)
        modal.bind("<Escape>", lambda _event: closed())
        self._place_modal_on_app_monitor(modal)

        worker = threading.Thread(
            target=self._recommendations_worker,
            args=(seq, self.context, object_ids),
            daemon=True,
        )
        worker.start()
        modal.grab_set()
        modal.focus_set()

    def _recommendations_worker(
        self,
        seq: int,
        context: core.VehicleContext,
        object_ids: list[str],
    ) -> None:
        try:
            recommendations = build_mode_recommendations(context, object_ids)
            self.worker_queue.put(("recommendations_success", (seq, context, recommendations)))
        except Exception as exc:
            self.worker_queue.put(("recommendations_error", (seq, exc)))

    def _handle_recommendations_success(self, payload: object) -> None:
        seq, context, recommendations = payload
        if seq != self.recommendation_seq or context is not self.context:
            return
        modal = self.recommendation_modal
        tree = self.recommendation_tree
        if modal is None or tree is None or not modal.winfo_exists():
            return
        for item in tree.get_children():
            tree.delete(item)
        self.recommendation_rows = {}
        label_universe = sorted({
            str(recommendation.get(key) or "")
            for recommendation in recommendations
            for key in ("object_id", "source_id")
            if recommendation.get(key)
        })
        for index, recommendation in enumerate(recommendations):
            row_id = f"rec_{index}"
            self.recommendation_rows[row_id] = recommendation
            object_id = recommendation["object_id"]
            source_id = recommendation.get("source_id", "")
            current = self._recommendation_current_label(recommendation)
            tree.insert(
                "",
                "end",
                iid=row_id,
                tags=self._row_tags(index),
                values=(
                    "Y",
                    mode_label(recommendation["mode"]),
                    self._part_option_label(object_id, label_universe),
                    self._part_option_label(source_id, label_universe) if source_id else "",
                    "Y" if recommendation.get("equivalent") else "N",
                    current,
                    recommendation.get("reason", ""),
                ),
            )
        count = len(recommendations)
        self.recommendation_status_var.set(
            f"{count} recommendation(s) found for {len(self.resolved_part_ids or self.current_part_ids)} used part(s)."
        )
        state = "normal" if count else "disabled"
        self.recommendation_select_all_button.configure(state=state)
        self.recommendation_clear_all_button.configure(state=state)
        self.apply_recommendations_button.configure(state=state)

    def _handle_recommendations_error(self, payload: object) -> None:
        seq, exc = payload
        if seq != self.recommendation_seq:
            return
        if self.recommendation_modal is not None and self.recommendation_modal.winfo_exists():
            self.recommendation_status_var.set("Recommendation scan failed.")
        self._show_error("Recommendations failed", str(exc))

    def _recommendation_current_label(self, recommendation: dict[str, str]) -> str:
        object_id = recommendation["object_id"]
        mode = str(self._get_part_setting(object_id, "mode", core.MODE_SKIP))
        source_id = recommendation.get("source_id", "")
        if not source_id:
            return mode_label(mode)
        source_mode = str(self._get_part_setting(source_id, "mode", core.MODE_SKIP))
        return f"{mode_label(mode)} / {mode_label(source_mode)}"

    def _recommendation_click(self, event: tk.Event) -> str | None:
        tree = self.recommendation_tree
        if tree is None:
            return None
        if not self._tree_body_click(tree, event):
            return None
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if not item or self._tree_column_name(tree, column) != "apply":
            return None
        current = str(tree.set(item, "apply"))
        tree.set(item, "apply", "N" if current == "Y" else "Y")
        return "break"

    def _set_all_recommendations(self, selected: bool) -> None:
        tree = self.recommendation_tree
        if tree is None:
            return
        value = "Y" if selected else "N"
        for item in tree.get_children():
            tree.set(item, "apply", value)

    def _apply_selected_recommendations(self) -> None:
        if self.context is None or self.recommendation_tree is None:
            return
        selected_rows = [
            self.recommendation_rows[item]
            for item in self.recommendation_tree.get_children()
            if self.recommendation_tree.set(item, "apply") == "Y"
        ]
        if not selected_rows:
            self._show_error("No recommendations selected", "Select at least one recommendation to apply.")
            return

        applied = 0
        equivalences = 0
        for recommendation in selected_rows:
            mode = recommendation["mode"]
            object_id = recommendation["object_id"]
            source_id = recommendation.get("source_id", "")
            if recommendation.get("equivalent") and source_id:
                core.set_side_pair(
                    self.conversion,
                    object_id,
                    source_id,
                    kind=str(recommendation.get("pair_kind") or "part"),
                )
                equivalences += 1
            if mode == core.MODE_MIRROR_STRUCTURAL and source_id:
                self._apply_structural_pair(object_id, source_id)
                applied += 2
            else:
                # A seat's mesh stays untransformed: the equivalent parts row
                # is what moves it across, so Skip also clears any structural
                # pair a previous pass left on the two meshes.
                self._apply_single_part_mode(object_id, mode)
                applied += 1

        self._refresh_parts()
        self._refresh_slots()
        self._refresh_delta_label()
        self._update_detail()
        if self.recommendation_modal is not None and self.recommendation_modal.winfo_exists():
            self.recommendation_modal.destroy()
        self.recommendation_modal = None
        if self.recommendation_tree is not None:
            self._tree_sort.pop(self.recommendation_tree, None)
            self._tree_heading_text.pop(self.recommendation_tree, None)
        self.recommendation_tree = None
        self.recommendation_rows = {}
        summary = f"Applied {len(selected_rows)} recommendation(s) to {applied} part setting(s)"
        if equivalences:
            summary = f"{summary} and {equivalences} equivalent parts row(s)"
        self.status_var.set(summary)

    def _apply_single_part_mode(self, object_id: str, mode: str) -> None:
        settings = self._part_settings(object_id)
        if settings.get("mode") == core.MODE_MIRROR_STRUCTURAL:
            self._clear_structural_pair(object_id)
            settings = self._part_settings(object_id)
        settings["mode"] = mode
        settings["mirrorSource"] = None

    def _apply_structural_pair(self, object_id: str, source_id: str) -> None:
        self._clear_structural_pair(object_id)
        self._clear_structural_pair(source_id)
        settings = self._part_settings(object_id)
        source_settings = self._part_settings(source_id)
        settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        settings["mirrorSource"] = source_id
        source_settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        source_settings["mirrorSource"] = object_id
