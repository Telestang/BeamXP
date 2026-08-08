from __future__ import annotations

from .shared import *

# Appended to the heading of the column a table is sorted by. Column fitting
# leaves room for one, so they live here rather than inline.
TREE_SORT_ASCENDING = " ▲"
TREE_SORT_DESCENDING = " ▼"


class WindowingMixin:
    """Window lifecycle, monitor placement, modal helpers, and Treeview sorting."""

    def _on_close(self) -> None:
        self._close_tree_combo_editor()
        self.part_resolver.shutdown(wait=False, cancel_futures=True)
        self.part_table_builder.shutdown(wait=False, cancel_futures=True)
        self.variant_detector.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def _set_app_icon(self) -> None:
        icon_path = app_icon_path()
        if icon_path is None:
            return
        try:
            if sys.platform == "win32":
                self.iconbitmap(default=str(icon_path))
            else:
                self.iconbitmap(str(icon_path))
        except tk.TclError:
            pass

    @staticmethod
    def _is_widget_or_child(widget: tk.Widget, parent: tk.Widget) -> bool:
        widget_path = str(widget)
        parent_path = str(parent)
        return widget_path == parent_path or widget_path.startswith(parent_path + ".")

    def _clear_part_filter_focus_on_click(self, event: tk.Event) -> None:
        filter_entry = self.part_filter_entry
        if filter_entry is None:
            return
        try:
            if self.focus_get() is not filter_entry:
                return
        except (tk.TclError, KeyError):
            return
        clicked = event.widget
        if clicked is not None and self._is_widget_or_child(clicked, filter_entry):
            return
        try:
            clicked.focus_set()
        except Exception:
            self.focus_set()

    def _part_display_name(self, object_id: str) -> str:
        if hasattr(self, "_part_row_mesh_id"):
            object_id = self._part_row_mesh_id(object_id)
        if self.context is None:
            return object_id
        obj = self.context.objects.get(object_id)
        if obj is not None and not obj.dae_path and obj.name and obj.name != object_id:
            return f"{obj.name} [{object_id}]"
        prefix = f"{self.context.vehicle_id}_"
        if object_id.startswith(prefix):
            return object_id[len(prefix) :]
        return object_id

    def _part_display_label(self, object_id: str, object_ids: list[str] | tuple[str, ...] | None = None) -> str:
        if hasattr(self, "_part_row_mesh_id"):
            object_id = self._part_row_mesh_id(object_id)
        display = self._part_display_name(object_id)
        if object_ids is None:
            object_ids = tuple(self.resolved_part_ids or self.current_part_ids or [])
        duplicates = [
            candidate
            for candidate in object_ids
            if candidate != object_id
            and self._part_display_name(candidate) == display
        ]
        if not duplicates:
            return display
        ordered = sorted(
            [object_id, *duplicates],
            key=lambda candidate: (
                getattr(self.context.objects.get(candidate), "x", 0.0) if self.context is not None else 0.0,
                candidate,
            ),
        )
        return f"{display} #{ordered.index(object_id) + 1}"

    def _configure_theme(self) -> None:
        self.ttk_style = ttk.Style(self)
        for theme in ("clam", "alt", "default"):
            if theme in self.ttk_style.theme_names():
                self.ttk_style.theme_use(theme)
                return

    def _maximize_on_start(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

    def _current_monitor_work_area(self) -> tuple[int, int, int, int]:
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class RECT(ctypes.Structure):
                    _fields_ = (
                        ("left", wintypes.LONG),
                        ("top", wintypes.LONG),
                        ("right", wintypes.LONG),
                        ("bottom", wintypes.LONG),
                    )

                class MONITORINFO(ctypes.Structure):
                    _fields_ = (
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", RECT),
                        ("rcWork", RECT),
                        ("dwFlags", wintypes.DWORD),
                    )

                monitor = ctypes.windll.user32.MonitorFromWindow(self.winfo_id(), 2)
                info = MONITORINFO()
                info.cbSize = ctypes.sizeof(MONITORINFO)
                if monitor and ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                    work = info.rcWork
                    return work.left, work.top, work.right, work.bottom
            except Exception:
                pass
        left = self.winfo_vrootx()
        top = self.winfo_vrooty()
        return left, top, left + self.winfo_vrootwidth(), top + self.winfo_vrootheight()

    def _place_modal_on_app_monitor(self, modal: tk.Toplevel) -> None:
        self.update_idletasks()
        modal.update_idletasks()

        width = modal.winfo_width()
        height = modal.winfo_height()
        if width <= 1:
            width = modal.winfo_reqwidth()
        if height <= 1:
            height = modal.winfo_reqheight()

        parent_x = self.winfo_rootx()
        parent_y = self.winfo_rooty()
        parent_w = max(self.winfo_width(), 1)
        parent_h = max(self.winfo_height(), 1)
        x = parent_x + (parent_w - width) // 2
        y = parent_y + (parent_h - height) // 2

        work_left, work_top, work_right, work_bottom = self._current_monitor_work_area()
        x = min(max(x, work_left), max(work_left, work_right - width))
        y = min(max(y, work_top), max(work_top, work_bottom - height))
        modal.geometry(f"{width}x{height}+{x}+{y}")

    def _show_error(self, title: str, message: str, *, parent: tk.Widget | None = None) -> None:
        messagebox.showerror(title, message, parent=parent or self)

    def _ask_open_filename(self, **options) -> str:
        return filedialog.askopenfilename(parent=self, **options)

    def _ask_directory(self, **options) -> str:
        return filedialog.askdirectory(parent=self, **options)

    def _configure_tree_rows(self, tree: ttk.Treeview) -> None:
        tree.tag_configure("evenrow", background="#ffffff")
        tree.tag_configure("oddrow", background="#cccccc")

    def _row_tags(self, index: int) -> tuple[str, ...]:
        return ("oddrow",) if index % 2 else ("evenrow",)

    # ----- one-shot column fitting -----------------------------------------

    # Room for the cell padding either side of the text, and for the
    # disclosure area ttk reserves in front of a #0 label.
    TREE_CELL_PADDING = 18
    TREE_INDENT_PADDING = 26

    def _tree_font(self, style_name: str, fallback: str) -> tkfont.Font:
        spec = self.ttk_style.lookup(style_name, "font") or fallback
        try:
            return tkfont.Font(root=self, font=spec)
        except tk.TclError:
            return tkfont.Font(root=self)

    def _fit_tree_columns(self, tree: ttk.Treeview, *, max_width: int = 460) -> None:
        """Widen every column to the widest thing in it.

        Column widths are the user's to drag, so this is a one-shot: the
        caller fires it when a table first has rows and then leaves the
        widths alone. Re-running it on every refresh would snap a column back
        the moment any cell in it changed, which is worse than never fitting.

        Cell text repeats heavily down a column -- most of them hold a mode
        name or a Y/N -- so only the distinct strings are measured. Headings
        are measured with the sort arrow allowed for, since any of them can
        be clicked and a heading fitted exactly would then clip it.

        The rows are read a row at a time rather than a cell at a time: each
        read crosses into Tcl, and the parts table is four hundred rows by
        fourteen columns.
        """
        rows = tree.get_children("")
        if not rows:
            return
        with timed_ui("_fit_tree_columns"):
            body_font = self._tree_font("Treeview", "TkDefaultFont")
            heading_font = self._tree_font("Treeview.Heading", "TkHeadingFont")
            # the plain labels, so a table fitted while sorted doesn't leave
            # room for two arrows
            labels = self._tree_heading_text.get(tree, {})
            # tk hands "show" back as Tcl string objects, so compare on str()
            raw_show = tree["show"]
            show = {
                str(part)
                for part in (raw_show.split() if isinstance(raw_show, str) else raw_show)
            }
            columns = list(tree["columns"])
            texts_by_column: dict[str, set[str]] = {column: set() for column in columns}
            for row in rows:
                for column, value in zip(columns, tree.item(row, "values")):
                    texts_by_column[column].add(str(value))
            if "tree" in show:
                columns.insert(0, "#0")
                texts_by_column["#0"] = {str(tree.item(row, "text")) for row in rows}
            for column in columns:
                extra = self.TREE_INDENT_PADDING if column == "#0" else self.TREE_CELL_PADDING
                texts = texts_by_column[column]
                widest = max((body_font.measure(text) for text in texts), default=0)
                if "headings" in show:
                    label = labels.get(column) or str(tree.heading(column, "text"))
                    widest = max(widest, heading_font.measure(label + TREE_SORT_DESCENDING))
                minwidth = int(tree.column(column, "minwidth") or 0)
                tree.column(column, width=max(minwidth, min(widest + extra, max_width)))

    # ----- generic click-to-sort for all table views -----------------------

    def _tree_column_name(self, tree: ttk.Treeview, column_id: str) -> str | None:
        """Map a display column id ('#3') to its logical column name so click
        handlers stay correct no matter how many columns a table has. Returns
        None for the tree column ('#0') or on any mismatch."""
        if not column_id or column_id == "#0":
            return None
        try:
            index = int(column_id[1:]) - 1
        except ValueError:
            return None
        columns = tree["columns"]
        if 0 <= index < len(columns):
            return str(columns[index])
        return None

    def _register_tree_headings(self, tree: ttk.Treeview, headings: dict[str, str]) -> None:
        """Record each heading's plain label and wire its heading button to sort
        the table by that column. `headings` maps a column id ('#0' or a column
        name) to its display label."""
        self._tree_heading_text[tree] = dict(headings)
        for column in headings:
            tree.heading(column, command=lambda c=column, t=tree: self._sort_tree(t, c))

    def _sort_tree(self, tree: ttk.Treeview, column: str) -> None:
        self._close_tree_combo_editor()
        prev_column, prev_descending = self._tree_sort.get(tree, (None, False))
        descending = column == prev_column and not prev_descending
        self._tree_sort[tree] = (column, descending)
        self._apply_tree_sort(tree)

    def _scroll_tree(self, tree: ttk.Treeview, axis: str, *args: object) -> None:
        """Close a cell overlay before moving the rows beneath it."""
        self._close_tree_combo_editor()
        getattr(tree, axis)(*args)

    @staticmethod
    def _sort_key(value: object) -> tuple[int, object]:
        # Numeric-parseable cells sort numerically ahead of text cells, so
        # coordinate/offset columns order by value while Y/N and text columns
        # order alphabetically -- and float is never compared against str.
        text = str(value).strip()
        try:
            return (0, float(text))
        except ValueError:
            return (1, text.lower())

    def _apply_tree_sort(self, tree: ttk.Treeview) -> None:
        """Reorder the rows in place per the tree's current sort selection.
        Row iids are preserved (only their visual order changes) so selection,
        preview picking, and part/config identity mapping are unaffected."""
        entry = self._tree_sort.get(tree)
        if not entry or entry[0] is None:
            return
        column, descending = entry
        children = list(tree.get_children(""))
        if not children:
            return
        if column == "#0":
            cell = lambda iid: tree.item(iid, "text")
        else:
            cell = lambda iid: tree.set(iid, column)
        ordered = sorted(children, key=lambda iid: self._sort_key(cell(iid)), reverse=descending)
        for index, iid in enumerate(ordered):
            tree.move(iid, "", index)
            tree.item(iid, tags=self._row_tags(index))
        self._update_sort_indicators(tree)

    def _restore_tree_order(self, tree: ttk.Treeview, previous_order: list[str]) -> None:
        children = list(tree.get_children(""))
        if not children:
            return
        existing = set(children)
        seen: set[str] = set()
        ordered: list[str] = []
        for iid in previous_order:
            if iid in existing and iid not in seen:
                ordered.append(iid)
                seen.add(iid)
        ordered.extend(iid for iid in children if iid not in seen)
        for index, iid in enumerate(ordered):
            tree.move(iid, "", index)
            tree.item(iid, tags=self._row_tags(index))

    @staticmethod
    def _tree_body_click(tree: ttk.Treeview, event: tk.Event) -> bool:
        return tree.identify_region(event.x, event.y) in {"tree", "cell"}

    def _update_sort_indicators(self, tree: ttk.Treeview) -> None:
        base = self._tree_heading_text.get(tree)
        if not base:
            return
        entry = self._tree_sort.get(tree)
        sort_column, descending = entry if entry else (None, False)
        arrow = TREE_SORT_DESCENDING if descending else TREE_SORT_ASCENDING
        for column, label in base.items():
            tree.heading(column, text=label + arrow if column == sort_column else label)
