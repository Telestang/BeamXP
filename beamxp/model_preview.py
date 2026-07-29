from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk


class ModelPreview(ttk.Frame):
    BOX_EDGES = (
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    def __init__(
        self,
        parent: tk.Widget,
        preview_by_id: dict[str, dict[str, object]],
    ) -> None:
        super().__init__(parent)
        self.preview_by_id = preview_by_id
        self.visible_object_ids: list[str] = []
        self.selected_object_ids: set[str] = set()
        self.dimmed_object_ids: set[str] = set()
        self.target = (0.0, 0.0, 0.0)
        self.base_zoom = 120.0
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.yaw = -0.75
        self.pitch = 0.45
        self.drag_start: tuple[int, int] | None = None
        self.drag_mode = "rotate"

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(toolbar, text="Preview").pack(side="left")
        ttk.Button(toolbar, text="Reset", command=self.reset_view, width=8).pack(side="right")
        ttk.Button(toolbar, text="Focus", command=self.focus_selected, width=8).pack(side="right", padx=(0, 6))

        self.canvas = tk.Canvas(self, width=420, height=420, background="#111317", highlightthickness=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<ButtonPress-1>", lambda event: self._start_drag(event, "rotate"))
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonPress-2>", lambda event: self._start_drag(event, "pan"))
        self.canvas.bind("<B2-Motion>", self._drag)
        self.canvas.bind("<ButtonPress-3>", lambda event: self._start_drag(event, "pan"))
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom(1.12))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom(1 / 1.12))

    def set_visible_ids(self, object_ids: list[str], *, reset: bool = False) -> None:
        self.visible_object_ids = [object_id for object_id in object_ids if object_id in self.preview_by_id]
        visible = set(self.visible_object_ids)
        self.selected_object_ids &= set(self.visible_object_ids)
        self.dimmed_object_ids &= visible
        if reset:
            self.reset_view()
        else:
            self.draw()

    def set_selected_ids(self, object_ids: set[str]) -> None:
        self.selected_object_ids = {
            object_id
            for object_id in object_ids
            if object_id in self.preview_by_id and object_id in set(self.visible_object_ids)
        }
        self.draw()

    def set_dimmed_ids(self, object_ids: set[str]) -> None:
        visible = set(self.visible_object_ids)
        self.dimmed_object_ids = {
            object_id
            for object_id in object_ids
            if object_id in self.preview_by_id and object_id in visible
        }
        self.draw()

    def reset_view(self) -> None:
        bounds = self._combined_bounds(self.visible_object_ids)
        if bounds is not None:
            self.target = self._bounds_center(bounds)
            self.base_zoom = self._zoom_for_bounds(bounds, margin=1.35)
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.draw()

    def focus_selected(self) -> None:
        ids = list(self.selected_object_ids) or self.visible_object_ids
        bounds = self._combined_bounds(ids)
        if bounds is None:
            return
        self.target = self._bounds_center(bounds)
        self.base_zoom = self._zoom_for_bounds(bounds, margin=2.6)
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.draw()

    def _start_drag(self, event: tk.Event, mode: str) -> None:
        self.drag_start = (event.x, event.y)
        self.drag_mode = mode

    def _drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        last_x, last_y = self.drag_start
        dx = event.x - last_x
        dy = event.y - last_y
        self.drag_start = (event.x, event.y)
        if self.drag_mode == "pan":
            self.pan_x += dx
            self.pan_y += dy
        else:
            self.yaw += dx * 0.01
            self.pitch = max(-1.25, min(1.25, self.pitch + dy * 0.01))
        self.draw()

    def _mouse_wheel(self, event: tk.Event) -> None:
        self._zoom(1.12 if event.delta > 0 else 1 / 1.12)

    def _zoom(self, factor: float) -> None:
        self.zoom_factor = max(0.08, min(80.0, self.zoom_factor * factor))
        self.draw()

    def _combined_bounds(
        self,
        object_ids: list[str],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        bounds_list = [
            self.preview_by_id[object_id]["bounds"]
            for object_id in object_ids
            if object_id in self.preview_by_id
        ]
        if not bounds_list:
            return None
        min_points = [bounds[0] for bounds in bounds_list]
        max_points = [bounds[1] for bounds in bounds_list]
        return (
            (
                min(point[0] for point in min_points),
                min(point[1] for point in min_points),
                min(point[2] for point in min_points),
            ),
            (
                max(point[0] for point in max_points),
                max(point[1] for point in max_points),
                max(point[2] for point in max_points),
            ),
        )

    def _bounds_center(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        min_point, max_point = bounds
        return (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        )

    def _zoom_for_bounds(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        *,
        margin: float,
    ) -> float:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        min_point, max_point = bounds
        max_dim = max(
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2],
            0.08,
        )
        return min(width, height) / (max_dim * margin)

    def _expanded_bounds(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        min_point, max_point = bounds
        spans = (
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2],
        )
        pad = max(max(spans) * 0.02, 0.015)
        out_min = list(min_point)
        out_max = list(max_point)
        for idx, span in enumerate(spans):
            if span < pad:
                out_min[idx] -= pad
                out_max[idx] += pad
        return tuple(out_min), tuple(out_max)

    def _bounds_corners(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        min_point, max_point = self._expanded_bounds(bounds)
        min_x, min_y, min_z = min_point
        max_x, max_y, max_z = max_point
        return [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (min_x, max_y, min_z),
            (max_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (min_x, max_y, max_z),
            (max_x, max_y, max_z),
        ]

    def _project(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        x = point[0] - self.target[0]
        y = point[1] - self.target[1]
        z = point[2] - self.target[2]

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        x1 = cos_yaw * x - sin_yaw * y
        y1 = sin_yaw * x + cos_yaw * y

        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)
        y2 = cos_pitch * y1 - sin_pitch * z
        z2 = sin_pitch * y1 + cos_pitch * z

        scale = self.base_zoom * self.zoom_factor
        screen_x = self.canvas.winfo_width() / 2 + self.pan_x + x1 * scale
        screen_y = self.canvas.winfo_height() / 2 + self.pan_y - z2 * scale
        return screen_x, screen_y, y2

    def draw(self) -> None:
        self.canvas.delete("all")
        if not self.visible_object_ids:
            return

        selected = self.selected_object_ids
        context_ids = [object_id for object_id in self.visible_object_ids if object_id not in selected]
        context_ids.sort(key=self._depth_for_object, reverse=True)

        for object_id in context_ids:
            color = "#24282d" if object_id in self.dimmed_object_ids else "#53606c"
            self._draw_box(object_id, color, 1)

        for object_id in selected:
            self._draw_sample_points(object_id)
            self._draw_box(object_id, "#ffcc33", 3)
            self._draw_marker(object_id)

        self._draw_axes()

    def _depth_for_object(self, object_id: str) -> float:
        center = self.preview_by_id[object_id]["center"]
        return self._project(center)[2]

    def _draw_box(self, object_id: str, color: str, width: int) -> None:
        bounds = self.preview_by_id[object_id]["bounds"]
        points = [self._project(point) for point in self._bounds_corners(bounds)]
        for start, end in self.BOX_EDGES:
            x1, y1, _ = points[start]
            x2, y2, _ = points[end]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _draw_sample_points(self, object_id: str) -> None:
        points = self.preview_by_id[object_id].get("sample_points", [])
        for point in points:
            x, y, _ = self._project(point)
            self.canvas.create_rectangle(x - 1, y - 1, x + 1, y + 1, outline="", fill="#ff8c42")

    def _draw_marker(self, object_id: str) -> None:
        center = self.preview_by_id[object_id]["center"]
        x, y, _ = self._project(center)
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#ffffff", width=2)
        self.canvas.create_text(x + 8, y - 8, text=object_id, fill="#ffffff", anchor="sw")

    def _draw_axes(self) -> None:
        origin = (50.0, self.canvas.winfo_height() - 45.0)
        axes = [
            ((0.35, 0.0, 0.0), "X", "#ff6b6b"),
            ((0.0, 0.35, 0.0), "Y", "#74c69d"),
            ((0.0, 0.0, 0.35), "Z", "#8ecae6"),
        ]
        old_target = self.target
        old_base_zoom = self.base_zoom
        old_zoom_factor = self.zoom_factor
        old_pan = (self.pan_x, self.pan_y)
        self.target = (0.0, 0.0, 0.0)
        self.base_zoom = 90.0
        self.zoom_factor = 1.0
        self.pan_x = origin[0] - self.canvas.winfo_width() / 2
        self.pan_y = origin[1] - self.canvas.winfo_height() / 2
        for point, label, color in axes:
            x, y, _ = self._project(point)
            self.canvas.create_line(origin[0], origin[1], x, y, fill=color, width=2)
            self.canvas.create_text(x + 4, y, text=label, fill=color, anchor="w")
        self.target = old_target
        self.base_zoom = old_base_zoom
        self.zoom_factor = old_zoom_factor
        self.pan_x, self.pan_y = old_pan
