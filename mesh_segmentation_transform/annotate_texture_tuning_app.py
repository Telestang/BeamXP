#!/usr/bin/env python3
"""Interactive tuning harness for the texture region annotator.

Loads a BeamNG vehicle ZIP, resolves a part's trim textures the same way the
mesh transform POC does, rasterises that material's UV islands, then runs
``annotate_texture_regions`` and shows what every filtering stage kept and
removed on top of the flat texture.

The pipeline itself lives in ``annotate_texture_regions``; this harness only
drives it, so what is shown here is exactly what a production run produces.
Tune the parameters on the right, press Run, and step through the stage tabs to
see which filter removed a region.

Usage:
    python mesh_segmentation_transform/annotate_texture_tuning_app.py
"""

from __future__ import annotations

import queue
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform.annotate_texture_regions import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_UV_ISLAND_SYMMETRY_CONFIG,
    DetectionRun,
    DetectionStage,
    MserConfig,
    UvIslandSymmetryConfig,
    UvIslandSymmetryMatch,
    analyse_uv_island_symmetry,
    load_image,
    run_detection,
)
from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (  # noqa: E402
    PRIMITIVE_TAGS,
    ArchiveTextureBinding,
    DaePart,
    LoadedDae,
    VehicleArchive,
    _blend_archive_preview_texture,
    _material_targets_by_symbol,
    _normalise_material_alias,
    _resolve_collada_material_for_symbol,
    archive_texture_choices_for_part,
    extract_archive_member,
    load_dae,
    load_dialog_directories,
    local_name,
    material_names_for_part,
    qname,
    save_dialog_directories,
    scan_vehicle_archive,
)
from mesh_segmentation_transform.extract_uv_island_paths import (  # noqa: E402
    uv_island_mask,
)

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

APP_NAME = "BeamXP Texture Annotation Tuning Harness"
VIEWER_BACKGROUND = "#1b1b1b"
KEPT_COLOUR = (60, 255, 60)
REJECTED_COLOUR = (255, 70, 70)

# UV island overlay.  Everything outside the UV domain is painted out solidly
# rather than tinted: those pixels carry no geometry, so they must read as
# "ignore this", not as texture you could still judge.  All widths below are in
# texture pixels and are baked once at full resolution, so they zoom with the
# image and the outline sits wholly inside the excluded area — no valid pixel is
# ever covered.
SHOW_UV_DOMAIN_BY_DEFAULT = True
UV_BLOCKED_REGION = "outside"  # "outside" the islands, or "inside" them
UV_BLOCKED_FILL = (0, 0, 0)
UV_BLOCKED_HATCH_COLOUR = (200, 35, 35)
UV_BLOCKED_HATCH_SPACING_PX = 14
UV_BLOCKED_HATCH_WIDTH_PX = 3
UV_ISLAND_OUTLINE_COLOUR = (255, 60, 60)
UV_ISLAND_OUTLINE_WIDTH_PX = 1
BOX_OUTLINE_WIDTH = 2
MINIMUM_BOX_DISPLAY_PX = 3  # draw tiny boxes at least this big so they stay visible
PREVIEW_LEVEL_MAX_PX = 2048  # cached half-resolution level keeps panning smooth
INITIAL_ZOOM = 1.0  # open at 1:1 on the middle of the atlas, not fitted
ZOOM_STEP = 1.25
MIN_SCALE = 0.02
MAX_SCALE = 16.0

# Parameters are grouped in this order; anything MserConfig gains later that is
# not listed here still appears, under "Other".
PARAMETER_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "MSER detector",
        ("delta", "min_area", "max_area", "max_variation", "min_diversity"),
    ),
    (
        "Box filtering",
        (
            "min_box_width_px",
            "min_box_height_px",
            "enable_aspect_ratio_filter",
            "min_aspect",
            "max_aspect",
        ),
    ),
    (
        "Box features",
        (
            "enable_box_feature_filter",
            "min_box_uv_coverage",
            "box_feature_colour_tolerance",
            "box_feature_context_px",
            "box_feature_min_domain_px",
            "box_min_feature_px",
        ),
    ),
    (
        "Repeating pattern",
        (
            "enable_pattern_group_filter",
            "max_pattern_autocorrelation",
            "pattern_window_scale",
            "pattern_min_window_px",
            "pattern_min_period_px",
            "pattern_max_period_px",
        ),
    ),
    (
        "Grouping",
        (
            "merge_distance_px",
            "min_group_union_region_px",
            "enable_island_bounded_grouping",
            "enable_overlap_group_merge",
            "enable_circular_groups",
            "circular_group_min_squareness",
            "circular_group_padding_px",
            "circular_group_colour_tolerance",
            "circular_group_max_corner_content",
            "enable_region_domain_filter",
            "min_region_uv_coverage",
        ),
    ),
    (
        "Final filters",
        (
            "enable_final_size_filter",
            "final_min_width_px",
            "final_min_height_px",
            "final_min_area_px",
            "enable_final_aspect_filter",
            "final_max_aspect",
            "final_region_padding_px",
        ),
    ),
)

# Driven by the loaded part or irrelevant to detection.
HIDDEN_PARAMETERS = frozenset(
    {
        "uv_island_mask_path",
        "green_colour",
        "red_colour",
        "green_thickness",
        "red_thickness",
    }
)

UV_SYMMETRY_HIDDEN_PARAMETERS = frozenset({"blue_colour", "blue_thickness"})
UV_SYMMETRY_OUTLINE_WIDTH = 2


# ---------------------------------------------------------------------------
# Detection run
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TextureSource:
    """A texture and its UV island mask, cached between runs."""

    key: tuple[str, str]
    image: np.ndarray
    uv_mask: np.ndarray
    uv_stats: dict[str, object]
    texture_path: Path
    material_symbols: tuple[str, ...] = ()


def material_symbols_for_binding(
    loaded: LoadedDae,
    binding: ArchiveTextureBinding,
) -> tuple[str, ...]:
    """Resolve which COLLADA primitive symbols a materials-JSON binding paints.

    The binding names a material ("scintilla_interior") while primitives name a
    symbol ("scintilla_interior-material"), so match on the same normalised
    aliases the export path uses rather than on the raw string.
    """
    root = loaded.tree.getroot()
    namespace = loaded.namespace
    materials_library = root.find(f"./{qname(namespace, 'library_materials')}")
    material_by_id = {
        material.get("id", ""): material
        for material in (
            materials_library.findall(qname(namespace, "material"))
            if materials_library is not None
            else []
        )
        if material.get("id")
    }
    targets_by_symbol = _material_targets_by_symbol(root, namespace)
    wanted = {
        _normalise_material_alias(value)
        for value in (binding.dae_material, binding.material_key)
        if value
    }

    symbols: list[str] = []
    for primitive in root.iter():
        if local_name(primitive.tag) not in PRIMITIVE_TAGS:
            continue
        symbol = (primitive.get("material") or "").strip()
        if not symbol or symbol in symbols:
            continue
        _material, aliases = _resolve_collada_material_for_symbol(
            symbol, material_by_id, targets_by_symbol
        )
        if any(_normalise_material_alias(alias) in wanted for alias in aliases):
            symbols.append(symbol)
    return tuple(symbols)


@dataclass(slots=True)
class RunResult:
    source: TextureSource
    run: DetectionRun
    symmetry_config: UvIslandSymmetryConfig
    symmetric_uv_islands: tuple[UvIslandSymmetryMatch, ...]
    seconds: float

    @property
    def stages(self) -> list[DetectionStage]:
        return self.run.stages


def build_texture_source(
    archive: VehicleArchive,
    loaded: LoadedDae,
    binding: ArchiveTextureBinding,
) -> TextureSource:
    """Extract the trim texture and rasterise its UV islands.

    UV islands come from the whole source DAE, so the mask covers every triangle
    sharing the atlas rather than one part's slice of it.
    """
    texture_path = extract_archive_member(archive, binding.texture_member)
    texture_path = _blend_archive_preview_texture(texture_path, archive, binding)
    image = load_image(texture_path)
    height, width = image.shape[:2]

    symbols = material_symbols_for_binding(loaded, binding)
    if not symbols:
        raise ValueError(
            f"No COLLADA primitive symbol resolved for material {binding.dae_material!r}."
        )

    root = loaded.tree.getroot()
    mask = np.zeros((height, width), dtype=bool)
    triangles = 0
    used: list[str] = []
    failures: list[str] = []
    for symbol in symbols:
        try:
            symbol_mask, stats = uv_island_mask(root, symbol, (width, height))
        except ValueError as exc:  # a symbol with no triangle primitives
            failures.append(f"{symbol}: {exc}")
            continue
        mask |= symbol_mask
        triangles += int(stats.get("triangles", 0))
        used.append(symbol)

    if not used:
        raise ValueError("; ".join(failures) or "no UV triangles found")

    return TextureSource(
        key=(binding.texture_member, binding.dae_material),
        image=image,
        uv_mask=mask,
        uv_stats={"triangles": triangles, "symbols": len(used)},
        texture_path=texture_path,
        material_symbols=tuple(used),
    )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


def build_masked_texture(rgb: np.ndarray, blocked: np.ndarray) -> Image.Image:
    """Return the texture with the excluded region painted out.

    Built once at full texture resolution, so the hatch pitch and the outline
    are measured in texture pixels and scale with the image.  The outline is
    eroded inwards from the boundary, keeping it entirely within the excluded
    region: every pixel the detector can actually see stays visible.
    """
    pixels = rgb.copy()
    pixels[blocked] = UV_BLOCKED_FILL

    height, width = blocked.shape
    spacing = max(UV_BLOCKED_HATCH_SPACING_PX, 1)
    columns = np.arange(width, dtype=np.int32)[None, :]
    rows = np.arange(height, dtype=np.int32)[:, None]
    hatch = ((columns + rows) % spacing) < max(UV_BLOCKED_HATCH_WIDTH_PX, 1)
    pixels[blocked & hatch] = UV_BLOCKED_HATCH_COLOUR

    thickness = max(UV_ISLAND_OUTLINE_WIDTH_PX, 1)
    interior = cv2.erode(
        blocked.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (thickness * 2 + 1,) * 2),
    )
    pixels[blocked & (interior == 0)] = UV_ISLAND_OUTLINE_COLOUR
    return Image.fromarray(pixels)


@dataclass(slots=True)
class ViewState:
    """Shared pan/zoom, so switching stage tabs holds the same framing."""

    scale: float = INITIAL_ZOOM
    origin_x: float = 0.0
    origin_y: float = 0.0
    initialised: bool = False


class StageCanvas:
    """One stage tab: the texture with that stage's boxes drawn over it."""

    def __init__(self, parent: tk.Widget, app: "TuningApp", stage_key: str) -> None:
        self.app = app
        self.stage_key = stage_key
        self.frame = ttk.Frame(parent)
        self.canvas = tk.Canvas(
            self.frame,
            background=VIEWER_BACKGROUND,
            highlightthickness=0,
            cursor="fleur",
        )
        self.canvas.pack(fill="both", expand=True)
        self.photo: ImageTk.PhotoImage | None = None
        self.last_render: Image.Image | None = None  # what this tab last drew
        self.dirty = True
        self._drag_anchor: tuple[int, int] | None = None

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Motion>", self._on_motion)

    # -- events ----------------------------------------------------------
    def _on_configure(self, _event: object) -> None:
        self.app.invalidate_views()
        self.app.render_active()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_anchor = (event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_anchor is None:
            return
        view = self.app.view
        dx = (event.x - self._drag_anchor[0]) / max(view.scale, 1e-6)
        dy = (event.y - self._drag_anchor[1]) / max(view.scale, 1e-6)
        self._drag_anchor = (event.x, event.y)
        view.origin_x -= dx
        view.origin_y -= dy
        self.app.clamp_view(self.canvas)
        self.app.invalidate_views()
        self.render()

    def _on_wheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            delta = 120 if getattr(event, "num", 0) == 4 else -120
        view = self.app.view
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        new_scale = min(max(view.scale * factor, MIN_SCALE), MAX_SCALE)
        if new_scale == view.scale:
            return
        # Keep the texture point under the cursor pinned while zooming.
        source_x = view.origin_x + event.x / view.scale
        source_y = view.origin_y + event.y / view.scale
        view.scale = new_scale
        view.origin_x = source_x - event.x / new_scale
        view.origin_y = source_y - event.y / new_scale
        self.app.clamp_view(self.canvas)
        self.app.invalidate_views()
        self.render()

    def _on_motion(self, event: tk.Event) -> None:
        view = self.app.view
        self.app.report_cursor(
            int(view.origin_x + event.x / max(view.scale, 1e-6)),
            int(view.origin_y + event.y / max(view.scale, 1e-6)),
        )

    # -- drawing ---------------------------------------------------------
    def render(self) -> None:
        """Redraw this tab from the shared view state."""
        self.dirty = False
        self.canvas.delete("all")
        result = self.app.result
        if result is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Load a vehicle ZIP, pick a part and a texture, then press Run.",
                fill="#888888",
            )
            return

        view = self.app.view
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        source_width, source_height = self.app.source_size()

        # Clamp the visible rectangle to the texture, then paste it where it
        # belongs on the canvas: the view may be larger than the texture.
        visible = Image.new("RGB", (width, height), VIEWER_BACKGROUND)
        vx0 = max(view.origin_x, 0.0)
        vy0 = max(view.origin_y, 0.0)
        vx1 = min(view.origin_x + width / view.scale, float(source_width))
        vy1 = min(view.origin_y + height / view.scale, float(source_height))
        if vx1 > vx0 and vy1 > vy0:
            base, base_scale = self.app.pyramid_level(view.scale)
            target_w = max(int(round((vx1 - vx0) * view.scale)), 1)
            target_h = max(int(round((vy1 - vy0) * view.scale)), 1)
            resample = Image.NEAREST if view.scale >= 1.0 else Image.BILINEAR
            box = (
                vx0 * base_scale,
                vy0 * base_scale,
                min(vx1 * base_scale, base.width),
                min(vy1 * base_scale, base.height),
            )
            patch = base.resize((target_w, target_h), resample=resample, box=box)
            visible.paste(
                patch,
                (
                    int(round((vx0 - view.origin_x) * view.scale)),
                    int(round((vy0 - view.origin_y) * view.scale)),
                ),
            )

        stage = self.app.stage_by_key(self.stage_key)
        if stage is not None:
            draw = ImageDraw.Draw(visible)
            if self.app.show_rejected.get():
                self._draw_boxes(draw, stage.rejected, REJECTED_COLOUR, view, width, height)
            if self.app.show_kept.get():
                self._draw_boxes(
                    draw,
                    stage.kept,
                    KEPT_COLOUR,
                    view,
                    width,
                    height,
                    stage.circles,
                )
            if self.app.show_uv_symmetry.get():
                self._draw_uv_symmetry(
                    draw,
                    result.symmetric_uv_islands,
                    result.symmetry_config,
                    view,
                )

        self.last_render = visible
        self.photo = ImageTk.PhotoImage(visible)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _draw_boxes(
        self,
        draw: ImageDraw.ImageDraw,
        boxes: tuple[tuple[int, int, int, int], ...],
        colour: tuple[int, int, int],
        view: ViewState,
        width: int,
        height: int,
        circles: tuple[int | None, ...] = (),
    ) -> None:
        for index, (x, y, w, h) in enumerate(boxes):
            x0 = (x - view.origin_x) * view.scale
            y0 = (y - view.origin_y) * view.scale
            x1 = x0 + max(w * view.scale, MINIMUM_BOX_DISPLAY_PX)
            y1 = y0 + max(h * view.scale, MINIMUM_BOX_DISPLAY_PX)
            if x1 < 0 or y1 < 0 or x0 > width or y0 > height:
                continue
            radius = circles[index] if index < len(circles) else None
            if radius:
                centre_x = (x + w / 2.0 - view.origin_x) * view.scale
                centre_y = (y + h / 2.0 - view.origin_y) * view.scale
                scaled = radius * view.scale
                draw.ellipse(
                    (
                        centre_x - scaled,
                        centre_y - scaled,
                        centre_x + scaled,
                        centre_y + scaled,
                    ),
                    outline=colour,
                    width=BOX_OUTLINE_WIDTH,
                )
                continue
            draw.rectangle((x0, y0, x1, y1), outline=colour, width=BOX_OUTLINE_WIDTH)

    def _draw_uv_symmetry(
        self,
        draw: ImageDraw.ImageDraw,
        matches: tuple[UvIslandSymmetryMatch, ...],
        config: UvIslandSymmetryConfig,
        view: ViewState,
    ) -> None:
        """Draw the independent blue UV-island contour overlay."""
        if not config.enable_uv_island_symmetry:
            return
        colour = tuple(reversed(config.blue_colour))
        for match in matches:
            for contour in match.contours:
                if len(contour) < 2:
                    continue
                points = [
                    (
                        (x - view.origin_x) * view.scale,
                        (y - view.origin_y) * view.scale,
                    )
                    for x, y in contour
                ]
                draw.line(
                    points + [points[0]],
                    fill=colour,
                    width=UV_SYMMETRY_OUTLINE_WIDTH,
                    joint="curve",
                )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class TuningApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1600x950")
        self.minsize(1200, 760)

        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.dialog_directories = load_dialog_directories()

        self.archive: VehicleArchive | None = None
        self.loaded: LoadedDae | None = None
        self.archive_dae_member: str | None = None
        self.part_by_label: dict[str, DaePart] = {}
        self.all_part_labels: tuple[str, ...] = ()
        self.archive_dae_by_label: dict[str, str] = {}
        self.texture_by_label: dict[str, ArchiveTextureBinding] = {}
        self.texture_source: TextureSource | None = None
        self.result: RunResult | None = None
        self.busy = False

        self.view = ViewState()
        self._syncing = False  # guards the summary <-> tab selection round trip
        self._pyramid: list[tuple[Image.Image, Image.Image, float]] = []  # plain, masked, ratio
        self.parameter_vars: dict[str, tk.Variable] = {}
        self.symmetry_parameter_vars: dict[str, tk.Variable] = {}
        self.stage_canvases: dict[str, StageCanvas] = {}

        self.source_var = tk.StringVar()
        self.archive_dae_var = tk.StringVar()
        self.search_var = tk.StringVar()
        self.part_var = tk.StringVar()
        self.texture_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a vehicle ZIP to begin.")
        self.detail_var = tk.StringVar(value="")
        self.cursor_var = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value="")
        self.show_kept = tk.BooleanVar(value=True)
        self.show_rejected = tk.BooleanVar(value=True)
        self.show_uv_domain = tk.BooleanVar(value=SHOW_UV_DOMAIN_BY_DEFAULT)
        self.show_uv_symmetry = tk.BooleanVar(value=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_handle: str | None = self.after(60, self._poll_worker)

    # -- construction ----------------------------------------------------
    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=(10, 8))
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Vehicle ZIP").grid(row=0, column=0, sticky="w", padx=(0, 8))
        entry = ttk.Entry(controls, textvariable=self.source_var)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<Return>", lambda _e: self._load_entry())
        ttk.Button(controls, text="Browse…", command=self._browse).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Button(controls, text="Load", command=self._load_entry).grid(
            row=0, column=3, padx=(6, 0)
        )

        ttk.Label(controls, text="Archive DAE").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        self.archive_dae_combo = ttk.Combobox(
            controls, textvariable=self.archive_dae_var, state="disabled"
        )
        self.archive_dae_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="Load DAE", command=self._load_selected_archive_dae).grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        ttk.Label(controls, text="Part filter").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        search = ttk.Entry(controls, textvariable=self.search_var)
        search.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        search.bind("<KeyRelease>", lambda _e: self._filter_parts())

        ttk.Label(controls, text="Part").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        self.part_combo = ttk.Combobox(
            controls, textvariable=self.part_var, state="disabled"
        )
        self.part_combo.grid(row=3, column=1, sticky="ew", pady=(6, 0))
        self.part_combo.bind("<<ComboboxSelected>>", lambda _e: self._part_changed())

        ttk.Label(controls, text="Texture").grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        self.texture_combo = ttk.Combobox(
            controls, textvariable=self.texture_var, state="disabled"
        )
        self.texture_combo.grid(row=4, column=1, sticky="ew", pady=(6, 0))
        self.texture_combo.bind("<<ComboboxSelected>>", lambda _e: self._texture_changed())
        self.run_button = ttk.Button(
            controls, text="Run detection", command=self._run, state="disabled"
        )
        self.run_button.grid(row=3, column=2, rowspan=2, columnspan=2, sticky="nsew", padx=(8, 0), pady=(6, 0))

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(4, 0))

        viewer = ttk.Frame(body)
        body.add(viewer, weight=4)

        overlay_bar = ttk.Frame(viewer)
        overlay_bar.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(
            overlay_bar, text="Kept (green)", variable=self.show_kept, command=self._redraw
        ).pack(side="left")
        ttk.Checkbutton(
            overlay_bar,
            text="Removed here (red)",
            variable=self.show_rejected,
            command=self._redraw,
        ).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(
            overlay_bar,
            text=f"Mask non-UV area ({UV_BLOCKED_REGION})",
            variable=self.show_uv_domain,
            command=self._redraw,
        ).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(
            overlay_bar,
            text="Symmetric UV islands (blue)",
            variable=self.show_uv_symmetry,
            command=self._redraw,
        ).pack(side="left", padx=(10, 0))
        ttk.Button(overlay_bar, text="Fit", command=self._fit_view).pack(side="left", padx=(16, 0))
        ttk.Button(overlay_bar, text="1:1", command=self._actual_size).pack(side="left", padx=(6, 0))
        ttk.Label(overlay_bar, textvariable=self.zoom_var).pack(side="left", padx=(12, 0))
        ttk.Label(overlay_bar, textvariable=self.cursor_var).pack(side="right")

        self.notebook = ttk.Notebook(viewer)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self._tab_changed())

        side = ttk.Frame(body)
        body.add(side, weight=2)
        self._build_side_panel(side)

        status = ttk.Frame(self, padding=(10, 6))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

        self._build_stage_tabs(self._placeholder_stages())

    def _build_side_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Stages", font=("", 10, "bold")).pack(anchor="w")
        self.summary = ttk.Treeview(
            parent,
            columns=("kept", "removed"),
            show="tree headings",
            height=8,
            selectmode="browse",
        )
        self.summary.heading("#0", text="Stage")
        self.summary.heading("kept", text="Kept")
        self.summary.heading("removed", text="Removed")
        self.summary.column("#0", width=170, stretch=True)
        self.summary.column("kept", width=60, anchor="e", stretch=False)
        self.summary.column("removed", width=70, anchor="e", stretch=False)
        self.summary.pack(fill="x", pady=(2, 2))
        self.summary.bind("<<TreeviewSelect>>", lambda _e: self._summary_selected())
        ttk.Label(
            parent,
            textvariable=self.detail_var,
            wraplength=330,
            justify="left",
            foreground="#4a6fa5",
        ).pack(fill="x", pady=(0, 8))

        header = ttk.Frame(parent)
        header.pack(fill="x")
        ttk.Label(header, text="Parameters", font=("", 10, "bold")).pack(side="left")
        ttk.Button(header, text="Reset", command=self._reset_parameters).pack(side="right")
        ttk.Button(header, text="Copy", command=self._copy_parameters).pack(
            side="right", padx=(0, 6)
        )

        # Scrollable parameter grid: MserConfig is long and grows over time.
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, pady=(4, 0))
        canvas = tk.Canvas(container, highlightthickness=0, width=340)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all(
            "<MouseWheel>",
            lambda event: (
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                if self._pointer_within(canvas, event)
                else None
            ),
        )
        self._build_parameter_widgets(inner)

    def _build_parameter_widgets(self, parent: ttk.Frame) -> None:
        by_name = {field.name: field for field in fields(MserConfig)}
        listed = {name for _title, names in PARAMETER_SECTIONS for name in names}
        remainder = tuple(
            name
            for name in by_name
            if name not in listed and name not in HIDDEN_PARAMETERS
        )
        sections = PARAMETER_SECTIONS + (("Other", remainder),) if remainder else PARAMETER_SECTIONS

        row = 0
        parent.columnconfigure(0, weight=1)
        for title, names in sections:
            ttk.Label(parent, text=title, font=("", 9, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(10, 2), padx=4
            )
            row += 1
            for name in names:
                field = by_name.get(name)
                if field is None or name in HIDDEN_PARAMETERS:
                    continue
                value = getattr(DEFAULT_CONFIG, name)
                if isinstance(value, bool):
                    variable: tk.Variable = tk.BooleanVar(value=value)
                    ttk.Checkbutton(parent, text=name, variable=variable).grid(
                        row=row, column=0, columnspan=2, sticky="w", padx=4
                    )
                else:
                    variable = tk.StringVar(value=str(value))
                    ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=4)
                    entry = ttk.Entry(parent, textvariable=variable, width=11)
                    entry.grid(row=row, column=1, sticky="e", padx=4)
                    entry.bind("<Return>", lambda _e: self._run())
                self.parameter_vars[name] = variable
                row += 1

        ttk.Label(parent, text="UV island symmetry", font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2), padx=4
        )
        row += 1
        for field in fields(UvIslandSymmetryConfig):
            name = field.name
            if name in UV_SYMMETRY_HIDDEN_PARAMETERS:
                continue
            value = getattr(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, name)
            if isinstance(value, bool):
                variable = tk.BooleanVar(value=value)
                ttk.Checkbutton(parent, text=name, variable=variable).grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=4
                )
            else:
                variable = tk.StringVar(value=str(value))
                ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=4)
                entry = ttk.Entry(parent, textvariable=variable, width=11)
                entry.grid(row=row, column=1, sticky="e", padx=4)
                entry.bind("<Return>", lambda _e: self._run())
            self.symmetry_parameter_vars[name] = variable
            row += 1

    def _placeholder_stages(self) -> list[DetectionStage]:
        return [
            DetectionStage(key=key, title=title, kept=())
            for key, title in (
                ("mser", "MSER boxes"),
                ("box_filter", "Box filtering"),
                ("grouped", "Initial grouping"),
                ("region_domain", "Domain recovery"),
                ("overlap_group", "Post-circle forced merge"),
                ("pattern_group", "Repeating pattern"),
                ("size", "Final size"),
                ("final_padding", "Final padding"),
            )
        ]

    def _build_stage_tabs(self, stages: list[DetectionStage]) -> None:
        """Rebuild the stage tabs, holding whichever stage was being viewed."""
        previous = self._active_canvas()
        previous_key = previous.stage_key if previous is not None else None
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.stage_canvases.clear()
        for index, stage in enumerate(stages):
            view = StageCanvas(self.notebook, self, stage.key)
            self.stage_canvases[stage.key] = view
            self.notebook.add(view.frame, text=f"{index + 1}. {stage.title}")
        if previous_key in self.stage_canvases:
            self._syncing = True
            try:
                self.notebook.select(
                    self.notebook.tabs()[list(self.stage_canvases).index(previous_key)]
                )
            finally:
                self._syncing = False

    # -- parameters ------------------------------------------------------
    def current_config(self) -> MserConfig:
        """Build an MserConfig from the widgets, raising on a bad value."""
        overrides: dict[str, object] = {}
        for field in fields(MserConfig):
            variable = self.parameter_vars.get(field.name)
            if variable is None:
                continue
            default = getattr(DEFAULT_CONFIG, field.name)
            raw = variable.get()
            if isinstance(default, bool):
                overrides[field.name] = bool(raw)
                continue
            text = str(raw).strip()
            try:
                overrides[field.name] = (
                    int(text) if isinstance(default, int) else float(text)
                )
            except ValueError as exc:
                raise ValueError(f"{field.name}: {text!r} is not a valid number") from exc
        return replace(DEFAULT_CONFIG, **overrides)

    def current_uv_symmetry_config(self) -> UvIslandSymmetryConfig:
        """Build the independent UV symmetry configuration from its widgets."""
        overrides: dict[str, object] = {}
        for field in fields(UvIslandSymmetryConfig):
            variable = self.symmetry_parameter_vars.get(field.name)
            if variable is None:
                continue
            default = getattr(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, field.name)
            raw = variable.get()
            if isinstance(default, bool):
                overrides[field.name] = bool(raw)
                continue
            text = str(raw).strip()
            try:
                overrides[field.name] = (
                    int(text) if isinstance(default, int) else float(text)
                )
            except ValueError as exc:
                raise ValueError(
                    f"{field.name}: {text!r} is not a valid number"
                ) from exc
        return replace(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, **overrides)

    def _reset_parameters(self) -> None:
        for name, variable in self.parameter_vars.items():
            default = getattr(DEFAULT_CONFIG, name)
            variable.set(default if isinstance(default, bool) else str(default))
        for name, variable in self.symmetry_parameter_vars.items():
            default = getattr(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, name)
            variable.set(default if isinstance(default, bool) else str(default))
        self.status_var.set("Parameters reset to the module defaults.")

    def _copy_parameters(self) -> None:
        """Copy changed values as MserConfig assignments, ready to paste."""
        try:
            config = self.current_config()
            symmetry_config = self.current_uv_symmetry_config()
        except ValueError as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        lines = [
            f"    {field.name}: {'bool' if isinstance(getattr(config, field.name), bool) else type(getattr(config, field.name)).__name__} = "
            f"{getattr(config, field.name)!r}"
            for field in fields(MserConfig)
            if field.name not in HIDDEN_PARAMETERS
            and getattr(config, field.name) != getattr(DEFAULT_CONFIG, field.name)
        ]
        symmetry_lines = [
            f"    {field.name}: {'bool' if isinstance(getattr(symmetry_config, field.name), bool) else type(getattr(symmetry_config, field.name)).__name__} = "
            f"{getattr(symmetry_config, field.name)!r}"
            for field in fields(UvIslandSymmetryConfig)
            if field.name not in UV_SYMMETRY_HIDDEN_PARAMETERS
            and getattr(symmetry_config, field.name)
            != getattr(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, field.name)
        ]
        if symmetry_lines:
            if lines:
                lines.append("")
            lines.append("# UvIslandSymmetryConfig")
            lines.extend(symmetry_lines)
        if not lines:
            self.status_var.set("No parameters differ from the module defaults.")
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines) + "\n")
        self.status_var.set(f"Copied {len(lines)} changed parameter(s) to the clipboard.")

    # -- loading ---------------------------------------------------------
    def _dialog_initial_directory(self) -> str | None:
        remembered = self.dialog_directories.get("source")
        if remembered and Path(remembered).is_dir():
            return remembered
        return None

    def _browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a BeamNG vehicle ZIP",
            initialdir=self._dialog_initial_directory(),
            filetypes=(("Vehicle archive", "*.zip"), ("COLLADA DAE", "*.dae")),
        )
        if not selected:
            return
        self.source_var.set(selected)
        self._load_entry()

    def _load_entry(self) -> None:
        text = self.source_var.get().strip().strip('"')
        if not text:
            return
        path = Path(text)
        if not path.is_file():
            messagebox.showerror(APP_NAME, f"File not found:\n{path}")
            return
        self.dialog_directories["source"] = str(path.parent)
        save_dialog_directories(self.dialog_directories)
        if path.suffix.lower() == ".zip":
            self._start_archive_scan(path)
        elif path.suffix.lower() == ".dae":
            messagebox.showinfo(
                APP_NAME,
                "Trim textures are resolved from a vehicle ZIP's materials JSON.\n"
                "Load the vehicle ZIP instead of a bare DAE.",
            )
        else:
            messagebox.showerror(APP_NAME, "Choose a .zip vehicle archive.")

    def _start_worker(self, job) -> None:
        def work() -> None:
            try:
                self.worker_queue.put(job())
            except Exception:
                self.worker_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _start_archive_scan(self, path: Path) -> None:
        self._cleanup_archive()
        self._set_busy(True, f"Indexing {path.name}…")

        def job() -> tuple[str, object]:
            workspace = Path(tempfile.mkdtemp(prefix="beamxp_tuning_zip_"))
            try:
                return "archive", scan_vehicle_archive(path, workspace)
            except Exception:
                shutil.rmtree(workspace, ignore_errors=True)
                raise

        self._start_worker(job)

    def _load_selected_archive_dae(self) -> None:
        member = self.archive_dae_by_label.get(self.archive_dae_var.get())
        archive = self.archive
        if archive is None or member is None:
            messagebox.showerror(APP_NAME, "Choose a DAE from the archive.")
            return
        self._set_busy(True, f"Loading {PurePosixPath(member).name}…")

        def job() -> tuple[str, object]:
            dae_path = extract_archive_member(archive, member)
            return "dae", (load_dae(dae_path), member)

        self._start_worker(job)

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                self._handle_worker_result(kind, payload)
        except queue.Empty:
            pass
        self._poll_handle = self.after(60, self._poll_worker)

    def _handle_worker_result(self, kind: str, payload: object) -> None:
        if kind == "error":
            self._set_busy(False, "Failed.")
            messagebox.showerror(APP_NAME, str(payload))
            return
        if kind == "archive":
            self._archive_loaded(payload)  # type: ignore[arg-type]
        elif kind == "dae":
            loaded, member = payload  # type: ignore[misc]
            self._dae_loaded(loaded, member)
        elif kind == "run":
            self._run_finished(payload)  # type: ignore[arg-type]

    def _archive_loaded(self, archive: VehicleArchive) -> None:
        self.archive = archive
        self.archive_dae_by_label = {
            f"{member}  ({archive.member_sizes.get(member, 0) / 1024 / 1024:.1f} MB)": member
            for member in archive.dae_members
        }
        labels = tuple(self.archive_dae_by_label)
        self.archive_dae_combo.configure(values=labels, state="readonly")
        if labels:
            self.archive_dae_var.set(labels[0])
        self._set_busy(
            False,
            f"{archive.path.name}: {len(archive.dae_members)} DAE(s), "
            f"{len(archive.materials)} materials-JSON entries. Pick a DAE and load it.",
        )

    def _dae_loaded(self, loaded: LoadedDae, member: str) -> None:
        self.loaded = loaded
        self.archive_dae_member = member
        self.part_by_label = {part.label: part for part in loaded.parts}
        self.all_part_labels = tuple(self.part_by_label)
        self.part_combo.configure(values=self.all_part_labels, state="readonly")
        if self.all_part_labels:
            self.part_var.set(self.all_part_labels[0])
        self._set_busy(False, f"{PurePosixPath(member).name}: {len(loaded.parts)} part(s).")
        self._part_changed()

    def _filter_parts(self) -> None:
        query = self.search_var.get().strip().lower()
        labels = [label for label in self.all_part_labels if query in label.lower()]
        self.part_combo.configure(values=labels)
        if self.part_var.get() not in labels:
            self.part_var.set(labels[0] if labels else "")
            self._part_changed()

    def _part_changed(self) -> None:
        self.texture_by_label.clear()
        self.texture_var.set("")
        self.texture_combo.configure(values=(), state="disabled")
        self.run_button.configure(state="disabled")

        part = self.part_by_label.get(self.part_var.get())
        if self.archive is None or self.loaded is None or part is None:
            return
        try:
            choices = archive_texture_choices_for_part(self.archive, self.loaded, part)
        except Exception as exc:
            self.status_var.set(f"Material resolution failed: {exc}")
            return
        if not choices:
            names = ", ".join(material_names_for_part(self.loaded, part))
            self.status_var.set(
                f"No materials-JSON base colour resolved for: {names or 'no bound material'}"
            )
            return

        filename_counts: dict[str, int] = defaultdict(int)
        for binding in choices:
            filename_counts[PurePosixPath(binding.texture_member).name.lower()] += 1
        for binding in choices:
            filename = PurePosixPath(binding.texture_member).name
            label = filename
            if filename_counts[filename.lower()] > 1:
                label += f"  —  {binding.dae_material}"
            suffix = 2
            base_label = label
            while label in self.texture_by_label:
                label = f"{base_label} #{suffix}"
                suffix += 1
            self.texture_by_label[label] = binding

        labels = tuple(self.texture_by_label)
        self.texture_combo.configure(values=labels, state="readonly")
        self.texture_var.set(labels[0])
        self.run_button.configure(state="disabled" if self.busy else "normal")
        self.status_var.set(f"{part.label}: {len(labels)} trim texture(s) resolved.")

    def _texture_changed(self) -> None:
        self.run_button.configure(state="disabled" if self.busy else "normal")

    # -- detection -------------------------------------------------------
    def _run(self) -> None:
        binding = self.texture_by_label.get(self.texture_var.get())
        archive, loaded = self.archive, self.loaded
        if binding is None or archive is None or loaded is None:
            return
        try:
            config = self.current_config()
            symmetry_config = self.current_uv_symmetry_config()
        except ValueError as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return

        cached = self.texture_source
        reuse = cached is not None and cached.key == (
            binding.texture_member,
            binding.dae_material,
        )
        self._set_busy(
            True,
            "Detecting…" if reuse else "Extracting texture and rasterising UV islands…",
        )

        previous = self.result.run if reuse and self.result is not None else None

        def job() -> tuple[str, object]:
            source = cached if reuse and cached is not None else build_texture_source(
                archive, loaded, binding
            )
            started = time.perf_counter()
            run = run_detection(source.image, source.uv_mask, config, previous)
            symmetric_uv_islands = analyse_uv_island_symmetry(
                source.uv_mask, symmetry_config
            )
            return "run", RunResult(
                source=source,
                run=run,
                symmetry_config=symmetry_config,
                symmetric_uv_islands=symmetric_uv_islands,
                seconds=time.perf_counter() - started,
            )

        self._start_worker(job)

    def _run_finished(self, result: RunResult) -> None:
        first_load = (
            self.texture_source is None or self.texture_source.key != result.source.key
        )
        self.texture_source = result.source
        self.result = result
        if first_load:
            self._build_pyramid(result.source)
            self.view.initialised = False
        self._build_stage_tabs(result.stages)
        self._update_summary(result)
        if not self.view.initialised:
            self._initial_view()
        else:
            self.invalidate_views()
            self.render_active()
        height, width = result.source.image.shape[:2]
        islands = int(result.source.uv_mask.sum())
        resumed = result.run.resumed_from
        reused = (
            "full run"
            if resumed == 0
            else f"resumed at {result.stages[resumed].title}"
            if resumed < len(result.stages)
            else "nothing to redo"
        )
        self._set_busy(
            False,
            f"{result.source.texture_path.name}  {width}x{height}  "
            f"UV domain {islands / (width * height):.1%}  "
            f"{len(result.symmetric_uv_islands):,} symmetric UV islands  "
            f"{result.source.uv_stats.get('triangles', 0):,} UV triangles from "
            f"{', '.join(result.source.material_symbols)}  "
            f"{reused} in {result.seconds:.2f} s",
        )

    def _update_summary(self, result: RunResult) -> None:
        self.summary.delete(*self.summary.get_children())
        for index, stage in enumerate(result.stages):
            self.summary.insert(
                "",
                "end",
                iid=stage.key,
                text=f"{index + 1}. {stage.title}",
                values=(f"{len(stage.kept):,}", f"{len(stage.rejected):,}" if stage.rejected else ""),
            )
        # Follow whichever tab is open rather than jumping to the last stage:
        # re-running with a tweaked parameter should not move the view.
        active = self._active_canvas()
        showing = self.stage_by_key(active.stage_key) if active else result.stages[-1]
        showing = showing or result.stages[-1]
        self._syncing = True
        try:
            self.summary.selection_set(showing.key)
        finally:
            self._syncing = False
        self._show_stage_detail(showing)

    def _summary_selected(self) -> None:
        selection = self.summary.selection()
        if not selection or self._syncing:
            return
        key = selection[0]
        keys = list(self.stage_canvases)
        tabs = self.notebook.tabs()
        self._syncing = True
        try:
            if key in keys and keys.index(key) < len(tabs):
                self.notebook.select(tabs[keys.index(key)])
        finally:
            self._syncing = False
        self._show_stage_detail(self.stage_by_key(key))

    def _show_stage_detail(self, stage: DetectionStage | None) -> None:
        if stage is None:
            self.detail_var.set("")
            return
        parts = [f"{len(stage.kept):,} kept"]
        if stage.rejected:
            parts.append(f"{len(stage.rejected):,} removed here")
        if stage.adjusted:
            action = (
                "absorbed"
                if stage.key == "overlap_group"
                else "padded"
                if stage.key == "final_padding"
                else "adjusted"
            )
            parts.append(f"{stage.adjusted:,} {action}")
        detail = f"{stage.title}: " + ", ".join(parts)
        if stage.detail:
            detail += f"\n{stage.detail}"
        self.detail_var.set(detail)

    # -- view ------------------------------------------------------------
    def _build_pyramid(self, source: TextureSource) -> None:
        """Cache full and reduced levels so panning a 4k atlas stays responsive.

        Each level holds the plain texture and a pre-masked copy, so toggling
        the overlay is a choice of source image rather than per-frame work.
        """
        rgb = source.image[:, :, ::-1]
        blocked = ~source.uv_mask if UV_BLOCKED_REGION == "outside" else source.uv_mask
        full = Image.fromarray(rgb)
        masked = build_masked_texture(rgb, blocked)
        self._pyramid = [(full, masked, 1.0)]

        longest = max(full.size)
        if longest > PREVIEW_LEVEL_MAX_PX:
            ratio = PREVIEW_LEVEL_MAX_PX / longest
            size = (
                max(int(full.width * ratio), 1),
                max(int(full.height * ratio), 1),
            )
            self._pyramid.append(
                (
                    full.resize(size, Image.BILINEAR),
                    masked.resize(size, Image.BILINEAR),
                    ratio,
                )
            )

    def pyramid_level(self, scale: float) -> tuple[Image.Image, float]:
        """Return the smallest cached level that still exceeds the view scale."""
        chosen = self._pyramid[0]
        for level in self._pyramid[1:]:
            if scale <= level[2]:
                chosen = level
        plain, masked, ratio = chosen
        return (masked if self.show_uv_domain.get() else plain), ratio

    def source_size(self) -> tuple[int, int]:
        if self.result is None:
            return (1, 1)
        height, width = self.result.source.image.shape[:2]
        return (width, height)

    def clamp_view(self, canvas: tk.Canvas) -> None:
        """Keep the texture in view, centring it on whichever axis has slack."""
        width, height = self.source_size()
        view_w = max(canvas.winfo_width(), 1) / max(self.view.scale, 1e-6)
        view_h = max(canvas.winfo_height(), 1) / max(self.view.scale, 1e-6)
        self.view.origin_x = (
            (width - view_w) / 2.0
            if view_w >= width
            else min(max(self.view.origin_x, 0.0), width - view_w)
        )
        self.view.origin_y = (
            (height - view_h) / 2.0
            if view_h >= height
            else min(max(self.view.origin_y, 0.0), height - view_h)
        )
        self.zoom_var.set(f"{self.view.scale * 100:.0f}%")

    def _initial_view(self) -> None:
        """Open a newly loaded texture at 1:1, centred on the atlas."""
        canvas = self._active_canvas()
        if canvas is None or self.result is None:
            return
        width, height = self.source_size()
        self.view.scale = INITIAL_ZOOM
        self.view.origin_x = (width - canvas.canvas.winfo_width() / INITIAL_ZOOM) / 2.0
        self.view.origin_y = (height - canvas.canvas.winfo_height() / INITIAL_ZOOM) / 2.0
        self.view.initialised = True
        self.clamp_view(canvas.canvas)
        self.invalidate_views()
        self.render_active()

    def _fit_view(self) -> None:
        canvas = self._active_canvas()
        if canvas is None or self.result is None:
            return
        width, height = self.source_size()
        self.view.scale = min(
            max(canvas.canvas.winfo_width(), 1) / width,
            max(canvas.canvas.winfo_height(), 1) / height,
        )
        self.view.initialised = True
        self.clamp_view(canvas.canvas)
        self.invalidate_views()
        self.render_active()

    def _actual_size(self) -> None:
        canvas = self._active_canvas()
        if canvas is None:
            return
        self.view.scale = 1.0
        self.clamp_view(canvas.canvas)
        self.invalidate_views()
        self.render_active()

    def _active_canvas(self) -> StageCanvas | None:
        current = self.notebook.select()
        tabs = self.notebook.tabs()
        keys = list(self.stage_canvases)
        if not current or current not in tabs:
            return None
        index = tabs.index(current)
        return self.stage_canvases[keys[index]] if index < len(keys) else None

    def stage_by_key(self, key: str) -> DetectionStage | None:
        if self.result is None:
            return None
        return next((stage for stage in self.result.stages if stage.key == key), None)

    def invalidate_views(self) -> None:
        for view in self.stage_canvases.values():
            view.dirty = True

    def render_active(self) -> None:
        canvas = self._active_canvas()
        if canvas is not None:
            canvas.render()

    def _tab_changed(self) -> None:
        canvas = self._active_canvas()
        if canvas is None:
            return
        if canvas.dirty:
            canvas.render()
        stage = self.stage_by_key(canvas.stage_key)
        if stage is None or self._syncing:
            return
        self._syncing = True
        try:
            self.summary.selection_set(stage.key)
        finally:
            self._syncing = False
        self._show_stage_detail(stage)

    def _redraw(self) -> None:
        self.invalidate_views()
        self.render_active()

    def report_cursor(self, x: int, y: int) -> None:
        width, height = self.source_size()
        if 0 <= x < width and 0 <= y < height:
            self.cursor_var.set(f"x={x}  y={y}")
        else:
            self.cursor_var.set("")

    def _pointer_within(self, widget: tk.Widget, event: tk.Event) -> bool:
        try:
            x, y = widget.winfo_pointerxy()
        except tk.TclError:
            return False
        left, top = widget.winfo_rootx(), widget.winfo_rooty()
        return (
            left <= x < left + widget.winfo_width()
            and top <= y < top + widget.winfo_height()
        )

    # -- lifecycle -------------------------------------------------------
    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy
        self.status_var.set(text)
        self.run_button.configure(
            state="disabled" if busy or not self.texture_by_label else "normal"
        )
        self.part_combo.configure(
            state="disabled" if busy or not self.part_by_label else "readonly"
        )
        self.texture_combo.configure(
            state="disabled" if busy or not self.texture_by_label else "readonly"
        )
        self.configure(cursor="watch" if busy else "")
        self.update_idletasks()

    def _cleanup_archive(self) -> None:
        archive = self.archive
        self.archive = None
        self.loaded = None
        self.texture_source = None
        self.result = None
        self.part_by_label.clear()
        self.texture_by_label.clear()
        if archive is not None:
            shutil.rmtree(archive.workspace, ignore_errors=True)

    def _on_close(self) -> None:
        if self._poll_handle is not None:
            self.after_cancel(self._poll_handle)
            self._poll_handle = None
        self._cleanup_archive()
        self.destroy()


def main() -> None:
    TuningApp().mainloop()


if __name__ == "__main__":
    main()