"""Glyph Lab: the whole corpus on screen at once, detected under one config.

The point of this app, and the reason it is not the old one with a tab added:
a change to detection is judged by what it does across a *variety* of
materials, and the old harness could only show one atlas per vehicle load.
Here every texture is on screen under the same parameters, so a change that
rescues a dashboard legend and destroys a badge sheet says so immediately.

    python -m glyph_lab.app [corpus_dir]

Click a tile to open it.  Inside a tile the wheel zooms about the pointer,
dragging pans, double-click refits, and Escape or a right-click returns to the
sheet.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from glyph_lab.corpus import (
    CorpusTexture,
    UvDomainGeometry,
    domain_boundaries,
    load_corpus,
)
from glyph_lab.detect import (
    FLAT_GROUPED,
    LAST_STOP,
    STAGE_FIRST,
    STAGE_OF,
    STAGES,
    ContrastConfig,
    Detection,
    LabConfig,
    detect,
)
from glyph_lab.gpu_render import GpuRenderUnavailable, GpuViewRenderer
from glyph_lab.normal_detect import (
    NORMAL_GROUPED_FLAT_GROUND,
    NORMAL_STAGE_OF,
    NORMAL_STAGES,
    NormalMapConfig,
    detect_normal,
)
from glyph_lab.normal_map import normal_edge_steepness
from glyph_lab.region_grouping import GroupingConfig
from glyph_lab.render import ViewOptions, compose_view, draw_uv_geometry, fit_rect
from glyph_lab.uv_geometry import (
    DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
    SimpleUvGeometryConfig,
    source_seam_guides,
    wrapped_axis_copies,
)

BACKGROUND = "#141414"
FOREGROUND = "#c8c8c8"
CELL_PAD = 8
LABEL_HEIGHT = 18
# 2026-08-26: measured 16.6 s serial against 10.1 s on four threads over the
# four-texture corpus -- 1.6x, not 4x, because each texture is a loop over its
# own domains and the small ones finish early.  Worth having; not worth
# building a domain-level pool for.  (Threading the *old* pipeline's per-box
# work measured 0.6x: numpy holds the GIL on arrays of a few hundred texels.
# These are whole domains, which is the other end of that trade.)
DETECT_WORKERS = 4
# Sized from the longest field name and the longest counts line, so neither
# the parameter panel nor the breakdown beside it clips what it is showing.
LABEL_WIDTH = 38
COUNTS_WIDTH = 58
# The parameter panel, section by section, in the order the pipeline applies
# them.  Field names are unique across the sections, which is what lets
# `config_vars` stay a flat dict keyed by name.
SECTIONS: tuple[tuple[str, type, str], ...] = (
    ("Local-contrast detector", ContrastConfig, "contrast"),
)


def panel_groups(
    title: str,
    config_type: type,
    stage_of: dict[str, int] = STAGE_OF,
    stages: tuple[str, ...] = STAGES,
) -> list[tuple[str, tuple[str, ...]]]:
    """The panel's headings and the parameters under each, in pipeline order.

    2026-08-28: grouped by the stage that reads a parameter, not by the order
    the dataclass happens to declare it in.  The panel is read while watching a
    stage -- pick one, turn its knobs, look -- and a flat list makes that a
    hunt: the fields for stage 10 sat above those for stage 7 purely because
    they were written first.  `STAGE_OF` already knows the answer and is
    already the thing `_apply_stop` hides by, so the panel now agrees with it.

    Stage 0 is the pipeline control itself, which keeps the section's own
    title and stays at the top.  A config type with no stage mapping at all
    comes back as one group under its title, exactly as before.
    """
    by_stage: dict[int, list[str]] = {}
    for entry in fields(config_type):
        by_stage.setdefault(stage_of.get(entry.name, 0), []).append(entry.name)
    if set(by_stage) <= {0}:
        return [(title, tuple(by_stage.get(0, ())))]
    groups: list[tuple[str, tuple[str, ...]]] = []
    if 0 in by_stage:
        groups.append((title, tuple(by_stage.pop(0))))
    for number in sorted(by_stage):
        heading = (
            stages[number - 1] if 1 <= number <= len(stages)
            else f"stage {number}"
        )
        groups.append((heading, tuple(by_stage[number])))
    return groups
# Friendly names for the overlay toggles; anything without one shows its field
# name, so a new ViewOptions field appears whether or not it is listed here.
# Tuning is a long conversation with the pictures, and losing a session's
# worth of it to closing a window is not acceptable.
#
# 2026-08-27: kept in the repository root rather than under LOCALAPPDATA, so
# that the values in play are visible and editable alongside the code that
# reads them -- this is a tuning sandbox, and "what was it set to when that
# screenshot was taken" is a question that comes up constantly.  Matches the
# `hand_drive_tool_settings.json` convention: machine-local, and git-ignored.
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "glyph_lab_params.json"


def load_settings() -> dict:
    """Whatever was saved last, or nothing.  Never raises.

    The parameter set changes most sessions in this package, so a saved file
    is always read as a set of *suggestions*: a key that no longer exists is
    ignored, one that has appeared since falls back to its default, and one
    that will not parse as its field's type is dropped rather than carried
    into the panel where it would fail validation on every run.
    """
    try:
        stored = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    # 2026-08-29: the first rim-artifact gate also required a component to be
    # solid.  It spared the low-fill, discrete pieces of edge stitching, so
    # the replacement is a 32 px minor-axis limit and the fill knob is gone.
    # Migrate only that short-lived cached form; all other cached values remain
    # the user's values.
    legacy_rim_fill_gate = "rim_artifact_min_fill" in stored
    kept: dict = {}
    for config_type in (
        [t for _title, t, _a in SECTIONS]
        + [GroupingConfig, SimpleUvGeometryConfig, ViewOptions]
    ):
        defaults = config_type()
        for entry in fields(config_type):
            if entry.name not in stored:
                continue
            default = getattr(defaults, entry.name)
            raw = stored[entry.name]
            try:
                kept[entry.name] = (
                    bool(raw) if isinstance(default, bool)
                    else int(raw) if isinstance(default, int)
                    else float(raw)
                )
            except (TypeError, ValueError):
                continue
    normal_defaults = NormalMapConfig()
    for entry in fields(NormalMapConfig):
        key = f"{NORMAL_PARAM_PREFIX}{entry.name}"
        if key not in stored:
            continue
        default = getattr(normal_defaults, entry.name)
        raw = stored[key]
        try:
            kept[key] = (
                bool(raw) if isinstance(default, bool)
                else int(raw) if isinstance(default, int)
                else float(raw)
            )
        except (TypeError, ValueError):
            continue
    if legacy_rim_fill_gate and kept.get("rim_artifact_max_span_px") == 16:
        kept["rim_artifact_max_span_px"] = 32
    # 2026-08-27: a saved stop that names a stage earlier than the code's own
    # default is stale, and taking it at face value hides the stage that was
    # just built.
    #
    # `stop_after_stage` is not an ordinary preference.  Its default states how
    # far the rebuild has reached, so it moves every time a stage is added --
    # and the panel hides the parameters of any stage past the stop.  A file
    # written before the new stage existed therefore switches the new stage's
    # knobs off, and its entry out of the stage picker, on the first run after
    # it lands: the work looks not to have been done.
    #
    # 0 is honoured, because 0 means "run all of it" rather than "stop early",
    # and so does any value at or beyond the default.  Only a stop that
    # predates the pipeline is moved.
    fresh = ContrastConfig().stop_after_stage
    stop = kept.get("stop_after_stage")
    if isinstance(stop, int) and 0 < stop < fresh:
        kept["stop_after_stage"] = fresh
    # 2026-09-01: normal-map stages can be appended independently of the
    # colour pipeline.  As above, 0 keeps its explicit "run all" meaning.
    normal_stop_key = f"{NORMAL_PARAM_PREFIX}stop_after_stage"
    normal_stop = kept.get(normal_stop_key)
    if isinstance(normal_stop, int) and 0 < normal_stop < len(NORMAL_STAGES):
        kept[normal_stop_key] = NormalMapConfig().stop_after_stage
    return kept


def save_settings(values: dict) -> None:
    """Best effort.  A read-only profile is not a reason to fail a redraw."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(values, indent=2, sort_keys=True))
    except (OSError, TypeError, ValueError):
        pass


# What the stage picker shows when it is not showing a stage.
NO_STAGE = "(the response)"

VIEW_LABELS = {
    "show_texels": "Detector texels (amber)",
    "show_regions": "Regions (green)",
    "dim_outside_uv": "Dim outside the UV domain",
    "show_uv_edges": "UV domain edges (blue)",
    "show_mesh_seams": "Mesh seams (violet)",
    "show_uv_geometry": "Simple UV geometry (cyan / orange)",
    "show_sparse": "The sparse field (purple)",
}
SOURCE_COLOUR = "Colour"
SOURCE_NORMAL_EDGE_STEEPNESS = "Normal map (edge steepness)"
SOURCES = (SOURCE_COLOUR, SOURCE_NORMAL_EDGE_STEEPNESS)
NORMAL_PARAM_PREFIX = "normal_"


def is_normal_source(source: str) -> bool:
    return source == SOURCE_NORMAL_EDGE_STEEPNESS


@dataclass(frozen=True, slots=True)
class RenderTile:
    """Everything a background raster job needs for one canvas image."""

    name: str
    texture: CorpusTexture
    fired: np.ndarray | None
    outlines: np.ndarray | None
    sparse: np.ndarray | None
    rect: tuple[float, float, float, float]
    out_size: tuple[int, int]
    view: ViewOptions
    boundaries: np.ndarray | None
    needs_boundaries: bool
    uv_geometry: tuple[UvDomainGeometry, ...]
    x: int
    y: int
    text_x: int
    text_y: int
    caption: str


@dataclass(frozen=True, slots=True)
class RenderTask:
    """A complete paint request, deliberately free of tkinter objects."""

    token: int
    tiles: tuple[RenderTile, ...]


def resolved_geometry(
    texture: CorpusTexture, config: SimpleUvGeometryConfig,
) -> tuple[UvDomainGeometry, ...]:
    """Apply the current guide thresholds without asking a Tk variable."""
    guides: list[UvDomainGeometry] = []
    seen_axes: set[tuple[float, ...]] = set()
    for item in texture.uv_geometry:
        if item.kind == "seam" and item.source_segments:
            for kind, data, score in source_seam_guides(
                item.source_segments, (texture.shape[1], texture.shape[0]), config,
            ):
                if kind == "symmetric":
                    for axis in wrapped_axis_copies(
                        data, (texture.shape[1], texture.shape[0]),
                    ):
                        key = tuple(round(value, 4) for point in axis for value in point)
                        if key in seen_axes:
                            continue
                        seen_axes.add(key)
                        guides.append(UvDomainGeometry(
                            item.label, kind, axis=axis, symmetry_score=score,
                        ))
                elif kind == "parallelogram":
                    guides.append(UvDomainGeometry(item.label, kind, corners=data))
                else:
                    guides.append(UvDomainGeometry(item.label, "parallelogram",
                                                   segments=data))
            continue
        # 2026-08-30: corpora from before raw seam sidecars remain viewable.
        if (item.kind != "symmetric" or item.symmetry_score is None
                or (item.symmetry_score >= config.seam_symmetry_min_score
                    and (item.symmetry_area_px is None
                         or item.symmetry_area_px >= config.seam_symmetry_min_area_px))):
            guides.append(item)
    return tuple(guides)


def render_tile(
    tile: RenderTile, renderer: GpuViewRenderer | None = None,
) -> tuple[RenderTile, Image.Image, np.ndarray | None]:
    """Compose pixels only; tkinter image creation remains on the main thread."""
    boundaries = tile.boundaries
    if tile.needs_boundaries:
        boundaries = domain_boundaries(tile.texture.domains)
    # The GPU composes every source-sized pixel layer.  The vector guides are
    # only a few lines, so drawing them after readback retains acceleration
    # without needing a second OpenGL line-rendering pipeline.
    if renderer is not None:
        image = renderer.compose(
            tile.texture.colour, tile.texture.domains, tile.fired, tile.outlines,
            tile.rect, tile.out_size, tile.view, boundaries, tile.texture.seams,
            tile.sparse, tile.texture.geometry_seams if tile.uv_geometry else None,
        )
        draw_uv_geometry(image, tile.rect, tile.uv_geometry)
        return tile, image, boundaries
    image = compose_view(
        tile.texture.colour, tile.texture.domains, tile.fired, tile.outlines,
        tile.rect, tile.out_size, tile.view, boundaries, tile.texture.seams,
        tile.sparse, tile.uv_geometry,
        tile.texture.geometry_seams if tile.uv_geometry else None,
    )
    return tile, image, boundaries


class GlyphLab(tk.Tk):
    def __init__(self, corpus_dir: Path) -> None:
        super().__init__()
        self.title("BeamXP Glyph Lab")
        self.geometry("1750x950")
        self.configure(background=BACKGROUND)

        self._colour_textures: list[CorpusTexture] = load_corpus(corpus_dir)
        if not self._colour_textures:
            raise SystemExit(f"no textures in {corpus_dir}")
        self._normal_textures: dict[tuple[str, str], CorpusTexture] = {}
        self.textures = list(self._colour_textures)
        self.results: dict[str, Detection] = {}
        self.zoomed: str | None = None
        self.view_origin = (0.0, 0.0)
        self.view_scale = 1.0
        self._drag: tuple[int, int] | None = None
        self._photo: object = None
        self._tiles: list[tuple[int, int, int, int, str]] = []
        self._boundaries: dict[str, np.ndarray] = {}
        self._resolved_geometry: dict[
            tuple[str, SimpleUvGeometryConfig], tuple[UvDomainGeometry, ...]
        ] = {}
        # The last set of view values that parsed; redraw falls back to
        # it rather than raising out of a paint method.
        self.view = ViewOptions()
        self._queue: queue.Queue = queue.Queue()
        self._render_requests: queue.Queue[RenderTask | None] = queue.Queue()
        self._busy = False
        self._rerun = False
        # 2026-08-31: composing a contact sheet over several 4k atlases is
        # still hundreds of milliseconds even after the image-path shortcut.
        # Raster work must not run in Tk's event loop: a paint that stalls the
        # loop also stalls the pan, wheel, and Escape events that would change
        # it.  One pending task replaces any earlier one, so a drag cannot
        # create an obsolete queue of full-atlas renders.
        self._render_busy = False
        self._render_pending: RenderTask | None = None
        self._render_token = 0
        self._render_backend = "initialising GPU renderer"
        self._cursor_busy = False
        self._closing = False
        self._poll_job: str | None = None
        self._panel_shell: tk.Canvas | None = None
        self._render_worker = threading.Thread(
            target=self._render_worker_main,
            name="Glyph Lab GPU renderer",
            daemon=True,
        )
        self._render_worker.start()
        # Which pipeline stage to draw in place of the response, or None for
        # the response itself.  Held here rather than in `ViewOptions` because
        # it changes what detection *records*, not merely how it is painted.
        self.stage: str | None = None
        self._saved = load_settings()

        self._build_widgets()
        self._poll_job = self.after(50, self._poll)
        self.run_detection()

    def destroy(self) -> None:
        """Take the interpreter down with nothing still pointed at it.

        2026-08-27: Tk does not cancel pending ``after`` callbacks on destroy,
        so the poll timer fires once more into a dead interpreter and prints
        ``invalid command name ..._poll``.  ``Variable.__del__`` and
        ``PhotoImage.__del__`` are worse: they call *into* the interpreter
        whenever CPython gets round to collecting them, on whichever thread,
        and a Tcl async handler released off the main loop aborts the process
        with ``Tcl_AsyncDelete`` rather than raising.  One window gets away
        with it; a process that opens several does not, which is how this
        showed up -- as a roughly one-in-three failure of a random test in the
        suite, always at the *next* window's construction.
        """
        self.remember()
        self._closing = True
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:                 # already gone; nothing to do
                pass
            self._poll_job = None
        self._photo = None
        self._render_token += 1
        self._render_pending = None
        self._render_requests.put(None)
        # The wheel handler is bound at application level and closes over the
        # shell, so let go of it before the widgets are torn down.
        self._panel_shell = None
        # Found by type rather than by name: the list of variables has grown
        # every time this app gained a checkbox, and a name missed off it is
        # invisible until the crash lands in some unrelated test.
        for name, value in list(vars(self).items()):
            if isinstance(value, tk.Variable):
                setattr(self, name, None)
        self.config_vars = {}
        self.view_vars = {}
        super().destroy()

    # -- layout -----------------------------------------------------------
    def _build_widgets(self) -> None:
        side = ttk.Frame(self, width=460)
        side.pack(side="right", fill="y")

        # The readouts are pinned and only the controls scroll.  They are the
        # output of every change made above them, and a result you have to
        # scroll back to find is one you stop reading.  Packed first and from
        # the bottom, so they claim their space before the scrolling area takes
        # what is left -- the same ordering trap the status line already had.
        self.status = ttk.Label(side, text="", anchor="w", wraplength=300)
        self.status.pack(side="bottom", fill="x", padx=8, pady=(6, 8))
        self.counts = tk.Text(
            side, width=COUNTS_WIDTH, height=14, background=BACKGROUND,
            foreground=FOREGROUND, borderwidth=0,
        )
        self.counts.pack(side="bottom", fill="x", padx=8)
        ttk.Separator(side).pack(side="bottom", fill="x", padx=8, pady=8)

        # 2026-08-27: what each stage costs, over the whole corpus.  Pinned
        # beside the counts rather than hidden behind the stage picker: the
        # price of a step is half of judging it, and a number you have to turn
        # something on to see is a number nobody looks at.
        self.costs = tk.Text(
            side, width=COUNTS_WIDTH, height=11, background=BACKGROUND,
            foreground=FOREGROUND, borderwidth=0,
        )
        self.costs.pack(side="bottom", fill="x", padx=8)

        # A frame cannot scroll on its own, so the controls live in a canvas
        # and the canvas scrolls.  2026-08-27: they overflowed a 950-tall
        # window once the filters grew a second stage, and the fields at the
        # bottom were not merely awkward to reach but invisible.
        shell = tk.Canvas(side, background=BACKGROUND, highlightthickness=0)
        bar = ttk.Scrollbar(side, orient="vertical", command=shell.yview)
        shell.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        shell.pack(side="left", fill="both", expand=True)
        self._panel_shell = shell

        panel = ttk.Frame(shell, padding=8)
        window = shell.create_window((0, 0), window=panel, anchor="nw")

        def _panel_resized(_event: object = None) -> None:
            # The scrollregion has to follow the content, and the canvas has to
            # ask for the width the content needs -- a canvas defaults to
            # something far narrower and would clip the entry boxes instead.
            shell.configure(scrollregion=shell.bbox("all"),
                            width=panel.winfo_reqwidth())

        def _shell_resized(event: tk.Event) -> None:
            # Stretch the inner frame to the canvas, so `fill="x"` rows inside
            # it fill the panel rather than their own requested width.
            shell.itemconfigure(window, width=event.width)

        panel.bind("<Configure>", _panel_resized)
        shell.bind("<Configure>", _shell_resized)
        # Bound on the application rather than on the canvas.  Enter/Leave is
        # the usual way to arm this, and it does not work here: Tk sends the
        # parent a Leave when the pointer moves onto a child, so it would
        # disarm the moment the pointer touched one of the entry boxes -- which
        # is everywhere the panel actually is.  The handler decides for itself
        # from the pointer instead.
        self.bind_all("<MouseWheel>", self._on_panel_wheel)

        self.config_vars: dict[str, tk.StringVar] = {}
        self.geometry_vars: dict[str, tk.StringVar] = {}
        # 2026-08-27: kept so `_apply_stop` can take a parameter off the panel
        # when the stage that reads it is not running.  A box that still takes
        # a value for a stage that no longer happens is worse than no box.
        self._config_rows: dict[str, tk.Widget] = {}
        self._section_rows: list[tuple[tk.Widget, tuple[str, ...]]] = []
        self._colour_section_rows: list[tuple[tk.Widget, tuple[str, ...]]] = []
        self._colour_detector_widgets: set[tk.Widget] = set()
        self._normal_detector_widgets: set[tk.Widget] = set()
        written = 0
        for title, config_type, _attribute in SECTIONS:
            defaults = config_type()
            for group, names in panel_groups(title, config_type):
                heading = ttk.Label(panel, text=group)
                heading.pack(anchor="w", pady=(8 if written else 0, 2))
                self._section_rows.append((heading, names))
                self._colour_section_rows.append((heading, names))
                self._colour_detector_widgets.add(heading)
                written += 1
                for name in names:
                    default = self._saved.get(name, getattr(defaults, name))
                    row = ttk.Frame(panel)
                    row.pack(fill="x", pady=1)
                    # master=self, not the default root.  An unmastered
                    # Variable binds to tkinter's global `_default_root`,
                    # which is whichever Tk was created first in the process
                    # -- so the panel would drive a window it does not belong
                    # to, and be finalised against an interpreter that may
                    # already be gone.
                    var: tk.Variable
                    if isinstance(default, bool):
                        var = tk.BooleanVar(master=self, value=default)
                        ttk.Checkbutton(
                            row, text=name, variable=var,
                            command=self._on_colour_config_change,
                        ).pack(side="left")
                    else:
                        # Wide enough for the longest field name there is.  At
                        # 18 the panel silently truncated
                        # `uv_edge_margin_percent` to `uv_edge_margin_per`,
                        # which reads as a parameter that is missing rather
                        # than one that is merely clipped.
                        ttk.Label(row, text=name, width=LABEL_WIDTH).pack(
                            side="left"
                        )
                        var = tk.StringVar(master=self, value=str(default))
                        box = ttk.Entry(row, textvariable=var, width=9)
                        box.pack(side="left")
                        box.bind(
                            "<Return>",
                            lambda _event: self._on_colour_config_change(),
                        )
                    self.config_vars[name] = var
                    self._config_rows[name] = row
                    self._colour_detector_widgets.add(row)

        normal_separator = ttk.Separator(panel)
        normal_separator.pack(fill="x", pady=8)
        self._normal_detector_widgets.add(normal_separator)
        self.normal_config_vars: dict[str, tk.Variable] = {}
        self._normal_config_rows: dict[str, tk.Widget] = {}
        normal_defaults = NormalMapConfig()
        for group, names in panel_groups(
            "Normal-map detector", NormalMapConfig, NORMAL_STAGE_OF, NORMAL_STAGES,
        ):
            heading = ttk.Label(panel, text=f"Normal map — {group}")
            heading.pack(anchor="w", pady=(8 if written else 0, 2))
            self._section_rows.append((heading, names))
            self._normal_detector_widgets.add(heading)
            written += 1
            for name in names:
                default = self._saved.get(
                    f"{NORMAL_PARAM_PREFIX}{name}",
                    getattr(normal_defaults, name),
                )
                row = ttk.Frame(panel)
                row.pack(fill="x", pady=1)
                var: tk.Variable
                if isinstance(default, bool):
                    var = tk.BooleanVar(master=self, value=default)
                    ttk.Checkbutton(
                        row, text=name, variable=var,
                        command=self._on_normal_config_change,
                    ).pack(side="left")
                else:
                    ttk.Label(row, text=name, width=LABEL_WIDTH).pack(side="left")
                    var = tk.StringVar(master=self, value=str(default))
                    box = ttk.Entry(row, textvariable=var, width=9)
                    box.pack(side="left")
                    box.bind(
                        "<Return>",
                        lambda _event: self._on_normal_config_change(),
                    )
                self.normal_config_vars[name] = var
                self._normal_config_rows[name] = row
                self._normal_detector_widgets.add(row)

        # 2026-09-01: both detectors apply this final pass, so it remains visible when the
        # source-specific controls above are switched out.
        grouping_separator = ttk.Separator(panel)
        grouping_separator.pack(fill="x", pady=8)
        grouping_heading = ttk.Label(panel, text="Grouping across flat ground")
        grouping_heading.pack(
            anchor="w", pady=(0, 2),
        )
        self._grouping_widgets: set[tk.Widget] = {
            grouping_separator, grouping_heading,
        }
        self.grouping_vars: dict[str, tk.Variable] = {}
        grouping_defaults = GroupingConfig()
        for entry in fields(GroupingConfig):
            name = entry.name
            default = self._saved.get(name, getattr(grouping_defaults, name))
            row = ttk.Frame(panel)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=name, width=LABEL_WIDTH).pack(side="left")
            var = tk.StringVar(master=self, value=str(default))
            box = ttk.Entry(row, textvariable=var, width=9)
            box.pack(side="left")
            box.bind("<Return>", lambda _event: self._on_grouping_config_change())
            self.grouping_vars[name] = var
            self._grouping_widgets.add(row)

        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="Simple UV geometry").pack(anchor="w", pady=(0, 2))
        for entry in fields(SimpleUvGeometryConfig):
            default = self._saved.get(
                entry.name, getattr(DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG, entry.name),
            )
            row = ttk.Frame(panel)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=entry.name, width=LABEL_WIDTH).pack(side="left")
            var = tk.StringVar(master=self, value=str(default))
            box = ttk.Entry(row, textvariable=var, width=9)
            box.pack(side="left")
            box.bind("<Return>", lambda _event: self._on_geometry_change())
            self.geometry_vars[entry.name] = var

        ttk.Button(panel, text="Run detection", command=self.run_detection).pack(
            fill="x", pady=(8, 2)
        )
        ttk.Separator(panel).pack(fill="x", pady=8)

        # Generated from ViewOptions for the same reason the sections above are
        # generated from the config dataclasses: a field added without a widget
        # is a parameter nobody can reach, and hand-listing them is how that
        # keeps happening.
        self.view_vars: dict[str, tk.Variable] = {}
        view_defaults = ViewOptions()
        for entry in fields(ViewOptions):
            default = self._saved.get(entry.name, getattr(view_defaults, entry.name))
            var: tk.Variable
            if isinstance(default, bool):
                var = tk.BooleanVar(master=self, value=default)
                ttk.Checkbutton(
                    panel, text=VIEW_LABELS.get(entry.name, entry.name),
                    variable=var, command=self._on_view_change,
                ).pack(anchor="w")
            else:
                row = ttk.Frame(panel)
                row.pack(fill="x", pady=1)
                ttk.Label(row, text=entry.name, width=LABEL_WIDTH).pack(
                    side="left"
                )
                var = tk.StringVar(master=self, value=str(default))
                box = ttk.Entry(row, textvariable=var, width=9)
                box.pack(side="left")
                box.bind("<Return>", lambda _event: self._on_view_change())
            self.view_vars[entry.name] = var
            # Named aliases, so the overlays can still be reached by name.
            setattr(self, entry.name, var)
        self.dim_outside = self.view_vars["dim_outside_uv"]

        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="Detect and show").pack(anchor="w", pady=(0, 2))
        self.source_var = tk.StringVar(master=self, value=SOURCE_COLOUR)
        self._source_box = ttk.Combobox(
            panel, textvariable=self.source_var, state="readonly",
            values=SOURCES, width=LABEL_WIDTH + 6,
        )
        self._source_box.pack(fill="x", pady=(2, 0))
        self._source_box.bind("<<ComboboxSelected>>", self._on_source_change)

        # Stage picker.  A mark that arrives whole and leaves in pieces should
        # be watchable losing its shape, rather than inferred about from the
        # two ends -- which is how the last several arguments about a run of
        # seat stitching were conducted, one throwaway script at a time.
        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="Show stage in place of the response").pack(anchor="w")
        self.stage_var = tk.StringVar(master=self, value=NO_STAGE)
        self._stage_box = ttk.Combobox(
            panel, textvariable=self.stage_var, state="readonly",
            values=(NO_STAGE, *STAGES), width=LABEL_WIDTH + 6,
        )
        self._stage_box.pack(fill="x", pady=(2, 0))
        self._stage_box.bind("<<ComboboxSelected>>", self._on_stage_change)

        # The panel's own running order, captured once it is complete.  Hiding
        # a row is `pack_forget`, and showing it again has to put it back where
        # it was rather than at the bottom -- so the whole column is re-packed
        # from this list, in order, skipping what is hidden.
        self._panel_layout = [
            (child, child.pack_info()) for child in panel.pack_slaves()
        ]
        self._apply_stop()

        self.canvas = tk.Canvas(self, background=BACKGROUND, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        # Three ways back, because losing the corpus behind a texture you
        # cannot find your way out of is the one navigation failure that
        # matters here.
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.bind("<Escape>", lambda _event: self._show_sheet())

    def _on_panel_wheel(self, event: tk.Event) -> None:
        """Scroll the controls, but only with the pointer over them.

        This is bound at application level, so it fires for every wheel turn
        anywhere in the window -- including over the image, where the canvas's
        own binding is busy zooming.  Hence the containment test: work out what
        is under the pointer and climb to see whether the shell is an ancestor.

        It also declines when everything already fits, so that a wheel turn
        over a short panel does nothing rather than scrolling the content out
        of sight and leaving a blank column.
        """
        if self._closing:
            return
        shell = self._panel_shell
        if shell is None:
            return
        try:
            node = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            # 2026-09-01: Windows' ttk combobox popup is a native child called
            # `popdown`, not a widget Tkinter can resolve through
            # `nametowidget`.  A wheel event while it is open therefore has no
            # panel owner to scroll.
            return
        while node is not None and node is not shell:
            node = getattr(node, "master", None)
        if node is not shell:
            return
        box = shell.bbox("all")
        if box is None or box[3] - box[1] <= shell.winfo_height():
            return
        shell.yview_scroll(-int(event.delta / 120), "units")

    def _refresh_busy_cursor(self) -> None:
        """Make in-flight detection or rendering visible without blocking input."""
        busy = self._busy or self._render_busy or self._render_pending is not None
        if busy == self._cursor_busy:
            return
        self._cursor_busy = busy
        cursor = "wait" if busy else ""
        # A cursor set only on the toplevel is not inherited consistently by
        # ttk controls on Windows, where the slow work is most noticeable.
        pending = [self]
        while pending:
            widget = pending.pop()
            try:
                widget.configure(cursor=cursor)
                pending.extend(widget.winfo_children())
            except tk.TclError:                 # a closing child needs no cursor
                continue

    # -- settings ---------------------------------------------------------
    def remember(self) -> None:
        """Write the panel back to disk, so a session survives the window.

        Saved as typed rather than as parsed values: a half-finished number in
        a box is still what the user last wanted there, and validation happens
        on the way back in.
        """
        values: dict = {}
        for name, var in (
            list(self.config_vars.items())
            + list(self.grouping_vars.items())
            + list(self.geometry_vars.items())
            + list(self.view_vars.items())
        ):
            try:
                values[name] = var.get()
            except tk.TclError:                 # the window is going away
                return
        for name, var in self.normal_config_vars.items():
            try:
                values[f"{NORMAL_PARAM_PREFIX}{name}"] = var.get()
            except tk.TclError:
                return
        save_settings(values)

    def _on_view_change(self) -> None:
        self.redraw()
        self.remember()

    def _on_geometry_change(self) -> None:
        self.redraw()
        self.remember()

    def _on_colour_config_change(self) -> None:
        """Keep colour edits from re-running the normal-map detector."""
        if not is_normal_source(self.source_var.get()):
            self.run_detection()
        else:
            self.remember()

    def _on_normal_config_change(self) -> None:
        """Keep normal-map experiments out of the colour detector."""
        if is_normal_source(self.source_var.get()):
            self.run_detection()
        else:
            self.remember()

    def _on_grouping_config_change(self) -> None:
        """The final shared pass is live for either map source."""
        self.run_detection()

    def _normal_texture(
        self, texture: CorpusTexture, source: str,
    ) -> CorpusTexture:
        """Build each normal-derived source once, when it is selected."""
        key = ("edge_steepness", texture.name)
        cached = self._normal_textures.get(key)
        if cached is None:
            cached = normal_edge_steepness(texture)
            self._normal_textures[key] = cached
        return cached

    def _on_source_change(self, _event: object = None) -> None:
        """Switch the corpus view and detector input together."""
        source = self.source_var.get()
        if is_normal_source(source):
            self.textures = [
                self._normal_texture(texture, source)
                for texture in self._colour_textures
                if texture.normal is not None
            ]
        else:
            self.textures = list(self._colour_textures)
        self.results = {}
        self.zoomed = None
        self._drag = None
        self.run_detection()
        self.redraw()

    # -- detection --------------------------------------------------------
    def current_config(self) -> LabConfig:
        sections: dict[str, object] = {}
        for _title, config_type, attribute in SECTIONS:
            defaults = config_type()
            overrides: dict[str, object] = {}
            for entry in fields(config_type):
                raw = self.config_vars[entry.name].get()
                default = getattr(defaults, entry.name)
                # bool before int: bool *is* an int in Python, so testing the
                # other way round would send a checkbox through int().
                if isinstance(default, bool):
                    overrides[entry.name] = bool(raw)
                elif isinstance(default, int):
                    overrides[entry.name] = int(raw)
                else:
                    overrides[entry.name] = float(raw)
            sections[attribute] = replace(defaults, **overrides)
        return LabConfig(**sections, grouping=self.current_grouping_config())

    def current_grouping_config(self) -> GroupingConfig:
        """Read the final shared controls without either detector owning them."""
        defaults = GroupingConfig()
        overrides = {
            entry.name: int(self.grouping_vars[entry.name].get())
            for entry in fields(GroupingConfig)
        }
        return replace(defaults, **overrides)

    def current_normal_config(self) -> NormalMapConfig:
        """The normal-map parameter copy, with no colour settings mixed in."""
        defaults = NormalMapConfig()
        overrides: dict[str, object] = {}
        for entry in fields(NormalMapConfig):
            raw = self.normal_config_vars[entry.name].get()
            default = getattr(defaults, entry.name)
            if isinstance(default, bool):
                overrides[entry.name] = bool(raw)
            elif isinstance(default, int):
                overrides[entry.name] = int(raw)
            else:
                overrides[entry.name] = float(raw)
        return replace(defaults, **overrides)

    def _on_stage_change(self, _event: object = None) -> None:
        """Pick a stage.  Turning capture on or off needs a fresh run.

        Switching between two captured stages does not -- they were all
        recorded by the same pass -- so that redraws instantly.
        """
        chosen = self.stage_var.get()
        wanted = None if chosen == NO_STAGE else chosen
        capturing = self.stage is not None
        self.stage = wanted
        if (wanted is not None) != capturing:
            self.run_detection()
        else:
            self.redraw()

    def active_stages(self) -> tuple[str, ...]:
        """Stage labels belonging to the detector currently on screen."""
        return (
            NORMAL_STAGES if is_normal_source(self.source_var.get())
            else STAGES
        )

    def stage_layer(self, found: Detection | None) -> np.ndarray | None:
        """What to draw as the response layer: a stage, or the response."""
        if found is None:
            return None
        if self.stage is None:
            return found.fired
        return found.stage_mask(self.stage)

    def stage_overlays(self, found: Detection | None) -> dict[str, object]:
        """The other layers, as they stood at the selected stage.

        An overlay whose stage has not happened yet comes back as None and is
        drawn as nothing.  The alternative -- leaving the finished article on
        screen under an early stage's texels -- reads as though the pipeline
        had already decided, which is exactly the confusion a stage viewer is
        supposed to remove.
        """
        if found is None:
            return dict.fromkeys(("outlines",))
        if self.stage is None:
            return {"outlines": found.outlines}
        if is_normal_source(self.source_var.get()):
            drawn = found.stage_outline(self.stage)
            return {"outlines": found.outlines if drawn is None else drawn}
        if STAGES.index(self.stage) < STAGE_FIRST["outlines"]:
            return {"outlines": None}
        # 2026-08-29: the stage's own paths, falling back to the finished ones
        # only where a stage never recorded any.  Before this the finished
        # outlines were drawn over every stage, so a mark a middle stage threw
        # away still had a green box round it at that stage, and there was no
        # way to see which stage had dropped it.
        drawn = found.stage_outline(self.stage)
        return {"outlines": found.outlines if drawn is None else drawn}

    def stage_count(self, found: Detection | None) -> int:
        """How many regions stood at the selected stage, not at the end.

        A stage that never recorded a count is one that does not deal in
        regions at all -- the legacy tail -- so it falls back to the finished
        tally rather than claiming zero.
        """
        if found is None:
            return 0
        if self.stage is None:
            return len(found.regions)
        return found.stage_counts.get(self.stage, len(found.regions))

    def run_detection(self) -> None:
        # Validate before the busy check, or a bad value typed during a run is
        # swallowed in silence -- and since the window starts a detection as it
        # opens, during a run is exactly when you are typing.
        try:
            normal_source = is_normal_source(self.source_var.get())
            grouping = self.current_grouping_config()
            normal_config = self.current_normal_config()
            config = (
                normal_config
                if normal_source else self.current_config()
            )
        except ValueError as exc:
            self.status.configure(text=f"bad parameter: {exc}")
            return
        self._apply_stop()
        if self._busy:
            # Detection over four 4k atlases takes seconds, which is long
            # enough to change your mind in.  Take the newest parameters once
            # the run in flight lands rather than dropping the request.
            self._rerun = True
            self.status.configure(text="detecting... (a re-run is queued)")
            return
        self._busy = True
        self._rerun = False
        self._refresh_busy_cursor()
        self.remember()
        self.status.configure(text="detecting...")

        # Bound as locals rather than reached for through `self`: a worker
        # holding the window keeps the widget tree alive past `destroy`, and
        # Tk objects finalised on a thread that is not the main loop take the
        # interpreter down with "async handler deleted by the wrong thread".
        # The thread needs the textures and somewhere to put the answer, and
        # touches no Tk at all.
        textures = list(self.textures)
        results_queue = self._queue
        capture = self.stage is not None
        colour_sources = {
            texture.name: texture.colour for texture in self._colour_textures
        }

        def job() -> None:
            started = time.perf_counter()
            try:
                with ThreadPoolExecutor(max_workers=DETECT_WORKERS) as pool:
                    if normal_source:
                        found = list(pool.map(
                            lambda texture: (
                                texture.name,
                                detect_normal(
                                    texture, config, capture, grouping,
                                    colour_sources[texture.name],
                                ),
                            ), textures,
                        ))
                    else:
                        found = list(pool.map(
                            lambda texture: (
                                texture.name,
                                detect(
                                    texture, config, capture,
                                    normal_config=normal_config,
                                ),
                            ), textures,
                        ))
                results_queue.put(
                    ("done", dict(found), time.perf_counter() - started)
                )
            except Exception:
                results_queue.put(("error", traceback.format_exc(), 0.0))

        threading.Thread(target=job, daemon=True).start()

    def _poll(self) -> None:
        if self._closing:
            return
        try:
            while True:
                kind, payload, seconds = self._queue.get_nowait()
                if kind == "rendered":
                    task, rendered = payload
                    self._finish_render(task, rendered)
                    continue
                if kind == "render_error":
                    task, error = payload
                    self._finish_render(task, ())
                    if task.token == self._render_token:
                        self.status.configure(text="render failed; see console")
                        print(error, file=sys.stderr)
                    continue
                if kind == "render_backend":
                    self._render_backend = payload
                    renderer_name = (
                        "GPU compositor" if payload.startswith("GPU:")
                        else "CPU compositor"
                    )
                    self.title(f"BeamXP Glyph Lab — {renderer_name}")
                    continue

                self._busy = False
                if kind == "error":
                    # A paint method has nowhere to put an exception, so
                    # detection reports its own rather than dying silently.
                    self.status.configure(text="detection failed; see console")
                    print(payload, file=sys.stderr)
                    continue
                self.results = payload
                total = sum(self.stage_count(found) for found in payload.values())
                texels = sum(found.kept_texels for found in payload.values())
                # 2026-08-27: say so when the pipeline is cut short.  With
                # `stop_after_stage` set there are no regions by construction,
                # and "0 regions" with no explanation reads as a broken
                # parameter rather than as the switch being on.
                stop = self.stage_stop()
                stages = self.active_stages()
                ending = (
                    f" -- stopped after {stages[stop - 1]}"
                    if 1 <= stop <= len(stages) else ""
                )
                self.status.configure(
                    text=f"{total} regions over {len(payload)} textures "
                         f"in {seconds:.1f}s ({texels:,} texels)"
                         f"{ending}"
                )
                self._write_counts()
                self._write_costs()
                self.redraw()
                if self._rerun:
                    self.run_detection()
        except queue.Empty:
            pass
        self._refresh_busy_cursor()
        self._poll_job = self.after(50, self._poll)

    def _apply_stop(self) -> None:
        """Show only the parameters and stages a truncated pipeline reaches.

        Called from `run_detection`, which is the one path every change to the
        panel goes through -- typing a value and pressing Return, and the
        button, both land there.

        The stage picker is trimmed the same way and for the same reason: the
        stages past the stop are not captured, so offering them means offering
        a blank picture with nothing to say why.
        """
        stop = self.stage_stop()
        if not 1 <= stop <= LAST_STOP:
            stop = 0

        normal_source = is_normal_source(self.source_var.get())
        stage_of = NORMAL_STAGE_OF if normal_source else STAGE_OF

        def dead(name: str) -> bool:
            return bool(stop) and stage_of.get(name, 0) > stop
        hidden = set(
            self._colour_detector_widgets
            if normal_source else self._normal_detector_widgets
        )
        grouping_stage = (
            NORMAL_STAGES.index(NORMAL_GROUPED_FLAT_GROUND) + 1 if normal_source
            else STAGES.index(FLAT_GROUPED) + 1
        )
        if stop and grouping_stage > stop:
            hidden.update(self._grouping_widgets)
        rows = self._normal_config_rows if normal_source else self._config_rows
        hidden.update(row for name, row in rows.items() if dead(name))
        # A heading with nothing left under it goes too.  Only inspect the
        # active detector: the other detector's entire panel is source-hidden.
        headings = (
            self._section_rows
            if normal_source else self._colour_section_rows
        )
        hidden.update(
            heading for heading, names in headings
            if all(dead(name) for name in names)
        )
        for child, _info in self._panel_layout:
            child.pack_forget()
        for child, info in self._panel_layout:
            if child not in hidden:
                child.pack(info)

        stages = self.active_stages()
        offered = stages[:stop] if stop else stages
        self._stage_box.configure(values=(NO_STAGE, *offered))
        if self.stage is not None and self.stage not in offered:
            # Set directly rather than through `_on_stage_change`: that would
            # call back into `run_detection`, which is what called this.
            self.stage_var.set(NO_STAGE)
            self.stage = None

    def stage_stop(self) -> int:
        """The stage the panel says the pipeline ends on, or 0 for all of it."""
        try:
            variables = (self.normal_config_vars if is_normal_source(
                self.source_var.get()
            ) else self.config_vars)
            return int(float(variables["stop_after_stage"].get()))
        except (KeyError, TypeError, ValueError):
            return 0

    def _write_costs(self) -> None:
        """Stage cost across the corpus, in pipeline order.

        Summed over textures rather than shown per texture: the question is
        what a *step* costs, and one texture's domains are too small a sample
        to answer it -- scintilla_badges finishes in a fraction of the time
        etk800 takes and would drown the reading it contributes.
        """
        self.costs.delete("1.0", "end")
        totals: dict[str, float] = {}
        for found in self.results.values():
            for name, seconds in found.stage_times.items():
                totals[name] = totals.get(name, 0.0) + seconds
        if not totals:
            self.costs.insert("end", "stage cost\n   -\n")
            return
        whole = sum(totals.values())
        self.costs.insert(
            "end",
            f"stage cost, {len(self.results)} textures, {whole:.2f}s\n",
        )
        for name in self.active_stages():
            seconds = totals.get(name)
            if seconds is None:
                continue
            self.costs.insert(
                "end",
                f" {name:<32}{seconds:>6.2f}s{seconds / whole:>5.0%}\n",
            )

    def _write_counts(self) -> None:
        self.counts.delete("1.0", "end")
        for texture in self.textures:
            found = self.results.get(texture.name)
            if found is None:
                self.counts.insert("end", f"{texture.name}\n   -\n")
                continue
            sizes = sorted(int(region.texels) for region in found.regions)
            middle = sizes[len(sizes) // 2] if sizes else 0
            self.counts.insert(
                "end",
                f"{texture.name}\n"
                f"   {self.stage_count(found)} regions,"
                f" {found.kept_texels:,} texels\n"
                f"   median {middle:,}, largest {sizes[-1] if sizes else 0:,}"
                f" over {texture.domain_count} islands\n",
            )

    # -- drawing ----------------------------------------------------------
    def view_options(self) -> ViewOptions:
        # Not `_options`: tkinter's Widget._options(cnf, kw) is called by
        # `configure`, and a subclass that shadows it breaks the window before
        # it opens.  Nothing outside a running Tk would notice.
        defaults = ViewOptions()
        values: dict[str, object] = {}
        for entry in fields(ViewOptions):
            default = getattr(defaults, entry.name)
            raw = self.view_vars[entry.name].get()
            if isinstance(default, bool):
                values[entry.name] = bool(raw)
            elif isinstance(default, int):
                values[entry.name] = int(raw)
            else:
                values[entry.name] = float(raw)
        return replace(defaults, **values)

    def geometry_config(self) -> SimpleUvGeometryConfig:
        defaults = DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG
        values = {
            entry.name: (
                int(self.geometry_vars[entry.name].get())
                if isinstance(getattr(defaults, entry.name), int)
                else float(self.geometry_vars[entry.name].get())
            )
            for entry in fields(SimpleUvGeometryConfig)
        }
        return replace(DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG, **values)

    def boundaries_for(self, texture: CorpusTexture) -> np.ndarray:
        """The texture's UV edges, derived once and kept.

        They come from the DAE, not from detection, so they survive every
        parameter change and there is no reason to recompute them per frame.
        """
        cached = self._boundaries.get(texture.name)
        if cached is None:
            cached = domain_boundaries(texture.domains)
            self._boundaries[texture.name] = cached
        return cached

    def geometry_seams_for(self, texture: CorpusTexture) -> np.ndarray | None:
        """Mesh discontinuities that give the source-UV frames their context."""
        return texture.geometry_seams if texture.uv_geometry else None

    def geometry_for(
        self,
        texture: CorpusTexture,
        config: SimpleUvGeometryConfig | None = None,
    ) -> tuple[UvDomainGeometry, ...]:
        """Draw the individually preserved seam-region guides from the sidecar."""
        config = self.geometry_config() if config is None else config
        key = (texture.name, config)
        cached = self._resolved_geometry.get(key)
        if cached is None:
            cached = resolved_geometry(texture, config)
            self._resolved_geometry[key] = cached
        return cached

    def redraw(self) -> None:
        # A view value is typed by hand, so it can be nonsense, and this is a
        # paint method -- an exception here has nowhere to go and blanks the
        # canvas.  Report it and keep the last good picture instead.
        try:
            self.view = self.view_options()
            geometry_config = (
                self.geometry_config() if self.view.show_uv_geometry else None
            )
        except ValueError as exc:
            self.status.configure(text=f"bad view value: {exc}")
            return
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if self.zoomed is None:
            task = self._sheet_task(width, height, geometry_config)
        else:
            task = self._zoom_task(width, height, geometry_config)
        self._request_render(task)

    def _sheet_task(
        self, width: int, height: int,
        geometry_config: SimpleUvGeometryConfig | None,
    ) -> RenderTask:
        count = len(self.textures)
        columns = max(1, min(count, int(np.ceil(np.sqrt(count)))))
        rows = int(np.ceil(count / columns))
        cell_w = max((width - CELL_PAD) // columns - CELL_PAD, 32)
        cell_h = max(
            (height - CELL_PAD) // rows - CELL_PAD - LABEL_HEIGHT, 32
        )
        tiles: list[RenderTile] = []
        self._tiles = []
        for index, texture in enumerate(self.textures):
            found = self.results.get(texture.name)
            layers = self.stage_overlays(found)
            x = CELL_PAD + (index % columns) * (cell_w + CELL_PAD)
            y = CELL_PAD + (index // columns) * (cell_h + CELL_PAD + LABEL_HEIGHT)
            self._tiles.append((x, y, cell_w, cell_h, texture.name))
            boundaries = self._boundaries.get(texture.name)
            geometry = (
                self.geometry_for(texture, geometry_config)
                if geometry_config is not None else ()
            )
            tiles.append(RenderTile(
                texture.name, texture, self.stage_layer(found), layers["outlines"],
                found.sparse if found else None,
                fit_rect(texture.shape, (cell_w, cell_h)), (cell_w, cell_h),
                self.view, boundaries,
                self.view.show_uv_edges and boundaries is None, geometry,
                x, y, x, y + cell_h + 2,
                f"{texture.name}   {self.stage_count(found)} regions",
            ))
        return RenderTask(self._render_token + 1, tuple(tiles))

    def _zoom_task(
        self, width: int, height: int,
        geometry_config: SimpleUvGeometryConfig | None,
    ) -> RenderTask:
        texture = next(t for t in self.textures if t.name == self.zoomed)
        found = self.results.get(texture.name)
        origin_x, origin_y = self.view_origin
        rect = (
            origin_x, origin_y,
            origin_x + width / self.view_scale,
            origin_y + height / self.view_scale,
        )
        layers = self.stage_overlays(found)
        boundaries = self._boundaries.get(texture.name)
        geometry = (
            self.geometry_for(texture, geometry_config)
            if geometry_config is not None else ()
        )
        return RenderTask(self._render_token + 1, (RenderTile(
            texture.name, texture, self.stage_layer(found), layers["outlines"],
            found.sparse if found else None, rect, (width, height), self.view,
            boundaries, self.view.show_uv_edges and boundaries is None,
            geometry, 0, 0, 8, 8,
            f"{texture.name}   {self.view_scale * 100:.0f}%   "
            f"{self.stage_count(found)} regions   "
            + (f"[{self.stage}]   " if self.stage else "")
            + "(Esc or right-click for the sheet, double-click to refit)",
        ),))

    def _request_render(self, task: RenderTask) -> None:
        self._render_token += 1
        task = replace(task, token=self._render_token)
        self._render_pending = task
        if not self._render_busy:
            self._start_render()

    def _start_render(self) -> None:
        task = self._render_pending
        if task is None or self._closing:
            return
        self._render_pending = None
        self._render_busy = True
        self._refresh_busy_cursor()
        self._render_requests.put(task)

    def _render_worker_main(self) -> None:
        """Own one GL context for the window's whole rendering lifetime."""
        renderer: GpuViewRenderer | None = None
        task: RenderTask | None = None
        try:
            try:
                renderer = GpuViewRenderer()
                self._queue.put(("render_backend", f"GPU: {renderer.renderer}", 0.0))
            except GpuRenderUnavailable:
                self._queue.put(("render_backend", "CPU fallback", 0.0))
            while True:
                task = self._render_requests.get()
                if task is None:
                    return
                try:
                    rendered = tuple(render_tile(tile, renderer) for tile in task.tiles)
                    if renderer is not None:
                        renderer.retain_masks(tuple(
                            mask
                            for tile, _image, boundaries in rendered
                            for mask in (
                                tile.fired, tile.outlines, tile.sparse,
                                boundaries, tile.texture.seams,
                                tile.texture.geometry_seams,
                            )
                        ))
                except Exception:
                    if renderer is None:
                        raise
                    # A driver reset must not take Glyph Lab's view down with
                    # it.  Discard the suspect context and redo this frame on
                    # the proven CPU path.
                    renderer.close()
                    renderer = None
                    self._queue.put(("render_backend", "CPU fallback", 0.0))
                    rendered = tuple(render_tile(tile) for tile in task.tiles)
                self._queue.put(("rendered", (task, rendered), 0.0))
        except Exception:
            if task is not None:
                self._queue.put((
                    "render_error", (task, traceback.format_exc()), 0.0,
                ))
        finally:
            if renderer is not None:
                renderer.close()

    def _finish_render(
        self,
        task: RenderTask,
        rendered: tuple[tuple[RenderTile, Image.Image, np.ndarray | None], ...],
    ) -> None:
        self._render_busy = False
        if task.token == self._render_token and rendered and not self._closing:
            for tile, _image, boundaries in rendered:
                if boundaries is not None:
                    self._boundaries.setdefault(tile.name, boundaries)
            self.canvas.delete("all")
            # Tk drops an image the moment nothing references it, so the tiles
            # are retained only after this main-thread handoff.
            photos = []
            for tile, image, _boundaries in rendered:
                photo = ImageTk.PhotoImage(image)
                photos.append(photo)
                self.canvas.create_image(tile.x, tile.y, anchor="nw", image=photo)
                self.canvas.create_text(
                    tile.text_x, tile.text_y, anchor="nw", fill=FOREGROUND,
                    text=tile.caption,
                )
            self._photo = photos
        if self._render_pending is not None:
            self._start_render()
        self._refresh_busy_cursor()

    # -- interaction ------------------------------------------------------
    def _on_click(self, event: tk.Event) -> None:
        if self.zoomed is not None:
            self._drag = (event.x, event.y)
            return
        for x, y, cell_w, cell_h, name in self._tiles:
            if x <= event.x < x + cell_w and y <= event.y < y + cell_h:
                self._open(name)
                return

    def _on_release(self, _event: tk.Event) -> None:
        self._drag = None

    def _on_right_click(self, _event: tk.Event) -> None:
        self._show_sheet()

    def _on_double_click(self, _event: tk.Event) -> None:
        """Refit the open texture, whatever the view has wandered to."""
        if self.zoomed is not None:
            self._open(self.zoomed)

    def _open(self, name: str) -> None:
        texture = next(t for t in self.textures if t.name == name)
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self.zoomed = name
        self.view_scale = min(
            width / texture.shape[1], height / texture.shape[0]
        )
        self.view_origin = (0.0, 0.0)
        self.redraw()

    def _show_sheet(self) -> None:
        self.zoomed = None
        self.redraw()

    def _on_drag(self, event: tk.Event) -> None:
        if self.zoomed is None or self._drag is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.view_origin = (
            self.view_origin[0] - dx / self.view_scale,
            self.view_origin[1] - dy / self.view_scale,
        )
        self.redraw()

    def _on_wheel(self, event: tk.Event) -> None:
        if self.zoomed is None:
            return
        step = 1.25 if event.delta > 0 else 1 / 1.25
        # Keep the texel under the pointer under the pointer.
        before_x = self.view_origin[0] + event.x / self.view_scale
        before_y = self.view_origin[1] + event.y / self.view_scale
        self.view_scale = max(min(self.view_scale * step, 32.0), 0.02)
        self.view_origin = (
            before_x - event.x / self.view_scale,
            before_y - event.y / self.view_scale,
        )
        self.redraw()


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("glyph_corpus")
    GlyphLab(corpus).mainloop()


if __name__ == "__main__":
    main()
