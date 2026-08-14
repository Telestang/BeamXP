from __future__ import annotations

from .build_and_preview import BuildAndPreviewMixin
from .layout import LayoutMixin
from .part_editing import PartEditingMixin
from .parts_workflow import PartsWorkflowMixin
from .plates_workflow import PlatesWorkflowMixin
from .recommendations_ui import RecommendationsUIMixin
from .shared import *
from .slots_workflow import SlotsWorkflowMixin
from .triggers_workflow import TriggersWorkflowMixin
from .variant_workflow import VariantWorkflowMixin
from .vehicle_browser import VehicleBrowserMixin
from .vehicle_workflow import VehicleWorkflowMixin
from .windowing import WindowingMixin
from .worker_handlers import WorkerHandlersMixin


class HandDriveToolApp(
    WindowingMixin,
    LayoutMixin,
    VehicleBrowserMixin,
    VehicleWorkflowMixin,
    VariantWorkflowMixin,
    PartsWorkflowMixin,
    SlotsWorkflowMixin,
    TriggersWorkflowMixin,
    PlatesWorkflowMixin,
    RecommendationsUIMixin,
    PartEditingMixin,
    BuildAndPreviewMixin,
    WorkerHandlersMixin,
    tk.Tk,
):
    """Thin application composition root.

    Behaviour lives in workflow-focused mixins; this class owns only shared
    application state and startup orchestration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("BeamXP - BeamNG Vehicle eXPort Services")
        self._set_app_icon()
        self.geometry("1480x840")
        self.minsize(480, 360)

        self.context: core.VehicleContext | None = None
        self.conversion: dict[str, object] = {}
        self.source_zip: Path | None = None
        self.vehicle_ids: list[str] = []
        # Model dropdown: combo label -> (zip path, vehicle id)
        self.model_entries: dict[str, tuple[Path, str]] = {}
        self.model_load_busy = False
        # Vehicles discovered by scanning the configured game/mods folders.
        self.vehicle_listings: list[object] = []
        self.inventory_scan_seq = 0
        # Thumbnail beside the Model dropdown; the hover override previews the
        # highlighted entry before the user commits to it.
        self.model_preview_photo = None
        self.model_preview_hover: str | None = None
        # Last label the user actually committed to, used to put the box back
        # when they click the recents divider.
        self.last_model_label = ""
        self.preview_photo_cache: dict[str, object] = {}
        self._model_popdown_listbox: str | None = None
        self._model_hover_after: str | None = None
        self.settings = core.load_app_settings()
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_running = False
        self.part_resolver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rhd-parts")
        self.part_table_builder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rhd-part-table")
        self.variant_detector = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rhd-variants")
        self.variant_detection_seq = 0
        self.variant_detection_running = False
        self.variant_detection_pending = False
        self.variant_detected_hands: dict[str, str] = {}
        self.variant_detection_complete = False
        self.part_refresh_after_id: str | None = None
        self.part_refresh_running = False
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self.part_refresh_seq = 0
        self.part_table_seq = 0
        self.part_table_running = False
        self.part_table_pending = False
        self.part_table_pending_reset = False
        self.part_table_requested_key: tuple[object, ...] | None = None
        self.part_table_snapshot_key: tuple[object, ...] | None = None
        self.part_table_snapshot: dict[str, object] | None = None
        self.resolved_part_ids: list[str] = []
        self.vehicle_load_seq = 0
        self.recommendation_seq = 0
        self.recommendation_modal: tk.Toplevel | None = None
        self.plate_editor_modal: PlateEditorDialog | None = None
        self.plate_library_modal: PlateLibraryDialog | None = None
        self.recommendation_tree: ttk.Treeview | None = None
        self.recommendation_rows: dict[str, dict[str, str]] = {}
        self.structural_prompt_after_id: str | None = None
        self.structural_prompt_part_id: str | None = None
        self.structural_prompt_previous_mode: str = core.MODE_SKIP
        self.structural_prompt_open = False
        # Per-table click-to-sort state: tree -> (column id or None, descending)
        self._tree_sort: dict[ttk.Treeview, tuple[str | None, bool]] = {}
        self._tree_heading_text: dict[ttk.Treeview, dict[str, str]] = {}
        # Treeview has no native cell editors.  Keep one temporary combobox
        # overlay at a time and manage its popdown/focus lifecycle explicitly.
        self._tree_combo_editor: ttk.Combobox | None = None
        self._tree_combo_focus_after_id: str | None = None
        self.part_filter_entry: ttk.Entry | None = None

        self.source_var = tk.StringVar(value="No source zip loaded")
        self.vehicle_var = tk.StringVar()
        self.project_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        self.filter_var = tk.StringVar()
        self.slot_filter_var = tk.StringVar()
        self.trigger_filter_var = tk.StringVar()
        self.auto_delta_var = tk.StringVar(value="")
        self.manual_delta_enabled = tk.BooleanVar(value=False)
        self.manual_delta_var = tk.StringVar(value="")
        self.plate_choice_var = tk.StringVar(value="Off")
        self.plate_choice_to_id: dict[str, str] = {}
        self.mods_folder_var = tk.StringVar(value=str(self.settings.get("modsFolder") or ""))
        self.folder_hint_var = tk.StringVar(value="")
        self.include_automation_var = tk.BooleanVar(
            value=bool(self.settings.get("includeAutomationVehicles"))
        )
        self.blender_var = tk.StringVar(value=str(self.settings.get("blenderExecutable") or ""))
        self.texture_quality_var = tk.StringVar(value="")
        self.texture_quality_to_tier: dict[str, str] = {}
        self.preview_output_var = tk.StringVar(value="")
        self.derived_output_var = tk.StringVar(value="")
        self.preview_output_to_config: dict[str, str] = {}
        self.preview_output_to_output: dict[str, str] = {}
        # While the Config dropdown list is open, the highlighted (not
        # yet confirmed) entry hot-loads into the preview via this override.
        self.preview_output_hover: str | None = None
        self._preview_popdown_listbox: str | None = None
        self._preview_hover_after: str | None = None

        self.viewer: ModelPreview | None = None
        # Box-viewer preview data for the trim on screen; ModelPreview holds
        # this by reference, so it is mutated rather than replaced.
        self.box_preview_by_id: dict[str, dict[str, object]] = {}
        self._box_preview_config: str | None = None
        self.viewer_supports_scene = False
        self.mesh_scene_seq = 0
        self.mesh_scene_after: str | None = None
        self.mesh_scene_running = False
        self.mesh_scene_pending = False
        self.mesh_scene_hash: str | None = None
        self.mesh_scene_reset_pending = True
        self.current_part_ids: list[str] = []
        self.part_row_mesh_ids: dict[str, str] = {}
        self.part_row_side_refs: dict[str, str] = {}
        self.part_row_positions: dict[str, tuple[tuple[float, float, float], bool]] = {}
        self.part_instance_rows: dict[str, dict[str, object]] = {}
        self.part_row_labels: dict[str, str] = {}
        self.part_child_overrides: dict[str, dict[str, str]] = {}
        # Armed on every vehicle load and spent on the first refresh that has
        # rows, so the column widths fit the vehicle on screen and then stay
        # wherever the user drags them.
        self.part_columns_need_fit = True
        self.mesh_instance_numbering_key: tuple[object, str] | None = None
        self.mesh_instance_numbering_cache: dict[str, dict[str, int]] = {}
        self.mesh_instance_keys_cache: dict[str, list[str]] = {}
        self.current_slot_ids: list[str] = []
        self.slot_usage_key: tuple[str, ...] | None = None
        self.slot_usage_cache: dict[str, object] | None = None
        self.side_pair_pick_target: dict[str, str] | None = None
        self.trigger_rows_by_iid: dict[str, dict[str, object]] = {}
        self.trigger_owner_cache: tuple[object, object] | None = None
        self._trigger_box_cache: tuple[object, list] | None = None
        # Whether the trigger boxes are drawn in the preview. A view toggle, so
        # it lives here rather than in the conversion, and it survives a vehicle
        # change the way the parts table's Visible column does not.
        self.triggers_visible = True

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_theme()
        self._build_ui()
        self.bind("<KeyPress-h>", self._toggle_selected_parts_visibility_shortcut)
        self.bind("<KeyPress-H>", self._toggle_selected_parts_visibility_shortcut)
        self.bind("<Control-t>", self._focus_trigger_table_shortcut)
        self.bind("<Control-T>", self._focus_trigger_table_shortcut)
        self.bind("<Control-p>", self._focus_side_pair_table_shortcut)
        self.bind("<Control-P>", self._focus_side_pair_table_shortcut)
        for hotkey, hotkey_mode in MODE_HOTKEYS.items():
            self.bind(f"<KeyPress-{hotkey}>", lambda event, m=hotkey_mode: self._set_selected_mode_shortcut(event, m))
            self.bind(f"<KeyPress-{hotkey.upper()}>", lambda event, m=hotkey_mode: self._set_selected_mode_shortcut(event, m))
        self.bind_all("<Button-1>", self._clear_part_filter_focus_on_click, add="+")
        self._refresh_folder_buttons()
        self._rebuild_model_combo()
        self.after_idle(self._maximize_on_start)
        self.after(120, self._poll_worker_queue)
        # Folder scan is ~1s over 190 zips, so it runs off the UI thread once
        # the window is up rather than blocking startup.
        self.after(60, self._start_inventory_scan)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic BeamNG hand-drive visual conversion tool")
    parser.add_argument("--source", help="Vehicle source zip to open")
    parser.add_argument("--vehicle", help="Vehicle catalog id to load from the source zip")
    parser.add_argument("--validate", action="store_true", help="Print detected inventory and exit")
    return parser.parse_args()


def validate_source(source: Path, vehicle: str | None) -> None:
    context = core.load_vehicle_context(source, vehicle)
    conversion, loaded = core.load_or_create_conversion(context)
    print(f"Source: {context.source_zip}")
    print(f"Vehicle: {context.vehicle_id}")
    print(f"Project: {context.project_dir}")
    print(f"Project config loaded: {loaded}")
    print(f"DAE files: {len(context.dae_paths)}")
    print(f"Variants: {len(context.variants)}")
    print(f"DAE objects: {len(context.objects)}")
    print(f"Auto delta magnitude: {fmt_float(core.auto_delta_magnitude(context, conversion))}")


def main() -> None:
    args = parse_args()
    if args.validate:
        if not args.source:
            raise SystemExit("--validate requires --source")
        validate_source(Path(args.source), args.vehicle)
        return

    app = HandDriveToolApp()
    if args.source:
        app.after(50, lambda: app._load_source_zip(Path(args.source), args.vehicle))
    else:
        last_source = str(app.settings.get("lastVehicleZipPath") or "")
        last_vehicle = str(app.settings.get("lastVehicleId") or "")
        if last_source and Path(last_source).exists():
            app.after(
                50,
                lambda: app._load_source_zip(
                    Path(last_source),
                    last_vehicle or None,
                ),
            )
    app.mainloop()


if __name__ == "__main__":
    main()
