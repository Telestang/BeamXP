from __future__ import annotations

from .shared import *


class PartsWorkflowMixin:
    """Part inventory refresh, filtering, selection, viewer synchronisation, and variant interaction."""

    def _part_row_mesh_id(self, row_id: object) -> str:
        text = str(row_id)
        return getattr(self, "part_row_mesh_ids", {}).get(text, text.split("@@", 1)[0])

    def _part_row_side_ref(self, row_id: object) -> str:
        text = str(row_id)
        return getattr(self, "part_row_side_refs", {}).get(text, text)

    def _selected_part_mesh_ids(self) -> set[str]:
        if not hasattr(self, "part_tree"):
            return set()
        return {
            self._part_row_mesh_id(row_id)
            for row_id in self.part_tree.selection()
            if self.part_tree.exists(row_id)
        }

    def _selected_preview_ids(self) -> set[str]:
        if not hasattr(self, "part_tree"):
            return set()
        selected = {
            str(row_id)
            for row_id in self.part_tree.selection()
            if self.part_tree.exists(row_id)
        }
        if not selected:
            return set()
        selected_meshes = {
            self._part_row_mesh_id(row_id)
            for row_id in self.part_tree.selection()
            if self.part_tree.exists(row_id)
        }
        selected_refs = {
            self._part_row_side_ref(row_id)
            for row_id in self.part_tree.selection()
            if self.part_tree.exists(row_id)
        }
        selected.update(ref for ref in selected_refs if ref)
        scene = getattr(self.viewer, "scene", None) if getattr(self, "viewer", None) is not None else None
        groups = getattr(scene, "groups", {}) if scene is not None else {}
        if isinstance(groups, dict):
            for mesh_id in selected_meshes:
                if (
                    mesh_id not in selected
                    and mesh_id in groups
                    and len(groups.get(mesh_id, ())) == 1
                ):
                    selected.add(mesh_id)
        pick_to_row = getattr(scene, "pick_to_row", {}) if scene is not None else {}
        if isinstance(pick_to_row, dict):
            for preview_mesh, row_mesh in pick_to_row.items():
                preview_mesh = str(preview_mesh)
                row_mesh = str(row_mesh)
                if row_mesh in selected or row_mesh in selected_meshes:
                    selected.add(preview_mesh)
        return selected

    def _part_row_label(
        self,
        row_id: str,
        mesh_id: str,
        label_universe: list[str],
    ) -> str:
        display = self._part_display_label(mesh_id, label_universe)
        row = getattr(self, "part_instance_rows", {}).get(row_id)
        if not row:
            return display
        if row.get("display_count_for_mesh", row.get("count_for_mesh", 1)) > 1:
            display = f"{display} #{row.get('display_ordinal_for_mesh', row.get('ordinal_for_mesh', 1))}"
        return display

    @staticmethod
    def _mesh_instance_number_key(instance: object) -> str:
        slot_id = str(getattr(instance, "slot_id", "") or "")
        if slot_id:
            base = f"slot:{slot_id}"
        slot_path = str(getattr(instance, "slot_path", "") or "")
        if not slot_id and slot_path:
            base = f"path:{slot_path}"
        if not slot_id and not slot_path:
            base = str(getattr(instance, "instance_id", "") or "")
        position = getattr(instance, "position", None)
        if isinstance(position, tuple) and position:
            x = float(position[0])
            if x > 1e-4:
                return f"{base}|x:+"
            if x < -1e-4:
                return f"{base}|x:-"
            return f"{base}|x:0"
        return base

    def _mesh_instance_number_for_ref(self, ref: str) -> tuple[int, int] | None:
        if self.context is None:
            return None
        mesh_id = ref.split("@@", 1)[0]
        if not mesh_id:
            return None
        numbering = self._vehicle_mesh_instance_numbering().get(mesh_id, {})
        if not numbering:
            return None
        for config_name in sorted(self.context.variants):
            try:
                instances = self._mesh_transform_instances_for_config(config_name)
            except Exception:
                continue
            for instance in instances:
                if instance.mesh_id != mesh_id:
                    continue
                if instance.instance_id != ref:
                    continue
                ordinal = numbering.get(self._mesh_instance_number_key(instance))
                if ordinal is not None:
                    return ordinal, len(numbering)
        return None

    def _mesh_instance_label_for_ref(
        self,
        ref: str,
        label_universe: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        mesh_id = ref.split("@@", 1)[0]
        display = self._part_display_name(mesh_id)
        numbered = self._mesh_instance_number_for_ref(ref)
        if numbered is None:
            return self._part_display_label(mesh_id, label_universe)
        ordinal, count = numbered
        return f"{display} #{ordinal}" if count > 1 else display

    def _mesh_transform_instances_for_config(self, config_name: str) -> list[core.MeshTransformInstance]:
        if self.context is None:
            return []
        conversion = getattr(self, "conversion", {})
        if not isinstance(conversion, dict):
            conversion = {}
        plan = core.slot_pair_plans_for_variants(self.context, conversion, [config_name]).get(config_name)
        if plan is None:
            return core.selected_mesh_transform_instances_for_config(self.context, config_name)
        pc = core.load_pc(self.context.source_zip, self.context.variants[config_name].pc_path)
        core.apply_hand_authored_group(pc, plan)
        selected = core.resolve_selected_parts(
            pc,
            self.context.jbeam_texts,
            vehicle_id=self.context.source_vehicle_id,
            part_body_index=self.context.part_body_index,
        )
        return core.selected_mesh_transform_instances_for_selection(self.context, selected)

    def _mesh_numbering_cache_key(self) -> tuple[object, str]:
        conversion = getattr(self, "conversion", {})
        if not isinstance(conversion, dict):
            return (id(self.context), "")
        try:
            fingerprint = json.dumps(
                {
                    "slotPairs": core.normalized_slot_pairs(conversion.get("slotPairs")),
                    "sidePairs": core.normalized_side_pairs(conversion.get("sidePairs")),
                },
                sort_keys=True,
            )
        except Exception:
            fingerprint = ""
        return (id(self.context), fingerprint)

    def _vehicle_mesh_instance_numbering(self) -> dict[str, dict[str, int]]:
        if self.context is None:
            return {}
        cache_key = self._mesh_numbering_cache_key()
        if (
            getattr(self, "mesh_instance_numbering_key", None) == cache_key
            and isinstance(getattr(self, "mesh_instance_numbering_cache", None), dict)
        ):
            return self.mesh_instance_numbering_cache
        keys_by_mesh: dict[str, set[str]] = {}
        for config_name in sorted(self.context.variants):
            try:
                instances = self._mesh_transform_instances_for_config(config_name)
            except Exception:
                continue
            for instance in instances:
                if instance.count_for_mesh <= 1:
                    continue
                mesh_id = str(instance.mesh_id)
                key = self._mesh_instance_number_key(instance)
                if mesh_id and key:
                    keys_by_mesh.setdefault(mesh_id, set()).add(key)
        numbering = {
            mesh_id: {
                key: index + 1
                for index, key in enumerate(sorted(keys, key=str.lower))
            }
            for mesh_id, keys in keys_by_mesh.items()
            if len(keys) > 1
        }
        self.mesh_instance_numbering_key = cache_key
        self.mesh_instance_numbering_cache = numbering
        return numbering

    def _part_table_rows(self, ids: list[str]) -> list[dict[str, object]]:
        if self.context is None:
            return []
        config = self._mesh_scene_config()
        vehicle_numbering = self._vehicle_mesh_instance_numbering()
        rows: list[dict[str, object]] = []
        used: set[str] = set()
        row_id_counts: dict[str, int] = {}
        if config is not None and config in self.context.variants:
            try:
                instances = self._mesh_transform_instances_for_config(config)
            except Exception:
                instances = []
            for instance in instances:
                mesh_id = instance.mesh_id
                if mesh_id not in ids or mesh_id not in self.context.objects:
                    continue
                used.add(mesh_id)
                number_key = self._mesh_instance_number_key(instance)
                mesh_numbering = vehicle_numbering.get(mesh_id, {})
                if number_key in mesh_numbering:
                    display_ordinal = mesh_numbering[number_key]
                    display_count = len(mesh_numbering)
                else:
                    display_ordinal = instance.ordinal_for_mesh
                    display_count = instance.count_for_mesh
                base_row_id = f"{mesh_id}@@{display_ordinal}" if display_count > 1 else mesh_id
                row_occurrence = row_id_counts.get(base_row_id, 0) + 1
                row_id_counts[base_row_id] = row_occurrence
                row_id = base_row_id if row_occurrence == 1 else f"{base_row_id}@@row{row_occurrence}"
                rows.append({
                    "row_id": row_id,
                    "mesh_id": mesh_id,
                    "side_ref": instance.instance_id,
                    "part_id": instance.part_id,
                    "slot_id": instance.slot_id,
                    "slot_path": instance.slot_path,
                    "position": instance.position,
                    "count_for_mesh": instance.count_for_mesh,
                    "ordinal_for_mesh": instance.ordinal_for_mesh,
                    "display_count_for_mesh": display_count,
                    "display_ordinal_for_mesh": display_ordinal,
                    "instance_number_key": number_key,
                })
        for mesh_id in ids:
            if mesh_id in used or mesh_id not in self.context.objects:
                continue
            rows.append({
                "row_id": mesh_id,
                "mesh_id": mesh_id,
                "side_ref": mesh_id,
                "part_id": "",
                "slot_id": "",
                "slot_path": "",
                "position": None,
                "count_for_mesh": 1,
                "ordinal_for_mesh": 1,
            })
        rows.sort(key=lambda row: (str(row["mesh_id"]).lower(), str(row["row_id"])))
        return rows

    def _replace_source_child_overrides(
        self,
        rows: list[dict[str, object]],
        label_universe: list[str],
    ) -> dict[str, dict[str, str]]:
        parts = self.conversion.get("parts", {})
        if not isinstance(parts, dict):
            return {}
        parents: list[dict[str, str]] = []
        for row in rows:
            row_id = str(row.get("row_id") or "")
            mesh_id = str(row.get("mesh_id") or "")
            slot_path = str(row.get("slot_path") or "")
            settings = parts.get(mesh_id)
            if (
                not row_id
                or not mesh_id
                or not slot_path
                or not isinstance(settings, dict)
                or settings.get("mode") != core.MODE_REPLACE_SOURCE
                or not settings.get("includeChildren")
            ):
                continue
            parents.append(
                {
                    "row_id": row_id,
                    "mesh_id": mesh_id,
                    "part_id": str(row.get("part_id") or ""),
                    "slot_path": slot_path,
                    "label": self._part_row_label(row_id, mesh_id, label_universe),
                }
            )
        overrides: dict[str, dict[str, str]] = {}
        for row in rows:
            row_id = str(row.get("row_id") or "")
            slot_path = str(row.get("slot_path") or "")
            if not row_id or not slot_path:
                continue
            matches = [
                parent
                for parent in parents
                if parent["row_id"] != row_id
                and (
                    (
                        slot_path == parent["slot_path"]
                        and str(row.get("part_id") or "") == parent.get("part_id", "")
                    )
                    or (
                        slot_path != parent["slot_path"]
                        and slot_path.startswith(parent["slot_path"])
                    )
                )
            ]
            if not matches:
                continue
            parent = max(matches, key=lambda item: len(item["slot_path"]))
            override = dict(parent)
            override["kind"] = "child"
            overrides[row_id] = override
        scene = getattr(self.viewer, "scene", None) if getattr(self, "viewer", None) is not None else None
        pick_to_row = getattr(scene, "pick_to_row", {}) if scene is not None else {}
        if isinstance(pick_to_row, dict):
            by_mesh = {str(row.get("mesh_id") or ""): row for row in rows}
            parents_by_mesh = {parent["mesh_id"]: parent for parent in parents}
            pick_to_parent = getattr(scene, "pick_to_parent", {}) if scene is not None else {}
            for preview_mesh, row_mesh in pick_to_row.items():
                preview_mesh = str(preview_mesh)
                row_mesh = str(row_mesh)
                if "@@" in preview_mesh:
                    continue
                parent = overrides.get(row_mesh)
                if not parent and isinstance(pick_to_parent, dict):
                    parent = parents_by_mesh.get(str(pick_to_parent.get(preview_mesh) or ""))
                if not parent:
                    continue
                preview_label = self._part_display_label(preview_mesh, [*label_universe, preview_mesh])
                if row_mesh != preview_mesh and row_mesh != parent["mesh_id"]:
                    replaced = dict(parent)
                    replaced["kind"] = "replaced"
                    replaced["preview_mesh"] = preview_mesh
                    replaced["source_label"] = f"{preview_label} via {parent['label']}"
                    overrides[row_mesh] = replaced
                if preview_mesh in by_mesh:
                    source = dict(parent)
                    source["kind"] = "source"
                    source["preview_mesh"] = preview_mesh
                    source["source_label"] = f"via {parent['label']}"
                    overrides[preview_mesh] = source
        return overrides

    def _part_child_override(self, row_id: object) -> dict[str, str] | None:
        override = getattr(self, "part_child_overrides", {}).get(str(row_id))
        return override if isinstance(override, dict) else None

    def _part_child_source_label(self, override: dict[str, str]) -> str:
        return str(override.get("source_label") or override.get("label") or "")

    def _part_override_mode_label(self, override: dict[str, str]) -> str:
        kind = str(override.get("kind") or "child")
        if kind == "source":
            return "Source"
        if kind == "replaced":
            return "Replaced"
        return "Child"

    def _part_override_children_label(self, override: dict[str, str]) -> str:
        return "Child" if str(override.get("kind") or "child") == "child" else "N/A"

    def _part_override_status_label(self, override: dict[str, str]) -> str:
        source = self._part_child_source_label(override)
        kind = str(override.get("kind") or "child")
        if kind == "source":
            return f"Source geometry shown {source}"
        if kind == "replaced":
            return f"Replaced by {source}"
        return f"Child transform inherited from {source}"

    def _part_children_label(self, object_id: str, settings: object) -> str:
        if not isinstance(settings, dict) or settings.get("mode") != core.MODE_REPLACE_SOURCE:
            return "N/A"
        return yn_label(settings.get("includeChildren"))

    def _refresh_parts(self, *, reset_view: bool = False) -> None:
        if self.context is None:
            if hasattr(self, "part_tree"):
                for item in self.part_tree.get_children():
                    self.part_tree.delete(item)
            self.current_part_ids = []
            self.part_row_mesh_ids = {}
            self.part_row_side_refs = {}
            self.part_row_positions = {}
            self.part_instance_rows = {}
            self.part_child_overrides = {}
            self._refresh_slots()
            self._refresh_derived_output_summary()
            return
        query = self.filter_var.get().strip().lower()
        # The x/y/z columns read placed geometry, so make sure it matches the
        # trim on screen before the rows are built.
        self._refresh_box_preview()
        keep = set(self.part_tree.selection())
        previous_order = list(self.part_tree.get_children(""))
        for item in self.part_tree.get_children():
            self.part_tree.delete(item)

        parts = self.conversion.setdefault("parts", {})
        ids = self.resolved_part_ids
        active_ids = self._preview_active_ids()
        selected_variants = self._selected_variant_names()
        flexbody_meshes, prop_meshes, _all_meshes = core.selected_mesh_roles(
            self.context,
            selected_variants,
        )
        label_universe = [object_id for object_id in ids if object_id in self.context.objects]
        table_rows = self._part_table_rows(ids)
        self.part_child_overrides = self._replace_source_child_overrides(table_rows, label_universe)
        self.part_row_mesh_ids = {str(row["row_id"]): str(row["mesh_id"]) for row in table_rows}
        self.part_row_side_refs = {str(row["row_id"]): str(row["side_ref"]) for row in table_rows}
        self.part_instance_rows = {str(row["row_id"]): row for row in table_rows}
        self.part_row_positions = {
            str(row["row_id"]): (
                row["position"],
                str(row["mesh_id"]) in self.context.variant_dependent_meshes,
            )
            for row in table_rows
            if row.get("position") is not None
        }
        displayed: list[str] = []
        row_index = 0
        for row in table_rows:
            row_id = str(row["row_id"])
            object_id = str(row["mesh_id"])
            obj = self.context.objects.get(object_id)
            if obj is None:
                continue
            settings = parts.setdefault(
                object_id,
                {
                    "mode": core.MODE_SKIP,
                    "translateOffset": None,
                    "steeringRef": False,
                    "includeChildren": False,
                    "viewerVisible": True,
                    "viewerSolo": False,
                },
            )
            if not isinstance(settings, dict):
                continue
            mode = str(settings.get("mode", core.MODE_SKIP))
            display_name = self._part_display_name(object_id)
            display_label = self._part_row_label(row_id, object_id, label_universe)
            child_override = self._part_child_override(row_id)
            mode_display = self._part_override_mode_label(child_override) if child_override else mode_label(mode)
            source_display = (
                self._part_child_source_label(child_override)
                if child_override
                else self._swap_source_label(object_id, settings)
            )
            part_type = part_type_label(object_id, flexbody_meshes, prop_meshes)
            if (
                query
                and query not in object_id.lower()
                and query not in display_name.lower()
                and query not in display_label.lower()
                and query not in mode_display.lower()
                and query not in source_display.lower()
                and query not in str(row.get("slot_path") or "").lower()
                and query not in str(row.get("part_id") or "").lower()
                and query not in mode
                and query not in part_type.lower()
            ):
                continue
            displayed.append(row_id)
            self.part_tree.insert(
                "",
                "end",
                iid=row_id,
                text=display_label,
                tags=self._row_tags(row_index),
                values=(
                    part_type,
                    yn_label(settings.get("viewerVisible", True)),
                    yn_label(settings.get("viewerSolo")),
                    yn_label(object_id in active_ids),
                    mode_display,
                    source_display,
                    self._part_override_children_label(child_override)
                    if child_override
                    else self._part_children_label(object_id, settings),
                    offset_display(
                        core.MODE_SKIP if child_override else mode,
                        settings.get("translateOffset"),
                        manual_delta=self.manual_delta_enabled.get(),
                    ),
                    yn_label(settings.get("steeringRef")),
                    *position_labels(*self._table_position(row_id)),
                ),
            )
            row_index += 1
        self.current_part_ids = displayed
        self._refresh_slots()
        self._restore_tree_order(self.part_tree, previous_order)
        visible_keep = [item for item in keep if self.part_tree.exists(item)]
        if visible_keep:
            self.part_tree.selection_set(visible_keep)
        self._refresh_viewer(reset=reset_view)

    def _schedule_parts_refresh(self, *, reset_view: bool = False) -> None:
        self.part_refresh_pending_reset = self.part_refresh_pending_reset or reset_view
        if self.part_refresh_running:
            self.part_refresh_pending = True
            self.part_refresh_seq += 1
            return
        if self.part_refresh_after_id is not None:
            return
        self.part_refresh_after_id = self.after_idle(self._run_scheduled_parts_refresh)

    def _run_scheduled_parts_refresh(self) -> None:
        self.part_refresh_after_id = None
        reset_view = self.part_refresh_pending_reset
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self._start_parts_refresh(reset_view=reset_view)

    def _start_parts_refresh(self, *, reset_view: bool = False) -> None:
        self.part_refresh_after_id = None
        if self.context is None:
            self.resolved_part_ids = []
            self._refresh_parts(reset_view=reset_view)
            return
        selected = tuple(self._selected_variant_names())
        self.part_refresh_seq += 1
        seq = self.part_refresh_seq
        if not selected:
            self.resolved_part_ids = []
            self._refresh_parts(reset_view=reset_view)
            self.status_var.set("No trims selected; 0 used part(s) displayed")
            return
        context = self.context
        cached_ids = core.load_cached_part_ids(context, selected)
        if cached_ids is not None:
            self.resolved_part_ids = [part_id for part_id in cached_ids if part_id in context.objects]
            self._refresh_parts(reset_view=reset_view)
            self._update_detail()
            self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed (parts cache)")
            return
        self.status_var.set(f"Resolving used parts for {len(selected)} trim(s)...")
        self.part_refresh_running = True
        future = self.part_resolver.submit(self._resolve_part_ids_worker, context, selected)
        future.add_done_callback(
            lambda completed, current_seq=seq, current_context=context, should_reset=reset_view, current_selected=selected: self.worker_queue.put(
                ("parts_success", (current_seq, current_context, should_reset, current_selected, completed))
            )
        )

    @staticmethod
    def _resolve_part_ids_worker(
        context: core.VehicleContext,
        selected: tuple[str, ...],
    ) -> list[str]:
        _flex, _props, all_meshes = core.selected_mesh_roles(context, list(selected))
        return sorted(mesh for mesh in all_meshes if mesh in context.objects)

    def _selected_variant_names(self) -> list[str]:
        if self.context is None:
            return []
        variants = self.conversion.get("variants", {})
        if not isinstance(variants, dict):
            return []
        return [
            name
            for name, settings in variants.items()
            if name in self.context.variants
            and isinstance(settings, dict)
            and core.variant_build_mode(settings) != core.BUILD_OFF
        ]

    def _preview_base_part_ids(self) -> list[str]:
        if self.context is None:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for row_id in (self.resolved_part_ids or self.current_part_ids):
            object_id = self._part_row_mesh_id(row_id)
            if object_id in self.context.objects and object_id not in seen:
                seen.add(object_id)
                out.append(object_id)
        return out

    def _resolved_visible_ids(self) -> set[str]:
        """The set of parts actually present in the active preview / final
        visible output for the current variant selection: Solo (if any part is
        soloed) or per-part Visible toggles, over the resolved used-part set.
        Table selection deliberately has no effect here -- Visible/Solo have the
        final say over what the preview and the converted output contain."""
        if self.context is None:
            return set()
        parts = self.conversion.get("parts", {})
        base_ids = self._preview_base_part_ids()
        solo_ids = {
            object_id
            for object_id in base_ids
            if isinstance(parts, dict)
            and isinstance(parts.get(object_id), dict)
            and parts[object_id].get("viewerSolo")
        }
        if solo_ids:
            return solo_ids
        return {
            object_id
            for object_id in base_ids
            if not isinstance(parts, dict)
            or not isinstance(parts.get(object_id), dict)
            or parts[object_id].get("viewerVisible", True)
        }

    def _refresh_viewer(self, *, reset: bool = False) -> None:
        if self.viewer is None:
            return
        visible_ids = self._resolved_visible_ids()
        # Selected inactive parts are temporarily injected into the GPU scene
        # (scene.extra); show them while they stay selected. Intersecting with
        # the live selection hides a stale extra instantly after deselection,
        # before the scene rebuild that drops it has landed.
        scene = getattr(self.viewer, "scene", None)
        selected_meshes = self._selected_part_mesh_ids()
        pick_to_row = getattr(scene, "pick_to_row", {}) if scene is not None else {}
        if isinstance(pick_to_row, dict):
            for preview_mesh, row_mesh in pick_to_row.items():
                if str(row_mesh) in visible_ids:
                    visible_ids.add(str(preview_mesh))
        visible_ids |= set(getattr(scene, "extra", ()) or ()) & selected_meshes
        current_mesh_ids = {self._part_row_mesh_id(row_id) for row_id in self.current_part_ids}
        dimmed_ids = visible_ids - current_mesh_ids
        self.viewer.set_visible_ids(list(visible_ids), reset=reset)
        if hasattr(self.viewer, "set_dimmed_ids"):
            self.viewer.set_dimmed_ids(dimmed_ids)
        # Selection only drives the highlight outline (skipped for hidden parts
        # in the renderer); it never adds a part to the visible set above.
        selected_ids = self._selected_preview_ids() if self.viewer_supports_scene else selected_meshes
        self.viewer.set_selected_ids(selected_ids)

    def _preview_active_ids(self) -> set[str]:
        """Object ids present on the trim currently shown in the moderngl
        preview -- i.e. the config chosen in the Config dropdown. This
        indicates which parts the converted trim actually uses; it is NOT
        affected by the viewer Visible/Solo toggles (those only filter what is
        drawn). Ground truth is the built scene's mesh groups (keyed by object
        id, already excluding inactive/geometry-less rows for this config)."""
        scene = getattr(self.viewer, "scene", None) if self.viewer is not None else None
        groups = getattr(scene, "groups", None)
        if groups:
            # Temporarily-shown inactive parts (scene.extra) are in the scene
            # but not part of the previewed trim; they are never Active.
            return set(groups.keys()) - set(getattr(scene, "extra", ()) or ())
        # No GPU scene yet (box-viewer fallback, or the preview is still
        # building): resolve the previewed config's meshes directly. Roles are
        # cached per config on the context, so this stays cheap.
        if self.context is None:
            return set()
        config = self._mesh_scene_config()
        if config is None:
            return set()
        try:
            _flex, _props, all_meshes = core.selected_mesh_roles(self.context, [config])
        except Exception:
            return set()
        return {mesh for mesh in all_meshes if mesh in self.context.objects}

    def _selected_extra_preview_ids(self) -> list[str]:
        """Inactive table meshes that should still be present in the GPU scene.

        Selection and Solo are both explicit user requests to look at a mesh.
        If the focused trim does not use that mesh, inject it as ``extra`` so
        the visibility filter cannot collapse the scene to an absent id.
        Active meshes are never in this list, so normal trim preview behaviour
        is unchanged.
        """
        if self.context is None or not hasattr(self, "part_tree"):
            return []
        config = self._mesh_scene_config()
        if config is None:
            return []
        try:
            _flex, _props, all_meshes = core.selected_mesh_roles(self.context, [config])
        except Exception:
            return []
        parts = self.conversion.get("parts", {})
        solo_ids = {
            object_id
            for object_id in self._preview_base_part_ids()
            if isinstance(parts, dict)
            and isinstance(parts.get(object_id), dict)
            and parts[object_id].get("viewerSolo")
        }
        return sorted(
            object_id
            for object_id in self._selected_part_mesh_ids() | solo_ids
            if object_id in self.context.objects and object_id not in all_meshes
        )

    def _refresh_active_cells(self) -> None:
        """Update the parts table Active (Y/N) column for every displayed row to
        reflect the trim currently shown in the moderngl preview."""
        if not hasattr(self, "part_tree") or self.part_tree is None:
            return
        active_ids = self._preview_active_ids()
        for row_id in self.part_tree.get_children():
            self.part_tree.set(row_id, "active", yn_label(self._part_row_mesh_id(row_id) in active_ids))

    def _refresh_derived_output_summary(self) -> None:
        if self.context is None:
            self.derived_output_var.set("")
            return
        modes = core.active_part_modes(self.conversion)
        move = sum(1 for mode in modes.values() if mode == core.MODE_TRANSLATE)
        mirror_move = sum(1 for mode in modes.values() if mode == core.MODE_MIRROR_POSITION)
        mirror = sum(1 for mode in modes.values() if mode == core.MODE_MIRROR)
        swap = sum(1 for mode in modes.values() if mode == core.MODE_MIRROR_STRUCTURAL)
        replace = sum(1 for mode in modes.values() if mode == core.MODE_REPLACE_SOURCE)
        pairs = len(core.active_slot_pairs(self.conversion)) + len(
            [
                pair
                for pair in core.normalized_side_pairs(self.conversion.get("sidePairs"))
                if pair.get("enabled", True)
            ]
        )
        selected = len(self._selected_variant_names())
        self.derived_output_var.set(
            f"{selected} config(s); Move {move}, Mirror Move {mirror_move}, Mirror {mirror}, "
            f"Swap Mesh {swap}, Replace Source {replace}; "
            f"{pairs} equivalent pair(s); auto fixes: lights, cameras, bridge parts"
        )

    def _refresh_delta_label(self) -> None:
        if self.context is None:
            self.auto_delta_var.set("")
            return
        auto = core.auto_delta_magnitude(self.context, self.conversion)
        source_refs = core.auto_delta_source_refs(self.context, self.conversion)
        if source_refs:
            names = ", ".join(self._part_display_name(object_id) for object_id in source_refs)
            source = f"found using {names}"
        else:
            # No steering ref selected (or the selected one has no usable
            # off-center X), so the auto delta is just its default.
            source = "no steering ref found"
        self.auto_delta_var.set(f"{fmt_float(auto)} ({source})")

    def _update_detail(self) -> None:
        if self.context is None:
            self.detail_var.set("")
            return
        # Every conversion mutation (mode, translate offset, equivalent parts,
        # steering ref, manual delta, variant hand override) funnels through here
        # as its final UI step, so this is where we keep the GPU preview live.
        # _schedule_mesh_scene is snapshot-guarded: pure selection/visibility
        # changes leave the fingerprint unchanged and cost only a cheap compare.
        self._schedule_mesh_scene()
        self._refresh_derived_output_summary()
        selected_parts = self.part_tree.selection()
        if selected_parts:
            row_id = selected_parts[0]
            object_id = self._part_row_mesh_id(row_id)
            obj = self.context.objects.get(object_id)
            settings = self.conversion.get("parts", {}).get(object_id, {})
            if obj:
                display_name = self._part_row_label(row_id, object_id, self.resolved_part_ids)
                mode = str(settings.get("mode", core.MODE_SKIP)) if isinstance(settings, dict) else core.MODE_SKIP
                child_override = self._part_child_override(row_id)
                mode_display = self._part_override_mode_label(child_override) if child_override else mode_label(mode)
                part_offset = (
                    offset_display(
                        mode,
                        settings.get("translateOffset") if isinstance(settings, dict) else None,
                        manual_delta=self.manual_delta_enabled.get(),
                    )
                    if mode == core.MODE_TRANSLATE and not child_override
                    else "N/A"
                )
                position, varies = self._table_position(row_id)
                source = self._swap_source_label(object_id, settings) if isinstance(settings, dict) else ""
                if child_override:
                    source = self._part_child_source_label(child_override)
                ref_note = ""
                side_ref = self._part_row_side_ref(row_id)
                if side_ref != object_id:
                    row = getattr(self, "part_instance_rows", {}).get(row_id) or {}
                    ref_note = f", instance #{row.get('ordinal_for_mesh', 1)}"
                self.detail_var.set(
                    f"{display_name}: {mode_display}, "
                    f"full id {object_id}{ref_note}, source {source or 'N/A'}, x {fmt_float(position[0])}, offset {part_offset}, "
                    f"dae {obj.dae_path}{self._variant_position_note(object_id) if varies else ''}"
                )
                return
        active = len(core.active_part_modes(self.conversion))
        selected_variants = len(self._selected_variant_names())
        self.detail_var.set(
            f"{len(self.current_part_ids)} displayed mesh(es), {active} transform setting(s), "
            f"{selected_variants} selected variant(s)"
        )

    def _set_all_variants_selected(self, selected: bool) -> None:
        if self.context is None:
            return
        variants = self.conversion.setdefault("variants", {})
        for config_name in self.context.variants:
            settings = variants.setdefault(config_name, {})
            if isinstance(settings, dict):
                core.set_variant_build_mode(settings, core.BUILD_CONVERTED if selected else core.BUILD_OFF)
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()
        self.status_var.set(
            f"{'All trims selected' if selected else 'All trims cleared'}; updating used parts..."
        )

    def _toggle_variant_selected(self, config_name: str) -> None:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            mode = core.variant_build_mode(settings)
            core.set_variant_build_mode(
                settings,
                core.BUILD_OFF if mode != core.BUILD_OFF else core.BUILD_CONVERTED,
            )
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()
        state = BUILD_LABELS[core.variant_build_mode(settings)] if isinstance(settings, dict) else "Off"
        self.status_var.set(f"{config_name} {state}; updating used parts...")

    def _variant_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.variant_tree, event):
            return None
        item = self.variant_tree.identify_row(event.y)
        column = self.variant_tree.identify_column(event.x)
        if not item or self.context is None:
            return
        name = self._tree_column_name(self.variant_tree, column)
        if name == "build":
            settings = self.conversion.setdefault("variants", {}).setdefault(item, {})
            current = BUILD_LABELS[core.variant_build_mode(settings)] if isinstance(settings, dict) else BUILD_LABELS[core.BUILD_OFF]
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                list(BUILD_LABELS.values()),
                current,
                lambda value: self._set_variant_build_label(item, value),
            )
            return "break"
        if name == "stock_hand":
            settings = self.conversion.get("variants", {}).get(item, {})
            if core.variant_build_mode(settings) not in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                return "break"
            labels, mapping = self._variant_stock_hand_choices(item, settings)
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                self._variant_stock_hand_label(item, settings),
                lambda value: self._set_variant_setting(item, "sourceHandOverride", mapping[value]),
            )
            return "break"
        if name == "plate":
            labels, mapping = self._variant_plate_choices(item)
            current_label = self._variant_plate_label(item, self.conversion.get("variants", {}).get(item, {}))
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                current_label,
                lambda value: self._set_variant_plate_choice(item, mapping[value]),
            )
            return "break"
        if name in {"front_plate", "rear_plate"}:
            side = "front" if name == "front_plate" else "rear"
            key = "frontPlate" if side == "front" else "rearPlate"
            choices = plate_generator.plate_part_choices_for_config(self.context, item, side)
            labels = [choice.label for choice in choices]
            values_by_label = {choice.label: choice.value for choice in choices}
            current_value = self._get_variant_setting(item, key, plate_generator.PLATE_PART_AUTO)
            current_label = plate_generator.plate_part_label_for_config(
                self.context,
                item,
                side,
                current_value,
            )
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                current_label,
                lambda value: self._set_variant_setting(item, key, values_by_label[value]),
            )
            return "break"
        if name is not None:
            # Clicking the descriptive cells keeps the old quick on/off action.
            self._toggle_variant_selected(item)
            return "break"
        return None

    def _variant_double_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.variant_tree, event):
            return None
        item = self.variant_tree.identify_row(event.y)
        column = self.variant_tree.identify_column(event.x)
        if not item:
            return
        name = self._tree_column_name(self.variant_tree, column)
        if name == "plate":
            self._open_plate_editor(item)
        elif name == "stock_hand":
            settings = self.conversion.get("variants", {}).get(item, {})
            if core.variant_build_mode(settings) not in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                return "break"
            labels, mapping = self._variant_stock_hand_choices(item, settings)
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                self._variant_stock_hand_label(item, settings),
                lambda value: self._set_variant_setting(item, "sourceHandOverride", mapping[value]),
            )
