from __future__ import annotations

from .shared import *


# What a pairing will actually do for the trim on screen, which is the thing
# the user cannot work out from the two slot names alone.
PLAN_SYMMETRIC = "Symmetric"
PLAN_SWAP = "Swap"
PLAN_RELOCATE = "Relocate"
PLAN_MOVE = "Move"
PLAN_UNPAIRED = ""
SIDE_PAIR_REF_SEP = "@@"
SIDE_PAIR_KIND_LABELS = {
    "seat": "Seat",
    "door": "Door card",
    "mirror": "Wing mirror",
    "part": "Part",
}
SIDE_PAIR_KIND_BY_LABEL = {label: key for key, label in SIDE_PAIR_KIND_LABELS.items()}
SIDE_PAIR_KIND_OPTIONS = ["Seat", "Door card", "Wing mirror"]
# 2026-09-07: the counterpart modal opens just after the click that asked for
# it, never during. A grab taken inside a Treeview button press leaves the tree
# believing the button is still down. The Swap Mesh source prompt waits the
# same 80 ms before it opens, for the same reason.
SIDE_PAIR_MODAL_DELAY_MS = 80


class SlotsWorkflowMixin:
    """Vehicle-level equivalent parts and legacy slot-pair helpers."""

    def _slot_usage(self) -> dict[str, object]:
        if self.context is None:
            return {}
        variants = tuple(self._selected_variant_names())
        if not variants:
            return {}
        if self.slot_usage_key == variants and self.slot_usage_cache is not None:
            return self.slot_usage_cache
        usage = core.slot_usage_for_configs(self.context, variants)
        self.slot_usage_key = variants
        self.slot_usage_cache = usage
        return usage

    def _invalidate_slot_usage(self) -> None:
        self.slot_usage_key = None
        self.slot_usage_cache = None

    def _focused_slot_config(self) -> str | None:
        """The trim the Part and Plan columns describe."""
        selected = self._selected_variant_names()
        if not selected:
            return None
        focused = self.variant_tree.focus()
        if focused and focused in selected:
            return focused
        return sorted(selected)[0]

    def _slot_pair_plan_label(self, slot_type: str, partner: str) -> str:
        if not partner or self.context is None:
            return PLAN_UNPAIRED
        config_name = self._focused_slot_config()
        if config_name is None:
            return PLAN_UNPAIRED
        usage = self._slot_usage()
        usage_a = usage.get(slot_type)
        usage_b = usage.get(partner)
        if usage_a is None or usage_b is None:
            return PLAN_UNPAIRED
        part_a = str(usage_a.part_by_config.get(config_name) or "")
        part_b = str(usage_b.part_by_config.get(config_name) or "")
        if not part_a and not part_b:
            return PLAN_SYMMETRIC

        plan = core.resolve_slot_pair_plan(self.context, config_name, [(slot_type, partner)])
        if plan is None:
            return PLAN_SYMMETRIC
        if core.slot_pair_plan_relocations(plan):
            return PLAN_RELOCATE
        return PLAN_SWAP if part_a and part_b else PLAN_MOVE

    def _slot_occupancy_label(self, slot_usage: object, configs: list[str]) -> str:
        if not configs:
            return ""
        occupied = sum(
            1
            for config_name in configs
            if str(slot_usage.part_by_config.get(config_name) or "")
        )
        if occupied == 0:
            return "empty"
        if occupied == len(configs):
            return "filled"
        return f"{occupied}/{len(configs)} filled"

    def _side_pair_kind(self, *values: str) -> str:
        text = " ".join(value.lower() for value in values if value)
        if any(token in text for token in ("light", "lamp", "bulb", "skin", "material", "paint")):
            return ""
        if re.search(r"\b(seat|bucket|racingseat)\b", text) or any(
            token in text for token in ("seat_", "_seat", "racingseat", "seatbase", "seat_base")
        ):
            return "Seat"
        if "mirror" in text:
            return "Mirror"
        if any(token in text for token in ("doorpanel", "door_panel", "doorcard", "door_card")):
            return "Door"
        if "door" in text and any(token in text for token in ("panel", "card", "release", "control", "handle")):
            return "Door"
        return ""

    def _part_instance_display_label(
        self,
        instance: dict[str, object] | None,
        peers: list[dict[str, object]],
    ) -> str:
        if instance is None:
            return "(empty)"
        part_id = str(instance.get("part_id") or "")
        if not part_id:
            return "(empty)"
        display = part_id
        if self.context is not None:
            found = core.part_body_for_context(self.context, part_id)
            if found is not None:
                info_name = core.part_information_name(found[0])
                if info_name:
                    display = info_name
        same = [
            peer
            for peer in peers
            if str(peer.get("part_id") or "") == part_id
        ]
        if len(same) > 1:
            ordered = sorted(
                same,
                key=lambda peer: (
                    str(peer.get("slot_path") or ""),
                    str(peer.get("instance_id") or ""),
                ),
            )
            display = f"{display} #{ordered.index(instance) + 1}"
        return part_id if display == part_id else f"{display} ({part_id})"

    @staticmethod
    def _side_pair_ref(part_id: str, slot_path: str) -> str:
        return f"{part_id}{SIDE_PAIR_REF_SEP}{slot_path}"

    @staticmethod
    def _split_side_pair_ref(ref: str) -> tuple[str, str]:
        if SIDE_PAIR_REF_SEP not in ref:
            return ref, ""
        part_id, slot_path = ref.split(SIDE_PAIR_REF_SEP, 1)
        return part_id, slot_path

    def _side_pair_ref_part_id(self, ref: str) -> str:
        part_id, _slot_path = self._split_side_pair_ref(ref)
        return part_id

    def _side_pair_instance_candidates(self) -> list[dict[str, object]]:
        instances = [
            instance
            for instance in self._part_instances_for_focused_config()
            if str(instance.get("part_id") or "")
        ]
        candidates: list[dict[str, object]] = []
        for instance in instances:
            part_id = str(instance.get("part_id") or "")
            slot_path = str(instance.get("slot_path") or "")
            ref = self._side_pair_ref(part_id, slot_path)
            candidates.append({
                "ref": ref,
                "part_id": part_id,
                "slot_path": slot_path,
                "label": self._part_instance_display_label(instance, instances),
            })
        candidates.sort(key=lambda item: (str(item["label"]).lower(), str(item["slot_path"])))
        return candidates

    def _explicit_side_pair_label(self, object_id: str, peers: list[str]) -> str:
        part_id, slot_path = self._split_side_pair_ref(object_id)
        if slot_path:
            for candidate in self._side_pair_instance_candidates():
                if candidate["ref"] == object_id:
                    return str(candidate["label"])
            numbered = self._mesh_instance_number_for_ref(object_id)
            if numbered is not None:
                return self._mesh_instance_label_for_ref(object_id, peers)
            object_id = part_id
        if object_id in peers and self.context is not None and object_id in self.context.objects:
            return self._part_option_label(object_id, peers)
        if self.context is not None:
            found = core.part_body_for_context(self.context, object_id)
            if found is not None:
                info_name = core.part_information_name(found[0])
                if info_name:
                    return f"{info_name} ({object_id})"
        return object_id

    def _part_instances_for_focused_config(self) -> list[dict[str, object]]:
        if self.context is None:
            return []
        config_name = self._focused_slot_config()
        if config_name is None:
            return []
        try:
            selected = core.selected_parts_for_config(self.context, config_name)
        except Exception:
            return []
        return [dict(instance) for instance in core.selected_part_instances(selected)]

    def _side_pair_rows(self) -> list[dict[str, object]]:
        instances = self._part_instances_for_focused_config()
        by_slot: dict[str, list[dict[str, object]]] = {}
        for instance in instances:
            slot_id = str(instance.get("slot_id") or "")
            if not slot_id:
                continue
            by_slot.setdefault(slot_id, []).append(instance)

        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        usage = self._slot_usage()
        for slot_id in sorted(by_slot):
            partner_slot = core.mirror_lateral_node_id(slot_id)
            if partner_slot == slot_id:
                continue
            pair_key = tuple(sorted((slot_id, partner_slot)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            left_slot, right_slot = sorted(pair_key)
            left_instances = by_slot.get(left_slot, [])
            right_instances = by_slot.get(right_slot, [])
            text_values = [left_slot, right_slot]
            text_values.extend(str(instance.get("part_id") or "") for instance in left_instances + right_instances)
            text_values.extend(str(instance.get("slot_path") or "") for instance in left_instances + right_instances)
            kind = self._side_pair_kind(*text_values)
            if not kind:
                continue
            left = left_instances[0] if left_instances else None
            right = right_instances[0] if right_instances else None
            left_part = str(left.get("part_id") or "") if left is not None else ""
            right_part = str(right.get("part_id") or "") if right is not None else ""
            left_ref = self._side_pair_ref(left_part, str(left.get("slot_path") or "")) if left is not None else ""
            right_ref = self._side_pair_ref(right_part, str(right.get("slot_path") or "")) if right is not None else ""
            plan = self._slot_pair_plan_label(left_slot, right_slot)
            if not plan and left is not None and right is not None:
                left_part = str(left.get("part_id") or "")
                right_part = str(right.get("part_id") or "")
                if left_part == right_part:
                    plan = "Same part"
                else:
                    plan = "Review"
            if not plan:
                plan = "Move empty side"
            active = core.slot_pair_partner(self.conversion, left_slot) == right_slot
            rows.append({
                "id": f"{left_slot}||{right_slot}",
                "label": f"{left_slot} <-> {right_slot}",
                "left": self._part_instance_display_label(left, instances),
                "right": self._part_instance_display_label(right, instances),
                "kind": kind,
                "slots": f"{left_slot} / {right_slot}",
                "plan": f"Applied: {plan}" if active else plan,
                "active": "Y" if active else "N",
                "left_id": left_ref,
                "right_id": right_ref,
                "slot_a": left_slot,
                "slot_b": right_slot,
            })
        explicit_pairs = core.normalized_side_pairs(self.conversion.get("sidePairs"))
        explicit_ids = [
            str(pair.get(field) or "")
            for pair in explicit_pairs
            for field in ("left", "right")
            if pair.get(field)
        ]
        seen_explicit: set[tuple[str, str]] = set()
        for pair in explicit_pairs:
            left = str(pair["left"])
            right = str(pair["right"])
            key = tuple(sorted((left, right)))
            if key in seen_explicit:
                continue
            seen_explicit.add(key)
            rows.append({
                "id": f"sidepair||{left}||{right}",
                "label": f"{left} <-> {right}",
                "left": self._explicit_side_pair_label(left, explicit_ids),
                "right": self._explicit_side_pair_label(right, explicit_ids),
                "slots": "equivalent parts",
                "plan": "Equivalent",
                "active": "Y",
                "left_id": left,
                "right_id": right,
                "slot_a": "",
                "slot_b": "",
            })
        rows.sort(key=lambda row: (str(row["kind"]), str(row["label"])))
        return rows

    def _refresh_slots(self) -> None:
        with timed_ui("_refresh_slots"):
            if hasattr(self, "slot_tree"):
                keep = set(self.slot_tree.selection())
                previous_order = list(self.slot_tree.get_children(""))
                for item in self.slot_tree.get_children():
                    self.slot_tree.delete(item)
            else:
                keep = set()
                previous_order = []
            self.side_pair_rows_by_iid = {}
            if self.context is None:
                return
            query = self.slot_filter_var.get().strip().lower()
            context = self._side_pair_table_context()
            rows = self._side_pair_table_rows(context)
            displayed: list[str] = []
            row_index = 0
            for row in rows:
                values = (
                    row["left"],
                    row["right"],
                )
                if (
                    query
                    and all(query not in str(value).lower() for value in values)
                ):
                    continue
                row_id = str(row["id"])
                displayed.append(row_id)
                self.side_pair_rows_by_iid[row_id] = row["pair"]
                self.slot_tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    tags=self._row_tags(row_index),
                    values=values,
                )
                row_index += 1
            self.current_slot_ids = displayed
            self._restore_tree_order(self.slot_tree, previous_order)
            # 2026-09-07: a pick used to be a half-filled row waiting on a cell,
            # so a rebuild had to re-stamp its prompt or drop the pick when the
            # row went. New holds the prompt in the status bar and no row until
            # both sides are chosen, so a rebuild has nothing to say to it.
            visible_keep = [item for item in keep if self.slot_tree.exists(item)]
            if visible_keep:
                self.slot_tree.selection_set(visible_keep)

    def _side_pair_pair_key(self, left: object, right: object) -> tuple[str, str]:
        return tuple(sorted((str(left or ""), str(right or ""))))

    def _side_pair_table_context(self) -> dict[str, object]:
        source_candidates = self._side_pair_source_candidates()
        known_source_refs = {str(candidate["ref"]) for candidate in source_candidates}
        for pair in core.normalized_side_pairs(self.conversion.get("sidePairs")):
            for field in ("left", "right"):
                ref = str(pair.get(field) or "")
                if not ref or ref in known_source_refs:
                    continue
                part_id = self._side_pair_ref_part_id(ref)
                source_candidates.append({
                    "ref": ref,
                    "part_id": part_id,
                    "slot_path": self._split_side_pair_ref(ref)[1],
                    "label": self._explicit_side_pair_label(ref, [ref]),
                })
                known_source_refs.add(ref)
        value_by_label, label_by_value = self._side_pair_source_option_maps(source_candidates)
        return {
            "source_candidates": source_candidates,
            "value_by_label": value_by_label,
            "label_by_value": label_by_value,
        }

    def _side_pair_label_for(self, value: object, context: dict[str, object]) -> str:
        text = str(value or "")
        if not text:
            return "(choose)"
        label_by_value = context.get("label_by_value", {})
        if isinstance(label_by_value, dict) and text in label_by_value:
            return str(label_by_value[text])
        return self._mesh_instance_label_for_ref(text, self._side_pair_mesh_candidates())

    def _side_pair_kind_label(self, value: object) -> str:
        kind = str(value or "part").lower()
        return SIDE_PAIR_KIND_LABELS.get(kind, "Part")

    def _side_pair_kind_value(self, label: str) -> str:
        return SIDE_PAIR_KIND_BY_LABEL.get(label, "part")

    def _side_pair_table_rows(self, context: dict[str, object]) -> list[dict[str, object]]:
        table_pairs = core.normalized_side_pairs(self.conversion.get("sidePairs"))
        rows: list[dict[str, object]] = []
        for index, pair in enumerate(table_pairs):
            if not isinstance(pair, dict):
                continue
            rows.append({
                "id": f"pair_{index}",
                "pair": pair,
                "kind": self._side_pair_kind_label(pair.get("kind")),
                "left": self._side_pair_label_for(pair.get("left"), context),
                "right": self._side_pair_label_for(pair.get("right"), context),
            })
        return rows

    def _slot_pair_from_row_id(self, row_id: str) -> tuple[str, str] | None:
        if "||" not in row_id:
            return None
        left, right = row_id.split("||", 1)
        if not left or not right:
            return None
        return left, right

    def _slot_click(self, event: tk.Event) -> str | None:
        """Select the clicked row and nothing else.

        2026-09-07: a click on the Left or Right cell used to arm a part pick,
        so the click that meant "select this row so Remove can take it" put the
        row back into editing instead. Rows are made by New and taken away by
        Remove; the table itself only ever reports.
        """
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.slot_tree, event):
            return None
        item = self.slot_tree.identify_row(event.y)
        if not item:
            return None
        self.slot_tree.selection_set([item])
        self.slot_tree.focus(item)
        return "break"

    def _clear_side_pair_pick_target(self) -> None:
        self.side_pair_pick_target = None

    def _start_new_side_pair(self) -> None:
        """New: pick a part, then name its opposite number in the modal.

        2026-09-07: this used to drop a blank row into the table and wait for
        two separate part picks to fill its cells. A row with neither side
        chosen states nothing, and it made every later click on the table
        ambiguous between selecting and editing, so a pair is now assembled
        off-table and only lands once both sides are known.
        """
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        self.side_pair_pick_target = {"mode": "new"}
        self.status_var.set(
            "Equivalent Parts: click a row in Mesh Transforms or a mesh in the preview"
        )

    def _commit_side_pair_part_pick_from_row(self, row_id: object) -> bool:
        """A part was clicked while New was waiting for one.

        Answers whether the click was consumed, not whether a row was written:
        the modal opens on a timer once the click has finished being handled.
        """
        if not isinstance(getattr(self, "side_pair_pick_target", None), dict):
            return False
        self._clear_side_pair_pick_target()
        ref = ""
        if hasattr(self, "part_tree") and self.part_tree.exists(row_id):
            ref = self._part_row_side_ref(row_id) or self._part_row_mesh_id(row_id)
        if not ref:
            self.status_var.set("Equivalent part picker cancelled")
            return False
        ref = str(ref)
        self.after(SIDE_PAIR_MODAL_DELAY_MS, lambda: self._choose_and_commit_side_pair(ref))
        return True

    def _side_pair_partner_ref(self, ref: str) -> str:
        """The other side of the row this ref already sits on, if any."""
        for pair in self._side_pair_current_pairs():
            if str(pair.get("left") or "") == ref:
                return str(pair.get("right") or "")
            if str(pair.get("right") or "") == ref:
                return str(pair.get("left") or "")
        return ""

    def _side_pair_ref_position(self, ref: str) -> tuple[float, float, float] | None:
        """Where a ref sits, for a ref the candidate list does not carry."""
        if self.context is None:
            return None
        obj = self.context.objects.get(self._side_pair_ref_part_id(ref))
        if obj is None:
            return None
        return (float(obj.x), float(obj.y), float(obj.z))

    @staticmethod
    def _mirrored_position_ref(position: object, candidates: list[dict[str, object]]) -> str:
        """The candidate closest to where this one's reflection would be.

        The same score the Swap Mesh geometry test uses -- distance from the
        reflection, lateral weighted four times the fore/aft and vertical drift
        -- but over instance positions rather than mesh positions, because two
        Equivalent Parts rows are routinely two instances of one mesh and a
        mesh-level position cannot tell those apart.
        """
        if not isinstance(position, (tuple, list)) or len(position) < 3:
            return ""
        x, y, z = (float(position[0]), float(position[1]), float(position[2]))
        best: tuple[float, str] | None = None
        for candidate in candidates:
            other = candidate.get("position")
            if not isinstance(other, (tuple, list)) or len(other) < 3:
                continue
            other_x, other_y, other_z = (float(other[0]), float(other[1]), float(other[2]))
            if abs(x) > 0.02 and abs(other_x) > 0.02 and x * other_x > 0:
                continue  # both off-centre on the same side, so not a counterpart
            ref = str(candidate.get("ref") or "")
            if not ref:
                continue
            score = abs(x + other_x) * 4.0 + abs(y - other_y) + abs(z - other_z)
            if best is None or score < best[0]:
                best = (score, ref)
        return best[1] if best is not None else ""

    def _suggest_side_pair_counterpart(
        self,
        subject: dict[str, object],
        candidates: list[dict[str, object]],
    ) -> str:
        """The part most likely to be the picked one's opposite number.

        Name first -- an _FL mesh names its _FR -- then position, which is what
        answers the case a name cannot: one mesh fitted twice, once per side.
        """
        mesh_id = str(subject.get("mesh_id") or subject.get("part_id") or "")
        position = subject.get("position")
        mesh_ids = sorted(
            {
                str(candidate.get("mesh_id") or candidate.get("part_id") or "")
                for candidate in candidates
            }
            - {"", mesh_id}
        )
        named = self._name_pair_candidate(mesh_id, mesh_ids) if mesh_id and mesh_ids else None
        if named:
            pool = [
                candidate
                for candidate in candidates
                if str(candidate.get("mesh_id") or candidate.get("part_id") or "") == named
            ]
            if len(pool) == 1:
                return str(pool[0].get("ref") or "")
            mirrored = self._mirrored_position_ref(position, pool)
            if mirrored:
                return mirrored
            if pool:
                return str(pool[0].get("ref") or "")
        return self._mirrored_position_ref(position, candidates)

    def _choose_and_commit_side_pair(self, ref: str) -> bool:
        """Ask which part answers ``ref``, then write the row."""
        context = self._side_pair_table_context()
        source_candidates = context.get("source_candidates", [])
        if not isinstance(source_candidates, list):
            source_candidates = []
        by_ref = {
            str(candidate.get("ref") or ""): candidate
            for candidate in source_candidates
            if isinstance(candidate, dict)
        }
        subject = by_ref.get(ref) or {
            "ref": ref,
            "mesh_id": self._side_pair_ref_part_id(ref),
            "position": self._side_pair_ref_position(ref),
        }
        candidates = [
            candidate
            for candidate in source_candidates
            if isinstance(candidate, dict)
            and str(candidate.get("ref") or "")
            and str(candidate.get("ref")) != ref
        ]
        if not candidates:
            self._show_error("Equivalent Parts", "No other part is available to pair with.")
            return False
        value_by_label, _label_by_value = self._side_pair_source_option_maps(candidates)
        existing = self._side_pair_partner_ref(ref)
        partner = self._choose_counterpart_modal(
            title="Equivalent Parts",
            subject_heading="Part",
            subject_text=self._side_pair_label_for(ref, context),
            choice_heading="Equivalent To",
            value_by_label=value_by_label,
            suggested=self._suggest_side_pair_counterpart(subject, candidates),
            initial=existing,
            subject_value=ref,
            empty_choice_message="Select the part on the other side.",
            self_choice_message="Equivalent parts need two different entries.",
        )
        if not partner:
            self.status_var.set("Equivalent part picker cancelled")
            return False
        kind = self._default_side_pair_kind(
            str(subject.get("mesh_id") or self._side_pair_ref_part_id(ref)),
            str(by_ref.get(partner, {}).get("mesh_id") or self._side_pair_ref_part_id(partner)),
        )
        return self._commit_side_pair(
            {"left": ref, "right": partner, "kind": kind.lower()},
            action="Updated" if existing else "Added",
        )

    def _side_pair_current_pairs(self) -> list[dict[str, object]]:
        return core.normalized_side_pairs(self.conversion.get("sidePairs"))

    def _select_side_pair_row(self, left: str, right: str) -> None:
        key = self._side_pair_pair_key(left, right)
        rows_by_iid = getattr(self, "side_pair_rows_by_iid", {})
        if not isinstance(rows_by_iid, dict):
            return
        for row_iid, row in rows_by_iid.items():
            if self._side_pair_pair_key(row.get("left"), row.get("right")) == key:
                self.slot_tree.selection_set([row_iid])
                self.slot_tree.focus(row_iid)
                self.slot_tree.see(row_iid)
                return

    def _commit_side_pair(
        self,
        pair: dict[str, object],
        *,
        action: str = "Updated",
    ) -> bool:
        """Write one finished pair.

        No old-row bookkeeping: ``set_side_pair`` drops every row either side
        already sits on, so re-pairing a part replaces its row rather than
        leaving the part on two.
        """
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return False
        left = str(pair.get("left") or "")
        right = str(pair.get("right") or "")
        if not left or not right:
            self._show_error("Equivalent Parts", "Choose both parts.")
            return False
        if left == right:
            self._show_error("Equivalent Parts", "Equivalent parts need two different entries.")
            return False
        core.set_side_pair(
            self.conversion,
            left,
            right,
            kind=str(pair.get("kind") or "part"),
        )
        self._refresh_parts()
        self._refresh_slots()
        self._update_detail()
        self._select_side_pair_row(left, right)
        self.status_var.set(
            f"{action} equivalent parts: {self._explicit_side_pair_label(left, [left, right])} <-> "
            f"{self._explicit_side_pair_label(right, [left, right])}"
        )
        return True

    def _remove_selected_side_pair(self) -> None:
        self._clear_side_pair_pick_target()
        selection = self.slot_tree.selection()
        iid = str(selection[0]) if selection else str(self.slot_tree.focus() or "")
        if not iid:
            return
        rows_by_iid = getattr(self, "side_pair_rows_by_iid", {})
        pair = rows_by_iid.get(iid) if isinstance(rows_by_iid, dict) else None
        if not isinstance(pair, dict):
            return
        left = str(pair.get("left") or "")
        right = str(pair.get("right") or "")
        old_children = list(self.slot_tree.get_children())
        old_index = old_children.index(iid) if iid in old_children else 0
        self.conversion["sidePairs"] = [
            existing
            for existing in self._side_pair_current_pairs()
            if self._side_pair_pair_key(existing.get("left"), existing.get("right"))
            != self._side_pair_pair_key(left, right)
        ]
        self._refresh_parts()
        self._refresh_slots()
        self._update_detail()
        next_children = list(self.slot_tree.get_children())
        if next_children:
            next_iid = next_children[min(old_index, len(next_children) - 1)]
            self.slot_tree.selection_set([next_iid])
            self.slot_tree.focus(next_iid)
            self.slot_tree.see(next_iid)
        else:
            self.slot_tree.selection_set([])
        context = self._side_pair_table_context()
        self.status_var.set(
            f"Removed equivalent parts: {self._side_pair_label_for(left, context)} <-> "
            f"{self._side_pair_label_for(right, context)}"
        )

    def _focus_side_pair_table_shortcut(self, event: tk.Event) -> str | None:
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in {"Entry", "TEntry", "TCombobox", "Text"}:
            return None
        if hasattr(self, "slot_tree"):
            self.slot_tree.focus_set()
            children = self.slot_tree.get_children()
            if children and not self.slot_tree.selection():
                self.slot_tree.selection_set([children[0]])
                self.slot_tree.focus(children[0])
        return "break"

    def _set_slot_partner(self, slot_type: str, partner: str) -> None:
        core.set_slot_pair(self.conversion, slot_type, partner.strip())
        self._refresh_slots()
        self._update_detail()
        if partner:
            self.status_var.set(
                f"Paired {slot_type} with {partner} "
                f"({self._slot_pair_plan_label(slot_type, partner) or 'no change'} on this trim)"
            )
        else:
            self.status_var.set(f"Unpaired {slot_type}")

    def _clear_slot_pairs(self) -> None:
        self._clear_side_pair_pick_target()
        self.conversion["slotPairs"] = []
        core.clear_side_pairs(self.conversion)
        self._refresh_slots()
        self._update_detail()
        self.status_var.set("Cleared equivalent parts")

    def _side_pair_mesh_candidates(self) -> list[str]:
        if self.context is None:
            return []
        candidates = [
            self._part_row_mesh_id(object_id)
            for object_id in (self.resolved_part_ids or self.current_part_ids)
            if self._part_row_mesh_id(object_id) in self.context.objects
        ]
        candidates = sorted(set(candidates))
        if candidates:
            return candidates
        return sorted(self.context.objects)

    def _focused_mesh_instance_candidates(self) -> list[dict[str, object]]:
        if self.context is None or not hasattr(self, "part_tree"):
            return []
        label_universe = self._side_pair_mesh_candidates()
        candidates: list[dict[str, object]] = []
        # 2026-09-07: every row the table holds, not just the ones its filter is
        # showing. The Equivalent Parts modal offers this list, and the filter
        # typed to find the first part would otherwise hide its counterpart.
        for row_id in getattr(self, "part_instance_rows", {}):
            mesh_id = self._part_row_mesh_id(row_id)
            if mesh_id not in self.context.objects:
                continue
            row = getattr(self, "part_instance_rows", {}).get(row_id, {})
            candidates.append({
                "ref": self._part_row_side_ref(row_id),
                "part_id": mesh_id,
                "mesh_id": mesh_id,
                "slot_path": str(row.get("slot_path") or ""),
                "slot_id": str(row.get("slot_id") or ""),
                "owner_part_id": str(row.get("part_id") or ""),
                "kind": self._default_side_pair_kind(mesh_id, ""),
                "label": self._part_row_label(str(row_id), mesh_id, label_universe),
                "position": row.get("position"),
            })
        if candidates:
            return candidates
        if self.context is None:
            return []
        config_name = self._focused_slot_config()
        if config_name is None:
            return []
        try:
            instances = core.selected_mesh_transform_instances_for_config(self.context, config_name)
        except Exception:
            return []
        vehicle_numbering = self._vehicle_mesh_instance_numbering()
        candidates: list[dict[str, object]] = []
        for instance in instances:
            mesh_id = instance.mesh_id
            base_label = self._part_display_label(mesh_id, self._side_pair_mesh_candidates())
            label = base_label
            number_key = self._mesh_instance_number_key(instance)
            mesh_numbering = vehicle_numbering.get(mesh_id, {})
            display_count = len(mesh_numbering) if mesh_numbering else instance.count_for_mesh
            display_ordinal = mesh_numbering.get(number_key, instance.ordinal_for_mesh)
            if display_count > 1:
                label = f"{label} #{display_ordinal}"
            candidates.append({
                "ref": instance.instance_id,
                "part_id": mesh_id,
                "mesh_id": mesh_id,
                "slot_path": instance.slot_path,
                "slot_id": instance.slot_id,
                "owner_part_id": instance.part_id,
                "kind": self._default_side_pair_kind(mesh_id, ""),
                "label": label,
                "position": instance.position,
            })
        candidates.sort(
            key=lambda item: (
                str(item.get("kind") or ""),
                str(item.get("label") or "").lower(),
                str(item.get("slot_path") or ""),
            )
        )
        return candidates

    def _side_pair_source_candidates(self) -> list[dict[str, object]]:
        candidates = self._focused_mesh_instance_candidates()
        if candidates:
            return candidates
        return [
            {
                "ref": object_id,
                "part_id": object_id,
                "mesh_id": object_id,
                "slot_path": "",
                "kind": self._default_side_pair_kind(object_id, ""),
                "label": self._part_display_label(object_id, self._side_pair_mesh_candidates()),
            }
            for object_id in self._side_pair_mesh_candidates()
        ]

    def _side_pair_source_option_maps(
        self,
        candidates: list[dict[str, object]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        value_by_label: dict[str, str] = {}
        label_by_value: dict[str, str] = {}
        label_counts: dict[str, int] = {}
        for candidate in candidates:
            label = str(candidate["label"])
            ref = str(candidate["ref"])
            label_counts[label] = label_counts.get(label, 0) + 1
            label_by_value[ref] = label
            if label in value_by_label:
                label = f"{label} #{label_counts[label]}"
            value_by_label[label] = ref
        return value_by_label, label_by_value

    def _default_side_pair_kind(self, left: str, right: str) -> str:
        kind = self._side_pair_kind(left, right)
        return kind if kind in {"Seat", "Mirror", "Door"} else "Part"

    def _open_side_pair_modal_shortcut(self, event: tk.Event) -> str | None:
        return self._focus_side_pair_table_shortcut(event)

    def _open_side_pair_modal(self) -> None:
        self._focus_side_pair_table_shortcut(None)
