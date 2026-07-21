# Handover: agreed next changes to the spatial Recommend Modes classifier

## Status at handover

The spatial classifier is in its **last validated state** — all 90 tests pass;
per-trim agreement vs the hand-verified baselines is etk800 **99.98 %**, pickup
**98.19 %**, sunburst2 **94.37 %**. The five changes below were designed and
signed off with the user but **not yet implemented**. This document is the spec to implement
them cleanly from here.

Code lives in `beamng_hand_drive_core.py` (pure geometry) and
`beamng_hand_drive_tool.py` (`_classify_meshes_for_trim`, `_resolve_trim_pairs`,
`_inherit_mounted_parts`, `build_mode_recommendations`). Method reference:
`docs/SPATIAL_CLASSIFIER_WRITEUP.md`.

---

## The evidence that drove these decisions (don't re-derive)

We measured symmetry residuals on the real clouds, **capped 350-point preview
vs full DAE vertex set**, and found the current self-symmetry test is measuring
an artifact:

- `sample_points` (in `beamng_transform_helpers.py`) is
  `points[::stride][:max_points]` — strided **and** truncated. On a symmetric
  mesh this breaks mirrored vertex pairs and can drop a contiguous chunk of one
  half, so a **truly symmetric mesh reads wildly asymmetric from its capped
  cloud**.
- Full-vertex residuals are decisive and clean. Truly symmetric parts
  (headliners, benches, seats_R, carpet, sunvisors, `facelift_seats_F`) come
  back **exactly 0.0** (worst stray: etk800 `seats_R` at 0.7 cm — one vertex of
  the centre headrest, a dev modelling slip, and not a transform candidate
  anyway). Genuinely one-sided detail (`shifter_M` bent lever, `shifter_T`,
  `grp_mirror` arm, `desert_cage_dashseals`, `console`, `rollcage` door bar)
  shows **hundreds of orphan vertices at 4–17 cm**.
- Cross-mesh **pair** residuals are unaffected (twin meshes have mirrored
  buffers, so stride truncates both the same way — this is why pairing has been
  reliable while self-symmetry lied). **Only the self-symmetry test needs the
  full vertex set.**

Conclusion the user endorsed: **decide self-symmetry on the full vertex set with
a zero threshold** (skip only truly symmetric meshes; everything else mirrors).
The user explicitly prefers being trigger-happy (benign extra mirrors) over
missing a real asymmetry. Note that Mirror generates a mirrored mesh copy in the
build, so etk800's agreement will drop as its deliberately-skipped-symmetric
parts stay correctly skipped (they measure 0.0) — actually etk800 should hold,
because the zero-threshold keeps the true-symmetric parts skipped; the flips are
the shifters/seals/console/etc. that the baseline mirrors anyway.

I also **fabricated physical explanations** ("dome light bosses", "sunroof
cutout", "60/40 split") for the capped-cloud phantom tails before verifying —
those were wrong; the cause is purely the sampler truncation above. The
write-up's symmetry section must be corrected to say so.

---

## Change 1 — Full-vertex, zero-threshold self-symmetry  *(highest value, do first)*

**Core additions:**

- `reflected_orphan_stats(points, center_x, exact_tol=1e-4, coarse_tol=0.02) ->
  (orphans, coarse_fraction)`: reflect across `x=center_x`; for every vertex,
  is there a reflected vertex within tol? `orphans` = count with none within
  `exact_tol` (0.1 mm — float dust of a true mirror pair); `coarse_fraction` =
  fraction with none within `coarse_tol` (for confidence grading only).
  Deterministic, no sampling. A voxel-hash membership test (round coords to
  `tol`, put reflected set in a `set`, probe the 2×2×2 neighbourhood of each
  query cell) is exact enough and linear; verify by re-running the earlier
  measurement script if in doubt.
- `full_vertex_clouds_for_ids(context, ids) -> {id: ndarray}`: parse each mesh's
  DAE **once** with `preview_data_from_tree(..., max_points_per_object=huge)` so
  no cap applies; **follow `obj.dae_source_zip or context.source_zip`** (shared
  accessories like `dino` live in common packs, not the vehicle zip); align the
  authored cloud onto the placed preview centre (translation only — jbeam-placed
  shared meshes are authored at origin); cache on the context
  (`_full_clouds` + a parsed-file set); on any parse failure fall back to the
  preview cloud (degrades toward "asymmetric" = a benign Mirror).

**Tool wiring** (in `_classify_meshes_for_trim`, replacing the
`cloud_symmetry_residual` block):

- Add a small `_mesh_symmetry(context, object_id, points, center_x)` wrapper
  that pulls the full cloud, shifts it by this trim's x offset (compare the
  passed `points` bbox-centre-x to `preview_by_id[id]["center"][0]`), and caches
  the `(orphans, coarse_fraction)` per `(id, shift_x, center_x)`.
- Replace `residual < SPATIAL_SYMMETRIC_RESIDUAL` → **`orphans == 0`** for the
  skip/fascia/display branch. Replace the borderline confidence
  (`residual < SPATIAL_BORDERLINE_RESIDUAL`) with `coarse_fraction < 0.05`.
  Replace the "barely-seen centred blob" guard's `residual < 0.15` with
  `coarse_fraction < 0.15`.
- Delete `SPATIAL_SYMMETRIC_RESIDUAL` / `SPATIAL_BORDERLINE_RESIDUAL`.
  `cloud_symmetry_residual` in core can stay (still used nowhere else) or be
  removed — check for other callers first.

**Pitfall found last attempt:** get the `center_x` argument threaded through
consistently (the classifier uses `cx0 = frame.center_x`). Don't reference a
non-existent `context._center_x`.

---

## Change 2 — Driver-seat transparency + under-seat base pairing

The eye sits *in* the driver seat, so the seat occludes its own base. Treat the
driver seat like the wheel: **add it to the `transparent` set** in
`_classify_meshes_for_trim` so the scan sees through it.

- Find the seat **geometrically, not by name**: a large mesh (diag ≥ ~0.5 m)
  with a meaningful fraction (≥ ~20 %) of its points directly under the eye
  (|x−eye_x|<0.25, |y−eye_y|<0.35, eye_z−0.75 < z < eye_z−0.10).
- Under-seat bases then enter scope and, where L/R bases exist as separate
  meshes, pair structurally. The existing build fallback
  (`core.fallback_structural_part_modes`) already demotes a structural part to
  aesthetic Mirror when its twin is absent in a trim, so **seat-delete /
  single-seater trims put the lone base on the correct side via Mirror, never
  skip**. Verify this end-to-end on a pickup single-seat trim.

Targets this should rescue: sunburst2 `racingseat_base`, pickup
`racing_seat_base`.

---

## Change 3 — Passenger-footwell cone (blind-spot admission)

The driver eye can't see the passenger footwell, so parts there (e.g. sunburst2
`grp_footplate`, the passenger footrest) are missed. Add **one 30° FOV cone**
from the eye, aimed at the **mirror of the translate-classified control cluster**:

- After a trim's meshes are classified, compute the centroid of all meshes
  verdict==`translate` that sit **below the wheel** (the pedal box etc.), then
  **reflect that centroid across the centreline** → the aim direction.
- Any mesh whose centroid lies within 30° of that axis (and within cabin range)
  becomes a **forced candidate**: it bypasses the scope-admission gate (thread a
  `forced: frozenset` into `_classify_meshes_for_trim`, OR it into the
  `cand_*` disjunction, and exempt it from the y-front veto) and then flows
  through the **normal** symmetry / cone / pair logic — forcing *candidacy* only,
  never the verdict.
- This needs the translate set, so it's a **second pass**: classify once, derive
  the cone, re-classify the still-unclassified meshes with `forced` populated.
  Keep it cheap (only meshes with no verdict yet).

User's exact words: "30 degree fov", aim at "the average position of translated
meshes/props/etc positioned below the steering wheel but mirrored onto the other
side".

---

## Change 4 — Sightline inheritance for floating meshes

A scoped-in mesh that is **floating** (unattached — fails the 3 cm contact test
in `_inherit_mounted_parts`) but sits in the driver's line of sight *in front of*
a transformed mesh should inherit Mirror.

- Rule (user-confirmed, generalised): for a floating in-scope mesh, take the
  eye→point directions it occupies; if those directions **continue on and
  terminate on any mesh with a non-skip verdict** (translate, mirror, OR
  structural — *any* transform), **at any distance**, the floater inherits
  Mirror. Rationale: every transformed mesh is cabin/mirror furniture, so
  something in front of one, in the driver's view, is almost certainly cabin too.
- Cheap to compute with the existing angular-bin machinery in
  `core.visibility_scan` — it already finds, per direction, what's nearest and
  what's behind; extend it (or add a sibling) to report the **verdict class** of
  the mesh behind, not just its existence. Run this as an extra pass in/after
  `_inherit_mounted_parts`.
- **Post-handover correction:** only points in the driver's forward 180°
  hemisphere participate. The earlier unrestricted directions incorrectly
  mirrored the ETK rear-window third brake light through rearward bodywork.
- Targets: laptop-mount poles (sunburst2 `police_laptop_mount_*`) and similar
  floaters between the dash and other transformed meshes.

---

## Change 5 — Functionally-sided pairs (mirror signals)  *(classifier: skip-with-reason now; build fix later)*

`sunburst2_mirrorsignal_L/R` and `generic_flasher_led_*` are geometric twins the
classifier *would* pair, but the baseline skips them **on purpose**: the flashing
direction is encoded in the **material** (`sunburst2_signal_l` vs `signal_r`), so
swapping the meshes without swapping the materials makes the indicators sweep the
wrong way.

- **Detector (name-free):** a structural pair whose two members bind **different
  material symbols** is functionally sided. `core.mesh_material_symbols` already
  gives per-mesh symbols.
- **Now:** in `_resolve_trim_pairs`, when a candidate pair's members have
  differing material symbols, **do not pair** — leave both `skip` with an
  explicit reason ("functionally sided: materials differ, needs build-side
  material rebind"). Confirm with the user whether they'd rather it emit a
  low-confidence Mirror instead of skip.
- **Later (build side, NOT this round unless asked):** in `generate_daes`, when
  swapping such a pair, **rebind each generated mesh to its twin's materials**
  (L housing keeps the L-flashing material) so the pair can be safely paired.
  That is the real fix the user wants eventually; the classifier change above is
  the safe interim.

---

## Decisions locked (do NOT reopen)

- **Superseded after handover:** generic driver-visible admission is restricted
  to the forward 180° hemisphere. The 360° shell remains for enclosure and
  backing, but a rearward mesh cannot enter scope from visibility alone. The
  ETK baseline's paired rear door cards were confirmed as a baseline mistake.
- **Shifter/nav/console furniture** should all Mirror under the spatial method;
  the sunburst2-vs-etk800 baseline disagreement there is a baseline
  inconsistency, not a bug. Don't tune toward either baseline. The user will
  revisit the auto-gate-asymmetry nuance separately, later.
- **`pickup_fueltank_short` / `pickup_shocktop_R_offroad`** low-confidence leaks:
  benign, leave them.
- The structural→aesthetic-Mirror fallback for a twin-absent trim already exists
  on both the recommender side (per-trim pairing emits Mirror "twin absent") and
  the build side (`fallback_structural_part_modes`). The per-trim "pair vs
  mirror" rows in the old diff tables were a **validation-script artifact**
  (script called the recommender per-trim); the global recommendation is already
  the paired verdict. Verify, don't re-engineer.

---

## Validation protocol (unchanged)

Scratch validators used last session live under the session scratchpad; the
pattern:

- `python -m unittest discover -s tests` — must stay green (90 tests).
- Run `python scripts/validate_spatial_classifier.py`. It loads each cached
  context, obtains the global recommendation once, applies structural pairs
  per trim, excludes the desired aesthetic fallback when a twin is absent,
  and diffs modes against `conversion.json`; pairs count BOTH members. Projects at
  `C:\Users\ashle\AppData\Local\BeamXP\handedness_conversion_projects\{etk800,
  pickup,sunburst2}`.
- Watch for regressions the last attempt hit: SPOTLIGHT-type sub-resolution
  dummies must stay excluded from any new inheritance; underbody parts
  (`axlebrace`) must not chain in via sightline/cone — bound everything by the
  cabin radius (~1.6 m of the eye).
- After each change, re-diff all three vehicles and **categorise** every new
  disagreement before moving on (right-by-the-brief vs genuine miss).

## Then

Update `docs/SPATIAL_CLASSIFIER_WRITEUP.md` §3 (symmetry) with the corrected
full-vertex method and the sampler-truncation cause (strike the fabricated
physical explanations), refresh the agreement numbers, and update the memory
file `beamxp-test-environment.md`.
