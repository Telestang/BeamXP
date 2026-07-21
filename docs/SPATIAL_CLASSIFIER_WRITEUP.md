# Recommend Modes: the eye-anchored spatial classifier

`build_mode_recommendations` no longer reasons from part names. It builds a
**driver eye-frame** from the jbeam internal camera, sweeps a
**nearest-surface shell** around that eye to scope the interior, and assigns
each mesh a mode from geometric evidence: visibility, backing layers,
self-symmetry, twin geometry, and an ergonomic control cone. Names appear in
exactly one place — the steering-column resolution-floor hint — and that use
is explicit, confined, and always low-confidence.

Pure geometry lives in `beamng_hand_drive_core.py` (`DriverFrame`,
`driver_frame_for_context`, `visibility_scan`, `floor_height_from_shell`,
`glass_beyond_fractions`, `cloud_symmetry_residual`, `mirror_pair_residual`,
`principal_extent_sds`, `material_flags_for_context`,
`mesh_material_symbols`). Orchestration lives in `beamng_hand_drive_tool.py`
(`_classify_meshes_for_trim`, `_resolve_trim_pairs`,
`build_mode_recommendations`).

## 1. The eye-frame (Step 0)

The interior is defined by the driver's viewpoint, so the classifier anchors
on the **internal driver camera**: every `["dash"|"driver", x, y, z, fov, …]`
row in any `camerasInternal` array is parsed (bracket-matched from the raw
jbeam; literal coordinates, with node references resolved through
`node_positions` as a fallback), and the eye **E** is the component-wise
median. The steering wheel — located with the already-sanctioned
`steering_ref_score` helpers — corroborates the frame and fixes the forward
direction **f** = horizontal(E→wheel). The wheel centre is estimated
robustly from the cloud points *near and below the eye*, because wheel meshes
sometimes include the whole steering shaft (the D-series wheel cloud spans a
metre of column). The driver side is `sign(E_x − median node x)`.

Degradation is explicit: camera without wheel keeps BeamNG's −y-forward
convention; wheel without camera estimates the eye 0.60 m behind and 0.35 m
above the rim and marks the frame degraded; **neither camera nor wheel means
no spatial frame, and the classifier emits nothing** — never a name-based
guess.

Measured frames: etk800 E=(0.40, 0.29, 1.19), pickup E=(0.462, 0.10, 1.53),
sunburst2 E=(0.374, 0.18, 1.09). The 44 cm spread in eye height between the
truck and the hatchback is why every threshold downstream is eye- or
shell-relative rather than absolute.

## 2. Interior scope: candidates, not absolutes (Step 1)

`visibility_scan` bins every present mesh's sample points into 6° equal-angle
(elevation, azimuth) bins around E and takes the per-bin nearest opaque point
as the **shell**. Steering-scored meshes are transparent (so the cluster,
stalks and pedals behind the wheel are reachable). Per mesh it reports:

- `vf` — fraction of points on/inside the shell (range-scaled tolerance
  0.05 + 0.04·r);
- `backed` — fraction with *any other mesh* somewhere behind them;
- `lined` — fraction with another mesh **3–30 cm** behind in the same
  direction. This is the key layer signal: a door card has the skin close
  behind it; an outermost skin has nothing. Own-mesh thickness never counts;
- `depth` — mean distance points sit behind the shell;
- `min_r` — nearest approach to the eye.

A mesh becomes an interior **candidate** through any of four channels:
visible (`vf ≥ 0.28`), enclosed (backed, shallow, inside the envelope),
exterior-fitment (compact, at beltline height, just outboard of the shell,
visible once glass is removed — wing mirrors), or the control cone.
Candidacy is deliberately high-recall; **corroborating vetoes** then remove
what the porous shell let through:

- **beyond a glass plane**: large planar translucent panes (windscreen, door
  and rear glass, identified from `*.materials.json`, not names) bound the
  glasshouse; ≥40 % of points beyond a pane within its angular footprint ⇒
  exterior (wipers, hood, the truck bed through the rear window). Panes
  smaller than 0.75 m are instrument lenses and never bound anything;
- **past the cabin shell**: the envelope half-width comes from the *lined
  walls* (p60 of their |x| reaches — robust to the whole body shell or an
  exposed sheet joining the list); protruding beyond it ⇒ door skin;
- **at the shell with nothing behind** (`lined < 0.35` and `backed < 0.35`)
  ⇒ an exposed exterior surface;
- **shell wall rising past the beltline** (thin lateral wall whose z70 is
  above the eye) ⇒ a door frame/skin, even when wheel-arch clutter behind it
  fakes a backing signal;
- **above the headliner / outside the cabin y-range / under the floor** —
  ceiling and y-bounds from the upper-cabin lining, the floor read off the
  shell in the forward-down footwell sector (straight-down rays end on the
  seat cushion and are excluded).

### The stripped-trim exposed-skin case

This is the crux the scope must not get wrong, and it is rejected by
corroboration, not trusted visibility. When a trim strips a door card the
skin becomes the nearest surface with a high `vf` — but it has **no close
backing** (`lined ≈ 0`: the card that would have been in front of it is gone,
and nothing sits 3–30 cm behind it), and it **protrudes past the envelope**
built from the surviving lined walls (the driver-side card, or the dash when
both cards are gone — the p60-of-walls half-width is exactly what keeps one
exposed sheet from dragging the envelope out to itself). The missing twin is
then read as a *signal*: the surviving card pairs with nothing, so it falls
back to aesthetic Mirror with the reason "twin absent in this trim" — convert
nothing where the trim stripped the interior, never promote the skin.
`tests/test_recommendations.py::test_stripped_trim_rejects_exposed_skin_and_keeps_lone_card`
pins this behaviour, and the sunburst2 drift/rally trims exercise it for
real (see §7).

## 3. Symmetry and structural pairing (Steps 2–3)

`cloud_symmetry_residual` reflects a candidate's cloud across the centreline
and takes the median nearest-neighbour distance, normalised by the bbox
diagonal. Below 0.045 the reflection is a visual no-op ⇒ **skip**, with two
exceptions: the **dashboard fascia** (wide, forward of the eye, spanning the
view band — symmetric at cloud resolution but carrying sub-sampling driver
detail) mirrors, and a **directional display** (emissive material on a small
planar surface) mirrors with `textureFlip`. Between 0.045 and 0.09 the
verdict exists but is low-confidence; above, the mesh is one-sided ⇒
**pairable**.

Pairing is **relational and per trim**: each pairable seeks a geometric twin
among *the meshes present in that trim* — mirrored centroid (±14 cm),
overlapping y/z, comparable size, and `mirror_pair_residual ≤ 0.10`. Twins
may come from the latent pool (the passenger-side counterpart the eye barely
sees) but never from the vetoed-exterior set, and each twin is consumed once
per trim, so mutually exclusive variants (the recast lhd/rhd case) can never
pair across trims. Twin present ⇒ one `kind:"pair"`
`MODE_MIRROR_STRUCTURAL` entry naming the driver-side member; twin absent in
every trim ⇒ aesthetic `MODE_MIRROR` with the reason saying so. Near-centred
pairables (|x| < 8 cm) never pair — a centred fitment has nothing opposite.

## 4. The control cone (Step 4)

A candidate translates iff it sits in the driver control assembly, defined
entirely in the eye/wheel frame: ahead of the eye by 0.20 m up to
wheel-distance + 1.0 m, laterally from **0.22 m inboard of the column to
0.33 m outboard** (signed toward the driver's door — the foot parking brake
by the kick panel is in; the console shifter and the driveline under the
tunnel are out), from just above the eye down to the shell floor, and within
**ergonomic reach** (`min_r ≤ 1.35 m` — every hand/foot control starts within
reach of the eye; 4-link suspension mounts in the cone's shadow at 1.40 m do
not). Broad lateral walls (an unpaired door card at the driver's shoulder)
are exempt — trim, not controls. Admission still needs evidence the eye can
interact with it: visibility, nearness to the wheel, or a fully-lined shallow
position (the race pedal box bolted behind the dash with `vf ≈ 0`). The
wheel anchor itself is exempt from every test because its mesh may span the
shaft. A small **glass** pane inside the cone is an instrument cover and
translates with the cluster (the D-series `gauges_cover_M`), which is why
"glass ⇒ skip" is decided after the cone check.

## 4b. Assembly propagation (mounted parts)

Failure analysis against the baselines exposed one dominant pattern: almost
every genuine miss was a part **physically touching (< 1.5 cm) an interior
part the classifier already handled** -- the hazard button on the dash face
(cloud gap 0.000 m), the handbrake lever on its body (0.001), the shifter
knob on its lever (0.000), dash seals on the fascia (0.004), the console on
the dash (0.002). Individually each is near-centred and symmetric, so its
own verdict is skip; as a component of an assembly it should move with its
host. `_inherit_mounted_parts` therefore propagates: a small
(diag <= 0.7 m), non-glass, non-dummy mesh with a skip/none verdict, inside
the cabin radius (<= 1.6 m of the eye), whose cloud touches (<= 3 cm) a mesh
classified aesthetic Mirror -- including an unpaired pairable, which emits as
Mirror -- inherits Mirror at low confidence ("mounted on <host>"). Two
passes resolve chains (button -> console -> dash). Translate and pair
verdicts are never overridden, structural hosts never confer (their
satellites pair on their own), and the radius bound keeps underbody
bracketry from chaining off a leak.

## 5. Directional-texture flip (Step 5)

Per-mesh materials are now surfaced minimally: `preview_data_from_tree`
records each node's `<instance_material>` symbols (older cached contexts
lazily re-parse the vehicle's own DAEs once), and
`material_flags_for_context` reads every `*.materials.json` in the vehicle
and common zips — `emissiveMap` in any stage ⇒ emissive display;
`translucent` *without* emissive ⇒ window glass. A mirrored, small, planar,
emissive surface gets `textureFlip`. Across all three reference vehicles
this flags exactly one mesh — `etk800_screen` — which is precisely the
baseline's flip set. If materials cannot be loaded the flip is skipped, not
guessed.

## 6. Batch model and caching (§3 of the brief)

The intrinsic class is a property of the mesh, not the trim. The classifier
walks trims in order; each unique mesh is classified once, in the first trim
containing it, and memoised. Only meshes that are simultaneously
**low-confidence and variant-dependent** (`context.variant_dependent_meshes`)
are re-solved in later trims, and a trim that resolves the borderline case
upgrades the memo. The only inherently per-trim step is pairing, which runs
once per trim over that trim's present set. All state is cached on the
context, so the cost profile is
`O(unique_meshes × scan + trims × pairables)`: the etk800 union pass takes
~19 s cold, and each subsequent per-trim call ~0.2 s. Per-trim point clouds
come from `preview_entries_for_config`, so a trim that moves a mesh is judged
on that trim's geometry. Meshes with **multiple simultaneous placements**
(a wheel at four corners) are excluded outright — their resolved position is
an average, a fictitious point the classifier must not reason about.

## 7. Validation against the hand-verified baselines

Per trim (not a union), diffing every used mesh's mode against the
baseline `conversion.json`:

| vehicle | trims | per-trim mode checks | agreement |
|---|---|---|---|
| etk800 | 29 | 4 516 | **99.98 %** (1 diff) |
| pickup | 63 | 12 335 | **98.19 %** |
| sunburst2 | 38 | 7 561 | **94.37 %** |

(The sunburst2 figure is lower than the others largely because assembly
propagation now mirrors dash/console furniture -- shifter dressing, nav
pods, the radio -- that the sunburst2 baseline leaves at skip while the
etk800 baseline mirrors the identical furniture; the classifier follows the
etk800 convention consistently. See category B.)

The whole-vehicle etk800 diff is **zero for all 1 109 parts**, including the
single correct `textureFlip`. Every remaining disagreement falls into one of
these categories:

**A. Per-trim relational verdicts the baseline's single global mode cannot
express (ours correct by the brief's own definition).** etk800
`racing_seat_FL` in `844_track_M`: the trim carries only the driver racing
seat, so the classifier says "aesthetic mirror, twin absent in this trim"
while the global baseline says structural — the brief explicitly calls this
correct. Same shape: `pickup_mirror_L` (one trim without its twin),
`pickup_mirror_R_facelift`, `sunburst2_seats_FR`, `sunburst2_doorpanel_FL`
(one trim each).

**B. Baseline internal inconsistencies where the classifier is the
consistent one.** The etk800 baseline pairs its rear door cards
(`doorpanel_RL/RR` structural); the pickup and sunburst2 baselines leave
their rear/crew door cards at skip. The classifier pairs rear cards on all
three vehicles (~30 % of pickup and sunburst2 per-trim diffs). Likewise
crew-cab interior door handles (the front `windowhandle_L/R` are structural
in the baseline; the identical rear ones were left unset), and
`n2o_bottle_10lb` (baseline mirror on pickup, skip on sunburst2; ours:
mirror on both).

**C. Mirror↔skip of near-symmetric parts — visual no-ops at cloud
resolution.** `pickup_bench`, `facelift_seats_F` and `facelift_console`,
`sunburst2_console`, centred floor shifters (`pickup_shifter_M/T`,
`grp_hb_lever_a`, `grp_shifter_knob_a`), dash seals, `hazard_button`,
shifter-panel buttons, `sunburst2_grp_mirror`, `rollcage_simple`, `dino`.
The brief's own rules (bench ⇒ skip, centred shifter ⇒ skip) side with the
geometry here; the hand baseline chose mirror defensively. The genuinely
sub-resolution content (a shift-pattern decal, the rear-view glass angle) is
below what 350 positions can carry — that is the resolution floor, and these
verdicts sit at or near the low-confidence band.

**D. Structural swaps of mirror-identical exterior pairs — render
identically.** Desert-truck cage items the open buggy genuinely exposes to
the eye (side mufflers, window nets) and wing-mirror satellites
(`mirrorsignal_L/R`, flasher LEDs) pair with their twins; the baseline skips
them. Swapping exact mirror twins produces the same image, so these are
no-ops, but skip would be cleaner — the desert cab has no doors or glass, so
the glasshouse boundary the vetoes rely on does not exist there.

**E. Genuine scope misses — the eye cannot see them.** Deep hidden kit the
baseline mirrors but the shell never reaches: `racingseat_base` (sunburst2,
under the seats), `grp_footplate` (passenger footrest behind the dash line),
`grp_extinguisher`, `gro_stand_S`, `grp_hb_line_a`, two of the four
police-laptop mount variants, `racing_seat_base` and `dino` on the pickup.
Visibility is a high-recall scope, not a perfect one; these are honest
misses, and mirroring-vs-skipping hidden hardware is near-invisible either
way.

**F. Control-cone boundary judgment calls.** `sunburst2_dash_key` (ignition
barrel beside the column: we translate with the column kit, the user
mirrored it with the dash), `grp_hood_release(_mount)` (at the driver's knee:
cone says translate, user said mirror), `pickup_facelift_shifter_T_buttons`
(overdrive buttons on the column shifter: we translate; the baseline mirrors
them while translating the shifter they sit on). Centimetre-scale calls
where reasonable conversions disagree.

**G. Residual leaks (wrong, and known).** `pickup_fueltank_short` (7 trims)
and `pickup_shocktop_R_offroad` (2 trims) still read as enclosed interior on
some configurations — under-body geometry whose occluders are too sparse to
close the shell. Both surface as Mirror of parts nobody sees; both carry
low/med confidence.

## 8. Honesty: where this is fragile

- **Occlusion on ~350-point clouds is the weak joint.** The shell is porous:
  a 74-point windscreen leaves 6°-bin gaps, so `vf` for any single mesh can
  be off by ±0.2. Dilating the occlusion field was tried and rejected — it
  over-occludes catastrophically (door cards fell to `vf` 0.02). The design
  answer is that no admission or veto rests on `vf` alone; every decision
  pairs it with backing, lining, envelope, glass-plane or symmetry evidence,
  and conflicts surface as low confidence rather than a confident wrong
  answer.
- **The `lined` signal saves the card-vs-skin call but is not free.** Rear
  wheel-arch clutter can fake a backing for a rear door skin (hence the
  beltline veto), and a fender flare bolted to a gutted race door sheet makes
  the sheet itself read "lined" — those sheets survive as low-confidence
  pairs flagged "wall at the cabin shell (verify)".
- **The resolution floor is real.** The steering-column top vs body split
  (centimetres, sliding with column length) is delegated to the one
  sanctioned name hint and marked low-confidence. Sub-sampling content —
  shift-pattern decals, mirror glass angle, badge text — cannot be recovered
  from positions-only clouds at this density; those verdicts are the
  skip-vs-mirror band in category C.
- **Multi-instance meshes are outside the model.** A mesh placed four times
  has no single position; it is excluded rather than mis-reasoned about.
- **Open vehicles weaken the shell.** The desert trucks have no doors or
  glass, so "inside the glasshouse" degenerates and cage-mounted exterior
  lights become spatially interior. The verdicts there are no-op pair swaps,
  but it is the honest limit of an eye-centred interior definition.
