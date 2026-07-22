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
`glass_beyond_fractions`, `full_vertex_clouds_for_ids`,
`full_surface_triangles_for_ids`, `surface_visibility_stats`,
`reflected_orphan_stats`, `mirror_pair_residual`,
`directional_verdict_backing`, `principal_extent_sds`,
`material_flags_for_context`, `mesh_material_symbols`). Orchestration lives in
`beamng_hand_drive_tool.py` (`_classify_meshes_for_trim`,
`_resolve_trim_pairs`, `_inherit_mounted_parts`,
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
as the **360° shell**. Driver-visible admission is narrower: a point must also
have a non-negative projection onto the driver's forward vector, giving the
requested **180° forward hemisphere**. Geometry behind the eye can still
contribute enclosure/backing evidence, but cannot become a candidate merely
because it wins a sparse rearward angular bin. Steering-scored meshes are
transparent (so the cluster,
stalks and pedals behind the wheel are reachable). The large mesh that
geometrically surrounds the eye and extends below it is also transparent: the
camera sits inside the driver's seat, and an opaque seat otherwise hides its
own rails/base. Per mesh the scan reports:

- `vf` — fraction of points on/inside the 360° shell (range-scaled tolerance
  0.05 + 0.04·r), retained for enclosure and backing;
- `front_vf` — the same visibility test intersected with the forward 180°
  hemisphere;
- `backed` — fraction with *any other mesh* somewhere behind them;
- `lined` — fraction with another mesh **3–30 cm** behind in the same
  direction. This is the key layer signal: a door card has the skin close
  behind it; an outermost skin has nothing. Own-mesh thickness never counts;
- `depth` — mean distance points sit behind the shell;
- `min_r` — nearest approach to the eye.

The 6° point shell is now a **broad phase**, not the final exposure test. A
vertex-only shell cannot see the filled interior of a triangle: this was why
all 350 points of `etk800_exhaust_R` could appear to fall through gaps between
the sparse vertices of the body, carpet and heatshield. The classifier parses
indexed COLLADA `triangles`, `polylist` and `polygons`, applies each object's
node and per-trim placement matrices, and traces every candidate point's open
eye segment against those filled triangles. A point is visible only when no
opaque triangle intersects before it (the final 2 mm is ignored so the point's
own incident face does not hide itself). In `844_150_M` the old shell reported
53/350 visible exhaust points, including 25/350 forward; the exact surface
test reports **0/350 visible and 350/350 blocked**.

A mesh becomes an interior **candidate** through the four general channels:
driver-visible (`front_vf ≥ 0.28`), enclosed (backed, shallow, inside the envelope),
exterior-fitment (compact, at beltline height, just outboard of the shell,
visible once glass is removed — wing mirrors), or the control cone. There is
also a narrow under-seat channel for compact hardware directly below the
geometrically detected driver seat; it grants candidacy only, allowing a seat
rail/base to reach ordinary symmetry and pairing even when carpet hides most
eye rays. Its centroid must remain in the forward hemisphere, so rearward
fixtures cannot enter through it.
Candidacy is deliberately high-recall; **corroborating vetoes** then remove
what the porous shell let through:

- **beyond a glass plane**: large planar translucent panes (windscreen, door
  and rear glass, identified from `*.materials.json`, not names) bound the
  glasshouse; ≥40 % of points beyond a pane within its angular footprint ⇒
  exterior (wipers, hood, the truck bed through the rear window). The sparse
  point result is refined against the panes' filled triangles before this veto
  fires; this is why the common-pack `dino` mascot is correctly inside despite
  the point bins falsely reporting 67.7% beyond glass. Panes
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

Self-symmetry is decided on the **uncapped DAE vertex set**, not the 350-point
preview. `sample_points` is `points[::stride][:max_points]`: the final
truncation can discard a contiguous tail after striding and split authored
mirror pairs. That sampling artifact — not a physical feature of the mesh —
made symmetric meshes appear asymmetric.

`full_vertex_clouds_for_ids` follows each object's
`dae_source_zip or context.source_zip`, parses every source DAE at most once,
and translates the authored cloud onto the placed preview centre. Shared
accessories therefore come from the common pack that actually owns them. A
trim's x placement is applied before testing. Parse failure falls back to the
preview cloud, deliberately degrading toward an extra benign Mirror rather
than a missed asymmetric part.

`reflected_orphan_stats` reflects every vertex across the vehicle centreline
and performs deterministic voxel-hash membership checks. A vertex is an exact
orphan when no reflected partner exists within 0.1 mm; **only
`exact_orphans == 0` skips**. The separate fraction unmatched within 2 cm is
used for low-confidence grading (`< 0.05`) and the barely-seen centred-blob
guard (`< 0.15`), never for the skip decision. There are two established
exceptions to a zero-orphan no-op: the **dashboard fascia** still mirrors, and
a small planar emissive **directional display** mirrors with `textureFlip`.
Any exact orphan otherwise makes the mesh pairable. This is intentionally
trigger-happy: small modelling offsets can produce an extra Mirror, which is
preferred to missing real one-sided detail.

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

A prospective pair whose non-empty material-symbol sets are **disjoint** is
protected as `functional_skip` with the reason “functionally sided: materials
differ, needs build-side material rebind”. This catches material-animated
directional lights while allowing ordinary multi-material housings that share
a body material. The real build-side solution remains a later material rebind;
until then, Skip is safer than swapping directional animation materials.

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

### Passenger-footwell blind-spot pass

After the ordinary pass, the classifier averages the centroids of Translate
meshes at least 10 cm below the steering wheel, reflects that point across the
centreline, and aims a **30° field-of-view cone** from the eye at the opposite
footwell. Only still-unclassified meshes enter this second pass. A forced mesh
is queued for the same filled-surface visibility refinement used on the driver
side even when the sparse point shell saw nothing. Cone membership does **not**
grant admission: after refinement the mesh must satisfy the ordinary visible,
enclosed, fitment or control evidence. The cone locates a blind spot; it does
not provide x-ray vision.

The pass is bounded to below-wheel geometry whose 80th-percentile point range
is within the 1.6 m cabin radius. Hard exterior, roof, rear-cab and under-floor
vetoes persist between trims and cannot be reopened by the cone. The real
`grp_footplate` is recovered because its sparse shell says 0% visible while
filled triangles show 44%; ETK's covered driveshaft and transfer case are
rejected at 0% and 2.25% respectively.

## 4b. Assembly propagation (mounted parts)

Failure analysis against the baselines exposed one dominant pattern: almost
every genuine miss was a part **physically touching (nominally 3 cm) an interior
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
The implementation allows 0.1 mm of numerical dust on that boundary; this
keeps `police_laptop_mount_b_alt` attached to its mirrored mount at a measured
30.0219 mm rather than recovering it through the passenger cone.

A second propagation rule handles genuinely floating scoped meshes. After
the 3 cm contact passes, `directional_verdict_backing` reuses the 6° angular
bins to ask whether a floater's **forward-hemisphere** eye rays continue at
least 3 cm farther into geometry classified Translate, Mirror, or
Pairable/structural. Rearward points are removed before this test, matching
the general 180° driver-visible rule; this prevents a rear-window third brake
light from inheriting through bodywork behind it. Any qualifying forward
backing, at any farther distance along the ray, makes the floater inherit
low-confidence Mirror and records the verdict class behind it. Both floater
and backing geometry are bounded to the 1.6 m cabin radius; glass,
furniture-sized meshes over 0.7 m, and sub-resolution dummies are excluded.
This covers detached laptop-mount poles without promoting SPOTLIGHT or axle
braces.

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
containing it, and memoised. Low-confidence or no-verdict meshes whose
placement is variant-dependent (`context.variant_dependent_meshes`) are
re-solved in later trims until a decisive placement upgrades the memo. The
only inherently per-trim step is pairing, which runs
once per trim over that trim's present set. All state is cached on the
context: intrinsic verdicts, scoped IDs, persistent hard vetoes, pair votes,
full DAE clouds and triangle surfaces, parsed-file keys and exact symmetry
results. Candidate-ray tests for separate meshes are independent, so up to
four run concurrently while stateful trim ordering, memo updates and pairing
remain sequential. The cost profile is
`O(unique vertices + candidate rays × scene triangles + trims × pairables)`;
subsequent calls reuse uncapped clouds rather than reparsing a DAE.

Per-trim point clouds normally come from `preview_entries_for_config`, so a
trim that translates a mesh is judged at that trim's location. A flexbody is
rebuilt from the authored DAE and every real matrix in that trim. Multiple
simultaneous placements are represented by the union of their transformed
clouds/surfaces, never their fictitious average. If a symmetric multi-instance
placement produces no verdict, a later variant-dependent placement (for
example a single driver racing-seat base) can supply the decisive intrinsic
verdict.

## 7. Validation against the hand-verified baselines

`python scripts/validate_spatial_classifier.py` obtains one global
recommendation from the union of meshes used by the selected trims, then
applies that recommendation to each trim before diffing every used mesh
against the baseline `conversion.json`. Structural pairs count both members;
when only one member is fitted, the desired aesthetic-Mirror fallback is
treated as semantic agreement rather than a Structural→Mirror mismatch. This
models the UI/build path and avoids the old state-cache artifact caused by
calling the global recommender independently for each trim. A baseline
Structural→Skip result is likewise excluded when the classifier explicitly
identifies the pair as functionally sided by differing materials: Skip is the
deliberate safe interim until generated meshes can rebind their twin's
materials. Other Structural→Skip results remain visible.

| vehicle | trims | per-trim mode checks | agreement |
|---|---|---|---|
| etk800 | 29 | 4 516 | **99.38 %** (28 diffs) |
| pickup | 73 | 12 335 | **97.94 %** (254 diffs) |
| sunburst2 | 39 | 7 561 | **92.87 %** (539 diffs) |

These figures include the signed-off exact-orphan policy: the literal zero
threshold mirrors small DAE modelling offsets that the hand baselines skip.
That is the chosen false-positive/false-negative trade: a benign extra mirrored
copy is preferred to leaving a real one-sided control behind. Restricting
driver-visible admission to the forward hemisphere removed rearward
visibility-only candidates. The ETK baseline was also corrected so its rear
door cards are Skip. Current true-mismatch transition counts are:

- ETK: 28 Skip→Mirror; three expected functionally-sided skips are excluded;
- pickup: 63 Skip→Mirror, 82 Skip→Structural, 104 Mirror→Skip,
  and 5 Mirror→Translate; two desired twin-absent fallbacks and 14 expected
  functionally-sided skips are excluded;
- Sunburst: 392 Skip→Mirror, 64 Skip→Structural, 40 Mirror→Skip,
  and 43 Mirror→Translate; two desired twin-absent fallbacks and 18 expected
  functionally-sided skips are excluded.

Every disagreement falls into one of these categories:

**A. Per-trim relational verdicts the baseline's single global mode cannot
express.** The validator now normalises these before counting differences: a
globally structural pair remains structural when both members are fitted and
becomes aesthetic Mirror when only one is fitted. The latter is explicitly
reported as an excluded fallback count, never as a mismatch. The
`racing_seat_FL/FR` Structural→Skip outcomes are not missing-twin noise: their
disjoint `racing_seat_fl`/`racing_seat_fr` material bindings trigger the
signed-off functionally-sided safe Skip pending material rebinding. The
validator reports these in its excluded count instead of its mismatch table.

**B. Forward-hemisphere corrections and baseline inconsistencies.** The
general visibility channel is now the forward 180° only. The ETK baseline's
structural rear-door-card modes were confirmed as a hand-baseline mistake and
corrected to Skip; `doorpanel_RL/RR` now agree as Skip. Sunburst's two rear-card
Skip→Structural rows likewise disappear. Pickup's ordinary rear-card leaks
and all five carpet-family leaks disappear; rear parts with independent
enclosure/wall evidence are still allowed because the 360° shell remains an
enclosure measurement, not driver visibility. Crew-cab interior door handles
remain a baseline inconsistency: the front `windowhandle_L/R` are structural
in the baseline while the identical rear ones were left unset.

**C. Mirror↔skip of visually near-symmetric parts.** Exact full-vertex testing
now separates authored equality from “looks symmetric at preview density.”
Benches and exactly mirrored carpets still skip; millimetre-scale offsets in
sunvisors, roof covers, rear seats and driveline/exhaust meshes create exact
orphans and therefore Mirror. Conversely, an exactly symmetric bench can skip
where the baseline mirrored defensively. The coarse 2 cm fraction lowers
confidence but never changes the zero-orphan decision.

**D. Structural swaps of mirror-identical exterior pairs — render
identically.** Desert-truck cage items the open buggy genuinely exposes to
the eye (side mufflers, window nets) can still pair; the desert cab has no
doors or glass, so the glasshouse boundary degenerates. Directional
wing-mirror signals and generic flasher LEDs no longer pair: their disjoint
material bindings produce protected Skip pending build-side material rebind.
The global validator now correctly shows the flasher pair as agreement; the
old two-row `_b` mismatch was solely a subset-call artifact.

**E. Former sightline blind spots now covered.** Driver-seat transparency,
under-seat candidacy and real multi-instance rebuilding recover
`racingseat_base` and `racing_seat_base`; the reflected 30° passenger-footwell
cone requests the exact rescan that admits
`grp_footplate`; angular verdict backing handles all three
`police_laptop_mount_*` meshes. These targets match the baseline in every trim
that uses them. Remaining hidden-kit differences are parts outside the scoped
cone/radius, not name-based omissions.

**F. Control-cone boundary judgment calls.** `sunburst2_dash_key` (ignition
barrel beside the column: we translate with the column kit, the user
mirrored it with the dash), `grp_hood_release(_mount)` (at the driver's knee:
cone says translate, user said mirror), `pickup_facelift_shifter_T_buttons`
(overdrive buttons on the column shifter: we translate; the baseline mirrors
them while translating the shifter they sit on). Centimetre-scale calls
where reasonable conversions disagree.

**G. Residual leaks (wrong, and known).** Filled-surface occlusion fixes the
covered ETK exhaust, driveshaft and transfer case: all three are absent from
every mismatch row, as are `pickup_fueltank_short`,
`pickup_shocktop_R_offroad` and the 32-trim `sunburst2_exhaust_pipe` leak. The
remaining pickup and Sunburst driveline rows enter through other
contact/pair/scope channels and need separate tracing.

## 8. Honesty: where this is fragile

- **The point shell remains approximate backing/envelope evidence.** Final
  candidate exposure and disputed glass crossings use filled DAE triangles,
  but `backed`, `lined`, `depth` and the initial cabin envelope still come from
  6° preview bins. Their sparsity can affect non-visibility channels,
  particularly contact inheritance and non-visibility pairing channels. The
  passenger blind-spot cone itself now uses exact surface exposure.
- **The `lined` signal saves the card-vs-skin call but is not free.** Rear
  wheel-arch clutter can fake a backing for a rear door skin (hence the
  beltline veto), and a fender flare bolted to a gutted race door sheet makes
  the sheet itself read "lined" — those sheets survive as low-confidence
  pairs flagged "wall at the cabin shell (verify)".
- **The resolution floor is real.** The steering-column top vs body split
  (centimetres, sliding with column length) is delegated to the one
  sanctioned name hint and marked low-confidence. Self-symmetry now uses all
  DAE positions, but texture-only content — shift-pattern decals, mirror-glass
  direction, badge text — is still not recoverable from positions.
- **Multi-instance surfaces cost more.** They are represented honestly as the
  union of every placement, so four-wheel and two-seat meshes no longer
  collapse to an average or disappear. This increases triangle count for that
  trim but keeps spatial reasoning correct.
- **Functionally sided pairs await a build fix.** Disjoint material symbols
  are enough to prevent an unsafe structural swap, but safe conversion needs
  generated meshes rebound to their twins' directional materials. Until that
  build-side work lands, the classifier intentionally emits no recommendation.
- **Open vehicles weaken the shell.** The desert trucks have no doors or
  glass, so "inside the glasshouse" degenerates and cage-mounted exterior
  lights become spatially interior. The verdicts there are no-op pair swaps,
  but it is the honest limit of an eye-centred interior definition.
