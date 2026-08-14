# BeamXP — BeamNG LHD/RHD Vehicle Converter

**BeamXP (BeamNG Vehicle eXPort Services)** converts BeamNG.drive vehicles between left-hand drive and right-hand drive, with automatic correction of meshes, textures, controls, cameras, mirrors, lighting, configuration parts, and licence plates.

**[Download BeamXP 0.3.1-alpha](https://github.com/Telestang/BeamXP/raw/main/release/BeamXP-0.3.1-alpha-windows.zip)** — extract it anywhere and run the exe.

> BeamXP was previously named **BeamHDC (BeamNG Hand Drive Converter)**.

![Automatic texture correction on a converted interior](Screenshots/interior_reveal_720_orange_pingpong.gif)

BeamXP is designed to turn an existing vehicle into a convincing opposite-hand-drive version rather than simply mirror the whole interior. Where possible it preserves existing geometry, reuses opposite-side meshes or authored parts, and only mirrors the parts that actually need mirroring. The remaining directional texture and relief detail is then corrected automatically.

The source vehicle's drivetrain, suspension, tyres, handling, and core vehicle structure are left alone unless a configured opposite-side or authored part is deliberately substituted.

---

## Demo

A new end-to-end tutorial/demo is planned for this release.

Until then, the previous ETK 800-Series conversion video remains available:

<a href="https://www.youtube.com/shorts/5jT2sWg6tlI"><img src="https://img.youtube.com/vi/5jT2sWg6tlI/oardefault.jpg" width="300" alt="Previous BeamXP ETK 800-Series conversion demo"></a>

The old **39 trims in 4 minutes 30 seconds** figure is no longer used as a benchmark. Automatic texture correction adds a significant analysis step, and a new end-to-end timing will be published after measuring the current workflow from starting a conversion to driving the finished vehicle in game.

---

## Quick Start

1. **[Download the release zip](https://github.com/Telestang/BeamXP/raw/main/release/BeamXP-0.3.1-alpha-windows.zip)**, extract it, and run the exe (or run from source — see Requirements).
2. Open the app settings and configure:
   - your BeamNG vehicle-content folder;
   - your BeamNG mods folder.
3. Select a vehicle from the vehicle browser.
4. Select the configurations/trims you want to convert.
5. Run **Recommend Transforms** to fill the common cases.
6. Review the result in the live 3D preview.
7. Adjust parts using **Move**, **Mirror**, **Swap Mesh**, `Move X`, Equivalent Parts, or an existing authored opposite-hand-drive part where appropriate.
8. Select and transform any triggers that need to follow moved controls.
9. Optionally configure custom front/rear licence plates.
10. Use the in-app preview, or the optional Blender preview, to inspect the result.
11. Click **Build + Install** to generate the conversion and install it into the configured BeamNG mods folder.
12. Launch BeamNG.drive and select the generated configuration.

The goal is not to make every vehicle completely automatic. BeamXP handles the repetitive conversion work and provides a fast visual workflow for the vehicle-specific decisions that remain.

Enjoying driving from the other side? Star the repo to help other people find it, or support development on [Ko-fi](https://ko-fi.com/telestang).

---

## What's New in 0.3.x

The 0.3 line is a major conversion-quality and workflow overhaul. It introduced automatic mesh and texture correction, then spent the releases since making that pipeline correct and affordable — much of it found while converting the Lexus LC500 and Civetta Scintilla.

### Texture correction is scoped to the mesh that uses the texture

A UV layout belongs to a **mesh's use of a material**, and that is now the unit a correction is scoped to. Previously corrections were pooled per texture file, which let one trim's layout erase another's, and let a small UV island be mirrored about the whole atlas centre rather than about itself.

- Corrections are emitted per mesh, per material, and merged only where the domains come out byte-identical.
- A mesh that no correction names stays on the texture it shipped with, instead of inheriting another mesh's corrected copy.
- Where one material alias corrects into several, each fork now mints its own DAE handle, glow-map entry, and skin variants.

### Glyph detection stays inside its own UV chart

Detection regions no longer sprawl across unrelated charts that merely share atlas space. Two charts are treated as one island only where the **meshes actually meet**, and a glyph merge is refused across a chart boundary.

This is what had been welding neighbouring labels into a single block — mirror-select icons, padlocks, and whole legends stopped existing as separate candidates and so stayed mirrored.

### Builds no longer pay to re-encode untouched textures

Roughly half of a vehicle's planned corrections turn out to be no-ops, and each was still doing a full PNG write and BC7 encode to arrive back at its own input.

- Empty correction plans now emit nothing. On a Scintilla build this removed 215s from a 629s texture pass.
- The **block-encoder profile is a build setting** (basic / fast / veryfast), beside the Blender path in Build Settings, and rides in `conversion.json`.
- Alpha-searching encoder variants are chosen by reading the image rather than assumed, which is free speed on the two thirds of interior textures that are fully opaque.
- The inspection PNG beside each corrected DDS is no longer written during a build — that alone was 3.39 GB against 0.74 GB of shipped DDS on the Scintilla.

### Skins follow the material they skin

A converted vehicle selecting an interior skin rendered the base interior instead — ten of the Scintilla's sixteen trims. Corrected skins are now named so the engine's runtime composition still finds them, and the prune pass keeps a skin exactly when it keeps the base it skins.

### Handed headlight patterns are resolved from the bulb

`$lightPattern` is read by the **bulb** a slot pulls in, not by the part mounting the lamp, and an unset value is a US pattern rather than a lamp opting out. BeamXP now resolves each bulb slot against the bulb it actually mounts and inserts the pattern where it is missing.

Across the stock fleet this converts 316 pattern-reading bulb rows with none left behind, and converts 15 lamp parts that previously converted to nothing — including vehicles such as the Wendover, which had its dipped beam converted but left its full beam shining the American way.

### Live screens belong to the conversion that draws them

A conversion spawned beside its donor left one of the two instrument clusters black, because the render-to-texture tag is global to whichever vehicle asks first. A conversion now ships its own retagged copy of the controller, with a reflected copy of the page, reflected about the window the quad actually samples rather than the middle of the page.

### Fixes

- **Texture Fix decides whether a Swap Mesh is corrected.** Swap Mesh hands the object the opposite side's authored mesh, whose texture is already the one that side wants — but any Swap Mesh row sharing an atlas with a corrected mesh was pulled into correction regardless of its own Y/N, and had its whole texture domain flipped. The column is now read. Mirror is unchanged: with no authored counterpart to fall back on, a shared atlas is the only thing it can follow.
- **Mirrors keep reflecting after texture correction.** A corrected mesh is split into a deforming carrier plus rigid pieces, which left the `mirrors` row naming a mesh that no longer exists. The row now finds the piece holding the reflective surface.
- **Trigger answers reach the build.** A trigger set to Mirror in the Triggers table could come out unconverted: the table and the build were keying the same box off different node maps, and a part holding only an answered trigger was never cloned. Fourteen of sixteen Scintilla trims failed to recognise their own saved answer.
- **Accented display names survive.** Config names such as *Velocità* were written as `à`, which BeamNG's SJSON parser does not decode — the vehicle selector showed *Velocitu00e0*. Everything written for the game is now raw UTF-8.

### Interface

- The **Triggers** table drops the coordinates after each name and gains a single Hide/Show control for the whole table; hidden boxes are not drawn, not pickable, and carry no selection outline.
- **Mesh Transforms** loses the Role and X/Y/Z columns, and the filter no longer matches on hidden role text. The selected row still reports its position in the detail line.
- **Equivalent Parts** columns start equal and share the pane instead of overflowing it.
- The preview's opacity slider is gone; trigger-box blending is unchanged.

### Automatic mesh and texture correction

Previously, large parts such as dashboards and centre consoles could be converted by mirroring the complete mesh. That moved the geometry to the correct side, but also mirrored text, symbols, directional trim, and normal-map relief.

BeamXP now segments these meshes first.

- Symmetric sub-meshes are detected so they can be **rotated and translated instead of mirrored**.
- The remaining asymmetric portion of the mesh — referred to internally as the **husk** — is mirrored.
- Symmetric UV islands on the husk are detected automatically.
- Remaining candidate regions are detected with a **dominant-background foreground mask on each archive-backed colour/emissive layer**, independently inside each topological UV island; **MSER remains available for comparison**, and edge detection handles normal/relief maps.
- The texture flip axis is chosen according to whether the associated surface normal is parallel to the **XY plane** or the **Z plane**.
- Colour and relief information can therefore be corrected without reverting the whole part to a simple mirrored texture.

The aim is to preserve text orientation, symbols, embossed or recessed details, and other directional surface features while still converting the overall geometry.

### Vanilla-quality mirror reflections

Converted wing mirrors now produce the same class of reflection as vanilla BeamNG mirrors, including the vehicle itself in the reflection.

Older BeamXP conversions could produce the rather conspicuous effect of the car being invisible in its own mirrors.

### Correct LHD/RHD headlight patterns

BeamXP now converts the headlight pattern to the appropriate LHD/RHD version **when the source vehicle implements headlight patterns using BeamNG's vanilla method**.

Vehicles using a custom headlight implementation may still require vehicle-specific handling.

### Equivalent Parts

A new **Equivalent Parts** system allows BeamXP to swap actual left/right configuration parts instead of relying on visual mirroring.

For example, a single-seat LHD race configuration might contain:

```text
raceseat_FL
<empty FR seat slot>
```

With this configured pair:

```text
raceseat_FL <-> raceseat_FR
```

an LHD -> RHD conversion can leave the left slot empty and place the authored right-hand race seat in the right slot.

Equivalent Parts also preserves mixed configurations. Given:

```text
Before:
raceseat_FL + seat_FR

Configured pairs:
raceseat_FL <-> raceseat_FR
seat_FL     <-> seat_FR

After LHD -> RHD:
seat_FL + raceseat_FR
```

the seat types move to the opposite sides instead of being replaced by visually mirrored copies.

### Existing authored LHD/RHD parts

A conversion no longer has to generate a new transformed part when the vehicle already contains an authored opposite-hand-drive version.

BeamXP can select an existing part from the configuration, walk down its slot tree, find the associated LHD/RHD parts and placements, and apply them as part of the conversion.

This is distinct from a mesh transform: BeamXP is using the vehicle author's own opposite-hand-drive configuration where one already exists.

### Trigger transforms

BeamNG trigger boxes are used for interactions such as:

- Doors
- Ignition
- Traction/stability-control buttons
- Hood releases
- Other clickable vehicle controls

Triggers can now use **Move** and **Mirror** transforms so the interaction point follows the converted control.

In the 3D preview, triggers are displayed as semi-transparent wire meshes at **0.5 opacity**. Their base display is red and transformed triggers use the same transform colour coding as vehicle meshes.

### Multi-selection

`Ctrl+Click` can now be used to select multiple meshes, parts, or triggers and apply a transform to the group.

Selection works from the relevant tables and directly in the 3D preview.

This is useful for assemblies made from several separate objects. A shifter, for example, may contain a knob, stick, base, trim, and related pieces that all need the same transform.

### Clearer transform names

The old transform names have been replaced:

| Previous name | New name |
| --- | --- |
| `Translate` | **Move** |
| `Mirror Aesthetic` | **Mirror** |
| `Mirror Structural` | **Swap Mesh** |

The new names describe the operation more directly and are used throughout the UI and this README.

### More flexible Move X adjustment

`Move X` is no longer limited to **Move**.

It can now be used with:

- **Move** — controls the lateral translation.
- **Mirror** — applies an additional lateral translation *after* the mirror operation.

`Move X` can also be negative, so the result can be adjusted in either direction along the vehicle X axis.

The meaning of the positive direction follows the conversion direction:

- LHD -> RHD: positive moves toward the RHD destination side.
- RHD -> LHD: positive moves toward the LHD destination side.

This keeps the control meaningful regardless of which direction the vehicle is being converted.

### Swap Mesh for more paired components

**Swap Mesh** can be used anywhere the vehicle already provides a suitable opposite-side mesh, including parts such as:

- Door cards
- Wing mirrors
- Wing-mirror indicators/signals
- Other paired left/right components

### Installed-vehicle browser

BeamXP no longer starts by asking the user to locate and open a vehicle ZIP manually.

Instead, configure the BeamNG vehicle-content folder and the user's mods folder once. For example:

```text
<SteamLibrary>\steamapps\common\BeamNG.drive\content\vehicles
%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods
```

BeamXP parses those locations and builds a vehicle selector using vehicle display information and the default vehicle preview image.

Vanilla and mod sources are identified separately. When several sources expose the same vehicle, labels such as `[mod]`, `#1`, and `#2` are used to distinguish them.

### JBeam parser improvements

The JBeam parser and writer have received general robustness improvements.

BeamNG vehicle files in the wild are not always perfectly consistent, so unusual write cases may still expose edge cases. Please report any vehicle that produces malformed or incorrect output.

### BeamNG 0.39 compatibility

General compatibility changes have been made for the BeamNG.drive 0.39 series.

The RHD driver-camera bug previously documented by BeamXP was officially fixed by BeamNG in **0.39.3**:

- [Original BeamNG forum report](https://www.beamng.com/threads/rhd-driver-camera-bug.110306/)
- [BeamNG.drive v0.39.3 patch notes](https://www.beamng.com/game/news/patch/beamng-drive-v0-39-3/#:~:text=Fixed%20driver%20camera%20not%20moving%20correctly%20when%20looking%20back%20in%20RHD%20vehicles)

### Licence-plate reliability fixes

Several bugs affecting custom plates have been fixed and front/rear plate generation now behaves reliably in normal use.

The current rear plate still does not reproduce the same emboss effect as the front plate. The rear relies on BeamNG's existing plate material properties, and adding a separate normal map would conflict with that setup. Replicating the required plate material properties independently is planned for a later update.

### Recommend Transforms

`Recommend Modes` has been renamed to **Recommend Transforms**.

The recommendation system has returned to a name-based heuristic approach. The previous release's spatial recommendation logic was slower and less accurate in practice.

Spatial classification may be revisited in the future, but the current priority is fast, predictable recommendations that the user can review before applying.

---

## Vehicle Discovery

BeamXP scans the configured BeamNG vehicle and mod locations instead of requiring individual ZIP selection.

The browser is designed to make large collections manageable:

- Vehicles are presented using their display information rather than requiring the user to recognise archive filenames.
- The default BeamNG vehicle preview image is shown to make a source easier to identify.
- Vanilla and mod sources are differentiated.
- Duplicate vehicle sources are disambiguated using markers such as `[mod]`, `#1`, and `#2`.

This means a vanilla vehicle and several mods based on the same vehicle ID can coexist in the selector without the user having to manually browse the filesystem for each conversion.

---

## Conversion Model

BeamXP follows a simple preference:

> **Preserve or reuse correct geometry first; mirror only what actually needs mirroring.**

A conversion can therefore combine several approaches in the same vehicle:

1. Use an existing authored LHD/RHD part when the vehicle already provides one.
2. Swap configured left/right equivalent parts where the correct opposite-side JBeam part exists.
3. Use **Swap Mesh** where the opposite-side visual mesh already exists.
4. Use **Move** where the original orientation should be preserved.
5. Segment large parts so symmetric components can be moved or rotated without reflection.
6. Use **Mirror** for the remaining asymmetric geometry.
7. Correct directional colour-map and normal-map regions after the mirror.

This mixed approach is what allows a converted cockpit to retain the source vehicle's design instead of looking like a simple reflection.

---

## Transforms

### Move

**Move** relocates the visual geometry laterally without reflecting it.

Typical uses include:

- Steering wheel
- Gauge cluster
- Needles and screens
- Pedals
- Stalks
- Gear selectors
- Driver-specific controls

For animated props, BeamXP preserves the original animation references and adds the required global translation so the visual object continues to animate around its translated position.

`Move X` can be used to tune the lateral placement in either direction.

### Mirror

**Mirror** reflects the geometry across the vehicle centreline.

It is appropriate for geometry that genuinely needs to exchange handedness rather than simply move across the cabin.

Large mirrored parts can also pass through BeamXP's automatic segmentation and texture-correction pipeline so directional text, symbols, trim, and relief information do not have to remain backwards.

`Move X` can be applied after the reflection when the mirrored result needs an additional lateral offset.

### Swap Mesh

**Swap Mesh** uses the corresponding opposite-side mesh on the existing source-side structure rather than geometrically reflecting the original mesh.

Typical uses include:

- Door cards
- Wing mirrors
- Wing-mirror signals
- Other paired left/right visual parts

Because the opposite-side mesh is already authored in the source vehicle, it can preserve details that would otherwise be unnecessarily mirrored.

### Move X

`Move X` is a lateral adjustment available to **Move** and **Mirror**.

For **Move**, it controls the translation itself.

For **Mirror**, it is an extra translation applied after the reflection.

Positive and negative values are supported. The positive direction follows the selected conversion direction rather than being hard-coded to one absolute side of the vehicle.

---

## Equivalent Parts

Equivalent Parts operate at the configuration/slot level rather than merely changing a mesh.

The user defines left/right pairs, for example:

```text
seat_FL      <-> seat_FR
raceseat_FL  <-> raceseat_FR
```

BeamXP can then exchange occupied and empty slots, or preserve mixed part types while moving them to the appropriate side.

This is particularly useful for:

- Single-seat race configurations
- Different driver/passenger seat types
- Left/right hardware that already exists as separate JBeam parts
- Configurations where visual mirroring would leave the physical part on the wrong side

---

## Existing Authored Hand-Drive Parts

Some vehicles already contain separate LHD and RHD parts.

BeamXP can use those authored parts instead of generating another transformed version.

When such a part is selected, BeamXP walks down the relevant slot tree, finds the associated handed parts and placements, and applies the existing vehicle configuration.

Where the source vehicle already contains the correct solution, reusing it is preferable to recreating it.

---

## Automatic Mesh and Texture Correction

This pipeline is primarily intended for large interior components such as dashboards and centre consoles.

A simplified view is:

```text
Source mesh
    |
    v
Mesh segmentation
    |
    +--> Symmetric sub-meshes
    |        |
    |        +--> rotate / translate without reflection
    |
    +--> Remaining asymmetric geometry ("husk")
             |
             +--> mirror geometry
                     |
                     v
              UV-region analysis
                     |
                     +--> symmetric UV-island detection
                     +--> foreground mask on colour/emissive layers (MSER comparison mode available)
                     +--> edge detection on normal map
                     |
                     v
              choose correction axis
                     |
                     v
          corrected colour + relief mapping
```

### Why segmentation matters

A centre console may contain a mixture of:

- symmetric trim;
- buttons with readable symbols;
- asymmetric controls;
- display panels;
- decorative relief;
- geometry that only needs to move rather than reflect.

Mirroring the entire object treats all of those features as though they have the same handedness.

Segmentation lets BeamXP preserve the orientation of symmetric components and restrict mirroring to the geometry that actually needs it.

### UV and texture correction

After the asymmetric husk is mirrored, BeamXP analyses its UV mapping.

It uses:

- symmetry testing on UV islands;
- dominant-background foreground components in colour/emissive layers (with MSER available for comparison);
- edge information in the normal/relief map.

Candidate regions can then be reflected independently so the geometry changes side without forcing text, icons, or relief features to read backwards.

The selected correction axis depends on the orientation of the associated surface normal relative to the XY and Z planes.

---

## Triggers and Interactive Controls

BeamNG uses trigger boxes for many clickable interactions.

BeamXP exposes those triggers in the preview so they can follow converted controls.

Supported trigger transforms currently include:

- **Move**
- **Mirror**

The trigger wireframes are semi-transparent and use the same transform colour scheme as normal meshes.

Multiple triggers can be selected with `Ctrl+Click`, which is useful when several interactions belong to one transformed control assembly.

---

## Mirrors and Lighting

### Wing-mirror reflections

BeamXP now generates mirror setups that show the vehicle correctly in the reflection, matching the expected visual behaviour of vanilla BeamNG mirrors.

### Headlight handedness

When the source vehicle uses BeamNG's standard headlight-pattern system, BeamXP converts the pattern for the destination handedness automatically.

This is conditional on the source implementation. Custom vehicle-specific lighting systems may not expose the same vanilla data for BeamXP to transform.

---

## In-App Preview

![The tool with a conversion in progress: parts coloured by transform in the live 3D preview](Screenshots/sunburst2_tool.png)

The main window includes a live 3D preview of the selected configuration.

Changes are intended to be inspectable before building the mod.

- Left-drag orbits.
- Right- or middle-drag pans.
- Mouse wheel zooms.
- Click a part in the viewport to select it.
- `Ctrl+Click` adds/removes parts, meshes, or triggers from the current selection.
- `H` / `Shift+H` hides or unhides selected parts.
- The Triggers table's Hide/Show button takes every trigger box out of the preview, for a clear view of the cabin geometry underneath.
- The original-layout option removes the hand-drive transforms while keeping the currently selected configuration context.
- Meshes are colour-coded by transform:
  - grey — unchanged;
  - blue — **Move**;
  - orange — **Mirror**;
  - pink — **Swap Mesh**.
- Trigger boxes are shown as semi-transparent wire meshes and use the same transform colour coding.

---

## Blender Preview

The Blender preview remains optional.

Use it when you want to inspect the complete generated configuration using normal Blender tools; the in-app preview is the faster feedback loop for routine conversion work.

The preview:

- builds the current unpacked output first;
- uses the currently selected configuration;
- imports the resolved vehicle, including generated meshes and unchanged context meshes;
- does not require the BeamNG Blender JBeam Editor add-on;
- opens as a new unsaved Blender instance.

Nothing is written from Blender unless you save it manually.

BeamXP can still build and install conversions without Blender configured.

---

## Licence Plates

BeamXP can generate reusable custom plate designs and apply front/rear plate configurations per trim.

A trim can use:

- `Off`
- `Custom`
- a saved plate set from the global library

Front and rear physical plate meshes can be selected independently.

### Plate designs

Three plate families are supported:

- **EU** — wide format
- **US** — 2:1 format
- **JP** — 2:1 format

Design options include:

- fonts;
- colours;
- borders;
- side bands;
- background images;
- registration patterns;
- front/rear configuration;
- live preview.

Plate-style fonts are not bundled because many are not licensed for redistribution. User-supplied `.ttf` and `.otf` fonts can be placed in the BeamXP fonts folder.

Registration patterns use:

```text
@ = letter
# = digit
~ = letter or digit
. = centre dot
```

### Background images

Separate front and rear images can be supplied.

Recommended texture sizes:

```text
EU: 1024 x 196
US/JP: 512 x 256
```

Clean multiples of those sizes also map well. Other aspect ratios are centre-cropped.

### EU plates

EU-format controls include:

- separate front and rear background colours;
- font colour;
- horizontal text offset;
- character spacing;
- side band;
- country code;
- custom band colour;
- emblem or complete band image.

### US plates

US-format controls include:

- background colour;
- font colour;
- text scale;
- horizontal/vertical text offset;
- character spacing.

### JP plates

JP-format controls include:

- plate style;
- region;
- classification;
- kana;
- main registration number pattern.

### Rear emboss limitation

The front plate can reproduce the intended embossed appearance.

The rear plate currently cannot use the same emboss setup while retaining the material properties inherited from BeamNG's default plate system. A future implementation may recreate those material properties independently so the rear can use its own normal map without conflict.

### Plate library

The global library manages reusable sets with:

- New
- Duplicate
- Rename
- Delete
- Edit

Projects can reference saved sets rather than duplicating their contents.

`Export plates mod...` writes selected designs into a universal BeamXP plates mod so they can be selected on supported vehicles.

Each **Build + Install** also refreshes the exported plate library in the configured mods folder.

---

## Physics and Deformation Notes

BeamXP uses several different conversion mechanisms, so their physical behaviour is not identical.

### Visual mesh transforms

**Move** and **Mirror** operate primarily on visual geometry. They do not automatically relocate the source vehicle's underlying node/beam structure.

As a result, severe crash deformation can still follow the original physical side even when the visible control has moved to the opposite side.

### Swap Mesh

**Swap Mesh** places an authored opposite-side mesh onto the existing structure.

For paired components this can produce a better visual/deformation match than reflecting the original mesh.

### Equivalent Parts and authored parts

Equivalent Parts and existing authored LHD/RHD parts operate at the slot/configuration level.

Where the source vehicle provides a real opposite-side JBeam part, BeamXP can select that part rather than leaving the original physical component in place.

### Triggers

Trigger locations are handled separately from the vehicle's node/beam structure and can be moved or mirrored so interactive areas continue to align with converted controls.

BeamXP remains a conversion tool rather than a complete vehicle re-authoring system. Extremely vehicle-specific structures or unusual deformation setups may still require manual authoring for perfect crash behaviour.

---

## Output

Projects are stored under:

```text
%LOCALAPPDATA%/BeamXP/handedness_conversion_projects/<projectName>/
```

The app settings file is stored under:

```text
%LOCALAPPDATA%/BeamXP/
```

BeamHDC-era data under `%LOCALAPPDATA%/BeamHDC/` is migrated when required so replacing the application itself does not discard project data.

A project contains generated settings and build data such as:

```text
conversion.json
unpacked_output/
build/
blender_preview/
```

The configured BeamNG mods folder is used as the install target for generated output.

Vehicle conversion archives use the existing BeamXP conversion naming scheme, and generated conversion settings are embedded in the output so the attempted conversion can be reproduced later.

Reusable plate sets are stored separately under:

```text
%LOCALAPPDATA%/BeamXP/plates/
```

---

## Example Configs

The `examples/conversion_configs/` folder contains sample conversion settings, including existing vanilla-vehicle examples.

These are settings only, not redistributed BeamNG vehicle assets.

To use an example:

1. Make sure the matching vehicle is available in the configured vehicle/mod folders.
2. Select the vehicle in BeamXP.
3. Use `Import Config`.
4. Select the example configuration file.
5. Review the imported transforms against the current vehicle before building.

Converted vanilla vehicle archives are not distributed with BeamXP.

---

## Requirements

### Windows Build — Recommended

Download the Windows release, extract it anywhere, and run:

```text
BeamXP.exe
```

No Python installation is required.

Blender is optional and external.

The application itself does not need to be placed in the BeamNG mods folder. Configure the game vehicle folder and mods folder from inside BeamXP.

### Running From Source

- Windows
- Python 3.11 or newer recommended
- Tkinter
- BeamNG.drive installation/source vehicle content
- Blender 4.2+ optional

Install dependencies with:

```powershell
pip install -r requirements.txt
```

Then run BeamXP from source using the repository entry point.

### Linux

A user has reported the Windows build working under Wine.

BeamNG.drive itself commonly runs through Proton, so the configured mods directory can point at the BeamNG AppData path inside the relevant Proton prefix, for example:

```text
~/.steam/steam/steamapps/compatdata/284160/pfx/drive_c/users/steamuser/AppData/Local/BeamNG.drive/current/mods
```

Native source execution should also be possible where the Python/OpenGL dependencies are available, but it is not the primary tested environment.

---

## Status

BeamXP is still under active development.

The current release performs substantially more analysis than earlier builds, especially during automatic mesh and texture correction. That improves the visual result but also increases conversion time.

No replacement for the old 39-trim timing benchmark is published yet. A new benchmark will be based on the real end-to-end workflow rather than only the previous build stage.

If you encounter a vehicle that does not convert correctly, please report it. Real vehicle edge cases are particularly useful for improving the JBeam writer, authored-part resolution, texture correction, and transform heuristics.

---

## Known Limitations

- Unusual or inconsistent JBeam syntax may still expose parser/writer edge cases.
- Some community vehicles have off-centre geometry or inconsistent object origins and may require manual `Move X` adjustment.
- Some animated props may still need vehicle-specific attention.
- Texture paths in Blender preview may not resolve exactly like BeamNG's material system.
- Wheel-attached meshes such as wheels, hubcaps, and tyres may not appear in their final runtime positions in the preview because BeamNG places them using generated wheel node groups in game.
- Severe crash deformation of purely visual **Move** or **Mirror** conversions may still follow the original physical structure.
- Automatic headlight handedness conversion depends on the source vehicle using BeamNG's vanilla headlight-pattern implementation.
- Rear custom plates do not currently reproduce the same emboss treatment as the front plate.
- Automatic texture correction is computationally more expensive than the old whole-mesh mirroring path.

The previous RHD driver-camera lean bug is **not** a current BeamXP limitation: BeamNG fixed it officially in version 0.39.3.

---

## Windows Defender False Positive

Windows Defender, and potentially other antivirus software, may flag a BeamXP Windows build with a broad heuristic detection such as `Trojan:Script/Wacatac.B!ml`.

BeamXP performs several operations that can look suspicious to automated heuristics despite being legitimate:

- reads and writes files in BeamNG's AppData directories;
- extracts and modifies vehicle archives;
- creates new mod archives;
- writes files intended to be loaded by another application;
- is distributed as a small unsigned PyInstaller application.

BeamXP is source-available under the MIT licence. You can inspect the code or build the executable yourself instead of using the packaged Windows build.

If a release is flagged, please open an issue with the exact BeamXP version and a screenshot of the detection.

---

## Reporting Issues

Open a GitHub issue with enough information to reproduce the conversion.

Ideally include:

- the source vehicle or a link to where it can legally be obtained;
- the BeamXP `conversion.json`;
- the generated output/configuration involved;
- a description of the incorrect behaviour;
- screenshots where the problem is visual.

Generated mods also embed conversion settings so an attempted conversion can be reproduced without manually reconstructing every transform.

---

## Support

If BeamXP saved you from converting a vehicle by hand, consider supporting development on [Ko-fi](https://ko-fi.com/telestang).

Starring the repository also helps other BeamNG users find the project.

---

## License

BeamXP source code is licensed under the **MIT License**.

Generated conversion archives are **not automatically MIT licensed**. They may include or derive from the source vehicle's assets and remain subject to the licences and permissions that apply to those assets.
