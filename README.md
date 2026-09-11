# BeamXP — BeamNG LHD/RHD Vehicle Converter

**BeamXP (BeamNG Vehicle eXPort Services)** converts BeamNG.drive vehicles between **left-hand drive and right-hand drive**.

It handles much of the repetitive work automatically — including meshes, textures, controls, mirrors, cameras, lighting, configuration parts and licence plates — while giving you a live 3D preview to review and adjust the result before building the mod.

BeamXP works with both vanilla vehicles and compatible mods.

**Download:**

- [Microsoft Store](https://apps.microsoft.com/detail/9msncq8zhp84)
- [GitHub download (0.3.5-alpha)](https://github.com/Telestang/BeamXP/raw/main/release/BeamXP-0.3.5-alpha-windows.zip)

**v0.4.0 is still in development and has not been released.** The v0.4.0 changes described below are for the upcoming release.

![Automatic texture correction on a converted interior](Screenshots/interior_reveal_720_orange_pingpong.gif)

---

## What BeamXP Does

BeamXP can:

- Convert vehicles between LHD and RHD.
- Automatically recommend transforms for steering wheels, dashboards, controls and other interior parts.
- Move, mirror or swap individual meshes rather than simply reflecting the entire interior.
- Detect and correct mirrored text, symbols and surface relief.
- Reuse existing opposite-side meshes and authored LHD/RHD parts where available.
- Swap equivalent left/right configuration parts such as seats.
- Move interactive trigger boxes with their controls.
- Correct wing-mirror setups and standard BeamNG headlight handedness.
- Generate custom EU, US and Japanese licence plates.
- Preview the conversion in 3D before building it.
- Build and install the finished vehicle directly into your BeamNG mods folder.

BeamXP is designed to produce a good first-pass conversion for most vehicles with minimal manual work. It automates the repetitive parts of the process, while occasional vehicle-specific corrections may still be needed.

---

## Quick Start

1. Install BeamXP from the Microsoft Store, or download and extract the GitHub ZIP linked above.
2. Use the **Game vehicles** or **Mods** button at the top to choose the corresponding BeamNG vehicle or mods directory.
3. Select the vehicle from the **Model** drop-down list.
4. Wait for BeamXP to unpack the vehicle, automatically find and apply the recommended transforms and equivalent parts.
5. Review the result in the live 3D preview.
6. Correct anything that needs adjustment, including triggers and equivalent parts.
7. Optionally configure licence plates and, in the upcoming v0.4.0, the rear-window sticker.
8. Click **Build + Install**, then wait a few minutes while BeamXP builds and installs the converted vehicle.
9. Launch BeamNG.drive and select the generated configuration.

### See BeamXP in action

The current video demonstrates an ETK 800-Series conversion:

<a href="https://www.youtube.com/shorts/5jT2sWg6tlI"><img src="https://img.youtube.com/vi/5jT2sWg6tlI/oardefault.jpg" width="300" alt="BeamXP ETK 800-Series conversion demo"></a>

---

## Choosing a Transform

Most manual work comes down to choosing the correct transform for a part.

| Transform            | Use it when...                                                          | Typical examples                              |
| -------------------- | ----------------------------------------------------------------------- | --------------------------------------------- |
| **Move**             | The part needs to move to the other side but keep the same orientation. | Steering wheel, pedals, gauge cluster, stalks |
| **Mirror**           | The geometry itself needs to change handedness.                         | Dashboard sections, asymmetric trim           |
| **Swap Mesh**        | The vehicle already contains a suitable opposite-side mesh.             | Door cards, mirrors, mirror indicators        |
| **Equivalent Parts** | The vehicle has separate left/right configuration parts.                | Seats, race seats, other paired JBeam parts   |

### Move X

`Move X` fine-tunes the lateral position of a **Move** or **Mirror** transform.

Positive and negative values shift the result in either direction.

### Texture Fix

When texture correction is enabled, BeamXP analyses mirrored parts for directional details such as:

- text;
- icons;
- labels;
- dial markings;
- embossed or recessed details.

BeamXP also includes a symmetric fascia step for preserving symmetric meshes, such as radios and button faces, without sending them through the glyph-detection pipeline.

---

## 3D Preview

![The tool with a conversion in progress: parts coloured by transform in the live 3D preview](Screenshots/sunburst2_tool.png)

The main window includes a live preview of the selected configuration. Meshes can be selected either from the tables or directly in the vehicle.

### Controls

- **Left-drag** — orbit
- **Right/Middle-drag** — pan
- **Mouse wheel** — zoom
- **Click** — select
- **Ctrl+Click** — add/remove items from the selection
- **H** — hide selected parts
- **Shift+H** — unhide parts

Transform colours:

- **Grey** — unchanged
- **Blue** — Move
- **Orange** — Mirror
- **Pink** — Swap Mesh
- **Green** — Equivalent Parts

Triggers can also be displayed in the preview and transformed alongside the controls they belong to.

An optional Blender preview is available for more detailed inspection, but **Blender is not required to build conversions**.

---

## Coming in v0.4.0 (Unreleased)

The upcoming v0.4.0 focuses primarily on conversion quality, reliability and performance. Work is still in progress before the next release is built.

- **Much better texture correction.** Glyph Lab is now the default detector, improving recognition of text, symbols and relief while rejecting more seams, borders and repeating material patterns.
- **Better texture handling.** Corrections work more reliably across different texture resolutions, material layers and skins.
- **More accurate geometry correction.** Improved symmetry detection and grouping keep small details attached to the parts they belong to.
- **Faster builds.** Detection caching, reused calculations and optional GPU acceleration reduce repeated work.
- **Better recommendations and part pairing.** BeamXP uses more vehicle configuration evidence when recommending transforms and finding opposite-side parts.
- **Smoother workflow.** Build progress is clearer and several selection, state and output issues have been fixed.
- **Smaller, more portable saves.** Conversion files now store deviations from defaults and exported reports no longer contain machine-specific paths.
- **Experimental rear-window stickers.** Projects can optionally add a small BeamXP sticker which follows supported rear glass surfaces.

---

## Licence Plates

BeamXP includes a reusable custom licence-plate system supporting:

- **EU** wide plates
- **US** plates
- **Japanese** plates

Plate designs can include custom fonts, colours, borders, background images, side bands and registration patterns.

Front and rear plate meshes can be configured independently, and saved plate designs can be reused across multiple vehicle projects.

---

## Rear-Window Sticker

The upcoming v0.4.0 adds an experimental optional **BeamXP rear-window sticker**.

If you use BeamXP to author a mod that you publish, please consider enabling the BeamXP rear-window sticker. It includes a QR code linking to the BeamXP Microsoft Store page and helps other users discover the tool.

Glass that cannot currently be fitted reliably is skipped instead of producing a broken sticker.

---

## Installation

### Windows

The Microsoft Store version is the easiest way to install BeamXP.

For the GitHub version:

1. Download the GitHub ZIP linked above.
2. Extract the entire archive.
3. Run `BeamXP.exe`.

Keep `BeamXP.exe` beside its `_internal` directory. Moving the exe out of the extracted folder on its own will not work.

No Python installation is required.

### Running From Source

Python 3.11 or newer is recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Then launch BeamXP from the repository folder:

```bash
python -m beamxp.hand_drive_tool
```
