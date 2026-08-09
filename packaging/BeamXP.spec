# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "beamxp" / "hand_drive_tool.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "beamxp" / "blender_preview_backend.py"), "."),
        (str(ROOT / "assets" / "BeamXP_icon.ico"), "assets"),
        # Composited onto generated config previews; xp_sticker_path() looks
        # for it in sys._MEIPASS in frozen builds.
        (str(ROOT / "assets" / "xp_sticker.png"), "assets"),
    ],
    hiddenimports=[
        # Texture Fix is reached through a function-level import wrapped in a
        # try/except (build_pipeline.texture_correction_report), and
        # ispc_texcomp through another one inside the DDS writer. A module the
        # graph failed to follow would not break the build -- it would ship an
        # exe whose Texture Fix raises the moment anyone ticks the column. Name
        # them so that cannot happen quietly.
        "mesh_segmentation_transform.mirror_texture_for_rhd",
        "mesh_segmentation_transform.beamxp_transform_sym_mesh_POC",
        "ispc_texcomp",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BeamXP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX packs the exe with the same compression scheme malware droppers
    # commonly use to hide payloads; heuristic AV engines key on that pattern
    # (see the Windows Defender note in README.md's Status section).
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "BeamXP_icon.ico"),
)
