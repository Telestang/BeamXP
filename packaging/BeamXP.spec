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
    hiddenimports=[],
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
