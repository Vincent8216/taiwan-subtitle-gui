# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Taiwan Subtitle GUI (.app bundle).

Build with:
    .venv/bin/pyinstaller packaging/TaiwanSubtitle.spec --noconfirm

mlx / mlx-audio ship native Metal shader (.metallib) and binary assets that
PyInstaller's default import analysis does not pick up on its own, so both
packages are pulled in wholesale via collect_all() — without this, the built
app imports fine but MLX fails at the first GPU op ("failed to load default
metallib").
"""

import os

from PyInstaller.utils.hooks import collect_all

APP_VERSION = "0.1.0"
BUNDLE_ID = "com.vincent8216.taiwansubtitle"

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = []
binaries = []
hiddenimports = []
for package in ("mlx", "mlx_audio"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [os.path.join(project_root, "gui.py")],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name="TaiwanSubtitle",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TaiwanSubtitle",
)
app = BUNDLE(
    coll,
    name="TaiwanSubtitle.app",
    icon=None,
    bundle_identifier=BUNDLE_ID,
    version=APP_VERSION,
    info_plist={
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "台灣影片轉字幕工具",
    },
)
