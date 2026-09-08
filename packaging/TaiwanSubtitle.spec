# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Taiwan Subtitle GUI (.app bundle).

Build with:
    .venv/bin/pyinstaller packaging/TaiwanSubtitle.spec --noconfirm

Design: this is a THIN launcher. mlx / mlx-audio / numpy / soundfile /
opencc / huggingface-hub are deliberately EXCLUDED from the bundle — they're
large (mlx alone ships a ~130MB Metal shader library) and, like the AI
models, the app downloads them itself on first run via the "系統套件" panel
("安裝缺少的 Python 套件"), installing into
~/Library/Application Support/TaiwanSubtitle/pylibs and adding that to
sys.path at startup (see transcribe.py). That install path uses pip's
in-process API, so `pip` itself must be bundled — collect_all('pip') pulls
in its vendored dependencies too.
"""

import os

from PyInstaller.utils.hooks import collect_all

APP_VERSION = "0.1.0"
BUNDLE_ID = "com.vincent8216.taiwansubtitle"

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [(os.path.join(project_root, "requirements.txt"), ".")]
binaries = []
hiddenimports = []
for package in ("pip",):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# 這些是 transcribe.py 內部延遲載入（函式內 import）的重型套件；PyInstaller
# 的靜態分析仍看得到那些 import 陳述式，預設會硬把它們一起打包進來。明確
# excludes 掉，讓 App 保持精簡——第一次執行時才由使用者自己下載安裝。
heavy_excludes = [
    "mlx",
    "mlx_audio",
    "mlx_metal",
    "numpy",
    "soundfile",
    "opencc",
    "huggingface_hub",
]

a = Analysis(
    [os.path.join(project_root, "gui.py")],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=heavy_excludes,
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
