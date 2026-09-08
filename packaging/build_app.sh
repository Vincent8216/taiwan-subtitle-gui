#!/usr/bin/env bash
# 把 gui.py 打包成 TaiwanSubtitle.app，再包成可分發的 .dmg。
#
# 用法：
#   ./packaging/build_app.sh
#
# 需要先在 .venv 裝好 requirements.txt + pyinstaller：
#   source .venv/bin/activate
#   pip install -r requirements.txt pyinstaller

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYINSTALLER=".venv/bin/pyinstaller"
if [ ! -x "$VENV_PYINSTALLER" ]; then
  echo "找不到 $VENV_PYINSTALLER，請先在 .venv 執行：pip install pyinstaller" >&2
  exit 1
fi

APP_NAME="TaiwanSubtitle"
VERSION="0.1.0"
DIST_DIR="dist"
WORK_DIR="build"
DMG_PATH="$DIST_DIR/${APP_NAME}-${VERSION}.dmg"

echo "==> 清除舊的打包產物"
rm -rf "$DIST_DIR" "$WORK_DIR"

echo "==> 執行 PyInstaller"
"$VENV_PYINSTALLER" packaging/TaiwanSubtitle.spec \
  --noconfirm \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR"

APP_PATH="$DIST_DIR/${APP_NAME}.app"
if [ ! -d "$APP_PATH" ]; then
  echo "打包失敗：找不到 $APP_PATH" >&2
  exit 1
fi

echo "==> 打包 .dmg"
rm -f "$DMG_PATH"
hdiutil create \
  -volname "$APP_NAME $VERSION" \
  -srcfolder "$APP_PATH" \
  -ov -format UDZO \
  "$DMG_PATH"

echo ""
echo "完成："
echo "  App: $APP_PATH"
echo "  DMG: $DMG_PATH"
echo ""
echo "注意：這個 build 沒有 Apple Developer 簽章/公證，"
echo "使用者第一次開啟時 macOS 會擋下，需要在「系統設定 → 隱私權與安全性」允許，"
echo "或執行：xattr -dr com.apple.quarantine \"$APP_PATH\""
