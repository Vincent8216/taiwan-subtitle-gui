# 台灣影片 → 繁體中文字幕

在 Apple Silicon（M 系列）Mac 上，用 MLX 把本機影片或音訊轉成繁體中文 `.srt` 字幕，同時輸出 `.txt` 逐字稿與 `.json` 時間戳資料。

- **GUI**：`gui.py` — 圖形介面，套件、模型、轉錄狀態一目了然，適合大部分使用情境。
- **CLI**：`transcribe.py` — 命令列工具，適合批次處理或自動化腳本。

兩者共用同一套轉錄邏輯，`gui.py` 只是 `transcribe.py` 的圖形外殼。

## 0. 不想自己架環境？直接下載打包好的 App

到 [Releases](https://github.com/Vincent8216/taiwan-subtitle-gui/releases) 下載最新的 `TaiwanSubtitle-x.y.z.dmg`，打開後把 `TaiwanSubtitle.app` 拖進「應用程式」即可，不需要另外裝 Python／pip 套件（FFmpeg 仍需自行 `brew install ffmpeg`，AI 模型會在第一次執行時透過 App 內建的「下載 / 更新模型」取得）。

這個 build **沒有 Apple Developer 簽章／公證**，第一次打開會被 Gatekeeper 擋下：

- 對 App 圖示按右鍵 → 開啟 → 再次確認開啟，或
- 到「系統設定 → 隱私權與安全性」允許，或
- 在 Terminal 執行：`xattr -dr com.apple.quarantine /Applications/TaiwanSubtitle.app`

想自己從原始碼打包，見文末的「打包成 .app / .dmg」一節。若不想用 App，也可以照下面步驟自己架 Python 環境用 GUI 或 CLI。

## 1. 確認 Mac 與 Homebrew

打開 Terminal，確認使用原生 Apple Silicon 環境：

```bash
uname -m
```

應顯示 `arm64`。建議使用 Homebrew Python 3.13（本專案在 3.13.15 上測試）；不要使用系統 Python，也不要用更新的 3.14 ——`mlx`/`mlx-audio` 等套件目前還沒有對應的 wheel。本工具不需要 Rosetta、CUDA 或 NVIDIA 套件。

若尚未安裝 Homebrew，執行：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安裝好後確認：

```bash
brew --version
brew install python@3.13 ffmpeg
```

## 2. 取得專案

```bash
git clone https://github.com/Vincent8216/taiwan-subtitle-gui.git
cd taiwan-subtitle-gui
```

（或用 GitHub 頁面的 Code → Download ZIP 解壓縮。）

## 3. 建立虛擬環境

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如果 Homebrew 安裝位置不同，先用 `which python3.13` 或 `brew --prefix python@3.13` 找出實際路徑。

## 4. 安裝套件

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 涵蓋 GUI（customtkinter）與轉錄流程（mlx、mlx-audio、numpy、soundfile、opencc、huggingface-hub）需要的全部 Python 套件；FFmpeg／FFprobe 是系統工具，走 Homebrew 安裝，不在這份清單裡。

**不確定套件裝得夠不夠齊？** 兩種方式都能逐一確認每個套件的安裝狀態，不用自己猜：

```bash
python transcribe.py --check-deps     # 只檢查，不安裝
python transcribe.py --install-deps   # 安裝所有缺少的 Python 套件
```

即使是完全乾淨、什麼套件都沒裝的電腦，直接執行 `python gui.py` 也不會壞掉——沒有 customtkinter 時會先跳出一個只靠標準庫 tkinter 畫的安裝精靈，同樣逐一列出每個套件的狀態，按「安裝缺少的 Python 套件」裝完、按「重新啟動」即可進入正式畫面。

## 5. 啟動 GUI（建議）

```bash
python gui.py
```

畫面由上到下分三塊，各自獨立檢查/安裝，彼此不會互相卡住：

1. **系統套件** — customtkinter、mlx、mlx-audio、numpy、soundfile、opencc、huggingface-hub 與 FFmpeg／FFprobe，每個套件一列，顯示已安裝的版本或安裝路徑。「安裝缺少的 Python 套件」跑 `pip install`；「安裝 FFmpeg (brew)」跑 `brew install ffmpeg`。
2. **AI 語音模型** — TEA-ASR 與 Qwen3 ForcedAligner 的下載狀態與容量；「下載 / 更新模型」會先下載再驗證，「變更存放目錄」可以把模型改放到別的磁碟或路徑（選擇搬移既有檔案的話不用重新下載）。
3. **選擇影音檔並開始轉錄** — 系統套件與模型都就緒後才會解鎖；轉錄途中**不會**再觸發任何下載或安裝，缺什麼都會直接報錯並提示你回上面兩塊補齊。

## 6. 或用 CLI 製作字幕

第一次使用前，先把模型下載好（跟 GUI 的「下載 / 更新模型」是同一套邏輯）：

```bash
python transcribe.py --check-models      # 只檢查本機是否已有模型
python transcribe.py --download-models   # 下載齊全部必要模型
```

- 語音辨識：`Alkd/TEA-ASR-1.1-MLX-4bit`（約 1.3 GB）
- 時間對齊：`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`（約 1.2 GB）

預設存放位置是專案內的 `models/huggingface/`；想放別的地方：

```bash
python transcribe.py --model-dir /Volumes/External/subtitle-models --move-existing-models
```

`--move-existing-models` 會把舊目錄裡已下載的模型直接搬過去，不用重新下載；選擇的目錄會記在專案根目錄的 `config.json`，之後 GUI／CLI 都會自動沿用。若已另外設定 `HF_HOME` 環境變數，程式會優先尊重它。

模型備妥後就能轉錄：

```bash
python transcribe.py "$HOME/Downloads/影片.mp4" --output-dir output --verbose
```

也可以先輸入 `python transcribe.py `，把 Finder 的檔案拖進 Terminal，再輸入 ` --output-dir output` 並按 Enter。支援本機 `mp4`、`mov`、`mkv`、`mp3`、`wav`、`m4a` 等 ffmpeg 可解碼且含音訊的檔案；不直接接受 YouTube URL，也不需要 yt-dlp。

完成後可在 `output` 找到同名的 `.srt`、`.txt`、`.json`。若省略 `--output-dir`，預設輸出到輸入檔所在資料夾；同名結果會被覆寫，請先保存要保留的版本。

轉錄途中預設不會連網補下載模型；若接受這個風險（例如在有網路的機器上跑批次），可加 `--allow-download` 讓缺少的模型自動下載後繼續。

常用參數：

```bash
# SRT 不要標點（TXT、JSON 不受影響）
python transcribe.py "$HOME/Downloads/影片.mp4" --output-dir output --no-punctuation

# 追加專有名詞，幫助辨識
python transcribe.py "$HOME/Downloads/影片.mp4" --output-dir output --hotword "新產品名"

# 完整參數列表
python transcribe.py --help
```

辨識及時間對齊結果仍建議人工校對；對齊失敗時程式會記錄警告並採用較粗的時間戳。

## 專案結構

```text
taiwan-subtitle-gui/
├── gui.py               # GUI 入口（customtkinter；缺套件時自動退回 tkinter 安裝精靈）
├── transcribe.py        # 轉錄邏輯 + CLI 入口 + 套件/模型檢查與下載 API
├── requirements.txt     # Python 套件版本
├── test/                # 單元測試
├── config.json          # 使用者選定的模型存放目錄（首次執行才會產生，已加入 .gitignore）
└── models/huggingface/  # 預設模型快取位置（已加入 .gitignore，不進版控）
```

## 測試

```bash
python -m unittest discover -s test -v
python -m pip check
```

主要相依套件固定在 `requirements.txt`；間接相依套件與遠端模型 snapshot 並未全部鎖定，因此不保證未來安裝環境逐位元相同。

## 打包成 .app / .dmg

用 PyInstaller 把 `gui.py` 連同所有套件打包成一個獨立的 `TaiwanSubtitle.app`（使用者不需要另外裝 Python）：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt pyinstaller
./packaging/build_app.sh
```

會產生：

- `dist/TaiwanSubtitle.app`
- `dist/TaiwanSubtitle-<版本>.dmg`

版本號在 `packaging/TaiwanSubtitle.spec` 的 `APP_VERSION` 設定。`mlx`／`mlx-audio` 帶有 PyInstaller 預設分析抓不到的 Metal shader（`.metallib`）與其他二進位資產，spec 檔已用 `collect_all()` 特別處理；若之後升級這兩個套件版本，直接用同一份 spec 重新打包即可，不需要額外調整。

打包出來的 App 不含 AI 模型（開啟後用內建的「下載 / 更新模型」取得），模型與設定檔預設存放在 `~/Library/Application Support/TaiwanSubtitle/`（不是 App 本體內部，重新安裝/更新 App 不會清掉已下載的模型）。FFmpeg 仍依賴系統安裝的版本，不會被打包進 App。

這個 build 沒有 Apple Developer 簽章／公證，發佈前請先在自己機器上完整跑過一次「系統套件檢查 → 下載模型 → 選擇檔案轉錄」確認沒問題。

## 模型與套件來源

- [TEA-ASR 官方模型](https://huggingface.co/JacobLinCool/TEA-ASR-1.1)
- [TEA-ASR MLX 模型](https://huggingface.co/Alkd/TEA-ASR-1.1-MLX-4bit)
- [Qwen3 Forced Aligner MLX 模型](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)
- [MLX Audio](https://github.com/Blaizzy/mlx-audio)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

模型與第三方套件依各自授權條款使用。本專案不包含模型權重或影音素材。
