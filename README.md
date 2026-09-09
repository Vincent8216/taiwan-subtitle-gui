# 台灣影片 → 繁體中文字幕

在 Apple Silicon（M 系列）Mac 上，用 MLX 把本機影片或音訊轉成繁體中文 `.srt` 字幕，同時輸出 `.txt` 逐字稿與 `.json` 時間戳資料。

- **GUI**：`gui.py` — 圖形介面，套件、模型、轉錄狀態一目了然，適合大部分使用情境。
- **CLI**：`transcribe.py` — 命令列工具，適合批次處理或自動化腳本。

兩者共用同一套轉錄邏輯，`gui.py` 只是 `transcribe.py` 的圖形外殼。

## 📋 系統要求與效能基準

### 支援的環境
- **硬體**：Apple Silicon (M1/M1 Pro/M2/M3 系列) 原生執行
- **作業系統**：macOS 12.0 或更新
- **RAM**：最低 6 GB；**推薦 16 GB**（避免頻繁換頁）
- **磁碟空間**：至少 5 GB 空閒（模型 ~2.5 GB + 轉錄暫存 + 輸出檔）
- **Python**：3.13.x 專用（不支援 3.12、3.14）
- **FFmpeg**：透過 Homebrew 自動安裝；無最低版本要求（建議 6.1+）

### 不支援
- ❌ Intel Mac（需要 Rosetta；未測試且效能會大幅下降）
- ❌ 即時轉錄（不支援 Live 邊錄邊轉）
- ❌ 多語言混合（當前針對台灣華語最佳化）
- ❌ GPU 加速切換（MLX 已自動使用 Apple Silicon GPU；無法改用 NVIDIA）

### 效能基準

在 **M1 / 16 GB RAM / 本機 SSD** 環境測試（包含 ASR + Aligner）：

| 音訊長度 | 估計轉錄時間 | 備註 |
|---------|----------|------|
| 30 分鐘 | ~5-7 分鐘 | 短音訊，快速完成 |
| 1 小時 | ~10-15 分鐘 | 一般場景 |
| 2 小時 | ~20-30 分鐘 | 單進程；不支援並行 |

**影響因素**：
- 🎙️ **音訊品質**（雜訊多會延長 ASR 推理時間）
- 💾 **RAM 大小**（不足會頻繁換頁 ➜ 可能增加 50%+ 時間）
- 💿 **磁碟速度**（SSD 遠快於機械硬碟；外接 USB 可能成為瓶頸）
- 🔋 **Mac 溫度**（散熱不足時 M 系列會自動降頻）

**M2 Pro/M3/M3 Max 會更快；M1/M1 Pro 則作為基準。**

### 已知限制和風險

⚠️ **依賴版本控制不完整**：
- `requirements.txt` 鎖定了直接依賴版本，但間接依賴（例如 numpy 依賴的 BLAS）可能變化
- 風險：「在 A 機器能跑，在 B 機器不行」
- 改進計畫：預計 2026 Q3 改用 `poetry.lock` 或 `pip-compile` 解決

⚠️ **App 無開發者簽章**：
- 首次開啟需「右鍵 → 開啟」或執行 `xattr -dr com.apple.quarantine /Applications/TaiwanSubtitle.app`
- 影響：企業用戶無法使用（IT 政策不允許未簽章軟體）
- 改進計畫：評估成本效益後考慮申請 Apple Developer Program

⚠️ **模型降級無通知**：
- 若 TEA-ASR 失敗，會自動改用 Qwen3-ASR（使用者無感知）
- 風險：字幕品質可能略差（Qwen3 對台灣詞彙最佳化程度較低）
- 改進計畫：預計 2026 Q3 在 UI 中顯示降級通知

---

## 0. 不想自己架環境？直接下載打包好的 App

到 [Releases](https://github.com/Vincent8216/taiwan-subtitle-gui/releases) 下載最新的 `TaiwanSubtitle-x.y.z.dmg`（約 20 MB），打開後把 `TaiwanSubtitle.app` 拖進「應用程式」即可，不需要自己架 Python 環境。開啟 App 後，照畫面依序按兩顆按鈕就能補齊所有需要的東西：先按「系統套件」區塊的「安裝缺少的套件」（會一併裝好 Python 套件與 FFmpeg），再按「AI 語音模型」區塊的「下載 / 更新模型」。

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

### 推薦方式（精確可重現環境）⭐

```bash
python -m pip install -r requirements.lock
```

`requirements.lock` 是精確鎖定的依賴檔，包含所有直接和間接依賴的版本號。使用 lock 檔可保證環境完全可重現：
- ✅ 「在 A 機器能跑，在 B 機器也能跑」
- ✅ 依賴版本明確，no surprises
- ✅ CI/CD 環境最佳化

### 替代方式（更新依賴時使用）

如果想要最新版本的直接依賴，改用：
```bash
python -m pip install -r requirements.in
```

`requirements.in` 只包含直接依賴（customtkinter、mlx、numpy 等），pip 會自動解析和安裝間接依賴的最新相容版本。

### 更新依賴流程

若需要升級依賴（例如 mlx 從 0.32.2 → 0.33.0）：

```bash
# 1. 編輯 requirements.in
vim requirements.in  # 改 mlx==0.32.2 → mlx==0.33.0

# 2. 重新生成 lock 檔
pip-compile requirements.in -o requirements.lock

# 3. 本地測試新版本
pip install -r requirements.lock
python transcribe.py --check-deps

# 4. 確認無誤後提交
git add requirements.in requirements.lock
git commit -m "deps: upgrade mlx to 0.33.0"
```

### 詳細說明

`requirements.lock` 涵蓋 GUI（customtkinter）與轉錄流程（mlx、mlx-audio、numpy、soundfile、opencc、huggingface-hub）需要的**全部** Python 套件及其間接依賴；FFmpeg／FFprobe 是系統工具，走 Homebrew 安裝，不在這份清單裡。

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

1. **系統套件** — customtkinter、mlx、mlx-audio、numpy、soundfile、opencc、huggingface-hub 與 FFmpeg／FFprobe，每個套件一列，顯示已安裝的版本或安裝路徑。按一次「安裝缺少的套件」就會把 Python 套件（`pip install`）與 FFmpeg（`brew install ffmpeg`）一起裝好，不用分開點。
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

用 PyInstaller 把 `gui.py` 打包成一個獨立的 `TaiwanSubtitle.app`（使用者不需要另外裝 Python）：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt pyinstaller
./packaging/build_app.sh
```

會產生：

- `dist/TaiwanSubtitle.app`
- `dist/TaiwanSubtitle-<版本>.dmg`（約 20 MB）

版本號在 `packaging/TaiwanSubtitle.spec` 的 `APP_VERSION` 設定。

**這是一個精簡外殼，不含 mlx / mlx-audio / numpy / soundfile / opencc / huggingface-hub。** 這幾個套件加起來有幾百 MB（其中 `mlx` 的 Metal shader 檔就 130 MB），跟 AI 模型一樣，改成使用者開啟 App 後自己在「系統套件」面板按「安裝缺少的 Python 套件」才下載安裝——這樣 App 本體維持在 20 MB 左右，而不是把整個 ML 執行環境都塞進安裝檔。

實作方式：spec 檔用 `--exclude-module` 明確排除這幾個套件，並用 `collect_all('pip')` 把 pip 本身打包進去。按下「安裝缺少的套件」時，`transcribe.py` 呼叫 pip 的內部 API（因為打包後 `sys.executable` 是這個 App 自己，不是可執行 `-m pip` 的直譯器）把 Python 套件裝進 `~/Library/Application Support/TaiwanSubtitle/pylibs/`，同時視需要用 `brew install ffmpeg` 補裝 FFmpeg；執行期用 `sys.path` 接上 pylibs——同一個 arm64 + Python 3.13 的 ABI，裝在哪個資料夾都能被正常 import，包括 mlx 的 Metal shader 也能正確定位。AI 模型與設定檔也存在同一個 `Application Support/TaiwanSubtitle/` 資料夾（不是 App 本體內部），所以重新安裝/更新 App 不會清掉已下載的東西。

若之後升級 `mlx`／`mlx-audio` 版本，直接用同一份 spec 重新打包即可；使用者端下次按「安裝缺少的 Python 套件」（或「重新檢查」偵測到版本不同）就會抓到新版。

這個 build 沒有 Apple Developer 簽章／公證，發佈前請先在自己機器上完整跑過一次「系統套件檢查 → 安裝套件 → 下載模型 → 選擇檔案轉錄」確認沒問題。

## 常見問題與故障排除

### Q: 「缺少 numpy，請重新安裝 requirements.txt」
**A**: 重新安裝套件環境：
```bash
source .venv/bin/activate
python -m pip install --upgrade -r requirements.txt
# 或在 GUI 中按「重新檢查」和「安裝缺少的套件」
```

### Q: 轉錄卡住或進度緩慢
**A**: 
1. 檢查 Activity Monitor 的內存使用（是否達到 RAM 上限？）
2. 試試減小 `--chunk-size` (預設 240s)：
   ```bash
   python transcribe.py input.mp4 --chunk-size 120 --output-dir output
   ```
3. 查看 `output.json` 中的 `warnings` 欄位（是否有降級或失敗記錄？）
4. 確認外接硬碟的轉接速度（如果模型存在外接硬碟可能很慢）

### Q: 「Gatekeeper 擋住了 App」
**A**: 首次開啟打包版 App 時，按以下任一方式解除隔離：
- 方式 1：對 App 圖示按右鍵 → 開啟 → 再次確認開啟
- 方式 2：系統設定 → 隱私權與安全性 → 允許
- 方式 3：Terminal 執行 `xattr -dr com.apple.quarantine /Applications/TaiwanSubtitle.app`

### Q: 輸出的 SRT 時間碼不準
**A**:
- 這是 Forced Aligner 的限制（無法保證 100% 精準）
- 檢查 `output.json` 中的 `aligner.error` 欄位
- 如果顯示「fallback」，表示對齐失敗 ➜ 試執行 `--verbose` 查看詳細日誌：
  ```bash
  python transcribe.py input.mp4 --verbose --output-dir output
  ```
- 某些方言或口音可能需手動調整時間碼

### Q: GUI 中無法看到警告訊息
**A**:
- 警告會記錄在 `output.json` 的 `warnings` 欄位
- 目前 GUI 結果視窗不顯示詳細警告（計畫 2026 Q3 改進）
- 轉錄失敗時，查看 Terminal 或 GUI 的日誌輸出

### Q: 「找不到 FFmpeg」
**A**: FFmpeg 是系統工具，需透過 Homebrew 安裝：
```bash
# 確認是否安裝
which ffmpeg
brew install ffmpeg

# 確認版本
ffmpeg -version
```

### Q: 能在 Intel Mac 上跑嗎？
**A**: **不正式支援**。
- 理論上 MLX 可用 Rosetta 2 相容層執行，但效能會大幅下降（可能慢 10 倍）
- 預設 ASR 模型（TEA-ASR-MLX-4bit）針對 Apple Silicon 最佳化
- 如需在 Intel Mac 上使用，請考慮改用原始 PyTorch 版本的 TEA-ASR

### Q: 怎樣報告 Bug？
**A**: 請在 GitHub 上開 Issue，包含：
1. Mac 型號與 RAM（例如 M1 Pro / 16GB）
2. macOS 版本（執行 `sw_vers`）
3. Python 版本（執行 `python --version`）
4. 音訊檔的格式和長度
5. **`output.json` 中的診斷欄位**（若有產出的話）：
   ```json
   {
     "asr_model_used": "...",
     "asr_fallback_reason": "...",
     "warnings": [...]
   }
   ```
6. 完整的錯誤訊息（如有）

---

## 模型與套件來源

- [TEA-ASR 官方模型](https://huggingface.co/JacobLinCool/TEA-ASR-1.1)
- [TEA-ASR MLX 模型](https://huggingface.co/Alkd/TEA-ASR-1.1-MLX-4bit)
- [Qwen3 Forced Aligner MLX 模型](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)
- [MLX Audio](https://github.com/Blaizzy/mlx-audio)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

模型與第三方套件依各自授權條款使用。本專案不包含模型權重或影音素材。
