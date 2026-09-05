# 台灣影片 → 繁體中文字幕

在 Apple Silicon（M 系列）Mac 上，將本機影片或音訊轉成繁體中文 `.srt` 字幕，同時輸出 `.txt` 逐字稿與 `.json` 時間戳資料。程式入口是 **`transcribe.py`**。

## 1. 確認 Mac 與 Homebrew

打開 Terminal，確認使用原生 Apple Silicon 環境：

```bash
uname -m
```

應顯示 `arm64`。建議使用 Homebrew Python 3.13；不要使用系統 Python。本工具不需要 Rosetta、CUDA 或 NVIDIA 套件。

若尚未安裝 Homebrew，執行：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

若安裝程式要求安裝 Xcode Command Line Tools，依畫面完成。Apple Silicon 的 Homebrew 通常位於 `/opt/homebrew`；請完成安裝程式顯示的 PATH 設定，再確認：

```bash
brew --version
```

## 2. 把專案放進 Downloads

```bash
cd ~/Downloads
mkdir -p taiwan-subtitle
cd taiwan-subtitle
```

下載本專案的壓縮檔（GitHub 專案頁的 **Code → Download ZIP**，或 Releases 提供的壓縮檔），解壓縮後，把其中的 `transcribe.py` 與 `requirements.txt` 複製到上面建立的資料夾。也可一併複製 README 與 test 資料夾。

確認檔案直接放在 `~/Downloads/taiwan-subtitle`，不要多包一層資料夾：

```bash
ls transcribe.py requirements.txt
```

## 3. 建立獨立 Python 環境

在 `~/Downloads/taiwan-subtitle` 中執行：

```bash
brew install ffmpeg python@3.13
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

啟動後，Terminal 提示字元通常會出現 `(.venv)`。如果 Homebrew 安裝位置不同，先用 `which python3.13` 找出 Python 路徑，替換建立 venv 指令中的 `/opt/homebrew/bin/python3.13`。

## 4. 安裝套件並確認 MLX

`requirements.txt` 保留原專案已記錄的測試版本，包含 MLX Audio、MLX、OpenCC 與 Hugging Face 等套件：

```bash
python -m pip install -r requirements.txt
python -c 'import mlx.core as mx; print(mx.default_device()); print(mx.sum(mx.array([1, 2, 3])).item())'
ffmpeg -version
```

MLX 檢查正常應顯示 `Device(gpu, 0)` 與計算結果 `6`；ffmpeg 應顯示版本資訊。OpenCC 使用 Python 套件，不需要另外透過 Homebrew 安裝。

## 5. 模型自動下載

```bash
mkdir -p models output
```

第一次執行時需要網路，工具會自動下載：

- 語音辨識：`Alkd/TEA-ASR-1.1-MLX-4bit`
- 時間對齊：`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`

預設快取位置：

```text
~/Downloads/taiwan-subtitle/
├── transcribe.py
├── requirements.txt
├── models/
│   └── huggingface/
│       └── hub/
└── output/
```

依教學估計，TEA-ASR 約 1.3 GB，Forced Aligner 約 1.2 GB；建議至少預留 4 GB 給模型，影音、暫存音訊和輸出另計。模型更新或備援模型下載可能需要更多空間。ASR 完成後會先釋放模型，再載入 aligner。

不需要執行 `hf download`，也不需要手動建立模型名稱目錄。若已設定 `HF_HOME` 或 `HF_HUB_CACHE`，程式會優先尊重這些環境變數。

若 TEA 載入或辨識失敗，程式會記錄錯誤，再嘗試 Qwen3-ASR 1.7B 8bit 與 0.6B 8bit；這可能另外下載備援模型。

## 6. 開始製作字幕

每次開啟新的 Terminal，先切換資料夾並啟動環境：

```bash
cd ~/Downloads/taiwan-subtitle
source .venv/bin/activate
```

將下方範例路徑換成自己的影片或音訊完整路徑；保留引號以支援空白：

```bash
python transcribe.py "$HOME/Downloads/影片.mp4" --output-dir output --verbose
```

也可以先輸入 `python transcribe.py `，把 Finder 的檔案拖進 Terminal，再輸入 ` --output-dir output` 並按 Enter。

支援本機 `mp4`、`mov`、`mkv`、`mp3`、`wav`、`m4a` 等 ffmpeg 可解碼且含音訊的檔案。不直接接受 YouTube URL；請先用瀏覽器或自己的下載流程取得本機影片。本工具不需要安裝 yt-dlp。

完成後，可在 `output` 找到同名的 `.srt`、`.txt`、`.json`。若省略 `--output-dir output`，預設輸出到輸入檔所在資料夾。相同輸出資料夾內，同名結果會被覆寫，請先保存要保留的版本。

SRT 不要標點：

```bash
python transcribe.py "$HOME/Downloads/影片.mp4" --output-dir output --no-punctuation
```

這個選項只影響 SRT，TXT 與 JSON 保留原文字。追加專有名詞：

```bash
python transcribe.py "$HOME/Downloads/影片.mp4" --output-dir output --hotword "新產品名"
```

查看完整參數：

```bash
python transcribe.py --help
```

辨識及時間對齊結果仍建議人工校對；對齊失敗時程式會記錄警告並採用較粗的時間戳。

## 測試與版本

若有複製 `test` 資料夾，可執行：

```bash
python -m unittest discover -s test -v
python -m pip check
```

主要相依套件固定在 `requirements.txt`；間接相依套件與遠端模型 snapshot 並未全部鎖定，因此不保證未來安裝環境逐位元相同。

## 模型與套件來源

- [TEA-ASR 官方模型](https://huggingface.co/JacobLinCool/TEA-ASR-1.1)
- [TEA-ASR MLX 模型](https://huggingface.co/Alkd/TEA-ASR-1.1-MLX-4bit)
- [Qwen3 Forced Aligner MLX 模型](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)
- [MLX Audio](https://github.com/Blaizzy/mlx-audio)

模型與第三方套件依各自授權條款使用。本壓縮檔不包含模型權重或影音素材。
