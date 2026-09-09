"""Taiwan Subtitle 轉錄工具的 GUI。

三件事完全分開，各自有獨立的「檢查／安裝」欄位，轉錄途中都不會再觸發：
1. 系統套件（Python 套件 + FFmpeg）－ 模擬「全新未安裝過的電腦」也要能開機檢查。
2. AI 語音模型（TEA-ASR / Qwen3 ForcedAligner）－ 先下載才能離線轉錄。
3. 轉錄本身。

gui.py 只在最外層依賴 customtkinter；transcribe.py 本身不在模組層級 import
任何重型套件（mlx / numpy / soundfile / opencc / huggingface_hub 都是函式內
延遲載入），所以即使一個套件都沒裝，`import transcribe` 也不會壞掉 —— 這讓
下面的「乾淨電腦」偵測與安裝流程可以安全運作。
"""

import importlib.util
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import transcribe

READY_COLOR = "#2CC985"
WARN_COLOR = "#E5A83C"
ERROR_COLOR = "#E5534B"
MUTED_COLOR = "gray"

# (import 名稱 或 執行檔名稱, 顯示標籤) － 用來畫出每個套件各自獨立的狀態列。
DEP_ROWS_INFO = [
    (name, label) for name, _pip, label in transcribe.REQUIRED_PYTHON_PACKAGES
] + [
    (name, label) for name, _brew, label in transcribe.REQUIRED_BINARIES
]


def _has_customtkinter() -> bool:
    return importlib.util.find_spec("customtkinter") is not None


# ============================================================================
# 備援畫面：customtkinter 本身都還沒裝時使用（純標準庫 tkinter，一定能開啟）。
# ============================================================================


class DependencyBootstrap(tk.Tk):
    """全新電腦的第一個畫面：只用得到 tkinter，負責把套件裝齊再重啟主程式。"""

    BG = "#1e1e1e"
    FG = "#e6e6e6"

    def __init__(self):
        super().__init__()
        self.title("Taiwan Subtitle 轉錄工具 － 安裝必要套件")
        self.geometry("620x560")
        self.resizable(False, False)
        self.configure(bg=self.BG)

        tk.Label(
            self,
            text="偵測到本機尚未安裝 GUI 所需套件，請先在下方安裝：",
            bg=self.BG,
            fg=self.FG,
            font=("Helvetica", 13, "bold"),
            anchor="w",
        ).pack(pady=(16, 8), padx=16, anchor="w")

        self.rows_frame = tk.Frame(self, bg=self.BG)
        self.rows_frame.pack(fill="x", padx=16)
        self.row_labels: dict[str, tk.Label] = {}
        for key, label in DEP_ROWS_INFO:
            row = tk.Label(
                self.rows_frame,
                text=f"• {label}：檢查中...",
                bg=self.BG,
                fg=MUTED_COLOR,
                anchor="w",
                justify="left",
                wraplength=580,
            )
            row.pack(fill="x", pady=1)
            self.row_labels[key] = row

        btn_frame = tk.Frame(self, bg=self.BG)
        btn_frame.pack(pady=10)
        self.recheck_btn = tk.Button(btn_frame, text="重新檢查", command=self.start_check)
        self.recheck_btn.pack(side="left", padx=4)
        self.install_btn = tk.Button(
            btn_frame, text="安裝缺少的套件", command=self.start_install
        )
        self.install_btn.pack(side="left", padx=4)
        self.restart_btn = tk.Button(
            btn_frame,
            text="安裝完成，重新啟動",
            command=self.restart_app,
            state="disabled",
        )
        self.restart_btn.pack(side="left", padx=4)

        self.log_text = tk.Text(
            self, height=16, bg="#111111", fg="#dddddd", font=("Menlo", 11)
        )
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(6, 16))
        self.log_text.configure(state="disabled")

        self.busy = False
        self.deps_ready = False
        self.after(200, self.start_check)

    # ---------- 小工具 ----------

    def log(self, message: str):
        self.after(0, self._log, message)

    def _log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_busy(self, busy: bool):
        self.after(0, self._set_busy, busy)

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.recheck_btn.configure(state=state)
        self.install_btn.configure(state=state)

    # ---------- 檢查 ----------

    def start_check(self):
        self.set_busy(True)
        threading.Thread(target=self._check_task, daemon=True).start()

    def _check_task(self):
        try:
            statuses = transcribe.check_dependencies()
            self.after(0, self._render_statuses, statuses)
        finally:
            self.set_busy(False)

    def _render_statuses(self, statuses):
        for status in statuses:
            row = self.row_labels.get(status.key)
            if row is None:
                continue
            color = READY_COLOR if status.installed else (
                ERROR_COLOR if status.kind == "binary" else WARN_COLOR
            )
            row.configure(text=f"• {status.label}：{status.detail}", fg=color)

        self.deps_ready = all(status.installed for status in statuses)
        self.restart_btn.configure(state="normal" if self.deps_ready else "disabled")
        if self.deps_ready:
            self.log("所有必要套件皆已就緒！請按「安裝完成，重新啟動」進入主程式。")

    # ---------- 安裝 ----------

    def start_install(self):
        self.set_busy(True)
        self.log("\n開始安裝缺少的套件...")
        threading.Thread(target=self._install_task, daemon=True).start()

    def _install_task(self):
        try:
            transcribe.install_python_packages(progress=lambda m, f=None: self.log(m))
            self.log("Python 套件安裝完成。")
            if not transcribe.inspect_binary("ffmpeg", "ffmpeg", "FFmpeg").installed:
                self.log("接著安裝 FFmpeg（Homebrew）...")
                transcribe.install_binary("ffmpeg", "ffmpeg", progress=lambda m, f=None: self.log(m))
                self.log("FFmpeg 安裝完成。")
        except Exception as exc:
            self.log(f"安裝失敗：{exc}")
        finally:
            self._check_task()

    def restart_app(self):
        self.log("重新啟動中...")
        self.destroy()
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])


# ============================================================================
# 主程式：customtkinter 已就緒時使用。
# ============================================================================

if _has_customtkinter():
    import customtkinter as ctk

    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    class ModernTranscribeApp(ctk.CTk):
        def __init__(self):
            super().__init__()

            self.title("Taiwan Subtitle 轉錄工具 (MLX 加速版)")
            self.geometry("640x820")
            self.resizable(False, False)

            self.deps_ready = False
            self.models_ready = False
            self.busy = False

            self.card = ctk.CTkFrame(self, corner_radius=15)
            self.card.pack(padx=20, pady=20, fill="both", expand=True)

            self.title_label = ctk.CTkLabel(
                self.card,
                text="廣東話 / 台語 / 華語語音轉字幕",
                font=ctk.CTkFont(size=20, weight="bold"),
            )
            self.title_label.pack(pady=(15, 5))

            # --- 1. 系統套件面板（Python 套件 + FFmpeg，各自獨立一列） ---
            self.dep_box = ctk.CTkFrame(self.card, corner_radius=10)
            self.dep_box.pack(pady=(10, 0), padx=20, fill="x")

            self.dep_title = ctk.CTkLabel(
                self.dep_box,
                text="系統套件（模擬全新電腦，逐一確認安裝狀態）",
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            self.dep_title.pack(anchor="w", padx=15, pady=(8, 2))

            self.dep_rows: dict[str, ctk.CTkLabel] = {}
            for key, label in DEP_ROWS_INFO:
                row = ctk.CTkLabel(
                    self.dep_box,
                    text=f"• {label}：檢查中...",
                    font=ctk.CTkFont(size=12),
                    text_color=MUTED_COLOR,
                    anchor="w",
                    justify="left",
                    wraplength=520,
                )
                row.pack(anchor="w", padx=15, pady=1, fill="x")
                self.dep_rows[key] = row

            self.dep_btn_frame = ctk.CTkFrame(self.dep_box, fg_color="transparent")
            self.dep_btn_frame.pack(anchor="e", padx=15, pady=(6, 10))

            self.dep_recheck_btn = ctk.CTkButton(
                self.dep_btn_frame,
                text="重新檢查",
                height=28,
                width=100,
                font=ctk.CTkFont(size=12),
                fg_color="transparent",
                border_width=1,
                command=self.start_check_deps,
            )
            self.dep_recheck_btn.pack(side="left", padx=(0, 8))

            self.dep_install_btn = ctk.CTkButton(
                self.dep_btn_frame,
                text="安裝缺少的套件",
                height=28,
                width=170,
                font=ctk.CTkFont(size=12),
                command=self.start_install_deps,
            )
            self.dep_install_btn.pack(side="left")

            # --- 2. 語言模型狀態與下載面板 ---
            self.model_box = ctk.CTkFrame(self.card, corner_radius=10)
            self.model_box.pack(pady=10, padx=20, fill="x")

            self.model_title = ctk.CTkLabel(
                self.model_box,
                text="AI 語音模型（需事前下載）",
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            self.model_title.pack(anchor="w", padx=15, pady=(8, 2))

            self.model_rows = {}
            for model_id in transcribe.REQUIRED_MODELS:
                row = ctk.CTkLabel(
                    self.model_box,
                    text=f"• {model_id}：檢查中...",
                    font=ctk.CTkFont(size=12),
                    text_color=MUTED_COLOR,
                    anchor="w",
                    justify="left",
                    wraplength=520,
                )
                row.pack(anchor="w", padx=15, pady=1, fill="x")
                self.model_rows[model_id] = row

            self.cache_label = ctk.CTkLabel(
                self.model_box,
                text=f"存放目錄：{transcribe.get_model_dir()}",
                font=ctk.CTkFont(size=11),
                text_color=MUTED_COLOR,
                anchor="w",
                wraplength=520,
            )
            self.cache_label.pack(anchor="w", padx=15, pady=(4, 2))

            self.model_btn_frame = ctk.CTkFrame(self.model_box, fg_color="transparent")
            self.model_btn_frame.pack(anchor="e", padx=15, pady=(2, 10))

            self.change_dir_btn = ctk.CTkButton(
                self.model_btn_frame,
                text="變更存放目錄",
                height=28,
                width=110,
                font=ctk.CTkFont(size=12),
                fg_color="transparent",
                border_width=1,
                command=self.start_change_model_dir,
            )
            self.change_dir_btn.pack(side="left", padx=(0, 8))

            self.recheck_btn = ctk.CTkButton(
                self.model_btn_frame,
                text="重新檢查",
                height=28,
                width=100,
                font=ctk.CTkFont(size=12),
                fg_color="transparent",
                border_width=1,
                command=self.start_check_models,
            )
            self.recheck_btn.pack(side="left", padx=(0, 8))

            self.download_btn = ctk.CTkButton(
                self.model_btn_frame,
                text="下載 / 更新模型",
                height=28,
                width=150,
                font=ctk.CTkFont(size=12),
                command=self.start_download_models,
            )
            self.download_btn.pack(side="left")

            # --- 3. 檔案選擇與轉錄按鈕 ---
            self.select_btn = ctk.CTkButton(
                self.card,
                text="選擇影音檔並開始轉錄",
                height=40,
                corner_radius=8,
                font=ctk.CTkFont(size=14, weight="bold"),
                command=self.select_file,
                state="disabled",
            )
            self.select_btn.pack(pady=10, padx=30, fill="x")

            # --- 4. 日誌輸出區 ---
            self.log_textbox = ctk.CTkTextbox(
                self.card, height=180, font=ctk.CTkFont(family="Menlo", size=11)
            )
            self.log_textbox.pack(pady=5, padx=20, fill="both", expand=True)
            self.log_textbox.configure(state="disabled")

            # 進度條
            self.progressbar = ctk.CTkProgressBar(self.card, width=380)
            self.progressbar.pack(pady=(8, 2))
            self.progressbar.set(0)

            self.progress_label = ctk.CTkLabel(
                self.card, text="", font=ctk.CTkFont(size=11), text_color=MUTED_COLOR
            )
            self.progress_label.pack(pady=(0, 12))

            # 啟動時自動檢測（只讀本機狀態，不下載、不安裝任何東西）
            self.run_in_background(self.check_deps_now, True)
            self.run_in_background(self.check_models_now, True)

        # ---------- 執行緒與 UI 更新 ----------

        def run_in_background(self, target, *args):
            threading.Thread(target=target, args=args, daemon=True).start()

        def ui(self, func, *args):
            """從工作執行緒安全地回到 Tk 主執行緒更新畫面。"""
            self.after(0, lambda: func(*args))

        def log(self, message: str):
            self.ui(self._log, message)

        def _log(self, message: str):
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", f"{message}\n")
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")

        def set_progress(self, fraction, text=""):
            self.ui(self._set_progress, fraction, text)

        def _set_progress(self, fraction, text):
            if fraction is None:
                if self.progressbar.cget("mode") != "indeterminate":
                    self.progressbar.configure(mode="indeterminate")
                    self.progressbar.start()
            else:
                if self.progressbar.cget("mode") == "indeterminate":
                    self.progressbar.stop()
                    self.progressbar.configure(mode="determinate")
                self.progressbar.set(max(0.0, min(1.0, fraction)))
            self.progress_label.configure(text=text)

        def set_busy(self, busy: bool):
            self.ui(self._set_busy, busy)

        def _set_busy(self, busy: bool):
            self.busy = busy
            state = "disabled" if busy else "normal"
            self.dep_recheck_btn.configure(state=state)
            self.dep_install_btn.configure(state=state)
            self.change_dir_btn.configure(state=state)
            self.recheck_btn.configure(state=state)
            self.download_btn.configure(state=state)
            self._refresh_transcribe_button()

        def _refresh_transcribe_button(self):
            can_run = self.deps_ready and self.models_ready and not self.busy
            self.select_btn.configure(state="normal" if can_run else "disabled")

        # ---------- 系統套件檢查／安裝 ----------

        def start_check_deps(self):
            self.set_busy(True)
            self.log("\n重新檢查系統套件...")
            self.run_in_background(self._check_deps_task)

        def _check_deps_task(self):
            try:
                self.check_deps_now(verbose=True)
            finally:
                self.set_busy(False)

        def check_deps_now(self, verbose: bool = False):
            """只用 find_spec / which 檢查，不 import 任何重型套件。"""
            statuses = transcribe.check_dependencies()
            for status in statuses:
                row = self.dep_rows.get(status.key)
                if row is None:
                    continue
                color = READY_COLOR if status.installed else (
                    ERROR_COLOR if status.kind == "binary" else WARN_COLOR
                )
                text = f"• {status.label}：{status.detail}"
                self.ui(lambda r=row, t=text, c=color: r.configure(text=t, text_color=c))

            self.deps_ready = all(status.installed for status in statuses)
            self.ui(self._refresh_transcribe_button)

            if not verbose:
                return
            if self.deps_ready:
                self.log("系統套件檢查完成：全部已安裝。")
            else:
                missing = [s.label for s in statuses if not s.installed]
                self.log(
                    "尚未安裝：" + "、".join(missing) + "\n"
                    "請按「安裝缺少的套件」，Python 套件與 FFmpeg 會一併安裝。"
                )

        def start_install_deps(self):
            self.set_busy(True)
            self.set_progress(0.0, "準備安裝套件...")
            self.log("\n開始安裝缺少的套件...")
            self.run_in_background(self.install_deps_process)

        def install_deps_process(self):
            try:
                transcribe.install_python_packages(progress=self.on_deps_progress)
                self.log("Python 套件安裝完成！")
                if not transcribe.inspect_binary("ffmpeg", "ffmpeg", "FFmpeg").installed:
                    self.log("接著安裝 FFmpeg（Homebrew）...")
                    transcribe.install_binary("ffmpeg", "ffmpeg", progress=self.on_deps_progress)
                    self.log("FFmpeg 安裝完成！")
                self.check_deps_now()
                self.set_progress(1.0, "套件已安裝")
            except Exception as exc:
                self.log(f"安裝失敗：{exc}")
                self.set_progress(0.0, "安裝失敗")
            finally:
                self.set_busy(False)

        def on_deps_progress(self, message: str, fraction=None):
            self.set_progress(fraction, message if fraction is not None else message)
            self.log(message)

        # ---------- 模型存放目錄 ----------

        def start_change_model_dir(self):
            new_dir = filedialog.askdirectory(
                title="選擇 AI 模型存放目錄",
                initialdir=str(transcribe.get_model_dir()),
            )
            if not new_dir:
                return

            old_dir = transcribe.get_model_dir()
            move_existing = False
            old_hub = old_dir / "hub"
            if Path(new_dir).resolve() == old_dir:
                self.log(f"\n已經是目前的存放目錄：{old_dir}")
                return
            if old_hub.is_dir() and any(old_hub.iterdir()):
                move_existing = messagebox.askyesno(
                    "搬移現有模型？",
                    f"目前已下載的模型位於：\n{old_dir}\n\n"
                    "是否搬移到新目錄？\n"
                    "選「否」則新目錄視為空白，該重新下載一次模型。",
                )

            self.set_busy(True)
            self.set_progress(0.0, "變更存放目錄中...")
            self.log(f"\n變更模型存放目錄為：{new_dir}")
            self.run_in_background(self.change_model_dir_process, new_dir, move_existing)

        def change_model_dir_process(self, new_dir, move_existing):
            try:
                resolved = transcribe.set_model_dir(
                    new_dir, move_existing=move_existing, progress=self.on_deps_progress
                )
                self.ui(lambda: self.cache_label.configure(text=f"存放目錄：{resolved}"))
                self.log(f"模型存放目錄已更新：{resolved}")
                self.check_models_now(verbose=True)
                self.set_progress(1.0, "存放目錄已更新")
            except Exception as exc:
                self.log(f"變更存放目錄失敗：{exc}")
                self.set_progress(0.0, "變更失敗")
            finally:
                self.set_busy(False)

        # ---------- 模型檢查／下載 ----------

        def start_check_models(self):
            self.set_busy(True)
            self.log("\n重新檢查本機模型快取...")
            self.run_in_background(self._check_models_task)

        def _check_models_task(self):
            try:
                self.check_models_now(verbose=True)
            finally:
                self.set_busy(False)

        def check_models_now(self, verbose: bool = False):
            """只讀本機快取，不連網、不載入模型。"""
            statuses = transcribe.check_models(transcribe.REQUIRED_MODELS)
            for status in statuses:
                row = self.model_rows.get(status.model_id)
                if row is None:
                    continue
                color = READY_COLOR if status.ready else WARN_COLOR
                text = f"• {status.label}：{status.detail}"
                self.ui(lambda r=row, t=text, c=color: r.configure(text=t, text_color=c))

            self.models_ready = all(status.ready for status in statuses)
            self.ui(self._refresh_transcribe_button)

            if not verbose:
                return
            if self.models_ready:
                self.log("模型檢查完成：已可離線轉錄。")
            else:
                missing = [s.label for s in statuses if not s.ready]
                self.log(
                    "尚未備妥：" + "、".join(missing) + "\n請先按「下載 / 更新模型」，"
                    "下載完成後才會解鎖轉錄按鈕。"
                )

        def start_download_models(self):
            self.set_busy(True)
            self.set_progress(0.0, "準備下載...")
            self.log("\n開始下載模型（TEA-ASR / Qwen3 ForcedAligner）...")
            self.run_in_background(self.download_process)

        def download_process(self):
            try:
                # 全部已備妥時按下按鈕，視為「更新」：重新與 Hub 比對檔案。
                transcribe.download_models(
                    transcribe.REQUIRED_MODELS,
                    progress=self.on_download_progress,
                    force=self.models_ready,
                )
                self.log("模型下載與驗證完成！")
                self.check_models_now()
                self.set_progress(1.0, "模型已備妥")
            except Exception as exc:
                self.log(f"下載失敗：{exc}")
                self.set_progress(0.0, "下載失敗")
            finally:
                self.set_busy(False)

        def on_download_progress(self, message: str, fraction=None):
            if fraction is None:
                self.set_progress(None, message)
            else:
                self.set_progress(fraction, f"{message}　{fraction * 100:.1f}%")
            if fraction is None or not message.startswith("下載中"):
                self.log(message)

        # ---------- 轉錄 ----------

        def select_file(self):
            file_path = filedialog.askopenfilename(
                filetypes=[("Media Files", "*.mov *.mp4 *.mp3 *.wav *.m4a *.mkv")]
            )
            if not file_path:
                return
            self.set_busy(True)
            self.ui(lambda: self.select_btn.configure(text="轉錄中..."))
            self.set_progress(None, "轉錄進行中...")
            self.log(f"\n--- 開始轉錄：{os.path.basename(file_path)} ---")
            self.run_in_background(self.run_process, file_path)

        def run_process(self, file_path):
            try:
                # allow_download=False：模型缺少時直接報錯，不會在這裡才下載。
                srt_path = transcribe.transcribe_file(file_path, allow_download=False)
                self.log(f"轉錄成功！字幕檔已儲存至：\n{srt_path}")
                self.set_progress(1.0, "轉錄完成")
            except Exception as exc:
                self.log(f"轉錄失敗：{exc}")
                self.set_progress(0.0, "轉錄失敗")
            finally:
                self.ui(lambda: self.select_btn.configure(text="選擇影音檔並開始轉錄"))
                self.set_busy(False)


if __name__ == "__main__":
    import multiprocessing

    # 打包成 .app 後 sys.executable 變成這個凍結執行檔本身，不是通用的
    # python 直譯器；multiprocessing 在某些情況下會嘗試用它 spawn 輔助行程
    # （例如 resource_tracker），freeze_support() 是 PyInstaller 官方建議的
    # 標準防護，避免那類輔助行程失敗時印出無害但嚇人的警告。
    multiprocessing.freeze_support()

    if _has_customtkinter():
        app = ModernTranscribeApp()
    else:
        app = DependencyBootstrap()
    app.mainloop()
