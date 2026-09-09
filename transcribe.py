#!/usr/bin/env python3
"""Local Taiwan-Mandarin video transcription to Traditional Chinese SRT.

The default route probes the TEA-ASR MLX checkpoint first.  Its model-card
compatibility predicate is applied before loading, and the first real ASR call
must still succeed.  When it does not, the pipeline records the exception and
falls back to a maintained Qwen3 MLX checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parent

# 打包成 .app 後，__file__ 會指向 bundle 內部（例如 Contents/Frameworks/）——
# 那裡不該存放會持續變動、GB 等級的模型快取或使用者設定，而且下次重新打包
# 就會被整個清空。frozen 執行時改用系統慣例的使用者資料目錄；一般以原始碼
# 執行（python transcribe.py / python gui.py）時則沿用專案內的資料夾，行為不變。
if getattr(sys, "frozen", False):
    APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "TaiwanSubtitle"
else:
    APP_DATA_DIR = PROJECT_DIR

# 打包版的 App 本身不含 mlx / mlx-audio / numpy / soundfile / opencc /
# huggingface-hub 這些重型套件（幾百 MB 的 Metal shader、模型程式碼）——
# 這些跟 AI 模型一樣，改成使用者自己在「系統套件」面板按「安裝缺少的
# Python 套件」時才下載，裝到這個資料夾，執行期用 sys.path 接上去用。
# 一般以原始碼執行時完全不受影響（沿用目前 venv 已安裝的套件）。
PYLIBS_DIR = APP_DATA_DIR / "pylibs"
if getattr(sys, "frozen", False):
    PYLIBS_DIR.mkdir(parents=True, exist_ok=True)
    if str(PYLIBS_DIR) not in sys.path:
        sys.path.insert(0, str(PYLIBS_DIR))

DEFAULT_HF_HOME = APP_DATA_DIR / "models" / "huggingface"
TEA_ASR_MLX_MODEL = "Alkd/TEA-ASR-1.1-MLX-4bit"
TEA_ASR_MLX_COMPATIBILITY_NOTE = (
    "套用 TEA-ASR-1.1 MLX 模型卡要求的 quantization predicate，讓 8-bit audio tower "
    "與 4-bit text decoder 正確載入。"
)
TEA_ASR_MLX_TEXT_CLEANUP_NOTE = (
    "清除 tokenizer 相容性造成的 Unicode Private Use Area 殘留字元，不改動時間戳。"
)
DEFAULT_ASR_CANDIDATES = (
    TEA_ASR_MLX_MODEL,
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
)
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
DEFAULT_CHUNK_SECONDS = 240.0
DEFAULT_OVERLAP_SECONDS = 1.5
MAX_ALIGNMENT_SECONDS = 270.0

# Add project-specific terms here.  The current MLX-Audio API accepts this
# list as hotwords; older compatible APIs can receive the generated context.
HOTWORDS = [
    "NVIDIA",
    "GeForce",
    "RTX",
    "Blackwell",
    "Vera Rubin",
    "CoWoS",
    "TSMC",
    "OpenAI",
    "Gemini",
    "Claude",
    "Qwen",
    "Kimi",
    "Cerebras",
]

try:
    import numpy as np
except ImportError:  # pragma: no cover - gives a clearer runtime error later
    np = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - gives a clearer runtime error later
    sf = None


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class PreparedAudioChunk:
    chunk: AudioChunk
    path: Path


@dataclass
class ASRChunk:
    index: int
    start: float
    end: float
    text: str
    language: str
    audio_path: Path
    segments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TokenTimestamp:
    text: str
    start: float
    end: float
    source_chunk: int | None = None
    fallback: str | None = None


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass
class ModelInfo:
    name: str
    backend: str = "mlx"
    load_seconds: float = 0.0
    active_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    compatibility: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    tokens: list[TokenTimestamp]
    fallback: str | None = None
    error: str | None = None


@dataclass
class TranscriptBundle:
    full_transcript: str
    raw_transcript: str
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    characters: list[dict[str, Any]]
    model: dict[str, Any]
    aligner: dict[str, Any]
    processing_time: float
    audio_duration: float
    realtime_factor: float
    warnings: list[str]
    cues: list[SubtitleCue]


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("缺少 numpy，請在 taiwan-subtitle/.venv 重新安裝 requirements.txt")
    return np


def _require_soundfile() -> Any:
    if sf is None:
        raise RuntimeError("缺少 soundfile，請在 taiwan-subtitle/.venv 重新安裝 requirements.txt")
    return sf


CONFIG_PATH = APP_DATA_DIR / "config.json"


def load_config() -> dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_model_dir() -> Path:
    """Resolve where models live: env var > 使用者上次選定 > 專案內建預設值。"""

    env_value = os.environ.get("HF_HOME")
    if env_value:
        return Path(env_value).expanduser().resolve()
    configured = load_config().get("model_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_HF_HOME


def set_model_dir(
    path: str | Path, move_existing: bool = False, progress: Any = None
) -> Path:
    """把模型存放目錄換成 path，寫進 config.json 讓下次啟動也記得。

    move_existing=True 時會把舊目錄底下的 hub 快取整包搬過去，
    這樣換目錄不用重新下載一次已經有的模型。
    """

    report = _progress_reporter(progress)
    new_dir = Path(path).expanduser().resolve()
    old_dir = get_model_dir()
    new_dir.mkdir(parents=True, exist_ok=True)

    if move_existing and old_dir != new_dir:
        old_hub = old_dir / "hub"
        new_hub = new_dir / "hub"
        if old_hub.is_dir() and any(old_hub.iterdir()):
            if new_hub.exists() and any(new_hub.iterdir()):
                raise RuntimeError(
                    f"目標資料夾已有模型快取（{new_hub}），"
                    "請改選空資料夾，或關閉「搬移現有模型」再試一次。"
                )
            report(f"搬移模型快取：{old_hub} -> {new_hub}", None)
            shutil.move(str(old_hub), str(new_hub))
            report("搬移完成。", 1.0)

    config = load_config()
    config["model_dir"] = str(new_dir)
    save_config(config)

    os.environ["HF_HOME"] = str(new_dir)
    os.environ["HF_HUB_CACHE"] = str(new_dir / "hub")
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    return new_dir


def configure_huggingface_cache() -> Path:
    """Keep model downloads inside the configured 模型存放目錄。"""

    cache_home = get_model_dir()
    os.environ["HF_HOME"] = str(cache_home)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_home / "hub"))
    cache_home.mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    return cache_home


# --- 模型下載 / 檢查：與轉錄流程完全分離的獨立 API ---------------------------
#
# GUI（或 CLI 的 --check-models / --download-models）先用這組函式把模型備妥，
# run_transcription() 預設不再連網下載，避免使用者按下轉錄後才卡在下載。

REQUIRED_MODELS: tuple[str, ...] = (TEA_ASR_MLX_MODEL, DEFAULT_ALIGNER_MODEL)
MODEL_LABELS: dict[str, str] = {
    TEA_ASR_MLX_MODEL: "語音辨識模型 (TEA-ASR)",
    DEFAULT_ALIGNER_MODEL: "時間軸對齊模型 (Qwen3 ForcedAligner)",
}
_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin", ".gguf")


@dataclass
class ModelStatus:
    """One model's local-cache state, as shown on the GUI 模型面板."""

    model_id: str
    label: str
    ready: bool
    detail: str
    path: Path | None = None
    size_bytes: int = 0


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def model_cache_dir(model_id: str) -> Path:
    cache_home = configure_huggingface_cache()
    hub_dir = Path(os.environ.get("HF_HUB_CACHE", str(cache_home / "hub")))
    return hub_dir / ("models--" + model_id.replace("/", "--"))


def local_snapshot_dir(model_id: str) -> Path | None:
    """Locate the cached snapshot folder without importing huggingface_hub."""

    repo_dir = model_cache_dir(model_id)
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    head = repo_dir / "refs" / "main"
    if head.is_file():
        try:
            pinned = snapshots / head.read_text(encoding="utf-8").strip()
        except OSError:
            pinned = None
        if pinned is not None and pinned.is_dir():
            return pinned
    folders = [item for item in snapshots.iterdir() if item.is_dir()]
    if not folders:
        return None
    return max(folders, key=lambda item: item.stat().st_mtime)


def _snapshot_files(snapshot: Path) -> list[Path]:
    return [item for item in snapshot.rglob("*") if item.is_file()]


def inspect_model(model_id: str) -> ModelStatus:
    """Report whether one model is fully downloaded in the local cache."""

    label = model_label(model_id)
    snapshot = local_snapshot_dir(model_id)
    if snapshot is None:
        return ModelStatus(model_id, label, False, "尚未下載")

    files = _snapshot_files(snapshot)
    if not any(item.name == "config.json" for item in files):
        return ModelStatus(model_id, label, False, "快取不完整（缺少 config.json）", snapshot)

    weights = [item for item in files if item.suffix in _WEIGHT_SUFFIXES]
    if not weights:
        return ModelStatus(model_id, label, False, "快取不完整（缺少權重檔）", snapshot)

    total = 0
    for item in files:
        try:
            total += item.stat().st_size
        except OSError:
            return ModelStatus(model_id, label, False, "快取損毀（檔案連結失效）", snapshot)
    if any(item.stat().st_size == 0 for item in weights):
        return ModelStatus(model_id, label, False, "快取不完整（權重檔為空）", snapshot)
    if any((model_cache_dir(model_id) / "blobs").glob("*.incomplete")):
        return ModelStatus(model_id, label, False, "上次下載未完成", snapshot)

    return ModelStatus(model_id, label, True, f"已備妥（{_format_bytes(total)}）", snapshot, total)


def check_models(model_ids: Sequence[str] = REQUIRED_MODELS) -> list[ModelStatus]:
    return [inspect_model(model_id) for model_id in model_ids]


def models_ready(model_ids: Sequence[str] = REQUIRED_MODELS) -> bool:
    return all(status.ready for status in check_models(model_ids))


def _progress_reporter(progress: Any) -> Any:
    """Normalise the optional callback into progress(message, fraction)."""

    if progress is None:
        return lambda message, fraction=None: None

    def report(message: str, fraction: float | None = None) -> None:
        try:
            progress(message, fraction)
        except Exception:
            pass

    return report


def _download_progress_tqdm(report: Any) -> Any:
    """A silent tqdm subclass that forwards aggregate byte progress instead."""

    try:
        from tqdm.auto import tqdm as base_tqdm
    except ImportError:
        return None

    import threading

    lock = threading.Lock()
    state = {"total": 0, "done": 0, "last": 0.0}

    class ProgressTqdm(base_tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            with lock:
                state["total"] += int(self.total or 0)

        def update(self, n: int | None = 1) -> Any:
            with lock:
                state["done"] += int(n or 0)
                total, done = state["total"], state["done"]
                now = time.monotonic()
                due = now - state["last"] >= 0.5
                if due:
                    state["last"] = now
            if due and total > 0:
                fraction = min(done / total, 1.0)
                report(
                    f"下載中：{_format_bytes(done)} / {_format_bytes(total)}",
                    fraction,
                )
            return super().update(n)

    return ProgressTqdm


class _CacheGrowthReporter:
    """Fallback progress: poll the repo cache size when tqdm hooks are absent."""

    def __init__(self, model_id: str, report: Any) -> None:
        import threading

        self._repo_dir = model_cache_dir(model_id)
        self._report = report
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _current_bytes(self) -> int:
        total = 0
        blobs = self._repo_dir / "blobs"
        if not blobs.is_dir():
            return 0
        for item in blobs.iterdir():
            try:
                total += item.stat().st_size
            except OSError:
                continue
        return total

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            self._report(f"下載中：{_format_bytes(self._current_bytes())}", None)

    def __enter__(self) -> "_CacheGrowthReporter":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()


def _snapshot_download_callable() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少 huggingface-hub 套件，請先執行：pip install -r requirements.txt"
        ) from exc
    return snapshot_download


def download_model(model_id: str, progress: Any = None, force: bool = False) -> ModelStatus:
    """Fetch one model into the project cache; no ASR model is loaded into RAM."""

    configure_huggingface_cache()
    report = _progress_reporter(progress)
    status = inspect_model(model_id)
    if status.ready and not force:
        report(f"{status.label}：{status.detail}，略過下載", 1.0)
        return status

    snapshot_download = _snapshot_download_callable()
    os.environ.pop("HF_HUB_OFFLINE", None)
    report(f"{model_label(model_id)}：開始下載 {model_id}", 0.0)

    kwargs: dict[str, Any] = {"repo_id": model_id}
    tqdm_class = _download_progress_tqdm(report)
    names, accepts_kwargs = _supported_parameters(snapshot_download)
    if tqdm_class is not None and ("tqdm_class" in names or accepts_kwargs):
        kwargs["tqdm_class"] = tqdm_class
    try:
        if "tqdm_class" in kwargs:
            snapshot_download(**kwargs)
        else:
            with _CacheGrowthReporter(model_id, report):
                snapshot_download(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"{model_label(model_id)} 下載失敗：{exc}") from exc

    status = inspect_model(model_id)
    if not status.ready:
        raise RuntimeError(f"{status.label} 下載後仍不完整：{status.detail}")
    report(f"{status.label}：{status.detail}", 1.0)
    return status


def download_models(
    model_ids: Sequence[str] = REQUIRED_MODELS,
    progress: Any = None,
    force: bool = False,
) -> list[ModelStatus]:
    """Download every required model, reporting one overall 0..1 fraction."""

    report = _progress_reporter(progress)
    model_ids = list(model_ids)
    statuses: list[ModelStatus] = []
    for index, model_id in enumerate(model_ids):

        def step(message: str, fraction: float | None = None, index: int = index) -> None:
            if fraction is None:
                report(message, None)
                return
            report(message, (index + fraction) / len(model_ids))

        statuses.append(download_model(model_id, progress=step, force=force))
    report("全部模型已備妥。", 1.0)
    return statuses


def _missing_models_error(missing: Sequence[ModelStatus]) -> RuntimeError:
    detail = "、".join(f"{status.label}（{status.detail}）" for status in missing)
    return RuntimeError(
        f"模型尚未備妥：{detail}。請先按下 GUI 的「下載 / 更新模型」按鈕，"
        "或執行 python transcribe.py --download-models。"
    )


def ensure_models_ready(
    model_ids: Sequence[str] = REQUIRED_MODELS,
    allow_download: bool = False,
    progress: Any = None,
) -> list[ModelStatus]:
    """Raise an actionable error instead of downloading mid-transcription."""

    statuses = check_models(model_ids)
    missing = [status for status in statuses if not status.ready]
    if not missing:
        return statuses
    if allow_download:
        download_models([status.model_id for status in missing], progress=progress)
        return check_models(model_ids)
    raise _missing_models_error(missing)


def resolve_asr_candidates(
    asr_model: str | None = None,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
    allow_download: bool = False,
    progress: Any = None,
) -> tuple[str, ...]:
    """Return the ASR ids usable offline, after checking the aligner is cached."""

    if allow_download:
        ensure_models_ready(
            [asr_model or TEA_ASR_MLX_MODEL, aligner_model],
            allow_download=True,
            progress=progress,
        )
        return (asr_model,) if asr_model else DEFAULT_ASR_CANDIDATES

    candidates = (asr_model,) if asr_model else DEFAULT_ASR_CANDIDATES
    ready = tuple(status.model_id for status in check_models(candidates) if status.ready)
    missing = [] if ready else [inspect_model(candidates[0])]
    aligner_status = inspect_model(aligner_model)
    if not aligner_status.ready:
        missing.append(aligner_status)
    if missing:
        raise _missing_models_error(missing)
    return ready


# --- 執行環境套件檢查 / 安裝：模擬「全新未安裝過的電腦」 ------------------------
#
# 這裡的檢查全部只用 importlib.util.find_spec / importlib.metadata 與
# shutil.which，不會 import 任何一個重型套件（mlx、numpy...），所以在完全
# 沒裝任何依賴的乾淨環境也能安全跑，用來畫出「每個套件各自獨立」的安裝狀態列。

@dataclass
class DependencyStatus:
    """One required Python package or system binary's install state."""

    key: str
    label: str
    kind: str  # "python" | "binary"
    installed: bool
    detail: str
    pip_requirement: str | None = None
    install_hint: str | None = None


# (import 名稱, requirements.txt 內對應套件名, 顯示用標籤)
REQUIRED_PYTHON_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("customtkinter", "customtkinter", "GUI 介面套件 (customtkinter)"),
    ("mlx", "mlx", "Apple MLX 運算框架"),
    ("mlx_audio", "mlx-audio", "MLX 語音辨識套件 (mlx-audio)"),
    ("numpy", "numpy", "數值運算套件 (numpy)"),
    ("soundfile", "soundfile", "音訊讀寫套件 (soundfile)"),
    ("opencc", "opencc-python-reimplemented", "簡繁轉換套件 (opencc)"),
    ("huggingface_hub", "huggingface-hub", "模型下載套件 (huggingface-hub)"),
)

# (執行檔名稱, brew 套件名稱, 顯示用標籤)
REQUIRED_BINARIES: tuple[tuple[str, str, str], ...] = (
    ("ffmpeg", "ffmpeg", "音訊/影片轉檔工具 (ffmpeg)"),
    ("ffprobe", "ffmpeg", "媒體資訊探測工具 (ffprobe)"),
)


def _requirement_line_for(pip_name: str) -> str:
    """Return the exact pinned line from requirements.txt, else the bare name."""

    base = pip_name.split("[")[0].lower()
    req_file = PROJECT_DIR / "requirements.txt"
    try:
        lines = req_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return pip_name
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\[=<>!~]", line, maxsplit=1)[0].strip().lower()
        if name == base:
            return line
    return pip_name


def inspect_python_package(import_name: str, pip_name: str, label: str) -> DependencyStatus:
    """Check via find_spec/metadata only — never actually imports the package."""

    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return DependencyStatus(
            import_name, label, "python", False, "尚未安裝",
            pip_requirement=_requirement_line_for(pip_name),
        )
    try:
        version = importlib_metadata.version(pip_name.split("[")[0])
    except importlib_metadata.PackageNotFoundError:
        version = None
    detail = f"已安裝（v{version}）" if version else "已安裝"
    return DependencyStatus(import_name, label, "python", True, detail)


def inspect_binary(binary_name: str, brew_package: str, label: str) -> DependencyStatus:
    # 首先用 shutil.which 查找（使用當前系統 PATH）
    path = shutil.which(binary_name)

    if not path:
        # 如果 shutil.which 找不到，檢查常見的 Homebrew 位置
        for common_path in [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            str(Path.home() / ".homebrew" / "bin"),
        ]:
            candidate = Path(common_path) / binary_name
            if candidate.exists():
                path = str(candidate)
                break

    if not path:
        # 都找不到時，嘗試直接執行看看是否能成功
        try:
            result = subprocess.run(
                [binary_name, "--version"],
                capture_output=True,
                timeout=2,
                check=True,
                text=True,
            )
            path = "（已安裝但路徑不在 PATH）"
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return DependencyStatus(
                binary_name, label, "binary", False, "尚未安裝",
                install_hint=f"brew install {brew_package}",
            )

    if path:
        return DependencyStatus(binary_name, label, "binary", True, f"已安裝（{path}）")
    return DependencyStatus(
        binary_name, label, "binary", False, "尚未安裝",
        install_hint=f"brew install {brew_package}",
    )


def check_dependencies() -> list[DependencyStatus]:
    statuses = [inspect_python_package(*item) for item in REQUIRED_PYTHON_PACKAGES]
    statuses += [inspect_binary(*item) for item in REQUIRED_BINARIES]
    return statuses


def dependencies_ready() -> bool:
    return all(status.installed for status in check_dependencies())


def _resolve_install_targets(packages: Sequence[str] | None) -> list[str]:
    statuses = [s for s in check_dependencies() if s.kind == "python"]
    if packages is None:
        return [s.pip_requirement or s.key for s in statuses if not s.installed]
    wanted = set(packages)
    return [s.pip_requirement or s.key for s in statuses if s.key in wanted]


class _LineTee:
    """A writable that splits pip's printed output into report() calls.

    pip's default progress bar overwrites itself with bare "\r" (no "\n"),
    so it's treated as a line break too — otherwise it never shows up until
    the whole install is done, sitting unseen in the buffer the entire time.
    """

    def __init__(self, report: Any) -> None:
        self._report = report
        self._buffer = ""

    def write(self, chunk: str) -> int:
        self._buffer += chunk.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._report(line, None)
        return len(chunk)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _install_python_packages_frozen(targets: list[str], report: Any) -> None:
    """打包版 App 沒有可執行的 sys.executable -m pip；改成呼叫 pip 內部 API，
    把套件裝進 PYLIBS_DIR，執行期再透過 sys.path 讀到（見檔頭的 bootstrap）。
    """

    import contextlib
    import threading
    import time as _time

    try:
        from pip._internal.cli.main import main as pip_main
    except ImportError as exc:
        raise RuntimeError(
            "這個 App build 沒有內建 pip，無法自動安裝套件。請回報這個問題，"
            "或改用原始碼＋requirements.txt 執行。"
        ) from exc

    PYLIBS_DIR.mkdir(parents=True, exist_ok=True)
    args = [
        "install",
        "--upgrade",
        "--target",
        str(PYLIBS_DIR),
        "--no-warn-script-location",
        "--progress-bar",
        "off",
        *targets,
    ]
    tee = _LineTee(report)

    # pip 內部用 logging/Rich 輸出，被 in-process 呼叫時不一定能可靠攔截到
    # 逐行進度（依 pip 版本而異）；用心跳執行緒保證 GUI 進度條在等待期間
    # 持續有動靜，而不是像卡住一樣停在 0%。
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        started = _time.monotonic()
        while not stop_heartbeat.wait(1.5):
            report(f"安裝中…（已進行 {int(_time.monotonic() - started)} 秒）", None)

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                exit_code = pip_main(args)
            except SystemExit as exc:
                # pip 的內部 argparse 出錯（例如未來版本不接受某個參數）時，
                # 會直接 sys.exit() 而不是回傳值；轉成一般例外，才不會把整個
                # 呼叫端（GUI 背景執行緒／CLI）也跟著吞掉、卻沒有任何訊息。
                exit_code = exc.code if isinstance(exc.code, int) else 1
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2.0)
    if tee._buffer:
        report(tee._buffer, None)
    if exit_code != 0:
        raise RuntimeError(f"pip install 失敗（exit code {exit_code}）：{', '.join(targets)}")

    # 剛裝好的套件要能在這次執行期就被 find_spec/import 找到。
    importlib.invalidate_caches()


def install_python_packages(
    packages: Sequence[str] | None = None,
    progress: Any = None,
) -> None:
    """安裝目前缺少（或指定）的必要 Python 套件。

    一般以原始碼執行：跑 `sys.executable -m pip install ...`，裝進當前
    venv，行為和過去一樣。打包成 App 時：sys.executable 是凍結後的執行檔本
    身，不是可用的直譯器，改成呼叫 pip 內部 API 把套件裝進
    ~/Library/Application Support/TaiwanSubtitle/pylibs，執行期用 sys.path
    接上去（跟 AI 模型一樣，App 本體不包含這些套件，第一次使用才下載）。
    """

    report = _progress_reporter(progress)
    targets = _resolve_install_targets(packages)

    if not targets:
        report("所有 Python 套件皆已安裝，略過安裝。", 1.0)
        return

    report(f"開始安裝：{', '.join(targets)}", 0.0)

    if getattr(sys, "frozen", False):
        _install_python_packages_frozen(targets, report)
        report("Python 套件安裝完成。", 1.0)
        return

    command = [sys.executable, "-m", "pip", "install", "--upgrade", *targets]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            report(line, None)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"pip install 失敗（exit code {process.returncode}）：{', '.join(targets)}")
    report("Python 套件安裝完成。", 1.0)


def install_binary(binary_name: str, brew_package: str, progress: Any = None) -> None:
    """Install one system tool via Homebrew; raises with guidance if brew is absent."""

    report = _progress_reporter(progress)
    brew = shutil.which("brew")

    # macOS App 通過 GUI 啟動時 PATH 可能不完整，檢查常見的 Homebrew 位置
    if not brew:
        for common_path in [
            "/opt/homebrew/bin/brew",
            "/usr/local/bin/brew",
            Path.home() / ".homebrew" / "bin" / "brew",
        ]:
            if Path(common_path).exists():
                brew = common_path
                break

    if not brew:
        raise RuntimeError(
            f"找不到 Homebrew，無法自動安裝 {binary_name}。\n"
            f"請先安裝 Homebrew (https://brew.sh)，\n"
            f"再執行：brew install {brew_package}"
        )
    report(f"開始安裝：brew install {brew_package}", 0.0)
    process = subprocess.Popen(
        [brew, "install", brew_package],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            report(line, None)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"brew install {brew_package} 失敗（exit code {process.returncode}）")

    # 安裝完成後，嘗試更新 PATH 以便後續檢查能找到新安裝的工具
    # （特別是 GUI App 啟動時 PATH 可能不完整的情況）
    for common_bin_path in ["/opt/homebrew/bin", "/usr/local/bin"]:
        if common_bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{common_bin_path}:{os.environ['PATH']}"

    report(f"{binary_name} 安裝完成。", 1.0)


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(item) for item in command],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到系統工具：{command[0]}，請先用 Homebrew 安裝。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"命令失敗：{command[0]} {detail}") from exc


def probe_media(path: Path) -> MediaInfo:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{path}")
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe 無法讀取媒體長度：{path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"輸入媒體沒有有效音訊長度：{path}")
    return MediaInfo(path=path, duration=duration)


def extract_audio(path: Path, destination: Path) -> Path:
    """Decode the first audio stream without touching the source media."""

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(Path(path).resolve()),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg 沒有產生有效暫存 WAV：{destination}")
    return destination


def scan_energy(wav_path: Path, frame_seconds: float = 0.25) -> list[float]:
    """Scan RMS energy in bounded memory for low-energy chunk boundaries."""

    audio_file = _require_soundfile()
    arrays = _require_numpy()
    if frame_seconds <= 0:
        raise ValueError("frame_seconds 必須大於 0")
    energies: list[float] = []
    with audio_file.SoundFile(str(wav_path), "r") as source:
        sample_rate = int(source.samplerate)
        frame_size = max(1, round(sample_rate * frame_seconds))
        block_size = sample_rate * 30
        while True:
            block = source.read(block_size, dtype="float32", always_2d=False)
            if len(block) == 0:
                break
            block_array = arrays.asarray(block, dtype=arrays.float32)
            for start in range(0, len(block_array), frame_size):
                frame = block_array[start : start + frame_size]
                if len(frame) == 0:
                    continue
                rms = arrays.sqrt(arrays.mean(arrays.square(frame)))
                energies.append(float(rms))
            del block_array
    return energies


def build_chunks(
    duration: float,
    energies: Sequence[float],
    target_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    frame_seconds: float = 0.25,
) -> list[AudioChunk]:
    """Build overlapping windows and prefer the quietest nearby boundary."""

    if duration <= 0:
        return []
    if target_seconds <= 0:
        raise ValueError("target_seconds 必須大於 0")
    if overlap_seconds < 0 or overlap_seconds >= target_seconds:
        raise ValueError("overlap_seconds 必須介於 0 與 target_seconds 之間")
    if frame_seconds <= 0:
        raise ValueError("frame_seconds 必須大於 0")

    target_seconds = min(float(target_seconds), MAX_ALIGNMENT_SECONDS - 10.0)
    half_overlap = float(overlap_seconds) / 2.0
    chunks: list[AudioChunk] = []
    cursor = 0.0

    def find_boundary(desired: float) -> float:
        if not energies:
            return min(duration, desired)
        search_start = max(cursor + min(30.0, target_seconds * 0.25), desired - 20.0)
        search_end = min(duration, desired + 20.0)
        first = max(0, int(math.floor(search_start / frame_seconds)))
        last = min(len(energies) - 1, int(math.ceil(search_end / frame_seconds)))
        if first > last:
            return min(duration, desired)
        best_index = min(range(first, last + 1), key=lambda index: energies[index])
        candidate = best_index * frame_seconds
        if candidate <= cursor + 5.0:
            candidate = desired
        return max(cursor + 5.0, min(duration, candidate))

    while cursor < duration - 1e-6:
        remaining = duration - cursor
        if remaining <= target_seconds + half_overlap:
            chunks.append(AudioChunk(len(chunks), round(cursor, 3), round(duration, 3)))
            break

        desired = cursor + target_seconds
        boundary = find_boundary(desired)
        window_end = min(duration, boundary + half_overlap)
        next_cursor = max(cursor + 5.0, boundary - half_overlap)
        chunks.append(AudioChunk(len(chunks), round(cursor, 3), round(window_end, 3)))
        if next_cursor <= cursor + 1e-6:
            next_cursor = min(duration, cursor + target_seconds)
        cursor = next_cursor

    if chunks:
        chunks[-1] = AudioChunk(chunks[-1].index, chunks[-1].start, round(duration, 3))
    return chunks


def write_audio_chunk(wav_path: Path, chunk: AudioChunk, destination: Path) -> Path:
    """Read only one bounded chunk from the normalized WAV and write a temp WAV."""

    audio_file = _require_soundfile()
    with audio_file.SoundFile(str(wav_path), "r") as source:
        sample_rate = int(source.samplerate)
        source.seek(max(0, int(round(chunk.start * sample_rate))))
        frames = max(1, int(round(chunk.duration * sample_rate)))
        audio = source.read(frames, dtype="float32", always_2d=False)
    if len(audio) == 0:
        raise RuntimeError(f"音訊 chunk 為空：{chunk.index}")
    destination = Path(destination)
    audio_file.write(str(destination), audio, sample_rate, subtype="PCM_16")
    del audio
    return destination


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = int(max(0.0, float(seconds)) * 1000.0 + 1e-7)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@lru_cache(maxsize=1)
def _opencc_converter() -> Any:
    try:
        from opencc import OpenCC
    except ImportError as exc:  # pragma: no cover - dependency smoke covers this
        raise RuntimeError(
            "缺少 opencc-python-reimplemented，請在 taiwan-subtitle/.venv 重新安裝 requirements.txt"
        ) from exc
    return OpenCC("s2twp")


def to_traditional(text: str) -> str:
    return _opencc_converter().convert(text)


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
    )


_UNIT_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._'’+\-/][A-Za-z0-9]+)*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\w\s]",
    re.UNICODE,
)


def _text_units(text: str) -> list[str]:
    return _UNIT_PATTERN.findall(" ".join(str(text).split()))


def _join_units(units: Iterable[str]) -> str:
    output = ""
    opening = "([［「『【《〈（{"
    closing = "，。！？；：、」』】》〉）]}>,.!?;:"
    for raw_unit in units:
        unit = str(raw_unit)
        if not unit:
            continue
        if not output:
            output = unit
            continue
        previous = output[-1]
        if unit[0] in closing or previous in opening:
            output += unit
        elif is_cjk_char(previous) and is_cjk_char(unit[0]):
            output += unit
        elif previous.isascii() and unit[0].isascii() and (
            previous.isalnum() and unit[0].isalnum()
        ):
            output += " " + unit
        elif is_cjk_char(previous) != is_cjk_char(unit[0]):
            output += " " + unit
        else:
            output += unit
    return output.strip()


def _smart_join_strings(left: str, right: str) -> str:
    return _join_units(_text_units(left) + _text_units(right))


def build_context(hotwords: Sequence[str]) -> str:
    cleaned = [str(word).strip() for word in hotwords if str(word).strip()]
    return f"Vocabulary: {', '.join(cleaned)}" if cleaned else ""


def _is_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(
        unicodedata.category(char).startswith("P") for char in stripped
    )


def _infer_missing_punctuation(
    tokens: Sequence[TokenTimestamp], raw_text: str
) -> list[TokenTimestamp]:
    """Restore ASR punctuation that a character-level aligner may omit."""

    if not tokens or not raw_text.strip():
        return list(tokens)

    raw_units = _text_units(raw_text)
    raw_events: list[tuple[int, str]] = []
    raw_nonpunct_count = 0
    for unit in raw_units:
        if _is_punctuation(unit):
            raw_events.append((raw_nonpunct_count, unit))
        else:
            raw_nonpunct_count += 1
    if not raw_events:
        return list(tokens)

    existing_events: list[tuple[int, str]] = []
    existing_nonpunct_count = 0
    token_unit_counts: list[int] = []
    for token in tokens:
        count = 0
        for unit in _text_units(token.text):
            if _is_punctuation(unit):
                existing_events.append((existing_nonpunct_count, unit))
            else:
                existing_nonpunct_count += 1
                count += 1
        token_unit_counts.append(count)

    used_existing: set[int] = set()
    missing: list[tuple[int, str]] = []
    for before_count, punctuation in raw_events:
        match = None
        for index, (existing_before, existing_text) in enumerate(existing_events):
            if index in used_existing:
                continue
            if existing_text == punctuation and abs(existing_before - before_count) <= 1:
                match = index
                break
        if match is None:
            missing.append((before_count, punctuation))
        else:
            used_existing.add(match)
    if not missing:
        return list(tokens)

    inferred: list[TokenTimestamp] = []
    for before_count, punctuation in missing:
        if before_count <= 0:
            anchor = tokens[0]
            end = max(0.001, float(anchor.start))
            start = max(0.0, end - 0.01)
            if end <= start:
                end = start + 0.001
        else:
            cumulative = 0
            anchor_index = len(tokens) - 1
            for index, count in enumerate(token_unit_counts):
                if before_count <= cumulative + count:
                    anchor_index = index
                    break
                cumulative += count
            anchor = tokens[anchor_index]
            start = max(0.0, float(anchor.end))
            next_start = (
                float(tokens[anchor_index + 1].start)
                if anchor_index + 1 < len(tokens)
                else start + 0.001
            )
            end = min(start + 0.02, next_start) if next_start > start else start + 0.001
            if end <= start:
                end = start + 0.001
        inferred.append(
            TokenTimestamp(
                text=punctuation,
                start=round(start, 3),
                end=round(end, 3),
                source_chunk=anchor.source_chunk,
                fallback="punctuation_inferred",
            )
        )

    return sorted([*tokens, *inferred], key=lambda token: (token.start, token.end))


def _is_sentence_boundary(text: str) -> bool:
    return bool(text.strip()) and text.strip()[-1] in "。！？!?；;"


def _display_unit_count(text: str) -> int:
    units = _text_units(text)
    return sum(1 for unit in units if not unit.isspace())


def _render_token_texts(tokens: Sequence[TokenTimestamp]) -> str:
    result = ""
    opening = "([［「『【《〈（{"
    closing = "，。！？；：、」』】》〉）]}>,.!?;:"
    for token in tokens:
        text = " ".join(str(token.text).split())
        if not text:
            continue
        if not result:
            result = text
            continue
        previous = result[-1]
        if text[0] in closing or previous in opening:
            result += text
        elif previous.isascii() and text[0].isascii() and previous.isalnum() and text[0].isalnum():
            result += " " + text
        elif is_cjk_char(previous) and is_cjk_char(text[0]):
            result += text
        elif is_cjk_char(previous) != is_cjk_char(text[0]):
            result += " " + text
        else:
            result += text
    return result.strip()


def _make_cue(tokens: Sequence[TokenTimestamp]) -> SubtitleCue:
    text = _render_token_texts(tokens)
    start = min(token.start for token in tokens)
    end = max(token.end for token in tokens)
    return SubtitleCue(start=max(0.0, start), end=max(start, end), text=text)


def _merge_cues(left: SubtitleCue, right: SubtitleCue) -> SubtitleCue:
    return SubtitleCue(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        text=_smart_join_strings(left.text, right.text),
    )


def build_subtitle_cues(
    tokens: Sequence[TokenTimestamp], duration: float
) -> list[SubtitleCue]:
    """Group aligned tokens into readable, non-overlapping subtitle cues."""

    ordered = sorted(
        (token for token in tokens if str(token.text).strip()),
        key=lambda token: (token.start, token.end),
    )
    if not ordered:
        return []

    raw_cues: list[SubtitleCue] = []
    current: list[TokenTimestamp] = []
    current_count = 0
    for token in ordered:
        token_count = max(1, _display_unit_count(token.text))
        too_long = current and (
            current_count + token_count > 20
            or token.start - current[0].start >= 6.0
        )
        if too_long:
            raw_cues.append(_make_cue(current))
            current = []
            current_count = 0
        current.append(token)
        current_count += token_count
        if (
            (_is_sentence_boundary(token.text) or _is_punctuation(token.text))
            and current_count >= 3
        ):
            raw_cues.append(_make_cue(current))
            current = []
            current_count = 0
    if current:
        raw_cues.append(_make_cue(current))

    # Merge tiny cues before enforcing the minimum display duration.
    cues = list(raw_cues)
    index = 0
    while index < len(cues):
        if _display_unit_count(cues[index].text) <= 2 and len(cues) > 1:
            if index + 1 < len(cues):
                cues[index : index + 2] = [_merge_cues(cues[index], cues[index + 1])]
                continue
            cues[index - 1 : index + 1] = [_merge_cues(cues[index - 1], cues[index])]
            index = max(0, index - 1)
            continue
        index += 1

    normalized: list[SubtitleCue] = []
    for cue in cues:
        start = max(0.0, min(float(duration), cue.start))
        end = max(start, min(float(duration), cue.end))
        if end - start < 0.5 and normalized:
            normalized[-1] = _merge_cues(normalized[-1], SubtitleCue(start, end, cue.text))
            continue
        if end - start < 0.5:
            end = min(float(duration), start + 0.5)
        normalized.append(SubtitleCue(start, end, cue.text))

    # Clip a very small accidental overlap caused by rounded model timestamps.
    clipped: list[SubtitleCue] = []
    for cue in normalized:
        if clipped and cue.start < clipped[-1].end:
            adjusted_start = clipped[-1].end
            if cue.end <= adjusted_start:
                clipped[-1] = _merge_cues(clipped[-1], cue)
                continue
            cue = SubtitleCue(adjusted_start, cue.end, cue.text)
        clipped.append(cue)
    return [
        SubtitleCue(
            start=max(0.0, min(float(duration), cue.start)),
            end=max(0.0, min(float(duration), cue.end)),
            text=cue.text.strip(),
        )
        for cue in clipped
        if cue.text.strip() and cue.end > cue.start
    ]


def validate_cues(cues: Sequence[SubtitleCue], duration: float) -> None:
    previous_end = 0.0
    for index, cue in enumerate(cues, start=1):
        if cue.start < -1e-6 or cue.end > duration + 1e-3:
            raise ValueError(f"SRT cue {index} 超出音訊範圍")
        if cue.end <= cue.start:
            raise ValueError(f"SRT cue {index} 的 end 不大於 start")
        if index > 1 and cue.start < previous_end - 1e-3:
            raise ValueError(f"SRT cue {index} 與前一條字幕重疊")
        previous_end = cue.end


def _item_value(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return default
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _split_aligned_text(text: str) -> list[str]:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    if not any(is_cjk_char(char) for char in cleaned):
        return [cleaned]

    parts: list[str] = []
    latin_buffer: list[str] = []

    def flush_latin() -> None:
        if latin_buffer:
            value = "".join(latin_buffer).strip()
            if value:
                parts.append(value)
            latin_buffer.clear()

    for char in cleaned:
        if is_cjk_char(char):
            flush_latin()
            parts.append(char)
        elif char.isspace():
            latin_buffer.append(char)
        elif unicodedata.category(char).startswith("P"):
            flush_latin()
            parts.append(char)
        else:
            latin_buffer.append(char)
    flush_latin()
    return parts


def normalize_alignment(
    items: Sequence[Any],
    offset: float = 0.0,
    raw_text: str = "",
    source_chunk: int | None = None,
) -> list[TokenTimestamp]:
    """Normalize MLX-Audio items and add the absolute chunk offset."""

    normalized: list[TokenTimestamp] = []
    previous_end = float(offset)
    for item in items:
        text = str(_item_value(item, "text", "word", default="")).strip()
        if not text:
            continue
        start_raw = _item_value(item, "start_time", "start", default=None)
        end_raw = _item_value(item, "end_time", "end", default=None)
        try:
            start = float(start_raw) + float(offset)
            end = float(end_raw) + float(offset)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        start = max(start, previous_end)
        if end <= start:
            # Qwen3 Forced Aligner can emit a valid character with identical
            # start/end values.  Keep it instead of silently dropping text.
            end = start + 0.001

        parts = _split_aligned_text(text)
        if not parts:
            continue
        span = end - start
        part_duration = span / len(parts)
        for part_index, part in enumerate(parts):
            part_start = start + part_index * part_duration
            part_end = end if part_index == len(parts) - 1 else start + (part_index + 1) * part_duration
            normalized.append(
                TokenTimestamp(
                    text=part,
                    start=round(part_start, 3),
                    end=round(part_end, 3),
                    source_chunk=source_chunk,
                    fallback="character_split_fallback" if len(parts) > 1 and not is_cjk_char(text[0]) else None,
                )
            )
        previous_end = end
    return _infer_missing_punctuation(normalized, raw_text)


def _mlx_memory() -> tuple[int | None, int | None]:
    try:
        import mlx.core as mx
    except ImportError:
        return None, None
    active = getattr(mx, "get_active_memory", None)
    peak = getattr(mx, "get_peak_memory", None)
    try:
        active_value = int(active()) if callable(active) else None
    except Exception:
        active_value = None
    try:
        peak_value = int(peak()) if callable(peak) else None
    except Exception:
        peak_value = None
    return active_value, peak_value


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None)
    if callable(clear):
        clear()
        return

    metal = getattr(mx, "metal", None)
    clear = getattr(metal, "clear_cache", None) if metal is not None else None
    if callable(clear):
        clear()


def _tea_model_quantization_predicate(_self: Any, _path: Any, _module: Any) -> bool:
    """Quantize every TEA module, including its already-quantized audio tower."""

    return True


def configure_tea_mlx_loader(qwen3_asr_module: Any | None = None) -> None:
    """Apply the compatibility hook documented by the TEA MLX model card."""

    if qwen3_asr_module is None:
        from mlx_audio.stt.models.qwen3_asr import qwen3_asr as qwen3_asr_module

    qwen3_asr_module.Qwen3ASRModel.model_quant_predicate = (
        _tea_model_quantization_predicate
    )


_PRIVATE_USE_AREA_RE = re.compile(r"[\uE000-\uF8FF]+")


def clean_asr_text(text: str) -> str:
    """Remove tokenizer artifacts while preserving normal transcript spacing."""

    cleaned = _PRIVATE_USE_AREA_RE.sub("", str(text or ""))
    cleaned = cleaned.replace("\x00", "")
    return " ".join(cleaned.replace("\n", " ").split())


def load_asr_model(model_id: str) -> tuple[Any, ModelInfo]:
    configure_huggingface_cache()
    from mlx_audio.stt import load

    started = time.perf_counter()
    compatibility: list[str] = []
    if model_id == TEA_ASR_MLX_MODEL:
        configure_tea_mlx_loader()
        compatibility.extend(
            [TEA_ASR_MLX_COMPATIBILITY_NOTE, TEA_ASR_MLX_TEXT_CLEANUP_NOTE]
        )
    model = load(model_id)
    active, peak = _mlx_memory()
    return model, ModelInfo(
        name=model_id,
        load_seconds=time.perf_counter() - started,
        active_memory_bytes=active,
        peak_memory_bytes=peak,
        compatibility=compatibility,
    )


def release_model(model: Any) -> None:
    if model is None:
        return
    close = getattr(model, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    del model
    gc.collect()
    clear_mlx_cache()


def _supported_parameters(callable_object: Any) -> tuple[set[str], bool]:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return set(), True
    return set(parameters), any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _asr_kwargs(
    model: Any,
    language: str,
    context: str,
    hotwords: Sequence[str],
    max_tokens: int,
    verbose: bool,
) -> dict[str, Any]:
    supported, accepts_kwargs = _supported_parameters(model.generate)
    kwargs: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if name in supported or accepts_kwargs:
            kwargs[name] = value

    add("language", language)
    add("max_tokens", max_tokens)
    add("temperature", 0.0)
    add("verbose", verbose)
    if "hotwords" in supported or accepts_kwargs:
        kwargs["hotwords"] = list(hotwords)
    if context:
        if "context" in supported:
            kwargs["context"] = context
        elif "system_prompt" in supported or accepts_kwargs:
            kwargs["system_prompt"] = context
    return kwargs


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        value = result.get("text", result.get("transcription", ""))
    else:
        value = getattr(result, "text", getattr(result, "transcription", ""))
    return clean_asr_text(str(value or ""))


def _result_language(result: Any, default: str = "Chinese") -> str:
    value = result.get("language") if isinstance(result, dict) else getattr(result, "language", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return str(value or default)


def _result_segments(result: Any, chunk: AudioChunk, language: str) -> list[dict[str, Any]]:
    raw_segments = result.get("segments") if isinstance(result, dict) else getattr(result, "segments", None)
    if not isinstance(raw_segments, list) or not raw_segments:
        return [
            {
                "text": _result_text(result),
                "start": chunk.start,
                "end": chunk.end,
                "language": language,
            }
        ]
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        text = clean_asr_text(str(_item_value(raw, "text", default="")))
        if not text:
            continue
        try:
            start = float(_item_value(raw, "start", "start_time", default=0.0)) + chunk.start
            end = float(_item_value(raw, "end", "end_time", default=chunk.duration)) + chunk.start
        except (TypeError, ValueError):
            start, end = chunk.start, chunk.end
        segments.append({"text": text, "start": start, "end": end, "language": language})
    return segments or [
        {"text": _result_text(result), "start": chunk.start, "end": chunk.end, "language": language}
    ]


def transcribe_audio_chunks(
    model: Any,
    chunks: Sequence[PreparedAudioChunk],
    context: str,
    hotwords: Sequence[str] = HOTWORDS,
    language: str = "Chinese",
    verbose: bool = False,
) -> list[ASRChunk]:
    results: list[ASRChunk] = []
    for prepared in chunks:
        chunk = prepared.chunk
        max_tokens = max(256, min(4096, int(math.ceil(chunk.duration * 10.0)) + 128))
        kwargs = _asr_kwargs(model, language, context, hotwords, max_tokens, verbose=False)
        if verbose:
            print(
                f"[ASR] chunk {chunk.index + 1}/{len(chunks)} "
                f"{format_srt_timestamp(chunk.start)}",
                flush=True,
            )
        result = model.generate(str(prepared.path), **kwargs)
        text = _result_text(result)
        detected_language = _result_language(result, language)
        if text:
            results.append(
                ASRChunk(
                    index=chunk.index,
                    start=chunk.start,
                    end=chunk.end,
                    text=text,
                    language=detected_language,
                    audio_path=prepared.path,
                    segments=_result_segments(result, chunk, detected_language),
                )
            )
        clear_mlx_cache()
    if not results:
        raise RuntimeError("ASR 對所有音訊 chunk 都沒有回傳文字")
    return results


def align_chunk(aligner: Any, chunk: ASRChunk, audio_path: Path) -> AlignmentResult:
    if not chunk.text.strip():
        return AlignmentResult(tokens=[], fallback="segment")
    result = aligner.generate(
        str(audio_path),
        text=chunk.text,
        language=chunk.language or "Chinese",
    )
    items = getattr(result, "items", result)
    if isinstance(items, dict):
        items = items.get("items", [])
    if not isinstance(items, (list, tuple)):
        items = list(items) if items is not None else []
    tokens = normalize_alignment(
        items,
        offset=chunk.start,
        raw_text=chunk.text,
        source_chunk=chunk.index,
    )
    if not tokens:
        return AlignmentResult(tokens=[], fallback="segment")
    return AlignmentResult(tokens=tokens)


def _segment_fallback_token(chunk: ASRChunk) -> TokenTimestamp:
    return TokenTimestamp(
        text=chunk.text,
        start=chunk.start,
        end=chunk.end,
        source_chunk=chunk.index,
        fallback="segment",
    )


def align_all_chunks(
    aligner_model_id: str,
    asr_chunks: Sequence[ASRChunk],
    wav_path: Path,
    warnings: list[str] | None = None,
    prepared_chunks: Sequence[PreparedAudioChunk] | None = None,
) -> tuple[list[TokenTimestamp], list[dict[str, Any]], list[str]]:
    del wav_path
    local_warnings = warnings if warnings is not None else []
    prepared_by_index = {
        prepared.chunk.index: prepared.path for prepared in (prepared_chunks or [])
    }
    tokens: list[TokenTimestamp] = []
    segment_records: list[dict[str, Any]] = []
    aligner = None
    try:
        aligner, _ = load_asr_model(aligner_model_id)
    except Exception as exc:
        message = f"ForcedAligner 載入失敗，全部改用 segment timestamp：{type(exc).__name__}: {exc}"
        local_warnings.append(message)
        for chunk in asr_chunks:
            tokens.append(_segment_fallback_token(chunk))
            segment_records.append(
                {
                    "text": chunk.text,
                    "start": chunk.start,
                    "end": chunk.end,
                    "source_chunk": chunk.index,
                    "fallback": "segment",
                }
            )
        return tokens, segment_records, local_warnings

    try:
        for chunk in asr_chunks:
            try:
                path = prepared_by_index.get(chunk.index, chunk.audio_path)
                result = align_chunk(aligner, chunk, path)
                if result.tokens:
                    tokens.extend(result.tokens)
                    segment_records.append(
                        {
                            "text": chunk.text,
                            "start": chunk.start,
                            "end": chunk.end,
                            "source_chunk": chunk.index,
                            "fallback": result.fallback,
                        }
                    )
                else:
                    raise RuntimeError("aligner 回傳空 token")
            except Exception as exc:
                message = (
                    f"chunk {chunk.index} 對齊失敗，改用 segment timestamp："
                    f"{type(exc).__name__}: {exc}"
                )
                local_warnings.append(message)
                tokens.append(_segment_fallback_token(chunk))
                segment_records.append(
                    {
                        "text": chunk.text,
                        "start": chunk.start,
                        "end": chunk.end,
                        "source_chunk": chunk.index,
                        "fallback": "segment",
                    }
                )
            clear_mlx_cache()
    finally:
        release_model(aligner)
    return _deduplicate_tokens(tokens), segment_records, local_warnings


def _deduplicate_tokens(tokens: Sequence[TokenTimestamp]) -> list[TokenTimestamp]:
    by_chunk: dict[int, list[TokenTimestamp]] = {}
    for token in tokens:
        by_chunk.setdefault(token.source_chunk if token.source_chunk is not None else -1, []).append(token)
    deduplicated: list[TokenTimestamp] = []
    previous: list[TokenTimestamp] = []
    for chunk_index in sorted(by_chunk):
        current = sorted(by_chunk[chunk_index], key=lambda token: (token.start, token.end))
        previous_texts = [token.text.strip() for token in previous[-32:]]
        current_texts = [token.text.strip() for token in current[:32]]
        overlap = 0
        for size in range(min(len(previous_texts), len(current_texts), 32), 1, -1):
            if previous_texts[-size:] == current_texts[:size]:
                overlap = size
                break
        current = current[overlap:]
        deduplicated.extend(current)
        previous.extend(current)
    ordered = sorted(deduplicated, key=lambda token: (token.start, token.end))
    result: list[TokenTimestamp] = []
    for token in ordered:
        if result and token.start < result[-1].end - 0.03 and token.text.strip() == result[-1].text.strip():
            continue
        result.append(token)
    return result


def _deduplicate_transcript(chunks: Sequence[ASRChunk]) -> str:
    units: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.index):
        current = _text_units(chunk.text)
        overlap = 0
        for size in range(min(len(units), len(current), 32), 1, -1):
            if units[-size:] == current[:size]:
                overlap = size
                break
        units.extend(current[overlap:])
    return _join_units(units)


def _expand_fallback_tokens(tokens: Sequence[TokenTimestamp]) -> list[TokenTimestamp]:
    expanded: list[TokenTimestamp] = []
    for token in tokens:
        if token.fallback != "segment" or _display_unit_count(token.text) <= 20:
            expanded.append(token)
            continue
        units = _text_units(token.text)
        groups: list[list[str]] = []
        group: list[str] = []
        for unit in units:
            group.append(unit)
            if len(group) >= 18 and (unit in "。！？!?；;，,、" or len(group) >= 20):
                groups.append(group)
                group = []
        if group:
            groups.append(group)
        span = max(0.5, token.end - token.start)
        step = span / max(1, len(groups))
        for index, group_units in enumerate(groups):
            expanded.append(
                TokenTimestamp(
                    text=_join_units(group_units),
                    start=token.start + step * index,
                    end=token.end if index == len(groups) - 1 else token.start + step * (index + 1),
                    source_chunk=token.source_chunk,
                    fallback="segment",
                )
            )
    return expanded


def _token_record(token: TokenTimestamp, converted_text: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "text": converted_text if converted_text is not None else token.text,
        "start": round(float(token.start), 3),
        "end": round(float(token.end), 3),
    }
    if token.source_chunk is not None:
        record["source_chunk"] = token.source_chunk
    if token.fallback:
        record["fallback"] = token.fallback
    return record


def _character_records(tokens: Sequence[TokenTimestamp]) -> list[dict[str, Any]]:
    characters: list[dict[str, Any]] = []
    for token in tokens:
        parts = _split_aligned_text(token.text)
        if not parts:
            continue
        span = max(0.0, token.end - token.start)
        step = span / max(1, len(parts))
        for index, part in enumerate(parts):
            part_start = token.start + step * index
            part_end = token.end if index == len(parts) - 1 else token.start + step * (index + 1)
            part_token = TokenTimestamp(
                part,
                part_start,
                part_end,
                token.source_chunk,
                token.fallback,
            )
            characters.append(_token_record(part_token, to_traditional(part)))
    return characters


def strip_srt_punctuation(text: str) -> str:
    """Remove standalone punctuation while preserving alphanumeric product names."""

    return _join_units(
        unit for unit in _text_units(text) if not _is_punctuation(unit)
    )


def _srt_text(cues: Sequence[SubtitleCue], no_punctuation: bool = False) -> str:
    blocks: list[str] = []
    for cue in cues:
        display_text = (
            strip_srt_punctuation(cue.text) if no_punctuation else cue.text.strip()
        )
        if not display_text:
            continue
        index = len(blocks) + 1
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}",
                    display_text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(
    input_path: Path,
    output_dir: Path,
    transcript: TranscriptBundle,
    no_punctuation: bool = False,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    srt_path = output_dir / f"{stem}.srt"
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"
    _atomic_write(srt_path, _srt_text(transcript.cues, no_punctuation=no_punctuation))
    _atomic_write(txt_path, transcript.full_transcript + ("\n" if transcript.full_transcript else ""))
    json_payload = {
        "full_transcript": transcript.full_transcript,
        "raw_transcript": transcript.raw_transcript,
        "segments": transcript.segments,
        "words": transcript.words,
        "characters": transcript.characters,
        "model": transcript.model,
        "aligner": transcript.aligner,
        "processing_time": round(transcript.processing_time, 3),
        "audio_duration": round(transcript.audio_duration, 3),
        "realtime_factor": round(transcript.realtime_factor, 5),
        "warnings": transcript.warnings,
    }
    _atomic_write(json_path, json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n")
    return srt_path, txt_path, json_path


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def run_transcription(
    input_path: Path,
    output_dir: Path | None = None,
    asr_model: str | None = None,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    hotwords: Sequence[str] = HOTWORDS,
    language: str = "Chinese",
    keep_temp: bool = False,
    no_punctuation: bool = False,
    verbose: bool = False,
    allow_download: bool = False,
) -> tuple[tuple[Path, Path, Path], TranscriptBundle]:
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("此工具只允許在 Apple Silicon arm64 上執行。")
    if chunk_seconds > MAX_ALIGNMENT_SECONDS - 10.0:
        raise ValueError("chunk_seconds 不可超過 260 秒，需保留 ForcedAligner 的長度安全邊界")

    # 模型必須事前備妥；預設不在轉錄途中連網下載。
    candidates = resolve_asr_candidates(
        asr_model=asr_model,
        aligner_model=aligner_model,
        allow_download=allow_download,
    )
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    started = time.perf_counter()
    media = probe_media(input_path)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else media.path.parent
    warnings: list[str] = []
    context = build_context(hotwords)
    temp_root = APP_DATA_DIR / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(temp_root)))
    model = None
    selected_info: ModelInfo | None = None
    asr_chunks: list[ASRChunk] | None = None

    try:
        normalized_wav = extract_audio(media.path, temp_dir / "audio-16k-mono.wav")
        energies = scan_energy(normalized_wav)
        chunks = build_chunks(
            media.duration,
            energies,
            target_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        if not chunks:
            raise RuntimeError("沒有可處理的音訊 chunk")
        prepared_chunks = [
            PreparedAudioChunk(
                chunk=chunk,
                path=write_audio_chunk(
                    normalized_wav,
                    chunk,
                    temp_dir / f"chunk-{chunk.index:05d}.wav",
                ),
            )
            for chunk in chunks
        ]

        candidate_errors: list[str] = []
        for candidate in candidates:
            candidate_model = None
            try:
                if verbose:
                    print(f"[ASR] 載入 {candidate}", flush=True)
                candidate_model, info = load_asr_model(candidate)
                candidate_results = transcribe_audio_chunks(
                    candidate_model,
                    prepared_chunks,
                    context=context,
                    hotwords=hotwords,
                    language=language,
                    verbose=verbose,
                )
                model = candidate_model
                candidate_model = None
                asr_chunks = candidate_results
                selected_info = info
                break
            except Exception as exc:
                message = f"ASR {candidate} 失敗：{type(exc).__name__}: {exc}"
                candidate_errors.append(message)
                warnings.append(message)
                if verbose:
                    print(f"[ASR] {message}", file=sys.stderr, flush=True)
            finally:
                release_model(candidate_model)

        if asr_chunks is None or selected_info is None:
            raise RuntimeError("所有 ASR candidate 都失敗：\n" + "\n".join(candidate_errors))
        release_model(model)
        model = None

        aligned_tokens, segment_records, alignment_warnings = align_all_chunks(
            aligner_model,
            asr_chunks,
            normalized_wav,
            warnings=[],
            prepared_chunks=prepared_chunks,
        )
        warnings.extend(alignment_warnings)

        raw_transcript = _deduplicate_transcript(asr_chunks)
        full_transcript = to_traditional(raw_transcript)
        converted_tokens = [
            TokenTimestamp(
                text=to_traditional(token.text),
                start=token.start,
                end=token.end,
                source_chunk=token.source_chunk,
                fallback=token.fallback,
            )
            for token in aligned_tokens
        ]
        cue_tokens = _expand_fallback_tokens(converted_tokens)
        cues = build_subtitle_cues(cue_tokens, media.duration)
        if not cues and full_transcript:
            cues = [SubtitleCue(0.0, min(media.duration, max(0.5, media.duration)), full_transcript)]
            warnings.append("沒有可用 token timestamp，SRT 使用整段 fallback cue")
        validate_cues(cues, media.duration)

        for record in segment_records:
            record["text"] = to_traditional(str(record.get("text", "")))
        words = [_token_record(token, to_traditional(token.text)) for token in aligned_tokens]
        characters = _character_records(aligned_tokens)
        processing_time = time.perf_counter() - started
        bundle = TranscriptBundle(
            full_transcript=full_transcript,
            raw_transcript=raw_transcript,
            segments=segment_records,
            words=words,
            characters=characters,
            model={
                "name": selected_info.name,
                "backend": selected_info.backend,
                "load_seconds": round(selected_info.load_seconds, 3),
                "active_memory": _format_bytes(selected_info.active_memory_bytes),
                "active_memory_bytes": selected_info.active_memory_bytes,
                "peak_memory": _format_bytes(selected_info.peak_memory_bytes),
                "peak_memory_bytes": selected_info.peak_memory_bytes,
                "compatibility": selected_info.compatibility,
            },
            aligner={"name": aligner_model, "backend": "mlx"},
            processing_time=processing_time,
            audio_duration=media.duration,
            realtime_factor=processing_time / media.duration,
            warnings=warnings,
            cues=cues,
        )
        paths = write_outputs(
            media.path,
            output_dir,
            bundle,
            no_punctuation=no_punctuation,
        )
        if verbose:
            print(
                f"[done] ASR={selected_info.name} active={_format_bytes(selected_info.active_memory_bytes)} "
                f"chunks={len(chunks)} RTF={bundle.realtime_factor:.3f}",
                flush=True,
            )
        return paths, bundle
    finally:
        release_model(model)
        if keep_temp:
            print(f"[temp] 保留暫存資料：{temp_dir}", flush=True)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 Apple Silicon Mac 上將影片/音訊轉成繁體中文字幕 SRT、TXT、JSON。"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="mp4/mov/mkv/mp3/wav/m4a 等本機檔案；搭配 --check-models/--download-models 時可省略",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="輸出資料夾；預設為輸入檔所在目錄")
    parser.add_argument("--asr-model", default=None, help="指定單一 ASR model；預設依序探測 TEA/Qwen")
    parser.add_argument("--aligner-model", default=DEFAULT_ALIGNER_MODEL)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--hotword", action="append", default=[], help="額外 hotword，可重複指定")
    parser.add_argument("--no-default-hotwords", action="store_true")
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="只從 SRT 顯示文字移除獨立標點；TXT、JSON 與時間戳不變",
    )
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="只檢查模型是否已下載到本機快取，不進行轉錄",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="事前下載所有必要模型，不進行轉錄",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="搭配 --download-models：即使已存在也重新下載",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允許轉錄途中補下載模型；預設關閉，模型缺少時直接報錯",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="檢查本機是否已安裝所有必要的 Python 套件與系統工具（ffmpeg），不進行轉錄",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="安裝所有缺少的 Python 套件（pip install），不進行轉錄；ffmpeg 等系統工具仍需自行安裝",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="指定 AI 模型的存放目錄並記住（寫入 config.json）；預設為 models/huggingface",
    )
    parser.add_argument(
        "--move-existing-models",
        action="store_true",
        help="搭配 --model-dir：把舊目錄底下已下載的模型搬到新目錄，而不是留在原地重新下載",
    )
    return parser.parse_args(argv)


def _print_model_progress(message: str, fraction: float | None = None) -> None:
    suffix = f" ({fraction * 100:.1f}%)" if fraction is not None else ""
    print(f"[MODEL] {message}{suffix}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.model_dir is not None:
        try:
            resolved = set_model_dir(
                args.model_dir,
                move_existing=args.move_existing_models,
                progress=_print_model_progress,
            )
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"模型存放目錄已設定為：{resolved}")

    if args.check_deps:
        statuses = check_dependencies()
        for status in statuses:
            mark = "OK" if status.installed else "MISSING"
            print(f"[{mark}] {status.label}: {status.detail}")
            if not status.installed and status.install_hint:
                print(f"       -> {status.install_hint}")
        return 0 if all(status.installed for status in statuses) else 1

    if args.install_deps:
        try:
            install_python_packages(progress=_print_model_progress)
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        for status in check_dependencies():
            mark = "OK" if status.installed else "MISSING"
            print(f"[{mark}] {status.label}: {status.detail}")
        return 0

    if args.download_models:
        try:
            statuses = download_models(
                progress=_print_model_progress, force=args.force_download
            )
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        for status in statuses:
            print(f"{status.label}: {status.detail}")
        return 0

    if args.check_models:
        print(f"模型存放目錄：{get_model_dir()}")
        statuses = check_models()
        for status in statuses:
            mark = "OK" if status.ready else "MISSING"
            print(f"[{mark}] {status.label} ({status.model_id}): {status.detail}")
        return 0 if all(status.ready for status in statuses) else 1

    if args.input is None:
        print("ERROR: 缺少輸入檔案；請提供影音檔或改用 --check-models/--download-models", file=sys.stderr)
        return 1

    configured_hotwords = [] if args.no_default_hotwords else list(HOTWORDS)
    configured_hotwords.extend(args.hotword)
    try:
        paths, bundle = run_transcription(
            input_path=args.input,
            output_dir=args.output_dir,
            asr_model=args.asr_model,
            aligner_model=args.aligner_model,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            hotwords=configured_hotwords,
            language=args.language,
            keep_temp=args.keep_temp,
            no_punctuation=args.no_punctuation,
            verbose=args.verbose,
            allow_download=args.allow_download,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"SRT: {paths[0]}")
    print(f"TXT: {paths[1]}")
    print(f"JSON: {paths[2]}")
    print(f"ASR: {bundle.model['name']}")
    print(f"Aligner: {bundle.aligner['name']}")
    print(f"RTF: {bundle.realtime_factor:.3f}")
    if bundle.warnings:
        print(f"Warnings: {len(bundle.warnings)}")
    return 0


# transcribe.py 最底部

def transcribe_file(
    file_path: str,
    output_dir: str | Path | None = None,
    allow_download: bool = False,
):
    """供 GUI 呼叫的轉錄入口函式。

    模型必須事前用 download_models() 備妥；allow_download 預設 False，
    因此轉錄途中不會突然開始下載。
    """

    paths, _bundle = run_transcription(
        input_path=Path(file_path),
        output_dir=Path(output_dir) if output_dir else None,
        verbose=True,
        allow_download=allow_download,
    )
    return paths[0]  # 回傳產出的 SRT 檔案路徑


if __name__ == "__main__":
    import multiprocessing

    # 若這支檔案本身也被凍結打包成獨立 CLI 執行檔，sys.executable 就是它
    # 自己而不是通用的 python 直譯器；freeze_support() 是官方建議的標準
    # 防護，避免 multiprocessing 內部（如 resource_tracker）想 spawn 輔助
    # 行程時失敗。以原始碼執行時這行完全沒有作用。
    multiprocessing.freeze_support()

    raise SystemExit(main())
