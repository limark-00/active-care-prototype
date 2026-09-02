#!/usr/bin/env python3
"""Install the released V3.1 delta checkpoint and its pinned MacBERT base."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "decision_model" / "outputs"
RUN_NAME = "partial-macbert-v31-20260829T132719Z"
CHECKPOINT_NAME = "best_delta.pt"
RELEASE_URL = (
    "https://github.com/limark-00/active-care-prototype/releases/download/"
    "model-v3.1/active-care-v3.1-delta.zip"
)
ARCHIVE_SHA256 = "f04357af3e131053974ddea4ce6dbc6daba2e25cf01304f8b772f59ef771723f"
CHECKPOINT_SHA256 = "1bfb03f22cb5eb4840db8c33583391cf8209ecd544cae128de9748fcb8a0645e"
BASE_MODEL = "hfl/chinese-macbert-base"
BASE_MODEL_REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
BASE_MODEL_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.bin",
    "*.safetensors",
]
CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"缺少{label}：{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256不匹配。\n期望：{expected}\n实际：{actual}\n文件：{path}"
        )


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "active-care-model-installer/1.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while chunk := response.read(CHUNK_BYTES):
            output.write(chunk)
            received += len(chunk)
            if total:
                print(
                    f"\r下载V3.1增量权重：{received / 1024 / 1024:.1f} / "
                    f"{total / 1024 / 1024:.1f} MB",
                    end="",
                    flush=True,
                )
        print()


def _safe_member_path(member_name: str) -> PurePosixPath:
    normalized = PurePosixPath(member_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise RuntimeError(f"模型压缩包包含不安全路径：{member_name}")
    if not normalized.parts or normalized.parts[0] != RUN_NAME:
        raise RuntimeError(f"模型压缩包包含未知顶层路径：{member_name}")
    return normalized


def extract_release_archive(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            _safe_member_path(member.filename)
            mode = member.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise RuntimeError(f"模型压缩包不允许符号链接：{member.filename}")
        bundle.extractall(destination)
    return destination / RUN_NAME


def install_delta_from_archive(
    archive: Path,
    outputs_dir: Path = OUTPUTS_DIR,
    archive_sha256: str = ARCHIVE_SHA256,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
) -> Path:
    verify_file(archive, archive_sha256, "模型压缩包")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    target = outputs_dir / RUN_NAME
    checkpoint = target / CHECKPOINT_NAME
    if target.exists():
        verify_file(checkpoint, checkpoint_sha256, "V3.1增量权重")
        return checkpoint
    with tempfile.TemporaryDirectory(prefix=".v31-install-", dir=outputs_dir) as temporary:
        extracted = extract_release_archive(archive, Path(temporary))
        extracted_checkpoint = extracted / CHECKPOINT_NAME
        verify_file(extracted_checkpoint, checkpoint_sha256, "V3.1增量权重")
        shutil.move(str(extracted), str(target))
    return checkpoint


def install_delta() -> Path:
    checkpoint = OUTPUTS_DIR / RUN_NAME / CHECKPOINT_NAME
    if checkpoint.is_file():
        verify_file(checkpoint, CHECKPOINT_SHA256, "V3.1增量权重")
        print(f"V3.1增量权重已存在并通过校验：{checkpoint}")
        return checkpoint
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="active-care-download-") as temporary:
        archive = Path(temporary) / "active-care-v3.1-delta.zip"
        print(f"从GitHub Release下载：{RELEASE_URL}")
        download_file(RELEASE_URL, archive)
        checkpoint = install_delta_from_archive(archive)
    print(f"V3.1增量权重安装完成：{checkpoint}")
    return checkpoint


def prepare_base_model(*, allow_download: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "缺少huggingface-hub。请先安装vision/requirements.in或vision/requirements.txt。"
        ) from error

    options = {
        "repo_id": BASE_MODEL,
        "revision": BASE_MODEL_REVISION,
        "allow_patterns": BASE_MODEL_PATTERNS,
        "local_files_only": not allow_download,
    }
    try:
        snapshot = Path(snapshot_download(**options))
    except Exception as error:
        action = "联网下载" if allow_download else "本机缓存校验"
        raise RuntimeError(
            f"基础MacBERT{action}失败：{BASE_MODEL}@{BASE_MODEL_REVISION}\n{error}"
        ) from error
    print(f"基础MacBERT已准备：{snapshot}")
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载并校验Active Care原型使用的V3.1文字决策模型。"
    )
    parser.add_argument(
        "--delta-only",
        action="store_true",
        help="只安装GitHub Release中的55MB增量权重，不准备基础MacBERT。",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅校验已有文件和本机缓存，不进行网络下载。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            checkpoint = OUTPUTS_DIR / RUN_NAME / CHECKPOINT_NAME
            verify_file(checkpoint, CHECKPOINT_SHA256, "V3.1增量权重")
            print(f"V3.1增量权重校验通过：{checkpoint}")
        else:
            install_delta()
        if not args.delta_only:
            prepare_base_model(allow_download=not args.verify_only)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"模型准备失败：{error}")
        return 1
    print("模型准备完成。现在可以启动本地伴护服务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
