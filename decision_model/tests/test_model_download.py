import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.download_models import (
    CHECKPOINT_NAME,
    RUN_NAME,
    extract_release_archive,
    install_delta_from_archive,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_archive(path: Path, payload: bytes = b"fixture checkpoint") -> tuple[str, str]:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(f"{RUN_NAME}/{CHECKPOINT_NAME}", payload)
        bundle.writestr(f"{RUN_NAME}/run.json", "{}")
    return digest(path), hashlib.sha256(payload).hexdigest()


def test_installs_verified_release_archive(tmp_path: Path):
    archive = tmp_path / "model.zip"
    archive_hash, checkpoint_hash = make_archive(archive)
    checkpoint = install_delta_from_archive(
        archive,
        tmp_path / "outputs",
        archive_hash,
        checkpoint_hash,
    )
    assert checkpoint.read_bytes() == b"fixture checkpoint"


def test_rejects_archive_hash_mismatch(tmp_path: Path):
    archive = tmp_path / "model.zip"
    _, checkpoint_hash = make_archive(archive)
    with pytest.raises(RuntimeError, match="模型压缩包 SHA256不匹配"):
        install_delta_from_archive(
            archive,
            tmp_path / "outputs",
            "0" * 64,
            checkpoint_hash,
        )


def test_rejects_zip_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")
    with pytest.raises(RuntimeError, match="不安全路径"):
        extract_release_archive(archive, tmp_path / "extract")
