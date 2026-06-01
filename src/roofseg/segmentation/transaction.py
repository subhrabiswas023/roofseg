import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, override

import torch

from roofseg.common.tracking import priority_staging_saver, save_to_jsonl
from roofseg.common.transaction import StagedTransaction, Transaction
from roofseg.common.typing import JsonDict, StateDict


def get_tmp_path(source_path: Path) -> Path:
    return source_path.with_suffix(".tmp")


def get_manifest_path(source_path: Path) -> Path:
    return source_path.with_suffix(".manifest.txt")


@dataclass(frozen=True)
class StagedSaveArtifact(StagedTransaction):
    target_path: Path

    @cached_property
    def tmp_path(self) -> Path:
        return get_tmp_path(self.target_path)

    @override
    def commit(self) -> None:
        os.replace(self.tmp_path, self.target_path)

    @property
    @override
    def is_failed(self) -> bool:
        return self.tmp_path.exists()

    @override
    def recover_if_failed(self) -> None:
        if not self.is_failed:
            return
        
        os.remove(self.tmp_path)


@dataclass(frozen=True)
class SaveArtifact(Transaction):
    artifact: StateDict
    target_path: Path

    @override
    def stage(self) -> StagedSaveArtifact:
        priority_staging_saver(torch.save)(
            self.artifact, get_tmp_path(self.target_path)
        )
        return StagedSaveArtifact(self.target_path)


@dataclass(frozen=True)
class StagedSaveMetrics(StagedTransaction):
    target_path: Path

    @cached_property
    def manifest_path(self) -> Path:
        return get_manifest_path(self.target_path)

    @override
    def commit(self) -> None:
        if not self.manifest_path.exists():
            return
        os.remove(self.manifest_path)

    @property
    @override
    def is_failed(self) -> bool:
        return self.manifest_path.exists()

    @override
    def recover_if_failed(self) -> None:
        if not self.is_failed:
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                safe_offset = int(f.read())
                with open(self.target_path, "a") as f:
                    f.truncate(safe_offset)

                    f.flush()
                    os.fsync(f.fileno())
        except ValueError:
            pass  # manifest corrupted, the target file's change is intact
        finally:
            os.remove(self.manifest_path)


@dataclass(frozen=True)
class SaveMetrics(Transaction):
    metrics: Iterable[JsonDict]
    target_path: Path

    @override
    def stage(self) -> StagedSaveMetrics:
        manifest_path = get_manifest_path(self.target_path)
        offset = self.target_path.stat().st_size if self.target_path.exists() else 0
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(str(offset))

            f.flush()
            os.fsync(f.fileno())

        priority_staging_saver(save_to_jsonl)(self.metrics, self.target_path)

        return StagedSaveMetrics(self.target_path)
