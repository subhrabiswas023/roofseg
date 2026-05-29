import os
from typing import Iterable, override
from pathlib import Path
from dataclasses import dataclass, field

import torch

from roofseg.common.tracking import priority_save_to_jsonl, priority_staging_saver
from roofseg.common.typing import JsonDict, StateDict
from roofseg.common.transaction import Transaction


@dataclass
class SaveArtifact(Transaction):
    artifact: StateDict | None
    target_path: Path
    tmp_path: Path = field(init=False)
    
    def __post_init__(self) -> None:
        self.tmp_path = self.target_path.with_suffix(".tmp")

    @override
    def stage(self) -> None:
        if not self.artifact:
            return
        priority_staging_saver(torch.save)(self.artifact, self.target_path)

    @override
    def commit(self) -> None:
        os.replace(self.tmp_path, self.target_path)
    
    @property 
    @override
    def is_failed(self) -> bool:
        return self.tmp_path.exists()

    @override
    def recover_if_failed(self) -> None:
        if self.is_failed:
            os.remove(self.tmp_path)


@dataclass
class SaveMetrics(Transaction):
    metrics: Iterable[JsonDict] | None
    target_path: Path
    manifest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.manifest_path = self.target_path.with_suffix(".manifest.txt")

    @override
    def stage(self) -> None:
        if not self.metrics:
            return
        
        offset = self.target_path.stat().st_size if self.target_path.exists() else 0
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(str(offset))
            
            f.flush()
            os.fsync(f.fileno())

        priority_save_to_jsonl(self.metrics, self.target_path)

    @override
    def commit(self) -> None:
        if self.manifest_path.exists():
            return
        os.remove(
            self.manifest_path
        )  # FIXME: We assume that the "commit" is called after "stage". So, the manifest path always exists. A future implementation of state machine will make it better.

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
        except ValueError:
            pass  # manifest corrupted, the target file's change is intact
        finally:
            os.remove(self.manifest_path)