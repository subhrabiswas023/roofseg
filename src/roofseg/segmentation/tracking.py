from functools import cached_property
import json
from pathlib import Path
from typing import override
from dataclasses import dataclass

import torch
import yaml

from roofseg.common.tracking import (
    ArtifactTracker,
    ArtifactRestorer,
    MetricTracker,
    MetricRestorer,
)
from roofseg.common.typing import JsonDict, StateDict


@dataclass(frozen=True)
class Tracker(
    MetricTracker[JsonDict],
    MetricRestorer[JsonDict],
    ArtifactTracker[StateDict],
    ArtifactRestorer[StateDict],
):
    target_root_dir: Path
    restoration_root_dir: Path
    config_file_name: str = "config.yaml"
    metrics_file_name: str = "metrics.jsonl"
    artifacts_dir_name: str = "artifacts"
    model_file_name: str = "model.pth"
    optimizer_file_name = "optimizer.pth"

    def __post_init__(
        self,
    ):
        self.target_root_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

    @cached_property
    def artifacts_path(self) -> Path:
        return self.target_root_dir / self.artifacts_dir_name

    @cached_property
    def artifacts_restoration_path(self) -> Path:
        return self.restoration_root_dir / self.artifacts_dir_name

    def save_config(self, config: JsonDict) -> None:
        config_path = self.target_root_dir / self.config_file_name

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

    @override
    def log_metrics(self, metrics: JsonDict) -> None:
        metrics_path = self.target_root_dir / self.metrics_file_name

        with open(metrics_path, "a", encoding="utf-8") as f:
            json.dump(metrics, f)
            f.write("\n")

    @override
    def restore_last_metrics(self) -> JsonDict | None:
        metrics_path = self.restoration_root_dir / self.metrics_file_name

        if not metrics_path.exists():
            return None

        with open(metrics_path, "r", encoding="utf-8") as f:
            last_dict = None
            for line in f:
                line = line.strip()
                if line:
                    last_dict = json.loads(line)

        if isinstance(last_dict, dict):
            return last_dict

        raise TypeError(f"Expected JSON object, got {type(last_dict).__name__}")

    def save_artifact(self, artifact: StateDict, filename: str) -> None:
        torch.save(artifact, self.artifacts_path / filename)

    @override
    def save_model(self, model_state: StateDict) -> None:
        self.save_artifact(model_state, self.model_file_name)

    @override
    def save_optimizer(self, optimizer_state: StateDict) -> None:
        self.save_artifact(optimizer_state, self.optimizer_file_name)

    def restore_artifact(self, filename: str) -> StateDict | None:
        artifact_file_path = self.artifacts_restoration_path / filename

        if not artifact_file_path.exists():
            return None

        return torch.load(artifact_file_path, weights_only=True)

    def restore_model(self) -> StateDict | None:
        return self.restore_artifact(self.model_file_name)

    def restore_optimizer(self) -> StateDict | None:
        return self.restore_artifact(self.optimizer_file_name)
