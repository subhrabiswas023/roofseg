from functools import cached_property
from pathlib import Path
from typing import override
from dataclasses import dataclass, field

import torch
import yaml

from roofseg.common.tracking import (
    ArtifactTracker,
    ArtifactRestorer,
    MetricTracker,
    MetricRestorer,
    load_from_jsonl,
    save_to_yaml,
)
from roofseg.common.transaction import BatchedTransaction, Transaction
from roofseg.common.typing import JsonDict, StateDict
from roofseg.segmentation.transaction import SaveArtifact, SaveMetrics


@dataclass(frozen=True)
class PathContext:
    root_dir: Path = Path(".")

    config_file_name: str = "config.yaml"
    metrics_file_name: str = "metrics.jsonl"
    artifacts_dir_name: str = "artifacts"
    model_file_name: str = "model.pth"
    optimizer_file_name = "optimizer.pth"

    @cached_property
    def config(self) -> Path:
        return self.root_dir / self.config_file_name

    @cached_property
    def metrics(self) -> Path:
        return self.root_dir / self.metrics_file_name

    @cached_property
    def artifacts(self) -> Path:
        return self.root_dir / self.artifacts_dir_name

    @cached_property
    def model(self) -> Path:
        return self.root_dir / self.model_file_name

    @cached_property
    def optimizer(self) -> Path:
        return self.root_dir / self.optimizer_file_name


@dataclass
class LocalRestorer(MetricRestorer[JsonDict], ArtifactRestorer[StateDict]):
    paths: PathContext

    def __post_init__(self):
        BatchedTransaction(
            [
                SaveMetrics(None, self.paths.metrics),
                SaveArtifact(None, self.paths.model),
                SaveArtifact(None, self.paths.optimizer),
            ]
        ).recover_if_failed()

    @override
    def restore_last_metrics(self) -> JsonDict | None:
        if not self.paths.metrics.exists():
            return None

        last_metrics = None

        for metrics in load_from_jsonl(self.paths.metrics):
            last_metrics = metrics

        if isinstance(last_metrics, dict):
            return last_metrics

        raise TypeError(f"Expected JSON object, got {type(last_metrics).__name__}")

    def _restore_artifact(self, artifact_path: Path) -> StateDict | None:
        if not artifact_path.exists():
            return None

        return torch.load(artifact_path, weights_only=True)

    @override
    def restore_model(self) -> StateDict | None:
        return self._restore_artifact(self.paths.model)

    @override
    def restore_optimizer(self) -> StateDict | None:
        return self._restore_artifact(self.paths.optimizer)


@dataclass
class LocalTracker(MetricTracker[JsonDict], ArtifactTracker[StateDict]):
    paths: PathContext
    _metrics_buffer: list[JsonDict] = field(default_factory=list)
    _transactions: list[Transaction] = field(default_factory=list)

    def __post_init__(self):
        self.paths.root_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: JsonDict) -> None:
        save_to_yaml(config, self.paths.config)

    @override
    def log_metrics(self, metrics: JsonDict) -> None:
        self._metrics_buffer.append(metrics)

    @override
    def save_model(self, model_state: StateDict) -> None:
        self._save_artifact(model_state, self.paths.model)

    @override
    def save_optimizer(self, optimizer_state: StateDict) -> None:
        self._save_artifact(optimizer_state, self.paths.optimizer)

    def _save_artifact(self, artifact: StateDict, artifact_path: Path) -> None:
        self._transactions.append(SaveArtifact(artifact, artifact_path))

    def prepare_transaction(self) -> Transaction:
        metrics_buffer = self._metrics_buffer
        transactions = self._transactions

        self._metrics_buffer = []
        self._transactions = []

        return BatchedTransaction(
            [
                SaveMetrics(metrics_buffer, self.paths.metrics),
                *transactions,
            ]
        )
