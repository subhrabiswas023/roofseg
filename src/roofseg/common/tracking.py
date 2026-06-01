import json
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Iterable, Protocol

import yaml

from roofseg.common.typing import JsonDict


class MetricTracker[T](Protocol):
    def log_metrics(self, metrics: T) -> None: ...


class MetricRestorer[T](Protocol):
    def restore_last_metrics(self) -> T | None: ...


class ArtifactTracker[T](Protocol):
    def save_model(self, model_state: T) -> None: ...
    def save_optimizer(self, optimizer_state) -> None: ...


class ArtifactRestorer[T](Protocol):
    def restore_model(self) -> T | None: ...
    def restore_optimizer(self) -> T | None: ...


def priority_staging_saver[T](saver: Callable[[T, Path], None]):
    @wraps(saver)
    def wrapper(data: T, target: Path):
        saver(data, target)

        with open(target, "a") as f:
            f.flush()
            os.fsync(f.fileno())

    return wrapper

def save_to_yaml(data: JsonDict, target: Path):
        with open(target, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

def save_to_jsonl(lines: Iterable[JsonDict], target: Path) -> None:
    with open(target, "a", encoding="utf-8") as f:
        for line in lines:
            json.dump(line, f)
            f.write("\n")


def load_from_jsonl(source: Path) -> Iterable[JsonDict]:
    with open(source, "r") as f:
        for line in f:
            yield json.loads(line)
