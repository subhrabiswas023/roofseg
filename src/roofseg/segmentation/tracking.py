import json
from pathlib import Path
from typing import override

import yaml

from roofseg.common import tracking
from roofseg.common.typing import Dataclass
    
class Tracker(tracking.Tracker):
    def __init__(
        self,
        root_dir: Path,
        config_file_name: str = "config.yaml",
        metrics_file_name: str = "metrics.jsonl",
        artifacts_dir_name: str = "artifacts",
    ):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_path = root_dir / config_file_name
        self.metrics_path = root_dir / metrics_file_name
        self._artifacts_path = root_dir / artifacts_dir_name
        self._artifacts_path.mkdir(parents=True, exist_ok=True)
    
    @property    
    @override
    def artifacts_path(self) -> Path:
        return self._artifacts_path
    
    @override
    def save_config(self, config: Dataclass) -> None:
        with open(self.config_path, 'w') as f:
            yaml.dump(config.asdict(), f)
      
    @override      
    def log_metrics(self, metrics: Dataclass) -> None:
        with open(self.metrics_path, 'a') as f:
            json.dump(metrics.asdict(), f)
            f.write("\n")
    
    @override
    def close(self):
        pass
    
    