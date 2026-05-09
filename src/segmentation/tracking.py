import yaml
import json
from pathlib import Path

from ..common import tracking
    
class Tracker(tracking.Tracker):
    def __init__(
        self,
        root_dir: Path,
        config_file_name: str = "config.yaml",
        metrics_file_name: str = "metrics.jsonl",
        artifacts_dir_name: str = "artifacts",
    ):
        self.root_dir = root_dir
        self.config_path = root_dir / config_file_name
        self.metrics_path = root_dir / metrics_file_name
        self._artifacts_path = root_dir / artifacts_dir_name
        
    @property        
    def artifacts_path(self) -> Path:
        return self._artifacts_path
        
    def save_config(self, config: dict) -> None:
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
            
    def log_metrics(self, metrics: dict[str, float]) -> None:
        with open(self.metrics_path, 'a') as f:
            json.dump(metrics, f)
            f.write("\n")
    
    def close(self):
        pass
    
    