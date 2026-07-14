import os
import json
from pathlib import Path

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_NOTEBOOK_SLUG = os.environ["KAGGLE_NOTEBOOK_SLUG"]
KAGGLE_NOTEBOOK_TITLE = os.environ["KAGGLE_NOTEBOOK_TITLE"]

DEFAULT_KERNEL_METADATA = {
    "id": f"{KAGGLE_USERNAME}/{KAGGLE_NOTEBOOK_SLUG}",
    "title": f"{KAGGLE_NOTEBOOK_TITLE}",
    "code_file": "bundle.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [
        "dhruvpanchal1/inria-rooftop-segmentation-dataset-1024x1024-png"
    ],
}

def serialize_metadata() -> None:
    deployment_dir = Path("deploy")
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    kernel_metadata_path = Path("deploy") / "kernel-metadata.json"
    kernel_metadata_path.write_text(json.dumps(DEFAULT_KERNEL_METADATA))
    
if __name__ == '__main__':
    serialize_metadata()