from dataclasses import dataclass


@dataclass(frozen=True)
class License:
    name: str


@dataclass(frozen=True)
class DatasetMetadata:
    title: str
    id: str
    licenses: list[License]


@dataclass(frozen=True)
class KernelMetadata:
    id: str
    title: str
    code_file: str
    language: str
    kernel_type: str
    is_private: bool
    enable_gpu: bool
    enable_internet: bool
    dataset_sources: list[str]

# Prepare metadata
DATASET_METADATA = DatasetMetadata(
    title="Rooftop Segmentation Package",
    id="subhrabiswas023/rooftop-segmentation-package",
    licenses=[License(name="CC0-1.0")],
)
KERNEL_METADATA = KernelMetadata(
    id="subhrabiswas023/rooftop-segmentation-from-cli",
    title="Rooftop Segmentation from CLI",
    code_file="run.py",
    language="python",
    kernel_type="script",
    is_private=True,
    enable_gpu=True,
    enable_internet=True,
    dataset_sources=[
        DATASET_METADATA.id,
        "dhruvpanchal1/inria-rooftop-segmentation-dataset-1024x1024-png",
    ],
)