import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

from .git import assert_clean_repo, GitStamp
from .build import build_wheel, get_wheel_path
from .metadata import DatasetMetadata, KernelMetadata, DATASET_METADATA, KERNEL_METADATA
from .executor import run_command


def write_to_json(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=4))


# pipeline


def validate_repo() -> GitStamp:
    assert_clean_repo()
    return GitStamp.create()


@dataclass(frozen=True)
class DeployPaths:
    dist: Path
    dataset: Path
    kernel_source: Path
    kernel_stage: Path


def prepare_stage() -> DeployPaths:
    # Cleaning the directories
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    deploy_dir = Path("deploy") / "kaggle"
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)

    # Making the directories
    dataset_dir = deploy_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path("scripts") / "kaggle"
    kernel_stage_dir = deploy_dir / "kernel"
    kernel_stage_dir.mkdir(parents=True, exist_ok=True)

    kernel_source_dir = scripts_dir / "kernel"

    return DeployPaths(
        dist=dist_dir,
        dataset=dataset_dir,
        kernel_source=kernel_source_dir,
        kernel_stage=kernel_stage_dir,
    )


def build_artifacts(
    paths: DeployPaths,
    dataset_metadata: DatasetMetadata,
    kernel_metadata: KernelMetadata,
) -> None:

    build_wheel()

    wheel_path = get_wheel_path(paths.dist)
    run_path = paths.kernel_source / kernel_metadata.code_file
    if not run_path.exists():
        raise FileNotFoundError(run_path)

    shutil.copy2(wheel_path, paths.dataset)
    shutil.copy2(run_path, paths.kernel_stage)

    # stage metadata
    write_to_json(
        asdict(dataset_metadata), paths.dataset / "dataset-metadata.json"
    )  # dataset
    write_to_json(
        asdict(kernel_metadata), paths.kernel_stage / "kernel-metadata.json"
    )  # kernel


def deploy(paths: DeployPaths, stamp: GitStamp) -> None:
    # Deploy the package as dataset
    print(run_command(
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(paths.dataset),
        "-m",
        f"Deploy {stamp.description}",
    ))
    # FIX ME: The line can be finished but the actual dataset may take time to update. Synchronization needed!

    # Deploy the main script
    print(run_command("kaggle", "kernels", "push", "-p", str(paths.kernel_stage)))


if __name__ == "__main__":
    stamp = validate_repo()
    paths = prepare_stage()
    build_artifacts(paths, DATASET_METADATA, KERNEL_METADATA)
    deploy(paths, stamp)
