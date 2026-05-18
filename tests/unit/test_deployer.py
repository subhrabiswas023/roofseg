from pathlib import Path
from unittest.mock import patch

from scripts.kaggle import deployer, metadata


# Should be put in integration testing
class TestPrepareStage:
    def test_paths_exist(self, tmp_path: Path):
        paths = deployer.prepare_stage(tmp_path)

        assert not paths.dist.exists()
        assert paths.dataset.exists()
        assert paths.kernel_stage.exists()

        assert paths.kernel_source == tmp_path / "scripts" / "kaggle" / "kernel"


class TestBuildArtifacts:
    def test_paths_exist(self, tmp_path: Path):
        paths = deployer.prepare_stage(tmp_path)

        wheel_path = paths.dist / "dummy_pkg-0.0.1-py3-none-any.whl"
        wheel_path.parent.mkdir(parents=True)
        wheel_path.write_text("fake wheel content")

        run_path = paths.kernel_source / metadata.KERNEL_METADATA.code_file
        run_path.parent.mkdir(parents=True)
        run_path.write_text("print('Hello')")

        with patch.object(deployer, deployer.build_wheel.__name__) as _:
            deployer.build_artifacts(
                paths, metadata.DATASET_METADATA, metadata.KERNEL_METADATA
            )

        assert (paths.dataset / wheel_path.name).exists()
        assert (paths.kernel_stage / run_path).exists()
        assert (paths.dataset / "dataset-metadata.json").exists()
        assert (paths.kernel_stage / "kernel-metadata.json").exists()
