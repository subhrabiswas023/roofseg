from pathlib import Path

import pytest
from torch.utils.data import DataLoader, Dataset

from roofseg import factories
from roofseg.pipeline import run_training_pipeline
from roofseg.segmentation.config import Config
from roofseg.segmentation.tracking import LocalRestorer, LocalTracker, PathContext

@pytest.mark.smoke
def test_pipeline(
    mock_config: Config,
    mock_dataset: Dataset,
    tmp_path: Path
):
    run_training_pipeline(
        config=mock_config,
        tracker=LocalTracker(PathContext(tmp_path)),
        restorer=LocalRestorer(PathContext(tmp_path)),
        environment_setter=factories.setup_environment,
        train_loader_factory=lambda _: DataLoader(mock_dataset),
        val_loader_factory=lambda _: DataLoader(mock_dataset),
        transformer_factory=factories.build_train_transformer,
        model_factory=factories.build_model,
        criterion_factory=factories.build_criterion,
        optimizer_factory=factories.build_optimizer,
    )
