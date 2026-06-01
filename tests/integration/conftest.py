import pytest

from roofseg.segmentation.config import Config, EnvironmentConfig

@pytest.fixture(scope="session")
def mock_config() -> Config:
    return Config(
       environment=EnvironmentConfig(
           device="cpu"
       )
    )