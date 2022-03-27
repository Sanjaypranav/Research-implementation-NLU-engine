from pathlib import Path

import pytest

from ruth.cli.cli import train
from ruth.shared.nlu.training_data.ruth_config import RuthConfig


@pytest.mark.parametrize(
    "training_data", "config",
    [
        (
                Path("data/test/classification/classification_data.json"),
                Path("data/test/pipelines/pipelines-basic.yml")
        )
    ]

)
def test_trainer_with_naive(training_data: Path, config: Path):
    config = RuthConfig(config)
    train(training_data, config)

