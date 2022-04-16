from pathlib import Path

import pytest
from cli import train


@pytest.mark.parametrize(
    "training_data, config",
    [
        (
            Path("data/test/classification/classification_data.json"),
            Path("data/test/pipelines/pipeline-basic.yml"),
        )
    ],
)
def test_trainer_with_naive(training_data: Path, config: Path):
    train(training_data, config)
