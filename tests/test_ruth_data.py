from pathlib import Path

import pytest
from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def example_data(example_data_path: Path) -> TrainData:
    return TrainData.build(example_data_path)


def test_ruth_data(example_data: TrainData):
    data = ""

    for example in example_data.training_examples:
        data = RuthData.build(intent=example.get(INTENT), text=example.get(TEXT))
        break

    assert data.get(INTENT) == "greet"
    assert data.get(TEXT) == "Hello!"
