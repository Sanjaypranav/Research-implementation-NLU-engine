import json
from pathlib import Path

import pytest
from transformers import BertTokenizer

from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:
    with open(example_classifier_data, "r") as f:
        examples = json.load(f)

    training_data = TrainData()
    for value in examples:
        training_data.add_example(RuthData(value))

    return training_data


def test_transformers_finetuner(
        classifier_data: TrainData,
):
    tokenizer = HFTokenizer({})
    tokenizer.train(classifier_data)
