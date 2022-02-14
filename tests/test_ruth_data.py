import json
from pathlib import Path

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_ruth_data(example_data_path: Path):
    with open(example_data_path, "r") as file_pointer:
        example_data = json.load(file_pointer)

    for example in example_data:
        data = RuthData.build(intent=example[INTENT], text=example[TEXT])
        break

    assert data.intent == "greet"
    assert data.text == "hi"
