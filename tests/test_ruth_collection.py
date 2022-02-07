import json
from pathlib import Path

from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_collections_positive(example_data_path: Path):
    messages = []
    with open(example_data_path, 'r') as f:
        example_data = json.load(f)
    for data in example_data:
        messages.append(RuthData(data))


