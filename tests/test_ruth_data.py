from pathlib import Path
from typing import Any, Dict, List, Text

import pytest
import yaml
from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def example_data(example_data_path: Path) -> List[Dict[Text, Any]]:
    with open(example_data_path, "r") as file_pointer:
        return yaml.safe_load(file_pointer)


def test_ruth_data(example_data: List[Dict[Text, Any]]):

    data = ""
    for example in example_data:
        data = RuthData.build(intent=example[INTENT], text=example[TEXT])
        break

    assert data.get(INTENT) == "greet"
    assert data.get(TEXT) == "Hello!"
