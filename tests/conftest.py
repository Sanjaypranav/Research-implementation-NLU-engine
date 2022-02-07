import pytest
from pathlib import Path


@pytest.fixture
def example_data_path() -> Path:
    return Path("data/test/ruth_example_data/training_example.json")
