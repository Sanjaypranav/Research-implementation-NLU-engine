from pathlib import Path
from typing import Any, Dict, Text

import pytest
from ruth.constants import PATH, TEXT, TOKENS
from ruth.nlu.tokenizer.tokenizer import Token

FEATURE = "feature"
TOKEN = "token"
INPUT_ID = "input_id"
ATTENTION_MASK = "attention_mask"


@pytest.fixture
def example_data_path() -> Path:
    return Path("data/test/ruth_example_data/nlu.yml")


@pytest.fixture
def count_featurizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "What is your name?",
        FEATURE: [[0, 1, 0, 1, 1]],
        PATH: Path("data/test/ruth_example_data/training_example.yml"),
    }


@pytest.fixture
def whitespace_example() -> Dict[Text, Any]:
    return {
        TEXT: "I am a developer",
        TOKENS: [
            Token("i", start=0, end=1),
            Token("am", start=2, end=4),
            Token("a", start=5, end=6),
            Token("developer", start=7, end=16),
        ],
        PATH: Path("data/test/ruth_example_data/training_example.yml"),
    }


@pytest.fixture
def example_classifier_data() -> Path:
    return Path("data/test/classification/classification_data.yml")


@pytest.fixture
def bert_tokenizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "He lived Happily",
        TOKEN: [[0, 0, 0]],
        PATH: Path("data/test/ruth_example_data/training_example.yml"),
    }


@pytest.fixture
def tfidf_featurizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "What is your name?",
        FEATURE: [
            [0.0, 0.5773502691896257, 0.0, 0.5773502691896257, 0.5773502691896257]
        ],
        PATH: Path("data/test/ruth_example_data/training_example.yml"),
    }


@pytest.fixture
def hf_tokenizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "hello",
        INPUT_ID: [101, 7592, 102],
        ATTENTION_MASK: [1, 1, 1],
        PATH: Path("data/test/ruth_example_data/training_example.yml"),
    }
