from pathlib import Path

from ruth.constants import TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.constants import ATTENTION_MASKS, INPUT_IDS
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_hf_tokenizer(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = registered_classes["HFTokenizer"].build({})
    tokenizer.train(training_data=training_data)
    message = RuthData(data={TEXT: "hi"})
    tokenizer.parse(message)
    assert message.data[INPUT_IDS] == [101, 7632, 102]
    assert message.data[ATTENTION_MASKS] == [1, 1, 1]
