from pathlib import Path

from ruth.nlu.tokenizer.hf_tokenizer import HFTokenizer
from ruth.shared.constants import INPUT_IDS, ATTENTION_MASKS
from ruth.shared.nlu.training_data.collections import TrainData


def test_hf_tokenizer(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = HFTokenizer.build({})
    tokenizer.train(training_data=training_data)
    assert training_data.training_examples[0].data[INPUT_IDS] == [101, 7632, 102]
    assert training_data.training_examples[0].data[ATTENTION_MASKS] == [1, 1, 1]

