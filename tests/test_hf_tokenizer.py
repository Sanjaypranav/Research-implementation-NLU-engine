from pathlib import Path

from nlu import HFTokenizer
from shared import TrainData


def test_hf_tokenizer(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = HFTokenizer.build({})
    tokenizer.train(training_data=training_data)
    print("dsdf")
