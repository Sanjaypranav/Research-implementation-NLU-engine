from pathlib import Path

from ruth.nlu import HFTokenizer
from ruth.shared import TrainData


def test_hf_tokenizer(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = HFTokenizer.build({})
    tokenizer.train(training_data=training_data)
    print("dsdf")
