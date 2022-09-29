from pathlib import Path

from ruth.constants import INTENT, TEXT
from ruth.nlu.classifiers.constants import BATCH_SIZE, EPOCHS
from ruth.nlu.registry import registered_classes
from ruth.shared.constants import INTENT_NAME_KEY
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_hf_classifier(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = registered_classes["HFTokenizer"].build({})
    tokenizer.train(training_data=training_data)
    classifier = registered_classes["HFClassifier"].build({EPOCHS: 10, BATCH_SIZE: 8})
    classifier.train(training_data=training_data)
    message = RuthData(data={TEXT: "Hello!"})
    tokenizer.parse(message)
    classifier.parse(message)
    assert message.data[INTENT][INTENT_NAME_KEY] == "greet"
