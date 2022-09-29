from pathlib import Path

import pytest
import yaml
from ruth.constants import INTENT, TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:
    with open(example_classifier_data, "r") as f:
        examples = yaml.safe_load(f)

    training_data = TrainData()
    for value in examples:
        training_data.add_example(RuthData(value))

    return training_data


def test_svm_classifier(
    classifier_data: TrainData,
):
    ftr = registered_classes["CountVectorFeaturizer"].build({})
    ftr.train(classifier_data)

    classifier = registered_classes["SVMClassifier"].build({})
    classifier.train(training_data=classifier_data)
    message = RuthData({TEXT: "hello"})
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)["name"] == "ham"

    message = RuthData(
        {
            TEXT: "WINNER!! As a valued network customer you have been"
            " selected to received £900 prize reward! "
            "To claim call 09061701461. Claim code KL341."
            " Valid 12 hours only."
        }
    )
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)["name"] == "spam"
