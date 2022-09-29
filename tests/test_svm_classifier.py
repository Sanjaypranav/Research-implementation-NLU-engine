from pathlib import Path

import pytest
from ruth.constants import TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:

    training_data = TrainData.build(example_classifier_data)

    return training_data


def test_svm_classifier(
    classifier_data: TrainData,
):
    ftr = registered_classes["TfidfVectorFeaturizer"].build({})
    ftr.train(classifier_data)

    classifier = registered_classes["SVMClassifier"].build({})
    classifier.train(training_data=classifier_data)
    message = RuthData({TEXT: "hello"})
    ftr.parse(message)
    classifier.parse(message)
    # assert message.get(INTENT)[INTENT_NAME_KEY] == "ham"
    #
    # message = RuthData(
    #     {
    #         TEXT: "WINNER!!"
    #     }
    # )
    # ftr.parse(message)
    # classifier.parse(message)
    # assert message.get(INTENT)[INTENT_NAME_KEY] == "spam"
