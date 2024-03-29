from pathlib import Path

import pytest
from ruth.constants import INTENT, TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:
    training_data = TrainData.build(example_classifier_data)

    return training_data


def test_naive_bayes_classifier(
    classifier_data: TrainData,
):
    ftr = registered_classes["CountVectorFeaturizer"].build({})
    ftr.train(classifier_data)

    classifier = registered_classes["NaiveBayesClassifier"].build({})
    classifier.train(training_data=classifier_data)
    message = RuthData({TEXT: "hello"})
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)["name"] == "ham"
    assert message.get(INTENT)["accuracy"] == 1.0
