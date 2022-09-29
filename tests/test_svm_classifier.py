from pathlib import Path

import pytest
from ruth.constants import INTENT, TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.constants import INTENT_NAME_KEY
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:

    training_data = TrainData.build(example_classifier_data)

    return training_data


def test_svm_classifier(
    classifier_data: TrainData,
):
    ftr = registered_classes["CountVectorFeaturizer"].build({})
    ftr.train(classifier_data)

    classifier = registered_classes["SVMClassifier"].build({})
    classifier.train(training_data=classifier_data)
    message = RuthData({TEXT: "hello bro, how are you"})
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)[INTENT_NAME_KEY] == "ham"

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
