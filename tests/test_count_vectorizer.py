from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT
from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import (
    CountVectorFeaturizer,
)
from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData

from tests.conftest import FEATURE


def test_count_vectorizer(count_featurizer_example: Dict[Text, Any]):
    training_data = TrainData.build(count_featurizer_example[PATH])
    print(training_data)

    featurizer = CountVectorFeaturizer({})
    featurizer.train(training_data)
    message = RuthData.build(text=count_featurizer_example[TEXT])
    featurizer.parse(message)

    assert (
        count_featurizer_example[FEATURE]
        == message.get_features().feature.toarray().tolist()
    )


def test_count_vectorizer_by_registry(count_featurizer_example: Dict[Text, Any]):
    training_data = TrainData.build(count_featurizer_example[PATH])

    featurizer = registered_classes["CountVectorFeaturizer"].build({})
    featurizer.train(training_data)
    message = RuthData.build(text=count_featurizer_example[TEXT])
    featurizer.parse(message)

    assert (
        count_featurizer_example[FEATURE]
        == message.get_features().feature.toarray().tolist()
    )
