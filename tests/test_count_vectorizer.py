import json
from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT
from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import (
    CountVectorFeaturizer,
)
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData

from tests.conftest import FEATURE


def test_count_vectorizer(count_featurizer_example: Dict[Text, Any]):
    messages = []
    with open(count_featurizer_example[PATH], "r") as f:
        example_data = json.load(f)
    for data in example_data:
        messages.append(RuthData(data=data))
    training_data = TrainData(messages)

    featurizer = CountVectorFeaturizer()
    featurizer.train(training_data)
    test_message = RuthData.build(text=count_featurizer_example[TEXT])
    featurized_text = featurizer.parse(test_message)

    assert count_featurizer_example[FEATURE] == featurized_text.toarray().tolist()
