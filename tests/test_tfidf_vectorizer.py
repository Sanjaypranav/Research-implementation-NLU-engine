from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData

from tests.conftest import FEATURE


def test_tfidf_vectorizer(tfidf_featurizer_example: Dict[Text, Any]):
    training_data = TrainData.build(tfidf_featurizer_example[PATH])
    featurizer = registered_classes["TfidfVectorFeaturizer"].build({})
    featurizer.train(training_data)
    message = RuthData.build(text=tfidf_featurizer_example[TEXT])
    featurizer.parse(message)
    assert (
        tfidf_featurizer_example[FEATURE]
        == message.get_features().feature.toarray().tolist()
    )
