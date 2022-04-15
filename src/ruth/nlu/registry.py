from typing import Dict, Text

from ruth.nlu.classifiers.naive_bayes_classifier import NaiveBayesClassifier
from ruth.nlu.elements import Element
from ruth.nlu.featurizers.dense_featurizers.hf_featurizer import HFFeaturizer
from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import (
    CountVectorFeaturizer,
)

element_classes = [
    # Featurizers
    CountVectorFeaturizer,
    # Classifiers
    NaiveBayesClassifier,
    HFFeaturizer,
]

registered_classes: Dict[Text, Element] = {cls.name: cls for cls in element_classes}
