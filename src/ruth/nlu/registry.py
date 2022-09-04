from typing import Dict, Text

from ruth.nlu.classifiers.naive_bayes_classifier import NaiveBayesClassifier
from ruth.nlu.classifiers.svm_classifier import SVMClassifier
from ruth.nlu.elements import Element
from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import (
    CountVectorFeaturizer,
)
from ruth.nlu.tokenizer.whitespace_tokenizer import WhiteSpaceTokenizer

element_classes = [
    # Tokenizer
    WhiteSpaceTokenizer,
    # Featurizers
    CountVectorFeaturizer,
    # Classifiers
    NaiveBayesClassifier,
    SVMClassifier,
]

registered_classes: Dict[Text, Element] = {cls.name: cls for cls in element_classes}
