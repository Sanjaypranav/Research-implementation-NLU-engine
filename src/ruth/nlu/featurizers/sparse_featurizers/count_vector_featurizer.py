from typing import List, Optional

from ruth.nlu.featurizers.sparse_featurizers.sparse_featurizer import SparseFeaturizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.features import Features
from ruth.shared.nlu.training_data.ruth_data import RuthData
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class CountVectorFeaturizer(SparseFeaturizer):
    def __init__(self, vectorizer: Optional["CountVectorizer"] = None):
        super(CountVectorFeaturizer, self).__init__()
        self.vectorizer = vectorizer or {}

    def _build_vectorizer(self) -> CountVectorizer:
        return CountVectorizer()

    def _check_attribute_vocabulary(self) -> bool:
        """Checks if trained vocabulary exists in attribute's count vectorizer."""
        try:
            return hasattr(self.vectorizer, "vocabulary_")
        except (AttributeError, KeyError):
            return False

    def create_vectors(self, examples: List[RuthData]) -> List[sparse.spmatrix]:
        features = []
        for message in examples:
            features.append(self.vectorizer.transform(message.text))
        return features

    def _get_featurizer_data(self, training_data: TrainData) -> List[sparse.spmatrix]:
        if self._check_attribute_vocabulary():
            return self.create_vectors(training_data.training_examples)
        else:
            return []

    @staticmethod
    def _add_features_to_data(
        training_examples: List[RuthData], features: List[sparse.spmatrix]
    ):
        for message, feature in zip(training_examples, features):
            message.add_features(Features(feature))

    def train(self, training_data: TrainData) -> CountVectorizer:
        self.vectorizer = self._build_vectorizer()
        self.vectorizer.fit(self.get_data(training_data))
        features = self._get_featurizer_data(training_data)
        self._add_features_to_data(training_data.training_examples, features)
        return self.vectorizer

    def parse(self, message: RuthData) -> sparse.coo_matrix:
        sentence_vec = self.vectorizer.transform([message.text])
        return sentence_vec.tocoo()
