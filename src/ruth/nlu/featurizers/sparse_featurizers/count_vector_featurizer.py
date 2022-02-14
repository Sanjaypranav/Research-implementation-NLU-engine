from ruth.nlu.featurizers.sparse_featurizers.sparse_featurizer import SparseFeaturizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from sklearn.feature_extraction.text import CountVectorizer


class CountVectorFeaturizer(SparseFeaturizer):
    def __init__(self, training_data: TrainData):
        super(CountVectorFeaturizer, self).__init__(training_data)
        self.vectorizer = self._build_vectorizer()

    def _build_vectorizer(self) -> CountVectorizer:
        return CountVectorizer()

    def train(self):
        self.vectorizer.fit(self.get_data())

    def parse(self, message: RuthData):
        return self.vectorizer.transform([message.text])
