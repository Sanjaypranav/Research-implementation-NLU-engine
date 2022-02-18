from typing import Any, Dict, Text

from ruth.nlu.classifiers.classifier import Classifier
from ruth.shared.nlu.training_data.collections import TrainData


class NaiveBayesClassifier(Classifier):
    def __init__(self, config: Dict[Text, Any] = None):
        super(NaiveBayesClassifier, self).__init__()
        self.config = config or {}

    def train(self, training_data: TrainData):

        x = [
            message.get_sparse_features() for message in training_data.training_examples
        ]
        return x
