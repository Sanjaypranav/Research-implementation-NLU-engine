from typing import Any, Dict, List, Text

import numpy as np
from ruth.nlu.classifiers.classifier import Classifier
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


class NaiveBayesClassifier(Classifier):
    def __init__(
        self,
        element_config: Dict[Text, Any],
        config: Dict[Text, Any] = None,
        le: LabelEncoder = None,
        clf: GridSearchCV = None,
    ):
        super(NaiveBayesClassifier, self).__init__(element_config=element_config)

        self.config = config or {}
        self.le = le or LabelEncoder()
        self.clf = clf

    def encode_the_str_to_int(self, labels: List[Text]) -> np.ndarray:
        return self.le.fit_transform(labels)

    def train(self, training_data: TrainData):
        X = [
            message.get_sparse_features() for message in training_data.training_examples
        ]
        y = self.encode_the_str_to_int(
            [message.intent for message in training_data.training_examples]
        )

        return X, y

    def parse(self, message: RuthData):
        pass
