from typing import Any, Dict, List, Text

import numpy as np
from numpy import ndarray
from ruth.nlu.elements import Element
from sklearn.preprocessing import LabelEncoder


class IntentClassifier(Element):
    MAX_LENGTH = 30000

    def __init__(self, element_config: Dict[Text, Any], le: LabelEncoder):
        self.le = le or LabelEncoder()
        super(IntentClassifier, self).__init__(element_config)

    def encode_the_str_to_int(self, labels: List[Text]) -> ndarray:
        return self.le.fit_transform(labels)

    def _change_int_to_text(self, prediction: ndarray) -> ndarray:
        return self.le.inverse_transform(prediction)

    @staticmethod
    def ravel_vector(vector: ndarray) -> ndarray:
        return np.ravel(vector)

    @staticmethod
    def pad_vector(vector: ndarray, max_length: int) -> ndarray:
        if len(vector) < max_length:
            vector = np.pad(vector, (0, max_length - len(vector)), "constant")
        if len(vector) > max_length:
            vector = vector[:max_length]
        return vector

    def get_max_length(self, vector: List[ndarray]):
        max_length = 0
        for i in vector:
            if max_length < i.shape[0]:
                max_length = i.shape[0]
            if max_length > self.MAX_LENGTH:
                return self.MAX_LENGTH
        return max_length

    @staticmethod
    def check_dense(X) -> bool:
        if isinstance(X, ndarray):
            return True
        return False
