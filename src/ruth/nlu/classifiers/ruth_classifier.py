from typing import Any, Dict, Text, List

from numpy import ndarray
from sklearn.preprocessing import LabelEncoder

from ruth.shared.nlu.ruth_elements import Element


class IntentClassifier(Element):
    def __init__(self, element_config: Dict[Text, Any], le: LabelEncoder):
        self.le = le or LabelEncoder()
        super(IntentClassifier, self).__init__(element_config)

    def encode_the_str_to_int(self, labels: List[Text]) -> ndarray:
        return self.le.fit_transform(labels)

    def _change_int_to_text(self, prediction: ndarray) -> ndarray:
        return self.le.inverse_transform(prediction)
