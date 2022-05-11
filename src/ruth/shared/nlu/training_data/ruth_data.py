import copy
from typing import Any, Dict, List, Optional, Text

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.features import Features


class RuthData:
    def __init__(
        self,
        data: Dict[Text, Any] = None,
        features: Optional[List[Features]] = None,
    ):
        self.features = features or []
        self.data = data or {}

    @classmethod
    def build(cls, intent: Text = None, text: Text = None) -> "RuthData":
        return cls(data={INTENT: intent, TEXT: text})

    def add_features(self, feature: Features) -> None:
        if feature is not None:
            self.features.append(feature)

    def set(self, key: Text, value: Any):
        self.data[key] = value

    def get(self, key: Text, default: Any = None):
        return self.data.get(key, default)

    @staticmethod
    def _combine_features(
        features: List[Features], featurizers: List[Text]
    ) -> Features:

        combined_features = None

        for feature in features:
            if not combined_features:
                combined_features = copy.deepcopy(feature)
                combined_features.origin = featurizers
            else:
                combined_features.combine_with_features(feature)
        return combined_features

    def get_sparse_features(self, featurizers: List[Text] = None) -> Features:
        combined_features = self._combine_features(self.features, featurizers)
        return combined_features

    def as_dict(self) -> Dict:
        return {key: value for key, value in self.data.items() if value is not None}
