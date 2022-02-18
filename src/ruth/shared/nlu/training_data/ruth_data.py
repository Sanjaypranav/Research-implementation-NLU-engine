from typing import Any, Dict, List, Optional, Text

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.features import Features


class RuthData:
    def __init__(
        self,
        features: Optional[List[Features]] = None,
        data: Dict[Text, Any] = None,
    ):
        self.intent: Text = data.get(INTENT, "__mis__")
        self.text: Text = data.get(TEXT, "__mis__")
        self.features = features or []

    @classmethod
    def build(cls, intent: Text = None, text: Text = None) -> "RuthData":
        return cls(data={INTENT: intent, TEXT: text})

    def add_features(self, feature) -> None:
        if feature is not None:
            self.features.append(feature)

    def get_sparse_features(self):
        sparse_features = [feature for feature in self.features if feature.is_sparse()]
        return sparse_features
