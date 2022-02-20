from typing import Any, Dict, List, Optional, Text

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.features import Features
from scipy import sparse


class RuthData:
    def __init__(
        self,
        data: Dict[Text, Any] = None,
        features: Optional[List[Features]] = None,
    ):
        data = data or {}
        self.intent: Text = data.get(INTENT, "__mis__")
        self.text: Text = data.get(TEXT, "__mis__")
        self.features = features or []

    @classmethod
    def build(cls, intent: Text = None, text: Text = None) -> "RuthData":
        return cls(data={INTENT: intent, TEXT: text})

    def add_features(self, feature: Features) -> None:
        if feature is not None:
            self.features.append(feature)

    def get_sparse_features(self) -> List[sparse.spmatrix]:
        sparse_features = [
            feature.features for feature in self.features if feature.is_sparse()
        ]
        return sparse_features
