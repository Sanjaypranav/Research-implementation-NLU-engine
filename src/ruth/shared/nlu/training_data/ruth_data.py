from typing import Any, Dict, List, Optional, Text

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.features import Features
from ruth.shared.nlu.training_data.tokens import Tokens


class RuthData:
    def __init__(
            self,
            features: Optional[List[Features]] = None,
            tokens: Optional[List[Tokens]] = None,
            attention_masks: Optional[List[Tokens]] = None,
            data: Dict[Text, Any] = None,
    ):
        data = data or {}
        self.intent: Text = data.get(INTENT, "__mis__")
        self.text: Text = data.get(TEXT, "__mis__")
        self.features = features or []
        self.tokens = tokens or []
        self.attention_masks = attention_masks or []

    @classmethod
    def build(cls, intent: Text = None, text: Text = None) -> "RuthData":
        return cls(data={INTENT: intent, TEXT: text})

    def add_features(self, feature: Features) -> None:
        if feature is not None:
            self.features.append(feature)

    def add_tokens(self, token) -> None:
        if token is not None:
            self.tokens.append(token)

    def add_attention_masks(self, attention_mask) -> None:
        if attention_mask is not None:
            self.attention_masks.append(attention_mask)

    def get_sparse_features(self):
        sparse_features = [feature for feature in self.features if feature.is_sparse()]
        return sparse_features

    # def get_tokens(self):
    #     tokens = [token for token in self.tokens if token.is_token()]

