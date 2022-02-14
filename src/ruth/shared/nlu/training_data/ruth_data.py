from typing import Any, Dict, Text

from ruth.constants import INTENT, TEXT


class RuthData:
    def __init__(self, data: Dict[Text, Any] = None):
        self.intent: Text = data.get(INTENT, "__mis__")
        self.text: Text = data.get(TEXT, "__mis__")

    @classmethod
    def build(cls, intent: Text = None, text: Text = None) -> "RuthData":
        return cls({INTENT: intent, TEXT: text})
