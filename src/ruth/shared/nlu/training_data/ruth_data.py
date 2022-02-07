from typing import Dict, Text, Any, List, Optional

from ruth.constants import INTENT, TEXT


class RuthData:
    def __init__(self, data: Dict[Text, Any] = None):
        self.intent = data[INTENT]
        self.text = data[TEXT]

    @classmethod
    def build(cls,
              intent: Text = None,
              text: Text = None) -> "RuthData":
        return cls({INTENT: intent, TEXT: text})
