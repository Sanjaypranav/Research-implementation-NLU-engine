from typing import Dict, Text, Any, List, Optional


class RuthData:
    def __init__(self, data: Dict[Text, Any] = None):
        ""
        self.intent = ""
        self.text = ""

    @classmethod
    def build(cls,
              intent: Text,
              text: Text) -> "RuthData":
        ...
