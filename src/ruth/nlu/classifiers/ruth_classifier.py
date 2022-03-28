from typing import Any, Dict, Text

from ruth.nlu.elements import Element
from ruth.shared.nlu.ruth_elements import Element


class IntentClassifier(Element):
    def __init__(self, element_config: Dict[Text, Any]):
        super(IntentClassifier, self).__init__(element_config)
