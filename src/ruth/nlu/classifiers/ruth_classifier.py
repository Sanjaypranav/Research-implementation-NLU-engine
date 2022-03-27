from typing import Any, Dict, Text

from ruth.shared.nlu.ruth_elements import Element


class Classifier(Element):
    def __init__(self, element_config: Dict[Text, Any]):
        super(Classifier, self).__init__(element_config)
