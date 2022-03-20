from typing import Any, Dict, Text

import ruth.nlu.ruth_elements


class Classifier(Element):
    def __init__(self, element_config: Dict[Text, Any]):
        super(Classifier, self).__init__(element_config)
