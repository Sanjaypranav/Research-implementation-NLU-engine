from typing import Dict, Text, Any

from ruth.nlu.tokenizer.tokenizer import Tokenizer


class WhiteSpaceTokenizer(Tokenizer):
    def _build_tokenizer(self, parameters: Dict[Text, Any]):
        pass

    def __init__(self, element_config):
        super().__init__(element_config)
        ...

    def process(self):
        ...
