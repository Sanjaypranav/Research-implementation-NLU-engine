from abc import ABC
from typing import List, Dict, Text, Any

from transformers import BertTokenizer
from ruth.nlu.tokenizer.tokenizer import Tokenizer
from ruth.shared.nlu.training_data.ruth_data import RuthData


class TokenizerBert(Tokenizer, ABC):

    def __init__(self, config):
        super().__init__(config)
        self.default_config = self.default_config()
        self.tokenizer = self._build_tokenizer(self)

    @staticmethod
    def default_config() -> Dict[Text, Any]:
        return {
            'bert_size': "bert-base-uncased",
            'do_lower_case': True
        }

    @staticmethod
    def _build_tokenizer(self) -> BertTokenizer:
        return BertTokenizer.from_pretrained(self.default_config['bert_size'],
                                             do_lower_case=self.default_config['do_lower_case'])

    def create_tokens(self, examples: List[RuthData]):
        features = []
        for message in examples:
            features.append(self.tokenizer.transform(message.text))
        return features

    def train(self):
        ...

    def parse(self):
        ...
