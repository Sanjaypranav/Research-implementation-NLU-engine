from typing import Dict, Text, Any, Optional, List

from transformers import AutoTokenizer

from ruth.constants import TEXT
from ruth.nlu.constants import ELEMENT_UNIQUE_NAME
from ruth.nlu.tokenizer.constants import MAX_LENGTH_FOR_PADDING
from ruth.nlu.tokenizer.tokenizer import Tokenizer
from ruth.shared.constants import TOKENS
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.nlu.training_data.tokens import Tokens


class HFTokenizer(Tokenizer):
    MODEL_NAME = "model_name"
    DO_LOWER_CASE = "do_lower_case"

    defaults = {
        MODEL_NAME: 'bert-base-uncased',
        DO_LOWER_CASE: True
    }

    def __init__(self, element_config: Optional[Dict[Text, Any]], tokenizer=None):
        super(HFTokenizer, self).__init__(element_config)
        self.tokenizer = tokenizer or {}

    def _build_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.element_config[self.MODEL_NAME])

    def _create_tokens(self, examples: TrainData):
        tokens = []
        # attention_masks = []
        for message in examples:
            token = self.tokenizer.tokenize(
                message.get(TEXT),
                add_special_tokens=True,
                max_length=MAX_LENGTH_FOR_PADDING,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            tokens.append(token)
            # attention_masks.append(encoded_dict['attention_mask'])
        # return torch.cat(tokens, dim=0), torch.cat(attention_masks, dim=0)
        return tokens

    def get_tokenized_data(self, training_data: TrainData):
        return self._create_tokens(training_data.training_examples)

    def _add_tokens_to_data(
            self,
            training_examples: List[RuthData],
            tokens: List[Text]
    ):
        for message, token in zip(training_examples, tokens):
            message.set(TOKENS, Tokens(token, self.element_config[ELEMENT_UNIQUE_NAME]))

    def train(self, training_data: TrainData):
        self.tokenizer = self._build_tokenizer()
        tokens = self.get_tokenized_data(training_data)
        self._add_tokens_to_data(training_data.training_examples, tokens)

    def parse(self, message: RuthData):
        parser_token = self.tokenizer.tokenize(
            message.get(TEXT),
            add_special_tokens=True,
            max_length=MAX_LENGTH_FOR_PADDING,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

