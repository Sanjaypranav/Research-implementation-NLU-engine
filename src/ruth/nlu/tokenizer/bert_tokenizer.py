from typing import Any, Dict, List, Optional, Text

import torch
from ruth.constants import TEXT
from ruth.nlu.constants import ELEMENT_UNIQUE_NAME
from ruth.nlu.tokenizer.constants import max_length_for_padding
from ruth.nlu.tokenizer.tokenizer import Tokenizer
from ruth.shared.constants import TOKENS
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.nlu.training_data.tokens import Tokens
from transformers import BertTokenizer


class TokenizerBert(Tokenizer):
    defaults = {"model_name": "bert-base-uncased", "do_lower_case": True}

    def __init__(
        self,
        element_config: Optional[Dict[Text, Any]] = None,
        tokenizer: Optional["BertTokenizer"] = None,
    ):
        super(TokenizerBert, self).__init__(element_config)
        self.tokenizer = tokenizer or {}

    def _build_tokenizer(self) -> BertTokenizer:
        return BertTokenizer.from_pretrained(self.element_config["model_name"], do_lower_case=True)

    def _create_tokens(self, examples: List[RuthData]) -> List[torch.Tensor]:
        tokens = []
        # attention_masks = []
        for message in examples:
            encoded_dict = self.tokenizer.encode_plus(
                message.get(TEXT),
                add_special_tokens=True,
                max_length=max_length_for_padding,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            tokens.append(encoded_dict["input_ids"])
            # attention_masks.append(encoded_dict['attention_mask'])
        # return torch.cat(tokens, dim=0), torch.cat(attention_masks, dim=0)
        return tokens

    def _get_tokenized_data(self, training_data: TrainData):
        return self._create_tokens(training_data.training_examples)

    def _add_tokens_to_data(
        self, training_examples: List[RuthData], tokens: List[torch.Tensor]
    ):
        for message, token in zip(training_examples, tokens):
            message.set(TOKENS, Tokens(token, self.element_config[ELEMENT_UNIQUE_NAME]))
            # message.set(TOKENS, Tokens(attention_mask, self.element_config[ELEMENT_UNIQUE_NAME]))

    def train(self, training_data: TrainData) -> BertTokenizer:
        self.tokenizer = self._build_tokenizer()
        tokens = self._get_tokenized_data(training_data)
        self._add_tokens_to_data(training_data.training_examples, tokens)
        return self.tokenizer

    def parse(self, message: RuthData):
        parse_encoded = self.tokenizer.encode_plus(
            message.get(TEXT),
            add_special_tokens=True,
            max_length=max_length_for_padding,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token_ids = parse_encoded["input_ids"]
        # attentions_masks = parse_encoded['attention_masks']

        message.set(TOKENS, Tokens(token_ids, self.element_config[ELEMENT_UNIQUE_NAME]))
