import json
from typing import Dict, Text, Any, Tuple

import torch

from ruth.constants import PATH
from ruth.nlu.tokenizer.bert_tokenizer import TokenizerBert
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from tests.conftest import bert_tokenizer_example, TOKEN


def test_bert_tokenizer(bert_tokenizer_example: Dict[Text, Any]):
    messages = []
    with open(bert_tokenizer_example[PATH], "r") as f:
        example_data = json.load(f)
    for data in example_data:
        messages.append(RuthData(data))
    training_data = TrainData(messages)

    tokenizer = TokenizerBert()
    tokenizer.train(training_data)

    assert type(bert_tokenizer_example[TOKEN]) == Tuple[torch.Tensor, torch.Tensor]



