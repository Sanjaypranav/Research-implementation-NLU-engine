import json
from typing import Any, Dict, Text

from constants import PATH
from nlu import TokenizerBert
from shared import RuthData, TrainData
from transformers import BertTokenizer


def test_bert_tokenizer(bert_tokenizer_example: Dict[Text, Any]):
    messages = []
    with open(bert_tokenizer_example[PATH], "r") as f:
        example_data = json.load(f)
    for data in example_data:
        messages.append(RuthData(data))
    training_data = TrainData(messages)

    tokenizer = TokenizerBert()
    s = tokenizer.train(training_data)

    assert isinstance(type(s), type(BertTokenizer))
