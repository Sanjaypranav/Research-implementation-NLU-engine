from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT, TOKENS
from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_with_ham_spam_with_empty_training_data(whitespace_example: Dict[Text, Any]):
    training_data = TrainData.build(whitespace_example[PATH])
    tokenizer = registered_classes["WhiteSpaceTokenizer"].build({})
    tokenizer.train(training_data)
    message = RuthData.build(text=whitespace_example[TEXT])
    tokenizer.parse(message)
    assert whitespace_example[TOKENS] == message.get(TOKENS)
