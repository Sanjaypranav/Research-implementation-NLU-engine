from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT
from ruth.nlu.registry import registered_classes
from ruth.shared.constants import ATTENTION_MASKS, INPUT_IDS
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData

from tests.conftest import ATTENTION_MASK, INPUT_ID


def test_hf_tokenizer(hf_tokenizer_example: Dict[Text, Any]):
    training_data = TrainData.build(hf_tokenizer_example[PATH])
    tokenizer = registered_classes["HFTokenizer"].build({})
    tokenizer.train(training_data=training_data)
    message = RuthData(data={TEXT: hf_tokenizer_example[TEXT]})
    tokenizer.parse(message)
    assert message.data[INPUT_IDS] == hf_tokenizer_example[INPUT_ID]
    assert message.data[ATTENTION_MASKS] == hf_tokenizer_example[ATTENTION_MASK]
