from typing import Dict, Text, Type

from transformers import PreTrainedTokenizer, BertTokenizer

model_class_dict: Dict[Text, Text] = {
    "bert": 'bert-base-uncased',
    "gpt2": '',  # TODO : Add GPT2 model path
}

model_weights_defaults = {
    "bert": '',
    "gpt2": '',
}

model_tokenizer_dict: Dict[Text, Type[PreTrainedTokenizer]] = {
    "bert": BertTokenizer,
}
