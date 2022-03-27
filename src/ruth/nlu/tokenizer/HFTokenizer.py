from typing import Dict, Text, Any, Optional

from transformers import AutoTokenizer

from ruth.nlu.tokenizer.tokenizer import Tokenizer


class HFTokenizer(Tokenizer):
    defaults = {
        'model_name': 'bert-base-uncased',
        'do_lower_case': True
    }

    def __init__(self, element_config: Optional[Dict[Text, Any]], tokenizer: AutoTokenizer):
        super(HFTokenizer, self).__init__(element_config)
        self.tokenizer = tokenizer or {}

    def _build_tokenizer(self,
                         parameters: Dict[Text, Any]
                         ):
        return AutoTokenizer.from_pretrained(self.element_config['model_name'])


