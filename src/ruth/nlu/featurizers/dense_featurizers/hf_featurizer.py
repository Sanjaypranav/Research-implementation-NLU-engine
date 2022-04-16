import logging
from typing import Any, Dict, List, Optional, Text, Type

from ruth.nlu.featurizers.dense_featurizers.dense_featurizer import DenseFeaturizer
from ruth.nlu.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTHS = {
    "bert": 512,
    "gpt2": 512,
}


class HFFeaturizer(DenseFeaturizer):

    defaults = {
        "model_name": "bert",
        "model_weights": None,
        "cache_dir": None,
    }

    @classmethod
    def required_element(cls) -> List[Type]:
        return [Tokenizer]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        skip_model_load: bool = False,
        hf_transformers_loaded: bool = False,
    ) -> None:
        super(HFFeaturizer, self).__init__(component_config)
        if hf_transformers_loaded:
            return
        self._load_metadata()
        self._load_model(skip_model_load)
