import logging
from typing import List, Type, Dict, Text, Any, Optional

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
    def required_components(cls) -> List[Type]:
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

    @classmethod
    def create(
            cls, element_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "DenseFeaturizer":
        if isinstance(config, Metadata):
            hf_transformers_loaded = "HFTransformersNLP" in [
                c["name"] for c in config.metadata["pipeline"]
            ]
        else:
            hf_transformers_loaded = "HFTransformersNLP" in config.component_names
        return cls(element_config, hf_transformers_loaded=hf_transformers_loaded)

    def _load_metadata(self) -> None:
        from ruth.nlu.utils.hugging_face_utils import (
            model_class_dict,
            model_weights_defaults,
        )

        self.model_name = model_class_dict["bert"]  # TODO: self._config["model_name"]

        self.max_model_sequence_length = MAX_SEQUENCE_LENGTHS["bert"]

        self.model_weights = model_weights_defaults["bert"]
        self.cache_dir = model_weights_defaults[]

    def _load_model(self) -> None:
        from ruth.nlu.utils.hugging_face_utils import (
            model_class_dict,
            model_tokenizer_dict,
        )

        self.tokenizer = model_tokenizer_dict["bert"].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        )
        self.model = model_class_dict[self.model_name].from_pretrained(  # type: ignore[no-untyped-call] # change chache dir
            self.model_weights, cache_dir=self.cache_dir
        )

        self.pad_token_id = self.tokenizer.unk_token_id

    def


