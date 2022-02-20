from typing import Any, Dict, List, Text

from ruth.shared.constants import ELEMENT_INDEX, KEY_LANGUAGE, KEY_PIPELINE


@property
def default_pipline() -> List[Dict[Text, Any]]:
    return [{"name": "CountVectorFeaturizer"}, {"name": "NaiveBayesClassifier"}]


class RuthConfig:
    def __init__(self, config: Dict[Text, Any]):
        self.config = config or {}

        self.language = config.get(KEY_LANGUAGE, "en")
        self.pipeline = self.index_the_pipeline(
            config.get(KEY_PIPELINE, default_pipline)
        )

    @staticmethod
    def index_the_pipeline(pipeline: List[Dict[Text, Any]]):
        if not pipeline:
            return []

        return [
            element.setdefault(ELEMENT_INDEX, index)
            for index, element in enumerate(pipeline)
        ]
