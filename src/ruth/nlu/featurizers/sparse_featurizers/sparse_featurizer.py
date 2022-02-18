from ruth.nlu.element import Element
from ruth.nlu.featurizers.sparse_featurizers.constants import (
    CLASS_FEATURIZER_UNIQUE_NAME,
)
from ruth.shared.nlu.training_data.collections import TrainData


class SparseFeaturizer(Element):
    def __init__(self, element_config):
        self.element_config = element_config or {}
        self.element_config.setdefault(
            CLASS_FEATURIZER_UNIQUE_NAME, self.create_unique_name
        )
        super().__init__(element_config)

    def _build_vectorizer(self):
        raise NotImplementedError

    @staticmethod
    def get_data(training_data: TrainData):
        return training_data.get_text_list()
