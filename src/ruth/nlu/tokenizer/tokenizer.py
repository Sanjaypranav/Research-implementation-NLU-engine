from typing import Dict, Text, Any

from ruth.shared.nlu.ruth_elements import Element
from ruth.nlu.constants import ELEMENT_UNIQUE_NAME
from ruth.shared.nlu.training_data.collections import TrainData


class Tokenizer(Element):
    def __init__(self, element_config):
        element_config = element_config or {}
        self.element_config = element_config
        element_config.setdefault(ELEMENT_UNIQUE_NAME, self.create_unique_name())
        super().__init__(element_config)

    # @abstractmethod
    def _build_tokenizer(self):
        raise NotImplementedError

    # @abstractmethod
    # def _create_tokens(self):
    #   raise NotImplementedError

    @staticmethod
    def get_data(training_data: TrainData):
        return training_data.get_text_list(
            training_examples=training_data.training_examples
        )

    # def train(self):
    #   raise NotImplementedError

    # def parse(self):
    #   raise NotImplementedError
