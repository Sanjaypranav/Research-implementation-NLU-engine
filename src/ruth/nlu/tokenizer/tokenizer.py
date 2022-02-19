from abc import ABC, abstractmethod
from typing import Dict, Text, Any, Optional

from ruth.shared.nlu.training_data.collections import TrainData


class Tokenizer(ABC):

    def __init__(self, config: Optional[Dict[Text, Any]] = None):
        self.config = config

    @abstractmethod
    def _build_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def _create_tokens(self):
        raise NotImplementedError

    @staticmethod
    def get_data(training_data: TrainData):
        return training_data.get_text_list()

    def train(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError
