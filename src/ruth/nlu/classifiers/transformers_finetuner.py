import logging
from typing import Any, Dict, Text

from ruth.nlu.classifiers.ruth_classifier import Classifier
from ruth.nlu.classifiers.constants import EPOCHS
from ruth.shared.nlu.training_data.collections import TrainData

logger = logging.getLogger(__name__)


class TransformerClassifier:
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}

    def preprocess_train_data(self):
        pass

    def __init__(
        self,
        config: Dict[Text, Any],
    ) -> None:
        self.element_config = config

    @classmethod
    def create(cls, config: Dict[Text, Any]):
        return cls(config)

    def train(self, training_data: TrainData):
        self.preprocess_train_data(training_data)

        model_data = self.preprocess_train_data(training_data)
        return model_data
