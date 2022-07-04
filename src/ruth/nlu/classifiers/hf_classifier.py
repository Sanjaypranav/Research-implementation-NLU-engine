import logging
from typing import Dict, Text, Any, List, Tuple

import sklearn
from numpy import reshape, ndarray, fliplr, argsort
from rich.console import Console
from ruth.constants import INTENT, INTENT_RANKING

from ruth.nlu.classifiers import LABEL_RANKING_LIMIT

from ruth.shared.utils import json_pickle, json_unpickle
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from ruth.nlu.classifiers.ruth_classifier import IntentClassifier
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData

from ruth.nlu.classifiers.ruth_classifier import IntentClassifier

logger = logging.getLogger(__name__)

console = Console()

class HFClassifier(IntentClassifier):
    def __init__(self, element_config: Dict[Text, Any],
                 le: LabelEncoder = None,
                 clf: GridSearchCV = None):
        self.clf = clf
        super().__init__(element_config, le)

    @staticmethod
    def get_features(message: RuthData) -> sparse.spmatrix:
        feature = message.get_sparse_features()
        if feature is not None:
            return feature.feature[0]
        raise ValueError("There is no sentence. Not able to train SVMClassifier")

    def train(self, training_data: TrainData):
        intents: List[Text] = [message.get(INTENT) for message in training_data.intent_examples]
        if len(set(intents)) < 2:
            logger.warning(
                "There are no enough intent. "
                "At least two unique intent are needed to train the model"
            )
            return

        X = [
            self.get_features(message).toarray()
            for message in training_data.intent_examples
        ]
        y = self.encode_the_str_to_int(intents)

        X = reshape(X, (len(X), -1))

