import logging
from typing import Any, Dict, List, Text

import numpy as np
import sklearn
from ruth.nlu.classifiers.classifier import Classifier
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class NaiveBayesClassifier(Classifier):
    defaults = {"priors": None, "var_smoothing": 1e-9}

    def __init__(
        self,
        element_config: Dict[Text, Any],
        config: Dict[Text, Any] = None,
        le: "sklearn.model_selection.GridSearchCV" = None,
        clf: "sklearn.preprocessing.LabelEncoder" = None,
    ):
        super(NaiveBayesClassifier, self).__init__(element_config=element_config)

        self.config = config or {}
        self.le = le or LabelEncoder()
        self.clf = clf

    def encode_the_str_to_int(self, labels: List[Text]) -> np.ndarray:
        return self.le.fit_transform(labels)

    def _create_classifier(self) -> "sklearn.naive_bayes.GaussianNB":
        from sklearn.naive_bayes import GaussianNB

        priors = self.element_config["priors"]
        var_smoothing = self.element_config["var_smoothing"]

        return GaussianNB(priors=priors, var_smoothing=var_smoothing)

    def train(self, training_data: TrainData):

        intents = [message.intent for message in training_data.intent_examples]
        if len(set(intents)) < 2:
            logger.warning(
                "There are no enough intent. "
                "Atleast two unique intent are needed to train the model"
            )
            return

        X = [message.get_sparse_features() for message in training_data.intent_examples]
        y = self.encode_the_str_to_int(
            [message.intent for message in training_data.training_examples]
        )

        X = np.reshape(X, (len(X), -1))
        self.clf = self._create_classifier()
        self.clf.fit(X, y)

    def parse(self, message: RuthData):
        pass
