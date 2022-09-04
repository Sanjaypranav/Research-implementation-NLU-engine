import io
from typing import Any, Dict, List, Optional, Text

import numpy
from ruth.nlu.classifiers.constants import MODEL_NAME
from ruth.nlu.tokenizer.tokenizer import Tokenizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from tqdm import tqdm


class FastTextFeaturizer(Tokenizer):
    DO_LOWER_CASE = "do_lower_case"

    defaults = {MODEL_NAME: "wiki-news-300d-1M.vec", DO_LOWER_CASE: True}

    def __init__(self, element_config: Optional[Dict[Text, Any]]):
        super(FastTextFeaturizer, self).__init__(element_config)
        self.vectors = None
        self.featurizer = {}
        self.file_path = "/home/subash/Downloads/wiki-news-300d-1M.vec"
        self.dimension = 300

    def train(self, training_data: TrainData):
        self.featurizer = self._build_featurizer()
        tokenized_data: List[List[Text]] = [
            message.get_tokenized_data() for message in training_data.intent_examples
        ]

        self.vectors = [
            self.get_vector_list(token_list) for token_list in tokenized_data
        ]

    def _build_featurizer(self):

        fasttext_corpus = io.open(
            self.file_path, "r", encoding="utf-8", newline="\n", errors="ignore"
        )
        model = {}
        for line in tqdm(fasttext_corpus, colour="red"):
            tokens = line.strip().split(" ")
            model[tokens[0]] = numpy.array(list(map(float, tokens[1:])))

        return model

    def get_vector_list(self, token_list):
        if self.featurizer == {}:
            self._build_featurizer()
        if not token_list:
            return numpy.zeros(self.dimension)
        return numpy.array([self.get_vector(token) for token in token_list])

    def get_vector(self, token):
        if token in self.featurizer and self.featurizer != {}:
            return self.featurizer[token]
        else:
            return numpy.zeros(self.dimension)

    def parse(self, message: RuthData):
        tokens = message.get_tokenized_data()
        return self.get_vector_list(tokens)
