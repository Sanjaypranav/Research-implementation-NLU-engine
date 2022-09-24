import io
import os
from typing import Any, Dict, List, Optional, Text
from urllib import request

import numpy
from progressbar import progressbar
from ruth.nlu.classifiers.constants import MODEL_NAME
from ruth.nlu.constants import ELEMENT_UNIQUE_NAME
from ruth.nlu.featurizers.dense_featurizers.dense_featurizer import DenseFeaturizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.feature import Feature
from ruth.shared.nlu.training_data.ruth_data import RuthData
from tqdm import tqdm

pbar = None


class FastTextFeaturizer(DenseFeaturizer):
    DO_LOWER_CASE = "do_lower_case"

    defaults = {MODEL_NAME: "wiki-news-300d-1M.vec.zip", DO_LOWER_CASE: True}
    DEFAULT_MODELS_DIR = os.path.join(
        os.path.expanduser("~"), ".cache", "ruth", "models"
    )

    MODELS = {
        "wiki-news-300d-1M.vec.zip": "https://dl.fbaipublicfiles.com/"
        "fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        "wiki-news-300d-1M-subword.vec.zip": "https://dl.fbaipublicfiles.com/"
        "fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
        "crawl-300d-2M.vec.zip": "https://dl.fbaipublicfiles.com/"
        "fasttext/vectors-english/crawl-300d-2M.vec.zip",
        "crawl-300d-2M-subword.zip": "https://dl.fbaipublicfiles.com/"
        "fasttext/vectors-english/crawl-300d-2M-subword.zip",
    }

    def __init__(self, element_config: Optional[Dict[Text, Any]]):
        super(FastTextFeaturizer, self).__init__(element_config)
        self.vectors = None
        self.featurizer = {}
        if self.element_config[MODEL_NAME] not in self.MODELS:
            raise ValueError(
                "Model name not found. Please choose from the following: "
                "{}".format(list(self.MODELS.keys()))
            )
        self.file_path = self.download_models(self.element_config[MODEL_NAME])
        self.dimension = 300

    def download_models(self, specific_models=None):
        os.makedirs(self.DEFAULT_MODELS_DIR, exist_ok=True)

        def show_progress(block_num, block_size, total_size):
            global pbar
            if pbar is None:
                pbar = progressbar.ProgressBar(maxval=total_size)
                pbar.start()

            downloaded = block_num * block_size
            if downloaded < total_size:
                pbar.update(downloaded)
            else:
                pbar.finish()
                pbar = None

        for model_name, url in self.MODELS.items():
            if specific_models is not None and str(model_name) not in str(
                specific_models
            ):
                continue
            model_path = os.path.join(self.DEFAULT_MODELS_DIR, model_name)
            if os.path.exists(model_path):
                model_path = model_path[:-4]
                return model_path

            request.urlretrieve(url, model_path, show_progress)

            import zipfile

            with zipfile.ZipFile(model_path, "r") as zip_ref:
                zip_ref.extractall(self.DEFAULT_MODELS_DIR)

            model_path = model_path[:-4]
            return model_path
        raise f"""Given model {specific_models} not found.
                Please check the documentation and give the
                right Fastext model name """

    def train(self, training_data: TrainData):
        self.featurizer = self._build_featurizer()
        tokenized_data: List[List[Text]] = [
            message.get_tokenized_data() for message in training_data.intent_examples
        ]

        self.vectors = [
            self.get_vector_list(token_list) for token_list in tokenized_data
        ]

        for message, vector in zip(training_data.training_examples, self.vectors):
            message.add_features(
                Feature(vector, self.element_config[ELEMENT_UNIQUE_NAME])
            )

    def _build_featurizer(self):

        fasttext_corpus = io.open(
            self.file_path, "r", encoding="utf-8", newline="\n", errors="ignore"
        )
        model = {}
        for line in tqdm(fasttext_corpus, colour="red"):
            tokens = line.strip().split(" ")
            model[tokens[0]] = numpy.array(list(tokens[1:]))

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
        vector = self.get_vector_list(tokens)
        message.add_features(Feature(vector, self.element_config[ELEMENT_UNIQUE_NAME]))
