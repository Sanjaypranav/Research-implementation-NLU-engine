import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Text, Tuple

import torch
from numpy import argsort, fliplr, ndarray
from rich.console import Console
from ruth.constants import INTENT, INTENT_RANKING
from ruth.nlu.classifiers import LABEL_RANKING_LIMIT
from ruth.nlu.classifiers.constants import BATCH_SIZE, EPOCHS, MODEL_NAME
from ruth.nlu.classifiers.ruth_classifier import IntentClassifier
from ruth.nlu.tokenizer.hf_tokenizer import HFTokenizer
from ruth.shared.constants import (
    ATTENTION_MASKS,
    DEVICE,
    INPUT_IDS,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.utils import get_device, json_pickle, json_unpickle
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import logging as transformer_logging

torch.cuda.empty_cache()
logger = logging.getLogger(__name__)
transformer_logging.set_verbosity_error()

console = Console()


class HFClassifier(IntentClassifier):
    defaults = {
        EPOCHS: 100,
        MODEL_NAME: "bert-base-uncased",
        BATCH_SIZE: 4,
        DEVICE: "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(
        self,
        element_config: Dict[Text, Any],
        le: LabelEncoder = None,
        model: Module = None,
    ):
        super().__init__(element_config, le)
        self.model = model
        self.device = get_device(self.element_config[DEVICE])

    def required_element(self):
        return [HFTokenizer]

    def _build_model(self, label_count):
        return AutoModelForSequenceClassification.from_pretrained(
            self.element_config[MODEL_NAME], num_labels=label_count
        )

    @staticmethod
    def get_input_ids(message: RuthData) -> Dict[Text, List[int]]:
        input_ids = message.get(INPUT_IDS)
        if input_ids is not None:
            return input_ids
        raise ValueError("There is no sentence. Not able to train HFClassifier")

    @staticmethod
    def get_attention_masks(message: RuthData) -> Dict[Text, List[int]]:
        attention_masks = message.get(ATTENTION_MASKS)
        if attention_masks is not None:
            return attention_masks
        raise ValueError("There is no sentence. Not able to train HFClassifier")

    @staticmethod
    def get_optimizer(model):
        return AdamW(model.parameters(), lr=5e-5)

    @property
    def get_params(self):
        return {
            EPOCHS: self.element_config[EPOCHS],
            BATCH_SIZE: self.element_config[BATCH_SIZE],
        }

    def train(self, training_data: TrainData):
        intents: List[Text] = [
            message.get(INTENT) for message in training_data.intent_examples
        ]
        if len(set(intents)) < 2:
            logger.warning(
                "There are no enough intent. "
                "At least two unique intent are needed to train the model"
            )
            return

        X = {
            "input_ids": [
                self.get_input_ids(message) for message in training_data.intent_examples
            ],
            "attention_masks": [
                self.get_attention_masks(message)
                for message in training_data.intent_examples
            ],
        }

        y = self.encode_the_str_to_int(intents)
        label_count = len(Counter(y).keys())

        params = self.get_params

        loaded_data = HFDatasetLoader(X, y)
        batched_data = DataLoader(
            loaded_data, batch_size=params[BATCH_SIZE], shuffle=True
        )

        self.model = self._build_model(label_count)

        optimizer = self.get_optimizer(self.model)
        console.print("device: " + str(self.device) + " is used")
        self.model.to(self.device)

        self.model.train()
        for epoch in range(params[EPOCHS]):
            for batch in tqdm(batched_data, desc="epoch " + str(epoch)):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_masks"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(
                    input_ids, attention_mask=attention_masks, labels=labels
                )
                loss = outputs[0]
                loss.backward()
                optimizer.step()

    def persist(self, file_name: Text, model_dir: Path):
        classifier_file_name = file_name + "_classifier"
        encoder_file_name = file_name + "_encoder.pkl"

        classifier_path = str(model_dir) + "/" + classifier_file_name
        encoder_path = str(model_dir) + "/" + encoder_file_name

        if self.model and self.le:
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_to_save.save_pretrained(classifier_path)
            json_pickle(encoder_path, self.le)

        return {"classifier": classifier_file_name, "encoder": encoder_file_name}

    @classmethod
    def load(cls, meta: Dict[Text, Any], model_dir: Path, **kwargs):
        classifier_file_name = model_dir / meta["classifier"]
        encoder_file_name = model_dir / meta["encoder"]

        classifier = AutoModelForSequenceClassification.from_pretrained(
            classifier_file_name
        )
        le = json_unpickle(Path(encoder_file_name))

        return cls(meta, model=classifier, le=le)

    def _predict(self, input_ids, attention_masks) -> Tuple[ndarray, ndarray]:
        predictions = self.predict_probabilities(input_ids, attention_masks)
        sorted_index = fliplr(argsort(predictions, axis=1))
        return sorted_index[0], predictions[:, sorted_index][0][0]

    def predict_probabilities(self, input_ids, attention_masks):
        self.model.to(self.device)
        self.model.eval()
        probabilities = self.model(
            torch.tensor(input_ids, device=self.device),
            attention_mask=torch.tensor(attention_masks, device=self.device),
        )[0]
        probabilities = nn.functional.softmax(probabilities, dim=-1)
        probabilities = probabilities.to(torch.device("cpu"))
        probabilities = probabilities.detach().numpy()
        return probabilities

    def parse(self, message: RuthData):
        input_ids = [message.get(INPUT_IDS)]
        attention_masks = [message.get(ATTENTION_MASKS)]
        index, probabilities = self._predict(input_ids, attention_masks)

        intents = self._change_int_to_text(index)
        probabilities = probabilities

        if intents.size > 0 and probabilities.size > 0:
            ranking = list(zip(list(intents), list(probabilities)))[
                :LABEL_RANKING_LIMIT
            ]
            intent = {
                INTENT_NAME_KEY: intents[0],
                PREDICTED_CONFIDENCE_KEY: float(probabilities[0]),
            }
            intent_rankings = [
                {INTENT_NAME_KEY: name, PREDICTED_CONFIDENCE_KEY: float(probability)}
                for name, probability in ranking
            ]
        else:
            intent = {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0}
            intent_rankings = []
        message.set(INTENT, intent)
        message.set(INTENT_RANKING, intent_rankings)


class HFDatasetLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
