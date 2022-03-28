from pathlib import Path

from ruth.nlu.constants import DEFAULT_MODEL_NAME
from ruth.nlu.model import Trainer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig


def train_pipeline(config: RuthConfig, training_data: TrainData):
    trainer = Trainer(config)
    trainer.train(training_data)

    model_dir = Path(DEFAULT_MODEL_NAME)
    model_dir.mkdir(exist_ok=True)
    trainer.persist(model_dir)

