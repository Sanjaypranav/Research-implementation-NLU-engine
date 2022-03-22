from ruth.nlu.model import Trainer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig


def train_pipeline(config: RuthConfig, training_data: TrainData):
    trainer = Trainer(config)
    trainer.train(training_data)
