import copy
import logging

from ruth.shared.nlu.ruth_elements import ElementBuilder
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: RuthConfig, element_builder: ElementBuilder = None):
        self.config = config
        self.training_data = None

        if element_builder is None:
            element_builder = ElementBuilder()

        self.pipeline = self._build_pipeline(config, element_builder)

    @staticmethod
    def _build_pipeline(config: RuthConfig, element_builder: ElementBuilder):
        pipeline = []

        for index, pipe_element in enumerate(config.pipeline):
            element_cnf = config.get_element(index)
            element = element_builder.create_element(pipe_element.name, element_cnf)
            pipeline.append(element)

        return pipeline

    def train(self, data: TrainData, **kwargs):
        self.training_data = data

        context = kwargs
        working_data: TrainData = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info(f"Starting to train element {component.name}")
            # component.prepare_partial_processing(self.pipeline[:i], context)
            component.train(working_data, self.config, **context)
            logger.info("Finished training element.")

        return Interpreter(self.pipeline, context)
