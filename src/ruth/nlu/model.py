import copy
import logging
from typing import Any, Dict, Text

from ruth.nlu.registry import registered_classes
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig

logger = logging.getLogger(__name__)


class ElementBuilder:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

        self.element_cache = {}

    @staticmethod
    def create_element(name: Text, element_config: Dict[Text, Any]):
        if name not in registered_classes:
            logger.error(
                f"Given {name} element is not an registered element. We won't support custom element now."
            )
        else:
            return registered_classes[name].build(element_config)


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
            element = element_builder.create_element(pipe_element["name"], element_cnf)
            pipeline.append(element)

        return pipeline

    def train(self, data: TrainData):
        self.training_data = data

        # context = kwargs
        working_data: TrainData = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info(f"Starting to train element {component.name}")
            # component.prepare_partial_processing(self.pipeline[:i], context)
            component.train(working_data)
            logger.info("Finished training element.")

        # return Interpreter(self.pipeline, context)
