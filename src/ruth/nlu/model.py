import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Text

import ruth
from ruth.constants import INTENT, TEXT
from ruth.nlu.constants import RUTH
from ruth.nlu.elements import Element
from ruth.nlu.registry import registered_classes
from ruth.nlu.utils import module_path_from_object
from ruth.shared.constants import INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.utils import json_pickle

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

    @staticmethod
    def load_element(name: Text, element_config: Dict[Text, Any], model_dir: Path):
        if name not in registered_classes:
            logger.error(
                f"Given {name} element is not an registered element. We won't support custom element now."
            )
        else:
            return registered_classes[name].load(element_config, model_dir=model_dir)


class MetaData:
    def __init__(self, metadata: Dict[Text, Any]):
        self.metadata = metadata

    def get(self, prop: Text, defaults: Any = None):
        return self.metadata.get(prop, defaults)

    def persist(self, model_dir: Path):
        metadata = self.metadata.copy()

        metadata.update(
            {"Training completed at": datetime.now(), "ruth_version": ruth.VERSION}
        )
        filename = model_dir / "metadata.json"
        json_pickle(filename, metadata, indent=4)


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

    @staticmethod
    def get_filename(index, name):
        return f"element_{index}_{name}"

    def train(self, data: TrainData):
        self.training_data = data

        working_data: TrainData = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info(f"Starting to train element {component.name}")
            # component.prepare_partial_processing(self.pipeline[:i], context)
            component.train(working_data)
            logger.info("Finished training element.")
        # return Interpreter(self.pipeline, context)

    def persist(self, path: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        model_name = RUTH + "_" + timestamp
        metadata = {"language": self.config.language, "pipeline": []}

        model_dir = path.absolute() / model_name
        model_dir.mkdir(exist_ok=True)

        for index, element in enumerate(self.pipeline):
            file_name = self.get_filename(index, name=element.name)
            custom_meta = element.persist(file_name, model_dir)
            element_meta = element.element_config
            if element_meta:
                element_meta.update(custom_meta)
            element_meta["class"] = module_path_from_object(element)

            metadata["pipeline"].append(element_meta)

        MetaData(metadata).persist(model_dir)
        return model_dir


class Interpreter:
    def __init__(
            self,
            pipeline: List[Element],
            model_metadata: Optional[MetaData] = None,
    ) -> None:
        self.pipeline = pipeline
        self.model_metadata = model_metadata

    def parse(
            self,
            text: Text,
    ) -> Dict[Text, Any]:
        """Parse the input text, classify it and return pipeline result.

        The pipeline result usually contains intent and entities."""

        if not text:
            output = self.default_output_attributes()
            output["text"] = ""
            return output

        data = self.default_output_attributes()
        data[TEXT] = text

        message = RuthData(data=data)

        for element in self.pipeline:
            element.parse(message)

        output = self.default_output_attributes()
        output.update(message.as_dict())
        return output

    @staticmethod
    def default_output_attributes():
        return {
            TEXT: "",
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
        }
