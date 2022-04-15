import logging
from pathlib import Path
from typing import Any, Dict, Text

from ruth.shared.constants import ELEMENT_INDEX, KEY_NAME
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.nlu.training_data.utils import override_defaults

logger = logging.getLogger(__name__)


class ElementMetaClass(type):
    @property
    def name(cls) -> Text:
        return cls.__name__


class Element(metaclass=ElementMetaClass):
    defaults = {}

    def __init__(self, element_config: Dict[Text, Any]):
        element_config = element_config or {}

        element_config[KEY_NAME] = self.name
        self.element_config = override_defaults(self.defaults, element_config)

    @property
    def name(self):
        return type(self).name

    def train(self, training_data: TrainData):
        pass

    def parse(self, message: RuthData):
        pass

    def create_unique_name(self) -> Text:
        idx = self.element_config.get(ELEMENT_INDEX)
        return self.name if idx is None else f"element_{idx}_{self.name}"

    @classmethod
    def build(cls, element_config: Dict[Text, Any]):
        return cls(element_config)

    def persist(self, file_name: Text, model_dir: Text):
        pass

    @classmethod
    def load(cls, meta: Dict[Text, Any], model_dir: Path, **kwargs: Any):
        return cls(meta)
