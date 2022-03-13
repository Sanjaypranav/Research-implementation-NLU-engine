from typing import Any, Dict, Text

from ruth.shared.constants import ELEMENT_INDEX, KEY_NAME
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.nlu.training_data.utils import override_defaults


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
    def build(cls, element_config: Dict[Text, Any], config: RuthConfig):

        return cls(element_config)


class ElementBuilder:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

        self.element_cache = {}
