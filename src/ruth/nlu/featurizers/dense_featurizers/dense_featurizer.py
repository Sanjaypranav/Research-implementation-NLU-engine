from abc import ABC

from ruth.nlu.constants import ELEMENT_UNIQUE_NAME
from ruth.nlu.elements import Element


class DenseFeaturizer(Element):

    def __init__(self, element_config):
        element_config = element_config or {}
        self.element_config = element_config
        element_config.setdefault(
            ELEMENT_UNIQUE_NAME, self.create_unique_name()
        )
        super().__init__(element_config)

    @staticmethod
    def get_default_config():
        pass

