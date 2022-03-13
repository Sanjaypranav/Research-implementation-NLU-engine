from ruth.nlu.element import ElementBuilder
from ruth.shared.nlu.training_data.ruth_config import RuthConfig


class Trainer:
    def __init__(self, config: RuthConfig, element_builder: ElementBuilder = None):
        self.config = config
        self.training_data = None

        if element_builder is None:
            element_builder = ElementBuilder()

        self.pipeline = self._build_pipeline(config, element_builder)

    # def _build_pipeline(self, config: RuthConfig, element_builder: ElementBuilder):
    #     # pipeline = []
    #
    #     for index, element in enumerate(config.pipeline):
    #         pass
