from typing import Optional, Dict, Text, Any


class RuthModelConfig:
    def __init__(self, configuration_values: Optional[Dict[Text, Any]] = None) -> None:

        if not configuration_values:
            configuration_values = {}

        self.language = "en"
        self.pipeline = []
        self.data = None

        self.override(configuration_values)

        if self.__dict__["pipeline"] is None:
            # replaces None with empty list
            self.__dict__["pipeline"] = []

        for key, value in self.items():
            setattr(self, key, value)
