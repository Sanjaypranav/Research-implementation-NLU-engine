import json
from pathlib import Path
from typing import List, Text

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.ruth_data import RuthData


class TrainData:
    def __init__(self, training_examples: List[RuthData]):
        self.training_examples = training_examples

    def __len__(self) -> int:
        return len(self.training_examples)

    @classmethod
    def build(cls, data_path: Path) -> "TrainData":
        training_examples = list()
        with open(data_path, "r") as f:
            messages = json.load(f)
        for message in messages:
            training_examples.append(
                RuthData.build(intent=message.get(INTENT), text=message.get(TEXT))
            )
        return cls(training_examples)

    def get_text_list(self) -> List[Text]:
        return [example.text for example in self.training_examples]
