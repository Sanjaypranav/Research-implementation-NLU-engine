from typing import List

from ruth.shared.nlu.training_data.ruth_data import RuthData


class TrainData:
    def __init__(self, training_examples: List[RuthData]):
        self.training_examples = training_examples

    def __len__(self) -> int:
        return len(self.training_examples)
