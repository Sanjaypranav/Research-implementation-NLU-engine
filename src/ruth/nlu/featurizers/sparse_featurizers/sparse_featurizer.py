from ruth.shared.nlu.training_data.collections import TrainData


class SparseFeaturizer:
    def __init__(self, training_data: TrainData):
        self.training_data = training_data

    def _build_vectorizer(self):
        raise NotImplementedError

    def get_data(self):
        return self.training_data.get_text_list()
