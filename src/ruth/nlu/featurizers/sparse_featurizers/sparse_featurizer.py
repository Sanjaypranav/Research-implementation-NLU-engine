from ruth.shared.nlu.training_data.collections import TrainData


class SparseFeaturizer:
    def __init__(self):
        pass

    def _build_vectorizer(self):
        raise NotImplementedError

    @staticmethod
    def get_data(training_data: TrainData):
        return training_data.get_text_list()
