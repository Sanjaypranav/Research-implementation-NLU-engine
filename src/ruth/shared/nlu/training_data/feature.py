from typing import Any, Dict, Text, Union

import numpy as np
from numpy import ndarray
from ruth.constants import TEXT
from scipy import sparse


class Feature:
    def __init__(self, feature: Union[sparse.spmatrix, ndarray], origin: Text):
        self.feature = feature
        self.origin = origin

    def is_sparse(self):
        return isinstance(self.feature, sparse.spmatrix)

    def is_dense(self):
        return not self.is_sparse()

    def _combine_sparse_features(
        self, additional_features: "Feature", message: Dict[Text, Any]
    ) -> None:
        from scipy.sparse import hstack

        if self.feature.shape[0] != additional_features.feature.shape[0]:
            raise ValueError(
                f"Cannot combine sparse features as sequence dimensions do not "
                f"match: {self.feature.shape[0]} != "
                f"{additional_features.feature.shape[0]}."
            )

        self.features = hstack([self.feature, additional_features.feature])

    def _combine_dense_features(
        self, additional_features: Any, message: Dict[Text, Any]
    ) -> Any:
        if len(self.feature.shape[0]) != len(additional_features.feature.shape[0]):
            raise ValueError(
                f"Cannot concatenate dense features as sequence dimension does not "
                f"match: {self.feature.shape[0]} != "
                f"{len(additional_features)}. Message: {message.get(TEXT, 'Text not available')}"
            )
        else:
            return np.concatenate((self.feature, additional_features), axis=-1)

    def combine_with_features(
        self, additional_features: "Feature", message: Dict[Text, Any]
    ) -> None:
        if additional_features is None:
            return

        if self.is_dense() and additional_features.is_dense():
            self._combine_dense_features(additional_features, message)
        elif self.is_sparse() and additional_features.is_sparse():
            self._combine_sparse_features(additional_features, message)
        else:
            raise ValueError("Cannot combine sparse and dense features.")
