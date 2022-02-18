from typing import Text

from scipy import sparse


class Features:
    def __init__(self, features: sparse.spmatrix, origin: Text):
        self.features = features
        self.origin = origin

    def is_sparse(self):
        return isinstance(self.features, sparse.spmatrix)

    def is_dense(self):
        return not self.is_sparse()
