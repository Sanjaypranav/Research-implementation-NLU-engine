from typing import Text

from scipy import sparse


class Features:
    def __init__(self, feature: sparse.spmatrix, origin: Text):
        self.feature = feature
        self.origin = origin

    def is_sparse(self):
        return isinstance(self.feature, sparse.spmatrix)

    def is_dense(self):
        return not self.is_sparse()
