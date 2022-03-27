from typing import Text

import torch


class Tokens:
    def __init__(self, tokens: torch.Tensor, origin: Text):
        self.tokens = tokens
        self.origin = origin

    def is_token(self):
        return isinstance(self.tokens, torch.Tensor)
